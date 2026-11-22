#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <math.h>

// Hardware Pin Definitions (ESP32-S3-DevKitC-1)
#define I2C_SDA_PIN         8
#define I2C_SCL_PIN         9
#define CURRENT_ADC_PIN     1     // ADC1_CH0 (ACS712 analog output)
#define TEMP_ONEWIRE_PIN    4     // DS18B20 digital pin
#define SAFETY_RELAY_PIN    15    // Optocoupler -> Solid-State Relay (Active HIGH = Normal, LOW = Cutoff)
#define STATUS_LED_PIN      2

// Real-Time DSP & Safety Interlock Parameters
#define WINDOW_SIZE         128   // 128 samples @ 1 kHz = 128 ms window
#define SAMPLE_PERIOD_US    1000  // 1 kHz deterministic sampling
#define FAULT_VOTE_LIMIT    3     // Consecutive anomaly windows required before cutoff

// System Operating States
enum SystemState {
    STATE_NORMAL = 0,
    STATE_WARNING = 1,
    STATE_CRITICAL_CUTOFF = 2
};

// Shared Telemetry Structure (Protected via Mutex)
struct TelemetryPacket {
    float vib_rms;
    float vib_crest;
    float current_rms;
    float temperature_c;
    uint32_t inference_us;
    SystemState state;
    bool relay_tripped;
};

Adafruit_MPU6050 mpu;
OneWire oneWire(TEMP_ONEWIRE_PIN);
DallasTemperature tempSensor(&oneWire);

SemaphoreHandle_t telemetryMutex;
TelemetryPacket sharedTelemetry = {0.0f, 0.0f, 0.0f, 25.0f, 0, STATE_NORMAL, false};

// Sensor Sample Buffers
float ax_buf[WINDOW_SIZE];
float ay_buf[WINDOW_SIZE];
float az_buf[WINDOW_SIZE];
float curr_buf[WINDOW_SIZE];

// Trigger immediate fail-safe hardware power isolation
void triggerHardwareCutoff() {
    digitalWrite(SAFETY_RELAY_PIN, LOW); // De-energize SSR (Fail-safe open)
    digitalWrite(STATUS_LED_PIN, HIGH);
}

// Compute Root Mean Square (RMS) of a signal window
float computeRMS(const float* data, int len) {
    float sum_sq = 0.0f;
    for (int i = 0; i < len; i++) {
        sum_sq += data[i] * data[i];
    }
    return sqrtf(sum_sq / (float)len);
}

// Compute Crest Factor (Peak / RMS) for early bearing/impulsive fault detection
float computeCrestFactor(const float* data, int len, float rms) {
    float peak = 0.0f;
    for (int i = 0; i < len; i++) {
        float abs_val = fabsf(data[i]);
        if (abs_val > peak) {
            peak = abs_val;
        }
    }
    return (rms > 1e-6f) ? (peak / rms) : 0.0f;
}

// Quantized Edge Inference Wrapper (Replace body with TFLite Micro invoke() when model_data.h is linked)
SystemState runEdgeInference(float vib_rms, float vib_crest, float curr_rms, float temp_c) {
    // Deterministic multi-modal anomaly scoring aligned with INT8 1D-CNN thresholds
    if (vib_rms > 1.15f || curr_rms > 5.0f || temp_c > 75.0f) {
        return STATE_CRITICAL_CUTOFF;
    }
    if (vib_rms > 0.45f || vib_crest > 3.8f || curr_rms > 3.2f || temp_c > 55.0f) {
        return STATE_WARNING;
    }
    return STATE_NORMAL;
}

// CORE 1: Deterministic Real-Time Acquisition, Inference & Hardware Cutoff Task
void core1ControlTask(void* pvParameters) {
    uint8_t consecutive_faults = 0;
    bool cutoff_latched = false;
    float latest_temp = 25.0f;
    uint32_t loop_counter = 0;

    for (;;) {
        // 1. High-Frequency Window Acquisition (1 kHz)
        for (int i = 0; i < WINDOW_SIZE; i++) {
            uint32_t t_start = micros();

            sensors_event_t a, g, temp;
            mpu.getEvent(&a, &g, &temp);

            ax_buf[i] = a.acceleration.x;
            ay_buf[i] = a.acceleration.y;
            az_buf[i] = a.acceleration.z - 9.81f; // Remove static gravity offset

            // Read ACS712 current sensor (12-bit ADC, 3.3V ref, 100 mV/A sensitivity)
            int raw_adc = analogRead(CURRENT_ADC_PIN);
            float voltage = (raw_adc / 4095.0f) * 3.3f;
            curr_buf[i] = fabsf((voltage - 1.65f) / 0.100f);

            // Maintain strict 1 ms sampling period
            while ((micros() - t_start) < SAMPLE_PERIOD_US) {
                // Busy-wait for sub-millisecond timing precision
            }
        }

        // Poll slow thermal sensor every ~1 second (every 8 windows)
        if (loop_counter % 8 == 0) {
            tempSensor.requestTemperatures();
            float t_read = tempSensor.getTempCByIndex(0);
            if (t_read > -50.0f && t_read < 150.0f) {
                latest_temp = t_read;
            }
        }
        loop_counter++;

        // 2. Time-Domain Feature Extraction & Edge Inference Timing
        uint32_t inf_start = micros();

        float rms_x = computeRMS(ax_buf, WINDOW_SIZE);
        float rms_y = computeRMS(ay_buf, WINDOW_SIZE);
        float rms_z = computeRMS(az_buf, WINDOW_SIZE);
        float total_vib_rms = sqrtf((rms_x * rms_x + rms_y * rms_y + rms_z * rms_z) / 3.0f);
        float crest_z = computeCrestFactor(az_buf, WINDOW_SIZE, rms_z);
        float curr_rms = computeRMS(curr_buf, WINDOW_SIZE);

        SystemState predicted_state = runEdgeInference(total_vib_rms, crest_z, curr_rms, latest_temp);

        // 3. Sliding-Window Consecutive Fault Voting & Hardware Safety Interlock
        if (predicted_state == STATE_CRITICAL_CUTOFF) {
            consecutive_faults++;
            if (consecutive_faults >= FAULT_VOTE_LIMIT && !cutoff_latched) {
                triggerHardwareCutoff();
                cutoff_latched = true;
            }
        } else {
            consecutive_faults = 0;
        }

        uint32_t inf_duration = micros() - inf_start;

        // 4. Publish Snapshot to Core 0 Telemetry Buffer
        if (xSemaphoreTake(telemetryMutex, pdMS_TO_TICKS(5)) == pdTRUE) {
            sharedTelemetry.vib_rms = total_vib_rms;
            sharedTelemetry.vib_crest = crest_z;
            sharedTelemetry.current_rms = curr_rms;
            sharedTelemetry.temperature_c = latest_temp;
            sharedTelemetry.inference_us = inf_duration;
            sharedTelemetry.state = cutoff_latched ? STATE_CRITICAL_CUTOFF : predicted_state;
            sharedTelemetry.relay_tripped = cutoff_latched;
            xSemaphoreGive(telemetryMutex);
        }
    }
}

// CORE 0: Asynchronous Telemetry & Diagnostics Task
void core0TelemetryTask(void* pvParameters) {
    TelemetryPacket localCopy;

    for (;;) {
        if (xSemaphoreTake(telemetryMutex, pdMS_TO_TICKS(20)) == pdTRUE) {
            localCopy = sharedTelemetry;
            xSemaphoreGive(telemetryMutex);
        }

        Serial.printf(
            "[SILICON-SENTINEL] VibRMS: %.3f m/s2 | Crest: %.2f | CurrRMS: %.2f A | Temp: %.1f C | InfTime: %lu us | State: %d | Cutoff: %s\n",
            localCopy.vib_rms,
            localCopy.vib_crest,
            localCopy.current_rms,
            localCopy.temperature_c,
            localCopy.inference_us,
            localCopy.state,
            localCopy.relay_tripped ? "TRIPPED" : "ARMED"
        );

        vTaskDelay(pdMS_TO_TICKS(250)); // 4 Hz diagnostic output
    }
}

void setup() {
    Serial.begin(115200);

    // Initialize Fail-Safe Hardware Interlock (Energize SSR on startup)
    pinMode(SAFETY_RELAY_PIN, OUTPUT);
    pinMode(STATUS_LED_PIN, OUTPUT);
    digitalWrite(SAFETY_RELAY_PIN, HIGH);
    digitalWrite(STATUS_LED_PIN, LOW);

    // Initialize I2C Accelerometer & OneWire Thermal Sensor
    Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
    if (!mpu.begin()) {
        Serial.println("ERROR: MPU6050 initialization failed. Halting safely.");
        triggerHardwareCutoff();
        while (1) { delay(100); }
    }
    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu.setFilterBandwidth(MPU6050_BAND_260_HZ);
    tempSensor.begin();

    telemetryMutex = xSemaphoreCreateMutex();

    // Pin Real-Time Control & Cutoff Loop to Core 1 (High Priority)
    xTaskCreatePinnedToCore(
        core1ControlTask,
        "Core1_Control",
        8192,
        NULL,
        configMAX_PRIORITIES - 1,
        NULL,
        1
    );

    // Pin Asynchronous Telemetry to Core 0 (Lower Priority)
    xTaskCreatePinnedToCore(
        core0TelemetryTask,
        "Core0_Telemetry",
        4096,
        NULL,
        1,
        NULL,
        0
    );
}

void loop() {
    vTaskDelay(portMAX_DELAY); // FreeRTOS scheduler manages execution
}
