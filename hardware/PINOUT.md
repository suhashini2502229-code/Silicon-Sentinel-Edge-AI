# Hardware Wiring and GPIO Pinout Specification

## ESP32-S3-DevKitC-1 Interface Mapping

| Peripheral Module | Module Pin | ESP32-S3 Pin | Protocol / Signal Type | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **MPU-6050 Accelerometer** | VCC | 3V3 | Power | 3.3V regulated supply |
| | GND | GND | Ground | Common system ground |
| | SDA | GPIO 8 | I2C Data | 400 kHz Fast Mode |
| | SCL | GPIO 9 | I2C Clock | 400 kHz Fast Mode |
| **ACS712 Current Sensor** | VCC | 5V | Power | Requires 5V supply for linear Hall IC |
| | GND | GND | Ground | Common system ground |
| | OUT | GPIO 1 | Analog (ADC1_CH0) | Use 10k/20k voltage divider to cap at 3.3V |
| **DS18B20 Temp Probe** | VDD | 3V3 | Power | 3.3V supply |
| | GND | GND | Ground | Common system ground |
| | DQ | GPIO 4 | OneWire Digital | Requires 4.7 kOhm pull-up resistor to 3V3 |
| **PC817 Optocoupler + SSR** | IN+ | GPIO 15 | Digital Output | HIGH = Normal Operation, LOW = Fail-Safe Cutoff |
| | IN- | GND | Ground | Galvanically isolates MCU from AC/DC load |
| **Fault Status Indicator** | Anode | GPIO 2 | Digital Output | 220 Ohm series current-limiting resistor |

## Safety Isolation Notes
* **Fail-Safe Default:** GPIO 15 must remain `HIGH` to keep the Solid-State Relay energized. In the event of microcontroller brownout, sensor disconnection, or confirmed anomaly detection, GPIO 15 drops `LOW`, immediately opening the load circuit.
* **ADC Protection:** Because the ACS712 operates at 5V, ensure the analog output is stepped down via a precision resistor divider before connecting to GPIO 1 to stay within the ESP32-S3 3.3V ADC limit.
