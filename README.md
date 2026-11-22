# Silicon Sentinel: Low-Cost Edge-AI Predictive Maintenance and Automatic Safety Cutoff Using ESP32-S3

## Overview
Silicon Sentinel is a deterministic Edge-AI condition monitoring and hardware safety interlock system designed for auxiliary semiconductor fabrication equipment, including vacuum roughing pumps, coolant chillers, and chemical delivery systems. 

By executing quantized neural network inference directly on an ESP32-S3 microcontroller, the system eliminates cloud network latency and prevents intellectual property exposure while providing sub-15 ms hardware power cutoff during critical fault conditions.

## Key Technical Features
* **Deterministic Dual-Core Execution:** Uses FreeRTOS task pinning on the ESP32-S3 to isolate asynchronous network telemetry (Core 0) from high-priority sensor acquisition, TinyML inference, and safety relay actuation (Core 1).
* **On-Device Quantized Inference:** Combines time- and frequency-domain feature extraction with an INT8-quantized 1D Convolutional Neural Network (1D-CNN) accelerated by Xtensa LX7 vector instructions.
* **False-Trigger Mitigation:** Implements a sliding-window consecutive-anomaly voting mechanism prior to hardware cutoff to prevent spurious shutdowns from transient mechanical noise.
* **Galvanic Safety Interlock:** Integrates an optocoupler-driven Solid-State Relay (SSR) to physically isolate equipment power independently of external network availability.

## Repository Structure
```text
Silicon-Sentinel-EdgeAI/
├── hardware/           # Circuit schematics, pinouts, and Bill of Materials
├── dataset/            # Labeled multi-sensor telemetry logs (CSV)
├── model_training/     # Python feature extraction, 1D-CNN training, and INT8 export
├── firmware_esp32s3/   # Dual-core FreeRTOS C++ firmware for ESP32-S3
├── LICENSE             # MIT License
└── README.md           # Project documentation
```

## Hardware Components
* **Microcontroller:** ESP32-S3-DevKitC-1 (N16R8)
* **Vibration Sensor:** ADXL345 or MPU-6050 (Tri-axial accelerometer)
* **Current Sensor:** ACS712 (20A) or SCT-013 current transducer
* **Temperature Sensor:** MAX31865 with PT100 RTD
* **Safety Actuator:** PC817 Optocoupler and Solid-State Relay (SSR)

## Usage
1. **Model Training and Quantization:**
   ```bash
   cd model_training
   pip install -r requirements.txt
   python train_tinyml.py
   ```
2. **Firmware Build and Flash (PlatformIO):**
   ```bash
   cd firmware_esp32s3
   pio run --target upload
   ```

## Citation
```bibtex
@misc{kumar2026siliconsentinel,
  author       = {Suhashini Kumar},
  title        = {Silicon Sentinel: Low-Cost Edge-AI Predictive Maintenance and Automatic Safety Cutoff Using ESP32-S3},
  year         = {2026},
  howpublished \(= {\url{https://github.com}} \)}
```

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

