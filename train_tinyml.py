import os
import numpy as np
import tensorflow as tf
from scipy.stats import kurtosis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Configuration parameters
WINDOW_SIZE = 128       # Samples per inference window (e.g., 128 ms at 1 kHz)
NUM_CHANNELS = 5        # [ax, ay, az, current, temperature]
NUM_CLASSES = 3         # 0: Normal, 1: Early Degradation (Warning), 2: Critical Fault (Cutoff)
SAMPLES_PER_CLASS = 800


def generate_synthetic_telemetry(samples_per_class=SAMPLES_PER_CLASS, window_size=WINDOW_SIZE):
    """
    Generates multi-sensor windows simulating auxiliary fab equipment states:
    - Class 0: Normal baseline operation
    - Class 1: Mechanical imbalance / bearing wear (elevated vibration harmonics)
    - Class 2: Critical stall / overheat / severe cavitation (triggers hardware safety cutoff)
    """
    X_raw, y = [], []
    t = np.linspace(0, 1, window_size)

    for label in range(NUM_CLASSES):
        for _ in range(samples_per_class):
            if label == 0:
                vib_amp, curr_base, temp_base, noise_std = 0.15, 2.5, 42.0, 0.04
            elif label == 1:
                vib_amp, curr_base, temp_base, noise_std = 0.55, 3.1, 56.0, 0.10
            else:
                vib_amp, curr_base, temp_base, noise_std = 1.45, 5.8, 78.0, 0.25

            ax = vib_amp * np.sin(2 * np.pi * 50 * t) + np.random.normal(0, noise_std, window_size)
            ay = vib_amp * np.cos(2 * np.pi * 50 * t) + np.random.normal(0, noise_std, window_size)
            az = 1.0 + (vib_amp * 0.6) * np.sin(2 * np.pi * 120 * t) + np.random.normal(0, noise_std, window_size)
            current = curr_base + np.random.normal(0, noise_std * 0.5, window_size)
            temp = temp_base + np.random.normal(0, 0.2, window_size)

            window = np.stack([ax, ay, az, current, temp], axis=-1)
            X_raw.append(window)
            y.append(label)

    return np.array(X_raw, dtype=np.float32), np.array(y, dtype=np.int32)


def extract_features(X_raw):
    """
    Extracts compact time- and frequency-domain features per channel:
    [RMS, Peak-to-Peak, Kurtosis, Crest Factor, Spectral Energy] -> (N, 5, 5)
    """
    num_samples, _, num_channels = X_raw.shape
    features = np.zeros((num_samples, num_channels, 5), dtype=np.float32)

    for i in range(num_samples):
        for c in range(num_channels):
            sig = X_raw[i, :, c]
            rms = np.sqrt(np.mean(sig ** 2)) + 1e-8
            p2p = np.ptp(sig)
            kurt = kurtosis(sig)
            crest = np.max(np.abs(sig)) / rms
            fft_mag = np.abs(np.fft.rfft(sig))
            spec_energy = np.sum(fft_mag ** 2) / len(fft_mag)

            features[i, c, :] = [rms, p2p, kurt, crest, spec_energy]

    return features


def build_1d_cnn(input_shape, num_classes):
    """Lightweight 1D-CNN optimized for ESP32-S3 Xtensa LX7 vector instructions."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv1D(filters=16, kernel_size=3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def export_to_c_header(tflite_model, output_path="model_data.h"):
    """Converts binary TFLite flatbuffer into a C byte array for ESP32-S3 firmware."""
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    hex_lines = []
    for i in range(0, len(tflite_model), 12):
        chunk = ", ".join(f"0x{b:02x}" for b in tflite_model[i:i + 12])
        hex_lines.append(f"  {chunk},")

    header_content = (
        "#ifndef SILICON_SENTINEL_MODEL_DATA_H\n"
        "#define SILICON_SENTINEL_MODEL_DATA_H\n\n"
        "#include <stdint.h>\n\n"
        f"const unsigned int g_model_data_len = {len(tflite_model)};\n"
        "alignas(16) const unsigned char g_model_data[] = {\n"
        + "\n".join(hex_lines) + "\n"
        "};\n\n"
        "#endif // SILICON_SENTINEL_MODEL_DATA_H\n"
    )

    with open(output_path, "w") as f:
        f.write(header_content)
    print(f"Exported C header to {output_path} ({len(tflite_model)} bytes)")


def main():
    print("Generating multi-sensor telemetry dataset...")
    X_raw, y = generate_synthetic_telemetry()
    X_feat = extract_features(X_raw)

    # Normalize features across channels
    num_samples, c_dim, f_dim = X_feat.shape
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_feat.reshape(-1, f_dim)).reshape(num_samples, c_dim, f_dim)

    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=0.2, random_state=SEED, stratify=y
    )

    model = build_1d_cnn(input_shape=(c_dim, f_dim), num_classes=NUM_CLASSES)
    model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.15, verbose=1)

    _, fp32_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"FP32 Test Accuracy: {fp32_acc * 100:.2f}%")

    # Full Integer (INT8) Post-Training Quantization for ESP32-S3
    def representative_dataset():
        for i in range(min(200, len(X_train))):
            yield [X_train[i:i + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_int8_model = converter.convert()
    export_to_c_header(tflite_int8_model, output_path="model_data.h")


if __name__ == "__main__":
    main()
