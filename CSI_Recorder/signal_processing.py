import os
import ast
import numpy as np
import pandas as pd
 
# ============================================================
# CONFIGURATION
# ============================================================
 
RAW_DATASET = r"C:\CSI_Project\Dataset"
OUTPUT_DATASET = r"C:\CSI_Project\Processed_Dataset_002"
 
MOVING_AVERAGE_WINDOW = 5
ENABLE_MOVING_AVERAGE = True
 
ENABLE_OUTLIER_REMOVAL = True
OUTLIER_STD_FACTOR = 3
 
# Per-packet min-max normalization REMOVED.
# It was forcing every recording into [0,1] regardless of
# actual signal strength, which erased the absolute amplitude
# differences between Empty/Sitting/Standing/Walking -- that
# absolute level is your main discriminative signal.
# If you need scaling later, fit a scaler on the TRAINING SET
# ONLY after feature extraction, and apply the same fitted
# scaler at inference time. Never normalize per packet.
ENABLE_NORMALIZATION = False
 
 
# ============================================================
# PARSE CSI (raw string -> raw I/Q array)
# ============================================================
 
def parse_csi(csi_string):
    try:
        start = csi_string.rfind("[")
        end = csi_string.rfind("]") + 1
 
        if start == -1 or end == 0:
            return None
 
        values = ast.literal_eval(csi_string[start:end])
 
        if not isinstance(values, list):
            return None
 
        return np.array(values, dtype=float)
 
    except Exception:
        return None
 
 
# ============================================================
# I/Q -> AMPLITUDE DECODING  <-- THE CRITICAL FIX
# ============================================================
# ESP32 CSI raw data is a flat array of interleaved
# [imaginary, real, imaginary, real, ...] pairs, one pair per
# subcarrier. The old code treated these 128 raw values as one
# generic 128-point "signal" and smoothed/normalized across it
# directly -- mixing imaginary and real components together and
# discarding the actual channel amplitude info activity
# detection depends on.
#
# This converts N interleaved I/Q values into N/2 real
# amplitude values: amplitude[k] = sqrt(I[k]^2 + Q[k]^2)
 
def decode_amplitude(raw_iq):
    raw_iq = np.asarray(raw_iq, dtype=np.float64)
 
    if raw_iq.ndim != 1 or len(raw_iq) % 2 != 0:
        return None
 
    imag = raw_iq[0::2]
    real = raw_iq[1::2]
 
    amplitude = np.sqrt(imag ** 2 + real ** 2)
 
    return amplitude
 
 
# ============================================================
# MOVING AVERAGE FILTER (now applied to decoded amplitude,
# not to raw interleaved I/Q)
# ============================================================
 
def moving_average(signal, window=5):
 
    if len(signal) < window:
        return signal
 
    kernel = np.ones(window) / window
 
    return np.convolve(signal, kernel, mode="same")
 
 
# ============================================================
# OUTLIER REMOVAL
# Kept as-is: clipping extreme spikes at +/-3 std doesn't
# rescale the whole signal, so it doesn't destroy inter-class
# amplitude differences the way min-max normalization did.
# ============================================================
 
def remove_outliers(signal):
 
    mean = np.mean(signal)
    std = np.std(signal)
 
    if std == 0:
        return signal
 
    upper = mean + OUTLIER_STD_FACTOR * std
    lower = mean - OUTLIER_STD_FACTOR * std
 
    return np.clip(signal, lower, upper)
 
 
# ============================================================
# PROCESS SINGLE SIGNAL (raw I/Q packet -> clean amplitude)
# ============================================================
 
def process_signal(raw_iq):
 
    amplitude = decode_amplitude(raw_iq)
 
    if amplitude is None:
        return None
 
    if ENABLE_MOVING_AVERAGE:
        amplitude = moving_average(amplitude, MOVING_AVERAGE_WINDOW)
 
    if ENABLE_OUTLIER_REMOVAL:
        amplitude = remove_outliers(amplitude)
 
    if ENABLE_NORMALIZATION:
        mn, mx = np.min(amplitude), np.max(amplitude)
        if mx - mn != 0:
            amplitude = (amplitude - mn) / (mx - mn)
 
    return amplitude
 
 
# ============================================================
# PROCESS SINGLE CSV
# ============================================================
 
def process_csv(input_csv, output_csv):
 
    df = pd.read_csv(input_csv)
 
    processed_rows = []
 
    for _, row in df.iterrows():
 
        raw_iq = parse_csi(row["CSI_Data"])
 
        if raw_iq is None:
            continue
 
        amplitude = process_signal(raw_iq)
 
        if amplitude is None:
            continue
 
        processed_rows.append({
            "Packet_No": row["Packet_No"],
            "CSI_Data": np.round(amplitude, 6).astype(float).tolist()
        })
 
    processed_df = pd.DataFrame(processed_rows)
 
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
 
    processed_df.to_csv(output_csv, index=False)
 
 
# ============================================================
# PROCESS ENTIRE DATASET
# ============================================================
 
def process_dataset():
 
    for activity in os.listdir(RAW_DATASET):
 
        activity_path = os.path.join(RAW_DATASET, activity)
 
        if not os.path.isdir(activity_path):
            continue
 
        output_activity = os.path.join(OUTPUT_DATASET, activity)
        os.makedirs(output_activity, exist_ok=True)
 
        for file in os.listdir(activity_path):
 
            if not file.endswith(".csv"):
                continue
 
            input_csv = os.path.join(activity_path, file)
            output_csv = os.path.join(output_activity, file)
 
            process_csv(input_csv, output_csv)
 
            print(f"Processed: {activity}/{file}")
 
 
# ============================================================
# MAIN
# ============================================================
 
if __name__ == "__main__":
 
    process_dataset()
 
    print("\nSignal Processing Completed Successfully.")
    print(f"Output: {OUTPUT_DATASET}")
    print("\nNext: point dataset_builder.py's DATASET_PATH at this")
    print("new output folder, then rebuild feature_dataset.csv.")