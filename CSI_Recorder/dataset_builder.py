import os
import ast
import numpy as np
import pandas as pd
 
from feature_extractor import extract_features
 
 
# ==========================================================
# SESSION LOOKUP (from build_session_map.py output)
# ==========================================================
# Filenames alone don't carry session info in your existing
# dataset, so sessions are reconstructed from raw file
# timestamps instead. Run build_session_map.py FIRST, then
# this script will look up each file's session from that map.
 
SESSION_MAP_FILE = "session_map.csv"
 
 
def load_session_map(path):
    if not os.path.exists(path):
        print(
            f"WARNING: {path} not found. Run build_session_map.py "
            f"first if you want session-aware grouping. Falling back "
            f"to treating every file as its own session (not ideal)."
        )
        return {}
 
    df = pd.read_csv(path)
    return {
        (row["Activity"], row["Filename"]): row["SessionID"]
        for _, row in df.iterrows()
    }
 
 
# ==========================================================
# CONFIGURATION
# ==========================================================
 
# Point this at the OUTPUT of the fixed signal_processing.py
# (i.e. OUTPUT_DATASET from that file), not the raw dataset.
DATASET_PATH = "Processed_Dataset_002"
OUTPUT_FILE = "feature_dataset.csv"
 
# After I/Q -> amplitude decoding, each packet has HALF the
# raw length (128 raw I/Q values -> 64 amplitude values).
# Change this if your ESP32 CSI config gives a different count.
EXPECTED_SUBCARRIERS = 64
 
 
# ==========================================================
# BUILD DATASET
# ==========================================================
 
def build_dataset(dataset_path):
 
    X = []
    y = []
    sessions = []
 
    session_map = load_session_map(SESSION_MAP_FILE)
 
    activities = sorted(
        os.listdir(dataset_path)
    )
 
    for activity in activities:
 
        activity_path = os.path.join(
            dataset_path,
            activity
        )
 
        if not os.path.isdir(activity_path):
            continue
 
        print(f"\nProcessing: {activity}")
 
        files = sorted(
            [
                f for f in os.listdir(activity_path)
                if f.endswith(".csv")
            ]
        )
 
        for filename in files:
 
            filepath = os.path.join(
                activity_path,
                filename
            )
 
            try:
 
                df = pd.read_csv(filepath)
 
                if "CSI_Data" not in df.columns:
                    print(
                        f"Skipping {filename}: "
                        f"CSI_Data column missing"
                    )
                    continue
 
                csi_packets = []
 
                for value in df["CSI_Data"]:
 
                    try:
 
                        if isinstance(value, str):
                            parsed = ast.literal_eval(value)
                        else:
                            parsed = value
 
                        if parsed is None:
                            continue
 
                        # CHANGED: expect decoded amplitude
                        # length (64), not raw I/Q length (128)
                        if len(parsed) != EXPECTED_SUBCARRIERS:
                            continue
 
                        csi_packets.append(
                            np.asarray(
                                parsed,
                                dtype=np.float64
                            )
                        )
 
                    except Exception:
                        continue
 
                if len(csi_packets) == 0:
 
                    print(
                        f"Skipping {filename}: "
                        f"No valid CSI packets"
                    )
 
                    continue
 
                csi_matrix = np.array(
                    csi_packets
                )
 
                features = extract_features(
                    csi_matrix
                )
 
                X.append(features)
                y.append(activity)
                sessions.append(
                    session_map.get((activity, filename), filename)
                )
 
            except Exception as e:
 
                print(
                    f"Error processing "
                    f"{filename}: {e}"
                )
 
    return np.array(X), np.array(y), np.array(sessions)
 
 
# ==========================================================
# MAIN
# ==========================================================
 
if __name__ == "__main__":
 
    print("=" * 50)
    print("        CSI FEATURE DATASET BUILDER")
    print("=" * 50)
 
    if not os.path.exists(DATASET_PATH):
 
        print(
            f"\nERROR: "
            f"{DATASET_PATH} not found."
        )
 
        print(
            "\nRun the fixed signal_processing.py first."
        )
 
        exit()
 
    X, y, sessions = build_dataset(
        DATASET_PATH
    )
 
    if len(X) == 0:
 
        print(
            "\nERROR: No valid data found."
        )
 
        exit()
 
    print("\n===================================")
    print("Dataset Successfully Built")
    print("===================================")
 
    print(
        f"Feature Matrix Shape : {X.shape}"
    )
 
    print(
        f"Label Shape          : {y.shape}"
    )
 
    columns = [
        f"Feature_{i + 1}"
        for i in range(X.shape[1])
    ]
 
    dataset = pd.DataFrame(
        X,
        columns=columns
    )
 
    dataset["Label"] = y
    dataset["Session"] = sessions
 
    dataset.to_csv(
        OUTPUT_FILE,
        index=False
    )
 
    print(
        f"\n{OUTPUT_FILE} created successfully!"
    )
 
    print(
        f"Total Samples : {len(dataset)}"
    )
 
    print(
        f"Total Features: {X.shape[1]}"
    )
 
    print("\nClass Distribution:")
 
    print(
        dataset["Label"].value_counts()
    )