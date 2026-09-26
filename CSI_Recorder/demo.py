import os
import time
import joblib
import serial
import numpy as np
import pandas as pd

from serial_manager import connect_serial
from parser.csi_parser import parse_csi
from signal_processing import process_signal
from feature_extractor import extract_features

# ==========================================================
# CONFIGURATION
# ==========================================================

DURATION = 15
MODEL_DIR = "ML_Models"
FEATURE_DATASET = "feature_dataset.csv"

REQUIRED_MODELS = {
    "rf": "random_forest_model.pkl",
    "svm": "svm_model.pkl",
    "scaler": "svm_scaler.pkl",
    "xgb": "xgboost_model.pkl",
    "encoder": "label_encoder.pkl"
}

# ==========================================================
# GLOBAL VARIABLES
# ==========================================================

rf_model = None
svm_model = None
svm_scaler = None
xgb_model = None
label_encoder = None

feature_columns = None
ser = None

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def print_header(title):
    print("\n" + "=" * 60)
    print(title.center(60))
    print("=" * 60)


def load_models():
    global rf_model
    global svm_model
    global svm_scaler
    global xgb_model
    global label_encoder

    print_header("Loading Models")

    rf_model = joblib.load(
        os.path.join(MODEL_DIR, REQUIRED_MODELS["rf"])
    )

    svm_model = joblib.load(
        os.path.join(MODEL_DIR, REQUIRED_MODELS["svm"])
    )

    svm_scaler = joblib.load(
        os.path.join(MODEL_DIR, REQUIRED_MODELS["scaler"])
    )

    xgb_model = joblib.load(
        os.path.join(MODEL_DIR, REQUIRED_MODELS["xgb"])
    )

    label_encoder = joblib.load(
        os.path.join(MODEL_DIR, REQUIRED_MODELS["encoder"])
    )

    print("Models Loaded Successfully.")


def load_feature_columns():
    global feature_columns

    print_header("Loading Feature Columns")

    df = pd.read_csv(FEATURE_DATASET)

    feature_columns = [
        col for col in df.columns
        if col != "Label"
    ]

    print(f"Total Features : {len(feature_columns)}")


def initialize_serial():
    global ser

    print_header("Connecting ESP32")

    ser = connect_serial()

    if ser is None:
        raise Exception("Failed to connect to ESP32.")

    print("ESP32 Connected Successfully.")


def initialize():
    load_models()
    load_feature_columns()
    initialize_serial()

    # ==========================================================
# LIVE CSI RECORDING
# ==========================================================

def record_csi():

    print_header("Recording CSI")

    packets = []

    print("\nRecording starts in...")

    for i in range(5, 0, -1):
        print(f"{i}...")
        time.sleep(1)

    print("\nRecording Started...\n")

    start_time = time.time()
    invalid_packets = 0

    while (time.time() - start_time) < DURATION:

        try:

            line = ser.readline().decode(
                "utf-8",
                errors="ignore"
            ).strip()

            if not line:
                continue

            if "CSI_DATA" not in line:
                continue

            values = parse_csi(line)

            if values is None:
                invalid_packets += 1
                continue

            if len(values) != 128:
                invalid_packets += 1
                continue

            packets.append(
                np.asarray(values, dtype=np.float32)
            )

        except Exception:
            invalid_packets += 1
            continue

    print(f"\nValid Packets   : {len(packets)}")
    print(f"Invalid Packets : {invalid_packets}")

    if len(packets) == 0:
        print("No valid CSI packets collected.")
        return None

    return np.asarray(
        packets,
        dtype=np.float32
    )


def packet_statistics(csi_matrix):

    print_header("Packet Statistics")

    print(f"Packets      : {csi_matrix.shape[0]}")
    print(f"Subcarriers  : {csi_matrix.shape[1]}")


# ==========================================================
# SIGNAL PROCESSING
# ==========================================================

def preprocess_packets(csi_matrix):

    print_header("Signal Processing")

    if csi_matrix is None:
        return None

    processed_packets = []

    for packet in csi_matrix:

        try:

            packet = np.asarray(
                packet,
                dtype=np.float32
            )

            packet = np.nan_to_num(packet)

            filtered_packet = process_signal(packet)

            processed_packets.append(filtered_packet)

        except Exception:
            continue

    if len(processed_packets) == 0:
        print("No packets after processing.")
        return None

    processed_packets = np.asarray(
        processed_packets,
        dtype=np.float32
    )

    print(
        f"Processed Packets : "
        f"{processed_packets.shape[0]}"
    )

    return processed_packets


def validate_csi_matrix(csi_matrix):

    if csi_matrix is None:
        return False

    if csi_matrix.ndim != 2:
        return False

    if csi_matrix.shape[1] != 128:
        return False

    if np.isnan(csi_matrix).any():
        return False

    if np.isinf(csi_matrix).any():
        return False

    return True


# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

def extract_feature_vector(csi_matrix):

    print_header("Feature Extraction")

    try:

        feature_vector = extract_features(csi_matrix)

        feature_vector = np.asarray(
            feature_vector,
            dtype=np.float32
        )

        return feature_vector

    except Exception as e:

        print(f"Feature Extraction Error : {e}")

        return None


def validate_feature_vector(feature_vector):

    if feature_vector is None:
        return False

    if len(feature_vector) != len(feature_columns):

        print(
            f"Feature Count Mismatch\n"
            f"Expected : {len(feature_columns)}\n"
            f"Got      : {len(feature_vector)}"
        )

        return False

    if np.isnan(feature_vector).any():
        return False

    if np.isinf(feature_vector).any():
        return False

    return True


def create_feature_dataframe(feature_vector):

    return pd.DataFrame(
        [feature_vector],
        columns=feature_columns
    )


def prepare_features(csi_matrix):

    feature_vector = extract_feature_vector(
        csi_matrix
    )

    if not validate_feature_vector(
        feature_vector
    ):
        return None

    X = create_feature_dataframe(
        feature_vector
    )

    print("Features Prepared Successfully.")

    return X


# ==========================================================
# PREDICTION
# ==========================================================

def predict_activity(X):

    print_header("Activity Prediction")


    try:

        # Random Forest
        rf_prediction = rf_model.predict(X)[0]

        rf_probability = np.max(
            rf_model.predict_proba(X)
        )

        # SVM
        X_scaled = svm_scaler.transform(X)

        svm_prediction = svm_model.predict(
            X_scaled
        )[0]

        svm_probability = np.max(
            svm_model.predict_proba(
                X_scaled
            )
        )

        # XGBoost
        xgb_prediction = xgb_model.predict(X)[0]

        xgb_prediction = label_encoder.inverse_transform(
            [int(xgb_prediction)]
        )[0]

        predictions = [
            rf_prediction,
            svm_prediction,
            xgb_prediction
        ]

        final_prediction = max(
            set(predictions),
            key=predictions.count
        )

        xgb_probability = np.max(
            xgb_model.predict_proba(X)
        )

        confidence = (
            rf_probability +
            svm_probability +
            xgb_probability
        ) / 3 * 100

        return {
            "rf": rf_prediction,
            "svm": svm_prediction,
            "xgb": xgb_prediction,
            "final": final_prediction,
            "confidence": confidence
        }

    except Exception as e:

        print(f"Prediction Error : {e}")

        return None


def display_results(result):

    print_header("RESULT")

    print(f"Random Forest : {result['rf']}")
    print(f"SVM           : {result['svm']}")
    print(f"XGBoost       : {result['xgb']}")

    print("\n" + "=" * 60)

    print(f"Detected Activity : {result['final']}")

    print(
        f"Confidence        : "
        f"{result['confidence']:.2f}%"
    )

    print("=" * 60)

    # ==========================================================
# MAIN
# ==========================================================

def main():

    print_header("WiFi CSI Activity Detection")

    try:

        initialize()

    except Exception as e:

        print(f"\nInitialization Error : {e}")
        return

    while True:

        user_input = input(
            "\nPress ENTER to record CSI "
            "or type 'q' to quit: "
        ).strip().lower()

        if user_input == "q":
            break

        try:
   

        

            csi_matrix = record_csi()

            if csi_matrix is None:
                continue

            packet_statistics(csi_matrix)

            processed_matrix = preprocess_packets(
                csi_matrix
            )

            if not validate_csi_matrix(
                processed_matrix
            ):
                print("Invalid CSI Matrix.")
                continue

            X = prepare_features(
                processed_matrix
            )

            if X is None:
                continue

            result = predict_activity(X)

            if result is None:
                continue

            display_results(result)

        except KeyboardInterrupt:

            print("\nInterrupted by User.")
            break

        except Exception as e:

            print(f"\nRuntime Error : {e}")

    if ser is not None:
        ser.close()

    print("\nProgram Closed.")


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()