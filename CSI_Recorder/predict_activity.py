import os
import sys
import joblib
import numpy as np
import pandas as pd

from parser.csi_parser import parse_csi
from signal_processing import process_signal
from feature_extractor import extract_features

# ==========================================================
# LOAD MODELS
# ==========================================================

MODEL_DIR = "ML_Models"

required_models = [
    "random_forest_model.pkl",
    "svm_model.pkl",
    "svm_scaler.pkl",
    "xgboost_model.pkl",
    "label_encoder.pkl",
    "knn_model.pkl",
    "knn_scaler.pkl"
]

for model_name in required_models:
    path = os.path.join(MODEL_DIR, model_name)

    if not os.path.exists(path):
        print(f"ERROR: Missing model -> {path}")
        sys.exit()

rf_model = joblib.load(
    os.path.join(MODEL_DIR, "random_forest_model.pkl")
)

svm_model = joblib.load(
    os.path.join(MODEL_DIR, "svm_model.pkl")
)

svm_scaler = joblib.load(
    os.path.join(MODEL_DIR, "svm_scaler.pkl")
)

xgb_model = joblib.load(
    os.path.join(MODEL_DIR, "xgboost_model.pkl")
)

label_encoder = joblib.load(
    os.path.join(MODEL_DIR, "label_encoder.pkl")
)

knn_model = joblib.load(
    os.path.join(MODEL_DIR, "knn_model.pkl")
)

knn_scaler = joblib.load(
    os.path.join(MODEL_DIR, "knn_scaler.pkl")
)

# ==========================================================
# LOAD FEATURE COLUMN NAMES
# ==========================================================

FEATURE_DATASET = "feature_dataset.csv"

if not os.path.exists(FEATURE_DATASET):
    print("ERROR: feature_dataset.csv not found.")
    sys.exit()

feature_columns = (
    pd.read_csv(FEATURE_DATASET, nrows=1)
    .drop("Label", axis=1)
    .columns
)

# ==========================================================
# INPUT FILE
# ==========================================================

csv_path = input("\nEnter CSI CSV file path : ").strip()

# Remove optional Python raw-string notation and quotes
if csv_path.startswith('r"') and csv_path.endswith('"'):
    csv_path = csv_path[2:-1]
elif csv_path.startswith("r'") and csv_path.endswith("'"):
    csv_path = csv_path[2:-1]
elif csv_path.startswith('"') and csv_path.endswith('"'):
    csv_path = csv_path[1:-1]
elif csv_path.startswith("'") and csv_path.endswith("'"):
    csv_path = csv_path[1:-1]

if not os.path.exists(csv_path):
    print("\nERROR: File not found.")
    sys.exit()

print("\n========================================")
print("Reading File")
print("========================================")
print(csv_path)

try:
    df = pd.read_csv(csv_path)
except Exception as e:
    print(f"Unable to read CSV\n{e}")
    sys.exit()

if "CSI_Data" not in df.columns:
    print("ERROR: 'CSI_Data' column not found.")
    sys.exit()

# ==========================================================
# PARSE + SIGNAL PROCESSING
# ==========================================================

csi_packets = []

for _, row in df.iterrows():

    try:
        values = parse_csi(row["CSI_Data"])

        if values is None:
            continue

        if len(values) != 128:
            continue

        values = np.asarray(values, dtype=np.float64)

        values = process_signal(values)

        csi_packets.append(values)

    except Exception:
        continue

if len(csi_packets) == 0:
    print("No valid CSI packets found.")
    sys.exit()

print(f"\nValid CSI Packets : {len(csi_packets)}")

# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

csi_matrix = np.array(csi_packets)

features = extract_features(csi_matrix)

pd.DataFrame([features]).to_csv("test_features.csv", index=False)

print("test_features.csv saved")



if len(features) != len(feature_columns):

    print("\nERROR")
    print("Feature Length Mismatch")
    print(f"Expected : {len(feature_columns)}")
    print(f"Received : {len(features)}")

    sys.exit()

print(f"Extracted Features : {len(features)}")

X = pd.DataFrame(
    [features],
    columns=feature_columns
)

# ==========================================================
# PREDICTIONS + CONFIDENCE ENSEMBLE
# ==========================================================
# ==========================================================
# MODEL INPUT SCALING
# ==========================================================

X_knn = knn_scaler.transform(X)

X_scaled = svm_scaler.transform(X)
# ----------------------------------------------------------
# RANDOM FOREST
# ----------------------------------------------------------

rf_prob = rf_model.predict_proba(X)[0]
rf_classes = rf_model.classes_

rf_scores = dict(zip(rf_classes, rf_prob))
rf_prediction = rf_model.predict(X)[0]


# ----------------------------------------------------------
# XGBOOST
# ----------------------------------------------------------

xgb_prob = xgb_model.predict_proba(X)[0]

# Convert XGBoost numeric classes to activity names
xgb_class_names = label_encoder.inverse_transform(
    np.arange(len(xgb_prob))
)

xgb_scores = dict(zip(xgb_class_names, xgb_prob))

xgb_prediction_num = np.argmax(xgb_prob)

xgb_prediction = label_encoder.inverse_transform(
    [xgb_prediction_num]
)[0]


# ----------------------------------------------------------
# KNN
# ----------------------------------------------------------

knn_prob = knn_model.predict_proba(X_knn)[0]
knn_classes = knn_model.classes_

knn_scores = dict(zip(knn_classes, knn_prob))
knn_prediction = knn_model.predict(X_knn)[0]


# ----------------------------------------------------------
# SVM
# ----------------------------------------------------------

try:

    svm_prob = svm_model.predict_proba(X_scaled)[0]
    svm_classes = svm_model.classes_

    svm_scores = dict(zip(svm_classes, svm_prob))

except Exception:

    svm_prediction_temp = svm_model.predict(X_scaled)[0]

    svm_scores = {
        "Empty": 0.0,
        "Sitting": 0.0,
        "Standing": 0.0,
        "Walking": 0.0
    }

    svm_scores[svm_prediction_temp] = 1.0

svm_prediction = svm_model.predict(X_scaled)[0]


# ==========================================================
# FINAL PREDICTION LOGIC
# ==========================================================

if (
    knn_prediction == "Walking"
    and rf_prediction == "Walking"
):
    final_prediction = "Walking"

else:
    final_prediction = xgb_prediction


# ==========================================================
# RESULT
# ==========================================================

print("\n========================================")
print("ACTIVITY PREDICTION")
print("========================================")

print(f"Random Forest : {rf_prediction}")
print(f"SVM           : {svm_prediction}")
print(f"XGBoost       : {xgb_prediction}")
print(f"KNN           : {knn_prediction}")

print("----------------------------------------")
print(f"Final Result  : {final_prediction}")
print("========================================")