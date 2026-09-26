import os
import joblib
import numpy as np
import pandas as pd
 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix
 
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
 
# ==========================================================
# CONFIG
# ==========================================================
 
FEATURE_DATASET = "feature_dataset.csv"
MODEL_DIR = "ML_Models"
N_SPLITS = 5  # will auto-shrink if you have fewer sessions
 
os.makedirs(MODEL_DIR, exist_ok=True)
 
 
def main():
 
    df = pd.read_csv(FEATURE_DATASET)
 
    if "Session" not in df.columns:
        print(
            "ERROR: feature_dataset.csv has no 'Session' column.\n"
            "Rebuild it with the updated dataset_builder.py, and make\n"
            "sure your raw CSVs were recorded with the updated\n"
            "recorder.py (which tags each run with a session ID)."
        )
        return
 
    X = df.drop(["Label", "Session"], axis=1)
    y = df["Label"]
    groups = df["Session"]
 
    n_sessions = groups.nunique()
    print(f"Total samples  : {len(df)}")
    print(f"Total sessions : {n_sessions}")
    print(f"Class counts   :\n{y.value_counts()}\n")
 
    if n_sessions < 2:
        print(
            "ERROR: Only 1 unique session found. You cannot measure\n"
            "real generalization until you collect data across\n"
            "multiple separate recording sessions (different times/\n"
            "days). Collect more sessions before trusting any accuracy\n"
            "number from this dataset."
        )
        return
 
    n_splits = min(N_SPLITS, n_sessions)
    gkf = GroupKFold(n_splits=n_splits)
 
    # ------------------------------------------------------
    # SESSION-AWARE CROSS-VALIDATION (the real number)
    # ------------------------------------------------------
 
    print(f"Running GroupKFold (n_splits={n_splits}), sessions held out per fold...\n")
 
    fold_accuracies = []
 
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
 
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
 
        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(X_train, y_train)
        acc = rf.score(X_test, y_test)
        fold_accuracies.append(acc)
 
        test_sessions = groups.iloc[test_idx].unique()
        print(f"Fold {fold}: held-out session(s)={list(test_sessions)} -> accuracy={acc:.3f}")
 
    fold_accuracies = np.array(fold_accuracies)
    print(f"\nSession-aware CV accuracy: {fold_accuracies.mean():.3f} +/- {fold_accuracies.std():.3f}")
    print(
        "\nThis number -- not train accuracy, not row-level shuffled CV --"
        "\nis the honest estimate of how the model will do on a new session."
    )
 
    # ------------------------------------------------------
    # FINAL MODELS: train on everything, save for deployment
    # (only do this once the session-aware CV number above
    # looks reasonable -- don't ship a model whose real
    # accuracy you haven't measured this way)
    # ------------------------------------------------------
 
    print("\nTraining final models on full dataset for deployment...")
 
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
 
    rf_final = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_final.fit(X, y)
 
    svm_scaler = StandardScaler()
    X_scaled = svm_scaler.fit_transform(X)
    svm_final = SVC(probability=True, random_state=42)
    svm_final.fit(X_scaled, y)
 
    knn_scaler = StandardScaler()
    X_knn_scaled = knn_scaler.fit_transform(X)
    knn_final = KNeighborsClassifier(n_neighbors=5)
    knn_final.fit(X_knn_scaled, y)
 
    joblib.dump(rf_final, os.path.join(MODEL_DIR, "random_forest_model.pkl"))
    joblib.dump(svm_final, os.path.join(MODEL_DIR, "svm_model.pkl"))
    joblib.dump(svm_scaler, os.path.join(MODEL_DIR, "svm_scaler.pkl"))
    joblib.dump(knn_final, os.path.join(MODEL_DIR, "knn_model.pkl"))
    joblib.dump(knn_scaler, os.path.join(MODEL_DIR, "knn_scaler.pkl"))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))
 
    if HAS_XGB:
        xgb_final = XGBClassifier(eval_metric="mlogloss", random_state=42)
        xgb_final.fit(X, y_encoded)
        joblib.dump(xgb_final, os.path.join(MODEL_DIR, "xgboost_model.pkl"))
    else:
        print("xgboost not installed -- skipped xgb model.")
 
    print(f"\nModels saved to {MODEL_DIR}/")
    print("Report the session-aware CV accuracy above, not train accuracy,")
    print("as your project's real performance number.")
 
 
if __name__ == "__main__":
    main()