import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# PROJECT PATHS
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.dirname(BASE_DIR)

OUTPUT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "Graphs",
    "ML_Results"
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

print("Saving graphs to:", OUTPUT_FOLDER)
os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

# ==========================================================
# IMPORT MODEL RESULTS
# ==========================================================

import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

sys.path.append(os.path.join(PROJECT_ROOT, "ML_Models"))

import random_forest
import svm_model
import xgboost_model

import random_forest
import svm_model
import xgboost_model

rf = random_forest.results
svm = svm_model.results
xgb = xgboost_model.results

print("✓ Models Loaded")

print("Random Forest imported")
print("SVM imported")
print("XGBoost imported")


# ==========================================================
# ACCURACY COMPARISON
# ==========================================================

models = [
    "Random Forest",
    "SVM",
    "XGBoost"
]

accuracies = [

    rf["accuracy"] * 100,

    svm["accuracy"] * 100,

    xgb["accuracy"] * 100

]

plt.figure(figsize=(7,5))

bars = plt.bar(
    models,
    accuracies
)

plt.ylabel("Accuracy (%)")

plt.title(
    "Model Accuracy Comparison",
    fontweight="bold"
)

plt.ylim(0,100)

for bar in bars:

    plt.text(

        bar.get_x()+bar.get_width()/2,

        bar.get_height()+1,

        f"{bar.get_height():.1f}%",

        ha="center"

    )

plt.tight_layout()

plt.savefig(

    os.path.join(

        OUTPUT_FOLDER,

        "Accuracy_Comparison.png"

    ),

    dpi=600

)

plt.close()

print("✓ Accuracy Comparison Saved")



from sklearn.metrics import ConfusionMatrixDisplay

# ==========================================================
# CONFUSION MATRICES
# ==========================================================

models = [
    ("Random Forest", rf),
    ("SVM", svm),
    ("XGBoost", xgb)
]

class_names = ["Empty", "Sitting", "Standing", "Walking"]

for name, result in models:

    plt.figure(figsize=(6,5))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=result["confusion_matrix"],
        display_labels=class_names
    )

    disp.plot(
        cmap="Blues",
        values_format="d",
        colorbar=False
    )

    plt.title(
        f"{name} Confusion Matrix",
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            f"{name.replace(' ','_')}_Confusion_Matrix.png"
        ),
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()

print("✓ Confusion Matrices Saved")    