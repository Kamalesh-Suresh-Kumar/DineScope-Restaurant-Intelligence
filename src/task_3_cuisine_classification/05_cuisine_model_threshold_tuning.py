from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import joblib

from scipy.sparse import load_npz

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    hamming_loss
)


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "task_3_cuisine_classification"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "task_3"

MODEL_DIR = PROJECT_ROOT / "models" / "task_3"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

VALIDATION_SIZE = 0.20

THRESHOLD_VALUES = np.round(
    np.arange(0.05, 0.501, 0.025),
    3
)


# ============================================================
# DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS
# TASK 03: CUISINE CLASSIFICATION
# PROGRAM 05: CUISINE MODEL THRESHOLD TUNING
# ============================================================

print("=" * 75)
print("DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS")
print("TASK 03: CUISINE CLASSIFICATION")
print("PROGRAM 05 - CUISINE MODEL THRESHOLD TUNING")
print("=" * 75)


# ============================================================
# 2. LOAD TRAINING DATA
# ============================================================

print("\n" + "=" * 65)
print("1. LOAD TRAINING DATA")
print("=" * 65)

required_files = [
    "cuisine_X_train_engineered.npz",
    "cuisine_y_train.npy"
]

for filename in required_files:

    file_path = DATA_DIR / filename

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )

label_mapping_path = (
    REPORT_DIR / "cuisine_label_mapping.csv"
)

if not label_mapping_path.exists():

    raise FileNotFoundError(
        f"Label mapping not found: {label_mapping_path}"
    )


X_train = load_npz(
    DATA_DIR / "cuisine_X_train_engineered.npz"
).tocsr()

y_train = np.load(
    DATA_DIR / "cuisine_y_train.npy"
)

label_mapping = pd.read_csv(
    label_mapping_path
)

cuisine_classes = (
    label_mapping["Cuisine Label"]
    .astype(str)
    .to_list()
)

print("Training feature matrix:", X_train.shape)
print("Training target matrix:", y_train.shape)
print("Cuisine classes:", len(cuisine_classes))


# ============================================================
# 3. VALIDATE TRAINING DATA
# ============================================================

print("\n" + "=" * 65)
print("2. VALIDATE TRAINING DATA")
print("=" * 65)

assert X_train.shape[0] == y_train.shape[0]

assert y_train.shape[1] == len(cuisine_classes)

assert set(np.unique(y_train)).issubset({0, 1})

assert (y_train.sum(axis=1) >= 1).all()

assert (y_train.sum(axis=0) > 0).all()

assert len(cuisine_classes) == len(set(cuisine_classes))

print("Feature and target alignment: PASS")
print("Binary target validation: PASS")
print("Training class coverage: PASS")

print(
    "Training records:",
    X_train.shape[0]
)

print(
    "Total cuisine classes:",
    len(cuisine_classes)
)


# ============================================================
# 4. CREATE INTERNAL VALIDATION SPLIT
# ============================================================

print("\n" + "=" * 65)
print("3. CREATE INTERNAL VALIDATION SPLIT")
print("=" * 65)

# Preserve at least one positive training example
# for every cuisine class.

rng = np.random.default_rng(RANDOM_STATE)

total_records = len(y_train)

validation_records = int(
    round(total_records * VALIDATION_SIZE)
)

class_support = y_train.sum(axis=0).astype(int)

remaining_support = class_support.copy()

shuffled_indices = rng.permutation(total_records)

validation_indices = []

for index in shuffled_indices:

    if len(validation_indices) >= validation_records:
        break

    active_labels = np.flatnonzero(y_train[index])

    # Do not remove the last positive training
    # example of any cuisine class.

    if np.all(remaining_support[active_labels] > 1):

        validation_indices.append(index)

        remaining_support[active_labels] -= 1


validation_indices = np.asarray(
    validation_indices,
    dtype=int
)

training_indices = np.setdiff1d(
    np.arange(total_records),
    validation_indices
)

if len(validation_indices) == 0:

    raise ValueError(
        "Unable to create an internal validation split."
    )

X_subtrain = X_train[training_indices]

X_validation = X_train[validation_indices]

y_subtrain = y_train[training_indices]

y_validation = y_train[validation_indices]


print("Sub-training records:", len(training_indices))

print("Validation records:", len(validation_indices))

print(
    "Validation percentage:",
    round(
        len(validation_indices) / total_records * 100,
        2
    ),
    "%"
)

print(
    "Classes represented in sub-training:",
    int((y_subtrain.sum(axis=0) > 0).sum())
)

print(
    "Classes represented in validation:",
    int((y_validation.sum(axis=0) > 0).sum())
)


# ============================================================
# 5. VALIDATE INTERNAL SPLIT
# ============================================================

print("\n" + "=" * 65)
print("4. VALIDATE INTERNAL SPLIT")
print("=" * 65)

assert len(training_indices) + len(validation_indices) == total_records

assert len(np.intersect1d(
    training_indices,
    validation_indices
)) == 0

assert (y_subtrain.sum(axis=0) > 0).all()

assert X_subtrain.shape[0] == y_subtrain.shape[0]

assert X_validation.shape[0] == y_validation.shape[0]

assert y_subtrain.shape[1] == y_validation.shape[1]

print("No training-validation overlap: PASS")
print("All classes retained in sub-training: PASS")
print("Sub-training alignment: PASS")
print("Validation alignment: PASS")


# ============================================================
# 6. DEFINE LOGISTIC REGRESSION MODEL
# ============================================================

print("\n" + "=" * 65)
print("5. DEFINE LOGISTIC REGRESSION MODEL")
print("=" * 65)

model = OneVsRestClassifier(
    LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        random_state=RANDOM_STATE
    ),
    n_jobs=-1
)

print("Model: Logistic Regression")

print("Classification strategy: Binary Relevance")

print("Implementation: OneVsRestClassifier")


# ============================================================
# 7. TRAIN MODEL ON SUB-TRAINING DATA
# ============================================================

print("\n" + "=" * 65)
print("6. TRAIN MODEL ON SUB-TRAINING DATA")
print("=" * 65)

print("Training started...")

with warnings.catch_warnings(record=True) as caught_warnings:

    warnings.simplefilter("always")

    model.fit(
        X_subtrain,
        y_subtrain
    )

print("Training completed.")

if caught_warnings:

    print(
        "Warnings during training:",
        len(caught_warnings)
    )

    for warning_item in caught_warnings[:10]:

        print(
            "-",
            str(warning_item.message)
        )


# ============================================================
# 8. GENERATE VALIDATION PROBABILITIES
# ============================================================

print("\n" + "=" * 65)
print("7. GENERATE VALIDATION PROBABILITIES")
print("=" * 65)

validation_probabilities = model.predict_proba(
    X_validation
)

validation_probabilities = np.asarray(
    validation_probabilities,
    dtype=np.float64
)

assert validation_probabilities.shape == y_validation.shape

assert np.isfinite(validation_probabilities).all()

assert (
    (validation_probabilities >= 0)
    & (validation_probabilities <= 1)
).all()

print(
    "Probability matrix shape:",
    validation_probabilities.shape
)

print("Probability generation: PASS")


# ============================================================
# 9. EVALUATE DIFFERENT PREDICTION THRESHOLDS
# ============================================================

print("\n" + "=" * 65)
print("8. EVALUATE PREDICTION THRESHOLDS")
print("=" * 65)

threshold_results = []

for threshold in THRESHOLD_VALUES:

    predictions = (
        validation_probabilities >= threshold
    ).astype(np.uint8)

    micro_precision = precision_score(
        y_validation,
        predictions,
        average="micro",
        zero_division=0
    )

    micro_recall = recall_score(
        y_validation,
        predictions,
        average="micro",
        zero_division=0
    )

    micro_f1 = f1_score(
        y_validation,
        predictions,
        average="micro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_validation,
        predictions,
        average="macro",
        zero_division=0
    )

    h_loss = hamming_loss(
        y_validation,
        predictions
    )

    threshold_results.append({

        "Threshold": float(threshold),

        "Micro Precision": micro_precision,

        "Micro Recall": micro_recall,

        "Micro F1": micro_f1,

        "Macro F1": macro_f1,

        "Hamming Loss": h_loss,

        "Predicted Labels": int(
            predictions.sum()
        )

    })


threshold_results_df = pd.DataFrame(
    threshold_results
)

print(
    threshold_results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)


# ============================================================
# 10. SELECT OPTIMAL THRESHOLD
# ============================================================

print("\n" + "=" * 65)
print("9. SELECT OPTIMAL THRESHOLD")
print("=" * 65)

threshold_results_df = threshold_results_df.sort_values(
    by=[
        "Micro F1",
        "Micro Recall",
        "Threshold"
    ],
    ascending=[
        False,
        False,
        False
    ]
).reset_index(drop=True)

best_threshold = float(
    threshold_results_df.iloc[0]["Threshold"]
)

best_validation_result = threshold_results_df.iloc[0]

print("Selected threshold:", best_threshold)

print(
    "Validation Micro Precision:",
    round(
        best_validation_result["Micro Precision"],
        4
    )
)

print(
    "Validation Micro Recall:",
    round(
        best_validation_result["Micro Recall"],
        4
    )
)

print(
    "Validation Micro F1:",
    round(
        best_validation_result["Micro F1"],
        4
    )
)

print(
    "Validation Macro F1:",
    round(
        best_validation_result["Macro F1"],
        4
    )
)

print(
    "Validation Hamming Loss:",
    round(
        best_validation_result["Hamming Loss"],
        4
    )
)


# ============================================================
# 11. SAVE THRESHOLD EVALUATION REPORT
# ============================================================

print("\n" + "=" * 65)
print("10. SAVE THRESHOLD EVALUATION REPORT")
print("=" * 65)

threshold_report_path = (
    REPORT_DIR / "cuisine_threshold_comparison.csv"
)

threshold_results_df.to_csv(
    threshold_report_path,
    index=False
)

print(
    "Threshold comparison saved:",
    threshold_report_path.name
)


# ============================================================
# 12. REFIT MODEL ON COMPLETE TRAINING DATA
# ============================================================

print("\n" + "=" * 65)
print("11. REFIT MODEL ON COMPLETE TRAINING DATA")
print("=" * 65)

final_model = OneVsRestClassifier(
    LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        random_state=RANDOM_STATE
    ),
    n_jobs=-1
)

print("Final model training started...")

with warnings.catch_warnings(record=True) as caught_warnings:

    warnings.simplefilter("always")

    final_model.fit(
        X_train,
        y_train
    )

print("Final model training completed.")

if caught_warnings:

    print(
        "Warnings during final training:",
        len(caught_warnings)
    )

    for warning_item in caught_warnings[:10]:

        print(
            "-",
            str(warning_item.message)
        )


# ============================================================
# 13. SAVE TUNED MODEL AND THRESHOLD
# ============================================================

print("\n" + "=" * 65)
print("12. SAVE TUNED MODEL AND THRESHOLD")
print("=" * 65)

model_path = (
    MODEL_DIR / "tuned_cuisine_classification_model.joblib"
)

threshold_path = (
    MODEL_DIR / "cuisine_classification_threshold.joblib"
)

joblib.dump(
    final_model,
    model_path
)

joblib.dump(
    best_threshold,
    threshold_path
)

print("Tuned model saved:")
print(model_path.name)

print("Selected threshold saved:")
print(threshold_path.name)


# ============================================================
# 14. SAVE TUNING SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("13. SAVE TUNING SUMMARY")
print("=" * 65)

summary = pd.DataFrame({

    "Metric": [

        "Original Training Records",

        "Sub-training Records",

        "Validation Records",

        "Cuisine Classes",

        "Model",

        "Classification Strategy",

        "Threshold Search Range",

        "Threshold Search Step",

        "Selected Threshold",

        "Validation Micro Precision",

        "Validation Micro Recall",

        "Validation Micro F1",

        "Validation Macro F1",

        "Validation Hamming Loss",

        "Final Model Training Records"

    ],

    "Value": [

        len(y_train),

        len(y_subtrain),

        len(y_validation),

        len(cuisine_classes),

        "Logistic Regression",

        "OneVsRestClassifier",

        "0.05 to 0.50",

        "0.025",

        best_threshold,

        best_validation_result["Micro Precision"],

        best_validation_result["Micro Recall"],

        best_validation_result["Micro F1"],

        best_validation_result["Macro F1"],

        best_validation_result["Hamming Loss"],

        len(y_train)

    ]

})

print(
    summary.to_string(index=False)
)

summary_path = (
    REPORT_DIR / "cuisine_threshold_tuning_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)

print("Tuning summary saved.")


# ============================================================
# 15. FINAL THRESHOLD TUNING VALIDATION
# ============================================================

print("\n" + "=" * 65)
print("14. FINAL THRESHOLD TUNING VALIDATION")
print("=" * 65)

assert 0 < best_threshold < 1

assert len(threshold_results_df) == len(THRESHOLD_VALUES)

assert threshold_results_df["Micro F1"].between(0, 1).all()

assert threshold_results_df["Micro Recall"].between(0, 1).all()

assert threshold_results_df["Micro Precision"].between(0, 1).all()

assert model_path.exists()

assert threshold_path.exists()

assert threshold_report_path.exists()

assert summary_path.exists()

# Verify saved model and threshold can be loaded.

loaded_model = joblib.load(model_path)

loaded_threshold = joblib.load(threshold_path)

assert loaded_threshold == best_threshold

assert isinstance(
    loaded_model,
    OneVsRestClassifier
)

print("Threshold selection validation: PASS")

print("Model artifact validation: PASS")

print("Threshold artifact validation: PASS")

print("Report generation validation: PASS")

print("Saved model reload validation: PASS")


# ============================================================
# 16. DISPLAY GENERATED FILES
# ============================================================

print("\n" + "=" * 65)
print("15. GENERATED FILES")
print("=" * 65)

print("\nGenerated models:")

print("- tuned_cuisine_classification_model.joblib")

print("- cuisine_classification_threshold.joblib")

print("\nGenerated reports:")

print("- cuisine_threshold_comparison.csv")

print("- cuisine_threshold_tuning_summary.csv")


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 75)

print("TASK 03 - CUISINE CLASSIFICATION")

print("PROGRAM 05 - CUISINE MODEL THRESHOLD TUNING COMPLETED")

print("=" * 75)
