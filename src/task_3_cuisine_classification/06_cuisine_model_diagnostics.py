from pathlib import Path

import warnings

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.sparse import load_npz

from sklearn.metrics import (
    accuracy_score,
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

FIGURE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "task_3"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

SELECTED_THRESHOLD = 0.225


# ============================================================
# DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS
# TASK 03: CUISINE CLASSIFICATION
# PROGRAM 06: CUISINE MODEL DIAGNOSTICS
# ============================================================

print("=" * 75)
print("DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS")
print("TASK 03: CUISINE CLASSIFICATION")
print("PROGRAM 06 - CUISINE MODEL DIAGNOSTICS")
print("=" * 75)


# ============================================================
# 2. DEFINE INPUT AND OUTPUT PATHS
# ============================================================

print("\n" + "=" * 65)
print("1. CONFIGURE INPUT AND OUTPUT FILES")
print("=" * 65)

X_TEST_PATH = DATA_DIR / "cuisine_X_test_engineered.npz"
Y_TEST_PATH = DATA_DIR / "cuisine_y_test.npy"

BASELINE_MODEL_PATH = (
    MODEL_DIR / "best_cuisine_classification_model.joblib"
)

TUNED_MODEL_PATH = (
    MODEL_DIR / "tuned_cuisine_classification_model.joblib"
)

THRESHOLD_PATH = (
    MODEL_DIR / "cuisine_classification_threshold.joblib"
)

LABEL_MAPPING_PATH = (
    REPORT_DIR / "cuisine_label_mapping.csv"
)

required_files = [
    X_TEST_PATH,
    Y_TEST_PATH,
    BASELINE_MODEL_PATH,
    TUNED_MODEL_PATH,
    THRESHOLD_PATH,
    LABEL_MAPPING_PATH
]

for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )


# ============================================================
# 3. LOAD TEST DATA AND MODEL ARTIFACTS
# ============================================================

print("\n" + "=" * 65)
print("2. LOAD TEST DATA AND MODEL ARTIFACTS")
print("=" * 65)

X_test = load_npz(X_TEST_PATH).tocsr()

y_test = np.load(Y_TEST_PATH)

baseline_model = joblib.load(BASELINE_MODEL_PATH)

tuned_model = joblib.load(TUNED_MODEL_PATH)

saved_threshold = joblib.load(THRESHOLD_PATH)

label_mapping = pd.read_csv(LABEL_MAPPING_PATH)

cuisine_classes = (
    label_mapping["Cuisine Label"]
    .astype(str)
    .to_list()
)

print("Testing feature matrix:", X_test.shape)
print("Testing target matrix:", y_test.shape)

print("Cuisine classes:", len(cuisine_classes))

print("Baseline model:", type(baseline_model).__name__)

print("Tuned model:", type(tuned_model).__name__)

print("Saved threshold:", saved_threshold)


# ============================================================
# 4. VALIDATE TEST DATA AND MODEL ARTIFACTS
# ============================================================

print("\n" + "=" * 65)
print("3. VALIDATE TEST DATA AND MODEL ARTIFACTS")
print("=" * 65)

assert X_test.shape[0] == y_test.shape[0]

assert y_test.shape[1] == len(cuisine_classes)

assert set(np.unique(y_test)).issubset({0, 1})

assert X_test.shape[1] > 0

assert np.isfinite(X_test.data).all()

assert 0.0 < float(saved_threshold) < 1.0

assert np.isclose(
    float(saved_threshold),
    SELECTED_THRESHOLD
)

test_support = y_test.sum(axis=0)

supported_class_indices = np.where(
    test_support > 0
)[0]

unsupported_class_indices = np.where(
    test_support == 0
)[0]

assert len(supported_class_indices) > 0

print("Feature and target alignment: PASS")
print("Binary target validation: PASS")
print("Cuisine label alignment: PASS")
print("Test feature validation: PASS")
print("Threshold artifact validation: PASS")

print(
    "Classes represented in test:",
    len(supported_class_indices)
)

print(
    "Classes absent from test:",
    len(unsupported_class_indices)
)


# ============================================================
# 5. GENERATE BASELINE MODEL PREDICTIONS
# ============================================================

print("\n" + "=" * 65)
print("4. GENERATE BASELINE MODEL PREDICTIONS")
print("=" * 65)

print("Generating predictions using Program 04 model...")

baseline_predictions = baseline_model.predict(X_test)

baseline_predictions = np.asarray(
    baseline_predictions,
    dtype=np.uint8
)

assert baseline_predictions.shape == y_test.shape

assert set(np.unique(baseline_predictions)).issubset({0, 1})

print("Baseline predictions generated.")

print("Baseline prediction shape:", baseline_predictions.shape)

print(
    "Baseline predicted labels:",
    int(baseline_predictions.sum())
)


# ============================================================
# 6. GENERATE TUNED MODEL PREDICTIONS
# ============================================================

print("\n" + "=" * 65)
print("5. GENERATE TUNED MODEL PREDICTIONS")
print("=" * 65)

print("Generating probabilities using tuned Logistic Regression...")

if not hasattr(tuned_model, "predict_proba"):

    raise AttributeError(
        "Tuned model does not support predict_proba()."
    )

tuned_probabilities = tuned_model.predict_proba(X_test)

tuned_probabilities = np.asarray(
    tuned_probabilities,
    dtype=np.float64
)

assert tuned_probabilities.shape == y_test.shape

assert np.isfinite(tuned_probabilities).all()

assert (
    (tuned_probabilities >= 0.0)
    & (tuned_probabilities <= 1.0)
).all()

tuned_predictions = (
    tuned_probabilities >= float(saved_threshold)
).astype(np.uint8)

assert tuned_predictions.shape == y_test.shape

print("Tuned probabilities generated.")

print("Probability matrix shape:", tuned_probabilities.shape)

print("Selected threshold:", saved_threshold)

print("Tuned predictions generated.")

print(
    "Tuned predicted labels:",
    int(tuned_predictions.sum())
)


# ============================================================
# 7. DEFINE EVALUATION FUNCTION
# ============================================================

print("\n" + "=" * 65)
print("6. DEFINE MULTI-LABEL EVALUATION METRICS")
print("=" * 65)


def evaluate_model(model_name, predictions):

    metrics = {

        "Model": model_name,

        "Micro Precision": precision_score(
            y_test,
            predictions,
            average="micro",
            zero_division=0
        ),

        "Micro Recall": recall_score(
            y_test,
            predictions,
            average="micro",
            zero_division=0
        ),

        "Micro F1": f1_score(
            y_test,
            predictions,
            average="micro",
            zero_division=0
        ),

        "Macro F1 - All Classes": f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        ),

        "Macro F1 - Test Supported": f1_score(
            y_test[:, supported_class_indices],
            predictions[:, supported_class_indices],
            average="macro",
            zero_division=0
        ),

        "Weighted F1": f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),

        "Subset Accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "Hamming Loss": hamming_loss(
            y_test,
            predictions
        ),

        "Predicted Labels": int(
            predictions.sum()
        ),

        "Actual Labels": int(
            y_test.sum()
        )
    }

    return metrics


# ============================================================
# 8. COMPARE BASELINE AND TUNED MODELS
# ============================================================

print("\n" + "=" * 65)
print("7. COMPARE BASELINE AND TUNED MODELS")
print("=" * 65)

baseline_metrics = evaluate_model(
    "Baseline - Program 04",
    baseline_predictions
)

tuned_metrics = evaluate_model(
    "Tuned Logistic Regression - Program 05",
    tuned_predictions
)

comparison_df = pd.DataFrame([
    baseline_metrics,
    tuned_metrics
])

comparison_df.to_csv(
    REPORT_DIR / "cuisine_model_diagnostics_comparison.csv",
    index=False
)

print("\nModel comparison:")

print(
    comparison_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)

print("\nModel comparison report saved.")


# ============================================================
# 9. CALCULATE PER-CUISINE DIAGNOSTICS
# ============================================================

print("\n" + "=" * 65)
print("8. CALCULATE PER-CUISINE DIAGNOSTICS")
print("=" * 65)

per_class_metrics = []

for index, cuisine in enumerate(cuisine_classes):

    actual = y_test[:, index]

    baseline_predicted = baseline_predictions[:, index]

    tuned_predicted = tuned_predictions[:, index]

    true_positive = int(
        np.sum((actual == 1) & (tuned_predicted == 1))
    )

    false_positive = int(
        np.sum((actual == 0) & (tuned_predicted == 1))
    )

    false_negative = int(
        np.sum((actual == 1) & (tuned_predicted == 0))
    )

    true_negative = int(
        np.sum((actual == 0) & (tuned_predicted == 0))
    )

    baseline_f1 = f1_score(
        actual,
        baseline_predicted,
        zero_division=0
    )

    tuned_precision = precision_score(
        actual,
        tuned_predicted,
        zero_division=0
    )

    tuned_recall = recall_score(
        actual,
        tuned_predicted,
        zero_division=0
    )

    tuned_f1 = f1_score(
        actual,
        tuned_predicted,
        zero_division=0
    )

    specificity = (
        true_negative / (true_negative + false_positive)
        if (true_negative + false_positive) > 0
        else 0.0
    )

    per_class_metrics.append({

        "Cuisine": cuisine,

        "Training Support": np.nan,

        "Testing Support": int(test_support[index]),

        "Baseline Predicted Support": int(
            baseline_predicted.sum()
        ),

        "Tuned Predicted Support": int(
            tuned_predicted.sum()
        ),

        "True Positive": true_positive,

        "False Positive": false_positive,

        "False Negative": false_negative,

        "True Negative": true_negative,

        "Baseline F1": baseline_f1,

        "Tuned Precision": tuned_precision,

        "Tuned Recall": tuned_recall,

        "Tuned F1": tuned_f1,

        "Tuned Specificity": specificity,

        "Test Support Available": bool(
            test_support[index] > 0
        )
    })


per_class_df = pd.DataFrame(per_class_metrics)

# Add training support from the original training targets.
y_train = np.load(
    DATA_DIR / "cuisine_y_train.npy"
)

assert y_train.shape[1] == len(cuisine_classes)

per_class_df["Training Support"] = (
    y_train.sum(axis=0).astype(int)
)

per_class_df.to_csv(
    REPORT_DIR / "cuisine_model_detailed_diagnostics.csv",
    index=False
)

print("Per-cuisine diagnostics generated.")

print(
    "Per-cuisine report saved:",
    "cuisine_model_detailed_diagnostics.csv"
)


# ============================================================
# 10. ANALYSE FALSE POSITIVES AND FALSE NEGATIVES
# ============================================================

print("\n" + "=" * 65)
print("9. ANALYSE FALSE POSITIVES AND FALSE NEGATIVES")
print("=" * 65)

total_false_positives = int(
    per_class_df["False Positive"].sum()
)

total_false_negatives = int(
    per_class_df["False Negative"].sum()
)

total_true_positives = int(
    per_class_df["True Positive"].sum()
)

total_true_negatives = int(
    per_class_df["True Negative"].sum()
)

error_summary = pd.DataFrame({

    "Metric": [
        "True Positives",
        "False Positives",
        "False Negatives",
        "True Negatives",
        "Actual Positive Labels",
        "Predicted Positive Labels"
    ],

    "Value": [
        total_true_positives,
        total_false_positives,
        total_false_negatives,
        total_true_negatives,
        int(y_test.sum()),
        int(tuned_predictions.sum())
    ]
})

error_summary.to_csv(
    REPORT_DIR / "cuisine_prediction_error_analysis.csv",
    index=False
)

print(error_summary.to_string(index=False))

print("\nError analysis report saved.")


# ============================================================
# 11. ANALYSE COMMON AND RARE CUISINES
# ============================================================

print("\n" + "=" * 65)
print("10. ANALYSE COMMON AND RARE CUISINES")
print("=" * 65)

supported_df = per_class_df[
    per_class_df["Test Support Available"]
].copy()

supported_df = supported_df.sort_values(
    "Testing Support",
    ascending=False
)

common_cuisines = supported_df.head(15)

rare_cuisines = supported_df.tail(15).sort_values(
    "Testing Support",
    ascending=True
)

print("\nTop 15 cuisines by test support:")

print(
    common_cuisines[
        [
            "Cuisine",
            "Testing Support",
            "Tuned Precision",
            "Tuned Recall",
            "Tuned F1"
        ]
    ].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)

print("\n15 least-supported cuisines represented in test:")

print(
    rare_cuisines[
        [
            "Cuisine",
            "Testing Support",
            "Tuned Precision",
            "Tuned Recall",
            "Tuned F1"
        ]
    ].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)

common_cuisines.to_csv(
    REPORT_DIR / "cuisine_common_class_diagnostics.csv",
    index=False
)

rare_cuisines.to_csv(
    REPORT_DIR / "cuisine_rare_class_diagnostics.csv",
    index=False
)


# ============================================================
# 12. SAVE TUNED TEST PREDICTIONS
# ============================================================

print("\n" + "=" * 65)
print("11. SAVE TUNED TEST PREDICTIONS")
print("=" * 65)

prediction_df = pd.DataFrame(
    tuned_predictions,
    columns=cuisine_classes
)

prediction_df.insert(
    0,
    "Test Row Number",
    np.arange(len(tuned_predictions))
)

prediction_df.to_csv(
    REPORT_DIR / "cuisine_tuned_test_predictions.csv",
    index=False
)

np.save(
    DATA_DIR / "cuisine_tuned_test_predictions.npy",
    tuned_predictions
)

np.save(
    DATA_DIR / "cuisine_tuned_test_probabilities.npy",
    tuned_probabilities
)

print("Tuned test predictions saved.")

print("Tuned test probabilities saved.")


# ============================================================
# 13. GENERATE MODEL COMPARISON FIGURE
# ============================================================

print("\n" + "=" * 65)
print("12. GENERATE MODEL COMPARISON FIGURE")
print("=" * 65)

metric_names = [
    "Micro Precision",
    "Micro Recall",
    "Micro F1",
    "Macro F1 - Test Supported",
    "Weighted F1",
    "Subset Accuracy"
]

baseline_values = [
    baseline_metrics[name]
    for name in metric_names
]

tuned_values = [
    tuned_metrics[name]
    for name in metric_names
]

x = np.arange(len(metric_names))

width = 0.35

plt.figure(figsize=(12, 7))

plt.bar(
    x - width / 2,
    baseline_values,
    width,
    label="Baseline"
)

plt.bar(
    x + width / 2,
    tuned_values,
    width,
    label="Tuned Logistic Regression"
)

plt.xticks(
    x,
    metric_names,
    rotation=25,
    ha="right"
)

plt.ylim(0, 1)

plt.ylabel("Score")

plt.xlabel("Evaluation Metrics")

plt.title(
    "DineScope - Baseline vs Tuned Cuisine Classification"
)

plt.legend()

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

comparison_figure_path = (
    FIGURE_DIR / "01_cuisine_model_comparison.png"
)

plt.savefig(
    comparison_figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", comparison_figure_path)


# ============================================================
# 14. GENERATE PER-CUISINE F1 FIGURE
# ============================================================

print("\n" + "=" * 65)
print("13. GENERATE PER-CUISINE F1 FIGURE")
print("=" * 65)

plot_data = supported_df.sort_values(
    "Tuned F1",
    ascending=True
).tail(20)

plt.figure(figsize=(12, 9))

plt.barh(
    plot_data["Cuisine"],
    plot_data["Tuned F1"]
)

plt.xlabel("F1 Score")

plt.ylabel("Cuisine")

plt.xlim(0, 1)

plt.title(
    "DineScope - Top 20 Test-Supported Cuisines by F1 Score"
)

plt.grid(
    axis="x",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

per_class_figure_path = (
    FIGURE_DIR / "02_cuisine_per_class_f1.png"
)

plt.savefig(
    per_class_figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", per_class_figure_path)


# ============================================================
# 15. GENERATE SUPPORT VS F1 FIGURE
# ============================================================

print("\n" + "=" * 65)
print("14. GENERATE CUISINE SUPPORT VS F1 FIGURE")
print("=" * 65)

plt.figure(figsize=(11, 7))

plt.scatter(
    supported_df["Testing Support"],
    supported_df["Tuned F1"],
    alpha=0.7
)

plt.xlabel("Testing Support")

plt.ylabel("Tuned F1 Score")

plt.ylim(0, 1.05)

plt.title(
    "DineScope - Cuisine Test Support vs Classification F1"
)

plt.grid(
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

support_figure_path = (
    FIGURE_DIR / "03_cuisine_support_vs_f1.png"
)

plt.savefig(
    support_figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", support_figure_path)


# ============================================================
# 16. GENERATE DIAGNOSTIC SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("15. GENERATE DIAGNOSTIC SUMMARY")
print("=" * 65)

summary = pd.DataFrame({

    "Metric": [

        "Testing Records",

        "Total Cuisine Classes",

        "Classes Represented in Test",

        "Classes Absent from Test",

        "Baseline Model",

        "Tuned Model",

        "Selected Threshold",

        "Baseline Micro F1",

        "Tuned Micro F1",

        "Baseline Micro Precision",

        "Tuned Micro Precision",

        "Baseline Micro Recall",

        "Tuned Micro Recall",

        "Baseline Macro F1 - Test Supported",

        "Tuned Macro F1 - Test Supported",

        "Baseline Hamming Loss",

        "Tuned Hamming Loss",

        "Tuned True Positives",

        "Tuned False Positives",

        "Tuned False Negatives",

        "Tuned True Negatives"
    ],

    "Value": [

        len(y_test),

        len(cuisine_classes),

        len(supported_class_indices),

        len(unsupported_class_indices),

        "Program 04 selected model",

        "Program 05 tuned Logistic Regression",

        float(saved_threshold),

        baseline_metrics["Micro F1"],

        tuned_metrics["Micro F1"],

        baseline_metrics["Micro Precision"],

        tuned_metrics["Micro Precision"],

        baseline_metrics["Micro Recall"],

        tuned_metrics["Micro Recall"],

        baseline_metrics["Macro F1 - Test Supported"],

        tuned_metrics["Macro F1 - Test Supported"],

        baseline_metrics["Hamming Loss"],

        tuned_metrics["Hamming Loss"],

        total_true_positives,

        total_false_positives,

        total_false_negatives,

        total_true_negatives
    ]
})

summary.to_csv(
    REPORT_DIR / "cuisine_model_diagnostics_summary.csv",
    index=False
)

print(summary.to_string(index=False))

print("\nDiagnostic summary saved.")


# ============================================================
# 17. FINAL DIAGNOSTIC VALIDATION
# ============================================================

print("\n" + "=" * 65)
print("16. FINAL DIAGNOSTIC VALIDATION")
print("=" * 65)

assert baseline_predictions.shape == y_test.shape

assert tuned_predictions.shape == y_test.shape

assert tuned_probabilities.shape == y_test.shape

assert np.isin(
    baseline_predictions,
    [0, 1]
).all()

assert np.isin(
    tuned_predictions,
    [0, 1]
).all()

assert np.isfinite(
    tuned_probabilities
).all()

assert comparison_df["Micro F1"].between(0, 1).all()

assert comparison_df["Hamming Loss"].between(0, 1).all()

assert per_class_df["Tuned F1"].between(0, 1).all()

assert per_class_df["Tuned Precision"].between(0, 1).all()

assert per_class_df["Tuned Recall"].between(0, 1).all()

assert (
    per_class_df["True Positive"]
    + per_class_df["False Negative"]
    == per_class_df["Testing Support"]
).all()

assert (
    per_class_df["True Positive"]
    + per_class_df["False Positive"]
    == per_class_df["Tuned Predicted Support"]
).all()

assert (
    REPORT_DIR / "cuisine_model_diagnostics_comparison.csv"
).exists()

assert (
    REPORT_DIR / "cuisine_model_detailed_diagnostics.csv"
).exists()

assert (
    REPORT_DIR / "cuisine_prediction_error_analysis.csv"
).exists()

assert (
    REPORT_DIR / "cuisine_tuned_test_predictions.csv"
).exists()

assert comparison_figure_path.exists()

assert per_class_figure_path.exists()

assert support_figure_path.exists()

print("Baseline prediction validation: PASS")
print("Tuned prediction validation: PASS")
print("Probability validation: PASS")
print("Model comparison validation: PASS")
print("Per-cuisine diagnostics validation: PASS")
print("False-positive validation: PASS")
print("False-negative validation: PASS")
print("Report generation validation: PASS")
print("Figure generation validation: PASS")


# ============================================================
# 18. DISPLAY GENERATED FILES
# ============================================================

print("\n" + "=" * 65)
print("17. GENERATED FILES")
print("=" * 65)

print("\nGenerated reports:")

generated_reports = [

    "cuisine_model_diagnostics_comparison.csv",

    "cuisine_model_detailed_diagnostics.csv",

    "cuisine_prediction_error_analysis.csv",

    "cuisine_common_class_diagnostics.csv",

    "cuisine_rare_class_diagnostics.csv",

    "cuisine_tuned_test_predictions.csv",

    "cuisine_model_diagnostics_summary.csv"
]

for filename in generated_reports:

    file_path = REPORT_DIR / filename

    assert file_path.exists()

    print("-", filename)


print("\nGenerated model data:")

print("- cuisine_tuned_test_predictions.npy")

print("- cuisine_tuned_test_probabilities.npy")


print("\nGenerated figures:")

print("- 01_cuisine_model_comparison.png")

print("- 02_cuisine_per_class_f1.png")

print("- 03_cuisine_support_vs_f1.png")


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 75)

print("TASK 03 - CUISINE CLASSIFICATION")

print("PROGRAM 06 - CUISINE MODEL DIAGNOSTICS COMPLETED")

print("=" * 75)
