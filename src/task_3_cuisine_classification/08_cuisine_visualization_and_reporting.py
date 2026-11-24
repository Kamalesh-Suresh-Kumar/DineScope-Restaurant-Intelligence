from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    hamming_loss,
    precision_recall_fscore_support
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

FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures" / "task_3"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TOP_N_CUISINES = 20
TOP_N_COMMON_CLASSES = 15
RARE_CLASS_THRESHOLD = 10


# ============================================================
# DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS
# TASK 03: CUISINE CLASSIFICATION
# PROGRAM 08: CUISINE VISUALIZATION AND REPORTING
# ============================================================

print("=" * 75)
print("DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS")
print("TASK 03: CUISINE CLASSIFICATION")
print("PROGRAM 08 - CUISINE VISUALIZATION AND REPORTING")
print("=" * 75)


# ============================================================
# 2. DEFINE INPUT FILES
# ============================================================

print("\n" + "=" * 65)
print("1. CONFIGURE INPUT FILES")
print("=" * 65)

ACTUAL_PATH = DATA_DIR / "cuisine_y_test.npy"

BASELINE_PREDICTION_PATH = (
    DATA_DIR / "cuisine_best_model_predictions.npy"
)

TUNED_PREDICTION_PATH = (
    DATA_DIR / "cuisine_tuned_test_predictions.npy"
)

PIPELINE_PREDICTION_PATH = (
    DATA_DIR / "cuisine_pipeline_predictions.npy"
)

PROBABILITY_PATH = (
    DATA_DIR / "cuisine_pipeline_probabilities.npy"
)

LABEL_MAPPING_PATH = (
    REPORT_DIR / "cuisine_label_mapping.csv"
)

THRESHOLD_PATH = (
    MODEL_DIR / "cuisine_classification_threshold.joblib"
)

required_files = [
    ACTUAL_PATH,
    BASELINE_PREDICTION_PATH,
    TUNED_PREDICTION_PATH,
    PIPELINE_PREDICTION_PATH,
    PROBABILITY_PATH,
    LABEL_MAPPING_PATH,
    THRESHOLD_PATH
]

for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )

print("All required input files are available.")


# ============================================================
# 3. LOAD TEST LABELS AND PREDICTIONS
# ============================================================

print("\n" + "=" * 65)
print("2. LOAD TEST LABELS AND PREDICTIONS")
print("=" * 65)

y_test = np.load(ACTUAL_PATH)

baseline_predictions = np.load(
    BASELINE_PREDICTION_PATH
)

tuned_predictions = np.load(
    TUNED_PREDICTION_PATH
)

pipeline_predictions = np.load(
    PIPELINE_PREDICTION_PATH
)

probabilities = np.load(
    PROBABILITY_PATH
)

label_mapping = pd.read_csv(
    LABEL_MAPPING_PATH
)

cuisine_classes = (
    label_mapping["Cuisine Label"]
    .astype(str)
    .to_list()
)

threshold = float(
    joblib.load(THRESHOLD_PATH)
)

print("Actual test labels:", y_test.shape)

print("Baseline predictions:", baseline_predictions.shape)

print("Tuned predictions:", tuned_predictions.shape)

print("Pipeline predictions:", pipeline_predictions.shape)

print("Prediction probabilities:", probabilities.shape)

print("Cuisine classes:", len(cuisine_classes))

print("Selected threshold:", threshold)


# ============================================================
# 4. VALIDATE INPUT MATRICES
# ============================================================

print("\n" + "=" * 65)
print("3. VALIDATE INPUT MATRICES")
print("=" * 65)

expected_shape = (
    y_test.shape[0],
    len(cuisine_classes)
)

assert y_test.ndim == 2

assert y_test.shape == expected_shape

assert baseline_predictions.shape == expected_shape

assert tuned_predictions.shape == expected_shape

assert pipeline_predictions.shape == expected_shape

assert probabilities.shape == expected_shape

assert np.isin(y_test, [0, 1]).all()

assert np.isin(baseline_predictions, [0, 1]).all()

assert np.isin(tuned_predictions, [0, 1]).all()

assert np.isin(pipeline_predictions, [0, 1]).all()

assert np.isfinite(probabilities).all()

assert (
    (probabilities >= 0.0)
    & (probabilities <= 1.0)
).all()

assert 0.0 < threshold < 1.0

assert len(cuisine_classes) == len(set(cuisine_classes))

assert np.array_equal(
    tuned_predictions,
    pipeline_predictions
)

print("Actual label matrix validation: PASS")

print("Baseline prediction validation: PASS")

print("Tuned prediction validation: PASS")

print("Probability matrix validation: PASS")

print("Cuisine mapping validation: PASS")

print("Program 06 and Program 07 consistency: PASS")


# ============================================================
# 5. ANALYSE DATASET LABEL COVERAGE
# ============================================================

print("\n" + "=" * 65)
print("4. ANALYSE CUISINE LABEL COVERAGE")
print("=" * 65)

actual_support = y_test.sum(axis=0)

baseline_support = baseline_predictions.sum(axis=0)

tuned_support = tuned_predictions.sum(axis=0)

label_coverage = pd.DataFrame({

    "Cuisine": cuisine_classes,

    "Actual Test Support": actual_support,

    "Baseline Predicted Count": baseline_support,

    "Tuned Predicted Count": tuned_support

})

label_coverage["Test Representation"] = np.where(
    label_coverage["Actual Test Support"] > 0,
    "Represented",
    "Not Represented"
)

represented_classes = int(
    (actual_support > 0).sum()
)

unrepresented_classes = int(
    (actual_support == 0).sum()
)

rare_classes = int(
    (
        (actual_support > 0)
        & (actual_support < RARE_CLASS_THRESHOLD)
    ).sum()
)

print("Total cuisine classes:", len(cuisine_classes))

print("Classes represented in test:", represented_classes)

print("Classes absent from test:", unrepresented_classes)

print(
    f"Represented classes with fewer than "
    f"{RARE_CLASS_THRESHOLD} test examples:",
    rare_classes
)

label_coverage.to_csv(
    REPORT_DIR / "cuisine_prediction_label_distribution.csv",
    index=False
)

print("Cuisine label distribution report saved.")


# ============================================================
# 6. CALCULATE MODEL PERFORMANCE METRICS
# ============================================================

print("\n" + "=" * 65)
print("5. CALCULATE MODEL PERFORMANCE METRICS")
print("=" * 65)


def calculate_model_metrics(
    y_true,
    y_pred,
    model_name
):

    micro_precision = precision_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0
    )

    micro_recall = recall_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0
    )

    micro_f1 = f1_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0
    )

    macro_f1_all = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    subset_accuracy = accuracy_score(
        y_true,
        y_pred
    )

    hamming = hamming_loss(
        y_true,
        y_pred
    )

    per_class_precision, per_class_recall, per_class_f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average=None,
            zero_division=0
        )
    )

    per_class_f1 = np.asarray(
        per_class_f1,
        dtype=np.float64
    )

    support = np.asarray(
        support,
        dtype=np.int64
    )

    supported_mask = support > 0

    macro_f1_supported = float(
        np.mean(per_class_f1[supported_mask])
    )

    return {

        "Model": model_name,

        "Micro Precision": micro_precision,

        "Micro Recall": micro_recall,

        "Micro F1": micro_f1,

        "Macro F1 All Classes": macro_f1_all,

        "Macro F1 Test-Supported": macro_f1_supported,

        "Weighted F1": weighted_f1,

        "Subset Accuracy": subset_accuracy,

        "Hamming Loss": hamming,

        "Predicted Labels": int(y_pred.sum()),

        "Actual Positive Labels": int(y_true.sum()),

        "True Positives": int(
            np.logical_and(
                y_true == 1,
                y_pred == 1
            ).sum()
        ),

        "False Positives": int(
            np.logical_and(
                y_true == 0,
                y_pred == 1
            ).sum()
        ),

        "False Negatives": int(
            np.logical_and(
                y_true == 1,
                y_pred == 0
            ).sum()
        ),

        "True Negatives": int(
            np.logical_and(
                y_true == 0,
                y_pred == 0
            ).sum()
        )

    }


baseline_metrics = calculate_model_metrics(
    y_test,
    baseline_predictions,
    "Baseline Logistic Regression"
)

tuned_metrics = calculate_model_metrics(
    y_test,
    tuned_predictions,
    "Tuned Logistic Regression"
)

comparison_df = pd.DataFrame([
    baseline_metrics,
    tuned_metrics
])

print("\nMODEL PERFORMANCE COMPARISON")

print(
    comparison_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)

comparison_df.to_csv(
    REPORT_DIR / "cuisine_final_model_comparison.csv",
    index=False
)

print("\nModel comparison report saved.")


# ============================================================
# 7. GENERATE PER-CUISINE PERFORMANCE REPORT
# ============================================================

print("\n" + "=" * 65)
print("6. GENERATE PER-CUISINE PERFORMANCE REPORT")
print("=" * 65)

baseline_precision, baseline_recall, baseline_f1, support = (
    precision_recall_fscore_support(
        y_test,
        baseline_predictions,
        average=None,
        zero_division=0
    )
)

tuned_precision, tuned_recall, tuned_f1, _ = (
    precision_recall_fscore_support(
        y_test,
        tuned_predictions,
        average=None,
        zero_division=0
    )
)

per_class_df = pd.DataFrame({

    "Cuisine": cuisine_classes,

    "Test Support": support,

    "Baseline Precision": baseline_precision,

    "Baseline Recall": baseline_recall,

    "Baseline F1": baseline_f1,

    "Tuned Precision": tuned_precision,

    "Tuned Recall": tuned_recall,

    "Tuned F1": tuned_f1

})

per_class_df["F1 Improvement"] = (
    per_class_df["Tuned F1"]
    - per_class_df["Baseline F1"]
)

per_class_df["Class Category"] = np.where(
    per_class_df["Test Support"] == 0,
    "Absent from test",
    np.where(
        per_class_df["Test Support"] < RARE_CLASS_THRESHOLD,
        "Rare",
        "Common"
    )
)

per_class_df = per_class_df.sort_values(
    by="Test Support",
    ascending=False
).reset_index(drop=True)

per_class_df.to_csv(
    REPORT_DIR / "cuisine_final_per_class_metrics.csv",
    index=False
)

print("Per-cuisine metrics generated.")

print("\nTop 15 cuisines by test support:")

print(
    per_class_df.head(15).to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)

print("\nPer-cuisine performance report saved.")


# ============================================================
# 8. VISUALIZE BASELINE AND TUNED PERFORMANCE
# ============================================================

print("\n" + "=" * 65)
print("7. VISUALIZE MODEL PERFORMANCE COMPARISON")
print("=" * 65)

performance_metrics = [
    "Micro Precision",
    "Micro Recall",
    "Micro F1",
    "Macro F1 All Classes",
    "Macro F1 Test-Supported",
    "Weighted F1",
    "Subset Accuracy"
]

x = np.arange(len(performance_metrics))

width = 0.35

plt.figure(figsize=(13, 6))

plt.bar(
    x - width / 2,
    comparison_df.iloc[0][performance_metrics].to_numpy(dtype=float),
    width,
    label="Baseline"
)

plt.bar(
    x + width / 2,
    comparison_df.iloc[1][performance_metrics].to_numpy(dtype=float),
    width,
    label="Tuned"
)

plt.xticks(
    x,
    performance_metrics,
    rotation=25,
    ha="right"
)

plt.ylim(0, 1)

plt.ylabel("Metric Score")

plt.xlabel("Evaluation Metric")

plt.title(
    "DineScope - Baseline vs Tuned Cuisine Classification"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "01_cuisine_model_performance_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: 01_cuisine_model_performance_comparison.png")


# ============================================================
# 9. VISUALIZE PER-CUISINE F1 SCORES
# ============================================================

print("\n" + "=" * 65)
print("8. VISUALIZE PER-CUISINE F1 SCORES")
print("=" * 65)

common_class_data = per_class_df[
    per_class_df["Test Support"] >= RARE_CLASS_THRESHOLD
].head(TOP_N_COMMON_CLASSES).copy()

plot_data = common_class_data.sort_values(
    by="Tuned F1",
    ascending=True
)

if not plot_data.empty:

    y_positions = np.arange(len(plot_data))

    plt.figure(figsize=(11, 8))

    plt.barh(
        y_positions - 0.18,
        plot_data["Baseline F1"],
        height=0.35,
        label="Baseline"
    )

    plt.barh(
        y_positions + 0.18,
        plot_data["Tuned F1"],
        height=0.35,
        label="Tuned"
    )

    plt.yticks(
        y_positions,
        plot_data["Cuisine"].astype(str).tolist()
    )

    plt.xlabel("F1 Score")

    plt.ylabel("Cuisine")

    plt.title(
        "DineScope - Per-Cuisine F1 Score Comparison"
    )

    plt.xlim(0, 1)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "02_cuisine_per_class_f1_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved: 02_cuisine_per_class_f1_comparison.png")


# ============================================================
# 10. VISUALIZE SUPPORT VS MODEL PERFORMANCE
# ============================================================

print("\n" + "=" * 65)
print("9. VISUALIZE CUISINE SUPPORT VS F1 SCORE")
print("=" * 65)

supported_data = per_class_df[
    per_class_df["Test Support"] > 0
].copy()

plt.figure(figsize=(10, 7))

plt.scatter(
    supported_data["Test Support"],
    supported_data["Tuned F1"],
    alpha=0.75
)

plt.xscale("log")

plt.xlabel("Actual Test Support (Log Scale)")

plt.ylabel("Tuned F1 Score")

plt.ylim(-0.05, 1.05)

plt.title(
    "DineScope - Cuisine Support vs Classification Performance"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "03_cuisine_support_vs_f1.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: 03_cuisine_support_vs_f1.png")


# ============================================================
# 11. VISUALIZE PREDICTED CUISINE DISTRIBUTION
# ============================================================

print("\n" + "=" * 65)
print("10. VISUALIZE PREDICTED CUISINE DISTRIBUTION")
print("=" * 65)

prediction_distribution = pd.DataFrame({

    "Cuisine": cuisine_classes,

    "Actual Count": y_test.sum(axis=0),

    "Baseline Predicted Count": baseline_predictions.sum(axis=0),

    "Tuned Predicted Count": tuned_predictions.sum(axis=0)

})

prediction_distribution = prediction_distribution.sort_values(
    by="Tuned Predicted Count",
    ascending=False
).head(TOP_N_CUISINES)

plot_distribution = prediction_distribution.sort_values(
    by="Tuned Predicted Count",
    ascending=True
)

y_positions = np.arange(len(plot_distribution))

plt.figure(figsize=(12, 8))

plt.barh(
    y_positions - 0.25,
    plot_distribution["Actual Count"],
    height=0.25,
    label="Actual"
)

plt.barh(
    y_positions,
    plot_distribution["Baseline Predicted Count"],
    height=0.25,
    label="Baseline Predicted"
)

plt.barh(
    y_positions + 0.25,
    plot_distribution["Tuned Predicted Count"],
    height=0.25,
    label="Tuned Predicted"
)

plt.yticks(
    y_positions,
    plot_distribution["Cuisine"].astype(str).tolist()
)

plt.xlabel("Restaurant Label Count")

plt.ylabel("Cuisine")

plt.title(
    "DineScope - Actual and Predicted Cuisine Distribution"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "04_cuisine_prediction_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

prediction_distribution.to_csv(
    REPORT_DIR / "cuisine_prediction_distribution.csv",
    index=False
)

print("Saved: 04_cuisine_prediction_distribution.png")

print("Cuisine prediction distribution report saved.")


# ============================================================
# 12. VISUALIZE PREDICTION COUNT PER RESTAURANT
# ============================================================

print("\n" + "=" * 65)
print("11. VISUALIZE PREDICTION COUNT PER RESTAURANT")
print("=" * 65)

actual_counts = y_test.sum(axis=1)

baseline_counts = baseline_predictions.sum(axis=1)

tuned_counts = tuned_predictions.sum(axis=1)

count_distribution = pd.DataFrame({

    "Actual Cuisine Count": actual_counts,

    "Baseline Predicted Cuisine Count": baseline_counts,

    "Tuned Predicted Cuisine Count": tuned_counts

})

count_distribution.to_csv(
    REPORT_DIR / "cuisine_prediction_count_per_restaurant.csv",
    index=False
)

plt.figure(figsize=(10, 6))

bins = (
    np.arange(
        0,
        max(
            int(actual_counts.max()),
            int(baseline_counts.max()),
            int(tuned_counts.max())
        ) + 2
    ) - 0.5
).tolist()

plt.hist(
    actual_counts,
    bins=bins,
    alpha=0.55,
    label="Actual"
)

plt.hist(
    baseline_counts,
    bins=bins,
    alpha=0.55,
    label="Baseline Predicted"
)

plt.hist(
    tuned_counts,
    bins=bins,
    alpha=0.55,
    label="Tuned Predicted"
)

plt.xlabel("Number of Cuisine Labels per Restaurant")

plt.ylabel("Number of Restaurants")

plt.title(
    "DineScope - Cuisine Labels per Restaurant"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "05_cuisine_prediction_count_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: 05_cuisine_prediction_count_distribution.png")

print("Prediction count report saved.")


# ============================================================
# 13. VISUALIZE TOP PREDICTION PROBABILITIES
# ============================================================

print("\n" + "=" * 65)
print("12. VISUALIZE PREDICTION PROBABILITY DISTRIBUTION")
print("=" * 65)

maximum_probabilities = probabilities.max(axis=1)

selected_probabilities = probabilities[
    pipeline_predictions == 1
]

plt.figure(figsize=(10, 6))

plt.hist(
    maximum_probabilities,
    bins=25,
    alpha=0.75,
    label="Maximum Cuisine Probability per Restaurant"
)

plt.axvline(
    threshold,
    linestyle="--",
    label=f"Selected Threshold ({threshold:.3f})"
)

plt.xlabel("Model Probability")

plt.ylabel("Number of Restaurants")

plt.title(
    "DineScope - Maximum Cuisine Probability Distribution"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "06_cuisine_probability_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: 06_cuisine_probability_distribution.png")

print(
    "Average maximum probability:",
    round(float(maximum_probabilities.mean()), 4)
)

if selected_probabilities.size > 0:

    print(
        "Average probability of predicted labels:",
        round(float(selected_probabilities.mean()), 4)
    )

else:

    print("No cuisine labels crossed the prediction threshold.")


# ============================================================
# 14. GENERATE FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("13. GENERATE FINAL CLASSIFICATION SUMMARY")
print("=" * 65)

summary = pd.DataFrame({

    "Metric": [

        "Test Restaurants",

        "Total Cuisine Classes",

        "Classes Represented in Test",

        "Classes Absent from Test",

        "Rare Classes in Test",

        "Prediction Threshold",

        "Baseline Micro F1",

        "Tuned Micro F1",

        "Baseline Macro F1 - All Classes",

        "Tuned Macro F1 - All Classes",

        "Baseline Macro F1 - Test-Supported",

        "Tuned Macro F1 - Test-Supported",

        "Baseline Weighted F1",

        "Tuned Weighted F1",

        "Baseline Subset Accuracy",

        "Tuned Subset Accuracy",

        "Baseline Hamming Loss",

        "Tuned Hamming Loss",

        "Baseline Predicted Labels",

        "Tuned Predicted Labels",

        "Restaurants With Tuned Predictions",

        "Restaurants Without Tuned Predictions",

        "Average Tuned Predictions Per Restaurant",

        "Program 06 and Program 07 Agreement"

    ],

    "Value": [

        len(y_test),

        len(cuisine_classes),

        represented_classes,

        unrepresented_classes,

        rare_classes,

        threshold,

        baseline_metrics["Micro F1"],

        tuned_metrics["Micro F1"],

        baseline_metrics["Macro F1 All Classes"],

        tuned_metrics["Macro F1 All Classes"],

        baseline_metrics["Macro F1 Test-Supported"],

        tuned_metrics["Macro F1 Test-Supported"],

        baseline_metrics["Weighted F1"],

        tuned_metrics["Weighted F1"],

        baseline_metrics["Subset Accuracy"],

        tuned_metrics["Subset Accuracy"],

        baseline_metrics["Hamming Loss"],

        tuned_metrics["Hamming Loss"],

        int(baseline_predictions.sum()),

        int(tuned_predictions.sum()),

        int((tuned_predictions.sum(axis=1) > 0).sum()),

        int((tuned_predictions.sum(axis=1) == 0).sum()),

        float(tuned_predictions.sum(axis=1).mean()),

        "PASS"

    ]

})

print(summary.to_string(index=False))

summary.to_csv(
    REPORT_DIR / "cuisine_final_classification_summary.csv",
    index=False
)

print("\nFinal classification summary saved.")


# ============================================================
# 15. GENERATE FINAL TEXT REPORT
# ============================================================

print("\n" + "=" * 65)
print("14. GENERATE FINAL TEXT REPORT")
print("=" * 65)

baseline_f1_value = baseline_metrics["Micro F1"]

tuned_f1_value = tuned_metrics["Micro F1"]

if baseline_f1_value > 0:

    micro_f1_change = (
        (tuned_f1_value - baseline_f1_value)
        / baseline_f1_value
    ) * 100

else:

    micro_f1_change = np.nan


report_lines = [

    "DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS",

    "TASK 03: CUISINE CLASSIFICATION",

    "FINAL VISUALIZATION AND REPORTING",

    "=" * 70,

    "",

    "1. DATASET AND CLASS INFORMATION",

    f"Test restaurants: {len(y_test)}",

    f"Total cuisine classes: {len(cuisine_classes)}",

    f"Classes represented in test: {represented_classes}",

    f"Classes absent from test: {unrepresented_classes}",

    f"Rare represented classes: {rare_classes}",

    "",

    "2. BASELINE MODEL PERFORMANCE",

    f"Micro Precision: {baseline_metrics['Micro Precision']:.4f}",

    f"Micro Recall: {baseline_metrics['Micro Recall']:.4f}",

    f"Micro F1: {baseline_metrics['Micro F1']:.4f}",

    f"Macro F1 - All Classes: {baseline_metrics['Macro F1 All Classes']:.4f}",

    f"Macro F1 - Test-Supported: {baseline_metrics['Macro F1 Test-Supported']:.4f}",

    f"Weighted F1: {baseline_metrics['Weighted F1']:.4f}",

    f"Subset Accuracy: {baseline_metrics['Subset Accuracy']:.4f}",

    f"Hamming Loss: {baseline_metrics['Hamming Loss']:.4f}",

    "",

    "3. TUNED MODEL PERFORMANCE",

    f"Micro Precision: {tuned_metrics['Micro Precision']:.4f}",

    f"Micro Recall: {tuned_metrics['Micro Recall']:.4f}",

    f"Micro F1: {tuned_metrics['Micro F1']:.4f}",

    f"Macro F1 - All Classes: {tuned_metrics['Macro F1 All Classes']:.4f}",

    f"Macro F1 - Test-Supported: {tuned_metrics['Macro F1 Test-Supported']:.4f}",

    f"Weighted F1: {tuned_metrics['Weighted F1']:.4f}",

    f"Subset Accuracy: {tuned_metrics['Subset Accuracy']:.4f}",

    f"Hamming Loss: {tuned_metrics['Hamming Loss']:.4f}",

    "",

    "4. MODEL COMPARISON",

    f"Micro F1 relative change: {micro_f1_change:.2f}%",

    f"Selected threshold: {threshold:.3f}",

    f"Baseline predicted labels: {baseline_predictions.sum()}",

    f"Tuned predicted labels: {tuned_predictions.sum()}",

    "",

    "5. PREDICTION PIPELINE",

    f"Restaurants with predictions: {(tuned_predictions.sum(axis=1) > 0).sum()}",

    f"Restaurants without predictions: {(tuned_predictions.sum(axis=1) == 0).sum()}",

    f"Average predicted cuisines per restaurant: {tuned_predictions.sum(axis=1).mean():.4f}",

    "Program 06 and Program 07 prediction agreement: 100%",

    "",

    "6. INTERPRETATION AND LIMITATIONS",

    "- The task is a multi-label cuisine classification problem.",

    "- A restaurant can be associated with multiple cuisine labels.",

    "- The tuned model improves recall and overall Micro F1 compared with the baseline.",

    "- The tuned model also produces more false-positive predictions.",

    "- The selected threshold controls the precision-recall trade-off.",

    "- Some cuisine classes have very limited test support.",

    "- Classes absent from the test set cannot be meaningfully evaluated on this test split.",

    "- Rare-class metrics should be interpreted cautiously.",

    "- Probability values are model estimates and have not been established as calibrated probabilities.",

    "- These results describe the held-out test set, not independent external validation.",

    "",

    "7. FINAL STATUS",

    "Program 08 visualization and reporting completed.",

    "Program 06 and Program 07 consistency: PASS",

    "",

    "=" * 70

]

text_report_path = (
    REPORT_DIR / "cuisine_final_classification_report.txt"
)

text_report_path.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

print("Final text report saved.")


# ============================================================
# 16. FINAL VALIDATION
# ============================================================

print("\n" + "=" * 65)
print("15. FINAL REPORTING VALIDATION")
print("=" * 65)

assert len(comparison_df) == 2

assert len(per_class_df) == len(cuisine_classes)

assert len(label_coverage) == len(cuisine_classes)

assert len(prediction_distribution) <= TOP_N_CUISINES

assert np.isfinite(
    comparison_df.select_dtypes(include=[np.number]).to_numpy()
).all()

assert np.isfinite(
    per_class_df.select_dtypes(include=[np.number]).to_numpy()
).all()

assert np.array_equal(
    tuned_predictions,
    pipeline_predictions
)

required_reports = [

    "cuisine_final_model_comparison.csv",

    "cuisine_final_per_class_metrics.csv",

    "cuisine_prediction_label_distribution.csv",

    "cuisine_prediction_distribution.csv",

    "cuisine_prediction_count_per_restaurant.csv",

    "cuisine_final_classification_summary.csv",

    "cuisine_final_classification_report.txt"

]

for filename in required_reports:

    assert (
        REPORT_DIR / filename
    ).exists(), f"Report missing: {filename}"


required_figures = [

    "01_cuisine_model_performance_comparison.png",

    "02_cuisine_per_class_f1_comparison.png",

    "03_cuisine_support_vs_f1.png",

    "04_cuisine_prediction_distribution.png",

    "05_cuisine_prediction_count_distribution.png",

    "06_cuisine_probability_distribution.png"

]

for filename in required_figures:

    assert (
        FIGURE_DIR / filename
    ).exists(), f"Figure missing: {filename}"


print("Model comparison validation: PASS")

print("Per-cuisine report validation: PASS")

print("Label coverage validation: PASS")

print("Prediction distribution validation: PASS")

print("Summary report validation: PASS")

print("Text report validation: PASS")

print("Visualization file validation: PASS")

print("Program 06 and Program 07 consistency: PASS")


# ============================================================
# 17. DISPLAY GENERATED OUTPUTS
# ============================================================

print("\n" + "=" * 65)
print("16. GENERATED REPORTS")
print("=" * 65)

for filename in required_reports:

    print("-", filename)


print("\nGenerated figures:")

for filename in required_figures:

    print("-", filename)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 75)

print("TASK 03 - CUISINE CLASSIFICATION")

print("PROGRAM 08 - CUISINE VISUALIZATION AND REPORTING COMPLETED")

print("=" * 75)
