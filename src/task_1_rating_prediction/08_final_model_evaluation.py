import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ============================================================
# DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS
# TASK 01: RESTAURANT RATING PREDICTION
# PROGRAM 08: FINAL MODEL EVALUATION AND SELECTION
# ============================================================

print("=" * 75)
print("DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS")
print("TASK 01: RESTAURANT RATING PREDICTION")
print("PROGRAM 08 - FINAL MODEL EVALUATION AND SELECTION")
print("=" * 75)

# ------------------------------------------------------------
# 1. PATH CONFIGURATION
# ------------------------------------------------------------

REPORT_DIR = str(PROJECT_ROOT / "reports" / "task_1")
FIGURE_DIR = str(PROJECT_ROOT / "outputs" / "figures" / "task_1")

TUNING_PATH = os.path.join(
    REPORT_DIR,
    "rating_tuning_comparison.csv"
)

DIAGNOSTICS_PATH = os.path.join(
    REPORT_DIR,
    "tuned_model_diagnostics_comparison.csv"
)

IMPORTANCE_WITHOUT_PATH = os.path.join(
    REPORT_DIR,
    "tuned_feature_importance_without_votes.csv"
)

IMPORTANCE_WITH_PATH = os.path.join(
    REPORT_DIR,
    "tuned_feature_importance_with_votes.csv"
)

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD PREVIOUS REPORTS
# ------------------------------------------------------------

print("\n1. LOADING PREVIOUS EVALUATION REPORTS")

required_files = [
    TUNING_PATH,
    DIAGNOSTICS_PATH,
    IMPORTANCE_WITHOUT_PATH,
    IMPORTANCE_WITH_PATH
]

for file_path in required_files:

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Required report not found: {file_path}"
        )

tuning_df = pd.read_csv(TUNING_PATH)
diagnostics_df = pd.read_csv(DIAGNOSTICS_PATH)

importance_without = pd.read_csv(IMPORTANCE_WITHOUT_PATH)
importance_with = pd.read_csv(IMPORTANCE_WITH_PATH)

print("Tuning comparison loaded:", tuning_df.shape)
print("Diagnostics comparison loaded:", diagnostics_df.shape)

print("Feature importance reports loaded successfully.")


# ------------------------------------------------------------
# 3. VALIDATE REQUIRED COLUMNS
# ------------------------------------------------------------

print("\n2. VALIDATING REPORT STRUCTURE")

tuning_columns = [
    "Configuration",
    "Best CV RMSE",
    "Baseline Test RMSE",
    "Tuned Test RMSE",
    "Baseline Test MAE",
    "Tuned Test MAE",
    "Baseline Test R2",
    "Tuned Test R2",
    "RMSE Improvement"
]

diagnostic_columns = [
    "Configuration",
    "Train MAE",
    "Train RMSE",
    "Train R2",
    "Test MAE",
    "Test MSE",
    "Test RMSE",
    "Test R2",
    "RMSE Gap",
    "R2 Gap",
    "Mean Residual",
    "Maximum Absolute Error"
]

for column in tuning_columns:

    if column not in tuning_df.columns:
        raise ValueError(
            f"Missing column in tuning report: {column}"
        )

for column in diagnostic_columns:

    if column not in diagnostics_df.columns:
        raise ValueError(
            f"Missing column in diagnostics report: {column}"
        )

expected_configs = {"without_votes", "with_votes"}

if set(tuning_df["Configuration"]) != expected_configs:
    raise ValueError("Unexpected configurations in tuning report.")

if set(diagnostics_df["Configuration"]) != expected_configs:
    raise ValueError("Unexpected configurations in diagnostics report.")

print("All required columns and configurations verified.")


# ------------------------------------------------------------
# 4. MERGE EVALUATION RESULTS
# ------------------------------------------------------------

print("\n3. MERGING EVALUATION RESULTS")

final_df = pd.merge(
    tuning_df,
    diagnostics_df,
    on="Configuration",
    suffixes=("_tuning", "_diagnostics"),
    validate="one_to_one"
)

# Use diagnostic metrics as the final training/test metrics.
final_df["Final Train RMSE"] = final_df["Train RMSE"]
final_df["Final Train R2"] = final_df["Train R2"]

final_df["Final Test MAE"] = final_df["Test MAE"]
final_df["Final Test MSE"] = final_df["Test MSE"]
final_df["Final Test RMSE"] = final_df["Test RMSE"]
final_df["Final Test R2"] = final_df["Test R2"]

# Calculate improvement from the baseline test RMSE.
final_df["RMSE Improvement Percentage"] = (
    final_df["RMSE Improvement"]
    / final_df["Baseline Test RMSE"]
) * 100

final_df["RMSE Gap"] = final_df["Test RMSE"] - final_df["Train RMSE"]

print("\nCombined evaluation:")
print(
    final_df[
        [
            "Configuration",
            "Best CV RMSE",
            "Final Test RMSE",
            "Final Test MAE",
            "Final Test R2",
            "RMSE Gap",
            "RMSE Improvement Percentage"
        ]
    ].to_string(index=False)
)


# ------------------------------------------------------------
# 5. CROSS-VALIDATION COMPARISON
# ------------------------------------------------------------

print("\n4. CROSS-VALIDATION COMPARISON")

cv_df = final_df[
    [
        "Configuration",
        "Best CV RMSE"
    ]
].copy()

cv_df = cv_df.sort_values(
    by="Best CV RMSE",
    ascending=True
)

print(cv_df.to_string(index=False))

best_cv_config = cv_df.iloc[0]["Configuration"]
best_cv_rmse = cv_df.iloc[0]["Best CV RMSE"]


# ------------------------------------------------------------
# 6. BASELINE VS TUNED COMPARISON
# ------------------------------------------------------------

print("\n5. BASELINE VS TUNED MODEL COMPARISON")

comparison_columns = [
    "Configuration",
    "Baseline Test RMSE",
    "Tuned Test RMSE",
    "Baseline Test MAE",
    "Tuned Test MAE",
    "Baseline Test R2",
    "Tuned Test R2",
    "RMSE Improvement",
    "RMSE Improvement Percentage"
]

print(
    final_df[comparison_columns].to_string(index=False)
)


# ------------------------------------------------------------
# 7. GENERALIZATION ANALYSIS
# ------------------------------------------------------------

print("\n6. GENERALIZATION ANALYSIS")

generalization_columns = [
    "Configuration",
    "Train RMSE",
    "Test RMSE",
    "RMSE Gap",
    "Train R2",
    "Test R2",
    "R2 Gap"
]

print(
    final_df[generalization_columns].to_string(index=False)
)

print("\nInterpretation:")
print("- Smaller RMSE gaps indicate a smaller train/test error difference.")
print("- A smaller gap alone does not prove better generalization.")
print("- Test performance and cross-validation must also be considered.")


# ------------------------------------------------------------
# 8. FEATURE IMPORTANCE COMPARISON
# ------------------------------------------------------------

print("\n7. FEATURE IMPORTANCE COMPARISON")

importance_without.insert(
    0,
    "Configuration",
    "without_votes"
)

importance_with.insert(
    0,
    "Configuration",
    "with_votes"
)

importance_df = pd.concat(
    [
        importance_without,
        importance_with
    ],
    ignore_index=True
)

importance_df = importance_df.sort_values(
    by=["Configuration", "Importance"],
    ascending=[True, False]
)

print(importance_df.to_string(index=False))


# ------------------------------------------------------------
# 9. FINAL MODEL SELECTION
# ------------------------------------------------------------

print("\n8. FINAL MODEL SELECTION")

# Select based on cross-validation RMSE.
# Test results are used as an independent confirmation.

selected_row = final_df[
    final_df["Configuration"] == best_cv_config
].iloc[0]

if best_cv_config == "with_votes":

    primary_model = "tuned_rating_model_with_votes.joblib"
    fallback_model = "tuned_rating_model_without_votes.joblib"

else:

    primary_model = "tuned_rating_model_without_votes.joblib"
    fallback_model = "tuned_rating_model_with_votes.joblib"

print("\nPrimary model:", primary_model)
print("Fallback model:", fallback_model)

print("\nSelection evidence:")
print(f"Best CV RMSE: {best_cv_rmse:.4f}")
print(f"Test RMSE: {selected_row['Final Test RMSE']:.4f}")
print(f"Test MAE: {selected_row['Final Test MAE']:.4f}")
print(f"Test R2: {selected_row['Final Test R2']:.4f}")


# ------------------------------------------------------------
# 10. SAVE FINAL EVALUATION REPORT
# ------------------------------------------------------------

print("\n9. SAVING FINAL REPORTS")

final_report_path = os.path.join(
    REPORT_DIR,
    "final_rating_model_evaluation.csv"
)

final_df.to_csv(
    final_report_path,
    index=False
)

importance_report_path = os.path.join(
    REPORT_DIR,
    "final_feature_importance_comparison.csv"
)

importance_df.to_csv(
    importance_report_path,
    index=False
)

cv_report_path = os.path.join(
    REPORT_DIR,
    "final_cross_validation_comparison.csv"
)

cv_df.to_csv(
    cv_report_path,
    index=False
)


# ------------------------------------------------------------
# 11. SAVE MODEL SELECTION SUMMARY
# ------------------------------------------------------------

summary_path = os.path.join(
    REPORT_DIR,
    "final_model_selection_summary.txt"
)

with open(summary_path, "w", encoding="utf-8") as file:

    file.write("DINESCOPE - FINAL MODEL SELECTION\n")
    file.write("=" * 55 + "\n\n")

    file.write(f"Primary model: {primary_model}\n")
    file.write(f"Fallback model: {fallback_model}\n\n")

    file.write(f"Selected configuration: {best_cv_config}\n")
    file.write(f"Best CV RMSE: {best_cv_rmse:.4f}\n")

    file.write(
        f"Test RMSE: {selected_row['Final Test RMSE']:.4f}\n"
    )

    file.write(
        f"Test MAE: {selected_row['Final Test MAE']:.4f}\n"
    )

    file.write(
        f"Test R2: {selected_row['Final Test R2']:.4f}\n"
    )

    file.write("\nSelection basis:\n")
    file.write(
        "Cross-validation RMSE was used for model selection.\n"
    )

    file.write(
        "The held-out test set was used for final evaluation.\n"
    )

    file.write(
        "The model with Votes requires vote information at prediction time.\n"
    )



# ------------------------------------------------------------
# 12. VISUALIZATION - BASELINE VS TUNED RMSE
# ------------------------------------------------------------

print("\n10. GENERATING COMPARISON VISUALIZATIONS")

plot_df = final_df.set_index("Configuration")

labels = ["Without Votes", "With Votes"]

baseline_rmse = np.array([
    plot_df.loc["without_votes", "Baseline Test RMSE"],
    plot_df.loc["with_votes", "Baseline Test RMSE"]
], dtype=float)

tuned_rmse = np.array([
    plot_df.loc["without_votes", "Tuned Test RMSE"],
    plot_df.loc["with_votes", "Tuned Test RMSE"]
], dtype=float)

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(9, 6))

plt.bar(
    x - width / 2,
    baseline_rmse,
    width,
    label="Baseline"
)

plt.bar(
    x + width / 2,
    tuned_rmse,
    width,
    label="Tuned"
)

plt.xticks(x, labels)
plt.ylabel("Test RMSE")
plt.title("Baseline vs Tuned Random Forest")
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "15_baseline_vs_tuned_rmse.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 13. VISUALIZATION - TEST R2
# ------------------------------------------------------------

baseline_r2 = np.array([
    plot_df.loc["without_votes", "Baseline Test R2"],
    plot_df.loc["with_votes", "Baseline Test R2"]
], dtype=float)

tuned_r2 = np.array([
    plot_df.loc["without_votes", "Tuned Test R2"],
    plot_df.loc["with_votes", "Tuned Test R2"]
], dtype=float)

plt.figure(figsize=(9, 6))

plt.bar(
    x - width / 2,
    baseline_r2,
    width,
    label="Baseline"
)

plt.bar(
    x + width / 2,
    tuned_r2,
    width,
    label="Tuned"
)

plt.xticks(x, labels)
plt.ylabel("Test R2")
plt.title("Baseline vs Tuned R2")
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "16_baseline_vs_tuned_r2.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 14. VISUALIZATION - GENERALIZATION GAP
# ------------------------------------------------------------

gap_values = np.array([
    plot_df.loc["without_votes", "RMSE Gap"],
    plot_df.loc["with_votes", "RMSE Gap"]
], dtype=float)

plt.figure(figsize=(9, 6))

plt.bar(
    labels,
    gap_values
)

plt.ylabel("RMSE Gap")
plt.title("Tuned Model Generalization Gap")

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "17_tuned_generalization_gap.png"
    ),
    dpi=300
)

plt.close()

# ------------------------------------------------------------
# 15. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("PROGRAM 08 COMPLETED SUCCESSFULLY")
print("=" * 75)

print("\nGenerated reports:")
print("- final_rating_model_evaluation.csv")
print("- final_feature_importance_comparison.csv")
print("- final_cross_validation_comparison.csv")
print("- final_model_selection_summary.txt")

print("\nGenerated visualizations:")
print("- 15_baseline_vs_tuned_rmse.png")
print("- 16_baseline_vs_tuned_r2.png")
print("- 17_tuned_generalization_gap.png")

print("\nExisting scripts and models were not modified.")
