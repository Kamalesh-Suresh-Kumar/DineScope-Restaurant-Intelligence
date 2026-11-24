import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.inspection import permutation_importance


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ============================================================
# DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS
# TASK 01: RESTAURANT RATING PREDICTION
# PROGRAM 07: TUNED MODEL DIAGNOSTICS
# ============================================================

print("=" * 75)
print("DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS")
print("TASK 01: RESTAURANT RATING PREDICTION")
print("PROGRAM 07 - TUNED MODEL DIAGNOSTICS")
print("=" * 75)

# ------------------------------------------------------------
# 1. PATH CONFIGURATION
# ------------------------------------------------------------

TRAIN_PATH = str(PROJECT_ROOT / "data" / "processed" / "task_1_rating_prediction" / "train.csv")
TEST_PATH = str(PROJECT_ROOT / "data" / "processed" / "task_1_rating_prediction" / "test.csv")

MODEL_WITHOUT_VOTES = str(PROJECT_ROOT / "models" / "task_1" / "tuned_rating_model_without_votes.joblib")
MODEL_WITH_VOTES = str(PROJECT_ROOT / "models" / "task_1" / "tuned_rating_model_with_votes.joblib")

OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / "figures" / "task_1")
REPORT_DIR = str(PROJECT_ROOT / "reports" / "task_1")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD DATASETS
# ------------------------------------------------------------

print("\n1. LOADING DATASETS")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Training shape:", train_df.shape)
print("Testing shape:", test_df.shape)

TARGET = "Aggregate rating"

if TARGET not in train_df.columns or TARGET not in test_df.columns:
    raise ValueError("Target column not found in datasets.")


# ------------------------------------------------------------
# 3. LOAD TUNED MODELS
# ------------------------------------------------------------

print("\n2. LOADING TUNED MODELS")

models = {
    "without_votes": joblib.load(MODEL_WITHOUT_VOTES),
    "with_votes": joblib.load(MODEL_WITH_VOTES)
}

print("Both tuned models loaded successfully.")


# ------------------------------------------------------------
# 4. DIAGNOSTIC FUNCTION
# ------------------------------------------------------------

def analyse_model(model_name, model):

    print("\n" + "=" * 70)
    print("ANALYSING:", model_name.upper())
    print("=" * 70)

    # Obtain the original input features expected by the model.
    if not hasattr(model, "feature_names_in_"):
        raise ValueError(
            f"{model_name}: Model does not contain feature names. "
            "Check how the model was originally trained."
        )

    features = list(model.feature_names_in_)

    X_train = train_df[features].copy()
    X_test = test_df[features].copy()

    y_train = train_df[TARGET]
    y_test = test_df[TARGET]

    print("\nFeatures used:")
    print(features)

    # --------------------------------------------------------
    # TRAIN AND TEST PREDICTIONS
    # --------------------------------------------------------

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # --------------------------------------------------------
    # PERFORMANCE METRICS
    # --------------------------------------------------------

    train_mae = mean_absolute_error(y_train, train_pred)
    train_mse = mean_squared_error(y_train, train_pred)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train, train_pred)

    test_mae = mean_absolute_error(y_test, test_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, test_pred)

    rmse_gap = test_rmse - train_rmse
    r2_gap = train_r2 - test_r2

    print("\nTRAINING PERFORMANCE")
    print(f"MAE:  {train_mae:.4f}")
    print(f"MSE:  {train_mse:.4f}")
    print(f"RMSE: {train_rmse:.4f}")
    print(f"R2:   {train_r2:.4f}")

    print("\nTESTING PERFORMANCE")
    print(f"MAE:  {test_mae:.4f}")
    print(f"MSE:  {test_mse:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"R2:   {test_r2:.4f}")

    print("\nGENERALIZATION GAP")
    print(f"RMSE Gap: {rmse_gap:.4f}")
    print(f"R2 Gap:   {r2_gap:.4f}")

    # --------------------------------------------------------
    # RESIDUAL ANALYSIS
    # --------------------------------------------------------

    residuals = y_test.to_numpy() - test_pred

    print("\nRESIDUAL ANALYSIS")
    print(f"Mean Residual: {np.mean(residuals):.4f}")
    print(f"Residual Std:  {np.std(residuals):.4f}")
    print(f"Maximum Error: {np.max(np.abs(residuals)):.4f}")

    # --------------------------------------------------------
    # SAVE PREDICTION RESULTS
    # --------------------------------------------------------

    prediction_df = pd.DataFrame({
        "Actual Rating": y_test.to_numpy(),
        "Predicted Rating": test_pred,
        "Residual": residuals,
        "Absolute Error": np.abs(residuals)
    })

    prediction_path = os.path.join(
        REPORT_DIR,
        f"tuned_predictions_{model_name}.csv"
    )

    prediction_df.to_csv(prediction_path, index=False)

    # --------------------------------------------------------
    # PLOT 1: ACTUAL VS PREDICTED
    # --------------------------------------------------------

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_test,
        test_pred,
        alpha=0.5
    )

    plt.plot(
        [1, 5],
        [1, 5],
        linestyle="--"
    )

    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")
    plt.title(f"Actual vs Predicted - {model_name}")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"11_tuned_actual_vs_predicted_{model_name}.png"
        ),
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # PLOT 2: RESIDUAL DISTRIBUTION
    # --------------------------------------------------------

    plt.figure(figsize=(8, 6))

    plt.hist(
        residuals,
        bins=30,
        edgecolor="black"
    )

    plt.axvline(
        0,
        linestyle="--"
    )

    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title(f"Residual Distribution - {model_name}")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"12_tuned_residual_distribution_{model_name}.png"
        ),
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # PLOT 3: RESIDUALS VS PREDICTED
    # --------------------------------------------------------

    plt.figure(figsize=(8, 6))

    plt.scatter(
        test_pred,
        residuals,
        alpha=0.5
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.xlabel("Predicted Rating")
    plt.ylabel("Residual")
    plt.title(f"Residuals vs Predicted - {model_name}")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"13_tuned_residuals_vs_predicted_{model_name}.png"
        ),
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # PERMUTATION FEATURE IMPORTANCE
    # --------------------------------------------------------

    print("\nCALCULATING PERMUTATION IMPORTANCE...")

    importance = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=5,
        random_state=42,
        scoring="neg_root_mean_squared_error",
        n_jobs=1
    )

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance["importances_mean"],
        "Importance Std": importance["importances_std"]
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nPERMUTATION FEATURE IMPORTANCE")
    print(importance_df.to_string(index=False))

    importance_path = os.path.join(
        REPORT_DIR,
        f"tuned_feature_importance_{model_name}.csv"
    )

    importance_df.to_csv(
        importance_path,
        index=False
    )

    # --------------------------------------------------------
    # PLOT 4: FEATURE IMPORTANCE
    # --------------------------------------------------------

    plot_df = importance_df.head(15).sort_values(
        by="Importance"
    )

    plt.figure(figsize=(9, 6))

    plt.barh(
        plot_df["Feature"],
        plot_df["Importance"],
        xerr=plot_df["Importance Std"]
    )

    plt.xlabel("Decrease in Model Performance")
    plt.ylabel("Feature")
    plt.title(f"Permutation Importance - {model_name}")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"14_tuned_feature_importance_{model_name}.png"
        ),
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # RETURN METRICS
    # --------------------------------------------------------

    return {
        "Configuration": model_name,
        "Train MAE": train_mae,
        "Train RMSE": train_rmse,
        "Train R2": train_r2,
        "Test MAE": test_mae,
        "Test MSE": test_mse,
        "Test RMSE": test_rmse,
        "Test R2": test_r2,
        "RMSE Gap": rmse_gap,
        "R2 Gap": r2_gap,
        "Mean Residual": np.mean(residuals),
        "Maximum Absolute Error": np.max(np.abs(residuals))
    }


# ------------------------------------------------------------
# 5. ANALYSE BOTH MODELS
# ------------------------------------------------------------

print("\n3. RUNNING MODEL DIAGNOSTICS")

results = []

for model_name, model in models.items():

    result = analyse_model(
        model_name,
        model
    )

    results.append(result)


# ------------------------------------------------------------
# 6. SAVE COMPARISON REPORT
# ------------------------------------------------------------

print("\n4. FINAL DIAGNOSTIC COMPARISON")

results_df = pd.DataFrame(results)

print(results_df.to_string(index=False))

results_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "tuned_model_diagnostics_comparison.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# 7. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TUNED MODEL DIAGNOSTICS COMPLETED")
print("=" * 70)

print("\nGenerated reports:")
print("- tuned_predictions_without_votes.csv")
print("- tuned_predictions_with_votes.csv")
print("- tuned_feature_importance_without_votes.csv")
print("- tuned_feature_importance_with_votes.csv")
print("- tuned_model_diagnostics_comparison.csv")

print("\nGenerated visualizations:")
print("- 11_tuned_actual_vs_predicted_*.png")
print("- 12_tuned_residual_distribution_*.png")
print("- 13_tuned_residuals_vs_predicted_*.png")
print("- 14_tuned_feature_importance_*.png")

print("\nOriginal scripts and models preserved.")
