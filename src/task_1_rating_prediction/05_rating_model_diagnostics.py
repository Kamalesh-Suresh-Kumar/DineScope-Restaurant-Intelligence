import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
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
# PROGRAM 05: RATING MODEL DIAGNOSTICS
# ============================================================

print("=" * 75)
print("DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS")
print("TASK 01: RESTAURANT RATING PREDICTION")
print("PROGRAM 05 - RATING MODEL DIAGNOSTICS")
print("=" * 75)

# ==========================================================
# 1. CONFIGURATION
# ==========================================================

TRAIN_PATH = str(PROJECT_ROOT / "data" / "processed" / "task_1_rating_prediction" / "train.csv")
TEST_PATH = str(PROJECT_ROOT / "data" / "processed" / "task_1_rating_prediction" / "test.csv")

MODEL_PATH = str(PROJECT_ROOT / "models" / "task_1")
REPORT_PATH = str(PROJECT_ROOT / "reports" / "task_1")
FIGURE_PATH = str(PROJECT_ROOT / "outputs" / "figures" / "task_1")

os.makedirs(REPORT_PATH, exist_ok=True)
os.makedirs(FIGURE_PATH, exist_ok=True)

RANDOM_STATE = 42

PERMUTATION_REPEATS = 10
TOP_FEATURES = 15

TARGET = "Aggregate rating"


# ==========================================================
# 2. LOAD DATASETS
# ==========================================================

print("\n1. LOADING DATASETS")

if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(
        f"Training dataset not found: {TRAIN_PATH}"
    )

if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(
        f"Testing dataset not found: {TEST_PATH}"
    )

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Training shape:", train_df.shape)
print("Testing shape:", test_df.shape)

if train_df.empty or test_df.empty:
    raise ValueError(
        "Training or testing dataset is empty."
    )

if TARGET not in train_df.columns:
    raise ValueError(
        f"Target column missing from training data: {TARGET}"
    )

if TARGET not in test_df.columns:
    raise ValueError(
        f"Target column missing from testing data: {TARGET}"
    )


# ==========================================================
# 3. DEFINE FEATURE CONFIGURATIONS
# ==========================================================

print("\n2. FEATURE CONFIGURATION")

base_categorical = [
    "Country Code",
    "City",
    "Cuisines",
    "Has Table booking",
    "Has Online delivery"
]

base_numerical = [
    "Longitude",
    "Latitude",
    "Price range"
]

feature_configs = {
    "without_votes": {
        "features": base_categorical + base_numerical
    },

    "with_votes": {
        "features": (
            base_categorical +
            base_numerical +
            ["Votes"]
        )
    }
}

print("Model A: Without Votes")
print("Model B: With Votes")


# ==========================================================
# 4. PREPARE TARGET VARIABLES
# ==========================================================

print("\n3. PREPARING TARGET VARIABLES")

y_train = pd.to_numeric(
    train_df[TARGET],
    errors="coerce"
)

y_test = pd.to_numeric(
    test_df[TARGET],
    errors="coerce"
)

if y_train.isnull().any() or y_test.isnull().any():
    raise ValueError(
        "Missing or invalid target values found."
    )

print("Training target records:", len(y_train))
print("Testing target records:", len(y_test))


# ==========================================================
# 5. DEFINE EVALUATION FUNCTION
# ==========================================================

def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    mse = mean_squared_error(
        actual,
        predicted
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        actual,
        predicted
    )

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }


# ==========================================================
# 6. DEFINE PLOTTING FUNCTIONS
# ==========================================================

def plot_actual_vs_predicted(
    actual,
    predicted,
    config_name
):

    plt.figure(figsize=(8, 6))

    plt.scatter(
        actual,
        predicted,
        alpha=0.5
    )

    minimum = min(
        actual.min(),
        predicted.min()
    )

    maximum = max(
        actual.max(),
        predicted.max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")

    plt.title(
        f"Actual vs Predicted Ratings - {config_name}"
    )

    plt.tight_layout()

    path = os.path.join(
        FIGURE_PATH,
        f"11_actual_vs_predicted_{config_name}.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", path)


def plot_residuals(
    predicted,
    residuals,
    config_name
):

    plt.figure(figsize=(8, 6))

    plt.scatter(
        predicted,
        residuals,
        alpha=0.5
    )

    plt.axhline(
        y=0,
        linestyle="--"
    )

    plt.xlabel("Predicted Rating")
    plt.ylabel("Residual (Actual - Predicted)")

    plt.title(
        f"Residuals vs Predicted - {config_name}"
    )

    plt.tight_layout()

    path = os.path.join(
        FIGURE_PATH,
        f"12_residuals_vs_predicted_{config_name}.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", path)


def plot_residual_distribution(
    residuals,
    config_name
):

    plt.figure(figsize=(8, 6))

    plt.hist(
        residuals,
        bins=30,
        edgecolor="black"
    )

    plt.axvline(
        x=0,
        linestyle="--"
    )

    plt.xlabel("Residual")
    plt.ylabel("Frequency")

    plt.title(
        f"Residual Distribution - {config_name}"
    )

    plt.tight_layout()

    path = os.path.join(
        FIGURE_PATH,
        f"13_residual_distribution_{config_name}.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", path)


# ==========================================================
# 7. LOAD MODELS AND GENERATE PREDICTIONS
# ==========================================================

print("\n4. LOADING TRAINED MODELS")

diagnostic_results = []
prediction_reports = []

for config_name, config in feature_configs.items():

    print("\n" + "=" * 70)
    print("DIAGNOSTICS:", config_name.upper())
    print("=" * 70)

    model_file = os.path.join(
        MODEL_PATH,
        f"best_rating_model_{config_name}.joblib"
    )

    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"Trained model not found: {model_file}. "
            "Run Program 04 first."
        )

    model = joblib.load(model_file)

    print("Loaded model:", model_file)

    selected_features = config["features"]

    missing_features = [
        feature for feature in selected_features
        if feature not in train_df.columns
        or feature not in test_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    X_train = train_df[selected_features]
    X_test = test_df[selected_features]

    # Verify that the saved pipeline has the expected
    # fitted preprocessing and regression components.

    if not hasattr(model, "predict"):
        raise TypeError(
            "Loaded model does not support prediction."
        )

    # Generate predictions.

    train_predictions = model.predict(
        X_train
    )

    test_predictions = model.predict(
        X_test
    )

    print("\nPredictions generated successfully.")


    # ======================================================
    # 8. TRAINING AND TESTING EVALUATION
    # ======================================================

    print("\n5. TRAINING AND TESTING PERFORMANCE")

    train_metrics = calculate_metrics(
        y_train,
        train_predictions
    )

    test_metrics = calculate_metrics(
        y_test,
        test_predictions
    )

    print("\nTraining performance:")

    for metric, value in train_metrics.items():
        print(f"Train {metric}: {value:.4f}")

    print("\nTesting performance:")

    for metric, value in test_metrics.items():
        print(f"Test {metric}: {value:.4f}")

    rmse_gap = (
        test_metrics["RMSE"] -
        train_metrics["RMSE"]
    )

    r2_gap = (
        train_metrics["R2"] -
        test_metrics["R2"]
    )

    print("\nGeneralization gaps:")

    print(f"RMSE gap: {rmse_gap:.4f}")
    print(f"R2 gap: {r2_gap:.4f}")

    diagnostic_results.append({
        "Configuration": config_name,

        "Train MAE": train_metrics["MAE"],
        "Train MSE": train_metrics["MSE"],
        "Train RMSE": train_metrics["RMSE"],
        "Train R2": train_metrics["R2"],

        "Test MAE": test_metrics["MAE"],
        "Test MSE": test_metrics["MSE"],
        "Test RMSE": test_metrics["RMSE"],
        "Test R2": test_metrics["R2"],

        "RMSE Gap": rmse_gap,
        "R2 Gap": r2_gap
    })


    # ======================================================
    # 9. PREDICTION ERROR ANALYSIS
    # ======================================================

    print("\n6. PREDICTION ERROR ANALYSIS")

    residuals = (
        y_test.to_numpy() -
        test_predictions
    )

    absolute_errors = np.abs(residuals)

    print(
        "Mean residual:",
        round(float(np.mean(residuals)), 4)
    )

    print(
        "Mean absolute error:",
        round(float(np.mean(absolute_errors)), 4)
    )

    print(
        "Maximum absolute error:",
        round(float(np.max(absolute_errors)), 4)
    )

    print(
        "Minimum absolute error:",
        round(float(np.min(absolute_errors)), 4)
    )

    prediction_df = pd.DataFrame({
        "Actual Rating": y_test.to_numpy(),

        "Predicted Rating": test_predictions,

        "Residual": residuals,

        "Absolute Error": absolute_errors
    })

    prediction_path = os.path.join(
        REPORT_PATH,
        f"rating_diagnostics_{config_name}.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False
    )

    print("Prediction diagnostics saved:", prediction_path)

    prediction_reports.append(prediction_df)


    # ======================================================
    # 10. ACTUAL VS PREDICTED PLOT
    # ======================================================

    print("\n7. ACTUAL VS PREDICTED ANALYSIS")

    plot_actual_vs_predicted(
        y_test.to_numpy(),
        test_predictions,
        config_name
    )


    # ======================================================
    # 11. RESIDUAL ANALYSIS
    # ======================================================

    print("\n8. RESIDUAL ANALYSIS")

    plot_residuals(
        test_predictions,
        residuals,
        config_name
    )

    plot_residual_distribution(
        residuals,
        config_name
    )


    # ======================================================
    # 12. RANDOM FOREST FEATURE IMPORTANCE
    # ======================================================

    print("\n9. RANDOM FOREST FEATURE IMPORTANCE")

    regressor = model.named_steps["regressor"]

    if hasattr(regressor, "feature_importances_"):

        preprocessor = model.named_steps["preprocessor"]

        transformed_features = (
            preprocessor.get_feature_names_out()
        )

        importances = regressor.feature_importances_

        if len(transformed_features) != len(importances):
            raise ValueError(
                "Feature names and importance values do not match."
            )

        importance_df = pd.DataFrame({
            "Feature": transformed_features,
            "Importance": importances
        })

        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False
        )

        print("\nTop 15 transformed features:")

        print(
            importance_df.head(TOP_FEATURES).to_string(
                index=False
            )
        )

        importance_path = os.path.join(
            REPORT_PATH,
            f"random_forest_feature_importance_{config_name}.csv"
        )

        importance_df.to_csv(
            importance_path,
            index=False
        )

        # Plot top features.

        top_importance = (
            importance_df
            .head(TOP_FEATURES)
            .sort_values(
                by="Importance",
                ascending=True
            )
        )

        plt.figure(figsize=(10, 7))

        plt.barh(
            top_importance["Feature"],
            top_importance["Importance"]
        )

        plt.xlabel("Feature Importance")
        plt.ylabel("Transformed Feature")

        plt.title(
            f"Random Forest Feature Importance - {config_name}"
        )

        plt.tight_layout()

        importance_figure = os.path.join(
            FIGURE_PATH,
            f"14_random_forest_feature_importance_{config_name}.png"
        )

        plt.savefig(
            importance_figure,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print("Importance report saved:", importance_path)
        print("Importance plot saved:", importance_figure)

    else:

        print(
            "The selected estimator does not provide "
            "built-in feature_importances_."
        )


    # ======================================================
    # 13. PERMUTATION FEATURE IMPORTANCE
    # ======================================================

    print("\n10. PERMUTATION FEATURE IMPORTANCE")

    print(
        "Evaluating the effect of shuffling each original feature..."
    )

    permutation = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="neg_root_mean_squared_error",
        n_repeats=PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=1
    )

    # Extract permutation importance values.
    # getattr avoids the current Pylance attribute warnings.

    importance_mean = getattr(
        permutation,
        "importances_mean"
    )

    importance_std = getattr(
        permutation,
        "importances_std"
    )

    permutation_df = pd.DataFrame({
        "Feature": selected_features,

        "Importance Mean": importance_mean,

        "Importance Std": importance_std
    })

    permutation_df = permutation_df.sort_values(
        by="Importance Mean",
        ascending=False
    )

    print("\nPermutation importance:")

    print(
        permutation_df.to_string(index=False)
    )

    permutation_path = os.path.join(
        REPORT_PATH,
        f"permutation_importance_{config_name}.csv"
    )

    permutation_df.to_csv(
        permutation_path,
        index=False
    )

    print(
        "Permutation importance report saved:",
        permutation_path
    )

    # Plot permutation importance.

    top_permutation = (
        permutation_df
        .head(TOP_FEATURES)
        .sort_values(
            by="Importance Mean",
            ascending=True
        )
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        top_permutation["Feature"],
        top_permutation["Importance Mean"],
        xerr=top_permutation["Importance Std"]
    )

    plt.axvline(
        x=0,
        linestyle="--"
    )

    plt.xlabel("Increase in RMSE after Shuffling")
    plt.ylabel("Original Feature")

    plt.title(
        f"Permutation Importance - {config_name}"
    )

    plt.tight_layout()

    permutation_figure = os.path.join(
        FIGURE_PATH,
        f"15_permutation_importance_{config_name}.png"
    )

    plt.savefig(
        permutation_figure,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Permutation importance plot saved:",
        permutation_figure
    )


# ==========================================================
# 14. SAVE FINAL DIAGNOSTICS REPORT
# ==========================================================

print("\n" + "=" * 70)
print("11. FINAL DIAGNOSTICS COMPARISON")
print("=" * 70)

diagnostics_df = pd.DataFrame(
    diagnostic_results
)

print(
    diagnostics_df.to_string(index=False)
)

diagnostics_path = os.path.join(
    REPORT_PATH,
    "rating_model_diagnostics_summary.csv"
)

diagnostics_df.to_csv(
    diagnostics_path,
    index=False
)

print("\nDiagnostics summary saved:", diagnostics_path)


# ==========================================================
# 15. FINAL INTERPRETATION
# ==========================================================

print("\n" + "=" * 70)
print("12. FINAL MODEL INTERPRETATION")
print("=" * 70)

for _, row in diagnostics_df.iterrows():

    print("\nConfiguration:", row["Configuration"])

    print("Training RMSE:", round(row["Train RMSE"], 4))
    print("Testing RMSE:", round(row["Test RMSE"], 4))

    print("Training R2:", round(row["Train R2"], 4))
    print("Testing R2:", round(row["Test R2"], 4))

    if row["RMSE Gap"] > 0.10:
        print(
            "Observation: A noticeable training-to-test "
            "RMSE gap exists. Investigate possible overfitting."
        )

    elif row["RMSE Gap"] > 0:
        print(
            "Observation: Test error is higher than training "
            "error, but the RMSE gap is relatively smaller."
        )

    else:
        print(
            "Observation: Test RMSE is not higher than "
            "training RMSE."
        )

    if row["Test R2"] >= 0.60:
        print(
            "Test R2 indicates a reasonably strong fit "
            "for this dataset."
        )

    elif row["Test R2"] >= 0.40:
        print(
            "Test R2 indicates moderate explanatory "
            "performance."
        )

    else:
        print(
            "Test R2 indicates limited explanatory "
            "performance."
        )


# ==========================================================
# 16. FINAL STATUS
# ==========================================================

print("\n" + "=" * 70)
print("MODEL DIAGNOSTICS COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated reports:")

print("- rating_diagnostics_without_votes.csv")
print("- rating_diagnostics_with_votes.csv")
print("- random_forest_feature_importance_without_votes.csv")
print("- random_forest_feature_importance_with_votes.csv")
print("- permutation_importance_without_votes.csv")
print("- permutation_importance_with_votes.csv")
print("- rating_model_diagnostics_summary.csv")

print("\nGenerated visualizations:")

print("- 11_actual_vs_predicted_*.png")
print("- 12_residuals_vs_predicted_*.png")
print("- 13_residual_distribution_*.png")
print("- 14_random_forest_feature_importance_*.png")
print("- 15_permutation_importance_*.png")

print("\nAll model diagnostics completed.")
