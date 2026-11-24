import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ============================================================
# DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS
# TASK 01: RESTAURANT RATING PREDICTION
# PROGRAM 09: RATING PREDICTION PIPELINE TESTING
# ============================================================

print("=" * 75)
print("DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS")
print("TASK 01: RESTAURANT RATING PREDICTION")
print("PROGRAM 09 - RATING PREDICTION PIPELINE TESTING")
print("=" * 75)

# ------------------------------------------------------------
# 1. PATH CONFIGURATION
# ------------------------------------------------------------

TEST_PATH = str(PROJECT_ROOT / "data" / "processed" / "task_1_rating_prediction" / "test.csv")

MODEL_WITHOUT_VOTES = (
    str(PROJECT_ROOT / "models" / "task_1" / "tuned_rating_model_without_votes.joblib")
)

MODEL_WITH_VOTES = (
    str(PROJECT_ROOT / "models" / "task_1" / "tuned_rating_model_with_votes.joblib")
)

REPORT_DIR = str(PROJECT_ROOT / "reports" / "task_1")

os.makedirs(REPORT_DIR, exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD TEST DATASET
# ------------------------------------------------------------

print("\n1. LOADING TEST DATASET")

if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(TEST_PATH)

test_df = pd.read_csv(TEST_PATH)

TARGET = "Aggregate rating"

if TARGET not in test_df.columns:
    raise ValueError("Target column is missing.")

print("Test dataset shape:", test_df.shape)


# ------------------------------------------------------------
# 3. LOAD TRAINED MODELS
# ------------------------------------------------------------

print("\n2. LOADING TRAINED MODELS")

model_paths = {
    "without_votes": MODEL_WITHOUT_VOTES,
    "with_votes": MODEL_WITH_VOTES
}

models = {}

for name, path in model_paths.items():

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    models[name] = joblib.load(path)

    if not hasattr(models[name], "feature_names_in_"):
        raise ValueError(
            f"{name}: Model feature names are unavailable."
        )

    print(f"{name}: Loaded successfully.")


# ------------------------------------------------------------
# 4. VALIDATE MODEL INPUT FEATURES
# ------------------------------------------------------------

print("\n3. VALIDATING MODEL INPUT FEATURES")

for name, model in models.items():

    features = list(model.feature_names_in_)

    missing_features = [
        col for col in features
        if col not in test_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"{name}: Missing features: {missing_features}"
        )

    print(f"\n{name.upper()}")

    print("Expected feature count:", len(features))
    print("Expected features:", features)


# ------------------------------------------------------------
# 5. RUN PREDICTION TESTS
# ------------------------------------------------------------

print("\n4. RUNNING PREDICTION TESTS")

all_predictions = []
summary_results = []

for name, model in models.items():

    print("\n" + "=" * 70)
    print("TESTING:", name.upper())
    print("=" * 70)

    features = list(model.feature_names_in_)

    X_test = test_df[features].copy()
    y_test = test_df[TARGET].copy()

    # --------------------------------------------------------
    # BATCH PREDICTION
    # --------------------------------------------------------

    print("\nA. BATCH PREDICTION")

    predictions = np.asarray(
        model.predict(X_test),
        dtype=float
    )

    print("Input records:", len(X_test))
    print("Prediction records:", len(predictions))

    if len(predictions) != len(X_test):
        raise ValueError("Prediction count does not match input count.")

    # --------------------------------------------------------
    # VALIDATE PREDICTIONS
    # --------------------------------------------------------

    print("\nB. PREDICTION VALIDATION")

    if not np.all(np.isfinite(predictions)):
        raise ValueError("Non-finite prediction values detected.")

    print("All predictions are finite.")

    if np.any(predictions < 1) or np.any(predictions > 5):
        raise ValueError("Predictions outside the valid 1-5 range.")

    print("All predictions are within the 1-5 rating range.")

    # --------------------------------------------------------
    # SINGLE-RECORD PREDICTION
    # --------------------------------------------------------

    print("\nC. SINGLE-RECORD PREDICTION")

    single_input = X_test.iloc[[0]]

    single_prediction = np.asarray(
        model.predict(single_input),
        dtype=float
    )

    print("Actual rating:", y_test.iloc[0])
    print("Predicted rating:", round(single_prediction[0], 4))

    # Verify that single prediction matches batch prediction.
    if not np.isclose(
        single_prediction[0],
        predictions[0]
    ):
        raise ValueError(
            "Single prediction does not match batch prediction."
        )

    print("Single and batch predictions match.")

    # --------------------------------------------------------
    # EVALUATION METRICS
    # --------------------------------------------------------

    print("\nD. PIPELINE PERFORMANCE CHECK")

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    prediction_df = pd.DataFrame({
        "Configuration": name,
        "Actual Rating": y_test.to_numpy(),
        "Predicted Rating": predictions,
        "Absolute Error": np.abs(
            y_test.to_numpy() - predictions
        )
    })

    # Include restaurant identifiers if available.
    for col in ["Restaurant ID", "Restaurant Name"]:
        if col in test_df.columns:
            prediction_df.insert(
                len(prediction_df.columns) - 3,
                col,
                test_df[col].to_numpy()
            )

    all_predictions.append(prediction_df)

    summary_results.append({
        "Configuration": name,
        "Input Records": len(X_test),
        "Prediction Records": len(predictions),
        "Single Prediction Test": "Passed",
        "Batch Prediction Test": "Passed",
        "Range Validation": "Passed",
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })


# ------------------------------------------------------------
# 6. SAVE COMBINED PREDICTIONS
# ------------------------------------------------------------

print("\n5. SAVING PREDICTION RESULTS")

combined_predictions = pd.concat(
    all_predictions,
    ignore_index=True
)

prediction_path = os.path.join(
    REPORT_DIR,
    "rating_pipeline_test_predictions.csv"
)

combined_predictions.to_csv(
    prediction_path,
    index=False
)

print("Prediction results saved:", prediction_path)


# ------------------------------------------------------------
# 7. SAVE PIPELINE VALIDATION SUMMARY
# ------------------------------------------------------------

summary_df = pd.DataFrame(summary_results)

summary_path = os.path.join(
    REPORT_DIR,
    "rating_pipeline_validation_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)

print("\n6. PIPELINE VALIDATION SUMMARY")

print(summary_df.to_string(index=False))


# ------------------------------------------------------------
# 8. DISPLAY SAMPLE PREDICTIONS
# ------------------------------------------------------------

print("\n7. SAMPLE PREDICTIONS")

sample_df = combined_predictions.head(10)

print(sample_df.to_string(index=False))


# ------------------------------------------------------------
# 9. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("RATING PREDICTION PIPELINE TESTING COMPLETED")
print("=" * 70)

print("\nGenerated reports:")
print("- rating_pipeline_test_predictions.csv")
print("- rating_pipeline_validation_summary.csv")

print("\nBoth trained models were tested.")
print("No model was retrained or overwritten.")
