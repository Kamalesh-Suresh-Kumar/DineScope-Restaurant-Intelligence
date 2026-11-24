import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.model_selection import KFold, cross_validate

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ============================================================
# DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS
# TASK 01: RESTAURANT RATING PREDICTION
# PROGRAM 04: RESTAURANT RATING PREDICTION
# ============================================================

print("=" * 75)
print("DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS")
print("TASK 01: RESTAURANT RATING PREDICTION")
print("PROGRAM 04 - RESTAURANT RATING PREDICTION")
print("=" * 75)

# ==========================================================
# 1. CONFIGURATION
# ==========================================================

TRAIN_PATH = str(PROJECT_ROOT / "data" / "processed" / "task_1_rating_prediction" / "train.csv")
TEST_PATH = str(PROJECT_ROOT / "data" / "processed" / "task_1_rating_prediction" /"test.csv")

MODEL_PATH = str(PROJECT_ROOT / "models" / "task_1")
REPORT_PATH = str(PROJECT_ROOT / "reports" / "task_1")

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(REPORT_PATH, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5

TARGET = "Aggregate rating"


# ==========================================================
# 2. LOAD DATA
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
        "categorical": base_categorical,
        "numerical": base_numerical
    },

    "with_votes": {
        "categorical": base_categorical,
        "numerical": base_numerical + ["Votes"]
    }
}

print("\nConfiguration A: Without Votes")
print(
    base_categorical + base_numerical
)

print("\nConfiguration B: With Votes")
print(
    base_categorical + base_numerical + ["Votes"]
)


# ==========================================================
# 4. VALIDATE INPUT FEATURES
# ==========================================================

print("\n3. INPUT DATA VALIDATION")

all_features = (
    base_categorical +
    base_numerical +
    ["Votes"]
)

required_columns = all_features + [TARGET]

for col in required_columns:

    if col not in train_df.columns:
        raise ValueError(
            f"Missing training column: {col}"
        )

    if col not in test_df.columns:
        raise ValueError(
            f"Missing testing column: {col}"
        )

# Convert target values safely.

train_df[TARGET] = pd.to_numeric(
    train_df[TARGET],
    errors="coerce"
)

test_df[TARGET] = pd.to_numeric(
    test_df[TARGET],
    errors="coerce"
)

if train_df[TARGET].isnull().any():
    raise ValueError(
        "Missing or invalid target values in training data."
    )

if test_df[TARGET].isnull().any():
    raise ValueError(
        "Missing or invalid target values in testing data."
    )

# Check target boundaries.

if not train_df[TARGET].between(1, 5).all():
    raise ValueError(
        "Training target ratings must be between 1 and 5."
    )

if not test_df[TARGET].between(1, 5).all():
    raise ValueError(
        "Testing target ratings must be between 1 and 5."
    )

# Validate numerical feature values.
# Missing numerical features are allowed because the
# pipeline handles them through SimpleImputer.

for col in base_numerical + ["Votes"]:

    train_df[col] = pd.to_numeric(
        train_df[col],
        errors="coerce"
    )

    test_df[col] = pd.to_numeric(
        test_df[col],
        errors="coerce"
    )

    if np.isinf(train_df[col].dropna()).any():
        raise ValueError(
            f"Infinite values found in training feature: {col}"
        )

    if np.isinf(test_df[col].dropna()).any():
        raise ValueError(
            f"Infinite values found in testing feature: {col}"
        )

print("All required columns are present.")
print("Target values are valid.")
print("Numerical features validated.")

print("\nMissing values in training data:")
print(train_df[required_columns].isnull().sum())

print("\nMissing values in testing data:")
print(test_df[required_columns].isnull().sum())


# ==========================================================
# 5. PREPARE TARGET VARIABLES
# ==========================================================

print("\n4. PREPARING TARGET VARIABLES")

y_train = train_df[TARGET]
y_test = test_df[TARGET]

print("Training target records:", len(y_train))
print("Testing target records:", len(y_test))


# ==========================================================
# 6. DEFINE REGRESSION MODELS
# ==========================================================

print("\n5. DEFINING REGRESSION MODELS")

models = {
    "Dummy Regressor": DummyRegressor(
        strategy="mean"
    ),

    "Linear Regression": LinearRegression(),

    "Ridge Regression": Ridge(
        alpha=10.0
    ),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=20,
        min_samples_leaf=5,
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=150,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}

print("Total algorithms:", len(models))

for model_name in models:
    print("-", model_name)


# ==========================================================
# 7. PREPROCESSING PIPELINE
# ==========================================================

print("\n6. CREATING PREPROCESSING PIPELINE")


def create_preprocessor(
    categorical_columns,
    numerical_columns
):

    # Numerical preprocessing.

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # Categorical preprocessing.

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Unknown"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=5
                )
            )
        ]
    )

    # Combine both preprocessing pipelines.

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_columns
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        ],
        remainder="drop"
    )

    return preprocessor


print("Numerical pipeline: Imputation + Scaling")
print("Categorical pipeline: Imputation + One-Hot Encoding")


# ==========================================================
# 8. CROSS-VALIDATION CONFIGURATION
# ==========================================================

print("\n7. CROSS-VALIDATION CONFIGURATION")

cv = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

scoring = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2"
}

cv_results = []

best_candidates = {}

print("Cross-validation folds:", N_SPLITS)
print("Random state:", RANDOM_STATE)


# ==========================================================
# 9. TRAIN AND COMPARE MODELS
# ==========================================================

print("\n8. MODEL TRAINING AND CROSS-VALIDATION")

for config_name, config in feature_configs.items():

    print("\n" + "=" * 70)
    print("EXPERIMENT:", config_name.upper())
    print("=" * 70)

    selected_features = (
        config["categorical"] +
        config["numerical"]
    )

    X_train = train_df[selected_features]

    best_rmse = float("inf")
    best_model_name = None
    best_pipeline = None
    best_rmse_std = float("inf")

    for model_name, model in models.items():

        print("\n" + "-" * 55)
        print("Training:", model_name)

        preprocessor = create_preprocessor(
            config["categorical"],
            config["numerical"]
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", clone(model))
            ]
        )

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
            error_score="raise"
        )

        # Convert negative scoring values into
        # positive error values.

        mean_mae = -scores["test_mae"].mean()

        mean_rmse = -scores["test_rmse"].mean()

        mean_r2 = scores["test_r2"].mean()

        std_rmse = scores["test_rmse"].std()

        print(f"CV MAE:       {mean_mae:.4f}")
        print(f"CV RMSE:      {mean_rmse:.4f}")
        print(f"CV RMSE STD:  {std_rmse:.4f}")
        print(f"CV R2:        {mean_r2:.4f}")

        cv_results.append({
            "Configuration": config_name,
            "Model": model_name,
            "CV_MAE": mean_mae,
            "CV_RMSE": mean_rmse,
            "CV_RMSE_Std": std_rmse,
            "CV_R2": mean_r2
        })

        # Select the model with the lowest CV RMSE.

        if mean_rmse < best_rmse:

            best_rmse = mean_rmse
            best_model_name = model_name
            best_pipeline = pipeline
            best_rmse_std = std_rmse

    # Store the best model for this configuration.

    best_candidates[config_name] = {
        "model_name": best_model_name,
        "pipeline": best_pipeline,
        "cv_rmse": best_rmse,
        "cv_rmse_std": best_rmse_std
    }

    print("\nBest model for", config_name)
    print("Algorithm:", best_model_name)
    print("CV RMSE:", round(best_rmse, 4))
    print("CV RMSE STD:", round(best_rmse_std, 4))


# ==========================================================
# 10. SAVE CROSS-VALIDATION RESULTS
# ==========================================================

print("\n" + "=" * 70)
print("9. CROSS-VALIDATION RESULTS")
print("=" * 70)

results_df = pd.DataFrame(cv_results)

results_df = results_df.sort_values(
    by="CV_RMSE",
    ascending=True
)

print(
    results_df.to_string(index=False)
)

cv_report_path = os.path.join(
    REPORT_PATH,
    "rating_model_comparison.csv"
)

results_df.to_csv(
    cv_report_path,
    index=False
)

print("\nCross-validation report saved:", cv_report_path)


# ==========================================================
# 11. FINAL MODEL TRAINING AND TEST EVALUATION
# ==========================================================

print("\n" + "=" * 70)
print("10. FINAL MODEL EVALUATION")
print("=" * 70)

final_results = []

for config_name, candidate in best_candidates.items():

    print("\n" + "=" * 70)
    print("FINAL EVALUATION:", config_name.upper())
    print("=" * 70)

    config = feature_configs[config_name]

    selected_features = (
        config["categorical"] +
        config["numerical"]
    )

    X_train = train_df[selected_features]

    X_test = test_df[selected_features]

    # Use a fresh pipeline for final training.

    final_pipeline = clone(
        candidate["pipeline"]
    )

    
    # Train using the complete training dataset.

    final_pipeline.fit(
        X_train,
        y_train
    )

    # Predict ratings on training data.

    train_predictions = final_pipeline.predict(
        X_train
    )

    # Predict ratings on unseen test data.

    predictions = final_pipeline.predict(
        X_test
    )

    # Evaluate training performance.

    train_mae = mean_absolute_error(
        y_train,
        train_predictions
    )

    train_mse = mean_squared_error(
        y_train,
        train_predictions
    )

    train_rmse = np.sqrt(train_mse)

    train_r2 = r2_score(
        y_train,
        train_predictions
    )

    print("\nTRAINING PERFORMANCE")

    print(f"Train MAE:  {train_mae:.4f}")
    print(f"Train MSE:  {train_mse:.4f}")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Train R2:   {train_r2:.4f}")


    # Calculate evaluation metrics.

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions
    )

    print("\nSelected model:", candidate["model_name"])

    print(f"CV RMSE:  {candidate['cv_rmse']:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test MSE: {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R2:   {r2:.4f}")

    # Compare CV and test RMSE.

    rmse_difference = rmse - candidate["cv_rmse"]

    print(
        f"Test RMSE - CV RMSE: {rmse_difference:.4f}"
    )

    # Save the trained model.

    model_file = os.path.join(
        MODEL_PATH,
        f"best_rating_model_{config_name}.joblib"
    )

    joblib.dump(
        final_pipeline,
        model_file
    )

    print("\nModel saved:", model_file)

    # Save predictions and errors.

    prediction_df = pd.DataFrame({
        "Actual Rating": y_test.to_numpy(),
        "Predicted Rating": predictions,
        "Absolute Error": np.abs(
            y_test.to_numpy() - predictions
        )
    })

    prediction_path = os.path.join(
        REPORT_PATH,
        f"predictions_{config_name}.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False
    )

    print("Predictions saved:", prediction_path)

    # Store final evaluation results.

    final_results.append({
        "Configuration": config_name,
        "Best Model": candidate["model_name"],
        "CV RMSE": candidate["cv_rmse"],
        "CV RMSE STD": candidate["cv_rmse_std"],
        "Test MAE": mae,
        "Test MSE": mse,
        "Test RMSE": rmse,
        "Test R2": r2,
        "Test-CV RMSE Difference": rmse_difference,
        "Train MAE": train_mae,
        "Train MSE": train_mse,
        "Train RMSE": train_rmse,
        "Train R2": train_r2
    })


# ==========================================================
# 12. FINAL MODEL COMPARISON
# ==========================================================

print("\n" + "=" * 70)
print("11. FINAL MODEL COMPARISON")
print("=" * 70)

final_df = pd.DataFrame(final_results)

print(
    final_df.to_string(index=False)
)

final_report_path = os.path.join(
    REPORT_PATH,
    "rating_final_results.csv"
)

final_df.to_csv(
    final_report_path,
    index=False
)

print("\nFinal evaluation report saved:", final_report_path)


# ----------------------------------------------------------
# 12.1 COMPARE THE TWO FEATURE CONFIGURATIONS
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("12. EFFECT OF INCLUDING VOTES")
print("=" * 70)

without_votes = final_df[
    final_df["Configuration"] == "without_votes"
].iloc[0]

with_votes = final_df[
    final_df["Configuration"] == "with_votes"
].iloc[0]

rmse_improvement = (
    without_votes["Test RMSE"] -
    with_votes["Test RMSE"]
)

mae_improvement = (
    without_votes["Test MAE"] -
    with_votes["Test MAE"]
)

r2_improvement = (
    with_votes["Test R2"] -
    without_votes["Test R2"]
)

print(
    f"RMSE improvement: {rmse_improvement:.4f}"
)

print(
    f"MAE improvement: {mae_improvement:.4f}"
)

print(
    f"R2 improvement: {r2_improvement:.4f}"
)

if without_votes["Test RMSE"] > 0:

    rmse_improvement_percentage = (
        rmse_improvement /
        without_votes["Test RMSE"]
    ) * 100

    print(
        "RMSE improvement percentage:",
        f"{rmse_improvement_percentage:.2f}%"
    )


# ----------------------------------------------------------
# 12.2 SELECT OVERALL MODEL USING CV RMSE
# ----------------------------------------------------------

best_overall = min(
    best_candidates.items(),
    key=lambda item: item[1]["cv_rmse"]
)

best_config_name = best_overall[0]

best_model_name = best_overall[1]["model_name"]

best_model_path = os.path.join(
    MODEL_PATH,
    f"best_rating_model_{best_config_name}.joblib"
)

# Save the overall selected model under a standard name.

best_overall_model = joblib.load(
    best_model_path
)

standard_model_path = os.path.join(
    MODEL_PATH,
    "best_rating_model.joblib"
)

joblib.dump(
    best_overall_model,
    standard_model_path
)

print("\n" + "=" * 70)
print("OVERALL MODEL SELECTION")
print("=" * 70)

print("Selected configuration:", best_config_name)
print("Selected algorithm:", best_model_name)
print("Selected model path:", best_model_path)
print("Standard model copy:", standard_model_path)


# ==========================================================
# 13. FINAL STATUS
# ==========================================================

print("\n" + "=" * 70)
print("RATING PREDICTION COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated reports:")

print("- rating_model_comparison.csv")
print("- rating_final_results.csv")
print("- predictions_without_votes.csv")
print("- predictions_with_votes.csv")

print("\nGenerated models:")

print("- best_rating_model_without_votes.joblib")
print("- best_rating_model_with_votes.joblib")
print("- best_rating_model.joblib")

print("\nModel training and evaluation completed.")
