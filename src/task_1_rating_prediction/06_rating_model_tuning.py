import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ============================================================
# DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS
# TASK 01: RESTAURANT RATING PREDICTION
# PROGRAM 06: RANDOM FOREST HYPERPARAMETER TUNING
# ============================================================

print("=" * 75)
print("DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS")
print("TASK 01: RESTAURANT RATING PREDICTION")
print("PROGRAM 06 - RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 75)

# ==========================================================
# 1. CONFIGURATION
# ==========================================================

TRAIN_PATH = str(PROJECT_ROOT / "data" / "processed" / "task_1_rating_prediction" / "train.csv")
TEST_PATH = str(PROJECT_ROOT / "data" / "processed" / "task_1_rating_prediction" / "test.csv")

MODEL_PATH = str(PROJECT_ROOT / "models" / "task_1")
REPORT_PATH = str(PROJECT_ROOT / "reports" / "task_1")

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(REPORT_PATH, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5
N_ITER = 20

TARGET = "Aggregate rating"


# ==========================================================
# 2. LOAD DATASETS
# ==========================================================

print("\n1. LOADING DATASETS")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Training shape:", train_df.shape)
print("Testing shape:", test_df.shape)

if train_df.empty or test_df.empty:
    raise ValueError("Training or testing dataset is empty.")


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


# ==========================================================
# 4. PREPARE TRAINING AND TESTING DATA
# ==========================================================

print("\n3. PREPARING DATA")

y_train = pd.to_numeric(
    train_df[TARGET],
    errors="coerce"
)

y_test = pd.to_numeric(
    test_df[TARGET],
    errors="coerce"
)

if y_train.isnull().any() or y_test.isnull().any():
    raise ValueError("Invalid target values detected.")

if not y_train.between(1, 5).all():
    raise ValueError("Invalid training ratings.")

if not y_test.between(1, 5).all():
    raise ValueError("Invalid testing ratings.")


# ==========================================================
# 5. DEFINE EVALUATION FUNCTION
# ==========================================================

def evaluate_model(actual, predicted):

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
# 6. DEFINE HYPERPARAMETER SEARCH SPACE
# ==========================================================

print("\n4. DEFINING HYPERPARAMETER SEARCH SPACE")

parameter_space = {
    "regressor__n_estimators": [
        150, 200, 250, 300, 350
    ],

    "regressor__max_depth": [
        None, 10, 15, 20, 25, 30
    ],

    "regressor__min_samples_split": [
        2, 5, 10
    ],

    "regressor__min_samples_leaf": [
        1, 2, 4, 6
    ],

    "regressor__max_features": [
        "sqrt", 0.7, 1.0
    ]
}

print("Number of search iterations:", N_ITER)
print("Cross-validation folds:", N_SPLITS)


# ==========================================================
# 7. CROSS-VALIDATION CONFIGURATION
# ==========================================================

cv = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ==========================================================
# 8. HYPERPARAMETER TUNING
# ==========================================================

print("\n5. STARTING HYPERPARAMETER TUNING")

tuning_results = []
best_tuned_models = {}

for config_name, config in feature_configs.items():

    print("\n" + "=" * 70)
    print("TUNING:", config_name.upper())
    print("=" * 70)

    selected_features = config["features"]

    X_train = train_df[selected_features]
    X_test = test_df[selected_features]

    # Load the original baseline model.

    baseline_path = os.path.join(
        MODEL_PATH,
        f"best_rating_model_{config_name}.joblib"
    )

    if not os.path.exists(baseline_path):
        raise FileNotFoundError(
            f"Baseline model not found: {baseline_path}"
        )

    baseline_model = joblib.load(
        baseline_path
    )

    # Keep the original model untouched.
    # Clone only its preprocessing pipeline.

    preprocessor = clone(
        baseline_model.named_steps["preprocessor"]
    )

    # Define a new Random Forest pipeline.

    tuning_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),

            (
                "regressor",
                RandomForestRegressor(
                    random_state=RANDOM_STATE,
                    n_jobs=1
                )
            )
        ]
    )

    # Perform randomized hyperparameter search.

    search = RandomizedSearchCV(
        estimator=tuning_pipeline,
        param_distributions=parameter_space,
        n_iter=N_ITER,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=1,
        return_train_score=True,
        refit=True
    )

    search.fit(
        X_train,
        y_train
    )

    print("\nBest parameters:")

    for parameter, value in search.best_params_.items():
        print(f"{parameter}: {value}")

    best_cv_rmse = -search.best_score_

    print("\nBest CV RMSE:", round(best_cv_rmse, 4))

    # The best estimator is already fitted on all
    # training records because refit=True.

    # The fitted RandomizedSearchCV object delegates
    # prediction to its best fitted estimator.

    tuned_model = search.best_estimator_

    tuned_predictions = search.predict(
        X_test
    )

    tuned_metrics = evaluate_model(
        y_test,
        tuned_predictions
    )

    print("\nTUNED MODEL TEST PERFORMANCE")

    for metric, value in tuned_metrics.items():
        print(f"Test {metric}: {value:.4f}")

    # Evaluate the original baseline model
    # on the exact same test records.

    baseline_predictions = baseline_model.predict(
        X_test
    )

    baseline_metrics = evaluate_model(
        y_test,
        baseline_predictions
    )

    print("\nORIGINAL BASELINE TEST PERFORMANCE")

    for metric, value in baseline_metrics.items():
        print(f"Baseline {metric}: {value:.4f}")

    # Calculate the RMSE improvement.

    rmse_improvement = (
        baseline_metrics["RMSE"] -
        tuned_metrics["RMSE"]
    )

    print(
        "\nTest RMSE improvement:",
        round(rmse_improvement, 4)
    )

    if baseline_metrics["RMSE"] > 0:

        improvement_percentage = (
            rmse_improvement /
            baseline_metrics["RMSE"]
        ) * 100

        print(
            "RMSE improvement percentage:",
            f"{improvement_percentage:.2f}%"
        )

    # Save tuned model separately.

    tuned_model_path = os.path.join(
        MODEL_PATH,
        f"tuned_rating_model_{config_name}.joblib"
    )

    joblib.dump(
        tuned_model,
        tuned_model_path
    )

    print("\nTuned model saved:", tuned_model_path)

    best_tuned_models[config_name] = tuned_model

    # Store comparison results.

    tuning_results.append({
        "Configuration": config_name,

        "Best CV RMSE": best_cv_rmse,

        "Baseline Test RMSE": baseline_metrics["RMSE"],

        "Tuned Test RMSE": tuned_metrics["RMSE"],

        "Baseline Test MAE": baseline_metrics["MAE"],

        "Tuned Test MAE": tuned_metrics["MAE"],

        "Baseline Test R2": baseline_metrics["R2"],

        "Tuned Test R2": tuned_metrics["R2"],

        "RMSE Improvement": rmse_improvement,

        "Best Parameters": str(search.best_params_)
    })

    # Save complete CV search results.

    search_results = pd.DataFrame(
        search.cv_results_
    )

    search_results = search_results.sort_values(
        by="rank_test_score"
    )

    search_report_path = os.path.join(
        REPORT_PATH,
        f"tuning_search_results_{config_name}.csv"
    )

    search_results.to_csv(
        search_report_path,
        index=False
    )

    print(
        "Search results saved:",
        search_report_path
    )


# ==========================================================
# 9. SAVE FINAL TUNING COMPARISON
# ==========================================================

print("\n" + "=" * 70)
print("6. FINAL TUNING COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(
    tuning_results
)

print(
    results_df.to_string(index=False)
)

comparison_path = os.path.join(
    REPORT_PATH,
    "rating_tuning_comparison.csv"
)

results_df.to_csv(
    comparison_path,
    index=False
)

print("\nTuning comparison saved:", comparison_path)


# ==========================================================
# 10. SELECT BEST TUNED CONFIGURATION
# ==========================================================

print("\n" + "=" * 70)
print("7. BEST TUNED CONFIGURATION")
print("=" * 70)

best_index = int(
    results_df["Best CV RMSE"].to_numpy().argmin()
)

best_row = results_df.iloc[best_index]

print(
    "Best configuration:",
    best_row["Configuration"]
)

print(
    "Best CV RMSE:",
    round(best_row["Best CV RMSE"], 4)
)

print(
    "Corresponding test RMSE:",
    round(best_row["Tuned Test RMSE"], 4)
)


# ==========================================================
# 11. FINAL STATUS
# ==========================================================

print("\n" + "=" * 70)
print("RANDOM FOREST TUNING COMPLETED")
print("=" * 70)

print("\nOriginal models preserved.")

print("\nNew tuned models:")

print("- tuned_rating_model_without_votes.joblib")
print("- tuned_rating_model_with_votes.joblib")

print("\nGenerated reports:")

print("- rating_tuning_comparison.csv")
print("- tuning_search_results_without_votes.csv")
print("- tuning_search_results_with_votes.csv")

print("\nTuning process completed successfully.")
