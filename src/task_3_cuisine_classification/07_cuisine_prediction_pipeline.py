from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

from scipy.sparse import issparse

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
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

RANDOM_STATE = 42

EXPECTED_THRESHOLD = 0.225

CATEGORICAL_FEATURES = [
    "Country Code",
    "City",
    "Currency",
    "Has Table booking",
    "Has Online delivery",
    "Is delivering now",
    "Switch to order menu"
]

NUMERICAL_FEATURES = [
    "Longitude",
    "Latitude",
    "Average Cost for two",
    "Price range",
    "Aggregate rating",
    "Votes"
]

SELECTED_FEATURES = (
    CATEGORICAL_FEATURES + NUMERICAL_FEATURES
)


# ============================================================
# DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS
# TASK 03: CUISINE CLASSIFICATION
# PROGRAM 07: END-TO-END CUISINE PREDICTION PIPELINE
# ============================================================

print("=" * 75)
print("DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS")
print("TASK 03: CUISINE CLASSIFICATION")
print("PROGRAM 07 - END-TO-END CUISINE PREDICTION PIPELINE")
print("=" * 75)


# ============================================================
# 2. DEFINE INPUT AND OUTPUT PATHS
# ============================================================

print("\n" + "=" * 65)
print("1. CONFIGURE PIPELINE FILES")
print("=" * 65)

INPUT_PATH = DATA_DIR / "cuisine_X_test.csv"

PREPROCESSOR_PATH = (
    MODEL_DIR / "cuisine_feature_preprocessor.joblib"
)

MODEL_PATH = (
    MODEL_DIR / "tuned_cuisine_classification_model.joblib"
)

THRESHOLD_PATH = (
    MODEL_DIR / "cuisine_classification_threshold.joblib"
)

LABEL_MAPPING_PATH = (
    REPORT_DIR / "cuisine_label_mapping.csv"
)

EXPECTED_PREDICTIONS_PATH = (
    DATA_DIR / "cuisine_tuned_test_predictions.npy"
)

required_files = [
    INPUT_PATH,
    PREPROCESSOR_PATH,
    MODEL_PATH,
    THRESHOLD_PATH,
    LABEL_MAPPING_PATH,
    EXPECTED_PREDICTIONS_PATH
]

for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )

print("Input dataset:", INPUT_PATH.name)

print("Feature preprocessor:", PREPROCESSOR_PATH.name)

print("Tuned model:", MODEL_PATH.name)

print("Threshold artifact:", THRESHOLD_PATH.name)

print("Cuisine label mapping:", LABEL_MAPPING_PATH.name)


# ============================================================
# 3. LOAD MODEL ARTIFACTS
# ============================================================

print("\n" + "=" * 65)
print("2. LOAD TRAINED PIPELINE ARTIFACTS")
print("=" * 65)

preprocessor = joblib.load(PREPROCESSOR_PATH)

model = joblib.load(MODEL_PATH)

threshold = float(
    joblib.load(THRESHOLD_PATH)
)

label_mapping = pd.read_csv(
    LABEL_MAPPING_PATH
)

cuisine_classes = (
    label_mapping["Cuisine Label"]
    .astype(str)
    .to_list()
)

print("Feature preprocessor loaded.")

print("Tuned classification model loaded.")

print("Selected threshold:", threshold)

print("Cuisine classes:", len(cuisine_classes))


# ============================================================
# 4. VALIDATE MODEL ARTIFACTS
# ============================================================

print("\n" + "=" * 65)
print("3. VALIDATE MODEL ARTIFACTS")
print("=" * 65)

assert 0.0 < threshold < 1.0

assert np.isclose(
    threshold,
    EXPECTED_THRESHOLD
)

assert len(cuisine_classes) == 145

assert len(cuisine_classes) == len(set(cuisine_classes))

assert hasattr(preprocessor, "transform")

assert hasattr(model, "predict_proba")

print("Threshold validation: PASS")

print("Cuisine class mapping validation: PASS")

print("Preprocessor artifact validation: PASS")

print("Model probability capability validation: PASS")


# ============================================================
# 5. LOAD INPUT RESTAURANT DATA
# ============================================================

print("\n" + "=" * 65)
print("4. LOAD RESTAURANT INPUT DATA")
print("=" * 65)

input_data = pd.read_csv(INPUT_PATH)

print("Input records:", len(input_data))

print("Input columns:", len(input_data.columns))

assert len(input_data) > 0

assert "Source Row Index" in input_data.columns

assert input_data["Source Row Index"].is_unique

for feature in SELECTED_FEATURES:

    assert feature in input_data.columns, (
        f"Required feature missing: {feature}"
    )

print("Input dataset loaded successfully.")

print("Source row traceability validation: PASS")

print("Required feature validation: PASS")


# ============================================================
# 6. PREPARE RAW MODEL INPUT FEATURES
# ============================================================

print("\n" + "=" * 65)
print("5. PREPARE RAW MODEL INPUT FEATURES")
print("=" * 65)

source_row_indices = (
    input_data["Source Row Index"].to_numpy()
)

X_input = input_data[
    SELECTED_FEATURES
].copy()

assert "Cuisines" not in X_input.columns

assert "Cuisine Labels" not in X_input.columns

assert "Source Row Index" not in X_input.columns

print("Selected model input features:", len(SELECTED_FEATURES))

print("Categorical features:", len(CATEGORICAL_FEATURES))

print("Numerical features:", len(NUMERICAL_FEATURES))

print("Target leakage validation: PASS")

print("Source row index excluded from model inputs: PASS")


# ============================================================
# 7. APPLY NUMERICAL PREPROCESSING
# ============================================================

print("\n" + "=" * 65)
print("6. APPLY NUMERICAL PREPROCESSING")
print("=" * 65)

for column in NUMERICAL_FEATURES:

    X_input[column] = pd.to_numeric(
        X_input[column],
        errors="coerce"
    )

    X_input[column] = X_input[column].replace(
        [np.inf, -np.inf],
        np.nan
    )


# Apply the same transformations used in Program 03.

for column in [
    "Average Cost for two",
    "Votes"
]:

    X_input[column] = np.log1p(
        X_input[column].clip(lower=0)
    )

print("Numerical conversion completed.")

print("Invalid numerical values handled.")

print("Applied log1p transformation to:")

print("- Average Cost for two")

print("- Votes")


# ============================================================
# 8. APPLY CATEGORICAL PREPROCESSING
# ============================================================

print("\n" + "=" * 65)
print("7. APPLY CATEGORICAL PREPROCESSING")
print("=" * 65)

for column in CATEGORICAL_FEATURES:

    X_input[column] = (
        X_input[column]
        .fillna("Unknown")
        .astype(str)
    )

print("Missing categorical values handled.")

print("Categorical feature preparation completed.")


# ============================================================
# 9. TRANSFORM FEATURES USING SAVED PREPROCESSOR
# ============================================================

print("\n" + "=" * 65)
print("8. TRANSFORM INPUT FEATURES")
print("=" * 65)

X_transformed = preprocessor.transform(X_input)

print("Transformed feature matrix:", X_transformed.shape)

assert X_transformed.shape[0] == len(input_data)

assert X_transformed.shape[1] == 170

assert issparse(X_transformed)

assert np.isfinite(X_transformed.data).all()

print("Feature transformation validation: PASS")

print("Sparse matrix validation: PASS")

print("Feature dimension validation: PASS")


# ============================================================
# 10. GENERATE CUISINE PROBABILITIES
# ============================================================

print("\n" + "=" * 65)
print("9. GENERATE CUISINE PROBABILITIES")
print("=" * 65)

probabilities = model.predict_proba(
    X_transformed
)

probabilities = np.asarray(
    probabilities,
    dtype=np.float64
)

print("Probability matrix shape:", probabilities.shape)

assert probabilities.shape == (
    len(input_data),
    len(cuisine_classes)
)

assert np.isfinite(probabilities).all()

assert (
    (probabilities >= 0.0)
    & (probabilities <= 1.0)
).all()

print("Probability generation validation: PASS")


# ============================================================
# 11. GENERATE CUISINE PREDICTIONS
# ============================================================

print("\n" + "=" * 65)
print("10. GENERATE CUISINE PREDICTIONS")
print("=" * 65)

predictions = (
    probabilities >= threshold
).astype(np.uint8)

assert predictions.shape == probabilities.shape

assert np.isin(predictions, [0, 1]).all()

print("Prediction threshold:", threshold)

print("Predicted cuisine labels:", int(predictions.sum()))

print(
    "Restaurants with at least one predicted cuisine:",
    int((predictions.sum(axis=1) > 0).sum())
)

print(
    "Restaurants with no predicted cuisine:",
    int((predictions.sum(axis=1) == 0).sum())
)

print("Cuisine prediction validation: PASS")


# ============================================================
# 12. GENERATE RESTAURANT-LEVEL PREDICTION RESULTS
# ============================================================

print("\n" + "=" * 65)
print("11. GENERATE RESTAURANT-LEVEL PREDICTIONS")
print("=" * 65)

prediction_records = []

for row_index in range(len(predictions)):

    predicted_indices = np.where(
        predictions[row_index] == 1
    )[0]

    predicted_cuisines = [
        cuisine_classes[index]
        for index in predicted_indices
    ]

    prediction_records.append({

        "Source Row Index": source_row_indices[row_index],

        "Predicted Cuisine Count": len(predicted_cuisines),

        "Predicted Cuisines": json.dumps(
            predicted_cuisines,
            ensure_ascii=False
        ),

        "Prediction Threshold": threshold

    })


prediction_results = pd.DataFrame(
    prediction_records
)

print("Restaurant-level prediction records generated.")

print("\nSample predictions:")

print(
    prediction_results.head(10).to_string(index=False)
)


# ============================================================
# 13. GENERATE TOP CUISINE CONFIDENCE RESULTS
# ============================================================

print("\n" + "=" * 65)
print("12. GENERATE TOP CUISINE CONFIDENCE RESULTS")
print("=" * 65)

top_cuisine_records = []

TOP_N = 5

for row_index in range(len(probabilities)):

    top_indices = np.argsort(
        probabilities[row_index]
    )[::-1][:TOP_N]

    top_cuisines = []

    for class_index in top_indices:

        top_cuisines.append({

            "cuisine": cuisine_classes[class_index],

            "probability": round(
                float(probabilities[row_index, class_index]),
                6
            )

        })

    top_cuisine_records.append({

        "Source Row Index": source_row_indices[row_index],

        "Top Cuisine Probabilities": json.dumps(
            top_cuisines,
            ensure_ascii=False
        )

    })


top_cuisine_results = pd.DataFrame(
    top_cuisine_records
)

print("Top cuisine confidence records generated.")

print("\nSample confidence results:")

print(
    top_cuisine_results.head(5).to_string(index=False)
)


# ============================================================
# 14. SAVE PREDICTION OUTPUTS
# ============================================================

print("\n" + "=" * 65)
print("13. SAVE PREDICTION OUTPUTS")
print("=" * 65)

prediction_output_path = (
    REPORT_DIR / "cuisine_pipeline_predictions.csv"
)

confidence_output_path = (
    REPORT_DIR / "cuisine_pipeline_top_confidence.csv"
)

probability_output_path = (
    REPORT_DIR / "cuisine_pipeline_probabilities.csv"
)

prediction_results.to_csv(
    prediction_output_path,
    index=False,
    encoding="utf-8"
)

top_cuisine_results.to_csv(
    confidence_output_path,
    index=False,
    encoding="utf-8"
)

probability_results = pd.DataFrame(
    probabilities,
    columns=cuisine_classes
)

probability_results.insert(
    0,
    "Source Row Index",
    source_row_indices
)

probability_results.to_csv(
    probability_output_path,
    index=False,
    encoding="utf-8"
)

np.save(
    DATA_DIR / "cuisine_pipeline_predictions.npy",
    predictions
)

np.save(
    DATA_DIR / "cuisine_pipeline_probabilities.npy",
    probabilities
)

print("Prediction output saved:", prediction_output_path.name)

print("Confidence output saved:", confidence_output_path.name)

print("Probability output saved:", probability_output_path.name)

print("NumPy prediction and probability files saved.")


# ============================================================
# 15. VERIFY PREDICTIONS AGAINST PROGRAM 06
# ============================================================

print("\n" + "=" * 65)
print("14. VERIFY PIPELINE CONSISTENCY")
print("=" * 65)

expected_predictions = np.load(
    EXPECTED_PREDICTIONS_PATH
)

assert expected_predictions.shape == predictions.shape

matching_predictions = np.array_equal(
    predictions,
    expected_predictions
)

matching_prediction_percentage = (
    np.mean(
        predictions == expected_predictions
    ) * 100
)

print(
    "Matching prediction percentage:",
    round(matching_prediction_percentage, 4),
    "%"
)

assert matching_predictions, (
    "Pipeline predictions do not match Program 06 predictions."
)

print("Program 06 prediction consistency: PASS")


# ============================================================
# 16. GENERATE PIPELINE SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("15. GENERATE PIPELINE SUMMARY")
print("=" * 65)

summary = pd.DataFrame({

    "Metric": [

        "Input Records",

        "Selected Input Features",

        "Categorical Features",

        "Numerical Features",

        "Engineered Features",

        "Cuisine Classes",

        "Prediction Threshold",

        "Total Predicted Cuisine Labels",

        "Restaurants With Predictions",

        "Restaurants Without Predictions",

        "Average Predicted Cuisines Per Restaurant",

        "Prediction Agreement With Program 06",

        "Prediction Agreement Percentage"
    ],

    "Value": [

        len(input_data),

        len(SELECTED_FEATURES),

        len(CATEGORICAL_FEATURES),

        len(NUMERICAL_FEATURES),

        X_transformed.shape[1],

        len(cuisine_classes),

        threshold,

        int(predictions.sum()),

        int((predictions.sum(axis=1) > 0).sum()),

        int((predictions.sum(axis=1) == 0).sum()),

        float(predictions.sum(axis=1).mean()),

        "PASS" if matching_predictions else "FAIL",

        round(matching_prediction_percentage, 4)
    ]
})

print(summary.to_string(index=False))

summary.to_csv(
    REPORT_DIR / "cuisine_prediction_pipeline_summary.csv",
    index=False
)

print("\nPipeline summary saved.")


# ============================================================
# 17. FINAL PIPELINE VALIDATION
# ============================================================

print("\n" + "=" * 65)
print("16. FINAL PIPELINE VALIDATION")
print("=" * 65)

assert len(prediction_results) == len(input_data)

assert len(top_cuisine_results) == len(input_data)

assert predictions.shape == (
    len(input_data),
    len(cuisine_classes)
)

assert probabilities.shape == predictions.shape

assert prediction_results[
    "Source Row Index"
].equals(
    input_data["Source Row Index"]
)

assert prediction_results[
    "Predicted Cuisine Count"
].between(
    0,
    len(cuisine_classes)
).all()

assert matching_predictions

assert prediction_output_path.exists()

assert confidence_output_path.exists()

assert probability_output_path.exists()

assert (
    REPORT_DIR / "cuisine_prediction_pipeline_summary.csv"
).exists()

assert (
    DATA_DIR / "cuisine_pipeline_predictions.npy"
).exists()

assert (
    DATA_DIR / "cuisine_pipeline_probabilities.npy"
).exists()

print("Input data validation: PASS")

print("Preprocessing validation: PASS")

print("Feature transformation validation: PASS")

print("Prediction generation validation: PASS")

print("Cuisine confidence validation: PASS")

print("Source row traceability validation: PASS")

print("Program 06 consistency validation: PASS")

print("Output file validation: PASS")


# ============================================================
# 18. DISPLAY GENERATED FILES
# ============================================================

print("\n" + "=" * 65)
print("17. GENERATED FILES")
print("=" * 65)

print("\nGenerated reports:")

generated_reports = [

    "cuisine_pipeline_predictions.csv",

    "cuisine_pipeline_top_confidence.csv",

    "cuisine_pipeline_probabilities.csv",

    "cuisine_prediction_pipeline_summary.csv"
]

for filename in generated_reports:

    file_path = REPORT_DIR / filename

    assert file_path.exists()

    print("-", filename)


print("\nGenerated prediction data:")

print("- cuisine_pipeline_predictions.npy")

print("- cuisine_pipeline_probabilities.npy")


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 75)

print("TASK 03 - CUISINE CLASSIFICATION")

print("PROGRAM 07 - END-TO-END CUISINE PREDICTION PIPELINE COMPLETED")

print("=" * 75)
