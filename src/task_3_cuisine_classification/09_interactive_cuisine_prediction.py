from pathlib import Path
from datetime import datetime
import json

import joblib
import numpy as np
import pandas as pd


# ============================================================
# DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS
# TASK 03: CUISINE CLASSIFICATION
# PROGRAM 09: INTERACTIVE CUISINE PREDICTION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "task_3_cuisine_classification"
)

MODEL_DIR = PROJECT_ROOT / "models" / "task_3"
REPORT_DIR = PROJECT_ROOT / "reports" / "task_3"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = (
    MODEL_DIR / "tuned_cuisine_classification_model.joblib"
)

PREPROCESSOR_PATH = (
    MODEL_DIR / "cuisine_feature_preprocessor.joblib"
)

THRESHOLD_PATH = (
    MODEL_DIR / "cuisine_classification_threshold.joblib"
)

LABEL_MAPPING_PATH = (
    REPORT_DIR / "cuisine_label_mapping.csv"
)

HISTORY_PATH = (
    REPORT_DIR / "interactive_cuisine_prediction_history.csv"
)

# Selected model input features.
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
# PROJECT HEADER
# ============================================================

print("=" * 75)
print("DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS")
print("TASK 03: CUISINE CLASSIFICATION")
print("PROGRAM 09 - INTERACTIVE CUISINE PREDICTION")
print("=" * 75)


# ============================================================
# 1. LOAD SAVED MODEL ARTIFACTS
# ============================================================

print("\n" + "=" * 65)
print("1. LOAD SAVED MODEL ARTIFACTS")
print("=" * 65)

required_files = [
    MODEL_PATH,
    PREPROCESSOR_PATH,
    THRESHOLD_PATH,
    LABEL_MAPPING_PATH
]

for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required model artifact not found: {file_path}"
        )

model = joblib.load(MODEL_PATH)

preprocessor = joblib.load(PREPROCESSOR_PATH)

threshold = float(
    np.asarray(joblib.load(THRESHOLD_PATH)).item()
)

label_mapping = pd.read_csv(LABEL_MAPPING_PATH)

if "Cuisine Label" not in label_mapping.columns:
    raise ValueError(
        "Cuisine Label column missing from label mapping."
    )

cuisine_classes = (
    label_mapping["Cuisine Label"]
    .astype(str)
    .to_list()
)

if not 0.0 < threshold < 1.0:
    raise ValueError("Invalid classification threshold.")

if len(cuisine_classes) != len(set(cuisine_classes)):
    raise ValueError("Duplicate cuisine labels detected.")

if len(cuisine_classes) != 145:
    raise ValueError(
        "Expected 145 cuisine classes. "
        f"Found {len(cuisine_classes)}."
    )

print("Tuned classification model loaded.")
print("Feature preprocessor loaded.")
print("Cuisine label mapping loaded.")
print("Selected threshold:", threshold)
print("Cuisine classes:", len(cuisine_classes))

print("\nAll model artifacts loaded successfully.")


# ============================================================
# 2. INPUT VALIDATION FUNCTIONS
# ============================================================

def read_text(prompt):

    while True:

        value = input(prompt).strip()

        if value:
            return value

        print("Input cannot be empty. Try again.")


def read_number(prompt, minimum=None, maximum=None):

    while True:

        try:

            value = float(input(prompt).strip())

            if not np.isfinite(value):
                raise ValueError

            if minimum is not None and value < minimum:
                print(f"Value must be at least {minimum}.")
                continue

            if maximum is not None and value > maximum:
                print(f"Value must not exceed {maximum}.")
                continue

            return value

        except ValueError:

            print("Enter a valid numeric value.")


def read_integer(prompt, minimum=None, maximum=None):

    while True:

        try:

            value = int(input(prompt).strip())

            if minimum is not None and value < minimum:
                print(f"Value must be at least {minimum}.")
                continue

            if maximum is not None and value > maximum:
                print(f"Value must not exceed {maximum}.")
                continue

            return value

        except ValueError:

            print("Enter a valid whole number.")


def read_yes_no(prompt):

    while True:

        value = input(prompt).strip().casefold()

        if value in ["yes", "y"]:
            return "Yes"

        if value in ["no", "n"]:
            return "No"

        print("Enter Yes or No.")


# ============================================================
# 3. COLLECT RESTAURANT DETAILS
# ============================================================

def collect_restaurant_input():

    print("\n" + "=" * 65)
    print("ENTER RESTAURANT DETAILS")
    print("=" * 65)

    print("\nCATEGORICAL FEATURES")

    country_code = read_text(
        "Country Code: "
    )

    city = read_text(
        "City: "
    )

    currency = read_text(
        "Currency: "
    )

    table_booking = read_yes_no(
        "Has Table Booking? (Yes/No): "
    )

    online_delivery = read_yes_no(
        "Has Online Delivery? (Yes/No): "
    )

    delivering_now = read_yes_no(
        "Is Delivering Now? (Yes/No): "
    )

    switch_to_order_menu = read_yes_no(
        "Switch to Order Menu? (Yes/No): "
    )

    print("\nNUMERICAL FEATURES")

    longitude = read_number(
        "Longitude (-180 to 180): ",
        -180,
        180
    )

    latitude = read_number(
        "Latitude (-90 to 90): ",
        -90,
        90
    )

    average_cost = read_number(
        "Average Cost for Two: ",
        0
    )

    price_range = read_integer(
        "Price Range (1-4): ",
        1,
        4
    )

    rating = read_number(
        "Aggregate Rating (0-5): ",
        0,
        5
    )

    votes = read_integer(
        "Votes: ",
        0
    )

    restaurant_data = {
        "Country Code": country_code,
        "City": city,
        "Currency": currency,
        "Has Table booking": table_booking,
        "Has Online delivery": online_delivery,
        "Is delivering now": delivering_now,
        "Switch to order menu": switch_to_order_menu,
        "Longitude": longitude,
        "Latitude": latitude,
        "Average Cost for two": average_cost,
        "Price range": price_range,
        "Aggregate rating": rating,
        "Votes": votes
    }

    return pd.DataFrame(
        [restaurant_data],
        columns=SELECTED_FEATURES
    )


# ============================================================
# 4. APPLY TRAINED PREPROCESSING
# ============================================================

def prepare_input(restaurant_df):

    X = restaurant_df.copy()

    # Convert numerical values safely.
    for column in NUMERICAL_FEATURES:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

        X[column] = X[column].replace(
            [np.inf, -np.inf],
            np.nan
        )

    # Apply the same log transformation used during
    # feature engineering in Program 03.
    for column in [
        "Average Cost for two",
        "Votes"
    ]:

        X[column] = np.log1p(
            X[column].clip(lower=0)
        )

    # Match the categorical preprocessing convention
    # used during model training.
    for column in CATEGORICAL_FEATURES:

        X[column] = (
            X[column]
            .fillna("Unknown")
            .astype(str)
        )

    transformed = preprocessor.transform(X)

    return transformed


# ============================================================
# 5. PREDICT CUISINES
# ============================================================

def predict_cuisines(restaurant_df):

    transformed_input = prepare_input(
        restaurant_df
    )

    probabilities = model.predict_proba(
        transformed_input
    )

    probabilities = np.asarray(probabilities)

    if probabilities.shape != (1, len(cuisine_classes)):
        raise ValueError(
            "Prediction probability dimensions do not "
            "match the cuisine label mapping."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Model generated invalid probability values."
        )

    if (
        (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError(
            "Model probabilities are outside the valid range."
        )

    probability_values = probabilities[0]

    prediction_results = pd.DataFrame({
        "Cuisine": cuisine_classes,
        "Confidence": probability_values
    })

    prediction_results = prediction_results.sort_values(
        by="Confidence",
        ascending=False
    ).reset_index(drop=True)

    predicted_cuisines = prediction_results[
        prediction_results["Confidence"] >= threshold
    ].copy()

    return prediction_results, predicted_cuisines


# ============================================================
# 6. DISPLAY PREDICTION RESULTS
# ============================================================

def display_prediction(prediction_results, predicted_cuisines):

    print("\n" + "=" * 75)
    print("CUISINE CLASSIFICATION RESULTS")
    print("=" * 75)

    print(f"\nSelected threshold: {threshold:.3f}")

    if not predicted_cuisines.empty:

        print("\nPREDICTED CUISINES")

        for index, row in predicted_cuisines.iterrows():

            print(
                f"{index + 1}. {row['Cuisine'].title():<30}"
                f"Confidence: {row['Confidence']:.2%}"
            )

        print(
            "\nTotal predicted cuisines:",
            len(predicted_cuisines)
        )

    else:

        print(
            "\nNo cuisine prediction crossed the "
            "selected classification threshold."
        )

    print("\nTOP 5 MODEL SCORES")

    for index, row in prediction_results.head(5).iterrows():

        status = (
            "Above threshold"
            if row["Confidence"] >= threshold
            else "Below threshold"
        )

        print(
            f"{index + 1}. {row['Cuisine'].title():<30}"
            f"{row['Confidence']:.2%}  ({status})"
        )

    print("\n" + "=" * 75)


# ============================================================
# 7. SAVE PREDICTION HISTORY
# ============================================================

def save_prediction_history(
    restaurant_df,
    prediction_results,
    predicted_cuisines
):

    history_record = restaurant_df.iloc[0].to_dict()

    history_record["Prediction Timestamp"] = (
        datetime.now().isoformat(timespec="seconds")
    )

    history_record["Selected Threshold"] = threshold

    history_record["Predicted Cuisines"] = json.dumps(
        predicted_cuisines["Cuisine"].to_list(),
        ensure_ascii=False
    )

    history_record["Predicted Cuisine Count"] = (
        len(predicted_cuisines)
    )

    history_record["Top Cuisine"] = (
        prediction_results.iloc[0]["Cuisine"]
    )

    history_record["Top Cuisine Confidence"] = (
        prediction_results.iloc[0]["Confidence"]
    )

    new_record = pd.DataFrame([history_record])

    if HISTORY_PATH.exists():

        existing_history = pd.read_csv(
            HISTORY_PATH
        )

        updated_history = pd.concat(
            [existing_history, new_record],
            ignore_index=True
        )

    else:

        updated_history = new_record

    updated_history.to_csv(
        HISTORY_PATH,
        index=False,
        encoding="utf-8"
    )

    print("\nPrediction history saved:")
    print(HISTORY_PATH)


# ============================================================
# 8. INTERACTIVE TESTING LOOP
# ============================================================

def interactive_testing():

    prediction_count = 0

    while True:

        restaurant_df = collect_restaurant_input()

        prediction_results, predicted_cuisines = (
            predict_cuisines(restaurant_df)
        )

        display_prediction(
            prediction_results,
            predicted_cuisines
        )

        save_prediction_history(
            restaurant_df,
            prediction_results,
            predicted_cuisines
        )

        prediction_count += 1

        print("\nWould you like to test another restaurant?")

        choice = input(
            "Enter Y for Yes or N for No: "
        ).strip().casefold()

        if choice not in ["y", "yes"]:

            break

    print("\n" + "=" * 75)
    print("INTERACTIVE CUISINE PREDICTION COMPLETED")
    print("=" * 75)

    print("Restaurants tested:", prediction_count)

    print("Prediction history:")
    print(HISTORY_PATH)

    print("=" * 75)


# ============================================================
# 9. PROGRAM EXECUTION
# ============================================================

if __name__ == "__main__":

    interactive_testing()
