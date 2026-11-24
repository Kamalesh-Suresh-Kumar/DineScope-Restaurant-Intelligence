import os
import csv
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ============================================================
# DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS
# TASK 01: RESTAURANT RATING PREDICTION
# PROGRAM 10: INTERACTIVE RATING PREDICTION
# ============================================================

print("=" * 75)
print("DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS")
print("TASK 01: RESTAURANT RATING PREDICTION")
print("PROGRAM 10 - INTERACTIVE RATING PREDICTION")
print("=" * 75)

# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

MODEL_WITHOUT_VOTES = (
    str(PROJECT_ROOT / "models" / "task_1" / "tuned_rating_model_without_votes.joblib")
)

MODEL_WITH_VOTES = (
    str(PROJECT_ROOT / "models" / "task_1" / "tuned_rating_model_with_votes.joblib")
)

OUTPUT_DIR = str(PROJECT_ROOT / "reports" / "task_1")

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "interactive_rating_predictions.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD MODELS
# ------------------------------------------------------------

print("\n1. LOADING TRAINED MODELS")

if not os.path.exists(MODEL_WITHOUT_VOTES):
    raise FileNotFoundError(MODEL_WITHOUT_VOTES)

if not os.path.exists(MODEL_WITH_VOTES):
    raise FileNotFoundError(MODEL_WITH_VOTES)

models = {
    "1": joblib.load(MODEL_WITHOUT_VOTES),
    "2": joblib.load(MODEL_WITH_VOTES)
}

print("Both models loaded successfully.")


# ------------------------------------------------------------
# 3. INPUT VALIDATION FUNCTIONS
# ------------------------------------------------------------

def get_text(prompt):

    while True:

        value = input(prompt).strip()

        if value:
            return value

        print("This field cannot be empty.")


def get_integer(prompt, minimum=None, maximum=None):

    while True:

        try:

            value = int(input(prompt).strip())

            if minimum is not None and value < minimum:
                print(f"Enter a value of at least {minimum}.")
                continue

            if maximum is not None and value > maximum:
                print(f"Enter a value no greater than {maximum}.")
                continue

            return value

        except ValueError:
            print("Please enter a valid whole number.")


def get_float(prompt, minimum=None, maximum=None):

    while True:

        try:

            value = float(input(prompt).strip())

            if not np.isfinite(value):
                print("Enter a finite numerical value.")
                continue

            if minimum is not None and value < minimum:
                print(f"Enter a value of at least {minimum}.")
                continue

            if maximum is not None and value > maximum:
                print(f"Enter a value no greater than {maximum}.")
                continue

            return value

        except ValueError:
            print("Please enter a valid numerical value.")


def get_yes_no(prompt):

    while True:

        value = input(prompt).strip().lower()

        if value in ["yes", "y"]:
            return "Yes"

        if value in ["no", "n"]:
            return "No"

        print("Enter Yes or No.")


# ------------------------------------------------------------
# 4. SELECT MODEL
# ------------------------------------------------------------

def select_model():

    print("\n2. SELECT PREDICTION MODEL")

    print("1. Model without Votes")
    print("2. Model with Votes")

    while True:

        choice = input("Choose model (1/2): ").strip()

        if choice in models:
            return choice

        print("Invalid selection. Choose 1 or 2.")


# ------------------------------------------------------------
# 5. COLLECT RESTAURANT DETAILS
# ------------------------------------------------------------

def collect_restaurant_details(use_votes):

    print("\n3. ENTER RESTAURANT DETAILS")
    print("-" * 50)

    restaurant_name = get_text("Restaurant name: ")

    country_code = get_integer(
        "Country code: ",
        minimum=1
    )

    city = get_text("City: ")

    cuisines = get_text(
        "Cuisines (example: North Indian, Chinese): "
    )

    table_booking = get_yes_no(
        "Has table booking? (Yes/No): "
    )

    online_delivery = get_yes_no(
        "Has online delivery? (Yes/No): "
    )

    longitude = get_float(
        "Longitude (-180 to 180): ",
        minimum=-180,
        maximum=180
    )

    latitude = get_float(
        "Latitude (-90 to 90): ",
        minimum=-90,
        maximum=90
    )

    price_range = get_integer(
        "Price range (1-4): ",
        minimum=1,
        maximum=4
    )

    details = {
        "Country Code": country_code,
        "City": city,
        "Cuisines": cuisines,
        "Has Table booking": table_booking,
        "Has Online delivery": online_delivery,
        "Longitude": longitude,
        "Latitude": latitude,
        "Price range": price_range
    }

    if use_votes:

        votes = get_integer(
            "Number of votes: ",
            minimum=0
        )

        details["Votes"] = votes

    return restaurant_name, details


# ------------------------------------------------------------
# 6. GENERATE PREDICTION
# ------------------------------------------------------------

def predict_rating(model, details):

    features = list(model.feature_names_in_)

    missing_features = [
        feature for feature in features
        if feature not in details
    ]

    if missing_features:

        raise ValueError(
            f"Missing model inputs: {missing_features}"
        )

    input_df = pd.DataFrame(
        [[details[feature] for feature in features]],
        columns=features
    )

    prediction = float(model.predict(input_df)[0])

    if not np.isfinite(prediction):
        raise ValueError("The model returned an invalid prediction.")

    if prediction < 1 or prediction > 5:
        raise ValueError(
            f"Prediction outside the valid rating range: {prediction}"
        )

    return prediction


# ------------------------------------------------------------
# 7. SAVE PREDICTION
# ------------------------------------------------------------

def save_prediction(
    restaurant_name,
    model_name,
    details,
    prediction
):

    # Keep one stable column layout for both model configurations.
    # The Votes value is blank when the model without Votes is selected.
    fieldnames = [
        "Restaurant Name",
        "Model Configuration",
        "Country Code",
        "City",
        "Cuisines",
        "Has Table booking",
        "Has Online delivery",
        "Longitude",
        "Latitude",
        "Price range",
        "Votes",
        "Predicted Rating"
    ]

    record = {
        "Restaurant Name": restaurant_name,
        "Model Configuration": model_name,
        "Country Code": details["Country Code"],
        "City": details["City"],
        "Cuisines": details["Cuisines"],
        "Has Table booking": details["Has Table booking"],
        "Has Online delivery": details["Has Online delivery"],
        "Longitude": details["Longitude"],
        "Latitude": details["Latitude"],
        "Price range": details["Price range"],
        "Votes": details.get("Votes", np.nan),
        "Predicted Rating": prediction
    }
    
    # Normalize an existing history file before appending. Earlier versions
    # could have omitted the Votes header or written an extra Votes field
    # only for with_votes rows. Existing prediction values are retained.
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.reader(file))

        if rows:
            old_header = rows[0]
            normalized_rows = []

            for row in rows[1:]:
                old_values = {}

                if len(row) == len(old_header):
                    old_values = dict(zip(old_header, row))

                elif len(row) == len(old_header) + 1 and "Votes" not in old_header:
                    # Legacy with_votes row: Votes was written immediately
                    # before Predicted Rating, but was absent from the header.
                    old_values = dict(zip(old_header[:-1], row[:-2]))
                    old_values["Votes"] = row[-2]
                    old_values["Predicted Rating"] = row[-1]

                elif len(row) == len(old_header) - 1 and "Votes" in old_header:
                    # Legacy without_votes row in a file whose header already
                    # contains Votes.
                    old_values = dict(zip(old_header[:-2], row[:-1]))
                    old_values["Votes"] = ""
                    old_values["Predicted Rating"] = row[-1]

                else:
                    raise ValueError(
                        "Existing prediction history contains an unexpected "
                        "CSV row structure. Preserve a backup and inspect the file."
                    )

                normalized_rows.append([
                    old_values.get(column, "") for column in fieldnames
                ])

            with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(fieldnames)
                writer.writerows(normalized_rows)

    file_exists = os.path.exists(OUTPUT_FILE)
    result_df = pd.DataFrame([record], columns=fieldnames)
    result_df.to_csv(
        OUTPUT_FILE,
        mode="a",
        header=not file_exists,
        index=False
    )


# ------------------------------------------------------------
# 8. MAIN INTERACTIVE LOOP
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PREDICTION INTERFACE READY")
print("=" * 70)

while True:

    try:

        choice = select_model()

        selected_model = models[choice]

        use_votes = choice == "2"

        model_name = (
            "with_votes"
            if use_votes
            else "without_votes"
        )

        restaurant_name, details = collect_restaurant_details(
            use_votes
        )

        print("\n4. GENERATING PREDICTION")

        prediction = predict_rating(
            selected_model,
            details
        )

        print("\n" + "=" * 55)
        print("DINESCOPE PREDICTION RESULT")
        print("=" * 55)

        print("Restaurant:", restaurant_name)
        print("Model:", model_name)
        print(f"Predicted rating: {prediction:.2f} / 5")

        print("-" * 55)

        if prediction >= 4.0:
            print("Prediction category: High predicted rating")

        elif prediction >= 3.0:
            print("Prediction category: Moderate predicted rating")

        else:
            print("Prediction category: Low predicted rating")

        save_prediction(
            restaurant_name,
            model_name,
            details,
            prediction
        )

        print("\nPrediction saved successfully.")
        print("Output file:", OUTPUT_FILE)

    except (ValueError, TypeError) as error:

        print("\nPrediction could not be completed.")
        print("Reason:", error)

    except Exception as error:

        print("\nUnexpected error:", error)

    again = input(
        "\nDo you want to predict another restaurant? (Yes/No): "
    ).strip().lower()

    if again not in ["yes", "y"]:
        break


# ------------------------------------------------------------
# 9. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DINESCOPE PREDICTION SESSION COMPLETED")
print("=" * 70)

print("Prediction history:", OUTPUT_FILE)
print("Trained models were not modified.")
