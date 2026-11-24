from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ============================================================
# DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS
# TASK 02: RESTAURANT RECOMMENDATION SYSTEM
# PROGRAM 09: INTERACTIVE RECOMMENDATION SYSTEM
# ============================================================

print("=" * 75)
print("DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS")
print("TASK 02: RESTAURANT RECOMMENDATION SYSTEM")
print("PROGRAM 09 - INTERACTIVE RECOMMENDATION SYSTEM")
print("=" * 75)


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "task_2_restaurant_recommendation"
REPORT_DIR = PROJECT_ROOT / "reports" / "task_2"

CUISINE_MATRIX_PATH = (
    PROCESSED_DIR / "recommendation_cuisine_features.csv"
)

METADATA_PATH = (
    PROCESSED_DIR / "recommendation_restaurant_metadata.csv"
)

CUISINE_LABELS_PATH = (
    REPORT_DIR / "recommendation_cuisine_labels.csv"
)

HISTORY_PATH = (
    REPORT_DIR / "interactive_recommendation_history.csv"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

CUISINE_WEIGHT = 0.85
RATING_WEIGHT = 0.15


# ============================================================
# 2. LOAD ENGINEERED DATA
# ============================================================

print("\n" + "=" * 65)
print("2. LOAD ENGINEERED DATA")
print("=" * 65)

required_files = [
    CUISINE_MATRIX_PATH,
    METADATA_PATH,
    CUISINE_LABELS_PATH,
]

for file_path in required_files:

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )

cuisine_features = pd.read_csv(CUISINE_MATRIX_PATH)
metadata = pd.read_csv(METADATA_PATH)
cuisine_labels = pd.read_csv(CUISINE_LABELS_PATH)

feature_columns = [
    column
    for column in cuisine_features.columns
    if column.startswith("cuisine_")
]

if not feature_columns:
    raise ValueError("No cuisine feature columns found.")

if len(cuisine_features) != len(metadata):
    raise ValueError(
        "Cuisine features and metadata have different row counts."
    )

if not cuisine_features["Restaurant ID"].equals(
    metadata["Restaurant ID"]
):
    raise ValueError(
        "Restaurant IDs do not match between feature files."
    )

if metadata["Restaurant ID"].duplicated().any():
    raise ValueError("Duplicate restaurant IDs detected.")

print(f"Restaurants loaded: {len(metadata)}")
print(f"Cuisine features loaded: {len(feature_columns)}")
print("Engineered data loaded successfully.")


# ============================================================
# 3. PREPARE FEATURE MATRIX AND CUISINE LOOKUP
# ============================================================

print("\n" + "=" * 65)
print("3. PREPARE CUISINE FEATURE MATRIX")
print("=" * 65)

cuisine_matrix = cuisine_features[
    feature_columns
].to_numpy(dtype=np.float32)

if not np.isfinite(cuisine_matrix).all():
    raise ValueError(
        "Cuisine feature matrix contains invalid values."
    )

if not np.isin(cuisine_matrix, [0, 1]).all():
    raise ValueError(
        "Cuisine feature matrix must contain only 0 and 1."
    )

cuisine_lookup = {}

for column in feature_columns:

    cuisine_name = column.replace(
        "cuisine_",
        "",
        1
    ).replace("_", " ")

    cuisine_lookup[cuisine_name.casefold()] = column

available_cuisines = sorted(cuisine_lookup.keys())

print(f"Feature matrix shape: {cuisine_matrix.shape}")
print(f"Available cuisines: {len(available_cuisines)}")


# ============================================================
# 4. DISPLAY CUISINE OPTIONS
# ============================================================

print("\n" + "=" * 65)
print("4. AVAILABLE CUISINES")
print("=" * 65)

for index, cuisine in enumerate(available_cuisines, start=1):

    print(f"{index:3}. {cuisine.title()}")


# ============================================================
# 5. COLLECT USER PREFERENCES
# ============================================================

print("\n" + "=" * 65)
print("5. COLLECT USER PREFERENCES")
print("=" * 65)


def get_cuisine_preferences():

    while True:

        user_input = input(
            "\nEnter preferred cuisines (comma-separated): "
        ).strip()

        if not user_input:
            print("Please enter at least one cuisine.")
            continue

        selected = [
            value.strip().casefold()
            for value in user_input.split(",")
            if value.strip()
        ]

        selected = list(dict.fromkeys(selected))

        invalid = [
            cuisine for cuisine in selected
            if cuisine not in cuisine_lookup
        ]

        if invalid:

            print("\nUnknown cuisines:")

            for cuisine in invalid:
                print(f"- {cuisine}")

            continue

        return selected


def get_optional_text(prompt):

    return input(prompt).strip()


def get_price_preferences():

    while True:

        value = input(
            "\nPreferred price ranges (1-4, comma-separated, "
            "or Enter to skip): "
        ).strip()

        if not value:
            return []

        try:

            prices = [
                int(item.strip())
                for item in value.split(",")
                if item.strip()
            ]

            if not prices:
                print("Enter valid price ranges.")
                continue

            if any(price not in [1, 2, 3, 4] for price in prices):
                print("Price ranges must be between 1 and 4.")
                continue

            return sorted(set(prices))

        except ValueError:
            print("Enter numeric price ranges separated by commas.")


def get_minimum_rating():

    while True:

        value = input(
            "\nMinimum rating (1-5, or Enter to skip): "
        ).strip()

        if not value:
            return None

        try:

            rating = float(value)

            if np.isfinite(rating) and 1 <= rating <= 5:
                return rating

            print("Rating must be between 1 and 5.")

        except ValueError:
            print("Enter a valid numeric rating.")


def get_result_count():

    while True:

        value = input(
            "\nNumber of recommendations (default 10): "
        ).strip()

        if not value:
            return 10

        try:

            count = int(value)

            if count > 0:
                return count

            print("Enter a positive number.")

        except ValueError:
            print("Enter a valid whole number.")


selected_cuisines = get_cuisine_preferences()

selected_city = get_optional_text(
    "Preferred city (or Enter to skip): "
).casefold()

selected_locality = get_optional_text(
    "Preferred locality (or Enter to skip): "
).casefold()

selected_prices = get_price_preferences()

minimum_rating = get_minimum_rating()

top_n = get_result_count()

print("\nUser preferences collected successfully.")


# ============================================================
# 6. GENERATE USER CUISINE VECTOR
# ============================================================

print("\n" + "=" * 65)
print("6. GENERATE USER CUISINE VECTOR")
print("=" * 65)

user_vector = np.zeros(
    len(feature_columns),
    dtype=np.float32
)

for cuisine in selected_cuisines:

    column = cuisine_lookup[cuisine]

    position = feature_columns.index(column)

    user_vector[position] = 1

print("Selected cuisines:")

for cuisine in selected_cuisines:
    print(f"- {cuisine.title()}")

print(f"Vector dimensions: {user_vector.shape}")


# ============================================================
# 7. CALCULATE CUISINE SIMILARITY
# ============================================================

print("\n" + "=" * 65)
print("7. CALCULATE CUISINE SIMILARITY")
print("=" * 65)

similarity_scores = cosine_similarity(
    cuisine_matrix,
    user_vector.reshape(1, -1)
).ravel()

if len(similarity_scores) != len(metadata):
    raise ValueError(
        "Similarity scores do not match restaurant records."
    )

if not np.isfinite(similarity_scores).all():
    raise ValueError(
        "Invalid similarity scores generated."
    )

recommendation_data = metadata.copy()

recommendation_data["Cuisine Similarity"] = similarity_scores

print("Cosine similarity calculated successfully.")

print(
    "Maximum similarity:",
    round(float(similarity_scores.max()), 4)
)


# ============================================================
# 8. APPLY USER PREFERENCE FILTERS
# ============================================================

print("\n" + "=" * 65)
print("8. APPLY USER PREFERENCE FILTERS")
print("=" * 65)

if selected_city:

    recommendation_data = recommendation_data[
        recommendation_data["City Normalized"] == selected_city
    ]

    print(f"City filter applied: {selected_city}")

if selected_locality:

    recommendation_data = recommendation_data[
        recommendation_data["Locality Normalized"]
        == selected_locality
    ]

    print(f"Locality filter applied: {selected_locality}")

if selected_prices:

    recommendation_data = recommendation_data[
        recommendation_data["Price range"].isin(selected_prices)
    ]

    print(f"Price ranges applied: {selected_prices}")

if minimum_rating is not None:

    recommendation_data = recommendation_data[
        recommendation_data["Aggregate rating"] >= minimum_rating
    ]

    print(f"Minimum rating applied: {minimum_rating}")

eligible_count = len(recommendation_data)

print(f"\nRestaurants remaining after filters: {eligible_count}")


# ============================================================
# 9. CALCULATE RECOMMENDATION SCORES
# ============================================================

print("\n" + "=" * 65)
print("9. CALCULATE RECOMMENDATION SCORES")
print("=" * 65)

if recommendation_data.empty:

    print("\nNo restaurants match the selected preferences.")
    print("Try changing your preferences.")

    recommendations = pd.DataFrame()

else:

    rating_scores = (
        recommendation_data["Aggregate rating"] / 5.0
    )

    # Unrated restaurants retain their cuisine similarity.
    rating_scores = rating_scores.fillna(
        recommendation_data["Cuisine Similarity"]
    )

    recommendation_data["Recommendation Score"] = (
        CUISINE_WEIGHT
        * recommendation_data["Cuisine Similarity"]
        + RATING_WEIGHT
        * rating_scores
    )

    recommendation_data["Recommendation Score"] = (
        recommendation_data["Recommendation Score"].clip(0, 1)
    )

    recommendation_data = recommendation_data.sort_values(
        by=[
            "Recommendation Score",
            "Cuisine Similarity",
            "Aggregate rating",
            "Votes",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
        na_position="last"
    )

    recommendations = recommendation_data.head(top_n).copy()

    recommendations.insert(
        0,
        "Recommendation Rank",
        range(1, len(recommendations) + 1)
    )

    print("Recommendation scores calculated.")
    print("Restaurants ranked successfully.")


# ============================================================
# 10. DISPLAY RECOMMENDATIONS
# ============================================================

print("\n" + "=" * 65)
print("10. DISPLAY RECOMMENDATIONS")
print("=" * 65)

result_columns = [
    "Recommendation Rank",
    "Restaurant ID",
    "Restaurant Name",
    "City",
    "Locality",
    "Cuisines",
    "Price range",
    "Aggregate rating",
    "Votes",
    "Cuisine Similarity",
    "Recommendation Score",
]

if not recommendations.empty:

    recommendations = recommendations[result_columns]

    print(f"\nTop {len(recommendations)} recommendations:\n")

    print(
        recommendations.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

else:

    print("No recommendations available.")


# ============================================================
# 11. SAVE RECOMMENDATION SESSION
# ============================================================

print("\n" + "=" * 65)
print("11. SAVE RECOMMENDATION SESSION")
print("=" * 65)

if not recommendations.empty:

    session = recommendations.copy()

    session.insert(
        0,
        "Selected Cuisines",
        ", ".join(selected_cuisines)
    )

    session.insert(
        1,
        "Selected City",
        selected_city or "Any"
    )

    session.insert(
        2,
        "Selected Locality",
        selected_locality or "Any"
    )

    session.insert(
        3,
        "Selected Price Ranges",
        ", ".join(map(str, selected_prices))
        if selected_prices else "Any"
    )

    session.insert(
        4,
        "Minimum Rating",
        minimum_rating
        if minimum_rating is not None else "Any"
    )

    session.insert(
        5,
        "Requested Recommendations",
        top_n
    )

    session.insert(
        6,
        "Eligible Restaurants",
        eligible_count
    )

    session.insert(
        7,
        "Session Timestamp",
        pd.Timestamp.now().isoformat(timespec="seconds")
    )

    if HISTORY_PATH.exists():

        old_history = pd.read_csv(HISTORY_PATH)

        session = pd.concat(
            [old_history, session],
            ignore_index=True
        )

    session.to_csv(
        HISTORY_PATH,
        index=False
    )

    print("Recommendation session saved successfully.")
    print(f"History file: {HISTORY_PATH}")

else:

    print("No recommendation session saved because no results were generated.")


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("TASK 02 - RESTAURANT RECOMMENDATION SYSTEM")
print("PROGRAM 09 - INTERACTIVE RECOMMENDATION SYSTEM COMPLETED")
print("=" * 75)

print(f"\nSelected cuisines: {', '.join(selected_cuisines)}")
print(f"Selected city: {selected_city or 'Any'}")
print(f"Selected locality: {selected_locality or 'Any'}")
print(f"Selected price ranges: {selected_prices or 'Any'}")

print(
    "Minimum rating:",
    minimum_rating if minimum_rating is not None else "Any"
)

print(f"Eligible restaurants: {eligible_count}")
print(f"Requested recommendations: {top_n}")
print(f"Recommendations generated: {len(recommendations)}")

print("\nGenerated file:")
print("interactive_recommendation_history.csv")

print("\nPROGRAM 09 EXECUTION COMPLETED.")

print("=" * 75)
