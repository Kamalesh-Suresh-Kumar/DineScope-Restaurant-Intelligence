from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================
# DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS
# TASK 04: LOCATION-BASED ANALYSIS
# PROGRAM 08: INTERACTIVE LOCATION EXPLORER AND DECISION SUPPORT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "task_4_location_analysis"
    / "features"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "task_4"
OUTPUT_DIR = REPORT_DIR / "interactive_explorer"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures" / "task_4"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = FEATURE_DIR / "restaurant_location_features.csv"

CITY_RECOMMENDATION_FILE = (
    REPORT_DIR / "location_city_recommendations.csv"
)

LOCALITY_RECOMMENDATION_FILE = (
    REPORT_DIR / "location_locality_recommendations.csv"
)

SESSION_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

print("=" * 75)
print("DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS")
print("TASK 04: LOCATION-BASED ANALYSIS")
print("PROGRAM 08: INTERACTIVE LOCATION EXPLORER")
print("=" * 75)


# ============================================================
# SECTION 1: LOAD REQUIRED DATASETS
# ============================================================

print("\n[1] LOADING REQUIRED DATASETS")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Restaurant feature dataset not found: {DATA_FILE}"
    )

restaurants = pd.read_csv(DATA_FILE)

print("Restaurant dataset:", restaurants.shape)

city_recommendations = pd.DataFrame()
locality_recommendations = pd.DataFrame()

if CITY_RECOMMENDATION_FILE.exists():
    city_recommendations = pd.read_csv(
        CITY_RECOMMENDATION_FILE
    )
    print(
        "City recommendation dataset:",
        city_recommendations.shape
    )
else:
    print("[WARNING] City recommendation file unavailable.")

if LOCALITY_RECOMMENDATION_FILE.exists():
    locality_recommendations = pd.read_csv(
        LOCALITY_RECOMMENDATION_FILE
    )
    print(
        "Locality recommendation dataset:",
        locality_recommendations.shape
    )
else:
    print("[WARNING] Locality recommendation file unavailable.")


# ============================================================
# SECTION 2: VALIDATE AND PREPARE DATA
# ============================================================

print("\n[2] VALIDATING REQUIRED COLUMNS")

required_columns = [
    "Restaurant ID",
    "Restaurant Name",
    "City",
    "Locality",
    "Cuisines",
    "Aggregate rating",
    "Votes",
    "Price range"
]

missing_columns = [
    column
    for column in required_columns
    if column not in restaurants.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("[PASS] Required restaurant columns available.")

restaurants["Aggregate rating"] = pd.to_numeric(
    restaurants["Aggregate rating"],
    errors="coerce"
)

restaurants["Votes"] = pd.to_numeric(
    restaurants["Votes"],
    errors="coerce"
)

restaurants["Price range"] = pd.to_numeric(
    restaurants["Price range"],
    errors="coerce"
)

restaurants["City"] = (
    restaurants["City"].fillna("Unknown").astype(str).str.strip()
)

restaurants["Locality"] = (
    restaurants["Locality"].fillna("Unknown").astype(str).str.strip()
)

restaurants["Cuisines"] = (
    restaurants["Cuisines"].fillna("Unknown").astype(str).str.strip()
)

restaurants["Restaurant Name"] = (
    restaurants["Restaurant Name"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
)

restaurants["Rated"] = (
    restaurants["Aggregate rating"].between(1, 5)
)

print("[PASS] Restaurant data prepared.")

TOTAL_RESTAURANTS = len(restaurants)

RATED_RESTAURANTS = int(restaurants["Rated"].sum())

GLOBAL_AVERAGE_RATING = (
    restaurants.loc[
        restaurants["Rated"],
        "Aggregate rating"
    ].mean()
)

print("Total restaurants:", TOTAL_RESTAURANTS)
print("Rated restaurants:", RATED_RESTAURANTS)
print(
    "Global average rating:",
    round(GLOBAL_AVERAGE_RATING, 2)
)


# ============================================================
# SECTION 3: HELPER FUNCTIONS
# ============================================================

def display_header(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def display_table(data, columns=None, limit=15):
    if data.empty:
        print("No matching records found.")
        return

    if columns is not None:
        available = [
            column
            for column in columns
            if column in data.columns
        ]
        data = data[available]

    print(
        data.head(limit).to_string(
            index=False
        )
    )

    if len(data) > limit:
        print(
            f"\nShowing {limit} of {len(data)} records."
        )


def get_numeric_input(
    message,
    minimum=None,
    maximum=None,
    default=None
):
    while True:
        value = input(message).strip()

        if value == "" and default is not None:
            return default

        try:
            number = float(value)

            if minimum is not None and number < minimum:
                print("Value is below the allowed minimum.")
                continue

            if maximum is not None and number > maximum:
                print("Value exceeds the allowed maximum.")
                continue

            return number

        except ValueError:
            print("Please enter a valid number.")


def get_text_input(message, default=""):
    value = input(message).strip()

    if value == "":
        return default

    return value


def save_dataframe(data, filename):
    path = OUTPUT_DIR / filename

    data.to_csv(
        path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"[SAVED] {path.name}")

    return path


def get_restaurant_display_columns(data):
    preferred = [
        "Restaurant ID",
        "Restaurant Name",
        "Country Code",
        "City",
        "Locality",
        "Cuisines",
        "Aggregate rating",
        "Votes",
        "Price range",
        "Average Cost for two",
        "Currency"
    ]

    return [
        column
        for column in preferred
        if column in data.columns
    ]


def normalize_text(value):
    return str(value).strip().casefold()


def get_city_data(city):
    return restaurants[
        restaurants["City"].map(normalize_text)
        == normalize_text(city)
    ].copy()


def get_locality_data(city, locality):
    data = get_city_data(city)

    return data[
        data["Locality"].map(normalize_text)
        == normalize_text(locality)
    ].copy()


# ============================================================
# SECTION 4: LOCATION OVERVIEW
# ============================================================

def show_location_overview():

    display_header("LOCATION OVERVIEW")

    city_count = restaurants["City"].nunique()

    locality_count = (
        restaurants[["City", "Locality"]]
        .drop_duplicates()
        .shape[0]
    )

    print("Total restaurants:", TOTAL_RESTAURANTS)
    print("Total cities:", city_count)
    print("Total city-locality groups:", locality_count)
    print("Rated restaurants:", RATED_RESTAURANTS)
    print(
        "Unrated restaurants:",
        TOTAL_RESTAURANTS - RATED_RESTAURANTS
    )

    print(
        "Rating coverage:",
        round(
            RATED_RESTAURANTS / TOTAL_RESTAURANTS * 100,
            2
        ),
        "%"
    )

    print(
        "Global average rating:",
        round(GLOBAL_AVERAGE_RATING, 2)
    )

    print("\nTOP 10 CITIES BY RESTAURANT COUNT")

    city_counts = (
        restaurants.groupby("City")
        .size()
        .reset_index(name="Restaurant_Count")
        .sort_values(
            "Restaurant_Count",
            ascending=False
        )
    )

    display_table(city_counts, limit=10)

    print("\nTOP 10 LOCALITIES BY RESTAURANT COUNT")

    locality_counts = (
        restaurants.groupby(
            ["City", "Locality"]
        )
        .size()
        .reset_index(name="Restaurant_Count")
        .sort_values(
            "Restaurant_Count",
            ascending=False
        )
    )

    display_table(locality_counts, limit=10)


# ============================================================
# SECTION 5: CITY EXPLORER
# ============================================================

def explore_city():

    display_header("CITY EXPLORER")

    cities = sorted(
        restaurants["City"].dropna().unique()
    )

    print("Available cities:", len(cities))

    for index, city in enumerate(cities, start=1):
        print(f"{index:3}. {city}")

    selected_city = get_text_input(
        "\nEnter city name: "
    )

    data = get_city_data(selected_city)

    if data.empty:
        print("City not found. Check the spelling.")
        return

    rated = data[
        data["Rated"]
    ]

    print("\nCITY SUMMARY")

    print("City:", selected_city)
    print("Restaurant count:", len(data))
    print("Rated restaurants:", len(rated))

    if not rated.empty:
        print(
            "Average rating:",
            round(
                rated["Aggregate rating"].mean(),
                2
            )
        )

        print(
            "Median rating:",
            round(
                rated["Aggregate rating"].median(),
                2
            )
        )

    print("\nPRICE RANGE DISTRIBUTION")

    price_distribution = (
        data["Price range"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("Price_Range")
        .reset_index(name="Restaurant_Count")
    )

    display_table(price_distribution)

    print("\nTOP CUISINES")

    cuisine_counts = {}

    for cuisine_string in data["Cuisines"].dropna():

        for cuisine in str(cuisine_string).split(","):

            cuisine = cuisine.strip()

            if cuisine and cuisine.lower() != "unknown":

                cuisine_counts[cuisine] = (
                    cuisine_counts.get(cuisine, 0) + 1
                )

    cuisine_data = pd.DataFrame(
        [
            {
                "Cuisine": cuisine,
                "Restaurant_Count": count
            }
            for cuisine, count in cuisine_counts.items()
        ]
    )

    if not cuisine_data.empty:
        cuisine_data = cuisine_data.sort_values(
            "Restaurant_Count",
            ascending=False
        )

        display_table(cuisine_data, limit=10)

    print("\nTOP RATED RESTAURANTS")

    top_rated = (
        rated.sort_values(
            ["Aggregate rating", "Votes"],
            ascending=[False, False]
        )
    )

    display_table(
        top_rated,
        get_restaurant_display_columns(data),
        limit=10
    )

    save_dataframe(
        data,
        f"city_explorer_{SESSION_TIMESTAMP}.csv"
    )


# ============================================================
# SECTION 6: LOCALITY EXPLORER
# ============================================================

def explore_locality():

    display_header("LOCALITY EXPLORER")

    city = get_text_input(
        "Enter city name: "
    )

    city_data = get_city_data(city)

    if city_data.empty:
        print("City not found.")
        return

    localities = sorted(
        city_data["Locality"].dropna().unique()
    )

    print("\nAvailable localities:")

    for index, locality in enumerate(
        localities,
        start=1
    ):
        print(f"{index:3}. {locality}")

    locality = get_text_input(
        "\nEnter locality name: "
    )

    data = get_locality_data(
        city,
        locality
    )

    if data.empty:
        print("Locality not found in the selected city.")
        return

    rated = data[data["Rated"]]

    print("\nLOCALITY SUMMARY")

    print("City:", city)
    print("Locality:", locality)
    print("Restaurant count:", len(data))
    print("Rated restaurants:", len(rated))

    if not rated.empty:
        print(
            "Average rating:",
            round(
                rated["Aggregate rating"].mean(),
                2
            )
        )

    print("\nRESTAURANTS IN THIS LOCALITY")

    display_table(
        data.sort_values(
            ["Aggregate rating", "Votes"],
            ascending=[False, False]
        ),
        get_restaurant_display_columns(data),
        limit=20
    )

    save_dataframe(
        data,
        f"locality_explorer_{SESSION_TIMESTAMP}.csv"
    )


# ============================================================
# SECTION 7: ADVANCED RESTAURANT FILTERING
# ============================================================

def advanced_restaurant_search():

    display_header("ADVANCED RESTAURANT SEARCH")

    data = restaurants.copy()

    country_column = "Country Code"

    if country_column in data.columns:

        use_country = get_text_input(
            "Filter by country code? (y/n): ",
            "n"
        ).lower()

        if use_country == "y":

            country = get_numeric_input(
                "Enter country code: ",
                minimum=0
            )

            data = data[
                pd.to_numeric(
                    data[country_column],
                    errors="coerce"
                ) == country
            ]

    city = get_text_input(
        "Enter city (leave blank for all): "
    )

    if city:
        data = data[
            data["City"].map(normalize_text)
            == normalize_text(city)
        ]

    locality = get_text_input(
        "Enter locality (leave blank for all): "
    )

    if locality:
        data = data[
            data["Locality"].map(normalize_text)
            == normalize_text(locality)
        ]

    cuisine = get_text_input(
        "Enter cuisine (leave blank for all): "
    )

    if cuisine:
        data = data[
            data["Cuisines"].str.contains(
                cuisine,
                case=False,
                na=False,
                regex=False
            )
        ]

    minimum_rating = get_numeric_input(
        "Minimum rating (0 to 5, default 0): ",
        minimum=0,
        maximum=5,
        default=0
    )

    if minimum_rating > 0:
        data = data[
            data["Aggregate rating"] >= minimum_rating
        ]

    minimum_price = get_numeric_input(
        "Minimum price range (1 to 4, default 1): ",
        minimum=1,
        maximum=4,
        default=1
    )

    maximum_price = get_numeric_input(
        "Maximum price range (1 to 4, default 4): ",
        minimum=1,
        maximum=4,
        default=4
    )

    if minimum_price > maximum_price:
        print("Minimum price cannot exceed maximum price.")
        return

    data = data[
        data["Price range"].between(
            minimum_price,
            maximum_price
        )
    ]

    minimum_votes = get_numeric_input(
        "Minimum votes (default 0): ",
        minimum=0,
        default=0
    )

    data = data[
        data["Votes"].fillna(0) >= minimum_votes
    ]

    print("\nFILTER RESULTS")

    print("Matching restaurants:", len(data))

    if data.empty:
        print("No restaurants match the selected filters.")
        return

    data = data.sort_values(
        ["Aggregate rating", "Votes"],
        ascending=[False, False]
    )

    display_table(
        data,
        get_restaurant_display_columns(data),
        limit=25
    )

    filename = (
        f"advanced_search_{SESSION_TIMESTAMP}.csv"
    )

    save_dataframe(data, filename)


# ============================================================
# SECTION 8: CITY RECOMMENDATION EXPLORER
# ============================================================

def show_city_recommendations():

    display_header("CITY RECOMMENDATION EXPLORER")

    if city_recommendations.empty:
        print("City recommendation dataset unavailable.")
        return

    required = [
        "Recommendation_Rank",
        "City",
        "Recommendation_Score",
        "Recommendation_Tier",
        "Restaurant_Count",
        "Average_Rating",
        "Adjusted_Average_Rating"
    ]

    available = [
        column
        for column in required
        if column in city_recommendations.columns
    ]

    display_table(
        city_recommendations.sort_values(
            "Recommendation_Score",
            ascending=False
        ),
        available,
        limit=20
    )

    selected_city = get_text_input(
        "\nEnter a city to inspect its recommendation: "
    )

    result = city_recommendations[
        city_recommendations["City"].map(normalize_text)
        == normalize_text(selected_city)
    ]

    if result.empty:
        print("City not present in eligible recommendations.")
        return

    print("\nSELECTED CITY RECOMMENDATION")

    display_table(result, available, limit=1)


# ============================================================
# SECTION 9: LOCALITY RECOMMENDATION EXPLORER
# ============================================================

def show_locality_recommendations():

    display_header("LOCALITY RECOMMENDATION EXPLORER")

    if locality_recommendations.empty:
        print("Locality recommendation dataset unavailable.")
        return

    required = [
        "Recommendation_Rank",
        "City",
        "Locality",
        "Recommendation_Score",
        "Recommendation_Tier",
        "Restaurant_Count",
        "Average_Rating",
        "Adjusted_Average_Rating"
    ]

    available = [
        column
        for column in required
        if column in locality_recommendations.columns
    ]

    display_table(
        locality_recommendations.sort_values(
            "Recommendation_Score",
            ascending=False
        ),
        available,
        limit=20
    )

    city = get_text_input(
        "\nEnter city: "
    )

    locality = get_text_input(
        "Enter locality: "
    )

    result = locality_recommendations[
        (
            locality_recommendations["City"].map(normalize_text)
            == normalize_text(city)
        )
        &
        (
            locality_recommendations["Locality"].map(normalize_text)
            == normalize_text(locality)
        )
    ]

    if result.empty:
        print("Locality not present in eligible recommendations.")
        return

    print("\nSELECTED LOCALITY RECOMMENDATION")

    display_table(result, available, limit=1)


# ============================================================
# SECTION 10: LOCATION COMPARISON
# ============================================================

def compare_locations():

    display_header("LOCATION COMPARISON")

    first_city = get_text_input(
        "Enter first city: "
    )

    second_city = get_text_input(
        "Enter second city: "
    )

    first_data = get_city_data(first_city)
    second_data = get_city_data(second_city)

    if first_data.empty or second_data.empty:
        print("One or both cities were not found.")
        return

    comparison = []

    for city_name, data in [
        (first_city, first_data),
        (second_city, second_data)
    ]:

        rated = data[data["Rated"]]

        comparison.append({
            "City": city_name,
            "Restaurant_Count": len(data),
            "Rated_Restaurant_Count": len(rated),
            "Average_Rating": (
                rated["Aggregate rating"].mean()
                if not rated.empty
                else np.nan
            ),
            "Rating_Coverage_Percentage": (
                len(rated) / len(data) * 100
                if len(data) > 0
                else 0
            ),
            "Unique_Localities": data["Locality"].nunique(),
            "Unique_Cuisine_Combinations": data["Cuisines"].nunique()
        })

    comparison_df = pd.DataFrame(comparison)

    print("\nCITY COMPARISON")

    display_table(comparison_df, limit=10)

    save_dataframe(
        comparison_df,
        f"city_comparison_{SESSION_TIMESTAMP}.csv"
    )


# ============================================================
# SECTION 11: GENERATE CUSTOM RESTAURANT SHORTLIST
# ============================================================

def generate_shortlist():

    display_header("CUSTOM RESTAURANT SHORTLIST")

    city = get_text_input(
        "Enter preferred city: "
    )

    data = get_city_data(city)

    if data.empty:
        print("City not found.")
        return

    cuisine = get_text_input(
        "Preferred cuisine (leave blank for any): "
    )

    if cuisine:
        data = data[
            data["Cuisines"].str.contains(
                cuisine,
                case=False,
                na=False,
                regex=False
            )
        ]

    minimum_rating = get_numeric_input(
        "Minimum rating (0 to 5): ",
        minimum=0,
        maximum=5,
        default=3
    )

    data = data[
        data["Aggregate rating"] >= minimum_rating
    ]

    maximum_price = get_numeric_input(
        "Maximum price range (1 to 4): ",
        minimum=1,
        maximum=4,
        default=4
    )

    data = data[
        data["Price range"] <= maximum_price
    ]

    shortlist_size = int(
        get_numeric_input(
            "Number of restaurants to shortlist: ",
            minimum=1,
            maximum=100,
            default=10
        )
    )

    data = data.sort_values(
        ["Aggregate rating", "Votes"],
        ascending=[False, False]
    )

    shortlist = data.head(shortlist_size).copy()

    if shortlist.empty:
        print("No restaurants match the preferences.")
        return

    shortlist.insert(
        0,
        "Shortlist_Rank",
        range(1, len(shortlist) + 1)
    )

    print("\nGENERATED SHORTLIST")

    display_table(
        shortlist,
        [
            "Shortlist_Rank"
        ] + get_restaurant_display_columns(shortlist),
        limit=shortlist_size
    )

    save_dataframe(
        shortlist,
        f"custom_restaurant_shortlist_{SESSION_TIMESTAMP}.csv"
    )


# ============================================================
# SECTION 12: SAVE SESSION REPORT
# ============================================================

def save_session_report():

    report_path = (
        OUTPUT_DIR
        / f"interactive_explorer_session_{SESSION_TIMESTAMP}.txt"
    )

    lines = [
        "DINESCOPE - INTERACTIVE LOCATION EXPLORER",
        "=" * 65,
        f"Session timestamp: {SESSION_TIMESTAMP}",
        f"Total restaurants: {TOTAL_RESTAURANTS}",
        f"Rated restaurants: {RATED_RESTAURANTS}",
        f"Total cities: {restaurants['City'].nunique()}",
        (
            "Global average rating: "
            f"{GLOBAL_AVERAGE_RATING:.2f}"
        ),
        "",
        "Available modules:",
        "1. Location overview",
        "2. City explorer",
        "3. Locality explorer",
        "4. Advanced restaurant search",
        "5. City recommendation explorer",
        "6. Locality recommendation explorer",
        "7. Location comparison",
        "8. Custom restaurant shortlist",
        "",
        "Generated files are stored in:",
        str(OUTPUT_DIR),
        "",
        "Note:",
        "Recommendations are based on historical dataset values.",
        "They do not represent live availability or current ratings."
    ]

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print("[SAVED]", report_path.name)


# ============================================================
# SECTION 13: INTERACTIVE MAIN MENU
# ============================================================

def main_menu():

    while True:

        display_header(
            "DINESCOPE - INTERACTIVE LOCATION EXPLORER"
        )

        print("1.  Location Overview")
        print("2.  Explore a City")
        print("3.  Explore a Locality")
        print("4.  Advanced Restaurant Search")
        print("5.  City Recommendation Explorer")
        print("6.  Locality Recommendation Explorer")
        print("7.  Compare Two Cities")
        print("8.  Generate Custom Restaurant Shortlist")
        print("9.  Save Session Report")
        print("0.  Exit")

        choice = input(
            "\nSelect an option: "
        ).strip()

        try:

            if choice == "1":
                show_location_overview()

            elif choice == "2":
                explore_city()

            elif choice == "3":
                explore_locality()

            elif choice == "4":
                advanced_restaurant_search()

            elif choice == "5":
                show_city_recommendations()

            elif choice == "6":
                show_locality_recommendations()

            elif choice == "7":
                compare_locations()

            elif choice == "8":
                generate_shortlist()

            elif choice == "9":
                save_session_report()

            elif choice == "0":

                save_session_report()

                print("\nThank you for using DineScope.")
                print("Interactive explorer closed.")

                break

            else:
                print("Invalid selection. Choose an option from 0 to 9.")

        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled.")

        except Exception as error:
            print("\n[ERROR]", error)
            print(
                "The current operation could not be completed."
            )


# ============================================================
# SECTION 14: FINAL VALIDATION
# ============================================================

print("\n[3] RUNNING INITIAL VALIDATION")

validation_checks = {
    "Restaurant dataset available": (
        not restaurants.empty
    ),
    "Restaurant IDs unique": (
        restaurants["Restaurant ID"].is_unique
    ),
    "City column available": (
        restaurants["City"].notna().all()
    ),
    "Locality column available": (
        restaurants["Locality"].notna().all()
    ),
    "Restaurant ratings within valid range": (
        restaurants["Aggregate rating"]
        .dropna()
        .between(0, 5)
        .all()
    ),
    "Price ranges within valid range": (
        restaurants["Price range"]
        .dropna()
        .between(1, 4)
        .all()
    ),
    "Output directory available": (
        OUTPUT_DIR.exists()
    )
}

for check_name, result in validation_checks.items():

    status = "PASS" if result else "FAIL"

    print(f"[{status}] {check_name}")

if not all(validation_checks.values()):
    raise RuntimeError(
        "Initial validation failed. Review the dataset."
    )


# ============================================================
# SECTION 15: EXECUTION SUMMARY AND START APPLICATION
# ============================================================

print("\n[4] EXECUTION SUMMARY")

print("Total restaurants:", TOTAL_RESTAURANTS)
print("Rated restaurants:", RATED_RESTAURANTS)
print("Total cities:", restaurants["City"].nunique())

print(
    "Total locality groups:",
    restaurants[["City", "Locality"]]
    .drop_duplicates()
    .shape[0]
)

print("Output directory:", OUTPUT_DIR)

print("\n" + "=" * 75)
print("PROGRAM 08 INITIALIZATION COMPLETED SUCCESSFULLY")
print("=" * 75)

main_menu()

print("\n" + "=" * 75)
print("PROGRAM 08 EXECUTION COMPLETED")
print("=" * 75)
