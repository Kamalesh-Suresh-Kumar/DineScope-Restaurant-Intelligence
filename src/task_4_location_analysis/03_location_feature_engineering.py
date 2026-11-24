from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "task_4_location_analysis"
    / "location_analysis_data.csv"
)

FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "task_4_location_analysis"
    / "features"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "task_4"
)

FEATURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS
# TASK 04: LOCATION-BASED ANALYSIS
# PROGRAM 03: LOCATION FEATURE ENGINEERING
# ============================================================

print("=" * 75)
print("DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS")
print("TASK 04: LOCATION-BASED ANALYSIS")
print("PROGRAM 03 - LOCATION FEATURE ENGINEERING")
print("=" * 75)


# ============================================================
# 1. LOAD PREPROCESSED DATASET
# ============================================================

print("\n[1] LOADING PREPROCESSED DATASET")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Preprocessed dataset not found: {DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE)

print(f"Dataset loaded: {df.shape}")
print(f"Total restaurants: {len(df)}")


# ============================================================
# 2. VERIFY REQUIRED FEATURES
# ============================================================

print("\n[2] VERIFYING REQUIRED FEATURES")

required_columns = [
    "Restaurant ID",
    "Country Code",
    "City",
    "Locality",
    "City Key",
    "Location Key",
    "Longitude",
    "Latitude",
    "Coordinate Valid",
    "Duplicate Coordinate",
    "Cuisines",
    "Aggregate rating",
    "Price range",
    "Average Cost for two",
    "Currency",
    "Votes"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required features missing: {missing_columns}"
    )

print("All required features are available.")


# ============================================================
# 3. PREPARE FEATURE DATA
# ============================================================

print("\n[3] PREPARING FEATURE DATA")

numeric_columns = [
    "Country Code",
    "Longitude",
    "Latitude",
    "Aggregate rating",
    "Price range",
    "Average Cost for two",
    "Votes"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df["Coordinate Valid"] = (
    df["Coordinate Valid"]
    .astype(str)
    .str.lower()
    .eq("true")
)

df["Duplicate Coordinate"] = (
    df["Duplicate Coordinate"]
    .astype(str)
    .str.lower()
    .eq("true")
)

df["Cuisines"] = (
    df["Cuisines"]
    .fillna("Unknown")
    .astype(str)
)

df["Currency"] = (
    df["Currency"]
    .fillna("Unknown")
    .astype(str)
)

# Exclude unknown cuisine labels from cuisine diversity analysis.
def extract_cuisines(value):

    if pd.isna(value):
        return []

    cuisines = []

    for cuisine in str(value).split(","):

        cuisine = cuisine.strip().casefold()

        if not cuisine or cuisine == "unknown":
            continue

        cuisines.append(cuisine)

    return list(dict.fromkeys(cuisines))


df["Cuisine Features"] = df["Cuisines"].apply(
    extract_cuisines
)

df["Known Cuisine Count"] = (
    df["Cuisine Features"].apply(len)
)

df["Has Known Cuisine"] = (
    df["Known Cuisine Count"] > 0
)

print("Numerical and categorical features prepared.")


# ============================================================
# 4. ENGINEER RESTAURANT-LEVEL FEATURES
# ============================================================

print("\n[4] ENGINEERING RESTAURANT-LEVEL FEATURES")

df["Rated Restaurant"] = (
    df["Aggregate rating"].notna()
)

df["High Rated Restaurant"] = (
    df["Aggregate rating"] >= 4.0
)

df["Low Rated Restaurant"] = (
    df["Aggregate rating"] < 2.5
)

df["Premium Price Restaurant"] = (
    df["Price range"] >= 3
)

df["Budget Price Restaurant"] = (
    df["Price range"] == 1
)

df["Has Valid Coordinates"] = (
    df["Coordinate Valid"]
)

df["Has Multiple Cuisine Categories"] = (
    df["Known Cuisine Count"] > 1
)

print("Restaurant-level features created.")


# ============================================================
# 5. CITY-LEVEL FEATURE ENGINEERING
# ============================================================

print("\n[5] ENGINEERING CITY-LEVEL FEATURES")

city_group = [
    "Country Code",
    "City",
    "City Key"
]

city_summary = (
    df.groupby(
        city_group,
        dropna=False
    )
    .agg(
        Restaurant_Count=(
            "Restaurant ID",
            "count"
        ),
        Rated_Restaurant_Count=(
            "Rated Restaurant",
            "sum"
        ),
        High_Rated_Restaurant_Count=(
            "High Rated Restaurant",
            "sum"
        ),
        Low_Rated_Restaurant_Count=(
            "Low Rated Restaurant",
            "sum"
        ),
        Average_Rating=(
            "Aggregate rating",
            "mean"
        ),
        Median_Rating=(
            "Aggregate rating",
            "median"
        ),
        Rating_Standard_Deviation=(
            "Aggregate rating",
            "std"
        ),
        Average_Price_Range=(
            "Price range",
            "mean"
        ),
        Average_Votes=(
            "Votes",
            "mean"
        ),
        Total_Votes=(
            "Votes",
            "sum"
        ),
        Restaurants_With_Coordinates=(
            "Coordinate Valid",
            "sum"
        ),
        Duplicate_Coordinate_Restaurants=(
            "Duplicate Coordinate",
            "sum"
        ),
        Unique_Coordinate_Pairs=(
            "Restaurant ID",
            "nunique"
        ),
        Cuisine_Diversity=(
            "Known Cuisine Count",
            "sum"
        ),
        Latitude_Center=(
            "Latitude",
            "mean"
        ),
        Longitude_Center=(
            "Longitude",
            "mean"
        ),
        Minimum_Latitude=(
            "Latitude",
            "min"
        ),
        Maximum_Latitude=(
            "Latitude",
            "max"
        ),
        Minimum_Longitude=(
            "Longitude",
            "min"
        ),
        Maximum_Longitude=(
            "Longitude",
            "max"
        )
    )
    .reset_index()
)

# Cuisine_Diversity above is the total number of cuisine assignments,
# not the number of distinct cuisine categories.
# The distinct count is calculated separately below.

city_cuisine_diversity = (
    df[
        ["Country Code", "City", "Cuisine Features"]
    ]
    .explode("Cuisine Features")
    .dropna(subset=["Cuisine Features"])
    .groupby(
        ["Country Code", "City"]
    )["Cuisine Features"]
    .nunique()
    .reset_index(
        name="Unique_Cuisine_Count"
    )
)

city_summary = city_summary.drop(
    columns=["Cuisine_Diversity"]
)

city_summary = city_summary.merge(
    city_cuisine_diversity,
    on=["Country Code", "City"],
    how="left"
)

city_summary["Unique_Cuisine_Count"] = (
    city_summary["Unique_Cuisine_Count"]
    .fillna(0)
    .astype(int)
)

city_summary["Rating_Coverage_Percentage"] = (
    city_summary["Rated_Restaurant_Count"]
    / city_summary["Restaurant_Count"]
    * 100
)

city_summary["Coordinate_Coverage_Percentage"] = (
    city_summary["Restaurants_With_Coordinates"]
    / city_summary["Restaurant_Count"]
    * 100
)

city_summary = city_summary.sort_values(
    "Restaurant_Count",
    ascending=False
)

print(f"Cities analyzed: {len(city_summary)}")
print("\nTop 10 cities:")

print(
    city_summary[
        [
            "City",
            "Restaurant_Count",
            "Average_Rating",
            "Unique_Cuisine_Count",
            "Rating_Coverage_Percentage"
        ]
    ]
    .head(10)
    .round(2)
    .to_string(index=False)
)


# ============================================================
# 6. LOCALITY-LEVEL FEATURE ENGINEERING
# ============================================================

print("\n[6] ENGINEERING LOCALITY-LEVEL FEATURES")

locality_group = [
    "Country Code",
    "City",
    "Locality",
    "Location Key"
]

locality_summary = (
    df.groupby(
        locality_group,
        dropna=False
    )
    .agg(
        Restaurant_Count=(
            "Restaurant ID",
            "count"
        ),
        Rated_Restaurant_Count=(
            "Rated Restaurant",
            "sum"
        ),
        High_Rated_Restaurant_Count=(
            "High Rated Restaurant",
            "sum"
        ),
        Low_Rated_Restaurant_Count=(
            "Low Rated Restaurant",
            "sum"
        ),
        Average_Rating=(
            "Aggregate rating",
            "mean"
        ),
        Median_Rating=(
            "Aggregate rating",
            "median"
        ),
        Rating_Standard_Deviation=(
            "Aggregate rating",
            "std"
        ),
        Average_Price_Range=(
            "Price range",
            "mean"
        ),
        Average_Votes=(
            "Votes",
            "mean"
        ),
        Total_Votes=(
            "Votes",
            "sum"
        ),
        Restaurants_With_Coordinates=(
            "Coordinate Valid",
            "sum"
        ),
        Duplicate_Coordinate_Restaurants=(
            "Duplicate Coordinate",
            "sum"
        ),
        Latitude_Center=(
            "Latitude",
            "mean"
        ),
        Longitude_Center=(
            "Longitude",
            "mean"
        )
    )
    .reset_index()
)

locality_cuisine_diversity = (
    df[
        [
            "Country Code",
            "City",
            "Locality",
            "Cuisine Features"
        ]
    ]
    .explode("Cuisine Features")
    .dropna(subset=["Cuisine Features"])
    .groupby(
        ["Country Code", "City", "Locality"]
    )["Cuisine Features"]
    .nunique()
    .reset_index(
        name="Unique_Cuisine_Count"
    )
)

locality_summary = locality_summary.merge(
    locality_cuisine_diversity,
    on=[
        "Country Code",
        "City",
        "Locality"
    ],
    how="left"
)

locality_summary["Unique_Cuisine_Count"] = (
    locality_summary["Unique_Cuisine_Count"]
    .fillna(0)
    .astype(int)
)

locality_summary["Rating_Coverage_Percentage"] = (
    locality_summary["Rated_Restaurant_Count"]
    / locality_summary["Restaurant_Count"]
    * 100
)

locality_summary["Coordinate_Coverage_Percentage"] = (
    locality_summary["Restaurants_With_Coordinates"]
    / locality_summary["Restaurant_Count"]
    * 100
)

locality_summary = locality_summary.sort_values(
    "Restaurant_Count",
    ascending=False
)

print(f"Localities analyzed: {len(locality_summary)}")

print("\nTop 10 localities:")

print(
    locality_summary[
        [
            "City",
            "Locality",
            "Restaurant_Count",
            "Average_Rating",
            "Unique_Cuisine_Count"
        ]
    ]
    .head(10)
    .round(2)
    .to_string(index=False)
)


# ============================================================
# 7. CITY-WISE CUISINE DISTRIBUTION
# ============================================================

print("\n[7] ENGINEERING CITY-WISE CUISINE DISTRIBUTION")

city_cuisine_summary = (
    df[
        [
            "Country Code",
            "City",
            "Restaurant ID",
            "Cuisine Features"
        ]
    ]
    .explode("Cuisine Features")
    .dropna(subset=["Cuisine Features"])
    .groupby(
        [
            "Country Code",
            "City",
            "Cuisine Features"
        ]
    )
    .agg(
        Restaurant_Count=(
            "Restaurant ID",
            "nunique"
        )
    )
    .reset_index()
    .rename(
        columns={
            "Cuisine Features": "Cuisine"
        }
    )
)

city_cuisine_summary = city_cuisine_summary.sort_values(
    [
        "City",
        "Restaurant_Count"
    ],
    ascending=[True, False]
)

print(
    f"City-cuisine combinations: {len(city_cuisine_summary)}"
)


# ============================================================
# 8. LOCALITY-WISE CUISINE DISTRIBUTION
# ============================================================

print("\n[8] ENGINEERING LOCALITY-WISE CUISINE DISTRIBUTION")

locality_cuisine_summary = (
    df[
        [
            "Country Code",
            "City",
            "Locality",
            "Restaurant ID",
            "Cuisine Features"
        ]
    ]
    .explode("Cuisine Features")
    .dropna(subset=["Cuisine Features"])
    .groupby(
        [
            "Country Code",
            "City",
            "Locality",
            "Cuisine Features"
        ]
    )
    .agg(
        Restaurant_Count=(
            "Restaurant ID",
            "nunique"
        )
    )
    .reset_index()
    .rename(
        columns={
            "Cuisine Features": "Cuisine"
        }
    )
)

print(
    "Locality-cuisine combinations:",
    len(locality_cuisine_summary)
)


# ============================================================
# 9. CURRENCY-AWARE COST ANALYSIS
# ============================================================

print("\n[9] ENGINEERING CURRENCY-AWARE COST FEATURES")

city_cost_summary = (
    df.groupby(
        [
            "Country Code",
            "City",
            "Currency"
        ],
        dropna=False
    )
    .agg(
        Restaurant_Count=(
            "Restaurant ID",
            "count"
        ),
        Average_Cost=(
            "Average Cost for two",
            "mean"
        ),
        Median_Cost=(
            "Average Cost for two",
            "median"
        ),
        Minimum_Cost=(
            "Average Cost for two",
            "min"
        ),
        Maximum_Cost=(
            "Average Cost for two",
            "max"
        )
    )
    .reset_index()
)

locality_cost_summary = (
    df.groupby(
        [
            "Country Code",
            "City",
            "Locality",
            "Currency"
        ],
        dropna=False
    )
    .agg(
        Restaurant_Count=(
            "Restaurant ID",
            "count"
        ),
        Average_Cost=(
            "Average Cost for two",
            "mean"
        ),
        Median_Cost=(
            "Average Cost for two",
            "median"
        ),
        Minimum_Cost=(
            "Average Cost for two",
            "min"
        ),
        Maximum_Cost=(
            "Average Cost for two",
            "max"
        )
    )
    .reset_index()
)

print(
    "City-currency groups:",
    len(city_cost_summary)
)

print(
    "Locality-currency groups:",
    len(locality_cost_summary)
)


# ============================================================
# 10. RESTAURANT DENSITY FEATURES
# ============================================================

print("\n[10] CALCULATING RESTAURANT DENSITY FEATURES")

city_density = city_summary[
    [
        "Country Code",
        "City",
        "City Key",
        "Restaurant_Count"
    ]
].copy()

city_density = city_density.rename(
    columns={
        "Restaurant_Count": "City_Restaurant_Count"
    }
)

locality_density = locality_summary[
    [
        "Country Code",
        "City",
        "Locality",
        "Location Key",
        "Restaurant_Count"
    ]
].copy()

locality_density = locality_density.rename(
    columns={
        "Restaurant_Count": "Locality_Restaurant_Count"
    }
)

locality_density = locality_density.merge(
    city_density[
        [
            "Country Code",
            "City",
            "City_Restaurant_Count"
        ]
    ],
    on=[
        "Country Code",
        "City"
    ],
    how="left"
)

locality_density["Locality_Share_Within_City"] = (
    locality_density["Locality_Restaurant_Count"]
    / locality_density["City_Restaurant_Count"]
    * 100
)

print("City and locality restaurant density features created.")


# ============================================================
# 11. LOCATION RATING COMPARISON FEATURES
# ============================================================

print("\n[11] CREATING LOCATION RATING COMPARISON FEATURES")

overall_average_rating = df[
    "Aggregate rating"
].mean()

city_summary["Rating_Difference_From_Overall"] = (
    city_summary["Average_Rating"]
    - overall_average_rating
)

locality_summary["Rating_Difference_From_Overall"] = (
    locality_summary["Average_Rating"]
    - overall_average_rating
)

print(
    "Overall rated-restaurant average:",
    round(overall_average_rating, 3)
)

print("Location rating comparison features created.")


# ============================================================
# 12. SAVE RESTAURANT-LEVEL ENGINEERED DATA
# ============================================================

print("\n[12] SAVING RESTAURANT-LEVEL ENGINEERED DATA")

restaurant_features = df.drop(
    columns=["Cuisine Features"]
)

restaurant_features.to_csv(
    FEATURE_DIR / "restaurant_location_features.csv",
    index=False
)

print("Restaurant-level features saved.")


# ============================================================
# 13. SAVE AGGREGATED FEATURE DATASETS
# ============================================================

print("\n[13] SAVING AGGREGATED FEATURE DATASETS")

output_datasets = {
    "city_location_features.csv": city_summary,
    "locality_location_features.csv": locality_summary,
    "city_cuisine_distribution.csv": city_cuisine_summary,
    "locality_cuisine_distribution.csv": locality_cuisine_summary,
    "city_currency_cost_features.csv": city_cost_summary,
    "locality_currency_cost_features.csv": locality_cost_summary,
    "city_restaurant_density.csv": city_density,
    "locality_restaurant_density.csv": locality_density
}

for filename, data in output_datasets.items():

    data.to_csv(
        FEATURE_DIR / filename,
        index=False
    )

    print(
        f"[SAVED] {filename}: {data.shape}"
    )


# ============================================================
# 14. FINAL FEATURE VALIDATION
# ============================================================

print("\n[14] FINAL FEATURE VALIDATION")

checks = {
    "Original restaurant count preserved": (
        len(df) == 9551
    ),
    "City features generated": (
        not city_summary.empty
    ),
    "Locality features generated": (
        not locality_summary.empty
    ),
    "City cuisine features generated": (
        not city_cuisine_summary.empty
    ),
    "Locality cuisine features generated": (
        not locality_cuisine_summary.empty
    ),
    "Currency-aware cost features generated": (
        not city_cost_summary.empty
        and not locality_cost_summary.empty
    ),
    "City restaurant counts reconcile": (
        int(city_summary["Restaurant_Count"].sum())
        == len(df)
    ),
    "Locality restaurant counts reconcile": (
        int(locality_summary["Restaurant_Count"].sum())
        == len(df)
    ),
    "Restaurant IDs remain unique": (
        df["Restaurant ID"].is_unique
    ),
    "Rating values remain within valid range": (
        df["Aggregate rating"].dropna().between(0, 5).all()
    ),
    "Coordinate values remain valid": (
        df.loc[
            df["Coordinate Valid"],
            "Latitude"
        ].between(-90, 90).all()
        and df.loc[
            df["Coordinate Valid"],
            "Longitude"
        ].between(-180, 180).all()
    )
}

for check, passed in checks.items():

    print(
        f"{'[PASS]' if passed else '[FAIL]'} {check}"
    )

if not all(checks.values()):
    raise RuntimeError(
        "One or more feature engineering validations failed."
    )


# ============================================================
# 15. SAVE FEATURE ENGINEERING SUMMARY
# ============================================================

print("\n[15] SAVING FEATURE ENGINEERING SUMMARY")

feature_summary = [
    ("Total Restaurants", len(df)),
    ("Total Cities", len(city_summary)),
    ("Total Localities", len(locality_summary)),
    ("City-Cuisine Combinations", len(city_cuisine_summary)),
    ("Locality-Cuisine Combinations", len(locality_cuisine_summary)),
    ("City-Currency Groups", len(city_cost_summary)),
    ("Locality-Currency Groups", len(locality_cost_summary)),
    ("Total Restaurant Cuisine Assignments", int(df["Known Cuisine Count"].sum())),
    ("Average Rating Across Rated Restaurants", round(overall_average_rating, 4)),
    ("Total City Restaurant Records", int(city_summary["Restaurant_Count"].sum())),
    ("Total Locality Restaurant Records", int(locality_summary["Restaurant_Count"].sum()))
]

feature_summary_df = pd.DataFrame(
    feature_summary,
    columns=["Metric", "Value"]
)

feature_summary_df.to_csv(
    REPORT_DIR / "location_feature_engineering_summary.csv",
    index=False
)


# ============================================================
# 16. SAVE FEATURE VALIDATION REPORT
# ============================================================

print("\n[16] SAVING FEATURE VALIDATION REPORT")

validation_report = pd.DataFrame(
    [
        {
            "Validation": check,
            "Status": "PASS" if passed else "FAIL"
        }
        for check, passed in checks.items()
    ]
)

validation_report.to_csv(
    REPORT_DIR / "location_feature_validation_report.csv",
    index=False
)


# ============================================================
# 17. SAVE FINAL TEXT REPORT
# ============================================================

print("\n[17] SAVING FINAL TEXT REPORT")

report_file = (
    REPORT_DIR
    / "location_feature_engineering_report.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS\n"
    )

    file.write(
        "TASK 04: LOCATION-BASED ANALYSIS\n"
    )

    file.write(
        "PROGRAM 03: LOCATION FEATURE ENGINEERING\n\n"
    )

    file.write("=" * 65 + "\n")
    file.write("FEATURE ENGINEERING SUMMARY\n")
    file.write("=" * 65 + "\n\n")

    for metric, value in feature_summary:
        file.write(
            f"{metric}: {value}\n"
        )

    file.write("\n")
    file.write("=" * 65 + "\n")
    file.write("VALIDATION RESULTS\n")
    file.write("=" * 65 + "\n\n")

    for check, passed in checks.items():
        file.write(
            f"{check}: {'PASS' if passed else 'FAIL'}\n"
        )

    file.write("\n")
    file.write(
        "Note: Average costs are grouped by currency. "
        "Coordinate centres are arithmetic means and are "
        "approximate summaries, not exact geographic centroids.\n"
    )


# ============================================================
# 18. FINAL OUTPUT VALIDATION
# ============================================================

print("\n[18] FINAL OUTPUT VALIDATION")

expected_files = [
    FEATURE_DIR / "restaurant_location_features.csv",
    FEATURE_DIR / "city_location_features.csv",
    FEATURE_DIR / "locality_location_features.csv",
    FEATURE_DIR / "city_cuisine_distribution.csv",
    FEATURE_DIR / "locality_cuisine_distribution.csv",
    FEATURE_DIR / "city_currency_cost_features.csv",
    FEATURE_DIR / "locality_currency_cost_features.csv",
    FEATURE_DIR / "city_restaurant_density.csv",
    FEATURE_DIR / "locality_restaurant_density.csv",
    REPORT_DIR / "location_feature_engineering_summary.csv",
    REPORT_DIR / "location_feature_validation_report.csv",
    REPORT_DIR / "location_feature_engineering_report.txt"
]

all_outputs_exist = True

for output_file in expected_files:

    exists = output_file.exists()

    print(
        f"{'[PASS]' if exists else '[FAIL]'} {output_file.name}"
    )

    if not exists:
        all_outputs_exist = False

if not all_outputs_exist:
    raise RuntimeError(
        "One or more feature engineering outputs are missing."
    )

print("\nFeature directory:")
print(FEATURE_DIR)

print("\nReport directory:")
print(REPORT_DIR)

print("\n" + "=" * 75)
print("PROGRAM 03 EXECUTION COMPLETED SUCCESSFULLY")
print("=" * 75)
