from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "Dataset.csv"

REPORT_DIR = PROJECT_ROOT / "reports" / "task_4"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS
# TASK 04: LOCATION-BASED ANALYSIS
# PROGRAM 01: LOCATION DATASET AUDIT
# ============================================================

print("=" * 75)
print("DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS")
print("TASK 04: LOCATION-BASED ANALYSIS")
print("PROGRAM 01 - LOCATION DATASET AUDIT")
print("=" * 75)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\n[1] LOADING DATASET")

if not RAW_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {RAW_FILE}"
    )

df = pd.read_csv(RAW_FILE, encoding="utf-8")

print(f"Dataset loaded successfully.")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


# ============================================================
# 2. VERIFY REQUIRED COLUMNS
# ============================================================

print("\n[2] VERIFYING REQUIRED COLUMNS")

required_columns = [
    "Restaurant ID",
    "Restaurant Name",
    "Country Code",
    "City",
    "Address",
    "Locality",
    "Locality Verbose",
    "Longitude",
    "Latitude",
    "Cuisines",
    "Average Cost for two",
    "Currency",
    "Price range",
    "Aggregate rating",
    "Votes"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns missing: {missing_columns}"
    )

print("All required columns are available.")


# ============================================================
# 3. DATASET STRUCTURE AND DUPLICATE AUDIT
# ============================================================

print("\n[3] DATASET STRUCTURE AND DUPLICATE AUDIT")

total_rows = len(df)
total_columns = len(df.columns)

duplicate_rows = int(df.duplicated().sum())

duplicate_ids = int(
    df["Restaurant ID"].duplicated().sum()
)

unique_restaurants = int(
    df["Restaurant ID"].nunique()
)

print(f"Total records: {total_rows}")
print(f"Total columns: {total_columns}")
print(f"Duplicate rows: {duplicate_rows}")
print(f"Duplicate restaurant IDs: {duplicate_ids}")
print(f"Unique restaurant IDs: {unique_restaurants}")


# ============================================================
# 4. GEOGRAPHICAL DATA QUALITY AUDIT
# ============================================================

print("\n[4] GEOGRAPHICAL DATA QUALITY AUDIT")

df["Longitude"] = pd.to_numeric(
    df["Longitude"], errors="coerce"
)

df["Latitude"] = pd.to_numeric(
    df["Latitude"], errors="coerce"
)

missing_longitude = int(
    df["Longitude"].isna().sum()
)

missing_latitude = int(
    df["Latitude"].isna().sum()
)

missing_both_coordinates = int(
    (
        df["Longitude"].isna()
        & df["Latitude"].isna()
    ).sum()
)

missing_any_coordinate = int(
    (
        df["Longitude"].isna()
        | df["Latitude"].isna()
    ).sum()
)

invalid_longitude = int(
    (
        (df["Longitude"] < -180)
        | (df["Longitude"] > 180)
    ).sum()
)

invalid_latitude = int(
    (
        (df["Latitude"] < -90)
        | (df["Latitude"] > 90)
    ).sum()
)

valid_coordinates = (
    df["Longitude"].between(-180, 180)
    & df["Latitude"].between(-90, 90)
)

valid_coordinate_count = int(
    valid_coordinates.sum()
)

invalid_coordinate_count = int(
    (~valid_coordinates).sum()
)

print(f"Missing longitude: {missing_longitude}")
print(f"Missing latitude: {missing_latitude}")
print(f"Missing both coordinates: {missing_both_coordinates}")
print(f"Missing at least one coordinate: {missing_any_coordinate}")
print(f"Invalid longitude values: {invalid_longitude}")
print(f"Invalid latitude values: {invalid_latitude}")
print(f"Valid coordinate pairs: {valid_coordinate_count}")
print(f"Invalid or incomplete coordinate pairs: {invalid_coordinate_count}")


# ============================================================
# 5. CITY AND LOCALITY COVERAGE
# ============================================================

print("\n[5] CITY AND LOCALITY COVERAGE")

unique_cities = int(
    df["City"].nunique()
)

unique_localities = int(
    df["Locality"].nunique()
)

unique_countries = int(
    df["Country Code"].nunique()
)

missing_cities = int(
    df["City"].isna().sum()
)

missing_localities = int(
    df["Locality"].isna().sum()
)

print(f"Unique countries: {unique_countries}")
print(f"Unique cities: {unique_cities}")
print(f"Unique localities: {unique_localities}")
print(f"Missing city values: {missing_cities}")
print(f"Missing locality values: {missing_localities}")


# ============================================================
# 6. RESTAURANT DISTRIBUTION BY CITY
# ============================================================

print("\n[6] RESTAURANT DISTRIBUTION BY CITY")

city_distribution = (
    df.groupby("City", dropna=False)
    .agg(
        Restaurant_Count=("Restaurant ID", "count"),
        Average_Rating=("Aggregate rating", "mean"),
        Average_Price_Range=("Price range", "mean"),
        Average_Cost=("Average Cost for two", "mean"),
        Average_Votes=("Votes", "mean"),
        Restaurants_With_Coordinates=("Latitude", "count")
    )
    .reset_index()
    .sort_values(
        "Restaurant_Count",
        ascending=False
    )
)

city_distribution["Average_Rating"] = (
    city_distribution["Average_Rating"].round(2)
)

city_distribution["Average_Price_Range"] = (
    city_distribution["Average_Price_Range"].round(2)
)

city_distribution["Average_Cost"] = (
    city_distribution["Average_Cost"].round(2)
)

city_distribution["Average_Votes"] = (
    city_distribution["Average_Votes"].round(2)
)

print("\nTop 10 cities by restaurant count:")

print(
    city_distribution.head(10).to_string(index=False)
)


# ============================================================
# 7. RESTAURANT DISTRIBUTION BY LOCALITY
# ============================================================

print("\n[7] RESTAURANT DISTRIBUTION BY LOCALITY")

locality_distribution = (
    df.groupby(
        ["City", "Locality"],
        dropna=False
    )
    .agg(
        Restaurant_Count=("Restaurant ID", "count"),
        Average_Rating=("Aggregate rating", "mean"),
        Average_Price_Range=("Price range", "mean"),
        Average_Cost=("Average Cost for two", "mean"),
        Average_Votes=("Votes", "mean")
    )
    .reset_index()
    .sort_values(
        "Restaurant_Count",
        ascending=False
    )
)

locality_distribution["Average_Rating"] = (
    locality_distribution["Average_Rating"].round(2)
)

locality_distribution["Average_Price_Range"] = (
    locality_distribution["Average_Price_Range"].round(2)
)

locality_distribution["Average_Cost"] = (
    locality_distribution["Average_Cost"].round(2)
)

locality_distribution["Average_Votes"] = (
    locality_distribution["Average_Votes"].round(2)
)

print("\nTop 10 localities by restaurant count:")

print(
    locality_distribution.head(10).to_string(index=False)
)


# ============================================================
# 8. RATING AND PRICE DISTRIBUTION
# ============================================================

print("\n[8] RATING AND PRICE DISTRIBUTION")

rating_distribution = (
    df["Aggregate rating"]
    .value_counts(dropna=False)
    .sort_index()
    .reset_index()
)

rating_distribution.columns = [
    "Aggregate_Rating",
    "Restaurant_Count"
]

price_distribution = (
    df["Price range"]
    .value_counts(dropna=False)
    .sort_index()
    .reset_index()
)

price_distribution.columns = [
    "Price_Range",
    "Restaurant_Count"
]

rated_restaurants = int(
    (df["Aggregate rating"] > 0).sum()
)

unrated_restaurants = int(
    (df["Aggregate rating"] == 0).sum()
)

print(f"Rated restaurants: {rated_restaurants}")
print(f"Unrated restaurants: {unrated_restaurants}")

print("\nPrice range distribution:")
print(price_distribution.to_string(index=False))


# ============================================================
# 9. CUISINE COVERAGE
# ============================================================

print("\n[9] CUISINE COVERAGE")

missing_cuisines = int(
    df["Cuisines"].isna().sum()
)

unique_cuisine_combinations = int(
    df["Cuisines"].nunique()
)

print(f"Missing cuisine records: {missing_cuisines}")
print(
    "Unique cuisine combinations:",
    unique_cuisine_combinations
)

cuisine_data = df[
    df["Cuisines"].notna()
].copy()

cuisine_data["Cuisine"] = (
    cuisine_data["Cuisines"]
    .astype(str)
    .str.split(",")
)

cuisine_data = cuisine_data.explode("Cuisine")

cuisine_data["Cuisine"] = (
    cuisine_data["Cuisine"]
    .str.strip()
)

cuisine_data = cuisine_data[
    cuisine_data["Cuisine"] != ""
]

cuisine_distribution = (
    cuisine_data.groupby("Cuisine")
    .agg(
        Restaurant_Count=("Restaurant ID", "count"),
        Average_Rating=("Aggregate rating", "mean")
    )
    .reset_index()
    .sort_values(
        "Restaurant_Count",
        ascending=False
    )
)

cuisine_distribution["Average_Rating"] = (
    cuisine_distribution["Average_Rating"].round(2)
)

print("\nTop 10 cuisines:")

print(
    cuisine_distribution.head(10).to_string(index=False)
)


# ============================================================
# 10. COORDINATE DUPLICATE ANALYSIS
# ============================================================

print("\n[10] COORDINATE DUPLICATE ANALYSIS")

valid_location_data = df.loc[
    valid_coordinates
].copy()

duplicate_coordinate_rows = int(
    valid_location_data.duplicated(
        subset=["Longitude", "Latitude"],
        keep=False
    ).sum()
)

unique_coordinate_pairs = int(
    valid_location_data[
        ["Longitude", "Latitude"]
    ].drop_duplicates().shape[0]
)

print(
    f"Rows sharing coordinate pairs: {duplicate_coordinate_rows}"
)

print(
    f"Unique coordinate pairs: {unique_coordinate_pairs}"
)

# Repeated coordinates are not automatically errors.
# Multiple restaurants can legitimately share one building or location.


# ============================================================
# 11. COUNTRY-WISE LOCATION SUMMARY
# ============================================================

print("\n[11] COUNTRY-WISE LOCATION SUMMARY")

country_summary = (
    df.groupby("Country Code", dropna=False)
    .agg(
        Restaurant_Count=("Restaurant ID", "count"),
        City_Count=("City", "nunique"),
        Average_Rating=("Aggregate rating", "mean"),
        Average_Price_Range=("Price range", "mean")
    )
    .reset_index()
    .sort_values(
        "Restaurant_Count",
        ascending=False
    )
)

country_summary["Average_Rating"] = (
    country_summary["Average_Rating"].round(2)
)

country_summary["Average_Price_Range"] = (
    country_summary["Average_Price_Range"].round(2)
)

print(country_summary.to_string(index=False))


# ============================================================
# 12. SAVE AUDIT REPORTS
# ============================================================

print("\n[12] SAVING AUDIT REPORTS")

city_distribution.to_csv(
    REPORT_DIR / "location_city_distribution.csv",
    index=False
)

locality_distribution.to_csv(
    REPORT_DIR / "location_locality_distribution.csv",
    index=False
)

rating_distribution.to_csv(
    REPORT_DIR / "location_rating_distribution.csv",
    index=False
)

price_distribution.to_csv(
    REPORT_DIR / "location_price_distribution.csv",
    index=False
)

cuisine_distribution.to_csv(
    REPORT_DIR / "location_cuisine_distribution.csv",
    index=False
)

country_summary.to_csv(
    REPORT_DIR / "location_country_summary.csv",
    index=False
)


# ============================================================
# 13. GENERATE DATASET AUDIT SUMMARY
# ============================================================

audit_summary = [
    ("Total Restaurants", total_rows),
    ("Total Columns", total_columns),
    ("Duplicate Rows", duplicate_rows),
    ("Duplicate Restaurant IDs", duplicate_ids),
    ("Unique Restaurants", unique_restaurants),
    ("Unique Countries", unique_countries),
    ("Unique Cities", unique_cities),
    ("Unique Localities", unique_localities),
    ("Missing Cities", missing_cities),
    ("Missing Localities", missing_localities),
    ("Missing Longitude", missing_longitude),
    ("Missing Latitude", missing_latitude),
    ("Missing Both Coordinates", missing_both_coordinates),
    ("Missing Any Coordinate", missing_any_coordinate),
    ("Invalid Longitude", invalid_longitude),
    ("Invalid Latitude", invalid_latitude),
    ("Valid Coordinate Pairs", valid_coordinate_count),
    ("Invalid or Incomplete Coordinate Pairs", invalid_coordinate_count),
    ("Rows Sharing Coordinate Pairs", duplicate_coordinate_rows),
    ("Unique Coordinate Pairs", unique_coordinate_pairs),
    ("Rated Restaurants", rated_restaurants),
    ("Unrated Restaurants", unrated_restaurants),
    ("Missing Cuisines", missing_cuisines),
    ("Unique Cuisine Combinations", unique_cuisine_combinations)
]

audit_df = pd.DataFrame(
    audit_summary,
    columns=["Metric", "Value"]
)

audit_df.to_csv(
    REPORT_DIR / "location_dataset_audit_summary.csv",
    index=False
)

summary_file = (
    REPORT_DIR / "location_dataset_audit_report.txt"
)

with open(summary_file, "w", encoding="utf-8") as file:

    file.write(
        "DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS\n"
    )

    file.write(
        "TASK 04: LOCATION-BASED ANALYSIS\n"
    )

    file.write(
        "PROGRAM 01: LOCATION DATASET AUDIT\n\n"
    )

    file.write("=" * 65 + "\n")
    file.write("DATASET AUDIT SUMMARY\n")
    file.write("=" * 65 + "\n\n")

    for metric, value in audit_summary:
        file.write(f"{metric}: {value}\n")

    file.write("\n")
    file.write("=" * 65 + "\n")
    file.write("AUDIT COMPLETED\n")
    file.write("=" * 65 + "\n")


# ============================================================
# 14. FINAL VALIDATION
# ============================================================

print("\n[14] FINAL VALIDATION")

expected_reports = [
    "location_city_distribution.csv",
    "location_locality_distribution.csv",
    "location_rating_distribution.csv",
    "location_price_distribution.csv",
    "location_cuisine_distribution.csv",
    "location_country_summary.csv",
    "location_dataset_audit_summary.csv",
    "location_dataset_audit_report.txt"
]

all_reports_exist = all(
    (REPORT_DIR / name).exists()
    for name in expected_reports
)

checks = {
    "Dataset loaded": len(df) > 0,
    "Required columns available": len(missing_columns) == 0,
    "Coordinate audit completed": (
        valid_coordinate_count + invalid_coordinate_count
        == total_rows
    ),
    "City analysis generated": not city_distribution.empty,
    "Locality analysis generated": not locality_distribution.empty,
    "Cuisine analysis generated": not cuisine_distribution.empty,
    "Audit reports saved": all_reports_exist
}

for check, passed in checks.items():
    print(
        f"{'[PASS]' if passed else '[FAIL]'} {check}"
    )

if not all(checks.values()):
    raise RuntimeError(
        "One or more audit validations failed."
    )

print("\nAudit reports saved to:")
print(REPORT_DIR)

print("\n" + "=" * 75)
print("PROGRAM 01 EXECUTION COMPLETED SUCCESSFULLY")
print("=" * 75)
