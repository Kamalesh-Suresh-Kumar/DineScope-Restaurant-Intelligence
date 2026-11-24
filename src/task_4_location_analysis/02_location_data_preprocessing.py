from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Dataset.csv"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "task_4_location_analysis"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "task_4"
)

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS
# TASK 04: LOCATION-BASED ANALYSIS
# PROGRAM 02: LOCATION DATA PREPROCESSING
# ============================================================

print("=" * 75)
print("DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS")
print("TASK 04: LOCATION-BASED ANALYSIS")
print("PROGRAM 02 - LOCATION DATA PREPROCESSING")
print("=" * 75)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\n[1] LOADING DATASET")

if not RAW_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {RAW_FILE}"
    )

df = pd.read_csv(
    RAW_FILE,
    encoding="utf-8"
)

original_rows = len(df)

print(f"Dataset loaded: {df.shape}")
print(f"Original records: {original_rows}")


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
# 3. PRESERVE ORIGINAL ROW IDENTIFICATION
# ============================================================

print("\n[3] PRESERVING SOURCE ROW IDENTIFICATION")

df.insert(
    0,
    "Source Row Index",
    np.arange(len(df))
)

print("Source Row Index created.")
print("Original row order preserved.")


# ============================================================
# 4. CLEAN TEXT COLUMNS
# ============================================================

print("\n[4] CLEANING TEXT COLUMNS")

text_columns = [
    "Restaurant Name",
    "City",
    "Address",
    "Locality",
    "Locality Verbose",
    "Cuisines",
    "Currency"
]


for column in text_columns:
    df[column] = (
        df[column]
        .astype("string")
        .str.strip()
    )

    df[column] = df[column].mask(
        df[column].str.fullmatch(r"\s*", na=False),
        pd.NA
    )


print("Text columns cleaned.")


# ============================================================
# 5. CORRECT KNOWN CITY ENCODING ISSUES
# ============================================================

print("\n[5] CORRECTING KNOWN CITY ENCODING ISSUES")

city_corrections = {
    "Bras�_lia": "Brasília",
    "S��o Paulo": "São Paulo",
    "��stanbul": "İstanbul"
}

city_correction_counts = {}

for old_name, new_name in city_corrections.items():

    count = int(
        (df["City"] == old_name).sum()
    )

    if count > 0:
        df["City"] = df["City"].replace(
            old_name,
            new_name
        )

        city_correction_counts[old_name] = count

        print(
            f"{old_name} -> {new_name}: {count} records"
        )

if not city_correction_counts:
    print("No known city corrections were required.")


# ============================================================
# 6. HANDLE MISSING LOCATION INFORMATION
# ============================================================

print("\n[6] HANDLING MISSING LOCATION INFORMATION")

missing_city_before = int(
    df["City"].isna().sum()
)

missing_locality_before = int(
    df["Locality"].isna().sum()
)

missing_address_before = int(
    df["Address"].isna().sum()
)

df["City"] = df["City"].fillna("Unknown")
df["Locality"] = df["Locality"].fillna("Unknown")
df["Address"] = df["Address"].fillna("Unknown")

df["Locality Verbose"] = (
    df["Locality Verbose"].fillna("Unknown")
)

print(f"Missing cities before: {missing_city_before}")
print(f"Missing localities before: {missing_locality_before}")
print(f"Missing addresses before: {missing_address_before}")

print("Missing location text values handled.")


# ============================================================
# 7. CLEAN GEOGRAPHICAL COORDINATES
# ============================================================

print("\n[7] CLEANING GEOGRAPHICAL COORDINATES")

df["Longitude"] = pd.to_numeric(
    df["Longitude"],
    errors="coerce"
)

df["Latitude"] = pd.to_numeric(
    df["Latitude"],
    errors="coerce"
)

missing_coordinates = int(
    (
        df["Longitude"].isna()
        | df["Latitude"].isna()
    ).sum()
)

valid_longitude = df["Longitude"].between(
    -180,
    180
)

valid_latitude = df["Latitude"].between(
    -90,
    90
)

df["Coordinate Valid"] = (
    valid_longitude
    & valid_latitude
    & df["Longitude"].notna()
    & df["Latitude"].notna()
)

valid_coordinates = int(
    df["Coordinate Valid"].sum()
)

invalid_coordinates = int(
    (~df["Coordinate Valid"]).sum()
)

print(f"Missing coordinate pairs: {missing_coordinates}")
print(f"Valid coordinate pairs: {valid_coordinates}")
print(f"Invalid or incomplete coordinate pairs: {invalid_coordinates}")


# ============================================================
# 8. IDENTIFY REPEATED COORDINATE PAIRS
# ============================================================

print("\n[8] IDENTIFYING REPEATED COORDINATE PAIRS")

df["Duplicate Coordinate"] = False

valid_mask = df["Coordinate Valid"]

df.loc[
    valid_mask,
    "Duplicate Coordinate"
] = df.loc[
    valid_mask
].duplicated(
    subset=["Longitude", "Latitude"],
    keep=False
)

duplicate_coordinate_rows = int(
    df["Duplicate Coordinate"].sum()
)

unique_coordinate_pairs = int(
    df.loc[
        valid_mask,
        ["Longitude", "Latitude"]
    ].drop_duplicates().shape[0]
)

print(
    f"Rows sharing coordinate pairs: {duplicate_coordinate_rows}"
)

print(
    f"Unique coordinate pairs: {unique_coordinate_pairs}"
)

print("Repeated coordinates retained.")


# ============================================================
# 9. CLEAN NUMERICAL FEATURES
# ============================================================

print("\n[9] CLEANING NUMERICAL FEATURES")

numeric_columns = [
    "Country Code",
    "Average Cost for two",
    "Price range",
    "Aggregate rating",
    "Votes"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

print("Numerical columns converted.")


# ============================================================
# 10. HANDLE UNRATED RESTAURANTS
# ============================================================

print("\n[10] HANDLING UNRATED RESTAURANTS")

unrated_count = int(
    (df["Aggregate rating"] == 0).sum()
)

df["Aggregate rating"] = (
    df["Aggregate rating"].replace(0, np.nan)
)

rated_count = int(
    df["Aggregate rating"].notna().sum()
)

print(f"Unrated restaurants converted to NaN: {unrated_count}")
print(f"Restaurants with valid ratings: {rated_count}")

print("Original rating information preserved through missing values.")


# ============================================================
# 11. VALIDATE PRICE, COST AND VOTES
# ============================================================

print("\n[11] VALIDATING PRICE, COST AND VOTES")

invalid_price_range = int(
    (
        ~df["Price range"].isin([1, 2, 3, 4])
        & df["Price range"].notna()
    ).sum()
)

negative_cost = int(
    (
        (df["Average Cost for two"] < 0)
        & df["Average Cost for two"].notna()
    ).sum()
)

negative_votes = int(
    (
        (df["Votes"] < 0)
        & df["Votes"].notna()
    ).sum()
)

print(f"Invalid price ranges: {invalid_price_range}")
print(f"Negative average costs: {negative_cost}")
print(f"Negative vote counts: {negative_votes}")

# Invalid numerical values are retained for audit visibility.
# They are not silently replaced or removed.


# ============================================================
# 12. HANDLE CUISINE INFORMATION
# ============================================================

print("\n[12] HANDLING CUISINE INFORMATION")

missing_cuisines_before = int(
    df["Cuisines"].isna().sum()
)

df["Cuisines"] = df["Cuisines"].fillna(
    "Unknown"
)

df["Cuisine List"] = df["Cuisines"].apply(
    lambda value: [
        cuisine.strip()
        for cuisine in str(value).split(",")
        if cuisine.strip()
        and cuisine.strip().casefold() != "unknown"
    ]
)

restaurants_without_known_cuisine = int(
    df["Cuisine List"].apply(len).eq(0).sum()
)

print(
    f"Missing cuisine records before: {missing_cuisines_before}"
)

print(
    "Restaurants without known cuisine:",
    restaurants_without_known_cuisine
)


# ============================================================
# 13. CREATE LOCATION IDENTIFIERS
# ============================================================

print("\n[13] CREATING LOCATION IDENTIFIERS")

df["Location Key"] = (
    df["Country Code"].astype("string")
    + "_"
    + df["City"].astype("string").str.casefold()
    + "_"
    + df["Locality"].astype("string").str.casefold()
)

df["City Key"] = (
    df["Country Code"].astype("string")
    + "_"
    + df["City"].astype("string").str.casefold()
)

print("City Key created.")
print("Location Key created.")


# ============================================================
# 14. FINAL DATA VALIDATION
# ============================================================

print("\n[14] FINAL DATA VALIDATION")

duplicate_ids = int(
    df["Restaurant ID"].duplicated().sum()
)

missing_ids = int(
    df["Restaurant ID"].isna().sum()
)

final_rows = len(df)

checks = {
    "Original row count preserved": final_rows == original_rows,
    "Restaurant IDs unique": duplicate_ids == 0,
    "Restaurant IDs available": missing_ids == 0,
    "Coordinate validity flag created": "Coordinate Valid" in df.columns,
    "Duplicate coordinate flag created": "Duplicate Coordinate" in df.columns,
    "Location identifiers created": "Location Key" in df.columns,
    "Missing cuisine values handled": df["Cuisines"].notna().all(),
    "Rating zero converted to missing": (
        not (df["Aggregate rating"] == 0).any()
    ),
    "Source row index preserved": (
        df["Source Row Index"].is_unique
        and df["Source Row Index"].min() == 0
        and df["Source Row Index"].max() == original_rows - 1
    )
}

for check, passed in checks.items():
    print(
        f"{'[PASS]' if passed else '[FAIL]'} {check}"
    )

if not all(checks.values()):
    raise RuntimeError(
        "One or more preprocessing validations failed."
    )


# ============================================================
# 15. SAVE PROCESSED DATASET
# ============================================================

print("\n[15] SAVING PROCESSED DATASET")

processed_file = (
    PROCESSED_DIR
    / "location_analysis_data.csv"
)

df.to_csv(
    processed_file,
    index=False
)

print(f"Processed dataset saved: {processed_file}")


# ============================================================
# 16. SAVE PREPROCESSING SUMMARY
# ============================================================

print("\n[16] SAVING PREPROCESSING SUMMARY")

summary = [
    ("Original Records", original_rows),
    ("Final Records", final_rows),
    ("Rows Removed", original_rows - final_rows),
    ("Unique Restaurants", df["Restaurant ID"].nunique()),
    ("Unique Cities", df["City"].nunique()),
    ("Unique Localities", df["Locality"].nunique()),
    ("Missing City Values Before Cleaning", missing_city_before),
    ("Missing Locality Values Before Cleaning", missing_locality_before),
    ("Missing Address Values Before Cleaning", missing_address_before),
    ("Valid Coordinates", valid_coordinates),
    ("Invalid or Incomplete Coordinates", invalid_coordinates),
    ("Rows Sharing Coordinate Pairs", duplicate_coordinate_rows),
    ("Unique Coordinate Pairs", unique_coordinate_pairs),
    ("Unrated Restaurants", unrated_count),
    ("Rated Restaurants", rated_count),
    ("Missing Cuisine Values Before Cleaning", missing_cuisines_before),
    ("Restaurants Without Known Cuisine", restaurants_without_known_cuisine),
    ("Invalid Price Ranges", invalid_price_range),
    ("Negative Average Costs", negative_cost),
    ("Negative Votes", negative_votes)
]

summary_df = pd.DataFrame(
    summary,
    columns=["Metric", "Value"]
)

summary_df.to_csv(
    REPORT_DIR / "location_preprocessing_summary.csv",
    index=False
)


# ============================================================
# 17. SAVE DATA QUALITY REPORT
# ============================================================


# Calculate unique values safely, including list-valued columns.

def calculate_unique_count(series):

    # Remove missing values first.
    values = series.dropna()

    # Convert each value to its string representation.
    # This safely handles lists, tuples, NumPy arrays and dictionaries.
    values = values.map(repr)

    return int(values.nunique())


quality_report = pd.DataFrame([
    {
        "Column": column,
        "Missing_Values": int(df[column].isna().sum()),
        "Data_Type": str(df[column].dtype),
        "Unique_Values": calculate_unique_count(df[column])
    }
    for column in df.columns
])


quality_report.to_csv(
    REPORT_DIR / "location_preprocessing_data_quality.csv",
    index=False
)


# ============================================================
# 18. SAVE FINAL TEXT REPORT
# ============================================================

report_file = (
    REPORT_DIR
    / "location_preprocessing_report.txt"
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
        "PROGRAM 02: LOCATION DATA PREPROCESSING\n\n"
    )

    file.write("=" * 65 + "\n")
    file.write("PREPROCESSING SUMMARY\n")
    file.write("=" * 65 + "\n\n")

    for metric, value in summary:
        file.write(f"{metric}: {value}\n")

    file.write("\n")
    file.write("CITY CORRECTIONS\n")
    file.write("-" * 65 + "\n")

    for old_name, count in city_correction_counts.items():
        file.write(
            f"{old_name}: {count} records corrected\n"
        )

    if not city_correction_counts:
        file.write("No known city corrections required.\n")

    file.write("\n")
    file.write("=" * 65 + "\n")
    file.write("PREPROCESSING COMPLETED\n")
    file.write("=" * 65 + "\n")


# ============================================================
# 19. FINAL OUTPUT VALIDATION
# ============================================================

print("\n[19] FINAL OUTPUT VALIDATION")

expected_files = [
    PROCESSED_DIR / "location_analysis_data.csv",
    REPORT_DIR / "location_preprocessing_summary.csv",
    REPORT_DIR / "location_preprocessing_data_quality.csv",
    REPORT_DIR / "location_preprocessing_report.txt"
]

for output_file in expected_files:
    exists = output_file.exists()

    print(
        f"{'[PASS]' if exists else '[FAIL]'} {output_file.name}"
    )

if not all(path.exists() for path in expected_files):
    raise RuntimeError(
        "One or more output files were not saved."
    )

print("\nProcessed dataset shape:", df.shape)

print("\nProcessed dataset saved to:")
print(PROCESSED_DIR)

print("\nReports saved to:")
print(REPORT_DIR)

print("\n" + "=" * 75)
print("PROGRAM 02 EXECUTION COMPLETED SUCCESSFULLY")
print("=" * 75)
