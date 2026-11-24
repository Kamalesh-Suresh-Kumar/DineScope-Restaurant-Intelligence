from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ============================================================
# DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS
# TASK 02: RESTAURANT RECOMMENDATION SYSTEM
# PROGRAM 02: RECOMMENDATION DATA PREPROCESSING
# ============================================================

print("=" * 75)
print("DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS")
print("TASK 02: RESTAURANT RECOMMENDATION SYSTEM")
print("PROGRAM 02 - RECOMMENDATION DATA PREPROCESSING")
print("=" * 75)


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Dataset.csv"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "task_2_restaurant_recommendation"
REPORT_DIR = PROJECT_ROOT / "reports" / "task_2"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = PROCESSED_DIR / "recommendation_data.csv"

SUMMARY_PATH = REPORT_DIR / "recommendation_preprocessing_summary.csv"

CUISINE_PATH = REPORT_DIR / "recommendation_cuisine_summary.csv"


# ============================================================
# 2. LOAD AND VALIDATE DATASET
# ============================================================

print("\n" + "=" * 65)
print("2. LOAD AND VALIDATE DATASET")
print("=" * 65)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, encoding="utf-8")

original_rows = len(df)

required_columns = [
    "Restaurant ID",
    "Restaurant Name",
    "City",
    "Locality",
    "Cuisines",
    "Price range",
    "Aggregate rating",
    "Votes",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns missing: {missing_columns}"
    )

if df["Restaurant ID"].isnull().any():
    raise ValueError("Restaurant ID contains missing values.")

if df["Restaurant ID"].duplicated().any():
    raise ValueError("Duplicate Restaurant IDs detected.")

print(f"Original records: {original_rows}")
print("Required columns validated successfully.")


# ============================================================
# 3. CLEAN TEXT FEATURES
# ============================================================

print("\n" + "=" * 65)
print("3. CLEAN TEXT FEATURES")
print("=" * 65)

text_columns = [
    "Restaurant Name",
    "City",
    "Locality",
    "Cuisines",
]

for column in text_columns:
    df[column] = df[column].astype("string").str.strip()

# Convert empty strings to missing values.
for column in text_columns:
    df[column] = df[column].replace("", pd.NA)

# Correct known corrupted city names.
city_corrections = {
    "Bras�_lia": "Brasília",
    "S��o Paulo": "São Paulo",
    "��stanbul": "İstanbul",
}

df["City"] = df["City"].replace(city_corrections)

print("Text cleaning completed.")

for old_name, new_name in city_corrections.items():
    count = (df["City"] == new_name).sum()
    print(f"{old_name} -> {new_name}: {count} records")


# ============================================================
# 4. HANDLE MISSING CUISINE VALUES
# ============================================================

print("\n" + "=" * 65)
print("4. HANDLE MISSING CUISINE VALUES")
print("=" * 65)

missing_cuisines_before = df["Cuisines"].isnull().sum()

# Retain restaurants without cuisine information.
# They will receive no cuisine match during recommendation.
df["Cuisines"] = df["Cuisines"].fillna("Unknown")

missing_cuisines_after = df["Cuisines"].isnull().sum()

print(f"Missing cuisines before: {missing_cuisines_before}")
print(f"Missing cuisines after: {missing_cuisines_after}")
print("Restaurants with unknown cuisines have been retained.")


# ============================================================
# 5. NORMALIZE CUISINE FEATURES
# ============================================================

print("\n" + "=" * 65)
print("5. NORMALIZE CUISINE FEATURES")
print("=" * 65)

def normalize_cuisines(value):
    if pd.isna(value):
        return "Unknown"

    cuisines = [
        cuisine.strip().casefold()
        for cuisine in str(value).split(",")
        if cuisine.strip()
    ]

    # Remove duplicate labels while preserving their order.
    cuisines = list(dict.fromkeys(cuisines))

    if not cuisines:
        return "Unknown"

    return ", ".join(cuisines)


df["Cuisines"] = df["Cuisines"].apply(normalize_cuisines)

# Create a separate list representation for later
# multi-label recommendation processing.
df["Cuisine List"] = df["Cuisines"].apply(
    lambda value: (
        []
        if value == "Unknown"
        else value.split(", ")
    )
)

unique_cuisines = sorted({
    cuisine
    for cuisine_list in df["Cuisine List"]
    for cuisine in cuisine_list
})

print(f"Unique individual cuisines: {len(unique_cuisines)}")
print("Cuisine normalization completed.")


# ============================================================
# 6. HANDLE RESTAURANT RATINGS
# ============================================================

print("\n" + "=" * 65)
print("6. HANDLE RESTAURANT RATINGS")
print("=" * 65)

unrated_before = (df["Aggregate rating"] == 0).sum()

# Zero represents Not Rated in this dataset.
df["Aggregate rating"] = df["Aggregate rating"].replace(
    0,
    np.nan
)

# Validate remaining non-missing ratings.
invalid_ratings = (
    df["Aggregate rating"].notna()
    & ~df["Aggregate rating"].between(1, 5)
)

if invalid_ratings.any():
    raise ValueError("Invalid restaurant ratings detected.")

unrated_after = df["Aggregate rating"].isnull().sum()

print(f"Unrated restaurants before: {unrated_before}")
print(f"Missing ratings after conversion: {unrated_after}")
print("Rating preprocessing completed.")


# ============================================================
# 7. VALIDATE PRICE RANGE AND VOTES
# ============================================================

print("\n" + "=" * 65)
print("7. VALIDATE PRICE RANGE AND VOTES")
print("=" * 65)

df["Price range"] = pd.to_numeric(
    df["Price range"],
    errors="coerce"
)

df["Votes"] = pd.to_numeric(
    df["Votes"],
    errors="coerce"
)

invalid_price = (
    df["Price range"].notna()
    & ~df["Price range"].isin([1, 2, 3, 4])
)

if invalid_price.any():
    raise ValueError("Invalid price range values detected.")

invalid_votes = (
    df["Votes"].notna()
    & (df["Votes"] < 0)
)

if invalid_votes.any():
    raise ValueError("Negative vote values detected.")

print("Price range validation passed.")
print("Votes validation passed.")


# ============================================================
# 8. FINAL DATASET VALIDATION
# ============================================================

print("\n" + "=" * 65)
print("8. FINAL DATASET VALIDATION")
print("=" * 65)

if len(df) != original_rows:
    raise ValueError(
        "Restaurant records were unexpectedly added or removed."
    )

if df["Restaurant ID"].duplicated().any():
    raise ValueError("Duplicate restaurant IDs detected.")

if df["Restaurant ID"].isnull().any():
    raise ValueError("Missing restaurant IDs detected.")

if df["Cuisines"].isnull().any():
    raise ValueError("Cuisine preprocessing failed.")

print(f"Final records: {len(df)}")
print(f"Final columns: {len(df.columns)}")
print("Restaurant ID validation passed.")
print("Record preservation validation passed.")


# ============================================================
# 9. GENERATE PREPROCESSING REPORTS
# ============================================================

print("\n" + "=" * 65)
print("9. GENERATE PREPROCESSING REPORTS")
print("=" * 65)

summary = pd.DataFrame([
    {
        "Metric": "Original Records",
        "Value": original_rows,
    },
    {
        "Metric": "Final Records",
        "Value": len(df),
    },
    {
        "Metric": "Records Removed",
        "Value": original_rows - len(df),
    },
    {
        "Metric": "Missing Cuisines Before",
        "Value": missing_cuisines_before,
    },
    {
        "Metric": "Missing Cuisines After",
        "Value": missing_cuisines_after,
    },
    {
        "Metric": "Unrated Restaurants",
        "Value": unrated_before,
    },
    {
        "Metric": "Missing Ratings After Conversion",
        "Value": unrated_after,
    },
    {
        "Metric": "Unique Individual Cuisines",
        "Value": len(unique_cuisines),
    },
    {
        "Metric": "Unique Cities",
        "Value": df["City"].nunique(),
    },
    {
        "Metric": "Duplicate Restaurant IDs",
        "Value": df["Restaurant ID"].duplicated().sum(),
    },
])

cuisine_summary = (
    df.explode("Cuisine List")
    .loc[lambda data: data["Cuisine List"] != ""]
    ["Cuisine List"]
    .value_counts()
    .rename_axis("Cuisine")
    .reset_index(name="Restaurant Count")
)

summary.to_csv(SUMMARY_PATH, index=False)

cuisine_summary.to_csv(CUISINE_PATH, index=False)

df.to_csv(OUTPUT_PATH, index=False)

print("Preprocessing reports saved successfully.")

print(f"\n1. {OUTPUT_PATH}")
print(f"2. {SUMMARY_PATH}")
print(f"3. {CUISINE_PATH}")


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("TASK 02 - RESTAURANT RECOMMENDATION SYSTEM")
print("PROGRAM 02 - RECOMMENDATION DATA PREPROCESSING COMPLETED")
print("=" * 75)

print(f"\nOriginal records: {original_rows}")
print(f"Processed records: {len(df)}")
print(f"Records removed: {original_rows - len(df)}")
print(f"Individual cuisines: {len(unique_cuisines)}")
print(f"Unrated restaurants retained: {unrated_after}")

print("\nGenerated files:")
print("1. recommendation_data.csv")
print("2. recommendation_preprocessing_summary.csv")
print("3. recommendation_cuisine_summary.csv")

print("\nPROGRAM 02 EXECUTION SUCCESSFUL.")

print("=" * 75)
