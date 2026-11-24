from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ============================================================
# DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS
# TASK 02: RESTAURANT RECOMMENDATION SYSTEM
# PROGRAM 01: DATASET AUDIT
# ============================================================

print("=" * 75)
print("DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS")
print("TASK 02: RESTAURANT RECOMMENDATION SYSTEM")
print("PROGRAM 01 - DATASET AUDIT")
print("=" * 75)


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Dataset.csv"

REPORT_DIR = PROJECT_ROOT / "reports" / "task_2"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n" + "=" * 65)
print("2. LOAD DATASET")
print("=" * 65)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, encoding="utf-8")

print("\nDataset loaded successfully.")
print(f"Total rows: {df.shape[0]}")
print(f"Total columns: {df.shape[1]}")


# ============================================================
# 3. CHECK REQUIRED COLUMNS
# ============================================================

print("\n" + "=" * 65)
print("3. CHECK REQUIRED COLUMNS")
print("=" * 65)

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
        f"Required columns missing from dataset: {missing_columns}"
    )

print("\nAll required recommendation columns are available.")


# ============================================================
# 4. DATASET QUALITY CHECKS
# ============================================================

print("\n" + "=" * 65)
print("4. DATASET QUALITY CHECKS")
print("=" * 65)

missing_values = df.isnull().sum()

duplicate_rows = df.duplicated().sum()

duplicate_ids = df["Restaurant ID"].duplicated().sum()

print(f"\nDuplicate rows: {duplicate_rows}")
print(f"Duplicate restaurant IDs: {duplicate_ids}")

print("\nMissing values:")

print(missing_values[missing_values > 0])


# ============================================================
# 5. RECOMMENDATION FEATURE ANALYSIS
# ============================================================

print("\n" + "=" * 65)
print("5. RECOMMENDATION FEATURE ANALYSIS")
print("=" * 65)

feature_columns = [
    "City",
    "Locality",
    "Cuisines",
    "Price range",
    "Aggregate rating",
    "Votes",
]

feature_summary = []

for column in feature_columns:

    feature_summary.append({
        "Feature": column,
        "Data Type": str(df[column].dtype),
        "Missing Values": int(df[column].isnull().sum()),
        "Unique Values": int(df[column].nunique(dropna=True)),
    })

    print(
        f"{column}: "
        f"{df[column].nunique(dropna=True)} unique values, "
        f"{df[column].isnull().sum()} missing"
    )

feature_summary_df = pd.DataFrame(feature_summary)


# ============================================================
# 6. VALUE DISTRIBUTION ANALYSIS
# ============================================================

print("\n" + "=" * 65)
print("6. VALUE DISTRIBUTION ANALYSIS")
print("=" * 65)


# ------------------------------------------------------------
# 6.1 PRICE RANGE DISTRIBUTION
# ------------------------------------------------------------

print("\n--- PRICE RANGE DISTRIBUTION ---")

price_distribution = (
    df["Price range"]
    .value_counts(dropna=False)
    .sort_index()
)

print(price_distribution)


# ------------------------------------------------------------
# 6.2 CITY DISTRIBUTION
# ------------------------------------------------------------

print("\n--- TOP 15 CITIES ---")

city_distribution = df["City"].value_counts().head(15)

print(city_distribution)


# ------------------------------------------------------------
# 6.3 CUISINE DISTRIBUTION
# ------------------------------------------------------------

print("\n--- TOP 15 CUISINE ENTRIES ---")

cuisine_distribution = df["Cuisines"].value_counts().head(15)

print(cuisine_distribution)


# ------------------------------------------------------------
# 6.4 RATING DISTRIBUTION
# ------------------------------------------------------------

print("\n--- RATING DISTRIBUTION ---")

rating_distribution = (
    df["Aggregate rating"]
    .value_counts(dropna=False)
    .sort_index()
)

print(rating_distribution)


# ============================================================
# 7. RECOMMENDATION DATASET SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("7. RECOMMENDATION DATASET SUMMARY")
print("=" * 65)

summary = pd.DataFrame([
    {
        "Metric": "Total Restaurants",
        "Value": len(df),
    },
    {
        "Metric": "Unique Restaurant IDs",
        "Value": df["Restaurant ID"].nunique(),
    },
    {
        "Metric": "Duplicate Rows",
        "Value": duplicate_rows,
    },
    {
        "Metric": "Duplicate Restaurant IDs",
        "Value": duplicate_ids,
    },
    {
        "Metric": "Unique Cities",
        "Value": df["City"].nunique(),
    },
    {
        "Metric": "Unique Localities",
        "Value": df["Locality"].nunique(),
    },
    {
        "Metric": "Unique Cuisine Entries",
        "Value": df["Cuisines"].nunique(),
    },
    {
        "Metric": "Missing Cuisine Values",
        "Value": df["Cuisines"].isnull().sum(),
    },
    {
        "Metric": "Unrated Restaurants",
        "Value": (df["Aggregate rating"] == 0).sum(),
    },
])


# ============================================================
# 8. SAVE AUDIT REPORTS
# ============================================================

print("\n" + "=" * 65)
print("8. SAVE AUDIT REPORTS")
print("=" * 65)

summary_path = REPORT_DIR / "recommendation_dataset_summary.csv"

features_path = REPORT_DIR / "recommendation_feature_audit_summary.csv"

missing_path = REPORT_DIR / "recommendation_missing_values.csv"

summary.to_csv(summary_path, index=False)

feature_summary_df.to_csv(features_path, index=False)

missing_values.rename("Missing Values").to_csv(missing_path)

print("\nAudit reports saved successfully.")

print(f"\n1. {summary_path}")
print(f"2. {features_path}")
print(f"3. {missing_path}")


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("TASK 02 - RESTAURANT RECOMMENDATION SYSTEM")
print("PROGRAM 01 - DATASET AUDIT COMPLETED")
print("=" * 75)

print(f"\nTotal restaurants analyzed: {len(df)}")
print(f"Unique restaurant IDs: {df['Restaurant ID'].nunique()}")
print(f"Unique cities: {df['City'].nunique()}")
print(f"Unique cuisine entries: {df['Cuisines'].nunique()}")
print(f"Duplicate rows: {duplicate_rows}")
print(f"Duplicate restaurant IDs: {duplicate_ids}")

print("\nGenerated reports:")
print("1. recommendation_dataset_summary.csv")
print("2. recommendation_feature_audit_summary.csv")
print("3. recommendation_missing_values.csv")

print("\nPROGRAM 01 EXECUTION SUCCESSFUL.")

print("=" * 75)
