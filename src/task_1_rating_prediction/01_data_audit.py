import os
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ============================================================
# DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS
# TASK 01: RESTAURANT RATING PREDICTION
# PROGRAM 01: DATASET AUDIT
# ============================================================

print("=" * 75)
print("DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS")
print("TASK 01: RESTAURANT RATING PREDICTION")
print("PROGRAM 01 - DATASET AUDIT")
print("=" * 75)

# ==========================================================
# 1. CONFIGURATION
# ==========================================================

DATA_PATH = str(PROJECT_ROOT / "data" / "raw" / "Dataset.csv")
REPORT_PATH = str(PROJECT_ROOT / "reports" / "task_1")

# Create the reports folder if it does not exist
os.makedirs(REPORT_PATH, exist_ok=True)

# ==========================================================
# 2. LOAD DATASET
# ==========================================================

df = pd.read_csv(DATA_PATH)

# ==========================================================
# 3. DATASET DIMENSIONS
# ==========================================================

print("\n1. DATASET SHAPE")

rows, columns = df.shape

print("Number of rows:", rows)
print("Number of columns:", columns)

# ==========================================================
# 4. COLUMN INFORMATION
# ==========================================================

print("\n2. COLUMN INFORMATION")

df.info()

# ==========================================================
# 5. MISSING VALUES
# ==========================================================

print("\n3. MISSING VALUES")

missing_values = df.isnull().sum()

print(missing_values)

print("\nTotal missing values:", missing_values.sum())

# ==========================================================
# 6. DUPLICATE RECORDS
# ==========================================================

print("\n4. DUPLICATE RECORDS")

duplicate_count = df.duplicated().sum()

print("Duplicate records:", duplicate_count)

# ==========================================================
# 7. STATISTICAL SUMMARY
# ==========================================================

print("\n5. STATISTICAL SUMMARY")

print(df.describe())

# ==========================================================
# 8. RATING DISTRIBUTION
# ==========================================================

print("\n6. AGGREGATE RATING DISTRIBUTION")

rating_counts = df["Aggregate rating"].value_counts().sort_index()

print(rating_counts)

# ==========================================================
# 9. UNRATED RESTAURANTS
# ==========================================================

print("\n7. UNRATED RESTAURANTS")

unrated_count = (df["Aggregate rating"] == 0).sum()

print("Number of unrated restaurants:", unrated_count)

# ==========================================================
# 10. CUISINE INFORMATION
# ==========================================================

print("\n8. CUISINE INFORMATION")

print("Missing cuisines:", df["Cuisines"].isnull().sum())

print("Unique cuisine combinations:", df["Cuisines"].nunique())

# ==========================================================
# 11. CITY INFORMATION
# ==========================================================

print("\n9. CITY INFORMATION")

print("Unique cities:", df["City"].nunique())

# ==========================================================
# 12. PRICE RANGE DISTRIBUTION
# ==========================================================

print("\n10. PRICE RANGE DISTRIBUTION")

print(df["Price range"].value_counts().sort_index())

# ==========================================================
# 13. SAVE AUDIT REPORT
# ==========================================================

print("\n11. SAVING AUDIT REPORT")

report_path = os.path.join(
    REPORT_PATH,
    "dataset_summary.csv"
)

df.describe(include="all").to_csv(report_path)

print("Audit report saved:", report_path)

# ==========================================================
# 14. FINAL STATUS
# ==========================================================

print("\n" + "=" * 60)
print("DATASET AUDIT COMPLETED SUCCESSFULLY")
print("=" * 60)
