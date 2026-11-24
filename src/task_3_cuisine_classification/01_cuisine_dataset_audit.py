from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "Dataset.csv"
REPORT_DIR = PROJECT_ROOT / "reports" / "task_3"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures" / "task_3"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS
# TASK 03: CUISINE CLASSIFICATION
# PROGRAM 01: CUISINE DATASET AUDIT
# ============================================================

print("=" * 75)
print("DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS")
print("TASK 03: CUISINE CLASSIFICATION")
print("PROGRAM 01 - CUISINE DATASET AUDIT")
print("=" * 75)

# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\n" + "=" * 65)
print("1. LOAD RAW DATASET")
print("=" * 65)

assert RAW_FILE.exists(), f"Dataset not found: {RAW_FILE}"

data = pd.read_csv(RAW_FILE, encoding="utf-8")

print("Dataset loaded successfully.")
print("Dataset shape:", data.shape)
print("Total records:", len(data))
print("Total columns:", len(data.columns))

assert "Cuisines" in data.columns, "Cuisines column not found."

# ============================================================
# 2. INSPECT DATASET STRUCTURE
# ============================================================

print("\n" + "=" * 65)
print("2. DATASET STRUCTURE")
print("=" * 65)

print("\nColumn names:")
print(data.columns.tolist())

print("\nData types:")
print(data.dtypes)

print("\nFirst five records:")
print(data.head())

print("\nDataset information:")
data.info()

# ============================================================
# 3. ANALYSE MISSING VALUES
# ============================================================

print("\n" + "=" * 65)
print("3. MISSING VALUE ANALYSIS")
print("=" * 65)

missing_values = pd.DataFrame({
    "Missing Count": data.isnull().sum(),
    "Missing Percentage": (
        data.isnull().sum() / len(data) * 100
    )
})

missing_values = missing_values.sort_values(
    by="Missing Count",
    ascending=False
)

print(missing_values)

missing_values.to_csv(
    REPORT_DIR / "cuisine_missing_values.csv"
)

print("\nMissing value report saved.")

# ============================================================
# 4. ANALYSE DUPLICATES
# ============================================================

print("\n" + "=" * 65)
print("4. DUPLICATE ANALYSIS")
print("=" * 65)

duplicate_rows = data.duplicated().sum()

print("Duplicate records:", duplicate_rows)

if "Restaurant ID" in data.columns:
    duplicate_ids = data["Restaurant ID"].duplicated().sum()
    print("Duplicate restaurant IDs:", duplicate_ids)
else:
    duplicate_ids = None

# ============================================================
# 5. ANALYSE CUISINE AVAILABILITY
# ============================================================

print("\n" + "=" * 65)
print("5. CUISINE AVAILABILITY ANALYSIS")
print("=" * 65)

cuisine_data = data["Cuisines"]

missing_cuisine_count = cuisine_data.isna().sum()

print("Restaurants with missing cuisine information:",
      missing_cuisine_count)

available_cuisines = cuisine_data.dropna()

print("Restaurants with cuisine information:",
      len(available_cuisines))

# ============================================================
# 6. EXTRACT INDIVIDUAL CUISINE LABELS
# ============================================================

print("\n" + "=" * 65)
print("6. INDIVIDUAL CUISINE EXTRACTION")
print("=" * 65)

def extract_cuisines(value):

    if pd.isna(value):
        return []

    cuisines = []

    for cuisine in str(value).split(","):

        cuisine = cuisine.strip().casefold()

        if cuisine and cuisine != "unknown":
            cuisines.append(cuisine)

    return list(dict.fromkeys(cuisines))


data["Cuisine Labels"] = data["Cuisines"].apply(
    extract_cuisines
)

data["Cuisine Count"] = data["Cuisine Labels"].apply(len)

all_cuisines = [
    cuisine
    for cuisines in data["Cuisine Labels"]
    for cuisine in cuisines
]

cuisine_frequency = (
    pd.Series(all_cuisines)
    .value_counts()
    .rename_axis("Cuisine")
    .reset_index(name="Restaurant Count")
)

print("Total individual cuisine labels:",
      len(cuisine_frequency))

print("\nTop 20 cuisines:")
print(cuisine_frequency.head(20).to_string(index=False))

cuisine_frequency.to_csv(
    REPORT_DIR / "cuisine_frequency_distribution.csv",
    index=False
)

# ============================================================
# 7. SINGLE VS MULTIPLE CUISINE ANALYSIS
# ============================================================

print("\n" + "=" * 65)
print("7. SINGLE AND MULTIPLE CUISINE ANALYSIS")
print("=" * 65)

single_cuisine = (data["Cuisine Count"] == 1).sum()

multiple_cuisines = (data["Cuisine Count"] > 1).sum()

no_cuisine = (data["Cuisine Count"] == 0).sum()

cuisine_type_summary = pd.DataFrame({
    "Category": [
        "Single Cuisine",
        "Multiple Cuisines",
        "No Known Cuisine"
    ],
    "Restaurant Count": [
        single_cuisine,
        multiple_cuisines,
        no_cuisine
    ]
})

print(cuisine_type_summary)

cuisine_type_summary.to_csv(
    REPORT_DIR / "cuisine_type_distribution.csv",
    index=False
)

# ============================================================
# 8. IDENTIFY RARE CUISINE CLASSES
# ============================================================

print("\n" + "=" * 65)
print("8. RARE CUISINE ANALYSIS")
print("=" * 65)

RARE_THRESHOLD = 10

rare_cuisines = cuisine_frequency[
    cuisine_frequency["Restaurant Count"] < RARE_THRESHOLD
]

common_cuisines = cuisine_frequency[
    cuisine_frequency["Restaurant Count"] >= RARE_THRESHOLD
]

print("Rare cuisine threshold:", RARE_THRESHOLD)
print("Rare cuisine classes:", len(rare_cuisines))
print("Common cuisine classes:", len(common_cuisines))

print("\nRare cuisines:")
print(rare_cuisines.to_string(index=False))

rare_cuisines.to_csv(
    REPORT_DIR / "rare_cuisine_classes.csv",
    index=False
)

# ============================================================
# 9. CUISINE COUNT DISTRIBUTION
# ============================================================

print("\n" + "=" * 65)
print("9. CUISINE COUNT DISTRIBUTION")
print("=" * 65)

cuisine_count_distribution = (
    data["Cuisine Count"]
    .value_counts()
    .sort_index()
    .rename_axis("Number of Cuisines")
    .reset_index(name="Restaurant Count")
)

print(cuisine_count_distribution)

cuisine_count_distribution.to_csv(
    REPORT_DIR / "cuisine_count_distribution.csv",
    index=False
)

# ============================================================
# 10. CUISINE FREQUENCY VISUALIZATION
# ============================================================

print("\n" + "=" * 65)
print("10. GENERATE CUISINE VISUALIZATIONS")
print("=" * 65)

top_cuisines = cuisine_frequency.head(15)

plt.figure(figsize=(12, 7))

plt.barh(
    top_cuisines["Cuisine"].iloc[::-1],
    top_cuisines["Restaurant Count"].iloc[::-1]
)

plt.title("Top 15 Most Common Cuisines")
plt.xlabel("Restaurant Count")
plt.ylabel("Cuisine")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "01_top_cuisine_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: 01_top_cuisine_distribution.png")

# ============================================================
# 11. SINGLE VS MULTIPLE CUISINE VISUALIZATION
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    cuisine_type_summary["Category"],
    cuisine_type_summary["Restaurant Count"]
)

plt.title("Single vs Multiple Cuisine Distribution")
plt.xlabel("Cuisine Category")
plt.ylabel("Restaurant Count")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "02_single_multiple_cuisine_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: 02_single_multiple_cuisine_distribution.png")

# ============================================================
# 12. RARE CUISINE VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    ["Common Cuisines", "Rare Cuisines"],
    [len(common_cuisines), len(rare_cuisines)]
)

plt.title("Common vs Rare Cuisine Classes")
plt.xlabel("Cuisine Category")
plt.ylabel("Number of Cuisine Classes")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "03_common_rare_cuisine_classes.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: 03_common_rare_cuisine_classes.png")

# ============================================================
# 13. GENERATE DATASET AUDIT SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("13. GENERATE AUDIT SUMMARY")
print("=" * 65)

audit_summary = pd.DataFrame({
    "Metric": [
        "Total Restaurants",
        "Total Dataset Columns",
        "Missing Cuisine Records",
        "Restaurants With Cuisine Information",
        "Total Individual Cuisine Classes",
        "Single Cuisine Restaurants",
        "Multiple Cuisine Restaurants",
        "Restaurants Without Known Cuisine",
        "Common Cuisine Classes",
        "Rare Cuisine Classes",
        "Duplicate Records",
        "Duplicate Restaurant IDs"
    ],
    "Value": [
        len(data),
        len(data.columns) - 2,
        missing_cuisine_count,
        len(available_cuisines),
        len(cuisine_frequency),
        single_cuisine,
        multiple_cuisines,
        no_cuisine,
        len(common_cuisines),
        len(rare_cuisines),
        duplicate_rows,
        duplicate_ids if duplicate_ids is not None else "Not Available"
    ]
})

print(audit_summary.to_string(index=False))

audit_summary.to_csv(
    REPORT_DIR / "cuisine_dataset_audit_summary.csv",
    index=False
)

# ============================================================
# 14. CLASSIFICATION STRATEGY OBSERVATION
# ============================================================

print("\n" + "=" * 65)
print("14. CLASSIFICATION STRATEGY OBSERVATION")
print("=" * 65)

print("Cuisine labels per restaurant:")

print("Single cuisine restaurants:", single_cuisine)
print("Multiple cuisine restaurants:", multiple_cuisines)

if multiple_cuisines > single_cuisine:
    print(
        "\nObservation: Multi-label classification deserves "
        "serious consideration because multiple cuisine labels "
        "are common in this dataset."
    )
else:
    print(
        "\nObservation: Compare single-label and multi-label "
        "approaches before selecting the final classification strategy."
    )

print(
    "\nRare cuisine classes must be reviewed before model training. "
    "Do not merge or remove them without documenting the decision."
)

# ============================================================
# 15. SAVE CUISINE AUDIT DATA
# ============================================================

print("\n" + "=" * 65)
print("15. SAVE AUDIT DATA")
print("=" * 65)

audit_data = data[
    ["Restaurant ID", "Restaurant Name", "Cuisines", "Cuisine Labels", "Cuisine Count"]
].copy() if all(
    col in data.columns
    for col in ["Restaurant ID", "Restaurant Name"]
) else data[
    ["Cuisines", "Cuisine Labels", "Cuisine Count"]
].copy()

audit_data["Cuisine Labels"] = audit_data["Cuisine Labels"].apply(
    lambda labels: ", ".join(labels)
)

audit_data.to_csv(
    REPORT_DIR / "cuisine_audit_records.csv",
    index=False
)

print("Audit records saved successfully.")

# ============================================================
# 16. FINAL VALIDATION
# ============================================================

print("\n" + "=" * 65)
print("16. FINAL DATASET AUDIT VALIDATION")
print("=" * 65)

assert len(data) == 9551, "Unexpected dataset row count."
assert "Cuisines" in data.columns
assert len(cuisine_frequency) > 0
assert single_cuisine + multiple_cuisines + no_cuisine == len(data)

print("Dataset validation: PASS")
print("Cuisine extraction validation: PASS")
print("Cuisine distribution validation: PASS")
print("Single/multiple cuisine validation: PASS")

print("\nGenerated reports:")
for file in sorted(REPORT_DIR.glob("*.csv")):
    print("-", file.name)

print("\nGenerated figures:")
for file in sorted(FIGURE_DIR.glob("*.png")):
    print("-", file.name)

print("\n" + "=" * 75)
print("TASK 03 - CUISINE CLASSIFICATION")
print("PROGRAM 01 - CUISINE DATASET AUDIT COMPLETED")
print("=" * 75)
