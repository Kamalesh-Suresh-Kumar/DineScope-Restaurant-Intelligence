import os
from pathlib import Path
import json
import pandas as pd

from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ============================================================
# DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS
# TASK 01: RESTAURANT RATING PREDICTION
# PROGRAM 03: DATA PREPROCESSING
# ============================================================

print("=" * 75)
print("DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS")
print("TASK 01: RESTAURANT RATING PREDICTION")
print("PROGRAM 03 - DATA PREPROCESSING")
print("=" * 75)

# ==========================================================
# 1. CONFIGURATION
# ==========================================================

DATA_PATH = str(PROJECT_ROOT / "data" / "raw" / "Dataset.csv")

PROCESSED_PATH = str(PROJECT_ROOT / "data" / "processed" / "task_1_rating_prediction")
REPORT_PATH = str(PROJECT_ROOT / "reports" / "task_1")

os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs(REPORT_PATH, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

# ==========================================================
# 2. LOAD DATASET
# ==========================================================

df = pd.read_csv(DATA_PATH)

print("\n1. ORIGINAL DATASET")

print("Shape:", df.shape)

required_columns = [
    "Country Code",
    "City",
    "Longitude",
    "Latitude",
    "Cuisines",
    "Has Table booking",
    "Has Online delivery",
    "Price range",
    "Votes",
    "Aggregate rating"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ==========================================================
# 3. REMOVE UNRATED RESTAURANTS
# ==========================================================

print("\n2. RATING CLEANING")

original_count = len(df)

# Zero represents Not Rated in this dataset.

df = df[
    df["Aggregate rating"] > 0
].copy()

removed_count = original_count - len(df)

print("Original records:", original_count)
print("Unrated records removed:", removed_count)
print("Rated records retained:", len(df))


# ==========================================================
# 4. HANDLE MISSING AND CORRUPTED VALUES
# ==========================================================

print("\n3. MISSING VALUE HANDLING")


# ----------------------------------------------------------
# 4.1 CUISINE CLEANING
# ----------------------------------------------------------

df["Cuisines"] = (
    df["Cuisines"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
    .replace("", "Unknown")
)


# ----------------------------------------------------------
# 4.2 CITY CLEANING
# ----------------------------------------------------------

df["City"] = (
    df["City"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
    .replace("", "Unknown")
)

# Known corrupted city names.

city_corrections = {
    "Bras�_lia": "Brasília",
    "S��o Paulo": "São Paulo",
    "��stanbul": "İstanbul"
}

# Count and correct known corrupted city names.

print("\nCorrected city names:")

corrupted_city_count = 0

for old_name, new_name in city_corrections.items():

    count = int(
        (df["City"] == old_name).sum()
    )

    corrupted_city_count += count

    df["City"] = df["City"].replace(
        old_name,
        new_name
    )

    print(
        f"{old_name} -> {new_name}: {count} records"
    )

print(
    "\nTotal corrupted city records corrected:",
    corrupted_city_count
)

# Replace any remaining corrupted city values.

remaining_corrupted = df["City"].str.contains(
    "\uFFFD",
    regex=False,
    na=False
)

remaining_corrupted_count = int(
    remaining_corrupted.sum()
)

df.loc[
    remaining_corrupted,
    "City"
] = "Unknown"

print(
    "Remaining corrupted city values replaced:",
    remaining_corrupted_count
)


# ----------------------------------------------------------
# 4.3 NORMALIZE CATEGORICAL TEXT
# ----------------------------------------------------------

categorical_columns = [
    "Has Table booking",
    "Has Online delivery"
]

for col in categorical_columns:

    df[col] = (
        df[col]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .str.title()
    )

# Country code is a category, not a continuous quantity.

df["Country Code"] = (
    df["Country Code"]
    .astype("Int64")
    .astype("string")
    .fillna("Unknown")
)


# ----------------------------------------------------------
# 4.4 CONVERT NUMERICAL FEATURES SAFELY
# ----------------------------------------------------------

numeric_columns = [
    "Longitude",
    "Latitude",
    "Price range",
    "Votes",
    "Aggregate rating"
]

for col in numeric_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

print("\nMissing values after cleaning:")

print(
    df[required_columns].isnull().sum()
)


# ==========================================================
# 5. VALIDATE DATA
# ==========================================================

print("\n4. DATA VALIDATION")


# ----------------------------------------------------------
# 5.1 VALIDATE TARGET RATINGS
# ----------------------------------------------------------

if df["Aggregate rating"].isnull().any():

    raise ValueError(
        "Missing target values found in Aggregate rating."
    )

invalid_ratings = (
    (df["Aggregate rating"] < 1) |
    (df["Aggregate rating"] > 5)
)

print(
    "Invalid rating records:",
    int(invalid_ratings.sum())
)

if invalid_ratings.any():

    raise ValueError(
        "Invalid aggregate rating values detected."
    )


# ----------------------------------------------------------
# 5.2 VALIDATE PRICE RANGE
# ----------------------------------------------------------

invalid_prices = (
    df["Price range"].notna() &
    ~df["Price range"].isin([1, 2, 3, 4])
)

print(
    "Invalid price range records:",
    int(invalid_prices.sum())
)

if invalid_prices.any():

    raise ValueError(
        "Unexpected price range values detected."
    )


# ----------------------------------------------------------
# 5.3 VALIDATE VOTES
# ----------------------------------------------------------

invalid_votes = (
    df["Votes"].notna() &
    (df["Votes"] < 0)
)

print(
    "Invalid vote records:",
    int(invalid_votes.sum())
)

if invalid_votes.any():

    raise ValueError(
        "Negative vote values detected."
    )


# ----------------------------------------------------------
# 5.4 REPORT MISSING NUMERICAL FEATURES
# ----------------------------------------------------------

print("\nMissing numerical values:")

print(
    df[numeric_columns].isnull().sum()
)

# Missing feature values are retained for the
# imputation pipeline in Program 04.


# ==========================================================
# 6. SELECT FEATURES
# ==========================================================

print("\n5. FEATURE SELECTION")

# Common features for both model configurations.

base_features = [
    "Country Code",
    "City",
    "Longitude",
    "Latitude",
    "Cuisines",
    "Has Table booking",
    "Has Online delivery",
    "Price range"
]

# Model A: Without Votes.

features_without_votes = base_features.copy()

# Model B: With Votes.

features_with_votes = base_features + [
    "Votes"
]

target = "Aggregate rating"

print("\nModel A features:")
print(features_without_votes)

print("\nModel B features:")
print(features_with_votes)

print("\nTarget variable:")
print(target)

# Create the modeling dataset.

model_df = df[
    features_with_votes + [target]
].copy()

print("\nModeling dataset shape:", model_df.shape)


# Check missing values in selected model features.

print("\nMissing values in model features:")

print(
    model_df[features_with_votes].isnull().sum()
)

# Target must not contain missing values.

if model_df[target].isnull().any():

    raise ValueError(
        "Target variable contains missing values."
    )


# ==========================================================
# 7. TRAIN-TEST SPLITTING
# ==========================================================

print("\n6. TRAIN-TEST SPLITTING")

# Randomly split the dataset into training and testing sets.
# random_state ensures that the split is reproducible.

X = model_df[features_with_votes]

y = model_df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

# Reconstruct training and testing datasets.

train_df = X_train.copy()

train_df[target] = y_train

test_df = X_test.copy()

test_df[target] = y_test

print("Training records:", len(train_df))
print("Testing records:", len(test_df))

print(
    "Training percentage:",
    round(len(train_df) / len(model_df) * 100, 2),
    "%"
)

print(
    "Testing percentage:",
    round(len(test_df) / len(model_df) * 100, 2),
    "%"
)

# Verify that the training and testing indices do not overlap.

overlap_count = len(
    set(train_df.index).intersection(
        set(test_df.index)
    )
)

print(
    "Overlapping records:",
    overlap_count
)

if overlap_count > 0:

    raise ValueError(
        "Training and testing datasets contain overlapping records."
    )


# ==========================================================
# 8. SAVE PROCESSED DATASETS
# ==========================================================

print("\n7. SAVING PROCESSED DATASETS")

train_path = os.path.join(
    PROCESSED_PATH,
    "train.csv"
)

test_path = os.path.join(
    PROCESSED_PATH,
    "test.csv"
)

train_df.to_csv(
    train_path,
    index=False
)

test_df.to_csv(
    test_path,
    index=False
)

print("Training dataset saved:", train_path)

print("Testing dataset saved:", test_path)


# ==========================================================
# 9. SAVE FEATURE CONFIGURATION
# ==========================================================

feature_config = {
    "target": target,

    "base_features": features_without_votes,

    "features_with_votes": features_with_votes,

    "random_state": RANDOM_STATE,

    "test_size": TEST_SIZE,

    "excluded_features": [
        "Restaurant ID",
        "Restaurant Name",
        "Address",
        "Locality",
        "Locality Verbose",
        "Average Cost for two",
        "Currency",
        "Rating color",
        "Rating text",
        "Is delivering now",
        "Switch to order menu"
    ]
}

config_path = os.path.join(
    REPORT_PATH,
    "feature_config.json"
)

with open(
    config_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        feature_config,
        file,
        indent=4
    )

print("Feature configuration saved:", config_path)


# ==========================================================
# 10. PREPROCESSING SUMMARY
# ==========================================================

print("\n8. PREPROCESSING SUMMARY")

summary = pd.DataFrame({
    "Metric": [
        "Original records",
        "Unrated records removed",
        "Rated records retained",
        "Corrupted city names replaced",
        "Remaining corrupted city values replaced",
        "Training records",
        "Testing records",
        "Overlapping records",
        "Selected features without votes",
        "Selected features with votes",
        "Target variable",
        "Duplicate records after cleaning"
    ],

    "Value": [
        original_count,
        removed_count,
        len(model_df),
        corrupted_city_count,
        remaining_corrupted_count,
        len(train_df),
        len(test_df),
        overlap_count,
        len(features_without_votes),
        len(features_with_votes),
        target,
        int(df.duplicated().sum())
    ]
})

print(summary.to_string(index=False))

summary.to_csv(
    os.path.join(
        REPORT_PATH,
        "preprocessing_summary.csv"
    ),
    index=False
)


# ==========================================================
# 11. TARGET DISTRIBUTION CHECK
# ==========================================================

print("\n9. TARGET DISTRIBUTION CHECK")

print("\nTraining target statistics:")

print(
    train_df[target].describe()
)

print("\nTesting target statistics:")

print(
    test_df[target].describe()
)

# Compare the rating distributions.

print("\nTraining target distribution:")

print(
    train_df[target]
    .value_counts()
    .sort_index()
)

print("\nTesting target distribution:")

print(
    test_df[target]
    .value_counts()
    .sort_index()
)

# Compare mean ratings.

print("\nMean rating comparison:")

print(
    "Training mean:",
    round(train_df[target].mean(), 4)
)

print(
    "Testing mean:",
    round(test_df[target].mean(), 4)
)


# ==========================================================
# 12. FINAL STATUS
# ==========================================================

print("\n" + "=" * 65)

print("DATA PREPROCESSING COMPLETED SUCCESSFULLY")

print("=" * 65)

print("\nGenerated files:")

print("\ndata/processed/task_1_rating_prediction/")

print("train.csv")
print("test.csv")

print("\nreports/")

print("feature_config.json")
print("preprocessing_summary.csv")

print("\nReady for model development.")
