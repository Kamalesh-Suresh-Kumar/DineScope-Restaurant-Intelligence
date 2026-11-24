from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ============================================================
# DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS
# TASK 02: RESTAURANT RECOMMENDATION SYSTEM
# PROGRAM 03: RECOMMENDATION FEATURE ENGINEERING
# ============================================================

print("=" * 75)
print("DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS")
print("TASK 02: RESTAURANT RECOMMENDATION SYSTEM")
print("PROGRAM 03 - RECOMMENDATION FEATURE ENGINEERING")
print("=" * 75)


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "task_2_restaurant_recommendation"
    / "recommendation_data.csv"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "task_2_restaurant_recommendation"
REPORT_DIR = PROJECT_ROOT / "reports" / "task_2"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CUISINE_MATRIX_PATH = (
    PROCESSED_DIR / "recommendation_cuisine_features.csv"
)

METADATA_PATH = (
    PROCESSED_DIR / "recommendation_restaurant_metadata.csv"
)

FEATURE_SUMMARY_PATH = (
    REPORT_DIR / "recommendation_feature_summary.csv"
)

CUISINE_LABELS_PATH = (
    REPORT_DIR / "recommendation_cuisine_labels.csv"
)


# ============================================================
# 2. LOAD PROCESSED DATASET
# ============================================================

print("\n" + "=" * 65)
print("2. LOAD PROCESSED DATASET")
print("=" * 65)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Processed dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

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
    raise ValueError("Restaurant IDs contain missing values.")

if df["Restaurant ID"].duplicated().any():
    raise ValueError("Duplicate restaurant IDs detected.")

print(f"Restaurants loaded: {len(df)}")
print(f"Dataset columns: {len(df.columns)}")

print("Processed dataset loaded successfully.")


# ============================================================
# 3. PREPARE CUISINE FEATURES
# ============================================================

print("\n" + "=" * 65)
print("3. PREPARE CUISINE FEATURES")
print("=" * 65)

def extract_cuisines(value):

    if pd.isna(value):
        return []

    cuisines = []

    # Normalize known corrupted labels.
    corrections = {
        "b�_rek": "börek",
        "d�_ner": "döner",
    }

    for cuisine in str(value).split(","):

        cuisine = cuisine.strip().casefold()

        if not cuisine or cuisine == "unknown":
            continue

        cuisine = corrections.get(cuisine, cuisine)

        cuisines.append(cuisine)

    # Remove duplicates while preserving order.
    return list(dict.fromkeys(cuisines))

cuisine_lists = df["Cuisines"].apply(extract_cuisines)

print("Cuisine lists prepared.")

restaurants_without_cuisine = int(
    (cuisine_lists.apply(len) == 0).sum()
)

print(
    "Restaurants without known cuisine information:",
    restaurants_without_cuisine
)


# ============================================================
# 4. GENERATE MULTI-HOT CUISINE ENCODING
# ============================================================

print("\n" + "=" * 65)
print("4. GENERATE MULTI-HOT CUISINE ENCODING")
print("=" * 65)

# Generate a dense multi-hot encoded array.
encoder = MultiLabelBinarizer(sparse_output=False)

# Pylance type clarification:
# With sparse_output=False, the runtime result is a NumPy array.
# cast() informs the static type checker of this expected type.
cuisine_array = cast(
    np.ndarray,
    encoder.fit_transform(cuisine_lists)
)

# Use compact integer storage for binary features.
cuisine_array = cuisine_array.astype(
    np.uint8,
    copy=False
)

cuisine_columns = [
    f"cuisine_{cuisine.replace(' ', '_')}"
    for cuisine in encoder.classes_
]

cuisine_features = pd.DataFrame(
    cuisine_array,
    columns=cuisine_columns,
    index=df.index
)

# Keep the original restaurant ID for reliable matching.
cuisine_features.insert(
    0,
    "Restaurant ID",
    df["Restaurant ID"].tolist()
)

print(f"Individual cuisine labels: {len(encoder.classes_)}")
print(f"Cuisine feature columns: {len(cuisine_columns)}")

print("\nCuisine feature matrix shape:")
print(cuisine_features.shape)


# ============================================================
# 5. PREPARE LOCATION AND PRICE FEATURES
# ============================================================

print("\n" + "=" * 65)
print("5. PREPARE LOCATION AND PRICE FEATURES")
print("=" * 65)

metadata = df[
    [
        "Restaurant ID",
        "Restaurant Name",
        "City",
        "Locality",
        "Cuisines",
        "Price range",
        "Aggregate rating",
        "Votes",
    ]
].copy()

# Normalize location fields for consistent matching.
metadata["City Normalized"] = (
    metadata["City"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.casefold()
)

metadata["Locality Normalized"] = (
    metadata["Locality"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.casefold()
)

# Ensure price range is numeric.
metadata["Price range"] = pd.to_numeric(
    metadata["Price range"],
    errors="coerce"
)

# Ensure ratings and votes are numeric.
metadata["Aggregate rating"] = pd.to_numeric(
    metadata["Aggregate rating"],
    errors="coerce"
)

metadata["Votes"] = pd.to_numeric(
    metadata["Votes"],
    errors="coerce"
)

# Explicitly identify restaurants with known ratings.
metadata["Is Rated"] = (
    metadata["Aggregate rating"].notna()
)

print("Location features prepared.")
print("Price range feature prepared.")
print("Rating and vote metadata prepared.")


# ============================================================
# 6. VALIDATE ENGINEERED FEATURES
# ============================================================

print("\n" + "=" * 65)
print("6. VALIDATE ENGINEERED FEATURES")
print("=" * 65)

if len(cuisine_features) != len(df):
    raise ValueError(
        "Cuisine feature row count does not match the dataset."
    )

if len(metadata) != len(df):
    raise ValueError(
        "Metadata row count does not match the dataset."
    )

if cuisine_features["Restaurant ID"].duplicated().any():
    raise ValueError(
        "Duplicate IDs found in cuisine feature matrix."
    )

if metadata["Restaurant ID"].duplicated().any():
    raise ValueError(
        "Duplicate IDs found in restaurant metadata."
    )

if not np.isin(
    cuisine_array,
    [0, 1]
).all():
    raise ValueError(
        "Cuisine encoding contains values other than 0 and 1."
    )

if not cuisine_features["Restaurant ID"].equals(
    metadata["Restaurant ID"]
):
    raise ValueError(
        "Restaurant ID order differs between feature outputs."
    )

print("Cuisine encoding validation passed.")
print("Restaurant ID validation passed.")
print("Feature row alignment validation passed.")


# ============================================================
# 7. GENERATE FEATURE SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("7. GENERATE FEATURE SUMMARY")
print("=" * 65)

feature_summary = pd.DataFrame([
    {
        "Feature Group": "Cuisine",
        "Feature Type": "Multi-hot encoded",
        "Number of Features": len(cuisine_columns),
        "Purpose": "Measure cuisine similarity",
    },
    {
        "Feature Group": "City",
        "Feature Type": "Normalized categorical metadata",
        "Number of Features": metadata["City Normalized"].nunique(),
        "Purpose": "Match restaurant city preferences",
    },
    {
        "Feature Group": "Locality",
        "Feature Type": "Normalized categorical metadata",
        "Number of Features": metadata["Locality Normalized"].nunique(),
        "Purpose": "Match restaurant locality preferences",
    },
    {
        "Feature Group": "Price Range",
        "Feature Type": "Numeric metadata",
        "Number of Features": metadata["Price range"].nunique(),
        "Purpose": "Match user budget preferences",
    },
    {
        "Feature Group": "Aggregate Rating",
        "Feature Type": "Numeric metadata",
        "Number of Features": 1,
        "Purpose": "Support rating-based ranking",
    },
    {
        "Feature Group": "Votes",
        "Feature Type": "Numeric metadata",
        "Number of Features": 1,
        "Purpose": "Provide popularity context",
    },
])

print(feature_summary.to_string(index=False))


# ============================================================
# 8. GENERATE CUISINE LABEL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("8. GENERATE CUISINE LABEL SUMMARY")
print("=" * 65)

cuisine_counts = cuisine_array.sum(axis=0)

cuisine_label_summary = pd.DataFrame({
    "Cuisine": encoder.classes_.tolist(),
    "Restaurant Count": cuisine_counts.tolist(),
})

cuisine_label_summary = (
    cuisine_label_summary
    .sort_values(
        by="Restaurant Count",
        ascending=False
    )
    .reset_index(drop=True)
)

print("\nTop 10 individual cuisines:")

print(
    cuisine_label_summary
    .head(10)
    .to_string(index=False)
)


# ============================================================
# 9. SAVE ENGINEERED FEATURES
# ============================================================

print("\n" + "=" * 65)
print("9. SAVE ENGINEERED FEATURES")
print("=" * 65)

cuisine_features.to_csv(
    CUISINE_MATRIX_PATH,
    index=False
)

metadata.to_csv(
    METADATA_PATH,
    index=False
)

feature_summary.to_csv(
    FEATURE_SUMMARY_PATH,
    index=False
)

cuisine_label_summary.to_csv(
    CUISINE_LABELS_PATH,
    index=False
)

print("\nFeature engineering outputs saved successfully.")

print(f"\n1. {CUISINE_MATRIX_PATH}")
print(f"2. {METADATA_PATH}")
print(f"3. {FEATURE_SUMMARY_PATH}")
print(f"4. {CUISINE_LABELS_PATH}")


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("TASK 02 - RESTAURANT RECOMMENDATION SYSTEM")
print("PROGRAM 03 - RECOMMENDATION FEATURE ENGINEERING COMPLETED")
print("=" * 75)

print(f"\nRestaurants processed: {len(df)}")
print(f"Individual cuisine features: {len(cuisine_columns)}")

print(
    "Restaurants without known cuisines:",
    restaurants_without_cuisine
)

print(f"Cuisine matrix shape: {cuisine_features.shape}")
print(f"Metadata shape: {metadata.shape}")

print("\nGenerated files:")
print("1. recommendation_cuisine_features.csv")
print("2. recommendation_restaurant_metadata.csv")
print("3. recommendation_feature_summary.csv")
print("4. recommendation_cuisine_labels.csv")

print("\nPROGRAM 03 EXECUTION SUCCESSFUL.")

print("=" * 75)
