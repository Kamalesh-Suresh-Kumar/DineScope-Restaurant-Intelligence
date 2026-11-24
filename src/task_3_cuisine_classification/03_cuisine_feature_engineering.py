from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from scipy.sparse import csr_matrix, save_npz, issparse

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "task_3_cuisine_classification"
REPORT_DIR = PROJECT_ROOT / "reports" / "task_3"
MODEL_DIR = PROJECT_ROOT / "models" / "task_3"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS
# TASK 03: CUISINE CLASSIFICATION
# PROGRAM 03: CUISINE FEATURE ENGINEERING
# ============================================================

print("=" * 75)
print("DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS")
print("TASK 03: CUISINE CLASSIFICATION")
print("PROGRAM 03 - CUISINE FEATURE ENGINEERING")
print("=" * 75)


# ============================================================
# 2. LOAD TRAINING AND TESTING DATA
# ============================================================

print("\n" + "=" * 65)
print("1. LOAD TRAINING AND TESTING DATA")
print("=" * 65)

required_files = [
    "cuisine_X_train.csv",
    "cuisine_X_test.csv",
    "cuisine_y_train.csv",
    "cuisine_y_test.csv"
]

for filename in required_files:
    file_path = DATA_DIR / filename
    assert file_path.exists(), f"Required file not found: {file_path}"

X_train_raw = pd.read_csv(DATA_DIR / "cuisine_X_train.csv")
X_test_raw = pd.read_csv(DATA_DIR / "cuisine_X_test.csv")

y_train_data = pd.read_csv(DATA_DIR / "cuisine_y_train.csv")
y_test_data = pd.read_csv(DATA_DIR / "cuisine_y_test.csv")

print("Training input shape:", X_train_raw.shape)
print("Testing input shape:", X_test_raw.shape)
print("Training target shape:", y_train_data.shape)
print("Testing target shape:", y_test_data.shape)


# ============================================================
# 3. VALIDATE SOURCE ROW ALIGNMENT
# ============================================================

print("\n" + "=" * 65)
print("2. VALIDATE SOURCE ROW ALIGNMENT")
print("=" * 65)

assert "Source Row Index" in X_train_raw.columns
assert "Source Row Index" in X_test_raw.columns

assert "Source Row Index" in y_train_data.columns
assert "Source Row Index" in y_test_data.columns

assert np.array_equal(
    X_train_raw["Source Row Index"].to_numpy(),
    y_train_data["Source Row Index"].to_numpy()
), "Training input and target row alignment failed."

assert np.array_equal(
    X_test_raw["Source Row Index"].to_numpy(),
    y_test_data["Source Row Index"].to_numpy()
), "Testing input and target row alignment failed."

print("Training row alignment: PASS")
print("Testing row alignment: PASS")


# ============================================================
# 4. PREPARE TARGET MATRICES
# ============================================================

print("\n" + "=" * 65)
print("3. PREPARE MULTI-LABEL TARGET MATRICES")
print("=" * 65)

y_train_columns = y_train_data.drop(
    columns=["Source Row Index"]
)

y_test_columns = y_test_data.drop(
    columns=["Source Row Index"]
)

y_train = y_train_columns.to_numpy(dtype=np.uint8)
y_test = y_test_columns.to_numpy(dtype=np.uint8)

cuisine_classes = y_train_columns.columns.to_list()

assert cuisine_classes == y_test_columns.columns.to_list()
assert y_train.shape[1] == y_test.shape[1]

print("Training target matrix:", y_train.shape)
print("Testing target matrix:", y_test.shape)
print("Cuisine classes:", len(cuisine_classes))


# ============================================================
# 5. SELECT MODEL INPUT FEATURES
# ============================================================

print("\n" + "=" * 65)
print("4. SELECT MODEL INPUT FEATURES")
print("=" * 65)

categorical_features = [
    "Country Code",
    "City",
    "Currency",
    "Has Table booking",
    "Has Online delivery",
    "Is delivering now",
    "Switch to order menu"
]

numerical_features = [
    "Longitude",
    "Latitude",
    "Average Cost for two",
    "Price range",
    "Aggregate rating",
    "Votes"
]

selected_features = categorical_features + numerical_features

for column in selected_features:
    assert column in X_train_raw.columns, (
        f"Missing training feature: {column}"
    )

    assert column in X_test_raw.columns, (
        f"Missing testing feature: {column}"
    )

X_train = X_train_raw[selected_features].copy()
X_test = X_test_raw[selected_features].copy()

print("Categorical features:", len(categorical_features))
print("Numerical features:", len(numerical_features))
print("Total selected features:", len(selected_features))

print("\nSelected features:")

for feature in selected_features:
    print("-", feature)


# ============================================================
# 6. HANDLE NUMERICAL VALUES
# ============================================================

print("\n" + "=" * 65)
print("5. HANDLE NUMERICAL VALUES")
print("=" * 65)

for column in numerical_features:

    X_train[column] = pd.to_numeric(
        X_train[column],
        errors="coerce"
    )

    X_test[column] = pd.to_numeric(
        X_test[column],
        errors="coerce"
    )

    X_train[column] = X_train[column].replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_test[column] = X_test[column].replace(
        [np.inf, -np.inf],
        np.nan
    )


# Apply log transformation to skewed numerical features.
# The transformation is applied consistently to training
# and testing data.

for column in ["Average Cost for two", "Votes"]:

    X_train[column] = np.log1p(
        X_train[column].clip(lower=0)
    )

    X_test[column] = np.log1p(
        X_test[column].clip(lower=0)
    )

print("Numerical conversion completed.")
print("Applied log1p transformation to:")
print("- Average Cost for two")
print("- Votes")


# ============================================================
# 7. PREPARE CATEGORICAL FEATURES
# ============================================================

print("\n" + "=" * 65)
print("6. PREPARE CATEGORICAL FEATURES")
print("=" * 65)

for column in categorical_features:

    X_train[column] = (
        X_train[column]
        .fillna("Unknown")
        .astype(str)
    )

    X_test[column] = (
        X_test[column]
        .fillna("Unknown")
        .astype(str)
    )

print("Categorical missing values handled.")
print("Categorical features prepared.")


# ============================================================
# 8. DEFINE FEATURE PREPROCESSING PIPELINES
# ============================================================

print("\n" + "=" * 65)
print("7. DEFINE FEATURE PREPROCESSING PIPELINES")
print("=" * 65)

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numeric_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ],
    remainder="drop"
)

print("Numerical pipeline prepared.")
print("Categorical pipeline prepared.")
print("Column transformer prepared.")


# ============================================================
# 9. FIT FEATURE PREPROCESSOR
# ============================================================

print("\n" + "=" * 65)
print("8. FIT FEATURE PREPROCESSOR")
print("=" * 65)

train_result = preprocessor.fit_transform(X_train)
test_result = preprocessor.transform(X_test)

# Convert both outputs into sparse CSR matrices.
X_train_transformed = csr_matrix(train_result)
X_test_transformed = csr_matrix(test_result)

print("Feature preprocessing completed.")
print("Training transformed shape:", X_train_transformed.shape)
print("Testing transformed shape:", X_test_transformed.shape)

print("Training representation: Sparse CSR Matrix")
print("Testing representation: Sparse CSR Matrix")


# ============================================================
# 10. EXTRACT ENGINEERED FEATURE NAMES
# ============================================================

print("\n" + "=" * 65)
print("9. EXTRACT ENGINEERED FEATURE NAMES")
print("=" * 65)

feature_names = [
    str(name)
    for name in preprocessor.get_feature_names_out()
]

print("Original selected features:", len(selected_features))
print("Engineered features:", len(feature_names))

feature_mapping = pd.DataFrame({
    "Feature Index": range(len(feature_names)),
    "Engineered Feature": feature_names
})

feature_mapping.to_csv(
    REPORT_DIR / "cuisine_engineered_feature_mapping.csv",
    index=False
)

print("Feature mapping saved.")


# ============================================================
# 11. SAVE ENGINEERED FEATURE MATRICES
# ============================================================

print("\n" + "=" * 65)
print("10. SAVE ENGINEERED FEATURE MATRICES")
print("=" * 65)

save_npz(
    DATA_DIR / "cuisine_X_train_engineered.npz",
    X_train_transformed
)

save_npz(
    DATA_DIR / "cuisine_X_test_engineered.npz",
    X_test_transformed
)

print("Training feature matrix saved.")
print("Testing feature matrix saved.")


# ============================================================
# 12. SAVE TARGET MATRICES
# ============================================================

print("\n" + "=" * 65)
print("11. SAVE MULTI-LABEL TARGET MATRICES")
print("=" * 65)

np.save(
    DATA_DIR / "cuisine_y_train.npy",
    y_train
)

np.save(
    DATA_DIR / "cuisine_y_test.npy",
    y_test
)

print("Training target matrix saved.")
print("Testing target matrix saved.")


# ============================================================
# 13. SAVE FEATURE PREPROCESSOR
# ============================================================

print("\n" + "=" * 65)
print("12. SAVE FEATURE PREPROCESSOR")
print("=" * 65)

joblib.dump(
    preprocessor,
    MODEL_DIR / "cuisine_feature_preprocessor.joblib"
)

print("Fitted feature preprocessor saved.")


# ============================================================
# 14. GENERATE FEATURE ENGINEERING SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("13. GENERATE FEATURE ENGINEERING SUMMARY")
print("=" * 65)

train_rows = len(X_train)
test_rows = len(X_test)

train_feature_shape = X_train_transformed.shape
test_feature_shape = X_test_transformed.shape

summary = pd.DataFrame({
    "Metric": [
        "Training Records",
        "Testing Records",
        "Original Selected Features",
        "Categorical Features",
        "Numerical Features",
        "Engineered Features",
        "Cuisine Classes",
        "Training Target Shape",
        "Testing Target Shape",
        "Training Feature Matrix Shape",
        "Testing Feature Matrix Shape",
        "Feature Preprocessing Method",
        "Numerical Scaling",
        "Categorical Encoding",
        "Target Representation"
    ],
    "Value": [
        train_rows,
        test_rows,
        len(selected_features),
        len(categorical_features),
        len(numerical_features),
        len(feature_names),
        len(cuisine_classes),
        str(y_train.shape),
        str(y_test.shape),
        str(train_feature_shape),
        str(test_feature_shape),
        "ColumnTransformer",
        "StandardScaler",
        "OneHotEncoder",
        "Multi-label binary matrix"
    ]
})

print(summary.to_string(index=False))

summary.to_csv(
    REPORT_DIR / "cuisine_feature_engineering_summary.csv",
    index=False
)


# ============================================================
# 15. FINAL FEATURE ENGINEERING VALIDATION
# ============================================================

print("\n" + "=" * 65)
print("14. FINAL FEATURE ENGINEERING VALIDATION")
print("=" * 65)

assert train_rows == y_train.shape[0]
assert test_rows == y_test.shape[0]

assert train_feature_shape == (
    train_rows,
    len(feature_names)
)

assert test_feature_shape == (
    test_rows,
    len(feature_names)
)

assert y_train.shape[1] == len(cuisine_classes)
assert y_test.shape[1] == len(cuisine_classes)

assert not any(
    "Cuisines" in feature
    for feature in feature_names
)

assert "Source Row Index" not in feature_names

assert len(feature_names) == len(set(feature_names))

assert (y_train.sum(axis=1) >= 1).all()
assert (y_test.sum(axis=1) >= 1).all()

assert issparse(X_train_transformed)
assert issparse(X_test_transformed)

print("Training feature validation: PASS")
print("Testing feature validation: PASS")
print("Target alignment validation: PASS")
print("Feature name validation: PASS")
print("Target leakage validation: PASS")
print("Multi-label target validation: PASS")
print("Sparse matrix validation: PASS")


# ============================================================
# 16. DISPLAY GENERATED FILES
# ============================================================

print("\n" + "=" * 65)
print("15. GENERATED FILES")
print("=" * 65)

print("\nGenerated feature datasets:")

for filename in [
    "cuisine_X_train_engineered.npz",
    "cuisine_X_test_engineered.npz",
    "cuisine_y_train.npy",
    "cuisine_y_test.npy"
]:
    file_path = DATA_DIR / filename
    assert file_path.exists(), f"Generated file missing: {filename}"
    print("-", filename)

print("\nGenerated reports:")

for file in sorted(REPORT_DIR.glob("cuisine_*feature*.csv")):
    print("-", file.name)

print("\nGenerated model artifacts:")

model_file = MODEL_DIR / "cuisine_feature_preprocessor.joblib"
assert model_file.exists()

print("- cuisine_feature_preprocessor.joblib")


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 75)
print("TASK 03 - CUISINE CLASSIFICATION")
print("PROGRAM 03 - CUISINE FEATURE ENGINEERING COMPLETED")
print("=" * 75)
