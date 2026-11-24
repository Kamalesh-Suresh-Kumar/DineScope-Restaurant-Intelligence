from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

from sklearn.preprocessing import MultiLabelBinarizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "Dataset.csv"

REPORT_DIR = PROJECT_ROOT / "reports" / "task_3"
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "task_3_cuisine_classification"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures" / "task_3"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE: float = 0.20
RARE_THRESHOLD = 10

# ============================================================
# DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS
# TASK 03: CUISINE CLASSIFICATION
# PROGRAM 02: CUISINE DATA PREPROCESSING
# ============================================================

print("=" * 75)
print("DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS")
print("TASK 03: CUISINE CLASSIFICATION")
print("PROGRAM 02 - CUISINE DATA PREPROCESSING")
print("=" * 75)

# ============================================================
# 1. LOAD RAW DATASET
# ============================================================

print("\n" + "=" * 65)
print("1. LOAD RAW DATASET")
print("=" * 65)

assert RAW_FILE.exists(), f"Dataset not found: {RAW_FILE}"

data = pd.read_csv(RAW_FILE, encoding="utf-8")

print("Dataset loaded successfully.")
print("Dataset shape:", data.shape)

assert "Cuisines" in data.columns

# Preserve original row position for traceability.
data["Source Row Index"] = np.arange(len(data))

# Correct known city encoding issues.
city_corrections = {
    "Bras�_lia": "Brasília",
    "S��o Paulo": "São Paulo",
    "��stanbul": "İstanbul"
}

if "City" in data.columns:
    data["City"] = data["City"].replace(city_corrections)

print("Known city encoding corrections applied.")

# ============================================================
# 2. INSPECT CUISINE INFORMATION
# ============================================================

print("\n" + "=" * 65)
print("2. INSPECT CUISINE INFORMATION")
print("=" * 65)

print("Total restaurants:", len(data))
print("Missing cuisine records:", data["Cuisines"].isna().sum())

print("\nOriginal cuisine examples:")
print(data["Cuisines"].dropna().head(10).to_list())

# ============================================================
# 3. HANDLE MISSING CUISINE RECORDS
# ============================================================

print("\n" + "=" * 65)
print("3. HANDLE MISSING CUISINE RECORDS")
print("=" * 65)

missing_cuisine_data = data[
    data["Cuisines"].isna()
].copy()

classification_data = data[
    data["Cuisines"].notna()
].copy()

print(
    "Restaurants with missing cuisines:",
    len(missing_cuisine_data)
)

print(
    "Restaurants retained for classification:",
    len(classification_data)
)

missing_cuisine_data.to_csv(
    REPORT_DIR / "cuisine_missing_records.csv",
    index=False,
    encoding="utf-8"
)

print("Missing cuisine records saved separately.")

# ============================================================
# 4. CORRECT KNOWN ENCODING ISSUES
# ============================================================

print("\n" + "=" * 65)
print("4. CORRECT KNOWN CUISINE ENCODING ISSUES")
print("=" * 65)

corrections = {
    "d�_ner": "döner",
    "b�_rek": "börek"
}

correction_summary = []

for incorrect, correct in corrections.items():

    affected_rows = classification_data["Cuisines"].str.contains(
        incorrect,
        case=False,
        regex=False,
        na=False
    ).sum()

    classification_data["Cuisines"] = (
        classification_data["Cuisines"].str.replace(
            incorrect,
            correct,
            regex=False
        )
    )

    correction_summary.append({
        "Original Label": incorrect,
        "Corrected Label": correct,
        "Affected Restaurant Records": int(affected_rows)
    })

    print(f"{incorrect} -> {correct}")
    print("Affected restaurant records:", affected_rows)

correction_summary = pd.DataFrame(correction_summary)

correction_summary.to_csv(
    REPORT_DIR / "cuisine_encoding_corrections.csv",
    index=False,
    encoding="utf-8"
)

# ============================================================
# 5. NORMALIZE CUISINE LABELS
# ============================================================

print("\n" + "=" * 65)
print("5. NORMALIZE CUISINE LABELS")
print("=" * 65)


def extract_cuisines(value):

    if pd.isna(value):
        return []

    cuisines = []

    for cuisine in str(value).split(","):

        cuisine = cuisine.strip().casefold()

        if not cuisine or cuisine == "unknown":
            continue

        cuisine = corrections.get(cuisine, cuisine)

        if cuisine not in cuisines:
            cuisines.append(cuisine)

    return cuisines


classification_data["Cuisine Labels"] = (
    classification_data["Cuisines"].apply(extract_cuisines)
)

classification_data["Cuisine Count"] = (
    classification_data["Cuisine Labels"].apply(len)
)

# Identify records without usable cuisine labels.
empty_label_data = classification_data[
    classification_data["Cuisine Count"] == 0
].copy()

classification_data = classification_data[
    classification_data["Cuisine Count"] > 0
].copy()

print(
    "Restaurants with usable cuisine labels:",
    len(classification_data)
)

print(
    "Records without usable labels:",
    len(empty_label_data)
)

if len(empty_label_data) > 0:

    empty_label_data.to_csv(
        REPORT_DIR / "cuisine_empty_label_records.csv",
        index=False,
        encoding="utf-8"
    )

# ============================================================
# 6. VALIDATE NORMALIZED CUISINE LABELS
# ============================================================

print("\n" + "=" * 65)
print("6. VALIDATE NORMALIZED CUISINE LABELS")
print("=" * 65)

all_cuisines = [
    cuisine
    for labels in classification_data["Cuisine Labels"]
    for cuisine in labels
]

cuisine_frequency = (
    pd.Series(all_cuisines)
    .value_counts()
    .rename_axis("Cuisine")
    .reset_index(name="Restaurant Count")
)

print(
    "Total unique cuisine labels:",
    len(cuisine_frequency)
)

print("\nTop 20 cuisines:")
print(cuisine_frequency.head(20).to_string(index=False))

print("\nKnown corrected labels:")

for label in ["döner", "börek"]:

    count = int(
        (cuisine_frequency["Cuisine"] == label).sum()
    )

    print(label, ":", count, "unique label entry")

corrupted_labels = [
    label
    for label in cuisine_frequency["Cuisine"]
    if "\ufffd" in label
]

print(
    "\nCuisine labels containing replacement characters:",
    len(corrupted_labels)
)

if corrupted_labels:
    print(corrupted_labels)

cuisine_frequency.to_csv(
    REPORT_DIR / "preprocessed_cuisine_frequency.csv",
    index=False,
    encoding="utf-8"
)

assert len(cuisine_frequency) > 0

# ============================================================
# 7. ANALYSE RARE CUISINE LABELS
# ============================================================

print("\n" + "=" * 65)
print("7. ANALYSE RARE CUISINE LABELS")
print("=" * 65)

rare_cuisines = cuisine_frequency[
    cuisine_frequency["Restaurant Count"] < RARE_THRESHOLD
].copy()

common_cuisines = cuisine_frequency[
    cuisine_frequency["Restaurant Count"] >= RARE_THRESHOLD
].copy()

print("Rare cuisine threshold:", RARE_THRESHOLD)
print("Common cuisine classes:", len(common_cuisines))
print("Rare cuisine classes:", len(rare_cuisines))

print("\nRare cuisine labels:")
print(rare_cuisines.to_string(index=False))

# Preserve every cuisine label.
rare_cuisines["Preprocessing Decision"] = (
    "Retained - review evaluation coverage"
)

common_cuisines["Preprocessing Decision"] = (
    "Retained - common label"
)

cuisine_class_analysis = pd.concat(
    [common_cuisines, rare_cuisines],
    ignore_index=True
).sort_values(
    "Restaurant Count",
    ascending=False
)

cuisine_class_analysis.to_csv(
    REPORT_DIR / "cuisine_class_preprocessing_analysis.csv",
    index=False,
    encoding="utf-8"
)

print("\nAll cuisine classes retained.")

# ============================================================
# 8. PREPARE MULTI-LABEL TARGET
# ============================================================

print("\n" + "=" * 65)
print("8. PREPARE MULTI-LABEL TARGET")
print("=" * 65)

mlb = MultiLabelBinarizer(sparse_output=False)

target_matrix = mlb.fit_transform(
    classification_data["Cuisine Labels"]
)

target_matrix = np.asarray(
    target_matrix,
    dtype=np.uint8
)

cuisine_classes = list(mlb.classes_)

print("Multi-label target prepared successfully.")
print("Target matrix shape:", target_matrix.shape)
print("Number of cuisine classes:", len(cuisine_classes))

print("\nSample target matrix:")

print(
    pd.DataFrame(
        target_matrix[:5],
        columns=cuisine_classes
    )
)

# Save complete target matrix.
target_data = pd.DataFrame(
    target_matrix,
    columns=cuisine_classes
)

target_data.insert(
    0,
    "Source Row Index",
    classification_data["Source Row Index"].to_numpy()
)

target_data.to_csv(
    DATA_DIR / "cuisine_target_matrix.csv",
    index=False,
    encoding="utf-8"
)

# Save cuisine label mapping.
label_mapping = pd.DataFrame({
    "Class Index": range(len(cuisine_classes)),
    "Cuisine Label": cuisine_classes
})

label_mapping.to_csv(
    REPORT_DIR / "cuisine_label_mapping.csv",
    index=False,
    encoding="utf-8"
)

# ============================================================
# 9. PREPARE PROCESSED CLASSIFICATION DATASET
# ============================================================

print("\n" + "=" * 65)
print("9. PREPARE PROCESSED CLASSIFICATION DATASET")
print("=" * 65)

processed_data = classification_data.copy()

processed_data["Cuisine Labels"] = (
    processed_data["Cuisine Labels"].apply(
        lambda labels: json.dumps(
            labels,
            ensure_ascii=False
        )
    )
)

processed_data.to_csv(
    DATA_DIR / "cuisine_classification_data.csv",
    index=False,
    encoding="utf-8"
)

print("Processed classification dataset saved.")
print("Shape:", processed_data.shape)

# ============================================================
# 10. PREPARE MODEL INPUT DATA
# ============================================================

print("\n" + "=" * 65)
print("10. PREPARE MODEL INPUT DATA")
print("=" * 65)

# IMPORTANT:
# Original Cuisines column is excluded because it directly
# contains the target labels and would cause target leakage.
#
# Source Row Index is retained only for traceability.
# It must not be used as a predictive feature.

X = classification_data.drop(
    columns=[
        "Cuisines",
        "Cuisine Labels",
        "Cuisine Count"
    ]
).copy()

y = target_matrix

print("Model input dataset shape:", X.shape)
print("Target matrix shape:", y.shape)

print("Original Cuisines column excluded: PASS")
print("Source Row Index retained for traceability: PASS")

# ============================================================
# 11. SPLIT TRAINING AND TESTING DATA
# ============================================================

print("\n" + "=" * 65)
print("11. SPLIT TRAINING AND TESTING DATA")
print("=" * 65)

split_method = "Random train_test_split"

try:

    from iterstrat.ml_stratifiers import (
        MultilabelStratifiedShuffleSplit
    )
    
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,  # type: ignore[arg-type]
        random_state=RANDOM_STATE
    )
    
    train_indices, test_indices = next(
        splitter.split(X, y)
    )

    split_method = "Iterative multi-label stratification"

except ImportError:

    print(
        "Optional package not found. "
        "Using reproducible random splitting."
    )

    from sklearn.model_selection import train_test_split

    indices = np.arange(len(X))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True
    )

X_train = X.iloc[train_indices].copy()
X_test = X.iloc[test_indices].copy()

y_train = y[train_indices]
y_test = y[test_indices]

print("Split method:", split_method)
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print(
    "Training percentage:",
    round(len(X_train) / len(X) * 100, 2)
)

print(
    "Testing percentage:",
    round(len(X_test) / len(X) * 100, 2)
)

# ============================================================
# 12. VALIDATE TRAINING AND TESTING LABEL COVERAGE
# ============================================================

print("\n" + "=" * 65)
print("12. VALIDATE TRAINING AND TESTING LABEL COVERAGE")
print("=" * 65)

train_label_counts = y_train.sum(axis=0)
test_label_counts = y_test.sum(axis=0)

label_coverage = pd.DataFrame({
    "Cuisine": cuisine_classes,
    "Total Records": target_matrix.sum(axis=0),
    "Training Records": train_label_counts,
    "Testing Records": test_label_counts
})

label_coverage["Training Coverage"] = (
    label_coverage["Training Records"] > 0
)

label_coverage["Testing Coverage"] = (
    label_coverage["Testing Records"] > 0
)

label_coverage["Training Only"] = (
    (label_coverage["Training Records"] > 0)
    & (label_coverage["Testing Records"] == 0)
)

label_coverage["Testing Only"] = (
    (label_coverage["Training Records"] == 0)
    & (label_coverage["Testing Records"] > 0)
)

print(
    "Cuisine labels in complete dataset:",
    len(cuisine_classes)
)

print(
    "Cuisine labels present in training:",
    int(label_coverage["Training Coverage"].sum())
)

print(
    "Cuisine labels present in testing:",
    int(label_coverage["Testing Coverage"].sum())
)

print(
    "Labels missing from training:",
    int((~label_coverage["Training Coverage"]).sum())
)

print(
    "Labels missing from testing:",
    int((~label_coverage["Testing Coverage"]).sum())
)

print("\nLabels with limited split coverage:")

limited_coverage = label_coverage[
    label_coverage["Training Only"]
    | label_coverage["Testing Only"]
]

if len(limited_coverage) > 0:

    print(limited_coverage.to_string(index=False))

else:

    print("All cuisine labels are represented in both splits.")

label_coverage.to_csv(
    REPORT_DIR / "cuisine_train_test_label_coverage.csv",
    index=False,
    encoding="utf-8"
)

# ============================================================
# 13. SAVE TRAINING AND TESTING DATA
# ============================================================

print("\n" + "=" * 65)
print("13. SAVE TRAINING AND TESTING DATA")
print("=" * 65)

X_train.to_csv(
    DATA_DIR / "cuisine_X_train.csv",
    index=False,
    encoding="utf-8"
)

X_test.to_csv(
    DATA_DIR / "cuisine_X_test.csv",
    index=False,
    encoding="utf-8"
)

y_train_data = pd.DataFrame(
    y_train,
    columns=cuisine_classes
)

y_test_data = pd.DataFrame(
    y_test,
    columns=cuisine_classes
)

y_train_data.insert(
    0,
    "Source Row Index",
    X_train["Source Row Index"].to_numpy()
)

y_test_data.insert(
    0,
    "Source Row Index",
    X_test["Source Row Index"].to_numpy()
)

y_train_data.to_csv(
    DATA_DIR / "cuisine_y_train.csv",
    index=False,
    encoding="utf-8"
)

y_test_data.to_csv(
    DATA_DIR / "cuisine_y_test.csv",
    index=False,
    encoding="utf-8"
)

print("Training and testing datasets saved.")

# ============================================================
# 14. GENERATE PREPROCESSING SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("14. GENERATE PREPROCESSING SUMMARY")
print("=" * 65)

split_summary = pd.DataFrame({
    "Metric": [
        "Original Dataset Records",
        "Missing Cuisine Records",
        "Records Without Usable Labels",
        "Records Used for Classification",
        "Total Cuisine Classes",
        "Common Cuisine Classes",
        "Rare Cuisine Classes",
        "Training Records",
        "Testing Records",
        "Training Percentage",
        "Testing Percentage",
        "Training Labels Present",
        "Testing Labels Present",
        "Labels Missing From Training",
        "Labels Missing From Testing"
    ],
    "Value": [
        len(data),
        len(missing_cuisine_data),
        len(empty_label_data),
        len(classification_data),
        len(cuisine_classes),
        len(common_cuisines),
        len(rare_cuisines),
        len(X_train),
        len(X_test),
        round(len(X_train) / len(X) * 100, 2),
        round(len(X_test) / len(X) * 100, 2),
        int(label_coverage["Training Coverage"].sum()),
        int(label_coverage["Testing Coverage"].sum()),
        int((~label_coverage["Training Coverage"]).sum()),
        int((~label_coverage["Testing Coverage"]).sum())
    ]
})

print(split_summary.to_string(index=False))

split_summary.to_csv(
    REPORT_DIR / "cuisine_preprocessing_summary.csv",
    index=False,
    encoding="utf-8"
)

# ============================================================
# 15. GENERATE PREPROCESSING VISUALIZATION
# ============================================================

print("\n" + "=" * 65)
print("15. GENERATE PREPROCESSING VISUALIZATION")
print("=" * 65)

split_counts = pd.DataFrame({
    "Dataset": ["Training", "Testing"],
    "Restaurant Count": [len(X_train), len(X_test)]
})

plt.figure(figsize=(8, 5))

plt.bar(
    split_counts["Dataset"],
    split_counts["Restaurant Count"]
)

plt.title("Cuisine Classification Training and Testing Split")
plt.xlabel("Dataset")
plt.ylabel("Restaurant Count")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "04_cuisine_train_test_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: 04_cuisine_train_test_distribution.png")

# ============================================================
# 16. FINAL PREPROCESSING VALIDATION
# ============================================================

print("\n" + "=" * 65)
print("16. FINAL PREPROCESSING VALIDATION")
print("=" * 65)

assert len(classification_data) == len(X)

assert len(X_train) + len(X_test) == len(classification_data)

assert y_train.shape[0] == len(X_train)
assert y_test.shape[0] == len(X_test)

assert y_train.shape[1] == len(cuisine_classes)
assert y_test.shape[1] == len(cuisine_classes)

assert target_matrix.dtype == np.uint8

assert not classification_data["Cuisine Labels"].apply(
    lambda labels: len(labels) == 0
).any()

assert classification_data["Source Row Index"].is_unique

assert X_train["Source Row Index"].is_unique
assert X_test["Source Row Index"].is_unique

assert set(X_train["Source Row Index"]).isdisjoint(
    set(X_test["Source Row Index"])
)

assert "Cuisines" not in X_train.columns
assert "Cuisines" not in X_test.columns

print("Dataset preprocessing validation: PASS")
print("Cuisine normalization validation: PASS")
print("Multi-label target validation: PASS")
print("Train/test split validation: PASS")
print("Source row traceability validation: PASS")
print("Target leakage validation: PASS")

print("\nSplit method:", split_method)

# ============================================================
# 17. DISPLAY GENERATED FILES
# ============================================================

print("\n" + "=" * 65)
print("17. GENERATED FILES")
print("=" * 65)

print("\nGenerated processed datasets:")

for file in sorted(DATA_DIR.glob("cuisine_*.csv")):
    print("-", file.name)

print("\nGenerated reports:")

for file in sorted(REPORT_DIR.glob("*.csv")):
    print("-", file.name)

print("\nGenerated figures:")

for file in sorted(FIGURE_DIR.glob("*.png")):
    print("-", file.name)

# ============================================================
# FINAL COMPLETION
# ============================================================

print("\n" + "=" * 75)
print("TASK 03 - CUISINE CLASSIFICATION")
print("PROGRAM 02 - CUISINE DATA PREPROCESSING COMPLETED")
print("=" * 75)
