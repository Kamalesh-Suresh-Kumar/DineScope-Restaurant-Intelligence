import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ============================================================
# DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS
# TASK 01: RESTAURANT RATING PREDICTION
# PROGRAM 02: EXPLORATORY DATA ANALYSIS
# ============================================================

print("=" * 75)
print("DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS")
print("TASK 01: RESTAURANT RATING PREDICTION")
print("PROGRAM 02 - EXPLORATORY DATA ANALYSIS")
print("=" * 75)

# ==========================================================
# 1. CONFIGURATION
# ==========================================================

DATA_PATH = str(PROJECT_ROOT / "data" / "raw" / "Dataset.csv")
FIGURE_PATH = str(PROJECT_ROOT / "outputs" / "figures" / "task_1")
REPORT_PATH = str(PROJECT_ROOT / "reports" / "task_1")

os.makedirs(FIGURE_PATH, exist_ok=True)
os.makedirs(REPORT_PATH, exist_ok=True)

sns.set_theme(style="whitegrid")

# ==========================================================
# 2. LOAD AND VALIDATE DATASET
# ==========================================================

df = pd.read_csv(DATA_PATH)

# Normalize known text-encoding issues in the analysis copy.

city_corrections = {
    "Bras�_lia": "Brasília",
    "S��o Paulo": "São Paulo",
    "��stanbul": "İstanbul"
}

df["City"] = df["City"].replace(city_corrections)

currency_corrections = {
    "Pounds(��)": "Pounds Sterling"
}

df["Currency"] = df["Currency"].replace(currency_corrections)

required_columns = [
    "Restaurant ID",
    "City",
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
        f"Missing required columns: {missing_columns}"
    )

print("\nDataset shape:", df.shape)

# Create a separate dataset containing rated restaurants.
rated_df = df[
    df["Aggregate rating"] > 0
].copy()

unrated_count = int(
    (df["Aggregate rating"] == 0).sum()
)

rated_count = len(rated_df)

print("\nRated restaurants:", rated_count)
print("Unrated restaurants:", unrated_count)

# ==========================================================
# 3. BASIC DATASET INFORMATION
# ==========================================================

print("\n1. BASIC DATASET INFORMATION")

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)

print("\nStatistical summary:")
print(df.describe())

# ==========================================================
# 4. RATING DISTRIBUTION
# ==========================================================

print("\n2. RATING ANALYSIS")

rating_counts = (
    df["Aggregate rating"]
    .value_counts()
    .sort_index()
)

print("\nRating distribution:")
print(rating_counts)

# Histogram of rated restaurants only

plt.figure(figsize=(10, 6))

plt.hist(
    rated_df["Aggregate rating"],
    bins=20,
    edgecolor="black"
)

plt.title("Distribution of Restaurant Ratings")
plt.xlabel("Aggregate Rating")
plt.ylabel("Number of Restaurants")

plt.tight_layout()

plt.savefig(
    f"{FIGURE_PATH}/01_rating_distribution.png",
    dpi=300
)

plt.close()

# Rated versus unrated restaurants

rating_status = [
    rated_count,
    unrated_count
]

rating_labels = [
    "Rated",
    "Not Rated"
]

plt.figure(figsize=(7, 7))

plt.pie(
    rating_status,
    labels=rating_labels,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Rated vs Unrated Restaurants")
plt.tight_layout()

plt.savefig(
    f"{FIGURE_PATH}/02_rating_status.png",
    dpi=300
)

plt.close()

# ==========================================================
# 5. CITY ANALYSIS
# ==========================================================

print("\n3. CITY ANALYSIS")

city_counts = (
    df["City"]
    .value_counts()
    .head(15)
)

print("\nTop 15 cities by restaurant count:")
print(city_counts)

plt.figure(figsize=(12, 7))

plt.barh(
    city_counts.index.astype(str).tolist()[::-1],
    city_counts.to_numpy(dtype=int)[::-1]
)

plt.title("Top 15 Cities by Number of Restaurants")
plt.xlabel("Number of Restaurants")
plt.ylabel("City")

plt.tight_layout()

plt.savefig(
    f"{FIGURE_PATH}/03_top_cities.png",
    dpi=300
)

plt.close()

# Restaurant count and average votes for all records

city_summary = df.groupby("City").agg(
    restaurant_count=("Restaurant ID", "count"),
    average_votes=("Votes", "mean")
)

# Average rating calculated using rated restaurants only

city_ratings = rated_df.groupby("City").agg(
    average_rating=("Aggregate rating", "mean"),
    rated_restaurant_count=("Aggregate rating", "count")
)

city_summary = city_summary.join(city_ratings)

city_summary = city_summary.sort_values(
    by="restaurant_count",
    ascending=False
)

city_summary.to_csv(
    f"{REPORT_PATH}/city_analysis.csv"
)

print("\nCity analysis saved successfully.")

# ==========================================================
# 6. PRICE RANGE ANALYSIS
# ==========================================================

print("\n4. PRICE RANGE ANALYSIS")

price_counts = (
    df["Price range"]
    .value_counts()
    .sort_index()
)

print("\nRestaurant distribution by price range:")
print(price_counts)

plt.figure(figsize=(8, 5))

plt.bar(
    price_counts.index.astype(str).tolist(),
    price_counts.to_numpy(dtype=int),
    edgecolor="black"
)

plt.title("Restaurant Distribution by Price Range")
plt.xlabel("Price Range")
plt.ylabel("Number of Restaurants")

plt.tight_layout()

plt.savefig(
    f"{FIGURE_PATH}/04_price_range.png",
    dpi=300
)

plt.close()

# ==========================================================
# 7. PRICE RANGE VS RATINGS
# ==========================================================

print("\n5. PRICE RANGE VS RATINGS")

price_rating = rated_df.groupby(
    "Price range"
)["Aggregate rating"].agg(
    ["mean", "median", "count"]
)

print("\nPrice range rating statistics:")
print(price_rating)

price_groups = [
    group["Aggregate rating"].to_numpy(dtype=float)
    for _, group in rated_df.groupby("Price range")
]

price_labels = [
    str(price)
    for price in sorted(rated_df["Price range"].unique())
]

plt.figure(figsize=(9, 6))

plt.boxplot(
    price_groups,
    tick_labels=price_labels
)

plt.title("Price Range vs Restaurant Ratings")
plt.xlabel("Price Range")
plt.ylabel("Aggregate Rating")

plt.tight_layout()

plt.savefig(
    f"{FIGURE_PATH}/05_price_vs_rating.png",
    dpi=300
)

plt.close()

price_rating.to_csv(
    f"{REPORT_PATH}/price_rating_analysis.csv"
)

# ==========================================================
# 8. VOTES VS RATINGS
# ==========================================================

print("\n6. VOTES VS RATINGS")

zero_vote_count = int(
    (rated_df["Votes"] == 0).sum()
)

print("Rated restaurants with zero votes:", zero_vote_count)

plt.figure(figsize=(10, 6))

plt.scatter(
    rated_df["Votes"],
    rated_df["Aggregate rating"],
    alpha=0.5
)

plt.title("Restaurant Votes vs Aggregate Ratings")
plt.xlabel("Number of Votes")
plt.ylabel("Aggregate Rating")

plt.tight_layout()

plt.savefig(
    f"{FIGURE_PATH}/06_votes_vs_rating.png",
    dpi=300
)

plt.close()

vote_rating_correlation = rated_df[
    ["Votes", "Aggregate rating"]
].corr()

print("\nVotes and rating correlation:")
print(vote_rating_correlation)

vote_rating_correlation.to_csv(
    f"{REPORT_PATH}/votes_rating_correlation.csv"
)

# ==========================================================
# 9. CUISINE ANALYSIS
# ==========================================================

print("\n7. CUISINE ANALYSIS")

missing_cuisine_count = int(
    df["Cuisines"].isnull().sum()
)

print("Restaurants with missing cuisines:", missing_cuisine_count)

# Preserve missing cuisine information as Unknown

cuisine_df = df["Cuisines"].fillna("Unknown")

# Separate multiple cuisines into individual labels

cuisine_list = (
    cuisine_df
    .astype(str)
    .str.split(",")
    .explode()
    .str.strip()
)

cuisine_list = cuisine_list[
    cuisine_list != ""
]

cuisine_counts = cuisine_list.value_counts()

print(
    "\nTotal unique individual cuisine labels:",
    cuisine_counts.size
)

print("\nTop 15 cuisines:")
print(cuisine_counts.head(15))

top_cuisines = cuisine_counts.head(15)

plt.figure(figsize=(12, 7))

plt.barh(
    top_cuisines.index.astype(str).tolist()[::-1],
    top_cuisines.to_numpy(dtype=int)[::-1]
)

plt.title("Top 15 Most Common Cuisines")
plt.xlabel("Restaurant-Cuisine Associations")
plt.ylabel("Cuisine")

plt.tight_layout()

plt.savefig(
    f"{FIGURE_PATH}/07_top_cuisines.png",
    dpi=300
)

plt.close()

cuisine_counts.to_csv(
    f"{REPORT_PATH}/cuisine_frequency.csv",
    header=["count"]
)

# ==========================================================
# 10. COST ANALYSIS BY CURRENCY
# ==========================================================

print("\n8. AVERAGE COST ANALYSIS BY CURRENCY")

cost_df = df[
    ["Currency", "Average Cost for two"]
].copy()

cost_df["Average Cost for two"] = pd.to_numeric(
    cost_df["Average Cost for two"],
    errors="coerce"
)

cost_df = cost_df.dropna(
    subset=["Currency", "Average Cost for two"]
)

cost_summary = cost_df.groupby("Currency").agg(
    restaurant_count=("Average Cost for two", "count"),
    mean_cost=("Average Cost for two", "mean"),
    median_cost=("Average Cost for two", "median"),
    minimum_cost=("Average Cost for two", "min"),
    maximum_cost=("Average Cost for two", "max"),
    zero_cost_count=(
        "Average Cost for two",
        lambda x: int((x == 0).sum())
    )
)

cost_summary = cost_summary.sort_values(
    by="restaurant_count",
    ascending=False
)

print("\nCost statistics grouped by currency:")
print(cost_summary)

cost_summary.to_csv(
    f"{REPORT_PATH}/cost_analysis_by_currency.csv"
)

print("\nTotal restaurants with zero cost:",
      int((cost_df["Average Cost for two"] == 0).sum()))

# Visualize separate distributions for the four most common currencies.
# Do not compare their cost magnitudes as equivalent monetary values.

top_currencies = (
    cost_df["Currency"]
    .value_counts()
    .head(4)
    .index
    .tolist()
)

fig, axes = plt.subplots(
    2, 2,
    figsize=(12, 9)
)

for ax, currency in zip(axes.flatten(), top_currencies):

    currency_cost = cost_df.loc[
        cost_df["Currency"] == currency,
        "Average Cost for two"
    ]

    ax.hist(
        np.log1p(currency_cost),
        bins=25,
        edgecolor="black"
    )

    ax.set_title(f"Cost Distribution - {currency}")
    ax.set_xlabel("log(1 + Average Cost for Two)")
    ax.set_ylabel("Number of Restaurants")

plt.tight_layout()

plt.savefig(
    f"{FIGURE_PATH}/08_cost_distribution_by_currency.png",
    dpi=300
)

plt.close()

# ==========================================================
# 11. NUMERICAL CORRELATION
# ==========================================================

print("\n9. NUMERICAL CORRELATION")

# Exclude Average Cost for two because different currencies
# cannot be directly compared in one correlation matrix.

numeric_columns = [
    "Price range",
    "Aggregate rating",
    "Votes"
]

correlation_matrix = rated_df[
    numeric_columns
].corr()

print("\nNumerical correlation matrix:")
print(correlation_matrix)

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    vmin=-1,
    vmax=1
)

plt.title("Numerical Feature Correlation")

plt.tight_layout()

plt.savefig(
    f"{FIGURE_PATH}/09_correlation_heatmap.png",
    dpi=300
)

plt.close()

correlation_matrix.to_csv(
    f"{REPORT_PATH}/correlation_matrix.csv"
)

# ==========================================================
# 12. MISSING VALUE ANALYSIS
# ==========================================================

print("\n10. MISSING VALUE ANALYSIS")

missing_values = df.isnull().sum()

missing_values = missing_values[
    missing_values > 0
]

print("\nColumns containing missing values:")
print(missing_values)

if not missing_values.empty:

    plt.figure(figsize=(9, 5))

    plt.barh(
        missing_values.index.astype(str).tolist()[::-1],
        missing_values.to_numpy(dtype=int)[::-1]
    )

    plt.title("Missing Values by Column")
    plt.xlabel("Missing Records")
    plt.ylabel("Column")

    plt.tight_layout()

    plt.savefig(
        f"{FIGURE_PATH}/10_missing_values.png",
        dpi=300
    )

    plt.close()

    missing_values.to_csv(
        f"{REPORT_PATH}/missing_values.csv",
        header=["missing_count"]
    )

else:

    print("No missing values found.")

# ==========================================================
# 13. FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 65)
print("EXPLORATORY DATA ANALYSIS SUMMARY")
print("=" * 65)

print("\nTotal restaurants:", len(df))
print("Rated restaurants:", rated_count)
print("Unrated restaurants:", unrated_count)
print("Unique cities:", df["City"].nunique())
print("Unique cuisine labels:", cuisine_counts.size)
print("Missing cuisine values:", missing_cuisine_count)
print("Duplicate records:", df.duplicated().sum())
print("Total missing values:", df.isnull().sum().sum())
print("Restaurants with zero votes (rated subset):", zero_vote_count)

summary = pd.DataFrame({
    "Metric": [
        "Total Restaurants",
        "Rated Restaurants",
        "Unrated Restaurants",
        "Unique Cities",
        "Unique Cuisine Labels",
        "Missing Cuisine Values",
        "Duplicate Records",
        "Total Missing Values",
        "Rated Restaurants with Zero Votes"
    ],

    "Value": [
        len(df),
        rated_count,
        unrated_count,
        df["City"].nunique(),
        cuisine_counts.size,
        missing_cuisine_count,
        df.duplicated().sum(),
        df.isnull().sum().sum(),
        zero_vote_count
    ]
})

summary.to_csv(
    f"{REPORT_PATH}/eda_summary.csv",
    index=False
)

# ==========================================================
# 14. GENERATED FILES
# ==========================================================

print("\n" + "=" * 65)
print("GENERATED VISUALIZATIONS")
print("=" * 65)

for file in sorted(os.listdir(FIGURE_PATH)):
    if file.endswith(".png"):
        print(file)

print("\n" + "=" * 65)
print("GENERATED REPORTS")
print("=" * 65)

for file in sorted(os.listdir(REPORT_PATH)):
    if file.endswith(".csv"):
        print(file)

print("\n" + "=" * 65)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 65)
