from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import (
    kruskal,
    spearmanr,
    mannwhitneyu
)


# ============================================================
# DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS
# TASK 04: LOCATION-BASED ANALYSIS
# PROGRAM 06: STATISTICAL ANALYSIS AND COMPARATIVE INSIGHTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "task_4_location_analysis"
    / "features"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "task_4"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "task_4"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 75)
print("DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS")
print("TASK 04: LOCATION-BASED ANALYSIS")
print("PROGRAM 06: STATISTICAL ANALYSIS AND COMPARATIVE INSIGHTS")
print("=" * 75)


# ============================================================
# 1. LOAD REQUIRED DATASETS
# ============================================================

print("\n[1] LOADING REQUIRED DATASETS")

restaurant_file = (
    FEATURE_DIR / "restaurant_location_features.csv"
)

city_file = (
    FEATURE_DIR / "city_location_features.csv"
)

locality_file = (
    FEATURE_DIR / "locality_location_features.csv"
)

for file_path in [
    restaurant_file,
    city_file,
    locality_file
]:

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required input file not found: {file_path}"
        )

restaurant_df = pd.read_csv(restaurant_file)
city_df = pd.read_csv(city_file)
locality_df = pd.read_csv(locality_file)

print("Restaurant dataset:", restaurant_df.shape)
print("City dataset:", city_df.shape)
print("Locality dataset:", locality_df.shape)


# ============================================================
# 2. VALIDATE REQUIRED COLUMNS
# ============================================================

print("\n[2] VALIDATING REQUIRED COLUMNS")

restaurant_required = [
    "Restaurant ID",
    "Country Code",
    "City",
    "Locality",
    "Aggregate rating",
    "Price range"
]

city_required = [
    "Country Code",
    "City",
    "Restaurant_Count",
    "Rated_Restaurant_Count",
    "Average_Rating",
    "Unique_Cuisine_Count",
    "Rating_Coverage_Percentage"
]

locality_required = [
    "Country Code",
    "City",
    "Locality",
    "Restaurant_Count",
    "Rated_Restaurant_Count",
    "Average_Rating"
]

for name, dataset, columns in [
    ("Restaurant", restaurant_df, restaurant_required),
    ("City", city_df, city_required),
    ("Locality", locality_df, locality_required)
]:

    missing = [
        column
        for column in columns
        if column not in dataset.columns
    ]

    if missing:
        raise ValueError(
            f"{name} dataset is missing columns: {missing}"
        )

print("[PASS] All required columns are available.")


# ============================================================
# 3. DATA CLEANING AND PREPARATION
# ============================================================

print("\n[3] PREPARING STATISTICAL DATA")

numeric_columns = [
    "Country Code",
    "Aggregate rating",
    "Price range"
]

for column in numeric_columns:

    restaurant_df[column] = pd.to_numeric(
        restaurant_df[column],
        errors="coerce"
    )

city_numeric_columns = [
    "Restaurant_Count",
    "Rated_Restaurant_Count",
    "Average_Rating",
    "Unique_Cuisine_Count",
    "Rating_Coverage_Percentage"
]

for column in city_numeric_columns:

    city_df[column] = pd.to_numeric(
        city_df[column],
        errors="coerce"
    )

locality_numeric_columns = [
    "Restaurant_Count",
    "Rated_Restaurant_Count",
    "Average_Rating"
]

for column in locality_numeric_columns:

    locality_df[column] = pd.to_numeric(
        locality_df[column],
        errors="coerce"
    )

# Remove missing and invalid rating records.
rated_df = restaurant_df.loc[
    restaurant_df["Aggregate rating"].notna()
    & restaurant_df["Aggregate rating"].between(0, 5)
].copy()

# Rating 0 represents Not Rated in this dataset.
rated_df = rated_df.loc[
    rated_df["Aggregate rating"] > 0
].copy()

if rated_df.empty:
    raise ValueError(
        "No rated restaurant records available."
    )

print("Total restaurant records:", len(restaurant_df))
print("Rated restaurant records:", len(rated_df))

print(
    "Unrated restaurant records:",
    len(restaurant_df) - len(rated_df)
)


# ============================================================
# 4. DEFINE STATISTICAL CONFIGURATION
# ============================================================

print("\n[4] DEFINING STATISTICAL CONFIGURATION")

MIN_CITY_RATED = 10
MIN_LOCALITY_RATED = 10
MIN_PRICE_GROUP = 30

MAX_CITY_GROUPS = 10
MAX_LOCALITY_GROUPS = 30

SIGNIFICANCE_LEVEL = 0.05

print(
    "Minimum rated restaurants per city:",
    MIN_CITY_RATED
)

print(
    "Minimum rated restaurants per locality:",
    MIN_LOCALITY_RATED
)

print(
    "Minimum rated restaurants per price group:",
    MIN_PRICE_GROUP
)

print(
    "Significance level:",
    SIGNIFICANCE_LEVEL
)


# ============================================================
# 5. CITY-WISE RATING DISTRIBUTION
# ============================================================

print("\n[5] ANALYZING CITY-WISE RATING DISTRIBUTION")

city_rating_data = (
    rated_df.groupby(
        ["Country Code", "City"],
        dropna=False
    )
    .agg(
        Rated_Restaurant_Count=(
            "Restaurant ID",
            "nunique"
        ),
        Average_Rating=(
            "Aggregate rating",
            "mean"
        ),
        Median_Rating=(
            "Aggregate rating",
            "median"
        ),
        Rating_Standard_Deviation=(
            "Aggregate rating",
            "std"
        ),
        Minimum_Rating=(
            "Aggregate rating",
            "min"
        ),
        Maximum_Rating=(
            "Aggregate rating",
            "max"
        )
    )
    .reset_index()
)

city_rating_data = city_rating_data.loc[
    city_rating_data["Rated_Restaurant_Count"]
    >= MIN_CITY_RATED
].copy()

city_rating_data = city_rating_data.sort_values(
    "Rated_Restaurant_Count",
    ascending=False
)

print(
    "Cities eligible for rating analysis:",
    len(city_rating_data)
)

print("\nTop 10 cities by rated restaurant count:")

print(
    city_rating_data.head(10)
    .round(3)
    .to_string(index=False)
)


# ============================================================
# 6. LOCALITY-WISE RATING DISTRIBUTION
# ============================================================

print("\n[6] ANALYZING LOCALITY-WISE RATING DISTRIBUTION")

locality_rating_data = (
    rated_df.groupby(
        [
            "Country Code",
            "City",
            "Locality"
        ],
        dropna=False
    )
    .agg(
        Rated_Restaurant_Count=(
            "Restaurant ID",
            "nunique"
        ),
        Average_Rating=(
            "Aggregate rating",
            "mean"
        ),
        Median_Rating=(
            "Aggregate rating",
            "median"
        ),
        Rating_Standard_Deviation=(
            "Aggregate rating",
            "std"
        )
    )
    .reset_index()
)

locality_rating_data = locality_rating_data.loc[
    locality_rating_data["Rated_Restaurant_Count"]
    >= MIN_LOCALITY_RATED
].copy()

locality_rating_data = locality_rating_data.sort_values(
    "Rated_Restaurant_Count",
    ascending=False
)

print(
    "Eligible localities:",
    len(locality_rating_data)
)

print("\nTop 10 localities by rated restaurant count:")

print(
    locality_rating_data.head(10)
    .round(3)
    .to_string(index=False)
)


# ============================================================
# 7. KRUSKAL-WALLIS CITY RATING COMPARISON
# ============================================================

print("\n[7] PERFORMING CITY-WISE STATISTICAL COMPARISON")

city_test_data = city_rating_data.head(
    MAX_CITY_GROUPS
).copy()

city_rating_groups = []

for _, row in city_test_data.iterrows():

    group = rated_df.loc[
        (rated_df["Country Code"] == row["Country Code"])
        & (rated_df["City"] == row["City"]),
        "Aggregate rating"
    ].dropna()

    if len(group) >= MIN_CITY_RATED:
        city_rating_groups.append(group.to_numpy())

city_kruskal_statistic = np.nan
city_kruskal_pvalue = np.nan
city_test_status = "Not performed"

if len(city_rating_groups) >= 2:

    city_statistic, city_pvalue = kruskal(
        *city_rating_groups
    )

    city_kruskal_statistic = float(city_statistic)
    city_kruskal_pvalue = float(city_pvalue)

    city_test_status = "Completed"

    print(
        "Kruskal-Wallis statistic:",
        round(city_kruskal_statistic, 5)
    )

    print(
        "P-value:",
        f"{city_kruskal_pvalue:.8f}"
    )

else:

    print(
        "Insufficient city groups for statistical comparison."
    )


# ============================================================
# 8. KRUSKAL-WALLIS LOCALITY RATING COMPARISON
# ============================================================

print("\n[8] PERFORMING LOCALITY-WISE STATISTICAL COMPARISON")

locality_test_data = locality_rating_data.head(
    MAX_LOCALITY_GROUPS
).copy()

locality_rating_groups = []

for _, row in locality_test_data.iterrows():

    group = rated_df.loc[
        (rated_df["Country Code"] == row["Country Code"])
        & (rated_df["City"] == row["City"])
        & (rated_df["Locality"] == row["Locality"]),
        "Aggregate rating"
    ].dropna()

    if len(group) >= MIN_LOCALITY_RATED:
        locality_rating_groups.append(group.to_numpy())

locality_kruskal_statistic = np.nan
locality_kruskal_pvalue = np.nan
locality_test_status = "Not performed"

if len(locality_rating_groups) >= 2:

    locality_statistic, locality_pvalue = kruskal(
        *locality_rating_groups
    )

    locality_kruskal_statistic = float(locality_statistic)
    locality_kruskal_pvalue = float(locality_pvalue)

    locality_test_status = "Completed"

    print(
        "Kruskal-Wallis statistic:",
        round(locality_kruskal_statistic, 5)
    )

    print(
        "P-value:",
        f"{locality_kruskal_pvalue:.8f}"
    )

else:

    print(
        "Insufficient locality groups for statistical comparison."
    )


# ============================================================
# 9. PRICE RANGE AND RESTAURANT RATING ANALYSIS
# ============================================================

print("\n[9] ANALYZING PRICE RANGE AND RATINGS")

price_rating_data = (
    rated_df.groupby("Price range")
    .agg(
        Rated_Restaurant_Count=(
            "Restaurant ID",
            "nunique"
        ),
        Average_Rating=(
            "Aggregate rating",
            "mean"
        ),
        Median_Rating=(
            "Aggregate rating",
            "median"
        ),
        Rating_Standard_Deviation=(
            "Aggregate rating",
            "std"
        )
    )
    .reset_index()
)

price_rating_data = price_rating_data.loc[
    price_rating_data["Rated_Restaurant_Count"]
    >= MIN_PRICE_GROUP
].copy()

print("\nRating statistics by price range:")

print(
    price_rating_data
    .round(3)
    .to_string(index=False)
)

price_rating_groups = []

for price_range in sorted(
    price_rating_data["Price range"].dropna().unique()
):

    group = rated_df.loc[
        rated_df["Price range"] == price_range,
        "Aggregate rating"
    ].dropna()

    if len(group) >= MIN_PRICE_GROUP:
        price_rating_groups.append(group.to_numpy())

price_kruskal_statistic = np.nan
price_kruskal_pvalue = np.nan
price_test_status = "Not performed"

if len(price_rating_groups) >= 2:

    price_statistic, price_pvalue = kruskal(
        *price_rating_groups
    )

    price_kruskal_statistic = float(price_statistic)
    price_kruskal_pvalue = float(price_pvalue)

    price_test_status = "Completed"

    print(
        "\nPrice range Kruskal-Wallis statistic:",
        round(price_kruskal_statistic, 5)
    )

    print(
        "P-value:",
        f"{price_kruskal_pvalue:.8f}"
    )

else:

    print(
        "Insufficient price groups for statistical comparison."
    )


# ============================================================
# 10. BUDGET VERSUS PREMIUM RATING COMPARISON
# ============================================================

print("\n[10] COMPARING BUDGET AND PREMIUM RESTAURANTS")

budget_ratings = rated_df.loc[
    rated_df["Price range"] == 1,
    "Aggregate rating"
].dropna()

premium_ratings = rated_df.loc[
    rated_df["Price range"] == 4,
    "Aggregate rating"
].dropna()

budget_premium_statistic = np.nan
budget_premium_pvalue = np.nan
budget_premium_status = "Not performed"

budget_mean = (
    float(budget_ratings.mean())
    if not budget_ratings.empty
    else np.nan
)

premium_mean = (
    float(premium_ratings.mean())
    if not premium_ratings.empty
    else np.nan
)

if (
    len(budget_ratings) >= MIN_PRICE_GROUP
    and len(premium_ratings) >= MIN_PRICE_GROUP
):

    budget_statistic, budget_pvalue = mannwhitneyu(
        budget_ratings,
        premium_ratings,
        alternative="two-sided"
    )

    budget_premium_statistic = float(budget_statistic)
    budget_premium_pvalue = float(budget_pvalue)

    budget_premium_status = "Completed"

    print(
        "Budget restaurant count:",
        len(budget_ratings)
    )

    print(
        "Premium restaurant count:",
        len(premium_ratings)
    )

    print(
        "Budget average rating:",
        round(budget_mean, 3)
    )

    print(
        "Premium average rating:",
        round(premium_mean, 3)
    )

    print(
        "Mann-Whitney U statistic:",
        round(budget_premium_statistic, 3)
    )

    print(
        "P-value:",
        f"{budget_premium_pvalue:.8f}"
    )

else:

    print(
        "Insufficient budget or premium samples."
    )



# ============================================================
# 11. CUISINE DIVERSITY AND AVERAGE RATING
# ============================================================

print("\n[11] ANALYZING CUISINE DIVERSITY AND RATINGS")

cuisine_rating_analysis = city_df[
    [
        "Country Code",
        "City",
        "Restaurant_Count",
        "Rated_Restaurant_Count",
        "Average_Rating",
        "Unique_Cuisine_Count"
    ]
].copy()

cuisine_rating_analysis = cuisine_rating_analysis.loc[
    (cuisine_rating_analysis["Rated_Restaurant_Count"] >= MIN_CITY_RATED)
    & cuisine_rating_analysis["Average_Rating"].notna()
    & cuisine_rating_analysis["Unique_Cuisine_Count"].notna()
].copy()

cuisine_rating_correlation = np.nan
cuisine_rating_pvalue = np.nan
cuisine_correlation_status = "Not performed"

if len(cuisine_rating_analysis) >= 3:

    cuisine_result = spearmanr(
        cuisine_rating_analysis["Unique_Cuisine_Count"],
        cuisine_rating_analysis["Average_Rating"]
    )

    cuisine_result_values = np.asarray(
        cuisine_result,
        dtype=float
    ).reshape(-1)

    cuisine_rating_correlation = float(
        cuisine_result_values[0]
    )

    cuisine_rating_pvalue = float(
        cuisine_result_values[1]
    )

    cuisine_correlation_status = "Completed"

    print(
        "Spearman correlation:",
        round(cuisine_rating_correlation, 5)
    )

    print(
        "P-value:",
        f"{cuisine_rating_pvalue:.8f}"
    )

else:

    print(
        "Insufficient city records for cuisine correlation."
    )



# ============================================================
# 12. RESTAURANT CONCENTRATION AND RATING COVERAGE
# ============================================================

print("\n[12] ANALYZING RESTAURANT CONCENTRATION AND RATING COVERAGE")

coverage_analysis = city_df[
    [
        "Country Code",
        "City",
        "Restaurant_Count",
        "Rated_Restaurant_Count",
        "Rating_Coverage_Percentage"
    ]
].copy()

coverage_analysis = coverage_analysis.dropna(
    subset=[
        "Restaurant_Count",
        "Rating_Coverage_Percentage"
    ]
)

concentration_coverage_correlation = np.nan
concentration_coverage_pvalue = np.nan
coverage_correlation_status = "Not performed"

if len(coverage_analysis) >= 3:

    coverage_result = spearmanr(
        coverage_analysis["Restaurant_Count"],
        coverage_analysis["Rating_Coverage_Percentage"]
    )

    coverage_result_values = np.asarray(
        coverage_result,
        dtype=float
    ).reshape(-1)

    concentration_coverage_correlation = float(
        coverage_result_values[0]
    )

    concentration_coverage_pvalue = float(
        coverage_result_values[1]
    )

    coverage_correlation_status = "Completed"

    print(
        "Spearman correlation:",
        round(concentration_coverage_correlation, 5)
    )

    print(
        "P-value:",
        f"{concentration_coverage_pvalue:.8f}"
    )

else:

    print(
        "Insufficient city records for coverage correlation."
    )



# ============================================================
# 13. RESTAURANT POPULARITY AND RATINGS
# ============================================================

print("\n[13] ANALYZING RESTAURANT VOTES AND RATINGS")

votes_analysis = None

votes_rating_correlation = np.nan
votes_rating_pvalue = np.nan
votes_correlation_status = "Not performed"

if "Votes" in rated_df.columns:

    rated_df["Votes"] = pd.to_numeric(
        rated_df["Votes"],
        errors="coerce"
    )

    votes_analysis = rated_df.dropna(
        subset=[
            "Votes",
            "Aggregate rating"
        ]
    ).copy()

    if len(votes_analysis) >= 3:

        votes_result = spearmanr(
            votes_analysis["Votes"],
            votes_analysis["Aggregate rating"]
        )

        votes_result_values = np.asarray(
            votes_result,
            dtype=float
        ).reshape(-1)

        votes_rating_correlation = float(
            votes_result_values[0]
        )

        votes_rating_pvalue = float(
            votes_result_values[1]
        )

        votes_correlation_status = "Completed"

        print(
            "Restaurants included:",
            len(votes_analysis)
        )

        print(
            "Spearman correlation:",
            round(votes_rating_correlation, 5)
        )

        print(
            "P-value:",
            f"{votes_rating_pvalue:.8f}"
        )

    else:

        print(
            "Insufficient records for votes correlation."
        )

else:

    print(
        "Votes column unavailable. Analysis skipped."
    )


# ============================================================
# 14. CREATE STATISTICAL TEST SUMMARY
# ============================================================

print("\n[14] GENERATING STATISTICAL TEST SUMMARY")

test_results = [
    {
        "Analysis": "City Rating Comparison",
        "Test": "Kruskal-Wallis",
        "Statistic": city_kruskal_statistic,
        "P_Value": city_kruskal_pvalue,
        "Status": city_test_status
    },
    {
        "Analysis": "Locality Rating Comparison",
        "Test": "Kruskal-Wallis",
        "Statistic": locality_kruskal_statistic,
        "P_Value": locality_kruskal_pvalue,
        "Status": locality_test_status
    },
    {
        "Analysis": "Price Range Rating Comparison",
        "Test": "Kruskal-Wallis",
        "Statistic": price_kruskal_statistic,
        "P_Value": price_kruskal_pvalue,
        "Status": price_test_status
    },
    {
        "Analysis": "Budget vs Premium Rating",
        "Test": "Mann-Whitney U",
        "Statistic": budget_premium_statistic,
        "P_Value": budget_premium_pvalue,
        "Status": budget_premium_status
    },
    {
        "Analysis": "Cuisine Diversity vs Rating",
        "Test": "Spearman Correlation",
        "Statistic": cuisine_rating_correlation,
        "P_Value": cuisine_rating_pvalue,
        "Status": cuisine_correlation_status
    },
    {
        "Analysis": "Restaurant Count vs Rating Coverage",
        "Test": "Spearman Correlation",
        "Statistic": concentration_coverage_correlation,
        "P_Value": concentration_coverage_pvalue,
        "Status": coverage_correlation_status
    },
    {
        "Analysis": "Votes vs Rating",
        "Test": "Spearman Correlation",
        "Statistic": votes_rating_correlation,
        "P_Value": votes_rating_pvalue,
        "Status": votes_correlation_status
    }
]

statistical_results_df = pd.DataFrame(
    test_results
)

statistical_results_df["Significant_At_0_05"] = (
    statistical_results_df["P_Value"]
    .lt(SIGNIFICANCE_LEVEL)
    .fillna(False)
)

statistical_results_df["Interpretation"] = np.select(
    [
        statistical_results_df["Status"].ne("Completed"),
        statistical_results_df["P_Value"].isna(),
        statistical_results_df["Significant_At_0_05"]
    ],
    [
        "Test not completed",
        "Result undefined",
        "Statistically significant at 0.05"
    ],
    default="Not statistically significant at 0.05"
)

print(
    statistical_results_df[
        [
            "Analysis",
            "Test",
            "Statistic",
            "P_Value",
            "Interpretation"
        ]
    ].to_string(index=False)
)


# ============================================================
# 15. GENERATE DESCRIPTIVE INSIGHTS
# ============================================================

print("\n[15] GENERATING DESCRIPTIVE INSIGHTS")

insights = []

if not city_rating_data.empty:

    highest_city = city_rating_data.sort_values(
        "Average_Rating",
        ascending=False
    ).iloc[0]

    insights.append({
        "Category": "City Rating",
        "Finding": (
            f"{highest_city['City']} has the highest average rating "
            "among cities meeting the minimum sample threshold."
        ),
        "Value": round(
            float(highest_city["Average_Rating"]),
            3
        )
    })

if not locality_rating_data.empty:

    highest_locality = locality_rating_data.sort_values(
        "Average_Rating",
        ascending=False
    ).iloc[0]

    insights.append({
        "Category": "Locality Rating",
        "Finding": (
            f"{highest_locality['Locality']}, "
            f"{highest_locality['City']} has the highest "
            "average rating among eligible localities."
        ),
        "Value": round(
            float(highest_locality["Average_Rating"]),
            3
        )
    })

if not price_rating_data.empty:

    highest_price_group = price_rating_data.sort_values(
        "Average_Rating",
        ascending=False
    ).iloc[0]

    insights.append({
        "Category": "Price Range",
        "Finding": (
            f"Price range {int(highest_price_group['Price range'])} "
            "has the highest average rating among eligible price groups."
        ),
        "Value": round(
            float(highest_price_group["Average_Rating"]),
            3
        )
    })

completed_tests = int(
    (statistical_results_df["Status"] == "Completed").sum()
)

significant_count = int(
    statistical_results_df["Significant_At_0_05"].sum()
)

insights.append({
    "Category": "Statistical Tests",
    "Finding": (
        f"{significant_count} of {completed_tests} completed tests "
        "have p-values below 0.05."
    ),
    "Value": significant_count
})

insights_df = pd.DataFrame(insights)

for _, row in insights_df.iterrows():

    print(
        f"\n{row['Category']}: {row['Finding']}"
    )


# ============================================================
# 16. VISUALIZE CITY-WISE RATING COMPARISON
# ============================================================

print("\n[16] GENERATING CITY RATING VISUALIZATION")

plot_city_data = city_rating_data.sort_values(
    "Average_Rating",
    ascending=False
).head(15)

fig, ax = plt.subplots(figsize=(13, 7))

ax.barh(
    plot_city_data["City"],
    plot_city_data["Average_Rating"]
)

ax.set_title(
    "Average Restaurant Ratings by City",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel("Average Rating")
ax.set_ylabel("City")
ax.invert_yaxis()

ax.set_xlim(0, 5)

ax.grid(
    axis="x",
    alpha=0.25
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "07_city_rating_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("[SAVED] 07_city_rating_comparison.png")


# ============================================================
# 17. VISUALIZE PRICE RANGE AND RATINGS
# ============================================================

print("\n[17] GENERATING PRICE-RATING VISUALIZATION")

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(
    price_rating_data["Price range"].astype(str),
    price_rating_data["Average_Rating"]
)

ax.set_title(
    "Average Restaurant Rating by Price Range",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel("Price Range")
ax.set_ylabel("Average Rating")

ax.set_ylim(0, 5)

ax.grid(
    axis="y",
    alpha=0.25
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "08_price_rating_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("[SAVED] 08_price_rating_comparison.png")


# ============================================================
# 18. VISUALIZE CUISINE DIVERSITY AND RATINGS
# ============================================================

print("\n[18] GENERATING CUISINE DIVERSITY VISUALIZATION")

fig, ax = plt.subplots(figsize=(10, 7))

ax.scatter(
    cuisine_rating_analysis["Unique_Cuisine_Count"],
    cuisine_rating_analysis["Average_Rating"],
    s=(
        cuisine_rating_analysis["Restaurant_Count"]
        .clip(lower=1)
        * 1.5
    ),
    alpha=0.7
)

ax.set_title(
    "Cuisine Diversity vs Average Restaurant Rating",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel("Number of Distinct Cuisine Labels")
ax.set_ylabel("Average Rating")

ax.set_ylim(0, 5)

ax.grid(
    True,
    alpha=0.25
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "09_cuisine_diversity_rating.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("[SAVED] 09_cuisine_diversity_rating.png")


# ============================================================
# 19. VISUALIZE RESTAURANT CONCENTRATION AND COVERAGE
# ============================================================

print("\n[19] GENERATING CONCENTRATION-COVERAGE VISUALIZATION")

fig, ax = plt.subplots(figsize=(10, 7))

ax.scatter(
    coverage_analysis["Restaurant_Count"],
    coverage_analysis["Rating_Coverage_Percentage"],
    alpha=0.7
)

ax.set_title(
    "Restaurant Concentration vs Rating Coverage",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel("Restaurant Count")
ax.set_ylabel("Rating Coverage (%)")

ax.grid(
    True,
    alpha=0.25
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "10_restaurant_concentration_rating_coverage.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print(
    "[SAVED] 10_restaurant_concentration_rating_coverage.png"
)


# ============================================================
# 20. SAVE ANALYSIS DATASETS
# ============================================================

print("\n[20] SAVING STATISTICAL ANALYSIS DATASETS")

analysis_outputs = {
    "location_statistical_city_ratings.csv": city_rating_data,
    "location_statistical_locality_ratings.csv": locality_rating_data,
    "location_statistical_price_ratings.csv": price_rating_data,
    "location_statistical_cuisine_rating.csv": cuisine_rating_analysis,
    "location_statistical_rating_coverage.csv": coverage_analysis,
    "location_statistical_test_results.csv": statistical_results_df,
    "location_statistical_insights.csv": insights_df
}

if votes_analysis is not None:

    analysis_outputs[
        "location_statistical_votes_rating.csv"
    ] = votes_analysis[
        [
            "Restaurant ID",
            "Votes",
            "Aggregate rating"
        ]
    ].copy()

for filename, dataset in analysis_outputs.items():

    dataset.to_csv(
        REPORT_DIR / filename,
        index=False
    )

    print(
        f"[SAVED] {filename}: {dataset.shape}"
    )


# ============================================================
# 21. SAVE STATISTICAL ANALYSIS SUMMARY
# ============================================================

print("\n[21] SAVING STATISTICAL ANALYSIS SUMMARY")

summary_data = [
    ("Total Restaurants", len(restaurant_df)),
    ("Rated Restaurants", len(rated_df)),
    ("Eligible Cities", len(city_rating_data)),
    ("Eligible Localities", len(locality_rating_data)),
    ("Eligible Price Groups", len(price_rating_data)),
    ("City Groups in Kruskal-Wallis Test", len(city_rating_groups)),
    (
        "Locality Groups in Kruskal-Wallis Test",
        len(locality_rating_groups)
    ),
    ("Price Groups in Kruskal-Wallis Test", len(price_rating_groups)),
    ("Budget Restaurant Sample Size", len(budget_ratings)),
    ("Premium Restaurant Sample Size", len(premium_ratings)),
    ("Cuisine Correlation Sample Size", len(cuisine_rating_analysis)),
    ("Coverage Correlation Sample Size", len(coverage_analysis)),
    (
        "Votes Correlation Sample Size",
        len(votes_analysis) if votes_analysis is not None else 0
    ),
    (
        "Completed Statistical Tests",
        completed_tests
    ),
    (
        "Significant Statistical Tests",
        significant_count
    )
]

summary_df = pd.DataFrame(
    summary_data,
    columns=["Metric", "Value"]
)

summary_df.to_csv(
    REPORT_DIR / "location_statistical_analysis_summary.csv",
    index=False
)


# ============================================================
# 22. SAVE TEXT REPORT
# ============================================================

print("\n[22] GENERATING STATISTICAL ANALYSIS REPORT")

report_path = (
    REPORT_DIR
    / "location_statistical_analysis_report.txt"
)

with open(
    report_path,
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
        "PROGRAM 06: STATISTICAL ANALYSIS AND COMPARATIVE INSIGHTS\n\n"
    )

    file.write("=" * 70 + "\n")
    file.write("ANALYSIS SUMMARY\n")
    file.write("=" * 70 + "\n\n")

    for metric, value in summary_data:

        file.write(
            f"{metric}: {value}\n"
        )

    file.write("\n")
    file.write("=" * 70 + "\n")
    file.write("STATISTICAL TEST RESULTS\n")
    file.write("=" * 70 + "\n\n")

    file.write(
        statistical_results_df.to_string(index=False)
    )

    file.write("\n\n")
    file.write("=" * 70 + "\n")
    file.write("DESCRIPTIVE INSIGHTS\n")
    file.write("=" * 70 + "\n\n")

    for _, row in insights_df.iterrows():

        file.write(
            f"{row['Category']}: {row['Finding']}\n"
        )

    file.write("\n")
    file.write("=" * 70 + "\n")
    file.write("INTERPRETATION LIMITATIONS\n")
    file.write("=" * 70 + "\n\n")

    file.write(
        "1. Statistical significance does not necessarily indicate "
        "practical importance.\n"
    )

    file.write(
        "2. Correlation does not establish causation.\n"
    )

    file.write(
        "3. Kruskal-Wallis tests indicate whether group distributions "
        "differ, but do not identify which specific groups differ.\n"
    )

    file.write(
        "4. Multiple group comparisons may require post-hoc testing "
        "and multiple-testing correction.\n"
    )

    file.write(
        "5. Groups are subject to minimum sample-size filters.\n"
    )

    file.write(
        "6. City and locality comparisons are descriptive of the "
        "available dataset and should not automatically be generalized "
        "to all restaurants in those locations.\n"
    )


# ============================================================
# 23. FINAL VALIDATION
# ============================================================

print("\n[23] FINAL VALIDATION")

completed_results = statistical_results_df.loc[
    statistical_results_df["Status"] == "Completed"
]

valid_pvalues = (
    completed_results["P_Value"].dropna()
    .between(0, 1).all()
)

checks = {
    "Restaurant dataset loaded": len(restaurant_df) > 0,

    "Rated restaurants available": len(rated_df) > 0,

    "City rating analysis generated": not city_rating_data.empty,

    "Locality rating analysis generated": (
        not locality_rating_data.empty
    ),

    "Price rating analysis generated": not price_rating_data.empty,

    "Cuisine rating analysis generated": (
        not cuisine_rating_analysis.empty
    ),

    "Rating coverage analysis generated": (
        not coverage_analysis.empty
    ),

    "Statistical results generated": (
        not statistical_results_df.empty
    ),

    "Statistical insight report generated": (
        not insights_df.empty
    ),

    "Completed test p-values are valid": valid_pvalues,

    "All rating values are within valid range": (
        rated_df["Aggregate rating"].between(0, 5).all()
    ),

    "City sample counts are valid": (
        city_rating_data["Rated_Restaurant_Count"]
        .ge(MIN_CITY_RATED).all()
    ),

    "Locality sample counts are valid": (
        locality_rating_data["Rated_Restaurant_Count"]
        .ge(MIN_LOCALITY_RATED).all()
    ),

    "Price group sample counts are valid": (
        price_rating_data["Rated_Restaurant_Count"]
        .ge(MIN_PRICE_GROUP).all()
    )
}

for check, passed in checks.items():

    print(
        f"{'[PASS]' if passed else '[FAIL]'} {check}"
    )

if not all(checks.values()):

    raise RuntimeError(
        "One or more statistical analysis validations failed."
    )


# ============================================================
# 24. VALIDATE OUTPUT FILES
# ============================================================

print("\n[24] VALIDATING OUTPUT FILES")

expected_files = [
    REPORT_DIR / filename
    for filename in analysis_outputs
]

expected_files.extend([
    REPORT_DIR / "location_statistical_analysis_summary.csv",
    REPORT_DIR / "location_statistical_analysis_report.txt",
    FIGURE_DIR / "07_city_rating_comparison.png",
    FIGURE_DIR / "08_price_rating_comparison.png",
    FIGURE_DIR / "09_cuisine_diversity_rating.png",
    FIGURE_DIR / "10_restaurant_concentration_rating_coverage.png"
])

all_outputs_exist = True

for output_file in expected_files:

    exists = (
        output_file.exists()
        and output_file.stat().st_size > 0
    )

    print(
        f"{'[PASS]' if exists else '[FAIL]'} {output_file.name}"
    )

    if not exists:
        all_outputs_exist = False

if not all_outputs_exist:

    raise RuntimeError(
        "One or more statistical analysis output files are missing."
    )


# ============================================================
# 25. EXECUTION SUMMARY
# ============================================================

print("\n[25] EXECUTION SUMMARY")

print("Total restaurants:", len(restaurant_df))
print("Rated restaurants:", len(rated_df))
print("Eligible cities:", len(city_rating_data))
print("Eligible localities:", len(locality_rating_data))

print(
    "Completed statistical tests:",
    completed_tests
)

print(
    "Significant statistical tests:",
    significant_count
)

print("Reports directory:", REPORT_DIR)
print("Figures directory:", FIGURE_DIR)

print("\n" + "=" * 75)
print("PROGRAM 06 EXECUTION COMPLETED SUCCESSFULLY")
print("=" * 75)
