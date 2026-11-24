from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS
# TASK 04: LOCATION-BASED ANALYSIS
# PROGRAM 07: LOCATION RECOMMENDATION AND DECISION SUPPORT
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
print("PROGRAM 07: LOCATION RECOMMENDATION AND DECISION SUPPORT")
print("=" * 75)


# ============================================================
# 1. LOAD ENGINEERED LOCATION DATASETS
# ============================================================

print("\n[1] LOADING ENGINEERED LOCATION DATASETS")

city_file = FEATURE_DIR / "city_location_features.csv"
locality_file = FEATURE_DIR / "locality_location_features.csv"
restaurant_file = FEATURE_DIR / "restaurant_location_features.csv"

required_files = [
    city_file,
    locality_file,
    restaurant_file
]

for file_path in required_files:

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required input file not found: {file_path}"
        )

city_df = pd.read_csv(city_file)
locality_df = pd.read_csv(locality_file)
restaurant_df = pd.read_csv(restaurant_file)

print("City dataset:", city_df.shape)
print("Locality dataset:", locality_df.shape)
print("Restaurant dataset:", restaurant_df.shape)


# ============================================================
# 2. VALIDATE REQUIRED COLUMNS
# ============================================================

print("\n[2] VALIDATING REQUIRED COLUMNS")

city_required = [
    "Country Code",
    "City",
    "Restaurant_Count",
    "Rated_Restaurant_Count",
    "Average_Rating"
]

locality_required = [
    "Country Code",
    "City",
    "Locality",
    "Restaurant_Count",
    "Rated_Restaurant_Count",
    "Average_Rating"
]

restaurant_required = [
    "Restaurant ID",
    "Restaurant Name",
    "Country Code",
    "City",
    "Locality",
    "Aggregate rating"
]

for name, dataset, required_columns in [
    ("City", city_df, city_required),
    ("Locality", locality_df, locality_required),
    ("Restaurant", restaurant_df, restaurant_required)
]:

    missing_columns = [
        column
        for column in required_columns
        if column not in dataset.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{name} dataset is missing columns: "
            f"{missing_columns}"
        )

print("[PASS] Required columns are available.")


# ============================================================
# 3. DATA CLEANING AND GLOBAL RATING BASELINE
# ============================================================

print("\n[3] PREPARING LOCATION DATA")

numeric_columns = [
    "Country Code",
    "Restaurant_Count",
    "Rated_Restaurant_Count",
    "Average_Rating"
]

for dataset in [city_df, locality_df]:

    for column in numeric_columns:

        dataset[column] = pd.to_numeric(
            dataset[column],
            errors="coerce"
        )

    if "Unique_Cuisine_Count" in dataset.columns:

        dataset["Unique_Cuisine_Count"] = pd.to_numeric(
            dataset["Unique_Cuisine_Count"],
            errors="coerce"
        )

    dataset["Restaurant_Count"] = (
        dataset["Restaurant_Count"].fillna(0)
    )

    dataset["Rated_Restaurant_Count"] = (
        dataset["Rated_Restaurant_Count"].fillna(0)
    )

    dataset["Average_Rating"] = (
        dataset["Average_Rating"].where(
            dataset["Average_Rating"].between(0, 5)
        )
    )

    # Recalculate rating coverage from counts.
    dataset["Calculated_Rating_Coverage"] = np.where(
        dataset["Restaurant_Count"] > 0,
        (
            dataset["Rated_Restaurant_Count"]
            / dataset["Restaurant_Count"]
        ) * 100,
        0
    )

    # Coverage cannot exceed 100%.
    dataset["Calculated_Rating_Coverage"] = (
        dataset["Calculated_Rating_Coverage"].clip(0, 100)
    )

restaurant_df["Aggregate rating"] = pd.to_numeric(
    restaurant_df["Aggregate rating"],
    errors="coerce"
)

# Use valid rated restaurants to calculate the global baseline.
rated_restaurant_data = restaurant_df.loc[
    restaurant_df["Aggregate rating"].between(0, 5)
    & (restaurant_df["Aggregate rating"] > 0)
].copy()

if rated_restaurant_data.empty:

    raise ValueError(
        "No valid rated restaurants available for rating baseline."
    )

GLOBAL_AVERAGE_RATING = float(
    rated_restaurant_data["Aggregate rating"].mean()
)

print("City records:", len(city_df))
print("Locality records:", len(locality_df))
print("Restaurant records:", len(restaurant_df))

print(
    "Global average restaurant rating:",
    round(GLOBAL_AVERAGE_RATING, 4)
)

print(
    "Restaurants used for rating baseline:",
    len(rated_restaurant_data)
)


# ============================================================
# 4. DEFINE RECOMMENDATION CONFIGURATION
# ============================================================

print("\n[4] DEFINING RECOMMENDATION CONFIGURATION")

MIN_CITY_RESTAURANTS = 10

MIN_LOCALITY_RESTAURANTS = 5

MIN_RATED_RESTAURANTS = 10

TOP_N = 20

# Prior strength for sample-size-adjusted ratings.
RATING_PRIOR_STRENGTH = 30

# Recommendation component weights.
RECOMMENDATION_WEIGHTS = {
    "Rating_Score": 0.40,
    "Cuisine_Diversity_Score": 0.20,
    "Rating_Coverage_Score": 0.20,
    "Restaurant_Availability_Score": 0.20
}

print(
    "Minimum restaurants per city:",
    MIN_CITY_RESTAURANTS
)

print(
    "Minimum restaurants per locality:",
    MIN_LOCALITY_RESTAURANTS
)

print(
    "Minimum rated restaurants:",
    MIN_RATED_RESTAURANTS
)

print(
    "Rating prior strength:",
    RATING_PRIOR_STRENGTH
)

print(
    "Global average rating:",
    round(GLOBAL_AVERAGE_RATING, 4)
)

print("Recommendation weights:")

for component, weight in RECOMMENDATION_WEIGHTS.items():

    print(
        f"{component}: {weight:.0%}"
    )


# ============================================================
# 5. DEFINE LOCATION RANKING FUNCTIONS
# ============================================================

print("\n[5] DEFINING LOCATION RANKING FUNCTIONS")


def calculate_percentile_score(
    series,
    higher_is_better=True
):

    values = pd.to_numeric(
        series,
        errors="coerce"
    )

    if values.notna().sum() == 0:

        return pd.Series(
            np.nan,
            index=series.index,
            dtype=float
        )

    scores = values.rank(
        method="average",
        pct=True,
        ascending=higher_is_better
    ) * 100

    return scores.astype(float)


def build_location_ranking(
    dataset,
    location_type,
    minimum_restaurants
):

    data = dataset.copy()

    # --------------------------------------------------------
    # FILTER ELIGIBLE LOCATIONS
    # --------------------------------------------------------

    data = data.loc[
        (data["Restaurant_Count"] >= minimum_restaurants)
        & (
            data["Rated_Restaurant_Count"]
            >= MIN_RATED_RESTAURANTS
        )
        & data["Average_Rating"].notna()
    ].copy()

    if data.empty:

        raise ValueError(
            f"No eligible {location_type} records available."
        )

    # --------------------------------------------------------
    # COMPONENT 1: SAMPLE-SIZE-ADJUSTED RATING
    # --------------------------------------------------------

    # Bayesian-style shrinkage:
    #
    # Adjusted Rating =
    # (n * observed average + m * global average) / (n + m)
    #
    # n = rated restaurant count for the location
    # m = prior strength
    #
    # Small samples are adjusted more strongly toward
    # the global average.
    #
    # Large samples remain closer to their observed average.

    rated_counts = data["Rated_Restaurant_Count"].clip(
        lower=0
    )

    data["Adjusted_Average_Rating"] = (
        (
            rated_counts * data["Average_Rating"]
            +
            RATING_PRIOR_STRENGTH * GLOBAL_AVERAGE_RATING
        )
        /
        (
            rated_counts + RATING_PRIOR_STRENGTH
        )
    )

    data["Adjusted_Average_Rating"] = (
        data["Adjusted_Average_Rating"].clip(0, 5)
    )

    data["Rating_Score"] = calculate_percentile_score(
        data["Adjusted_Average_Rating"]
    )

    # --------------------------------------------------------
    # COMPONENT 2: CUISINE DIVERSITY
    # --------------------------------------------------------

    available_components = [
        "Rating_Score",
        "Rating_Coverage_Score",
        "Restaurant_Availability_Score"
    ]

    if (
        "Unique_Cuisine_Count" in data.columns
        and data["Unique_Cuisine_Count"].notna().any()
    ):

        data["Cuisine_Diversity_Score"] = (
            calculate_percentile_score(
                data["Unique_Cuisine_Count"]
            )
        )

        available_components.append(
            "Cuisine_Diversity_Score"
        )

    else:

        data["Cuisine_Diversity_Score"] = np.nan

    # --------------------------------------------------------
    # COMPONENT 3: RATING COVERAGE
    # --------------------------------------------------------

    data["Rating_Coverage_Score"] = (
        calculate_percentile_score(
            data["Calculated_Rating_Coverage"]
        )
    )

    # --------------------------------------------------------
    # COMPONENT 4: RESTAURANT AVAILABILITY
    # --------------------------------------------------------

    # Log transformation prevents large cities from
    # dominating solely because of their restaurant count.

    data["Log_Restaurant_Count"] = np.log1p(
        data["Restaurant_Count"].clip(lower=0)
    )

    data["Restaurant_Availability_Score"] = (
        calculate_percentile_score(
            data["Log_Restaurant_Count"]
        )
    )

    # --------------------------------------------------------
    # NORMALIZE ACTIVE COMPONENT WEIGHTS
    # --------------------------------------------------------

    active_weights = {
        component: RECOMMENDATION_WEIGHTS[component]
        for component in available_components
    }

    total_weight = sum(active_weights.values())

    if total_weight <= 0:

        raise ValueError(
            "No valid recommendation weights available."
        )

    normalized_weights = {
        component: weight / total_weight
        for component, weight in active_weights.items()
    }

    # --------------------------------------------------------
    # CALCULATE COMPOSITE RECOMMENDATION SCORE
    # --------------------------------------------------------

    data["Recommendation_Score"] = 0.0

    for component, weight in normalized_weights.items():

        data["Recommendation_Score"] += (
            data[component].fillna(0) * weight
        )

    data["Recommendation_Score"] = (
        data["Recommendation_Score"].clip(0, 100)
    )

    # --------------------------------------------------------
    # ASSIGN RECOMMENDATION TIERS
    # --------------------------------------------------------

    data["Recommendation_Tier"] = np.select(
        [
            data["Recommendation_Score"] >= 80,
            data["Recommendation_Score"] >= 60,
            data["Recommendation_Score"] >= 40
        ],
        [
            "Highly Recommended",
            "Recommended",
            "Moderately Recommended"
        ],
        default="Lower Priority"
    )

    # --------------------------------------------------------
    # ASSIGN RECOMMENDATION RANKS
    # --------------------------------------------------------

    data = data.sort_values(
        [
            "Recommendation_Score",
            "Adjusted_Average_Rating",
            "Restaurant_Count"
        ],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    data["Recommendation_Rank"] = np.arange(
        1,
        len(data) + 1
    )

    data["Location_Type"] = location_type

    return data, normalized_weights


# ============================================================
# 6. GENERATE CITY RECOMMENDATIONS
# ============================================================

print("\n[6] GENERATING CITY RECOMMENDATIONS")

city_recommendations, city_weights = (
    build_location_ranking(
        city_df,
        "City",
        MIN_CITY_RESTAURANTS
    )
)

print(
    "Eligible cities:",
    len(city_recommendations)
)

city_display_columns = [
    "Recommendation_Rank",
    "Country Code",
    "City",
    "Restaurant_Count",
    "Rated_Restaurant_Count",
    "Average_Rating",
    "Adjusted_Average_Rating",
    "Calculated_Rating_Coverage",
    "Recommendation_Score",
    "Recommendation_Tier"
]

print("\nTop 10 recommended cities:")

print(
    city_recommendations[
        city_display_columns
    ].head(10).to_string(index=False)
)


# ============================================================
# 7. GENERATE LOCALITY RECOMMENDATIONS
# ============================================================

print("\n[7] GENERATING LOCALITY RECOMMENDATIONS")

locality_recommendations, locality_weights = (
    build_location_ranking(
        locality_df,
        "Locality",
        MIN_LOCALITY_RESTAURANTS
    )
)

print(
    "Eligible localities:",
    len(locality_recommendations)
)

locality_display_columns = [
    "Recommendation_Rank",
    "Country Code",
    "City",
    "Locality",
    "Restaurant_Count",
    "Rated_Restaurant_Count",
    "Average_Rating",
    "Adjusted_Average_Rating",
    "Calculated_Rating_Coverage",
    "Recommendation_Score",
    "Recommendation_Tier"
]

print("\nTop 10 recommended localities:")

print(
    locality_recommendations[
        locality_display_columns
    ].head(10).to_string(index=False)
)


# ============================================================
# 8. ANALYZE RECOMMENDATION TIERS
# ============================================================

print("\n[8] ANALYZING RECOMMENDATION TIERS")


def create_tier_summary(dataset):

    summary = (
        dataset.groupby("Recommendation_Tier")
        .agg(
            Location_Count=(
                "Recommendation_Tier",
                "size"
            ),
            Average_Recommendation_Score=(
                "Recommendation_Score",
                "mean"
            ),
            Average_Original_Rating=(
                "Average_Rating",
                "mean"
            ),
            Average_Adjusted_Rating=(
                "Adjusted_Average_Rating",
                "mean"
            ),
            Total_Restaurants=(
                "Restaurant_Count",
                "sum"
            )
        )
        .reset_index()
    )

    return summary


city_tier_summary = create_tier_summary(
    city_recommendations
)

locality_tier_summary = create_tier_summary(
    locality_recommendations
)

print("\nCity recommendation tiers:")

print(
    city_tier_summary.to_string(index=False)
)

print("\nLocality recommendation tiers:")

print(
    locality_tier_summary.to_string(index=False)
)


# ============================================================
# 9. IDENTIFY HIGH-QUALITY LOCATIONS
# ============================================================

print("\n[9] IDENTIFYING HIGH-QUALITY LOCATIONS")

high_quality_cities = city_recommendations.loc[
    city_recommendations["Recommendation_Tier"]
    == "Highly Recommended"
].copy()

high_quality_localities = locality_recommendations.loc[
    locality_recommendations["Recommendation_Tier"]
    == "Highly Recommended"
].copy()

print(
    "Highly recommended cities:",
    len(high_quality_cities)
)

print(
    "Highly recommended localities:",
    len(high_quality_localities)
)


# ============================================================
# 10. IDENTIFY LOCATIONS WITH STRONG RATING COVERAGE
# ============================================================

print("\n[10] ANALYZING RATING COVERAGE")

high_coverage_cities = city_recommendations.loc[
    city_recommendations["Calculated_Rating_Coverage"] >= 80
].copy()

high_coverage_localities = locality_recommendations.loc[
    locality_recommendations["Calculated_Rating_Coverage"] >= 80
].copy()

print(
    "Cities with at least 80% rating coverage:",
    len(high_coverage_cities)
)

print(
    "Localities with at least 80% rating coverage:",
    len(high_coverage_localities)
)


# ============================================================
# 11. GENERATE LOCATION COMPARISON DATA
# ============================================================

print("\n[11] GENERATING LOCATION COMPARISON DATA")

city_comparison_columns = [
    "Country Code",
    "City",
    "Restaurant_Count",
    "Rated_Restaurant_Count",
    "Average_Rating",
    "Adjusted_Average_Rating",
    "Calculated_Rating_Coverage",
    "Cuisine_Diversity_Score",
    "Rating_Score",
    "Rating_Coverage_Score",
    "Restaurant_Availability_Score",
    "Recommendation_Score",
    "Recommendation_Rank",
    "Recommendation_Tier"
]

locality_comparison_columns = [
    "Country Code",
    "City",
    "Locality",
    "Restaurant_Count",
    "Rated_Restaurant_Count",
    "Average_Rating",
    "Adjusted_Average_Rating",
    "Calculated_Rating_Coverage",
    "Cuisine_Diversity_Score",
    "Rating_Score",
    "Rating_Coverage_Score",
    "Restaurant_Availability_Score",
    "Recommendation_Score",
    "Recommendation_Rank",
    "Recommendation_Tier"
]

city_comparison = city_recommendations[
    city_comparison_columns
].copy()

locality_comparison = locality_recommendations[
    locality_comparison_columns
].copy()


# ============================================================
# 12. GENERATE RESTAURANT SHORTLISTS
# ============================================================

print("\n[12] GENERATING RESTAURANT SHORTLISTS")

rated_restaurants = restaurant_df.loc[
    restaurant_df["Aggregate rating"].between(0, 5)
    & (restaurant_df["Aggregate rating"] > 0)
].copy()

# Select the highest-ranked city.
top_city = city_recommendations.iloc[0]

top_city_restaurants = rated_restaurants.loc[
    (
        rated_restaurants["Country Code"]
        == top_city["Country Code"]
    )
    & (
        rated_restaurants["City"]
        == top_city["City"]
    )
].copy()

if not top_city_restaurants.empty:

    top_city_restaurants = (
        top_city_restaurants.sort_values(
            [
                "Aggregate rating",
                "Votes"
            ] if "Votes" in top_city_restaurants.columns
            else ["Aggregate rating"],
            ascending=False
        ).head(TOP_N)
    )

print(
    "Top recommended city:",
    top_city["City"]
)

print(
    "Restaurants in shortlist:",
    len(top_city_restaurants)
)

restaurant_shortlist_columns = [
    column
    for column in [
        "Restaurant ID",
        "Restaurant Name",
        "Country Code",
        "City",
        "Locality",
        "Cuisines",
        "Aggregate rating",
        "Votes",
        "Price range"
    ]
    if column in top_city_restaurants.columns
]

if not top_city_restaurants.empty:

    print("\nTop restaurants in the highest-ranked city:")

    print(
        top_city_restaurants[
            restaurant_shortlist_columns
        ].head(10).to_string(index=False)
    )


# ============================================================
# 13. GENERATE LOCATION DECISION INSIGHTS
# ============================================================

print("\n[13] GENERATING LOCATION DECISION INSIGHTS")

insights = []

best_city = city_recommendations.iloc[0]

best_locality = locality_recommendations.iloc[0]

insights.append({
    "Category": "Top City Recommendation",
    "Location": best_city["City"],
    "Score": round(
        float(best_city["Recommendation_Score"]),
        3
    ),
    "Reason": (
        "Highest composite recommendation score among "
        "eligible cities using sample-size-adjusted ratings."
    )
})

insights.append({
    "Category": "Top Locality Recommendation",
    "Location": (
        f"{best_locality['Locality']}, "
        f"{best_locality['City']}"
    ),
    "Score": round(
        float(best_locality["Recommendation_Score"]),
        3
    ),
    "Reason": (
        "Highest composite recommendation score among "
        "eligible localities using sample-size-adjusted ratings."
    )
})

if not high_coverage_cities.empty:

    coverage_city = high_coverage_cities.sort_values(
        "Calculated_Rating_Coverage",
        ascending=False
    ).iloc[0]

    insights.append({
        "Category": "Highest City Rating Coverage",
        "Location": coverage_city["City"],
        "Score": round(
            float(
                coverage_city["Calculated_Rating_Coverage"]
            ),
            3
        ),
        "Reason": (
            "Highest rating coverage among eligible cities "
            "with at least 80 percent coverage."
        )
    })

if not high_quality_localities.empty:

    quality_locality = high_quality_localities.iloc[0]

    insights.append({
        "Category": "Highly Recommended Locality",
        "Location": (
            f"{quality_locality['Locality']}, "
            f"{quality_locality['City']}"
        ),
        "Score": round(
            float(
                quality_locality["Recommendation_Score"]
            ),
            3
        ),
        "Reason": (
            "Highest-ranked locality within the highly "
            "recommended tier."
        )
    })

insights_df = pd.DataFrame(insights)

print("\nLocation decision insights:")

print(
    insights_df.to_string(index=False)
)


# ============================================================
# 14. VISUALIZE TOP RECOMMENDED CITIES
# ============================================================

print("\n[14] GENERATING CITY RECOMMENDATION VISUALIZATION")

plot_city = (
    city_recommendations.head(15)
    .sort_values(
        "Recommendation_Score",
        ascending=True
    )
)

fig, ax = plt.subplots(figsize=(12, 7))

ax.barh(
    plot_city["City"],
    plot_city["Recommendation_Score"]
)

ax.set_title(
    "Top Recommended Cities",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel("Recommendation Score")
ax.set_ylabel("City")
ax.set_xlim(0, 100)

ax.grid(
    axis="x",
    alpha=0.25
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "11_city_recommendation_ranking.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("[SAVED] 11_city_recommendation_ranking.png")


# ============================================================
# 15. VISUALIZE TOP RECOMMENDED LOCALITIES
# ============================================================

print("\n[15] GENERATING LOCALITY RECOMMENDATION VISUALIZATION")

plot_locality = (
    locality_recommendations.head(15)
    .sort_values(
        "Recommendation_Score",
        ascending=True
    )
    .copy()
)

plot_locality["Location_Label"] = (
    plot_locality["Locality"].astype(str)
    + ", "
    + plot_locality["City"].astype(str)
)

fig, ax = plt.subplots(figsize=(13, 8))

ax.barh(
    plot_locality["Location_Label"],
    plot_locality["Recommendation_Score"]
)

ax.set_title(
    "Top Recommended Localities",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel("Recommendation Score")
ax.set_ylabel("Locality")
ax.set_xlim(0, 100)

ax.grid(
    axis="x",
    alpha=0.25
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "12_locality_recommendation_ranking.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("[SAVED] 12_locality_recommendation_ranking.png")


# ============================================================
# 16. VISUALIZE RECOMMENDATION COMPONENTS
# ============================================================

print("\n[16] GENERATING RECOMMENDATION COMPONENT VISUALIZATION")

component_columns = [
    "Rating_Score",
    "Cuisine_Diversity_Score",
    "Rating_Coverage_Score",
    "Restaurant_Availability_Score"
]

component_means = (
    city_recommendations[component_columns]
    .mean()
)

# Explicit NumPy conversion avoids Pylance type ambiguity.
component_mean_values = component_means.to_numpy(
    dtype=np.float64
)

component_labels = [
    "Rating",
    "Cuisine Diversity",
    "Rating Coverage",
    "Availability"
]

fig, ax = plt.subplots(figsize=(11, 6))

ax.bar(
    component_labels,
    component_mean_values
)

ax.set_title(
    "Average Recommendation Component Scores",
    fontsize=15,
    fontweight="bold"
)

ax.set_ylabel("Average Component Score")
ax.set_ylim(0, 100)

ax.grid(
    axis="y",
    alpha=0.25
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "13_recommendation_component_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("[SAVED] 13_recommendation_component_analysis.png")


# ============================================================
# 17. VISUALIZE ORIGINAL VS ADJUSTED CITY RATINGS
# ============================================================

print("\n[17] GENERATING RATING SHRINKAGE VISUALIZATION")

rating_comparison = (
    city_recommendations.head(15)
    .sort_values(
        "Adjusted_Average_Rating",
        ascending=True
    )
    .copy()
)

y_positions = np.arange(
    len(rating_comparison)
)

fig, ax = plt.subplots(figsize=(12, 8))

ax.scatter(
    rating_comparison["Average_Rating"],
    y_positions,
    label="Original Average Rating",
    marker="o"
)

ax.scatter(
    rating_comparison["Adjusted_Average_Rating"],
    y_positions,
    label="Adjusted Average Rating",
    marker="x"
)

ax.set_yticks(y_positions)

ax.set_yticklabels(
    rating_comparison["City"]
)

ax.set_title(
    "Original vs Sample-Size-Adjusted City Ratings",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel("Restaurant Rating")
ax.set_ylabel("City")

ax.set_xlim(0, 5)

ax.legend()

ax.grid(
    axis="x",
    alpha=0.25
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "14_city_rating_shrinkage_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("[SAVED] 14_city_rating_shrinkage_comparison.png")


# ============================================================
# 18. SAVE RECOMMENDATION DATASETS
# ============================================================

print("\n[18] SAVING RECOMMENDATION DATASETS")

analysis_outputs = {
    "location_city_recommendations.csv": city_recommendations,
    "location_locality_recommendations.csv": locality_recommendations,
    "location_city_recommendation_comparison.csv": city_comparison,
    "location_locality_recommendation_comparison.csv": locality_comparison,
    "location_city_recommendation_tiers.csv": city_tier_summary,
    "location_locality_recommendation_tiers.csv": locality_tier_summary,
    "location_high_quality_cities.csv": high_quality_cities,
    "location_high_quality_localities.csv": high_quality_localities,
    "location_high_coverage_cities.csv": high_coverage_cities,
    "location_high_coverage_localities.csv": high_coverage_localities,
    "location_top_city_restaurant_shortlist.csv": top_city_restaurants,
    "location_decision_insights.csv": insights_df
}

for filename, dataset in analysis_outputs.items():

    dataset.to_csv(
        REPORT_DIR / filename,
        index=False
    )

    print(
        f"[SAVED] {filename}: {dataset.shape}"
    )


# ============================================================
# 19. SAVE RECOMMENDATION CONFIGURATION
# ============================================================

print("\n[19] SAVING RECOMMENDATION CONFIGURATION")

configuration_rows = []

for component, weight in RECOMMENDATION_WEIGHTS.items():

    configuration_rows.append({
        "Component": component,
        "Configured_Weight": weight,
        "City_Active_Weight": city_weights.get(
            component,
            0
        ),
        "Locality_Active_Weight": locality_weights.get(
            component,
            0
        )
    })

configuration_df = pd.DataFrame(
    configuration_rows
)

configuration_df.to_csv(
    REPORT_DIR / "location_recommendation_configuration.csv",
    index=False
)

print(
    "[SAVED] location_recommendation_configuration.csv"
)


# ============================================================
# 20. SAVE RECOMMENDATION SUMMARY
# ============================================================

print("\n[20] SAVING RECOMMENDATION SUMMARY")

summary_data = [
    ("Total Cities", len(city_df)),
    ("Eligible Cities", len(city_recommendations)),
    ("Total Localities", len(locality_df)),
    ("Eligible Localities", len(locality_recommendations)),
    ("Highly Recommended Cities", len(high_quality_cities)),
    ("Highly Recommended Localities", len(high_quality_localities)),
    ("Cities with High Rating Coverage", len(high_coverage_cities)),
    ("Localities with High Rating Coverage", len(high_coverage_localities)),
    ("Global Average Rating", GLOBAL_AVERAGE_RATING),
    ("Rating Prior Strength", RATING_PRIOR_STRENGTH),
    ("Minimum Rated Restaurants", MIN_RATED_RESTAURANTS),
    ("Top City", best_city["City"]),
    ("Top City Original Rating", best_city["Average_Rating"]),
    ("Top City Adjusted Rating", best_city["Adjusted_Average_Rating"]),
    ("Top City Score", best_city["Recommendation_Score"]),
    ("Top Locality", best_locality["Locality"]),
    ("Top Locality City", best_locality["City"]),
    ("Top Locality Original Rating", best_locality["Average_Rating"]),
    ("Top Locality Adjusted Rating", best_locality["Adjusted_Average_Rating"]),
    ("Top Locality Score", best_locality["Recommendation_Score"]),
    ("Restaurant Shortlist Size", len(top_city_restaurants))
]

summary_df = pd.DataFrame(
    summary_data,
    columns=["Metric", "Value"]
)

summary_df.to_csv(
    REPORT_DIR / "location_recommendation_summary.csv",
    index=False
)

print(
    "[SAVED] location_recommendation_summary.csv"
)


# ============================================================
# 21. GENERATE TEXT REPORT
# ============================================================

print("\n[21] GENERATING LOCATION RECOMMENDATION REPORT")

report_path = (
    REPORT_DIR
    / "location_recommendation_report.txt"
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
        "PROGRAM 07: LOCATION RECOMMENDATION AND DECISION SUPPORT\n\n"
    )

    file.write("=" * 70 + "\n")
    file.write("RECOMMENDATION CONFIGURATION\n")
    file.write("=" * 70 + "\n\n")

    file.write(
        configuration_df.to_string(index=False)
    )

    file.write("\n\n")
    file.write("=" * 70 + "\n")
    file.write("TOP 10 CITY RECOMMENDATIONS\n")
    file.write("=" * 70 + "\n\n")

    file.write(
        city_recommendations[
            city_display_columns
        ].head(10).to_string(index=False)
    )

    file.write("\n\n")
    file.write("=" * 70 + "\n")
    file.write("TOP 10 LOCALITY RECOMMENDATIONS\n")
    file.write("=" * 70 + "\n\n")

    file.write(
        locality_recommendations[
            locality_display_columns
        ].head(10).to_string(index=False)
    )

    file.write("\n\n")
    file.write("=" * 70 + "\n")
    file.write("LOCATION DECISION INSIGHTS\n")
    file.write("=" * 70 + "\n\n")

    file.write(
        insights_df.to_string(index=False)
    )

    file.write("\n\n")
    file.write("=" * 70 + "\n")
    file.write("METHODOLOGY AND LIMITATIONS\n")
    file.write("=" * 70 + "\n\n")

    file.write(
        "1. Recommendation scores are composite ranking scores, "
        "not predicted customer satisfaction.\n"
    )

    file.write(
        "2. Component scores use percentile ranking among eligible "
        "locations.\n"
    )

    file.write(
        "3. Restaurant availability uses log-transformed restaurant "
        "counts.\n"
    )

    file.write(
        "4. Cuisine diversity uses the number of distinct cuisine "
        "labels where available.\n"
    )

    file.write(
        "5. Rating coverage measures the proportion of restaurants "
        "with available ratings.\n"
    )

    file.write(
        "6. Currency-specific costs are not compared directly because "
        "the dataset contains multiple currencies.\n"
    )

    file.write(
        "7. A higher recommendation score indicates a stronger match "
        "to the configured analytical criteria.\n"
    )

    file.write(
        "8. Restaurant shortlists are ranked by aggregate rating "
        "within the highest-ranked city.\n"
    )

    file.write(
        "9. Adjusted ratings use Bayesian-style shrinkage toward "
        "the global average rating to reduce small-sample bias.\n"
    )

    file.write(
        "10. Rating prior strength is configurable and currently "
        f"set to {RATING_PRIOR_STRENGTH}.\n"
    )

    file.write(
        "11. Original average ratings are preserved separately "
        "from adjusted average ratings.\n"
    )

    file.write(
        "12. Rating shrinkage is a ranking adjustment and is not "
        "a fully fitted Bayesian hierarchical model.\n"
    )


# ============================================================
# 22. FINAL VALIDATION
# ============================================================

print("\n[22] FINAL VALIDATION")

checks = {
    "City dataset available": (
        len(city_df) > 0
    ),

    "Locality dataset available": (
        len(locality_df) > 0
    ),

    "Restaurant dataset available": (
        len(restaurant_df) > 0
    ),

    "Valid global rating baseline": (
        0 < GLOBAL_AVERAGE_RATING <= 5
    ),

    "City recommendations generated": (
        not city_recommendations.empty
    ),

    "Locality recommendations generated": (
        not locality_recommendations.empty
    ),

    "City recommendation scores within range": (
        city_recommendations["Recommendation_Score"]
        .between(0, 100).all()
    ),

    "Locality recommendation scores within range": (
        locality_recommendations["Recommendation_Score"]
        .between(0, 100).all()
    ),

    "City adjusted ratings within valid range": (
        city_recommendations["Adjusted_Average_Rating"]
        .between(0, 5).all()
    ),

    "Locality adjusted ratings within valid range": (
        locality_recommendations["Adjusted_Average_Rating"]
        .between(0, 5).all()
    ),

    "City ranks are unique": (
        city_recommendations["Recommendation_Rank"].is_unique
    ),

    "Locality ranks are unique": (
        locality_recommendations["Recommendation_Rank"].is_unique
    ),

    "City ranks are sequential": (
        city_recommendations["Recommendation_Rank"].tolist()
        == list(range(1, len(city_recommendations) + 1))
    ),

    "Locality ranks are sequential": (
        locality_recommendations["Recommendation_Rank"].tolist()
        == list(range(1, len(locality_recommendations) + 1))
    ),

    "City recommendation tiers valid": (
        city_recommendations["Recommendation_Tier"].isin(
            [
                "Highly Recommended",
                "Recommended",
                "Moderately Recommended",
                "Lower Priority"
            ]
        ).all()
    ),

    "Locality recommendation tiers valid": (
        locality_recommendations["Recommendation_Tier"].isin(
            [
                "Highly Recommended",
                "Recommended",
                "Moderately Recommended",
                "Lower Priority"
            ]
        ).all()
    ),

    "City recommendation insights generated": (
        not insights_df.empty
    ),

    "Recommendation configuration generated": (
        not configuration_df.empty
    ),

    "Restaurant IDs unique in shortlist": (
        top_city_restaurants["Restaurant ID"].is_unique
    ),

    "City adjusted ratings match shrinkage formula": (
        np.allclose(
            city_recommendations["Adjusted_Average_Rating"],
            (
                city_recommendations["Rated_Restaurant_Count"]
                * city_recommendations["Average_Rating"]
                +
                RATING_PRIOR_STRENGTH * GLOBAL_AVERAGE_RATING
            )
            /
            (
                city_recommendations["Rated_Restaurant_Count"]
                + RATING_PRIOR_STRENGTH
            ),
            equal_nan=False
        )
    ),
    "Locality adjusted ratings match shrinkage formula": (
        np.allclose(
            locality_recommendations["Adjusted_Average_Rating"],
            (
                locality_recommendations["Rated_Restaurant_Count"]
                * locality_recommendations["Average_Rating"]
                + RATING_PRIOR_STRENGTH * GLOBAL_AVERAGE_RATING
            )
            /
            (
                locality_recommendations["Rated_Restaurant_Count"]
                + RATING_PRIOR_STRENGTH
            ),
            equal_nan=False
        )
    ),
}

for check, passed in checks.items():

    print(
        f"{'[PASS]' if passed else '[FAIL]'} {check}"
    )

if not all(checks.values()):

    raise RuntimeError(
        "One or more recommendation validations failed."
    )


# ============================================================
# 23. VALIDATE OUTPUT FILES
# ============================================================

print("\n[23] VALIDATING OUTPUT FILES")

expected_files = [
    REPORT_DIR / filename
    for filename in analysis_outputs
]

expected_files.extend([
    REPORT_DIR / "location_recommendation_configuration.csv",
    REPORT_DIR / "location_recommendation_summary.csv",
    REPORT_DIR / "location_recommendation_report.txt",
    FIGURE_DIR / "11_city_recommendation_ranking.png",
    FIGURE_DIR / "12_locality_recommendation_ranking.png",
    FIGURE_DIR / "13_recommendation_component_analysis.png",
    FIGURE_DIR / "14_city_rating_shrinkage_comparison.png"
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
        "One or more recommendation output files are missing."
    )


# ============================================================
# 24. EXECUTION SUMMARY
# ============================================================

print("\n[24] EXECUTION SUMMARY")

print("Total cities:", len(city_df))
print("Eligible cities:", len(city_recommendations))

print("Total localities:", len(locality_df))
print("Eligible localities:", len(locality_recommendations))

print(
    "Global average rating:",
    round(GLOBAL_AVERAGE_RATING, 4)
)

print(
    "Rating prior strength:",
    RATING_PRIOR_STRENGTH
)

print(
    "Top recommended city:",
    best_city["City"]
)

print(
    "Original city rating:",
    round(float(best_city["Average_Rating"]), 4)
)

print(
    "Adjusted city rating:",
    round(float(best_city["Adjusted_Average_Rating"]), 4)
)

print(
    "Top recommended locality:",
    best_locality["Locality"],
    "-",
    best_locality["City"]
)

print(
    "Original locality rating:",
    round(float(best_locality["Average_Rating"]), 4)
)

print(
    "Adjusted locality rating:",
    round(float(best_locality["Adjusted_Average_Rating"]), 4)
)

print(
    "Highly recommended cities:",
    len(high_quality_cities)
)

print(
    "Highly recommended localities:",
    len(high_quality_localities)
)

print(
    "Restaurant shortlist size:",
    len(top_city_restaurants)
)

print("Reports directory:", REPORT_DIR)
print("Figures directory:", FIGURE_DIR)

print("\n" + "=" * 75)
print("PROGRAM 07 EXECUTION COMPLETED SUCCESSFULLY")
print("=" * 75)
