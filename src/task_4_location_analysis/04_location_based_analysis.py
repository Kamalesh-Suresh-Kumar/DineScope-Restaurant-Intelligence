from pathlib import Path

import numpy as np
import pandas as pd


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

FEATURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS
# TASK 04: LOCATION-BASED ANALYSIS
# PROGRAM 04: LOCATION-BASED ANALYSIS
# ============================================================

print("=" * 75)
print("DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS")
print("TASK 04: LOCATION-BASED ANALYSIS")
print("PROGRAM 04 - LOCATION-BASED ANALYSIS")
print("=" * 75)


# ============================================================
# 1. LOAD ENGINEERED DATASETS
# ============================================================

print("\n[1] LOADING ENGINEERED DATASETS")

required_files = {
    "restaurant": "restaurant_location_features.csv",
    "city": "city_location_features.csv",
    "locality": "locality_location_features.csv",
    "city_cuisine": "city_cuisine_distribution.csv",
    "locality_cuisine": "locality_cuisine_distribution.csv",
    "city_cost": "city_currency_cost_features.csv",
    "locality_cost": "locality_currency_cost_features.csv",
    "city_density": "city_restaurant_density.csv",
    "locality_density": "locality_restaurant_density.csv"
}

for name, filename in required_files.items():

    path = FEATURE_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Required input file not found: {path}"
        )

restaurant_df = pd.read_csv(
    FEATURE_DIR / required_files["restaurant"]
)

city_df = pd.read_csv(
    FEATURE_DIR / required_files["city"]
)

locality_df = pd.read_csv(
    FEATURE_DIR / required_files["locality"]
)

city_cuisine_df = pd.read_csv(
    FEATURE_DIR / required_files["city_cuisine"]
)

locality_cuisine_df = pd.read_csv(
    FEATURE_DIR / required_files["locality_cuisine"]
)

city_cost_df = pd.read_csv(
    FEATURE_DIR / required_files["city_cost"]
)

locality_cost_df = pd.read_csv(
    FEATURE_DIR / required_files["locality_cost"]
)

city_density_df = pd.read_csv(
    FEATURE_DIR / required_files["city_density"]
)

locality_density_df = pd.read_csv(
    FEATURE_DIR / required_files["locality_density"]
)

print(
    f"Restaurant records: {len(restaurant_df)}"
)

print(
    f"Cities: {len(city_df)}"
)

print(
    f"Locality groups: {len(locality_df)}"
)


# ============================================================
# 2. DEFINE ANALYSIS CONFIGURATION
# ============================================================

print("\n[2] DEFINING ANALYSIS CONFIGURATION")

MIN_RATED_RESTAURANTS = 10
MIN_LOCALITY_RESTAURANTS = 5

TOP_N = 20

print(
    "Minimum rated restaurants for city rating comparison:",
    MIN_RATED_RESTAURANTS
)

print(
    "Minimum restaurants for locality comparison:",
    MIN_LOCALITY_RESTAURANTS
)


# ============================================================
# 3. CITY-WISE RESTAURANT DISTRIBUTION
# ============================================================

print("\n[3] ANALYZING CITY-WISE RESTAURANT DISTRIBUTION")

city_distribution = city_df.sort_values(
    "Restaurant_Count",
    ascending=False
).copy()

total_restaurants = int(
    city_distribution["Restaurant_Count"].sum()
)

city_distribution["Restaurant_Share_Percentage"] = (
    city_distribution["Restaurant_Count"]
    / total_restaurants
    * 100
)

city_distribution["Cumulative_Restaurant_Share"] = (
    city_distribution["Restaurant_Share_Percentage"]
    .cumsum()
)

top_cities = city_distribution.head(TOP_N).copy()

print("\nTop 10 cities by restaurant count:")

print(
    top_cities[
        [
            "City",
            "Restaurant_Count",
            "Restaurant_Share_Percentage"
        ]
    ]
    .head(10)
    .round(2)
    .to_string(index=False)
)

top_3_city_share = float(
    city_distribution.head(3)[
        "Restaurant_Share_Percentage"
    ].sum()
)

top_10_city_share = float(
    city_distribution.head(10)[
        "Restaurant_Share_Percentage"
    ].sum()
)

print(
    f"\nTop 3 cities account for {top_3_city_share:.2f}% "
    "of all restaurants."
)

print(
    f"Top 10 cities account for {top_10_city_share:.2f}% "
    "of all restaurants."
)


# ============================================================
# 4. CITY-WISE RATING ANALYSIS
# ============================================================

print("\n[4] ANALYZING CITY-WISE RESTAURANT RATINGS")

city_rating_analysis = city_df.loc[
    city_df["Rated_Restaurant_Count"]
    >= MIN_RATED_RESTAURANTS
].copy()

city_rating_analysis = city_rating_analysis.sort_values(
    "Average_Rating",
    ascending=False
)

top_rated_cities = city_rating_analysis.head(
    TOP_N
).copy()

bottom_rated_cities = city_rating_analysis.tail(
    TOP_N
).sort_values(
    "Average_Rating",
    ascending=True
).copy()

print(
    "\nCities with highest average ratings "
    f"(minimum {MIN_RATED_RESTAURANTS} rated restaurants):"
)

print(
    top_rated_cities[
        [
            "City",
            "Rated_Restaurant_Count",
            "Average_Rating"
        ]
    ]
    .head(10)
    .round(2)
    .to_string(index=False)
)

print(
    "\nCities with lowest average ratings "
    f"(minimum {MIN_RATED_RESTAURANTS} rated restaurants):"
)

print(
    bottom_rated_cities[
        [
            "City",
            "Rated_Restaurant_Count",
            "Average_Rating"
        ]
    ]
    .head(10)
    .round(2)
    .to_string(index=False)
)


# ============================================================
# 5. LOCALITY-WISE RESTAURANT DISTRIBUTION
# ============================================================

print("\n[5] ANALYZING LOCALITY-WISE RESTAURANT DISTRIBUTION")

locality_distribution = locality_df.sort_values(
    "Restaurant_Count",
    ascending=False
).copy()

top_localities = locality_distribution.head(
    TOP_N
).copy()

print("\nTop 10 localities by restaurant count:")

print(
    top_localities[
        [
            "City",
            "Locality",
            "Restaurant_Count",
            "Unique_Cuisine_Count"
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# 6. LOCALITY-WISE RATING ANALYSIS
# ============================================================

print("\n[6] ANALYZING LOCALITY-WISE RATINGS")

locality_rating_analysis = locality_df.loc[
    (locality_df["Rated_Restaurant_Count"] >= MIN_LOCALITY_RESTAURANTS)
    & (locality_df["Average_Rating"].notna())
].copy()

locality_rating_analysis = (
    locality_rating_analysis.sort_values(
        "Average_Rating",
        ascending=False
    )
)

top_rated_localities = (
    locality_rating_analysis.head(TOP_N).copy()
)

bottom_rated_localities = (
    locality_rating_analysis.tail(TOP_N)
    .sort_values("Average_Rating")
    .copy()
)

print(
    "\nHighest-rated localities "
    f"(minimum {MIN_LOCALITY_RESTAURANTS} rated restaurants):"
)

print(
    top_rated_localities[
        [
            "City",
            "Locality",
            "Rated_Restaurant_Count",
            "Average_Rating"
        ]
    ]
    .head(10)
    .round(2)
    .to_string(index=False)
)

print(
    "\nLowest-rated localities "
    f"(minimum {MIN_LOCALITY_RESTAURANTS} rated restaurants):"
)

print(
    bottom_rated_localities[
        [
            "City",
            "Locality",
            "Rated_Restaurant_Count",
            "Average_Rating"
        ]
    ]
    .head(10)
    .round(2)
    .to_string(index=False)
)


# ============================================================
# 7. PRICE RANGE ANALYSIS BY CITY
# ============================================================

print("\n[7] ANALYZING PRICE RANGE DISTRIBUTION BY CITY")

price_city_df = restaurant_df[
    [
        "Country Code",
        "City",
        "Restaurant ID",
        "Price range"
    ]
].copy()

price_city_df = price_city_df.dropna(
    subset=["Price range"]
)

city_price_distribution = (
    price_city_df.groupby(
        [
            "Country Code",
            "City",
            "Price range"
        ]
    )
    .agg(
        Restaurant_Count=(
            "Restaurant ID",
            "nunique"
        )
    )
    .reset_index()
)

city_price_distribution["Price_Share_Percentage"] = (
    city_price_distribution["Restaurant_Count"]
    / city_price_distribution.groupby(
        ["Country Code", "City"]
    )["Restaurant_Count"].transform("sum")
    * 100
)

city_price_distribution = city_price_distribution.sort_values(
    [
        "City",
        "Price range"
    ]
)

print(
    f"City-price groups generated: {len(city_price_distribution)}"
)


# ============================================================
# 8. BUDGET AND PREMIUM LOCATION ANALYSIS
# ============================================================

print("\n[8] ANALYZING BUDGET AND PREMIUM RESTAURANT LOCATIONS")

budget_premium_analysis = (
    restaurant_df.groupby(
        [
            "Country Code",
            "City"
        ],
        dropna=False
    )
    .agg(
        Total_Restaurants=(
            "Restaurant ID",
            "count"
        ),
        Budget_Restaurants=(
            "Budget Price Restaurant",
            "sum"
        ),
        Premium_Restaurants=(
            "Premium Price Restaurant",
            "sum"
        )
    )
    .reset_index()
)

budget_premium_analysis["Budget_Share_Percentage"] = (
    budget_premium_analysis["Budget_Restaurants"]
    / budget_premium_analysis["Total_Restaurants"]
    * 100
)

budget_premium_analysis["Premium_Share_Percentage"] = (
    budget_premium_analysis["Premium_Restaurants"]
    / budget_premium_analysis["Total_Restaurants"]
    * 100
)

budget_premium_analysis = budget_premium_analysis.sort_values(
    "Total_Restaurants",
    ascending=False
)

print("\nTop 10 cities by budget restaurant share:")

print(
    budget_premium_analysis.sort_values(
        "Budget_Share_Percentage",
        ascending=False
    )[
        [
            "City",
            "Total_Restaurants",
            "Budget_Share_Percentage"
        ]
    ]
    .head(10)
    .round(2)
    .to_string(index=False)
)

print("\nTop 10 cities by premium restaurant share:")

print(
    budget_premium_analysis.sort_values(
        "Premium_Share_Percentage",
        ascending=False
    )[
        [
            "City",
            "Total_Restaurants",
            "Premium_Share_Percentage"
        ]
    ]
    .head(10)
    .round(2)
    .to_string(index=False)
)


# ============================================================
# 9. CUISINE POPULARITY BY CITY
# ============================================================

print("\n[9] ANALYZING CUISINE POPULARITY BY CITY")

city_cuisine_ranking = city_cuisine_df.copy()

city_cuisine_ranking = city_cuisine_ranking.sort_values(
    [
        "City",
        "Restaurant_Count"
    ],
    ascending=[True, False]
)

top_cuisines_by_city = (
    city_cuisine_ranking.groupby(
        ["Country Code", "City"],
        group_keys=False
    )
    .head(5)
    .copy()
)

print(
    "Top cuisine records generated:",
    len(top_cuisines_by_city)
)

print("\nTop cuisines in New Delhi:")

new_delhi_cuisines = city_cuisine_ranking.loc[
    city_cuisine_ranking["City"] == "New Delhi"
].head(10)

if new_delhi_cuisines.empty:
    print("New Delhi cuisine data not available.")
else:
    print(
        new_delhi_cuisines[
            [
                "Cuisine",
                "Restaurant_Count"
            ]
        ].to_string(index=False)
    )


# ============================================================
# 10. CUISINE DIVERSITY BY LOCATION
# ============================================================

print("\n[10] ANALYZING CUISINE DIVERSITY")

city_cuisine_diversity = city_df[
    [
        "Country Code",
        "City",
        "Restaurant_Count",
        "Unique_Cuisine_Count"
    ]
].copy()

city_cuisine_diversity["Cuisine_Diversity_Ratio"] = (
    city_cuisine_diversity["Unique_Cuisine_Count"]
    / city_cuisine_diversity["Restaurant_Count"]
)

city_cuisine_diversity = (
    city_cuisine_diversity.sort_values(
        "Unique_Cuisine_Count",
        ascending=False
    )
)

locality_cuisine_diversity = locality_df[
    [
        "Country Code",
        "City",
        "Locality",
        "Restaurant_Count",
        "Unique_Cuisine_Count"
    ]
].copy()

locality_cuisine_diversity["Cuisine_Diversity_Ratio"] = (
    locality_cuisine_diversity["Unique_Cuisine_Count"]
    / locality_cuisine_diversity["Restaurant_Count"]
)

print("\nTop 10 cities by cuisine diversity:")

print(
    city_cuisine_diversity[
        [
            "City",
            "Restaurant_Count",
            "Unique_Cuisine_Count"
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# 11. RESTAURANT RATING COVERAGE ANALYSIS
# ============================================================

print("\n[11] ANALYZING RATING COVERAGE")

rating_coverage_analysis = city_df[
    [
        "Country Code",
        "City",
        "Restaurant_Count",
        "Rated_Restaurant_Count",
        "Rating_Coverage_Percentage"
    ]
].copy()

rating_coverage_analysis = (
    rating_coverage_analysis.sort_values(
        "Rating_Coverage_Percentage",
        ascending=False
    )
)

print("\nHighest rating coverage cities:")

print(
    rating_coverage_analysis[
        [
            "City",
            "Restaurant_Count",
            "Rated_Restaurant_Count",
            "Rating_Coverage_Percentage"
        ]
    ]
    .head(10)
    .round(2)
    .to_string(index=False)
)


# ============================================================
# 12. LOCATION-LEVEL COST ANALYSIS
# ============================================================

print("\n[12] ANALYZING LOCATION-LEVEL COSTS")

city_cost_analysis = city_cost_df.sort_values(
    "Average_Cost",
    ascending=False
).copy()

locality_cost_analysis = locality_cost_df.sort_values(
    "Average_Cost",
    ascending=False
).copy()

print(
    "\nHighest average cost city-currency groups:"
)

print(
    city_cost_analysis[
        [
            "City",
            "Currency",
            "Restaurant_Count",
            "Average_Cost"
        ]
    ]
    .head(10)
    .round(2)
    .to_string(index=False)
)

print(
    "\nHighest average cost locality-currency groups:"
)

print(
    locality_cost_analysis[
        [
            "City",
            "Locality",
            "Currency",
            "Restaurant_Count",
            "Average_Cost"
        ]
    ]
    .head(10)
    .round(2)
    .to_string(index=False)
)


# ============================================================
# 13. GEOGRAPHICAL CONCENTRATION ANALYSIS
# ============================================================

print("\n[13] ANALYZING GEOGRAPHICAL CONCENTRATION")

coordinate_analysis = restaurant_df.loc[
    restaurant_df["Coordinate Valid"].astype(str).str.lower().eq("true")
].copy()

coordinate_analysis["Latitude"] = pd.to_numeric(
    coordinate_analysis["Latitude"],
    errors="coerce"
)

coordinate_analysis["Longitude"] = pd.to_numeric(
    coordinate_analysis["Longitude"],
    errors="coerce"
)

coordinate_analysis = coordinate_analysis.dropna(
    subset=["Latitude", "Longitude"]
)

coordinate_summary = pd.DataFrame([
    {
        "Metric": "Restaurants with valid coordinates",
        "Value": len(coordinate_analysis)
    },
    {
        "Metric": "Unique coordinate pairs",
        "Value": coordinate_analysis[
            ["Latitude", "Longitude"]
        ].drop_duplicates().shape[0]
    },
    {
        "Metric": "Latitude minimum",
        "Value": coordinate_analysis["Latitude"].min()
    },
    {
        "Metric": "Latitude maximum",
        "Value": coordinate_analysis["Latitude"].max()
    },
    {
        "Metric": "Longitude minimum",
        "Value": coordinate_analysis["Longitude"].min()
    },
    {
        "Metric": "Longitude maximum",
        "Value": coordinate_analysis["Longitude"].max()
    },
    {
        "Metric": "Mean latitude",
        "Value": coordinate_analysis["Latitude"].mean()
    },
    {
        "Metric": "Mean longitude",
        "Value": coordinate_analysis["Longitude"].mean()
    }
])

print(
    coordinate_summary.to_string(index=False)
)


# ============================================================
# 14. IDENTIFY LOCATION-LEVEL INSIGHTS
# ============================================================

print("\n[14] GENERATING LOCATION-LEVEL INSIGHTS")

insights = []

# Restaurant concentration
largest_city = city_distribution.iloc[0]

insights.append({
    "Insight_Category": "Restaurant Concentration",
    "Finding": (
        f"{largest_city['City']} has the highest restaurant count "
        f"with {int(largest_city['Restaurant_Count'])} restaurants."
    ),
    "Supporting_Value": round(
        float(largest_city["Restaurant_Share_Percentage"]),
        2
    ),
    "Unit": "Percentage of dataset"
})

# Highest-rated eligible city
if not top_rated_cities.empty:

    best_city = top_rated_cities.iloc[0]

    insights.append({
        "Insight_Category": "Highest Rated City",
        "Finding": (
            f"{best_city['City']} has the highest average rating "
            "among eligible cities."
        ),
        "Supporting_Value": round(
            float(best_city["Average_Rating"]),
            2
        ),
        "Unit": "Average rating"
    })

# Highest-rated eligible locality
if not top_rated_localities.empty:

    best_locality = top_rated_localities.iloc[0]

    insights.append({
        "Insight_Category": "Highest Rated Locality",
        "Finding": (
            f"{best_locality['Locality']}, {best_locality['City']} "
            "has the highest average rating among eligible localities."
        ),
        "Supporting_Value": round(
            float(best_locality["Average_Rating"]),
            2
        ),
        "Unit": "Average rating"
    })

# Cuisine diversity
most_diverse_city = city_cuisine_diversity.iloc[0]

insights.append({
    "Insight_Category": "Cuisine Diversity",
    "Finding": (
        f"{most_diverse_city['City']} has the highest number "
        "of distinct cuisine labels."
    ),
    "Supporting_Value": int(
        most_diverse_city["Unique_Cuisine_Count"]
    ),
    "Unit": "Distinct cuisine labels"
})

# Rating coverage
highest_coverage_city = (
    rating_coverage_analysis.iloc[0]
)

insights.append({
    "Insight_Category": "Rating Coverage",
    "Finding": (
        f"{highest_coverage_city['City']} has the highest "
        "rating coverage among cities."
    ),
    "Supporting_Value": round(
        float(
            highest_coverage_city[
                "Rating_Coverage_Percentage"
            ]
        ),
        2
    ),
    "Unit": "Percentage"
})

insights_df = pd.DataFrame(insights)

for _, row in insights_df.iterrows():

    print(
        f"\n{row['Insight_Category']}: "
        f"{row['Finding']}"
    )


# ============================================================
# 15. SAVE ANALYSIS OUTPUTS
# ============================================================

print("\n[15] SAVING ANALYSIS OUTPUTS")

analysis_outputs = {
    "location_city_distribution_analysis.csv": city_distribution,
    "location_city_rating_analysis.csv": city_rating_analysis,
    "location_top_rated_cities.csv": top_rated_cities,
    "location_bottom_rated_cities.csv": bottom_rated_cities,
    "location_locality_distribution_analysis.csv": locality_distribution,
    "location_locality_rating_analysis.csv": locality_rating_analysis,
    "location_top_rated_localities.csv": top_rated_localities,
    "location_bottom_rated_localities.csv": bottom_rated_localities,
    "location_city_price_distribution.csv": city_price_distribution,
    "location_budget_premium_analysis.csv": budget_premium_analysis,
    "location_top_cuisines_by_city.csv": top_cuisines_by_city,
    "location_city_cuisine_diversity.csv": city_cuisine_diversity,
    "location_locality_cuisine_diversity.csv": locality_cuisine_diversity,
    "location_rating_coverage_analysis.csv": rating_coverage_analysis,
    "location_city_cost_analysis.csv": city_cost_analysis,
    "location_locality_cost_analysis.csv": locality_cost_analysis,
    "location_coordinate_summary.csv": coordinate_summary,
    "location_key_insights.csv": insights_df
}

for filename, data in analysis_outputs.items():

    data.to_csv(
        REPORT_DIR / filename,
        index=False
    )

    print(
        f"[SAVED] {filename}: {data.shape}"
    )


# ============================================================
# 16. SAVE FINAL ANALYSIS SUMMARY
# ============================================================

print("\n[16] SAVING FINAL ANALYSIS SUMMARY")

analysis_summary = [
    ("Total Restaurants", len(restaurant_df)),
    ("Cities Analyzed", len(city_df)),
    ("Localities Analyzed", len(locality_df)),
    ("Top 3 City Restaurant Share (%)", round(top_3_city_share, 2)),
    ("Top 10 City Restaurant Share (%)", round(top_10_city_share, 2)),
    ("Eligible Cities for Rating Comparison", len(city_rating_analysis)),
    ("Eligible Localities for Rating Comparison", len(locality_rating_analysis)),
    ("City-Cuisine Records", len(city_cuisine_df)),
    ("Locality-Cuisine Records", len(locality_cuisine_df)),
    ("Cities with Cost Data", len(city_cost_df)),
    ("Localities with Cost Data", len(locality_cost_df)),
    ("Valid Coordinate Records", len(coordinate_analysis)),
    ("Key Insights Generated", len(insights_df))
]

analysis_summary_df = pd.DataFrame(
    analysis_summary,
    columns=["Metric", "Value"]
)

analysis_summary_df.to_csv(
    REPORT_DIR / "location_analysis_summary.csv",
    index=False
)


# ============================================================
# 17. SAVE TEXT REPORT
# ============================================================

print("\n[17] SAVING LOCATION ANALYSIS REPORT")

report_file = (
    REPORT_DIR
    / "location_based_analysis_report.txt"
)

with open(
    report_file,
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
        "PROGRAM 04: LOCATION-BASED ANALYSIS\n\n"
    )

    file.write("=" * 70 + "\n")
    file.write("ANALYSIS SUMMARY\n")
    file.write("=" * 70 + "\n\n")

    for metric, value in analysis_summary:
        file.write(f"{metric}: {value}\n")

    file.write("\n")
    file.write("=" * 70 + "\n")
    file.write("KEY INSIGHTS\n")
    file.write("=" * 70 + "\n\n")

    for _, row in insights_df.iterrows():

        file.write(
            f"{row['Insight_Category']}: "
            f"{row['Finding']}\n"
        )

    file.write("\n")
    file.write(
        "Rating comparisons use minimum sample-size filters. "
        "Average costs are grouped by currency. Restaurant "
        "density represents recorded restaurant counts per "
        "location, not restaurants per unit area.\n"
    )


# ============================================================
# 18. FINAL VALIDATION
# ============================================================

print("\n[18] FINAL VALIDATION")

checks = {
    "Restaurant dataset available": len(restaurant_df) > 0,
    "City analysis generated": not city_distribution.empty,
    "Locality analysis generated": not locality_distribution.empty,
    "City rating analysis generated": not city_rating_analysis.empty,
    "Locality rating analysis generated": not locality_rating_analysis.empty,
    "Price analysis generated": not city_price_distribution.empty,
    "Budget and premium analysis generated": not budget_premium_analysis.empty,
    "Cuisine analysis generated": not city_cuisine_ranking.empty,
    "Cost analysis generated": not city_cost_analysis.empty,
    "Coordinate analysis generated": not coordinate_summary.empty,
    "Key insights generated": not insights_df.empty,
    "City restaurant counts reconcile": (
        int(city_distribution["Restaurant_Count"].sum())
        == len(restaurant_df)
    ),
    "Locality restaurant counts reconcile": (
        int(locality_distribution["Restaurant_Count"].sum())
        == len(restaurant_df)
    ),
    "Restaurant IDs unique": restaurant_df["Restaurant ID"].is_unique,
    "Rating values valid": (
        restaurant_df["Aggregate rating"]
        .dropna()
        .between(0, 5)
        .all()
    )
}

for check, passed in checks.items():

    print(
        f"{'[PASS]' if passed else '[FAIL]'} {check}"
    )

if not all(checks.values()):
    raise RuntimeError(
        "One or more location analysis validations failed."
    )


# ============================================================
# 19. VALIDATE OUTPUT FILES
# ============================================================

print("\n[19] VALIDATING OUTPUT FILES")

expected_files = [
    REPORT_DIR / filename
    for filename in analysis_outputs
]

expected_files.extend([
    REPORT_DIR / "location_analysis_summary.csv",
    REPORT_DIR / "location_based_analysis_report.txt"
])

all_files_exist = True

for output_file in expected_files:

    exists = output_file.exists()

    print(
        f"{'[PASS]' if exists else '[FAIL]'} {output_file.name}"
    )

    if not exists:
        all_files_exist = False

if not all_files_exist:
    raise RuntimeError(
        "One or more analysis output files are missing."
    )

print("\nAnalysis reports saved to:")
print(REPORT_DIR)

print("\n" + "=" * 75)
print("PROGRAM 04 EXECUTION COMPLETED SUCCESSFULLY")
print("=" * 75)
