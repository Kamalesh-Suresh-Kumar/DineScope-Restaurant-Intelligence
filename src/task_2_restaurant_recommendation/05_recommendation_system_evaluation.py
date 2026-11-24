from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ============================================================
# DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS
# TASK 02: RESTAURANT RECOMMENDATION SYSTEM
# PROGRAM 05: RECOMMENDATION SYSTEM EVALUATION
# ============================================================

print("=" * 75)
print("DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS")
print("TASK 02: RESTAURANT RECOMMENDATION SYSTEM")
print("PROGRAM 05 - RECOMMENDATION SYSTEM EVALUATION")
print("=" * 75)


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "task_2_restaurant_recommendation"
REPORT_DIR = PROJECT_ROOT / "reports" / "task_2"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures" / "task_2"

CUISINE_MATRIX_PATH = (
    PROCESSED_DIR / "recommendation_cuisine_features.csv"
)

METADATA_PATH = (
    PROCESSED_DIR / "recommendation_restaurant_metadata.csv"
)

METRICS_PATH = (
    REPORT_DIR / "recommendation_evaluation_metrics.csv"
)

RESULTS_PATH = (
    REPORT_DIR / "recommendation_evaluation_results.csv"
)

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_FIGURE_PATH = (
    FIGURE_DIR / "01_recommendation_evaluation_metrics.png"
)

SIMILARITY_FIGURE_PATH = (
    FIGURE_DIR / "02_recommendation_similarity_analysis.png"
)

TOP_N = 10

CUISINE_WEIGHT = 0.85
RATING_WEIGHT = 0.15


# ============================================================
# 2. LOAD ENGINEERED FEATURES
# ============================================================

print("\n" + "=" * 65)
print("2. LOAD ENGINEERED FEATURES")
print("=" * 65)

if not CUISINE_MATRIX_PATH.exists():
    raise FileNotFoundError(
        f"Cuisine feature matrix not found: {CUISINE_MATRIX_PATH}"
    )

if not METADATA_PATH.exists():
    raise FileNotFoundError(
        f"Restaurant metadata not found: {METADATA_PATH}"
    )

cuisine_features = pd.read_csv(CUISINE_MATRIX_PATH)

metadata = pd.read_csv(METADATA_PATH)

feature_columns = [
    column
    for column in cuisine_features.columns
    if column.startswith("cuisine_")
]

if not feature_columns:
    raise ValueError("No cuisine feature columns found.")

if len(cuisine_features) != len(metadata):
    raise ValueError(
        "Cuisine features and metadata have different row counts."
    )

if not cuisine_features["Restaurant ID"].equals(
    metadata["Restaurant ID"]
):
    raise ValueError(
        "Restaurant IDs do not match between feature files."
    )

if metadata["Restaurant ID"].duplicated().any():
    raise ValueError("Duplicate restaurant IDs detected.")

print(f"Restaurants loaded: {len(metadata)}")
print(f"Cuisine features loaded: {len(feature_columns)}")

print("Engineered features loaded successfully.")


# ============================================================
# 3. PREPARE FEATURE MATRIX
# ============================================================

print("\n" + "=" * 65)
print("3. PREPARE FEATURE MATRIX")
print("=" * 65)

cuisine_matrix = cuisine_features[
    feature_columns
].to_numpy(dtype=np.float32)

if not np.isfinite(cuisine_matrix).all():
    raise ValueError(
        "Cuisine matrix contains invalid values."
    )

if not np.isin(cuisine_matrix, [0, 1]).all():
    raise ValueError(
        "Cuisine matrix must contain only binary values."
    )

metadata["Aggregate rating"] = pd.to_numeric(
    metadata["Aggregate rating"],
    errors="coerce"
)

metadata["Price range"] = pd.to_numeric(
    metadata["Price range"],
    errors="coerce"
)

metadata["Votes"] = pd.to_numeric(
    metadata["Votes"],
    errors="coerce"
)

# Aggregate rating = 0 means Not Rated in this dataset.
metadata.loc[
    metadata["Aggregate rating"] <= 0,
    "Aggregate rating"
] = np.nan

metadata["City Normalized"] = (
    metadata["City Normalized"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.casefold()
)

metadata["Locality Normalized"] = (
    metadata["Locality Normalized"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.casefold()
)

print(f"Cuisine matrix shape: {cuisine_matrix.shape}")
print("Feature matrix validation passed.")


# ============================================================
# 4. PREPARE CUISINE LOOKUP
# ============================================================

print("\n" + "=" * 65)
print("4. PREPARE CUISINE LOOKUP")
print("=" * 65)

cuisine_lookup = {}

for column in feature_columns:

    cuisine_name = (
        column.replace("cuisine_", "", 1)
        .replace("_", " ")
    )

    cuisine_lookup[cuisine_name.casefold()] = column

print(f"Available cuisines: {len(cuisine_lookup)}")


# ============================================================
# 5. DEFINE EVALUATION SCENARIOS
# ============================================================

print("\n" + "=" * 65)
print("5. DEFINE EVALUATION SCENARIOS")
print("=" * 65)

scenarios = [
    {
        "Scenario": "01_North_Indian",
        "Description": "North Indian cuisine preference",
        "Cuisines": ["north indian"],
        "City": "",
        "Locality": "",
        "Price Ranges": [],
        "Minimum Rating": None,
    },
    {
        "Scenario": "02_Multi_Cuisine_New_Delhi",
        "Description": "North Indian and Chinese in New Delhi",
        "Cuisines": ["north indian", "chinese"],
        "City": "new delhi",
        "Locality": "",
        "Price Ranges": [1, 2],
        "Minimum Rating": 3.5,
    },
    {
        "Scenario": "03_Italian",
        "Description": "Italian cuisine preference",
        "Cuisines": ["italian"],
        "City": "",
        "Locality": "",
        "Price Ranges": [],
        "Minimum Rating": None,
    },
    {
        "Scenario": "04_City_Price_Filter",
        "Description": "New Delhi with price ranges 1 and 2",
        "Cuisines": [],
        "City": "new delhi",
        "Locality": "",
        "Price Ranges": [1, 2],
        "Minimum Rating": None,
    },
    {
        "Scenario": "05_Empty_Result",
        "Description": "Nonexistent city to test empty results",
        "Cuisines": [],
        "City": "__no_matching_city__",
        "Locality": "",
        "Price Ranges": [],
        "Minimum Rating": None,
    },
]

print(f"Evaluation scenarios defined: {len(scenarios)}")

for scenario in scenarios:
    print(
        f"- {scenario['Scenario']}: "
        f"{scenario['Description']}"
    )


# ============================================================
# 6. DEFINE RECOMMENDATION EVALUATION FUNCTION
# ============================================================

print("\n" + "=" * 65)
print("6. DEFINE RECOMMENDATION EVALUATION FUNCTION")
print("=" * 65)


def evaluate_recommendations(scenario):

    selected_cuisines = scenario["Cuisines"]

    selected_city = scenario["City"]

    selected_locality = scenario["Locality"]

    selected_prices = scenario["Price Ranges"]

    minimum_rating = scenario["Minimum Rating"]

    # Generate the user's cuisine preference vector.
    user_vector = np.zeros(
        len(feature_columns),
        dtype=np.float32
    )

    for cuisine in selected_cuisines:

        if cuisine not in cuisine_lookup:
            raise ValueError(
                f"Unknown cuisine in evaluation scenario: {cuisine}"
            )

        column = cuisine_lookup[cuisine]

        position = feature_columns.index(column)

        user_vector[position] = 1

    # Calculate cuisine similarity.
    if selected_cuisines:

        similarity_scores = cosine_similarity(
            cuisine_matrix,
            user_vector.reshape(1, -1)
        ).ravel()

    else:

        # No cuisine preference means cuisine similarity
        # is not used for this scenario.
        similarity_scores = np.zeros(
            len(metadata),
            dtype=np.float32
        )

    recommendation_data = metadata.copy()

    recommendation_data["Cuisine Similarity"] = (
        similarity_scores
    )

    # Apply city filter.
    if selected_city:

        recommendation_data = recommendation_data[
            recommendation_data["City Normalized"]
            == selected_city
        ]

    # Apply locality filter.
    if selected_locality:

        recommendation_data = recommendation_data[
            recommendation_data["Locality Normalized"]
            == selected_locality
        ]

    # Apply price filter.
    if selected_prices:

        recommendation_data = recommendation_data[
            recommendation_data["Price range"].isin(
                selected_prices
            )
        ]

    # Apply minimum rating filter.
    if minimum_rating is not None:

        recommendation_data = recommendation_data[
            recommendation_data["Aggregate rating"]
            >= minimum_rating
        ]

    candidate_count = len(recommendation_data)

    # Handle empty results.
    if recommendation_data.empty:

        return {
            "recommendations": pd.DataFrame(),
            "candidate_count": candidate_count,
            "result_count": 0,
            "mean_similarity": np.nan,
            "exact_match_rate": np.nan,
            "zero_similarity_rate": np.nan,
            "mean_rating": np.nan,
            "rating_coverage": np.nan,
            "unique_cuisine_combinations": 0,
            "duplicate_ids": 0,
        }

    # Use the same ranking policy as Program 04.
    if selected_cuisines:

        # Cuisine similarity is primary.
        # Rating and votes are secondary tie-breakers.
        recommendation_data["Recommendation Score"] = (
            recommendation_data["Cuisine Similarity"]
            .clip(0.0, 1.0)
        )

    else:

        # Without cuisine preferences, use rating as the
        # primary score. Unrated restaurants remain last.
        recommendation_data["Recommendation Score"] = (
            recommendation_data["Aggregate rating"] / 5.0
        ).clip(0.0, 1.0)

    recommendation_data = recommendation_data.sort_values(
        by=[
            "Recommendation Score",
            "Cuisine Similarity",
            "Aggregate rating",
            "Votes",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
        na_position="last"
    ).reset_index(drop=True)

    recommendations = recommendation_data.head(TOP_N).copy()

    # Calculate descriptive evaluation metrics.
    if selected_cuisines:

        mean_similarity = float(
            recommendations["Cuisine Similarity"].mean()
        )

        exact_match_rate = float(
            (
                recommendations["Cuisine Similarity"]
                >= 0.999999
            ).mean() * 100
        )

        zero_similarity_rate = float(
            (
                recommendations["Cuisine Similarity"]
                == 0
            ).mean() * 100
        )

    else:

        mean_similarity = np.nan
        exact_match_rate = np.nan
        zero_similarity_rate = np.nan

    mean_rating = recommendations[
        "Aggregate rating"
    ].mean()

    rating_coverage = (
        recommendations["Aggregate rating"].notna().mean()
        * 100
    )

    unique_cuisine_combinations = (
        recommendations["Cuisines"].nunique()
    )

    duplicate_ids = int(
        recommendations["Restaurant ID"].duplicated().sum()
    )

    return {
        "recommendations": recommendations,
        "candidate_count": candidate_count,
        "result_count": len(recommendations),
        "mean_similarity": mean_similarity,
        "exact_match_rate": exact_match_rate,
        "zero_similarity_rate": zero_similarity_rate,
        "mean_rating": (
            float(mean_rating)
            if pd.notna(mean_rating)
            else np.nan
        ),
        "rating_coverage": float(rating_coverage),
        "unique_cuisine_combinations": int(
            unique_cuisine_combinations
        ),
        "duplicate_ids": duplicate_ids,
    }


print("Recommendation evaluation function prepared.")


# ============================================================
# 7. EXECUTE EVALUATION SCENARIOS
# ============================================================

print("\n" + "=" * 65)
print("7. EXECUTE EVALUATION SCENARIOS")
print("=" * 65)

evaluation_metrics = []
all_recommendations = []

for scenario in scenarios:

    scenario_name = scenario["Scenario"]

    print(f"\nEvaluating: {scenario_name}")

    result = evaluate_recommendations(scenario)

    recommendations = result["recommendations"]

    if result["candidate_count"] == 0:

        execution_status = "PASS - Empty result handled"

    elif result["duplicate_ids"] == 0:

        execution_status = "PASS"

    else:

        execution_status = "FAIL - Duplicate IDs detected"

    if result["candidate_count"] > 0:

        if len(recommendations) == 0:
            execution_status = "FAIL - Recommendations missing"

        if scenario["Minimum Rating"] is not None:

            if recommendations["Aggregate rating"].isna().any() or not (
                recommendations["Aggregate rating"]
                >= scenario["Minimum Rating"]
            ).all():

                execution_status = (
                    "FAIL - Minimum rating filter violated"
                )

        if scenario["City"]:
            if not (
                recommendations["City Normalized"]
                == scenario["City"]
            ).all():
                execution_status = "FAIL - City filter violated"

        if scenario["Locality"]:
            if not (
                recommendations["Locality Normalized"]
                == scenario["Locality"]
            ).all():
                execution_status = "FAIL - Locality filter violated"

        if scenario["Price Ranges"]:
            if not (
                recommendations["Price range"]
                .isin(scenario["Price Ranges"])
            ).all():
                execution_status = "FAIL - Price filter violated"

    if result["duplicate_ids"] > 0:
        execution_status = "FAIL - Duplicate recommendations"

    evaluation_metrics.append({
        "Scenario": scenario_name,
        "Description": scenario["Description"],
        "Cuisine Preferences": ", ".join(
            scenario["Cuisines"]
        ) or "Any",
        "City": scenario["City"] or "Any",
        "Price Ranges": ", ".join(
            map(str, scenario["Price Ranges"])
        ) or "Any",
        "Minimum Rating": (
            scenario["Minimum Rating"]
            if scenario["Minimum Rating"] is not None
            else "Any"
        ),
        "Eligible Restaurants": result["candidate_count"],
        "Recommendations Generated": result["result_count"],
        "Mean Cuisine Similarity": result["mean_similarity"],
        "Exact Cuisine Match Rate (%)": result["exact_match_rate"],
        "Zero Similarity Rate (%)": result["zero_similarity_rate"],
        "Mean Recommended Rating": result["mean_rating"],
        "Rating Coverage (%)": result["rating_coverage"],
        "Unique Cuisine Combinations": (
            result["unique_cuisine_combinations"]
        ),
        "Duplicate Recommendation IDs": result["duplicate_ids"],
        "Execution Status": execution_status,
    })

    if not recommendations.empty:

        recommendations = recommendations.copy()

        recommendations.insert(
            0,
            "Scenario",
            scenario_name
        )

        recommendations.insert(
            1,
            "Recommendation Rank",
            range(1, len(recommendations) + 1)
        )

        all_recommendations.append(recommendations)

    print(f"Eligible restaurants: {result['candidate_count']}")
    print(f"Recommendations generated: {result['result_count']}")
    print(f"Execution status: {execution_status}")


# ============================================================
# 8. GENERATE EVALUATION REPORTS
# ============================================================

print("\n" + "=" * 65)
print("8. GENERATE EVALUATION REPORTS")
print("=" * 65)

metrics_df = pd.DataFrame(evaluation_metrics)

if all_recommendations:

    recommendation_results = pd.concat(
        all_recommendations,
        ignore_index=True
    )

else:

    recommendation_results = pd.DataFrame()

metrics_df.to_csv(
    METRICS_PATH,
    index=False
)

recommendation_results.to_csv(
    RESULTS_PATH,
    index=False
)

print("Evaluation reports saved successfully.")

print(f"\n1. {METRICS_PATH}")
print(f"2. {RESULTS_PATH}")


# ============================================================
# 9. GENERATE EVALUATION VISUALIZATIONS
# ============================================================

print("\n" + "=" * 65)
print("9. GENERATE EVALUATION VISUALIZATIONS")
print("=" * 65)

# Figure 1: Eligible restaurants and recommendations.
plot_data = metrics_df[
    metrics_df["Eligible Restaurants"] > 0
].copy()

if not plot_data.empty:

    x = np.arange(len(plot_data))

    width = 0.35

    plt.figure(figsize=(12, 6))

    plt.bar(
        x - width / 2,
        plot_data["Eligible Restaurants"],
        width,
        label="Eligible Restaurants"
    )

    plt.bar(
        x + width / 2,
        plot_data["Recommendations Generated"],
        width,
        label="Recommendations Generated"
    )

    plt.xticks(
        x,
        plot_data["Scenario"].astype(str).tolist(),
        rotation=20,
        ha="right"
    )

    plt.ylabel("Number of Restaurants")
    plt.xlabel("Evaluation Scenario")

    plt.title(
        "DineScope - Recommendation Evaluation"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        METRICS_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {METRICS_FIGURE_PATH}")

# Figure 2: Mean cuisine similarity.
similarity_data = metrics_df.dropna(
    subset=["Mean Cuisine Similarity"]
)

if not similarity_data.empty:

    plt.figure(figsize=(10, 6))

    plt.bar(
        similarity_data["Scenario"],
        similarity_data["Mean Cuisine Similarity"]
    )

    plt.axhline(
        y=1.0,
        linestyle="--",
        label="Exact cuisine-vector similarity"
    )

    plt.ylim(0, 1.05)

    plt.ylabel("Mean Cuisine Similarity")
    plt.xlabel("Evaluation Scenario")

    plt.title(
        "DineScope - Cuisine Similarity Analysis"
    )

    plt.xticks(rotation=20, ha="right")

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        SIMILARITY_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {SIMILARITY_FIGURE_PATH}")


# ============================================================
# 10. FINAL EVALUATION SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("TASK 02 - RESTAURANT RECOMMENDATION SYSTEM")
print("PROGRAM 05 - RECOMMENDATION SYSTEM EVALUATION COMPLETED")
print("=" * 75)

print("\nEvaluation metrics:")

display_columns = [
    "Scenario",
    "Eligible Restaurants",
    "Recommendations Generated",
    "Mean Cuisine Similarity",
    "Exact Cuisine Match Rate (%)",
    "Mean Recommended Rating",
    "Execution Status",
]

print(
    metrics_df[display_columns].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)

total_scenarios = len(metrics_df)

passed_scenarios = int(
    metrics_df["Execution Status"]
    .str.startswith("PASS")
    .sum()
)

failed_scenarios = total_scenarios - passed_scenarios

print("\nOverall evaluation:")
print(f"Total scenarios: {total_scenarios}")
print(f"Passed scenarios: {passed_scenarios}")
print(f"Failed scenarios: {failed_scenarios}")

print("\nGenerated reports:")
print("1. recommendation_evaluation_metrics.csv")
print("2. recommendation_evaluation_results.csv")

print("\nGenerated figures:")
print("1. 01_recommendation_evaluation_metrics.png")
print("2. 02_recommendation_similarity_analysis.png")

if failed_scenarios == 0:

    print("\nPROGRAM 05 EXECUTION SUCCESSFUL.")

else:

    print("\nPROGRAM 05 COMPLETED WITH VALIDATION FAILURES.")

print("=" * 75)
