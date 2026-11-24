from pathlib import Path
import time
import tracemalloc

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ============================================================
# DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS
# TASK 02: RESTAURANT RECOMMENDATION SYSTEM
# PROGRAM 07: RECOMMENDATION SYSTEM PERFORMANCE ANALYSIS
# ============================================================

print("=" * 75)
print("DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS")
print("TASK 02: RESTAURANT RECOMMENDATION SYSTEM")
print("PROGRAM 07 - RECOMMENDATION SYSTEM PERFORMANCE ANALYSIS")
print("=" * 75)


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "task_2_restaurant_recommendation"
REPORT_DIR = PROJECT_ROOT / "reports" / "task_2"

CUISINE_MATRIX_PATH = (
    PROCESSED_DIR / "recommendation_cuisine_features.csv"
)

METADATA_PATH = (
    PROCESSED_DIR / "recommendation_restaurant_metadata.csv"
)

PERFORMANCE_REPORT_PATH = (
    REPORT_DIR / "recommendation_performance_summary.csv"
)

SCALABILITY_REPORT_PATH = (
    REPORT_DIR / "recommendation_scalability_analysis.csv"
)

REQUEST_REPORT_PATH = (
    REPORT_DIR / "recommendation_request_size_analysis.csv"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPEAT_COUNT = 5
TOP_N_VALUES = [5, 10, 25, 50]

# Fixed benchmark preferences.
SELECTED_CUISINES = ["north indian", "chinese"]
SELECTED_CITY = "new delhi"
SELECTED_PRICE_RANGES = [1, 2]
MINIMUM_RATING = 3.5

RANDOM_STATE = 42


# ============================================================
# 2. LOAD AND VALIDATE ENGINEERED DATA
# ============================================================

print("\n" + "=" * 65)
print("2. LOAD AND VALIDATE ENGINEERED DATA")
print("=" * 65)

for file_path in [
    CUISINE_MATRIX_PATH,
    METADATA_PATH,
]:

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )

cuisine_features = pd.read_csv(CUISINE_MATRIX_PATH)
metadata = pd.read_csv(METADATA_PATH)

feature_columns = [
    column
    for column in cuisine_features.columns
    if column.startswith("cuisine_")
]

required_columns = [
    "Restaurant ID",
    "Restaurant Name",
    "City",
    "Locality",
    "Price range",
    "Aggregate rating",
    "Votes",
    "City Normalized",
    "Locality Normalized",
]

missing_columns = [
    column for column in required_columns
    if column not in metadata.columns
]

if missing_columns:
    raise ValueError(
        f"Required metadata columns missing: {missing_columns}"
    )

if not feature_columns:
    raise ValueError("No cuisine features found.")

if len(cuisine_features) != len(metadata):
    raise ValueError(
        "Cuisine feature and metadata row counts differ."
    )

if not cuisine_features["Restaurant ID"].equals(
    metadata["Restaurant ID"]
):
    raise ValueError(
        "Restaurant ID alignment validation failed."
    )

if metadata["Restaurant ID"].duplicated().any():
    raise ValueError("Duplicate restaurant IDs detected.")

cuisine_matrix = cuisine_features[
    feature_columns
].to_numpy(dtype=np.float32)

if not np.isfinite(cuisine_matrix).all():
    raise ValueError(
        "Cuisine matrix contains non-finite values."
    )

if not np.isin(cuisine_matrix, [0, 1]).all():
    raise ValueError(
        "Cuisine matrix must contain only 0 and 1."
    )

metadata["Price range"] = pd.to_numeric(
    metadata["Price range"],
    errors="coerce"
)

metadata["Aggregate rating"] = pd.to_numeric(
    metadata["Aggregate rating"],
    errors="coerce"
)

metadata["Votes"] = pd.to_numeric(
    metadata["Votes"],
    errors="coerce"
)

# A zero rating means Not Rated in this dataset.
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

print(f"Restaurants loaded: {len(metadata)}")
print(f"Cuisine features: {len(feature_columns)}")
print(f"Cuisine matrix shape: {cuisine_matrix.shape}")

print("Dataset validation passed.")


# ============================================================
# 3. PREPARE BENCHMARK PREFERENCE VECTOR
# ============================================================

print("\n" + "=" * 65)
print("3. PREPARE BENCHMARK PREFERENCE VECTOR")
print("=" * 65)

cuisine_lookup = {
    column.replace("cuisine_", "", 1).replace("_", " ").casefold(): column
    for column in feature_columns
}

user_vector = np.zeros(
    len(feature_columns),
    dtype=np.float32
)

for cuisine in SELECTED_CUISINES:

    if cuisine not in cuisine_lookup:
        raise ValueError(
            f"Benchmark cuisine not found: {cuisine}"
        )

    column = cuisine_lookup[cuisine]
    position = feature_columns.index(column)

    user_vector[position] = 1

print("Selected cuisines:", SELECTED_CUISINES)
print("Selected city:", SELECTED_CITY)
print("Selected price ranges:", SELECTED_PRICE_RANGES)
print("Minimum rating:", MINIMUM_RATING)

print("Benchmark preference vector prepared.")


# ============================================================
# 4. DEFINE PERFORMANCE MEASUREMENT FUNCTION
# ============================================================

print("\n" + "=" * 65)
print("4. DEFINE PERFORMANCE MEASUREMENT FUNCTION")
print("=" * 65)


def measure_performance(function, repeat_count=REPEAT_COUNT):

    execution_times = []
    peak_memory_values = []

    # Warm-up execution before collecting measurements.
    function()

    for _ in range(repeat_count):

        tracemalloc.start()

        start_time = time.perf_counter()

        function()

        end_time = time.perf_counter()

        current_memory, peak_memory = tracemalloc.get_traced_memory()

        tracemalloc.stop()

        execution_times.append(
            (end_time - start_time) * 1000
        )

        peak_memory_values.append(
            peak_memory / (1024 * 1024)
        )

    return {
        "Mean Time (ms)": float(np.mean(execution_times)),
        "Median Time (ms)": float(np.median(execution_times)),
        "Minimum Time (ms)": float(np.min(execution_times)),
        "Maximum Time (ms)": float(np.max(execution_times)),
        "Peak Python Memory (MB)": float(
            np.max(peak_memory_values)
        ),
    }


print(
    f"Performance measurement configured for "
    f"{REPEAT_COUNT} repetitions."
)

print("Warm-up execution enabled.")
print("Timer: time.perf_counter()")
print("Memory measurement: tracemalloc")


# ============================================================
# 5. DEFINE RECOMMENDATION COMPONENTS
# ============================================================

print("\n" + "=" * 65)
print("5. DEFINE RECOMMENDATION COMPONENTS")
print("=" * 65)


def calculate_similarity(matrix):

    return cosine_similarity(
        matrix,
        user_vector.reshape(1, -1)
    ).ravel()


def apply_preference_filters(
    source_metadata,
    similarity_scores
):

    candidates = source_metadata.copy()

    candidates["Cuisine Similarity"] = similarity_scores

    candidates = candidates[
        candidates["City Normalized"] == SELECTED_CITY
    ]

    candidates = candidates[
        candidates["Price range"].isin(
            SELECTED_PRICE_RANGES
        )
    ]

    candidates = candidates[
        candidates["Aggregate rating"] >= MINIMUM_RATING
    ]

    return candidates


def rank_recommendations(candidates, top_n):

    if candidates.empty:
        return candidates.copy()

    ranked = candidates.copy()

    # This is a ranking-workload proxy for benchmarking.
    # It does not replace Program 04's production scoring formula.
    ranked["Benchmark Ranking Score"] = (
        ranked["Cuisine Similarity"].clip(0, 1)
    )

    ranked = ranked.sort_values(
        by=[
            "Benchmark Ranking Score",
            "Aggregate rating",
            "Votes",
        ],
        ascending=[False, False, False],
        na_position="last"
    )

    return ranked.head(top_n).reset_index(drop=True)


print("Similarity function prepared.")
print("Preference filtering function prepared.")
print("Ranking function prepared.")


# ============================================================
# 6. MEASURE INDIVIDUAL COMPONENT PERFORMANCE
# ============================================================

print("\n" + "=" * 65)
print("6. MEASURE INDIVIDUAL COMPONENT PERFORMANCE")
print("=" * 65)

similarity_scores = calculate_similarity(cuisine_matrix)

filtered_candidates = apply_preference_filters(
    metadata,
    similarity_scores
)

if filtered_candidates.empty:
    raise ValueError(
        "Benchmark preferences produced no eligible restaurants."
    )

print(
    "Eligible restaurants for benchmark:",
    len(filtered_candidates)
)

component_results = []

# Similarity computation.
similarity_metrics = measure_performance(
    lambda: calculate_similarity(cuisine_matrix)
)

component_results.append({
    "Component": "Cuisine Similarity Computation",
    "Input Restaurants": len(metadata),
    "Output Restaurants": len(metadata),
    **similarity_metrics,
})

# Preference filtering.
filter_metrics = measure_performance(
    lambda: apply_preference_filters(
        metadata,
        similarity_scores
    )
)

component_results.append({
    "Component": "Preference Filtering",
    "Input Restaurants": len(metadata),
    "Output Restaurants": len(filtered_candidates),
    **filter_metrics,
})

# Ranking workload.
ranking_metrics = measure_performance(
    lambda: rank_recommendations(
        filtered_candidates,
        10
    )
)

component_results.append({
    "Component": "Recommendation Ranking",
    "Input Restaurants": len(filtered_candidates),
    "Output Restaurants": min(10, len(filtered_candidates)),
    **ranking_metrics,
})

print("\nComponent performance:")

for result in component_results:

    print(
        f"{result['Component']}: "
        f"{result['Median Time (ms)']:.4f} ms"
    )


# ============================================================
# 7. MEASURE COMPLETE RECOMMENDATION PIPELINE
# ============================================================

print("\n" + "=" * 65)
print("7. MEASURE COMPLETE RECOMMENDATION PIPELINE")
print("=" * 65)


def complete_recommendation_pipeline(top_n=10):

    similarities = calculate_similarity(cuisine_matrix)

    candidates = apply_preference_filters(
        metadata,
        similarities
    )

    recommendations = rank_recommendations(
        candidates,
        top_n
    )

    return recommendations


pipeline_metrics = measure_performance(
    lambda: complete_recommendation_pipeline(10)
)

final_recommendations = complete_recommendation_pipeline(10)

print("Complete pipeline benchmark completed.")
print("Recommendations generated:", len(final_recommendations))
print(
    "Median execution time:",
    f"{pipeline_metrics['Median Time (ms)']:.4f} ms"
)

component_results.append({
    "Component": "Complete Recommendation Pipeline",
    "Input Restaurants": len(metadata),
    "Output Restaurants": len(final_recommendations),
    **pipeline_metrics,
})


# ============================================================
# 8. ANALYZE RECOMMENDATION REQUEST SIZES
# ============================================================

print("\n" + "=" * 65)
print("8. ANALYZE RECOMMENDATION REQUEST SIZES")
print("=" * 65)

request_results = []

for top_n in TOP_N_VALUES:

    metrics = measure_performance(
        lambda n=top_n: complete_recommendation_pipeline(n)
    )

    recommendations = complete_recommendation_pipeline(top_n)

    request_results.append({
        "Requested Recommendations": top_n,
        "Eligible Restaurants": len(filtered_candidates),
        "Recommendations Generated": len(recommendations),
        **metrics,
    })

    print(
        f"Top {top_n}: "
        f"{len(recommendations)} recommendations, "
        f"{metrics['Median Time (ms)']:.4f} ms"
    )


# ============================================================
# 9. ANALYZE DATASET SCALABILITY
# ============================================================

print("\n" + "=" * 65)
print("9. ANALYZE DATASET SCALABILITY")
print("=" * 65)

scaling_percentages = [25, 50, 75, 100]

scaling_results = []

rng = np.random.default_rng(RANDOM_STATE)

for percentage in scaling_percentages:

    sample_size = max(
        1,
        int(len(metadata) * percentage / 100)
    )

    selected_indices = np.sort(
        rng.choice(
            len(metadata),
            size=sample_size,
            replace=False
        )
    )

    sample_metadata = (
        metadata.iloc[selected_indices]
        .reset_index(drop=True)
    )

    sample_matrix = cuisine_matrix[selected_indices]

    def run_sample_pipeline(
        current_metadata=sample_metadata,
        current_matrix=sample_matrix
    ):

        similarities = cosine_similarity(
            current_matrix,
            user_vector.reshape(1, -1)
        ).ravel()

        candidates = current_metadata.copy()

        candidates["Cuisine Similarity"] = similarities

        candidates = candidates[
            candidates["City Normalized"] == SELECTED_CITY
        ]

        candidates = candidates[
            candidates["Price range"].isin(
                SELECTED_PRICE_RANGES
            )
        ]

        candidates = candidates[
            candidates["Aggregate rating"] >= MINIMUM_RATING
        ]

        if candidates.empty:
            return candidates

        candidates["Benchmark Ranking Score"] = (
            candidates["Cuisine Similarity"].clip(0, 1)
        )

        return candidates.sort_values(
            by=[
                "Benchmark Ranking Score",
                "Aggregate rating",
                "Votes",
            ],
            ascending=[False, False, False],
            na_position="last"
        ).head(10)

    metrics = measure_performance(
        run_sample_pipeline
    )

    sample_recommendations = run_sample_pipeline()

    scaling_results.append({
        "Dataset Percentage": percentage,
        "Restaurants Processed": sample_size,
        "Eligible Restaurants": int(
            (
                (sample_metadata["City Normalized"] == SELECTED_CITY)
                & sample_metadata["Price range"].isin(
                    SELECTED_PRICE_RANGES
                )
                & (
                    sample_metadata["Aggregate rating"]
                    >= MINIMUM_RATING
                )
            ).sum()
        ),
        "Recommendations Generated": len(sample_recommendations),
        **metrics,
    })

    print(
        f"{percentage}% dataset: "
        f"{sample_size} restaurants, "
        f"{metrics['Median Time (ms)']:.4f} ms"
    )


# ============================================================
# 10. SAVE PERFORMANCE REPORTS
# ============================================================

print("\n" + "=" * 65)
print("10. SAVE PERFORMANCE REPORTS")
print("=" * 65)

component_summary = pd.DataFrame(component_results)

request_summary = pd.DataFrame(request_results)

scalability_summary = pd.DataFrame(scaling_results)

component_summary.to_csv(
    PERFORMANCE_REPORT_PATH,
    index=False
)

request_summary.to_csv(
    REQUEST_REPORT_PATH,
    index=False
)

scalability_summary.to_csv(
    SCALABILITY_REPORT_PATH,
    index=False
)

print("Performance reports saved successfully.")

print(f"\n1. {PERFORMANCE_REPORT_PATH}")
print(f"2. {REQUEST_REPORT_PATH}")
print(f"3. {SCALABILITY_REPORT_PATH}")


# ============================================================
# 11. DISPLAY PERFORMANCE RESULTS
# ============================================================

print("\n" + "=" * 65)
print("11. DISPLAY PERFORMANCE RESULTS")
print("=" * 65)

print("\n--- COMPONENT PERFORMANCE ---")

print(
    component_summary[
        [
            "Component",
            "Input Restaurants",
            "Output Restaurants",
            "Median Time (ms)",
            "Peak Python Memory (MB)",
        ]
    ].round(4).to_string(index=False)
)

print("\n--- REQUEST SIZE ANALYSIS ---")

print(
    request_summary[
        [
            "Requested Recommendations",
            "Recommendations Generated",
            "Median Time (ms)",
            "Peak Python Memory (MB)",
        ]
    ].round(4).to_string(index=False)
)

print("\n--- DATASET SCALABILITY ANALYSIS ---")

print(
    scalability_summary[
        [
            "Dataset Percentage",
            "Restaurants Processed",
            "Recommendations Generated",
            "Median Time (ms)",
            "Peak Python Memory (MB)",
        ]
    ].round(4).to_string(index=False)
)


# ============================================================
# 12. FINAL PERFORMANCE SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("TASK 02 - RESTAURANT RECOMMENDATION SYSTEM")
print("PROGRAM 07 - RECOMMENDATION PERFORMANCE ANALYSIS COMPLETED")
print("=" * 75)

print(f"\nTotal restaurants: {len(metadata)}")
print(f"Cuisine features: {len(feature_columns)}")
print(f"Benchmark eligible restaurants: {len(filtered_candidates)}")

print(
    "\nComplete pipeline median time:",
    f"{pipeline_metrics['Median Time (ms)']:.4f} ms"
)

print(
    "Complete pipeline peak Python memory:",
    f"{pipeline_metrics['Peak Python Memory (MB)']:.4f} MB"
)

print("\nGenerated reports:")
print("1. recommendation_performance_summary.csv")
print("2. recommendation_request_size_analysis.csv")
print("3. recommendation_scalability_analysis.csv")

print("\nPROGRAM 07 EXECUTION SUCCESSFUL.")

print("=" * 75)
