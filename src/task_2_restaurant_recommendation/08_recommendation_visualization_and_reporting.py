from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ============================================================
# DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS
# TASK 02: RESTAURANT RECOMMENDATION SYSTEM
# PROGRAM 08: RECOMMENDATION VISUALIZATION AND REPORTING
# ============================================================

print("=" * 75)
print("DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS")
print("TASK 02: RESTAURANT RECOMMENDATION SYSTEM")
print("PROGRAM 08 - RECOMMENDATION VISUALIZATION AND REPORTING")
print("=" * 75)


# ============================================================
# 1. CONFIGURATION
# ============================================================

REPORT_DIR = PROJECT_ROOT / "reports" / "task_2"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures" / "task_2"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RECOMMENDATION_PATH = (
    REPORT_DIR / "recommendation_results.csv"
)

EVALUATION_PATH = (
    REPORT_DIR / "recommendation_evaluation_metrics.csv"
)

TEST_SUMMARY_PATH = (
    REPORT_DIR / "recommendation_system_test_summary.csv"
)

PERFORMANCE_PATH = (
    REPORT_DIR / "recommendation_performance_summary.csv"
)

REQUEST_SIZE_PATH = (
    REPORT_DIR / "recommendation_request_size_analysis.csv"
)

SCALABILITY_PATH = (
    REPORT_DIR / "recommendation_scalability_analysis.csv"
)

SUMMARY_PATH = (
    REPORT_DIR / "recommendation_final_summary.csv"
)

TEXT_REPORT_PATH = (
    REPORT_DIR / "recommendation_final_report.txt"
)


# ============================================================
# 2. LOAD GENERATED REPORTS
# ============================================================

print("\n" + "=" * 65)
print("2. LOAD GENERATED REPORTS")
print("=" * 65)


def load_report(path, report_name):
    if not path.exists():
        raise FileNotFoundError(
            f"{report_name} not found: {path}"
        )

    data = pd.read_csv(path)

    if data.empty:
        raise ValueError(
            f"{report_name} is empty: {path}"
        )

    print(
        f"{report_name} loaded: "
        f"{len(data)} records"
    )

    return data


recommendations = load_report(
    RECOMMENDATION_PATH,
    "Recommendation results"
)

evaluation = load_report(
    EVALUATION_PATH,
    "Evaluation metrics"
)

test_summary = load_report(
    TEST_SUMMARY_PATH,
    "System test summary"
)

performance = load_report(
    PERFORMANCE_PATH,
    "Performance summary"
)

request_size = load_report(
    REQUEST_SIZE_PATH,
    "Request size analysis"
)

scalability = load_report(
    SCALABILITY_PATH,
    "Scalability analysis"
)


# ============================================================
# 3. VALIDATE REPORT COLUMNS
# ============================================================

print("\n" + "=" * 65)
print("3. VALIDATE REPORT STRUCTURES")
print("=" * 65)

required_recommendation_columns = [
    "Recommendation Rank",
    "Restaurant ID",
    "Restaurant Name",
    "Aggregate rating",
    "Cuisine Similarity",
    "Recommendation Score",
]

required_evaluation_columns = [
    "Scenario",
    "Eligible Restaurants",
    "Recommendations Generated",
    "Execution Status",
]

required_test_columns = [
    "Metric",
    "Value",
]

required_performance_columns = [
    "Component",
    "Median Time (ms)",
]

required_request_columns = [
    "Requested Recommendations",
    "Recommendations Generated",
    "Median Time (ms)",
]

required_scalability_columns = [
    "Dataset Percentage",
    "Restaurants Processed",
    "Median Time (ms)",
]

def validate_columns(data, required, report_name):

    missing = [
        column for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"{report_name} is missing columns: {missing}"
        )


validate_columns(
    recommendations,
    required_recommendation_columns,
    "Recommendation results"
)

validate_columns(
    evaluation,
    required_evaluation_columns,
    "Evaluation metrics"
)

validate_columns(
    test_summary,
    required_test_columns,
    "Test summary"
)

validate_columns(
    performance,
    required_performance_columns,
    "Performance summary"
)

validate_columns(
    request_size,
    required_request_columns,
    "Request size analysis"
)

validate_columns(
    scalability,
    required_scalability_columns,
    "Scalability analysis"
)

print("All report structures validated successfully.")


# ============================================================
# 4. PREPARE NUMERIC DATA
# ============================================================

print("\n" + "=" * 65)
print("4. PREPARE VISUALIZATION DATA")
print("=" * 65)

numeric_columns = [
    "Recommendation Rank",
    "Aggregate rating",
    "Cuisine Similarity",
    "Recommendation Score",
]

for column in numeric_columns:
    recommendations[column] = pd.to_numeric(
        recommendations[column],
        errors="coerce"
    )

for column in [
    "Eligible Restaurants",
    "Recommendations Generated",
]:
    evaluation[column] = pd.to_numeric(
        evaluation[column],
        errors="coerce"
    )

for column in [
    "Median Time (ms)",
]:
    performance[column] = pd.to_numeric(
        performance[column],
        errors="coerce"
    )

for column in [
    "Requested Recommendations",
    "Recommendations Generated",
    "Median Time (ms)",
]:
    request_size[column] = pd.to_numeric(
        request_size[column],
        errors="coerce"
    )

for column in [
    "Dataset Percentage",
    "Restaurants Processed",
    "Median Time (ms)",
]:
    scalability[column] = pd.to_numeric(
        scalability[column],
        errors="coerce"
    )

print("Numeric report fields prepared.")


# ============================================================
# 5. VISUALIZE TOP RECOMMENDATIONS
# ============================================================

print("\n" + "=" * 65)
print("5. VISUALIZE TOP RECOMMENDATIONS")
print("=" * 65)

top_recommendations = (
    recommendations
    .sort_values("Recommendation Rank")
    .head(10)
    .sort_values("Recommendation Score")
)

if not top_recommendations.empty:

    plt.figure(figsize=(11, 7))

    plt.barh(
        top_recommendations["Restaurant Name"].astype(str),
        top_recommendations["Recommendation Score"]
    )

    plt.xlabel("Recommendation Score")
    plt.ylabel("Restaurant")
    plt.title("DineScope - Top 10 Restaurant Recommendations")
    plt.tight_layout()

    path = FIGURE_DIR / "03_top_recommendations.png"

    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {path}")


# ============================================================
# 6. ANALYZE CUISINE SIMILARITY AND RATINGS
# ============================================================

print("\n" + "=" * 65)
print("6. ANALYZE CUISINE SIMILARITY AND RATINGS")
print("=" * 65)

similarity_data = recommendations.dropna(
    subset=[
        "Cuisine Similarity",
        "Aggregate rating",
    ]
)

if not similarity_data.empty:

    plt.figure(figsize=(9, 6))

    plt.scatter(
        similarity_data["Cuisine Similarity"],
        similarity_data["Aggregate rating"],
        alpha=0.75
    )

    plt.xlabel("Cuisine Similarity")
    plt.ylabel("Aggregate Rating")
    plt.title("Cuisine Similarity vs Restaurant Rating")
    plt.tight_layout()

    path = (
        FIGURE_DIR
        / "04_cuisine_similarity_vs_rating.png"
    )

    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {path}")

else:
    print(
        "Skipped similarity analysis: "
        "no valid rating and similarity pairs."
    )


# ============================================================
# 7. VISUALIZE EVALUATION SCENARIOS
# ============================================================

print("\n" + "=" * 65)
print("7. VISUALIZE EVALUATION SCENARIOS")
print("=" * 65)

plot_evaluation = evaluation.copy()

plot_evaluation["Scenario"] = (
    plot_evaluation["Scenario"].astype(str)
)

x = np.arange(len(plot_evaluation))
width = 0.38

plt.figure(figsize=(12, 6))

plt.bar(
    x - width / 2,
    plot_evaluation["Eligible Restaurants"],
    width,
    label="Eligible Restaurants"
)

plt.bar(
    x + width / 2,
    plot_evaluation["Recommendations Generated"],
    width,
    label="Recommendations Generated"
)

plt.xticks(
    x,
    plot_evaluation["Scenario"].astype(str).tolist(),
    rotation=25,
    ha="right"
)

plt.ylabel("Number of Restaurants")
plt.xlabel("Evaluation Scenario")
plt.title("DineScope - Recommendation Evaluation")
plt.legend()
plt.tight_layout()

path = FIGURE_DIR / "05_recommendation_evaluation.png"

plt.savefig(path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {path}")


# ============================================================
# 8. VISUALIZE COMPONENT PERFORMANCE
# ============================================================

print("\n" + "=" * 65)
print("8. VISUALIZE COMPONENT PERFORMANCE")
print("=" * 65)

component_data = performance[
    performance["Component"]
    != "Complete Recommendation Pipeline"
].copy()

component_data = component_data.dropna(
    subset=["Median Time (ms)"]
)

if not component_data.empty:

    plt.figure(figsize=(10, 6))

    plt.bar(
        component_data["Component"],
        component_data["Median Time (ms)"]
    )

    plt.ylabel("Median Execution Time (ms)")
    plt.xlabel("Recommendation Component")
    plt.title("Recommendation Component Performance")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    path = FIGURE_DIR / "06_component_performance.png"

    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {path}")


# ============================================================
# 9. VISUALIZE REQUEST SIZE PERFORMANCE
# ============================================================

print("\n" + "=" * 65)
print("9. VISUALIZE REQUEST SIZE PERFORMANCE")
print("=" * 65)

request_size = request_size.dropna(
    subset=[
        "Requested Recommendations",
        "Median Time (ms)",
    ]
)

plt.figure(figsize=(9, 6))

plt.plot(
    request_size["Requested Recommendations"],
    request_size["Median Time (ms)"],
    marker="o"
)

plt.xlabel("Requested Recommendations")
plt.ylabel("Median Execution Time (ms)")
plt.title("Recommendation Request Size vs Execution Time")
plt.grid(True, alpha=0.3)
plt.tight_layout()

path = FIGURE_DIR / "07_request_size_performance.png"

plt.savefig(path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {path}")


# ============================================================
# 10. VISUALIZE DATASET SCALABILITY
# ============================================================

print("\n" + "=" * 65)
print("10. VISUALIZE DATASET SCALABILITY")
print("=" * 65)

scalability = scalability.dropna(
    subset=[
        "Restaurants Processed",
        "Median Time (ms)",
    ]
)

plt.figure(figsize=(9, 6))

plt.plot(
    scalability["Restaurants Processed"],
    scalability["Median Time (ms)"],
    marker="o"
)

plt.xlabel("Restaurants Processed")
plt.ylabel("Median Execution Time (ms)")
plt.title("Recommendation System Scalability")
plt.grid(True, alpha=0.3)
plt.tight_layout()

path = FIGURE_DIR / "08_recommendation_scalability.png"

plt.savefig(path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {path}")


# ============================================================
# 11. GENERATE CONSOLIDATED SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("11. GENERATE CONSOLIDATED SUMMARY")
print("=" * 65)

def get_metric(metric_name):

    matched = test_summary[
        test_summary["Metric"].astype(str).str.casefold()
        == metric_name.casefold()
    ]

    if matched.empty:
        return np.nan

    return pd.to_numeric(
        matched.iloc[0]["Value"],
        errors="coerce"
    )


pipeline_row = performance[
    performance["Component"]
    == "Complete Recommendation Pipeline"
]

if pipeline_row.empty:
    pipeline_time = np.nan
else:
    pipeline_time = pipeline_row.iloc[0]["Median Time (ms)"]


summary = pd.DataFrame([
    {
        "Metric": "Total Restaurants in Dataset",
        "Value": 9551,
    },
    {
        "Metric": "Recommendations Generated",
        "Value": len(recommendations),
    },
    {
        "Metric": "Evaluation Scenarios",
        "Value": len(evaluation),
    },
    {
        "Metric": "Evaluation Scenarios Passed",
        "Value": (
            evaluation["Execution Status"]
            .astype(str)
            .str.startswith("PASS")
            .sum()
        ),
    },
    {
        "Metric": "Total Test Cases",
        "Value": get_metric("Total Test Cases"),
    },
    {
        "Metric": "Passed Tests",
        "Value": get_metric("Passed Tests"),
    },
    {
        "Metric": "Failed Tests",
        "Value": get_metric("Failed Tests"),
    },
    {
        "Metric": "Test Pass Rate (%)",
        "Value": get_metric("Test Pass Rate (%)"),
    },
    {
        "Metric": "Complete Pipeline Median Time (ms)",
        "Value": pipeline_time,
    },
])

summary.to_csv(
    SUMMARY_PATH,
    index=False
)

print(summary.to_string(index=False))

print(f"\nSummary saved: {SUMMARY_PATH}")


# ============================================================
# 12. GENERATE FINAL TEXT REPORT
# ============================================================

print("\n" + "=" * 65)
print("12. GENERATE FINAL TEXT REPORT")
print("=" * 65)

passed_evaluations = (
    evaluation["Execution Status"]
    .astype(str)
    .str.startswith("PASS")
    .sum()
)

total_evaluations = len(evaluation)

total_tests = get_metric("Total Test Cases")
passed_tests = get_metric("Passed Tests")
failed_tests = get_metric("Failed Tests")
pass_rate = get_metric("Test Pass Rate (%)")

report_lines = [
    "DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS",
    "TASK 02: RESTAURANT RECOMMENDATION SYSTEM",
    "PROGRAM 08: FINAL VISUALIZATION AND REPORTING",
    "",
    "=" * 65,
    "1. RECOMMENDATION OVERVIEW",
    "=" * 65,
    f"Recommendations in latest result file: {len(recommendations)}",
    "",
    "=" * 65,
    "2. EVALUATION OVERVIEW",
    "=" * 65,
    f"Total evaluation scenarios: {total_evaluations}",
    f"Passed evaluation scenarios: {passed_evaluations}",
    f"Failed evaluation scenarios: {total_evaluations - passed_evaluations}",
    "",
    "=" * 65,
    "3. AUTOMATED TESTING",
    "=" * 65,
    f"Total test cases: {total_tests}",
    f"Passed tests: {passed_tests}",
    f"Failed tests: {failed_tests}",
    f"Test pass rate (%): {pass_rate}",
    "",
    "=" * 65,
    "4. PERFORMANCE OVERVIEW",
    "=" * 65,
    f"Complete pipeline median time (ms): {pipeline_time}",
    "",
    "=" * 65,
    "5. GENERATED VISUALIZATIONS",
    "=" * 65,
    "03_top_recommendations.png",
    "04_cuisine_similarity_vs_rating.png",
    "05_recommendation_evaluation.png",
    "06_component_performance.png",
    "07_request_size_performance.png",
    "08_recommendation_scalability.png",
    "",
    "=" * 65,
    "6. INTERPRETATION AND LIMITATIONS",
    "=" * 65,
    "This report consolidates outputs from the recommendation",
    "engine, evaluation, automated testing and performance analysis.",
    "",
    "The evaluation scenarios and automated tests validate",
    "implemented behavior against predefined test conditions.",
    "They do not establish real-world user satisfaction.",
    "",
    "Execution timings are local benchmark measurements and",
    "may vary with hardware, system load and runtime conditions.",
    "",
    "Recommendation scores are implementation-specific ranking",
    "scores and should not be interpreted as probabilities.",
    "",
    "=" * 65,
    "END OF REPORT",
    "=" * 65,
]

with open(
    TEXT_REPORT_PATH,
    "w",
    encoding="utf-8"
) as file:
    file.write("\n".join(report_lines))

print(f"Final report saved: {TEXT_REPORT_PATH}")


# ============================================================
# 13. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 75)
print("TASK 02 - RESTAURANT RECOMMENDATION SYSTEM")
print("PROGRAM 08 - VISUALIZATION AND REPORTING COMPLETED")
print("=" * 75)

print("\nGenerated visualizations:")
print("1. 03_top_recommendations.png")
print("2. 04_cuisine_similarity_vs_rating.png")
print("3. 05_recommendation_evaluation.png")
print("4. 06_component_performance.png")
print("5. 07_request_size_performance.png")
print("6. 08_recommendation_scalability.png")

print("\nGenerated reports:")
print("1. recommendation_final_summary.csv")
print("2. recommendation_final_report.txt")

print("\nPROGRAM 08 EXECUTION SUCCESSFUL.")

print("=" * 75)
