from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ============================================================
# DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS
# TASK 02: RESTAURANT RECOMMENDATION SYSTEM
# PROGRAM 06: RECOMMENDATION SYSTEM TESTING
# ============================================================

print("=" * 75)
print("DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS")
print("TASK 02: RESTAURANT RECOMMENDATION SYSTEM")
print("PROGRAM 06 - RECOMMENDATION SYSTEM TESTING")
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

TEST_RESULTS_PATH = (
    REPORT_DIR / "recommendation_system_test_results.csv"
)

TEST_SUMMARY_PATH = (
    REPORT_DIR / "recommendation_system_test_summary.csv"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 10


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

required_metadata_columns = [
    "Restaurant ID",
    "Restaurant Name",
    "City",
    "Locality",
    "Cuisines",
    "Price range",
    "Aggregate rating",
    "Votes",
    "City Normalized",
    "Locality Normalized",
]

missing_columns = [
    column for column in required_metadata_columns
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
        "Cuisine features and metadata row counts differ."
    )

if not cuisine_features["Restaurant ID"].equals(
    metadata["Restaurant ID"]
):
    raise ValueError(
        "Restaurant ID alignment validation failed."
    )

if metadata["Restaurant ID"].duplicated().any():
    raise ValueError(
        "Duplicate restaurant IDs detected."
    )

cuisine_matrix = cuisine_features[
    feature_columns
].to_numpy(dtype=np.float32)

if not np.isfinite(cuisine_matrix).all():
    raise ValueError(
        "Cuisine matrix contains non-finite values."
    )

if not np.isin(cuisine_matrix, [0, 1]).all():
    raise ValueError(
        "Cuisine matrix contains values other than 0 and 1."
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

# In this dataset, rating 0 means Not Rated.
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

print("Engineered dataset validation passed.")


# ============================================================
# 3. PREPARE CUISINE LOOKUP
# ============================================================

print("\n" + "=" * 65)
print("3. PREPARE CUISINE LOOKUP")
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
# 4. DEFINE RECOMMENDATION FUNCTION
# ============================================================

print("\n" + "=" * 65)
print("4. DEFINE RECOMMENDATION FUNCTION")
print("=" * 65)


def generate_recommendations(
    cuisines=None,
    city="",
    locality="",
    price_ranges=None,
    minimum_rating=None,
    top_n=TOP_N,
):

    cuisines = cuisines or []
    price_ranges = price_ranges or []

    cuisines = [
        str(cuisine).strip().casefold()
        for cuisine in cuisines
    ]

    cuisines = list(dict.fromkeys(cuisines))

    if top_n <= 0:
        raise ValueError("top_n must be positive.")

    if minimum_rating is not None:
        if not 1 <= minimum_rating <= 5:
            raise ValueError(
                "Minimum rating must be between 1 and 5."
            )

    if any(price not in [1, 2, 3, 4] for price in price_ranges):
        raise ValueError(
            "Price ranges must be between 1 and 4."
        )

    invalid_cuisines = [
        cuisine
        for cuisine in cuisines
        if cuisine not in cuisine_lookup
    ]

    if invalid_cuisines:
        raise ValueError(
            f"Unknown cuisines: {invalid_cuisines}"
        )

    # Generate cuisine preference vector.
    user_vector = np.zeros(
        len(feature_columns),
        dtype=np.float32
    )

    for cuisine in cuisines:

        column = cuisine_lookup[cuisine]

        position = feature_columns.index(column)

        user_vector[position] = 1

    # Calculate similarity.
    if cuisines:

        similarity_scores = cosine_similarity(
            cuisine_matrix,
            user_vector.reshape(1, -1)
        ).ravel()

    else:

        similarity_scores = np.zeros(
            len(metadata),
            dtype=np.float32
        )

    recommendation_data = metadata.copy()

    recommendation_data["Cuisine Similarity"] = (
        similarity_scores
    )

    # Apply city filter.
    if city:

        recommendation_data = recommendation_data[
            recommendation_data["City Normalized"]
            == city.strip().casefold()
        ]

    # Apply locality filter.
    if locality:

        recommendation_data = recommendation_data[
            recommendation_data["Locality Normalized"]
            == locality.strip().casefold()
        ]

    # Apply price filter.
    if price_ranges:

        recommendation_data = recommendation_data[
            recommendation_data["Price range"].isin(
                price_ranges
            )
        ]

    # Apply minimum rating filter.
    if minimum_rating is not None:

        recommendation_data = recommendation_data[
            recommendation_data["Aggregate rating"]
            >= minimum_rating
        ]

    if recommendation_data.empty:
        return recommendation_data

    # Generate ranking score.
    if cuisines:

        recommendation_data["Recommendation Score"] = (
            recommendation_data["Cuisine Similarity"]
            .clip(0.0, 1.0)
        )

    else:

        recommendation_data["Recommendation Score"] = (
            recommendation_data["Aggregate rating"] / 5.0
        ).clip(0.0, 1.0)

    # Apply ranking.
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

    return recommendation_data.head(top_n)


print("Recommendation function prepared.")


# ============================================================
# 5. DEFINE TEST CASES
# ============================================================

print("\n" + "=" * 65)
print("5. DEFINE TEST CASES")
print("=" * 65)

test_cases = [
    {
        "Test ID": "TC01",
        "Test Name": "Single Cuisine Recommendation",
        "Cuisines": ["north indian"],
        "City": "",
        "Locality": "",
        "Price Ranges": [],
        "Minimum Rating": None,
        "Expected": "Recommendations should be generated",
    },
    {
        "Test ID": "TC02",
        "Test Name": "Multiple Cuisine Recommendation",
        "Cuisines": ["north indian", "chinese"],
        "City": "",
        "Locality": "",
        "Price Ranges": [],
        "Minimum Rating": None,
        "Expected": "Recommendations should match cuisine preferences",
    },
    {
        "Test ID": "TC03",
        "Test Name": "City Filter",
        "Cuisines": ["north indian"],
        "City": "new delhi",
        "Locality": "",
        "Price Ranges": [],
        "Minimum Rating": None,
        "Expected": "All recommendations should belong to New Delhi",
    },
    {
        "Test ID": "TC04",
        "Test Name": "Price Filter",
        "Cuisines": ["north indian"],
        "City": "",
        "Locality": "",
        "Price Ranges": [1, 2],
        "Minimum Rating": None,
        "Expected": "All recommendations should have price range 1 or 2",
    },
    {
        "Test ID": "TC05",
        "Test Name": "Minimum Rating Filter",
        "Cuisines": ["north indian"],
        "City": "",
        "Locality": "",
        "Price Ranges": [],
        "Minimum Rating": 4.0,
        "Expected": "All recommendations should have rating at least 4",
    },
    {
        "Test ID": "TC06",
        "Test Name": "Combined Preference Filter",
        "Cuisines": ["north indian", "chinese"],
        "City": "new delhi",
        "Locality": "",
        "Price Ranges": [1, 2],
        "Minimum Rating": 3.5,
        "Expected": "All selected filters should be satisfied",
    },
    {
        "Test ID": "TC07",
        "Test Name": "No Cuisine Preference",
        "Cuisines": [],
        "City": "new delhi",
        "Locality": "",
        "Price Ranges": [],
        "Minimum Rating": None,
        "Expected": "Restaurants should be ranked by rating",
    },
    {
        "Test ID": "TC08",
        "Test Name": "Uncommon Cuisine",
        "Cuisines": ["afghani"],
        "City": "",
        "Locality": "",
        "Price Ranges": [],
        "Minimum Rating": None,
        "Expected": "Valid uncommon cuisine should be processed",
    },
    {
        "Test ID": "TC09",
        "Test Name": "Empty Result Handling",
        "Cuisines": [],
        "City": "__nonexistent_city__",
        "Locality": "",
        "Price Ranges": [],
        "Minimum Rating": None,
        "Expected": "No recommendations should be returned",
    },
    {
        "Test ID": "TC10",
        "Test Name": "Large Recommendation Request",
        "Cuisines": ["north indian"],
        "City": "",
        "Locality": "",
        "Price Ranges": [],
        "Minimum Rating": None,
        "Expected": "Return all available candidates when fewer than requested",
        "Top N": len(metadata) + 100,
    },
]

print(f"Total test cases defined: {len(test_cases)}")


# ============================================================
# 6. EXECUTE AUTOMATED TESTS
# ============================================================

print("\n" + "=" * 65)
print("6. EXECUTE AUTOMATED TESTS")
print("=" * 65)

test_results = []

for test in test_cases:

    test_id = test["Test ID"]
    test_name = test["Test Name"]

    print(f"\nRunning {test_id}: {test_name}")

    status = "PASS"
    details = ""
    
    # Initialize before the try block to avoid an unbound variable.
    recommendations = pd.DataFrame()

    try:

        top_n = test.get("Top N", TOP_N)

        recommendations = generate_recommendations(
            cuisines=test["Cuisines"],
            city=test["City"],
            locality=test["Locality"],
            price_ranges=test["Price Ranges"],
            minimum_rating=test["Minimum Rating"],
            top_n=top_n,
        )

        # TC09: Empty result handling.
        if test_id == "TC09":

            if not recommendations.empty:
                status = "FAIL"
                details = "Expected an empty result."

            else:
                details = "Empty result handled correctly."

        else:

            if recommendations.empty:
                status = "FAIL"
                details = "Expected recommendations but none were returned."

            # Check duplicate IDs.
            if recommendations["Restaurant ID"].duplicated().any():
                status = "FAIL"
                details = "Duplicate restaurant IDs detected."

            # Check city filter.
            if test["City"]:

                if not (
                    recommendations["City Normalized"]
                    == test["City"]
                ).all():

                    status = "FAIL"
                    details = "City filter validation failed."

            # Check locality filter.
            if test["Locality"]:

                if not (
                    recommendations["Locality Normalized"]
                    == test["Locality"]
                ).all():

                    status = "FAIL"
                    details = "Locality filter validation failed."

            # Check price filter.
            if test["Price Ranges"]:

                if not (
                    recommendations["Price range"]
                    .isin(test["Price Ranges"])
                ).all():

                    status = "FAIL"
                    details = "Price filter validation failed."

            # Check rating filter.
            if test["Minimum Rating"] is not None:

                ratings_valid = (
                    recommendations["Aggregate rating"].notna()
                    & (
                        recommendations["Aggregate rating"]
                        >= test["Minimum Rating"]
                    )
                )

                if not ratings_valid.all():

                    status = "FAIL"
                    details = "Minimum rating filter validation failed."

            # Check similarity bounds.
            similarity = recommendations[
                "Cuisine Similarity"
            ]

            if not similarity.between(0, 1).all():

                status = "FAIL"
                details = "Cuisine similarity outside valid range."

            # Check recommendation score bounds.
            scores = recommendations[
                "Recommendation Score"
            ]

            if not scores.between(0, 1).all():

                status = "FAIL"
                details = "Recommendation score outside valid range."

            # Check ranking order.
            ranking_columns = [
                "Recommendation Score",
                "Cuisine Similarity",
                "Aggregate rating",
                "Votes",
            ]

            expected_order = recommendations.sort_values(
                by=ranking_columns,
                ascending=[False, False, False, False],
                na_position="last"
            ).reset_index(drop=True)

            if not recommendations[
                "Restaurant ID"
            ].reset_index(drop=True).equals(
                expected_order["Restaurant ID"]
            ):

                status = "FAIL"
                details = "Recommendation ranking order is incorrect."

            if status == "PASS":

                details = (
                    f"{len(recommendations)} recommendations validated."
                )

    except ValueError as error:

        status = "FAIL"
        details = str(error)

    except Exception as error:

        status = "ERROR"
        details = f"{type(error).__name__}: {error}"

    test_results.append({
        "Test ID": test_id,
        "Test Name": test_name,
        "Expected Behaviour": test["Expected"],
        "Recommendations Generated": (
            len(recommendations)
            if "recommendations" in locals()
            and isinstance(recommendations, pd.DataFrame)
            else 0
        ),
        "Status": status,
        "Details": details,
    })

    print(f"Status: {status}")

    if details:
        print(f"Details: {details}")


# ============================================================
# 7. TEST INVALID USER INPUTS
# ============================================================

print("\n" + "=" * 65)
print("7. TEST INVALID USER INPUTS")
print("=" * 65)

invalid_input_tests = [
    {
        "Test ID": "TC11",
        "Test Name": "Unknown Cuisine",
        "Arguments": {
            "cuisines": ["nonexistent cuisine"],
        },
    },
    {
        "Test ID": "TC12",
        "Test Name": "Invalid Price Range",
        "Arguments": {
            "price_ranges": [5],
        },
    },
    {
        "Test ID": "TC13",
        "Test Name": "Invalid Minimum Rating",
        "Arguments": {
            "minimum_rating": 6,
        },
    },
    {
        "Test ID": "TC14",
        "Test Name": "Invalid Recommendation Count",
        "Arguments": {
            "top_n": 0,
        },
    },
]

for test in invalid_input_tests:

    print(f"\nRunning {test['Test ID']}: {test['Test Name']}")

    try:

        generate_recommendations(
            **test["Arguments"]
        )

        status = "FAIL"
        details = "Invalid input was not rejected."

    except ValueError as error:

        status = "PASS"
        details = f"Invalid input rejected: {error}"

    except Exception as error:

        status = "ERROR"
        details = f"{type(error).__name__}: {error}"

    test_results.append({
        "Test ID": test["Test ID"],
        "Test Name": test["Test Name"],
        "Expected Behaviour": "Invalid input should raise ValueError",
        "Recommendations Generated": 0,
        "Status": status,
        "Details": details,
    })

    print(f"Status: {status}")
    print(f"Details: {details}")


# ============================================================
# 8. GENERATE TEST REPORT
# ============================================================

print("\n" + "=" * 65)
print("8. GENERATE TEST REPORT")
print("=" * 65)

test_results_df = pd.DataFrame(test_results)

test_results_df.to_csv(
    TEST_RESULTS_PATH,
    index=False
)

total_tests = len(test_results_df)

passed_tests = int(
    (test_results_df["Status"] == "PASS").sum()
)

failed_tests = int(
    (test_results_df["Status"] == "FAIL").sum()
)

error_tests = int(
    (test_results_df["Status"] == "ERROR").sum()
)

test_summary = pd.DataFrame([
    {
        "Metric": "Total Test Cases",
        "Value": total_tests,
    },
    {
        "Metric": "Passed Tests",
        "Value": passed_tests,
    },
    {
        "Metric": "Failed Tests",
        "Value": failed_tests,
    },
    {
        "Metric": "Execution Errors",
        "Value": error_tests,
    },
    {
        "Metric": "Test Pass Rate (%)",
        "Value": (
            round(passed_tests / total_tests * 100, 2)
            if total_tests else 0
        ),
    },
])

test_summary.to_csv(
    TEST_SUMMARY_PATH,
    index=False
)

print("Test reports saved successfully.")

print(f"\n1. {TEST_RESULTS_PATH}")
print(f"2. {TEST_SUMMARY_PATH}")


# ============================================================
# 9. DISPLAY TEST RESULTS
# ============================================================

print("\n" + "=" * 65)
print("9. DISPLAY TEST RESULTS")
print("=" * 65)

print(
    test_results_df[
        [
            "Test ID",
            "Test Name",
            "Status",
            "Recommendations Generated",
            "Details",
        ]
    ].to_string(index=False)
)

print("\nTest summary:")

print(test_summary.to_string(index=False))


# ============================================================
# 10. FINAL TEST SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("TASK 02 - RESTAURANT RECOMMENDATION SYSTEM")
print("PROGRAM 06 - RECOMMENDATION SYSTEM TESTING COMPLETED")
print("=" * 75)

print(f"\nTotal test cases: {total_tests}")
print(f"Passed tests: {passed_tests}")
print(f"Failed tests: {failed_tests}")
print(f"Execution errors: {error_tests}")

print(
    f"Test pass rate: "
    f"{passed_tests / total_tests * 100:.2f}%"
    if total_tests else "Test pass rate: 0%"
)

print("\nGenerated reports:")
print("1. recommendation_system_test_results.csv")
print("2. recommendation_system_test_summary.csv")

if failed_tests == 0 and error_tests == 0:

    print("\nPROGRAM 06 EXECUTION SUCCESSFUL.")

else:

    print("\nPROGRAM 06 COMPLETED WITH TEST FAILURES.")

print("=" * 75)
