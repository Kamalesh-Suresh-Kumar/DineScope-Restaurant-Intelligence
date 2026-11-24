from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN


# ============================================================
# DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS
# TASK 04: LOCATION-BASED ANALYSIS
# PROGRAM 05: GEOSPATIAL ANALYSIS AND VISUALIZATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "task_4_location_analysis"
    / "features"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "task_4"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "task_4"
)

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 75)
print("DINESCOPE - RESTAURANT INTELLIGENCE & ANALYTICS")
print("TASK 04: LOCATION-BASED ANALYSIS")
print("PROGRAM 05: GEOSPATIAL ANALYSIS AND VISUALIZATION")
print("=" * 75)


# ============================================================
# 1. LOAD RESTAURANT LOCATION DATA
# ============================================================

print("\n[1] LOADING RESTAURANT LOCATION DATA")

input_file = FEATURE_DIR / "restaurant_location_features.csv"

if not input_file.exists():
    raise FileNotFoundError(
        f"Input dataset not found: {input_file}"
    )

df = pd.read_csv(input_file)

required_columns = [
    "Restaurant ID",
    "Country Code",
    "City",
    "Locality",
    "Latitude",
    "Longitude",
    "Aggregate rating",
    "Coordinate Valid"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("Dataset shape:", df.shape)



# ============================================================
# 2. CLEAN AND VALIDATE GEOGRAPHICAL COORDINATES
# ============================================================

print("\n[2] VALIDATING GEOGRAPHICAL COORDINATES")

df["Latitude"] = pd.to_numeric(
    df["Latitude"],
    errors="coerce"
)

df["Longitude"] = pd.to_numeric(
    df["Longitude"],
    errors="coerce"
)

df["Aggregate rating"] = pd.to_numeric(
    df["Aggregate rating"],
    errors="coerce"
)

df["Country Code"] = pd.to_numeric(
    df["Country Code"],
    errors="coerce"
)

coordinate_valid = (
    df["Coordinate Valid"]
    .astype(str)
    .str.strip()
    .str.lower()
    .isin(["true", "1", "yes"])
)

valid_range = (
    df["Latitude"].between(-90, 90)
    & df["Longitude"].between(-180, 180)
)

# Technically valid coordinates, including (0, 0).
technical_valid_mask = (
    coordinate_valid
    & valid_range
    & df["Latitude"].notna()
    & df["Longitude"].notna()
)

technical_valid_df = df.loc[
    technical_valid_mask
].copy()

# Identify suspicious null-island coordinates.
suspicious_coordinate_mask = (
    technical_valid_mask
    & df["Latitude"].eq(0)
    & df["Longitude"].eq(0)
)

suspicious_coordinates_df = df.loc[
    suspicious_coordinate_mask
].copy()

suspicious_coordinates_df[
    "Coordinate_Quality_Status"
] = "Suspicious - Zero Coordinates"

# Exclude suspicious coordinates only from spatial analysis.
usable_coordinate_mask = (
    technical_valid_mask
    & ~suspicious_coordinate_mask
)

geo_df = df.loc[
    usable_coordinate_mask
].copy()

geo_df["Coordinate_Quality_Status"] = "Usable"

if geo_df.empty:
    raise ValueError(
        "No usable geographical coordinates found."
    )

print("Total restaurant records:", len(df))

print(
    "Technically valid coordinate records:",
    len(technical_valid_df)
)

print(
    "Suspicious zero-coordinate records:",
    len(suspicious_coordinates_df)
)

print(
    "Usable coordinate records:",
    len(geo_df)
)

print(
    "Invalid coordinate records:",
    len(df) - len(technical_valid_df)
)

print(
    "Latitude range:",
    geo_df["Latitude"].min(),
    "to",
    geo_df["Latitude"].max()
)

print(
    "Longitude range:",
    geo_df["Longitude"].min(),
    "to",
    geo_df["Longitude"].max()
)

# ============================================================
# 3. GLOBAL RESTAURANT COORDINATE DISTRIBUTION
# ============================================================

print("\n[3] ANALYZING GLOBAL RESTAURANT DISTRIBUTION")

global_coordinate_summary = pd.DataFrame([
    {
        "Metric": "Total valid coordinate records",
        "Value": len(geo_df)
    },
    {
        "Metric": "Unique coordinate pairs",
        "Value": geo_df[
            ["Latitude", "Longitude"]
        ].drop_duplicates().shape[0]
    },
    {
        "Metric": "Unique countries",
        "Value": geo_df["Country Code"].nunique()
    },
    {
        "Metric": "Unique cities",
        "Value": geo_df[
            ["Country Code", "City"]
        ].drop_duplicates().shape[0]
    }
])

print(
    global_coordinate_summary.to_string(index=False)
)


# ============================================================
# 4. GLOBAL GEOGRAPHICAL DISTRIBUTION VISUALIZATION
# ============================================================

print("\n[4] GENERATING GLOBAL DISTRIBUTION VISUALIZATION")

fig, ax = plt.subplots(figsize=(15, 8))

scatter = ax.scatter(
    geo_df["Longitude"],
    geo_df["Latitude"],
    c=geo_df["Aggregate rating"],
    cmap="viridis",
    s=9,
    alpha=0.65,
    edgecolors="none"
)

colorbar = fig.colorbar(
    scatter,
    ax=ax
)

colorbar.set_label("Aggregate Restaurant Rating")

ax.set_title(
    "Global Restaurant Geographical Distribution",
    fontsize=16,
    fontweight="bold"
)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)

ax.grid(
    True,
    alpha=0.25,
    linestyle="--"
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "01_global_restaurant_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print(
    "[SAVED] 01_global_restaurant_distribution.png"
)


# ============================================================
# 5. INDIA-FOCUSED GEOGRAPHICAL DISTRIBUTION
# ============================================================

print("\n[5] ANALYZING INDIA RESTAURANT DISTRIBUTION")

# In this dataset, Country Code 1 represents India.
india_df = geo_df.loc[
    geo_df["Country Code"] == 1
].copy()

if india_df.empty:
    raise ValueError(
        "No India records found for Country Code 1."
    )

print(
    "India restaurant records:",
    len(india_df)
)

print(
    "India cities:",
    india_df["City"].nunique()
)

fig, ax = plt.subplots(figsize=(12, 10))

scatter = ax.scatter(
    india_df["Longitude"],
    india_df["Latitude"],
    c=india_df["Aggregate rating"],
    cmap="viridis",
    s=12,
    alpha=0.65,
    edgecolors="none"
)

colorbar = fig.colorbar(
    scatter,
    ax=ax
)

colorbar.set_label("Aggregate Restaurant Rating")

ax.set_title(
    "Restaurant Geographical Distribution - India",
    fontsize=16,
    fontweight="bold"
)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

ax.grid(
    True,
    alpha=0.25,
    linestyle="--"
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "02_india_restaurant_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print(
    "[SAVED] 02_india_restaurant_distribution.png"
)


# ============================================================
# 6. CITY-WISE GEOGRAPHICAL DISTRIBUTION
# ============================================================

print("\n[6] ANALYZING CITY-WISE GEOGRAPHICAL DISTRIBUTION")

city_coordinate_summary = (
    geo_df.groupby(
        ["Country Code", "City"],
        dropna=False
    )
    .agg(
        Restaurant_Count=(
            "Restaurant ID",
            "nunique"
        ),
        Latitude_Center=(
            "Latitude",
            "mean"
        ),
        Longitude_Center=(
            "Longitude",
            "mean"
        ),
        Rated_Restaurant_Count=(
            "Aggregate rating",
            "count"
        ),
        Average_Rating=(
            "Aggregate rating",
            "mean"
        )
    )
    .reset_index()
)

city_coordinate_summary = (
    city_coordinate_summary.sort_values(
        "Restaurant_Count",
        ascending=False
    )
)

top_20_cities = city_coordinate_summary.head(
    20
).copy()

fig, ax = plt.subplots(figsize=(13, 9))

scatter = ax.scatter(
    city_coordinate_summary["Longitude_Center"],
    city_coordinate_summary["Latitude_Center"],
    s=(
        city_coordinate_summary["Restaurant_Count"]
        .clip(lower=1)
        * 1.5
    ),
    c=city_coordinate_summary["Average_Rating"],
    cmap="plasma",
    alpha=0.7,
    edgecolors="black",
    linewidths=0.3
)

colorbar = fig.colorbar(
    scatter,
    ax=ax
)

colorbar.set_label("Average Restaurant Rating")

ax.set_title(
    "City-Level Restaurant Geographical Distribution",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel("Mean Longitude")
ax.set_ylabel("Mean Latitude")

ax.grid(
    True,
    alpha=0.25,
    linestyle="--"
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "03_city_geographical_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print(
    "[SAVED] 03_city_geographical_distribution.png"
)


# ============================================================
# 7. GEOGRAPHICAL CLUSTERING USING DBSCAN
# ============================================================

print("\n[7] PERFORMING GEOGRAPHICAL CLUSTERING")

# DBSCAN uses the Haversine distance metric.
# Coordinates must be converted from degrees to radians.
#
# EPSILON = 0.5 km
# MINIMUM SAMPLES = 5
#
# A cluster represents restaurants located within a
# geographical neighborhood according to these parameters.

EARTH_RADIUS_KM = 6371.0088

EPSILON_KM = 0.5
MIN_SAMPLES = 5

epsilon_radians = (
    EPSILON_KM / EARTH_RADIUS_KM
)

coordinates_radians = np.radians(
    geo_df[
        [
            "Latitude",
            "Longitude"
        ]
    ].to_numpy()
)

dbscan = DBSCAN(
    eps=epsilon_radians,
    min_samples=MIN_SAMPLES,
    metric="haversine",
    algorithm="ball_tree"
)

cluster_labels = dbscan.fit_predict(
    coordinates_radians
)

geo_df["Geographical_Cluster"] = cluster_labels

cluster_count = len(
    set(cluster_labels) - {-1}
)

noise_count = int(
    (cluster_labels == -1).sum()
)

clustered_count = int(
    (cluster_labels != -1).sum()
)

print("Geographical clusters identified:", cluster_count)
print("Restaurants assigned to clusters:", clustered_count)
print("Restaurants classified as noise:", noise_count)

print(
    "Cluster coverage:",
    round(
        clustered_count / len(geo_df) * 100,
        2
    ),
    "%"
)


# ============================================================
# 8. GEOGRAPHICAL CLUSTER SUMMARY
# ============================================================

print("\n[8] GENERATING GEOGRAPHICAL CLUSTER SUMMARY")

cluster_summary = (
    geo_df.groupby(
        "Geographical_Cluster"
    )
    .agg(
        Restaurant_Count=(
            "Restaurant ID",
            "nunique"
        ),
        Latitude_Center=(
            "Latitude",
            "mean"
        ),
        Longitude_Center=(
            "Longitude",
            "mean"
        ),
        Rated_Restaurant_Count=(
            "Aggregate rating",
            "count"
        ),
        Average_Rating=(
            "Aggregate rating",
            "mean"
        ),
        City_Count=(
            "City",
            "nunique"
        )
    )
    .reset_index()
)

cluster_summary["Cluster_Type"] = np.where(
    cluster_summary["Geographical_Cluster"] == -1,
    "Noise",
    "Cluster"
)

cluster_summary = cluster_summary.sort_values(
    "Restaurant_Count",
    ascending=False
)

print("\nLargest geographical clusters:")

print(
    cluster_summary.loc[
        cluster_summary["Geographical_Cluster"] != -1
    ]
    .head(10)[
        [
            "Geographical_Cluster",
            "Restaurant_Count",
            "Rated_Restaurant_Count",
            "Average_Rating",
            "City_Count"
        ]
    ]
    .round(3)
    .to_string(index=False)
)


# ============================================================
# 9. VISUALIZE GEOGRAPHICAL CLUSTERS
# ============================================================

print("\n[9] VISUALIZING GEOGRAPHICAL CLUSTERS")

fig, ax = plt.subplots(figsize=(15, 9))

noise_df = geo_df.loc[
    geo_df["Geographical_Cluster"] == -1
]

clustered_df = geo_df.loc[
    geo_df["Geographical_Cluster"] != -1
]

if not noise_df.empty:

    ax.scatter(
        noise_df["Longitude"],
        noise_df["Latitude"],
        color="lightgray",
        s=5,
        alpha=0.25,
        label="Noise / Unclustered"
    )

if not clustered_df.empty:

    scatter = ax.scatter(
        clustered_df["Longitude"],
        clustered_df["Latitude"],
        c=clustered_df["Geographical_Cluster"],
        cmap="turbo",
        s=8,
        alpha=0.7,
        edgecolors="none"
    )

    colorbar = fig.colorbar(
        scatter,
        ax=ax
    )

    colorbar.set_label("Geographical Cluster ID")

ax.set_title(
    "DBSCAN Geographical Restaurant Clusters",
    fontsize=16,
    fontweight="bold"
)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

ax.grid(
    True,
    alpha=0.25,
    linestyle="--"
)

ax.legend(loc="best")

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "04_geographical_dbscan_clusters.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print(
    "[SAVED] 04_geographical_dbscan_clusters.png"
)


# ============================================================
# 10. INDIA GEOGRAPHICAL CLUSTER VISUALIZATION
# ============================================================

print("\n[10] VISUALIZING INDIA GEOGRAPHICAL CLUSTERS")

india_clustered = geo_df.loc[
    geo_df["Country Code"] == 1
].copy()

fig, ax = plt.subplots(figsize=(12, 10))

india_noise = india_clustered.loc[
    india_clustered["Geographical_Cluster"] == -1
]

india_groups = india_clustered.loc[
    india_clustered["Geographical_Cluster"] != -1
]

if not india_noise.empty:

    ax.scatter(
        india_noise["Longitude"],
        india_noise["Latitude"],
        color="lightgray",
        s=5,
        alpha=0.3,
        label="Noise / Unclustered"
    )

if not india_groups.empty:

    scatter = ax.scatter(
        india_groups["Longitude"],
        india_groups["Latitude"],
        c=india_groups["Geographical_Cluster"],
        cmap="turbo",
        s=9,
        alpha=0.7,
        edgecolors="none"
    )

    colorbar = fig.colorbar(
        scatter,
        ax=ax
    )

    colorbar.set_label("Global Cluster ID")

ax.set_title(
    "India Restaurant Geographical Clusters",
    fontsize=16,
    fontweight="bold"
)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

ax.grid(
    True,
    alpha=0.25,
    linestyle="--"
)

ax.legend(loc="best")

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "05_india_geographical_clusters.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print(
    "[SAVED] 05_india_geographical_clusters.png"
)


# ============================================================
# 11. COORDINATE OVERLAP ANALYSIS
# ============================================================

print("\n[11] ANALYZING COORDINATE OVERLAPS")

coordinate_groups = (
    geo_df.groupby(
        [
            "Latitude",
            "Longitude"
        ],
        dropna=False
    )
    .agg(
        Restaurant_Count=(
            "Restaurant ID",
            "nunique"
        ),
        City_Count=(
            "City",
            "nunique"
        ),
        Restaurant_IDs=(
            "Restaurant ID",
            lambda values: ", ".join(
                map(str, values)
            )
        ),
        Cities=(
            "City",
            lambda values: ", ".join(
                sorted(set(values.dropna().astype(str)))
            )
        )
    )
    .reset_index()
)

duplicate_coordinates = coordinate_groups.loc[
    coordinate_groups["Restaurant_Count"] > 1
].copy()

duplicate_coordinates = duplicate_coordinates.sort_values(
    "Restaurant_Count",
    ascending=False
)

print(
    "Unique coordinate pairs:",
    len(coordinate_groups)
)

print(
    "Coordinate pairs shared by multiple restaurants:",
    len(duplicate_coordinates)
)

print("\nLargest coordinate overlaps:")

print(
    duplicate_coordinates.head(10)[
        [
            "Latitude",
            "Longitude",
            "Restaurant_Count",
            "City_Count",
            "Cities"
        ]
    ].to_string(index=False)
)


# ============================================================
# 12. COORDINATE OVERLAP VISUALIZATION
# ============================================================

print("\n[12] VISUALIZING COORDINATE OVERLAPS")

overlap_plot_df = coordinate_groups.loc[
    coordinate_groups["Restaurant_Count"] > 1
].copy()

if not overlap_plot_df.empty:

    fig, ax = plt.subplots(figsize=(13, 8))

    scatter = ax.scatter(
        overlap_plot_df["Longitude"],
        overlap_plot_df["Latitude"],
        s=(
            overlap_plot_df["Restaurant_Count"]
            * 5
        ),
        c=overlap_plot_df["Restaurant_Count"],
        cmap="Reds",
        alpha=0.7,
        edgecolors="black",
        linewidths=0.3
    )

    colorbar = fig.colorbar(
        scatter,
        ax=ax
    )

    colorbar.set_label(
        "Restaurants Sharing Coordinate Pair"
    )

    ax.set_title(
        "Geographical Coordinate Overlap Analysis",
        fontsize=15,
        fontweight="bold"
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax.grid(
        True,
        alpha=0.25,
        linestyle="--"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "06_coordinate_overlap_analysis.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        "[SAVED] 06_coordinate_overlap_analysis.png"
    )

else:
    print(
        "No duplicate coordinate pairs available for plotting."
    )


# ============================================================
# 13. RATING ANALYSIS ACROSS GEOGRAPHICAL CLUSTERS
# ============================================================

print("\n[13] ANALYZING RATINGS ACROSS GEOGRAPHICAL CLUSTERS")

rated_cluster_df = geo_df.loc[
    geo_df["Aggregate rating"].notna()
    & (geo_df["Geographical_Cluster"] != -1)
].copy()

cluster_rating_analysis = (
    rated_cluster_df.groupby(
        "Geographical_Cluster"
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

cluster_rating_analysis = cluster_rating_analysis.loc[
    cluster_rating_analysis["Rated_Restaurant_Count"] >= 5
].copy()

cluster_rating_analysis = (
    cluster_rating_analysis.sort_values(
        "Average_Rating",
        ascending=False
    )
)

print(
    "Clusters with at least five rated restaurants:",
    len(cluster_rating_analysis)
)

print("\nHighest-rated geographical clusters:")

print(
    cluster_rating_analysis.head(10)
    .round(3)
    .to_string(index=False)
)

# ============================================================
# 14. GENERATING COUNTRY-WISE GEOGRAPHICAL SUMMARY
# ============================================================

print("\n[14] GENERATING COUNTRY-WISE GEOGRAPHICAL SUMMARY")

# Count actual distinct latitude-longitude pairs per country.
country_coordinate_counts = (
    geo_df[
        [
            "Country Code",
            "Latitude",
            "Longitude"
        ]
    ]
    .drop_duplicates()
    .groupby("Country Code")
    .size()
    .rename("Unique_Coordinate_Pairs")
    .reset_index()
)

country_geo_summary = (
    geo_df.groupby("Country Code")
    .agg(
        Restaurant_Count=(
            "Restaurant ID",
            "nunique"
        ),
        Unique_Cities=(
            "City",
            "nunique"
        ),
        Average_Latitude=(
            "Latitude",
            "mean"
        ),
        Average_Longitude=(
            "Longitude",
            "mean"
        ),
        Rated_Restaurant_Count=(
            "Aggregate rating",
            "count"
        ),
        Average_Rating=(
            "Aggregate rating",
            "mean"
        )
    )
    .reset_index()
)

country_geo_summary = country_geo_summary.merge(
    country_coordinate_counts,
    on="Country Code",
    how="left"
)

country_geo_summary = country_geo_summary.sort_values(
    "Restaurant_Count",
    ascending=False
)

print(
    country_geo_summary[
        [
            "Country Code",
            "Restaurant_Count",
            "Unique_Cities",
            "Unique_Coordinate_Pairs",
            "Average_Latitude",
            "Average_Longitude",
            "Rated_Restaurant_Count",
            "Average_Rating"
        ]
    ]
    .round(3)
    .head(15)
    .to_string(index=False)
)

# ============================================================
# 15. SAVE GEOSPATIAL DATASETS
# ============================================================

print("\n[15] SAVING GEOSPATIAL DATASETS")

geo_df.to_csv(
    REPORT_DIR / "restaurant_geospatial_clusters.csv",
    index=False
)

cluster_summary.to_csv(
    REPORT_DIR / "geospatial_cluster_summary.csv",
    index=False
)

cluster_rating_analysis.to_csv(
    REPORT_DIR / "geospatial_cluster_rating_analysis.csv",
    index=False
)

duplicate_coordinates.to_csv(
    REPORT_DIR / "geospatial_coordinate_overlaps.csv",
    index=False
)

city_coordinate_summary.to_csv(
    REPORT_DIR / "geospatial_city_coordinate_summary.csv",
    index=False
)

country_geo_summary.to_csv(
    REPORT_DIR / "geospatial_country_summary.csv",
    index=False
)

global_coordinate_summary.to_csv(
    REPORT_DIR / "geospatial_global_summary.csv",
    index=False
)

suspicious_coordinates_df.to_csv(
    REPORT_DIR / "geospatial_suspicious_coordinates.csv",
    index=False
)

print("[SAVED] geospatial_suspicious_coordinates.csv")
print("[SAVED] restaurant_geospatial_clusters.csv")
print("[SAVED] geospatial_cluster_summary.csv")
print("[SAVED] geospatial_cluster_rating_analysis.csv")
print("[SAVED] geospatial_coordinate_overlaps.csv")
print("[SAVED] geospatial_city_coordinate_summary.csv")
print("[SAVED] geospatial_country_summary.csv")
print("[SAVED] geospatial_global_summary.csv")


# ============================================================
# 16. SAVE GEOSPATIAL ANALYSIS SUMMARY
# ============================================================

print("\n[16] SAVING GEOSPATIAL ANALYSIS SUMMARY")

analysis_summary = [
    ("Total Restaurant Records", len(df)),
    ("Technically Valid Coordinate Records", len(technical_valid_df)),
    ("Suspicious Zero Coordinate Records", len(suspicious_coordinates_df)),
    ("Usable Coordinate Records", len(geo_df)),
    ("Invalid Coordinate Records", len(df) - len(technical_valid_df)),
    ("Unique Coordinate Pairs", len(coordinate_groups)),
    ("Coordinate Pairs with Multiple Restaurants", len(duplicate_coordinates)),
    ("Unique Countries", geo_df["Country Code"].nunique()),
    (
        "Unique Cities",
        geo_df[["Country Code", "City"]].drop_duplicates().shape[0]
    ),
    ("India Restaurant Records", len(india_df)),
    ("DBSCAN Clusters", cluster_count),
    ("Restaurants Assigned to Clusters", clustered_count),
    ("Restaurants Classified as Noise", noise_count),
    ("DBSCAN Epsilon in Kilometers", EPSILON_KM),
    ("DBSCAN Minimum Samples", MIN_SAMPLES)
]

analysis_summary_df = pd.DataFrame(
    analysis_summary,
    columns=[
        "Metric",
        "Value"
    ]
)

analysis_summary_df.to_csv(
    REPORT_DIR / "geospatial_analysis_summary.csv",
    index=False
)


# ============================================================
# 17. GENERATE TEXT REPORT
# ============================================================

print("\n[17] GENERATING GEOSPATIAL ANALYSIS REPORT")

report_path = (
    REPORT_DIR
    / "geospatial_analysis_report.txt"
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
        "PROGRAM 05: GEOSPATIAL ANALYSIS AND VISUALIZATION\n\n"
    )

    file.write("=" * 70 + "\n")
    file.write("GEOSPATIAL ANALYSIS SUMMARY\n")
    file.write("=" * 70 + "\n\n")

    for metric, value in analysis_summary:
        file.write(
            f"{metric}: {value}\n"
        )

    file.write("\n")
    file.write("=" * 70 + "\n")
    file.write("CLUSTERING CONFIGURATION\n")
    file.write("=" * 70 + "\n\n")

    file.write(
        f"Algorithm: DBSCAN\n"
        f"Distance Metric: Haversine\n"
        f"Earth Radius: {EARTH_RADIUS_KM} km\n"
        f"Epsilon: {EPSILON_KM} km\n"
        f"Minimum Samples: {MIN_SAMPLES}\n"
    )

    file.write("\n")
    file.write("=" * 70 + "\n")
    file.write("INTERPRETATION NOTES\n")
    file.write("=" * 70 + "\n\n")

    file.write(
        "1. Geographical clusters are based on coordinate proximity.\n"
    )

    file.write(
        "2. DBSCAN noise points are not necessarily invalid records.\n"
    )

    file.write(
        "3. Cluster IDs are algorithmic labels, not location names.\n"
    )

    file.write(
        "4. Coordinate overlaps do not automatically indicate errors.\n"
    )

    file.write(
        "5. Cluster ratings are descriptive and do not establish "
        "that geographical proximity causes rating differences.\n"
    )

    file.write(
        "6. City centers are arithmetic coordinate averages, "
        "not exact geographic centroids.\n"
    )

    file.write(
        "7. The visualizations show coordinate distributions "
        "without street maps or administrative boundaries.\n"
    )


# ============================================================
# 18. FINAL VALIDATION
# ============================================================

print("\n[18] FINAL VALIDATION")

checks = {
    "Input dataset available": len(df) > 0,

    "Technically valid coordinates identified": (
        len(technical_valid_df) > 0
    ),

    "Suspicious zero coordinates identified": (
        len(suspicious_coordinates_df) > 0
    ),

    "Suspicious coordinates excluded from spatial analysis": (
        not (
            geo_df["Latitude"].eq(0)
            & geo_df["Longitude"].eq(0)
        ).any()
    ),

    "Usable coordinates available": len(geo_df) > 0,

    "Latitude values within valid range": (
        geo_df["Latitude"].between(-90, 90).all()
    ),

    "Longitude values within valid range": (
        geo_df["Longitude"].between(-180, 180).all()
    ),

    "India records identified": len(india_df) > 0,

    "DBSCAN labels generated": (
        len(geo_df["Geographical_Cluster"]) == len(geo_df)
    ),

    "Cluster summary generated": not cluster_summary.empty,

    "Coordinate overlap analysis generated": (
        not coordinate_groups.empty
    ),

    "Country summary generated": not country_geo_summary.empty,

    "Cluster rating analysis generated": (
        isinstance(cluster_rating_analysis, pd.DataFrame)
    ),

    "Usable restaurant IDs preserved": (
        set(geo_df["Restaurant ID"])
        == set(
            df.loc[usable_coordinate_mask, "Restaurant ID"]
        )
    ),

    "Cluster counts reconcile": (
        int(cluster_summary["Restaurant_Count"].sum())
        == len(geo_df)
    ),

    "Clustered and noise counts reconcile": (
        clustered_count + noise_count == len(geo_df)
    ),

    "Coordinate groups reconcile": (
        int(coordinate_groups["Restaurant_Count"].sum())
        == len(geo_df)
    ),

    "Country coordinate counts reconcile": (
        int(
            country_geo_summary[
                "Unique_Coordinate_Pairs"
            ].sum()
        )
        == len(coordinate_groups)
    )
}

for check, passed in checks.items():

    print(
        f"{'[PASS]' if passed else '[FAIL]'} {check}"
    )

if not all(checks.values()):
    raise RuntimeError(
        "One or more geospatial validation checks failed."
    )


# ============================================================
# 19. VALIDATE GENERATED OUTPUT FILES
# ============================================================

print("\n[19] VALIDATING OUTPUT FILES")

expected_files = [
    REPORT_DIR / "restaurant_geospatial_clusters.csv",
    REPORT_DIR / "geospatial_cluster_summary.csv",
    REPORT_DIR / "geospatial_global_summary.csv",
    REPORT_DIR / "geospatial_cluster_rating_analysis.csv",
    REPORT_DIR / "geospatial_coordinate_overlaps.csv",
    REPORT_DIR / "geospatial_city_coordinate_summary.csv",
    REPORT_DIR / "geospatial_country_summary.csv",
    REPORT_DIR / "geospatial_global_summary.csv",
    REPORT_DIR / "geospatial_suspicious_coordinates.csv",
    REPORT_DIR / "geospatial_analysis_summary.csv",
    REPORT_DIR / "geospatial_analysis_report.txt",
    FIGURE_DIR / "01_global_restaurant_distribution.png",
    FIGURE_DIR / "02_india_restaurant_distribution.png",
    FIGURE_DIR / "03_city_geographical_distribution.png",
    FIGURE_DIR / "04_geographical_dbscan_clusters.png",
    FIGURE_DIR / "05_india_geographical_clusters.png"
]

# Coordinate overlap visualization is conditional.
if not duplicate_coordinates.empty:

    expected_files.append(
        FIGURE_DIR / "06_coordinate_overlap_analysis.png"
    )

all_outputs_exist = True

for output_file in expected_files:

    exists = (
        output_file.exists()
        and output_file.stat().st_size > 0
    )

    print(
        f"{'[PASS]' if exists else '[FAIL]'} "
        f"{output_file.name}"
    )

    if not exists:
        all_outputs_exist = False

if not all_outputs_exist:
    raise RuntimeError(
        "One or more geospatial output files are missing."
    )


# ============================================================
# 20. EXECUTION SUMMARY
# ============================================================

print("\n[20] EXECUTION SUMMARY")

print("Restaurant records analyzed:", len(geo_df))
print("India records analyzed:", len(india_df))
print("Geographical clusters:", cluster_count)
print("Coordinate overlaps:", len(duplicate_coordinates))
print("Figures directory:", FIGURE_DIR)
print("Reports directory:", REPORT_DIR)

print("\n" + "=" * 75)
print("PROGRAM 05 EXECUTION COMPLETED SUCCESSFULLY")
print("=" * 75)
