# 03_extract_n02.py

import os
import math
import ee
import geemap
import geopandas as gpd
import pandas as pd


# ============================================================
# 1. Settings
# ============================================================

PROJECT_ID = "carbon-detection"

GRID_PATH = "data/grid/delhi_ncr_1km_grid.geojson"
OUTPUT_DIR = "data/interim"
BATCH_DIR = os.path.join(OUTPUT_DIR, "no2_batches_2024_01")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "stream_a_no2_2024_01.csv")

START_DATE = "2024-01-01"
END_DATE = "2024-02-01"
DATE_LABEL = "2024-01"

NO2_COLLECTION = "COPERNICUS/S5P/OFFL/L3_NO2"
NO2_BAND = "tropospheric_NO2_column_number_density"

BATCH_SIZE = 500


# ============================================================
# 2. Initialize Earth Engine
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BATCH_DIR, exist_ok=True)

ee.Initialize(project=PROJECT_ID)

print("Google Earth Engine initialized successfully.")
print(f"Project ID: {PROJECT_ID}")


# ============================================================
# 3. Load local grid
# ============================================================

grid = gpd.read_file(GRID_PATH)

required_columns = ["grid_id", "row", "col", "lon", "lat"]
missing_columns = [col for col in required_columns if col not in grid.columns]

if missing_columns:
    raise ValueError(f"Missing columns in grid file: {missing_columns}")

grid = grid.reset_index(drop=True)

print(f"Total grid cells: {len(grid)}")


# ============================================================
# 4. Load Sentinel-5P NO2 monthly mean image
# ============================================================

no2_collection = (
    ee.ImageCollection(NO2_COLLECTION)
    .filterDate(START_DATE, END_DATE)
    .select(NO2_BAND)
)

count = no2_collection.size().getInfo()
print(f"Number of Sentinel-5P NO2 images: {count}")

if count == 0:
    raise ValueError("No Sentinel-5P NO2 images found for the selected date range.")

no2_monthly_mean = no2_collection.mean().rename("no2_mean")


# ============================================================
# 5. Batch extraction function
# ============================================================

def extract_batch(batch_gdf, batch_idx):
    batch_path = os.path.join(BATCH_DIR, f"no2_batch_{batch_idx:03d}.csv")

    # Skip if batch already exists
    if os.path.exists(batch_path):
        print(f"[Batch {batch_idx:03d}] Already exists. Skipping.")
        return pd.read_csv(batch_path)

    print(f"[Batch {batch_idx:03d}] Extracting {len(batch_gdf)} cells...")

    batch_ee = geemap.geopandas_to_ee(batch_gdf)

    reduced = no2_monthly_mean.reduceRegions(
        collection=batch_ee,
        reducer=ee.Reducer.mean(),
        scale=1000
    )

    result_gdf = geemap.ee_to_gdf(reduced)

    result_df = pd.DataFrame(result_gdf.drop(columns="geometry"))

    if "mean" in result_df.columns and "no2_mean" not in result_df.columns:
        result_df = result_df.rename(columns={"mean": "no2_mean"})

    result_df["date"] = DATE_LABEL

    keep_columns = [
        "grid_id",
        "row",
        "col",
        "lon",
        "lat",
        "date",
        "no2_mean",
    ]

    result_df = result_df[keep_columns]
    result_df.to_csv(batch_path, index=False)

    print(f"[Batch {batch_idx:03d}] Saved: {batch_path}")

    return result_df


# ============================================================
# 6. Run batch extraction
# ============================================================

num_batches = math.ceil(len(grid) / BATCH_SIZE)
print(f"Batch size: {BATCH_SIZE}")
print(f"Number of batches: {num_batches}")

all_batches = []

for batch_idx in range(num_batches):
    start_idx = batch_idx * BATCH_SIZE
    end_idx = min((batch_idx + 1) * BATCH_SIZE, len(grid))

    batch_gdf = grid.iloc[start_idx:end_idx].copy()
    batch_df = extract_batch(batch_gdf, batch_idx)

    all_batches.append(batch_df)


# ============================================================
# 7. Merge all batches
# ============================================================

final_df = pd.concat(all_batches, ignore_index=True)

# Sort by row/col or grid_id for consistency
final_df = final_df.sort_values(["col", "row"]).reset_index(drop=True)

final_df.to_csv(OUTPUT_CSV, index=False)


# ============================================================
# 8. Print summary
# ============================================================

print()
print("NO2 batch extraction completed.")
print(f"Saved final CSV to: {OUTPUT_CSV}")
print(f"Total rows: {len(final_df)}")
print(f"Missing NO2 values: {final_df['no2_mean'].isna().sum()}")
print()
print(final_df.head())
print()
print(final_df["no2_mean"].describe())