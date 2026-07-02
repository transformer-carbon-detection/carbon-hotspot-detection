# 06_extract_nightlight.py
# almost same as 03_extract_no2.py

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
BATCH_DIR = os.path.join(OUTPUT_DIR, "nightlight_batches_2024_01")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "stream_b_nightlight_2024_01.csv")

START_DATE = "2024-01-01"
END_DATE = "2024-02-01"
DATE_LABEL = "2024-01"

NTL_COLLECTION = "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG"
NTL_BAND = "avg_rad"

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
# 4. Load VIIRS Nighttime Light monthly image
# ============================================================

ntl_collection = (
    ee.ImageCollection(NTL_COLLECTION)
    .filterDate(START_DATE, END_DATE)
    .select(NTL_BAND)
)

count = ntl_collection.size().getInfo()
print(f"Number of VIIRS Nighttime Light images: {count}")

if count == 0:
    raise ValueError("No VIIRS Nighttime Light images found for the selected date range.")

# Usually there is one monthly composite image.
# mean() is used for safety in case multiple images exist.
ntl_monthly_mean = ntl_collection.mean().rename("nightlight_mean")


# ============================================================
# 5. Batch extraction function
# ============================================================

def extract_batch(batch_gdf, batch_idx):
    batch_path = os.path.join(BATCH_DIR, f"nightlight_batch_{batch_idx:03d}.csv")

    if os.path.exists(batch_path):
        print(f"[Batch {batch_idx:03d}] Already exists. Skipping.")
        return pd.read_csv(batch_path)

    print(f"[Batch {batch_idx:03d}] Extracting {len(batch_gdf)} cells...")

    batch_ee = geemap.geopandas_to_ee(batch_gdf)

    reduced = ntl_monthly_mean.reduceRegions(
        collection=batch_ee,
        reducer=ee.Reducer.mean(),
        scale=500
    )

    result_gdf = geemap.ee_to_gdf(reduced)

    result_df = pd.DataFrame(result_gdf.drop(columns="geometry"))

    if "mean" in result_df.columns and "nightlight_mean" not in result_df.columns:
        result_df = result_df.rename(columns={"mean": "nightlight_mean"})

    result_df["date"] = DATE_LABEL

    keep_columns = [
        "grid_id",
        "row",
        "col",
        "lon",
        "lat",
        "date",
        "nightlight_mean",
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

final_df = final_df.sort_values(["col", "row"]).reset_index(drop=True)

final_df.to_csv(OUTPUT_CSV, index=False)


# ============================================================
# 8. Print summary
# ============================================================

print()
print("Nighttime Light batch extraction completed.")
print(f"Saved final CSV to: {OUTPUT_CSV}")
print(f"Total rows: {len(final_df)}")
print(f"Missing nightlight values: {final_df['nightlight_mean'].isna().sum()}")
print()
print(final_df.head())
print()
print(final_df["nightlight_mean"].describe())