# 10_extract_odiac_label.py

import os
import geopandas as gpd
import pandas as pd
import rasterio
from rasterstats import zonal_stats


# ============================================================
# 1. Paths
# ============================================================

GRID_PATH = "data/grid/delhi_ncr_1km_grid.geojson"
ODIAC_TIF_PATH = "data/raw/odiac/odiac2025_1km_excl_intl_2401.tif"

OUTPUT_DIR = "data/interim"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "labels_odiac_2024_01.csv")

DATE_LABEL = "2024-01"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. Load grid
# ============================================================

grid = gpd.read_file(GRID_PATH)

required_columns = ["grid_id", "row", "col", "lon", "lat"]
missing_columns = [col for col in required_columns if col not in grid.columns]

if missing_columns:
    raise ValueError(f"Missing columns in grid file: {missing_columns}")

print(f"Grid loaded: {len(grid)} cells")
print("Grid CRS:", grid.crs)


# ============================================================
# 3. Check ODIAC raster
# ============================================================

with rasterio.open(ODIAC_TIF_PATH) as src:
    print("ODIAC raster loaded successfully.")
    print("ODIAC CRS:", src.crs)
    print("ODIAC resolution:", src.res)
    print("ODIAC bounds:", src.bounds)
    print("ODIAC nodata:", src.nodata)

    raster_crs = src.crs
    affine = src.transform
    nodata = src.nodata


# ============================================================
# 4. Match CRS
# ============================================================

if grid.crs != raster_crs:
    print("Reprojecting grid to ODIAC CRS...")
    grid_for_stats = grid.to_crs(raster_crs)
else:
    grid_for_stats = grid.copy()


# ============================================================
# 5. Zonal statistics
# ============================================================
# We use mean because each 1km grid cell overlaps with ODIAC 1km pixels.
# Later, depending on ODIAC unit, we may decide whether sum is more appropriate.

print("Running zonal statistics...")

stats = zonal_stats(
    vectors=grid_for_stats,
    raster=ODIAC_TIF_PATH,
    stats=["mean", "max", "min"],
    nodata=nodata,
    all_touched=True
)

print("Zonal statistics completed.")


# ============================================================
# 6. Build output DataFrame
# ============================================================

stats_df = pd.DataFrame(stats)

result_df = grid[["grid_id", "row", "col", "lon", "lat"]].copy()
result_df["date"] = DATE_LABEL
result_df["label_reg"] = stats_df["mean"]

# Optional diagnostic columns
result_df["odiac_min"] = stats_df["min"]
result_df["odiac_max"] = stats_df["max"]


# ============================================================
# 7. Save
# ============================================================

result_df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved ODIAC labels to: {OUTPUT_CSV}")
print()
print(result_df.head())
print()
print("label_reg summary:")
print(result_df["label_reg"].describe())
print()
print("Missing label_reg values:", result_df["label_reg"].isna().sum())
print("Zero label_reg values:", (result_df["label_reg"] == 0).sum())