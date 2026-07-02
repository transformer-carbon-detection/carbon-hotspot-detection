# 01_create_grid.py
# Delhi NCR 영역을 1km × 1km grid로 나누고,
# 각 grid cell에 grid_id, row, col, lon, lat를 부여해서 저장한다.

import os
import numpy as np
import geopandas as gpd
from shapely.geometry import box


# ============================================================
# 1. Output path
# ============================================================

OUTPUT_DIR = "data/grid"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GEOJSON_PATH = os.path.join(OUTPUT_DIR, "delhi_ncr_1km_grid.geojson")
CSV_PATH = os.path.join(OUTPUT_DIR, "delhi_ncr_1km_grid.csv")


# ============================================================
# 2. Delhi NCR approximate bounding box
# ============================================================
# This is an approximate region for initial preprocessing.
# Later, this can be replaced with an official Delhi NCR boundary polygon.

MIN_LON = 76.5
MAX_LON = 77.8
MIN_LAT = 28.1
MAX_LAT = 29.3


# ============================================================
# 3. Create bounding box in WGS84
# ============================================================

bbox_wgs84 = gpd.GeoDataFrame(
    {"name": ["Delhi_NCR_bbox"]},
    geometry=[box(MIN_LON, MIN_LAT, MAX_LON, MAX_LAT)],
    crs="EPSG:4326"
)


# ============================================================
# 4. Convert to UTM Zone 43N
# ============================================================
# Delhi region is located in UTM Zone 43N.
# EPSG:32643 allows us to create a 1000m x 1000m grid accurately.

bbox_utm = bbox_wgs84.to_crs("EPSG:32643")


# ============================================================
# 5. Create 1km grid
# ============================================================

minx, miny, maxx, maxy = bbox_utm.total_bounds

cell_size = 1000  # meters

grid_cells = []
row_indices = []
col_indices = []

x_coords = np.arange(minx, maxx, cell_size)
y_coords = np.arange(miny, maxy, cell_size)

for col, x in enumerate(x_coords):
    for row, y in enumerate(y_coords):
        cell = box(x, y, x + cell_size, y + cell_size)
        grid_cells.append(cell)
        row_indices.append(row)
        col_indices.append(col)


grid_utm = gpd.GeoDataFrame(
    {
        "row": row_indices,
        "col": col_indices,
    },
    geometry=grid_cells,
    crs="EPSG:32643"
)


# ============================================================
# 6. Clip grid to bounding box
# ============================================================

grid_utm = gpd.overlay(grid_utm, bbox_utm, how="intersection")


# ============================================================
# 7. Add grid_id
# ============================================================

grid_utm = grid_utm.reset_index(drop=True)
grid_utm["grid_id"] = [f"DELHI_{i:05d}" for i in range(len(grid_utm))]


# ============================================================
# 8. Add centroid lon/lat
# ============================================================

centroids_wgs84 = grid_utm.geometry.centroid.to_crs("EPSG:4326")

grid_utm["lon"] = centroids_wgs84.x
grid_utm["lat"] = centroids_wgs84.y


# ============================================================
# 9. Save outputs
# ============================================================

grid_wgs84 = grid_utm.to_crs("EPSG:4326")

grid_wgs84.to_file(GEOJSON_PATH, driver="GeoJSON")

grid_csv = grid_wgs84.drop(columns="geometry")
grid_csv.to_csv(CSV_PATH, index=False)


# ============================================================
# 10. Print summary
# ============================================================

print("Delhi NCR 1km grid created successfully.")
print(f"Number of grid cells: {len(grid_wgs84)}")
print(f"GeoJSON saved to: {GEOJSON_PATH}")
print(f"CSV saved to: {CSV_PATH}")
print()
print(grid_csv.head())