# 02_check_grid.py

import os
import geopandas as gpd
import matplotlib.pyplot as plt


# ============================================================
# 1. Input / output paths
# ============================================================

GRID_PATH = "data/grid/delhi_ncr_1km_grid.geojson"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

FIG_PATH = os.path.join(OUTPUT_DIR, "delhi_ncr_1km_grid_check.png")


# ============================================================
# 2. Load grid
# ============================================================

grid = gpd.read_file(GRID_PATH)

print("Grid loaded successfully.")
print(f"Number of grid cells: {len(grid)}")
print("CRS:", grid.crs)
print()
print(grid.head())


# ============================================================
# 3. Basic validation
# ============================================================

required_columns = ["grid_id", "row", "col", "lon", "lat"]

missing_columns = [col for col in required_columns if col not in grid.columns]

if missing_columns:
    print("Missing columns:", missing_columns)
else:
    print("All required columns exist.")


print()
print("Longitude range:", grid["lon"].min(), "to", grid["lon"].max())
print("Latitude range:", grid["lat"].min(), "to", grid["lat"].max())


# ============================================================
# 4. Plot grid
# ============================================================

fig, ax = plt.subplots(figsize=(8, 8))

grid.boundary.plot(ax=ax, linewidth=0.2)

ax.set_title("Delhi NCR 1km Grid Check")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_aspect("equal")

plt.tight_layout()
plt.savefig(FIG_PATH, dpi=300)
plt.close()

print()
print(f"Grid check figure saved to: {FIG_PATH}")