# 04_visualize_no2.py

import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Paths
# ============================================================

GRID_PATH = "data/grid/delhi_ncr_1km_grid.geojson"
NO2_CSV_PATH = "data/interim/stream_a_no2_2022_01.csv"

OUTPUT_DIR = "outputs"
OUTPUT_FIG = os.path.join(OUTPUT_DIR, "no2_2025_01_map.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. Load data
# ============================================================

grid = gpd.read_file(GRID_PATH)
no2 = pd.read_csv(NO2_CSV_PATH)

print("Grid cells:", len(grid))
print("NO2 rows:", len(no2))
print("Missing NO2 values:", no2["no2_mean"].isna().sum())


# ============================================================
# 3. Merge grid geometry with NO2 values
# ============================================================

merged = grid.merge(
    no2[["grid_id", "date", "no2_mean"]],
    on="grid_id",
    how="left"
)

print("Merged rows:", len(merged))
print("Merged missing NO2 values:", merged["no2_mean"].isna().sum())


# ============================================================
# 4. Plot NO2 map
# ============================================================

fig, ax = plt.subplots(figsize=(9, 9))

merged.plot(
    column="no2_mean",
    ax=ax,
    legend=True,
    missing_kwds={
        "color": "lightgrey",
        "label": "Missing values"
    }
)

ax.set_title("Sentinel-5P NO2 Mean - Delhi NCR, 2025-01")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_aspect("equal")

plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=300)
plt.close()

print(f"NO2 map saved to: {OUTPUT_FIG}")


# ============================================================
# 5. Print summary statistics
# ============================================================

print()
print(no2["no2_mean"].describe())