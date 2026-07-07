# 07_visualize_nightlight.py

import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Paths
# ============================================================

GRID_PATH = "data/grid/delhi_ncr_1km_grid.geojson"
NTL_CSV_PATH = "data/interim/stream_b_nightlight_2025_01.csv"

OUTPUT_DIR = "outputs"
OUTPUT_FIG = os.path.join(OUTPUT_DIR, "nightlight_2025_01_map.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. Load data
# ============================================================

grid = gpd.read_file(GRID_PATH)
ntl = pd.read_csv(NTL_CSV_PATH)

print("Grid cells:", len(grid))
print("Nightlight rows:", len(ntl))
print("Missing nightlight values:", ntl["nightlight_mean"].isna().sum())


# ============================================================
# 3. Merge grid geometry with nightlight values
# ============================================================

merged = grid.merge(
    ntl[["grid_id", "date", "nightlight_mean"]],
    on="grid_id",
    how="left"
)

print("Merged rows:", len(merged))
print("Merged missing nightlight values:", merged["nightlight_mean"].isna().sum())


# ============================================================
# 4. Plot nightlight map
# ============================================================

fig, ax = plt.subplots(figsize=(9, 9))

merged.plot(
    column="nightlight_mean",
    ax=ax,
    legend=True,
    missing_kwds={
        "color": "lightgrey",
        "label": "Missing values"
    }
)

ax.set_title("VIIRS Nighttime Light Mean - Delhi NCR, 2025-01")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_aspect("equal")

plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=300)
plt.close()

print(f"Nighttime Light map saved to: {OUTPUT_FIG}")


# ============================================================
# 5. Print summary statistics
# ============================================================

print()
print(ntl["nightlight_mean"].describe())