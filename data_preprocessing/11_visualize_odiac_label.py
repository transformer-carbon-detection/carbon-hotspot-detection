# 11_visualize_odiac_label.py

import os
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Paths
# ============================================================

GRID_PATH = "data/grid/delhi_ncr_1km_grid.geojson"
LABEL_CSV_PATH = "data/interim/labels_odiac_2024_01.csv"

OUTPUT_DIR = "outputs"
OUTPUT_RAW_FIG = os.path.join(OUTPUT_DIR, "odiac_label_2024_01_map.png")
OUTPUT_LOG_FIG = os.path.join(OUTPUT_DIR, "odiac_label_2024_01_log_map.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. Load data
# ============================================================

grid = gpd.read_file(GRID_PATH)
label = pd.read_csv(LABEL_CSV_PATH)

print("Grid cells:", len(grid))
print("Label rows:", len(label))
print("Missing label_reg:", label["label_reg"].isna().sum())
print()
print(label["label_reg"].describe())


# ============================================================
# 3. Merge geometry with label
# ============================================================

merged = grid.merge(
    label[["grid_id", "date", "label_reg"]],
    on="grid_id",
    how="left"
)

print()
print("Merged rows:", len(merged))
print("Merged missing label_reg:", merged["label_reg"].isna().sum())


# ============================================================
# 4. Raw label map
# ============================================================

fig, ax = plt.subplots(figsize=(9, 9))

merged.plot(
    column="label_reg",
    ax=ax,
    legend=True,
    missing_kwds={
        "color": "lightgrey",
        "label": "Missing values"
    }
)

ax.set_title("ODIAC CO2 Emission Label - Delhi NCR, 2024-01")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_aspect("equal")

plt.tight_layout()
plt.savefig(OUTPUT_RAW_FIG, dpi=300)
plt.close()

print(f"Raw ODIAC label map saved to: {OUTPUT_RAW_FIG}")


# ============================================================
# 5. Log-transformed label map
# ============================================================

merged["label_reg_log"] = np.log1p(merged["label_reg"])

fig, ax = plt.subplots(figsize=(9, 9))

merged.plot(
    column="label_reg_log",
    ax=ax,
    legend=True,
    missing_kwds={
        "color": "lightgrey",
        "label": "Missing values"
    }
)

ax.set_title("ODIAC CO2 Emission Label log1p - Delhi NCR, 2024-01")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_aspect("equal")

plt.tight_layout()
plt.savefig(OUTPUT_LOG_FIG, dpi=300)
plt.close()

print(f"Log ODIAC label map saved to: {OUTPUT_LOG_FIG}")