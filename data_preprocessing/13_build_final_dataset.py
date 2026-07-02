# 13_build_final_dataset.py

import os
import pandas as pd


# ============================================================
# 1. Paths
# ============================================================

NO2_CSV = "data/processed/stream_a_no2_2024_01_clean.csv"
NIGHTLIGHT_CSV = "data/processed/stream_b_nightlight_2024_01_clean.csv"
LABEL_CSV = "data/processed/labels_odiac_2024_01_with_cls.csv"

OUTPUT_DIR = "data/processed"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "final_ms_mca_dataset_2024_01.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. Load data
# ============================================================

no2 = pd.read_csv(NO2_CSV)
nightlight = pd.read_csv(NIGHTLIGHT_CSV)
label = pd.read_csv(LABEL_CSV)

print("NO2 rows:", len(no2))
print("Nightlight rows:", len(nightlight))
print("Label rows:", len(label))


# ============================================================
# 3. Basic column check
# ============================================================

required_base_cols = ["grid_id", "row", "col", "lon", "lat", "date"]

for name, df in [
    ("NO2", no2),
    ("Nightlight", nightlight),
    ("Label", label),
]:
    missing = [col for col in required_base_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{name} file missing columns: {missing}")


if "no2_mean" not in no2.columns:
    raise ValueError("NO2 file must contain 'no2_mean' column.")

if "nightlight_mean" not in nightlight.columns:
    raise ValueError("Nightlight file must contain 'nightlight_mean' column.")

for col in ["label_reg", "label_reg_log", "label_cls"]:
    if col not in label.columns:
        raise ValueError(f"Label file must contain '{col}' column.")


# ============================================================
# 4. Keep only necessary columns
# ============================================================

no2 = no2[
    [
        "grid_id",
        "row",
        "col",
        "lon",
        "lat",
        "date",
        "no2_mean",
    ]
]

nightlight = nightlight[
    [
        "grid_id",
        "date",
        "nightlight_mean",
    ]
]

label = label[
    [
        "grid_id",
        "date",
        "label_reg",
        "label_reg_log",
        "label_cls",
    ]
]


# ============================================================
# 5. Merge datasets
# ============================================================

df = no2.merge(
    nightlight,
    on=["grid_id", "date"],
    how="inner"
)

df = df.merge(
    label,
    on=["grid_id", "date"],
    how="inner"
)


# ============================================================
# 6. Sort and validate
# ============================================================

df = df.sort_values(["col", "row"]).reset_index(drop=True)

print()
print("Final rows:", len(df))
print("Missing values by column:")
print(df.isna().sum())

expected_rows = len(no2)

if len(df) != expected_rows:
    print()
    print("Warning: final row count does not match NO2 row count.")
    print(f"Expected rows: {expected_rows}")
    print(f"Final rows: {len(df)}")


# ============================================================
# 7. Save final dataset
# ============================================================

df.to_csv(OUTPUT_CSV, index=False)

print()
print(f"Saved final dataset to: {OUTPUT_CSV}")
print()
print(df.head())

print()
print("label_cls distribution:")
print(df["label_cls"].value_counts().sort_index())

print()
print("Feature summary:")
print(df[["no2_mean", "nightlight_mean", "label_reg", "label_reg_log"]].describe())