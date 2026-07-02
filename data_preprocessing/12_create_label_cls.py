# 12_create_label_cls.py

import os
import numpy as np
import pandas as pd


# ============================================================
# 1. Paths
# ============================================================

INPUT_CSV = "data/interim/labels_odiac_2024_01.csv"

OUTPUT_DIR = "data/processed"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "labels_odiac_2024_01_with_cls.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. Load label_reg data
# ============================================================

df = pd.read_csv(INPUT_CSV)

print("Rows:", len(df))
print("Missing label_reg:", df["label_reg"].isna().sum())
print("Zero label_reg:", (df["label_reg"] == 0).sum())
print()
print(df["label_reg"].describe())


# ============================================================
# 3. Create log-transformed regression label
# ============================================================
# label_reg has very large outliers.
# log1p keeps zero values as zero and reduces extreme scale.

df["label_reg_log"] = np.log1p(df["label_reg"])


# ============================================================
# 4. Create classification label
# ============================================================
# Rule:
# - label_reg == 0 -> class 0
# - label_reg > 0 -> quantile classes 1~4

df["label_cls"] = 0

nonzero_mask = df["label_reg"] > 0
nonzero_values = df.loc[nonzero_mask, "label_reg"]

# qcut creates 4 quantile bins.
# labels=[1,2,3,4] assigns class numbers.
df.loc[nonzero_mask, "label_cls"] = pd.qcut(
    nonzero_values,
    q=4,
    labels=[1, 2, 3, 4]
).astype(int)


# ============================================================
# 5. Save output
# ============================================================

keep_columns = [
    "grid_id",
    "row",
    "col",
    "lon",
    "lat",
    "date",
    "label_reg",
    "label_reg_log",
    "label_cls",
]

df = df[keep_columns]

df.to_csv(OUTPUT_CSV, index=False)


# ============================================================
# 6. Print summary
# ============================================================

print()
print(f"Saved label file to: {OUTPUT_CSV}")
print()
print(df.head())

print()
print("label_cls distribution:")
print(df["label_cls"].value_counts().sort_index())

print()
print("label_reg_log summary:")
print(df["label_reg_log"].describe())