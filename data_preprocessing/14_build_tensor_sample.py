# 14_build_tensor_sample.py

import os
import numpy as np
import pandas as pd


# ============================================================
# 1. Paths
# ============================================================

INPUT_CSV = "data/processed/final_ms_mca_dataset_2024_01.csv"

OUTPUT_DIR = "data/processed/tensors"
OUTPUT_NPZ = os.path.join(OUTPUT_DIR, "ms_mca_sample_2024_01.npz")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. Load final dataset
# ============================================================

df = pd.read_csv(INPUT_CSV)

print("Rows:", len(df))
print("Missing values:")
print(df.isna().sum())


# ============================================================
# 3. Determine grid shape
# ============================================================

H = int(df["row"].max()) + 1
W = int(df["col"].max()) + 1

print()
print(f"Grid shape H x W: {H} x {W}")


# ============================================================
# 4. Initialize arrays
# ============================================================
# Fill with 0 initially.
# Some row/col locations may be outside the clipped bbox.

no2_grid = np.zeros((H, W), dtype=np.float32)
nightlight_grid = np.zeros((H, W), dtype=np.float32)

label_cls_grid = np.zeros((H, W), dtype=np.int64)
label_reg_grid = np.zeros((H, W), dtype=np.float32)

mask_grid = np.zeros((H, W), dtype=np.uint8)


# ============================================================
# 5. Fill arrays by row/col
# ============================================================

for _, r in df.iterrows():
    row = int(r["row"])
    col = int(r["col"])

    no2_grid[row, col] = r["no2_mean"]
    nightlight_grid[row, col] = r["nightlight_mean"]

    label_cls_grid[row, col] = int(r["label_cls"])
    label_reg_grid[row, col] = r["label_reg_log"]

    mask_grid[row, col] = 1


# ============================================================
# 6. Build stream tensors
# ============================================================
# Shape:
# stream_a: [C_A, H, W]
# stream_b: [C_B, H, W]

stream_a = np.stack(
    [
        no2_grid,
    ],
    axis=0
).astype(np.float32)

stream_b = np.stack(
    [
        nightlight_grid,
    ],
    axis=0
).astype(np.float32)


# ============================================================
# 7. Save tensor sample
# ============================================================

np.savez_compressed(
    OUTPUT_NPZ,
    stream_a=stream_a,
    stream_b=stream_b,
    label_cls=label_cls_grid,
    label_reg=label_reg_grid,
    mask=mask_grid,
)

print()
print(f"Saved tensor sample to: {OUTPUT_NPZ}")

print()
print("Tensor shapes:")
print("stream_a:", stream_a.shape)
print("stream_b:", stream_b.shape)
print("label_cls:", label_cls_grid.shape)
print("label_reg:", label_reg_grid.shape)
print("mask:", mask_grid.shape)

print()
print("Valid grid cells:", mask_grid.sum())
print("Total tensor cells:", H * W)