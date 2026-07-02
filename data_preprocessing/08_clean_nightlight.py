# 08_clean_nightlight.py

import os
import pandas as pd


# ============================================================
# 1. Paths
# ============================================================

INPUT_CSV = "data/interim/stream_b_nightlight_2024_01.csv"
OUTPUT_DIR = "data/processed"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "stream_b_nightlight_2024_01_clean.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. Load Nighttime Light data
# ============================================================

df = pd.read_csv(INPUT_CSV)

print("Rows:", len(df))
print("Missing before:", df["nightlight_mean"].isna().sum())


# ============================================================
# 3. Fill missing values
# ============================================================

fill_value = df["nightlight_mean"].mean()

df["nightlight_mean"] = df["nightlight_mean"].fillna(fill_value)

print("Nightlight fill value:", fill_value)
print("Missing after:", df["nightlight_mean"].isna().sum())


# ============================================================
# 4. Save cleaned data
# ============================================================

df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved cleaned Nighttime Light data to: {OUTPUT_CSV}")
print()
print(df.head())
print()
print(df["nightlight_mean"].describe())