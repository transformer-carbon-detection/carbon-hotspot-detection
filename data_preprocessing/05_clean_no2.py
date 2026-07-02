# 05_clean_no2.py

import os
import pandas as pd


# ============================================================
# 1. Paths
# ============================================================

INPUT_CSV = "data/interim/stream_a_no2_2024_01.csv"
OUTPUT_DIR = "data/processed"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "stream_a_no2_2024_01_clean.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. Load NO2 data
# ============================================================

df = pd.read_csv(INPUT_CSV)

print("Rows:", len(df))
print("Missing before:", df["no2_mean"].isna().sum())


# ============================================================
# 3. Fill missing values
# ============================================================

no2_mean_value = df["no2_mean"].mean()

df["no2_mean"] = df["no2_mean"].fillna(no2_mean_value)

print("NO2 fill value:", no2_mean_value)
print("Missing after:", df["no2_mean"].isna().sum())


# ============================================================
# 4. Save cleaned data
# ============================================================

df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved cleaned NO2 data to: {OUTPUT_CSV}")
print()
print(df.head())
print()
print(df["no2_mean"].describe())