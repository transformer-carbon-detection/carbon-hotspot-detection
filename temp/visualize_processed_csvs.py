# visualize_processed_csvs.py

import argparse
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


# Keep Matplotlib cache inside the project/temp area when the user home cache is
# not writable.
MPLCONFIGDIR = Path("outputs") / ".matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt


# ============================================================
# 1. Defaults
# ============================================================

GRID_PATH = "data/grid/delhi_ncr_1km_grid.geojson"
PROCESSED_DIR = "data/processed"
OUTPUT_DIR = "outputs/processed_maps"

ID_COLUMNS = {"grid_id", "row", "col", "lon", "lat", "date"}
PLOT_LOG1P_COLUMNS = {"label_reg"}


# ============================================================
# 2. Helpers
# ============================================================

def find_value_columns(df: pd.DataFrame, all_columns: bool) -> list[str]:
    numeric_columns = [
        column
        for column in df.columns
        if column not in ID_COLUMNS and pd.api.types.is_numeric_dtype(df[column])
    ]
    if not numeric_columns:
        return []

    if all_columns:
        return numeric_columns

    preferred_columns = [
        column
        for column in numeric_columns
        if (
            column.endswith("_mean")
            or column.endswith("_sum")
            or column in {"label_reg", "label_reg_log", "urban_fraction"}
        )
    ]
    if preferred_columns:
        return [preferred_columns[0]]

    return [numeric_columns[0]]


def plot_column_name(value_column: str) -> str:
    if value_column in PLOT_LOG1P_COLUMNS:
        return f"{value_column}_log1p_for_plot"
    return value_column


def plot_label(value_column: str) -> str:
    if value_column in PLOT_LOG1P_COLUMNS:
        return f"log1p({value_column})"
    return value_column


def make_output_path(csv_path: Path, processed_dir: Path, value_column: str, output_dir: Path) -> Path:
    relative_name = csv_path.relative_to(processed_dir).with_suffix("")
    safe_name = "__".join(relative_name.parts)
    return output_dir / f"{safe_name}__{plot_column_name(value_column)}_map.png"


def make_title(csv_path: Path, value_column: str, date_value: str | None) -> str:
    name = csv_path.stem.replace("_", " ")
    if date_value:
        return f"{plot_label(value_column)} - {name} ({date_value})"
    return f"{plot_label(value_column)} - {name}"


def plot_column(
    grid: gpd.GeoDataFrame,
    df: pd.DataFrame,
    csv_path: Path,
    processed_dir: Path,
    output_dir: Path,
    value_column: str,
) -> bool:
    keep_columns = ["grid_id", value_column]
    if "date" in df.columns:
        keep_columns.insert(1, "date")

    merged = grid.merge(df[keep_columns], on="grid_id", how="left")
    map_column = value_column
    if value_column in PLOT_LOG1P_COLUMNS:
        map_column = plot_column_name(value_column)
        merged[map_column] = np.log1p(merged[value_column].clip(lower=0))

    output_fig = make_output_path(csv_path, processed_dir, value_column, output_dir)
    output_fig.parent.mkdir(parents=True, exist_ok=True)

    date_value = None
    if "date" in df.columns and not df["date"].dropna().empty:
        date_value = str(df["date"].dropna().iloc[0])

    print("-" * 60)
    print(f"Value column: {value_column}")
    if map_column != value_column:
        print(f"Plot column: {map_column} ({plot_label(value_column)})")
    print(f"Missing {value_column}: {df[value_column].isna().sum()}")
    print(f"Merged missing {map_column}: {merged[map_column].isna().sum()}")

    fig, ax = plt.subplots(figsize=(9, 9))
    merged.plot(
        column=map_column,
        ax=ax,
        legend=True,
        missing_kwds={
            "color": "lightgrey",
            "label": "Missing values",
        },
    )

    ax.set_title(make_title(csv_path, value_column, date_value))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_fig, dpi=300)
    plt.close()

    print(f"Map saved to: {output_fig}")
    print()
    if map_column != value_column:
        print(merged[map_column].describe())
    else:
        print(df[value_column].describe())
    return True


def visualize_csv(
    grid: gpd.GeoDataFrame,
    csv_path: Path,
    processed_dir: Path,
    output_dir: Path,
    all_columns: bool,
) -> int:
    df = pd.read_csv(csv_path)

    if "grid_id" not in df.columns:
        print(f"[SKIP] {csv_path}: missing grid_id column")
        return 0

    value_columns = find_value_columns(df, all_columns)
    if not value_columns:
        print(f"[SKIP] {csv_path}: no numeric feature column found")
        return 0

    print("=" * 60)
    print(f"CSV: {csv_path}")
    print(f"Rows: {len(df)}")
    print(f"Value columns: {', '.join(value_columns)}")

    saved_count = 0
    for value_column in value_columns:
        if plot_column(grid, df, csv_path, processed_dir, output_dir, value_column):
            saved_count += 1
    return saved_count


# ============================================================
# 3. Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize CSV files under data/processed as map images.",
    )
    parser.add_argument("--grid", default=GRID_PATH, help="Grid GeoJSON path.")
    parser.add_argument("--processed-dir", default=PROCESSED_DIR, help="Directory containing processed CSV files.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for output PNG maps.")
    parser.add_argument(
        "--all-columns",
        action="store_true",
        help="Create one map for every numeric value column in each CSV.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional maximum number of CSV files to visualize.",
    )
    args = parser.parse_args()

    grid_path = Path(args.grid)
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)

    if not grid_path.exists():
        raise FileNotFoundError(f"Grid file not found: {grid_path}")
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed directory not found: {processed_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    grid = gpd.read_file(grid_path)
    csv_paths = sorted(processed_dir.rglob("*.csv"))
    if args.max_files is not None:
        csv_paths = csv_paths[: args.max_files]

    print("Grid cells:", len(grid))
    print("CSV files found:", len(csv_paths))
    print("Output directory:", output_dir)

    saved_count = 0
    for csv_path in csv_paths:
        saved_count += visualize_csv(
            grid=grid,
            csv_path=csv_path,
            processed_dir=processed_dir,
            output_dir=output_dir,
            all_columns=args.all_columns,
        )

    print("=" * 60)
    print(f"Saved maps: {saved_count}")


if __name__ == "__main__":
    main()
