# Transformer-Based Carbon Emission Hotspot Detection

This project builds a preprocessing pipeline for carbon emission hotspot
detection using multi-modal satellite and spatial proxy data.

The first real preprocessing target is monthly training data for Delhi NCR from
2020-01 to 2024-12. Monthly aggregation is intentional: the ODIAC target label is
monthly emission intensity in tonne C/km2/month, so NO2 and nighttime light
inputs are aligned as monthly means.

## Project Structure

```text
.
|-- configs/
|   `-- delhi_2020_2024.yaml
|-- src/
|   |-- data/
|   |   `-- spatial_grid_dataset.py
|   |-- preprocessing/
|   |   |-- grid.py
|   |   |-- gee_extract.py
|   |   |-- cleaning.py
|   |   |-- raster_label.py
|   |   |-- labeling.py
|   |   |-- merge.py
|   |   `-- tensor_builder.py
|   `-- utils/
|       |-- config.py
|       |-- paths.py
|       `-- validation.py
|-- scripts/
|   |-- run_preprocessing.py
|   `-- check_odiac_tif.py
|-- data/
|   |-- raw/
|   |   `-- odiac/
|   |-- grid/
|   |-- interim/
|   `-- processed/
`-- data_preprocessing/
```

The old numbered scripts remain available as development history, while the new
pipeline exposes reusable project-level functions.

## Pipeline Overview

The main pipeline performs these steps:

1. Create a 1 km x 1 km Delhi NCR grid.
2. Extract monthly mean Sentinel-5P TROPOMI NO2, SO2, and CO from Google Earth Engine.
3. Extract monthly mean VIIRS nighttime light from Google Earth Engine.
4. Extract yearly WorldPop population and MODIS urban fraction as monthly CSVs.
5. Extract static WRI GPPD power plant features as monthly CSVs.
6. Clean missing image feature values.
7. Extract ODIAC monthly CO2 labels from GeoTIFF files.
8. Create 5-class hotspot labels using global 2020-2024 quantile thresholds.
9. Merge features and labels into monthly final CSVs.
10. Build MS-MCA-compatible `.npz` tensor samples.
11. Build combined CSV and tensor index files for the full range.

## Expected Input Files

ODIAC monthly GeoTIFF files should be placed in:

```text
data/raw/odiac/
```

The default filename pattern is:

```text
data/raw/odiac/odiac2025_1km_excl_intl_{yy}{mm}.tif
```

Examples:

```text
2020-01 -> data/raw/odiac/odiac2025_1km_excl_intl_2001.tif
2024-12 -> data/raw/odiac/odiac2025_1km_excl_intl_2412.tif
```

Google Earth Engine access must be authenticated before running feature
extraction.

## Run The Pipeline

```bash
python scripts/run_preprocessing.py --config configs/delhi_2020_2024.yaml
```

The config controls all run flags. To run only selected stages, edit the `run`
section in `configs/delhi_2020_2024.yaml`.

## Expected Outputs

For each month, the pipeline writes:

```text
data/interim/stream_a_no2_YYYY_MM.csv
data/interim/stream_a_so2_YYYY_MM.csv
data/interim/stream_a_co_YYYY_MM.csv
data/interim/stream_b_nightlight_YYYY_MM.csv
data/interim/stream_b_population_YYYY_MM.csv
data/interim/stream_b_urban_fraction_YYYY_MM.csv
data/processed/stream_a_no2_YYYY_MM_clean.csv
data/processed/stream_a_so2_YYYY_MM_clean.csv
data/processed/stream_a_co_YYYY_MM_clean.csv
data/processed/stream_b_nightlight_YYYY_MM_clean.csv
data/processed/stream_b_population_YYYY_MM_clean.csv
data/processed/stream_b_urban_fraction_YYYY_MM_clean.csv
data/processed/stream_b_power_plants_YYYY_MM_clean.csv
data/processed/odiac_label_YYYY_MM.csv
data/processed/odiac_label_YYYY_MM_cls.csv
data/processed/final_ms_mca_dataset_YYYY_MM.csv
data/processed/ms_mca_tensor_delhi_ncr_YYYY_MM.npz
```

The full-range pipeline also writes:

```text
data/processed/label_thresholds_2020_2024_global.csv
data/processed/final_ms_mca_dataset_2020_2024.csv
data/processed/tensor_index_2020_2024.csv
```

## Tensor Contents

Each monthly tensor `.npz` contains:

- `stream_a`: `[C_A, H, W]`
- `stream_b`: `[C_B, H, W]`
- `label_cls`: `[H, W]`
- `label_reg`: `[H, W]`
- `mask`: `[H, W]`

Current streams:

- `stream_a`: `no2_mean`, `so2_mean`, `co_mean`
- `stream_b`: `nightlight_mean`, `population_sum`, `urban_fraction`, `power_plant_count`, `fossil_capacity_mw`

The config keeps `temporal.mode: monthly_mean` for now and records
`daily_sequence` as a future supported mode.

## MS-MCA Tensor Dataset Format

Monthly tensor files are stored as compressed `.npz` files, for example:

```text
data/processed/ms_mca_tensor_delhi_ncr_2020_01.npz
```

Each file contains:

```text
stream_a:  [3, H, W]
stream_b:  [5, H, W]
label_cls: [H, W]
label_reg: [H, W]
mask:      [H, W]
```

Stream A channel order:

```text
0. no2_mean
1. so2_mean
2. co_mean
```

Stream B channel order:

```text
0. nightlight_mean
1. population_sum
2. urban_fraction
3. power_plant_count
4. fossil_capacity_mw
```

The spatial MS-MCA dataset lives in:

```text
src/data/spatial_grid_dataset.py
```

It supports either direct `.npz` paths or a tensor index CSV:

```python
from src.data.spatial_grid_dataset import SpatialGridDataset

dataset = SpatialGridDataset(
    tensor_index_csv="data/processed/tensor_index_2020_2024.csv",
    crop_size=16,
)
```

Each sample is a patch centered on a valid grid cell:

```text
stream_a:  [3, crop_size, crop_size]
stream_b:  [5, crop_size, crop_size]
label_cls: scalar class label, 0 to 4
label_reg: scalar ODIAC emission intensity
mask:      [crop_size, crop_size]
date:      string
row:       integer
col:       integer
```

Expected PyTorch DataLoader batch format:

```text
batch["stream_a"]:  [B, 3, crop_size, crop_size]
batch["stream_b"]:  [B, 5, crop_size, crop_size]
batch["label_cls"]: [B]
batch["label_reg"]: [B]
batch["mask"]:      [B, crop_size, crop_size]
```

The existing `src/dataset.py` and `src/dataloader.py` remain for the tabular MLP
baseline only.
