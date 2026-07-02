# CDI Carbon Detection Data Preprocessing

This folder contains the first working preprocessing pipeline for the CDI internship project:

> **Transformer / MS-MCA based carbon emission hotspot detection using multi-modal satellite and spatial data**

The goal of this preprocessing stage is to convert heterogeneous geospatial datasets into one common format for model training.

For the current test version, we use:

- Region: **Delhi NCR approximate bounding box**
- Spatial unit: **1 km × 1 km grid**
- Time step: **monthly**
- Test month: **2024-01**
- Stream A: **Sentinel-5P NO₂**
- Stream B: **VIIRS Nighttime Light**
- Label: **ODIAC fossil fuel CO₂ emission**

The current final output is:

```text
data/processed/final_ms_mca_dataset_2024_01.csv
data/processed/tensors/ms_mca_sample_2024_01.npz
```

---

## 1. Project Structure

Recommended folder structure:

```text
Carbon_Detction/
├── data/
│   ├── raw/
│   │   └── odiac/
│   ├── interim/
│   ├── processed/
│   │   └── tensors/
│   └── grid/
├── data_preprocessing/
├── outputs/
└── README.md
```

Meaning of each data folder:

| Folder | Purpose |
|---|---|
| `data/raw/` | Original downloaded files, for example ODIAC `.tif` |
| `data/grid/` | Delhi NCR 1 km grid files |
| `data/interim/` | Intermediate extracted feature/label CSV files |
| `data/processed/` | Cleaned final CSV files for model input |
| `data/processed/tensors/` | Final `.npz` tensor samples |
| `outputs/` | Visualization images for checking preprocessing results |

---

## 2. Python Environment

We used a conda environment named `carbon-detection`.

Example setup:

```bash
conda create -n carbon-detection python=3.14
conda activate carbon-detection
```

Install the main packages:

```bash
conda install -c conda-forge geopandas shapely pyproj fiona numpy pandas matplotlib rasterio rasterstats
```

Install Google Earth Engine related packages:

```bash
pip install earthengine-api geemap
```

If conda dependency solving is very slow, using `mamba` is recommended:

```bash
conda install -c conda-forge mamba
mamba install -c conda-forge geopandas rasterio rasterstats
```

---

## 3. Google Earth Engine Setup

Sentinel-5P NO₂ and VIIRS Nighttime Light are accessed through the **Google Earth Engine Data Catalog**.

Important clarification:

> GEE is not the original data producer. It is a cloud-based geospatial data access and processing platform. The original datasets are produced by agencies such as ESA/Copernicus and NOAA, while GEE provides convenient access and server-side processing.

### 3.1 Create or Register a Google Cloud Project

1. Go to the Google Earth Engine registration / project setup page.
2. Create a Google Cloud Project or register an existing project.
3. For this project, we used:

```python
PROJECT_ID = "your-google-cloud-project-id"
```

If your project ID is different, change `PROJECT_ID` in the scripts.

### 3.2 Authenticate Earth Engine

Run:

```bash
earthengine authenticate
```

If credentials already exist, the terminal may show:

```text
Authenticate: Credentials already exist. Use --force to refresh.
```

This is okay.

### 3.3 Set the Earth Engine Project

Run:

```bash
earthengine set_project YOUR_PROJECT_ID
```

Example:

```bash
earthengine set_project carbon-detection
```

### 3.4 Initialize Earth Engine in Python

In Python scripts, use:

```python
import ee

ee.Initialize(project="carbon-detection")
```

If your project ID is different, replace `carbon-detection` with your own project ID.

---

## 4. Dataset Concept

All datasets are converted into this common table format:

```text
grid_id, row, col, lon, lat, date, feature_value
```

For individual feature files, the final value column can have a specific name such as `no2_mean` or `nightlight_mean`.

For the final merged dataset, feature-specific names are necessary:

```text
grid_id, row, col, lon, lat, date,
no2_mean,
nightlight_mean,
label_reg,
label_reg_log,
label_cls
```

### Stream Definition

In this project, a stream is not necessarily one dataset. A stream is a group of features with similar physical meaning.

Current minimal version:

```text
Stream A: Atmospheric / environmental proxy
- no2_mean

Stream B: Human activity / urban proxy
- nightlight_mean

Labels:
- label_reg: ODIAC CO2 emission value
- label_reg_log: log1p(label_reg), recommended for regression
- label_cls: hotspot class, 0 to 4
```

---

## 5. Preprocessing Pipeline

Run scripts from the project root directory.

Example:

```bash
python data_preprocessing/01_create_grid.py
```

---

### Step 1. Create Delhi NCR 1 km Grid

Script:

```text
data_preprocessing/01_create_grid.py
```

Purpose:

- Create an approximate Delhi NCR bounding box.
- Convert the region to UTM Zone 43N, `EPSG:32643`.
- Generate 1000 m × 1000 m grid cells.
- Save grid files.

Outputs:

```text
data/grid/delhi_ncr_1km_grid.geojson
data/grid/delhi_ncr_1km_grid.csv
```

Important columns:

| Column | Meaning |
|---|---|
| `grid_id` | Unique grid cell ID |
| `row` | Grid row index, used to reconstruct `[H, W]` tensor |
| `col` | Grid column index, used to reconstruct `[H, W]` tensor |
| `lon` | Cell centroid longitude |
| `lat` | Cell centroid latitude |

Note:

Most internal grid cells are exactly 1 km × 1 km. Edge cells may be smaller because the grid is clipped to the bounding box.

---

### Step 2. Check Grid Visualization

Script:

```text
data_preprocessing/02_check_grid.py
```

Purpose:

- Load the generated grid.
- Check required columns.
- Plot the grid boundary.

Output:

```text
outputs/delhi_ncr_1km_grid_check.png
```

---

### Step 3. Check Sentinel-5P NO₂ Availability

Script:

```text
data_preprocessing/03_check_no2_availability.py
```

Purpose:

- Check first and latest available date for Sentinel-5P NO₂ in GEE.
- Count images by year.

GEE dataset:

```text
COPERNICUS/S5P/OFFL/L3_NO2
```

Band used:

```text
tropospheric_NO2_column_number_density
```

---

### Step 4. Extract Sentinel-5P NO₂

Script:

```text
data_preprocessing/03_extract_no2.py
```

Purpose:

- Load Sentinel-5P NO₂ from GEE.
- Create a monthly mean image.
- Extract grid-level mean values using `reduceRegions`.
- Process in batches to avoid Earth Engine payload size limits.

Important settings for 2024-01:

```python
START_DATE = "2024-01-01"
END_DATE = "2024-02-01"
DATE_LABEL = "2024-01"
```

Output:

```text
data/interim/stream_a_no2_2024_01.csv
```

Expected columns:

```text
grid_id, row, col, lon, lat, date, no2_mean
```

Why batching is used:

The full grid has 17,156 polygons. Sending all polygons to Earth Engine at once can exceed the request payload limit. Therefore, the script processes the grid in batches of 500 cells.

---

### Step 5. Clean NO₂ Missing Values

Script:

```text
data_preprocessing/05_clean_no2.py
```

Purpose:

- Fill missing `no2_mean` values.
- Current strategy: replace NaN with the global mean of `no2_mean`.

Output:

```text
data/processed/stream_a_no2_2024_01_clean.csv
```

---

### Step 6. Extract VIIRS Nighttime Light

Script:

```text
data_preprocessing/06_extract_nightlight.py
```

Purpose:

- Load VIIRS monthly nighttime light from GEE.
- Extract grid-level mean radiance values.
- Process in batches.

GEE dataset:

```text
NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG
```

Band used:

```text
avg_rad
```

Important settings for 2024-01:

```python
START_DATE = "2024-01-01"
END_DATE = "2024-02-01"
DATE_LABEL = "2024-01"
```

Output:

```text
data/interim/stream_b_nightlight_2024_01.csv
```

Expected columns:

```text
grid_id, row, col, lon, lat, date, nightlight_mean
```

---

### Step 7. Visualize Nighttime Light

Script:

```text
data_preprocessing/07_visualize_nightlight.py
```

Purpose:

- Join grid geometry with `nightlight_mean`.
- Plot Nighttime Light spatial distribution.

Output:

```text
outputs/nightlight_2024_01_map.png
```

---

### Step 8. Clean Nighttime Light Missing Values

Script:

```text
data_preprocessing/08_clean_nightlight.py
```

Purpose:

- Fill missing `nightlight_mean` values.
- Current strategy: replace NaN with the global mean of `nightlight_mean`.

Output:

```text
data/processed/stream_b_nightlight_2024_01_clean.csv
```

---

### Step 9. Prepare ODIAC CO₂ Label Data

ODIAC is not handled in the same way as NO₂ and Nighttime Light in this pipeline.

For this version, we manually downloaded the ODIAC 1 km GeoTIFF file and processed it locally.

Recommended file:

```text
odiac2025_1km_excl_intl_2401.tif
```

Place it here:

```text
data/raw/odiac/odiac2025_1km_excl_intl_2401.tif
```

We used the 1 km GeoTIFF instead of the 1-degree NetCDF because our model grid is 1 km × 1 km. A 1-degree file is too coarse for this task.

---

### Step 10. Check ODIAC GeoTIFF

Script:

```text
data_preprocessing/09_check_odiac_tif.py
```

Purpose:

- Open the ODIAC `.tif` file.
- Check CRS, bounds, resolution, data type, and Delhi NCR window values.

Expected ODIAC raster properties:

```text
CRS: EPSG:4326
Resolution: 0.008333333 degrees, about 30 arc-second / about 1 km
Bounds: global extent
```

---

### Step 11. Extract ODIAC label_reg

Script:

```text
data_preprocessing/10_extract_odiac_label.py
```

Purpose:

- Run zonal statistics over the Delhi NCR 1 km grid.
- Assign ODIAC CO₂ emission value to each grid cell.

Output:

```text
data/interim/labels_odiac_2024_01.csv
```

Expected columns:

```text
grid_id, row, col, lon, lat, date, label_reg, odiac_min, odiac_max
```

For the initial test, `label_reg` is calculated as the mean ODIAC value inside each grid cell.

Note:

The ODIAC label distribution is highly skewed. Some cells can have very large values due to strong point sources such as power plants or large industrial facilities. These should not be removed automatically because they may be true emission hotspots.

---

### Step 12. Visualize ODIAC Label

Script:

```text
data_preprocessing/11_visualize_odiac_label.py
```

Purpose:

- Plot raw ODIAC label map.
- Plot log-transformed ODIAC label map.

Outputs:

```text
outputs/odiac_label_2024_01_map.png
outputs/odiac_label_2024_01_log_map.png
```

The log map is usually more useful because `label_reg` has strong outliers.

---

### Step 13. Create Classification Label

Script:

```text
data_preprocessing/12_create_label_cls.py
```

Purpose:

- Create `label_reg_log` using `np.log1p(label_reg)`.
- Create `label_cls` from ODIAC CO₂ emission values.

Classification rule:

```text
label_reg == 0  -> class 0
label_reg > 0   -> divided into 4 quantile groups, classes 1 to 4
```

Class meaning:

| Class | Meaning |
|---:|---|
| 0 | zero / very low emission |
| 1 | low emission |
| 2 | medium emission |
| 3 | high emission |
| 4 | hotspot |

Output:

```text
data/processed/labels_odiac_2024_01_with_cls.csv
```

Expected columns:

```text
grid_id, row, col, lon, lat, date, label_reg, label_reg_log, label_cls
```

---

### Step 14. Build Final MS-MCA Dataset CSV

Script:

```text
data_preprocessing/13_build_final_dataset.py
```

Purpose:

Merge:

```text
Stream A: stream_a_no2_2024_01_clean.csv
Stream B: stream_b_nightlight_2024_01_clean.csv
Label: labels_odiac_2024_01_with_cls.csv
```

Final output:

```text
data/processed/final_ms_mca_dataset_2024_01.csv
```

Expected columns:

```text
grid_id, row, col, lon, lat, date,
no2_mean,
nightlight_mean,
label_reg,
label_reg_log,
label_cls
```

In the current successful run:

```text
Rows: 17156
Missing values: 0
```

---

### Step 15. Build Tensor Sample for PyTorch

Script:

```text
data_preprocessing/14_build_tensor_sample.py
```

Purpose:

Convert the final CSV table into model-ready NumPy tensors.

Output:

```text
data/processed/tensors/ms_mca_sample_2024_01.npz
```

Arrays inside `.npz`:

```text
stream_a:  [1, H, W]
stream_b:  [1, H, W]
label_cls: [H, W]
label_reg: [H, W]
mask:      [H, W]
```

Current mapping:

```text
stream_a[0] = no2_mean
stream_b[0] = nightlight_mean
label_cls = 0 to 4 hotspot class
label_reg = label_reg_log
mask = valid grid cell mask
```

`mask` is important because the rectangular tensor may include empty/padded locations. During model training, loss should be calculated only where `mask == 1`.

---

## 6. Final Dataset Summary

The current minimal dataset is:

```text
Date: 2024-01
Region: Delhi NCR approximate bbox
Grid cells: 17156
Stream A channels: 1
Stream B channels: 1
Classification classes: 5
Regression target: log-transformed ODIAC CO2 emission
```

Final CSV:

```text
data/processed/final_ms_mca_dataset_2024_01.csv
```

Final tensor sample:

```text
data/processed/tensors/ms_mca_sample_2024_01.npz
```

---

## 7. What Is Finished?

Finished:

```text
1. Delhi NCR 1 km grid generation
2. Sentinel-5P NO2 extraction
3. VIIRS Nighttime Light extraction
4. ODIAC CO2 label extraction
5. label_cls creation
6. Final CSV creation
7. Final tensor sample creation
```

The preprocessing stage for a **minimal one-month model test sample** is complete.

---

## 8. What Comes Next?

Next model-side steps:

```text
1. Create PyTorch Dataset class for the .npz file
2. Create DataLoader
3. Build baseline multi-stream model
4. Define classification and regression losses
5. Use mask during loss calculation
6. Run a simple training loop
```

Recommended first model input shapes:

```text
stream_a:  [B, 1, H, W]
stream_b:  [B, 1, H, W]
label_cls: [B, H, W]
label_reg: [B, H, W]
mask:      [B, H, W]
```

---

## 9. Important Notes for Team Members

### 9.1 Do not commit large raw files if possible

Large files such as `.tif`, `.nc`, `.zip`, `.gz`, and `.npz` may be too large for GitHub.

Recommended `.gitignore` entries:

```text
data/raw/
data/interim/
data/processed/tensors/
*.tif
*.tif.gz
*.nc
*.zip
*.npz
```

If data files are needed for sharing, use Google Drive, institutional storage, or Git LFS.

### 9.2 Keep scripts and data separate

Scripts should be committed:

```text
data_preprocessing/*.py
README.md
```

Large generated data may be excluded from GitHub.

### 9.3 Project ID must be changed by each user

Each teammate may have a different Earth Engine project ID.

Check and update this line in GEE scripts:

```python
PROJECT_ID = "carbon-detection"
```

### 9.4 Current Delhi NCR boundary is approximate

This version uses an approximate bounding box:

```text
Longitude: 76.5 to 77.8
Latitude: 28.1 to 29.3
```

Later, this should be replaced with an official Delhi NCR boundary polygon if higher geographic accuracy is required.

### 9.5 Current dataset is only one month

This is a minimal test sample for checking the full pipeline.

For real training, the team should repeat the same process for multiple months, for example:

```text
2023-01 to 2024-12
```

Then each month can become one training sample.

---

## 10. Short Explanation for the Team

This preprocessing pipeline creates a common 1 km grid over Delhi NCR and converts multiple geospatial datasets into the same spatial and temporal format. Sentinel-5P NO₂ is used as Stream A, VIIRS Nighttime Light is used as Stream B, and ODIAC CO₂ emissions are used to create regression and classification labels. The final output is a model-ready `.npz` file containing stream tensors, labels, and a valid-cell mask.
