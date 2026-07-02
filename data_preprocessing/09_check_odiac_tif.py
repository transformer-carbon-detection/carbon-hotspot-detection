# 09_check_odiac_tif.py

import rasterio
from rasterio.windows import from_bounds


# ============================================================
# 1. Path
# ============================================================

ODIAC_TIF_PATH = "data/raw/odiac/odiac2025_1km_excl_intl_2401.tif"


# ============================================================
# 2. Delhi NCR approximate bbox
# ============================================================

MIN_LON = 76.5
MAX_LON = 77.8
MIN_LAT = 28.1
MAX_LAT = 29.3


# ============================================================
# 3. Open ODIAC GeoTIFF
# ============================================================

with rasterio.open(ODIAC_TIF_PATH) as src:
    print("ODIAC GeoTIFF loaded successfully.")
    print()
    print("File path:", ODIAC_TIF_PATH)
    print("CRS:", src.crs)
    print("Width:", src.width)
    print("Height:", src.height)
    print("Number of bands:", src.count)
    print("Bounds:", src.bounds)
    print("Resolution:", src.res)
    print("Nodata value:", src.nodata)
    print("Data type:", src.dtypes)
    print()

    # ========================================================
    # 4. Read small Delhi NCR window
    # ========================================================

    window = from_bounds(
        MIN_LON,
        MIN_LAT,
        MAX_LON,
        MAX_LAT,
        transform=src.transform
    )

    arr = src.read(1, window=window)

    print("Delhi NCR window shape:", arr.shape)
    print("Raw min:", arr.min())
    print("Raw max:", arr.max())
    print("Raw mean:", arr.mean())