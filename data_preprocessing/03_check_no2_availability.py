# 03_check_no2_availability.py

import ee


# ============================================================
# 1. Settings
# ============================================================

PROJECT_ID = "carbon-detection"
NO2_COLLECTION = "COPERNICUS/S5P/OFFL/L3_NO2"


# ============================================================
# 2. Initialize Earth Engine
# ============================================================

ee.Initialize(project=PROJECT_ID)

print("Google Earth Engine initialized successfully.")
print(f"Project ID: {PROJECT_ID}")


# ============================================================
# 3. Load Sentinel-5P NO2 collection
# ============================================================

collection = ee.ImageCollection(NO2_COLLECTION)

count = collection.size().getInfo()

print()
print(f"Total number of images: {count}")


# ============================================================
# 4. Get first and latest image date
# ============================================================

first_image = collection.sort("system:time_start").first()
latest_image = collection.sort("system:time_start", False).first()

first_date = ee.Date(first_image.get("system:time_start")).format("YYYY-MM-dd HH:mm:ss").getInfo()
latest_date = ee.Date(latest_image.get("system:time_start")).format("YYYY-MM-dd HH:mm:ss").getInfo()

print()
print(f"First available date: {first_date}")
print(f"Latest available date: {latest_date}")


# ============================================================
# 5. Check number of images by year
# ============================================================

print()
print("Image count by year:")

for year in range(2018, 2027):
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"

    yearly_count = (
        collection
        .filterDate(start, end)
        .size()
        .getInfo()
    )

    print(f"{year}: {yearly_count}")