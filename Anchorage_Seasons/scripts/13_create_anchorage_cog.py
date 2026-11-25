from pathlib import Path

import numpy as np
import rasterio
from rasterio.shutil import copy

ROOT = Path(__file__).resolve().parent.parent

SOURCE = ROOT / "data/processed/anchorage_s2_mosaic_10m.tif"
OUTPUT = ROOT / "data/processed/anchorage_s2_cog_10m.tif"

# --------------------------------------------------
# 1. Convert GeoTIFF to Cloud Optimized GeoTIFF
# --------------------------------------------------

print("Creating Cloud Optimized GeoTIFF...")

copy(
    SOURCE,
    OUTPUT,
    driver="COG",
    COMPRESS="DEFLATE",
    PREDICTOR="YES",
    BLOCKSIZE=512,
    OVERVIEW_RESAMPLING="AVERAGE",
    BIGTIFF="IF_SAFER",
)

# --------------------------------------------------
# 2. Verify metadata and pixel values
# --------------------------------------------------

with rasterio.open(SOURCE) as original:
    with rasterio.open(OUTPUT) as cog:

        print("\nCOG VERIFICATION")
        print("CRS:", cog.crs)
        print("Resolution:", cog.res)
        print("Dimensions:", cog.width, "x", cog.height)
        print("Bands:", cog.descriptions)
        print("Data types:", cog.dtypes)
        print("NoData:", cog.nodata)
        print("Scales:", cog.scales)
        print("Offsets:", cog.offsets)
        print("Block shapes:", cog.block_shapes)
        print("Overviews:", cog.overviews(1))

        layout = cog.tags(ns="IMAGE_STRUCTURE").get("LAYOUT")
        print("Layout:", layout)

        assert layout == "COG"
        assert cog.crs == original.crs
        assert cog.transform == original.transform
        assert cog.shape == original.shape
        assert cog.count == original.count
        assert cog.dtypes == original.dtypes
        assert cog.descriptions == original.descriptions
        assert cog.nodata == original.nodata
        assert cog.scales == original.scales
        assert cog.offsets == original.offsets

        print("\nComparing original and COG pixels...")

        for _, window in original.block_windows(1):
            original_pixels = original.read(window=window)
            cog_pixels = cog.read(window=window)

            if not np.array_equal(original_pixels, cog_pixels):
                raise RuntimeError("Pixel mismatch detected.")

        print("All full-resolution pixels match!")

print(f"\nOriginal size: {SOURCE.stat().st_size / 1e6:.2f} MB")
print(f"COG size: {OUTPUT.stat().st_size / 1e6:.2f} MB")

print("\nSUCCESS: Cloud Optimized GeoTIFF created!")
print("Output:", OUTPUT)