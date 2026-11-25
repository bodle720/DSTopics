
from pathlib import Path
import rasterio
import numpy as np

# --------------------------------------------------
# 1. Locate our GeoTIFF
# --------------------------------------------------

project_dir = Path(__file__).resolve().parent.parent

raster_path = (
    project_dir
    / "data"
    / "qgis_sample_data"
    / "raster"
    / "SR_50M_alaska_nad.tif"
)

# --------------------------------------------------
# 2. Open the raster using Rasterio
# --------------------------------------------------

with rasterio.open(raster_path) as src:

    print("FORMAT:", src.driver)
    print("CRS:", src.crs)
    print("WIDTH:", src.width)
    print("HEIGHT:", src.height)
    print("BANDS:", src.count)
    print("DATA TYPE:", src.dtypes)
    print("PIXEL SIZE:", src.res)
    print("NODATA:", src.nodata)

    # Geographic extent
    print("\nGEOGRAPHIC BOUNDS:")
    print(src.bounds)

    # Pixel-to-map coordinate transformation
    print("\nAFFINE TRANSFORM:")
    print(src.transform)

    # --------------------------------------------------
    # 3. Read band 1 into a NumPy array
    # --------------------------------------------------

    band = src.read(1, masked=True)

    print("\nNUMPY ARRAY:")
    print("Shape:", band.shape)
    print("Data type:", band.dtype)
    print("Minimum valid value:", band.min())
    print("Maximum valid value:", band.max())
    print("Valid pixels:", np.ma.count(band))

    # --------------------------------------------------
    # 4. Find the geographic location of a pixel
    # --------------------------------------------------

    row = src.height // 2
    col = src.width // 2

    x, y = src.xy(row, col)

    print("\nCENTER PIXEL:")
    print("Row:", row)
    print("Column:", col)
    print("Pixel value:", band[row, col])
    print("Map X:", x)
    print("Map Y:", y)
