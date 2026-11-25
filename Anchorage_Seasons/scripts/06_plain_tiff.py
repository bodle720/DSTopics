
from pathlib import Path
import rasterio

project_dir = Path(__file__).resolve().parent.parent

source_path = (
    project_dir / "data" / "qgis_sample_data"
    / "raster" / "SR_50M_alaska_nad.tif"
)

output_path = (
    project_dir / "data" / "processed"
    / "alaska_without_georeferencing.tif"
)

# Read the original GeoTIFF
with rasterio.open(source_path) as src:
    pixels = src.read(1)

    # Create a new TIFF with pixel data only.
    # No CRS or affine transform is supplied.
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=src.height,
        width=src.width,
        count=1,
        dtype=src.dtypes[0],
    ) as dst:
        dst.write(pixels, 1)

print("Ordinary TIFF created:")
print(output_path)

# Check what geographic information remains
with rasterio.open(output_path) as src:
    print("\nCRS:", src.crs)
    print("Transform:", src.transform)
    print("Dimensions:", src.width, "x", src.height)
    print("Bands:", src.count)
