
from pathlib import Path

import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_geom
from pystac_client import Client
from shapely.geometry import shape, box, mapping

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

SCENE_ID = "S2C_T06VUN_20250612T212656_L2A"

# Smaller Anchorage area (WGS84 longitude/latitude)
AOI = (-150.05, 61.13, -149.75, 61.25)

BANDS = {
    "blue": "B02",
    "green": "B03",
    "red": "B04",
    "nir": "B08",
}

project_dir = Path(__file__).resolve().parent.parent
output_dir = project_dir / "data" / "sentinel2"
output_dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Retrieve the Sentinel-2 scene
# --------------------------------------------------

catalog = Client.open(
    "https://earth-search.aws.element84.com/v1"
)

collection = catalog.get_collection("sentinel-2-c1-l2a")
scene = collection.get_item(SCENE_ID)

if scene is None:
    raise ValueError("Sentinel-2 scene not found.")

print("Selected scene:", scene.id)
print("Cloud cover:", scene.properties.get("eo:cloud_cover"))

# --------------------------------------------------
# 3. Check geographic coverage
# --------------------------------------------------

scene_footprint = shape(scene.geometry)
requested_area = box(*AOI)

# Use only the part covered by this satellite tile
clip_area = requested_area.intersection(scene_footprint)

if clip_area.is_empty:
    raise ValueError("Selected scene does not overlap Anchorage AOI.")

coverage = 100 * clip_area.area / requested_area.area

print(f"AOI coverage: {coverage:.1f}%")
print("Clipping to available scene coverage.")

# --------------------------------------------------
# 4. Download four clipped GeoTIFF bands
# --------------------------------------------------

for band_name, band_id in BANDS.items():

    url = scene.assets[band_name].href
    output_path = output_dir / f"{band_id}_anchorage.tif"

    print(f"\nProcessing {band_id} ({band_name})...")

    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"
    ):
        with rasterio.open(url) as src:

            # Transform our WGS84 polygon into the raster CRS
            clip_geometry = transform_geom(
                "EPSG:4326",
                src.crs,
                mapping(clip_area),
            )

            # Read only the geographic area we need
            pixels, transform = mask(
                src,
                [clip_geometry],
                crop=True,
                nodata=src.nodata if src.nodata is not None else 0,
            )

            # Preserve relevant raster metadata
            profile = src.profile.copy()

            profile.update(
                driver="GTiff",
                height=pixels.shape[1],
                width=pixels.shape[2],
                count=1,
                transform=transform,
                nodata=src.nodata if src.nodata is not None else 0,
                compress="deflate",
            )

            # Save our local GeoTIFF
            with rasterio.open(
                output_path, "w", **profile
            ) as dst:

                dst.write(pixels)
                dst.scales = src.scales
                dst.offsets = src.offsets

            print("Saved:", output_path.name)
            print("Dimensions:", pixels.shape[2], "x", pixels.shape[1])
            print("CRS:", src.crs)
            print("Resolution:", src.res)

# --------------------------------------------------
# 5. Finished
# --------------------------------------------------

print("\nSUCCESS: Four Sentinel-2 GeoTIFF bands saved!")
print("Output directory:", output_dir)
