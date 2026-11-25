import math
import shutil
from contextlib import ExitStack
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, bounds, from_bounds, transform
from shapely.geometry import box, mapping, shape

# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
BOUNDARY_FILE = ROOT / "data/processed/anchorage_municipality.geojson"
OUTPUT = ROOT / "data/processed/anchorage_s2_mosaic_10m.tif"
TEMP_OUTPUT = OUTPUT.with_suffix(".partial.tif")

TARGET_CRS = "EPSG:3338"
RESOLUTION = 10
BLOCK = 512

SCENE_IDS = [
    "S2C_T05VPH_20250612T212656_L2A",
    "S2C_T06VUN_20250612T212656_L2A",
    "S2C_T06VVN_20250612T212656_L2A",
    "S2C_T05VPJ_20250612T212656_L2A",
    "S2C_T06VUP_20250612T212656_L2A",
    "S2C_T06VVP_20250612T212656_L2A",
]

BANDS = ["blue", "green", "red", "nir"]

# --------------------------------------------------
# 1. Municipality boundary and output grid
# --------------------------------------------------

municipality = (
    gpd.read_file(BOUNDARY_FILE)
    .to_crs(TARGET_CRS)
    .geometry.union_all()
)

municipality_geometry = mapping(municipality)

minx, miny, maxx, maxy = municipality.bounds

left = math.floor(minx / RESOLUTION) * RESOLUTION
bottom = math.floor(miny / RESOLUTION) * RESOLUTION
right = math.ceil(maxx / RESOLUTION) * RESOLUTION
top = math.ceil(maxy / RESOLUTION) * RESOLUTION

width = round((right - left) / RESOLUTION)
height = round((top - bottom) / RESOLUTION)

grid_transform = from_origin(
    left, top, RESOLUTION, RESOLUTION
)

size_bytes = width * height * 4 * 2
size_gb = size_bytes / 1e9

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

free_bytes = shutil.disk_usage(OUTPUT.parent).free

print("Output dimensions:", width, "x", height)
print("Target CRS:", TARGET_CRS)
print("Resolution:", RESOLUTION, "meters")
print(f"Uncompressed pixel storage: {size_gb:.2f} GB")
print(f"Available disk space: {free_bytes / 1e9:.2f} GB")

# Conservative disk-space safeguard
if free_bytes < size_bytes * 2 + 2_000_000_000:
    raise RuntimeError(
        "Insufficient free disk space for safe mosaic creation."
    )

# --------------------------------------------------
# 2. Retrieve the six scenes
# --------------------------------------------------

catalog = Client.open(
    "https://earth-search.aws.element84.com/v1"
)

collection = catalog.get_collection("sentinel-2-c1-l2a")

scenes = []

for scene_id in SCENE_IDS:
    scene = collection.get_item(scene_id)

    if scene is None:
        raise ValueError(f"Missing scene: {scene_id}")

    footprint = gpd.GeoSeries(
        [shape(scene.geometry)],
        crs="EPSG:4326",
    ).to_crs(TARGET_CRS).iloc[0]

    relevant_area = footprint.intersection(municipality)

    if not relevant_area.is_empty:
        scenes.append((
            scene.properties.get("eo:cloud_cover", 100),
            scene,
            relevant_area,
        ))

# Prefer lower-cloud scenes in overlapping regions
scenes.sort(key=lambda record: record[0])

print("Scenes to mosaic:", len(scenes))

if len(scenes) != len(SCENE_IDS):
    raise RuntimeError("Unexpected number of relevant scenes.")

# --------------------------------------------------
# 3. Create the four-band GeoTIFF
# --------------------------------------------------

profile = {
    "driver": "GTiff",
    "width": width,
    "height": height,
    "count": 4,
    "dtype": "uint16",
    "crs": TARGET_CRS,
    "transform": grid_transform,
    "nodata": 0,
    "tiled": True,
    "blockxsize": BLOCK,
    "blockysize": BLOCK,
    "compress": "deflate",
    "predictor": 2,
    "BIGTIFF": "YES",
}

# Create the file first in write mode
with rasterio.open(TEMP_OUTPUT, "w", **profile) as dst:

    dst.scales = (0.0001,) * 4
    dst.offsets = (-0.1,) * 4

    for band_number, band_name in enumerate(BANDS, 1):
        dst.set_band_description(band_number, band_name)

# --------------------------------------------------
# 4. Reopen in read/write mode and mosaic
# --------------------------------------------------

with rasterio.Env(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    GDAL_HTTP_MAX_RETRY="3",
    GDAL_HTTP_RETRY_DELAY="1",
):
    with rasterio.open(TEMP_OUTPUT, "r+") as dst:

        for scene_number, (cloud, scene, area) in enumerate(
            scenes, 1
        ):
            print(
                f"\nScene {scene_number}/{len(scenes)}: "
                f"{scene.id}"
            )
            print(f"Cloud cover: {cloud:.2f}%")

            coverage_window = from_bounds(
                *area.bounds,
                transform=grid_transform,
            )

            col_start = max(
                0,
                math.floor(coverage_window.col_off / BLOCK) * BLOCK,
            )

            row_start = max(
                0,
                math.floor(coverage_window.row_off / BLOCK) * BLOCK,
            )

            col_end = min(
                width,
                math.ceil(
                    (coverage_window.col_off + coverage_window.width)
                    / BLOCK
                ) * BLOCK,
            )

            row_end = min(
                height,
                math.ceil(
                    (coverage_window.row_off + coverage_window.height)
                    / BLOCK
                ) * BLOCK,
            )

            written_pixels = 0

            # Open all four bands from the same scene
            with ExitStack() as stack:

                vrts = []

                for band_name in BANDS:

                    url = scene.assets[band_name].href

                    src = stack.enter_context(
                        rasterio.open(url)
                    )

                    if not (
                        np.isclose(src.scales[0], 0.0001)
                        and np.isclose(src.offsets[0], -0.1)
                    ):
                        raise ValueError(
                            f"Unexpected reflectance metadata: "
                            f"{scene.id}, {band_name}"
                        )

                    vrt = stack.enter_context(
                        WarpedVRT(
                            src,
                            crs=TARGET_CRS,
                            transform=grid_transform,
                            width=width,
                            height=height,
                            src_nodata=0,
                            nodata=0,
                            resampling=Resampling.bilinear,
                        )
                    )

                    vrts.append(vrt)

                # Process the scene block by block
                for row in range(row_start, row_end, BLOCK):
                    for col in range(col_start, col_end, BLOCK):

                        window = Window(
                            col,
                            row,
                            min(BLOCK, width - col),
                            min(BLOCK, height - row),
                        )

                        window_polygon = box(
                            *bounds(window, grid_transform)
                        )

                        if not area.intersects(window_polygon):
                            continue

                        current = dst.read(window=window)

                        inside = geometry_mask(
                            [municipality_geometry],
                            out_shape=current.shape[1:],
                            transform=transform(
                                window, grid_transform
                            ),
                            invert=True,
                        )

                        # Do not overwrite previously filled pixels
                        available = (
                            inside
                            & np.all(current == 0, axis=0)
                        )

                        if not np.any(available):
                            continue

                        # Read all four spectral bands
                        pixels = np.ma.stack([
                            vrt.read(
                                1,
                                window=window,
                                masked=True,
                            )
                            for vrt in vrts
                        ])

                        # Require valid data in every band
                        valid = (
                            available
                            & ~np.any(
                                np.ma.getmaskarray(pixels),
                                axis=0,
                            )
                            & np.all(pixels.data != 0, axis=0)
                        )

                        if np.any(valid):
                            current[:, valid] = pixels.data[:, valid]

                            dst.write(
                                current,
                                window=window,
                            )

                            written_pixels += np.count_nonzero(valid)

            print(f"New pixels filled: {written_pixels:,}")

# --------------------------------------------------
# 5. Verify the completed mosaic
# --------------------------------------------------

inside_count = 0
valid_count = 0

with rasterio.open(TEMP_OUTPUT) as src:

    print("\nMOSAIC VERIFICATION")
    print("CRS:", src.crs)
    print("Dimensions:", src.width, "x", src.height)
    print("Band count:", src.count)
    print("Band descriptions:", src.descriptions)
    print("Pixel size:", src.res)
    print("Data types:", src.dtypes)
    print("NoData:", src.nodata)
    print("Scales:", src.scales)
    print("Offsets:", src.offsets)

    for _, window in src.block_windows(1):

        pixels = src.read(window=window)

        inside = geometry_mask(
            [municipality_geometry],
            out_shape=pixels.shape[1:],
            transform=src.window_transform(window),
            invert=True,
        )

        valid = np.all(pixels != 0, axis=0)

        inside_count += np.count_nonzero(inside)
        valid_count += np.count_nonzero(inside & valid)

    if inside_count == 0 or valid_count == 0:
        raise RuntimeError("No valid municipality imagery found.")

    coverage = 100 * valid_count / inside_count

    print(f"Municipality pixels: {inside_count:,}")
    print(f"Valid four-band pixels: {valid_count:,}")
    print(f"Valid pixel coverage: {coverage:.2f}%")

# Publish the completed file only after verification
TEMP_OUTPUT.replace(OUTPUT)

print(f"File size: {OUTPUT.stat().st_size / 1e6:.2f} MB")
print("\nSUCCESS: Municipality mosaic saved!")
print("Output:", OUTPUT)