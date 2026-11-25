import argparse
from pathlib import Path
from time import perf_counter

import boto3
import numpy as np
import rasterio
from rasterio.session import AWSSession
from rasterio.warp import transform
from rasterio.windows import Window

# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser(
    description="Read a 200 x 200 pixel downtown Anchorage window from an S3 COG."
)
parser.add_argument("--source", required=True, help="S3 URI of an Anchorage COG you can access")
parser.add_argument("--profile", help="Optional local AWS profile; otherwise use the normal credential chain")
parser.add_argument("--region", help="Optional AWS region; otherwise use the normal AWS configuration")
args = parser.parse_args()
if not args.source.startswith("s3://"):
    parser.error("--source must be an s3:// URI")

SOURCE = args.source
OUTPUT = ROOT / "data/processed/anchorage_downtown_clip.tif"

LONGITUDE = -149.9003
LATITUDE = 61.2181

WINDOW_SIZE = 200  # 200 pixels = 2 kilometers

# --------------------------------------------------
# 1. Authenticate with AWS
# --------------------------------------------------

session = boto3.Session(
    profile_name=args.profile,
    region_name=args.region,
)

aws_session = AWSSession(session)

print("Opening remote COG...")
start = perf_counter()

# --------------------------------------------------
# 2. Open the remote COG
# --------------------------------------------------

with rasterio.Env(
    session=aws_session,
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    GDAL_HTTP_MAX_RETRY="5",
    GDAL_HTTP_RETRY_DELAY="2",
):
    with rasterio.open(SOURCE) as src:

        print("Remote dimensions:", src.width, "x", src.height)
        print("CRS:", src.crs)
        print("Resolution:", src.res)
        print("Layout:", src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT"))

        # Convert longitude/latitude into the raster CRS
        xs, ys = transform(
            "EPSG:4326",
            src.crs,
            [LONGITUDE],
            [LATITUDE],
        )

        row, col = src.index(xs[0], ys[0])

        # Define a small window around downtown Anchorage
        window = Window(
            col - WINDOW_SIZE // 2,
            row - WINDOW_SIZE // 2,
            WINDOW_SIZE,
            WINDOW_SIZE,
        )

        if (
            window.col_off < 0
            or window.row_off < 0
            or window.col_off + window.width > src.width
            or window.row_off + window.height > src.height
        ):
            raise ValueError("Requested window is outside the raster.")

        # --------------------------------------------------
        # 3. Retrieve only the requested window
        # --------------------------------------------------

        print("\nRetrieving geographic subset...")

        pixels = src.read(window=window)

        profile = {
            "driver": "GTiff",
            "width": WINDOW_SIZE,
            "height": WINDOW_SIZE,
            "count": src.count,
            "dtype": src.dtypes[0],
            "crs": src.crs,
            "transform": src.window_transform(window),
            "nodata": src.nodata,
            "compress": "deflate",
        }

        descriptions = src.descriptions
        scales = src.scales
        offsets = src.offsets

elapsed = perf_counter() - start

# --------------------------------------------------
# 4. Save the extracted imagery locally
# --------------------------------------------------

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with rasterio.open(OUTPUT, "w", **profile) as dst:

    dst.write(pixels)
    dst.descriptions = descriptions
    dst.scales = scales
    dst.offsets = offsets

valid = np.all(pixels != 0, axis=0)

print("\nREMOTE EXTRACTION RESULTS")
print("Array shape:", pixels.shape)
print("Valid pixel coverage:", valid.mean() * 100, "%")
print(f"Open and read time: {elapsed:.2f} seconds")
print(f"Uncompressed array: {pixels.nbytes / 1e6:.2f} MB")
print(f"Saved file size: {OUTPUT.stat().st_size / 1e6:.2f} MB")

print("\nSUCCESS: Remote COG subset retrieved!")
print("Output:", OUTPUT)