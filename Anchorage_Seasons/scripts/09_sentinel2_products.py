
from pathlib import Path

import numpy as np
import rasterio

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

root = Path(__file__).resolve().parent.parent
source_dir = root / "data" / "sentinel2"
output_dir = root / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

bands = {}

# --------------------------------------------------
# 2. Read the four Sentinel-2 bands
# --------------------------------------------------

for band_id in ["B02", "B03", "B04", "B08"]:

    path = source_dir / f"{band_id}_anchorage.tif"

    with rasterio.open(path) as src:

        if not bands:
            profile = src.profile.copy()
            reference = (src.crs, src.transform, src.shape)
        elif (src.crs, src.transform, src.shape) != reference:
            raise ValueError(f"{band_id} is not spatially aligned")

        pixels = src.read(1, masked=True)

        # Convert stored values to surface reflectance.
        # Scale/offset were copied from the original COGs.
        scale = src.scales[0]
        offset = src.offsets[0]

        if scale == 1.0 and offset == 0.0:
            raise ValueError(
                f"{band_id}: reflectance scale/offset missing"
            )

        bands[band_id] = (
            pixels.astype("float32").filled(np.nan)
            * scale + offset
        )

        print(f"Loaded {band_id}: scale={scale}, offset={offset}")

# --------------------------------------------------
# 3. Create display-ready RGB composites
# --------------------------------------------------

def make_rgb(band_order):
    channels = []

    for band_id in band_order:
        array = bands[band_id]
        valid = np.isfinite(array)

        # Contrast stretch for visualization
        low, high = np.percentile(
            array[valid], [2, 98]
        )

        stretched = np.clip(
            (array - low) / max(high - low, 1e-6),
            0, 1
        )

        channel = np.where(
            valid, stretched * 254 + 1, 0
        ).astype("uint8")

        channels.append(channel)

    return np.stack(channels)


def save_rgb(filename, band_order):
    rgb = make_rgb(band_order)

    rgb_profile = profile.copy()
    rgb_profile.update(
        driver="GTiff",
        count=3,
        dtype="uint8",
        nodata=0,
        compress="deflate",
    )

    path = output_dir / filename

    with rasterio.open(path, "w", **rgb_profile) as dst:
        dst.write(rgb)

    print("Saved:", filename)


# Natural colors: Red, Green, Blue
save_rgb(
    "anchorage_true_color.tif",
    ["B04", "B03", "B02"]
)

# False color: NIR, Red, Green
save_rgb(
    "anchorage_false_color.tif",
    ["B08", "B04", "B03"]
)

# --------------------------------------------------
# 4. Calculate NDVI
# --------------------------------------------------

red = bands["B04"]
nir = bands["B08"]

denominator = nir + red

valid = (
    np.isfinite(red)
    & np.isfinite(nir)
    & (red >= 0)
    & (nir >= 0)
    & (denominator > 0)
)

ndvi = np.full(red.shape, -9999.0, dtype="float32")

ndvi[valid] = (
    (nir[valid] - red[valid]) / denominator[valid]
)

# Save NDVI with geographic metadata
ndvi_profile = profile.copy()
ndvi_profile.update(
    driver="GTiff",
    count=1,
    dtype="float32",
    nodata=-9999.0,
    compress="deflate",
)

with rasterio.open(
    output_dir / "anchorage_ndvi.tif",
    "w",
    **ndvi_profile
) as dst:
    dst.write(ndvi, 1)

print("Saved: anchorage_ndvi.tif")
print("NDVI range:",
      round(float(ndvi[valid].min()), 3),
      round(float(ndvi[valid].max()), 3))

print("\nSUCCESS: All three satellite products created!")
