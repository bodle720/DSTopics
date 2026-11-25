
from pathlib import Path

import geopandas as gpd
from pystac_client import Client
from shapely.geometry import shape
from shapely.ops import unary_union

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

root = Path(__file__).resolve().parent.parent

boundary_path = (
    root / "data" / "processed"
    / "anchorage_municipality.geojson"
)

output_path = (
    root / "data" / "processed"
    / "sentinel2_tile_footprints.geojson"
)

# --------------------------------------------------
# 2. Load Anchorage's actual municipal boundary
# --------------------------------------------------

anchorage = gpd.read_file(boundary_path).to_crs(4326)

boundary = anchorage.geometry.union_all()

print("Loaded Anchorage municipality boundary.")

# --------------------------------------------------
# 3. Search Sentinel-2 tiles
# --------------------------------------------------

catalog = Client.open(
    "https://earth-search.aws.element84.com/v1"
)

search = catalog.search(
    collections=["sentinel-2-c1-l2a"],
    bbox=boundary.bounds,
    datetime="2025-06-12",
    max_items=200,
)

scenes = list(search.items())

# Select tiles from our original acquisition
selected = [
    scene for scene in scenes
    if scene.id.startswith("S2C_")
    and "_20250612T212656_" in scene.id
    and shape(scene.geometry).intersects(boundary)
]

if not selected:
    raise ValueError("No matching tiles found.")

print(f"\nMatching tiles: {len(selected)}")

# --------------------------------------------------
# 4. Inspect tile footprints
# --------------------------------------------------

records = []

for scene in selected:

    footprint = shape(scene.geometry)

    print(
        scene.id,
        "| Cloud cover:",
        scene.properties.get("eo:cloud_cover"),
    )

    records.append({
        "scene_id": scene.id,
        "cloud_pct": scene.properties.get("eo:cloud_cover"),
        "geometry": footprint,
    })

# --------------------------------------------------
# 5. Calculate combined geographic coverage
# --------------------------------------------------

footprints = gpd.GeoDataFrame(
    records,
    geometry="geometry",
    crs="EPSG:4326",
)

# Project to Alaska Albers in meters for area calculations
boundary_m = gpd.GeoSeries(
    [boundary], crs="EPSG:4326"
).to_crs(3338).iloc[0]

tiles_m = footprints.to_crs(3338)

combined = unary_union(tiles_m.geometry)

covered_area = combined.intersection(boundary_m).area
total_area = boundary_m.area

coverage = 100 * covered_area / total_area

print(f"\nMunicipality covered: {coverage:.2f}%")

# --------------------------------------------------
# 6. Export footprints for QGIS
# --------------------------------------------------

footprints.to_file(output_path, driver="GeoJSON")

print("\nSaved tile footprints:", output_path)

if coverage < 99.9:
    print("Some areas are not covered by this acquisition.")
    print("We may need additional imagery.")
else:
    print("This acquisition covers the municipality!")

print("\nNo satellite imagery downloaded yet.")
