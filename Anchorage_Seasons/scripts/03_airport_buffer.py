
from pathlib import Path
import geopandas as gpd

# Project paths
project_dir = Path(__file__).resolve().parent.parent

airports_path = (
    project_dir / "data" / "qgis_sample_data"
    / "shapefiles" / "airports.shp"
)

output_path = (
    project_dir / "data" / "processed"
    / "anchorage_airport_2km_buffer.shp"
)

# Load airport points
airports = gpd.read_file(airports_path)

# Select Anchorage International Airport
airport = airports.loc[
    airports["NAME"] == "ANCHORAGE INTL"
].copy()

if airport.empty:
    raise ValueError("Anchorage airport not found!")

print("Selected airport:", airport["NAME"].iloc[0])
print("Original CRS:", airport.crs.to_epsg())

# Reproject into a CRS that uses meters:
# NAD83 / UTM zone 6N (appropriate for Anchorage)
airport_m = airport.to_crs(epsg=26906)

# Shapely: turn the airport point into a 2 km buffer polygon
buffer_polygon = airport_m.copy()
buffer_polygon["geometry"] = airport_m.geometry.buffer(2000)

print("New geometry:", buffer_polygon.geom_type.iloc[0])
print("Buffer area (sq km):",
      round(buffer_polygon.geometry.area.iloc[0] / 1_000_000, 2))

# Save the new polygon as a shapefile
output_path.parent.mkdir(parents=True, exist_ok=True)
buffer_polygon.to_file(output_path)

print("Saved:", output_path)
