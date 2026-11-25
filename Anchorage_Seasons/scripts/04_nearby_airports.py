
from pathlib import Path
import geopandas as gpd

# Locate our data
project_dir = Path(__file__).resolve().parent.parent

airports_path = (
    project_dir / "data" / "qgis_sample_data"
    / "shapefiles" / "airports.shp"
)

output_path = (
    project_dir / "data" / "processed"
    / "nearby_anchorage_airports.shp"
)

# Load airport locations
airports = gpd.read_file(airports_path)

# Convert coordinates to meters (NAD83 / UTM zone 6N)
airports = airports.to_crs(epsg=26906)

# Find Anchorage International Airport
anchorage = airports.loc[
    airports["NAME"] == "ANCHORAGE INTL"
].geometry.iloc[0]

# Create a 25-kilometer search area
search_area = anchorage.buffer(25_000)

# Find airports inside the search area
nearby = airports.loc[
    airports.geometry.intersects(search_area)
].copy()

# Calculate distances from Anchorage International
nearby["distance_km"] = (
    nearby.geometry.distance(anchorage) / 1000
)

# Sort by distance
nearby = nearby.sort_values("distance_km")

print("Airports within 25 km of Anchorage International:\n")
print(nearby[["NAME", "distance_km"]].to_string(index=False))

# Save the result as a new shapefile
nearby.to_file(output_path)

print("\nSaved:", output_path)
