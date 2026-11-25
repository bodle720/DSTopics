
from pathlib import Path
import geopandas as gpd

# Project paths
project_dir = Path(__file__).resolve().parent.parent
airports_path = (
    project_dir
    / "data"
    / "qgis_sample_data"
    / "shapefiles"
    / "airports.shp"
)

# Load the airport shapefile
airports = gpd.read_file(airports_path)

# Basic inspection
print("Number of airports:", len(airports))
print("Columns:", airports.columns.tolist())
print("Geometry types:")
print(airports.geom_type.value_counts())

# Inspect airport attributes without printing lengthy geometries
print("\nFirst 10 airport records:")
print(
    airports.drop(columns="geometry")
    .head(10)
    .to_string(index=False)
)

# Look at one actual Shapely geometry
print("\nFirst airport geometry:")
print(airports.geometry.iloc[0])
