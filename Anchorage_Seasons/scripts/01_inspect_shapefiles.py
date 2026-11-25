from pathlib import Path
import geopandas as gpd

# Locate our project's original shapefiles
project_dir = Path(__file__).resolve().parent.parent
shapefile = (
    project_dir
    / "data"
    / "qgis_sample_data"
    / "shapefiles"
    / "alaska.shp"
)

# Read the Alaska shapefile
alaska = gpd.read_file(shapefile)

# Inspect the dataset
print("Number of features:", len(alaska))
print("Coordinate reference system:", alaska.crs)
print("Columns:", alaska.columns.tolist())
print("\nFirst five records:")
print(alaska.head())
