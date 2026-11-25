
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import geopandas as gpd

root = Path(__file__).resolve().parent.parent

output = (
    root / "data" / "processed"
    / "anchorage_municipality.geojson"
)

# Census Bureau boundary service
endpoint = (
    "https://tigerweb.geo.census.gov/arcgis/rest/services/"
    "TIGERweb/State_County/MapServer/1/query"
)

params = {
    "where": "GEOID='02020'",
    "outFields": "NAME,GEOID",
    "returnGeometry": "true",
    "outSR": 4326,
    "f": "geojson",
}

# Download the municipal boundary
url = endpoint + "?" + urlencode(params)

with urlopen(url, timeout=60) as response:
    data = json.load(response)

if len(data.get("features", [])) != 1:
    raise ValueError("Expected one Anchorage boundary.")

output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(data), encoding="utf-8")

# Inspect with GeoPandas
anchorage = gpd.read_file(output)

print("Saved:", output)
print("CRS:", anchorage.crs)
print("Geometry:", anchorage.geom_type.iloc[0])
print("Bounds:", anchorage.total_bounds)
