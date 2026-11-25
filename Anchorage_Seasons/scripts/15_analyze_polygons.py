from itertools import combinations
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union


DATA = Path(__file__).resolve().parents[1] / "data" / "processed"
AREA_CRS = "EPSG:3338"       # Alaska Albers equal area (meters)
DISTANCE_CRS = "EPSG:32606"  # Local UTM Zone 6N (meters)
OUTPUT_CRS = "EPSG:4326"     # GeoJSON longitude/latitude


def check_polygons(gdf, label):
    if gdf.empty or gdf.crs is None:
        raise ValueError(f"{label}: no features or missing CRS.")
    for index, geom in gdf.geometry.items():
        if not isinstance(geom, (Polygon, MultiPolygon)) or geom.is_empty:
            raise ValueError(f"{label}: feature {index} is not a nonempty polygon.")
        if not geom.is_valid:
            raise ValueError(f"{label}: feature {index} has invalid geometry.")


def rectangle_dimensions(rectangle):
    # Measure the four edges of the minimum rotated rectangle.
    coordinates = list(rectangle.exterior.coords)
    edges = [
        ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        for (x1, y1), (x2, y2) in zip(coordinates, coordinates[1:])
    ]
    return max(edges) / 1000, min(edges) / 1000


def main():
    drawn = gpd.read_file(DATA / "anchorage_study_areas.geojson")
    municipality = gpd.read_file(DATA / "anchorage_municipality.geojson")
    check_polygons(drawn, "Study areas")
    check_polygons(municipality, "Municipality")
    if not {"name", "category"}.issubset(drawn.columns):
        raise ValueError("Study areas must have name and category fields.")
    if drawn[["name", "category"]].isna().any().any():
        raise ValueError("Study areas contain missing names or categories.")
    drawn["name"] = drawn["name"].str.strip()
    drawn["category"] = drawn["category"].str.strip().str.lower()
    if (drawn["name"] == "").any() or not (drawn["category"] == "developed").all():
        raise ValueError("Expected named study areas in the developed category.")

    # Project before geometric calculations; combine features with the same name.
    regions = drawn.to_crs(AREA_CRS).dissolve(by=["name", "category"], as_index=False)
    municipal_geom = unary_union(municipality.to_crs(AREA_CRS).geometry)
    municipal_km2 = municipal_geom.area / 1_000_000
    check_polygons(regions, "Dissolved study areas")

    # Clip to the administrative boundary, but do not remove water yet.
    original_areas = regions.geometry.area.copy()
    regions.geometry = regions.geometry.intersection(municipal_geom)
    check_polygons(regions, "Clipped study areas")
    regions["area_km2"] = regions.geometry.area / 1_000_000
    regions["outside_km2"] = (original_areas - regions.geometry.area).clip(lower=0) / 1_000_000
    regions["pct_municipality"] = 100 * regions["area_km2"] / municipal_km2

    # Use local UTM coordinates for perimeter and rotated-rectangle measurements.
    local = regions.to_crs(DISTANCE_CRS)
    regions["perimeter_km"] = local.geometry.length / 1000
    rectangles = local.geometry.apply(lambda geom: geom.minimum_rotated_rectangle)
    dimensions = [rectangle_dimensions(rect) for rect in rectangles]
    regions["approx_length_km"] = [length for length, width in dimensions]
    regions["approx_width_km"] = [width for length, width in dimensions]
    regions["geometry_type"] = regions.geometry.geom_type

    print(f"Original features: {len(drawn)}")
    print(f"Dissolved regions: {len(regions)}")
    print(f"Municipality area (including water): {municipal_km2:,.2f} km²")
    print("\nREGION MEASUREMENTS")
    columns = ["name", "geometry_type", "area_km2", "perimeter_km",
               "approx_length_km", "approx_width_km", "pct_municipality", "outside_km2"]
    print(regions[columns].to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    print("\nPAIRWISE OVERLAPPING AREA")
    for (_, a), (_, b) in combinations(regions.iterrows(), 2):
        overlap = a.geometry.intersection(b.geometry).area / 1_000_000
        print(f"{a['name']} / {b['name']}: {overlap:,.6f} km²")

    developed = unary_union(regions.geometry)
    developed_km2 = developed.area / 1_000_000
    print(f"\nCombined study area (no double-counting): {developed_km2:,.2f} km²")
    print(f"Share of administrative municipality: {100 * developed_km2 / municipal_km2:.2f}%")
    print("Note: this is not a land-only or impervious-surface percentage.")

    # Axis-aligned rectangles in the raster's CRS: useful for future COG windows.
    boxes = gpd.GeoDataFrame(regions[["name"]].copy(),
                            geometry=regions.geometry.envelope, crs=AREA_CRS)
    rotated = gpd.GeoDataFrame(regions[["name"]].copy(),
                               geometry=rectangles, crs=DISTANCE_CRS)

    outputs = {
        "anchorage_study_areas_analyzed.geojson": regions,
        "anchorage_study_area_bounding_boxes.geojson": boxes,
        "anchorage_study_area_rotated_rectangles.geojson": rotated,
    }
    print("\nOUTPUT FILES")
    for filename, gdf in outputs.items():
        path = DATA / filename
        gdf.to_crs(OUTPUT_CRS).to_file(path, driver="GeoJSON", index=False)
        print(path)

    print("\nPolygon analysis complete. Original files were not modified.")


if __name__ == "__main__":
    main()
