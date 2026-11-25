"""Create mutually exclusive land study areas using Census mapped water.

Run from the Anchorage_Seasons project, with this script in scripts/.
Requires GeoPandas and Shapely 2.0+. Original inputs are never modified.
"""
from itertools import combinations
from pathlib import Path

import geopandas as gpd
from shapely import make_valid
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
AREA_CRS = 'EPSG:3338'
OUTPUT_CRS = 'EPSG:4326'
WATER_FILENAME = 'tl_2025_02020_areawater.shp'
TOLERANCE_M2 = 1.0  # numerical overlay tolerance, not a mapping accuracy claim


def polygonal(geom):
    """Repair invalid geometry and retain polygon components only."""
    if geom is None or geom.is_empty:
        return Polygon()
    if not geom.is_valid:
        geom = make_valid(geom)
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    if hasattr(geom, 'geoms'):
        parts = [polygonal(part) for part in geom.geoms]
        return unary_union([part for part in parts if not part.is_empty])
    return Polygon()


def load_polygons(path):
    gdf = gpd.read_file(path)
    if gdf.empty or gdf.crs is None:
        raise ValueError(f'Empty dataset or missing CRS: {path}')
    gdf = gdf.to_crs(AREA_CRS)
    invalid = sum(not geom.is_valid for geom in gdf.geometry if geom is not None)
    gdf.geometry = gdf.geometry.apply(polygonal)
    empty = int(gdf.geometry.is_empty.sum())
    gdf = gdf.loc[~gdf.geometry.is_empty].copy()
    if gdf.empty:
        raise ValueError(f'No polygonal geometry in {path}')
    print(f'{path.name}: {len(gdf)} polygon features; {invalid} repaired; {empty} empty/nonpolygonal omitted')
    return gdf


def geometry_counts(geom):
    parts = [geom] if isinstance(geom, Polygon) else list(geom.geoms)
    return len(parts), sum(len(part.interiors) for part in parts)


def km2(geom):
    return geom.area / 1_000_000


def check_area(label, value_m2):
    if abs(value_m2) > TOLERANCE_M2:
        raise ValueError(f'{label} failed: {value_m2:,.6f} m²')
    print(f'PASS: {label} ({value_m2:.6f} m²)')


def save_geojson(gdf, filename):
    path = DATA / filename
    gdf.to_crs(OUTPUT_CRS).to_file(path, driver='GeoJSON', index=False)
    print(path)


def main():
    # Search nested extraction directories as well as data/shapefiles itself.
    matches = sorted((ROOT / 'data' / 'shapefiles').rglob(WATER_FILENAME))
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected exactly one {WATER_FILENAME} in data/shapefiles; found {len(matches)}: {matches}')
    municipal = load_polygons(DATA / 'anchorage_municipality.geojson')
    drawn = load_polygons(DATA / 'anchorage_study_areas.geojson')
    waters = load_polygons(matches[0])
    if not {'name', 'category'}.issubset(drawn.columns):
        raise ValueError('Study areas must contain name and category fields.')
    if drawn[['name', 'category']].isna().any().any():
        raise ValueError('Missing study-area name or category.')
    drawn['name'] = drawn['name'].str.strip()
    drawn['category'] = drawn['category'].str.strip().str.lower()
    if (drawn['name'] == '').any() or not (drawn['category'] == 'developed').all():
        raise ValueError('Expected named study areas with category developed.')
    drawn = drawn.dissolve(by=['name', 'category'], as_index=False)

    municipality = polygonal(unary_union(municipal.geometry))
    # Union first, to avoid counting overlapping water polygons twice.
    water = polygonal(unary_union(waters.geometry).intersection(municipality))
    land = polygonal(municipality.difference(water))
    if water.is_empty or land.is_empty:
        raise ValueError('Expected both mapped water and land within the municipality; check inputs.')

    print('\nMUNICIPAL AREA BREAKDOWN')
    print(f'Administrative municipality: {km2(municipality):,.3f} km²')
    print(f'Mapped water:                {km2(water):,.3f} km²')
    print(f'Municipal land:              {km2(land):,.3f} km²')

    rows = []
    for _, row in drawn.iterrows():
        original = polygonal(row.geometry)
        within = polygonal(original.intersection(municipality))
        clipped = polygonal(within.intersection(land))
        if clipped.is_empty:
            raise ValueError(f"Region has no municipal land: {row['name']}")
        parts, holes = geometry_counts(clipped)
        rows.append(dict(name=row['name'], category='developed',
                         drawn_km2=km2(original),
                         outside_km2=km2(original.difference(municipality)),
                         water_removed_km2=km2(within.intersection(water)),
                         area_km2=km2(clipped),
                         pct_land=100 * clipped.area / land.area,
                         geometry_type=clipped.geom_type,
                         polygon_parts=parts, holes=holes, geometry=clipped))
    developed = polygonal(unary_union([row['geometry'] for row in rows]))
    remaining = polygonal(land.difference(developed))
    if remaining.is_empty:
        raise ValueError('Expected a nonempty remaining land region.')
    parts, holes = geometry_counts(remaining)
    rows.append(dict(name='Remaining Municipal Land', category='remaining_land',
                     drawn_km2=None, outside_km2=None, water_removed_km2=None,
                     area_km2=km2(remaining), pct_land=100 * remaining.area / land.area,
                     geometry_type=remaining.geom_type, polygon_parts=parts,
                     holes=holes, geometry=remaining))
    regions = gpd.GeoDataFrame(rows, geometry='geometry', crs=AREA_CRS)

    print('\nREGION MEASUREMENTS')
    cols = ['name', 'drawn_km2', 'water_removed_km2', 'outside_km2', 'area_km2', 'pct_land', 'geometry_type', 'polygon_parts', 'holes']
    print(regions[cols].to_string(index=False, float_format=lambda x: f'{x:,.3f}', na_rep='—'))
    print(f'\nCombined developed study land: {km2(developed):,.3f} km² ({100 * developed.area / land.area:.2f}%)')
    print(f'Remaining municipal land:      {km2(remaining):,.3f} km² ({100 * remaining.area / land.area:.2f}%)')
    print('Note: approximate study-area extents, not impervious-surface coverage.')

    print('\nGEOMETRY AND AREA CHECKS')
    check_area('Municipal land + water = administrative area', land.area + water.area - municipality.area)
    check_area('Developed + remaining = municipal land', developed.area + remaining.area - land.area)
    check_area('Study region union matches municipal land', unary_union(regions.geometry).symmetric_difference(land).area)
    for _, a in regions.iterrows():
        if not a.geometry.is_valid:
            raise ValueError(f"Invalid output geometry: {a['name']}")
        check_area(f"{a['name']} outside municipal land", a.geometry.difference(land).area)
        check_area(f"{a['name']} water overlap", a.geometry.intersection(water).area)
    for (_, a), (_, b) in combinations(regions.iterrows(), 2):
        check_area(f"{a['name']} / {b['name']} overlap", a.geometry.intersection(b.geometry).area)
    check_area('Sum of all region areas = municipal land', regions.geometry.area.sum() - land.area)

    print('\nOUTPUT FILES')
    save_geojson(gpd.GeoDataFrame([dict(name='Mapped Municipal Water', area_km2=km2(water), geometry=water)], crs=AREA_CRS), 'anchorage_municipal_water.geojson')
    save_geojson(gpd.GeoDataFrame([dict(name='Municipal Land', area_km2=km2(land), geometry=land)], crs=AREA_CRS), 'anchorage_municipal_land.geojson')
    save_geojson(regions, 'anchorage_land_study_regions.geojson')
    print('\nLand-region analysis complete. Original files were not modified.')


if __name__ == '__main__':
    main()
