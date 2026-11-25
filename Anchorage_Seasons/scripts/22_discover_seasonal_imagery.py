"""Discover 2025 Anchorage Sentinel-2 acquisitions using STAC metadata only.
Run normally in PyCharm. No raster reads, imagery downloads or private AWS operations.
Footprint coverage and scene cloud metadata are screening proxies, NOT pixel quality.
"""
from __future__ import annotations
import argparse
import calendar
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
from pyproj import Transformer
from shapely import make_valid
from shapely.geometry import shape
from shapely.ops import transform, unary_union

STAC_URL = 'https://earth-search.aws.element84.com/v1'
COLLECTION = 'sentinel-2-c1-l2a'
ASSETS = ('blue', 'green', 'red', 'nir', 'swir16', 'scl')
# Matches the acquisition naming convention already verified in Scripts 17 and 19.
ID_PATTERN = re.compile(r'^(S2[A-Z])_(T\d{2}[A-Z]{3})_(\d{8}T\d{6})_L2A(?:_.*)?$')
MIN_FOOTPRINT = 99.9  # provisional discovery gate; actual quality tested later
TOP_PER_MONTH = 3
REFERENCE = 'S2C_20250612T212656'
PROJECT = Transformer.from_crs(4326, 3338, always_xy=True).transform


def polygonal(geom):
    if geom.is_empty:
        return geom
    if not geom.is_valid:
        geom = make_valid(geom)
    if geom.geom_type in ('Polygon', 'MultiPolygon'):
        return geom
    if geom.geom_type == 'GeometryCollection':
        return unary_union([polygonal(g) for g in geom.geoms if g.geom_type in
                            ('Polygon', 'MultiPolygon', 'GeometryCollection')])
    return unary_union([])


def cloud_value(item):
    try:
        v = float(item['properties']['eo:cloud_cover'])
        return v if math.isfinite(v) and 0 <= v <= 100 else None
    except (KeyError, TypeError, ValueError):
        return None


def parse_item(item, municipality):
    match = ID_PATTERN.fullmatch(item['id'])
    if match is None:
        raise ValueError(f"Unexpected Earth Search item ID: {item['id']}")
    sat, tile, stamp = match.groups()
    dt = datetime.strptime(stamp, '%Y%m%dT%H%M%S').replace(tzinfo=timezone.utc)
    geom = item.get('geometry')
    if not geom:
        raise ValueError(f"Missing STAC footprint: {item['id']}")
    clipped = polygonal(transform(PROJECT, polygonal(shape(geom)))).intersection(municipality)
    if clipped.is_empty or clipped.area <= 0:
        return None
    missing = [a for a in ASSETS if not item.get('assets', {}).get(a, {}).get('href')]
    return {'id': item['id'], 'tile': tile, 'group': f'{sat}_{stamp}',
            'date': dt.date().isoformat(), 'datetime_utc': dt.isoformat(),
            'cloud': cloud_value(item), 'missing_assets': missing, 'geometry': clipped}


def evaluate(group, scenes, municipality, land):
    # Use all complete-asset scenes, prioritizing known low cloud metadata as in Script 17.
    # Unique footprint areas avoid counting overlaps multiple times.
    eligible = [s for s in scenes if not s['missing_assets']]
    eligible.sort(key=lambda s: (s['cloud'] is None, s['cloud'] if s['cloud'] is not None else 100, s['id']))
    covered = unary_union([])
    weighted = known_area = missing_area = 0.0
    records = []
    for scene in eligible:
        unique = scene['geometry'].difference(covered)
        a = unique.intersection(land).area
        if scene['cloud'] is None:
            missing_area += a
        else:
            known_area += a
            weighted += a * scene['cloud']
        records.append({'id': scene['id'], 'tile': scene['tile'],
                        'scene_cloud_pct': scene['cloud'], 'unique_land_footprint_km2': round(a/1e6, 6)})
        covered = covered.union(scene['geometry'])
    total_land_covered = covered.intersection(land).area
    land_pct = min(100.0, 100 * total_land_covered / land.area)
    muni_pct = min(100.0, 100 * covered.area / municipality.area)
    # Missing scene cloud values penalized as 100 ONLY for ranking, not reported as measurements.
    proxy = (weighted + 100 * missing_area) / total_land_covered if total_land_covered > 0 else None
    return {'acquisition': group, 'date': scenes[0]['date'], 'datetime_utc': scenes[0]['datetime_utc'],
            'scene_count': len(scenes), 'complete_asset_scene_count': len(eligible),
            'municipal_footprint_pct': muni_pct, 'land_footprint_pct': land_pct,
            'land_weighted_scene_cloud_proxy_pct': proxy,
            'known_cloud_metadata_land_pct': 100 * known_area / total_land_covered if total_land_covered else 0,
            'passes_footprint_gate': muni_pct >= MIN_FOOTPRINT and land_pct >= MIN_FOOTPRINT,
            'scenes': records, 'excluded_missing_assets': [
                {'id': s['id'], 'missing_assets': s['missing_assets']} for s in scenes if s['missing_assets']]}


def rank_key(row):
    return (not row['passes_footprint_gate'],
            0 if row['passes_footprint_gate'] else -min(row['land_footprint_pct'], row['municipal_footprint_pct']),
            row['land_weighted_scene_cloud_proxy_pct'] if row['land_weighted_scene_cloud_proxy_pct'] is not None else 101,
            -row['known_cloud_metadata_land_pct'], -row['land_footprint_pct'], row['acquisition'])


def shortlist(rows, year):
    selected = []
    for month in range(1, 13):
        subset = sorted([r for r in rows if r['date'].startswith(f'{year}-{month:02d}')], key=rank_key)
        seen = set()
        for r in subset:
            if not r['passes_footprint_gate'] or r['date'] in seen:
                continue
            seen.add(r['date'])
            selected.append(r)
            if len(seen) == TOP_PER_MONTH:
                break
    return selected


def load_geography(processed):
    m = gpd.read_file(processed / 'anchorage_municipality.geojson').to_crs(3338)
    l = gpd.read_file(processed / 'anchorage_land_study_regions.geojson').to_crs(3338)
    expected = {'Anchorage Bowl', 'Chugiak - Eagle River', 'Girdwood - Alyeska', 'Remaining Municipal Land'}
    if len(l) != 4 or set(l['name']) != expected:
        raise ValueError('Expected four permanent study regions')
    def checked(g):
        before = g.area
        fixed = polygonal(g)
        if fixed.is_empty or not fixed.is_valid or abs(fixed.area - before) > max(100., before * 1e-7):
            raise ValueError('Invalid geometry or material repair')
        return fixed
    municipality = unary_union([checked(g) for g in m.geometry])
    land = unary_union([checked(g) for g in l.geometry])
    if land.difference(municipality).area > max(100., land.area*1e-7):
        raise ValueError('Study land extends outside municipal boundary')
    if abs(sum(g.area for g in l.geometry) - land.area) > max(100.,land.area*1e-7):
        raise ValueError('Study land overlap or material area inconsistency')
    print(f'Municipality: {municipality.area/1e6:,.3f} km2; study land: {land.area/1e6:,.3f} km2', flush=True)
    return municipality, land


def run(processed, year):
    from pystac_client import Client
    municipality, land = load_geography(processed)
    wgs84 = gpd.GeoSeries([municipality], crs=3338).to_crs(4326).iloc[0]
    bbox = list(wgs84.bounds)
    client = Client.open(STAC_URL)
    groups = defaultdict(list)
    seen = set()
    print('Metadata only: no imagery pixels will be downloaded.', flush=True)
    for month in range(1, 13):
        last = calendar.monthrange(year, month)[1]
        interval = f'{year}-{month:02d}-01T00:00:00Z/{year}-{month:02d}-{last:02d}T23:59:59.999999Z'
        raw = 0
        accepted = 0
        for item in client.search(collections=[COLLECTION], bbox=bbox, datetime=interval, limit=100).items():
            raw += 1
            if item.id in seen:
                continue
            seen.add(item.id)
            rec = parse_item(item.to_dict(), municipality)
            if rec is not None:
                if not rec['date'].startswith(f'{year}-'):
                    raise ValueError(f'Unexpected year in acquisition ID: {rec["id"]}')
                groups[rec['group']].append(rec)
                accepted += 1
        print(f'{year}-{month:02d}: {raw} catalog items; {accepted} unique intersecting scenes', flush=True)
    if not groups:
        raise RuntimeError('No intersecting acquisitions found; no outputs written')
    rows = [evaluate(k, v, municipality, land) for k, v in sorted(groups.items())]
    chosen = shortlist(rows, year)
    selected_ids = {r['acquisition'] for r in chosen}
    reference = next((r for r in rows if r['acquisition'] == REFERENCE), None)
    print('\nMONTHLY CANDIDATES (cloud proxy is NOT measured local cloud cover)')
    print('Month   Groups  Full*   Date        Acquisition           Municipal%   Land%  Cloud proxy%  Scenes')
    for month in range(1, 13):
        subset = [r for r in rows if r['date'].startswith(f'{year}-{month:02d}')]
        picks = [r for r in chosen if r['date'].startswith(f'{year}-{month:02d}')]
        full = sum(r['passes_footprint_gate'] for r in subset)
        print(f'{year}-{month:02d} {len(subset):6d} {full:6d}', end='')
        if not picks:
            print('   No candidate meets footprint gate')
        for i, r in enumerate(picks):
            if i: print(' ' * 21, end='')
            p = r['land_weighted_scene_cloud_proxy_pct']
            ptext = f'{p:6.2f}' if p is not None else '   N/A'
            print(f'   {r["date"]}  {r["acquisition"]:22s} {r["municipal_footprint_pct"]:10.3f} {r["land_footprint_pct"]:8.3f} {ptext:>12s} {r["complete_asset_scene_count"]:7d}')
    print(f'* Full = >= {MIN_FOOTPRINT}% of both municipal and study-land footprints.')
    if year == 2025:
        if reference:
            print(f'June reference: {REFERENCE}; {reference["complete_asset_scene_count"]} complete scenes; '
                  f'municipal footprint {reference["municipal_footprint_pct"]:.3f}%; '
                  f'cloud proxy {reference["land_weighted_scene_cloud_proxy_pct"]}%')
        else:
            print('WARNING: verified June reference not found in current catalog response.')
    csv_path = processed / f'anchorage_{year}_acquisition_candidates.csv'
    json_path = processed / f'anchorage_{year}_acquisition_shortlist.json'
    keys = ['acquisition', 'date', 'datetime_utc', 'scene_count', 'complete_asset_scene_count',
            'municipal_footprint_pct', 'land_footprint_pct', 'land_weighted_scene_cloud_proxy_pct',
            'known_cloud_metadata_land_pct', 'passes_footprint_gate', 'shortlisted', 'reference']
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in sorted(rows, key=lambda r: (r['date'], rank_key(r))):
            row = {k: r[k] for k in keys if k in r}
            row.update(shortlisted=r['acquisition'] in selected_ids, reference=r['acquisition'] == REFERENCE)
            writer.writerow(row)
    with json_path.open('w', encoding='utf-8') as f:
        json.dump({'year': year, 'stac_url': STAC_URL, 'collection': COLLECTION,
                   'required_assets': ASSETS, 'grouping': 'satellite + acquisition timestamp embedded in item ID (UTC)',
                   'footprint_gate_pct': MIN_FOOTPRINT, 'top_distinct_dates_per_month': TOP_PER_MONTH,
                   'cloud_proxy': 'Area-weighted scene-wide cloud metadata on nonoverlapping study-land footprints; missing metadata penalized at 100 for ranking only. Not measured local cloud or usable pixels.',
                   'limitations': 'Footprints do not establish actual valid data coverage; no SCL or reflectance pixels inspected. Winter and snow are not assumed unusable or cloud-free.',
                   'acquisition_count': len(rows), 'shortlist': chosen, 'verified_june_reference': reference,
                   'all_acquisitions': rows}, f, indent=2, allow_nan=False)
    print(f'\nTotal acquisition groups: {len(rows)}; shortlisted: {len(chosen)}')
    print(f'Created: {csv_path}\nCreated: {json_path}')
    print('Complete. No raster pixels downloaded. Existing imagery and polygons unchanged.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--year', type=int, default=2025)
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[1] / 'data' / 'processed', args.year)
