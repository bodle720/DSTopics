"""Script 25: Prepare the selected 2025 Sentinel-2 seasonal acquisitions.

Place in Anchorage_Seasons/scripts and run with the existing Python environment.
Default: verify the existing June reference, then prepare seven new dates.
Optional: --date YYYY-MM-DD to prepare a single selected date (June is still verified).
Uses public Earth Search HTTPS assets; no AWS uploads or private AWS operations.
Do not run multiple instances concurrently. Script 17 remains unchanged.
"""
from contextlib import ExitStack
from pathlib import Path
import argparse
import json
import re
import shutil
import time

import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.shutil import copy as rio_copy
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import bounds as window_bounds

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
REFERENCE = DATA / 'anchorage_s2_cog_10m.tif'
BOUNDARY = DATA / 'anchorage_municipality.geojson'
INVENTORY = DATA / 'anchorage_2025_acquisition_candidates.csv'
API = 'https://earth-search.aws.element84.com/v1'
COLLECTION = 'sentinel-2-c1-l2a'
JUNE = '2025-06-12'
# Acquisition identifiers from the actual Script 24 console output.
ACQUISITIONS = {
    '2025-02-07': 'S2B_20250207T212545',
    '2025-03-29': 'S2B_20250329T212803',
    '2025-05-23': 'S2C_20250523T212539',
    '2025-06-12': 'S2C_20250612T212656',
    '2025-07-27': 'S2B_20250727T212521',
    '2025-08-11': 'S2C_20250811T212539',
    '2025-09-30': 'S2C_20250930T212536',
    '2025-11-19': 'S2C_20251119T212811',
}
# The six tile identifiers used by the June reference and the expanded screening.
TILES = ['05VPH', '06VUN', '06VVN', '05VPJ', '06VUP', '06VVP']
KEYS = ['blue', 'green', 'red', 'nir', 'swir16']
NAMES = ['B02 Blue', 'B03 Green', 'B04 Red', 'B08 NIR', 'B11 SWIR']
STAGE_NAMES = ('spectral.tif', 'quality.tif', 'spectral_cog.tif', 'quality_cog.tif', 'sources.json', 'sources.json.tmp')
METHOD = 'first valid B02/B03/B04/B08, fixed scene priority; no cross-scene filling'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def pinned_ids(date):
    satellite, stamp = ACQUISITIONS[date].split('_')
    return [f'{satellite}_T{tile}_{stamp}_L2A' for tile in TILES]


def check_inventory():
    """Cross-check pinned groups and scene lists when present in the original CSV.

    No assumptions about CSV column names. Exact group and item tokens are used.
    Some inventory formats may store only groups or tile counts; in that case
    the exact six constructed IDs must still all resolve in STAC before building.
    """
    text = INVENTORY.read_text(encoding='utf-8-sig')
    for date, group in ACQUISITIONS.items():
        satellite, stamp = group.split('_')
        pattern = rf'{satellite}_T[0-9]{{2}}[A-Z]{{3}}_{stamp}_L2A'
        found = set(re.findall(pattern, text))
        require(group in text or bool(found), f'Acquisition absent from Script 22 inventory: {group}')
        if found:
            require(found == set(pinned_ids(date)), f'Inventory scene set differs for {group}: {sorted(found)}')
        elif date != JUNE:
            print(f'  {group}: inventory group found; exact six scene IDs will be verified in STAC.')
    print('Pinned acquisition inventory checks: PASS', flush=True)


def output_paths(date):
    prefix = f"anchorage_{date.replace('-', '')}"
    return (DATA / f'{prefix}_spectral_10m.tif',
            DATA / f'{prefix}_quality_10m.tif',
            DATA / f'{prefix}_sources.json')


def read_items(ids, preserve_order=False):
    from pystac_client import Client
    client = Client.open(API)
    # Exact IDs, not a fresh cloud/date search which could change the inputs.
    items = {i.id: i for i in client.search(collections=[COLLECTION], ids=ids).items()}
    missing = set(ids) - set(items)
    if missing:
        raise RuntimeError(f'Missing pinned STAC items: {sorted(missing)}')
    records = []
    for scene_id in ids:
        item = items[scene_id]
        assets = {}
        for key in KEYS + ['scl']:
            asset = item.assets[key]
            band = asset.extra_fields.get('raster:bands', [{}])[0]
            if key != 'scl' and not all(k in band for k in ('scale', 'offset', 'nodata')):
                raise RuntimeError(f'Missing radiometric metadata: {scene_id} {key}: {band}')
            if key != 'scl' and float(band['nodata']) != 0:
                raise RuntimeError(f'Unexpected reflectance NoData: {scene_id} {key}')
            if key == 'scl' and float(band.get('nodata', 0)) != 0:
                raise RuntimeError(f'Unexpected SCL NoData: {scene_id}')
            assets[key] = {'href': asset.get_absolute_href(), 'raster:bands': band}
            if not assets[key]['href'] or not assets[key]['href'].startswith('https://'):
                raise RuntimeError(f'Expected public HTTPS asset: {scene_id} {key}')
        records.append({'id': scene_id, 'datetime': item.properties.get('datetime'),
                        'cloud_cover': item.properties.get('eo:cloud_cover'), 'assets': assets})
    for record in records:
        cloud = record['cloud_cover']
        if not isinstance(cloud, (int, float)) or not np.isfinite(cloud) or not 0 <= cloud <= 100:
            raise RuntimeError(f'Invalid scene cloud metadata: {record["id"]}: {cloud}')
    if not preserve_order:
        records.sort(key=lambda r: (r['cloud_cover'], r['id']))
    scales, offsets = [], []
    for key in KEYS:
        pairs = {(float(r['assets'][key]['raster:bands']['scale']),
                  float(r['assets'][key]['raster:bands']['offset'])) for r in records}
        if len(pairs) != 1:
            raise RuntimeError(f'Inconsistent source encodings for {key}: {pairs}')
        scale, offset = pairs.pop()
        if not np.isfinite(scale) or scale <= 0 or not np.isfinite(offset):
            raise RuntimeError(f'Invalid reflectance encoding for {key}: {scale}, {offset}')
        scales.append(scale)
        offsets.append(offset)
        print(f'{key}: reflectance = DN * {scale} + ({offset})')
    return records, scales, offsets


def fill_window(current, quality, new_rgbnir, swir, scl, inside, scene_code):
    """First valid four-band scene wins; never borrow SWIR or SCL from another scene."""
    eligible = inside & (quality[1] == 0) & np.all(new_rgbnir != 0, axis=0)
    if not np.any(eligible):
        return 0
    if np.any(scl[eligible] > 11):
        raise RuntimeError('Unexpected SCL class outside 0..11')
    current[:4, eligible] = new_rgbnir[:, eligible]
    current[4, eligible] = swir[eligible]
    quality[0, eligible] = scl[eligible]
    quality[1, eligible] = scene_code
    return int(eligible.sum())


def overlaps(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def build(date, records, scales, offsets, grid, geometry, spectral_stage, quality_stage):
    common = dict(driver='GTiff', **grid, tiled=True, blockxsize=512, blockysize=512,
                  compress='DEFLATE', nodata=0, BIGTIFF='IF_SAFER')
    vrt_grid = {k: grid[k] for k in ('crs', 'transform', 'width', 'height')}
    with rio.open(spectral_stage, 'w+', count=5, dtype='uint16', **common) as out, \
         rio.open(quality_stage, 'w+', count=2, dtype='uint8', **common) as qout:
        out.descriptions = tuple(NAMES)
        out.scales = tuple(scales)
        out.offsets = tuple(offsets)
        out.update_tags(acquisition_date=date, source_collection=COLLECTION,
                        B11_native_resolution_m=20, resampling='nearest')
        qout.descriptions = ('SCL (0-11)', 'Source scene code (1-6)')
        qout.update_tags(acquisition_date=date, SCL_native_resolution_m=20,
                         resampling='nearest', **{f'scene_{i}': r['id'] for i, r in enumerate(records, 1)})
        # Build the municipality mask once. Pixel center rule; no boundary double-counting.
        inside = geometry_mask(geometry, out_shape=(out.height, out.width),
                               transform=out.transform, invert=True)
        windows = [w for _, w in out.block_windows(1)]
        for code, record in enumerate(records, 1):
            print(f'\nScene {code}: {record["id"]} (cloud metadata: {record["cloud_cover"]}%)', flush=True)
            with ExitStack() as stack:
                sources = [stack.enter_context(rio.open(record['assets'][key]['href']))
                           for key in KEYS + ['scl']]
                for key, src in zip(KEYS + ['scl'], sources):
                    expected = 20 if key in ('swir16', 'scl') else 10
                    if src.count != 1 or not np.allclose(src.res, (expected, expected)):
                        raise RuntimeError(f'Unexpected source geometry for {key}: {src.res}')
                    expected_dtype = 'uint8' if key == 'scl' else 'uint16'
                    if src.dtypes[0] != expected_dtype:
                        raise RuntimeError(f'Unexpected dtype for {key}: {src.dtypes[0]}')
                    if src.nodata is not None and src.nodata != 0:
                        raise RuntimeError(f'Unexpected source NoData for {key}: {src.nodata}')
                scene_bounds = transform_bounds(sources[0].crs, out.crs, *sources[0].bounds, densify_pts=21)
                # Pad the bounding box only for window selection, never for the actual pixel mask.
                scene_bounds = (scene_bounds[0]-100, scene_bounds[1]-100,
                                scene_bounds[2]+100, scene_bounds[3]+100)
                vrts = [stack.enter_context(WarpedVRT(src, **vrt_grid, src_nodata=0,
                         nodata=0, resampling=Resampling.nearest)) for src in sources]
                filled = 0
                for w in windows:
                    if not overlaps(window_bounds(w, out.transform), scene_bounds):
                        continue
                    rows, cols = w.toslices()
                    mask = inside[rows, cols]
                    if not mask.any():
                        continue
                    q = qout.read(window=w)
                    if not np.any(mask & (q[1] == 0)):
                        continue
                    rgbnir = np.stack([v.read(1, window=w) for v in vrts[:4]])
                    if not np.any(mask & (q[1] == 0) & np.all(rgbnir != 0, axis=0)):
                        continue
                    swir = vrts[4].read(1, window=w)
                    scl = vrts[5].read(1, window=w)
                    current = out.read(window=w)
                    n = fill_window(current, q, rgbnir, swir, scl, mask, code)
                    if n:
                        out.write(current, window=w)
                        qout.write(q, window=w)
                        filled += n
                print(f'  Newly assigned pixels: {filled:,}', flush=True)
        return int(inside.sum())


def verify(spectral, quality, reference_grid, scales, offsets, inside, date, ids):
    total = int(inside.sum())
    counts = np.zeros(12, dtype=np.int64)
    scenes = np.zeros(len(ids)+1, dtype=np.int64)
    b11_valid = 0
    with rio.open(spectral) as s, rio.open(quality) as q:
        for src, bands, dtype in [(s, 5, 'uint16'), (q, 2, 'uint8')]:
            require(src.count == bands and all(t == dtype for t in src.dtypes), 'Band count or dtype mismatch')
            require(src.nodata == 0, 'NoData mismatch')
            require(src.crs == reference_grid['crs'] and src.transform == reference_grid['transform'], 'CRS or transform mismatch')
            require((src.width, src.height) == (reference_grid['width'], reference_grid['height']), 'Raster dimensions mismatch')
            require(src.tags(ns='IMAGE_STRUCTURE').get('LAYOUT') == 'COG', 'Not a COG')
            require(src.tags().get('acquisition_date') == date, 'Date tag mismatch')
            require(all(shape == (512, 512) for shape in src.block_shapes), 'COG block size mismatch')
            require(bool(src.overviews(1)), 'COG overviews missing')
        require(tuple(s.descriptions) == tuple(NAMES), 'Spectral band order mismatch')
        require(tuple(q.descriptions) == ('SCL (0-11)', 'Source scene code (1-6)'), 'Quality band order mismatch')
        require(all(q.tags().get(f'scene_{i}') == sid for i, sid in enumerate(ids, 1)), 'Scene tag mismatch')
        require(np.allclose(s.scales, scales, rtol=0, atol=1e-12) and np.allclose(s.offsets, offsets, rtol=0, atol=1e-12), 'Reflectance metadata mismatch')
        for _, w in s.block_windows(1):
            a, b = s.read(window=w), q.read(window=w)
            assigned = b[1] > 0
            rows, cols = w.toslices()
            require(not np.any(assigned & ~inside[rows, cols]), "Data assigned outside municipality")
            if np.any(b[1] > len(ids)) or np.any(b[0] > 11):
                raise RuntimeError('Invalid quality values')
            if not np.array_equal(assigned, np.all(a[:4] != 0, axis=0)):
                raise RuntimeError('Source-scene map disagrees with valid reflectance')
            if np.any(a[:, ~assigned] != 0) or np.any(b[0, ~assigned] != 0):
                raise RuntimeError('Unassigned pixel contains data')
            counts += np.bincount(b[0, assigned], minlength=12)
            scenes += np.bincount(b[1, assigned], minlength=len(ids)+1)
            b11_valid += int(np.count_nonzero(assigned & (a[4] != 0)))
    n = int(scenes.sum())
    if not 0 < n <= total:
        raise RuntimeError(f'Unexpected coverage: {n}/{total}')
    print('\nVERIFICATION')
    print(f'  Municipality pixels: {total:,}')
    print(f'  Assigned four-band pixels: {n:,} ({100*n/total:.4f}%)')
    print(f'  Valid B11 pixels: {b11_valid:,}')
    print(f'  Classified SCL pixels: {n-counts[0]:,}')
    for code, count in enumerate(counts):
        print(f'  SCL {code:2}: {count:12,}')
    print('  Grid, COG layout, reflectance metadata and source-map checks: PASS')
    return {'municipality_pixels': total, 'assigned_pixels': n, 'valid_b11': b11_valid,
            'scl_counts': counts.tolist(), 'scene_counts': scenes.tolist()}



def verify_manifest(date, paths, grid, inside):
    spectral, quality, manifest = paths
    data = json.loads(manifest.read_text(encoding='utf-8'))
    require(data['date'] == date and data['collection'] == COLLECTION, 'Manifest date/collection mismatch')
    require(data['spectral_band_order'] == NAMES, 'Manifest band order mismatch')
    require(data['quality_band_order'] == ['SCL', 'source scene code'], 'Manifest quality order mismatch')
    require(data['source_selection'] == METHOD, 'Manifest source-selection rule mismatch')
    records = data['source_scene_priority']
    ids = [r['id'] for r in records]
    require(len(ids) == len(pinned_ids(date)) and set(ids) == set(pinned_ids(date)), 'Manifest pinned scene IDs mismatch')
    if date == JUNE:
        require(ids == pinned_ids(date), 'June original scene priority mismatch')
    else:
        require(records == sorted(records, key=lambda r: (r['cloud_cover'], r['id'])), 'Scene priority mismatch')
    scales, offsets = data['scale'], data['offset']
    require(len(scales) == len(offsets) == 5, 'Manifest reflectance metadata length mismatch')
    for i, key in enumerate(KEYS):
        require(np.isfinite(scales[i]) and scales[i] > 0 and np.isfinite(offsets[i]), 'Invalid manifest encoding')
        for record in records:
            band = record['assets'][key]['raster:bands']
            require(float(band['scale']) == scales[i] and float(band['offset']) == offsets[i] and float(band['nodata']) == 0, 'Manifest asset encoding mismatch')
    stats = verify(spectral, quality, grid, scales, offsets, inside, date, ids)
    require(stats == data['statistics'], 'Raster statistics differ from manifest')
    if stats['assigned_pixels'] / stats['municipality_pixels'] < 0.999:
        print('  WARNING: assigned geographic imagery coverage is below 99.9%; review before analysis.')
    return stats


def check_cog_pixels(stage, cog):
    """Lossless full-resolution COG conversion check, independent of overviews."""
    with rio.open(stage) as src, rio.open(cog) as dst:
        require((src.count, src.width, src.height) == (dst.count, dst.width, dst.height), 'COG conversion shape mismatch')
        for _, w in src.block_windows(1):
            require(np.array_equal(src.read(window=w), dst.read(window=w)), f'COG conversion changed pixels: {w}')


def publish_ready(date, stage, final, grid, inside, verified_stats=None):
    """Resume a completed staging transaction, publishing the manifest last."""
    ready = tuple(stage / n for n in ('spectral_cog.tif', 'quality_cog.tif', 'sources.json'))
    chosen = []
    for src, dst in zip(ready, final):
        require(src.exists() != dst.exists(), f'Ambiguous or missing staged/final output: {src.name}; no files overwritten')
        chosen.append(dst if dst.exists() else src)
    stats = verify_manifest(date, chosen, grid, inside) if verified_stats is None else verified_stats
    for src, dst in zip(ready, final):
        if not dst.exists():
            src.rename(dst)
    clean_stage(stage)
    return stats


def clean_stage(stage):
    """Delete only Script 25's named intermediate files, never existing products."""
    if not stage.exists():
        return
    require(stage.is_dir() and not stage.is_symlink(), f'Unsafe staging directory: {stage}')
    entries = list(stage.iterdir())
    require(all(p.name in STAGE_NAMES and p.is_file() and not p.is_symlink() for p in entries), f'Unexpected staging files: {stage}; inspect manually')
    for p in entries:
        p.unlink()
    stage.rmdir()


def prepare(date, grid, geometry, inside):
    final = output_paths(date)
    stage = DATA / f".script25_staging_{date.replace('-', '')}"
    print(f'\n=== {date} | {ACQUISITIONS[date]} ===', flush=True)
    if all(p.is_file() for p in final):
        stats = verify_manifest(date, final, grid, inside)
        print('  Existing outputs verified and reused (no downloads).', flush=True)
        # Do not modify the original June outputs or any June staging directory.
        if date != JUNE and stage.exists():
            clean_stage(stage)
        return 'reused', stats
    if date == JUNE:
        raise RuntimeError('June reference is missing or incomplete. Preserve it; Script 25 will not rebuild June.')
    if stage.exists():
        require(stage.is_dir() and not stage.is_symlink(), f'Unsafe staging path: {stage}')
        # A valid manifest written after all checks is the ready-to-publish marker.
        if (stage / 'sources.json').is_file():
            print('  Recovering previously verified staged outputs.', flush=True)
            return 'recovered', publish_ready(date, stage, final, grid, inside)
        require(not any(p.exists() for p in final), 'Partial final outputs with no ready staging manifest; inspect manually')
        print('  Restarting interrupted acquisition; removing only Script 25 intermediate files.', flush=True)
        clean_stage(stage)
    require(not any(p.exists() for p in final), 'Incomplete existing outputs; move aside deliberately after inspection. Nothing overwritten.')
    require(shutil.disk_usage(DATA).free >= 4 * 1024**3, 'Leave at least 4 GiB free for staging and COG creation.')
    records, scales, offsets = read_items(pinned_ids(date))
    stage.mkdir()
    start = time.monotonic()
    with rio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', GDAL_HTTP_MAX_RETRY='5',
                 GDAL_HTTP_RETRY_DELAY='3', GDAL_CACHEMAX=128*1024**2):
        total = build(date, records, scales, offsets, grid, geometry, stage/'spectral.tif', stage/'quality.tif')
        require(total == int(inside.sum()), 'Municipality mask count mismatch')
        for name, resampling in [('spectral', 'AVERAGE'), ('quality', 'NEAREST')]:
            print(f'  Creating {name} COG...', flush=True)
            rio_copy(stage/f'{name}.tif', stage/f'{name}_cog.tif', driver='COG',
                     COMPRESS='DEFLATE', BLOCKSIZE=512, BIGTIFF='IF_SAFER',
                     OVERVIEW_RESAMPLING=resampling)
            check_cog_pixels(stage/f'{name}.tif', stage/f'{name}_cog.tif')
            print(f'  {name} COG full-resolution pixel equality: PASS', flush=True)
        stats = verify(stage/'spectral_cog.tif', stage/'quality_cog.tif', grid,
                       scales, offsets, inside, date, [r['id'] for r in records])
    if stats['assigned_pixels'] / stats['municipality_pixels'] < 0.999:
        print('  WARNING: assigned geographic imagery coverage is below 99.9%; review before analysis.')
    manifest = dict(date=date, collection=COLLECTION, source_scene_priority=records,
                    spectral_band_order=NAMES, scale=scales, offset=offsets,
                    source_selection=METHOD,
                    resampling='nearest for all full-resolution bands; B11 and SCL native 20 m',
                    quality_band_order=['SCL', 'source scene code'], statistics=stats,
                    acquisition_id=ACQUISITIONS[date],
                    priority_rule='ascending STAC scene cloud cover; ties by scene ID',
                    generated_by='25_prepare_seasonal_imagery.py')
    # Write the ready marker atomically. If interrupted before this, rerun the date.
    (stage/'sources.json.tmp').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (stage/'sources.json.tmp').rename(stage/'sources.json')
    stats = publish_ready(date, stage, final, grid, inside, verified_stats=stats)
    for p in final:
        print(f'  {p} ({p.stat().st_size / 1e6:.2f} MB)', flush=True)
    print(f'  Date completed in {(time.monotonic()-start)/60:.1f} minutes.', flush=True)
    return 'created', stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', choices=list(ACQUISITIONS), help='Only prepare this selected date; always verify June first')
    args = parser.parse_args()
    check_inventory()
    with rio.open(REFERENCE) as ref:
        grid = dict(crs=ref.crs, transform=ref.transform, width=ref.width, height=ref.height)
        require(ref.crs.to_epsg() == 3338 and np.allclose(ref.res, (10, 10)), 'Reference must be EPSG:3338, 10 m')
        require((ref.width, ref.height) == (10739, 8299), 'Reference dimensions changed')
    geometry = list(gpd.read_file(BOUNDARY).to_crs(grid['crs']).geometry)
    require(bool(geometry) and all(not g.is_empty and g.is_valid for g in geometry), 'Invalid municipal boundary')
    inside = geometry_mask(geometry, out_shape=(grid['height'], grid['width']), transform=grid['transform'], invert=True)
    require(int(inside.sum()) == 50362515, 'Municipality mask differs from verified Script 17 reference')
    print(f"Grid: {grid['width']:,} x {grid['height']:,}, EPSG:3338, 10 m; municipality pixels: {inside.sum():,}", flush=True)
    selected = [JUNE] + [d for d in ACQUISITIONS if d != JUNE and (args.date is None or d == args.date)]
    results = {}
    for i, date in enumerate(selected, 1):
        print(f'\nAcquisition {i}/{len(selected)}', flush=True)
        status, stats = prepare(date, grid, geometry, inside)
        results[date] = (status, stats)
    print('\nSEASONAL PREPARATION SUMMARY (geographic coverage, not SCL-usable coverage)')
    for date, (status, stats) in sorted(results.items()):
        coverage = 100 * stats['assigned_pixels'] / stats['municipality_pixels']
        print(f'  {date}: {status:9s} | assigned {coverage:8.4f}% | B11 {stats["valid_b11"]:,} | SCL classified {stats["assigned_pixels"]-stats["scl_counts"][0]:,}')
    print(f'Successful: {len(results)}/{len(selected)}; failed: 0')
    print('Original Script 17 and June imagery unchanged. No private AWS operations. No upload.')


if __name__ == '__main__':
    main()
