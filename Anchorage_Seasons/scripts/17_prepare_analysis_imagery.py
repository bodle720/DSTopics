"""Build aligned Sentinel-2 spectral and quality COGs for 2025-06-12.

Run from the Anchorage_Seasons project with its existing virtual environment.
Source imagery is public Earth Search data; no AWS upload is performed.
The original four-band COG is only used as a grid reference.
"""
from contextlib import ExitStack
from pathlib import Path
import json
import shutil
import tempfile

import geopandas as gpd
import numpy as np
import rasterio as rio
from pystac_client import Client
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
DATE = '2025-06-12'
API = 'https://earth-search.aws.element84.com/v1'
COLLECTION = 'sentinel-2-c1-l2a'
# Fixed priority from the original mosaic's ascending scene cloud percentages.
TILES = ['05VPH', '06VUN', '06VVN', '05VPJ', '06VUP', '06VVP']
IDS = [f'S2C_T{tile}_20250612T212656_L2A' for tile in TILES]
KEYS = ['blue', 'green', 'red', 'nir', 'swir16']
NAMES = ['B02 Blue', 'B03 Green', 'B04 Red', 'B08 NIR', 'B11 SWIR']
SPECTRAL = DATA / 'anchorage_20250612_spectral_10m.tif'
QUALITY = DATA / 'anchorage_20250612_quality_10m.tif'
MANIFEST = DATA / 'anchorage_20250612_sources.json'


def read_items():
    client = Client.open(API)
    # Exact IDs, not a fresh cloud/date search which could change the inputs.
    items = {i.id: i for i in client.search(collections=[COLLECTION], ids=IDS).items()}
    missing = set(IDS) - set(items)
    if missing:
        raise RuntimeError(f'Missing pinned STAC items: {sorted(missing)}')
    records = []
    for scene_id in IDS:
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
    scales, offsets = [], []
    for key in KEYS:
        pairs = {(float(r['assets'][key]['raster:bands']['scale']),
                  float(r['assets'][key]['raster:bands']['offset'])) for r in records}
        if len(pairs) != 1:
            raise RuntimeError(f'Inconsistent source encodings for {key}: {pairs}')
        scale, offset = pairs.pop()
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


def build(records, scales, offsets, grid, geometry, spectral_stage, quality_stage):
    common = dict(driver='GTiff', **grid, tiled=True, blockxsize=512, blockysize=512,
                  compress='DEFLATE', nodata=0, BIGTIFF='IF_SAFER')
    vrt_grid = {k: grid[k] for k in ('crs', 'transform', 'width', 'height')}
    with rio.open(spectral_stage, 'w+', count=5, dtype='uint16', **common) as out, \
         rio.open(quality_stage, 'w+', count=2, dtype='uint8', **common) as qout:
        out.descriptions = tuple(NAMES)
        out.scales = tuple(scales)
        out.offsets = tuple(offsets)
        out.update_tags(acquisition_date=DATE, source_collection=COLLECTION,
                        B11_native_resolution_m=20, resampling='nearest')
        qout.descriptions = ('SCL (0-11)', 'Source scene code (1-6)')
        qout.update_tags(acquisition_date=DATE, SCL_native_resolution_m=20,
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


def verify(spectral, quality, reference_grid, scales, offsets, total):
    counts = np.zeros(12, dtype=np.int64)
    scenes = np.zeros(len(IDS)+1, dtype=np.int64)
    b11_valid = 0
    with rio.open(spectral) as s, rio.open(quality) as q:
        for src, bands in [(s, 5), (q, 2)]:
            assert src.count == bands
            assert src.crs == reference_grid['crs'] and src.transform == reference_grid['transform']
            assert (src.width, src.height) == (reference_grid['width'], reference_grid['height'])
            assert src.tags(ns='IMAGE_STRUCTURE').get('LAYOUT') == 'COG'
        assert np.allclose(s.scales, scales) and np.allclose(s.offsets, offsets)
        for _, w in s.block_windows(1):
            a, b = s.read(window=w), q.read(window=w)
            assigned = b[1] > 0
            if np.any(b[1] > len(IDS)) or np.any(b[0] > 11):
                raise RuntimeError('Invalid quality values')
            if not np.array_equal(assigned, np.all(a[:4] != 0, axis=0)):
                raise RuntimeError('Source-scene map disagrees with valid reflectance')
            if np.any(a[:, ~assigned] != 0) or np.any(b[0, ~assigned] != 0):
                raise RuntimeError('Unassigned pixel contains data')
            counts += np.bincount(b[0, assigned], minlength=12)
            scenes += np.bincount(b[1, assigned], minlength=len(IDS)+1)
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


def main():
    if any(p.exists() for p in (SPECTRAL, QUALITY, MANIFEST)):
        raise FileExistsError('Analysis outputs already exist. Move them aside before deliberately rebuilding.')
    with rio.open(REFERENCE) as ref:
        grid = dict(crs=ref.crs, transform=ref.transform, width=ref.width, height=ref.height)
        if ref.crs.to_epsg() != 3338 or not np.allclose(ref.res, (10, 10)):
            raise RuntimeError('Reference must be EPSG:3338, 10 m')
    if shutil.disk_usage(DATA).free < 4 * 1024**3:
        raise RuntimeError('Please leave at least 4 GiB of free disk space for staging and COG creation.')
    geometry = list(gpd.read_file(BOUNDARY).to_crs(grid['crs']).geometry)
    records, scales, offsets = read_items()
    # Public HTTPS reads; do not use private bucket credentials or upload anything.
    with tempfile.TemporaryDirectory(prefix='s2_analysis_', dir=DATA) as td:
        tmp = Path(td)
        with rio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', GDAL_HTTP_MAX_RETRY='5',
                     GDAL_HTTP_RETRY_DELAY='3', GDAL_CACHEMAX=128*1024**2):
            total = build(records, scales, offsets, grid, geometry, tmp/'spectral.tif', tmp/'quality.tif')
            print('\nCreating cloud-optimized GeoTIFFs...', flush=True)
            for name, resampling in [('spectral', 'AVERAGE'), ('quality', 'NEAREST')]:
                rio_copy(tmp/f'{name}.tif', tmp/f'{name}_cog.tif', driver='COG',
                         COMPRESS='DEFLATE', BLOCKSIZE=512, BIGTIFF='IF_SAFER',
                         OVERVIEW_RESAMPLING=resampling)
            stats = verify(tmp/'spectral_cog.tif', tmp/'quality_cog.tif', grid, scales, offsets, total)
        manifest = dict(date=DATE, collection=COLLECTION, source_scene_priority=records,
                        spectral_band_order=NAMES, scale=scales, offset=offsets,
                        source_selection='first valid B02/B03/B04/B08, fixed scene priority; no cross-scene filling',
                        resampling='nearest for all full-resolution bands; B11 and SCL native 20 m',
                        quality_band_order=['SCL', 'source scene code'], statistics=stats)
        (tmp/'sources.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        (tmp/'spectral_cog.tif').replace(SPECTRAL)
        (tmp/'quality_cog.tif').replace(QUALITY)
        (tmp/'sources.json').replace(MANIFEST)
    for p in (SPECTRAL, QUALITY, MANIFEST):
        print(f'{p} ({p.stat().st_size / 1e6:.2f} MB)')
    print('Complete. Original COG unchanged. No AWS upload performed.')


if __name__ == '__main__':
    main()
