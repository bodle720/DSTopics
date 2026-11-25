"""Batch the verified Script 18 scientific analysis over eight 2025 acquisitions.

Keep 18_analyze_vegetation_and_snow.py (corrected version) beside this file.
Run normally in PyCharm or: python scripts/26_analyze_seasonal_imagery.py
Only local inputs are read. No downloads, AWS operations or input modifications.
Completed dates are checked against their actual rasters and reused. Partial or
inconsistent output sets are preserved and cause a stop, not a silent overwrite.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path
import shutil
import tempfile

import geopandas as gpd
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / 'data' / 'processed'
DATES = ('20250207', '20250329', '20250523', '20250612', '20250727',
         '20250811', '20250930', '20251119')
GROUPS = dict(zip(DATES, ('S2B_20250207T212545', 'S2B_20250329T212803',
    'S2C_20250523T212539', 'S2C_20250612T212656', 'S2B_20250727T212521',
    'S2C_20250811T212539', 'S2C_20250930T212536', 'S2C_20251119T212811')))
TILES = ('05VPH', '05VPJ', '06VUN', '06VUP', '06VVN', '06VVP')
REGIONS_FILE = 'anchorage_land_study_regions.geojson'
COMBINED = 'anchorage_2025_seasonal_regional_statistics.csv'
SUFFIXES = ('indices_10m.tif', 'snow_ice_10m.tif', 'regional_statistics.csv', 'analysis_method.json')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_analysis(path):
    require(path.is_file(), f'Missing corrected Script 18: {path}')
    spec = importlib.util.spec_from_file_location('anchorage_analysis18', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require((module.NDSI_THRESHOLD, module.GREEN_REFLECTANCE_MIN, module.NDVI_THRESHOLD,
             module.EPS, module.INDEX_NODATA, module.HIST_BINS) == (.4, .1, .3, .0001, -9999., 1000),
            'Script 18 threshold contract changed')
    return module


def paths_for(data, date):
    return [data / f'anchorage_{date}_{suffix}' for suffix in SUFFIXES]


def grid_of(ds):
    return ds.crs, ds.transform, ds.width, ds.height


def validate_source(data, date, grid, a):
    prefix = data / f'anchorage_{date}'
    s_path = Path(str(prefix) + '_spectral_10m.tif')
    q_path = Path(str(prefix) + '_quality_10m.tif')
    m_path = Path(str(prefix) + '_sources.json')
    m = json.loads(m_path.read_text(encoding='utf-8-sig'))
    satellite, stamp = GROUPS[date].split('_', 1)
    ids = {f'{satellite}_T{tile}_{stamp}_L2A' for tile in TILES}
    records = m['source_scene_priority']
    require(m['date'].replace('-', '') == date and m['collection'] == 'sentinel-2-c1-l2a', f'Source manifest date/collection mismatch: {date}')
    require(len(records) == 6 and {r['id'] for r in records} == ids, f'Source scene inventory mismatch: {date}')
    with rasterio.open(s_path) as s, rasterio.open(q_path) as q:
        require(grid_of(s) == grid_of(q) == grid, f'Input grid mismatch: {date}')
        scales, offsets = a.verify_inputs(s, q)
        require(np.array_equal(scales, np.asarray(m['scale'])) and np.array_equal(offsets, np.asarray(m['offset'])),
                f'Manifest radiometry mismatch: {date}')
        require(s.tags().get('acquisition_date') == q.tags().get('acquisition_date') == m['date'],
                f'Input acquisition tags mismatch: {date}')
        require(all(q.tags().get(f'scene_{i}') == r['id'] for i,r in enumerate(records, 1)),
                f'Quality scene code tags mismatch: {date}')
    return s_path, q_path


def prepare_labels(data, grid, a):
    crs, transform, width, height = grid
    g, _ = a.prepare_regions(gpd.read_file(data / REGIONS_FILE), crs)
    areas = g.geometry.area.to_numpy() / 1e6
    labels = a.rasterize([(geom, i+1) for i, geom in enumerate(g.geometry)], out_shape=(height, width),
                         transform=transform, fill=0, all_touched=False, dtype='uint8')
    union = a.rasterize([(a.unary_union(list(g.geometry)), 1)], out_shape=labels.shape,
                        transform=transform, fill=0, all_touched=False, dtype='uint8')
    require(np.array_equal(labels > 0, union > 0), 'Fixed region mask differs from union')
    pixel_area = abs(transform.a * transform.e) / 1e6
    counts = np.bincount(labels.ravel(), minlength=5)[1:]
    require(np.all(counts > 0) and np.all(abs(counts * pixel_area - areas) / areas < .02), 'Region area checks failed')
    print('Fixed study-region grid and areas: PASS', flush=True)
    return labels, areas, pixel_area


def check_method(path, date, a):
    m = json.loads(path.read_text(encoding='utf-8-sig'))
    expected = {'date': date, 'crs': 'EPSG:3338', 'pixel_spacing_m': 10,
                'quality_eligible': 'all five DNs nonzero, source ID positive, SCL 4/5/11',
                'invalid_index': 'negative reflectance or denominator <= 0.0001 excluded; not zero-filled'}
    for k,v in expected.items():
        require(m.get(k) == v, f'{date}: method metadata mismatch: {k}')
    require(m['snow_proxy']['ndsi_threshold'] == a.NDSI_THRESHOLD and m['snow_proxy']['green_min'] == a.GREEN_REFLECTANCE_MIN
            and m['vegetation_proxy']['ndvi_threshold'] == a.NDVI_THRESHOLD
            and m['vegetation_proxy']['snow_free'] == 'NDSI-valid, not NDSI snow proxy, SCL != 11', f'{date}: method rules mismatch')


def check_csv(path, expected):
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames == list(expected[0]), f'Unexpected CSV schema: {path}')
        actual = list(reader)
    require(len(actual) == len(expected), f'Unexpected CSV row count: {path}')
    for got, want in zip(actual, expected):
        for k,v in want.items():
            x = got[k]
            if v is None:
                ok = x == ''
            elif isinstance(v, str):
                ok = x == v
            elif isinstance(v, int):
                ok = x == str(v)
            else:
                try:
                    ok = math.isclose(float(x), float(v), rel_tol=0, abs_tol=1.1e-6)
                except (ValueError, TypeError):
                    ok = False
            require(ok, f'{path.name}: {want["region"]}/{k}: CSV={x!r}, recomputed={v!r}')
    return actual


def audit_date(data, date, grid, fixed, a):
    """Recompute with the same scientific kernel, compare every pixel and CSV field."""
    labels, areas, pixel_area = fixed
    ix_path, snow_path, csv_path, method_path = paths_for(data, date)
    check_method(method_path, date, a)
    s_path, q_path = validate_source(data, date, grid, a)
    stats = [a.Stats() for _ in a.REGIONS]
    with rasterio.open(s_path) as s, rasterio.open(q_path) as q, \
         rasterio.open(ix_path) as ix, rasterio.open(snow_path) as sm:
        for ds, count, dtype, nodata in ((ix,2,'float32',a.INDEX_NODATA), (sm,1,'uint8',0)):
            require(grid_of(ds) == grid and ds.count == count and all(d == dtype for d in ds.dtypes)
                    and ds.nodata == nodata and ds.tags(ns='IMAGE_STRUCTURE').get('LAYOUT') == 'COG'
                    and ds.tags().get('DATE') == date, f'{date}: incorrect output raster metadata: {ds.name}')
        require(ix.tags().get('SCL_ACCEPTED') == '4,5,11' and float(ix.tags()['NDVI_THRESHOLD']) == a.NDVI_THRESHOLD
                and float(ix.tags()['NDSI_THRESHOLD']) == a.NDSI_THRESHOLD
                and float(sm.tags()['NDSI_THRESHOLD']) == a.NDSI_THRESHOLD
                and float(sm.tags()['MIN_GREEN_REFLECTANCE']) == a.GREEN_REFLECTANCE_MIN,
                f'{date}: output threshold metadata mismatch')
        scales, offsets = np.array(s.scales), np.array(s.offsets)
        nblocks = math.ceil(ix.width / 512) * math.ceil(ix.height / 512)
        for j, (_, win) in enumerate(ix.block_windows(1), 1):
            rs,cs = win.toslices(); lab = labels[rs,cs]; land = lab > 0
            x = ix.read(window=win); c = sm.read(1, window=win)
            if land.any():
                scl, source = q.read(window=win)
                require(not np.any(land & ((source > 6) | (source < 0) | ((source > 0) & ((scl < 0) | (scl > 11))))), f'{date}: invalid source or SCL code')
                result = a.classify(s.read(window=win), scl, source, land, scales, offsets)
                require(np.array_equal(x[0], result[0]) and np.array_equal(x[1], result[1]) and np.array_equal(c, result[2]), f'{date}: output pixels differ at {win}')
                for i in np.unique(lab):
                    if i:
                        stats[int(i)-1].add(lab == i, scl, source, result)
            else:
                require(np.all(x == a.INDEX_NODATA) and np.all(c == 0), f'{date}: data outside study land')
            if j % max(1,nblocks//4) == 0 or j == nblocks:
                print(f'  Audited {j}/{nblocks} blocks', flush=True)
    total = a.Stats()
    for st in stats:
        total.merge(st)
    rows = [st.row(name, area, pixel_area, date) for st,name,area in zip(stats,a.REGIONS,areas)]
    rows.append(total.row('All Municipal Land', float(areas.sum()), pixel_area, date))
    actual = check_csv(csv_path, rows)
    print(f'  {date}: full-resolution scientific pixel and regional CSV checks: PASS', flush=True)
    return actual


def write_combined(data, rows):
    dest = data / COMBINED
    with tempfile.TemporaryDirectory(prefix='seasonal_summary_', dir=data) as folder:
        stage = Path(folder) / dest.name
        with stage.open('w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        if dest.exists():
            require(dest.read_bytes() == stage.read_bytes(), f'Existing summary differs: {dest}. Preserved; inspect before replacing.')
            print(f'Existing combined CSV verified: {dest}')
        else:
            stage.rename(dest)
            print(f'Created: {dest}')


def main():
    a = load_analysis(Path(__file__).with_name('18_analyze_vegetation_and_snow.py'))
    with rasterio.open(PROCESSED / 'anchorage_s2_cog_10m.tif') as ref:
        grid = grid_of(ref)
        require(ref.crs.to_epsg() == 3338 and ref.res == (10,10) and (ref.width,ref.height) == (10739,8299), 'Unexpected reference grid')
    # Preflight all inputs/output states before any new science outputs are written.
    for date in DATES:
        validate_source(PROCESSED, date, grid, a)
        states = [p.exists() for p in paths_for(PROCESSED,date)]
        require(all(states) or not any(states), f'{date}: partial output set; preserved. Inspect/move those outputs before restarting.')
        require(date != '20250612' or all(states), 'Verified June outputs must be present; do not regenerate them')
    fixed = prepare_labels(PROCESSED, grid, a)
    collected = {}
    # Verify June first to establish compatibility with the original run.
    for date in ('20250612',) + tuple(d for d in DATES if d != '20250612'):
        print(f'\n=== {date} ===', flush=True)
        paths = paths_for(PROCESSED,date)
        if all(p.is_file() for p in paths):
            status = 'reused'
        else:
            require(not any(p.exists() for p in paths), f'{date}: partial output set; preserved')
            require(shutil.disk_usage(PROCESSED).free >= 3*1024**3, 'Leave at least 3 GiB free for staging this date')
            a.run(PROCESSED,date)
            status = 'created'
        collected[date] = (status, audit_date(PROCESSED,date,grid,fixed,a))
    rows = [r for d in DATES for r in collected[d][1]]
    require(len(rows) == 40 and len({(r['date'],r['region']) for r in rows}) == 40, 'Incomplete seasonal table')
    write_combined(PROCESSED,rows)
    print('\nSEASONAL RESULTS — All Municipal Land (coverage and conditional estimates)')
    print('Date       Status   Snow usable%  Snow/ice%*  NDVI usable%  Mean NDVI  Green%**')
    for d in DATES:
        status, rr = collected[d]; r = rr[-1]
        def fmt(k, decimals=2):
            return f'{float(r[k]):.{decimals}f}' if r[k] else 'NA'
        print(f'{d}   {status:7} {fmt("snow_usable_pct_of_land"):>12} {fmt("snow_ice_pct_of_snow_usable"):>11} {fmt("ndvi_usable_pct_of_land"):>13} {fmt("ndvi_mean",3):>10} {fmt("green_pct_of_ndvi_usable"):>9}')
    print('* Of NDSI-usable land. ** Of conservative snow-free NDVI-usable land.')
    print('NDVI usable% is of ALL land; snow exclusion is not necessarily bad observation quality.')
    print('Missing estimates stay blank; no extrapolation or coverage-based suppression applied.')
    print('Successful: 8/8; regional rows: 40/40. Original inputs unchanged. No AWS operations.')


if __name__ == '__main__':
    main()
