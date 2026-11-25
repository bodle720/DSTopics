"""Anchorage Through the Seasons — first land-only NDVI/NDSI analysis.

Run from PyCharm or: python scripts/18_analyze_vegetation_and_snow.py
Optional --date YYYYMMDD uses the same input/output naming convention.
No downloads or AWS operations. Existing inputs are never modified.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from pathlib import Path
from datetime import datetime

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.shutil import copy as rio_copy
from rasterio.crs import CRS
from shapely import make_valid
from shapely.ops import unary_union
from shapely.validation import explain_validity

DATE = '20250612'
REGIONS = ('Anchorage Bowl', 'Chugiak - Eagle River', 'Girdwood - Alyeska',
           'Remaining Municipal Land')
# Exploratory, fixed thresholds; not a locally validated land-cover classifier.
NDSI_THRESHOLD = 0.40
GREEN_REFLECTANCE_MIN = 0.10
NDVI_THRESHOLD = 0.30
EPS = 0.0001
INDEX_NODATA = -9999.0
HIST_BINS = 1000  # approximate median to within a 0.002-wide index bin
SCL_NAMES = ('nodata', 'defective', 'terrain_shadow', 'cloud_shadow', 'vegetation',
             'nonvegetated', 'water', 'unclassified', 'cloud_medium', 'cloud_high',
             'cirrus', 'snow_ice')
CLASS_NAMES = {0: 'outside_study_land', 1: 'unusable_or_missing',
               2: 'not_detected_as_snow_ice', 3: 'ndsi_snow_ice_proxy'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pct(n, d):
    return round(100 * n / d, 5) if d else None


def ratio(a, b, eligible):
    """Reject negative corrected reflectance and near-zero denominators."""
    denom = a + b
    good = eligible & np.isfinite(a) & np.isfinite(b) & (a >= 0) & (b >= 0) & (denom > EPS)
    out = np.full(a.shape, INDEX_NODATA, dtype=np.float32)
    np.divide(a - b, denom, out=out, where=good)
    # Roundoff protection only; negative-input ratios have already been excluded.
    out[good] = np.clip(out[good], -1, 1)
    return out, good


def classify(dn, scl, source, land, scales, offsets):
    """Band order: blue, green, red, NIR, SWIR1; quality: SCL, source ID."""
    complete = land & (source > 0) & np.all(dn != 0, axis=0)
    quality = complete & np.isin(scl, (4, 5, 11))
    # Compute in float64 for reliable comparisons at decimal reflectance thresholds.
    refl = dn.astype(np.float64) * scales[:, None, None] + offsets[:, None, None]
    # Numerical roundoff around a physically zero reflectance is not negative signal.
    refl[np.abs(refl) < 1e-12] = 0
    green, red, nir, swir = refl[1:]
    ndsi, snow_valid = ratio(green, swir, quality)
    snow = snow_valid & (ndsi >= NDSI_THRESHOLD) & (green >= GREEN_REFLECTANCE_MIN - 1e-12)
    # SCL snow and NDSI snow are deliberately not treated as independent truth.
    # Conservative NDVI: exclude either snow indicator, plus invalid snow observations.
    snow_free = snow_valid & ~snow & (scl != 11)
    ndvi, veg_valid = ratio(nir, red, snow_free)
    veg = veg_valid & (ndvi >= NDVI_THRESHOLD)
    classes = np.zeros(land.shape, dtype=np.uint8)
    classes[land] = 1
    classes[snow_valid] = 2
    classes[snow] = 3
    return ndvi, ndsi, classes, complete, quality, snow_valid, snow, snow_free, veg_valid, veg


class Stats:
    def __init__(self):
        self.counts = {k: 0 for k in ('land', 'assigned', 'spectral_complete', 'quality_eligible',
                      'cloud', 'shadow', 'scl_water', 'ndsi_valid', 'ndsi_snow', 'scl_snow_on_ndsi_valid',
                      'snow_agreement', 'snow_disagreement', 'snow_free_candidate', 'ndvi_valid', 'green')}
        self.scl = np.zeros(12, dtype=np.int64)
        self.sums = {'ndvi': 0.0, 'ndsi': 0.0}
        self.hist = {k: np.zeros(HIST_BINS, dtype=np.int64) for k in self.sums}

    def add(self, m, scl, source, result):
        ndvi, ndsi, _, complete, quality, sv, snow, sf, vv, veg = result
        assigned = source > 0
        scl_snow = scl == 11
        # SCL histogram is over land cells with source provenance, not absent cells.
        self.scl += np.bincount(scl[m & assigned].astype(np.int64), minlength=12)
        masks = {'land': m, 'assigned': assigned, 'spectral_complete': complete,
                 'quality_eligible': quality, 'cloud': assigned & np.isin(scl, (8, 9, 10)),
                 'shadow': assigned & np.isin(scl, (2, 3)), 'scl_water': assigned & (scl == 6),
                 'ndsi_valid': sv, 'ndsi_snow': snow, 'scl_snow_on_ndsi_valid': sv & scl_snow,
                 'snow_agreement': sv & (snow == scl_snow),
                 'snow_disagreement': sv & (snow != scl_snow), 'snow_free_candidate': sf,
                 'ndvi_valid': vv, 'green': veg}
        for k, mask in masks.items():
            self.counts[k] += int(np.count_nonzero(m & mask))
        for k, arr, valid in [('ndvi', ndvi, vv), ('ndsi', ndsi, sv)]:
            vals = arr[m & valid]
            self.sums[k] += float(vals.sum(dtype=np.float64))
            self.hist[k] += np.histogram(vals, bins=HIST_BINS, range=(-1, 1))[0]

    def merge(self, other):
        for k in self.counts:
            self.counts[k] += other.counts[k]
        self.scl += other.scl
        for k in self.sums:
            self.sums[k] += other.sums[k]
            self.hist[k] += other.hist[k]

    def row(self, name, area, pixel_area, date):
        c = self.counts
        n, s, v = c['land'], c['ndsi_valid'], c['ndvi_valid']
        require(0 <= c['green'] <= v <= c['snow_free_candidate'] <= s <= c['quality_eligible'] <= c['spectral_complete'] <= c['assigned'] <= n,
                f'Inconsistent counts: {name}')
        require(c['ndsi_snow'] <= s and c['snow_agreement'] + c['snow_disagreement'] == s,
                f'Inconsistent snow counts: {name}')
        require(int(self.scl.sum()) == c['assigned'], f'Inconsistent SCL counts: {name}')
        r = {'date': date, 'region': name, 'vector_land_km2': round(area, 6),
             'raster_land_km2': round(n * pixel_area, 6)}
        r.update({k + '_pixels': val for k, val in c.items()})
        r.update({'missing_source_pixels': n - c['assigned'], 'snow_unusable_pixels': n - s,
                  'assigned_pct_of_land': pct(c['assigned'], n),
                  'cloud_pct_of_land': pct(c['cloud'], n), 'shadow_pct_of_land': pct(c['shadow'], n),
                  'quality_eligible_pct_of_land': pct(c['quality_eligible'], n),
                  'snow_usable_pct_of_land': pct(s, n),
                  'snow_ice_pct_of_snow_usable': pct(c['ndsi_snow'], s),
                  'snow_ice_observed_km2': round(c['ndsi_snow'] * pixel_area, 6),
                  'scl_snow_pct_on_same_snow_usable': pct(c['scl_snow_on_ndsi_valid'], s),
                  'ndsi_scl_agreement_pct': pct(c['snow_agreement'], s),
                  'ndvi_usable_pct_of_land': pct(v, n),
                  'ndvi_usable_pct_of_snow_free_candidate': pct(v, c['snow_free_candidate']),
                  'green_pct_of_ndvi_usable': pct(c['green'], v),
                  'green_observed_km2': round(c['green'] * pixel_area, 6)})
        for k, count in [('ndvi', v), ('ndsi', s)]:
            require(int(self.hist[k].sum()) == count, f'Histogram count mismatch: {name}/{k}')
            r[k + '_mean'] = round(self.sums[k] / count, 6) if count else None
            if count:
                cum = self.hist[k].cumsum()
                lo = np.searchsorted(cum, (count + 1) // 2)
                hi = np.searchsorted(cum, count // 2 + 1)
                r[k + '_median_approx'] = round(-1 + (float(lo + hi) / 2 + .5) * 2 / HIST_BINS, 6)
            else:
                r[k + '_median_approx'] = None
        r.update({f'scl_{i:02d}_{name}_pixels': int(self.scl[i]) for i, name in enumerate(SCL_NAMES)})
        return r


def verify_inputs(s, q):
    require(s.crs == q.crs == CRS.from_epsg(3338), 'Expected EPSG:3338 on both COGs')
    require(s.shape == q.shape and s.transform.almost_equals(q.transform, precision=1e-9), 'COG grids differ')
    t = s.transform
    require(np.allclose([t.a, t.b, t.d, t.e], [10, 0, 0, -10], atol=1e-9, rtol=0), 'Expected north-up 10 m grid')
    require(s.count == 5 and q.count == 2, 'Expected five spectral and two quality bands')
    require(all(x == 'uint16' for x in s.dtypes) and s.nodata == 0, 'Unexpected spectral dtype/NoData')
    require(all(np.issubdtype(np.dtype(x), np.integer) for x in q.dtypes), 'Expected integer quality bands')
    require(all(ds.tags(ns='IMAGE_STRUCTURE').get('LAYOUT') == 'COG' for ds in (s,q)), 'Expected verified COG inputs')
    scales, offsets = np.array(s.scales), np.array(s.offsets)
    require(np.allclose(scales, .0001, rtol=0, atol=1e-12) and np.allclose(offsets, -.1, rtol=0, atol=1e-12),
            f'Reflectance metadata differs from verified Script 17 contract: {scales}, {offsets}')
    # Guard against accidentally swapped quality bands if descriptions identify them.
    d1, d2 = ((x or '').lower() for x in q.descriptions)
    require('source' not in d1 and 'scl' not in d2 and 'classification' not in d2, 'Unexpected quality band order')
    print(f'Grid: {s.width:,} x {s.height:,}, {s.crs}, 10 m')
    print(f'Spectral bands: {s.descriptions}; scales={s.scales}; offsets={s.offsets}')
    print(f'Quality bands: {q.descriptions} (SCL, source-scene ID)')
    return scales, offsets



def polygon_parts(geom):
    """Extract polygonal components, including those in make_valid collections."""
    if geom.geom_type == 'Polygon':
        if not geom.is_empty:
            yield geom
    elif geom.geom_type in ('MultiPolygon', 'GeometryCollection'):
        for part in geom.geoms:
            yield from polygon_parts(part)


def prepare_regions(g, target_crs):
    """Validate after reprojection; repair only in memory, with area checks."""
    require(g.crs is not None and 'name' in g, 'Regions need CRS and name field')
    require(len(g) == 4 and set(g['name']) == set(REGIONS), 'Expected exactly four named study regions')
    g = g.set_index('name').loc[list(REGIONS)].copy()
    require(not g.geometry.isna().any() and not g.geometry.is_empty.any(), 'Missing or empty input region')
    source_valid = g.geometry.is_valid.to_dict()
    g = g.to_crs(target_crs)
    audit = []
    print('Checking land-region geometry before and after reprojection...', flush=True)
    for name in REGIONS:
        geom = g.geometry.loc[name]
        before = float(geom.area)
        valid = bool(geom.is_valid)
        reason = explain_validity(geom)
        require(geom.geom_type in ('Polygon', 'MultiPolygon') and not geom.is_empty and
                math.isfinite(before) and before > 0, f'Unexpected region geometry: {name}')
        fixed = geom
        if not valid:
            print(f'  {name}: {reason}; repairing in memory', flush=True)
            parts = list(polygon_parts(make_valid(geom)))
            require(bool(parts), f'No polygonal area after repair: {name}')
            fixed = unary_union(parts)
        require(fixed.is_valid and not fixed.is_empty and fixed.geom_type in ('Polygon', 'MultiPolygon'),
                f'Region still invalid after repair: {name}: {explain_validity(fixed)}')
        after = float(fixed.area)
        delta = abs(after - before)
        # Permit only small numerical/topological repairs: 100 m2 or 0.00001% of area.
        tolerance = max(100.0, before * 1e-7)
        require(math.isfinite(after) and after > 0 and delta <= tolerance,
                f'Repair changed area too much: {name}: {delta:.6f} m2 (limit {tolerance:.6f})')
        g.at[name, g.geometry.name] = fixed
        print(f'  {name}: source valid={source_valid[name]}, projected valid={valid}, '
              f'final valid=True, area={after/1e6:,.6f} km2, area change={delta:.6f} m2', flush=True)
        audit.append({'region': name, 'source_valid': bool(source_valid[name]), 'projected_valid': valid,
                      'projected_validity_reason': reason, 'repaired_in_memory': not valid,
                      'area_before_m2': before, 'area_after_m2': after, 'absolute_area_change_m2': delta,
                      'area_tolerance_m2': tolerance})
    overlaps = []
    for i in range(4):
        for j in range(i):
            overlap = float(g.geometry.iloc[i].intersection(g.geometry.iloc[j]).area)
            require(overlap <= 100.0, f'Overlapping study regions: {REGIONS[i]} / {REGIONS[j]}: {overlap:.6f} m2')
            overlaps.append({'regions': [REGIONS[i], REGIONS[j]], 'overlap_m2': overlap})
    print(f'  Geometry checks: PASS; largest pairwise overlap={max(x["overlap_m2"] for x in overlaps):.6f} m2', flush=True)
    return g, {'regions': audit, 'pairwise_overlaps': overlaps, 'pairwise_overlap_limit_m2': 100.0,
               'original_geojson_modified': False}

def make_cog(src, dst, overview, predictor):
    print(f'Creating COG: {dst.name} ...', flush=True)
    rio_copy(src, dst, driver='COG', COMPRESS='DEFLATE', LEVEL=4, BLOCKSIZE=512,
             PREDICTOR=predictor, OVERVIEW_RESAMPLING=overview, NUM_THREADS='2', BIGTIFF='IF_SAFER')
    with rasterio.open(src) as a, rasterio.open(dst) as b:
        require(b.tags(ns='IMAGE_STRUCTURE').get('LAYOUT') == 'COG', 'COG layout missing')
        require(a.shape == b.shape and a.crs == b.crs and a.transform == b.transform and
                a.count == b.count and a.nodata == b.nodata and a.descriptions == b.descriptions,
                'COG metadata changed')
        for _, win in a.block_windows(1):
            require(np.array_equal(a.read(window=win), b.read(window=win)), 'COG full-resolution pixels differ')
    print('  COG layout, metadata and full-resolution pixels: PASS', flush=True)


def run(processed, date):
    prefix = f'anchorage_{date}'
    spectral = processed / f'{prefix}_spectral_10m.tif'
    quality = processed / f'{prefix}_quality_10m.tif'
    sources = processed / f'{prefix}_sources.json'
    regions_path = processed / 'anchorage_land_study_regions.geojson'
    outputs = {k: processed / f'{prefix}_{suffix}' for k, suffix in
               [('indices', 'indices_10m.tif'), ('snow', 'snow_ice_10m.tif'),
                ('csv', 'regional_statistics.csv'), ('method', 'analysis_method.json')]}
    for path in (spectral, quality, sources, regions_path):
        require(path.is_file(), f'Missing input: {path}')
    for path in outputs.values():
        require(not path.exists(), f'Output already exists (preserved): {path}. Move prior outputs before rerunning.')
    json.loads(sources.read_text(encoding='utf-8-sig'))  # verify provenance file readable
    with tempfile.TemporaryDirectory(prefix='script18_', dir=processed) as folder:
        tmp = Path(folder)
        with rasterio.open(spectral) as s, rasterio.open(quality) as q:
            scales, offsets = verify_inputs(s, q)
            g, geometry_audit = prepare_regions(gpd.read_file(regions_path), s.crs)
            areas = g.geometry.area.to_numpy() / 1e6
            pixel_area = abs(s.transform.a * s.transform.e) / 1e6
            print('Rasterizing four permanent land polygons (pixel-center rule)...', flush=True)
            labels = rasterize([(geom, i+1) for i, geom in enumerate(g.geometry)], out_shape=s.shape,
                               transform=s.transform, fill=0, all_touched=False, dtype='uint8')
            label_counts = np.bincount(labels.ravel(), minlength=5)
            # Independently verify that the union of regions covers the same grid cells.
            union_mask = rasterize([(unary_union(list(g.geometry)), 1)], out_shape=s.shape,
                                  transform=s.transform, fill=0, all_touched=False, dtype='uint8')
            mismatch = int(np.count_nonzero((labels > 0) != (union_mask > 0)))
            del union_mask
            require(mismatch == 0, f'Region labels and union mask differ at {mismatch} pixel centers')
            geometry_audit['union_mask_mismatch_pixels'] = mismatch
            print('  Region labels versus union mask: PASS', flush=True)
            for i, name in enumerate(REGIONS):
                area = label_counts[i+1] * pixel_area
                require(area > 0 and abs(area - areas[i]) / areas[i] < .02,
                        f'Region grid area differs by >=2%: {name}: {area} vs {areas[i]} km2')
                print(f'  {name}: {label_counts[i+1]:,} land cells; {area:,.3f} km2 raster / {areas[i]:,.3f} km2 vector')
            base = dict(driver='GTiff', width=s.width, height=s.height, crs=s.crs, transform=s.transform,
                        tiled=True, blockxsize=512, blockysize=512, compress='deflate', BIGTIFF='IF_SAFER')
            index_tmp, snow_tmp = tmp/'indices_work.tif', tmp/'snow_work.tif'
            stats = [Stats() for _ in REGIONS]
            total_blocks = math.ceil(s.width / 512) * math.ceil(s.height / 512)
            step = max(1, total_blocks // 10)
            with rasterio.open(index_tmp, 'w', **base, count=2, dtype='float32', nodata=INDEX_NODATA, predictor=3) as ix, \
                 rasterio.open(snow_tmp, 'w', **base, count=1, dtype='uint8', nodata=0) as sm:
                ix.set_band_description(1, 'NDVI (conservative snow-free usable land)')
                ix.set_band_description(2, 'NDSI (usable land; SWIR native 20 m)')
                sm.set_band_description(1, 'Exploratory NDSI snow/ice classification')
                ix.update_tags(DATE=date, NDVI_THRESHOLD=str(NDVI_THRESHOLD), NDSI_THRESHOLD=str(NDSI_THRESHOLD),
                               SCL_ACCEPTED='4,5,11', NATIVE_SWIR_SCL_RESOLUTION='20 m')
                sm.update_tags(DATE=date, CLASS_NAMES=json.dumps(CLASS_NAMES), NDSI_THRESHOLD=str(NDSI_THRESHOLD),
                               MIN_GREEN_REFLECTANCE=str(GREEN_REFLECTANCE_MIN))
                sm.write_colormap(1, {0:(0,0,0,0), 1:(115,115,115,255), 2:(120,153,103,255), 3:(220,241,255,255)})
                for j, (_, win) in enumerate(ix.block_windows(1), 1):
                    rs, cs = win.toslices()
                    lab = labels[rs, cs]
                    land = lab > 0
                    if land.any():
                        dn = s.read(window=win)
                        scl, source = q.read(window=win)
                        require(not np.any(land & (source > 0) & ((scl < 0) | (scl > 11))), 'Unknown SCL value in assigned land')
                        require(not np.any(land & (source < 0)), 'Negative source ID')
                        result = classify(dn, scl, source, land, scales, offsets)
                        for i in np.unique(lab):
                            if i:
                                stats[int(i)-1].add(lab == i, scl, source, result)
                        ix.write(result[0], 1, window=win)
                        ix.write(result[1], 2, window=win)
                        sm.write(result[2], 1, window=win)
                    else:
                        ix.write(np.full((2, int(win.height), int(win.width)), INDEX_NODATA, np.float32), window=win)
                        sm.write(np.zeros(land.shape, np.uint8), 1, window=win)
                    if j % step == 0 or j == total_blocks:
                        print(f'  Analyzed {j:,}/{total_blocks:,} blocks ({100*j/total_blocks:.0f}%)', flush=True)
            total = Stats()
            for st in stats:
                total.merge(st)
            require(total.counts['land'] == int(label_counts[1:].sum()), 'Regional totals do not cover rasterized land')
            rows = [st.row(name, area, pixel_area, date) for st, name, area in zip(stats, REGIONS, areas)]
            rows.append(total.row('All Municipal Land', float(areas.sum()), pixel_area, date))
            # Check staged outputs' actual counts before translating to COGs.
            observed = np.zeros(4, dtype=np.int64)
            valid_counts = np.zeros(2, dtype=np.int64)
            with rasterio.open(index_tmp) as ix, rasterio.open(snow_tmp) as sm:
                for _, win in ix.block_windows(1):
                    x = ix.read(window=win)
                    c = sm.read(1, window=win)
                    observed += np.bincount(c.ravel(), minlength=4)
                    valid_counts += (x != INDEX_NODATA).sum(axis=(1,2))
                    require(np.array_equal(x[1] != INDEX_NODATA, c >= 2), 'NDSI and snow-map coverage mismatch')
                    require(not np.any((x[0] != INDEX_NODATA) & (c != 2)), 'NDVI includes snow or invalid cells')
            require(int(observed[1:].sum()) == total.counts['land'] and observed[3] == total.counts['ndsi_snow'] and
                    observed[1] == total.counts['land'] - total.counts['ndsi_valid'] and
                    np.array_equal(valid_counts, [total.counts['ndvi_valid'], total.counts['ndsi_valid']]),
                    'Output pixel counts do not match statistics')
            print('Regional count, histogram and raster consistency checks: PASS', flush=True)
        staged = {k: tmp / p.name for k,p in outputs.items()}
        make_cog(index_tmp, staged['indices'], 'AVERAGE', 'FLOATING_POINT')
        make_cog(snow_tmp, staged['snow'], 'NEAREST', 'NO')
        with staged['csv'].open('w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        method = {'date': date, 'inputs': [p.name for p in (spectral, quality, sources, regions_path)],
                  'crs': 'EPSG:3338', 'pixel_spacing_m': 10, 'pixel_area_km2': pixel_area,
                  'region_rasterization': 'pixel center; fixed mapped-water-excluded polygons; any boundary ties assigned to last region in fixed order',
                  'geometry_audit': geometry_audit,
                  'spectral_encoding': {'scale': scales.tolist(), 'offset': offsets.tolist(), 'nodata_dn': 0},
                  'quality_eligible': 'all five DNs nonzero, source ID positive, SCL 4/5/11',
                  'invalid_index': 'negative reflectance or denominator <= 0.0001 excluded; not zero-filled',
                  'snow_proxy': {'rule': 'NDSI >= 0.40 and green reflectance >= 0.10',
                                 'ndsi_threshold': NDSI_THRESHOLD, 'green_min': GREEN_REFLECTANCE_MIN,
                                 'denominator': 'NDSI-valid quality-eligible land', 'locally_validated': False},
                  'vegetation_proxy': {'rule': 'NDVI >= 0.30 on valid conservative snow-free land',
                                       'ndvi_threshold': NDVI_THRESHOLD,
                                       'snow_free': 'NDSI-valid, not NDSI snow proxy, SCL != 11', 'locally_validated': False},
                  'area': 'observed classified area only; no extrapolation to unobserved land',
                  'median': 'approximate histogram-based median, 0.002 bin width',
                  'snow_classes': CLASS_NAMES, 'scl_classes': dict(enumerate(SCL_NAMES)),
                  'limitations': ['SWIR and SCL native 20 m, resampled to 10 m; not independent 10 m snow measurements',
                                  'Snow and glacial ice are not separable by this rule',
                                  'Thresholds exploratory; SCL comparison not independent ground truth',
                                  'NDVI threshold is a greenness proxy, not fractional vegetation cover',
                                  'Usable land may vary by acquisition; coverage must accompany comparisons',
                                  'Source mosaic prioritized scene-level cloud metadata, not per-pixel quality'],
                  'threshold_background': 'https://modis-snow-ice.gsfc.nasa.gov/?c=atbd'}
        staged['method'].write_text(json.dumps(method, indent=2), encoding='utf-8')
        # No outputs are published until all computation and verification pass.
        for k, path in outputs.items():
            require(not path.exists(), f'Output appeared during processing: {path}')
            staged[k].rename(path)
    print('\nREGIONAL RESULTS (percentages use the denominators shown)')
    print(f'{"Region":26} {"Land km2":>10} {"Snow usable %":>14} {"Snow/ice %*":>12} {"SCL snow %*":>12} {"NDVI usable %":>14} {"Mean NDVI":>10} {"Green %**":>10}')
    def fmt(v, decimals=2):
        return 'NA' if v is None else f'{v:.{decimals}f}'
    for r in rows:
        print(f'{r["region"]:26} {fmt(r["raster_land_km2"]):>10} {fmt(r["snow_usable_pct_of_land"]):>14} '
              f'{fmt(r["snow_ice_pct_of_snow_usable"]):>12} {fmt(r["scl_snow_pct_on_same_snow_usable"]):>12} '
              f'{fmt(r["ndvi_usable_pct_of_land"]):>14} {fmt(r["ndvi_mean"],3):>10} {fmt(r["green_pct_of_ndvi_usable"]):>10}')
    print('* Of NDSI-usable land. ** Of NDVI-usable conservative snow-free land.')
    print('NDVI usable % is relative to all rasterized land, including snow-excluded areas.')
    print('Snow/ice and green estimates are exploratory proxies, not validated land-cover measurements.')
    for p in outputs.values():
        print(f'{p} ({p.stat().st_size/1e6:.2f} MB)')
    print('Complete. Existing inputs unchanged. No AWS upload performed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', default=DATE, help='Acquisition date YYYYMMDD')
    args = parser.parse_args()
    require(len(args.date) == 8, 'Date must be YYYYMMDD')
    datetime.strptime(args.date, '%Y%m%d')
    run(Path(__file__).resolve().parents[1] / 'data' / 'processed', args.date)
