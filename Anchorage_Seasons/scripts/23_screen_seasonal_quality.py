"""Screen Anchorage 2025 candidates using SCL only, on a fixed 20 m grid.

Run normally in PyCharm. Reads Script 22 shortlist and existing June quality COG.
Public STAC/SCL reads only; no new full-resolution imagery or private AWS operations.
Successful acquisition summaries are cached for safe resume after interruptions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.vrt import WarpedVRT
from shapely import make_valid
from shapely.ops import unary_union
from shapely.validation import explain_validity

YEAR = 2025
REFERENCE = 'S2C_20250612T212656'
REGIONS = ('Anchorage Bowl', 'Chugiak - Eagle River', 'Girdwood - Alyeska', 'Remaining Municipal Land')
ALL = 'All Municipal Land'
TARGET = 90.0
ACCEPTED = (4, 5, 11)
NAMES = ('nodata', 'defective', 'terrain_shadow', 'cloud_shadow', 'vegetation', 'nonvegetated',
         'water', 'unclassified', 'cloud_medium', 'cloud_high', 'cirrus', 'snow_ice')
VERSION = '23-v1'
STAC_URL = 'https://earth-search.aws.element84.com/v1'
COLLECTION = 'sentinel-2-c1-l2a'


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def polygon_parts(g):
    if g.geom_type == 'Polygon' and not g.is_empty:
        yield g
    elif g.geom_type in ('MultiPolygon', 'GeometryCollection'):
        for p in g.geoms:
            yield from polygon_parts(p)


def repair(g, name):
    require(g is not None and not g.is_empty and g.geom_type in ('Polygon', 'MultiPolygon'), f'Invalid input: {name}')
    area = g.area
    if not g.is_valid:
        print(f'  {name}: {explain_validity(g)}; repairing in memory', flush=True)
        g = unary_union(list(polygon_parts(make_valid(g))))
    require(g.is_valid and not g.is_empty and area > 0 and abs(g.area-area) <= max(100., area*1e-7), f'Material geometry repair: {name}')
    return g


def load_grid(processed):
    reference = processed / 'anchorage_20250612_quality_10m.tif'
    with rasterio.open(reference) as ds:
        require(ds.crs == rasterio.crs.CRS.from_epsg(3338) and ds.count == 2 and
                np.allclose([ds.transform.a, ds.transform.b, ds.transform.d, ds.transform.e], [10,0,0,-10], rtol=0, atol=1e-9),
                'Unexpected June reference grid')
        require('scl' in (ds.descriptions[0] or '').lower() and 'source' in (ds.descriptions[1] or '').lower(), 'Unexpected reference band order')
        grid = dict(crs=ds.crs, transform=ds.transform * Affine.scale(2),
                    width=math.ceil(ds.width/2), height=math.ceil(ds.height/2))
    g = gpd.read_file(processed / 'anchorage_land_study_regions.geojson')
    require(g.crs is not None and len(g)==4 and set(g['name'])==set(REGIONS), 'Expected four permanent study regions')
    g = g.set_index('name').loc[list(REGIONS)].to_crs(grid['crs'])
    geoms = [repair(g.geometry.loc[n], n) for n in REGIONS]
    for i in range(4):
        for j in range(i):
            require(geoms[i].intersection(geoms[j]).area <= 100., 'Region overlap exceeds tolerance')
    shape = (grid['height'], grid['width'])
    labels = rasterize([(p,i+1) for i,p in enumerate(geoms)], out_shape=shape, transform=grid['transform'], dtype='uint8', all_touched=False)
    union = rasterize([(unary_union(geoms),1)], out_shape=shape, transform=grid['transform'], dtype='uint8', all_touched=False)
    require(np.array_equal(labels>0,union>0),'Region labels differ from land union')
    for i,(n,p) in enumerate(zip(REGIONS,geoms),1):
        area = int(np.count_nonzero(labels==i)) * 400
        require(area > 0 and abs(area-p.area)/p.area < .02, f'20 m raster area differs by >=2%: {n}')
        print(f'  {n}: {area/1e6:,.3f} km2 on screening grid',flush=True)
    print(f'Grid: {grid["width"]:,} x {grid["height"]:,}, EPSG:3338, 20 m; geometry checks PASS',flush=True)
    return grid, labels, reference


def read_scl(path, grid, local=False):
    with rasterio.open(path) as ds:
        require(ds.crs is not None and ds.count == (2 if local else 1), f'Unexpected SCL dataset: {path}')
        require(np.issubdtype(np.dtype(ds.dtypes[0]), np.integer), 'SCL must be categorical integer data')
        require(ds.nodata in (None,0), f'Unexpected SCL NoData value: {ds.nodata}')
        if not local:
            require(np.allclose([abs(ds.transform.a), abs(ds.transform.e), ds.transform.b, ds.transform.d], [20,20,0,0], rtol=0,atol=1e-6), 'Unexpected native SCL grid')
        with WarpedVRT(ds, **grid, src_nodata=0, nodata=0, resampling=Resampling.nearest) as vrt:
            scl = vrt.read(1)
            if local:
                source = vrt.read(2)
                scl[source==0] = 0
        require(not np.any((scl<0)|(scl>11)), f'Unexpected SCL class in {path}')
        return scl.astype(np.uint8,copy=False)


def fill(mosaic, scene, labels):
    take = (labels>0)&(mosaic==0)&(scene!=0)
    mosaic[take] = scene[take]
    return int(take.sum())


def pct(n,d):
    return 100*n/d if d else None


def summarize(scl, labels):
    require(scl.shape == labels.shape and not np.any(scl>11), 'Invalid mosaic')
    hist = np.bincount(labels.ravel().astype(np.int16)*12+scl.ravel(),minlength=60).reshape(5,12)[1:]
    require(int(hist.sum()) == int(np.count_nonzero(labels)), 'Histogram count mismatch')
    rows=[]
    for name, h in zip((*REGIONS,ALL),[*hist,hist.sum(axis=0)]):
        h = [int(v) for v in h]
        n = sum(h)
        good = sum(h[i] for i in ACCEPTED)
        require(n>0 and good<=n, 'Invalid region counts')
        row = dict(region=name, land_pixels=n, scl_nonzero_pixels=n-h[0], scl_eligible_pixels=good,
                   scl_eligible_pct_of_land=pct(good,n), scl_missing_pct_of_land=pct(h[0],n),
                   cloud_pct_of_land=pct(sum(h[8:11]),n), shadow_pct_of_land=pct(h[2]+h[3],n),
                   water_pct_of_land=pct(h[6],n), scl_snow_pct_of_eligible=pct(h[11],good))
        row.update({f'scl_{i:02d}_{k}_pixels':h[i] for i,k in enumerate(NAMES)})
        rows.append(row)
    return rows


def atomic_json(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def screen(candidate, grid, labels, reference, client):
    group=candidate['acquisition']
    scenes=candidate['scenes']
    ids=[s['id'] for s in scenes]
    require(ids and len(set(ids))==len(ids), f'Empty or duplicate scene IDs: {group}')
    sat,stamp=group.split('_')
    require(all(x.startswith(sat+'_') and '_'+stamp+'_L2A' in x for x in ids), 'Scene/acquisition mismatch')
    details=[]
    if group==REFERENCE:
        mosaic=read_scl(reference,grid,local=True)
        method='local verified June quality COG sampled to 20 m'
    else:
        found={it.id:it for it in client.search(collections=[COLLECTION],ids=ids).items()}
        require(set(found)==set(ids), f'Missing exact STAC scene IDs: {sorted(set(ids)-set(found))}')
        mosaic=np.zeros(labels.shape,dtype=np.uint8)
        method='public native SCL, nearest neighbor, first nonzero in Script 22 priority order'
        with rasterio.Env(AWS_NO_SIGN_REQUEST='YES',GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
                          GDAL_HTTP_MAX_RETRY='3',GDAL_HTTP_RETRY_DELAY='2',GDAL_HTTP_TIMEOUT='90'):
            for j,scene_id in enumerate(ids,1):
                print(f'    Scene {j}/{len(ids)}: {scene_id}',flush=True)
                item=found[scene_id]
                require('scl' in item.assets and item.assets['scl'].href, f'Missing SCL: {scene_id}')
                scene=read_scl(item.assets['scl'].href,grid)
                assigned=fill(mosaic,scene,labels)
                details.append({'id':scene_id,'new_nonzero_land_pixels':assigned})
                print(f'      Newly classified land cells: {assigned:,}',flush=True)
    rows=summarize(mosaic,labels)
    if group!=REFERENCE:
        require(sum(x['new_nonzero_land_pixels'] for x in details)==rows[-1]['scl_nonzero_pixels'],'Scene assignment count mismatch')
    return {'acquisition':group,'date':candidate['date'],'status':'ok','method':method,
            'scene_ids':ids,'scene_assignments':details,'regions':rows,
            'cloud_proxy_pct':candidate['land_weighted_scene_cloud_proxy_pct']}


def metrics(result):
    r=result['regions']
    return min(x['scl_eligible_pct_of_land'] for x in r[:-1]),r[-1]['scl_eligible_pct_of_land']


def rank(result):
    worst,overall=metrics(result)
    return (worst<TARGET,-worst,-overall,result['date'],result['acquisition'])


def select_months(results):
    out=[]
    for m in range(1,13):
        group=[r for r in results if r['date'].startswith(f'{YEAR}-{m:02d}')]
        good=sorted([r for r in group if r['status']=='ok'],key=rank)
        errors=[r['acquisition'] for r in group if r['status']!='ok']
        chosen=next((r for r in good if r['acquisition']==REFERENCE),None) or (good[0] if good else None)
        status='no shortlisted acquisition' if not group else 'no successful screen'
        record={'month':f'{YEAR}-{m:02d}','status':status,'failed_acquisitions':errors,
                'candidate_count':len(group),'successful_count':len(good),'recommendation':None}
        if chosen:
            worst,overall=metrics(chosen)
            status='June verified reference' if chosen['acquisition']==REFERENCE else ('meets provisional SCL target' if worst>=TARGET else 'review: below provisional target')
            if errors: status+='; incomplete comparison'
            record.update(status=status,recommendation={'acquisition':chosen['acquisition'],'date':chosen['date'],
                'min_region_eligible_pct':worst,'all_land_eligible_pct':overall,'scene_ids':chosen['scene_ids']})
        out.append(record)
    return out


def run(processed):
    shortlist_path=processed/f'anchorage_{YEAR}_acquisition_shortlist.json'
    payload=json.loads(shortlist_path.read_text(encoding='utf-8-sig'))
    require(payload['year']==YEAR and payload['stac_url']==STAC_URL and payload['collection']==COLLECTION,'Unexpected shortlist source')
    candidates=list(payload['shortlist'])
    if not any(x['acquisition']==REFERENCE for x in candidates):
        ref=payload.get('verified_june_reference')
        require(ref and ref['acquisition']==REFERENCE,'Missing verified June reference in shortlist metadata')
        candidates.append(ref)
    require(len({x['acquisition'] for x in candidates})==len(candidates),'Duplicate acquisition groups')
    grid,labels,reference=load_grid(processed)
    cache=processed/'seasonal_scl_screen_cache'; cache.mkdir(exist_ok=True)
    # Fingerprint algorithm, input geometry, shortlist and reference file identity.
    digest=hashlib.sha256(Path(__file__).read_bytes())
    for p in (shortlist_path,processed/'anchorage_land_study_regions.geojson'):
        digest.update(p.read_bytes())
    digest.update(str((reference.stat().st_size,reference.stat().st_mtime_ns)).encode())
    signature=digest.hexdigest()
    client=None
    results=[]
    for i,c in enumerate(candidates,1):
        group=c['acquisition']
        require(group.replace('_','').isalnum(), 'Invalid acquisition identifier')
        print(f'\n[{i}/{len(candidates)}] {group}',flush=True)
        path=cache/f'{group}.json'
        try:
            cached=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        except (OSError,ValueError):
            cached={}
        if cached.get('signature')==signature and cached.get('result',{}).get('status')=='ok':
            result=cached['result']
            print('  Reusing completed local screening result',flush=True)
        else:
            try:
                if group!=REFERENCE and client is None:
                    from pystac_client import Client
                    client=Client.open(STAC_URL)
                result=screen(c,grid,labels,reference,client)
                atomic_json(path,{'signature':signature,'result':result})
            except Exception as exc:
                result={'acquisition':group,'date':c['date'],'status':'error','error':f'{type(exc).__name__}: {exc}'}
                print(f'  SCREEN FAILED (not a quality observation): {result["error"]}',flush=True)
        results.append(result)
        if result['status']=='ok':
            for r in result['regions']:
                print(f'  {r["region"]:26s} eligible {r["scl_eligible_pct_of_land"]:6.2f}% | cloud {r["cloud_pct_of_land"]:6.2f}% | shadow {r["shadow_pct_of_land"]:6.2f}%',flush=True)
    months=select_months(results)
    rows=[dict(acquisition=r['acquisition'],date=r['date'],method=r['method'],**x) for r in results if r['status']=='ok' for x in r['regions']]
    if rows:
        out=processed/f'anchorage_{YEAR}_scl_quality_screen.csv'
        tmp=out.with_suffix('.csv.tmp')
        with tmp.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        tmp.replace(out)
        print(f'Created: {out}')
    out=processed/f'anchorage_{YEAR}_scl_quality_selection.json'
    atomic_json(out,{'year':YEAR,'version':VERSION,'complete':all(r['status']=='ok' for r in results),
        'target_pct_each_region':TARGET,'eligible_scl_classes':list(ACCEPTED),
        'grid':{'crs':'EPSG:3338','pixel_size_m':20,'width':grid['width'],'height':grid['height'],'transform':list(grid['transform'])[:6]},
        'limitations':'SCL-only preliminary screening; not final spectral validity. Nonzero SCL first-fill may differ from complete spectral mosaic. Snow accepted; low illumination, cloud errors and reflectance quality still require review. Dates below target require review; no missing observations imputed.',
        'months':months,'results':results})
    print('\nMONTHLY RECOMMENDATIONS (preliminary SCL screening, not final acceptance)')
    for m in months:
        r=m['recommendation']
        print(f'{m["month"]}: '+(f'{r["date"]} | worst region {r["min_region_eligible_pct"]:.2f}% | all land {r["all_land_eligible_pct"]:.2f}% | ' if r else '')+m['status'])
    failures=sum(r['status']!='ok' for r in results)
    print(f'\nSuccessful: {len(results)-failures}/{len(results)}; failed: {failures}')
    print(f'Created: {out}')
    print('Existing imagery and polygons unchanged. No private AWS operations.')
    if failures:
        print('Partial results only. Rerun to retry failed candidates; successful screens will be reused.')
        raise SystemExit(2)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run(Path(__file__).resolve().parents[1]/'data'/'processed')
