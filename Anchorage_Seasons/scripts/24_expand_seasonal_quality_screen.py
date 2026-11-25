"""Expand Script 23's SCL screen to every Script 22 footprint-qualified acquisition.

Place beside the unchanged 23_screen_seasonal_quality.py and run in PyCharm.
Uses the full acquisition inventory already saved by Script 22. No new discovery
query, spectral downloads, or private AWS operations. Original outputs retained.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_screen_module(path):
    require(path.is_file(), f'Missing Script 23: {path}')
    spec = importlib.util.spec_from_file_location('anchorage_screen23', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.VERSION == '23-v1' and module.YEAR == 2025,
            'Unexpected Script 23 version; keep the original completed script.')
    return module


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def inventory(payload, s):
    require(payload['year'] == s.YEAR and payload['stac_url'] == s.STAC_URL
            and payload['collection'] == s.COLLECTION, 'Unexpected Script 22 inventory source')
    gate = payload['footprint_gate_pct']
    require(gate == 99.9, 'Unexpected footprint threshold')
    rows = payload['all_acquisitions']
    require(len(rows) == payload['acquisition_count'], 'Inventory count mismatch')
    by_id = {r['acquisition']: r for r in rows}
    require(len(by_id) == len(rows), 'Duplicate acquisition IDs')
    for r in rows:
        require(r['passes_footprint_gate'] == (r['municipal_footprint_pct'] >= gate
                and r['land_footprint_pct'] >= gate), 'Footprint flag mismatch')
    eligible = sorted((r for r in rows if r['passes_footprint_gate']),
                      key=lambda r: (r['date'], r['acquisition']))
    original = {r['acquisition']: r for r in payload['shortlist']}
    if s.REFERENCE not in original:
        original[s.REFERENCE] = payload['verified_june_reference']
    require(all(by_id.get(k) == r and r['passes_footprint_gate'] for k,r in original.items()),
            'Original shortlist differs from full inventory')
    require(s.REFERENCE in original, 'Missing June reference')
    for r in eligible:
        ids = [x['id'] for x in r['scenes']]
        group = r['acquisition']
        sat, stamp = group.split('_')
        require(group.replace('_','').isalnum() and len(ids) == r['complete_asset_scene_count']
                and len(ids) == len(set(ids)) and len(ids)>0
                and all(x.startswith(sat+'_') and '_'+stamp+'_L2A' in x for x in ids),
                f'Invalid scene metadata: {group}')
    return eligible, original


def signature23(script, shortlist, geometry, reference):
    # Exact original Script 23 fingerprint: do not use Script 24's bytes.
    h = hashlib.sha256(script.read_bytes())
    for p in (shortlist, geometry):
        h.update(p.read_bytes())
    h.update(str((reference.stat().st_size, reference.stat().st_mtime_ns)).encode())
    return h.hexdigest()


def validate_result(result, candidate, counts, s):
    require(result['status']=='ok' and result['acquisition']==candidate['acquisition']
            and result['date']==candidate['date']
            and result['scene_ids']==[x['id'] for x in candidate['scenes']],
            'Cached result identity mismatch')
    rows = result['regions']
    require([x['region'] for x in rows] == list((*s.REGIONS,s.ALL)), 'Region ordering mismatch')
    hists=[]
    for row,n in zip(rows,counts):
        h=[row[f'scl_{i:02d}_{k}_pixels'] for i,k in enumerate(s.NAMES)]
        require(all(isinstance(x,int) and x>=0 for x in h) and sum(h)==n
                and row['land_pixels']==n, 'Invalid cached histogram')
        good=sum(h[i] for i in s.ACCEPTED)
        require(row['scl_eligible_pixels']==good and row['scl_nonzero_pixels']==n-h[0],
                'Cached counts mismatch')
        for key,num,den in [('scl_eligible_pct_of_land',good,n),('scl_missing_pct_of_land',h[0],n),
                            ('cloud_pct_of_land',sum(h[8:11]),n),('shadow_pct_of_land',h[2]+h[3],n),
                            ('water_pct_of_land',h[6],n),('scl_snow_pct_of_eligible',h[11],good)]:
            value=row[key]
            require((value is None if not den else value is not None and math.isclose(value,100*num/den,abs_tol=1e-8)),
                    f'Cached percentage mismatch: {key}')
        hists.append(h)
    require(np.array_equal(np.sum(hists[:-1],axis=0),hists[-1]),'Municipal histogram mismatch')
    return result


def cached_result(path, signature, candidate, counts, s):
    if not path.exists():
        return None
    try:
        entry=read_json(path)
        if entry.get('signature')!=signature:
            return None
        return validate_result(entry['result'],candidate,counts,s)
    except (ValueError,KeyError,TypeError,OSError):
        return None


def run(processed, script23):
    s=load_screen_module(script23)
    shortlist_path=processed/'anchorage_2025_acquisition_shortlist.json'
    candidates, original=inventory(read_json(shortlist_path),s)
    print(f'Footprint-qualified acquisitions: {len(candidates)}; original shortlist: {len(original)}; additional: {len(candidates)-len(original)}',flush=True)
    grid,labels,reference=s.load_grid(processed)
    counts=[int(x) for x in np.bincount(labels.ravel(),minlength=5)[1:5]]
    counts.append(sum(counts))
    signature=signature23(script23,shortlist_path,processed/'anchorage_land_study_regions.geojson',reference)
    old_cache=processed/'seasonal_scl_screen_cache'
    new_cache=processed/'seasonal_scl_expanded_cache'
    new_cache.mkdir(exist_ok=True)
    # Check all old screens before starting remote work. Never silently redownload them.
    prior={k:cached_result(old_cache/f'{k}.json',signature,c,counts,s) for k,c in original.items()}
    missing=[k for k,v in prior.items() if v is None]
    require(not missing, 'Cannot verify original Script 23 caches: '+', '.join(missing)+
            '. Check that Script 23, the shortlist, June quality TIFF and region GeoJSON are unchanged. No original screens were repeated.')
    print(f'Original cache verification: PASS ({len(prior)}/{len(original)})',flush=True)
    results=[]
    client=None
    reused_new=attempted=0
    for i,c in enumerate(candidates,1):
        group=c['acquisition']
        print(f'\n[{i}/{len(candidates)}] {group}',flush=True)
        if group in prior:
            result=prior[group]
            print('  Reusing original Script 23 result',flush=True)
        else:
            path=new_cache/f'{group}.json'
            result=cached_result(path,signature,c,counts,s)
            if result is not None:
                reused_new+=1
                print('  Reusing completed expanded-screen result',flush=True)
            else:
                attempted+=1
                try:
                    if client is None:
                        from pystac_client import Client
                        client=Client.open(s.STAC_URL)
                    result=validate_result(s.screen(c,grid,labels,reference,client),c,counts,s)
                    s.atomic_json(path,{'signature':signature,'result':result})
                except Exception as exc:
                    result={'acquisition':group,'date':c['date'],'status':'error',
                            'error':f'{type(exc).__name__}: {exc}'}
                    print('  SCREEN FAILED (not a quality measurement): '+result['error'],flush=True)
        results.append(result)
        if result['status']=='ok':
            for r in result['regions']:
                print(f'  {r["region"]:26s} eligible {r["scl_eligible_pct_of_land"]:6.2f}% | cloud {r["cloud_pct_of_land"]:6.2f}% | shadow {r["shadow_pct_of_land"]:6.2f}%',flush=True)
    months=s.select_months(results)
    before=s.select_months(list(prior.values()))
    failures=sum(r['status']!='ok' for r in results)
    rows=[dict(acquisition=r['acquisition'],date=r['date'],method=r['method'],**x)
          for r in results if r['status']=='ok' for x in r['regions']]
    out=processed/'anchorage_2025_scl_quality_screen_expanded.csv'
    tmp=out.with_suffix('.csv.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    tmp.replace(out)
    selection=processed/'anchorage_2025_scl_quality_selection_expanded.json'
    s.atomic_json(selection,{'year':2025,'version':'24-v1','complete':failures==0,
        'target_pct_each_region':s.TARGET,'eligible_scl_classes':list(s.ACCEPTED),
        'grid':{'crs':'EPSG:3338','pixel_size_m':20,'width':grid['width'],'height':grid['height'],'transform':list(grid['transform'])[:6]},
        'source_signature':signature,'original_screen_count':len(prior),'additional_candidate_count':len(candidates)-len(prior),
        'limitations':'SCL-only screening; not final spectral validity. First nonzero SCL may differ from spectral mosaic. Below-target dates require review. No missing observations imputed. June reference pinned.',
        'original_months':before,'months':months,'results':results})
    print('\nMONTHLY COMPARISON (old -> expanded; worst-region and all-land eligible percentages)')
    for old,new in zip(before,months):
        a,b=old['recommendation'],new['recommendation']
        def fmt(r):
            return f'{r["date"]} ({r["min_region_eligible_pct"]:.2f}% / {r["all_land_eligible_pct"]:.2f}%)' if r else 'none'
        print(f'{new["month"]}: {fmt(a)} -> {fmt(b)} | {new["status"]}')
    print('\nEXPANDED MONTHLY REGIONAL COVERAGE (percent of each region)')
    by_id={r['acquisition']:r for r in results if r['status']=='ok'}
    print('Date          Bowl  Eagle River  Girdwood  Remaining  All land')
    for m in months:
        pick=m['recommendation']
        if pick:
            r=by_id[pick['acquisition']]
            print(f'{r["date"]} '+ ' '.join(f'{x["scl_eligible_pct_of_land"]:10.2f}' for x in r['regions']))
    print(f'\nOriginal screens reused: {len(prior)}; additional cached: {reused_new}; additional attempted: {attempted}')
    print(f'Successful: {len(results)-failures}/{len(results)}; failed: {failures}')
    print(f'Created: {out}\nCreated: {selection}')
    print('Original Script 22/23 outputs, imagery and polygons unchanged. No private AWS operations.')
    if failures:
        print('Partial comparison only. Rerun Script 24 to retry failures; completed screens are cached.')
        raise SystemExit(2)


if __name__=='__main__':
    here=Path(__file__).resolve().parent
    run(here.parent/'data'/'processed',here/'23_screen_seasonal_quality.py')
