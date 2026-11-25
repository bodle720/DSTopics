"""Anchorage Through the Seasons | Map 02: vegetation.

Run in PyCharm or: python scripts/20_create_vegetation_map.py
Requires the revised Script 19 in the same scripts directory and its local
RGB context cache. No network access, no changes to scientific inputs.
Optional --date YYYYMMDD --overwrite (replaces only this PNG).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
from rasterio.windows import Window
from shapely.geometry import mapping
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable

# Reuse the accepted Map 01 layout and geometry policy, not its network code.
path19 = Path(__file__).with_name('19_create_geographic_overview.py')
spec = importlib.util.spec_from_file_location('anchorage_map01', path19)
if spec is None or spec.loader is None or not path19.is_file():
    raise FileNotFoundError(f'Revised Script 19 must be beside Script 20: {path19}')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
require = base.require
DATE = '20250612'
BG, FG, MUTED = base.BG, base.FG, base.MUTED
SNOW = '#D9ECF8'
UNUSABLE = '#796D89'
WATER = base.WATER_LINE
CMAP = plt.get_cmap('RdYlGn')
NORM = Normalize(-1, 1, clip=True)


def read_stats(processed, date, areas):
    prefix = processed / f'anchorage_{date}'
    with Path(str(prefix) + '_regional_statistics.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    names = (*base.REGIONS, 'All Municipal Land')
    require(len(rows) == 5 and {r['region'] for r in rows} == set(names), 'Unexpected statistics regions')
    rows = {r['region']: r for r in rows}
    method = json.loads(Path(str(prefix) + '_analysis_method.json').read_text(encoding='utf-8-sig'))
    require(method['date'] == date and method['crs'] == 'EPSG:3338' and method['pixel_spacing_m'] == 10,
            'Analysis date/grid mismatch')
    require(method['vegetation_proxy']['ndvi_threshold'] == .30 and
            method['snow_proxy']['ndsi_threshold'] == .40 and method['snow_proxy']['green_min'] == .10,
            'Unexpected analysis thresholds; review map labels before proceeding')
    keys = ('land_pixels', 'ndvi_valid_pixels', 'green_pixels', 'ndsi_valid_pixels', 'ndsi_snow_pixels', 'snow_free_candidate_pixels')
    for name, area in zip(names, [*areas, sum(areas)]):
        r = rows[name]
        require(r['date'] == date and abs(float(r['vector_land_km2']) - area) < .001,
                f'Statistics date or region area mismatch: {name}')
        for k in keys: r[k] = int(r[k])
        n, v, green, sv, snow, sf = (r[k] for k in keys)
        require(0 <= green <= v <= sf <= sv <= n and 0 <= snow <= sv and sf <= sv-snow and n > 0,
                f'Invalid statistic counts: {name}')
        for k, numerator, denom in [('green_pct_of_ndvi_usable',green,v), ('ndvi_usable_pct_of_land',v,n)]:
            if denom:
                require(abs(float(r[k]) - 100*numerator/denom) < .00002, f'Percentage mismatch: {name}/{k}')
            else:
                require(not r[k], f'Expected missing percentage: {name}/{k}')
        require((v and -1 <= float(r['ndvi_mean']) <= 1) or (not v and not r['ndvi_mean']), 'Invalid mean NDVI')
    total = rows[names[-1]]
    for k in keys:
        require(sum(rows[n][k] for n in names[:-1]) == total[k], f'Regional totals mismatch: {k}')
    if total['ndvi_valid_pixels']:
        mean = sum(float(rows[n]['ndvi_mean'] or 0)*rows[n]['ndvi_valid_pixels'] for n in names[:-1])/total['ndvi_valid_pixels']
        require(abs(mean-float(total['ndvi_mean'])) < .000002, 'Weighted NDVI mean mismatch')
    print('  CSV totals, denominators and analysis thresholds: PASS', flush=True)
    return rows


def local_context(processed, date, rgb, extent, crs, municipality):
    """Read only the validated local cache created by revised Script 19."""
    full, expanded = base.context_grid(rgb, extent)
    h,w = full.shape[:2]
    left,right,bottom,top = expanded
    inside = geometry_mask([mapping(municipality)], (h,w), from_bounds(left,bottom,right,top,w,h), invert=True)
    cache = processed / f'anchorage_{date}_overview_context_v2.npz'
    require(cache.is_file(), f'Missing Map 01 local context cache: {cache}. Run revised Script 19 first.')
    with np.load(cache, allow_pickle=False) as data:
        context = data['rgba']
        stamp = str(data['stamp'].item())
        require(context.shape == full.shape and context.dtype == np.uint8 and
                np.allclose(data['extent'],expanded,rtol=0,atol=1e-6) and stamp.startswith(date+'T'),
                'Context cache date or display grid mismatch')
        if date in base.ACQUISITIONS:
            require(stamp == base.ACQUISITIONS[date], 'Context acquisition mismatch')
    outside = ~inside
    full[outside] = context[outside]
    valid = full[...,3] != 0
    # Background imagery is intentionally dimmed both outside and under mapped water.
    full[valid,:3] = np.rint(full[valid,:3].astype(np.float32)*base.CONTEXT_DIM).astype(np.uint8)
    missing = ~valid
    yy,xx = np.indices((h,w), sparse=True)
    full[missing,:3] = (47,57,64)
    full[missing & (((xx+yy)//9)%2==0),:3] = (63,74,82)
    full[...,3] = 255
    print(f'  Local context cache: {cache.name}; outside coverage {100*(valid & outside).sum()/outside.sum():.2f}%',flush=True)
    return full, expanded


def sample_and_verify(processed, date, shape, rows):
    """Read native-resolution strips; sample one identical center pixel per display
    cell, avoiding mismatched COG overview resampling between continuous and categorical bands.
    Check full-resolution counts against Script 18 before rendering.
    """
    prefix = processed / f'anchorage_{date}'
    paths = [Path(str(prefix)+suffix) for suffix in ('_indices_10m.tif','_snow_ice_10m.tif','_quality_10m.tif','_spectral_10m.tif')]
    h,w = shape
    with rasterio.open(paths[0]) as ix, rasterio.open(paths[1]) as sm, rasterio.open(paths[2]) as q, rasterio.open(paths[3]) as sp:
        require(all(ds.crs == sp.crs and ds.shape == sp.shape and ds.transform == sp.transform and
                    ds.tags(ns='IMAGE_STRUCTURE').get('LAYOUT') == 'COG' for ds in (ix,sm,q,sp)), 'Input COGs not on same grid')
        require(ix.count == 2 and ix.nodata == -9999 and 'ndvi' in ix.descriptions[0].lower() and
                sm.count == 1 and sm.nodata == 0 and q.count == 2 and 'scl' in q.descriptions[0].lower(), 'Unexpected bands or NoData')
        require(ix.tags().get('DATE') == date and sm.tags().get('DATE') == date, 'Raster date mismatch')
        # A display sample is the native pixel containing the display pixel center.
        rr = np.floor((np.arange(h)+.5)*ix.height/h).astype(int)
        cc = np.floor((np.arange(w)+.5)*ix.width/w).astype(int)
        ndvi = np.full((h,w),np.nan,np.float32)
        category = np.zeros((h,w),np.uint8) # 0 outside land, 1 unusable, 2 snow excluded, 3 valid NDVI
        count = dict(land_pixels=0,ndvi_valid_pixels=0,green_pixels=0,ndsi_valid_pixels=0,ndsi_snow_pixels=0,snow_free_candidate_pixels=0)
        sum_ndvi = 0.
        for y in range(0,ix.height,256):
            stop = min(y+256,ix.height)
            win = Window(0,y,ix.width,stop-y)
            a = ix.read(1,window=win)
            c = sm.read(1,window=win)
            scl = q.read(1,window=win)
            require(np.all(c<=3), 'Unexpected snow class')
            valid = a != ix.nodata
            require(np.all(np.isfinite(a[valid])) and np.all(np.abs(a[valid])<=1) and not np.any(valid & (c!=2)), 'Invalid NDVI samples')
            # Snow exclusion takes precedence only where NDSI was usable, matching Script 18.
            snow = (c==3) | ((c==2) & (scl==11))
            require(not np.any(valid & snow), 'NDVI includes snow-excluded pixels')
            candidate = (c==2) & (scl!=11)
            count['land_pixels'] += int((c>0).sum())
            count['ndvi_valid_pixels'] += int(valid.sum())
            count['green_pixels'] += int((valid & (a>=.30)).sum())
            count['ndsi_valid_pixels'] += int((c>=2).sum())
            count['ndsi_snow_pixels'] += int((c==3).sum())
            count['snow_free_candidate_pixels'] += int(candidate.sum())
            sum_ndvi += a[valid].sum(dtype=np.float64)
            select = np.flatnonzero((rr>=y)&(rr<stop))
            if select.size:
                p = np.ix_(rr[select]-y,cc)
                sample = np.where(c[p]>0,1,0).astype(np.uint8)
                sample[snow[p]]=2
                sample[valid[p]]=3
                category[select]=sample
                ndvi[select]=np.where(valid[p],a[p],np.nan)
            if y//256 % 8 == 0 or stop == ix.height:
                print(f'  Checked {stop:,}/{ix.height:,} native raster rows',flush=True)
        total = rows['All Municipal Land']
        require(all(total[k]==v for k,v in count.items()), f'Raster and CSV counts differ: {count}')
        if count['ndvi_valid_pixels']:
            require(abs(sum_ndvi/count['ndvi_valid_pixels']-float(total['ndvi_mean']))<.000002, 'Raster mean and CSV differ')
        print('  Full-resolution NDVI, snow-mask and CSV consistency: PASS',flush=True)
    return ndvi, category


def compose(background, ndvi, cat):
    h,w=cat.shape
    H,W=background.shape[:2]
    py,px=(H-h)//2,(W-w)//2
    require((H-h)%2==0 and (W-w)%2==0, 'Display padding mismatch')
    target=background[py:py+h,px:px+w]
    for k,color in [(1,UNUSABLE),(2,SNOW)]:
        target[cat==k,:3]=np.rint(np.array(to_rgb(color))*255).astype(np.uint8)
    valid=cat==3
    target[valid,:3]=np.rint(CMAP(NORM(ndvi[valid]))[:,:3]*255).astype(np.uint8)
    return background


def render(output, date, rgba, extent, crs, regions, municipality, land, rows):
    plt.rcParams.update({'font.family':'DejaVu Sans','savefig.facecolor':BG,'figure.facecolor':BG,'text.color':FG})
    fig=plt.figure(figsize=(16,10),facecolor=BG)
    accent='#77C5D5'
    fig.text(.035,.955,'ANCHORAGE THROUGH THE SEASONS',fontsize=10,weight='bold',color=accent)
    fig.text(.035,.902,'Anchorage: A landscape of green',fontsize=29,weight='bold')
    fig.text(.037,.866,'Normalized Difference Vegetation Index (NDVI)  |  Municipality of Anchorage, Alaska',fontsize=12.5,color=MUTED)
    fig.text(.968,.949,date.strftime('%B %d, %Y').upper(),ha='right',fontsize=13,weight='bold')
    fig.text(.968,.918,'SOUTH-CENTRAL ALASKA  /  MAP 02',ha='right',fontsize=10,color=MUTED)
    ax=fig.add_axes([.035,.115,.655,.71],facecolor='#293A45')
    ax.imshow(rgba,extent=extent,origin='upper',interpolation='nearest',zorder=1)
    water=municipality.difference(land)
    gpd.GeoSeries(list(base.polygon_parts(water)),crs=crs).plot(ax=ax,facecolor=WATER,edgecolor='none',alpha=.12,zorder=3,aspect=None)
    for i in range(3):
        edge=regions.geometry.iloc[i].boundary.difference(water.boundary.buffer(.5))
        if not edge.is_empty:
            # Dark under-stroke makes the region outline distinct from the NDVI ramp.
            s=gpd.GeoSeries([edge],crs=crs)
            s.plot(ax=ax,color=BG,linewidth=2.8,zorder=4,aspect=None)
            s.plot(ax=ax,color=base.COLORS[i],linewidth=1.3,zorder=5,aspect=None)
    base.boundary(ax,water,WATER,.65,.9,'solid',6)
    base.boundary(ax,municipality,FG,1.5,1.,(0,(4,3)),7)
    ax.set_xlim(extent[:2]);ax.set_ylim(extent[2:]);ax.set_aspect('equal',adjustable='box')
    ax.set_xticks([]);ax.set_yticks([]);ax.set_xlabel('');ax.set_ylabel('')
    for spine in ax.spines.values():spine.set_color('#50636E');spine.set_linewidth(.8)
    base.add_scale_and_north(ax,extent,crs,municipality)
    base.add_callouts(ax,regions)
    ax.text(.99,.015,'Outside boundary: dimmed RGB context',transform=ax.transAxes,ha='right',va='bottom',fontsize=8,color=FG,bbox=dict(fc=BG,ec='none',alpha=.85,pad=3),zorder=13)
    panel=fig.add_axes([.72,.115,.25,.71]);panel.set_axis_off()
    def text(x,y,s,**kw):panel.text(x,y,s,transform=panel.transAxes,va='top',**kw)
    def fmt(r,k,n=2,suffix=''):
        return f'{float(r[k]):.{n}f}{suffix}' if r[k] not in ('',None) else 'N/A'
    total=rows['All Municipal Land']
    text(0,1,'VEGETATION SIGNAL',fontsize=11,weight='bold',color=accent)
    text(0,.94,fmt(total,'ndvi_mean',3),fontsize=26,weight='bold')
    text(.53,.94,fmt(total,'ndvi_usable_pct_of_land',1,'%'),fontsize=23,weight='bold')
    text(0,.875,'Mean NDVI',fontsize=10.5,color=MUTED)
    text(.53,.875,'Land with valid NDVI',fontsize=10,color=MUTED)
    text(0,.812,'REGIONAL RESULTS',fontsize=10,weight='bold',color=accent)
    text(0,.774,'Mean NDVI   /   Green proxy¹   /   Usable land²',fontsize=9.4,color=MUTED)
    for i,(name,label,color) in enumerate(zip(base.REGIONS,base.DISPLAY_NAMES,base.COLORS)):
        r=rows[name];y=.725-i*.092
        text(0,y,label,fontsize=11.2,weight='bold',color=color)
        text(0,y-.039,f"{fmt(r,'ndvi_mean',3)}       /       {fmt(r,'green_pct_of_ndvi_usable',1,'%')}       /       {fmt(r,'ndvi_usable_pct_of_land',1,'%')}",fontsize=10.1)
    panel.plot([0,1],[.36,.36],color='#40535F',lw=.8,transform=panel.transAxes)
    text(0,.338,'NDVI  •  fixed scale',fontsize=10.5,weight='bold')
    cax=panel.inset_axes([0,.267,.99,.031]); cb=fig.colorbar(ScalarMappable(norm=NORM,cmap=CMAP),cax=cax,orientation='horizontal',ticks=[-1,0,.3,1])
    cb.ax.tick_params(colors=MUTED,labelsize=8.5,length=3);cb.outline.set_edgecolor('#50636E')
    text(0,.222,'Lower signal',fontsize=8.5,color=MUTED);text(1,.222,'Higher signal',fontsize=8.5,color=MUTED,ha='right')
    for y,color,label in [(.184,SNOW,'Snow/ice indicator: excluded'),(.145,UNUSABLE,'Other unusable / missing observation'),(.106,WATER,'Mapped water: outside study land')]:
        panel.add_patch(Rectangle((0,y-.019),.055,.024,transform=panel.transAxes,facecolor=color,edgecolor='none'))
        text(.083,y+.004,label,fontsize=9,color=MUTED)
    panel.plot([0,.055],[.058,.058],color=FG,lw=1.2,ls='--',transform=panel.transAxes)
    text(.083,.069,'Municipality of Anchorage boundary',fontsize=9,color=MUTED)
    text(0,.022,'¹ NDVI ≥ 0.30, of valid NDVI pixels.  ² Of all region land.',fontsize=8.5,color=MUTED)
    fig.add_artist(Line2D([.035,.968],[.088,.088],transform=fig.transFigure,color='#40535F',lw=.8))
    fig.text(.035,.060,f'Contains modified Copernicus Sentinel data {date.year}  •  Earth Search / Element 84  •  U.S. Census TIGER/Line 2025 hydrography',fontsize=9,color=MUTED)
    fig.text(.035,.035,'EPSG:3338  |  10 m source grid; display sampled  |  NDVI excludes both SCL and NDSI snow indicators where NDSI is usable; green proxy is not fractional cover.',fontsize=8.3,color=MUTED)
    fig.text(.968,.035,'VEGETATION ANALYSIS',ha='right',fontsize=9,weight='bold',color=accent)
    try:fig.savefig(output,dpi=240,facecolor=BG)
    finally:plt.close(fig)


def run(project,date_string=DATE,overwrite=False):
    date=datetime.strptime(date_string,'%Y%m%d')
    require(date.strftime('%Y%m%d')==date_string,'Date must be YYYYMMDD')
    processed=project/'data'/'processed'
    output=project/'portfolio'/f'anchorage_{date_string}_vegetation.png'
    require(overwrite or not output.exists(),f'Output exists: {output}. Use --overwrite to replace only this PNG.')
    print('Reading existing RGB and verified study regions...',flush=True)
    rgb,extent,crs=base.read_true_color(processed/f'anchorage_{date_string}_spectral_10m.tif')
    regions,municipality,land,areas=base.load_regions(processed,crs)
    rows=read_stats(processed,date_string,areas)
    background,expanded=local_context(processed,date_string,rgb,extent,crs,municipality)
    print('Reading NDVI and distinguishing exclusion categories...',flush=True)
    ndvi,category=sample_and_verify(processed,date_string,rgb.shape[:2],rows)
    rgba=compose(background,ndvi,category)
    print('Rendering vegetation portfolio map...',flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    temp=output.with_name(output.stem+'.tmp.png')
    try:
        render(temp,date,rgba,expanded,crs,regions,municipality,land,rows)
        require(temp.stat().st_size>0,'Empty PNG')
        temp.replace(output)
    finally:temp.unlink(missing_ok=True)
    print(f'Created: {output} ({output.stat().st_size/1e6:.2f} MB)')
    print('Complete. Existing inputs and QGIS project unchanged. No network or AWS operations.')
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date',default=DATE)
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    run(Path(__file__).resolve().parents[1],args.date,args.overwrite)
