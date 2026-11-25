"""Anchorage Through the Seasons | Map 03: snow and ice.
Run in PyCharm or: python scripts/21_create_snow_ice_map.py
Requires Scripts 19 (revision 2) and 20 alongside this file and the existing
local RGB context cache. No network access or changes to scientific inputs.
Optional --date YYYYMMDD --overwrite (replaces only this PNG).
"""
from __future__ import annotations
import argparse
import importlib.util
from datetime import datetime
from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

path20 = Path(__file__).with_name('20_create_vegetation_map.py')
if not path20.is_file():
    raise FileNotFoundError(f'Script 20 must be beside Script 21: {path20}')
spec = importlib.util.spec_from_file_location('anchorage_map02', path20)
vegetation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vegetation)
base = vegetation.base
require = base.require
DATE = '20250612'
BG, FG, MUTED = base.BG, base.FG, base.MUTED
SNOW = '#D9ECF8'
NO_SNOW = '#648E65'
UNUSABLE = '#796D89'
WATER = base.WATER_LINE
KEYS = ('land_pixels', 'ndsi_valid_pixels', 'ndsi_snow_pixels',
        'scl_snow_on_ndsi_valid_pixels', 'snow_agreement_pixels', 'snow_disagreement_pixels')
PERCENTAGES = (('snow_usable_pct_of_land', 1, 0),
               ('snow_ice_pct_of_snow_usable', 2, 1),
               ('scl_snow_pct_on_same_snow_usable', 3, 1),
               ('ndsi_scl_agreement_pct', 4, 1))


def read_stats(processed, date, areas):
    rows = vegetation.read_stats(processed, date, areas)
    for name, r in rows.items():
        for k in KEYS:
            r[k] = int(r[k])
        n, v, snow, scl, agree, disagree = (r[k] for k in KEYS)
        require(0 <= snow <= v <= n and 0 <= scl <= v and agree >= 0 and disagree >= 0 and
                agree + disagree == v and abs(snow-scl) <= disagree <= min(snow+scl, 2*v-snow-scl),
                f'Invalid snow comparison counts: {name}')
        for key, a, b in PERCENTAGES:
            num, denom = r[KEYS[a]], r[KEYS[b]]
            require((denom > 0 and abs(float(r[key])-100*num/denom) < .00002) or
                    (denom == 0 and r[key] in ('', None)), f'Invalid percentage: {name}/{key}')
    for key in KEYS:
        require(sum(rows[n][key] for n in base.REGIONS) == rows['All Municipal Land'][key], f'Invalid total {key}')
    return rows


def sample_and_verify(processed, date, shape, rows, regions):
    """Verify each region at native resolution, then sample matching display centers.
    Reuse the fixed polygon order and pixel-center rule from Script 18.
    """
    prefix = processed / f'anchorage_{date}'
    paths = [Path(str(prefix)+suffix) for suffix in ('_indices_10m.tif','_snow_ice_10m.tif','_quality_10m.tif','_spectral_10m.tif')]
    h,w = shape
    with rasterio.open(paths[0]) as ix, rasterio.open(paths[1]) as sm, rasterio.open(paths[2]) as q, rasterio.open(paths[3]) as sp:
        require(all(ds.crs == sp.crs and ds.shape == sp.shape and ds.transform == sp.transform and
                    ds.tags(ns='IMAGE_STRUCTURE').get('LAYOUT') == 'COG' for ds in (ix,sm,q,sp)), 'Input COGs not on same grid')
        require(ix.count == 2 and ix.nodata == -9999 and 'ndsi' in (ix.descriptions[1] or '').lower() and
                sm.count == 1 and sm.nodata == 0 and q.count == 2 and 'scl' in (q.descriptions[0] or '').lower(), 'Unexpected bands or NoData')
        require(ix.tags().get('DATE') == date and sm.tags().get('DATE') == date, 'Raster date mismatch')
        require(float(sm.tags()['NDSI_THRESHOLD']) == .4 and float(sm.tags()['MIN_GREEN_REFLECTANCE']) == .1, 'Snow raster threshold mismatch')
        labels = rasterize([(geom,i+1) for i,geom in enumerate(regions.geometry)], out_shape=sp.shape,
                           transform=sp.transform, fill=0, all_touched=False, dtype='uint8')
        rr = np.floor((np.arange(h)+.5)*ix.height/h).astype(int)
        cc = np.floor((np.arange(w)+.5)*ix.width/w).astype(int)
        cat = np.zeros((h,w),np.uint8)
        counts = np.zeros((4,len(KEYS)),dtype=np.int64)
        for y in range(0,ix.height,256):
            stop = min(y+256,ix.height)
            win = Window(0,y,ix.width,stop-y)
            a = ix.read(2,window=win)
            c = sm.read(1,window=win)
            scl, source = q.read(window=win)
            dn = sp.read(2,window=win)
            lab = labels[y:stop]
            valid = c>=2
            snow = c==3
            require(np.all(c<=3) and np.array_equal(c>0,lab>0), 'Unexpected snow class or land mask mismatch')
            require(np.array_equal(a!=ix.nodata,valid) and np.all(np.isfinite(a[valid])) and
                    np.all(np.abs(a[valid])<=1), 'NDSI validity mismatch')
            require(not np.any(valid & ((source<=0) | ~np.isin(scl,(4,5,11)))), 'Unusable SCL/source in valid snow map')
            green = dn.astype(np.float64)*sp.scales[1]+sp.offsets[1]
            expected = valid & (a>=np.float32(.4)) & (green>=.1-1e-12)
            require(np.array_equal(snow,expected), 'Snow mask differs from NDSI/green threshold')
            masks = (lab>0,valid,snow,valid & (scl==11),valid & (snow==(scl==11)),valid & (snow!=(scl==11)))
            for j,m in enumerate(masks):
                counts[:,j] += np.bincount(lab[m],minlength=5)[1:5]
            select = np.flatnonzero((rr>=y)&(rr<stop))
            if select.size:
                cat[select] = c[np.ix_(rr[select]-y,cc)]
            if y//256 % 8 == 0 or stop == ix.height:
                print(f'  Checked {stop:,}/{ix.height:,} native raster rows',flush=True)
        for i,name in enumerate(base.REGIONS):
            require(all(int(counts[i,j])==rows[name][k] for j,k in enumerate(KEYS)),f'Raster and CSV mismatch: {name}: {counts[i]}')
        require(all(int(counts[:,j].sum())==rows['All Municipal Land'][k] for j,k in enumerate(KEYS)), 'Municipal totals mismatch')
        print('  Full-resolution NDSI, snow-mask, SCL comparison and regional CSV checks: PASS',flush=True)
    return cat


def compose(background, cat):
    h,w=cat.shape
    H,W=background.shape[:2]
    py,px=(H-h)//2,(W-w)//2
    require((H-h)%2==0 and (W-w)%2==0, 'Display padding mismatch')
    target=background[py:py+h,px:px+w]
    for k,color in [(1,UNUSABLE),(2,NO_SNOW),(3,SNOW)]:
        target[cat==k,:3]=np.rint(np.array(to_rgb(color))*255).astype(np.uint8)
    return background


def render(output, date, rgba, extent, crs, regions, municipality, land, rows):
    plt.rcParams.update({'font.family':'DejaVu Sans','savefig.facecolor':BG,'figure.facecolor':BG,'text.color':FG})
    fig=plt.figure(figsize=(16,10),facecolor=BG)
    accent='#77C5D5'
    fig.text(.035,.955,'ANCHORAGE THROUGH THE SEASONS',fontsize=10,weight='bold',color=accent)
    fig.text(.035,.902,'Anchorage: Where winter lingers',fontsize=29,weight='bold')
    fig.text(.037,.866,'NDSI-derived snow and ice classification  |  Municipality of Anchorage, Alaska',fontsize=12.5,color=MUTED)
    fig.text(.968,.949,date.strftime('%B %d, %Y').upper(),ha='right',fontsize=13,weight='bold')
    fig.text(.968,.918,'SOUTH-CENTRAL ALASKA  /  MAP 03',ha='right',fontsize=10,color=MUTED)
    ax=fig.add_axes([.035,.115,.655,.71],facecolor='#293A45')
    ax.imshow(rgba,extent=extent,origin='upper',interpolation='nearest',zorder=1)
    water=municipality.difference(land)
    gpd.GeoSeries(list(base.polygon_parts(water)),crs=crs).plot(ax=ax,facecolor=WATER,edgecolor='none',alpha=.12,zorder=3,aspect=None)
    for i in range(3):
        edge=regions.geometry.iloc[i].boundary.difference(water.boundary.buffer(.5))
        if not edge.is_empty:
            # Dark under-stroke makes the region outline distinct from the snow classification.
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
    text(0,1,'SNOW & ICE SIGNAL',fontsize=11,weight='bold',color=accent)
    text(0,.94,fmt(total,'snow_ice_pct_of_snow_usable',1,'%'),fontsize=26,weight='bold')
    text(.53,.94,fmt(total,'snow_usable_pct_of_land',1,'%'),fontsize=23,weight='bold')
    text(0,.875,'Detected snow/ice¹',fontsize=10.5,color=MUTED)
    text(.53,.875,'Land with valid NDSI',fontsize=10,color=MUTED)
    text(0,.812,'REGIONAL RESULTS',fontsize=10,weight='bold',color=accent)
    text(0,.774,'NDSI snow¹   /   SCL snow¹   /   Usable land²',fontsize=9.4,color=MUTED)
    for i,(name,label,color) in enumerate(zip(base.REGIONS,base.DISPLAY_NAMES,base.COLORS)):
        r=rows[name];y=.725-i*.083
        text(0,y,label,fontsize=11.2,weight='bold',color=color)
        text(0,y-.039,f"{fmt(r,'snow_ice_pct_of_snow_usable',2,'%')}       /       {fmt(r,'scl_snow_pct_on_same_snow_usable',2,'%')}       /       {fmt(r,'snow_usable_pct_of_land',1,'%')}",fontsize=10.1)
    panel.plot([0,1],[.398,.398],color='#40535F',lw=.8,transform=panel.transAxes)
    text(0,.375,'NDSI vs. SCL  •  all municipal land',fontsize=10.5,weight='bold')
    if total['ndsi_valid_pixels']:
        a=float(total['snow_ice_pct_of_snow_usable']); b=float(total['scl_snow_pct_on_same_snow_usable'])
        text(0,.334,f'NDSI {a:.1f}%   |   SCL {b:.1f}%   |   Δ {b-a:+.1f} pp³',fontsize=10,color=FG)
        text(0,.298,f"Pixel agreement: {fmt(total,'ndsi_scl_agreement_pct',1,'%')} (not ground truth)",fontsize=9,color=MUTED)
    else:
        text(0,.334,'No usable observations for comparison',fontsize=9,color=MUTED)
    for y,color,label in [(.247,SNOW,'Snow/ice detected (NDSI rule)'),(.207,NO_SNOW,'Snow/ice not detected'),(.167,UNUSABLE,'Unusable / missing observation'),(.127,WATER,'Mapped water: outside study land')]:
        panel.add_patch(Rectangle((0,y-.019),.055,.024,transform=panel.transAxes,facecolor=color,edgecolor='none'))
        text(.083,y+.004,label,fontsize=9,color=MUTED)
    panel.plot([0,.055],[.079,.079],color=FG,lw=1.2,ls='--',transform=panel.transAxes)
    text(.083,.09,'Municipality of Anchorage boundary',fontsize=9,color=MUTED)
    text(0,.049,'¹ Of NDSI-usable land.  ² Of all region land.',fontsize=8.5,color=MUTED)
    text(0,.018,'³ SCL minus NDSI, percentage points; same usable pixels.',fontsize=8.5,color=MUTED)
    fig.add_artist(Line2D([.035,.968],[.088,.088],transform=fig.transFigure,color='#40535F',lw=.8))
    fig.text(.035,.060,f'Contains modified Copernicus Sentinel data {date.year}  •  Earth Search / Element 84  •  U.S. Census TIGER/Line 2025 hydrography',fontsize=9,color=MUTED)
    fig.text(.035,.035,'EPSG:3338  |  10 m grid; SWIR/SCL native 20 m  |  NDSI ≥ 0.40 and green reflectance ≥ 0.10  |  Exploratory proxy; seasonal snow and glacial ice not separated.',fontsize=8.3,color=MUTED)
    fig.text(.968,.035,'SNOW & ICE ANALYSIS',ha='right',fontsize=9,weight='bold',color=accent)
    try:fig.savefig(output,dpi=240,facecolor=BG)
    finally:plt.close(fig)


def run(project,date_string=DATE,overwrite=False):
    date=datetime.strptime(date_string,'%Y%m%d')
    require(date.strftime('%Y%m%d')==date_string,'Date must be YYYYMMDD')
    processed=project/'data'/'processed'
    output=project/'portfolio'/f'anchorage_{date_string}_snow_ice.png'
    require(overwrite or not output.exists(),f'Output exists: {output}. Use --overwrite to replace only this PNG.')
    print('Reading existing RGB and verified study regions...',flush=True)
    rgb,extent,crs=base.read_true_color(processed/f'anchorage_{date_string}_spectral_10m.tif')
    regions,municipality,land,areas=base.load_regions(processed,crs)
    rows=read_stats(processed,date_string,areas)
    background,expanded=vegetation.local_context(processed,date_string,rgb,extent,crs,municipality)
    print('Reading snow classes and verifying regional NDSI/SCL comparison...',flush=True)
    category=sample_and_verify(processed,date_string,rgb.shape[:2],rows,regions)
    rgba=compose(background,category)
    print('Rendering snow and ice portfolio map...',flush=True)
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
