"""Anchorage Through the Seasons — synchronized 2025 satellite animation.

Keep the original Scripts 19 (revision 2), 20 and 21 beside this script.
Run normally in PyCharm, or python scripts/27_create_seasonal_animation.py.
Default: local imagery only; neutral background outside the municipality.
Optional --same-date-context retrieves public surrounding RGB through Script 19
(and reuses its June cache); never uses another date as a substitute.
Optional --preview YYYYMMDD renders one frame in a separate output directory.
All scientific inputs and existing static maps are preserved.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from datetime import datetime
from pathlib import Path
import tempfile

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
from shapely.geometry import mapping
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
from PIL import Image

DATES = ('20250207','20250329','20250523','20250612','20250727','20250811','20250930','20251119')
GROUPS = dict(zip(DATES, ('S2B_20250207T212545','S2B_20250329T212803','S2C_20250523T212539',
    'S2C_20250612T212656','S2B_20250727T212521','S2C_20250811T212539','S2C_20250930T212536','S2C_20251119T212811')))
TILES = ('05VPH','05VPJ','06VUN','06VUP','06VVN','06VVP')
ACCENT = '#77C5D5'
NEUTRAL = '#293A45'
DURATION_MS = 1900
END_DURATION_MS = 2600
SIZE = (1920,1080)


def load_maps(script_dir):
    path = script_dir / '21_create_snow_ice_map.py'
    if not path.is_file():
        raise FileNotFoundError(f'Keep Scripts 19, 20 and 21 beside Script 27: {path}')
    spec = importlib.util.spec_from_file_location('anchorage_map03',path)
    snow = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(snow)
    return snow, snow.vegetation, snow.base


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_combined(data, base):
    path = data / 'anchorage_2025_seasonal_regional_statistics.csv'
    with path.open(encoding='utf-8-sig',newline='') as f:
        rows = list(csv.DictReader(f))
    names = (*base.REGIONS,'All Municipal Land')
    expected = {(d,n) for d in DATES for n in names}
    require(len(rows)==40 and {(r['date'],r['region']) for r in rows}==expected,
            'Expected exactly 40 distinct date/region records from Script 26')
    return {(r['date'],r['region']):r for r in rows}


def verify_source(data,date):
    p = data / f'anchorage_{date}_sources.json'
    m = json.loads(p.read_text(encoding='utf-8-sig'))
    sat,stamp = GROUPS[date].split('_',1)
    ids = {f'{sat}_T{tile}_{stamp}_L2A' for tile in TILES}
    records = m['source_scene_priority']
    require(m['date'].replace('-','')==date and m['collection']=='sentinel-2-c1-l2a' and
            len(records)==6 and {r['id'] for r in records}==ids,f'Wrong acquisition provenance: {date}')


def check_combined(data,date,combined):
    with (data/f'anchorage_{date}_regional_statistics.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    for r in rows:
        expected=combined[(date,r['region'])]
        require(all(expected.get(k)==v for k,v in r.items()),f'Combined CSV differs from date CSV: {date}/{r["region"]}')


def local_background(rgb,extent,municipality,base):
    full, expanded = base.context_grid(rgb,extent)
    h,w=full.shape[:2]
    l,r,b,t=expanded
    inside=geometry_mask([mapping(municipality)],(h,w),from_bounds(l,b,r,t,w,h),invert=True)
    missing=inside & (full[...,3]==0)
    full[~inside,:3]=np.array([41,58,69],np.uint8)
    full[missing,:3]=np.array([63,74,82],np.uint8)
    yy,xx=np.indices((h,w),sparse=True)
    full[missing & (((xx+yy)//6)%2==0),:3]=(47,57,64)
    full[...,3]=255
    return full,expanded


def prepare_frame(data,date,regions,municipality,areas,grid,combined,snow,veg,base,context):
    verify_source(data,date)
    check_combined(data,date,combined)
    spectral=data/f'anchorage_{date}_spectral_10m.tif'
    with rasterio.open(spectral) as ds:
        require((ds.crs,ds.transform,ds.width,ds.height)==grid,f'Grid differs: {date}')
        require(ds.tags().get('acquisition_date','').replace('-','')==date,f'RGB date mismatch: {date}')
    rgb,extent,crs=base.read_true_color(spectral)
    rows=snow.read_stats(data,date,areas)
    # Reuse verified map samplers: read full-resolution strips and preserve exact
    # display-center classification, rather than averaging categorical overviews.
    ndvi,veg_cat=veg.sample_and_verify(data,date,rgb.shape[:2],rows)
    snow_cat=snow.sample_and_verify(data,date,rgb.shape[:2],rows,regions)
    require(np.array_equal(veg_cat>0,snow_cat>0),'Display land masks differ')
    if context:
        full,expanded,coverage,used=base.add_surrounding_context(rgb,extent,crs,municipality,date,data)
        sat,stamp=GROUPS[date].split('_',1)
        require(all(i.startswith(sat+'_') and f'_{stamp}_' in i for i in used),'Context provenance mismatch')
        print(f'  Same-date context coverage: {coverage:.2f}%',flush=True)
    else:
        full,expanded=local_background(rgb,extent,municipality,base)
    background=full.copy()
    # Scientific overlays retain dimmed RGB only under excluded water and context.
    h,w=rgb.shape[:2]; H,W=full.shape[:2]; py,px=(H-h)//2,(W-w)//2
    target=background[py:py+h,px:px+w]
    left,right,bottom,top=extent
    inside=geometry_mask([mapping(municipality)],(h,w),from_bounds(left,bottom,right,top,w,h),invert=True)
    dim=inside & (rgb[...,3]>0)
    target[dim,:3]=np.rint(target[dim,:3].astype(np.float32)*base.CONTEXT_DIM).astype(np.uint8)
    return (full,veg.compose(background.copy(),ndvi,veg_cat),snow.compose(background.copy(),snow_cat)),expanded,rows


def legend_chip(fig,x,y,color,label,fg,fontsize=8.4):
    fig.add_artist(Rectangle((x,y-.004),.010,.010,transform=fig.transFigure,facecolor=color,edgecolor='none'))
    fig.text(x+.014,y,label,fontsize=fontsize,color=fg,va='center')


def draw_frame(output,date,images,extent,crs,regions,municipality,land,rows,snow,veg,base,context):
    bg,fg,muted=base.BG,base.FG,base.MUTED
    plt.rcParams.update({'font.family':'DejaVu Sans','figure.facecolor':bg,'savefig.facecolor':bg,'text.color':fg})
    fig=plt.figure(figsize=(16,9),facecolor=bg)
    try:
        date_obj=datetime.strptime(date,'%Y%m%d')
        fig.text(.034,.948,'SOUTH-CENTRAL ALASKA  /  2025 SEASONAL STUDY',fontsize=10,weight='bold',color=ACCENT)
        fig.text(.033,.891,'Anchorage through the seasons',fontsize=29,weight='bold')
        fig.text(.966,.948,f'{DATES.index(date)+1:02d} / {len(DATES):02d}  •  ACTUAL ACQUISITION',ha='right',fontsize=10,color=muted)
        fig.text(.966,.895,date_obj.strftime('%B %d, %Y'),ha='right',fontsize=22,weight='bold')
        water=municipality.difference(land)
        edges=[regions.geometry.iloc[i].boundary.difference(water.boundary.buffer(.5)) for i in range(3)]
        xs=(.034,.355,.676)
        for i,(x,im,title,subtitle) in enumerate(zip(xs,images,('01  True color','02  Vegetation / NDVI','03  Snow and ice'),
               ('Sentinel-2 visible bands','Snow-excluded spectral greenness','NDSI-derived classification'))):
            fig.text(x,.845,title,fontsize=14,weight='bold')
            fig.text(x,.819,subtitle,fontsize=9,color=muted)
            ax=fig.add_axes([x,.408,.29,.393],facecolor=NEUTRAL)
            ax.imshow(im,extent=extent,origin='upper',interpolation='nearest',zorder=1)
            for j,e in enumerate(edges):
                if not e.is_empty:
                    line=gpd.GeoSeries([e],crs=crs)
                    line.plot(ax=ax,color=bg,linewidth=1.6,zorder=4,aspect=None)
                    line.plot(ax=ax,color=base.COLORS[j],linewidth=.85,zorder=5,aspect=None)
            base.boundary(ax,water,base.WATER_LINE,.36,.85,'solid',6)
            base.boundary(ax,municipality,fg,.85,1.,(0,(4,3)),7)
            ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:]); ax.set_aspect('equal',adjustable='box')
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_color('#50636E'); spine.set_linewidth(.7)
            if i==0:
                base.add_scale_and_north(ax,extent,crs,municipality)
        fig.text(xs[0],.391,'Fixed RGB display stretch  •  clouds and shadows visible',fontsize=8.6,color=muted)
        fig.text(xs[0],.369,'Outside: same-date dimmed RGB' if context else 'Outside municipality: not displayed',fontsize=8.6,color=muted)
        fig.text(xs[0],.347,'Blue outlines: mapped water  •  dashed: municipality',fontsize=8.5,color=muted)
        cax=fig.add_axes([xs[1],.374,.29,.014])
        cb=fig.colorbar(ScalarMappable(norm=veg.NORM,cmap=veg.CMAP),cax=cax,orientation='horizontal',ticks=[-1,0,.3,1])
        cb.ax.tick_params(colors=muted,labelsize=8,length=2,pad=2);cb.outline.set_edgecolor('#50636E')
        legend_chip(fig,xs[1],.337,veg.SNOW,'Snow excluded',muted)
        legend_chip(fig,xs[1]+.135,.337,veg.UNUSABLE,'Unusable / missing',muted)
        legend_chip(fig,xs[2],.386,snow.SNOW,'Snow / ice detected',muted)
        legend_chip(fig,xs[2],.365,snow.NO_SNOW,'Snow / ice not detected',muted)
        legend_chip(fig,xs[2],.344,snow.UNUSABLE,'Unusable / missing',muted)
        fig.add_artist(Line2D([.034,.966],[.317,.317],transform=fig.transFigure,color='#40535F',lw=.8))
        # One table makes the geographic colors and conditional denominators explicit.
        fig.text(.034,.291,'REGIONAL OBSERVATIONS',fontsize=10,weight='bold',color=ACCENT)
        columns=(.36,.51,.66,.81,.953)
        headers=('Snow / ice¹','NDSI usable²','Mean NDVI³','Green proxy³','NDVI usable²')
        for x,hdr in zip(columns,headers): fig.text(x,.291,hdr,ha='right',fontsize=10,color=muted)
        names=(*base.REGIONS,'All Municipal Land')
        labels=(*base.DISPLAY_NAMES,'All municipal land')
        colors=(*base.COLORS,fg)
        def fmt(r,k,decimals,suffix=''):
            return 'N/A' if r[k] in ('',None) else f'{float(r[k]):.{decimals}f}{suffix}'
        for i,(name,label,color) in enumerate(zip(names,labels,colors)):
            r=rows[name]; y=.264-i*.027
            fig.text(.034,y,label,fontsize=10,color=color,weight='bold' if i==4 else 'normal')
            values=(fmt(r,'snow_ice_pct_of_snow_usable',1,'%'),fmt(r,'snow_usable_pct_of_land',1,'%'),
                    fmt(r,'ndvi_mean',3),fmt(r,'green_pct_of_ndvi_usable',1,'%'),fmt(r,'ndvi_usable_pct_of_land',1,'%'))
            for x,v in zip(columns,values): fig.text(x,y,v,ha='right',fontsize=10,weight='bold' if i==4 else 'normal')
        fig.text(.034,.135,'¹ Of NDSI-usable land.  ² Of all region land.  ³ Of usable conservative snow-free observations; green = NDVI ≥ 0.30.',fontsize=8.6,color=muted)
        fig.text(.034,.115,'Low coverage can bias estimates; low winter NDVI coverage also reflects snow exclusion. Snow and glacial ice are not separated.',fontsize=8.6,color=muted)
        # Position actual observation dates in proportion to elapsed calendar time.
        start=datetime(2025,1,1); days=[(datetime.strptime(d,'%Y%m%d')-start).days for d in DATES]
        x0,x1=.045,.955; y=.078
        fig.add_artist(Line2D([x0,x1],[y,y],transform=fig.transFigure,color='#50636E',lw=1.1))
        for i,(d,day) in enumerate(zip(DATES,days)):
            x=x0+(x1-x0)*day/364
            active=d==date
            fig.add_artist(Line2D([x],[y],transform=fig.transFigure,marker='o',markersize=6 if active else 3.6,
                                  color=ACCENT if active else muted,linestyle='none'))
            dt=datetime.strptime(d,'%Y%m%d'); label=f'{dt:%b} {dt.day}'
            fig.text(x,y-.019,label,ha='center',va='top',fontsize=8.3,color=ACCENT if active else muted,weight='bold' if active else 'normal')
        fig.text(.034,.022,'Modified Copernicus Sentinel data 2025  •  Earth Search / Element 84  •  U.S. Census TIGER/Line 2025  |  EPSG:3338; 10 m grid, SWIR/SCL native 20 m',fontsize=7.7,color=muted)
        fig.text(.966,.022,'Discrete observations, not daily imagery',ha='right',fontsize=7.7,color=muted)
        fig.savefig(output,dpi=120,facecolor=bg)
    finally:
        plt.close(fig)
    with Image.open(output) as im:
        require(im.size==SIZE and im.format=='PNG','Unexpected frame size or format')


def make_gif(paths,output):
    frames=[]
    try:
        for p in paths:
            with Image.open(p) as im:
                require(im.size==SIZE,'Animation frames have inconsistent dimensions')
                frames.append(im.convert('RGB').quantize(colors=256,method=Image.Quantize.MEDIANCUT,dither=Image.Dither.NONE))
        durations=[DURATION_MS]*len(frames); durations[-1]=END_DURATION_MS
        frames[0].save(output,save_all=True,append_images=frames[1:],duration=durations,loop=0,disposal=2,optimize=False)
        with Image.open(output) as check:
            require(check.n_frames==len(paths) and check.size==SIZE and check.info.get('loop')==0,'GIF frame count, size or loop mismatch')
            for i,duration in enumerate(durations):
                check.seek(i); check.load()
                require(check.info.get('duration')==duration,f'GIF timing mismatch: frame {i+1}')
        print(f'  GIF frame count, size, timing and loop: PASS ({len(paths)} frames)',flush=True)
    finally:
        for f in frames: f.close()


def run(project,preview=None,context=False):
    data=project/'data'/'processed'
    snow,veg,base=load_maps(project/'scripts')
    base.ACQUISITIONS.update({d:g.split('_',1)[1] for d,g in GROUPS.items()})
    if not context: base.DISPLAY_WIDTH=1200
    combined=read_combined(data,base)
    with rasterio.open(data/'anchorage_20250612_spectral_10m.tif') as ds:
        grid=(ds.crs,ds.transform,ds.width,ds.height)
        require(ds.crs.to_epsg()==3338 and ds.res==(10,10) and ds.shape==(8299,10739),'Unexpected reference grid')
    regions,municipality,land,areas=base.load_regions(data,grid[0])
    dates=(preview,) if preview else DATES
    parent=project/'portfolio'; parent.mkdir(exist_ok=True)
    name=('anchorage_2025_seasonal_preview_'+preview) if preview else 'anchorage_2025_seasonal_animation'
    if context: name+='_context'
    output=parent/name
    require(not output.exists(),f'Output directory already exists (preserved): {output}; move it aside before rerunning.')
    with tempfile.TemporaryDirectory(prefix='seasonal_animation_',dir=parent) as td:
        stage=Path(td)/name; stage.mkdir()
        frame_paths=[]
        for i,date in enumerate(dates,1):
            print(f'\nFrame {i}/{len(dates)} — {date}',flush=True)
            images,extent,rows=prepare_frame(data,date,regions,municipality,areas,grid,combined,snow,veg,base,context)
            path=stage/f'anchorage_{date}_seasonal_frame.png'
            draw_frame(path,date,images,extent,grid[0],regions,municipality,land,rows,snow,veg,base,context)
            frame_paths.append(path)
            print(f'  Frame verified: {path.name} ({path.stat().st_size/1e6:.2f} MB)',flush=True)
        if not preview:
            make_gif(frame_paths,stage/'anchorage_2025_through_the_seasons.gif')
        manifest={'dates':list(dates),'acquisitions':{d:GROUPS[d] for d in dates},'frame_size':list(SIZE),
                  'frame_duration_ms':None if preview else [DURATION_MS]*(len(dates)-1)+[END_DURATION_MS],
                  'context':'same-date public RGB' if context else 'outside municipality not displayed',
                  'rgb_display':'fixed reflectance 0..0.45, power 0.65; average downsample',
                  'scientific_display':'native pixel-center sample; fixed NDVI scale -1..1; categorical snow mask',
                  'source':'local Script 25/26 products; conditional statistics from regional CSVs',
                  'limitations':'Unequal observation coverage; no interpolation between acquisition dates. GIF palette is limited to 256 colors per frame; PNGs preserve full display color.'}
        (stage/'animation_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        require(not output.exists(),'Output appeared during rendering; refusing overwrite')
        stage.rename(output)
    print(f'\nCreated: {output}',flush=True)
    for p in sorted(output.iterdir()): print(f'  {p.name}: {p.stat().st_size/1e6:.2f} MB')
    print('Complete. Existing scientific inputs and approved static maps unchanged. No private AWS operations.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview',choices=DATES,help='Render just one preview PNG in a separate folder')
    parser.add_argument('--same-date-context',action='store_true',help='Use public same-acquisition context; can download additional RGB')
    args=parser.parse_args()
    run(Path(__file__).resolve().parents[1],args.preview,args.same_date_context)
