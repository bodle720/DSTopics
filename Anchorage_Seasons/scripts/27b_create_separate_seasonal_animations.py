"""Three uncluttered seasonal animations; local Alaska vector context, no downloads.
Keep Scripts 19, 20, 21 and 27 with their original names in scripts/.
Run with --preview 20250612 for three stills, or without arguments for all GIFs.
Optional --alaska-boundary PATH selects the existing Alaska polygon dataset.
Original scientific rasters, maps and combined GIF are read-only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path
import tempfile

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
from shapely.geometry import box, Point, mapping
from shapely import make_valid
from shapely.ops import unary_union
from pyproj import Transformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
from PIL import Image

KINDS = ('true_color', 'vegetation', 'snow_ice')
TITLES = ('True color', 'Vegetation / NDVI', 'Snow and ice')
SEA, LAND, COAST = '#243D4B', '#52675F', '#82978C'
ACCENT = '#77C5D5'


def load27(script_dir):
    p = script_dir / '27_create_seasonal_animation.py'
    if not p.is_file():
        raise FileNotFoundError(f'Keep original Script 27 beside this script: {p}')
    spec = importlib.util.spec_from_file_location('seasonal27', p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def polygon_parts(geom):
    if geom.geom_type == 'Polygon' and not geom.is_empty:
        yield geom
    elif hasattr(geom, 'geoms'):
        for part in geom.geoms:
            yield from polygon_parts(part)


def read_alaska(path, crs):
    g = gpd.read_file(path)
    if g.crs is None or g.empty:
        raise ValueError('Boundary needs a CRS and polygon geometry')
    # Support both a stand-alone Alaska layer and a US states layer.
    for field in g.columns:
        if str(field).lower() in ('name','state_name','statename','state','stusps','statefp','statefp20','statefp10'):
            values = g[field].astype(str).str.strip().str.lower()
            selected = values.isin(('alaska','ak','02','2'))
            if selected.any():
                g = g.loc[selected]
                break
    g = g.to_crs(crs)
    parts = []
    for geom in g.geometry:
        if geom is not None and not geom.is_empty:
            parts.extend(polygon_parts(make_valid(geom) if not geom.is_valid else geom))
    if not parts:
        raise ValueError('No polygon components')
    state = unary_union(parts)
    # Sanity checks reject municipality, US-wide layers and wrong geography.
    lonlat = Transformer.from_crs(4326, crs, always_xy=True)
    points = [Point(*lonlat.transform(lon,lat)) for lon,lat in [(-149.9,61.2),(-147.7,64.84),(-134.42,58.3),(-156.8,71.29)]]
    if not state.is_valid or not 1.2e12 < state.area < 2.1e12 or not all(state.distance(p)<25000 for p in points):
        raise ValueError('Not a plausible statewide Alaska polygon in EPSG:3338')
    return state


def find_alaska(project, explicit, crs):
    if explicit is not None:
        p = Path(explicit).expanduser()
        if not p.is_absolute():
            p = project / p
        return read_alaska(p,crs),p
    candidates=[]
    for p in (project/'data').rglob('*'):
        if p.suffix.lower() not in ('.shp','.geojson','.gpkg') or not p.is_file():
            continue
        n=p.stem.lower()
        if 'alaska' in n or 'state' in n or 'boundary' in n or n=='ak':
            candidates.append(p)
    candidates.sort(key=lambda p:(p.stem.lower() not in ('alaska','ak','alaska_boundary'),p.suffix.lower()!='.shp',len(str(p)),str(p)))
    for p in candidates:
        try:
            state=read_alaska(p,crs)
        except Exception as exc:
            print(f'  Skipping {p.name}: {exc}',flush=True)
            continue
        return state,p
    raise FileNotFoundError('No suitable Alaska statewide boundary was discovered under data/. '
                            'Rerun with --alaska-boundary "path\\to\\alaska.shp" (the boundary used in Script 01/QGIS).')


def geography(state,extent,municipality,regions,land):
    l,r,b,t=extent
    # A slightly wider map uses the available canvas without changing raster geometry.
    target=1.49
    extra=max(0,((t-b)*target-(r-l))/2)
    view=(l-extra,r+extra,b,t)
    clip=box(view[0],view[2],view[1],view[3])
    local=state.intersection(clip)
    water=municipality.difference(land)
    edges=[regions.geometry.iloc[i].boundary.difference(water.boundary.buffer(.5)) for i in range(3)]
    return view,local,water,edges


def draw_polygon(ax,geom,crs,color,edge='none',lw=.5,z=1):
    parts=list(polygon_parts(geom))
    if parts:
        gpd.GeoSeries(parts,crs=crs).plot(ax=ax,facecolor=color,edgecolor=edge,linewidth=lw,zorder=z,aspect=None)


def setup_ax(ax,extent):
    ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:]); ax.set_aspect('equal',adjustable='box')
    ax.set_xticks([]);ax.set_yticks([]);ax.set_xlabel('');ax.set_ylabel('')
    for spine in ax.spines.values():
        spine.set_color('#50636E');spine.set_linewidth(.8)


def chip(fig,x,y,color,label,fg,style='fill'):
    if style=='fill':
        fig.add_artist(Rectangle((x,y-.008),.014,.016,transform=fig.transFigure,facecolor=color,edgecolor='none'))
    else:
        fig.add_artist(Line2D([x,x+.016],[y,y],transform=fig.transFigure,color=color,lw=2,ls='--' if style=='dash' else '-'))
    fig.text(x+.022,y,label,fontsize=12.5,color=fg,va='center')


def draw_frame(output,date,image,extent,crs,regions,municipality,state,geo,snow,veg,base,original,kind):
    bg,fg,muted=base.BG,base.FG,base.MUTED
    view,local,water,edges=geo
    plt.rcParams.update({'font.family':'DejaVu Sans','figure.facecolor':bg,'savefig.facecolor':bg,'text.color':fg})
    fig=plt.figure(figsize=(16,9),facecolor=bg)
    try:
        dt=datetime.strptime(date,'%Y%m%d')
        fig.text(.038,.926,TITLES[kind],fontsize=33,weight='bold',va='center')
        fig.text(.962,.926,f'{dt:%B} {dt.day}, {dt.year}',fontsize=29,weight='bold',ha='right',va='center')
        ax=fig.add_axes([.038,.157,.688,.714],facecolor=SEA)
        draw_polygon(ax,local,crs,LAND,COAST,.55,1)
        # Script 27 has an opaque neutral exterior. Replace only that exterior with
        # transparent pixels so the vector land/sea basemap is visible underneath.
        h,w=image.shape[:2]
        l,r,b,t=extent
        inside=geometry_mask([mapping(municipality)],(h,w),from_bounds(l,b,r,t,w,h),invert=True)
        image=image.copy(); image[...,3]=inside.astype(np.uint8)*255
        ax.imshow(image,extent=extent,origin='upper',interpolation='nearest',zorder=2)
        for j,e in enumerate(edges):
            if not e.is_empty:
                gs=gpd.GeoSeries([e],crs=crs)
                gs.plot(ax=ax,color=bg,linewidth=2.4,zorder=4,aspect=None)
                gs.plot(ax=ax,color=base.COLORS[j],linewidth=1.3,zorder=5,aspect=None)
        base.boundary(ax,water,base.WATER_LINE,.55,.85,'solid',6)
        base.boundary(ax,municipality,fg,1.3,1.,(0,(4,3)),7)
        setup_ax(ax,view)
        base.add_scale_and_north(ax,view,crs,municipality)
        # Standalone Alaska locator; marker is at the city, not the inland municipal centroid.
        fig.text(.751,.849,'Alaska',fontsize=18,weight='bold')
        inset=fig.add_axes([.751,.65,.211,.18],facecolor=SEA)
        draw_polygon(inset,state,crs,LAND,COAST,.45,1)
        sl,sb,sr,st=state.bounds; dx=sr-sl;dy=st-sb
        p=Point(*Transformer.from_crs(4326,crs,always_xy=True).transform(-149.9003,61.2181))
        inset.scatter([p.x],[p.y],s=45,c=ACCENT,edgecolors=bg,linewidths=.9,zorder=4)
        inset.annotate('Anchorage',(p.x,p.y),xytext=(6,-12),textcoords='offset points',fontsize=9.5,color=fg,weight='bold',zorder=5,
                       bbox=dict(fc=bg,ec='none',alpha=.85,pad=1.5))
        setup_ax(inset,(sl-dx*.035,sr+dx*.035,sb-dy*.08,st+dy*.08))
        fig.text(.751,.618,'Map key',fontsize=17,weight='bold')
        if kind==0:
            fig.text(.751,.582,'Visible satellite bands',fontsize=13,color=muted)
            fig.text(.751,.55,'Clouds and shadows visible',fontsize=12.5,color=muted)
            fig.text(.751,.518,'Hatched gray: missing imagery',fontsize=11.5,color=muted)
            y=.470
        elif kind==1:
            cax=fig.add_axes([.751,.558,.211,.023]);cb=fig.colorbar(ScalarMappable(norm=veg.NORM,cmap=veg.CMAP),cax=cax,orientation='horizontal',ticks=[-1,0,.3,1])
            cb.ax.tick_params(colors=muted,labelsize=11,length=3,pad=3);cb.outline.set_edgecolor('#50636E')
            fig.text(.751,.503,'Lower signal',fontsize=11,color=muted)
            fig.text(.962,.503,'Higher signal',fontsize=11,color=muted,ha='right')
            chip(fig,.751,.466,veg.SNOW,'Snow excluded',fg)
            chip(fig,.751,.433,veg.UNUSABLE,'Unusable / missing',fg)
            y=.383
        else:
            chip(fig,.751,.576,snow.SNOW,'Snow / ice detected',fg)
            chip(fig,.751,.543,snow.NO_SNOW,'Snow / ice not detected',fg)
            chip(fig,.751,.510,snow.UNUSABLE,'Unusable / missing',fg)
            y=.460
        chip(fig,.751,y,fg,'Municipal boundary',muted,'dash')
        chip(fig,.751,y-.032,base.WATER_LINE,'Mapped water',muted,'line')
        for i,name in enumerate(base.DISPLAY_NAMES[:3]):
            chip(fig,.751,y-.064-.032*i,base.COLORS[i],name,fg,'line')
        chip(fig,.751,y-.174,LAND,'Surrounding land (vector)',muted)
        chip(fig,.751,y-.206,SEA,'Surrounding water',muted)
        # Calendar-proportional timeline; no frame counter or tiny footnotes.
        dts=[datetime.strptime(d,'%Y%m%d') for d in original.DATES]
        first,last=dts[0],dts[-1]
        x0,x1=.068,.932; y=.098
        fig.add_artist(Line2D([x0,x1],[y,y],transform=fig.transFigure,color='#50636E',lw=1.5))
        for d,when in zip(original.DATES,dts):
            x=x0+(x1-x0)*(when-first).days/(last-first).days
            active=d==date
            fig.add_artist(Line2D([x],[y],transform=fig.transFigure,marker='o',markersize=9 if active else 5,
                                   color=ACCENT if active else muted,linestyle='none'))
            fig.text(x,y-.024,f'{when:%b} {when.day}',ha='center',va='top',fontsize=13.5,color=ACCENT if active else muted,weight='bold' if active else 'normal')
        fig.savefig(output,dpi=120,facecolor=bg)
    finally:
        plt.close(fig)
    with Image.open(output) as im:
        original.require(im.size==original.SIZE and im.format=='PNG','Unexpected frame dimensions')


def run(project,preview=None,alaska_boundary=None):
    original=load27(project/'scripts')
    snow,veg,base=original.load_maps(project/'scripts')
    data=project/'data'/'processed'
    base.DISPLAY_WIDTH=2000
    combined=original.read_combined(data,base)
    with rasterio.open(data/'anchorage_20250612_spectral_10m.tif') as ds:
        grid=(ds.crs,ds.transform,ds.width,ds.height)
        original.require(ds.crs.to_epsg()==3338 and ds.res==(10,10) and ds.shape==(8299,10739),'Unexpected reference grid')
    regions,municipality,land,areas=base.load_regions(data,grid[0])
    state,path=find_alaska(project,alaska_boundary,grid[0])
    print(f'Alaska vector context: {path}',flush=True)
    parent=project/'portfolio';parent.mkdir(exist_ok=True)
    name='anchorage_2025_separate_animations'+('_preview_'+preview if preview else '')
    output=parent/name
    original.require(not output.exists(),f'Output exists, preserved: {output}. Move it aside before rerunning.')
    dates=(preview,) if preview else original.DATES
    with tempfile.TemporaryDirectory(prefix='separate_animation_',dir=parent) as td:
        stage=Path(td)/name;stage.mkdir()
        paths=[[] for _ in KINDS]; geo=None; reference_extent=None
        for i,date in enumerate(dates,1):
            print(f'\nAcquisition {i}/{len(dates)} — {date}',flush=True)
            images,extent,rows=original.prepare_frame(data,date,regions,municipality,areas,grid,combined,snow,veg,base,False)
            if geo is None:
                geo=geography(state,extent,municipality,regions,land);reference_extent=extent
            original.require(extent==reference_extent,'Display extent changed')
            for k,kind in enumerate(KINDS):
                folder=stage/kind;folder.mkdir(exist_ok=True)
                file=folder/f'anchorage_{date}_{kind}.png'
                draw_frame(file,date,images[k],extent,grid[0],regions,municipality,state,geo,snow,veg,base,original,k)
                paths[k].append(file)
                print(f'  Rendered: {file.name}',flush=True)
        if not preview:
            for kind,frames in zip(KINDS,paths):
                original.make_gif(frames,stage/f'anchorage_2025_{kind}.gif')
        try: source=str(path.relative_to(project))
        except ValueError: source=str(path)
        manifest={'dates':list(dates),'acquisitions':{d:original.GROUPS[d] for d in dates},'size':list(original.SIZE),
                  'timing_ms':None if preview else [original.DURATION_MS]*7+[original.END_DURATION_MS],
                  'context':{'source':source,'description':'Static vector land/water context only; no outside-municipality satellite observations displayed.'},
                  'display':{'rgb':'fixed 0..0.45 reflectance, power .65','ndvi':'fixed -1..1, snow-excluded','snow':'existing NDSI classes','sampling':'Script 27 / 20 / 21; full-resolution counts verified'},
                  'limitations':'Discrete dates; missing observations distinct; conditional metrics and coverage in original regional CSV; GIF palette limited to 256 colors per frame.',
                  'attribution':'Contains modified Copernicus Sentinel data 2025; Earth Search / Element 84; U.S. Census TIGER/Line 2025 hydrography. Include Alaska vector dataset attribution on public website (verify source metadata).'}
        (stage/'animation_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        original.require(not output.exists(),'Output appeared during rendering; refusing overwrite')
        stage.rename(output)
    print(f'\nCreated: {output}',flush=True)
    print('Complete. Original animations, static maps, and scientific inputs unchanged. No network or AWS operations.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview',choices=('20250207','20250329','20250523','20250612','20250727','20250811','20250930','20251119'))
    parser.add_argument('--alaska-boundary',help='Explicit path to local Alaska polygon dataset')
    args=parser.parse_args()
    run(Path(__file__).resolve().parents[1],args.preview,args.alaska_boundary)
