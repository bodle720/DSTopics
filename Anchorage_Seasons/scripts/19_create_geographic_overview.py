"""Anchorage Through the Seasons | geographic overview, portfolio map.

Run in PyCharm or from the project root:
    python scripts/19_create_geographic_overview.py
Optional: --date YYYYMMDD --overwrite
Revision 2: adds public Earth Search Sentinel-2 RGB context at display resolution.
Does not change QGIS, existing GeoJSON/GeoTIFFs, or private AWS resources.
Writes a reusable local context cache and a separate v2 portfolio PNG.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')  # render to a PNG without opening a GUI window
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from shapely.geometry import mapping, Point
from pyproj import CRS, Transformer, Geod
from shapely import make_valid
from shapely.ops import unary_union
from shapely.validation import explain_validity

DATE = '20250612'
REGIONS = ('Anchorage Bowl', 'Chugiak - Eagle River', 'Girdwood - Alyeska',
           'Remaining Municipal Land')
DISPLAY_NAMES = ('Anchorage Bowl', 'Chugiak – Eagle River', 'Girdwood – Alyeska',
                 'Remaining municipal land')
COLORS = ('#FFD166', '#54E4E5', '#FF92AA', '#A6C3D3')
BG, FG, MUTED = '#101A22', '#F4F7F8', '#B3C3CD'
WATER_LINE = '#65B7FF'
CONTEXT_DIM = .62
CONTEXT_PADDING = .12
STAC_URL = 'https://earth-search.aws.element84.com/v1'
COLLECTION = 'sentinel-2-c1-l2a'
# The only currently verified acquisition; fail rather than substitute another date.
ACQUISITIONS = {'20250612': '20250612T212656'}
# Fixed display transfer function; never use per-date histogram stretching.
RGB_MIN, RGB_MAX, RGB_POWER = 0.0, 0.45, 0.65
DISPLAY_WIDTH = 2600


def require(condition, message):
    if not condition:
        raise ValueError(message)


def polygon_parts(geom):
    if geom.geom_type == 'Polygon' and not geom.is_empty:
        yield geom
    elif geom.geom_type in ('MultiPolygon', 'GeometryCollection'):
        for part in geom.geoms:
            yield from polygon_parts(part)


def repair_polygon(geom, name):
    """Same area-preserving in-memory repair policy as Script 18."""
    require(geom is not None and not geom.is_empty and
            geom.geom_type in ('Polygon', 'MultiPolygon'), f'Invalid polygon type: {name}')
    before = float(geom.area)
    require(math.isfinite(before) and before > 0, f'Invalid polygon area: {name}')
    if not geom.is_valid:
        print(f'  {name}: {explain_validity(geom)}; repairing in memory')
        parts = list(polygon_parts(make_valid(geom)))
        require(bool(parts), f'No polygonal area after repair: {name}')
        geom = unary_union(parts)
    require(geom.is_valid and not geom.is_empty, f'Unusable geometry: {name}')
    require(abs(geom.area - before) <= max(100., before * 1e-7),
            f'Geometry repair changed area materially: {name}')
    return geom


def load_regions(processed, crs):
    g = gpd.read_file(processed / 'anchorage_land_study_regions.geojson')
    require(g.crs is not None and 'name' in g and len(g) == 4 and
            set(g['name']) == set(REGIONS), 'Expected four permanent named regions')
    g = g.set_index('name').loc[list(REGIONS)].to_crs(crs).copy()
    for name in REGIONS:
        g.at[name, g.geometry.name] = repair_polygon(g.geometry.loc[name], name)
    for i in range(4):
        for j in range(i):
            require(g.geometry.iloc[i].intersection(g.geometry.iloc[j]).area <= 100.,
                    'Study polygons overlap materially')
    m = gpd.read_file(processed / 'anchorage_municipality.geojson')
    require(m.crs is not None and len(m) > 0, 'Missing municipal boundary or CRS')
    m = m.to_crs(crs)
    municipality = unary_union([repair_polygon(x, 'Municipality') for x in m.geometry])
    land = unary_union(list(g.geometry))
    require(land.difference(municipality).area <= max(100., land.area * 1e-7),
            'Study land extends outside municipality')
    areas = g.geometry.area.to_numpy() / 1e6
    print('  Geometry checks: PASS')
    return g, municipality, land, areas


def read_true_color(path):
    """Read RGB at bounded display size; preserve NoData transparency."""
    with rasterio.open(path) as ds:
        require(ds.crs == rasterio.crs.CRS.from_epsg(3338) and ds.count == 5,
                'Expected Script 17 five-band EPSG:3338 COG')
        require(ds.tags(ns='IMAGE_STRUCTURE').get('LAYOUT') == 'COG', 'Input is not marked COG')
        require(ds.nodata == 0 and all(x == 'uint16' for x in ds.dtypes), 'Unexpected DN storage')
        require(np.allclose([ds.transform.a, ds.transform.b, ds.transform.d, ds.transform.e],
                            [10, 0, 0, -10], rtol=0, atol=1e-9), 'Expected north-up 10 m grid')
        require(np.allclose(ds.scales, .0001, rtol=0, atol=1e-12) and
                np.allclose(ds.offsets, -.1, rtol=0, atol=1e-12), 'Unexpected reflectance encoding')
        require(all(token in (ds.descriptions[i] or '').lower()
                    for i, token in enumerate(('blue','green','red','nir','swir'))), 'Unexpected band order')
        w = min(ds.width, DISPLAY_WIDTH)
        h = max(1, round(ds.height * w / ds.width))
        # RGB = B04, B03, B02; average is requested for display downsampling.
        indexes = [3, 2, 1]
        dn = ds.read(indexes, out_shape=(3, h, w), masked=True,
                     resampling=Resampling.average)
        valid = ~np.ma.getmaskarray(dn).any(axis=0) & np.all(dn.data != 0, axis=0)
        require(valid.any(), 'No valid RGB samples')
        refl = dn.data.astype(np.float32) * np.float32(.0001) - np.float32(.1)
        rgb = np.clip((refl - RGB_MIN) / (RGB_MAX - RGB_MIN), 0, 1) ** RGB_POWER
        rgba = np.empty((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = np.rint(np.moveaxis(rgb, 0, -1) * 255).astype(np.uint8)
        rgba[:, :, 3] = valid.astype(np.uint8) * 255
        b = ds.bounds
        extent = (b.left, b.right, b.bottom, b.top)
        print(f'  Source: {ds.width:,} x {ds.height:,}, {ds.crs}; RGB display: {w:,} x {h:,}')
        return rgba, extent, ds.crs



def context_grid(rgba, extent):
    """Pad on the exact display pixel lattice; scientific grid is untouched."""
    h, w = rgba.shape[:2]
    px, py = round(w * CONTEXT_PADDING), round(h * CONTEXT_PADDING)
    left, right, bottom, top = extent
    dx, dy = (right-left)/w, (top-bottom)/h
    expanded = (left-px*dx, right+px*dx, bottom-py*dy, top+py*dy)
    full = np.zeros((h+2*py, w+2*px, 4), dtype=np.uint8)
    full[py:py+h, px:px+w] = rgba
    return full, expanded


def rgb_from_dn(dn):
    refl = dn.astype(np.float32) * np.float32(.0001) - np.float32(.1)
    rgb = np.clip((refl-RGB_MIN)/(RGB_MAX-RGB_MIN),0,1) ** RGB_POWER
    return np.rint(np.moveaxis(rgb,0,-1)*255).astype(np.uint8)


def fetch_context(extent, shape, crs, outside, date_string):
    """Mosaic only same-datatake public RGB assets onto a coarse display grid."""
    from pystac_client import Client
    h,w = shape
    left,right,bottom,top = extent
    transform = from_bounds(left,bottom,right,top,w,h)
    bbox = transform_bounds(crs, 'EPSG:4326', left,bottom,right,top,densify_pts=21)
    stamp = ACQUISITIONS.get(date_string)
    require(stamp is not None, 'No verified same-datatake context configured for this date')
    day = datetime.strptime(date_string,'%Y%m%d').strftime('%Y-%m-%d')
    print(f'  Querying public Earth Search for {day} / datatake {stamp}...',flush=True)
    items = list(Client.open(STAC_URL).search(collections=[COLLECTION], bbox=bbox,
                         datetime=f'{day}T00:00:00Z/{day}T23:59:59Z').items())
    items = [it for it in items if f'_{stamp}_' in it.id]
    require(items, 'No matching same-datatake scenes found; no alternate-date imagery substituted')
    items.sort(key=lambda it: (float(it.properties.get('eo:cloud_cover',100) or 0),it.id))
    context = np.zeros((h,w,4),dtype=np.uint8)
    used = []
    with rasterio.Env(AWS_NO_SIGN_REQUEST='YES', GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
                      GDAL_HTTP_MAX_RETRY='3', GDAL_HTTP_RETRY_DELAY='2', GDAL_HTTP_TIMEOUT='90'):
        for item in items:
            if np.all(context[...,3][outside] != 0):
                break
            print(f'  Context scene: {item.id}',flush=True)
            # All three channels must come from the same source scene.
            channels=[]
            for key in ('red','green','blue'):
                asset = item.assets[key]
                bands = asset.extra_fields.get('raster:bands',[])
                require(len(bands)==1 and np.isclose(bands[0].get('scale',np.nan),.0001) and
                        np.isclose(bands[0].get('offset',np.nan),-.1),
                        f'Unexpected or missing reflectance encoding: {item.id}/{key}')
                with rasterio.open(asset.href) as ds:
                    require(ds.count==1 and ds.crs is not None and ds.dtypes[0]=='uint16',
                            f'Unexpected RGB source: {item.id}/{key}')
                    with WarpedVRT(ds,crs=crs,transform=transform,width=w,height=h,
                                   src_nodata=0,nodata=0,resampling=Resampling.average) as vrt:
                        channels.append(vrt.read(1))
            dn = np.stack(channels)
            take = outside & (context[...,3]==0) & np.all(dn!=0,axis=0)
            if take.any():
                rgb=rgb_from_dn(dn)
                context[take,:3]=rgb[take]
                context[take,3]=255
                used.append(item.id)
            print(f'    New context display pixels: {take.sum():,}',flush=True)
    require(used, 'No valid surrounding RGB pixels retrieved')
    return context, used


def add_surrounding_context(rgba, extent, crs, municipality, date_string, processed):
    full, expanded = context_grid(rgba, extent)
    h,w = full.shape[:2]
    left,right,bottom,top = expanded
    transform = from_bounds(left,bottom,right,top,w,h)
    inside = geometry_mask([mapping(municipality)],out_shape=(h,w),transform=transform,invert=True)
    outside = ~inside
    # Cache is display-only, separate from all analysis inputs.
    cache = processed / f'anchorage_{date_string}_overview_context_v2.npz'
    context = None
    if cache.exists():
        with np.load(cache,allow_pickle=False) as data:
            if (data['rgba'].shape==full.shape and np.allclose(data['extent'],expanded,rtol=0,atol=1e-6)
                and str(data['stamp'].item())==ACQUISITIONS.get(date_string)):
                context = data['rgba']
                used = data['scenes'].tolist()
        if context is not None:
            print(f'  Reusing context cache: {cache.name}',flush=True)
    if context is None:
        context, used = fetch_context(expanded,(h,w),crs,outside,date_string)
        temp=cache.with_name(cache.stem+'.tmp.npz')
        try:
            np.savez_compressed(temp,rgba=context,extent=expanded,stamp=ACQUISITIONS[date_string],scenes=np.array(used))
            temp.replace(cache)
        finally:
            temp.unlink(missing_ok=True)
    # The original local municipality RGB always wins inside the boundary.
    # No substitution of outside imagery for missing scientific pixels.
    full[outside] = context[outside]
    valid = outside & (full[...,3]!=0)
    full[valid,:3]=np.rint(full[valid,:3].astype(np.float32)*CONTEXT_DIM).astype(np.uint8)
    coverage=100*int(valid.sum())/int(outside.sum())
    print(f'  Surrounding RGB coverage: {coverage:.2f}% of outside display pixels',flush=True)
    print(f'  Context source scenes: {len(used)}',flush=True)
    if coverage < 99.9:
        print('  Some context pixels have no same-datatake imagery; explicitly marked unavailable.',flush=True)
    # Mark all unavailable imagery, not pretend gray is water.
    missing=full[...,3]==0
    yy,xx=np.indices((h,w),sparse=True)
    hatch=((xx+yy)//9)%2==0
    full[missing,:3]=(47,57,64)
    full[missing & hatch,:3]=(63,74,82)
    full[missing,3]=255
    return full, expanded, coverage, used


def boundary(ax, geom, color, width, alpha=1., linestyle='solid', z=3):
    gpd.GeoSeries([geom], crs='EPSG:3338').boundary.plot(
        ax=ax, color=color, linewidth=width, alpha=alpha, linestyle=linestyle, zorder=z,
        aspect=None)


def add_scale_and_north(ax, extent, crs, municipality):
    left, right, bottom, top = extent
    dx, dy = right-left, top-bottom
    length_km = 20 if dx >= 85000 else max(k for k in (0.1, .2, .5, 1, 2, 5, 10, 20)
                    if k * 1000 <= dx * .22)
    x0, y0 = left + dx*.055, bottom + dy*.065
    length = length_km * 1000
    # The equal-area map's scale is projected grid distance, not exact geodesic distance.
    effect = [patheffects.withStroke(linewidth=3.5, foreground=BG)]
    ax.plot([x0,x0+length], [y0,y0], color=FG, lw=2.2, zorder=9, path_effects=effect)
    for x in (x0,x0+length/2,x0+length):
        ax.plot([x,x], [y0-dy*.005,y0+dy*.005], color=FG, lw=1.6, zorder=9, path_effects=effect)
    ax.text(x0+length/2, y0+dy*.014, f'{length_km:g} km (grid)', ha='center', va='bottom',
            fontsize=9, color=FG, path_effects=effect, zorder=9)
    # True north at the arrow location, using a short geodesic due north.
    # Use the upper-left context, well outside the municipality in Anchorage.
    x, y = left + dx*.065, bottom + dy*.84
    require(not municipality.covers(Point(x,y)), 'North arrow anchor overlaps municipality')
    inv = Transformer.from_crs(crs, 4326, always_xy=True)
    fwd = Transformer.from_crs(4326, crs, always_xy=True)
    lon, lat = inv.transform(x, y)
    lon2, lat2, _ = Geod(ellps='WGS84').fwd(lon, lat, 0, 1000)
    nx, ny = fwd.transform(lon2, lat2)
    vx, vy = nx-x, ny-y
    norm = math.hypot(vx,vy)
    vx, vy = vx/norm*dy*.072, vy/norm*dy*.072
    arrow = FancyArrowPatch((x,y), (x+vx,y+vy), arrowstyle='-|>', mutation_scale=16,
                            color=FG, linewidth=1.8, zorder=10, path_effects=effect)
    ax.add_patch(arrow)
    ax.text(x+vx, y+vy+dy*.013, 'N', color=FG, fontsize=11, weight='bold',
            ha='center', va='bottom', zorder=10, path_effects=effect)


def add_callouts(ax, g):
    # Fixed Anchorage-specific callout positions; anchor points are actual polygon interiors.
    # Keeping label positions stable is useful for future date-to-date visual comparisons.
    locations = [(0.045,.48), (.15,.84), (.66,.12)]
    labels = ['Anchorage Bowl', 'Chugiak – Eagle River', 'Girdwood – Alyeska']
    for i, (pos, label) in enumerate(zip(locations, labels)):
        p = g.geometry.iloc[i].representative_point()
        ax.annotate(label, xy=(p.x,p.y), xycoords='data', xytext=pos, textcoords='axes fraction',
                    color=COLORS[i], fontsize=10, fontweight='bold', ha='left', va='center',
                    bbox=dict(boxstyle='round,pad=.38', fc=BG, ec=COLORS[i], alpha=.96, lw=.8),
                    arrowprops=dict(arrowstyle='-', color=COLORS[i], lw=1.2, shrinkA=4, shrinkB=2),
                    zorder=12, annotation_clip=False)
        ax.scatter([p.x],[p.y], s=10, color=COLORS[i], zorder=11)


def render_map(rgba, extent, crs, regions, municipality, land, areas, date, output, coverage):
    plt.rcParams.update({'font.family':'DejaVu Sans', 'savefig.facecolor':BG,
                         'figure.facecolor':BG, 'text.color':FG})
    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    fig.text(.035,.955,'ANCHORAGE THROUGH THE SEASONS',fontsize=10,weight='bold',color='#77C5D5')
    fig.text(.035,.902,'Anchorage: A municipality of contrasts',fontsize=29,weight='bold')
    fig.text(.037,.866,'True-color Sentinel-2 imagery  |  Municipality of Anchorage, Alaska',fontsize=13,color=MUTED)
    fig.text(.968,.949,date.strftime('%B %d, %Y').upper(),ha='right',fontsize=13,weight='bold')
    fig.text(.968,.918,'SOUTH-CENTRAL ALASKA  /  MAP 01',ha='right',fontsize=10,color=MUTED)
    ax = fig.add_axes([.035,.115,.655,.71], facecolor='#293A45')
    ax.imshow(rgba, extent=extent, origin='upper', interpolation='nearest', zorder=1)
    # Exactly the water excluded from the four verified study regions.
    # Draw it independently of region ownership; interior holes are water-blue,
    # not the neighboring community color. The outline is clipped to the municipality.
    water = municipality.difference(land)
    gpd.GeoSeries(list(polygon_parts(water)),crs=crs).plot(
        ax=ax,facecolor=WATER_LINE,edgecolor='none',alpha=.15,zorder=3,aspect=None)
    water_edge = water.boundary
    for i in range(3):
        edge = regions.geometry.iloc[i].boundary.difference(water_edge.buffer(.5))
        if not edge.is_empty:
            gpd.GeoSeries([edge],crs=crs).plot(ax=ax,color=COLORS[i],linewidth=1.65,zorder=5,aspect=None)
    boundary(ax, water, WATER_LINE, .85, .95, 'solid', 6)
    boundary(ax, municipality, FG, 1.5, 1., (0,(4,3)), 7)
    ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:]); ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(''); ax.set_ylabel('')
    for spine in ax.spines.values():
        spine.set_color('#50636E'); spine.set_linewidth(.8)
    add_scale_and_north(ax, extent, crs, municipality)
    add_callouts(ax, regions)
    # Sidebar: calculated areas, not official census land-area statistics.
    panel = fig.add_axes([.72,.12,.25,.70]); panel.set_axis_off()
    total = float(areas.sum()); developed = float(areas[:3].sum())
    panel.text(0,1.,'THE STUDY AREA',fontsize=11,weight='bold',color='#77C5D5',va='top')
    panel.text(0,.931,f'{total:,.2f} km²',fontsize=26,weight='bold',va='top')
    panel.text(0,.866,'Calculated municipal land',fontsize=11,color=MUTED,va='top')
    panel.text(0,.805,'Four fixed, mapped-water-excluded regions',fontsize=11,va='top')
    for i,(name,area,color) in enumerate(zip(DISPLAY_NAMES,areas,COLORS)):
        y = .735-i*.126
        if i < 3:
            panel.plot([0,.065],[y-.017,y-.017],color=color,lw=3,solid_capstyle='round',
                       transform=panel.transAxes,clip_on=False)
        panel.text(.09,y,name,fontsize=11.5,weight='bold',color=color,va='top',transform=panel.transAxes)
        panel.text(.09,y-.046,f'{area:,.2f} km²   •   {area/total*100:.2f}%',fontsize=10.5,color=MUTED,
                   va='top',transform=panel.transAxes)
    panel.text(.09,.265,'All other land inside the boundary',fontsize=9,color=MUTED,va='top',transform=panel.transAxes)
    panel.plot([0,1],[.229,.229],color='#40535F',lw=.8,transform=panel.transAxes)
    panel.text(0,.203,f'{developed/total*100:.2f}%  developed study extents',fontsize=11,weight='bold',va='top',transform=panel.transAxes)
    panel.text(0,.155,'Approximate community extents, not\nimpervious surface or building footprints.',fontsize=10,color=MUTED,va='top',linespacing=1.4,transform=panel.transAxes)
    panel.plot([0,.065],[.060,.060],color=FG,lw=1.2,ls='--',transform=panel.transAxes)
    panel.text(.09,.060,'Municipality of Anchorage boundary',fontsize=9.0,va='center',color=MUTED,transform=panel.transAxes)
    panel.plot([0,.065],[.015,.015],color=WATER_LINE,lw=1.2,transform=panel.transAxes)
    panel.text(.09,.015,'Mapped water excluded from areas',fontsize=9.0,va='center',color=MUTED,transform=panel.transAxes)
    # All first panel text coordinates are axes fractions, unaffected by legend lines.
    for text in panel.texts:
        text.set_transform(panel.transAxes)
    fig.add_artist(Line2D([.035,.968],[.088,.088],transform=fig.transFigure,color='#40535F',lw=.8))
    fig.text(.035,.060,f'Contains modified Copernicus Sentinel data {date.year}  •  Earth Search / Element 84  •  U.S. Census TIGER/Line 2025 hydrography',fontsize=9,color=MUTED)
    fig.text(.035,.035, 'EPSG:3338  |  Source RGB: 10 m; display downsampled  |  Outside boundary: dimmed same-acquisition imagery',fontsize=9,color=MUTED)
    ax.text(.99,.015, ('Hatched gray: imagery unavailable' if coverage<99.999 else 'Outside boundary: dimmed context'), transform=ax.transAxes,ha='right',va='bottom',fontsize=8,color=FG,bbox=dict(fc=BG,ec='none',alpha=.85,pad=3),zorder=13)
    fig.text(.968,.035,'GEOGRAPHIC OVERVIEW',ha='right',fontsize=9,weight='bold',color='#77C5D5')
    try:
        fig.savefig(output,dpi=240,facecolor=BG)
    finally:
        plt.close(fig)


def run(project, date_string=DATE, overwrite=False):
    date = datetime.strptime(date_string, '%Y%m%d')
    require(date.strftime('%Y%m%d') == date_string, 'Date must be YYYYMMDD')
    processed = project / 'data' / 'processed'
    spectral = processed / f'anchorage_{date_string}_spectral_10m.tif'
    sources = processed / f'anchorage_{date_string}_sources.json'
    paths = (spectral, sources, processed/'anchorage_land_study_regions.geojson', processed/'anchorage_municipality.geojson')
    for p in paths:
        require(p.is_file(), f'Missing input: {p}')
    json.loads(sources.read_text(encoding='utf-8-sig'))
    output = project/'portfolio'/f'anchorage_{date_string}_geographic_overview_v2.png'
    require(overwrite or not output.exists(), f'Output exists: {output}. Use --overwrite to replace only this PNG.')
    print('Reading true-color imagery (display resolution only)...',flush=True)
    rgba, extent, crs = read_true_color(spectral)
    print('Loading and verifying study regions...',flush=True)
    g, municipality, land, areas = load_regions(processed, crs)
    for name, area in zip(REGIONS, areas):
        print(f'  {name}: {area:,.3f} km²')
    print('Retrieving surrounding same-acquisition RGB context...',flush=True)
    rgba, extent, coverage, used = add_surrounding_context(rgba, extent, crs, municipality, date_string, processed)
    print('Rendering portfolio map...',flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    temp = output.with_name(output.stem+'.tmp.png')
    try:
        render_map(rgba, extent, crs, g, municipality, land, areas, date, temp, coverage)
        require(temp.stat().st_size > 0, 'Empty output')
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)
    print(f'Created: {output} ({output.stat().st_size/1e6:.2f} MB)')
    print('Complete. Existing inputs and QGIS project unchanged. No private AWS operations.')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date',default=DATE,help='Acquisition date YYYYMMDD')
    parser.add_argument('--overwrite',action='store_true',help='Replace existing overview PNG only')
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[1], args.date, args.overwrite)
