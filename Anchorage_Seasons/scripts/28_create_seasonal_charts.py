"""Anchorage Through the Seasons — publication charts and interactive data table.
Run from PyCharm or: python scripts/28_create_seasonal_charts.py
Reads Script 26's local CSV. No satellite, AWS or website operations.
Requires matplotlib. All output is staged in a new portfolio directory.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

DATES = ('20250207','20250329','20250523','20250612','20250727','20250811','20250930','20251119')
REGIONS = ('Anchorage Bowl','Chugiak - Eagle River','Girdwood - Alyeska','Remaining Municipal Land','All Municipal Land')
LABELS = ('Anchorage Bowl','Chugiak–Eagle River','Girdwood–Alyeska','Remaining municipal land','All municipal land')
COLORS = ('#FFD166','#54E4E5','#FF92AA','#A6C3D3','#F4F7F8')
BG, FG, MUTED, ACCENT, GRID = '#101A22','#F4F7F8','#B3C3CD','#77C5D5','#40535F'
SNOW, SCOV = 'snow_ice_pct_of_snow_usable','snow_usable_pct_of_land'
NDVI, GREEN, VCOV = 'ndvi_mean','green_pct_of_ndvi_usable','ndvi_usable_pct_of_land'
CORE = (SNOW,SCOV,NDVI,GREEN,VCOV)
EXTRA = ('scl_snow_pct_on_same_snow_usable','ndsi_scl_agreement_pct','cloud_pct_of_land','shadow_pct_of_land','snow_ice_observed_km2','green_observed_km2','ndvi_usable_pct_of_snow_free_candidate')
COUNTS = ('land_pixels','ndsi_valid_pixels','ndsi_snow_pixels','ndvi_valid_pixels','green_pixels','snow_free_candidate_pixels','scl_snow_on_ndsi_valid_pixels','snow_agreement_pixels','cloud_pixels','shadow_pixels')
RATIOS = {SNOW:('ndsi_snow_pixels','ndsi_valid_pixels'), SCOV:('ndsi_valid_pixels','land_pixels'), GREEN:('green_pixels','ndvi_valid_pixels'), VCOV:('ndvi_valid_pixels','land_pixels'), EXTRA[0]:('scl_snow_on_ndsi_valid_pixels','ndsi_valid_pixels'),EXTRA[1]:('snow_agreement_pixels','ndsi_valid_pixels'),EXTRA[2]:('cloud_pixels','land_pixels'),EXTRA[3]:('shadow_pixels','land_pixels'),EXTRA[6]:('ndvi_valid_pixels','snow_free_candidate_pixels')}
COVERAGE_GUIDE = 50.0  # Display-only flag, not a validated quality threshold.

def require(ok,msg):
    if not ok: raise ValueError(msg)

def number(x):
    if x is None or x.strip()=='': return None
    v=float(x); require(math.isfinite(v),'Nonfinite numeric value'); return v

def read_data(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f); fields=reader.fieldnames or []
        require(set(('date','region','vector_land_km2','raster_land_km2',*CORE,*EXTRA,*COUNTS))<=set(fields),'CSV schema differs from Script 26')
        raw=list(reader)
    require(len(raw)==40,'Expected exactly 40 rows')
    data={d:{} for d in DATES}
    for r in raw:
        d,n=r['date'],r['region']; require(d in data and n in REGIONS and n not in data[d],f'Unexpected or duplicate key: {d}/{n}')
        row={k:number(r[k]) for k in (*CORE,*EXTRA,'vector_land_km2','raster_land_km2')}
        for k in COUNTS:
            row[k]=int(r[k]); require(row[k]>=0,f'Negative count: {d}/{n}/{k}')
        N,S,s,V,g,F,cs,agree,cl,sh=(row[k] for k in COUNTS)
        require(0<=g<=V<=F<=S-s<=S<=N and N>0 and 0<=cs<=S and 0<=agree<=S and cl+sh<=N,f'Invalid counts: {d}/{n}')
        for k,(num,den) in RATIOS.items():
            x=row[k]; expected=100*row[num]/row[den] if row[den] else None
            require((x is None if expected is None else x is not None and 0<=x<=100 and abs(x-expected)<.00002),f'Invalid ratio: {d}/{n}/{k}')
        x=row[NDVI]; require((x is None if not V else x is not None and -1<=x<=1),f'Invalid NDVI: {d}/{n}')
        for k,c in ((EXTRA[4],s),(EXTRA[5],g),('raster_land_km2',N)):
            require(row[k] is not None and abs(row[k]-c*.0001)<.000002,f'Area mismatch: {d}/{n}/{k}')
        require(row['vector_land_km2'] is not None and row['vector_land_km2']>0,'Invalid vector area')
        data[d][n]=row
    for d,rows in data.items():
        require(set(rows)==set(REGIONS),f'Missing regions: {d}')
        total=rows[REGIONS[-1]]
        for k in COUNTS:
            require(sum(rows[n][k] for n in REGIONS[:-1])==total[k],f'Total mismatch: {d}/{k}')
        if total['ndvi_valid_pixels']:
            mean=sum((rows[n][NDVI] or 0)*rows[n]['ndvi_valid_pixels'] for n in REGIONS[:-1])/total['ndvi_valid_pixels']
            require(abs(mean-total[NDVI])<.000002,f'Weighted NDVI mismatch: {d}')
        for n in REGIONS:
            for k in ('land_pixels','vector_land_km2','raster_land_km2'):
                require(rows[n][k]==data[DATES[0]][n][k],f'Fixed region changed: {d}/{n}/{k}')
    print('40 rows, fixed regions, denominators, observed areas and weighted means: PASS')
    return data


def setup():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'text.color':FG,'axes.labelcolor':FG,'xtick.color':MUTED,'ytick.color':MUTED,'axes.facecolor':BG,'figure.facecolor':BG,'savefig.facecolor':BG,'axes.edgecolor':GRID})

def axis(ax,lim,label):
    ax.set_xlim(datetime(2025,1,1),datetime(2025,12,31)); ax.set_ylim(*lim)
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.set_ylabel(label,labelpad=10); ax.grid(axis='y',color=GRID,linewidth=.7,alpha=.65); ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False); ax.tick_params(length=0,pad=8)

def draw(ax,data,key,coverage=None):
    xs=[datetime.strptime(d,'%Y%m%d') for d in DATES]
    # Municipality aggregate overlaps the regions; visually distinguished with a dashed line.
    for n,label,color in zip(REGIONS,LABELS,COLORS):
        vals=[data[d][n][key] for d in DATES]; ys=[float('nan') if x is None else x for x in vals]
        ax.plot(xs,ys,color=color,lw=1.6,ls='--' if n==REGIONS[-1] else '-',alpha=.8,zorder=2)
        for d,x,y in zip(DATES,xs,vals):
            if y is None: continue
            low=coverage is not None and data[d][n][coverage]<COVERAGE_GUIDE
            ax.plot(x,y,marker='o' if n!=REGIONS[-1] else 'D',ms=7,ls='none',mec=color,mew=1.6,mfc=BG if low else color,zorder=4,clip_on=False)

def legend(fig):
    handles=[Line2D([0],[0],color=c,marker='D' if i==4 else 'o',ls='--' if i==4 else '-',lw=1.6,label=n) for i,(c,n) in enumerate(zip(COLORS,LABELS))]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.071,.845),ncol=3,frameon=False,labelcolor=FG,fontsize=11,columnspacing=1.6)

def heading(fig,title,sub):
    fig.text(.075,.95,title,fontsize=27,weight='bold',va='top'); fig.text(.965,.95,'2025',fontsize=19,weight='bold',color=ACCENT,ha='right',va='top'); fig.text(.075,.887,sub,fontsize=12,color=MUTED,va='top')

def footer(fig,lines):
    fig.text(.075,.025,lines,fontsize=10,color=MUTED,va='bottom',linespacing=1.45)

def save_chart(data,out,key,title,sub,ylim,ylabel,coverage,notes):
    fig,ax=plt.subplots(figsize=(13,7.5)); fig.subplots_adjust(left=.10,right=.965,top=.72,bottom=.19)
    heading(fig,title,sub); legend(fig); axis(ax,ylim,ylabel); draw(ax,data,key,coverage)
    footer(fig,notes+'\nOpen markers: less than 50% of region land usable for this index (display guide only). Lines connect discrete observations.')
    try: fig.savefig(out,dpi=160)
    finally: plt.close(fig)

def coverage_chart(data,out):
    fig,axs=plt.subplots(2,1,figsize=(13,9),sharex=True); fig.subplots_adjust(left=.10,right=.965,top=.72,bottom=.15,hspace=.29)
    heading(fig,'What the satellite can observe','Usable observations as a percentage of the fixed mapped-water-excluded land area'); legend(fig)
    for ax,k,label in zip(axs,(SCOV,VCOV),('NDSI usable land (%)','NDVI usable land (%)')):
        axis(ax,(0,100),label); draw(ax,data,k); ax.axhline(50,color=MUTED,ls=':',lw=1,alpha=.6)
    footer(fig,'NDSI excludes clouds, shadows, water and invalid measurements. NDVI also excludes both snow indicators.\nLow NDVI coverage in winter can reflect snow exclusion, not poor imagery. Dotted line: 50% display guide, not a scientific cutoff.')
    try: fig.savefig(out,dpi=160)
    finally: plt.close(fig)

HTML = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Anchorage | Explore the regional statistics</title>
<style>
:root{color-scheme:dark}body{margin:0;background:#101a22;color:#f4f7f8;font:16px/1.55 system-ui,sans-serif}#anchorage-data{max-width:1200px;margin:auto;padding:clamp(16px,4vw,40px)}#anchorage-data h1{font-size:clamp(25px,4vw,38px);line-height:1.2;margin:0 0 8px}#anchorage-data h2{font-size:21px;margin:24px 0 8px}#anchorage-data p{color:#b3c3cd}#anchorage-data .dates{display:flex;gap:8px;flex-wrap:wrap;margin:24px 0}#anchorage-data button{font:inherit;background:#1d2b35;color:#f4f7f8;border:1px solid #50636e;border-radius:7px;padding:9px 13px;cursor:pointer}#anchorage-data button[aria-pressed=true]{background:#77c5d5;color:#101a22;border-color:#77c5d5;font-weight:700}#anchorage-data button:focus-visible,#anchorage-data summary:focus-visible{outline:3px solid #ffd166;outline-offset:3px}#anchorage-data .scroll{overflow-x:auto}#anchorage-data table{border-collapse:collapse;width:100%;min-width:720px;font-variant-numeric:tabular-nums}#anchorage-data th,#anchorage-data td{padding:12px 10px;border-bottom:1px solid #40535f;text-align:right}#anchorage-data th:first-child{text-align:left;min-width:170px}#anchorage-data thead{color:#b3c3cd;font-size:14px}#anchorage-data tbody tr:last-child{font-weight:700;background:#1d2b35}#anchorage-data .note{font-size:14px}#anchorage-data .flag{color:#ffd166}#anchorage-data details{margin:24px 0}#anchorage-data summary{cursor:pointer;color:#77c5d5;font-weight:600}#anchorage-data caption{text-align:left;font-size:14px;color:#b3c3cd;margin-bottom:8px}
</style></head><body><section id="anchorage-data"><h1>Explore the regional statistics</h1><p>Choose one of the eight Sentinel-2 acquisition dates in 2025.</p><div class="dates" role="group" aria-label="Acquisition date"></div><h2 id="ad-date" aria-live="polite"></h2><div class="scroll"><table id="ad-core"><caption>Snow and vegetation measurements with their observation coverage</caption><thead></thead><tbody></tbody></table></div><p class="note">¹ Of NDSI-usable land. ² Of all region land. ³ Of usable, conservatively snow-free NDVI land; green proxy uses NDVI ≥ 0.30. <span class="flag">*</span> Less than 50% of total region land usable for the corresponding index (display guide only). Low NDVI coverage can be due to snow exclusion. N/A means no valid denominator, not zero.</p><details><summary>Additional measurements and quality information</summary><div class="scroll"><table id="ad-extra"><caption>Observed areas are not extrapolated to unobserved land</caption><thead></thead><tbody></tbody></table></div><p class="note">SCL snow and agreement use the same NDSI-valid pixels. Cloud and shadow percentages use all region land. NDVI retention is of conservative snow-free candidate pixels. SCL is not independent ground truth.</p></details><p class="note">All municipal land is the combined total of the four fixed regions, not a separate fifth region. These are exploratory snow/ice and greenness proxies, not validated land-cover fractions. Snow and glacial ice are not distinguished. Eight discrete observations; coverage and visible land differ by date.</p><noscript>JavaScript is required to select dates.</noscript></section>
<script type="application/json" id="ad-data">__DATA__</script><script>
(()=>{'use strict';const root=document.getElementById('anchorage-data'), data=JSON.parse(document.getElementById('ad-data').textContent), buttons=root.querySelector('.dates');const core=__CORE__,extra=__EXTRA__;const make=(tag,text)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;return e};
function table(id,cols,rows){const t=root.querySelector(id),head=make('tr');for(const c of ['Region',...cols.map(x=>x[1])]){const h=make('th',c);h.scope='col';head.append(h)}t.tHead.replaceChildren(head);const body=t.tBodies[0];body.replaceChildren();rows.forEach((r,i)=>{const tr=make('tr'),h=make('th',r.label);h.scope='row';h.style.color=r.color;tr.append(h);cols.forEach(([key,label,digits,suffix,cov])=>{const v=r.values[key],low=v!==null&&cov&&r.values[cov]!==null&&r.values[cov]<50,td=make('td',v===null?'N/A':v.toFixed(digits)+suffix);if(low){const a=make('span',' *');a.className='flag';a.title='Less than 50% of region land usable';td.append(a)}tr.append(td)});body.append(tr)})}
function select(d){for(const b of buttons.children)b.setAttribute('aria-pressed',String(b.dataset.date===d));const item=data[d];root.querySelector('#ad-date').textContent=item.long;table('#ad-core',core,item.rows);table('#ad-extra',extra,item.rows)}
for(const [d,item] of Object.entries(data)){const b=make('button',item.short);b.type='button';b.dataset.date=d;b.addEventListener('click',()=>select(d));buttons.append(b)}select('20250612');})();
</script></body></html>'''

def html_table(data,out):
    core=[[SNOW,'Snow / ice¹',2,'%',SCOV],[SCOV,'NDSI usable²',2,'%',None],[NDVI,'Mean NDVI³',3,'',VCOV],[GREEN,'Green proxy³',2,'%',VCOV],[VCOV,'NDVI usable²',2,'%',None]]
    extra=[[k,label,2,suffix,None] for k,label,suffix in zip(EXTRA,('SCL snow','NDSI–SCL agreement','Cloud','Shadow','Observed snow / ice','Observed green','NDVI retention'),('%','%','%','%',' km²',' km²','%'))]
    payload={}
    for d in DATES:
        dt=datetime.strptime(d,'%Y%m%d'); payload[d]={'short':f'{dt:%b} {dt.day}','long':f'{dt:%B} {dt.day}, 2025','rows':[{'label':label,'color':color,'values':{k:data[d][n][k] for k in (*CORE,*EXTRA)}} for n,label,color in zip(REGIONS,LABELS,COLORS)]}
    js=lambda x:json.dumps(x,ensure_ascii=True,allow_nan=False).replace('<','\\u003c')
    out.write_text(HTML.replace('__DATA__',js(payload)).replace('__CORE__',js(core)).replace('__EXTRA__',js(extra)),encoding='utf-8')


def run(project):
    src=project/'data'/'processed'/'anchorage_2025_seasonal_regional_statistics.csv'; require(src.is_file(),f'Missing Script 26 CSV: {src}')
    out=project/'portfolio'/'anchorage_2025_seasonal_charts'; require(not out.exists(),f'Output already exists; preserved: {out}. Rename it before rerunning.')
    data=read_data(src); setup(); out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='script28_',dir=out.parent) as tmp:
        stage=Path(tmp)/'output'; stage.mkdir()
        save_chart(data,stage/'anchorage_2025_snow_ice_chart.png',SNOW,'Where winter lingers','NDSI-derived snow and ice proxy across eight acquisitions',(0,100),'Snow / ice (%)',SCOV,'Percentage of NDSI-usable land, not all land. Snow and glacial ice are not separated.')
        save_chart(data,stage/'anchorage_2025_ndvi_chart.png',NDVI,'The growing season','Mean NDVI on usable, conservatively snow-free land',(-1,1),'Mean NDVI',VCOV,'NDVI is spectral greenness, not vegetation fraction. Winter estimates may describe very small snow-free subsets.')
        save_chart(data,stage/'anchorage_2025_green_proxy_chart.png',GREEN,'A landscape of green','Share of usable NDVI observations with NDVI ≥ 0.30',(0,100),'Green proxy (%)',VCOV,'Percentage of NDVI-usable land, not all land. The threshold is exploratory, not fractional vegetation cover.')
        coverage_chart(data,stage/'anchorage_2025_observation_coverage_chart.png')
        html_table(data,stage/'anchorage_2025_interactive_statistics.html')
        manifest={'source':src.name,'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'dates':list(DATES),'regions':list(REGIONS),'rows':40,'display_coverage_guide_pct':COVERAGE_GUIDE,'missing_values':'not plotted; N/A in table','files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in stage.iterdir()}}
        (stage/'charts_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        require(not out.exists(),'Output appeared during processing; refusing to overwrite'); stage.rename(out)
    print(f'Created: {out}'); [print(f'  {p.name}: {p.stat().st_size/1e6:.2f} MB') for p in sorted(out.iterdir())]; print('Complete. Original CSV, GIFs and maps unchanged. No network or AWS operations.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1]); run(p.parse_args().project)
