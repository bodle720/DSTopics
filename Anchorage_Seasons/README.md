# Anchorage Through the Seasons

**Cloud-Native Geospatial Analysis Using Sentinel-2**

A personal geospatial case study exploring seasonal vegetation greenness and snow and ice throughout the Municipality of Anchorage, Alaska, using eight selected observations from 2025. The investigation covers four fixed, mapped-water-excluded study regions totaling approximately 4,359 km². It is a descriptive remote-sensing study, not a trained machine learning model.

## Explore the results

- [Read the complete Jupyter notebook](Anchorage_Seasons.ipynb): embedded maps, charts, scientific results, methodology, equations and saved example outputs. No satellite downloads are required to read it on GitHub.
- [Download the regional statistics CSV](results/anchorage_2025_seasonal_regional_statistics.csv): 40 observations (eight dates × five reporting categories, including the municipal total), 50 fields.
- [Browse the processing scripts](scripts/).
- [Explore the reproducibility inputs and historical acquisition records](reproducibility/README.md).

Municipal mean NDVI reached 0.741 on July 27 among usable, conservatively snow-free observations. The lowest observed municipal snow/ice proxy was 19.24% of valid NDSI observations on September 30. Coverage varies substantially by date and region: on March 29, just 342 of 46,077 Girdwood–Alyeska land cells were usable, despite complete assigned imagery. These conditional measurements should not be treated as estimates for unobserved land.

## Technologies

QGIS, GeoPandas, Shapely, pyproj, Rasterio, NumPy, pandas, Matplotlib, Pillow, STAC / pystac-client, Sentinel-2 multispectral imagery, GeoTIFF and Cloud Optimized GeoTIFF (COG), AWS S3 / Boto3, ESRI shapefiles, GeoJSON, and HTML/CSS/JavaScript for the separately generated interactive statistics viewer.

## Repository layout

```text
Anchorage_Seasons/
├── Anchorage_Seasons.ipynb      # GitHub-readable scientific walkthrough
├── README.md
├── requirements.txt
├── .gitignore
├── results/
│   └── anchorage_2025_seasonal_regional_statistics.csv
├── scripts/                    # 29 source scripts
└── reproducibility/            # 27 original small input and audit files
    ├── README.md
    ├── geometries/
    ├── acquisitions/
    └── methodology/
```

The original local working directory also contains `data/`, `portfolio/`, and `alaska_refresher.qgz`. The large source imagery, derived rasters, cached datasets, QGIS project configuration, full-resolution visualizations, virtual environments and IDE metadata are intentionally excluded from the public repository. Notebook images are embedded publication-size copies; the original maps and GIFs are not changed.

## Read or run the notebook

The notebook is pre-executed and can be read directly in GitHub. To rerun its three small CSV examples, launch Jupyter from this project directory with Python, pandas and a Jupyter kernel installed. The notebook does not call AWS, download imagery or perform the full scientific processing.

For the processing scripts, the original project used Python 3.11 and QGIS. Set up an environment and install dependencies from this directory:

```shell
python -m pip install -r requirements.txt
```

Use a dedicated environment for a new installation. Jupyter Notebook or JupyterLab and an IPython kernel are additional requirements for interactive notebook execution. QGIS Desktop is installed separately. The historical dependency list is not a lockfile; compatibility with a new environment should be tested.

## Processing workflow

The scripts are organized chronologically. The first group comprises learning exercises; not every exercise is a prerequisite for the final analysis. The later stages reuse verified intermediate files and may require existing local source datasets, manually created geometries, and public catalog access.

| Scripts | Purpose |
|---|---|
| 01–04 | Inspect Alaska shapefiles and airports; construct an airport buffer and nearby-airport query. |
| 05–06 | Inspect raster metadata and compare ordinary TIFF with GeoTIFF. |
| 07–09 | Discover Sentinel-2 imagery, retrieve initial scenes, produce RGB/false color and NDVI. |
| 10–13 | Retrieve Anchorage municipality boundary, discover covering scenes, mosaic June imagery and produce a COG. |
| 14 | Optional demonstration of remote S3 COG windowed access using user-supplied AWS configuration. |
| 15–16 | Analyze community polygons, subtract mapped water, create and validate fixed land regions. |
| 17–18 | Prepare June spectral/quality rasters and calculate NDVI/NDSI and regional statistics. |
| 19–21 | Generate geographic overview, vegetation and snow/ice maps. |
| 22–24 | Discover 2025 acquisitions and screen 62 footprint-qualified groups for observation quality. |
| 25–26 | Prepare and analyze the eight selected acquisitions, validate aligned grids, consolidate statistics. |
| 27 | Generate the original synchronized three-panel GIF (retained as an earlier design). |
| 27b | Generate the three final separate seasonal animations with statewide geographic context. |
| 28 | Generate four charts and the standalone interactive HTML statistics viewer. |

### Reproduction notes

A full end-to-end rebuild from a fresh clone is **not currently turnkey**, but the exact manually drawn community polygons, fixed water-excluded study regions, historical acquisition discovery and screening summaries, source-scene manifests and scientific methodology records are included under [`reproducibility/`](reproducibility/README.md). These 27 files preserve the original bytes. The separate [reproducibility guide](reproducibility/README.md) explains how to restore selected inputs to the ignored local processing directory. Original raster data, per-acquisition screening caches and QGIS sample data are not distributed; rebuild or retrieve those separately as required. Do not substitute newly drawn polygons or different imagery and describe the results as an exact replication.

Local input locations expected by the processing scripts include:

- `data/qgis_sample_data/` — QGIS Alaska sample dataset for early exercises and the statewide animation context.
- `data/shapefiles/` — extracted 2025 TIGER/Line Anchorage area-water shapefile and associated sidecar files (`tl_2025_02020_areawater.shp`).
- `data/processed/anchorage_study_areas.geojson` — original community study polygons prepared in QGIS; required by scripts 15–16.
- `data/processed/anchorage_municipality.geojson` — municipal boundary generated by script 10.
- `data/processed/anchorage_s2_cog_10m.tif` — four-band June grid reference produced by scripts 12–13.
- `data/processed/anchorage_land_study_regions.geojson` — fixed land-region geometry produced by script 16.
- The 2025 acquisition inventory from script 22, local June spectral/quality COGs from script 17, and subsequent seasonal outputs as required by the later scripts.

The published summary CSV is under `results/` for convenient viewing. Script 28 expects it at `data/processed/anchorage_2025_seasonal_regional_statistics.csv`; to regenerate only the charts and interactive HTML, copy the published CSV into that local location and run `python scripts/28_create_seasonal_charts.py` after installing the dependencies. This does not require satellite imagery.

The original workflow used several gigabytes of local storage plus staging space. Public catalog contents and service interfaces may change, so historical discovery results are not guaranteed to repeat. Consult individual scripts for specific inputs, arguments and output handling before execution. The original run reported successful geometry checks, full-resolution COG equality, grid checks and regional accounting; those raster checks were not rerun for this documentation package.

The optional remote access demo is independent of the seasonal statistics. Provide your own accessible COG and authenticated AWS setup, e.g.:

```shell
python scripts/14_read_remote_cog.py --source s3://YOUR-BUCKET/YOUR-COG.tif --profile YOUR-PROFILE
```

No private bucket identifiers or credentials are required to read the published notebook. AWS use may incur costs; the seasonal analysis used public source imagery and local processing.

## Scientific methodology and limitations

The fixed 10 m EPSG:3338 raster grid supports projected area calculations. B11 and SCL are native 20 m data resampled to this grid; resampling does not improve their independent measurement resolution. Reflectance is decoded as `DN × 0.0001 − 0.1` before computing NDVI and NDSI. NDSI snow/ice requires NDSI ≥ 0.40 and green reflectance ≥ 0.10; the green proxy requires NDVI ≥ 0.30. SCL classes 4, 5 and 11 are initially eligible; NDVI additionally requires valid NDSI and excludes snow indicated by either rule or SCL. Invalid observations remain missing. The notebook explains all key denominators and the coverage limitations.

These thresholds are exploratory, and SCL comparison is not independent ground truth. The green proxy is not fractional vegetation cover. Snow and ice are not separated, observed areas are not extrapolated to unobserved land, and the study does not establish causal effects of urbanization. Eight discrete dates are not daily or continuous monthly data.

## Data sources and attribution

- **Contains modified Copernicus Sentinel data 2025.** [Copernicus Sentinel data legal notice](https://ewds.climate.copernicus.eu/licences/ec-sentinel).
- Sentinel-2 Collection 1 Level-2A imagery: [Earth Search / Element 84](https://earth-search.aws.element84.com/v1) (`sentinel-2-c1-l2a`), [AWS Open Data registry](https://registry.opendata.aws/sentinel-2-l2a-cogs/).
- Municipality boundary and derived mapped-water and land geometries: U.S. Census Bureau [TIGERweb](https://tigerweb.geo.census.gov/); area-water geometry: [2025 TIGER/Line shapefiles](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.2025.html). The derived study-region products are independently prepared, not official Census Bureau products; original TIGER/Line data are available free of charge from the Census Bureau.
- Earlier vector exercises and the separate animation locator context: [QGIS Alaska sample dataset](https://github.com/qgis/QGIS-Sample-Data). The upstream project identifies the Alaska political boundary as adapted VMAP0 data and notes multiple licenses across the sample collection. The raw sample dataset and separate animation files are not redistributed in this package. Consult upstream terms before any additional redistribution.

## Development approach

I directed the project, selected and reviewed the scientific methodology, executed the workflows, inspected outputs and investigated observation-quality issues. AI-assisted code generation was used extensively. The source scripts are included for transparency and technical review; this project is not a claim of independently hand-writing every line.
