# Reproducibility inputs and historical records

These 27 files are byte-for-byte copies of the inputs and records from the original 2025 Anchorage analysis. They supplement the published notebook, scripts and regional results. The original source rasters and large local caches are not included.

## Contents

| Folder | Files | Purpose |
| --- | ---: | --- |
| `geometries/` | 5 | Manually drawn community study extents, Census municipality boundary, derived mapped water, land and fixed reporting regions |
| `acquisitions/` | 14 | Six discovery/screening records and eight source-scene manifests, including public asset URLs |
| `methodology/` | 8 | Per-date reflectance, masking, grid, thresholds, geometry and scientific audit records |

The original `anchorage_land_study_regions.geojson` intentionally retains a tiny invalid geometry component. Script 18 repairs it in memory, and the original methodology records document the repair and area change. Do not silently edit this archived input.

## Restore inputs for a local run

The processing scripts expect their local inputs under `data/processed/`, not under `reproducibility/`. Keep this archive unchanged. On a **fresh clone** of the project, run these commands from `Anchorage_Seasons/` in Windows CMD to restore the five geometry files and historical discovery shortlist:

```cmd
if not exist "data\processed" mkdir "data\processed"
copy /-Y "reproducibility\geometries\*.geojson" "data\processed\"
copy /-Y "reproducibility\acquisitions\anchorage_2025_acquisition_shortlist.json" "data\processed\"
```

The `/-Y` flag prompts before overwriting existing files. Do not overwrite completed local analysis products unnecessarily. Other archived acquisition and methodology records are primarily evidence and reference files: they do not substitute for the rasters or intermediate caches they describe. Some scripts use the presence of output manifests to determine whether a run is complete; do not copy those manifests into an otherwise empty processing directory and assume the outputs exist.

See the [main README](../README.md) and individual scripts for prerequisites. In particular, scripts 12–13 and 17 create the June reference imagery required by later processing. Script 16 additionally requires the original area-water shapefile if regenerating regions; the archived fixed region file permits using the already established regions instead. Scripts 23–24 use per-acquisition screening caches and a reference-file signature (including file metadata); archived summary CSV/JSON records alone cannot restore that cache. Script 25 uses the fixed eight acquisitions and verifies the June reference. Recreating the historical screening pipeline or all raster outputs was not tested from a clean clone as part of this publication package.

## Sources and attribution

- Community study polygons were manually defined for this project; municipal boundary from [U.S. Census Bureau TIGERweb](https://tigerweb.geo.census.gov/), water exclusions derived from [2025 TIGER/Line area-water data](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.2025.html). Census data are available free of charge from the Census Bureau. The derived study regions are not official Census Bureau products.
- Contains modified Copernicus Sentinel data 2025. Public Sentinel-2 Collection 1 Level-2A scene assets discovered through [Earth Search / Element 84](https://earth-search.aws.element84.com/v1). See [Sentinel data legal notice](https://ewds.climate.copernicus.eu/licences/ec-sentinel).
- The raw QGIS Alaska sample data and animation context shapefile are not included here.

## File integrity

The SHA-256 hashes below identify the exact archived files. They are provided for comparison, not as proof of external data accuracy.

| File | SHA-256 |
| --- | --- |
| `methodology/anchorage_20250207_analysis_method.json` | `e2b21bbd69e06973b888f10bca7834f5df3959c8b925da2a77eed319470da9f0` |
| `acquisitions/anchorage_20250207_sources.json` | `3c41ad3f85c57e8eb2f5cb28a5124c4a1c68ae4374f35d7804c2b53102d6f510` |
| `methodology/anchorage_20250329_analysis_method.json` | `0aed51efa7215d7ca0dd996065e00300f050dfd2874334c06a6f23c579f96584` |
| `acquisitions/anchorage_20250329_sources.json` | `6b6a8b018d1f0852a974c18e84b1e24a88e6b4b458799fc3c82bce875a66ce45` |
| `methodology/anchorage_20250523_analysis_method.json` | `824094fcd7d065b1b15ce4715216b23e5d852217e9be2eb4b935dc44091a781a` |
| `acquisitions/anchorage_20250523_sources.json` | `42129e2b066601000e911568ca17081bed3fafc2b6c88033c006b680b947fddd` |
| `methodology/anchorage_20250612_analysis_method.json` | `1d7a35fa479b25628984e233396af1200856968298354f80739b4cacb748072e` |
| `acquisitions/anchorage_20250612_sources.json` | `c418aae4fd8fc4f80a3716b481227123cee445e8249c569b38fdd934fa225a8c` |
| `methodology/anchorage_20250727_analysis_method.json` | `3bb77e0101da7145f30c82eae5fa71ef56fcd49bd806a2fd89c1fd9260ae0205` |
| `acquisitions/anchorage_20250727_sources.json` | `46e4bd4543283ace5279d61d747e90b2592692e6d7dfd0d2e19326c0e8a96374` |
| `methodology/anchorage_20250811_analysis_method.json` | `eb170be711cb9c35fae0b438fca2c16104f5565e4857b40c81c73bcbf51e7663` |
| `acquisitions/anchorage_20250811_sources.json` | `16564ff4457f4b366606ccf722edd85347411ccaa9d2559f53582763326bf879` |
| `methodology/anchorage_20250930_analysis_method.json` | `cf52438e705273e087dce53b69a1a163f908c7d4ce316f552a62e89362902677` |
| `acquisitions/anchorage_20250930_sources.json` | `89acb338bf1753e86e8a686e4dfaa874780c4dfac2e433cff09230da44c7c5a1` |
| `methodology/anchorage_20251119_analysis_method.json` | `ffcc30e4515dff57859beebf4ae3a818947d6173001f83845152f35e2b932023` |
| `acquisitions/anchorage_20251119_sources.json` | `d2e6c54c3c485ca60dac6a6b311545fac14cac0cf8c37dc31cd56db97295626c` |
| `acquisitions/anchorage_2025_acquisition_candidates.csv` | `be160772385f701f030608bc2857461b649c06a5b7a982df7b278d1db54729a3` |
| `acquisitions/anchorage_2025_acquisition_shortlist.json` | `d7b20f0087f5833119be7c1e6d6b350e13e88758df5d840106a4d636ecd10214` |
| `acquisitions/anchorage_2025_scl_quality_screen.csv` | `bf64c81a28a04f139f129982270be0b1e9a45103cda3b8f6bee45d8d7af53d78` |
| `acquisitions/anchorage_2025_scl_quality_screen_expanded.csv` | `5410467bbe99267e77f6a6f79cae6a9d29ef1a30a7f15fc627ee4edf97187867` |
| `acquisitions/anchorage_2025_scl_quality_selection.json` | `a75941fa61f270dabe26c9ac45554e29aa145b3dc56b0ed973b40fdecae2d951` |
| `acquisitions/anchorage_2025_scl_quality_selection_expanded.json` | `2c3397cfd6def6450162417c166aacd8f87443bff9af03c7d80355752e72c800` |
| `geometries/anchorage_land_study_regions.geojson` | `47d1c54eeb621edb4c2e7cdc0506d3b52454689f267522e349bb9a6e15137c81` |
| `geometries/anchorage_municipal_land.geojson` | `5e85f238c3685f964e90ebae0918fc5a47d4ff1506eaf6b545597c39feabede5` |
| `geometries/anchorage_municipal_water.geojson` | `3b576a07717d07e0c4663870114a04c471b866c2b6d9ac8ec54f16255b40c516` |
| `geometries/anchorage_municipality.geojson` | `1f8ce6152314a549970e0bd665b6868e5c82ddc573d5c57cc780fc4d6ac3520f` |
| `geometries/anchorage_study_areas.geojson` | `4c695d9cbe63e26a5ff9b872f8b0eb21ce41232eabdecbfae4cc3d3a978ae89b` |
