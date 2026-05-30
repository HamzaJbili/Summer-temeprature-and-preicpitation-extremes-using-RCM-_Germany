# Summer Temperature and Precipitation Extremes over Germany
## Assessment Using ICON-CLM Regional Climate Modelling vs E-OBS

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This repository contains the complete Python analysis workflow for a master's thesis assessing **summer (JJA) temperature and precipitation trends and extremes over Germany** using high-resolution regional climate modelling.

The study evaluates the performance of **ICON-CLM** (~12 km) against the **E-OBS** gridded observational dataset (v28e, 0.25°) over the period **1950–2022**. It computes **8 annual extreme indices** selected for their scientific relevance to Germany's summer hazards (heat waves, heavy precipitation, drought), applies non-parametric trend statistics (Theil-Sen + Mann-Kendall with Yue-Wang autocorrelation correction), and links extreme-summer patterns to atmospheric and land-surface process drivers. All figures are produced in **IPCC AR6 publication style**.

---

## Scientific Context

| Item | Detail |
|---|---|
| Model | ICON-CLM (~12 km, ERA5-driven hindcast) |
| Observations | E-OBS v28e (0.25°) |
| Domain | Germany |
| Season | JJA (June–July–August) |
| Analysis period | 1950–2022 |
| Threshold reference | 1961–1990 (WMO recommendation for extreme trend studies) |
| Anomaly reference | 1991–2020 (current WMO climatological normal) |
| Trend method | Theil-Sen slope + Mann-Kendall (Yue-Wang autocorrelation correction) |
| Figure style | IPCC AR6 WGI (BrBG / RdBu_r diverging, 5-yr & 11-yr rolling means, 600 dpi) |

---

## Repository Structure

```
.
├── utils.py                  # Shared utilities (imported by all scripts)
├── script1_mean_climate.py   # JJA mean temperature and precipitation: climatology, bias, trends
├── script2_extremes.py       # 8 annual extreme indices + trend maps + summary figures
├── script3_drivers.py        # Process-driver composite and correlation analysis
│
├── requirements.txt          # Python dependencies
├── .gitignore                # Files excluded from version control
├── LICENSE                   # MIT License
└── README.md                 # This file
```

### Script execution order

```
script1  →  script2  →  script3
```

Script 3 reads annual index NetCDF files produced by script 2 and raises a
clear error if script 2 has not been run first.

---

## Extreme Indices Computed (8 total)

The index set is deliberately compact, retaining only indices with a clear physical
interpretation, high signal-to-noise ratio in Germany's summer climate, and direct
relevance to documented hazards.  Only daily mean temperature (Tmean) is used —
indices requiring Tmax or Tmin (TXx, TX90p, TNx, TN90p) are not applicable to
this dataset.

### Temperature (3)

| Index | Definition | Threshold / method | Hazard relevance |
|---|---|---|---|
| **T90p** | JJA days with Tmean > local 90th pct | 90th pct JJA Tmean, 1961–1990 | Hot-day frequency |
| **HWN** | Number of heatwave events (≥ 3 consecutive T90p days) | T90p | Persistent heat — frequency |
| **HWD** | Mean heatwave event duration (days event⁻¹) | T90p; computed from same runs as HWN | Persistent heat — severity |

The 90th percentile (≈ 9 hot days per summer in 1961–1990) follows the ETCCDI
convention for percentile-based warm-day indices (cf. TX90p/TN90p) and is more
sensitive to moderate heat extremes than the 95th, giving better-populated
heatwave maps and continuous Germany-average series.
Together, HWN × HWD describes total heatwave exposure per season (comparable to
a hazard dose); disentangling frequency from duration is essential for heat-mortality
and energy-demand impact assessments.

### Precipitation — Heavy events (4)

| Index | Definition | Threshold / method | Hazard relevance |
|---|---|---|---|
| **R10mm** | JJA days with P ≥ 10 mm day⁻¹ | Fixed 10 mm threshold | Heavy precipitation frequency |
| **Rx1day** | Annual maximum 1-day precipitation (mm day⁻¹) | — | Peak convective / flash-flood intensity |
| **Rx5day** | Annual maximum consecutive 5-day precipitation (mm) | Sliding 5-day window | Basin-scale river flooding |
| **SDII** | Mean precipitation on wet days (mm wet-day⁻¹) | Wet day: P ≥ 1 mm day⁻¹ | Per-event intensity change independent of frequency |

### Precipitation — Drought (1)

| Index | Definition | Threshold / method | Hazard relevance |
|---|---|---|---|
| **CDD** | Maximum consecutive dry days in JJA | P < 1 mm day⁻¹ | Drought spell persistence; soil-moisture depletion |

---

## Process Driver Variables (script 3)

| Variable | Description | Units |
|---|---|---|
| Z500 | Geopotential height at 500 hPa | m |
| SHF | Surface sensible heat flux | W m⁻² |
| LHF | Surface latent heat flux | W m⁻² |
| SM | Soil moisture (uppermost layer) | kg m⁻² |
| WIND | 10-metre wind speed | m s⁻¹ |
| CAPE | Convective Available Potential Energy | J kg⁻¹ |
| CIN | Convective Inhibition | J kg⁻¹ |

For each index, script 3 identifies upper-quartile (top 25%) summers by Germany-average index value and computes composite anomaly maps and Pearson / Spearman correlations for each driver.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/hamzajbili/summer-temeprature-and-preicpitation-extremes-using-rcm-_germany.git
cd summer-temeprature-and-preicpitation-extremes-using-rcm-_germany
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# or
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Data Requirements

The scripts expect the following input files in the working directory.
These files are **not included** in this repository (data sharing restrictions).

### Temperature
| File | Variable | Description |
|---|---|---|
| `ICONCLM_tas_daily_1950_2022_0.25deg_Germany.nc` | `tas` | ICON-CLM daily mean temperature |
| `EOBS_tg_daily_1950_2022_0.25deg_Germany.nc` | `tg` | E-OBS daily mean temperature |

### Precipitation
| File | Variable | Description |
|---|---|---|
| `ICONCLM_pr_daily_1950_2022_0.25deg_Germany.nc` | `pr` | ICON-CLM daily precipitation |
| `EOBS_rr_daily_1950_2022_0.25deg_Germany.nc` | `rr` | E-OBS daily precipitation |

### Driver variables (script 3)
| File | Variable | Description |
|---|---|---|
| `z500_daily_1950_2022_0.25deg_Germany.nc` | `z` | Geopotential height 500 hPa |
| `sensible_heat_flux_daily_1950_2022_0.25deg_Germany.nc` | `hfss` | Surface sensible heat flux |
| `latent_heat_flux_daily_1950_2022_0.25deg_Germany.nc` | `hfls` | Surface latent heat flux |
| `soil_moisture_daily_1950_2022_0.25deg_Germany.nc` | `mrso` | Soil moisture |
| `wind_sfc_daily_1950_2022_0.25deg_Germany.nc` | `sfcWind` | 10-metre wind speed |
| `cape_daily_1950_2022_0.25deg_Germany.nc` | `cape` | CAPE |
| `cin_daily_1950_2022_0.25deg_Germany.nc` | `cin` | CIN |

### Germany shapefile
```
/work/jbiliham/shapefile_Germany/gadm41_DEU_0.shp
```
Update the `GERMANY_SHP` constant at the top of each script to match your system.

> **Pre-processing note:** Input data were pre-processed using CDO (Schulzweida, 2023).
> ICON-CLM fields were remapped from the native triangular icosahedral grid to
> the E-OBS 0.25° regular grid using first-order conservative remapping for
> precipitation (`cdo remapcon`) and bilinear interpolation for temperature
> (`cdo remapbil`).

---

## Usage

All scripts are run from the directory containing the input data files.

### Script 1 — Mean climate: climatology, bias, and trends

```bash
python script1_mean_climate.py
```

**Outputs** → `output_mean_climate/`
```
figures/
  JJA_mean_temperature_climatology_map.png     ← 3-panel: E-OBS | ICON | bias (1991-2020)
  JJA_mean_temperature_trend_map.png           ← paired trend map with stippling
  JJA_mean_temperature_germany_series.png      ← time series with rolling means + trend annotation
  JJA_total_precipitation_climatology_map.png
  JJA_total_precipitation_trend_map.png
  JJA_total_precipitation_germany_series.png
tables/
  mean_climate_summary.csv
```

### Script 2 — Extreme indices

```bash
python script2_extremes.py
```

**Outputs** → `output_extremes/`
```
figures/
  {index}_trend_map.png          ← paired E-OBS / ICON-CLM trend map (8 indices)
  {index}_germany_series.png     ← Germany-average time series with trend lines (8 indices)
  T90p_exceedance_days_spatial_mean.png  ← T90p mean field (E-OBS | ICON-CLM | bias)
  taylor_diagram.png             ← ICON-CLM model skill (correlation / normalised std dev)
  trend_heatmap.png              ← all-index trend summary with significance markers
tables/
  extreme_indices_summary.csv
netcdf/
  {index}_{ICON|EOBS}_annual.nc  ← annual index arrays (input for script 3)
```

### Script 3 — Process-driver analysis

**Run script 2 first.**

```bash
python script3_drivers.py
```

**Outputs** → `output_drivers/`
```
figures/
  {index}_all_drivers_composite.png   ← 7-panel composite per index
  {index}_{driver}_composite.png      ← individual driver composite maps
tables/
  {index}_driver_correlations.csv
  {index}_top_quartile_years.csv
```

---

## Output Description

### Climatology maps (`{variable}_climatology_map.png`)

Three-panel figure (script 1 only): (a) E-OBS 1991-2020 mean, (b) ICON-CLM 1991-2020 mean,
(c) model bias (ICON minus E-OBS). Panels (a) and (b) use a sequential colormap calibrated
to the observed range; panel (c) uses a symmetric diverging colormap centred on zero.

### Trend maps (`{index}_trend_map.png`)

Two-panel figure (E-OBS left, ICON-CLM right) showing the Theil-Sen slope per
decade at each grid cell. Significance stippling marks grid cells where the
Mann-Kendall p-value < 0.05. Colourmap: BrBG for precipitation (wetter = green),
RdBu_r for temperature (warmer = red). T90p additionally carries a third
**Bias (ICON − E-OBS)** panel.

### Time series (`{index}_germany_series.png`)

Single-panel figure (script 2): annual Germany-average values for E-OBS (black)
and ICON-CLM (red) as thin lines, with the Theil-Sen trend per dataset drawn as
bold dashed lines and the slope per decade annotated in the legend.

### Taylor diagram (`taylor_diagram.png`)

Polar diagram comparing ICON-CLM to E-OBS for all 8 indices simultaneously.
Each point's angular position encodes the Pearson correlation; radial position
encodes the normalised standard deviation (σ_model / σ_obs). RMSE contours
are drawn from the reference point (E-OBS = ★ at r=1, θ=0).

### Trend heatmap (`trend_heatmap.png`)

Compact row-normalised heatmap summarising the direction and significance of all
index trends. Each cell shows the Sen slope per decade with * (p < 0.05) or
** (p < 0.01). Useful for rapid cross-index comparison.

### Summary CSV (`extreme_indices_summary.csv`)

One row per index. Key columns:

| Column | Description |
|---|---|
| `EOBS_threshold_mean` | Germany-mean threshold value (E-OBS) |
| `ICON_threshold_mean` | Same for ICON-CLM |
| `threshold_bias_ICON_minus_EOBS` | Model threshold bias |
| `EOBS_mean_gridcell_trend` | Mean Sen slope across Germany grid cells (E-OBS) |
| `ICON_mean_gridcell_trend` | Same for ICON-CLM |
| `EOBS_sig_grid_fraction` | Fraction of grid cells with MK p < 0.05 (E-OBS) |
| `ICON_sig_grid_fraction` | Same for ICON-CLM |
| `EOBS_series_sen_slope_decade` | Sen slope of Germany-average annual series |
| `ICON_series_sen_slope_decade` | Same for ICON-CLM |
| `EOBS_series_MK_p` | MK p-value for Germany-average series (E-OBS) |
| `ICON_series_MK_p` | Same for ICON-CLM |
| `series_mean_bias_ICON_minus_EOBS` | Mean annual bias |
| `series_RMSE` | RMSE between ICON-CLM and E-OBS Germany-average series |

### Correlation CSV (`{index}_driver_correlations.csv`)

One file per extreme index. Columns: `driver`, `n`, `pearson_r`, `pearson_p`,
`spearman_r`, `spearman_p`. Both Pearson and Spearman coefficients are reported
because the Germany-average annual series may not be normally distributed.

---

## Key Methodological Notes

**Threshold reference period (1961–1990):** Following WMO (2017) recommendations,
thresholds are estimated from 1961–1990, a pre-acceleration baseline. This ensures
exceedance-count trends reflect real climate change rather than a shifting baseline.

**Anomaly reference period (1991–2020):** Anomalies are computed relative to the
current WMO 30-year climatological normal (1991–2020).

**Autocorrelation correction:** The Mann-Kendall test uses the Yue-Wang
modification (`mk.yue_wang_modification_test`), which applies trend-free
pre-whitening (TFPW) to correct for positive lag-1 autocorrelation that would
otherwise inflate the significance of the trend statistic (Yue & Wang, 2004).
Falls back to the original MK test if the correction fails for a given cell.

**Wet-day restriction:** SDII uses a wet-day threshold of P ≥ 1 mm day⁻¹,
consistent with ETCCDI definitions (Zhang et al., 2011).

**Rx5day computation:** The 5-day rolling sum is computed using a sliding window
approach (equivalent to `pandas.rolling(5).sum()`), with the maximum over all
windows in the JJA season taken as the annual value.

**IPCC figure style:** All figures apply `set_ipcc_style()` from `utils.py`, which
sets rcParams matching IPCC AR6 WGI standards (Helvetica/Arial font family, 9 pt
body text, 600 dpi output, BrBG / RdBu_r diverging colormaps).

---

## Dependencies

See `requirements.txt` for pinned versions. Core libraries:

| Library | Purpose |
|---|---|
| `xarray` | NetCDF handling and labelled array operations |
| `numpy` | Numerical computations and run-length encoding |
| `pandas` | Summary tables, CSV output, rolling means |
| `scipy` | Theil-Sen slope estimation (`theilslopes`) |
| `pymannkendall` | Mann-Kendall test with autocorrelation corrections |
| `matplotlib` | All figure production (IPCC-style maps, time series, Taylor diagram) |
| `geopandas` | Germany shapefile loading and boundary drawing |
| `shapely` | Polygon-to-path conversion for contour clipping |

---

## References

- Alexander, L. V., et al. (2006). Global observed changes in daily climate extremes of temperature and precipitation. *Journal of Geophysical Research: Atmospheres*, 111, D05109. https://doi.org/10.1029/2005JD006290
- Collaud Coen, M., et al. (2020). Identifying European air quality stations for detection of long-term changes. *Atmospheric Measurement Techniques*, 13, 6945–6964. https://doi.org/10.5194/amt-13-6945-2020
- Cornes, R. C., et al. (2018). An Ensemble Version of the E-OBS Temperature and Precipitation Data Sets. *Journal of Geophysical Research: Atmospheres*, 123, 9391–9409. https://doi.org/10.1029/2017JD028200
- Donat, M. G., et al. (2013). Updated analyses of temperature and precipitation extreme indices since the beginning of the twentieth century. *Journal of Geophysical Research: Atmospheres*, 118, 2098–2118. https://doi.org/10.1002/jgrd.50150
- Fischer, E. M., & Knutti, R. (2015). Anthropogenic contribution to global occurrence of heavy-precipitation and high-temperature extremes. *Nature Climate Change*, 5, 560–564. https://doi.org/10.1038/nclimate2617
- Fischer, E. M., et al. (2014). Robust spatially aggregated projections of climate extremes. *Nature Climate Change*, 4, 713–717. https://doi.org/10.1038/nclimate2317
- Frich, P., et al. (2002). Observed coherent changes in climatic extremes during the second half of the twentieth century. *Climate Research*, 19, 193–212. https://doi.org/10.3354/cr019193
- Hersbach, H., et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146, 1999–2049. https://doi.org/10.1002/qj.3803
- Moberg, A., & Jones, P. D. (2005). Trends in indices for extremes in daily temperature and precipitation in central and western Europe. *International Journal of Climatology*, 25, 1149–1171. https://doi.org/10.1002/joc.1163
- Perkins, S. E., & Alexander, L. V. (2013). On the measurement of heat waves. *Journal of Climate*, 26, 4500–4517. https://doi.org/10.1175/JCLI-D-12-00383.1
- Schulzweida, U. (2023). CDO User Guide v2.2.2. Zenodo. https://doi.org/10.5281/zenodo.10020826
- Sen, P. K. (1968). Estimates of the regression coefficient based on Kendall's tau. *Journal of the American Statistical Association*, 63, 1379–1389.
- WMO (2017). *Guidelines on the Calculation of Climate Normals*. WMO-No. 1203.
- Yue, S., & Wang, C. (2004). The Mann-Kendall test modified by effective sample size to detect trend in serially correlated hydrological series. *Water Resources Research*, 40, W08201. https://doi.org/10.1029/2004WR003024
- Zaengl, G., et al. (2015). The ICON (ICOsahedral Non-hydrostatic) modelling framework of DWD and MPI-M. *Quarterly Journal of the Royal Meteorological Society*, 141, 563–579. https://doi.org/10.1002/qj.2378
- Zhang, X., et al. (2011). Indices for monitoring changes in extremes based on daily temperature and precipitation data. *WIREs Climate Change*, 2, 851–870. https://doi.org/10.1002/wcc.147
- Zolina, O., et al. (2010). Improving estimates of heavy and extreme precipitation using daily records from European rain gauges. *Journal of Hydrometeorology*, 11, 771–785. https://doi.org/10.1175/2010JHM1252.1
- Zolina, O., et al. (2014). Precipitation variability and extremes in central Europe: new view from ENSEMBLES regional climate models. *Climate Dynamics*, 42, 881–898. https://doi.org/10.1007/s00382-013-1818-5

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Hamza Jbili** — M.Sc Environmental and Resource Management  
BTU Cottbus-Senftenberg, Germany  
Climate change science · Environmental engineering · Data analysis
