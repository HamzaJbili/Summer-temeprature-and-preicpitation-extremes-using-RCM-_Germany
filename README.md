# Summer Temperature and Precipitation Extremes over Germany
## Assessment Using ICON-CLM Regional Climate Modelling vs E-OBS

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This repository contains the complete Python analysis workflow for a master's thesis assessing **summer (JJA) temperature and precipitation trends and extremes over Germany** using high-resolution regional climate modelling.

The study evaluates the performance of **ICON-CLM** (~12 km) against the **E-OBS** gridded observational dataset (v28e, 0.25°) over the period **1950–2022**, computing six annual extreme indices, non-parametric trend statistics, and a process-driver analysis linking extreme patterns to atmospheric and land-surface conditions.

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
| Trend method | Theil-Sen slope + Mann-Kendall test (Yue-Wang autocorrelation correction) |

---

## Repository Structure

```
.
├── utils.py                  # Shared utilities (imported by all scripts)
├── script1_mean_climate.py   # JJA mean temperature and precipitation trends
├── script2_extremes.py       # Six annual extreme indices + trend maps
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

Script 3 reads annual index NetCDF files produced by script 2 and will raise
a clear error if script 2 has not been run first.

---

## Extreme Indices Computed

| Index | Variable | Definition | Threshold | Hazard |
|---|---|---|---|---|
| T95 | Temperature | JJA days with Tmean > T95(i,j) | 95th pct JJA Tmean, 1961–1990 | Warm day frequency |
| HWN | Temperature | ≥ 3 consecutive T95 days = 1 event | T95 | Persistent heat |
| R95p | Precipitation | JJA wet days with P > R95(i,j) | 95th pct wet-day JJA precip, 1961–1990 | Heavy precipitation |
| R99p | Precipitation | JJA wet days with P > R99(i,j) | 99th pct wet-day JJA precip, 1961–1990 | Very heavy precipitation |
| P<1mm | Precipitation | JJA days with P < 1 mm d⁻¹ | Fixed: 1 mm d⁻¹ | Summer dryness |
| CDD | Precipitation | Max consecutive dry days in JJA | Fixed: 1 mm d⁻¹ | Dry spell persistence |

---

## Process Driver Variables

The driver analysis (script 3) uses the following atmospheric and
land-surface variables derived from ICON-CLM and ERA5:

| Variable | Description | Units |
|---|---|---|
| Z500 | Geopotential height at 500 hPa | m |
| SHF | Surface sensible heat flux | W m⁻² |
| LHF | Surface latent heat flux | W m⁻² |
| SM | Soil moisture (uppermost layer) | kg m⁻² |
| WIND | 10-metre wind speed | m s⁻¹ |
| CAPE | Convective Available Potential Energy | J kg⁻¹ |
| CIN | Convective Inhibition | J kg⁻¹ |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/summer-extremes-germany.git
cd summer-extremes-germany
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
Update the `GERMANY_SHP` path at the top of each script to match your system.

> **Note:** The data files listed above were pre-processed (remapped and masked
> to Germany) using CDO (Schulzweida, 2023). ICON-CLM fields were remapped from
> the native triangular icosahedral grid to the E-OBS 0.25° regular grid using
> first-order conservative remapping for precipitation (`cdo remapcon`) and
> bilinear interpolation for temperature (`cdo remapbil`).

---

## Usage

All scripts are run from the directory containing the input data files.

### Script 1 — Mean climate trends

```bash
python script1_mean_climate.py
```

**Outputs** → `output_mean_climate/`
- `figures/JJA_mean_temperature_trend_map.png`
- `figures/JJA_total_precipitation_trend_map.png`
- `figures/JJA_mean_temperature_germany_series.png`
- `figures/JJA_total_precipitation_germany_series.png`
- `tables/mean_climate_summary.csv`

### Script 2 — Extreme indices

```bash
python script2_extremes.py
```

**Outputs** → `output_extremes/`
- `figures/{index}_trend_map.png` (six indices × two panels each)
- `figures/{index}_germany_series.png`
- `tables/extreme_indices_summary.csv`
- `netcdf/{index}_{ICON|EOBS}_annual.nc` (annual index arrays for script 3)

### Script 3 — Process-driver analysis

**Run script 2 first.**

```bash
python script3_drivers.py
```

**Outputs** → `output_drivers/`
- `figures/{index}_all_drivers_composite.png` (seven-panel composite per index)
- `figures/{index}_{driver}_composite.png` (individual composites)
- `tables/{index}_driver_correlations.csv`
- `tables/{index}_top_quartile_years.csv`

---

## Output Description

### Summary CSV (scripts 1 and 2)

Each row corresponds to one index. Columns include:

| Column | Description |
|---|---|
| `EOBS_threshold_mean` | Germany-mean T95 or R95/R99 threshold value (E-OBS) |
| `ICON_threshold_mean` | Same for ICON-CLM |
| `threshold_bias_ICON_minus_EOBS` | Threshold bias |
| `EOBS_mean_gridcell_trend` | Mean Sen slope across Germany grid cells (E-OBS) |
| `ICON_mean_gridcell_trend` | Same for ICON-CLM |
| `EOBS_sig_grid_fraction` | Fraction of grid cells with MK p < 0.05 (E-OBS) |
| `ICON_sig_grid_fraction` | Same for ICON-CLM |
| `EOBS_series_sen_slope_decade` | Sen slope of Germany-average annual series |
| `ICON_series_sen_slope_decade` | Same for ICON-CLM |
| `series_mean_bias_ICON_minus_EOBS` | Mean annual bias |
| `series_RMSE` | RMSE between ICON-CLM and E-OBS Germany-average series |

### Correlation CSV (script 3)

One file per extreme index. Columns: `driver`, `n`, `pearson_r`, `pearson_p`,
`spearman_r`, `spearman_p`. Both Pearson and Spearman coefficients are provided
because the Germany-average annual series may not be normally distributed.

---

## Key Methodological Notes

**Threshold reference period (1961–1990):** Following WMO (2017) recommendations
for extreme trend studies, thresholds are estimated from the pre-acceleration
baseline 1961–1990. This ensures that exceedance trends reflect real climate
change relative to the historical climate, not a shifting baseline.

**Anomaly reference period (1991–2020):** Anomalies are computed relative to the
current WMO 30-year climatological normal (WMO, 2017).

**Autocorrelation correction:** The Mann-Kendall test uses the Yue-Wang
modification (`mk.yue_wang_modification_test`) which implements trend-free
pre-whitening (TFPW) to correct for positive lag-1 autocorrelation that would
otherwise inflate significance (Yue et al., 2002; Collaud Coen et al., 2020).

**Wet-day restriction:** R95p and R99p thresholds are computed from wet days
only (P ≥ 1 mm d⁻¹), consistent with ETCCDI definitions (Zhang et al., 2011).

---

## Dependencies

See `requirements.txt` for pinned versions. Core libraries:

| Library | Purpose |
|---|---|
| `xarray` | NetCDF handling and labeled array operations |
| `numpy` | Numerical computations |
| `pandas` | Summary tables and CSV output |
| `scipy` | Theil-Sen slope estimation |
| `pymannkendall` | Mann-Kendall test with autocorrelation corrections |
| `matplotlib` | All figure production |
| `geopandas` | Germany shapefile masking |
| `shapely` | Geometry operations |

---

## References

- Collaud Coen et al. (2020). Atmospheric Measurement Techniques, 13, 6945–6964. https://doi.org/10.5194/amt-13-6945-2020
- Cornes et al. (2018). Journal of Geophysical Research: Atmospheres, 123, 9391–9409. https://doi.org/10.1029/2017JD028200
- Hersbach et al. (2020). Quarterly Journal of the Royal Meteorological Society, 146, 1999–2049. https://doi.org/10.1002/qj.3803
- Jones (1999). Monthly Weather Review, 127, 2204–2210. https://doi.org/10.1175/1520-0493(1999)127<2204:FASOCR>2.0.CO;2
- Schulzweida (2023). CDO User Guide v2.2.2. Zenodo. https://doi.org/10.5281/zenodo.10020826
- Sen (1968). Journal of the American Statistical Association, 63, 1379–1389.
- WMO (2017). Guidelines on the Calculation of Climate Normals. WMO-No. 1203.
- Yue et al. (2002). Hydrological Processes, 16, 1807–1829. https://doi.org/10.1002/hyp.1095
- Zaengl et al. (2015). Quarterly Journal of the Royal Meteorological Society, 141, 563–579. https://doi.org/10.1002/qj.2378
- Zhang et al. (2011). WIREs Climate Change, 2, 851–870. https://doi.org/10.1002/wcc.147

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Author

Master's Thesis — Department of Meteorology / Climate Science  
Analysis period: 1950–2022 | Domain: Germany | Season: JJA
