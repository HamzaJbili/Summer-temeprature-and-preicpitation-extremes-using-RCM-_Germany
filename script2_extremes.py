"""
script2_extremes.py
-------------------
Annual JJA extreme climate indices for ICON-CLM vs E-OBS over Germany,
1950–2022.

Indices computed (6 total)
~~~~~~~~~~~~~~~~~~~~~~~~~~
  Temperature (3)
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ T90p    Hot days exceeding the local 90th-pct JJA Tmean (1961-1990)    │
  │ HWN     Heatwave Number: events of ≥3 consecutive T90p days            │
  │ HWD     Heatwave Duration: mean length of heatwave events (days/event) │
  └─────────────────────────────────────────────────────────────────────────┘

  Precipitation — Drought / Intensity (3)
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ CDD     Maximum consecutive dry days per JJA season                    │
  │ SDII    Simple Daily Intensity Index (mm wet-day⁻¹)                    │
  │ SPI     Standardised Precipitation Index (JJA seasonal total)          │
  └─────────────────────────────────────────────────────────────────────────┘

Index selection rationale
~~~~~~~~~~~~~~~~~~~~~~~~~
  The set is distilled from the full ETCCDI catalogue to the indices with
  the highest signal-to-noise ratio, clearest physical interpretation, and
  greatest relevance to Germany's documented summer hazards.

  Temperature: T90p quantifies raw hot-day frequency using the 90th percentile
  threshold — sensitive to moderate heat extremes and therefore informative
  for early-period trends when intense heatwaves were rare; HWN counts distinct
  heatwave events; HWD measures mean event length.  Together they describe the
  full hazard profile: a doubling of both HWN and HWD implies four times more
  heatwave exposure per season than if frequency alone doubled.

  Precipitation — Drought / Intensity: CDD is the ETCCDI benchmark drought-
  duration index, directly coupled to soil-moisture deficits and heat–drought
  co-occurrence; SDII isolates intensity changes from frequency changes; SPI
  provides a standardised, distribution-normalised drought severity metric.

  Excluded indices: R10mm, Rx1day, Rx5day (heavy-event frequency / peak
  intensity) are reserved for later analysis; R95p and R95pTOT were dropped
  because their JJA signal is weak and spatially incoherent over Germany;
  R99p, R20mm, CWD are either redundant or lower signal-to-noise in
  Germany's maritime-continental summer climate.

Methodological choices
~~~~~~~~~~~~~~~~~~~~~~
  Threshold / percentile reference period : 1961-1990  (WMO recommendation
      for extreme-trend studies; ensures exceedance-count trends reflect
      climate change relative to a pre-acceleration baseline)
  Anomaly reference period                : 1991-2020  (current WMO normal)
  Wet-day threshold                       : P ≥ 1.0 mm day⁻¹
  Heatwave minimum length                 : ≥ 3 consecutive hot days
  Trend method                            : Theil-Sen slope (per decade) +
      Mann-Kendall with Yue-Wang trend-free pre-whitening (TFPW) to account
      for lag-1 autocorrelation (Yue & Wang, 2004)
  Significance level                      : p < 0.05 (two-tailed)

Outputs per index
~~~~~~~~~~~~~~~~~
  figures/  {index}_trend_map.png       — paired E-OBS / ICON trend map
            {index}_germany_series.png  — Germany-average time series
  tables/   extreme_indices_summary.csv — bias, trend, RMSE table
  netcdf/   {index}_{ICON|EOBS}_annual.nc — annual arrays for script3

Summary outputs (end of script)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  figures/  taylor_diagram.png          — model skill assessment
            trend_heatmap.png           — all-index trend summary

Scientific references
~~~~~~~~~~~~~~~~~~~~~
  Alexander et al. (2006) JGR-Atmos 111, D05109
  Fischer & Knutti (2015) Nature Clim. Change 5, 560-564
  Fischer et al. (2014) Nature Clim. Change 4, 713-717
  Frich et al. (2002) Clim. Res. 19, 193-212
  Perkins & Alexander (2013) J. Climate 26, 4500-4517
  Yue & Wang (2004) Water Resour. Res. 40, W08201
  Zolina et al. (2010) J. Hydrometeor. 11, 771-785
  Zolina et al. (2014) Clim. Dyn. 42, 881-898
"""

import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from utils import (
    # data helpers
    load_field, keep_jja, reference_mean, compute_anomalies,
    area_mean, rmse, compute_trend_maps, series_stats,
    # geometry
    load_country_shape,
    # precipitation index functions (defined in utils)
    annual_sdii, annual_spi_jja,
    # visualisation
    set_ipcc_style, plot_obs_bias_maps, plot_germany_series,
    taylor_diagram, plot_trend_heatmap,
    plot_climatology_maps,
    # constants
    REF_START, REF_END, ANOM_START, ANOM_END, DPI, DRY_DAY_MAX,
)

# Apply IPCC publication style to all matplotlib figures in this script
set_ipcc_style()

# ── file paths ─────────────────────────────────────────────────────────────────
MODEL_T_FILE = "Tas_degrees_DE-0.25_ERA5_JJA_evaluation_r1i1p1f1_CLMcom-Hereon_ICON-CLM-202407-1-1_v1-r2_day_1950-2022.nc"
OBS_T_FILE   = "tg_JJA_eobs_dd_0.25deg_reg_v28.0e.nc"
MODEL_P_FILE = "TOT_PREC_DE-0.25_ERA5_JJA_evaluation_r1i1p1f1_CLMcom-Hereon_ICON-CLM-202407-1-1_v1-r2_day_1950-2022.nc"
OBS_P_FILE   = "rr_JJA_eobs_dd_0.25deg_reg_v28.0e.nc"
GERMANY_SHP  = "/work/jbiliham/shapefile_Germany/gadm41_DEU_0.shp"

FIGDIR = os.path.join("output_extremes", "figures")
TABDIR = os.path.join("output_extremes", "tables")
NCDIR  = os.path.join("output_extremes", "netcdf")   # annual index NetCDFs
for d in [FIGDIR, TABDIR, NCDIR]:
    os.makedirs(d, exist_ok=True)

HW_MIN_LEN = 3   # minimum consecutive T90p days required to define a heatwave

# ── IPCC-standard colormap palettes ───────────────────────────────────────────
TEMP_COLORS = [
    "#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#f7f7f7",
    "#fee0b6", "#fdb863", "#f4a582", "#d6604d", "#b2182b",
]

# Precipitation trend: brown (drier) → white → green (wetter)
PREC_COLORS = [
    "#7f3b08", "#b35806", "#e08214", "#fdb863", "#fee0b6", "#f7f7f7",
    "#d8f0ed", "#a6dba0", "#5aae61", "#1b7837",
]

# CDD drought palette: blue (fewer dry days) → pale → dark brown (more dry days)
# Positive trend = more drought → warm brown/orange tones.
# NOTE: 10 colours for the 10 intervals defined by the 11-boundary CDD_LEVELS.
CDD_COLORS = [
    "#4575b4",  # blue            (-8 .. -6)  strongly wetter
    "#74add1",  # medium blue     (-6 .. -4)
    "#abd9e9",  # light blue      (-4 .. -2)
    "#e0f3f8",  # very light blue (-2 .. -1)
    "#ffffd4",  # pale near-zero  (-1 ..  0)
    "#fee391",  # warm yellow     ( 0 ..  1)  drought starts here
    "#fec44f",  # amber           ( 1 ..  2)
    "#fe9929",  # orange          ( 2 ..  4)
    "#d95f0e",  # dark orange     ( 4 ..  6)
    "#8c2d04",  # dark brown      ( 6 ..  8)  strongly drier
]

# ── Colormap level definitions (Theil-Sen slope, unit per decade) ──────────────
# ── Trend map colormap levels (Theil-Sen slope per decade) ────────────────────
TEMP_LEVELS = [-8, -6, -4, -2, -1,    0,   1,    2,    4,    6,    8]  # days/decade
HW_LEVELS   = [-2.0, -1.5, -1.0, -0.5, -0.25, 0, 0.25, 0.5, 1.0, 1.5, 2.0]
HWD_LEVELS  = [-3.0, -2.0, -1.5, -1.0, -0.5,  0,  0.5, 1.0, 1.5, 2.0, 3.0]
CDD_LEVELS  = [-3.0, -2.0, -1.5, -1.0, -0.5,  0,  0.5, 1.0, 1.5, 2.0, 3.0]  # days/decade
SDII_LEVELS = [-1.0, -0.75, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 0.75, 1.0]
SPI_LEVELS  = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]  # SPI/decade

# ── Mean-field (climatology) palettes and levels ───────────────────────────────
# T90p threshold — actual 90th-pct JJA Tmean values (°C; adjust if data is in K)
T90P_THR_COLORS = [
    "#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c",
    "#fc4e2a", "#e31a1c", "#bd0026", "#800026", "#4d0019",
]
T90P_THR_LEVELS = [17, 19, 21, 22, 23, 24, 25, 26, 27, 28, 30]  # °C

# HWN mean — events per summer
HWN_CLIM_COLORS = [
    "#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c",
    "#fc4e2a", "#e31a1c", "#bd0026", "#800026", "#4d0019",
]
HWN_CLIM_LEVELS = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0, 2.5]

# HWD mean — days per event
HWD_CLIM_COLORS = [
    "#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c",
    "#fc4e2a", "#e31a1c", "#bd0026", "#800026", "#4d0019",
]
HWD_CLIM_LEVELS = [3, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.5]

# SDII mean — mm per wet day
SDII_CLIM_COLORS = [
    "#f0f9ff", "#d0e7f5", "#a8d1e8", "#7db8d8", "#4f99c4",
    "#2b7ab3", "#1560a0", "#0b4d89", "#073b71", "#042a58",
]
SDII_CLIM_LEVELS = [3, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 9]  # mm/wet-day

# CDD mean — max consecutive dry days per summer
CDD_CLIM_COLORS = [
    "#fff7bc", "#fee391", "#fec44f", "#fe9929", "#ec7014",
    "#cc4c02", "#993404", "#772b00", "#5a1f00", "#3d1200",
]
CDD_CLIM_LEVELS = [4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16]  # days

# Sequential warm palette + levels for the T90p mean spatial-distribution map
# (climatological hot-day frequency, days summer⁻¹; not a trend → sequential)
T90P_DIST_COLORS = [
    "#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c",
    "#fc4e2a", "#e31a1c", "#bd0026", "#800026", "#4d0019",
]
T90P_DIST_LEVELS = [5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18]  # days summer⁻¹


# ══════════════════════════════════════════════════════════════════════════════
#  Threshold computation
# ══════════════════════════════════════════════════════════════════════════════

def percentile_threshold(daily_jja, q):
    """
    Compute the local q-th percentile from the 1961-1990 reference period.

    Used for the T90p hot-day threshold (90th percentile of JJA Tmean over the
    fixed 1961-1990 baseline) so that exceedance-count trends reflect genuine
    climate change relative to a pre-acceleration reference.

    Parameters
    ----------
    daily_jja : xr.DataArray (time, lat, lon) — JJA daily data
    q         : float — percentile (0-100)

    Returns
    -------
    xr.DataArray (lat, lon) — local threshold field (float32)
    """
    ref = daily_jja.sel(time=slice(REF_START, REF_END))
    thr = ref.quantile(q / 100.0, dim="time", skipna=True)
    if "quantile" in thr.dims:
        thr = thr.squeeze("quantile", drop=True)
    return thr.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  Temperature index functions (computed from daily Tmean only)
# ══════════════════════════════════════════════════════════════════════════════

def annual_exceedance_days(daily_jja, threshold):
    """
    Count JJA days per year that exceed a spatially-varying threshold.

    Used for T90p (hot days above the local 90th-percentile Tmean).  The
    threshold is computed once from the 1961-1990 reference period so that any
    trend in exceedance counts reflects genuine climate change.
    """
    return (daily_jja > threshold).astype(np.float32).groupby("time.year").sum("time")


def annual_cdd(daily_jja):
    """
    CDD — Maximum consecutive dry days (P < 1 mm day⁻¹) per JJA season.

    The longest unbroken dry spell per summer.  CDD is the ETCCDI benchmark
    drought-duration index; it is tightly coupled to soil-moisture depletion,
    crop stress, and the positive land–atmosphere feedback that amplifies
    summer heat extremes in central Europe (Fischer et al., 2007).

    Implementation: run-length encoding via np.diff on a padded binary array.
    """
    years = np.unique(daily_jja["time"].dt.year.values).astype(int)
    lat   = daily_jja["lat"].values
    lon   = daily_jja["lon"].values
    out   = np.zeros((len(years), len(lat), len(lon)), dtype=np.float32)

    for yi, yr in enumerate(years):
        dry = (daily_jja.sel(time=str(yr)).values < DRY_DAY_MAX).astype(np.int8)
        for i in range(len(lat)):
            for j in range(len(lon)):
                d = dry[:, i, j]
                if d.sum() == 0:
                    out[yi, i, j] = 0.0
                    continue
                padded = np.concatenate([[0], d, [0]])
                diff   = np.diff(padded)
                starts = np.where(diff ==  1)[0]
                ends   = np.where(diff == -1)[0]
                out[yi, i, j] = float(np.max(ends - starts))

    return xr.DataArray(
        out, coords={"year": years, "lat": lat, "lon": lon},
        dims=("year", "lat", "lon"), name="CDD",
    )


def annual_heatwave_number(daily_jja, threshold):
    """
    HWN — Count of heatwave events per JJA season at each grid cell.

    A heatwave event is defined as a run of ≥ HW_MIN_LEN (3) consecutive
    days with Tmean > local T90p threshold (1961-1990).  Using Tmean rather
    than Tmax captures the integrated thermal load (overnight as well as
    daytime heat), which is the physiologically relevant metric for heat-
    related mortality.

    Reference: Perkins & Alexander (2013); Fischer & Knutti (2015).
    """
    years    = np.unique(daily_jja["time"].dt.year.values).astype(int)
    lat      = daily_jja["lat"].values
    lon      = daily_jja["lon"].values
    out      = np.zeros((len(years), len(lat), len(lon)), dtype=np.float32)
    thr_vals = threshold.values

    for yi, yr in enumerate(years):
        hot = (daily_jja.sel(time=str(yr)).values > thr_vals).astype(np.int8)
        for i in range(len(lat)):
            for j in range(len(lon)):
                d = hot[:, i, j]
                if d.sum() < HW_MIN_LEN:
                    out[yi, i, j] = 0.0
                    continue
                padded  = np.concatenate([[0], d, [0]])
                diff    = np.diff(padded)
                starts  = np.where(diff ==  1)[0]
                ends    = np.where(diff == -1)[0]
                lengths = ends - starts
                out[yi, i, j] = float(np.sum(lengths >= HW_MIN_LEN))

    return xr.DataArray(
        out, coords={"year": years, "lat": lat, "lon": lon},
        dims=("year", "lat", "lon"), name="HWN",
    )


def annual_hwd(daily_jja, threshold):
    """
    HWD — Mean heatwave duration (days per event) per JJA season.

    Defined as:  HWD = Σ(heatwave-event lengths) / number_of_events
    Returns NaN for grid cells and years with no qualifying heatwave.

    Together with HWN, HWD characterises the nature of changing heat-wave
    hazard: if HWN increases but HWD remains stable, events become more
    frequent but not longer; if both increase, the exposure scales as
    HWN × HWD.  Disentangling frequency from duration is essential for
    impact modelling of heat mortality and energy demand.

    Implementation: identical run-length encoding to HWN; qualifying event
    lengths are averaged rather than counted.

    Reference: Perkins & Alexander (2013, J. Climate 26, 4500-4517);
               Fischer & Knutti (2015, Nature Clim. Change 5, 560-564).
    """
    years    = np.unique(daily_jja["time"].dt.year.values).astype(int)
    lat      = daily_jja["lat"].values
    lon      = daily_jja["lon"].values
    out      = np.full((len(years), len(lat), len(lon)), np.nan, dtype=np.float32)
    thr_vals = threshold.values

    for yi, yr in enumerate(years):
        hot = (daily_jja.sel(time=str(yr)).values > thr_vals).astype(np.int8)
        for i in range(len(lat)):
            for j in range(len(lon)):
                d = hot[:, i, j]
                if d.sum() < HW_MIN_LEN:
                    continue              # no qualifying heatwave; out stays NaN
                padded  = np.concatenate([[0], d, [0]])
                diff    = np.diff(padded)
                starts  = np.where(diff ==  1)[0]
                ends    = np.where(diff == -1)[0]
                lengths = ends - starts
                hw_len  = lengths[lengths >= HW_MIN_LEN]
                if len(hw_len) > 0:
                    out[yi, i, j] = float(np.mean(hw_len))

    return xr.DataArray(
        out, coords={"year": years, "lat": lat, "lon": lon},
        dims=("year", "lat", "lon"), name="HWD",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Save annual index to NetCDF (for script3 driver analysis)
# ══════════════════════════════════════════════════════════════════════════════

def save_index(da, name, dataset_label):
    """Save an annual-index DataArray to NetCDF for ingestion by script3."""
    path = os.path.join(NCDIR, f"{name}_{dataset_label}_annual.nc")
    da.to_netcdf(path)
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  Core processing pipeline (one call per index)
# ══════════════════════════════════════════════════════════════════════════════

def process_index(
    name, long_name,
    annual_model, annual_obs,
    thr_model, thr_obs,
    unit, trend_unit, trend_levels, colors, tick_fmt,
    gdf, geom,
    summary_rows, ts_obs_store, ts_mod_store, heatmap_rows,
    obs_sequential=False,
):
    """
    Run the full analysis pipeline for one extreme index.

    Steps
    -----
    1.  Compute gridcell-wise Theil-Sen slopes and Mann-Kendall p-values with
        Yue-Wang TFPW autocorrelation correction (compute_trend_maps).
    2.  Plot paired trend map (E-OBS | ICON-CLM) with significance stippling.
    3.  Compute Germany-average time series and 1991-2020 anomalies.
    4.  Plot enhanced time series (absolute + anomaly, with 5-yr/11-yr rolling
        means and trend annotation).
    5.  Compute and store summary statistics (bias, RMSE, trend).
    6.  Store metadata in shared dicts for summary figures.

    Parameters
    ----------
    name        : str — short identifier (used in file names)
    long_name   : str — verbose name including unit (used in row labels)
    annual_model, annual_obs : xr.DataArray (year, lat, lon)
    thr_model, thr_obs : xr.DataArray (lat, lon) or None
        Percentile threshold; None for fixed-threshold indices (Rx1day, etc.)
    unit        : str — index unit (e.g. "days summer⁻¹")
    trend_unit  : str — trend unit (e.g. "days decade⁻¹")
    trend_levels: list — colormap boundary levels
    colors      : list — colormap colours (len = len(trend_levels)-1)
    tick_fmt    : str — colorbar tick format string
    gdf, geom   : Germany boundary GeoDataFrame and Shapely geometry
    summary_rows : list — appended with per-index statistics dict
    ts_obs_store, ts_mod_store : dict — Germany-average series for Taylor diagram
    heatmap_rows   : list — stores trend stats for the trend heatmap
    obs_sequential : bool — if True, the E-OBS panel uses a one-sided sequential
        palette (for indices with a dominant trend direction); False (default)
        allows diverging scale so both positive and negative trend areas appear.
    """
    print(f"  • {name}: computing trend maps …")
    trend_obs   = compute_trend_maps(annual_obs)
    trend_model = compute_trend_maps(annual_model)

    # ── 1. Trend map: E-OBS | Diff (ICON − E-OBS) ────────────────────────────
    plot_obs_bias_maps(
        obs_slope   = trend_obs["sen_slope"],
        model_slope = trend_model["sen_slope"],
        obs_pval    = trend_obs["mk_pvalue"],
        model_pval  = trend_model["mk_pvalue"],
        gdf=gdf, geom=geom,
        outfile     = os.path.join(FIGDIR, f"{name}_trend_map.png"),
        obs_levels  = trend_levels, obs_colors=colors,
        cbar_label  = trend_unit, tick_fmt=tick_fmt,
        suptitle    = f"{long_name} — Theil-Sen Trend 1950–2022",
        obs_sequential=obs_sequential,
    )

    # ── 2. Germany-average series and 1991-2020 anomalies ────────────────────
    obs_series   = area_mean(annual_obs)
    model_series = area_mean(annual_model)
    obs_clim     = reference_mean(obs_series,   ANOM_START, ANOM_END)
    mod_clim     = reference_mean(model_series, ANOM_START, ANOM_END)
    obs_anom     = compute_anomalies(obs_series,   obs_clim)
    model_anom   = compute_anomalies(model_series, mod_clim)

    obs_stats   = series_stats(obs_series)
    model_stats = series_stats(model_series)

    # ── 3. Enhanced time series figure ───────────────────────────────────────
    plot_germany_series(
        obs_series, model_series,
        obs_anom,   model_anom,
        ylabel      = unit,
        title       = f"{long_name}  —  Germany average, JJA 1950–2022",
        outfile     = os.path.join(FIGDIR, f"{name}_germany_series.png"),
        obs_stats   = obs_stats,
        model_stats = model_stats,
        simple      = True,
    )

    # ── 4. Summary statistics ─────────────────────────────────────────────────
    thr_obs_mean   = float(np.nanmean(thr_obs.values))   if thr_obs   is not None else np.nan
    thr_model_mean = float(np.nanmean(thr_model.values)) if thr_model is not None else np.nan
    thr_bias = (thr_model_mean - thr_obs_mean) if np.isfinite(thr_obs_mean) else np.nan

    summary_rows.append({
        "index":      name,
        "unit":       unit,
        "trend_unit": trend_unit,

        # Percentile threshold (NaN for fixed-threshold indices)
        "EOBS_threshold_mean":            _fmt(thr_obs_mean),
        "ICON_threshold_mean":            _fmt(thr_model_mean),
        "threshold_bias_ICON_minus_EOBS": _fmt(thr_bias),

        # Gridcell-average Theil-Sen slope
        "EOBS_mean_gridcell_trend":       _fmt(np.nanmean(trend_obs["sen_slope"].values)),
        "ICON_mean_gridcell_trend":       _fmt(np.nanmean(trend_model["sen_slope"].values)),
        "trend_bias_ICON_minus_EOBS":     _fmt(np.nanmean(
            trend_model["sen_slope"].values - trend_obs["sen_slope"].values)),

        # Fraction of grid cells with statistically significant trend (p < 0.05)
        "EOBS_sig_grid_fraction":  _fmt(np.nanmean(
            (trend_obs["mk_pvalue"].values   < 0.05).astype(float))),
        "ICON_sig_grid_fraction":  _fmt(np.nanmean(
            (trend_model["mk_pvalue"].values < 0.05).astype(float))),

        # Germany-average series — Theil-Sen + Mann-Kendall (Yue-Wang)
        "EOBS_series_sen_slope_decade": _fmt(obs_stats["sen_slope_decade"]),
        "ICON_series_sen_slope_decade": _fmt(model_stats["sen_slope_decade"]),
        "EOBS_series_MK_p":             _fmt4(obs_stats["mk_p"]),
        "ICON_series_MK_p":             _fmt4(model_stats["mk_p"]),

        # Model–observation agreement
        "series_mean_bias_ICON_minus_EOBS": _fmt(
            float(np.nanmean(model_series.values - obs_series.values))),
        "series_RMSE": _fmt(rmse(model_series.values, obs_series.values)),
    })

    # ── 5. Store for summary figures (Taylor diagram, trend heatmap) ───────────
    ts_obs_store[name] = obs_series.values
    ts_mod_store[name] = model_series.values

    heatmap_rows.append({
        "name":       long_name,
        "obs_slope":  obs_stats["sen_slope_decade"],
        "mod_slope":  model_stats["sen_slope_decade"],
        "obs_pval":   obs_stats["mk_p"],
        "mod_pval":   model_stats["mk_p"],
        "trend_unit": trend_unit,
    })


# ── helpers ────────────────────────────────────────────────────────────────────
def _fmt(v):
    return round(float(v), 3) if np.isfinite(float(v)) else np.nan

def _fmt4(v):
    return round(float(v), 4) if np.isfinite(float(v)) else np.nan


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Load Germany boundary ─────────────────────────────────────────────────
    print("Loading Germany boundary shapefile …")
    gdf, geom = load_country_shape(GERMANY_SHP)

    # ── Load daily data ───────────────────────────────────────────────────────
    print("Loading daily temperature and precipitation data …")
    tas_model = keep_jja(load_field(MODEL_T_FILE, "Tas"))
    tas_obs   = keep_jja(load_field(OBS_T_FILE,   "tg"))
    pr_model  = keep_jja(load_field(MODEL_P_FILE, "p"))
    pr_obs    = keep_jja(load_field(OBS_P_FILE,   "rr"))

    # Shared data stores for the summary figures generated at the end
    summary_rows   = []   # per-index statistics → CSV
    ts_obs_store   = {}   # Germany-avg series   → Taylor diagram
    ts_mod_store   = {}
    heatmap_rows   = []   # trend stats          → heatmap

    # ── Pre-compute T90p percentile threshold (1961-1990 reference) ───────────
    print("\nComputing T90p percentile threshold (1961-1990 reference) …")
    t90_model = percentile_threshold(tas_model, 90)
    t90_obs   = percentile_threshold(tas_obs,   90)

    # ══════════════════════════════════════════════════════════════════════════
    #  TEMPERATURE INDICES
    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== Temperature Indices ===")

    # ── T90p: Hot days (Tmean > local 90th pct) — days per JJA season ─────────
    print("T90p: computing annual hot-day counts …")
    t90_days_model = annual_exceedance_days(tas_model, t90_model)
    t90_days_obs   = annual_exceedance_days(tas_obs,   t90_obs)
    save_index(t90_days_model, "T90p_days", "ICON")
    save_index(t90_days_obs,   "T90p_days", "EOBS")

    process_index(
        name="T90p_exceedance_days",
        long_name="T90p — Hot days > 90th pct Tmean [days summer⁻¹]",
        annual_model=t90_days_model, annual_obs=t90_days_obs,
        thr_model=t90_model,         thr_obs=t90_obs,
        unit="days summer⁻¹",        trend_unit="days decade⁻¹",
        trend_levels=TEMP_LEVELS,    colors=TEMP_COLORS, tick_fmt="%.1f",
        gdf=gdf, geom=geom,
        summary_rows=summary_rows,
        ts_obs_store=ts_obs_store, ts_mod_store=ts_mod_store,
        heatmap_rows=heatmap_rows,
    )

    # ── T90p mean hot-day frequency (E-OBS | ICON-CLM | Bias) ────────────────
    print("T90p: plotting mean hot-day frequency …")
    plot_climatology_maps(
        obs_clim = t90_days_obs.mean("year",   skipna=True),
        mod_clim = t90_days_model.mean("year", skipna=True),
        gdf=gdf, geom=geom,
        outfile  = os.path.join(FIGDIR, "T90p_exceedance_days_spatial_mean.png"),
        levels   = T90P_DIST_LEVELS, colors=T90P_DIST_COLORS,
        cbar_label = "T90p hot-day frequency [days summer⁻¹]", tick_fmt="%.0f",
        suptitle = "T90p — Hot days > 90th pct Tmean [days summer⁻¹] — Mean 1950–2022",
    )

    # ── T90p threshold field (actual 90th-pct temperature values) ─────────────
    # Shows the spatial pattern of the threshold itself (°C), i.e. how warm
    # it needs to be at each grid cell to qualify as a "hot day".
    print("T90p: plotting 90th-pct temperature threshold field …")
    plot_climatology_maps(
        obs_clim = t90_obs,   mod_clim = t90_model,
        gdf=gdf, geom=geom,
        outfile  = os.path.join(FIGDIR, "T90p_threshold_spatial.png"),
        levels   = T90P_THR_LEVELS, colors=T90P_THR_COLORS,
        cbar_label = "90th-pct Tmean threshold [°C]", tick_fmt="%.1f",
        suptitle = "T90p threshold — 90th-pct JJA Tmean 1961–1990",
    )

    # ── HWN: Heatwave Number (≥3 consecutive T90p days) ──────────────────────
    print("HWN: computing heatwave number (may take several minutes) …")
    hwn_model = annual_heatwave_number(tas_model, t90_model)
    hwn_obs   = annual_heatwave_number(tas_obs,   t90_obs)
    save_index(hwn_model, "HWN", "ICON")
    save_index(hwn_obs,   "HWN", "EOBS")

    process_index(
        name="Heatwave_number",
        long_name="HWN — Heatwave Number [events summer⁻¹]",
        annual_model=hwn_model, annual_obs=hwn_obs,
        thr_model=t90_model,    thr_obs=t90_obs,
        unit="events summer⁻¹", trend_unit="events decade⁻¹",
        trend_levels=HW_LEVELS, colors=TEMP_COLORS, tick_fmt="%.2f",
        gdf=gdf, geom=geom,
        summary_rows=summary_rows,
        ts_obs_store=ts_obs_store, ts_mod_store=ts_mod_store,
        heatmap_rows=heatmap_rows,
    )
    plot_climatology_maps(
        obs_clim = hwn_obs.mean("year",   skipna=True),
        mod_clim = hwn_model.mean("year", skipna=True),
        gdf=gdf, geom=geom,
        outfile  = os.path.join(FIGDIR, "Heatwave_number_spatial_mean.png"),
        levels   = HWN_CLIM_LEVELS, colors=HWN_CLIM_COLORS,
        cbar_label = "HWN mean [events summer⁻¹]", tick_fmt="%.1f",
        suptitle = "HWN — Mean heatwave number 1950–2022",
    )

    # ── HWD: Heatwave Duration (mean days per heatwave event) ─────────────────
    # Computed from the same hot-day binary mask as HWN but averaging event
    # lengths instead of counting events.  Seasons with no heatwave are NaN.
    print("HWD: computing mean heatwave duration (may take several minutes) …")
    hwd_model = annual_hwd(tas_model, t90_model)
    hwd_obs   = annual_hwd(tas_obs,   t90_obs)
    save_index(hwd_model, "HWD", "ICON")
    save_index(hwd_obs,   "HWD", "EOBS")

    process_index(
        name="Heatwave_duration",
        long_name="HWD — Mean Heatwave Duration [days event⁻¹]",
        annual_model=hwd_model, annual_obs=hwd_obs,
        thr_model=t90_model,    thr_obs=t90_obs,
        unit="days event⁻¹",    trend_unit="days event⁻¹ decade⁻¹",
        trend_levels=HWD_LEVELS, colors=TEMP_COLORS, tick_fmt="%.2f",
        gdf=gdf, geom=geom,
        summary_rows=summary_rows,
        ts_obs_store=ts_obs_store, ts_mod_store=ts_mod_store,
        heatmap_rows=heatmap_rows,
    )
    plot_climatology_maps(
        obs_clim = hwd_obs.mean("year",   skipna=True),
        mod_clim = hwd_model.mean("year", skipna=True),
        gdf=gdf, geom=geom,
        outfile  = os.path.join(FIGDIR, "Heatwave_duration_spatial_mean.png"),
        levels   = HWD_CLIM_LEVELS, colors=HWD_CLIM_COLORS,
        cbar_label = "HWD mean [days event⁻¹]", tick_fmt="%.1f",
        suptitle = "HWD — Mean heatwave duration 1950–2022",
    )

    # ══════════════════════════════════════════════════════════════════════════
    #  PRECIPITATION INDICES — DROUGHT / INTENSITY
    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== Precipitation Indices — Drought / Intensity ===")

    # ── SDII: Simple Daily Intensity Index ────────────────────────────────────
    # Wet-day mean intensity; decouples intensity changes from frequency.
    # A rising SDII with stable or declining totals is the hallmark of
    # precipitation intensification (fewer, more intense events).
    print("SDII: computing simple daily intensity index …")
    sdii_model = annual_sdii(pr_model)
    sdii_obs   = annual_sdii(pr_obs)
    save_index(sdii_model, "SDII", "ICON")
    save_index(sdii_obs,   "SDII", "EOBS")

    process_index(
        name="SDII",
        long_name="SDII — Simple Daily Intensity Index [mm wet-day⁻¹]",
        annual_model=sdii_model, annual_obs=sdii_obs,
        thr_model=None,           thr_obs=None,
        unit="mm wet-day⁻¹",      trend_unit="mm wet-day⁻¹ decade⁻¹",
        trend_levels=SDII_LEVELS,  colors=PREC_COLORS, tick_fmt="%.2f",
        gdf=gdf, geom=geom,
        summary_rows=summary_rows,
        ts_obs_store=ts_obs_store, ts_mod_store=ts_mod_store,
        heatmap_rows=heatmap_rows,
    )
    plot_climatology_maps(
        obs_clim = sdii_obs.mean("year",   skipna=True),
        mod_clim = sdii_model.mean("year", skipna=True),
        gdf=gdf, geom=geom,
        outfile  = os.path.join(FIGDIR, "SDII_spatial_mean.png"),
        levels   = SDII_CLIM_LEVELS, colors=SDII_CLIM_COLORS,
        cbar_label = "SDII mean [mm wet-day⁻¹]", tick_fmt="%.1f",
        suptitle = "SDII — Mean daily intensity 1950–2022",
    )

    print("\n=== Precipitation Indices — Drought ===")

    # ── CDD: Maximum consecutive dry days ─────────────────────────────────────
    # The ETCCDI benchmark drought-duration index.  CDD increasing means
    # longer dry spells within summer, amplifying heat extremes via the
    # soil-moisture–temperature feedback and increasing agricultural water
    # stress even in seasons with unchanged total rainfall.
    print("CDD: computing maximum consecutive dry days (may take a few minutes) …")
    cdd_model = annual_cdd(pr_model)
    cdd_obs   = annual_cdd(pr_obs)
    save_index(cdd_model, "CDD", "ICON")
    save_index(cdd_obs,   "CDD", "EOBS")

    process_index(
        name="CDD",
        long_name="CDD — Consecutive dry days [days summer⁻¹]",
        annual_model=cdd_model, annual_obs=cdd_obs,
        thr_model=None,          thr_obs=None,
        unit="days summer⁻¹",    trend_unit="days decade⁻¹",
        trend_levels=CDD_LEVELS, colors=CDD_COLORS, tick_fmt="%.2f",
        gdf=gdf, geom=geom,
        summary_rows=summary_rows,
        ts_obs_store=ts_obs_store, ts_mod_store=ts_mod_store,
        heatmap_rows=heatmap_rows,
    )
    plot_climatology_maps(
        obs_clim = cdd_obs.mean("year",   skipna=True),
        mod_clim = cdd_model.mean("year", skipna=True),
        gdf=gdf, geom=geom,
        outfile  = os.path.join(FIGDIR, "CDD_spatial_mean.png"),
        levels   = CDD_CLIM_LEVELS, colors=CDD_CLIM_COLORS,
        cbar_label = "CDD mean [days summer⁻¹]", tick_fmt="%.0f",
        suptitle = "CDD — Mean max consecutive dry days 1950–2022",
    )

    # ── SPI: Standardised Precipitation Index (JJA seasonal total) ───────────
    print("SPI: computing standardised precipitation index …")
    spi_model = annual_spi_jja(pr_model)
    spi_obs   = annual_spi_jja(pr_obs)
    save_index(spi_model, "SPI", "ICON")
    save_index(spi_obs,   "SPI", "EOBS")

    process_index(
        name="SPI",
        long_name="SPI — Standardised Precipitation Index [dimensionless]",
        annual_model=spi_model, annual_obs=spi_obs,
        thr_model=None,          thr_obs=None,
        unit="SPI [-]",          trend_unit="SPI per decade",
        trend_levels=SPI_LEVELS,  colors=PREC_COLORS, tick_fmt="%.2f",
        gdf=gdf, geom=geom,
        summary_rows=summary_rows,
        ts_obs_store=ts_obs_store, ts_mod_store=ts_mod_store,
        heatmap_rows=heatmap_rows,
    )
    # SPI mean ≈ 0 by construction — no climatology map

    # ══════════════════════════════════════════════════════════════════════════
    #  SUMMARY FIGURES
    # ══════════════════════════════════════════════════════════════════════════

    # ── Taylor diagram: ICON-CLM model skill for all indices ──────────────────
    print("Generating Taylor diagram …")
    taylor_diagram(
        obs_dict=ts_obs_store,
        mod_dict=ts_mod_store,
        outfile=os.path.join(FIGDIR, "taylor_diagram.png"),
        title="Model Skill — ICON-CLM vs E-OBS\n(Germany average, JJA 1950–2022)",
    )

    # ── Trend heatmap: all-index trend summary ────────────────────────────────
    print("Generating trend heatmap …")
    plot_trend_heatmap(
        heatmap_rows=heatmap_rows,
        outfile=os.path.join(FIGDIR, "trend_heatmap.png"),
        title="Theil-Sen Trends per Decade — All JJA Extreme Indices",
    )

    # ── Save statistics table ─────────────────────────────────────────────────
    df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(TABDIR, "extreme_indices_summary.csv")
    df.to_csv(csv_path, index=False)

    # ── Final report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("Script 2 complete.")
    print(f"  Individual figures  → {FIGDIR}/")
    for idx in ["T90p_exceedance_days", "Heatwave_number", "Heatwave_duration",
                "SDII", "CDD", "SPI"]:
        print(f"    {idx}_trend_map.png")
        print(f"    {idx}_germany_series.png")
    print(f"  Summary figures     → {FIGDIR}/taylor_diagram.png")
    print(f"                        {FIGDIR}/trend_heatmap.png")
    print(f"  Statistics table    → {csv_path}")
    print(f"  Annual NetCDF files → {NCDIR}/")
    print(f"  Indices processed   : {len(summary_rows)} total")
    print("  (Run script3_drivers.py next to analyse process drivers)")
    print("=" * 66)
