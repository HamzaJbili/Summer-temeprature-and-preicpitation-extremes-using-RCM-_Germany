"""
script2_extremes.py
-------------------
Annual JJA extreme climate indices for ICON-CLM vs E-OBS over Germany,
1950–2022.

Indices computed
~~~~~~~~~~~~~~~~
  Temperature (2)
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ T95     Hot days exceeding the local 95th-pct JJA Tmean (1961-1990)    │
  │ HWN     Heatwave Number: events of ≥3 consecutive T95 days             │
  └─────────────────────────────────────────────────────────────────────────┘

  Precipitation — Frequency (4)
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ R95p    Wet days > 95th pct of 1961-1990 wet-day distribution          │
  │ R99p    Wet days > 99th pct of 1961-1990 wet-day distribution          │
  │ R10mm   Days with P ≥ 10 mm day⁻¹  (heavy precipitation)              │
  │ R20mm   Days with P ≥ 20 mm day⁻¹  (very heavy precipitation)         │
  └─────────────────────────────────────────────────────────────────────────┘

  Precipitation — Intensity (3)
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ Rx1day  Annual maximum 1-day precipitation (mm day⁻¹)                  │
  │ Rx5day  Annual maximum consecutive 5-day precipitation (mm)             │
  │ SDII    Simple Daily Intensity Index (mm wet-day⁻¹)                    │
  └─────────────────────────────────────────────────────────────────────────┘

  Precipitation — Concentration (1)
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ R95pTOT Fraction of wet-day precip from very heavy events (%)          │
  └─────────────────────────────────────────────────────────────────────────┘

  Precipitation — Duration (3)
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ Dry     Dry-day count (P < 1 mm day⁻¹)                                │
  │ CDD     Maximum consecutive dry days                                    │
  │ CWD     Maximum consecutive wet days                                    │
  └─────────────────────────────────────────────────────────────────────────┘

Methodological choices
~~~~~~~~~~~~~~~~~~~~~~
  Threshold / percentile reference period : 1961-1990  (WMO recommendation
      for extreme-trend studies; ensures exceedance-count trends reflect
      climate change measured from a pre-acceleration baseline)
  Anomaly reference period                : 1991-2020  (current WMO normal)
  Wet-day threshold                       : P ≥ 1.0 mm day⁻¹
  Trend method                            : Theil-Sen slope + Mann-Kendall
      (Yue-Wang autocorrelation correction, i.e. trend-free pre-whitening)
  Significance level                      : p < 0.05 (two-tailed)

Outputs per index
~~~~~~~~~~~~~~~~~
  figures/  {index}_trend_map.png       — paired E-OBS / ICON trend map
            {index}_germany_series.png  — Germany-average time series
  tables/   extreme_indices_summary.csv — bias, trend, RMSE table
  netcdf/   {index}_{ICON|EOBS}_annual.nc — annual arrays for script3

Summary outputs (end of script)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  figures/  precipitation_overview.png  — all precipitation trends (one figure)
            taylor_diagram.png          — model skill assessment
            trend_heatmap.png           — all-index trend summary

Scientific references
~~~~~~~~~~~~~~~~~~~~~
  Alexander et al. (2006) JGR-Atmos 111, D05109
  Donat et al. (2013) JGR-Atmos 118, 2098-2118
  Fischer et al. (2014) Nature Clim. Change 4, 713-717
  Frich et al. (2002) Clim. Res. 19, 193-212
  Moberg & Jones (2005) Int. J. Climatol. 25, 1149-1171
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
    # new precipitation index functions
    annual_rx1day, annual_rx5day,
    annual_r10mm, annual_r20mm,
    annual_sdii, annual_r95ptot, annual_cwd,
    # visualisation
    set_ipcc_style, plot_paired_trend_maps, plot_germany_series,
    plot_precipitation_overview, taylor_diagram, plot_trend_heatmap,
    # constants
    REF_START, REF_END, ANOM_START, ANOM_END, DPI, WET_DAY_MIN, DRY_DAY_MAX,
)

# Apply IPCC publication style to all matplotlib figures in this script
set_ipcc_style()

# ── file paths ─────────────────────────────────────────────────────────────────
MODEL_T_FILE = "ICONCLM_tas_daily_1950_2022_0.25deg_Germany.nc"
OBS_T_FILE   = "EOBS_tg_daily_1950_2022_0.25deg_Germany.nc"
MODEL_P_FILE = "ICONCLM_pr_daily_1950_2022_0.25deg_Germany.nc"
OBS_P_FILE   = "EOBS_rr_daily_1950_2022_0.25deg_Germany.nc"
GERMANY_SHP  = "/work/jbiliham/shapefile_Germany/gadm41_DEU_0.shp"

FIGDIR = os.path.join("output_extremes", "figures")
TABDIR = os.path.join("output_extremes", "tables")
NCDIR  = os.path.join("output_extremes", "netcdf")     # annual index NetCDFs
for d in [FIGDIR, TABDIR, NCDIR]:
    os.makedirs(d, exist_ok=True)

HW_MIN_LEN = 3   # minimum consecutive T95 days defining a heatwave event

# ── IPCC-standard colormap palettes ───────────────────────────────────────────
# Temperature: blue-white-red (warm anomaly = red, consistent with IPCC AR6)
TEMP_COLORS = [
    "#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#f7f7f7",
    "#fee0b6", "#fdb863", "#f4a582", "#d6604d", "#b2182b",
]

# Precipitation: brown-white-green (wetter = green, IPCC/WMO convention)
PREC_COLORS = [
    "#7f3b08", "#b35806", "#e08214", "#fdb863", "#fee0b6", "#f7f7f7",
    "#d8f0ed", "#a6dba0", "#5aae61", "#1b7837",
]

# ── Colormap level definitions (Theil-Sen slope, unit per decade) ──────────────
# These ranges are calibrated to typical observed summer trends in Germany;
# extend="both" in contourf handles values outside the range.
TEMP_LEVELS   = [-8, -6, -4, -2, -1,    0,   1,  2,   4,   6,   8]  # days/decade
HW_LEVELS     = [-2.0, -1.5, -1.0, -0.5, -0.25, 0, 0.25, 0.5, 1.0, 1.5, 2.0]  # events/decade
PREC_LEVELS   = [-8, -6, -4, -2, -1,    0,   1,  2,   4,   6,   8]  # days/decade
RX1DAY_LEVELS = [-5, -4, -3, -2, -1,    0,   1,  2,   3,   4,   5]  # mm/decade
RX5DAY_LEVELS = [-10, -8, -6, -4, -2,   0,   2,  4,   6,   8,  10]  # mm/decade
R10MM_LEVELS  = [-3.0, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 3.0]   # days/decade
R20MM_LEVELS  = [-1.5, -1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0, 1.5]  # days/decade
SDII_LEVELS   = [-1.0, -0.75, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 0.75, 1.0]  # mm/wd/decade
R95TOT_LEVELS = [-8, -6, -4, -2, -1,    0,   1,  2,   4,   6,   8]  # %/decade
CWD_LEVELS    = [-3.0, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 3.0]   # days/decade


# ══════════════════════════════════════════════════════════════════════════════
#  Threshold computation
# ══════════════════════════════════════════════════════════════════════════════

def percentile_threshold(daily_jja, q, wet_only=False):
    """
    Compute the local q-th percentile from the 1961-1990 reference period.

    Parameters
    ----------
    daily_jja : xr.DataArray (time, lat, lon) — JJA daily data
    q         : float — percentile (0-100)
    wet_only  : bool — if True, restrict to wet days (P ≥ WET_DAY_MIN)
                before computing the quantile.  Required for R95p/R99p
                (ETCCDI definition: percentile of the wet-day distribution).

    Returns
    -------
    xr.DataArray (lat, lon) — local threshold field (float32)
    """
    ref = daily_jja.sel(time=slice(REF_START, REF_END))
    if wet_only:
        # Retain only wet days in the reference period for quantile computation
        ref = ref.where(ref >= WET_DAY_MIN)
    thr = ref.quantile(q / 100.0, dim="time", skipna=True)
    if "quantile" in thr.dims:
        thr = thr.squeeze("quantile", drop=True)
    return thr.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  Temperature index functions (defined here; precipitation in utils.py)
# ══════════════════════════════════════════════════════════════════════════════

def annual_exceedance_days(daily_jja, threshold):
    """
    Count JJA days per year that exceed a spatially-varying threshold.

    Used for T95 (hot days > 95th pct temperature) and R95p / R99p
    (wet days > 95th / 99th pct wet-day precipitation).

    Parameters
    ----------
    daily_jja : xr.DataArray (time, lat, lon)
    threshold : xr.DataArray (lat, lon) — local exceedance threshold
    """
    return (daily_jja > threshold).astype(np.float32).groupby("time.year").sum("time")


def annual_dry_days(daily_jja):
    """
    Count JJA days per year with P < DRY_DAY_MAX (1 mm day⁻¹).

    Dry-day frequency is a primary indicator of summer drying trends and
    complements the wet-extreme indices (R95p, CWD) to give a full picture
    of the precipitation distribution shift.
    """
    return (daily_jja < DRY_DAY_MAX).astype(np.float32).groupby("time.year").sum("time")


def annual_cdd(daily_jja):
    """
    Maximum consecutive dry days (CDD) per JJA season at each grid cell.

    CDD measures the longest unbroken sequence of dry days (P < 1 mm day⁻¹)
    within the summer season.  It is a key drought indicator and is closely
    linked to heat-stress amplification through soil-moisture feedbacks.

    Implementation: run-length encoding via np.diff on a padded binary array.
    One Python loop over years; inner loop over grid cells (unavoidable for
    run-length encoding without additional libraries).
    """
    years = np.unique(daily_jja["time"].dt.year.values).astype(int)
    lat   = daily_jja["lat"].values
    lon   = daily_jja["lon"].values
    out   = np.zeros((len(years), len(lat), len(lon)), dtype=np.float32)

    for yi, yr in enumerate(years):
        dry = (daily_jja.sel(time=str(yr)).values < DRY_DAY_MAX).astype(np.int8)
        # dry shape: (time, lat, lon)
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
    Count heatwave events per JJA season (Germany-wide at each grid cell).

    A heatwave event is defined as a run of ≥ HW_MIN_LEN (3) consecutive
    days with Tmean > local T95 threshold.  The index captures the frequency
    of distinct heatwave occurrences rather than total hot-day counts.

    Implementation: run-length encoding identical to annual_cdd, applied to
    a binary 'hot-day' array; only runs meeting the minimum-length criterion
    are counted.
    """
    years    = np.unique(daily_jja["time"].dt.year.values).astype(int)
    lat      = daily_jja["lat"].values
    lon      = daily_jja["lon"].values
    out      = np.zeros((len(years), len(lat), len(lon)), dtype=np.float32)
    thr_vals = threshold.values   # (lat, lon)

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
    summary_rows, overview_store, ts_obs_store, ts_mod_store, heatmap_rows,
):
    """
    Run the full analysis pipeline for one extreme index.

    Steps
    -----
    1.  Compute gridcell-wise Theil-Sen slopes and Mann-Kendall p-values.
    2.  Plot paired trend map (E-OBS | ICON-CLM) with stippling.
    3.  Compute Germany-average time series and 1991-2020 anomalies.
    4.  Plot enhanced time series (absolute + anomaly, with rolling means).
    5.  Compute and store summary statistics (bias, RMSE, trend).
    6.  Store metadata in shared dicts for the summary figures.

    Parameters
    ----------
    name        : str — short index identifier (used in filenames)
    long_name   : str — verbose name with unit (used in overview row labels)
    annual_model, annual_obs : xr.DataArray (year, lat, lon)
    thr_model, thr_obs : xr.DataArray (lat, lon) or None
        Percentile threshold; None for fixed-threshold indices.
    unit        : str — index unit (e.g. "days summer⁻¹")
    trend_unit  : str — trend unit (e.g. "days decade⁻¹")
    trend_levels: list — colormap boundary levels
    colors      : list — colormap colours (len = len(trend_levels)-1)
    tick_fmt    : str — colorbar tick format
    gdf, geom   : Germany boundary
    summary_rows : list — appended with statistics dict
    overview_store : dict — stores trend arrays for precipitation overview figure
    ts_obs_store, ts_mod_store : dict — Germany-average series (for Taylor diagram)
    heatmap_rows   : list — stores trend stats for heatmap
    """
    print(f"  • {name}: computing trend maps …")
    trend_obs   = compute_trend_maps(annual_obs)
    trend_model = compute_trend_maps(annual_model)

    # ── 1. Paired trend map ───────────────────────────────────────────────────
    plot_paired_trend_maps(
        obs_slope=trend_obs["sen_slope"],   model_slope=trend_model["sen_slope"],
        obs_pval =trend_obs["mk_pvalue"],   model_pval =trend_model["mk_pvalue"],
        gdf=gdf, geom=geom,
        outfile=os.path.join(FIGDIR, f"{name}_trend_map.png"),
        levels=trend_levels, colors=colors,
        cbar_label=trend_unit, tick_fmt=tick_fmt,
        suptitle=f"{long_name} — Theil-Sen Trend 1950–2022",
    )

    # ── 2. Germany-average series and anomalies ───────────────────────────────
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
        ylabel=unit,
        ylabel_anom=f"Anomaly [{unit}]",
        title=f"{long_name}  —  Germany average, JJA 1950–2022",
        outfile=os.path.join(FIGDIR, f"{name}_germany_series.png"),
        obs_stats=obs_stats, model_stats=model_stats,
    )

    # ── 4. Summary statistics ─────────────────────────────────────────────────
    thr_obs_mean   = float(np.nanmean(thr_obs.values))   if thr_obs   is not None else np.nan
    thr_model_mean = float(np.nanmean(thr_model.values)) if thr_model is not None else np.nan
    thr_bias = (thr_model_mean - thr_obs_mean) if np.isfinite(thr_obs_mean) else np.nan

    summary_rows.append({
        "index":      name,
        "unit":       unit,
        "trend_unit": trend_unit,

        # Threshold statistics (NaN for fixed-threshold indices)
        "EOBS_threshold_mean":            _fmt(thr_obs_mean),
        "ICON_threshold_mean":            _fmt(thr_model_mean),
        "threshold_bias_ICON_minus_EOBS": _fmt(thr_bias),

        # Gridcell-average trend
        "EOBS_mean_gridcell_trend":       _fmt(np.nanmean(trend_obs["sen_slope"].values)),
        "ICON_mean_gridcell_trend":       _fmt(np.nanmean(trend_model["sen_slope"].values)),
        "trend_bias_ICON_minus_EOBS":     _fmt(np.nanmean(
            trend_model["sen_slope"].values - trend_obs["sen_slope"].values)),

        # Fraction of grid cells with significant trend (p < 0.05)
        "EOBS_sig_grid_fraction":  _fmt(np.nanmean(
            (trend_obs["mk_pvalue"].values   < 0.05).astype(float))),
        "ICON_sig_grid_fraction":  _fmt(np.nanmean(
            (trend_model["mk_pvalue"].values < 0.05).astype(float))),

        # Germany-average series trend (Theil-Sen + MK)
        "EOBS_series_sen_slope_decade": _fmt(obs_stats["sen_slope_decade"]),
        "ICON_series_sen_slope_decade": _fmt(model_stats["sen_slope_decade"]),
        "EOBS_series_MK_p":             _fmt4(obs_stats["mk_p"]),
        "ICON_series_MK_p":             _fmt4(model_stats["mk_p"]),

        # Model–observation agreement
        "series_mean_bias_ICON_minus_EOBS": _fmt(
            float(np.nanmean(model_series.values - obs_series.values))),
        "series_RMSE": _fmt(rmse(model_series.values, obs_series.values)),
    })

    # ── 5. Store for summary / overview figures ───────────────────────────────
    # Only precipitation indices go into the multi-panel overview figure
    if _is_precip(name):
        overview_store[name] = {
            "obs_slope":  trend_obs["sen_slope"],
            "mod_slope":  trend_model["sen_slope"],
            "obs_pval":   trend_obs["mk_pvalue"],
            "mod_pval":   trend_model["mk_pvalue"],
            "levels":     trend_levels,
            "colors":     colors,
            "cbar_label": trend_unit,
            "tick_fmt":   tick_fmt,
            "long_name":  long_name,
        }

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

def _is_precip(name):
    """Return True for precipitation-related indices (for overview selection)."""
    PRECIP_NAMES = {
        "R95p_exceedance_days", "R99p_exceedance_days",
        "Rx1day", "Rx5day", "R10mm", "R20mm",
        "SDII", "R95pTOT", "Dry_days", "CDD", "CWD",
    }
    return name in PRECIP_NAMES


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Load Germany boundary ─────────────────────────────────────────────────
    print("Loading Germany boundary shapefile …")
    gdf, geom = load_country_shape(GERMANY_SHP)

    # ── Load daily data ───────────────────────────────────────────────────────
    print("Loading daily temperature and precipitation data …")
    tas_model = keep_jja(load_field(MODEL_T_FILE, "tas"))
    tas_obs   = keep_jja(load_field(OBS_T_FILE,   "tg"))
    pr_model  = keep_jja(load_field(MODEL_P_FILE, "pr"))
    pr_obs    = keep_jja(load_field(OBS_P_FILE,   "rr"))

    # Shared data stores for the summary figures generated at the end
    summary_rows   = []          # per-index statistics → CSV
    overview_store = {}          # precip trend maps   → overview figure
    ts_obs_store   = {}          # Germany-avg series  → Taylor diagram
    ts_mod_store   = {}
    heatmap_rows   = []          # trend stats         → heatmap

    # ── Pre-compute percentile thresholds ─────────────────────────────────────
    print("\nComputing percentile thresholds (1961-1990 reference) …")
    t95_model  = percentile_threshold(tas_model, 95)
    t95_obs    = percentile_threshold(tas_obs,   95)
    r95_model  = percentile_threshold(pr_model,  95, wet_only=True)
    r95_obs    = percentile_threshold(pr_obs,    95, wet_only=True)
    r99_model  = percentile_threshold(pr_model,  99, wet_only=True)
    r99_obs    = percentile_threshold(pr_obs,    99, wet_only=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  TEMPERATURE INDICES
    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== Temperature Indices ===")

    # ── T95: Hot days ─────────────────────────────────────────────────────────
    print("T95: computing annual hot-day counts …")
    t95_days_model = annual_exceedance_days(tas_model, t95_model)
    t95_days_obs   = annual_exceedance_days(tas_obs,   t95_obs)
    save_index(t95_days_model, "T95_days", "ICON")
    save_index(t95_days_obs,   "T95_days", "EOBS")

    process_index(
        name="T95_exceedance_days",
        long_name="T95 — Hot days > 95th pct [days summer⁻¹]",
        annual_model=t95_days_model, annual_obs=t95_days_obs,
        thr_model=t95_model,         thr_obs=t95_obs,
        unit="days summer⁻¹",        trend_unit="days decade⁻¹",
        trend_levels=TEMP_LEVELS,    colors=TEMP_COLORS, tick_fmt="%.0f",
        gdf=gdf, geom=geom,
        summary_rows=summary_rows, overview_store=overview_store,
        ts_obs_store=ts_obs_store, ts_mod_store=ts_mod_store,
        heatmap_rows=heatmap_rows,
    )

    # ── HWN: Heatwave Number ──────────────────────────────────────────────────
    print("HWN: computing heatwave number (may take several minutes) …")
    hwn_model = annual_heatwave_number(tas_model, t95_model)
    hwn_obs   = annual_heatwave_number(tas_obs,   t95_obs)
    save_index(hwn_model, "HWN", "ICON")
    save_index(hwn_obs,   "HWN", "EOBS")

    process_index(
        name="Heatwave_number",
        long_name="HWN — Heatwave Number [events summer⁻¹]",
        annual_model=hwn_model, annual_obs=hwn_obs,
        thr_model=t95_model,    thr_obs=t95_obs,
        unit="events summer⁻¹",  trend_unit="events decade⁻¹",
        trend_levels=HW_LEVELS,  colors=TEMP_COLORS, tick_fmt="%.2f",
        gdf=gdf, geom=geom,
        summary_rows=summary_rows, overview_store=overview_store,
        ts_obs_store=ts_obs_store, ts_mod_store=ts_mod_store,
        heatmap_rows=heatmap_rows,
    )

    # ══════════════════════════════════════════════════════════════════════════
    #  PRECIPITATION INDICES — FREQUENCY
    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== Precipitation Indices — Frequency ===")

    # ── R95p: Very heavy rain days (>95th pct wet-day distribution) ───────────
    print("R95p: computing exceedance day counts …")
    r95_days_model = annual_exceedance_days(pr_model, r95_model)
    r95_days_obs   = annual_exceedance_days(pr_obs,   r95_obs)
    save_index(r95_days_model, "R95p_days", "ICON")
    save_index(r95_days_obs,   "R95p_days", "EOBS")

    process_index(
        name="R95p_exceedance_days",
        long_name="R95p — Very heavy rain days > 95th pct [days summer⁻¹]",
        annual_model=r95_days_model, annual_obs=r95_days_obs,
        thr_model=r95_model,          thr_obs=r95_obs,
        unit="days summer⁻¹",         trend_unit="days decade⁻¹",
        trend_levels=PREC_LEVELS,     colors=PREC_COLORS, tick_fmt="%.0f",
        gdf=gdf, geom=geom,
        summary_rows=summary_rows, overview_store=overview_store,
        ts_obs_store=ts_obs_store, ts_mod_store=ts_mod_store,
        heatmap_rows=heatmap_rows,
    )

    # ── R99p: Extremely heavy rain days (>99th pct wet-day distribution) ──────
    print("R99p: computing exceedance day counts …")
    r99_days_model = annual_exceedance_days(pr_model, r99_model)
    r99_days_obs   = annual_exceedance_days(pr_obs,   r99_obs)
    save_index(r99_days_model, "R99p_days", "ICON")
    save_index(r99_days_obs,   "R99p_days", "EOBS")

    process_index(
        name="R99p_exceedance_days",
        long_name="R99p — Extremely heavy rain days > 99th pct [days summer⁻¹]",
        annual_model=r99_days_model, annual_obs=r99_days_obs,
        thr_model=r99_model,          thr_obs=r99_obs,
        unit="days summer⁻¹",         trend_unit="days decade⁻¹",
        trend_levels=PREC_LEVELS,     colors=PREC_COLORS, tick_fmt="%.0f",
        gdf=gdf, geom=geom,
        summary_rows=summary_rows, overview_store=overview_store,
        ts_obs_store=ts_obs_store, ts_mod_store=ts_mod_store,
        heatmap_rows=heatmap_rows,
    )

    # ── R10mm: Heavy precipitation days (P ≥ 10 mm day⁻¹) ───────────────────
    print("R10mm: computing heavy-rain day counts …")
    r10_model = annual_r10mm(pr_model)
    r10_obs   = annual_r10mm(pr_obs)
    save_index(r10_model, "R10mm", "ICON")
    save_index(r10_obs,   "R10mm", "EOBS")

    process_index(
        name="R10mm",
        long_name="R10mm — Heavy rain days (P ≥ 10 mm) [days summer⁻¹]",
        annual_model=r10_model, annual_obs=r10_obs,
        thr_model=None,          thr_obs=None,
        unit="days summer⁻¹",    trend_unit="days decade⁻¹",
        trend_levels=R10MM_LEVELS, colors=PREC_COLORS, tick_fmt="%.1f",
        gdf=gdf, geom=geom,
        summary_rows=summary_rows, overview_store=overview_store,
        ts_obs_store=ts_obs_store, ts_mod_store=ts_mod_store,
        heatmap_rows=heatmap_rows,
    )

    # ── R20mm: Very heavy precipitation days (P ≥ 20 mm day⁻¹) ──────────────
    print("R20mm: computing very-heavy-rain day counts …")
    r20_model = annual_r20mm(pr_model)
    r20_obs   = annual_r20mm(pr_obs)
    save_index(r20_model, "R20mm", "ICON")
    save_index(r20_obs,   "R20mm", "EOBS")

    process_index(
        name="R20mm",
        long_name="R20mm — Very heavy rain days (P ≥ 20 mm) [days summer⁻¹]",
        annual_model=r20_model, annual_obs=r20_obs,
        thr_model=None,          thr_obs=None,
        unit="days summer⁻¹",    trend_unit="days decade⁻¹",
        trend_levels=R20MM_LEVELS, colors=PREC_COLORS, tick_fmt="%.2f",
        gdf=gdf, geom=geom,
        summary_rows=summary_rows, overview_store=overview_store,
        ts_obs_store=ts_obs_store, ts_mod_store=ts_mod_store,
        heatmap_rows=heatmap_rows,
    )

    # ══════════════════════════════════════════════════════════════════════════
    #  PRECIPITATION INDICES — INTENSITY
    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== Precipitation Indices — Intensity ===")

    # ── Rx1day: Maximum 1-day precipitation amount ────────────────────────────
    print("Rx1day: computing annual maximum 1-day precipitation …")
    rx1_model = annual_rx1day(pr_model)
    rx1_obs   = annual_rx1day(pr_obs)
    save_index(rx1_model, "Rx1day", "ICON")
    save_index(rx1_obs,   "Rx1day", "EOBS")

    process_index(
        name="Rx1day",
        long_name="Rx1day — Max 1-day precipitation [mm day⁻¹]",
        annual_model=rx1_model, annual_obs=rx1_obs,
        thr_model=None,          thr_obs=None,
        unit="mm day⁻¹",         trend_unit="mm decade⁻¹",
        trend_levels=RX1DAY_LEVELS, colors=PREC_COLORS, tick_fmt="%.0f",
        gdf=gdf, geom=geom,
        summary_rows=summary_rows, overview_store=overview_store,
        ts_obs_store=ts_obs_store, ts_mod_store=ts_mod_store,
        heatmap_rows=heatmap_rows,
    )

    # ── Rx5day: Maximum consecutive 5-day precipitation ──────────────────────
    print("Rx5day: computing annual maximum 5-day precipitation …")
    rx5_model = annual_rx5day(pr_model)
    rx5_obs   = annual_rx5day(pr_obs)
    save_index(rx5_model, "Rx5day", "ICON")
    save_index(rx5_obs,   "Rx5day", "EOBS")

    process_index(
        name="Rx5day",
        long_name="Rx5day — Max 5-day precipitation [mm]",
        annual_model=rx5_model, annual_obs=rx5_obs,
        thr_model=None,          thr_obs=None,
        unit="mm",                trend_unit="mm decade⁻¹",
        trend_levels=RX5DAY_LEVELS, colors=PREC_COLORS, tick_fmt="%.0f",
        gdf=gdf, geom=geom,
        summary_rows=summary_rows, overview_store=overview_store,
        ts_obs_store=ts_obs_store, ts_mod_store=ts_mod_store,
        heatmap_rows=heatmap_rows,
    )

    # ── SDII: Simple Daily Intensity Index ────────────────────────────────────
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
        summary_rows=summary_rows, overview_store=overview_store,
        ts_obs_store=ts_obs_store, ts_mod_store=ts_mod_store,
        heatmap_rows=heatmap_rows,
    )

    # ══════════════════════════════════════════════════════════════════════════
    #  PRECIPITATION INDICES — CONCENTRATION
    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== Precipitation Indices — Concentration ===")

    # ── R95pTOT: Fraction of precipitation from very heavy events ─────────────
    # Uses the R95p threshold already computed above.
    print("R95pTOT: computing fraction of precipitation from extreme events …")
    r95tot_model = annual_r95ptot(pr_model, r95_model)
    r95tot_obs   = annual_r95ptot(pr_obs,   r95_obs)
    save_index(r95tot_model, "R95pTOT", "ICON")
    save_index(r95tot_obs,   "R95pTOT", "EOBS")

    process_index(
        name="R95pTOT",
        long_name="R95pTOT — Fraction of precip from very heavy events [%]",
        annual_model=r95tot_model, annual_obs=r95tot_obs,
        thr_model=r95_model,        thr_obs=r95_obs,
        unit="%",                    trend_unit="% decade⁻¹",
        trend_levels=R95TOT_LEVELS,  colors=PREC_COLORS, tick_fmt="%.0f",
        gdf=gdf, geom=geom,
        summary_rows=summary_rows, overview_store=overview_store,
        ts_obs_store=ts_obs_store, ts_mod_store=ts_mod_store,
        heatmap_rows=heatmap_rows,
    )

    # ══════════════════════════════════════════════════════════════════════════
    #  PRECIPITATION INDICES — DURATION
    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== Precipitation Indices — Duration ===")

    # ── Dry days: P < 1 mm day⁻¹ ─────────────────────────────────────────────
    print("Dry_days: computing dry-day counts …")
    dry_model = annual_dry_days(pr_model)
    dry_obs   = annual_dry_days(pr_obs)
    save_index(dry_model, "Dry_days", "ICON")
    save_index(dry_obs,   "Dry_days", "EOBS")

    process_index(
        name="Dry_days",
        long_name="Dry days — P < 1 mm [days summer⁻¹]",
        annual_model=dry_model, annual_obs=dry_obs,
        thr_model=None,          thr_obs=None,
        unit="days summer⁻¹",    trend_unit="days decade⁻¹",
        trend_levels=PREC_LEVELS, colors=PREC_COLORS, tick_fmt="%.0f",
        gdf=gdf, geom=geom,
        summary_rows=summary_rows, overview_store=overview_store,
        ts_obs_store=ts_obs_store, ts_mod_store=ts_mod_store,
        heatmap_rows=heatmap_rows,
    )

    # ── CDD: Maximum consecutive dry days ─────────────────────────────────────
    print("CDD: computing maximum consecutive dry days (may take a few minutes) …")
    cdd_model = annual_cdd(pr_model)
    cdd_obs   = annual_cdd(pr_obs)
    save_index(cdd_model, "CDD", "ICON")
    save_index(cdd_obs,   "CDD", "EOBS")

    process_index(
        name="CDD",
        long_name="CDD — Max consecutive dry days [days summer⁻¹]",
        annual_model=cdd_model, annual_obs=cdd_obs,
        thr_model=None,          thr_obs=None,
        unit="days summer⁻¹",    trend_unit="days decade⁻¹",
        trend_levels=PREC_LEVELS, colors=PREC_COLORS, tick_fmt="%.0f",
        gdf=gdf, geom=geom,
        summary_rows=summary_rows, overview_store=overview_store,
        ts_obs_store=ts_obs_store, ts_mod_store=ts_mod_store,
        heatmap_rows=heatmap_rows,
    )

    # ── CWD: Maximum consecutive wet days ─────────────────────────────────────
    print("CWD: computing maximum consecutive wet days (may take a few minutes) …")
    cwd_model = annual_cwd(pr_model)
    cwd_obs   = annual_cwd(pr_obs)
    save_index(cwd_model, "CWD", "ICON")
    save_index(cwd_obs,   "CWD", "EOBS")

    process_index(
        name="CWD",
        long_name="CWD — Max consecutive wet days [days summer⁻¹]",
        annual_model=cwd_model, annual_obs=cwd_obs,
        thr_model=None,          thr_obs=None,
        unit="days summer⁻¹",    trend_unit="days decade⁻¹",
        trend_levels=CWD_LEVELS,  colors=PREC_COLORS, tick_fmt="%.1f",
        gdf=gdf, geom=geom,
        summary_rows=summary_rows, overview_store=overview_store,
        ts_obs_store=ts_obs_store, ts_mod_store=ts_mod_store,
        heatmap_rows=heatmap_rows,
    )

    # ══════════════════════════════════════════════════════════════════════════
    #  SUMMARY FIGURES
    # ══════════════════════════════════════════════════════════════════════════

    # ── Precipitation overview: all precip indices in one flagship figure ──────
    print("\nGenerating precipitation overview figure …")
    plot_precipitation_overview(
        index_meta=overview_store,
        gdf=gdf, geom=geom,
        outfile=os.path.join(FIGDIR, "precipitation_overview.png"),
    )

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
    print(f"  Summary figures     → {FIGDIR}/precipitation_overview.png")
    print(f"                        {FIGDIR}/taylor_diagram.png")
    print(f"                        {FIGDIR}/trend_heatmap.png")
    print(f"  Statistics table    → {csv_path}")
    print(f"  Annual NetCDF files → {NCDIR}/")
    print(f"  Indices processed   : {len(summary_rows)} total")
    print("  (Run script3_drivers.py next to analyse process drivers)")
    print("=" * 66)
