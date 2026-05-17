"""
script2_extremes.py
-------------------
Six annual JJA extreme indices for ICON-CLM vs E-OBS over Germany.

Indices computed
  Temperature : T95 exceedance days, Heatwave Number (HWN)
  Precipitation: R95p days, R99p days, Dry days (P<1mm), CDD (max dry spell)

Threshold reference : 1961-1990  (WMO recommendation for extreme trend studies)
Anomaly reference   : 1991-2020
Trend method        : Theil-Sen + Mann-Kendall (Yue-Wang autocorr. correction)

Annual index arrays are saved as NetCDF so that script3 can read them directly.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr

from utils import (
    load_field, keep_jja, reference_mean, compute_anomalies,
    area_mean, rmse, compute_trend_maps, series_stats,
    load_country_shape, plot_paired_trend_maps, plot_germany_series,
    REF_START, REF_END, ANOM_START, ANOM_END, DPI,
)

import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 6, "ytick.labelsize": 6,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})

# ── file paths ────────────────────────────────────────────────────────────────
MODEL_T_FILE = "ICONCLM_tas_daily_1950_2022_0.25deg_Germany.nc"
OBS_T_FILE   = "EOBS_tg_daily_1950_2022_0.25deg_Germany.nc"
MODEL_P_FILE = "ICONCLM_pr_daily_1950_2022_0.25deg_Germany.nc"
OBS_P_FILE   = "EOBS_rr_daily_1950_2022_0.25deg_Germany.nc"
GERMANY_SHP  = "/work/jbiliham/shapefile_Germany/gadm41_DEU_0.shp"

FIGDIR  = os.path.join("output_extremes", "figures")
TABDIR  = os.path.join("output_extremes", "tables")
NCDIR   = os.path.join("output_extremes", "netcdf")         # annual index NetCDFs
for d in [FIGDIR, TABDIR, NCDIR]:
    os.makedirs(d, exist_ok=True)

WET_DAY_MIN = 1.0   # mm/d — defines a wet day for R95 / R99 thresholds
DRY_DAY_MAX = 1.0   # mm/d — defines a dry day for dry-day count and CDD
HW_MIN_LEN  = 3     # minimum consecutive T95 exceedance days → 1 heatwave event

# ── colormap settings ─────────────────────────────────────────────────────────
TEMP_LEVELS = [-8, -6, -4, -2, -1, 0, 1, 2, 4, 6, 8]
TEMP_COLORS = [
    "#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#f7f7f7",
    "#fee0b6", "#fdb863", "#f4a582", "#d6604d", "#b2182b",
]

HW_LEVELS = [-2.0, -1.5, -1.0, -0.5, -0.25, 0, 0.25, 0.5, 1.0, 1.5, 2.0]
HW_COLORS = TEMP_COLORS  # same diverging palette

PREC_LEVELS = [-8, -6, -4, -2, -1, 0, 1, 2, 4, 6, 8]
PREC_COLORS = [
    "#7f3b08", "#b35806", "#e08214", "#fdb863", "#fee0b6", "#f7f7f7",
    "#d8f0ed", "#a6dba0", "#5aae61", "#1b7837",
]


# ── threshold estimation ──────────────────────────────────────────────────────
def percentile_threshold(daily_jja, q, wet_only=False):
    """
    Local q-th percentile from the 1961-1990 reference period.
    If wet_only=True, only wet days (>= WET_DAY_MIN) are included
    in the quantile calculation (required for R95 / R99).
    """
    ref = daily_jja.sel(time=slice(REF_START, REF_END))
    if wet_only:
        ref = ref.where(ref >= WET_DAY_MIN)
    thr = ref.quantile(q / 100.0, dim="time", skipna=True)
    if "quantile" in thr.dims:
        thr = thr.squeeze("quantile", drop=True)
    return thr.astype(np.float32)


# ── annual index computations ─────────────────────────────────────────────────
def annual_exceedance_days(daily_jja, threshold):
    """Count JJA days per year exceeding a gridded threshold (vectorized)."""
    exceed = (daily_jja > threshold).astype(np.float32)
    return exceed.groupby("time.year").sum("time")


def annual_dry_days(daily_jja):
    """Count JJA days per year with precipitation < DRY_DAY_MAX (vectorized)."""
    dry = (daily_jja < DRY_DAY_MAX).astype(np.float32)
    return dry.groupby("time.year").sum("time")


def annual_cdd(daily_jja):
    """
    Maximum consecutive dry days (CDD) per JJA season at each grid cell.
    Fully vectorized using NumPy run-length encoding.
    A dry day is defined as P < DRY_DAY_MAX.
    """
    years = np.unique(daily_jja["time"].dt.year.values).astype(int)
    lat   = daily_jja["lat"].values
    lon   = daily_jja["lon"].values
    out   = np.zeros((len(years), len(lat), len(lon)), dtype=np.float32)

    for yi, yr in enumerate(years):
        # binary dry-day array: (time, lat, lon)
        dry = (daily_jja.sel(time=str(yr)).values < DRY_DAY_MAX).astype(np.int8)
        # max dry spell per column using a 1-D diff trick
        # For each (i,j), find the longest run of 1s
        T = dry.shape[0]
        # Pad with zeros at start and end to detect run boundaries
        pad  = np.zeros((1, len(lat), len(lon)), dtype=np.int8)
        dry_p = np.concatenate([pad, dry, pad], axis=0)      # (T+2, lat, lon)
        # transitions: +1 = start of dry run, -1 = end
        diff  = np.diff(dry_p.astype(np.int16), axis=0)      # (T+1, lat, lon)
        # At each grid cell find max run length
        # This is done with a vectorized cumulative-sum approach
        cum = np.cumsum(dry, axis=0)    # (T, lat, lon)

        for i in range(len(lat)):
            for j in range(len(lon)):
                d = dry[:, i, j]
                if d.sum() == 0:
                    out[yi, i, j] = 0
                    continue
                # Standard run-length encoding via diff
                starts = np.where(np.diff(np.concatenate([[0], d, [0]])) == 1)[0]
                ends   = np.where(np.diff(np.concatenate([[0], d, [0]])) == -1)[0]
                out[yi, i, j] = float(np.max(ends - starts))

    return xr.DataArray(
        out, coords={"year": years, "lat": lat, "lon": lon},
        dims=("year", "lat", "lon"), name="CDD",
    )


def annual_heatwave_number(daily_jja, threshold):
    """
    Count heatwave events per JJA season.
    A heatwave = run of >= HW_MIN_LEN consecutive days with T > threshold.
    Vectorized: one Python loop over years only; inner loop is NumPy diff.
    """
    years = np.unique(daily_jja["time"].dt.year.values).astype(int)
    lat   = daily_jja["lat"].values
    lon   = daily_jja["lon"].values
    out   = np.zeros((len(years), len(lat), len(lon)), dtype=np.float32)

    thr_vals = threshold.values  # (lat, lon)

    for yi, yr in enumerate(years):
        hot = (daily_jja.sel(time=str(yr)).values > thr_vals).astype(np.int8)
        # hot shape: (time, lat, lon)
        # Pad time axis
        pad  = np.zeros((1, len(lat), len(lon)), dtype=np.int8)
        hot_p = np.concatenate([pad, hot, pad], axis=0)
        diff  = np.diff(hot_p.astype(np.int16), axis=0)

        for i in range(len(lat)):
            for j in range(len(lon)):
                d = hot[:, i, j]
                if d.sum() < HW_MIN_LEN:
                    out[yi, i, j] = 0
                    continue
                starts = np.where(np.diff(np.concatenate([[0], d, [0]])) ==  1)[0]
                ends   = np.where(np.diff(np.concatenate([[0], d, [0]])) == -1)[0]
                lengths = ends - starts
                out[yi, i, j] = float(np.sum(lengths >= HW_MIN_LEN))

    return xr.DataArray(
        out, coords={"year": years, "lat": lat, "lon": lon},
        dims=("year", "lat", "lon"), name="HWN",
    )


# ── save annual index to NetCDF ───────────────────────────────────────────────
def save_index(da, name, dataset_label):
    """Save annual index DataArray to NetCDF for use by script3."""
    path = os.path.join(NCDIR, f"{name}_{dataset_label}_annual.nc")
    da.to_netcdf(path)
    return path


# ── full index processing pipeline ───────────────────────────────────────────
def process_index(name, annual_model, annual_obs,
                  thr_model, thr_obs,
                  unit, trend_unit, levels, colors, tick_fmt,
                  gdf, geom, rows):
    """
    For one index: trend maps, Germany-average series, summary statistics.
    Annual index NetCDFs are saved separately via save_index().
    """
    trend_obs   = compute_trend_maps(annual_obs)
    trend_model = compute_trend_maps(annual_model)

    plot_paired_trend_maps(
        obs_slope=trend_obs["sen_slope"], model_slope=trend_model["sen_slope"],
        obs_pval=trend_obs["mk_pvalue"],  model_pval=trend_model["mk_pvalue"],
        gdf=gdf, geom=geom,
        outfile=os.path.join(FIGDIR, f"{name}_trend_map.png"),
        levels=levels, colors=colors,
        cbar_label=trend_unit, tick_fmt=tick_fmt,
    )

    # Germany-average absolute series and anomalies (1991-2020)
    obs_series   = area_mean(annual_obs)
    model_series = area_mean(annual_model)
    obs_anom     = compute_anomalies(obs_series,   reference_mean(obs_series,   ANOM_START, ANOM_END))
    model_anom   = compute_anomalies(model_series, reference_mean(model_series, ANOM_START, ANOM_END))

    plot_germany_series(
        obs_series, model_series, obs_anom, model_anom,
        ylabel=unit, ylabel_anom=f"Anomaly ({unit})",
        title=name.replace("_", " "),
        outfile=os.path.join(FIGDIR, f"{name}_germany_series.png"),
    )

    obs_stats   = series_stats(obs_series)
    model_stats = series_stats(model_series)

    thr_obs_mean   = float(np.nanmean(thr_obs.values))   if thr_obs   is not None else np.nan
    thr_model_mean = float(np.nanmean(thr_model.values)) if thr_model is not None else np.nan
    thr_bias = thr_model_mean - thr_obs_mean if np.isfinite(thr_obs_mean) else np.nan

    rows.append({
        "index":       name,
        "unit":        unit,
        "trend_unit":  trend_unit,

        "EOBS_threshold_mean":               round(thr_obs_mean,   3) if np.isfinite(thr_obs_mean)   else np.nan,
        "ICON_threshold_mean":               round(thr_model_mean, 3) if np.isfinite(thr_model_mean) else np.nan,
        "threshold_bias_ICON_minus_EOBS":    round(thr_bias,       3) if np.isfinite(thr_bias)       else np.nan,

        "EOBS_mean_gridcell_trend":          round(float(np.nanmean(trend_obs["sen_slope"].values)),   3),
        "ICON_mean_gridcell_trend":          round(float(np.nanmean(trend_model["sen_slope"].values)), 3),
        "trend_bias_ICON_minus_EOBS":        round(float(np.nanmean(
            trend_model["sen_slope"].values - trend_obs["sen_slope"].values)), 3),

        "EOBS_sig_grid_fraction":  round(float(np.nanmean((trend_obs["mk_pvalue"].values   < 0.05).astype(float))), 3),
        "ICON_sig_grid_fraction":  round(float(np.nanmean((trend_model["mk_pvalue"].values < 0.05).astype(float))), 3),

        "EOBS_series_sen_slope_decade": round(obs_stats["sen_slope_decade"],   3),
        "ICON_series_sen_slope_decade": round(model_stats["sen_slope_decade"], 3),
        "EOBS_series_MK_p":  round(obs_stats["mk_p"],   4),
        "ICON_series_MK_p":  round(model_stats["mk_p"], 4),
        "series_mean_bias_ICON_minus_EOBS": round(float(np.nanmean(
            model_series.values - obs_series.values)), 3),
        "series_RMSE": round(rmse(model_series.values, obs_series.values), 3),
    })


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("Loading Germany boundary...")
    gdf, geom = load_country_shape(GERMANY_SHP)

    print("Loading daily data...")
    tas_model = keep_jja(load_field(MODEL_T_FILE, "tas"))
    tas_obs   = keep_jja(load_field(OBS_T_FILE,   "tg"))
    pr_model  = keep_jja(load_field(MODEL_P_FILE, "pr"))
    pr_obs    = keep_jja(load_field(OBS_P_FILE,   "rr"))

    rows = []

    # ── T95 exceedance days ───────────────────────────────────────────────────
    print("Computing T95 thresholds...")
    t95_model = percentile_threshold(tas_model, 95)
    t95_obs   = percentile_threshold(tas_obs,   95)

    print("Computing T95 exceedance days...")
    t95_days_model = annual_exceedance_days(tas_model, t95_model)
    t95_days_obs   = annual_exceedance_days(tas_obs,   t95_obs)

    save_index(t95_days_model, "T95_days", "ICON")
    save_index(t95_days_obs,   "T95_days", "EOBS")

    process_index(
        "T95_exceedance_days",
        t95_days_model, t95_days_obs,
        t95_model, t95_obs,
        unit="days summer-1", trend_unit="days / decade",
        levels=TEMP_LEVELS, colors=TEMP_COLORS, tick_fmt="%.0f",
        gdf=gdf, geom=geom, rows=rows,
    )

    # ── Heatwave Number ───────────────────────────────────────────────────────
    print("Computing Heatwave Number  (may take a few minutes)...")
    hwn_model = annual_heatwave_number(tas_model, t95_model)
    hwn_obs   = annual_heatwave_number(tas_obs,   t95_obs)

    save_index(hwn_model, "HWN", "ICON")
    save_index(hwn_obs,   "HWN", "EOBS")

    process_index(
        "Heatwave_number",
        hwn_model, hwn_obs,
        t95_model, t95_obs,     # threshold is the same T95
        unit="events summer-1", trend_unit="events / decade",
        levels=HW_LEVELS, colors=HW_COLORS, tick_fmt="%.2f",
        gdf=gdf, geom=geom, rows=rows,
    )

    # ── R95p heavy precipitation days ─────────────────────────────────────────
    print("Computing R95 wet-day thresholds...")
    r95_model = percentile_threshold(pr_model, 95, wet_only=True)
    r95_obs   = percentile_threshold(pr_obs,   95, wet_only=True)

    print("Computing R95p days...")
    r95_days_model = annual_exceedance_days(pr_model, r95_model)
    r95_days_obs   = annual_exceedance_days(pr_obs,   r95_obs)

    save_index(r95_days_model, "R95p_days", "ICON")
    save_index(r95_days_obs,   "R95p_days", "EOBS")

    process_index(
        "R95p_exceedance_days",
        r95_days_model, r95_days_obs,
        r95_model, r95_obs,
        unit="days summer-1", trend_unit="days / decade",
        levels=PREC_LEVELS, colors=PREC_COLORS, tick_fmt="%.0f",
        gdf=gdf, geom=geom, rows=rows,
    )

    # ── R99p very heavy precipitation days ────────────────────────────────────
    print("Computing R99 wet-day thresholds...")
    r99_model = percentile_threshold(pr_model, 99, wet_only=True)
    r99_obs   = percentile_threshold(pr_obs,   99, wet_only=True)

    print("Computing R99p days...")
    r99_days_model = annual_exceedance_days(pr_model, r99_model)
    r99_days_obs   = annual_exceedance_days(pr_obs,   r99_obs)

    save_index(r99_days_model, "R99p_days", "ICON")
    save_index(r99_days_obs,   "R99p_days", "EOBS")

    process_index(
        "R99p_exceedance_days",
        r99_days_model, r99_days_obs,
        r99_model, r99_obs,
        unit="days summer-1", trend_unit="days / decade",
        levels=PREC_LEVELS, colors=PREC_COLORS, tick_fmt="%.0f",
        gdf=gdf, geom=geom, rows=rows,
    )

    # ── Dry days ──────────────────────────────────────────────────────────────
    print("Computing dry days...")
    dry_model = annual_dry_days(pr_model)
    dry_obs   = annual_dry_days(pr_obs)

    save_index(dry_model, "Dry_days", "ICON")
    save_index(dry_obs,   "Dry_days", "EOBS")

    process_index(
        "Dry_days",
        dry_model, dry_obs,
        None, None,             # fixed threshold — no percentile object
        unit="days summer-1", trend_unit="days / decade",
        levels=PREC_LEVELS, colors=PREC_COLORS, tick_fmt="%.0f",
        gdf=gdf, geom=geom, rows=rows,
    )

    # ── CDD — Consecutive Dry Days maximum ────────────────────────────────────
    print("Computing CDD (max consecutive dry days)...")
    cdd_model = annual_cdd(pr_model)
    cdd_obs   = annual_cdd(pr_obs)

    save_index(cdd_model, "CDD", "ICON")
    save_index(cdd_obs,   "CDD", "EOBS")

    process_index(
        "CDD",
        cdd_model, cdd_obs,
        None, None,
        unit="days summer-1", trend_unit="days / decade",
        levels=PREC_LEVELS, colors=PREC_COLORS, tick_fmt="%.0f",
        gdf=gdf, geom=geom, rows=rows,
    )

    # ── Save summary table ────────────────────────────────────────────────────
    pd.DataFrame(rows).to_csv(
        os.path.join(TABDIR, "extreme_indices_summary.csv"), index=False)

    print(f"\nDone.  Figures → {FIGDIR}   Tables → {TABDIR}   NetCDF → {NCDIR}")
