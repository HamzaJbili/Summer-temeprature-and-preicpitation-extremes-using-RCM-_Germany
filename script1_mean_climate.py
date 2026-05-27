"""
script1_mean_climate.py
-----------------------
JJA mean temperature (°C) and mean precipitation (mm day⁻¹) climatology,
anomalies, and trend maps for ICON-CLM vs E-OBS over Germany.

Input data        : CDO-preprocessed annual JJA means (one value per year per grid cell)
                    Temperature : annual JJA mean (°C)
                    Precipitation : annual JJA mean (mm day⁻¹)
Anomaly reference : 1991-2020  (WMO current normal)
Trend method      : Theil-Sen slope + Mann-Kendall (Yue-Wang autocorr. correction)
Output            : paired trend maps, Germany-average time series, summary CSV
"""

import os
import numpy as np
import pandas as pd
import xarray as xr

from utils import (
    load_field,
    reference_mean, compute_anomalies, area_mean, rmse,
    compute_trend_maps, series_stats,
    load_country_shape, build_mask,
    plot_paired_trend_maps, plot_obs_bias_maps, plot_germany_series,
    set_ipcc_style,
    START_YEAR, END_YEAR, ANOM_START, ANOM_END, DPI,
)

# ── file paths ────────────────────────────────────────────────────────────────
MODEL_T_FILE = "ICONCLM_tas_daily_1950_2022_0.25deg_Germany.nc"
OBS_T_FILE   = "EOBS_tg_daily_1950_2022_0.25deg_Germany.nc"
MODEL_P_FILE = "ICONCLM_pr_daily_1950_2022_0.25deg_Germany.nc"
OBS_P_FILE   = "EOBS_rr_daily_1950_2022_0.25deg_Germany.nc"
GERMANY_SHP  = "/work/jbiliham/shapefile_Germany/gadm41_DEU_0.shp"

# ── output directories ────────────────────────────────────────────────────────
FIGDIR = os.path.join("output_mean_climate", "figures")
TABDIR = os.path.join("output_mean_climate", "tables")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(TABDIR, exist_ok=True)


# ── colormap settings ─────────────────────────────────────────────────────────
# Sequential red palette for temperature trends.
# Germany JJA trends are all positive (warming); a diverging palette centred
# on zero wastes the blue half and compresses all values into a narrow dark-red
# band.  A sequential palette from near-white to deep red gives full
# perceptual discrimination across the observed 0.0–0.60 °C/decade range.
# Rule: len(LEVELS) = len(COLORS) + 1
TEMP_LEVELS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
TEMP_COLORS = [
    "#fff5f0",   # near-white  (0.00–0.05)
    "#fee0d2",   # very pale   (0.05–0.10)
    "#fcbba1",   # light pink  (0.10–0.15)
    "#fc9272",   # light red   (0.15–0.20)
    "#fb6a4a",   # medium-light(0.20–0.25)
    "#ef3b2c",   # medium red  (0.25–0.30)
    "#cb181d",   # strong red  (0.30–0.35)
    "#a50f15",   # dark red    (0.35–0.40)
    "#820808",   # darker red  (0.40–0.45)
    "#67000d",   # very dark   (0.45–0.50)
    "#3d0007",   # deepest red (0.50–0.60)
]

PREC_LEVELS = [-0.30, -0.20, -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20, 0.30]
PREC_COLORS = [
    "#7f3b08", "#b35806", "#e08214", "#fdb863", "#fee0b6", "#f7f7f7",
    "#d8f0ed", "#a6dba0", "#5aae61", "#1b7837",
]

import matplotlib.pyplot as plt


# ── CDO file converter ────────────────────────────────────────────────────────
def _to_annual(da):
    """
    Convert a CDO annual-mean file to a year-indexed DataArray.

    CDO writes annual means with a 'time' dimension whose coordinate values
    are full datetime stamps (e.g. 1950-07-16).  Downstream functions
    (reference_mean, compute_trend_maps, …) expect a 'year' integer dimension.
    This function extracts the year from each time stamp and swaps the
    dimension so the result is indexed by year (e.g. 1950, 1951, …, 2022).
    """
    years = da["time"].dt.year.values                     # e.g. array([1950, 1951, …, 2022])
    return (
        da.assign_coords(year=("time", years))            # add 'year' as a coordinate on 'time'
          .swap_dims({"time": "year"})                    # make 'year' the active dimension
          .drop_vars("time", errors="ignore")             # remove the old datetime coordinate
    )


# ── processing helper ─────────────────────────────────────────────────────────
def process_mean_index(name, annual_model, annual_obs, unit,
                       trend_unit, levels, colors, tick_fmt,
                       gdf, geom, rows):
    """
    Full pipeline for one mean-climate variable:
    anomalies, trend maps, Germany-average plots, summary stats.
    """
    # Anomalies relative to 1991-2020
    clim_obs   = reference_mean(annual_obs,   ANOM_START, ANOM_END)
    clim_model = reference_mean(annual_model, ANOM_START, ANOM_END)
    anom_obs   = compute_anomalies(annual_obs,   clim_obs)
    anom_model = compute_anomalies(annual_model, clim_model)

    # Trend maps
    trend_obs   = compute_trend_maps(annual_obs)
    trend_model = compute_trend_maps(annual_model)

    # Paired trend maps figure
    plot_paired_trend_maps(
        obs_slope   = trend_obs["sen_slope"],
        model_slope = trend_model["sen_slope"],
        obs_pval    = trend_obs["mk_pvalue"],
        model_pval  = trend_model["mk_pvalue"],
        gdf=gdf, geom=geom,
        outfile     = os.path.join(FIGDIR, f"{name}_trend_map.png"),
        levels=levels, colors=colors,
        cbar_label  = trend_unit, tick_fmt=tick_fmt,
    )

    # Supplementary: E-OBS trend + bias side by side with independent colorbars
    plot_obs_bias_maps(
        obs_slope   = trend_obs["sen_slope"],
        model_slope = trend_model["sen_slope"],
        obs_pval    = trend_obs["mk_pvalue"],
        model_pval  = trend_model["mk_pvalue"],
        gdf=gdf, geom=geom,
        outfile     = os.path.join(FIGDIR, f"{name}_obs_bias_map.png"),
        obs_levels=levels, obs_colors=colors,
        cbar_label  = trend_unit, tick_fmt=tick_fmt,
    )

    # Germany-average series
    obs_series   = area_mean(annual_obs)
    model_series = area_mean(annual_model)
    obs_anom_series   = area_mean(anom_obs)
    model_anom_series = area_mean(anom_model)

    plot_germany_series(
        obs_series, model_series,
        obs_anom_series, model_anom_series,
        ylabel      = unit,
        ylabel_anom = f"Anomaly ({unit})",
        title       = name.replace("_", " "),
        outfile     = os.path.join(FIGDIR, f"{name}_germany_series.png"),
    )

    # Summary statistics
    obs_stats   = series_stats(obs_series)
    model_stats = series_stats(model_series)
    mean_bias   = float(np.nanmean(model_series.values - obs_series.values))

    # Germany mask — used to count only German grid cells for the sig fraction
    de_mask_obs = build_mask(trend_obs["lon"].values,   trend_obs["lat"].values,   geom)
    de_mask_mod = build_mask(trend_model["lon"].values, trend_model["lat"].values, geom)

    rows.append({
        "index":    name,
        "unit":     unit,
        "trend_unit": trend_unit,

        # Climatology (1991-2020) bias
        "EOBS_climatology_mean_1991_2020":  round(float(clim_obs.mean(skipna=True).values),   3),
        "ICON_climatology_mean_1991_2020":  round(float(clim_model.mean(skipna=True).values), 3),
        "climatology_bias_ICON_minus_EOBS": round(
            float(clim_model.mean(skipna=True).values - clim_obs.mean(skipna=True).values), 3),

        # Trend maps
        "EOBS_mean_gridcell_trend":  round(float(np.nanmean(trend_obs["sen_slope"].values)),   3),
        "ICON_mean_gridcell_trend":  round(float(np.nanmean(trend_model["sen_slope"].values)), 3),
        "trend_bias_ICON_minus_EOBS": round(
            float(np.nanmean(trend_model["sen_slope"].values
                             - trend_obs["sen_slope"].values)), 3),

        # Fraction of significant grid cells — Germany cells only, expressed as %
        "EOBS_sig_grid_pct": round(
            float((trend_obs["mk_pvalue"].values[de_mask_obs]   < 0.05).mean()) * 100, 1),
        "ICON_sig_grid_pct": round(
            float((trend_model["mk_pvalue"].values[de_mask_mod] < 0.05).mean()) * 100, 1),

        # Germany-average series
        "EOBS_series_sen_slope_decade": round(obs_stats["sen_slope_decade"],   3),
        "ICON_series_sen_slope_decade": round(model_stats["sen_slope_decade"], 3),
        "EOBS_series_MK_p":  round(obs_stats["mk_p"],   4),
        "ICON_series_MK_p":  round(model_stats["mk_p"], 4),
        "series_mean_bias_ICON_minus_EOBS": round(mean_bias, 3),
        "series_RMSE": round(rmse(model_series.values, obs_series.values), 3),
    })


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    set_ipcc_style()   # IPCC AR6 WGI fonts, sizes, and figure defaults

    print("Loading Germany boundary...")
    gdf, geom = load_country_shape(GERMANY_SHP)

    # Input files are CDO-preprocessed annual JJA means — no in-Python
    # seasonal extraction or aggregation needed.  _to_annual() converts the
    # CDO 'time' dimension to an integer 'year' dimension expected downstream.

    print("Loading temperature data (annual JJA means, °C)...")
    tas_model_annual = _to_annual(load_field(MODEL_T_FILE, "tas"))
    tas_obs_annual   = _to_annual(load_field(OBS_T_FILE,   "tg"))

    print("Loading precipitation data (annual JJA means, mm day⁻¹)...")
    pr_model_annual  = _to_annual(load_field(MODEL_P_FILE, "pr"))
    pr_obs_annual    = _to_annual(load_field(OBS_P_FILE,   "rr"))

    rows = []

    print("Processing JJA mean temperature...")
    process_mean_index(
        name="JJA_mean_temperature",
        annual_model=tas_model_annual,
        annual_obs=tas_obs_annual,
        unit="°C",
        trend_unit="°C decade⁻¹",
        levels=TEMP_LEVELS,
        colors=TEMP_COLORS,
        tick_fmt="%.2f",
        gdf=gdf, geom=geom, rows=rows,
    )

    print("Processing JJA mean precipitation...")
    process_mean_index(
        name="JJA_mean_precipitation",
        annual_model=pr_model_annual,
        annual_obs=pr_obs_annual,
        unit="mm day⁻¹",
        trend_unit="mm day⁻¹ decade⁻¹",
        levels=PREC_LEVELS,
        colors=PREC_COLORS,
        tick_fmt="%.2f",
        gdf=gdf, geom=geom, rows=rows,
    )

    pd.DataFrame(rows).to_csv(
        os.path.join(TABDIR, "mean_climate_summary.csv"), index=False)

    print(f"\nDone.  Figures → {FIGDIR}   Tables → {TABDIR}")
