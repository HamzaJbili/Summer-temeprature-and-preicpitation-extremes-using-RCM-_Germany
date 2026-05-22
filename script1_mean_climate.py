"""
script1_mean_climate.py
-----------------------
JJA mean temperature and total precipitation: climatology, bias, anomalies,
and trend maps for ICON-CLM vs E-OBS over Germany, 1950–2022.

What this script produces
~~~~~~~~~~~~~~~~~~~~~~~~~
For each variable (JJA mean temperature, JJA total precipitation):

  1. Climatology map   — 3-panel: E-OBS | ICON-CLM | bias (ICON−E-OBS)
                         based on the 1991-2020 WMO normal.
  2. Trend map         — 2-panel: E-OBS | ICON-CLM Theil-Sen slopes
                         with significance stippling (MK p < 0.05).
  3. Time series       — 2-panel: annual Germany-average with 5-yr / 11-yr
                         running means, trend annotation, and anomaly bars.
  4. Summary CSV       — bias, trend magnitude, RMSE per variable.

Methodological choices
~~~~~~~~~~~~~~~~~~~~~~
  Anomaly / climatology reference : 1991-2020 (current WMO 30-yr normal)
  Trend method                    : Theil-Sen slope (per decade) +
                                    Mann-Kendall with Yue-Wang autocorr. correction
  Figure style                    : IPCC AR6 Working Group I (600 dpi)
"""

import os
import numpy as np
import pandas as pd

from utils import (
    load_field, keep_jja, annual_jja_mean, annual_jja_sum,
    reference_mean, compute_anomalies, area_mean, rmse,
    compute_trend_maps, series_stats,
    load_country_shape, plot_paired_trend_maps, plot_germany_series,
    plot_climatology_maps,
    set_ipcc_style,
    START_YEAR, END_YEAR, ANOM_START, ANOM_END, DPI,
)

# Apply IPCC publication style to all figures in this script
set_ipcc_style()

# ── file paths ─────────────────────────────────────────────────────────────────
MODEL_T_FILE = "ICONCLM_tas_daily_1950_2022_0.25deg_Germany.nc"
OBS_T_FILE   = "EOBS_tg_daily_1950_2022_0.25deg_Germany.nc"
MODEL_P_FILE = "ICONCLM_pr_daily_1950_2022_0.25deg_Germany.nc"
OBS_P_FILE   = "EOBS_rr_daily_1950_2022_0.25deg_Germany.nc"
GERMANY_SHP  = "/work/jbiliham/shapefile_Germany/gadm41_DEU_0.shp"

# ── output directories ─────────────────────────────────────────────────────────
FIGDIR = os.path.join("output_mean_climate", "figures")
TABDIR = os.path.join("output_mean_climate", "tables")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(TABDIR, exist_ok=True)

# ── colormap specifications ────────────────────────────────────────────────────
# Temperature trend map: blue → white → red (warming = red, IPCC RdBu_r)
TEMP_TREND_LEVELS = [-0.35, -0.25, -0.15, -0.10, -0.05,
                      0.00,  0.05,  0.10,  0.15,  0.20, 0.25, 0.35]
TEMP_TREND_COLORS = [
    "#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#f4f4f4", "#f7f7f7",
    "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#8c0d1c",
]

# Temperature climatology map: cool → warm sequential
TEMP_CLIM_LEVELS = [12, 13, 14, 15, 16, 17, 17.5, 18, 18.5, 19, 20, 21]
TEMP_CLIM_COLORS = [
    "#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8",
    "#ffffbf", "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026",
]

# Precipitation trend map: brown → white → green (wetter = green, IPCC BrBG)
PREC_TREND_LEVELS = [-60, -45, -30, -15, -5, 0, 5, 15, 30, 45, 60]
PREC_TREND_COLORS = [
    "#7f3b08", "#b35806", "#e08214", "#fdb863", "#fee0b6", "#f7f7f7",
    "#d8f0ed", "#a6dba0", "#5aae61", "#1b7837",
]

# Precipitation climatology map: white → blue sequential
PREC_CLIM_LEVELS = [100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 400]
PREC_CLIM_COLORS = [
    "#ffffff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6",
    "#4292c6", "#2171b5", "#08519c", "#08306b", "#041f5d", "#011037",
]


# ══════════════════════════════════════════════════════════════════════════════
#  Processing pipeline (one call per climate variable)
# ══════════════════════════════════════════════════════════════════════════════

def process_mean_index(
    name, long_name,
    annual_model, annual_obs,
    unit, trend_unit,
    trend_levels, trend_colors, tick_fmt_trend,
    clim_levels,  clim_colors,  tick_fmt_clim,
    gdf, geom, rows,
):
    """
    Full pipeline for one mean-climate variable.

    Steps
    -----
    1.  Compute 1991-2020 climatology and model bias.
    2.  Plot three-panel climatology / bias map.
    3.  Compute Theil-Sen slope and Mann-Kendall p-value at each grid cell.
    4.  Plot paired trend map (E-OBS | ICON-CLM) with stippling.
    5.  Compute Germany-average annual series and 1991-2020 anomalies.
    6.  Plot enhanced time series (rolling means, trend annotation, anomaly bars).
    7.  Compute and store summary statistics.

    Parameters
    ----------
    name, long_name : str
        Short identifier (used in filenames) and verbose display name.
    annual_model, annual_obs : xr.DataArray (year, lat, lon)
        Annual aggregates (mean for temperature, sum for precipitation).
    unit       : str — index unit  (e.g. "°C" or "mm season⁻¹")
    trend_unit : str — trend unit  (e.g. "°C decade⁻¹")
    trend_levels, trend_colors, tick_fmt_trend : trend map colormap spec
    clim_levels,  clim_colors,  tick_fmt_clim  : climatology colormap spec
    gdf, geom  : Germany shapefile and unified geometry
    rows       : list — appended with the statistics dict for CSV output
    """
    # ── 1. Climatology and bias ───────────────────────────────────────────────
    clim_obs   = reference_mean(annual_obs,   ANOM_START, ANOM_END)
    clim_model = reference_mean(annual_model, ANOM_START, ANOM_END)
    anom_obs   = compute_anomalies(annual_obs,   clim_obs)
    anom_model = compute_anomalies(annual_model, clim_model)

    # ── 2. Climatology map (E-OBS | ICON | bias) ─────────────────────────────
    print(f"    → climatology map …")
    plot_climatology_maps(
        obs_clim=clim_obs, mod_clim=clim_model,
        gdf=gdf, geom=geom,
        outfile=os.path.join(FIGDIR, f"{name}_climatology_map.png"),
        levels=clim_levels, colors=clim_colors,
        cbar_label=f"{long_name} [{unit}]",
        tick_fmt=tick_fmt_clim,
        suptitle=f"{long_name} — Climatology 1991–2020",
    )

    # ── 3. Trend maps ─────────────────────────────────────────────────────────
    print(f"    → trend maps …")
    trend_obs   = compute_trend_maps(annual_obs)
    trend_model = compute_trend_maps(annual_model)

    # ── 4. Paired trend map figure ────────────────────────────────────────────
    plot_paired_trend_maps(
        obs_slope   = trend_obs["sen_slope"],
        model_slope = trend_model["sen_slope"],
        obs_pval    = trend_obs["mk_pvalue"],
        model_pval  = trend_model["mk_pvalue"],
        gdf=gdf, geom=geom,
        outfile     = os.path.join(FIGDIR, f"{name}_trend_map.png"),
        levels=trend_levels, colors=trend_colors,
        cbar_label  = trend_unit, tick_fmt=tick_fmt_trend,
        suptitle    = f"{long_name} — Theil-Sen Trend 1950–2022",
    )

    # ── 5. Germany-average series ─────────────────────────────────────────────
    obs_series   = area_mean(annual_obs)
    model_series = area_mean(annual_model)
    obs_anom_s   = area_mean(anom_obs)
    mod_anom_s   = area_mean(anom_model)

    obs_stats   = series_stats(obs_series)
    model_stats = series_stats(model_series)

    # ── 6. Enhanced time series figure ───────────────────────────────────────
    print(f"    → time series …")
    plot_germany_series(
        obs_series, model_series,
        obs_anom_s, mod_anom_s,
        ylabel      = f"{long_name} [{unit}]",
        ylabel_anom = f"Anomaly [{unit}]",
        title       = f"{long_name}  —  Germany average, JJA 1950–2022",
        outfile     = os.path.join(FIGDIR, f"{name}_germany_series.png"),
        obs_stats   = obs_stats,
        model_stats = model_stats,
    )

    # ── 7. Summary statistics ─────────────────────────────────────────────────
    mean_bias = float(np.nanmean(model_series.values - obs_series.values))

    rows.append({
        "index":      name,
        "unit":       unit,
        "trend_unit": trend_unit,

        # 1991-2020 climatology
        "EOBS_climatology_mean_1991_2020":  _fmt(float(clim_obs.mean(skipna=True).values)),
        "ICON_climatology_mean_1991_2020":  _fmt(float(clim_model.mean(skipna=True).values)),
        "climatology_bias_ICON_minus_EOBS": _fmt(
            float(clim_model.mean(skipna=True).values)
            - float(clim_obs.mean(skipna=True).values)),

        # Gridcell-average trend (Theil-Sen per decade)
        "EOBS_mean_gridcell_trend":   _fmt(float(np.nanmean(trend_obs["sen_slope"].values))),
        "ICON_mean_gridcell_trend":   _fmt(float(np.nanmean(trend_model["sen_slope"].values))),
        "trend_bias_ICON_minus_EOBS": _fmt(float(np.nanmean(
            trend_model["sen_slope"].values - trend_obs["sen_slope"].values))),

        # Fraction of grid cells with significant trend (MK p < 0.05)
        "EOBS_sig_grid_fraction": _fmt(float(np.nanmean(
            (trend_obs["mk_pvalue"].values   < 0.05).astype(float)))),
        "ICON_sig_grid_fraction": _fmt(float(np.nanmean(
            (trend_model["mk_pvalue"].values < 0.05).astype(float)))),

        # Germany-average series: Theil-Sen + Mann-Kendall
        "EOBS_series_sen_slope_decade": _fmt(obs_stats["sen_slope_decade"]),
        "ICON_series_sen_slope_decade": _fmt(model_stats["sen_slope_decade"]),
        "EOBS_series_MK_p":  _fmt4(obs_stats["mk_p"]),
        "ICON_series_MK_p":  _fmt4(model_stats["mk_p"]),

        # Model–observation agreement
        "series_mean_bias_ICON_minus_EOBS": _fmt(mean_bias),
        "series_RMSE": _fmt(rmse(model_series.values, obs_series.values)),
    })


def _fmt(v):
    return round(float(v), 3) if np.isfinite(float(v)) else float("nan")

def _fmt4(v):
    return round(float(v), 4) if np.isfinite(float(v)) else float("nan")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("Loading Germany boundary shapefile …")
    gdf, geom = load_country_shape(GERMANY_SHP)

    print("Loading daily data …")
    tas_model_jja = keep_jja(load_field(MODEL_T_FILE, "tas"))
    tas_obs_jja   = keep_jja(load_field(OBS_T_FILE,   "tg"))
    pr_model_jja  = keep_jja(load_field(MODEL_P_FILE, "pr"))
    pr_obs_jja    = keep_jja(load_field(OBS_P_FILE,   "rr"))

    print("Computing annual aggregates …")
    tas_model_annual = annual_jja_mean(tas_model_jja)
    tas_obs_annual   = annual_jja_mean(tas_obs_jja)
    pr_model_annual  = annual_jja_sum(pr_model_jja)
    pr_obs_annual    = annual_jja_sum(pr_obs_jja)

    rows = []

    # ── JJA mean temperature ──────────────────────────────────────────────────
    print("\nProcessing JJA mean temperature …")
    process_mean_index(
        name      = "JJA_mean_temperature",
        long_name = "JJA Mean Temperature",
        annual_model = tas_model_annual,
        annual_obs   = tas_obs_annual,
        unit       = "°C",
        trend_unit = "°C decade⁻¹",
        trend_levels = TEMP_TREND_LEVELS,
        trend_colors = TEMP_TREND_COLORS,
        tick_fmt_trend = "%.2f",
        clim_levels  = TEMP_CLIM_LEVELS,
        clim_colors  = TEMP_CLIM_COLORS,
        tick_fmt_clim = "%.0f",
        gdf=gdf, geom=geom, rows=rows,
    )

    # ── JJA total precipitation ───────────────────────────────────────────────
    print("\nProcessing JJA total precipitation …")
    process_mean_index(
        name      = "JJA_total_precipitation",
        long_name = "JJA Total Precipitation",
        annual_model = pr_model_annual,
        annual_obs   = pr_obs_annual,
        unit       = "mm season⁻¹",
        trend_unit = "mm decade⁻¹",
        trend_levels = PREC_TREND_LEVELS,
        trend_colors = PREC_TREND_COLORS,
        tick_fmt_trend = "%.0f",
        clim_levels  = PREC_CLIM_LEVELS,
        clim_colors  = PREC_CLIM_COLORS,
        tick_fmt_clim = "%.0f",
        gdf=gdf, geom=geom, rows=rows,
    )

    # ── Save summary CSV ──────────────────────────────────────────────────────
    csv_path = os.path.join(TABDIR, "mean_climate_summary.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print("\n" + "=" * 60)
    print("Script 1 complete.")
    print(f"  Figures → {FIGDIR}/")
    print(f"    JJA_mean_temperature_climatology_map.png")
    print(f"    JJA_mean_temperature_trend_map.png")
    print(f"    JJA_mean_temperature_germany_series.png")
    print(f"    JJA_total_precipitation_climatology_map.png")
    print(f"    JJA_total_precipitation_trend_map.png")
    print(f"    JJA_total_precipitation_germany_series.png")
    print(f"  Table  → {csv_path}")
    print("  (Run script2_extremes.py next)")
    print("=" * 60)
