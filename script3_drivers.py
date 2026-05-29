"""
script3_drivers.py
------------------
Process-driver composite and correlation analysis.

For each extreme index, identifies summers in the upper quartile
of the Germany-average annual series (high-index years) and computes:
  (a) composite anomaly maps for each driver variable
  (b) a combined six-panel composite figure
  (c) Germany-average Pearson and Spearman correlations vs each driver

Composite anomaly definition
  composite = mean(driver anomaly | high-index years)
            - mean(driver anomaly | all other years)
  This is the standard definition; the driver variable is first expressed
  as an anomaly relative to the 1961-1990 reference climatology.

Requires annual index NetCDF files produced by script2_extremes.py.
Driver files (ICON-CLM or ERA5) must be provided externally.

Driver variables expected
  Z500  : geopotential height at 500 hPa   (m)
  SHF   : surface sensible heat flux        (W m-2)
  LHF   : surface latent heat flux          (W m-2)
  SM    : soil moisture (uppermost layer)   (m3 m-3 or kg m-2)
  WIND  : 10-metre wind speed               (m s-1)
  CAPE  : convective available PE           (J kg-1)
  CIN   : convective inhibition             (J kg-1)
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import pearsonr, spearmanr

from utils import (
    load_field, keep_jja, annual_jja_mean,
    reference_mean, compute_anomalies, area_mean,
    load_country_shape, interp_display, build_mask, apply_mask,
    clip_contourf, style_axis,
    START_YEAR, END_YEAR, REF_START, REF_END, DPI,
)

from utils import set_ipcc_style
set_ipcc_style()

# ── output directories ────────────────────────────────────────────────────────
FIGDIR = os.path.join("output_drivers", "figures")
TABDIR = os.path.join("output_drivers", "tables")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(TABDIR, exist_ok=True)

GERMANY_SHP  = "/work/jbiliham/shapefile_Germany/gadm41_DEU_0.shp"
INDEX_NC_DIR = "output_extremes/netcdf"   # produced by script2

# ── indices to analyse ────────────────────────────────────────────────────────
# Tuple: (display name, ICON nc stem, EOBS nc stem, dataset to use)
# "ICON" uses ICON-CLM driver files; "EOBS" uses ERA5 driver files.
# Both are listed; you can run one or both by setting DATASETS below.
INDICES = [
    # Temperature extremes
    ("T90p_exceedance_pct",  "T90p_pct"),
    ("Heatwave_number",      "HWN"),
    ("Heatwave_duration",    "HWD"),
    # Precipitation — heavy events
    ("R95p_exceedance_days", "R95p_days"),
    ("Rx1day",               "Rx1day"),
    ("Rx5day",               "Rx5day"),
    ("SDII",                 "SDII"),
    # Precipitation — concentration
    ("R95pTOT",              "R95pTOT"),
    # Precipitation — drought
    ("CDD",                  "CDD"),
]

# ── driver file configuration ─────────────────────────────────────────────────
# Provide one set of driver files per dataset label.
# Paths shown here are illustrative; adjust to your actual file names.
DRIVER_FILES = {
    "Z500": "z500_daily_1950_2022_0.25deg_Germany.nc",
    "SHF":  "sensible_heat_flux_daily_1950_2022_0.25deg_Germany.nc",
    "LHF":  "latent_heat_flux_daily_1950_2022_0.25deg_Germany.nc",
    "SM":   "soil_moisture_daily_1950_2022_0.25deg_Germany.nc",
    "WIND": "wind_sfc_daily_1950_2022_0.25deg_Germany.nc",
    "CAPE": "cape_daily_1950_2022_0.25deg_Germany.nc",
    "CIN":  "cin_daily_1950_2022_0.25deg_Germany.nc",
}

DRIVER_VARS = {
    "Z500": "z",
    "SHF":  "hfss",
    "LHF":  "hfls",
    "SM":   "mrso",
    "WIND": "sfcWind",
    "CAPE": "cape",
    "CIN":  "cin",
}

DRIVER_UNITS = {
    "Z500": "m",
    "SHF":  "W m$^{-2}$",
    "LHF":  "W m$^{-2}$",
    "SM":   "kg m$^{-2}$",
    "WIND": "m s$^{-1}$",
    "CAPE": "J kg$^{-1}$",
    "CIN":  "J kg$^{-1}$",
}

# Composite anomaly colormap levels for each driver (adjust after first run)
DRIVER_LEVELS = {
    "Z500": [-60, -40, -20, -10, -5, 0,  5, 10, 20, 40, 60],
    "SHF":  [-30, -20, -10,  -5, -2, 0,  2,  5, 10, 20, 30],
    "LHF":  [-30, -20, -10,  -5, -2, 0,  2,  5, 10, 20, 30],
    "SM":   [-30, -20, -10,  -5, -2, 0,  2,  5, 10, 20, 30],
    "WIND": [ -2,  -1, -.5,-.25,-.1, 0, .1, .25, .5,  1,  2],
    "CAPE": [-200,-150,-100,-50,-20, 0, 20, 50,100,150,200],
    "CIN":  [ -40, -30, -20,-10, -5, 0,  5, 10, 20, 30, 40],
}

# Generic diverging blue-white-red palette
DIV_COLORS = [
    "#2166ac","#4393c3","#92c5de","#d1e5f0","#f7f7f7",
    "#fddbc7","#f4a582","#d6604d","#b2182b","#67001f",
]


# ── helper: load annual index from script2 output ─────────────────────────────
def load_index(nc_stem, dataset_label):
    """Load annual index array produced by script2_extremes.py."""
    path = os.path.join(INDEX_NC_DIR, f"{nc_stem}_{dataset_label}_annual.nc")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Annual index file not found: {path}\n"
            "Run script2_extremes.py first to generate the NetCDF files."
        )
    ds = xr.open_dataset(path)
    da = ds[list(ds.data_vars)[0]]
    return da.sortby("lat").sortby("lon")


# ── composite definition ──────────────────────────────────────────────────────
def upper_quartile_years(annual_index, n_total=None):
    """
    Return years in the upper quartile of the Germany-average annual index.
    Upper quartile = top 25% of years = floor(n/4) years.
    """
    series = area_mean(annual_index)
    df = pd.DataFrame({
        "year":  series["year"].values.astype(int),
        "value": series.values,
    }).dropna()
    n_top = max(1, len(df) // 4)          # upper quartile
    top = df.nlargest(n_top, "value")
    return top["year"].values.astype(int), top


def composite_high_vs_rest(driver_anom, high_years, all_years):
    """
    Composite anomaly = mean over high-index years minus mean over all other years.
    Both means are of the driver variable expressed as anomaly from its climatology.
    """
    low_years = np.setdiff1d(all_years, high_years)
    mean_high = driver_anom.sel(year=high_years).mean("year", skipna=True)
    mean_low  = driver_anom.sel(year=low_years ).mean("year", skipna=True)
    return mean_high - mean_low


# ── plotting helpers ──────────────────────────────────────────────────────────
def plot_single_composite(da, gdf, geom, outfile, levels, cbar_label, title):
    cmap = mcolors.ListedColormap(DIV_COLORS)
    norm = mcolors.BoundaryNorm(levels, cmap.N)

    fig, ax = plt.subplots(1, 1, figsize=(4.2, 4.0))
    fig.patch.set_facecolor("white")

    fine = interp_display(da)
    mask = build_mask(fine["lon"].values, fine["lat"].values, geom)
    arr  = apply_mask(fine.values, mask)

    cf = ax.contourf(fine["lon"].values, fine["lat"].values, arr,
                     levels=levels, cmap=cmap, norm=norm,
                     extend="both", antialiased=True)
    clip_contourf(cf, ax, geom)
    gdf.boundary.plot(ax=ax, color="black", linewidth=0.45, zorder=5)
    ax.set_title(title, fontsize=8, pad=3)
    style_axis(ax)

    plt.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.22)
    cax = fig.add_axes([0.18, 0.09, 0.64, 0.045])
    cb  = ColorbarBase(cax, cmap=cmap, norm=norm, boundaries=levels,
                       ticks=levels, orientation="horizontal", extend="both")
    cb.ax.tick_params(labelsize=6, pad=1)
    cb.ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    cb.set_label(cbar_label, fontsize=7, labelpad=2)

    fig.savefig(outfile, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_six_panel_composites(composites, gdf, geom, outfile, suptitle):
    """
    2×4 grid figure (7 drivers; one spare slot used for a title block).
    Each panel has its own colorbar strip.
    """
    driver_names = list(composites.keys())
    nrows, ncols = 2, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.0, 6.4))
    fig.patch.set_facecolor("white")
    axes = axes.ravel()

    for k, dname in enumerate(driver_names):
        ax     = axes[k]
        da     = composites[dname]
        levels = DRIVER_LEVELS[dname]
        unit   = DRIVER_UNITS[dname]

        cmap = mcolors.ListedColormap(DIV_COLORS)
        norm = mcolors.BoundaryNorm(levels, cmap.N)

        fine = interp_display(da)
        mask = build_mask(fine["lon"].values, fine["lat"].values, geom)
        arr  = apply_mask(fine.values, mask)

        cf = ax.contourf(fine["lon"].values, fine["lat"].values, arr,
                         levels=levels, cmap=cmap, norm=norm,
                         extend="both", antialiased=True)
        clip_contourf(cf, ax, geom)
        gdf.boundary.plot(ax=ax, color="black", linewidth=0.40, zorder=5)
        ax.set_title(dname, fontsize=7, pad=2)
        style_axis(ax)

    # Hide the unused 8th panel
    axes[len(driver_names)].set_visible(False)

    fig.suptitle(suptitle, fontsize=9, y=0.99)
    plt.subplots_adjust(left=0.04, right=0.98, top=0.92,
                        bottom=0.10, wspace=0.18, hspace=0.32)

    # Add per-panel colorbars AFTER adjusting layout
    for k, dname in enumerate(driver_names):
        ax     = axes[k]
        levels = DRIVER_LEVELS[dname]
        unit   = DRIVER_UNITS[dname]
        cmap   = mcolors.ListedColormap(DIV_COLORS)
        norm   = mcolors.BoundaryNorm(levels, cmap.N)

        bbox = ax.get_position()
        cax  = fig.add_axes([bbox.x0 + 0.01, bbox.y0 - 0.028,
                              bbox.width - 0.02, 0.012])
        cb   = ColorbarBase(cax, cmap=cmap, norm=norm, boundaries=levels,
                            ticks=[levels[0], 0, levels[-1]],
                            orientation="horizontal", extend="both")
        cb.ax.tick_params(labelsize=5, pad=1)
        cb.ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        cb.set_label(unit, fontsize=5, labelpad=1)

    fig.savefig(outfile, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ── correlation analysis ──────────────────────────────────────────────────────
def compute_correlations(index_series, driver_anoms):
    """
    Germany-average annual Pearson and Spearman correlations
    between an extreme index and each driver variable.
    """
    idx_df = pd.DataFrame({
        "year":  index_series["year"].values.astype(int),
        "index": index_series.values,
    }).dropna()

    rows = []
    for dname, da in driver_anoms.items():
        drv_series = area_mean(da)
        drv_df = pd.DataFrame({
            "year":  drv_series["year"].values.astype(int),
            "driver": drv_series.values,
        }).dropna()

        merged = idx_df.merge(drv_df, on="year")
        n = len(merged)

        if n < 10:
            rows.append({"driver": dname, "n": n,
                         "pearson_r": np.nan, "pearson_p": np.nan,
                         "spearman_r": np.nan, "spearman_p": np.nan})
            continue

        pr, pp = pearsonr(merged["index"], merged["driver"])
        sr, sp = spearmanr(merged["index"], merged["driver"])
        rows.append({
            "driver": dname, "n": n,
            "pearson_r":  round(float(pr), 3), "pearson_p":  round(float(pp), 4),
            "spearman_r": round(float(sr), 3), "spearman_p": round(float(sp), 4),
        })
    return pd.DataFrame(rows)


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("Loading Germany boundary...")
    gdf, geom = load_country_shape(GERMANY_SHP)

    # ── pre-load and pre-compute all driver annual anomalies once ─────────────
    print("Loading and computing driver anomalies (1961-1990 reference)...")
    driver_annual_anoms = {}

    for dname, fpath in DRIVER_FILES.items():
        vname = DRIVER_VARS[dname]
        print(f"  {dname}...")
        try:
            da    = keep_jja(load_field(fpath, vname))
            annual = annual_jja_mean(da)
            clim   = reference_mean(annual, REF_START, REF_END)
            anom   = compute_anomalies(annual, clim)
            driver_annual_anoms[dname] = anom
        except FileNotFoundError as e:
            print(f"  WARNING: {e}. Skipping {dname}.")

    available_drivers = list(driver_annual_anoms.keys())
    all_years = np.arange(int(START_YEAR), int(END_YEAR) + 1)

    # ── process each extreme index ────────────────────────────────────────────
    for display_name, nc_stem in INDICES:
        print(f"\nProcessing drivers for: {display_name}")

        # Load ICON-CLM annual index (use ICON version for driver composites)
        try:
            index_annual = load_index(nc_stem, "ICON")
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
            continue

        # Identify upper-quartile (high-index) years
        high_years, top_df = upper_quartile_years(index_annual)
        top_df.to_csv(
            os.path.join(TABDIR, f"{display_name}_top_quartile_years.csv"),
            index=False,
        )
        print(f"  Upper-quartile years ({len(high_years)}): {high_years}")

        # Composite anomaly for each driver
        composites = {}
        for dname, anom in driver_annual_anoms.items():
            common_years = np.intersect1d(high_years, anom["year"].values)
            if len(common_years) < 3:
                print(f"  {dname}: not enough overlapping years, skipping.")
                continue
            comp = composite_high_vs_rest(anom, high_years, all_years)
            composites[dname] = comp

            # Individual composite map
            plot_single_composite(
                comp, gdf, geom,
                outfile   = os.path.join(FIGDIR, f"{display_name}_{dname}_composite.png"),
                levels    = DRIVER_LEVELS[dname],
                cbar_label = f"{dname} anomaly ({DRIVER_UNITS[dname]})",
                title     = f"{display_name.replace('_',' ')} | {dname} composite",
            )

        # Six-panel composite figure (all available drivers)
        if len(composites) >= 2:
            plot_six_panel_composites(
                composites, gdf, geom,
                outfile  = os.path.join(FIGDIR, f"{display_name}_all_drivers_composite.png"),
                suptitle = f"Driver anomalies | upper-quartile {display_name.replace('_',' ')} summers",
            )

        # Germany-average correlations
        index_series = area_mean(index_annual)
        corr_df = compute_correlations(index_series, driver_annual_anoms)
        corr_df.insert(0, "index", display_name)
        corr_df.to_csv(
            os.path.join(TABDIR, f"{display_name}_driver_correlations.csv"),
            index=False,
        )
        print(f"  Correlations saved.")

    print(f"\nDone.  Figures → {FIGDIR}   Tables → {TABDIR}")
