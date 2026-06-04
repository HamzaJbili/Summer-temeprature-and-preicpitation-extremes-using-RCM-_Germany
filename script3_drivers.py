"""
script3_drivers.py
------------------
Process-driver composite and correlation analysis.

For each extreme index, identifies summers in the upper quartile
of the Germany-average annual series (high-index years) and computes:
  (a) composite anomaly maps for each driver variable
  (b) a seven-panel composite figure (2×4 grid, one spare slot)
  (c) Germany-average Pearson and Spearman correlations vs each driver

Composite anomaly definition
  composite = mean(driver anomaly | high-index years)
            - mean(driver anomaly | all other years)
  Driver variable is first expressed as anomaly relative to the
  1961–1990 reference climatology.

Driver variables used (all ICON-CLM ERA5-driven, EUR-12 CORDEX):
  PSL   : sea-level pressure            (Pa → hPa via ×0.01)
  SHF   : surface sensible heat flux    (W m⁻²)
  LHF   : surface latent heat flux      (W m⁻²)
  CLT   : total cloud cover             (%)
  WIND  : 10-metre wind speed           (m s⁻¹)
  CAPE  : convective available PE       (J kg⁻¹)
  CIN   : convective inhibition         (J kg⁻¹)

Physical rationale for driver choices
  - Z500 (mid-troposphere geopotential) is replaced by PSL (sea-level pressure),
    which equivalently captures blocking anticyclones and the Azores-High
    extension that govern German summer heat events.
  - Soil moisture (mrso) is not available.  LHF serves as an indirect proxy:
    when surface soil dries, latent heat flux decreases and sensible heat flux
    increases (Bowen-ratio shift), capturing the land–atmosphere coupling that
    amplifies heat extremes.
  - CLT (cloud cover) is added because incoming solar radiation — modulated
    by cloud amount — is the primary surface energy-balance driver of
    temperature extremes on daily-to-seasonal timescales.

Requires annual index NetCDF files produced by script2_extremes.py.
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

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from utils import (
    load_field,
    reference_mean, compute_anomalies, area_mean,
    load_country_shape, interp_display, build_mask, apply_mask,
    style_axis,
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
INDEX_NC_DIR = "/archive1/hamza_data/DE_files/DE_1950-2022/extremes_indices/output_extremes/netcdf"   # produced by script2

# ── extreme indices to analyse ────────────────────────────────────────────────
# Tuple: (internal_name, nc_stem)
INDICES = [
    ("T90p_exceedance_days", "T90p_days"),
    ("Heatwave_number",      "HWN"),
    ("Heatwave_duration",    "HWD"),
    ("SDII",                 "SDII"),
    ("CDD",                  "CDD"),
    ("SPI",                  "SPI"),
]

# ── driver file configuration ─────────────────────────────────────────────────
# JJA seasonal mean files produced by CDO (one value per year, 1950-2022).
# Naming convention: {var}_DE-0.25_JJA_1950-2022.nc
_SUFFIX = "DE-0.25_JJA_1950-2022.nc"

DRIVER_FILES = {
    "PSL":  f"psl_{_SUFFIX}",
    "SHF":  f"hfss_{_SUFFIX}",
    "LHF":  f"hfls_{_SUFFIX}",
    "CLT":  f"clt_{_SUFFIX}",
    "WIND": f"sfcWind_{_SUFFIX}",
    "CAPE": f"cape_{_SUFFIX}",
    "CIN":  f"cin_{_SUFFIX}",
}

# NetCDF variable name inside each file
DRIVER_VARS = {
    "PSL":  "psl",
    "SHF":  "hfss",
    "LHF":  "hfls",
    "CLT":  "clt",
    "WIND": "sfcWind",
    "CAPE": "cape",
    "CIN":  "cin",
}

# Physical unit scaling applied AFTER loading (before climatology computation).
# PSL is stored in Pa by CORDEX convention; multiply by 0.01 → hPa.
# All other drivers are already in their display units.
DRIVER_SCALE = {
    "PSL":  0.01,
}

# Units for colorbar labels
DRIVER_UNITS = {
    "PSL":  "hPa",
    "SHF":  "W m$^{-2}$",
    "LHF":  "W m$^{-2}$",
    "CLT":  "%",
    "WIND": "m s$^{-1}$",
    "CAPE": "J kg$^{-1}$",
    "CIN":  "J kg$^{-1}$",
}

# Composite anomaly colormap boundaries (adjust after first visual inspection).
# Symmetric around zero; blue = anomaly below composite mean, red = above.
DRIVER_LEVELS = {
    "PSL":  [-6,   -4,   -2,  -1,  -0.5, 0,  0.5,  1,   2,   4,   6],   # hPa
    "SHF":  [-30,  -20,  -10,  -5,  -2,  0,   2,    5,  10,  20,  30],   # W m⁻²
    "LHF":  [-30,  -20,  -10,  -5,  -2,  0,   2,    5,  10,  20,  30],   # W m⁻²
    "CLT":  [-15,  -10,   -7,  -5,  -2,  0,   2,    5,   7,  10,  15],   # %
    "WIND": [ -2,   -1,  -0.5,-0.25,-0.1, 0,  0.1, 0.25, 0.5, 1,   2],  # m s⁻¹
    "CAPE": [-300, -200, -100, -50, -20,  0,  20,   50, 100, 200, 300],  # J kg⁻¹
    "CIN":  [ -40,  -30,  -20, -10,  -5,  0,   5,   10,  20,  30,  40], # J kg⁻¹
}

# Physical descriptions for axis titles and CSV output
DRIVER_LONG = {
    "PSL":  "Sea-level pressure",
    "SHF":  "Sensible heat flux",
    "LHF":  "Latent heat flux",
    "CLT":  "Total cloud cover",
    "WIND": "Surface wind speed",
    "CAPE": "Convective available PE",
    "CIN":  "Convective inhibition",
}

# Blue-white-red diverging palette
DIV_COLORS = [
    "#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#f7f7f7",
    "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f",
]

MAP_EXTENT = [5.8, 15.2, 47.4, 55.1]
PROJ       = ccrs.LambertConformal(central_longitude=10, central_latitude=51)
PC         = ccrs.PlateCarree()


# ── helper: load annual index from script2 output ─────────────────────────────
def load_index(nc_stem, dataset_label):
    """Load annual index array produced by script2_extremes.py."""
    path = os.path.join(INDEX_NC_DIR, f"{nc_stem}_{dataset_label}_annual.nc")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Annual index file not found: {path}\n"
            "Run script2_extremes.py first."
        )
    ds = xr.open_dataset(path)
    da = ds[list(ds.data_vars)[0]]
    return da.sortby("lat").sortby("lon")


# ── composite definition ──────────────────────────────────────────────────────
def upper_quartile_years(annual_index):
    """Return years in the upper quartile of the Germany-average annual index."""
    series = area_mean(annual_index)
    df = pd.DataFrame({
        "year":  series["year"].values.astype(int),
        "value": series.values,
    }).dropna()
    n_top = max(1, len(df) // 4)
    top   = df.nlargest(n_top, "value")
    return top["year"].values.astype(int), top


def composite_high_vs_rest(driver_anom, high_years, all_years):
    """
    Composite anomaly = mean over high-index summers − mean over all other summers.
    Both operands are the driver expressed as anomaly from its 1961–1990 climatology.
    """
    low_years  = np.setdiff1d(all_years, high_years)
    mean_high  = driver_anom.sel(year=high_years).mean("year", skipna=True)
    mean_low   = driver_anom.sel(year=low_years ).mean("year", skipna=True)
    return mean_high - mean_low


# ── plotting helpers (cartopy maps for consistency with script2) ───────────────
def _draw_composite_panel(ax, da, gdf, geom, levels, fmt, title, tag=None):
    """Draw one composite anomaly panel into an existing cartopy axes."""
    cmap = mcolors.ListedColormap(
        [c for c in DIV_COLORS[:len(levels) - 1]] if len(DIV_COLORS) == len(levels) - 1
        else _interp_colors_local(DIV_COLORS, len(levels) - 1)
    )
    norm = mcolors.BoundaryNorm(levels, cmap.N)
    cmap.set_under(DIV_COLORS[0])
    cmap.set_over(DIV_COLORS[-1])

    ax.set_extent(MAP_EXTENT, crs=PC)
    ax.set_facecolor("#d6e8f2")
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#ebebeb", zorder=1)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"),
                   linewidth=0.3, edgecolor="0.45", zorder=2)

    fine = interp_display(da)
    mask = build_mask(fine["lon"].values, fine["lat"].values, geom)
    arr  = apply_mask(fine.values, mask)

    ax.contourf(fine["lon"].values, fine["lat"].values, arr,
                levels=levels, cmap=cmap, norm=norm,
                transform=PC, extend="both", antialiased=True, zorder=3)
    ax.add_geometries(gdf.geometry, PC, facecolor="none",
                      edgecolor="black", linewidth=0.55, zorder=6)

    if tag:
        ax.text(0.03, 0.97, tag, transform=ax.transAxes,
                ha="left", va="top", fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    ax.set_title(title, fontsize=8.5, fontweight="bold", pad=3)
    style_axis(ax)

    # Germany-mean annotation
    de_mask_c = build_mask(da["lon"].values, da["lat"].values, geom)
    if de_mask_c.any():
        mean_v = float(np.nanmean(da.values[de_mask_c]))
        sign   = "+" if mean_v >= 0 else ""
        ax.text(0.03, 0.03, f"DE: {sign}{fmt % mean_v}",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5,
                color="#222222",
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="#aaaaaa", alpha=0.88, lw=0.4))

    return cmap, norm


def _interp_colors_local(palette, n):
    """Linearly interpolate palette to n colours."""
    from matplotlib.colors import to_rgba
    import numpy as np
    rgba = np.array([to_rgba(c) for c in palette])
    xs   = np.linspace(0, 1, len(palette))
    xn   = np.linspace(0, 1, n)
    out  = np.column_stack([np.interp(xn, xs, rgba[:, i]) for i in range(4)])
    return [tuple(row) for row in out]


def plot_single_composite(da, gdf, geom, outfile, levels, cbar_label, title):
    """Single-index, single-driver composite map (publication cartopy style)."""
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(5.5, 5.2))
    fig.patch.set_facecolor("white")
    ax  = fig.add_subplot(1, 1, 1, projection=PROJ)

    fmt  = "%.1f"
    cmap, norm = _draw_composite_panel(ax, da, gdf, geom, levels, fmt, title)

    cax = ax.inset_axes([1.015, 0.0, 0.04, 1.0])
    cb  = ColorbarBase(cax, cmap=cmap, norm=norm, boundaries=levels,
                       ticks=levels, orientation="vertical", extend="neither")
    cb.ax.tick_params(labelsize=6, pad=2)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))
    cb.outline.set_linewidth(0.5)
    cb.set_label(cbar_label, fontsize=7, labelpad=4)

    fig.savefig(outfile, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_driver_panel_figure(composites, gdf, geom, outfile, suptitle):
    """
    2×4 grid composite figure — one panel per driver variable.
    Cartopy LambertConformal projection, slim vertical colorbars,
    rectangular ends, Germany-mean annotation on each panel.
    """
    from matplotlib.gridspec import GridSpec

    driver_names = list(composites.keys())
    n = len(driver_names)
    nrows, ncols = 2, 4
    n_total = nrows * ncols   # 8 slots; last slot hidden if n < 8

    fig = plt.figure(figsize=(14.0, 7.8))
    fig.patch.set_facecolor("white")
    if suptitle:
        fig.suptitle(suptitle, fontsize=10, fontweight="normal", y=1.00)

    gs = GridSpec(nrows, ncols * 2,
                  width_ratios=([1, 0.06] * ncols),
                  left=0.02, right=0.98, top=0.94, bottom=0.04,
                  hspace=0.14, wspace=0.0)

    tags = list("abcdefgh")
    for k, dname in enumerate(driver_names):
        row  = k // ncols
        col  = (k % ncols) * 2

        ax   = fig.add_subplot(gs[row, col], projection=PROJ)
        da   = composites[dname]
        lvls = DRIVER_LEVELS[dname]
        unit = DRIVER_UNITS[dname]
        fmt  = "%.1f"

        cmap, norm = _draw_composite_panel(
            ax, da, gdf, geom, lvls, fmt,
            title=DRIVER_LONG[dname],
            tag=f"({tags[k]})",
        )

        # Slim vertical colorbar
        cax = ax.inset_axes([1.015, 0.0, 0.06, 1.0])
        cb  = ColorbarBase(cax, cmap=cmap, norm=norm, boundaries=lvls,
                           ticks=[lvls[0], 0, lvls[-1]],
                           orientation="vertical", extend="neither")
        cb.ax.tick_params(labelsize=5, pad=1)
        cb.ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))
        cb.outline.set_linewidth(0.4)
        cb.set_label(unit, fontsize=5.5, labelpad=2)

    # Hide unused slots
    for k in range(n, n_total):
        row = k // ncols
        col = (k % ncols) * 2
        try:
            fig.add_subplot(gs[row, col]).set_visible(False)
        except Exception:
            pass

    fig.savefig(outfile, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ── correlation bar chart ────────────────────────────────────────────────────
def plot_correlation_bars(corr_df, index_name, outfile):
    """
    Horizontal bar chart of Pearson r (Germany-avg index vs Germany-avg driver).
    Bars are coloured by sign; hatched bars indicate p < 0.05.
    """
    df = corr_df.dropna(subset=["pearson_r"]).copy()
    df = df.sort_values("pearson_r", ascending=True)

    fig, ax = plt.subplots(figsize=(5.5, 0.5 * len(df) + 1.2))
    fig.patch.set_facecolor("white")

    colors = ["#d73027" if r >= 0 else "#4575b4" for r in df["pearson_r"]]
    bars   = ax.barh(df["driver"], df["pearson_r"],
                     color=colors, height=0.55, edgecolor="k", linewidth=0.4)

    # Hatching for non-significant bars
    for bar, pval in zip(bars, df["pearson_p"]):
        if pval >= 0.05:
            bar.set_hatch("///")
            bar.set_edgecolor("0.50")

    # Annotate r values
    for bar, r, p in zip(bars, df["pearson_r"], df["pearson_p"]):
        sig  = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        xpos = bar.get_width() + (0.01 if r >= 0 else -0.01)
        ha   = "left" if r >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"{r:+.2f}{sig}", va="center", ha=ha, fontsize=8)

    ax.axvline(0, color="0.3", lw=0.7)
    ax.set_xlim(-1.1, 1.1)
    ax.set_xlabel("Pearson r  (Germany average)", fontsize=9)
    ax.set_title(f"{index_name.replace('_', ' ')} — driver correlations\n"
                 "(hatched = p ≥ 0.05; * p < 0.05; ** p < 0.01)",
                 fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
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
            "year":   drv_series["year"].values.astype(int),
            "driver": drv_series.values,
        }).dropna()

        merged = idx_df.merge(drv_df, on="year")
        n_obs  = len(merged)

        if n_obs < 10:
            rows.append({"driver": dname, "n": n_obs,
                         "pearson_r": np.nan, "pearson_p": np.nan,
                         "spearman_r": np.nan, "spearman_p": np.nan})
            continue

        pr, pp = pearsonr( merged["index"], merged["driver"])
        sr, sp = spearmanr(merged["index"], merged["driver"])
        rows.append({
            "driver":     dname,
            "driver_long": DRIVER_LONG[dname],
            "unit":        DRIVER_UNITS[dname],
            "n":           n_obs,
            "pearson_r":   round(float(pr), 3), "pearson_p":   round(float(pp), 4),
            "spearman_r":  round(float(sr), 3), "spearman_p":  round(float(sp), 4),
        })
    return pd.DataFrame(rows)


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("Loading Germany boundary ...")
    gdf, geom = load_country_shape(GERMANY_SHP)

    # ── pre-load and pre-compute driver annual anomalies once ─────────────────
    print("Loading driver variables and computing 1961–1990 anomalies ...")
    driver_annual_anoms = {}

    for dname, fpath in DRIVER_FILES.items():
        vname = DRIVER_VARS[dname]
        scale = DRIVER_SCALE.get(dname, 1.0)
        print(f"  {dname} ({DRIVER_LONG[dname]}) ...")
        try:
            da = load_field(fpath, vname)
            if scale != 1.0:
                da = da * scale
            # CDO JJA mean: time dim has one step per year → extract year
            annual = da.assign_coords(year=("time", da["time"].dt.year.values))
            annual = annual.swap_dims({"time": "year"}).drop_vars("time")
            annual = annual.sel(year=slice(int(START_YEAR), int(END_YEAR)))
            clim   = reference_mean(annual, REF_START, REF_END)
            anom   = compute_anomalies(annual, clim)
            driver_annual_anoms[dname] = anom
            print(f"    OK — {len(annual.year)} years, "
                  f"mean={float(anom.mean()):.3f} {DRIVER_UNITS[dname]}")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}  →  skipping {dname}.")
        except Exception as e:
            print(f"  ERROR loading {dname}: {e}  →  skipping.")

    available_drivers = list(driver_annual_anoms.keys())
    all_years = np.arange(int(START_YEAR), int(END_YEAR) + 1)

    print(f"\nAvailable drivers: {available_drivers}")

    # ── process each extreme index ────────────────────────────────────────────
    all_corr_rows = []

    for display_name, nc_stem in INDICES:
        print(f"\n{'='*60}")
        print(f"Processing drivers for: {display_name}")

        # Load ICON-CLM annual index (driver composites use model fields)
        try:
            index_annual = load_index(nc_stem, "ICON")
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
            continue

        # Upper-quartile (high-index) years
        high_years, top_df = upper_quartile_years(index_annual)
        top_df.to_csv(
            os.path.join(TABDIR, f"{display_name}_top_quartile_years.csv"),
            index=False,
        )
        print(f"  Upper-quartile years ({len(high_years)}): {high_years}")

        # Composite anomaly for each driver
        composites = {}
        for dname, anom in driver_annual_anoms.items():
            anom_years   = anom["year"].values.astype(int)
            common_high  = np.intersect1d(high_years, anom_years).astype(int)
            if len(common_high) < 3:
                print(f"  {dname}: fewer than 3 overlapping high-index years — skipping.")
                continue
            comp = composite_high_vs_rest(anom, common_high, anom_years)
            composites[dname] = comp

            # Individual map
            plot_single_composite(
                comp, gdf, geom,
                outfile    = os.path.join(FIGDIR,
                             f"{display_name}_{dname}_composite.png"),
                levels     = DRIVER_LEVELS[dname],
                cbar_label = f"Anomaly [{DRIVER_UNITS[dname]}]",
                title      = (f"{display_name.replace('_', ' ')}  ·  "
                              f"{DRIVER_LONG[dname]}"),
            )

        # Multi-panel composite figure (all available drivers)
        if len(composites) >= 2:
            plot_driver_panel_figure(
                composites, gdf, geom,
                outfile  = os.path.join(FIGDIR,
                           f"{display_name}_all_drivers_composite.png"),
                suptitle = (f"Driver anomalies — upper-quartile "
                            f"{display_name.replace('_', ' ')} summers"),
            )

        # Germany-average correlations
        index_series = area_mean(index_annual)
        corr_df = compute_correlations(index_series, driver_annual_anoms)
        corr_df.insert(0, "index", display_name)
        corr_df.to_csv(
            os.path.join(TABDIR, f"{display_name}_driver_correlations.csv"),
            index=False,
        )
        all_corr_rows.append(corr_df)

        # Correlation bar chart
        plot_correlation_bars(
            corr_df, display_name,
            outfile = os.path.join(FIGDIR,
                      f"{display_name}_driver_correlations_bar.png"),
        )
        print(f"  Figures and table saved.")

    # ── cross-index correlation summary CSV ───────────────────────────────────
    if all_corr_rows:
        pd.concat(all_corr_rows, ignore_index=True).to_csv(
            os.path.join(TABDIR, "all_indices_driver_correlations.csv"),
            index=False,
        )
        print(f"\nCombined correlation table → {TABDIR}/all_indices_driver_correlations.csv")

    print(f"\nDone.  Figures → {FIGDIR}   Tables → {TABDIR}")
