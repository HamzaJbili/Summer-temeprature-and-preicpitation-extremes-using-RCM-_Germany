"""
script3_drivers.py
------------------
Driver analysis for German summer extremes — correlation approach.

The analysis has four parts:

  1.  DRIVER OVERVIEW
      A multi-panel time-series figure of every driver variable
      (Germany-average JJA mean, 1950–2022) with its linear trend, so
      the interannual behaviour that produces the correlations is visible.

  2.  GROUPED INDEX–DRIVER CORRELATION HEATMAPS
      Instead of a blanket 6-index × 7-driver matrix the indices are split
      into two physically motivated groups, each correlated only against
      its mechanistically relevant drivers:

        TEMPERATURE   T90p, HWN, HWD  ×  PSL, SHF, LHF, CLT      (3×4)
        PRECIPITATION SDII, CDD, SPI  ×  PSL, LHF, CLT, CAPE, CIN (3×5)

      Each group yields a Pearson and a Spearman heatmap plus per-index
      correlation bar charts.  This quantifies HOW STRONG each link is.

  3.  DRIVER–DRIVER CORRELATION MATRIX
      A symmetric Pearson matrix of all drivers against each other,
      documenting the multicollinearity among the predictors
      (PSL/SHF/LHF/CLT are physically linked: high pressure → clear sky →
      dry soil → high sensible / low latent heat flux).  This lets the
      index–driver correlations be interpreted with the right caution.

  4.  COMPOSITE ANOMALY MAPS
      For ONE representative index per group (T90p for temperature, SDII
      for precipitation) the upper-tercile (hottest / wettest-intensity)
      summers are selected and the anomaly field of each relevant driver
      is composited over those years, as a single multi-panel map figure
      per group.  This shows WHERE in Germany the driver signal sits —
      the spatial complement to the (non-spatial) correlation heatmaps.
      Restricting composites to one index per group keeps the output to
      two map figures instead of one map per index×driver pair.  The
      composite-during-extreme-years method follows Hirschi et al. (2011)
      and Whan et al. (2015).

Why grouped correlation (not a blanket matrix, not composite maps)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Correlating an extreme-index time series with candidate driver
  variables is the standard, compact way to quantify the strength and
  sign of each index–driver link (Hirschi et al. 2011; Mueller &
  Seneviratne 2012).  Two grouped heatmaps summarise the physically
  meaningful relationships, avoiding both the 40+ separate composite maps
  a full composite treatment would need and the physically unmotivated
  pairings of a blanket matrix (e.g. CAPE vs heat-wave duration).

Interpretation note — model-internal physical-consistency analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Both the extreme indices and the drivers are taken from the SAME
  ICON-CLM ERA5-driven hindcast.  This analysis therefore measures
  whether the model produces its extremes through physically consistent
  internal relationships (e.g. hot summers under high pressure and
  suppressed latent cooling) — a "process-based / physical-consistency"
  evaluation, not an attribution of the observed climate.

  This framing is defensible for an ERA5-driven hindcast because the
  large-scale circulation is constrained by the driving reanalysis at
  the lateral boundaries, so the dynamical drivers (PSL) remain reliable
  even where the model has local biases in precipitation extremes
  (Kotlarski et al. 2014; Vautard et al. 2021).  The same model-internal
  driver approach is used by Loikith et al. (2015) for reanalysis-driven
  RCM temperature extremes.  For precipitation indices (CDD, SDII, SPI)
  the links reflect the model's internal physics rather than necessarily
  the observed system, and are interpreted as such.

Driver variables (all ICON-CLM, JJA seasonal mean, DE-0.25 domain)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  PSL   : sea-level pressure            (Pa → hPa via ×0.01)  blocking / anticyclones
  SHF   : surface sensible heat flux    (W m⁻²)               surface heating
  LHF   : surface latent heat flux      (W m⁻²)               soil-moisture proxy (Bowen)
  CLT   : total cloud cover             (%)                   incoming solar radiation
  WIND  : 10-metre wind speed           (m s⁻¹)               advection / ventilation
  CAPE  : convective available PE       (J kg⁻¹)              convective potential
  CIN   : convective inhibition         (J kg⁻¹)              convective suppression

Scientific references
~~~~~~~~~~~~~~~~~~~~~~~
  Hirschi et al. (2011)  Nat. Geosci. 4, 17-21        — soil-moisture / hot-extreme correlation
  Mueller & Seneviratne (2012)  PNAS 109, 12398        — precip-deficit / hot-day correlation
  Loikith et al. (2015)  Clim. Dyn. 45, 3257           — reanalysis-driven RCM extreme drivers
  Kotlarski et al. (2014)  Geosci. Model Dev. 7, 1297  — EURO-CORDEX evaluation framework
  Vautard et al. (2021)  JGR-Atmos 126, e2019JD032344  — ERA5-driven CORDEX, constrained circulation
  Seneviratne et al. (2010)  Earth-Sci. Rev. 99, 125   — soil-moisture–climate coupling review
  Whan et al. (2015)  Weather Clim. Extremes 9, 57      — soil-moisture composites for hot extremes

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

from scipy.stats import pearsonr, spearmanr, linregress

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from utils import (
    load_field, area_mean,
    reference_mean, compute_anomalies,
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

# ── extreme indices, split into two physically motivated groups ───────────────
# Tuple: (display_name, nc_stem, row_label)
TEMP_INDICES = [
    ("T90p_exceedance_days", "T90p_days", "T90p"),
    ("Heatwave_number",      "HWN",       "HWN"),
    ("Heatwave_duration",    "HWD",       "HWD"),
]
PRECIP_INDICES = [
    ("SDII",                 "SDII",      "SDII"),
    ("CDD",                  "CDD",       "CDD"),
    ("SPI",                  "SPI",       "SPI"),
]

# Drivers assigned to each group on physical grounds.
#   Temperature extremes:  anticyclonic blocking (PSL), surface energy
#                          partition (SHF, LHF), radiation control (CLT).
#   Precipitation extremes: blocking (PSL), moisture supply (LHF), cloud
#                          cover (CLT), convective fuel / trigger (CAPE, CIN).
TEMP_DRIVERS   = ["PSL", "SHF", "LHF", "CLT"]
PRECIP_DRIVERS = ["PSL", "LHF", "CLT", "CAPE", "CIN"]

# ── driver file configuration ─────────────────────────────────────────────────
# JJA seasonal mean files produced by CDO (one value per year, 1950-2022).
# Naming convention: {var}_DE-0.25_JJA_1950-2022.nc
_SUFFIX = "DE-0.25_JJA_1950-2022.nc"

# Full driver list (used for the overview time series and the driver matrix).
DRIVER_ORDER = ["PSL", "SHF", "LHF", "CLT", "WIND", "CAPE", "CIN"]

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

# Physical unit scaling applied AFTER loading.
# PSL is stored in Pa by CORDEX convention; multiply by 0.01 → hPa.
DRIVER_SCALE = {
    "PSL":  0.01,
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

# Units for the driver time-series y-axes
DRIVER_UNITS = {
    "PSL":  "hPa",
    "SHF":  "W m$^{-2}$",
    "LHF":  "W m$^{-2}$",
    "CLT":  "%",
    "WIND": "m s$^{-1}$",
    "CAPE": "J kg$^{-1}$",
    "CIN":  "J kg$^{-1}$",
}

# Diverging blue-white-red palette for the correlation heatmap
CORR_COLORS = [
    "#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#f7f7f7",
    "#fddbc7", "#f4a582", "#d6604d", "#b2182b",
]

# ── composite-map configuration ───────────────────────────────────────────────
# Symmetric anomaly contour levels per driver (blue = below, red = above the
# all-summers mean).  Adjust after first visual inspection if needed.
DRIVER_LEVELS = {
    "PSL":  [-6,   -4,   -2,  -1,  -0.5, 0,  0.5,  1,   2,   4,   6],   # hPa
    "SHF":  [-30,  -20,  -10,  -5,  -2,  0,   2,    5,  10,  20,  30],   # W m⁻²
    "LHF":  [-30,  -20,  -10,  -5,  -2,  0,   2,    5,  10,  20,  30],   # W m⁻²
    "CLT":  [-15,  -10,   -7,  -5,  -2,  0,   2,    5,   7,  10,  15],   # %
    "WIND": [ -2,   -1,  -0.5,-0.25,-0.1, 0,  0.1, 0.25, 0.5, 1,   2],  # m s⁻¹
    "CAPE": [-300, -200, -100, -50, -20,  0,  20,   50, 100, 200, 300],  # J kg⁻¹
    "CIN":  [ -40,  -30,  -20, -10,  -5,  0,   5,   10,  20,  30,  40],  # J kg⁻¹
}

# Blue-white-red diverging palette for the composite maps
DIV_COLORS = [
    "#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#f7f7f7",
    "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f",
]

MAP_EXTENT = [5.8, 15.2, 47.4, 55.1]
PROJ       = ccrs.LambertConformal(central_longitude=10, central_latitude=51)
PC         = ccrs.PlateCarree()

# Representative index per group for the composite figure (one strong, clean
# index per group keeps the spatial story to two multi-panel figures instead
# of one map per index×driver pair).
COMPOSITE_REPRESENTATIVE = {
    "Temperature":   ("T90p_exceedance_days", "T90p_days", "T90p"),
    "Precipitation": ("SDII",                 "SDII",      "SDII"),
}


# ── helpers ────────────────────────────────────────────────────────────────────
def load_index_series(nc_stem, dataset_label="ICON"):
    """Germany-average annual series of an index produced by script2."""
    path = os.path.join(INDEX_NC_DIR, f"{nc_stem}_{dataset_label}_annual.nc")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Annual index file not found: {path}\nRun script2_extremes.py first."
        )
    ds = xr.open_dataset(path)
    da = ds[list(ds.data_vars)[0]].sortby("lat").sortby("lon")
    return area_mean(da)


def load_driver_series(dname):
    """Germany-average annual JJA-mean series of one driver variable."""
    fpath = DRIVER_FILES[dname]
    vname = DRIVER_VARS[dname]
    scale = DRIVER_SCALE.get(dname, 1.0)

    da = load_field(fpath, vname)
    if scale != 1.0:
        da = da * scale
    # CDO JJA mean: time dim has one step per year → convert to a 'year' dim
    da = da.assign_coords(year=("time", da["time"].dt.year.values))
    da = da.swap_dims({"time": "year"}).drop_vars("time")
    da = da.sel(year=slice(int(START_YEAR), int(END_YEAR)))
    return area_mean(da)


def correlate(index_series, driver_series):
    """Pearson and Spearman correlation on the common years of two series."""
    idx = pd.DataFrame({"year": index_series["year"].values.astype(int),
                        "index": index_series.values}).dropna()
    drv = pd.DataFrame({"year": driver_series["year"].values.astype(int),
                        "driver": driver_series.values}).dropna()
    merged = idx.merge(drv, on="year")
    n = len(merged)
    if n < 10:
        return dict(n=n, pearson_r=np.nan, pearson_p=np.nan,
                    spearman_r=np.nan, spearman_p=np.nan)
    pr, pp = pearsonr(merged["index"], merged["driver"])
    sr, sp = spearmanr(merged["index"], merged["driver"])
    return dict(n=n,
                pearson_r=float(pr), pearson_p=float(pp),
                spearman_r=float(sr), spearman_p=float(sp))


def _stars(p):
    """Significance stars from a p-value."""
    if not np.isfinite(p):
        return ""
    return "**" if p < 0.01 else ("*" if p < 0.05 else "")


# ── driver overview: multi-panel time series ───────────────────────────────────
def plot_driver_overview(series_dict, drivers, outfile):
    """
    Multi-panel time-series of every driver (Germany-average JJA mean,
    1950–2022) with a linear trend line and slope/significance annotation.
    """
    n = len(drivers)
    ncol = 2
    nrow = int(np.ceil(n / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 2.1 * nrow),
                             sharex=True)
    fig.patch.set_facecolor("white")
    axes = np.atleast_1d(axes).ravel()

    for ax, dname in zip(axes, drivers):
        s = series_dict[dname]
        years = s["year"].values.astype(int)
        vals  = s.values

        good = np.isfinite(vals)
        ax.plot(years[good], vals[good], color="#2166ac", lw=1.2,
                marker="o", ms=2.5, mfc="white", mec="#2166ac", mew=0.5)

        # Linear trend
        if good.sum() >= 10:
            sl, ic, r, p, se = linregress(years[good], vals[good])
            ax.plot(years[good], ic + sl * years[good],
                    color="#b2182b", lw=1.4, ls="--")
            decade = sl * 10.0
            ax.set_title(
                f"{dname} — {DRIVER_LONG[dname]}   "
                f"(trend {decade:+.2g} {DRIVER_UNITS[dname]}/dec{_stars(p)})",
                fontsize=8.5)
        else:
            ax.set_title(f"{dname} — {DRIVER_LONG[dname]}", fontsize=8.5)

        ax.set_ylabel(DRIVER_UNITS[dname], fontsize=8)
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", lw=0.3, alpha=0.5)

    # Hide any unused panels
    for ax in axes[n:]:
        ax.set_visible(False)

    # X label only on the bottom row of visible panels
    for ax in axes[:n][-ncol:]:
        ax.set_xlabel("Year", fontsize=8)

    fig.suptitle("ICON-CLM JJA-mean drivers, Germany average, 1950–2022\n"
                 "(red dashed = linear trend; * p<0.05, ** p<0.01)",
                 fontsize=10, y=1.0)
    plt.tight_layout()
    fig.savefig(outfile, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ── driver–driver correlation matrix ───────────────────────────────────────────
def plot_driver_matrix(series_dict, drivers, outfile):
    """
    Symmetric Pearson correlation matrix among the driver variables,
    documenting the multicollinearity of the predictors.  Returns the
    matrix as a DataFrame for CSV export.
    """
    n = len(drivers)
    mat = np.full((n, n), np.nan)
    for i, di in enumerate(drivers):
        for j, dj in enumerate(drivers):
            si, sj = series_dict[di], series_dict[dj]
            a = pd.DataFrame({"year": si["year"].values.astype(int),
                              "a": si.values}).dropna()
            b = pd.DataFrame({"year": sj["year"].values.astype(int),
                              "b": sj.values}).dropna()
            m = a.merge(b, on="year")
            if len(m) >= 10:
                mat[i, j] = pearsonr(m["a"], m["b"])[0]

    cmap = mcolors.LinearSegmentedColormap.from_list("corr", CORR_COLORS)
    norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)

    fig, ax = plt.subplots(figsize=(0.85 * n + 2.0, 0.85 * n + 1.6))
    fig.patch.set_facecolor("white")
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")

    for i in range(n):
        for j in range(n):
            r = mat[i, j]
            if not np.isfinite(r):
                continue
            txt_color = "white" if abs(r) > 0.55 else "#1a1a1a"
            ax.text(j, i, f"{r:+.2f}", ha="center", va="center",
                    fontsize=8, color=txt_color)

    ax.set_xticks(np.arange(n)); ax.set_yticks(np.arange(n))
    ax.set_xticklabels(drivers, fontsize=9)
    ax.set_yticklabels(drivers, fontsize=9, fontweight="bold")
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.set_title("Driver–driver correlation (Pearson)\n"
                 "ICON-CLM Germany average, JJA 1950–2022", fontsize=10, pad=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Pearson r", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    plt.tight_layout()
    fig.savefig(outfile, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame(mat, index=drivers, columns=drivers)


# ── composite anomaly maps ─────────────────────────────────────────────────────
def load_index_field(nc_stem, dataset_label="ICON"):
    """Full 2-D annual index field (lat, lon, year) produced by script2."""
    path = os.path.join(INDEX_NC_DIR, f"{nc_stem}_{dataset_label}_annual.nc")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Annual index file not found: {path}\nRun script2_extremes.py first.")
    ds = xr.open_dataset(path)
    return ds[list(ds.data_vars)[0]].sortby("lat").sortby("lon")


def load_driver_anom(dname):
    """Full 2-D driver anomaly field (lat, lon, year) vs 1961–1990 climatology."""
    da = load_field(DRIVER_FILES[dname], DRIVER_VARS[dname])
    scale = DRIVER_SCALE.get(dname, 1.0)
    if scale != 1.0:
        da = da * scale
    annual = da.assign_coords(year=("time", da["time"].dt.year.values))
    annual = annual.swap_dims({"time": "year"}).drop_vars("time")
    annual = annual.sel(year=slice(int(START_YEAR), int(END_YEAR)))
    clim   = reference_mean(annual, REF_START, REF_END)
    return compute_anomalies(annual, clim)


def top_tercile_years(index_field):
    """Years in the upper tercile (~top third) of the Germany-average index."""
    s = area_mean(index_field)
    df = pd.DataFrame({"year": s["year"].values.astype(int),
                       "value": s.values}).dropna()
    n_top = max(1, len(df) // 3)
    return df.nlargest(n_top, "value")["year"].values.astype(int)


def composite_high_vs_rest(driver_anom, high_years, all_years):
    """Composite = mean over high-index summers − mean over all other summers."""
    low_years = np.setdiff1d(all_years, high_years)
    mh = driver_anom.sel(year=high_years).mean("year", skipna=True)
    ml = driver_anom.sel(year=low_years ).mean("year", skipna=True)
    return mh - ml


def _interp_colors_local(palette, n):
    from matplotlib.colors import to_rgba
    rgba = np.array([to_rgba(c) for c in palette])
    xs = np.linspace(0, 1, len(palette)); xn = np.linspace(0, 1, n)
    return [tuple(r) for r in np.column_stack(
        [np.interp(xn, xs, rgba[:, i]) for i in range(4)])]


def _draw_composite_panel(ax, da, gdf, geom, levels, fmt, title, tag=None):
    """Draw one composite-anomaly panel into an existing cartopy axes."""
    cmap = mcolors.ListedColormap(_interp_colors_local(DIV_COLORS, len(levels) - 1))
    norm = mcolors.BoundaryNorm(levels, cmap.N)
    cmap.set_under(DIV_COLORS[0]); cmap.set_over(DIV_COLORS[-1])

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
        ax.text(0.03, 0.97, tag, transform=ax.transAxes, ha="left", va="top",
                fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    ax.set_title(title, fontsize=8.5, fontweight="bold", pad=3)
    style_axis(ax)

    de_mask = build_mask(da["lon"].values, da["lat"].values, geom)
    if de_mask.any():
        mean_v = float(np.nanmean(da.values[de_mask]))
        sign = "+" if mean_v >= 0 else ""
        ax.text(0.03, 0.03, f"DE: {sign}{fmt % mean_v}",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5,
                color="#222222",
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="#aaaaaa", alpha=0.88, lw=0.4))
    return cmap, norm


def plot_driver_panel_figure(composites, gdf, geom, outfile, suptitle):
    """
    One multi-panel composite figure — one panel per driver — showing the
    driver-anomaly field composited over the upper-tercile summers of the
    representative index.  Grid adapts to the number of drivers.
    """
    from matplotlib.gridspec import GridSpec

    driver_names = list(composites.keys())
    n = len(driver_names)
    ncols = 2 if n <= 4 else 3
    nrows = int(np.ceil(n / ncols))

    fig = plt.figure(figsize=(3.5 * ncols + 0.6, 3.7 * nrows))
    fig.patch.set_facecolor("white")
    if suptitle:
        fig.suptitle(suptitle, fontsize=10, y=1.00)

    gs = GridSpec(nrows, ncols * 2, width_ratios=([1, 0.06] * ncols),
                  left=0.02, right=0.96, top=0.93, bottom=0.04,
                  hspace=0.16, wspace=0.0)

    tags = list("abcdefgh")
    for k, dname in enumerate(driver_names):
        row = k // ncols
        col = (k % ncols) * 2
        ax  = fig.add_subplot(gs[row, col], projection=PROJ)
        da  = composites[dname]
        lvls = DRIVER_LEVELS[dname]
        cmap, norm = _draw_composite_panel(
            ax, da, gdf, geom, lvls, "%.1f",
            title=DRIVER_LONG[dname], tag=f"({tags[k]})")

        cax = ax.inset_axes([1.015, 0.0, 0.06, 1.0])
        cb  = ColorbarBase(cax, cmap=cmap, norm=norm, boundaries=lvls,
                           ticks=[lvls[0], 0, lvls[-1]],
                           orientation="vertical", extend="neither")
        cb.ax.tick_params(labelsize=5, pad=1)
        cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        cb.outline.set_linewidth(0.4)
        cb.set_label(DRIVER_UNITS[dname], fontsize=5.5, labelpad=2)

    fig.savefig(outfile, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def run_composite_group(group_name, drivers, driver_anoms, all_years,
                        gdf, geom):
    """Build the single composite panel figure for one group's representative index."""
    display_name, nc_stem, label = COMPOSITE_REPRESENTATIVE[group_name]
    try:
        index_field = load_index_field(nc_stem, "ICON")
    except FileNotFoundError as e:
        print(f"  [{group_name}] composite skipped — {e}")
        return

    high_years = top_tercile_years(index_field)
    print(f"  [{group_name}] {label}: upper-tercile years "
          f"({len(high_years)}) → composite")

    composites = {}
    for dname in drivers:
        if dname not in driver_anoms:
            continue
        anom = driver_anoms[dname]
        anom_years = anom["year"].values.astype(int)
        common = np.intersect1d(high_years, anom_years).astype(int)
        if len(common) < 3:
            print(f"    {dname}: <3 overlapping high years — skipping panel.")
            continue
        composites[dname] = composite_high_vs_rest(anom, common, anom_years)

    if len(composites) >= 2:
        plot_driver_panel_figure(
            composites, gdf, geom,
            outfile=os.path.join(FIGDIR, f"composite_{group_name.lower()}.png"),
            suptitle=(f"{group_name} drivers — composite anomalies during "
                      f"upper-tercile {label} summers (ICON-CLM, JJA 1950–2022)"))
        print(f"  [{group_name}] composite figure written.")


# ── heatmap figure ─────────────────────────────────────────────────────────────
def plot_correlation_heatmap(r_mat, p_mat, row_labels, col_labels,
                             outfile, title, cbar_label="Correlation r"):
    """
    Index × driver correlation heatmap.  Cell colour = correlation coefficient
    (diverging, −1…+1); cell text = r value with significance stars
    (* p<0.05, ** p<0.01).  Non-significant cells are left unstarred.
    """
    n_rows, n_cols = r_mat.shape

    cmap = mcolors.LinearSegmentedColormap.from_list("corr", CORR_COLORS)
    norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)

    fig, ax = plt.subplots(figsize=(0.95 * n_cols + 2.2, 0.75 * n_rows + 1.8))
    fig.patch.set_facecolor("white")

    im = ax.imshow(r_mat, cmap=cmap, norm=norm, aspect="auto")

    # Cell annotations
    for i in range(n_rows):
        for j in range(n_cols):
            r = r_mat[i, j]
            if not np.isfinite(r):
                continue
            txt = f"{r:+.2f}{_stars(p_mat[i, j])}"
            # White text on dark (strong) cells, dark text on pale cells
            txt_color = "white" if abs(r) > 0.55 else "#1a1a1a"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8, color=txt_color)

    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticklabels(row_labels, fontsize=9, fontweight="bold")
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    # Thin gridlines between cells
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.set_title(title + "\n(* p<0.05, ** p<0.01)", fontsize=10, pad=8)

    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, extend="neither")
    cb.set_label(cbar_label, fontsize=9)
    cb.ax.tick_params(labelsize=8)

    plt.tight_layout()
    fig.savefig(outfile, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ── per-index correlation bar chart ────────────────────────────────────────────
def plot_correlation_bars(corr_df, index_label, outfile):
    """
    Horizontal bar chart of Pearson r for one index vs all drivers.
    Bars coloured by sign; hatched = non-significant (p ≥ 0.05).
    """
    df = corr_df.dropna(subset=["pearson_r"]).copy()
    df = df.sort_values("pearson_r", ascending=True)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(5.5, 0.5 * len(df) + 1.2))
    fig.patch.set_facecolor("white")

    colors = ["#d73027" if r >= 0 else "#4575b4" for r in df["pearson_r"]]
    bars   = ax.barh(df["driver"], df["pearson_r"],
                     color=colors, height=0.55, edgecolor="k", linewidth=0.4)

    for bar, pval in zip(bars, df["pearson_p"]):
        if not np.isfinite(pval) or pval >= 0.05:
            bar.set_hatch("///")
            bar.set_edgecolor("0.50")

    for bar, r, p in zip(bars, df["pearson_r"], df["pearson_p"]):
        xpos = bar.get_width() + (0.01 if r >= 0 else -0.01)
        ha   = "left" if r >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"{r:+.2f}{_stars(p)}", va="center", ha=ha, fontsize=8)

    ax.axvline(0, color="0.3", lw=0.7)
    ax.set_xlim(-1.1, 1.1)
    ax.set_xlabel("Pearson r  (Germany average)", fontsize=9)
    ax.set_title(f"{index_label} — driver correlations\n"
                 "(hatched = p ≥ 0.05; * p<0.05; ** p<0.01)", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    fig.savefig(outfile, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ── one correlation group (a set of indices vs a set of drivers) ────────────────
def run_group(group_name, indices, drivers, index_series_loader,
              driver_series, long_rows):
    """
    Compute correlation matrices for one index group against its drivers,
    write the Pearson and Spearman heatmaps, per-index bar charts, and the
    wide-format CSV matrices.  Appends tidy rows to ``long_rows``.
    """
    avail_rows = []
    rP, pP, rS, pS = [], [], [], []

    for display_name, nc_stem, row_label in indices:
        try:
            index_series = index_series_loader(nc_stem, "ICON")
        except FileNotFoundError as e:
            print(f"  Skipping {display_name}: {e}")
            continue

        print(f"\n[{group_name}] Correlating: {display_name}")
        avail_rows.append((display_name, row_label))

        rowP, rowPp, rowS, rowSp = [], [], [], []
        for dname in drivers:
            res = correlate(index_series, driver_series[dname])
            rowP.append(res["pearson_r"]);   rowPp.append(res["pearson_p"])
            rowS.append(res["spearman_r"]);  rowSp.append(res["spearman_p"])

            long_rows.append({
                "group":       group_name,
                "index":       display_name,
                "index_label": row_label,
                "driver":      dname,
                "driver_long": DRIVER_LONG[dname],
                "n":           res["n"],
                "pearson_r":   round(res["pearson_r"], 3)  if np.isfinite(res["pearson_r"])  else np.nan,
                "pearson_p":   round(res["pearson_p"], 4)  if np.isfinite(res["pearson_p"])  else np.nan,
                "spearman_r":  round(res["spearman_r"], 3) if np.isfinite(res["spearman_r"]) else np.nan,
                "spearman_p":  round(res["spearman_p"], 4) if np.isfinite(res["spearman_p"]) else np.nan,
            })
            print(f"  {dname:5s}: r={res['pearson_r']:+.2f} (p={res['pearson_p']:.3f})  "
                  f"rho={res['spearman_r']:+.2f} (p={res['spearman_p']:.3f})")

        rP.append(rowP); pP.append(rowPp); rS.append(rowS); pS.append(rowSp)

        # Per-index Pearson bar chart
        idx_df = pd.DataFrame({"driver": drivers,
                               "pearson_r": rowP, "pearson_p": rowPp})
        plot_correlation_bars(
            idx_df, row_label,
            outfile=os.path.join(FIGDIR, f"{display_name}_driver_correlations_bar.png"))

    if not avail_rows:
        print(f"  [{group_name}] No index files available — skipping heatmaps.")
        return

    rP = np.array(rP); pP = np.array(pP)
    rS = np.array(rS); pS = np.array(pS)
    row_labels = [lbl for _, lbl in avail_rows]
    tag = group_name.lower()

    plot_correlation_heatmap(
        rP, pP, row_labels, drivers,
        outfile=os.path.join(FIGDIR, f"heatmap_{tag}_pearson.png"),
        title=f"{group_name} extremes — driver correlation (Pearson)\n"
              "ICON-CLM Germany average, JJA 1950–2022",
        cbar_label="Pearson r")
    plot_correlation_heatmap(
        rS, pS, row_labels, drivers,
        outfile=os.path.join(FIGDIR, f"heatmap_{tag}_spearman.png"),
        title=f"{group_name} extremes — driver correlation (Spearman rank)\n"
              "ICON-CLM Germany average, JJA 1950–2022",
        cbar_label="Spearman ρ")

    pd.DataFrame(rP, index=row_labels, columns=drivers).to_csv(
        os.path.join(TABDIR, f"correlation_matrix_{tag}_pearson.csv"))
    pd.DataFrame(rS, index=row_labels, columns=drivers).to_csv(
        os.path.join(TABDIR, f"correlation_matrix_{tag}_spearman.csv"))
    print(f"  [{group_name}] heatmaps + matrices written.")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("Loading Germany boundary ...")
    gdf, geom = load_country_shape(GERMANY_SHP)

    # ── Pre-load all driver Germany-average series once ───────────────────────
    print("Loading driver variables (JJA seasonal means) ...")
    driver_series = {}
    for dname in DRIVER_ORDER:
        try:
            driver_series[dname] = load_driver_series(dname)
            n = int(np.isfinite(driver_series[dname].values).sum())
            print(f"  OK  {dname:5s} ({DRIVER_LONG[dname]}) — {n} years")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}  →  skipping {dname}.")
        except Exception as e:
            print(f"  ERROR loading {dname}: {e}  →  skipping.")

    avail_drivers = [d for d in DRIVER_ORDER if d in driver_series]
    if not avail_drivers:
        raise SystemExit("No driver files could be loaded — check filenames.")
    print(f"\nAvailable drivers: {avail_drivers}")

    # ── 1. Driver overview time series (all available drivers) ────────────────
    print("\nGenerating driver overview time series ...")
    plot_driver_overview(
        driver_series, avail_drivers,
        outfile=os.path.join(FIGDIR, "driver_overview_timeseries.png"))

    # ── 2. Driver–driver correlation matrix ───────────────────────────────────
    print("Generating driver–driver correlation matrix ...")
    dmat = plot_driver_matrix(
        driver_series, avail_drivers,
        outfile=os.path.join(FIGDIR, "driver_correlation_matrix.png"))
    dmat.to_csv(os.path.join(TABDIR, "driver_correlation_matrix.csv"))

    # ── 3. Grouped index–driver correlation heatmaps ──────────────────────────
    long_rows = []

    temp_drivers   = [d for d in TEMP_DRIVERS   if d in driver_series]
    precip_drivers = [d for d in PRECIP_DRIVERS if d in driver_series]

    run_group("Temperature", TEMP_INDICES, temp_drivers,
              load_index_series, driver_series, long_rows)
    run_group("Precipitation", PRECIP_INDICES, precip_drivers,
              load_index_series, driver_series, long_rows)

    if not long_rows:
        raise SystemExit("No index files could be loaded — run script2 first.")

    # ── Tidy CSV table (all index–driver pairs, both groups) ──────────────────
    pd.DataFrame(long_rows).to_csv(
        os.path.join(TABDIR, "all_indices_driver_correlations.csv"), index=False)

    # ── 4. Composite anomaly maps (one figure per group, representative index) ─
    print("\nGenerating composite anomaly maps ...")
    driver_anoms = {}
    for dname in avail_drivers:
        try:
            driver_anoms[dname] = load_driver_anom(dname)
        except Exception as e:
            print(f"  ERROR loading anomaly field for {dname}: {e}")

    all_years = np.arange(int(START_YEAR), int(END_YEAR) + 1)
    run_composite_group("Temperature",   temp_drivers,   driver_anoms,
                        all_years, gdf, geom)
    run_composite_group("Precipitation", precip_drivers, driver_anoms,
                        all_years, gdf, geom)

    print("\n" + "=" * 60)
    print("Script 3 complete.")
    print(f"  Driver overview  → {FIGDIR}/driver_overview_timeseries.png")
    print(f"  Driver matrix    → {FIGDIR}/driver_correlation_matrix.png")
    print(f"  Temp heatmaps    → {FIGDIR}/heatmap_temperature_[pearson|spearman].png")
    print(f"  Precip heatmaps  → {FIGDIR}/heatmap_precipitation_[pearson|spearman].png")
    print(f"  Composite maps   → {FIGDIR}/composite_temperature.png")
    print(f"                     {FIGDIR}/composite_precipitation.png")
    print(f"  Bar charts       → {FIGDIR}/<index>_driver_correlations_bar.png")
    print(f"  Tables           → {TABDIR}/all_indices_driver_correlations.csv")
    print(f"                     {TABDIR}/correlation_matrix_<group>_[pearson|spearman].csv")
    print(f"                     {TABDIR}/driver_correlation_matrix.csv")
    print("=" * 60)
