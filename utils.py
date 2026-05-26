"""
utils.py
--------
Shared utilities for the summer-extremes analysis.

All functions are imported by the main scripts; nothing runs on import.

Scientific references for index definitions
-------------------------------------------
- ETCCDI: Expert Team on Climate Change Detection and Indices
  (https://etccdi.pacificclimate.org/)
- Alexander et al. (2006) Global observed changes in daily climate extremes
  of temperature and precipitation, JGR-Atmos.
- Frich et al. (2002) Observed coherent changes in climatic extremes during
  the second half of the twentieth century, Clim. Res.
- Zolina et al. (2010) Improving estimates of heavy and extreme precipitation
  using daily records from European rain gauges, J. Hydrometeor.
- Zolina et al. (2014) Precipitation variability and extremes in central Europe,
  Clim. Dyn.
- Fischer et al. (2014) Robust spatially aggregated projections of climate
  extremes, Nature Clim. Change.
- Donat et al. (2013) Updated analyses of temperature and precipitation extreme
  indices since the beginning of the twentieth century, JGR-Atmos.
- Moberg & Jones (2005) Trends in indices for extremes in daily temperature and
  precipitation in central and western Europe, Int. J. Climatol.
- Yue & Wang (2004) The Mann-Kendall test modified by effective sample size to
  detect trend in serially correlated hydrological series, Water Resour. Res.
"""

import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import theilslopes
import pymannkendall as mk
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon

# ── shared settings (imported by all scripts) ─────────────────────────────────
START_YEAR     = "1950"
END_YEAR       = "2022"
REF_START      = "1961"   # percentile threshold reference period (WMO)
REF_END        = "1990"
ANOM_START     = "1991"   # anomaly reference period (current WMO normal)
ANOM_END       = "2020"
ALPHA          = 0.05     # significance level for Mann-Kendall
MIN_VALID      = 10       # minimum valid years required for a trend estimate
DPI            = 600      # output resolution
MAP_EXTENT     = [5.8, 15.2, 47.4, 55.1]   # Germany [lon_min, lon_max, lat_min, lat_max]
DISPLAY_FACTOR = 5        # bilinear upsampling factor for map display (visual only)
WET_DAY_MIN    = 1.0      # mm/day — wet-day threshold (R95p, R99p, SDII, CWD, R95pTOT)
DRY_DAY_MAX    = 1.0      # mm/day — dry-day threshold (Dry_days, CDD)


# ── IPCC / publication style ──────────────────────────────────────────────────
def set_ipcc_style():
    """
    Apply IPCC AR6 Working Group I figure style globally.

    Call once at the start of each script.  Sets fonts, sizes, and defaults
    to match the standards used in IPCC AR6 and major climate journals.
    All subsequent matplotlib calls inherit these settings.
    """
    mpl.rcParams.update({
        "font.family":          "sans-serif",
        "font.sans-serif":      ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":            9,
        "axes.titlesize":       10,
        "axes.titleweight":     "bold",
        "axes.labelsize":       9,
        "xtick.labelsize":      8,
        "ytick.labelsize":      8,
        "legend.fontsize":      8,
        "legend.framealpha":    0.85,
        "legend.edgecolor":     "0.75",
        "figure.facecolor":     "white",
        "figure.dpi":           150,
        "savefig.dpi":          DPI,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.05,
        "savefig.facecolor":    "white",
        "axes.linewidth":       0.8,
        "grid.linewidth":       0.35,
        "grid.color":           "#cccccc",
        "grid.alpha":           0.55,
        "lines.linewidth":      1.5,
        "patch.linewidth":      0.6,
        "hatch.linewidth":      0.4,
    })


# ── data loading ───────────────────────────────────────────────────────────────
def load_field(path, varname, start=START_YEAR, end=END_YEAR):
    """
    Load a variable from a NetCDF file; harmonise lat/lon coordinate names.

    Parameters
    ----------
    path    : str  — path to NetCDF file
    varname : str  — variable name inside the file
    start, end : str — year strings for time slicing (e.g. "1950", "2022")

    Returns
    -------
    xr.DataArray sorted by lat and lon, sliced to [start, end].
    """
    ds = xr.open_dataset(path)
    rename = {}
    if "latitude"  in ds.coords: rename["latitude"]  = "lat"
    if "longitude" in ds.coords: rename["longitude"] = "lon"
    if rename:
        ds = ds.rename(rename)
    if varname not in ds:
        raise KeyError(
            f"Variable '{varname}' not found in {path}. "
            f"Available variables: {list(ds.data_vars)}"
        )
    da = ds[varname].sel(time=slice(start, end))
    return da.sortby("lat").sortby("lon")


def keep_jja(da):
    """
    Subset a daily DataArray to June–July–August days only.

    Returns a DataArray with only the JJA time steps retained.
    """
    return da.where(da["time"].dt.month.isin([6, 7, 8]), drop=True)


# ── climatology and anomalies ─────────────────────────────────────────────────
def annual_jja_mean(da):
    """Annual JJA mean (temperature or any continuous variable, °C or K)."""
    return da.groupby("time.year").mean("time").astype(np.float32)


def annual_jja_sum(da):
    """Annual JJA total (precipitation, mm season⁻¹)."""
    return da.groupby("time.year").sum("time").astype(np.float32)


def reference_mean(annual_da, start=ANOM_START, end=ANOM_END):
    """
    Climatological mean over a reference period (default: 1991-2020 WMO normal).

    Parameters
    ----------
    annual_da : xr.DataArray with 'year' dimension
    start, end : str — boundary years (inclusive)
    """
    return annual_da.sel(year=slice(int(start), int(end))).mean("year", skipna=True)


def compute_anomalies(annual_da, climatology):
    """Subtract a climatology DataArray from every year of *annual_da*."""
    return annual_da - climatology


def area_mean(da):
    """Unweighted spatial mean over all valid (non-NaN) Germany grid cells."""
    return da.mean(dim=("lat", "lon"), skipna=True)


def rmse(a, b):
    """Root-mean-square error between two 1-D arrays (NaN-safe)."""
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((a[valid] - b[valid]) ** 2)))


# ── trend estimation ──────────────────────────────────────────────────────────
def compute_trend_maps(annual_da):
    """
    Compute per-gridcell Theil-Sen slope (unit per decade) and Mann-Kendall
    p-value with Yue-Wang autocorrelation correction at every grid point.

    The Yue-Wang modification applies trend-free pre-whitening (TFPW) before
    the MK test, which removes inflation of the test statistic caused by
    lag-1 positive autocorrelation (Yue & Wang, 2004).  Falls back to the
    original MK test if the correction raises an exception.

    Grid cells with fewer than MIN_VALID valid years are left as NaN.

    Returns
    -------
    xr.Dataset with variables:
        sen_slope : Theil-Sen slope [unit / decade]
        mk_pvalue : Mann-Kendall two-tailed p-value
        mk_z      : Mann-Kendall standardised test statistic
    """
    years = annual_da["year"].values.astype(float)
    lat   = annual_da["lat"].values
    lon   = annual_da["lon"].values
    arr   = annual_da.values          # (year, lat, lon)

    slope_map = np.full((len(lat), len(lon)), np.nan)
    p_map     = np.full((len(lat), len(lon)), np.nan)
    z_map     = np.full((len(lat), len(lon)), np.nan)

    for i in range(len(lat)):
        for j in range(len(lon)):
            series = arr[:, i, j]
            valid  = np.isfinite(series)
            if valid.sum() < MIN_VALID:
                continue
            yy, xx = series[valid], years[valid]

            # Theil-Sen slope converted from per-year to per-decade
            slope, _, _, _ = theilslopes(yy, xx, 0.95)
            slope_map[i, j] = slope * 10.0

            # Mann-Kendall with Yue-Wang TFPW autocorrelation correction
            try:
                res = mk.yue_wang_modification_test(yy)
            except Exception:
                res = mk.original_test(yy)
            p_map[i, j] = res.p
            z_map[i, j] = res.z

    return xr.Dataset(
        {"sen_slope": (("lat", "lon"), slope_map),
         "mk_pvalue": (("lat", "lon"), p_map),
         "mk_z":      (("lat", "lon"), z_map)},
        coords={"lat": lat, "lon": lon},
    )


def series_stats(annual_series):
    """
    Trend statistics for a 1-D Germany-average annual time series.

    Returns
    -------
    dict with keys:
        sen_slope_decade : Theil-Sen slope per decade
        mk_p             : Mann-Kendall two-tailed p-value (Yue-Wang)
        mk_z             : Mann-Kendall Z statistic
    """
    years = annual_series["year"].values.astype(float)
    y     = annual_series.values
    valid = np.isfinite(y)
    if valid.sum() < MIN_VALID:
        return {"sen_slope_decade": np.nan, "mk_p": np.nan, "mk_z": np.nan}
    slope, _, _, _ = theilslopes(y[valid], years[valid], 0.95)
    try:
        res = mk.yue_wang_modification_test(y[valid])
    except Exception:
        res = mk.original_test(y[valid])
    return {
        "sen_slope_decade": float(slope * 10.0),
        "mk_p":             float(res.p),
        "mk_z":             float(res.z),
    }


# ── Germany shapefile and masking ─────────────────────────────────────────────
def load_country_shape(shp_path):
    """
    Load the Germany boundary shapefile and return (GeoDataFrame, unified geometry).

    The GeoDataFrame is reprojected to WGS-84 (EPSG:4326).  The unified geometry
    is the shapely union of all sub-geometries (used for clipping and masking).
    """
    gdf  = gpd.read_file(shp_path).to_crs("EPSG:4326")
    geom = unary_union(gdf.geometry)
    return gdf, geom


def _polygon_to_path(geom):
    """Convert a Shapely Polygon or MultiPolygon to a matplotlib Path."""
    vertices, codes = [], []

    def _add(poly):
        x, y = poly.exterior.coords.xy
        pts  = np.column_stack([x, y])
        c    = np.full(len(pts), Path.LINETO, dtype=np.uint8)
        c[0] = Path.MOVETO
        vertices.extend(pts.tolist())
        codes.extend(c.tolist())
        vertices.append((0, 0))
        codes.append(Path.CLOSEPOLY)

    if isinstance(geom, Polygon):
        _add(geom)
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            _add(poly)
    else:
        raise ValueError(f"Unsupported geometry type: {type(geom)}")
    return Path(np.asarray(vertices, float), np.asarray(codes))


def build_mask(lon, lat, geom):
    """
    Boolean array (True inside Germany) with shape (len(lat), len(lon)).

    Uses the matplotlib Path approach: a grid cell is 'inside' if its centre
    point lies inside the Germany boundary polygon.
    """
    path   = _polygon_to_path(geom)
    lo, la = np.meshgrid(lon, lat)
    pts    = np.column_stack([lo.ravel(), la.ravel()])
    return path.contains_points(pts).reshape(len(lat), len(lon))


def apply_mask(arr2d, mask2d):
    """Set grid cells outside *mask2d* (False) to NaN."""
    out = np.array(arr2d, dtype=float)
    out[~mask2d] = np.nan
    return out


def clip_contourf(cf_obj, ax, geom):
    """Clip a filled-contour collection to the Germany boundary polygon."""
    clip = PathPatch(_polygon_to_path(geom), transform=ax.transData, facecolor="none")
    for coll in cf_obj.collections:
        coll.set_clip_path(clip)


def interp_display(da2d, factor=DISPLAY_FACTOR):
    """
    Bilinearly upsample a 2-D DataArray for smoother map rendering.

    This is purely cosmetic (display quality); all statistics are computed
    on the original grid and are never affected by this function.
    """
    lo      = da2d["lon"].values
    la      = da2d["lat"].values
    lo_fine = np.linspace(lo.min(), lo.max(), len(lo) * factor)
    la_fine = np.linspace(la.min(), la.max(), len(la) * factor)
    return da2d.interp(lon=lo_fine, lat=la_fine, method="linear")


# ══════════════════════════════════════════════════════════════════════════════
#  Climatology and bias map figures (used by script1)
# ══════════════════════════════════════════════════════════════════════════════

def plot_climatology_maps(obs_clim, mod_clim, gdf, geom, outfile,
                          levels, colors, cbar_label, tick_fmt="%.0f",
                          suptitle=None):
    """
    Three-panel figure: (a) E-OBS climatology, (b) ICON-CLM climatology,
    (c) model bias (ICON minus E-OBS).

    Designed for showing the mean-state spatial pattern alongside the
    systematic model bias.  Panels (a) and (b) share one sequential colormap;
    panel (c) uses a diverging colormap centred on zero.

    Parameters
    ----------
    obs_clim, mod_clim : xr.DataArray (lat, lon)
        1991-2020 climatological mean for E-OBS and ICON-CLM.
    gdf  : GeoDataFrame
    geom : Shapely geometry
    outfile : str
    levels  : list — sequential boundary levels for panels (a) and (b)
    colors  : list — one colour per interval (len = len(levels)-1)
    cbar_label : str — colorbar label including units
    tick_fmt   : str — colorbar tick format
    suptitle   : str, optional
    """
    bias   = mod_clim - obs_clim
    n_lvl  = len(levels)
    maxabs = float(np.nanmax(np.abs(bias.values)))
    if maxabs < 1e-6:
        maxabs = 1.0
    # Symmetric diverging levels for the bias panel
    step     = maxabs / 5
    bias_lvl = [round(-maxabs + k * step, 3) for k in range(11)]
    bias_col = [
        "#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#f7f7f7",
        "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f",
    ]

    cmap_seq  = mcolors.ListedColormap(colors)
    norm_seq  = mcolors.BoundaryNorm(levels, cmap_seq.N)
    cmap_div  = mcolors.ListedColormap(bias_col)
    norm_div  = mcolors.BoundaryNorm(bias_lvl, cmap_div.N)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.4))
    fig.patch.set_facecolor("white")
    if suptitle:
        fig.suptitle(suptitle, fontsize=10, fontweight="bold", y=0.99)

    panel_labels = ["(a)", "(b)", "(c)"]
    panel_titles = ["E-OBS", "ICON-CLM", "Bias (ICON − E-OBS)"]

    for k, (ax, da, cmap, norm, lvls, title, plabel) in enumerate(zip(
            axes,
            [obs_clim, mod_clim, bias],
            [cmap_seq, cmap_seq, cmap_div],
            [norm_seq, norm_seq, norm_div],
            [levels,   levels,   bias_lvl],
            panel_titles, panel_labels,
    )):
        fine = interp_display(da)
        mask = build_mask(fine["lon"].values, fine["lat"].values, geom)
        arr  = apply_mask(fine.values, mask)

        cf = ax.contourf(
            fine["lon"].values, fine["lat"].values, arr,
            levels=lvls, cmap=cmap, norm=norm,
            extend="both", antialiased=True,
        )
        clip_contourf(cf, ax, geom)
        gdf.boundary.plot(ax=ax, color="black", linewidth=0.55, zorder=5)
        ax.text(0.03, 0.97, plabel, transform=ax.transAxes,
                ha="left", va="top", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75))
        ax.set_title(title, fontsize=9.5, fontweight="bold", pad=4)
        style_axis(ax)

        # Individual colorbar below each panel
        plt.subplots_adjust(left=0.04, right=0.98, top=0.88,
                            bottom=0.22, wspace=0.18)
        pos  = ax.get_position()
        cax  = fig.add_axes([pos.x0 + 0.01, 0.10, pos.width - 0.02, 0.040])
        fmt  = tick_fmt if k < 2 else "%.2f"
        ticks = [lvls[0], lvls[len(lvls)//2], lvls[-1]] if k == 2 else lvls
        cb   = ColorbarBase(cax, cmap=cmap, norm=norm, boundaries=lvls,
                            ticks=ticks, orientation="horizontal", extend="both")
        cb.ax.tick_params(labelsize=6.5, pad=1.5)
        cb.ax.xaxis.set_major_formatter(FormatStrFormatter(fmt))
        if k == 1:
            cb.set_label(cbar_label, fontsize=8, labelpad=3)
        elif k == 2:
            cb.set_label(f"Bias [{cbar_label.split('[')[-1].rstrip(']') if '[' in cbar_label else cbar_label}]",
                         fontsize=8, labelpad=3)

    fig.savefig(outfile, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  NEW: Annual precipitation extreme index functions
#  All functions operate on a daily JJA DataArray (time, lat, lon).
#  Output: (year, lat, lon) DataArray with integer year coordinate.
# ══════════════════════════════════════════════════════════════════════════════

def annual_rx1day(daily_jja):
    """
    Rx1day — Maximum 1-day precipitation amount (mm day⁻¹) per JJA season.

    The single highest daily total recorded in June–July–August each year.
    Rx1day directly captures the intensity of the most extreme single rainfall
    event in the season; high values are associated with flash-flood hazard
    and convective precipitation extremes in central Europe.

    Reference: ETCCDI; Frich et al. (2002); Alexander et al. (2006).
    """
    return daily_jja.groupby("time.year").max("time").astype(np.float32)


def annual_rx5day(daily_jja):
    """
    Rx5day — Maximum consecutive 5-day precipitation amount (mm) per JJA season.

    The maximum sum of any five consecutive days within JJA.  This multi-day
    accumulation metric is closely linked to basin-scale hydrological flooding
    risk: a prolonged heavy-rain event saturates catchments and can produce
    river floods even when individual days do not exceed flash-flood thresholds.

    Implementation: sliding window of width 5 shifted one day at a time;
    maximum over all windows in the season.

    Reference: ETCCDI; Zolina et al. (2014).
    """
    years = np.unique(daily_jja["time"].dt.year.values).astype(int)
    lat   = daily_jja["lat"].values
    lon   = daily_jja["lon"].values
    out   = np.full((len(years), len(lat), len(lon)), np.nan, dtype=np.float32)

    for yi, yr in enumerate(years):
        season = daily_jja.sel(time=str(yr)).values   # (T, lat, lon)
        T = season.shape[0]
        if T < 5:
            continue
        # Compute all 5-day rolling sums efficiently via cumulative sum
        # roll_sums[k] = season[k] + … + season[k+4]  for k = 0 … T-5
        roll_sums = np.zeros((T - 4, len(lat), len(lon)), dtype=np.float32)
        for k in range(5):
            roll_sums += season[k: T - 4 + k]
        out[yi] = roll_sums.max(axis=0)

    return xr.DataArray(
        out,
        coords={"year": years, "lat": lat, "lon": lon},
        dims=("year", "lat", "lon"),
        name="Rx5day",
    )


def annual_r10mm(daily_jja):
    """
    R10mm — Count of heavy precipitation days (P ≥ 10 mm day⁻¹) per JJA season.

    The number of days per summer season with daily precipitation at or above
    10 mm.  This threshold broadly marks the onset of rapid surface runoff in
    most soil types and is among the most widely cited ETCCDI indices in
    European precipitation studies.

    Reference: ETCCDI; Donat et al. (2013).
    """
    return (
        (daily_jja >= 10.0)
        .astype(np.float32)
        .groupby("time.year")
        .sum("time")
    )


def annual_r20mm(daily_jja):
    """
    R20mm — Count of very heavy precipitation days (P ≥ 20 mm day⁻¹) per JJA season.

    The number of days per summer season with daily precipitation at or above
    20 mm.  This higher threshold is broadly associated with urban flash-flood
    triggering in Germany and targets the most intense convective rainfall events.

    Reference: ETCCDI; Zeder & Fischer (2020).
    """
    return (
        (daily_jja >= 20.0)
        .astype(np.float32)
        .groupby("time.year")
        .sum("time")
    )


def annual_sdii(daily_jja, wet_min=WET_DAY_MIN):
    """
    SDII — Simple Daily Intensity Index (mm wet-day⁻¹) per JJA season.

    Ratio of total wet-day precipitation to the number of wet days (P ≥ 1 mm).
    SDII isolates changes in precipitation *intensity* from changes in
    *frequency*, which is critical for understanding whether Germany's summer
    precipitation is becoming more or less intense per event independent of
    whether events become more or less frequent.

    Reference: ETCCDI; Zolina et al. (2010).
    """
    wet_mask  = daily_jja >= wet_min
    wet_total = (
        daily_jja.where(wet_mask)
        .groupby("time.year")
        .sum("time", skipna=True)
    )
    wet_count = (
        wet_mask.astype(np.float32)
        .groupby("time.year")
        .sum("time")
    )
    # Avoid division by zero for grid cells / years with no wet days
    sdii = wet_total / wet_count.where(wet_count > 0)
    return sdii.astype(np.float32)


def annual_r95ptot(daily_jja, threshold_r95, wet_min=WET_DAY_MIN):
    """
    R95pTOT — Fraction of total wet-day precipitation from very heavy events (%).

    Percentage of the seasonal wet-day precipitation total that falls on days
    exceeding the R95p threshold (95th percentile of the 1961-1990 wet-day
    distribution).  High R95pTOT values indicate that a large share of seasonal
    rainfall is delivered in a few extreme events — a key fingerprint of
    precipitation intensification and concentration independent of total amount.

    Reference: Fischer et al. (2014); Zolina et al. (2014).
    """
    # Precipitation on days exceeding the local R95p threshold
    heavy = (
        daily_jja.where(daily_jja > threshold_r95)
        .groupby("time.year")
        .sum("time", skipna=True)
    )
    # Total wet-day precipitation (denominator)
    total = (
        daily_jja.where(daily_jja >= wet_min)
        .groupby("time.year")
        .sum("time", skipna=True)
    )
    frac = (heavy / total.where(total > 0)) * 100.0
    return frac.astype(np.float32)


def annual_cwd(daily_jja, wet_min=WET_DAY_MIN):
    """
    CWD — Maximum consecutive wet days (P ≥ 1 mm day⁻¹) per JJA season.

    The length of the longest unbroken sequence of wet days in JJA.  CWD
    complements CDD: while CDD captures drought duration, CWD quantifies
    sustained rainfall episodes that saturate soils and significantly elevate
    flood risk through cumulative antecedent moisture, even when individual
    daily totals remain moderate.

    Implementation: run-length encoding via np.diff on the binary wet-day mask,
    identical in structure to annual_cdd for consistency.

    Reference: ETCCDI; Moberg & Jones (2005).
    """
    years = np.unique(daily_jja["time"].dt.year.values).astype(int)
    lat   = daily_jja["lat"].values
    lon   = daily_jja["lon"].values
    out   = np.zeros((len(years), len(lat), len(lon)), dtype=np.float32)

    for yi, yr in enumerate(years):
        wet = (daily_jja.sel(time=str(yr)).values >= wet_min).astype(np.int8)
        # wet shape: (time, lat, lon)
        for i in range(len(lat)):
            for j in range(len(lon)):
                d = wet[:, i, j]
                if d.sum() == 0:
                    out[yi, i, j] = 0.0
                    continue
                # Run-length encoding: find start and end indices of wet spells
                padded = np.concatenate([[0], d, [0]])
                diff   = np.diff(padded)
                starts = np.where(diff ==  1)[0]
                ends   = np.where(diff == -1)[0]
                out[yi, i, j] = float(np.max(ends - starts))

    return xr.DataArray(
        out,
        coords={"year": years, "lat": lat, "lon": lon},
        dims=("year", "lat", "lon"),
        name="CWD",
    )


# ── map axis styling ──────────────────────────────────────────────────────────
def style_axis(ax):
    """
    Apply uniform IPCC-style cartographic formatting to a map axis.

    Sets background colour, lon/lat ticks, gridlines, axis borders, and
    fixes the aspect ratio for Germany's domain.
    """
    ax.set_facecolor("#d9e9f2")   # light blue ocean/background
    ax.set_xlim(MAP_EXTENT[0], MAP_EXTENT[1])
    ax.set_ylim(MAP_EXTENT[2], MAP_EXTENT[3])
    ax.set_box_aspect(1)
    ax.set_xticks(np.arange(6, 16, 3))
    ax.set_yticks(np.arange(48, 56, 2))
    ax.set_xticklabels([f"{v}°E" for v in np.arange(6, 16, 3)], fontsize=6)
    ax.set_yticklabels([f"{v}°N" for v in np.arange(48, 56, 2)], fontsize=6)
    ax.tick_params(axis="both", which="both", direction="out",
                   top=False, right=False, pad=1.5)
    ax.grid(True, linestyle="--", linewidth=0.35, color="0.60", alpha=0.60, zorder=0)
    for sp in ax.spines.values():
        sp.set_linewidth(0.55)


# ── paired trend map figure (E-OBS vs ICON-CLM) ───────────────────────────────
def plot_paired_trend_maps(
    obs_slope, model_slope,
    obs_pval,  model_pval,
    gdf, geom,
    outfile, levels, colors,
    cbar_label,
    title_obs="E-OBS", title_model="ICON-CLM",
    tick_fmt="%.1f",
    suptitle=None,
):
    """
    Two-panel trend map: (a) E-OBS, (b) ICON-CLM.

    - Shared horizontal colorbar at the bottom.
    - Stippling (small dots) marks grid cells where the Mann-Kendall trend
      is statistically significant at p < 0.05 (Yue-Wang correction).
    - Maps are bilinearly upsampled for display smoothness (DISPLAY_FACTOR).
    - Germany boundary is drawn from the shapefile.

    Parameters
    ----------
    obs_slope, model_slope : xr.DataArray (lat, lon)
        Theil-Sen trend slope per decade.
    obs_pval, model_pval : xr.DataArray (lat, lon)
        Mann-Kendall p-values.
    gdf   : GeoDataFrame — Germany shapefile for boundary drawing.
    geom  : Shapely geometry — unified Germany polygon for contour clipping.
    outfile : str — output file path (PNG, DPI=600).
    levels : list of float — colour boundary levels.
    colors : list of str — one colour per interval.
    cbar_label : str — colorbar axis label (e.g. "days decade⁻¹").
    tick_fmt : str — colorbar tick format string.
    suptitle : str, optional — figure-level title.
    """
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(levels, cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
    fig.patch.set_facecolor("white")
    if suptitle:
        fig.suptitle(suptitle, fontsize=10, fontweight="bold", y=0.99)

    for ax, slope, pval, ds_title, panel_label in [
        (axes[0], obs_slope,   obs_pval,   title_obs,   "(a)"),
        (axes[1], model_slope, model_pval, title_model, "(b)"),
    ]:
        # Bilinear upsampling for display quality
        fine = interp_display(slope)
        mask = build_mask(fine["lon"].values, fine["lat"].values, geom)
        arr  = apply_mask(fine.values, mask)

        cf = ax.contourf(
            fine["lon"].values, fine["lat"].values, arr,
            levels=levels, cmap=cmap, norm=norm,
            extend="both", antialiased=True,
        )
        clip_contourf(cf, ax, geom)
        gdf.boundary.plot(ax=ax, color="black", linewidth=0.60, zorder=5)

        # Stippling: small dots on the original (coarse) grid where p < ALPHA
        lo2d, la2d = np.meshgrid(slope["lon"].values, slope["lat"].values)
        sig_mask   = pval.values < ALPHA
        ax.scatter(
            lo2d[sig_mask], la2d[sig_mask],
            s=0.55, c="#1a1a1a", alpha=0.28, zorder=6, rasterized=True,
        )

        # Panel label (top-left inset box)
        ax.text(0.03, 0.97, panel_label, transform=ax.transAxes,
                ha="left", va="top", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18", fc="white",
                          ec="none", alpha=0.75))
        ax.set_title(ds_title, fontsize=9.5, fontweight="bold", pad=4)
        style_axis(ax)

    plt.subplots_adjust(left=0.05, right=0.97, top=0.88, bottom=0.21, wspace=0.14)

    # Shared horizontal colorbar
    cax = fig.add_axes([0.15, 0.09, 0.70, 0.042])
    cb  = ColorbarBase(
        cax, cmap=cmap, norm=norm, boundaries=levels,
        ticks=levels, orientation="horizontal", extend="both",
    )
    cb.ax.tick_params(labelsize=7, pad=1.5)
    cb.ax.xaxis.set_major_formatter(FormatStrFormatter(tick_fmt))
    cb.set_label(cbar_label, fontsize=8, labelpad=3)

    fig.savefig(outfile, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ── Germany-average time series plot ─────────────────────────────────────────
def plot_germany_series(
    obs_series, model_series,
    obs_anom,   model_anom,
    ylabel, title, outfile,
    ylabel_anom=None,
    obs_stats=None, model_stats=None,
):
    """
    Two-panel IPCC-style time series figure for Germany-average values.

    Top panel — Absolute annual values
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - Annual values: semi-transparent scatter points (E-OBS: green, ICON: red)
    - 5-yr running mean: dashed line (centre-aligned, min 3 years)
    - 11-yr running mean: bold solid line (centre-aligned, min 6 years)
    - Linear trend annotation (Sen slope/decade + MK p-value)

    Bottom panel — Anomalies relative to 1991-2020
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - E-OBS anomaly: colour bars (red = positive, blue = negative)
    - ICON-CLM anomaly: stepped grey line
    - 11-yr running E-OBS anomaly mean: bold green line
    - 1991-2020 reference period: light grey background band

    Parameters
    ----------
    obs_series, model_series : xr.DataArray (year,)
        Annual Germany-average absolute index values.
    obs_anom, model_anom : xr.DataArray (year,)
        Anomalies relative to 1991-2020.
    ylabel : str — y-axis label for the absolute panel (includes unit).
    ylabel_anom : str, optional — y-axis label for the anomaly panel.
    title : str — figure-level title.
    outfile : str — output path (PNG, 600 dpi).
    obs_stats, model_stats : dict, optional
        Output of series_stats(); used for trend annotation.
        Keys: sen_slope_decade, mk_p.
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7.0), sharex=True,
        gridspec_kw={"hspace": 0.05, "height_ratios": [1.45, 1]},
    )
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=11, fontweight="bold", y=0.98)

    OBS_COL = "#1b7837"    # E-OBS: dark green
    MOD_COL = "#b2182b"    # ICON-CLM: dark red

    years      = obs_series["year"].values.astype(int)
    obs_vals   = obs_series.values.astype(float)
    mod_vals   = model_series.values.astype(float)

    # ── Top panel: absolute values ────────────────────────────────────────────
    for vals, col, lbl in [
        (obs_vals, OBS_COL, "E-OBS"),
        (mod_vals, MOD_COL, "ICON-CLM"),
    ]:
        s = pd.Series(vals, index=years, dtype=float)

        # Raw annual scatter (small, semi-transparent)
        ax1.scatter(years, vals, color=col, s=9, alpha=0.30, zorder=2, linewidths=0)

        # 5-yr running mean (dashed) — slightly lighter than 11-yr for visual separation
        rm5 = s.rolling(5, center=True, min_periods=3).mean()
        ax1.plot(years, rm5.values, "--", color=col, lw=1.5, alpha=0.80, zorder=3)

        # 10-yr running mean (solid, bold) — IPCC AR6 standard decadal smoother
        rm10 = s.rolling(10, center=True, min_periods=5).mean()
        ax1.plot(years, rm10.values, "-", color=col, lw=2.2, alpha=0.92,
                 label=lbl, zorder=4)

    # Trend annotation boxes
    if obs_stats and model_stats:
        for stats, col, xpos in [
            (obs_stats,   OBS_COL, 0.02),
            (model_stats, MOD_COL, 0.52),
        ]:
            sl = stats.get("sen_slope_decade", np.nan)
            pv = stats.get("mk_p", np.nan)
            if not (np.isfinite(sl) and np.isfinite(pv)):
                continue
            sig_str = "**" if pv < 0.01 else ("*" if pv < 0.05 else " ns")
            sign    = "+" if sl >= 0 else ""
            unit_short = ylabel.split("[")[-1].rstrip("]") if "[" in ylabel else ylabel
            txt = f"Trend: {sign}{sl:.2f} {unit_short} dec⁻¹\np = {pv:.3f}{sig_str}"
            ax1.text(xpos, 0.97, txt, transform=ax1.transAxes,
                     ha="left", va="top", fontsize=7.5, color=col,
                     bbox=dict(boxstyle="round,pad=0.25", fc="white",
                               ec=col, alpha=0.85, lw=0.8))

    ax1.set_ylabel(ylabel, fontsize=9)
    ax1.legend(fontsize=8.5, frameon=True, framealpha=0.85,
               edgecolor="0.70", loc="lower right")
    ax1.tick_params(labelsize=8)
    ax1.grid(True, linestyle="--", linewidth=0.35, alpha=0.55, zorder=0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines[["left", "bottom"]].set_linewidth(0.7)
    ax1.text(0.005, 0.98, "(a)", transform=ax1.transAxes,
             fontsize=10, fontweight="bold", va="top")

    # Legend entry for running means
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color="0.40", lw=1.5, ls="--", label="5-yr mean"),
        Line2D([0], [0], color="0.40", lw=2.2, ls="-",  label="10-yr mean"),
        Line2D([0], [0], color="0.40", marker="o", ms=4,
               ls="none", alpha=0.4, label="Annual"),
    ]
    ax1.legend(
        handles=list(ax1.get_legend_handles_labels()[0]) + custom_lines,
        fontsize=7.5, frameon=True, framealpha=0.85, edgecolor="0.70",
        loc="lower right", ncol=2,
    )

    # ── Bottom panel: anomalies ───────────────────────────────────────────────
    anom_yrs = obs_anom["year"].values.astype(int)
    obs_a    = obs_anom.values.astype(float)
    mod_a    = model_anom.values.astype(float)

    # Reference period shading (1991-2020)
    ax2.axvspan(1991, 2020, color="#f0f0f0", zorder=0)
    ax2.axhline(0, color="0.35", lw=0.80, zorder=1)

    # E-OBS anomaly bars coloured by sign (IPCC convention: warm=red, cool=blue)
    bar_w   = 0.42
    obs_pos = np.where(obs_a >= 0, obs_a, 0.0)
    obs_neg = np.where(obs_a < 0,  obs_a, 0.0)
    ax2.bar(anom_yrs - bar_w / 2, obs_pos, bar_w,
            color="#d73027", alpha=0.78, label="E-OBS +anom", zorder=2)
    ax2.bar(anom_yrs - bar_w / 2, obs_neg, bar_w,
            color="#4575b4", alpha=0.78, label="E-OBS −anom", zorder=2)

    # 11-yr running means for both — smoothed decadal signal for direct comparison
    s_obs_a  = pd.Series(obs_a, index=anom_yrs, dtype=float)
    s_mod_a  = pd.Series(mod_a, index=anom_yrs, dtype=float)
    rm10_obs = s_obs_a.rolling(10, center=True, min_periods=5).mean()
    rm10_mod = s_mod_a.rolling(10, center=True, min_periods=5).mean()

    ax2.plot(anom_yrs, rm10_obs.values, "-", color=OBS_COL, lw=2.0,
             label="E-OBS 10-yr", zorder=5, alpha=0.90)
    ax2.plot(anom_yrs, rm10_mod.values, "-", color=MOD_COL, lw=2.0,
             label="ICON-CLM 10-yr", zorder=5, alpha=0.90)

    ylabel_anom = ylabel_anom or f"Anomaly [{ylabel}]"
    ax2.set_ylabel(ylabel_anom, fontsize=9)
    ax2.set_xlabel("Year", fontsize=9)
    ax2.legend(fontsize=7.5, frameon=True, framealpha=0.85, edgecolor="0.70",
               loc="lower left", ncol=2)
    ax2.tick_params(labelsize=8)
    ax2.grid(True, linestyle="--", linewidth=0.35, alpha=0.55, zorder=0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines[["left", "bottom"]].set_linewidth(0.7)
    ax2.text(0.005, 0.98, "(b)", transform=ax2.transAxes,
             fontsize=10, fontweight="bold", va="top")
    ax2.text(0.79, 0.96, "grey band = 1991-2020 ref. period",
             transform=ax2.transAxes, fontsize=6.5, color="0.55", va="top")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outfile, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  NEW: Multi-panel precipitation extreme overview figure
# ══════════════════════════════════════════════════════════════════════════════

def plot_precipitation_overview(index_meta, gdf, geom, outfile):
    """
    Flagship multi-panel figure: trend maps for all precipitation indices.

    Creates an N×2 grid (E-OBS left | ICON-CLM right) for N indices.
    Per-row vertical colorbars on the right side.  Stippling for p < 0.05.
    Designed to be the thesis overview figure for precipitation extremes.

    Parameters
    ----------
    index_meta : dict[str, dict]
        Keys: short index name.
        Each value must contain:
          obs_slope, mod_slope : xr.DataArray (lat, lon)
          obs_pval,  mod_pval  : xr.DataArray (lat, lon)
          levels, colors       : colormap specification
          cbar_label, tick_fmt : colorbar formatting
          long_name            : row label (e.g. "Rx1day [mm day⁻¹]")
    gdf  : GeoDataFrame  — Germany boundary.
    geom : Shapely geometry — for contour clipping.
    outfile : str
    """
    n    = len(index_meta)
    fig, axes = plt.subplots(n, 2, figsize=(8.8, n * 2.8))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "JJA Precipitation Extreme Indices — Theil-Sen Trends 1950–2022\n"
        "Stippling: statistically significant trend (MK p < 0.05, Yue-Wang correction)",
        fontsize=9.5, fontweight="bold", y=0.998,
    )

    letters = "abcdefghijklmnopqrstuvwxyz"
    letter_idx = 0

    for row, (name, d) in enumerate(index_meta.items()):
        cmap = mcolors.ListedColormap(d["colors"])
        norm = mcolors.BoundaryNorm(d["levels"], cmap.N)

        for col, (slope, pval, ds_title) in enumerate([
            (d["obs_slope"], d["obs_pval"], "E-OBS"),
            (d["mod_slope"], d["mod_pval"], "ICON-CLM"),
        ]):
            ax = axes[row, col] if n > 1 else axes[col]

            fine = interp_display(slope)
            mask = build_mask(fine["lon"].values, fine["lat"].values, geom)
            arr  = apply_mask(fine.values, mask)

            cf = ax.contourf(
                fine["lon"].values, fine["lat"].values, arr,
                levels=d["levels"], cmap=cmap, norm=norm,
                extend="both", antialiased=True,
            )
            clip_contourf(cf, ax, geom)
            gdf.boundary.plot(ax=ax, color="black", linewidth=0.50, zorder=5)

            # Stippling on original grid where p < ALPHA
            lo2d, la2d = np.meshgrid(slope["lon"].values, slope["lat"].values)
            sig_mask   = pval.values < ALPHA
            ax.scatter(lo2d[sig_mask], la2d[sig_mask],
                       s=0.35, c="#111111", alpha=0.26,
                       zorder=6, rasterized=True)

            # Panel letter label
            ax.text(0.03, 0.97, f"({letters[letter_idx]})",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white",
                              ec="none", alpha=0.72))
            letter_idx += 1

            # Column title (top row only)
            if row == 0:
                ax.set_title(ds_title, fontsize=9, fontweight="bold", pad=3)

            # Row label on left column only
            if col == 0:
                ax.set_ylabel(d["long_name"], fontsize=7.5, labelpad=2)

            style_axis(ax)

            # Vertical colorbar attached to the right panel of each row
            if col == 1:
                cb_ax = ax.inset_axes([1.05, 0.0, 0.07, 1.0])
                cb = ColorbarBase(
                    cb_ax, cmap=cmap, norm=norm,
                    boundaries=d["levels"], ticks=d["levels"],
                    orientation="vertical", extend="both",
                )
                cb.ax.tick_params(labelsize=5.5, pad=1)
                cb.ax.yaxis.set_major_formatter(FormatStrFormatter(d["tick_fmt"]))
                cb.set_label(d["cbar_label"], fontsize=6, labelpad=3)

    plt.subplots_adjust(
        left=0.10, right=0.88, top=0.97, bottom=0.01,
        hspace=0.07, wspace=0.12,
    )
    fig.savefig(outfile, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  NEW: Taylor diagram
# ══════════════════════════════════════════════════════════════════════════════

def taylor_diagram(obs_dict, mod_dict, outfile,
                   title="Model Skill — ICON-CLM vs E-OBS (Germany average)"):
    """
    Taylor diagram comparing ICON-CLM to E-OBS for all extreme indices.

    Each point represents one index and is placed at:
    - Angular position: correlation coefficient r (θ = arccos r)
    - Radial position:  normalised standard deviation σ_model / σ_obs
    Centred RMSE contours (arcs from the reference point) are also drawn.

    The reference point (obs) sits at (θ=0°, r_norm=1.0).
    A perfect model would coincide with the reference point.

    Parameters
    ----------
    obs_dict : dict {index_name: np.ndarray}
        Germany-average annual observed series for each index.
    mod_dict : dict {index_name: np.ndarray}
        Corresponding ICON-CLM series.
    outfile  : str — output path.
    title    : str — figure title.
    """
    # ── Compute per-index statistics ──────────────────────────────────────────
    names, corrs, norm_stds = [], [], []
    for name, obs_vals in obs_dict.items():
        mod_vals = mod_dict.get(name)
        if mod_vals is None:
            continue
        n = min(len(obs_vals), len(mod_vals))
        o, m = obs_vals[:n].astype(float), mod_vals[:n].astype(float)
        valid = np.isfinite(o) & np.isfinite(m)
        if valid.sum() < 5:
            continue
        o, m = o[valid], m[valid]
        corr    = float(np.corrcoef(o, m)[0, 1])
        std_obs = float(np.std(o, ddof=1))
        if std_obs < 1e-9:
            continue
        nstd = float(np.std(m, ddof=1)) / std_obs
        names.append(name)
        corrs.append(np.clip(corr, -1.0, 1.0))
        norm_stds.append(nstd)

    if not names:
        warnings.warn("taylor_diagram: no valid index pairs found; skipping.")
        return

    # ── Build polar plot ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.0, 6.0),
                           subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("white")

    ax.set_theta_zero_location("N")   # θ=0 at top (r=1.0 = perfect correlation)
    ax.set_theta_direction(-1)        # clockwise increasing θ
    ax.set_thetamin(0)
    ax.set_thetamax(180)

    # Draw std-dev reference arcs
    for r_ref in [0.5, 1.0, 1.5, 2.0]:
        arc = np.linspace(0, np.pi, 200)
        ax.plot(arc, np.full_like(arc, r_ref), "--", color="0.72",
                lw=0.55, zorder=1)
        ax.text(np.pi * 0.80, r_ref + 0.04, f"{r_ref:.1f}",
                fontsize=6.5, color="0.50", ha="center")

    ax.text(np.pi * 0.80, 2.20, "Norm. Std Dev",
            fontsize=7, color="0.45", ha="center")

    # Draw correlation radial lines and labels
    for r_corr in [0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        ang = np.arccos(r_corr)
        ax.plot([ang, ang], [0, 2.05], ":", color="0.65", lw=0.5, zorder=1)
        ax.text(ang, 2.15, f"{r_corr:.2f}", ha="center", va="bottom",
                fontsize=6.5, color="0.50")
    ax.text(np.arccos(0.4) / 2, 2.35, "Correlation",
            fontsize=7, color="0.45", ha="center")

    # RMSE contours centred on the reference point (1, 0)
    theta_arc = np.linspace(0, np.pi, 400)
    for rms_v in [0.5, 1.0, 1.5]:
        # For a point at (theta, nstd): RMSE² = 1 + nstd² − 2·nstd·cos(theta)
        # → nstd² − 2·cos(theta)·nstd + (1 − RMSE²) = 0
        b = -2.0 * np.cos(theta_arc)
        c = 1.0 - rms_v ** 2
        discriminant = b ** 2 - 4.0 * c
        r_arc = np.full_like(theta_arc, np.nan)
        ok = discriminant >= 0
        r_arc[ok] = (-b[ok] + np.sqrt(discriminant[ok])) / 2.0
        r_arc[(r_arc < 0) | (r_arc > 2.5)] = np.nan
        ax.plot(theta_arc, r_arc, ":", color="#d73027", lw=0.8, alpha=0.60)
        mid = np.nanargmin(np.abs(theta_arc - np.pi / 5))
        if np.isfinite(r_arc[mid]):
            ax.text(theta_arc[mid], r_arc[mid] + 0.06,
                    f"RMSE={rms_v:.1f}", fontsize=6, color="#d73027", alpha=0.80)

    # Reference (obs) marker
    ax.plot([0], [1.0], "*", color="0.20", ms=13, zorder=7,
            markeredgecolor="k", markeredgewidth=0.5)
    ax.text(0.07, 1.08, "E-OBS (ref.)", fontsize=7.5, color="0.25")

    # Model markers
    palette = plt.cm.tab10(np.linspace(0, 1, len(names)))
    for i, (name, theta, nstd) in enumerate(
            zip(names, [np.arccos(c) for c in corrs], norm_stds)):
        ax.plot(theta, nstd, "o", color=palette[i], ms=9, zorder=6,
                markeredgecolor="k", markeredgewidth=0.5)
        short = name.replace("_exceedance_days", "").replace("_days", "")
        ax.text(theta + 0.06, nstd + 0.07, short,
                fontsize=7, ha="left", va="bottom", color=palette[i],
                fontweight="bold")

    ax.set_rlim(0, 2.2)
    ax.set_rticks([])
    ax.set_title(title, fontsize=9, fontweight="bold", pad=20)

    fig.savefig(outfile, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  NEW: Trend summary heatmap
# ══════════════════════════════════════════════════════════════════════════════

def plot_trend_heatmap(heatmap_rows, outfile,
                       title="Theil-Sen Trends per Decade — All Indices"):
    """
    Compact heatmap summarising all index trends for E-OBS and ICON-CLM.

    Each row is one extreme index; columns are E-OBS and ICON-CLM.  Cell
    colour reflects the sign and magnitude of the trend relative to the
    maximum absolute trend across both datasets (row-normalised).
    Significance is annotated (* p<0.05, ** p<0.01).

    Parameters
    ----------
    heatmap_rows : list of dict, each with keys:
        name       : str — display name
        obs_slope  : float — E-OBS Theil-Sen slope per decade
        mod_slope  : float — ICON-CLM Theil-Sen slope per decade
        obs_pval   : float — E-OBS MK p-value
        mod_pval   : float — ICON-CLM MK p-value
        trend_unit : str — unit string
    outfile : str
    """
    n         = len(heatmap_rows)
    names     = [d["name"] for d in heatmap_rows]
    obs_sl    = np.array([d["obs_slope"]  for d in heatmap_rows], dtype=float)
    mod_sl    = np.array([d["mod_slope"]  for d in heatmap_rows], dtype=float)
    obs_pv    = np.array([d["obs_pval"]   for d in heatmap_rows], dtype=float)
    mod_pv    = np.array([d["mod_pval"]   for d in heatmap_rows], dtype=float)
    units     = [d["trend_unit"] for d in heatmap_rows]

    # Row-normalise so every index spans [−1, +1] in colour space
    maxabs = np.maximum(np.abs(obs_sl), np.abs(mod_sl))
    maxabs[maxabs < 1e-9] = 1.0
    heat   = np.column_stack([obs_sl / maxabs, mod_sl / maxabs])  # (n, 2)

    fig_h  = max(4.0, 0.52 * n + 1.6)
    fig, ax = plt.subplots(figsize=(5.0, fig_h))
    fig.patch.set_facecolor("white")

    im = ax.imshow(heat, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")

    # Annotate each cell with the actual trend value and significance
    for i in range(n):
        for j, (sl, pv) in enumerate(
                [(obs_sl[i], obs_pv[i]), (mod_sl[i], mod_pv[i])]):
            sig = "**" if pv < 0.01 else ("*" if pv < 0.05 else "")
            cell_val = heat[i, j]
            txt_col  = "white" if abs(cell_val) > 0.60 else "black"
            ax.text(j, i,
                    f"{sl:+.2f}{sig}\n{units[i]}",
                    ha="center", va="center", fontsize=7.5,
                    color=txt_col, fontweight="bold" if sig else "normal")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["E-OBS", "ICON-CLM"], fontsize=10, fontweight="bold")
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_title(f"{title}\n(* p<0.05, ** p<0.01 — Mann-Kendall, Yue-Wang correction)",
                 fontsize=9, fontweight="bold", pad=6)
    ax.tick_params(axis="both", which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    cb = fig.colorbar(im, ax=ax, orientation="vertical",
                      fraction=0.040, pad=0.04, shrink=0.85)
    cb.set_label("Row-normalised trend", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)

    plt.tight_layout()
    fig.savefig(outfile, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
