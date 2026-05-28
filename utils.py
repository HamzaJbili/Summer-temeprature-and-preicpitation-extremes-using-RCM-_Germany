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
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import theilslopes
import pymannkendall as mk
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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


# ── Colorbar helpers ──────────────────────────────────────────────────────────
def _interp_colors(palette, n):
    """Return n evenly-interpolated hex colors from palette (no index repeats)."""
    cmap_tmp = mcolors.LinearSegmentedColormap.from_list('_', palette)
    return [mcolors.to_hex(cmap_tmp(x)) for x in np.linspace(0, 1, n)]


def _smart_ticks(lvls):
    """Return all level boundaries as tick positions (every band labeled)."""
    return list(lvls)


def _auto_scale_palette(data_arr, base_levels, base_colors):
    """Auto-scale levels and palette to one panel's data range.

    Returns (levels, colors, cmap, norm).
    """
    colors = list(base_colors)
    levels = list(base_levels)
    base_is_div = min(levels) < 0 < max(levels)
    valid = np.asarray(data_arr).ravel()
    valid = valid[np.isfinite(valid)]
    if len(valid) <= 10:
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(levels, cmap.N)
        return levels, colors, cmap, norm
    p2, p98      = np.percentile(valid, 2), np.percentile(valid, 98)
    is_diverging = base_is_div and p2 < 0
    if is_diverging:
        ext    = max(abs(p2), abs(p98))
        lo, hi = -ext, ext
    else:
        lo, hi = p2, p98
    n_target = 14 if is_diverging else 12
    span     = hi - lo
    raw_step = span / n_target
    mag      = 10.0 ** np.floor(np.log10(raw_step))
    step     = min([f * mag for f in [1, 2, 5, 10]],
                   key=lambda s: abs(span / s - n_target))
    if is_diverging:
        n_half = max(1, round(hi / step))
        if n_half % 2 != 0:
            n_half += 1
        nice = [round(i * step, 10) for i in range(-n_half, n_half + 1)]
    else:
        start = np.floor(lo / step) * step
        nice  = np.round(np.arange(start, hi + step * 0.01, step), 10).tolist()
        if hi < 0:
            nice = [t for t in nice if t < 0]
        elif lo > 0:
            nice = [t for t in nice if t > 0]
    margin = span * 0.08
    nice   = [float(t) for t in nice if (lo - margin) <= t <= (hi + margin)]
    if len(nice) >= 3:
        n_new  = len(nice) - 1
        colors = _interp_colors(colors, n_new)   # full palette → max contrast per panel
        levels = nice
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(levels, cmap.N)
    return levels, colors, cmap, norm


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

    PC   = ccrs.PlateCarree()
    PROJ = ccrs.LambertConformal(central_longitude=10, central_latitude=51)

    cmap_seq  = mcolors.ListedColormap(colors)
    norm_seq  = mcolors.BoundaryNorm(levels, cmap_seq.N)
    cmap_div  = mcolors.ListedColormap(bias_col)
    norm_div  = mcolors.BoundaryNorm(bias_lvl, cmap_div.N)

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(13.0, 5.2))
    gs  = GridSpec(2, 3, height_ratios=[1, 0.07],
                   left=0.04, right=0.98, top=0.92, bottom=0.04,
                   hspace=0.10, wspace=0.15)
    axes = [fig.add_subplot(gs[0, k], projection=PROJ) for k in range(3)]
    caxs = [fig.add_subplot(gs[1, k]) for k in range(3)]

    fig.patch.set_facecolor("white")
    if suptitle:
        fig.suptitle(suptitle, fontsize=10, fontweight="bold", y=0.99)

    panel_labels = ["(a)", "(b)", "(c)"]
    panel_titles = ["E-OBS", "ICON-CLM", "Bias (ICON − E-OBS)"]

    for k, (ax, cax, da, cmap, norm, lvls, title, plabel) in enumerate(zip(
            axes, caxs,
            [obs_clim, mod_clim, bias],
            [cmap_seq, cmap_seq, cmap_div],
            [norm_seq, norm_seq, norm_div],
            [levels,   levels,   bias_lvl],
            panel_titles, panel_labels,
    )):
        ax.set_extent(MAP_EXTENT, crs=PC)
        ax.set_facecolor("#d6e8f2")
        ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#ebebeb", zorder=1)
        ax.add_feature(cfeature.BORDERS.with_scale("10m"),
                       linewidth=0.3, edgecolor="0.45", zorder=2)

        fine = interp_display(da)
        mask = build_mask(fine["lon"].values, fine["lat"].values, geom)
        arr  = apply_mask(fine.values, mask)

        ax.contourf(
            fine["lon"].values, fine["lat"].values, arr,
            levels=lvls, cmap=cmap, norm=norm,
            transform=PC, extend="both", antialiased=True, zorder=3,
        )
        ax.add_geometries(gdf.geometry, PC, facecolor="none",
                          edgecolor="black", linewidth=0.55, zorder=6)
        ax.text(0.03, 0.97, plabel, transform=ax.transAxes,
                ha="left", va="top", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75))
        ax.set_title(title, fontsize=9.5, fontweight="bold", pad=4)
        style_axis(ax)

        fmt   = tick_fmt if k < 2 else "%.2f"
        ticks = [lvls[0], lvls[len(lvls)//2], lvls[-1]] if k == 2 else lvls
        cb    = ColorbarBase(cax, cmap=cmap, norm=norm, boundaries=lvls,
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
    """Apply cartographic formatting to a Cartopy GeoAxes (Germany domain)."""
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=True,
        linewidth=0.35, color="0.55", alpha=0.55, linestyle=":", zorder=0,
        xlocs=np.arange(6, 16, 3), ylocs=np.arange(48, 56, 2),
    )
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}
    for sp in ax.spines.values():
        sp.set_linewidth(0.65)


# ── paired trend map figure (E-OBS vs ICON-CLM) ───────────────────────────────
def plot_paired_trend_maps(
    obs_slope, model_slope,
    obs_pval,  model_pval,
    gdf, geom,
    outfile, levels, colors,
    cbar_label,
    title_obs="E-OBS", title_model="ICON-CLM",
    tick_fmt="%.2f",
    suptitle=None,
):
    """
    Publication-quality two-panel trend map: (a) E-OBS, (b) ICON-CLM.

    Each panel has its own slim vertical colorbar (same scale, all boundary
    ticks labeled).  Significance stippling and domain-mean annotated per panel.
    """
    from matplotlib.ticker import MaxNLocator
    from matplotlib.gridspec import GridSpec

    PC   = ccrs.PlateCarree()
    PROJ = ccrs.LambertConformal(central_longitude=10, central_latitude=51)

    # Template palette kept unchanged; each panel gets its own independent scale
    levels_base = list(levels)
    colors_base = list(colors)

    # GridSpec: map_a | gap | map_b  (colorbars via inset_axes)
    fig = plt.figure(figsize=(10.5, 5.5))
    fig.patch.set_facecolor("white")
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=0.99)

    gs = GridSpec(1, 3, width_ratios=[1, 0.08, 1],
                  left=0.03, right=0.93, top=0.91, bottom=0.04,
                  wspace=0.0)

    for i, (slope, pval, ds_title, panel_label) in enumerate([
        (obs_slope,   obs_pval,   title_obs,   "(a)"),
        (model_slope, model_pval, title_model, "(b)"),
    ]):
        # Independent auto-scale to this panel's own data range
        lvls, _, cmap, norm = _auto_scale_palette(slope.values, levels_base, colors_base)

        ax = fig.add_subplot(gs[0, 0 if i == 0 else 2], projection=PROJ)
        ax.set_extent(MAP_EXTENT, crs=PC)
        ax.set_facecolor("#d6e8f2")
        ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#ebebeb", zorder=1)
        ax.add_feature(cfeature.BORDERS.with_scale("10m"),
                       linewidth=0.3, edgecolor="0.45", zorder=2)

        fine    = interp_display(slope)
        de_mask = build_mask(fine["lon"].values, fine["lat"].values, geom)
        arr     = apply_mask(fine.values, de_mask)

        ax.contourf(
            fine["lon"].values, fine["lat"].values, arr,
            levels=lvls, cmap=cmap, norm=norm,
            transform=PC, extend="both", antialiased=True, zorder=3,
        )
        ax.add_geometries(gdf.geometry, PC, facecolor="none",
                          edgecolor="white",   linewidth=1.8, zorder=5)
        ax.add_geometries(gdf.geometry, PC, facecolor="none",
                          edgecolor="#1a1a1a", linewidth=0.70, zorder=6)

        de_mask_c  = build_mask(slope["lon"].values, slope["lat"].values, geom)
        sig_mask   = pval.values < ALPHA
        lo2d, la2d = np.meshgrid(slope["lon"].values, slope["lat"].values)
        stip_mask  = sig_mask & de_mask_c
        ax.scatter(lo2d[stip_mask], la2d[stip_mask],
                   s=2.0, c="#1a1a1a", alpha=0.40, marker=".", zorder=7,
                   rasterized=True, transform=PC)

        n_de     = int(de_mask_c.sum())
        sig_frac = stip_mask.sum() / n_de * 100 if n_de > 0 else 0.0

        ax.text(0.03, 0.97, panel_label, transform=ax.transAxes,
                ha="left", va="top", fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.22", fc="white",
                          ec="#888888", alpha=0.92, lw=0.6))
        ax.set_title(ds_title, fontsize=11, fontweight="bold", pad=6)

        ax.text(0.97, 0.03, f"Sig. area: {sig_frac:.0f}%",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7.5, color="#222222",
                bbox=dict(boxstyle="round,pad=0.20", fc="white",
                          ec="#aaaaaa", alpha=0.92, lw=0.5))

        mean_v = float(np.nanmean(slope.values[de_mask_c]))
        sign   = "+" if mean_v >= 0 else ""
        ax.text(0.03, 0.03, f"Mean: {sign}{mean_v:.3f}",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=7.5, color="#222222",
                bbox=dict(boxstyle="round,pad=0.20", fc="white",
                          ec="#aaaaaa", alpha=0.92, lw=0.5))

        style_axis(ax)

        # Slim vertical colorbar — all boundary ticks labeled, rectangular ends
        cax = ax.inset_axes([1.015, 0.0, 0.035, 1.0])
        cb  = ColorbarBase(cax, cmap=cmap, norm=norm, boundaries=levels,
                           ticks=_smart_ticks(levels), orientation="vertical", extend="neither")
        cb.ax.tick_params(labelsize=6, pad=2, length=3, width=0.5, direction="out")
        cb.ax.yaxis.set_major_formatter(FormatStrFormatter(tick_fmt))
        cb.outline.set_linewidth(0.5)
        cb.set_label(cbar_label, fontsize=8, labelpad=4, fontweight="normal")

    fig.savefig(outfile, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ── E-OBS trend + diff supplementary figure ───────────────────────────────────
def plot_obs_bias_maps(
    obs_slope, model_slope,
    obs_pval, model_pval,
    gdf, geom,
    outfile,
    obs_levels, obs_colors,
    cbar_label,
    title_obs="E-OBS", title_diff="Diff (ICON − E-OBS)",
    tick_fmt="%.2f",
    suptitle=None,
    obs_sequential=True,
    diff_colors=None, diff_levels=None,
):
    """
    Supplementary two-panel figure: (a) E-OBS trend with significance stippling,
    (b) ICON − E-OBS arithmetic difference (no significance).

    Both panels use the same base color palette (obs_colors), each independently
    auto-scaled to its own data range.
    """
    from matplotlib.ticker import MaxNLocator

    PC   = ccrs.PlateCarree()
    PROJ = ccrs.LambertConformal(central_longitude=10, central_latitude=51)

    obs_colors_orig  = list(obs_colors)
    diff_colors_orig = list(diff_colors) if diff_colors is not None else obs_colors_orig
    diff_levels_base = list(diff_levels) if diff_levels is not None else obs_levels

    def _auto_scale(data_arr, base_colors, base_levels, obs_panel=False):
        """Auto-scale levels and resample base_colors to data range.

        obs_panel=True forces sequential (never diverging) so the E-OBS
        panel always shows only the dominant-sign half of the palette.
        """
        valid       = data_arr[np.isfinite(data_arr)]
        base_is_div = min(base_levels) < 0 < max(base_levels)
        if len(valid) <= 10:
            cmap = mcolors.ListedColormap(base_colors)
            norm = mcolors.BoundaryNorm(base_levels, cmap.N)
            return list(base_levels), list(base_colors), cmap, norm
        p2, p98  = np.percentile(valid, 2), np.percentile(valid, 98)
        med_val  = float(np.median(valid))
        # Diff panel: diverging whenever the base palette is diverging (obs_panel never diverges)
        is_div   = (not obs_panel) and base_is_div
        n_target = 14 if is_div else 12
        if is_div:
            ext    = max(abs(p2), abs(p98))
            lo, hi = -ext, ext
        else:
            lo = np.percentile(valid, 5)
            hi = np.percentile(valid, 95)
            # Negative-dominant: extend one step above 0 so colorbar doesn't stop abruptly
            if base_is_div and med_val < 0:
                rough_mag = 10.0 ** np.floor(np.log10(abs(lo) / n_target))
                step0     = min([f * rough_mag for f in [1, 2, 5, 10]],
                                key=lambda s: abs(abs(lo) / s - n_target))
                hi = step0
        span     = hi - lo
        raw_step = span / n_target
        mag      = 10.0 ** np.floor(np.log10(raw_step))
        step     = min([f * mag for f in [1, 2, 5, 10]],
                       key=lambda s: abs(span / s - n_target))
        if is_div:
            n_half = max(1, round(hi / step))
            if n_half % 2 != 0:
                n_half += 1          # force even → zero at even index → always labeled
            nice   = [round(i * step, 10) for i in range(-n_half, n_half + 1)]
        else:
            start  = np.floor(lo / step) * step
            nice   = np.round(np.arange(start, hi + step * 0.01, step), 10).tolist()
            if lo > 0:
                nice = [t for t in nice if t > 0]
        margin = span * 0.08
        nice   = [float(t) for t in nice if (lo - margin) <= t <= (hi + margin)]
        if len(nice) >= 3:
            n_new  = len(nice) - 1
            if base_is_div and not is_div:
                center = len(base_colors) // 2
                sub    = base_colors[:center + 1] if med_val < 0 else base_colors[center:]
                colors = _interp_colors(sub, n_new)
            else:
                colors = _interp_colors(base_colors, n_new)
            levels = nice
        else:
            colors = list(base_colors)
            levels = list(base_levels)
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(levels, cmap.N)
        return levels, colors, cmap, norm

    diff = model_slope - obs_slope

    # E-OBS: sequential when obs_sequential=True (temperature); full diverging otherwise (precipitation)
    # Diff: uses diff_colors/diff_levels when provided (e.g. TEMP_COLORS for precipitation Diff)
    obs_lvls,  _, cmap_obs,  norm_obs  = _auto_scale(
        obs_slope.values, obs_colors_orig, obs_levels, obs_panel=obs_sequential)
    diff_lvls, _, cmap_diff, norm_diff = _auto_scale(
        diff.values, diff_colors_orig, diff_levels_base)

    # ── Figure layout ─────────────────────────────────────────────────────────
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(10.5, 5.2))
    fig.patch.set_facecolor("white")
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=0.99)

    gs  = GridSpec(1, 3, width_ratios=[1, 0.08, 1],
                   left=0.03, right=0.93, top=0.91, bottom=0.04,
                   wspace=0.0)
    axs = [fig.add_subplot(gs[0, 0], projection=PROJ),
           fig.add_subplot(gs[0, 2], projection=PROJ)]

    panels = [
        dict(ax=axs[0], da=obs_slope,
             cmap=cmap_obs,  norm=norm_obs,  lvls=obs_lvls,
             title=title_obs,  cbar_lbl=cbar_label,
             stipple=True,  pval=obs_pval, tag="(a)"),
        dict(ax=axs[1], da=diff,
             cmap=cmap_diff, norm=norm_diff, lvls=diff_lvls,
             title=title_diff, cbar_lbl=cbar_label,   # no "Diff" prefix — title already says it
             stipple=False, pval=None,     tag="(b)"),
    ]

    for p in panels:
        ax  = p["ax"]
        da  = p["da"]
        cmap, norm, lvls = p["cmap"], p["norm"], p["lvls"]

        ax.set_extent(MAP_EXTENT, crs=PC)
        ax.set_facecolor("#d6e8f2")
        ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#ebebeb", zorder=1)
        ax.add_feature(cfeature.BORDERS.with_scale("10m"),
                       linewidth=0.3, edgecolor="0.45", zorder=2)

        fine    = interp_display(da)
        de_mask = build_mask(fine["lon"].values, fine["lat"].values, geom)
        arr     = apply_mask(fine.values, de_mask)

        ax.contourf(
            fine["lon"].values, fine["lat"].values, arr,
            levels=lvls, cmap=cmap, norm=norm,
            transform=PC, extend="both", antialiased=True, zorder=3,
        )
        ax.add_geometries(gdf.geometry, PC, facecolor="none",
                          edgecolor="white",   linewidth=1.8, zorder=5)
        ax.add_geometries(gdf.geometry, PC, facecolor="none",
                          edgecolor="#1a1a1a", linewidth=0.70, zorder=6)

        de_mask_c = build_mask(da["lon"].values, da["lat"].values, geom)

        # Significance stippling — E-OBS panel only
        if p["stipple"]:
            sig_mask  = p["pval"].values < ALPHA
            stip_mask = sig_mask & de_mask_c
            lo2d, la2d = np.meshgrid(da["lon"].values, da["lat"].values)
            ax.scatter(lo2d[stip_mask], la2d[stip_mask],
                       s=2.0, c="#1a1a1a", alpha=0.40, marker=".", zorder=7,
                       rasterized=True, transform=PC)
            n_de     = int(de_mask_c.sum())
            sig_frac = stip_mask.sum() / n_de * 100 if n_de > 0 else 0.0
            ax.text(0.97, 0.03, f"Sig. area: {sig_frac:.0f}%",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7.5, color="#222222",
                    bbox=dict(boxstyle="round,pad=0.20", fc="white",
                              ec="#aaaaaa", alpha=0.92, lw=0.5))

        ax.text(0.03, 0.97, p["tag"], transform=ax.transAxes,
                ha="left", va="top", fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.22", fc="white",
                          ec="#888888", alpha=0.92, lw=0.6))
        ax.set_title(p["title"], fontsize=11, fontweight="bold", pad=6)

        mean_v = float(np.nanmean(da.values[de_mask_c]))
        sign   = "+" if mean_v >= 0 else ""
        ax.text(0.03, 0.03, f"Mean: {sign}{mean_v:.3f}",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=7.5, color="#222222",
                bbox=dict(boxstyle="round,pad=0.20", fc="white",
                          ec="#aaaaaa", alpha=0.92, lw=0.5))

        style_axis(ax)

        # Slim vertical colorbar — rectangular ends, all boundary ticks labeled
        cax = ax.inset_axes([1.015, 0.0, 0.035, 1.0])

        cb = ColorbarBase(cax, cmap=cmap, norm=norm, boundaries=lvls,
                          ticks=_smart_ticks(lvls), orientation="vertical", extend="neither")
        cb.ax.tick_params(labelsize=6, pad=2, length=3, width=0.5, direction="out")
        cb.ax.yaxis.set_major_formatter(FormatStrFormatter(tick_fmt))
        cb.outline.set_linewidth(0.5)
        cb.set_label(p["cbar_lbl"], fontsize=8, labelpad=4, fontweight="normal")

    fig.savefig(outfile, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_germany_series(
    obs_series, model_series,
    obs_anom,   model_anom,
    ylabel, title, outfile,
    ylabel_anom=None,
    obs_stats=None, model_stats=None,
):
    """
    Two-panel stacked figure for Germany-average values.

    Top panel (a)    — annual scatter + 5-yr dashed + 10-yr solid for both datasets.
    Bottom panel (b) — E-OBS sign-coloured bars + ICON-CLM raw annual line (dark grey).
    """
    from matplotlib.lines import Line2D

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7.0), sharex=True,
        gridspec_kw={"hspace": 0.05, "height_ratios": [1.45, 1]},
    )
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=11, fontweight="bold", y=0.98)

    OBS_COL = "#1b7837"    # E-OBS: dark green
    MOD_COL = "#b2182b"    # ICON-CLM: dark red

    years    = obs_series["year"].values.astype(int)
    obs_vals = obs_series.values.astype(float)
    mod_vals = model_series.values.astype(float)

    # ── Top panel: absolute annual values ────────────────────────────────────
    for vals, col, lbl in [
        (obs_vals, OBS_COL, "E-OBS"),
        (mod_vals, MOD_COL, "ICON-CLM"),
    ]:
        s = pd.Series(vals, index=years, dtype=float)
        ax1.scatter(years, vals, color=col, s=9, alpha=0.30, zorder=2, linewidths=0)
        rm5  = s.rolling(5,  center=True, min_periods=3).mean()
        ax1.plot(years, rm5.values,  "--", color=col, lw=1.5, alpha=0.80, zorder=3)
        rm10 = s.rolling(10, center=True, min_periods=5).mean()
        ax1.plot(years, rm10.values, "-",  color=col, lw=2.2, alpha=0.92,
                 label=lbl, zorder=4)

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
    ax1.set_ylabel(ylabel, fontsize=9)
    ax1.tick_params(labelsize=8)
    ax1.grid(True, linestyle="--", linewidth=0.35, alpha=0.55, zorder=0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines[["left", "bottom"]].set_linewidth(0.7)
    ax1.text(0.005, 0.98, "(a)", transform=ax1.transAxes,
             fontsize=10, fontweight="bold", va="top")

    # ── Bottom panel: anomalies ───────────────────────────────────────────────
    anom_yrs = obs_anom["year"].values.astype(int)
    obs_a    = obs_anom.values.astype(float)
    mod_a    = model_anom.values.astype(float)

    ax2.axvspan(1991, 2020, color="#f0f0f0", zorder=0)
    ax2.text(2005.5, 0, "1991–2020\nreference", ha="center", va="bottom",
             fontsize=6.5, color="0.55", style="italic", zorder=1)
    ax2.axhline(0, color="0.35", lw=0.80, zorder=1)

    # E-OBS sign-coloured bars
    bar_w   = 0.42
    obs_pos = np.where(obs_a >= 0, obs_a, 0.0)
    obs_neg = np.where(obs_a < 0,  obs_a, 0.0)
    ax2.bar(anom_yrs - bar_w / 2, obs_pos, bar_w,
            color="#d73027", alpha=0.78, label="E-OBS +anom", zorder=2)
    ax2.bar(anom_yrs - bar_w / 2, obs_neg, bar_w,
            color="#4575b4", alpha=0.78, label="E-OBS −anom", zorder=2)

    # ICON-CLM: dark grey step plot (rectangular appearance)
    ax2.plot(anom_yrs, mod_a, color="0.25", lw=1.0,
             drawstyle="steps-mid", label="ICON-CLM", zorder=4)

    # E-OBS 10-yr running mean (bold green)
    s_obs_a  = pd.Series(obs_a, index=anom_yrs, dtype=float)
    rm10_obs = s_obs_a.rolling(10, center=True, min_periods=5).mean()
    ax2.plot(anom_yrs, rm10_obs.values, "-", color=OBS_COL, lw=2.0,
             label="E-OBS 10-yr", zorder=5, alpha=0.90)

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
    PC   = ccrs.PlateCarree()
    PROJ = ccrs.LambertConformal(central_longitude=10, central_latitude=51)

    n    = len(index_meta)
    fig, axes = plt.subplots(n, 2, figsize=(8.8, n * 2.8),
                              subplot_kw={"projection": PROJ})
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "JJA Precipitation Extreme Indices — Theil-Sen Trends 1950–2022\n"
        "Stippling: statistically significant trend (MK p < 0.05, Yue-Wang correction)",
        fontsize=9.5, fontweight="bold", y=0.998,
    )

    letters    = "abcdefghijklmnopqrstuvwxyz"
    letter_idx = 0

    for row, (name, d) in enumerate(index_meta.items()):
        cmap = mcolors.ListedColormap(d["colors"])
        norm = mcolors.BoundaryNorm(d["levels"], cmap.N)

        for col, (slope, pval, ds_title) in enumerate([
            (d["obs_slope"], d["obs_pval"], "E-OBS"),
            (d["mod_slope"], d["mod_pval"], "ICON-CLM"),
        ]):
            ax = axes[row, col] if n > 1 else axes[col]
            ax.set_extent(MAP_EXTENT, crs=PC)
            ax.set_facecolor("#d6e8f2")
            ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#ebebeb", zorder=1)
            ax.add_feature(cfeature.BORDERS.with_scale("10m"),
                           linewidth=0.3, edgecolor="0.45", zorder=2)

            fine = interp_display(slope)
            mask = build_mask(fine["lon"].values, fine["lat"].values, geom)
            arr  = apply_mask(fine.values, mask)

            ax.contourf(
                fine["lon"].values, fine["lat"].values, arr,
                levels=d["levels"], cmap=cmap, norm=norm,
                transform=PC, extend="both", antialiased=True, zorder=3,
            )
            ax.add_geometries(gdf.geometry, PC, facecolor="none",
                              edgecolor="black", linewidth=0.50, zorder=5)

            # Stippling — Germany cells only
            de_mask_c  = build_mask(slope["lon"].values, slope["lat"].values, geom)
            sig_mask   = pval.values < ALPHA
            lo2d, la2d = np.meshgrid(slope["lon"].values, slope["lat"].values)
            stip_mask  = sig_mask & de_mask_c
            ax.scatter(lo2d[stip_mask], la2d[stip_mask],
                       s=0.35, c="#111111", alpha=0.26, zorder=6,
                       rasterized=True, transform=PC)

            ax.text(0.03, 0.97, f"({letters[letter_idx]})",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white",
                              ec="none", alpha=0.72))
            letter_idx += 1

            if row == 0:
                ax.set_title(ds_title, fontsize=9, fontweight="bold", pad=3)

            if col == 0:
                ax.text(-0.12, 0.5, d["long_name"], transform=ax.transAxes,
                        fontsize=7.5, va="center", ha="right", rotation=90)

            style_axis(ax)

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
        left=0.12, right=0.88, top=0.97, bottom=0.01,
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
