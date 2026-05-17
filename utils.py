"""
utils.py
--------
Shared utilities for all three analysis scripts.
All functions are imported into the main scripts; nothing runs on import.
"""

import os
import numpy as np
import xarray as xr
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
START_YEAR   = "1950"
END_YEAR     = "2022"
REF_START    = "1961"   # threshold / percentile reference period
REF_END      = "1990"
ANOM_START   = "1991"   # anomaly reference period  (WMO 1991-2020)
ANOM_END     = "2020"
ALPHA        = 0.05
MIN_VALID    = 10
DPI          = 600
MAP_EXTENT   = [5.8, 15.2, 47.4, 55.1]
DISPLAY_FACTOR = 5

# ── data loading ───────────────────────────────────────────────────────────────
def load_field(path, varname, start=START_YEAR, end=END_YEAR):
    """Load a variable from a NetCDF file; harmonize lat/lon names."""
    ds = xr.open_dataset(path)
    rename = {}
    if "latitude"  in ds.coords: rename["latitude"]  = "lat"
    if "longitude" in ds.coords: rename["longitude"] = "lon"
    if rename:
        ds = ds.rename(rename)
    if varname not in ds:
        raise KeyError(f"'{varname}' not in {path}. Available: {list(ds.data_vars)}")
    da = ds[varname].sel(time=slice(start, end))
    return da.sortby("lat").sortby("lon")


def keep_jja(da):
    """Subset a daily DataArray to JJA days only."""
    return da.where(da["time"].dt.month.isin([6, 7, 8]), drop=True)


# ── climatology and anomalies ─────────────────────────────────────────────────
def annual_jja_mean(da):
    """Annual JJA mean (temperature or any continuous variable)."""
    return da.groupby("time.year").mean("time").astype(np.float32)


def annual_jja_sum(da):
    """Annual JJA sum (precipitation total)."""
    return da.groupby("time.year").sum("time").astype(np.float32)


def reference_mean(annual_da, start=ANOM_START, end=ANOM_END):
    """Mean over an arbitrary reference period (default 1991-2020)."""
    return annual_da.sel(year=slice(int(start), int(end))).mean("year", skipna=True)


def compute_anomalies(annual_da, climatology):
    """Subtract a climatology from an annual DataArray."""
    return annual_da - climatology


def area_mean(da):
    """Spatial mean over all valid Germany grid cells."""
    return da.mean(dim=("lat", "lon"), skipna=True)


def rmse(a, b):
    """RMSE between two 1-D arrays; NaN-safe."""
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((a[valid] - b[valid]) ** 2)))


# ── trend estimation ──────────────────────────────────────────────────────────
def compute_trend_maps(annual_da):
    """
    Theil-Sen slope (per decade) and Mann-Kendall p-value at every grid cell.
    Uses mk.yue_wang_modification_test which corrects for lag-1 autocorrelation
    (equivalent to trend-free pre-whitening; Yue et al. 2002).
    Falls back to original_test if the correction raises an error.
    Requires MIN_VALID valid years; cells with fewer are left as NaN.
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
            yy = series[valid]
            xx = years[valid]

            # Theil-Sen slope → per decade
            slope, _, _, _ = theilslopes(yy, xx, 0.95)
            slope_map[i, j] = slope * 10.0

            # Mann-Kendall with autocorrelation correction (TFPW)
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
    """Trend statistics for a Germany-average annual time series."""
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
    return {"sen_slope_decade": float(slope * 10.0),
            "mk_p": float(res.p),
            "mk_z": float(res.z)}


# ── Germany shapefile and masking ─────────────────────────────────────────────
def load_country_shape(shp_path):
    gdf  = gpd.read_file(shp_path).to_crs("EPSG:4326")
    geom = unary_union(gdf.geometry)
    return gdf, geom


def _polygon_to_path(geom):
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
        raise ValueError(f"Unsupported geometry: {type(geom)}")
    return Path(np.asarray(vertices, float), np.asarray(codes))


def build_mask(lon, lat, geom):
    """Boolean mask: True where grid cell centre is inside the Germany polygon."""
    path   = _polygon_to_path(geom)
    lo, la = np.meshgrid(lon, lat)
    pts    = np.column_stack([lo.ravel(), la.ravel()])
    return path.contains_points(pts).reshape(len(lat), len(lon))


def apply_mask(arr2d, mask2d):
    out = np.array(arr2d, dtype=float)
    out[~mask2d] = np.nan
    return out


def clip_contourf(cf_obj, ax, geom):
    clip = PathPatch(_polygon_to_path(geom), transform=ax.transData, facecolor="none")
    for coll in cf_obj.collections:
        coll.set_clip_path(clip)


def interp_display(da2d, factor=DISPLAY_FACTOR):
    """
    Bilinear upsampling for display smoothness only.
    Does NOT affect any computed values.
    """
    lo = da2d["lon"].values
    la = da2d["lat"].values
    lo_fine = np.linspace(lo.min(), lo.max(), len(lo) * factor)
    la_fine = np.linspace(la.min(), la.max(), len(la) * factor)
    return da2d.interp(lon=lo_fine, lat=la_fine, method="linear")


# ── map axis styling ──────────────────────────────────────────────────────────
def style_axis(ax):
    ax.set_facecolor("#d9d9d9")
    ax.set_xlim(MAP_EXTENT[0], MAP_EXTENT[1])
    ax.set_ylim(MAP_EXTENT[2], MAP_EXTENT[3])
    ax.set_box_aspect(1)
    ax.set_xticks(np.arange(6, 16, 2))
    ax.set_yticks(np.arange(48, 56, 2))
    ax.tick_params(axis="both", which="both", direction="out",
                   top=False, right=False, labelsize=6, pad=1)
    ax.grid(True, linestyle="--", linewidth=0.40, color="0.55", alpha=0.65)
    for sp in ax.spines.values():
        sp.set_linewidth(0.50)


# ── paired trend map (reused by scripts 1 and 2) ─────────────────────────────
def plot_paired_trend_maps(
    obs_slope, model_slope,
    obs_pval,  model_pval,
    gdf, geom,
    outfile, levels, colors,
    cbar_label,
    title_obs="E-OBS", title_model="ICON-CLM",
    tick_fmt="%.1f",
):
    """
    Two-panel trend map: (a) obs, (b) model.
    Shared horizontal colorbar. Significance stippling on each panel.
    """
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(levels, cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.65))
    fig.patch.set_facecolor("white")

    for ax, slope, pval, title, label in [
        (axes[0], obs_slope,   obs_pval,   title_obs,   "(a)"),
        (axes[1], model_slope, model_pval, title_model, "(b)"),
    ]:
        fine = interp_display(slope)
        mask = build_mask(fine["lon"].values, fine["lat"].values, geom)
        arr  = apply_mask(fine.values, mask)

        cf = ax.contourf(fine["lon"].values, fine["lat"].values, arr,
                         levels=levels, cmap=cmap, norm=norm,
                         extend="both", antialiased=True)
        clip_contourf(cf, ax, geom)
        gdf.boundary.plot(ax=ax, color="black", linewidth=0.45, zorder=5)

        # Stippling on original (non-interpolated) grid
        lo2d, la2d = np.meshgrid(slope["lon"].values, slope["lat"].values)
        sig = pval.values < ALPHA
        ax.scatter(lo2d[sig], la2d[sig], s=0.40, c="0.10", alpha=0.22, zorder=6)

        ax.text(0.03, 0.97, label, transform=ax.transAxes,
                ha="left", va="top", fontsize=8, fontweight="bold")
        ax.set_title(title, fontsize=8, pad=3)
        style_axis(ax)

    plt.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.22, wspace=0.16)

    cax = fig.add_axes([0.18, 0.095, 0.64, 0.045])
    cb  = ColorbarBase(cax, cmap=cmap, norm=norm, boundaries=levels,
                       ticks=levels, orientation="horizontal", extend="both")
    cb.ax.tick_params(labelsize=6, pad=1)
    cb.ax.xaxis.set_major_formatter(FormatStrFormatter(tick_fmt))
    cb.set_label(cbar_label, fontsize=7, labelpad=2)

    fig.savefig(outfile, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ── Germany-average time series plot ─────────────────────────────────────────
def plot_germany_series(
    obs_series, model_series,
    obs_anom,   model_anom,
    ylabel, title, outfile,
    ylabel_anom=None,
):
    """
    Two-panel figure:
    Left  — annual Germany-average absolute values (obs vs model)
    Right — anomalies relative to 1991-2020
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    fig.patch.set_facecolor("white")

    years_abs  = obs_series["year"].values
    years_anom = obs_anom["year"].values

    # Absolute
    axes[0].plot(years_abs, obs_series.values,   color="#1b7837", lw=1.2, label="E-OBS")
    axes[0].plot(years_abs, model_series.values, color="#b2182b", lw=1.2, label="ICON-CLM", alpha=0.85)
    axes[0].axhline(0, color="0.6", lw=0.5, ls="--")
    axes[0].set_xlabel("Year", fontsize=8)
    axes[0].set_ylabel(ylabel, fontsize=8)
    axes[0].set_title(f"{title} — absolute", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False)
    axes[0].tick_params(labelsize=7)
    axes[0].grid(True, linestyle="--", linewidth=0.35, alpha=0.5)

    # Anomaly
    axes[1].bar(years_anom, obs_anom.values,
                color=["#b2182b" if v > 0 else "#2166ac" for v in obs_anom.values],
                width=0.4, align="edge", label="E-OBS", alpha=0.75)
    axes[1].plot(years_anom, model_anom.values,
                 color="0.25", lw=1.0, label="ICON-CLM", zorder=5)
    axes[1].axhline(0, color="0.4", lw=0.7)
    axes[1].set_xlabel("Year", fontsize=8)
    axes[1].set_ylabel(ylabel_anom or ylabel, fontsize=8)
    axes[1].set_title(f"{title} — anomaly vs 1991-2020", fontsize=8)
    axes[1].legend(fontsize=7, frameon=False)
    axes[1].tick_params(labelsize=7)
    axes[1].grid(True, linestyle="--", linewidth=0.35, alpha=0.5)

    plt.tight_layout()
    fig.savefig(outfile, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
