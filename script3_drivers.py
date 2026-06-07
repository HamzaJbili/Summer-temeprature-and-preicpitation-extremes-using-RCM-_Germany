"""
script3_drivers.py
------------------
Drivers of German summer extremes — correlation + composite analysis.

Standard methodology (Hirschi et al. 2011; Mueller & Seneviratne 2012;
Loikith et al. 2015; Whan et al. 2015):

  1. CORRELATION HEATMAPS  — quantify HOW STRONG every driver link is.
     Two grouped heatmaps (Pearson r + significance stars):
       Temperature   T90p, HWN, HWD  ×  PSL, SHF, LHF, CLT     (3×4)
       Precipitation SDII, CDD, SPI  ×  PSL, LHF, CLT, CAPE, CIN (3×5)

  2. COMPOSITE ANOMALY MAPS — show WHERE in Germany the driver signal
     sits during the most extreme summers.  Upper-tercile years of EACH
     index are selected; the anomaly field of each relevant driver is
     composited and shown as a single multi-panel map figure per index
     (NOT one map per pair).
       Temperature   one 4-panel figure per index → 3 figures (T90p, HWN, HWD)
       Precipitation one 5-panel figure per index → 3 figures (SDII, CDD, SPI)

Output: 8 figures + 1 CSV. Nothing else.

Interpretation note
~~~~~~~~~~~~~~~~~~~~
  Both indices and drivers are from the SAME ICON-CLM ERA5-driven hindcast.
  This is a model-internal physical-consistency diagnosis: does the model
  produce its extremes through physically consistent relationships?
  Large-scale dynamics (PSL) are constrained by ERA5 lateral boundaries
  (Kotlarski et al. 2014; Vautard et al. 2021).  For precipitation indices
  the links reflect model internal physics and are interpreted as such
  (Loikith et al. 2015).

References
~~~~~~~~~~~
  Hirschi et al. (2011)  Nat. Geosci. 4, 17-21
  Mueller & Seneviratne (2012)  PNAS 109, 12398
  Seneviratne et al. (2010)  Earth-Sci. Rev. 99, 125
  Loikith et al. (2015)  Clim. Dyn. 45, 3257
  Whan et al. (2015)  Weather Clim. Extremes 9, 57
  Kotlarski et al. (2014)  Geosci. Model Dev. 7, 1297
  Vautard et al. (2021)  JGR-Atmos 126, e2019JD032344
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
    load_field, area_mean,
    reference_mean, compute_anomalies,
    load_country_shape, interp_display, build_mask, apply_mask,
    style_axis,
    START_YEAR, END_YEAR, REF_START, REF_END, DPI,
)
from utils import set_ipcc_style
set_ipcc_style()

# ── output directories ─────────────────────────────────────────────────────────
FIGDIR = os.path.join("output_drivers", "figures")
TABDIR = os.path.join("output_drivers", "tables")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(TABDIR, exist_ok=True)

GERMANY_SHP  = "/work/jbiliham/shapefile_Germany/gadm41_DEU_0.shp"
INDEX_NC_DIR = "/archive1/hamza_data/DE_files/DE_1950-2022/extremes_indices/output_extremes/netcdf"

# ── index groups ───────────────────────────────────────────────────────────────
# (display_name, nc_stem, row_label)
TEMP_INDICES = [
    ("T90p_exceedance_days", "T90p_days", "T90p"),
    ("Heatwave_number",      "HWN",       "HWN"),
    ("Heatwave_duration",    "HWD",       "HWD"),
]
PRECIP_INDICES = [
    ("SDII", "SDII", "SDII"),
    ("CDD",  "CDD",  "CDD"),
    ("SPI",  "SPI",  "SPI"),
]

TEMP_DRIVERS   = ["PSL", "SHF", "LHF", "CLT"]
PRECIP_DRIVERS = ["PSL", "LHF", "CLT", "CAPE", "CIN"]

# ── driver file config ─────────────────────────────────────────────────────────
_SUFFIX = "DE-0.25_JJA_1950-2022.nc"
DRIVER_FILES = {
    "PSL":  f"psl_{_SUFFIX}",   "SHF": f"hfss_{_SUFFIX}",
    "LHF":  f"hfls_{_SUFFIX}",  "CLT": f"clt_{_SUFFIX}",
    "CAPE": f"cape_{_SUFFIX}",  "CIN": f"cin_{_SUFFIX}",
}
DRIVER_VARS = {
    "PSL": "psl",  "SHF": "hfss", "LHF": "hfls",
    "CLT": "clt",  "CAPE": "cape", "CIN": "cin",
}
DRIVER_SCALE = {"PSL": 0.01}   # Pa → hPa
DRIVER_LONG = {
    "PSL":  "Sea-level pressure",      "SHF":  "Sensible heat flux",
    "LHF":  "Latent heat flux",        "CLT":  "Total cloud cover",
    "CAPE": "Convective available PE", "CIN":  "Convective inhibition",
}
DRIVER_UNITS = {
    "PSL":  "hPa",            "SHF":  "W m$^{-2}$",
    "LHF":  "W m$^{-2}$",    "CLT":  "%",
    "CAPE": "J kg$^{-1}$",   "CIN":  "J kg$^{-1}$",
}
# Symmetric anomaly contour levels per driver
DRIVER_LEVELS = {
    "PSL":  [-6,  -4,  -2,  -1, -0.5, 0, 0.5,  1,   2,   4,   6],
    "SHF":  [-30, -20, -10,  -5,  -2, 0,  2,   5,  10,  20,  30],
    "LHF":  [-30, -20, -10,  -5,  -2, 0,  2,   5,  10,  20,  30],
    "CLT":  [-15, -10,  -7,  -5,  -2, 0,  2,   5,   7,  10,  15],
    "CAPE": [-300,-200,-100, -50, -20, 0, 20,  50, 100, 200, 300],
    "CIN":  [-40, -30, -20, -10,  -5, 0,  5,  10,  20,  30,  40],
}

# Palettes
CORR_COLORS = ["#2166ac","#4393c3","#92c5de","#d1e5f0","#f7f7f7",
               "#fddbc7","#f4a582","#d6604d","#b2182b"]
DIV_COLORS  = ["#2166ac","#4393c3","#92c5de","#d1e5f0","#f7f7f7",
               "#fddbc7","#f4a582","#d6604d","#b2182b","#67001f"]

MAP_EXTENT = [5.8, 15.2, 47.4, 55.1]
PROJ = ccrs.LambertConformal(central_longitude=10, central_latitude=51)
PC   = ccrs.PlateCarree()


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════
def load_index_series(nc_stem, dataset_label="ICON"):
    """Germany-average annual index series (for correlation)."""
    path = os.path.join(INDEX_NC_DIR, f"{nc_stem}_{dataset_label}_annual.nc")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}\nRun script2 first.")
    ds = xr.open_dataset(path)
    da = ds[list(ds.data_vars)[0]].sortby("lat").sortby("lon")
    return area_mean(da)


def load_index_field(nc_stem, dataset_label="ICON"):
    """Full 2-D annual index field (lat×lon×year) for composite selection."""
    path = os.path.join(INDEX_NC_DIR, f"{nc_stem}_{dataset_label}_annual.nc")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}\nRun script2 first.")
    ds = xr.open_dataset(path)
    return ds[list(ds.data_vars)[0]].sortby("lat").sortby("lon")


def load_driver_series(dname):
    """Germany-average annual JJA-mean driver series (for correlation)."""
    da = load_field(DRIVER_FILES[dname], DRIVER_VARS[dname])
    if DRIVER_SCALE.get(dname, 1.0) != 1.0:
        da = da * DRIVER_SCALE[dname]
    da = da.assign_coords(year=("time", da["time"].dt.year.values))
    da = da.swap_dims({"time": "year"}).drop_vars("time")
    return area_mean(da.sel(year=slice(int(START_YEAR), int(END_YEAR))))


def load_driver_anom_field(dname):
    """Full 2-D driver anomaly field (vs 1961-1990) for composite maps."""
    da = load_field(DRIVER_FILES[dname], DRIVER_VARS[dname])
    if DRIVER_SCALE.get(dname, 1.0) != 1.0:
        da = da * DRIVER_SCALE[dname]
    da = da.assign_coords(year=("time", da["time"].dt.year.values))
    da = da.swap_dims({"time": "year"}).drop_vars("time")
    da = da.sel(year=slice(int(START_YEAR), int(END_YEAR)))
    clim = reference_mean(da, REF_START, REF_END)
    return compute_anomalies(da, clim)


# ══════════════════════════════════════════════════════════════════════════════
# Part 1 — Correlation heatmaps
# ══════════════════════════════════════════════════════════════════════════════
def _stars(p):
    if not np.isfinite(p):
        return ""
    return "**" if p < 0.01 else ("*" if p < 0.05 else "")


def correlate(s_idx, s_drv):
    idx = pd.DataFrame({"year": s_idx["year"].values.astype(int),
                        "v": s_idx.values}).dropna()
    drv = pd.DataFrame({"year": s_drv["year"].values.astype(int),
                        "v": s_drv.values}).dropna()
    m = idx.merge(drv, on="year")
    if len(m) < 10:
        return dict(n=len(m), pearson_r=np.nan, pearson_p=np.nan,
                    spearman_r=np.nan, spearman_p=np.nan)
    pr, pp = pearsonr(m["v_x"], m["v_y"])
    sr, sp = spearmanr(m["v_x"], m["v_y"])
    return dict(n=len(m), pearson_r=float(pr), pearson_p=float(pp),
                spearman_r=float(sr), spearman_p=float(sp))


def plot_heatmap(r_mat, p_mat, row_labels, col_labels, outfile, title):
    n_rows, n_cols = r_mat.shape
    cmap = mcolors.LinearSegmentedColormap.from_list("corr", CORR_COLORS)
    norm = mcolors.Normalize(-1.0, 1.0)
    fig, ax = plt.subplots(figsize=(0.95 * n_cols + 2.2, 0.75 * n_rows + 1.8))
    fig.patch.set_facecolor("white")
    im = ax.imshow(r_mat, cmap=cmap, norm=norm, aspect="auto")
    for i in range(n_rows):
        for j in range(n_cols):
            r = r_mat[i, j]
            if not np.isfinite(r):
                continue
            c = "white" if abs(r) > 0.55 else "#1a1a1a"
            ax.text(j, i, f"{r:+.2f}{_stars(p_mat[i, j])}",
                    ha="center", va="center", fontsize=9, color=c)
    ax.set_xticks(range(n_cols)); ax.set_yticks(range(n_rows))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticklabels(row_labels, fontsize=10, fontweight="bold")
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title + "\n(* p<0.05,  ** p<0.01)", fontsize=10, pad=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Pearson r", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    plt.tight_layout()
    fig.savefig(outfile, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outfile}")


def run_correlation_group(group_name, indices, drivers, driver_series, rows_out):
    avail, rP, pP = [], [], []
    for display_name, nc_stem, label in indices:
        try:
            s = load_index_series(nc_stem, "ICON")
        except FileNotFoundError as e:
            print(f"  Skipping {display_name}: {e}")
            continue
        avail.append(label)
        rowP, rowPp = [], []
        for d in drivers:
            res = correlate(s, driver_series[d])
            rowP.append(res["pearson_r"]); rowPp.append(res["pearson_p"])
            rows_out.append({
                "group": group_name, "index": label, "driver": d,
                "n": res["n"],
                "pearson_r":  round(res["pearson_r"],  3) if np.isfinite(res["pearson_r"])  else np.nan,
                "pearson_p":  round(res["pearson_p"],  4) if np.isfinite(res["pearson_p"])  else np.nan,
                "spearman_r": round(res["spearman_r"], 3) if np.isfinite(res["spearman_r"]) else np.nan,
                "spearman_p": round(res["spearman_p"], 4) if np.isfinite(res["spearman_p"]) else np.nan,
            })
            print(f"  {label:5s} × {d:5s}: r={res['pearson_r']:+.2f} (p={res['pearson_p']:.3f})")
        rP.append(rowP); pP.append(rowPp)

    if not avail:
        print(f"  [{group_name}] no index files found — skipping heatmap.")
        return

    plot_heatmap(
        np.array(rP), np.array(pP), avail, drivers,
        outfile=os.path.join(FIGDIR, f"heatmap_{group_name.lower()}.png"),
        title=f"{group_name} extremes — driver correlation (Pearson r)\n"
              "ICON-CLM  ·  Germany average  ·  JJA 1950–2022")


# ══════════════════════════════════════════════════════════════════════════════
# Part 2 — Composite anomaly maps
# ══════════════════════════════════════════════════════════════════════════════
def top_tercile_years(index_field):
    s = area_mean(index_field)
    df = pd.DataFrame({"year": s["year"].values.astype(int),
                       "v": s.values}).dropna()
    n_top = max(1, len(df) // 3)
    return df.nlargest(n_top, "v")["year"].values.astype(int)


def composite_anomaly(driver_anom, high_years, all_years):
    low_years = np.setdiff1d(all_years, high_years)
    return (driver_anom.sel(year=high_years).mean("year", skipna=True) -
            driver_anom.sel(year=low_years ).mean("year", skipna=True))


def _interp_colors(palette, n):
    from matplotlib.colors import to_rgba
    rgba = np.array([to_rgba(c) for c in palette])
    xs = np.linspace(0, 1, len(palette)); xn = np.linspace(0, 1, n)
    return [tuple(r) for r in
            np.column_stack([np.interp(xn, xs, rgba[:, i]) for i in range(4)])]


def _draw_panel(ax, da, gdf, geom, levels, title, tag=None):
    cmap = mcolors.ListedColormap(_interp_colors(DIV_COLORS, len(levels) - 1))
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
        ax.text(0.03, 0.97, tag, transform=ax.transAxes,
                ha="left", va="top", fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.75))
    ax.set_title(title, fontsize=8.5, fontweight="bold", pad=3)
    style_axis(ax)

    de_mask = build_mask(da["lon"].values, da["lat"].values, geom)
    if de_mask.any():
        mv = float(np.nanmean(da.values[de_mask]))
        ax.text(0.03, 0.03, f"DE: {mv:+.1f}",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=6.5, color="#222222",
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="#aaaaaa", alpha=0.88, lw=0.4))
    return cmap, norm


def plot_composite_figure(composites, gdf, geom, outfile, suptitle):
    """One grouped multi-panel composite figure — one panel per driver."""
    from matplotlib.gridspec import GridSpec
    names = list(composites.keys())
    n = len(names)
    ncols = 2 if n <= 4 else 3
    nrows = int(np.ceil(n / ncols))

    fig = plt.figure(figsize=(3.6 * ncols + 0.4, 3.8 * nrows + 0.5))
    fig.patch.set_facecolor("white")
    fig.suptitle(suptitle, fontsize=10, y=1.01)

    gs = GridSpec(nrows, ncols * 2, width_ratios=([1, 0.06] * ncols),
                  left=0.02, right=0.97, top=0.93, bottom=0.04,
                  hspace=0.18, wspace=0.0)

    for k, dname in enumerate(names):
        row = k // ncols; col = (k % ncols) * 2
        ax = fig.add_subplot(gs[row, col], projection=PROJ)
        cmap, norm = _draw_panel(
            ax, composites[dname], gdf, geom,
            DRIVER_LEVELS[dname], DRIVER_LONG[dname],
            tag=f"({chr(97 + k)})")

        cax = ax.inset_axes([1.015, 0.0, 0.06, 1.0])
        _lv = DRIVER_LEVELS[dname]; _n = len(_lv)
        cb  = ColorbarBase(cax, cmap=cmap, norm=norm,
                           boundaries=_lv,
                           ticks=[_lv[0], _lv[_n // 4], 0, _lv[3 * _n // 4], _lv[-1]],
                           orientation="vertical", extend="neither")
        cb.ax.tick_params(labelsize=5, pad=1)
        cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        cb.outline.set_linewidth(0.4)
        cb.set_label(f"Anomaly [{DRIVER_UNITS[dname]}]",
                     fontsize=5.5, labelpad=2)

    fig.savefig(outfile, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outfile}")


def run_composite_group(group_name, indices, drivers, gdf, geom):
    """One composite figure per index — all relevant drivers in each figure."""
    for _display, nc_stem, label in indices:
        try:
            index_field = load_index_field(nc_stem, "ICON")
        except FileNotFoundError as e:
            print(f"  [{group_name}/{label}] composite skipped — {e}")
            continue

        all_years  = index_field["year"].values.astype(int)
        high_years = top_tercile_years(index_field)
        print(f"  [{group_name}] {label}: top-tercile years ({len(high_years)}): "
              f"{sorted(high_years)}")

        composites = {}
        for dname in drivers:
            try:
                anom = load_driver_anom_field(dname)
            except Exception as e:
                print(f"    {dname}: could not load — {e}")
                continue
            anom_years = anom["year"].values.astype(int)
            common     = np.intersect1d(high_years, anom_years).astype(int)
            if len(common) < 3:
                print(f"    {dname}: <3 overlapping years — skipping.")
                continue
            composites[dname] = composite_anomaly(anom, common, anom_years)

        if len(composites) < 2:
            print(f"  [{group_name}/{label}] not enough drivers — skipping composite figure.")
            continue

        plot_composite_figure(
            composites, gdf, geom,
            outfile=os.path.join(FIGDIR, f"composite_{label.lower()}.png"),
            suptitle=(f"Driver anomalies during upper-tercile {label} summers  "
                      f"·  ICON-CLM  ·  JJA 1950–2022"))


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── load Germany boundary ─────────────────────────────────────────────────
    print("Loading Germany boundary ...")
    gdf, geom = load_country_shape(GERMANY_SHP)

    # ── pre-load Germany-average driver series (for correlation) ──────────────
    print("\nLoading driver area-mean series ...")
    driver_series = {}
    for d in set(TEMP_DRIVERS) | set(PRECIP_DRIVERS):
        try:
            driver_series[d] = load_driver_series(d)
            print(f"  OK  {d}")
        except Exception as e:
            print(f"  WARNING {d}: {e}")

    temp_drv   = [d for d in TEMP_DRIVERS   if d in driver_series]
    precip_drv = [d for d in PRECIP_DRIVERS if d in driver_series]

    # ── Part 1: correlation heatmaps ──────────────────────────────────────────
    rows = []
    print("\n[1/2] Correlation heatmaps")
    print("Temperature group:")
    run_correlation_group("Temperature",   TEMP_INDICES,   temp_drv,
                          driver_series, rows)
    print("Precipitation group:")
    run_correlation_group("Precipitation", PRECIP_INDICES, precip_drv,
                          driver_series, rows)

    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(TABDIR, "driver_correlations.csv"), index=False)
        print(f"  Saved: {TABDIR}/driver_correlations.csv")

    # ── Part 2: composite anomaly maps ────────────────────────────────────────
    print("\n[2/2] Composite anomaly maps")
    run_composite_group("Temperature",   TEMP_INDICES,   temp_drv,   gdf, geom)
    run_composite_group("Precipitation", PRECIP_INDICES, precip_drv, gdf, geom)

    print("\n" + "=" * 56)
    print("Done. Output:")
    print(f"  {FIGDIR}/heatmap_temperature.png")
    print(f"  {FIGDIR}/heatmap_precipitation.png")
    for _, _, lbl in TEMP_INDICES + PRECIP_INDICES:
        print(f"  {FIGDIR}/composite_{lbl.lower()}.png")
    print(f"  {TABDIR}/driver_correlations.csv")
    print("=" * 56)
