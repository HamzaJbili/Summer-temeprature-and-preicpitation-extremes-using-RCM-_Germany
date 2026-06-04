"""
script3_drivers.py
------------------
Driver analysis for German summer extremes — correlation approach.

For every extreme index (Germany-average JJA series, 1950–2022) the
Pearson and Spearman rank correlations are computed against each
large-scale / surface driver variable (also Germany-average JJA means).
The full 6-index × 7-driver correlation structure is condensed into a
single heatmap figure, complemented by per-index correlation bar charts
and CSV tables.

Why correlation (not composite maps)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Correlating an extreme-index time series with candidate driver
  variables is the standard, compact way to quantify the strength and
  sign of each index–driver link (Hirschi et al. 2011; Mueller &
  Seneviratne 2012).  One heatmap summarises all 42 relationships,
  avoiding the 40+ separate composite maps that a full composite
  treatment would require.

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
  the lateral boundaries, so the dynamical drivers (PSL, wind) remain
  reliable even where the model has local biases in precipitation
  extremes (Kotlarski et al. 2014; Vautard et al. 2021).  The same
  model-internal driver approach is used by Loikith et al. (2015) for
  reanalysis-driven RCM temperature extremes.  For precipitation indices
  (CDD, SDII, SPI) the links reflect the model's internal physics rather
  than necessarily the observed system, and are interpreted as such.

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

Requires annual index NetCDF files produced by script2_extremes.py.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from scipy.stats import pearsonr, spearmanr

from utils import (
    load_field, area_mean,
    load_country_shape,
    START_YEAR, END_YEAR, DPI,
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
# Tuple: (display_name, nc_stem, row_label)
INDICES = [
    ("T90p_exceedance_days", "T90p_days", "T90p"),
    ("Heatwave_number",      "HWN",       "HWN"),
    ("Heatwave_duration",    "HWD",       "HWD"),
    ("SDII",                 "SDII",      "SDII"),
    ("CDD",                  "CDD",       "CDD"),
    ("SPI",                  "SPI",       "SPI"),
]

# ── driver file configuration ─────────────────────────────────────────────────
# JJA seasonal mean files produced by CDO (one value per year, 1950-2022).
# Naming convention: {var}_DE-0.25_JJA_1950-2022.nc
_SUFFIX = "DE-0.25_JJA_1950-2022.nc"

# Driver order defines the heatmap column order.
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

# Diverging blue-white-red palette for the correlation heatmap
CORR_COLORS = [
    "#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#f7f7f7",
    "#fddbc7", "#f4a582", "#d6604d", "#b2182b",
]


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

    # ── Build correlation matrices (indices × drivers) ────────────────────────
    avail_index_rows = []        # (display_name, row_label) actually present
    rP = []                      # Pearson r rows
    pP = []                      # Pearson p rows
    rS = []                      # Spearman r rows
    pS = []                      # Spearman p rows
    long_rows = []               # tidy CSV rows

    for display_name, nc_stem, row_label in INDICES:
        try:
            index_series = load_index_series(nc_stem, "ICON")
        except FileNotFoundError as e:
            print(f"  Skipping {display_name}: {e}")
            continue

        print(f"\n{'='*60}\nCorrelating: {display_name}")
        avail_index_rows.append((display_name, row_label))

        rowP, rowPp, rowS, rowSp = [], [], [], []
        for dname in avail_drivers:
            res = correlate(index_series, driver_series[dname])
            rowP.append(res["pearson_r"]);   rowPp.append(res["pearson_p"])
            rowS.append(res["spearman_r"]);  rowSp.append(res["spearman_p"])

            long_rows.append({
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
        idx_df = pd.DataFrame({
            "driver":    avail_drivers,
            "pearson_r": rowP,
            "pearson_p": rowPp,
        })
        plot_correlation_bars(
            idx_df, row_label,
            outfile=os.path.join(FIGDIR, f"{display_name}_driver_correlations_bar.png"),
        )

    if not avail_index_rows:
        raise SystemExit("No index files could be loaded — run script2 first.")

    rP = np.array(rP); pP = np.array(pP)
    rS = np.array(rS); pS = np.array(pS)
    row_labels = [lbl for _, lbl in avail_index_rows]

    # ── Main figures: Pearson and Spearman heatmaps ───────────────────────────
    print("\nGenerating correlation heatmaps ...")
    plot_correlation_heatmap(
        rP, pP, row_labels, avail_drivers,
        outfile=os.path.join(FIGDIR, "driver_correlation_heatmap_pearson.png"),
        title="Extreme index — driver correlation (Pearson)\n"
              "ICON-CLM Germany average, JJA 1950–2022",
        cbar_label="Pearson r",
    )
    plot_correlation_heatmap(
        rS, pS, row_labels, avail_drivers,
        outfile=os.path.join(FIGDIR, "driver_correlation_heatmap_spearman.png"),
        title="Extreme index — driver correlation (Spearman rank)\n"
              "ICON-CLM Germany average, JJA 1950–2022",
        cbar_label="Spearman ρ",
    )

    # ── Tidy CSV table (all index–driver pairs) ───────────────────────────────
    pd.DataFrame(long_rows).to_csv(
        os.path.join(TABDIR, "all_indices_driver_correlations.csv"), index=False)

    # ── Wide CSV matrices (Pearson r and Spearman rho) ────────────────────────
    pd.DataFrame(rP, index=row_labels, columns=avail_drivers).to_csv(
        os.path.join(TABDIR, "correlation_matrix_pearson.csv"))
    pd.DataFrame(rS, index=row_labels, columns=avail_drivers).to_csv(
        os.path.join(TABDIR, "correlation_matrix_spearman.csv"))

    print("\n" + "=" * 60)
    print("Script 3 complete.")
    print(f"  Heatmaps  → {FIGDIR}/driver_correlation_heatmap_pearson.png")
    print(f"              {FIGDIR}/driver_correlation_heatmap_spearman.png")
    print(f"  Bar charts→ {FIGDIR}/<index>_driver_correlations_bar.png")
    print(f"  Tables    → {TABDIR}/all_indices_driver_correlations.csv")
    print(f"              {TABDIR}/correlation_matrix_pearson.csv")
    print(f"              {TABDIR}/correlation_matrix_spearman.csv")
    print("=" * 60)
