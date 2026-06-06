"""
script3_drivers.py
------------------
Drivers of German summer extremes — correlation analysis.

Each extreme index (Germany-average JJA series, 1950–2022) is correlated
with its physically relevant ICON-CLM JJA-mean drivers.  A strong,
statistically significant correlation identifies a driver of that extreme.
This is the standard, compact way to diagnose drivers of climate extremes
(Hirschi et al. 2011, Nat. Geosci.; Mueller & Seneviratne 2012, PNAS).

Two physically motivated groups (no blanket index×driver matrix):

    TEMPERATURE    T90p, HWN, HWD   ×  PSL, SHF, LHF, CLT       (3×4)
    PRECIPITATION  SDII, CDD, SPI   ×  PSL, LHF, CLT, CAPE, CIN  (3×5)

Output (3 files only):
    heatmap_temperature.png      correlation heatmap, temperature group
    heatmap_precipitation.png    correlation heatmap, precipitation group
    driver_correlations.csv      every index–driver r (Pearson + Spearman), p

Both indices and drivers come from the SAME ICON-CLM ERA5-driven hindcast,
so this is a model-internal physical-consistency diagnosis (Loikith et al.
2015): does the model produce its extremes through physically consistent
relationships?  The large-scale driver (PSL) is constrained by the ERA5
boundary conditions (Kotlarski et al. 2014; Vautard et al. 2021).

Requires the annual index NetCDF files produced by script2_extremes.py and
the CDO JJA-mean driver files.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from scipy.stats import pearsonr, spearmanr

from utils import load_field, area_mean, START_YEAR, END_YEAR, DPI
from utils import set_ipcc_style
set_ipcc_style()

# ── output directories ────────────────────────────────────────────────────────
FIGDIR = os.path.join("output_drivers", "figures")
TABDIR = os.path.join("output_drivers", "tables")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(TABDIR, exist_ok=True)

INDEX_NC_DIR = "/archive1/hamza_data/DE_files/DE_1950-2022/extremes_indices/output_extremes/netcdf"

# ── two physically motivated index groups ─────────────────────────────────────
# Tuple: (display_name, nc_stem, row_label)
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

# ── driver files (CDO JJA seasonal means, one value per year 1950-2022) ────────
_SUFFIX = "DE-0.25_JJA_1950-2022.nc"
DRIVER_FILES = {
    "PSL":  f"psl_{_SUFFIX}",   "SHF":  f"hfss_{_SUFFIX}",
    "LHF":  f"hfls_{_SUFFIX}",  "CLT":  f"clt_{_SUFFIX}",
    "CAPE": f"cape_{_SUFFIX}",  "CIN":  f"cin_{_SUFFIX}",
}
DRIVER_VARS = {
    "PSL": "psl", "SHF": "hfss", "LHF": "hfls",
    "CLT": "clt", "CAPE": "cape", "CIN": "cin",
}
DRIVER_SCALE = {"PSL": 0.01}   # Pa → hPa
DRIVER_LONG = {
    "PSL": "Sea-level pressure", "SHF": "Sensible heat flux",
    "LHF": "Latent heat flux",   "CLT": "Total cloud cover",
    "CAPE": "Convective available PE", "CIN": "Convective inhibition",
}

# Diverging blue-white-red palette for the heatmap
CORR_COLORS = ["#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#f7f7f7",
               "#fddbc7", "#f4a582", "#d6604d", "#b2182b"]


# ── data loading ───────────────────────────────────────────────────────────────
def load_index_series(nc_stem, dataset_label="ICON"):
    """Germany-average annual series of an index produced by script2."""
    path = os.path.join(INDEX_NC_DIR, f"{nc_stem}_{dataset_label}_annual.nc")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Annual index file not found: {path}\nRun script2_extremes.py first.")
    ds = xr.open_dataset(path)
    da = ds[list(ds.data_vars)[0]].sortby("lat").sortby("lon")
    return area_mean(da)


def load_driver_series(dname):
    """Germany-average annual JJA-mean series of one driver variable."""
    da = load_field(DRIVER_FILES[dname], DRIVER_VARS[dname])
    scale = DRIVER_SCALE.get(dname, 1.0)
    if scale != 1.0:
        da = da * scale
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
    m = idx.merge(drv, on="year")
    if len(m) < 10:
        return dict(n=len(m), pearson_r=np.nan, pearson_p=np.nan,
                    spearman_r=np.nan, spearman_p=np.nan)
    pr, pp = pearsonr(m["index"], m["driver"])
    sr, sp = spearmanr(m["index"], m["driver"])
    return dict(n=len(m), pearson_r=float(pr), pearson_p=float(pp),
                spearman_r=float(sr), spearman_p=float(sp))


def _stars(p):
    if not np.isfinite(p):
        return ""
    return "**" if p < 0.01 else ("*" if p < 0.05 else "")


# ── the one figure people actually use: a correlation heatmap ──────────────────
def plot_heatmap(r_mat, p_mat, row_labels, col_labels, outfile, title):
    """Index × driver correlation heatmap (cell = Pearson r, stars = p-value)."""
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

    ax.set_title(title + "\n(* p<0.05, ** p<0.01)", fontsize=10, pad=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Pearson r", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    plt.tight_layout()
    fig.savefig(outfile, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def run_group(group_name, indices, drivers, driver_series, rows_out):
    """Correlate one index group against its drivers → one heatmap + CSV rows."""
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
            rows_out.append({"group": group_name, "index": display_name,
                             "driver": d, "driver_long": DRIVER_LONG[d],
                             "n": res["n"],
                             "pearson_r":  round(res["pearson_r"], 3)  if np.isfinite(res["pearson_r"])  else np.nan,
                             "pearson_p":  round(res["pearson_p"], 4)  if np.isfinite(res["pearson_p"])  else np.nan,
                             "spearman_r": round(res["spearman_r"], 3) if np.isfinite(res["spearman_r"]) else np.nan,
                             "spearman_p": round(res["spearman_p"], 4) if np.isfinite(res["spearman_p"]) else np.nan})
            print(f"  {label:5s} × {d:5s}: r={res['pearson_r']:+.2f} (p={res['pearson_p']:.3f})")
        rP.append(rowP); pP.append(rowPp)

    if not avail:
        print(f"  [{group_name}] no index files — skipping heatmap.")
        return
    plot_heatmap(np.array(rP), np.array(pP), avail, drivers,
                 outfile=os.path.join(FIGDIR, f"heatmap_{group_name.lower()}.png"),
                 title=f"{group_name} extremes — driver correlation\n"
                       "ICON-CLM Germany average, JJA 1950–2022")
    print(f"  [{group_name}] heatmap written.")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading drivers (JJA seasonal means) ...")
    driver_series = {}
    for d in set(TEMP_DRIVERS) | set(PRECIP_DRIVERS):
        try:
            driver_series[d] = load_driver_series(d)
        except Exception as e:
            print(f"  WARNING: could not load {d}: {e}")

    temp_drivers   = [d for d in TEMP_DRIVERS   if d in driver_series]
    precip_drivers = [d for d in PRECIP_DRIVERS if d in driver_series]

    rows = []
    print("\nTemperature group:")
    run_group("Temperature", TEMP_INDICES, temp_drivers, driver_series, rows)
    print("\nPrecipitation group:")
    run_group("Precipitation", PRECIP_INDICES, precip_drivers, driver_series, rows)

    if not rows:
        raise SystemExit("No index files could be loaded — run script2 first.")

    pd.DataFrame(rows).to_csv(
        os.path.join(TABDIR, "driver_correlations.csv"), index=False)

    print("\n" + "=" * 56)
    print("Done. 3 files:")
    print(f"  {FIGDIR}/heatmap_temperature.png")
    print(f"  {FIGDIR}/heatmap_precipitation.png")
    print(f"  {TABDIR}/driver_correlations.csv")
    print("=" * 56)
