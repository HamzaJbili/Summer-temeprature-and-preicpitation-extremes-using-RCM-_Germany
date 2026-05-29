# Scientific & Code Review — Script 2 (Summer Extremes, Germany 1950–2022)

**Scope:** `script2_extremes.py` + supporting `utils.py`; cross-checks against
`script1_mean_climate.py` and `script3_drivers.py`. Covers methodology,
correctness, your reported observations on the figures, a literature benchmark
for the *expected* results, and the "% of days" suggestion.

> No numerical results are reproduced here: the input NetCDFs live on the HPC
> (`/work/jbiliham/…`) and outputs are git-ignored. The "results review" below
> therefore evaluates **what the code will produce and whether that is
> scientifically defensible**, anchored to (a) the index definitions in code and
> (b) the published literature for German summer extremes.

---

## 1. Executive summary

The pipeline is methodologically sound and publication-aligned: percentile
thresholds on the WMO 1961–1990 base, Theil–Sen slopes, Mann–Kendall with
Yue–Wang TFPW autocorrelation correction, IPCC-style figures. The recent fixes
(T90p threshold, shared paired-map colour scale, corrected `tick_fmt`, CDD
drought palette, Cartesian Taylor diagram, readable heatmap) resolve the
visual problems you reported.

Four concrete **defects were found and fixed in this pass** (Section 2.1). The
remaining recommendations (Section 2.3) are about **statistical rigour**
(field significance), **presentation** (the "% of days" idea), and
**consistency** between scripts 1 and 2 — none block the analysis.

---

## 2. Code review

### 2.1 Defects found and fixed in this pass

| # | Severity | File | Problem | Fix |
|---|----------|------|---------|-----|
| C1 | **Critical** | `utils.py` / `script3` | `script3` imports `clip_contourf` from `utils`, but the function **did not exist** → `ImportError` on import; script 3 could never run. | Implemented `clip_contourf(cf, ax, geom)` (clips a contourf to the Germany polygon on a plain Axes; handles matplotlib ≥3.8 and older). |
| C2 | **Major** | `script3` | `INDICES` still referenced `("T95_exceedance_days","T95_days")`, but script 2 now writes `T90p_days_*.nc`. The temperature index would hit `FileNotFoundError`, be silently caught, and **dropped from the driver analysis**. | Updated to `("T90p_exceedance_days","T90p_days")`. |
| C3 | **Major (visual)** | `script2` | `CDD_COLORS` had **11 colours for 10 intervals** (`CDD_LEVELS` = 11 boundaries). `BoundaryNorm` then mis-maps bins → wrong colours in the CDD panel of `precipitation_overview.png`. | Trimmed to a clean 10-colour blue→pale→brown drought ramp. |
| C4 | **Minor** | `README.md` | Still documented T95. | Updated temperature table and rationale to T90p. |

All four files parse cleanly after the edits, and every overview palette now
satisfies `len(colors) == len(levels) − 1`.

### 2.2 Methodology — correct and defensible

- **Threshold base period (1961–1990).** Correct choice for *trend* studies:
  it predates the strong post-1990 acceleration, so exceedance-count trends
  reflect climate change rather than a baseline absorbing the signal
  (WMO guidance; Alexander et al. 2006).
- **Trend estimator.** Theil–Sen + Mann–Kendall is the field standard for
  non-Gaussian, outlier-prone climate-index series. The **Yue–Wang TFPW**
  modification is the right call — raw MK over-rejects the null under positive
  lag-1 autocorrelation (Yue & Wang 2004). Graceful fallback to `original_test`
  is sensible.
- **Wet-day handling.** R95p threshold computed on **wet days only**
  (`P ≥ 1 mm`) — matches the ETCCDI definition (percentile of the wet-day
  distribution, not all days). R95pTOT and SDII guard against divide-by-zero.
- **Heatwave definition.** ≥3 consecutive days over the local percentile, using
  Tmean (integrated thermal load, the physiologically relevant quantity) —
  consistent with Perkins & Alexander (2013). HWN counts events; HWD averages
  event length; together they separate frequency from persistence. Good.
- **Run-length encoding** (CDD, HWN, HWD, CWD) via padded `np.diff` is correct
  and handles edge spells (start/end of season) properly.

### 2.3 Recommendations (not yet applied — your call)

1. **Field significance / multiple-testing control (highest-value rigour fix).**
   Per-gridcell stippling at raw `p < 0.05` over ~hundreds of German cells will
   flag ≈5 % of cells as "significant" purely by chance. Best practice is a
   **false-discovery-rate** control (Benjamini–Hochberg; Wilks 2016, *BAMS*).
   Recommend applying FDR to the `mk_pvalue` field before stippling, and
   reporting the FDR-controlled significant-area fraction. Low effort, high
   credibility — reviewers of climate-trend maps increasingly expect it.

2. **Percentile-index base-period inhomogeneity.** Exceedance counts *within*
   the 1961–1990 base are slightly biased relative to outside it (the threshold
   "sees" the same data). The standard remedy is the bootstrap of
   Zhang et al. (2005, *J. Climate*). For a thesis this can simply be
   **acknowledged as a caveat**; full implementation is optional.

3. **Latitude weighting in `area_mean`.** Currently an unweighted mean over
   lat/lon. Across Germany's ~7.7° latitude span the `cos(lat)` correction is
   <1 %, so conclusions don't change, but a weighted mean is more defensible in
   the methods text. One-line change if you want it.

4. **Paired-map colour convention consistency.** Script 2 calls
   `plot_paired_trend_maps` with the default `force_diverging=False`, so an
   all-positive temperature trend renders as a **sequential warm ramp** (no blue
   half). Script 1 effectively uses the symmetric blue–red. Both are valid, but
   decide which convention the thesis should use **consistently**. If you want
   T90p/HWN/HWD to keep the blue–red zero-centred bar like script 1, pass
   `force_diverging=True` for the temperature indices.

5. **Performance (not correctness).** CDD/HWN/HWD use triple Python loops
   (`year × lat × lon`) → minutes per index. If runtime becomes painful, these
   vectorise well with `xarray.apply_ufunc` over a Numba/`scipy.ndimage.label`
   run-length kernel. Optional.

---

## 3. Results review — your reported observations, point by point

Your earlier observations and how the code now addresses them, with the
*expected* physical outcome (benchmarked in Section 4):

| Your observation | Diagnosis | Status |
|------------------|-----------|--------|
| "CDD time-series title says *maximum*" | `long_name` carried "Max"; positive trend wasn't the issue, the label was | **Fixed** — `CDD — Consecutive dry days` |
| "Stars in the legend" (time series) | Significance asterisks appended to the trend label | **Fixed** — slope value only |
| "CDD map colour is green" | Used the wet (BrBG) palette where green = wetter; drought should read brown | **Fixed** — dedicated blue→pale→brown ramp; positive (drought) = brown |
| "ICON colorbar gives 0,0,0,1,1,1" (CDD/HWD/R95p/R95TOT/T95) | `tick_fmt="%.0f"` rounded small decimal trends (0.05…0.5) to 0/1 | **Fixed** — `%.1f`/`%.2f` per index |
| "HWD curve disrupted" / "HWN not informative" | At the **95th** pct only ~3–4 days/summer exceed → many years had **zero** ≥3-day heatwaves → NaN gaps in the series and near-empty maps | **Fixed** by moving to **T90p** (~9 days/summer): events occur nearly every year → continuous curve, populated maps |
| "Taylor diagram is just a semi-circle, no SD axes" | Polar projection with `set_rticks([])` hid the radial scale | **Fixed** — rewritten in Cartesian coords with labelled x/y standard-deviation axes, SD arcs, RMSE arcs, correlation rays |
| "Trend heatmap totally missed / not readable" | Units crammed into each cell at tiny font | **Fixed** — units moved to row labels, larger cell font, wider figure |
| "Same numbers & colorbar for E-OBS vs ICON" | Panels were auto-scaled independently | **Fixed** — both panels pooled into one shared scale |

**Interpretation caveat to state in the thesis:** HWD is a *conditional* mean
(defined only when a heatwave occurs). The Germany-average HWD series therefore
averages over a varying set of cells/years; with T90p the conditioning set is
near-complete, but the wording in the methods should make the conditional nature
explicit.

---

## 4. Literature benchmark — what the results *should* show

Use this as a sanity check when you run the pipeline. "Expected sign/strength"
is the consensus for **JJA, Germany/Central Europe, ~1950–2022**.

### Temperature

| Index | Expected (literature) | Sources |
|-------|-----------------------|---------|
| **T90p** (hot-day frequency) | Strong, **significant positive** trend almost everywhere; Germany hot-day counts rising significantly, record years 2003/2015/2018/2022 | UBA *Hot days* indicator; Copernicus ESOTC 2022 |
| **HWN / HWD** | Marked increase in heatwave frequency **and** duration, concentrated post-1990; W/Central-European heat extremes rising **faster than RCMs simulate** owing to circulation trends | Vautard et al. 2023 (*Nat. Commun.* s41467-023-42143-3); RCM heatwave evaluation NHESS 24/265/2024 |
| **Model note** | ERA5-driven ICON-CLM should reproduce the *sign* and interannual *timing* well (high Taylor correlation for T indices) but is likely to **under-estimate the magnitude** of the observed hot-extreme trend. | Vautard et al. 2023 |

### Heavy precipitation

| Index | Expected (literature) | Sources |
|-------|-----------------------|---------|
| **Rx1day, Rx5day, R95p, SDII** | Long-term **intensification** of sub-daily–to–5-day extremes across Central Europe since ~1901; **but the summer signal is spatially heterogeneous and frequently *not* field-significant** — positive in the mountainous south & northern coast, weaker/negative in central Germany | Zolina et al. 2014; "Observed extreme precip & scaling in Central Europe" (ScienceDirect S2212094719301720); MDPI *Water* 12/1950 |
| **R95pTOT** | Slight positive (greater share of rainfall delivered by extremes) — the "intensification of extremes" fingerprint | Fischer et al. 2014; Zolina et al. 2014 |
| **Model note** | Convective summer extremes are the **hardest** for a ~12 km RCM (no convection-permitting); expect **lower Taylor correlation and larger scatter** than for temperature, and a possible σ-ratio bias. | general RCM evaluation literature |

**Implication:** if your maps show *weak, patchy, largely non-significant* JJA
precipitation trends with a small positive median and a positive R95pTOT, that
is **consistent with the literature**, not a bug. This is exactly why field
significance (Rec. 1) matters — it prevents over-claiming from scattered
significant cells.

### Drought

| Index | Expected (literature) | Sources |
|-------|-----------------------|---------|
| **CDD** | **Increasing** summer dry-spell length, strongest in the **NE and Rhine-Main**; low-soil-moisture days up significantly since 1961; 2018–2019 unprecedented in 250 yr; 2022 severe | Helmholtz Climate Initiative; UBA; REKLIM; NHESS 25/1293/2025 |
| **Model note** | RCMs reproduce drought *characteristics* with known biases; expect ICON-CLM to capture the positive tendency but possibly mis-site the maximum. | NHESS 22/3875/2022 |

---

## 5. The "% of days" suggestion — scientific assessment

Your idea (use *% of days* instead of a raw day-count where the map is
uninformative) is well-founded — **but only for some indices**:

| Index | Current unit | "% of days"? | Verdict |
|-------|--------------|--------------|---------|
| **T90p** | days summer⁻¹ | **% of JJA days** | **Recommended.** This is in fact the **ETCCDI convention** for percentile warm-day indices (TX90p/TN90p are *defined* as "percentage of days exceeding the percentile"). Conversion is `days / 92 × 100`. Makes it season-length-independent and directly comparable to published TX90p figures. |
| **R95p** | days summer⁻¹ | **% of wet days** (optional) | Defensible — frames "how often heavy rain occurs" relative to rain opportunities. Optional; the raw count is also standard. |
| **HWN** | events summer⁻¹ | No | It counts **events**, not days; a percentage is not meaningful. Sparsity was the real problem and T90p fixed it. |
| **CDD** | days summer⁻¹ | No | It is a **duration** (longest consecutive run), not a frequency; "% of days" has no clean interpretation. Keep days. |
| **Rx1day/Rx5day/SDII** | mm | No | Intensities, not counts. |

**Recommendation:** convert **T90p to "% of JJA days"** (aligns with ETCCDI,
improves readability and cross-study comparability) and optionally R95p to
"% of wet days". Leave HWN/CDD/Rx*/SDII as they are. This is a definitional
change to the index, so it's left for you to approve before I apply it.

---

## 6. Suggested action list

**Applied in this pass:** C1–C4 (clip_contourf, script3 T90p, CDD palette,
README).

**Awaiting your go-ahead (each is independent):**
1. Convert **T90p → % of JJA days** (ETCCDI-consistent).
2. Add **FDR field-significance** control to trend-map stippling.
3. Pass `force_diverging=True` for temperature paired maps (script-1/2
   colour-convention consistency).
4. Latitude-weight `area_mean`.
5. Shrink the `precipitation_overview.png` title / optionally move **HWN** to an
   annex (you raised both earlier).

Tell me which of 1–5 to implement and I'll make the changes.

---

### References cited
- Alexander, L. V. et al. (2006). *JGR-Atmos.* 111, D05109.
- Fischer, E. M. et al. (2014). *Nat. Clim. Change* 4, 713–717.
- Perkins, S. E. & Alexander, L. V. (2013). *J. Climate* 26, 4500–4517.
- Vautard, R. et al. (2023). *Nat. Commun.* 14, 6803 (s41467-023-42143-3).
- Wilks, D. S. (2016). *BAMS* 97, 2263–2273 (FDR for field significance).
- Yue, S. & Wang, C. Y. (2004). *Water Resour. Res.* 40, W08201.
- Zhang, X. et al. (2005). *J. Climate* 18, 1641–1651 (base-period bootstrap).
- Zolina, O. et al. (2014). *Clim. Dyn.* 42, 881–898.
- Umweltbundesamt (UBA) — *Hot days* indicator; Helmholtz Climate Initiative;
  REKLIM; Copernicus ESOTC 2022; NHESS 24/265/2024, 22/3875/2022, 25/1293/2025.
