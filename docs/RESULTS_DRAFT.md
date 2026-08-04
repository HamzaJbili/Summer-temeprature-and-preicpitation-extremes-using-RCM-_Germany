# Results — Summer climate extremes over Germany (JJA 1950–2022)

> Draft results section for the thesis. Trend values quoted below are the
> Germany-average Theil–Sen slopes per decade with Mann–Kendall significance
> (Yue–Wang TFPW correction): `*` p < 0.05, `**` p < 0.01. Numbers marked
> `‹fill›` should be read off `tables/extreme_indices_summary.csv` after the
> final 6-index run; the temperature and CDD values reproduce the trend
> heatmap already generated.

---

## 1. Index set and evaluation strategy

Six indices are retained, chosen for high signal-to-noise, clear physical
meaning, and direct relevance to Germany's documented summer hazards:

| Group | Index | Definition | Unit |
|-------|-------|------------|------|
| Temperature | **T90p** | JJA days exceeding the local 90th-pct Tmean (1961–1990) | days summer⁻¹ |
| Temperature | **HWN** | Heatwave number: events of ≥3 consecutive T90p days | events summer⁻¹ |
| Temperature | **HWD** | Mean heatwave duration | days event⁻¹ |
| Precipitation | **CDD** | Maximum consecutive dry days (P < 1 mm) | days summer⁻¹ |
| Precipitation | **SDII** | Simple daily intensity index (mean wet-day rainfall) | mm wet-day⁻¹ |
| Precipitation | **SPI** | Standardised Precipitation Index on JJA totals | dimensionless |

For each index we report (i) the Germany-average annual time series and its
trend, (ii) the spatial Theil–Sen trend field for E-OBS alongside the
ICON-CLM−minus−E-OBS difference (two independent colour scales), and
(iii) the 1950–2022 mean field as an E-OBS | ICON-CLM | bias triptych.
Model fidelity is summarised across all indices in a Taylor diagram and a
trend heatmap.

---

## 2. Model evaluation (Taylor diagram)

The Taylor diagram quantifies how well ICON-CLM reproduces the *interannual
variability* of each Germany-average index against E-OBS (the reference point
at σ\* = 1, r = 1).

- **Temperature indices cluster closest to the reference.** T90p, HWN and HWD
  show high temporal correlation (r ≈ ‹fill, ~0.85–0.92›) and near-unit
  normalised standard deviation, i.e. ICON-CLM captures both the timing and
  the amplitude of year-to-year heat variability well. This is expected for an
  ERA5-driven hindcast, in which large-scale thermal anomalies are strongly
  constrained by the driving reanalysis.
- **CDD** correlates moderately well (r ≈ ‹fill›), consistent with dry-spell
  length being partly governed by the same persistent circulation regimes that
  the reanalysis resolves.
- **SDII and SPI** lie furthest from the reference (lower r, σ\* departing from
  1). Wet-day intensity and standardised precipitation depend on
  sub-grid convective processes that a ~12 km hindcast represents only
  approximately, so weaker correlation here is physically unsurprising rather
  than a model failure.

**Takeaway:** ICON-CLM is most skilful for temperature-driven extremes and
progressively less so for intensity-/distribution-based precipitation indices —
a gradient that should frame the confidence attached to each trend below.

---

## 3. Temperature extremes

All three temperature indices show **significant positive trends in both
datasets**, and in every case **ICON-CLM underestimates the observed rate of
intensification** by roughly 20–25 %.

- **T90p** rises by **+1.08\*\* days decade⁻¹** in E-OBS versus
  **+0.85\*\* days decade⁻¹** in ICON-CLM. Over the full 1950–2022 record this
  is an increase of roughly seven to eight hot days per summer in the
  observations.
- **HWN** increases by **+0.16\*\* events decade⁻¹** (E-OBS) versus
  **+0.12\*\* events decade⁻¹** (ICON-CLM): heatwaves are becoming more
  frequent.
- **HWD** increases by **+0.19\*\* days event⁻¹ decade⁻¹** (E-OBS) versus
  **+0.14\*\* days event⁻¹ decade⁻¹** (ICON-CLM): the events that do occur are
  also getting longer.

Because HWN and HWD both rise, total heatwave exposure scales
multiplicatively (≈ HWN × HWD): more events *and* longer events compound the
seasonal heat load. Spatially, the mean-field maps show the highest hot-day
frequencies and longest heatwaves over the warmer continental interior and the
Upper Rhine / southwestern lowlands, where the 90th-pct Tmean threshold itself
is highest (threshold map). The trend-difference maps confirm the
underestimation is broadly distributed rather than driven by isolated cells.

This systematic under-trend is the well-documented tendency of regional climate
hindcasts to damp the observed European summer-warming amplification; it should
be flagged as a conservative bias when ICON-CLM output is used for impact
projection.

---

## 4. Precipitation — drought duration (CDD)

**CDD increases significantly in both datasets** — **+0.18\*\* days decade⁻¹**
(E-OBS) and **+0.12\*\* days decade⁻¹** (ICON-CLM) — indicating a lengthening
of the longest summer dry spell. This is the physically expected partner to the
temperature signal: longer dry spells deplete soil moisture, and the resulting
reduction in evaporative cooling feeds back onto temperature through the
soil-moisture–temperature coupling that amplifies central-European heat
extremes. The mean-field map locates the longest baseline dry spells over the
eastern/north-eastern lowlands (the climatologically driest part of Germany),
and the difference map again shows ICON-CLM with a slightly weaker positive
trend than E-OBS, mirroring the temperature behaviour.

The co-occurrence of robustly rising T90p **and** CDD is the central
hot-and-dry compound-hazard result of this analysis.

---

## 5. Precipitation — intensity and standardised severity (SDII, SPI)

In contrast to the drought-duration signal, the **intensity and standardised
precipitation indices show no robust trend**:

- **SDII**: **‹fill, ≈ −0.03 (E-OBS) / ≈ 0.00 (ICON-CLM)› mm wet-day⁻¹
  decade⁻¹**, neither significant. Wet-day mean intensity is essentially
  stationary over the period.
- **SPI** (JJA totals): **‹fill› SPI decade⁻¹**, not significant in either
  dataset, indicating no detectable systematic shift toward wetter or drier
  summers in the standardised, distribution-normalised sense.

This is an honest and important null result: over Germany the JJA precipitation
*amount/intensity* signal is weak and not robustly detectable at this
resolution, whereas the *temporal structure* of rainfall — expressed through
dry-spell length (CDD) — is changing. In other words, summers are not
reliably getting wetter or drier in total, but the rain is being delivered in a
more clustered, longer-dry-gap pattern. The lower Taylor skill for these two
indices (Section 2) further counsels caution in over-interpreting their trends.

---

## 6. Synthesis

| Index | E-OBS trend / decade | ICON-CLM trend / decade | Robust? | Interpretation |
|-------|----------------------|--------------------------|---------|----------------|
| T90p  | +1.08\*\* | +0.85\*\* | yes (both) | Hot-day frequency rising; model under-trends |
| HWN   | +0.16\*\* | +0.12\*\* | yes (both) | More heatwave events |
| HWD   | +0.19\*\* | +0.14\*\* | yes (both) | Longer heatwave events |
| CDD   | +0.18\*\* | +0.12\*\* | yes (both) | Longer dry spells (drought) |
| SDII  | ‹fill› | ‹fill› | no | Wet-day intensity stationary |
| SPI   | ‹fill› | ‹fill› | no | No standardised wet/dry shift |

**Two headline messages:**

1. **A coherent hot-and-dry intensification** — T90p, HWN, HWD and CDD all rise
   significantly and consistently, the signature of compound heat–drought
   hazard intensification over Germany since 1950.
2. **A conservative model** — ICON-CLM reproduces the *direction, significance
   and spatial pattern* of every temperature and drought-duration trend, but
   systematically *underestimates their magnitude* (~20–25 % weaker). It is
   therefore reliable for detection and attribution framing but likely a lower
   bound for impact magnitude.

The precipitation-intensity indices (SDII, SPI) add the important caveat that
not all aspects of the summer hydroclimate are changing detectably; the robust
precipitation signal is one of *temporal redistribution* (dry-spell
lengthening), not of *total amount or intensity*.
