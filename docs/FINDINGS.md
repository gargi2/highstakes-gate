# Pillar C — findings from the cross-domain gate benchmark

What the reused validation gate (`src/gate/`, unchanged from FedCheck) actually
does when run across three high-stakes domains — **healthcare** (MIMIC-IV
in-hospital mortality), **finance** (SEC EDGAR adverse-earnings), and
**government** (CMS Open Payments food-and-beverage classification) — on **real
data**. Every number here is quoted from a named results CSV; nothing is from
memory (project hard rule 1).

## Sources (read these, don't trust this prose)

| File | What it holds |
|---|---|
| `experiments/results/crossdomain_healthcare.csv` | 325 grid cells, real MIMIC-IV (`provenance=real:data/healthcare`). |
| `experiments/results/crossdomain_finance.csv` | 325 cells, real EDGAR (`real:data`, ~11k multi-quarter filings). |
| `experiments/results/crossdomain_government.csv` | 325 cells, real Open Payments (`real:data/PGYR2025_…`). |
| `experiments/results/metrics_summary.csv` | 60 rows = (domain × mode × ε) aggregates over 5 seeds × 3 severities, produced by `experiments/analyze.py`. |
| `paper/figures/fig1…fig5.{pdf,png}` | The five figures below, produced by `experiments/analyze.py`. |

Each grid is `mode × severity × ε × seed`, with `n_sources=6`, `n_degraded=2`,
ε ∈ {0.1, 0.5, 1, 5, ∞}, seeds 0–4. Reproduce the analysis with:

```bash
.venv/Scripts/python.exe experiments/analyze.py
```

Definitions used throughout (from the grid's own header):
- **detection recall** — of the 2 degraded sources, the fraction the mode's *own*
  gate check flagged. **precision** — of all flags, the fraction truly degraded.
- **withheld fraction** — `degraded_withheld / 2`: the fraction of degraded
  sources the gate's READY/FIXABLE/EXCLUDE decision actually kept out. Detection
  and withholding are **different columns** and they differ (see Finding 2).
- **clean false-flag rate** — on undegraded (`mode=none`) runs, the fraction of
  clean sources wrongly flagged.
- **auc_delta** — downstream ROC-AUC with gating minus without, on a clean 20%
  holdout that is never degraded.

---

## Finding 1 — What the gate catches, and what it misses (at ε=∞)

Quoted from `metrics_summary.csv` (ε=inf rows) and shown in
**`paper/figures/fig1_detection_by_mode.png`**.

| Mode | Healthcare recall | Finance recall | Government recall | Verdict |
|---|---|---|---|---|
| Structural | 1.00 | 1.00 | 1.00 | **Caught everywhere.** |
| Completeness | 0.67 | 1.00 | 1.00 | Caught; healthcare partial. |
| Distributional | 0.33 | 0.73 | 1.00 | **Domain-dependent.** |
| Label | 0.67 | **0.00** | 0.67 | **Misses balanced-label finance.** |

- **Structural degradations are caught perfectly in every domain** (recall = 1.00,
  precision = 1.00; `crossdomain_*.csv`, mode=structural). Schema/dtype breaks are
  unambiguous and transfer without tuning.
- **The label-rate detector is blind on finance** (recall 0.00 at every ε;
  `crossdomain_finance.csv`, mode=label). Finance's `adverse` label is ≈0.42
  positive — near-balanced — and a symmetric label flip barely moves the positive
  *rate* the detector watches. On rare-label tasks (healthcare mortality,
  government food-and-beverage) the same detector reaches 0.67. This is the single
  clearest **negative result** and is a property of a marginal-based detector, not
  a bug.
- **Distributional recall varies 0.33 → 1.00 across domains** at ε=∞. In
  healthcare the injected feature shift lands on a comparatively weak feature, so
  only strong shifts clear the divergence threshold without DP noise; government's
  features separate cleanly so recall is 1.00. Read this mode's ε sweep, never a
  single number (Finding 3).

---

## Finding 2 — Detection is *not* withholding (the governance nuance)

**`paper/figures/fig2_detect_vs_withhold.png`.** A source can be *flagged* by a
consensus check yet still be *kept* by the readiness decision, because a single
consensus failure is one check among ~15 in `score_source`. Quoted from
`metrics_summary.csv` (ε=inf, `withheld_frac_mean`):

| Mode | Detected (recall) | Actually withheld (frac) |
|---|---|---|
| Structural (all domains) | 1.00 | **1.00** — schema-incompatible sources are excluded. |
| Completeness (healthcare) | 0.67 | 1.00 — multiple null failures flip the decision. |
| Completeness (finance, government) | 1.00 | **0.00** — detected but kept. |
| Distributional (healthcare) | 0.33 | 1.00; (finance, government) | 0.00 |
| Label (all domains) | 0.00–0.67 | **0.00** — never withheld. |

The honest reading: **the gate reliably *withholds* only structural (and, in
healthcare, completeness) degradations.** Distributional and label degradations
are *detected and reported* but usually *not excluded*. The benchmark reports
both because they are not the same governance guarantee — "we measured and flagged
this" (MEASURE) is weaker than "we withheld it," and the paper should not conflate
them.

---

## Finding 3 — Privacy vs utility: DP noise trades precision for recall

**`paper/figures/fig3_privacy_utility.png`**, distributional mode, from
`crossdomain_*.csv` (mode=distributional and mode=none):

- As ε **falls** (more privacy), DP noise inflates *every* source's cross-source
  divergence. Degraded sources stay flagged (recall rises toward 1.0) but clean
  sources get **falsely** flagged — precision drops and the clean false-flag rate
  climbs. Healthcare distributional precision: `1.00 (ε=∞) → 0.83 (ε=0.5) → 0.33
  (ε=0.1)`; recall moves the opposite way `0.33 → 0.93 → 1.00`
  (`metrics_summary.csv`, healthcare/distributional).
- **The usable privacy knee is ε ≥ 0.5.** On clean data the false-flag rate is
  **0.0 at every ε ≥ 1 in all three domains** (`metrics_summary.csv`,
  `clean_false_flag_rate`), but at **ε = 0.1 it is 1.00 / 1.00 / 0.63** for
  healthcare / finance / government — the gate flags essentially everything.
- **Government precision caps at ~0.56 even at ε=∞** (`crossdomain_government.csv`,
  mode=distributional): its clean sources already diverge enough to trip the
  threshold, so this is a calibration ceiling, not a DP effect. Reported, not hidden.

No clean source was ever fully **EXCLUDED** in any domain (`clean_excluded`
max = 0 across all three CSVs), so the false *flags* above did not cascade into
false *withholding* — consistent with Finding 2.

---

## Finding 4 — Cross-domain transfer: label-rate transfers, divergence does not

**`paper/figures/fig4_cross_domain.png`.** Per-mode spread of detection recall
across the three domains (σ over domains, ε=∞; `crossdomain_*.csv`):

| Mode | σ across domains | Transfer verdict |
|---|---|---|
| Structural | **0.00** | Transfers perfectly. |
| Completeness | 0.19 | Transfers well. |
| Distributional | 0.34 | Does **not** transfer cleanly. |
| Label | 0.38 | Does **not** transfer — blind on balanced labels. |

This is the **FedCheck real-data lesson, reproduced across domains**: the
*schema/structural* and *label-rate* checks are simple marginals that behave
predictably, but *divergence-based distributional* detection depends on feature
scale, bin resolution, and base rate, so a threshold calibrated in one domain does
not carry to another (each domain needed its own `divergence_threshold`; see
`docs/WALKTHROUGH.md` "Adding a new domain"). The label-rate detector **transfers
across the two rare-label domains** (healthcare, government both 0.67) — it is the
*balanced base rate* in finance, not a domain change per se, that defeats it.

---

## Finding 5 — Gating barely moves the downstream model (an honest negative)

**`paper/figures/fig5_downstream.png`**, `auc_delta` at ε=∞ from
`crossdomain_*.csv`:

- **|ΔAUC| < 0.005 in every single cell.** The largest is healthcare/distributional
  at **+0.0049**; finance/structural is **−0.0002**; most are exactly 0.000
  (`metrics_summary.csv`, `auc_delta_mean`).
- Clean-baseline AUC is healthy (healthcare 0.855, finance 0.669, government 0.928;
  `mode=none`, `auc_nogate`), so the models train fine — gating just doesn't change
  them, because removing 2 of 6 sources leaves a similar training set and the clean
  holdout is unaffected by the degradation.

**The claim this does NOT support:** "gating improves downstream accuracy." In this
setup it does not. **The claim it DOES support:** the gate's value is *governance
and withholding of unfit sources* (Findings 1–2, mapped to NIST MEASURE in
`docs/NIST_MAPPING.md`), not a headline accuracy lift. Stating this plainly is more
defensible than fishing for a metric that moved.

---

## Reading the figures (for someone new to the area)

- **Fig 1 — detection heatmaps.** Rows = the four injectable failure modes; columns
  = the three domains. Left = *recall* (did the gate flag the bad sources?), right
  = *precision* (were its flags right?). Green = 1.0 (good), red = 0.0. The one red
  cell (finance/label recall) is the headline blind spot. "n/a" = precision
  undefined because the detector raised no flags at all.
- **Fig 2 — detection vs withholding.** For each domain, solid bars = *detected*,
  hatched bars = *actually withheld by the gate's decision*. Where a solid bar has
  no hatched twin, the gate saw the problem but still let the source through — the
  gap between "measuring" and "acting."
- **Fig 3 — privacy–utility.** x-axis = the differential-privacy budget ε, most
  private on the left. Three lines per domain: detection precision, detection
  recall, and the false-alarm rate on clean data. The scissor shape (recall up,
  precision down as you go left) is the core privacy/utility tradeoff; the orange
  line shooting up at ε=0.1 is the gate crying wolf under heavy noise.
- **Fig 4 — cross-domain transfer.** Grouped bars: one cluster per mode, one bar
  per domain. σ (printed above each cluster) is how much the bars disagree — σ=0
  (structural) means the check behaves identically everywhere; large σ (label,
  distributional) means "your mileage varies by domain."
- **Fig 5 — downstream effect.** Change in model AUC (×10⁻³) from turning the gate
  on. Bars hug zero on purpose: this is the negative result that gating is a
  data-governance control here, not an accuracy booster.

---

## Summary for the paper

1. **Catches and transfers:** structural (perfectly) and completeness (well) —
   these are the gate's dependable, domain-agnostic guarantees.
2. **Catches but domain-dependent:** distributional — reported as an ε sweep with a
   usable knee at ε ≥ 0.5; needs per-domain threshold calibration.
3. **Misses:** subtle label noise on **balanced-label** tasks (finance) — a stated
   limitation of the marginal label-rate detector.
4. **Detection ≠ withholding:** only structural/completeness reliably change the
   READY/EXCLUDE decision; distributional/label are flagged but usually kept.
5. **No downstream accuracy lift** in this setup — the contribution is governance
   (MEASURE mapping in `docs/NIST_MAPPING.md`), not model performance.

All five are quoted from the CSVs named above and are reproducible via
`experiments/analyze.py`. Known limitations and calibration caveats are cross-listed
in [`docs/WALKTHROUGH.md`](WALKTHROUGH.md) and
[`verification/VERIFICATION.md`](../verification/VERIFICATION.md).
