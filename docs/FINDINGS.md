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
  `crossdomain_finance.csv`, mode=label) — and this is a **proven limitation with a
  derivation**, not an empirical surprise. The detector compares a source's
  positive *rate* against consensus. For a base rate `p` and a **symmetric** flip
  of a fraction `r` of labels (`src/domains/degradations.py`, `label_noise`), the
  post-flip positive rate is

  > `p' = p(1−r) + (1−p)r = p + r(1−2p)`.

  The shift is `r·(1−2p)`, which **vanishes as p→0.5**. Finance's `adverse` base
  rate is ≈0.42, so even the `high` flip (`r=0.45`) moves the marginal only
  ≈`0.45·(1−0.84)=0.072` — below the detector's deviation threshold → recall 0. On
  rare-label tasks (healthcare mortality, government food-and-beverage, `p`≪0.5)
  the same `r` produces a large shift, hence recall 0.67. **Result: marginal
  label-rate checks are provably blind to symmetric label noise on balanced
  targets.** Two consequences for the paper, both future work in
  `docs/NEXT_EXPERIMENTS.md`: (i) test **asymmetric** noise, where `p'` *does*
  move, to turn the binary miss into a detectability-threshold curve; (ii) add a
  detector with a mechanism to see symmetric noise (confident-learning /
  cross-validated disagreement) so we report a fix, not just a hole. **Both are now
  built and the fix works — see Finding 8.**
- **Distributional recall varies 0.33 → 1.00 across domains** at ε=∞. In
  healthcare the injected feature shift lands on a comparatively weak feature, so
  only strong shifts clear the divergence threshold without DP noise; government's
  features separate cleanly so recall is 1.00. Read this mode's ε sweep, never a
  single number (Finding 3).

---

## Finding 2 — THE SPINE: composite readiness scoring detects far more than it withholds

**`paper/figures/fig6_detection_action_gap.png`** and
**`paper/figures/fig2_detect_vs_withhold.png`.** This is the paper's central
finding, and it is robust because it appears two independent ways (across the
cross-domain grid here, and in the single-domain "site at 30 % nulls still scores
READY" observation from FedCheck — one phenomenon, not two loose ends).

**Aggregated over all three domains at ε=∞** (`crossdomain_*.csv`,
`det_recall` vs `degraded_withheld/2`):

- the gate **detects 0.76** (95% CI [0.74, 0.76]) of degraded sources but
  **withholds only 0.42** — a **0.34 gap (95% CI [0.32, 0.35]**, seed-level bootstrap,
  10k resamples); the CI excludes 0 by a wide margin, so the gap is real, not seed noise;
- **74 of 180 degraded cells (41 %)** are *detected-but-not-withheld*;
- the gap is **0.00 for structural** (schema-incompatible → hard EXCLUDE) but
  **0.44–0.56 for completeness/distributional/label**.

**Mechanism (not a mystery — it is the scoring rule, `src/gate/readiness.py`):**
readiness = `0.4·check_pass_rate + 0.3·schema_score + 0.3·divergence_score`, with
READY at ≥ 0.8. A single failed check among ~15 moves `check_pass_rate` from 1.00
to ~0.93, i.e. the composite score from ~0.97 to ~0.94 — still READY. **A severe
single-dimension defect is averaged away by a composite score.** This is a general
property of weighted-mean readiness/data-quality scores, stated here as a
measurable result with the two metrics (detection rate, gate-action rate) reported
separately everywhere.

Per-mode detail (`metrics_summary.csv`, ε=inf, `withheld_frac_mean`):

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

## Finding 5 — The downstream-accuracy arm is UNDERPOWERED (uninformative, not a negative)

**`paper/figures/fig5_downstream.png`**, `auc_delta` at ε=∞ from
`crossdomain_*.csv`. An earlier draft read the flat `auc_delta` as "the gate is
governance-only." That reading is **not licensed by this experiment**, and we
retract it. The three-way check that decides the question
(clean → corrupted-ungated → corrupted-gated; `auc_nogate` on `mode=none` vs on a
corrupted mode vs `auc_gate`) shows the corruption **never degraded the model in
the first place**:

| Domain | clean AUC | worst corrupted-ungated AUC | max drop (clean→corrupt) |
|---|---|---|---|
| healthcare | 0.8553 | 0.8508 (distributional) | **0.0045** |
| finance | 0.6694 | 0.6670 (distributional) | **0.0024** |
| government | 0.9276 | 0.9276 | **0.0000** |

Because clean ≈ corrupted-ungated in **every** cell, the downstream arm measures
nothing about the gate: if removing the corruption doesn't move accuracy, neither
can gating it out. Two design reasons, both fixable:

1. **Source-level dilution.** Only 2 of 6 sources are corrupted, so ≥4 clean
   sources dominate the pooled training set (`n_sources=6`, `n_degraded=2`).
2. **Aggregate metric on a clean holdout.** Overall AUC on an undegraded 20%
   holdout is the least sensitive thing we could measure; representation/label
   damage shows up in **worst-group** accuracy and **calibration**, not the mean.

**This arm was redesigned (item 1) and now yields a real result — see Finding 7.**
The aggregate-AUC-only arm is superseded; keep this section only as the record of
why aggregate AUC on a clean holdout was the wrong instrument.

## Finding 7 — Redesigned downstream arm: label noise wrecks CALIBRATION, and the gate does not recover it

Source: `experiments/results/downstream_{healthcare,finance,government}.csv`
(real data, 5 seeds × 3 severities, `git=43759f7`), summarized in
`downstream_summary.csv`; figures **figD1/figD2/figD3**. Produced by
`experiments/run_downstream.py` + `experiments/analyze_downstream.py`. Four training
conditions per cell: `clean_only` (= **oracle** ceiling), `all_ungated` (diluted),
`corrupt_only` (**undiluted**), `gate_realistic` (what the gate kept); metrics on the
clean holdout are AUC, accuracy, **worst-group accuracy**, **subgroup FNR gap**, and
**expected calibration error (ECE)**.

**(a) The sensitive metric is calibration (ECE), and it exposes harm AUC hides.**
Under undiluted label corruption:

| Domain | ECE clean→corrupt | AUC clean→corrupt | worst-group acc clean→corrupt |
|---|---|---|---|
| healthcare | 0.032 → **0.288** (9×) | 0.856 → 0.728 | 0.955 → 0.955 |
| government | 0.046 → **0.288** (6×) | 0.928 → **0.927** | 0.911 → 0.879 |
| finance | 0.066 → 0.077 | 0.669 → 0.663 | 0.502 → 0.502 |

The **government** row is the headline: label noise leaves AUC essentially unchanged
(0.928→0.927) while calibration collapses 6× — exactly the failure the old arm was
blind to. **Worst-group accuracy is mostly flat on real data** (an honest correction
to the synthetic smoke, where it moved); ECE is the metric that carries the signal.

**(b) The gate does not recover label harm — the spine, confirmed downstream.**
For label, `gate_realistic` ≈ `all_ungated` (the gate keeps the label-corrupted
sources, Finding 2), so residual ECE stays high (healthcare 0.105, government 0.120,
vs clean 0.03–0.05) and worst-group accuracy stays at the ungated value
(government 0.902 < oracle 0.911). A miscalibrated model ships.

**(c) The gate DOES recover structural harm to the oracle ceiling.** Government
structural: undiluted corruption drops worst-group accuracy 0.911→0.853 and AUC
0.928→0.871, and the realistic gate **recovers worst-group accuracy to 0.911 = the
oracle** (vs 0.905 ungated). This is the clearest positive: where the gate withholds
(structural), it recovers the downstream metric to the best achievable.

**(d) Item 2 — harm scales with severity AND with dilution (figD3, figD4).** ECE of
the undiluted label model climbs low→med→high: healthcare 0.147 → 0.287 → 0.431,
government similar; all other modes stay on the clean baseline (figD3). And the
diluted (all-sources) model's label ECE rises with the **number** of corrupted
sources — a 2-point curve (n_degraded ∈ {1,2}): government 0.080 → 0.120, healthcare
0.064 → 0.105, finance ~flat 0.066 → 0.074 (figD4). Together these quantify the
2-of-6 dilution that flattened the old aggregate-AUC arm. (Run `--n-degraded 3` to
extend the dilution curve; `figD4` redraws automatically.)

**Net:** Blocker 1 is resolved. The downstream story is now a real, honest result —
*label noise is a calibration failure invisible to AUC; the gate recovers the
degradations it withholds (structural) and fails to recover those it only flags
(label)* — not an uninformative null.

---

## Finding 8 — A confident-learning detector closes the label blind spot (Blocker 2 fixed)

Source: `experiments/run_label_detector.py` + `src/domains/label_quality.py`
(extension — the frozen gate is untouched), analyzed by
`experiments/analyze_label_detector.py` → `label_detector_summary.csv`, figures
**figL1/figL2**. The new detector is a per-source **cross-validated
self-disagreement rate** (out-of-fold predicted class ≠ given label): a rows-free
scalar, DP-noisable like the gate's other released statistics, so it fits the
private/federated architecture. A flipped source keeps disagreeing with its own
labels even when its positive-rate marginal does not move — so it is visible where
the marginal detector is blind.

**(a) Symmetric noise (the blind spot) — closed.** Self-disagreement separates clean
from 45 %-flipped sources in every domain, including balanced finance: clean 0.390
(±0.003 at ~6.7k rows/source) vs noisy 0.489 — a +0.10 gap the marginal cannot see.
Smoke recall (marginal → confident): **finance 0.00 → 0.67, healthcare 0.75 → 1.00,
government 0.75 → 1.00** (real-data numbers pending the human run; figL1). The
consensus threshold (`0.05`) is calibrated on **clean** sources — like
`divergence_threshold`, not on the detection outcome (no leakage).

**(b) Asymmetric noise — the marginal regains signal (derivation confirmed).** With
positives→negatives flips the marginal moves (`p' = p − p·rate`), so the gate's own
label-rate detector recovers recall as the flip rate grows (finance symmetric 0.00 →
asymmetric ~0.13 on smoke; figL2). This turns the binary miss into a
detectability-threshold curve and validates the Finding-1 derivation empirically.

**Honest caveats:** finance's high base error (AUC ~0.67) makes its clean/noisy gap
the smallest, so its confident-learning recall is the lowest of the three and the
most threshold-sensitive; asymmetric-noise detection on rare-label domains
(healthcare base 0.06) is weak for *both* detectors, because flipping the few
positives barely changes anything. A rigorous DP analysis of a CV-derived statistic
is future work (the module ships an approximate `epsilon` option, flagged as such).
**Run for the paper:** the human runs `run_label_detector.py` on real data (command
in `docs/NEXT_EXPERIMENTS.md` §3); `analyze_label_detector.py` then draws figL1/L2.

## Finding 9 — The spine is a FIXABLE rule choice — but the fix trades DP-robustness

Quoted from `aggregation_summary.csv` (real data, `run_aggregation_ablation.py` →
`analyze_aggregation.py`); **`paper/figures/figA1_gap_by_rule.png`, figA2**. The gate's
decision rule is a weighted mean of three subscores (READY≥0.8); Finding 2's gap is
what that averaging hides. We re-decide the **same reports** under two alternatives —
`max_violation` (worst subscore) and `any_critical` (any failed check → EXCLUDE) — and
measure the detection→action gap and the clean false-positive rate.

**Gap (det_recall − withhold rate), ε=∞:**

| Domain | weighted_mean | max_violation | any_critical |
|---|---|---|---|
| finance | **0.433** | 0.350 | **0.000** |
| government | **0.667** | 0.575 | **0.000** |
| healthcare | −0.083 | −0.083 | −0.250 |

- **`any_critical` closes the gap to 0.000** on the two domains where composite masking
  bites (finance, government), and `max_violation` is only a *partial* fix — the single
  failed check hidden inside pass_rate=0.93 escapes it (the two-layer masking).
- **Healthcare's gap is already ≤0** — its degradations trip *other* checks, so the
  composite already withholds them (action 0.75 > detection 0.667); the masking is
  domain/mode-specific, worst on government/finance.
- **The honest cost — and the real finding (figA2).** Clean false-positive rate by rule
  × ε (all domains): weighted_mean stays **0.000 at every ε**; `any_critical` is **0.000
  for ε≥1, 0.011 at ε=0.5, and 0.878 at ε=0.1.** The averaging that *causes* the masking
  is the same averaging that makes the gate robust to DP noise, so the zero-tolerance fix
  reintroduces DP sensitivity. **Recommendation:** `any_critical` closes the gap at no
  clean-FPR cost for **ε≥0.5**; below that, the composite's noise-robustness is worth its
  masking, or a hybrid is needed. The masking is fixable, but not for free — a
  privacy↔aggregation tradeoff, not a strict improvement. No threshold was tuned on the
  outcome (rules fixed a priori; gate thresholds reused).

## Finding 10 — External baselines: the gate's edge is the layer, not the checks

Quoted from `baselines_summary.csv` (`baselines_detection.csv` from the headless Docker
run, `experiments/baselines/run_baselines.py`, py3.11 + Spark 3.5) + `crossdomain_*.csv`;
**`paper/figures/figBL_gate_vs_baselines.png`**. **Great Expectations (expectation-style)
and Deequ (constraint-style) ran; TFDV did not** (its `pyfarmhash`/`apache-beam[gcp]`
build fails on slim py3.11 — omitted for packaging, not a result). Both baselines used
**reference-derived** thresholds (GE: existence, not-null, `between` on feature quantiles,
a label mean-band; Deequ: completeness + feature mean-bands, **no label constraint**) —
reasonably-tuned, not zero-config.

**Detection recall (severity=high, ε=∞):**

| Mode | gate (hc/fin/gov) | GE (hc/fin/gov) | Deequ (hc/fin/gov) |
|---|---|---|---|
| structural | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 |
| completeness | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 |
| distributional | 1.0 / 0.8 / 1.0 | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 |
| label | 1.0 / **0.0** / 1.0 | 1.0 / **0.2** / 1.0 | **0.0 / 0.0 / 0.0** |

**Clean false-flag rate (mode=none):** gate 0.000/0.000/0.000 · GE **0.333**/0.067/0.000 ·
Deequ 0.000/0.000/0.000.

- **On structural/completeness all three tie at 1.0** — off-the-shelf validators are built
  for exactly these; **we claim no raw-check novelty**, and this makes that precise.
- **A precision↔recall split across the two baseline families.** Deequ (constraints) is the
  most **precise** — 0.000 clean false-flag everywhere — but with a completeness+distribution
  constraint set it flags **no** label corruption (0.0 all domains). GE (expectations) catches
  more (rare-label 1.0) but pays a **false-flag cost** (a third of clean healthcare sources).
  The gate sits at the good corner of both: rare-label recall 1.0 (hc/gov) **and** 0.000
  clean false-flag.
- **Nobody catches balanced-label symmetric noise on finance** (gate 0.0, GE 0.2, Deequ 0.0):
  a symmetric flip moves no marginal, so every threshold/constraint baseline is blind — the
  exact blind spot the **confident-learning detector (Finding 8)** closes (finance 0.63).
  This is the paper's point: the contribution is not a better single check, it is the
  cross-source self-disagreement statistic + the DP-noised operation (Finding 3) + the
  readiness decision (Finding 9).
- **Caveats (honest):** Deequ's label 0.0 partly reflects the adapter (no label constraint,
  symmetric to GE's mean-band would still miss balanced finance); both baselines used tuned
  thresholds; **TFDV absent** (packaging). `analyze_baselines.py` reports absent baselines as
  ABSENT and redraws the figure for whatever subset ran.

## Finding 11 — Thresholds do not transfer: recall survives, clean false-flags don't

Quoted from `threshold_transfer_summary.csv` (`run_threshold_transfer.py` →
`analyze_threshold_transfer.py`); **`paper/figures/figT1_threshold_transfer.png`**. We
calibrate the distributional `divergence_threshold` on one domain and apply it, unchanged,
to another (distributional mode, ε=∞).

**Detection recall** (row=target, col=threshold-source):

| target ↓ / thr → | finance | government | healthcare |
|---|---|---|---|
| finance | 0.73 | 1.00 | 1.00 |
| government | 1.00 | 1.00 | 1.00 |
| healthcare | 0.00 | 0.00 | 0.33 |

**Clean false-flag** (same layout): finance 0.00/0.00/**0.98** · government 0.67/0.67/**1.00** ·
healthcare 0.00/0.00/0.00.

- **Recall is roughly portable** — mean 0.69 with a domain's own threshold vs 0.67 transferred.
- **The cost lands on precision:** applying **healthcare's tight threshold** to finance/government
  drives the clean false-flag rate to **0.98 / 1.00** — nearly every clean source flagged. Mean
  clean false-flag is 0.22 (own) → **0.44 (transferred)**.
- **So thresholds are NOT portable**; the "industry-agnostic" claim must be stated as *each domain
  needs its own clean-source calibration*, and the retuning cost is a false-positive explosion, not
  a recall loss. Quantifies Finding F (σ=0.34). (Government's own-threshold false-flag 0.67 is
  itself high — its distributional detector is precision-limited even in-domain, consistent with
  fig3's ~0.56 government precision cap.)

## Finding 12 — The confident-learning detector is DP-robust (and its threshold is calibrated)

Quoted from `label_epsilon_summary.csv` (`run_label_epsilon.py` →
`analyze_label_epsilon.py`); **`paper/figures/figE1_confident_epsilon.png`**. ε-sweep of the DP
budget on the self-disagreement statistic, symmetric label noise.

**Confident-detector recall by ε:**

| Domain | ε=0.1 | 0.5 | 1 | 5 | ∞ |
|---|---|---|---|---|---|
| finance | 0.57 | 0.60 | 0.63 | 0.63 | 0.63 |
| government | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| healthcare | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

- **Recall barely degrades under DP:** finance 0.63→0.57 from ε=∞ to ε=0.1; healthcare/government
  hold **1.00** all the way to ε=0.1. This is the key contrast with Finding 3, where the
  *divergence* detector's clean false-flag exploded to ≈1.0 at ε=0.1. **The confident-learning
  detector is the DP-robust label detector** — its statistic is a bounded proportion (sensitivity
  1/n), so Laplace noise at ε=0.1 is small relative to the ~0.10 clean-vs-noisy gap.
- **Calibration (3.3-iii):** clean self-disagreement std is tiny — **finance 0.0073**, government
  0.0016, healthcare 0.0013 — all far below the 0.05 flag threshold, so 0.05 separates clean from
  noisy with wide margin in every domain (finance, the sensitive one, is widest and still 7× under).
- **Still open (3.3-i):** a rigorous DP accounting of the CV-derived statistic (the fold-models are
  treated as post-processing; the released proportion carries the ε). Reported as approximate, not
  overclaimed.

## Finding 13 — Pillar A reliability: the coarse filter holds, the fine coding does not

Quoted from `taxonomy_agreement.txt` (`code_agreement.py`, n=30 double-coded: machine
`code_incidents.py` vs an independent human coding transcribed from
`docs/taxonomy_sample_for_human_review.md`). **This is an honest negative on the
fine-grained coding — reported, not hidden.**

- **Fine coding is NOT reliable.** Assigned **mode**: percent agreement 0.233, **Cohen's
  κ = 0.082**; pre-training **detectability**: agreement 0.333, **κ = 0.038** — both at or
  near chance (Landis–Koch "slight"). Mechanism: the machine coder systematically
  **over-assigns** `distributional` (14/30 vs the human's 6) and `partial` detectability
  (24/30 vs 9); that marginal skew collapses κ.
- **The coarse filter IS reliable.** The human judged **28 of 30** sampled incidents to be
  genuine data-readiness failures (2 "other") → **filter precision 0.933**. So the automated
  *is-this-a-data-readiness-failure* screen (`code_incidents.py`) is trustworthy even though
  the downstream *which-mode / how-detectable* labels are not.
- **Implication for the paper (scope honestly):** do **not** claim the fine mode proportions
  or the machine-coded "12% detectable" as a reliable measurement — a second coder does not
  reproduce them. Claim only (a) the filter precision (0.93, human-validated) and (b) the
  taxonomy as a *qualitative* organizing frame, with the low κ stated as a limitation.
  AIID reports are media-derived and root cause is often under-determined, which is exactly
  why fine coding is subjective; this is consistent with the "undeterminable rate" caveat.
- **Caveats:** n=30, a single human coder, no adjudication; κ has a wide interval at this n.
  Future work: coding guidelines + adjudication, or collapsing to the modes the human can
  reproduce.
- **Intra-rater pass (DONE) — the fine coding is unstable even within one coder.** A blind
  re-code of the same 30 incidents by the same coder after a >3-week washout
  (`taxonomy_agreement_intra.txt`, via `code_agreement.py --intra`; pass-2 from the reshuffled
  `docs/taxonomy_sample_pass2_blind.md`, seed=202) agrees **no better than chance with pass 1**:
  **mode** percent agreement 0.100, **Cohen's κ = −0.048**; **detectability** agreement 0.233,
  **κ = −0.144**. So the near-chance *inter*-rater κ above is **not** merely a machine-vs-human
  artifact — the fine mode/detectability scheme is intrinsically subjective and not reproducible
  even by a single rater over time. This *strengthens* the scoping decision: rely only on the
  human-validated **filter precision (0.93)** and treat the taxonomy as a qualitative frame.
  (The inter-rater κ above is unchanged; the intra result is written to a separate file.)

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
  on. Bars hug zero — but read Finding 5: they hug zero because the corruption never
  moved the model (underpowered arm), *not* because gating is proven harmless.
- **Fig 6 — the spine.** For each failure mode (and the overall mean), the blue bar
  is how often a bad source is *detected* and the orange bar how often it is
  *withheld*; the red arrow is the gap. Everything except structural has a large
  gap — the gate sees the problem but its composite score lets the source through.
- **Fig D1 — downstream calibration harm (item 1).** Per domain, red = how much
  *undiluted* corruption worsens calibration (ECE) vs clean; blue = residual harm in
  the realistic gated pipeline. Only label has tall bars, and its blue bar stays up —
  the gate keeps the label sources, so a miscalibrated model ships.
- **Fig D2 — worst-group-accuracy recovery (item 1).** Three bars per mode: ungated,
  gate-kept, oracle (clean-only). When the gate bar reaches the black oracle bar it
  fully recovered — government structural is the clear case; label does not recover.
- **Fig D3 — severity curve (item 2).** ECE of the undiluted model vs corruption
  severity. The label line climbs low→med→high; every other mode hugs the clean
  baseline. This is the harm-vs-severity curve the single-point arm couldn't produce.

---

## Summary for the paper

The honest framing is a **detectability-boundary study**, not "we built a gate and
it works": *what pre-training data validation provably can catch, what it provably
cannot, and why "flagged" and "blocked" diverge.*

1. **Spine — detection ≠ action:** the gate detects 0.76 of degraded sources but
   withholds 0.42 (0.34 gap; 41 % of cells detected-but-kept), because a composite
   readiness score averages away single-dimension defects. Reported as two separate
   metrics everywhere (Finding 2, fig6).
2. **Catches and transfers:** structural (perfectly, σ=0.00 across domains) and
   completeness (well) — the dependable, domain-agnostic guarantees.
3. **Catches but domain-dependent:** distributional — an ε sweep with a usable knee
   at ε ≥ 0.5; needs per-domain threshold calibration (does not transfer cold).
4. **Provably misses:** symmetric label noise on **balanced** targets — a derived
   result (`p'=p+r(1−2p)→0` as `p→0.5`), not a strawman graph (Finding 1).
5. **DP-noise ↔ false-positive tradeoff:** validating on differentially-private
   summaries trades precision for recall as ε falls — the most technically original
   piece; reported as FPR/FNR-vs-ε with a stated usable range (Finding 3).
6. **Downstream (redesigned, item 1):** aggregate AUC was the wrong instrument
   (Finding 5). With subgroup/calibration metrics, **label noise is a calibration
   failure invisible to AUC** (government ECE 0.046→0.288 at AUC 0.928→0.927), it
   **scales with severity**, and the gate **recovers the modes it withholds
   (structural → oracle) but not those it only flags (label)** (Finding 7,
   figD1–D3). The dilution sweep is the remaining piece.

All are quoted from the CSVs named above and reproducible via
`experiments/analyze.py`. Open items and the ranked plan to close them are in
[`docs/NEXT_EXPERIMENTS.md`](NEXT_EXPERIMENTS.md); limitations are cross-listed in
[`docs/WALKTHROUGH.md`](WALKTHROUGH.md) and
[`verification/VERIFICATION.md`](../verification/VERIFICATION.md).
