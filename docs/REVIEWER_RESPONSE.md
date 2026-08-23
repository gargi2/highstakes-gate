# Response to review — findings and changes

We treated every point in the review as a **hypothesis to check against the actual
result files**, not as a verdict to accept or reject. Below: what we verified, what
we changed, and what we found — every number quoted from a named committed CSV and
reproducible from the scripts listed. Where a diagnosis turned out to be partly
wrong, we say so; where it was right, we fixed it and re-ran.

**Headline reframe.** The paper is now a **data-readiness *detectability-boundary*
study** — what pre-training data validation provably catches, what it provably
cannot, and why a source that is *flagged* is often not *blocked* — not a
"we-built-a-gate-and-it-works" tool paper. Our own results forced this framing, and
it is the more defensible contribution.

---

## 1. Blockers: verdict, evidence, and fix

### Blocker 1 — "the accuracy null may be uninformative." ✅ Confirmed; fixed.
Verified from `experiments/results/crossdomain_*.csv`: the three-way comparison
(clean → corrupted-ungated → corrupted-gated) shows **clean ≈ corrupted-ungated in
every cell** (max ΔAUC **0.0045**, most 0.000). The corruption never dented the
model, so the old aggregate-AUC arm licensed no conclusion. We **retracted** the
earlier "governance-only" reading.

**Fix (built + run on real data):** a redesigned downstream arm
(`experiments/run_downstream.py`, `src/domains/metrics.py`) with four training
conditions — `clean_only` (oracle ceiling), `all_ungated` (diluted),
`corrupt_only` (undiluted), `gate_realistic` — and sensitive metrics: **worst-group
accuracy, subgroup FNR gap, calibration error (ECE)**. Result below (Finding B).

### Blocker 2 — "the finance label miss is arithmetic, a strawman." ✅ Confirmed; fixed.
The injection is a symmetric flip, and for base rate `p` the post-flip positive rate
is `p' = p + r(1−2p)`, which vanishes as `p → 0.5`. Finance `adverse ≈ 0.42`, so the
marginal detector cannot move → recall **0.00** at every ε
(`crossdomain_finance.csv`). We now state this **as a derivation**, not a graph.

**Fix (built + run on real data):** (i) a **confident-learning detector**
(`src/domains/label_quality.py`) — a per-source cross-validated **self-disagreement
rate**, a rows-free scalar that fits the private architecture; (ii) **asymmetric
noise** (`degrade_labels_asymmetric`) that moves the marginal, giving a threshold
curve. Result below (Finding D).

### Blocker 3 — "single annotator, no reliability statistic." ⚠️ Partly incorrect.
An inter-rater pipeline already exists (`experiments/code_agreement.py`): a seeded
30-incident sample, **Cohen's κ** on both assigned mode and pre-training
detectability, filter precision, and a machine coding as a disclosed second
annotator. What is genuinely missing is only the **human coding pass** (the labels
file is still a template). We have written the plan to close it — intra-rater blind
re-annotation and reporting the *undeterminable* rate — and now **position against
prior taxonomies explicitly** (AIID CSET/GMF and the MIT AI Risk Repository classify
*causes*; this work asks the narrower **pre-training-detectability** question) in
`docs/TAXONOMY.md`.

### On the spine you identified — ✅ confirmed as the strongest result (Finding A).

### On the NIST mapping — corrected and scoped.
The crosswalk is **not** claimed as a research contribution. It is now verified
verbatim against **NIST AI 100-1 (AI RMF 1.0), Table 3, pp. 29–31** (fetched from
nvlpubs.nist.gov), stated at **MEASURE subcategory** granularity (2.1, 2.11, …); the
full RMF has 72 subcategories across four functions and we map only MEASURE (21) by
design. Details and the framing caveat (we do **not** claim MEASURE 2.2 for data
completeness — 2.2 is human-subject-evaluation-specific) are in
`docs/NIST_MAPPING.md`.

---

## 2. Findings (each tied to a file/figure)

### Finding A — The spine: composite scoring detects far more than it withholds
`crossdomain_*.csv`; `paper/figures/fig6_detection_action_gap`, `fig2`.
Aggregated over three domains at ε=∞, the gate **detects 0.76** of degraded sources
but **withholds only 0.42** — a **0.34 gap**; **74 of 180 degraded cells (41 %)** are
detected-but-not-withheld. The gap is **0.00 for structural** and **0.44–0.56** for
completeness/distributional/label. Mechanism (from `src/gate/readiness.py`):
readiness = `0.4·pass_rate + 0.3·schema + 0.3·divergence`, READY ≥ 0.8, so one failed
check of ~15 (pass-rate 1.00→0.93) is averaged away → still READY. **A severe
single-dimension defect is masked by a composite score.** Reported as two separate
metrics (detection rate, gate-action rate) throughout.

### Finding B — Downstream: label noise is a *calibration* failure invisible to AUC
`downstream_*.csv` → `downstream_summary.csv`; `figD1/D2/D3`.
Under undiluted label corruption:

| Domain | ECE clean→corrupt | AUC clean→corrupt | worst-group acc |
|---|---|---|---|
| healthcare | 0.032 → **0.288** (9×) | 0.856 → 0.728 | 0.955 → 0.955 |
| government | 0.046 → **0.288** (6×) | 0.928 → **0.927** | 0.911 → 0.879 |
| finance | 0.066 → 0.077 | 0.669 → 0.663 | 0.502 → 0.502 |

The **government** row is the point: calibration collapses 6× while **AUC is
unchanged** — exactly the harm aggregate AUC hid. Worst-group accuracy is mostly
flat on real data (an honest correction to our own synthetic pilot, where it moved);
**ECE is the sensitive metric.** The gate **recovers structural to the oracle**
(government worst-group 0.911 = clean-only, vs 0.905 ungated) but **not label**
(gate = ungated < oracle) — the spine, confirmed downstream.

### Finding C — Harm scales with severity and with dilution
`figD3`, `figD4`. Undiluted label ECE climbs low→med→high (healthcare 0.147 → 0.287 →
0.431); all other modes stay on the clean baseline. Diluted (all-sources) label ECE
rises with the number of corrupted sources (government 0.080 → 0.120, healthcare
0.064 → 0.105 as n_degraded 1→2) — quantifying the 2-of-6 dilution that flattened the
old arm.

### Finding D — A confident-learning detector closes the label blind spot (real data)
`label_detector_*.csv` → `label_detector_summary.csv`; `figL1/L2`. Detection recall
over the corrupted sources, **precision in parentheses**:

| Domain | marginal (symmetric) | confident (symmetric) | marginal (asymmetric) |
|---|---|---|---|
| finance | **0.00** (—) | **0.63** (1.00) | 0.25 (1.00) |
| healthcare | 0.67 (1.00) | **1.00** (1.00) | 0.00 |
| government | 0.67 (1.00) | **1.00** (1.00) | 0.75 (1.00) |

The confident-learning detector **closes the finance symmetric blind spot
(0.00 → 0.63) at perfect precision** and lifts the rare-label domains to 1.00. The
two detectors are **complementary**: symmetric noise → confident-learning;
asymmetric noise → the marginal regains signal (finance 0.00 → 0.25, government 0.75),
confirming the §1 Blocker-2 derivation. The detector works because a source's
cross-validated self-disagreement stays elevated under symmetric noise even when its
positive-rate marginal does not move.

### Finding E — Privacy–utility of DP measurement
`crossdomain_*.csv`; `fig3`. Validating on differentially-private summaries trades
precision for recall as ε falls: clean-data false-flag rate is **0.0 at every ε ≥ 1**
in all domains but **1.0 / 1.0 / 0.63** (healthcare/finance/government) at ε=0.1. The
usable knee is ε ≥ 0.5. This is the most technically original piece and we report it
as an ε sweep, not a single number.

### Finding F — Cross-domain transfer
`crossdomain_*.csv`; `fig1/fig4`. Structural detection transfers perfectly (σ=0.00
across domains); completeness well (σ=0.19); **distributional (σ=0.34) and the
label-rate detector (σ=0.38) do not transfer** — the label-rate detector works on
rare-label domains but is blind on balanced finance (Blocker 2). Each domain needs
its own `divergence_threshold`, calibrated on clean sources.

---

## 3. What we are NOT claiming (honest scope)
- **No downstream accuracy lift.** Gating changes AUC by |Δ| < 0.005 everywhere; its
  value is governance/withholding and calibration, not an accuracy boost.
- **Finance is the weak domain** (AUC ~0.67, worst-group ~0.50; its balanced label
  mutes calibration harm and gives the confident detector its smallest margin) —
  reported, not hidden.
- **The confident-learning statistic's DP** is approximate (the module ships an
  `epsilon` option flagged as such); a rigorous DP analysis of a CV-derived statistic
  is future work.
- **The NIST crosswalk is connective tissue**, not a contribution.

## 4. Still open (planned, `docs/NEXT_EXPERIMENTS.md`)
Aggregation-rule ablation (does max-violation / any-critical close the spine gap
without an FPR explosion?), cold cross-domain threshold transfer with a quantified
retuning cost, external baselines (TFDV / Deequ / Great Expectations), bootstrap CIs
on every headline number, and the Pillar-A human κ pass. None of these change the
findings above; they harden them.

---

### Reproducibility
All figures/tables regenerate from the committed CSVs with fixed seeds:
`experiments/analyze.py` (fig1–6), `experiments/analyze_downstream.py` (figD1–4),
`experiments/analyze_label_detector.py` (figL1–2). The gate (`src/gate/`) is reused
unmodified from FedCheck; every new detector/metric is an additive extension in
`src/domains/`. 45 tests pass. Verification log: `verification/VERIFICATION.md`
(V6–V11). Restricted clinical data (MIMIC-IV, DUA) is used aggregate-only; no rows
appear in any committed file.
