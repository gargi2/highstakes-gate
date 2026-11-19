# Next experiments — ranked plan to close the review blockers

Prioritized after **verifying each blocker against the committed CSVs** (see
`docs/FINDINGS.md` and `verification/VERIFICATION.md` V8). Ordering is by
value-per-effort for a submittable **detectability-boundary** paper. Items 1–2 are
gating: until they run, the downstream story and the label story are not yet
defensible. Heavy grids are run by the author (CONVENTIONS.md hard rule 4); the code
is drafted and smoke-tested first.

Status legend: 🔴 blocks submission · 🟡 strengthens · 🟢 polish.

---

## 1. ✅ IMPLEMENTED — Redesign the downstream arm (subgroup metrics + undiluted signal)
**Closes Blocker 1 (verified: clean ≈ corrupted-ungated, max ΔAUC 0.0045 — the old
arm measured nothing).** Built as a NEW, focused script so the detection grids are
untouched:

- `experiments/run_downstream.py` — four training conditions per cell:
  `clean_only` (= oracle ceiling, the undegraded sources), `all_ungated` (diluted
  floor), `corrupt_only` (**undiluted** damage), `gate_realistic` (what the real
  gate kept). One CSV per domain: `downstream_<domain>.csv`.
- `src/domains/metrics.py` — numpy-only **worst-group accuracy**, **subgroup FNR
  gap**, **ECE**, alongside AUC/accuracy; subgroups = quantile bins of an equity
  axis (`Domain.subgroup_column`, default `feature_columns[0]`: healthcare=age,
  finance=log_assets, government=log_amount) on the **clean holdout**.
- `tests/test_downstream.py` — 7 tests guard the metrics + four-condition wiring.

**Smoke evidence (synthetic — not for the paper, but confirms sensitivity):** on
`--smoke`, label corruption drops worst-group accuracy by 0.49 (healthcare) /
0.43 (government) and pushes ECE from ~0.02 to ~0.34 between `clean_only` and
`corrupt_only`, while aggregate AUC barely moved — exactly the metric the old arm
was blind to. `gate_realistic == all_ungated` for label (gate didn't withhold it),
so it does not recover — consistent with the spine.

**The human runs the real grid** (heavy I/O — reads each extract once/domain):
```bash
.venv/Scripts/python.exe experiments/run_downstream.py --seeds 5 --max-rows 40000 \
  --data-root healthcare=data/healthcare --data-root finance=data \
  --data-root government=data/PGYR2025_P06302026_06032026
```
Then: read `downstream_*.csv` and report, per mode/domain, the `clean_only −
corrupt_only` gaps in worst-group accuracy / FNR / ECE, and whether `gate_realistic`
recovers toward `clean_only`. If those gaps move, the null becomes an honest
**positive**; if not, report "no downstream effect even with sensitive metrics" — a
real negative, not the current uninformative one. (A figure/summary for `analyze.py`
is deferred until the real CSVs exist, to avoid plotting synthetic smoke.)

**Still open (do next within this item):** the severity×dilution sweep + oracle is
item 2; `run_downstream.py` already accepts `--n-degraded` and `--severities` to
drive it.

## 2. 🟡 Severity × dilution sweep + oracle ceiling — PARTLY DONE
**Also Blocker 1; turns single points into severity–detectability curves.**

- **Severity curve: DONE.** `run_downstream.py` already varies `--severities`
  {low,med,high}; `analyze_downstream.py` plots ECE-vs-severity (`figD3`). Result:
  label calibration harm scales monotonically (healthcare ECE 0.147→0.287→0.431),
  all other modes flat. (`docs/FINDINGS.md` Finding 7d.)
- **Oracle ceiling: DONE.** The `clean_only` condition *is* the omniscient gate
  (excludes exactly the degraded sources); `gate_realistic` is measured against it
  (`figD2`). Government structural recovers to the oracle; label does not.
- **Dilution sweep: DONE (2-point, extendable).** Ran `--n-degraded 1` into
  `experiments/results/dilution_nd1/`; combined with the nd=2 baseline,
  `analyze_downstream.py` drew `figD4`: diluted label ECE rises with #corrupted
  sources (government 0.080→0.120, healthcare 0.064→0.105; finance ~flat). Run
  `--n-degraded 3` into `dilution_nd3/` to add the third point. Command:
```bash
# repeat for --n-degraded 1 and 3 (keep <50% of n_sources=6 for consensus),
# writing to distinct dirs, then concatenate the CSVs before analyze_downstream.
.venv/Scripts/python.exe experiments/run_downstream.py --seeds 5 --max-rows 40000 \
  --n-degraded 1 --out-dir experiments/results/dilution_nd1 \
  --data-root healthcare=data/healthcare --data-root finance=data \
  --data-root government=data/PGYR2025_P06302026_06032026
```
  Run all three levels (1, 2, 3) into `dilution_nd{1,2,3}/` with the current code so
  every file carries the `n_degraded` column, then point `analyze_downstream` at the
  concatenation. (The committed `downstream_*.csv` — used for figD1–D3 — predates the
  column and is n_degraded=2; keep it for the item-1 figures, don't mix it into the
  dilution concat.)

## 3. ✅ IMPLEMENTED — Asymmetric label noise + a confident-learning detector
**Closes Blocker 2 (verified arithmetic: `p'=p+r(1−2p)→0` as `p→0.5`).** Built as an
extension; the frozen gate is untouched:

- `src/domains/degradations.py::degrade_labels_asymmetric` — class-conditional flip
  (positives→negatives). Marginals move (`p'=p−p·rate`), so the gate's label-rate
  detector regains recall as asymmetry grows (figL2) — the derivation, confirmed.
- `src/domains/label_quality.py` — the confident-learning detector: per-source
  cross-validated **self-disagreement rate** (rows-free scalar, DP-noisable via
  `gate.privacy.laplace_noise`) + a one-sided consensus check
  (`label_quality_checks`, threshold 0.05 calibrated on CLEAN sources).
- `experiments/run_label_detector.py` + `experiments/analyze_label_detector.py`
  (figL1/L2 + summary); `tests/test_label_quality.py` (7 tests).

**Smoke evidence (synthetic; real run pending):** symmetric-noise recall marginal →
confident: **finance 0.00 → 0.67, healthcare/government 0.75 → 1.00** — the blind
spot closed, precision ~1.0. Self-disagreement clean vs 45 %-noisy: finance
0.390±0.003 → 0.489.

**The human runs the real grid** (per-source CV is heavier than the marginal detector,
but the frame cache reads each extract once):
```bash
.venv/Scripts/python.exe experiments/run_label_detector.py --seeds 5 --max-rows 40000 \
  --data-root healthcare=data/healthcare --data-root finance=data \
  --data-root government=data/PGYR2025_P06302026_06032026
.venv/Scripts/python.exe experiments/analyze_label_detector.py
```
**Still open (3.3):** (i) a rigorous DP analysis of the CV-derived disagreement statistic
(analysis/writing — module ships an approximate `epsilon` option, flagged); (ii) ε-sweep +
(iii) per-domain clean-disagreement calibration are now BUILT in
`experiments/run_label_epsilon.py` (V18). Human runs the real grid:
```bash
.venv/Scripts/python.exe experiments/run_label_epsilon.py --seeds 5 --max-rows 40000 \
  --data-root healthcare=data/healthcare --data-root finance=data \
  --data-root government=data/PGYR2025_P06302026_06032026
```
Report confident-detector recall vs ε (does it survive DP noise?) and `clean_disagree_std`
per domain (the calibration basis for the 0.05 threshold; finance is the sensitive one).

## 4. ✅ DONE (real data + analyzed) — Ablate the aggregation rule (the spine's "fix")
**Result (Finding 9):** `any_critical` closes the gap to 0.000 on finance/government at
**zero clean-FPR for ε≥0.5**, but clean-FPR hits **0.878 at ε=0.1** — the fix is
DP-noise-fragile (privacy↔aggregation tradeoff). `max_violation` only partial.
Figures figA1/figA2, `aggregation_summary.csv`. Analyzer: `analyze_aggregation.py`.

**Directly tests the paper's spine (Finding A — the 0.34 detection–action gap).**
Built as an additive extension; the frozen gate (`src/gate/`) is untouched
(`git diff src/gate/` empty).

- `src/domains/aggregation.py::decide` — three rules on the SAME reports:
  **weighted_mean** (= `score_source` verbatim, the reference), **max_violation**
  (worst-of-three-subscores), **any_critical** (any single failed check → EXCLUDE).
- `experiments/run_aggregation_ablation.py` — per rule: `det_recall` (rule-independent
  detection), `action_rate` (decision≠READY), `gap = det_recall−action_rate`,
  `clean_fpr` (clean sources not marked READY). One CSV per domain.
- `tests/test_aggregation.py` — 6 tests; incl. `weighted_mean`≡`score_source`, and the
  two-layer-masking property (a single failed check hidden in pass_rate=0.93 escapes
  both weighted_mean AND max_violation, but not any_critical).

**Smoke evidence (synthetic — not paper numbers):** mean gap (degraded) / clean_fpr
(mode=none) per rule — healthcare 0.250/0.000 → 0.250/0.000 → 0.000/0.100; finance
0.500/0.000 → 0.000/0.000 → 0.000/0.000; government 0.750/0.000 → 0.500/0.000 →
0.000/0.000 (weighted_mean → max_violation → any_critical). `max_violation` partially
closes the gap; `any_critical` fully closes it at a small FPR cost — the "fixable
design choice" story, to confirm on real data. No threshold tuned on the outcome
(rules fixed a priori; gate thresholds reused).

**The human runs the real grid:**
```bash
.venv/Scripts/python.exe experiments/run_aggregation_ablation.py --seeds 5 --max-rows 40000 \
  --data-root healthcare=data/healthcare --data-root finance=data \
  --data-root government=data/PGYR2025_P06302026_06032026
```
Then report, per rule/domain: the gap at ε=∞ and the clean_fpr, and whether
max_violation / any_critical close the gap **without** an FPR explosion. (Analyzer +
figure deferred until the real CSVs exist, to avoid plotting synthetic smoke.)

## 5. ✅ DONE (real + analyzed) — Cold threshold transfer (3.2)
**Result (Finding 11):** recall transfers (own 0.69 vs transferred 0.67) but clean
false-flag does NOT — healthcare's tight threshold on finance/government → 0.98/1.00.
Thresholds not portable; retuning cost = false positives. figT1, `threshold_transfer_summary.csv`.
Analyzer: `analyze_threshold_transfer.py`. (Runner below.)

### (runner)
`experiments/run_threshold_transfer.py` — calibrate `divergence_threshold` on one
domain, apply unchanged to the others; per (target × threshold-source): distributional
det_recall + clean false_flag (diagonal = own threshold). Smoke-tested (V18). Human runs:
```bash
.venv/Scripts/python.exe experiments/run_threshold_transfer.py --seeds 5 --max-rows 40000 \
  --data-root healthcare=data/healthcare --data-root finance=data \
  --data-root government=data/PGYR2025_P06302026_06032026
```
Report the 3×3 recall/false_flag matrix; the off-diagonal retuning cost is the honest
form of the cross-domain claim (expected non-transfer from σ=0.34).

## 6. ✅ DONE (GE + Deequ real + analyzed; TFDV omitted) — Baselines
**Result (Finding 10):** GE + Deequ both tie the gate at 1.0 on structural/completeness.
Precision↔recall split: Deequ 0.000 clean false-flag but 0.0 on label (no label
constraint); GE catches rare-label but false-flags (hc 0.333); the gate has both. **All
three miss balanced-finance label** (0.0/0.2/0.0) — the confident-learning detector
(Finding 8) is the only thing that catches it (0.63). Figure figBL (3 series),
`baselines_summary.csv`. Analyzer: `analyze_baselines.py`. **TFDV omitted** (packaging;
footnote it). Ran headless via the Docker command above. **Still pending:** get TFDV + Deequ (they need Python ≤3.11; Colab is 3.13, Kaggle 3.12).
Reliable path is the headless runner `experiments/baselines/run_baselines.py` in a
py3.11 Docker container (from repo root, Docker Desktop running):
```powershell
docker run --rm -v "${PWD}:/repo" -w /repo python:3.11-slim bash -lc "apt-get update && apt-get install -y default-jre-headless >/dev/null && pip install great_expectations==0.18.19 pydeequ pyspark==3.5.1 && SPARK_VERSION=3.5 python experiments/baselines/run_baselines.py"
```
Notes: **pydeequ needs Spark 3.5/3.3/3.2 (NOT 3.4)** — hence pyspark==3.5.1 +
SPARK_VERSION=3.5. This command gives GE+Deequ (two baseline families). **TFDV is
omitted** — its `apache-beam[gcp]` dep makes pip backtrack through the whole
google-cloud SDK and it fights py3.11; footnote it as omitted for packaging, or add
`(pip install 'apache-beam[gcp]==2.53.0' tensorflow-data-validation==1.15.1 || echo skip) &&`
before the run to attempt it non-fatally.
Writes `experiments/results/baselines_detection.csv` (all three), then re-run
`analyze_baselines.py`. Synthetic only (no data/ read). Grid skips any baseline that
fails to import; check the "baselines with results" line it prints.

Run **TFDV**, **Deequ**, and **Great Expectations** defaults on the same degraded
sources; report per-mode detection vs the gate. Framing: the contribution is the
DP-private + cross-source + readiness-decision layer, **not** raw check novelty —
baselines make that precise.

- `experiments/baselines/gen_baseline_sources.py` — regenerates the gate's EXACT
  per-cell sources (same seeds/splits/degrade as `run_crossdomain`), so baselines
  validate byte-identical inputs. `tests/test_baseline_sources.py` (5 tests) guards it.
- `experiments/baselines/baselines_colab.ipynb` — installs GE (pinned 0.18.19) +
  TFDV + Deequ (pydeequ/Spark), runs each baseline's default checks per source,
  writes `experiments/results/baselines_detection.csv` (per-mode `det_recall` +
  `false_flag_rate`, aligned with the gate), and prints a gate-vs-baseline table.

**Why Colab:** TFDV/Deequ don't run on the author's Windows box (TFDV is Linux-wheel
only; Deequ needs a JVM). The notebook defaults to **synthetic** data — healthcare
MUST stay synthetic (MIMIC DUA); finance/government public extracts are optional via
`DATA_ROOT`. **The baseline cells are not pre-run** — expect minor GE/Deequ
API fixups on first run. Expected finding: baselines match the gate on
structural/completeness (no raw-check-novelty claim) and fall behind on
distributional/label — the latter is the symmetric-flip blind spot that Finding D's
confident-learning detector closes.
**Human:** open the notebook in Colab, set `REPO_URL`, Run all, copy
`baselines_detection.csv` back into `experiments/results/`.

## 7. ✅ DONE — Confidence intervals (the grid already has 5 seeds)
Shared helper `experiments/ci_utils.py` (seed-level percentile bootstrap, 10k resamples,
seed=42). Wired into: **fig6** (spine — error bars + the gap CI in-caption), **figA1**
(ablation gap), **figBL** (baselines), and CI columns (`det_recall_ci_*`,
`withheld_frac_ci_*`) in `metrics_summary.csv`. Headline result:
**detect 0.76 [0.74,0.76], withhold 0.42, gap 0.34 (95% CI [0.32, 0.35])** — the spine
excludes 0 by a wide margin. No new runs (used the existing 5 seeds).
Optional extension: error bars on fig2/fig3/fig4 too (bar/line figures) — same helper.

## 8. ✅ DONE (real κ — an honest negative) — Pillar A reliability pass
**Result (Finding 13):** human codings were transcribed from the filled worksheet
`docs/taxonomy_sample_for_human_review.md` into the CSV; `code_agreement.py` gives
**mode κ 0.082, detectability κ 0.038** (near chance — fine coding NOT reliable; machine
over-assigns distributional/partial) but **filter precision 0.933** (coarse screen IS
reliable). Paper implication: claim the filter precision, not the fine proportions/12%.
**Still pending (human):** an intra-rater blind second pass for the within-coder κ.

The IRR pipeline is built (`experiments/code_agreement.py`: seeded 30-incident
sample, Cohen's κ on mode **and** detectability, filter precision, machine coding as
a 2nd annotator). It is **unrun** only because the human labels file is a template.
To close Blocker 3:
- Fill `experiments/results/taxonomy_sample_human_labels.csv` (blind), run
  `code_agreement.py`, report κ.
- Add an **intra-rater** pass (blind re-annotation separated in time) for sole
  authorship, and report the **undeterminable rate** (AIID reports are media-derived;
  root cause is often unknowable) — quote the 12 % over the *determinable* subset and
  name the selection bias toward high-profile consumer-facing incidents.
- **Position against prior work explicitly** in `docs/TAXONOMY.md`: existing AIID
  failure-cause taxonomies and the MIT AI Risk Repository classify *causes*; this work
  asks the narrower operational question of **pre-training detectability**. Differentiate
  in the intro or a reviewer will cite them.

---

## Not a paper contribution (do separately) — the NIST crosswalk as a deliverable
The MEASURE mapping (`docs/NIST_MAPPING.md`) is **not** a research contribution — NIST
hosts many crosswalks. But a crosswalk **published on a NIST AIRC .gov page** is
government-published evidence, valuable independent of peer review and on a shorter
timeline. If pursued: state every cell at MEASURE **subcategory** granularity (already
done), and defend every cell — one wrong mapping sinks the whole crosswalk.

## Housekeeping before submission
- **Data licenses (verify, don't assume):** SEC EDGAR = U.S. Gov public domain (OK);
  CMS Open Payments = federal Open Data (derived publication OK — confirm the current
  data-use notice); MIMIC-IV = PhysioNet **DUA, aggregate-only** (already enforced —
  no rows in prompts/outputs, CONVENTIONS.md). Confirm in writing before any extract text
  or row-derived number enters the paper.
- **Petition framing:** any brief citing this work must carry the
  measurement-and-assurance framing; "the gate did not improve accuracy" without it
  reads as no-impact to a non-specialist adjudicator.
