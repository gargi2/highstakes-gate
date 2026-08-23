# Next experiments — ranked plan to close the review blockers

Prioritized after **verifying each blocker against the committed CSVs** (see
`docs/FINDINGS.md` and `verification/VERIFICATION.md` V8). Ordering is by
value-per-effort for a submittable **detectability-boundary** paper. Items 1–2 are
gating: until they run, the downstream story and the label story are not yet
defensible. Heavy grids are run by the human (CLAUDE.md hard rule 4); the assistant
drafts and smoke-tests the code first.

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
**Still open:** (i) a rigorous DP analysis of the CV-derived disagreement statistic
(module ships an approximate `epsilon` option, flagged); (ii) an ε-sweep of the
confident detector for the privacy–utility story; (iii) per-domain threshold
calibration on clean sources (finance is the sensitive one).

## 4. 🟡 Ablate the aggregation rule (the spine's "fix")
**Directly tests the paper's spine (Finding 2 — the 0.34 detection–action gap).**

In `src/gate/readiness.py`, compare decision rules on the *same* reports:
- current **weighted mean** (0.4/0.3/0.3, READY≥0.8);
- **max-violation** (worst subscore drives the decision);
- **any-critical-fails** (any schema/null-critical check → EXCLUDE).

Report, per rule: the detection–action gap, `degraded_withheld`, **and** the clean
false-positive rate. The claim to support: max-violation / any-critical closes the
gap **without** an FPR explosion — i.e. the masking is a fixable design choice, not
inherent. Do NOT tune thresholds on the outcome (leakage; keep the sensitivity
analysis the file's docstring already promises).

## 5. 🟡 Cold threshold transfer (the industry-agnostic claim)
- Calibrate `divergence_threshold` on **one** domain, apply **unchanged** to the
  other two; measure the detection/FPR cost vs each domain's own calibrated value.
- If thresholds don't transfer (expected from the σ=0.34 distributional spread),
  **quantify the retuning cost** — still a result, and the honest form of the
  cross-domain claim.

## 6. 🟡 Baselines (avoid desk-reject)
Run **TFDV**, **Deequ**, and **Great Expectations** defaults on the same degraded
sources; report per-mode detection vs the gate. Framing: the contribution is the
DP-private + cross-source + readiness-decision layer, **not** raw check novelty —
baselines make that precise.

## 7. 🟢 Confidence intervals (the grid already has 5 seeds)
Add bootstrap / across-seed **95 % CIs** to every headline number and error bars to
every figure in `experiments/analyze.py`. No new runs needed — the seeds exist.

## 8. 🟢 Pillar A — run the reliability pass that already exists
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
  no rows in prompts/outputs, CLAUDE.md). Confirm in writing before any extract text
  or row-derived number enters the paper.
- **Petition framing:** any brief citing this work must carry the
  measurement-and-assurance framing; "the gate did not improve accuracy" without it
  reads as no-impact to a non-specialist adjudicator.
