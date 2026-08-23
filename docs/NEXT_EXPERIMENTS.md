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

## 2. 🔴 Severity × dilution sweep + oracle ceiling
**Also Blocker 1; turns single points into severity–detectability curves.**

- Sweep `--n-degraded` ∈ {1,2,3} (keep <50 % for consensus) and reuse the existing
  `severities` {low,med,high} — the grid *already* varies severity, so plot
  detection/withholding/subgroup-harm **vs severity** per mode per domain.
- Add an **oracle** condition: an omniscient gate that excludes exactly the degraded
  sources, to bound the best achievable downstream number (the ceiling the real gate
  is measured against).

Command shape (human runs):
```bash
.venv/Scripts/python.exe experiments/run_crossdomain.py --seeds 5 \
  --n-degraded 1 --data-root ... ; # repeat for 2, 3
```

## 3. 🟡 Asymmetric label noise + a detector that can see symmetric noise
**Closes Blocker 2 (verified arithmetic: `p'=p+r(1−2p)→0` as `p→0.5`).**

- In `src/domains/degradations.py`, add an **asymmetric** label mode (flip only
  positives→negatives, or a class-conditional rate). Marginals *do* move → the
  existing label-rate detector should produce a **detectability-threshold curve**
  vs asymmetry, converting the finance "0.00" into a boundary result.
- Add a **confident-learning / cross-validated-disagreement** detector (out-of-fold
  model disagreement rate per source) as a new consensus check, so symmetric noise
  on balanced targets becomes detectable. Report it beside the marginal detector —
  "here is the hole *and* the fix."
- Keep the derivation in `docs/FINDINGS.md` Finding 1 as the theoretical anchor.

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
