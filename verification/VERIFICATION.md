# Verification log

Per CONVENTIONS.md hard rule 4, the author reviews all code before commit and
records verifications here. Each entry: what was run, the command, and the
outcome. Numbers are quoted from files or command output, never memory.

---

## Session 3 — Pillar B: cross-domain gate harness (2026-08-23)

Code: `src/domains/` (base, healthcare, finance, government, degradations),
`experiments/run_crossdomain.py`, `tests/test_domains.py`. git=`432276d` (parent).

### V1 — full test suite green
```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```
Outcome: **31 passed** (10 gate: test_privacy 5, test_consensus 5; 21 domain:
registry 1, synthetic-schema 3, clean-sources-READY 3, degradation-detection
4×3=12, roc_auc 1, gating 1). The gate in `src/gate/` was NOT modified.

### V2 — smoke on each domain (<2 min total)
```bash
.venv/Scripts/python.exe experiments/run_crossdomain.py --domains healthcare  --smoke
.venv/Scripts/python.exe experiments/run_crossdomain.py --domains finance     --smoke
.venv/Scripts/python.exe experiments/run_crossdomain.py --domains government   --smoke
```
Outcome: 20 cells each; wrote `experiments/results/crossdomain_<domain>_smoke.csv`.
Mean per-mode detection recall on degraded cells: healthcare 0.969, finance
0.875, government 1.000. Full `--smoke` (all three) wall-clock ~3.7 s.

### V3 — loaders emit aggregates only (no rows)
```bash
PYTHONPATH=src .venv/Scripts/python.exe -m domains.healthcare
PYTHONPATH=src .venv/Scripts/python.exe -m domains.finance
PYTHONPATH=src .venv/Scripts/python.exe -m domains.government
```
Outcome: each prints `provenance=synthetic` + a rows-free summary (shapes, label
positive-rate, per-feature null/min/median/max). Confirmed no data row is
printed. Synthetic label base rates: healthcare mortality ≈ 0.056, finance
adverse ≈ 0.417, government disputed ≈ 0.06.

### V4 — privacy-utility gradient sanity check (full grid scale, synthetic)
```bash
.venv/Scripts/python.exe experiments/run_crossdomain.py \
    --modes none distributional --epsilons inf 5.0 1.0 --seeds 3 --n-rows 4000 \
    --out-dir <tmp>
```
Outcome (mean distributional detection **precision** over 3 seeds, all
severities): healthcare inf/5/1 = 0.93 / 0.56 / 0.33; finance = 0.93 / 0.89 /
0.33; government = 0.72(recall)/0.71(prec) at inf, 0.69/0.33 as ε falls. Recall
stays ≈1.0 at ε≥5. Interpretation: the DP-noised consensus divergence detector
loses **precision** (not recall) as the privacy budget tightens — the intended,
measurable tradeoff. Divergence thresholds are calibrated per domain on clean
sources at ε=∞ (healthcare 0.026, finance 0.055, government 0.045) and passed to
`gate.consensus` as a hyperparameter — the gate is unchanged.

### Known, verified findings (not bugs) — see docs/WALKTHROUGH.md §Limitations
- **Label-rate detector is blind to label noise on near-balanced labels.**
  finance `adverse` ≈ 0.42, so flipping 45 % of labels moves the positive-rate
  marginal only ≈0.42→0.49 (< the 0.15 deviation threshold) → recall ≈ 0.
  Rare-label domains (healthcare, government) detect it easily. Encoded in
  `test_domains.py::test_degradation_detection_noiseless`.
- **Extreme distributional shifts leave the shared histogram support** →
  empty noised histogram → NaN divergence → detector goes blind. Severity
  factors are capped in-range (1.7/2.2/2.8) for this reason.
- **cross-source** and **provenance** taxonomy modes are not row-injectable and
  are out of scope for this harness (documented).

### V5 — real-data validation (small grid, aggregates only) — 2026-08-23
Real extracts placed under `data/` (gitignored): EDGAR quarters 2024q1–2026q1
(`num.txt`), CMS Open Payments general payments PGYR2025 (8.6 GB CSV), and the
FedCheck MIMIC-IV full cohort `data/healthcare/mimic-full-cohort.csv.gz`
(546,029 rows, DUA-restricted — used read-only, aggregates only).
```bash
.venv/Scripts/python.exe experiments/run_crossdomain.py \
  --modes none completeness distributional label structural \
  --epsilons inf --seeds 2 --severities high --max-rows 20000 \
  --data-root healthcare=data/healthcare --data-root finance=data \
  --data-root government=data/PGYR2025_P06302026_06032026 --out-dir <tmp>
```
Outcome (ε=∞): **all three domains clean(none) ready=6/6, clean_excluded=0,
false_flag=0.0** (healthcare 0.30 null tolerance and government Food/Beverage
label both verified against real data). Detection recall=1.0 for all modes
except finance/label (balanced-label blind spot). Downstream AUC on real data:
healthcare ≈0.86, finance ≈0.66, government ≈0.92 (all defined). Confirmed
detection ≠ withholding (distributional/label detected but `degraded_withheld`≈0).
The one-time per-domain frame cache (`_FRAME_CACHE`) makes this feasible: the
8.6 GB CSV / EDGAR set is read once per domain, not per cell.

Real-data findings recorded (not bugs):
- MIMIC labs ~23-24 % baseline null → healthcare `null_threshold=0.30`.
- Open Payments dispute rate ≈0.002 % (59 / 3,000,000) → label switched to
  payment-nature = Food-and-Beverage (~0.93 positive on the real 50k head).
- One EDGAR quarter yields only ~1.4k filings with all five tags → finance
  loader reads every `num.txt` under the root (7 quarters ≈ 11k filings).

### Pending (run by the author)
Full grid + real extractions — see `--print-commands` and the commands below.
Results land in `experiments/results/crossdomain_<domain>.csv`.

---

## Session 4 — Pillar C: NIST MEASURE mapping + analysis (2026-08-23)

Deliverables: `docs/NIST_MAPPING.md`, `experiments/analyze.py`,
`experiments/results/metrics_summary.csv`, `paper/figures/fig1…fig5`,
`docs/FINDINGS.md`. Full grids (325 cells/domain, real data) consumed, not rerun.

### V6 — NIST MEASURE subcategories verified against official source
The MEASURE category/subcategory numbers and wording in `docs/NIST_MAPPING.md`
were read directly from **NIST AI 100-1 (AI RMF 1.0), Table 3, pp. 29–31**,
fetched as the official PDF from
`https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf` (verified 2026-08-23) —
not from memory (hard rule 2). MEASURE = 4 categories, 21 Core subcategories
(1.1–1.3, 2.1–2.13, 3.1–3.3, 4.1–4.3). The placeholder subcategory numbers in
`docs/PIVOT_SCOPE.md` §3 and the `TODO-VERIFY` rows in `docs/TAXONOMY.md` are
superseded by `docs/NIST_MAPPING.md`. Note recorded: MEASURE 2.2 is deliberately
NOT claimed for data completeness (2.2 is human-subject-evaluation-specific).

### V7 — analysis reproducible from committed CSVs
```bash
.venv/Scripts/python.exe experiments/analyze.py
```
Reads `experiments/results/crossdomain_{healthcare,finance,government}.csv`
(975 rows total; `git=ada8fc3`, `seed=42`), writes 5 figures (PDF+PNG) and
`metrics_summary.csv` (60 rows). Deterministic — no experiment is rerun. Every
number in `docs/FINDINGS.md` is quoted from `metrics_summary.csv` or the
per-domain CSVs. Headline results confirmed against the CSVs:
- structural detection recall/precision = 1.00, withheld = 1.00, **all domains**;
- finance/label detection recall = **0.00 at every ε** (balanced-label blind spot);
- distributional detection is precision-limited under DP (clean false-flag 0.0 at
  ε≥1 in all domains; 1.00/1.00/0.63 at ε=0.1);
- cross-domain recall spread σ: structural 0.00, completeness 0.19,
  distributional 0.34, label 0.38 (label-rate transfers to rare-label domains,
  divergence does not transfer cleanly);
- `|auc_delta| < 0.005` in every cell (gating is governance, not an accuracy lift);
- `clean_excluded` max = 0 across all three CSVs (no clean source ever excluded).

### V8 — reviewer blockers checked against the CSVs (2026-08-23)
Diagnoses from an external review were **verified against the committed grids**
before acting (not accepted on assertion). Findings, all reproducible from
`crossdomain_*.csv`:
- **B1 (downstream null is uninformative): CONFIRMED.** Three-way AUC at ε=∞ shows
  clean ≈ corrupted-ungated in every cell (max drop 0.0045 healthcare/distributional,
  0.0000 government). Corruption never dented the model (2-of-6 source dilution +
  aggregate metric), so the arm measures nothing about the gate. The prior
  "governance-only" reading in FINDINGS.md was retracted.
- **B2 (finance label-miss is arithmetic): CONFIRMED.** `label_noise` is a symmetric
  flip; `p'=p+r(1−2p)` → shift vanishes as p→0.5 (finance `adverse`≈0.42). Recall 0.00
  at every ε (`crossdomain_finance.csv`). Now stated with the derivation, not just a graph.
- **B3 (no reliability method): PARTLY REFUTED.** `experiments/code_agreement.py`
  already implements a seeded 30-incident κ pipeline (mode + detectability) with a
  machine 2nd annotator; only the human labels file is unrun (template exists).
- **Spine (composite masking): CONFIRMED & strong.** Detection 0.756 vs action 0.417
  (0.34 gap); 74/180 degraded cells (41 %) detected-but-not-withheld; gap 0.00 for
  structural only. Mechanism = weighted-mean score in `src/gate/readiness.py`
  (one failed check of ~15 → pass-rate 1.00→0.93 → still ≥0.8 READY).
- **NIST granularity: ALREADY SATISFIED.** Mapping is at MEASURE subcategory level;
  full RMF = 72 subcategories, this maps only MEASURE (21) by design.
New: `paper/figures/fig6_detection_action_gap.{pdf,png}` (the spine),
`docs/NEXT_EXPERIMENTS.md` (ranked plan to close B1/B2 and strengthen the rest).

### V9 — downstream-arm redesign (item 1) drafted + smoke-tested (2026-08-23)
New code (reviewed before the full run — CONVENTIONS.md hard rule 4):
`experiments/run_downstream.py`, `src/domains/metrics.py`,
`tests/test_downstream.py`, plus a backward-compatible `Domain.subgroup_column`
field. Four training conditions per cell (clean_only=oracle, all_ungated=diluted,
corrupt_only=undiluted, gate_realistic); sensitive metrics worst-group accuracy /
subgroup FNR gap / ECE (numpy-only, no new dependency).
```bash
.venv/Scripts/python.exe experiments/run_downstream.py --smoke   # <1 min, synthetic
.venv/Scripts/python.exe -m pytest tests/ -q                     # 38 passed
```
Smoke outcome (synthetic — sanity, NOT a paper result): label corruption drops
worst-group accuracy by 0.49 (healthcare) / 0.43 (government) and raises ECE from
~0.02 to ~0.34 (clean_only vs corrupt_only) while aggregate AUC barely moved —
confirming the redesigned arm sees what the old one missed. `gate_realistic`
withholds structural (n_train 6→4) but not label (stays 6), matching the spine.
Real-data grid pending (human runs the command in NEXT_EXPERIMENTS §1).

### V10 — downstream arm: real-data results (items 1–2) — 2026-08-23
Human ran the full real-data grid (`git=43759f7`, 5 seeds × 3 severities, all three
domains). CSVs: `experiments/results/downstream_{healthcare,finance,government}.csv`.
Analyzed by `experiments/analyze_downstream.py` → `downstream_summary.csv`,
`paper/figures/figD1/D2/D3`. Numbers quoted from `downstream_summary.csv`:
- **Blocker 1 RESOLVED.** Label corruption raises ECE 9× healthcare (0.032→0.288) and
  6× government (0.046→0.288) **while AUC is unchanged** (government 0.928→0.927) —
  the harm the aggregate-AUC arm was blind to. Worst-group accuracy mostly flat on
  real data (corrects the synthetic-smoke expectation).
- **Spine confirmed downstream.** `gate_realistic` ≈ `all_ungated` for label (not
  withheld → residual ECE 0.105/0.120), but recovers worst-group accuracy to the
  oracle for government structural (gate 0.911 = clean_only, vs 0.905 ungated).
- **Item 2 severity DONE:** label ECE scales low→med→high (healthcare 0.147/0.287/0.431);
  other modes flat. **Dilution sweep PENDING** (`--n-degraded` 1/3; `figD4` auto-draws).
Tests: 38 passed (`test_downstream.py`). Finance is the weak domain (AUC~0.67,
worst-group~0.50, balanced label mutes calibration harm) — reported, not hidden.

### V11 — confident-learning label detector (item 3, Blocker 2 fix) — 2026-08-23
New extension (frozen gate untouched): `src/domains/label_quality.py` (per-source
cross-validated self-disagreement rate + one-sided consensus check),
`degrade_labels_asymmetric` in `degradations.py`, `experiments/run_label_detector.py`,
`experiments/analyze_label_detector.py`, `tests/test_label_quality.py`.
```bash
.venv/Scripts/python.exe experiments/run_label_detector.py --smoke
.venv/Scripts/python.exe -m pytest tests/ -q      # 45 passed
```
Prototype + smoke (synthetic — sanity, not paper numbers):
- Self-disagreement separates clean vs 45%-symmetric-noisy in ALL domains including
  balanced finance: finance clean 0.390±0.003 → noisy 0.489 (gap +0.10) at ~6.7k
  rows/source; the marginal detector cannot see this (Blocker 2).
- Symmetric-noise recall marginal→confident: finance 0.00→0.67, healthcare 0.75→1.00,
  government 0.75→1.00; precision ~1.0.
- Asymmetric flips move the marginal (finance base 0.42 → 0.23 at rate 0.45), so the
  gate's label-rate detector regains recall (derivation confirmed).
- Threshold 0.05 calibrated on CLEAN sources (not on outcome — no leakage), matching
  the `divergence_threshold` discipline. confidence_margin>0 backfires (symmetric noise
  removes confident predictions) → plain disagreement is used.
Honest caveats recorded: finance gap smallest (high base error); asymmetric detection
on rare-label domains weak for both detectors; rigorous DP of a CV statistic is future
work (approximate `epsilon` option flagged). Real-data grid pending (human).

---

## Session 5 — novelty pass + positioning anchors (2026-08-25)

### V12 — spine novelty pass + citation verification (2026-08-25)
Verified the composite-masking spine against the literature by fetching primary
sources (WebSearch + WebFetch). Outcome: **spine survives** — no fetched source
claims composite/threshold readiness scoring masks severe single-dimension defects
with a detection-rate vs gate-action-rate split, benchmarked cross-domain. Framing
adjusted to "known-in-principle, never quantified as a gate-action gap."

Anchors fetched and stamped `[VERIFIED-FETCH 2026-08-25]` in `paper/refs.bib`
(title / ordered authors / venue confirmed against the source):
- `breck2019datavalidation` — proceedings.mlsys.org (MLSys 2019).
- `schelter2018deequ` — PVLDB 11(12):1781--1794, 2018.
- `lawrence2017datareadiness` — arXiv:1705.02245.
- `afzal2020datareadinessreport` — arXiv:2010.07213 (abstract only; CAVEAT: specific
  ML checks NOT confirmed — do not attribute in prose until full text read).
- `rokvic2022lia` — arXiv:2205.11518 / IEEE BigData 2024.
- `hiniduma2025aidrin` — arXiv:2505.18213 (full text read). Nearest neighbor:
  report-only, per-dimension, no gate decision, no masking discussion → delta line
  recorded in the entry.
Left `TODO-VERIFY`: `caveness2020tfdv` (ACM 403, dblp API 500 on 2026-08-25 — author
list not source-confirmed; cite `breck2019datavalidation` for the method meanwhile).
No code run; no result CSVs changed. See `memory/novelty-pass-masking-spine.md`.

### V13 — aggregation-rule ablation (item 4, the spine's "fix") drafted + smoke-tested (2026-08-25)
New extension (frozen gate untouched — `git diff src/gate/` empty): decision rules
`src/domains/aggregation.py` (`weighted_mean`=score_source verbatim, `max_violation`,
`any_critical`), experiment `experiments/run_aggregation_ablation.py`, tests
`tests/test_aggregation.py`. Detection = `deg.flagged_for_mode` (rule-independent);
action = `decision != READY`; gap = det_recall - action_rate; clean_fpr = clean
sources not marked READY.
```bash
.venv/Scripts/python.exe -m pytest tests/ -q                          # 51 passed
.venv/Scripts/python.exe experiments/run_aggregation_ablation.py --smoke
```
Smoke (synthetic — sanity, NOT paper numbers): mean gap over degraded cells /
clean_fpr at mode=none, per rule:
- healthcare: weighted_mean 0.250 / 0.000 -> max_violation 0.250 / 0.000 ->
  any_critical 0.000 / 0.100
- finance:    weighted_mean 0.500 / 0.000 -> max_violation 0.000 / 0.000 ->
  any_critical 0.000 / 0.000
- government: weighted_mean 0.750 / 0.000 -> max_violation 0.500 / 0.000 ->
  any_critical 0.000 / 0.000
Confirms the predicted shape: `max_violation` is a PARTIAL fix (catches tanked
dimensions but not a single check hidden in pass_rate=0.93 — healthcare unchanged);
`any_critical` closes the gap fully at a small, measurable clean-FPR cost. Real-data
grid + a figure/summary are pending (human) — analyzer deferred to avoid plotting
synthetic smoke, per repo discipline.

### V14 — external-baseline harness + Colab notebook (item 6) — 2026-08-25
New (nothing in `src/gate/` touched): `experiments/baselines/gen_baseline_sources.py`
(regenerates the gate's exact per-cell sources — same seeds/splits/degrade as
run_crossdomain), `tests/test_baseline_sources.py` (5 tests), and
`experiments/baselines/baselines_colab.ipynb` (GE 0.18.19 + TFDV + Deequ default
checks → `baselines_detection.csv`, per-mode det_recall + false_flag_rate aligned to
the gate).
```bash
.venv/Scripts/python.exe -m pytest tests/ -q                 # 56 passed
.venv/Scripts/python.exe experiments/baselines/gen_baseline_sources.py   # rows-free self-check
.venv/Scripts/python.exe -c "import json;json.load(open('experiments/baselines/baselines_colab.ipynb'))"  # valid ipynb
```
Source-regenerator verified locally (structural drops columns in degraded sources
only; completeness raises nulls >0.3 in degraded only; reproducible per seed; correct
degraded counts/indices). **The three baseline adapters (GE/TFDV/Deequ) were NOT run**
— TFDV/Deequ don't run on this Windows box; the human runs the notebook in Colab and
copies `baselines_detection.csv` back. Data safety: synthetic by default; healthcare
stays synthetic (MIMIC DUA); finance/government public extracts optional via DATA_ROOT.

### V15 — real-data analysis of items 4 & 6 (2026-08-25)
Human ran the real aggregation-ablation grid (`aggregation_{healthcare,finance,
government}.csv`, 2925 cells) and the Colab baseline notebook
(`baselines_detection.csv`; **GE ran, TFDV/Deequ empty**). Analyzers added:
`experiments/analyze_aggregation.py` (figA1/figA2 + `aggregation_summary.csv`) and
`experiments/analyze_baselines.py` (figBL + `baselines_summary.csv`).
```bash
.venv/Scripts/python.exe experiments/analyze_aggregation.py
.venv/Scripts/python.exe experiments/analyze_baselines.py
```
Numbers quoted in `docs/FINDINGS.md` Findings 9–10, all from the summary CSVs:
- **Item 4 (Finding 9):** gap (det−action, ε=∞) weighted_mean → any_critical: finance
  0.433→0.000, government 0.667→0.000, healthcare −0.083→−0.250; max_violation partial.
  Clean-FPR by rule×ε: any_critical **0.000 (ε≥1) → 0.011 (ε=0.5) → 0.878 (ε=0.1)**;
  weighted_mean 0.000 at every ε. → the fix closes the gap free for ε≥0.5 but is
  DP-noise-fragile at ε=0.1 (privacy↔aggregation tradeoff). Honest, not a strict win.
- **Item 6 (Finding 10):** gate vs GE det_recall ties at 1.0 on structural/completeness;
  both miss balanced-finance label (gate 0.0, GE 0.2); GE clean false-flag hc 0.333 /
  fin 0.067 / gov 0.000 vs gate 0.000 everywhere. GE used reference-derived expectations
  (noted); TFDV/Deequ absent (reported as ABSENT, not zero). 56 tests still pass.

### V16 — Deequ baseline added via headless Docker runner (2026-08-26)
`experiments/baselines/run_baselines.py` (new, py3.11/Docker) run in `python:3.11-slim`
with `pyspark==3.5.1` + `SPARK_VERSION=3.5` (pydeequ rejects Spark 3.4). **GE + Deequ ran;
TFDV skipped** (pyfarmhash needs g++, absent in slim; apache-beam[gcp] resolver). Re-ran
`analyze_baselines.py` → figBL now has 3 series, `baselines_summary.csv` = 45 rows.
Deequ (from `baselines_detection.csv`): det_recall structural/completeness/distributional
= 1.0 all domains, **label = 0.0 all domains** (adapter has no label constraint), clean
false_flag = 0.000 all domains. Three-way reading in FINDINGS Finding 10. No src/ changed;
synthetic-only in the container (no data/ read).

### V17 — bootstrap 95% CIs (item 7) — 2026-08-26
Added `experiments/ci_utils.py` (seed-level percentile bootstrap, 10k resamples,
seed=42) and wired CIs into analyze.py (fig6 + metrics_summary CI columns),
analyze_aggregation.py (figA1 error bars), analyze_baselines.py (figBL error bars).
```bash
.venv/Scripts/python.exe experiments/analyze.py
.venv/Scripts/python.exe experiments/analyze_aggregation.py
.venv/Scripts/python.exe experiments/analyze_baselines.py
.venv/Scripts/python.exe -m pytest tests/ -q      # 56 passed
```
Headline (from crossdomain_*.csv, eps=inf, all modes/domains): detect 0.756
[0.739,0.764], withhold 0.417 [0.417,0.417], GAP 0.339 (95% CI [0.322,0.347]) —
excludes 0 widely, so the spine is not seed noise. No new experiment runs (existing
5 seeds); no src/ changed (git diff src/gate empty). FINDINGS Finding 2 updated with
the CI.

### V18 — runners for items 3.2 (threshold transfer) & 3.3 (confident ε-sweep) — 2026-08-26
New (frozen gate untouched — git diff src/gate empty): `experiments/run_threshold_transfer.py`
(cold cross-domain divergence_threshold transfer: diagonal=own, off-diagonal=transferred;
det_recall + clean false_flag) and `experiments/run_label_epsilon.py` (DP ε-sweep of the
confident-learning self-disagreement detector + per-domain clean-disagreement calibration
stats). Both smoke-tested (synthetic):
```bash
.venv/Scripts/python.exe experiments/run_threshold_transfer.py --smoke   # 18 rows
.venv/Scripts/python.exe experiments/run_label_epsilon.py --smoke        # 12 rows; finance confident recall 0.75 @ eps=1 & inf
.venv/Scripts/python.exe -m pytest tests/ -q                             # 56 passed
```
Real grids pending (human — commands in docs/NEXT_EXPERIMENTS.md §5 and §3.3). Smoke
numbers are synthetic sanity only, NOT paper numbers (tiny n inflates false_flag).

### V19 — real-data analysis of items 3.2 & 3.3; item 2.3 NOT done (2026-08-26)
Human ran the real grids. Analyzers added: `experiments/analyze_threshold_transfer.py`
(figT1 + threshold_transfer_summary.csv) and `experiments/analyze_label_epsilon.py`
(figE1 + label_epsilon_summary.csv). Numbers in FINDINGS Findings 11–12:
- **3.2 (Finding 11):** recall transfers (own 0.69 vs transferred 0.67) but clean false-flag
  does NOT — healthcare's threshold applied to finance/government → false-flag 0.98/1.00;
  mean 0.22 (own) → 0.44 (transferred). Thresholds are not portable.
- **3.3 (Finding 12):** confident detector DP-robust — recall finance 0.63→0.57 (ε=∞→0.1),
  healthcare/government hold 1.00 to ε=0.1 (vs divergence detector's ε=0.1 blowup, Finding 3).
  Clean self-disagreement std finance 0.0073 / gov 0.0016 / hc 0.0013 → 0.05 threshold safely
  calibrated. Open: rigorous DP accounting (3.3-i).
- **2.3 κ NOT done:** `taxonomy_sample_human_labels.csv` is BLANK (0/30 mode+detectability
  filled); `taxonomy_agreement.txt` shows κ=0.000 / all-nan confusion — an empty-file artifact,
  NOT a result. The human must code the 30 incidents (blind) before κ is meaningful. Not
  fabricated. 56 tests still pass; src/gate untouched.

### V20 — Pillar A κ, real (2026-08-26)
The human worksheet docs/taxonomy_sample_for_human_review.md was already filled (30/30
valid codings); those were transcribed VERBATIM (regex parse — no values invented) into
experiments/results/taxonomy_sample_human_labels.csv, then:
```bash
.venv/Scripts/python.exe experiments/code_agreement.py
```
Outcome (taxonomy_agreement.txt): mode percent-agreement 0.233 / Cohen's κ 0.082;
detectability agreement 0.333 / κ 0.038 (both ~chance — machine over-assigns
distributional 14/30 vs human 6, and partial 24/30 vs 9). Filter precision 0.933 (human
judged 28/30 ARE data-readiness failures). Honest negative on fine coding → FINDINGS
Finding 13: claim filter precision only, not fine proportions/12%. Intra-rater pass still
pending. AI did NOT author any human label (transcription only, from the human-filled .md).

### V21 — paper draft + 3 citation verifications (2026-11-03)
Verified via Crossref/source and graduated TODO-VERIFY -> VERIFIED-FETCH in refs.bib:
`dwork2014` (DOI 10.1561/0400000042, FnT TCS 9(3-4):211-487, 2014), `jensenshannon`
(Lin, DOI 10.1109/18.61115, IEEE TIT 37(1):145-151, 1991), `greatexpectations`
(github.com/great-expectations/great_expectations, Apache-2.0).
Wrote `paper/main.tex` (arXiv preprint draft, sole author). Validated: 23 \cite keys all
resolve in refs.bib and NONE are TODO-VERIFY; all 5 \includegraphics figures exist.
No local LaTeX toolchain to compile. FedCheck (prior, unpublished) and TFDV (omitted for
packaging) are referenced in prose with no \cite — no fabricated bib details. Every number
is quoted from docs/FINDINGS.md / experiments/results/*.csv (hard rule 1); 3.3-(i) DP
accounting is folded into §Privacy-utility as "approximate", not overclaimed.

### V22 — arXiv-review revision + intra-rater scaffolding (2026-11-03)
Paper (`paper/main.tex`) revised against an external arXiv-readiness review and
recompiled clean (0 undefined citations, 0 undefined refs; 10 pages):
- Figures regenerated without internal "Item N:" titles and without the gray
  `fig.text` provenance footnotes (edited `experiments/analyze.py`,
  `analyze_aggregation.py`, `analyze_label_epsilon.py`, `analyze_threshold_transfer.py`);
  provenance consolidated into the paper's Reproducibility paragraph.
- Title de-emphasizes the taxonomy; `\date` no longer says "Draft"; author email set
  to gargi.desai20@gmail.com and ORCID 0009-0007-9491-2949 added.
- 0.34-vs-41% reconciled in prose as two denominators (mean-rate difference vs 74/180
  cells; 74/180 quoted from FINDINGS Finding 2 line "74 of 180 (41%)").
- DP claim softened to "DP-style"/"approximate"; "DP-robust" scoped to empirical recall.
- NIST MEASURE mapping added as a body TABLE, transcribed from docs/NIST_MAPPING.md.
Citations verified by fetching (hard rule 2) and written into refs.bib:
- `northcutt2021confidentlearning` ADDED + cited (JAIR vol. 70, 2021; DOI
  10.1613/jair.1.12125; arXiv:1911.00068) — JAIR + arXiv pages.
- `mimiciv` full 13-author list from Crossref (DOI 10.1038/s41597-022-01899-x,
  published 2023); TODO-VERIFY note removed.
- `paeth2024aiid` full 5-author list from arXiv:2409.16425; TODO-VERIFY note removed.
- `nistairmf2023` MEASURE subcategory IDs (1.1,1.3,2.1,2.3,2.5,2.8,2.10,2.11,2.13,3.1)
  re-confirmed present in the official NIST AI 100-1 PDF (pypdf text extraction);
  MEASURE 2 = 2.1–2.13. Stale "Session 4" note removed.
Intra-rater reliability scaffolding built (NOT yet run on data):
- `experiments/make_pass2_worksheet.py` → `docs/taxonomy_sample_pass2_blind.md`
  (30 incidents, pass-1 labels stripped, reshuffled seed=202; verified 60 blank
  fields, 0 filled).
- `experiments/code_agreement.py --intra` added: compares pass1 vs
  `taxonomy_sample_human_labels_pass2.csv`, writes a SEPARATE
  `taxonomy_agreement_intra.txt` (inter-rater `taxonomy_agreement.txt` untouched);
  inter-rater path regression-checked (mode agr 0.233, det 0.333, filter 0.933 —
  unchanged).
- INTRA-RATER κ IS PENDING: it requires a human blind pass-2 coding after a washout.
  No κ number has been written to FINDINGS or §5.6; §5.6 still correctly says "an
  intra-rater pass is future work" until the real pass-2 exists. AI authors no labels.

### V23 — intra-rater κ computed on the human pass-2 (2026-11-03)
Human completed the blind pass-2 coding (docs/taxonomy_sample_pass2_blind.md), transcribed
via `make_pass2_worksheet.py --transcribe` to
`experiments/results/taxonomy_sample_human_labels_pass2.csv`, then:
    .venv/Scripts/python.exe experiments/code_agreement.py --intra
Outcome (quoted from `taxonomy_agreement_intra.txt`, n=30 coded in both passes):
  assigned mode — percent agreement 0.100, Cohen's κ = -0.048;
  detectability  — percent agreement 0.233, Cohen's κ = -0.144.
Both κ are at/below chance: the same coder does NOT reproduce their own fine codings after a
washout. Interpretation (honest negative, strengthens §5.6): the fine mode/detectability
scheme is intrinsically unstable, not merely a machine-vs-human artifact — so the paper relies
only on the coarse human-validated filter precision (0.93), not the fine proportions.
Written to docs/FINDINGS.md (Finding 13 addendum) and paper §5.6 parenthetical (mode κ≈-0.05,
detectability κ≈-0.14); paper recompiled clean (0 undefined cites/refs, 10 pp.). The
INTER-rater result (taxonomy_agreement.txt: mode κ 0.082, det κ 0.038, filter 0.933) was NOT
modified. AI computed the statistic from human-authored labels; AI authored no labels.
