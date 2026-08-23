# Verification log

Per CLAUDE.md hard rule 4, the human reads AI-written code before commit and
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

### Pending (human runs; no assistant credits)
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
New code (human reviews before the full run — CLAUDE.md hard rule 4):
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
