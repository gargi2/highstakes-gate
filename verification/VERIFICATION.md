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

### Pending (human runs; no assistant credits)
Full grid + real extractions — see `--print-commands`. Results land in
`experiments/results/crossdomain_<domain>.csv`.
