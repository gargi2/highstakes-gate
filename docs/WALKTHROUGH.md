# Pillar B — cross-domain gate benchmark: how it works

Pillar B runs the **same** validation gate (`src/gate/`, reused unchanged from
FedCheck) and the **same** taxonomy-derived degradations across three
high-stakes domains — healthcare, finance, government — to test the claim that
the gate generalizes beyond its single validated domain. It answers: *for each
pre-training data-readiness failure mode, does the gate detect it, does gating
change the readiness decision, and does it change a downstream model — and how
does differential privacy (ε) affect that?*

Nothing here modifies the gate. Domains and degradations are new; the gate is
imported and called.

## Pieces

| File | Role |
|---|---|
| `src/domains/base.py` | The `Domain` contract + registry; dependency-free downstream model (numpy logistic regression) and ROC-AUC; chunked CSV reader. |
| `src/domains/healthcare.py` | In-hospital mortality. Synthetic = FedCheck's MIMIC-like generator; real = a de-identified clinical CSV via `--data-root`. |
| `src/domains/finance.py` | SEC EDGAR Financial Statement Data Sets; adverse-earnings (NetIncomeLoss<0) from log balance-sheet magnitudes. |
| `src/domains/government.py` | CMS Open Payments; disputed-payment flag from payment magnitude/frequency. |
| `src/domains/degradations.py` | Maps the four row-injectable taxonomy modes → a `gate.partition.degrade` spec (REUSED), and → the gate check(s) that should catch each. |
| `experiments/run_crossdomain.py` | The grid: domain × mode × severity × ε × seed → one CSV per domain. `--smoke` for a <2 min run. |
| `tests/test_domains.py` | Contract + wiring + one end-to-end cell per domain. |

## The gate pipeline, per grid cell

For each `(domain, mode, severity, ε, seed)`:

1. **Load** data (`Domain.get_frame`): real via the domain loader (streamed in
   chunks, capped at `--max-rows`) if `--data-root` is given, else synthetic.
   A clean 20 % holdout is reserved for the downstream metric and is never
   degraded.
2. **Split** the rest into `n_sources` simulated data providers
   (`gate.partition.split_sources`).
3. **Degrade** the first `n_degraded` sources with the mode's spec
   (`gate.partition.degrade` via `degradations.build_spec`).
4. **Report** each source through the gate: `gate.expectations.run_suite`
   produces a DP-noised, rows-free `SourceQualityReport` (schema, noised null
   fractions, noised feature histograms, noised positive-label rate).
5. **Gate-side analysis** on the reports only (never rows):
   `schema_compatibility`, `consensus_checks` (label-rate + divergence, with the
   domain's calibrated `divergence_threshold`), then `readiness.score_source`
   → READY / FIXABLE / EXCLUDE per source.
6. **Score** the cell:
   - detection **precision/recall** for the active mode over sources (was each
     degraded source flagged by the mode's own gate check?);
   - readiness decision counts, `degraded_withheld`, `clean_excluded`;
   - downstream **AUC with vs without gating**: train the tiny logistic model on
     all sources vs on only the kept (READY/FIXABLE) sources, evaluate both on
     the clean holdout.

## The four degradations ↔ taxonomy modes ↔ gate checks

| Mode (docs/TAXONOMY.md) | Injection (`degrade`) | Gate check that catches it |
|---|---|---|
| structural | drop / dtype-corrupt a column | `column_exists`, `column_dtype`, `schema_compatibility` |
| completeness | null out a fraction of a column | `null_fraction_max` |
| distributional | scale-shift a feature | consensus `distribution_divergence` (JS) |
| label | flip a fraction of labels | consensus `label_rate_divergence` |

`cross-source` and `provenance/documentation` (the other two taxonomy modes) are
**not row-injectable** — cross-source is a multi-provider schema/semantic
mismatch and provenance is about lineage/consent, neither expressible as a
single-table degradation — so they are out of scope for this harness. (The gate
*does* address cross-source via `schema_compatibility`/`pairwise_divergence`;
that is exercised implicitly whenever sources disagree.)

## Running it

```bash
# fast dev/CI check (all three domains, synthetic, <2 min)
.venv/Scripts/python.exe experiments/run_crossdomain.py --smoke

# the exact real-extraction + full-grid commands (printed, not run by the assistant)
.venv/Scripts/python.exe experiments/run_crossdomain.py --print-commands
```

Every results CSV carries a `#` reproduction header (command, git rev, grid
config). Read results past the header with `skiprows` (titles/values are plain,
but keep the convention consistent with `taxonomy_coding.csv`).

## Adding a new domain (the plug-in contract)

Create `src/domains/<newdomain>.py` that builds and `register`s one `Domain`:

```python
from .base import Domain, register, read_csv_chunked, aggregate_summary

def synthetic(n_rows, seed): ...      # returns a DataFrame with the columns below
def load(data_root, chunksize): ...   # yields chunks; SELECT only modelling cols

DOMAIN = register(Domain(
    name="newdomain",
    label_column="y",                 # binary 0/1
    feature_columns=[...],            # numeric; used for model + histograms + divergence
    required_columns={...},           # name -> "int"/"float"/"object"; drives suite + schema check
    histogram_ranges={col: (lo, hi)}, # SHARED bins — required for meaningful divergence
    synthetic=synthetic,
    loader=load,
    divergence_threshold=<calibrated>,# see below
))
```

Then it is automatically in the grid (import it in `src/domains/__init__.py`).
Two rules that the tests enforce:

1. **Clean synthetic data must pass the domain's own suite** — the dtype in
   `required_columns` must match what `synthetic`/`load` actually produce, or
   every clean source is wrongly EXCLUDED
   (`test_synthetic_matches_required_schema`, `test_clean_sources_are_ready`).
2. **Calibrate `divergence_threshold` on clean sources at ε=∞** — run the
   distributional probe (see `verification/VERIFICATION.md` V4) and set the
   threshold just above the clean-source mean JS divergence and below the
   degraded-source value. A single fixed 0.08 does not transfer across domains
   because the JS baseline depends on feature scale and bin resolution.

## Limitations (verified — see verification/VERIFICATION.md)

- **Label-rate detector is blind to label noise on near-balanced labels.** It
  watches the positive-rate marginal, which barely moves under symmetric flips
  when the base rate is ≈0.5 (finance `adverse` ≈0.42 → label-noise recall ≈0).
  Rare-label tasks (healthcare, government) detect it easily. This is a property
  of a marginal-based detector, reported rather than hidden.
- **Distributional detection is precision-limited, not recall-limited, under
  DP.** As ε falls, noise inflates every source's cross-source divergence, so
  clean sources get flagged (precision drops from ~0.9 at ε=∞ toward ~0.33 at
  ε=1) while degraded sources stay flagged (recall ≈1 at ε≥5). Report the ε
  sweep, not a single number.
- **Extreme shifts leave the histogram support** and become invisible (empty
  noised histogram → NaN divergence); severity factors are capped in-range.
- **Detection ≠ withholding.** The consensus detectors *flag* distributional and
  label degradations (detection recall ≈1 at ε=∞), but a single consensus
  failure is one check among ~15 in `readiness.score_source`, so the READY/
  EXCLUDE decision usually does NOT withhold those sources (`degraded_withheld`
  ≈ 0 for distributional/label). Structural and completeness degradations *are*
  withheld (schema-incompatible or multiple null failures). The
  `det_*` columns measure detection; `degraded_withheld` measures the gate
  action — the benchmark reports both because they differ.
- **Real-data label choices.** healthcare uses a 0.30 null tolerance (real MIMIC
  labs are ~23-24 % missing at baseline); government predicts payment nature =
  Food-and-Beverage (~0.9 prevalence, learnable from amount/frequency) because
  the dispute flag is ~0.002 % positive and yields no downstream metric.
- **Consensus assumes <50 % of sources are degraded** (median/mean consensus);
  the default grid degrades 2 of 6.
- **Real loaders are format-specific.** finance expects EDGAR `num.txt`
  (tab-delimited); government expects an Open Payments general-payments CSV;
  healthcare expects a CSV already carrying the `required_columns`. Sanity-check
  a real extract first with `python -m domains.<name> --data-root <path>` (prints
  aggregates only) before running the grid on it.
