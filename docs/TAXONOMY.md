# Pillar A — A data-readiness failure taxonomy from production incidents

**Axis:** data-readiness failure modes that are, in principle, detectable
*before* training — the lens that differentiates this work from the AI Incident
Database's own taxonomies (CSET, GMF, MIT) and from Sambasivan et al.'s data
cascades. We do **not** claim a new catalogue of incidents; we re-project a
subset of AIID incidents onto the question *"would a pre-training validation
gate have caught the data problem?"* and map each mode to NIST AI RMF MEASURE
(§ NIST mapping below, refined in Session 4).

All counts in this document are quoted from
[`experiments/results/taxonomy_coding.csv`](../experiments/results/taxonomy_coding.csv),
produced by [`experiments/code_incidents.py`](../experiments/code_incidents.py)
over the pinned AIID snapshot (2026-08-17; provenance in
[`docs/DATA_AIID.md`](DATA_AIID.md)). Re-generate with:

```bash
.venv/Scripts/python.exe experiments/code_incidents.py
```

> **Reading the CSV:** it has a leading `#` reproduction-header block, but
> incident titles contain `#` (e.g. `#LekkiMassacre`). Read past the header with
> `skiprows` (use `read_results_csv()` in `experiments/code_incidents.py`) —
> **not** `pd.read_csv(comment="#")`, which would corrupt those rows.

## Method (transparent, reproducible, conservative)

Each of the 1,632 AIID incidents is scored against the six modes using two
signal sources, both auditable in the script:

1. **Keyword lexicons** over the lowercased title + description — a STRONG list
   (weight 1.0) and a WEAK list (weight 0.5) per mode.
2. **AIID's own GMF expert labels** (`Known/Potential AI Technical Failure`),
   the data-related subset only, mapped to our modes (weight 2.0). This borrows
   AIID's root-cause attribution; we add only the detectability lens.

An incident is **coded as data-readiness** iff some mode scores ≥ 1.0; the
assigned mode is the argmax (ties broken by a fixed priority). Every row carries
a **confidence** flag (high / med / low) and a **one-line rationale** listing the
signals that fired. The filter is deliberately precision-oriented — it will miss
data-rooted incidents described only in deployment terms (see Limitations).

> **Rigor / this is qualitative coding.** The codings are a reproducible rule,
> not ground truth. A seeded 30-incident sample
> ([`docs/taxonomy_sample_for_human_review.md`](taxonomy_sample_for_human_review.md))
> is set aside for **independent human coding**; inter-rater agreement (percent
> agreement + Cohen's κ for both mode and detectability, plus filter precision)
> is computed by [`experiments/code_agreement.py`](../experiments/code_agreement.py)
> once the human labels are returned. Do not cite reliability numbers until that
> runs.

## Coverage

- **195 of 1,632** incidents (12.0%) were coded as plausibly data-readiness-rooted.
- **Confidence:** 139 high · 31 med · 25 low (quote from CSV).
- **Pre-training detectability (as coded):** 38 `yes` · 157 `partial` · 0 `no`.

The `no`-detectability count being 0 is expected: an incident only enters the
table if it matched a *data-readiness* signal, and all six modes are at least
partially pre-training-detectable by construction. Non-detectable harms
(misuse, deployment, model-capability failures) are correctly *excluded* by the
filter rather than coded `no`.

## The six modes

Counts, per-mode detectability split, and confidence split are all quoted from
`taxonomy_coding.csv`.

### 1. Structural / schema — **n = 13** (13 `yes`; 13 high-confidence)
Missing columns, type mismatches, encoding/format drift, faulty preprocessing.
In AIID these surface as GMF's *"Faulty or Inadequate Preprocessing"* rather
than as raw schema errors (public reports rarely expose the schema), so this
mode leans entirely on the expert label — the most human-review-dependent mode.

Examples:
- **22 — Waze Navigates Motorists into Wildfires** — routing over
  mis-preprocessed / stale road-state data.
- **70 — Self-driving cars in winter** — sensor/feature preprocessing degrades
  under distribution of winter inputs.
- **6 — Microsoft's TayBot posts abusive content** — GMF attributes inadequate
  input preprocessing/filtering.

**Gate detects pre-training? YES.** Direct: `column_exists`, `column_dtype`,
`value_range` (expectation suite) and `schema_compatibility`
(`src/gate/compatibility.py`).

### 2. Completeness — **n = 21** (21 `yes`; 21 high-confidence)
Missing values, missing subpopulations, insufficient attribute capture,
coverage gaps. In AIID: GMF's *"Incomplete Data Attribute Capture."*

Examples:
- **57 — Australian Automated Debt Assessment ("Robodebt")** — averaged/
  incomplete income records drove false debt notices.
- **26 — Hackers Break Apple Face ID** — training captured too few attributes to
  separate a mask from a face.
- **9 — NYC Value-Added Teacher Ratings released** — sparse per-teacher data
  yielded unstable, disputed scores.

**Gate detects pre-training? YES.** Direct: `null_fraction_max`,
`min_row_count` (expectation suite).

### 3. Distributional — **n = 94** (4 `yes`, 90 `partial`; 67 high / 2 med / 25 low)
Population skew, under/over-representation, sampling bias, covariate/label
shift, concept drift. The largest mode, largely via GMF *"Distributional Bias"*,
*"Dataset Imbalance"*, *"Inadequate Data Sampling"*, *"Concept Drift"* plus
demographic keywords.

Examples:
- **16 — Images of Black People Labeled as Gorillas** — training set
  under-represented darker skin tones.
- **14 — Biased Sentiment Analysis** — skewed training corpus.
- **12 — Common Biases of Vector Embeddings** — distributional bias baked into
  pretraining data.

**Gate detects pre-training? PARTIAL.** `distribution_divergence` /
`js_divergence` / `pairwise_divergence` detect skew **against a reference or
across sources**; a lone training set with no reference population cannot be
flagged as "skewed" from values alone — hence 90/94 coded `partial`. The 4 `yes`
are cases where an explicit, self-contained training-set imbalance is stated
(measurable directly).

### 4. Label — **n = 17** (17 `partial`; 16 high / 1 med)
Label noise, annotation error, mislabeling, leakage. In AIID: GMF's *"Data or
Labelling Noise."*

Examples:
- **113 — Facebook's AI puts "Primates" label on video of Black men** —
  label/annotation failure rooted in labeled training data.
- **13 — High-Toxicity assessed on text involving women and minority groups** —
  noisy toxicity labels.
- **36 — Woman on side of bus shamed for jaywalking** — mislabeled detection.

**Gate detects pre-training? PARTIAL.** `label_rate_divergence`
(`src/gate/consensus.py`) flags anomalous label rates *across sources*; subtle
annotation error *within* a source is not detectable from label rates alone.

### 5. Provenance / documentation — **n = 50** (50 `partial`; 22 high / 28 med)
Unknown lineage, unauthorized/scraped data, undocumented sources, missing
consent — the Datasheets / Data-Statements gap. In AIID: GMF's *"Unauthorized
Data"* plus consent/scraping keywords.

Examples:
- **190 — ByteDance trained "For You" on content scraped without consent** —
  provenance/authorization failure.
- **122 — Facebook "Tag Suggestions" stored biometric data without consent.**
- **106 — Korean chatbot Luda trained on chat data without consent.**

**Gate detects pre-training? PARTIAL — leaning NO for the harm.** The gate
*enforces the presence* of a structured quality/lineage report
(`SourceQualityReport`, `src/gate/reports.py`) — a datasheet-like artifact — so
it catches *undocumented* data. It cannot, from values alone, determine that
data was scraped or unauthorized. Coded `partial` to reflect the
documentation-presence check, not detection of the underlying consent violation.

### 6. Cross-source incompatibility — **n = 0**
Multi-site schema/semantic misalignment (the original FedCheck motivation).

**No AIID incidents were coded to this mode.** This is a genuine, reportable
finding, not a coder bug: a corpus probe for multi-source/merge/harmonization
language found 39 candidate hits, all of which were *"emergency"*, lane-merge,
or corporate-merger noise — none described a data-integration failure. Public
incident reports describe deployment-facing harms, not an organization's
internal cross-site data-integration problems, so this mode is essentially
**invisible to incident-database-derived taxonomies**. It remains in the axis
because it is real and gate-detectable (validated in FedCheck's multi-site
setting).

**Gate detects pre-training? YES.** `schema_compatibility` +
`pairwise_divergence` (`src/gate/compatibility.py`) are purpose-built for it.

## Gate-detectability summary

| # | Mode | AIID n | Coded detectability | Gate mechanism | Detects pre-training? |
|---|------|-------:|---------------------|----------------|-----------------------|
| 1 | Structural / schema | 13 | 13 yes | `column_exists`, `column_dtype`, `value_range`, `schema_compatibility` | **Yes** |
| 2 | Completeness | 21 | 21 yes | `null_fraction_max`, `min_row_count` | **Yes** |
| 3 | Distributional | 94 | 4 yes / 90 partial | `distribution_divergence`, `js_divergence` (needs reference/across-source) | **Partial** |
| 4 | Label | 17 | 17 partial | `label_rate_divergence` (across-source only) | **Partial** |
| 5 | Provenance / documentation | 50 | 50 partial | `SourceQualityReport` presence (not consent/lineage truth) | **Partial→No** |
| 6 | Cross-source incompatibility | 0 | — | `schema_compatibility`, `pairwise_divergence` | **Yes** (absent from AIID) |

*n* and detectability splits quoted from `taxonomy_coding.csv`; gate mechanisms
quoted from `src/gate/` and `suites/clinical_v1.json`.

## NIST AI RMF MEASURE cross-reference (first draft — refine in Session 4)

Category-level only. **Subcategory numbers are `TODO-VERIFY` against the
official NIST AI 100-1 text and MEASURE playbook** before any draft asserts
them (project hard rule 2; see `docs/PIVOT_SCOPE.md` §3).

| Mode | MEASURE category | Governance claim | Status |
|------|------------------|------------------|--------|
| Structural / schema | MEASURE 2 (TEVV; validity) | data validity measured pre-deployment | subcat TODO-VERIFY |
| Completeness | MEASURE 2 (representativeness) | coverage/representativeness assessed | subcat TODO-VERIFY |
| Distributional | MEASURE 2 (bias / drift, ~2.11) | population skew detected before training | subcat TODO-VERIFY |
| Label | MEASURE 2 (TEVV; data quality) | label integrity measured | subcat TODO-VERIFY |
| Provenance / documentation | MEASURE 2 + GOVERN (data documentation) | lineage/consent documented | subcat TODO-VERIFY |
| Cross-source incompatibility | MEASURE 2 (bias/validity across sources) | multi-site consistency measured | subcat TODO-VERIFY |
| (all) Readiness score + gate decision | MEASURE 1 (metrics selected/applied) | documented go/no-go metric | subcat TODO-VERIFY |

The MAP→MEASURE loop is the framing: the incident-derived taxonomy (MAP) sets
*what to measure*; the gate *measures* it (MEASURE); the readiness score is the
documented, pre-deployment metric.

## Limitations (for the reviewer who will ask)

1. **Precision over recall.** The filter targets high precision; incidents whose
   data root cause is described only in deployment language are missed. Coverage
   (195/1,632) is a lower bound on data-readiness-rooted incidents, not an
   estimate of their true prevalence.
2. **Reliance on AIID's GMF labels.** Structural, label, and much of the
   distributional/completeness signal comes from AIID's expert `Technical
   Failure` labels. We re-project their attribution; we do not re-verify each
   root cause. The human-review κ quantifies how much this matters.
3. **Detectability is coded at mode level with two refinements** (distributional
   static-imbalance → `yes`, drift → `partial`), not per-incident forensics. It
   states whether the *class* of failure is gate-detectable, not that the gate
   would have prevented that specific incident.
4. **Cross-source mode is empirically empty in AIID** — a property of public
   incident reporting, addressed by the cross-domain benchmark (Pillar B), not
   by this corpus.
5. **Inter-rater reliability is pending** the human coding of the seeded sample;
   no agreement statistic is reported here until `code_agreement.py` runs on real
   labels.
