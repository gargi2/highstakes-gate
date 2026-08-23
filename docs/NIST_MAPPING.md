# Pillar C — mapping the data-readiness gate to NIST AI RMF MEASURE

This document maps (a) each **gate capability** and (b) each **taxonomy failure
mode** to specific, **source-verified** subcategories of the MEASURE function of
the NIST AI Risk Management Framework. It replaces the placeholder, category-level
draft in [`docs/PIVOT_SCOPE.md`](PIVOT_SCOPE.md) §3 and the `TODO-VERIFY` rows in
[`docs/TAXONOMY.md`](TAXONOMY.md) §"NIST AI RMF MEASURE cross-reference".

The MAP→MEASURE loop is the paper's governance spine: the incident-derived
taxonomy (a MAP artifact) enumerates *what to measure*; the gate *measures* it
pre-training (MEASURE); the readiness score is the documented, pre-deployment
metric that MEASURE 1 asks for.

## Source — verified against the official NIST text

Every subcategory number and its wording below was read directly from **Table 3,
"Categories and subcategories for the MEASURE function," in NIST AI 100-1
(AI RMF 1.0), pp. 29–31**, retrieved as the official PDF from NIST:

> National Institute of Standards and Technology (2023). *Artificial Intelligence
> Risk Management Framework (AI RMF 1.0)*, NIST AI 100-1.
> <https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf> (verified 2026-08-23).

MEASURE has **4 categories** and **21 subcategories** in the Core:
MEASURE 1 (1.1–1.3), MEASURE 2 (2.1–2.13), MEASURE 3 (3.1–3.3), MEASURE 4 (4.1–4.3).
The subcategories this project relies on, quoted verbatim from that table:

| Subcat | Verbatim wording (NIST AI 100-1, Table 3) |
|--------|-------------------------------------------|
| **1.1** | "Approaches and metrics for measurement of AI risks enumerated during the MAP function are selected for implementation starting with the most significant AI risks. The risks or trustworthiness characteristics that will not — or cannot — be measured are properly documented." |
| **1.2** | "Appropriateness of AI metrics and effectiveness of existing controls are regularly assessed and updated, including reports of errors and potential impacts on affected communities." |
| **1.3** | "Internal experts who did not serve as front-line developers for the system and/or independent assessors are involved in regular assessments and updates. Domain experts, users, AI actors external to the team … and affected communities are consulted …" |
| **2.1** | "Test sets, metrics, and details about the tools used during TEVV are documented." |
| **2.2** | "Evaluations involving human subjects meet applicable requirements (including human subject protection) and are representative of the relevant population." |
| **2.3** | "AI system performance or assurance criteria are measured qualitatively or quantitatively and demonstrated for conditions similar to deployment setting(s). Measures are documented." |
| **2.5** | "The AI system to be deployed is demonstrated to be valid and reliable. Limitations of the generalizability beyond the conditions under which the technology was developed are documented." |
| **2.8** | "Risks associated with transparency and accountability — as identified in the MAP function — are examined and documented." |
| **2.10** | "Privacy risk of the AI system — as identified in the MAP function — is examined and documented." |
| **2.11** | "Fairness and bias — as identified in the MAP function — are evaluated and results are documented." |
| **2.13** | "Effectiveness of the employed TEVV metrics and processes in the MEASURE function are evaluated and documented." |
| **3.1** | "Approaches, personnel, and documentation are in place to regularly identify and track existing, unanticipated, and emergent AI risks …" |

## Framing caveat (read before the tables)

NIST MEASURE is written about measuring the **AI system's** trustworthiness,
largely at or after model evaluation. This gate measures the **training data**
*before* a model exists. The mapping therefore claims that the gate applies
MEASURE-style, documented, pre-deployment TEVV discipline to the *data artifact*
that feeds the system — an upstream instantiation of the same measurement intent,
not a claim that the gate discharges the subcategory for the deployed model. Two
consequences we state honestly rather than paper over:

- **MEASURE 2.2 is deliberately *not* claimed for completeness/representativeness.**
  2.2 is specifically about *human-subject evaluations* being representative, not
  about a training set's coverage. Coverage/representativeness of data maps to
  2.5 (generalizability) and 2.11 (bias), not 2.2.
- **Each mapping is "measured and documented," not "risk eliminated."** The gate
  produces the documented measurement MEASURE requires; MANAGE (out of scope here)
  is where the risk is treated.

## A. Gate capability → MEASURE subcategory

Gate mechanisms are quoted from `src/gate/`; capabilities are exercised in
Pillar B (`experiments/run_crossdomain.py`) and evidenced in
[`docs/FINDINGS.md`](FINDINGS.md).

| Gate capability (code) | Primary MEASURE | Secondary | Governance claim (what the gate lets you assert) |
|---|---|---|---|
| Schema / dtype / range checks — `column_exists`, `column_dtype`, `value_range` (`gate.expectations`), `schema_compatibility` (`gate.compatibility`) | **2.1** | 2.5 | Pre-training data validity is measured and documented by a versioned test suite (the suite *is* the "test set … documented"). |
| Completeness thresholds — `null_fraction_max`, `min_row_count` (`gate.expectations`) | **2.5** | 2.11 | Coverage/generalizability limits are measured; missing subpopulations (a bias source) are surfaced. *(Not 2.2 — see caveat.)* |
| Distribution divergence — `distribution_divergence` / `js_divergence` / `pairwise_divergence` (`gate.consensus`, `gate.compatibility`) | **2.11** | 2.5 | Population skew / covariate shift across sources is evaluated and documented before training. |
| Label-rate divergence — `label_rate_divergence` (`gate.consensus`) | **2.3** | 2.5 | Label integrity is measured as a documented assurance criterion (across-source label-rate anomaly). |
| Readiness score + gate decision — `score_source` → READY/FIXABLE/EXCLUDE (`gate.readiness`) | **1.1** | 1.2, 2.13 | A single documented go/no-go metric is selected and applied per source, starting with the most significant readiness risks. |
| DP-noised quality reports — `SourceQualityReport` with ε budget (`gate.reports`, `gate.expectations`) | **2.10** | GOVERN | The measurement itself is privacy-preserving (rows-free, DP-noised aggregates), so privacy risk of the *measurement process* is examined and documented. |
| Source quality/lineage report *presence* — `SourceQualityReport` (`gate.reports`) | **2.8** | 2.1 | A datasheet-like artifact is required per source, so undocumented data is caught and transparency/accountability is documented. |
| Cross-source consistency — `schema_compatibility` + `pairwise_divergence` (`gate.compatibility`) | **2.5** | 2.11 | Multi-site schema/semantic consistency is measured across providers. |
| Incident-derived check selection — the taxonomy (`docs/TAXONOMY.md`) drives which checks run | **1.1** | 3.1 | The checks implemented are exactly the risks enumerated in MAP (the MAP→MEASURE loop, verbatim in 1.1). |
| Documented blind spots — label-detector blindness on balanced labels; detection≠withholding (`docs/FINDINGS.md`, `verification/VERIFICATION.md`) | **1.1** | 2.13 | Risks that "will not — or cannot — be measured are properly documented" (1.1's own words); TEVV effectiveness is itself evaluated (2.13). |
| Independent human coding of the taxonomy — seeded 30-incident sample + Cohen's κ (`experiments/code_agreement.py`) | **1.3** | 1.2 | Assessors who were not the front-line developers validate the measurement rule. |

## B. Taxonomy failure mode → MEASURE subcategory

Modes and detectability are from [`docs/TAXONOMY.md`](TAXONOMY.md) (counts quoted
from `experiments/results/taxonomy_coding.csv`); cross-domain detectability is
from `experiments/results/crossdomain_*.csv` (see `docs/FINDINGS.md`).

| # | Taxonomy mode | MEASURE (primary / secondary) | Rationale |
|---|---|---|---|
| 1 | Structural / schema | **2.1** / 2.5 | Schema/dtype/range are documented TEVV checks on data validity. |
| 2 | Completeness | **2.5** / 2.11 | Coverage gaps limit generalizability; missing subpopulations are a bias vector. |
| 3 | Distributional | **2.11** / 2.5 | Population skew / shift is the fairness-and-bias measurement, evaluated across sources. |
| 4 | Label | **2.3** / 2.5 | Label integrity is a documented assurance/data-quality criterion. |
| 5 | Provenance / documentation | **2.8** / 2.1 | Lineage/consent documentation is the transparency-and-accountability measurement. |
| 6 | Cross-source incompatibility | **2.5** / 2.11 | Multi-site validity/bias consistency (empirically absent from AIID; see TAXONOMY §6). |
| — | Readiness score + gate decision (spans all modes) | **1.1** / 1.2, 2.13 | The documented, pre-deployment go/no-go metric. |

## How this differentiates the paper (the intersection claim)

No prior work operationalizes a **pre-training data gate** *and* maps each of its
checks to a **specific, verified** NIST MEASURE subcategory *and* benchmarks the
checks **across domains** (Pillar B). Datasheets/Data-Statements (Gebru et al.,
Bender & Friedman) supply the documentation artifact that maps to 2.8/2.1;
Sambasivan et al. establish that data quality matters; the AI Incident Database
supplies the incidents (MAP). This project is the *integration* — incidents (MAP)
→ a private, benchmarked, pre-training gate (MEASURE 1.1/2.x) → documented
governance — and the mapping above is the connective tissue, now anchored to the
official NIST text rather than to remembered subcategory numbers.

## Verification note (project hard rule 2)

The subcategory numbers and wording were confirmed by fetching NIST AI 100-1
directly (URL above) and reading Table 3, not from memory. Record this in
[`verification/VERIFICATION.md`](../verification/VERIFICATION.md). If a reviewer
cites the NIST AI RMF **Playbook** (the companion that expands MEASURE 2 into
~13 actionable items), note that the Playbook's *suggested actions* elaborate the
same Core subcategories used here; this mapping is stated at the Core-subcategory
level, which is the authoritative, versioned numbering.
