# Pivot Scoping Pack — judge feasibility before committing

For deciding whether to commit to *"Pre-training data-readiness gating for
high-stakes AI systems: a failure taxonomy from production incidents, a
validation gate benchmarked across domains, mapped to NIST AI RMF MEASURE."*
Companion to `PIVOT_MAP.md`. Read the feasibility verdict (§4) first.

## 1. Reference backbone — VERIFIED against source (2026-08-06)

Enough to anchor all four pillars; each confirmed by title/authors/venue/DOI.

**Governance & data documentation**
- NIST AI RMF 1.0 (NIST AI 100-1), MEASURE function — nist.gov.
- Mitchell et al., "Model Cards for Model Reporting," FAccT 2019 — DOI 10.1145/3287560.3287596.
- Gebru et al., "Datasheets for Datasets," CACM 2021 — DOI 10.1145/3458723.
- Bender & Friedman, "Data Statements for NLP," TACL 2018 — ACL Q18-1041.

**Failure / incident literature**
- McGregor, "Preventing Repeated Real World AI Failures … AI Incident Database," arXiv:2011.08512 (AAAI/IAAI 2021).
- Paeth et al., "Lessons for Editors of AI Incidents from the AIID," arXiv:2409.16425 (2024).
- Sambasivan et al., "Data Cascades in High-Stakes AI," CHI 2021 — DOI 10.1145/3411764.3445518 (CLOSEST PRIOR WORK).
- Sculley et al., "Hidden Technical Debt in Machine Learning Systems," NeurIPS 2015.
- Amodei et al., "Concrete Problems in AI Safety," arXiv:1606.06565 (2016).
- Zhang et al., "Machine Learning Testing: Survey, Landscapes and Horizons," IEEE TSE 2020 — DOI 10.1109/TSE.2019.2962027.

**Reusable from FedCheck refs.bib:** Great Expectations, DQ-tools survey
(zhou2024dqtools), tabular DQ effects (mohammed2025dataquality), MIMIC-IV,
Dwork–Roth (if DP retained).

**Still to verify during execution (not blocking the decision):** EU AI Act
(2024 regulation), ISO/IEC 42001 (2023), a data-centric-AI survey, and citable
dataset sources for the cross-domain benchmark (SEC EDGAR, NYC TLC Trip
Records, CMS OpenPayments).

## 2. Failure taxonomy — proposed structure (the qualitative contribution)

Axis: **data-readiness failure modes detectable BEFORE training**, each grounded
in real AIID incidents and mapped to a NIST MEASURE subcategory. Draft top level:

1. **Structural / schema** — missing columns, type mismatches, encoding drift.
2. **Completeness** — nulls, missing subpopulations, coverage gaps.
3. **Distributional** — population skew, covariate/label shift, non-representative sampling.
4. **Label** — label noise, annotation error, leakage, definitional drift.
5. **Provenance / documentation** — unknown lineage, undocumented data (the Datasheets/Data-Statements gap).
6. **Cross-source incompatibility** — multi-site schema/semantic misalignment.

Each taxonomy entry carries: example incident (AIID id) · does our gate detect
it pre-training? (yes/partial/no) · NIST MEASURE subcategory. This table *is*
the paper's spine — it connects incidents → gate → governance.

*Novelty note:* AIID already ships two incident taxonomies and Sambasivan
already typologizes data cascades. The novelty is NOT "a taxonomy of AI
incidents" — it is the **pre-training-detectability lens + the operational gate
+ the NIST MEASURE mapping**. Frame it that way or a reviewer will cite prior
taxonomies.

## 3. NIST AI RMF MEASURE mapping — first draft (category level)

MEASURE has four categories; exact subcategory numbers to be confirmed against
the official NIST playbook before the paper asserts them.

| Gate capability | MEASURE category (subcat. TBD-verify) | Governance claim |
|---|---|---|
| Schema / dtype / range checks | MEASURE 2 (TEVV; validity) | data validity measured pre-deployment |
| Completeness thresholds | MEASURE 2 (representativeness) | coverage/representativeness assessed |
| Distribution divergence | MEASURE 2 (~2.11 bias; drift) | population skew detected before training |
| Readiness score + gate decision | MEASURE 1 (metrics selected/applied) | documented go/no-go metric |
| DP-noised quality reports | MEASURE 2 (privacy) + GOVERN | privacy-preserving measurement |
| Incident-derived checks | MAP→MEASURE loop | taxonomy sets what to measure |

This mapping is the strongest differentiation: no prior work operationalizes a
pre-training gate *and* maps it to NIST MEASURE *and* benchmarks it across
domains. That intersection is the defensible contribution.

## 4. Feasibility verdict (honest)

| Pillar | Effort (solo) | Reuse | Novelty risk |
|---|---|---|---|
| A. Incident taxonomy | ~3–4 wks (AIID is public, ~750+ incidents) | none | Medium — prior taxonomies exist; lean on the readiness lens |
| B. Cross-domain benchmark | ~4–6 wks (2–3 new pipelines) | gate harness reuses | Low — mostly engineering |
| C. NIST MEASURE mapping | ~1–2 wks (writing) | — | Low — high differentiation |
| D. High-stakes reframing | writing | FedCheck findings | Low |

**Total ~3–4 months.** The individual pieces each have prior art (AIID
taxonomies, Sambasivan cascades, NIST RMF, data-doc frameworks); the
**integration is genuinely novel and useful** and that is the whole pitch:
incidents → a private, benchmarked, pre-training gate → NIST MEASURE compliance.

**Recommendation stands:** ship FedCheck first (it becomes this paper's
validated-gate citation), then build this as the ambitious follow-up. But the
pivot is feasible and the reference/where-it-fits picture is now concrete enough
to commit if you want to.

## 5. Minimum-viable version (if you want a faster first output)

A leaner paper that still lands: taxonomy (Pillar A) + NIST MEASURE mapping
(Pillar C) + the *existing* healthcare gate results as the single-domain
instantiation — deferring cross-domain (Pillar B) to future work. ~6–8 weeks,
reuses everything, and the incidents→gate→NIST spine is intact.
