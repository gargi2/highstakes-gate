# Pivot Map — "Pre-Training Data-Readiness Gating for High-Stakes AI Systems"

**Proposed new framing:** a failure taxonomy derived from production incidents,
a validation gate benchmarked across domains, mapped to NIST AI RMF MEASURE.

**Honest scope note.** This is a *different and larger* paper than the
near-submittable FedCheck draft, not an edit of it. FedCheck's validation gate
is reusable as ONE component, but the pivot adds three pillars we have no
results for yet (incident taxonomy, cross-domain benchmark, NIST mapping) and
drops the federated-learning framing as the centerpiece. Estimated 3–4 months.
See "Strategy" at the bottom for the two-paper option.

---

## 1. What we already have (reusable assets)

| Asset | Reuse in pivot | Notes |
|---|---|---|
| Expectation-suite engine (5 checks), readiness score, blocking gate | **Core** — becomes "the validation gate" | Domain-agnostic already |
| Consensus detectors (label-rate, divergence) | Partial | FL-flavored; label-rate transfers, divergence needs robust threshold |
| DP mechanism (Laplace, verified) | Optional | Central only if privacy stays in scope; may be secondary in the new framing |
| FedAvg + poison experiment harness | Partial | FL-specific; new framing is "high-stakes AI," not necessarily federated |
| MIMIC-IV cohort + extraction pipeline | **1 of N domains** (healthcare) | Need 2–3 more domains |
| 28 verified references | ~10 reusable | Data-quality/validation/DP ones carry over; FL-specific ones less so |
| Honest findings (null + poison boundary) | Evidence of gate efficacy | Recast as "when the gate changes outcomes" |

## 2. What the new framing requires (gap analysis)

| Pillar | Have? | What's missing |
|---|---|---|
| **A. Failure taxonomy from production incidents** | ❌ none | Incident corpus + qualitative coding method + taxonomy structure. Net-new empirical work. |
| **B. Validation gate benchmarked across domains** | ⚠️ 1 domain | 2–3 non-healthcare domains (finance, government), run the gate, measure. |
| **C. NIST AI RMF MEASURE mapping** | ❌ none | Map each gate check/output to specific MEASURE subcategories; governance argument. |
| **D. "High-stakes AI" reframing (beyond FL)** | ⚠️ partial | Generalize gate framing; position against data-cascades / data-documentation literature. |

## 3. Reference shopping list (by pillar)

### Verified (2026-08-06)
- NIST AI RMF 1.0 (NIST AI 100-1), MEASURE function — nist.gov — Pillar C.
- Sambasivan et al., "Data Cascades in High-Stakes AI," CHI 2021, DOI 10.1145/3411764.3445518 — Pillars A, D (closest prior work).
- McGregor, "Preventing Repeated Real World AI Failures … AI Incident Database," arXiv:2011.08512 (AAAI/IAAI 2021) — Pillar A (corpus).
- Paeth et al., "Lessons for Editors of AI Incidents from the AIID," arXiv:2409.16425 (2024) — Pillar A (taxonomy precedent).
- Reusable from current refs.bib: Great Expectations, TFDV/DQ-tools survey (zhou2024dqtools), tabular DQ effects (mohammed2025dataquality), Dwork–Roth (if DP kept), MIMIC-IV.

### To gather + verify (per hard rule 4, fetch each before citing)
- **Pillar A (incidents/failure):** AIAAIC repository; ML testing survey (Zhang et al.); "Concrete Problems in AI Safety" (Amodei et al.); an incident-taxonomy methodology (grounded theory / thematic analysis reference).
- **Pillar C (governance/assurance):** EU AI Act (risk management articles); ISO/IEC 42001 (AI management system); Model Cards (Mitchell et al. 2019); Datasheets for Datasets (Gebru et al.); Data Statements (Bender & Friedman); "Hidden Technical Debt in ML Systems" (Sculley et al. 2015).
- **Pillar B (cross-domain datasets):** SEC EDGAR; NYC TLC Trip Records; CMS OpenPayments — each needs a citable source/dataset reference.
- **Pillar D:** data-validation-for-ML surveys; data-centric AI references.

Target: ~30–45 verified references (the 56-item FL reading list only partially
applies — much of it is FL-specific and will be dropped).

## 4. NIST AI RMF MEASURE mapping (first draft — to refine against the official playbook)

| Gate capability | MEASURE subcategory (approx.) | Note |
|---|---|---|
| Schema / dtype / range checks | MEASURE 2.x (TEVV; data validity) | "appropriate methods … applied" |
| Completeness (null thresholds) | MEASURE 2.x (data quality) | representativeness of data |
| Cross-site distribution divergence | MEASURE 2.11 / 2.x (bias, drift) | population/skew detection |
| Readiness score + gate decision | MEASURE 1.x (metrics selected/applied) | documented, pre-deployment |
| DP-noised quality reports | MEASURE 2.x (privacy) + GOVERN | privacy-preserving assessment |
| Incident-derived checks (new) | MAP → MEASURE loop | taxonomy informs what to measure |

**VERIFY** every subcategory number against the official NIST AI 100-1 text and
the MEASURE playbook before the paper claims a mapping — the numbers above are
placeholders from a summary, not the source.

## 5. Strategy (honest recommendation)

Do NOT throw away the near-done FedCheck paper to chase this. A cleaner path:
- **Ship FedCheck now** (workshop/JDIQ) as the *validated gate* — it becomes the
  empirical backbone this pivot cites.
- **Build the high-stakes / NIST paper as the bigger follow-up**, reusing the
  gate and adding the taxonomy + cross-domain + governance mapping. Two coherent
  papers with a throughline beat one delayed, over-scoped paper — and it fits
  the NIW "sustained contribution" narrative better.

## 6. Proceed plan (concrete next steps)

1. Lock the incident corpus: download the public AI Incident Database export
   (incidentdatabase.ai) + review its two existing taxonomies (don't reinvent).
2. Batch-verify the ~15–20 Pillar A/C/D anchor references (governance + failure).
3. Finalize the NIST MEASURE mapping table against the official playbook.
4. Choose 2–3 benchmark domains + citable datasets; extend the gate harness.
5. Only then: taxonomy coding → cross-domain runs → write-up.
