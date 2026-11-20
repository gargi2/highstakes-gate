# highstakes-gate

Pre-training **data-readiness gating** for high-stakes AI systems.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24204022.svg)](https://doi.org/10.5281/zenodo.24204022)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Preprint:** [doi:10.5281/zenodo.24204022](https://doi.org/10.5281/zenodo.24204022)
(this DOI covers all versions).

**Data:** evaluated on MIMIC-IV, SEC EDGAR, and CMS Open Payments; no restricted
data rows are included in this repository. Committed results are aggregates only.

A failure taxonomy derived from production incidents, a privacy-preserving
validation gate benchmarked across domains, mapped to the NIST AI RMF MEASURE
function. Sole author: Gargi Desai.

The validation gate is reused from **FedCheck** (its validated single-domain
instantiation) and generalized here to be domain-agnostic. See `docs/PIVOT_MAP.md`
and `docs/PIVOT_SCOPE.md` for the four pillars, the taxonomy, and the NIST mapping.

## Layout
- `src/gate/` — the reused validation gate (expectation suite, consensus
  detectors, readiness score, DP reports, cross-source compatibility)
- `tests/` — verification tests (DP statistical properties, consensus detectors)
- `suites/` — expectation suites (JSON)
- `experiments/` — benchmark harness + results CSVs
- `docs/` — scoping, taxonomy, NIST mapping, findings
- `paper/` — LaTeX + `refs.bib`
- `data/` — **gitignored**; restricted data is read-only, aggregates-only in output

## Setup
```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pytest tests/ -q
```

See `CONVENTIONS.md` for the hard rules (no fabricated results, verified citations,
fixed seeds, human review before commit).

## Citation
See [`CITATION.cff`](CITATION.cff), or cite the preprint:
Desai, G. (2026). *Pre-training data-readiness gating for high-stakes AI systems.*
Zenodo. https://doi.org/10.5281/zenodo.24204022

## License
Code and original documentation are released under the [MIT License](LICENSE).

Third-party content keeps its own terms:
- Incident titles and descriptions quoted from the
  [AI Incident Database](https://incidentdatabase.ai/) (in `docs/taxonomy_sample_*.md`,
  `experiments/results/taxonomy_coding.csv`, and
  `experiments/results/taxonomy_sample_machine_labels.csv`) are CC BY-SA 4.0.
  Attribution: AI Incident Database. See `docs/DATA_AIID.md`.
- `docs/financial-statement-data-sets.pdf` is SEC documentation (a U.S. Government work).
- The datasets themselves (MIMIC-IV, SEC EDGAR, CMS Open Payments, AIID) are not
  redistributed here. Obtain them from their providers under their own terms;
  MIMIC-IV requires PhysioNet credentialing and its data use agreement.
