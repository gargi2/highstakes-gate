# highstakes-gate

Pre-training **data-readiness gating** for high-stakes AI systems.

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

See `CLAUDE.md` for the hard rules (no fabricated results, verified citations,
fixed seeds, human review before commit).
