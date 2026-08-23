# highstakes-gate — Development Conventions

## Project
Research project (sole author: Gargi Desai) for the paper:
**"Pre-training data-readiness gating for high-stakes AI systems: a failure
taxonomy from production incidents, a validation gate benchmarked across
domains, mapped to NIST AI RMF MEASURE."**

The contribution is the **integration** — production incidents → a private,
benchmarked, pre-training validation gate → NIST AI RMF MEASURE — not any single
pillar. It is explicitly *not* "a taxonomy of AI incidents" (the AI Incident
Database already ships those) and *not* "data quality matters" (Sambasivan's
data-cascades work already showed that). Differentiate from both in every draft.

The validation gate is **reused from FedCheck** (the validated single-domain
instantiation, cited as such); this project generalizes it across domains and
maps it to governance.

## HARD RULES (carried over from FedCheck — do not relax)
1. **Never fabricate results.** A result exists only if it has been written to a
   file. Read the file and quote it. Never report a number from memory or
   estimation.
2. **Every citation is verified by fetching the actual source (arXiv ID / DOI /
   publisher) before it enters a draft.** No citations from memory. Anything
   unverified is marked `TODO-VERIFY` in `paper/refs.bib` and never cited in
   prose until confirmed.
3. **Fixed seeds.** Every experiment uses a fixed seed, and every results file
   records the exact command that reproduces it in its header.
4. **Human reviews AI-written code before commit.** AI drafts and smoke-tests;
   the human reads it before it is committed. Heavy compute (full extractions,
   full grids) runs outside the assistant.

## Stack
- Python 3.10+ (`.venv` in repo root)
- Core gate: numpy, pandas (a small custom expectation engine — no hard GX dep)
- Tests: pytest
- Later pillars add: matplotlib (figures), domain loaders (Pillar B)

## Package Layout
- Source code: `src/gate/` (import as `gate.*`; pytest resolves `src/` via
  `pyproject.toml` `pythonpath`)
- Tests: `tests/`
- Experiments: `experiments/` (code) + `experiments/results/` (CSVs, committed)
- Docs / scoping: `docs/`
- Paper: `paper/` (LaTeX + `refs.bib`)
- Expectation suites: `suites/`
- Verification log: `verification/VERIFICATION.md`

## Local data paths
- `data/` is **gitignored** — never commit raw data files of any kind.
- Any restricted dataset (e.g. PhysioNet/MIMIC under a DUA) is treated as
  **read-only**. Restricted rows must never appear in an assistant prompt.
- Output from any restricted source is **aggregates / shapes only** (counts,
  schemas, histograms, noised statistics) — **never rows**.

## Running Tests
```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```
Run from the repo root (the consensus tests load `suites/clinical_v1.json` by
relative path).
