#!/usr/bin/env python
"""Pillar A — inter-rater reliability for the data-readiness coding.

Compares the machine coding (experiments/results/taxonomy_sample_machine_labels.csv)
against an independent human coding of the same seeded 30-incident sample, and
reports percent agreement + Cohen's kappa for (a) assigned mode and
(b) pre-training detectability. Also reports filter precision: the fraction of
sampled incidents the human judged NOT to be a data-readiness failure ("other").

The human coding is produced by filling docs/taxonomy_sample_for_human_review.md
and transcribing the answers into:

    experiments/results/taxonomy_sample_human_labels.csv
    columns: incident_id,mode,detectability

`mode` must be one of: structural_schema, completeness, distributional, label,
provenance_documentation, cross_source, other.
`detectability` must be one of: yes, partial, no.

Reproduce (after the human file exists):
    .venv/Scripts/python.exe experiments/code_agreement.py

If the human file is absent, this prints an empty template and exits 0 (nothing
is fabricated — agreement stats appear only once real human labels are present).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
KEY = RESULTS / "taxonomy_sample_machine_labels.csv"
HUMAN = RESULTS / "taxonomy_sample_human_labels.csv"
OUT = RESULTS / "taxonomy_agreement.txt"


def cohen_kappa(a: list[str], b: list[str]) -> tuple[float, float]:
    """Return (observed_agreement, cohen_kappa) for two aligned label lists."""
    n = len(a)
    assert n == len(b) and n > 0
    labels = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    # expected agreement from marginals
    pe = 0.0
    for lab in labels:
        pa = sum(1 for x in a if x == lab) / n
        pb = sum(1 for x in b if x == lab) / n
        pe += pa * pb
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 1e-12 else float("nan")
    return po, kappa


def confusion(a: list[str], b: list[str]) -> str:
    df = pd.crosstab(pd.Series(a, name="machine"), pd.Series(b, name="human"))
    return df.to_string()


def main() -> int:
    if not HUMAN.exists():
        RESULTS.mkdir(parents=True, exist_ok=True)
        key = pd.read_csv(KEY)
        tmpl = key[["incident_id"]].copy()
        tmpl["mode"] = ""
        tmpl["detectability"] = ""
        tmpl_path = RESULTS / "taxonomy_sample_human_labels.TEMPLATE.csv"
        tmpl.to_csv(tmpl_path, index=False)
        print(
            "No human labels yet.\n"
            f"  Fill docs/taxonomy_sample_for_human_review.md, then transcribe to:\n"
            f"    {HUMAN}\n"
            f"  A blank template was written to:\n    {tmpl_path}\n"
            "  Re-run this script afterwards to compute agreement."
        )
        return 0

    key = pd.read_csv(KEY).set_index("incident_id")
    hum = pd.read_csv(HUMAN).set_index("incident_id")
    common = sorted(set(key.index) & set(hum.index))
    if not common:
        print("ERROR: no overlapping incident_id between machine key and human labels.")
        return 1

    m_mode = [str(key.loc[i, "mode"]).strip() for i in common]
    h_mode = [str(hum.loc[i, "mode"]).strip().lower() for i in common]
    m_det = [str(key.loc[i, "detectability"]).strip() for i in common]
    h_det = [str(hum.loc[i, "detectability"]).strip().lower() for i in common]

    po_mode, k_mode = cohen_kappa(m_mode, h_mode)
    po_det, k_det = cohen_kappa(m_det, h_det)
    n_other = sum(1 for x in h_mode if x == "other")
    filter_precision = 1 - n_other / len(common)

    lines = [
        "# Pillar A inter-rater reliability",
        f"n (sampled & double-coded): {len(common)}",
        "",
        "## Assigned mode",
        f"  percent agreement: {po_mode:.3f}",
        f"  Cohen's kappa:     {k_mode:.3f}",
        "",
        "## Pre-training detectability",
        f"  percent agreement: {po_det:.3f}",
        f"  Cohen's kappa:     {k_det:.3f}",
        "",
        "## Filter precision (human judged sampled incident IS data-readiness)",
        f"  human 'other' count: {n_other} / {len(common)}",
        f"  filter precision:    {filter_precision:.3f}",
        "",
        "## Mode confusion (rows=machine, cols=human)",
        confusion(m_mode, h_mode),
        "",
        "## Detectability confusion (rows=machine, cols=human)",
        confusion(m_det, h_det),
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
