#!/usr/bin/env python
"""Regenerate the SAME degraded sources the gate sees — for the external-baseline
comparison (NEXT_EXPERIMENTS item 6, run in Colab: GE / TFDV / Deequ).

Reuses the frozen gate's `split_sources` / `degrade` and the domain registry with
the identical seed arithmetic as `experiments/run_crossdomain.py`, so the baselines
validate byte-identical inputs to the gate — apples to apples.

DATA SAFETY (CONVENTIONS.md "Local data paths"): defaults to SYNTHETIC frames, so nothing
leaves the machine. A PUBLIC extract may be passed via `data_root` for finance
(EDGAR) or government (CMS Open Payments); healthcare MUST stay synthetic in any
shared/Colab notebook — MIMIC-IV is DUA-bound and its rows must never be uploaded.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gate.partition import split_sources, degrade   # noqa: E402
from domains import REGISTRY                          # noqa: E402
from domains import degradations as deg               # noqa: E402


def make_cell_sources(
    domain_name: str,
    mode: str,
    severity: str,
    seed: int,
    *,
    n_sources: int = 6,
    n_degraded: int = 2,
    n_rows: int = 4000,
    max_rows: int = 40000,
    data_root: str | None = None,
) -> dict:
    """Return the sources for one (domain, mode, severity, seed) cell.

    Mirrors run_crossdomain.run_cell's construction EXACTLY (same permutation,
    holdout, split, degraded indices, and degrade seeds), so a baseline run here
    is comparable to the gate's `det_recall` in `crossdomain_<domain>.csv`.

    Returns dict:
      domain      : the Domain object (for columns/label/thresholds)
      provenance  : "synthetic" | "real:<path>"
      reference   : clean holdout DataFrame (build the baseline's schema from this)
      sources     : list of (source_df, is_degraded) — the datasets to validate
    """
    if domain_name not in REGISTRY:
        raise KeyError(f"unknown domain {domain_name!r}; known: {list(REGISTRY)}")
    domain = REGISTRY[domain_name]
    df, prov = domain.get_frame(n_rows, seed, data_root, max_rows)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_test = max(20, int(0.2 * len(df)))
    reference = df.iloc[idx[:n_test]].reset_index(drop=True)     # clean holdout
    train_df = df.iloc[idx[n_test:]].reset_index(drop=True)

    sources = split_sources(train_df, n_sources, seed)
    degraded_idx = set(range(n_degraded)) if mode != "none" else set()
    spec = (deg.build_spec(mode, severity, domain.feature_columns, domain.label_column)
            if mode != "none" else {})

    out = []
    for i, src in enumerate(sources):
        if i in degraded_idx:
            src = degrade(src, spec, seed * 97 + i)
        out.append((src.reset_index(drop=True), i in degraded_idx))

    return {"domain": domain, "provenance": prov, "reference": reference, "sources": out}


if __name__ == "__main__":  # rows-free self-check
    for m in ["none"] + deg.MODES:
        cell = make_cell_sources("healthcare", m, "high", seed=0, n_rows=400, n_sources=5)
        n_deg = sum(is_d for _, is_d in cell["sources"])
        cols0 = list(cell["sources"][0][0].columns)
        print(f"mode={m:>14}  n_sources={len(cell['sources'])}  n_degraded={n_deg}  "
              f"src0_cols={len(cols0)}  prov={cell['provenance']}")
