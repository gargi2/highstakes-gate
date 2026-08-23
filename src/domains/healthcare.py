"""Healthcare domain (Pillar B).

Synthetic path reuses FedCheck's MIMIC-like generator (`gate.partition.
make_synthetic_clinical`) so tests/smoke never touch restricted rows. Real path
reads a de-identified public health table via --data-root (a CSV whose columns
match `required_columns`); MIMIC-IV under a PhysioNet DUA is read-only and must
be pointed to locally by the human, never uploaded (CLAUDE.md).

Task: in-hospital mortality (binary) from a small tabular feature set.
"""
from __future__ import annotations

import glob
import os
from typing import Iterator

import pandas as pd

from gate.partition import make_synthetic_clinical
from .base import Domain, read_csv_chunked, register, aggregate_summary

FEATURES = ["age", "los_days", "lab_creatinine", "lab_hemoglobin", "n_diagnoses"]
LABEL = "mortality"
REQUIRED = {
    "age": "int", "los_days": "float", "lab_creatinine": "float",
    "lab_hemoglobin": "float", "n_diagnoses": "int", "mortality": "int",
}
HIST_RANGES = {
    "age": (18.0, 100.0), "los_days": (0.0, 60.0), "lab_creatinine": (0.0, 15.0),
    "lab_hemoglobin": (4.0, 20.0), "n_diagnoses": (0.0, 20.0),
}


def _resolve_csv(data_root: str) -> str:
    if os.path.isfile(data_root):
        return data_root
    hits = sorted(glob.glob(os.path.join(data_root, "*.csv")))
    if not hits:
        raise RuntimeError(f"healthcare: no CSV found under {data_root}")
    return hits[0]


def load(data_root: str, chunksize: int) -> Iterator[pd.DataFrame]:
    """Stream a public clinical CSV, selecting only the modelling columns."""
    path = _resolve_csv(data_root)
    want = list(REQUIRED)
    for chunk in read_csv_chunked(path, chunksize=chunksize):
        cols = [c for c in want if c in chunk.columns]
        yield chunk[cols].copy()


DOMAIN = register(Domain(
    name="healthcare",
    label_column=LABEL,
    feature_columns=FEATURES,
    required_columns=REQUIRED,
    histogram_ranges=HIST_RANGES,
    synthetic=make_synthetic_clinical,
    loader=load,
    divergence_threshold=0.026,  # calibrated on clean sources at epsilon=inf
    notes="In-hospital mortality; synthetic = MIMIC-like generator (FedCheck).",
))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Healthcare loader — aggregates only, no rows printed.")
    ap.add_argument("--data-root", default=None, help="CSV file or dir; omit for synthetic.")
    ap.add_argument("--n-rows", type=int, default=4000)
    ap.add_argument("--max-rows", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    df, prov = DOMAIN.get_frame(a.n_rows, a.seed, a.data_root, a.max_rows)
    print(f"provenance={prov}")
    print(aggregate_summary(df, DOMAIN))
