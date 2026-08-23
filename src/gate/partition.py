"""Split a dataset across simulated sources and inject controlled quality issues.

TEST / SYNTHETIC FIXTURE. This module exists so that all AI-assisted
development and the smoke tests run on fake data — real restricted rows
(e.g. MIMIC-IV under a PhysioNet DUA) must never appear in an LLM prompt.
Real per-domain loaders arrive with Pillar B (see docs/WALKTHROUGH.md);
`degrade` and `split_sources` here are domain-agnostic and are reused by the
cross-domain degradation-injection harness.

Injectable degradations (per source, configurable):
  - null_rate: randomly null out a fraction of a column
  - drop_column: remove a required column
  - type_corrupt: cast numeric column to string
  - distribution_shift: resample to skew a column's distribution
  - label_noise: flip a fraction of labels
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_clinical(n_rows: int, seed: int) -> pd.DataFrame:
    """Synthetic table with a MIMIC-like schema — for development/tests only."""
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(65, 15, n_rows), 18, 100).round(0)
    los = np.clip(rng.gamma(2.0, 2.5, n_rows), 0.1, 60)
    lab_creatinine = np.clip(rng.lognormal(0.1, 0.5, n_rows), 0.2, 15)
    lab_hemoglobin = np.clip(rng.normal(12.5, 2.0, n_rows), 4, 20)
    n_diagnoses = rng.poisson(6, n_rows)
    # Mortality correlated with age, creatinine, LOS — logistic link
    logit = -6.0 + 0.04 * age + 0.5 * np.log(lab_creatinine + 0.1) + 0.05 * los
    mortality = (rng.random(n_rows) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame(
        {
            "age": age.astype(int),
            "gender": rng.choice(["M", "F"], n_rows),
            "los_days": los,
            "lab_creatinine": lab_creatinine,
            "lab_hemoglobin": lab_hemoglobin,
            "n_diagnoses": n_diagnoses,
            "mortality": mortality,
        }
    )


def split_sources(df: pd.DataFrame, n_sources: int, seed: int) -> list[pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    return [df.iloc[part].reset_index(drop=True) for part in np.array_split(idx, n_sources)]


def degrade(df: pd.DataFrame, spec: dict, seed: int) -> pd.DataFrame:
    """Apply a degradation spec to one source's data. Returns a copy."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    for column, rate in spec.get("null_rate", {}).items():
        mask = rng.random(len(out)) < rate
        out.loc[mask, column] = np.nan
    for column in spec.get("drop_column", []):
        out = out.drop(columns=[column], errors="ignore")
    for column in spec.get("type_corrupt", []):
        if column in out.columns:
            out[column] = out[column].astype(str)
    shift = spec.get("distribution_shift")
    if shift:
        col, factor = shift["column"], shift["factor"]
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") * factor
    ln = spec.get("label_noise")
    if ln:
        col, rate = ln["column"], ln["rate"]
        flip = rng.random(len(out)) < rate
        out.loc[flip, col] = 1 - out.loc[flip, col]
    return out
