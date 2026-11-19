"""Shared bootstrap 95% CI helper for the analysis scripts (NEXT_EXPERIMENTS item 7).

The unit of variability is the SEED: each seed is an independent run of the grid, so
resampling seeds (not individual rows) gives an honest run-to-run confidence interval
rather than one that shrinks artificially with the number of rows in a cell. Fixed RNG
seed keeps the CI reproducible (project hard rule 3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CI_SEED = 42


def per_seed_means(df: pd.DataFrame, value_col: str, seed_col: str = "seed") -> np.ndarray:
    """Collapse to one number per seed (mean of value_col within each seed).

    That per-seed vector is the bootstrap resample unit.
    """
    s = pd.to_numeric(df[value_col], errors="coerce")
    g = df.assign(_v=s).dropna(subset=["_v"]).groupby(seed_col)["_v"].mean()
    return g.to_numpy(dtype=float)


def bootstrap_ci(values, b: int = 10000, alpha: float = 0.05, seed: int = CI_SEED):
    """Percentile bootstrap CI for the MEAN of `values` (a per-seed vector).

    Returns (mean, lo, hi). Degenerate inputs are handled: empty -> all NaN;
    a single seed -> (v, v, v) (no spread to estimate).
    """
    v = np.asarray([x for x in np.asarray(values, dtype=float) if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    if v.size == 1:
        return (float(v[0]), float(v[0]), float(v[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(b, v.size))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(v.mean()), float(lo), float(hi))


def ci_from_df(df: pd.DataFrame, value_col: str, seed_col: str = "seed",
               b: int = 10000, seed: int = CI_SEED):
    """Convenience: per-seed means of value_col in df -> (mean, lo, hi)."""
    return bootstrap_ci(per_seed_means(df, value_col, seed_col), b=b, seed=seed)
