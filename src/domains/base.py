"""Cross-domain benchmark: the Domain contract + shared, dependency-free helpers.

Pillar B runs the SAME validation gate (`src/gate/`, reused from FedCheck,
untouched) and the SAME taxonomy-derived degradations (`src/domains/degradations.py`,
wrapping `gate.partition.degrade`) across several high-stakes domains. A domain
plugs in by constructing one `Domain` here; see `docs/WALKTHROUGH.md`.

Everything in this module is numpy/pandas only — no sklearn/scipy — so the
harness has no new hard dependency (see CONVENTIONS.md Stack). The downstream model
is a tiny seeded logistic regression with a numpy ROC-AUC, enough to show
"does gating change the downstream metric?" cheaply.

AGGREGATES ONLY. Loaders and the CLI here print shapes/counts/decisions, never
rows — restricted rows (e.g. a DUA-bound clinical table) must never reach a
prompt or a committed file (CONVENTIONS.md "Local data paths").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np
import pandas as pd

from gate.expectations import Expectation

# Chunked-read default: keep memory bounded on the real multi-GB extracts.
DEFAULT_CHUNKSIZE = 200_000


@dataclass
class Domain:
    """Everything the harness needs to benchmark one domain.

    feature_columns are the numeric features fed to the gate's histograms /
    divergence detectors AND the downstream model. required_columns (name ->
    dtype-prefix) drive the expectation suite and gate schema-compatibility.
    """

    name: str
    label_column: str
    feature_columns: list[str]
    required_columns: dict[str, str]
    histogram_ranges: dict[str, tuple[float, float]]
    synthetic: Callable[[int, int], pd.DataFrame]
    loader: Callable[..., Iterator[pd.DataFrame]]
    null_threshold: float = 0.05
    min_rows: int = 50
    # Cross-source JS-divergence flag threshold for the consensus detector.
    # A gate HYPERPARAMETER (passed to gate.consensus, gate untouched), calibrated
    # per domain just above the clean-source divergence at epsilon=inf, because
    # the JS baseline depends on each domain's feature scale and bin resolution.
    divergence_threshold: float = 0.08
    # Equity axis for the downstream subgroup metrics (experiments/run_downstream.py).
    # None -> the harness falls back to feature_columns[0]. Binned into quantiles on
    # the clean holdout; worst-group accuracy / FNR gap / ECE are computed over the bins.
    subgroup_column: str | None = None
    notes: str = ""

    @property
    def histogram_columns(self) -> list[str]:
        return list(self.feature_columns)

    def build_suite(self) -> list[Expectation]:
        """A per-domain expectation suite over the SAME five gate check kinds."""
        exps: list[Expectation] = []
        for col, dt in self.required_columns.items():
            exps.append(Expectation(f"exists_{col}", "column_exists", column=col, severity="blocking"))
            if dt in ("int", "float"):
                exps.append(
                    Expectation(f"dtype_{col}", "column_dtype", column=col,
                                params={"dtype": dt}, severity="blocking")
                )
        for col in self.feature_columns:
            exps.append(
                Expectation(f"null_{col}", "null_fraction_max", column=col,
                            params={"max": self.null_threshold}, severity="blocking")
            )
        exps.append(Expectation("min_rows", "min_row_count",
                                params={"min": self.min_rows}, severity="major"))
        return exps

    def get_frame(self, n_rows: int, seed: int, data_root: str | None,
                  max_rows: int, chunksize: int = DEFAULT_CHUNKSIZE) -> tuple[pd.DataFrame, str]:
        """Return (frame, provenance). Real data if data_root given, else synthetic.

        Real path streams chunks and stops once max_rows is reached — never
        loads the whole extract, never returns identifiers (loaders select only
        the modelling columns).
        """
        if data_root:
            parts, total = [], 0
            for chunk in self.loader(data_root=data_root, chunksize=chunksize):
                parts.append(chunk)
                total += len(chunk)
                if total >= max_rows:
                    break
            if not parts:
                raise RuntimeError(f"{self.name}: loader yielded no rows from {data_root}")
            df = pd.concat(parts, ignore_index=True).head(max_rows)
            return df, f"real:{data_root}"
        return self.synthetic(n_rows, seed), "synthetic"


# --------------------------------------------------------------------------
# Dependency-free downstream model + metric (numpy only).
# --------------------------------------------------------------------------
def _prep_xy(df: pd.DataFrame, features: list[str], label: str) -> tuple[np.ndarray, np.ndarray]:
    """Coerce features to numeric, mean-impute, standardize; return (X, y).

    Robust to degraded frames: a dropped column becomes an all-imputed zero
    column, a string-corrupted column coerces to NaN then imputes. This is
    deliberately how a naive 'train on whatever survived' pipeline would behave.
    """
    cols = []
    for c in features:
        s = pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(np.nan, index=df.index)
        cols.append(s)
    X = pd.concat(cols, axis=1).to_numpy(dtype=float)
    mean = np.nanmean(X, axis=0)
    mean = np.where(np.isnan(mean), 0.0, mean)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(mean, inds[1])
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X = (X - X.mean(axis=0)) / std
    y = pd.to_numeric(df[label], errors="coerce").fillna(0).to_numpy(dtype=float)
    return X, (y == 1).astype(float)


def fit_logreg(X: np.ndarray, y: np.ndarray, seed: int, steps: int = 300, lr: float = 0.1) -> np.ndarray:
    """Tiny L2-regularized logistic regression by full-batch gradient descent."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    w = rng.normal(0, 0.01, d + 1)
    lam = 1e-3
    for _ in range(steps):
        z = Xb @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad = Xb.T @ (p - y) / n + lam * w
        w -= lr * grad
    return w


def predict_proba(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    return 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC via the rank (Mann-Whitney U) identity. NaN if only one class."""
    y_true = np.asarray(y_true)
    pos = y_true == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def train_eval(train_df: pd.DataFrame, test_df: pd.DataFrame,
               features: list[str], label: str, seed: int) -> float:
    """Fit on train_df, return ROC-AUC on the clean test_df. NaN if unusable."""
    if train_df is None or len(train_df) < 20:
        return float("nan")
    Xtr, ytr = _prep_xy(train_df, features, label)
    if len(np.unique(ytr)) < 2:
        return float("nan")
    w = fit_logreg(Xtr, ytr, seed=seed)
    Xte, yte = _prep_xy(test_df, features, label)
    return roc_auc(yte, predict_proba(Xte, w))


# --------------------------------------------------------------------------
# Shared chunked CSV reader for the real loaders.
# --------------------------------------------------------------------------
def read_csv_chunked(path: str, usecols: list[str] | None = None,
                     chunksize: int = DEFAULT_CHUNKSIZE, **kw) -> Iterator[pd.DataFrame]:
    """Thin wrapper so every real loader streams identically (aggregates only)."""
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize,
                             low_memory=False, **kw):
        yield chunk


def aggregate_summary(df: pd.DataFrame, domain: "Domain") -> str:
    """A rows-free summary string for a loaded frame (for loader __main__)."""
    lines = [f"[{domain.name}] rows={len(df):,} cols={list(df.columns)}"]
    if domain.label_column in df.columns:
        lab = pd.to_numeric(df[domain.label_column], errors="coerce")
        pos = float((lab == 1).mean())
        lines.append(f"  label '{domain.label_column}': positive_rate={pos:.4f}")
    for c in domain.feature_columns:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            lines.append(f"  {c}: null={s.isna().mean():.3f} "
                         f"min={np.nanmin(s):.3g} med={np.nanmedian(s):.3g} max={np.nanmax(s):.3g}")
    return "\n".join(lines)


# Registry — domains self-register on import (see domains/__init__.py).
REGISTRY: dict[str, Domain] = {}


def register(domain: Domain) -> Domain:
    REGISTRY[domain.name] = domain
    return domain
