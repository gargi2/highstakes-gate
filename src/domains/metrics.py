"""Downstream evaluation metrics for the redesigned downstream arm (Pillar C, item 1).

The original benchmark measured only aggregate ROC-AUC on a clean holdout, which is
the LEAST sensitive thing to data corruption (verified: clean ≈ corrupted-ungated,
`docs/FINDINGS.md` Finding 5). Representation/label damage shows up in **worst-group
accuracy**, **subgroup false-negative-rate gap**, and **calibration (ECE)**, not the
mean. This module adds those, numpy-only (no new dependency — CLAUDE.md Stack),
reusing the tiny logistic model in `domains.base`.

Subgroups are quantile bins of one equity-axis feature evaluated on the clean
holdout (so the subgrouping is never contaminated by the injected corruption, which
is applied only to training sources). Groups too small or single-class for a stable
statistic are dropped from that statistic and counted in `n_groups_used`.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import _prep_xy, fit_logreg, predict_proba, roc_auc

MIN_GROUP = 10        # a subgroup needs this many rows to enter worst-group accuracy
MIN_GROUP_POS = 5     # ...and this many positives to enter the FNR gap
ECE_BINS = 10


@dataclass
class DownstreamMetrics:
    auc: float
    acc: float
    worst_group_acc: float
    subgroup_fnr_gap: float
    ece: float
    n_groups_used: int


def _accuracy(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> float:
    return float(((p >= thr).astype(float) == y).mean())


def _quantile_bins(values: np.ndarray, k: int) -> np.ndarray:
    """Assign each value to one of up to k quantile bins (ties collapse bins)."""
    v = np.asarray(values, dtype=float)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return np.zeros(len(v), dtype=int)
    edges = np.unique(np.quantile(finite, np.linspace(0, 1, k + 1)))
    if edges.size < 2:
        return np.zeros(len(v), dtype=int)
    # np.digitize with interior edges -> bins 0..len(edges)-2
    return np.clip(np.digitize(v, edges[1:-1], right=False), 0, len(edges) - 2)


def _worst_group_acc(y, p, groups, thr=0.5) -> tuple[float, int]:
    accs = []
    for g in np.unique(groups):
        m = groups == g
        if m.sum() >= MIN_GROUP:
            accs.append(_accuracy(y[m], p[m], thr))
    return (min(accs) if accs else float("nan")), len(accs)


def _fnr_gap(y, p, groups, thr=0.5) -> float:
    """max-min false-negative rate across groups with enough positives."""
    fnrs = []
    for g in np.unique(groups):
        m = groups == g
        yp, pp = y[m], p[m]
        pos = yp == 1
        if pos.sum() >= MIN_GROUP_POS:
            fn = ((pp < thr) & pos).sum()
            fnrs.append(fn / pos.sum())
    return float(max(fnrs) - min(fnrs)) if len(fnrs) >= 2 else float("nan")


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = ECE_BINS) -> float:
    """Standard ECE: sum_b (n_b/n) * |acc_b - conf_b| over equal-width prob bins."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    n = len(p)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        nb = int(m.sum())
        if nb == 0:
            continue
        conf = float(p[m].mean())
        acc = float(y[m].mean())  # fraction actually positive == empirical accuracy of the confidence
        ece += (nb / n) * abs(acc - conf)
    return float(ece)


def evaluate_downstream(train_df: pd.DataFrame, test_df: pd.DataFrame,
                        features: list[str], label: str, subgroup_column: str,
                        seed: int, k_groups: int = 4) -> DownstreamMetrics:
    """Fit the tiny logreg on train_df; score sensitive metrics on the clean test_df.

    All-NaN when train is unusable (too small / single-class), matching train_eval.
    Subgroups come from the RAW (unstandardized) equity-axis column in test_df.
    """
    nan = DownstreamMetrics(*( [float("nan")] * 5 + [0] ))
    if train_df is None or len(train_df) < 20:
        return nan
    # A degraded frame can present an all-NaN (dropped) column -> nanmean of an
    # empty slice; _prep_xy handles it (imputes 0), so silence the expected warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        Xtr, ytr = _prep_xy(train_df, features, label)
        if len(np.unique(ytr)) < 2:
            return nan
        w = fit_logreg(Xtr, ytr, seed=seed)
        Xte, yte = _prep_xy(test_df, features, label)
        p = predict_proba(Xte, w)

    col = subgroup_column if subgroup_column in test_df.columns else features[0]
    raw = pd.to_numeric(test_df[col], errors="coerce").to_numpy(dtype=float)
    groups = _quantile_bins(raw, k_groups)

    wga, n_used = _worst_group_acc(yte, p, groups)
    return DownstreamMetrics(
        auc=roc_auc(yte, p),
        acc=_accuracy(yte, p),
        worst_group_acc=wga,
        subgroup_fnr_gap=_fnr_gap(yte, p, groups),
        ece=expected_calibration_error(yte, p),
        n_groups_used=n_used,
    )
