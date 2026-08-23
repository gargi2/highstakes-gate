"""Confident-learning label-noise detector (Pillar C item 3) — closes Blocker 2.

The reused gate's consensus label detector watches only the DP-noised positive-label
RATE, so it is provably blind to SYMMETRIC label noise on balanced targets
(`p' = p + r(1-2p) -> 0` as `p -> 0.5`; finance `adverse` ~= 0.42 -> recall 0.00;
see docs/FINDINGS.md Finding 1). This module ADDS a detector with a mechanism that
can see such noise, without modifying the frozen FedCheck gate (`src/gate/`).

Idea (Northcutt et al., "Confident Learning," JAIR 2021, arXiv:1911.00068 —
VERIFY before citing): a model cross-validated on a source's own rows will keep
DISAGREEING with labels that were flipped, because the flip is not learnable from
the features. The per-source **self-disagreement rate** (out-of-fold predicted class
!= given label) is therefore elevated on a noisy source EVEN WHEN THE MARGINAL DOES
NOT MOVE. It is a single scalar aggregate — rows never leave the source — so it fits
the gate's private, federated architecture and is DP-noisable like the other
released statistics (sensitivity 1/n as a noised proportion; a rigorous DP analysis
of a CV-derived statistic is future work, flagged not overclaimed).

Consensus then flags sources whose disagreement rate exceeds the cross-source
consensus (median) by a margin — the same >50%-clean assumption the gate's other
consensus detectors use, but ONE-SIDED (only high disagreement is suspicious).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gate.privacy import laplace_noise
from gate.reports import CheckResult
from .base import _prep_xy, fit_logreg, predict_proba

LABEL_QUALITY_CHECK_ID = "consensus_label_quality"


def self_disagreement_rate(frame: pd.DataFrame, features: list[str], label: str,
                           seed: int, k: int = 4, confidence_margin: float = 0.0,
                           epsilon: float | None = None) -> float:
    """Out-of-fold disagreement between the model's prediction and the given label.

    Rows-free scalar in [0, 1]: fit a k-fold cross-validated logistic model on the
    source's own rows, predict each held-out fold, and return the fraction whose
    predicted class disagrees with the label. With confidence_margin > 0, count only
    CONFIDENT disagreements (|p - 0.5| > margin) — Northcutt-style, more robust to a
    weak model. Returns NaN if the source is too small or single-class.

    epsilon (optional): release the rate under epsilon-DP by adding Laplace(1/(n*eps))
    noise (sensitivity 1/n for a proportion), then clamp to [0, 1]. Approximate —
    treats the fitted fold-models as post-processing; see module docstring.
    """
    X, y = _prep_xy(frame, features, label)
    n = len(y)
    if n < max(20, 2 * k) or len(np.unique(y)) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    oof = np.full(n, np.nan)
    for f in folds:
        tr = np.setdiff1d(idx, f, assume_unique=False)
        if len(np.unique(y[tr])) < 2:
            oof[f] = float(y[tr].mean()) if len(tr) else 0.5
            continue
        w = fit_logreg(X[tr], y[tr], seed=seed)
        oof[f] = predict_proba(X[f], w)
    pred = (oof >= 0.5).astype(float)
    disagree = pred != y
    if confidence_margin > 0.0:
        confident = np.abs(oof - 0.5) > confidence_margin
        rate = float((disagree & confident).sum()) / max(int(confident.sum()), 1)
    else:
        rate = float(disagree.mean())
    if epsilon is not None:
        rate += laplace_noise(sensitivity=1.0 / n, epsilon=epsilon, rng=rng)
        rate = float(min(max(rate, 0.0), 1.0))
    return rate


def label_quality_checks(disagreements: dict[str, float],
                         max_abs_deviation: float = 0.05) -> dict[str, list[CheckResult]]:
    """Flag sources whose self-disagreement exceeds the cross-source consensus.

    One-sided: only disagreement ABOVE (median + max_abs_deviation) is flagged
    (high disagreement = likely label noise; low is fine). Mirrors the gate's
    consensus pattern (median = robust to <50% degraded sources). Sources with a
    NaN rate (unusable) pass by default. Returns source_id -> [CheckResult].

    max_abs_deviation is a detector hyperparameter, calibrated on CLEAN sources
    (like `Domain.divergence_threshold`; docs/WALKTHROUGH.md), NOT on detection
    outcome — no leakage. At realistic source sizes the clean self-disagreement rate
    is very tight (std ~0.003 at ~6-7k rows), and a flipped source sits ~+0.10 above,
    so 0.05 separates them with margin. Small/high-variance sources may need a larger
    value; recalibrate per domain on clean data.
    """
    valid = {s: r for s, r in disagreements.items() if r == r}  # drop NaN
    out: dict[str, list[CheckResult]] = {s: [] for s in disagreements}
    if len(valid) < 2:
        return out
    consensus = float(np.median(list(valid.values())))
    for s, r in disagreements.items():
        if r != r:  # NaN -> pass (no evidence)
            out[s].append(CheckResult(LABEL_QUALITY_CHECK_ID, True, "rate=nan (unusable)"))
            continue
        excess = r - consensus
        ok = excess <= max_abs_deviation
        out[s].append(CheckResult(
            LABEL_QUALITY_CHECK_ID, ok,
            f"disagree={r:.3f} vs median={consensus:.3f} (+{excess:.3f}) thr={max_abs_deviation}"))
    return out
