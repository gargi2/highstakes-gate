"""Differential privacy mechanisms for the gate's quality reports.

This module is the highest-verification-priority code in the project.
Every noised statistic must document its sensitivity justification.

Laplace mechanism: to release f(D) with epsilon-DP, add Laplace(0, b) noise
where b = sensitivity(f) / epsilon, and sensitivity is the max change in f
from adding/removing one record (unbounded DP) or replacing one record
(bounded DP). We use UNBOUNDED (add/remove) sensitivity throughout —
state this in the paper.

VERIFY against diffprivlib before trusting:
    from diffprivlib.mechanisms import Laplace
    Laplace(epsilon=1.0, sensitivity=1.0).randomise(value)
"""

from __future__ import annotations

import numpy as np


def laplace_noise(sensitivity: float, epsilon: float, rng: np.random.Generator) -> float:
    """Draw one sample of Laplace(0, b) with b = sensitivity / epsilon."""
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    if sensitivity <= 0:
        raise ValueError(f"sensitivity must be > 0, got {sensitivity}")
    scale = sensitivity / epsilon
    return float(rng.laplace(loc=0.0, scale=scale))


def noised_count(true_count: int, epsilon: float, rng: np.random.Generator) -> float:
    """Row count with epsilon-DP.

    Sensitivity = 1: adding or removing one record changes a count by
    exactly 1. (Standard result; Dwork & Roth Ch. 3.)
    """
    return true_count + laplace_noise(sensitivity=1.0, epsilon=epsilon, rng=rng)


def noised_histogram(
    counts: np.ndarray, epsilon: float, rng: np.random.Generator
) -> np.ndarray:
    """Histogram release with epsilon-DP.

    Sensitivity = 1 for the whole histogram under add/remove DP: one record
    falls in exactly one bin, so removing it changes the L1 norm of the count
    vector by exactly 1. Therefore we add Laplace(1/epsilon) noise to EACH bin
    and the entire histogram release costs epsilon total (NOT epsilon per bin).
    (Dwork & Roth, Example 3.2 — VERIFY this citation before the paper.)

    Post-processing: clamp negatives to 0. Post-processing preserves DP.
    """
    counts = np.asarray(counts, dtype=float)
    noise = rng.laplace(loc=0.0, scale=1.0 / epsilon, size=counts.shape)
    return np.clip(counts + noise, 0.0, None)


def noised_null_fraction(
    null_count: int, total_count: int, epsilon: float, rng: np.random.Generator
) -> float:
    """Null fraction with epsilon-DP, computed from two noised counts.

    We split the budget: epsilon/2 to the null count, epsilon/2 to the total
    count, then divide. By sequential composition the release costs epsilon.
    Each count has sensitivity 1 (one record changes each count by at most 1).

    NOTE: the ratio of two noised counts is a biased estimator of the true
    fraction. For the readiness gate this is acceptable at moderate epsilon;
    quantify the bias empirically in the privacy-utility experiment.
    """
    n_null = null_count + laplace_noise(1.0, epsilon / 2.0, rng)
    n_total = total_count + laplace_noise(1.0, epsilon / 2.0, rng)
    if n_total <= 0:
        return 0.0
    return float(np.clip(n_null / n_total, 0.0, 1.0))


class PrivacyAccountant:
    """Tracks cumulative epsilon spent per source per reporting round.

    Basic (sequential) composition only: total epsilon = sum of epsilons.
    This is loose but simple and defensible for a v1 paper. Advanced
    composition / RDP is future work — say so in limitations.
    """

    def __init__(self, epsilon_budget: float):
        self.budget = epsilon_budget
        self.spent = 0.0
        self.ledger: list[tuple[str, float]] = []

    def spend(self, label: str, epsilon: float) -> None:
        if self.spent + epsilon > self.budget + 1e-9:
            raise RuntimeError(
                f"Privacy budget exceeded: spent {self.spent:.3f}, "
                f"requested {epsilon:.3f}, budget {self.budget:.3f}"
            )
        self.spent += epsilon
        self.ledger.append((label, epsilon))
