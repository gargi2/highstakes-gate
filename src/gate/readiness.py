"""Composite readiness score and gate decision per source.

v1 scoring is deliberately simple and interpretable (weighted average of
three subscores in [0,1]). Weights are a hyperparameter — report a
sensitivity analysis over weights in the paper, do NOT tune them on the
test outcome (that would be leakage into the headline claim).

Gate:
  score >= ready_threshold          -> READY
  fixable_threshold <= s < ready    -> FIXABLE (schema ok, quality issues)
  below, or schema incompatible     -> EXCLUDE
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .reports import SourceQualityReport


@dataclass
class ReadinessResult:
    source_id: str
    score: float
    check_pass_rate: float
    schema_ok: bool
    mean_divergence: float | None
    decision: str  # READY | FIXABLE | EXCLUDE


def score_source(
    report: SourceQualityReport,
    schema_ok: bool,
    mean_divergence: float | None,
    weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
    divergence_ceiling: float = float(np.log(2)),
    ready_threshold: float = 0.8,
    fixable_threshold: float = 0.5,
) -> ReadinessResult:
    w_checks, w_schema, w_div = weights
    checks = report.check_results
    pass_rate = sum(c.passed for c in checks) / len(checks) if checks else 0.0
    schema_score = 1.0 if schema_ok else 0.0
    if mean_divergence is None or np.isnan(mean_divergence):
        div_score = 0.5  # unknown — neutral
    else:
        div_score = 1.0 - min(mean_divergence / divergence_ceiling, 1.0)

    score = w_checks * pass_rate + w_schema * schema_score + w_div * div_score

    if not schema_ok:
        decision = "EXCLUDE"
    elif score >= ready_threshold:
        decision = "READY"
    elif score >= fixable_threshold:
        decision = "FIXABLE"
    else:
        decision = "EXCLUDE"

    return ReadinessResult(
        source_id=report.source_id,
        score=float(score),
        check_pass_rate=float(pass_rate),
        schema_ok=schema_ok,
        mean_divergence=mean_divergence,
        decision=decision,
    )
