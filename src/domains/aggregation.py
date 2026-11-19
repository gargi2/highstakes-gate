"""Alternative readiness-decision rules — the aggregation-rule ablation.

Directly tests the paper's spine (the detection->action gap, Finding A). The
frozen gate (`gate.readiness.score_source`) decides via a WEIGHTED MEAN of three
subscores (pass_rate, schema, divergence), READY >= 0.8. The spine is that this
composite MASKS a severe single-dimension defect: one failed check of ~15 pulls
pass_rate only 1.00 -> 0.93, which the 0.4 weight averages away, so a *flagged*
source is still marked READY and never withheld (detection != action).

This module holds ALTERNATIVE decision rules that consume the SAME subscores /
reports, so we can ask whether a different aggregation closes the gap WITHOUT
exploding the clean false-positive rate. Nothing here modifies `gate/` —
`score_source` stays the reference rule (reused, unmodified, from FedCheck).

No leakage: the rules are fixed a priori and reuse the gate's own thresholds and
ceiling; none is tuned on the detection/withholding outcome.

Rules
-----
weighted_mean : the reference gate (score_source), verbatim.
max_violation : the WORST of the three subscores drives the decision. Catches a
                tanked dimension (distributional div_score, or heavy completeness
                that drags pass_rate below 0.8) but, by design, still passes a
                single failed check hidden inside pass_rate=0.93 — a *partial*
                fix, and the honest way to show the masking has two layers.
any_critical  : zero-tolerance at the CHECK level — any single failed check (or an
                incompatible schema) -> EXCLUDE. Closes the gap by construction;
                the experiment measures what it costs in clean false positives.
"""
from __future__ import annotations

import numpy as np

from gate.reports import SourceQualityReport
from gate.readiness import score_source

RULES = ["weighted_mean", "max_violation", "any_critical"]

_DIV_CEILING = float(np.log(2))


def _subscores(
    report: SourceQualityReport,
    schema_ok: bool,
    mean_divergence: float | None,
    divergence_ceiling: float,
) -> tuple[float, float, float]:
    """The three subscores, computed EXACTLY as gate.readiness.score_source does."""
    checks = report.check_results
    pass_rate = sum(c.passed for c in checks) / len(checks) if checks else 0.0
    schema_score = 1.0 if schema_ok else 0.0
    if mean_divergence is None or np.isnan(mean_divergence):
        div_score = 0.5  # unknown — neutral (matches score_source)
    else:
        div_score = 1.0 - min(mean_divergence / divergence_ceiling, 1.0)
    return pass_rate, schema_score, div_score


def decide(
    report: SourceQualityReport,
    schema_ok: bool,
    mean_divergence: float | None,
    rule: str,
    *,
    divergence_ceiling: float = _DIV_CEILING,
    ready_threshold: float = 0.8,
    fixable_threshold: float = 0.5,
) -> str:
    """Return the READY | FIXABLE | EXCLUDE decision under `rule`.

    `weighted_mean` delegates to the frozen `score_source` so the reference rule
    is guaranteed identical to the deployed gate.
    """
    if rule == "weighted_mean":
        return score_source(
            report, schema_ok, mean_divergence,
            divergence_ceiling=divergence_ceiling,
            ready_threshold=ready_threshold,
            fixable_threshold=fixable_threshold,
        ).decision

    if not schema_ok:  # schema incompatibility is fatal under every rule
        return "EXCLUDE"

    if rule == "max_violation":
        pass_rate, schema_score, div_score = _subscores(
            report, schema_ok, mean_divergence, divergence_ceiling)
        worst = min(pass_rate, schema_score, div_score)
        if worst >= ready_threshold:
            return "READY"
        if worst >= fixable_threshold:
            return "FIXABLE"
        return "EXCLUDE"

    if rule == "any_critical":
        if any(not c.passed for c in report.check_results):
            return "EXCLUDE"
        return "READY"

    raise ValueError(f"unknown rule: {rule!r}; known: {RULES}")
