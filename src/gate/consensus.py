"""Gate-side consensus detectors (Part 2).

Part 1's checks are LOCAL: each source tests its own data against fixed rules
(column exists, null fraction, range, ...). Those miss the two degradations
that most damage a downstream model: label noise and distribution shift.
Both are only visible by COMPARING a source to its peers, which the gate can
do from the DP-noised reports without seeing any rows.

Two detectors, each emitting one extra CheckResult per source so they slot
into the existing gate/scoring machinery (with a severity, so the blocking
gate can act on them):

  label_rate_divergence   flag a source whose DP-noised positive-label rate is
                          far from the cross-source consensus (median). Heavy
                          label noise moves the positive rate massively (e.g.
                          ~0.45 vs a ~0.05 base rate), so this is easy to see
                          even through DP noise.
  distribution_divergence flag a source whose mean pairwise Jensen-Shannon
                          divergence to other sources (over the noised feature
                          histograms) exceeds a threshold. Catches a shifted
                          feature distribution.

Consensus uses the MEDIAN across sources, which stays meaningful as long as
fewer than half the sources are degraded; past ~50% the "consensus" is itself
corrupted — a documented limitation and a natural place to report a breakdown
point.
"""

from __future__ import annotations

import numpy as np

from .compatibility import pairwise_divergence
from .reports import CheckResult, SourceQualityReport

LABEL_RATE_CHECK_ID = "consensus_label_rate"
DIVERGENCE_CHECK_ID = "consensus_divergence"


def label_rate_divergence_checks(
    reports: list[SourceQualityReport], max_abs_deviation: float = 0.15
) -> dict[str, CheckResult]:
    """Flag sources whose positive-label rate is far from the cross-source median.

    Requires positive_rate_noised on the reports (run_suite given a
    label_column). Sources without it are passed through (not flagged).
    """
    rates = {r.source_id: r.positive_rate_noised for r in reports
             if r.positive_rate_noised >= 0.0}
    out: dict[str, CheckResult] = {}
    if not rates:
        return out
    consensus = float(np.median(list(rates.values())))
    for sid, rate in rates.items():
        dev = abs(rate - consensus)
        ok = dev <= max_abs_deviation
        out[sid] = CheckResult(
            LABEL_RATE_CHECK_ID, ok,
            f"|{rate:.3f}-{consensus:.3f}|={dev:.3f} vs {max_abs_deviation}")
    return out


def mean_divergence_per_source(
    reports: list[SourceQualityReport], columns: list[str]
) -> dict[str, float]:
    """Each source's mean pairwise JS divergence to all other sources, over columns."""
    per_source: dict[str, list[float]] = {r.source_id: [] for r in reports}
    for col in columns:
        for (a, b), d in pairwise_divergence(reports, col).items():
            if not np.isnan(d):
                per_source[a].append(d)
                per_source[b].append(d)
    return {sid: (float(np.mean(v)) if v else float("nan"))
            for sid, v in per_source.items()}


def distribution_divergence_checks(
    reports: list[SourceQualityReport], columns: list[str], max_divergence: float = 0.08
) -> dict[str, CheckResult]:
    """Flag sources whose mean cross-source JS divergence exceeds a threshold."""
    means = mean_divergence_per_source(reports, columns)
    out: dict[str, CheckResult] = {}
    for sid, md in means.items():
        if np.isnan(md):
            continue  # no divergence signal (e.g. no histograms) -> don't flag
        ok = md <= max_divergence
        out[sid] = CheckResult(DIVERGENCE_CHECK_ID, ok,
                               f"mean_div={md:.3f} vs {max_divergence}")
    return out


def consensus_checks(
    reports: list[SourceQualityReport],
    divergence_columns: list[str],
    label_max_abs_deviation: float = 0.15,
    max_divergence: float = 0.08,
) -> dict[str, list[CheckResult]]:
    """Both consensus detectors, keyed by source_id -> list of extra CheckResults.

    Returned checks are intended to be BLOCKING severity (they target the
    degradations that most damage downstream models), appended to each
    source's local check_results before scoring.
    """
    lab = label_rate_divergence_checks(reports, label_max_abs_deviation)
    div = distribution_divergence_checks(reports, divergence_columns, max_divergence)
    out: dict[str, list[CheckResult]] = {r.source_id: [] for r in reports}
    for sid in out:
        if sid in lab:
            out[sid].append(lab[sid])
        if sid in div:
            out[sid].append(div[sid])
    return out
