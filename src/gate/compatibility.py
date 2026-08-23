"""Gate-side cross-source analysis: schema compatibility + divergence.

Jensen-Shannon divergence is used (not KL) because it is symmetric and
bounded: 0 <= JS <= ln(2) in nats. We CROSS-CHECK our implementation
against scipy.spatial.distance.jensenshannon (which returns the sqrt of
JS divergence in base-2 by default — mind the convention difference).
"""

from __future__ import annotations

import numpy as np

from .reports import SourceQualityReport


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in nats. Inputs are count vectors."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def schema_compatibility(
    reports: list[SourceQualityReport], required_columns: dict[str, str]
) -> dict[str, dict]:
    """For each source: which required columns are missing or wrongly typed."""
    out: dict[str, dict] = {}
    for r in reports:
        cols = {c.name: c.dtype for c in r.schema}
        missing = [c for c in required_columns if c not in cols]
        wrong_type = [
            c for c, want in required_columns.items()
            if c in cols and not cols[c].startswith(want)
        ]
        out[r.source_id] = {"missing": missing, "wrong_type": wrong_type,
                            "compatible": not missing and not wrong_type}
    return out


def pairwise_divergence(
    reports: list[SourceQualityReport], column: str
) -> dict[tuple[str, str], float]:
    """JS divergence between every pair of sources for one histogram column.

    Uses the DP-noised histograms — the divergence estimate itself is
    therefore noisy. Quantify the noise-vs-epsilon effect in experiments.
    """
    hists = {}
    for r in reports:
        for c in r.schema:
            if c.name == column and c.histogram_noised is not None:
                hists[r.source_id] = np.array(c.histogram_noised)
    out = {}
    ids = sorted(hists)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            out[(a, b)] = js_divergence(hists[a], hists[b])
    return out
