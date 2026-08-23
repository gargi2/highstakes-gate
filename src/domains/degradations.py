"""Taxonomy-derived degradations, shared across all domains.

The four PRE-TRAINING-detectable data-readiness modes from docs/TAXONOMY.md that
map onto a controllable injection (cross-source and provenance are not row-level
injectable, so they are out of scope for this harness — see WALKTHROUGH):

    structural      -> drop a column / corrupt its dtype   (schema checks)
    completeness    -> null out a fraction of a column     (null_fraction_max)
    distributional  -> scale-shift a feature distribution  (consensus divergence)
    label           -> flip a fraction of labels           (consensus label-rate)

`build_spec` returns a spec for `gate.partition.degrade` (REUSED, not forked) so
the exact same injection code runs in every domain. `flagged_for_mode` says
whether the gate's own checks caught a given source, which is how we score
detection precision/recall.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gate.consensus import DIVERGENCE_CHECK_ID, LABEL_RATE_CHECK_ID
from gate.reports import CheckResult

MODES = ["structural", "completeness", "distributional", "label"]
SEVERITIES = ["low", "med", "high"]


def degrade_labels_asymmetric(df: pd.DataFrame, column: str, rate_pos: float,
                              rate_neg: float, seed: int) -> pd.DataFrame:
    """Class-conditional (ASYMMETRIC) label flip — an item-3 extension.

    Flips positives->0 at rate_pos and negatives->1 at rate_neg. Unlike the gate's
    symmetric `label_noise` (which cannot move a balanced marginal), an asymmetric
    flip shifts P(y=1) by `(1-p)*rate_neg - p*rate_pos`, so the marginal detector
    regains a signal and we can trace a detectability-threshold curve. Applied
    locally (the frozen gate.partition is untouched). Returns a copy.
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    y = pd.to_numeric(out[column], errors="coerce").to_numpy(dtype=float)
    pos, neg = (y == 1), (y == 0)
    flip_p = pos & (rng.random(len(out)) < rate_pos)
    flip_n = neg & (rng.random(len(out)) < rate_neg)
    y[flip_p] = 0.0
    y[flip_n] = 1.0
    out[column] = y.astype(int)
    return out


def build_spec(mode: str, severity: str, features: list[str], label: str) -> dict:
    """Return a `gate.partition.degrade` spec for (mode, severity)."""
    f0 = features[0]
    f1 = features[1] if len(features) > 1 else features[0]
    if mode == "structural":
        return {"low": {"type_corrupt": [f0]},
                "med": {"drop_column": [f0]},
                "high": {"drop_column": [f0, f1]}}[severity]
    if mode == "completeness":
        return {"low": {"null_rate": {f0: 0.10}},
                "med": {"null_rate": {f0: 0.35, f1: 0.35}},
                "high": {"null_rate": {c: 0.60 for c in features}}}[severity]
    if mode == "distributional":
        # In-range multiplicative shifts. Larger factors push mass OUT of the
        # shared histogram range (empty noised histogram -> NaN divergence ->
        # the detector goes blind), so severity tops out where the shift is
        # still measurable; the out-of-support blind spot is a documented
        # limitation (docs/WALKTHROUGH.md), not more "severity".
        factor = {"low": 1.7, "med": 2.2, "high": 2.8}[severity]
        return {"distribution_shift": {"column": f0, "factor": factor}}
    if mode == "label":
        rate = {"low": 0.15, "med": 0.30, "high": 0.45}[severity]
        return {"label_noise": {"column": label, "rate": rate}}
    raise ValueError(f"unknown mode: {mode}")


def flagged_for_mode(mode: str, checks: list[CheckResult], schema_ok: bool) -> bool:
    """Did the gate's mode-specific check(s) flag this source?"""
    failed = {c.check_id for c in checks if not c.passed}
    if mode == "structural":
        return (not schema_ok) or any(
            cid.startswith("exists_") or cid.startswith("dtype_") for cid in failed)
    if mode == "completeness":
        return any(cid.startswith("null_") for cid in failed)
    if mode == "distributional":
        return DIVERGENCE_CHECK_ID in failed
    if mode == "label":
        return LABEL_RATE_CHECK_ID in failed
    return False


def flagged_any(checks: list[CheckResult], schema_ok: bool) -> bool:
    """Flagged by ANY mode detector — used for the clean-source false-flag rate."""
    return any(flagged_for_mode(m, checks, schema_ok) for m in MODES)
