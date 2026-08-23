"""Tests for the Part 2 consensus detectors + run_suite label-rate release."""

import numpy as np

from gate.consensus import (
    DIVERGENCE_CHECK_ID,
    LABEL_RATE_CHECK_ID,
    consensus_checks,
    distribution_divergence_checks,
    label_rate_divergence_checks,
)
from gate.expectations import load_suite, run_suite
from gate.partition import degrade, make_synthetic_clinical, split_sources

SUITE = load_suite("suites/clinical_v1.json")
HIST_COLS = ["age", "lab_creatinine", "lab_hemoglobin"]
# Gate-defined monitoring ranges -> shared bins across sources.
HIST_RANGES = {"age": (0.0, 120.0), "lab_creatinine": (0.0, 20.0),
               "lab_hemoglobin": (0.0, 25.0)}


def _reports(sources, eps=float("inf"), label="mortality"):
    return [run_suite(s, SUITE, source_id=f"source_{i}", epsilon_budget=eps, seed=i,
                      histogram_columns=HIST_COLS, label_column=label,
                      histogram_ranges=HIST_RANGES)
            for i, s in enumerate(sources)]


def test_run_suite_label_column_backward_compatible():
    """Without label_column, positive_rate stays -1 and Part 1 output is unchanged."""
    df = make_synthetic_clinical(2000, seed=0)
    r_no = run_suite(df, SUITE, "s", epsilon_budget=1.0, seed=1, histogram_columns=HIST_COLS)
    assert r_no.positive_rate_noised == -1.0
    r_yes = run_suite(df, SUITE, "s", epsilon_budget=1.0, seed=1,
                      histogram_columns=HIST_COLS, label_column="mortality")
    assert 0.0 <= r_yes.positive_rate_noised <= 1.0


def test_label_rate_detector_flags_label_noise_source():
    """A heavy-label-noise source sits far from consensus and is flagged; clean pass."""
    df = make_synthetic_clinical(9000, seed=3)
    sources = split_sources(df, 6, seed=4)
    sources[0] = degrade(sources[0], {"label_noise": {"column": "mortality", "rate": 0.45}}, seed=5)
    checks = label_rate_divergence_checks(_reports(sources), max_abs_deviation=0.15)
    assert checks["source_0"].passed is False   # degraded
    assert all(checks[f"source_{i}"].passed for i in range(1, 6))  # clean


def test_divergence_detector_flags_distribution_shift_source():
    df = make_synthetic_clinical(9000, seed=6)
    sources = split_sources(df, 6, seed=7)
    sources[1] = degrade(sources[1], {"distribution_shift": {"column": "lab_creatinine", "factor": 4.0}}, seed=8)
    checks = distribution_divergence_checks(_reports(sources), HIST_COLS, max_divergence=0.08)
    assert checks["source_1"].passed is False
    # at least the shifted source is flagged; clean sources should mostly pass
    flagged = [sid for sid, c in checks.items() if not c.passed]
    assert flagged == ["source_1"]


def test_consensus_checks_combines_both():
    df = make_synthetic_clinical(9000, seed=9)
    sources = split_sources(df, 6, seed=10)
    sources[2] = degrade(sources[2], {"label_noise": {"column": "mortality", "rate": 0.45},
                                      "distribution_shift": {"column": "lab_creatinine", "factor": 4.0}}, seed=11)
    extra = consensus_checks(_reports(sources), HIST_COLS)
    ids = {c.check_id for c in extra["source_2"]}
    assert ids == {LABEL_RATE_CHECK_ID, DIVERGENCE_CHECK_ID}
    assert all(not c.passed for c in extra["source_2"])  # both fire on the degraded source


def test_label_rate_survives_dp_noise_at_moderate_epsilon():
    """The label-noise signal is huge, so detection holds even with DP noise."""
    df = make_synthetic_clinical(9000, seed=12)
    sources = split_sources(df, 6, seed=13)
    sources[0] = degrade(sources[0], {"label_noise": {"column": "mortality", "rate": 0.45}}, seed=14)
    checks = label_rate_divergence_checks(_reports(sources, eps=1.0), max_abs_deviation=0.15)
    assert checks["source_0"].passed is False
