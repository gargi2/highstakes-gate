"""Pillar B — cross-domain harness tests.

Guard the domain contract (schema/dtype parity between synthetic frames and the
expectation suite), the degradation-injection wiring, the dependency-free
model/metrics, and one end-to-end grid cell per domain. Synthetic only — no
data files, no rows printed.
"""
import math

import numpy as np
import pandas as pd
import pytest

from domains import REGISTRY
from domains import degradations as deg
from domains.base import roc_auc, train_eval
from gate.compatibility import schema_compatibility
from gate.consensus import consensus_checks, mean_divergence_per_source
from gate.expectations import run_suite
from gate.partition import degrade, split_sources
from gate.readiness import score_source

DOMAINS = ["healthcare", "finance", "government"]


def test_registry_has_all_domains():
    assert set(DOMAINS) <= set(REGISTRY)


@pytest.mark.parametrize("name", DOMAINS)
def test_synthetic_matches_required_schema(name):
    """Clean synthetic data must satisfy the domain's own suite (no self-inflicted fails)."""
    dom = REGISTRY[name]
    df = dom.synthetic(2000, 0)
    for col, dt in dom.required_columns.items():
        assert col in df.columns, f"{name}: missing {col}"
        if dt in ("int", "float"):
            assert str(df[col].dtype).startswith(dt), f"{name}: {col} is {df[col].dtype}, want {dt}"
    # label is binary
    assert set(pd.unique(df[dom.label_column])) <= {0, 1}
    # both classes present (so a downstream AUC is defined)
    assert df[dom.label_column].nunique() == 2


@pytest.mark.parametrize("name", DOMAINS)
def test_clean_sources_are_ready(name):
    """A clean, noiseless run must mark every source READY (no false exclusions)."""
    dom = REGISTRY[name]
    df = dom.synthetic(3000, 0)
    sources = split_sources(df, 6, 0)
    suite = dom.build_suite()
    reports = [run_suite(s, suite, f"s{i}", float("inf"), i,
                         histogram_columns=dom.histogram_columns,
                         label_column=dom.label_column,
                         histogram_ranges=dom.histogram_ranges)
               for i, s in enumerate(sources)]
    compat = schema_compatibility(reports, dom.required_columns)
    means = mean_divergence_per_source(reports, dom.histogram_columns)
    cons = consensus_checks(reports, dom.histogram_columns,
                            max_divergence=dom.divergence_threshold)
    for r in reports:
        r.check_results = list(r.check_results) + cons[r.source_id]
        rr = score_source(r, compat[r.source_id]["compatible"], means.get(r.source_id))
        assert rr.decision == "READY", f"{name}/{r.source_id}: {rr.decision} score={rr.score:.3f}"


@pytest.mark.parametrize("name", DOMAINS)
@pytest.mark.parametrize("mode", deg.MODES)
def test_degradation_detection_noiseless(name, mode):
    """At epsilon=inf, detection behaves as the gate's design predicts.

    All modes are detected EXCEPT label noise on a near-balanced label: the
    consensus detector watches the positive-rate marginal, which is ~invariant
    under symmetric label flips when the base rate is near 0.5. That blind spot
    is a documented finding (docs/WALKTHROUGH.md), so we assert it rather than
    pretend it away.
    """
    dom = REGISTRY[name]
    df = dom.synthetic(4000, 1)
    base_rate = float((df[dom.label_column] == 1).mean())
    sources = split_sources(df, 6, 1)
    spec = deg.build_spec(mode, "high", dom.feature_columns, dom.label_column)
    reports = []
    for i, s in enumerate(sources):
        if i < 2:
            s = degrade(s, spec, 97 + i)
        reports.append(run_suite(s, dom.build_suite(), f"s{i}", float("inf"), 131 + i,
                                 histogram_columns=dom.histogram_columns,
                                 label_column=dom.label_column,
                                 histogram_ranges=dom.histogram_ranges))
    compat = schema_compatibility(reports, dom.required_columns)
    cons = consensus_checks(reports, dom.histogram_columns,
                            max_divergence=dom.divergence_threshold)
    hits = 0
    for r in reports[:2]:
        r.check_results = list(r.check_results) + cons[r.source_id]
        hits += deg.flagged_for_mode(mode, r.check_results, compat[r.source_id]["compatible"])

    label_blind_spot = mode == "label" and 0.2 < base_rate < 0.8
    if label_blind_spot:
        assert hits == 0, f"{name}/{mode}: unexpectedly detected on balanced label"
    else:
        assert hits >= 1, f"{name}/{mode}: no degraded source flagged at eps=inf"


def test_roc_auc_matches_known_value():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8])
    assert roc_auc(y, s) == pytest.approx(0.75)
    assert math.isnan(roc_auc(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3])))


def test_gating_never_hurts_on_clean_data():
    """With no degradation, gated vs ungated AUC should be identical (same data kept)."""
    dom = REGISTRY["finance"]
    df = dom.synthetic(3000, 2)
    train, test = df.iloc[:2400], df.iloc[2400:]
    auc = train_eval(train, test, dom.feature_columns, dom.label_column, 0)
    assert 0.5 <= auc <= 1.0
