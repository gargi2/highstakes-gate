"""Pillar C item 1 — downstream-arm redesign tests.

Guard the subgroup/calibration metrics and the four-condition wiring in
experiments/run_downstream.py. Synthetic only, tiny grids — no data files, no rows.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from domains import REGISTRY                                    # noqa: E402
from domains.metrics import (                                   # noqa: E402
    evaluate_downstream, expected_calibration_error, _quantile_bins, _fnr_gap,
)
import run_downstream as rd                                     # noqa: E402

DOMAINS = ["healthcare", "finance", "government"]


def _in_unit(x):
    return x == "" or (isinstance(x, float) and math.isnan(x)) or (0.0 <= float(x) <= 1.0)


def test_ece_perfect_and_worst():
    y = np.array([0, 0, 1, 1], dtype=float)
    assert expected_calibration_error(y, np.array([0.0, 0.0, 1.0, 1.0])) == pytest.approx(0.0, abs=1e-9)
    # confident but always wrong -> ECE 1.0
    assert expected_calibration_error(y, np.array([1.0, 1.0, 0.0, 0.0])) == pytest.approx(1.0, abs=1e-9)


def test_quantile_bins_and_fnr_gap_shapes():
    v = np.arange(100.0)
    b = _quantile_bins(v, 4)
    assert set(np.unique(b)).issubset({0, 1, 2, 3})
    # constant column collapses to a single bin (no crash)
    assert set(np.unique(_quantile_bins(np.ones(50), 4))) == {0}
    # fnr gap undefined with <2 usable groups -> NaN
    y = np.ones(10); p = np.zeros(10)
    assert math.isnan(_fnr_gap(y, p, np.zeros(10, dtype=int)))


def test_evaluate_downstream_ranges():
    dom = REGISTRY["healthcare"]
    df = dom.synthetic(2000, seed=0)
    idx = np.random.default_rng(0).permutation(len(df))
    te, tr = df.iloc[idx[:400]], df.iloc[idx[400:]]
    m = evaluate_downstream(tr, te, dom.feature_columns, dom.label_column,
                            dom.subgroup_column or dom.feature_columns[0], seed=0)
    for v in (m.auc, m.acc, m.worst_group_acc, m.ece):
        assert math.isnan(v) or 0.0 <= v <= 1.0
    assert m.n_groups_used >= 1


@pytest.mark.parametrize("domain", DOMAINS)
def test_run_cell_conditions_and_wiring(domain):
    dom = REGISTRY[domain]
    cfg = dict(n_rows=1500, max_rows=3000, n_sources=6, n_degraded=2,
               epsilon=math.inf, data_root={})
    # a degraded mode yields exactly the four conditions; metrics are in range
    rows = rd.run_cell(dom, "structural", "high", math.inf, seed=0, cfg=cfg)
    conds = {r["condition"] for r in rows}
    assert conds == {"clean_only", "all_ungated", "corrupt_only", "gate_realistic"}
    for r in rows:
        for k in ("auc", "acc", "worst_group_acc", "ece"):
            assert _in_unit(r[k])
    by = {r["condition"]: r for r in rows}
    # dilution floor trains on all 6; oracle ceiling on the 4 clean
    assert by["all_ungated"]["n_train_sources"] == 6
    assert by["clean_only"]["n_train_sources"] == 4
    # structural is schema-incompatible -> the gate withholds it, so gate_realistic
    # keeps only the clean sources (matches the detection-vs-action story).
    assert by["gate_realistic"]["n_train_sources"] == 4

    # mode=none yields a single clean baseline row
    none_rows = rd.run_cell(dom, "none", "none", math.inf, seed=0, cfg=cfg)
    assert [r["condition"] for r in none_rows] == ["clean_baseline"]


def test_label_corruption_hits_subgroup_not_just_auc():
    """The whole point of item 1: corruption that aggregate AUC hides shows up in
    worst-group accuracy / calibration. On a rare-label domain, corrupt_only should
    be no better than clean_only on worst-group accuracy (usually worse)."""
    dom = REGISTRY["healthcare"]
    cfg = dict(n_rows=4000, max_rows=8000, n_sources=6, n_degraded=2,
               epsilon=math.inf, data_root={})
    rows = rd.run_cell(dom, "label", "high", math.inf, seed=0, cfg=cfg)
    by = {r["condition"]: r for r in rows}
    wga_clean = float(by["clean_only"]["worst_group_acc"])
    wga_corrupt = float(by["corrupt_only"]["worst_group_acc"])
    assert wga_corrupt <= wga_clean + 1e-6
