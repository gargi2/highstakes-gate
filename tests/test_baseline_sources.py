"""Tests for the external-baseline source regenerator (NEXT_EXPERIMENTS item 6).

Synthetic only. Guards that make_cell_sources reproduces the gate's cell structure
so a Colab baseline run (GE/TFDV/Deequ) sees the same inputs the gate scored:
counts, which sources are degraded, and that each mode actually injects its defect.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "baselines"))

from gen_baseline_sources import make_cell_sources          # noqa: E402
from domains import REGISTRY                                  # noqa: E402


def test_counts_and_degraded_indices():
    cell = make_cell_sources("finance", "completeness", "high", seed=0,
                             n_sources=6, n_degraded=2, n_rows=2000)
    assert len(cell["sources"]) == 6
    assert sum(is_d for _, is_d in cell["sources"]) == 2
    # degraded are the first n_degraded sources (matches run_crossdomain)
    flags = [is_d for _, is_d in cell["sources"]]
    assert flags[:2] == [True, True] and not any(flags[2:])


def test_mode_none_has_no_degraded():
    cell = make_cell_sources("healthcare", "none", "none", seed=1, n_rows=1000)
    assert sum(is_d for _, is_d in cell["sources"]) == 0


def test_structural_drops_columns_in_degraded_only():
    dom = REGISTRY["government"]
    cell = make_cell_sources("government", "structural", "high", seed=2, n_rows=2000)
    f0, f1 = dom.feature_columns[0], dom.feature_columns[1]
    for src, is_deg in cell["sources"]:
        if is_deg:
            assert f0 not in src.columns and f1 not in src.columns   # high drops two cols
        else:
            assert f0 in src.columns


def test_completeness_raises_nulls_in_degraded_only():
    dom = REGISTRY["healthcare"]
    cell = make_cell_sources("healthcare", "completeness", "high", seed=3, n_rows=2000)
    f0 = dom.feature_columns[0]
    for src, is_deg in cell["sources"]:
        null_frac = pd.to_numeric(src[f0], errors="coerce").isna().mean()
        if is_deg:
            assert null_frac > 0.3      # high nulls out >=0.60 of features
        else:
            assert null_frac < 0.1


def test_reproducible_given_seed():
    a = make_cell_sources("finance", "distributional", "med", seed=7, n_rows=1500)
    b = make_cell_sources("finance", "distributional", "med", seed=7, n_rows=1500)
    sa, sb = a["sources"][0][0], b["sources"][0][0]
    pd.testing.assert_frame_equal(sa, sb)
