"""Pillar C item 3 — confident-learning label detector tests (Blocker 2 fix).

Synthetic only. Guards: (1) self-disagreement is elevated by SYMMETRIC label noise
even on a balanced target (where the marginal detector is blind); (2) the consensus
check flags a high-disagreement source; (3) the asymmetric flip moves the marginal.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from domains import REGISTRY                                       # noqa: E402
from domains.degradations import degrade_labels_asymmetric        # noqa: E402
from domains.label_quality import (                               # noqa: E402
    self_disagreement_rate, label_quality_checks, LABEL_QUALITY_CHECK_ID,
)


def _flip_symmetric(df, label, rate, seed):
    rng = np.random.default_rng(seed)
    out = df.copy()
    flip = rng.random(len(out)) < rate
    out.loc[flip, label] = 1 - out.loc[flip, label]
    return out


@pytest.mark.parametrize("domain", ["finance", "healthcare", "government"])
def test_self_disagreement_elevated_by_symmetric_noise(domain):
    dom = REGISTRY[domain]
    df = dom.synthetic(4000, seed=0)
    clean = self_disagreement_rate(df, dom.feature_columns, dom.label_column, seed=1)
    noisy = self_disagreement_rate(_flip_symmetric(df, dom.label_column, 0.45, 2),
                                   dom.feature_columns, dom.label_column, seed=1)
    # noisy must be clearly higher — including on balanced finance (the blind spot)
    assert noisy > clean + 0.05


def test_self_disagreement_nan_on_tiny_or_singleclass():
    dom = REGISTRY["finance"]
    df = dom.synthetic(4000, seed=0)
    tiny = df.head(10)
    assert np.isnan(self_disagreement_rate(tiny, dom.feature_columns, dom.label_column, seed=0))
    single = df.copy(); single[dom.label_column] = 1
    assert np.isnan(self_disagreement_rate(single, dom.feature_columns, dom.label_column, seed=0))


def test_consensus_flags_high_disagreement_source():
    # 4 clean sources ~0.30, one noisy at 0.45 -> only the noisy one is flagged
    dis = {"s0": 0.30, "s1": 0.31, "s2": 0.29, "s3": 0.30, "s4": 0.45}
    checks = label_quality_checks(dis, max_abs_deviation=0.05)
    flagged = {s for s, cs in checks.items()
               if any(c.check_id == LABEL_QUALITY_CHECK_ID and not c.passed for c in cs)}
    assert flagged == {"s4"}


def test_nan_rate_passes():
    checks = label_quality_checks({"s0": 0.30, "s1": 0.31, "s2": float("nan")})
    assert all(c.passed for c in checks["s2"])  # no evidence -> pass


def test_asymmetric_flip_moves_the_marginal():
    dom = REGISTRY["finance"]  # balanced ~0.42
    df = dom.synthetic(4000, seed=0)
    base = (df[dom.label_column] == 1).mean()
    asym = degrade_labels_asymmetric(df, dom.label_column, rate_pos=0.45, rate_neg=0.0, seed=1)
    after = (asym[dom.label_column] == 1).mean()
    # positives->negatives only: marginal must drop meaningfully (unlike symmetric)
    assert base - after > 0.10
