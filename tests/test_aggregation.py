"""Aggregation-rule ablation tests (NEXT_EXPERIMENTS item 4).

Guards, on hand-built reports (no data rows):
  1. `weighted_mean` is byte-for-byte the frozen gate (score_source).
  2. `any_critical` withholds on a single failed check; passes when all pass.
  3. `max_violation` catches a tanked dimension (high divergence) but, by design,
     still passes a single failed check hidden inside pass_rate=0.93 — the
     two-layer-masking property, made explicit.
  4. On exactly that masked single-check case, `any_critical` closes the
     detection->action gap the weighted mean leaves open.
  5. Schema incompatibility -> EXCLUDE under every rule.
  6. Unknown rule raises.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gate.reports import SourceQualityReport, CheckResult   # noqa: E402
from gate.readiness import score_source                     # noqa: E402
from domains.aggregation import decide, RULES               # noqa: E402

_CEILING = float(np.log(2))


def _report(n_pass: int, n_fail: int, source_id: str = "s0") -> SourceQualityReport:
    checks = [CheckResult(f"ok_{i}", True) for i in range(n_pass)]
    checks += [CheckResult(f"null_{i}", False, "injected") for i in range(n_fail)]
    return SourceQualityReport(source_id=source_id, row_count_noised=1000.0,
                               epsilon_spent=1.0, check_results=checks)


def test_weighted_mean_matches_frozen_gate():
    cases = [
        (_report(15, 0), True, None),        # all pass, divergence unknown
        (_report(14, 1), True, 0.0),         # one failed check, low divergence
        (_report(15, 0), True, _CEILING),    # all pass, max divergence
        (_report(15, 0), False, 0.0),        # schema incompatible
    ]
    for report, schema_ok, div in cases:
        assert decide(report, schema_ok, div, "weighted_mean") == \
            score_source(report, schema_ok, div).decision


def test_any_critical_zero_tolerance():
    assert decide(_report(15, 0), True, 0.0, "any_critical") == "READY"
    assert decide(_report(14, 1), True, 0.0, "any_critical") == "EXCLUDE"


def test_max_violation_catches_tanked_dimension_but_not_hidden_single_check():
    # high divergence tanks div_score -> max_violation is stricter than weighted mean
    hi_div = _report(15, 0)
    assert decide(hi_div, True, _CEILING, "max_violation") == "EXCLUDE"
    assert decide(hi_div, True, _CEILING, "weighted_mean") != "EXCLUDE"
    # but a single failed check (pass_rate 14/15=0.93) hides inside pass_rate:
    # both weighted_mean and max_violation still say READY (two-layer masking)
    masked = _report(14, 1)
    assert decide(masked, True, 0.0, "weighted_mean") == "READY"
    assert decide(masked, True, 0.0, "max_violation") == "READY"


def test_any_critical_closes_the_gap_weighted_mean_leaves_open():
    # the exact masked case: flagged (a check failed) yet weighted_mean keeps it
    # READY (not withheld) -> the spine's gap. any_critical withholds it.
    masked = _report(14, 1)
    assert decide(masked, True, 0.0, "weighted_mean") == "READY"        # not acted on
    assert decide(masked, True, 0.0, "any_critical") != "READY"         # acted on


def test_schema_incompatible_excludes_under_every_rule():
    for rule in RULES:
        assert decide(_report(15, 0), False, 0.0, rule) == "EXCLUDE"


def test_unknown_rule_raises():
    with pytest.raises(ValueError):
        decide(_report(15, 0), True, 0.0, "median_vote")
