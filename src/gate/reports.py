"""Structured quality report exchanged between a data source and the gate.

Contains METADATA ONLY. No raw rows, no identifiers, ever.

Generalized from FedCheck: a "source" here is any contributing dataset /
data provider (a hospital site, a filing entity, a data vendor). The gate
never sees rows — only these noised metadata reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json


@dataclass
class ColumnReport:
    name: str
    dtype: str
    null_fraction_noised: float
    cardinality_bucket: str  # "low" | "medium" | "high" — coarse by design
    histogram_noised: list[float] | None = None
    histogram_bin_edges: list[float] | None = None


@dataclass
class CheckResult:
    check_id: str
    passed: bool
    detail: str = ""


@dataclass
class SourceQualityReport:
    source_id: str
    row_count_noised: float
    epsilon_spent: float
    schema: list[ColumnReport] = field(default_factory=list)
    check_results: list[CheckResult] = field(default_factory=list)
    positive_rate_noised: float = -1.0  # DP-noised P(label=1); -1.0 = not reported.
    # Used by the consensus label-rate detector (Part 2). Off unless run_suite
    # is given a label_column, so Part 1 reports are unchanged.

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(s: str) -> "SourceQualityReport":
        d = json.loads(s)
        d["schema"] = [ColumnReport(**c) for c in d["schema"]]
        d["check_results"] = [CheckResult(**c) for c in d["check_results"]]
        return SourceQualityReport(**d)
