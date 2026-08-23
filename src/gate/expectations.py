"""Expectation suite: the gate-defined rules each data source is checked against.

Deliberately a small custom engine rather than a hard dependency on
Great Expectations — GX's API churn is a maintenance risk and we only need
five check types for v1. Cite GX/dbt as the inspiration in the paper.

Check types (v1):
  - column_exists
  - column_dtype
  - null_fraction_max
  - value_range          (numeric min/max)
  - min_row_count
"""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd

from .privacy import PrivacyAccountant, noised_count, noised_histogram, noised_null_fraction
from .reports import CheckResult, ColumnReport, SourceQualityReport


@dataclass
class Expectation:
    check_id: str
    kind: str
    column: str | None = None
    params: dict | None = None
    severity: str = "major"  # "blocking" | "major" | "minor" — used by the
    # blocking-gate calibration condition (dbt error-vs-warn semantics);
    # ignored by the control and reweight conditions.


def load_suite(path: str) -> list[Expectation]:
    with open(path) as f:
        raw = json.load(f)
    return [Expectation(**e) for e in raw["expectations"]]


def _cardinality_bucket(n_unique: int) -> str:
    if n_unique <= 10:
        return "low"
    if n_unique <= 1000:
        return "medium"
    return "high"


def run_suite(
    df: pd.DataFrame,
    suite: list[Expectation],
    source_id: str,
    epsilon_budget: float,
    seed: int,
    histogram_columns: list[str] | None = None,
    n_bins: int = 10,
    label_column: str | None = None,
    histogram_ranges: dict[str, tuple[float, float]] | None = None,
) -> SourceQualityReport:
    """Run all checks locally; emit a DP-noised metadata report.

    Budget split (documented for the paper):
      no label_column (Part 1):  25% row count / 50% nulls / 25% histograms
      with label_column (Part 2): 20% count / 40% nulls / 20% hist / 20% label
    The label share funds a DP-noised positive-label rate (Part 2's consensus
    label-rate detector); when label_column is None the split and outputs are
    IDENTICAL to Part 1.
    Pass/fail booleans are released WITHOUT noise in v1 — this leaks one bit
    per check. Acceptable for v1; flag in limitations, cite as future work
    (randomized response on booleans).
    """
    use_label = label_column is not None and label_column in df.columns
    b_count, b_nulls, b_hist, b_label = (
        (0.20, 0.40, 0.20, 0.20) if use_label else (0.25, 0.50, 0.25, 0.0))
    rng = np.random.default_rng(seed)
    acc = PrivacyAccountant(epsilon_budget)
    results: list[CheckResult] = []

    # --- checks (computed on true data; only booleans + noised stats leave)
    for e in suite:
        if e.kind == "column_exists":
            ok = e.column in df.columns
            results.append(CheckResult(e.check_id, ok, "" if ok else f"missing column {e.column}"))
        elif e.kind == "column_dtype":
            want = e.params["dtype"]
            ok = e.column in df.columns and str(df[e.column].dtype).startswith(want)
            results.append(CheckResult(e.check_id, ok))
        elif e.kind == "null_fraction_max":
            if e.column not in df.columns:
                results.append(CheckResult(e.check_id, False, "column missing"))
                continue
            frac = df[e.column].isna().mean()
            ok = frac <= e.params["max"]
            results.append(CheckResult(e.check_id, ok, f"threshold {e.params['max']}"))
        elif e.kind == "value_range":
            if e.column not in df.columns:
                results.append(CheckResult(e.check_id, False, "column missing"))
                continue
            col = pd.to_numeric(df[e.column], errors="coerce").dropna()
            ok = bool(len(col) == 0 or (col.between(e.params["min"], e.params["max"]).all()))
            results.append(CheckResult(e.check_id, ok))
        elif e.kind == "min_row_count":
            results.append(CheckResult(e.check_id, len(df) >= e.params["min"]))
        else:
            raise ValueError(f"unknown check kind: {e.kind}")

    # --- noised aggregates
    eps_count = b_count * epsilon_budget
    acc.spend("row_count", eps_count)
    row_count = noised_count(len(df), eps_count, rng)

    null_cols = sorted({e.column for e in suite if e.kind == "null_fraction_max" and e.column in df.columns})
    hist_cols = [c for c in (histogram_columns or []) if c in df.columns]

    eps_nulls_each = (b_nulls * epsilon_budget / len(null_cols)) if null_cols else 0.0
    eps_hist_each = (b_hist * epsilon_budget / len(hist_cols)) if hist_cols else 0.0

    schema: list[ColumnReport] = []
    for col in df.columns:
        nf = -1.0
        if col in null_cols:
            acc.spend(f"null_frac:{col}", eps_nulls_each)
            nf = noised_null_fraction(int(df[col].isna().sum()), len(df), eps_nulls_each, rng)
        hist, edges = None, None
        if col in hist_cols:
            acc.spend(f"hist:{col}", eps_hist_each)
            vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
            # Gate-defined shared bin edges (histogram_ranges) are what make
            # cross-source divergence meaningful: without a shared range each
            # source bins over its own min/max, so a scale/location shift
            # produces the same-shaped vector and is invisible to a bin-wise
            # comparison.
            rng_col = (histogram_ranges or {}).get(col)
            counts, bin_edges = np.histogram(vals, bins=n_bins, range=rng_col)
            hist = noised_histogram(counts, eps_hist_each, rng).tolist()
            edges = bin_edges.tolist()
        schema.append(
            ColumnReport(
                name=col,
                dtype=str(df[col].dtype),
                null_fraction_noised=nf,
                cardinality_bucket=_cardinality_bucket(df[col].nunique()),
                histogram_noised=hist,
                histogram_bin_edges=edges,
            )
        )

    positive_rate = -1.0
    if use_label:
        eps_label = b_label * epsilon_budget
        acc.spend("positive_rate", eps_label)
        lab = pd.to_numeric(df[label_column], errors="coerce")
        # Ratio of two DP-noised counts (same mechanism as null fractions):
        # positive count and total. Sensitivity 1 each; epsilon split evenly.
        positive_rate = noised_null_fraction(
            int((lab == 1).sum()), len(df), eps_label, rng)

    return SourceQualityReport(
        source_id=source_id,
        row_count_noised=row_count,
        epsilon_spent=acc.spent,
        schema=schema,
        check_results=results,
        positive_rate_noised=positive_rate,
    )
