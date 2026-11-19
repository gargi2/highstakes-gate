#!/usr/bin/env python
"""Headless external-baseline runner (GE / TFDV / Deequ) — NEXT_EXPERIMENTS item 6.

Same adapters and grid as `baselines_colab.ipynb`, but a plain script for a real
Python 3.11 environment (local Docker), where all three actually install. Writes
`experiments/results/baselines_detection.csv`. Any baseline that isn't importable
is SKIPPED (its rows get empty det_recall); the analyzer reports it as ABSENT.

DATA SAFETY: synthetic only (make_cell_sources default) — nothing sensitive is read.

Run (inside a py3.11 container; see the docker command in docs/NEXT_EXPERIMENTS.md §6):
    SPARK_VERSION=3.4 python experiments/baselines/run_baselines.py
"""
from __future__ import annotations

import argparse
import sys
import types
import typing
from pathlib import Path

import numpy as np  # noqa: F401  (repo domains import numpy/pandas transitively)
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "baselines"))

from gen_baseline_sources import make_cell_sources   # noqa: E402
from domains import degradations as deg               # noqa: E402


def numeric_bounds(ref, col, lo_q=0.001, hi_q=0.999):
    s = pd.to_numeric(ref[col], errors="coerce").dropna()
    return (float(s.quantile(lo_q)), float(s.quantile(hi_q)), float(s.mean())) if len(s) else (None, None, None)


# --- Great Expectations (legacy 0.18 PandasDataset) ---
try:
    import great_expectations as ge
    HAVE_GE = True
except Exception as e:  # noqa: BLE001
    HAVE_GE = False
    print("GE unavailable ->", type(e).__name__, str(e)[:100])


def ge_flag(source_df, reference_df, domain):
    if not HAVE_GE:
        return None
    d = ge.from_pandas(source_df)
    ok = True
    for col in domain.required_columns:
        ok &= d.expect_column_to_exist(col).success
    for col in domain.feature_columns:
        if col in source_df.columns:
            ok &= d.expect_column_values_to_not_be_null(col, mostly=1.0 - domain.null_threshold).success
            lo, hi, _ = numeric_bounds(reference_df, col)
            if lo is not None and hi > lo:
                ok &= d.expect_column_values_to_be_between(col, min_value=lo, max_value=hi, mostly=0.99).success
    lab = domain.label_column
    if lab in source_df.columns and lab in reference_df.columns:
        _, _, m = numeric_bounds(reference_df, lab)
        if m is not None:
            ok &= d.expect_column_mean_to_be_between(lab, min_value=0.7 * m, max_value=1.3 * m + 1e-9).success
    return not ok


# --- TFDV (schema + JSD drift comparator) ---
try:
    import tensorflow_data_validation as tfdv
    HAVE_TFDV = True
except Exception as e:  # noqa: BLE001
    HAVE_TFDV = False
    print("TFDV unavailable ->", type(e).__name__, str(e)[:100])


def tfdv_schema(reference_df, domain, drift_threshold=0.1):
    if not HAVE_TFDV:
        return None, None
    ref_stats = tfdv.generate_statistics_from_dataframe(reference_df)
    schema = tfdv.infer_schema(ref_stats)
    for col in domain.feature_columns:
        try:
            f = tfdv.get_feature(schema, col)
            f.drift_comparator.jensen_shannon_divergence.threshold = drift_threshold
        except Exception:  # noqa: BLE001
            pass
    return schema, ref_stats


def tfdv_flag(source_df, reference_df, domain, schema, ref_stats):
    if not HAVE_TFDV or schema is None:
        return None
    stats = tfdv.generate_statistics_from_dataframe(source_df)
    anomalies = tfdv.validate_statistics(stats, schema, previous_statistics=ref_stats)
    return len(anomalies.anomaly_info) > 0


# --- Deequ (pydeequ + Spark); shim typing.io for py>=3.13 ---
for _sub, _names in (("typing.io", ("IO", "TextIO", "BinaryIO")), ("typing.re", ("Pattern", "Match"))):
    if _sub not in sys.modules:
        _m = types.ModuleType(_sub)
        for _n in _names:
            try:
                setattr(_m, _n, getattr(typing, _n))
            except AttributeError:
                pass
        sys.modules[_sub] = _m
try:
    import pydeequ
    from pyspark.sql import SparkSession
    from pydeequ.checks import Check, CheckLevel
    from pydeequ.verification import VerificationSuite, VerificationResult
    spark = (SparkSession.builder
             .config("spark.jars.packages", pydeequ.deequ_maven_coord)
             .config("spark.jars.excludes", pydeequ.f2j_maven_coord)
             .getOrCreate())
    HAVE_DEEQU = True
except Exception as e:  # noqa: BLE001
    HAVE_DEEQU = False
    print("Deequ unavailable ->", type(e).__name__, str(e)[:120])


def deequ_flag(source_df, reference_df, domain):
    if not HAVE_DEEQU:
        return None
    missing = [c for c in domain.required_columns if c not in source_df.columns]
    if missing:
        return True
    sdf = spark.createDataFrame(source_df)
    chk = Check(spark, CheckLevel.Error, "readiness")
    for col in domain.feature_columns:
        if col in source_df.columns:
            chk = chk.isComplete(col)
            lo, hi, _ = numeric_bounds(reference_df, col)
            if lo is not None and hi > lo:
                chk = chk.hasMean(col, lambda m, lo=lo, hi=hi: lo <= m <= hi)
    res = VerificationSuite(spark).onData(sdf).addCheck(chk).run()
    rdf = VerificationResult.checkResultsAsDataFrame(spark, res)
    return any(r["constraint_status"] != "Success" for r in rdf.collect())


ADAPTERS = ["great_expectations", "tfdv", "deequ"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domains", nargs="*", default=["healthcare", "finance", "government"])
    ap.add_argument("--severities", nargs="*", default=["high"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-sources", type=int, default=6)
    ap.add_argument("--n-degraded", type=int, default=2)
    ap.add_argument("--n-rows", type=int, default=4000)
    ap.add_argument("--max-rows", type=int, default=40000)
    ap.add_argument("--out", default=str(ROOT / "experiments" / "results" / "baselines_detection.csv"))
    a = ap.parse_args()

    have = {"great_expectations": HAVE_GE, "tfdv": HAVE_TFDV, "deequ": HAVE_DEEQU}
    print("available baselines:", [b for b in ADAPTERS if have[b]] or "NONE")
    modes = ["none"] + deg.MODES
    seeds = list(range(a.seeds))
    rows = []
    for dom in a.domains:
        for mode in modes:
            sev_list = ["none"] if mode == "none" else a.severities
            for severity in sev_list:
                for seed in seeds:
                    cell = make_cell_sources(dom, mode, severity, seed, n_sources=a.n_sources,
                                             n_degraded=a.n_degraded, n_rows=a.n_rows, max_rows=a.max_rows)
                    d, ref, srcs = cell["domain"], cell["reference"], cell["sources"]
                    schema = ref_stats = None
                    per = {b: {"tp": 0, "deg": 0, "cf": 0, "cl": 0} for b in ADAPTERS}
                    for src, is_deg in srcs:
                        for b in ADAPTERS:
                            try:
                                if b == "great_expectations":
                                    fl = ge_flag(src, ref, d)
                                elif b == "tfdv":
                                    if schema is None:
                                        schema, ref_stats = tfdv_schema(ref, d)
                                    fl = tfdv_flag(src, ref, d, schema, ref_stats)
                                else:
                                    fl = deequ_flag(src, ref, d)
                            except Exception:  # noqa: BLE001
                                fl = None
                            if fl is None:
                                continue
                            if is_deg:
                                per[b]["deg"] += 1; per[b]["tp"] += int(fl)
                            else:
                                per[b]["cl"] += 1; per[b]["cf"] += int(fl)
                    for b, c in per.items():
                        rows.append({
                            "domain": dom, "mode": mode, "severity": severity, "seed": seed, "baseline": b,
                            "det_recall": round(c["tp"] / c["deg"], 4) if c["deg"] else "",
                            "false_flag_rate": round(c["cf"] / c["cl"], 4) if c["cl"] else "",
                        })
    bdf = pd.DataFrame(rows)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write("# External baselines (GE/TFDV/Deequ) default checks on the gate's sources\n")
        f.write("# generated by experiments/baselines/run_baselines.py (headless, Python 3.11 / Docker)\n")
        f.write(f"# domains={a.domains} severities={a.severities} seeds={a.seeds} "
                f"n_sources={a.n_sources} n_degraded={a.n_degraded}\n")
        bdf.to_csv(f, index=False)
    ran = [b for b in ADAPTERS if pd.to_numeric(bdf[bdf.baseline == b]["det_recall"], errors="coerce").notna().any()]
    print(f"wrote {out} ({len(bdf)} rows)")
    print("baselines with results:", ran or "NONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
