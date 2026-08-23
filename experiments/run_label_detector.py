#!/usr/bin/env python
"""Pillar C item 3 — does a confident-learning detector close the label blind spot?

The reused gate's marginal label-rate detector is provably blind to SYMMETRIC label
noise on balanced targets (Blocker 2; finance recall 0.00). This script benchmarks a
NEW confident-learning detector (`domains.label_quality`, an extension — the frozen
gate is untouched) against the marginal one, over:
  * symmetric noise (rates low/med/high) — where the marginal fails, and
  * asymmetric noise (positives->negatives at a swept rate) — where the marginal
    regains a signal, giving a detectability-threshold curve.

Per cell it splits the data into sources, degrades the first n_degraded with the
label noise, and scores each detector's per-source detection precision/recall. All
output is metrics — never a row. One CSV per domain: label_detector_<domain>.csv.

Smoke:  .venv/Scripts/python.exe experiments/run_label_detector.py --smoke
Full (human runs on real data):
  .venv/Scripts/python.exe experiments/run_label_detector.py --seeds 5 --max-rows 40000 \
    --data-root healthcare=data/healthcare --data-root finance=data \
    --data-root government=data/PGYR2025_P06302026_06032026
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gate.partition import split_sources, degrade                     # noqa: E402
from gate.expectations import run_suite                               # noqa: E402
from gate.consensus import label_rate_divergence_checks, LABEL_RATE_CHECK_ID  # noqa: E402
from domains import REGISTRY                                          # noqa: E402
from domains import degradations as deg                               # noqa: E402
from domains.label_quality import (                                   # noqa: E402
    self_disagreement_rate, label_quality_checks, LABEL_QUALITY_CHECK_ID,
)

COLUMNS = ["domain", "noise_type", "rate", "seed", "detector",
           "det_precision", "det_recall", "n_sources", "n_degraded"]
SYM_RATES = [0.15, 0.30, 0.45]
ASYM_RATES = [0.10, 0.20, 0.30, 0.45]
_FRAME_CACHE: dict[str, tuple[pd.DataFrame, str]] = {}


def _frame_for(domain, seed, cfg):
    dr = cfg["data_root"].get(domain.name)
    if dr is None:
        return domain.get_frame(cfg["n_rows"], seed, None, cfg["max_rows"])
    if domain.name not in _FRAME_CACHE:
        _FRAME_CACHE[domain.name] = domain.get_frame(cfg["n_rows"], seed, dr, cfg["max_rows"])
    return _FRAME_CACHE[domain.name]


def _pr(flags: list[bool], degraded_idx: set[int]) -> tuple[float, float]:
    tp = sum(1 for i, f in enumerate(flags) if f and i in degraded_idx)
    fp = sum(1 for i, f in enumerate(flags) if f and i not in degraded_idx)
    fn = sum(1 for i, f in enumerate(flags) if (not f) and i in degraded_idx)
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    return prec, rec


def run_cell(domain, noise_type, rate, seed, cfg) -> list[dict]:
    df, _ = _frame_for(domain, seed, cfg)
    sources = split_sources(df, cfg["n_sources"], seed)
    degraded_idx = set(range(cfg["n_degraded"]))
    frames = []
    for i, src in enumerate(sources):
        if i in degraded_idx:
            if noise_type == "symmetric":
                src = degrade(src, {"label_noise": {"column": domain.label_column, "rate": rate}}, seed * 97 + i)
            else:  # asymmetric: positives -> negatives at `rate`
                src = deg.degrade_labels_asymmetric(src, domain.label_column, rate_pos=rate, rate_neg=0.0, seed=seed * 97 + i)
        frames.append(src)

    # (a) marginal label-rate detector (the frozen gate), no DP noise (epsilon=inf)
    reports = [run_suite(f, domain.build_suite(), source_id=f"s{i}", epsilon_budget=math.inf,
                         seed=seed * 131 + i, histogram_columns=domain.histogram_columns,
                         label_column=domain.label_column, histogram_ranges=domain.histogram_ranges)
               for i, f in enumerate(frames)]
    marg = label_rate_divergence_checks(reports)  # dict sid -> CheckResult (may be absent)
    marg_flags = [(r.source_id in marg and not marg[r.source_id].passed) for r in reports]

    # (b) confident-learning detector (the extension)
    dis = {f"s{i}": self_disagreement_rate(f, domain.feature_columns, domain.label_column, seed=seed * 131 + i)
           for i, f in enumerate(frames)}
    cl = label_quality_checks(dis)
    cl_flags = [any(c.check_id == LABEL_QUALITY_CHECK_ID and not c.passed for c in cl[f"s{i}"])
                for i in range(len(frames))]

    rows = []
    for name, flags in (("marginal", marg_flags), ("confident", cl_flags)):
        p, r = _pr(flags, degraded_idx)
        rows.append({"domain": domain.name, "noise_type": noise_type, "rate": rate,
                     "seed": seed, "detector": name,
                     "det_precision": "" if math.isnan(p) else round(p, 4),
                     "det_recall": "" if math.isnan(r) else round(r, 4),
                     "n_sources": cfg["n_sources"], "n_degraded": cfg["n_degraded"]})
    return rows


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def write_csv(path: Path, rows: list[dict], cfg, cmd: str) -> None:
    header = [
        f"# Pillar C item 3 — label-noise detector comparison — domain={rows[0]['domain']}",
        f"# Reproduce: {cmd}",
        f"# git={git_rev()} seeds={cfg['seeds']} n_sources={cfg['n_sources']} "
        f"n_degraded={cfg['n_degraded']} n_rows={cfg['n_rows']} max_rows={cfg['max_rows']}",
        "# detectors: marginal=gate consensus label-rate (positive_rate_noised); "
        "confident=cross-validated self-disagreement (domains.label_quality, extension)",
        "# noise: symmetric=gate label_noise flip; asymmetric=positives->negatives at `rate` (marginal regains signal)",
        "# columns: " + ",".join(COLUMNS),
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        for line in header:
            f.write(line + "\n")
        pd.DataFrame(rows, columns=COLUMNS).to_csv(f, index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domains", nargs="*", default=list(REGISTRY))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-sources", type=int, default=6)
    ap.add_argument("--n-degraded", type=int, default=2)
    ap.add_argument("--n-rows", type=int, default=4000)
    ap.add_argument("--max-rows", type=int, default=40000)
    ap.add_argument("--data-root", action="append", default=[], help="DOMAIN=PATH (repeatable)")
    ap.add_argument("--out-dir", default=str(ROOT / "experiments" / "results"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    data_root = {}
    for item in a.data_root:
        if "=" not in item:
            ap.error(f"--data-root must be DOMAIN=PATH, got {item!r}")
        k, v = item.split("=", 1)
        data_root[k] = v

    if a.smoke:
        seeds = list(range(2))
        cfg = dict(n_rows=1200, max_rows=4000, n_sources=6, n_degraded=2)
    else:
        seeds = list(range(a.seeds))
        cfg = dict(n_rows=a.n_rows, max_rows=a.max_rows, n_sources=a.n_sources, n_degraded=a.n_degraded)
    cfg["data_root"] = data_root
    cfg["seeds"] = len(seeds)

    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ".venv/Scripts/python.exe experiments/run_label_detector.py " + " ".join(sys.argv[1:])

    total = 0
    for dname in a.domains:
        if dname not in REGISTRY:
            ap.error(f"unknown domain {dname!r}")
        domain = REGISTRY[dname]
        rows = []
        for nt, rates in (("symmetric", SYM_RATES), ("asymmetric", ASYM_RATES)):
            for rate in rates:
                for seed in seeds:
                    rows.extend(run_cell(domain, nt, rate, seed, cfg))
        suffix = "_smoke" if a.smoke else ""
        path = out_dir / f"label_detector_{dname}{suffix}.csv"
        write_csv(path, rows, cfg, cmd)
        total += len(rows)
        dfp = pd.DataFrame(rows)
        sym = dfp[dfp["noise_type"] == "symmetric"]
        rec = sym.groupby("detector")["det_recall"].apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
        print(f"wrote {path}  ({len(rows)} rows) | symmetric recall: {rec.round(3).to_dict()}")
    print(f"done: {total} rows across {len(a.domains)} domain(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
