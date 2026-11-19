#!/usr/bin/env python
"""ε-sweep of the confident-learning label detector under DP (NEXT_EXPERIMENTS 3.3).

Finding D showed the confident-learning detector (cross-validated self-disagreement)
closes the symmetric-label blind spot at ε=∞. This sweeps the DP budget ε on that
statistic (`self_disagreement_rate(epsilon=...)`, Laplace 1/(n·ε)) to trace the
privacy–utility curve of the NEW detector — the piece the marginal ε-sweep (Finding
3) cannot show, because the marginal is blind to symmetric noise regardless of ε.

Also reports the per-domain CLEAN self-disagreement mean/std (the basis for the
`label_quality_checks` threshold calibration, item 3.3-iii) — computed on the
undegraded sources, never on the detection outcome.

Symmetric noise only (the case the confident detector is FOR). Output is metrics.

Smoke:  .venv/Scripts/python.exe experiments/run_label_epsilon.py --smoke
Real (human runs):
  .venv/Scripts/python.exe experiments/run_label_epsilon.py --seeds 5 --max-rows 40000 \
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

from gate.partition import split_sources, degrade                        # noqa: E402
from domains import REGISTRY                                             # noqa: E402
from domains.label_quality import (                                      # noqa: E402
    self_disagreement_rate, label_quality_checks, LABEL_QUALITY_CHECK_ID,
)

COLUMNS = ["domain", "epsilon", "rate", "seed", "det_precision", "det_recall",
           "clean_disagree_mean", "clean_disagree_std", "n_sources", "n_degraded"]
SYM_RATES = [0.15, 0.30, 0.45]
EPSILONS = [0.1, 0.5, 1.0, 5.0, math.inf]
_FRAME_CACHE: dict[str, tuple[pd.DataFrame, str]] = {}


def _eps_str(e: float) -> str:
    return "inf" if math.isinf(e) else str(e)


def _frame_for(domain, seed, cfg):
    dr = cfg["data_root"].get(domain.name)
    if dr is None:
        return domain.get_frame(cfg["n_rows"], seed, None, cfg["max_rows"])
    if domain.name not in _FRAME_CACHE:
        _FRAME_CACHE[domain.name] = domain.get_frame(cfg["n_rows"], seed, dr, cfg["max_rows"])
    return _FRAME_CACHE[domain.name]


def _pr(flags, degraded_idx):
    tp = sum(1 for i, f in enumerate(flags) if f and i in degraded_idx)
    fp = sum(1 for i, f in enumerate(flags) if f and i not in degraded_idx)
    fn = sum(1 for i, f in enumerate(flags) if (not f) and i in degraded_idx)
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    return prec, rec


def run_cell(domain, epsilon, rate, seed, cfg) -> dict:
    df, _ = _frame_for(domain, seed, cfg)
    sources = split_sources(df, cfg["n_sources"], seed)
    degraded_idx = set(range(cfg["n_degraded"]))
    frames = []
    for i, src in enumerate(sources):
        if i in degraded_idx:
            src = degrade(src, {"label_noise": {"column": domain.label_column, "rate": rate}}, seed * 97 + i)
        frames.append(src)

    eps = None if math.isinf(epsilon) else epsilon
    dis = {f"s{i}": self_disagreement_rate(f, domain.feature_columns, domain.label_column,
                                           seed=seed * 131 + i, epsilon=eps)
           for i, f in enumerate(frames)}
    cl = label_quality_checks(dis)
    flags = [any(c.check_id == LABEL_QUALITY_CHECK_ID and not c.passed for c in cl[f"s{i}"])
             for i in range(len(frames))]
    prec, rec = _pr(flags, degraded_idx)

    # clean-source disagreement calibration stat (undegraded sources only)
    clean_vals = [dis[f"s{i}"] for i in range(len(frames)) if i not in degraded_idx]
    clean_vals = [v for v in clean_vals if v == v]  # drop NaN
    cmean = float(np.mean(clean_vals)) if clean_vals else float("nan")
    cstd = float(np.std(clean_vals)) if len(clean_vals) > 1 else float("nan")

    return {"domain": domain.name, "epsilon": _eps_str(epsilon), "rate": rate, "seed": seed,
            "det_precision": "" if math.isnan(prec) else round(prec, 4),
            "det_recall": "" if math.isnan(rec) else round(rec, 4),
            "clean_disagree_mean": "" if math.isnan(cmean) else round(cmean, 4),
            "clean_disagree_std": "" if math.isnan(cstd) else round(cstd, 4),
            "n_sources": cfg["n_sources"], "n_degraded": cfg["n_degraded"]}


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def write_csv(path: Path, rows: list[dict], cfg, cmd: str) -> None:
    header = [
        f"# 3.3 confident-detector ε-sweep — domain={rows[0]['domain']}",
        f"# Reproduce: {cmd}",
        f"# git={git_rev()} seeds={cfg['seeds']} n_sources={cfg['n_sources']} "
        f"n_degraded={cfg['n_degraded']} n_rows={cfg['n_rows']} max_rows={cfg['max_rows']}",
        "# confident-learning self-disagreement under Laplace DP (1/(n·ε)); symmetric noise only",
        "# clean_disagree_* = mean/std self-disagreement on the undegraded sources (threshold calibration)",
        "# columns: " + ",".join(COLUMNS),
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        for line in header:
            f.write(line + "\n")
        pd.DataFrame(rows, columns=COLUMNS).to_csv(f, index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domains", nargs="*", default=list(REGISTRY))
    ap.add_argument("--epsilons", nargs="*", type=float, default=None)
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
        seeds = list(range(2)); epsilons = [1.0, math.inf]; rates = [0.45]
        cfg = dict(n_rows=1200, max_rows=4000, n_sources=6, n_degraded=2)
    else:
        seeds = list(range(a.seeds))
        epsilons = a.epsilons if a.epsilons is not None else EPSILONS
        rates = SYM_RATES
        cfg = dict(n_rows=a.n_rows, max_rows=a.max_rows, n_sources=a.n_sources, n_degraded=a.n_degraded)
    cfg["data_root"] = data_root
    cfg["seeds"] = len(seeds)

    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ".venv/Scripts/python.exe experiments/run_label_epsilon.py " + " ".join(sys.argv[1:])

    total = 0
    for dname in a.domains:
        if dname not in REGISTRY:
            ap.error(f"unknown domain {dname!r}")
        domain = REGISTRY[dname]
        rows = [run_cell(domain, eps, rate, seed, cfg)
                for eps in epsilons for rate in rates for seed in seeds]
        suffix = "_smoke" if a.smoke else ""
        path = out_dir / f"label_epsilon_{dname}{suffix}.csv"
        write_csv(path, rows, cfg, cmd)
        total += len(rows)
        dfp = pd.DataFrame(rows)
        dfp["det_recall"] = pd.to_numeric(dfp["det_recall"], errors="coerce")
        rec = dfp.groupby("epsilon")["det_recall"].mean()
        print(f"wrote {path}  ({len(rows)} rows) | recall by eps: {rec.round(3).to_dict()}")
    print(f"done: {total} rows across {len(a.domains)} domain(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
