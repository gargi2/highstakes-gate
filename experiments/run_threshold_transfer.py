#!/usr/bin/env python
"""Cold cross-domain threshold transfer (NEXT_EXPERIMENTS item 5 / 3.2).

The distributional detector's `divergence_threshold` is calibrated PER DOMAIN on
clean sources (a gate hyperparameter; docs/WALKTHROUGH.md). Finding F showed the
distributional detector does NOT transfer cleanly (σ=0.34). This quantifies the
COST of transfer: calibrate the threshold on one domain, apply it UNCHANGED to
another, and measure the distributional detection recall + the clean false-flag
rate vs each domain using its OWN calibrated threshold.

For each target domain × threshold-source domain (its `divergence_threshold`), over
distributional-degraded cells at ε=∞: recall over the degraded sources, false-flag
over the clean sources in the same cell. The diagonal (target == source) is the
"own threshold" baseline; off-diagonal is cold transfer. Output is metrics only.

The gate is imported and called, never modified.

Smoke:  .venv/Scripts/python.exe experiments/run_threshold_transfer.py --smoke
Real (human runs):
  .venv/Scripts/python.exe experiments/run_threshold_transfer.py --seeds 5 --max-rows 40000 \
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
from gate.expectations import run_suite                                  # noqa: E402
from gate.consensus import consensus_checks, DIVERGENCE_CHECK_ID         # noqa: E402
from domains import REGISTRY                                             # noqa: E402
from domains import degradations as deg                                  # noqa: E402

COLUMNS = ["target_domain", "threshold_source", "threshold_value", "severity",
           "seed", "det_recall", "false_flag_rate", "n_degraded", "n_clean"]
_FRAME_CACHE: dict[str, tuple[pd.DataFrame, str]] = {}


def _frame_for(domain, seed, cfg):
    dr = cfg["data_root"].get(domain.name)
    if dr is None:
        return domain.get_frame(cfg["n_rows"], seed, None, cfg["max_rows"])
    if domain.name not in _FRAME_CACHE:
        _FRAME_CACHE[domain.name] = domain.get_frame(cfg["n_rows"], seed, dr, cfg["max_rows"])
    return _FRAME_CACHE[domain.name]


def _base_reports(domain, severity, seed, cfg):
    """Distributional-degraded reports for one cell (ε=∞), plus the degraded index set."""
    df, _ = _frame_for(domain, seed, cfg)
    sources = split_sources(df, cfg["n_sources"], seed)
    degraded_idx = set(range(cfg["n_degraded"]))
    spec = deg.build_spec("distributional", severity, domain.feature_columns, domain.label_column)
    reports = []
    for i, src in enumerate(sources):
        if i in degraded_idx:
            src = degrade(src, spec, seed * 97 + i)
        reports.append(run_suite(
            src, domain.build_suite(), source_id=f"s{i}", epsilon_budget=math.inf,
            seed=seed * 131 + i, histogram_columns=domain.histogram_columns,
            label_column=domain.label_column, histogram_ranges=domain.histogram_ranges))
    return reports, degraded_idx


def _flagged_at(reports, domain, threshold):
    """Distributional flag per source under a given divergence threshold."""
    cons = consensus_checks(reports, domain.histogram_columns, max_divergence=threshold)
    flags = []
    for r in reports:
        failed = {c.check_id for c in cons[r.source_id] if not c.passed}
        flags.append(DIVERGENCE_CHECK_ID in failed)
    return flags


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def write_csv(path: Path, rows: list[dict], cfg, cmd: str) -> None:
    header = [
        "# Cold cross-domain threshold transfer (NEXT_EXPERIMENTS item 5 / 3.2)",
        f"# Reproduce: {cmd}",
        f"# git={git_rev()} seeds={cfg['seeds']} n_sources={cfg['n_sources']} "
        f"n_degraded={cfg['n_degraded']} n_rows={cfg['n_rows']} max_rows={cfg['max_rows']}",
        "# distributional detection at eps=inf; threshold_source supplies its "
        "divergence_threshold, applied to target_domain data (diagonal = own threshold).",
        "# det_recall over degraded sources; false_flag_rate over clean sources in the same cell",
        "# columns: " + ",".join(COLUMNS),
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        for line in header:
            f.write(line + "\n")
        pd.DataFrame(rows, columns=COLUMNS).to_csv(f, index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domains", nargs="*", default=list(REGISTRY))
    ap.add_argument("--severities", nargs="*", default=deg.SEVERITIES)
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
        seeds = list(range(2)); severities = ["high"]
        cfg = dict(n_rows=800, max_rows=2000, n_sources=5, n_degraded=2)
    else:
        seeds = list(range(a.seeds)); severities = a.severities
        cfg = dict(n_rows=a.n_rows, max_rows=a.max_rows, n_sources=a.n_sources, n_degraded=a.n_degraded)
    cfg["data_root"] = data_root
    cfg["seeds"] = len(seeds)

    domains = [REGISTRY[d] for d in a.domains if d in REGISTRY]
    thresholds = {d.name: d.divergence_threshold for d in domains}

    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ".venv/Scripts/python.exe experiments/run_threshold_transfer.py " + " ".join(sys.argv[1:])

    rows = []
    for target in domains:
        for severity in severities:
            for seed in seeds:
                reports, degraded_idx = _base_reports(target, severity, seed, cfg)
                n_deg = len(degraded_idx)
                n_clean = len(reports) - n_deg
                for src_name, thr in thresholds.items():
                    flags = _flagged_at(reports, target, thr)
                    tp = sum(1 for i, f in enumerate(flags) if f and i in degraded_idx)
                    fp = sum(1 for i, f in enumerate(flags) if f and i not in degraded_idx)
                    rows.append({
                        "target_domain": target.name, "threshold_source": src_name,
                        "threshold_value": round(thr, 5), "severity": severity, "seed": seed,
                        "det_recall": round(tp / n_deg, 4) if n_deg else "",
                        "false_flag_rate": round(fp / n_clean, 4) if n_clean else "",
                        "n_degraded": n_deg, "n_clean": n_clean,
                    })
    suffix = "_smoke" if a.smoke else ""
    path = out_dir / f"threshold_transfer{suffix}.csv"
    write_csv(path, rows, cfg, cmd)
    print(f"wrote {path}  ({len(rows)} rows)")
    # rows-free digest: recall on the diagonal (own) vs mean off-diagonal (transferred)
    dfp = pd.DataFrame(rows)
    dfp["det_recall"] = pd.to_numeric(dfp["det_recall"], errors="coerce")
    dfp["false_flag_rate"] = pd.to_numeric(dfp["false_flag_rate"], errors="coerce")
    own = dfp[dfp.target_domain == dfp.threshold_source]
    xfer = dfp[dfp.target_domain != dfp.threshold_source]
    print(f"   own-threshold: recall {own.det_recall.mean():.3f}  false_flag {own.false_flag_rate.mean():.3f}")
    print(f"   transferred:   recall {xfer.det_recall.mean():.3f}  false_flag {xfer.false_flag_rate.mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
