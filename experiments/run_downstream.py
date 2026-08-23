#!/usr/bin/env python
"""Pillar C item 1 — redesigned downstream arm (subgroup metrics + undiluted signal).

WHY THIS EXISTS. The cross-domain grid (`run_crossdomain.py`) measured downstream
impact only as aggregate ROC-AUC on a clean holdout, and that arm turned out to be
underpowered: clean ≈ corrupted-ungated in every cell (`docs/FINDINGS.md` Finding 5),
because (a) only 2 of 6 sources are corrupted (dilution) and (b) aggregate AUC is the
least sensitive metric. This script fixes both, WITHOUT touching the detection grids:

  * it trains FOUR conditions so the corruption's effect is observable at all —
      clean_only     : train on the undegraded sources    (= the ORACLE ceiling)
      all_ungated    : train on all sources               (diluted, the old auc_nogate)
      corrupt_only   : train on the degraded sources only  (UNDILUTED damage)
      gate_realistic : train on the sources the real gate kept (READY/FIXABLE)
  * it reports sensitive metrics — worst-group accuracy, subgroup FNR gap, and
    calibration error (ECE) — alongside AUC (`domains.metrics`).

The gate, degradations, split, and model are all REUSED (imported, not modified).
All output is metrics/decisions — never a data row (CLAUDE.md).

Smoke (synthetic, <1 min; CI/dev):
    .venv/Scripts/python.exe experiments/run_downstream.py --smoke

Full real-data run (the HUMAN runs this; heavy I/O reads the extracts once/domain):
    .venv/Scripts/python.exe experiments/run_downstream.py --seeds 5 --max-rows 40000 \
      --data-root healthcare=data/healthcare --data-root finance=data \
      --data-root government=data/PGYR2025_P06302026_06032026
One CSV per domain: experiments/results/downstream_<domain>.csv (reproduction header).
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

from gate.partition import split_sources, degrade                       # noqa: E402
from gate.expectations import run_suite                                 # noqa: E402
from gate.compatibility import schema_compatibility                     # noqa: E402
from gate.consensus import consensus_checks, mean_divergence_per_source  # noqa: E402
from gate.readiness import score_source                                 # noqa: E402
from domains import REGISTRY                                            # noqa: E402
from domains import degradations as deg                                 # noqa: E402
from domains.metrics import evaluate_downstream, DownstreamMetrics      # noqa: E402

COLUMNS = [
    "domain", "mode", "severity", "epsilon", "seed", "provenance", "condition",
    "n_train_sources", "auc", "acc", "worst_group_acc", "subgroup_fnr_gap", "ece",
    "n_groups_used",
]
_FRAME_CACHE: dict[str, tuple[pd.DataFrame, str]] = {}


def _eps_str(e: float) -> str:
    return "inf" if math.isinf(e) else str(e)


def _frame_for(domain, seed, cfg):
    data_root = cfg["data_root"].get(domain.name)
    if data_root is None:
        return domain.get_frame(cfg["n_rows"], seed, None, cfg["max_rows"])
    if domain.name not in _FRAME_CACHE:
        _FRAME_CACHE[domain.name] = domain.get_frame(cfg["n_rows"], seed, data_root, cfg["max_rows"])
    return _FRAME_CACHE[domain.name]


def _gate_kept(reports, domain, epsilon) -> list[bool]:
    """Run the REUSED gate and return keep[i] (READY/FIXABLE) per source."""
    compat = schema_compatibility(reports, domain.required_columns)
    means = mean_divergence_per_source(reports, domain.histogram_columns)
    cons = consensus_checks(reports, domain.histogram_columns,
                            max_divergence=domain.divergence_threshold)
    keep = []
    for r in reports:
        r.check_results = list(r.check_results) + cons[r.source_id]
        rr = score_source(r, compat[r.source_id]["compatible"], means.get(r.source_id))
        keep.append(rr.decision in ("READY", "FIXABLE"))
    return keep


def _concat(frames):
    return pd.concat(frames, ignore_index=True) if frames else None


def run_cell(domain, mode, severity, epsilon, seed, cfg) -> list[dict]:
    df, prov = _frame_for(domain, seed, cfg)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_test = max(20, int(0.2 * len(df)))
    test_df = df.iloc[idx[:n_test]].reset_index(drop=True)          # clean holdout
    train_df = df.iloc[idx[n_test:]].reset_index(drop=True)

    sources = split_sources(train_df, cfg["n_sources"], seed)
    degraded_idx = set(range(cfg["n_degraded"])) if mode != "none" else set()
    spec = deg.build_spec(mode, severity, domain.feature_columns, domain.label_column) if mode != "none" else {}

    frames, reports = [], []
    suite = domain.build_suite()
    for i, src in enumerate(sources):
        if i in degraded_idx:
            src = degrade(src, spec, seed * 97 + i)
        frames.append(src)
        reports.append(run_suite(
            src, suite, source_id=f"s{i}", epsilon_budget=epsilon, seed=seed * 131 + i,
            histogram_columns=domain.histogram_columns, label_column=domain.label_column,
            histogram_ranges=domain.histogram_ranges,
        ))

    clean_frames = [frames[i] for i in range(len(frames)) if i not in degraded_idx]
    corrupt_frames = [frames[i] for i in degraded_idx]
    if mode == "none":
        conditions = {"clean_baseline": frames}   # all sources are clean
    else:
        keep = _gate_kept(reports, domain, epsilon)
        conditions = {
            "clean_only": clean_frames,                                   # oracle ceiling
            "all_ungated": frames,                                        # dilution floor
            "corrupt_only": corrupt_frames,                              # undiluted damage
            "gate_realistic": [frames[i] for i in range(len(frames)) if keep[i]],
        }

    subgroup = domain.subgroup_column or domain.feature_columns[0]
    rows = []
    for cond, fr in conditions.items():
        m: DownstreamMetrics = evaluate_downstream(
            _concat(fr), test_df, domain.feature_columns, domain.label_column,
            subgroup, seed)
        rows.append({
            "domain": domain.name, "mode": mode, "severity": severity,
            "epsilon": _eps_str(epsilon), "seed": seed, "provenance": prov,
            "condition": cond, "n_train_sources": len(fr),
            "auc": _r(m.auc), "acc": _r(m.acc), "worst_group_acc": _r(m.worst_group_acc),
            "subgroup_fnr_gap": _r(m.subgroup_fnr_gap), "ece": _r(m.ece),
            "n_groups_used": m.n_groups_used,
        })
    return rows


def _r(x):
    return round(x, 4) if (x is not None and not (isinstance(x, float) and math.isnan(x))) else ""


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def write_csv(path: Path, rows: list[dict], cfg, cmd: str, subgroup: str) -> None:
    header = [
        f"# Pillar C downstream arm (item 1) — domain={rows[0]['domain']}",
        f"# Reproduce: {cmd}",
        f"# git={git_rev()} epsilon={_eps_str(cfg['epsilon'])} seeds={cfg['seeds']} "
        f"n_sources={cfg['n_sources']} n_degraded={cfg['n_degraded']} "
        f"n_rows={cfg['n_rows']} max_rows={cfg['max_rows']} subgroup={subgroup}",
        "# conditions: clean_only=oracle ceiling (undegraded sources); all_ungated=diluted; "
        "corrupt_only=undiluted damage; gate_realistic=what the real gate kept",
        "# metrics on the clean holdout: auc, acc, worst_group_acc (min over subgroup "
        "quantile bins), subgroup_fnr_gap (max-min FNR), ece (calibration error)",
        "# columns: " + ",".join(COLUMNS),
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        for line in header:
            f.write(line + "\n")
        pd.DataFrame(rows, columns=COLUMNS).to_csv(f, index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domains", nargs="*", default=list(REGISTRY))
    ap.add_argument("--modes", nargs="*", default=["none"] + deg.MODES)
    ap.add_argument("--severities", nargs="*", default=deg.SEVERITIES)
    ap.add_argument("--epsilon", type=float, default=math.inf,
                    help="single DP budget for the gate_realistic condition (default inf).")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-sources", type=int, default=6)
    ap.add_argument("--n-degraded", type=int, default=2)
    ap.add_argument("--n-rows", type=int, default=4000)
    ap.add_argument("--max-rows", type=int, default=40000)
    ap.add_argument("--data-root", action="append", default=[], help="DOMAIN=PATH (repeatable)")
    ap.add_argument("--out-dir", default=str(ROOT / "experiments" / "results"))
    ap.add_argument("--smoke", action="store_true", help="tiny synthetic grid, <1 min")
    a = ap.parse_args()

    data_root = {}
    for item in a.data_root:
        if "=" not in item:
            ap.error(f"--data-root must be DOMAIN=PATH, got {item!r}")
        k, v = item.split("=", 1)
        data_root[k] = v

    if a.smoke:
        seeds = list(range(2))
        modes = ["none"] + deg.MODES
        severities = ["high"]
        cfg = dict(n_rows=800, max_rows=4000, n_sources=6, n_degraded=2, epsilon=math.inf)
    else:
        seeds = list(range(a.seeds))
        modes, severities = a.modes, a.severities
        cfg = dict(n_rows=a.n_rows, max_rows=a.max_rows, n_sources=a.n_sources,
                   n_degraded=a.n_degraded, epsilon=a.epsilon)
    cfg["data_root"] = data_root
    cfg["seeds"] = len(seeds)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ".venv/Scripts/python.exe experiments/run_downstream.py " + " ".join(sys.argv[1:])

    total = 0
    for dname in a.domains:
        if dname not in REGISTRY:
            ap.error(f"unknown domain {dname!r}; known: {list(REGISTRY)}")
        domain = REGISTRY[dname]
        rows = []
        for mode in modes:
            sev_list = ["none"] if mode == "none" else severities
            for severity in sev_list:
                for seed in seeds:
                    rows.extend(run_cell(domain, mode, severity, cfg["epsilon"], seed, cfg))
        subgroup = domain.subgroup_column or domain.feature_columns[0]
        suffix = "_smoke" if a.smoke else ""
        path = out_dir / f"downstream_{dname}{suffix}.csv"
        write_csv(path, rows, cfg, cmd, subgroup)
        total += len(rows)
        print(f"wrote {path}  ({len(rows)} rows; subgroup={subgroup})")
        # rows-free digest: worst-group harm from corruption (clean_only vs corrupt_only)
        dfp = pd.DataFrame(rows)
        for c in ["worst_group_acc", "auc"]:
            dfp[c] = pd.to_numeric(dfp[c], errors="coerce")
        deg_rows = dfp[dfp["mode"] != "none"]
        if len(deg_rows):
            piv = deg_rows.pivot_table(index="mode", columns="condition", values="worst_group_acc")
            if {"clean_only", "corrupt_only"}.issubset(piv.columns):
                gap = (piv["clean_only"] - piv["corrupt_only"]).round(3)
                print(f"   worst-group-acc drop (clean_only - corrupt_only) by mode: {gap.to_dict()}")
    print(f"done: {total} rows across {len(a.domains)} domain(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
