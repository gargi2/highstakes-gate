#!/usr/bin/env python
"""Pillar B — cross-domain gate benchmark grid.

For each domain x degradation-mode x severity x epsilon x seed: split the data
into simulated sources, degrade a subset with the taxonomy-derived injection,
run the REUSED validation gate (src/gate/), and record detection
precision/recall per mode, the readiness decisions, and the downstream model
AUC with vs without gating. One CSV per domain, each with a reproduction header.

The gate is imported and called, never modified. Degradations come from
src/domains/degradations.py (which wraps gate.partition.degrade). All output is
aggregates/decisions/metrics — never a data row.

Full grid (the human runs this — see the commands printed by --print-commands):
    epsilons {0.1,0.5,1,5,inf} x severities {low,med,high} x >=5 seeds.

Smoke (<2 min, CI/dev — synthetic only):
    .venv/Scripts/python.exe experiments/run_crossdomain.py --smoke

Reproduce a full synthetic run:
    .venv/Scripts/python.exe experiments/run_crossdomain.py --seeds 5

Real data (per domain; never run full grids from the assistant):
    ... --data-root healthcare=/path/to/clinic.csv --data-root finance=/path/to/edgar \\
        --data-root government=/path/to/openpayments
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

from gate.partition import split_sources, degrade                      # noqa: E402
from gate.expectations import run_suite                                # noqa: E402
from gate.compatibility import schema_compatibility                    # noqa: E402
from gate.consensus import consensus_checks, mean_divergence_per_source  # noqa: E402
from gate.readiness import score_source                                # noqa: E402
from domains import REGISTRY                                           # noqa: E402
from domains.base import train_eval                                    # noqa: E402
from domains import degradations as deg                                # noqa: E402

EPSILONS = [0.1, 0.5, 1.0, 5.0, math.inf]
COLUMNS = [
    "domain", "mode", "severity", "epsilon", "seed", "provenance",
    "n_sources", "n_degraded", "det_precision", "det_recall", "false_flag_rate",
    "ready", "fixable", "exclude", "degraded_withheld", "clean_excluded",
    "auc_nogate", "auc_gate", "auc_delta",
]


def _eps_str(e: float) -> str:
    return "inf" if math.isinf(e) else str(e)


def run_cell(domain, mode, severity, epsilon, seed, cfg) -> dict:
    df, prov = domain.get_frame(cfg["n_rows"], seed, cfg["data_root"].get(domain.name),
                                cfg["max_rows"])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_test = max(20, int(0.2 * len(df)))
    test_df = df.iloc[idx[:n_test]].reset_index(drop=True)          # clean holdout
    train_df = df.iloc[idx[n_test:]].reset_index(drop=True)

    sources = split_sources(train_df, cfg["n_sources"], seed)
    degraded_idx = set(range(cfg["n_degraded"])) if mode != "none" else set()
    spec = deg.build_spec(mode, severity, domain.feature_columns, domain.label_column) if mode != "none" else {}

    suite = domain.build_suite()
    frames, reports = [], []
    for i, src in enumerate(sources):
        if i in degraded_idx:
            src = degrade(src, spec, seed * 97 + i)
        frames.append(src)
        reports.append(run_suite(
            src, suite, source_id=f"s{i}", epsilon_budget=epsilon, seed=seed * 131 + i,
            histogram_columns=domain.histogram_columns, label_column=domain.label_column,
            histogram_ranges=domain.histogram_ranges,
        ))

    compat = schema_compatibility(reports, domain.required_columns)
    means = mean_divergence_per_source(reports, domain.histogram_columns)
    cons = consensus_checks(reports, domain.histogram_columns,
                            max_divergence=domain.divergence_threshold)

    tp = fp = fn = 0
    clean_flagged = clean_total = 0
    ready = fixable = exclude = 0
    degraded_withheld = clean_excluded = 0
    kept_frames = []
    for i, r in enumerate(reports):
        r.check_results = list(r.check_results) + cons[r.source_id]
        schema_ok = compat[r.source_id]["compatible"]
        rr = score_source(r, schema_ok, means.get(r.source_id))
        is_deg = i in degraded_idx
        flagged = deg.flagged_for_mode(mode, r.check_results, schema_ok) if mode != "none" else False
        if is_deg and flagged:
            tp += 1
        elif is_deg and not flagged:
            fn += 1
        elif (not is_deg) and flagged:
            fp += 1
        if not is_deg:
            clean_total += 1
            if deg.flagged_any(r.check_results, schema_ok):
                clean_flagged += 1
        ready += rr.decision == "READY"
        fixable += rr.decision == "FIXABLE"
        exclude += rr.decision == "EXCLUDE"
        if is_deg and rr.decision != "READY":
            degraded_withheld += 1
        if (not is_deg) and rr.decision == "EXCLUDE":
            clean_excluded += 1
        if rr.decision in ("READY", "FIXABLE"):
            kept_frames.append(frames[i])

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    false_flag_rate = clean_flagged / clean_total if clean_total else float("nan")

    auc_nogate = auc_gate = auc_delta = float("nan")
    if cfg["model"]:
        train_nogate = pd.concat(frames, ignore_index=True) if frames else None
        train_gate = pd.concat(kept_frames, ignore_index=True) if kept_frames else None
        auc_nogate = train_eval(train_nogate, test_df, domain.feature_columns, domain.label_column, seed)
        auc_gate = train_eval(train_gate, test_df, domain.feature_columns, domain.label_column, seed)
        if not (math.isnan(auc_nogate) or math.isnan(auc_gate)):
            auc_delta = auc_gate - auc_nogate

    return {
        "domain": domain.name, "mode": mode, "severity": severity,
        "epsilon": _eps_str(epsilon), "seed": seed, "provenance": prov,
        "n_sources": cfg["n_sources"], "n_degraded": len(degraded_idx),
        "det_precision": round(precision, 4) if not math.isnan(precision) else "",
        "det_recall": round(recall, 4) if not math.isnan(recall) else "",
        "false_flag_rate": round(false_flag_rate, 4) if not math.isnan(false_flag_rate) else "",
        "ready": ready, "fixable": fixable, "exclude": exclude,
        "degraded_withheld": degraded_withheld, "clean_excluded": clean_excluded,
        "auc_nogate": round(auc_nogate, 4) if not math.isnan(auc_nogate) else "",
        "auc_gate": round(auc_gate, 4) if not math.isnan(auc_gate) else "",
        "auc_delta": round(auc_delta, 4) if not math.isnan(auc_delta) else "",
    }


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def write_csv(path: Path, rows: list[dict], cfg, cmd: str) -> None:
    header = [
        f"# Pillar B cross-domain gate benchmark — domain={rows[0]['domain']}",
        f"# Reproduce: {cmd}",
        f"# git={git_rev()} epsilons={[_eps_str(e) for e in cfg['epsilons']]} "
        f"seeds={cfg['seeds']} n_sources={cfg['n_sources']} n_degraded={cfg['n_degraded']} "
        f"n_rows={cfg['n_rows']} max_rows={cfg['max_rows']} model={cfg['model']}",
        "# gate reused from src/gate/ (unmodified); degradations from src/domains/degradations.py",
        "# metrics: det_precision/recall = per-mode detection over sources; "
        "degraded_withheld = gate did not mark a degraded source READY; "
        "auc_* = downstream model on a clean holdout",
        "# columns: " + ",".join(COLUMNS),
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        for line in header:
            f.write(line + "\n")
        pd.DataFrame(rows, columns=COLUMNS).to_csv(f, index=False)


def print_commands() -> None:
    py = ".venv/Scripts/python.exe"
    print("# --- (a) Real per-domain extraction sanity-check (aggregates only) ---")
    print(f"{py} -m domains.healthcare  --data-root <CLINICAL_CSV_OR_DIR>")
    print(f"{py} -m domains.finance     --data-root <EDGAR_DIR_WITH_num.txt>")
    print(f"{py} -m domains.government  --data-root <OPENPAYMENTS_CSV_OR_DIR>")
    print("#   (run these from ./src, or set PYTHONPATH=src)")
    print()
    print("# --- (b) Full cross-domain grid (synthetic; ~minutes) ---")
    print(f"{py} experiments/run_crossdomain.py --seeds 5")
    print()
    print("# --- (b') Full grid on REAL data, per domain ---")
    print(f"{py} experiments/run_crossdomain.py --seeds 5 \\")
    print("     --data-root healthcare=<CLINICAL_CSV> \\")
    print("     --data-root finance=<EDGAR_DIR> \\")
    print("     --data-root government=<OPENPAYMENTS_CSV> --max-rows 200000")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domains", nargs="*", default=list(REGISTRY))
    ap.add_argument("--modes", nargs="*", default=["none"] + deg.MODES)
    ap.add_argument("--severities", nargs="*", default=deg.SEVERITIES)
    ap.add_argument("--epsilons", nargs="*", type=float, default=None)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-sources", type=int, default=6)
    ap.add_argument("--n-degraded", type=int, default=2)
    ap.add_argument("--n-rows", type=int, default=4000)
    ap.add_argument("--max-rows", type=int, default=40000)
    ap.add_argument("--data-root", action="append", default=[],
                    help="DOMAIN=PATH (repeatable); omit for synthetic.")
    ap.add_argument("--out-dir", default=str(ROOT / "experiments" / "results"))
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny synthetic grid, <2 min")
    ap.add_argument("--print-commands", action="store_true",
                    help="print the exact extraction + full-grid commands and exit")
    a = ap.parse_args()

    if a.print_commands:
        print_commands()
        return 0

    data_root = {}
    for item in a.data_root:
        if "=" not in item:
            ap.error(f"--data-root must be DOMAIN=PATH, got {item!r}")
        k, v = item.split("=", 1)
        data_root[k] = v

    if a.smoke:
        epsilons = [1.0, math.inf]
        seeds = list(range(2))
        modes = ["none"] + deg.MODES
        severities = ["high"]
        cfg = dict(n_rows=400, max_rows=2000, n_sources=5, n_degraded=2, model=True)
    else:
        epsilons = a.epsilons if a.epsilons is not None else EPSILONS
        seeds = list(range(a.seeds))
        modes = a.modes
        severities = a.severities
        cfg = dict(n_rows=a.n_rows, max_rows=a.max_rows, n_sources=a.n_sources,
                   n_degraded=a.n_degraded, model=not a.no_model)
    cfg["data_root"] = data_root
    cfg["epsilons"] = epsilons
    cfg["seeds"] = len(seeds)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ".venv/Scripts/python.exe experiments/run_crossdomain.py " + " ".join(sys.argv[1:])

    total = 0
    for dname in a.domains:
        if dname not in REGISTRY:
            ap.error(f"unknown domain {dname!r}; known: {list(REGISTRY)}")
        domain = REGISTRY[dname]
        rows = []
        for mode in modes:
            sev_list = ["none"] if mode == "none" else severities
            for severity in sev_list:
                for epsilon in epsilons:
                    for seed in seeds:
                        rows.append(run_cell(domain, mode, severity, epsilon, seed, cfg))
        suffix = "_smoke" if a.smoke else ""
        path = out_dir / f"crossdomain_{dname}{suffix}.csv"
        write_csv(path, rows, cfg, cmd)
        total += len(rows)
        print(f"wrote {path}  ({len(rows)} rows)")
        # tiny rows-free digest
        dfp = pd.DataFrame(rows)
        deg_rows = dfp[dfp["mode"] != "none"]
        if len(deg_rows):
            rec = pd.to_numeric(deg_rows["det_recall"], errors="coerce").mean()
            print(f"   mean per-mode detection recall (degraded cells): {rec:.3f}")
    print(f"done: {total} cells across {len(a.domains)} domain(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
