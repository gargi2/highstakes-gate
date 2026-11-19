#!/usr/bin/env python
"""Aggregation-rule ablation — does a different decision rule close the spine's
detection->action gap without an FPR explosion? (NEXT_EXPERIMENTS item 4.)

For each domain x mode x severity x epsilon x seed we build the SAME reports as
`run_crossdomain.py` (split -> degrade -> run_suite -> consensus -> compat ->
mean divergence), then decide each source under THREE rules on those identical
reports (src/domains/aggregation.py):

    weighted_mean : the frozen gate (gate.readiness.score_source), verbatim.
    max_violation : worst-of-three-subscores drives the decision.
    any_critical  : any single failed check -> EXCLUDE.

Detection is rule-INDEPENDENT (deg.flagged_for_mode — a mode check fired). Action
is rule-DEPENDENT (decision != READY -> the source is withheld). We report, per
rule: detection recall, action (withhold) rate, the gap between them, and the
clean false-positive rate (clean source not marked READY). The paper's claim to
test: max_violation / any_critical shrink the gap WITHOUT blowing up clean FPR.

The gate is imported and called, never modified. Output is aggregates/decisions
only — never a data row.

Smoke (<1 min, CI/dev — synthetic only):
    .venv/Scripts/python.exe experiments/run_aggregation_ablation.py --smoke

Real data (the human runs the full grid):
    .venv/Scripts/python.exe experiments/run_aggregation_ablation.py --seeds 5 --max-rows 40000 \
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
from gate.compatibility import schema_compatibility                      # noqa: E402
from gate.consensus import consensus_checks, mean_divergence_per_source  # noqa: E402
from domains import REGISTRY                                             # noqa: E402
from domains import degradations as deg                                  # noqa: E402
from domains.aggregation import decide, RULES                           # noqa: E402

EPSILONS = [0.1, 0.5, 1.0, 5.0, math.inf]
COLUMNS = [
    "domain", "mode", "severity", "epsilon", "seed", "rule",
    "n_sources", "n_degraded", "n_clean",
    "det_recall", "action_rate", "gap", "clean_fpr",
    "ready", "fixable", "exclude",
]

_FRAME_CACHE: dict[str, tuple[pd.DataFrame, str]] = {}


def _eps_str(e: float) -> str:
    return "inf" if math.isinf(e) else str(e)


def _frame_for(domain, seed, cfg) -> tuple[pd.DataFrame, str]:
    data_root = cfg["data_root"].get(domain.name)
    if data_root is None:
        return domain.get_frame(cfg["n_rows"], seed, None, cfg["max_rows"])
    if domain.name not in _FRAME_CACHE:
        _FRAME_CACHE[domain.name] = domain.get_frame(cfg["n_rows"], seed, data_root, cfg["max_rows"])
    return _FRAME_CACHE[domain.name]


def _build_reports(domain, mode, severity, epsilon, seed, cfg):
    """Reproduce run_crossdomain's per-cell report construction, verbatim."""
    df, prov = _frame_for(domain, seed, cfg)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_test = max(20, int(0.2 * len(df)))
    train_df = df.iloc[idx[n_test:]].reset_index(drop=True)

    sources = split_sources(train_df, cfg["n_sources"], seed)
    degraded_idx = set(range(cfg["n_degraded"])) if mode != "none" else set()
    spec = deg.build_spec(mode, severity, domain.feature_columns, domain.label_column) if mode != "none" else {}

    suite = domain.build_suite()
    reports = []
    for i, src in enumerate(sources):
        if i in degraded_idx:
            src = degrade(src, spec, seed * 97 + i)
        reports.append(run_suite(
            src, suite, source_id=f"s{i}", epsilon_budget=epsilon, seed=seed * 131 + i,
            histogram_columns=domain.histogram_columns, label_column=domain.label_column,
            histogram_ranges=domain.histogram_ranges,
        ))

    compat = schema_compatibility(reports, domain.required_columns)
    means = mean_divergence_per_source(reports, domain.histogram_columns)
    cons = consensus_checks(reports, domain.histogram_columns,
                            max_divergence=domain.divergence_threshold)
    for r in reports:
        r.check_results = list(r.check_results) + cons[r.source_id]
    return reports, compat, means, degraded_idx, prov


def run_cell(domain, mode, severity, epsilon, seed, cfg) -> list[dict]:
    reports, compat, means, degraded_idx, prov = _build_reports(
        domain, mode, severity, epsilon, seed, cfg)

    # Per-source: is it degraded? was it flagged (rule-independent detection)?
    flagged, is_deg, schema_oks = [], [], []
    for i, r in enumerate(reports):
        schema_ok = compat[r.source_id]["compatible"]
        schema_oks.append(schema_ok)
        is_deg.append(i in degraded_idx)
        flagged.append(
            deg.flagged_for_mode(mode, r.check_results, schema_ok) if mode != "none" else False)

    n_deg = sum(is_deg)
    n_clean = len(reports) - n_deg
    det_tp = sum(1 for i in range(len(reports)) if is_deg[i] and flagged[i])
    det_recall = det_tp / n_deg if n_deg else float("nan")

    rows = []
    for rule in cfg["rules"]:
        decisions = [
            decide(r, schema_oks[i], means.get(r.source_id), rule)
            for i, r in enumerate(reports)
        ]
        withheld_deg = sum(1 for i, d in enumerate(decisions) if is_deg[i] and d != "READY")
        clean_not_ready = sum(1 for i, d in enumerate(decisions) if not is_deg[i] and d != "READY")
        action_rate = withheld_deg / n_deg if n_deg else float("nan")
        clean_fpr = clean_not_ready / n_clean if n_clean else float("nan")
        gap = (det_recall - action_rate) if not math.isnan(det_recall) else float("nan")
        rows.append({
            "domain": domain.name, "mode": mode, "severity": severity,
            "epsilon": _eps_str(epsilon), "seed": seed, "rule": rule,
            "n_sources": cfg["n_sources"], "n_degraded": n_deg, "n_clean": n_clean,
            "det_recall": round(det_recall, 4) if not math.isnan(det_recall) else "",
            "action_rate": round(action_rate, 4) if not math.isnan(action_rate) else "",
            "gap": round(gap, 4) if not math.isnan(gap) else "",
            "clean_fpr": round(clean_fpr, 4) if not math.isnan(clean_fpr) else "",
            "ready": sum(d == "READY" for d in decisions),
            "fixable": sum(d == "FIXABLE" for d in decisions),
            "exclude": sum(d == "EXCLUDE" for d in decisions),
        })
    return rows


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def write_csv(path: Path, rows: list[dict], cfg, cmd: str) -> None:
    header = [
        f"# Aggregation-rule ablation — domain={rows[0]['domain']} (NEXT_EXPERIMENTS item 4)",
        f"# Reproduce: {cmd}",
        f"# git={git_rev()} rules={cfg['rules']} epsilons={[_eps_str(e) for e in cfg['epsilons']]} "
        f"seeds={cfg['seeds']} n_sources={cfg['n_sources']} n_degraded={cfg['n_degraded']} "
        f"n_rows={cfg['n_rows']} max_rows={cfg['max_rows']}",
        "# gate reused from src/gate/ (unmodified); decision rules from src/domains/aggregation.py",
        "# detection = deg.flagged_for_mode (rule-independent); action = decision != READY;",
        "# gap = det_recall - action_rate; clean_fpr = clean sources not marked READY",
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
    ap.add_argument("--rules", nargs="*", default=RULES)
    ap.add_argument("--epsilons", nargs="*", type=float, default=None)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-sources", type=int, default=6)
    ap.add_argument("--n-degraded", type=int, default=2)
    ap.add_argument("--n-rows", type=int, default=4000)
    ap.add_argument("--max-rows", type=int, default=40000)
    ap.add_argument("--data-root", action="append", default=[],
                    help="DOMAIN=PATH (repeatable); omit for synthetic.")
    ap.add_argument("--out-dir", default=str(ROOT / "experiments" / "results"))
    ap.add_argument("--smoke", action="store_true", help="tiny synthetic grid, <1 min")
    a = ap.parse_args()

    data_root = {}
    for item in a.data_root:
        if "=" not in item:
            ap.error(f"--data-root must be DOMAIN=PATH, got {item!r}")
        k, v = item.split("=", 1)
        data_root[k] = v

    for r in a.rules:
        if r not in RULES:
            ap.error(f"unknown rule {r!r}; known: {RULES}")

    if a.smoke:
        epsilons = [math.inf]
        seeds = list(range(2))
        modes = ["none"] + deg.MODES
        severities = ["high"]
        cfg = dict(n_rows=400, max_rows=2000, n_sources=5, n_degraded=2)
    else:
        epsilons = a.epsilons if a.epsilons is not None else EPSILONS
        seeds = list(range(a.seeds))
        modes = a.modes
        severities = a.severities
        cfg = dict(n_rows=a.n_rows, max_rows=a.max_rows, n_sources=a.n_sources,
                   n_degraded=a.n_degraded)
    cfg["data_root"] = data_root
    cfg["epsilons"] = epsilons
    cfg["seeds"] = len(seeds)
    cfg["rules"] = a.rules

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ".venv/Scripts/python.exe experiments/run_aggregation_ablation.py " + " ".join(sys.argv[1:])

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
                        rows.extend(run_cell(domain, mode, severity, epsilon, seed, cfg))
        suffix = "_smoke" if a.smoke else ""
        path = out_dir / f"aggregation_{dname}{suffix}.csv"
        write_csv(path, rows, cfg, cmd)
        total += len(rows)
        print(f"wrote {path}  ({len(rows)} rows)")
        # rows-free digest: mean gap and clean_fpr per rule over degraded cells
        dfp = pd.DataFrame(rows)
        deg_rows = dfp[dfp["mode"] != "none"]
        clean_rows = dfp[dfp["mode"] == "none"]
        for rule in a.rules:
            g = pd.to_numeric(deg_rows[deg_rows["rule"] == rule]["gap"], errors="coerce").mean()
            fpr = pd.to_numeric(clean_rows[clean_rows["rule"] == rule]["clean_fpr"], errors="coerce").mean()
            print(f"   {rule:>14}: mean gap (degraded) {g:.3f} | clean_fpr (mode=none) {fpr:.3f}")
    print(f"done: {total} rows across {len(a.domains)} domain(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
