#!/usr/bin/env python
"""Item 4 — analysis of the aggregation-rule ablation (the spine's "fix").

Reads the real per-domain aggregation_*.csv (from run_aggregation_ablation.py) and
emits seeded figures + a summary CSV. Two questions:
  figA1  Does a different decision rule CLOSE the detection->action gap? (gap by
         domain x rule, epsilon=inf.)
  figA2  Does it do so WITHOUT an FPR explosion — including under DP noise? (clean
         false-positive rate vs epsilon, per rule.)

Run:  .venv/Scripts/python.exe experiments/analyze_aggregation.py
Out:  paper/figures/figA1_gap_by_rule.{pdf,png}, figA2_fpr_vs_epsilon.{pdf,png},
      experiments/results/aggregation_summary.csv
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ci_utils import per_seed_means, bootstrap_ci  # noqa: E402  (experiments/ on sys.path[0])

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
FIGDIR = ROOT / "paper" / "figures"
SEED = 42
DOMAINS = ["healthcare", "finance", "government"]
RULES = ["weighted_mean", "max_violation", "any_critical"]
RULE_LABEL = {"weighted_mean": "weighted mean (gate)",
              "max_violation": "max-violation",
              "any_critical": "any-critical"}
RULE_COLOR = {"weighted_mean": "#999999", "max_violation": "#E69F00", "any_critical": "#0072B2"}


def read_results_csv(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        skip = sum(1 for line in f if line.startswith("#"))
    return pd.read_csv(path, skiprows=skip)


def load_all() -> pd.DataFrame:
    frames = []
    for d in DOMAINS:
        p = RESULTS / f"aggregation_{d}.csv"
        if p.exists():
            df = read_results_csv(p)
            frames.append(df)
    if not frames:
        sys.exit("no real aggregation_*.csv — run experiments/run_aggregation_ablation.py first")
    df = pd.concat(frames, ignore_index=True)
    df["epsilon"] = df["epsilon"].astype(str)
    for c in ("det_recall", "action_rate", "gap", "clean_fpr"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  wrote paper/figures/{name}.pdf + .png")


def fig_gap_by_rule(df: pd.DataFrame) -> None:
    """Gap (det_recall - action_rate) by domain x rule at epsilon=inf, over degraded cells."""
    deg = df[(df["mode"] != "none") & (df["epsilon"] == "inf")]
    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    x = np.arange(len(DOMAINS)); w = 0.26
    for j, rule in enumerate(RULES):
        vals, err = [], [[], []]
        for d in DOMAINS:
            m, lo, hi = bootstrap_ci(per_seed_means(deg[(deg.domain == d) & (deg.rule == rule)], "gap"))
            vals.append(m); err[0].append(m - lo); err[1].append(hi - m)
        ax.bar(x + (j - 1) * w, vals, w, yerr=err, capsize=2.5, ecolor="#333",
               error_kw={"lw": 0.9}, label=RULE_LABEL[rule], color=RULE_COLOR[rule])
        for i, v in enumerate(vals):
            ax.text(i + (j - 1) * w, v + (0.015 if v >= 0 else -0.03), f"{v:.2f}",
                    ha="center", fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, [d.capitalize() for d in DOMAINS])
    ax.set_ylabel("detection → action gap  (recall − withhold rate)")
    ax.set_ylim(-0.42, 0.85)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.set_title("The detection→action gap by aggregation rule", fontsize=11.5)
    # (provenance/annotation moved to the paper's Reproducibility paragraph)
    _save(fig, "figA1_gap_by_rule")


def fig_fpr_vs_epsilon(df: pd.DataFrame) -> None:
    """Clean false-positive rate (clean sources not marked READY) vs epsilon, per rule."""
    clean = df[df["mode"] == "none"].copy()
    order = ["0.1", "0.5", "1.0", "5.0", "inf"]
    clean = clean[clean["epsilon"].isin(order)]
    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    xs = np.arange(len(order))
    for rule in RULES:
        g = clean[clean["rule"] == rule]
        ys = [pd.to_numeric(g[g["epsilon"] == e]["clean_fpr"], errors="coerce").mean() for e in order]
        ax.plot(xs, ys, "-o", label=RULE_LABEL[rule], color=RULE_COLOR[rule])
    ax.set_xticks(xs, order)
    ax.set_xlabel("privacy budget ε  (← more DP noise · ∞ = no noise)")
    ax.set_ylabel("clean false-positive rate")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(ls=":", alpha=0.5)
    ax.legend(fontsize=9)
    ax.set_title("Clean false-positive rate vs privacy budget, by rule", fontsize=11.5)
    # (provenance/annotation moved to the paper's Reproducibility paragraph)
    _save(fig, "figA2_fpr_vs_epsilon")


def write_summary(df: pd.DataFrame) -> None:
    deg = df[df["mode"] != "none"]
    clean = df[df["mode"] == "none"]
    rows = []
    for eps in ["inf", "1.0", "0.5"]:
        for dom in DOMAINS:
            for rule in RULES:
                d = deg[(deg.domain == dom) & (deg.rule == rule) & (deg.epsilon == eps)]
                c = clean[(clean.domain == dom) & (clean.rule == rule) & (clean.epsilon == eps)]
                if len(d) == 0:
                    continue
                rows.append({
                    "epsilon": eps, "domain": dom, "rule": rule,
                    "det_recall": round(d["det_recall"].mean(), 4),
                    "action_rate": round(d["action_rate"].mean(), 4),
                    "gap": round(d["gap"].mean(), 4),
                    "clean_fpr": round(c["clean_fpr"].mean(), 4) if len(c) else "",
                })
    g = pd.DataFrame(rows)
    out = RESULTS / "aggregation_summary.csv"
    header = [
        "# Item 4 aggregation-rule ablation summary — from aggregation_*.csv",
        f"# Reproduce: .venv/Scripts/python.exe experiments/analyze_aggregation.py (git={git_rev()})",
        "# one row per (epsilon, domain, rule); means over modes x severities x seeds",
        "# gap = det_recall - action_rate (positive = flagged-but-not-withheld = the spine)",
    ]
    with open(out, "w", encoding="utf-8", newline="") as f:
        for line in header:
            f.write(line + "\n")
        g.to_csv(f, index=False)
    print(f"  wrote {out.relative_to(ROOT)} ({len(g)} rows)")


def main() -> int:
    np.random.seed(SEED)
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 200, "pdf.fonttype": 42})
    df = load_all()
    print(f"loaded {len(df)} rows (git={git_rev()})")
    fig_gap_by_rule(df)
    fig_fpr_vs_epsilon(df)
    write_summary(df)
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
