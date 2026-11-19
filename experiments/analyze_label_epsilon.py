#!/usr/bin/env python
"""Item 3.3 — analysis of the confident-detector ε-sweep.

Reads label_epsilon_*.csv (run_label_epsilon.py) and emits a recall-vs-ε figure
(one line per domain, seed-level bootstrap 95% CI band) plus a summary CSV. The
story: unlike the divergence/marginal detectors (Finding 3, false-flags explode at
ε=0.1), the confident-learning self-disagreement detector holds its recall down to
ε=0.1 — it is the DP-robust label detector. Also reports the per-domain clean
self-disagreement std, the basis for the 0.05 flag threshold (calibration, 3.3-iii).

Run:  .venv/Scripts/python.exe experiments/analyze_label_epsilon.py
Out:  paper/figures/figE1_confident_epsilon.{pdf,png},
      experiments/results/label_epsilon_summary.csv
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

from ci_utils import per_seed_means, bootstrap_ci  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
FIGDIR = ROOT / "paper" / "figures"
DOMAINS = ["healthcare", "finance", "government"]
DOM_COLOR = {"healthcare": "#0072B2", "finance": "#D55E00", "government": "#009E73"}
EPS_ORDER = ["0.1", "0.5", "1.0", "5.0", "inf"]


def read_results_csv(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        skip = sum(1 for line in f if line.startswith("#"))
    return pd.read_csv(path, skiprows=skip)


def load_all() -> pd.DataFrame:
    frames = []
    for d in DOMAINS:
        p = RESULTS / f"label_epsilon_{d}.csv"
        if p.exists():
            df = read_results_csv(p); df["domain"] = d; frames.append(df)
    if not frames:
        sys.exit("no label_epsilon_*.csv — run experiments/run_label_epsilon.py first")
    df = pd.concat(frames, ignore_index=True)
    df["epsilon"] = df["epsilon"].astype(str)
    for c in ("det_recall", "det_precision", "clean_disagree_std", "clean_disagree_mean"):
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


def main() -> int:
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 200, "pdf.fonttype": 42})
    df = load_all()
    xs = np.arange(len(EPS_ORDER))

    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    summ_rows = []
    for d in DOMAINS:
        means, los, his = [], [], []
        for e in EPS_ORDER:
            sub = df[(df.domain == d) & (df.epsilon == e)]
            m, lo, hi = bootstrap_ci(per_seed_means(sub, "det_recall"))
            means.append(m); los.append(lo); his.append(hi)
            cstd = sub["clean_disagree_std"].mean()
            summ_rows.append({"domain": d, "epsilon": e, "det_recall": round(m, 4),
                              "det_recall_ci_lo": round(lo, 4), "det_recall_ci_hi": round(hi, 4),
                              "clean_disagree_std": round(cstd, 4) if cstd == cstd else ""})
        ax.plot(xs, means, "-o", color=DOM_COLOR[d], label=d.capitalize())
        ax.fill_between(xs, los, his, color=DOM_COLOR[d], alpha=0.15)
    ax.set_xticks(xs, ["0.1", "0.5", "1", "5", "∞"])
    ax.set_xlabel("privacy budget ε  (← more DP noise · ∞ = no noise)")
    ax.set_ylabel("confident-detector recall (symmetric noise)")
    ax.set_ylim(0, 1.08)
    ax.grid(ls=":", alpha=0.5)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_title("Confident-learning detector recall vs privacy budget", fontsize=11.5)
    # (provenance/annotation moved to the paper's Reproducibility paragraph)
    _save(fig, "figE1_confident_epsilon")

    out = RESULTS / "label_epsilon_summary.csv"
    header = [
        "# Item 3.3 confident-detector ε-sweep summary — from label_epsilon_*.csv",
        f"# Reproduce: .venv/Scripts/python.exe experiments/analyze_label_epsilon.py (git={git_rev()})",
        "# det_recall = confident detector over the corrupted sources; seed-level bootstrap 95% CI;",
        "# clean_disagree_std = self-disagreement std on undegraded sources (0.05-threshold calibration basis)",
    ]
    with open(out, "w", encoding="utf-8", newline="") as f:
        for line in header:
            f.write(line + "\n")
        pd.DataFrame(summ_rows).to_csv(f, index=False)
    print(f"  wrote {out.relative_to(ROOT)} ({len(summ_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
