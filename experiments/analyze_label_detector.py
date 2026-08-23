#!/usr/bin/env python
"""Pillar C item 3 — analysis of the label-detector comparison.

Reads the real per-domain label_detector_*.csv (from run_label_detector.py) and
emits seeded figures + a summary CSV. Two questions:
  figL1  SYMMETRIC noise: does the confident-learning detector close the marginal
         detector's blind spot (finance recall 0.00)?
  figL2  ASYMMETRIC noise: does the MARGINAL detector regain recall as asymmetry
         grows (the derivation p'=p+... predicts it)?

Run:  .venv/Scripts/python.exe experiments/analyze_label_detector.py
Out:  paper/figures/figL1_symmetric.{pdf,png}, figL2_asymmetric.{pdf,png},
      experiments/results/label_detector_summary.csv
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

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
FIGDIR = ROOT / "paper" / "figures"
SEED = 42
DOMAINS = ["healthcare", "finance", "government"]
DET_COLOR = {"marginal": "#999999", "confident": "#0072B2"}


def read_results_csv(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        skip = sum(1 for line in f if line.startswith("#"))
    return pd.read_csv(path, skiprows=skip)


def load_all() -> pd.DataFrame:
    frames = []
    for d in DOMAINS:
        p = RESULTS / f"label_detector_{d}.csv"
        if p.exists():
            df = read_results_csv(p); df["domain"] = d; frames.append(df)
    if not frames:
        sys.exit("no real label_detector_*.csv — run experiments/run_label_detector.py first")
    df = pd.concat(frames, ignore_index=True)
    for c in ("det_precision", "det_recall", "rate"):
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


def fig_symmetric(df: pd.DataFrame) -> None:
    sym = df[df["noise_type"] == "symmetric"]
    rec = sym.groupby(["domain", "detector"])["det_recall"].mean().unstack()
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(DOMAINS)); w = 0.38
    ax.bar(x - w / 2, [rec.loc[d, "marginal"] for d in DOMAINS], w,
           label="marginal (gate label-rate)", color=DET_COLOR["marginal"])
    ax.bar(x + w / 2, [rec.loc[d, "confident"] for d in DOMAINS], w,
           label="confident-learning (new)", color=DET_COLOR["confident"])
    for i, d in enumerate(DOMAINS):
        ax.text(i - w / 2, rec.loc[d, "marginal"] + 0.02, f"{rec.loc[d,'marginal']:.2f}", ha="center", fontsize=9)
        ax.text(i + w / 2, rec.loc[d, "confident"] + 0.02, f"{rec.loc[d,'confident']:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x, [d.capitalize() for d in DOMAINS])
    ax.set_ylabel("detection recall (symmetric label noise)")
    ax.set_ylim(0, 1.12)
    ax.legend(loc="lower center", fontsize=9)
    ax.set_title("Item 3: confident-learning closes the symmetric-label-noise blind spot",
                 fontsize=12)
    fig.text(0.5, -0.04,
             "Recall over the 2 label-corrupted sources, averaged over rates {0.15,0.30,0.45} and seeds. "
             "The marginal detector is blind on balanced-label FINANCE (recall ~0.00, Blocker 2);\n"
             "the confident-learning detector recovers it and also lifts the rare-label domains. "
             "Source: label_detector_*.csv (noise_type=symmetric).",
             ha="center", fontsize=8.5)
    _save(fig, "figL1_symmetric")


def fig_asymmetric(df: pd.DataFrame) -> None:
    asy = df[df["noise_type"] == "asymmetric"]
    rates = sorted(asy["rate"].dropna().unique())
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for d in DOMAINS:
        g = asy[(asy["domain"] == d) & (asy["detector"] == "marginal")]
        ys = [g[g["rate"] == r]["det_recall"].mean() for r in rates]
        ax.plot(rates, ys, "-o", label=d.capitalize())
    ax.set_xlabel("asymmetric flip rate (positives → negatives)")
    ax.set_ylabel("marginal-detector recall")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=9)
    ax.grid(ls=":", alpha=0.5)
    ax.set_title("Item 3: under ASYMMETRIC noise the marginal detector regains signal",
                 fontsize=12)
    fig.text(0.5, -0.04,
             "Asymmetric flips move the positive-rate marginal (p' = p − p·rate), so the gate's "
             "label-rate detector — blind to symmetric noise — recovers recall as the flip rate grows.\n"
             "This is the detectability-threshold curve the derivation predicts. "
             "Source: label_detector_*.csv (noise_type=asymmetric, detector=marginal).",
             ha="center", fontsize=8.5)
    _save(fig, "figL2_asymmetric")


def write_summary(df: pd.DataFrame) -> None:
    g = df.groupby(["domain", "noise_type", "detector"]).agg(
        n=("seed", "count"),
        det_recall=("det_recall", "mean"),
        det_precision=("det_precision", "mean")).reset_index()
    for c in ("det_recall", "det_precision"):
        g[c] = g[c].round(4)
    out = RESULTS / "label_detector_summary.csv"
    header = [
        "# Pillar C item 3 label-detector summary — from label_detector_*.csv",
        f"# Reproduce: .venv/Scripts/python.exe experiments/analyze_label_detector.py (seed={SEED}, git={git_rev()})",
        "# one row per (domain, noise_type, detector); means over rates x seeds",
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
    print(f"loaded {len(df)} rows (git={git_rev()}, seed={SEED})")
    fig_symmetric(df)
    fig_asymmetric(df)
    write_summary(df)
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
