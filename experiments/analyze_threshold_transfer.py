#!/usr/bin/env python
"""Item 3.2 — analysis of cold cross-domain threshold transfer.

Reads threshold_transfer.csv (run_threshold_transfer.py) and emits a two-panel
heatmap (detection recall + clean false-flag, target × threshold-source) and a
summary CSV. The story: recall transfers, but a mismatched threshold inflates the
clean false-positive rate — so each domain needs its own calibration (Finding F).

Run:  .venv/Scripts/python.exe experiments/analyze_threshold_transfer.py
Out:  paper/figures/figT1_threshold_transfer.{pdf,png},
      experiments/results/threshold_transfer_summary.csv
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


def read_results_csv(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        skip = sum(1 for line in f if line.startswith("#"))
    return pd.read_csv(path, skiprows=skip)


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
    p = RESULTS / "threshold_transfer.csv"
    if not p.exists():
        sys.exit("no threshold_transfer.csv — run experiments/run_threshold_transfer.py first")
    df = read_results_csv(p)
    for c in ("det_recall", "false_flag_rate"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    rec = df.pivot_table(index="target_domain", columns="threshold_source",
                         values="det_recall").reindex(index=DOMAINS, columns=DOMAINS)
    ff = df.pivot_table(index="target_domain", columns="threshold_source",
                        values="false_flag_rate").reindex(index=DOMAINS, columns=DOMAINS)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    fig.subplots_adjust(wspace=0.5)
    for ax, mat, title, cmap in ((axes[0], rec, "Detection recall", "RdYlGn"),
                                 (axes[1], ff, "Clean false-flag rate", "RdYlGn_r")):
        im = ax.imshow(mat.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(DOMAINS)), [d.capitalize() for d in DOMAINS])
        ax.set_yticks(range(len(DOMAINS)), [d.capitalize() for d in DOMAINS])
        ax.set_xlabel("threshold calibrated on →")
        ax.set_ylabel("applied to (target)")
        ax.set_title(title, fontsize=12, pad=8)
        for i in range(len(DOMAINS)):
            for j in range(len(DOMAINS)):
                v = mat.values[i, j]
                edge = i == j
                ax.text(j, i, "n/a" if np.isnan(v) else f"{v:.2f}", ha="center", va="center",
                        color="black", fontsize=10, fontweight="bold" if edge else "normal")
                if edge:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                               edgecolor="#000", lw=2))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Cold threshold transfer: recall survives, clean false-flags do not "
                 "(boxed = own threshold)", fontsize=12.5, y=1.02)
    # (provenance/annotation moved to the paper's Reproducibility paragraph)
    _save(fig, "figT1_threshold_transfer")

    # summary CSV with seed-level bootstrap CIs per (target, source)
    rows = []
    for t in DOMAINS:
        for s in DOMAINS:
            sub = df[(df.target_domain == t) & (df.threshold_source == s)]
            rm, rlo, rhi = bootstrap_ci(per_seed_means(sub, "det_recall"))
            fm, flo, fhi = bootstrap_ci(per_seed_means(sub, "false_flag_rate"))
            rows.append({"target_domain": t, "threshold_source": s, "own": t == s,
                         "det_recall": round(rm, 4), "det_recall_ci_lo": round(rlo, 4),
                         "det_recall_ci_hi": round(rhi, 4), "false_flag_rate": round(fm, 4),
                         "false_flag_ci_lo": round(flo, 4), "false_flag_ci_hi": round(fhi, 4)})
    out = RESULTS / "threshold_transfer_summary.csv"
    header = [
        "# Item 3.2 cold threshold-transfer summary — from threshold_transfer.csv",
        f"# Reproduce: .venv/Scripts/python.exe experiments/analyze_threshold_transfer.py (git={git_rev()})",
        "# own=True is the diagonal (domain's own calibrated threshold); seed-level bootstrap 95% CIs",
    ]
    with open(out, "w", encoding="utf-8", newline="") as f:
        for line in header:
            f.write(line + "\n")
        pd.DataFrame(rows).to_csv(f, index=False)
    print(f"  wrote {out.relative_to(ROOT)} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
