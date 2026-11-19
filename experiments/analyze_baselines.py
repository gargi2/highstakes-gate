#!/usr/bin/env python
"""Item 6 — external-baseline vs gate comparison.

Reads experiments/results/baselines_detection.csv (from the Colab notebook) and the
committed crossdomain_*.csv (the gate), and emits a figure + summary. The claim:
off-the-shelf validators match the gate on structural/completeness (no raw-check
novelty claimed) and either miss the harder modes or pay a false-flag cost — the
gate's value is the DP + cross-source + decision layer, not the individual checks.

Baselines that did not run in the notebook (empty det_recall) are reported as
ABSENT, not zero. Comparison is at the baseline's grid: severity=high, epsilon=inf.

Run:  .venv/Scripts/python.exe experiments/analyze_baselines.py
Out:  paper/figures/figBL_gate_vs_baselines.{pdf,png},
      experiments/results/baselines_summary.csv
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
DOMAINS = ["healthcare", "finance", "government"]
MODES = ["structural", "completeness", "distributional", "label"]
BASELINES = ["great_expectations", "tfdv", "deequ"]
SERIES_COLOR = {"gate": "#0072B2", "great_expectations": "#E69F00",
                "tfdv": "#009E73", "deequ": "#CC79A7"}
SERIES_LABEL = {"gate": "gate", "great_expectations": "Great Expectations",
                "tfdv": "TFDV", "deequ": "Deequ"}


def read_results_csv(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        skip = sum(1 for line in f if line.startswith("#"))
    return pd.read_csv(path, skiprows=skip)


def load_baselines() -> pd.DataFrame:
    p = RESULTS / "baselines_detection.csv"
    if not p.exists():
        sys.exit("no baselines_detection.csv — run experiments/baselines/baselines_colab.ipynb in Colab first")
    df = read_results_csv(p)
    df["det_recall"] = pd.to_numeric(df["det_recall"], errors="coerce")
    df["false_flag_rate"] = pd.to_numeric(df["false_flag_rate"], errors="coerce")
    return df


def load_gate() -> pd.DataFrame:
    frames = []
    for d in DOMAINS:
        p = RESULTS / f"crossdomain_{d}.csv"
        if not p.exists():
            continue
        g = read_results_csv(p)
        g["epsilon"] = g["epsilon"].astype(str)
        g = g[(g["epsilon"] == "inf") & (g["severity"].isin(["high", "none"]))]
        frames.append(g)
    if not frames:
        sys.exit("no crossdomain_*.csv for the gate comparison")
    g = pd.concat(frames, ignore_index=True)
    for c in ("det_recall", "false_flag_rate"):
        g[c] = pd.to_numeric(g[c], errors="coerce")
    return g


def which_ran(base: pd.DataFrame) -> list[str]:
    ran = []
    for b in BASELINES:
        if base[base["baseline"] == b]["det_recall"].notna().any():
            ran.append(b)
    return ran


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


def fig_compare(base, gate, ran):
    """Detection recall by mode: gate vs each baseline that ran, averaged over domains."""
    series = ["gate"] + ran
    gdeg = gate[gate["mode"] != "none"]

    def rows_for(s, m):
        if s == "gate":
            return gdeg[gdeg["mode"] == m]
        return base[(base["baseline"] == s) & (base["mode"] == m)]

    fig, ax = plt.subplots(figsize=(9.2, 4.7))
    x = np.arange(len(MODES)); w = 0.8 / len(series)
    for j, s in enumerate(series):
        vals, err = [], [[], []]
        for m in MODES:
            mm, lo, hi = bootstrap_ci(per_seed_means(rows_for(s, m), "det_recall"))
            vals.append(mm); err[0].append(mm - lo); err[1].append(hi - mm)
        ax.bar(x + (j - (len(series) - 1) / 2) * w, vals, w, yerr=err, capsize=2.5,
               ecolor="#333", error_kw={"lw": 0.9},
               label=SERIES_LABEL[s], color=SERIES_COLOR[s])
    ax.set_xticks(x, [m.capitalize() for m in MODES])
    ax.set_ylabel("detection recall (severity=high, ε=∞)")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9, ncol=len(series))
    absent = [SERIES_LABEL[b] for b in BASELINES if b not in ran]
    note = f"  [absent this run: {', '.join(absent)}]" if absent else ""
    ax.set_title("Item 6: gate vs off-the-shelf validators, by degradation mode" + note, fontsize=11.5)
    fig.text(0.5, -0.05,
             "Recall over degraded sources, averaged over domains x seeds; whiskers = seed-level bootstrap 95% CI. "
             "Baselines match the gate on "
             "structural/completeness (no raw-check novelty claimed). The gap opens on the label mode:\n"
             "on balanced-label FINANCE a symmetric flip moves no marginal, so an expectation baseline misses it "
             "— the blind spot the confident-learning detector (Finding D) closes. Source: baselines_detection.csv, crossdomain_*.csv.",
             ha="center", fontsize=8.3)
    _save(fig, "figBL_gate_vs_baselines")


def write_summary(base, gate, ran):
    rows = []
    gdeg = gate[gate["mode"] != "none"]
    gclean = gate[gate["mode"] == "none"]
    for dom in DOMAINS:
        for m in MODES:
            rows.append({"domain": dom, "mode": m, "series": "gate",
                         "det_recall": round(gdeg[(gdeg.domain == dom) & (gdeg["mode"] == m)]["det_recall"].mean(), 4)})
        rows.append({"domain": dom, "mode": "none", "series": "gate",
                     "det_recall": "", "false_flag_rate": round(
                         gclean[gclean.domain == dom]["false_flag_rate"].mean(), 4)})
    for b in ran:
        sub = base[base["baseline"] == b]
        for dom in DOMAINS:
            for m in MODES:
                rows.append({"domain": dom, "mode": m, "series": b,
                             "det_recall": round(sub[(sub.domain == dom) & (sub["mode"] == m)]["det_recall"].mean(), 4)})
            rows.append({"domain": dom, "mode": "none", "series": b, "det_recall": "",
                         "false_flag_rate": round(
                             sub[(sub.domain == dom) & (sub["mode"] == "none")]["false_flag_rate"].mean(), 4)})
    g = pd.DataFrame(rows)
    out = RESULTS / "baselines_summary.csv"
    header = [
        "# Item 6 baseline-vs-gate summary — from baselines_detection.csv + crossdomain_*.csv",
        f"# Reproduce: .venv/Scripts/python.exe experiments/analyze_baselines.py (git={git_rev()})",
        f"# baselines that ran: {ran}; absent: {[b for b in BASELINES if b not in ran]}",
        "# det_recall over degraded sources; false_flag_rate on mode=none (clean); severity=high, eps=inf",
    ]
    with open(out, "w", encoding="utf-8", newline="") as f:
        for line in header:
            f.write(line + "\n")
        g.to_csv(f, index=False)
    print(f"  wrote {out.relative_to(ROOT)} ({len(g)} rows)")


def main() -> int:
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 200, "pdf.fonttype": 42})
    base = load_baselines()
    gate = load_gate()
    ran = which_ran(base)
    print(f"baselines that ran: {ran}  (git={git_rev()})")
    if not ran:
        sys.exit("no baseline produced det_recall — check the Colab run")
    fig_compare(base, gate, ran)
    write_summary(base, gate, ran)
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
