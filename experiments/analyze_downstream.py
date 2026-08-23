#!/usr/bin/env python
"""Pillar C items 1–2 — analysis of the redesigned downstream arm.

Reads the real per-domain downstream CSVs (experiments/results/downstream_*.csv,
produced by run_downstream.py) and emits seeded, publication-quality figures plus a
summary CSV. Never re-runs an experiment; only summarizes.

It answers the two questions the aggregate-AUC arm could not:
  (item 1) does data corruption damage the model on real data, in metrics AUC hides
           (worst-group accuracy, subgroup FNR gap, calibration/ECE), and does the
           gate recover it toward the oracle (clean_only) ceiling?
  (item 2) how does the harm scale with corruption SEVERITY, and (if a dilution
           sweep is present) with the number of degraded sources?

Figures (all from downstream_{healthcare,finance,government}.csv):
  figD1_ece_harm_recovery  ECE harm (corrupt−clean) and gate residual (gate−clean).
  figD2_wga_recovery       worst-group accuracy: ungated vs gate vs oracle ceiling.
  figD3_severity           ECE of corrupt_only vs severity (harm scales with severity).
  figD4_dilution           (only if >1 n_degraded present) harm vs #degraded sources.

Run:  .venv/Scripts/python.exe experiments/analyze_downstream.py
Out:  paper/figures/figD*.{pdf,png}, experiments/results/downstream_summary.csv
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
MODES = ["structural", "completeness", "distributional", "label"]
SEVERITIES = ["low", "med", "high"]
MODE_SHORT = {"structural": "struct.", "completeness": "complete.",
              "distributional": "distrib.", "label": "label"}
DOM_COLOR = {"healthcare": "#0072B2", "finance": "#D55E00", "government": "#009E73"}
METRICS = ["auc", "acc", "worst_group_acc", "subgroup_fnr_gap", "ece", "n_train_sources"]


def read_results_csv(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        skip = sum(1 for line in f if line.startswith("#"))
    return pd.read_csv(path, skiprows=skip)


def load_all() -> pd.DataFrame:
    """The main nd=2 downstream grid — drives figD1/D2/D3 and the summary."""
    frames = []
    for d in DOMAINS:
        path = RESULTS / f"downstream_{d}.csv"
        if not path.exists():
            continue
        df = read_results_csv(path)
        df["domain"] = d
        frames.append(df)
    if not frames:
        sys.exit("no real downstream_*.csv found — run experiments/run_downstream.py first")
    df = pd.concat(frames, ignore_index=True)
    for c in METRICS + (["n_degraded"] if "n_degraded" in df.columns else []):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_dilution() -> pd.DataFrame | None:
    """Concatenate the main grid (labelled n_degraded=2) with any
    experiments/results/dilution_nd*/downstream_*.csv (each carrying its own
    n_degraded), for the dilution curve figD4. Returns None if no sweep dir exists.
    """
    frames = []
    # main files are the committed n_degraded=2 baseline (predate the column)
    for d in DOMAINS:
        path = RESULTS / f"downstream_{d}.csv"
        if path.exists():
            df = read_results_csv(path); df["domain"] = d
            if "n_degraded" not in df.columns:
                df["n_degraded"] = 2
            frames.append(df)
    sweep = sorted(RESULTS.glob("dilution_nd*/downstream_*.csv"))
    if not sweep:
        return None
    for path in sweep:
        d = path.stem.replace("downstream_", "")
        if d not in DOMAINS:
            continue
        df = read_results_csv(path); df["domain"] = d
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    for c in METRICS + ["n_degraded"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _save(fig, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  wrote paper/figures/{name}.pdf + .png")


def _cond_means(df, metric):
    """(domain, mode, condition) -> mean(metric) over seeds+severities."""
    return df[df["mode"] != "none"].groupby(
        ["domain", "mode", "condition"])[metric].mean()


# ---------------------------------------------------------------------------
# figD1 — ECE harm (corrupt−clean) and gate residual (gate−clean)
# ---------------------------------------------------------------------------
def fig_ece_harm_recovery(df: pd.DataFrame) -> None:
    m = _cond_means(df, "ece")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
    x = np.arange(len(MODES)); w = 0.38
    for ax, d in zip(axes, DOMAINS):
        harm = [m.get((d, mo, "corrupt_only"), np.nan) - m.get((d, mo, "clean_only"), np.nan) for mo in MODES]
        resid = [m.get((d, mo, "gate_realistic"), np.nan) - m.get((d, mo, "clean_only"), np.nan) for mo in MODES]
        ax.bar(x - w / 2, harm, w, label="corruption harm (corrupt − clean)", color="#B00020")
        ax.bar(x + w / 2, resid, w, label="after gate (gate − clean)", color="#6699CC")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(d.capitalize(), fontsize=12)
        ax.set_xticks(x, [MODE_SHORT[mo] for mo in MODES], fontsize=9)
        ax.grid(axis="y", ls=":", alpha=0.5)
    axes[0].set_ylabel("Δ calibration error (ECE)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Item 1: label corruption wrecks calibration — and the gate does "
                 "not recover it", fontsize=13, y=1.01)
    fig.subplots_adjust(bottom=0.2)
    fig.text(0.5, -0.09,
             "Higher = worse calibration than clean. Red = UNDILUTED harm (corrupt_only); label "
             "raises ECE by ~0.25 (healthcare, government) — invisible to AUC (government AUC "
             "0.928→0.927). Blue = residual harm in the realistic gated pipeline (diluted):\n"
             "for label it stays clearly >0 (~0.07) because the gate keeps the label-corrupted "
             "sources — a miscalibrated model ships. For the other modes the gate leaves ≈0 "
             "residual. Detection≠action, now confirmed DOWNSTREAM.\n"
             "Source: downstream_{healthcare,finance,government}.csv (condition ECE means).",
             ha="center", fontsize=8.5)
    _save(fig, "figD1_ece_harm_recovery")


# ---------------------------------------------------------------------------
# figD2 — worst-group accuracy: ungated vs gate vs oracle ceiling
# ---------------------------------------------------------------------------
def fig_wga_recovery(df: pd.DataFrame) -> None:
    m = _cond_means(df, "worst_group_acc")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    x = np.arange(len(MODES)); w = 0.26
    for ax, d in zip(axes, DOMAINS):
        ung = [m.get((d, mo, "all_ungated"), np.nan) for mo in MODES]
        gat = [m.get((d, mo, "gate_realistic"), np.nan) for mo in MODES]
        ora = [m.get((d, mo, "clean_only"), np.nan) for mo in MODES]
        ax.bar(x - w, ung, w, label="ungated (all sources)", color="#CCCCCC")
        ax.bar(x, gat, w, label="gate kept", color=DOM_COLOR[d])
        ax.bar(x + w, ora, w, label="oracle (clean only)", color="#333333")
        ax.set_title(d.capitalize(), fontsize=12)
        ax.set_xticks(x, [MODE_SHORT[mo] for mo in MODES], fontsize=9)
        lo = np.nanmin(ung + gat + ora); ax.set_ylim(max(0, lo - 0.08), 1.0)
        ax.grid(axis="y", ls=":", alpha=0.5)
    axes[0].set_ylabel("worst-group accuracy")
    axes[0].legend(loc="lower left", fontsize=7.5)
    fig.suptitle("Item 1: where the gate recovers worst-group accuracy toward the "
                 "oracle (and where it does not)", fontsize=12.5, y=1.01)
    fig.subplots_adjust(bottom=0.18)
    fig.text(0.5, -0.06,
             "Gate bar reaching the oracle (black) = full recovery. Government STRUCTURAL is the "
             "clear win (gate 0.91 = oracle, vs 0.90 ungated); LABEL shows gate = ungated < oracle "
             "(not withheld → not recovered).\n"
             "Worst-group accuracy is otherwise flat on real data — ECE (figD1) is the sensitive "
             "metric here. Source: downstream_*.csv (worst_group_acc condition means).",
             ha="center", fontsize=8.5)
    _save(fig, "figD2_wga_recovery")


# ---------------------------------------------------------------------------
# figD3 — ECE of corrupt_only vs severity (harm scales with severity)
# ---------------------------------------------------------------------------
def fig_severity(df: pd.DataFrame) -> None:
    sub = df[(df["condition"] == "corrupt_only") & (df["mode"] != "none")]
    xs = list(range(len(SEVERITIES)))
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
    markers = {"structural": "o", "completeness": "s", "distributional": "^", "label": "D"}
    for ax, d in zip(axes, DOMAINS):
        g = sub[sub["domain"] == d]
        base = df[(df["domain"] == d) & (df["condition"] == "clean_only")]["ece"].mean()
        for mo in MODES:
            ys = [g[(g["mode"] == mo) & (g["severity"] == s)]["ece"].mean() for s in SEVERITIES]
            ax.plot(xs, ys, marker=markers[mo], label=MODE_SHORT[mo])
        ax.axhline(base, ls="--", color="grey", alpha=0.7, label="clean baseline")
        ax.set_title(d.capitalize(), fontsize=12)
        ax.set_xticks(xs, SEVERITIES)
        ax.set_xlabel("corruption severity")
        ax.grid(ls=":", alpha=0.5)
    axes[0].set_ylabel("calibration error (ECE), corrupt_only")
    axes[0].legend(loc="upper left", fontsize=7.5)
    fig.suptitle("Item 2: calibration harm scales monotonically with corruption "
                 "severity (label mode)", fontsize=12.5, y=1.01)
    fig.subplots_adjust(bottom=0.2)
    fig.text(0.5, -0.08,
             "ECE of the undiluted corrupt_only model vs severity. Label ECE climbs low→med→high "
             "(e.g. healthcare 0.15→0.29→0.43); other modes stay near the clean baseline (dashed). "
             "This is the severity–detectability curve the single-point arm could not show.\n"
             "Source: downstream_*.csv (condition=corrupt_only, ECE by severity).",
             ha="center", fontsize=8.5)
    _save(fig, "figD3_severity")


# ---------------------------------------------------------------------------
# figD4 — harm vs #degraded sources (only if a dilution sweep is present)
# ---------------------------------------------------------------------------
def fig_dilution(df: pd.DataFrame | None) -> bool:
    if df is None or "n_degraded" not in df.columns or df["n_degraded"].nunique() < 2:
        return False
    sub = df[(df["condition"] == "all_ungated") & (df["mode"] == "label")]
    levels = sorted(int(x) for x in df["n_degraded"].dropna().unique() if x > 0)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for d in DOMAINS:
        g = sub[sub["domain"] == d]
        ys = [g[g["n_degraded"] == n]["ece"].mean() for n in levels]
        ax.plot(levels, ys, "-o", color=DOM_COLOR[d], label=d.capitalize())
    ax.set_xlabel("# degraded sources (of 6)")
    ax.set_ylabel("ECE, all_ungated (diluted), label mode")
    ax.set_xticks(levels)
    ax.legend(fontsize=9)
    ax.set_title("Item 2: dilution — how many bad sources before pooled training "
                 "calibration breaks", fontsize=11.5)
    ax.grid(ls=":", alpha=0.5)
    fig.subplots_adjust(bottom=0.2)
    npts = len(levels)
    fig.text(0.5, -0.11,
             "As more of the 6 sources carry label noise, the DILUTED (all-sources) model's "
             "calibration degrades — quantifying the 2-of-6 dilution that flattened the old arm. "
             f"Government/healthcare climb steeply; finance (balanced label) barely moves.\n"
             f"({npts}-point curve, n_degraded ∈ {levels}; run --n-degraded 3 to extend it.) "
             "Source: downstream_*.csv + dilution_nd*/ (condition=all_ungated, mode=label).",
             ha="center", fontsize=8.5)
    _save(fig, "figD4_dilution")
    return True


def write_summary(df: pd.DataFrame) -> None:
    deg = df[df["mode"] != "none"]
    g = deg.groupby(["domain", "mode", "condition"])
    summ = g.agg(n=("seed", "count"),
                 auc=("auc", "mean"), acc=("acc", "mean"),
                 worst_group_acc=("worst_group_acc", "mean"),
                 subgroup_fnr_gap=("subgroup_fnr_gap", "mean"),
                 ece=("ece", "mean")).reset_index()
    for c in ["auc", "acc", "worst_group_acc", "subgroup_fnr_gap", "ece"]:
        summ[c] = summ[c].round(4)
    out = RESULTS / "downstream_summary.csv"
    header = [
        "# Pillar C downstream summary — aggregated from experiments/results/downstream_*.csv",
        f"# Reproduce: .venv/Scripts/python.exe experiments/analyze_downstream.py (seed={SEED}, git={git_rev()})",
        "# one row per (domain, mode, condition); means over seeds x severities",
        "# conditions: clean_only=oracle ceiling, all_ungated=diluted, corrupt_only=undiluted, gate_realistic=gate kept",
    ]
    with open(out, "w", encoding="utf-8", newline="") as f:
        for line in header:
            f.write(line + "\n")
        summ.to_csv(f, index=False)
    print(f"  wrote {out.relative_to(ROOT)} ({len(summ)} rows)")


def main() -> int:
    np.random.seed(SEED)
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 200, "pdf.fonttype": 42})
    df = load_all()
    print(f"loaded {len(df)} rows (git={git_rev()}, seed={SEED})")
    fig_ece_harm_recovery(df)
    fig_wga_recovery(df)
    fig_severity(df)
    if fig_dilution(load_dilution()):
        print("  (dilution sweep detected — figD4 written)")
    else:
        print("  (no dilution_nd*/ sweep — figD4 skipped; run the sweep to enable it)")
    write_summary(df)
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
