#!/usr/bin/env python
"""Pillar C — analysis of the cross-domain gate benchmark (Pillar B grids).

Reads the committed per-domain result CSVs (experiments/results/crossdomain_*.csv,
each with a leading ``#`` reproduction header) and emits publication-quality,
seeded figures (PDF + PNG) plus one machine-readable metrics summary CSV. It does
NOT re-run any experiment; it only summarizes what the grids already recorded.

Every number a figure shows is traceable to a named source CSV, and every figure
is regenerated deterministically from those CSVs (no randomness; seed is fixed
and recorded for the project's fixed-seed rule and matplotlib font hashing).

Figures (all from experiments/results/crossdomain_{healthcare,finance,government}.csv):
    fig1_detection_by_mode   detection recall + precision, mode x domain, eps=inf.
    fig2_detect_vs_withhold  detection recall vs the gate ACTION (share of
                             degraded sources withheld) — they differ.
    fig3_privacy_utility     distributional detection precision/recall and the
                             clean-data false-flag rate vs privacy budget eps.
    fig4_cross_domain        per-mode detection recall across the three domains
                             (transfer/consistency); structural transfers,
                             label/distributional do not.
    fig5_downstream          downstream AUC change from gating (auc_delta),
                             mode x domain — mostly ~0 (reported honestly).

Run:
    .venv/Scripts/python.exe experiments/analyze.py
Outputs:
    paper/figures/*.pdf, paper/figures/*.png
    experiments/results/metrics_summary.csv  (one row per (domain,mode,epsilon))
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless, deterministic
import matplotlib.pyplot as plt  # noqa: E402

from ci_utils import per_seed_means, bootstrap_ci  # noqa: E402  (experiments/ on sys.path[0])

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
FIGDIR = ROOT / "paper" / "figures"

SEED = 42  # project fixed-seed rule; analysis is deterministic, kept for reproducibility
DOMAINS = ["healthcare", "finance", "government"]
MODES = ["structural", "completeness", "distributional", "label"]
EPS_ORDER = [0.1, 0.5, 1.0, 5.0, np.inf]
N_DEGRADED = 2  # sources degraded per cell in the Pillar B grid (n_degraded)

# Colorblind-safe (Wong/Okabe-Ito) palette, one hue per domain.
DOM_COLOR = {"healthcare": "#0072B2", "finance": "#D55E00", "government": "#009E73"}
MODE_LABEL = {
    "structural": "Structural\n(schema/dtype)",
    "completeness": "Completeness\n(nulls)",
    "distributional": "Distributional\n(feature shift)",
    "label": "Label\n(flip)",
}
# Short single-line labels for the narrow grouped-bar panels (fig2/4/5).
MODE_SHORT = {"structural": "struct.", "completeness": "complete.",
              "distributional": "distrib.", "label": "label"}

# Numeric columns to coerce (blank cells = NaN, e.g. precision when tp+fp==0).
NUM_COLS = [
    "epsilon", "det_precision", "det_recall", "false_flag_rate",
    "ready", "fixable", "exclude", "degraded_withheld", "clean_excluded",
    "auc_nogate", "auc_gate", "auc_delta",
]


def read_results_csv(path: Path) -> pd.DataFrame:
    """Skip only the contiguous leading ``#`` reproduction-header lines.

    Do NOT use ``pd.read_csv(comment='#')`` on project result CSVs: some string
    fields elsewhere in the project contain ``#``. Here we skip the header block
    explicitly, matching experiments/code_incidents.py.
    """
    with open(path, encoding="utf-8") as f:
        skip = sum(1 for line in f if line.startswith("#"))
    return pd.read_csv(path, skiprows=skip)


def load_all() -> pd.DataFrame:
    frames = []
    for d in DOMAINS:
        path = RESULTS / f"crossdomain_{d}.csv"
        df = read_results_csv(path)
        df["domain"] = d
        df["source_csv"] = path.name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")  # "inf" -> np.inf
    return df


def git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _eps_label(e: float) -> str:
    return "∞" if np.isinf(e) else (f"{e:g}")


def _save(fig, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  wrote paper/figures/{name}.pdf + .png")


# ----------------------------------------------------------------------------
# Figure 1 — detection recall & precision, mode x domain, at eps=inf (no DP noise)
# ----------------------------------------------------------------------------
def fig_detection_by_mode(df: pd.DataFrame) -> None:
    sub = df[(df["mode"] != "none") & np.isinf(df["epsilon"])]
    rec = sub.groupby(["mode", "domain"])["det_recall"].mean().unstack()
    prec = sub.groupby(["mode", "domain"])["det_precision"].mean().unstack()
    rec = rec.reindex(index=MODES, columns=DOMAINS)
    prec = prec.reindex(index=MODES, columns=DOMAINS)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    fig.subplots_adjust(wspace=0.55)
    for k, (ax, mat, title) in enumerate((
            (axes[0], rec, "Detection recall"),
            (axes[1], prec, "Detection precision"))):
        im = ax.imshow(mat.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(DOMAINS)), [d.capitalize() for d in DOMAINS])
        if k == 0:
            ax.set_yticks(range(len(MODES)), [MODE_LABEL[m] for m in MODES])
        else:
            ax.set_yticks(range(len(MODES)), [""] * len(MODES))
        ax.set_title(title, fontsize=12, pad=8)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                txt = "n/a" if np.isnan(v) else f"{v:.2f}"
                ax.text(j, i, txt, ha="center", va="center",
                        color="black", fontsize=10, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Per-mode detection at ε=∞ (no DP noise), real data",
                 fontsize=13, y=1.02)
    fig.text(0.5, -0.06,
             "Recall = share of the 2 degraded sources the mode's own gate check flagged; "
             "precision = share of flags that were truly degraded.\n"
             "Structural detection is perfect everywhere; the finance label cell is 0.00 "
             "(balanced labels defeat the label-rate detector).\n"
             "Source: experiments/results/crossdomain_{healthcare,finance,government}.csv",
             ha="center", fontsize=8.5)
    _save(fig, "fig1_detection_by_mode")


# ----------------------------------------------------------------------------
# Figure 2 — detection vs withholding (the gate action) at eps=inf
# ----------------------------------------------------------------------------
def fig_detect_vs_withhold(df: pd.DataFrame) -> None:
    sub = df[(df["mode"] != "none") & np.isinf(df["epsilon"])].copy()
    sub["withheld_frac"] = sub["degraded_withheld"] / N_DEGRADED
    rec = sub.groupby(["domain", "mode"])["det_recall"].mean().unstack().reindex(
        index=DOMAINS, columns=MODES)
    wh = sub.groupby(["domain", "mode"])["withheld_frac"].mean().unstack().reindex(
        index=DOMAINS, columns=MODES)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    x = np.arange(len(MODES))
    w = 0.38
    for ax, d in zip(axes, DOMAINS):
        ax.bar(x - w / 2, rec.loc[d].values, w, label="detected (recall)",
               color=DOM_COLOR[d], alpha=0.95)
        ax.bar(x + w / 2, wh.loc[d].values, w, label="withheld by gate",
               color=DOM_COLOR[d], alpha=0.45, hatch="//")
        ax.set_title(d.capitalize(), fontsize=12)
        ax.set_xticks(x, [MODE_SHORT[m] for m in MODES], fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", ls=":", alpha=0.5)
    axes[0].set_ylabel("fraction of degraded sources")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.subplots_adjust(bottom=0.16)
    fig.suptitle("Detection is not withholding: a flagged source is often still "
                 "kept (ε=∞)", fontsize=13, y=1.00)
    fig.text(0.5, -0.08,
             "Solid = detection recall (a consensus check fired); hatched = the gate's "
             "READY/EXCLUDE decision actually withheld the source.\n"
             "Only structural (schema-incompatible) is reliably withheld; distributional/label "
             "flags are one check among ~15 and rarely flip the decision.\n"
             "Source: crossdomain_{healthcare,finance,government}.csv (columns det_recall, degraded_withheld).",
             ha="center", fontsize=8.5)
    _save(fig, "fig2_detect_vs_withhold")


# ----------------------------------------------------------------------------
# Figure 3 — privacy vs utility: distributional detection & clean false-flags vs eps
# ----------------------------------------------------------------------------
def fig_privacy_utility(df: pd.DataFrame) -> None:
    dist = df[df["mode"] == "distributional"]
    none = df[df["mode"] == "none"]
    xs = list(range(len(EPS_ORDER)))
    xlabels = [_eps_label(e) for e in EPS_ORDER]

    def series(g, col):
        m = g.groupby("epsilon")[col].mean()
        return [m.get(e, np.nan) for e in EPS_ORDER]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True)
    for ax, d in zip(axes, DOMAINS):
        gp = dist[dist["domain"] == d]
        gn = none[none["domain"] == d]
        ax.plot(xs, series(gp, "det_precision"), "-o", color="#0072B2",
                label="precision (degraded)")
        ax.plot(xs, series(gp, "det_recall"), "-s", color="#009E73",
                label="recall (degraded)")
        ax.plot(xs, series(gn, "false_flag_rate"), "--^", color="#D55E00",
                label="false-flag (clean)")
        ax.set_title(d.capitalize(), fontsize=12)
        ax.set_xticks(xs, xlabels)
        ax.set_ylim(-0.03, 1.05)
        ax.grid(ls=":", alpha=0.5)
    axes[0].set_ylabel("rate")
    axes[0].legend(loc="center left", fontsize=8)
    fig.supxlabel("privacy budget ε  (left = more private)", fontsize=10, y=0.02)
    fig.subplots_adjust(bottom=0.18)
    fig.suptitle("Privacy–utility of divergence-based detection: DP noise trades "
                 "precision for recall", fontsize=13, y=1.01)
    fig.text(0.5, -0.10,
             "As ε falls (more privacy), noise inflates every source's cross-source divergence: "
             "recall on degraded sources rises but clean sources get falsely flagged "
             "(precision and clean false-flag rate both worsen).\n"
             "At ε=0.1 clean false-flag ≈ 1.0; the usable knee is around ε≥0.5. "
             "Government precision caps at ~0.56 even at ε=∞.\n"
             "Source: crossdomain_*.csv (mode=distributional and mode=none).",
             ha="center", fontsize=8.5)
    _save(fig, "fig3_privacy_utility")


# ----------------------------------------------------------------------------
# Figure 4 — cross-domain consistency of detection recall (transfer)
# ----------------------------------------------------------------------------
def fig_cross_domain(df: pd.DataFrame) -> None:
    sub = df[(df["mode"] != "none") & np.isinf(df["epsilon"])]
    rec = sub.groupby(["mode", "domain"])["det_recall"].mean().unstack().reindex(
        index=MODES, columns=DOMAINS)
    spread = rec.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(MODES))
    w = 0.25
    for k, d in enumerate(DOMAINS):
        ax.bar(x + (k - 1) * w, rec[d].values, w, label=d.capitalize(),
               color=DOM_COLOR[d])
    for i, m in enumerate(MODES):
        ax.text(i, 1.04, f"σ={spread[m]:.2f}", ha="center", fontsize=9,
                color="#444")
    ax.set_xticks(x, [MODE_LABEL[m] for m in MODES])
    ax.set_ylabel("detection recall @ ε=∞")
    ax.set_ylim(0, 1.15)
    ax.axhline(1.0, ls=":", color="grey", alpha=0.6)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Cross-domain transfer of the gate's detectors "
                 "(σ = spread across domains)", fontsize=12)
    fig.subplots_adjust(bottom=0.24)
    fig.text(0.5, -0.12,
             "Structural detection transfers perfectly (σ=0.00). Completeness is close "
             "(σ=0.19). Distributional (σ=0.34) and especially the label-rate detector "
             "(σ=0.39) do NOT transfer:\n"
             "label-rate works on rare-label domains (healthcare, government ≈ 0.67) but is "
             "blind on balanced-label finance (0.00). Source: crossdomain_*.csv, det_recall @ ε=∞.",
             ha="center", fontsize=8.5)
    _save(fig, "fig4_cross_domain")


# ----------------------------------------------------------------------------
# Figure 5 — downstream impact of gating (auc_delta), mode x domain
# ----------------------------------------------------------------------------
def fig_downstream(df: pd.DataFrame) -> None:
    sub = df[(df["mode"] != "none") & np.isinf(df["epsilon"])]
    delta = sub.groupby(["mode", "domain"])["auc_delta"].mean().unstack().reindex(
        index=MODES, columns=DOMAINS)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(MODES))
    w = 0.25
    for k, d in enumerate(DOMAINS):
        ax.bar(x + (k - 1) * w, delta[d].values * 1000, w, label=d.capitalize(),
               color=DOM_COLOR[d])
    ax.set_xticks(x, [MODE_LABEL[m] for m in MODES])
    ax.set_ylabel("downstream AUC change ×10⁻³  (gate − no-gate)")
    ax.axhline(0, color="black", lw=0.8)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Downstream effect of gating on the clean holdout (ε=∞)",
                 fontsize=12)
    fig.subplots_adjust(bottom=0.24)
    fig.text(0.5, -0.12,
             "AUC is reported ×10⁻³. Gating changes the downstream model almost "
             "not at all here (|ΔAUC| < 0.005 in every cell): removing 2 of 6 sources leaves "
             "a similar model,\n"
             "and the clean holdout is unaffected by the degradation. An honest negative: the gate's "
             "value is governance/withholding, not a downstream accuracy lift in this setup.\n"
             "Source: crossdomain_*.csv, auc_delta @ ε=∞.",
             ha="center", fontsize=8.5)
    _save(fig, "fig5_downstream")


# ----------------------------------------------------------------------------
# Figure 6 — THE SPINE: detection rate vs gate-action rate, and the gap
# ----------------------------------------------------------------------------
def fig_detection_action_gap(df: pd.DataFrame) -> None:
    """Composite readiness scoring masks single-dimension defects: a source is
    detected far more often than it is withheld. Aggregated over all 3 domains
    at eps=inf (no DP noise), so the gap is a property of the scoring rule, not
    of privacy noise."""
    sub = df[(df["mode"] != "none") & np.isinf(df["epsilon"])].copy()
    sub["withheld_frac"] = sub["degraded_withheld"] / N_DEGRADED

    # per-mode + overall means with a seed-level bootstrap 95% CI (whiskers)
    d, a, d_err, a_err = [], [], [[], []], [[], []]
    groups = [(m, sub[sub["mode"] == m]) for m in MODES] + [("ALL", sub)]
    for _, g in groups:
        dm, dlo, dhi = bootstrap_ci(per_seed_means(g, "det_recall"))
        am, alo, ahi = bootstrap_ci(per_seed_means(g, "withheld_frac"))
        d.append(dm); a.append(am)
        d_err[0].append(dm - dlo); d_err[1].append(dhi - dm)
        a_err[0].append(am - alo); a_err[1].append(ahi - am)
    det_all, act_all = d[-1], a[-1]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(MODES) + 1)
    labels = [MODE_SHORT[m] for m in MODES] + ["ALL\n(mean)"]
    w = 0.38
    ax.bar(x - w / 2, d, w, yerr=d_err, capsize=3, ecolor="#333",
           error_kw={"lw": 1}, label="detected (a check fired)", color="#0072B2")
    ax.bar(x + w / 2, a, w, yerr=a_err, capsize=3, ecolor="#333",
           error_kw={"lw": 1}, label="withheld (gate excluded it)", color="#E69F00")
    for i in range(len(x)):
        gap = d[i] - a[i]
        ax.annotate("", xy=(x[i] + w / 2, a[i]), xytext=(x[i] + w / 2, d[i]),
                    arrowprops=dict(arrowstyle="<->", color="#B00020", lw=1.3))
        ax.text(x[i], max(d[i], a[i]) + 0.03, f"gap {gap:.2f}",
                ha="center", fontsize=8.5, color="#B00020", fontweight="bold")
    ax.axvline(len(MODES) - 0.5, ls=":", color="grey", alpha=0.6)
    ax.set_xticks(x, labels)
    ax.set_ylabel("fraction of degraded sources")
    ax.set_ylim(0, 1.18)
    ax.set_title("The spine: composite readiness scoring detects far more than it "
                 "withholds", fontsize=12.5)
    ax.legend(loc="upper center", ncol=2, fontsize=9)
    fig.subplots_adjust(bottom=0.2)
    # (provenance/annotation moved to the paper's Reproducibility paragraph)
    _save(fig, "fig6_detection_action_gap")


# ----------------------------------------------------------------------------
# Metrics summary CSV — one row per (domain, mode, epsilon)
# ----------------------------------------------------------------------------
def write_summary(df: pd.DataFrame) -> Path:
    deg = df[df["mode"] != "none"].copy()
    deg["withheld_frac"] = deg["degraded_withheld"] / N_DEGRADED
    g = deg.groupby(["domain", "mode", "epsilon"])
    summ = g.agg(
        n_cells=("seed", "count"),
        det_precision_mean=("det_precision", "mean"),
        det_recall_mean=("det_recall", "mean"),
        withheld_frac_mean=("withheld_frac", "mean"),
        auc_delta_mean=("auc_delta", "mean"),
    ).reset_index()

    # seed-level bootstrap 95% CIs for the two headline rates
    ci_rows = []
    for (dom, mode, eps), grp in g:
        _, rlo, rhi = bootstrap_ci(per_seed_means(grp, "det_recall"))
        _, wlo, whi = bootstrap_ci(per_seed_means(grp, "withheld_frac"))
        ci_rows.append({"domain": dom, "mode": mode, "epsilon": eps,
                        "det_recall_ci_lo": round(rlo, 4), "det_recall_ci_hi": round(rhi, 4),
                        "withheld_frac_ci_lo": round(wlo, 4), "withheld_frac_ci_hi": round(whi, 4)})
    summ = summ.merge(pd.DataFrame(ci_rows), on=["domain", "mode", "epsilon"], how="left")

    # clean-data false-flag rate (mode=none) joined per (domain, epsilon)
    none = df[df["mode"] == "none"].groupby(["domain", "epsilon"])[
        "false_flag_rate"].mean().reset_index().rename(
        columns={"false_flag_rate": "clean_false_flag_rate"})
    summ = summ.merge(none, on=["domain", "epsilon"], how="left")

    for c in ["det_precision_mean", "det_recall_mean", "withheld_frac_mean",
              "auc_delta_mean", "clean_false_flag_rate"]:
        summ[c] = summ[c].round(4)
    summ["epsilon"] = summ["epsilon"].map(lambda e: "inf" if np.isinf(e) else f"{e:g}")
    summ = summ.sort_values(["domain", "mode", "epsilon"]).reset_index(drop=True)

    out = RESULTS / "metrics_summary.csv"
    header = [
        "# Pillar C metrics summary — aggregated from experiments/results/crossdomain_*.csv",
        f"# Reproduce: .venv/Scripts/python.exe experiments/analyze.py  (seed={SEED}, git={git_rev()})",
        "# one row per (domain, mode, epsilon); means over the 5 seeds x 3 severities in each grid cell",
        "# det_* = per-mode detection over the 2 degraded sources; withheld_frac = degraded_withheld/2 "
        "(gate ACTION); clean_false_flag_rate from mode=none rows; auc_delta = downstream (gate - no-gate)",
        "# *_ci_lo/hi = seed-level percentile bootstrap 95% CI (10k resamples, seed=42) for det_recall "
        "and withheld_frac",
    ]
    with open(out, "w", encoding="utf-8", newline="") as f:
        for line in header:
            f.write(line + "\n")
        summ.to_csv(f, index=False)
    print(f"  wrote {out.relative_to(ROOT)}  ({len(summ)} rows)")
    return out


def main() -> int:
    np.random.seed(SEED)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 120, "savefig.dpi": 200,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    df = load_all()
    print(f"loaded {len(df)} rows from {len(DOMAINS)} domain CSVs "
          f"(git={git_rev()}, seed={SEED})")
    fig_detection_by_mode(df)
    fig_detect_vs_withhold(df)
    fig_privacy_utility(df)
    fig_cross_domain(df)
    fig_downstream(df)
    fig_detection_action_gap(df)
    write_summary(df)
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
