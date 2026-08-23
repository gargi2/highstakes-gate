#!/usr/bin/env python
"""Pillar A — data-readiness failure-mode coding of the AI Incident Database.

Transparent, reproducible, keyword + expert-label-assisted qualitative coding.
For each AIID incident *plausibly rooted in a pre-training data-readiness
failure*, emit one row: incident_id, short title, assigned mode (one of the six
data-readiness modes from docs/PIVOT_SCOPE.md §2, or "other"), pre-training
detectability (yes/partial/no), a confidence flag, and a one-line rationale.

This is QUALITATIVE coding executed by a rule. It is deliberately conservative
(precision over recall on the "is this data-readiness?" filter) and every coding
carries a confidence flag so low-confidence rows can be down-weighted. A seeded
30-incident sample is emitted for independent human validation
(docs/taxonomy_sample_for_human_review.md); inter-rater agreement is computed by
experiments/code_agreement.py once the human labels are returned.

Signals used (all transparent):
  1. Keyword lexicons over lowercased title + description. Each mode has a
     STRONG list (weight 1.0) and a WEAK list (weight 0.5).
  2. Expert labels from AIID's own GMF taxonomy (classifications_GMF.csv,
     "Known/Potential AI Technical Failure"), mapped to our modes (weight 2.0).
     These are AIID's labels, not ours — we only re-project the data-related
     subset onto the pre-training-detectability axis.

An incident is INCLUDED (coded as data-readiness) iff some mode scores >= 1.0.
The assigned mode is the argmax score; ties break by MODE_PRIORITY.

We do NOT claim novelty over AIID's taxonomies. The contribution is the
pre-training-detectability lens (does a validation gate catch it BEFORE
training?), not the incident catalogue.

Reproduce:
    .venv/Scripts/python.exe experiments/code_incidents.py

Inputs  : data/aiid/snapshot_csv/{incidents,classifications_GMF}.csv
Outputs : experiments/results/taxonomy_coding.csv           (all coded incidents)
          experiments/results/taxonomy_sample_machine_labels.csv  (blind-review key)
          docs/taxonomy_sample_for_human_review.md          (blind human-coding form)

Seed: 42 (used only for the human-review sample draw; coding itself is
deterministic).
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

SEED = 42
SAMPLE_N = 30
TITLE_MAXLEN = 90

ROOT = Path(__file__).resolve().parents[1]
AIID = ROOT / "data" / "aiid" / "snapshot_csv"
RESULTS = ROOT / "experiments" / "results"
DOCS = ROOT / "docs"

# The six data-readiness modes (docs/PIVOT_SCOPE.md §2). Order = tie-break
# priority: more structurally specific / higher-precision modes win ties.
MODES = [
    "structural_schema",
    "label",
    "provenance_documentation",
    "cross_source",
    "completeness",
    "distributional",
]
MODE_PRIORITY = {m: i for i, m in enumerate(MODES)}

# ---------------------------------------------------------------------------
# Keyword lexicons. STRONG = 1.0, WEAK = 0.5. Kept deliberately specific; every
# term is auditable here. Matched as word-ish substrings on lowercased text.
# ---------------------------------------------------------------------------
STRONG: dict[str, list[str]] = {
    "structural_schema": [
        "schema", "wrong column", "column mismatch", "data type", "dtype",
        "encoding error", "malformed", "parsing error", "wrong format",
        "unit mismatch", "field mismatch", "corrupt data", "corrupted data",
        "preprocessing error", "faulty preprocessing", "date format",
    ],
    "completeness": [
        "missing data", "incomplete data", "missing field", "missing value",
        "null value", "coverage gap", "missing records", "sparse data",
        "not captured", "omitted data", "gaps in the data", "insufficient data",
        "lack of data", "no data on", "not enough data",
    ],
    "distributional": [
        "underrepresent", "under-represent", "over-represent", "overrepresent",
        "imbalanced", "class imbalance", "dataset imbalance", "sampling bias",
        "non-representative", "not representative", "unrepresentative",
        "skin tone", "darker skin", "lighter skin", "predominantly white",
        "predominantly male", "mostly white", "mostly men", "trained mostly on",
        "demographic skew", "population skew", "skewed data", "distribution shift",
        "concept drift", "covariate shift", "training data was biased",
        "biased training data", "biased dataset", "biased data set",
        "unbalanced dataset", "unbalanced data",
    ],
    "label": [
        "mislabel", "mislabeled", "mislabelled", "label noise", "labelling noise",
        "labeling noise", "annotation error", "incorrect label", "wrong label",
        "labeling error", "labelling error", "ground truth error", "noisy label",
        "data leakage", "label leakage", "target leakage", "mislabeling",
    ],
    "provenance_documentation": [
        "unauthorized data", "scraped without", "scraped from", "unknown source",
        "undocumented", "no datasheet", "data provenance", "unknown provenance",
        "unclear provenance", "without consent", "copyrighted data",
        "without permission", "data lineage", "unlicensed data",
        "unvetted data", "unverified source", "unknown origin",
    ],
    "cross_source": [
        "different hospitals", "different sites", "multiple sources",
        "cross-site", "multi-site", "merging datasets", "merged datasets",
        "combining datasets", "incompatible data", "inconsistent across sources",
        "different institutions", "harmonize", "harmonization",
        "semantic mismatch", "different coding systems",
    ],
}
WEAK: dict[str, list[str]] = {
    "structural_schema": ["glitch", "software bug", "formatting", "misconfigured"],
    "completeness": ["incomplete", "missing", "sparse", "gap in"],
    "distributional": [
        "bias", "biased", "imbalance", "skew", "underfitting", "overfitting",
        "trained on", "training data", "demographic", "dialect", "accent",
        "race", "racial", "gender", "ethnic",
    ],
    "label": ["label", "annotation", "annotated", "ground truth"],
    "provenance_documentation": ["scraped", "consent", "provenance", "sourced from"],
    "cross_source": ["multiple hospitals", "across sites", "data sources"],
}

# ---------------------------------------------------------------------------
# AIID GMF expert-label -> our mode. Only the data-related subset is mapped;
# non-data failure labels (Misuse, Software Bug, Generalization Failure,
# Underspecification, Lack of Transparency, ...) are intentionally NOT mapped,
# so they contribute no data-readiness signal.
# ---------------------------------------------------------------------------
GMF_LABEL_TO_MODE: dict[str, str] = {
    "Distributional Bias": "distributional",
    "Dataset Imbalance": "distributional",
    "Inadequate Data Sampling": "distributional",
    "Concept Drift": "distributional",
    "Incomplete Data Attribute Capture": "completeness",
    "Data or Labelling Noise": "label",
    "Unauthorized Data": "provenance_documentation",
    "Faulty or Inadequate Preprocessing": "structural_schema",
}

# Per-mode default pre-training detectability by the FedCheck gate + a refinement
# rule for the distributional mode (static imbalance is measurable pre-training;
# temporal drift is not). Documented in docs/TAXONOMY.md.
DETECTABILITY = {
    "structural_schema": "yes",
    "completeness": "yes",
    "cross_source": "yes",
    "distributional": "partial",  # refined below
    "label": "partial",
    "provenance_documentation": "partial",
    "other": "no",
}
# distributional refinement: drift/shift terms -> partial; static-imbalance terms -> yes
DRIFT_TERMS = ["drift", "shift", "changed over time", "over time", "seasonal"]
STATIC_IMBALANCE_TERMS = [
    "imbalance", "underrepresent", "under-represent", "over-represent",
    "overrepresent", "representative", "skin tone", "predominantly",
    "mostly white", "mostly men", "dataset imbalance", "unbalanced",
    "trained mostly on", "biased dataset", "biased training data",
]


def read_results_csv(path: Path) -> pd.DataFrame:
    """Read a results CSV that carries a leading ``#`` reproduction-header block.

    Do NOT use ``pd.read_csv(path, comment="#")`` on these files: incident titles
    contain ``#`` (hashtags, e.g. "#LekkiMassacre"), and ``comment=`` strips from
    the first ``#`` anywhere on a line, silently corrupting those rows. This skips
    only the contiguous leading ``#`` lines.
    """
    with open(path, encoding="utf-8") as f:
        skip = 0
        for line in f:
            if line.startswith("#"):
                skip += 1
            else:
                break
    return pd.read_csv(path, skiprows=skip)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower())


def _hits(text: str, terms: list[str]) -> list[str]:
    return [t for t in terms if t in text]


def load_gmf_signals() -> dict[int, dict[str, list[str]]]:
    """incident_id -> {mode: [gmf labels that fired]} from GMF Technical Failure."""
    gmf = pd.read_csv(AIID / "classifications_GMF.csv")
    id_col = "Incident ID"
    cols = ["Known AI Technical Failure", "Potential AI Technical Failure"]
    out: dict[int, dict[str, list[str]]] = {}
    for _, row in gmf.iterrows():
        try:
            iid = int(row[id_col])
        except (ValueError, TypeError):
            continue
        fired: dict[str, list[str]] = {}
        for col in cols:
            val = row.get(col)
            if pd.isna(val):
                continue
            # cells are comma-joined label lists
            for label in str(val).split(","):
                label = label.strip()
                mode = GMF_LABEL_TO_MODE.get(label)
                if mode:
                    fired.setdefault(mode, []).append(label)
        if fired:
            # dedupe labels within a mode (Known/Potential may repeat a label)
            out[iid] = {m: sorted(set(lbls)) for m, lbls in fired.items()}
    return out


def code_incident(text: str, gmf: dict[str, list[str]]) -> dict:
    """Score every mode; return assignment or None if below inclusion threshold."""
    scores: dict[str, float] = {m: 0.0 for m in MODES}
    fired: dict[str, list[str]] = {m: [] for m in MODES}

    for mode in MODES:
        for kw in _hits(text, STRONG[mode]):
            scores[mode] += 1.0
            fired[mode].append(f"kw:{kw}")
        for kw in _hits(text, WEAK[mode]):
            scores[mode] += 0.5
            fired[mode].append(f"weak:{kw}")
    for mode, labels in gmf.items():
        scores[mode] += 2.0 * len(labels)
        fired[mode].extend(f"gmf:{lb}" for lb in labels)

    best = max(MODES, key=lambda m: (scores[m], -MODE_PRIORITY[m]))
    if scores[best] < 1.0:
        return {"included": False}

    # confidence
    has_gmf = any(s.startswith("gmf:") for s in fired[best])
    n_strong = sum(1 for s in fired[best] if s.startswith("kw:"))
    if has_gmf or n_strong >= 2:
        confidence = "high"
    elif n_strong == 1:
        confidence = "med"
    else:  # only weak keywords summed to >= 1.0
        confidence = "low"

    # detectability
    detect = DETECTABILITY[best]
    if best == "distributional":
        if any(t in text for t in STATIC_IMBALANCE_TERMS):
            detect = "yes"
        elif any(t in text for t in DRIFT_TERMS):
            detect = "partial"

    return {
        "included": True,
        "mode": best,
        "detectability": detect,
        "confidence": confidence,
        "signals": "; ".join(fired[best]),
        "score": round(scores[best], 2),
    }


def one_line_rationale(mode: str, detect: str, signals: str) -> str:
    sig = signals.replace("kw:", "").replace("weak:", "~").replace("gmf:", "GMF=")
    return (
        f"{mode.replace('_', '/')}; pre-training detectable={detect}; "
        f"signals: {sig}"
    )


def git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    inc = pd.read_csv(AIID / "incidents.csv")
    gmf_signals = load_gmf_signals()

    rows = []
    for _, r in inc.iterrows():
        iid = int(r["incident_id"])
        title = str(r["title"])
        text = _norm(f"{title} {r['description']}")
        res = code_incident(text, gmf_signals.get(iid, {}))
        if not res["included"]:
            continue
        short = title if len(title) <= TITLE_MAXLEN else title[: TITLE_MAXLEN - 1] + "…"
        rows.append(
            {
                "incident_id": iid,
                "title": short,
                "mode": res["mode"],
                "detectability": res["detectability"],
                "confidence": res["confidence"],
                "rationale": one_line_rationale(res["mode"], res["detectability"], res["signals"]),
                "score": res["score"],
                "signals": res["signals"],
            }
        )

    coded = pd.DataFrame(rows).sort_values(["mode", "incident_id"]).reset_index(drop=True)

    header = [
        "# Pillar A: AIID data-readiness failure-mode coding",
        f"# Reproduce: .venv/Scripts/python.exe experiments/code_incidents.py",
        f"# git={git_rev()} seed={SEED} inputs=data/aiid/snapshot_csv/{{incidents,classifications_GMF}}.csv",
        f"# corpus=1632 incidents; coded (data-readiness)={len(coded)}",
        "# modes per docs/PIVOT_SCOPE.md sec2; scoring per experiments/code_incidents.py docstring",
        "# READ WITH skiprows (see read_results_csv); titles contain '#', so comment='#' corrupts rows",
        "# columns: incident_id,title,mode,detectability,confidence,rationale,score,signals",
    ]
    out_csv = RESULTS / "taxonomy_coding.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        for line in header:
            f.write(line + "\n")
        coded.to_csv(f, index=False)
    print(f"wrote {out_csv}  ({len(coded)} coded incidents)")
    print(coded["mode"].value_counts().to_string())
    print("\nconfidence:")
    print(coded["confidence"].value_counts().to_string())
    print("\ndetectability:")
    print(coded["detectability"].value_counts().to_string())

    # ---- seeded blind-review sample -------------------------------------
    sample = coded.sample(n=min(SAMPLE_N, len(coded)), random_state=SEED).sort_values(
        "incident_id"
    )
    # machine-label key (kept out of the human form so review is blind)
    key = sample[["incident_id", "title", "mode", "detectability", "confidence"]]
    key.to_csv(RESULTS / "taxonomy_sample_machine_labels.csv", index=False)

    # blind human-coding form: id + title + description, blank fields to fill
    desc_by_id = dict(zip(inc["incident_id"].astype(int), inc["description"]))
    lines = [
        "# Taxonomy sample for human review (blind)",
        "",
        f"Seeded random sample of {len(sample)} incidents coded as data-readiness",
        "failures by `experiments/code_incidents.py` "
        f"(seed={SEED}, git={git_rev()}).",
        "",
        "**Instructions.** For each incident, without looking at the machine key,",
        "assign ONE mode and a pre-training detectability. This gives an",
        "independent second coding for inter-rater reliability",
        "(`experiments/code_agreement.py`).",
        "",
        "Modes: `structural_schema`, `completeness`, `distributional`, `label`,",
        "`provenance_documentation`, `cross_source`, or `other` (not a",
        "data-readiness failure — this also validates the filter's precision).",
        "",
        "Detectability (can a pre-training validation gate catch it?):",
        "`yes` / `partial` / `no`.",
        "",
        "Fill the two blanks per incident:",
        "",
        "---",
        "",
    ]
    for _, r in sample.iterrows():
        iid = int(r["incident_id"])
        lines += [
            f"## Incident {iid}",
            f"**Title:** {inc.loc[inc['incident_id'] == iid, 'title'].iloc[0]}",
            "",
            f"{str(desc_by_id.get(iid, '')).strip()}",
            "",
            "- Your mode: `________`",
            "- Your detectability: `________`",
            "- (optional) note: ",
            "",
            "---",
            "",
        ]
    (DOCS / "taxonomy_sample_for_human_review.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"\nwrote docs/taxonomy_sample_for_human_review.md ({len(sample)} incidents)")
    print("wrote experiments/results/taxonomy_sample_machine_labels.csv (blind key)")


if __name__ == "__main__":
    sys.exit(main())
