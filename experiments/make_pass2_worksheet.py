#!/usr/bin/env python
"""Generate a blind, reshuffled PASS-2 worksheet for the intra-rater reliability pass.

Reads the pass-1 worksheet (docs/taxonomy_sample_for_human_review.md), strips every
filled-in label, reshuffles the 30 incidents with a fixed seed, and writes a fresh
blank worksheet (docs/taxonomy_sample_pass2_blind.md).

The coder fills this AFTER a washout period, WITHOUT looking at pass 1 or the
machine key. Transcribing the result to
    experiments/results/taxonomy_sample_human_labels_pass2.csv
and running `experiments/code_agreement.py --intra` yields the intra-rater kappa
(coding stability), reported separately from the inter-rater kappa.

No labels are copied — the mode/detectability fields are emitted blank. Nothing is
fabricated; the human does the pass-2 coding.

Reproduce (fixed seed in the header of the output file):
    .venv/Scripts/python.exe experiments/make_pass2_worksheet.py --seed 202
"""
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "taxonomy_sample_for_human_review.md"
OUT = ROOT / "docs" / "taxonomy_sample_pass2_blind.md"
PASS2_CSV = ROOT / "experiments" / "results" / "taxonomy_sample_human_labels_pass2.csv"

MODES = {"structural_schema", "completeness", "distributional", "label",
         "provenance_documentation", "cross_source", "other"}
DETECT = {"yes", "partial", "no"}


def parse_blocks(text: str) -> list[tuple[str, str, str]]:
    """Return [(incident_id, title, description)] from the pass-1 worksheet."""
    blocks: list[tuple[str, str, str]] = []
    for part in re.split(r"\n## Incident ", text)[1:]:
        iid = re.match(r"(\d+)", part).group(1)
        tmatch = re.search(r"\*\*Title:\*\*\s*(.+)", part)
        title = tmatch.group(1).strip()
        desc = part[tmatch.end():].split("- Your mode:")[0].strip()
        blocks.append((iid, title, desc))
    return blocks


def transcribe() -> int:
    """Parse the FILLED blind worksheet -> pass-2 CSV (incident_id,mode,detectability)."""
    if not OUT.exists():
        print(f"ERROR: {OUT} not found. Generate it first (run without --transcribe).")
        return 1
    rows, blanks, bad = [], [], []
    for part in re.split(r"\n## Incident ", OUT.read_text(encoding="utf-8"))[1:]:
        iid = re.match(r"(\d+)", part).group(1)
        mode = (re.search(r"- Your mode:\s*`([^`]*)`", part) or [None, ""])[1].strip().lower()
        det = (re.search(r"- Your detectability:\s*`([^`]*)`", part) or [None, ""])[1].strip().lower()
        if not mode or not det:
            blanks.append(iid)
        if mode and mode not in MODES:
            bad.append(f"{iid}: mode='{mode}'")
        if det and det not in DETECT:
            bad.append(f"{iid}: detectability='{det}'")
        rows.append((iid, mode, det))
    if blanks:
        print(f"ERROR: {len(blanks)} incident(s) not coded yet: {blanks}")
        return 1
    if bad:
        print("ERROR: invalid label(s):\n  " + "\n  ".join(bad))
        print(f"  modes must be one of {sorted(MODES)}")
        print(f"  detectability must be one of {sorted(DETECT)}")
        return 1
    PASS2_CSV.parent.mkdir(parents=True, exist_ok=True)
    lines = ["incident_id,mode,detectability"] + [f"{i},{m},{d}" for i, m, d in rows]
    PASS2_CSV.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {PASS2_CSV} ({len(rows)} rows)")
    print("Next: .venv/Scripts/python.exe experiments/code_agreement.py --intra")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=202,
                    help="fixed shuffle seed (recorded in the output header)")
    ap.add_argument("--transcribe", action="store_true",
                    help="parse the FILLED blind worksheet into the pass-2 CSV")
    args = ap.parse_args()

    if args.transcribe:
        return transcribe()

    blocks = parse_blocks(SRC.read_text(encoding="utf-8"))
    assert len(blocks) == 30, f"expected 30 incidents, parsed {len(blocks)}"

    order = blocks[:]
    random.Random(args.seed).shuffle(order)

    out = [
        "# Taxonomy sample — PASS 2 (blind intra-rater re-code)",
        "",
        f"The same 30 incidents as `docs/taxonomy_sample_for_human_review.md`, "
        f"reshuffled with a fixed seed (seed={args.seed}) and with the pass-1 labels "
        "removed. Code each incident from the text ALONE — do not look at your "
        "pass-1 answers or the machine key.",
        "",
        "This is the INTRA-rater pass: comparing it against pass 1 measures how "
        "consistently you apply the coding scheme over time (coding stability), "
        "separately from the machine-vs-human inter-rater kappa.",
        "",
        "Modes: `structural_schema`, `completeness`, `distributional`, `label`, "
        "`provenance_documentation`, `cross_source`, or `other` (not a "
        "data-readiness failure).",
        "Detectability (can a pre-training validation gate catch it?): "
        "`yes` / `partial` / `no`.",
        "",
        "Fill the two blanks per incident:",
        "",
        "---",
        "",
    ]
    for iid, title, desc in order:
        out += [
            f"## Incident {iid}",
            f"**Title:** {title}",
            "",
            desc,
            "",
            "- Your mode: ``",
            "- Your detectability: ``",
            "- (optional) note: ",
            "",
            "---",
            "",
        ]

    OUT.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {OUT} ({len(order)} incidents, seed={args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
