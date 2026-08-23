"""Finance domain (Pillar B) — SEC EDGAR Financial Statement Data Sets.

Public, citable dataset: SEC "Financial Statement Data Sets"
(https://www.sec.gov/dera/data/financial-statement-data-sets), quarterly ZIPs
containing a tab-delimited `num.txt` (one row per reported numeric fact) and
`sub.txt` (submissions). The real loader streams `num.txt`, keeps a handful of
XBRL tags, and pivots to one row per filing (adsh).

Task: adverse-earnings flag (NetIncomeLoss < 0) from balance-sheet magnitudes.
Features are log10 magnitudes so histogram ranges are stable across filers.
"""
from __future__ import annotations

import glob
import os
from typing import Iterator

import numpy as np
import pandas as pd

from .base import Domain, register, aggregate_summary

FEATURES = ["log_assets", "log_liabilities", "log_revenues", "log_equity"]
LABEL = "adverse"
REQUIRED = {
    "log_assets": "float", "log_liabilities": "float",
    "log_revenues": "float", "log_equity": "float", "adverse": "int",
}
HIST_RANGES = {  # log10 dollars; most filers sit in ~1e5..1e12
    "log_assets": (3.0, 13.0), "log_liabilities": (3.0, 13.0),
    "log_revenues": (3.0, 13.0), "log_equity": (2.0, 13.0),
}
# XBRL tags we pull from num.txt (us-gaap namespace).
TAGS = {
    "Assets": "assets", "Liabilities": "liabilities",
    "Revenues": "revenues", "StockholdersEquity": "equity",
    "NetIncomeLoss": "net_income",
}


def _log10m(x: pd.Series) -> pd.Series:
    return np.log10(np.abs(pd.to_numeric(x, errors="coerce")) + 1.0)


def _build_features(pivot: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=pivot.index)
    out["log_assets"] = _log10m(pivot.get("assets"))
    out["log_liabilities"] = _log10m(pivot.get("liabilities"))
    out["log_revenues"] = _log10m(pivot.get("revenues"))
    out["log_equity"] = _log10m(pivot.get("equity"))
    out["adverse"] = (pd.to_numeric(pivot.get("net_income"), errors="coerce") < 0).astype(int)
    return out.dropna(subset=FEATURES).reset_index(drop=True)


def synthetic(n_rows: int, seed: int) -> pd.DataFrame:
    """Synthetic filings with EDGAR-like log-magnitude features."""
    rng = np.random.default_rng(seed)
    log_assets = np.clip(rng.normal(8.0, 1.6, n_rows), 3, 13)
    leverage = np.clip(rng.beta(2, 2, n_rows), 0.02, 0.98)
    log_liabilities = np.clip(log_assets + np.log10(leverage), 3, 13)
    log_equity = np.clip(log_assets + np.log10(np.clip(1 - leverage, 1e-3, None)), 2, 13)
    log_revenues = np.clip(log_assets + rng.normal(-0.2, 0.6, n_rows), 3, 13)
    # adverse earnings more likely with high leverage / low revenue
    logit = -0.5 + 3.0 * (leverage - 0.5) - 0.6 * (log_revenues - log_assets)
    adverse = (rng.random(n_rows) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "log_assets": log_assets, "log_liabilities": log_liabilities,
        "log_revenues": log_revenues, "log_equity": log_equity, "adverse": adverse,
    })


def _resolve_num(data_root: str) -> str:
    if os.path.isfile(data_root):
        return data_root
    hits = sorted(glob.glob(os.path.join(data_root, "**", "num.txt"), recursive=True))
    if not hits:
        raise RuntimeError(f"finance: no num.txt found under {data_root}")
    return hits[0]


def load(data_root: str, chunksize: int) -> Iterator[pd.DataFrame]:
    """Stream num.txt, keep target tags, pivot to one row per filing (adsh)."""
    path = _resolve_num(data_root)
    keep = set(TAGS)
    collected: dict[str, dict[str, float]] = {}
    for chunk in pd.read_csv(path, sep="\t", usecols=["adsh", "tag", "qtrs", "value"],
                             chunksize=chunksize, low_memory=False):
        sub = chunk[chunk["tag"].isin(keep)]
        for adsh, tag, val in zip(sub["adsh"], sub["tag"], sub["value"]):
            collected.setdefault(adsh, {})[TAGS[tag]] = val
    if not collected:
        return
    pivot = pd.DataFrame.from_dict(collected, orient="index")
    yield _build_features(pivot)


DOMAIN = register(Domain(
    name="finance",
    label_column=LABEL,
    feature_columns=FEATURES,
    required_columns=REQUIRED,
    histogram_ranges=HIST_RANGES,
    synthetic=synthetic,
    loader=load,
    divergence_threshold=0.055,  # calibrated on clean sources at epsilon=inf
    notes="SEC EDGAR Financial Statement Data Sets; adverse = NetIncomeLoss<0.",
))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Finance loader — aggregates only, no rows printed.")
    ap.add_argument("--data-root", default=None, help="num.txt file or EDGAR dataset dir; omit for synthetic.")
    ap.add_argument("--n-rows", type=int, default=4000)
    ap.add_argument("--max-rows", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    df, prov = DOMAIN.get_frame(a.n_rows, a.seed, a.data_root, a.max_rows)
    print(f"provenance={prov}")
    print(aggregate_summary(df, DOMAIN))
