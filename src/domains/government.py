"""Government domain (Pillar B) — CMS Open Payments (general payments).

Public, citable dataset: CMS Open Payments
(https://openpaymentsdata.cms.gov/), the annual "General Payments" detail CSV
(OP_DTL_GNRL_PGYR{year}_*.csv). Chosen over NYC TLC because Open Payments ships
CSV (chunk-friendly with pandas) whereas TLC is parquet-only and this
environment has no pyarrow.

Task: predict whether a payment's nature is "Food and Beverage" from its
magnitude/frequency. Food/beverage payments are small and frequent, larger
categories (consulting, travel) are not — a genuinely learnable target with
usable prevalence (~0.9). The natural dispute flag was dropped as a label: it is
~0.002% positive in the real file (59 in 3M rows), so a head-slice has one
class and no downstream metric is defined.
"""
from __future__ import annotations

import glob
import os
from typing import Iterator

import numpy as np
import pandas as pd

from .base import Domain, register, aggregate_summary

FEATURES = ["log_amount", "log_per_payment", "n_payments"]
LABEL = "is_food_beverage"
REQUIRED = {
    "log_amount": "float", "log_per_payment": "float",
    # n_payments is a count but kept float: real Open Payments rows can carry
    # blanks (-> NaN -> float on load), and a float column matches both the
    # synthetic and the real extract without a spurious dtype failure.
    "n_payments": "float", "is_food_beverage": "int",
}
HIST_RANGES = {
    "log_amount": (0.0, 7.0),        # log10 USD; $1 .. $10M
    "log_per_payment": (0.0, 7.0),
    "n_payments": (1.0, 50.0),
}
# Real CSV column names (long) -> our short names.
COLMAP = {
    "Total_Amount_of_Payment_USDollars": "amount",
    "Number_of_Payments_Included_in_Total_Amount": "n_payments",
    "Nature_of_Payment_or_Transfer_of_Value": "nature",
}


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    amount = pd.to_numeric(df["amount"], errors="coerce")
    n_pay = pd.to_numeric(df["n_payments"], errors="coerce").clip(lower=1)
    nature = df["nature"].astype(str).str.strip().str.lower()
    out = pd.DataFrame({
        "log_amount": np.log10(amount.clip(lower=0) + 1.0),
        "log_per_payment": np.log10((amount.clip(lower=0) / n_pay) + 1.0),
        "n_payments": n_pay.astype("float"),
        "is_food_beverage": nature.str.startswith("food and bev").astype(int),
    })
    return out.dropna(subset=["log_amount", "log_per_payment", "n_payments"]).reset_index(drop=True)


def synthetic(n_rows: int, seed: int) -> pd.DataFrame:
    """Synthetic Open-Payments-like records."""
    rng = np.random.default_rng(seed)
    amount = rng.lognormal(mean=4.0, sigma=1.8, size=n_rows)          # ~ $50 median
    n_payments = 1 + rng.poisson(1.5, n_rows)
    per_payment = amount / n_payments
    log_amount = np.log10(amount + 1.0)
    log_per = np.log10(per_payment + 1.0)
    # Food/beverage is the majority class (~0.85), and is the SMALL, frequent
    # payments — so a low amount predicts it. Matches the real ~0.9 prevalence
    # and gives the downstream model genuine, non-circular signal.
    logit = 2.4 - 1.3 * (log_amount - 1.7)
    is_fb = (rng.random(n_rows) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "log_amount": log_amount, "log_per_payment": log_per,
        "n_payments": n_payments.astype(float), "is_food_beverage": is_fb,
    })


def _resolve_csv(data_root: str) -> str:
    if os.path.isfile(data_root):
        return data_root
    hits = sorted(glob.glob(os.path.join(data_root, "**", "OP_DTL_GNRL_*.csv"), recursive=True))
    if not hits:
        hits = sorted(glob.glob(os.path.join(data_root, "*.csv")))
    if not hits:
        raise RuntimeError(f"government: no Open Payments CSV found under {data_root}")
    return hits[0]


def load(data_root: str, chunksize: int) -> Iterator[pd.DataFrame]:
    path = _resolve_csv(data_root)
    src_cols = list(COLMAP)
    for chunk in pd.read_csv(path, usecols=lambda c: c in src_cols,
                             chunksize=chunksize, low_memory=False):
        chunk = chunk.rename(columns=COLMAP)
        for needed in ("amount", "n_payments", "dispute"):
            if needed not in chunk.columns:
                chunk[needed] = np.nan
        yield _build_features(chunk)


DOMAIN = register(Domain(
    name="government",
    label_column=LABEL,
    feature_columns=FEATURES,
    required_columns=REQUIRED,
    histogram_ranges=HIST_RANGES,
    synthetic=synthetic,
    loader=load,
    min_rows=50,
    divergence_threshold=0.045,  # calibrated on clean sources at epsilon=inf
    notes="CMS Open Payments general payments; disputed = Dispute_Status=='Yes'.",
))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Government loader — aggregates only, no rows printed.")
    ap.add_argument("--data-root", default=None, help="Open Payments CSV or dir; omit for synthetic.")
    ap.add_argument("--n-rows", type=int, default=4000)
    ap.add_argument("--max-rows", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    df, prov = DOMAIN.get_frame(a.n_rows, a.seed, a.data_root, a.max_rows)
    print(f"provenance={prov}")
    print(aggregate_summary(df, DOMAIN))
