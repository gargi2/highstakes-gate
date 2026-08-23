"""gate — a domain-agnostic pre-training data-readiness validation gate.

Reused from FedCheck (the validated single-domain instantiation) and
generalized: a "source" is any contributing dataset / data provider. The gate
sees only DP-noised metadata reports, never rows.

Modules:
  expectations   — local expectation suite + DP-noised report emission
  reports        — the metadata report exchanged with the gate (no rows)
  privacy        — Laplace DP mechanisms (highest verification priority)
  compatibility  — cross-source schema compatibility + JS divergence
  consensus      — cross-source consensus detectors (label-rate, divergence)
  readiness      — composite readiness score + READY/FIXABLE/EXCLUDE decision
  partition      — synthetic fixtures + domain-agnostic degradation injection
"""
