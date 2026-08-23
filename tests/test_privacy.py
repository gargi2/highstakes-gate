"""Verification tests for privacy.py — hand-computable, not AI-trusted.

These tests verify the STATISTICAL PROPERTIES of the noise, which is the
part AI-generated code most often gets subtly wrong (wrong scale, wrong
sensitivity, epsilon applied per-release vs per-bin).

Run: pytest tests/test_privacy.py -v
Then cross-check one value against diffprivlib (see test_cross_check note).
"""

import numpy as np
import pytest

from gate.privacy import (
    PrivacyAccountant,
    laplace_noise,
    noised_count,
    noised_histogram,
)


def test_laplace_scale_is_sensitivity_over_epsilon():
    """Empirical std of Laplace(0,b) must be b*sqrt(2).

    Hand computation: sensitivity=1, epsilon=0.5 -> b = 1/0.5 = 2.0
    -> std = 2.0 * sqrt(2) ≈ 2.8284. With 200k samples, tolerance 2%.
    """
    rng = np.random.default_rng(42)
    samples = np.array([laplace_noise(1.0, 0.5, rng) for _ in range(200_000)])
    expected_std = 2.0 * np.sqrt(2.0)
    assert abs(samples.std() - expected_std) / expected_std < 0.02
    assert abs(samples.mean()) < 0.05  # centered at 0


def test_noised_count_unbiased():
    """Mean of many noised releases of count=1000 must be ~1000."""
    rng = np.random.default_rng(7)
    vals = [noised_count(1000, epsilon=1.0, rng=rng) for _ in range(100_000)]
    assert abs(np.mean(vals) - 1000) < 0.5


def test_histogram_noise_scale_is_one_over_epsilon_not_bins_over_epsilon():
    """THE classic mistake: splitting epsilon across bins when sensitivity=1
    already covers the whole histogram. Each bin's noise must have scale
    1/epsilon regardless of bin count.

    epsilon=1.0 -> per-bin Laplace scale 1.0 -> std = sqrt(2) ≈ 1.4142.
    If someone 'fixes' the code to scale = n_bins/epsilon, this test fails.
    Note: clipping at 0 truncates the noise distribution, so we test on
    large counts where clipping never triggers.
    """
    rng = np.random.default_rng(3)
    true = np.full(10, 10_000.0)  # large so clipping never binds
    releases = np.stack([noised_histogram(true, 1.0, rng) for _ in range(50_000)])
    per_bin_std = (releases - true).std(axis=0)
    expected = np.sqrt(2.0)
    assert np.all(np.abs(per_bin_std - expected) / expected < 0.03)


def test_accountant_blocks_overspend():
    acc = PrivacyAccountant(epsilon_budget=1.0)
    acc.spend("row_count", 0.5)
    acc.spend("histogram_age", 0.5)
    with pytest.raises(RuntimeError):
        acc.spend("extra", 0.01)


def test_epsilon_zero_rejected():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        laplace_noise(1.0, 0.0, rng)


# MANUAL CROSS-CHECK (do once, log in verification/VERIFICATION.md):
#   pip install diffprivlib
#   from diffprivlib.mechanisms import Laplace
#   m = Laplace(epsilon=0.5, sensitivity=1.0)
#   samples = [m.randomise(0.0) for _ in range(200_000)]
#   -> std should also be ≈ 2.8284, matching test_laplace_scale above.
