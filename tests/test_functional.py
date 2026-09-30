"""Unit tests for SRVF transformation, DP time warping, and Fisher-Rao metrics."""

import numpy as np
from spatiotemporal_atlas.functional import (
    curve_to_srvf,
    align_srvf_dp_clamped,
    calculate_fisher_rao_distance,
)


def test_srvf_shape_and_speed():
    """Verify curve_to_srvf produces correct dimensions and handles constant velocity."""
    time_grid = np.linspace(0.0, 1.0, 50)
    # Linear curve with constant velocity v = [3, 4, 0], speed = 5
    curve = np.outer(time_grid, np.array([3.0, 4.0, 0.0]))
    srvf = curve_to_srvf(curve, time_grid)

    assert srvf.shape == (50, 3)
    # v / sqrt(||v||) = [3, 4, 0] / sqrt(5)
    expected = np.array([3.0, 4.0, 0.0]) / np.sqrt(5.0)
    # Interior gradient matches closely
    assert np.allclose(srvf[5:45], expected, atol=1e-2)


def test_dp_warp_identity():
    """Verify that aligning identical curves yields the identity warping gamma(t) = t."""
    time_grid = np.linspace(0.0, 1.0, 40)
    curve = np.column_stack([
        np.sin(2 * np.pi * time_grid),
        np.cos(2 * np.pi * time_grid),
        time_grid,
    ])
    q = curve_to_srvf(curve, time_grid)

    gamma = align_srvf_dp_clamped(q, q, time_grid)

    assert np.isclose(gamma[0], 0.0, atol=1e-5)
    assert np.isclose(gamma[-1], 1.0, atol=1e-5)
    # Monotonicity check
    assert np.all(np.diff(gamma) >= -1e-6)
    # Closeness to identity
    assert np.allclose(gamma, time_grid, atol=0.05)


def test_fisher_rao_identity_is_zero():
    """Verify that Fisher-Rao distance of identity warping is zero."""
    time_grid = np.linspace(0.0, 1.0, 100)
    gamma_id = time_grid.copy()
    d_fr = calculate_fisher_rao_distance(gamma_id, time_grid)
    assert np.isclose(d_fr, 0.0, atol=1e-4)


def test_fisher_rao_warped_positive():
    """Verify that a non-linear warping yields positive distance."""
    time_grid = np.linspace(0.0, 1.0, 100)
    gamma_warped = time_grid ** 2  # Accelerated warping
    d_fr = calculate_fisher_rao_distance(gamma_warped, time_grid)
    assert d_fr > 0.0
