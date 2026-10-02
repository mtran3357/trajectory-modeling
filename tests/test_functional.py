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


def test_regularized_monotonic_time_warp_identity():
    """Verify regularized_monotonic_time_warp recovers identity when query matches template."""
    from spatiotemporal_atlas.functional.time_warp import regularized_monotonic_time_warp
    tau_grid = np.linspace(0.0, 25.0, 26)
    template_curve = np.column_stack([
        tau_grid * 0.8,
        np.sin(tau_grid * 0.2) * 5.0,
        np.cos(tau_grid * 0.2) * 3.0,
    ])
    template_centered = template_curve - np.mean(template_curve, axis=0)

    gamma_hat, rms_warp, slopes = regularized_monotonic_time_warp(
        tau_obs=tau_grid,
        coords_aligned=template_centered,
        template_curve=template_centered,
        tau_grid=tau_grid,
        lambda_reg=10.0,
        slope_bounds=(0.5, 2.0),
    )

    assert np.isclose(rms_warp, 0.0, atol=0.05)
    assert np.allclose(gamma_hat, tau_grid, atol=0.1)
    assert np.all(slopes >= 0.5 - 1e-4)
    assert np.all(slopes <= 2.0 + 1e-4)


def test_regularized_monotonic_time_warp_slope_bounds():
    """Verify regularized_monotonic_time_warp strictly respects slope bounds [0.5, 2.0]."""
    from spatiotemporal_atlas.functional.time_warp import regularized_monotonic_time_warp
    tau_grid = np.linspace(0.0, 30.0, 31)
    template = np.column_stack([tau_grid * 1.0, np.zeros_like(tau_grid), np.zeros_like(tau_grid)])

    # Extremely distorted query (e.g. 5x faster progression)
    tau_obs = np.linspace(0.0, 30.0, 31)
    query_distorted = np.column_stack([tau_obs * 5.0, np.zeros_like(tau_obs), np.zeros_like(tau_obs)])

    gamma_hat, rms_warp, slopes = regularized_monotonic_time_warp(
        tau_obs=tau_obs,
        coords_aligned=query_distorted,
        template_curve=template,
        tau_grid=tau_grid,
        lambda_reg=10.0,
        slope_bounds=(0.5, 2.0),
    )

    assert np.all(slopes >= 0.5 - 1e-3)
    assert np.all(slopes <= 2.0 + 1e-3)
    # Monotonicity check
    assert np.all(np.diff(gamma_hat) >= -1e-5)


def test_regularized_monotonic_time_warp_synthetic_recovery():
    """Verify regularized monotonic warping accurately recovers synthetic non-linear warping."""
    from spatiotemporal_atlas.functional.time_warp import regularized_monotonic_time_warp
    tau_grid = np.linspace(0.0, 20.0, 21)
    template = np.column_stack([
        tau_grid * 1.2,
        np.sin(tau_grid * 0.3) * 4.0,
        np.cos(tau_grid * 0.3) * 3.0,
    ])
    template_centered = template - np.mean(template, axis=0)

    # Synthetic acceleration: gamma(t) = 20 * (t/20)^1.3 (slope starts at 0 and grows)
    tau_obs = tau_grid.copy()
    u = tau_obs / 20.0
    gamma_syn = 20.0 * (u ** 1.3)
    # Sample template at gamma_syn to create synthetic warped query
    coords_syn = np.column_stack([
        np.interp(gamma_syn, tau_grid, template_centered[:, d]) for d in range(3)
    ])

    gamma_hat, rms_warp, slopes = regularized_monotonic_time_warp(
        tau_obs=tau_obs,
        coords_aligned=coords_syn,
        template_curve=template_centered,
        tau_grid=tau_grid,
        lambda_reg=10.0,
        slope_bounds=(0.5, 2.0),
    )

    # Recovered warping should correlate very strongly with true synthetic warping
    corr = np.corrcoef(gamma_hat, gamma_syn)[0, 1]
    assert corr > 0.95
    assert rms_warp > 0.1  # should detect positive warping


def test_compute_warp_metrics():
    """Verify compute_warp_metrics calculates rms_warp_min and signed_warp_area."""
    from spatiotemporal_atlas.functional.time_warp import compute_warp_metrics
    tau_grid = np.linspace(0.0, 10.0, 11)
    gamma = tau_grid + 1.0  # constant shift of 1 minute
    slopes = np.ones(10)

    metrics = compute_warp_metrics(gamma, tau_grid, slopes)
    assert np.isclose(metrics["rms_warp_min"], 1.0, atol=1e-5)
    assert np.isclose(metrics["max_warp_min"], 1.0, atol=1e-5)
    assert np.isclose(metrics["signed_warp_area"], 10.0, atol=1e-4)
    assert np.isclose(metrics["mean_slope"], 1.0, atol=1e-5)

