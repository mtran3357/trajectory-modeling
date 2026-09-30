"""Unit tests for 1D RANSAC temporal pacing and pose estimation."""

import numpy as np
from spatiotemporal_atlas.temporal import compute_ransac_temporal_pose


def test_ransac_temporal_pose_clean():
    """Verify 1D RANSAC recovers known Ke and dt0 with no noise."""
    t_ref = np.linspace(10.0, 100.0, 10)
    ke_true = 1.2
    dt0_true = 5.0
    t_obs = ke_true * t_ref + dt0_true

    ke, dt0, inliers = compute_ransac_temporal_pose(
        t_mid_obs=t_obs,
        t_mid_ref=t_ref,
        max_inlier_dist_canon=1.0,
        n_iterations=100,
        random_state=42,
    )

    assert np.all(inliers)
    assert np.isclose(ke, ke_true, atol=1e-4)
    assert np.isclose(dt0, dt0_true, atol=1e-3)


def test_ransac_temporal_pose_with_outliers():
    """Verify 1D RANSAC ignores outliers in observed timings."""
    rng = np.random.default_rng(42)
    t_ref = np.linspace(10.0, 120.0, 15)
    ke_true = 0.95
    dt0_true = 12.0
    t_obs = ke_true * t_ref + dt0_true

    # Corrupt 3 cells with huge timing delays/skips
    t_obs[2] += 40.0
    t_obs[7] -= 35.0
    t_obs[12] += 50.0

    ke, dt0, inliers = compute_ransac_temporal_pose(
        t_mid_obs=t_obs,
        t_mid_ref=t_ref,
        max_inlier_dist_canon=5.0,
        n_iterations=200,
        random_state=42,
    )

    assert np.isclose(ke, ke_true, atol=0.05)
    assert np.isclose(dt0, dt0_true, atol=1.5)
    assert not inliers[2]
    assert not inliers[7]
    assert not inliers[12]
