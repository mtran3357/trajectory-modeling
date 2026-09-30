"""Unit tests for 3D geometry, Umeyama SVD, and RANSAC similarity alignment."""

import numpy as np
from spatiotemporal_atlas.geometry import (
    umeyama_similarity_transform,
    compute_ransac_similarity_pose,
)


def test_umeyama_identity():
    """Verify Umeyama returns identity transform for identical point sets."""
    P = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 1.0, 2.0],
        [2.0, 8.0, 4.0],
    ])
    s, R, t = umeyama_similarity_transform(P, P)
    assert np.isclose(s, 1.0)
    assert np.allclose(R, np.eye(3), atol=1e-6)
    assert np.allclose(t, np.zeros(3), atol=1e-6)


def test_umeyama_so3_no_reflection():
    """Verify that Umeyama guarantees det(R) == +1 and never returns a reflection."""
    rng = np.random.default_rng(42)
    P = rng.standard_normal((10, 3))
    # Create reflected target: Q = P with Z negated
    Q = P.copy()
    Q[:, 2] = -Q[:, 2]

    s, R, t = umeyama_similarity_transform(P, Q, allow_scaling=True)
    det_R = np.linalg.det(R)
    assert np.isclose(det_R, 1.0, atol=1e-5), f"det(R) should be +1, got {det_R}"


def test_umeyama_known_rigid_transform():
    """Verify Umeyama recovers known scale, 90-degree rotation, and translation."""
    P = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [2.0, 1.0, 3.0],
    ])
    # 90-deg rotation around Z
    R_true = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ])
    s_true = 2.5
    t_true = np.array([10.0, -5.0, 2.0])

    Q = s_true * (R_true @ P.T).T + t_true
    s, R, t = umeyama_similarity_transform(P, Q, allow_scaling=True)

    assert np.isclose(s, s_true, atol=1e-5)
    assert np.allclose(R, R_true, atol=1e-5)
    assert np.allclose(t, t_true, atol=1e-5)


def test_ransac_similarity_pose_with_outliers():
    """Verify RANSAC correctly rejects outliers and finds the true consensus pose."""
    rng = np.random.default_rng(123)
    P_inliers = rng.standard_normal((20, 3)) * 10.0

    R_true = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ])
    s_true = 1.0
    t_true = np.array([5.0, 2.0, -3.0])

    Q_inliers = s_true * (R_true @ P_inliers.T).T + t_true

    # Add 5 aggressive outliers
    P_outliers = rng.standard_normal((5, 3)) * 10.0
    Q_outliers = rng.standard_normal((5, 3)) * 50.0

    P = np.vstack([P_inliers, P_outliers])
    Q = np.vstack([Q_inliers, Q_outliers])

    s, R, t, inliers = compute_ransac_similarity_pose(
        P, Q, max_inlier_dist_um=1.0, n_iterations=300, random_state=42
    )

    # Inlier mask should have flagged the first 20 as inliers and last 5 as outliers
    assert np.sum(inliers[:20]) >= 18
    assert np.sum(inliers[20:]) == 0
    assert np.isclose(s, s_true, atol=0.1)
    assert np.allclose(t, t_true, atol=0.5)
