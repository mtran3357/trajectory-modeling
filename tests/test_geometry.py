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


def test_kabsch_curve_so3_identity():
    """Verify kabsch_curve_so3 returns identity and 0 deg for identical curves."""
    from spatiotemporal_atlas.geometry import kabsch_curve_so3
    s = np.linspace(0, 1, 20)
    P = np.column_stack([np.sin(s * np.pi), np.cos(s * np.pi), s])
    P_centered = P - np.mean(P, axis=0)

    R, theta = kabsch_curve_so3(P_centered, P_centered)
    assert np.allclose(R, np.eye(3), atol=1e-6)
    assert np.isclose(theta, 0.0, atol=1e-6)


def test_kabsch_curve_so3_known_rotation():
    """Verify kabsch_curve_so3 recovers exact rotation angle and matrix."""
    from spatiotemporal_atlas.geometry import kabsch_curve_so3
    s = np.linspace(0, 1, 20)
    P = np.column_stack([s, np.sin(s * 2 * np.pi), np.cos(s * 2 * np.pi)])
    P_centered = P - np.mean(P, axis=0)

    # 45 deg rotation around Y axis
    angle_rad = np.radians(45.0)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    R_true = np.array([
        [cos_a,  0.0, sin_a],
        [0.0,    1.0, 0.0],
        [-sin_a, 0.0, cos_a],
    ])

    Q_centered = (R_true @ P_centered.T).T
    R_rec, theta_rec = kabsch_curve_so3(P_centered, Q_centered)

    assert np.allclose(R_rec, R_true, atol=1e-5)
    assert np.isclose(theta_rec, 45.0, atol=1e-4)


def test_kabsch_curve_so3_no_reflection():
    """Verify kabsch_curve_so3 enforces det(R) = +1 even when target is reflected."""
    from spatiotemporal_atlas.geometry import kabsch_curve_so3
    rng = np.random.default_rng(42)
    P = rng.standard_normal((15, 3))
    P_centered = P - np.mean(P, axis=0)

    # Reflected target
    Q_centered = P_centered.copy()
    Q_centered[:, 2] = -Q_centered[:, 2]

    R, theta = kabsch_curve_so3(P_centered, Q_centered, allow_reflection=False)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5)


def test_generalized_procrustes_curves():
    """Verify GPA converges and produces consensus template curve."""
    from spatiotemporal_atlas.geometry import generalized_procrustes_curves
    s_grid = np.linspace(0, 1, 21)

    # Base curve
    base = np.column_stack([s_grid * 10, np.sin(s_grid * np.pi) * 3, np.cos(s_grid * np.pi) * 2])
    base_centered = base - np.mean(base, axis=0)

    # Generate 3 rotated versions
    trajectories = {}
    angles = [0.0, 15.0, -20.0]
    for i, ang in enumerate(angles):
        rad = np.radians(ang)
        R_z = np.array([
            [np.cos(rad), -np.sin(rad), 0],
            [np.sin(rad),  np.cos(rad), 0],
            [0,            0,           1],
        ])
        coords_rot = (R_z @ base_centered.T).T
        trajectories[f"emb_{i}"] = (s_grid, coords_rot)

    res = generalized_procrustes_curves(trajectories, s_grid=s_grid, max_iters=20)
    assert "aligned_trajectories" in res
    assert "template_curve" in res
    assert "angles_deg" in res
    assert len(res["aligned_trajectories"]) == 3
    assert res["template_curve"].shape == (21, 3)
    assert res["mean_angle_deg"] >= 0.0


def test_register_curve_to_template():
    """Verify register_curve_to_template aligns query to template."""
    from spatiotemporal_atlas.geometry import register_curve_to_template
    s_grid = np.linspace(0, 1, 21)
    template = np.column_stack([s_grid * 5, np.sin(s_grid * np.pi) * 3, np.cos(s_grid * np.pi) * 2])
    template_centered = template - np.mean(template, axis=0)

    # Rotated query by 30 degrees around Z
    rad = np.radians(30.0)
    R_z = np.array([
        [np.cos(rad), -np.sin(rad), 0],
        [np.sin(rad),  np.cos(rad), 0],
        [0,            0,           1],
    ])
    s_obs = np.linspace(0, 1, 15)
    query_raw = np.column_stack([s_obs * 5, np.sin(s_obs * np.pi) * 3, np.cos(s_obs * np.pi) * 2])
    query_centered = query_raw - np.mean(query_raw, axis=0)
    query_rot = (R_z @ query_centered.T).T

    R_test, theta_deg, coords_aligned = register_curve_to_template(
        s_obs=s_obs,
        coords_centered=query_rot,
        template_curve=template_centered,
        s_grid=s_grid,
    )

    assert np.isclose(theta_deg, 30.0, atol=0.5)
    assert coords_aligned.shape == query_centered.shape
    # After rotation, aligned coordinates should match query_centered
    assert np.allclose(coords_aligned, query_centered, atol=0.2)


def test_masked_generalized_procrustes():
    """Verify masked GPA converges with variable-lifespan trajectories."""
    from spatiotemporal_atlas.geometry import masked_generalized_procrustes
    tau_grid = np.linspace(0.0, 30.0, 31)

    # Base curve: spiral trajectory in physical minutes
    base_fn = lambda t: np.column_stack([t * 0.8, np.sin(t * 0.2) * 4.0, np.cos(t * 0.2) * 3.0])

    # 3 embryos with different lifespans: 20 min, 25 min, 30 min
    trajectories = {}
    durations = [20.0, 25.0, 30.0]
    angles = [0.0, 20.0, -15.0]

    for i, (dur, ang) in enumerate(zip(durations, angles)):
        t_obs = np.linspace(0.0, dur, int(dur) + 1)
        coords = base_fn(t_obs)
        coords_centered = coords - np.mean(coords, axis=0)

        rad = np.radians(ang)
        R_z = np.array([
            [np.cos(rad), -np.sin(rad), 0.0],
            [np.sin(rad),  np.cos(rad), 0.0],
            [0.0,          0.0,         1.0],
        ])
        coords_rot = (R_z @ coords_centered.T).T
        trajectories[f"emb_{i}"] = (t_obs, coords_rot)

    res = masked_generalized_procrustes(trajectories, tau_grid=tau_grid, max_iters=25)

    assert "aligned_trajectories" in res
    assert "template_curve" in res
    assert "angles_deg" in res
    assert len(res["aligned_trajectories"]) == 3
    assert res["template_curve"].shape == (31, 3)
    assert res["mean_angle_deg"] >= 0.0
    for emb_id in trajectories:
        assert emb_id in res["aligned_trajectories"]
        orig_tau, _ = trajectories[emb_id]
        aligned_tau, aligned_coords = res["aligned_trajectories"][emb_id]
        assert aligned_coords.shape == (len(orig_tau), 3)


def test_register_curve_to_template_with_mask():
    """Verify register_curve_to_template correctly masks active nodes when query duration < grid."""
    from spatiotemporal_atlas.geometry import register_curve_to_template
    tau_grid = np.linspace(0.0, 30.0, 31)
    template = np.column_stack([tau_grid * 0.5, np.sin(tau_grid * 0.3) * 3.0, np.cos(tau_grid * 0.3) * 2.0])
    template_centered = template - np.mean(template, axis=0)

    # Query trajectory only observed up to 18.0 min (shorter than 30.0 min template)
    tau_obs = np.linspace(0.0, 18.0, 19)
    query_raw = np.column_stack([tau_obs * 0.5, np.sin(tau_obs * 0.3) * 3.0, np.cos(tau_obs * 0.3) * 2.0])
    query_centered = query_raw - np.mean(query_raw, axis=0)

    # Rotate query by 25 degrees around Z axis
    rad = np.radians(25.0)
    R_z = np.array([
        [np.cos(rad), -np.sin(rad), 0.0],
        [np.sin(rad),  np.cos(rad), 0.0],
        [0.0,          0.0,         1.0],
    ])
    query_rot = (R_z @ query_centered.T).T

    R_test, theta_deg, coords_aligned = register_curve_to_template(
        s_obs=tau_obs,
        coords_centered=query_rot,
        template_curve=template_centered,
        s_grid=tau_grid,
    )

    assert np.isclose(theta_deg, 25.0, atol=0.5)
    assert coords_aligned.shape == query_centered.shape
    assert np.allclose(coords_aligned, query_centered, atol=0.2)



