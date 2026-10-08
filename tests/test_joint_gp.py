"""Unit tests for the Kronecker Separable 3D Joint GP model."""

import numpy as np
from spatiotemporal_atlas.models.joint_gp import (
    matern52_kernel,
    fit_kronecker_joint_gp,
    predict_kronecker_joint_gp,
    compute_3d_mahalanobis_residuals,
)
from spatiotemporal_atlas.models.gp import fit_coordinate_gps


def test_matern52_kernel_properties():
    """Verify Matérn 5/2 kernel symmetry, unit diagonal, and distance decay."""
    X = np.array([[0.0], [0.5], [1.0]])
    K = matern52_kernel(X, X, length_scale=0.3)

    assert K.shape == (3, 3)
    assert np.allclose(np.diag(K), 1.0)
    assert np.allclose(K, K.T)
    # K(0, 0) > K(0, 1) > K(0, 2)
    assert K[0, 0] > K[0, 1] > K[0, 2]


def test_joint_gp_fit_and_prediction():
    """Verify Kronecker joint GP produces valid positive-definite 3D covariance and smooth mean."""
    rng = np.random.default_rng(42)
    s = np.linspace(0.0, 1.0, 25).reshape(-1, 1)
    # Trajectory with coupled x and y coordinates
    x = np.sin(2 * np.pi * s.flatten())
    y = 0.8 * x + rng.normal(scale=0.05, size=25)  # Correlated with x
    z = np.cos(2 * np.pi * s.flatten())
    Y = np.column_stack([x, y, z])
    Y = Y - np.mean(Y, axis=0)  # zero-mean trajectory shape

    model = fit_kronecker_joint_gp(s, Y, length_scale=0.3, noise_level=0.05)

    B = model["B"]
    assert B.shape == (3, 3)
    assert np.allclose(B, B.T)
    # B must be positive definite (all eigenvalues > 0)
    eigvals = np.linalg.eigvalsh(B)
    assert np.all(eigvals > 0.0)

    # Cross-covariance between x and y should be distinctly non-zero
    assert abs(B[0, 1]) > 0.05

    # Predict on a dense grid
    s_dense = np.linspace(0.0, 1.0, 100)
    mu_pred, var_t, cov_3d = predict_kronecker_joint_gp(model, s_dense)

    assert mu_pred.shape == (100, 3)
    assert var_t.shape == (100,)
    assert cov_3d.shape == (100, 3, 3)

    # Test return_cov=False fast mean-only path
    mu_fast, var_fast, cov_fast = predict_kronecker_joint_gp(model, s_dense, return_cov=False)
    assert np.allclose(mu_fast, mu_pred)
    assert var_fast is None
    assert cov_fast is None

    # Check Mahalanobis residual calculation on test points
    d_spat_shape, rmse = compute_3d_mahalanobis_residuals(Y[:10], mu_pred[:10], cov_3d[:10])
    assert d_spat_shape >= 0.0
    assert rmse >= 0.0


def test_joint_vs_independent_mean_concordance():
    """Verify that mean predictions from Kronecker GP are identical to independent GPs on zero-mean shapes."""
    s = np.linspace(0.0, 1.0, 25).reshape(-1, 1)
    Y = np.column_stack([
        np.sin(np.pi * s.flatten()),
        np.cos(np.pi * s.flatten()),
        s.flatten() ** 2,
    ])
    Y = Y - np.mean(Y, axis=0)  # zero-mean shape (matching xyz_shape)

    # 1. Independent GPs
    spatial_cols = ["x", "y", "z"]
    ind_gps = fit_coordinate_gps(
        s, Y, spatial_cols=spatial_cols, length_scale=0.3, noise_level=0.05, optimizer=None
    )
    s_query = np.linspace(0.0, 1.0, 30).reshape(-1, 1)
    ind_mu = np.column_stack([
        ind_gps[col].predict(s_query) for col in spatial_cols
    ])

    # 2. Kronecker Joint GP
    joint_model = fit_kronecker_joint_gp(s, Y, length_scale=0.3, noise_level=0.05)
    joint_mu, _, _ = predict_kronecker_joint_gp(joint_model, s_query.flatten())

    # The means from separable formulation and independent GPs share the same Matérn temporal basis
    # and should be algebraically identical to within numerical tolerance (< 1e-5)
    assert np.allclose(joint_mu, ind_mu, atol=1e-5)


def test_extract_joint_trajectory_ribbon_serialization():
    """Verify extract_joint_trajectory_ribbon creates valid ribbons and round-trips via to_dict/from_dict."""
    from spatiotemporal_atlas.models.joint_gp import extract_joint_trajectory_ribbon
    from spatiotemporal_atlas.types import TrajectoryRibbon

    s = np.linspace(0.0, 1.0, 20).reshape(-1, 1)
    Y = np.column_stack([
        np.sin(np.pi * s.flatten()),
        np.cos(np.pi * s.flatten()),
        s.flatten(),
    ])
    spatial_cols = ["x", "y", "z"]
    joint_model = fit_kronecker_joint_gp(s, Y, length_scale=0.3, noise_level=0.1)

    ribbon = extract_joint_trajectory_ribbon(
        cell_name="ABa",
        joint_model=joint_model,
        spatial_cols=spatial_cols,
        mu_com=np.array([10.0, 20.0, 30.0]),
        inv_cov_com=np.eye(3),
        mu_fr=0.15,
        std_fr=0.03,
        mu_srvf=np.zeros((100, 3)),
        time_grid=np.linspace(0.0, 1.0, 100),
        n_dense_samples=100,
    )

    # Check ribbon attributes and dictionary access
    assert isinstance(ribbon, TrajectoryRibbon)
    assert ribbon.dense_mu_3d.shape == (100, 3)
    assert ribbon.dense_cov_3d.shape == (100, 3, 3)
    assert ribbon.B_cov.shape == (3, 3)
    assert ribbon["cell_name"] == "ABa"
    assert np.allclose(ribbon["dense_mu_3d"], ribbon.dense_mu_3d)

    # Check 1D marginals backwards-compatibility
    for col in spatial_cols:
        assert col in ribbon.dense_pred
        assert "mu" in ribbon.dense_pred[col]
        assert "std" in ribbon.dense_pred[col]
        assert len(ribbon.dense_pred[col]["mu"]) == 100

    # Serialization round-trip
    d = ribbon.to_dict()
    assert "dense_mu_3d" in d
    assert "dense_cov_3d" in d
    assert "B_cov" in d

    ribbon_restored = TrajectoryRibbon.from_dict(d)
    assert ribbon_restored.cell_name == "ABa"
    assert np.allclose(ribbon_restored.dense_mu_3d, ribbon.dense_mu_3d)
    assert np.allclose(ribbon_restored.dense_cov_3d, ribbon.dense_cov_3d)
    assert np.allclose(ribbon_restored.B_cov, ribbon.B_cov)
    assert ribbon_restored.length_scale == ribbon.length_scale
    assert ribbon_restored.noise_level == ribbon.noise_level


def test_full_joint_mahalanobis_residuals():
    """Verify full joint GP Mahalanobis residual properties: zero residual, smoothness sensitivity."""
    from spatiotemporal_atlas.models.joint_gp import compute_full_joint_mahalanobis_residuals

    T = 20
    u_eval = np.linspace(0.0, 1.0, T)
    mu_pred = np.zeros((T, 3))
    B = np.eye(3) * 2.0

    # 1. Exact match produces zero distance
    d_zero, rmse_zero = compute_full_joint_mahalanobis_residuals(
        Y_observed=mu_pred,
        mu_pred=mu_pred,
        B=B,
        u_eval=u_eval,
        length_scale=0.3,
        noise_level=0.1,
    )
    assert np.isclose(d_zero, 0.0)
    assert np.isclose(rmse_zero, 0.0)

    # 2. Smooth coherent drift vs high-frequency oscillation with identical Euclidean RMSE
    # Smooth drift: low curvature
    smooth_y = 1.0 * np.sin(np.pi * u_eval)[:, None] * np.ones((1, 3))
    # Oscillating signal: high frequency alternating signs
    osc_signs = np.array([1.0, -1.0] * (T // 2))[:, None]
    osc_y = 1.0 * osc_signs * np.ones((1, 3))

    # Match Euclidean norms exactly
    norm_smooth = np.sqrt(np.mean(smooth_y ** 2))
    norm_osc = np.sqrt(np.mean(osc_y ** 2))
    smooth_y = smooth_y * (norm_osc / norm_smooth)

    d_smooth, rmse_smooth = compute_full_joint_mahalanobis_residuals(
        Y_observed=smooth_y,
        mu_pred=mu_pred,
        B=B,
        u_eval=u_eval,
        length_scale=0.3,
        noise_level=0.05,
    )
    d_osc, rmse_osc = compute_full_joint_mahalanobis_residuals(
        Y_observed=osc_y,
        mu_pred=mu_pred,
        B=B,
        u_eval=u_eval,
        length_scale=0.3,
        noise_level=0.05,
    )

    # Both have the same Euclidean RMSE
    assert np.isclose(rmse_smooth, rmse_osc, rtol=1e-4)

    # High frequency oscillation violates temporal continuity and receives significantly higher penalty!
    assert d_osc > d_smooth * 1.5

