"""Unit tests for analytical GP fitting and TrajectoryRibbon interpolation."""

import numpy as np
from spatiotemporal_atlas.models.gp import fit_coordinate_gps, extract_trajectory_ribbon


def test_gp_and_ribbon_interpolation_accuracy():
    """Verify that TrajectoryRibbon interpolation closely matches direct GP evaluation."""
    rng = np.random.default_rng(42)
    s_train = np.linspace(0.0, 1.0, 30).reshape(-1, 1)
    # Synthetic 3D trajectory
    y_train = np.column_stack([
        np.sin(np.pi * s_train.flatten()),
        np.cos(np.pi * s_train.flatten()),
        s_train.flatten() ** 2,
    ]) + rng.normal(scale=0.02, size=(30, 3))

    spatial_cols = ["x", "y", "z"]
    gps = fit_coordinate_gps(
        X_warped=s_train,
        Y_shape=y_train,
        spatial_cols=spatial_cols,
        length_scale=0.3,
        noise_level=0.1,
        optimizer=None,
    )

    ribbon = extract_trajectory_ribbon(
        cell_name="ABa",
        gps=gps,
        spatial_cols=spatial_cols,
        mu_com=np.array([1.0, 2.0, 3.0]),
        inv_cov_com=np.eye(3),
        mu_fr=0.1,
        std_fr=0.02,
        mu_srvf=np.zeros((40, 3)),
        time_grid=np.linspace(0.0, 1.0, 40),
        n_dense_samples=100,
    )

    # Test on query points
    s_query = np.array([0.15, 0.42, 0.78, 0.91])
    ribbon_mu, ribbon_std = ribbon.predict(s_query, spatial_cols)

    # Direct GP prediction
    gp_mu = np.zeros((len(s_query), 3))
    gp_std = np.zeros((len(s_query), 3))
    for i, col in enumerate(spatial_cols):
        m, s = gps[col].predict(s_query.reshape(-1, 1), return_std=True)
        gp_mu[:, i] = m
        gp_std[:, i] = s

    # Ribbon interpolation error should be minimal (< 0.01)
    assert np.allclose(ribbon_mu, gp_mu, atol=1e-2)
    assert np.allclose(ribbon_std, gp_std, atol=1e-2)
