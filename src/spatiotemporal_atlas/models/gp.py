"""Analytical Gaussian Process trajectory fitting and 1D ribbon extraction."""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel
from ..types import TrajectoryRibbon


def fit_coordinate_gps(
    X_warped: np.ndarray,
    Y_shape: np.ndarray,
    spatial_cols: list[str] | tuple[str, ...],
    length_scale: float = 0.3,
    noise_level: float = 1.0,
    length_scale_bounds: tuple[float, float] | str = (0.05, 3.0),
    noise_level_bounds: tuple[float, float] | str = (1e-4, 1e2),
    c_bounds: tuple[float, float] | str = (1e-3, 1e3),
    optimizer: str | None = None,
    n_restarts_optimizer: int = 10,
    random_state: int = 42,
) -> dict[str, GaussianProcessRegressor]:
    """Fits analytical or optimized Gaussian Process Regressors per spatial coordinate.
    
    Uses Matérn nu=2.5 kernel with WhiteKernel noise. Setting optimizer=None
    enables closed-form Cholesky decomposition without gradient descent overhead.
    When an optimizer is provided, hyperparameter bounds and restarts are applied.
    
    Parameters
    ----------
    X_warped : np.ndarray
        Warped developmental time points of shape (N, 1) in [0, 1].
    Y_shape : np.ndarray
        Trajectory spatial shape coordinates (zero-COM) of shape (N, D).
    spatial_cols : list or tuple of str
        Names of spatial coordinate columns.
    length_scale : float, default=0.3
        Matern kernel length scale.
    noise_level : float, default=1.0
        Observation noise variance.
    length_scale_bounds : tuple or str, default=(0.05, 3.0)
        Optimization bounds for length scale.
    noise_level_bounds : tuple or str, default=(1e-4, 1e2)
        Optimization bounds for noise level.
    c_bounds : tuple or str, default=(1e-3, 1e3)
        Optimization bounds for constant amplitude.
    optimizer : str or None, default=None
        Kernel hyperparameter optimizer. Set to None for analytical solve.
    n_restarts_optimizer : int, default=10
        Number of optimizer restarts if optimizer is not None.
    random_state : int, default=42
        Random seed.
        
    Returns
    -------
    gps : dict[str, GaussianProcessRegressor]
        Fitted GP models indexed by coordinate name.
    """
    gps = {}
    ls_bounds = "fixed" if optimizer is None else length_scale_bounds
    nl_bounds = "fixed" if optimizer is None else noise_level_bounds
    const_bounds = "fixed" if optimizer is None else c_bounds
    n_restarts = 0 if optimizer is None else n_restarts_optimizer

    for idx, col in enumerate(spatial_cols):
        kernel = C(1.0, const_bounds) * Matern(
            length_scale=length_scale, length_scale_bounds=ls_bounds, nu=2.5
        ) + WhiteKernel(noise_level=noise_level, noise_level_bounds=nl_bounds)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts,
            alpha=1e-6,
            random_state=random_state,
            normalize_y=True,
        )
        gp.fit(X_warped, Y_shape[:, idx])
        gps[col] = gp
    return gps



def extract_trajectory_ribbon(
    cell_name: str,
    gps: dict[str, GaussianProcessRegressor],
    spatial_cols: list[str] | tuple[str, ...],
    mu_com: np.ndarray,
    inv_cov_com: np.ndarray,
    mu_fr: float,
    std_fr: float,
    mu_srvf: np.ndarray,
    time_grid: np.ndarray,
    n_dense_samples: int = 100,
    mu_rot_deg: float = 0.0,
    std_rot_deg: float = 1.0,
    template_curve: np.ndarray | None = None,
) -> TrajectoryRibbon:
    """Pre-evaluates GP trajectories into a lightweight 100-point ribbon object.
    
    Strips internal sklearn GP objects to minimize serialized footprint and speed up inference.
    """
    s_dense = np.linspace(0.0, 1.0, n_dense_samples).reshape(-1, 1)
    dense_pred = {}
    for col in spatial_cols:
        m_d, s_d = gps[col].predict(s_dense, return_std=True)
        dense_pred[col] = {
            "mu": m_d.astype(np.float32),
            "std": s_d.astype(np.float32),
        }

    return TrajectoryRibbon(
        cell_name=cell_name,
        s_dense=s_dense.flatten().astype(np.float32),
        dense_pred=dense_pred,
        mu_com=mu_com.astype(np.float32),
        inv_cov_com=inv_cov_com.astype(np.float32),
        mu_fr=float(mu_fr),
        std_fr=float(std_fr),
        mu_srvf=mu_srvf.astype(np.float32),
        time_grid=time_grid.astype(np.float32),
        mu_rot_deg=float(mu_rot_deg),
        std_rot_deg=float(std_rot_deg),
        template_curve=template_curve,
    )

