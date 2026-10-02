"""Kronecker Separable 3D Joint Gaussian Process model for spatial trajectories."""

import numpy as np
from scipy.linalg import solve_triangular
from scipy.spatial.distance import cdist
from ..types import TrajectoryRibbon


def matern52_kernel(X1: np.ndarray, X2: np.ndarray, length_scale: float = 0.3) -> np.ndarray:
    """Computes Matérn 5/2 kernel between 1D inputs X1 (N1, 1) and X2 (N2, 1)."""
    dists = cdist(X1, X2, metric="euclidean")
    sqrt5 = np.sqrt(5.0)
    scaled_dist = sqrt5 * dists / max(length_scale, 1e-6)
    return (1.0 + scaled_dist + (5.0 * dists ** 2) / (3.0 * length_scale ** 2)) * np.exp(-scaled_dist)


def fit_kronecker_joint_gp(
    s_obs: np.ndarray,
    Y_shape: np.ndarray,
    length_scale: float = 0.3,
    noise_level: float = 0.1,
    ridge_eps: float = 1e-4,
) -> dict:
    """Fits an analytical Kronecker separable 3D Joint Gaussian Process.
    
    Decomposes the joint covariance into:
        K_joint = B_3x3 (x) K_temporal(s, s')
        
    Parameters
    ----------
    s_obs : np.ndarray
        Canonical warped time coordinates of shape (N,) or (N, 1) in [0, 1].
    Y_shape : np.ndarray
        Zero-centered 3D spatial coordinates of shape (N, 3).
    length_scale : float, default=0.3
        Temporal Matérn 5/2 kernel length scale.
    noise_level : float, default=0.1
        Observation noise variance sigma_n^2.
    ridge_eps : float, default=1e-4
        Numerical stability ridge added to B.
        
    Returns
    -------
    model : dict
        Model parameters containing Cholesky factors, weight matrix, B matrix,
        and training coordinates.
    """
    s = s_obs.reshape(-1, 1).astype(np.float64)
    Y = Y_shape.astype(np.float64)
    N = len(s)

    # 1. Temporal Kernel Matrix K_T (N, N)
    K_T = matern52_kernel(s, s, length_scale=length_scale)
    K_T_reg = K_T + (noise_level + 1e-6) * np.eye(N)

    # Cholesky solve: L_T @ L_T.T = K_T_reg
    L_T = np.linalg.cholesky(K_T_reg)

    # 2. Solve alpha = K_T_reg^{-1} @ Y via triangular solves
    # L_T @ v = Y  =>  v = L_T^{-1} @ Y
    v = solve_triangular(L_T, Y, lower=True)
    # L_T.T @ alpha = v  =>  alpha = L_T^{-T} @ v
    alpha = solve_triangular(L_T.T, v, lower=False)

    # 3. Closed-form ML Spatial Covariance Matrix B (3, 3)
    # B_hat = (1/N) * Y.T @ K_T_reg^{-1} @ Y = (1/N) * v.T @ v
    B_hat = (v.T @ v) / max(N, 1)
    B = B_hat + (noise_level + ridge_eps) * np.eye(3)

    return {
        "s_train": s,
        "Y_train": Y,
        "L_T": L_T,
        "alpha": alpha,
        "B": B,
        "length_scale": length_scale,
        "noise_level": noise_level,
    }


def predict_kronecker_joint_gp(
    model: dict,
    s_query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluates the Kronecker joint GP at query times s_query.
    
    Parameters
    ----------
    model : dict
        Fitted model dictionary from fit_kronecker_joint_gp.
    s_query : np.ndarray
        Query timepoints of shape (M,) or (M, 1) in [0, 1].
        
    Returns
    -------
    mu_pred : np.ndarray
        Predicted mean coordinates of shape (M, 3).
    var_temporal : np.ndarray
        Marginal temporal variance scalar per point of shape (M,).
    cov_3d : np.ndarray
        Full 3D spatial covariance tensor of shape (M, 3, 3).
    """
    s_q = s_query.reshape(-1, 1).astype(np.float64)
    s_train = model["s_train"]
    L_T = model["L_T"]
    alpha = model["alpha"]
    B = model["B"]
    length_scale = model["length_scale"]
    noise_level = model["noise_level"]

    # Cross kernel K_T_* of shape (N, M)
    K_star = matern52_kernel(s_train, s_q, length_scale=length_scale)

    # Predicted mean: mu_* = K_star.T @ alpha (M, 3)
    mu_pred = (K_star.T @ alpha).astype(np.float32)

    # Predictive temporal variance: v = L_T^{-1} @ K_star
    v = solve_triangular(L_T, K_star, lower=True)
    # var_t = k(s*, s*) - sum(v^2)
    var_temporal = np.maximum(1.0 - np.sum(v ** 2, axis=0) + noise_level, noise_level)

    # 3D spatial covariance per point: Sigma(s_*) = var_temporal(s_*) * B
    # Vectorized computation: shape (M, 3, 3)
    cov_3d = (var_temporal[:, None, None] * B[None, :, :]).astype(np.float32)

    return mu_pred, var_temporal.astype(np.float32), cov_3d


def compute_3d_mahalanobis_residuals(
    Y_observed: np.ndarray,
    mu_pred: np.ndarray,
    cov_3d: np.ndarray,
) -> tuple[float, float]:
    """Computes normalized 3D Mahalanobis residual distance and Euclidean RMSE.
    
    d_spat_shape = sqrt( mean( e(t)^T @ Sigma(t)^{-1} @ e(t) / 3 ) )
    
    Parameters
    ----------
    Y_observed : np.ndarray
        Observed coordinates of shape (T, 3).
    mu_pred : np.ndarray
        Predicted mean coordinates of shape (T, 3).
    cov_3d : np.ndarray
        Predicted covariance tensor of shape (T, 3, 3).
        
    Returns
    -------
    d_spat_shape : float
        Normalized 3D Mahalanobis residual distance.
    rmse_3d_um : float
        Euclidean 3D RMSE in microns.
    """
    diff = (Y_observed - mu_pred).astype(np.float64)
    T = len(diff)
    if T == 0:
        return 0.0, 0.0

    try:
        inv_cov = np.linalg.inv(cov_3d.astype(np.float64))
        # delta2 = sum_i sum_j diff_i * inv_cov_ij * diff_j
        delta2 = np.sum((diff[:, None, :] @ inv_cov).squeeze(1) * diff, axis=1)
    except np.linalg.LinAlgError:
        delta2 = np.zeros(T, dtype=np.float64)
        for t in range(T):
            inv_cov_t = np.linalg.pinv(cov_3d[t].astype(np.float64))
            delta2[t] = diff[t].T @ inv_cov_t @ diff[t]

    d_spat_shape = float(np.sqrt(np.mean(np.maximum(delta2, 0.0) / 3.0)))
    rmse_3d_um = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
    return d_spat_shape, rmse_3d_um


def extract_joint_trajectory_ribbon(
    cell_name: str,
    joint_model: dict,
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
    tau_cutoff: float = 0.0,
    mu_warp_min: float = 0.0,
    std_warp_min: float = 1.0,
) -> TrajectoryRibbon:
    """Pre-evaluates Kronecker joint GP into a lightweight 100-point ribbon object with 3D covariance.
    
    Populates both backwards-compatible 1D marginals and full 3D spatial covariance.
    """
    s_dense = np.linspace(0.0, 1.0, n_dense_samples).reshape(-1, 1)
    mu_dense, var_temporal, cov_3d_dense = predict_kronecker_joint_gp(
        joint_model, s_dense.ravel()
    )

    dense_pred = {}
    for idx, col in enumerate(spatial_cols):
        dense_pred[col] = {
            "mu": mu_dense[:, idx].astype(np.float32),
            "std": np.sqrt(np.maximum(cov_3d_dense[:, idx, idx], 1e-6)).astype(np.float32),
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
        dense_mu_3d=mu_dense.astype(np.float32),
        dense_cov_3d=cov_3d_dense.astype(np.float32),
        B_cov=joint_model["B"].astype(np.float32),
        mu_rot_deg=float(mu_rot_deg),
        std_rot_deg=float(std_rot_deg),
        template_curve=template_curve,
        tau_cutoff=float(tau_cutoff),
        mu_warp_min=float(mu_warp_min),
        std_warp_min=float(std_warp_min),
    )

