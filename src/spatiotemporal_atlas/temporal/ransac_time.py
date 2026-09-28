"""1D Affine temporal pose estimation using RANSAC in canonical units."""

import numpy as np


def compute_ransac_temporal_pose(
    t_mid_obs: np.ndarray,
    t_mid_ref: np.ndarray,
    max_inlier_dist_canon: float = 5.0,
    n_iterations: int = 200,
    sample_size: int = 2,
    k_bounds: tuple[float, float] = (0.3, 3.0),
    random_state: int | None = None,
) -> tuple[float, float, np.ndarray]:
    """Fits 1D affine developmental pacing: t_mid_obs approx Ke * t_mid_ref + dt0 using RANSAC.
    
    Evaluates inlier consensus directly in canonical reference time units:
      r_canon = |(t_mid_obs - dt0) / Ke - t_mid_ref| < max_inlier_dist_canon
      
    Parameters
    ----------
    t_mid_obs : np.ndarray
        Observed cell midpoint times of shape (N,).
    t_mid_ref : np.ndarray
        Reference canonical cell midpoint times of shape (N,).
    max_inlier_dist_canon : float, default=5.0
        Maximum inlier residual cutoff in canonical minutes.
    n_iterations : int, default=200
        Number of RANSAC sampling iterations.
    sample_size : int, default=2
        Minimal sample size for 1D line fitting.
    k_bounds : tuple of (float, float), default=(0.3, 3.0)
        Allowable developmental pace multiplier bounds.
    random_state : int or None, default=None
        Random seed for reproducibility.
        
    Returns
    -------
    best_ke : float
        Fitted pacing rate multiplier.
    best_dt0 : float
        Fitted developmental time offset.
    best_inliers : np.ndarray
        Boolean inlier mask of shape (N,).
    """
    n_pts = len(t_mid_obs)
    if n_pts < sample_size:
        return 1.0, 0.0, np.ones(n_pts, dtype=bool)

    rng = np.random.default_rng(random_state)
    best_inliers = np.zeros(n_pts, dtype=bool)
    best_count = -1
    best_ke = 1.0
    best_dt0 = 0.0

    for _ in range(n_iterations):
        idx = rng.choice(n_pts, size=sample_size, replace=False)
        x_sub, y_sub = t_mid_ref[idx], t_mid_obs[idx]

        dx = x_sub[1] - x_sub[0]
        if abs(dx) < 1e-4:
            continue

        ke_cand = (y_sub[1] - y_sub[0]) / dx
        if not (k_bounds[0] <= ke_cand <= k_bounds[1]):
            continue

        dt0_cand = y_sub[0] - ke_cand * x_sub[0]

        residuals_canon = np.abs((t_mid_obs - dt0_cand) / ke_cand - t_mid_ref)
        inliers = residuals_canon < max_inlier_dist_canon
        inlier_count = int(np.sum(inliers))

        if inlier_count > best_count:
            best_count = inlier_count
            best_inliers = inliers
            best_ke = ke_cand
            best_dt0 = dt0_cand

    # Polish slope and intercept over all consensus inliers via linear regression
    if best_count >= sample_size:
        x_in = t_mid_ref[best_inliers]
        y_in = t_mid_obs[best_inliers]
        cov_matrix = np.cov(x_in, y_in)
        var_x = cov_matrix[0, 0] if cov_matrix.ndim == 2 else 0.0
        if var_x > 1e-8:
            best_ke = float(cov_matrix[0, 1] / var_x)
            best_dt0 = float(np.mean(y_in) - best_ke * np.mean(x_in))

    return best_ke, best_dt0, best_inliers
