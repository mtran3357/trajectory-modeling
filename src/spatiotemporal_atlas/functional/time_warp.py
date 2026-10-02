"""Regularized monotonic time-warping optimization aligning 3D trajectories."""

import numpy as np
from scipy.optimize import minimize


def regularized_monotonic_time_warp(
    tau_obs: np.ndarray,
    coords_aligned: np.ndarray,
    template_curve: np.ndarray,
    tau_grid: np.ndarray,
    lambda_reg: float = 10.0,
    slope_bounds: tuple[float, float] = (0.5, 2.0),
) -> tuple[np.ndarray, float, np.ndarray]:
    """Optimizes a regularized monotonic time-warping function gamma(tau) aligning
    a query trajectory to a reference template curve.
    
    Minimizes:
        L(gamma) = sum_k || coords(gamma(tau_k)) - template(tau_k) ||^2
                   + lambda_reg * sum_k ((dgamma/dtau - 1)^2) * delta_tau
                   
    subject to:
        gamma(0) = 0
        slope_min <= (gamma_{k+1} - gamma_k) / delta_tau <= slope_max
        
    Parameters
    ----------
    tau_obs : np.ndarray of shape (N,)
        Observed elapsed times.
    coords_aligned : np.ndarray of shape (N, 3)
        Spatially rotated query coordinates.
    template_curve : np.ndarray of shape (K, 3)
        Consensus reference template curve evaluated on tau_grid.
    tau_grid : np.ndarray of shape (K,)
        Uniform evaluation grid in [0, tau_max].
    lambda_reg : float, default=10.0
        Regularization penalty strength on velocity deviation from unit speed.
    slope_bounds : tuple[float, float], default=(0.5, 2.0)
        Allowable instantaneous pacing bounds [s_min, s_max].
        
    Returns
    -------
    gamma_opt : np.ndarray of shape (K_act,)
        Optimized warped timestamps evaluated at active tau_grid nodes.
    rms_warp_min : float
        Root-mean-square temporal warping deviation: sqrt(mean((gamma_opt - tau_act)^2)) in minutes.
    pacing_derivatives : np.ndarray of shape (K_act - 1,)
        Instantaneous slope values (dgamma/dtau).
    """
    tau_obs = np.asarray(tau_obs, dtype=float)
    coords_aligned = np.asarray(coords_aligned, dtype=float)
    tau_max_obs = float(tau_obs.max())

    # Active grid nodes
    active_mask = tau_grid <= tau_max_obs + 1e-4
    if not np.any(active_mask):
        active_mask[0] = True
    tau_act = tau_grid[active_mask]
    K_act = len(tau_act)

    if K_act <= 1:
        return tau_act, 0.0, np.array([1.0])

    Q_act = template_curve[active_mask]
    delta_tau = np.diff(tau_act)
    s_min, s_max = slope_bounds

    # Precompute velocity profile for analytical gradient
    v_obs = np.gradient(coords_aligned, tau_obs, axis=0) if len(tau_obs) > 1 else np.zeros_like(coords_aligned)

    # Parameterization: step increments delta_k = gamma_{k+1} - gamma_k
    delta_init = delta_tau.copy()
    bounds = [(s_min * dt, s_max * dt) for dt in delta_tau]

    def loss_and_grad(deltas: np.ndarray) -> tuple[float, np.ndarray]:
        gamma = np.zeros(K_act, dtype=float)
        gamma[1:] = np.cumsum(deltas)
        gamma_clipped = np.clip(gamma, tau_obs[0], tau_max_obs)

        # Fast vectorized 1D interpolation
        coords_eval = np.column_stack([
            np.interp(gamma_clipped, tau_obs, coords_aligned[:, d]) for d in range(3)
        ])
        v_eval = np.column_stack([
            np.interp(gamma_clipped, tau_obs, v_obs[:, d]) for d in range(3)
        ])

        diff = coords_eval - Q_act
        spatial_loss = float(np.sum(diff ** 2))

        slopes = deltas / delta_tau
        reg_loss = float(lambda_reg * np.sum(((slopes - 1.0) ** 2) * delta_tau))
        total_loss = spatial_loss + reg_loss

        # Exact analytical gradient:
        # dS/dgamma = 2 * (coords_eval - Q_act) . v_eval
        dS_dgamma = 2.0 * np.sum(diff * v_eval, axis=1)
        # dS/ddelta_m = sum_{k > m} dS/dgamma_k
        dS_ddelta = np.cumsum(dS_dgamma[1:][::-1])[::-1]
        dreg_ddelta = 2.0 * lambda_reg * (slopes - 1.0)
        grad = (dS_ddelta + dreg_ddelta).astype(float)

        return total_loss, grad

    res = minimize(
        loss_and_grad,
        x0=delta_init,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 60, "ftol": 1e-6},
    )

    deltas_opt = res.x if res.success or res.x is not None else delta_init
    gamma_opt = np.zeros(K_act, dtype=float)
    gamma_opt[1:] = np.cumsum(deltas_opt)

    slopes_opt = deltas_opt / delta_tau
    rms_warp_min = float(np.sqrt(np.mean((gamma_opt - tau_act) ** 2)))

    return gamma_opt, rms_warp_min, slopes_opt


def compute_warp_metrics(
    gamma_opt: np.ndarray,
    tau_act: np.ndarray,
    slopes_opt: np.ndarray | None = None,
) -> dict[str, float]:
    """Computes standard temporal warping summary metrics.
    
    Parameters
    ----------
    gamma_opt : np.ndarray
        Warped elapsed times.
    tau_act : np.ndarray
        Unwarped evaluation times.
    slopes_opt : np.ndarray, optional
        Precomputed derivatives dgamma/dtau.
        
    Returns
    -------
    dict[str, float]
        Dictionary of summary metrics:
        - rms_warp_min: Root-mean-square displacement from identity in minutes.
        - max_warp_min: Maximum absolute displacement from identity in minutes.
        - signed_warp_area: Integral of (gamma - tau) over tau.
        - mean_slope: Average instantaneous speed dgamma/dtau.
    """
    gamma_opt = np.asarray(gamma_opt, dtype=float)
    tau_act = np.asarray(tau_act, dtype=float)
    
    diff = gamma_opt - tau_act
    rms_warp_min = float(np.sqrt(np.mean(diff ** 2)))
    max_warp_min = float(np.max(np.abs(diff)))
    signed_warp_area = float(np.trapezoid(diff, tau_act)) if len(tau_act) > 1 else 0.0

    if slopes_opt is None:
        delta_tau = np.diff(tau_act)
        slopes_opt = np.diff(gamma_opt) / np.maximum(delta_tau, 1e-6) if len(delta_tau) > 0 else np.array([1.0])
    mean_slope = float(np.mean(slopes_opt)) if len(slopes_opt) > 0 else 1.0

    return {
        "rms_warp_min": rms_warp_min,
        "max_warp_min": max_warp_min,
        "signed_warp_area": signed_warp_area,
        "mean_slope": mean_slope,
    }
