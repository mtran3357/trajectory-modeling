"""Iterative Karcher mean SRVF template computation."""

import numpy as np
from scipy.interpolate import interp1d
from .dp_warp import align_srvf_dp_clamped


def compute_karcher_mean_srvf(
    srvf_list: list[np.ndarray],
    time_grid: np.ndarray,
    max_iter: int = 5,
    tol: float = 1e-3,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Iteratively computes the Karcher Mean SRVF template across training tracks.
    
    Parameters
    ----------
    srvf_list : list of np.ndarray
        List of M SRVF curves, each of shape (N, 3).
    time_grid : np.ndarray
        Common time grid of shape (N,) spanning [0, 1].
    max_iter : int, default=5
        Maximum iterations.
    tol : float, default=1e-3
        Convergence tolerance for relative update norm.
        
    Returns
    -------
    mu_q : np.ndarray
        Karcher mean SRVF template of shape (N, 3).
    gammas : list of np.ndarray
        List of M optimal warping functions gamma_i(t).
    """
    M = len(srvf_list)
    if M == 0:
        raise ValueError("Cannot compute Karcher mean on an empty srvf_list.")
    if M == 1:
        return srvf_list[0].copy(), [time_grid.copy()]

    mu_q = np.mean(srvf_list, axis=0)
    gammas = [time_grid.copy() for _ in range(M)]

    for _ in range(max_iter):
        warped_srvfs = []
        for i in range(M):
            gamma_i = align_srvf_dp_clamped(mu_q, srvf_list[i], time_grid)
            gammas[i] = gamma_i
            interp_q = interp1d(time_grid, srvf_list[i], axis=0, fill_value="extrapolate")
            dot_gamma = np.maximum(np.gradient(gamma_i, time_grid), 1e-4)[:, None]
            warped_srvfs.append(interp_q(gamma_i) * np.sqrt(dot_gamma))

        mu_q_new = np.mean(warped_srvfs, axis=0)
        diff = np.linalg.norm(mu_q_new - mu_q) / (np.linalg.norm(mu_q) + 1e-6)
        mu_q = mu_q_new
        if diff < tol:
            break

    return mu_q, gammas
