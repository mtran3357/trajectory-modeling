"""Closed-form Umeyama similarity pose estimation with SO(3) reflection correction."""

import numpy as np


def umeyama_similarity_transform(
    P: np.ndarray,
    Q: np.ndarray,
    allow_scaling: bool = True,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Computes optimal similarity transformation: Q approx s * (R @ P) + t.
    
    Guarantees proper rotation R in SO(3) with det(R) = +1 via reflection correction.
    
    Parameters
    ----------
    P : np.ndarray
        Source points of shape (N, D).
    Q : np.ndarray
        Target points of shape (N, D).
    allow_scaling : bool, default=True
        Whether to estimate uniform isotropic scale factor s.
        
    Returns
    -------
    s : float
        Uniform isotropic scale factor.
    R : np.ndarray
        Proper rotation matrix of shape (D, D) in SO(D).
    t : np.ndarray
        Translation vector of shape (D,).
    """
    n_pts, m_dim = P.shape
    if n_pts == 0:
        return 1.0, np.eye(m_dim), np.zeros(m_dim)

    mu_p = np.mean(P, axis=0)
    mu_q = np.mean(Q, axis=0)

    P_c = P - mu_p
    Q_c = Q - mu_q

    var_p = np.sum(P_c ** 2) / n_pts
    if var_p < 1e-10:
        return 1.0, np.eye(m_dim), mu_q - mu_p

    H = (P_c.T @ Q_c) / n_pts
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T

    # SO(D) proper rotation check: ensure det(R) == +1 (no reflections)
    d = np.linalg.det(V @ U.T)
    S_diag = np.ones(m_dim)
    if d < 0:
        S_diag[-1] = -1.0

    R = V @ np.diag(S_diag) @ U.T
    s = float((1.0 / var_p) * np.sum(S * S_diag)) if allow_scaling else 1.0
    t = mu_q - s * (R @ mu_p)

    return s, R, t
