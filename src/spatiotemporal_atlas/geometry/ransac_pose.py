"""3D RANSAC similarity pose estimation combining random sample consensus with Umeyama SVD."""

import numpy as np
from .umeyama import umeyama_similarity_transform


def compute_ransac_similarity_pose(
    P: np.ndarray,
    Q: np.ndarray,
    max_inlier_dist_um: float = 4.0,
    n_iterations: int = 500,
    sample_size: int = 4,
    allow_scaling: bool = True,
    random_state: int | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Computes robust similarity pose: Q approx s * (R @ P) + t.
    
    Guarantees proper rotation R in SO(3) via Umeyama SVD reflection correction.
    
    Returns
    -------
    best_s : float
        Uniform scale factor.
    best_R : np.ndarray
        (3, 3) rotation matrix in SO(3).
    best_t : np.ndarray
        (3,) translation vector.
    best_inliers : np.ndarray
        (N,) boolean inlier mask.
    """
    n_pts, m_dim = P.shape
    if n_pts < sample_size:
        mu_p, mu_q = np.mean(P, axis=0), np.mean(Q, axis=0)
        return 1.0, np.eye(m_dim), mu_q - mu_p, np.ones(n_pts, dtype=bool)

    rng = np.random.default_rng(random_state)
    best_inliers = np.zeros(n_pts, dtype=bool)
    best_count = -1
    best_s = 1.0
    best_R = np.eye(m_dim)
    best_t = np.zeros(m_dim)

    for _ in range(n_iterations):
        idx = rng.choice(n_pts, size=sample_size, replace=False)
        P_sub, Q_sub = P[idx], Q[idx]

        s_cand, R_cand, t_cand = umeyama_similarity_transform(
            P_sub, Q_sub, allow_scaling=allow_scaling
        )

        P_trans = s_cand * (R_cand @ P.T).T + t_cand
        residuals = np.linalg.norm(Q - P_trans, axis=1)
        inliers = residuals < max_inlier_dist_um
        inlier_count = int(np.sum(inliers))

        if inlier_count > best_count:
            best_count = inlier_count
            best_inliers = inliers
            best_s = s_cand
            best_R = R_cand
            best_t = t_cand

    if best_count >= sample_size:
        best_s, best_R, best_t = umeyama_similarity_transform(
            P[best_inliers], Q[best_inliers], allow_scaling=allow_scaling
        )

    return best_s, best_R, best_t, best_inliers
