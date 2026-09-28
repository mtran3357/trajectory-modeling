"""Dynamic programming time-warping alignment for SRVF curves on [0, 1]."""

import numpy as np
from scipy.interpolate import interp1d


def align_srvf_dp_clamped(
    q_template: np.ndarray, q_query: np.ndarray, time_grid: np.ndarray
) -> np.ndarray:
    """Dynamic programming alignment between two 3D SRVF curves on [0, 1].
    
    Finds optimal monotonic diffeomorphism gamma: [0, 1] -> [0, 1] minimizing:
        ||q_template(t) - q_query(gamma(t)) * sqrt(dot_gamma(t))||^2
    
    Parameters
    ----------
    q_template : np.ndarray
        Template SRVF curve of shape (N, 3).
    q_query : np.ndarray
        Query SRVF curve of shape (N, 3).
    time_grid : np.ndarray
        Time points of shape (N,) spanning [0, 1].
        
    Returns
    -------
    gamma : np.ndarray
        Monotonic time-warping function gamma(t) of shape (N,) on [0, 1].
    """
    N = len(time_grid)
    cost_matrix = np.sum((q_template[:, None, :] - q_query[None, :, :]) ** 2, axis=-1)

    D = np.full((N, N), np.inf)
    D[0, 0] = cost_matrix[0, 0]
    phi = np.zeros((N, N, 2), dtype=int)

    for i in range(N):
        for j in range(N):
            if i == 0 and j == 0:
                continue
            min_val = np.inf
            best_prev = (0, 0)
            for di, dj in [(1, 1), (1, 2), (2, 1)]:
                pi, pj = i - di, j - dj
                if pi >= 0 and pj >= 0:
                    val = D[pi, pj] + cost_matrix[i, j]
                    if val < min_val:
                        min_val = val
                        best_prev = (pi, pj)
            if not np.isinf(min_val):
                D[i, j] = min_val
                phi[i, j] = best_prev

    path = [(N - 1, N - 1)]
    curr = (N - 1, N - 1)
    while curr != (0, 0):
        prev = tuple(phi[curr[0], curr[1]])
        if prev == (0, 0) and curr != (0, 0) and curr != (1, 1):
            path.append((0, 0))
            break
        path.append(prev)
        curr = prev
    path = np.array(path[::-1])

    t_t_raw = time_grid[path[:, 0]]
    t_q_raw = time_grid[path[:, 1]]

    _, unique_idx = np.unique(t_q_raw, return_index=True)
    t_q_clean = t_q_raw[unique_idx]
    t_t_clean = t_t_raw[unique_idx]

    if t_q_clean[0] > 0.0:
        t_q_clean = np.insert(t_q_clean, 0, 0.0)
        t_t_clean = np.insert(t_t_clean, 0, 0.0)
    if t_q_clean[-1] < 1.0:
        t_q_clean = np.append(t_q_clean, 1.0)
        t_t_clean = np.append(t_t_clean, 1.0)

    gamma_interp = interp1d(t_q_clean, t_t_clean, kind="linear", fill_value="extrapolate")
    return np.clip(gamma_interp(time_grid), 0.0, 1.0)
