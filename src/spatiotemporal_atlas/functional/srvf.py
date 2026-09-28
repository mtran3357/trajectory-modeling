"""Square-Root Velocity Function (SRVF) transformation for 3D trajectory curves."""

import numpy as np


def curve_to_srvf(curve: np.ndarray, time_grid: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Transforms a 3D trajectory curve to its Square-Root Velocity Function (SRVF).
    
    q(t) = v(t) / sqrt(||v(t)|| + eps)
    
    Parameters
    ----------
    curve : np.ndarray
        Trajectory coordinates of shape (N, 3).
    time_grid : np.ndarray
        Time points of shape (N,) spanning [0, 1].
    eps : float, default=1e-6
        Regularization epsilon to prevent division by zero at zero-speed points.
        
    Returns
    -------
    q : np.ndarray
        SRVF representation of shape (N, 3).
    """
    v = np.gradient(curve, time_grid, axis=0)
    speed = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.sqrt(speed + eps)
