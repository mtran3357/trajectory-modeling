"""Riemannian metrics and geodesic distances for elastic shape analysis."""

import numpy as np


def calculate_fisher_rao_distance(gamma: np.ndarray, time_grid: np.ndarray) -> float:
    """Calculates geodesic Fisher-Rao distance on the unit Hilbert sphere.
    
    Given a time-warping function gamma: [0, 1] -> [0, 1], its representation
    psi = sqrt(dot_gamma) lies on the unit sphere of L2([0, 1]).
    The geodesic distance to the identity warping gamma_id(t) = t is:
        theta = arccos( <psi, psi_id> )
    
    Parameters
    ----------
    gamma : np.ndarray
        Time warping function values of shape (N,).
    time_grid : np.ndarray
        Time grid of shape (N,).
        
    Returns
    -------
    distance : float
        Fisher-Rao geodesic distance in radians.
    """
    dot_gamma = np.maximum(np.gradient(gamma, time_grid), 0.0)
    psi = np.sqrt(dot_gamma)
    dt = np.gradient(time_grid)
    inner_prod = np.clip(np.sum(psi * dt), -1.0, 1.0)
    return float(np.arccos(inner_prod))
