"""Multiple testing correction and false discovery rate (FDR) control."""

import numpy as np


def bh_qvalues(p_values: np.ndarray | list[float]) -> np.ndarray:
    """Computes Benjamini-Hochberg False Discovery Rate (FDR) q-values.
    
    Guarantees monotonicity: q_(i) <= q_(i+1).
    
    Parameters
    ----------
    p_values : array-like of shape (M,)
        Raw p-values.
        
    Returns
    -------
    q_values : np.ndarray of shape (M,)
        FDR-adjusted q-values clipped to [0, 1].
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return p.copy()
    order = np.argsort(p)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, m + 1)
    q = (p * m) / np.maximum(ranks, 1)
    sorted_q = q[order]
    monotone = np.minimum.accumulate(sorted_q[::-1])[::-1]
    q[order] = np.clip(monotone, 0.0, 1.0)
    return q
