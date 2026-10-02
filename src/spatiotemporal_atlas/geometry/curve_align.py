"""Local 3D trajectory curve alignment via SO(3) Kabsch rotation and Generalized Procrustes Analysis (GPA)."""

import numpy as np
from scipy.interpolate import interp1d


def kabsch_curve_so3(
    P: np.ndarray,
    Q: np.ndarray,
    allow_reflection: bool = False,
) -> tuple[np.ndarray, float]:
    """Computes the optimal SO(3) rigid rotation matrix aligning centered trajectory P to Q.
    
    Minimizes:
        min_{R in SO(3)} || R @ P.T - Q.T ||_F^2
        
    Parameters
    ----------
    P : np.ndarray of shape (N, 3)
        Source centered coordinate trajectory.
    Q : np.ndarray of shape (N, 3)
        Target centered coordinate trajectory.
    allow_reflection : bool, default=False
        If False, enforces det(R) = +1 via Umeyama singular value adjustment
        to strictly prevent coordinate reflections or inversions.
        
    Returns
    -------
    R : np.ndarray of shape (3, 3)
        Optimal rotation matrix in SO(3).
    theta_deg : float
        Geodesic rotation angle in degrees: arccos((trace(R) - 1) / 2).
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)

    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Enforce proper rotation (det(R) = +1)
    if not allow_reflection and np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T

    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    if np.isclose(tr, 1.0, atol=1e-12):
        theta_deg = 0.0
    elif np.isclose(tr, -1.0, atol=1e-12):
        theta_deg = 180.0
    else:
        theta_deg = float(np.degrees(np.arccos(tr)))
    return R, theta_deg



def interpolate_curve_to_grid(
    s_obs: np.ndarray,
    coords: np.ndarray,
    s_grid: np.ndarray,
) -> np.ndarray:
    """Resamples a 3D coordinate trajectory onto uniform developmental progression nodes.
    
    Parameters
    ----------
    s_obs : np.ndarray of shape (N,)
        Observed developmental progression values in [0, 1].
    coords : np.ndarray of shape (N, 3)
        Observed 3D spatial coordinates.
    s_grid : np.ndarray of shape (K,)
        Target uniform evaluation nodes in [0, 1].
        
    Returns
    -------
    coords_resampled : np.ndarray of shape (K, 3)
        Resampled 3D trajectory coordinates.
    """
    s_obs = np.asarray(s_obs, dtype=float)
    coords = np.asarray(coords, dtype=float)

    # Sort by developmental progression and remove duplicate timestamps
    idx_sort = np.argsort(s_obs)
    s_sorted = s_obs[idx_sort]
    coords_sorted = coords[idx_sort]

    u_s, u_idx = np.unique(s_sorted, return_index=True)
    u_coords = coords_sorted[u_idx]

    if len(u_s) < 2:
        return np.repeat(u_coords[0:1], len(s_grid), axis=0)

    interp = interp1d(u_s, u_coords, axis=0, kind="linear", fill_value="extrapolate")
    return interp(s_grid)


def generalized_procrustes_curves(
    trajectories_dict: dict[str, tuple[np.ndarray, np.ndarray]],
    s_grid: np.ndarray,
    max_iters: int = 25,
    tol: float = 1e-5,
) -> dict:
    """Iteratively aligns an ensemble of replicate trajectories onto a consensus Fréchet
    mean shape via Generalized Procrustes Analysis (GPA) with SO(3) Kabsch rotation.
    
    An orientation anchor constraint is enforced at each iteration: the consensus template
    is rigidly re-anchored to the initial unrotated cohort orientation. This guarantees
    zero rotational drift relative to the embryo's canonical AP/DV/LR axes.
    
    Parameters
    ----------
    trajectories_dict : dict
        Mapping embryo_id -> (s_obs, coords_centered) where coords_centered has shape (N, 3).
    s_grid : np.ndarray of shape (K,)
        Uniform nodes in [0, 1] for discrete shape representation.
    max_iters : int, default=25
        Maximum GPA refinement iterations.
    tol : float, default=1e-5
        Convergence tolerance on mean template Frobenius displacement.
        
    Returns
    -------
    result : dict
        Dictionary containing:
        - "aligned_trajectories": dict of embryo_id -> (s_obs, coords_aligned)
        - "rotations": dict of embryo_id -> R (3, 3)
        - "angles_deg": dict of embryo_id -> theta_deg
        - "template_curve": np.ndarray of shape (K, 3)
        - "mean_angle_deg": float
        - "std_angle_deg": float
    """
    embryo_ids = sorted(list(trajectories_dict.keys()))
    M = len(embryo_ids)
    if M == 0:
        raise ValueError("Cannot perform GPA on an empty trajectories dictionary.")

    # Resample all centered trajectories onto uniform grid
    resampled_stack = np.zeros((M, len(s_grid), 3), dtype=float)
    for i, emb_id in enumerate(embryo_ids):
        s_obs, coords = trajectories_dict[emb_id]
        resampled_stack[i] = interpolate_curve_to_grid(s_obs, coords, s_grid)

    # Initial consensus template: unrotated ensemble mean
    initial_seed_template = np.mean(resampled_stack, axis=0)
    consensus_template = initial_seed_template.copy()

    rotations = {emb_id: np.eye(3) for emb_id in embryo_ids}
    angles_deg = {emb_id: 0.0 for emb_id in embryo_ids}

    for iteration in range(max_iters):
        aligned_stack = np.zeros_like(resampled_stack)

        for i, emb_id in enumerate(embryo_ids):
            P = resampled_stack[i]
            Q = consensus_template
            R, theta = kabsch_curve_so3(P, Q, allow_reflection=False)
            rotations[emb_id] = R
            angles_deg[emb_id] = theta
            aligned_stack[i] = (R @ P.T).T

        new_consensus = np.mean(aligned_stack, axis=0)

        # Re-anchor consensus template to initial orientation to eliminate rotational drift
        R_anchor, _ = kabsch_curve_so3(new_consensus, initial_seed_template, allow_reflection=False)
        new_consensus = (R_anchor @ new_consensus.T).T

        # Update per-embryo rotations with anchor correction
        for emb_id in embryo_ids:
            rotations[emb_id] = R_anchor @ rotations[emb_id]
            tr = np.clip((np.trace(rotations[emb_id]) - 1.0) / 2.0, -1.0, 1.0)
            angles_deg[emb_id] = float(np.degrees(np.arccos(tr)))

        shift = np.linalg.norm(new_consensus - consensus_template)
        consensus_template = new_consensus
        if shift < tol:
            break

    # Apply final optimal rotations to original observed coordinate tracks
    aligned_trajectories = {}
    for emb_id in embryo_ids:
        s_obs, coords = trajectories_dict[emb_id]
        R = rotations[emb_id]
        coords_rot = (R @ coords.T).T
        aligned_trajectories[emb_id] = (s_obs, coords_rot)

    angle_vals = list(angles_deg.values())
    mean_angle = float(np.mean(angle_vals)) if angle_vals else 0.0
    std_angle = float(np.std(angle_vals, ddof=1)) if len(angle_vals) > 1 else 1.0

    return {
        "aligned_trajectories": aligned_trajectories,
        "rotations": rotations,
        "angles_deg": angles_deg,
        "template_curve": consensus_template,
        "mean_angle_deg": mean_angle,
        "std_angle_deg": std_angle,
    }


def masked_generalized_procrustes(
    trajectories_dict: dict[str, tuple[np.ndarray, np.ndarray]],
    tau_grid: np.ndarray,
    max_iters: int = 25,
    tol: float = 1e-5,
) -> dict:
    """Iteratively aligns an ensemble of variable-duration trajectories onto a consensus
    template via Generalized Procrustes Analysis (GPA) with SO(3) Kabsch rotation.
    
    For each replicate, centering and Kabsch rotation are computed strictly over observed
    nodes tau <= tau_dur. An orientation anchor constraint is enforced at each iteration:
    the consensus template is rigidly re-anchored to the initial unrotated cohort orientation.
    This guarantees zero rotational drift relative to the embryo's canonical axes.
    
    Parameters
    ----------
    trajectories_dict : dict
        Mapping embryo_id -> (tau_obs, coords_centered) where coords_centered has shape (N, 3).
    tau_grid : np.ndarray of shape (K,)
        Uniform evaluation nodes in [0, tau_max] for discrete shape representation.
    max_iters : int, default=25
        Maximum GPA refinement iterations.
    tol : float, default=1e-5
        Convergence tolerance on mean template Frobenius displacement.
        
    Returns
    -------
    result : dict
        Dictionary containing:
        - "aligned_trajectories": dict of embryo_id -> (tau_obs, coords_aligned)
        - "rotations": dict of embryo_id -> R (3, 3)
        - "angles_deg": dict of embryo_id -> theta_deg
        - "template_curve": np.ndarray of shape (K, 3)
        - "mean_angle_deg": float
        - "std_angle_deg": float
        - "tau_grid": np.ndarray of shape (K,)
        - "resampled_stack": np.ndarray of shape (M, K, 3)
        - "active_masks": np.ndarray of shape (M, K) boolean
    """
    embryo_ids = sorted(list(trajectories_dict.keys()))
    M = len(embryo_ids)
    K = len(tau_grid)
    if M == 0:
        raise ValueError("Cannot perform GPA on an empty trajectories dictionary.")

    resampled_stack = np.zeros((M, K, 3), dtype=float)
    active_masks = np.zeros((M, K), dtype=bool)

    for i, emb_id in enumerate(embryo_ids):
        tau_obs, coords = trajectories_dict[emb_id]
        tau_max_emb = float(tau_obs.max())
        mask_i = tau_grid <= tau_max_emb + 1e-4
        if not np.any(mask_i):
            mask_i[0] = True
        active_masks[i] = mask_i

        active_tau = tau_grid[mask_i]
        coords_resampled = interpolate_curve_to_grid(tau_obs, coords, active_tau)
        resampled_stack[i, mask_i] = coords_resampled

    # Initial seed consensus template: mean over active embryos at each node
    initial_seed_template = np.zeros((K, 3), dtype=float)
    for k in range(K):
        actives = active_masks[:, k]
        if np.any(actives):
            initial_seed_template[k] = np.mean(resampled_stack[actives, k], axis=0)
        else:
            initial_seed_template[k] = initial_seed_template[max(k - 1, 0)]

    consensus_template = initial_seed_template.copy()
    rotations = {emb_id: np.eye(3) for emb_id in embryo_ids}
    angles_deg = {emb_id: 0.0 for emb_id in embryo_ids}

    for iteration in range(max_iters):
        aligned_stack = np.zeros_like(resampled_stack)

        for i, emb_id in enumerate(embryo_ids):
            mask_i = active_masks[i]
            if np.sum(mask_i) < 2:
                aligned_stack[i] = resampled_stack[i]
                continue

            P_act = resampled_stack[i, mask_i]
            Q_act = consensus_template[mask_i]

            P_act_c = P_act - np.mean(P_act, axis=0)
            Q_act_c = Q_act - np.mean(Q_act, axis=0)

            R, theta = kabsch_curve_so3(P_act_c, Q_act_c, allow_reflection=False)
            rotations[emb_id] = R
            angles_deg[emb_id] = theta

            aligned_stack[i, mask_i] = (R @ P_act.T).T

        new_consensus = np.zeros((K, 3), dtype=float)
        for k in range(K):
            actives = active_masks[:, k]
            if np.any(actives):
                new_consensus[k] = np.mean(aligned_stack[actives, k], axis=0)
            else:
                new_consensus[k] = new_consensus[max(k - 1, 0)]

        # Orientation anchor constraint: rigidly re-anchor consensus to initial seed orientation
        R_anchor, _ = kabsch_curve_so3(
            new_consensus - np.mean(new_consensus, axis=0),
            initial_seed_template - np.mean(initial_seed_template, axis=0),
            allow_reflection=False,
        )
        new_consensus = (R_anchor @ new_consensus.T).T

        for emb_id in embryo_ids:
            rotations[emb_id] = R_anchor @ rotations[emb_id]
            tr = np.clip((np.trace(rotations[emb_id]) - 1.0) / 2.0, -1.0, 1.0)
            angles_deg[emb_id] = float(np.degrees(np.arccos(tr)))

        shift = np.linalg.norm(new_consensus - consensus_template)
        consensus_template = new_consensus
        if shift < tol:
            break

    aligned_trajectories = {}
    for emb_id in embryo_ids:
        tau_obs, coords = trajectories_dict[emb_id]
        R = rotations[emb_id]
        coords_rot = (R @ coords.T).T
        aligned_trajectories[emb_id] = (tau_obs, coords_rot)

    angle_vals = list(angles_deg.values())
    mean_angle = float(np.mean(angle_vals)) if angle_vals else 0.0
    std_angle = float(np.std(angle_vals, ddof=1)) if len(angle_vals) > 1 else 1.0

    return {
        "aligned_trajectories": aligned_trajectories,
        "rotations": rotations,
        "angles_deg": angles_deg,
        "template_curve": consensus_template,
        "mean_angle_deg": mean_angle,
        "std_angle_deg": std_angle,
        "tau_grid": tau_grid,
        "resampled_stack": resampled_stack,
        "active_masks": active_masks,
    }


def register_curve_to_template(
    s_obs: np.ndarray,
    coords_centered: np.ndarray,
    template_curve: np.ndarray,
    s_grid: np.ndarray,
    template_s: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Rigidly registers a centered query trajectory to a reference template curve in SO(3).
    
    Both query and template trajectories are evaluated on progress nodes s_grid.
    If the query track duration is shorter than s_grid[-1], registration is evaluated
    strictly over the active observed nodes.
    
    Parameters
    ----------
    s_obs : np.ndarray of shape (N,)
        Observed progression or elapsed values for query track.
    coords_centered : np.ndarray of shape (N, 3)
        Zero-mean centered query 3D coordinates.
    template_curve : np.ndarray of shape (K, 3) or (M, 3)
        Template curve coordinates.
    s_grid : np.ndarray of shape (K,)
        Uniform evaluation nodes.
    template_s : np.ndarray of shape (M,), optional
        Progression nodes corresponding to template_curve if not already on s_grid.
        
    Returns
    -------
    R_test : np.ndarray of shape (3, 3)
        Optimal rotation matrix in SO(3).
    theta_deg : float
        Geodesic rotation angle in degrees.
    coords_aligned : np.ndarray of shape (N, 3)
        Rotated query coordinates: (R_test @ coords_centered.T).T.
    """
    coords_centered = np.asarray(coords_centered, dtype=float)
    s_obs = np.asarray(s_obs, dtype=float)

    s_max_obs = float(s_obs.max())
    active_mask = s_grid <= s_max_obs + 1e-4
    if not np.any(active_mask):
        active_mask[0] = True
    s_act = s_grid[active_mask]

    P_grid = interpolate_curve_to_grid(s_obs, coords_centered, s_act)

    if template_s is not None and len(template_curve) == len(template_s):
        Q_full = interpolate_curve_to_grid(template_s, template_curve, s_grid)
    elif len(template_curve) == len(s_grid):
        Q_full = template_curve
    else:
        s_src = np.linspace(s_grid[0], s_grid[-1], len(template_curve))
        Q_full = interpolate_curve_to_grid(s_src, template_curve, s_grid)

    Q_grid = Q_full[active_mask]

    P_grid_centered = P_grid - np.mean(P_grid, axis=0)
    Q_grid_centered = Q_grid - np.mean(Q_grid, axis=0)

    R_test, theta_deg = kabsch_curve_so3(P_grid_centered, Q_grid_centered, allow_reflection=False)
    coords_aligned = (R_test @ coords_centered.T).T

    return R_test, theta_deg, coords_aligned

