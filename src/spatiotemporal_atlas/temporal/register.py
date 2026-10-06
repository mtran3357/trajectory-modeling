"""Registration of individual embryo lifespans into canonical developmental time."""

import numpy as np
import pandas as pd
from .ransac_time import compute_ransac_temporal_pose


def register_embryo_to_temporal_atlas(
    cycles_df: pd.DataFrame,
    temporal_atlas: dict,
    cell_col: str = "cell",
    min_anchor_cells: int = 3,
    max_inlier_dist_canon: float = 5.0,
    n_ransac_iter: int = 200,
) -> tuple[pd.DataFrame, dict]:
    """Registers query embryo into canonical developmental time using 1D RANSAC.
    
    Parameters
    ----------
    cycles_df : pd.DataFrame
        Blastomere lifespans and midpoints for a single embryo.
    temporal_atlas : dict
        Reference temporal atlas with 'mu_mid' and 'mu_phys'.
    cell_col : str, default='cell'
        Blastomere cell name column.
    min_anchor_cells : int, default=3
        Minimum anchor cells required.
    max_inlier_dist_canon : float, default=5.0
        Cutoff in canonical minutes for RANSAC inlier consensus.
    n_ransac_iter : int, default=200
        RANSAC iteration count.
        
    Returns
    -------
    canon_df : pd.DataFrame
        DataFrame with canonical birth, divide, duration, mid, and log_dur.
    reg_meta : dict
        Registration parameters (k_test, dt0_test) and error diagnostics.
    """
    df_te = cycles_df.copy()
    df_te[cell_col] = df_te[cell_col].astype(str).str.strip()
    clean_atlas = {str(k).strip(): v for k, v in temporal_atlas.items()}

    valid_cells = [c for c in df_te[cell_col] if c in clean_atlas]
    if len(valid_cells) < min_anchor_cells:
        raise ValueError(
            f"Insufficient anchor blastomeres for RANSAC temporal registration "
            f"(found {len(valid_cells)}, required >= {min_anchor_cells})."
        )

    sub = df_te.set_index(cell_col).loc[valid_cells]
    t_mid_obs = sub["t_mid"].values.astype(float)
    t_mid_ref = np.array([clean_atlas[c]["mu_mid"] for c in valid_cells], dtype=float)

    # Solve Ke and dt0 via 1D Temporal RANSAC
    k_test, dt0_test, inlier_mask = compute_ransac_temporal_pose(
        t_mid_obs=t_mid_obs,
        t_mid_ref=t_mid_ref,
        max_inlier_dist_canon=max_inlier_dist_canon,
        n_iterations=n_ransac_iter,
    )

    canon_df = df_te.copy()
    canon_df["canon_birth"] = (canon_df["t_birth"] - dt0_test) / k_test
    canon_df["canon_divide"] = (canon_df["t_divide"] - dt0_test) / k_test
    canon_df["canon_duration"] = canon_df["canon_divide"] - canon_df["canon_birth"]
    canon_df["canon_mid"] = 0.5 * (canon_df["canon_birth"] + canon_df["canon_divide"])
    canon_df["canon_log_dur"] = np.log(canon_df["canon_duration"])
    canon_df["k_e"] = k_test
    canon_df["dt0_e"] = dt0_test

    # Error diagnostics
    pct_dur_errors = []
    mid_residuals = []
    for _, row in canon_df.iterrows():
        c_name = row[cell_col]
        if c_name in clean_atlas:
            ref_dur = clean_atlas[c_name]["mu_phys"]
            ref_mid = clean_atlas[c_name]["mu_mid"]
            pct_dur_errors.append(abs(row["canon_duration"] - ref_dur) / (ref_dur + 1e-6) * 100.0)
            mid_residuals.append(abs(row["canon_mid"] - ref_mid))

    reg_meta = {
        "k_test": k_test,
        "dt0_test": dt0_test,
        "n_anchor_cells": len(valid_cells),
        "n_temporal_inliers": int(np.sum(inlier_mask)),
        "temporal_inlier_ratio": float(np.mean(inlier_mask)),
        "mean_abs_dur_err_pct": float(np.mean(pct_dur_errors)) if pct_dur_errors else np.nan,
        "median_abs_mid_err": float(np.median(mid_residuals)) if mid_residuals else np.nan,
    }
    return canon_df, reg_meta
