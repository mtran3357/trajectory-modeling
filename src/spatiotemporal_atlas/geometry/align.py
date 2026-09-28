"""Embryo coordinate alignment to canonical spatial template."""

import numpy as np
import pandas as pd
from .ransac_pose import compute_ransac_similarity_pose


def align_embryo_to_spatial_template(
    emb_coords_df: pd.DataFrame,
    spatial_template_df: pd.DataFrame,
    micron_cols: list[str] | tuple[str, ...] = ("x_um", "y_um", "z_um"),
    aligned_cols: list[str] | tuple[str, ...] = ("x_aligned_um", "y_aligned_um", "z_aligned_um"),
    cell_col: str = "cell",
    max_inlier_dist_um: float = 4.0,
    n_ransac_iter: int = 500,
    allow_scaling: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Aligns an individual embryo's coordinates to the canonical spatial template.
    
    Returns
    -------
    df_out : pd.DataFrame
        DataFrame with aligned coordinates populated in aligned_cols.
    reg_metrics : dict
        Registration quality diagnostics including scale, rotation, translation,
        inlier ratio, residuals, and per-cell inlier mapping.
    """
    micron_cols = list(micron_cols)
    aligned_cols = list(aligned_cols)

    df_out = emb_coords_df.copy()
    for col in aligned_cols:
        df_out[col] = np.nan

    emb_cents = (
        df_out.dropna(subset=micron_cols)
        .groupby(cell_col)[micron_cols]
        .mean()
    )
    common = sorted(list(set(emb_cents.index).intersection(set(spatial_template_df.index))))

    if len(common) < 4:
        P_raw = df_out[micron_cols].values
        mu_p = np.mean(P_raw, axis=0) if len(P_raw) else np.zeros(3)
        mu_q = np.mean(spatial_template_df[micron_cols].values, axis=0)
        t_fallback = mu_q - mu_p
        for i, a_col in enumerate(aligned_cols):
            df_out[a_col] = P_raw[:, i] + t_fallback[i] if len(P_raw) else np.nan

        reg_metrics = {
            "scale_s": 1.0,
            "R": np.eye(3),
            "t": t_fallback,
            "inlier_ratio": 0.0,
            "n_inliers": 0,
            "n_total": len(emb_cents),
            "n_common": len(common),
            "mean_inlier_res_um": np.nan,
            "mean_all_res_um": np.nan,
            "is_spatial_inlier_map": {c: False for c in emb_cents.index},
        }
        return df_out, reg_metrics

    P_arr = emb_cents.loc[common, micron_cols].values
    Q_arr = spatial_template_df.loc[common, micron_cols].values

    s_e, R_e, t_e, inlier_mask = compute_ransac_similarity_pose(
        P=P_arr,
        Q=Q_arr,
        max_inlier_dist_um=max_inlier_dist_um,
        n_iterations=n_ransac_iter,
        sample_size=4,
        allow_scaling=allow_scaling,
    )

    raw_coords = df_out[micron_cols].values
    aligned_coords = s_e * (R_e @ raw_coords.T).T + t_e
    for i, a_col in enumerate(aligned_cols):
        df_out[a_col] = aligned_coords[:, i]

    P_trans = s_e * (R_e @ P_arr.T).T + t_e
    residuals = np.linalg.norm(Q_arr - P_trans, axis=1)
    inlier_res = residuals[inlier_mask]

    is_inlier_map = {c: False for c in emb_cents.index}
    for c_name, is_in in zip(common, inlier_mask):
        is_inlier_map[c_name] = bool(is_in)

    reg_metrics = {
        "scale_s": float(s_e),
        "R": R_e,
        "t": t_e,
        "inlier_ratio": float(np.mean(inlier_mask)),
        "n_inliers": int(np.sum(inlier_mask)),
        "n_total": len(emb_cents),
        "n_common": len(common),
        "mean_inlier_res_um": float(np.mean(inlier_res)) if len(inlier_res) else np.nan,
        "mean_all_res_um": float(np.mean(residuals)),
        "is_spatial_inlier_map": is_inlier_map,
    }
    return df_out, reg_metrics
