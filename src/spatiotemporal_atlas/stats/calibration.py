"""Empirical null calibration and multi-modal trajectory anomaly scoring."""

import numpy as np
import pandas as pd
from .fdr import bh_qvalues


def calc_emp_pval(val: float, ref_arr: np.ndarray, two_sided: bool = False) -> float:
    """Calculates non-parametric empirical p-value against a reference distribution.
    
    Formula: (sum(ref >= target) + 1) / (N_ref + 1)
    
    Parameters
    ----------
    val : float
        Observed test statistic.
    ref_arr : np.ndarray
        Reference null distribution values.
    two_sided : bool, default=False
        Whether to evaluate two-sided hypothesis (|val| >= ref).
        
    Returns
    -------
    p_val : float
        Empirical p-value in (0, 1].
    """
    ref = np.asarray(ref_arr, dtype=float)
    ref = ref[np.isfinite(ref)]
    if ref.size == 0 or not np.isfinite(val):
        return 1.0
    target = abs(val) if two_sided else val
    return float((np.sum(ref >= target) + 1.0) / (ref.size + 1.0))


def apply_empirical_calibration_to_inference(
    test_res_df: pd.DataFrame,
    oof_null_df: pd.DataFrame,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, dict]:
    """Scores 6 trajectory modalities against pooled WT OOF null and computes per-embryo BH-FDR.
    
    Modalities scored:
      Temporal Domain:
        1. temp_shape: autonomous duration log-deviation (|z_temp_shape|)
        2. temp_shift: ancestral birth-time deviation (|z_temp_shift|)
        3. warp: monotonic time-warping pacing distortion (rms_warp_min)
      Spatial Domain:
        4. spat_shift: center-of-mass Mahalanobis shift (d_spat_shift via Joint GP B)
        5. spat_rot: local trajectory SO(3) Kabsch rotation angle (rot_angle_deg)
        6. spat_shape: GP ribbon trajectory path residual error (d_spat_shape)
      
    Parameters
    ----------
    test_res_df : pd.DataFrame
        Uncalibrated blastomere scores for an embryo.
    oof_null_df : pd.DataFrame
        Pooled out-of-fold WT reference null distribution.
    alpha : float, default=0.05
        FDR significance threshold.
        
    Returns
    -------
    calibrated_df : pd.DataFrame
        DataFrame augmented with pval_*, qval_*, hit_*, and outlier counts.
    cal_meta : dict
        Null metadata and empirical 95th percentiles.
    """
    out = test_res_df.copy()
    null_specs = {
        "temp_shape": ("emp_temp_shape", True),
        "temp_shift": ("emp_temp_shift", True),
        "warp": ("emp_warp", False),
        "spat_shift": ("emp_spat_shift", False),
        "spat_rot": ("emp_rot_angle", False),
        "spat_shape": ("emp_spat_shape", False),
    }

    cal_meta = {
        "n_null_records": len(oof_null_df),
        "n_null_embryos": int(oof_null_df["null_embryo_id"].nunique()) if not oof_null_df.empty and "null_embryo_id" in oof_null_df else 0,
    }

    # Cell-specific linear standardization for spatial and warping metrics
    cell_col = "cell" if "cell" in oof_null_df.columns and "cell" in out.columns and not oof_null_df.empty else None
    cell_stats = {}
    if cell_col is not None:
        for metric, (null_col, _) in null_specs.items():
            if metric in ["temp_shape", "temp_shift"]:
                continue
            if null_col in oof_null_df.columns:
                grp = oof_null_df.groupby(cell_col)[null_col]
                means = grp.mean().to_dict()
                stds = grp.std().to_dict()
                g_mu = float(oof_null_df[null_col].mean())
                g_sd = float(oof_null_df[null_col].std())
                cell_stats[metric] = (means, stds, g_mu, g_sd)

    for metric, (null_col, two_sided) in null_specs.items():
        p_col = f"pval_{metric}"
        q_col = f"qval_{metric}"
        hit_col = f"hit_{metric}"

        if metric == "temp_shape":
            test_score = np.abs(out["z_temp_shape"].to_numpy(dtype=float))
        elif metric == "temp_shift":
            test_score = np.abs(out["z_temp_shift"].to_numpy(dtype=float))
        elif metric == "warp":
            test_score = out["rms_warp_min"].to_numpy(dtype=float) if "rms_warp_min" in out.columns else out["z_warp"].to_numpy(dtype=float)
        elif metric == "spat_shift":
            test_score = out["d_spat_shift"].to_numpy(dtype=float)
        elif metric == "spat_rot":
            test_score = out["rot_angle_deg"].to_numpy(dtype=float)
        elif metric == "spat_shape":
            test_score = out["d_spat_shape"].to_numpy(dtype=float)
        else:
            test_score = np.zeros(len(out), dtype=float)

        # Standardize test scores and reference null by cell baseline
        if cell_col is not None and metric in cell_stats:
            means, stds, g_mu, g_sd = cell_stats[metric]
            z_test = []
            for c_val, s_val in zip(out[cell_col], test_score):
                mu = means.get(c_val, g_mu)
                sd = stds.get(c_val, g_sd)
                z_test.append((s_val - mu) / max(sd, 1e-6))
            test_score = np.array(z_test, dtype=float)

            if metric == "spat_shape":
                out["z_spat_shape"] = test_score
            elif metric == "spat_rot":
                out["z_rot_angle"] = test_score
            elif metric == "warp":
                out["z_warp"] = test_score
            elif metric == "spat_shift":
                out["z_spat_shift"] = test_score

            raw_ref = oof_null_df[null_col].to_numpy(dtype=float)
            cells_ref = oof_null_df[cell_col].values
            z_ref = []
            for c_val, r_val in zip(cells_ref, raw_ref):
                mu = means.get(c_val, g_mu)
                sd = stds.get(c_val, g_sd)
                z_ref.append((r_val - mu) / max(sd, 1e-6))
            ref = np.array(z_ref, dtype=float)
        else:
            ref = oof_null_df[null_col].dropna().to_numpy() if (not oof_null_df.empty and null_col in oof_null_df) else np.array([])

        # Empirical p-values and per-embryo BH FDR
        out[p_col] = [calc_emp_pval(v, ref, two_sided=two_sided) for v in test_score]
        out[q_col] = bh_qvalues(out[p_col].to_numpy(dtype=float))
        out[hit_col] = out[q_col] < alpha
        cal_meta[f"q95_{metric}_emp"] = float(np.percentile(ref, 95.0)) if ref.size else np.nan

    metrics_suite = ["temp_shape", "temp_shift", "warp", "spat_shift", "spat_rot", "spat_shape"]
    hit_cols = [f"hit_{m}" for m in metrics_suite]
    out["n_outlier_modalities"] = out[hit_cols].sum(axis=1)
    out["is_any_outlier"] = out["n_outlier_modalities"] > 0
    return out, cal_meta
