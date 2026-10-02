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
    """Scores 5 trajectory modalities against pooled WT OOF null and computes per-embryo BH-FDR.
    
    Modalities scored:
      1. temp_shape: canonical duration deviation (|z_temp_shape|)
      2. temp_shift: ancestral-propagated midpoint deviation (|z_temp_shift|)
      3. spat_shift: center-of-mass Mahalanobis shift (d_spat_shift)
      4. spat_shape: GP ribbon trajectory residual error (d_spat_shape)
      5. warp: Fisher-Rao warping distance z-score (z_warp)
      
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
        "spat_shift": ("emp_spat_shift", False),
        "spat_shape": ("emp_spat_shape", False),
        "warp": ("emp_warp", False),
    }

    cal_meta = {
        "n_null_records": len(oof_null_df),
        "n_null_embryos": int(oof_null_df["null_embryo_id"].nunique()) if not oof_null_df.empty and "null_embryo_id" in oof_null_df else 0,
    }

    for metric, (null_col, two_sided) in null_specs.items():
        p_col = f"pval_{metric}"
        q_col = f"qval_{metric}"
        hit_col = f"hit_{metric}"

        ref = oof_null_df[null_col].dropna().to_numpy() if (not oof_null_df.empty and null_col in oof_null_df) else np.array([])

        if metric == "temp_shape":
            test_score = np.abs(out["z_temp_shape"].to_numpy(dtype=float))
        elif metric == "temp_shift":
            test_score = np.abs(out["z_temp_shift"].to_numpy(dtype=float))
        elif metric == "spat_shift":
            test_score = out["d_spat_shift"].to_numpy(dtype=float)
        elif metric == "spat_shape":
            test_score = out["d_spat_shape"].to_numpy(dtype=float)
        else:
            test_score = out["rms_warp_min"].to_numpy(dtype=float) if "rms_warp_min" in out.columns else out["z_warp"].to_numpy(dtype=float)

        # Empirical p-values and per-embryo BH FDR
        out[p_col] = [calc_emp_pval(v, ref, two_sided=two_sided) for v in test_score]
        out[q_col] = bh_qvalues(out[p_col].to_numpy(dtype=float))
        out[hit_col] = out[q_col] < alpha
        cal_meta[f"q95_{metric}_emp"] = float(np.percentile(ref, 95.0)) if ref.size else np.nan

    metrics_suite = ["temp_shape", "temp_shift", "spat_shift", "spat_shape", "warp"]
    hit_cols = [f"hit_{m}" for m in metrics_suite]
    out["n_outlier_modalities"] = out[hit_cols].sum(axis=1)
    out["is_any_outlier"] = out["n_outlier_modalities"] > 0
    return out, cal_meta
