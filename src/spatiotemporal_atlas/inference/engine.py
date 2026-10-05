"""Stage 2: Inference pipeline for query / perturbed embryos."""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from ..geometry.align import align_embryo_to_spatial_template
from ..functional.time_warp import regularized_monotonic_time_warp, compute_warp_metrics
from ..lineage.graph import parse_lineage_graph, get_ancestral_path_in_interval
from ..temporal.lifespans import extract_cell_lifespans
from ..temporal.register import register_embryo_to_temporal_atlas
from ..stats.calibration import apply_empirical_calibration_to_inference
from ..models.joint_gp import compute_3d_mahalanobis_residuals
from ..geometry.curve_align import register_curve_to_template
from ..types import ReferenceAtlas, TrajectoryRibbon



def run_embryo_inference(
    query_pos_df: pd.DataFrame,
    lineage_df: pd.DataFrame,
    atlas_bundle: dict | ReferenceAtlas,
    alpha: float = 0.05,
    min_observations: int = 3,
    max_inlier_dist_canon: float | None = None,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Executes spatiotemporal inference for query embryos against pre-fit WT atlas.
    
    Parameters
    ----------
    query_pos_df : pd.DataFrame
        Query embryo tracking coordinates.
    lineage_df : pd.DataFrame
        Lineage graph tree dataframe.
    atlas_bundle : dict or ReferenceAtlas
        Stage 1 fitted atlas bundle.
    alpha : float, default=0.05
        FDR significance threshold.
    min_observations : int, default=3
        Minimum observations per cell track.
    max_inlier_dist_canon : float or None, default=None
        Maximum inlier cutoff for 1D temporal RANSAC.
    n_jobs : int, default=1
        Worker count (reserved for batch inference).
        
    Returns
    -------
    cell_scores_df : pd.DataFrame
        Scores across 5 modalities with empirical p-values and BH q-values.
    embryo_qc_df : pd.DataFrame
        Embryo-level spatial and temporal registration quality diagnostics.
    """
    bundle = atlas_bundle.to_dict() if isinstance(atlas_bundle, ReferenceAtlas) else atlas_bundle

    embryo_col = bundle["embryo_col"]
    cell_col = bundle["cell_col"]
    time_col = bundle["time_col"]
    micron_cols = bundle["micron_cols"]
    aligned_cols = bundle["aligned_cols"]
    raw_spatial_cols = bundle["raw_spatial_cols"]
    voxel_size_xyz = bundle["voxel_size_xyz"]
    spatial_template = bundle["spatial_template"]
    temporal_atlas = bundle["temporal_atlas"]
    cell_models = bundle["cell_models"]
    oof_null_df = bundle["oof_null_df"]
    max_inlier_dist_um = bundle["max_inlier_dist_um"]
    local_trajectory_alignment = bundle.get("local_trajectory_alignment", True)
    warping_lambda = float(bundle.get("warping_lambda", 10.0))
    warping_slope_bounds = tuple(bundle.get("warping_slope_bounds", (0.5, 2.0)))
    tau_cutoffs = bundle.get("tau_cutoffs", {})


    if max_inlier_dist_canon is None:
        max_inlier_dist_canon = bundle.get("max_inlier_dist_canon", 5.0)

    # 1. Scale physical coordinates
    df = query_pos_df.copy()
    df[embryo_col] = df[embryo_col].astype(str)
    df[cell_col] = df[cell_col].astype(str).str.strip()

    for raw_c, mic_c, scale in zip(raw_spatial_cols, micron_cols, voxel_size_xyz):
        df[mic_c] = df[raw_c].astype(float) * scale

    parent_map = parse_lineage_graph(lineage_df)
    pooled_log_stds = [v["std_log"] for v in temporal_atlas.values() if v["std_log"] > 0]
    sigma_log_pool = float(np.median(pooled_log_stds)) if pooled_log_stds else 0.05
    var_tempo_estimation = (sigma_log_pool ** 2) / max(len(temporal_atlas), 1)

    all_cell_scores = []
    embryo_qc_records = []

    for q_id, q_grp in df.groupby(embryo_col):
        # Step A: Align query embryo to consensus spatial template
        aligned_q, s_meta = align_embryo_to_spatial_template(
            emb_coords_df=q_grp,
            spatial_template_df=spatial_template,
            micron_cols=micron_cols,
            aligned_cols=aligned_cols,
            cell_col=cell_col,
            max_inlier_dist_um=max_inlier_dist_um,
            allow_scaling=True,
        )

        # Step B: Register query embryo into canonical developmental time via 1D RANSAC
        cycles_df = extract_cell_lifespans(
            aligned_q, time_col=time_col, embryo_col=embryo_col, cell_col=cell_col
        )
        try:
            canon_cycles, t_meta = register_embryo_to_temporal_atlas(
                cycles_df=cycles_df,
                temporal_atlas=temporal_atlas,
                cell_col=cell_col,
                max_inlier_dist_canon=max_inlier_dist_canon,
            )
        except ValueError as e:
            print(f"[Warning] Embryo '{q_id}' failed temporal registration: {e}")
            continue

        # Record embryo-level spatiotemporal registration metrics
        embryo_qc_records.append({
            "embryo_id": q_id,
            "inlier_ratio": s_meta["inlier_ratio"],
            "n_inliers": s_meta["n_inliers"],
            "n_total": s_meta["n_total"],
            "n_common_spatial": s_meta["n_common"],
            "mean_inlier_res_um": s_meta["mean_inlier_res_um"],
            "mean_all_res_um": s_meta["mean_all_res_um"],
            "scale_s": s_meta["scale_s"],
            "k_test": t_meta["k_test"],
            "dt0_test": t_meta["dt0_test"],
            "n_anchor_temporal": t_meta["n_anchor_cells"],
            "n_temporal_inliers": t_meta["n_temporal_inliers"],
            "temporal_inlier_ratio": t_meta["temporal_inlier_ratio"],
            "mean_abs_dur_err_pct": t_meta["mean_abs_dur_err_pct"],
            "median_abs_mid_err": t_meta["median_abs_mid_err"],
        })

        # Step C: Score each observed cell track against reference models
        cell_records = []
        for _, c_row in canon_cycles.iterrows():
            c_name = c_row[cell_col]
            if c_name not in cell_models or c_name not in temporal_atlas:
                continue

            m_info = cell_models[c_name]
            t_stat = temporal_atlas[c_name]
            tau_grid = m_info["time_grid"]
            tau_cutoff = float(m_info.get("tau_cutoff", tau_cutoffs.get(c_name, tau_grid[-1] if len(tau_grid) > 0 else 30.0)))
            if tau_cutoff <= 0:
                tau_cutoff = tau_grid[-1] if len(tau_grid) > 0 else 30.0

            emb_cell_df = aligned_q[aligned_q[cell_col] == c_name].sort_values(time_col)
            t_vals = emb_cell_df[time_col].values.astype(float)
            t_birth = float(t_vals.min())
            tau_vals = t_vals - t_birth
            valid_mask = tau_vals <= tau_cutoff + 1e-4
            if np.sum(valid_mask) < min_observations:
                continue

            tau_test = tau_vals[valid_mask]
            xyz = emb_cell_df[aligned_cols].values.astype(float)[valid_mask]
            com = np.mean(xyz, axis=0)
            xyz_shape = xyz - com

            # 1. Temporal Shape
            log_dur_obs = float(c_row["canon_log_dur"])
            z_temp_shape = float((log_dur_obs - t_stat["mu_log"]) / (t_stat["std_log"] + 1e-6))
            delta_dur_min = float(c_row["canon_duration"] - t_stat["mu_phys"])
            pct_dur_dev = float((np.exp(log_dur_obs - t_stat["mu_log"]) - 1.0) * 100.0)

            # 2. Temporal Shift (Birth Time Anchored)
            local_root, path_ancestors = get_ancestral_path_in_interval(
                c_name, parent_map, set(temporal_atlas.keys())
            )
            root_stat = temporal_atlas[local_root]
            ancestor_mus = [temporal_atlas[a]["mu_phys"] for a in path_ancestors]
            ancestor_vars = [temporal_atlas[a]["var_phys"] for a in path_ancestors]

            # Lineage-propagated birth expectation and variance
            e_birth_path = root_stat["mu_birth"] + sum(ancestor_mus)
            var_birth_path = root_stat["var_birth"] + sum(ancestor_vars)
            elapsed_birth_time = max(e_birth_path - root_stat["mu_birth"], 0.0)
            total_std_birth = np.sqrt(max(var_birth_path + (elapsed_birth_time ** 2) * var_tempo_estimation, 1e-4))

            canon_birth_obs = float(c_row["canon_birth"]) if "canon_birth" in c_row else t_birth
            delta_birth_min = float(canon_birth_obs - e_birth_path)
            z_temp_shift = float(delta_birth_min / total_std_birth) if total_std_birth > 0 else 0.0

            # Midpoint diagnostics (preserved for reporting)
            e_mid_path = e_birth_path + 0.5 * t_stat["mu_phys"]
            obs_mid = float(c_row["canon_mid"])
            delta_mid = float(obs_mid - e_mid_path)

            # 3. Spatial Shift (Mahalanobis COM via Joint GP B)
            diff_com = com - m_info["mu_com"]
            if m_info.get("B_cov") is not None:
                inv_B = np.linalg.pinv(m_info["B_cov"])
                d2_shift_raw = float(diff_com.T @ inv_B @ diff_com)
            else:
                d2_shift_raw = float(diff_com.T @ m_info["inv_cov_com"] @ diff_com)
            d_spat_shift = float(np.sqrt(max(d2_shift_raw, 0.0)))
            com_shift_um = float(np.linalg.norm(diff_com))

            # 4. Spatial Rotation (SO(3) alignment to consensus template)
            template_curve = m_info.get("template_curve")
            if local_trajectory_alignment and template_curve is not None:
                R_test, rot_angle_deg, coords_aligned = register_curve_to_template(
                    s_obs=tau_test,
                    coords_centered=xyz_shape,
                    template_curve=template_curve,
                    s_grid=tau_grid,
                )
            else:
                rot_angle_deg = 0.0
                coords_aligned = xyz_shape

            mu_rot_ref = float(m_info.get("mu_rot_deg", 0.0))
            std_rot_ref = float(m_info.get("std_rot_deg", 1.0))
            z_rot_angle = float((rot_angle_deg - mu_rot_ref) / (std_rot_ref + 1e-6))

            # 5. Regularized Monotonic Time Warping in Rotated Physical Space
            target_curve = template_curve if template_curve is not None else np.zeros((len(tau_grid), 3))
            gamma_test, rms_warp_min, slopes_test = regularized_monotonic_time_warp(
                tau_obs=tau_test,
                coords_aligned=coords_aligned,
                template_curve=target_curve,
                tau_grid=tau_grid,
                lambda_reg=warping_lambda,
                slope_bounds=warping_slope_bounds,
            )
            active_tau_test = tau_grid[tau_grid <= tau_test.max() + 1e-4]
            warp_met = compute_warp_metrics(gamma_test, active_tau_test, slopes_test)
            max_warp_min = warp_met["max_warp_min"]
            signed_warp_area = warp_met["signed_warp_area"]

            mu_warp_ref = float(m_info.get("mu_warp_min", m_info.get("mu_fr", 0.0)))
            std_warp_ref = float(m_info.get("std_warp_min", m_info.get("std_fr", 1.0)))
            z_warp = float((rms_warp_min - mu_warp_ref) / (std_warp_ref + 1e-6))

            # 6. Spatial Shape Residuals in Warped Progression Coordinates
            u_eval = np.clip(gamma_test / max(tau_cutoff, 1e-3), 0.0, 1.0)
            xyz_eval = np.column_stack([np.interp(gamma_test, tau_test, coords_aligned[:, d]) for d in range(3)])

            if m_info.get("dense_cov_3d") is not None and m_info.get("dense_mu_3d") is not None:
                grid_u = m_info["s_dense"]
                pred_mu_xyz = interp1d(
                    grid_u, m_info["dense_mu_3d"], axis=0, kind="linear", fill_value="extrapolate"
                )(u_eval)
                pred_cov_3d = interp1d(
                    grid_u, m_info["dense_cov_3d"], axis=0, kind="linear", fill_value="extrapolate"
                )(u_eval)

                d_spat_shape, rmse_3d_um = compute_3d_mahalanobis_residuals(
                    xyz_eval, pred_mu_xyz, pred_cov_3d
                )
            else:
                delta2 = np.zeros(len(u_eval))
                pred_mu_xyz = np.zeros_like(xyz_eval)
                grid_u = m_info["s_dense"]

                for idx_col, col_name in enumerate(aligned_cols):
                    ref_mu = m_info["dense_pred"][col_name]["mu"]
                    pred_mu_xyz[:, idx_col] = interp1d(grid_u, ref_mu, kind="linear", fill_value="extrapolate")(u_eval)
                    ref_std = m_info["dense_pred"][col_name]["std"]
                    s_gp = interp1d(grid_u, ref_std, kind="linear", fill_value="extrapolate")(u_eval)
                    delta2 += ((xyz_eval[:, idx_col] - pred_mu_xyz[:, idx_col]) ** 2) / np.maximum(s_gp ** 2, 1e-4)

                d_spat_shape = float(np.sqrt(np.mean(delta2 / 3.0)))
                rmse_3d_um = float(np.sqrt(np.mean(np.sum((xyz_eval - pred_mu_xyz) ** 2, axis=1))))

            cell_records.append({
                "embryo_id": q_id,
                "cell": c_name,
                "is_spatial_inlier": s_meta["is_spatial_inlier_map"].get(c_name, False),
                "k_test": t_meta["k_test"],
                "dt0_test": t_meta["dt0_test"],
                "scale_s": s_meta["scale_s"],
                "z_temp_shape": z_temp_shape,
                "delta_duration_min": delta_dur_min,
                "pct_duration_deviation": pct_dur_dev,
                "canon_duration": float(c_row["canon_duration"]),
                "canon_mid": obs_mid,
                "z_temp_shift": z_temp_shift,
                "delta_midpoint_min": delta_mid,
                "delta_birth_min": delta_birth_min,
                "d_spat_shift": d_spat_shift,
                "com_shift_um": com_shift_um,
                "rot_angle_deg": rot_angle_deg,
                "z_rot_angle": z_rot_angle,
                "d_spat_shape": d_spat_shape,
                "rmse_3d_um": rmse_3d_um,
                "d_warp_fr_rad": rms_warp_min,
                "rms_warp_min": rms_warp_min,
                "z_warp": z_warp,
                "signed_warp_area": signed_warp_area,
                "max_warp_dist": max_warp_min,
                "max_warp_min": max_warp_min,
                "tau_cutoff": tau_cutoff,
            })


        if cell_records:
            emb_scored_df = pd.DataFrame(cell_records)
            emb_calibrated_df, _ = apply_empirical_calibration_to_inference(
                test_res_df=emb_scored_df,
                oof_null_df=oof_null_df,
                alpha=alpha,
            )
            all_cell_scores.append(emb_calibrated_df)

    cell_scores_df = (
        pd.concat(all_cell_scores, axis=0, ignore_index=True)
        if all_cell_scores
        else pd.DataFrame()
    )
    embryo_qc_df = pd.DataFrame(embryo_qc_records)
    return cell_scores_df, embryo_qc_df
