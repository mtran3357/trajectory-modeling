"""Stage 2: Inference pipeline for query / perturbed embryos."""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from ..geometry.align import align_embryo_to_spatial_template
from ..functional.srvf import curve_to_srvf
from ..functional.dp_warp import align_srvf_dp_clamped
from ..functional.metrics import calculate_fisher_rao_distance
from ..lineage.graph import parse_lineage_graph, get_ancestral_path_in_interval
from ..temporal.lifespans import extract_cell_lifespans
from ..temporal.register import register_embryo_to_temporal_atlas
from ..stats.calibration import apply_empirical_calibration_to_inference
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
            time_grid = m_info["time_grid"]

            emb_cell_df = aligned_q[aligned_q[cell_col] == c_name].sort_values(time_col)
            if len(emb_cell_df) < min_observations:
                continue

            t_vals = emb_cell_df[time_col].values.astype(float)
            t_min, t_max = t_vals.min(), t_vals.max()
            t_rel = (t_vals - t_min) / (t_max - t_min) if t_max > t_min else np.zeros_like(t_vals)

            xyz = emb_cell_df[aligned_cols].values.astype(float)
            com = np.mean(xyz, axis=0)
            xyz_shape = xyz - com

            curve_interp = interp1d(t_rel, xyz_shape, axis=0, kind="linear", fill_value="extrapolate")
            dense_curve = curve_interp(time_grid)
            srvf = curve_to_srvf(dense_curve, time_grid)

            # 1. Temporal Shape
            log_dur_obs = float(c_row["canon_log_dur"])
            z_temp_shape = float((log_dur_obs - t_stat["mu_log"]) / (t_stat["std_log"] + 1e-6))
            delta_dur_min = float(c_row["canon_duration"] - t_stat["mu_phys"])
            pct_dur_dev = float((np.exp(log_dur_obs - t_stat["mu_log"]) - 1.0) * 100.0)

            # 2. Temporal Shift
            local_root, path_ancestors = get_ancestral_path_in_interval(
                c_name, parent_map, set(temporal_atlas.keys())
            )
            root_stat = temporal_atlas[local_root]
            ancestor_mus = [temporal_atlas[a]["mu_phys"] for a in path_ancestors]
            ancestor_vars = [temporal_atlas[a]["var_phys"] for a in path_ancestors]
            e_mid_path = root_stat["mu_birth"] + sum(ancestor_mus) + 0.5 * t_stat["mu_phys"]
            var_path = root_stat["var_birth"] + sum(ancestor_vars) + 0.25 * t_stat["var_phys"]
            elapsed_time = max(e_mid_path - root_stat["mu_birth"], 0.0)
            total_std_shift = np.sqrt(max(var_path + (elapsed_time ** 2) * var_tempo_estimation, 1e-4))
            obs_mid = float(c_row["canon_mid"])
            delta_mid = float(obs_mid - e_mid_path)
            z_temp_shift = float(delta_mid / total_std_shift) if total_std_shift > 0 else 0.0

            # 3. Spatial Shift
            diff_com = com - m_info["mu_com"]
            d2_shift_raw = float(diff_com.T @ m_info["inv_cov_com"] @ diff_com)
            d_spat_shift = float(np.sqrt(max(d2_shift_raw, 0.0)))
            com_shift_um = float(np.linalg.norm(diff_com))

            # 4. Spatial Shape (Interpolated from precomputed ribbon)
            gamma_test = align_srvf_dp_clamped(m_info["mu_srvf"], srvf, time_grid)
            g_interp = interp1d(time_grid, gamma_test, kind="linear", fill_value="extrapolate")
            s_obs = np.clip(g_interp(t_rel), 0.0, 1.0)

            delta2 = np.zeros(len(emb_cell_df))
            pred_mu_xyz = np.zeros_like(xyz_shape)

            for idx_col, col_name in enumerate(aligned_cols):
                grid_s = m_info["s_dense"]
                ref_mu = m_info["dense_pred"][col_name]["mu"]
                ref_std = m_info["dense_pred"][col_name]["std"]

                m_gp = interp1d(grid_s, ref_mu, kind="linear", fill_value="extrapolate")(s_obs)
                s_gp = interp1d(grid_s, ref_std, kind="linear", fill_value="extrapolate")(s_obs)

                pred_mu_xyz[:, idx_col] = m_gp
                delta2 += ((xyz_shape[:, idx_col] - m_gp) ** 2) / np.maximum(s_gp ** 2, 1e-4)

            d_spat_shape = float(np.sqrt(np.mean(delta2 / 3.0)))
            rmse_3d_um = float(np.sqrt(np.mean(np.sum((xyz_shape - pred_mu_xyz) ** 2, axis=1))))

            # 5. Warp
            d_fr_test = max(calculate_fisher_rao_distance(gamma_test, time_grid), 1e-4)
            z_warp = float((d_fr_test - m_info["mu_fr"]) / (m_info["std_fr"] + 1e-6))
            signed_warp_area = float(np.trapezoid(gamma_test - time_grid, time_grid))
            max_warp_dist = float(np.max(np.abs(gamma_test - time_grid)))

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
                "d_spat_shift": d_spat_shift,
                "com_shift_um": com_shift_um,
                "d_spat_shape": d_spat_shape,
                "rmse_3d_um": rmse_3d_um,
                "d_warp_fr_rad": d_fr_test,
                "z_warp": z_warp,
                "signed_warp_area": signed_warp_area,
                "max_warp_dist": max_warp_dist,
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
