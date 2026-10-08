"""Cell-wise parallel Gaussian Process and Regularized Time-Warping trajectory fitting and scoring."""

import joblib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..functional.time_warp import regularized_monotonic_time_warp, compute_warp_metrics
from ..geometry.curve_align import (
    interpolate_curve_to_grid,
    masked_generalized_procrustes,
    register_curve_to_template,
)
from ..lineage.graph import parse_lineage_graph, get_ancestral_path_in_interval
from ..temporal.lifespans import extract_cell_lifespans
from ..temporal.atlas import build_canonical_temporal_atlas
from ..temporal.register import register_embryo_to_temporal_atlas
from .gp import fit_coordinate_gps, extract_trajectory_ribbon
from .joint_gp import (
    fit_kronecker_joint_gp,
    predict_kronecker_joint_gp,
    compute_3d_mahalanobis_residuals,
    compute_full_joint_mahalanobis_residuals,
    extract_joint_trajectory_ribbon,
)


def _fit_and_score_single_cell_worker(
    c: str,
    sub_c: pd.DataFrame,
    training_embryos: list[str],
    test_embryo_ids: list[str],
    canon_train_df: pd.DataFrame,
    canon_test_by_embryo: dict,
    temporal_atlas: dict,
    parent_map: dict,
    var_tempo_estimation: float,
    spatial_cols: list[str],
    time_col: str,
    embryo_col: str,
    cell_col: str,
    grid_points: int,
    tau_cutoff: float,
    min_observations: int,
    min_train_embryos: int,
    optimizer: str | None,
    length_scale: float,
    noise_level: float,
    random_state: int,
    return_models: bool,
    model_type: str = "joint_gp",
    n_dense_samples: int = 100,
    local_trajectory_alignment: bool = True,
    warping_lambda: float = 1.0,
    warping_slope_bounds: tuple[float, float] = (0.5, 2.0),
    shape_metric_mode: str = "pointwise_marginal",
) -> tuple[list[dict], dict | None]:
    """Fits analytical GP + regularized time-warping models for one blastomere and scores test tracks."""
    embs_present = set(sub_c[embryo_col].unique())
    train_embs = [
        e for e in training_embryos
        if e in embs_present and e in canon_train_df[embryo_col].values
    ]

    if len(train_embs) < min_train_embryos:
        return [], None

    train_ke_map = (
        canon_train_df.groupby(embryo_col)["k_e"].first().to_dict()
        if "k_e" in canon_train_df.columns
        else {}
    )
    train_tracks = {}
    for emb in train_embs:
        emb_df = sub_c[sub_c[embryo_col] == emb].sort_values(time_col)
        t_vals = emb_df[time_col].values.astype(float)
        t_birth = float(t_vals.min())
        k_e = float(train_ke_map.get(emb, 1.0))
        tau_vals = (t_vals - t_birth) / k_e
        valid_mask = tau_vals <= tau_cutoff + 1e-4
        if np.sum(valid_mask) < min_observations:
            continue

        tau_obs = tau_vals[valid_mask]
        xyz = emb_df[spatial_cols].values.astype(float)[valid_mask]
        com = np.mean(xyz, axis=0)
        xyz_shape = xyz - com

        train_tracks[emb] = {
            "tau_obs": tau_obs,
            "xyz_shape": xyz_shape,
            "com_xyz": com,
            "t_birth": t_birth,
            "k_e": k_e,
            "n_frames": len(tau_obs),
        }

    if len(train_tracks) < min_train_embryos:
        return [], None

    train_coms = np.array([v["com_xyz"] for v in train_tracks.values()])
    M_train = len(train_tracks)

    # Evaluation nodes in physical minutes [0, tau_cutoff]
    tau_grid = np.linspace(0.0, tau_cutoff, grid_points)

    # Local trajectory rigid alignment (Masked GPA)
    tracks_dict = {emb: (data["tau_obs"], data["xyz_shape"]) for emb, data in train_tracks.items()}
    if local_trajectory_alignment and len(tracks_dict) >= 2:
        gpa_res = masked_generalized_procrustes(tracks_dict, tau_grid=tau_grid)
        aligned_train_dict = gpa_res["aligned_trajectories"]
        mu_rot_train = gpa_res["mean_angle_deg"]
        std_rot_train = gpa_res["std_angle_deg"]
        template_curve = gpa_res["template_curve"]
    else:
        aligned_train_dict = tracks_dict
        mu_rot_train = 0.0
        std_rot_train = 1.0
        res_list = [interpolate_curve_to_grid(d["tau_obs"], d["xyz_shape"], tau_grid) for d in train_tracks.values()]
        template_curve = np.mean(res_list, axis=0)

    # Regularized monotonic time-warping in rotated physical space
    u_warped_all = []
    xyz_shape_all = []
    train_warp_rms = []

    for emb, data in train_tracks.items():
        tau_obs, coords_aligned = aligned_train_dict[emb]
        gamma_e, rms_w, _ = regularized_monotonic_time_warp(
            tau_obs=tau_obs,
            coords_aligned=coords_aligned,
            template_curve=template_curve,
            tau_grid=tau_grid,
            lambda_reg=warping_lambda,
            slope_bounds=warping_slope_bounds,
        )
        train_warp_rms.append(rms_w)
        # Normalize time coordinate by tau_cutoff for GP modeling: u in [0, 1]
        u_obs = np.clip(gamma_e / max(tau_cutoff, 1e-3), 0.0, 1.0)
        c_warped = np.column_stack([np.interp(gamma_e, tau_obs, coords_aligned[:, d]) for d in range(3)])
        u_warped_all.extend(u_obs)
        xyz_shape_all.extend(c_warped)

    X_warped = np.array(u_warped_all).reshape(-1, 1)
    Y_shape = np.array(xyz_shape_all)

    # Fit trajectory model: Joint 3D Kronecker GP (default) or Independent 1D GPs
    if model_type == "joint_gp":
        joint_model = fit_kronecker_joint_gp(
            s_obs=X_warped,
            Y_shape=Y_shape,
            length_scale=length_scale,
            noise_level=noise_level,
        )
        gps = None
    else:
        joint_model = None
        gps = fit_coordinate_gps(
            X_warped=X_warped,
            Y_shape=Y_shape,
            spatial_cols=spatial_cols,
            length_scale=length_scale,
            noise_level=noise_level,
            optimizer=optimizer,
            random_state=random_state,
        )

    mu_com = np.mean(train_coms, axis=0)
    if model_type == "joint_gp" and joint_model is not None and "B" in joint_model:
        inv_cov_com = np.linalg.pinv(joint_model["B"])
    elif len(train_coms) > 2:
        cov_com = np.cov(train_coms, rowvar=False) + np.eye(3) * 0.25
        inv_cov_com = np.linalg.pinv(cov_com)
    elif len(train_coms) == 2:
        diff = train_coms[0] - train_coms[1]
        cov_com = np.outer(diff, diff) + np.eye(3) * 0.25
        inv_cov_com = np.linalg.pinv(cov_com)
    else:
        cov_com = np.eye(3) * 0.25
        inv_cov_com = np.linalg.pinv(cov_com)
    mu_warp = float(np.mean(train_warp_rms)) if train_warp_rms else 0.0
    std_warp = float(np.std(train_warp_rms, ddof=1)) if len(train_warp_rms) > 1 else 0.1

    factor_pred = 1.0 + (1.0 / M_train)
    factor_pred_1d = np.sqrt(factor_pred)
    t_stat = temporal_atlas[c]

    cell_records = []

    # Score held-out test tracks
    for te_id in test_embryo_ids:
        if te_id not in canon_test_by_embryo:
            continue
        canon_test_df = canon_test_by_embryo[te_id]
        te_cell_row = canon_test_df[canon_test_df[cell_col] == c]
        if te_cell_row.empty:
            continue

        emb_df = sub_c[sub_c[embryo_col] == te_id].sort_values(time_col)
        t_vals = emb_df[time_col].values.astype(float)
        t_birth = float(t_vals.min())
        k_test = float(te_cell_row["k_e"].iloc[0]) if "k_e" in te_cell_row.columns else 1.0
        tau_vals = (t_vals - t_birth) / k_test
        valid_mask = tau_vals <= tau_cutoff + 1e-4
        if np.sum(valid_mask) < min_observations:
            continue

        tau_test = tau_vals[valid_mask]
        xyz_test = emb_df[spatial_cols].values.astype(float)[valid_mask]
        com_test = np.mean(xyz_test, axis=0)
        xyz_shape_test = xyz_test - com_test

        # 1. Temporal Shape
        log_dur_obs = float(te_cell_row["canon_log_dur"].iloc[0])
        z_temp_shape = float((log_dur_obs - t_stat["mu_log"]) / (t_stat["std_log"] * factor_pred_1d))
        delta_dur_min = float(te_cell_row["canon_duration"].iloc[0] - t_stat["mu_phys"])
        pct_dur_dev = float((np.exp(log_dur_obs - t_stat["mu_log"]) - 1.0) * 100.0)

        # 2. Temporal Shift (Birth Time Anchored)
        local_root, path_ancestors = get_ancestral_path_in_interval(
            c, parent_map, set(temporal_atlas.keys())
        )
        root_stat = temporal_atlas[local_root]
        ancestor_mus = [temporal_atlas[a]["mu_phys"] for a in path_ancestors]
        ancestor_vars = [temporal_atlas[a]["var_phys"] for a in path_ancestors]

        # Lineage-propagated birth expectation and variance
        e_birth_path = root_stat["mu_birth"] + sum(ancestor_mus)
        var_birth_path = root_stat["var_birth"] + sum(ancestor_vars)
        elapsed_birth_time = max(e_birth_path - root_stat["mu_birth"], 0.0)
        total_std_birth = np.sqrt(max(var_birth_path + (elapsed_birth_time ** 2) * var_tempo_estimation, 1e-4))

        canon_birth_obs = float(te_cell_row["canon_birth"].iloc[0]) if "canon_birth" in te_cell_row.columns else t_birth
        delta_birth_min = float(canon_birth_obs - e_birth_path)
        z_temp_shift = float(delta_birth_min / (total_std_birth * factor_pred_1d)) if total_std_birth > 0 else 0.0

        # Midpoint diagnostics (preserved for reporting)
        e_mid_path = e_birth_path + 0.5 * t_stat["mu_phys"]
        obs_mid = float(te_cell_row["canon_mid"].iloc[0])
        delta_mid = float(obs_mid - e_mid_path)

        # 3. Spatial Shift
        diff_com = com_test - mu_com
        d2_shift_raw = float(diff_com.T @ inv_cov_com @ diff_com)
        d_spat_shift = float(np.sqrt(max(d2_shift_raw / factor_pred, 0.0)))
        com_shift_um = float(np.linalg.norm(diff_com))

        # 4. Spatial Rotation
        if local_trajectory_alignment and template_curve is not None:
            R_test, rot_angle_deg, coords_aligned = register_curve_to_template(
                s_obs=tau_test,
                coords_centered=xyz_shape_test,
                template_curve=template_curve,
                s_grid=tau_grid,
            )
        else:
            R_test = np.eye(3)
            rot_angle_deg = 0.0
            coords_aligned = xyz_shape_test

        z_rot_angle = float((rot_angle_deg - mu_rot_train) / (std_rot_train * factor_pred_1d))

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
        z_warp = float((rms_warp_min - mu_warp) / max(std_warp * np.sqrt(factor_pred), 1e-6))

        # 6. Spatial Shape Residuals in Warped Progression Coordinates
        u_eval = np.clip(gamma_test / max(tau_cutoff, 1e-3), 0.0, 1.0)
        xyz_eval = np.column_stack([np.interp(gamma_test, tau_test, coords_aligned[:, d]) for d in range(3)])

        if model_type == "joint_gp" and joint_model is not None:
            if shape_metric_mode == "full_joint_gp":
                pred_mu_xyz, _, _ = predict_kronecker_joint_gp(joint_model, u_eval, return_cov=False)
                d_spat_shape, rmse_3d_um = compute_full_joint_mahalanobis_residuals(
                    Y_observed=xyz_eval,
                    mu_pred=pred_mu_xyz,
                    B=joint_model["B"],
                    u_eval=u_eval,
                    length_scale=length_scale,
                    noise_level=noise_level,
                )
            else:
                pred_mu_xyz, _, cov_3d_test = predict_kronecker_joint_gp(joint_model, u_eval)
                d_spat_shape, rmse_3d_um = compute_3d_mahalanobis_residuals(xyz_eval, pred_mu_xyz, cov_3d_test)
        else:
            delta2 = np.zeros(len(u_eval))
            pred_mu_xyz = np.zeros_like(xyz_eval)
            u_eval_col = u_eval.reshape(-1, 1)
            for idx_col, col_name in enumerate(spatial_cols):
                m_gp, s_gp = gps[col_name].predict(u_eval_col, return_std=True)
                pred_mu_xyz[:, idx_col] = m_gp
                delta2 += ((xyz_eval[:, idx_col] - m_gp) ** 2) / np.maximum(s_gp ** 2, 1e-4)

            d_spat_shape = float(np.sqrt(np.mean(delta2 / 3.0)))
            rmse_3d_um = float(np.sqrt(np.mean(np.sum((xyz_eval - pred_mu_xyz) ** 2, axis=1))))

        cell_records.append({
            embryo_col: te_id,
            cell_col: c,
            "k_test": k_test,
            "z_temp_shape": z_temp_shape,
            "delta_duration_min": delta_dur_min,
            "pct_duration_deviation": pct_dur_dev,
            "canon_duration": float(te_cell_row["canon_duration"].iloc[0]),
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

    # STRIP MEMORY: Pre-evaluate ribbons and discard internal GP objects before returning
    lightweight_model = None
    if return_models:
        if model_type == "joint_gp" and joint_model is not None:
            ribbon = extract_joint_trajectory_ribbon(
                cell_name=c,
                joint_model=joint_model,
                spatial_cols=spatial_cols,
                mu_com=mu_com,
                inv_cov_com=inv_cov_com,
                mu_fr=0.0,
                std_fr=1.0,
                mu_srvf=np.zeros((1, 3)),
                time_grid=tau_grid,
                n_dense_samples=n_dense_samples,
                mu_rot_deg=mu_rot_train,
                std_rot_deg=std_rot_train,
                template_curve=template_curve,
                tau_cutoff=tau_cutoff,
                mu_warp_min=mu_warp,
                std_warp_min=std_warp,
            )
        else:
            ribbon = extract_trajectory_ribbon(
                cell_name=c,
                gps=gps,
                spatial_cols=spatial_cols,
                mu_com=mu_com,
                inv_cov_com=inv_cov_com,
                mu_fr=0.0,
                std_fr=1.0,
                mu_srvf=np.zeros((1, 3)),
                time_grid=tau_grid,
                n_dense_samples=n_dense_samples,
                mu_rot_deg=mu_rot_train,
                std_rot_deg=std_rot_train,
                template_curve=template_curve,
                tau_cutoff=tau_cutoff,
                mu_warp_min=mu_warp,
                std_warp_min=std_warp,
            )
        lightweight_model = ribbon.to_dict()

    del gps, joint_model, X_warped, Y_shape, train_tracks
    return cell_records, lightweight_model


def score_embryos_batch_parallel(
    test_embryo_ids: list[str],
    pos_df: pd.DataFrame,
    lineage_df: pd.DataFrame,
    training_embryos: list[str],
    spatial_cols: list[str],
    time_col: str,
    embryo_col: str,
    cell_col: str,
    grid_points: int = 40,
    min_observations: int = 3,
    min_train_embryos: int = 2,
    optimizer: str | None = None,
    length_scale: float = 0.3,
    noise_level: float = 1.0,
    model_type: str = "joint_gp",
    random_state: int = 42,
    return_models: bool = True,
    n_jobs: int = 4,
    desc: str = "Fitting Cells",
    max_inlier_dist_canon: float = 5.0,
    local_trajectory_alignment: bool = True,
    warping_lambda: float = 1.0,
    warping_slope_bounds: tuple[float, float] = (0.5, 2.0),
    tau_percentile_cutoff: float = 95.0,
    cycles_df: pd.DataFrame | None = None,
    shape_metric_mode: str = "pointwise_marginal",
) -> tuple[pd.DataFrame, dict, dict]:
    """Coordinates parallel worker execution across blastomeres."""
    parent_map = parse_lineage_graph(lineage_df)

    pos_work = pos_df.copy()
    pos_work[embryo_col] = pos_work[embryo_col].astype(str)
    pos_work[cell_col] = pos_work[cell_col].astype(str).str.strip()

    if cycles_df is not None:
        all_cycles = cycles_df.copy()
    else:
        all_cycles = extract_cell_lifespans(
            pos_work, time_col=time_col, embryo_col=embryo_col, cell_col=cell_col
        )
    all_cycles[embryo_col] = all_cycles[embryo_col].astype(str)
    all_cycles[cell_col] = all_cycles[cell_col].astype(str).str.strip()

    training_embryos = [str(e) for e in training_embryos]
    test_embryo_ids = [str(e) for e in test_embryo_ids]

    train_cycles = all_cycles[all_cycles[embryo_col].isin(training_embryos)]
    test_cycles_all = all_cycles[all_cycles[embryo_col].isin(test_embryo_ids)]

    if train_cycles.empty:
        return pd.DataFrame(), {}, {}

    canon_train_df, temporal_atlas, _ = build_canonical_temporal_atlas(
        train_cycles,
        cell_col=cell_col,
        embryo_col=embryo_col,
        max_inlier_dist_canon=max_inlier_dist_canon,
    )
    if canon_train_df.empty or not temporal_atlas:
        return pd.DataFrame(), {}, {}

    # Calculate empirical tau_cutoff (95th percentile) per cell type from WT training cohort in canonical minutes
    tau_cutoffs = {}
    for cell_name, grp in canon_train_df.groupby(cell_col):
        durs = grp["canon_duration"].dropna().values.astype(float)
        if len(durs) >= 5:
            cutoff = float(np.percentile(durs, tau_percentile_cutoff))
        elif len(durs) > 0:
            cutoff = float(np.max(durs))
        else:
            cutoff = 30.0
        tau_cutoffs[cell_name] = max(cutoff, 1.0)

    for cell_name in temporal_atlas.keys():
        if cell_name not in tau_cutoffs:
            tau_cutoffs[cell_name] = max(float(temporal_atlas[cell_name].get("mu_phys", 30.0)), 1.0)
        temporal_atlas[cell_name]["tau_cutoff"] = tau_cutoffs[cell_name]

    pooled_log_stds = [v["std_log"] for v in temporal_atlas.values() if v["std_log"] > 0]
    sigma_log_pool = float(np.median(pooled_log_stds)) if pooled_log_stds else 0.05
    n_anchor_cells = max(len(temporal_atlas), 1)
    var_tempo_estimation = (sigma_log_pool ** 2) / n_anchor_cells

    reg_meta_by_test = {}
    canon_test_by_embryo = {}
    for te_id in test_embryo_ids:
        te_cyc = test_cycles_all[test_cycles_all[embryo_col] == te_id]
        if te_cyc.empty:
            continue
        try:
            c_df, r_meta = register_embryo_to_temporal_atlas(
                te_cyc,
                temporal_atlas,
                cell_col=cell_col,
                max_inlier_dist_canon=max_inlier_dist_canon,
            )
            reg_meta_by_test[te_id] = r_meta
            c_df["k_e"] = r_meta["k_test"]
            canon_test_by_embryo[te_id] = c_df
        except ValueError:
            continue

    unique_cells = list(canon_train_df[cell_col].unique())
    req_cols = [embryo_col, cell_col, time_col] + list(spatial_cols)
    cell_subsets = {c: pos_work.loc[pos_work[cell_col] == c, req_cols].copy() for c in unique_cells}

    tasks = (
        joblib.delayed(_fit_and_score_single_cell_worker)(
            c=c,
            sub_c=cell_subsets[c],
            training_embryos=training_embryos,
            test_embryo_ids=test_embryo_ids,
            canon_train_df=canon_train_df,
            canon_test_by_embryo=canon_test_by_embryo,
            temporal_atlas=temporal_atlas,
            parent_map=parent_map,
            var_tempo_estimation=var_tempo_estimation,
            spatial_cols=spatial_cols,
            time_col=time_col,
            embryo_col=embryo_col,
            cell_col=cell_col,
            grid_points=grid_points,
            tau_cutoff=tau_cutoffs.get(c, 30.0),
            min_observations=min_observations,
            min_train_embryos=min_train_embryos,
            optimizer=optimizer,
            length_scale=length_scale,
            noise_level=noise_level,
            random_state=random_state,
            return_models=return_models,
            model_type=model_type,
            local_trajectory_alignment=local_trajectory_alignment,
            warping_lambda=warping_lambda,
            warping_slope_bounds=warping_slope_bounds,
            shape_metric_mode=shape_metric_mode,
        )
        for c in unique_cells
    )

    worker_results = joblib.Parallel(
        n_jobs=n_jobs,
        backend="loky",
        batch_size="auto",
        max_nbytes=None,
    )(tqdm(tasks, total=len(unique_cells), desc=desc, leave=False, unit="cell"))

    all_scored_records = []
    fitted_models = {}

    for records, model_dict in worker_results:
        if records:
            all_scored_records.extend(records)
        if model_dict is not None:
            fitted_models[model_dict["cell_name"]] = model_dict

    del cell_subsets, tasks, worker_results

    raw_test_df = pd.DataFrame(all_scored_records)
    return (
        raw_test_df,
        fitted_models,
        {
            "temporal_atlas": temporal_atlas,
            "reg_meta_by_test": reg_meta_by_test,
            "tau_cutoffs": tau_cutoffs,
        },
    )
