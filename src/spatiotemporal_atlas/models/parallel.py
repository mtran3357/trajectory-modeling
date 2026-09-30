"""Cell-wise parallel Gaussian Process and SRVF trajectory fitting and scoring."""

import joblib
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm.auto import tqdm

from ..functional.srvf import curve_to_srvf
from ..functional.dp_warp import align_srvf_dp_clamped
from ..functional.karcher_mean import compute_karcher_mean_srvf
from ..functional.metrics import calculate_fisher_rao_distance
from ..lineage.graph import parse_lineage_graph, get_ancestral_path_in_interval
from ..temporal.lifespans import extract_cell_lifespans
from ..temporal.atlas import build_canonical_temporal_atlas
from ..temporal.register import register_embryo_to_temporal_atlas
from .gp import fit_coordinate_gps, extract_trajectory_ribbon
from .joint_gp import (
    fit_kronecker_joint_gp,
    predict_kronecker_joint_gp,
    compute_3d_mahalanobis_residuals,
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
    time_grid: np.ndarray,
    min_observations: int,
    min_train_embryos: int,
    optimizer: str | None,
    length_scale: float,
    noise_level: float,
    random_state: int,
    return_models: bool,
    model_type: str = "joint_gp",
    n_dense_samples: int = 100,
) -> tuple[list[dict], dict | None]:
    """Fits analytical GP + SRVF models for one blastomere and scores test tracks."""
    embs_present = set(sub_c[embryo_col].unique())
    train_embs = [
        e for e in training_embryos
        if e in embs_present and e in canon_train_df[embryo_col].values
    ]

    if len(train_embs) < min_train_embryos:
        return [], None

    train_tracks = {}
    for emb in train_embs:
        emb_df = sub_c[sub_c[embryo_col] == emb].sort_values(time_col)
        if len(emb_df) < min_observations:
            continue

        t_vals = emb_df[time_col].values.astype(float)
        t_min, t_max = t_vals.min(), t_vals.max()
        t_rel = (t_vals - t_min) / (t_max - t_min) if t_max > t_min else np.zeros_like(t_vals)

        xyz = emb_df[spatial_cols].values.astype(float)
        com = np.mean(xyz, axis=0)
        xyz_shape = xyz - com

        curve_interp = interp1d(t_rel, xyz_shape, axis=0, kind="linear", fill_value="extrapolate")
        dense_curve = curve_interp(time_grid)
        srvf = curve_to_srvf(dense_curve, time_grid)

        train_tracks[emb] = {
            "t_rel": t_rel,
            "xyz_shape": xyz_shape,
            "com_xyz": com,
            "srvf": srvf,
            "n_frames": len(emb_df),
        }

    if len(train_tracks) < min_train_embryos:
        return [], None

    train_srvfs = [v["srvf"] for v in train_tracks.values()]
    train_coms = np.array([v["com_xyz"] for v in train_tracks.values()])
    M_train = len(train_srvfs)

    mu_srvf, train_gammas = compute_karcher_mean_srvf(train_srvfs, time_grid)

    s_warped_all = []
    xyz_shape_all = []
    train_fr_distances = []

    for i, (emb, data) in enumerate(train_tracks.items()):
        gamma_e = train_gammas[i]
        gamma_interp = interp1d(time_grid, gamma_e, kind="linear", fill_value="extrapolate")
        s_obs = np.clip(gamma_interp(data["t_rel"]), 0.0, 1.0)
        s_warped_all.extend(s_obs)
        xyz_shape_all.extend(data["xyz_shape"])
        d_fr = max(calculate_fisher_rao_distance(gamma_e, time_grid), 1e-4)
        train_fr_distances.append(d_fr)

    X_warped = np.array(s_warped_all).reshape(-1, 1)
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
    if len(train_coms) > 2:
        cov_com = np.cov(train_coms, rowvar=False) + np.eye(3) * 0.25
    elif len(train_coms) == 2:
        diff = train_coms[0] - train_coms[1]
        cov_com = np.outer(diff, diff) + np.eye(3) * 0.25
    else:
        cov_com = np.eye(3) * 0.25

    inv_cov_com = np.linalg.pinv(cov_com)
    mu_fr = float(np.mean(train_fr_distances))
    std_fr = float(np.std(train_fr_distances, ddof=1)) if M_train > 1 else 0.05

    factor_pred = 1.0 + (1.0 / M_train)
    factor_pred_1d = np.sqrt(factor_pred)
    t_stat = temporal_atlas[c]

    cell_records = []

    # Score held-out test tracks
    for te_id in test_embryo_ids:
        if te_id not in canon_test_by_embryo:
            continue
        canon_te_df = canon_test_by_embryo[te_id]
        te_cell_row = canon_te_df[canon_te_df[cell_col] == c]
        if te_cell_row.empty:
            continue

        emb_df = sub_c[sub_c[embryo_col] == te_id].sort_values(time_col)
        if len(emb_df) < min_observations:
            continue

        t_vals = emb_df[time_col].values.astype(float)
        t_min, t_max = t_vals.min(), t_vals.max()
        t_rel = (t_vals - t_min) / (t_max - t_min) if t_max > t_min else np.zeros_like(t_vals)

        xyz = emb_df[spatial_cols].values.astype(float)
        com = np.mean(xyz, axis=0)
        xyz_shape = xyz - com

        curve_interp = interp1d(t_rel, xyz_shape, axis=0, kind="linear", fill_value="extrapolate")
        dense_curve = curve_interp(time_grid)
        srvf = curve_to_srvf(dense_curve, time_grid)

        # 1. Temporal Shape
        log_dur_obs = float(te_cell_row["canon_log_dur"].iloc[0])
        z_temp_shape = float((log_dur_obs - t_stat["mu_log"]) / (t_stat["std_log"] * factor_pred_1d))
        delta_dur_min = float(te_cell_row["canon_duration"].iloc[0] - t_stat["mu_phys"])
        pct_dur_dev = float((np.exp(log_dur_obs - t_stat["mu_log"]) - 1.0) * 100.0)

        # 2. Temporal Shift
        local_root, path_ancestors = get_ancestral_path_in_interval(
            c, parent_map, set(temporal_atlas.keys())
        )
        root_stat = temporal_atlas[local_root]
        ancestor_mus = [temporal_atlas[a]["mu_phys"] for a in path_ancestors]
        ancestor_vars = [temporal_atlas[a]["var_phys"] for a in path_ancestors]
        e_mid_path = root_stat["mu_birth"] + sum(ancestor_mus) + 0.5 * t_stat["mu_phys"]
        var_path = root_stat["var_birth"] + sum(ancestor_vars) + 0.25 * t_stat["var_phys"]
        elapsed_time = max(e_mid_path - root_stat["mu_birth"], 0.0)
        total_std_shift = np.sqrt(max(var_path + (elapsed_time ** 2) * var_tempo_estimation, 1e-4))
        obs_mid = float(te_cell_row["canon_mid"].iloc[0])
        delta_mid = float(obs_mid - e_mid_path)
        z_temp_shift = float(delta_mid / total_std_shift) if total_std_shift > 0 else 0.0

        # 3. Spatial Shift
        diff_com = com - mu_com
        d2_shift_raw = float(diff_com.T @ inv_cov_com @ diff_com)
        d_spat_shift = float(np.sqrt(max(d2_shift_raw / factor_pred, 0.0)))
        com_shift_um = float(np.linalg.norm(diff_com))

        # 4. Spatial Shape
        gamma_test = align_srvf_dp_clamped(mu_srvf, srvf, time_grid)
        g_interp = interp1d(time_grid, gamma_test, kind="linear", fill_value="extrapolate")
        s_obs = np.clip(g_interp(t_rel), 0.0, 1.0)

        if model_type == "joint_gp" and joint_model is not None:
            pred_mu_xyz, _, cov_3d_test = predict_kronecker_joint_gp(joint_model, s_obs)
            d_spat_shape, rmse_3d_um = compute_3d_mahalanobis_residuals(xyz_shape, pred_mu_xyz, cov_3d_test)
        else:
            delta2 = np.zeros(len(emb_df))
            pred_mu_xyz = np.zeros_like(xyz_shape)
            s_obs_col = s_obs.reshape(-1, 1)
            for idx_col, col_name in enumerate(spatial_cols):
                m_gp, s_gp = gps[col_name].predict(s_obs_col, return_std=True)
                pred_mu_xyz[:, idx_col] = m_gp
                delta2 += ((xyz_shape[:, idx_col] - m_gp) ** 2) / np.maximum(s_gp ** 2, 1e-4)

            d_spat_shape = float(np.sqrt(np.mean(delta2 / 3.0)))
            rmse_3d_um = float(np.sqrt(np.mean(np.sum((xyz_shape - pred_mu_xyz) ** 2, axis=1))))

        # 5. Warp
        d_fr_test = max(calculate_fisher_rao_distance(gamma_test, time_grid), 1e-4)
        z_warp = float((d_fr_test - mu_fr) / (std_fr * np.sqrt(factor_pred)))
        signed_warp_area = float(np.trapezoid(gamma_test - time_grid, time_grid))
        max_warp_dist = float(np.max(np.abs(gamma_test - time_grid)))

        cell_records.append({
            embryo_col: te_id,
            cell_col: c,
            "z_temp_shape": z_temp_shape,
            "delta_duration_min": delta_dur_min,
            "pct_duration_deviation": pct_dur_dev,
            "canon_duration": float(te_cell_row["canon_duration"].iloc[0]),
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
                mu_fr=mu_fr,
                std_fr=std_fr,
                mu_srvf=mu_srvf,
                time_grid=time_grid,
                n_dense_samples=n_dense_samples,
            )
        else:
            ribbon = extract_trajectory_ribbon(
                cell_name=c,
                gps=gps,
                spatial_cols=spatial_cols,
                mu_com=mu_com,
                inv_cov_com=inv_cov_com,
                mu_fr=mu_fr,
                std_fr=std_fr,
                mu_srvf=mu_srvf,
                time_grid=time_grid,
                n_dense_samples=n_dense_samples,
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
) -> tuple[pd.DataFrame, dict, dict]:
    """Coordinates parallel worker execution across blastomeres."""
    time_grid = np.linspace(0.0, 1.0, grid_points)
    parent_map = parse_lineage_graph(lineage_df)

    pos_work = pos_df.copy()
    pos_work[embryo_col] = pos_work[embryo_col].astype(str)
    pos_work[cell_col] = pos_work[cell_col].astype(str).str.strip()

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
            canon_test_by_embryo[te_id] = c_df
        except ValueError:
            continue

    unique_cells = list(canon_train_df[cell_col].unique())
    cell_subsets = {c: pos_work[pos_work[cell_col] == c].copy() for c in unique_cells}

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
            time_grid=time_grid,
            min_observations=min_observations,
            min_train_embryos=min_train_embryos,
            optimizer=optimizer,
            length_scale=length_scale,
            noise_level=noise_level,
            random_state=random_state,
            return_models=return_models,
            model_type=model_type,
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
        {"temporal_atlas": temporal_atlas, "reg_meta_by_test": reg_meta_by_test},
    )
