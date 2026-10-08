"""Single-cell trajectory diagnostic dashboard across the 6-modality spatiotemporal suite."""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    go = None
    make_subplots = None

from ..functional.time_warp import regularized_monotonic_time_warp, compute_warp_metrics
from ..geometry.curve_align import register_curve_to_template
from ..models.joint_gp import compute_3d_mahalanobis_residuals
from .colors import get_canonical_clade_color


def _prepare_trajectory_dashboard_data(
    cell_name: str,
    embryo_id: str,
    pos_df: pd.DataFrame,
    target_row: pd.Series | dict | None = None,
    cell_model: dict | None = None,
    temporal_atlas: dict | None = None,
    artifacts_dir: str | Path | None = "./cv_artifacts",
    raw_time_col: str = "time",
    spatial_cols: list[str] = ["x_aligned_um", "y_aligned_um", "z_aligned_um"],
    embryo_col: str = "series",
    cell_col: str = "cell",
    k_pacing: float | None = None,
    embryo_qc: pd.DataFrame | dict | None = None,
) -> dict:
    """Extracts, registers, and prepares trajectory data for 2D/3D diagnostic dashboards."""
    embryo_id = str(embryo_id)

    # 1. Resolve model and scores if not directly supplied
    if cell_model is None or temporal_atlas is None or target_row is None:
        art_path = Path(artifacts_dir) if artifacts_dir else Path("./cv_artifacts")
        score_files = sorted(art_path.glob("scores_fold_*.parquet"))
        target_fold = None
        for s_file in score_files:
            df_f = pd.read_parquet(s_file)
            e_col_f = "embryo_id" if "embryo_id" in df_f.columns else ("series" if "series" in df_f.columns else embryo_col)
            c_col_f = "cell" if "cell" in df_f.columns else ("cell_name" if "cell_name" in df_f.columns else cell_col)
            match = df_f[(df_f[e_col_f].astype(str) == embryo_id) & (df_f[c_col_f] == cell_name)]
            if not match.empty:
                if target_row is None:
                    target_row = match.iloc[0]
                target_fold = int(s_file.stem.split("_")[-1])
                break

        if target_fold is not None:
            if cell_model is None:
                m_dict = joblib.load(art_path / f"models_fold_{target_fold}.joblib")
                cell_model = m_dict[cell_name]
            if temporal_atlas is None:
                meta = joblib.load(art_path / f"metadata_fold_{target_fold}.joblib")
                temporal_atlas = meta["temporal_atlas"]

    if cell_model is None or temporal_atlas is None:
        raise ValueError(f"Could not resolve cell model or temporal atlas for '{cell_name}' in embryo '{embryo_id}'.")

    model_dict = cell_model.to_dict() if hasattr(cell_model, "to_dict") else cell_model

    # 2. Extract observed track for this cell and embryo
    e_col_pos = embryo_col if embryo_col in pos_df.columns else ("series" if "series" in pos_df.columns else "embryo_id")
    c_col_pos = cell_col if cell_col in pos_df.columns else ("cell" if "cell" in pos_df.columns else "cell_name")
    sub_obs = pos_df[
        (pos_df[e_col_pos].astype(str) == embryo_id) & (pos_df[c_col_pos] == cell_name)
    ].sort_values(raw_time_col)

    if sub_obs.empty:
        raise ValueError(f"No coordinate records found in pos_df for cell '{cell_name}' in embryo '{embryo_id}'.")

    t_raw = sub_obs[raw_time_col].values.astype(float)
    t_birth = float(t_raw.min())
    tau_raw = t_raw - t_birth

    # Resolve embryo developmental pacing factor Ke to normalize raw time to canonical minutes
    k_rate = None
    if k_pacing is not None and float(k_pacing) > 0:
        k_rate = float(k_pacing)
    elif embryo_qc is not None:
        if isinstance(embryo_qc, pd.DataFrame):
            e_col_qc = "embryo_id" if "embryo_id" in embryo_qc.columns else ("series" if "series" in embryo_qc.columns else embryo_col)
            m_qc = embryo_qc[embryo_qc[e_col_qc].astype(str) == embryo_id]
            if not m_qc.empty and "k_test" in m_qc.columns and pd.notna(m_qc.iloc[0]["k_test"]):
                k_rate = float(m_qc.iloc[0]["k_test"])
        elif isinstance(embryo_qc, dict):
            if embryo_id in embryo_qc and isinstance(embryo_qc[embryo_id], dict) and "k_test" in embryo_qc[embryo_id]:
                k_rate = float(embryo_qc[embryo_id]["k_test"])
            elif "k_test" in embryo_qc:
                k_rate = float(embryo_qc["k_test"])

    if k_rate is None and target_row is not None:
        if "k_test" in target_row and pd.notna(target_row["k_test"]) and float(target_row["k_test"]) > 0:
            k_rate = float(target_row["k_test"])
        elif "k_e" in target_row and pd.notna(target_row["k_e"]) and float(target_row["k_e"]) > 0:
            k_rate = float(target_row["k_e"])
        elif "canon_duration" in target_row and pd.notna(target_row["canon_duration"]) and float(target_row["canon_duration"]) > 0:
            raw_dur = float(t_raw.max() - t_birth)
            canon_dur = float(target_row["canon_duration"])
            if raw_dur > 0:
                k_rate = raw_dur / canon_dur

    if k_rate is None or k_rate <= 0:
        k_rate = 1.0

    # Pacing-normalized elapsed canonical time
    tau_obs = tau_raw / max(k_rate, 1e-4)
    tau_obs_max = float(tau_obs.max())

    tau_grid = np.asarray(model_dict.get("time_grid", np.linspace(0, 30, 30)), dtype=float)
    tau_cutoff = float(model_dict.get("tau_cutoff", tau_grid[-1] if len(tau_grid) > 0 else 30.0))
    if tau_cutoff <= 0:
        tau_cutoff = tau_grid[-1] if len(tau_grid) > 0 else 30.0

    valid_mask = tau_obs <= tau_cutoff + 1e-4
    if np.sum(valid_mask) < 2:
        valid_mask = np.ones(len(tau_obs), dtype=bool)

    tau_active = tau_obs[valid_mask]
    xyz_raw_full = sub_obs[spatial_cols].values.astype(float)
    xyz_raw = xyz_raw_full[valid_mask]
    com_obs = np.mean(xyz_raw, axis=0)
    xyz_centered = xyz_raw - com_obs
    xyz_centered_full = xyz_raw_full - com_obs

    # 3. Rigid SO(3) Kabsch rotation
    template_curve = model_dict.get("template_curve")
    if template_curve is not None:
        R_test, rot_angle_deg, coords_aligned = register_curve_to_template(
            s_obs=tau_active,
            coords_centered=xyz_centered,
            template_curve=template_curve,
            s_grid=tau_grid,
        )
    else:
        R_test = np.eye(3)
        rot_angle_deg = 0.0
        coords_aligned = xyz_centered
        template_curve = np.zeros((len(tau_grid), 3))

    coords_aligned_full = (R_test @ xyz_centered_full.T).T
    mu_com = np.asarray(model_dict["mu_com"], dtype=float)
    obs_3d_full = coords_aligned_full + mu_com
    obs_3d_active = coords_aligned + mu_com

    # 4. Regularized Monotonic Time Warping
    gamma_test, rms_warp, slopes = regularized_monotonic_time_warp(
        tau_obs=tau_active,
        coords_aligned=coords_aligned,
        template_curve=template_curve,
        tau_grid=tau_grid,
        lambda_reg=float(model_dict.get("warping_lambda", 1.0)),
        slope_bounds=(0.5, 2.0),
    )
    active_tau_grid = tau_grid[tau_grid <= tau_active.max() + 1e-4]
    warp_met = compute_warp_metrics(gamma_test, active_tau_grid, slopes)
    u_eval = np.clip(gamma_test / max(tau_cutoff, 1e-3), 0.0, 1.0)
    xyz_eval = np.column_stack([np.interp(gamma_test, tau_active, coords_aligned[:, d]) for d in range(3)])

    if len(active_tau_grid) > 1:
        g_dot = np.maximum(np.gradient(gamma_test, active_tau_grid), 0.0)
    else:
        g_dot = np.ones_like(active_tau_grid, dtype=float)

    # 5. GP Predictions along dense ribbon
    s_dense = np.asarray(model_dict["s_dense"], dtype=float).flatten()
    tau_dense = s_dense * tau_cutoff
    n_eval = len(gamma_test)
    pred_at_obs = np.zeros((n_eval, 3))
    gp_preds = {}

    if model_dict.get("dense_mu_3d") is not None and model_dict.get("dense_cov_3d") is not None:
        dense_mu = np.asarray(model_dict["dense_mu_3d"], dtype=float)
        dense_cov = np.asarray(model_dict["dense_cov_3d"], dtype=float)
        for idx, col in enumerate(spatial_cols):
            gp_preds[col] = {
                "mu": dense_mu[:, idx],
                "std": np.sqrt(np.maximum(dense_cov[:, idx, idx], 1e-6)),
            }
            pred_at_obs[:, idx] = interp1d(s_dense, dense_mu[:, idx], kind="linear", fill_value="extrapolate")(u_eval)
        pred_cov_eval = interp1d(s_dense, dense_cov, axis=0, kind="linear", fill_value="extrapolate")(u_eval)
        d_spat_shape_calc, rmse_3d_calc = compute_3d_mahalanobis_residuals(xyz_eval, pred_at_obs, pred_cov_eval)
    else:
        for idx, col in enumerate(spatial_cols):
            m_d = np.asarray(model_dict["dense_pred"][col]["mu"], dtype=float)
            s_d = np.asarray(model_dict["dense_pred"][col]["std"], dtype=float)
            gp_preds[col] = {"mu": m_d, "std": s_d}
            pred_at_obs[:, idx] = interp1d(s_dense, m_d, kind="linear", fill_value="extrapolate")(u_eval)
        rmse_3d_calc = float(np.sqrt(np.mean(np.sum((xyz_eval - pred_at_obs) ** 2, axis=1))))
        d_spat_shape_calc = rmse_3d_calc

    # 6. Canonical reference 3D path
    canon_3d = np.column_stack([
        gp_preds[spatial_cols[0]]["mu"] + mu_com[0],
        gp_preds[spatial_cols[1]]["mu"] + mu_com[1],
        gp_preds[spatial_cols[2]]["mu"] + mu_com[2],
    ])

    # 7. Splitting Reference and Observed curves for solid vs. dashed rendering
    # Case A: Reference curve overhang if observed trajectory is shorter than reference curve
    if tau_obs_max < tau_cutoff - 1e-4:
        ref_active_mask = tau_dense <= tau_obs_max + 1e-4
        if np.any(ref_active_mask) and np.any(~ref_active_mask):
            p_split = interp1d(tau_dense, canon_3d, axis=0)(tau_obs_max)
            canon_solid = np.vstack([canon_3d[ref_active_mask], p_split[None, :]])
            canon_dashed = np.vstack([p_split[None, :], canon_3d[~ref_active_mask]])
        else:
            canon_solid = canon_3d
            canon_dashed = None
    else:
        canon_solid = canon_3d
        canon_dashed = None

    # Case B: Observed trajectory overhang if longer than reference curve
    obs_solid = obs_3d_active
    if np.any(~valid_mask):
        obs_dashed = np.vstack([obs_3d_active[-1:], obs_3d_full[~valid_mask]])
        tau_overhang = np.concatenate([tau_active[-1:], tau_obs[~valid_mask]])
    else:
        obs_dashed = None
        tau_overhang = None

    # Start and End points
    p_ref_start = canon_3d[0]
    p_ref_end = canon_3d[-1]
    p_obs_start = obs_3d_full[0]
    p_obs_end = obs_3d_full[-1]

    # 8. Center of Mass & Spatial Shift via Joint GP B
    diff_com = com_obs - mu_com
    d_shift_euclid = float(np.linalg.norm(diff_com))
    if model_dict.get("B_cov") is not None:
        inv_B = np.linalg.pinv(model_dict["B_cov"])
        d_shift_score = float(np.sqrt(max(diff_com.T @ inv_B @ diff_com, 0.0)))
    else:
        inv_cov_com = np.asarray(model_dict.get("inv_cov_com", np.eye(3)), dtype=float)
        d_shift_score = float(np.sqrt(max(diff_com.T @ inv_cov_com @ diff_com, 0.0)))

    # 9. Timing & Birth Offset
    t_stat = temporal_atlas[cell_name]
    ref_b = float(t_stat["mu_birth"])
    ref_dur = float(t_stat["mu_phys"])
    ref_d = ref_b + ref_dur

    if target_row is not None and "canon_birth" in target_row and pd.notna(target_row["canon_birth"]):
        obs_b = float(target_row["canon_birth"])
    elif target_row is not None and "canon_mid" in target_row and "canon_duration" in target_row:
        obs_b = float(target_row["canon_mid"]) - 0.5 * float(target_row["canon_duration"])
    elif target_row is not None and "delta_birth_min" in target_row and pd.notna(target_row["delta_birth_min"]):
        obs_b = ref_b + float(target_row["delta_birth_min"])
    elif target_row is not None and "k_test" in target_row and "dt0_test" in target_row:
        obs_b = (t_birth - float(target_row["dt0_test"])) / max(float(target_row["k_test"]), 1e-4)
    else:
        obs_b = t_birth

    if target_row is not None and "canon_duration" in target_row and pd.notna(target_row["canon_duration"]):
        obs_dur = float(target_row["canon_duration"])
    elif target_row is not None and "k_test" in target_row:
        obs_dur = float(t_raw.max() - t_birth) / max(float(target_row["k_test"]), 1e-4)
    else:
        obs_dur = tau_obs_max

    obs_d = obs_b + obs_dur
    dur_ratio = float(obs_dur / max(ref_dur, 1e-4))
    pct_dur_dev = float((dur_ratio - 1.0) * 100.0)

    delta_birth_val = float(target_row["delta_birth_min"]) if target_row is not None and "delta_birth_min" in target_row else float(obs_b - ref_b)
    z_birth_shift = float(target_row["z_temp_shift"]) if target_row is not None and "z_temp_shift" in target_row else 0.0
    z_temp_shape = float(target_row["z_temp_shape"]) if target_row is not None and "z_temp_shape" in target_row else 0.0
    spat_shape_score = float(target_row["d_spat_shape"]) if target_row is not None and "d_spat_shape" in target_row else d_spat_shape_calc
    rot_angle_val = float(target_row["rot_angle_deg"]) if target_row is not None and "rot_angle_deg" in target_row else rot_angle_deg
    rms_warp_val = float(target_row["rms_warp_min"]) if target_row is not None and "rms_warp_min" in target_row else rms_warp

    std_birth_err = abs(delta_birth_val / z_birth_shift) if abs(z_birth_shift) > 1e-4 else float(np.sqrt(max(t_stat.get("var_birth", 1.0), 0.5)))

    # Q-value callouts
    def _q_str(col_name: str) -> str:
        if target_row is not None and col_name in target_row and pd.notna(target_row[col_name]):
            q = float(target_row[col_name])
            return f", $q = {q:.3e}$" if q < 0.05 else f", $q = {q:.2f}$"
        return ""

    return {
        "cell_name": cell_name,
        "embryo_id": embryo_id,
        "k_pacing": k_rate,
        "target_row": target_row,
        "cell_model": cell_model,
        "model_dict": model_dict,
        "temporal_atlas": temporal_atlas,
        "t_stat": t_stat,
        "sub_obs": sub_obs,
        "spatial_cols": spatial_cols,
        "t_raw": t_raw,
        "t_birth": t_birth,
        "tau_obs": tau_obs,
        "tau_obs_max": tau_obs_max,
        "tau_grid": tau_grid,
        "tau_cutoff": tau_cutoff,
        "valid_mask": valid_mask,
        "tau_active": tau_active,
        "xyz_raw_full": xyz_raw_full,
        "xyz_raw": xyz_raw,
        "com_obs": com_obs,
        "xyz_centered": xyz_centered,
        "xyz_centered_full": xyz_centered_full,
        "template_curve": template_curve,
        "R_test": R_test,
        "rot_angle_deg": rot_angle_deg,
        "coords_aligned": coords_aligned,
        "coords_aligned_full": coords_aligned_full,
        "gamma_test": gamma_test,
        "rms_warp": rms_warp,
        "slopes": slopes,
        "active_tau_grid": active_tau_grid,
        "g_dot": g_dot,
        "warp_met": warp_met,
        "u_eval": u_eval,
        "xyz_eval": xyz_eval,
        "s_dense": s_dense,
        "tau_dense": tau_dense,
        "n_eval": n_eval,
        "pred_at_obs": pred_at_obs,
        "gp_preds": gp_preds,
        "d_spat_shape_calc": d_spat_shape_calc,
        "rmse_3d_calc": rmse_3d_calc,
        "mu_com": mu_com,
        "diff_com": diff_com,
        "d_shift_euclid": d_shift_euclid,
        "d_shift_score": d_shift_score,
        "ref_b": ref_b,
        "ref_dur": ref_dur,
        "ref_d": ref_d,
        "obs_b": obs_b,
        "obs_dur": obs_dur,
        "obs_d": obs_d,
        "dur_ratio": dur_ratio,
        "pct_dur_dev": pct_dur_dev,
        "delta_birth_val": delta_birth_val,
        "z_birth_shift": z_birth_shift,
        "z_temp_shape": z_temp_shape,
        "spat_shape_score": spat_shape_score,
        "rot_angle_val": rot_angle_val,
        "rms_warp_val": rms_warp_val,
        "std_birth_err": std_birth_err,
        "canon_3d": canon_3d,
        "obs_3d_full": obs_3d_full,
        "obs_3d_active": obs_3d_active,
        "canon_solid": canon_solid,
        "canon_dashed": canon_dashed,
        "obs_solid": obs_solid,
        "obs_dashed": obs_dashed,
        "tau_overhang": tau_overhang,
        "p_ref_start": p_ref_start,
        "p_ref_end": p_ref_end,
        "p_obs_start": p_obs_start,
        "p_obs_end": p_obs_end,
        "q_shape_t": _q_str("qval_temp_shape"),
        "q_shift_t": _q_str("qval_temp_shift"),
        "q_warp_t": _q_str("qval_warp"),
        "q_shift_s": _q_str("qval_spat_shift"),
        "q_rot_s": _q_str("qval_spat_rot"),
        "q_shape_s": _q_str("qval_spat_shape"),
    }


def plot_interactive_3d_trajectory(
    cell_name: str,
    embryo_id: str,
    pos_df: pd.DataFrame,
    target_row: pd.Series | dict | None = None,
    cell_model: dict | None = None,
    temporal_atlas: dict | None = None,
    artifacts_dir: str | Path | None = "./cv_artifacts",
    raw_time_col: str = "time",
    spatial_cols: list[str] = ["x_aligned_um", "y_aligned_um", "z_aligned_um"],
    embryo_col: str = "series",
    cell_col: str = "cell",
    save_html: str | Path | None = None,
    width: int = 1200,
    height: int = 650,
    k_pacing: float | None = None,
    embryo_qc: pd.DataFrame | dict | None = None,
):
    """Generates an interactive dual-scene 3D WebGL trajectory diagnostic visualization using Plotly.
    
    Features:
      - Dual-scene synchronized layout: Scene 1 shows the 3D locally registered trajectory against
        the WT reference; Scene 2 shows the rigid spatial transformation (center-of-mass translation
        vector and SO(3) coordinate frame rotation triads).
      - Camera locking: Rotating, panning, or zooming either viewport synchronizes the camera angle
        in both viewports in real time.
      - 360-degree rotation, pan, zoom, and coordinate hover tooltips.
      - Start and End point annotations for Reference and Observed trajectories.
      - Solid line for the temporal mask where fit is evaluated; dashed for overhang.
      - Solid line for reference path where observed track exists; dashed for overhang if observed is shorter.
      
    Parameters
    ----------
    cell_name : str
        Blastomere identifier (e.g. 'MSapa').
    embryo_id : str
        Embryo series identifier.
    pos_df : pd.DataFrame
        Tracking coordinates DataFrame.
    target_row : pd.Series or dict, optional
        Precomputed anomaly scores row containing q-values.
    cell_model : dict, optional
        Fitted TrajectoryRibbon model dictionary.
    temporal_atlas : dict, optional
        Reference temporal atlas dictionary.
    artifacts_dir : str or Path, default='./cv_artifacts'
        Artifacts directory for fallback model loading.
    raw_time_col : str, default='time'
        Time column name.
    spatial_cols : list of str, default=['x_aligned_um', 'y_aligned_um', 'z_aligned_um']
        Spatial coordinate columns.
    embryo_col : str, default='series'
        Embryo identifier column name.
    cell_col : str, default='cell'
        Cell name column name.
    save_html : str or Path, optional
        If provided, writes a self-contained interactive HTML file with synchronized cameras to this path.
    width : int, default=1200
        Figure width in pixels.
    height : int, default=650
        Figure height in pixels.
    k_pacing : float, optional
        Embryo pacing rate factor Ke for canonical developmental time normalization.
    embryo_qc : pd.DataFrame or dict, optional
        Embryo-level QC metrics containing 'k_test'.
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive Plotly dual-scene 3D Figure.
    """
    if go is None or make_subplots is None:
        raise ImportError("Plotly is required for plot_interactive_3d_trajectory. Install via 'pip install plotly'.")

    d = _prepare_trajectory_dashboard_data(
        cell_name=cell_name,
        embryo_id=embryo_id,
        pos_df=pos_df,
        target_row=target_row,
        cell_model=cell_model,
        temporal_atlas=temporal_atlas,
        artifacts_dir=artifacts_dir,
        raw_time_col=raw_time_col,
        spatial_cols=spatial_cols,
        embryo_col=embryo_col,
        cell_col=cell_col,
        k_pacing=k_pacing,
        embryo_qc=embryo_qc,
    )

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=[
            f"<b>3D Locally Registered Trajectory</b> (Canonical Frame)",
            f"<b>3D Spatial Transformation</b> (Translation + SO(3) Rotation)",
        ],
        horizontal_spacing=0.03,
    )

    # === Scene 1: Locally Registered Trajectory ===
    # 1. Reference Curve
    if d["canon_dashed"] is not None:
        fig.add_trace(go.Scatter3d(
            x=d["canon_solid"][:, 0], y=d["canon_solid"][:, 1], z=d["canon_solid"][:, 2],
            mode="lines",
            line=dict(color="#2c3e50", width=6),
            name=f"Reference Path (Active, τ ≤ {d['tau_obs_max']:.1f} min)",
            hoverinfo="text",
            hovertext=[f"Ref (Active): X={pt[0]:.2f}, Y={pt[1]:.2f}, Z={pt[2]:.2f} μm" for pt in d["canon_solid"]],
        ), row=1, col=1)
        fig.add_trace(go.Scatter3d(
            x=d["canon_dashed"][:, 0], y=d["canon_dashed"][:, 1], z=d["canon_dashed"][:, 2],
            mode="lines",
            line=dict(color="#2c3e50", width=5, dash="dash"),
            name=f"Reference Overhang (τ > {d['tau_obs_max']:.1f} min)",
            hoverinfo="text",
            hovertext=[f"Ref (Overhang): X={pt[0]:.2f}, Y={pt[1]:.2f}, Z={pt[2]:.2f} μm" for pt in d["canon_dashed"]],
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter3d(
            x=d["canon_solid"][:, 0], y=d["canon_solid"][:, 1], z=d["canon_solid"][:, 2],
            mode="lines",
            line=dict(color="#2c3e50", width=6),
            name="Canonical Reference Path",
            hoverinfo="text",
            hovertext=[f"Ref: τ={t:.1f} min<br>X={pt[0]:.2f}, Y={pt[1]:.2f}, Z={pt[2]:.2f} μm" for t, pt in zip(d["tau_dense"], d["canon_solid"])],
        ), row=1, col=1)

    # 2. Observed Track
    fig.add_trace(go.Scatter3d(
        x=d["obs_solid"][:, 0], y=d["obs_solid"][:, 1], z=d["obs_solid"][:, 2],
        mode="lines+markers",
        line=dict(color="#d63031", width=5),
        marker=dict(size=4, color="#d63031"),
        name=f"Observed Track ({embryo_id}, Fitted)",
        hoverinfo="text",
        hovertext=[f"Obs (Fitted): τ={t:.1f} min<br>X={pt[0]:.2f}, Y={pt[1]:.2f}, Z={pt[2]:.2f} μm" for t, pt in zip(d["tau_active"], d["obs_solid"])],
    ), row=1, col=1)

    if d["obs_dashed"] is not None:
        fig.add_trace(go.Scatter3d(
            x=d["obs_dashed"][:, 0], y=d["obs_dashed"][:, 1], z=d["obs_dashed"][:, 2],
            mode="lines+markers",
            line=dict(color="#d63031", width=4, dash="dash"),
            marker=dict(size=4, color="#e74c3c", symbol="diamond-open"),
            name=f"Observed Overhang (τ > {d['tau_cutoff']:.1f} min)",
            hoverinfo="text",
            hovertext=[f"Obs (Overhang): τ={t:.1f} min<br>X={pt[0]:.2f}, Y={pt[1]:.2f}, Z={pt[2]:.2f} μm" for t, pt in zip(d["tau_overhang"], d["obs_dashed"])],
        ), row=1, col=1)

    # 3. Start and End annotations
    fig.add_trace(go.Scatter3d(
        x=[d["p_ref_start"][0]], y=[d["p_ref_start"][1]], z=[d["p_ref_start"][2]],
        mode="markers+text",
        marker=dict(size=8, color="#2c3e50", symbol="circle", line=dict(color="white", width=1.5)),
        text=["Ref Start (0m)"],
        textposition="top center",
        name="Ref Start",
        hoverinfo="text",
        hovertext=[f"Ref Start: τ=0.0 min<br>X={d['p_ref_start'][0]:.2f}, Y={d['p_ref_start'][1]:.2f}, Z={d['p_ref_start'][2]:.2f} μm"],
    ), row=1, col=1)

    fig.add_trace(go.Scatter3d(
        x=[d["p_ref_end"][0]], y=[d["p_ref_end"][1]], z=[d["p_ref_end"][2]],
        mode="markers+text",
        marker=dict(size=8, color="#2c3e50", symbol="square", line=dict(color="white", width=1.5)),
        text=[f"Ref End ({d['tau_cutoff']:.1f}m)"],
        textposition="top center",
        name="Ref End",
        hoverinfo="text",
        hovertext=[f"Ref End: τ={d['tau_cutoff']:.1f} min<br>X={d['p_ref_end'][0]:.2f}, Y={d['p_ref_end'][1]:.2f}, Z={d['p_ref_end'][2]:.2f} μm"],
    ), row=1, col=1)

    fig.add_trace(go.Scatter3d(
        x=[d["p_obs_start"][0]], y=[d["p_obs_start"][1]], z=[d["p_obs_start"][2]],
        mode="markers+text",
        marker=dict(size=8, color="#d63031", symbol="circle", line=dict(color="white", width=1.5)),
        text=["Obs Start (0m)"],
        textposition="bottom center",
        name="Obs Start",
        hoverinfo="text",
        hovertext=[f"Obs Start: τ=0.0 min<br>X={d['p_obs_start'][0]:.2f}, Y={d['p_obs_start'][1]:.2f}, Z={d['p_obs_start'][2]:.2f} μm"],
    ), row=1, col=1)

    fig.add_trace(go.Scatter3d(
        x=[d["p_obs_end"][0]], y=[d["p_obs_end"][1]], z=[d["p_obs_end"][2]],
        mode="markers+text",
        marker=dict(size=10, color="#d63031", symbol="diamond", line=dict(color="white", width=1.5)),
        text=[f"Obs End ({d['tau_obs_max']:.1f}m)"],
        textposition="bottom center",
        name="Obs End",
        hoverinfo="text",
        hovertext=[f"Obs End: τ={d['tau_obs_max']:.1f} min<br>X={d['p_obs_end'][0]:.2f}, Y={d['p_obs_end'][1]:.2f}, Z={d['p_obs_end'][2]:.2f} μm"],
    ), row=1, col=1)

    # === Scene 2: Spatial Transformation ===
    c_init = d["com_obs"]
    c_final = d["mu_com"]
    vec_trans = c_final - c_init
    d_trans = float(np.linalg.norm(vec_trans))
    R = d["R_test"]
    theta_deg = float(d["rot_angle_deg"])
    triad_len = max(d_trans * 0.35, 3.5)

    # Alphad out trajectories
    xyz_raw = d["xyz_raw"]
    fig.add_trace(go.Scatter3d(
        x=xyz_raw[:, 0], y=xyz_raw[:, 1], z=xyz_raw[:, 2],
        mode="lines",
        line=dict(color="#e67e22", width=3, dash="dash"),
        opacity=0.4,
        name="Initial Track (Global)",
        hoverinfo="text",
        hovertext="Raw Observed Track in Global Frame",
    ), row=1, col=2)

    fig.add_trace(go.Scatter3d(
        x=d["obs_solid"][:, 0], y=d["obs_solid"][:, 1], z=d["obs_solid"][:, 2],
        mode="lines",
        line=dict(color="#d63031", width=3),
        opacity=0.45,
        name="Registered Track",
        hoverinfo="text",
        hovertext="Locally Registered Track",
    ), row=1, col=2)

    fig.add_trace(go.Scatter3d(
        x=d["canon_solid"][:, 0], y=d["canon_solid"][:, 1], z=d["canon_solid"][:, 2],
        mode="lines",
        line=dict(color="#2c3e50", width=2.5, dash="dot"),
        opacity=0.35,
        name="WT Reference",
        hoverinfo="text",
        hovertext="WT Reference Consensus Path",
    ), row=1, col=2)

    # Centroids
    fig.add_trace(go.Scatter3d(
        x=[c_init[0]], y=[c_init[1]], z=[c_init[2]],
        mode="markers+text",
        marker=dict(size=8, color="#e67e22", line=dict(color="black", width=1.5)),
        text=["c_obs"],
        textposition="top center",
        name="Initial Centroid",
    ), row=1, col=2)

    fig.add_trace(go.Scatter3d(
        x=[c_final[0]], y=[c_final[1]], z=[c_final[2]],
        mode="markers+text",
        marker=dict(size=8, color="#2c3e50", line=dict(color="black", width=1.5)),
        text=["μ_com"],
        textposition="top center",
        name="Target Centroid",
    ), row=1, col=2)

    # Translation Vector
    fig.add_trace(go.Scatter3d(
        x=[c_init[0], c_final[0]], y=[c_init[1], c_final[1]], z=[c_init[2], c_final[2]],
        mode="lines+text",
        line=dict(color="#8e44ad", width=7),
        text=["", f"T (Δc={d_trans:.2f}μm)"],
        textposition="middle right",
        textfont=dict(color="#8e44ad", size=11),
        name=f"Translation Vector (d={d_trans:.2f}μm)",
    ), row=1, col=2)

    v_norm = vec_trans / max(d_trans, 1e-6)
    fig.add_trace(go.Cone(
        x=[c_final[0]], y=[c_final[1]], z=[c_final[2]],
        u=[v_norm[0]], v=[v_norm[1]], w=[v_norm[2]],
        sizemode="absolute",
        sizeref=1.5,
        colorscale=[[0, "#8e44ad"], [1, "#8e44ad"]],
        showscale=False,
        showlegend=False,
    ), row=1, col=2)

    # Coordinate Frame Triads
    colors_triad = ["#e74c3c", "#27ae60", "#2980b9"]
    labels_canon = ["X (AP)", "Y (DV)", "Z (LR)"]
    for k in range(3):
        ek = np.zeros(3)
        ek[k] = triad_len
        end_pt = c_init + ek
        fig.add_trace(go.Scatter3d(
            x=[c_init[0], end_pt[0]], y=[c_init[1], end_pt[1]], z=[c_init[2], end_pt[2]],
            mode="lines+text",
            line=dict(color=colors_triad[k], width=4, dash="dash"),
            text=["", labels_canon[k]],
            textposition="top right",
            textfont=dict(color=colors_triad[k], size=10),
            showlegend=False,
        ), row=1, col=2)

    labels_rot = ["X' (AP)", "Y' (DV)", "Z' (LR)"]
    for k in range(3):
        ek = np.zeros(3)
        ek[k] = triad_len
        rk = R @ ek
        end_pt = c_final + rk
        fig.add_trace(go.Scatter3d(
            x=[c_final[0], end_pt[0]], y=[c_final[1], end_pt[1]], z=[c_final[2], end_pt[2]],
            mode="lines+text",
            line=dict(color=colors_triad[k], width=6),
            text=["", labels_rot[k]],
            textposition="top right",
            textfont=dict(color=colors_triad[k], size=11),
            showlegend=False,
        ), row=1, col=2)
        rk_norm = rk / np.linalg.norm(rk)
        fig.add_trace(go.Cone(
            x=[end_pt[0]], y=[end_pt[1]], z=[end_pt[2]],
            u=[rk_norm[0]], v=[rk_norm[1]], w=[rk_norm[2]],
            sizemode="absolute",
            sizeref=1.0,
            colorscale=[[0, colors_triad[k]], [1, colors_triad[k]]],
            showscale=False,
            showlegend=False,
        ), row=1, col=2)

    for k in range(3):
        ek = np.zeros(3)
        ek[k] = triad_len * 0.7
        end_pt = c_final + ek
        fig.add_trace(go.Scatter3d(
            x=[c_final[0], end_pt[0]], y=[c_final[1], end_pt[1]], z=[c_final[2], end_pt[2]],
            mode="lines",
            line=dict(color=colors_triad[k], width=2, dash="dot"),
            opacity=0.35,
            showlegend=False,
        ), row=1, col=2)

    # Title with metrics
    q_rot = d["q_rot_s"].replace("$", "")
    q_shape = d["q_shape_s"].replace("$", "")
    q_shift = d["q_shift_s"].replace("$", "")
    pacing_tag = f" [Pacing Ke = {d['k_pacing']:.3f}]" if abs(d.get("k_pacing", 1.0) - 1.0) > 0.005 else ""
    title_text = (
        f"<b>Interactive 3D Trajectory & Transformation: '{cell_name}' (Embryo {embryo_id})</b><br>"
        f"<span style=\"font-size:12px; color:#555;\">Translation: Δc = {d_trans:.2f} μm{q_shift} | "
        f"SO(3) Rotation: {d['rot_angle_val']:.1f}°{q_rot} | Shape Residual: {d['spat_shape_score']:.2f}{q_shape} | "
        f"Lifespan: {d['tau_obs_max']:.1f}m (Ref: {d['tau_cutoff']:.1f}m){pacing_tag}</span>"
    )

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=13, color="#2c3e50"), x=0.5),
        scene=dict(
            xaxis_title="X (μm)",
            yaxis_title="Y (μm)",
            zaxis_title="Z (μm)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.7, y=1.7, z=1.2)),
        ),
        scene2=dict(
            xaxis_title="X (μm)",
            yaxis_title="Y (μm)",
            zaxis_title="Z (μm)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.7, y=1.7, z=1.2)),
        ),
        width=width,
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.85)",
            bordercolor="rgba(0, 0, 0, 0.15)",
            borderwidth=1,
            font=dict(size=10),
        ),
        margin=dict(l=20, r=20, t=75, b=20),
        template="plotly_white",
    )

    if save_html is not None:
        save_p = Path(save_html)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        html_str = fig.to_html(include_plotlyjs="cdn", full_html=True)
        all_scenes_json = "['scene', 'scene2']"
        sync_script = f"""
<script>
document.addEventListener("DOMContentLoaded", function() {{
    var gd = document.getElementsByClassName('plotly-graph-div')[0];
    if (!gd) return;
    var isUpdating = false;
    var allScenes = {all_scenes_json};
    gd.on('plotly_relayout', function(eventData) {{
        if (isUpdating || !eventData) return;
        var changedCamera = null;
        for (var key in eventData) {{
            if (key.indexOf('camera') !== -1) {{
                if (key.endsWith('.camera')) {{
                    changedCamera = eventData[key];
                    break;
                }}
                var parts = key.split('.');
                var sceneName = parts[0];
                if (gd.layout && gd.layout[sceneName] && gd.layout[sceneName].camera) {{
                    changedCamera = gd.layout[sceneName].camera;
                    break;
                }}
            }}
        }}
        if (changedCamera) {{
            isUpdating = true;
            var update = {{}};
            for (var i = 0; i < allScenes.length; i++) {{
                update[allScenes[i] + '.camera'] = changedCamera;
            }}
            Plotly.relayout(gd, update).then(function() {{
                isUpdating = false;
            }}).catch(function() {{
                isUpdating = false;
            }});
        }}
    }});
}});
</script>
</body>
"""
        html_synced = html_str.replace("</body>", sync_script)
        save_p.write_text(html_synced, encoding="utf-8")

    return fig


def plot_cell_trajectory_diagnostic_dashboard(
    cell_name: str,
    embryo_id: str,
    pos_df: pd.DataFrame,
    target_row: pd.Series | dict | None = None,
    cell_model: dict | None = None,
    temporal_atlas: dict | None = None,
    artifacts_dir: str | Path | None = "./cv_artifacts",
    raw_time_col: str = "time",
    spatial_cols: list[str] = ["x_aligned_um", "y_aligned_um", "z_aligned_um"],
    embryo_col: str = "series",
    cell_col: str = "cell",
    figsize: tuple[float, float] = (22.0, 14.5),
    dpi: int = 130,
    save_interactive_html: str | Path | None = None,
    return_interactive_3d: bool = False,
    interactive_3d: bool = False,
    k_pacing: float | None = None,
    embryo_qc: pd.DataFrame | dict | None = None,
):
    """Renders a comprehensive 7-panel single-cell spatiotemporal trajectory diagnostic dashboard.
    
    Panels:
      1-3: X(tau), Y(tau), Z(tau) progression profiles with Joint GP ribbon, aligned/warped
           query points, and raw-centered trajectory (light dashed line) for visual contrast.
      4:   Regularized monotonic time-warping diffeomorphism gamma(tau) on [0, tau_cutoff].
      5:   3D spatial alignment showing canonical reference path and locally registered track,
           with solid fit segments, dashed overhang segments, and Start/End annotations.
      6:   3D spatial transformation showing center-of-mass translation T(Delta c) and SO(3)
           coordinate frame rotation triads (X, Y, Z -> X', Y', Z') with contextual alphad trajectories.
      7:   Birth-anchored developmental lifespan interval comparison with tree-propagated uncertainty.
      
    Parameters
    ----------
    cell_name : str
        Blastomere identifier (e.g. 'Caap').
    embryo_id : str
        Embryo identifier (e.g. '200113_2').
    pos_df : pd.DataFrame
        Tracking coordinates dataframe.
    target_row : pd.Series or dict, optional
        Precomputed anomaly scores row containing q-values and metrics.
    cell_model : dict, optional
        Fitted TrajectoryRibbon or dictionary for cell_name.
    temporal_atlas : dict, optional
        Reference temporal atlas dictionary.
    artifacts_dir : str or Path, default='./cv_artifacts'
        Directory containing CV models/metadata if not passed explicitly.
    raw_time_col : str, default='time'
        Time column name.
    spatial_cols : list of str, default=['x_aligned_um', 'y_aligned_um', 'z_aligned_um']
        3D coordinate column names.
    embryo_col : str, default='series'
        Embryo ID column name.
    cell_col : str, default='cell'
        Cell name column name.
    figsize : tuple of (float, float), default=(22.0, 14.5)
        Matplotlib figure dimensions.
    dpi : int, default=130
        Resolution.
    save_interactive_html : str or Path, optional
        If provided, generates and exports the interactive 3D WebGL trajectory HTML file to this path.
    return_interactive_3d : bool, default=False
        If True, also generates and returns the interactive Plotly 3D Figure (fig, axes_dict, fig_3d).
    interactive_3d : bool, default=False
        If True and running in an interactive notebook, displays the Plotly 3D Figure inline.
    k_pacing : float, optional
        Embryo pacing rate factor Ke for canonical developmental time normalization.
    embryo_qc : pd.DataFrame or dict, optional
        Embryo-level QC metrics containing 'k_test'.
        
    Returns
    -------
    fig : plt.Figure
        Rendered figure.
    axes_dict : dict of str -> plt.Axes
        Dictionary of panel axes.
    fig_3d : plotly.graph_objects.Figure, optional
        Returned only when return_interactive_3d=True.
    """

    d = _prepare_trajectory_dashboard_data(
        cell_name=cell_name,
        embryo_id=embryo_id,
        pos_df=pos_df,
        target_row=target_row,
        cell_model=cell_model,
        temporal_atlas=temporal_atlas,
        artifacts_dir=artifacts_dir,
        raw_time_col=raw_time_col,
        spatial_cols=spatial_cols,
        embryo_col=embryo_col,
        cell_col=cell_col,
        k_pacing=k_pacing,
        embryo_qc=embryo_qc,
    )

    # Render Matplotlib Figure with nested GridSpec:
    # Left column (width ratio 1.45): 3 rows [X, Y], [Z, gamma], and timing (spanning cols 0-1)
    # Right column (width ratio 1.0): 2 rows [3D locally registered], [3D spatial transformation]
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.16)

    gs_left = gs[0].subgridspec(3, 2, height_ratios=[1.0, 1.0, 0.70], hspace=0.36, wspace=0.24)
    ax_x = fig.add_subplot(gs_left[0, 0])
    ax_y = fig.add_subplot(gs_left[0, 1])
    ax_z = fig.add_subplot(gs_left[1, 0])
    ax_gamma = fig.add_subplot(gs_left[1, 1])
    ax_timing = fig.add_subplot(gs_left[2, :])

    gs_right = gs[1].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.26)
    ax_3d = fig.add_subplot(gs_right[0, 0], projection="3d")
    ax_trans = fig.add_subplot(gs_right[1, 0], projection="3d")

    coord_axes = [ax_x, ax_y, ax_z]
    axis_labels = [r"$X$", r"$Y$", r"$Z$"]
    norm_warp = TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=1.5)
    cmap_warp = plt.get_cmap("coolwarm")

    # Panels 1-3: Coordinate Fits
    for idx, (ax, col, lbl) in enumerate(zip(coord_axes, d["spatial_cols"], axis_labels)):
        mu = d["gp_preds"][col]["mu"]
        std = d["gp_preds"][col]["std"]

        # GP Reference Ribbon
        ax.plot(d["tau_dense"], mu, color="#2c3e50", linewidth=2.2, label="Joint GP Reference", zorder=3)
        if np.any(std > 0):
            ax.fill_between(
                d["tau_dense"], mu - 1.96 * std, mu + 1.96 * std,
                color="#3498db", alpha=0.22, label=r"$\pm 1.96\sigma$ Ribbon", zorder=2,
            )

        # Raw centered trajectory (light dashed line) showing pre-aligned contrast
        ax.plot(
            d["tau_active"], d["xyz_centered"][:, idx],
            color="#95a5a6", linestyle="--", linewidth=1.5, alpha=0.75,
            label="Raw Centered Track", zorder=4,
        )

        # Aligned & warped query points colored by instantaneous warping velocity g_dot(tau)
        ax.scatter(
            d["active_tau_grid"], d["xyz_eval"][:, idx],
            c=d["g_dot"], cmap=cmap_warp, norm=norm_warp,
            edgecolor="black", linewidth=0.6, s=36,
            label=rf"Query $\tau_k$ ($\dot{{\gamma}}$ colored)", zorder=5,
        )

        # Residual drop-lines to GP mean
        for k in range(d["n_eval"]):
            ax.plot(
                [d["active_tau_grid"][k], d["active_tau_grid"][k]],
                [d["pred_at_obs"][k, idx], d["xyz_eval"][k, idx]],
                color="#7f8c8d", linestyle=":", linewidth=1.1, zorder=4,
            )

        ax.set_title(rf"{lbl}($\tau$) Progression Profile ($\tau \in [0, {d['tau_cutoff']:.1f}]$ min)", fontsize=10.5, fontweight="bold")
        ax.set_xlabel(r"Elapsed Canonical Time $\tau$ (min)", fontsize=9)
        ax.set_ylabel(rf"{lbl} ($\mu\mathrm{{m}}$)", fontsize=9)
        ax.set_xlim([0.0, d["tau_cutoff"]])
        ax.grid(True, linestyle=":", alpha=0.5)
        if idx == 0:
            ax.legend(frameon=True, fontsize=7.2, loc="best")

    # Panel 4: Monotonic Warping Diffeomorphism gamma(tau)
    ax_gamma.plot([0, d["tau_cutoff"]], [0, d["tau_cutoff"]], "k--", linewidth=1.4, label=r"Uniform Pacing ($\gamma_{\mathrm{id}}(\tau) = \tau$)")
    ax_gamma.plot(d["active_tau_grid"], d["gamma_test"], color="#e67e22", linewidth=2.4, label=rf"$\hat{{\gamma}}(\tau)$ ($\mathrm{{RMS}} = {d['rms_warp_val']:.2f}\,$min)")
    ax_gamma.fill_between(d["active_tau_grid"], d["active_tau_grid"], d["gamma_test"], color="#f39c12", alpha=0.22, label="Pacing Distortion Area")
    sc_gamma = ax_gamma.scatter(
        d["active_tau_grid"], d["gamma_test"],
        c=d["g_dot"], cmap=cmap_warp, norm=norm_warp,
        s=36, edgecolor="black", linewidth=0.6, zorder=5,
        label=rf"Query $\tau_k$ ($\dot{{\gamma}}$ colored)",
    )

    cax_warp = inset_axes(ax_gamma, width="40%", height="5.5%", loc="lower right", borderpad=1.4)
    cbar_warp = fig.colorbar(sc_gamma, cax=cax_warp, orientation="horizontal")
    cbar_warp.set_ticks([0.5, 1.0, 1.5])
    cbar_warp.ax.tick_params(labelsize=6.8, pad=1.5)
    cbar_warp.set_label(r"Velocity $\dot{\gamma}(\tau)$ (Delay $\leftarrow 1 \rightarrow$ Fast)", fontsize=7.2, labelpad=2)

    ax_gamma.set_title(rf"Monotonic Warping $\hat{{\gamma}}(\tau)$ ($\mathrm{{RMS}}_{{\mathrm{{warp}}}} = {d['rms_warp_val']:.2f}\,$min{d['q_warp_t']})", fontsize=10.5, fontweight="bold")
    ax_gamma.set_xlabel(r"Elapsed Canonical Time $\tau$ (min)", fontsize=9)
    ax_gamma.set_ylabel(r"Warped Progression $\hat{\gamma}(\tau)$ (min)", fontsize=9)
    ax_gamma.set_xlim([0.0, d["tau_cutoff"]])
    ax_gamma.set_ylim([0.0, max(d["tau_cutoff"], float(d["gamma_test"].max()) + 1.0)])
    ax_gamma.grid(True, linestyle=":", alpha=0.5)
    ax_gamma.legend(frameon=True, fontsize=7.2, loc="upper left")

    # Panel 5: 3D Trajectory & Spatial Alignment
    # 1. Canonical Reference Path (Solid active, Dashed overhang if observed is shorter)
    if d["canon_dashed"] is not None:
        ax_3d.plot(
            d["canon_solid"][:, 0], d["canon_solid"][:, 1], d["canon_solid"][:, 2],
            color="#2c3e50", linewidth=2.4, linestyle="-",
            label=f"Canonical Ref (Active, $\\tau \\leq {d['tau_obs_max']:.1f}$ min)", zorder=3,
        )
        ax_3d.plot(
            d["canon_dashed"][:, 0], d["canon_dashed"][:, 1], d["canon_dashed"][:, 2],
            color="#2c3e50", linewidth=2.0, linestyle="--", alpha=0.75,
            label=f"Canonical Ref Overhang ($\\tau > {d['tau_obs_max']:.1f}$ min)", zorder=3,
        )
    else:
        ax_3d.plot(
            d["canon_solid"][:, 0], d["canon_solid"][:, 1], d["canon_solid"][:, 2],
            color="#2c3e50", linewidth=2.4, linestyle="-",
            label="Canonical Reference Path", zorder=3,
        )

    # 2. Observed Track (Solid within fit mask, Dashed overhang if longer)
    ax_3d.plot(
        d["obs_solid"][:, 0], d["obs_solid"][:, 1], d["obs_solid"][:, 2],
        color="#d63031", linewidth=2.0, linestyle="-", marker="o", markersize=3.5,
        label=f"Observed Track ({d['embryo_id']}, Fitted)", zorder=4,
    )
    if d["obs_dashed"] is not None:
        ax_3d.plot(
            d["obs_dashed"][:, 0], d["obs_dashed"][:, 1], d["obs_dashed"][:, 2],
            color="#d63031", linewidth=1.8, linestyle="--", marker="s", markersize=3.0, alpha=0.85,
            label=f"Observed Track Overhang ($\\tau > {d['tau_cutoff']:.1f}$ min)", zorder=4,
        )

    # 3. Start & End Markers and Annotations
    ax_3d.scatter([d["p_ref_start"][0]], [d["p_ref_start"][1]], [d["p_ref_start"][2]], color="#2c3e50", edgecolor="#ffffff", linewidth=1.2, s=70, marker="o", zorder=6)
    ax_3d.text(d["p_ref_start"][0], d["p_ref_start"][1], d["p_ref_start"][2], "  Ref Start", fontsize=8.0, fontweight="bold", color="#2c3e50", zorder=7)

    ax_3d.scatter([d["p_ref_end"][0]], [d["p_ref_end"][1]], [d["p_ref_end"][2]], color="#2c3e50", edgecolor="#ffffff", linewidth=1.2, s=70, marker="s", zorder=6)
    ax_3d.text(d["p_ref_end"][0], d["p_ref_end"][1], d["p_ref_end"][2], f"  Ref End ({d['tau_cutoff']:.1f}m)", fontsize=8.0, fontweight="bold", color="#2c3e50", zorder=7)

    ax_3d.scatter([d["p_obs_start"][0]], [d["p_obs_start"][1]], [d["p_obs_start"][2]], color="#d63031", edgecolor="#ffffff", linewidth=1.2, s=75, marker="o", zorder=6)
    ax_3d.text(d["p_obs_start"][0], d["p_obs_start"][1], d["p_obs_start"][2], "  Obs Start", fontsize=8.0, fontweight="bold", color="#d63031", zorder=7)

    ax_3d.scatter([d["p_obs_end"][0]], [d["p_obs_end"][1]], [d["p_obs_end"][2]], color="#d63031", edgecolor="#ffffff", linewidth=1.2, s=85, marker="D", zorder=6)
    ax_3d.text(d["p_obs_end"][0], d["p_obs_end"][1], d["p_obs_end"][2], f"  Obs End ({d['tau_obs_max']:.1f}m)", fontsize=8.0, fontweight="bold", color="#d63031", zorder=7)

    ax_3d.set_title(
        rf"3D Locally Registered Trajectory" + "\n"
        rf"($\mathcal{{M}}_{{\mathrm{{shape}}}} = {d['spat_shape_score']:.2f}${d['q_shape_s']})",
        fontsize=10.5,
        fontweight="bold",
        pad=10,
    )
    ax_3d.set_xlabel(r"$X$ ($\mu\mathrm{m}$)", fontsize=8.5)
    ax_3d.set_ylabel(r"$Y$ ($\mu\mathrm{m}$)", fontsize=8.5)
    ax_3d.set_zlabel(r"$Z$ ($\mu\mathrm{m}$)", fontsize=8.5)

    # Enforce isotropic physical dimensions for 3D trajectory
    pts_3d = [d["obs_solid"], d["canon_solid"]]
    if d["obs_dashed"] is not None:
        pts_3d.append(d["obs_dashed"])
    if d["canon_dashed"] is not None:
        pts_3d.append(d["canon_dashed"])
    pts_3d_arr = np.vstack(pts_3d)
    mid_3d = np.mean(pts_3d_arr, axis=0)
    max_span_3d = max(float(np.ptp(pts_3d_arr, axis=0).max()), 2.0) * 0.55
    ax_3d.set_xlim(mid_3d[0] - max_span_3d, mid_3d[0] + max_span_3d)
    ax_3d.set_ylim(mid_3d[1] - max_span_3d, mid_3d[1] + max_span_3d)
    ax_3d.set_zlim(mid_3d[2] - max_span_3d, mid_3d[2] + max_span_3d)
    ax_3d.set_box_aspect((1, 1, 1))
    ax_3d.view_init(elev=20, azim=45)
    ax_3d.legend(frameon=True, fontsize=6.8, loc="upper left")

    # Panel 6: 3D Spatial Transformation (under 3D Locally Registered Trajectory)
    xyz_raw = d["xyz_raw"]
    c_init = d["com_obs"]
    c_final = d["mu_com"]
    vec_trans = c_final - c_init
    d_trans = float(np.linalg.norm(vec_trans))
    R = d["R_test"]
    theta_deg = float(d["rot_angle_deg"])
    triad_len = max(d_trans * 0.35, 3.2)

    # Alphad out trajectories
    ax_trans.plot(
        xyz_raw[:, 0], xyz_raw[:, 1], xyz_raw[:, 2],
        color="#e67e22", alpha=0.35, linewidth=2.2, linestyle="--",
        label="Initial Raw Track",
    )
    ax_trans.plot(
        d["obs_solid"][:, 0], d["obs_solid"][:, 1], d["obs_solid"][:, 2],
        color="#d63031", alpha=0.45, linewidth=2.2, linestyle="-",
        label="Locally Registered Track",
    )
    ax_trans.plot(
        d["canon_solid"][:, 0], d["canon_solid"][:, 1], d["canon_solid"][:, 2],
        color="#2c3e50", alpha=0.25, linewidth=1.8, linestyle=":",
        label="WT Reference Path",
    )

    # Centroids
    ax_trans.scatter([c_init[0]], [c_init[1]], [c_init[2]], color="#e67e22", edgecolor="black", linewidth=1.0, s=80, zorder=6, label=r"Initial Centroid $\mathbf{c}_{\mathrm{obs}}$")
    ax_trans.scatter([c_final[0]], [c_final[1]], [c_final[2]], color="#2c3e50", edgecolor="black", linewidth=1.0, s=80, zorder=6, label=r"Target Centroid $\boldsymbol{\mu}_{\mathrm{com}}$")

    # Translation Vector
    ax_trans.quiver(c_init[0], c_init[1], c_init[2], vec_trans[0], vec_trans[1], vec_trans[2], color="#8e44ad", linewidth=2.8, arrow_length_ratio=0.12, zorder=5)
    mid_t = 0.5 * (c_init + c_final)
    ax_trans.text(mid_t[0], mid_t[1], mid_t[2] + 0.5, rf" $\mathbf{{T}}$ ($\Delta c={d_trans:.2f}\,\mu\mathrm{{m}}$)", color="#8e44ad", fontsize=8.5, fontweight="bold", zorder=7)

    # Initial Canonical Triad at c_init
    colors_triad = ["#e74c3c", "#27ae60", "#2980b9"]
    labels_canon = ["X", "Y", "Z"]
    for k in range(3):
        ek = np.zeros(3)
        ek[k] = triad_len
        ax_trans.quiver(c_init[0], c_init[1], c_init[2], ek[0], ek[1], ek[2], color=colors_triad[k], linewidth=1.8, alpha=0.7, linestyle=":")
        ax_trans.text(c_init[0] + ek[0] * 1.1, c_init[1] + ek[1] * 1.1, c_init[2] + ek[2] * 1.1, f" {labels_canon[k]}", color=colors_triad[k], fontsize=7.5, fontweight="bold")

    # Rotated Triad at c_final
    labels_rot = ["X'", "Y'", "Z'"]
    for k in range(3):
        ek = np.zeros(3)
        ek[k] = triad_len
        rk = R @ ek
        ax_trans.quiver(c_final[0], c_final[1], c_final[2], rk[0], rk[1], rk[2], color=colors_triad[k], linewidth=2.4, arrow_length_ratio=0.15)
        ax_trans.text(c_final[0] + rk[0] * 1.15, c_final[1] + rk[1] * 1.15, c_final[2] + rk[2] * 1.15, f" {labels_rot[k]}", color=colors_triad[k], fontsize=8.5, fontweight="bold")

    # Faint reference axes at c_final
    for k in range(3):
        ek = np.zeros(3)
        ek[k] = triad_len * 0.7
        ax_trans.quiver(c_final[0], c_final[1], c_final[2], ek[0], ek[1], ek[2], color=colors_triad[k], linewidth=1.0, linestyle="--", alpha=0.35)

    ax_trans.set_title(
        rf"Spatial Transformation (Translation $\mathbf{{T}}$ + $SO(3)$ Rotation $R$)" + "\n"
        rf"($\Delta c={d_trans:.2f}\,\mu\mathrm{{m}}${d['q_shift_s']}, $\theta={theta_deg:.1f}^\circ${d['q_rot_s']})",
        fontsize=10.5,
        fontweight="bold",
        pad=10,
    )
    ax_trans.set_xlabel(r"$X$ ($\mu\mathrm{m}$)", fontsize=8.5)
    ax_trans.set_ylabel(r"$Y$ ($\mu\mathrm{m}$)", fontsize=8.5)
    ax_trans.set_zlabel(r"$Z$ ($\mu\mathrm{m}$)", fontsize=8.5)

    # Enforce isotropic physical dimensions for transformation panel so triads remain orthogonal
    pts_trans = [xyz_raw, d["obs_solid"], d["canon_solid"], [c_init], [c_final]]
    for k in range(3):
        ek = np.zeros(3)
        ek[k] = triad_len
        pts_trans.append([c_init + ek, c_final + ek, c_final + R @ ek])
    pts_trans_arr = np.vstack(pts_trans)
    mid_trans = np.mean(pts_trans_arr, axis=0)
    max_span_trans = max(float(np.ptp(pts_trans_arr, axis=0).max()), 2.0) * 0.55
    ax_trans.set_xlim(mid_trans[0] - max_span_trans, mid_trans[0] + max_span_trans)
    ax_trans.set_ylim(mid_trans[1] - max_span_trans, mid_trans[1] + max_span_trans)
    ax_trans.set_zlim(mid_trans[2] - max_span_trans, mid_trans[2] + max_span_trans)
    ax_trans.set_box_aspect((1, 1, 1))
    ax_trans.view_init(elev=22, azim=48)
    ax_trans.legend(frameon=True, fontsize=6.8, loc="upper left")

    # Panel 7: Birth-Anchored Developmental Lifespan Interval
    clade_color = get_canonical_clade_color(cell_name)
    norm_dur = TwoSlopeNorm(vmin=0.6, vcenter=1.0, vmax=1.4)
    cmap_dur = plt.get_cmap("coolwarm")
    obs_bar_color = cmap_dur(norm_dur(np.clip(d["dur_ratio"], 0.6, 1.4)))

    # Reference Lifespan Bar
    ax_timing.barh(
        y=1.0, width=d["ref_dur"], left=d["ref_b"], height=0.38,
        color=clade_color, edgecolor="#2c3e50", linewidth=1.2, alpha=0.92,
        label=f"Reference Lifespan ({d['ref_dur']:.1f} min)", zorder=3,
    )
    # Propagated birth time error bar
    ax_timing.errorbar(
        x=d["ref_b"], y=1.0, xerr=d["std_birth_err"],
        fmt="none", ecolor="#000000", elinewidth=2.2, capsize=6.0, capthick=1.8,
        zorder=5, label=rf"Propagated Birth Uncertainty ($\pm 1\sigma = {d['std_birth_err']:.2f}$ min)",
    )
    ax_timing.scatter([d["ref_b"]], [1.0], color="#ffffff", edgecolor="#000000", linewidth=1.5, s=70, zorder=6, label=r"Expected Birth $\mathbb{E}[t_{\mathrm{birth}}]$")

    # Observed Lifespan Bar
    ax_timing.barh(
        y=0.0, width=d["obs_dur"], left=d["obs_b"], height=0.38,
        color=obs_bar_color, edgecolor="#2c3e50", linewidth=1.4, alpha=0.95,
        label=f"Observed Lifespan ({d['obs_dur']:.1f} min, Ratio $\\rho = {d['dur_ratio']:.2f}$)", zorder=3,
    )
    ax_timing.scatter([d["obs_b"]], [0.0], color="#2c3e50", edgecolor="#ffffff", linewidth=1.5, s=75, zorder=6, label=r"Observed Birth $t_{\mathrm{birth}}$")

    # Connector line at birth anchor
    ax_timing.plot([d["ref_b"], d["obs_b"]], [1.0, 0.0], color="#7f8c8d", linestyle="--", linewidth=1.6, zorder=2)
    ax_timing.annotate(
        f"$\\Delta t_{{\\mathrm{{birth}}}} = {d['delta_birth_val']:+.1f}$ min\n($Z_{{\\mathrm{{shift}}}} = {d['z_birth_shift']:+.2f}\\sigma${d['q_shift_t']})",
        xy=(0.5 * (d["ref_b"] + d["obs_b"]), 0.5), xytext=(20, 0), textcoords="offset points",
        fontsize=8.5, fontweight="bold", color="#2c3e50",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ecf0f1", edgecolor="#bdc3c7"),
        arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.2), zorder=7,
    )

    t_bounds = [d["ref_b"] - d["std_birth_err"], d["ref_d"], d["obs_b"], d["obs_d"]]
    ax_timing.set_xlim([min(t_bounds) - 5.0, max(t_bounds) + 5.0])
    ax_timing.set_ylim([-0.55, 1.55])
    ax_timing.set_yticks([0.0, 1.0])
    emb_short = d['embryo_id'] if len(d['embryo_id']) <= 20 else d['embryo_id'][:17] + "..."
    ax_timing.set_yticklabels([f"Observed ({emb_short})", f"Reference ({cell_name})"], fontsize=9.0, fontweight="bold")
    ax_timing.set_xlabel(r"Canonical Developmental Time $t$ (min)", fontsize=9.5, fontweight="bold")
    pacing_note = f", $K_e = {d['k_pacing']:.3f}$" if abs(d.get("k_pacing", 1.0) - 1.0) > 0.005 else ""
    ax_timing.set_title(
        f"Birth-Anchored Lifespan Interval: Duration Ratio $\\rho = {d['dur_ratio']:.2f}$ ({d['pct_dur_dev']:+.1f}% | $Z_{{\\mathrm{{shape}}}} = {d['z_temp_shape']:+.2f}\\sigma${d['q_shape_t']}{pacing_note})",
        fontsize=10.5, fontweight="bold",
    )
    ax_timing.grid(True, linestyle=":", alpha=0.5, axis="x")
    ax_timing.legend(frameon=True, fontsize=7.2, loc="upper right", ncol=2)

    # Suptitle
    pacing_str = f" ($K_e = {d['k_pacing']:.3f}$)" if abs(d.get("k_pacing", 1.0) - 1.0) > 0.005 else ""
    plt.suptitle(
        f"Multi-Modal Trajectory Diagnostic Dashboard: '{cell_name}' (Embryo {d['embryo_id']}{pacing_str})\n"
        f"Temporal: $Z_{{\\mathrm{{shape}}}} = {d['z_temp_shape']:+.2f}\\sigma${d['q_shape_t']}  |  "
        f"$Z_{{\\mathrm{{shift}}}} = {d['z_birth_shift']:+.2f}\\sigma$ ({d['delta_birth_val']:+.1f} min{d['q_shift_t']})  |  "
        f"$\\mathrm{{RMS}}_{{\\mathrm{{warp}}}} = {d['rms_warp_val']:.2f}$ min{d['q_warp_t']}\n"
        f"Spatial: $\\mathcal{{D}}_{{\\mathrm{{shift}}}} = {d['d_shift_score']:.2f}$ ({d['d_shift_euclid']:.1f} $\\mu\\mathrm{{m}}${d['q_shift_s']})  |  "
        f"$\\theta_{{\\mathrm{{rot}}}} = {d['rot_angle_val']:.1f}^\\circ${d['q_rot_s']}  |  "
        f"$\\mathcal{{M}}_{{\\mathrm{{shape}}}} = {d['spat_shape_score']:.2f}$ (RMSE {d['rmse_3d_calc']:.1f} $\\mu\\mathrm{{m}}${d['q_shape_s']})",
        fontsize=12.0, fontweight="bold", y=0.992,
    )

    axes_dict = {
        "ax_x": ax_x, "ax_y": ax_y, "ax_z": ax_z,
        "ax_gamma": ax_gamma, "ax_3d": ax_3d, "ax_trans": ax_trans,
        "ax_timing": ax_timing,
    }

    fig_3d = None
    if save_interactive_html or return_interactive_3d or interactive_3d:
        html_dest = str(save_interactive_html) if isinstance(save_interactive_html, (str, Path)) else None
        fig_3d = plot_interactive_3d_trajectory(
            cell_name=cell_name,
            embryo_id=embryo_id,
            pos_df=pos_df,
            target_row=target_row,
            cell_model=cell_model,
            temporal_atlas=temporal_atlas,
            artifacts_dir=artifacts_dir,
            raw_time_col=raw_time_col,
            spatial_cols=spatial_cols,
            embryo_col=embryo_col,
            cell_col=cell_col,
            save_html=html_dest,
            k_pacing=d.get("k_pacing", k_pacing),
            embryo_qc=embryo_qc,
        )
        if interactive_3d:
            try:
                from IPython.display import display
                display(fig_3d)
            except Exception:
                pass

    if return_interactive_3d:
        return fig, axes_dict, fig_3d
    return fig, axes_dict


def plot_cell_cohort_3d_comparison(
    cell_name: str,
    pos_df: pd.DataFrame,
    cell_scores: pd.DataFrame,
    cell_model: dict | None = None,
    temporal_atlas: dict | None = None,
    atlas_bundle: dict | None = None,
    embryo_qc: pd.DataFrame | dict | None = None,
    spatial_cols: list[str] | None = None,
    raw_time_col: str = "time",
    embryo_col: str = "series",
    cell_col: str = "cell",
    ncols: int = 6,
    figsize_per_subplot: tuple[float, float] = (3.8, 3.8),
    save_html: str | Path | None = None,
    save_png: str | Path | None = None,
    dpi: int = 120,
    elev: float = 20.0,
    azim: float = 45.0,
) -> tuple[plt.Figure, list, any]:
    """Generates multi-embryo 3D aligned trajectory visualizer across a cohort for a chosen blastomere.

    Provides:
      1. Static Matplotlib figure: An N-embryo grid of 3D subplots showing the WT Consensus Reference
         path alongside each embryo's rigidly aligned trajectory in the canonical frame, with unified
         spatial bounds and orientation.
      2. Interactive Plotly 3D HTML figure: WebGL multi-scene visualization with synchronized/locked
         cameras across all viewports. Rotating, panning, or zooming any viewport updates all scenes
         in real-time.

    Parameters
    ----------
    cell_name : str
        Target blastomere identifier (e.g. 'MSapa').
    pos_df : pd.DataFrame
        Aligned tracking coordinates DataFrame.
    cell_scores : pd.DataFrame
        Table of inference scores containing 'embryo_id', 'cell', and outlier q-values.
    cell_model : dict, optional
        Fitted TrajectoryRibbon model dictionary for cell_name.
    temporal_atlas : dict, optional
        Reference temporal atlas dictionary.
    atlas_bundle : dict, optional
        Full reference atlas bundle containing 'cell_models' and 'temporal_atlas'.
    embryo_qc : pd.DataFrame or dict, optional
        Embryo QC table containing pacing rates 'k_test'.
    spatial_cols : list of str, optional
        3D spatial coordinate columns (defaults to ['x_aligned_um', 'y_aligned_um', 'z_aligned_um'] or atlas_bundle['aligned_cols']).
    raw_time_col : str, default='time'
        Time column name.
    embryo_col : str, default='series'
        Embryo series column name in pos_df.
    cell_col : str, default='cell'
        Cell name column name in pos_df.
    ncols : int, default=6
        Number of subplot columns.
    figsize_per_subplot : tuple of float, default=(3.8, 3.8)
        Size per 3D subplot in inches for static matplotlib figure.
    save_html : str or Path, optional
        Path to save interactive WebGL HTML with locked cameras.
    save_png : str or Path, optional
        Path to save static PNG figure.
    dpi : int, default=120
        DPI resolution for static PNG.
    elev : float, default=20.0
        Initial 3D elevation angle.
    azim : float, default=45.0
        Initial 3D azimuth angle.

    Returns
    -------
    fig_static : plt.Figure
        Static Matplotlib Figure.
    axes_list : list of Axes3D
        List of 3D axes objects.
    fig_plotly : plotly.graph_objects.Figure or None
        Interactive Plotly Figure (if plotly is installed).
    """
    if atlas_bundle is not None:
        if cell_model is None and "cell_models" in atlas_bundle:
            cell_model = atlas_bundle["cell_models"].get(cell_name)
        if temporal_atlas is None and "temporal_atlas" in atlas_bundle:
            temporal_atlas = atlas_bundle.get("temporal_atlas")
        if spatial_cols is None and "aligned_cols" in atlas_bundle:
            spatial_cols = atlas_bundle["aligned_cols"]

    if spatial_cols is None:
        spatial_cols = ["x_aligned_um", "y_aligned_um", "z_aligned_um"]

    if cell_model is None or temporal_atlas is None:
        raise ValueError(f"cell_model and temporal_atlas must be supplied for '{cell_name}'.")

    # Match scored embryos for this cell
    c_col_scores = "cell" if "cell" in cell_scores.columns else "cell_name"
    e_col_scores = "embryo_id" if "embryo_id" in cell_scores.columns else ("series" if "series" in cell_scores.columns else embryo_col)

    c_match = cell_scores[cell_scores[c_col_scores] == cell_name]
    if c_match.empty:
        raise ValueError(f"No records found in cell_scores for cell '{cell_name}'.")

    emb_ids = sorted(c_match[e_col_scores].astype(str).unique())

    # Prepare data for each embryo
    data_list = []
    for emb_id in emb_ids:
        sub_row = c_match[c_match[e_col_scores].astype(str) == emb_id]
        target_row = sub_row.iloc[0] if not sub_row.empty else None
        try:
            d = _prepare_trajectory_dashboard_data(
                cell_name=cell_name,
                embryo_id=emb_id,
                pos_df=pos_df,
                target_row=target_row,
                cell_model=cell_model,
                temporal_atlas=temporal_atlas,
                spatial_cols=spatial_cols,
                embryo_col=embryo_col,
                cell_col=cell_col,
                embryo_qc=embryo_qc,
            )
            data_list.append(d)
        except Exception as e:
            # Skip if missing observations in pos_df
            continue

    if not data_list:
        raise ValueError(f"Could not prepare trajectory data for cell '{cell_name}' across any embryos.")

    # Calculate global bounding box over all trajectories to enforce common scales
    all_pts = []
    for d in data_list:
        all_pts.append(d["canon_solid"])
        if d["canon_dashed"] is not None:
            all_pts.append(d["canon_dashed"])
        all_pts.append(d["obs_solid"])
        if d["obs_dashed"] is not None:
            all_pts.append(d["obs_dashed"])
    all_pts = np.vstack(all_pts)

    min_xyz = all_pts.min(axis=0)
    max_xyz = all_pts.max(axis=0)
    center_xyz = 0.5 * (min_xyz + max_xyz)
    max_span = np.max(max_xyz - min_xyz) * 0.55
    lims = [
        [center_xyz[0] - max_span, center_xyz[0] + max_span],
        [center_xyz[1] - max_span, center_xyz[1] + max_span],
        [center_xyz[2] - max_span, center_xyz[2] + max_span],
    ]

    # 1. Build Static Matplotlib Grid
    n_embs = len(data_list)
    cols = min(ncols, n_embs)
    rows = int(np.ceil(n_embs / cols))
    fig_w = figsize_per_subplot[0] * cols
    fig_h = figsize_per_subplot[1] * rows + 1.2

    fig_static = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    axes_list = []

    for i, d in enumerate(data_list):
        ax = fig_static.add_subplot(rows, cols, i + 1, projection="3d")
        axes_list.append(ax)

        # Reference
        ax.plot(d["canon_solid"][:, 0], d["canon_solid"][:, 1], d["canon_solid"][:, 2], color="#2c3e50", linewidth=2.0)
        if d["canon_dashed"] is not None:
            ax.plot(d["canon_dashed"][:, 0], d["canon_dashed"][:, 1], d["canon_dashed"][:, 2], color="#2c3e50", linewidth=1.4, linestyle="--", alpha=0.6)

        # Observed track
        is_outlier = bool(
            d["target_row"] is not None and (
                d["target_row"].get("hit_spat_shape", False)
                or d["target_row"].get("hit_spat_shift", False)
                or d["target_row"].get("is_any_outlier", False)
            )
        )
        obs_color = "#d63031" if is_outlier else "#2980b9"
        ax.plot(d["obs_solid"][:, 0], d["obs_solid"][:, 1], d["obs_solid"][:, 2], color=obs_color, linewidth=1.8, marker="o", markersize=2.2)
        if d["obs_dashed"] is not None:
            ax.plot(d["obs_dashed"][:, 0], d["obs_dashed"][:, 1], d["obs_dashed"][:, 2], color=obs_color, linewidth=1.2, linestyle="--", marker="s", markersize=1.8, alpha=0.7)

        # Start / End markers
        ax.scatter([d["p_ref_start"][0]], [d["p_ref_start"][1]], [d["p_ref_start"][2]], color="#2c3e50", s=32, marker="o")
        ax.scatter([d["p_obs_start"][0]], [d["p_obs_start"][1]], [d["p_obs_start"][2]], color=obs_color, s=32, marker="o")
        ax.scatter([d["p_ref_end"][0]], [d["p_ref_end"][1]], [d["p_ref_end"][2]], color="#2c3e50", s=32, marker="^")
        ax.scatter([d["p_obs_end"][0]], [d["p_obs_end"][1]], [d["p_obs_end"][2]], color=obs_color, s=32, marker="^")

        ax.set_xlim(lims[0])
        ax.set_ylim(lims[1])
        ax.set_zlim(lims[2])
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect([1, 1, 1])

        # Subplot Title
        emb_lbl = d["embryo_id"]
        if len(emb_lbl) > 22:
            emb_lbl = emb_lbl[:10] + "..." + emb_lbl[-9:]
        q_val = d["target_row"].get("qval_spat_shape", np.nan) if d["target_row"] is not None else np.nan
        q_str = f"q_shape={q_val:.2e}" if pd.notna(q_val) and q_val < 0.05 else (f"q_shape={q_val:.2f}" if pd.notna(q_val) else "")
        ax.set_title(f"{emb_lbl}\n{q_str}", fontsize=8.5, fontweight="bold", pad=2)
        ax.tick_params(labelsize=6, pad=0.5)

    fig_static.suptitle(
        f"{cell_name}: 3D Aligned Trajectories Across All {n_embs} pop-1(RNAi) Embryos\n"
        "(Black: WT Consensus Path; Red: pop-1 Aligned Track; Circles: Start; Triangles: End)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_png:
        png_path = Path(save_png)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig_static.savefig(png_path, dpi=dpi, bbox_inches="tight")

    # 2. Build Interactive Plotly 3D Figure with Locked Camera Synchronization
    fig_plotly = None
    if go is not None and make_subplots is not None:
        titles = []
        for d in data_list:
            emb_lbl = d["embryo_id"]
            if len(emb_lbl) > 22:
                emb_lbl = emb_lbl[:10] + "..." + emb_lbl[-9:]
            q_val = d["target_row"].get("qval_spat_shape", np.nan) if d["target_row"] is not None else np.nan
            q_str = f" (q={q_val:.2e})" if pd.notna(q_val) and q_val < 0.05 else ""
            titles.append(f"{emb_lbl}{q_str}")

        fig_plotly = make_subplots(
            rows=rows,
            cols=cols,
            specs=[[{"type": "scene"} for _ in range(cols)] for _ in range(rows)],
            subplot_titles=titles,
            horizontal_spacing=0.015,
            vertical_spacing=0.04,
        )

        all_scenes = []
        for idx, d in enumerate(data_list):
            r = (idx // cols) + 1
            c = (idx % cols) + 1
            scene_name = "scene" if idx == 0 else f"scene{idx + 1}"
            all_scenes.append(scene_name)

            show_leg = (idx == 0)
            # WT Ref
            fig_plotly.add_trace(go.Scatter3d(
                x=d["canon_solid"][:, 0], y=d["canon_solid"][:, 1], z=d["canon_solid"][:, 2],
                mode="lines",
                line=dict(color="#2c3e50", width=5),
                name="WT Reference Path",
                legendgroup="ref",
                showlegend=show_leg,
                hoverinfo="text",
                hovertext=[f"WT Ref: X={pt[0]:.2f}, Y={pt[1]:.2f}, Z={pt[2]:.2f} μm" for pt in d["canon_solid"]],
            ), row=r, col=c)

            if d["canon_dashed"] is not None:
                fig_plotly.add_trace(go.Scatter3d(
                    x=d["canon_dashed"][:, 0], y=d["canon_dashed"][:, 1], z=d["canon_dashed"][:, 2],
                    mode="lines",
                    line=dict(color="#2c3e50", width=4, dash="dash"),
                    name="WT Ref Overhang",
                    legendgroup="ref_overhang",
                    showlegend=show_leg,
                    hoverinfo="skip",
                ), row=r, col=c)

            # Observed
            is_outlier = bool(
                d["target_row"] is not None and (
                    d["target_row"].get("hit_spat_shape", False)
                    or d["target_row"].get("hit_spat_shift", False)
                    or d["target_row"].get("is_any_outlier", False)
                )
            )
            obs_color = "#d63031" if is_outlier else "#2980b9"
            fig_plotly.add_trace(go.Scatter3d(
                x=d["obs_solid"][:, 0], y=d["obs_solid"][:, 1], z=d["obs_solid"][:, 2],
                mode="lines+markers",
                line=dict(color=obs_color, width=4),
                marker=dict(size=3, color=obs_color),
                name="pop-1 Aligned Track",
                legendgroup="obs",
                showlegend=show_leg,
                hoverinfo="text",
                hovertext=[f"{d['embryo_id']}: τ={t:.1f} min<br>X={pt[0]:.2f}, Y={pt[1]:.2f}, Z={pt[2]:.2f} μm" for t, pt in zip(d["tau_active"], d["obs_solid"])],
            ), row=r, col=c)

            if d["obs_dashed"] is not None:
                fig_plotly.add_trace(go.Scatter3d(
                    x=d["obs_dashed"][:, 0], y=d["obs_dashed"][:, 1], z=d["obs_dashed"][:, 2],
                    mode="lines+markers",
                    line=dict(color=obs_color, width=3, dash="dash"),
                    marker=dict(size=2.5, color=obs_color, symbol="square"),
                    name="pop-1 Track Overhang",
                    legendgroup="obs_overhang",
                    showlegend=show_leg,
                    hoverinfo="skip",
                ), row=r, col=c)

            # Start and End markers
            fig_plotly.add_trace(go.Scatter3d(
                x=[d["p_ref_start"][0], d["p_obs_start"][0]],
                y=[d["p_ref_start"][1], d["p_obs_start"][1]],
                z=[d["p_ref_start"][2], d["p_obs_start"][2]],
                mode="markers",
                marker=dict(size=[6, 6], color=["#2c3e50", obs_color], symbol="circle"),
                name="Start Point",
                legendgroup="start",
                showlegend=show_leg,
                hoverinfo="skip",
            ), row=r, col=c)

            fig_plotly.add_trace(go.Scatter3d(
                x=[d["p_ref_end"][0], d["p_obs_end"][0]],
                y=[d["p_ref_end"][1], d["p_obs_end"][1]],
                z=[d["p_ref_end"][2], d["p_obs_end"][2]],
                mode="markers",
                marker=dict(size=[6, 6], color=["#2c3e50", obs_color], symbol="diamond"),
                name="End / Division Point",
                legendgroup="end",
                showlegend=show_leg,
                hoverinfo="skip",
            ), row=r, col=c)

        layout_updates = dict(
            title=dict(
                text=f"<b>{cell_name}</b>: 3D Aligned Trajectories Across All {n_embs} pop-1(RNAi) Embryos (Locked Camera Viewports)",
                font=dict(size=18),
                x=0.5,
            ),
            height=rows * 380,
            width=cols * 280,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(l=20, r=20, t=80, b=20),
        )

        for s in all_scenes:
            layout_updates[s] = dict(
                xaxis=dict(range=lims[0], title="X (μm)", showticklabels=True),
                yaxis=dict(range=lims[1], title="Y (μm)", showticklabels=True),
                zaxis=dict(range=lims[2], title="Z (μm)", showticklabels=True),
                aspectmode="data",
                camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
            )

        fig_plotly.update_layout(**layout_updates)

        if save_html:
            html_path = Path(save_html)
            html_path.parent.mkdir(parents=True, exist_ok=True)
            html_str = fig_plotly.to_html(include_plotlyjs="cdn", full_html=True)
            all_scenes_json = str(all_scenes)
            sync_script = f"""
<script>
document.addEventListener("DOMContentLoaded", function() {{
    var gd = document.getElementsByClassName('plotly-graph-div')[0];
    if (!gd) return;
    var isUpdating = false;
    var allScenes = {all_scenes_json};

    gd.on('plotly_relayout', function(eventData) {{
        if (isUpdating || !eventData) return;
        var changedCamera = null;
        for (var key in eventData) {{
            if (key.indexOf('camera') !== -1) {{
                if (key.endsWith('.camera')) {{
                    changedCamera = eventData[key];
                    break;
                }}
                var parts = key.split('.');
                var sceneName = parts[0];
                if (gd.layout && gd.layout[sceneName] && gd.layout[sceneName].camera) {{
                    changedCamera = gd.layout[sceneName].camera;
                    break;
                }}
            }}
        }}

        if (changedCamera) {{
            isUpdating = true;
            var update = {{}};
            for (var i = 0; i < allScenes.length; i++) {{
                update[allScenes[i] + '.camera'] = changedCamera;
            }}
            Plotly.relayout(gd, update).then(function() {{
                isUpdating = false;
            }}).catch(function() {{
                isUpdating = false;
            }});
        }}
    }});
}});
</script>
</body>
"""
            html_synced = html_str.replace("</body>", sync_script)
            html_path.write_text(html_synced, encoding="utf-8")

    return fig_static, axes_list, fig_plotly


def plot_wt_null_metric_distributions(
    scores_df: pd.DataFrame,
    figsize: tuple[float, float] = (14.0, 13.5),
    dpi: int = 140,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plots 6-panel dual-axis comparison of empirical physical vs. standardized scaled distributions.

    All temporal physical metrics are rendered in canonical minutes:
      1. Temporal Shape: Lifespan deviation |Delta t_dur| (min) vs. Standardized |Z_temp_shape|
      2. Temporal Shift: Midpoint shift |Delta t_birth| (min) vs. Path-normalized |Z_temp_shift|
      3. Spatiotemporal Warp: Pacing distortion RMS_warp (min) vs. Standardized Z_warp
      4. Spatial Shift: COM displacement ||Delta c|| (um) vs. Joint GP Mahalanobis D_spat_shift
      5. Spatial Orientation: Geodesic rotation theta_rot (deg) vs. Standardized Z_rot_angle
      6. Spatial Shape: Post-alignment 3D RMSE (um) vs. Kronecker GP Mahalanobis M_spat_shape

    Parameters
    ----------
    scores_df : pd.DataFrame
        Table containing empirical WT null scores across blastomeres (e.g. cv_scores_all_100.parquet).
    figsize : tuple of float, default=(14.0, 13.5)
        Matplotlib figure size.
    dpi : int, default=140
        Resolution for rasterization/export.
    save_path : str or Path, optional
        Path to save rendered PNG image.

    Returns
    -------
    fig : plt.Figure
        Rendered figure.
    axes : np.ndarray of Axes
        2D array of primary Matplotlib Axes (shape 3x2).
    """
    import seaborn as sns

    fig, axes = plt.subplots(3, 2, figsize=figsize, dpi=dpi)
    plt.subplots_adjust(hspace=0.55, wspace=0.28)

    # Resolve duration deviation in canonical minutes:
    if "delta_duration_min" in scores_df.columns:
        dur_phys = scores_df["delta_duration_min"].abs()
    elif "canon_duration" in scores_df.columns and "obs_duration" in scores_df.columns:
        dur_phys = (scores_df["obs_duration"] - scores_df["canon_duration"]).abs()
    elif "pct_duration_deviation" in scores_df.columns and "canon_duration" in scores_df.columns:
        dur_phys = (scores_df["pct_duration_deviation"].abs() / 100.0) * scores_df["canon_duration"]
    else:
        dur_phys = scores_df.get("delta_duration_min", scores_df.get("pct_duration_deviation", pd.Series(dtype=float))).abs()

    # Resolve birth / midpoint shift in canonical minutes:
    if "delta_birth_min" in scores_df.columns:
        shift_phys = scores_df["delta_birth_min"].abs()
    elif "delta_midpoint_min" in scores_df.columns:
        shift_phys = scores_df["delta_midpoint_min"].abs()
    else:
        shift_phys = scores_df.get("delta_birth_min", pd.Series(dtype=float)).abs()

    metric_pairs = [
        (
            axes[0, 0],
            dur_phys,
            scores_df["z_temp_shape"].abs(),
            "Temporal Shape",
            r"|Lifespan Deviation| $|\Delta t_{\mathrm{dur}}|$ (min)",
            "Standardized |Z_temp_shape|",
            "min",
            "#1f77b4", "#3498db"
        ),
        (
            axes[0, 1],
            shift_phys,
            scores_df["z_temp_shift"].abs(),
            "Temporal Shift",
            r"Midpoint Shift $|\Delta t_{\mathrm{birth}}|$ (min)",
            "Path-Normalized |Z_temp_shift|",
            "min",
            "#2ca02c", "#2ecc71"
        ),
        (
            axes[1, 0],
            scores_df["rms_warp_min"],
            scores_df["z_warp"],
            "Spatiotemporal Warp",
            r"Pacing Distortion $\mathrm{RMS}_{\mathrm{warp}}$ (min)",
            "Standardized Z_warp",
            "min",
            "#ff7f0e", "#e67e22"
        ),
        (
            axes[1, 1],
            scores_df["com_shift_um"],
            scores_df["d_spat_shift"],
            "Spatial Shift",
            r"COM Displacement $\|\Delta \mathbf{c}\|$ ($\mu\mathrm{m}$)",
            r"Joint GP Mahalanobis $\mathcal{D}_{\mathrm{spat\_shift}}$",
            "μm",
            "#d62728", "#e74c3c"
        ),
        (
            axes[2, 0],
            scores_df["rot_angle_deg"],
            scores_df["z_rot_angle"],
            "Spatial Orientation",
            r"Geodesic Rotation $\theta_{\mathrm{rot}}$ ($^\circ$)",
            "Standardized Z_rot_angle",
            "°",
            "#9467bd", "#9b59b6"
        ),
        (
            axes[2, 1],
            scores_df["rmse_3d_um"],
            scores_df["d_spat_shape"],
            "Spatial Shape",
            r"Post-Alignment 3D RMSE ($\mu\mathrm{m}$)",
            r"Kronecker GP Mahalanobis $\mathcal{M}_{\mathrm{spat\_shape}}$",
            "μm",
            "#8c564b", "#a0522d"
        ),
    ]

    for ax, phys_s, scaled_s, title, x_lbl_phys, x_lbl_scaled, unit_str, c_dark, c_light in metric_pairs:
        ax_twin = ax.twiny()

        # Plot Physical (bottom)
        sns.kdeplot(phys_s, ax=ax, color=c_dark, fill=True, alpha=0.25, linewidth=2.0, label="Physical Unit")
        ax.set_xlabel(x_lbl_phys, color=c_dark, fontweight="bold", fontsize=9.5)
        ax.tick_params(axis="x", labelcolor=c_dark)
        ax.set_ylabel("Empirical Density", fontsize=9.5)
        ax.grid(True, linestyle=":", alpha=0.5)

        # Plot Scaled (top)
        sns.kdeplot(scaled_s, ax=ax_twin, color="#2c3e50", linestyle="--", linewidth=2.0, label="Scaled (Dimensionless)")
        ax_twin.set_xlabel(x_lbl_scaled, color="#2c3e50", fontweight="bold", fontsize=9.5)
        ax_twin.tick_params(axis="x", labelcolor="#2c3e50")

        # Title positioned cleanly above top axis
        p_med = float(phys_s.median()) if not phys_s.empty else np.nan
        s_med = float(scaled_s.median()) if not scaled_s.empty else np.nan
        ax_twin.set_title(f"{title}: Median {p_med:.2f} {unit_str} vs {s_med:.2f} scaled", fontsize=11, fontweight="bold", pad=14)

    n_obs = len(scores_df)
    n_embs = scores_df["embryo_id"].nunique() if "embryo_id" in scores_df.columns else 100
    fig.suptitle(
        f"Empirical Metric Distributions: Physical vs. Standardized Scaled Units\n({n_obs:,} Blastomere Observations across {n_embs} Wild-Type Embryos)",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")

    return fig, axes

