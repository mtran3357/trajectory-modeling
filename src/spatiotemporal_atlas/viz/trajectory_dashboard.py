"""Single-cell trajectory diagnostic dashboard across the 6-modality spatiotemporal suite."""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from ..functional.time_warp import regularized_monotonic_time_warp, compute_warp_metrics
from ..geometry.curve_align import register_curve_to_template
from ..models.joint_gp import compute_3d_mahalanobis_residuals
from .colors import get_canonical_clade_color


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
    figsize: tuple[float, float] = (20.0, 14.5),
    dpi: int = 130,
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    """Renders a comprehensive 6-panel single-cell spatiotemporal trajectory diagnostic dashboard.
    
    Panels:
      1-3: X(tau), Y(tau), Z(tau) progression profiles with Joint GP ribbon, aligned/warped
           query points, and raw-centered trajectory (light dashed line) for visual contrast.
      4:   Regularized monotonic time-warping diffeomorphism gamma(tau) on [0, tau_cutoff].
      5:   3D morphogenetic trajectory, Center-of-Mass Mahalanobis displacement, and SO(3) rotation.
      6:   Birth-anchored developmental lifespan interval comparison with tree-propagated uncertainty.
      
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
    figsize : tuple of (float, float), default=(20, 14.5)
        Matplotlib figure dimensions.
    dpi : int, default=130
        Resolution.
        
    Returns
    -------
    fig : plt.Figure
        Rendered figure.
    axes_dict : dict of str -> plt.Axes
        Dictionary of panel axes.
    """
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
    tau_obs = t_raw - t_birth

    tau_grid = np.asarray(model_dict.get("time_grid", np.linspace(0, 30, 30)), dtype=float)
    tau_cutoff = float(model_dict.get("tau_cutoff", tau_grid[-1] if len(tau_grid) > 0 else 30.0))
    if tau_cutoff <= 0:
        tau_cutoff = tau_grid[-1] if len(tau_grid) > 0 else 30.0

    valid_mask = tau_obs <= tau_cutoff + 1e-4
    if np.sum(valid_mask) < 2:
        valid_mask = np.ones(len(tau_obs), dtype=bool)

    tau_active = tau_obs[valid_mask]
    xyz_raw = sub_obs[spatial_cols].values.astype(float)[valid_mask]
    com_obs = np.mean(xyz_raw, axis=0)
    xyz_centered = xyz_raw - com_obs

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

    # 4. Regularized Monotonic Time Warping
    gamma_test, rms_warp, slopes = regularized_monotonic_time_warp(
        tau_obs=tau_active,
        coords_aligned=coords_aligned,
        template_curve=template_curve,
        tau_grid=tau_grid,
        lambda_reg=10.0,
        slope_bounds=(0.5, 2.0),
    )
    active_tau_grid = tau_grid[tau_grid <= tau_active.max() + 1e-4]
    warp_met = compute_warp_metrics(gamma_test, active_tau_grid, slopes)
    u_eval = np.clip(gamma_test / max(tau_cutoff, 1e-3), 0.0, 1.0)
    xyz_eval = np.column_stack([np.interp(gamma_test, tau_active, coords_aligned[:, d]) for d in range(3)])

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

    # 6. Center of Mass & Spatial Shift via Joint GP B
    mu_com = np.asarray(model_dict["mu_com"], dtype=float)
    diff_com = com_obs - mu_com
    d_shift_euclid = float(np.linalg.norm(diff_com))
    if model_dict.get("B_cov") is not None:
        inv_B = np.linalg.pinv(model_dict["B_cov"])
        d_shift_score = float(np.sqrt(max(diff_com.T @ inv_B @ diff_com, 0.0)))
    else:
        inv_cov_com = np.asarray(model_dict.get("inv_cov_com", np.eye(3)), dtype=float)
        d_shift_score = float(np.sqrt(max(diff_com.T @ inv_cov_com @ diff_com, 0.0)))

    # 7. Timing & Birth Offset
    t_stat = temporal_atlas[cell_name]
    ref_b = float(t_stat["mu_birth"])
    ref_dur = float(t_stat["mu_phys"])
    ref_d = ref_b + ref_dur

    obs_b = float(target_row["canon_birth"]) if target_row is not None and "canon_birth" in target_row else t_birth
    obs_dur = float(target_row["canon_duration"]) if target_row is not None and "canon_duration" in target_row else float(t_raw.max() - t_birth)
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

    # Q-value callout strings
    def _q_str(col_name: str) -> str:
        if target_row is not None and col_name in target_row and pd.notna(target_row[col_name]):
            q = float(target_row[col_name])
            return f", $q = {q:.3e}$" if q < 0.05 else f", $q = {q:.2f}$"
        return ""

    q_shape_t = _q_str("qval_temp_shape")
    q_shift_t = _q_str("qval_temp_shift")
    q_warp_t = _q_str("qval_warp")
    q_shift_s = _q_str("qval_spat_shift")
    q_rot_s = _q_str("qval_spat_rot")
    q_shape_s = _q_str("qval_spat_shape")

    # 8. Render Figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(
        3, 3,
        height_ratios=[1.0, 1.0, 0.85],
        width_ratios=[1.0, 1.0, 1.25],
        hspace=0.38,
        wspace=0.26,
    )

    ax_x = fig.add_subplot(gs[0, 0])
    ax_y = fig.add_subplot(gs[0, 1])
    ax_z = fig.add_subplot(gs[1, 0])
    ax_gamma = fig.add_subplot(gs[1, 1])
    ax_3d = fig.add_subplot(gs[0:2, 2], projection="3d")
    ax_timing = fig.add_subplot(gs[2, :])

    coord_axes = [ax_x, ax_y, ax_z]
    axis_labels = [r"$X$", r"$Y$", r"$Z$"]

    # Panels 1-3: Coordinate Fits
    for idx, (ax, col, lbl) in enumerate(zip(coord_axes, spatial_cols, axis_labels)):
        mu = gp_preds[col]["mu"]
        std = gp_preds[col]["std"]

        # GP Reference Ribbon
        ax.plot(tau_dense, mu, color="#2c3e50", linewidth=2.2, label="Joint GP Reference", zorder=3)
        if np.any(std > 0):
            ax.fill_between(
                tau_dense, mu - 1.96 * std, mu + 1.96 * std,
                color="#3498db", alpha=0.22, label=r"$\pm 1.96\sigma$ Ribbon", zorder=2,
            )

        # Raw centered trajectory (light dashed line) showing pre-aligned/pre-warped contrast
        ax.plot(
            tau_active, xyz_centered[:, idx],
            color="#95a5a6", linestyle="--", linewidth=1.5, alpha=0.75,
            label="Raw Centered Track", zorder=4,
        )

        # Aligned & warped query points
        ax.scatter(
            active_tau_grid, xyz_eval[:, idx],
            color="#d63031", edgecolor="black", linewidth=0.6, s=34,
            label=f"Aligned & Warped ({embryo_id})", zorder=5,
        )

        # Residual drop-lines to GP mean
        for k in range(n_eval):
            ax.plot(
                [active_tau_grid[k], active_tau_grid[k]],
                [pred_at_obs[k, idx], xyz_eval[k, idx]],
                color="#d63031", linestyle=":", linewidth=1.0, zorder=4,
            )


        ax.set_title(rf"{lbl}($\tau$) Progression Profile ($\tau \in [0, {tau_cutoff:.1f}]$ min)", fontsize=10.5, fontweight="bold")
        ax.set_xlabel(r"Elapsed Canonical Time $\tau$ (min)", fontsize=9)
        ax.set_ylabel(rf"{lbl} ($\mu\mathrm{{m}}$)", fontsize=9)
        ax.set_xlim([0.0, tau_cutoff])
        ax.grid(True, linestyle=":", alpha=0.5)
        if idx == 0:
            ax.legend(frameon=True, fontsize=7.2, loc="best")

    # Panel 4: Monotonic Warping Diffeomorphism gamma(tau)
    ax_gamma.plot([0, tau_cutoff], [0, tau_cutoff], "k--", linewidth=1.4, label=r"Uniform Pacing ($\gamma_{\mathrm{id}}(\tau) = \tau$)")
    ax_gamma.plot(active_tau_grid, gamma_test, color="#e67e22", linewidth=2.4, label=rf"$\hat{{\gamma}}(\tau)$ ($\mathrm{{RMS}} = {rms_warp_val:.2f}\,$min)")
    ax_gamma.fill_between(active_tau_grid, active_tau_grid, gamma_test, color="#f39c12", alpha=0.22, label="Pacing Distortion Area")
    ax_gamma.scatter(active_tau_grid, gamma_test, color="#d63031", s=28, edgecolor="black", linewidth=0.5, zorder=4)


    ax_gamma.set_title(rf"Monotonic Warping $\hat{{\gamma}}(\tau)$ ($\mathrm{{RMS}}_{{\mathrm{{warp}}}} = {rms_warp_val:.2f}\,$min{q_warp_t})", fontsize=10.5, fontweight="bold")
    ax_gamma.set_xlabel(r"Elapsed Canonical Time $\tau$ (min)", fontsize=9)
    ax_gamma.set_ylabel(r"Warped Progression $\hat{\gamma}(\tau)$ (min)", fontsize=9)
    ax_gamma.set_xlim([0.0, tau_cutoff])
    ax_gamma.set_ylim([0.0, max(tau_cutoff, float(gamma_test.max()) + 1.0)])
    ax_gamma.grid(True, linestyle=":", alpha=0.5)
    ax_gamma.legend(frameon=True, fontsize=7.2, loc="upper left")

    # Panel 5: 3D Trajectory & Spatial Alignment
    canon_3d = np.column_stack([
        gp_preds[spatial_cols[0]]["mu"] + mu_com[0],
        gp_preds[spatial_cols[1]]["mu"] + mu_com[1],
        gp_preds[spatial_cols[2]]["mu"] + mu_com[2],
    ])
    raw_3d = xyz_raw

    # Canonical reference path
    ax_3d.plot(canon_3d[:, 0], canon_3d[:, 1], canon_3d[:, 2], color="#2c3e50", linewidth=2.4, label="Canonical Reference Path")
    # Raw observed track (pre-rotation)
    ax_3d.plot(raw_3d[:, 0], raw_3d[:, 1], raw_3d[:, 2], color="#95a5a6", linestyle="--", linewidth=1.4, alpha=0.75, label="Raw Track (pre-rotation)")
    # Rotated observed track
    rotated_3d = coords_aligned + com_obs
    ax_3d.plot(rotated_3d[:, 0], rotated_3d[:, 1], rotated_3d[:, 2], color="#d63031", linewidth=2.0, marker="o", markersize=3.5, label=f"Rotated Track ($\mathbf{{R}}_{{\mathrm{{test}}}}$)")


    # Center of Mass markers
    ax_3d.scatter(mu_com[0], mu_com[1], mu_com[2], color="#2c3e50", s=65, marker="^", label=r"$\boldsymbol{\mu}_{\mathrm{COM}}$")
    ax_3d.scatter(com_obs[0], com_obs[1], com_obs[2], color="#d63031", s=65, marker="^", label=r"$\bar{\mathbf{x}}_{\mathrm{obs}}$")
    ax_3d.plot(
        [mu_com[0], com_obs[0]], [mu_com[1], com_obs[1]], [mu_com[2], com_obs[2]],
        "k--", linewidth=1.2,
        label=rf"$\mathcal{{D}}_{{\mathrm{{shift}}}} = {d_shift_score:.2f}$ ({d_shift_euclid:.1f} $\mu\mathrm{{m}}$)",
    )

    ax_3d.set_title(rf"3D Spatial Alignment ($\theta_{{\mathrm{{rot}}}} = {rot_angle_val:.1f}^\circ${q_rot_s})", fontsize=11, fontweight="bold")
    ax_3d.set_xlabel(r"$X$ ($\mu\mathrm{m}$)", fontsize=8.5)
    ax_3d.set_ylabel(r"$Y$ ($\mu\mathrm{m}$)", fontsize=8.5)
    ax_3d.set_zlabel(r"$Z$ ($\mu\mathrm{m}$)", fontsize=8.5)
    ax_3d.legend(frameon=True, fontsize=6.8, loc="upper right")

    # Panel 6: Birth-Anchored Developmental Lifespan Interval
    clade_color = get_canonical_clade_color(cell_name)
    norm_dur = TwoSlopeNorm(vmin=0.6, vcenter=1.0, vmax=1.4)
    cmap_dur = plt.get_cmap("coolwarm")
    obs_bar_color = cmap_dur(norm_dur(np.clip(dur_ratio, 0.6, 1.4)))

    # Reference Lifespan Bar
    ax_timing.barh(
        y=1.0, width=ref_dur, left=ref_b, height=0.38,
        color=clade_color, edgecolor="#2c3e50", linewidth=1.2, alpha=0.92,
        label=f"Reference Lifespan ({ref_dur:.1f} min)", zorder=3,
    )
    # Propagated birth time error bar
    ax_timing.errorbar(
        x=ref_b, y=1.0, xerr=std_birth_err,
        fmt="none", ecolor="#000000", elinewidth=2.2, capsize=6.0, capthick=1.8,
        zorder=5, label=rf"Propagated Birth Uncertainty ($\pm 1\sigma = {std_birth_err:.2f}$ min)",
    )
    ax_timing.scatter([ref_b], [1.0], color="#ffffff", edgecolor="#000000", linewidth=1.5, s=70, zorder=6, label=r"Expected Birth $\mathbb{E}[t_{\mathrm{birth}}]$")

    # Observed Lifespan Bar
    ax_timing.barh(
        y=0.0, width=obs_dur, left=obs_b, height=0.38,
        color=obs_bar_color, edgecolor="#2c3e50", linewidth=1.4, alpha=0.95,
        label=f"Observed Lifespan ({obs_dur:.1f} min, Ratio $\\rho = {dur_ratio:.2f}$)", zorder=3,
    )
    ax_timing.scatter([obs_b], [0.0], color="#2c3e50", edgecolor="#ffffff", linewidth=1.5, s=75, zorder=6, label=r"Observed Birth $t_{\mathrm{birth}}$")

    # Connector line at birth anchor
    ax_timing.plot([ref_b, obs_b], [1.0, 0.0], color="#7f8c8d", linestyle="--", linewidth=1.6, zorder=2)
    ax_timing.annotate(
        f"$\\Delta t_{{\\mathrm{{birth}}}} = {delta_birth_val:+.1f}$ min\n($Z_{{\\mathrm{{shift}}}} = {z_birth_shift:+.2f}\\sigma${q_shift_t})",
        xy=(0.5 * (ref_b + obs_b), 0.5), xytext=(20, 0), textcoords="offset points",
        fontsize=9.0, fontweight="bold", color="#2c3e50",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ecf0f1", edgecolor="#bdc3c7"),
        arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.2), zorder=7,
    )

    t_bounds = [ref_b - std_birth_err, ref_d, obs_b, obs_d]
    ax_timing.set_xlim([min(t_bounds) - 5.0, max(t_bounds) + 5.0])
    ax_timing.set_ylim([-0.55, 1.55])
    ax_timing.set_yticks([0.0, 1.0])
    ax_timing.set_yticklabels([f"Observed ({embryo_id})", f"Reference ({cell_name})"], fontsize=10.5, fontweight="bold")
    ax_timing.set_xlabel(r"Canonical Developmental Time $t$ (min)", fontsize=10.5, fontweight="bold")
    ax_timing.set_title(
        f"Birth-Anchored Lifespan Interval: Duration Ratio $\\rho = {dur_ratio:.2f}$ ({pct_dur_dev:+.1f}% | $Z_{{\\mathrm{{shape}}}} = {z_temp_shape:+.2f}\\sigma${q_shape_t})",
        fontsize=11.5, fontweight="bold",
    )
    ax_timing.grid(True, linestyle=":", alpha=0.5, axis="x")
    ax_timing.legend(frameon=True, fontsize=8.0, loc="upper right", ncol=3)

    # Suptitle
    plt.suptitle(
        f"Multi-Modal Trajectory Diagnostic Dashboard: '{cell_name}' (Embryo {embryo_id})\n"
        f"Temporal: $Z_{{\\mathrm{{shape}}}} = {z_temp_shape:+.2f}\\sigma${q_shape_t}  |  "
        f"$Z_{{\\mathrm{{shift}}}} = {z_birth_shift:+.2f}\\sigma$ ({delta_birth_val:+.1f} min{q_shift_t})  |  "
        f"$\\mathrm{{RMS}}_{{\\mathrm{{warp}}}} = {rms_warp_val:.2f}$ min{q_warp_t}\n"
        f"Spatial: $\\mathcal{{D}}_{{\\mathrm{{shift}}}} = {d_shift_score:.2f}$ ({d_shift_euclid:.1f} $\\mu\\mathrm{{m}}${q_shift_s})  |  "
        f"$\\theta_{{\\mathrm{{rot}}}} = {rot_angle_val:.1f}^\\circ${q_rot_s}  |  "
        f"$\\mathcal{{M}}_{{\\mathrm{{shape}}}} = {spat_shape_score:.2f}$ (RMSE {rmse_3d_calc:.1f} $\\mu\\mathrm{{m}}${q_shape_s})",
        fontsize=12.0, fontweight="bold", y=0.992,
    )



    axes_dict = {
        "ax_x": ax_x, "ax_y": ax_y, "ax_z": ax_z,
        "ax_gamma": ax_gamma, "ax_3d": ax_3d, "ax_timing": ax_timing,
    }
    return fig, axes_dict
