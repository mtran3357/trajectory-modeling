"""5-Metric Manhattan plot across multi-modal empirical trajectory anomaly benchmarks."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .colors import LINEAGE_PALETTE, assign_lineage


def plot_5metric_manhattan(
    test_results_df: pd.DataFrame,
    calib_meta: dict | None = None,
    alpha: float = 0.05,
    top_n_labels: int = 4,
    figsize: tuple[float, float] = (20.0, 22.0),
    dpi: int = 130,
) -> tuple[plt.Figure, np.ndarray]:
    """Generates a 5-panel Manhattan plot across the full empirical benchmark suite.
    
    Panels:
      A. Temporal Shape: Autonomous Duration Anomaly (-log10 p_temp_shape)
      B. Temporal Shift: Propagated Tree Phase Drift (-log10 p_temp_shift)
      C. Spatial Shift: Center-of-Mass Misplacement (-log10 p_spat_shift)
      D. Spatial Shape: GP Trajectory Path Residual (-log10 p_spat_shape)
      E. Spatiotemporal Warp: Monotonic Pacing Distortion (-log10 p_warp)
      
    Parameters
    ----------
    test_results_df : pd.DataFrame
        DataFrame of blastomere scores for a single query embryo.
    calib_meta : dict or None, default=None
        Calibration metadata containing k_test and dt0_test.
    alpha : float, default=0.05
        FDR significance cutoff.
    top_n_labels : int, default=4
        Number of top significant blastomeres to annotate per panel.
    figsize : tuple of (float, float), default=(20, 22)
        Matplotlib figure dimensions.
    dpi : int, default=130
        Figure resolution.
        
    Returns
    -------
    fig : plt.Figure
        Matplotlib Figure object.
    axes : np.ndarray of plt.Axes
        Array of 5 panel Axes.
    """
    plot_df = test_results_df.copy()
    test_embryo_id = str(plot_df["embryo_id"].iloc[0]) if "embryo_id" in plot_df.columns else "Query"
    if calib_meta is None:
        calib_meta = {}

    # Column name fallback: handles 'cell' (inference) vs 'cell_name' (CV artifact)
    c_col = "cell" if "cell" in plot_df.columns else "cell_name"

    plot_df["lineage"] = plot_df[c_col].apply(assign_lineage)
    lineage_order = ["AB", "MS", "E", "C", "D", "Germline/P", "Other"]
    plot_df["lineage_cat"] = pd.Categorical(
        plot_df["lineage"], categories=lineage_order, ordered=True
    )
    plot_df = plot_df.sort_values(["lineage_cat", c_col]).reset_index(drop=True)

    n_cells = len(plot_df)
    plot_df["x_coord"] = np.arange(n_cells)
    eps = 1e-12

    fig, axes = plt.subplots(5, 1, figsize=figsize, sharex=True, dpi=dpi)

    neg_log_nom = -np.log10(alpha)
    neg_log_bonf = -np.log10(alpha / max(n_cells, 1))

    panels = [
        (
            axes[0],
            "pval_temp_shape",
            "qval_temp_shape",
            "hit_temp_shape",
            "pct_duration_deviation",
            "%",
            r"A. Temporal Shape: Autonomous Duration Anomaly ($-\log_{10} p_{Z_{\mathrm{temp\_shape}}}$)",
        ),
        (
            axes[1],
            "pval_temp_shift",
            "qval_temp_shift",
            "hit_temp_shift",
            "delta_midpoint_min",
            " min",
            r"B. Temporal Shift: Propagated Tree Phase Drift ($-\log_{10} p_{Z_{\mathrm{temp\_shift}}}$)",
        ),
        (
            axes[2],
            "pval_spat_shift",
            "qval_spat_shift",
            "hit_spat_shift",
            "com_shift_um",
            r" $\mu$m",
            r"C. Spatial Shift: Center-of-Mass Misplacement ($-\log_{10} p_{D_{\mathrm{spat\_shift}}}$)",
        ),
        (
            axes[3],
            "pval_spat_shape",
            "qval_spat_shape",
            "hit_spat_shape",
            "rmse_3d_um",
            r" $\mu$m RMSE",
            r"D. Spatial Shape: GP Trajectory Path Residual ($-\log_{10} p_{D_{\mathrm{spat\_shape}}}$)",
        ),
        (
            axes[4],
            "pval_warp",
            "qval_warp",
            "hit_warp",
            "rms_warp_min" if "rms_warp_min" in plot_df.columns else "signed_warp_area",
            r" min RMS" if "rms_warp_min" in plot_df.columns else r" $\Delta A_\gamma$",
            r"E. Spatiotemporal Warp: Monotonic Pacing Distortion ($-\log_{10} p_{\mathrm{warp}}$)",
        ),
    ]

    for ax, p_col, q_col, hit_col, eff_col, unit, title in panels:
        if p_col not in plot_df.columns:
            continue

        p_vals = np.clip(plot_df[p_col].astype(float).values, eps, 1.0)
        neg_log_p = -np.log10(p_vals)
        plot_df[f"neg_log_{p_col}"] = neg_log_p

        for lin_name, grp in plot_df.groupby("lineage_cat", observed=True):
            if grp.empty:
                continue
            col_val = LINEAGE_PALETTE.get(str(lin_name), "#7f8c8d")
            ax.scatter(
                grp["x_coord"],
                grp[f"neg_log_{p_col}"],
                color=col_val,
                s=52,
                alpha=0.9,
                edgecolors="black",
                linewidth=0.6,
                zorder=3,
            )

        sorted_indices = np.argsort(p_vals)
        sorted_p = p_vals[sorted_indices]
        ranks = np.arange(1, n_cells + 1)
        bh_critical_p = (ranks / n_cells) * alpha
        passed_bh = sorted_p <= bh_critical_p

        if np.any(passed_bh):
            max_passed_k = np.max(np.where(passed_bh)[0])
            bh_threshold_p = sorted_p[max_passed_k]
            neg_log_bh = -np.log10(bh_threshold_p)
            bh_label = rf"BH-FDR $q < {alpha}$ (Cutoff $-\log_{{10}} p = {neg_log_bh:.2f}$ | {np.sum(passed_bh)} Hits)"
        else:
            neg_log_bh = -np.log10(alpha / n_cells)
            bh_label = rf"BH-FDR $q < {alpha}$ (Theoretical Max Cutoff $-\log_{{10}} p = {neg_log_bh:.2f}$)"

        ax.axhline(neg_log_bh, color="#e67e22", linestyle="-.", linewidth=1.3, label=bh_label, zorder=2)
        ax.axhline(neg_log_nom, color="#d63031", linestyle="--", linewidth=1.1, label=rf"Nominal $\alpha = {alpha}$ ($-\log_{{10}} p = {neg_log_nom:.2f}$)", zorder=2)
        ax.axhline(neg_log_bonf, color="#7f8c8d", linestyle=":", linewidth=1.1, label=rf"Bonferroni ($-\log_{{10}} p = {neg_log_bonf:.2f}$)", zorder=2)

        sig_hits = plot_df[plot_df[hit_col]].copy() if hit_col in plot_df else pd.DataFrame()
        if sig_hits.empty:
            top_hits = plot_df.nlargest(top_n_labels, f"neg_log_{p_col}")
            label_candidates = top_hits[top_hits[f"neg_log_{p_col}"] >= neg_log_nom]
        else:
            label_candidates = sig_hits.nlargest(top_n_labels, f"neg_log_{p_col}")

        for _, hit in label_candidates.iterrows():
            val_eff = hit.get(eff_col, np.nan)
            val_str = f"{val_eff:+.1f}" if isinstance(val_eff, float) else f"{float(val_eff):+.1f}"
            tag = f"{hit[c_col]} ({val_str}{unit})"
            ax.annotate(
                tag,
                xy=(hit["x_coord"], hit[f"neg_log_{p_col}"]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5),
                zorder=5,
            )

        ax.set_title(title, fontsize=11.5, fontweight="bold", loc="left")
        ax.set_ylabel(r"$-\log_{10}(p)$", fontsize=10.0)
        y_max = max(plot_df[f"neg_log_{p_col}"].max() * 1.15, neg_log_bonf + 0.6)
        ax.set_ylim([0.0, max(y_max, 3.0)])
        ax.grid(True, linestyle=":", alpha=0.5, axis="y")
        ax.legend(frameon=True, fontsize=8.5, loc="upper right")

    lineage_midpoints = []
    lineage_labels = []

    for lin_name, grp in plot_df.groupby("lineage_cat", observed=True):
        if not grp.empty:
            mid = grp["x_coord"].mean()
            lineage_midpoints.append(mid)
            lineage_labels.append(f"{lin_name}\n(N={len(grp)})")
            boundary = grp["x_coord"].max() + 0.5
            if boundary < n_cells - 0.5:
                for ax in axes:
                    ax.axvline(boundary, color="#bdc3c7", linestyle="-", linewidth=1.0, zorder=1)

    axes[-1].set_xticks(lineage_midpoints)
    axes[-1].set_xticklabels(lineage_labels, fontsize=10, fontweight="bold")
    axes[-1].set_xlim([-1, n_cells])
    axes[-1].set_xlabel("Embryonic Lineage Clades", fontsize=11.5, fontweight="bold")

    k_val = calib_meta.get("k_test", plot_df["k_test"].iloc[0] if "k_test" in plot_df.columns else 1.0)
    dt0_val = calib_meta.get("dt0_test", plot_df["dt0_test"].iloc[0] if "dt0_test" in plot_df.columns else 0.0)

    plt.suptitle(
        f"Unified 5-Metric Empirical Anomaly Profile: Test Embryo '{test_embryo_id}'\n"
        rf"Affine Registration: Pace $K_e = {k_val:.4f}$, Phase $\Delta t_0 = {dt0_val:+.2f}$ min",
        fontsize=13.5,
        fontweight="bold",
        y=0.992,
    )
    plt.tight_layout(rect=[0, 0.01, 1, 0.975])
    return fig, axes
