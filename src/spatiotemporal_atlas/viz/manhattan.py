"""6-Metric Manhattan plot across multi-modal empirical trajectory anomaly benchmarks."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from .colors import LINEAGE_PALETTE, assign_lineage
from ..stats.fdr import bh_qvalues


def plot_6metric_manhattan(
    test_results_df: pd.DataFrame,
    calib_meta: dict | None = None,
    alpha: float = 0.05,
    top_n_labels: int = 4,
    figsize: tuple[float, float] = (20.0, 25.0),
    dpi: int = 130,
) -> tuple[plt.Figure, np.ndarray]:
    """Generates a 6-panel Manhattan plot across the full empirical benchmark suite.
    
    Panels:
      Temporal Domain:
        A. Temporal Shape: Autonomous Duration Anomaly (-log10 p_temp_shape)
        B. Temporal Shift: Lineage Birth Time Drift (-log10 p_temp_shift)
        C. Spatiotemporal Warp: Monotonic Pacing Distortion (-log10 p_warp)
      Spatial Domain:
        D. Spatial Shift: Center-of-Mass Misplacement (-log10 p_spat_shift)
        E. Spatial Orientation: Trajectory Migration Deflection (-log10 p_spat_rot)
        F. Spatial Shape: GP Trajectory Path Residual (-log10 p_spat_shape)
      
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
    figsize : tuple of (float, float), default=(20, 25)
        Matplotlib figure dimensions.
    dpi : int, default=130
        Figure resolution.
        
    Returns
    -------
    fig : plt.Figure
        Matplotlib Figure object.
    axes : np.ndarray of plt.Axes
        Array of 6 panel Axes.
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

    fig, axes = plt.subplots(6, 1, figsize=figsize, sharex=True, dpi=dpi)

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
            r"A. Temporal Shape: Autonomous Duration Anomaly ($-\log_{10} p_{\mathrm{temp\_shape}}$)",
        ),
        (
            axes[1],
            "pval_temp_shift",
            "qval_temp_shift",
            "hit_temp_shift",
            "delta_birth_min" if "delta_birth_min" in plot_df.columns else "delta_midpoint_min",
            " min",
            r"B. Temporal Shift: Lineage Birth Time Drift ($-\log_{10} p_{\mathrm{temp\_shift}}$)",
        ),
        (
            axes[2],
            "pval_warp",
            "qval_warp",
            "hit_warp",
            "rms_warp_min" if "rms_warp_min" in plot_df.columns else "signed_warp_area",
            r" min RMS" if "rms_warp_min" in plot_df.columns else r" $\Delta A_\gamma$",
            r"C. Spatiotemporal Warp: Monotonic Pacing Distortion ($-\log_{10} p_{\mathrm{warp}}$)",
        ),
        (
            axes[3],
            "pval_spat_shift",
            "qval_spat_shift",
            "hit_spat_shift",
            "com_shift_um",
            r" $\mu$m",
            r"D. Spatial Shift: Center-of-Mass Misplacement ($-\log_{10} p_{\mathrm{spat\_shift}}$)",
        ),
        (
            axes[4],
            "pval_spat_rot",
            "qval_spat_rot",
            "hit_spat_rot",
            "rot_angle_deg",
            r"$^\circ$",
            r"E. Spatial Orientation: Trajectory Migration Deflection ($-\log_{10} p_{\mathrm{spat\_rot}}$)",
        ),
        (
            axes[5],
            "pval_spat_shape",
            "qval_spat_shape",
            "hit_spat_shape",
            "rmse_3d_um",
            r" $\mu$m RMSE",
            r"F. Spatial Shape: GP Trajectory Path Residual ($-\log_{10} p_{\mathrm{spat\_shape}}$)",
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
        f"Unified 6-Metric Empirical Anomaly Profile: Test Embryo '{test_embryo_id}'\n"
        rf"Affine Registration: Pace $K_e = {k_val:.4f}$, Phase $\Delta t_0 = {dt0_val:+.2f}$ min",
        fontsize=13.5,
        fontweight="bold",
        y=0.992,
    )
    plt.tight_layout(rect=[0, 0.01, 1, 0.975])
    return fig, axes


# Backward-compatible alias
plot_5metric_manhattan = plot_6metric_manhattan


COHORT_MODALITY_CONFIG = [
    (
        "temp_shape",
        "z_temp_shape",
        "emp_temp_shape",
        True,
        "pct_duration_deviation",
        "%",
        r"A. Temporal Shape: Autonomous Duration Anomaly ($-\log_{10} p_{\mathrm{temp\_shape}}$)",
    ),
    (
        "temp_shift",
        "z_temp_shift",
        "emp_temp_shift",
        True,
        "delta_midpoint_min",
        " min",
        r"B. Temporal Shift: Lineage Midpoint Time Drift ($-\log_{10} p_{\mathrm{temp\_shift}}$)",
    ),
    (
        "warp",
        "rms_warp_min",
        "emp_warp",
        False,
        "rms_warp_min",
        " min RMS",
        r"C. Spatiotemporal Warp: Monotonic Pacing Distortion ($-\log_{10} p_{\mathrm{warp}}$)",
    ),
    (
        "spat_shift",
        "d_spat_shift",
        "emp_spat_shift",
        False,
        "com_shift_um",
        r" $\mu$m",
        r"D. Spatial Shift: Center-of-Mass Misplacement ($-\log_{10} p_{\mathrm{spat\_shift}}$)",
    ),
    (
        "spat_rot",
        "rot_angle_deg",
        "emp_rot_angle",
        False,
        "rot_angle_deg",
        r"$^\circ$",
        r"E. Spatial Orientation: Trajectory Migration Deflection ($-\log_{10} p_{\mathrm{spat\_rot}}$)",
    ),
    (
        "spat_shape",
        "d_spat_shape",
        "emp_spat_shape",
        False,
        "rmse_3d_um",
        r" $\mu$m RMSE",
        r"F. Spatial Shape: GP Trajectory Path Residual ($-\log_{10} p_{\mathrm{spat\_shape}}$)",
    ),
]


def _calc_cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    """Calculates Cohen's d effect size between two independent samples."""
    x_clean = np.asarray(x, dtype=float)[np.isfinite(x)]
    y_clean = np.asarray(y, dtype=float)[np.isfinite(y)]
    if len(x_clean) < 2 or len(y_clean) < 2:
        return 0.0
    nx, ny = len(x_clean), len(y_clean)
    vx, vy = np.var(x_clean, ddof=1), np.var(y_clean, ddof=1)
    pooled_sd = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / max(nx + ny - 2, 1))
    if pooled_sd < 1e-12:
        return 0.0
    return float((np.mean(x_clean) - np.mean(y_clean)) / pooled_sd)


def compute_cohort_differential_stats(
    query_scores_df: pd.DataFrame,
    oof_null_df: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Performs blastomere-level two-sample Wilcoxon (Mann-Whitney U) tests across cohort.

    Compares query mutant blastomere distributions against the wild-type reference
    out-of-fold null distribution across all 6 spatiotemporal modalities, calculating
    both uncorrected nominal p-values and Benjamini-Hochberg FDR q-values.

    Parameters
    ----------
    query_scores_df : pd.DataFrame
        DataFrame of blastomere scores across query embryos (must contain 'cell').
    oof_null_df : pd.DataFrame
        DataFrame of WT out-of-fold reference null records.
    alpha : float, default=0.05
        FDR and nominal significance threshold.

    Returns
    -------
    diff_stats_df : pd.DataFrame
        DataFrame containing p-values, q-values, significance flags, and effect sizes
        for each of the 142 blastomeres across all 6 modalities.
    """
    c_col = "cell" if "cell" in query_scores_df.columns else "cell_name"
    cells = sorted(query_scores_df[c_col].unique())
    rows = []

    null_cell_col = "cell" if "cell" in oof_null_df.columns else ("cell_name" if "cell_name" in oof_null_df.columns else None)

    for cell in cells:
        q_cell = query_scores_df[query_scores_df[c_col] == cell]
        w_cell = oof_null_df[oof_null_df[null_cell_col] == cell] if null_cell_col is not None else oof_null_df

        rec = {
            "cell": cell,
            "lineage": assign_lineage(cell),
            "n_query": len(q_cell),
            "n_null": len(w_cell),
        }

        for m_key, q_col_name, n_col_name, abs_val, eff_col_name, unit_str, _ in COHORT_MODALITY_CONFIG:
            # Fallback for metric column names
            q_col = q_col_name if q_col_name in q_cell.columns else ("z_warp" if m_key == "warp" else q_col_name)
            n_col = n_col_name if n_col_name in w_cell.columns else n_col_name

            qv = q_cell[q_col].dropna().to_numpy(dtype=float) if q_col in q_cell.columns else np.array([])
            wv = w_cell[n_col].dropna().to_numpy(dtype=float) if n_col in w_cell.columns else np.array([])

            if abs_val:
                qv = np.abs(qv)
                wv = np.abs(wv)

            if len(qv) >= 2 and len(wv) >= 2:
                try:
                    stat, pval = mannwhitneyu(qv, wv, alternative="greater")
                except Exception:
                    pval = 1.0
            else:
                pval = 1.0

            eff_col = eff_col_name if eff_col_name in q_cell.columns else ("delta_birth_min" if m_key == "temp_shift" else eff_col_name)
            med_q = float(q_cell[eff_col].median()) if eff_col in q_cell.columns and len(q_cell[eff_col].dropna()) > 0 else np.nan
            med_w = float(w_cell[eff_col].median()) if eff_col in w_cell.columns and len(w_cell[eff_col].dropna()) > 0 else np.nan

            rec[f"pval_{m_key}"] = float(pval)
            rec[f"median_query_{m_key}"] = med_q
            rec[f"median_null_{m_key}"] = med_w
            rec[f"delta_median_{m_key}"] = med_q - med_w if np.isfinite(med_q) and np.isfinite(med_w) else np.nan
            rec[f"cohen_d_{m_key}"] = _calc_cohen_d(qv, wv)

        rows.append(rec)

    diff_stats_df = pd.DataFrame(rows)

    # Compute BH FDR q-values and significance flags across all blastomeres
    for m_key, _, _, _, _, _, _ in COHORT_MODALITY_CONFIG:
        pvals = diff_stats_df[f"pval_{m_key}"].to_numpy(dtype=float)
        qvals = bh_qvalues(pvals)
        diff_stats_df[f"qval_{m_key}"] = qvals
        diff_stats_df[f"hit_nom_{m_key}"] = pvals < alpha
        diff_stats_df[f"hit_bh_{m_key}"] = qvals < alpha

    return diff_stats_df


def plot_cohort_differential_manhattan(
    diff_stats_df: pd.DataFrame,
    alpha: float = 0.05,
    top_n_labels: int = 4,
    cohort_name: str = "pop-1(RNAi)",
    figsize: tuple[float, float] = (20.0, 25.0),
    dpi: int = 130,
) -> tuple[plt.Figure, np.ndarray]:
    """Generates a 6-panel Manhattan plot of cohort-aggregated two-sample Wilcoxon differential tests.

    Displays -log10(p) across all scored blastomeres ordered by lineage clade, annotating
    nominal significance (p < alpha), Benjamini-Hochberg significance (q < alpha), and Bonferroni
    thresholds, with top aberrant blastomeres highlighted by physical effect sizes.

    Parameters
    ----------
    diff_stats_df : pd.DataFrame
        Output of compute_cohort_differential_stats.
    alpha : float, default=0.05
        Significance cutoff.
    top_n_labels : int, default=4
        Number of top significant blastomeres to annotate per panel.
    cohort_name : str, default='pop-1(RNAi)'
        Name of query perturbation cohort for suptitle.
    figsize : tuple of (float, float), default=(20, 25)
        Figure size.
    dpi : int, default=130
        Figure DPI.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object.
    axes : np.ndarray of plt.Axes
        Array of 6 subplot Axes.
    """
    plot_df = diff_stats_df.copy()
    c_col = "cell" if "cell" in plot_df.columns else "cell_name"

    lineage_order = ["AB", "MS", "E", "C", "D", "Germline/P", "Other"]
    if "lineage" not in plot_df.columns:
        plot_df["lineage"] = plot_df[c_col].apply(assign_lineage)

    plot_df["lineage_cat"] = pd.Categorical(
        plot_df["lineage"], categories=lineage_order, ordered=True
    )
    plot_df = plot_df.sort_values(["lineage_cat", c_col]).reset_index(drop=True)

    n_cells = len(plot_df)
    plot_df["x_coord"] = np.arange(n_cells)
    eps = 1e-15

    fig, axes = plt.subplots(6, 1, figsize=figsize, sharex=True, dpi=dpi)

    neg_log_nom = -np.log10(alpha)
    neg_log_bonf = -np.log10(alpha / max(n_cells, 1))

    for idx, (m_key, _, _, _, _, unit_str, panel_title) in enumerate(COHORT_MODALITY_CONFIG):
        ax = axes[idx]
        p_col = f"pval_{m_key}"
        q_col = f"qval_{m_key}"
        hit_bh_col = f"hit_bh_{m_key}"
        hit_nom_col = f"hit_nom_{m_key}"

        if p_col not in plot_df.columns:
            continue

        p_vals = np.clip(plot_df[p_col].astype(float).values, eps, 1.0)
        neg_log_p = -np.log10(p_vals)
        plot_df[f"neg_log_{p_col}"] = neg_log_p

        # Scatter plot colored by lineage clade
        for lin_name, grp in plot_df.groupby("lineage_cat", observed=True):
            if grp.empty:
                continue
            col_val = LINEAGE_PALETTE.get(str(lin_name), "#7f8c8d")
            ax.scatter(
                grp["x_coord"],
                grp[f"neg_log_{p_col}"],
                color=col_val,
                s=54,
                alpha=0.92,
                edgecolors="black",
                linewidth=0.6,
                zorder=3,
            )

        # Compute BH critical threshold cutoff
        sorted_indices = np.argsort(p_vals)
        sorted_p = p_vals[sorted_indices]
        ranks = np.arange(1, n_cells + 1)
        bh_critical_p = (ranks / n_cells) * alpha
        passed_bh = sorted_p <= bh_critical_p
        n_bh = int(np.sum(passed_bh))
        n_nom = int(np.sum(p_vals < alpha))

        if n_bh > 0:
            max_passed_k = np.max(np.where(passed_bh)[0])
            bh_threshold_p = sorted_p[max_passed_k]
            neg_log_bh = -np.log10(bh_threshold_p)
            bh_label = rf"BH-FDR $q < {alpha}$ (Cutoff $-\log_{{10}} p = {neg_log_bh:.2f}$ | {n_bh} Hits)"
        else:
            neg_log_bh = -np.log10(alpha / n_cells)
            bh_label = rf"BH-FDR $q < {alpha}$ (No Hits | Min Theoretical $-\log_{{10}} p = {neg_log_bh:.2f}$)"

        nom_label = rf"Nominal $p < {alpha}$ ($-\log_{{10}} p = {neg_log_nom:.2f}$ | {n_nom} Hits)"
        bonf_label = rf"Bonferroni ($-\log_{{10}} p = {neg_log_bonf:.2f}$ | {int(np.sum(p_vals < alpha / n_cells))} Hits)"

        # Horizontal threshold lines
        ax.axhline(neg_log_bh, color="#e67e22", linestyle="-.", linewidth=1.4, label=bh_label, zorder=2)
        ax.axhline(neg_log_nom, color="#d63031", linestyle="--", linewidth=1.2, label=nom_label, zorder=2)
        ax.axhline(neg_log_bonf, color="#7f8c8d", linestyle=":", linewidth=1.1, label=bonf_label, zorder=2)

        # Annotate top aberrant blastomeres
        sig_hits = plot_df[plot_df[hit_bh_col]].copy() if hit_bh_col in plot_df else pd.DataFrame()
        if sig_hits.empty:
            top_candidates = plot_df.nlargest(top_n_labels, f"neg_log_{p_col}")
            label_candidates = top_candidates[top_candidates[f"neg_log_{p_col}"] >= neg_log_nom]
        else:
            label_candidates = sig_hits.nlargest(top_n_labels, f"neg_log_{p_col}")

        # Stagger annotations to prevent overlapping text boxes
        label_candidates = label_candidates.sort_values("x_coord").reset_index(drop=True)
        prev_x = -999.0
        y_offset_idx = 0
        y_offsets = [9, 23, 37]

        for _, hit in label_candidates.iterrows():
            curr_x = hit["x_coord"]
            if abs(curr_x - prev_x) < 7.0:
                y_offset_idx = (y_offset_idx + 1) % len(y_offsets)
            else:
                y_offset_idx = 0
            prev_x = curr_x
            y_off = y_offsets[y_offset_idx]

            eff_val = hit.get(f"median_query_{m_key}", np.nan)
            if np.isfinite(eff_val):
                val_str = f"{eff_val:+.1f}" if "%" in unit_str or "min" in unit_str else f"{eff_val:.2f}"
                tag = f"{hit[c_col]} ({val_str}{unit_str})"
            else:
                tag = f"{hit[c_col]}"

            ax.annotate(
                tag,
                xy=(curr_x, hit[f"neg_log_{p_col}"]),
                xytext=(0, y_off),
                textcoords="offset points",
                ha="center",
                fontsize=7.8,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#2c3e50", lw=0.6, alpha=0.95),
                zorder=5,
            )

        title_str = f"{panel_title}  —  [{n_bh} BH Hits ({n_bh/n_cells:.1%}) | {n_nom} Nominal ({n_nom/n_cells:.1%}) / {n_cells} Cells]"
        ax.set_title(title_str, fontsize=11.5, fontweight="bold", loc="left", pad=6)
        ax.set_ylabel(r"$-\log_{10}(p)$", fontsize=10.0, fontweight="bold")
        y_max = max(plot_df[f"neg_log_{p_col}"].max() * 1.25, neg_log_bonf + 1.2, 4.0)
        ax.set_ylim([0.0, max(y_max, 4.5)])
        ax.grid(True, linestyle=":", alpha=0.5, axis="y")
        ax.legend(frameon=True, fontsize=8.5, loc="upper right", framealpha=0.92)

    # Vertical clade boundary lines and clade labels on bottom axis
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
    axes[-1].set_xticklabels(lineage_labels, fontsize=10.5, fontweight="bold")
    axes[-1].set_xlim([-1, n_cells])
    axes[-1].set_xlabel("Embryonic Lineage Clades (Sulston Order)", fontsize=11.5, fontweight="bold")

    n_q_max = int(plot_df["n_query"].max()) if "n_query" in plot_df.columns else 18
    n_null_max = int(plot_df["n_null"].max()) if "n_null" in plot_df.columns else 266

    plt.suptitle(
        f"Cohort-Aggregated Blastomere Differential Trajectory Profile: {cohort_name} vs. WT Reference Atlas\n"
        f"Two-Sample Wilcoxon (Mann-Whitney U) Tests Across All {n_cells} Blastomeres (N={n_q_max} {cohort_name} vs. N={n_null_max} WT Null)",
        fontsize=13.5,
        fontweight="bold",
        y=0.992,
    )
    plt.tight_layout(rect=[0, 0.01, 1, 0.975])
    return fig, axes
