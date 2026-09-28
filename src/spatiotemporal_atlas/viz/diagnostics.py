"""Unified embryo diagnostics dashboard rendering Manhattan and Dual-Lineage plots."""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from ..types import ReferenceAtlas
from .manhattan import plot_5metric_manhattan
from .lineage_tree import plot_warping_velocity_dual_lineage


def visualize_embryo_diagnostics(
    target_embryo_id: str,
    cell_scores: pd.DataFrame | str | Path,
    atlas_bundle: dict | ReferenceAtlas | str | Path,
    lineage_data: pd.DataFrame | str | Path,
    query_pos_data: pd.DataFrame | str | Path | None = None,
    embryo_qc: pd.DataFrame | str | Path | None = None,
    alpha: float = 0.05,
    show: bool = True,
) -> tuple[plt.Figure, plt.Figure]:
    """Renders the 5-Metric Manhattan Plot and Dual Lineage Tree for an embryo
    evaluated via run_embryo_inference.
    
    Parameters
    ----------
    target_embryo_id : str
        Identifier of the embryo to plot (e.g., '20080923_lin-32_1_L1').
    cell_scores : pd.DataFrame or str or Path
        DataFrame or file path to cell_scores (parquet/csv) from run_embryo_inference.
    atlas_bundle : dict or ReferenceAtlas or str or Path
        Stage 1 atlas bundle or path to saved atlas joblib file.
    lineage_data : pd.DataFrame or str or Path
        DataFrame or file path to lineage_tree.csv/parquet.
    query_pos_data : pd.DataFrame or str or Path or None, default=None
        Optional raw query positions to compute instantaneous warping velocity gamma_dot(t).
    embryo_qc : pd.DataFrame or str or Path or None, default=None
        Optional DataFrame or file path to embryo_qc from run_embryo_inference.
    alpha : float, default=0.05
        FDR significance cutoff.
    show : bool, default=True
        Whether to call plt.show() after rendering.
        
    Returns
    -------
    fig_manhattan : plt.Figure
        Figure handle for 5-metric Manhattan plot.
    fig_lineage : plt.Figure
        Figure handle for Dual Lineage tree plot.
    """
    target_embryo_id = str(target_embryo_id)

    # 1. Resolve cell_scores
    if isinstance(cell_scores, (str, Path)):
        p = str(cell_scores)
        scores_df = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
    else:
        scores_df = cell_scores.copy()

    c_embryo = "embryo_id" if "embryo_id" in scores_df.columns else "series"
    scores_df[c_embryo] = scores_df[c_embryo].astype(str)
    target_df = scores_df[scores_df[c_embryo] == target_embryo_id].copy()

    if target_df.empty:
        raise ValueError(f"Embryo '{target_embryo_id}' not found in provided cell_scores.")

    print(f"Found Embryo '{target_embryo_id}' ({len(target_df)} blastomeres scored).")

    # 2. Resolve atlas_bundle
    if isinstance(atlas_bundle, (str, Path)):
        atlas = joblib.load(atlas_bundle)
    elif isinstance(atlas_bundle, ReferenceAtlas):
        atlas = atlas_bundle.to_dict()
    else:
        atlas = atlas_bundle

    # 3. Resolve lineage_df
    if isinstance(lineage_data, (str, Path)):
        p = str(lineage_data)
        lineage_df = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
    else:
        lineage_df = lineage_data.copy()

    # 4. Resolve query_pos_df
    pos_df = None
    if query_pos_data is not None:
        if isinstance(query_pos_data, (str, Path)):
            p = str(query_pos_data)
            pos_df = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
        else:
            pos_df = query_pos_data.copy()

    # 5. Extract embryo registration metadata
    calib_meta = {
        "temporal_atlas": atlas.get("temporal_atlas", {}),
        "fitted_models": atlas.get("cell_models", {}),
    }

    if embryo_qc is not None:
        if isinstance(embryo_qc, (str, Path)):
            p = str(embryo_qc)
            qc_df = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
        else:
            qc_df = embryo_qc.copy()
        qc_embryo = "embryo_id" if "embryo_id" in qc_df.columns else "series"
        qc_df[qc_embryo] = qc_df[qc_embryo].astype(str)
        emb_qc_row = qc_df[qc_df[qc_embryo] == target_embryo_id]
        if not emb_qc_row.empty:
            calib_meta["k_test"] = float(emb_qc_row["k_test"].iloc[0])
            calib_meta["dt0_test"] = float(emb_qc_row["dt0_test"].iloc[0])
    elif "k_test" in target_df.columns:
        calib_meta["k_test"] = float(target_df["k_test"].iloc[0])
        calib_meta["dt0_test"] = float(target_df["dt0_test"].iloc[0])
    else:
        calib_meta["k_test"] = 1.0
        calib_meta["dt0_test"] = 0.0

    # Panel 1: 5-Metric Manhattan Plot
    print("Rendering 5-Metric Manhattan plot...")
    fig_manhattan, _ = plot_5metric_manhattan(
        test_results_df=target_df,
        calib_meta=calib_meta,
        alpha=alpha,
        figsize=(20, 22),
    )
    if show:
        plt.show()

    # Panel 2: Dual Lineage Warp Velocity Tree
    print("Rendering Warping Velocity Dual-Lineage plot...")
    fig_lineage, _ = plot_warping_velocity_dual_lineage(
        test_results_df=target_df,
        calib_meta=calib_meta,
        lineage_df=lineage_df,
        query_pos_df=pos_df,
        atlas_bundle=atlas,
    )
    if show:
        plt.show()

    return fig_manhattan, fig_lineage
