"""Stage 1: WT Reference Atlas construction pipeline."""

import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

from ..geometry.gpa import compute_gpa_consensus_atlas
from ..geometry.align import align_embryo_to_spatial_template
from ..models.parallel import score_embryos_batch_parallel


def build_wt_reference_atlas(
    wt_pos_df: pd.DataFrame,
    lineage_df: pd.DataFrame,
    raw_spatial_cols: list[str] | tuple[str, ...] = ("x", "y", "z"),
    voxel_size_xyz: list[float] | tuple[float, ...] = (0.09, 0.09, 1.0),
    time_col: str = "time",
    embryo_col: str = "series",
    cell_col: str = "cell",
    max_inlier_dist_um: float = 4.0,
    max_inlier_dist_canon: float = 5.0,
    n_null_splits: int = 5,
    n_jobs: int = 4,
    model_type: str = "joint_gp",
    length_scale: float = 0.3,
    noise_level: float = 1.0,
    random_state: int = 42,
    local_trajectory_alignment: bool = True,
    warping_lambda: float = 10.0,
    warping_slope_bounds: tuple[float, float] = (0.5, 2.0),
    tau_percentile_cutoff: float = 95.0,
) -> dict:

    """Builds the complete WT reference atlas: consensus spatial template, 1D RANSAC
    temporal atlas, pooled out-of-fold empirical nulls, and lightweight GP models.
    
    Parameters
    ----------
    wt_pos_df : pd.DataFrame
        Tracked positions across wild-type (WT) training cohort.
    lineage_df : pd.DataFrame
        Lineage tree defining parent-daughter relationships.
    raw_spatial_cols : list or tuple of str, default=('x', 'y', 'z')
        Column names of raw voxel coordinates.
    voxel_size_xyz : list or tuple of float, default=(0.09, 0.09, 1.0)
        Physical voxel pitch in microns (dx, dy, dz).
    time_col : str, default='time'
        Time/frame column.
    embryo_col : str, default='series'
        Embryo identifier column.
    cell_col : str, default='cell'
        Blastomere name column.
    max_inlier_dist_um : float, default=4.0
        RANSAC spatial inlier threshold in microns.
    max_inlier_dist_canon : float, default=5.0
        RANSAC temporal inlier threshold in canonical minutes.
    n_null_splits : int, default=5
        Number of internal K-fold cross-validation splits for assembling empirical null.
    n_jobs : int, default=4
        Number of parallel CPU worker processes.
    random_state : int, default=42
        Random seed for reproducibility.
        
    Returns
    -------
    atlas_bundle : dict
        Complete reference atlas dictionary containing spatial_template,
        temporal_atlas, cell_models, oof_null_df, and associated configuration.
    """
    print("=" * 80)
    print("STAGE 1: BUILDING WT REFERENCE ATLAS")
    print("=" * 80)

    # 1. Scale raw voxel coordinates to microns
    micron_cols = ["x_um", "y_um", "z_um"]
    aligned_cols = ["x_aligned_um", "y_aligned_um", "z_aligned_um"]

    df = wt_pos_df.copy()
    df[embryo_col] = df[embryo_col].astype(str)
    df[cell_col] = df[cell_col].astype(str).str.strip()

    for raw_c, mic_c, scale in zip(raw_spatial_cols, micron_cols, voxel_size_xyz):
        df[mic_c] = df[raw_c].astype(float) * scale

    # 2. Build Generalized Procrustes consensus mean template
    centroids_df = (
        df.dropna(subset=micron_cols)
        .groupby([embryo_col, cell_col])[micron_cols]
        .mean()
        .reset_index()
    )

    print("[Spatial Registration] Computing scale-normalized GPA consensus template...")
    consensus_spatial_template, seed_id = compute_gpa_consensus_atlas(
        centroids_df=centroids_df,
        embryo_col=embryo_col,
        cell_col=cell_col,
        micron_cols=micron_cols,
        max_inlier_dist_um=max_inlier_dist_um,
        n_procrustes_iter=4,
        allow_scaling=True,
    )
    print(f"  GPA template converged across {len(consensus_spatial_template)} blastomeres (Seed: '{seed_id}').")

    # 3. Align all WT embryos to the consensus template
    print("[Spatial Registration] Aligning WT embryos to consensus template...")
    aligned_frames = []
    wt_spatial_reg_meta = {}
    for emb_id, grp_emb in df.groupby(embryo_col):
        aligned_emb, r_meta = align_embryo_to_spatial_template(
            emb_coords_df=grp_emb,
            spatial_template_df=consensus_spatial_template,
            micron_cols=micron_cols,
            aligned_cols=aligned_cols,
            cell_col=cell_col,
            max_inlier_dist_um=max_inlier_dist_um,
            allow_scaling=True,
        )
        aligned_frames.append(aligned_emb)
        wt_spatial_reg_meta[emb_id] = r_meta

    pos_df_aligned = pd.concat(aligned_frames, axis=0, ignore_index=True)

    # 4. Build empirical OOF null model via internal K-fold CV across WT embryos
    all_wt_embryos = np.array(pos_df_aligned[embryo_col].dropna().unique())
    actual_splits = min(n_null_splits, len(all_wt_embryos))
    print(f"[Null Model] Generating pooled empirical null across {actual_splits} internal folds...")

    kf = KFold(n_splits=actual_splits, shuffle=True, random_state=random_state)
    null_records = []

    for fold_idx, (tr_idx, val_idx) in enumerate(
        tqdm(kf.split(all_wt_embryos), total=actual_splits, desc="Internal Null Folds")
    ):
        inner_train = list(all_wt_embryos[tr_idx])
        inner_val = list(all_wt_embryos[val_idx])

        raw_val_df, _, _ = score_embryos_batch_parallel(
            test_embryo_ids=inner_val,
            pos_df=pos_df_aligned,
            lineage_df=lineage_df,
            training_embryos=inner_train,
            spatial_cols=aligned_cols,
            time_col=time_col,
            embryo_col=embryo_col,
            cell_col=cell_col,
            model_type=model_type,
            length_scale=length_scale,
            noise_level=noise_level,
            optimizer=None,
            return_models=False,
            n_jobs=n_jobs,
            desc=f"Null Fold {fold_idx + 1}/{actual_splits}",
            max_inlier_dist_canon=max_inlier_dist_canon,
            local_trajectory_alignment=local_trajectory_alignment,
            warping_lambda=warping_lambda,
            warping_slope_bounds=warping_slope_bounds,
            tau_percentile_cutoff=tau_percentile_cutoff,
        )

        if raw_val_df.empty:
            continue

        for _, row in raw_val_df.iterrows():
            null_records.append({
                "null_embryo_id": row[embryo_col],
                cell_col: row[cell_col],
                "emp_temp_shape": abs(float(row["z_temp_shape"])),
                "emp_temp_shift": abs(float(row["z_temp_shift"])),
                "emp_spat_shift": float(row["d_spat_shift"]),
                "emp_rot_angle": float(row.get("rot_angle_deg", 0.0)),
                "emp_spat_shape": float(row["d_spat_shape"]),
                "emp_warp": float(row.get("rms_warp_min", row.get("z_warp", 0.0))),
                "emp_delta_birth": float(row.get("delta_birth_min", 0.0)),
            })

    oof_null_df = pd.DataFrame(null_records)
    print(f"  Pooled OOF null model assembled: {len(oof_null_df)} blastomere observations.")
    gc.collect()

    # 5. Fit final reference trajectory models on 100% of WT embryos
    print("[Reference Models] Fitting final Gaussian Process trajectories on 100% of WT data...")
    _, final_cell_models, atlas_meta = score_embryos_batch_parallel(
        test_embryo_ids=[],
        pos_df=pos_df_aligned,
        lineage_df=lineage_df,
        training_embryos=list(all_wt_embryos),
        spatial_cols=aligned_cols,
        time_col=time_col,
        embryo_col=embryo_col,
        cell_col=cell_col,
        model_type=model_type,
        length_scale=length_scale,
        noise_level=noise_level,
        optimizer=None,
        return_models=True,
        n_jobs=n_jobs,
        desc="Final Reference Fit",
        max_inlier_dist_canon=max_inlier_dist_canon,
        local_trajectory_alignment=local_trajectory_alignment,
        warping_lambda=warping_lambda,
        warping_slope_bounds=warping_slope_bounds,
        tau_percentile_cutoff=tau_percentile_cutoff,
    )

    atlas_bundle = {
        "spatial_template": consensus_spatial_template,
        "seed_embryo_id": seed_id,
        "temporal_atlas": atlas_meta["temporal_atlas"],
        "cell_models": final_cell_models,
        "oof_null_df": oof_null_df,
        "wt_spatial_reg_meta": wt_spatial_reg_meta,
        "tau_cutoffs": atlas_meta.get("tau_cutoffs", {}),
        "voxel_size_xyz": list(voxel_size_xyz),
        "raw_spatial_cols": list(raw_spatial_cols),
        "aligned_cols": aligned_cols,
        "micron_cols": micron_cols,
        "time_col": time_col,
        "embryo_col": embryo_col,
        "cell_col": cell_col,
        "max_inlier_dist_um": max_inlier_dist_um,
        "max_inlier_dist_canon": max_inlier_dist_canon,
        "local_trajectory_alignment": local_trajectory_alignment,
        "warping_lambda": warping_lambda,
        "warping_slope_bounds": list(warping_slope_bounds),
        "tau_percentile_cutoff": tau_percentile_cutoff,
    }


    print("=" * 80)
    print(f"ATLAS CONSTRUCTION COMPLETE ({len(final_cell_models)} blastomere models registered).")
    print("=" * 80)
    return atlas_bundle
