"""End-to-end integration test verifying Stage 1 Atlas Building and Stage 2 Inference."""

import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

from spatiotemporal_atlas.atlas import build_wt_reference_atlas, save_atlas, load_atlas
from spatiotemporal_atlas.inference import run_embryo_inference


def test_end_to_end_pipeline():
    """Verify Stage 1 atlas building, persistence, and Stage 2 inference on uncensored toy cohort."""
    repo_root = Path(__file__).parent.parent
    toy_pos_path = repo_root / "data" / "nuclei_pos_toy.csv"
    lineage_path = repo_root / "data" / "lineage_tree.csv"

    if not toy_pos_path.exists() or not lineage_path.exists():
        print(f"Skipping end-to-end test: {toy_pos_path} or {lineage_path} not found.")
        return

    full_pos_df = pd.read_csv(toy_pos_path)
    lineage_df = pd.read_csv(lineage_path)

    # In nuclei_pos_toy.csv, every cell is uncensored and present in all embryos.
    # Select the first 8 unique cells for a fast, representative test.
    test_cells = list(full_pos_df["cell_name"].unique()[:8])
    sub_df = full_pos_df[full_pos_df["cell_name"].isin(test_cells)].copy()

    # Split into 3 training embryos and 1 query embryo
    all_embryos = list(sub_df["embryo_id"].unique())
    train_embryos = all_embryos[:3]
    query_embryo = all_embryos[3]

    train_df = sub_df[sub_df["embryo_id"].isin(train_embryos)].copy()
    query_df = sub_df[sub_df["embryo_id"] == query_embryo].copy()

    print(f"  Training cohort: {train_embryos} across {len(test_cells)} cells.")
    print(f"  Query embryo: '{query_embryo}'.")

    # STAGE 1: Build WT reference atlas
    atlas_bundle = build_wt_reference_atlas(
        wt_pos_df=train_df,
        lineage_df=lineage_df,
        raw_spatial_cols=["x_um", "y_um", "z_um"],
        voxel_size_xyz=[1.0, 1.0, 1.0],  # Already in microns
        time_col="time_idx",
        embryo_col="embryo_id",
        cell_col="cell_name",
        n_null_splits=2,
        n_jobs=1,
    )

    assert "spatial_template" in atlas_bundle
    assert "temporal_atlas" in atlas_bundle
    assert "cell_models" in atlas_bundle
    assert "oof_null_df" in atlas_bundle
    assert "emp_rot_angle" in atlas_bundle["oof_null_df"].columns
    assert "emp_warp" in atlas_bundle["oof_null_df"].columns
    assert "emp_delta_birth" in atlas_bundle["oof_null_df"].columns
    assert len(atlas_bundle["cell_models"]) > 0

    # Test persistence (save and load)
    with tempfile.TemporaryDirectory() as tmp_dir:
        atlas_path = Path(tmp_dir) / "test_atlas.joblib"
        save_atlas(atlas_bundle, atlas_path)
        loaded_atlas = load_atlas(atlas_path)

        assert loaded_atlas.seed_embryo_id == atlas_bundle["seed_embryo_id"]
        assert len(loaded_atlas.cell_models) == len(atlas_bundle["cell_models"])

        # STAGE 2: Run inference on query embryo
        cell_scores_df, embryo_qc_df = run_embryo_inference(
            query_pos_df=query_df,
            lineage_df=lineage_df,
            atlas_bundle=loaded_atlas,
            alpha=0.05,
            min_observations=3,
        )

        assert not cell_scores_df.empty, "Inference produced empty cell_scores_df."
        assert not embryo_qc_df.empty, "Inference produced empty embryo_qc_df."

        expected_metric_cols = [
            "z_temp_shape", "z_temp_shift", "d_spat_shift", "d_spat_shape", "z_warp",
            "rot_angle_deg", "z_rot_angle",
            "rms_warp_min", "delta_birth_min", "tau_cutoff",
            "pval_temp_shape", "qval_temp_shape", "hit_temp_shape",
            "pval_temp_shift", "qval_temp_shift", "hit_temp_shift",
            "pval_warp", "qval_warp", "hit_warp",
            "pval_spat_shift", "qval_spat_shift", "hit_spat_shift",
            "pval_spat_rot", "qval_spat_rot", "hit_spat_rot",
            "pval_spat_shape", "qval_spat_shape", "hit_spat_shape",
            "is_any_outlier", "n_outlier_modalities",
        ]
        for col in expected_metric_cols:
            assert col in cell_scores_df.columns, f"Missing expected column: {col}"

        assert (cell_scores_df["n_outlier_modalities"] >= 0).all()
        assert (cell_scores_df["n_outlier_modalities"] <= 6).all()


        qc_row = embryo_qc_df.iloc[0]
        assert "scale_s" in qc_row
        assert "inlier_ratio" in qc_row
        assert "k_test" in qc_row
        assert "dt0_test" in qc_row
        print(f"  [OK] Successfully scored {len(cell_scores_df)} blastomere tracks.")
        print(f"  [OK] Query embryo scale={qc_row['scale_s']:.3f}, pacing Ke={qc_row['k_test']:.3f}.")

        # STAGE 2 (Full Joint GP): Run inference with shape_metric_mode="full_joint_gp"
        cell_scores_joint_df, _ = run_embryo_inference(
            query_pos_df=query_df,
            lineage_df=lineage_df,
            atlas_bundle=loaded_atlas,
            alpha=0.05,
            min_observations=3,
            shape_metric_mode="full_joint_gp",
        )
        assert not cell_scores_joint_df.empty
        assert (cell_scores_joint_df["d_spat_shape"] >= 0.0).all()
        print(f"  [OK] Successfully scored {len(cell_scores_joint_df)} tracks with shape_metric_mode='full_joint_gp'.")
