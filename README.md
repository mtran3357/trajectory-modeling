# Spatiotemporal Atlas

A modular framework for constructing 4D spatiotemporal atlases of *C. elegans* embryonic development and running inference on perturbed/query embryos (e.g. RNAi).

## Architecture

- **Stage 1 (Atlas Building)**:
  - Scale-normalized Generalized Procrustes Analysis (GPA) consensus spatial template.
  - 1D RANSAC affine temporal pacing atlas ($K_e, \Delta t_0$).
  - Internal K-fold cross-validation across WT embryos to assemble pooled out-of-fold empirical nulls.
  - Kronecker Separable Joint 3D Gaussian Process regression evaluated on canonical active domains $[0, \tau_{\text{cutoff}}]$.
- **Stage 2 (Inference Pipeline)**:
  - Spatial alignment to consensus template (Umeyama similarity registration).
  - Temporal alignment to canonical temporal atlas (affine pacing $K_e$, timing offset $\Delta t_0$).
  - **Symmetric 3 × 3 Spatiotemporal Anomaly Suite** (6 orthogonal modalities):
    - *Temporal Domain*:
      1. **Temporal Shape**: Autonomous cell cycle duration log-deviation ($Z_{\text{temp\_shape}}$).
      2. **Temporal Shift**: Lineage-propagated birth time drift ($Z_{\text{temp\_shift}}$).
      3. **Spatiotemporal Warp**: Monotonic pacing distortion ($\text{RMS}_{\text{warp}}$, $Z_{\text{warp}}$).
    - *Spatial Domain*:
      4. **Spatial Shift**: Center-of-mass Mahalanobis translation ($\mathcal{D}_{\text{spat\_shift}}$ via Joint GP $B$).
      5. **Spatial Orientation**: Rigid $SO(3)$ Kabsch trajectory rotation angle ($\theta_{\text{rot}}$, $Z_{\text{rot\_angle}}$).
      6. **Spatial Shape**: Intrinsic Kronecker GP path curvature residual ($\mathcal{M}_{\text{spat\_shape}}$).
  - Cell-standardized empirical null calibration ($Z_c = \frac{s_c - \mu_c}{\sigma_c}$) and per-embryo Benjamini-Hochberg FDR control ($q < 0.05$).
- **Visualization Suite (`spatiotemporal_atlas.viz`)**:
  - Unified 6-panel Manhattan plots showing genome-wide anomaly significance across modalities.
  - Canonical Sulston-ordered dual-lineage trees colored by instantaneous warping velocity $\dot{\gamma}(t)$ with whole-embryo QC headers.
  - 7-panel single-cell trajectory diagnostic dashboard (`plot_cell_trajectory_diagnostic_dashboard`) featuring:
    - 3D spatial transformation panel (translation vector $\mathbf{T}$ and $SO(3)$ rotation triads $X, Y, Z \to X', Y', Z'$).
    - True isometric 1:1:1 physical bounding boxes preserving 3D Euclidean angles and triad orthogonality.
    - Pacing velocity $\dot{\gamma}(t)$ dot coloring on coordinate trajectories and monotonic warping curves.
    - Canonical developmental lifespan intervals.
  - Multi-embryo cohort 3D comparison visualizer (`plot_cell_cohort_3d_comparison`) with synchronized camera WebGL HTML export.
  - Empirical null distribution visualizer (`plot_wt_null_metric_distributions`) comparing physical units (canonical minutes) vs. standardized scaled metrics.

---

## Installation

Within your conda environment (`cell_typing_3d`):

```bash
pip install -e .
```

---

## Quick Start

### 1. Build Reference Atlas (Stage 1)

```python
import pandas as pd
from spatiotemporal_atlas import build_wt_reference_atlas, save_atlas

wt_pos_df = pd.read_csv("data/EPIC_merged_data_subset_oriented.csv")
lineage_df = pd.read_csv("data/lineage_tree.csv")

atlas = build_wt_reference_atlas(
    wt_pos_df=wt_pos_df,
    lineage_df=lineage_df,
    raw_spatial_cols=["x", "y", "z"],
    voxel_size_xyz=[0.0897, 0.0897, 0.8958],
    time_col="time",
    embryo_col="series",
    cell_col="cell",
    n_null_splits=5,
    n_jobs=4,
)

save_atlas(atlas, "models/wt_atlas_v1.joblib")
```

### 2. Run Inference on Query/RNAi Embryos (Stage 2)

```python
import pandas as pd
from spatiotemporal_atlas import load_atlas, run_embryo_inference

atlas = load_atlas("models/wt_atlas_v1.joblib")
rnai_pos_df = pd.read_csv("data/EPIC_RNAi_merged_data_subset_oriented.csv")
lineage_df = pd.read_csv("data/lineage_tree.csv")

cell_scores_df, embryo_qc_df = run_embryo_inference(
    query_pos_df=rnai_pos_df,
    lineage_df=lineage_df,
    atlas_bundle=atlas,
    alpha=0.05,
)
```

### 3. Visual Diagnostics Dashboard

```python
from spatiotemporal_atlas.viz import (
    visualize_embryo_diagnostics,
    plot_cell_trajectory_diagnostic_dashboard,
)

# Embryo-level 6-metric Manhattan plot & dual lineage tree
fig_manhattan, fig_lineage = visualize_embryo_diagnostics(
    target_embryo_id="20080602_hnd-1_6_pop1i_L1",
    cell_scores=cell_scores_df,
    atlas_bundle=atlas,
    lineage_data=lineage_df,
    query_pos_data=rnai_pos_df,
    embryo_qc=embryo_qc_df,
    alpha=0.05,
    show=True,
)

# 7-panel single-cell trajectory diagnostic dashboard
fig_dash, axes_dash = plot_cell_trajectory_diagnostic_dashboard(
    cell_name="MSapa",
    embryo_id="20080602_hnd-1_6_pop1i_L1",
    pos_df=rnai_pos_df,
    target_row=cell_scores_df[(cell_scores_df["cell"] == "MSapa") & (cell_scores_df["embryo_id"] == "20080602_hnd-1_6_pop1i_L1")].iloc[0],
    cell_model=atlas["cell_models"]["MSapa"],
    temporal_atlas=atlas["temporal_atlas"],
    embryo_qc=embryo_qc_df,
    save_interactive_html="MSapa_3d_interactive.html",
)
```

---

## Running Tests

```bash
python tests/run_tests.py
```
