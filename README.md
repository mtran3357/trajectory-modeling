# Spatiotemporal Atlas

A modular framework for constructing 4D spatiotemporal atlases of *C. elegans* embryonic development and running inference on perturbed/query embryos (e.g. RNAi).

## Architecture

- **Stage 1 (Atlas Building)**:
  - Scale-normalized Generalized Procrustes Analysis (GPA) consensus spatial template.
  - 1D RANSAC affine temporal pacing atlas ($K_e, \Delta t_0$).
  - Internal K-fold cross-validation across WT embryos to assemble pooled out-of-fold empirical nulls.
  - Analytical Cholesky GP regression stripped to lightweight 100-point ribbons.
- **Stage 2 (Inference Pipeline)**:
  - Spatial alignment to consensus template.
  - Temporal alignment to canonical temporal atlas.
  - 5-modality trajectory scoring:
    1. **Temporal Shape**: Autonomous cell cycle duration deviation (*Z*<sub>temp_shape</sub>).
    2. **Temporal Shift**: Lineage-propagated midpoint phase drift (*Z*<sub>temp_shift</sub>).
    3. **Spatial Shift**: Center-of-mass Mahalanobis misplacement (*D*<sub>spat_shift</sub>).
    4. **Spatial Shape**: GP trajectory path residual RMSE (*D*<sub>spat_shape</sub>).
    5. **Spatiotemporal Warp**: Fisher-Rao geodesic pacing distance (*Z*<sub>warp</sub>).
  - Empirical p-values against WT null distribution and per-embryo Benjamini-Hochberg FDR control.
- **Visualization Suite (`spatiotemporal_atlas.viz`)**:
  - Unified 5-panel Manhattan plots showing genome-wide anomaly significance per lineage clade.
  - Paired reference vs. observed dual-lineage trees colored by instantaneous warping velocity $\dot{\gamma}(t)$.
  - Embryo diagnostics dashboard (`visualize_embryo_diagnostics`).

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
from spatiotemporal_atlas.viz import visualize_embryo_diagnostics

fig_manhattan, fig_lineage = visualize_embryo_diagnostics(
    target_embryo_id="20080508_pha-4_3E3C5_1yy_L2",
    cell_scores=cell_scores_df,
    atlas_bundle=atlas,
    lineage_data=lineage_df,
    query_pos_data=rnai_pos_df,
    embryo_qc=embryo_qc_df,
    alpha=0.05,
    show=True,
)
```

---

## Running Tests

```bash
python tests/run_tests.py
```
