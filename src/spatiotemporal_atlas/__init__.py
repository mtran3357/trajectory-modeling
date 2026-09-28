"""Spatiotemporal Atlas Building and Inference Framework for C. elegans Development."""

import os

# Prevent thread over-subscription on macOS/ARM when joblib spawns multiple processes
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from .config import (
    ColumnConfig,
    SpatialConfig,
    TemporalConfig,
    TrajectoryConfig,
    AtlasBuildConfig,
    InferenceConfig,
)
from .types import (
    SpatialPose,
    TemporalPose,
    CellTemporalStats,
    TrajectoryRibbon,
    ReferenceAtlas,
)
from .geometry import (
    umeyama_similarity_transform,
    compute_ransac_similarity_pose,
    compute_gpa_consensus_atlas,
    align_embryo_to_spatial_template,
)
from .functional import (
    curve_to_srvf,
    align_srvf_dp_clamped,
    compute_karcher_mean_srvf,
    calculate_fisher_rao_distance,
)
from .lineage import (
    parse_lineage_graph,
    get_ancestral_path_in_interval,
)
from .temporal import (
    extract_cell_lifespans,
    compute_ransac_temporal_pose,
    build_canonical_temporal_atlas,
    register_embryo_to_temporal_atlas,
)
from .models import (
    fit_coordinate_gps,
    extract_trajectory_ribbon,
    score_embryos_batch_parallel,
)
from .stats import (
    bh_qvalues,
    calc_emp_pval,
    apply_empirical_calibration_to_inference,
)
from .atlas import (
    build_wt_reference_atlas,
    save_atlas,
    load_atlas,
)
from .inference import (
    run_embryo_inference,
)
from .viz import (
    get_canonical_clade_color,
    plot_5metric_manhattan,
    build_interval_tree_layout,
    plot_warping_velocity_dual_lineage,
    visualize_embryo_diagnostics,
)

__all__ = [
    # Configs
    "ColumnConfig",
    "SpatialConfig",
    "TemporalConfig",
    "TrajectoryConfig",
    "AtlasBuildConfig",
    "InferenceConfig",
    # Types
    "SpatialPose",
    "TemporalPose",
    "CellTemporalStats",
    "TrajectoryRibbon",
    "ReferenceAtlas",
    # Geometry
    "umeyama_similarity_transform",
    "compute_ransac_similarity_pose",
    "compute_gpa_consensus_atlas",
    "align_embryo_to_spatial_template",
    # Functional
    "curve_to_srvf",
    "align_srvf_dp_clamped",
    "compute_karcher_mean_srvf",
    "calculate_fisher_rao_distance",
    # Lineage
    "parse_lineage_graph",
    "get_ancestral_path_in_interval",
    # Temporal
    "extract_cell_lifespans",
    "compute_ransac_temporal_pose",
    "build_canonical_temporal_atlas",
    "register_embryo_to_temporal_atlas",
    # Models
    "fit_coordinate_gps",
    "extract_trajectory_ribbon",
    "score_embryos_batch_parallel",
    # Stats
    "bh_qvalues",
    "calc_emp_pval",
    "apply_empirical_calibration_to_inference",
    # Stage 1: Atlas
    "build_wt_reference_atlas",
    "save_atlas",
    "load_atlas",
    # Stage 2: Inference
    "run_embryo_inference",
    # Visualization
    "get_canonical_clade_color",
    "plot_5metric_manhattan",
    "build_interval_tree_layout",
    "plot_warping_velocity_dual_lineage",
    "visualize_embryo_diagnostics",
]
