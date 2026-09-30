"""3D Rigid and similarity alignment, pose estimation, and GPA consensus templates."""

from .umeyama import umeyama_similarity_transform
from .ransac_pose import compute_ransac_similarity_pose
from .gpa import compute_gpa_consensus_atlas
from .align import align_embryo_to_spatial_template
from .curve_align import (
    kabsch_curve_so3,
    interpolate_curve_to_grid,
    generalized_procrustes_curves,
    register_curve_to_template,
)

__all__ = [
    "umeyama_similarity_transform",
    "compute_ransac_similarity_pose",
    "compute_gpa_consensus_atlas",
    "align_embryo_to_spatial_template",
    "kabsch_curve_so3",
    "interpolate_curve_to_grid",
    "generalized_procrustes_curves",
    "register_curve_to_template",
]

