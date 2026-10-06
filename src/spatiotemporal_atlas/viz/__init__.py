"""Visualization routines: 5-metric Manhattan plots, dual-lineage trees, and clade styling."""

from .colors import get_canonical_clade_color, assign_lineage, LINEAGE_PALETTE
from .manhattan import plot_5metric_manhattan, plot_6metric_manhattan
from .lineage_tree import build_interval_tree_layout, plot_warping_velocity_dual_lineage
from .diagnostics import visualize_embryo_diagnostics
from .trajectory_dashboard import (
    plot_cell_trajectory_diagnostic_dashboard,
    plot_interactive_3d_trajectory,
    plot_cell_cohort_3d_comparison,
    plot_wt_null_metric_distributions,
)

__all__ = [
    "get_canonical_clade_color",
    "assign_lineage",
    "LINEAGE_PALETTE",
    "plot_5metric_manhattan",
    "plot_6metric_manhattan",
    "build_interval_tree_layout",
    "plot_warping_velocity_dual_lineage",
    "visualize_embryo_diagnostics",
    "plot_cell_trajectory_diagnostic_dashboard",
    "plot_interactive_3d_trajectory",
    "plot_cell_cohort_3d_comparison",
    "plot_wt_null_metric_distributions",
]

