"""Unit tests for visualization routines."""

import matplotlib
matplotlib.use("Agg")  # Headless backend for testing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spatiotemporal_atlas.viz import (
    get_canonical_clade_color,
    assign_lineage,
    build_interval_tree_layout,
    plot_5metric_manhattan,
    plot_warping_velocity_dual_lineage,
    visualize_embryo_diagnostics,
)


def test_clade_colors_and_lineages():
    """Verify clade color assignments and lineage clade categorization."""
    assert get_canonical_clade_color("ABa") == "#1f77b4"
    assert get_canonical_clade_color("ABp") == "#2ca02c"
    assert get_canonical_clade_color("MS") == "#ff7f0e"
    assert get_canonical_clade_color("E") == "#2ecc71"
    assert get_canonical_clade_color("C") == "#9467bd"

    assert assign_lineage("ABala") == "AB"
    assert assign_lineage("MSa") == "MS"
    assert assign_lineage("Ea") == "E"
    assert assign_lineage("P2") == "Germline/P"


def test_build_interval_tree_layout():
    """Verify horizontal tree coordinate assignment."""
    lineage_df = pd.DataFrame([
        {"parent": "AB", "daughter1": "ABa", "daughter2": "ABp"},
        {"parent": "P0", "daughter1": "AB", "daughter2": "P1"},
    ])
    valid_cells = {"P0", "AB", "P1", "ABa", "ABp"}
    x_coords, children_map, local_roots = build_interval_tree_layout(valid_cells, lineage_df)

    assert "P0" in local_roots
    assert "AB" in children_map
    assert set(children_map["AB"]) == {"ABa", "ABp"}
    assert all(c in x_coords for c in valid_cells)


def test_plot_5metric_manhattan():
    """Verify Manhattan plot creates 5 axes and annotations without errors."""
    mock_df = pd.DataFrame([
        {
            "embryo_id": "test_emb",
            "cell": "ABa",
            "pval_temp_shape": 0.001,
            "qval_temp_shape": 0.005,
            "hit_temp_shape": True,
            "pct_duration_deviation": 12.5,
            "pval_temp_shift": 0.02,
            "qval_temp_shift": 0.04,
            "hit_temp_shift": True,
            "delta_midpoint_min": 3.2,
            "pval_spat_shift": 0.1,
            "qval_spat_shift": 0.2,
            "hit_spat_shift": False,
            "com_shift_um": 1.2,
            "pval_spat_shape": 0.0005,
            "qval_spat_shape": 0.002,
            "hit_spat_shape": True,
            "rmse_3d_um": 2.1,
            "pval_warp": 0.5,
            "qval_warp": 0.6,
            "hit_warp": False,
            "signed_warp_area": 0.02,
            "k_test": 1.05,
            "dt0_test": -2.0,
        },
        {
            "embryo_id": "test_emb",
            "cell": "EMS",
            "pval_temp_shape": 0.5,
            "qval_temp_shape": 0.6,
            "hit_temp_shape": False,
            "pct_duration_deviation": -1.0,
            "pval_temp_shift": 0.4,
            "qval_temp_shift": 0.5,
            "hit_temp_shift": False,
            "delta_midpoint_min": -0.5,
            "pval_spat_shift": 0.001,
            "qval_spat_shift": 0.005,
            "hit_spat_shift": True,
            "com_shift_um": 4.5,
            "pval_spat_shape": 0.3,
            "qval_spat_shape": 0.4,
            "hit_spat_shape": False,
            "rmse_3d_um": 0.8,
            "pval_warp": 0.002,
            "qval_warp": 0.008,
            "hit_warp": True,
            "signed_warp_area": 0.15,
            "k_test": 1.05,
            "dt0_test": -2.0,
        },
    ])

    fig, axes = plot_5metric_manhattan(
        test_results_df=mock_df,
        figsize=(12, 14),
        dpi=80,
    )
    assert len(axes) == 5
    plt.close(fig)


def test_plot_dual_lineage_and_diagnostics():
    """Verify dual lineage plot and diagnostics wrapper."""
    mock_df = pd.DataFrame([
        {
            "embryo_id": "test_emb",
            "cell": "ABa",
            "canon_duration": 18.0,
            "canon_mid": 25.0,
            "hit_temp_shape": True,
            "hit_temp_shift": False,
            "hit_spat_shift": False,
            "hit_spat_shape": False,
            "hit_warp": False,
            "k_test": 1.0,
            "dt0_test": 0.0,
            "pval_temp_shape": 0.01,
            "pval_temp_shift": 0.5,
            "pval_spat_shift": 0.5,
            "pval_spat_shape": 0.5,
            "pval_warp": 0.5,
            "qval_temp_shape": 0.02,
            "qval_temp_shift": 0.6,
            "qval_spat_shift": 0.6,
            "qval_spat_shape": 0.6,
            "qval_warp": 0.6,
            "pct_duration_deviation": 10.0,
            "delta_midpoint_min": 0.0,
            "com_shift_um": 0.5,
            "rmse_3d_um": 0.5,
            "signed_warp_area": 0.01,
        },
        {
            "embryo_id": "test_emb",
            "cell": "ABp",
            "canon_duration": 19.0,
            "canon_mid": 26.0,
            "hit_temp_shape": False,
            "hit_temp_shift": False,
            "hit_spat_shift": False,
            "hit_spat_shape": False,
            "hit_warp": False,
            "k_test": 1.0,
            "dt0_test": 0.0,
            "pval_temp_shape": 0.5,
            "pval_temp_shift": 0.5,
            "pval_spat_shift": 0.5,
            "pval_spat_shape": 0.5,
            "pval_warp": 0.5,
            "qval_temp_shape": 0.6,
            "qval_temp_shift": 0.6,
            "qval_spat_shift": 0.6,
            "qval_spat_shape": 0.6,
            "qval_warp": 0.6,
            "pct_duration_deviation": 0.0,
            "delta_midpoint_min": 0.0,
            "com_shift_um": 0.5,
            "rmse_3d_um": 0.5,
            "signed_warp_area": 0.0,
        },
    ])

    lineage_df = pd.DataFrame([
        {"parent": "AB", "daughter1": "ABa", "daughter2": "ABp"}
    ])

    atlas_mock = {
        "temporal_atlas": {
            "ABa": {"mu_birth": 16.0, "mu_phys": 18.0, "var_birth": 0.5, "var_phys": 1.0, "mu_mid": 25.0},
            "ABp": {"mu_birth": 16.5, "mu_phys": 19.0, "var_birth": 0.5, "var_phys": 1.0, "mu_mid": 26.0},
        },
        "cell_models": {},
    }

    fig_m, fig_l = visualize_embryo_diagnostics(
        target_embryo_id="test_emb",
        cell_scores=mock_df,
        atlas_bundle=atlas_mock,
        lineage_data=lineage_df,
        show=False,
    )

    assert fig_m is not None
    assert fig_l is not None
    plt.close(fig_m)
    plt.close(fig_l)
