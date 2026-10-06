"""Unit tests for visualization routines."""

import matplotlib
matplotlib.use("Agg")  # Headless backend for testing
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spatiotemporal_atlas.viz import (
    get_canonical_clade_color,
    assign_lineage,
    build_interval_tree_layout,
    plot_5metric_manhattan,
    plot_6metric_manhattan,
    plot_cell_trajectory_diagnostic_dashboard,
    plot_interactive_3d_trajectory,
    plot_cell_cohort_3d_comparison,
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
    assert x_coords["ABa"] < x_coords["ABp"]

    # Verify canonical Sulston founder clade ordering: ABa < ABp < MS < E < C < D < P4
    multi_lineage_df = pd.DataFrame([
        {"parent": "EMS", "daughter1": "MS", "daughter2": "E"},
        {"parent": "P2", "daughter1": "C", "daughter2": "P3"},
        {"parent": "P3", "daughter1": "D", "daughter2": "P4"},
    ])
    clade_cells = {"ABa", "ABp", "MS", "E", "C", "D", "P4"}
    x_clades, _, _ = build_interval_tree_layout(clade_cells, multi_lineage_df)
    assert x_clades["ABa"] < x_clades["ABp"] < x_clades["MS"] < x_clades["E"] < x_clades["C"] < x_clades["D"] < x_clades["P4"]


def test_plot_6metric_manhattan():
    """Verify Manhattan plot creates 6 axes across the spatiotemporal suite without errors."""
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
            "delta_birth_min": 3.2,
            "pval_spat_shift": 0.1,
            "qval_spat_shift": 0.2,
            "hit_spat_shift": False,
            "com_shift_um": 1.2,
            "pval_spat_rot": 0.03,
            "qval_spat_rot": 0.05,
            "hit_spat_rot": True,
            "rot_angle_deg": 25.0,
            "pval_spat_shape": 0.0005,
            "qval_spat_shape": 0.002,
            "hit_spat_shape": True,
            "rmse_3d_um": 2.1,
            "pval_warp": 0.5,
            "qval_warp": 0.6,
            "hit_warp": False,
            "rms_warp_min": 0.02,
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
            "delta_birth_min": -0.5,
            "pval_spat_shift": 0.001,
            "qval_spat_shift": 0.005,
            "hit_spat_shift": True,
            "com_shift_um": 4.5,
            "pval_spat_rot": 0.4,
            "qval_spat_rot": 0.5,
            "hit_spat_rot": False,
            "rot_angle_deg": 5.0,
            "pval_spat_shape": 0.3,
            "qval_spat_shape": 0.4,
            "hit_spat_shape": False,
            "rmse_3d_um": 0.8,
            "pval_warp": 0.002,
            "qval_warp": 0.008,
            "hit_warp": True,
            "rms_warp_min": 0.15,
            "k_test": 1.05,
            "dt0_test": -2.0,
        },
    ])

    fig, axes = plot_6metric_manhattan(
        test_results_df=mock_df,
        figsize=(12, 14),
        dpi=80,
    )
    assert len(axes) == 6
    plt.close(fig)

    # Backward compatibility alias test
    fig5, axes5 = plot_5metric_manhattan(
        test_results_df=mock_df,
        figsize=(12, 14),
        dpi=80,
    )
    assert len(axes5) == 6
    plt.close(fig5)


def test_plot_cell_trajectory_diagnostic_dashboard():
    """Verify single-cell trajectory diagnostic dashboard renders 6 panels."""
    time_grid = np.linspace(0, 20, 20)
    mock_pos_df = pd.DataFrame({
        "series": ["test_emb"] * 15,
        "cell": ["ABa"] * 15,
        "time": np.linspace(10, 25, 15),
        "x_aligned_um": np.sin(np.linspace(0, 2, 15)),
        "y_aligned_um": np.cos(np.linspace(0, 2, 15)),
        "z_aligned_um": np.linspace(0, 5, 15),
    })

    mock_model = {
        "cell": "ABa",
        "time_grid": time_grid,
        "tau_cutoff": 20.0,
        "template_curve": np.zeros((20, 3)),
        "s_dense": np.linspace(0, 1, 50),
        "dense_mu_3d": np.zeros((50, 3)),
        "dense_cov_3d": np.array([np.eye(3) * 0.1 for _ in range(50)]),
        "dense_pred": {
            col: {"mu": np.zeros(50), "std": np.ones(50) * 0.1}
            for col in ["x_aligned_um", "y_aligned_um", "z_aligned_um"]
        },
        "mu_com": np.array([0.0, 0.0, 2.5]),
    }

    mock_temporal_atlas = {
        "ABa": {
            "mu_birth": 10.0,
            "mu_phys": 15.0,
            "var_birth": 0.5,
            "var_phys": 1.0,
            "var_path": 0.8,
        }
    }

    mock_target_row = pd.Series({
        "cell": "ABa",
        "series": "test_emb",
        "pval_temp_shape": 0.05,
        "qval_temp_shape": 0.05,
        "hit_temp_shape": False,
        "pval_temp_shift": 0.05,
        "qval_temp_shift": 0.05,
        "hit_temp_shift": False,
        "pval_warp": 0.05,
        "qval_warp": 0.05,
        "hit_warp": False,
        "pval_spat_shift": 0.05,
        "qval_spat_shift": 0.05,
        "hit_spat_shift": False,
        "pval_spat_rot": 0.05,
        "qval_spat_rot": 0.05,
        "hit_spat_rot": False,
        "pval_spat_shape": 0.05,
        "qval_spat_shape": 0.05,
        "hit_spat_shape": False,
        "n_outlier_modalities": 0,
        "rms_warp_min": 0.2,
        "delta_birth_min": 0.5,
        "com_shift_um": 0.8,
        "rot_angle_deg": 12.0,
        "rmse_3d_um": 0.4,
    })

    fig, axes_dict = plot_cell_trajectory_diagnostic_dashboard(
        cell_name="ABa",
        embryo_id="test_emb",
        pos_df=mock_pos_df,
        target_row=mock_target_row,
        cell_model=mock_model,
        temporal_atlas=mock_temporal_atlas,
        figsize=(12, 10),
        dpi=80,
    )

    assert fig is not None
    assert len(axes_dict) == 7
    for key in ["ax_x", "ax_y", "ax_z", "ax_gamma", "ax_3d", "ax_trans", "ax_timing"]:
        assert key in axes_dict
    plt.close(fig)

    # Test return_interactive_3d=True
    fig2, axes_dict2, fig_3d = plot_cell_trajectory_diagnostic_dashboard(
        cell_name="ABa",
        embryo_id="test_emb",
        pos_df=mock_pos_df,
        target_row=mock_target_row,
        cell_model=mock_model,
        temporal_atlas=mock_temporal_atlas,
        figsize=(12, 10),
        dpi=80,
        return_interactive_3d=True,
    )
    assert fig2 is not None
    assert fig_3d is not None
    plt.close(fig2)


def test_plot_interactive_3d_trajectory():
    """Verify plot_interactive_3d_trajectory generates an interactive Plotly 3D figure with traces."""
    time_grid = np.linspace(0, 20, 20)
    mock_pos_df = pd.DataFrame({
        "series": ["test_emb"] * 15,
        "cell": ["ABa"] * 15,
        "time": np.linspace(10, 25, 15),
        "x_aligned_um": np.sin(np.linspace(0, 2, 15)),
        "y_aligned_um": np.cos(np.linspace(0, 2, 15)),
        "z_aligned_um": np.linspace(0, 5, 15),
    })
    mock_model = {
        "cell": "ABa",
        "time_grid": time_grid,
        "tau_cutoff": 20.0,
        "template_curve": np.zeros((20, 3)),
        "s_dense": np.linspace(0, 1, 50),
        "dense_mu_3d": np.zeros((50, 3)),
        "dense_cov_3d": np.array([np.eye(3) * 0.1 for _ in range(50)]),
        "dense_pred": {
            col: {"mu": np.zeros(50), "std": np.ones(50) * 0.1}
            for col in ["x_aligned_um", "y_aligned_um", "z_aligned_um"]
        },
        "mu_com": np.array([0.0, 0.0, 2.5]),
    }
    mock_temporal_atlas = {
        "ABa": {
            "mu_birth": 10.0,
            "mu_phys": 15.0,
            "var_birth": 0.5,
            "var_phys": 1.0,
            "var_path": 0.8,
        }
    }
    mock_target_row = pd.Series({
        "cell": "ABa",
        "series": "test_emb",
        "pval_spat_rot": 0.05,
        "qval_spat_rot": 0.05,
        "rot_angle_deg": 12.0,
        "pval_spat_shape": 0.05,
        "qval_spat_shape": 0.05,
        "rmse_3d_um": 0.4,
    })

    fig = plot_interactive_3d_trajectory(
        cell_name="ABa",
        embryo_id="test_emb",
        pos_df=mock_pos_df,
        target_row=mock_target_row,
        cell_model=mock_model,
        temporal_atlas=mock_temporal_atlas,
    )
    assert fig is not None
    # Traces: Reference path, Observed track, Ref Start, Ref End, Obs Start, Obs End
    assert len(fig.data) >= 5
    trace_names = [t.name for t in fig.data]
    assert any("Ref Start" in n for n in trace_names)
    assert any("Ref End" in n for n in trace_names)
    assert any("Obs Start" in n for n in trace_names)
    assert any("Obs End" in n for n in trace_names)


def test_plot_cell_trajectory_diagnostic_dashboard_save_interactive_html():
    """Verify plot_cell_trajectory_diagnostic_dashboard saves interactive 3D HTML when requested."""
    import tempfile
    from pathlib import Path

    time_grid = np.linspace(0, 20, 20)
    mock_pos_df = pd.DataFrame({
        "series": ["test_emb"] * 15,
        "cell": ["ABa"] * 15,
        "time": np.linspace(10, 25, 15),
        "x_aligned_um": np.sin(np.linspace(0, 2, 15)),
        "y_aligned_um": np.cos(np.linspace(0, 2, 15)),
        "z_aligned_um": np.linspace(0, 5, 15),
    })
    mock_model = {
        "cell": "ABa",
        "time_grid": time_grid,
        "tau_cutoff": 20.0,
        "template_curve": np.zeros((20, 3)),
        "s_dense": np.linspace(0, 1, 50),
        "dense_mu_3d": np.zeros((50, 3)),
        "dense_cov_3d": np.array([np.eye(3) * 0.1 for _ in range(50)]),
        "dense_pred": {
            col: {"mu": np.zeros(50), "std": np.ones(50) * 0.1}
            for col in ["x_aligned_um", "y_aligned_um", "z_aligned_um"]
        },
        "mu_com": np.array([0.0, 0.0, 2.5]),
    }
    mock_temporal_atlas = {
        "ABa": {
            "mu_birth": 10.0,
            "mu_phys": 15.0,
            "var_birth": 0.5,
            "var_phys": 1.0,
            "var_path": 0.8,
        }
    }
    mock_target_row = pd.Series({
        "cell": "ABa",
        "series": "test_emb",
        "pval_temp_shape": 0.05,
        "qval_temp_shape": 0.05,
        "hit_temp_shape": False,
        "pval_temp_shift": 0.05,
        "qval_temp_shift": 0.05,
        "hit_temp_shift": False,
        "pval_warp": 0.05,
        "qval_warp": 0.05,
        "hit_warp": False,
        "pval_spat_shift": 0.05,
        "qval_spat_shift": 0.05,
        "hit_spat_shift": False,
        "pval_spat_rot": 0.05,
        "qval_spat_rot": 0.05,
        "hit_spat_rot": False,
        "pval_spat_shape": 0.05,
        "qval_spat_shape": 0.05,
        "hit_spat_shape": False,
        "n_outlier_modalities": 0,
        "rms_warp_min": 0.2,
        "delta_birth_min": 0.5,
        "com_shift_um": 0.8,
        "rot_angle_deg": 12.0,
        "rmse_3d_um": 0.4,
    })

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_html = Path(tmp_dir) / "test_saved_3d.html"
        fig, axes_dict = plot_cell_trajectory_diagnostic_dashboard(
            cell_name="ABa",
            embryo_id="test_emb",
            pos_df=mock_pos_df,
            target_row=mock_target_row,
            cell_model=mock_model,
            temporal_atlas=mock_temporal_atlas,
            save_interactive_html=out_html,
        )
        assert fig is not None
        assert out_html.exists()
        assert out_html.stat().st_size > 0
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


def test_trajectory_dashboard_pacing_normalization():
    """Verify trajectory dashboard scales tau by embryo pacing Ke."""
    time_grid = np.linspace(0, 30, 30)
    mock_pos_df = pd.DataFrame({
        "series": ["test_emb"] * 31,
        "cell": ["Cpaa"] * 31,
        "time": np.linspace(0, 30, 31),  # raw span: 30 minutes
        "x_aligned_um": np.zeros(31),
        "y_aligned_um": np.zeros(31),
        "z_aligned_um": np.zeros(31),
    })
    mock_model = {
        "cell": "Cpaa",
        "time_grid": time_grid,
        "tau_cutoff": 35.0,
        "template_curve": np.zeros((30, 3)),
        "s_dense": np.linspace(0, 1, 50),
        "dense_mu_3d": np.zeros((50, 3)),
        "dense_cov_3d": np.array([np.eye(3) * 0.1 for _ in range(50)]),
        "dense_pred": {
            col: {"mu": np.zeros(50), "std": np.ones(50) * 0.1}
            for col in ["x_aligned_um", "y_aligned_um", "z_aligned_um"]
        },
        "mu_com": np.array([0.0, 0.0, 0.0]),
    }
    mock_temporal_atlas = {
        "Cpaa": {
            "mu_birth": 50.0,
            "mu_phys": 25.0,
            "var_birth": 0.5,
            "var_phys": 1.0,
            "var_path": 0.8,
        }
    }
    mock_target_row = pd.Series({
        "cell": "Cpaa",
        "series": "test_emb",
        "k_test": 1.25,
        "canon_duration": 24.0,
    })

    fig, axes_dict = plot_cell_trajectory_diagnostic_dashboard(
        cell_name="Cpaa",
        embryo_id="test_emb",
        pos_df=mock_pos_df,
        target_row=mock_target_row,
        cell_model=mock_model,
        temporal_atlas=mock_temporal_atlas,
        figsize=(12, 10),
        dpi=80,
    )
    assert fig is not None
    plt.close(fig)


def test_plot_cell_cohort_3d_comparison():
    """Verify multi-embryo 3D cohort comparison generates static grid and interactive HTML."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        time_grid = np.linspace(0, 20, 20)
        mock_pos_df = pd.DataFrame({
            "series": ["emb_1"] * 10 + ["emb_2"] * 10,
            "cell": ["ABa"] * 20,
            "time": list(np.linspace(10, 25, 10)) + list(np.linspace(11, 26, 10)),
            "x_aligned_um": np.sin(np.linspace(0, 2, 20)),
            "y_aligned_um": np.cos(np.linspace(0, 2, 20)),
            "z_aligned_um": np.linspace(0, 5, 20),
        })

        mock_scores = pd.DataFrame([
            {"embryo_id": "emb_1", "cell": "ABa", "k_test": 1.0, "canon_duration": 15.0, "qval_spat_shape": 0.02, "hit_spat_shape": True},
            {"embryo_id": "emb_2", "cell": "ABa", "k_test": 1.1, "canon_duration": 15.0, "qval_spat_shape": 0.40, "hit_spat_shape": False},
        ])

        mock_model = {
            "cell": "ABa",
            "time_grid": time_grid,
            "tau_cutoff": 20.0,
            "template_curve": np.zeros((20, 3)),
            "s_dense": np.linspace(0, 1, 50),
            "dense_mu_3d": np.zeros((50, 3)),
            "dense_cov_3d": np.array([np.eye(3) * 0.1 for _ in range(50)]),
            "dense_pred": {
                col: {"mu": np.zeros(50), "std": np.ones(50) * 0.1}
                for col in ["x_aligned_um", "y_aligned_um", "z_aligned_um"]
            },
            "mu_com": np.array([0.0, 0.0, 2.5]),
        }

        mock_temporal_atlas = {
            "ABa": {
                "mu_birth": 10.0,
                "mu_phys": 15.0,
                "var_birth": 0.5,
                "var_phys": 1.0,
                "var_path": 0.8,
            }
        }

        out_html = tmp_path / "test_cohort_locked.html"
        out_png = tmp_path / "test_cohort_grid.png"

        fig_static, axes, fig_plotly = plot_cell_cohort_3d_comparison(
            cell_name="ABa",
            pos_df=mock_pos_df,
            cell_scores=mock_scores,
            cell_model=mock_model,
            temporal_atlas=mock_temporal_atlas,
            save_html=out_html,
            save_png=out_png,
            ncols=2,
            dpi=80,
        )

        assert fig_static is not None
        assert len(axes) == 2
        assert out_png.exists()
        assert out_html.exists()
        html_content = out_html.read_text(encoding="utf-8")
        assert "plotly_relayout" in html_content
        assert "camera" in html_content
        plt.close(fig_static)


def test_plot_wt_null_metric_distributions():
    import tempfile
    from spatiotemporal_atlas.viz import plot_wt_null_metric_distributions

    np.random.seed(42)
    n = 30
    mock_df = pd.DataFrame({
        "embryo_id": [f"emb_{i % 5}" for i in range(n)],
        "cell": ["ABa"] * n,
        "delta_duration_min": np.random.normal(1.0, 0.5, n),
        "delta_birth_min": np.random.normal(0.8, 0.4, n),
        "rms_warp_min": np.abs(np.random.normal(0.7, 0.3, n)),
        "com_shift_um": np.abs(np.random.normal(1.5, 0.6, n)),
        "rot_angle_deg": np.abs(np.random.normal(35.0, 15.0, n)),
        "rmse_3d_um": np.abs(np.random.normal(1.2, 0.4, n)),
        "z_temp_shape": np.random.normal(0.0, 1.0, n),
        "z_temp_shift": np.random.normal(0.0, 1.0, n),
        "z_warp": np.random.normal(0.0, 1.0, n),
        "d_spat_shift": np.abs(np.random.normal(1.2, 0.5, n)),
        "z_rot_angle": np.random.normal(0.0, 1.0, n),
        "d_spat_shape": np.abs(np.random.normal(0.6, 0.2, n)),
    })

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_png = Path(tmp_dir) / "test_wt_null.png"
        fig, axes = plot_wt_null_metric_distributions(
            scores_df=mock_df,
            figsize=(10.0, 9.0),
            dpi=80,
            save_path=out_png,
        )

        assert fig is not None
        assert axes.shape == (3, 2)
        assert out_png.exists()
        plt.close(fig)


