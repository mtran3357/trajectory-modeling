"""Dual lineage tree layout and instantaneous warping velocity visualization."""

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import seaborn as sns

from ..functional.srvf import curve_to_srvf
from ..functional.dp_warp import align_srvf_dp_clamped
from ..functional.time_warp import regularized_monotonic_time_warp
from ..geometry.curve_align import register_curve_to_template
from ..geometry.align import align_embryo_to_spatial_template
from ..types import ReferenceAtlas
from .colors import get_canonical_clade_color


def get_canonical_clade_rank(cell_name: str) -> int:
    """Returns canonical Sulston founder clade rank (ABa < ABp < MS < E < C < D < P)."""
    c = str(cell_name).strip()
    if c.startswith("ABa"):
        return 0
    if c.startswith("ABp"):
        return 1
    if c.startswith("AB"):
        return 2
    if c.startswith("EMS") or c.startswith("MS"):
        return 3
    if c.startswith("E"):
        return 4
    if c.startswith("C"):
        return 5
    if c.startswith("D"):
        return 6
    if c.startswith("P") or c.startswith("Z"):
        return 7
    return 8


def build_interval_tree_layout(
    valid_cells: set[str], lineage_df: pd.DataFrame
) -> tuple[dict[str, float], dict[str, list[str]], set[str]]:
    """Constructs 2D horizontal coordinates for the lineage tree.
    
    Parameters
    ----------
    valid_cells : set of str
        Blastomeres present in the analysis interval.
    lineage_df : pd.DataFrame
        DataFrame with columns ['parent', 'daughter1', 'daughter2'].
        
    Returns
    -------
    x_coords : dict[str, float]
        Horizontal plotting coordinate per cell.
    children_map : dict[str, list[str]]
        Mapping from parent cell to valid child cells.
    local_roots : set of str
        Roots of trees/subtrees present in valid_cells.
    """
    G = nx.DiGraph()
    children_map: dict[str, list[str]] = {}
    parent_map: dict[str, str] = {}

    for _, row in lineage_df.iterrows():
        p = str(row["parent"]).strip() if pd.notna(row["parent"]) else None
        d1 = str(row["daughter1"]).strip() if pd.notna(row["daughter1"]) else None
        d2 = str(row["daughter2"]).strip() if pd.notna(row["daughter2"]) else None

        if p:
            kids = [d for d in [d1, d2] if d and d in valid_cells]
            if p in valid_cells and kids:
                children_map[p] = kids
                for k in kids:
                    G.add_edge(p, k)
                    parent_map[k] = p
            elif kids:
                for k in kids:
                    if k not in G:
                        G.add_node(k)

    for c in valid_cells:
        if c not in G:
            G.add_node(c)

    local_roots = {n for n in valid_cells if n not in parent_map}
    leaves = sorted(
        [n for n in G.nodes() if G.out_degree(n) == 0],
        key=lambda c: (get_canonical_clade_rank(c), c),
    )
    x_coords = {leaf: float(i * 2.5) for i, leaf in enumerate(leaves)}

    if nx.is_directed_acyclic_graph(G):
        for n in reversed(list(nx.topological_sort(G))):
            kids = list(G.successors(n))
            if kids and all(k in x_coords for k in kids):
                x_coords[n] = float(np.mean([x_coords[k] for k in kids]))
            elif n not in x_coords:
                x_coords[n] = 0.0
    else:
        for i, n in enumerate(G.nodes()):
            if n not in x_coords:
                x_coords[n] = float(i * 2.0)

    return x_coords, children_map, local_roots


def plot_warping_velocity_dual_lineage(
    test_results_df: pd.DataFrame,
    calib_meta: dict,
    lineage_df: pd.DataFrame,
    query_pos_df: pd.DataFrame | None = None,
    atlas_bundle: dict | ReferenceAtlas | None = None,
    dx_offset: float = 0.32,
    figsize: tuple[float, float] | None = None,
    base_height: float = 15.0,
    width_per_cell: float = 0.45,
    dpi: int = 130,
    annotate_outliers: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Paired reference vs observed lineage tree with instantaneous warp coloring.
    
    Self-contained: Performs on-the-fly spatial alignment of query_pos_df against
    the consensus template if aligned coordinates are not already present.
    """
    sns.set_theme(style="ticks")
    test_embryo_id = str(test_results_df["embryo_id"].iloc[0])

    bundle = atlas_bundle.to_dict() if isinstance(atlas_bundle, ReferenceAtlas) else atlas_bundle

    if bundle is not None:
        temporal_atlas = bundle.get("temporal_atlas", {})
        cell_models = bundle.get("cell_models", {})
    else:
        temporal_atlas = calib_meta.get("temporal_atlas", {})
        cell_models = calib_meta.get("fitted_models", {})

    if not temporal_atlas:
        raise ValueError("Missing temporal_atlas dictionary in calib_meta/atlas_bundle.")

    c_col = "cell" if "cell" in test_results_df.columns else "cell_name"
    valid_cells = set(test_results_df[c_col].unique())
    x_coords, children_map, local_roots = build_interval_tree_layout(valid_cells, lineage_df)

    if figsize is None:
        calc_width = max(26.0, len(x_coords) * width_per_cell)
        figsize = (calc_width, base_height)

    computed_gamma_dots = {}
    time_grid = np.linspace(0.0, 1.0, 40)

    # Compute gamma_dot(t) for observed tracks if query coordinate data is supplied
    if query_pos_df is not None:
        e_col = bundle.get("embryo_col", "series") if bundle else "series"
        cell_key = bundle.get("cell_col", "cell") if bundle else "cell"
        t_key = bundle.get("time_col", "time") if bundle else "time"

        emb_pos = query_pos_df[query_pos_df[e_col].astype(str) == test_embryo_id].copy()

        # Check if coordinates need alignment to consensus template
        aligned_cols = ["x_aligned_um", "y_aligned_um", "z_aligned_um"]
        if not all(col in emb_pos.columns for col in aligned_cols) and bundle is not None:
            emb_pos_aligned, _ = align_embryo_to_spatial_template(
                emb_coords_df=emb_pos,
                spatial_template_df=bundle["spatial_template"],
                micron_cols=bundle["micron_cols"],
                aligned_cols=aligned_cols,
                cell_col=cell_key,
                max_inlier_dist_um=bundle["max_inlier_dist_um"],
                allow_scaling=True,
            )
            emb_pos = emb_pos_aligned

        if all(col in emb_pos.columns for col in aligned_cols):
            for c in valid_cells:
                if c in cell_models:
                    model_dict = cell_models[c].to_dict() if hasattr(cell_models[c], "to_dict") else cell_models[c]
                    template_curve = model_dict.get("template_curve")
                    if template_curve is not None:
                        tau_grid = np.asarray(model_dict.get("time_grid", time_grid), dtype=float)
                        tau_cutoff = float(model_dict.get("tau_cutoff", tau_grid[-1] if len(tau_grid) > 0 else 30.0))
                        c_sub = emb_pos[emb_pos[cell_key] == c].sort_values(t_key)
                        if len(c_sub) >= 3:
                            t_vals = c_sub[t_key].values.astype(float)
                            t_birth = float(t_vals.min())
                            tau_vals = t_vals - t_birth
                            valid_mask = tau_vals <= tau_cutoff + 1e-4
                            if np.sum(valid_mask) >= 3:
                                tau_test = tau_vals[valid_mask]
                                xyz = c_sub[aligned_cols].values.astype(float)[valid_mask]
                                xyz_shape = xyz - np.mean(xyz, axis=0)
                                _, _, coords_aligned = register_curve_to_template(
                                    s_obs=tau_test,
                                    coords_centered=xyz_shape,
                                    template_curve=template_curve,
                                    s_grid=tau_grid,
                                )
                                gamma_test, _, _ = regularized_monotonic_time_warp(
                                    tau_obs=tau_test,
                                    coords_aligned=coords_aligned,
                                    template_curve=template_curve,
                                    tau_grid=tau_grid,
                                    lambda_reg=float(bundle.get("warping_lambda", 10.0)) if bundle else 10.0,
                                )
                                active_tau = tau_grid[tau_grid <= tau_test.max() + 1e-4]
                                g_dot = np.maximum(np.gradient(gamma_test, active_tau), 0.0)
                                t_norm_grid = (active_tau - active_tau[0]) / max(active_tau[-1] - active_tau[0], 1e-4)
                                computed_gamma_dots[c] = (t_norm_grid, g_dot)
                    elif "mu_srvf" in model_dict and np.sum(np.abs(model_dict["mu_srvf"])) > 0:
                        c_sub = emb_pos[emb_pos[cell_key] == c].sort_values(t_key)
                        if len(c_sub) >= 3:
                            t_vals = c_sub[t_key].values.astype(float)
                            t_min, t_max = t_vals.min(), t_vals.max()
                            t_rel = (t_vals - t_min) / max(t_max - t_min, 1e-6)
                            xyz = c_sub[aligned_cols].values.astype(float)
                            xyz_shape = xyz - np.mean(xyz, axis=0)
                            curve_interp = interp1d(t_rel, xyz_shape, axis=0, kind="linear", fill_value="extrapolate")
                            test_srvf = curve_to_srvf(curve_interp(time_grid), time_grid)
                            mu_srvf = np.asarray(model_dict["mu_srvf"], dtype=float)
                            gamma_test = align_srvf_dp_clamped(mu_srvf, test_srvf, time_grid)
                            g_dot = np.maximum(np.gradient(gamma_test, time_grid), 0.0)
                            computed_gamma_dots[c] = (time_grid, g_dot)

    ref_data = {}
    obs_data = {}

    for c in valid_cells:
        if c not in temporal_atlas:
            continue
        t_stat = temporal_atlas[c]
        row_te = test_results_df[test_results_df[c_col] == c].iloc[0]

        ref_b = float(t_stat["mu_birth"])
        ref_d = ref_b + float(t_stat["mu_phys"])
        std_b = np.sqrt(max(t_stat.get("var_birth", 0.5), 1e-4))
        std_d = np.sqrt(max(t_stat.get("var_birth", 0.5) + t_stat.get("var_phys", 1.0), 1e-4))

        ref_data[c] = {
            "t_birth": ref_b,
            "t_divide": ref_d,
            "t_mid": 0.5 * (ref_b + ref_d),
            "std_b": std_b,
            "std_d": std_d,
            "color": get_canonical_clade_color(c),
        }

        obs_dur = (
            float(row_te["canon_duration"])
            if "canon_duration" in row_te
            else float(t_stat["mu_phys"] + row_te.get("delta_duration_min", 0.0))
        )
        obs_mid = (
            float(row_te["canon_mid"])
            if "canon_mid" in row_te
            else float(ref_data[c]["t_mid"] + row_te.get("delta_midpoint_min", 0.0))
        )

        obs_b = obs_mid - 0.5 * obs_dur
        obs_d = obs_mid + 0.5 * obs_dur

        hit_columns = [
            "hit_temp_shape",
            "hit_temp_shift",
            "hit_warp",
            "hit_spat_shift",
            "hit_spat_rot",
            "hit_spat_shape",
        ]
        modality_hits = {
            col.replace("hit_", ""): bool(row_te.get(col, False))
            for col in hit_columns
        }
        n_hits = int(sum(modality_hits.values()))

        obs_data[c] = {
            "t_birth": obs_b,
            "t_divide": obs_d,
            "t_mid": obs_mid,
            "hits": modality_hits,
            "n_hits": n_hits,
            "is_outlier": n_hits > 0,
        }

    norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=2.0)
    cmap = plt.get_cmap("coolwarm")
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Parent-child connectors
    for parent, kids in children_map.items():
        if parent not in ref_data or parent not in x_coords:
            continue
        xp = x_coords[parent]
        ref_pd = ref_data[parent]["t_divide"]
        obs_pd = obs_data[parent]["t_divide"]

        for k in kids:
            if k not in ref_data or k not in x_coords:
                continue
            xk = x_coords[k]
            ref_kb = ref_data[k]["t_birth"]
            obs_kb = obs_data[k]["t_birth"]

            ax.plot(
                [xp - dx_offset, xk - dx_offset],
                [ref_pd, ref_kb],
                color="#7f8c8d",
                linestyle="--",
                linewidth=1.0,
                alpha=0.5,
                zorder=2,
            )
            ax.plot(
                [xp + dx_offset, xk + dx_offset],
                [obs_pd, obs_kb],
                color="#34495e",
                linestyle=":",
                linewidth=1.1,
                alpha=0.6,
                zorder=2,
            )

    # Render tracks
    for c, xc in x_coords.items():
        if c not in ref_data:
            continue
        xr = xc - dx_offset
        xo = xc + dx_offset
        r_info = ref_data[c]
        o_info = obs_data[c]

        # Reference track with +/- 1 SD caps
        ax.plot(
            [xr, xr],
            [r_info["t_birth"], r_info["t_divide"]],
            color=r_info["color"],
            linewidth=4.5,
            solid_capstyle="round",
            alpha=0.95,
            zorder=3,
        )
        ax.errorbar(
            x=xr,
            y=r_info["t_birth"],
            yerr=r_info["std_b"],
            fmt="none",
            ecolor="#000000",
            elinewidth=1.2,
            capsize=3.0,
            capthick=1.2,
            zorder=5,
        )
        ax.errorbar(
            x=xr,
            y=r_info["t_divide"],
            yerr=r_info["std_d"],
            fmt="none",
            ecolor="#000000",
            elinewidth=1.2,
            capsize=3.0,
            capthick=1.2,
            zorder=5,
        )

        # Observed track colored by gamma_dot(t)
        if c in computed_gamma_dots:
            t_grid_c, g_dot = computed_gamma_dots[c]
            y_pts = o_info["t_birth"] + t_grid_c * (o_info["t_divide"] - o_info["t_birth"])
            points = np.array([np.full_like(y_pts, xo), y_pts]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(
                segments,
                cmap=cmap,
                norm=norm,
                linewidth=4.5,
                zorder=4,
                capstyle="round",
            )
            lc.set_array(0.5 * (g_dot[:-1] + g_dot[1:]))
            ax.add_collection(lc)
        else:
            ax.plot(
                [xo, xo],
                [o_info["t_birth"], o_info["t_divide"]],
                color="#95a5a6",
                linewidth=4.5,
                solid_capstyle="round",
                zorder=3,
            )

        # Outlier markers
        if annotate_outliers and o_info["is_outlier"]:
            hits = o_info["hits"]
            if o_info["n_hits"] >= 3:
                marker_color = "#8e44ad"
                marker_size = 75
            elif hits.get("temp_shift", False) or hits.get("warp", False):
                marker_color = "#e74c3c"
                marker_size = 60
            elif hits.get("spat_shift", False) or hits.get("spat_rot", False) or hits.get("spat_shape", False):
                marker_color = "#e67e22"
                marker_size = 60
            else:
                marker_color = "#3498db"
                marker_size = 60

            ax.scatter(
                [xo],
                [o_info["t_mid"]],
                s=marker_size,
                color=marker_color,
                edgecolors="#ffffff",
                linewidth=1.4,
                zorder=7,
            )

            hit_labels = [
                name.replace("temp_shape", "Tshape")
                .replace("temp_shift", "Tshift")
                .replace("spat_shift", "Sshift")
                .replace("spat_shape", "Sshape")
                .replace("warp", "Warp")
                for name, is_hit in hits.items()
                if is_hit
            ]
            if hit_labels:
                ax.text(
                    xo + 0.12,
                    o_info["t_mid"],
                    ", ".join(hit_labels),
                    fontsize=6.8,
                    color=marker_color,
                    fontweight="bold",
                    va="center",
                    ha="left",
                    zorder=8,
                )

        # Midpoint connector & label
        ax.plot(
            [xr, xo],
            [r_info["t_mid"], o_info["t_mid"]],
            color="#95a5a6",
            linestyle=":",
            linewidth=1.0,
            alpha=0.8,
            zorder=1,
        )
        ax.text(
            xc,
            min(r_info["t_birth"], o_info["t_birth"]) - 2.8,
            c,
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
            rotation=90,
            color="#2c3e50",
        )

    all_times = (
        [v["t_birth"] - v["std_b"] for v in ref_data.values()]
        + [v["t_divide"] + v["std_d"] for v in ref_data.values()]
        + [v["t_divide"] for v in obs_data.values()]
    )
    ax.set_ylim(max(all_times) + 8.0, min(all_times) - 8.0)
    ax.set_xlim(min(x_coords.values()) - 2.0, max(x_coords.values()) + 2.0)
    sorted_items = sorted(x_coords.items(), key=lambda t: t[1])
    ax.set_xticks([t[1] for t in sorted_items])
    ax.set_xticklabels([t[0] for t in sorted_items], rotation=90, fontsize=8)
    ax.set_ylabel("Canonical Developmental Time (min)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Lineage Blastomeres", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.4, axis="y")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.015, aspect=35)
    cbar.set_label(
        r"Observed instantaneous warping velocity $\dot{\gamma}(t)$"
        "\n"
        r"($>1$: accelerated / ahead, $<1$: decelerated / behind)",
        fontsize=10,
        fontweight="bold",
    )

    legend_handles = [
        Line2D([0], [0], color="#1f77b4", lw=4.0, label="Reference: ABa clade"),
        Line2D([0], [0], color="#2ca02c", lw=4.0, label="Reference: ABp clade"),
        Line2D([0], [0], color="#ff7f0e", lw=4.0, label="Reference: EMS / MS clade"),
        Line2D([0], [0], color="#2ecc71", lw=4.0, label="Reference: E clade"),
        Line2D([0], [0], color="#9467bd", lw=4.0, label="Reference: C clade"),
        Line2D([0], [0], color="#8c564b", lw=4.0, label="Reference: D clade"),
        Line2D([0], [0], color="#e377c2", lw=4.0, label="Reference: Germline / P4 clade"),
        Line2D([0], [0], marker="_", color="#000000", mew=1.4, ms=8, label=r"Reference timing ($\pm1\sigma$)"),
        Line2D([0], [0], color="#b0bec5", lw=4.0, label=r"Observed: colored by $\dot{\gamma}(t)$"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c", markeredgecolor="#ffffff", ms=7, label="Empirical temporal-shift / warp outlier"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e67e22", markeredgecolor="#ffffff", ms=7, label="Empirical spatial outlier"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#8e44ad", markeredgecolor="#ffffff", ms=7, label="Multi-modal outlier (>=3 modalities)"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        frameon=True,
        fontsize=8.0,
        ncol=2,
        title="Lineage Atlas & Anomaly Modalities",
    )

    if "k_test" in calib_meta:
        k_val = float(calib_meta["k_test"])
    elif "k_test" in test_results_df.columns:
        k_val = float(test_results_df["k_test"].iloc[0])
    else:
        k_val = 1.0

    if "dt0_test" in calib_meta:
        dt0_val = float(calib_meta["dt0_test"])
    elif "dt0_test" in test_results_df.columns:
        dt0_val = float(test_results_df["dt0_test"].iloc[0])
    else:
        dt0_val = 0.0

    qc_items = []
    if "scale_s" in calib_meta:
        qc_items.append(f"Spatial Scale $s = {float(calib_meta['scale_s']):.3f}$")
    elif "scale_s" in test_results_df.columns:
        qc_items.append(f"Spatial Scale $s = {float(test_results_df['scale_s'].iloc[0]):.3f}$")

    if "inlier_ratio" in calib_meta:
        qc_items.append(f"Landmark Inliers = {float(calib_meta['inlier_ratio']) * 100:.1f}%")
    elif "inlier_ratio" in test_results_df.columns:
        qc_items.append(f"Landmark Inliers = {float(test_results_df['inlier_ratio'].iloc[0]) * 100:.1f}%")

    if "mean_inlier_res_um" in calib_meta:
        qc_items.append(f"Mean Inlier Residual = {float(calib_meta['mean_inlier_res_um']):.1f} $\\mu\\mathrm{{m}}$")
    elif "mean_inlier_res_um" in test_results_df.columns:
        qc_items.append(f"Mean Inlier Residual = {float(test_results_df['mean_inlier_res_um'].iloc[0]):.1f} $\\mu\\mathrm{{m}}$")

    if "temporal_inlier_ratio" in calib_meta:
        qc_items.append(f"Temporal Inliers = {float(calib_meta['temporal_inlier_ratio']) * 100:.1f}%")
    elif "temporal_inlier_ratio" in test_results_df.columns:
        qc_items.append(f"Temporal Inliers = {float(test_results_df['temporal_inlier_ratio'].iloc[0]) * 100:.1f}%")

    qc_subtitle = "  |  ".join(qc_items)
    full_title = (
        f"Paired Reference vs. Observed Lineage Atlas: Test Embryo '{test_embryo_id}'\n"
        rf"Affine Pace $K_e = {k_val:.4f}$, Timing Offset $dt_0 = {dt0_val:+.2f}$ min"
    )
    if qc_subtitle:
        full_title += f"\nWhole-Embryo QC: {qc_subtitle}"

    plt.suptitle(
        full_title,
        fontsize=12.5,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    return fig, ax
