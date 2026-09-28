"""Scale-normalized Generalized Procrustes Analysis (GPA) consensus atlas generation."""

import numpy as np
import pandas as pd
from .ransac_pose import compute_ransac_similarity_pose


def compute_gpa_consensus_atlas(
    centroids_df: pd.DataFrame,
    embryo_col: str = "series",
    cell_col: str = "cell",
    micron_cols: list[str] | tuple[str, ...] = ("x_um", "y_um", "z_um"),
    max_inlier_dist_um: float = 4.0,
    n_procrustes_iter: int = 4,
    allow_scaling: bool = True,
    n_ransac_iter: int = 300,
) -> tuple[pd.DataFrame, str]:
    """Constructs a scale-normalized Fréchet mean consensus atlas across blastomeres.
    
    Parameters
    ----------
    centroids_df : pd.DataFrame
        DataFrame of blastomere spatial centroids with columns [embryo_col, cell_col, *micron_cols].
    embryo_col : str
        Column containing embryo identifiers.
    cell_col : str
        Column containing blastomere cell names.
    micron_cols : list or tuple of str
        Names of spatial coordinate columns in microns.
    max_inlier_dist_um : float
        RANSAC inlier distance threshold in microns.
    n_procrustes_iter : int
        Number of Procrustean mean updating iterations.
    allow_scaling : bool
        Whether to permit uniform isotropic scaling.
    n_ransac_iter : int
        Number of RANSAC iterations per alignment.
        
    Returns
    -------
    consensus_template_df : pd.DataFrame
        Consensus spatial template indexed by cell_col with columns micron_cols.
    seed_embryo_id : str
        ID of the embryo chosen as the seed reference.
    """
    micron_cols = list(micron_cols)
    counts = centroids_df.groupby(embryo_col)[cell_col].count()
    seed_embryo_id = str(counts.idxmax())

    seed_cents = centroids_df[centroids_df[embryo_col] == seed_embryo_id]
    M_seed = seed_cents.set_index(cell_col)[micron_cols].copy()
    M_curr = M_seed.copy()

    all_embryos = list(centroids_df[embryo_col].unique())

    pop_centroid_sizes = []
    for emb_id in all_embryos:
        emb_pts = centroids_df[centroids_df[embryo_col] == emb_id][micron_cols].values
        if len(emb_pts) >= 4:
            pts_centered = emb_pts - np.mean(emb_pts, axis=0)
            rms_size = np.sqrt(np.mean(np.sum(pts_centered ** 2, axis=1)))
            pop_centroid_sizes.append(rms_size)

    target_rms_size = float(np.median(pop_centroid_sizes)) if pop_centroid_sizes else 1.0

    for _ in range(n_procrustes_iter):
        aligned_snapshots = []

        for emb_id in all_embryos:
            emb_cents = centroids_df[centroids_df[embryo_col] == emb_id].set_index(cell_col)
            common = sorted(list(set(emb_cents.index).intersection(set(M_curr.index))))
            if len(common) < 4:
                continue

            P_arr = emb_cents.loc[common, micron_cols].values
            Q_arr = M_curr.loc[common, micron_cols].values

            s, R, t, _ = compute_ransac_similarity_pose(
                P=P_arr,
                Q=Q_arr,
                max_inlier_dist_um=max_inlier_dist_um,
                n_iterations=n_ransac_iter,
                allow_scaling=allow_scaling,
            )

            P_all = emb_cents[micron_cols].values
            P_trans = s * (R @ P_all.T).T + t

            sub_df = pd.DataFrame(P_trans, columns=micron_cols)
            sub_df[cell_col] = list(emb_cents.index)
            aligned_snapshots.append(sub_df)

        if not aligned_snapshots:
            break

        all_aligned = pd.concat(aligned_snapshots, axis=0, ignore_index=True)
        M_mean = all_aligned.groupby(cell_col)[micron_cols].mean()

        # Scale normalization back to empirical population centroid radius
        mean_pts = M_mean[micron_cols].values
        mean_centered = mean_pts - np.mean(mean_pts, axis=0)
        curr_rms_size = np.sqrt(np.mean(np.sum(mean_centered ** 2, axis=1)))
        scale_correction = target_rms_size / (curr_rms_size + 1e-10)

        M_mean_scaled = mean_centered * scale_correction + np.mean(mean_pts, axis=0)
        M_mean_df = pd.DataFrame(M_mean_scaled, index=M_mean.index, columns=micron_cols)

        # Re-anchor orientation to canonical seed frame
        common_seed = sorted(list(set(M_mean_df.index).intersection(set(M_seed.index))))
        _, R_fix, t_fix, _ = compute_ransac_similarity_pose(
            P=M_mean_df.loc[common_seed, micron_cols].values,
            Q=M_seed.loc[common_seed, micron_cols].values,
            max_inlier_dist_um=max_inlier_dist_um,
            n_iterations=n_ransac_iter,
            allow_scaling=False,
        )

        M_curr = pd.DataFrame(
            (R_fix @ M_mean_df[micron_cols].values.T).T + t_fix,
            index=M_mean_df.index,
            columns=micron_cols,
        )

    return M_curr, seed_embryo_id
