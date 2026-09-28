"""Construction of canonical temporal atlas across training embryos."""

import numpy as np
import pandas as pd
from .ransac_time import compute_ransac_temporal_pose


def build_canonical_temporal_atlas(
    train_cycles: pd.DataFrame,
    cell_col: str = "cell",
    embryo_col: str = "series",
    min_anchor_cells: int = 3,
    max_inlier_dist_canon: float = 5.0,
    n_ransac_iter: int = 200,
) -> tuple[pd.DataFrame, dict, dict]:
    """Fits 1D RANSAC affine temporal atlas across training cohort.
    
    Parameters
    ----------
    train_cycles : pd.DataFrame
        Empirical lifespans across training embryos.
    cell_col : str, default='cell'
        Blastomere column name.
    embryo_col : str, default='series'
        Embryo identifier column name.
    min_anchor_cells : int, default=3
        Minimum anchor cells required to fit embryo pacing.
    max_inlier_dist_canon : float, default=5.0
        Cutoff in canonical minutes for RANSAC inlier consensus.
    n_ransac_iter : int, default=200
        RANSAC iteration count.
        
    Returns
    -------
    canon_train_df : pd.DataFrame
        Per-embryo normalized canonical timings.
    temporal_atlas : dict
        Mapping from cell_name to canonical lifespan and midpoint distributions.
    seeds : dict
        Seed reference values for initialization.
    """
    df_tr = train_cycles.copy()
    if df_tr.empty:
        return pd.DataFrame(), {}, {}

    df_tr[cell_col] = df_tr[cell_col].astype(str).str.strip()

    seed_log = df_tr.groupby(cell_col)["log_dur"].median().to_dict()
    seed_mid = df_tr.groupby(cell_col)["t_mid"].median().to_dict()

    records = []
    for eid, grp in df_tr.groupby(embryo_col):
        valid_seed_cells = [c for c in grp[cell_col] if c in seed_mid]
        if len(valid_seed_cells) < min_anchor_cells:
            continue

        sub = grp.set_index(cell_col).loc[valid_seed_cells]
        t_mid_obs = sub["t_mid"].values.astype(float)
        t_mid_ref = np.array([seed_mid[c] for c in valid_seed_cells], dtype=float)

        k_e, dt0_e, _ = compute_ransac_temporal_pose(
            t_mid_obs=t_mid_obs,
            t_mid_ref=t_mid_ref,
            max_inlier_dist_canon=max_inlier_dist_canon,
            n_iterations=n_ransac_iter,
        )

        for _, row in grp.iterrows():
            tb_c = (row["t_birth"] - dt0_e) / k_e
            td_c = (row["t_divide"] - dt0_e) / k_e
            dur_c = td_c - tb_c
            records.append({
                embryo_col: eid,
                cell_col: row[cell_col],
                "canon_birth": tb_c,
                "canon_divide": td_c,
                "canon_duration": dur_c,
                "canon_mid": 0.5 * (tb_c + td_c),
                "canon_log_dur": np.log(dur_c),
                "k_e": k_e,
                "dt0_e": dt0_e,
            })

    if not records:
        return pd.DataFrame(), {}, {}

    canon_train_df = pd.DataFrame(records)
    temporal_atlas = {}
    for c, grp in canon_train_df.groupby(cell_col):
        l_dur = grp["canon_log_dur"]
        p_dur = grp["canon_duration"]
        b_time = grp["canon_birth"]
        m_time = grp["canon_mid"]

        temporal_atlas[c] = {
            "mu_log": float(l_dur.mean()),
            "std_log": float(l_dur.std(ddof=1)) if len(l_dur) > 1 else 0.04,
            "mu_phys": float(p_dur.mean()),
            "var_phys": float(p_dur.var(ddof=1)) if len(p_dur) > 1 else 1.0,
            "mu_birth": float(b_time.mean()),
            "var_birth": float(b_time.var(ddof=1)) if len(b_time) > 1 else 0.5,
            "mu_mid": float(m_time.mean()),
            "n_replicates": len(grp),
        }

    return canon_train_df, temporal_atlas, {"seed_log": seed_log, "seed_mid": seed_mid}
