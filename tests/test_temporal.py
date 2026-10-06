"""Unit tests for 1D RANSAC temporal pacing and pose estimation."""

import numpy as np
from spatiotemporal_atlas.temporal import compute_ransac_temporal_pose


def test_ransac_temporal_pose_clean():
    """Verify 1D RANSAC recovers known Ke and dt0 with no noise."""
    t_ref = np.linspace(10.0, 100.0, 10)
    ke_true = 1.2
    dt0_true = 5.0
    t_obs = ke_true * t_ref + dt0_true

    ke, dt0, inliers = compute_ransac_temporal_pose(
        t_mid_obs=t_obs,
        t_mid_ref=t_ref,
        max_inlier_dist_canon=1.0,
        n_iterations=100,
        random_state=42,
    )

    assert np.all(inliers)
    assert np.isclose(ke, ke_true, atol=1e-4)
    assert np.isclose(dt0, dt0_true, atol=1e-3)


def test_ransac_temporal_pose_with_outliers():
    """Verify 1D RANSAC ignores outliers in observed timings."""
    rng = np.random.default_rng(42)
    t_ref = np.linspace(10.0, 120.0, 15)
    ke_true = 0.95
    dt0_true = 12.0
    t_obs = ke_true * t_ref + dt0_true

    # Corrupt 3 cells with huge timing delays/skips
    t_obs[2] += 40.0
    t_obs[7] -= 35.0
    t_obs[12] += 50.0

    ke, dt0, inliers = compute_ransac_temporal_pose(
        t_mid_obs=t_obs,
        t_mid_ref=t_ref,
        max_inlier_dist_canon=5.0,
        n_iterations=200,
        random_state=42,
    )

    assert np.isclose(ke, ke_true, atol=0.05)
    assert np.isclose(dt0, dt0_true, atol=1.5)
    assert not inliers[2]
    assert not inliers[7]
    assert not inliers[12]


def test_register_embryo_to_temporal_atlas_records_ke():
    """Verify register_embryo_to_temporal_atlas scales duration by Ke and records k_e/dt0_e."""
    import pandas as pd
    from spatiotemporal_atlas.temporal import register_embryo_to_temporal_atlas

    cells = ["ABa", "ABp", "EMS", "P2"]
    ke_true = 1.25
    dt0_true = 10.0

    ref_atlas = {
        c: {"mu_mid": 20.0 + i * 15.0, "mu_phys": 25.0}
        for i, c in enumerate(cells)
    }

    # Synthesize cycles with Ke=1.25, dt0=10.0
    cycles_data = []
    for i, c in enumerate(cells):
        t_mid_ref = ref_atlas[c]["mu_mid"]
        dur_ref = ref_atlas[c]["mu_phys"]
        t_birth_ref = t_mid_ref - 0.5 * dur_ref
        t_divide_ref = t_mid_ref + 0.5 * dur_ref

        t_birth_obs = ke_true * t_birth_ref + dt0_true
        t_divide_obs = ke_true * t_divide_ref + dt0_true
        cycles_data.append({
            "cell": c,
            "t_birth": t_birth_obs,
            "t_divide": t_divide_obs,
            "duration": t_divide_obs - t_birth_obs,
            "t_mid": 0.5 * (t_birth_obs + t_divide_obs),
        })

    cycles_df = pd.DataFrame(cycles_data)
    canon_df, reg_meta = register_embryo_to_temporal_atlas(cycles_df, ref_atlas)

    assert "k_e" in canon_df.columns
    assert "dt0_e" in canon_df.columns
    assert np.isclose(reg_meta["k_test"], ke_true, atol=1e-3)
    assert np.isclose(canon_df["k_e"].iloc[0], ke_true, atol=1e-3)

    # Verify that canon_duration == duration / Ke == 25.0 canonical minutes
    for _, row in canon_df.iterrows():
        assert np.isclose(row["canon_duration"], 25.0, atol=1e-3)
        assert np.isclose(row["duration"] / row["k_e"], row["canon_duration"], atol=1e-4)
