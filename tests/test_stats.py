"""Unit tests for FDR control and empirical calibration."""

import numpy as np
import pandas as pd
from spatiotemporal_atlas.stats import (
    bh_qvalues,
    calc_emp_pval,
    apply_empirical_calibration_to_inference,
)


def test_bh_qvalues_monotonicity():
    """Verify Benjamini-Hochberg q-values are monotonically sorted with p-values."""
    p_vals = np.array([0.001, 0.04, 0.02, 0.5, 0.9])
    q_vals = bh_qvalues(p_vals)

    assert len(q_vals) == len(p_vals)
    assert np.all(q_vals >= 0.0) and np.all(q_vals <= 1.0)
    # The smallest p-value should have the smallest q-value
    assert np.argmin(q_vals) == np.argmin(p_vals)
    assert q_vals[0] < q_vals[1]


def test_calc_emp_pval_limits():
    """Verify empirical p-value computation on known reference arrays."""
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # N=5
    # val = 6.0 > all ref: p = (0 + 1) / (5 + 1) = 1/6
    assert np.isclose(calc_emp_pval(6.0, ref, two_sided=False), 1.0 / 6.0)
    # val = 0.0 < all ref: p = (5 + 1) / (5 + 1) = 1.0
    assert np.isclose(calc_emp_pval(0.0, ref, two_sided=False), 1.0)
    # val = 3.0: 3, 4, 5 >= 3: count = 3: p = (3 + 1) / 6 = 4/6
    assert np.isclose(calc_emp_pval(3.0, ref, two_sided=False), 4.0 / 6.0)


def test_apply_empirical_calibration():
    """Verify empirical calibration and FDR hit calling across 5 modalities."""
    test_df = pd.DataFrame([{
        "cell": "ABa",
        "z_temp_shape": 4.5,
        "z_temp_shift": 0.2,
        "d_spat_shift": 5.0,
        "d_spat_shape": 0.5,
        "z_warp": 0.1,
    }])
    null_df = pd.DataFrame({
        "emp_temp_shape": [0.1, 0.5, 1.0, 1.2],
        "emp_temp_shift": [0.1, 0.2, 0.3, 0.4],
        "emp_spat_shift": [0.2, 0.4, 0.6, 0.8],
        "emp_spat_shape": [0.1, 0.2, 0.3, 0.4],
        "emp_warp": [0.1, 0.2, 0.3, 0.4],
        "null_embryo_id": ["E1", "E2", "E3", "E4"],
    })

    cal_df, meta = apply_empirical_calibration_to_inference(test_df, null_df, alpha=0.25)

    assert "pval_temp_shape" in cal_df.columns
    assert "qval_temp_shape" in cal_df.columns
    assert "hit_temp_shape" in cal_df.columns
    assert "n_outlier_modalities" in cal_df.columns
    # Outlier count should be > 0 because temp_shape and spat_shift are extreme
    assert cal_df["n_outlier_modalities"].iloc[0] > 0
    assert cal_df["is_any_outlier"].iloc[0] is True or cal_df["is_any_outlier"].iloc[0] == 1
