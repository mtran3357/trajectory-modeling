"""Domain entities, data contracts, and result containers."""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SpatialPose:
    """3D similarity pose parameters: Q approx s * (R @ P) + t."""
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    inliers: np.ndarray
    inlier_ratio: float
    mean_inlier_res_um: float
    mean_all_res_um: float
    is_inlier_map: dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class TemporalPose:
    """1D affine developmental pacing parameters: t_mid_obs approx Ke * t_mid_ref + dt0."""
    ke: float
    dt0: float
    inliers: np.ndarray
    inlier_ratio: float
    n_anchor_cells: int
    n_temporal_inliers: int
    mean_abs_dur_err_pct: float
    median_abs_mid_err: float


@dataclass(frozen=True)
class CellTemporalStats:
    """Canonical temporal statistics for a single blastomere."""
    cell_name: str
    mu_log: float
    std_log: float
    mu_phys: float
    var_phys: float
    mu_birth: float
    var_birth: float
    mu_mid: float
    n_replicates: int


@dataclass
class TrajectoryRibbon:
    """Lightweight 100-point ribbon representation of a blastomere 3D trajectory."""
    cell_name: str
    s_dense: np.ndarray
    dense_pred: dict[str, dict[str, np.ndarray]]
    mu_com: np.ndarray
    inv_cov_com: np.ndarray
    mu_fr: float
    std_fr: float
    mu_srvf: np.ndarray
    time_grid: np.ndarray
    dense_mu_3d: np.ndarray | None = None
    dense_cov_3d: np.ndarray | None = None
    B_cov: np.ndarray | None = None
    mu_rot_deg: float = 0.0
    std_rot_deg: float = 1.0
    template_curve: np.ndarray | None = None

    def __getitem__(self, item: str):
        return getattr(self, item)

    def get(self, item: str, default=None):
        return getattr(self, item, default)

    def predict(self, s_obs: np.ndarray, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Fast 1D linear interpolation of predicted mean and standard deviation along ribbon."""
        pred_mu = np.zeros((len(s_obs), len(cols)), dtype=np.float32)
        pred_std = np.zeros((len(s_obs), len(cols)), dtype=np.float32)
        for i, c in enumerate(cols):
            pred_mu[:, i] = np.interp(s_obs, self.s_dense, self.dense_pred[c]["mu"])
            pred_std[:, i] = np.interp(s_obs, self.s_dense, self.dense_pred[c]["std"])
        return pred_mu, pred_std

    def to_dict(self) -> dict:
        """Converts to dictionary representation for backward compatibility."""
        d = {
            "cell_name": self.cell_name,
            "s_dense": self.s_dense,
            "dense_pred": self.dense_pred,
            "mu_com": self.mu_com,
            "inv_cov_com": self.inv_cov_com,
            "mu_fr": self.mu_fr,
            "std_fr": self.std_fr,
            "mu_srvf": self.mu_srvf,
            "time_grid": self.time_grid,
            "mu_rot_deg": self.mu_rot_deg,
            "std_rot_deg": self.std_rot_deg,
        }
        if self.dense_mu_3d is not None:
            d["dense_mu_3d"] = self.dense_mu_3d
        if self.dense_cov_3d is not None:
            d["dense_cov_3d"] = self.dense_cov_3d
        if self.B_cov is not None:
            d["B_cov"] = self.B_cov
        if self.template_curve is not None:
            d["template_curve"] = self.template_curve
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TrajectoryRibbon":
        """Instantiates TrajectoryRibbon from dictionary."""
        return cls(
            cell_name=d["cell_name"],
            s_dense=d["s_dense"],
            dense_pred=d["dense_pred"],
            mu_com=d["mu_com"],
            inv_cov_com=d["inv_cov_com"],
            mu_fr=float(d["mu_fr"]),
            std_fr=float(d["std_fr"]),
            mu_srvf=d["mu_srvf"],
            time_grid=d["time_grid"],
            dense_mu_3d=d.get("dense_mu_3d"),
            dense_cov_3d=d.get("dense_cov_3d"),
            B_cov=d.get("B_cov"),
            mu_rot_deg=float(d.get("mu_rot_deg", 0.0)),
            std_rot_deg=float(d.get("std_rot_deg", 1.0)),
            template_curve=d.get("template_curve"),
        )



@dataclass
class ReferenceAtlas:
    """Stage 1 complete WT reference atlas bundle."""
    spatial_template: pd.DataFrame
    seed_embryo_id: str
    temporal_atlas: dict[str, dict]
    cell_models: dict[str, TrajectoryRibbon | dict]
    oof_null_df: pd.DataFrame
    wt_spatial_reg_meta: dict[str, dict]
    config: dict

    def to_dict(self) -> dict:
        """Serializes bundle into dictionary compatible with prototype atlas_bundle."""
        return {
            "spatial_template": self.spatial_template,
            "seed_embryo_id": self.seed_embryo_id,
            "temporal_atlas": self.temporal_atlas,
            "cell_models": {
                k: (v.to_dict() if isinstance(v, TrajectoryRibbon) else v)
                for k, v in self.cell_models.items()
            },
            "oof_null_df": self.oof_null_df,
            "wt_spatial_reg_meta": self.wt_spatial_reg_meta,
            **self.config,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReferenceAtlas":
        """Reconstructs ReferenceAtlas from dictionary."""
        config_keys = [
            "voxel_size_xyz", "raw_spatial_cols", "aligned_cols", "micron_cols",
            "time_col", "embryo_col", "cell_col", "max_inlier_dist_um", "max_inlier_dist_canon"
        ]
        config = {k: d[k] for k in config_keys if k in d}
        cell_models = {}
        for k, v in d.get("cell_models", {}).items():
            if isinstance(v, dict):
                cell_models[k] = TrajectoryRibbon.from_dict(v)
            else:
                cell_models[k] = v

        return cls(
            spatial_template=d["spatial_template"],
            seed_embryo_id=d["seed_embryo_id"],
            temporal_atlas=d["temporal_atlas"],
            cell_models=cell_models,
            oof_null_df=d["oof_null_df"],
            wt_spatial_reg_meta=d.get("wt_spatial_reg_meta", {}),
            config=config,
        )
