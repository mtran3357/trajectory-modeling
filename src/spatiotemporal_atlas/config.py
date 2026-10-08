"""Configuration dataclasses for the Spatiotemporal Atlas framework."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ColumnConfig:
    """Column names and dimensional scaling definitions."""
    raw_spatial_cols: tuple[str, str, str] = ("x", "y", "z")
    micron_cols: tuple[str, str, str] = ("x_um", "y_um", "z_um")
    aligned_cols: tuple[str, str, str] = ("x_aligned_um", "y_aligned_um", "z_aligned_um")
    voxel_size_xyz: tuple[float, float, float] = (0.09, 0.09, 1.0)
    time_col: str = "time"
    embryo_col: str = "series"
    cell_col: str = "cell"


@dataclass(frozen=True)
class SpatialConfig:
    """Settings for 3D RANSAC / Umeyama rigid-similarity registration and GPA."""
    max_inlier_dist_um: float = 4.0
    n_ransac_iter: int = 500
    sample_size: int = 4
    allow_scaling: bool = True
    n_procrustes_iter: int = 4


@dataclass(frozen=True)
class TemporalConfig:
    """Settings for 1D RANSAC affine temporal pacing and registration."""
    max_inlier_dist_canon: float = 5.0
    n_ransac_iter: int = 200
    sample_size: int = 2
    k_bounds: tuple[float, float] = (0.3, 3.0)
    min_anchor_cells: int = 3


@dataclass(frozen=True)
class TrajectoryConfig:
    """Settings for SRVF curve alignment and analytical GP trajectory modeling."""
    model_type: str = "joint_gp"
    grid_points: int = 40
    min_observations: int = 3
    min_train_embryos: int = 2
    length_scale: float = 0.3
    noise_level: float = 1.0
    length_scale_bounds: tuple[float, float] = (0.05, 3.0)
    noise_level_bounds: tuple[float, float] = (1e-4, 1e2)
    c_bounds: tuple[float, float] = (1e-3, 1e3)
    n_restarts_optimizer: int = 10
    random_state: int = 42
    n_dense_samples: int = 100
    optimizer: str | None = None
    local_trajectory_alignment: bool = True
    warping_lambda: float = 1.0
    warping_slope_bounds: tuple[float, float] = (0.5, 2.0)
    tau_percentile_cutoff: float = 95.0


@dataclass(frozen=True)
class AtlasBuildConfig:
    """Settings for Stage 1 WT reference atlas construction and null assembly."""
    columns: ColumnConfig = field(default_factory=ColumnConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    n_null_splits: int = 5
    n_jobs: int = 4
    random_state: int = 42


@dataclass(frozen=True)
class InferenceConfig:
    """Settings for Stage 2 query embryo inference and scoring."""
    alpha: float = 0.05
    min_observations: int = 3
    max_inlier_dist_canon: float | None = None
    n_jobs: int = 1
