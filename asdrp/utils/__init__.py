"""Utility modules for gait analysis.

This package provides reusable utilities for geometric calculations,
signal smoothing, and configuration management.
"""

from .geometry import (
    calculate_angle,
    calculate_angle_2d,
    calculate_signed_angle,
    euclidean_distance,
    euclidean_distance_2d,
    point_to_line_distance,
    calculate_velocity,
    calculate_speed,
    normalize_vector,
    rotation_matrix_2d,
    rotate_point_2d,
    calculate_centroid,
    calculate_bounding_box,
    scale_points,
    project_point_onto_line,
)

from .smoothing import (
    gaussian_smooth,
    moving_average,
    savitzky_golay,
    butterworth_filter,
    exponential_smoothing,
    median_filter,
    smooth_trajectory,
    adaptive_smooth,
    interpolate_missing_values,
)

from .config import (
    VideoConfig,
    PoseEstimationConfig,
    GaitAnalysisConfig,
    VisualizationConfig,
    PipelineConfig,
    create_default_config,
)

__all__ = [
    # Geometry
    "calculate_angle",
    "calculate_angle_2d",
    "calculate_signed_angle",
    "euclidean_distance",
    "euclidean_distance_2d",
    "point_to_line_distance",
    "calculate_velocity",
    "calculate_speed",
    "normalize_vector",
    "rotation_matrix_2d",
    "rotate_point_2d",
    "calculate_centroid",
    "calculate_bounding_box",
    "scale_points",
    "project_point_onto_line",
    # Smoothing
    "gaussian_smooth",
    "moving_average",
    "savitzky_golay",
    "butterworth_filter",
    "exponential_smoothing",
    "median_filter",
    "smooth_trajectory",
    "adaptive_smooth",
    "interpolate_missing_values",
    # Config
    "VideoConfig",
    "PoseEstimationConfig",
    "GaitAnalysisConfig",
    "VisualizationConfig",
    "PipelineConfig",
    "create_default_config",
]
