"""ASDRP - Advanced Sports Data Research Platform for Running Gait Analysis.

This package provides a comprehensive toolkit for analyzing running gait patterns
using computer vision and biomechanics principles. It includes modules for video
processing, pose estimation, gait metrics calculation, and visualization.

Main Components:
    - pipeline: Orchestrated workflow for complete gait analysis
    - video: Video input/output handling
    - pose: Pose estimation using MediaPipe
    - analysis: Gait metrics and pattern analysis
    - visualization: Plotting and report generation
    - utils: Utility functions and configuration management

Quick Start:
    >>> from asdrp import GaitAnalysisPipeline
    >>> from asdrp.utils import create_default_config
    >>> from pathlib import Path
    >>>
    >>> # Create configuration
    >>> config = create_default_config(
    ...     video_path=Path("running_video.mp4"),
    ...     model_path=Path("pose_landmarker.task"),
    ...     output_directory=Path("./results")
    ... )
    >>>
    >>> # Run complete pipeline
    >>> pipeline = GaitAnalysisPipeline(config)
    >>> results = pipeline.run()
    >>>
    >>> # Access results
    >>> metrics = results['metrics']
    >>> print(f"Cadence: {metrics.cadence:.1f} steps/min")
    >>> print(f"Stride time: {metrics.stride_time:.3f} seconds")
    >>> print(f"Symmetry index: {metrics.symmetry_index:.3f}")

Advanced Usage:
    >>> # Custom configuration
    >>> from asdrp.utils import PipelineConfig, VideoConfig, PoseEstimationConfig
    >>>
    >>> config = PipelineConfig(
    ...     video=VideoConfig(
    ...         input_path=Path("video.mp4"),
    ...         start_time=5.0,
    ...         end_time=30.0
    ...     ),
    ...     pose=PoseEstimationConfig(
    ...         model_path=Path("model.task"),
    ...         min_detection_confidence=0.7
    ...     ),
    ...     # ... other config
    ... )
    >>>
    >>> # Use as context manager for automatic cleanup
    >>> with GaitAnalysisPipeline(config) as pipeline:
    ...     results = pipeline.run()
"""

# Pipeline
from .pipeline import GaitAnalysisPipeline

# Configuration
from .utils import (
    PipelineConfig,
    VideoConfig,
    PoseEstimationConfig,
    GaitAnalysisConfig,
    VisualizationConfig,
    create_default_config,
)

# Core data structures
from .analysis import (
    GaitMetrics,
    GaitEvent,
    GaitEventType,
    Foot,
)

from .pose import (
    PoseLandmarks,
    PoseLandmarkIndex,
)

# Analyzers (for advanced users)
from .analysis import (
    GaitAnalyzer,
    StrideAnalyzer,
    CadenceAnalyzer,
    SymmetryAnalyzer,
    BaseMetricCalculator,
)

# Video processing (for advanced users)
from .video import (
    VideoFileReader,
    VideoWriter,
    FrameData,
)

# Pose estimation (for advanced users)
from .pose import (
    MediaPipePoseEstimator,
    PoseTracker,
    LandmarkProcessor,
)

# Visualization (for advanced users)
from .visualization import (
    PoseOverlay,
    MetricsPlotter,
    ReportGenerator,
)

# Utilities (for advanced users)
from .utils import (
    calculate_angle,
    euclidean_distance,
    gaussian_smooth,
    savitzky_golay,
    smooth_trajectory,
)

__all__ = [
    # Main pipeline
    "GaitAnalysisPipeline",
    # Configuration
    "PipelineConfig",
    "VideoConfig",
    "PoseEstimationConfig",
    "GaitAnalysisConfig",
    "VisualizationConfig",
    "create_default_config",
    # Core data structures
    "GaitMetrics",
    "GaitEvent",
    "GaitEventType",
    "Foot",
    "PoseLandmarks",
    "PoseLandmarkIndex",
    # Analyzers
    "GaitAnalyzer",
    "StrideAnalyzer",
    "CadenceAnalyzer",
    "SymmetryAnalyzer",
    "BaseMetricCalculator",
    # Video
    "VideoFileReader",
    "VideoWriter",
    "FrameData",
    # Pose
    "MediaPipePoseEstimator",
    "PoseTracker",
    "LandmarkProcessor",
    # Visualization
    "PoseOverlay",
    "MetricsPlotter",
    "ReportGenerator",
    # Utilities
    "calculate_angle",
    "euclidean_distance",
    "gaussian_smooth",
    "savitzky_golay",
    "smooth_trajectory",
]

# Version information
__version__ = "0.1.0"
__author__ = "ASDRP Team"
__description__ = "Advanced Sports Data Research Platform for Running Gait Analysis"

# Module-level documentation
__doc__ = """
ASDRP - Running Gait Analysis Platform
=======================================

A comprehensive toolkit for analyzing running biomechanics from video data.

Installation:
    pip install -e .

Dependencies:
    - opencv-python
    - numpy
    - scipy
    - matplotlib
    - mediapipe
    - pandas

Key Features:
    1. Video Processing: Efficient reading and writing of video files
    2. Pose Estimation: MediaPipe-based pose detection and tracking
    3. Gait Analysis: Comprehensive biomechanics metrics (cadence, stride, symmetry)
    4. Visualization: Pose overlays, plots, and HTML reports
    5. Configuration: Type-safe configuration management

Documentation:
    See individual module docstrings for detailed API documentation.

    Main modules:
    - asdrp.pipeline: Main workflow orchestration
    - asdrp.video: Video I/O operations
    - asdrp.pose: Pose estimation and tracking
    - asdrp.analysis: Gait metrics calculation
    - asdrp.visualization: Plotting and reporting
    - asdrp.utils: Utilities and configuration

Example Workflow:
    1. Create configuration with video and model paths
    2. Initialize GaitAnalysisPipeline with configuration
    3. Run pipeline to process video, analyze gait, and generate outputs
    4. Access results including metrics, visualizations, and reports

For more examples, see the examples/ directory in the repository.
"""
