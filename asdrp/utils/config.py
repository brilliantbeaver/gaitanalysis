"""Configuration management for gait analysis pipeline.

This module provides dataclasses and utilities for managing configuration
parameters across the gait analysis pipeline, ensuring type safety and
validation of settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Literal
import json


@dataclass
class VideoConfig:
    """Configuration for video processing.

    Attributes:
        input_path: Path to input video file
        output_path: Optional path for output video/results
        fps: Target frames per second (None = use source fps)
        start_time: Start time in seconds (None = from beginning)
        end_time: End time in seconds (None = to end)
        max_frames: Maximum number of frames to process (None = all)
        scale_factor: Scale factor for resizing video (1.0 = no scaling)
    """

    input_path: Path
    output_path: Optional[Path] = None
    fps: Optional[float] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    max_frames: Optional[int] = None
    scale_factor: float = 1.0

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.input_path = Path(self.input_path)
        if self.output_path is not None:
            self.output_path = Path(self.output_path)

        if self.scale_factor <= 0:
            raise ValueError(f"scale_factor must be positive, got {self.scale_factor}")

        if self.start_time is not None and self.start_time < 0:
            raise ValueError(f"start_time must be non-negative, got {self.start_time}")

        if self.end_time is not None and self.end_time < 0:
            raise ValueError(f"end_time must be non-negative, got {self.end_time}")

        if (
            self.start_time is not None
            and self.end_time is not None
            and self.end_time <= self.start_time
        ):
            raise ValueError(
                f"end_time ({self.end_time}) must be greater than start_time ({self.start_time})"
            )


@dataclass
class PoseEstimationConfig:
    """Configuration for pose estimation.

    Attributes:
        model_path: Path to MediaPipe pose model file
        min_detection_confidence: Minimum confidence for detection [0, 1]
        min_tracking_confidence: Minimum confidence for tracking [0, 1]
        running_mode: MediaPipe running mode ('IMAGE' or 'VIDEO')
        smooth_landmarks: Whether to apply smoothing to detected landmarks
        smoothing_window: Window size for landmark smoothing
    """

    model_path: Path
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    running_mode: Literal["IMAGE", "VIDEO"] = "VIDEO"
    smooth_landmarks: bool = True
    smoothing_window: int = 5

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.model_path = Path(self.model_path)

        if not 0.0 <= self.min_detection_confidence <= 1.0:
            raise ValueError(
                f"min_detection_confidence must be in [0, 1], got {self.min_detection_confidence}"
            )

        if not 0.0 <= self.min_tracking_confidence <= 1.0:
            raise ValueError(
                f"min_tracking_confidence must be in [0, 1], got {self.min_tracking_confidence}"
            )

        if self.running_mode not in ("IMAGE", "VIDEO"):
            raise ValueError(
                f"running_mode must be 'IMAGE' or 'VIDEO', got {self.running_mode}"
            )

        if self.smoothing_window < 1:
            raise ValueError(
                f"smoothing_window must be at least 1, got {self.smoothing_window}"
            )


@dataclass
class GaitAnalysisConfig:
    """Configuration for gait analysis.

    Attributes:
        fps: Video frame rate for temporal calculations
        calculate_cadence: Whether to calculate cadence metrics
        calculate_stride: Whether to calculate stride metrics
        calculate_symmetry: Whether to calculate symmetry metrics
        calculate_kinematics: Whether to calculate kinematic angles
        min_visibility: Minimum landmark visibility threshold [0, 1]
        stride_detection_method: Method for stride detection ('peak', 'threshold')
    """

    fps: float = 30.0
    calculate_cadence: bool = True
    calculate_stride: bool = True
    calculate_symmetry: bool = True
    calculate_kinematics: bool = True
    min_visibility: float = 0.5
    stride_detection_method: Literal["peak", "threshold"] = "peak"

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")

        if not 0.0 <= self.min_visibility <= 1.0:
            raise ValueError(
                f"min_visibility must be in [0, 1], got {self.min_visibility}"
            )


@dataclass
class VisualizationConfig:
    """Configuration for visualization and output.

    Attributes:
        create_overlay_video: Whether to create video with pose overlay
        create_plots: Whether to generate analysis plots
        create_report: Whether to generate HTML report
        plot_cadence: Whether to plot cadence over time
        plot_stride: Whether to plot stride metrics
        plot_angles: Whether to plot joint angles
        plot_symmetry: Whether to plot symmetry metrics
        output_format: Format for plots ('png', 'pdf', 'svg')
        dpi: Resolution for plot output
        video_codec: Codec for output video ('mp4v', 'avc1', 'h264')
    """

    create_overlay_video: bool = True
    create_plots: bool = True
    create_report: bool = True
    plot_cadence: bool = True
    plot_stride: bool = True
    plot_angles: bool = True
    plot_symmetry: bool = True
    output_format: Literal["png", "pdf", "svg"] = "png"
    dpi: int = 150
    video_codec: str = "mp4v"

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.dpi < 50:
            raise ValueError(f"dpi must be at least 50, got {self.dpi}")


@dataclass
class PipelineConfig:
    """Master configuration for the entire gait analysis pipeline.

    Attributes:
        video: Video processing configuration
        pose: Pose estimation configuration
        analysis: Gait analysis configuration
        visualization: Visualization configuration
        project_name: Name of the analysis project
        output_directory: Base directory for all outputs
        verbose: Whether to print detailed progress information
    """

    video: VideoConfig
    pose: PoseEstimationConfig
    analysis: GaitAnalysisConfig
    visualization: VisualizationConfig
    project_name: str = "Gait Analysis"
    output_directory: Path = Path("./output")
    verbose: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.output_directory = Path(self.output_directory)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PipelineConfig":
        """Create configuration from dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters

        Returns:
            PipelineConfig instance

        Example:
            >>> config_dict = {
            ...     'video': {'input_path': 'video.mp4'},
            ...     'pose': {'model_path': 'model.task'},
            ...     'analysis': {'fps': 30.0},
            ...     'visualization': {}
            ... }
            >>> config = PipelineConfig.from_dict(config_dict)
        """
        video_config = VideoConfig(**config_dict.get("video", {}))
        pose_config = PoseEstimationConfig(**config_dict.get("pose", {}))
        analysis_config = GaitAnalysisConfig(**config_dict.get("analysis", {}))
        viz_config = VisualizationConfig(**config_dict.get("visualization", {}))

        return cls(
            video=video_config,
            pose=pose_config,
            analysis=analysis_config,
            visualization=viz_config,
            project_name=config_dict.get("project_name", "Gait Analysis"),
            output_directory=Path(config_dict.get("output_directory", "./output")),
            verbose=config_dict.get("verbose", True),
        )

    @classmethod
    def from_json(cls, json_path: Path) -> "PipelineConfig":
        """Load configuration from JSON file.

        Args:
            json_path: Path to JSON configuration file

        Returns:
            PipelineConfig instance

        Raises:
            FileNotFoundError: If JSON file doesn't exist
            json.JSONDecodeError: If JSON is invalid
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {json_path}")

        with open(json_path, "r") as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "video": {
                "input_path": str(self.video.input_path),
                "output_path": str(self.video.output_path) if self.video.output_path else None,
                "fps": self.video.fps,
                "start_time": self.video.start_time,
                "end_time": self.video.end_time,
                "max_frames": self.video.max_frames,
                "scale_factor": self.video.scale_factor,
            },
            "pose": {
                "model_path": str(self.pose.model_path),
                "min_detection_confidence": self.pose.min_detection_confidence,
                "min_tracking_confidence": self.pose.min_tracking_confidence,
                "running_mode": self.pose.running_mode,
                "smooth_landmarks": self.pose.smooth_landmarks,
                "smoothing_window": self.pose.smoothing_window,
            },
            "analysis": {
                "fps": self.analysis.fps,
                "calculate_cadence": self.analysis.calculate_cadence,
                "calculate_stride": self.analysis.calculate_stride,
                "calculate_symmetry": self.analysis.calculate_symmetry,
                "calculate_kinematics": self.analysis.calculate_kinematics,
                "min_visibility": self.analysis.min_visibility,
                "stride_detection_method": self.analysis.stride_detection_method,
            },
            "visualization": {
                "create_overlay_video": self.visualization.create_overlay_video,
                "create_plots": self.visualization.create_plots,
                "create_report": self.visualization.create_report,
                "plot_cadence": self.visualization.plot_cadence,
                "plot_stride": self.visualization.plot_stride,
                "plot_angles": self.visualization.plot_angles,
                "plot_symmetry": self.visualization.plot_symmetry,
                "output_format": self.visualization.output_format,
                "dpi": self.visualization.dpi,
                "video_codec": self.visualization.video_codec,
            },
            "project_name": self.project_name,
            "output_directory": str(self.output_directory),
            "verbose": self.verbose,
        }

    def to_json(self, json_path: Path, indent: int = 2) -> None:
        """Save configuration to JSON file.

        Args:
            json_path: Path where to save JSON file
            indent: Number of spaces for indentation
        """
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=indent)

    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"PipelineConfig(\n"
            f"  project_name='{self.project_name}',\n"
            f"  video.input_path={self.video.input_path},\n"
            f"  output_directory={self.output_directory},\n"
            f"  analysis.fps={self.analysis.fps},\n"
            f"  verbose={self.verbose}\n"
            f")"
        )


def create_default_config(
    video_path: Path,
    model_path: Path,
    output_directory: Optional[Path] = None,
) -> PipelineConfig:
    """Create a default pipeline configuration with sensible defaults.

    Args:
        video_path: Path to input video file
        model_path: Path to MediaPipe pose model
        output_directory: Optional output directory (default: './output')

    Returns:
        PipelineConfig with default settings

    Example:
        >>> config = create_default_config(
        ...     video_path=Path("running_video.mp4"),
        ...     model_path=Path("pose_landmarker.task")
        ... )
    """
    video_config = VideoConfig(input_path=video_path)

    pose_config = PoseEstimationConfig(
        model_path=model_path,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode="VIDEO",
        smooth_landmarks=True,
        smoothing_window=5,
    )

    analysis_config = GaitAnalysisConfig(
        fps=30.0,
        calculate_cadence=True,
        calculate_stride=True,
        calculate_symmetry=True,
        calculate_kinematics=True,
        min_visibility=0.5,
        stride_detection_method="peak",
    )

    viz_config = VisualizationConfig(
        create_overlay_video=True,
        create_plots=True,
        create_report=True,
        plot_cadence=True,
        plot_stride=True,
        plot_angles=True,
        plot_symmetry=True,
        output_format="png",
        dpi=150,
        video_codec="mp4v",
    )

    output_dir = output_directory if output_directory else Path("./output")

    return PipelineConfig(
        video=video_config,
        pose=pose_config,
        analysis=analysis_config,
        visualization=viz_config,
        project_name="Running Gait Analysis",
        output_directory=output_dir,
        verbose=True,
    )
