"""
Pose estimation module for running gait analysis.

This module provides pose estimation capabilities using MediaPipe Pose Landmarker,
including landmark detection, tracking, and processing utilities.

Key Components:
    - BasePoseEstimator: Abstract base class for pose estimators
    - MediaPipePoseEstimator: MediaPipe-based implementation
    - PoseLandmarks: Data structure for pose detection results
    - PoseLandmarkIndex: Enum of MediaPipe landmark indices
    - PoseTracker: Multi-frame tracking with temporal smoothing
    - LandmarkProcessor: Utility functions for landmark analysis

Example Usage:
    >>> from asdrp.pose import MediaPipePoseEstimator, PoseTracker, PoseLandmarkIndex
    >>>
    >>> # Initialize estimator
    >>> estimator = MediaPipePoseEstimator(
    ...     model_path="pose_landmarker.task",
    ...     min_detection_confidence=0.5
    ... )
    >>>
    >>> # Process a frame
    >>> landmarks = estimator.estimate(image)
    >>>
    >>> # Track over multiple frames
    >>> tracker = PoseTracker(window_size=5)
    >>> tracker.add_detection(landmarks)
    >>> smoothed = tracker.get_smoothed_landmarks()
    >>>
    >>> # Analyze landmarks
    >>> from asdrp.pose import LandmarkProcessor
    >>> knee_angle = LandmarkProcessor.get_joint_angle(
    ...     landmarks,
    ...     PoseLandmarkIndex.LEFT_HIP,
    ...     PoseLandmarkIndex.LEFT_KNEE,
    ...     PoseLandmarkIndex.LEFT_ANKLE
    ... )
"""

from .base import BasePoseEstimator, PoseLandmarkIndex, PoseLandmarks
from .estimator import MediaPipePoseEstimator
from .landmarker import LandmarkProcessor
from .tracker import PoseTracker

__all__ = [
    # Base classes and data structures
    "PoseLandmarks",
    "BasePoseEstimator",
    "PoseLandmarkIndex",
    # Implementations
    "MediaPipePoseEstimator",
    "PoseTracker",
    # Utilities
    "LandmarkProcessor",
]

# Version information
__version__ = "0.1.0"
