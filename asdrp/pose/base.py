"""
Base classes and data structures for pose estimation.

This module defines the core abstractions for pose estimation, including
data classes for landmarks and abstract base classes for pose estimators.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np
import numpy.typing as npt


@dataclass
class PoseLandmarks:
    """
    Container for pose landmark detection results.

    Attributes:
        landmarks: Array of shape (N, 3) containing (x, y, z) coordinates in image space.
                  x and y are normalized to [0, 1], z represents depth.
        visibility: Array of shape (N,) containing visibility scores [0, 1] for each landmark.
        world_landmarks: Array of shape (N, 3) containing (x, y, z) coordinates in world space
                        (meters relative to hip center).
        timestamp: Timestamp of the frame in milliseconds.
        frame_number: Sequential frame number in the video/stream.
    """

    landmarks: npt.NDArray[np.float32]
    visibility: npt.NDArray[np.float32]
    world_landmarks: npt.NDArray[np.float32]
    timestamp: float
    frame_number: int

    def __post_init__(self) -> None:
        """Validate landmark data after initialization."""
        if self.landmarks.shape[0] != self.visibility.shape[0]:
            raise ValueError(
                f"Landmarks ({self.landmarks.shape[0]}) and visibility "
                f"({self.visibility.shape[0]}) must have same length"
            )
        if self.landmarks.shape[0] != self.world_landmarks.shape[0]:
            raise ValueError(
                f"Landmarks ({self.landmarks.shape[0]}) and world_landmarks "
                f"({self.world_landmarks.shape[0]}) must have same length"
            )
        if self.landmarks.shape[1] != 3:
            raise ValueError(f"Landmarks must have 3 coordinates, got {self.landmarks.shape[1]}")
        if self.world_landmarks.shape[1] != 3:
            raise ValueError(
                f"World landmarks must have 3 coordinates, got {self.world_landmarks.shape[1]}"
            )

    @property
    def num_landmarks(self) -> int:
        """Return the number of detected landmarks."""
        return self.landmarks.shape[0]

    def get_landmark(self, index: int) -> tuple[float, float, float]:
        """
        Get image-space coordinates for a specific landmark.

        Args:
            index: Index of the landmark to retrieve.

        Returns:
            Tuple of (x, y, z) coordinates.

        Raises:
            IndexError: If index is out of bounds.
        """
        if index < 0 or index >= self.num_landmarks:
            raise IndexError(f"Landmark index {index} out of range [0, {self.num_landmarks})")
        return tuple(self.landmarks[index])  # type: ignore

    def get_world_landmark(self, index: int) -> tuple[float, float, float]:
        """
        Get world-space coordinates for a specific landmark.

        Args:
            index: Index of the landmark to retrieve.

        Returns:
            Tuple of (x, y, z) coordinates in meters.

        Raises:
            IndexError: If index is out of bounds.
        """
        if index < 0 or index >= self.num_landmarks:
            raise IndexError(f"Landmark index {index} out of range [0, {self.num_landmarks})")
        return tuple(self.world_landmarks[index])  # type: ignore

    def get_visibility(self, index: int) -> float:
        """
        Get visibility score for a specific landmark.

        Args:
            index: Index of the landmark to retrieve.

        Returns:
            Visibility score in range [0, 1].

        Raises:
            IndexError: If index is out of bounds.
        """
        if index < 0 or index >= self.num_landmarks:
            raise IndexError(f"Landmark index {index} out of range [0, {self.num_landmarks})")
        return float(self.visibility[index])


class PoseLandmarkIndex(IntEnum):
    """
    MediaPipe Pose landmark indices.

    These indices correspond to the 33 landmarks detected by MediaPipe Pose Landmarker.
    Reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
    """

    # Face
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10

    # Upper body
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22

    # Lower body
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


class BasePoseEstimator(ABC):
    """
    Abstract base class for pose estimation.

    This class defines the interface that all pose estimators must implement.
    Concrete implementations should handle specific pose estimation backends
    (e.g., MediaPipe, OpenPose, etc.).
    """

    @abstractmethod
    def estimate(
        self, image: npt.NDArray[np.uint8], timestamp: float = 0.0, frame_number: int = 0
    ) -> Optional[PoseLandmarks]:
        """
        Estimate pose landmarks from a single image.

        Args:
            image: Input image as numpy array in RGB format with shape (H, W, 3).
            timestamp: Timestamp of the frame in milliseconds (default: 0.0).
            frame_number: Sequential frame number (default: 0).

        Returns:
            PoseLandmarks object if pose is detected, None otherwise.

        Raises:
            ValueError: If image format is invalid.
            RuntimeError: If pose estimation fails.
        """
        pass

    @abstractmethod
    def estimate_batch(
        self,
        images: list[npt.NDArray[np.uint8]],
        timestamps: Optional[list[float]] = None,
        frame_numbers: Optional[list[int]] = None,
    ) -> list[Optional[PoseLandmarks]]:
        """
        Estimate pose landmarks from a batch of images.

        Args:
            images: List of input images in RGB format with shape (H, W, 3).
            timestamps: List of timestamps in milliseconds (default: sequential).
            frame_numbers: List of frame numbers (default: sequential starting from 0).

        Returns:
            List of PoseLandmarks objects. None entries indicate no pose detected.

        Raises:
            ValueError: If input lists have inconsistent lengths or invalid format.
            RuntimeError: If pose estimation fails.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Release resources used by the pose estimator.

        This method should be called when the estimator is no longer needed
        to free up memory and release any held resources.
        """
        pass

    def __enter__(self) -> "BasePoseEstimator":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: object) -> None:
        """Context manager exit with automatic resource cleanup."""
        self.close()
