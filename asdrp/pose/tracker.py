"""
Pose tracking and temporal smoothing.

This module provides functionality for tracking pose landmarks across multiple frames
and applying temporal smoothing to reduce jitter and noise.
"""

import logging
from collections import deque
from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter1d

from .base import PoseLandmarks

logger = logging.getLogger(__name__)


class PoseTracker:
    """
    Multi-frame pose tracker with temporal smoothing.

    This class maintains a history of pose detections and applies temporal
    smoothing to reduce jitter and improve stability of landmark positions.

    Attributes:
        window_size: Number of frames to keep in history for smoothing.
        sigma: Standard deviation for Gaussian smoothing kernel.
    """

    def __init__(self, window_size: int = 5, sigma: float = 1.0) -> None:
        """
        Initialize the pose tracker.

        Args:
            window_size: Number of frames to keep in history (default: 5).
                        Larger values provide smoother results but increase latency.
            sigma: Standard deviation for Gaussian smoothing (default: 1.0).
                  Larger values provide more smoothing.

        Raises:
            ValueError: If window_size < 1 or sigma <= 0.
        """
        if window_size < 1:
            raise ValueError(f"window_size must be at least 1, got {window_size}")

        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        self.window_size = window_size
        self.sigma = sigma

        # History buffer using deque for efficient append/pop operations
        self._history: deque[PoseLandmarks] = deque(maxlen=window_size)

        logger.info(f"Initialized PoseTracker with window_size={window_size}, sigma={sigma}")

    def add_detection(self, landmarks: Optional[PoseLandmarks]) -> None:
        """
        Add a new pose detection to the tracking history.

        Args:
            landmarks: Detected pose landmarks, or None if no pose was detected.

        Note:
            If landmarks is None, the detection is skipped and not added to history.
            This prevents gaps in the temporal sequence used for smoothing.
        """
        if landmarks is None:
            logger.debug("Skipping None detection, not adding to history")
            return

        self._history.append(landmarks)
        logger.debug(
            f"Added detection for frame {landmarks.frame_number}, "
            f"history size: {len(self._history)}"
        )

    def get_smoothed_landmarks(
        self, mode: str = "gaussian"
    ) -> Optional[PoseLandmarks]:
        """
        Get temporally smoothed landmarks from the tracking history.

        Args:
            mode: Smoothing method to use. Options:
                 - "gaussian": Gaussian-weighted smoothing (default)
                 - "average": Simple moving average
                 - "median": Median filtering

        Returns:
            Smoothed PoseLandmarks object using the most recent timestamp and frame number,
            or None if history is empty.

        Raises:
            ValueError: If mode is not supported.
        """
        if not self._history:
            logger.debug("No detections in history, returning None")
            return None

        if mode not in ("gaussian", "average", "median"):
            raise ValueError(f"Unsupported smoothing mode: {mode}. Use 'gaussian', 'average', or 'median'")

        # If we only have one frame, return it as-is
        if len(self._history) == 1:
            logger.debug("Only one frame in history, returning without smoothing")
            return self._history[0]

        try:
            # Stack landmarks from history into arrays
            # Shape: (window_size, num_landmarks, 3)
            landmarks_stack = np.stack([det.landmarks for det in self._history], axis=0)
            world_landmarks_stack = np.stack(
                [det.world_landmarks for det in self._history], axis=0
            )
            visibility_stack = np.stack([det.visibility for det in self._history], axis=0)

            # Apply smoothing based on mode
            if mode == "gaussian":
                smoothed_landmarks = self._gaussian_smooth(landmarks_stack)
                smoothed_world = self._gaussian_smooth(world_landmarks_stack)
                smoothed_visibility = self._gaussian_smooth(visibility_stack)
            elif mode == "average":
                smoothed_landmarks = np.mean(landmarks_stack, axis=0)
                smoothed_world = np.mean(world_landmarks_stack, axis=0)
                smoothed_visibility = np.mean(visibility_stack, axis=0)
            else:  # median
                smoothed_landmarks = np.median(landmarks_stack, axis=0)
                smoothed_world = np.median(world_landmarks_stack, axis=0)
                smoothed_visibility = np.median(visibility_stack, axis=0)

            # Use the most recent frame's metadata
            latest = self._history[-1]

            logger.debug(
                f"Applied {mode} smoothing over {len(self._history)} frames "
                f"for frame {latest.frame_number}"
            )

            return PoseLandmarks(
                landmarks=smoothed_landmarks.astype(np.float32),
                visibility=smoothed_visibility.astype(np.float32),
                world_landmarks=smoothed_world.astype(np.float32),
                timestamp=latest.timestamp,
                frame_number=latest.frame_number,
            )

        except Exception as e:
            logger.error(f"Smoothing failed: {e}")
            # Fall back to returning the most recent detection
            return self._history[-1]

    def _gaussian_smooth(self, data: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Apply Gaussian smoothing along the temporal axis.

        Args:
            data: Input array with shape (window_size, ...).

        Returns:
            Smoothed array with the same shape, returning the last (most recent) frame.
        """
        # Apply Gaussian filter along axis 0 (time)
        # We need at least 3 points for Gaussian filtering
        if data.shape[0] < 3:
            return data[-1]

        smoothed = gaussian_filter1d(data, sigma=self.sigma, axis=0, mode="nearest")
        # Return the most recent frame after smoothing
        return smoothed[-1]

    def get_raw_latest(self) -> Optional[PoseLandmarks]:
        """
        Get the most recent raw (unsmoothed) detection.

        Returns:
            Most recent PoseLandmarks object, or None if history is empty.
        """
        if not self._history:
            return None
        return self._history[-1]

    def clear(self) -> None:
        """Clear all tracking history."""
        self._history.clear()
        logger.debug("Cleared tracking history")

    @property
    def history_size(self) -> int:
        """Return the current number of frames in the tracking history."""
        return len(self._history)

    @property
    def is_tracking(self) -> bool:
        """Return True if there are any detections in the tracking history."""
        return len(self._history) > 0

    def get_history(self) -> list[PoseLandmarks]:
        """
        Get a copy of the full tracking history.

        Returns:
            List of PoseLandmarks in chronological order (oldest to newest).
        """
        return list(self._history)

    def interpolate_missing(self, max_gap: int = 3) -> Optional[PoseLandmarks]:
        """
        Interpolate landmarks for missing frames using linear interpolation.

        This method is useful when a pose is temporarily not detected but we want
        to estimate its position based on the trajectory before and after the gap.

        Args:
            max_gap: Maximum number of frames to interpolate across (default: 3).
                    If gap is larger, returns None.

        Returns:
            Interpolated PoseLandmarks object, or None if interpolation is not possible.

        Note:
            This method requires at least 2 frames in history and is intended for
            filling small gaps in detection. It should not be used as the primary
            tracking mechanism.
        """
        if len(self._history) < 2:
            logger.debug("Not enough history for interpolation")
            return None

        # Get the two most recent detections
        prev = self._history[-2]
        curr = self._history[-1]

        # Check if gap is within interpolation limit
        frame_gap = curr.frame_number - prev.frame_number
        if frame_gap > max_gap + 1:
            logger.debug(f"Gap too large for interpolation: {frame_gap} frames")
            return None

        # Simple linear interpolation for the next frame
        alpha = 1.0 / frame_gap  # Weight for extrapolation

        interpolated_landmarks = curr.landmarks + alpha * (curr.landmarks - prev.landmarks)
        interpolated_world = curr.world_landmarks + alpha * (
            curr.world_landmarks - prev.world_landmarks
        )
        interpolated_visibility = np.minimum(curr.visibility, prev.visibility)

        logger.debug(f"Interpolated landmarks across {frame_gap} frame gap")

        return PoseLandmarks(
            landmarks=interpolated_landmarks.astype(np.float32),
            visibility=interpolated_visibility.astype(np.float32),
            world_landmarks=interpolated_world.astype(np.float32),
            timestamp=curr.timestamp + (curr.timestamp - prev.timestamp) * alpha,
            frame_number=curr.frame_number + 1,
        )
