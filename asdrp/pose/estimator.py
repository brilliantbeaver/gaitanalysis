"""
MediaPipe-based pose estimation implementation.

This module provides a concrete implementation of pose estimation using
Google's MediaPipe Pose Landmarker API.
"""

import logging
from pathlib import Path
from typing import Optional

import mediapipe as mp
import numpy as np
import numpy.typing as npt
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .base import BasePoseEstimator, PoseLandmarks

logger = logging.getLogger(__name__)


class MediaPipePoseEstimator(BasePoseEstimator):
    """
    Pose estimator using MediaPipe Pose Landmarker.

    This class wraps MediaPipe's Pose Landmarker API to provide pose estimation
    capabilities for images and video frames.

    Attributes:
        model_path: Path to the MediaPipe pose landmarker model file (.task).
        min_detection_confidence: Minimum confidence for pose detection [0, 1].
        min_tracking_confidence: Minimum confidence for pose tracking [0, 1].
    """

    def __init__(
        self,
        model_path: str | Path,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        running_mode: str = "IMAGE",
    ) -> None:
        """
        Initialize the MediaPipe pose estimator.

        Args:
            model_path: Path to the MediaPipe pose landmarker model file.
                       Download from: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
            min_detection_confidence: Minimum confidence score [0, 1] for pose detection
                                     to be considered successful (default: 0.5).
            min_tracking_confidence: Minimum confidence score [0, 1] for pose tracking
                                    to be considered successful (default: 0.5).
            running_mode: Mode for the estimator: "IMAGE" or "VIDEO" (default: "IMAGE").

        Raises:
            FileNotFoundError: If model file does not exist.
            ValueError: If confidence values are not in range [0, 1] or running_mode is invalid.
            RuntimeError: If MediaPipe initialization fails.
        """
        self.model_path = Path(model_path)
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Validate inputs
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        if not 0.0 <= min_detection_confidence <= 1.0:
            raise ValueError(
                f"min_detection_confidence must be in [0, 1], got {min_detection_confidence}"
            )

        if not 0.0 <= min_tracking_confidence <= 1.0:
            raise ValueError(
                f"min_tracking_confidence must be in [0, 1], got {min_tracking_confidence}"
            )

        if running_mode not in ("IMAGE", "VIDEO"):
            raise ValueError(f"running_mode must be 'IMAGE' or 'VIDEO', got {running_mode}")

        self.running_mode = running_mode

        # Initialize MediaPipe
        try:
            self._initialize_landmarker()
            logger.info(
                f"Initialized MediaPipePoseEstimator with model: {self.model_path}, "
                f"mode: {running_mode}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
            raise RuntimeError(f"MediaPipe initialization failed: {e}") from e

    def _initialize_landmarker(self) -> None:
        """Initialize the MediaPipe Pose Landmarker."""
        base_options = python.BaseOptions(model_asset_path=str(self.model_path))

        # Map string running mode to enum
        mode_map = {
            "IMAGE": vision.RunningMode.IMAGE,
            "VIDEO": vision.RunningMode.VIDEO,
        }

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mode_map[self.running_mode],
            min_pose_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            output_segmentation_masks=False,
        )

        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        logger.debug("MediaPipe Pose Landmarker initialized successfully")

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
        # Validate input image
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Image must have shape (H, W, 3), got {image.shape}")

        if image.dtype != np.uint8:
            raise ValueError(f"Image must be uint8, got {image.dtype}")

        try:
            # Convert to MediaPipe Image format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

            # Run pose detection
            if self.running_mode == "VIDEO":
                timestamp_ms = int(timestamp)
                detection_result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
            else:
                detection_result = self._landmarker.detect(mp_image)

            # Extract landmarks if pose is detected
            if not detection_result.pose_landmarks:
                logger.debug(f"No pose detected in frame {frame_number}")
                return None

            # MediaPipe returns a list of poses, take the first one
            pose = detection_result.pose_landmarks[0]
            world_pose = detection_result.pose_world_landmarks[0]

            # Convert to numpy arrays
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in pose], dtype=np.float32)

            visibility = np.array([lm.visibility for lm in pose], dtype=np.float32)

            world_landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in world_pose], dtype=np.float32
            )

            logger.debug(f"Detected pose with {len(landmarks)} landmarks in frame {frame_number}")

            return PoseLandmarks(
                landmarks=landmarks,
                visibility=visibility,
                world_landmarks=world_landmarks,
                timestamp=timestamp,
                frame_number=frame_number,
            )

        except Exception as e:
            logger.error(f"Pose estimation failed for frame {frame_number}: {e}")
            raise RuntimeError(f"Pose estimation failed: {e}") from e

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
        if not images:
            return []

        # Generate default timestamps and frame numbers if not provided
        if timestamps is None:
            timestamps = [float(i) for i in range(len(images))]

        if frame_numbers is None:
            frame_numbers = list(range(len(images)))

        # Validate input lengths
        if len(timestamps) != len(images):
            raise ValueError(
                f"Number of timestamps ({len(timestamps)}) must match "
                f"number of images ({len(images)})"
            )

        if len(frame_numbers) != len(images):
            raise ValueError(
                f"Number of frame_numbers ({len(frame_numbers)}) must match "
                f"number of images ({len(images)})"
            )

        # Process each image
        results: list[Optional[PoseLandmarks]] = []
        for img, ts, fn in zip(images, timestamps, frame_numbers):
            try:
                result = self.estimate(img, timestamp=ts, frame_number=fn)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to process frame {fn}: {e}")
                results.append(None)

        logger.info(
            f"Processed batch of {len(images)} images, "
            f"detected poses in {sum(1 for r in results if r is not None)} frames"
        )

        return results

    def close(self) -> None:
        """
        Release resources used by the pose estimator.

        This method should be called when the estimator is no longer needed
        to free up memory and release any held resources.
        """
        if hasattr(self, "_landmarker"):
            self._landmarker.close()
            logger.debug("MediaPipe Pose Landmarker closed")

    def __del__(self) -> None:
        """Destructor to ensure resources are released."""
        self.close()
