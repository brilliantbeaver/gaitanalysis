"""
Landmark processing utilities.

This module provides utility functions for processing and analyzing pose landmarks,
including geometric calculations and normalization operations.
"""

import logging
from typing import Optional, Union

import numpy as np
import numpy.typing as npt

from .base import PoseLandmarks, PoseLandmarkIndex

logger = logging.getLogger(__name__)


class LandmarkProcessor:
    """
    Utility class for processing pose landmarks.

    This class provides static methods for common geometric operations on landmarks,
    such as calculating joint angles, distances, and normalization.
    """

    @staticmethod
    def get_joint_angle(
        landmarks: PoseLandmarks,
        point1_idx: Union[int, PoseLandmarkIndex],
        point2_idx: Union[int, PoseLandmarkIndex],
        point3_idx: Union[int, PoseLandmarkIndex],
        use_world_coordinates: bool = False,
    ) -> float:
        """
        Calculate the angle formed by three landmarks (point1 -> point2 -> point3).

        The angle is calculated at point2 (the middle point) formed by the vectors
        from point2 to point1 and from point2 to point3.

        Args:
            landmarks: PoseLandmarks object containing the pose data.
            point1_idx: Index of the first landmark.
            point2_idx: Index of the middle landmark (vertex of the angle).
            point3_idx: Index of the third landmark.
            use_world_coordinates: If True, use world coordinates for calculation (default: False).

        Returns:
            Angle in degrees [0, 180].

        Raises:
            IndexError: If any landmark index is invalid.
            ValueError: If landmarks are collinear or invalid.

        Example:
            # Calculate knee angle (hip -> knee -> ankle)
            knee_angle = LandmarkProcessor.get_joint_angle(
                landmarks,
                PoseLandmarkIndex.LEFT_HIP,
                PoseLandmarkIndex.LEFT_KNEE,
                PoseLandmarkIndex.LEFT_ANKLE
            )
        """
        # Convert enum to int if necessary
        idx1 = int(point1_idx)
        idx2 = int(point2_idx)
        idx3 = int(point3_idx)

        try:
            # Get landmark coordinates
            if use_world_coordinates:
                p1 = np.array(landmarks.get_world_landmark(idx1))
                p2 = np.array(landmarks.get_world_landmark(idx2))
                p3 = np.array(landmarks.get_world_landmark(idx3))
            else:
                p1 = np.array(landmarks.get_landmark(idx1))
                p2 = np.array(landmarks.get_landmark(idx2))
                p3 = np.array(landmarks.get_landmark(idx3))

            # Calculate vectors from the middle point
            v1 = p1 - p2
            v2 = p3 - p2

            # Calculate the angle using dot product
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

            # Clamp to [-1, 1] to handle numerical errors
            cos_angle = np.clip(cos_angle, -1.0, 1.0)

            # Convert to degrees
            angle = np.arccos(cos_angle) * 180.0 / np.pi

            logger.debug(f"Calculated angle at landmark {idx2}: {angle:.2f} degrees")

            return float(angle)

        except ZeroDivisionError as e:
            logger.error(f"Collinear points detected: {idx1}, {idx2}, {idx3}")
            raise ValueError("Cannot calculate angle for collinear points") from e

    @staticmethod
    def get_distance(
        landmarks: PoseLandmarks,
        point1_idx: Union[int, PoseLandmarkIndex],
        point2_idx: Union[int, PoseLandmarkIndex],
        use_world_coordinates: bool = False,
    ) -> float:
        """
        Calculate Euclidean distance between two landmarks.

        Args:
            landmarks: PoseLandmarks object containing the pose data.
            point1_idx: Index of the first landmark.
            point2_idx: Index of the second landmark.
            use_world_coordinates: If True, use world coordinates (meters) for calculation.
                                  If False, use normalized image coordinates (default: False).

        Returns:
            Distance between the two landmarks. Units depend on coordinate system:
            - Image coordinates: normalized units (typically [0, 1] range)
            - World coordinates: meters

        Raises:
            IndexError: If any landmark index is invalid.

        Example:
            # Calculate distance between left hip and left knee in meters
            distance = LandmarkProcessor.get_distance(
                landmarks,
                PoseLandmarkIndex.LEFT_HIP,
                PoseLandmarkIndex.LEFT_KNEE,
                use_world_coordinates=True
            )
        """
        # Convert enum to int if necessary
        idx1 = int(point1_idx)
        idx2 = int(point2_idx)

        # Get landmark coordinates
        if use_world_coordinates:
            p1 = np.array(landmarks.get_world_landmark(idx1))
            p2 = np.array(landmarks.get_world_landmark(idx2))
        else:
            p1 = np.array(landmarks.get_landmark(idx1))
            p2 = np.array(landmarks.get_landmark(idx2))

        # Calculate Euclidean distance
        distance = float(np.linalg.norm(p1 - p2))

        logger.debug(
            f"Distance between landmarks {idx1} and {idx2}: {distance:.4f} "
            f"({'world' if use_world_coordinates else 'image'} coordinates)"
        )

        return distance

    @staticmethod
    def normalize_to_reference(
        landmarks: PoseLandmarks,
        reference_idx1: Union[int, PoseLandmarkIndex],
        reference_idx2: Union[int, PoseLandmarkIndex],
    ) -> PoseLandmarks:
        """
        Normalize landmarks by scaling to a reference distance.

        This is useful for making measurements invariant to the subject's distance
        from the camera. The reference distance (typically hip width or height)
        is scaled to 1.0, and all other landmarks are scaled proportionally.

        Args:
            landmarks: PoseLandmarks object to normalize.
            reference_idx1: Index of first reference landmark (e.g., left hip).
            reference_idx2: Index of second reference landmark (e.g., right hip).

        Returns:
            New PoseLandmarks object with normalized coordinates.

        Raises:
            IndexError: If any landmark index is invalid.
            ValueError: If reference distance is zero or too small.

        Example:
            # Normalize using hip width as reference
            normalized = LandmarkProcessor.normalize_to_reference(
                landmarks,
                PoseLandmarkIndex.LEFT_HIP,
                PoseLandmarkIndex.RIGHT_HIP
            )
        """
        # Convert enum to int if necessary
        idx1 = int(reference_idx1)
        idx2 = int(reference_idx2)

        # Calculate reference distance in world coordinates
        ref_distance = LandmarkProcessor.get_distance(
            landmarks, idx1, idx2, use_world_coordinates=True
        )

        if ref_distance < 1e-6:
            raise ValueError(
                f"Reference distance too small: {ref_distance}. "
                "Cannot normalize with near-zero reference."
            )

        # Calculate scaling factor
        scale_factor = 1.0 / ref_distance

        # Scale world landmarks
        normalized_world = landmarks.world_landmarks * scale_factor

        logger.debug(
            f"Normalized landmarks with reference distance {ref_distance:.4f}m "
            f"(scale factor: {scale_factor:.4f})"
        )

        # Return new PoseLandmarks with normalized coordinates
        return PoseLandmarks(
            landmarks=landmarks.landmarks.copy(),  # Image coords unchanged
            visibility=landmarks.visibility.copy(),
            world_landmarks=normalized_world.astype(np.float32),
            timestamp=landmarks.timestamp,
            frame_number=landmarks.frame_number,
        )

    @staticmethod
    def get_midpoint(
        landmarks: PoseLandmarks,
        point1_idx: Union[int, PoseLandmarkIndex],
        point2_idx: Union[int, PoseLandmarkIndex],
        use_world_coordinates: bool = False,
    ) -> tuple[float, float, float]:
        """
        Calculate the midpoint between two landmarks.

        Args:
            landmarks: PoseLandmarks object containing the pose data.
            point1_idx: Index of the first landmark.
            point2_idx: Index of the second landmark.
            use_world_coordinates: If True, use world coordinates (default: False).

        Returns:
            Tuple of (x, y, z) coordinates of the midpoint.

        Raises:
            IndexError: If any landmark index is invalid.

        Example:
            # Get hip center (midpoint between left and right hip)
            hip_center = LandmarkProcessor.get_midpoint(
                landmarks,
                PoseLandmarkIndex.LEFT_HIP,
                PoseLandmarkIndex.RIGHT_HIP,
                use_world_coordinates=True
            )
        """
        # Convert enum to int if necessary
        idx1 = int(point1_idx)
        idx2 = int(point2_idx)

        # Get landmark coordinates
        if use_world_coordinates:
            p1 = np.array(landmarks.get_world_landmark(idx1))
            p2 = np.array(landmarks.get_world_landmark(idx2))
        else:
            p1 = np.array(landmarks.get_landmark(idx1))
            p2 = np.array(landmarks.get_landmark(idx2))

        # Calculate midpoint
        midpoint = (p1 + p2) / 2.0

        return tuple(midpoint)  # type: ignore

    @staticmethod
    def calculate_velocity(
        landmarks_prev: PoseLandmarks,
        landmarks_curr: PoseLandmarks,
        landmark_idx: Union[int, PoseLandmarkIndex],
        use_world_coordinates: bool = True,
    ) -> tuple[float, float, float]:
        """
        Calculate velocity of a landmark between two frames.

        Args:
            landmarks_prev: PoseLandmarks from the previous frame.
            landmarks_curr: PoseLandmarks from the current frame.
            landmark_idx: Index of the landmark to track.
            use_world_coordinates: If True, use world coordinates (default: True).

        Returns:
            Tuple of (vx, vy, vz) velocity components. Units:
            - World coordinates: meters/second
            - Image coordinates: normalized units/second

        Raises:
            IndexError: If landmark index is invalid.
            ValueError: If timestamps are identical.

        Example:
            # Calculate ankle velocity
            velocity = LandmarkProcessor.calculate_velocity(
                prev_landmarks,
                curr_landmarks,
                PoseLandmarkIndex.LEFT_ANKLE
            )
            speed = np.linalg.norm(velocity)  # Total speed in m/s
        """
        # Convert enum to int if necessary
        idx = int(landmark_idx)

        # Get landmark positions
        if use_world_coordinates:
            p_prev = np.array(landmarks_prev.get_world_landmark(idx))
            p_curr = np.array(landmarks_curr.get_world_landmark(idx))
        else:
            p_prev = np.array(landmarks_prev.get_landmark(idx))
            p_curr = np.array(landmarks_curr.get_landmark(idx))

        # Calculate time difference in seconds
        dt = (landmarks_curr.timestamp - landmarks_prev.timestamp) / 1000.0

        if dt <= 0:
            raise ValueError(
                f"Invalid time difference: {dt}. Current timestamp must be greater than previous."
            )

        # Calculate velocity
        velocity = (p_curr - p_prev) / dt

        logger.debug(
            f"Velocity of landmark {idx}: ({velocity[0]:.3f}, {velocity[1]:.3f}, {velocity[2]:.3f})"
        )

        return tuple(velocity)  # type: ignore

    @staticmethod
    def check_visibility(
        landmarks: PoseLandmarks,
        landmark_indices: list[Union[int, PoseLandmarkIndex]],
        threshold: float = 0.5,
    ) -> bool:
        """
        Check if all specified landmarks meet the visibility threshold.

        Args:
            landmarks: PoseLandmarks object containing the pose data.
            landmark_indices: List of landmark indices to check.
            threshold: Minimum visibility score required [0, 1] (default: 0.5).

        Returns:
            True if all landmarks meet the threshold, False otherwise.

        Raises:
            ValueError: If threshold is not in range [0, 1].

        Example:
            # Check if leg landmarks are visible
            leg_visible = LandmarkProcessor.check_visibility(
                landmarks,
                [PoseLandmarkIndex.LEFT_HIP, PoseLandmarkIndex.LEFT_KNEE,
                 PoseLandmarkIndex.LEFT_ANKLE],
                threshold=0.7
            )
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}")

        for idx in landmark_indices:
            visibility = landmarks.get_visibility(int(idx))
            if visibility < threshold:
                logger.debug(f"Landmark {idx} visibility {visibility:.3f} below threshold {threshold}")
                return False

        return True
