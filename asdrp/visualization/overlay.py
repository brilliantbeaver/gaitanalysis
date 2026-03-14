"""
Pose overlay visualization for running gait analysis.

This module provides classes and functions for drawing pose landmarks and connections
on video frames using OpenCV, with support for MediaPipe pose landmarks.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class GaitEvent(Enum):
    """Gait event types for visualization."""
    FOOT_STRIKE = "foot_strike"
    TOE_OFF = "toe_off"
    MID_STANCE = "mid_stance"
    MID_SWING = "mid_swing"


@dataclass
class LandmarkPoint:
    """Represents a single pose landmark point."""
    x: float
    y: float
    z: float
    visibility: float

    def to_pixel_coords(self, width: int, height: int) -> Tuple[int, int]:
        """Convert normalized coordinates to pixel coordinates."""
        return (int(self.x * width), int(self.y * height))


# MediaPipe Pose landmark indices
POSE_LANDMARKS = {
    'nose': 0,
    'left_eye_inner': 1,
    'left_eye': 2,
    'left_eye_outer': 3,
    'right_eye_inner': 4,
    'right_eye': 5,
    'right_eye_outer': 6,
    'left_ear': 7,
    'right_ear': 8,
    'mouth_left': 9,
    'mouth_right': 10,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_pinky': 17,
    'right_pinky': 18,
    'left_index': 19,
    'right_index': 20,
    'left_thumb': 21,
    'right_thumb': 22,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28,
    'left_heel': 29,
    'right_heel': 30,
    'left_foot_index': 31,
    'right_foot_index': 32,
}

# MediaPipe Pose connections
POSE_CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19),
    # Right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (18, 20),
    # Left leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # Right leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]


class PoseOverlay:
    """
    Class for drawing pose landmarks and connections on video frames.

    This class provides methods to visualize MediaPipe pose landmarks,
    connections, and gait-specific events on video frames.
    """

    def __init__(
        self,
        landmark_color: Tuple[int, int, int] = (0, 255, 0),
        connection_color: Tuple[int, int, int] = (255, 255, 255),
        landmark_radius: int = 5,
        connection_thickness: int = 2,
        visibility_threshold: float = 0.5,
    ):
        """
        Initialize the PoseOverlay.

        Args:
            landmark_color: BGR color for landmarks (default: green)
            connection_color: BGR color for connections (default: white)
            landmark_radius: Radius of landmark circles
            connection_thickness: Thickness of connection lines
            visibility_threshold: Minimum visibility score to draw a landmark
        """
        self.landmark_color = landmark_color
        self.connection_color = connection_color
        self.landmark_radius = landmark_radius
        self.connection_thickness = connection_thickness
        self.visibility_threshold = visibility_threshold

        # Gait event colors
        self.event_colors = {
            GaitEvent.FOOT_STRIKE: (0, 0, 255),      # Red
            GaitEvent.TOE_OFF: (255, 0, 0),          # Blue
            GaitEvent.MID_STANCE: (0, 255, 255),     # Yellow
            GaitEvent.MID_SWING: (255, 0, 255),      # Magenta
        }

    def draw_pose(
        self,
        frame: np.ndarray,
        landmarks: List[LandmarkPoint],
        draw_landmarks: bool = True,
        draw_connections: bool = True,
    ) -> np.ndarray:
        """
        Draw complete pose on a frame.

        Args:
            frame: Input video frame (BGR)
            landmarks: List of pose landmarks
            draw_landmarks: Whether to draw landmark points
            draw_connections: Whether to draw skeleton connections

        Returns:
            Frame with pose overlay
        """
        output_frame = frame.copy()
        height, width = frame.shape[:2]

        if draw_connections:
            output_frame = self.draw_connections(output_frame, landmarks)

        if draw_landmarks:
            output_frame = self.draw_landmarks(output_frame, landmarks)

        return output_frame

    def draw_landmarks(
        self,
        frame: np.ndarray,
        landmarks: List[LandmarkPoint],
        highlight_indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Draw pose landmarks on a frame.

        Args:
            frame: Input video frame (BGR)
            landmarks: List of pose landmarks
            highlight_indices: Optional list of landmark indices to highlight

        Returns:
            Frame with landmarks drawn
        """
        output_frame = frame.copy()
        height, width = frame.shape[:2]

        for idx, landmark in enumerate(landmarks):
            if landmark.visibility < self.visibility_threshold:
                continue

            x, y = landmark.to_pixel_coords(width, height)

            # Determine color and radius
            if highlight_indices and idx in highlight_indices:
                color = (0, 0, 255)  # Red for highlighted landmarks
                radius = self.landmark_radius + 2
            else:
                color = self.landmark_color
                radius = self.landmark_radius

            # Draw landmark
            cv2.circle(output_frame, (x, y), radius, color, -1)
            cv2.circle(output_frame, (x, y), radius + 1, (0, 0, 0), 1)  # Black border

        return output_frame

    def draw_connections(
        self,
        frame: np.ndarray,
        landmarks: List[LandmarkPoint],
        connections: Optional[List[Tuple[int, int]]] = None,
    ) -> np.ndarray:
        """
        Draw skeleton connections between landmarks.

        Args:
            frame: Input video frame (BGR)
            landmarks: List of pose landmarks
            connections: Optional list of landmark pairs to connect
                        (defaults to POSE_CONNECTIONS)

        Returns:
            Frame with connections drawn
        """
        output_frame = frame.copy()
        height, width = frame.shape[:2]

        if connections is None:
            connections = POSE_CONNECTIONS

        for start_idx, end_idx in connections:
            if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                continue

            start_landmark = landmarks[start_idx]
            end_landmark = landmarks[end_idx]

            # Check visibility
            if (start_landmark.visibility < self.visibility_threshold or
                end_landmark.visibility < self.visibility_threshold):
                continue

            # Get pixel coordinates
            start_pos = start_landmark.to_pixel_coords(width, height)
            end_pos = end_landmark.to_pixel_coords(width, height)

            # Draw connection
            cv2.line(
                output_frame,
                start_pos,
                end_pos,
                self.connection_color,
                self.connection_thickness,
            )

        return output_frame

    def draw_gait_events(
        self,
        frame: np.ndarray,
        events: Dict[str, Any],
        landmarks: List[LandmarkPoint],
    ) -> np.ndarray:
        """
        Draw gait-specific events on the frame.

        Args:
            frame: Input video frame (BGR)
            events: Dictionary of gait events with structure:
                    {
                        'left_foot': GaitEvent or None,
                        'right_foot': GaitEvent or None,
                    }
            landmarks: List of pose landmarks

        Returns:
            Frame with gait events visualized
        """
        output_frame = frame.copy()
        height, width = frame.shape[:2]

        # Define foot landmark indices
        foot_landmarks = {
            'left_foot': [
                POSE_LANDMARKS['left_ankle'],
                POSE_LANDMARKS['left_heel'],
                POSE_LANDMARKS['left_foot_index']
            ],
            'right_foot': [
                POSE_LANDMARKS['right_ankle'],
                POSE_LANDMARKS['right_heel'],
                POSE_LANDMARKS['right_foot_index']
            ]
        }

        for foot_side, event in events.items():
            if event is None:
                continue

            if foot_side not in foot_landmarks:
                continue

            # Get foot center position
            foot_indices = foot_landmarks[foot_side]
            visible_landmarks = [
                landmarks[idx] for idx in foot_indices
                if idx < len(landmarks) and landmarks[idx].visibility >= self.visibility_threshold
            ]

            if not visible_landmarks:
                continue

            # Calculate average position
            avg_x = sum(lm.x for lm in visible_landmarks) / len(visible_landmarks)
            avg_y = sum(lm.y for lm in visible_landmarks) / len(visible_landmarks)
            center = (int(avg_x * width), int(avg_y * height))

            # Draw event marker
            color = self.event_colors.get(event, (255, 255, 255))

            # Draw circle around foot
            cv2.circle(output_frame, center, 30, color, 3)

            # Draw event label
            label = event.value.replace('_', ' ').title()
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2

            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )

            # Draw text background
            text_x = center[0] - text_width // 2
            text_y = center[1] + 50
            cv2.rectangle(
                output_frame,
                (text_x - 5, text_y - text_height - 5),
                (text_x + text_width + 5, text_y + baseline + 5),
                (0, 0, 0),
                -1,
            )

            # Draw text
            cv2.putText(
                output_frame,
                label,
                (text_x, text_y),
                font,
                font_scale,
                color,
                thickness,
            )

        return output_frame

    def draw_angle(
        self,
        frame: np.ndarray,
        landmarks: List[LandmarkPoint],
        joint_indices: Tuple[int, int, int],
        angle_value: float,
        label: Optional[str] = None,
    ) -> np.ndarray:
        """
        Draw an angle measurement between three landmarks.

        Args:
            frame: Input video frame (BGR)
            landmarks: List of pose landmarks
            joint_indices: Tuple of (point1, joint, point2) landmark indices
            angle_value: Angle value in degrees
            label: Optional label for the angle

        Returns:
            Frame with angle visualization
        """
        output_frame = frame.copy()
        height, width = frame.shape[:2]

        p1_idx, joint_idx, p2_idx = joint_indices

        # Check if indices are valid and landmarks are visible
        if (p1_idx >= len(landmarks) or joint_idx >= len(landmarks) or
            p2_idx >= len(landmarks)):
            return output_frame

        p1 = landmarks[p1_idx]
        joint = landmarks[joint_idx]
        p2 = landmarks[p2_idx]

        if (p1.visibility < self.visibility_threshold or
            joint.visibility < self.visibility_threshold or
            p2.visibility < self.visibility_threshold):
            return output_frame

        # Get pixel coordinates
        p1_pos = p1.to_pixel_coords(width, height)
        joint_pos = joint.to_pixel_coords(width, height)
        p2_pos = p2.to_pixel_coords(width, height)

        # Draw lines
        cv2.line(output_frame, joint_pos, p1_pos, (255, 200, 0), 2)
        cv2.line(output_frame, joint_pos, p2_pos, (255, 200, 0), 2)

        # Draw arc for angle
        angle_radius = 30
        # Calculate angles for arc
        v1 = np.array([p1_pos[0] - joint_pos[0], p1_pos[1] - joint_pos[1]])
        v2 = np.array([p2_pos[0] - joint_pos[0], p2_pos[1] - joint_pos[1]])

        angle1 = np.degrees(np.arctan2(v1[1], v1[0]))
        angle2 = np.degrees(np.arctan2(v2[1], v2[0]))

        cv2.ellipse(
            output_frame,
            joint_pos,
            (angle_radius, angle_radius),
            0,
            angle1,
            angle2,
            (255, 200, 0),
            2,
        )

        # Draw angle text
        text = f"{angle_value:.1f}°"
        if label:
            text = f"{label}: {text}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2

        # Position text near the joint
        text_pos = (joint_pos[0] + 10, joint_pos[1] - 10)

        # Draw text background
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        cv2.rectangle(
            output_frame,
            (text_pos[0] - 2, text_pos[1] - text_height - 2),
            (text_pos[0] + text_width + 2, text_pos[1] + baseline + 2),
            (0, 0, 0),
            -1,
        )

        # Draw text
        cv2.putText(
            output_frame,
            text,
            text_pos,
            font,
            font_scale,
            (255, 200, 0),
            thickness,
        )

        return output_frame

    def add_info_panel(
        self,
        frame: np.ndarray,
        info: Dict[str, Any],
        position: str = "top-left",
    ) -> np.ndarray:
        """
        Add an information panel to the frame.

        Args:
            frame: Input video frame (BGR)
            info: Dictionary of information to display
            position: Panel position ("top-left", "top-right", "bottom-left", "bottom-right")

        Returns:
            Frame with information panel
        """
        output_frame = frame.copy()
        height, width = frame.shape[:2]

        # Prepare text
        lines = [f"{key}: {value}" for key, value in info.items()]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        line_spacing = 30
        padding = 10

        # Calculate panel size
        max_text_width = 0
        for line in lines:
            (text_width, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
            max_text_width = max(max_text_width, text_width)

        panel_width = max_text_width + 2 * padding
        panel_height = len(lines) * line_spacing + 2 * padding

        # Determine panel position
        if position == "top-left":
            x, y = 10, 10
        elif position == "top-right":
            x, y = width - panel_width - 10, 10
        elif position == "bottom-left":
            x, y = 10, height - panel_height - 10
        else:  # bottom-right
            x, y = width - panel_width - 10, height - panel_height - 10

        # Draw panel background
        cv2.rectangle(
            output_frame,
            (x, y),
            (x + panel_width, y + panel_height),
            (0, 0, 0),
            -1,
        )
        cv2.rectangle(
            output_frame,
            (x, y),
            (x + panel_width, y + panel_height),
            (255, 255, 255),
            2,
        )

        # Draw text
        for i, line in enumerate(lines):
            text_y = y + padding + (i + 1) * line_spacing - 5
            cv2.putText(
                output_frame,
                line,
                (x + padding, text_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
            )

        return output_frame
