"""Stride analysis for running gait.

This module implements stride detection and analysis, including heel strike
and toe off event detection, as well as stride length and timing calculations.
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from scipy.signal import find_peaks, savgol_filter

from .metrics import BaseMetricCalculator, GaitEvent, GaitEventType, Foot


class StrideAnalyzer(BaseMetricCalculator):
    """Analyzer for stride metrics and gait event detection.

    This class detects heel strikes and toe offs using kinematic analysis
    of foot landmarks, and calculates stride-related metrics.
    """

    def __init__(self, fps: float = 30.0, min_stride_time: float = 0.3,
                 max_stride_time: float = 2.0):
        """Initialize stride analyzer.

        Args:
            fps: Video frame rate (frames per second)
            min_stride_time: Minimum expected stride duration in seconds
            max_stride_time: Maximum expected stride duration in seconds
        """
        super().__init__(fps)
        self.min_stride_time = min_stride_time
        self.max_stride_time = max_stride_time
        # Calculate minimum distance between peaks in frames
        self.min_distance = int(min_stride_time * fps)

    def calculate(self, landmarks_sequence: List[Dict[str, Any]],
                  events: Optional[List[GaitEvent]] = None) -> Dict[str, Any]:
        """Calculate stride metrics and detect gait events.

        Args:
            landmarks_sequence: List of pose landmarks for each frame
            events: Optional list of previously detected events (unused here)

        Returns:
            Dictionary containing:
                - events: List of GaitEvent objects
                - stride_time: Average stride time in seconds
                - stride_length: Average stride length (normalized)
                - stance_phase_duration: Average stance phase duration
                - swing_phase_duration: Average swing phase duration
                - total_strides: Total number of complete strides
                - left_metrics: Left foot specific metrics
                - right_metrics: Right foot specific metrics
        """
        # Detect heel strikes and toe offs
        heel_strikes = self.detect_heel_strikes(landmarks_sequence)
        toe_offs = self.detect_toe_offs(landmarks_sequence)

        # Combine all events
        all_events = heel_strikes + toe_offs
        all_events.sort(key=lambda e: e.frame_number)

        # Calculate stride metrics
        stride_times = self._calculate_stride_times(heel_strikes)
        stride_lengths = self._calculate_stride_lengths(landmarks_sequence, heel_strikes)
        phase_durations = self._calculate_phase_durations(heel_strikes, toe_offs)

        # Separate metrics by foot
        left_heel_strikes = [e for e in heel_strikes if e.foot == Foot.LEFT]
        right_heel_strikes = [e for e in heel_strikes if e.foot == Foot.RIGHT]

        left_metrics = self._compute_side_metrics(
            left_heel_strikes, landmarks_sequence, Foot.LEFT
        )
        right_metrics = self._compute_side_metrics(
            right_heel_strikes, landmarks_sequence, Foot.RIGHT
        )

        return {
            'events': all_events,
            'stride_time': np.mean(stride_times) if stride_times else 0.0,
            'stride_length': np.mean(stride_lengths) if stride_lengths else 0.0,
            'stance_phase_duration': phase_durations['stance'],
            'swing_phase_duration': phase_durations['swing'],
            'total_strides': len(heel_strikes) - 2,  # -2 to count complete strides
            'left_metrics': left_metrics,
            'right_metrics': right_metrics
        }

    def detect_heel_strikes(self, landmarks_sequence: List[Dict[str, Any]]) -> List[GaitEvent]:
        """Detect heel strike events using vertical velocity and position.

        Heel strike occurs when:
        1. Foot reaches minimum vertical position (lowest point)
        2. Vertical velocity transitions from negative (downward) to positive (upward)
        3. Foot is in front of the body (hip)

        Args:
            landmarks_sequence: List of pose landmarks for each frame

        Returns:
            List of GaitEvent objects for heel strikes
        """
        heel_strikes = []

        for foot_type in [Foot.LEFT, Foot.RIGHT]:
            # Extract foot trajectory
            foot_y_positions = []
            foot_x_positions = []
            hip_x_positions = []

            foot_key = 'left_ankle' if foot_type == Foot.LEFT else 'right_ankle'
            hip_key = 'left_hip' if foot_type == Foot.LEFT else 'right_hip'

            for frame_idx, landmarks in enumerate(landmarks_sequence):
                if foot_key in landmarks and hip_key in landmarks:
                    foot_y_positions.append(landmarks[foot_key]['y'])
                    foot_x_positions.append(landmarks[foot_key]['x'])
                    hip_x_positions.append(landmarks[hip_key]['x'])
                else:
                    # Use interpolation for missing data
                    if foot_y_positions:
                        foot_y_positions.append(foot_y_positions[-1])
                        foot_x_positions.append(foot_x_positions[-1])
                        hip_x_positions.append(hip_x_positions[-1])
                    else:
                        foot_y_positions.append(0.0)
                        foot_x_positions.append(0.0)
                        hip_x_positions.append(0.0)

            foot_y = np.array(foot_y_positions)
            foot_x = np.array(foot_x_positions)
            hip_x = np.array(hip_x_positions)

            # Smooth the signal to reduce noise
            if len(foot_y) > 5:
                window_length = min(11, len(foot_y) if len(foot_y) % 2 == 1 else len(foot_y) - 1)
                foot_y_smooth = savgol_filter(foot_y, window_length, 3)
            else:
                foot_y_smooth = foot_y

            # Find peaks in vertical position (lowest points = highest y values in image coords)
            # Note: In image coordinates, y increases downward
            peaks, properties = find_peaks(
                foot_y_smooth,
                distance=self.min_distance,
                prominence=0.01  # Minimum prominence to avoid noise
            )

            # Validate peaks (foot should be in front of hip)
            valid_peaks = []
            for peak in peaks:
                # Check if foot is roughly in front of or aligned with hip
                # This helps distinguish heel strike from other foot positions
                if abs(foot_x[peak] - hip_x[peak]) < 0.3:  # Reasonable threshold
                    valid_peaks.append(peak)

            # Create GaitEvent objects
            for peak in valid_peaks:
                if peak < len(landmarks_sequence):
                    landmark_data = self._extract_landmark_data(
                        landmarks_sequence[peak], foot_type
                    )
                    event = GaitEvent(
                        event_type=GaitEventType.HEEL_STRIKE,
                        timestamp=peak / self.fps,
                        frame_number=peak,
                        foot=foot_type,
                        landmark_data=landmark_data
                    )
                    heel_strikes.append(event)

        # Sort by frame number
        heel_strikes.sort(key=lambda e: e.frame_number)
        return heel_strikes

    def detect_toe_offs(self, landmarks_sequence: List[Dict[str, Any]]) -> List[GaitEvent]:
        """Detect toe off events using vertical velocity and position.

        Toe off occurs when:
        1. Foot starts moving upward rapidly (beginning of swing phase)
        2. Toe is behind the body (hip)
        3. Vertical acceleration is positive

        Args:
            landmarks_sequence: List of pose landmarks for each frame

        Returns:
            List of GaitEvent objects for toe offs
        """
        toe_offs = []

        for foot_type in [Foot.LEFT, Foot.RIGHT]:
            # Extract foot trajectory
            foot_y_positions = []
            foot_x_positions = []
            hip_x_positions = []

            foot_key = 'left_foot_index' if foot_type == Foot.LEFT else 'right_foot_index'
            hip_key = 'left_hip' if foot_type == Foot.LEFT else 'right_hip'

            # Fallback to ankle if foot_index not available
            alt_foot_key = 'left_ankle' if foot_type == Foot.LEFT else 'right_ankle'

            for frame_idx, landmarks in enumerate(landmarks_sequence):
                # Try foot_index first, then ankle
                if foot_key in landmarks:
                    current_foot_key = foot_key
                elif alt_foot_key in landmarks:
                    current_foot_key = alt_foot_key
                else:
                    current_foot_key = None

                if current_foot_key and hip_key in landmarks:
                    foot_y_positions.append(landmarks[current_foot_key]['y'])
                    foot_x_positions.append(landmarks[current_foot_key]['x'])
                    hip_x_positions.append(landmarks[hip_key]['x'])
                else:
                    if foot_y_positions:
                        foot_y_positions.append(foot_y_positions[-1])
                        foot_x_positions.append(foot_x_positions[-1])
                        hip_x_positions.append(hip_x_positions[-1])
                    else:
                        foot_y_positions.append(0.0)
                        foot_x_positions.append(0.0)
                        hip_x_positions.append(0.0)

            foot_y = np.array(foot_y_positions)
            foot_x = np.array(foot_x_positions)
            hip_x = np.array(hip_x_positions)

            # Smooth the signal
            if len(foot_y) > 5:
                window_length = min(11, len(foot_y) if len(foot_y) % 2 == 1 else len(foot_y) - 1)
                foot_y_smooth = savgol_filter(foot_y, window_length, 3)
            else:
                foot_y_smooth = foot_y

            # Find valleys (minimum y = foot leaving ground)
            # Invert signal to find valleys as peaks
            peaks, properties = find_peaks(
                -foot_y_smooth,
                distance=self.min_distance,
                prominence=0.01
            )

            # Validate peaks (foot should be behind hip at toe off)
            valid_peaks = []
            for peak in peaks:
                # For running from left to right, left foot behind means x < hip_x
                # For right to left, reverse this logic
                # We use absolute difference as a simple heuristic
                if abs(foot_x[peak] - hip_x[peak]) < 0.3:
                    valid_peaks.append(peak)

            # Create GaitEvent objects
            for peak in valid_peaks:
                if peak < len(landmarks_sequence):
                    landmark_data = self._extract_landmark_data(
                        landmarks_sequence[peak], foot_type
                    )
                    event = GaitEvent(
                        event_type=GaitEventType.TOE_OFF,
                        timestamp=peak / self.fps,
                        frame_number=peak,
                        foot=foot_type,
                        landmark_data=landmark_data
                    )
                    toe_offs.append(event)

        # Sort by frame number
        toe_offs.sort(key=lambda e: e.frame_number)
        return toe_offs

    def _calculate_stride_times(self, heel_strikes: List[GaitEvent]) -> List[float]:
        """Calculate stride times from heel strike events.

        A stride is from one heel strike to the next heel strike of the same foot.

        Args:
            heel_strikes: List of heel strike events

        Returns:
            List of stride times in seconds
        """
        stride_times = []

        # Separate by foot
        for foot_type in [Foot.LEFT, Foot.RIGHT]:
            foot_strikes = [e for e in heel_strikes if e.foot == foot_type]

            # Calculate time between consecutive strikes
            for i in range(len(foot_strikes) - 1):
                stride_time = foot_strikes[i + 1].timestamp - foot_strikes[i].timestamp
                # Validate stride time is within reasonable range
                if self.min_stride_time <= stride_time <= self.max_stride_time:
                    stride_times.append(stride_time)

        return stride_times

    def _calculate_stride_lengths(self, landmarks_sequence: List[Dict[str, Any]],
                                  heel_strikes: List[GaitEvent]) -> List[float]:
        """Calculate stride lengths from heel strike positions.

        Stride length is the horizontal distance traveled between
        successive heel strikes of the same foot.

        Args:
            landmarks_sequence: List of pose landmarks
            heel_strikes: List of heel strike events

        Returns:
            List of normalized stride lengths
        """
        stride_lengths = []

        for foot_type in [Foot.LEFT, Foot.RIGHT]:
            foot_strikes = [e for e in heel_strikes if e.foot == foot_type]

            for i in range(len(foot_strikes) - 1):
                # Get positions at two consecutive heel strikes
                frame1 = foot_strikes[i].frame_number
                frame2 = foot_strikes[i + 1].frame_number

                if frame1 < len(landmarks_sequence) and frame2 < len(landmarks_sequence):
                    ankle_key = 'left_ankle' if foot_type == Foot.LEFT else 'right_ankle'

                    if ankle_key in landmarks_sequence[frame1] and \
                       ankle_key in landmarks_sequence[frame2]:
                        pos1 = np.array([
                            landmarks_sequence[frame1][ankle_key]['x'],
                            landmarks_sequence[frame1][ankle_key]['y']
                        ])
                        pos2 = np.array([
                            landmarks_sequence[frame2][ankle_key]['x'],
                            landmarks_sequence[frame2][ankle_key]['y']
                        ])

                        # Calculate Euclidean distance (normalized units)
                        stride_length = np.linalg.norm(pos2 - pos1)
                        stride_lengths.append(stride_length)

        return stride_lengths

    def _calculate_phase_durations(self, heel_strikes: List[GaitEvent],
                                   toe_offs: List[GaitEvent]) -> Dict[str, float]:
        """Calculate average stance and swing phase durations.

        Stance phase: heel strike to toe off (foot on ground)
        Swing phase: toe off to heel strike (foot in air)

        Args:
            heel_strikes: List of heel strike events
            toe_offs: List of toe off events

        Returns:
            Dictionary with 'stance' and 'swing' durations in seconds
        """
        stance_durations = []
        swing_durations = []

        for foot_type in [Foot.LEFT, Foot.RIGHT]:
            foot_heel_strikes = [e for e in heel_strikes if e.foot == foot_type]
            foot_toe_offs = [e for e in toe_offs if e.foot == foot_type]

            # Calculate stance phase (heel strike to toe off)
            for hs in foot_heel_strikes:
                # Find the next toe off after this heel strike
                next_toe_offs = [to for to in foot_toe_offs if to.timestamp > hs.timestamp]
                if next_toe_offs:
                    stance_duration = next_toe_offs[0].timestamp - hs.timestamp
                    if 0.05 <= stance_duration <= 0.5:  # Reasonable range for running
                        stance_durations.append(stance_duration)

            # Calculate swing phase (toe off to heel strike)
            for to in foot_toe_offs:
                # Find the next heel strike after this toe off
                next_heel_strikes = [hs for hs in foot_heel_strikes if hs.timestamp > to.timestamp]
                if next_heel_strikes:
                    swing_duration = next_heel_strikes[0].timestamp - to.timestamp
                    if 0.1 <= swing_duration <= 0.8:  # Reasonable range for running
                        swing_durations.append(swing_duration)

        return {
            'stance': np.mean(stance_durations) if stance_durations else 0.0,
            'swing': np.mean(swing_durations) if swing_durations else 0.0
        }

    def _compute_side_metrics(self, heel_strikes: List[GaitEvent],
                             landmarks_sequence: List[Dict[str, Any]],
                             foot: Foot) -> Dict[str, float]:
        """Compute metrics specific to one side (left or right foot).

        Args:
            heel_strikes: Heel strike events for this foot
            landmarks_sequence: Full landmark sequence
            foot: Which foot (LEFT or RIGHT)

        Returns:
            Dictionary of side-specific metrics
        """
        if len(heel_strikes) < 2:
            return {
                'stride_count': 0,
                'avg_stride_time': 0.0,
                'std_stride_time': 0.0
            }

        # Calculate stride times
        stride_times = []
        for i in range(len(heel_strikes) - 1):
            stride_time = heel_strikes[i + 1].timestamp - heel_strikes[i].timestamp
            if self.min_stride_time <= stride_time <= self.max_stride_time:
                stride_times.append(stride_time)

        return {
            'stride_count': len(stride_times),
            'avg_stride_time': np.mean(stride_times) if stride_times else 0.0,
            'std_stride_time': np.std(stride_times) if stride_times else 0.0
        }

    def _extract_landmark_data(self, landmarks: Dict[str, Any],
                               foot: Foot) -> Dict[str, Dict[str, float]]:
        """Extract relevant landmarks for a gait event.

        Args:
            landmarks: Dictionary of all landmarks for a frame
            foot: Which foot the event belongs to

        Returns:
            Dictionary of relevant landmarks with coordinates
        """
        side = 'left' if foot == Foot.LEFT else 'right'
        landmark_keys = [
            f'{side}_hip',
            f'{side}_knee',
            f'{side}_ankle',
            f'{side}_heel',
            f'{side}_foot_index'
        ]

        landmark_data = {}
        for key in landmark_keys:
            if key in landmarks:
                landmark_data[key] = landmarks[key].copy()

        return landmark_data
