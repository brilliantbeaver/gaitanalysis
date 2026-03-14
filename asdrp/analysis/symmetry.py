"""Symmetry analysis for running gait.

This module compares left and right side metrics to assess gait symmetry.
Asymmetry can indicate biomechanical issues, injury, or training imbalances.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from scipy.stats import pearsonr

from .metrics import BaseMetricCalculator, GaitEvent, GaitEventType, Foot


class SymmetryAnalyzer(BaseMetricCalculator):
    """Analyzer for assessing bilateral gait symmetry.

    Symmetry is assessed using multiple methods:
    1. Symmetry Index (SI): Normalized difference between sides
    2. Symmetry Ratio (SR): Ratio between sides
    3. Gait Asymmetry (GA): Percentage difference
    4. Correlation between left and right side kinematics
    """

    def __init__(self, fps: float = 30.0):
        """Initialize symmetry analyzer.

        Args:
            fps: Video frame rate (frames per second)
        """
        super().__init__(fps)

    def calculate(self, landmarks_sequence: List[Dict[str, Any]],
                  events: Optional[List[GaitEvent]] = None) -> Dict[str, Any]:
        """Calculate symmetry metrics comparing left and right sides.

        Args:
            landmarks_sequence: List of pose landmarks for each frame
            events: Optional list of detected gait events

        Returns:
            Dictionary containing:
                - symmetry_index: Overall symmetry index (0-1, 1 = perfect)
                - temporal_symmetry: Symmetry of timing metrics
                - spatial_symmetry: Symmetry of spatial metrics
                - kinematic_symmetry: Symmetry of joint angles
                - left_metrics: Left side specific metrics
                - right_metrics: Right side specific metrics
        """
        results = {
            'symmetry_index': 0.0,
            'temporal_symmetry': 0.0,
            'spatial_symmetry': 0.0,
            'kinematic_symmetry': 0.0
        }

        if events is None or len(events) < 4:
            # Need sufficient events for meaningful symmetry analysis
            return results

        # Calculate temporal symmetry (stride times, stance times, etc.)
        temporal_symmetry = self._calculate_temporal_symmetry(events)

        # Calculate spatial symmetry (stride lengths, positions)
        spatial_symmetry = self._calculate_spatial_symmetry(landmarks_sequence, events)

        # Calculate kinematic symmetry (joint angles)
        kinematic_symmetry = self._calculate_kinematic_symmetry(landmarks_sequence, events)

        # Compute overall symmetry index as weighted average
        overall_symmetry = (
            0.4 * temporal_symmetry['index'] +
            0.3 * spatial_symmetry['index'] +
            0.3 * kinematic_symmetry['index']
        )

        results.update({
            'symmetry_index': overall_symmetry,
            'temporal_symmetry': temporal_symmetry['index'],
            'spatial_symmetry': spatial_symmetry['index'],
            'kinematic_symmetry': kinematic_symmetry['index'],
            'temporal_details': temporal_symmetry,
            'spatial_details': spatial_symmetry,
            'kinematic_details': kinematic_symmetry
        })

        return results

    def _calculate_temporal_symmetry(self, events: List[GaitEvent]) -> Dict[str, float]:
        """Calculate symmetry of temporal gait parameters.

        Compares stride times, stance times, and swing times between left and right.

        Args:
            events: List of gait events

        Returns:
            Dictionary with temporal symmetry metrics
        """
        # Separate events by foot
        left_heel_strikes = [e for e in events if e.event_type == GaitEventType.HEEL_STRIKE
                            and e.foot == Foot.LEFT]
        right_heel_strikes = [e for e in events if e.event_type == GaitEventType.HEEL_STRIKE
                             and e.foot == Foot.RIGHT]
        left_toe_offs = [e for e in events if e.event_type == GaitEventType.TOE_OFF
                        and e.foot == Foot.LEFT]
        right_toe_offs = [e for e in events if e.event_type == GaitEventType.TOE_OFF
                         and e.foot == Foot.RIGHT]

        # Calculate stride times for each foot
        left_stride_times = self._calculate_stride_times_for_foot(left_heel_strikes)
        right_stride_times = self._calculate_stride_times_for_foot(right_heel_strikes)

        # Calculate stance times for each foot
        left_stance_times = self._calculate_stance_times(left_heel_strikes, left_toe_offs)
        right_stance_times = self._calculate_stance_times(right_heel_strikes, right_toe_offs)

        # Calculate swing times for each foot
        left_swing_times = self._calculate_swing_times(left_toe_offs, left_heel_strikes)
        right_swing_times = self._calculate_swing_times(right_toe_offs, right_heel_strikes)

        # Compute symmetry indices
        stride_symmetry = self._compute_symmetry_index(
            np.mean(left_stride_times) if left_stride_times else 0.0,
            np.mean(right_stride_times) if right_stride_times else 0.0
        )

        stance_symmetry = self._compute_symmetry_index(
            np.mean(left_stance_times) if left_stance_times else 0.0,
            np.mean(right_stance_times) if right_stance_times else 0.0
        )

        swing_symmetry = self._compute_symmetry_index(
            np.mean(left_swing_times) if left_swing_times else 0.0,
            np.mean(right_swing_times) if right_swing_times else 0.0
        )

        # Overall temporal symmetry
        temporal_index = np.mean([stride_symmetry, stance_symmetry, swing_symmetry])

        return {
            'index': temporal_index,
            'stride_symmetry': stride_symmetry,
            'stance_symmetry': stance_symmetry,
            'swing_symmetry': swing_symmetry,
            'left_avg_stride_time': np.mean(left_stride_times) if left_stride_times else 0.0,
            'right_avg_stride_time': np.mean(right_stride_times) if right_stride_times else 0.0,
            'left_avg_stance_time': np.mean(left_stance_times) if left_stance_times else 0.0,
            'right_avg_stance_time': np.mean(right_stance_times) if right_stance_times else 0.0
        }

    def _calculate_spatial_symmetry(self, landmarks_sequence: List[Dict[str, Any]],
                                    events: List[GaitEvent]) -> Dict[str, float]:
        """Calculate symmetry of spatial gait parameters.

        Compares stride lengths and step widths between left and right.

        Args:
            landmarks_sequence: List of pose landmarks
            events: List of gait events

        Returns:
            Dictionary with spatial symmetry metrics
        """
        # Get heel strike events
        left_heel_strikes = [e for e in events if e.event_type == GaitEventType.HEEL_STRIKE
                            and e.foot == Foot.LEFT]
        right_heel_strikes = [e for e in events if e.event_type == GaitEventType.HEEL_STRIKE
                             and e.foot == Foot.RIGHT]

        # Calculate stride lengths
        left_stride_lengths = self._calculate_stride_lengths_for_foot(
            landmarks_sequence, left_heel_strikes, 'left_ankle'
        )
        right_stride_lengths = self._calculate_stride_lengths_for_foot(
            landmarks_sequence, right_heel_strikes, 'right_ankle'
        )

        # Calculate step widths (lateral distance between feet)
        step_widths = self._calculate_step_widths(landmarks_sequence, events)

        # Compute symmetry index for stride lengths
        stride_length_symmetry = self._compute_symmetry_index(
            np.mean(left_stride_lengths) if left_stride_lengths else 0.0,
            np.mean(right_stride_lengths) if right_stride_lengths else 0.0
        )

        # Overall spatial symmetry
        spatial_index = stride_length_symmetry

        return {
            'index': spatial_index,
            'stride_length_symmetry': stride_length_symmetry,
            'left_avg_stride_length': np.mean(left_stride_lengths) if left_stride_lengths else 0.0,
            'right_avg_stride_length': np.mean(right_stride_lengths) if right_stride_lengths else 0.0,
            'avg_step_width': np.mean(step_widths) if step_widths else 0.0
        }

    def _calculate_kinematic_symmetry(self, landmarks_sequence: List[Dict[str, Any]],
                                      events: List[GaitEvent]) -> Dict[str, float]:
        """Calculate symmetry of kinematic parameters (joint angles).

        Compares knee and hip angles between left and right sides.

        Args:
            landmarks_sequence: List of pose landmarks
            events: List of gait events

        Returns:
            Dictionary with kinematic symmetry metrics
        """
        # Extract joint angles throughout the sequence
        left_knee_angles = []
        right_knee_angles = []
        left_hip_angles = []
        right_hip_angles = []

        for landmarks in landmarks_sequence:
            # Left knee angle
            if all(k in landmarks for k in ['left_hip', 'left_knee', 'left_ankle']):
                hip = self._extract_landmark_coords(landmarks['left_hip'])
                knee = self._extract_landmark_coords(landmarks['left_knee'])
                ankle = self._extract_landmark_coords(landmarks['left_ankle'])
                left_knee_angles.append(self._calculate_angle(hip, knee, ankle))

            # Right knee angle
            if all(k in landmarks for k in ['right_hip', 'right_knee', 'right_ankle']):
                hip = self._extract_landmark_coords(landmarks['right_hip'])
                knee = self._extract_landmark_coords(landmarks['right_knee'])
                ankle = self._extract_landmark_coords(landmarks['right_ankle'])
                right_knee_angles.append(self._calculate_angle(hip, knee, ankle))

            # Left hip angle (using shoulder, hip, knee)
            if all(k in landmarks for k in ['left_shoulder', 'left_hip', 'left_knee']):
                shoulder = self._extract_landmark_coords(landmarks['left_shoulder'])
                hip = self._extract_landmark_coords(landmarks['left_hip'])
                knee = self._extract_landmark_coords(landmarks['left_knee'])
                left_hip_angles.append(self._calculate_angle(shoulder, hip, knee))

            # Right hip angle
            if all(k in landmarks for k in ['right_shoulder', 'right_hip', 'right_knee']):
                shoulder = self._extract_landmark_coords(landmarks['right_shoulder'])
                hip = self._extract_landmark_coords(landmarks['right_hip'])
                knee = self._extract_landmark_coords(landmarks['right_knee'])
                right_hip_angles.append(self._calculate_angle(shoulder, hip, knee))

        # Calculate range of motion for each joint
        left_knee_rom = max(left_knee_angles) - min(left_knee_angles) if left_knee_angles else 0.0
        right_knee_rom = max(right_knee_angles) - min(right_knee_angles) if right_knee_angles else 0.0
        left_hip_rom = max(left_hip_angles) - min(left_hip_angles) if left_hip_angles else 0.0
        right_hip_rom = max(right_hip_angles) - min(right_hip_angles) if right_hip_angles else 0.0

        # Compute symmetry indices
        knee_rom_symmetry = self._compute_symmetry_index(left_knee_rom, right_knee_rom)
        hip_rom_symmetry = self._compute_symmetry_index(left_hip_rom, right_hip_rom)

        # Calculate correlation between left and right angle patterns
        knee_correlation = 0.0
        hip_correlation = 0.0

        if len(left_knee_angles) == len(right_knee_angles) and len(left_knee_angles) > 2:
            try:
                knee_correlation, _ = pearsonr(left_knee_angles, right_knee_angles)
                knee_correlation = abs(knee_correlation)  # Use absolute value
            except:
                knee_correlation = 0.0

        if len(left_hip_angles) == len(right_hip_angles) and len(left_hip_angles) > 2:
            try:
                hip_correlation, _ = pearsonr(left_hip_angles, right_hip_angles)
                hip_correlation = abs(hip_correlation)
            except:
                hip_correlation = 0.0

        # Overall kinematic symmetry
        kinematic_index = np.mean([
            knee_rom_symmetry,
            hip_rom_symmetry,
            knee_correlation,
            hip_correlation
        ])

        return {
            'index': kinematic_index,
            'knee_rom_symmetry': knee_rom_symmetry,
            'hip_rom_symmetry': hip_rom_symmetry,
            'knee_correlation': knee_correlation,
            'hip_correlation': hip_correlation,
            'left_knee_rom': left_knee_rom,
            'right_knee_rom': right_knee_rom,
            'left_hip_rom': left_hip_rom,
            'right_hip_rom': right_hip_rom
        }

    def _compute_symmetry_index(self, left_value: float, right_value: float) -> float:
        """Compute symmetry index using Robinson's formula.

        SI = 1 - |left - right| / (0.5 * (left + right))

        A value of 1.0 indicates perfect symmetry, 0.0 indicates complete asymmetry.

        Args:
            left_value: Metric value for left side
            right_value: Metric value for right side

        Returns:
            Symmetry index between 0 and 1
        """
        if left_value == 0.0 and right_value == 0.0:
            return 1.0

        avg_value = 0.5 * (left_value + right_value)
        if avg_value == 0.0:
            return 0.0

        asymmetry = abs(left_value - right_value) / avg_value
        symmetry = 1.0 - min(asymmetry, 1.0)  # Clamp to [0, 1]

        return symmetry

    def _calculate_stride_times_for_foot(self, heel_strikes: List[GaitEvent]) -> List[float]:
        """Calculate stride times for a single foot.

        Args:
            heel_strikes: List of heel strike events for one foot

        Returns:
            List of stride times in seconds
        """
        stride_times = []
        for i in range(len(heel_strikes) - 1):
            stride_time = heel_strikes[i + 1].timestamp - heel_strikes[i].timestamp
            if 0.3 <= stride_time <= 2.0:  # Reasonable range
                stride_times.append(stride_time)
        return stride_times

    def _calculate_stance_times(self, heel_strikes: List[GaitEvent],
                                toe_offs: List[GaitEvent]) -> List[float]:
        """Calculate stance phase durations.

        Args:
            heel_strikes: List of heel strike events
            toe_offs: List of toe off events

        Returns:
            List of stance times in seconds
        """
        stance_times = []
        for hs in heel_strikes:
            next_toe_offs = [to for to in toe_offs if to.timestamp > hs.timestamp]
            if next_toe_offs:
                stance_time = next_toe_offs[0].timestamp - hs.timestamp
                if 0.05 <= stance_time <= 0.5:
                    stance_times.append(stance_time)
        return stance_times

    def _calculate_swing_times(self, toe_offs: List[GaitEvent],
                               heel_strikes: List[GaitEvent]) -> List[float]:
        """Calculate swing phase durations.

        Args:
            toe_offs: List of toe off events
            heel_strikes: List of heel strike events

        Returns:
            List of swing times in seconds
        """
        swing_times = []
        for to in toe_offs:
            next_heel_strikes = [hs for hs in heel_strikes if hs.timestamp > to.timestamp]
            if next_heel_strikes:
                swing_time = next_heel_strikes[0].timestamp - to.timestamp
                if 0.1 <= swing_time <= 0.8:
                    swing_times.append(swing_time)
        return swing_times

    def _calculate_stride_lengths_for_foot(self, landmarks_sequence: List[Dict[str, Any]],
                                          heel_strikes: List[GaitEvent],
                                          ankle_key: str) -> List[float]:
        """Calculate stride lengths for a single foot.

        Args:
            landmarks_sequence: List of pose landmarks
            heel_strikes: List of heel strike events for one foot
            ankle_key: Key for ankle landmark ('left_ankle' or 'right_ankle')

        Returns:
            List of stride lengths in normalized units
        """
        stride_lengths = []
        for i in range(len(heel_strikes) - 1):
            frame1 = heel_strikes[i].frame_number
            frame2 = heel_strikes[i + 1].frame_number

            if frame1 < len(landmarks_sequence) and frame2 < len(landmarks_sequence):
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
                    stride_length = np.linalg.norm(pos2 - pos1)
                    stride_lengths.append(stride_length)

        return stride_lengths

    def _calculate_step_widths(self, landmarks_sequence: List[Dict[str, Any]],
                              events: List[GaitEvent]) -> List[float]:
        """Calculate step widths (lateral distance between feet at heel strike).

        Args:
            landmarks_sequence: List of pose landmarks
            events: List of gait events

        Returns:
            List of step widths in normalized units
        """
        step_widths = []

        heel_strikes = [e for e in events if e.event_type == GaitEventType.HEEL_STRIKE]

        for event in heel_strikes:
            frame = event.frame_number
            if frame < len(landmarks_sequence):
                landmarks = landmarks_sequence[frame]

                if 'left_ankle' in landmarks and 'right_ankle' in landmarks:
                    left_x = landmarks['left_ankle']['x']
                    right_x = landmarks['right_ankle']['x']
                    step_width = abs(left_x - right_x)
                    step_widths.append(step_width)

        return step_widths

    def compare_sides(self, left_metrics: Dict[str, float],
                     right_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Compare left and right side metrics directly.

        Args:
            left_metrics: Dictionary of left side metrics
            right_metrics: Dictionary of right side metrics

        Returns:
            Dictionary with comparison results and symmetry scores
        """
        comparisons = {}

        # Compare each common metric
        for key in left_metrics:
            if key in right_metrics:
                left_val = left_metrics[key]
                right_val = right_metrics[key]

                symmetry = self._compute_symmetry_index(left_val, right_val)
                difference = right_val - left_val
                percent_diff = (difference / left_val * 100) if left_val != 0 else 0.0

                comparisons[key] = {
                    'left': left_val,
                    'right': right_val,
                    'difference': difference,
                    'percent_difference': percent_diff,
                    'symmetry_index': symmetry
                }

        return comparisons
