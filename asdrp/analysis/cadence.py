"""Cadence analysis for running gait.

This module calculates cadence (steps per minute) from gait events or
directly from pose landmark sequences.
"""

from typing import Dict, List, Any, Optional
import numpy as np

from .metrics import BaseMetricCalculator, GaitEvent, GaitEventType, Foot


class CadenceAnalyzer(BaseMetricCalculator):
    """Analyzer for calculating running cadence.

    Cadence is measured as the number of steps per minute. A step occurs
    at each heel strike or foot contact. For running, typical cadence ranges
    from 160-180 steps per minute for recreational runners, and 180-200+
    for elite runners.
    """

    def __init__(self, fps: float = 30.0):
        """Initialize cadence analyzer.

        Args:
            fps: Video frame rate (frames per second)
        """
        super().__init__(fps)

    def calculate(self, landmarks_sequence: List[Dict[str, Any]],
                  events: Optional[List[GaitEvent]] = None) -> Dict[str, Any]:
        """Calculate cadence from gait events or landmark sequence.

        If gait events are provided, uses heel strike events to calculate cadence.
        Otherwise, estimates cadence from vertical foot motion.

        Args:
            landmarks_sequence: List of pose landmarks for each frame
            events: Optional list of detected gait events

        Returns:
            Dictionary containing:
                - cadence: Overall cadence in steps per minute
                - left_cadence: Cadence for left foot
                - right_cadence: Cadence for right foot
                - step_count: Total number of steps detected
                - avg_step_time: Average time between steps (seconds)
        """
        if events is not None and len(events) > 0:
            # Use provided events (preferred method)
            result = self._calculate_from_events(events, landmarks_sequence)
        else:
            # Estimate from landmarks
            result = self._estimate_from_landmarks(landmarks_sequence)

        return result

    def _calculate_from_events(self, events: List[GaitEvent],
                               landmarks_sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate cadence from detected gait events.

        Args:
            events: List of gait events (heel strikes and toe offs)
            landmarks_sequence: Landmark sequence for duration calculation

        Returns:
            Dictionary of cadence metrics
        """
        # Filter to only heel strike events
        heel_strikes = [e for e in events if e.event_type == GaitEventType.HEEL_STRIKE]

        if len(heel_strikes) < 2:
            return {
                'cadence': 0.0,
                'left_cadence': 0.0,
                'right_cadence': 0.0,
                'step_count': len(heel_strikes),
                'avg_step_time': 0.0
            }

        # Calculate total duration
        duration = len(landmarks_sequence) / self.fps

        # Total steps
        total_steps = len(heel_strikes)

        # Calculate overall cadence (steps per minute)
        cadence = (total_steps / duration) * 60.0 if duration > 0 else 0.0

        # Calculate separate cadence for each foot
        left_strikes = [e for e in heel_strikes if e.foot == Foot.LEFT]
        right_strikes = [e for e in heel_strikes if e.foot == Foot.RIGHT]

        left_cadence = (len(left_strikes) / duration) * 60.0 if duration > 0 else 0.0
        right_cadence = (len(right_strikes) / duration) * 60.0 if duration > 0 else 0.0

        # Calculate average time between steps
        step_times = []
        sorted_strikes = sorted(heel_strikes, key=lambda e: e.timestamp)
        for i in range(len(sorted_strikes) - 1):
            step_time = sorted_strikes[i + 1].timestamp - sorted_strikes[i].timestamp
            step_times.append(step_time)

        avg_step_time = np.mean(step_times) if step_times else 0.0

        return {
            'cadence': cadence,
            'left_cadence': left_cadence,
            'right_cadence': right_cadence,
            'step_count': total_steps,
            'avg_step_time': avg_step_time
        }

    def _estimate_from_landmarks(self, landmarks_sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estimate cadence directly from foot position analysis.

        This method is used as a fallback when gait events are not provided.
        It analyzes vertical foot motion to estimate step frequency.

        Args:
            landmarks_sequence: List of pose landmarks for each frame

        Returns:
            Dictionary of estimated cadence metrics
        """
        from scipy.signal import find_peaks, savgol_filter

        # Extract foot vertical positions
        left_foot_y = []
        right_foot_y = []

        for landmarks in landmarks_sequence:
            # Use ankle landmarks as proxy for foot position
            if 'left_ankle' in landmarks:
                left_foot_y.append(landmarks['left_ankle']['y'])
            else:
                left_foot_y.append(left_foot_y[-1] if left_foot_y else 0.0)

            if 'right_ankle' in landmarks:
                right_foot_y.append(landmarks['right_ankle']['y'])
            else:
                right_foot_y.append(right_foot_y[-1] if right_foot_y else 0.0)

        left_foot_y = np.array(left_foot_y)
        right_foot_y = np.array(right_foot_y)

        # Smooth signals
        if len(left_foot_y) > 5:
            window_length = min(11, len(left_foot_y) if len(left_foot_y) % 2 == 1 else len(left_foot_y) - 1)
            left_foot_y = savgol_filter(left_foot_y, window_length, 3)
            right_foot_y = savgol_filter(right_foot_y, window_length, 3)

        # Find peaks (foot contacts - highest y values in image coordinates)
        min_distance = int(0.3 * self.fps)  # Minimum 0.3s between steps

        left_peaks, _ = find_peaks(left_foot_y, distance=min_distance, prominence=0.01)
        right_peaks, _ = find_peaks(right_foot_y, distance=min_distance, prominence=0.01)

        # Calculate cadence
        duration = len(landmarks_sequence) / self.fps
        total_steps = len(left_peaks) + len(right_peaks)

        cadence = (total_steps / duration) * 60.0 if duration > 0 else 0.0
        left_cadence = (len(left_peaks) / duration) * 60.0 if duration > 0 else 0.0
        right_cadence = (len(right_peaks) / duration) * 60.0 if duration > 0 else 0.0

        # Estimate average step time
        all_peaks = sorted(list(left_peaks) + list(right_peaks))
        step_times = []
        for i in range(len(all_peaks) - 1):
            step_time = (all_peaks[i + 1] - all_peaks[i]) / self.fps
            step_times.append(step_time)

        avg_step_time = np.mean(step_times) if step_times else 0.0

        return {
            'cadence': cadence,
            'left_cadence': left_cadence,
            'right_cadence': right_cadence,
            'step_count': total_steps,
            'avg_step_time': avg_step_time
        }

    def calculate_instantaneous_cadence(self, events: List[GaitEvent],
                                       window_size: float = 10.0) -> List[Dict[str, float]]:
        """Calculate instantaneous cadence over sliding time windows.

        This method provides cadence as a time series, useful for analyzing
        changes in running rhythm over time.

        Args:
            events: List of gait events
            window_size: Size of time window in seconds for calculating
                        instantaneous cadence

        Returns:
            List of dictionaries with 'timestamp' and 'cadence' keys
        """
        heel_strikes = [e for e in events if e.event_type == GaitEventType.HEEL_STRIKE]

        if len(heel_strikes) < 2:
            return []

        # Sort by timestamp
        heel_strikes.sort(key=lambda e: e.timestamp)

        instantaneous_cadence = []

        # Calculate cadence for each time window
        start_time = heel_strikes[0].timestamp
        end_time = heel_strikes[-1].timestamp

        # Create time windows
        current_time = start_time
        while current_time <= end_time:
            # Find events in this window
            window_start = current_time
            window_end = current_time + window_size

            events_in_window = [
                e for e in heel_strikes
                if window_start <= e.timestamp < window_end
            ]

            # Calculate cadence for this window
            if len(events_in_window) >= 2:
                steps_in_window = len(events_in_window)
                cadence = (steps_in_window / window_size) * 60.0

                instantaneous_cadence.append({
                    'timestamp': current_time + window_size / 2,  # Center of window
                    'cadence': cadence
                })

            # Move to next window (with 50% overlap)
            current_time += window_size / 2

        return instantaneous_cadence

    def analyze_cadence_variability(self, events: List[GaitEvent]) -> Dict[str, float]:
        """Analyze variability in cadence over time.

        High variability may indicate inconsistent running rhythm or fatigue.

        Args:
            events: List of gait events

        Returns:
            Dictionary containing:
                - mean_cadence: Average cadence
                - std_cadence: Standard deviation of step times
                - cv_cadence: Coefficient of variation (CV)
                - variability_index: Normalized variability measure
        """
        heel_strikes = [e for e in events if e.event_type == GaitEventType.HEEL_STRIKE]
        heel_strikes.sort(key=lambda e: e.timestamp)

        if len(heel_strikes) < 3:
            return {
                'mean_cadence': 0.0,
                'std_cadence': 0.0,
                'cv_cadence': 0.0,
                'variability_index': 0.0
            }

        # Calculate step times
        step_times = []
        for i in range(len(heel_strikes) - 1):
            step_time = heel_strikes[i + 1].timestamp - heel_strikes[i].timestamp
            step_times.append(step_time)

        step_times = np.array(step_times)

        # Calculate statistics
        mean_step_time = np.mean(step_times)
        std_step_time = np.std(step_times)
        cv = (std_step_time / mean_step_time) * 100 if mean_step_time > 0 else 0.0

        # Convert to cadence
        mean_cadence = (1.0 / mean_step_time) * 60.0 if mean_step_time > 0 else 0.0

        # Variability index (normalized standard deviation)
        variability_index = cv / 100.0

        return {
            'mean_cadence': mean_cadence,
            'std_cadence': std_step_time,
            'cv_cadence': cv,
            'variability_index': variability_index
        }
