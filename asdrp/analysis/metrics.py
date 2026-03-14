"""Core data structures and orchestrator for gait analysis.

This module defines the fundamental data structures for gait events and metrics,
as well as the base classes and orchestrator for performing gait analysis.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import numpy as np


class GaitEventType(Enum):
    """Types of gait events during running cycle."""
    HEEL_STRIKE = "heel_strike"  # Initial contact with ground
    TOE_OFF = "toe_off"  # End of contact with ground
    MID_STANCE = "mid_stance"  # Middle of stance phase
    MID_SWING = "mid_swing"  # Middle of swing phase


class Foot(Enum):
    """Foot identifier."""
    LEFT = "left"
    RIGHT = "right"


@dataclass
class GaitEvent:
    """Represents a single gait event during running.

    Attributes:
        event_type: Type of gait event (e.g., heel strike, toe off)
        timestamp: Time of event in seconds
        frame_number: Video frame number where event occurred
        foot: Which foot (left or right)
        landmark_data: Dictionary of relevant MediaPipe pose landmarks at this event
            Expected keys include: 'hip', 'knee', 'ankle', 'heel', 'toe'
            Values are dicts with 'x', 'y', 'z' coordinates and 'visibility'
    """
    event_type: GaitEventType
    timestamp: float
    frame_number: int
    foot: Foot
    landmark_data: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"GaitEvent({self.event_type.value}, {self.foot.value}, "
                f"frame={self.frame_number}, t={self.timestamp:.3f}s)")


@dataclass
class GaitMetrics:
    """Comprehensive gait analysis metrics for running.

    All temporal metrics are in seconds, spatial metrics in meters (when calibrated),
    angular metrics in degrees, and ratios/indices are dimensionless.

    Attributes:
        cadence: Steps per minute (total steps for both feet)
        stride_length: Average distance between successive heel strikes of same foot (m)
        stride_time: Average time between successive heel strikes of same foot (s)
        stance_phase_duration: Average time foot is in contact with ground (s)
        swing_phase_duration: Average time foot is in air (s)
        knee_flexion_max: Maximum knee flexion angle during swing phase (degrees)
        hip_extension_max: Maximum hip extension angle during stance phase (degrees)
        symmetry_index: Measure of left-right symmetry (0-1, where 1 is perfect symmetry)
        events: List of all detected gait events
        left_metrics: Side-specific metrics for left foot
        right_metrics: Side-specific metrics for right foot
        analysis_duration: Total duration of analyzed video (s)
        total_strides: Total number of complete strides detected
    """
    cadence: float = 0.0
    stride_length: float = 0.0
    stride_time: float = 0.0
    stance_phase_duration: float = 0.0
    swing_phase_duration: float = 0.0
    knee_flexion_max: float = 0.0
    hip_extension_max: float = 0.0
    symmetry_index: float = 0.0
    events: List[GaitEvent] = field(default_factory=list)
    left_metrics: Dict[str, float] = field(default_factory=dict)
    right_metrics: Dict[str, float] = field(default_factory=dict)
    analysis_duration: float = 0.0
    total_strides: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format for export."""
        return {
            'cadence': self.cadence,
            'stride_length': self.stride_length,
            'stride_time': self.stride_time,
            'stance_phase_duration': self.stance_phase_duration,
            'swing_phase_duration': self.swing_phase_duration,
            'knee_flexion_max': self.knee_flexion_max,
            'hip_extension_max': self.hip_extension_max,
            'symmetry_index': self.symmetry_index,
            'left_metrics': self.left_metrics,
            'right_metrics': self.right_metrics,
            'analysis_duration': self.analysis_duration,
            'total_strides': self.total_strides,
            'total_events': len(self.events)
        }

    def __repr__(self) -> str:
        return (f"GaitMetrics(cadence={self.cadence:.1f} spm, "
                f"stride_time={self.stride_time:.3f}s, "
                f"symmetry={self.symmetry_index:.3f})")


class BaseMetricCalculator(ABC):
    """Abstract base class for metric calculators.

    All metric calculators should inherit from this class and implement
    the calculate method to compute specific gait metrics.
    """

    def __init__(self, fps: float = 30.0):
        """Initialize calculator.

        Args:
            fps: Video frame rate (frames per second)
        """
        self.fps = fps

    @abstractmethod
    def calculate(self, landmarks_sequence: List[Dict[str, Any]],
                  events: Optional[List[GaitEvent]] = None) -> Dict[str, Any]:
        """Calculate specific gait metrics.

        Args:
            landmarks_sequence: List of pose landmarks for each frame
                Each entry is a dict with landmark names as keys and
                coordinate dicts (x, y, z, visibility) as values
            events: Optional list of previously detected gait events

        Returns:
            Dictionary of calculated metrics
        """
        pass

    def _calculate_angle(self, point1: np.ndarray, point2: np.ndarray,
                        point3: np.ndarray) -> float:
        """Calculate angle between three points using vectors.

        Args:
            point1: First point (e.g., hip) as [x, y, z]
            point2: Vertex point (e.g., knee) as [x, y, z]
            point3: Third point (e.g., ankle) as [x, y, z]

        Returns:
            Angle in degrees (0-180)
        """
        # Create vectors
        vector1 = point1 - point2
        vector2 = point3 - point2

        # Calculate angle using dot product
        cos_angle = np.dot(vector1, vector2) / (
            np.linalg.norm(vector1) * np.linalg.norm(vector2) + 1e-8
        )
        # Clip to valid range to avoid numerical errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))

        return angle

    def _extract_landmark_coords(self, landmark: Dict[str, float]) -> np.ndarray:
        """Extract coordinates from landmark dictionary.

        Args:
            landmark: Dictionary with 'x', 'y', 'z' keys

        Returns:
            NumPy array [x, y, z]
        """
        return np.array([landmark['x'], landmark['y'], landmark['z']])


class GaitAnalyzer:
    """Orchestrator class for comprehensive gait analysis.

    This class coordinates multiple metric calculators to perform
    complete gait analysis on running videos.
    """

    def __init__(self, fps: float = 30.0):
        """Initialize gait analyzer.

        Args:
            fps: Video frame rate (frames per second)
        """
        self.fps = fps
        self.calculators: List[BaseMetricCalculator] = []

    def add_calculator(self, calculator: BaseMetricCalculator) -> None:
        """Add a metric calculator to the analysis pipeline.

        Args:
            calculator: Instance of BaseMetricCalculator
        """
        self.calculators.append(calculator)

    def analyze(self, landmarks_sequence: List[Dict[str, Any]]) -> GaitMetrics:
        """Perform comprehensive gait analysis.

        Args:
            landmarks_sequence: List of pose landmarks for each frame
                Each entry is a dict with landmark names as keys

        Returns:
            GaitMetrics object containing all computed metrics
        """
        metrics = GaitMetrics()
        metrics.analysis_duration = len(landmarks_sequence) / self.fps

        # Run all calculators in sequence
        events = None
        all_results = {}

        for calculator in self.calculators:
            results = calculator.calculate(landmarks_sequence, events)
            all_results.update(results)

            # If this calculator detected events, use them for subsequent calculators
            if 'events' in results:
                events = results['events']

        # Aggregate results into metrics object
        self._aggregate_results(metrics, all_results)

        return metrics

    def _aggregate_results(self, metrics: GaitMetrics,
                          results: Dict[str, Any]) -> None:
        """Aggregate results from all calculators into metrics object.

        Args:
            metrics: GaitMetrics object to populate
            results: Dictionary of results from all calculators
        """
        # Map result keys to metric attributes
        metric_mapping = {
            'cadence': 'cadence',
            'stride_length': 'stride_length',
            'stride_time': 'stride_time',
            'stance_phase_duration': 'stance_phase_duration',
            'swing_phase_duration': 'swing_phase_duration',
            'knee_flexion_max': 'knee_flexion_max',
            'hip_extension_max': 'hip_extension_max',
            'symmetry_index': 'symmetry_index',
            'events': 'events',
            'left_metrics': 'left_metrics',
            'right_metrics': 'right_metrics',
            'total_strides': 'total_strides'
        }

        for result_key, metric_attr in metric_mapping.items():
            if result_key in results:
                setattr(metrics, metric_attr, results[result_key])
