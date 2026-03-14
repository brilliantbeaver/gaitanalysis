"""
Visualization module for running gait analysis.

This module provides tools for visualizing pose landmarks, plotting gait metrics,
and generating comprehensive analysis reports.
"""

from .overlay import (
    PoseOverlay,
    LandmarkPoint,
    GaitEvent,
    POSE_LANDMARKS,
    POSE_CONNECTIONS,
)
from .plots import MetricsPlotter
from .report import ReportGenerator

__all__ = [
    # Overlay classes and constants
    'PoseOverlay',
    'LandmarkPoint',
    'GaitEvent',
    'POSE_LANDMARKS',
    'POSE_CONNECTIONS',
    # Plotting
    'MetricsPlotter',
    # Reporting
    'ReportGenerator',
]

__version__ = '0.1.0'
