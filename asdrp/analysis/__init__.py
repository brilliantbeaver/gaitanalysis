"""Gait analysis module for running biomechanics.

This module provides comprehensive gait analysis tools including:
- Event detection (heel strikes, toe offs)
- Stride analysis (length, timing, phases)
- Cadence calculation
- Symmetry assessment
- Joint angle analysis

Example usage:
    >>> from asdrp.analysis import GaitAnalyzer, StrideAnalyzer, CadenceAnalyzer, SymmetryAnalyzer
    >>>
    >>> # Initialize analyzer
    >>> analyzer = GaitAnalyzer(fps=30.0)
    >>> analyzer.add_calculator(StrideAnalyzer(fps=30.0))
    >>> analyzer.add_calculator(CadenceAnalyzer(fps=30.0))
    >>> analyzer.add_calculator(SymmetryAnalyzer(fps=30.0))
    >>>
    >>> # Analyze landmarks
    >>> metrics = analyzer.analyze(landmarks_sequence)
    >>> print(f"Cadence: {metrics.cadence:.1f} steps/min")
    >>> print(f"Symmetry: {metrics.symmetry_index:.3f}")
"""

from .metrics import (
    GaitEvent,
    GaitEventType,
    GaitMetrics,
    Foot,
    BaseMetricCalculator,
    GaitAnalyzer
)

from .stride import StrideAnalyzer
from .cadence import CadenceAnalyzer
from .symmetry import SymmetryAnalyzer

__all__ = [
    # Core data structures
    'GaitEvent',
    'GaitEventType',
    'GaitMetrics',
    'Foot',

    # Base classes
    'BaseMetricCalculator',
    'GaitAnalyzer',

    # Analyzers
    'StrideAnalyzer',
    'CadenceAnalyzer',
    'SymmetryAnalyzer',
]

__version__ = '0.1.0'
