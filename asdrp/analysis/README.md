# Gait Analysis Module

Comprehensive biomechanics analysis for running gait using MediaPipe Pose landmarks.

## Overview

This module provides tools for analyzing running gait patterns, detecting gait events, and computing biomechanical metrics from video data processed with MediaPipe Pose Landmarker. It implements state-of-the-art biomechanics algorithms with a clean, extensible architecture.

## Features

- **Event Detection**: Automatic detection of heel strikes and toe offs using multi-criteria kinematic algorithms
- **Stride Analysis**: Calculate stride length, stride time, stance/swing phases
- **Cadence Calculation**: Measure running rhythm (steps per minute) with primary and fallback methods
- **Symmetry Assessment**: Compare left vs right side biomechanics across temporal, spatial, and kinematic domains
- **Joint Angle Analysis**: Compute knee and hip angles throughout gait cycle
- **Signal Processing**: Savitzky-Golay filtering, peak detection with physiological constraints, missing data interpolation, outlier filtering
- **Statistical Analysis**: Mean/SD, coefficient of variation, Pearson correlation for pattern matching, multiple symmetry index formulas
- **Error Handling**: Graceful handling of missing landmarks, validation of physiological ranges, fallback algorithms when primary methods fail

## Module Structure

```
analysis/
├── __init__.py          # Module exports and public API
├── metrics.py           # Core data structures, base classes, and orchestrator
├── stride.py            # Stride detection and analysis
├── cadence.py           # Cadence calculation
├── symmetry.py          # Bilateral symmetry assessment
└── README.md            # This documentation
```

### metrics.py

Core data structures and the analysis orchestrator.

- `GaitEventType` (Enum): HEEL_STRIKE, TOE_OFF, MID_STANCE, MID_SWING
- `Foot` (Enum): LEFT, RIGHT
- `GaitEvent` (dataclass): A single gait event with timestamp, frame number, foot, and landmark data
- `GaitMetrics` (dataclass): Comprehensive container for all gait analysis results
- `BaseMetricCalculator` (ABC): Abstract base for all analyzers — provides angle calculation, coordinate extraction, and enforces a consistent `calculate()` interface
- `GaitAnalyzer`: Coordinates multiple calculators in sequence, aggregates results into a unified `GaitMetrics` object, and passes detected events between calculators

### stride.py

Stride detection and analysis using kinematic algorithms. Detects heel strikes (vertical position + velocity + foot-hip alignment with Savitzky-Golay filtering and `scipy.signal.find_peaks`) and toe offs (vertical acceleration and position transitions). Computes stride time, stride length, stance/swing phase durations per side. Enforces physiological range validation (0.3–2.0 s stride time) and filters outliers.

### cadence.py

Cadence calculation with two methods: a primary event-based method (`_calculate_from_events`) that counts heel strikes over duration, and a fallback landmark-based method (`_estimate_from_landmarks`) that analyzes vertical foot oscillation via peak detection. Also provides `calculate_instantaneous_cadence()` (sliding-window time series) and `analyze_cadence_variability()` (mean, SD, CV, variability index).

### symmetry.py

Bilateral symmetry assessment across three domains — temporal (stride/stance/swing time), spatial (stride length, step width, lateral deviation), and kinematic (joint angle ROM, knee flexion and hip extension patterns via cross-correlation with Pearson coefficient). Uses Robinson's Symmetry Index and combines sub-indices into an overall symmetry score.

## Quick Start

```python
from asdrp.analysis import (
    GaitAnalyzer,
    StrideAnalyzer,
    CadenceAnalyzer,
    SymmetryAnalyzer
)

# Initialize analyzer
analyzer = GaitAnalyzer(fps=30.0)

# Add calculators
analyzer.add_calculator(StrideAnalyzer(fps=30.0))
analyzer.add_calculator(CadenceAnalyzer(fps=30.0))
analyzer.add_calculator(SymmetryAnalyzer(fps=30.0))

# Analyze (landmarks_sequence from MediaPipe)
metrics = analyzer.analyze(landmarks_sequence)

# Access results
print(f"Cadence: {metrics.cadence:.1f} steps/min")
print(f"Stride Time: {metrics.stride_time:.3f} s")
print(f"Symmetry Index: {metrics.symmetry_index:.3f}")
print(f"Events Detected: {len(metrics.events)}")

# Export
metrics_dict = metrics.to_dict()
```

## Data Structures

### GaitEvent

Represents a single gait event (heel strike or toe off).

**Attributes:**
- `event_type`: Type of event (HEEL_STRIKE, TOE_OFF, MID_STANCE, MID_SWING)
- `timestamp`: Time in seconds
- `frame_number`: Video frame number
- `foot`: Which foot (LEFT or RIGHT)
- `landmark_data`: Relevant pose landmarks at this moment

### GaitMetrics

Comprehensive gait analysis results.

**Key Metrics:**
- `cadence`: Steps per minute
- `stride_length`: Average stride length (m)
- `stride_time`: Average stride duration (s)
- `stance_phase_duration`: Time foot is on ground (s)
- `swing_phase_duration`: Time foot is in air (s)
- `knee_flexion_max`: Maximum knee bend angle (degrees)
- `hip_extension_max`: Maximum hip extension angle (degrees)
- `symmetry_index`: Left-right symmetry (0-1, 1=perfect)
- `events`: List of all detected gait events
- `left_metrics`: Left side specific metrics
- `right_metrics`: Right side specific metrics

## Analyzers

### StrideAnalyzer

Detects gait events and calculates stride metrics.

**Algorithm:**
- Uses vertical foot position and velocity for heel strike detection
- Identifies toe off events from vertical acceleration
- Calculates stride timing and spatial metrics
- Separates stance and swing phases
- Validates physiological ranges (0.3–2.0 s stride time), filters outliers, and handles missing landmarks gracefully

**Key Methods:**
- `detect_heel_strikes()`: Find initial contact events
- `detect_toe_offs()`: Find end of contact events
- `calculate()`: Compute all stride metrics

**Metric Calculations:**
- Stride time: Time between consecutive heel strikes of same foot
- Stride length: Euclidean distance traveled between strikes
- Stance phase: Heel strike to toe off duration
- Swing phase: Toe off to next heel strike duration

### CadenceAnalyzer

Calculates running cadence (steps per minute).

**Primary Method** (`_calculate_from_events`):
- Counts heel strikes over analysis duration
- Computes separate cadence for each foot
- Calculates average step time

**Fallback Method** (`_estimate_from_landmarks`):
- Direct landmark analysis when events are unavailable
- Analyzes vertical foot oscillation
- Uses peak detection on smoothed signals
- Estimates step frequency

**Typical Values:**
- Recreational runners: 160–180 steps/min
- Elite runners: 180–200+ steps/min

**Key Methods:**
- `calculate()`: Compute overall cadence
- `calculate_instantaneous_cadence()`: Cadence time series (sliding window)
- `analyze_cadence_variability()`: Rhythm consistency metrics (mean, SD, CV, variability index)

### SymmetryAnalyzer

Assesses bilateral gait symmetry across three domains.

**Three-Component Analysis:**

1. **Temporal Symmetry**: Stride time, stance time, swing time
2. **Spatial Symmetry**: Stride length, step width, lateral deviation
3. **Kinematic Symmetry**: Joint angle ROM, knee flexion patterns, hip extension patterns, cross-correlation of angle time series (Pearson coefficient)

**Algorithm:**
- Uses Robinson's Symmetry Index: SI = 1 - |L - R| / (0.5 * (L + R))
- Compares range of motion between sides
- Calculates correlation between left/right angle patterns
- Combines sub-indices into overall symmetry index

**Key Methods:**
- `calculate()`: Compute all symmetry metrics
- `compare_sides()`: Direct left-right comparison with percentage differences
- `_compute_symmetry_index()`: Standardized symmetry calculation

## Biomechanics Background

### Gait Cycle Phases

1. **Stance Phase** (30-40% of cycle in running)
   - Initial Contact (heel strike)
   - Mid-stance
   - Toe Off

2. **Swing Phase** (60-70% of cycle in running)
   - Initial Swing
   - Mid-swing
   - Terminal Swing

### Normal Running Gait Values

- **Cadence**: 160-200 steps/min (elite: 180-200)
- **Stride Time**: 0.6-0.8 seconds
- **Stance Phase**: 0.2-0.3 seconds
- **Swing Phase**: 0.3-0.5 seconds
- **Knee Flexion**: 120-140 degrees during swing
- **Hip Extension**: 180-185 degrees during stance

### Symmetry Interpretation

- **SI > 0.95**: Excellent symmetry
- **SI 0.90-0.95**: Good symmetry
- **SI 0.85-0.90**: Moderate asymmetry
- **SI < 0.85**: Significant asymmetry (may indicate issues)

## Input Data Format

The analyzers expect pose landmark data in this format:

```python
landmarks_sequence = [
    {  # Frame 0
        'left_hip': {'x': 0.45, 'y': 0.55, 'z': -0.1, 'visibility': 0.99},
        'left_knee': {'x': 0.44, 'y': 0.72, 'z': -0.05, 'visibility': 0.98},
        'left_ankle': {'x': 0.43, 'y': 0.88, 'z': 0.0, 'visibility': 0.97},
        'left_heel': {'x': 0.42, 'y': 0.90, 'z': 0.01, 'visibility': 0.96},
        'left_foot_index': {'x': 0.45, 'y': 0.91, 'z': 0.02, 'visibility': 0.95},
        'right_hip': {'x': 0.55, 'y': 0.55, 'z': -0.1, 'visibility': 0.99},
        # ... more landmarks ...
    },
    # Frame 1, Frame 2, ...
]
```

**Required Landmarks:**
- Hip: `left_hip`, `right_hip`
- Knee: `left_knee`, `right_knee`
- Ankle: `left_ankle`, `right_ankle`
- Foot: `left_heel`, `right_heel`, `left_foot_index`, `right_foot_index`

**Optional for Enhanced Analysis:**
- Shoulder: `left_shoulder`, `right_shoulder`

## Advanced Usage

### Custom Metric Calculators

Create your own analyzer by extending `BaseMetricCalculator`:

```python
from asdrp.analysis import BaseMetricCalculator

class CustomAnalyzer(BaseMetricCalculator):
    def calculate(self, landmarks_sequence, events=None):
        # Your analysis logic here
        return {
            'custom_metric': value,
            # ... more metrics
        }

# Use it
analyzer = GaitAnalyzer(fps=30.0)
analyzer.add_calculator(CustomAnalyzer(fps=30.0))
```

### Export Results

```python
# Get metrics as dictionary
metrics_dict = metrics.to_dict()

# Save to JSON
import json
with open('gait_metrics.json', 'w') as f:
    json.dump(metrics_dict, f, indent=2)

# Convert to DataFrame
import pandas as pd
df = pd.DataFrame([metrics_dict])
df.to_csv('gait_metrics.csv', index=False)
```

### Analyze Specific Events

```python
# Get individual events
for event in metrics.events:
    print(f"{event.event_type.value} - {event.foot.value} - "
          f"Frame {event.frame_number} at {event.timestamp:.3f}s")

# Filter by type
heel_strikes = [e for e in metrics.events
                if e.event_type == GaitEventType.HEEL_STRIKE]
```

## Algorithm Details

### Heel Strike Detection

Uses multi-criteria approach:
1. Vertical foot position (local maximum in y-coordinate)
2. Foot alignment with hip (x-coordinate proximity)
3. Signal filtering with Savitzky-Golay filter
4. Peak detection with prominence and distance constraints

### Symmetry Index Calculation

Robinson's formula:
```
SI = 1 - |Left - Right| / (0.5 * (Left + Right))
```

Where:
- SI = 1.0 indicates perfect symmetry
- SI = 0.0 indicates complete asymmetry
- Values clamped to [0, 1] range

### Joint Angle Calculation

Uses vector dot product method:
```python
angle = arccos((v1 · v2) / (|v1| * |v2|))
```

Where v1 and v2 are vectors from joint center to adjacent landmarks.

## Architecture & Design

- **Separation of Concerns**: Events, metrics, and calculators are cleanly separated
- **Extensibility**: The `BaseMetricCalculator` ABC pattern makes it straightforward to add new analyzers
- **Type Safety**: Full type annotations and dataclasses for immutable data structures
- **Orchestration**: `GaitAnalyzer` manages calculator sequencing and result aggregation
- **Robustness**: Reasonable defaults for edge cases, fallback algorithms when primary methods fail

## Integration Points

The module is designed to integrate with the rest of the pipeline:

1. **Video Module** (`asdrp.video`): Receives video frames
2. **Pose Module** (`asdrp.pose`): Receives MediaPipe landmarks via `LandmarkProcessor.to_dict()`
3. **Visualization Module** (`asdrp.visualization`): Provides metrics for plotting and overlay
4. **Utils Module** (`asdrp.utils`): Uses shared geometry and smoothing utilities

## Performance Considerations

- **Frame Rate**: Higher frame rates (60+ fps) improve event detection accuracy
- **Video Quality**: Clear visibility of lower body landmarks is essential
- **Camera Angle**: Side view provides best results for sagittal plane analysis
- **Lighting**: Consistent lighting helps MediaPipe tracking
- **Duration**: Minimum 3-5 strides recommended for reliable statistics
- **Speed**: ~1000 frames/sec processing (hardware-dependent)
- **Memory**: O(n) where n is number of frames

## Testing Recommendations

1. **Unit Tests**: Event detection accuracy, angle calculation precision, symmetry index formulas
2. **Integration Tests**: Full pipeline with synthetic data, edge cases (very fast/slow running), missing landmark handling
3. **Validation Tests**: Comparison with motion capture systems, manual annotation correlation, inter-rater reliability

## Validation

The algorithms have been designed based on established biomechanics research:

- Zeni et al. (2008) - Heel strike/toe off detection
- Robinson et al. (1987) - Symmetry index
- Novacheck (1998) - Running gait kinematics
- Cavanagh & LaFortune (1980) - Ground reaction forces

## Limitations

- Requires clear visibility of lower body landmarks
- Assumes sagittal plane motion (side view)
- Stride length in normalized units (requires calibration for absolute values)
- Does not include kinetic measurements (ground reaction forces)
- Best suited for steady-state running (not acceleration/deceleration)

## Future Enhancements

Potential additions:
- 3D kinematic analysis
- Joint power and work calculations
- Footstrike pattern classification (heel/midfoot/forefoot)
- Ground contact time estimation
- Vertical oscillation analysis
- Running economy metrics
- Fatigue detection algorithms
- Spatial calibration for absolute measurements
- Tutorial notebooks

## References

1. Zeni, J. A., Richards, J. G., & Higginson, J. S. (2008). Two simple methods for determining gait events during treadmill and overground walking using kinematic data. Gait & Posture, 27(4), 710-714.

2. Robinson, R. O., Herzog, W., & Nigg, B. M. (1987). Use of force platform variables to quantify the effects of chiropractic manipulation on gait symmetry. Journal of Manipulative and Physiological Therapeutics, 10(4), 172-176.

3. Novacheck, T. F. (1998). The biomechanics of running. Gait & Posture, 7(1), 77-95.

4. Cavanagh, P. R., & LaFortune, M. A. (1980). Ground reaction forces in distance running. Journal of Biomechanics, 13(5), 397-406.
