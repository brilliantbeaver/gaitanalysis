# Visualization Module

The visualization module provides comprehensive tools for visualizing running gait analysis results, including pose overlays, metric plots, and report generation.

## Components

### 1. PoseOverlay (`overlay.py`)

Class for drawing pose landmarks and connections on video frames using OpenCV.

**Features:**
- Draw pose landmarks with visibility thresholding
- Draw skeleton connections (MediaPipe format)
- Visualize gait events (foot strike, toe-off, etc.)
- Display joint angle measurements
- Add information panels to frames

**Example Usage:**

```python
from asdrp.visualization import PoseOverlay, LandmarkPoint, GaitEvent

# Initialize overlay
overlay = PoseOverlay(
    landmark_color=(0, 255, 0),
    connection_color=(255, 255, 255),
    landmark_radius=5,
)

# Draw pose on frame
output_frame = overlay.draw_pose(frame, landmarks)

# Draw gait events
events = {
    'left_foot': GaitEvent.FOOT_STRIKE,
    'right_foot': GaitEvent.TOE_OFF,
}
output_frame = overlay.draw_gait_events(output_frame, events, landmarks)

# Draw joint angle
output_frame = overlay.draw_angle(
    output_frame,
    landmarks,
    joint_indices=(hip_idx, knee_idx, ankle_idx),
    angle_value=145.0,
    label="Knee Angle"
)
```

**Key Classes:**
- `PoseOverlay`: Main class for drawing pose visualizations
- `LandmarkPoint`: Dataclass representing a single pose landmark
- `GaitEvent`: Enum for gait event types

**Constants:**
- `POSE_LANDMARKS`: Dictionary mapping landmark names to indices
- `POSE_CONNECTIONS`: List of landmark pairs defining the skeleton

### 2. MetricsPlotter (`plots.py`)

Class for creating publication-quality plots of gait metrics using matplotlib and seaborn.

**Features:**
- Plot joint angles over time
- Visualize gait cycles (normalized 0-100%)
- Symmetry analysis plots (scatter, correlation, difference)
- Stride metrics summary plots
- Correlation heatmaps
- Multi-condition comparisons

**Example Usage:**

```python
from asdrp.visualization import MetricsPlotter
import pandas as pd

# Initialize plotter
plotter = MetricsPlotter(figsize=(12, 8), dpi=150)

# Plot angles over time
fig = plotter.plot_angles_over_time(
    data,
    angles=['left_knee_angle', 'right_knee_angle'],
    title="Joint Angles During Running",
    save_path="angles.png"
)

# Plot gait cycle
fig = plotter.plot_gait_cycle(
    cycle_data,
    angle_columns=['left_knee_angle'],
    show_phases=True,
    save_path="gait_cycle.png"
)

# Plot symmetry analysis
fig = plotter.plot_symmetry(
    data,
    left_column='left_knee_angle',
    right_column='right_knee_angle',
    save_path="symmetry.png"
)

# Plot stride metrics
stride_metrics = {
    'Stride Time (s)': [0.72, 0.68, 0.75, ...],
    'Cadence (spm)': [175, 180, 172, ...],
}
fig = plotter.plot_stride_metrics(
    stride_metrics,
    save_path="stride_metrics.png"
)
```

**Available Plot Types:**
- `plot_angles_over_time()`: Time series of joint angles
- `plot_gait_cycle()`: Angles across normalized gait cycle
- `plot_symmetry()`: Left vs right comparison (scatter, time series, histogram)
- `plot_stride_metrics()`: Box plots with statistics
- `plot_heatmap()`: Correlation heatmap
- `plot_comparison()`: Multi-condition comparison (violin + box plots)

### 3. ReportGenerator (`report.py`)

Class for generating comprehensive analysis reports in HTML and text formats.

**Features:**
- Professional HTML reports with embedded visualizations
- Text summaries for quick review
- Automatic metric interpretation
- Recommendations based on findings
- Responsive design for viewing on any device

**Example Usage:**

```python
from asdrp.visualization import ReportGenerator

# Initialize report generator
report_gen = ReportGenerator(
    project_name="Running Gait Analysis",
    author="Dr. Smith"
)

# Define metrics
metrics = {
    'temporal': {
        'stride_time': {'mean': 0.72, 'std': 0.05},
        'cadence': {'mean': 175.5, 'std': 4.2},
    },
    'kinematics': {
        'left_knee_angle': {'mean': 142.5, 'std': 8.3},
    },
    'symmetry': {
        'overall_symmetry_index': {'mean': 96.5},
    }
}

# Generate HTML report
html_content = report_gen.generate_html(
    metrics,
    figures={'Angles': fig1, 'Gait Cycle': fig2},
    summary_text="Analysis shows good running form.",
    output_path="report.html"
)

# Generate text summary
summary = report_gen.generate_summary(
    metrics,
    output_path="summary.txt"
)
```

**Report Sections:**
- Executive Summary
- Key Metrics (visual cards)
- Visualizations (embedded figures)
- Detailed Tables
- Interpretation & Recommendations
- Metadata (author, date, etc.)

## Data Structures

### LandmarkPoint

```python
@dataclass
class LandmarkPoint:
    x: float              # Normalized x coordinate (0-1)
    y: float              # Normalized y coordinate (0-1)
    z: float              # Normalized z coordinate (depth)
    visibility: float     # Visibility score (0-1)
```

### Metrics Dictionary Format

```python
metrics = {
    'temporal': {
        'metric_name': {
            'mean': float,
            'std': float,
            'min': float,
            'max': float,
        }
    },
    'kinematics': {
        'joint_angle': {...}
    },
    'symmetry': {
        'symmetry_index': float or dict
    }
}
```

## Color Schemes

### Default Colors

**Body Sides:**
- Left: `#2E86AB` (Blue)
- Right: `#A23B72` (Purple/Red)

**Joint Angles:**
- Hip: `#E63946` (Red)
- Knee: `#2A9D8F` (Teal)
- Ankle: `#E9C46A` (Yellow)
- Trunk: `#264653` (Dark Blue)

**Gait Events:**
- Foot Strike: Red `(0, 0, 255)`
- Toe Off: Blue `(255, 0, 0)`
- Mid Stance: Yellow `(0, 255, 255)`
- Mid Swing: Magenta `(255, 0, 255)`

## Best Practices

### Pose Overlay
- Use visibility threshold to filter unreliable landmarks
- Highlight key landmarks during specific gait phases
- Add information panels for context (frame number, time, metrics)
- Use contrasting colors for overlays on different video backgrounds

### Plotting
- Always include error bars or standard deviation bands
- Use consistent color schemes across related plots
- Add gait phase regions to cycle plots for context
- Save plots at high DPI (150+) for publications
- Include both mean trends and individual data points

### Reports
- Provide both HTML (detailed) and text (quick) versions
- Embed high-quality figures in HTML reports
- Include interpretation and actionable recommendations
- Use consistent metric naming across all outputs
- Add metadata (date, author, software version)

## Dependencies

- OpenCV (`cv2`): Video frame manipulation and drawing
- NumPy: Numerical computations
- Pandas: Data manipulation
- Matplotlib: Base plotting library
- Seaborn: Statistical visualization
- Base64: Image encoding for HTML reports

## Examples

See `/examples/visualization_example.py` for complete working examples of all visualization components.

## Notes

- All coordinates are in normalized format (0-1) and converted to pixels internally
- Figures are automatically closed after saving to prevent memory leaks
- HTML reports are fully self-contained with embedded CSS and images
- All visualization functions support optional file saving

## Future Enhancements

Potential additions:
- 3D pose visualization using matplotlib 3D
- Animated GIF generation from frame sequences
- Interactive HTML reports with plotly
- PDF report generation
- Video overlay with progress bar
- Real-time visualization streaming
