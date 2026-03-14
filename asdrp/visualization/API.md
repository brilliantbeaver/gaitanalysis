# Visualization Module API Reference

Complete API documentation for the `asdrp.visualization` module.

## Table of Contents

1. [overlay.py](#overlaypy)
2. [plots.py](#plotspy)
3. [report.py](#reportpy)

---

## overlay.py

### Classes

#### `LandmarkPoint`

Dataclass representing a single pose landmark.

**Attributes:**
- `x` (float): Normalized x coordinate (0-1)
- `y` (float): Normalized y coordinate (0-1)
- `z` (float): Normalized z coordinate (depth)
- `visibility` (float): Visibility/confidence score (0-1)

**Methods:**
- `to_pixel_coords(width: int, height: int) -> Tuple[int, int]`: Convert normalized coordinates to pixel coordinates

#### `GaitEvent`

Enum defining gait event types.

**Values:**
- `FOOT_STRIKE`: Initial contact of foot with ground
- `TOE_OFF`: Foot leaving the ground
- `MID_STANCE`: Middle of stance phase
- `MID_SWING`: Middle of swing phase

#### `PoseOverlay`

Main class for drawing pose landmarks and connections on video frames.

**Constructor:**
```python
PoseOverlay(
    landmark_color: Tuple[int, int, int] = (0, 255, 0),
    connection_color: Tuple[int, int, int] = (255, 255, 255),
    landmark_radius: int = 5,
    connection_thickness: int = 2,
    visibility_threshold: float = 0.5,
)
```

**Parameters:**
- `landmark_color`: BGR color for landmarks (default: green)
- `connection_color`: BGR color for connections (default: white)
- `landmark_radius`: Radius of landmark circles in pixels
- `connection_thickness`: Thickness of connection lines in pixels
- `visibility_threshold`: Minimum visibility score to draw a landmark (0-1)

**Methods:**

##### `draw_pose()`
```python
draw_pose(
    frame: np.ndarray,
    landmarks: List[LandmarkPoint],
    draw_landmarks: bool = True,
    draw_connections: bool = True,
) -> np.ndarray
```
Draw complete pose on a frame with landmarks and connections.

**Parameters:**
- `frame`: Input video frame (BGR format)
- `landmarks`: List of 33 pose landmarks
- `draw_landmarks`: Whether to draw landmark points
- `draw_connections`: Whether to draw skeleton connections

**Returns:** Frame with pose overlay

##### `draw_landmarks()`
```python
draw_landmarks(
    frame: np.ndarray,
    landmarks: List[LandmarkPoint],
    highlight_indices: Optional[List[int]] = None,
) -> np.ndarray
```
Draw pose landmarks on a frame.

**Parameters:**
- `frame`: Input video frame (BGR)
- `landmarks`: List of pose landmarks
- `highlight_indices`: Optional list of landmark indices to highlight in red

**Returns:** Frame with landmarks drawn

##### `draw_connections()`
```python
draw_connections(
    frame: np.ndarray,
    landmarks: List[LandmarkPoint],
    connections: Optional[List[Tuple[int, int]]] = None,
) -> np.ndarray
```
Draw skeleton connections between landmarks.

**Parameters:**
- `frame`: Input video frame (BGR)
- `landmarks`: List of pose landmarks
- `connections`: Optional list of (start_idx, end_idx) pairs (defaults to POSE_CONNECTIONS)

**Returns:** Frame with connections drawn

##### `draw_gait_events()`
```python
draw_gait_events(
    frame: np.ndarray,
    events: Dict[str, Any],
    landmarks: List[LandmarkPoint],
) -> np.ndarray
```
Draw gait-specific events on the frame.

**Parameters:**
- `frame`: Input video frame (BGR)
- `events`: Dictionary with structure:
  ```python
  {
      'left_foot': GaitEvent or None,
      'right_foot': GaitEvent or None,
  }
  ```
- `landmarks`: List of pose landmarks

**Returns:** Frame with gait events visualized

##### `draw_angle()`
```python
draw_angle(
    frame: np.ndarray,
    landmarks: List[LandmarkPoint],
    joint_indices: Tuple[int, int, int],
    angle_value: float,
    label: Optional[str] = None,
) -> np.ndarray
```
Draw an angle measurement between three landmarks.

**Parameters:**
- `frame`: Input video frame (BGR)
- `landmarks`: List of pose landmarks
- `joint_indices`: Tuple of (point1, joint, point2) landmark indices
- `angle_value`: Angle value in degrees
- `label`: Optional label for the angle

**Returns:** Frame with angle visualization

##### `add_info_panel()`
```python
add_info_panel(
    frame: np.ndarray,
    info: Dict[str, Any],
    position: str = "top-left",
) -> np.ndarray
```
Add an information panel to the frame.

**Parameters:**
- `frame`: Input video frame (BGR)
- `info`: Dictionary of key-value pairs to display
- `position`: Panel position ("top-left", "top-right", "bottom-left", "bottom-right")

**Returns:** Frame with information panel

### Constants

#### `POSE_LANDMARKS`
Dictionary mapping landmark names to indices (0-32).

**Example entries:**
```python
{
    'nose': 0,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28,
    ...
}
```

#### `POSE_CONNECTIONS`
List of landmark index pairs defining the skeleton structure.

---

## plots.py

### Classes

#### `MetricsPlotter`

Class for creating publication-quality plots of gait metrics.

**Constructor:**
```python
MetricsPlotter(
    style: str = "seaborn-v0_8-darkgrid",
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 150,
)
```

**Parameters:**
- `style`: Matplotlib style to use
- `figsize`: Default figure size (width, height) in inches
- `dpi`: Resolution for saved figures

**Attributes:**
- `side_colors`: Dict mapping 'left'/'right' to colors
- `angle_colors`: Dict mapping joint names to colors

**Methods:**

##### `plot_angles_over_time()`
```python
plot_angles_over_time(
    data: pd.DataFrame,
    angles: List[str],
    time_column: str = 'time',
    title: str = "Joint Angles Over Time",
    xlabel: str = "Time (s)",
    ylabel: str = "Angle (degrees)",
    save_path: Optional[Path] = None,
    show_events: bool = True,
    events_data: Optional[pd.DataFrame] = None,
) -> plt.Figure
```
Plot joint angles over time with optional gait events.

**Parameters:**
- `data`: DataFrame with time and angle columns
- `angles`: List of angle column names to plot
- `time_column`: Name of time column
- `title`: Plot title
- `xlabel`: X-axis label
- `ylabel`: Y-axis label
- `save_path`: Optional path to save figure
- `show_events`: Whether to show gait event markers
- `events_data`: DataFrame with 'time', 'event_type', 'side' columns

**Returns:** matplotlib Figure object

##### `plot_gait_cycle()`
```python
plot_gait_cycle(
    data: pd.DataFrame,
    angle_columns: List[str],
    cycle_percentage_column: str = 'cycle_percent',
    title: str = "Gait Cycle Analysis",
    save_path: Optional[Path] = None,
    show_phases: bool = True,
) -> plt.Figure
```
Plot joint angles across normalized gait cycle (0-100%).

**Parameters:**
- `data`: DataFrame with cycle percentage and angle columns
- `angle_columns`: List of angle column names
- `cycle_percentage_column`: Name of cycle percentage column (0-100)
- `title`: Plot title
- `save_path`: Optional path to save
- `show_phases`: Whether to show stance/swing phase regions

**Returns:** matplotlib Figure object

**Note:** If data contains 'cycle_id' column, plots mean ± std across cycles.

##### `plot_symmetry()`
```python
plot_symmetry(
    data: pd.DataFrame,
    left_column: str,
    right_column: str,
    title: str = "Left vs Right Symmetry",
    save_path: Optional[Path] = None,
) -> plt.Figure
```
Create comprehensive symmetry analysis with 3 subplots:
1. Time series comparison
2. Scatter plot with line of identity
3. Difference histogram

**Parameters:**
- `data`: DataFrame with left and right measurements
- `left_column`: Name of left side column
- `right_column`: Name of right side column
- `title`: Plot title
- `save_path`: Optional path to save

**Returns:** matplotlib Figure object

##### `plot_stride_metrics()`
```python
plot_stride_metrics(
    metrics: Dict[str, List[float]],
    title: str = "Stride Metrics",
    save_path: Optional[Path] = None,
) -> plt.Figure
```
Create box plots with individual points and statistics for stride metrics.

**Parameters:**
- `metrics`: Dict with metric names as keys, lists of values
- `title`: Plot title
- `save_path`: Optional path to save

**Returns:** matplotlib Figure object

##### `plot_heatmap()`
```python
plot_heatmap(
    data: pd.DataFrame,
    title: str = "Correlation Heatmap",
    save_path: Optional[Path] = None,
    cmap: str = "coolwarm",
) -> plt.Figure
```
Create correlation heatmap of metrics.

**Parameters:**
- `data`: DataFrame with metrics as columns
- `title`: Plot title
- `save_path`: Optional path to save
- `cmap`: Matplotlib colormap name

**Returns:** matplotlib Figure object

##### `plot_comparison()`
```python
plot_comparison(
    data_dict: Dict[str, pd.DataFrame],
    metric: str,
    title: str = "Metric Comparison",
    ylabel: str = "Value",
    save_path: Optional[Path] = None,
) -> plt.Figure
```
Compare a metric across multiple conditions with violin and box plots.

**Parameters:**
- `data_dict`: Dict with condition names as keys, DataFrames as values
- `metric`: Name of metric column to compare
- `title`: Plot title
- `ylabel`: Y-axis label
- `save_path`: Optional path to save

**Returns:** matplotlib Figure object

##### `close_all()`
```python
close_all() -> None
```
Close all open matplotlib figures to free memory.

---

## report.py

### Classes

#### `ReportGenerator`

Class for generating comprehensive HTML and text reports.

**Constructor:**
```python
ReportGenerator(
    project_name: str = "Running Gait Analysis",
    author: Optional[str] = None,
)
```

**Parameters:**
- `project_name`: Name of the analysis project
- `author`: Optional author name

**Methods:**

##### `generate_html()`
```python
generate_html(
    metrics: Dict[str, Any],
    figures: Optional[Dict[str, plt.Figure]] = None,
    summary_text: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> str
```
Generate a comprehensive HTML report with embedded visualizations.

**Parameters:**
- `metrics`: Dictionary of computed metrics (see structure below)
- `figures`: Optional dict mapping titles to matplotlib figures
- `summary_text`: Optional executive summary text
- `output_path`: Optional path to save HTML file

**Returns:** HTML content as string

**Metrics Structure:**
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
        'joint_angle_name': {...}
    },
    'symmetry': {
        'symmetry_metric': float or dict
    }
}
```

**Report Sections:**
1. Header with project info
2. Executive summary (if provided)
3. Key metrics cards (visual)
4. Embedded visualizations
5. Detailed metric tables
6. Interpretation and recommendations
7. Footer with metadata

##### `generate_summary()`
```python
generate_summary(
    metrics: Dict[str, Any],
    output_path: Optional[Path] = None,
) -> str
```
Generate a text summary of key findings.

**Parameters:**
- `metrics`: Dictionary of computed metrics
- `output_path`: Optional path to save text file

**Returns:** Summary text as string

**Summary Sections:**
1. Header with metadata
2. Temporal metrics
3. Kinematic metrics
4. Symmetry analysis
5. Recommendations

---

## Usage Examples

### Complete Workflow Example

```python
from asdrp.visualization import (
    PoseOverlay,
    MetricsPlotter,
    ReportGenerator,
    LandmarkPoint,
    GaitEvent,
)
import cv2
import pandas as pd
from pathlib import Path

# 1. Pose Overlay on Video Frames
overlay = PoseOverlay()
frame = cv2.imread('frame.jpg')
landmarks = [...]  # List of LandmarkPoint objects

# Draw pose
frame_with_pose = overlay.draw_pose(frame, landmarks)

# Add gait events
events = {'left_foot': GaitEvent.FOOT_STRIKE}
frame_with_events = overlay.draw_gait_events(frame_with_pose, events, landmarks)

# Save
cv2.imwrite('output_frame.jpg', frame_with_events)

# 2. Create Plots
plotter = MetricsPlotter(dpi=150)

# Load data
data = pd.read_csv('gait_data.csv')

# Generate plots
fig1 = plotter.plot_angles_over_time(
    data,
    angles=['left_knee_angle', 'right_knee_angle'],
    save_path='angles.png'
)

fig2 = plotter.plot_symmetry(
    data,
    'left_knee_angle',
    'right_knee_angle',
    save_path='symmetry.png'
)

# 3. Generate Report
reporter = ReportGenerator(
    project_name="My Running Analysis",
    author="Dr. Smith"
)

metrics = {
    'temporal': {
        'cadence': {'mean': 175, 'std': 5},
        'stride_length': {'mean': 2.1, 'std': 0.1},
    },
    'symmetry': {
        'overall_symmetry_index': {'mean': 96.5},
    }
}

# HTML report with embedded figures
html = reporter.generate_html(
    metrics,
    figures={'Angles': fig1, 'Symmetry': fig2},
    output_path='report.html'
)

# Text summary
summary = reporter.generate_summary(metrics, output_path='summary.txt')

# Cleanup
plotter.close_all()
```

---

## Type Hints

All functions include complete type hints for parameters and return values:

```python
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
```

---

## Error Handling

### Common Issues

1. **Missing landmarks**: Functions check for visibility threshold and skip invalid landmarks
2. **Invalid indices**: Boundary checks prevent index errors
3. **Empty data**: Functions handle empty DataFrames gracefully
4. **File I/O**: Path objects are converted and directories are created as needed

### Best Practices

- Always check landmark visibility before processing
- Use try-except blocks when loading external data
- Validate DataFrame column names before plotting
- Close figures after saving to prevent memory leaks
- Use appropriate DPI for target output (screen vs. print)

---

## Performance Considerations

- **Overlay operations**: O(n) where n is number of landmarks (33 for MediaPipe)
- **Plotting**: Vectorized operations via pandas/numpy for efficiency
- **Memory**: Close figures after use; avoid accumulating open plots
- **File I/O**: Save plots incrementally rather than accumulating in memory

---

## Dependencies Version Requirements

Minimum versions:
- OpenCV: >= 4.8.0
- NumPy: >= 1.24.0
- Pandas: >= 2.0.0
- Matplotlib: >= 3.7.0
- Seaborn: >= 0.12.0

---

## License

Part of the Running Gait Analysis project (runninggait).
