# Visualization Module Quick Start

Get started with the visualization module in 5 minutes.

## Installation

The visualization module is part of the `runninggait` package. Ensure all dependencies are installed:

```bash
pip install -e .
```

## Quick Examples

### 1. Draw Pose on a Video Frame (30 seconds)

```python
import cv2
from asdrp.visualization import PoseOverlay, LandmarkPoint

# Load a video frame
frame = cv2.imread('running_frame.jpg')

# Create landmarks (normally from MediaPipe)
landmarks = [
    LandmarkPoint(x=0.5, y=0.3, z=0.0, visibility=0.95),
    # ... 32 more landmarks ...
]

# Create overlay and draw
overlay = PoseOverlay()
result = overlay.draw_pose(frame, landmarks)

# Save
cv2.imwrite('output.jpg', result)
```

### 2. Plot Joint Angles (1 minute)

```python
import pandas as pd
from asdrp.visualization import MetricsPlotter

# Load your gait data
data = pd.DataFrame({
    'time': [0.0, 0.01, 0.02, ...],
    'left_knee_angle': [140, 142, 145, ...],
    'right_knee_angle': [138, 141, 143, ...],
})

# Create plotter and generate figure
plotter = MetricsPlotter()
fig = plotter.plot_angles_over_time(
    data,
    angles=['left_knee_angle', 'right_knee_angle'],
    title="Knee Angles During Running",
    save_path="knee_angles.png"
)

plotter.close_all()
```

### 3. Generate HTML Report (2 minutes)

```python
from asdrp.visualization import ReportGenerator

# Define your metrics
metrics = {
    'temporal': {
        'cadence': {'mean': 175.5, 'std': 4.2},
        'stride_length': {'mean': 2.1, 'std': 0.1},
    },
    'symmetry': {
        'overall_symmetry_index': {'mean': 96.5},
    }
}

# Generate report
reporter = ReportGenerator(
    project_name="My Running Analysis",
    author="Your Name"
)

html_report = reporter.generate_html(
    metrics,
    summary_text="Analysis shows good running form with minimal asymmetry.",
    output_path="report.html"
)

print("Report saved to report.html")
```

## Common Workflows

### Workflow 1: Video Frame Annotation

```python
import cv2
from asdrp.visualization import PoseOverlay, GaitEvent, POSE_LANDMARKS

# Initialize
overlay = PoseOverlay(
    landmark_color=(0, 255, 0),
    connection_color=(255, 255, 255),
)

# Process video
cap = cv2.VideoCapture('running_video.mp4')
ret, frame = cap.read()

# Get landmarks from your pose detector
landmarks = your_pose_detector.detect(frame)

# Draw complete pose
frame = overlay.draw_pose(frame, landmarks)

# Mark gait events
events = {'left_foot': GaitEvent.FOOT_STRIKE}
frame = overlay.draw_gait_events(frame, events, landmarks)

# Draw specific angle
frame = overlay.draw_angle(
    frame,
    landmarks,
    joint_indices=(
        POSE_LANDMARKS['left_hip'],
        POSE_LANDMARKS['left_knee'],
        POSE_LANDMARKS['left_ankle']
    ),
    angle_value=145.0,
    label="Knee"
)

# Add info panel
frame = overlay.add_info_panel(
    frame,
    {'Frame': 100, 'Time': '3.33s', 'Phase': 'Stance'},
    position="top-left"
)

cv2.imwrite('annotated_frame.jpg', frame)
```

### Workflow 2: Comprehensive Metric Visualization

```python
import pandas as pd
from asdrp.visualization import MetricsPlotter

# Load data
data = pd.read_csv('gait_metrics.csv')

# Initialize plotter
plotter = MetricsPlotter(figsize=(12, 8), dpi=150)

# Create multiple plots
plots = {}

# 1. Time series
plots['angles'] = plotter.plot_angles_over_time(
    data,
    angles=['left_knee_angle', 'right_knee_angle', 'left_hip_angle', 'right_hip_angle'],
    save_path='angles_time.png'
)

# 2. Symmetry analysis
plots['symmetry'] = plotter.plot_symmetry(
    data,
    'left_knee_angle',
    'right_knee_angle',
    save_path='symmetry.png'
)

# 3. Stride metrics
stride_data = {
    'Stride Time (s)': data['stride_time'].tolist(),
    'Cadence (spm)': data['cadence'].tolist(),
}
plots['strides'] = plotter.plot_stride_metrics(
    stride_data,
    save_path='stride_metrics.png'
)

# Cleanup
plotter.close_all()

print("All plots saved!")
```

### Workflow 3: Complete Analysis Report

```python
import pandas as pd
from asdrp.visualization import MetricsPlotter, ReportGenerator

# 1. Calculate metrics from your data
data = pd.read_csv('gait_data.csv')

metrics = {
    'temporal': {
        'stride_time': {
            'mean': data['stride_time'].mean(),
            'std': data['stride_time'].std(),
            'min': data['stride_time'].min(),
            'max': data['stride_time'].max(),
        },
        'cadence': {
            'mean': data['cadence'].mean(),
            'std': data['cadence'].std(),
        },
    },
    'kinematics': {
        'left_knee_angle': {
            'mean': data['left_knee_angle'].mean(),
            'std': data['left_knee_angle'].std(),
        },
        'right_knee_angle': {
            'mean': data['right_knee_angle'].mean(),
            'std': data['right_knee_angle'].std(),
        },
    },
    'symmetry': {
        'overall_symmetry_index': {
            'mean': 96.5,  # Your calculation
        },
    },
}

# 2. Create visualizations
plotter = MetricsPlotter()

figures = {
    'Joint Angles': plotter.plot_angles_over_time(
        data,
        ['left_knee_angle', 'right_knee_angle']
    ),
    'Symmetry Analysis': plotter.plot_symmetry(
        data,
        'left_knee_angle',
        'right_knee_angle'
    ),
}

# 3. Generate report
reporter = ReportGenerator(
    project_name="Running Gait Analysis",
    author="Your Name"
)

# HTML report with embedded figures
reporter.generate_html(
    metrics,
    figures=figures,
    summary_text="Analysis shows good running form with minimal asymmetry and optimal cadence.",
    output_path="complete_report.html"
)

# Text summary
reporter.generate_summary(
    metrics,
    output_path="summary.txt"
)

# Cleanup
plotter.close_all()

print("Complete report generated!")
print("  - complete_report.html")
print("  - summary.txt")
```

## Customization Tips

### Custom Colors

```python
# Define your color scheme
overlay = PoseOverlay(
    landmark_color=(255, 0, 0),      # Blue landmarks
    connection_color=(0, 255, 255),   # Yellow connections
)

# Or modify plotter colors
plotter = MetricsPlotter()
plotter.side_colors['left'] = '#FF5733'
plotter.side_colors['right'] = '#33FF57'
```

### Custom Figure Sizes

```python
# For presentations
plotter = MetricsPlotter(figsize=(16, 9), dpi=200)

# For papers
plotter = MetricsPlotter(figsize=(8, 6), dpi=300)

# For posters
plotter = MetricsPlotter(figsize=(20, 15), dpi=300)
```

### Custom Report Styling

```python
reporter = ReportGenerator(
    project_name="Elite Runner Study - Subject 001",
    author="Dr. Smith, PhD"
)

# The CSS styling is embedded in the report.py file
# You can modify reporter.css_style for custom styling
```

## Tips and Tricks

### 1. Batch Processing

```python
import glob
from pathlib import Path

overlay = PoseOverlay()

for video_path in glob.glob('videos/*.mp4'):
    # Process each video
    process_video(video_path, overlay)
```

### 2. Memory Management

```python
# Always close figures after saving
plotter = MetricsPlotter()

for i in range(100):
    fig = plotter.plot_angles_over_time(data[i], ...)
    # Figure is automatically saved

plotter.close_all()  # Close all at end
```

### 3. High-Quality Output

```python
# For publication-quality figures
plotter = MetricsPlotter(
    figsize=(10, 6),
    dpi=300,  # High resolution
)

# Save as vector format
fig = plotter.plot_angles_over_time(...)
fig.savefig('figure.pdf', format='pdf', bbox_inches='tight')
fig.savefig('figure.svg', format='svg', bbox_inches='tight')
```

### 4. Quick Checks

```python
# Verify landmark visibility
visible_count = sum(1 for lm in landmarks if lm.visibility > 0.5)
print(f"Visible landmarks: {visible_count}/33")

# Check data integrity
print(data.describe())  # Statistical summary
print(data.isnull().sum())  # Missing values
```

## Troubleshooting

### Issue: Landmarks not visible
**Solution**: Check visibility threshold
```python
overlay = PoseOverlay(visibility_threshold=0.3)  # Lower threshold
```

### Issue: Plots look cramped
**Solution**: Increase figure size
```python
plotter = MetricsPlotter(figsize=(16, 10))
```

### Issue: HTML report too large
**Solution**: Save figures separately and reduce DPI
```python
plotter = MetricsPlotter(dpi=100)  # Lower DPI
```

## Next Steps

1. Check out `/examples/visualization_example.py` for complete working examples
2. Read `README.md` for detailed documentation
3. See `API.md` for complete API reference
4. Explore the source code in `overlay.py`, `plots.py`, and `report.py`

## Getting Help

- Check the examples directory for working code
- Review the API documentation for function signatures
- Examine the source code for implementation details
- Test with sample data before processing real datasets

Happy visualizing!
