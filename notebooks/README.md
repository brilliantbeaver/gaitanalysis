# Running Gait Analysis Notebooks

This directory contains Jupyter notebooks demonstrating the use of the running gait analysis system.

## Notebooks

### 1. gait_analysis_demo.ipynb

**Comprehensive Running Gait Analysis Tutorial**

A complete, end-to-end demonstration of running gait analysis using MediaPipe Pose Landmarker.

#### What This Notebook Covers

1. **Video Loading and Preprocessing**
   - Loading video files using the VideoFileReader
   - Inspecting video properties (resolution, FPS, duration)
   - Displaying video preview

2. **Pose Detection Setup**
   - Initializing MediaPipe Pose Landmarker
   - Configuring detection and tracking confidence thresholds
   - Setting up temporal smoothing with PoseTracker

3. **Frame-by-Frame Processing**
   - Extracting pose landmarks from each video frame
   - Applying Gaussian smoothing for temporal consistency
   - Converting landmarks to analysis-ready format

4. **Pose Visualization**
   - Drawing landmarks and skeleton connections on frames
   - Visualizing pose detection results
   - Creating annotated frame sequences

5. **Gait Analysis**
   - Computing temporal metrics (cadence, stride time, stance/swing phases)
   - Computing spatial metrics (stride length)
   - Computing angular metrics (knee/hip/ankle flexion and extension)
   - Analyzing left-right symmetry

6. **Joint Angle Extraction**
   - Calculating knee angles over time
   - Calculating hip angles over time
   - Calculating ankle angles over time
   - Extracting angle statistics and ranges

7. **Symmetry Analysis**
   - Comparing left and right side metrics
   - Computing correlation coefficients
   - Visualizing asymmetries

8. **Visualization Dashboard**
   - Creating comprehensive multi-panel visualizations
   - Plotting time series of joint angles
   - Generating symmetry scatter plots
   - Building analysis summary dashboard

9. **Results Export**
   - Exporting metrics to CSV
   - Saving joint angle data
   - Generating detailed text reports
   - Saving all visualization plots

10. **Summary and Next Steps**
    - Review of accomplishments
    - Suggestions for extending the analysis
    - References and resources

#### Prerequisites

- Python 3.12+
- All dependencies installed (see `pyproject.toml`)
- MediaPipe Pose Landmarker model file (`pose_landmarker.task`)
- Sample video file (`data/runner_example.mp4`)

#### Running the Notebook

1. Ensure you're in the virtual environment:
   ```bash
   source ../.venv/bin/activate  # On Unix/macOS
   # or
   ../.venv\Scripts\activate  # On Windows
   ```

2. Start Jupyter:
   ```bash
   jupyter notebook
   ```

3. Open `gait_analysis_demo.ipynb` and run all cells

4. Results will be saved to `../data/outputs/`

#### Expected Outputs

The notebook generates the following files in `data/outputs/`:

- `gait_metrics.csv` - Numerical gait metrics
- `joint_angles.csv` - Time series of joint angles
- `gait_analysis_report.txt` - Detailed text report
- `pose_visualization.png` - Sample frames with pose overlay
- `knee_angles.png` - Knee angle time series plot
- `hip_angles.png` - Hip angle time series plot
- `ankle_angles.png` - Ankle angle time series plot
- `knee_symmetry.png` - Knee symmetry analysis
- `hip_symmetry.png` - Hip symmetry analysis
- `gait_dashboard.png` - Comprehensive analysis dashboard

#### Key Concepts

**MediaPipe Pose Landmarker**
- Detects 33 body landmarks in each frame
- Provides 3D world coordinates and normalized 2D coordinates
- Includes visibility scores for each landmark

**Temporal Smoothing**
- Applies Gaussian filtering over a sliding window
- Reduces noise and jitter in landmark positions
- Improves stability of derived metrics

**Gait Metrics**
- **Cadence**: Steps per minute
- **Stride Time**: Time between successive heel strikes (same foot)
- **Stride Length**: Distance between successive heel strikes (same foot)
- **Stance Phase**: Duration foot is on the ground
- **Swing Phase**: Duration foot is in the air
- **Symmetry Index**: Measure of left-right balance (0-1)

**Joint Angles**
- Calculated using vector geometry between three landmarks
- Example: Knee angle = angle between hip-knee and knee-ankle vectors
- Measured in degrees (0-180)

#### Troubleshooting

**Issue**: Model file not found
- **Solution**: Download the pose_landmarker.task model from [MediaPipe website](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) and place it in `data/models/`

**Issue**: Video file not found
- **Solution**: Ensure `runner_example.mp4` exists in `data/` directory

**Issue**: Low detection rate
- **Solution**: Adjust `min_detection_confidence` and `min_tracking_confidence` parameters

**Issue**: Noisy angle measurements
- **Solution**: Increase the `window_size` parameter in PoseTracker for more smoothing

#### Customization

You can customize the analysis by:

- **Changing video**: Update `video_path` to analyze different videos
- **Adjusting smoothing**: Modify `window_size` and `sigma` in PoseTracker
- **Adding metrics**: Extend the gait analyzers with custom calculations
- **Modifying visualizations**: Change colors, styles, and plot types
- **Processing subsets**: Analyze only specific frame ranges for faster testing

#### Performance Notes

- Processing time depends on video length and resolution
- Typical processing: ~1-2 seconds per frame on modern hardware
- For long videos, consider processing a subset of frames
- Memory usage scales with number of frames kept for visualization

## Getting Started

If you're new to the project, start with `gait_analysis_demo.ipynb` which provides a complete walkthrough of the entire analysis pipeline.

## Project Structure

```
notebooks/
├── README.md                    # This file
└── gait_analysis_demo.ipynb    # Main tutorial notebook

../data/
├── runner_example.mp4          # Sample video
├── models/
│   └── pose_landmarker.task   # MediaPipe model
└── outputs/                    # Generated results
    ├── *.csv                   # Data exports
    ├── *.txt                   # Text reports
    └── *.png                   # Visualizations

../asdrp/                       # Main package
├── pose/                       # Pose estimation
├── video/                      # Video processing
├── analysis/                   # Gait analysis
└── visualization/              # Plotting and overlays
```

## Additional Resources

- [MediaPipe Documentation](https://developers.google.com/mediapipe)
- [Project GitHub Repository](https://github.com/yourusername/runninggait)
- [Running Biomechanics Research](https://pubmed.ncbi.nlm.nih.gov/)
- [Gait Analysis Standards](https://www.clinicalgaitanalysis.com/)

## Contributing

To add new notebooks:

1. Create a descriptive filename (e.g., `advanced_symmetry_analysis.ipynb`)
2. Include clear markdown documentation
3. Follow the existing code style
4. Update this README with notebook description
5. Ensure all cells run without errors

## Support

For questions or issues:
- Check the main project README
- Review example scripts in `examples/`
- Consult the API documentation in `docs/`
- Open an issue on GitHub

---

**Happy Analyzing!** 🏃‍♂️📊
