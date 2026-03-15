# Running Gait Analysis Notebooks

Jupyter notebooks for the running gait analysis pipeline (MediaPipe Pose Landmarker → gait metrics, joint angles, symmetry → CSV, plots, report).

## Notebook

### gait_analysis.ipynb

**End-to-end gait analysis walkthrough**

The notebook follows the same pipeline as `GaitAnalysisPipeline.run()`, with each stage run explicitly so you can inspect intermediate results. It is organized into **6 stages** (aligned with the [main README](../README.md)#notebook-walkthrough):

| Stage | Notebook sections | Description |
|-------|-------------------|-------------|
| **1. Video Loading** | §2 | Load video with `VideoFileReader`; inspect FPS, resolution, frame count |
| **2. Pose Detection Setup** | §3 | Initialize `MediaPipePoseEstimator`; set detection/tracking confidence |
| **3. Process Frames & Smoothing** | §4 | Run pose on each frame; apply `PoseTracker` (Gaussian smoothing); build `landmarks_sequence` (list of per-frame dicts) |
| **4. Gait Analysis** | §6 | `GaitAnalyzer` + `StrideAnalyzer`, `CadenceAnalyzer`, `SymmetryAnalyzer`; `analyze(landmarks_sequence)` → `GaitMetrics` (events, cadence, stride time, joint angles, symmetry) |
| **5. Symmetry Analysis** | §8 | Left vs right comparison; symmetry metrics and scatter plots |
| **6. Visualization & Export** | §5, §7, §9, §10 | Pose overlay on frames (§5); joint angle time series (§7); dashboard (§9); export CSV, report, PNGs (§10) |

Section 1 is imports and setup. The table above maps the remaining sections to the six stages.

#### Prerequisites

- Python 3.12+
- Dependencies installed (`uv sync` from project root)
- MediaPipe model at `data/models/pose_landmarker.task`
- Sample video at `data/runner_example0.mp4`

#### Running the notebook

From the project root:

```bash
uv run jupyter notebook notebooks/gait_analysis.ipynb
```

Or activate the venv and start Jupyter from any directory:

```bash
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
jupyter notebook
```

Then open `gait_analysis.ipynb`. Outputs are written to `data/outputs/`.

#### Outputs (data/outputs/)

| File | Contents |
|------|----------|
| `gait_metrics.csv` | Scalar gait metrics (one row) |
| `joint_angles.csv` | Per-frame knee, hip, ankle angles |
| `gait_analysis_report.txt` | Text summary |
| `pose_visualization.png` | Sample frames with skeleton overlay |
| `knee_angles.png`, `hip_angles.png`, `ankle_angles.png` | Angle time series |
| `knee_symmetry.png`, `hip_symmetry.png` | Left vs right scatter |
| `gait_dashboard.png` | Multi-panel dashboard |

#### Concepts

- **MediaPipe Pose Landmarker**: 33 body landmarks per frame; normalized (x, y, z), optional world coords, visibility.
- **Temporal smoothing**: `PoseTracker` with Gaussian window; reduces jitter before gait analysis.
- **Gait metrics**: Cadence (steps/min), stride time/length, stance/swing phases, symmetry index (0–1). Joint angles via `LandmarkProcessor.get_joint_angle()` (e.g. knee = angle at knee between hip–knee and knee–ankle vectors).

#### Troubleshooting

| Issue | Fix |
|-------|-----|
| Model not found | Download `pose_landmarker.task` from [MediaPipe](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) and place in `data/models/` |
| Video not found | Use `data/runner_example0.mp4` or set `video_path` to your file |
| Low detection rate | Increase `min_detection_confidence` / `min_tracking_confidence` |
| Noisy angles | Increase `PoseTracker` `window_size` (e.g. 7) or adjust `sigma` |

## Getting started

Start with `gait_analysis.ipynb` for a full walkthrough. The [main README](../README.md) has Quick Start (single `pipeline.run()` call), project structure, and stage details.

## Project structure

```
notebooks/
├── README.md              # This file
└── gait_analysis.ipynb    # Main walkthrough

../data/
├── runner_example0.mp4    # Sample video
├── models/
│   └── pose_landmarker.task
└── outputs/               # Generated CSV, PNG, TXT, HTML

../asdrp/
├── pipeline.py
├── video/   pose/   analysis/   visualization/   utils/
```

## Resources

- [MediaPipe Pose Landmarker](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
- Main project [README](../README.md)
