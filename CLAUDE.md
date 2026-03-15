# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses `uv` for dependency management.

```bash
# Install dependencies
uv sync

# Install with dev dependencies (pytest, ruff, black, mypy)
uv sync --extra dev

# Run all tests
uv run pytest

# Run a single test
uv run pytest tests/test_foo.py::test_bar_function

# Lint
uv run ruff check asdrp/

# Format
uv run black asdrp/

# Type check
uv run mypy asdrp/

# Launch Jupyter for notebooks
uv run jupyter notebook
```

Line length is 100 (black and ruff both configured for this).

## Architecture

The importable package is `asdrp` (Advanced Sports Data Research Platform), built as a Python 3.12+ library for running gait analysis from video.

### Pipeline flow

`GaitAnalysisPipeline` in `asdrp/pipeline.py` is the central orchestrator. It executes five sequential stages:

1. **setup()** — initializes all components from a `PipelineConfig`
2. **process_video()** — reads frames via `VideoFileReader`, runs per-frame pose estimation via `MediaPipePoseEstimator` (MediaPipe), accumulates `PoseLandmarks` sequence
3. **analyze_gait()** — converts landmarks to dict format via `LandmarkProcessor.to_dict()`, runs `GaitAnalyzer` (which delegates to plugged-in calculators: `StrideAnalyzer`, `CadenceAnalyzer`, `SymmetryAnalyzer`), produces a `GaitMetrics` object
4. **create_visualizations()** — generates overlay video via `PoseOverlay` + `VideoWriter`, creates plots via `MetricsPlotter`
5. **generate_report()** — produces HTML via `ReportGenerator`

`pipeline.run()` calls all five stages and returns `{'metrics', 'visualizations', 'report_path', 'config'}`. It also supports use as a context manager for automatic cleanup.

### Module layout

| Module | Key classes |
|---|---|
| `asdrp/pipeline.py` | `GaitAnalysisPipeline` |
| `asdrp/video/` | `VideoFileReader`, `VideoWriter`, `FrameData` |
| `asdrp/pose/` | `MediaPipePoseEstimator`, `PoseTracker`, `LandmarkProcessor`, `PoseLandmarks`, `PoseLandmarkIndex` |
| `asdrp/analysis/` | `GaitAnalyzer`, `StrideAnalyzer`, `CadenceAnalyzer`, `SymmetryAnalyzer`, `GaitMetrics`, `GaitEvent`, `GaitEventType`, `Foot`, `BaseMetricCalculator` |
| `asdrp/visualization/` | `PoseOverlay`, `MetricsPlotter`, `ReportGenerator` |
| `asdrp/utils/` | `PipelineConfig`, `VideoConfig`, `PoseEstimationConfig`, `GaitAnalysisConfig`, `VisualizationConfig`, `create_default_config`, geometry/smoothing utilities |

### Configuration

All config is type-safe dataclasses in `asdrp/utils/config.py`. Use `create_default_config(video_path, model_path)` for quick setup, or construct `PipelineConfig` manually for fine-grained control. Config can be loaded from / saved to JSON via `PipelineConfig.from_json()` / `to_json()`.

### Data files

- `data/runner_example0.mp4` — sample video (in repo)
- `data/models/pose_landmarker.task` — MediaPipe Pose Landmarker model (must be downloaded separately from MediaPipe)
- `data/outputs/` — generated results (plots, CSVs, reports)

### Demo notebook

`notebooks/gait_analysis_demo.ipynb` is a full end-to-end walkthrough: video loading, pose detection, gait metrics, joint angles, symmetry analysis, and export.
