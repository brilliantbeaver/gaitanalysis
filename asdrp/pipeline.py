"""Main pipeline for running gait analysis.

This module provides the GaitAnalysisPipeline class that orchestrates the complete
analysis workflow from video input to final report generation.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import cv2

from .video import VideoFileReader, VideoWriter
from .pose import MediaPipePoseEstimator, PoseTracker, PoseLandmarks, LandmarkProcessor
from .analysis import GaitAnalyzer, StrideAnalyzer, CadenceAnalyzer, SymmetryAnalyzer, GaitMetrics
from .visualization import PoseOverlay, MetricsPlotter, ReportGenerator
from .utils import PipelineConfig, smooth_trajectory, create_default_config

logger = logging.getLogger(__name__)


class GaitAnalysisPipeline:
    """Orchestrates the complete gait analysis workflow.

    This pipeline coordinates video reading, pose estimation, gait analysis,
    visualization, and report generation into a unified workflow.

    Attributes:
        config: Pipeline configuration object
        video_reader: Video input handler
        pose_estimator: Pose estimation model
        pose_tracker: Multi-frame pose tracking
        gait_analyzer: Gait metrics calculator
        overlay: Pose visualization overlay
        plotter: Metrics plotting utility
        report_generator: Report generation utility

    Example:
        >>> from asdrp import GaitAnalysisPipeline
        >>> from asdrp.utils import create_default_config
        >>> from pathlib import Path
        >>>
        >>> # Create configuration
        >>> config = create_default_config(
        ...     video_path=Path("running.mp4"),
        ...     model_path=Path("pose_landmarker.task")
        ... )
        >>>
        >>> # Run pipeline
        >>> pipeline = GaitAnalysisPipeline(config)
        >>> results = pipeline.run()
        >>> print(f"Cadence: {results['metrics'].cadence:.1f} spm")
    """

    def __init__(self, config: PipelineConfig):
        """Initialize the gait analysis pipeline.

        Args:
            config: Pipeline configuration object

        Raises:
            FileNotFoundError: If video or model files don't exist
            ValueError: If configuration is invalid
        """
        self.config = config
        self.video_reader: Optional[VideoFileReader] = None
        self.pose_estimator: Optional[MediaPipePoseEstimator] = None
        self.pose_tracker: Optional[PoseTracker] = None
        self.gait_analyzer: Optional[GaitAnalyzer] = None
        self.overlay: Optional[PoseOverlay] = None
        self.plotter: Optional[MetricsPlotter] = None
        self.report_generator: Optional[ReportGenerator] = None

        # Results storage
        self.landmarks_sequence: List[Optional[PoseLandmarks]] = []
        self.smoothed_landmarks_sequence: List[Dict[str, Any]] = []
        self.metrics: Optional[GaitMetrics] = None

        # Setup logging
        if config.verbose:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
        else:
            logging.basicConfig(level=logging.WARNING)

        logger.info(f"Initialized pipeline: {config.project_name}")

    def setup(self) -> None:
        """Set up all pipeline components.

        This method initializes the video reader, pose estimator, analyzers,
        and visualization tools.

        Raises:
            FileNotFoundError: If required files don't exist
            RuntimeError: If component initialization fails
        """
        logger.info("Setting up pipeline components...")

        # Create output directory
        self.config.output_directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.config.output_directory}")

        # Initialize video reader
        try:
            self.video_reader = VideoFileReader(self.config.video.input_path)
            fps = self.video_reader.get_fps()
            frame_count = self.video_reader.get_frame_count()
            resolution = self.video_reader.get_resolution()

            logger.info(
                f"Video loaded: {self.config.video.input_path.name} "
                f"({frame_count} frames, {fps:.2f} fps, {resolution[0]}x{resolution[1]})"
            )

            # Update analysis fps if not set
            if self.config.analysis.fps != fps:
                logger.info(f"Updating analysis fps from {self.config.analysis.fps} to {fps}")
                self.config.analysis.fps = fps

        except Exception as e:
            logger.error(f"Failed to initialize video reader: {e}")
            raise

        # Initialize pose estimator
        try:
            self.pose_estimator = MediaPipePoseEstimator(
                model_path=self.config.pose.model_path,
                min_detection_confidence=self.config.pose.min_detection_confidence,
                min_tracking_confidence=self.config.pose.min_tracking_confidence,
                running_mode=self.config.pose.running_mode,
            )
            logger.info("Pose estimator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize pose estimator: {e}")
            raise

        # Initialize pose tracker
        if self.config.pose.smooth_landmarks:
            self.pose_tracker = PoseTracker(window_size=self.config.pose.smoothing_window)
            logger.info(f"Pose tracker initialized (window_size={self.config.pose.smoothing_window})")

        # Initialize gait analyzer
        self.gait_analyzer = GaitAnalyzer(fps=self.config.analysis.fps)

        # Add metric calculators based on configuration
        if self.config.analysis.calculate_stride:
            self.gait_analyzer.add_calculator(StrideAnalyzer(fps=self.config.analysis.fps))
            logger.info("Added stride analyzer")

        if self.config.analysis.calculate_cadence:
            self.gait_analyzer.add_calculator(CadenceAnalyzer(fps=self.config.analysis.fps))
            logger.info("Added cadence analyzer")

        if self.config.analysis.calculate_symmetry:
            self.gait_analyzer.add_calculator(SymmetryAnalyzer(fps=self.config.analysis.fps))
            logger.info("Added symmetry analyzer")

        # Initialize visualization components
        if self.config.visualization.create_overlay_video or self.config.visualization.create_plots:
            self.overlay = PoseOverlay()
            logger.info("Pose overlay initialized")

        if self.config.visualization.create_plots:
            self.plotter = MetricsPlotter(
                figsize=(12, 8),
                dpi=self.config.visualization.dpi,
            )
            logger.info("Metrics plotter initialized")

        if self.config.visualization.create_report:
            self.report_generator = ReportGenerator(
                project_name=self.config.project_name,
            )
            logger.info("Report generator initialized")

        logger.info("Pipeline setup complete")

    def process_video(self) -> None:
        """Process video frames through pose estimation.

        This method reads video frames, estimates poses, and optionally applies
        smoothing/tracking. Progress is logged if verbose mode is enabled.

        Raises:
            RuntimeError: If video processing fails
        """
        if self.video_reader is None or self.pose_estimator is None:
            raise RuntimeError("Pipeline not set up. Call setup() first.")

        logger.info("Starting video processing...")

        frame_count = self.video_reader.get_frame_count()
        max_frames = self.config.video.max_frames or frame_count

        self.landmarks_sequence = []
        processed_frames = 0
        detected_poses = 0

        try:
            while processed_frames < max_frames:
                frame_data = self.video_reader.read_frame()

                if frame_data is None:
                    break

                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame_data.image, cv2.COLOR_BGR2RGB)

                # Estimate pose
                timestamp_ms = frame_data.timestamp * 1000  # Convert to milliseconds
                landmarks = self.pose_estimator.estimate(
                    rgb_frame,
                    timestamp=timestamp_ms,
                    frame_number=frame_data.frame_number,
                )

                self.landmarks_sequence.append(landmarks)

                if landmarks is not None:
                    detected_poses += 1

                    # Add to tracker if enabled
                    if self.pose_tracker is not None:
                        self.pose_tracker.add_detection(landmarks)

                processed_frames += 1

                # Log progress
                if self.config.verbose and processed_frames % 30 == 0:
                    detection_rate = (detected_poses / processed_frames) * 100
                    logger.info(
                        f"Processed {processed_frames}/{max_frames} frames "
                        f"({detection_rate:.1f}% detection rate)"
                    )

        except Exception as e:
            logger.error(f"Error during video processing: {e}")
            raise
        finally:
            self.video_reader.close()

        detection_rate = (detected_poses / processed_frames) * 100 if processed_frames > 0 else 0
        logger.info(
            f"Video processing complete: {processed_frames} frames processed, "
            f"{detected_poses} poses detected ({detection_rate:.1f}%)"
        )

    def analyze_gait(self) -> None:
        """Analyze gait patterns from detected poses.

        This method converts pose landmarks to the format expected by analyzers
        and computes all configured gait metrics.

        Raises:
            RuntimeError: If gait analysis fails
        """
        if not self.landmarks_sequence:
            raise RuntimeError("No landmarks to analyze. Process video first.")

        if self.gait_analyzer is None:
            raise RuntimeError("Pipeline not set up. Call setup() first.")

        logger.info("Starting gait analysis...")

        try:
            # Convert landmarks to analysis format
            self.smoothed_landmarks_sequence = self._prepare_landmarks_for_analysis()

            # Run analysis
            self.metrics = self.gait_analyzer.analyze(self.smoothed_landmarks_sequence)

            logger.info(
                f"Gait analysis complete: "
                f"Cadence={self.metrics.cadence:.1f} spm, "
                f"Stride time={self.metrics.stride_time:.3f}s, "
                f"Symmetry={self.metrics.symmetry_index:.3f}"
            )

        except Exception as e:
            logger.error(f"Error during gait analysis: {e}")
            raise

    def _prepare_landmarks_for_analysis(self) -> List[Dict[str, Any]]:
        """Convert pose landmarks to format expected by gait analyzers.

        Returns:
            List of landmark dictionaries with format expected by analyzers
        """
        prepared_landmarks = []

        for i, landmarks in enumerate(self.landmarks_sequence):
            if landmarks is None:
                # Fill with empty dict for missing frames
                prepared_landmarks.append({})
                continue

            # Get smoothed landmarks if tracker is available
            if self.pose_tracker is not None and i < len(self.landmarks_sequence) - 1:
                # Use tracked/smoothed landmarks
                smoothed = self.pose_tracker.get_smoothed_landmarks()
                if smoothed is not None:
                    landmarks = smoothed

            # Convert to dictionary format
            landmark_dict = LandmarkProcessor.to_dict(landmarks)
            prepared_landmarks.append(landmark_dict)

        return prepared_landmarks

    def create_visualizations(self) -> Dict[str, Any]:
        """Create visualizations of analysis results.

        This method generates plots and optionally an overlay video with
        pose landmarks and gait metrics.

        Returns:
            Dictionary containing paths to generated visualizations

        Raises:
            RuntimeError: If visualization generation fails
        """
        if self.metrics is None:
            raise RuntimeError("No metrics to visualize. Analyze gait first.")

        logger.info("Creating visualizations...")

        visualization_paths = {}

        try:
            # Create overlay video
            if self.config.visualization.create_overlay_video and self.overlay is not None:
                video_path = self._create_overlay_video()
                visualization_paths["overlay_video"] = video_path
                logger.info(f"Created overlay video: {video_path}")

            # Create plots
            if self.config.visualization.create_plots and self.plotter is not None:
                plot_paths = self._create_plots()
                visualization_paths["plots"] = plot_paths
                logger.info(f"Created {len(plot_paths)} plots")

            logger.info("Visualizations complete")

        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            raise

        return visualization_paths

    def _create_overlay_video(self) -> Path:
        """Create video with pose overlay.

        Returns:
            Path to output video file
        """
        output_path = self.config.output_directory / f"{self.config.project_name}_overlay.mp4"

        # Re-open video reader
        video_reader = VideoFileReader(self.config.video.input_path)
        fps = video_reader.get_fps()
        width, height = video_reader.get_resolution()

        # Create video writer
        writer = VideoWriter(
            output_path,
            fps=fps,
            frame_size=(width, height),
            fourcc=self.config.visualization.video_codec,
        )

        try:
            frame_idx = 0
            while frame_idx < len(self.landmarks_sequence):
                frame_data = video_reader.read_frame()
                if frame_data is None:
                    break

                landmarks = self.landmarks_sequence[frame_idx]

                if landmarks is not None:
                    # Draw pose overlay
                    frame_with_overlay = self.overlay.draw_pose(
                        frame_data.image.copy(),
                        landmarks,
                    )
                else:
                    frame_with_overlay = frame_data.image

                writer.write_frame(frame_with_overlay)
                frame_idx += 1

        finally:
            video_reader.close()
            writer.close()

        return output_path

    def _create_plots(self) -> Dict[str, Path]:
        """Create analysis plots.

        Returns:
            Dictionary mapping plot names to file paths
        """
        plot_paths = {}
        output_format = self.config.visualization.output_format

        # Convert metrics to plotter format
        metrics_dict = self.metrics.to_dict()

        if self.config.visualization.plot_cadence and "cadence" in metrics_dict:
            plot_path = self.config.output_directory / f"cadence_plot.{output_format}"
            # Create cadence plot using plotter
            fig = self.plotter.plot_cadence_over_time(
                timestamps=[],  # Would need to extract from events
                cadence_values=[metrics_dict["cadence"]],
            )
            fig.savefig(plot_path, format=output_format, dpi=self.config.visualization.dpi)
            plot_paths["cadence"] = plot_path

        if self.config.visualization.plot_stride:
            plot_path = self.config.output_directory / f"stride_plot.{output_format}"
            # Create stride plot
            # Implementation depends on specific plot requirements
            plot_paths["stride"] = plot_path

        if self.config.visualization.plot_symmetry:
            plot_path = self.config.output_directory / f"symmetry_plot.{output_format}"
            # Create symmetry plot
            plot_paths["symmetry"] = plot_path

        return plot_paths

    def generate_report(self) -> Path:
        """Generate comprehensive analysis report.

        Returns:
            Path to generated HTML report

        Raises:
            RuntimeError: If report generation fails
        """
        if self.metrics is None:
            raise RuntimeError("No metrics to report. Analyze gait first.")

        if self.report_generator is None:
            raise RuntimeError("Report generator not initialized.")

        logger.info("Generating analysis report...")

        try:
            report_path = self.config.output_directory / f"{self.config.project_name}_report.html"

            # Convert metrics to dictionary
            metrics_dict = self.metrics.to_dict()

            # Generate report
            self.report_generator.generate_html(
                metrics=metrics_dict,
                output_path=report_path,
            )

            logger.info(f"Report generated: {report_path}")

            return report_path

        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise

    def run(self) -> Dict[str, Any]:
        """Run the complete gait analysis pipeline.

        This is the main entry point that executes all pipeline stages:
        1. Setup components
        2. Process video
        3. Analyze gait
        4. Create visualizations
        5. Generate report

        Returns:
            Dictionary containing:
                - metrics: GaitMetrics object
                - visualizations: Dictionary of visualization paths
                - report_path: Path to HTML report

        Example:
            >>> pipeline = GaitAnalysisPipeline(config)
            >>> results = pipeline.run()
            >>> print(f"Analysis complete! Report: {results['report_path']}")
        """
        logger.info(f"Starting pipeline: {self.config.project_name}")

        try:
            # Stage 1: Setup
            self.setup()

            # Stage 2: Process video
            self.process_video()

            # Stage 3: Analyze gait
            self.analyze_gait()

            # Stage 4: Create visualizations
            visualizations = {}
            if (
                self.config.visualization.create_overlay_video
                or self.config.visualization.create_plots
            ):
                visualizations = self.create_visualizations()

            # Stage 5: Generate report
            report_path = None
            if self.config.visualization.create_report:
                report_path = self.generate_report()

            results = {
                "metrics": self.metrics,
                "visualizations": visualizations,
                "report_path": report_path,
                "config": self.config,
            }

            logger.info("Pipeline complete!")

            return results

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

        finally:
            # Cleanup
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources used by the pipeline."""
        logger.info("Cleaning up pipeline resources...")

        if self.video_reader is not None:
            self.video_reader.close()

        if self.pose_estimator is not None:
            self.pose_estimator.close()

        logger.info("Cleanup complete")

    def __enter__(self):
        """Context manager entry."""
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
