"""Video writing functionality for outputting annotated videos.

This module provides classes for writing video files with annotations,
pose overlays, and other visual augmentations.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from .frame import FrameData


class VideoWriter:
    """Write frames to a video file using OpenCV.

    This class provides functionality to write video files frame-by-frame,
    typically used for outputting annotated videos with pose overlays,
    gait analysis visualizations, or other processed content.

    Attributes:
        output_path: Path where the video file will be written.
        fps: Frames per second for the output video.
        frame_size: Video frame dimensions as (width, height).
        fourcc: Four-character code for video codec.
        frames_written: Number of frames written so far.

    Example:
        >>> writer = VideoWriter(
        ...     output_path="output.mp4",
        ...     fps=30.0,
        ...     frame_size=(1280, 720)
        ... )
        >>> with writer:
        ...     for frame in frames:
        ...         writer.write_frame(frame)
    """

    # Common codec mappings
    CODECS = {
        "mp4": "mp4v",  # MPEG-4
        "avi": "XVID",  # Xvid
        "mov": "mp4v",  # QuickTime
        "mkv": "X264",  # H.264
    }

    def __init__(
        self,
        output_path: str | Path,
        fps: float,
        frame_size: Tuple[int, int],
        fourcc: Optional[str] = None,
        is_color: bool = True,
    ) -> None:
        """Initialize the video writer.

        Args:
            output_path: Path where the output video will be saved.
            fps: Frames per second for the output video.
            frame_size: Video dimensions as (width, height) in pixels.
            fourcc: Four-character codec code (e.g., 'mp4v', 'XVID'). If None,
                   will be inferred from file extension.
            is_color: Whether to write color (True) or grayscale (False) video.

        Raises:
            ValueError: If fps or frame_size are invalid.
        """
        self.output_path = Path(output_path)
        self.fps = fps
        self.frame_size = frame_size
        self.is_color = is_color
        self.frames_written = 0

        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")

        if len(self.frame_size) != 2 or any(d <= 0 for d in self.frame_size):
            raise ValueError(f"Invalid frame_size: {self.frame_size}")

        # Determine codec
        if fourcc is None:
            ext = self.output_path.suffix.lstrip(".").lower()
            fourcc = self.CODECS.get(ext, "mp4v")

        self.fourcc = fourcc
        self._fourcc_code = cv2.VideoWriter_fourcc(*fourcc)

        # Create output directory if it doesn't exist
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize the VideoWriter
        self._writer = cv2.VideoWriter(
            str(self.output_path),
            self._fourcc_code,
            self.fps,
            self.frame_size,
            self.is_color
        )

        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for: {self.output_path}")

    def write_frame(self, frame: Union[np.ndarray, FrameData]) -> bool:
        """Write a single frame to the video file.

        Args:
            frame: Either a numpy array (image) or FrameData object to write.

        Returns:
            True if the frame was written successfully, False otherwise.

        Raises:
            ValueError: If frame dimensions don't match the video size.
            RuntimeError: If the writer is not open.
        """
        if not self.is_opened():
            raise RuntimeError("VideoWriter is not opened")

        # Extract image from FrameData if necessary
        if isinstance(frame, FrameData):
            image = frame.image
        else:
            image = frame

        # Validate frame dimensions
        height, width = image.shape[:2]
        expected_width, expected_height = self.frame_size

        if width != expected_width or height != expected_height:
            raise ValueError(
                f"Frame dimensions ({width}x{height}) don't match video size "
                f"({expected_width}x{expected_height})"
            )

        # Validate color channels
        if self.is_color and (image.ndim != 3 or image.shape[2] != 3):
            raise ValueError(
                f"Expected color image with 3 channels, got shape {image.shape}"
            )

        if not self.is_color and image.ndim != 2:
            raise ValueError(
                f"Expected grayscale image with 2 dimensions, got shape {image.shape}"
            )

        # Write the frame
        self._writer.write(image)
        self.frames_written += 1
        return True

    def write_frames(self, frames: list[Union[np.ndarray, FrameData]]) -> int:
        """Write multiple frames to the video file.

        Args:
            frames: List of frames (numpy arrays or FrameData objects) to write.

        Returns:
            Number of frames successfully written.
        """
        count = 0
        for frame in frames:
            if self.write_frame(frame):
                count += 1
        return count

    def is_opened(self) -> bool:
        """Check if the video writer is open and ready to write.

        Returns:
            True if opened, False otherwise.
        """
        return self._writer is not None and self._writer.isOpened()

    def get_expected_duration(self) -> float:
        """Calculate the expected duration based on frames written.

        Returns:
            Duration in seconds as a float.
        """
        if self.fps > 0:
            return self.frames_written / self.fps
        return 0.0

    def close(self) -> None:
        """Release the video writer resources and finalize the video file."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> "VideoWriter":
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager and release resources."""
        self.close()

    def __repr__(self) -> str:
        """Return string representation of the writer."""
        status = "opened" if self.is_opened() else "closed"
        return (
            f"VideoWriter(path={self.output_path}, "
            f"fps={self.fps:.2f}, size={self.frame_size}, "
            f"codec={self.fourcc}, frames_written={self.frames_written}, "
            f"status={status})"
        )

    def __del__(self) -> None:
        """Destructor to ensure resources are released."""
        self.close()


class AnnotatedVideoWriter(VideoWriter):
    """Extended video writer with built-in annotation capabilities.

    This class extends VideoWriter to provide convenient methods for
    adding common annotations like text overlays, progress bars, and
    frame counters to the output video.

    Example:
        >>> writer = AnnotatedVideoWriter(
        ...     output_path="annotated.mp4",
        ...     fps=30.0,
        ...     frame_size=(1280, 720),
        ...     show_frame_number=True,
        ...     show_timestamp=True
        ... )
        >>> with writer:
        ...     for frame_data in frames:
        ...         writer.write_annotated_frame(frame_data)
    """

    def __init__(
        self,
        output_path: str | Path,
        fps: float,
        frame_size: Tuple[int, int],
        fourcc: Optional[str] = None,
        is_color: bool = True,
        show_frame_number: bool = False,
        show_timestamp: bool = False,
        font_scale: float = 0.7,
        font_thickness: int = 2,
        text_color: Tuple[int, int, int] = (255, 255, 255),
        bg_color: Optional[Tuple[int, int, int]] = (0, 0, 0),
    ) -> None:
        """Initialize the annotated video writer.

        Args:
            output_path: Path where the output video will be saved.
            fps: Frames per second for the output video.
            frame_size: Video dimensions as (width, height) in pixels.
            fourcc: Four-character codec code. If None, inferred from extension.
            is_color: Whether to write color (True) or grayscale (False) video.
            show_frame_number: Whether to overlay frame numbers on output.
            show_timestamp: Whether to overlay timestamps on output.
            font_scale: Scale factor for annotation text.
            font_thickness: Thickness of annotation text.
            text_color: RGB color tuple for annotation text.
            bg_color: RGB color tuple for text background. None for no background.
        """
        super().__init__(output_path, fps, frame_size, fourcc, is_color)

        self.show_frame_number = show_frame_number
        self.show_timestamp = show_timestamp
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.text_color = text_color
        self.bg_color = bg_color
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def write_annotated_frame(self, frame_data: FrameData) -> bool:
        """Write a frame with automatic annotations.

        Args:
            frame_data: FrameData object to write with annotations.

        Returns:
            True if the frame was written successfully, False otherwise.
        """
        # Create a copy to avoid modifying the original
        image = frame_data.image.copy()

        # Add frame number annotation
        if self.show_frame_number:
            text = f"Frame: {frame_data.frame_number}"
            self._add_text_overlay(image, text, position="top-left")

        # Add timestamp annotation
        if self.show_timestamp:
            text = f"Time: {frame_data.timestamp:.2f}s"
            self._add_text_overlay(image, text, position="top-right")

        return self.write_frame(image)

    def _add_text_overlay(
        self,
        image: np.ndarray,
        text: str,
        position: str = "top-left",
        padding: int = 10
    ) -> None:
        """Add text overlay to an image.

        Args:
            image: Image to add text to (modified in-place).
            text: Text string to overlay.
            position: Text position ("top-left", "top-right", "bottom-left", "bottom-right").
            padding: Padding from edges in pixels.
        """
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.font, self.font_scale, self.font_thickness
        )

        # Calculate position
        height, width = image.shape[:2]
        if position == "top-left":
            x, y = padding, padding + text_height
        elif position == "top-right":
            x, y = width - text_width - padding, padding + text_height
        elif position == "bottom-left":
            x, y = padding, height - padding
        elif position == "bottom-right":
            x, y = width - text_width - padding, height - padding
        else:
            x, y = padding, padding + text_height

        # Draw background rectangle if specified
        if self.bg_color is not None:
            cv2.rectangle(
                image,
                (x - 5, y - text_height - 5),
                (x + text_width + 5, y + baseline + 5),
                self.bg_color,
                -1
            )

        # Draw text
        cv2.putText(
            image,
            text,
            (x, y),
            self.font,
            self.font_scale,
            self.text_color,
            self.font_thickness,
            cv2.LINE_AA
        )
