"""Video reading functionality for gait analysis.

This module provides classes for reading video files and extracting frames
with their associated metadata.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .frame import FrameData


class BaseVideoReader(ABC):
    """Abstract base class for video readers.

    This class defines the interface that all video readers must implement,
    ensuring consistent behavior across different video sources.
    """

    @abstractmethod
    def read_frame(self) -> Optional[FrameData]:
        """Read the next frame from the video source.

        Returns:
            FrameData object containing the frame and its metadata, or None if
            no more frames are available.
        """
        pass

    @abstractmethod
    def get_fps(self) -> float:
        """Get the frames per second of the video.

        Returns:
            Frame rate as a float.
        """
        pass

    @abstractmethod
    def get_frame_count(self) -> int:
        """Get the total number of frames in the video.

        Returns:
            Total frame count as an integer.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Release resources associated with the video reader."""
        pass

    @abstractmethod
    def seek(self, frame_number: int) -> bool:
        """Seek to a specific frame in the video.

        Args:
            frame_number: Zero-indexed frame number to seek to.

        Returns:
            True if seek was successful, False otherwise.
        """
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """Check if the video source is opened and ready to read.

        Returns:
            True if opened, False otherwise.
        """
        pass

    def __enter__(self) -> "BaseVideoReader":
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager and release resources."""
        self.close()


class VideoFileReader(BaseVideoReader):
    """Read frames from a video file using OpenCV.

    This class provides functionality to read video files frame-by-frame,
    extracting both the image data and temporal metadata.

    Attributes:
        video_path: Path to the video file.
        current_frame_number: Current position in the video (zero-indexed).

    Example:
        >>> with VideoFileReader("path/to/video.mp4") as reader:
        ...     print(f"FPS: {reader.get_fps()}")
        ...     print(f"Total frames: {reader.get_frame_count()}")
        ...     while True:
        ...         frame = reader.read_frame()
        ...         if frame is None:
        ...             break
        ...         # Process frame
    """

    def __init__(self, video_path: str | Path) -> None:
        """Initialize the video file reader.

        Args:
            video_path: Path to the video file to read.

        Raises:
            FileNotFoundError: If the video file doesn't exist.
            ValueError: If the video file cannot be opened.
        """
        self.video_path = Path(video_path)

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        self._cap = cv2.VideoCapture(str(self.video_path))

        if not self._cap.isOpened():
            raise ValueError(f"Failed to open video file: {self.video_path}")

        self.current_frame_number = 0
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read_frame(self) -> Optional[FrameData]:
        """Read the next frame from the video file.

        Returns:
            FrameData object containing the frame image and metadata, or None
            if the end of the video is reached or an error occurs.
        """
        if not self.is_opened():
            return None

        ret, frame = self._cap.read()

        if not ret or frame is None:
            return None

        timestamp = self.current_frame_number / self._fps if self._fps > 0 else 0.0

        frame_data = FrameData(
            frame_number=self.current_frame_number,
            timestamp=timestamp,
            image=frame,
            metadata={
                "source": str(self.video_path),
                "width": self._width,
                "height": self._height,
                "fps": self._fps,
            }
        )

        self.current_frame_number += 1
        return frame_data

    def get_fps(self) -> float:
        """Get the frames per second of the video.

        Returns:
            Frame rate as a float.
        """
        return self._fps

    def get_frame_count(self) -> int:
        """Get the total number of frames in the video.

        Returns:
            Total frame count as an integer.
        """
        return self._frame_count

    def get_resolution(self) -> Tuple[int, int]:
        """Get the video resolution.

        Returns:
            Tuple of (width, height) in pixels.
        """
        return (self._width, self._height)

    def get_duration(self) -> float:
        """Get the video duration in seconds.

        Returns:
            Duration in seconds as a float.
        """
        if self._fps > 0:
            return self._frame_count / self._fps
        return 0.0

    def seek(self, frame_number: int) -> bool:
        """Seek to a specific frame in the video.

        Args:
            frame_number: Zero-indexed frame number to seek to.

        Returns:
            True if seek was successful, False otherwise.
        """
        if not self.is_opened():
            return False

        if frame_number < 0 or frame_number >= self._frame_count:
            return False

        success = self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        if success:
            self.current_frame_number = frame_number

        return success

    def is_opened(self) -> bool:
        """Check if the video file is opened and ready to read.

        Returns:
            True if opened, False otherwise.
        """
        return self._cap is not None and self._cap.isOpened()

    def reset(self) -> bool:
        """Reset the reader to the beginning of the video.

        Returns:
            True if reset was successful, False otherwise.
        """
        return self.seek(0)

    def close(self) -> None:
        """Release the video capture resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __repr__(self) -> str:
        """Return string representation of the reader."""
        status = "opened" if self.is_opened() else "closed"
        return (
            f"VideoFileReader(path={self.video_path}, "
            f"fps={self._fps:.2f}, frames={self._frame_count}, "
            f"resolution={self._width}x{self._height}, status={status})"
        )
