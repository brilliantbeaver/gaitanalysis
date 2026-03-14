"""Video processing module for running gait analysis.

This module provides classes and utilities for reading video files,
extracting frames, and writing annotated output videos for gait analysis.

Key Components:
    - FrameData: Data structure for video frames with metadata
    - BaseVideoReader: Abstract interface for video readers
    - VideoFileReader: Concrete implementation for reading video files
    - VideoWriter: Write frames to output video files
    - AnnotatedVideoWriter: Extended writer with annotation capabilities

Example:
    >>> from asdrp.video import VideoFileReader, AnnotatedVideoWriter
    >>>
    >>> # Read input video
    >>> with VideoFileReader("input.mp4") as reader:
    ...     fps = reader.get_fps()
    ...     resolution = reader.get_resolution()
    ...
    ...     # Write annotated output
    ...     with AnnotatedVideoWriter(
    ...         "output.mp4", fps, resolution,
    ...         show_frame_number=True,
    ...         show_timestamp=True
    ...     ) as writer:
    ...         while True:
    ...             frame = reader.read_frame()
    ...             if frame is None:
    ...                 break
    ...             # Process frame (e.g., pose detection)
    ...             writer.write_annotated_frame(frame)
"""

from .frame import FrameData
from .reader import BaseVideoReader, VideoFileReader
from .writer import AnnotatedVideoWriter, VideoWriter

__all__ = [
    "FrameData",
    "BaseVideoReader",
    "VideoFileReader",
    "VideoWriter",
    "AnnotatedVideoWriter",
]

__version__ = "0.1.0"
