"""Frame data structure for video processing.

This module defines the data structures used to represent individual video frames
along with their associated metadata.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class FrameData:
    """Container for video frame data and metadata.

    This dataclass encapsulates a single video frame along with its temporal
    information and any associated metadata from processing steps.

    Attributes:
        frame_number: Zero-indexed frame number in the video sequence.
        timestamp: Time in seconds from the start of the video.
        image: The frame image as a numpy array in BGR format (OpenCV standard).
        metadata: Optional dictionary for storing additional frame-specific data
                 such as detection results, pose estimations, or annotations.

    Example:
        >>> frame = FrameData(
        ...     frame_number=42,
        ...     timestamp=1.4,
        ...     image=np.zeros((720, 1280, 3), dtype=np.uint8),
        ...     metadata={'detected_poses': 2}
        ... )
    """

    frame_number: int
    timestamp: float
    image: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate frame data after initialization."""
        if self.frame_number < 0:
            raise ValueError(f"frame_number must be non-negative, got {self.frame_number}")

        if self.timestamp < 0:
            raise ValueError(f"timestamp must be non-negative, got {self.timestamp}")

        if not isinstance(self.image, np.ndarray):
            raise TypeError(f"image must be a numpy array, got {type(self.image)}")

        if self.image.ndim not in (2, 3):
            raise ValueError(
                f"image must be 2D (grayscale) or 3D (color), got shape {self.image.shape}"
            )

    @property
    def height(self) -> int:
        """Get frame height in pixels."""
        return self.image.shape[0]

    @property
    def width(self) -> int:
        """Get frame width in pixels."""
        return self.image.shape[1]

    @property
    def channels(self) -> int:
        """Get number of color channels (1 for grayscale, 3 for BGR)."""
        return self.image.shape[2] if self.image.ndim == 3 else 1

    @property
    def shape(self) -> tuple:
        """Get frame dimensions as (height, width, channels)."""
        return self.image.shape

    def copy(self) -> "FrameData":
        """Create a deep copy of the frame data.

        Returns:
            A new FrameData instance with copied image and metadata.
        """
        return FrameData(
            frame_number=self.frame_number,
            timestamp=self.timestamp,
            image=self.image.copy(),
            metadata=self.metadata.copy()
        )

    def add_metadata(self, key: str, value: Any) -> None:
        """Add or update a metadata entry.

        Args:
            key: Metadata key.
            value: Metadata value.
        """
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Retrieve a metadata value.

        Args:
            key: Metadata key to retrieve.
            default: Default value if key doesn't exist.

        Returns:
            The metadata value or default if key not found.
        """
        return self.metadata.get(key, default)
