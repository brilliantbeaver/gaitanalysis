"""Geometric calculations for gait analysis.

This module provides utilities for geometric computations including angle calculations,
distance measurements, and coordinate transformations commonly used in pose analysis.
"""

from typing import Tuple, Optional
import numpy as np
import numpy.typing as npt


def calculate_angle(
    point1: npt.NDArray[np.float32],
    point2: npt.NDArray[np.float32],
    point3: npt.NDArray[np.float32],
) -> float:
    """Calculate angle between three points using vectors.

    The angle is calculated at point2 (the vertex) formed by the lines
    connecting point1-point2 and point2-point3.

    Args:
        point1: First point as array [x, y] or [x, y, z]
        point2: Vertex point as array [x, y] or [x, y, z]
        point3: Third point as array [x, y] or [x, y, z]

    Returns:
        Angle in degrees (0-180)

    Example:
        >>> import numpy as np
        >>> hip = np.array([0.5, 0.5, 0.0])
        >>> knee = np.array([0.5, 0.7, 0.0])
        >>> ankle = np.array([0.5, 0.9, 0.0])
        >>> angle = calculate_angle(hip, knee, ankle)
        >>> print(f"Knee angle: {angle:.1f} degrees")
    """
    # Create vectors from vertex to other points
    vector1 = point1 - point2
    vector2 = point3 - point2

    # Calculate norms
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    # Check for zero-length vectors
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0

    # Calculate dot product and angle
    cos_angle = np.dot(vector1, vector2) / (norm1 * norm2)

    # Clip to valid range to avoid numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Convert to degrees
    angle = np.degrees(np.arccos(cos_angle))

    return float(angle)


def calculate_angle_2d(
    point1: npt.NDArray[np.float32],
    point2: npt.NDArray[np.float32],
    point3: npt.NDArray[np.float32],
) -> float:
    """Calculate angle between three points in 2D plane (ignoring z-coordinate).

    This is useful when analyzing gait from a side view where only x-y coordinates matter.

    Args:
        point1: First point as array [x, y] or [x, y, z]
        point2: Vertex point as array [x, y] or [x, y, z]
        point3: Third point as array [x, y] or [x, y, z]

    Returns:
        Angle in degrees (0-180)
    """
    # Use only x and y coordinates
    p1_2d = point1[:2]
    p2_2d = point2[:2]
    p3_2d = point3[:2]

    return calculate_angle(p1_2d, p2_2d, p3_2d)


def calculate_signed_angle(
    point1: npt.NDArray[np.float32],
    point2: npt.NDArray[np.float32],
    point3: npt.NDArray[np.float32],
) -> float:
    """Calculate signed angle between three points in 2D.

    The sign indicates the direction of rotation from vector1 to vector2.
    Positive values indicate counter-clockwise rotation, negative clockwise.

    Args:
        point1: First point as array [x, y] or [x, y, z]
        point2: Vertex point as array [x, y] or [x, y, z]
        point3: Third point as array [x, y] or [x, y, z]

    Returns:
        Signed angle in degrees (-180 to 180)
    """
    # Create 2D vectors
    v1 = point1[:2] - point2[:2]
    v2 = point3[:2] - point2[:2]

    # Calculate angle using atan2
    angle1 = np.arctan2(v1[1], v1[0])
    angle2 = np.arctan2(v2[1], v2[0])

    # Calculate difference
    angle_diff = angle2 - angle1

    # Normalize to [-pi, pi]
    angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

    return float(np.degrees(angle_diff))


def euclidean_distance(
    point1: npt.NDArray[np.float32],
    point2: npt.NDArray[np.float32],
) -> float:
    """Calculate Euclidean distance between two points.

    Args:
        point1: First point as array [x, y] or [x, y, z]
        point2: Second point as array [x, y] or [x, y, z]

    Returns:
        Euclidean distance between points
    """
    return float(np.linalg.norm(point1 - point2))


def euclidean_distance_2d(
    point1: npt.NDArray[np.float32],
    point2: npt.NDArray[np.float32],
) -> float:
    """Calculate Euclidean distance between two points in 2D (ignoring z).

    Args:
        point1: First point as array [x, y] or [x, y, z]
        point2: Second point as array [x, y] or [x, y, z]

    Returns:
        2D Euclidean distance between points
    """
    return float(np.linalg.norm(point1[:2] - point2[:2]))


def point_to_line_distance(
    point: npt.NDArray[np.float32],
    line_point1: npt.NDArray[np.float32],
    line_point2: npt.NDArray[np.float32],
) -> float:
    """Calculate perpendicular distance from a point to a line.

    The line is defined by two points. Works in 2D and 3D.

    Args:
        point: Point to measure distance from
        line_point1: First point defining the line
        line_point2: Second point defining the line

    Returns:
        Perpendicular distance from point to line
    """
    # Vector from line_point1 to line_point2
    line_vec = line_point2 - line_point1
    line_length = np.linalg.norm(line_vec)

    if line_length < 1e-8:
        # Line is actually a point
        return euclidean_distance(point, line_point1)

    # Vector from line_point1 to point
    point_vec = point - line_point1

    # Calculate perpendicular distance using cross product
    cross = np.cross(line_vec, point_vec)
    distance = np.linalg.norm(cross) / line_length

    return float(distance)


def calculate_velocity(
    position1: npt.NDArray[np.float32],
    position2: npt.NDArray[np.float32],
    time_delta: float,
) -> npt.NDArray[np.float32]:
    """Calculate velocity vector between two positions.

    Args:
        position1: Starting position as array [x, y] or [x, y, z]
        position2: Ending position as array [x, y] or [x, y, z]
        time_delta: Time difference in seconds

    Returns:
        Velocity vector in units per second

    Raises:
        ValueError: If time_delta is zero or negative
    """
    if time_delta <= 0:
        raise ValueError(f"time_delta must be positive, got {time_delta}")

    displacement = position2 - position1
    velocity = displacement / time_delta

    return velocity


def calculate_speed(
    position1: npt.NDArray[np.float32],
    position2: npt.NDArray[np.float32],
    time_delta: float,
) -> float:
    """Calculate speed (magnitude of velocity) between two positions.

    Args:
        position1: Starting position as array [x, y] or [x, y, z]
        position2: Ending position as array [x, y] or [x, y, z]
        time_delta: Time difference in seconds

    Returns:
        Speed in units per second

    Raises:
        ValueError: If time_delta is zero or negative
    """
    velocity = calculate_velocity(position1, position2, time_delta)
    return float(np.linalg.norm(velocity))


def normalize_vector(vector: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Normalize a vector to unit length.

    Args:
        vector: Input vector as numpy array

    Returns:
        Normalized vector with length 1.0

    Raises:
        ValueError: If vector has zero length
    """
    norm = np.linalg.norm(vector)

    if norm < 1e-8:
        raise ValueError("Cannot normalize zero-length vector")

    return vector / norm


def rotation_matrix_2d(angle_degrees: float) -> npt.NDArray[np.float32]:
    """Create 2D rotation matrix for given angle.

    Args:
        angle_degrees: Rotation angle in degrees (positive = counter-clockwise)

    Returns:
        2x2 rotation matrix
    """
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    return np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)


def rotate_point_2d(
    point: npt.NDArray[np.float32],
    angle_degrees: float,
    origin: Optional[npt.NDArray[np.float32]] = None,
) -> npt.NDArray[np.float32]:
    """Rotate a 2D point around an origin.

    Args:
        point: Point to rotate as [x, y]
        angle_degrees: Rotation angle in degrees (positive = counter-clockwise)
        origin: Center of rotation as [x, y]. If None, uses [0, 0]

    Returns:
        Rotated point as [x, y]
    """
    if origin is None:
        origin = np.array([0.0, 0.0], dtype=np.float32)

    # Translate to origin
    translated = point[:2] - origin[:2]

    # Rotate
    rot_matrix = rotation_matrix_2d(angle_degrees)
    rotated = rot_matrix @ translated

    # Translate back
    result = rotated + origin[:2]

    return result


def calculate_centroid(points: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Calculate centroid (center of mass) of multiple points.

    Args:
        points: Array of points with shape (n_points, n_dimensions)

    Returns:
        Centroid point with same dimensionality as input points

    Raises:
        ValueError: If points array is empty
    """
    if len(points) == 0:
        raise ValueError("Cannot calculate centroid of empty point set")

    return np.mean(points, axis=0)


def calculate_bounding_box(
    points: npt.NDArray[np.float32],
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Calculate axis-aligned bounding box for a set of points.

    Args:
        points: Array of points with shape (n_points, n_dimensions)

    Returns:
        Tuple of (min_point, max_point) defining the bounding box

    Raises:
        ValueError: If points array is empty
    """
    if len(points) == 0:
        raise ValueError("Cannot calculate bounding box of empty point set")

    min_point = np.min(points, axis=0)
    max_point = np.max(points, axis=0)

    return min_point, max_point


def scale_points(
    points: npt.NDArray[np.float32],
    scale_factor: float,
    center: Optional[npt.NDArray[np.float32]] = None,
) -> npt.NDArray[np.float32]:
    """Scale points by a factor around a center point.

    Args:
        points: Array of points with shape (n_points, n_dimensions)
        scale_factor: Scaling factor (e.g., 2.0 for doubling)
        center: Center point for scaling. If None, uses centroid

    Returns:
        Scaled points with same shape as input

    Raises:
        ValueError: If scale_factor is zero or negative
    """
    if scale_factor <= 0:
        raise ValueError(f"scale_factor must be positive, got {scale_factor}")

    if center is None:
        center = calculate_centroid(points)

    # Translate to center, scale, translate back
    translated = points - center
    scaled = translated * scale_factor
    result = scaled + center

    return result


def project_point_onto_line(
    point: npt.NDArray[np.float32],
    line_point1: npt.NDArray[np.float32],
    line_point2: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Project a point onto a line (find closest point on line).

    Args:
        point: Point to project
        line_point1: First point defining the line
        line_point2: Second point defining the line

    Returns:
        Projected point on the line
    """
    # Vector from line_point1 to line_point2
    line_vec = line_point2 - line_point1
    line_length_sq = np.dot(line_vec, line_vec)

    if line_length_sq < 1e-8:
        # Line is actually a point
        return line_point1.copy()

    # Vector from line_point1 to point
    point_vec = point - line_point1

    # Calculate projection parameter
    t = np.dot(point_vec, line_vec) / line_length_sq

    # Calculate projected point
    projected = line_point1 + t * line_vec

    return projected
