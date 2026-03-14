"""Signal smoothing utilities for gait analysis.

This module provides various smoothing techniques for time-series data including
pose landmarks and derived metrics. Smoothing helps reduce noise and extract
meaningful patterns from noisy sensor data.
"""

from typing import Optional, Union
import numpy as np
import numpy.typing as npt
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.ndimage import gaussian_filter1d


def gaussian_smooth(
    data: npt.NDArray[np.float32],
    sigma: float = 1.0,
    axis: int = 0,
) -> npt.NDArray[np.float32]:
    """Apply Gaussian smoothing to data.

    Gaussian smoothing is good for general noise reduction while preserving
    important features. The sigma parameter controls the amount of smoothing.

    Args:
        data: Input data array (can be 1D or multi-dimensional)
        sigma: Standard deviation of Gaussian kernel (larger = more smoothing)
        axis: Axis along which to apply smoothing (default: 0 for time axis)

    Returns:
        Smoothed data with same shape as input

    Example:
        >>> import numpy as np
        >>> noisy_signal = np.random.randn(100) + np.sin(np.linspace(0, 4*np.pi, 100))
        >>> smooth_signal = gaussian_smooth(noisy_signal, sigma=2.0)
    """
    return gaussian_filter1d(data, sigma=sigma, axis=axis)


def moving_average(
    data: npt.NDArray[np.float32],
    window_size: int = 5,
    mode: str = "valid",
) -> npt.NDArray[np.float32]:
    """Apply moving average smoothing to 1D data.

    Moving average is a simple smoothing technique that replaces each value
    with the average of nearby values.

    Args:
        data: Input 1D data array
        window_size: Size of the averaging window (must be odd for 'same' mode)
        mode: Padding mode - 'valid' (no padding), 'same' (zero-pad), or 'full'

    Returns:
        Smoothed data. Length depends on mode:
        - 'valid': length = len(data) - window_size + 1
        - 'same': length = len(data)
        - 'full': length = len(data) + window_size - 1

    Example:
        >>> data = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
        >>> smoothed = moving_average(data, window_size=3, mode='valid')
    """
    if window_size < 1:
        raise ValueError(f"window_size must be at least 1, got {window_size}")

    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode=mode)


def savitzky_golay(
    data: npt.NDArray[np.float32],
    window_length: int = 11,
    polyorder: int = 3,
    deriv: int = 0,
    axis: int = 0,
) -> npt.NDArray[np.float32]:
    """Apply Savitzky-Golay filter for smoothing and differentiation.

    The Savitzky-Golay filter is particularly good for preserving features
    like peaks and valleys while smoothing. It can also compute derivatives.

    Args:
        data: Input data array
        window_length: Length of filter window (must be odd and > polyorder)
        polyorder: Order of polynomial to fit (typically 2-5)
        deriv: Order of derivative to compute (0 = smoothing only, 1 = first derivative, etc.)
        axis: Axis along which to apply filter

    Returns:
        Filtered/smoothed data or derivative with same shape as input

    Raises:
        ValueError: If window_length is invalid

    Example:
        >>> # Smooth a signal
        >>> smoothed = savitzky_golay(signal, window_length=11, polyorder=3)
        >>> # Compute smoothed first derivative (velocity)
        >>> velocity = savitzky_golay(position, window_length=11, polyorder=3, deriv=1)
    """
    if window_length < 1 or window_length % 2 == 0:
        raise ValueError(f"window_length must be odd and >= 1, got {window_length}")

    if polyorder >= window_length:
        raise ValueError(
            f"polyorder ({polyorder}) must be less than window_length ({window_length})"
        )

    return savgol_filter(data, window_length, polyorder, deriv=deriv, axis=axis)


def butterworth_filter(
    data: npt.NDArray[np.float32],
    cutoff_freq: float,
    sampling_freq: float,
    order: int = 4,
    filter_type: str = "low",
) -> npt.NDArray[np.float32]:
    """Apply Butterworth filter for frequency-based smoothing.

    Butterworth filters are excellent for removing high-frequency noise while
    preserving low-frequency trends. They provide a flat frequency response
    in the passband.

    Args:
        data: Input 1D data array
        cutoff_freq: Cutoff frequency in Hz (for low/high pass) or tuple for bandpass
        sampling_freq: Sampling frequency of the data in Hz
        order: Filter order (higher = sharper cutoff, default: 4)
        filter_type: Type of filter - 'low', 'high', 'bandpass', or 'bandstop'

    Returns:
        Filtered data with same shape as input

    Example:
        >>> # Remove high-frequency noise (keep frequencies below 5 Hz)
        >>> smoothed = butterworth_filter(signal, cutoff_freq=5.0, sampling_freq=30.0)
        >>> # Remove low-frequency drift (keep frequencies above 0.5 Hz)
        >>> detrended = butterworth_filter(signal, cutoff_freq=0.5, sampling_freq=30.0, filter_type='high')
    """
    nyquist = sampling_freq / 2.0

    if isinstance(cutoff_freq, (list, tuple)):
        # Bandpass or bandstop
        normalized_cutoff = [f / nyquist for f in cutoff_freq]
    else:
        # Low or high pass
        normalized_cutoff = cutoff_freq / nyquist

    if filter_type not in ["low", "high", "bandpass", "bandstop"]:
        raise ValueError(
            f"filter_type must be 'low', 'high', 'bandpass', or 'bandstop', got {filter_type}"
        )

    # Design filter
    b, a = butter(order, normalized_cutoff, btype=filter_type)

    # Apply filter (forward and backward to avoid phase shift)
    filtered = filtfilt(b, a, data)

    return filtered


def exponential_smoothing(
    data: npt.NDArray[np.float32],
    alpha: float = 0.3,
) -> npt.NDArray[np.float32]:
    """Apply exponential smoothing (exponentially weighted moving average).

    This gives more weight to recent observations while still considering
    historical data. Good for real-time applications.

    Args:
        data: Input 1D data array
        alpha: Smoothing factor between 0 and 1
               - Close to 0: More smoothing (slower response)
               - Close to 1: Less smoothing (faster response)

    Returns:
        Smoothed data with same shape as input

    Raises:
        ValueError: If alpha is not in valid range

    Example:
        >>> # Smooth with moderate response to changes
        >>> smoothed = exponential_smoothing(signal, alpha=0.3)
    """
    if not 0 < alpha <= 1:
        raise ValueError(f"alpha must be in range (0, 1], got {alpha}")

    result = np.zeros_like(data)
    result[0] = data[0]

    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]

    return result


def median_filter(
    data: npt.NDArray[np.float32],
    window_size: int = 5,
) -> npt.NDArray[np.float32]:
    """Apply median filter for robust outlier removal.

    Median filtering is particularly good at removing outliers and spike noise
    while preserving edges. It's more robust than mean-based filters.

    Args:
        data: Input 1D data array
        window_size: Size of the sliding window (must be odd)

    Returns:
        Filtered data with same length as input

    Raises:
        ValueError: If window_size is invalid

    Example:
        >>> # Remove outliers from noisy signal
        >>> cleaned = median_filter(noisy_signal, window_size=5)
    """
    from scipy.ndimage import median_filter as scipy_median

    if window_size < 1 or window_size % 2 == 0:
        raise ValueError(f"window_size must be odd and >= 1, got {window_size}")

    return scipy_median(data, size=window_size)


def smooth_trajectory(
    trajectory: npt.NDArray[np.float32],
    method: str = "savgol",
    **kwargs,
) -> npt.NDArray[np.float32]:
    """Smooth a multi-dimensional trajectory (e.g., pose landmark positions over time).

    This is a convenience function that applies smoothing to each dimension
    of a trajectory independently.

    Args:
        trajectory: Array of shape (n_frames, n_dimensions)
        method: Smoothing method - 'savgol', 'gaussian', 'butterworth', or 'median'
        **kwargs: Additional arguments passed to the chosen smoothing function

    Returns:
        Smoothed trajectory with same shape as input

    Example:
        >>> # Smooth ankle position over time (100 frames x 3 coordinates)
        >>> ankle_positions = np.random.randn(100, 3)
        >>> smooth_ankle = smooth_trajectory(ankle_positions, method='savgol', window_length=11)
    """
    if trajectory.ndim == 1:
        # 1D data, just apply smoothing directly
        if method == "savgol":
            return savitzky_golay(trajectory, **kwargs)
        elif method == "gaussian":
            return gaussian_smooth(trajectory, **kwargs)
        elif method == "butterworth":
            return butterworth_filter(trajectory, **kwargs)
        elif method == "median":
            return median_filter(trajectory, **kwargs)
        else:
            raise ValueError(f"Unknown smoothing method: {method}")

    # Multi-dimensional data, smooth each dimension
    smoothed = np.zeros_like(trajectory)

    for dim in range(trajectory.shape[1]):
        if method == "savgol":
            smoothed[:, dim] = savitzky_golay(trajectory[:, dim], **kwargs)
        elif method == "gaussian":
            smoothed[:, dim] = gaussian_smooth(trajectory[:, dim], **kwargs)
        elif method == "butterworth":
            smoothed[:, dim] = butterworth_filter(trajectory[:, dim], **kwargs)
        elif method == "median":
            smoothed[:, dim] = median_filter(trajectory[:, dim], **kwargs)
        else:
            raise ValueError(f"Unknown smoothing method: {method}")

    return smoothed


def adaptive_smooth(
    data: npt.NDArray[np.float32],
    base_window: int = 5,
    sensitivity: float = 1.0,
) -> npt.NDArray[np.float32]:
    """Apply adaptive smoothing with variable window size based on local variance.

    This method uses larger windows in smooth regions and smaller windows
    near features/transitions, preserving detail while reducing noise.

    Args:
        data: Input 1D data array
        base_window: Base window size for smoothing
        sensitivity: Controls adaptation sensitivity (higher = more adaptive)

    Returns:
        Adaptively smoothed data with same length as input

    Example:
        >>> # Smooth with adaptation to local signal characteristics
        >>> smoothed = adaptive_smooth(signal, base_window=7, sensitivity=1.5)
    """
    result = np.zeros_like(data)
    n = len(data)

    # Calculate local variance
    local_var = np.zeros(n)
    for i in range(n):
        start = max(0, i - base_window // 2)
        end = min(n, i + base_window // 2 + 1)
        local_var[i] = np.var(data[start:end])

    # Normalize variance
    max_var = np.max(local_var) if np.max(local_var) > 0 else 1.0
    local_var = local_var / max_var

    # Apply adaptive smoothing
    for i in range(n):
        # Adapt window size based on local variance
        adapt_factor = 1.0 - sensitivity * local_var[i]
        adapt_factor = np.clip(adapt_factor, 0.3, 1.0)
        window = max(3, int(base_window * adapt_factor))

        # Ensure window is odd
        if window % 2 == 0:
            window += 1

        # Apply smoothing with adapted window
        start = max(0, i - window // 2)
        end = min(n, i + window // 2 + 1)
        result[i] = np.mean(data[start:end])

    return result


def interpolate_missing_values(
    data: npt.NDArray[np.float32],
    missing_mask: Optional[npt.NDArray[np.bool_]] = None,
    method: str = "linear",
) -> npt.NDArray[np.float32]:
    """Interpolate missing or invalid values in data.

    This is useful for handling frames where pose detection failed or
    landmarks have low visibility.

    Args:
        data: Input data array with missing values (as NaN or specified by mask)
        missing_mask: Boolean mask indicating missing values (True = missing)
                     If None, NaN values are considered missing
        method: Interpolation method - 'linear', 'cubic', or 'nearest'

    Returns:
        Data with missing values interpolated

    Example:
        >>> # Interpolate NaN values in trajectory
        >>> data_with_gaps = np.array([1.0, 2.0, np.nan, np.nan, 5.0, 6.0])
        >>> filled = interpolate_missing_values(data_with_gaps, method='linear')
    """
    from scipy.interpolate import interp1d

    if missing_mask is None:
        missing_mask = np.isnan(data)

    if not np.any(missing_mask):
        # No missing values
        return data.copy()

    if np.all(missing_mask):
        # All values missing, can't interpolate
        return data.copy()

    result = data.copy()
    valid_indices = np.where(~missing_mask)[0]
    valid_values = data[~missing_mask]

    # Create interpolation function
    if len(valid_indices) < 2:
        # Not enough points for interpolation, use nearest
        method = "nearest"

    if method == "cubic" and len(valid_indices) < 4:
        # Cubic needs at least 4 points
        method = "linear"

    interpolator = interp1d(
        valid_indices,
        valid_values,
        kind=method,
        bounds_error=False,
        fill_value=(valid_values[0], valid_values[-1]),
    )

    # Interpolate missing values
    missing_indices = np.where(missing_mask)[0]
    result[missing_indices] = interpolator(missing_indices)

    return result
