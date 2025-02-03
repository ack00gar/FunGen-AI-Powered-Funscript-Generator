import numpy as np


def boost_amplitude(signal, boost_factor, min_value=0, max_value=100):
    """
    Boosts the amplitude of a signal and ensures it stays within a specified range.

    Parameters:
        signal (list or np.ndarray): The input signal to boost.
        boost_factor (float): The factor by which to boost the signal's amplitude.
        min_value (int, optional): Minimum value of the range. Default is 0.
        max_value (int, optional): Maximum value of the range. Default is 100.

    Returns:
        np.ndarray: The boosted signal clamped to the specified range.
    """
    # Convert to numpy array for easier manipulation
    signal = np.array(signal)

    # Calculate the midpoint for rescaling
    midpoint = (min_value + max_value) / 2

    # Boost the signal
    boosted_signal = midpoint + (signal - midpoint) * boost_factor

    # Clamp the signal to the specified range
    boosted_signal = np.clip(boosted_signal, min_value, max_value)

    return boosted_signal


def filter_positions(positions, fps):
    """
    Filters positions to remove unnecessary points while preserving key features.

    Args:
        positions (list or np.array): List of [timestamp, value] pairs.
        fps (int): Frames per second of the video.

    Returns:
        list: Filtered list of [timestamp, value] pairs.
    """
    if not positions:
        return []

    # Ensure positions is a list of [timestamp, value] pairs
    positions = np.array(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must be a list of [timestamp, value] pairs")

    filtered_positions = [positions[0]]  # Start with the first position

    min_interval_ms = 50  # Minimum interval between points in milliseconds
    slope_threshold = 0.2  # Adjusted slope threshold for gradual changes

    for i in range(1, len(positions) - 1):
        current_pos = positions[i]
        prev_pos = positions[i - 1]
        next_pos = positions[i + 1]

        # Skip consecutive duplicate positions
        if current_pos[1] == filtered_positions[-1][1] and current_pos[1] == next_pos[1]:
            continue

        # Calculate slopes
        time_diff_prev = current_pos[0] - prev_pos[0]
        time_diff_next = next_pos[0] - current_pos[0]

        slope_prev = (current_pos[1] - prev_pos[1]) / time_diff_prev if time_diff_prev != 0 else 0
        slope_next = (next_pos[1] - current_pos[1]) / time_diff_next if time_diff_next != 0 else 0
        slope_diff = abs(slope_next - slope_prev)

        # Check if the current position is a local extreme
        is_local_extreme = (
                (current_pos[1] >= prev_pos[1] and current_pos[1] > next_pos[1]) or
                (current_pos[1] > prev_pos[1] and current_pos[1] >= next_pos[1]) or
                (current_pos[1] <= prev_pos[1] and current_pos[1] < next_pos[1]) or
                (current_pos[1] < prev_pos[1] and current_pos[1] <= next_pos[1])
        )

        # Calculate time difference in milliseconds
        time_diff_ms = (current_pos[0] - filtered_positions[-1][0]) * 1000 / fps

        # Add to filtered list based on conditions
        if (is_local_extreme or slope_diff > slope_threshold) and time_diff_ms > min_interval_ms:
            filtered_positions.append(current_pos)

    # Ensure the last point meets the interval requirement
    if len(filtered_positions) > 1 and (
            positions[-1][0] - filtered_positions[-1][0]) * 1000 / fps >= min_interval_ms:
        filtered_positions.append(positions[-1])

    return filtered_positions

