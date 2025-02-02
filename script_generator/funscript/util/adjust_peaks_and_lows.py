import numpy as np


def adjust_peaks_and_lows(positions, peak_boost=10, low_reduction=10, max_flat_length=5):
    """
    Adjusts the peaks and lows of a funscript while avoiding long flat sections at 0 or 100.

    Args:
        positions (list or np.array): List or array of positions (0-100) from the funscript.
        peak_boost (int): Amount to increase peaks by.
        low_reduction (int): Amount to decrease lows by.
        max_flat_length (int): Maximum allowed length of flat sections at 0 or 100.

    Returns:
        list: Adjusted positions.
    """

    # Ensure positions is a list of [timestamp, value] pairs
    if isinstance(positions, np.ndarray):
        positions = positions.tolist()
    elif not isinstance(positions, list):
        raise ValueError("positions must be a list or numpy array of [timestamp, value] pairs")

    if not positions or len(positions) < 3:
        return positions

    # Convert positions to a numpy array for easier manipulation
    positions = np.array(positions, dtype=float)

    # Identify plateaus before boosting
    original_plateaus = _find_plateaus(positions)
    # Identify peaks and lows
    peaks = _find_local_maxima(positions)
    lows = _find_local_minima(positions)
    # Adjust peaks and lows
    positions[peaks] = np.clip(positions[peaks] + peak_boost, 0, 100)
    positions[lows] = np.clip(positions[lows] - low_reduction, 0, 100)
    # Identify plateaus after boosting
    adjusted_plateaus = _find_plateaus(positions)
    # Compare plateaus and adjust artificially created flats
    positions = _compare_and_adjust_plateaus(positions, original_plateaus, adjusted_plateaus, max_flat_length)
    return positions.tolist()

def _find_local_maxima(positions):
    """
    Identifies local maxima (peaks) in the positions.

    Args:
        positions (np.array): Array of positions.

    Returns:
        np.array: Boolean array where True indicates a peak.
    """
    peaks = np.zeros_like(positions, dtype=bool)
    for i in range(1, len(positions) - 1):
        if positions[i] > positions[i - 1] and positions[i] > positions[i + 1]:
            peaks[i] = True
    return peaks

def _find_local_minima(positions):
    """
    Identifies local minima (lows) in the positions.

    Args:
        positions (np.array): Array of positions.

    Returns:
        np.array: Boolean array where True indicates a low.
    """
    lows = np.zeros_like(positions, dtype=bool)
    for i in range(1, len(positions) - 1):
        if positions[i] < positions[i - 1] and positions[i] < positions[i + 1]:
            lows[i] = True
    return lows

def _find_plateaus(positions):
    """
    Identifies flat sections (plateaus) in the positions.

    Args:
        positions (np.array): Array of positions.

    Returns:
        list: List of tuples (start, end) representing the indices of plateaus.
    """
    plateaus = []
    start = 0
    for i in range(1, len(positions)):
        if positions[i] != positions[i - 1]:
            if i - start > 1:  # Plateau must have at least 2 points
                plateaus.append((start, i - 1))
            start = i
    if len(positions) - start > 1:  # Check the last plateau
        plateaus.append((start, len(positions) - 1))
    return plateaus

def _compare_and_adjust_plateaus(positions, original_plateaus, adjusted_plateaus, max_flat_length):
    """
    Compares plateaus before and after adjustments and breaks artificially created flats.

    Args:
        positions (np.array): Array of positions.
        original_plateaus (list): List of plateaus before adjustments.
        adjusted_plateaus (list): List of plateaus after adjustments.
        max_flat_length (int): Maximum allowed length of flat sections at 0 or 100.

    Returns:
        np.array: Adjusted positions.
    """
    positions = positions.copy()  # Work on a copy to avoid modifying the original array
    for plateau in adjusted_plateaus:
        start, end = plateau
        value = positions[start]

        # Check if the plateau is at 0 or 100 and was not present in the original data
        if (value == 0 or value == 100) and not _is_plateau_in_original(plateau, original_plateaus):
            # Check if the plateau exceeds the maximum allowed length
            if end - start + 1 > max_flat_length:
                # Break the plateau by adjusting the values
                for i in range(start, end + 1):
                    positions[i] = positions[i] + 1 if value == 100 else positions[i] - 1

    return positions

def _is_plateau_in_original(plateau, original_plateaus):
    """
    Checks if a plateau was present in the original data.

    Args:
        plateau (tuple): Tuple (start, end) representing the indices of the plateau.
        original_plateaus (list): List of plateaus in the original data.

    Returns:
        bool: True if the plateau was present in the original data, False otherwise.
    """
    start, end = plateau
    for original_start, original_end in original_plateaus:
        if start >= original_start and end <= original_end:
            return True
    return False