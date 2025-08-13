"""
RTS (Rauch-Tung-Striebel) Smoother Implementation

A dedicated class for high-performance Kalman filtering and RTS smoothing
of tracking data. This class provides a clean, reusable interface for
smoothing trajectory data with configurable parameters.
"""

import numpy as np
from typing import Tuple, Optional


class RTSSmoother:
    """
    A high-performance RTS smoother implementation for trajectory smoothing.
    
    The RTS smoother combines a forward Kalman filter pass with a backward
    smoothing pass to provide optimal estimates of state variables given
    all available data.
    """
    
    def __init__(self, 
                 process_noise_diag: Optional[np.ndarray] = None,
                 measurement_noise_diag: Optional[np.ndarray] = None):
        """
        Initialize the RTS smoother with configurable noise parameters.
        
        Args:
            process_noise_diag: Diagonal values for process noise covariance matrix Q.
                               Default: [0.03, 0.03, 100.0, 100.0, 0.03, 0.03, 50.0, 50.0]
            measurement_noise_diag: Diagonal values for measurement noise covariance matrix R.
                                   Default: [0.1, 0.1, 0.1, 0.1]
        """
        # Default process noise (position, position, width, height, velocity x4)
        if process_noise_diag is None:
            process_noise_diag = np.array([0.03, 0.03, 100.0, 100.0, 0.03, 0.03, 50.0, 50.0])
        
        # Default measurement noise for [cx, cy, w, h]
        if measurement_noise_diag is None:
            measurement_noise_diag = np.array([0.1, 0.1, 0.1, 0.1])
        
        self.q_ext = np.diag(process_noise_diag).astype(np.float32)
        self.r_ext = np.diag(measurement_noise_diag).astype(np.float32)
        
        # Measurement matrix H - maps state to observations
        self.h_ext = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],  # cx
            [0, 1, 0, 0, 0, 0, 0, 0],  # cy
            [0, 0, 1, 0, 0, 0, 0, 0],  # w
            [0, 0, 0, 1, 0, 0, 0, 0]   # h
        ], dtype=np.float32)
    
    def smooth_trajectory(self, track_array: np.ndarray, window_size: Optional[int] = None) -> np.ndarray:
        """
        Apply RTS smoothing to trajectory data with optional windowing.
        
        Args:
            track_array: A 2D NumPy array with columns [frame_id, cx, cy, width, height].
            window_size: Optional window size for localized smoothing (e.g., ±20 frames).
        
        Returns:
            A 2D NumPy array with smoothed [cx, cy, width, height] data.
        """
        if track_array.shape[0] < 2:
            # Not enough data points for smoothing
            return track_array[:, 1:5] if track_array.shape[0] > 0 else np.array([])
        
        if window_size is None:
            # Standard full-track smoothing
            return self._smooth_full_trajectory(track_array)
        else:
            # Windowed smoothing for gap filling
            return self._smooth_windowed_trajectory(track_array, window_size)
    
    def _smooth_full_trajectory(self, track_array: np.ndarray) -> np.ndarray:
        """Standard full trajectory smoothing."""
        n = track_array.shape[0]
        frame_ids = track_array[:, 0]
        measurements = track_array[:, 1:5]  # cx, cy, w, h
        
        # Calculate time deltas
        dts = self._calculate_time_deltas(frame_ids)
        
        # Initialize storage arrays
        x_filtered, P_filtered, x_pred_list, P_pred_list = self._initialize_arrays(n, measurements, dts)
        
        # Forward pass (Kalman filtering)
        self._forward_pass(n, dts, measurements, x_filtered, P_filtered, x_pred_list, P_pred_list)
        
        # Backward pass (RTS smoothing)
        x_smoothed = self._backward_pass(n, dts, x_filtered, P_filtered, x_pred_list, P_pred_list)
        
        return x_smoothed[:, :4, 0]  # Return only [cx, cy, w, h]
    
    def _smooth_windowed_trajectory(self, track_array: np.ndarray, window_size: int) -> np.ndarray:
        """Windowed trajectory smoothing for gap filling."""
        n = track_array.shape[0]
        frame_ids = track_array[:, 0]
        measurements = track_array[:, 1:5]  # cx, cy, w, h
        
        # Create expanded window if possible
        total_frames = n
        start_expand = max(0, -window_size)
        end_expand = min(total_frames, n + window_size)
        
        # Use the available data within the window
        windowed_array = track_array[start_expand:end_expand]
        windowed_result = self._smooth_full_trajectory(windowed_array)
        
        # Return only the original range
        original_start = abs(start_expand)
        original_end = original_start + n
        return windowed_result[original_start:original_end] if windowed_result.shape[0] > original_end else windowed_result
    
    def _calculate_time_deltas(self, frame_ids: np.ndarray) -> np.ndarray:
        """Calculate time deltas between frames, ensuring positive values."""
        dts = np.diff(frame_ids, prepend=frame_ids[0])
        dts[dts <= 0] = 1.0
        return dts
    
    def _initialize_arrays(self, n: int, measurements: np.ndarray, dts: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Initialize state and covariance arrays for filtering."""
        x_filtered = np.zeros((n, 8, 1), dtype=np.float32)
        P_filtered = np.zeros((n, 8, 8), dtype=np.float32)
        x_pred_list = np.zeros((n, 8, 1), dtype=np.float32)
        P_pred_list = np.zeros((n, 8, 8), dtype=np.float32)
        
        # Initial state: position + estimated initial velocity
        if n > 1:
            initial_v = (measurements[1] - measurements[0]) / dts[1]
        else:
            initial_v = np.zeros(4)
        
        x_filtered[0] = np.hstack([measurements[0], initial_v]).reshape(8, 1)
        P_filtered[0] = np.eye(8, dtype=np.float32) * 100.0
        
        return x_filtered, P_filtered, x_pred_list, P_pred_list
    
    def _forward_pass(self, n: int, dts: np.ndarray, measurements: np.ndarray,
                     x_filtered: np.ndarray, P_filtered: np.ndarray,
                     x_pred_list: np.ndarray, P_pred_list: np.ndarray) -> None:
        """Perform forward Kalman filtering pass."""
        A = np.eye(8, dtype=np.float32)
        
        for i in range(1, n):
            dt = dts[i]
            # Update state transition matrix with time step
            A[0, 4] = A[1, 5] = A[2, 6] = A[3, 7] = dt
            
            # Prediction step
            x_pred = A @ x_filtered[i - 1]
            P_pred = A @ P_filtered[i - 1] @ A.T + self.q_ext
            
            x_pred_list[i] = x_pred
            P_pred_list[i] = P_pred
            
            # Update step
            z = measurements[i].reshape(4, 1)
            y = z - (self.h_ext @ x_pred)  # Innovation
            S = self.h_ext @ P_pred @ self.h_ext.T + self.r_ext  # Innovation covariance
            K = P_pred @ self.h_ext.T @ np.linalg.pinv(S)  # Kalman gain
            
            x_filtered[i] = x_pred + K @ y
            P_filtered[i] = (np.eye(8) - K @ self.h_ext) @ P_pred
    
    def _backward_pass(self, n: int, dts: np.ndarray, x_filtered: np.ndarray,
                      P_filtered: np.ndarray, x_pred_list: np.ndarray, P_pred_list: np.ndarray) -> np.ndarray:
        """Perform backward RTS smoothing pass."""
        x_smoothed = np.copy(x_filtered)
        A = np.eye(8, dtype=np.float32)
        
        for i in range(n - 2, -1, -1):
            dt = dts[i + 1]
            A[0, 4] = A[1, 5] = A[2, 6] = A[3, 7] = dt
            
            # Compute smoother gain
            P_pred_inv = np.linalg.pinv(P_pred_list[i + 1])
            J = P_filtered[i] @ A.T @ P_pred_inv
            
            # Backward smoothing update
            x_smoothed[i] += J @ (x_smoothed[i + 1] - x_pred_list[i + 1])
        
        return x_smoothed
    
    def set_process_noise(self, diagonal_values: np.ndarray) -> None:
        """Update the process noise covariance matrix."""
        self.q_ext = np.diag(diagonal_values).astype(np.float32)
    
    def set_measurement_noise(self, diagonal_values: np.ndarray) -> None:
        """Update the measurement noise covariance matrix."""
        self.r_ext = np.diag(diagonal_values).astype(np.float32)
    
    def apply_temporal_size_smoothing(self, smoothed_coords: np.ndarray, frame_ids: np.ndarray, 
                                     fps: float = 30.0) -> np.ndarray:
        """
        Apply temporal size smoothing to resist sudden size changes.
        
        Args:
            smoothed_coords: Array with [cx, cy, w, h] from RTS smoothing
            frame_ids: Frame ID array for time calculations
            fps: Frames per second for adaptive smoothing window
            
        Returns:
            Array with temporally smoothed sizes
        """
        if smoothed_coords.shape[0] < 2:
            return smoothed_coords
            
        # Size smoothing frames (~1 second window)
        size_smoothing_frames = int(fps)
        
        result_coords = smoothed_coords.copy()
        prev_w, prev_h = smoothed_coords[0, 2], smoothed_coords[0, 3]
        prev_frame_id = frame_ids[0]
        
        for i in range(1, smoothed_coords.shape[0]):
            raw_w, raw_h = smoothed_coords[i, 2], smoothed_coords[i, 3]
            current_frame_id = frame_ids[i]
            
            # Calculate time difference
            dt = max(1.0, current_frame_id - prev_frame_id)
            
            # Adaptive smoothing factor
            alpha = dt / (size_smoothing_frames + dt)
            
            # Apply temporal smoothing
            smoothed_w = prev_w + alpha * (raw_w - prev_w)
            smoothed_h = prev_h + alpha * (raw_h - prev_h)
            
            # Update results
            result_coords[i, 2] = smoothed_w
            result_coords[i, 3] = smoothed_h
            
            # Update previous values
            prev_w, prev_h = smoothed_w, smoothed_h
            prev_frame_id = current_frame_id
            
        return result_coords
    
    def get_config(self) -> dict:
        """Get current smoother configuration."""
        return {
            'process_noise_diagonal': np.diag(self.q_ext),
            'measurement_noise_diagonal': np.diag(self.r_ext),
            'measurement_matrix_shape': self.h_ext.shape
        }


# Convenience function for backward compatibility
def run_rts_smoother_numpy(track_array: np.ndarray) -> np.ndarray:
    """
    Convenience function that maintains the original API.
    
    Args:
        track_array: A 2D NumPy array with columns [frame_id, cx, cy, width, height].
    
    Returns:
        A 2D NumPy array with smoothed [cx, cy, width, height] data.
    """
    smoother = RTSSmoother()
    return smoother.smooth_trajectory(track_array)