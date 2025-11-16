"""
OpenCV-based thumbnail extractor for fast random frame access.

This module provides efficient thumbnail generation using OpenCV's VideoCapture,
which maintains a persistent video connection for faster seeking compared to
spawning new FFmpeg processes.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple
from threading import Lock


class ThumbnailExtractor:
    """
    Fast thumbnail extractor using OpenCV VideoCapture.

    Advantages over FFmpeg spawning:
    - Persistent video connection (no process creation overhead)
    - Faster seeking for random frame access
    - Optional GPU unwarp integration for VR content
    """

    def __init__(self, video_path: str, logger: Optional[logging.Logger] = None,
                 gpu_unwarp_worker=None, output_size: int = 640,
                 vr_input_format: str = None):
        """
        Initialize thumbnail extractor.

        Args:
            video_path: Path to video file
            logger: Optional logger instance
            gpu_unwarp_worker: Optional GPU unwarp worker for VR content
            output_size: Size for output thumbnails (width and height)
            vr_input_format: VR format (e.g., 'fisheye_sbs', 'he_tb') for panel cropping.
                           If None, video is treated as 2D and will be padded to square.
        """
        self.video_path = video_path
        self.logger = logger or logging.getLogger(__name__)
        self.gpu_unwarp_worker = gpu_unwarp_worker
        self.output_size = output_size
        self.vr_input_format = vr_input_format

        # Thread-safe VideoCapture access
        self.lock = Lock()
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_open = False

        # Video properties
        self.fps = 0.0
        self.total_frames = 0
        self.width = 0
        self.height = 0

        # VR format detection
        # SBS-left includes: _sbs, _lr (left-right) - crop to left panel
        # SBS-right includes: _rl (right-left) - crop to right panel
        # TB includes: _tb (top-bottom) - crop to top panel
        vr_fmt = (vr_input_format or '').lower()
        self.is_sbs_left = '_sbs' in vr_fmt or '_lr' in vr_fmt
        self.is_sbs_right = '_rl' in vr_fmt
        self.is_tb = '_tb' in vr_fmt

        # 2D video handling (if vr_input_format is None, treat as 2D)
        self.is_2d = vr_input_format is None

        # Open video
        self._open_video()

    def _open_video(self) -> bool:
        """Open video with OpenCV VideoCapture."""
        try:
            with self.lock:
                self.cap = cv2.VideoCapture(self.video_path)

                if not self.cap.isOpened():
                    self.logger.error(f"ThumbnailExtractor: Failed to open video: {self.video_path}")
                    return False

                # Get video properties
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                self.is_open = True
                self.logger.info(
                    f"ThumbnailExtractor opened: {self.width}x{self.height} @ {self.fps:.2f} FPS, "
                    f"{self.total_frames} frames"
                )
                return True

        except Exception as e:
            self.logger.error(f"ThumbnailExtractor: Error opening video: {e}")
            return False

    def get_frame(self, frame_index: int, use_gpu_unwarp: bool = True) -> Optional[np.ndarray]:
        """
        Extract a single frame at the specified index.

        Args:
            frame_index: Frame index to extract
            use_gpu_unwarp: Whether to apply GPU unwarp for VR content (if worker available)

        Returns:
            Frame as numpy array (BGR24 format, self.output_size x self.output_size)
            or None if extraction failed
        """
        if not self.is_open or self.cap is None:
            self.logger.warning("ThumbnailExtractor: Video not open")
            return None

        # Validate frame index
        if frame_index < 0 or (self.total_frames > 0 and frame_index >= self.total_frames):
            self.logger.warning(
                f"ThumbnailExtractor: Frame index {frame_index} out of range [0, {self.total_frames})"
            )
            return None

        try:
            with self.lock:
                # Seek to frame - this is much faster than spawning FFmpeg!
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

                # Read frame
                ret, frame = self.cap.read()

                if not ret or frame is None:
                    self.logger.warning(f"ThumbnailExtractor: Failed to read frame {frame_index}")
                    return None

            # Crop VR panel if needed (left/right for SBS, top for TB)
            if self.is_sbs_left and frame.shape[1] > 0:
                # Side-by-side (SBS/LR): crop to left half
                half_width = frame.shape[1] // 2
                frame = frame[:, :half_width]
                self.logger.debug(f"Cropped SBS/LR to left panel: {frame.shape}")
            elif self.is_sbs_right and frame.shape[1] > 0:
                # Right-left (RL): crop to right half
                half_width = frame.shape[1] // 2
                frame = frame[:, half_width:]
                self.logger.debug(f"Cropped RL to right panel: {frame.shape}")
            elif self.is_tb and frame.shape[0] > 0:
                # Top-bottom: crop to top half
                half_height = frame.shape[0] // 2
                frame = frame[:half_height, :]
                self.logger.debug(f"Cropped TB to top panel: {frame.shape}")

            # For 2D videos, pad to square. For VR, resize (already cropped to single eye)
            if self.is_2d:
                # Pad 2D video to square (letterbox/pillarbox)
                h, w = frame.shape[:2]

                if h == w:
                    # Already square, just resize
                    frame_resized = cv2.resize(frame, (self.output_size, self.output_size),
                                              interpolation=cv2.INTER_AREA)
                else:
                    # Calculate scaling to fit within output_size while maintaining aspect ratio
                    scale = self.output_size / max(h, w)
                    new_h = int(h * scale)
                    new_w = int(w * scale)

                    # Resize maintaining aspect ratio
                    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                    # Create black canvas
                    frame_resized = np.zeros((self.output_size, self.output_size, 3), dtype=np.uint8)

                    # Center the resized frame on the canvas
                    y_offset = (self.output_size - new_h) // 2
                    x_offset = (self.output_size - new_w) // 2
                    frame_resized[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

                    self.logger.debug(f"Padded 2D frame from {w}x{h} to {self.output_size}x{self.output_size}")
            else:
                # VR content: just resize (already cropped to single eye)
                frame_resized = cv2.resize(frame, (self.output_size, self.output_size),
                                          interpolation=cv2.INTER_AREA)

            # Apply GPU unwarp for VR content if available
            if use_gpu_unwarp and self.gpu_unwarp_worker is not None:
                # Convert BGR to RGBA for GPU unwarp
                rgba_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGBA)

                # Submit to GPU unwarp worker
                timestamp_ms = (frame_index / self.fps * 1000.0) if self.fps > 0 else 0.0
                self.gpu_unwarp_worker.submit_frame(
                    frame_index, rgba_frame, timestamp_ms=timestamp_ms, timeout=0.1
                )

                # Get unwrapped result
                unwarp_result = self.gpu_unwarp_worker.get_unwrapped_frame(timeout=0.5)

                if unwarp_result is not None:
                    _, frame_resized, _ = unwarp_result
                    self.logger.debug(f"GPU unwarp applied to thumbnail frame {frame_index}")
                else:
                    self.logger.debug(f"GPU unwarp timeout for thumbnail frame {frame_index}, using original")
                    # Keep original frame if unwarp fails

            return frame_resized

        except Exception as e:
            self.logger.error(f"ThumbnailExtractor: Error extracting frame {frame_index}: {e}")
            return None

    def get_frame_at_time(self, time_seconds: float, use_gpu_unwarp: bool = True) -> Optional[np.ndarray]:
        """
        Extract frame at specified timestamp.

        Args:
            time_seconds: Time in seconds
            use_gpu_unwarp: Whether to apply GPU unwarp for VR content

        Returns:
            Frame as numpy array or None
        """
        if self.fps <= 0:
            self.logger.warning("ThumbnailExtractor: Invalid FPS")
            return None

        frame_index = int(time_seconds * self.fps)
        return self.get_frame(frame_index, use_gpu_unwarp=use_gpu_unwarp)

    def close(self):
        """Release VideoCapture resources."""
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
                self.is_open = False
                self.logger.info("ThumbnailExtractor closed")

    def __del__(self):
        """Ensure VideoCapture is released on deletion."""
        self.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
