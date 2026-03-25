"""
FFmpeg-based thumbnail extractor for frame-accurate random access.

Uses FFmpeg with input-level -ss seeking for reliable, frame-accurate
single-frame extraction across all codecs (H.264, HEVC, AV1, etc.).

For VR content, applies the same v360 CPU dewarp filter used by the main
processing pipeline (benchmark shows it adds negligible cost vs decode).
"""

import subprocess
import sys
import numpy as np
import logging
from typing import Optional, Tuple
from threading import Lock


class ThumbnailExtractor:
    """
    Frame-accurate thumbnail extractor using FFmpeg subprocess.

    Video properties (fps, dimensions, etc.) are passed from the video
    processor at init time — no redundant ffprobe calls.
    """

    def __init__(self, video_path: str, fps: float, total_frames: int,
                 output_size: int = 320, vr_input_format: str = None,
                 vr_fov: int = 190, vr_pitch: float = 0.0,
                 display_dimensions: Optional[Tuple[int, int]] = None,
                 logger: Optional[logging.Logger] = None):
        self.video_path = video_path
        self.logger = logger or logging.getLogger(__name__)
        self.output_size = output_size
        self.fps = fps
        self.total_frames = total_frames
        self.vr_input_format = vr_input_format
        self.vr_fov = vr_fov
        self.vr_pitch = vr_pitch

        # HD display dimensions (width, height) — when set, 2D output is non-square
        self._display_w, self._display_h = display_dimensions if display_dimensions else (output_size, output_size)

        self.lock = Lock()
        self.is_open = fps > 0 and total_frames > 0

        # VR format detection
        vr_fmt = (vr_input_format or '').lower()
        self.is_sbs_left = '_sbs' in vr_fmt or '_lr' in vr_fmt
        self.is_sbs_right = '_rl' in vr_fmt
        self.is_tb = '_tb' in vr_fmt
        self.is_vr = vr_input_format is not None

        # Pre-build the filter string once
        self._vf = self._build_vf_filters()

        if self.is_open:
            self.logger.debug(
                f"ThumbnailExtractor: {fps:.2f} FPS, "
                f"{total_frames} frames, output={self._display_w}x{self._display_h}"
            )

    def _build_vf_filters(self) -> str:
        """Build FFmpeg video filter string."""
        filters = []

        if self.is_vr:
            # VR: crop to single eye panel
            if self.is_sbs_left:
                filters.append('crop=iw/2:ih:0:0')
            elif self.is_sbs_right:
                filters.append('crop=iw/2:ih:iw/2:0')
            elif self.is_tb:
                filters.append('crop=iw:ih/2:0:0')

            # v360 dewarp (CPU) — negligible cost vs decode on 8K HEVC
            base_fmt = (self.vr_input_format or '').replace(
                '_sbs', '').replace('_tb', '').replace('_lr', '').replace('_rl', '')
            filters.append(
                f'v360={base_fmt}:in_stereo=0:output=sg:'
                f'iv_fov={self.vr_fov}:ih_fov={self.vr_fov}:'
                f'd_fov={self.vr_fov}:'
                f'v_fov=90:h_fov=90:'
                f'pitch={self.vr_pitch}:yaw=0:roll=0:'
                f'w={self.output_size}:h={self.output_size}:interp=linear'
            )
        else:
            # 2D: scale to display dimensions (no padding when HD, padded square when standard)
            if self._display_w != self.output_size or self._display_h != self.output_size:
                # HD mode: scale to exact display dimensions, no padding
                filters.append(f'scale={self._display_w}:{self._display_h}')
            else:
                # Standard mode: scale preserving aspect ratio, pad to square
                filters.append(
                    f'scale={self.output_size}:{self.output_size}'
                    f':force_original_aspect_ratio=decrease,'
                    f'pad={self.output_size}:{self.output_size}'
                    f':(ow-iw)/2:(oh-ih)/2:black'
                )

        return ','.join(filters)

    def get_frame(self, frame_index: int, **_kwargs) -> Optional[np.ndarray]:
        """
        Extract a single frame at the specified index using FFmpeg.

        Returns:
            Frame as BGR24 numpy array (display_h x display_w x 3) or None.
        """
        if not self.is_open:
            return None

        if frame_index < 0 or frame_index >= self.total_frames:
            return None

        try:
            with self.lock:
                seek_time = frame_index / self.fps

                cmd = [
                    'ffmpeg', '-hide_banner', '-nostats', '-loglevel', 'error',
                    '-ss', f'{seek_time:.6f}',
                    '-i', self.video_path,
                    '-vf', self._vf,
                    '-frames:v', '1',
                    '-pix_fmt', 'bgr24',
                    '-f', 'rawvideo',
                    'pipe:1'
                ]

                creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                proc = subprocess.run(
                    cmd, capture_output=True, timeout=10,
                    creationflags=creation_flags
                )

                frame_size = self._display_w * self._display_h * 3
                if len(proc.stdout) < frame_size:
                    self.logger.warning(
                        f"ThumbnailExtractor: Incomplete data for frame {frame_index} "
                        f"(got {len(proc.stdout)}/{frame_size})"
                    )
                    return None

                return np.frombuffer(
                    proc.stdout[:frame_size], dtype=np.uint8
                ).reshape(self._display_h, self._display_w, 3)

        except subprocess.TimeoutExpired:
            self.logger.warning(f"ThumbnailExtractor: FFmpeg timeout for frame {frame_index}")
            return None
        except Exception as e:
            self.logger.error(f"ThumbnailExtractor: Error extracting frame {frame_index}: {e}")
            return None

    def close(self):
        """No persistent connection to close."""
        self.is_open = False

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
