"""FFmpeg-subprocess thumbnail extractor for frame-accurate random access."""

from __future__ import annotations

import logging
import subprocess
from threading import Lock
from typing import Optional, Tuple

import numpy as np

from video.ffmpeg_helpers import find_ffmpeg, subprocess_flags


class ThumbnailExtractor:
    """Frame-accurate thumbnail extractor using FFmpeg subprocess."""

    def __init__(self, video_path: str, fps: float, total_frames: int,
                 output_size: int = 320, vr_input_format: Optional[str] = None,
                 vr_fov: int = 190, vr_pitch: float = 0.0,
                 display_dimensions: Optional[Tuple[int, int]] = None,
                 eye: str = 'left',
                 logger: Optional[logging.Logger] = None):
        self.video_path = video_path
        self.logger = logger or logging.getLogger(__name__)
        self.output_size = output_size
        self.fps = fps
        self.total_frames = total_frames
        self.vr_input_format = vr_input_format
        # Guard vr_fov=0 the same way the filter builder does. Zero iv/ih/d_fov
        # make libavfilter reject v360 with "Invalid argument".
        self.vr_fov = vr_fov if vr_fov and vr_fov > 0 else 190
        self.vr_pitch = vr_pitch
        self.eye = eye

        self._display_w, self._display_h = (
            display_dimensions if display_dimensions else (output_size, output_size)
        )

        self.lock = Lock()
        self.is_open = fps > 0 and total_frames > 0

        self.is_vr = vr_input_format is not None

        self._vf = self._build_vf_filters()
        self._ffmpeg = find_ffmpeg()
        self._proc_flags = subprocess_flags()

        if self.is_open:
            self.logger.debug(
                f"ThumbnailExtractor: {fps:.2f} FPS, {total_frames} frames, "
                f"output={self._display_w}x{self._display_h}"
            )

    def _build_vf_filters(self) -> str:
        filters = []
        if self.is_vr:
            from video import vr_panel
            eye = self.eye if self.eye != vr_panel.EYE_FULL else vr_panel.EYE_LEFT
            region = vr_panel.resolve_eye(self.vr_input_format, eye)
            if not region.is_full():
                def _frac(val: float, axis: str) -> str:
                    if val == 0.0:
                        return "0"
                    if val == 0.5:
                        return f"{axis}/2"
                    return axis
                filters.append(
                    f"crop={_frac(region.w, 'iw')}:{_frac(region.h, 'ih')}"
                    f":{_frac(region.x, 'iw')}:{_frac(region.y, 'ih')}")

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
            if self._display_w != self.output_size or self._display_h != self.output_size:
                filters.append(f'scale={self._display_w}:{self._display_h}')
            else:
                filters.append(
                    f'scale={self.output_size}:{self.output_size}'
                    f':force_original_aspect_ratio=decrease,'
                    f'pad={self.output_size}:{self.output_size}'
                    f':(ow-iw)/2:(oh-ih)/2:black'
                )

        return ','.join(filters)

    def get_frame(self, frame_index: int, **_kwargs) -> Optional[np.ndarray]:
        """Extract a single frame at ``frame_index``. Returns BGR24 ndarray or None."""
        if not self.is_open:
            return None
        if frame_index < 0 or frame_index >= self.total_frames:
            return None
        try:
            with self.lock:
                seek_time = frame_index / self.fps
                cmd = [
                    self._ffmpeg,
                    '-hide_banner', '-nostats', '-loglevel', 'error',
                    '-ss', f'{seek_time:.6f}',
                    '-i', self.video_path,
                    '-vf', self._vf,
                    '-frames:v', '1',
                    '-pix_fmt', 'bgr24',
                    '-f', 'rawvideo',
                    'pipe:1',
                ]
                proc = subprocess.run(
                    cmd, capture_output=True, timeout=10,
                    creationflags=self._proc_flags,
                )
                frame_size = self._display_w * self._display_h * 3
                if len(proc.stdout) < frame_size:
                    self.logger.warning(
                        f"ThumbnailExtractor: incomplete data for frame {frame_index} "
                        f"(got {len(proc.stdout)}/{frame_size})"
                    )
                    return None
                return np.frombuffer(
                    proc.stdout[:frame_size], dtype=np.uint8
                ).reshape(self._display_h, self._display_w, 3)
        except subprocess.TimeoutExpired:
            # Cold-start + v360 on 8K VR can exceed the 10s budget. Returning
            # None is fine; the caller debounces and retries. Log at debug
            # so steady-state hovers don't spam warnings.
            self.logger.debug(f"ThumbnailExtractor: ffmpeg timeout for frame {frame_index}")
            return None
        except Exception as e:
            self.logger.error(f"ThumbnailExtractor: error extracting frame {frame_index}: {e}")
            return None

    def close(self):
        self.is_open = False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
