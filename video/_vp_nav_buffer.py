"""VideoProcessor NavBufferMixin — extracted from video_processor.py."""

import subprocess
import sys
import numpy as np
from typing import Optional


class NavBufferMixin:
    """Mixin fragment for VideoProcessor."""

    def _compute_nav_buffer_size(self) -> int:
        """Compute arrow-nav buffer size based on available RAM.

        Each frame is yolo_input_size^2 * 3 bytes.  We cap the buffer at
        ~10 % of available RAM, with a floor of 120 frames.
        """
        frame_bytes = self.yolo_input_size * self.yolo_input_size * 3
        try:
            import psutil
            avail = psutil.virtual_memory().available
            budget = int(avail * 0.10)  # 10 % of free RAM
            max_frames = max(120, budget // frame_bytes)
            # Cap at a reasonable upper bound (1800 = ~1 min at 30fps)
            max_frames = min(max_frames, 1800)
        except (ImportError, OSError):
            max_frames = 300  # Safe default (~360 MB at 640px)
        self.logger.debug(f"Arrow-nav buffer: {max_frames} frames "
                         f"({max_frames * frame_bytes / (1024*1024):.0f} MB)")
        return max_frames

    def _start_unified_pipe(self, start_frame: int) -> bool:
        """Spawn a unified FFmpeg pipe at *start_frame* for arrow nav / pipe reuse.

        Uses the same filter chain as the processing loop so VR projection,
        scaling, etc. are visually identical to playback.
        """
        self._stop_unified_pipe()

        if not self.video_path or not self.video_info or self.fps <= 0:
            return False

        start_sec = start_frame / self.fps
        common_prefix = ['ffmpeg', '-hide_banner', '-nostats', '-loglevel', 'error']

        effective_vf = self.ffmpeg_filter_string or (
            f"scale={self.yolo_input_size}:{self.yolo_input_size}"
            f":force_original_aspect_ratio=decrease,"
            f"pad={self.yolo_input_size}:{self.yolo_input_size}:(ow-iw)/2:(oh-ih)/2:black"
        )

        hwaccel_args = self._get_ffmpeg_hwaccel_args()
        cmd = common_prefix + hwaccel_args[:]
        if start_sec > 0.001:
            cmd.extend(['-ss', str(start_sec)])
        cmd.extend(['-i', self._active_video_source_path, '-an', '-sn'])
        cmd.extend(['-vf', effective_vf])
        cmd.extend(['-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1'])

        try:
            creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            self._ffmpeg_pipe = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                bufsize=-1, creationflags=creation_flags,
            )
            self._ffmpeg_read_position = start_frame
            self.logger.debug(f"Unified FFmpeg pipe started at frame {start_frame}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start unified FFmpeg pipe: {e}")
            self._ffmpeg_pipe = None
            return False

    def _stop_unified_pipe(self):
        """Terminate the unified FFmpeg pipe if running."""
        if self._ffmpeg_pipe is not None:
            self._terminate_process(self._ffmpeg_pipe, "UnifiedPipe")
            self._ffmpeg_pipe = None

    def _pipe_read_one(self) -> Optional[np.ndarray]:
        """Read exactly one frame from the unified FFmpeg pipe."""
        if self._ffmpeg_pipe is None or self._ffmpeg_pipe.stdout is None:
            return None
        try:
            raw = self._ffmpeg_pipe.stdout.read(self.frame_size_bytes)
            if len(raw) < self.frame_size_bytes:
                return None
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                self._display_frame_h, self._display_frame_w, 3
            ).copy()
            self._ffmpeg_read_position += 1
            return frame
        except (OSError, ValueError):
            return None

    def _buffer_lookup(self, target_frame: int) -> Optional[np.ndarray]:
        """O(1) lookup in the frame buffer by computing offset from first entry."""
        with self._frame_buffer_lock:
            if not self._frame_buffer:
                return None
            first_idx = self._frame_buffer[0][0]
            offset = target_frame - first_idx
            if 0 <= offset < len(self._frame_buffer):
                stored_idx, frame_data = self._frame_buffer[offset]
                if stored_idx == target_frame:
                    return frame_data
            return None

    def _buffer_append(self, frame_index: int, frame_data: np.ndarray):
        """Append a frame to the buffer. Clear if non-contiguous."""
        with self._frame_buffer_lock:
            if self._frame_buffer:
                last_idx = self._frame_buffer[-1][0]
                if frame_index != last_idx + 1:
                    self._frame_buffer.clear()
            self._frame_buffer.append((frame_index, frame_data))

    def _clear_nav_state(self):
        """Clear all navigation state: buffer + pipe.

        Called on position jumps (seek, cache clear, FFmpeg teardown) so stale
        frames from a previous position are never served.
        """
        self._stop_unified_pipe()
        with self._frame_buffer_lock:
            self._frame_buffer.clear()

    @property
    def buffer_info(self) -> dict:
        """Return buffer stats for status strip display."""
        with self._frame_buffer_lock:
            size = len(self._frame_buffer)
            capacity = self._frame_buffer.maxlen or 0
            start = self._frame_buffer[0][0] if self._frame_buffer else -1
            end = self._frame_buffer[-1][0] if self._frame_buffer else -1
        return {'size': size, 'capacity': capacity, 'start': start, 'end': end,
                'current': self.current_frame_index}

    def _clear_cache(self):
        with self.frame_cache_lock:
            if self.frame_cache is not None:
                try:
                    if self.frame_cache is not None:
                        cache_len = len(self.frame_cache)
                    else:
                        cache_len = 0
                except (TypeError, AttributeError):
                    cache_len = 0
                if cache_len > 0:
                    self.logger.debug(f"Clearing frame cache (had {cache_len} items).")
                    self.frame_cache.clear()
        self._clear_nav_state()

    def arrow_nav_forward(self, target_frame: int) -> Optional[np.ndarray]:
        """Navigate forward using unified buffer + pipe.

        3-tier lookup (no cv2 fallback):
        1. Buffer hit — O(1), instant
        2. Pipe at correct position — read + append
        3. Restart pipe at target — read + append
        """
        if self.arrow_nav_in_progress:
            return None

        self.arrow_nav_in_progress = True
        try:
            # 1. Buffer lookup (O(1))
            frame = self._buffer_lookup(target_frame)
            if frame is not None:
                self.current_frame_index = target_frame
                return frame

            # 2. Pipe at correct position — sequential read
            if (self._ffmpeg_pipe is not None
                    and self._ffmpeg_pipe.poll() is None
                    and self._ffmpeg_read_position == target_frame):
                frame = self._pipe_read_one()
                if frame is not None:
                    self._buffer_append(target_frame, frame)
                    self.current_frame_index = target_frame
                    return frame
                else:
                    self.logger.warning(f"Forward nav pipe read failed at frame {target_frame}")

            # 3. Restart pipe at target
            if self._start_unified_pipe(target_frame):
                frame = self._pipe_read_one()
                if frame is not None:
                    self._buffer_append(target_frame, frame)
                    self.current_frame_index = target_frame
                    return frame
                else:
                    self.logger.warning(f"Forward nav pipe restart read failed at frame {target_frame}")
            else:
                self.logger.warning(f"Forward nav pipe restart failed at frame {target_frame}")

            return None
        finally:
            self.arrow_nav_in_progress = False

    def arrow_nav_backward(self, target_frame: int) -> Optional[np.ndarray]:
        """Navigate backward using unified buffer + pipe.

        2-tier lookup (no cv2 fallback):
        1. Buffer hit — O(1), instant
        2. Buffer miss — rebuild buffer centered on target (50% before, 50% after)
        """
        # 1. Buffer lookup (O(1))
        frame = self._buffer_lookup(target_frame)
        if frame is not None:
            self.current_frame_index = target_frame
            return frame

        # 2. Buffer miss: rebuild centered on target_frame
        #    Start pipe at target - capacity/2 so cursor lands in the middle.
        half_buf = (self._frame_buffer.maxlen or 300) // 2
        pipe_start = max(0, target_frame - half_buf)
        frames_to_skip = target_frame - pipe_start  # frames to read before target

        with self._frame_buffer_lock:
            self._frame_buffer.clear()

        if self._start_unified_pipe(pipe_start):
            # Pre-fill buffer with frames before target
            for i in range(frames_to_skip):
                pre_frame = self._pipe_read_one()
                if pre_frame is not None:
                    self._buffer_append(pipe_start + i, pre_frame)
                else:
                    break

            # Read the target frame itself
            frame = self._pipe_read_one()
            if frame is not None:
                self._buffer_append(target_frame, frame)
                self.current_frame_index = target_frame
                return frame
            else:
                self.logger.warning(f"Backward nav pipe read failed at frame {target_frame}")
        else:
            self.logger.warning(f"Backward nav pipe restart failed at frame {target_frame}")

        return None
