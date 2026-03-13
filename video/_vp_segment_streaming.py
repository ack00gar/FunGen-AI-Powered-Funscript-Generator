"""VideoProcessor SegmentStreamingMixin — extracted from video_processor.py."""

import shlex
import subprocess
import sys
import threading
import time
import numpy as np
from typing import Optional, Iterator, Tuple


class SegmentStreamingMixin:
    """Mixin fragment for VideoProcessor."""

    def _start_ffmpeg_for_segment_streaming(self, start_frame_abs_idx: int, num_frames_to_stream_hint: Optional[int] = None) -> bool:
        self._terminate_ffmpeg_processes()

        if not self.video_path or not self.video_info or self.video_info.get('fps', 0) <= 0:
            self.logger.warning("Cannot start FFmpeg for segment: no video/invalid FPS.")
            return False

        start_time_seconds = start_frame_abs_idx / self.video_info['fps']
        
        # Optimize ffmpeg for MAX_SPEED processing (segment streaming)
        common_ffmpeg_prefix = ['ffmpeg', '-hide_banner', '-nostats', '-loglevel', 'error']
        
        # Add MAX_SPEED optimizations if in MAX_SPEED mode
        if (hasattr(self.app, 'app_state_ui') and 
            hasattr(self.app.app_state_ui, 'selected_processing_speed_mode') and
            self.app.app_state_ui.selected_processing_speed_mode == constants.ProcessingSpeedMode.MAX_SPEED):
            # Same aggressive optimizations for segment streaming
            # Hardware acceleration: Handled by individual pipe paths (don't add to common prefix)
            
            # Add speed optimizations (hardware acceleration handled by pipe-specific code)
            # NOTE: -preset and -tune are encoding options, not decoding options
            common_ffmpeg_prefix.extend([
                '-fflags', '+genpts+fastseek', 
                '-threads', '0',
                '-probesize', '32',
                '-analyzeduration', '1'
            ])
            self.logger.info("FFmpeg segment streaming optimized for MAX_SPEED with fast decode")

        if self._is_10bit_cuda_pipe_needed():
            self.logger.info("Using 2-pipe FFmpeg command for 10-bit CUDA segment streaming.")
            video_height_for_crop = self.video_info.get('height', 0)
            if video_height_for_crop <= 0:
                self.logger.error("Cannot construct 10-bit CUDA pipe 1 for segment: video height is unknown.")
                return False

            pipe1_vf = f"crop={int(video_height_for_crop)}:{int(video_height_for_crop)}:0:0,scale_cuda=1000:1000"
            cmd1 = common_ffmpeg_prefix[:]
            cmd1.extend(['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda'])
            if start_time_seconds > 0.001: cmd1.extend(['-ss', str(start_time_seconds)])
            cmd1.extend(['-i', self._active_video_source_path, '-an', '-sn', '-vf', pipe1_vf])
            if num_frames_to_stream_hint and num_frames_to_stream_hint > 0:
                cmd1.extend(['-frames:v', str(num_frames_to_stream_hint)])
            cmd1.extend(['-c:v', 'hevc_nvenc', '-preset', 'fast', '-qp', '0', '-f', 'matroska', 'pipe:1'])

            cmd2 = common_ffmpeg_prefix[:]
            cmd2.extend(['-hwaccel', 'cuda', '-i', 'pipe:0', '-an', '-sn'])
            effective_vf_pipe2 = self.ffmpeg_filter_string or f"scale={self.yolo_input_size}:{self.yolo_input_size}:force_original_aspect_ratio=decrease,pad={self.yolo_input_size}:{self.yolo_input_size}:(ow-iw)/2:(oh-ih)/2:black"
            cmd2.extend(['-vf', effective_vf_pipe2])
            if num_frames_to_stream_hint and num_frames_to_stream_hint > 0:
                cmd2.extend(['-frames:v', str(num_frames_to_stream_hint)])
            # Always use BGR24 - GPU unwarp worker handles BGR->RGBA conversion internally
            cmd2.extend(['-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1'])

            self.logger.info(f"Segment Pipe 1 CMD: {' '.join(shlex.quote(str(x)) for x in cmd1)}")
            self.logger.info(f"Segment Pipe 2 CMD: {' '.join(shlex.quote(str(x)) for x in cmd2)}")
            try:
                creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                self.ffmpeg_pipe1_process = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1, creationflags=creation_flags)
                if self.ffmpeg_pipe1_process.stdout is None:
                    raise IOError("Segment Pipe 1 stdout is None.")
                self.ffmpeg_process = subprocess.Popen(cmd2, stdin=self.ffmpeg_pipe1_process.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1, creationflags=creation_flags)
                self.ffmpeg_pipe1_process.stdout.close()
                return True
            except Exception as e:
                self.logger.error(f"Failed to start 2-pipe FFmpeg for segment: {e}", exc_info=True)
                self._terminate_ffmpeg_processes()
                return False
        else:
            # Standard single FFmpeg process for 8-bit or non-CUDA accelerated video
            hwaccel_cmd_list = self._get_ffmpeg_hwaccel_args()
            ffmpeg_input_options = hwaccel_cmd_list[:]
            if start_time_seconds > 0.001: ffmpeg_input_options.extend(['-ss', str(start_time_seconds)])
            ffmpeg_cmd = common_ffmpeg_prefix + ffmpeg_input_options + ['-i', self._active_video_source_path, '-an', '-sn']
            effective_vf = self.ffmpeg_filter_string or f"scale={self.yolo_input_size}:{self.yolo_input_size}:force_original_aspect_ratio=decrease,pad={self.yolo_input_size}:{self.yolo_input_size}:(ow-iw)/2:(oh-ih)/2:black"
            ffmpeg_cmd.extend(['-vf', effective_vf])

            if num_frames_to_stream_hint and num_frames_to_stream_hint > 0:
                ffmpeg_cmd.extend(['-frames:v', str(num_frames_to_stream_hint)])

            # When using select filter, -vsync vfr is needed to actually drop non-selected frames
            if getattr(self, '_ffmpeg_vsync_mode', None):
                ffmpeg_cmd.extend(['-vsync', self._ffmpeg_vsync_mode])

            # Always BGR24 — GPU unwarp worker handles BGR→RGBA conversion internally
            ffmpeg_cmd.extend(['-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1'])
            self.logger.info(f"Segment CMD (single pipe): {' '.join(shlex.quote(str(x)) for x in ffmpeg_cmd)}")
            try:
                creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                self.ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1, creationflags=creation_flags)
                return True
            except Exception as e:
                self.logger.warning(f"Failed to start FFmpeg for segment: {e}", exc_info=True)
                self.ffmpeg_process = None
                return False

    def stream_frames_for_segment(self, start_frame_abs_idx: int, num_frames_to_read: int, stop_event: Optional[threading.Event] = None) -> Iterator[Tuple[int, np.ndarray, dict]]:
        if num_frames_to_read <= 0:
            self.logger.warning("num_frames_to_read is not positive, no frames to stream.")
            return

        if not self._start_ffmpeg_for_segment_streaming(start_frame_abs_idx, num_frames_to_read):
            self.logger.warning(f"Failed to start FFmpeg for segment from {start_frame_abs_idx}.")
            return

        frames_yielded = 0
        segment_ffmpeg_process = self.ffmpeg_process
        try:
            for i in range(num_frames_to_read):
                if stop_event and stop_event.is_set():
                    self.logger.info("Stop event detected in stream_frames_for_segment. Aborting stream.")
                    break

                if not segment_ffmpeg_process or segment_ffmpeg_process.stdout is None:
                    self.logger.warning("FFmpeg process or stdout not available during segment streaming.")
                    break

                if segment_ffmpeg_process.poll() is not None:
                    stderr_output = segment_ffmpeg_process.stderr.read(4096).decode(errors='ignore') if segment_ffmpeg_process.stderr else ""
                    self.logger.warning(
                        f"FFmpeg process (segment) terminated prematurely. Exit: {segment_ffmpeg_process.returncode}. Stderr: '{stderr_output.strip()}'")
                    break

                t_decode_start = time.time()
                raw_frame_bytes = segment_ffmpeg_process.stdout.read(self.frame_size_bytes)
                decode_ms = (time.time() - t_decode_start) * 1000.0
                if len(raw_frame_bytes) < self.frame_size_bytes:
                    stderr_on_short_read = segment_ffmpeg_process.stderr.read(4096).decode(errors='ignore') if segment_ffmpeg_process.stderr else ""
                    self.logger.info(
                        f"End of FFmpeg stream or error (read {len(raw_frame_bytes)}/{self.frame_size_bytes}) "
                        f"after {frames_yielded} frames for segment (start {start_frame_abs_idx}). Stderr: '{stderr_on_short_read.strip()}'")
                    break

                # Always use BGR24 format (3 bytes per pixel)
                expected_size = self.yolo_input_size * self.yolo_input_size * 3
                actual_bytes = len(raw_frame_bytes)

                # Validate frame size
                if actual_bytes != expected_size:
                    self.logger.error(f"Invalid frame size: {actual_bytes} bytes (expected {expected_size}). Skipping frame.")
                    continue

                frame_np = np.frombuffer(raw_frame_bytes, dtype=np.uint8).reshape(self.yolo_input_size, self.yolo_input_size, 3)

                # Apply GPU unwarp for VR frames if enabled
                unwarp_ms = 0.0
                if self.gpu_unwarp_enabled and self.gpu_unwarp_worker:
                    t_unwarp_start = time.time()
                    current_frame_id = start_frame_abs_idx + frames_yielded
                    self.gpu_unwarp_worker.submit_frame(current_frame_id, frame_np,
                                                       timestamp_ms=current_frame_id * (1000.0 / self.fps) if self.fps > 0 else 0.0,
                                                       timeout=0.1)
                    unwarp_result = self.gpu_unwarp_worker.get_unwrapped_frame(timeout=0.5)
                    if unwarp_result is not None:
                        _, frame_np, _ = unwarp_result
                    else:
                        self.logger.warning(f"GPU unwarp timeout for segment frame {current_frame_id}")
                    unwarp_ms = (time.time() - t_unwarp_start) * 1000.0

                current_frame_id = start_frame_abs_idx + frames_yielded
                yield current_frame_id, frame_np, {'decode_ms': decode_ms, 'unwarp_ms': unwarp_ms}
                frames_yielded += 1
        finally:
            self._terminate_ffmpeg_processes()
