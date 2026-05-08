"""VideoProcessor FFmpegBuildersMixin -- extracted from video_processor.py."""

import os
import platform
import subprocess
import numpy as np
from typing import Optional, List, Dict, Tuple

# (source_path, hwaccel_name, mtime) -> works_bool; one probe per source per session.
_HWACCEL_PROBE_CACHE: Dict[Tuple[str, str, float], bool] = {}


class FFmpegBuildersMixin:
    """Mixin fragment for VideoProcessor."""

    def _is_using_preprocessed_video(self) -> bool:
        """Checks if the active video source is a preprocessed file."""
        is_using_preprocessed_by_path_diff = self._active_video_source_path != self.video_path
        is_preprocessed_by_name = self._active_video_source_path.endswith("_preprocessed.mp4")
        return is_using_preprocessed_by_path_diff or is_preprocessed_by_name

    def _needs_hw_download(self) -> bool:
        """Determines if the FFmpeg filter chain requires a 'hwdownload' filter."""
        current_hw_args = self._get_ffmpeg_hwaccel_args()
        if '-hwaccel_output_format' in current_hw_args:
            try:
                idx = current_hw_args.index('-hwaccel_output_format')
                hw_output_format = current_hw_args[idx + 1]
                # These formats are on the GPU and need to be downloaded for CPU-based filters.
                if hw_output_format in ['cuda', 'nv12', 'p010le', 'qsv', 'vaapi', 'd3d11va', 'dxva2_vld']:
                    return True
            except (ValueError, IndexError):
                self.logger.warning("Could not properly parse -hwaccel_output_format from hw_args.")
        return False

    def _get_2d_video_filters(self) -> List[str]:
        """Builds the list of FFmpeg filter segments for standard 2D video.
        ffmpeg subprocess always outputs at yolo_input_size."""
        if not self.video_info:
            # Fallback if no video info
            return [
                f"scale={self.yolo_input_size}:{self.yolo_input_size}:force_original_aspect_ratio=decrease",
                f"pad={self.yolo_input_size}:{self.yolo_input_size}:(ow-iw)/2:(oh-ih)/2:black"
            ]

        width = self.video_info.get('width', 0)
        height = self.video_info.get('height', 0)

        if width == 0 or height == 0:
            # Fallback if dimensions unknown
            return [
                f"scale={self.yolo_input_size}:{self.yolo_input_size}:force_original_aspect_ratio=decrease",
                f"pad={self.yolo_input_size}:{self.yolo_input_size}:(ow-iw)/2:(oh-ih)/2:black"
            ]

        aspect_ratio = width / height

        # Check if video is square (or nearly square)
        if 0.95 < aspect_ratio < 1.05:
            # Square video - just scale, no padding needed
            return [f"scale={self.yolo_input_size}:{self.yolo_input_size}"]
        elif aspect_ratio > 1.05:
            # Wider than tall (landscape) - scale and pad top/bottom
            # Use -2 (not -1) to ensure even dimensions, avoiding filter errors on some FFmpeg builds
            return [
                f"scale={self.yolo_input_size}:-2:force_original_aspect_ratio=decrease",
                f"pad={self.yolo_input_size}:{self.yolo_input_size}:0:(oh-ih)/2:black"
            ]
        else:
            # Taller than wide (portrait) - scale and pad left/right
            return [
                f"scale=-2:{self.yolo_input_size}:force_original_aspect_ratio=decrease",
                f"pad={self.yolo_input_size}:{self.yolo_input_size}:(ow-iw)/2:0:black"
            ]

    def _init_thumbnail_extractor(self):
        """Hand the active video path to ThumbnailMpv (a second mpv instance).
        External pattern: thumbnails come from a paused mpv player scrubbed on
        hover. No ffmpeg subprocess per thumbnail.
        """
        if not self._active_video_source_path or not self.video_info:
            return
        gui = getattr(self.app, 'gui_instance', None)
        tmb = getattr(gui, 'thumbnail_mpv', None) if gui else None
        if tmb is not None:
            try:
                tmb.load(self._active_video_source_path)
            except Exception as e:
                self.logger.debug(f"ThumbnailMpv load failed: {e}")

    def get_thumbnail_frame(self, frame_index: int, **_kwargs) -> Optional[np.ndarray]:
        """Returns a BGR24 numpy frame for the given index (or None).

        Request size matches source aspect so eye-crop on the caller side
        (gui_preview_manager) produces correct geometry. mpv letterboxes
        when the request shape does not match source aspect, which would
        turn an SBS left-eye crop into a vertical strip.
        """
        gui = getattr(self.app, 'gui_instance', None)
        tq = getattr(gui, 'thumbnail_mpv_queue', None) if gui else None
        if tq is not None and self.fps > 0:
            try:
                time_sec = float(frame_index) / float(self.fps)
                long_side = int(self.yolo_input_size) if self.yolo_input_size else 320
                src_w = int((self.video_info or {}).get('width', 0) or 0)
                src_h = int((self.video_info or {}).get('height', 0) or 0)
                if src_w > 0 and src_h > 0:
                    if src_w >= src_h:
                        w = long_side
                        h = max(1, int(round(long_side * src_h / src_w)))
                    else:
                        h = long_side
                        w = max(1, int(round(long_side * src_w / src_h)))
                else:
                    w = h = long_side
                frame = tq.request(time_sec, w=w, h=h, timeout=3.0)
                if frame is not None:
                    return frame
            except Exception as e:
                self.logger.debug(f"ThumbnailMpv request failed: {e}")
        return self._get_specific_frame(
            frame_index, update_current_index=False, use_thumbnail=True)

    def _get_vr_video_filters(self) -> List[str]:
        """Builds the list of FFmpeg filter segments for VR video, including cropping and v360."""
        if not self.video_info:
            return []

        original_width = self.video_info.get('width', 0)
        original_height = self.video_info.get('height', 0)
        # Tracker output is square at yolo_input_size; v_fov=h_fov=90 for
        # the stereographic projection.
        out_w = self.yolo_input_size
        out_h = self.yolo_input_size
        v_fov = 90
        h_fov = 90

        vr_filters = []
        from video import vr_panel
        # Unified panel selection. Centralized helper applies the _rl
        # reversal so the same 'left' selector always yields the LEFT
        # EYE regardless of source layout flavor.
        eye = vr_panel.read_setting(getattr(self.app, 'app_settings', None),
                                    default=vr_panel.EYE_LEFT)
        # v360 (the default) expects a single-eye input; 'full' would
        # feed it a squashed SBS frame and produce garbage. Clamp to
        # 'left' unless the user explicitly chose crop-only mode AND
        # wants both halves preserved.
        if eye == vr_panel.EYE_FULL and self.vr_unwarp_method_override != 'none':
            eye = vr_panel.EYE_LEFT
        region = vr_panel.resolve_eye(self.vr_input_format, eye)
        crop = region.ffmpeg_crop(original_width, original_height)
        if crop:
            vr_filters.append(crop)
            self.logger.debug(
                f"Applying VR pre-crop: fmt={self.vr_input_format} "
                f"eye={eye} -> {crop}")

        # Unwarp method: 'none' (crop only) or default libavfilter v360.
        if self.vr_unwarp_method_override == 'none':
            vr_filters.append(f"scale={out_w}:{out_h}")
            self.logger.debug("Unwarp: None (crop only) - no dewarping applied, just crop+scale")
        else:
            base_v360_input_format = self.vr_input_format.replace('_sbs', '').replace('_tb', '').replace('_lr', '').replace('_rl', '')
            v360_filter_core = (
                f"v360={base_v360_input_format}:in_stereo=0:output=sg:"
                f"iv_fov={self.vr_fov}:ih_fov={self.vr_fov}:"
                f"d_fov={self.vr_fov}:"
                f"v_fov={v_fov}:h_fov={h_fov}:"
                f"pitch={self.vr_pitch}:yaw=0:roll=0:"
                f"w={out_w}:h={out_h}:interp=linear"
            )
            vr_filters.append(v360_filter_core)

        return vr_filters

    def _build_ffmpeg_filter_string(self) -> str:
        if self._is_using_preprocessed_video():
            self.logger.debug(f"Using preprocessed video source ('{os.path.basename(self._active_video_source_path)}'). No FFmpeg filters will be applied.")
            return ""

        if not self.video_info:
            return ''

        software_filter_segments = []
        if self.determined_video_type == '2D':
            software_filter_segments = self._get_2d_video_filters()
        elif self.determined_video_type == 'VR':
            software_filter_segments = self._get_vr_video_filters()

        final_filter_chain_parts = []
        if self._needs_hw_download() and software_filter_segments:
            final_filter_chain_parts.extend(["hwdownload", "format=nv12"])
            self.logger.info("Prepending 'hwdownload,format=nv12' to the software filter chain.")

        final_filter_chain_parts.extend(software_filter_segments)
        ffmpeg_filter = ",".join(final_filter_chain_parts)

        self.logger.debug(
            f"Built FFmpeg filter (effective for single pipe, or pipe2 of 10bit-CUDA): {ffmpeg_filter if ffmpeg_filter else 'No explicit filter, direct output.'}")
        return ffmpeg_filter

    def _probe_hwaccel_works(self, hwaccel_args: List[str]) -> bool:
        """Test-decode 1 frame; ffmpeg's metadata layer succeeds even when
        the hwaccel decode silently produces no output (e.g. qsv on non-intel)."""
        if not hwaccel_args:
            return True
        src = self._active_video_source_path
        if not src or not os.path.exists(src):
            return True
        try:
            mtime = os.path.getmtime(src)
        except OSError:
            return True
        try:
            hw_name = hwaccel_args[hwaccel_args.index('-hwaccel') + 1]
        except (ValueError, IndexError):
            hw_name = ' '.join(hwaccel_args)
        key = (src, hw_name, mtime)
        cached = _HWACCEL_PROBE_CACHE.get(key)
        if cached is not None:
            return cached
        from video.ffmpeg_helpers import find_ffmpeg, subprocess_flags
        cmd = [find_ffmpeg(), '-hide_banner', '-loglevel', 'error',
               *hwaccel_args, '-i', src, '-frames:v', '1', '-f', 'null', '-']
        try:
            rc = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=10,
                                creationflags=subprocess_flags()).returncode
            ok = (rc == 0)
        except (OSError, subprocess.TimeoutExpired) as e:
            self.logger.debug(f"hwaccel probe error ({hw_name}): {e}")
            ok = False
        _HWACCEL_PROBE_CACHE[key] = ok
        if not ok:
            self.logger.warning(
                f"Hardware acceleration '{hw_name}' failed runtime probe on "
                f"{os.path.basename(src)}; falling back to CPU decoding.")
        return ok

    def _get_ffmpeg_hwaccel_args(self) -> List[str]:
        """Determines FFmpeg hardware acceleration arguments based on app settings."""
        hwaccel_args: List[str] = []
        selected_hwaccel = getattr(self.app, 'hardware_acceleration_method', 'none') if self.app else "none"
        available_on_app = getattr(self.app, 'available_ffmpeg_hwaccels', []) if self.app else []

        # Force hardware acceleration to "none" for 10-bit or preprocessed videos
        is_10bit_video = self.video_info.get('bit_depth', 8) > 8
        is_preprocessed_video = self._is_using_preprocessed_video()
        
        if is_10bit_video or is_preprocessed_video:
            if is_10bit_video and is_preprocessed_video:
                self.logger.info("Hardware acceleration forced to 'none' for 10-bit preprocessed video (compatibility)")
            elif is_10bit_video:
                self.logger.info("Hardware acceleration forced to 'none' for 10-bit video (compatibility)")
            elif is_preprocessed_video:
                self.logger.info("Hardware acceleration forced to 'none' for preprocessed video (compatibility)")
            return []  # Return empty args = no hardware acceleration

        # NVDEC/CUDA cannot decode H.264 videos wider than 4096 pixels (QSV/VAAPI unaffected)
        video_width = self.video_info.get('width', 0)
        video_codec = self.video_info.get('codec_name', '').lower()
        is_cuda = selected_hwaccel in ('cuda', 'nvdec') or (
            selected_hwaccel == 'auto' and platform.system().lower() != 'darwin'
            and any(m in available_on_app for m in ('nvdec', 'cuda')))
        if is_cuda and video_width > 4096 and video_codec in ('h264', 'avc1', 'avc'):
            self.logger.info(f"CUDA hardware acceleration disabled: H.264 width {video_width} exceeds NVDEC limit of 4096")
            return []

        system = platform.system().lower()
        self.logger.debug(
            f"Determining HWAccel. Selected: '{selected_hwaccel}', OS: {system}, App Available: {available_on_app}")

        if selected_hwaccel == "auto":
            # macOS: CPU-only is 6x faster than VideoToolbox for filter chains
            # Benchmark: CPU 293 FPS vs VideoToolbox 47 FPS
            if system == 'darwin':
                hwaccel_args = []  # Use CPU-only decoding
                self.logger.debug("Auto-selected CPU-only for macOS (6x faster than VideoToolbox for sequential processing with filters).")
            # [REDUNDANCY REMOVED] - Combined Linux/Windows logic
            elif system in ['linux', 'windows']:
                if 'nvdec' in available_on_app or 'cuda' in available_on_app:
                    chosen_nvidia_accel = 'nvdec' if 'nvdec' in available_on_app else 'cuda'
                    # No -hwaccel_output_format=cuda: CPU filters can't take CUDA frames.
                    hwaccel_args = ['-hwaccel', chosen_nvidia_accel]
                    self.logger.debug(f"Auto-selected '{chosen_nvidia_accel}' (NVIDIA) for {system.capitalize()}.")
                elif 'qsv' in available_on_app:
                    hwaccel_args = ['-hwaccel', 'qsv', '-hwaccel_output_format', 'qsv']
                    self.logger.debug(f"Auto-selected 'qsv' (Intel) for {system.capitalize()}.")
                elif system == 'linux' and 'vaapi' in available_on_app:
                    hwaccel_args = ['-hwaccel', 'vaapi', '-hwaccel_output_format', 'vaapi']
                    self.logger.debug("Auto-selected 'vaapi' for Linux.")
                elif system == 'windows' and 'd3d11va' in available_on_app:
                    hwaccel_args = ['-hwaccel', 'd3d11va']
                    self.logger.debug("Auto-selected 'd3d11va' for Windows.")
                elif system == 'windows' and 'dxva2' in available_on_app:
                    hwaccel_args = ['-hwaccel', 'dxva2']
                    self.logger.debug("Auto-selected 'dxva2' for Windows.")

            if not hwaccel_args:
                self.logger.debug("Auto hardware acceleration: No compatible method found, using CPU decoding.")
        elif selected_hwaccel != "none" and selected_hwaccel:
            if selected_hwaccel in available_on_app:
                hwaccel_args = ['-hwaccel', selected_hwaccel]
                if selected_hwaccel == 'qsv':
                    hwaccel_args.extend(['-hwaccel_output_format', 'qsv'])
                elif selected_hwaccel in ['cuda', 'nvdec']:
                    # No cuda output format pin: CPU filters can't take CUDA frames.
                    pass
                elif selected_hwaccel == 'vaapi':
                    hwaccel_args.extend(['-hwaccel_output_format', 'vaapi'])
                self.logger.info(f"User-selected hardware acceleration: '{selected_hwaccel}'. Args: {hwaccel_args}")
            else:
                self.logger.warning(
                    f"Selected HW accel '{selected_hwaccel}' not in FFmpeg's available list. Using CPU.")
        else:
            self.logger.debug("Hardware acceleration explicitly disabled (CPU decoding).")
        # Probe; silent hwaccel failure used to produce 0-frame batch runs.
        if hwaccel_args and not self._probe_hwaccel_works(hwaccel_args):
            return []
        return hwaccel_args
