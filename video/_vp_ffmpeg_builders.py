"""VideoProcessor FFmpegBuildersMixin — extracted from video_processor.py."""

import os
import platform
import numpy as np
from typing import Optional, List
from video.thumbnail_extractor import ThumbnailExtractor


class FFmpegBuildersMixin:
    """Mixin fragment for VideoProcessor."""

    def _is_10bit_cuda_pipe_needed(self) -> bool:
        # TODO: Add bitshift processing for 10-bit videos (fast 10-bit to 8-bit conversion).
        # Optional: Scale to 640x640 on GPU using tensorrt. This will not use lanczos. So if Lanczos is absolutely necessary, you will have to use other solution.
        """Checks if the special 2-pipe FFmpeg command for 10-bit CUDA should be used."""
        if not self.video_info:
            return False

        is_high_bit_depth = self.video_info.get('bit_depth', 8) > 8
        hwaccel_args = self._get_ffmpeg_hwaccel_args()
        # [OPTIMIZED] Simpler check
        is_cuda_hwaccel = 'cuda' in hwaccel_args

        if is_high_bit_depth and is_cuda_hwaccel:
            self.logger.info("Conditions for 10-bit CUDA pipe met.")
            return True
        return False

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
        """Builds the list of FFmpeg filter segments for standard 2D video."""
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
            return [
                f"scale={self.yolo_input_size}:-1:force_original_aspect_ratio=decrease",
                f"pad={self.yolo_input_size}:{self.yolo_input_size}:0:(oh-ih)/2:black"
            ]
        else:
            # Taller than wide (portrait) - scale and pad left/right
            return [
                f"scale=-1:{self.yolo_input_size}:force_original_aspect_ratio=decrease",
                f"pad={self.yolo_input_size}:{self.yolo_input_size}:(ow-iw)/2:0:black"
            ]

    def _init_gpu_unwarp_worker(self):
        """Initialize GPU unwarp worker for VR video processing."""
        from config.constants import ENABLE_GPU_UNWARP, GPU_UNWARP_BACKEND

        # Check if user wants CPU v360 or no unwarp (crop only)
        if self.vr_unwarp_method_override in ('v360', 'none'):
            self.logger.debug(f"User selected {self.vr_unwarp_method_override} unwarp method - GPU unwarp disabled")
            # Clean up existing GPU unwarp worker
            if self.gpu_unwarp_worker:
                self.logger.debug(f"Stopping existing GPU unwarp worker (switching to {self.vr_unwarp_method_override})")
                self.gpu_unwarp_worker.stop()
                self.gpu_unwarp_worker = None
            self.gpu_unwarp_enabled = False
            return

        # Only initialize for VR videos when GPU unwarp is enabled
        if self.determined_video_type != 'VR' or not ENABLE_GPU_UNWARP:
            # Clean up GPU unwarp worker if it exists but shouldn't be used
            if self.gpu_unwarp_worker:
                self.logger.info("Stopping GPU unwarp worker (not VR or GPU unwarp disabled)")
                self.gpu_unwarp_worker.stop()
                self.gpu_unwarp_worker = None
            self.gpu_unwarp_enabled = False
            return

        # If already initialized, skip (prevents duplicate workers)
        if self.gpu_unwarp_enabled and self.gpu_unwarp_worker:
            self.logger.debug("GPU unwarp worker already initialized, skipping")
            return

        try:
            from video.gpu_unwarp_worker import GPUUnwarpWorker

            # Get projection type from VR input format
            projection_type = self.vr_input_format.replace('_sbs', '').replace('_tb', '')
            if 'fisheye' in projection_type or 'he' in projection_type:
                # Map VR format to projection type
                if projection_type == 'fisheye':
                    projection_type = f'fisheye{int(self.vr_fov)}'
                elif projection_type == 'he':
                    projection_type = 'equirect180'

            # Determine backend based on user override or default
            if self.vr_unwarp_method_override in ['metal', 'opengl']:
                backend = self.vr_unwarp_method_override
            else:
                backend = GPU_UNWARP_BACKEND  # Use default (auto)

            self.gpu_unwarp_worker = GPUUnwarpWorker(
                projection_type=projection_type,
                output_size=self.yolo_input_size,
                queue_size=16,  # Increased for batch processing
                backend=backend,
                pitch=self.vr_pitch,  # Pass directly (matches benchmark behavior)
                yaw=0.0,
                roll=0.0,
                batch_size=4,  # Enable batch processing (12% faster)
                input_format='bgr24'  # Input is BGR24 from FFmpeg, worker converts to RGBA internally
            )
            self.gpu_unwarp_worker.start()
            self.gpu_unwarp_enabled = True
            self.logger.info(f"GPU unwarp worker started (backend={backend}, projection={projection_type}, pitch={self.vr_pitch})")

        except Exception as e:
            self.logger.warning(f"Failed to initialize GPU unwarp worker: {e}. Falling back to CPU v360.")
            self.gpu_unwarp_worker = None
            self.gpu_unwarp_enabled = False

    def _init_thumbnail_extractor(self):
        """Initialize FFmpeg-based thumbnail extractor for frame-accurate random access."""
        # Close existing extractor if present
        if self.thumbnail_extractor:
            self.thumbnail_extractor.close()
            self.thumbnail_extractor = None

        # Only initialize if we have a valid video
        if not self._active_video_source_path or not self.video_info:
            return

        try:
            # Don't apply VR cropping if using preprocessed video (already cropped/unwrapped)
            vr_format = None
            if self.determined_video_type == 'VR' and not self._is_using_preprocessed_video():
                vr_format = self.vr_input_format

            extractor = ThumbnailExtractor(
                video_path=self._active_video_source_path,
                fps=self.fps,
                total_frames=self.total_frames,
                output_size=self.yolo_input_size,
                vr_input_format=vr_format,
                vr_fov=getattr(self, 'vr_fov', 190),
                vr_pitch=getattr(self, 'vr_pitch', 0.0),
                logger=self.logger,
            )
            if extractor.is_open:
                self.thumbnail_extractor = extractor
                source_type = "preprocessed" if self._is_using_preprocessed_video() else "original"
                self.logger.debug(f"Thumbnail extractor initialized (FFmpeg-based, {source_type} video)")
            else:
                self.logger.warning("Thumbnail extractor probe failed, falling back to FFmpeg batch")

        except Exception as e:
            self.logger.warning(f"Failed to initialize thumbnail extractor: {e}")
            self.thumbnail_extractor = None

    def get_thumbnail_frame(self, frame_index: int, **_kwargs) -> Optional[np.ndarray]:
        """
        Get a frame-accurate thumbnail using FFmpeg input-level seeking.

        For VR content, applies v360 CPU dewarp (negligible cost vs decode).
        Optimized for random frame access (timeline tooltips, seek preview).

        Returns:
            Frame as BGR24 numpy array (yolo_input_size x yolo_input_size) or None
        """
        if self.thumbnail_extractor is None:
            self.logger.debug("Thumbnail extractor not available, falling back to FFmpeg batch")
            return self._get_specific_frame(frame_index, update_current_index=False, use_thumbnail=True)

        try:
            return self.thumbnail_extractor.get_frame(frame_index)
        except Exception as e:
            self.logger.warning(f"Thumbnail extraction failed: {e}, falling back to FFmpeg batch")
            return self._get_specific_frame(frame_index, update_current_index=False, use_thumbnail=True)

    def _get_vr_video_filters(self) -> List[str]:
        """Builds the list of FFmpeg filter segments for VR video, including cropping and v360."""
        from config.constants import ENABLE_GPU_UNWARP

        if not self.video_info:
            return []

        original_width = self.video_info.get('width', 0)
        original_height = self.video_info.get('height', 0)
        v_h_FOV = 90  # Default vertical and horizontal FOV for the output projection

        vr_filters = []
        is_sbs_format = '_sbs' in self.vr_input_format
        is_tb_format = '_tb' in self.vr_input_format
        is_lr_format = '_lr' in self.vr_input_format
        is_rl_format = '_rl' in self.vr_input_format
        is_side_by_side = is_sbs_format or is_lr_format or is_rl_format

        # For "none" (crop only) mode, vr_crop_panel selects which panel to use
        use_second_panel = (self.vr_unwarp_method_override == 'none'
                            and getattr(self, 'vr_crop_panel', 'first') == 'second')

        if is_sbs_format and original_width > 0 and original_height > 0:
            crop_w = original_width / 2
            crop_h = original_height
            crop_x = int(original_width / 2) if use_second_panel else 0
            vr_filters.append(f"crop={int(crop_w)}:{int(crop_h)}:{crop_x}:0")
            panel_label = "right" if use_second_panel else "left"
            self.logger.debug(f"Applying SBS pre-crop ({panel_label}): w={int(crop_w)} h={int(crop_h)} x={crop_x} y=0")
        elif is_tb_format and original_width > 0 and original_height > 0:
            crop_w = original_width
            crop_h = original_height / 2
            crop_y = int(original_height / 2) if use_second_panel else 0
            vr_filters.append(f"crop={int(crop_w)}:{int(crop_h)}:0:{crop_y}")
            panel_label = "bottom" if use_second_panel else "top"
            self.logger.info(f"Applying TB pre-crop ({panel_label}): w={int(crop_w)} h={int(crop_h)} x=0 y={crop_y}")
        elif is_lr_format and original_width > 0 and original_height > 0:
            crop_w = original_width / 2
            crop_h = original_height
            crop_x = int(original_width / 2) if use_second_panel else 0
            vr_filters.append(f"crop={int(crop_w)}:{int(crop_h)}:{crop_x}:0")
            panel_label = "right" if use_second_panel else "left"
            self.logger.info(f"Applying LR pre-crop ({panel_label} panel): w={int(crop_w)} h={int(crop_h)} x={crop_x} y=0")
        elif is_rl_format and original_width > 0 and original_height > 0:
            # RL format: right panel is first (x=0), left panel is second (x=w/2)
            crop_w = original_width / 2
            crop_h = original_height
            crop_x = 0 if use_second_panel else int(original_width / 2)
            vr_filters.append(f"crop={int(crop_w)}:{int(crop_h)}:{crop_x}:0")
            panel_label = "left" if use_second_panel else "right"
            self.logger.info(f"Applying RL pre-crop ({panel_label} panel): w={int(crop_w)} h={int(crop_h)} x={crop_x} y=0")

        # Decide unwarp method: none (crop only), GPU unwarp, or CPU v360
        if self.vr_unwarp_method_override == 'none':
            # No unwarp — just scale the cropped panel to YOLO input size
            vr_filters.append(f"scale={self.yolo_input_size}:{self.yolo_input_size}")
            self.logger.info("Unwarp: None (crop only) - no dewarping applied, just crop+scale")
        elif ENABLE_GPU_UNWARP and self.vr_unwarp_method_override != 'v360':
            # GPU unwarp enabled — skip v360, just scale; unwrapping done by GPU worker
            vr_filters.append(f"scale={self.yolo_input_size}:{self.yolo_input_size}")
            self.logger.info(f"GPU unwarp enabled (method={self.vr_unwarp_method_override}) - using crop+scale (v360 skipped)")
        else:
            # CPU v360 dewarp (user override or GPU unwarp disabled)
            base_v360_input_format = self.vr_input_format.replace('_sbs', '').replace('_tb', '').replace('_lr', '').replace('_rl', '')
            v360_filter_core = (
                f"v360={base_v360_input_format}:in_stereo=0:output=sg:"
                f"iv_fov={self.vr_fov}:ih_fov={self.vr_fov}:"
                f"d_fov={self.vr_fov}:"
                f"v_fov={v_h_FOV}:h_fov={v_h_FOV}:"
                f"pitch={self.vr_pitch}:yaw=0:roll=0:"
                f"w={self.yolo_input_size}:h={self.yolo_input_size}:interp=linear"
            )
            vr_filters.append(v360_filter_core)
            self.logger.debug(f"Using CPU v360 filter: {v360_filter_core}")

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
                    hwaccel_args = ['-hwaccel', chosen_nvidia_accel, '-hwaccel_output_format', 'cuda']
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
                self.logger.info("Auto hardware acceleration: No compatible method found, using CPU decoding.")
        elif selected_hwaccel != "none" and selected_hwaccel:
            if selected_hwaccel in available_on_app:
                hwaccel_args = ['-hwaccel', selected_hwaccel]
                if selected_hwaccel == 'qsv':
                    hwaccel_args.extend(['-hwaccel_output_format', 'qsv'])
                elif selected_hwaccel in ['cuda', 'nvdec']:
                    hwaccel_args.extend(['-hwaccel_output_format', 'cuda'])
                elif selected_hwaccel == 'vaapi':
                    hwaccel_args.extend(['-hwaccel_output_format', 'vaapi'])
                self.logger.info(f"User-selected hardware acceleration: '{selected_hwaccel}'. Args: {hwaccel_args}")
            else:
                self.logger.warning(
                    f"Selected HW accel '{selected_hwaccel}' not in FFmpeg's available list. Using CPU.")
        else:
            self.logger.debug("Hardware acceleration explicitly disabled (CPU decoding).")
        return hwaccel_args
