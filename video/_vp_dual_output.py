"""VideoProcessor DualOutputMixin — extracted from video_processor.py."""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List


class DualOutputMixin:
    """Mixin fragment for VideoProcessor."""

    def set_target_fps(self, fps: float):
        self.target_fps = max(1.0, fps if fps > 0 else 1.0)

    def enable_dual_output_mode(self, fullscreen_resolution: Optional[Tuple[int, int]] = None) -> bool:
        """
        Enable single FFmpeg dual-output mode for perfect synchronization.
        
        Args:
            fullscreen_resolution: Target resolution for fullscreen frames
            
        Returns:
            True if enabled successfully
        """
        try:
            if self.dual_output_enabled:
                self.logger.warning("Dual-output mode already enabled")
                return True
            
            # Enable dual output processor
            self.dual_output_processor.enable_dual_output_mode(fullscreen_resolution)
            
            if self.dual_output_processor.dual_output_enabled:
                self.dual_output_enabled = True
                self.logger.info("VideoProcessor dual-output mode enabled")
                return True
            else:
                self.logger.error("Failed to enable dual-output processor")
                return False
                
        except Exception as e:
            self.logger.error(f"Error enabling dual-output mode: {e}")
            return False

    def disable_dual_output_mode(self) -> bool:
        """
        Disable dual-output mode and return to standard processing.
        
        Returns:
            True if disabled successfully
        """
        try:
            if not self.dual_output_enabled:
                self.logger.info("Dual-output mode already disabled")
                return True
            
            # Disable dual output processor
            self.dual_output_processor.disable_dual_output_mode()
            self.dual_output_enabled = False
            
            self.logger.info("VideoProcessor dual-output mode disabled")
            return True
            
        except Exception as e:
            self.logger.error(f"Error disabling dual-output mode: {e}")
            return False

    def is_dual_output_active(self) -> bool:
        """Check if dual-output mode is active."""
        return (self.dual_output_enabled and 
                self.dual_output_processor.is_dual_output_active())

    def get_dual_output_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get synchronized processing and fullscreen frames from dual output.
        
        Returns:
            Tuple of (processing_frame, fullscreen_frame)
        """
        if not self.dual_output_enabled:
            return None, None
        return self.dual_output_processor.get_dual_frames()

    def get_fullscreen_frame(self) -> Optional[np.ndarray]:
        """Get the latest fullscreen frame for display."""
        if not self.dual_output_enabled:
            return None
        return self.dual_output_processor.get_fullscreen_frame()

    def get_audio_buffer(self) -> Optional[np.ndarray]:
        """Get the latest audio buffer for sound."""
        if not self.dual_output_enabled:
            return None
        return self.dual_output_processor.get_audio_buffer()

    def get_dual_output_stats(self) -> Dict[str, Any]:
        """Get statistics about dual-output processing."""
        if not self.dual_output_enabled:
            return {'dual_output_enabled': False}
        return self.dual_output_processor.get_frame_stats()

    def _start_dual_output_ffmpeg_process(self, start_frame_abs_idx=0, num_frames_to_output_ffmpeg=None) -> bool:
        """
        Start FFmpeg process using the single FFmpeg dual-output architecture.
        
        Args:
            start_frame_abs_idx: Starting frame index
            num_frames_to_output_ffmpeg: Number of frames to output (optional)
            
        Returns:
            True if started successfully
        """
        try:
            if not self.dual_output_processor.dual_output_enabled:
                self.logger.error("Dual output processor not enabled")
                return False
            
            start_time_seconds = start_frame_abs_idx / self.video_info['fps']
            self.current_stream_start_frame_abs = start_frame_abs_idx
            self.frames_read_from_current_stream = 0
            
            # Build base FFmpeg command
            base_cmd = self._build_base_ffmpeg_command(start_time_seconds, num_frames_to_output_ffmpeg)
            
            # Enhance command for dual output
            dual_output_cmd = self.dual_output_processor.build_single_ffmpeg_dual_output_command(base_cmd)
            
            # Start the single FFmpeg process with dual outputs
            success = self.dual_output_processor.start_single_ffmpeg_process(dual_output_cmd)
            
            if success:
                self.logger.info("Single FFmpeg dual-output process started successfully")
                return True
            else:
                self.logger.error("Failed to start single FFmpeg dual-output process")
                return False
                
        except Exception as e:
            self.logger.error(f"Error starting dual-output FFmpeg process: {e}")
            return False

    def _build_base_ffmpeg_command(self, start_time_seconds: float, num_frames_to_output: Optional[int] = None) -> List[str]:
        """
        Build base FFmpeg command with input arguments and filters.
        
        Args:
            start_time_seconds: Start time in seconds
            num_frames_to_output: Number of frames to output (optional)
            
        Returns:
            Base FFmpeg command list
        """
        cmd = ['ffmpeg', '-hide_banner', '-nostats', '-loglevel', 'error']
        
        # Add hardware acceleration arguments
        hwaccel_args = self._get_ffmpeg_hwaccel_args()
        cmd.extend(hwaccel_args)
        
        # Add input file with seeking
        cmd.extend(['-ss', str(start_time_seconds), '-i', self.video_path])
        
        # Add frame limiting if specified
        if num_frames_to_output and num_frames_to_output > 0:
            cmd.extend(['-frames:v', str(num_frames_to_output)])
        
        # Add audio and subtitle options
        cmd.extend(['-an', '-sn'])  # No audio, no subtitles initially (dual processor handles audio separately)
        
        # Add video filter for processing
        effective_vf = self.ffmpeg_filter_string or f"scale={self.yolo_input_size}:{self.yolo_input_size}:force_original_aspect_ratio=decrease,pad={self.yolo_input_size}:{self.yolo_input_size}:(ow-iw)/2:(oh-ih)/2:black"
        cmd.extend(['-vf', effective_vf])
        
        return cmd
