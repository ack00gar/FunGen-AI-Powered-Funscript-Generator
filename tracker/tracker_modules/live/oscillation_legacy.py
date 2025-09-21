#!/usr/bin/env python3
"""
Legacy Oscillation Detector - Original oscillation tracking algorithm.

This tracker implements the legacy oscillation detection algorithm from commit c9e6fbd
that uses cohesion analysis, frequency weighting, and percentile-based amplification
for natural signal processing with superior amplitude detection.

Author: Migrated from legacy codebase
Version: 1.0.0
"""

import logging
import time
import numpy as np
import cv2
from collections import deque
from typing import Dict, Any, Optional, List, Tuple

try:
    from ..core.base_tracker import BaseTracker, TrackerMetadata, TrackerResult
except ImportError:
    from tracker.tracker_modules.core.base_tracker import BaseTracker, TrackerMetadata, TrackerResult


class OscillationLegacyTracker(BaseTracker):
    """
    Legacy oscillation detector with cohesion analysis and frequency weighting.
    
    This tracker excels at:
    - Superior signal amplification using percentile normalization
    - Cohesion analysis for natural motion cluster detection
    - Adaptive frequency weighting with bell curve at 2.5Hz
    - Live dynamic amplification with anti-plateau logic
    - Dense optical flow analysis with black bar culling
    """
    
    def __init__(self):
        super().__init__()
        
        # Core state
        self.prev_gray_oscillation = None
        self.flow_dense = None
        
        # Grid configuration - use smaller grid for better performance
        self.oscillation_grid_size = 10  # Match Experimental 2's performance
        self.oscillation_block_size = 64  # Will be calculated dynamically
        
        # Oscillation detection state (matches original tracker.py exactly)
        self.oscillation_history = {}
        self.oscillation_history_max_len = 60  # Original: 60 frames (2s * 30fps)
        self.oscillation_history_seconds = 2.0
        
        # Position tracking (matches original tracker.py exactly)
        self.oscillation_position_history = deque(maxlen=120)  # Original: 4 seconds @ 30fps
        self.oscillation_last_known_pos = 50.0
        self.oscillation_last_known_secondary_pos = 50.0
        self.oscillation_funscript_pos = 50
        self.oscillation_funscript_secondary_pos = 50
        
        # EMA smoothing
        self.oscillation_ema_alpha = 0.3
        
        # Live amplification
        self.live_amp_enabled = True
        
        # Frame preprocessing
        self.preprocess_frame_enabled = True
        
        # FPS tracking (missing from legacy implementation)
        self.current_fps = 0.0
        self._fps_last_time = 0.0
        
        # Buffer management for optimization (like experimental tracker)
        self._gray_buffer = None
        self._prev_gray_osc_buffer = None
    
    @property
    def metadata(self) -> TrackerMetadata:
        """Return metadata describing this tracker."""
        return TrackerMetadata(
            name="oscillation_legacy",
            display_name="Oscillation Detector (Legacy)",
            description="Original oscillation tracker with cohesion analysis and superior amplification",
            category="live",
            version="1.0.0",
            author="Legacy Codebase",
            tags=["oscillation", "optical-flow", "legacy", "amplification"],
            requires_roi=False,
            supports_dual_axis=True
        )
    
    def initialize(self, app_instance, **kwargs) -> bool:
        """Initialize the legacy oscillation detector."""
        try:
            self.app = app_instance
            
            # Load settings
            if hasattr(app_instance, 'app_settings'):
                settings = app_instance.app_settings
                
                # Use smaller grid size for better performance (can be overridden in settings)
                self.oscillation_grid_size = settings.get('oscillation_detector_grid_size', 10)
                self.live_amp_enabled = settings.get('live_oscillation_dynamic_amp_enabled', True)
                self.oscillation_sensitivity = settings.get('oscillation_detector_sensitivity', 1.0)
                self.oscillation_ema_alpha = settings.get('oscillation_ema_alpha', 0.3)
                self.preprocess_frame_enabled = settings.get('preprocess_frame_enabled', True)
                
                self.logger.info(f"Legacy oscillation settings: grid_size={self.oscillation_grid_size}, "
                               f"live_amp={self.live_amp_enabled}, sensitivity={self.oscillation_sensitivity}, ema_alpha={self.oscillation_ema_alpha}")
            
            # Calculate block size exactly like original tracker.py
            try:
                from config import constants
                self.oscillation_block_size = constants.YOLO_INPUT_SIZE // self.oscillation_grid_size
                self.logger.info(f"Block size calculated: {constants.YOLO_INPUT_SIZE} // {self.oscillation_grid_size} = {self.oscillation_block_size}")
            except ImportError:
                # Fallback if constants not available
                self.oscillation_block_size = 640 // self.oscillation_grid_size
                self.logger.warning(f"Using fallback YOLO_INPUT_SIZE=640, block size: {self.oscillation_block_size}")
            
            if self.oscillation_block_size <= 0:
                self.oscillation_block_size = 80  # Default fallback
            
            # Initialize optical flow - use DIS with ultrafast preset for better performance
            try:
                self.flow_dense = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
                self.logger.info("DIS optical flow initialized (ultrafast preset) for legacy oscillation")
            except AttributeError:
                try:
                    # Fallback to medium preset if ultrafast not available
                    self.flow_dense = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
                    self.logger.info("DIS optical flow initialized (medium preset) for legacy oscillation")
                except AttributeError:
                    self.logger.error("No DIS optical flow implementation available")
                    return False
            
            # Reset state
            self.prev_gray_oscillation = None
            self.oscillation_history.clear()
            self.oscillation_position_history.clear()
            self.oscillation_last_known_pos = 50.0
            self.oscillation_last_known_secondary_pos = 50.0
            
            self._initialized = True
            self.logger.info("Legacy oscillation detector initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray, frame_time_ms: int, 
                     frame_index: Optional[int] = None) -> TrackerResult:
        """
        Process a frame using legacy oscillation detection algorithm.
        
        This implementation follows the original c9e6fbd algorithm with:
        - Black bar detection and culling
        - Dense optical flow analysis
        - Cohesion analysis for motion clusters
        - Adaptive frequency weighting
        - Percentile-based amplification
        """
        try:
            self._update_fps()
            
            # No target height needed - video processor handles scaling
            
            if frame is None or frame.size == 0:
                return TrackerResult(
                    processed_frame=frame,
                    action_log=None,
                    debug_info={'error': 'Invalid frame'},
                    status_message="Error: Invalid frame"
                )
            
            # No resizing - video processor handles frame scaling via ffmpeg
            processed_frame = frame
            
            # Use buffer reuse optimization for performance
            target_h, target_w = processed_frame.shape[0], processed_frame.shape[1]
            
            if (self._gray_buffer is None or 
                self._gray_buffer.shape[:2] != (target_h, target_w)):
                self._gray_buffer = np.empty((target_h, target_w), dtype=np.uint8)
            
            cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY, dst=self._gray_buffer)
            current_gray = self._gray_buffer
            action_log_list = []
            
            # Initialize on first frame
            if self.prev_gray_oscillation is None or self.prev_gray_oscillation.shape != current_gray.shape:
                # Use buffer for previous frame to avoid memory allocation
                if (self._prev_gray_osc_buffer is None or 
                    self._prev_gray_osc_buffer.shape[:2] != (target_h, target_w)):
                    self._prev_gray_osc_buffer = np.empty((target_h, target_w), dtype=np.uint8)
                
                self._prev_gray_osc_buffer[:] = current_gray  # Copy data
                self.prev_gray_oscillation = self._prev_gray_osc_buffer
                return TrackerResult(
                    processed_frame=processed_frame,
                    action_log=None,
                    debug_info={'status': 'initializing'},
                    status_message="Initializing legacy oscillation detector..."
                )
            
            if not self.flow_dense:
                return TrackerResult(
                    processed_frame=processed_frame,
                    action_log=None,
                    debug_info={'error': 'No optical flow available'},
                    status_message="Error: Dense optical flow not available"
                )
            
            # Detect active video area to ignore black bars
            active_area_rect = self._detect_active_area(current_gray)
            
            # Calculate optical flow
            prev_gray_cont = np.ascontiguousarray(self.prev_gray_oscillation)
            current_gray_cont = np.ascontiguousarray(current_gray)
            flow = self.flow_dense.calc(prev_gray_cont, current_gray_cont, None)
            
            if flow is None:
                # Update previous frame using buffer copy for performance
                self._prev_gray_osc_buffer[:] = current_gray
                return TrackerResult(
                    processed_frame=processed_frame,
                    action_log=None,
                    debug_info={'error': 'Flow calculation failed'},
                    status_message="Error: Optical flow calculation failed"
                )
            
            # Analyze flow in grid blocks
            block_motions = self._analyze_grid_motion(flow, active_area_rect)
            
            # Detect camera motion
            is_camera_motion = self._detect_camera_motion(block_motions)
            
            # Find active blocks using cohesion analysis
            active_blocks = []
            if not is_camera_motion:
                active_blocks = self._perform_cohesion_analysis()
            
            # Calculate final positions
            primary_pos, secondary_pos = self._calculate_positions(active_blocks)
            
            # Apply live dynamic amplification
            amplified_primary = self._apply_live_amplification(primary_pos)
            
            # Apply EMA smoothing
            self.oscillation_last_known_pos = (self.oscillation_last_known_pos * (1 - self.oscillation_ema_alpha) + 
                                             amplified_primary * self.oscillation_ema_alpha)
            self.oscillation_last_known_secondary_pos = (self.oscillation_last_known_secondary_pos * (1 - self.oscillation_ema_alpha) + 
                                                       secondary_pos * self.oscillation_ema_alpha)
            
            self.oscillation_funscript_pos = int(round(self.oscillation_last_known_pos))
            self.oscillation_funscript_secondary_pos = int(round(self.oscillation_last_known_secondary_pos))
            
            # Generate funscript actions if tracking is active
            if self.tracking_active:
                action_log_list = self._generate_actions(frame_time_ms)
            
            # Apply visualization
            self._draw_visualization(processed_frame, block_motions, active_blocks, is_camera_motion)
            
            # Prepare debug info
            debug_info = {
                'primary_position': self.oscillation_funscript_pos,
                'secondary_position': self.oscillation_funscript_secondary_pos,
                'active_blocks': len(active_blocks),
                'total_blocks': len(block_motions),
                'camera_motion': is_camera_motion,
                'amplification_active': self.live_amp_enabled and len(self.oscillation_position_history) > 50,
                'tracking_active': self.tracking_active,
                'fps': round(self.current_fps, 1)
            }
            
            status_msg = f"Legacy | Pos: {self.oscillation_funscript_pos} | Active blocks: {len(active_blocks)}"
            if is_camera_motion:
                status_msg += " | Camera motion detected"
            
            # Update previous frame using buffer copy for performance
            self._prev_gray_osc_buffer[:] = current_gray
            
            return TrackerResult(
                processed_frame=processed_frame,
                action_log=action_log_list if action_log_list else None,
                debug_info=debug_info,
                status_message=status_msg
            )
            
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            return TrackerResult(
                processed_frame=frame,
                action_log=None,
                debug_info={'error': str(e)},
                status_message=f"Error: {e}"
            )
    
    def start_tracking(self) -> bool:
        """Start legacy oscillation tracking."""
        if not self._initialized:
            self.logger.error("Tracker not initialized")
            return False
        
        # Calculate dynamic history length based on FPS like original tracker.py
        fps = 30.0  # Default FPS
        if hasattr(self.app, 'processor') and self.app.processor and self.app.processor.fps > 0:
            fps = self.app.processor.fps
        
        # Update history max length dynamically (matches original)
        self.oscillation_history_max_len = int(self.oscillation_history_seconds * fps)
        
        # Update amplification history size (matches original)
        amp_window_ms = 4000  # Default 4 seconds
        if hasattr(self.app, 'app_settings'):
            amp_window_ms = self.app.app_settings.get("live_oscillation_amp_window_ms", 4000)
        new_maxlen = int((amp_window_ms / 1000.0) * fps)
        self.oscillation_position_history = deque(maxlen=new_maxlen)
        
        self.tracking_active = True
        self.oscillation_history.clear()
        self.prev_gray_oscillation = None
        self.oscillation_funscript_pos = 50
        
        self.logger.info(f"Legacy oscillation tracking started - FPS: {fps}, "
                        f"history_max_len: {self.oscillation_history_max_len}, "
                        f"position_history_maxlen: {new_maxlen}")
        return True
    
    def stop_tracking(self) -> bool:
        """Stop legacy oscillation tracking."""
        self.tracking_active = False
        self.logger.info("Legacy oscillation tracking stopped")
        return True
    
    def validate_settings(self, settings: Dict[str, Any]) -> bool:
        """Validate legacy oscillation settings."""
        try:
            grid_size = settings.get('oscillation_grid_size', self.oscillation_grid_size)
            if not isinstance(grid_size, int) or grid_size < 4 or grid_size > 16:
                self.logger.error("Grid size must be between 4 and 16")
                return False
            
            ema_alpha = settings.get('oscillation_ema_alpha', self.oscillation_ema_alpha)
            if not isinstance(ema_alpha, (int, float)) or not (0 < ema_alpha <= 1):
                self.logger.error("EMA alpha must be between 0 and 1")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Settings validation error: {e}")
            return False
    
    def get_status_info(self) -> Dict[str, Any]:
        """Get current status information."""
        base_status = super().get_status_info()
        
        custom_status = {
            'grid_size': self.oscillation_grid_size,
            'block_size': self.oscillation_block_size,
            'history_length': len(self.oscillation_history),
            'position_history_length': len(self.oscillation_position_history),
            'live_amplification': self.live_amp_enabled,
            'current_primary_pos': self.oscillation_funscript_pos,
            'current_secondary_pos': self.oscillation_funscript_secondary_pos
        }
        
        base_status.update(custom_status)
        return base_status
    
    def cleanup(self):
        """Clean up resources."""
        self.prev_gray_oscillation = None
        self.flow_dense = None
        self.oscillation_history.clear()
        self.oscillation_position_history.clear()
        # self.logger.info("Legacy oscillation detector cleaned up")
    
    # Private helper methods
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Basic frame preprocessing - kept for compatibility but not used."""
        return frame  # Preprocessing is done inline in process_frame method
    
    def _detect_active_area(self, gray_frame: np.ndarray) -> Tuple[int, int, int, int]:
        """Detect active video area to ignore black bars."""
        active_area_rect = (0, 0, gray_frame.shape[1], gray_frame.shape[0])
        
        if np.any(gray_frame):
            rows = np.any(gray_frame, axis=1)
            cols = np.any(gray_frame, axis=0)
            if np.any(rows) and np.any(cols):
                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]
                active_area_rect = (x_min, y_min, x_max, y_max)
        
        return active_area_rect
    
    def _analyze_grid_motion(self, flow: np.ndarray, active_area_rect: Tuple[int, int, int, int]) -> List[Dict]:
        """Analyze optical flow in grid blocks."""
        block_motions = []
        ax_min, ay_min, ax_max, ay_max = active_area_rect
        
        for r in range(self.oscillation_grid_size):
            for c in range(self.oscillation_grid_size):
                y_start, x_start = r * self.oscillation_block_size, c * self.oscillation_block_size
                
                # Skip blocks in black bar areas
                block_center_x = x_start + self.oscillation_block_size / 2
                block_center_y = y_start + self.oscillation_block_size / 2
                
                if not (ax_min < block_center_x < ax_max and ay_min < block_center_y < ay_max):
                    continue
                
                block_flow = flow[y_start:y_start + self.oscillation_block_size,
                             x_start:x_start + self.oscillation_block_size]
                
                if block_flow.size > 0:
                    dx, dy = np.median(block_flow[..., 0]), np.median(block_flow[..., 1])
                    mag = np.sqrt(dx ** 2 + dy ** 2)
                    block_motions.append({'dx': dx, 'dy': dy, 'mag': mag, 'pos': (r, c)})
                    
                    # Update oscillation history
                    block_pos = (r, c)
                    if block_pos not in self.oscillation_history:
                        self.oscillation_history[block_pos] = deque(maxlen=self.oscillation_history_max_len)
                    self.oscillation_history[block_pos].append({'dx': dx, 'dy': dy, 'mag': mag})
        
        return block_motions
    
    def _detect_camera_motion(self, block_motions: List[Dict]) -> bool:
        """Detect global camera motion."""
        if not block_motions:
            return False
        
        median_dx = np.median([m['dx'] for m in block_motions])
        median_dy = np.median([m['dy'] for m in block_motions])
        
        if np.sqrt(median_dx ** 2 + median_dy ** 2) <= 1.0:
            return False
        
        coherent_blocks = 0
        vec_median = np.array([median_dx, median_dy])
        
        for motion in block_motions:
            vec_block = np.array([motion['dx'], motion['dy']])
            if np.linalg.norm(vec_block) > 0.5:
                cosine_sim = np.dot(vec_block, vec_median) / (np.linalg.norm(vec_block) * np.linalg.norm(vec_median) + 1e-6)
                if cosine_sim > 0.8:
                    coherent_blocks += 1
        
        return (coherent_blocks / len(block_motions)) > 0.85
    
    def _perform_cohesion_analysis(self) -> List[Dict]:
        """Perform cohesion analysis to find natural motion clusters."""
        candidate_blocks = []
        
        for pos, history in self.oscillation_history.items():
            if len(history) < self.oscillation_history_max_len * 0.8:
                continue
            
            mags = [h['mag'] for h in history]
            dys = [h['dy'] for h in history]
            mean_mag = np.mean(mags)
            std_dev_dy = np.std(dys)
            
            if mean_mag < 0.5 or (mean_mag > 0 and std_dev_dy / mean_mag < 0.5):
                continue
            
            # Frequency analysis with smoothing
            smoothed_dys = np.convolve(dys, np.ones(5) / 5, mode='valid')
            if len(smoothed_dys) < 2:
                continue
            
            freq = (len(np.where(np.diff(np.sign(smoothed_dys)))[0]) / 2) / self.oscillation_history_seconds
            
            # Adaptive frequency weighting (bell curve centered at 2.5Hz)
            if 0.5 <= freq <= 7.0:
                freq_weight = np.exp(-((freq - 2.5) ** 2) / (2 * (1.5 ** 2)))
                score = mean_mag * freq * freq_weight
                candidate_blocks.append({
                    'pos': pos, 
                    'score': score, 
                    'dy': history[-1]['dy'], 
                    'dx': history[-1]['dx'], 
                    'mag': history[-1]['mag']
                })
        
        if not candidate_blocks:
            return []
        
        # Apply cohesion boost
        candidate_pos = {b['pos'] for b in candidate_blocks}
        for block in candidate_blocks:
            r, c = block['pos']
            cohesion_boost = 1.0
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    if (r + dr, c + dc) in candidate_pos:
                        cohesion_boost += 0.2  # 20% boost per active neighbor
            block['score'] *= cohesion_boost
        
        # Filter by relative score threshold
        max_score = max(b['score'] for b in candidate_blocks)
        active_blocks = [b for b in candidate_blocks if b['score'] > max_score * 0.6]
        
        return active_blocks
    
    def _calculate_positions(self, active_blocks: List[Dict]) -> Tuple[float, float]:
        """Calculate weighted primary and secondary positions."""
        if active_blocks:
            total_weight = sum(b['score'] for b in active_blocks)
            final_dy = sum(b['dy'] * b['score'] for b in active_blocks) / total_weight
            final_dx = sum(b['dx'] * b['score'] for b in active_blocks) / total_weight
            final_mag = sum(b['mag'] * b['score'] for b in active_blocks) / total_weight
            
            # Amplitude scaling based on magnitude
            amplitude_scaler = np.clip(final_mag / 4.0, 0.7, 1.5)
            max_deviation = 50 * amplitude_scaler
            
            scaled_dy = np.clip(final_dy * -10, -max_deviation, max_deviation)
            scaled_dx = np.clip(final_dx * 10, -max_deviation, max_deviation)
            
            primary_pos = np.clip(50 + scaled_dy, 0, 100)
            secondary_pos = np.clip(50 + scaled_dx, 0, 100)
        else:
            # Decay towards center when no motion
            primary_pos = self.oscillation_last_known_pos * 0.95 + 50 * 0.05
            secondary_pos = self.oscillation_last_known_secondary_pos * 0.95 + 50 * 0.05
        
        return primary_pos, secondary_pos
    
    def _apply_live_amplification(self, primary_pos: float) -> float:
        """Apply live dynamic amplification using percentiles."""
        self.oscillation_position_history.append(primary_pos)
        
        if not (self.live_amp_enabled and len(self.oscillation_position_history) > self.oscillation_position_history.maxlen * 0.5):
            return primary_pos
        
        p10 = np.percentile(self.oscillation_position_history, 10)
        p90 = np.percentile(self.oscillation_position_history, 90)
        effective_range = p90 - p10
        
        if effective_range > 15:  # Original amplification threshold
            normalized_pos = (primary_pos - p10) / effective_range
            return np.clip(normalized_pos * 100, 0, 100)
        
        return primary_pos
    
    def _generate_actions(self, frame_time_ms: int) -> List[Dict]:
        """Generate funscript actions based on tracking axis mode."""
        action_log_list = []
        
        # Get current tracking settings
        current_tracking_axis_mode = getattr(self.app, 'tracking_axis_mode', 'both')
        current_single_axis_output = getattr(self.app, 'single_axis_output_target', 'primary')
        
        primary_to_write, secondary_to_write = None, None
        
        if current_tracking_axis_mode == "both":
            primary_to_write, secondary_to_write = self.oscillation_funscript_pos, self.oscillation_funscript_secondary_pos
        elif current_tracking_axis_mode == "vertical":
            if current_single_axis_output == "primary":
                primary_to_write = self.oscillation_funscript_pos
            else:
                secondary_to_write = self.oscillation_funscript_pos
        elif current_tracking_axis_mode == "horizontal":
            if current_single_axis_output == "primary":
                primary_to_write = self.oscillation_funscript_secondary_pos
            else:
                secondary_to_write = self.oscillation_funscript_secondary_pos
        
        # Add action to funscript
        if hasattr(self.app, 'funscript'):
            self.app.funscript.add_action(
                timestamp_ms=frame_time_ms, 
                primary_pos=primary_to_write, 
                secondary_pos=secondary_to_write
            )
        
        action_log_list.append({
            "at": frame_time_ms, 
            "pos": primary_to_write, 
            "secondary_pos": secondary_to_write
        })
        
        return action_log_list
    
    def _draw_visualization(self, frame: np.ndarray, block_motions: List[Dict], 
                          active_blocks: List[Dict], is_camera_motion: bool):
        """Draw visualization overlays on the frame."""
        active_block_positions = {b['pos'] for b in active_blocks}
        
        for motion in block_motions:
            r, c = motion['pos']
            x1, y1 = c * self.oscillation_block_size, r * self.oscillation_block_size
            x2, y2 = x1 + self.oscillation_block_size, y1 + self.oscillation_block_size
            
            # Color coding: gray for inactive, orange for camera motion, green for active
            if is_camera_motion:
                color = (0, 165, 255)  # Orange for camera motion
            elif (r, c) in active_block_positions:
                color = (0, 255, 0)    # Green for active blocks
            else:
                color = (100, 100, 100)  # Gray for inactive blocks
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        
        # Add tracking indicator
        self._draw_tracking_indicator(frame)
    
    def _update_fps(self):
        """Update FPS calculation using high-performance delta time method."""
        current_time_sec = time.time()
        if self._fps_last_time > 0:
            delta_time = current_time_sec - self._fps_last_time
            if delta_time > 0.001:  # Avoid division by zero
                self.current_fps = 1.0 / delta_time
        self._fps_last_time = current_time_sec