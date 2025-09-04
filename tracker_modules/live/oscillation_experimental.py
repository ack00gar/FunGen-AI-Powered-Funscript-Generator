#!/usr/bin/env python3
"""
Original Experimental Oscillation Detector - Advanced oscillation tracking algorithm.

This tracker implements the original experimental oscillation detection algorithm (V9)
with advanced filtering, global motion cancellation, VR-specific focus, and adaptive
motion logic for precise timing and superior oscillation detection.

Author: Migrated from experimental codebase  
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
    from ..helpers.signal_amplifier import SignalAmplifier
except ImportError:
    from tracker_modules.core.base_tracker import BaseTracker, TrackerMetadata, TrackerResult
    from tracker_modules.helpers.signal_amplifier import SignalAmplifier

# Import constants with fallback
try:
    import sys
    import os
    # Add project root to path to find config module
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from config import constants
except ImportError:
    # Fallback constants
    class constants:
        DEFAULT_OSCILLATION_PROCESSING_TARGET_HEIGHT = 480


class OscillationExperimentalTracker(BaseTracker):
    """
    Original experimental oscillation detector with advanced features.
    
    This tracker excels at:
    - Global motion cancellation for camera pans/shakes
    - VR-specific central third focus for optimal detection
    - Advanced oscillation scoring with frequency/variance analysis
    - Adaptive motion logic (sparse vs dense motion paths)
    - Zero-crossing analysis for precise timing
    - Cell persistence tracking for stable detection
    - Optional decay mechanism with hold duration
    """
    
    def __init__(self):
        super().__init__()
        
        # Core state
        self.prev_gray_oscillation = None
        self.flow_dense_osc = None
        
        # Grid configuration
        self.oscillation_grid_size = 8
        self.oscillation_block_size = 80
        
        # Oscillation detection state
        self.oscillation_history = {}
        self.oscillation_history_max_len = 40
        self.oscillation_cell_persistence = {}
        self.OSCILLATION_PERSISTENCE_FRAMES = 5
        
        # Position tracking
        self.oscillation_last_known_pos = 50.0
        self.oscillation_last_known_secondary_pos = 50.0
        self.oscillation_funscript_pos = 50
        self.oscillation_funscript_secondary_pos = 50
        self.oscillation_last_active_time = 0
        
        # EMA smoothing and sensitivity
        self.oscillation_ema_alpha = 0.3
        self.oscillation_sensitivity = 1.0
        
        # Buffer management for optimization
        self._gray_roi_buffer = None
        self._gray_full_buffer = None
        self._prev_gray_osc_buffer = None
        
        # Settings
        self.show_masks = True
        self.show_grid_blocks = False
        
        # Performance tracking
        self.current_fps = 30.0
        self._fps_update_counter = 0
        self._fps_last_time = time.time()
        self._osc_instr_last_log_sec = 0.0
        
        # DisFlow settings (for performance logging)
        self.dis_flow_preset = "medium"
        self.dis_finest_scale = 1
    
    @property
    def metadata(self) -> TrackerMetadata:
        """Return metadata describing this tracker."""
        return TrackerMetadata(
            name="oscillation_experimental",
            display_name="Live Oscillation Detector (Experimental)",
            description="Advanced oscillation tracker with global motion cancellation and VR focus",
            category="live",
            version="1.0.0",
            author="Experimental Codebase",
            tags=["oscillation", "optical-flow", "experimental", "vr", "advanced"],
            requires_roi=False,
            supports_dual_axis=True
        )
    
    def initialize(self, app_instance, **kwargs) -> bool:
        """Initialize the experimental oscillation detector."""
        try:
            self.app = app_instance
            
            # Load settings
            if hasattr(app_instance, 'app_settings'):
                settings = app_instance.app_settings
                
                self.oscillation_grid_size = settings.get('oscillation_grid_size', 8)
                self.oscillation_ema_alpha = settings.get('oscillation_ema_alpha', 0.3)
                self.oscillation_sensitivity = settings.get('oscillation_sensitivity', 1.0)
                self.show_masks = settings.get('show_masks', True)
                self.show_grid_blocks = settings.get('show_grid_blocks', False)
                
                self.logger.info(f"Experimental oscillation settings: grid_size={self.oscillation_grid_size}, "
                               f"ema_alpha={self.oscillation_ema_alpha}, sensitivity={self.oscillation_sensitivity}")
            
            # Initialize SignalAmplifier helper
            self.signal_amplifier = SignalAmplifier(
                history_size=120,  # 4 seconds @ 30fps
                enable_live_amp=True,
                smoothing_alpha=self.oscillation_ema_alpha,
                logger=self.logger
            )
            
            # Calculate block size based on grid
            if hasattr(app_instance, 'get_video_dimensions'):
                width, height = app_instance.get_video_dimensions()
                if width and height:
                    self.oscillation_block_size = min(width, height) // self.oscillation_grid_size
            
            if self.oscillation_block_size <= 0:
                self.oscillation_block_size = 80  # Default fallback
            
            # Initialize optical flow - use DIS with ultrafast preset for better performance
            try:
                self.flow_dense_osc = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
                self.logger.info("DIS optical flow initialized (ultrafast preset) for experimental oscillation")
            except AttributeError:
                try:
                    # Fallback to medium preset if ultrafast not available
                    self.flow_dense_osc = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
                    self.logger.info("DIS optical flow initialized (medium preset) for experimental oscillation")
                except AttributeError:
                    self.logger.error("No DIS optical flow implementation available")
                    return False
            
            # Reset state
            self.prev_gray_oscillation = None
            self.oscillation_history.clear()
            self.oscillation_cell_persistence.clear()
            self.oscillation_last_known_pos = 50.0
            self.oscillation_last_known_secondary_pos = 50.0
            self.oscillation_last_active_time = 0
            
            # Clear buffers
            self._gray_roi_buffer = None
            self._gray_full_buffer = None
            self._prev_gray_osc_buffer = None
            
            self._initialized = True
            self.logger.info("Experimental oscillation detector initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray, frame_time_ms: int, 
                     frame_index: Optional[int] = None) -> TrackerResult:
        """
        Process a frame using experimental oscillation detection algorithm.
        
        This implementation includes:
        - Global motion cancellation
        - VR-specific central third focus
        - Advanced oscillation scoring
        - Adaptive motion logic (sparse/dense paths)
        - Cell persistence tracking
        """
        try:
            self._update_fps()
            
            # Get target processing height
            try:
                target_height = getattr(self.app.app_settings, 'get', lambda k, d=None: d)(
                    'oscillation_processing_target_height', 
                    constants.DEFAULT_OSCILLATION_PROCESSING_TARGET_HEIGHT
                )
            except Exception:
                target_height = constants.DEFAULT_OSCILLATION_PROCESSING_TARGET_HEIGHT
            
            if frame is None or frame.size == 0:
                return TrackerResult(
                    processed_frame=frame,
                    action_log=None,
                    debug_info={'error': 'Invalid frame'},
                    status_message="Error: Invalid frame"
                )
            
            # Frame preprocessing and ROI handling
            processed_frame, current_gray, use_oscillation_area, ax, ay, aw, ah = self._preprocess_frame_and_roi(frame, target_height)
            
            # Initialize on first frame
            if self.prev_gray_oscillation is None or self.prev_gray_oscillation.shape != current_gray.shape:
                self.prev_gray_oscillation = current_gray.copy()
                return TrackerResult(
                    processed_frame=processed_frame,
                    action_log=None,
                    debug_info={'status': 'initializing'},
                    status_message="Initializing experimental oscillation detector..."
                )
            
            if not self.flow_dense_osc:
                return TrackerResult(
                    processed_frame=processed_frame,
                    action_log=None,
                    debug_info={'error': 'No optical flow available'},
                    status_message="Error: Dense optical flow not available"
                )
            
            # Calculate grid parameters
            grid_params = self._calculate_grid_parameters(current_gray.shape)
            
            # Draw optional grid overlay
            if self.show_grid_blocks:
                self._draw_grid_overlay(processed_frame, grid_params, use_oscillation_area, ax, ay, aw, ah)
            
            # Calculate optical flow with global motion cancellation
            flow, global_dx, global_dy = self._calculate_flow_with_global_motion(current_gray)
            
            if flow is None:
                self.prev_gray_oscillation = current_gray.copy()
                return TrackerResult(
                    processed_frame=processed_frame,
                    action_log=None,
                    debug_info={'error': 'Flow calculation failed'},
                    status_message="Error: Optical flow calculation failed"
                )
            
            # Identify active cells with VR focus
            persistent_active_cells = self._identify_active_cells_with_vr_focus(
                current_gray, use_oscillation_area, grid_params
            )
            
            # Analyze localized motion in active cells
            active_blocks = self._analyze_localized_motion(
                flow, global_dx, global_dy, persistent_active_cells, grid_params
            )
            
            # Apply adaptive motion calculation
            final_dy, final_dx = self._apply_adaptive_motion_logic(active_blocks)
            
            # Enhanced signal processing with both simple and dynamic modes
            self._apply_enhanced_signal_processing(final_dy, final_dx, frame_time_ms)
            
            # Generate funscript actions if tracking is active
            action_log_list = []
            if self.tracking_active:
                action_log_list = self._generate_actions(frame_time_ms)
            
            # Apply visualization
            if self.show_masks:
                self._draw_visualization(processed_frame, persistent_active_cells, active_blocks, ax, ay, grid_params)
            
            # Update previous frame buffer
            self._update_previous_frame_buffer(current_gray)
            
            # Prepare debug info
            debug_info = {
                'primary_position': self.oscillation_funscript_pos,
                'secondary_position': self.oscillation_funscript_secondary_pos,
                'active_blocks': len(active_blocks),
                'persistent_cells': len(persistent_active_cells),
                'global_motion': (float(global_dx), float(global_dy)),
                'tracking_active': self.tracking_active,
                'vr_focus_active': self._is_vr_video() and not use_oscillation_area
            }
            
            # Performance logging
            if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                self._log_performance_info(current_gray, grid_params, use_oscillation_area)
            
            status_msg = f"Experimental | Pos: {self.oscillation_funscript_pos} | Cells: {len(persistent_active_cells)}"
            if abs(global_dx) > 1 or abs(global_dy) > 1:
                status_msg += f" | Global motion: ({global_dx:.1f}, {global_dy:.1f})"
            
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
        """Start experimental oscillation tracking."""
        if not self._initialized:
            self.logger.error("Tracker not initialized")
            return False
        
        self.tracking_active = True
        self.oscillation_last_active_time = 0
        
        # Reset signal amplifier for new tracking session
        if hasattr(self, 'signal_amplifier'):
            self.signal_amplifier.reset()
            
        self.logger.info("Experimental oscillation tracking started")
        return True
    
    def stop_tracking(self) -> bool:
        """Stop experimental oscillation tracking."""
        self.tracking_active = False
        self.logger.info("Experimental oscillation tracking stopped")
        return True
    
    def validate_settings(self, settings: Dict[str, Any]) -> bool:
        """Validate experimental oscillation settings."""
        try:
            grid_size = settings.get('oscillation_grid_size', self.oscillation_grid_size)
            if not isinstance(grid_size, int) or grid_size < 4 or grid_size > 16:
                self.logger.error("Grid size must be between 4 and 16")
                return False
            
            ema_alpha = settings.get('oscillation_ema_alpha', self.oscillation_ema_alpha)
            if not isinstance(ema_alpha, (int, float)) or not (0 < ema_alpha <= 1):
                self.logger.error("EMA alpha must be between 0 and 1")
                return False
            
            sensitivity = settings.get('oscillation_sensitivity', self.oscillation_sensitivity)
            if not isinstance(sensitivity, (int, float)) or sensitivity <= 0:
                self.logger.error("Sensitivity must be positive")
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
            'persistent_cells': len(self.oscillation_cell_persistence),
            'fps': round(self.current_fps, 1),
            'current_primary_pos': self.oscillation_funscript_pos,
            'current_secondary_pos': self.oscillation_funscript_secondary_pos,
            'vr_detection': self._is_vr_video()
        }
        
        base_status.update(custom_status)
        return base_status
    
    def cleanup(self):
        """Clean up resources."""
        self.prev_gray_oscillation = None
        self.flow_dense_osc = None
        self.oscillation_history.clear()
        self.oscillation_cell_persistence.clear()
        
        # Clear buffers
        self._gray_roi_buffer = None
        self._gray_full_buffer = None
        self._prev_gray_osc_buffer = None
        
        # self.logger.info("Experimental oscillation detector cleaned up")
    
    # Private helper methods
    
    def _update_fps(self):
        """Update FPS tracking."""
        self._fps_update_counter += 1
        if self._fps_update_counter >= 30:  # Update every 30 frames
            current_time = time.time()
            if self._fps_last_time > 0:
                self.current_fps = 30.0 / (current_time - self._fps_last_time)
            self._fps_last_time = current_time
            self._fps_update_counter = 0
    
    def _preprocess_frame_and_roi(self, frame: np.ndarray, target_height: int) -> Tuple:
        """Preprocess frame and handle ROI/full-frame processing."""
        # Frame resizing if needed
        src_h, src_w = frame.shape[:2]
        if target_height and src_h > target_height:
            scale = float(target_height) / float(src_h)
            new_w = max(1, int(round(src_w * scale)))
            processed_input = cv2.resize(frame, (new_w, target_height), interpolation=cv2.INTER_AREA)
        else:
            processed_input = frame
        
        # Check for oscillation area (ROI)
        oscillation_area_fixed = getattr(self.app, 'oscillation_area_fixed', None)
        use_oscillation_area = oscillation_area_fixed is not None
        
        if use_oscillation_area:
            ax, ay, aw, ah = oscillation_area_fixed
            processed_frame = self._preprocess_frame(frame)
            processed_frame_area = processed_frame[ay:ay+ah, ax:ax+aw]
            
            # Reuse/allocate ROI gray buffer
            if (self._gray_roi_buffer is None or 
                self._gray_roi_buffer.shape[:2] != (processed_frame_area.shape[0], processed_frame_area.shape[1])):
                self._gray_roi_buffer = np.empty((processed_frame_area.shape[0], processed_frame_area.shape[1]), dtype=np.uint8)
            
            cv2.cvtColor(processed_frame_area, cv2.COLOR_BGR2GRAY, dst=self._gray_roi_buffer)
            current_gray = self._gray_roi_buffer
        else:
            processed_frame = self._preprocess_frame(frame)
            target_h, target_w = processed_frame.shape[0], processed_frame.shape[1]
            
            if (self._gray_full_buffer is None or 
                self._gray_full_buffer.shape[:2] != (target_h, target_w)):
                self._gray_full_buffer = np.empty((target_h, target_w), dtype=np.uint8)
            
            cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY, dst=self._gray_full_buffer)
            current_gray = self._gray_full_buffer
            ax, ay = 0, 0
            aw, ah = processed_frame.shape[1], processed_frame.shape[0]
        
        return processed_frame, current_gray, use_oscillation_area, ax, ay, aw, ah
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Basic frame preprocessing."""
        # Placeholder for frame preprocessing if needed
        return frame.copy()
    
    def _calculate_grid_parameters(self, gray_shape: Tuple[int, int]) -> Dict:
        """Calculate grid parameters for current frame."""
        img_h, img_w = gray_shape[:2]
        grid_size = max(1, int(self.oscillation_grid_size))
        local_block_size = max(8, min(img_h // grid_size, img_w // grid_size))
        
        if local_block_size <= 0:
            local_block_size = 8
        
        num_rows = max(1, img_h // local_block_size)
        num_cols = max(1, img_w // local_block_size)
        min_cell_activation_pixels = (local_block_size * local_block_size) * 0.05
        
        return {
            'local_block_size': local_block_size,
            'num_rows': num_rows,
            'num_cols': num_cols,
            'min_cell_activation_pixels': min_cell_activation_pixels
        }
    
    def _draw_grid_overlay(self, processed_frame: np.ndarray, grid_params: Dict, 
                          use_oscillation_area: bool, ax: int, ay: int, aw: int, ah: int):
        """Draw optional static grid blocks overlay."""
        local_block_size = grid_params['local_block_size']
        num_cols = grid_params['num_cols']
        
        start_x, start_y = ax, ay
        end_x, end_y = ax + aw, ay + ah
        
        # If VR central-third focus is active (no ROI), align the grid to that region
        if not use_oscillation_area and self._is_vr_video():
            vr_central_third_start_local = num_cols // 3
            vr_central_third_end_local = 2 * num_cols // 3
            start_x = ax + (vr_central_third_start_local * local_block_size)
            end_x = ax + (vr_central_third_end_local * local_block_size)
        
        gray_color = (100, 100, 100)
        for y in range(start_y, end_y, local_block_size):
            row_h = min(local_block_size, end_y - y)
            if row_h <= 0:
                break
            for x in range(start_x, end_x, local_block_size):
                col_w = min(local_block_size, end_x - x)
                if col_w <= 0:
                    break
                cv2.rectangle(processed_frame, (x, y), (x + col_w, y + row_h), gray_color, 1)
    
    def _calculate_flow_with_global_motion(self, current_gray: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Calculate optical flow and detect global motion."""
        flow = self.flow_dense_osc.calc(self.prev_gray_oscillation, current_gray, None)
        
        if flow is None:
            return None, 0.0, 0.0
        
        # Calculate Global Motion to cancel out camera pans/shakes
        global_dx = np.median(flow[..., 0])
        global_dy = np.median(flow[..., 1])
        
        return flow, global_dx, global_dy
    
    def _identify_active_cells_with_vr_focus(self, current_gray: np.ndarray, 
                                           use_oscillation_area: bool, grid_params: Dict) -> List[Tuple[int, int]]:
        """Identify active cells with VR central-third focus."""
        min_motion_threshold = 15
        frame_diff = cv2.absdiff(current_gray, self.prev_gray_oscillation)
        _, motion_mask = cv2.threshold(frame_diff, min_motion_threshold, 255, cv2.THRESH_BINARY)
        
        # Check if the video is VR to apply the focus rule
        is_vr = self._is_vr_video()
        
        # Apply VR central-third focus only for full-frame scans
        apply_vr_central_focus = is_vr and not use_oscillation_area
        
        # Calculate VR focus bounds
        eff_cols = max(1, min(self.oscillation_grid_size, current_gray.shape[1] // self.oscillation_block_size))
        vr_central_third_start = eff_cols // 3
        vr_central_third_end = 2 * eff_cols // 3
        
        # Find newly active cells
        newly_active_cells = set()
        num_rows = grid_params['num_rows']
        num_cols = grid_params['num_cols']
        local_block_size = grid_params['local_block_size']
        min_cell_activation_pixels = grid_params['min_cell_activation_pixels']
        
        for r in range(num_rows):
            for c in range(num_cols):
                # If VR, skip cells outside the central third unless an oscillation ROI is set
                if is_vr and (not use_oscillation_area) and (c < vr_central_third_start or c > vr_central_third_end):
                    continue
                
                y_start, x_start = r * local_block_size, c * local_block_size
                mask_roi = motion_mask[y_start:y_start + local_block_size, x_start:x_start + local_block_size]
                
                if mask_roi.size == 0:
                    continue
                
                if cv2.countNonZero(mask_roi) > min_cell_activation_pixels:
                    newly_active_cells.add((r, c))
        
        # Update persistence counters
        for cell_pos in newly_active_cells:
            self.oscillation_cell_persistence[cell_pos] = self.OSCILLATION_PERSISTENCE_FRAMES
        
        # Remove expired cells
        expired_cells = [pos for pos, timer in self.oscillation_cell_persistence.items() if timer <= 1]
        for cell_pos in expired_cells:
            del self.oscillation_cell_persistence[cell_pos]
        
        # Decrement all timers
        for cell_pos in self.oscillation_cell_persistence:
            self.oscillation_cell_persistence[cell_pos] -= 1
        
        return list(self.oscillation_cell_persistence.keys())
    
    def _analyze_localized_motion(self, flow: np.ndarray, global_dx: float, global_dy: float,
                                 persistent_active_cells: List[Tuple[int, int]], grid_params: Dict) -> List[Dict]:
        """Analyze localized motion in active cells."""
        local_block_size = grid_params['local_block_size']
        block_motions = []
        
        for r, c in persistent_active_cells:
            y_start = r * local_block_size
            x_start = c * local_block_size
            
            # Sample the pre-computed flow field for this cell's ROI
            flow_patch = flow[y_start:y_start + local_block_size, x_start:x_start + local_block_size]
            
            if flow_patch.size > 0:
                # Subtract global motion to get true local motion
                local_dx = np.median(flow_patch[..., 0]) - global_dx
                local_dy = np.median(flow_patch[..., 1]) - global_dy
                
                mag = np.sqrt(local_dx ** 2 + local_dy ** 2)
                block_motions.append({'dx': local_dx, 'dy': local_dy, 'mag': mag, 'pos': (r, c)})
                
                # Update oscillation history
                if (r, c) not in self.oscillation_history:
                    self.oscillation_history[(r, c)] = deque(maxlen=self.oscillation_history_max_len)
                self.oscillation_history[(r, c)].append({'dx': local_dx, 'dy': local_dy, 'mag': mag})
        
        # Advanced oscillation scoring
        active_blocks = []
        if block_motions:
            candidate_blocks = []
            for motion in block_motions:
                history = self.oscillation_history.get(motion['pos'])
                
                # Only consider blocks with some history and current motion
                if history and len(history) > 10 and motion['mag'] > 0.2:
                    # Get stats from history
                    recent_dy = [h['dy'] for h in history]
                    mean_mag = np.mean([h['mag'] for h in history])
                    
                    # Calculate frequency score using zero-crossings
                    zero_crossings = np.sum(np.diff(np.sign(recent_dy)) != 0)
                    frequency_score = (zero_crossings / len(recent_dy)) * 10.0
                    
                    # Calculate variance score
                    variance_score = np.std(recent_dy)
                    
                    # Combine into final oscillation score
                    oscillation_score = mean_mag * (1 + frequency_score) * (1 + variance_score)
                    
                    if oscillation_score > 0.5:  # Filter out low-scoring blocks
                        candidate_blocks.append({**motion, 'score': oscillation_score})
            
            if candidate_blocks:
                max_score = max(b['score'] for b in candidate_blocks)
                # Take any block that has at least 40% of the max score
                active_blocks = [b for b in candidate_blocks if b['score'] >= max_score * 0.4]
        
        return active_blocks
    
    def _apply_adaptive_motion_logic(self, active_blocks: List[Dict]) -> Tuple[float, float]:
        """Apply adaptive motion logic (sparse vs dense paths)."""
        SPARSITY_THRESHOLD = 2  # Threshold for sparse vs dense motion
        
        final_dy, final_dx = 0.0, 0.0
        
        if 0 < len(active_blocks) <= SPARSITY_THRESHOLD:
            # Sparse Motion Path ("Follow the Leader")
            # Ideal for localized action like handjobs/blowjobs
            if hasattr(self, 'logger'):
                self.logger.debug(f"Sparse motion detected ({len(active_blocks)} blocks). Following the leader.")
            
            leader_block = max(active_blocks, key=lambda b: b['mag'])
            final_dy = leader_block['dy']
            final_dx = leader_block['dx']
            
        elif len(active_blocks) > SPARSITY_THRESHOLD:
            # Dense Motion Path (Weighted Average)
            # Ideal for full-body motion
            if hasattr(self, 'logger'):
                self.logger.debug(f"Dense motion detected ({len(active_blocks)} blocks). Using weighted average.")
            
            total_weight = sum(b['score'] for b in active_blocks)
            if total_weight > 0:
                final_dy = sum(b['dy'] * b['score'] for b in active_blocks) / total_weight
                final_dx = sum(b['dx'] * b['score'] for b in active_blocks) / total_weight
        
        return final_dy, final_dx
    
    def _apply_enhanced_signal_processing(self, final_dy: float, final_dx: float, frame_time_ms: int):
        """Apply enhanced signal processing with both simple and dynamic modes."""
        # Get settings
        use_simple_amplification = getattr(self.app.app_settings, 'get', lambda k, d: d)(
            'oscillation_use_simple_amplification', False
        ) if self.app else False
        
        enable_decay = getattr(self.app.app_settings, 'get', lambda k, d: d)(
            'oscillation_enable_decay', True
        ) if self.app else True
        
        if abs(final_dy) > 0.01 or abs(final_dx) > 0.01:
            # Update last active time for decay mechanism
            self.oscillation_last_active_time = frame_time_ms
            
            # Use SignalAmplifier for enhanced signal processing
            raw_primary_pos = 50  # Start from center
            raw_secondary_pos = 50
            
            # Apply enhanced signal mastering using helper module
            enhanced_primary, enhanced_secondary = self.signal_amplifier.enhance_signal(
                raw_primary_pos, raw_secondary_pos,
                final_dy, final_dx,
                sensitivity=self.oscillation_sensitivity * 10,  # Convert to standard 0-20 scale
                apply_smoothing=False  # We'll apply our own EMA smoothing
            )
            
            # Apply EMA smoothing as before
            alpha = self.oscillation_ema_alpha
            self.oscillation_last_known_pos = (self.oscillation_last_known_pos * (1 - alpha) + 
                                             enhanced_primary * alpha)
            self.oscillation_last_known_secondary_pos = (self.oscillation_last_known_secondary_pos * (1 - alpha) + 
                                                       enhanced_secondary * alpha)
            
            # Clip to valid range
            self.oscillation_last_known_pos = np.clip(self.oscillation_last_known_pos, 0, 100)
            self.oscillation_last_known_secondary_pos = np.clip(self.oscillation_last_known_secondary_pos, 0, 100)
        
        elif enable_decay:
            # Legacy-style decay mechanism when no motion is detected
            hold_duration_ms = getattr(self.app.app_settings, 'get', lambda k, d: d)(
                'oscillation_hold_duration_ms', 250
            ) if self.app else 250
            
            decay_factor = getattr(self.app.app_settings, 'get', lambda k, d: d)(
                'oscillation_decay_factor', 0.95
            ) if self.app else 0.95
            
            time_since_last_active = frame_time_ms - self.oscillation_last_active_time
            if time_since_last_active > hold_duration_ms:
                # Decay towards center after hold duration expires
                self.oscillation_last_known_pos = (self.oscillation_last_known_pos * decay_factor + 
                                                 50 * (1 - decay_factor))
                self.oscillation_last_known_secondary_pos = (self.oscillation_last_known_secondary_pos * decay_factor + 
                                                           50 * (1 - decay_factor))
        
        # Update final funscript positions
        self.oscillation_funscript_pos = int(round(self.oscillation_last_known_pos))
        self.oscillation_funscript_secondary_pos = int(round(self.oscillation_last_known_secondary_pos))
    
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
    
    def _draw_visualization(self, processed_frame: np.ndarray, persistent_active_cells: List[Tuple[int, int]], 
                          active_blocks: List[Dict], ax: int, ay: int, grid_params: Dict):
        """Draw visualization overlays on the frame."""
        local_block_size = grid_params['local_block_size']
        active_block_positions = {b['pos'] for b in active_blocks}
        
        for r, c in persistent_active_cells:
            x1, y1 = c * local_block_size + ax, r * local_block_size + ay
            color = (0, 255, 0) if (r, c) in active_block_positions else (180, 100, 100)
            cv2.rectangle(processed_frame, (x1, y1), (x1 + local_block_size, y1 + local_block_size), color, 1)
        
        # Draw position indicator
        pos_text = f"Exp: {self.oscillation_funscript_pos}, {self.oscillation_funscript_secondary_pos}"
        cv2.putText(processed_frame, pos_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if self.tracking_active:
            cv2.putText(processed_frame, "EXPERIMENTAL TRACKING", (10, processed_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    def _update_previous_frame_buffer(self, current_gray: np.ndarray):
        """Update previous frame buffer for next iteration."""
        if self._prev_gray_osc_buffer is None or self._prev_gray_osc_buffer.shape != current_gray.shape:
            self._prev_gray_osc_buffer = np.empty_like(current_gray)
        np.copyto(self._prev_gray_osc_buffer, current_gray)
        self.prev_gray_oscillation = self._prev_gray_osc_buffer
    
    def _log_performance_info(self, current_gray: np.ndarray, grid_params: Dict, use_oscillation_area: bool):
        """Log performance information for debugging."""
        try:
            now_sec = time.time()
            last = getattr(self, '_osc_instr_last_log_sec', 0.0)
            if now_sec - last >= 1.0:
                self._osc_instr_last_log_sec = now_sec
                cur_rows = grid_params['num_rows']
                cur_cols = grid_params['num_cols']
                self.logger.debug(
                    f"OSC perf: area={use_oscillation_area} img={current_gray.shape} grid={cur_rows}x{cur_cols} "
                    f"preset={self.dis_flow_preset} finest={self.dis_finest_scale} "
                    f"active_cells={len(self.oscillation_cell_persistence)} fps={self.current_fps:.1f}"
                )
        except Exception:
            pass
    
    def _is_vr_video(self) -> bool:
        """Detect if this is a VR video based on aspect ratio."""
        # Simple VR detection based on common VR aspect ratios
        try:
            if hasattr(self.app, 'get_video_dimensions'):
                width, height = self.app.get_video_dimensions()
                if width and height:
                    aspect_ratio = width / height
                    # Common VR aspect ratios: 2:1 for 360°, 16:9 for 180°, etc.
                    return aspect_ratio >= 1.8  # Threshold for VR detection
        except Exception:
            pass
        return False