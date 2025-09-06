"""
Oscillation Detector Experimental 2 - Hybrid Approach

This tracker combines the best aspects of the legacy and experimental oscillation
detectors, providing both timing precision and signal strength in a single
implementation.

Features:
- Zero-crossing analysis for precise peak/valley detection (from Experimental)
- Adaptive motion logic with sparse "Follow the Leader" mode (from Experimental)  
- Global motion cancellation to reduce camera shake (from Experimental)
- Live dynamic amplification with percentile normalization (from Legacy)
- Amplitude-aware scaling with natural response (from Legacy)
- Cohesion analysis for spatial consistency (from Legacy)
- Hybrid frequency scoring combining both approaches
"""

import logging
import time
import numpy as np
import cv2
from collections import deque
from typing import Optional, Dict, List, Set, Tuple, Any

try:
    from ..core.base_tracker import BaseTracker, TrackerMetadata, TrackerResult
    from ..helpers.signal_amplifier import SignalAmplifier
except ImportError:
    # Fallback for direct execution
    from tracker_modules.core.base_tracker import BaseTracker, TrackerMetadata, TrackerResult
    from tracker_modules.helpers.signal_amplifier import SignalAmplifier
import config.constants as constants


class OscillationExperimental2Tracker(BaseTracker):
    """
    Hybrid oscillation detector that combines experimental timing precision
    with legacy amplification and signal conditioning.
    """
    
    @property
    def metadata(self) -> TrackerMetadata:
        return TrackerMetadata(
            name="oscillation_experimental_2",
            display_name="Live Oscillation Detector (Experimental 2)",
            description="Hybrid approach combining experimental timing precision with legacy amplification and signal conditioning",
            category="live",
            version="1.0.0",
            author="VR Funscript AI Generator",
            tags=["oscillation", "hybrid", "live", "experimental", "amplitude", "timing"],
            requires_roi=False,  # ROI is optional
            supports_dual_axis=True
        )
    
    def initialize(self, app_instance, **kwargs) -> bool:
        """Initialize the hybrid oscillation detector."""
        try:
            self.app = app_instance
            self._initialized = False
            
            # Core tracking state
            self.tracking_active = False
            self.current_fps = 30.0
            
            # Initialize funscript connection
            # First check if funscript is provided through compatibility attributes (from bridge)
            if hasattr(self, 'funscript') and self.funscript:
                # Already have funscript from bridge
                pass
            elif hasattr(self.app, 'funscript') and self.app.funscript:
                self.funscript = self.app.funscript
            else:
                # Create our own funscript instance if not provided
                from funscript.dual_axis_funscript import DualAxisFunscript
                self.funscript = DualAxisFunscript(logger=self.logger)
                self.logger.info("Created local funscript instance for Oscillation Experimental 2")
            
            # Visual settings
            self.show_masks = kwargs.get('show_masks', True)
            self.show_roi = kwargs.get('show_roi', True)
            
            # Oscillation detection parameters
            self.oscillation_grid_size = 10
            self.oscillation_block_size = constants.YOLO_INPUT_SIZE // self.oscillation_grid_size
            self.oscillation_sensitivity = kwargs.get('oscillation_sensitivity', 1.0)
            
            # Motion history and persistence
            self.oscillation_history: Dict[Tuple[int, int], deque] = {}
            self.oscillation_history_max_len = 60
            self.oscillation_history_seconds = 2.0
            self.oscillation_cell_persistence: Dict[Tuple[int, int], int] = {}
            self.OSCILLATION_PERSISTENCE_FRAMES = 5
            
            # Position tracking and smoothing
            self.oscillation_last_known_pos = 50.0
            self.oscillation_last_known_secondary_pos = 50.0
            self.oscillation_last_active_time = 0
            self.oscillation_ema_alpha = 0.3
            self.oscillation_funscript_pos = 50
            self.oscillation_funscript_secondary_pos = 50
            
            # Enhanced signal mastering using helper module
            live_amp = self.app.app_settings.get("live_oscillation_dynamic_amp_enabled", True) if self.app else True
            self.signal_amplifier = SignalAmplifier(
                history_size=120,  # 4 seconds @ 30fps
                enable_live_amp=live_amp,
                smoothing_alpha=self.oscillation_ema_alpha,  # Use same EMA alpha
                logger=self.logger
            )
            
            # Optical flow
            self.flow_dense_osc = None
            self.prev_gray_oscillation = None
            
            # Region of interest (optional)
            self.oscillation_area_fixed: Optional[Tuple[int, int, int, int]] = None
            
            # Buffer management
            self._gray_roi_buffer = None
            self._gray_full_buffer = None
            self._prev_gray_osc_buffer = None
            
            # Performance instrumentation
            self._osc_instr_last_log_sec = 0.0
            
            self._initialize_optical_flow()
            self._initialized = True
            
            self.logger.info("Oscillation Experimental 2 tracker initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Oscillation Experimental 2 tracker: {e}", exc_info=True)
            return False
    
    def _initialize_optical_flow(self):
        """Initialize dense optical flow for oscillation detection."""
        try:
            # Try ultrafast preset first for better performance
            self.flow_dense_osc = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
            self.logger.info("DIS optical flow initialized (ultrafast preset) for Oscillation Experimental 2")
        except AttributeError:
            try:
                # Fallback to medium preset
                self.flow_dense_osc = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
                self.logger.info("DIS optical flow initialized (medium preset) for Oscillation Experimental 2")
            except Exception as e:
                self.logger.error(f"Failed to initialize DIS optical flow: {e}")
                self.flow_dense_osc = None
    
    def process_frame(self, frame: np.ndarray, frame_time_ms: int, 
                     frame_index: Optional[int] = None) -> TrackerResult:
        """Process frame using hybrid oscillation detection."""
        
        processed_frame, action_log = self._process_oscillation_experimental_2(
            frame, frame_time_ms, frame_index
        )
        
        debug_info = {
            'mode': 'experimental_2',
            'last_position': self.oscillation_last_known_pos,
            'last_secondary_position': self.oscillation_last_known_secondary_pos,
            'active_cells': len(self.oscillation_cell_persistence),
            'live_amp_enabled': self.signal_amplifier.live_amp_enabled if hasattr(self, 'signal_amplifier') else False,
            'fps': self.current_fps
        }
        
        return TrackerResult(
            processed_frame=processed_frame,
            action_log=action_log,
            debug_info=debug_info
        )
    
    def _process_oscillation_experimental_2(self, frame: np.ndarray, frame_time_ms: int, 
                                          frame_index: Optional[int] = None) -> Tuple[np.ndarray, Optional[List[Dict]]]:
        """
        [Experimental 2] Hybrid oscillation detector combining experimental timing precision 
        with legacy amplification and signal conditioning.
        """
        self._update_fps()

        try:
            target_height = getattr(self.app.app_settings, 'get', lambda k, d=None: d)('oscillation_processing_target_height', constants.DEFAULT_OSCILLATION_PROCESSING_TARGET_HEIGHT) if self.app else constants.DEFAULT_OSCILLATION_PROCESSING_TARGET_HEIGHT
        except Exception:
            target_height = constants.DEFAULT_OSCILLATION_PROCESSING_TARGET_HEIGHT

        if frame is None or frame.size == 0:
            return frame, None

        src_h, src_w = frame.shape[:2]
        if target_height and src_h > target_height:
            scale = float(target_height) / float(src_h)
            new_w = max(1, int(round(src_w * scale)))
            processed_input = cv2.resize(frame, (new_w, target_height), interpolation=cv2.INTER_AREA)
        else:
            processed_input = frame

        # --- Use oscillation area for detection if set ---
        use_oscillation_area = self.oscillation_area_fixed is not None
        if use_oscillation_area:
            ax, ay, aw, ah = self.oscillation_area_fixed
            processed_frame = self._preprocess_frame(frame)
            processed_frame_area = processed_frame[ay:ay+ah, ax:ax+aw]
            if self._gray_roi_buffer is None or self._gray_roi_buffer.shape[:2] != (processed_frame_area.shape[0], processed_frame_area.shape[1]):
                self._gray_roi_buffer = np.empty((processed_frame_area.shape[0], processed_frame_area.shape[1]), dtype=np.uint8)
            cv2.cvtColor(processed_frame_area, cv2.COLOR_BGR2GRAY, dst=self._gray_roi_buffer)
            current_gray = self._gray_roi_buffer
        else:
            processed_frame = self._preprocess_frame(frame)
            target_h, target_w = processed_frame.shape[0], processed_frame.shape[1]
            if self._gray_full_buffer is None or self._gray_full_buffer.shape[:2] != (target_h, target_w):
                self._gray_full_buffer = np.empty((target_h, target_w), dtype=np.uint8)
            cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY, dst=self._gray_full_buffer)
            current_gray = self._gray_full_buffer
            processed_frame_area = processed_frame
            ax, ay = 0, 0
            aw, ah = processed_frame.shape[1], processed_frame.shape[0]

        action_log_list = []
        active_blocks = set()

        # Compute dynamic grid based on current analysis image size
        img_h, img_w = current_gray.shape[:2]
        # Handle both tuple (8, 8) and integer 8 formats for grid_size
        if hasattr(self, 'oscillation_grid_size'):
            if isinstance(self.oscillation_grid_size, (tuple, list)):
                grid_size = max(1, int(self.oscillation_grid_size[0]))  # Use first element of tuple
            else:
                grid_size = max(1, int(self.oscillation_grid_size))
        else:
            grid_size = 8  # Default fallback
        local_block_size = max(8, min(img_h // grid_size, img_w // grid_size))
        if local_block_size <= 0:
            local_block_size = 8
        num_rows = max(1, img_h // local_block_size)
        num_cols = max(1, img_w // local_block_size)
        min_cell_activation_pixels = (local_block_size * local_block_size) * 0.05

        # --- Detection logic ---
        if self.prev_gray_oscillation is None or self.prev_gray_oscillation.shape != current_gray.shape:
            self.prev_gray_oscillation = current_gray.copy()
            return processed_frame, None

        if not hasattr(self, 'flow_dense_osc') or not self.flow_dense_osc:
            self.logger.warning("Dense optical flow not available for oscillation detection.")
            return processed_frame, None

        # --- Step 1: Calculate Global Optical Flow & Global Motion Vector (FROM EXPERIMENTAL) ---
        flow = self.flow_dense_osc.calc(self.prev_gray_oscillation, current_gray, None)
        if flow is None:
            self.prev_gray_oscillation = current_gray.copy()
            return processed_frame, None

        # Calculate Global Motion to cancel out camera pans/shakes
        global_dx = np.median(flow[..., 0])
        global_dy = np.median(flow[..., 1])

        # --- Step 2: Identify Active Cells & Apply VR Focus ---
        min_motion_threshold = 15
        frame_diff = cv2.absdiff(current_gray, self.prev_gray_oscillation)
        _, motion_mask = cv2.threshold(frame_diff, min_motion_threshold, 255, cv2.THRESH_BINARY)

        is_vr = self._is_vr_video()
        apply_vr_central_focus = is_vr and not use_oscillation_area
        eff_cols = max(1, min(self.oscillation_grid_size, current_gray.shape[1] // self.oscillation_block_size))
        vr_central_third_start = eff_cols // 3
        vr_central_third_end = 2 * eff_cols // 3

        newly_active_cells = set()
        for r in range(num_rows):
            for c in range(num_cols):
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

        expired_cells = [pos for pos, timer in self.oscillation_cell_persistence.items() if timer <= 1]
        for cell_pos in expired_cells:
            del self.oscillation_cell_persistence[cell_pos]

        for cell_pos in self.oscillation_cell_persistence:
            self.oscillation_cell_persistence[cell_pos] -= 1

        persistent_active_cells = list(self.oscillation_cell_persistence.keys())

        # --- Step 3: Analyze Localized Motion in Active Cells ---
        block_motions = []
        for r, c in persistent_active_cells:
            y_start = r * local_block_size
            x_start = c * local_block_size

            flow_patch = flow[y_start:y_start + local_block_size, x_start:x_start + local_block_size]

            if flow_patch.size > 0:
                # Subtract global motion to get true local motion (FROM EXPERIMENTAL)
                local_dx = np.median(flow_patch[..., 0]) - global_dx
                local_dy = np.median(flow_patch[..., 1]) - global_dy

                mag = np.sqrt(local_dx ** 2 + local_dy ** 2)
                block_motions.append({'dx': local_dx, 'dy': local_dy, 'mag': mag, 'pos': (r, c)})
                if (r, c) not in self.oscillation_history:
                    self.oscillation_history[(r, c)] = deque(maxlen=self.oscillation_history_max_len)
                self.oscillation_history[(r, c)].append({'dx': local_dx, 'dy': local_dy, 'mag': mag})

        # --- Step 4: HYBRID BLOCK SELECTION (EXPERIMENTAL + LEGACY) ---
        final_dy, final_dx = 0.0, 0.0
        active_blocks_list = []

        if block_motions:
            candidate_blocks = []
            for motion in block_motions:
                history = self.oscillation_history.get(motion['pos'])
                
                # EXPERIMENTAL: Advanced frequency/variance analysis
                if history and len(history) > 10 and motion['mag'] > 0.2:
                    recent_dy = [h['dy'] for h in history]
                    mean_mag = np.mean([h['mag'] for h in history])

                    # Zero crossing analysis for precise timing (FROM EXPERIMENTAL)
                    zero_crossings = np.sum(np.diff(np.sign(recent_dy)) != 0)
                    frequency_score = (zero_crossings / len(recent_dy)) * 10.0

                    # Variance analysis for oscillatory motion detection
                    variance_score = np.std(recent_dy)
                    
                    # Calculate frequency from smoothed data (FROM LEGACY)
                    smoothed_dys = np.convolve(recent_dy, np.ones(5) / 5, mode='valid')
                    if len(smoothed_dys) >= 2:
                        freq = (len(np.where(np.diff(np.sign(smoothed_dys)))[0]) / 2) / self.oscillation_history_seconds
                        
                        # LEGACY: Gaussian frequency weighting centered at 2.5Hz
                        if 0.5 <= freq <= 7.0:
                            freq_weight = np.exp(-((freq - 2.5) ** 2) / (2 * (1.5 ** 2)))
                            
                            # HYBRID SCORING: Combine experimental and legacy approaches
                            experimental_score = mean_mag * (1 + frequency_score) * (1 + variance_score)
                            legacy_score = mean_mag * freq * freq_weight
                            
                            # Weighted combination: favor experimental but boost with legacy
                            hybrid_score = (experimental_score * 0.7) + (legacy_score * 0.3)
                            
                            if hybrid_score > 0.5:
                                candidate_blocks.append({**motion, 'score': hybrid_score, 'freq': freq})

            if candidate_blocks:
                # LEGACY: Cohesion analysis for spatial consistency
                candidate_pos = {b['pos'] for b in candidate_blocks}
                for block in candidate_blocks:
                    r, c = block['pos']
                    cohesion_boost = 1.0
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0: continue
                            if (r + dr, c + dc) in candidate_pos:
                                cohesion_boost += 0.2  # 20% boost for each active neighbor
                    block['score'] *= cohesion_boost

                max_score = max(b['score'] for b in candidate_blocks)
                # Use legacy threshold (60%) for proven stability
                active_blocks_list = [b for b in candidate_blocks if b['score'] > max_score * 0.6]

        # --- Step 5: ADAPTIVE MOTION CALCULATION (FROM EXPERIMENTAL) ---
        SPARSITY_THRESHOLD = 2

        if 0 < len(active_blocks_list) <= SPARSITY_THRESHOLD:
            # Sparse Motion Path ("Follow the Leader")
            leader_block = max(active_blocks_list, key=lambda b: b['mag'])
            final_dy = leader_block['dy']
            final_dx = leader_block['dx']

        elif len(active_blocks_list) > SPARSITY_THRESHOLD:
            # Dense Motion Path (Weighted Average)
            total_weight = sum(b['score'] for b in active_blocks_list)
            if total_weight > 0:
                final_dy = sum(b['dy'] * b['score'] for b in active_blocks_list) / total_weight
                final_dx = sum(b['dx'] * b['score'] for b in active_blocks_list) / total_weight

        # --- Step 6: LEGACY SIGNAL CONDITIONING ---
        if active_blocks_list:
            # Use SignalAmplifier for enhanced signal processing
            # Start with neutral position since we'll use flow to enhance
            raw_primary_pos = 50
            raw_secondary_pos = 50
            
            # Apply enhanced signal mastering using helper module
            final_primary_pos, final_secondary_pos = self.signal_amplifier.enhance_signal(
                raw_primary_pos, raw_secondary_pos, 
                final_dy, final_dx,
                sensitivity=self.oscillation_sensitivity * 10,  # Convert to standard 0-20 scale
                apply_smoothing=False  # We'll apply our own EMA smoothing below
            )
        else:
            # Decay towards center when no motion
            decay_primary = self.oscillation_last_known_pos * 0.95 + 50 * 0.05
            decay_secondary = self.oscillation_last_known_secondary_pos * 0.95 + 50 * 0.05
            final_primary_pos = decay_primary
            final_secondary_pos = decay_secondary

        # Apply EMA smoothing to the final calculated positions
        self.oscillation_last_known_pos = self.oscillation_last_known_pos * (1 - self.oscillation_ema_alpha) + final_primary_pos * self.oscillation_ema_alpha
        self.oscillation_last_known_secondary_pos = self.oscillation_last_known_secondary_pos * (1 - self.oscillation_ema_alpha) + final_secondary_pos * self.oscillation_ema_alpha
        self.oscillation_funscript_pos = int(round(self.oscillation_last_known_pos))
        self.oscillation_funscript_secondary_pos = int(round(self.oscillation_last_known_secondary_pos))

        # --- Step 8: Action Logging ---
        if self.tracking_active:
            current_tracking_axis_mode = self.app.tracking_axis_mode if self.app else "both"
            current_single_axis_output = self.app.single_axis_output_target if self.app else "primary"
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

            self.funscript.add_action(timestamp_ms=frame_time_ms, primary_pos=primary_to_write, secondary_pos=secondary_to_write)
            action_log_list.append({"at": frame_time_ms, "pos": primary_to_write, "secondary_pos": secondary_to_write})

        # --- Step 9: Visualization ---
        if self.show_masks:
            active_block_positions = {b['pos'] for b in active_blocks_list}
            for r,c in list(self.oscillation_cell_persistence.keys()):
                x1, y1 = c * local_block_size + ax, r * local_block_size + ay
                color = (0, 255, 0) if (r, c) in active_block_positions else (180, 100, 100)
                cv2.rectangle(processed_frame, (x1, y1), (x1 + local_block_size, y1 + local_block_size), color, 1)

        # Keep reusable prev gray buffer
        if self._prev_gray_osc_buffer is None or self._prev_gray_osc_buffer.shape != current_gray.shape:
            self._prev_gray_osc_buffer = np.empty_like(current_gray)
        np.copyto(self._prev_gray_osc_buffer, current_gray)
        self.prev_gray_oscillation = self._prev_gray_osc_buffer

        return processed_frame, action_log_list if action_log_list else None
    
    def start_tracking(self) -> bool:
        """Start oscillation tracking."""
        try:
            # Check if properly initialized
            if not self._initialized:
                self.logger.error("Cannot start tracking - tracker not initialized")
                return False
                
            self.tracking_active = True
            self.oscillation_last_active_time = 0
            
            # Reset tracking state (check each exists first)
            if hasattr(self, 'oscillation_history') and self.oscillation_history:
                self.oscillation_history.clear()
            if hasattr(self, 'oscillation_cell_persistence') and self.oscillation_cell_persistence:
                self.oscillation_cell_persistence.clear()
            
            # Reset signal amplifier for new tracking session
            if hasattr(self, 'signal_amplifier'):
                self.signal_amplifier.reset()
                
            # Reset positions
            if hasattr(self, 'oscillation_last_known_pos'):
                self.oscillation_last_known_pos = 50.0
            if hasattr(self, 'oscillation_last_known_secondary_pos'):
                self.oscillation_last_known_secondary_pos = 50.0
            
            self.logger.info("Oscillation Experimental 2 tracking started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start tracking: {e}")
            return False
    
    def stop_tracking(self) -> bool:
        """Stop oscillation tracking."""
        try:
            self.tracking_active = False
            self.logger.info("Oscillation Experimental 2 tracking stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop tracking: {e}")
            return False
    
    def set_roi(self, roi: Tuple[int, int, int, int]) -> bool:
        """Set oscillation detection area (optional ROI)."""
        try:
            self.oscillation_area_fixed = roi
            self.logger.info(f"Set oscillation area: {roi}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to set ROI: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources."""
        try:
            # Safely clear collections if they exist
            if hasattr(self, 'oscillation_history') and self.oscillation_history:
                self.oscillation_history.clear()
            if hasattr(self, 'oscillation_cell_persistence') and self.oscillation_cell_persistence:
                self.oscillation_cell_persistence.clear()
            if hasattr(self, 'oscillation_position_history') and self.oscillation_position_history:
                self.oscillation_position_history.clear()
            
            # Safely clear references if they exist
            if hasattr(self, 'prev_gray_oscillation'):
                self.prev_gray_oscillation = None
            if hasattr(self, '_gray_roi_buffer'):
                self._gray_roi_buffer = None
            if hasattr(self, '_gray_full_buffer'):
                self._gray_full_buffer = None
            if hasattr(self, '_prev_gray_osc_buffer'):
                self._prev_gray_osc_buffer = None
            if hasattr(self, 'flow_dense_osc'):
                self.flow_dense_osc = None
            
            self.logger.debug("Oscillation Experimental 2 cleanup complete")
        except Exception as e:
            self.logger.warning(f"Cleanup error: {e}")
    
    # Helper methods
    def _update_fps(self):
        """Update current FPS estimate."""
        # Simple FPS calculation - could be improved
        self.current_fps = 30.0  # Default assumption
    
    def _is_vr_video(self) -> bool:
        """Check if the video appears to be VR format."""
        # Simple heuristic - VR videos are typically wider than 16:9
        if hasattr(self.app, 'processor') and self.app.processor:
            try:
                width = getattr(self.app.processor, 'frame_width', 1920)
                height = getattr(self.app.processor, 'frame_height', 1080)
                aspect_ratio = width / height if height > 0 else 1.0
                return aspect_ratio > 2.0  # Wider than 2:1 suggests VR
            except:
                pass
        return False
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Basic frame preprocessing."""
        return frame  # No preprocessing needed for oscillation detection
    
    def get_status_info(self) -> Dict[str, Any]:
        """Get detailed status information."""
        return {
            "tracker": self.metadata.display_name,
            "active": self.tracking_active,
            "initialized": self._initialized,
            "last_position": self.oscillation_funscript_pos,
            "last_secondary": self.oscillation_funscript_secondary_pos,
            "active_cells": len(self.oscillation_cell_persistence),
            "history_size": sum(len(h) for h in self.oscillation_history.values()),
            "live_amp_enabled": self.signal_amplifier.live_amp_enabled if hasattr(self, 'signal_amplifier') else False,
            "fps": self.current_fps
        }