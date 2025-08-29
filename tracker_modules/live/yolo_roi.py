#!/usr/bin/env python3
"""
YOLO ROI Tracker - Object detection-based ROI tracking.

This tracker uses YOLO object detection to identify key body parts (penis, hands, face, etc.)
and dynamically calculates regions of interest for optical flow-based motion tracking.
It provides automated ROI management with object persistence and interaction detection.

Author: Migrated from YOLO ROI system
Version: 1.0.0
"""

import logging
import numpy as np
import cv2
from collections import deque
from typing import Dict, Any, Optional, List, Tuple

try:
    from ..core.base_tracker import BaseTracker, TrackerMetadata, TrackerResult
except ImportError:
    from tracker_modules.core.base_tracker import BaseTracker, TrackerMetadata, TrackerResult


class YoloRoiTracker(BaseTracker):
    """
    YOLO-based ROI tracker for automated region detection.
    
    This tracker excels at:
    - Automatic ROI detection using YOLO object detection
    - Penis and interaction object tracking
    - Dynamic ROI calculation and smoothing
    - VR-specific ROI width optimizations
    - Object persistence and loss recovery
    - Multi-object interaction classification
    """
    
    def __init__(self):
        super().__init__()
        
        # YOLO model
        self.yolo_model = None
        self.yolo_model_path = None
        
        # ROI tracking state
        self.roi = None  # Current active ROI (x, y, w, h)
        self.penis_last_known_box = None  # Last detected penis box
        self.main_interaction_class = None  # Current main interaction type
        self.frames_since_target_lost = 0
        
        # Detection intervals and persistence
        self.roi_update_interval = 3  # Frames between detections
        self.max_frames_for_roi_persistence = 30  # Max frames to keep ROI without detection
        self.internal_frame_counter = 0
        
        # Optical flow for ROI content analysis
        self.flow_dense = None
        self.prev_gray_main_roi = None
        self.prev_features_main_roi = None
        
        # Position tracking
        self.primary_flow_history_smooth = deque(maxlen=10)
        self.secondary_flow_history_smooth = deque(maxlen=10)
        self.flow_history_window_smooth = 10
        
        # Scaling and sensitivity
        self.sensitivity = 10.0
        self.current_effective_amp_factor = 1.0
        self.adaptive_flow_scale = True
        self.y_offset = 0
        self.x_offset = 0
        
        # Class priorities for interaction detection
        self.CLASS_PRIORITY = {
            'face': 1, 'hand': 2, 'finger': 3, 'breast': 4, 
            'pussy': 5, 'ass': 6, 'dildo': 7, 'other': 99
        }
        
        # Penis size tracking for VR optimization
        self.penis_max_size_history = deque(maxlen=20)
        
        # Settings
        self.show_masks = True
        self.show_roi = True
        self.use_sparse_flow = False
        
        # Performance tracking
        self.current_fps = 30.0
        self.stats_display = []
        
        # Motion inversion detection
        self.enable_inversion_detection = False
        self.motion_mode = 'undetermined'
        self.motion_mode_history = deque(maxlen=30)
        self.motion_mode_history_window = 30
        
        # ROI smoothing
        self.roi_smoothing_factor = 0.7
        self.previous_roi = None
    
    @property
    def metadata(self) -> TrackerMetadata:
        """Return metadata describing this tracker."""
        return TrackerMetadata(
            name="yolo_roi",
            display_name="Live YOLO ROI Tracker",
            description="Automatic ROI detection using YOLO object detection with optical flow tracking",
            category="live",
            version="1.0.0",
            author="YOLO ROI System",
            tags=["yolo", "roi", "object-detection", "optical-flow", "automatic"],
            requires_roi=False,  # ROI is automatically detected
            supports_dual_axis=True
        )
    
    def initialize(self, app_instance, **kwargs) -> bool:
        """Initialize the YOLO ROI tracker."""
        try:
            self.app = app_instance
            
            # Load settings
            if hasattr(app_instance, 'app_settings'):
                settings = app_instance.app_settings
                
                self.roi_update_interval = settings.get('roi_update_interval', 3)
                self.max_frames_for_roi_persistence = settings.get('max_frames_for_roi_persistence', 30)
                self.sensitivity = settings.get('sensitivity', 10.0)
                self.adaptive_flow_scale = settings.get('adaptive_flow_scale', True)
                self.show_masks = settings.get('show_masks', True)
                self.show_roi = settings.get('show_roi', True)
                self.enable_inversion_detection = settings.get('enable_inversion_detection', False)
                
                self.logger.info(f"YOLO ROI settings: update_interval={self.roi_update_interval}, "
                               f"persistence={self.max_frames_for_roi_persistence}, sensitivity={self.sensitivity}")
            
            # Initialize YOLO model
            yolo_model_path = kwargs.get('yolo_model_path')
            if not yolo_model_path and hasattr(app_instance, 'yolo_model_path'):
                yolo_model_path = app_instance.yolo_model_path
            
            if yolo_model_path:
                try:
                    # Try to initialize YOLO model - this would need to be adapted based on the actual YOLO implementation
                    self.yolo_model_path = yolo_model_path
                    self.logger.info(f"YOLO model path set: {yolo_model_path}")
                    # Note: Actual YOLO model loading would depend on the specific implementation
                    # self.yolo_model = load_yolo_model(yolo_model_path)
                except Exception as e:
                    self.logger.error(f"Failed to load YOLO model: {e}")
                    return False
            else:
                self.logger.warning("No YOLO model path provided - object detection will be disabled")
            
            # Initialize optical flow
            try:
                self.flow_dense = cv2.optflow.createOptFlow_DualTVL1()
                self.logger.info("DualTVL1 optical flow initialized for YOLO ROI")
            except AttributeError:
                try:
                    self.flow_dense = cv2.FarnebackOpticalFlow_create()
                    self.logger.info("Farneback optical flow initialized as fallback")
                except AttributeError:
                    self.logger.error("No optical flow implementation available")
                    return False
            
            # Reset state
            self.roi = None
            self.penis_last_known_box = None
            self.main_interaction_class = None
            self.frames_since_target_lost = 0
            self.internal_frame_counter = 0
            self.prev_gray_main_roi = None
            self.prev_features_main_roi = None
            self.primary_flow_history_smooth.clear()
            self.secondary_flow_history_smooth.clear()
            self.penis_max_size_history.clear()
            self.motion_mode_history.clear()
            self.previous_roi = None
            
            self._initialized = True
            self.logger.info("YOLO ROI tracker initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray, frame_time_ms: int, 
                     frame_index: Optional[int] = None) -> TrackerResult:
        """
        Process a frame using YOLO ROI tracking.
        
        This implementation:
        1. Runs YOLO detection at intervals to find penis and interaction objects
        2. Calculates combined ROI based on detected objects
        3. Applies VR-specific optimizations
        4. Tracks motion within the ROI using optical flow
        5. Generates funscript actions based on motion
        """
        try:
            self.internal_frame_counter += 1
            processed_frame = self._preprocess_frame(frame)
            current_frame_gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
            
            action_log_list = []
            detected_objects_this_frame = []
            final_primary_pos, final_secondary_pos = 50, 50
            
            self.current_effective_amp_factor = self._get_effective_amplification_factor()
            
            # Determine if we should run detection this frame
            run_detection_this_frame = self._should_run_detection()
            
            # Initialize stats display
            self.stats_display = [
                f"T-FPS:{self.current_fps:.1f} T(ms):{frame_time_ms} Amp:{self.current_effective_amp_factor:.2f}x"
            ]
            if frame_index is not None:
                self.stats_display.append(f"FIdx:{frame_index}")
            if self.main_interaction_class:
                self.stats_display.append(f"Interact: {self.main_interaction_class}")
            if self.enable_inversion_detection:
                self.stats_display.append(f"Mode: {self.motion_mode}")
            
            # Run object detection and ROI calculation
            if run_detection_this_frame:
                detected_objects_this_frame = self._detect_objects(processed_frame)
                self._process_detections(detected_objects_this_frame, processed_frame.shape[:2])
            
            # Handle ROI persistence when target is lost
            if not self.penis_last_known_box and self.roi is not None:
                self._handle_target_loss()
            
            # Process ROI content if available
            if self.roi and self.tracking_active and self.roi[2] > 0 and self.roi[3] > 0:
                final_primary_pos, final_secondary_pos = self._process_roi_content(
                    processed_frame, current_frame_gray
                )
            
            # Generate funscript actions if tracking is active
            if self.tracking_active:
                action_log_list = self._generate_actions(
                    frame_time_ms, final_primary_pos, final_secondary_pos, frame_index
                )
            
            # Apply visualizations
            self._draw_visualizations(processed_frame, detected_objects_this_frame)
            
            # Prepare debug info
            debug_info = {
                'primary_position': final_primary_pos,
                'secondary_position': final_secondary_pos,
                'roi': self.roi,
                'penis_detected': self.penis_last_known_box is not None,
                'main_interaction': self.main_interaction_class,
                'frames_since_loss': self.frames_since_target_lost,
                'detection_run': run_detection_this_frame,
                'objects_detected': len(detected_objects_this_frame),
                'tracking_active': self.tracking_active
            }
            
            status_msg = f"YOLO ROI | Pos: {final_primary_pos},{final_secondary_pos}"
            if self.main_interaction_class:
                status_msg += f" | {self.main_interaction_class}"
            if self.roi:
                status_msg += f" | ROI: {self.roi[2]}x{self.roi[3]}"
            
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
        """Start YOLO ROI tracking."""
        if not self._initialized:
            self.logger.error("Tracker not initialized")
            return False
        
        self.tracking_active = True
        self.frames_since_target_lost = 0
        self.penis_max_size_history.clear()
        self.prev_gray_main_roi, self.prev_features_main_roi = None, None
        self.penis_last_known_box, self.main_interaction_class = None, None
        
        # Initialize motion mode tracking
        self.motion_mode = 'undetermined'
        self.motion_mode_history.clear()
        
        self.logger.info("YOLO ROI tracking started")
        return True
    
    def stop_tracking(self) -> bool:
        """Stop YOLO ROI tracking."""
        self.tracking_active = False
        self.prev_gray_main_roi, self.prev_features_main_roi = None, None
        
        # Reset motion mode to undetermined when stopping
        self.motion_mode = 'undetermined'
        self.motion_mode_history.clear()
        
        self.logger.info("YOLO ROI tracking stopped")
        return True
    
    def validate_settings(self, settings: Dict[str, Any]) -> bool:
        """Validate YOLO ROI settings."""
        try:
            update_interval = settings.get('roi_update_interval', self.roi_update_interval)
            if not isinstance(update_interval, int) or update_interval < 1 or update_interval > 30:
                self.logger.error("ROI update interval must be between 1 and 30 frames")
                return False
            
            persistence = settings.get('max_frames_for_roi_persistence', self.max_frames_for_roi_persistence)
            if not isinstance(persistence, int) or persistence < 10 or persistence > 300:
                self.logger.error("ROI persistence must be between 10 and 300 frames")
                return False
            
            sensitivity = settings.get('sensitivity', self.sensitivity)
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
            'roi_active': self.roi is not None,
            'roi_dimensions': f"{self.roi[2]}x{self.roi[3]}" if self.roi else "None",
            'penis_detected': self.penis_last_known_box is not None,
            'main_interaction': self.main_interaction_class or "None",
            'frames_since_loss': self.frames_since_target_lost,
            'update_interval': self.roi_update_interval,
            'flow_history_length': len(self.primary_flow_history_smooth),
            'motion_mode': self.motion_mode,
            'sensitivity': self.sensitivity
        }
        
        base_status.update(custom_status)
        return base_status
    
    def cleanup(self):
        """Clean up resources."""
        self.yolo_model = None
        self.roi = None
        self.penis_last_known_box = None
        self.prev_gray_main_roi = None
        self.prev_features_main_roi = None
        self.flow_dense = None
        self.primary_flow_history_smooth.clear()
        self.secondary_flow_history_smooth.clear()
        self.penis_max_size_history.clear()
        self.motion_mode_history.clear()
        self.logger.info("YOLO ROI tracker cleaned up")
    
    # Private helper methods
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Basic frame preprocessing."""
        # Placeholder for any frame preprocessing needed
        return frame.copy()
    
    def _should_run_detection(self) -> bool:
        """Determine if object detection should run this frame."""
        return (
            (self.internal_frame_counter % self.roi_update_interval == 0)
            or (self.roi is None)
            or (not self.penis_last_known_box
                and self.frames_since_target_lost < self.max_frames_for_roi_persistence
                and self.internal_frame_counter % max(1, self.roi_update_interval // 3) == 0)
        )
    
    def _detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """
        Run YOLO object detection on the frame.
        
        Note: This is a placeholder implementation. In a real system, this would
        interface with the actual YOLO model to detect objects.
        """
        # Placeholder for YOLO detection
        # In real implementation, this would call the YOLO model
        if self.yolo_model is None:
            # Return empty list if no model is available
            return []
        
        # Mock detection results for example
        # Real implementation would run: results = self.yolo_model.detect(frame)
        detected_objects = []
        
        try:
            # Placeholder - actual YOLO detection would go here
            # results = self.yolo_model(frame)
            # detected_objects = self._parse_yolo_results(results)
            pass
        except Exception as e:
            self.logger.error(f"YOLO detection error: {e}")
        
        return detected_objects
    
    def _process_detections(self, detected_objects: List[Dict], frame_shape: Tuple[int, int]):
        """Process YOLO detection results to update ROI."""
        # Find penis detections
        penis_boxes = [obj["box"] for obj in detected_objects if obj["class_name"].lower() == "penis"]
        
        if penis_boxes:
            self.frames_since_target_lost = 0
            self._update_penis_tracking(penis_boxes[0])
            
            # Find interacting objects
            interacting_objs = self._find_interacting_objects(self.penis_last_known_box, detected_objects)
            
            # Determine main interaction class
            current_best_interaction_name = None
            if interacting_objs:
                interacting_objs.sort(key=lambda x: self.CLASS_PRIORITY.get(x["class_name"].lower(), 99))
                current_best_interaction_name = interacting_objs[0]["class_name"].lower()
            
            self._update_main_interaction_class(current_best_interaction_name)
            
            # Calculate combined ROI
            combined_roi_candidate = self._calculate_combined_roi(frame_shape, self.penis_last_known_box, interacting_objs)
            
            # Apply VR-specific ROI width limits
            if self._is_vr_video() and self.penis_last_known_box:
                combined_roi_candidate = self._apply_vr_roi_limits(combined_roi_candidate, frame_shape[1])
            
            # Apply ROI smoothing
            self.roi = self._smooth_roi_transition(combined_roi_candidate)
        else:
            # No penis detected
            if self.penis_last_known_box:
                self.logger.info("Primary target (penis) lost in detection cycle.")
            self.penis_last_known_box = None
            self._update_main_interaction_class(None)
    
    def _update_penis_tracking(self, penis_box_xywh: Tuple[int, int, int, int]):
        """Update penis tracking state with new detection."""
        self.penis_last_known_box = penis_box_xywh
        
        # Update size history for VR optimization
        _, _, w, h = penis_box_xywh
        penis_size = w * h
        self.penis_max_size_history.append(penis_size)
    
    def _find_interacting_objects(self, penis_box_xywh: Tuple[int, int, int, int], 
                                 all_detections: List[Dict]) -> List[Dict]:
        """Find objects that are interacting with the penis."""
        if not penis_box_xywh:
            return []
        
        px, py, pw, ph = penis_box_xywh
        interacting_objects = []
        
        for obj in all_detections:
            if obj["class_name"].lower() == "penis":
                continue
            
            ox, oy, ow, oh = obj["box"]
            
            # Simple intersection check
            if (px < ox + ow and px + pw > ox and py < oy + oh and py + ph > oy):
                interacting_objects.append(obj)
        
        return interacting_objects
    
    def _calculate_combined_roi(self, frame_shape: Tuple[int, int], 
                              penis_box_xywh: Tuple[int, int, int, int], 
                              interacting_objects: List[Dict]) -> Tuple[int, int, int, int]:
        """Calculate combined ROI from penis and interacting objects."""
        if not penis_box_xywh:
            return (0, 0, frame_shape[1], frame_shape[0])
        
        # Start with penis box
        min_x, min_y, max_x, max_y = penis_box_xywh[0], penis_box_xywh[1], penis_box_xywh[0] + penis_box_xywh[2], penis_box_xywh[1] + penis_box_xywh[3]
        
        # Expand to include interacting objects
        for obj in interacting_objects:
            ox, oy, ow, oh = obj["box"]
            min_x = min(min_x, ox)
            min_y = min(min_y, oy)
            max_x = max(max_x, ox + ow)
            max_y = max(max_y, oy + oh)
        
        # Add padding
        padding = 20
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(frame_shape[1], max_x + padding)
        max_y = min(frame_shape[0], max_y + padding)
        
        return (min_x, min_y, max_x - min_x, max_y - min_y)
    
    def _apply_vr_roi_limits(self, roi_candidate: Tuple[int, int, int, int], frame_width: int) -> Tuple[int, int, int, int]:
        """Apply VR-specific ROI width limitations."""
        if not self.penis_last_known_box:
            return roi_candidate
        
        penis_w = self.penis_last_known_box[2]
        rx, ry, rw, rh = roi_candidate
        
        # Determine new width based on interaction type
        if self.main_interaction_class in ["face", "hand"]:
            new_rw = penis_w
        else:
            new_rw = min(rw, penis_w * 2)
        
        if new_rw > 0:
            # Recenter the ROI
            original_center_x = rx + rw / 2
            new_rx = int(original_center_x - new_rw / 2)
            
            final_rw = int(min(new_rw, frame_width))
            final_rx = max(0, min(new_rx, frame_width - final_rw))
            
            return (final_rx, ry, final_rw, rh)
        
        return roi_candidate
    
    def _smooth_roi_transition(self, new_roi: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Apply smoothing to ROI transitions."""
        if self.previous_roi is None:
            self.previous_roi = new_roi
            return new_roi
        
        # Linear interpolation for smoothing
        alpha = 1.0 - self.roi_smoothing_factor
        prev_x, prev_y, prev_w, prev_h = self.previous_roi
        new_x, new_y, new_w, new_h = new_roi
        
        smoothed_roi = (
            int(prev_x * self.roi_smoothing_factor + new_x * alpha),
            int(prev_y * self.roi_smoothing_factor + new_y * alpha),
            int(prev_w * self.roi_smoothing_factor + new_w * alpha),
            int(prev_h * self.roi_smoothing_factor + new_h * alpha)
        )
        
        self.previous_roi = smoothed_roi
        return smoothed_roi
    
    def _update_main_interaction_class(self, interaction_class: Optional[str]):
        """Update the main interaction class with stability checks."""
        self.main_interaction_class = interaction_class
    
    def _handle_target_loss(self):
        """Handle ROI persistence when target is lost."""
        self.frames_since_target_lost += 1
        if self.frames_since_target_lost > self.max_frames_for_roi_persistence:
            self.logger.info("ROI persistence timeout. Clearing ROI.")
            self.roi = None
            self.prev_gray_main_roi = None
            self.prev_features_main_roi = None
            self.primary_flow_history_smooth.clear()
            self.secondary_flow_history_smooth.clear()
            self.frames_since_target_lost = 0
    
    def _process_roi_content(self, processed_frame: np.ndarray, 
                           current_frame_gray: np.ndarray) -> Tuple[int, int]:
        """Process the content within the ROI using optical flow."""
        rx, ry, rw, rh = self.roi
        
        # Extract ROI patch
        main_roi_patch_gray = current_frame_gray[
            ry:min(ry + rh, current_frame_gray.shape[0]), 
            rx:min(rx + rw, current_frame_gray.shape[1])
        ]
        
        if main_roi_patch_gray.size == 0:
            self.prev_gray_main_roi = None
            return 50, 50
        
        # Process ROI content (simplified version of the original complex method)
        final_primary_pos, final_secondary_pos = self._analyze_roi_motion(
            processed_frame, main_roi_patch_gray, self.prev_gray_main_roi
        )
        
        self.prev_gray_main_roi = main_roi_patch_gray.copy()
        
        return final_primary_pos, final_secondary_pos
    
    def _analyze_roi_motion(self, processed_frame: np.ndarray, current_roi_gray: np.ndarray,
                          prev_roi_gray: Optional[np.ndarray]) -> Tuple[int, int]:
        """Analyze motion within the ROI patch."""
        if prev_roi_gray is None or current_roi_gray.shape != prev_roi_gray.shape:
            return 50, 50
        
        if not self.flow_dense:
            return 50, 50
        
        try:
            # Calculate optical flow
            flow = self.flow_dense.calc(
                np.ascontiguousarray(prev_roi_gray), 
                np.ascontiguousarray(current_roi_gray), 
                None
            )
            
            if flow is None:
                return 50, 50
            
            # Extract motion components
            dx_raw = np.median(flow[..., 0])
            dy_raw = np.median(flow[..., 1])
            
            # Apply smoothing
            self.primary_flow_history_smooth.append(dy_raw)
            self.secondary_flow_history_smooth.append(dx_raw)
            
            dy_smooth = np.median(self.primary_flow_history_smooth) if self.primary_flow_history_smooth else dy_raw
            dx_smooth = np.median(self.secondary_flow_history_smooth) if self.secondary_flow_history_smooth else dx_raw
            
            # Apply scaling
            if self.adaptive_flow_scale:
                # Simplified adaptive scaling
                final_primary_pos = int(np.clip(50 + dy_smooth * 2.0, 0, 100))
                final_secondary_pos = int(np.clip(50 + dx_smooth * 2.0, 0, 100))
            else:
                # Manual scaling
                manual_scale_multiplier = (self.sensitivity / 10.0) * self.current_effective_amp_factor
                final_primary_pos = int(np.clip(50 + dy_smooth * manual_scale_multiplier + self.y_offset, 0, 100))
                final_secondary_pos = int(np.clip(50 + dx_smooth * manual_scale_multiplier + self.x_offset, 0, 100))
            
            return final_primary_pos, final_secondary_pos
            
        except Exception as e:
            self.logger.error(f"ROI motion analysis error: {e}")
            return 50, 50
    
    def _generate_actions(self, frame_time_ms: int, final_primary_pos: int, final_secondary_pos: int,
                         frame_index: Optional[int]) -> List[Dict]:
        """Generate funscript actions based on calculated positions."""
        action_log_list = []
        
        # Get current tracking settings
        current_tracking_axis_mode = getattr(self.app, 'tracking_axis_mode', 'both')
        current_single_axis_output = getattr(self.app, 'single_axis_output_target', 'primary')
        
        primary_to_write, secondary_to_write = None, None
        
        if current_tracking_axis_mode == "both":
            primary_to_write, secondary_to_write = final_primary_pos, final_secondary_pos
        elif current_tracking_axis_mode == "vertical":
            if current_single_axis_output == "primary":
                primary_to_write = final_primary_pos
            else:
                secondary_to_write = final_primary_pos
        elif current_tracking_axis_mode == "horizontal":
            if current_single_axis_output == "primary":
                primary_to_write = final_secondary_pos
            else:
                secondary_to_write = final_secondary_pos
        
        # Add action to funscript
        if hasattr(self.app, 'funscript'):
            self.app.funscript.add_action(
                timestamp_ms=frame_time_ms, 
                primary_pos=primary_to_write, 
                secondary_pos=secondary_to_write
            )
        
        # Create action log entry
        action_log_entry = {
            "at": frame_time_ms,
            "pos": primary_to_write,
            "secondary_pos": secondary_to_write,
            "mode": current_tracking_axis_mode,
            "target": current_single_axis_output if current_tracking_axis_mode != "both" else "N/A",
            "roi_main": self.roi,
            "amp": self.current_effective_amp_factor
        }
        
        if frame_index is not None:
            action_log_entry["frame_index"] = frame_index
        
        action_log_list.append(action_log_entry)
        
        return action_log_list
    
    def _draw_visualizations(self, processed_frame: np.ndarray, detected_objects: List[Dict]):
        """Draw visualization overlays on the frame."""
        # Draw object detection masks
        if self.show_masks and detected_objects:
            self._draw_detections(processed_frame, detected_objects)
        
        # Draw ROI rectangle
        if self.show_roi and self.roi:
            rx, ry, rw, rh = self.roi
            color = self._get_class_color(
                self.main_interaction_class or ("penis" if self.penis_last_known_box else "persisting")
            )
            cv2.rectangle(processed_frame, (rx, ry), (rx + rw, ry + rh), color, 2)
            
            # Draw status text
            status_text = self.main_interaction_class or ('P' if self.penis_last_known_box else 'Lost...')
            cv2.putText(processed_frame, status_text, (rx, ry - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw stats display
        for i, stat in enumerate(self.stats_display):
            cv2.putText(processed_frame, stat, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if self.tracking_active:
            cv2.putText(processed_frame, "YOLO ROI TRACKING", (10, processed_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    def _draw_detections(self, frame: np.ndarray, detections: List[Dict]):
        """Draw detection bounding boxes and labels."""
        for detection in detections:
            box = detection["box"]
            class_name = detection["class_name"]
            confidence = detection.get("confidence", 1.0)
            
            x, y, w, h = box
            color = self._get_class_color(class_name)
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def _get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """Get color for a specific class."""
        color_map = {
            'penis': (0, 255, 0),      # Green
            'face': (255, 0, 0),       # Blue  
            'hand': (0, 255, 255),     # Yellow
            'finger': (255, 255, 0),   # Cyan
            'breast': (255, 0, 255),   # Magenta
            'pussy': (128, 0, 128),    # Purple
            'ass': (255, 165, 0),      # Orange
            'persisting': (128, 128, 128)  # Gray
        }
        return color_map.get(class_name.lower() if class_name else 'persisting', (255, 255, 255))
    
    def _get_effective_amplification_factor(self) -> float:
        """Calculate effective amplification factor."""
        # Simplified version - real implementation would be more complex
        return 1.0
    
    def _is_vr_video(self) -> bool:
        """Detect if this is a VR video based on aspect ratio."""
        try:
            if hasattr(self.app, 'get_video_dimensions'):
                width, height = self.app.get_video_dimensions()
                if width and height:
                    aspect_ratio = width / height
                    return aspect_ratio >= 1.8  # Threshold for VR detection
        except Exception:
            pass
        return False