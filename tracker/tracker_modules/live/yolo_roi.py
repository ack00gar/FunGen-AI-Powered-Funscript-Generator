#!/usr/bin/env python3
"""
YOLO ROI Tracker - Object detection-based ROI tracking.

This tracker uses YOLO object detection to identify key body parts (penis, hands, face, etc.)
and dynamically calculates regions of interest for optical flow-based motion tracking.
It provides automated ROI management with object persistence and interaction detection.

Author: Migrated from YOLO ROI system
Version: 1.1.0
"""

# Constants
MIN_FLOW_HISTORY_WINDOW = 3
POSITION_HISTORY_SMOOTH_SIZE = 2
SIGNAL_AMPLIFIER_HISTORY_SIZE = 120  # 4 seconds @ 30fps
SIGNAL_AMPLIFIER_SMOOTHING_ALPHA = 0.7
FINAL_SMOOTHING_ALPHA = 0.7
MIN_ROI_SIZE = 128
VR_ASPECT_RATIO_THRESHOLD = 1.8
VR_ROI_MULTIPLIER_FACE_HAND = 2.5
VR_ROI_MULTIPLIER_DEFAULT = 2
FPS_UPDATE_FRAME_COUNT = 30

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
    from tracker.tracker_modules.core.base_tracker import BaseTracker, TrackerMetadata, TrackerResult
    from tracker.tracker_modules.helpers.signal_amplifier import SignalAmplifier


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
        
        # Import constants once at the top
        from config import constants
        
        # YOLO model
        self.yolo_model = None
        self.yolo_model_path = None
        
        # ROI tracking state
        self.roi = None  # Current active ROI (x, y, w, h)
        self.penis_last_known_box = None  # Last detected penis box
        self.main_interaction_class = None  # Current main interaction type
        self.frames_since_target_lost = 0
        
        # Detection intervals and persistence (will be overridden by app settings)
        self.roi_update_interval = constants.DEFAULT_ROI_UPDATE_INTERVAL
        self.max_frames_for_roi_persistence = constants.DEFAULT_ROI_PERSISTENCE_FRAMES
        self.internal_frame_counter = 0
        
        # Optical flow for ROI content analysis
        self.flow_dense = None
        self.prev_gray_main_roi = None
        self.prev_features_main_roi = None
        
        # Position tracking (minimal smoothing for responsiveness)
        self.flow_history_window_smooth = max(MIN_FLOW_HISTORY_WINDOW, constants.DEFAULT_FLOW_HISTORY_SMOOTHING_WINDOW)
        self.primary_flow_history_smooth = deque(maxlen=self.flow_history_window_smooth)
        self.secondary_flow_history_smooth = deque(maxlen=self.flow_history_window_smooth)

        # Reduced smoothing for lower latency (<100ms total delay)
        self.position_history_smooth = deque(maxlen=POSITION_HISTORY_SMOOTH_SIZE)
        self.last_primary_position = None  # Initialize as None to detect first frame
        self.last_secondary_position = None
        self.first_valid_position_captured = False  # Flag for startup calibration

        # Enhanced signal mastering using helper module
        self.signal_amplifier = SignalAmplifier(
            history_size=SIGNAL_AMPLIFIER_HISTORY_SIZE,
            enable_live_amp=True,  # Enable dynamic amplification by default
            smoothing_alpha=SIGNAL_AMPLIFIER_SMOOTHING_ALPHA,
            logger=self.logger
        )
        
        # Adaptive flow range attributes (from original tracker.py)
        self.flow_min_primary_adaptive = -0.1
        self.flow_max_primary_adaptive = 0.1
        self.flow_min_secondary_adaptive = -0.1
        self.flow_max_secondary_adaptive = 0.1
        
        # VR and motion mode detection (from original tracker.py)
        self.enable_inversion_detection = False  # Disabled by default, enabled via settings
        self.motion_mode = 'undetermined'  # 'thrusting', 'riding', or 'undetermined'
        self.motion_mode_history = deque(maxlen=30)  # Use deque for automatic size management
        self.motion_inversion_threshold = 1.5
        self.motion_mode_history_window = 30
        self.inversion_detection_split_ratio = 3.0
        
        # Flow calculation modes
        self.use_sparse_flow = False
        
        # Scaling and sensitivity (will be overridden by app settings)
        self.sensitivity = constants.DEFAULT_LIVE_TRACKER_SENSITIVITY
        self.current_effective_amp_factor = 1.0
        self.adaptive_flow_scale = True
        self.y_offset = constants.DEFAULT_LIVE_TRACKER_Y_OFFSET
        self.x_offset = constants.DEFAULT_LIVE_TRACKER_X_OFFSET
        
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
        self._fps_counter = 0
        self._fps_last_time = time.time()
        self.stats_display = []

        # VR detection cache (to avoid repeated aspect ratio calculations)
        self._is_vr_video_cached = None

        # ROI smoothing (will be overridden by app settings)
        self.roi_smoothing_factor = constants.DEFAULT_ROI_SMOOTHING_FACTOR
        self.previous_roi = None
        
        # Class history tracking for stability 
        self.class_history = []
        self.class_stability_window = 10
        self.last_interaction_time = 0
        
        # Padding for ROI calculation (will be overridden by app settings)
        self.roi_padding = constants.DEFAULT_TRACKER_ROI_PADDING
        
        # Base amplification settings (will be overridden by app settings)
        self.base_amplification_factor = constants.DEFAULT_LIVE_TRACKER_BASE_AMPLIFICATION
        self.class_specific_amplification_multipliers = constants.DEFAULT_CLASS_AMP_MULTIPLIERS.copy()
        
        # Model classes list
        self.classes = []
        
        # Confidence threshold for detections (will be overridden by app settings)
        self.confidence_threshold = constants.DEFAULT_TRACKER_CONFIDENCE_THRESHOLD
    
    @property
    def metadata(self) -> TrackerMetadata:
        """Return metadata describing this tracker."""
        return TrackerMetadata(
            name="yolo_roi",
            display_name="YOLO ROI Tracker",
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
            
            # Load settings from control panel configuration
            if hasattr(app_instance, 'app_settings'):
                settings = app_instance.app_settings
                
                # Detection & ROI settings (from control panel)
                self.confidence_threshold = settings.get('live_tracker_confidence_threshold', 0.7)
                self.roi_padding = settings.get('live_tracker_roi_padding', 20)
                self.roi_update_interval = settings.get('live_tracker_roi_update_interval', 3)
                self.roi_smoothing_factor = settings.get('live_tracker_roi_smoothing_factor', 0.7)
                self.max_frames_for_roi_persistence = settings.get('live_tracker_roi_persistence_frames', 30)
                
                # Optical flow settings
                self.use_sparse_flow = settings.get('live_tracker_use_sparse_flow', False)
                dis_preset = settings.get('live_tracker_dis_flow_preset', 'ULTRAFAST').lower()
                self.dis_finest_scale = settings.get('live_tracker_dis_finest_scale', 0)
                
                # Output signal generation settings
                self.sensitivity = settings.get('live_tracker_sensitivity', 10.0)
                self.base_amplification_factor = settings.get('live_tracker_base_amplification', 1.0)
                self.class_specific_amplification_multipliers = settings.get('live_tracker_class_amp_multipliers', {
                    'face': 1.0, 'hand': 1.0, 'finger': 1.0, 'breast': 1.0,
                    'pussy': 1.2, 'ass': 1.2, 'dildo': 1.0
                })
                from config import constants as config_constants
                self.flow_history_window_smooth = settings.get('live_tracker_flow_smoothing_window', config_constants.DEFAULT_FLOW_HISTORY_SMOOTHING_WINDOW)
                
                # Update flow history deques with new window size
                self.primary_flow_history_smooth = deque(maxlen=self.flow_history_window_smooth)
                self.secondary_flow_history_smooth = deque(maxlen=self.flow_history_window_smooth)
                
                # UI display settings  
                self.show_masks = settings.get('show_masks', True)
                self.show_roi = settings.get('show_roi', True)
                self.enable_inversion_detection = settings.get('enable_inversion_detection', False)
                
                # Update flow dense preset
                if hasattr(self, 'flow_dense') and self.flow_dense:
                    self._update_flow_preset(dis_preset)
                
                self.logger.info(f"YOLO ROI settings loaded from control panel:")
                self.logger.info(f"  Detection: conf={self.confidence_threshold}, padding={self.roi_padding}")
                self.logger.info(f"  ROI: update_interval={self.roi_update_interval}, persistence={self.max_frames_for_roi_persistence}")
                self.logger.info(f"  Signal: sensitivity={self.sensitivity}, base_amp={self.base_amplification_factor}")
                self.logger.info(f"  Flow: sparse={self.use_sparse_flow}, preset={dis_preset}, smoothing={self.flow_history_window_smooth}")
            
            # Initialize YOLO model - primary path comes from bridge's tracker_model_path
            yolo_model_path = None
            
            # Primary: Check for model_path attribute (set by ModularTrackerBridge from tracker_model_path)
            if hasattr(self, 'model_path') and self.model_path:
                yolo_model_path = self.model_path
                self.logger.info(f"Using model path from bridge: {yolo_model_path}")
            
            # Fallback: Check kwargs
            if not yolo_model_path:
                yolo_model_path = kwargs.get('yolo_model_path')
            
            # Fallback: Check app settings for YOLO model path (most common location)
            if not yolo_model_path and hasattr(app_instance, 'app_settings'):
                yolo_model_path = app_instance.app_settings.get('yolo_det_model_path', None)
                if yolo_model_path:
                    self.logger.info(f"Using YOLO model path from app settings: {yolo_model_path}")
            
            # Fallback: Check app instance for model path
            if not yolo_model_path and hasattr(app_instance, 'tracker_model_path'):
                yolo_model_path = app_instance.tracker_model_path
            elif not yolo_model_path and hasattr(app_instance, 'det_model_path'):
                yolo_model_path = app_instance.det_model_path
            elif not yolo_model_path and hasattr(app_instance, 'yolo_model_path'):
                yolo_model_path = app_instance.yolo_model_path
            
            # Fallback: Check processor for model path
            if not yolo_model_path and app_instance and hasattr(app_instance, 'processor'):
                if hasattr(app_instance.processor, 'tracker_model_path'):
                    yolo_model_path = app_instance.processor.tracker_model_path
                elif hasattr(app_instance.processor, 'det_model_path'):
                    yolo_model_path = app_instance.processor.det_model_path
                    
            # Validate path exists
            if yolo_model_path:
                import os
                if os.path.exists(yolo_model_path):
                    self.logger.info(f"Found YOLO model at: {yolo_model_path}")
                else:
                    self.logger.warning(f"YOLO model path does not exist: {yolo_model_path}")
                    yolo_model_path = None
            
            if yolo_model_path:
                try:
                    # Load the actual YOLO model like the original tracker did
                    from ultralytics import YOLO
                    self.yolo_model_path = yolo_model_path
                    self.yolo_model = YOLO(yolo_model_path, task='detect')
                    self.logger.info(f"YOLO model loaded successfully from: {yolo_model_path}")
                    
                    # Load class names
                    names_attr = getattr(self.yolo_model, 'names', None)
                    if names_attr:
                        if isinstance(names_attr, dict):
                            self.classes = list(names_attr.values())
                        else:
                            self.classes = list(names_attr)
                        self.logger.info(f"Loaded {len(self.classes)} classes from YOLO model")
                except Exception as e:
                    self.logger.error(f"Failed to load YOLO model: {e}")
                    self.yolo_model = None
                    return False
            else:
                self.logger.warning("No YOLO model path provided - object detection will be disabled")
                self.yolo_model = None
            
            # Initialize optical flow - use DIS with ultrafast preset for better performance
            try:
                self.flow_dense = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
                self.logger.info("DIS optical flow initialized (ultrafast preset) for YOLO ROI")
            except AttributeError:
                try:
                    # Fallback to medium preset if ultrafast not available
                    self.flow_dense = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
                    self.logger.info("DIS optical flow initialized (medium preset) for YOLO ROI")
                except AttributeError:
                    self.logger.error("No DIS optical flow implementation available")
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
            self.logger.error(f"Initialization failed: {e}", exc_info=True)
            return False
    
    def update_settings(self, **kwargs) -> bool:
        """Update tracker settings dynamically from control panel changes."""
        try:
            if not self.app or not hasattr(self.app, 'app_settings'):
                return False
                
            settings = self.app.app_settings
            
            # Update detection & ROI settings
            self.confidence_threshold = settings.get('live_tracker_confidence_threshold', self.confidence_threshold)
            self.roi_padding = settings.get('live_tracker_roi_padding', self.roi_padding)
            self.roi_update_interval = settings.get('live_tracker_roi_update_interval', self.roi_update_interval)
            self.roi_smoothing_factor = settings.get('live_tracker_roi_smoothing_factor', self.roi_smoothing_factor)
            self.max_frames_for_roi_persistence = settings.get('live_tracker_roi_persistence_frames', self.max_frames_for_roi_persistence)
            
            # Update optical flow settings
            self.use_sparse_flow = settings.get('live_tracker_use_sparse_flow', self.use_sparse_flow)
            dis_preset = settings.get('live_tracker_dis_flow_preset', 'ULTRAFAST').lower()
            self.dis_finest_scale = settings.get('live_tracker_dis_finest_scale', self.dis_finest_scale)
            
            # Update signal generation settings
            old_sensitivity = self.sensitivity
            self.sensitivity = settings.get('live_tracker_sensitivity', self.sensitivity)
            self.base_amplification_factor = settings.get('live_tracker_base_amplification', self.base_amplification_factor)
            self.class_specific_amplification_multipliers = settings.get('live_tracker_class_amp_multipliers', self.class_specific_amplification_multipliers)
            
            from config import constants as config_constants
            self.flow_history_window_smooth = settings.get('live_tracker_flow_smoothing_window', self.flow_history_window_smooth)
            
            # Log significant changes
            if abs(old_sensitivity - self.sensitivity) > 0.1:
                self.logger.info(f"YOLO ROI tracker sensitivity updated: {old_sensitivity:.1f} -> {self.sensitivity:.1f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update settings: {e}", exc_info=True)
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
            # Validate frame dimensions
            if frame is None or frame.size == 0:
                self.logger.error("Received invalid frame (None or empty)")
                return TrackerResult(
                    processed_frame=frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8),
                    action_log=None,
                    debug_info={'error': 'Invalid frame'},
                    status_message="Error: Invalid frame"
                )

            if len(frame.shape) != 3 or frame.shape[2] != 3:
                self.logger.error(f"Invalid frame shape: {frame.shape}. Expected (H, W, 3)")
                return TrackerResult(
                    processed_frame=frame,
                    action_log=None,
                    debug_info={'error': f'Invalid frame shape: {frame.shape}'},
                    status_message=f"Error: Invalid frame shape {frame.shape}"
                )

            self.internal_frame_counter += 1
            self._update_fps()
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
            
            # Add model availability status for user feedback
            if not self.yolo_model:
                self.stats_display.append("No YOLO model - Manual ROI required")
            elif self.roi is None:
                self.stats_display.append("No ROI detected - Scanning for objects")
            
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
            self.logger.error(f"Frame processing error: {e}", exc_info=True)
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

        # Reset smoothing state to prevent position 50 fallback
        self.last_primary_position = None
        self.last_secondary_position = None
        self.first_valid_position_captured = False
        self.position_history_smooth.clear()

        # Reset signal amplifier for new tracking session
        self.signal_amplifier.reset()

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
            
            sensitivity = settings.get('sensitivity', self._get_current_sensitivity())
            if not isinstance(sensitivity, (int, float)) or sensitivity <= 0:
                self.logger.error("Sensitivity must be positive")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Settings validation error: {e}", exc_info=True)
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
        self._is_vr_video_cached = None
        self.logger.debug("YOLO ROI tracker cleaned up")
    
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
        Run YOLO object detection on the frame using the loaded model.
        This is the actual implementation ported from the original tracker.
        """
        detected_objects = []

        # Check if detection model is available
        if self.yolo_model is None:
            self.logger.debug("No YOLO model available for detection")
            return detected_objects

        # Get confidence threshold
        confidence_threshold = self._get_current_confidence_threshold()
        
        # Determine discarded classes based on app context if available
        discarded_classes_runtime = []
        if self.app and hasattr(self.app, 'discarded_tracking_classes'):
            discarded_classes_runtime = self.app.discarded_tracking_classes

        try:
            # Run YOLO detection - this is the actual detection call from original tracker
            from config import constants as config_constants
            device = getattr(config_constants, 'DEVICE', 'auto')
            
            results = self.yolo_model(frame, device=device, verbose=False, conf=confidence_threshold)

            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = None
                    
                    # Get class name from model names
                    if hasattr(self, 'classes') and self.classes and 0 <= class_id < len(self.classes):
                        class_name = self.classes[class_id]
                    else:
                        # Try names in result
                        rn = getattr(result, 'names', None)
                        if isinstance(rn, dict) and class_id in rn:
                            class_name = rn[class_id]
                        else:
                            class_name = f"unknown_{class_id}"

                    # Skip discarded classes
                    if class_name and class_name.lower() in discarded_classes_runtime:
                        continue

                    # Convert bounding box from xyxy to xywh format
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    w, h = x2 - x1, y2 - y1
                    
                    # Create detection object
                    detection = {
                        "class_name": class_name,
                        "confidence": conf,
                        "box": (x1, y1, w, h),  # xywh format
                        "class_id": class_id
                    }
                    
                    detected_objects.append(detection)
                    
        except Exception as e:
            self.logger.error(f"YOLO detection error: {e}", exc_info=True)
        
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
        """Find objects that are interacting with the penis (from original tracker)."""
        if not penis_box_xywh:
            return []
        
        px, py, pw, ph = penis_box_xywh
        pcx, pcy = px + pw / 2, py + ph / 2  # Penis center
        interacting_objects = []
        
        for obj in all_detections:
            if obj["class_name"].lower() == "penis":
                continue
            
            ox, oy, ow, oh = obj["box"]
            ocx, ocy = ox + ow / 2, oy + oh / 2  # Object center
            
            # Calculate distance between centers
            dist = np.sqrt((ocx - pcx) ** 2 + (ocy - pcy) ** 2)
            
            # Calculate maximum allowed distance (85% of combined half-diagonals)
            max_dist = (np.sqrt(pw ** 2 + ph ** 2) / 2 + np.sqrt(ow ** 2 + oh ** 2) / 2) * 0.85
            
            if dist < max_dist:
                interacting_objects.append(obj)
        
        return interacting_objects
    
    def _calculate_combined_roi(self, frame_shape: Tuple[int, int],
                              penis_box_xywh: Tuple[int, int, int, int],
                              interacting_objects: List[Dict]) -> Tuple[int, int, int, int]:
        """Calculate combined ROI from penis and interacting objects (from original tracker)."""
        entities = [penis_box_xywh] + [obj["box"] for obj in interacting_objects]

        min_x = min(e[0] for e in entities)
        min_y = min(e[1] for e in entities)
        max_x_coord = max(e[0] + e[2] for e in entities)
        max_y_coord = max(e[1] + e[3] for e in entities)

        # Apply padding
        rx1 = max(0, min_x - self.roi_padding)
        ry1 = max(0, min_y - self.roi_padding)
        rx2 = min(frame_shape[1], max_x_coord + self.roi_padding)
        ry2 = min(frame_shape[0], max_y_coord + self.roi_padding)

        rw = rx2 - rx1
        rh = ry2 - ry1

        # Ensure minimum size
        if rw < MIN_ROI_SIZE:
            deficit = MIN_ROI_SIZE - rw
            rx1 = max(0, rx1 - deficit // 2)
            rw = MIN_ROI_SIZE
        if rh < MIN_ROI_SIZE:
            deficit = MIN_ROI_SIZE - rh
            ry1 = max(0, ry1 - deficit // 2)
            rh = MIN_ROI_SIZE

        # Ensure ROI stays within frame bounds
        if rx1 + rw > frame_shape[1]:
            rx1 = frame_shape[1] - rw
        if ry1 + rh > frame_shape[0]:
            ry1 = frame_shape[0] - rh

        return (int(rx1), int(ry1), int(rw), int(rh))
    
    def _apply_vr_roi_limits(self, roi_candidate: Tuple[int, int, int, int], frame_width: int) -> Tuple[int, int, int, int]:
        """Apply VR-specific ROI width limitations."""
        if not self.penis_last_known_box:
            return roi_candidate

        penis_w = self.penis_last_known_box[2]
        rx, ry, rw, rh = roi_candidate

        # Determine new width based on interaction type
        # Increased width for face/hand to accommodate horizontal motion (blowjob/handjob)
        if self.main_interaction_class in ["face", "hand"]:
            new_rw = penis_w * VR_ROI_MULTIPLIER_FACE_HAND
        else:
            new_rw = min(rw, penis_w * VR_ROI_MULTIPLIER_DEFAULT)

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
    
    def _update_main_interaction_class(self, current_best_interaction_class_name: Optional[str]):
        """Update the main interaction class with stability checks (from original tracker)."""
        import time
        
        # Add to class history
        self.class_history.append(current_best_interaction_class_name)
        if len(self.class_history) > self.class_stability_window:
            self.class_history.pop(0)
        
        if not self.class_history:
            self.main_interaction_class = None
            return
        
        # Count occurrences
        counts = {}
        for cls_name in self.class_history:
            if cls_name:
                counts[cls_name] = counts.get(cls_name, 0) + 1
        
        if not counts:
            self.main_interaction_class = None
            return
        
        # Sort by priority (lower is better) and count (higher is better)
        sorted_cand = sorted(
            counts.items(), 
            key=lambda it: (self.CLASS_PRIORITY.get(it[0], 99), -it[1])
        )
        
        # Require stability - need at least half the window size + 1 occurrences
        stable_cls = None
        if sorted_cand and sorted_cand[0][1] >= self.class_stability_window // 2 + 1:
            stable_cls = sorted_cand[0][0]
        
        # Update main interaction class if changed
        if self.main_interaction_class != stable_cls:
            self.main_interaction_class = stable_cls
            if stable_cls:
                self.last_interaction_time = time.time()
                effective_amp = self._get_effective_amplification_factor()
                self.logger.info(f"Interaction: {stable_cls} (Effective Amp: {effective_amp:.2f}x)")
        
        # Timeout check - revert to None if no recent interaction
        if self.main_interaction_class and (time.time() - self.last_interaction_time > 3.0):
            self.logger.info(f"Interaction {self.main_interaction_class} timed out. Reverting to base amp.")
            self.main_interaction_class = None
        
        # Update current amplification factor
        self.current_effective_amp_factor = self._get_effective_amplification_factor()
    
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
        """Process the content within the ROI using original tracker logic."""
        rx, ry, rw, rh = self.roi
        
        # Extract ROI patch
        main_roi_patch_gray = current_frame_gray[
            ry:min(ry + rh, current_frame_gray.shape[0]), 
            rx:min(rx + rw, current_frame_gray.shape[1])
        ]
        
        if main_roi_patch_gray.size == 0:
            self.prev_gray_main_roi = None
            return 50, 50
        
        # Call the original process_main_roi_content logic
        final_primary_pos, final_secondary_pos, _, _, self.prev_features_main_roi = \
            self.process_main_roi_content(processed_frame, main_roi_patch_gray, self.prev_gray_main_roi, self.prev_features_main_roi)
        
        self.prev_gray_main_roi = main_roi_patch_gray.copy()
        
        return final_primary_pos, final_secondary_pos
    
    def process_main_roi_content(self, processed_frame_draw_target: np.ndarray, current_roi_patch_gray: np.ndarray, prev_roi_patch_gray: Optional[np.ndarray], prev_sparse_features: Optional[np.ndarray]) -> Tuple[int, int, float, float, Optional[np.ndarray]]:
        """EXACT original process_main_roi_content method from tracker.py"""

        updated_sparse_features_out = None
        dy_raw, dx_raw, lower_mag, upper_mag = 0.0, 0.0, 0.0, 0.0
        flow_field_for_vis = None  # Initialize here to ensure it's always defined

        if self.use_sparse_flow:
            dx_raw, dy_raw, _, updated_sparse_features_out = self._calculate_flow_in_patch(
                current_roi_patch_gray, prev_roi_patch_gray, use_sparse=True,
                prev_features_for_sparse=prev_sparse_features)
        else:
            # Use our sub-region analysis method for dense flow
            dy_raw, dx_raw, lower_mag, upper_mag, flow_field_for_vis = self._calculate_flow_in_sub_regions(
                current_roi_patch_gray, prev_roi_patch_gray)

        is_vr_video = self._is_vr_video()

        if self.enable_inversion_detection and is_vr_video:
            # This logic now ONLY runs for VR videos.
            current_dominant_motion = 'undetermined'
            motion_threshold = getattr(self, 'motion_inversion_threshold', 1.5)
            if lower_mag > upper_mag * motion_threshold:
                current_dominant_motion = 'thrusting'
            elif upper_mag > lower_mag * motion_threshold:
                current_dominant_motion = 'riding'

            self.motion_mode_history.append(current_dominant_motion)
            window_size = getattr(self, 'motion_mode_history_window', 30)
            if len(self.motion_mode_history) > window_size:
                self.motion_mode_history.pop(0)

            if self.motion_mode_history:
                from collections import Counter
                most_common_mode, count = Counter(self.motion_mode_history).most_common(1)[0]
                # Solidly switch mode if a new one is dominant in the history window.
                if count > window_size // 2 and self.motion_mode != most_common_mode:
                    self.logger.info(f"Motion mode switched: from '{self.motion_mode}' to '{most_common_mode}'.")
                    self.motion_mode = most_common_mode
        else:
            # If the feature is disabled or the video is 2D, ensure we are in the default, non-inverting state.
            self.motion_mode = 'thrusting'  # Default non-inverting mode

        # Smooth the raw dx/dy values from the overall flow
        self.primary_flow_history_smooth.append(dy_raw)
        self.secondary_flow_history_smooth.append(dx_raw)

        if len(self.primary_flow_history_smooth) > self.flow_history_window_smooth:
            self.primary_flow_history_smooth.pop(0)
        if len(self.secondary_flow_history_smooth) > self.flow_history_window_smooth:
            self.secondary_flow_history_smooth.pop(0)

        dy_smooth = np.median(self.primary_flow_history_smooth) if self.primary_flow_history_smooth else dy_raw
        dx_smooth = np.median(self.secondary_flow_history_smooth) if self.secondary_flow_history_smooth else dx_raw

        # Calculate the base positions before potential inversion
        size_factor = self.get_current_penis_size_factor()
        if self.adaptive_flow_scale:
            base_primary_pos = self._apply_adaptive_scaling(dy_smooth, "flow_min_primary_adaptive", "flow_max_primary_adaptive", size_factor, True)
            secondary_pos = self._apply_adaptive_scaling(dx_smooth, "flow_min_secondary_adaptive", "flow_max_secondary_adaptive", size_factor, False)
        else:
            effective_amp_factor = self._get_effective_amplification_factor()
            manual_scale_multiplier = (self._get_current_sensitivity() / 10.0) * (1.0 / size_factor) * effective_amp_factor
            base_primary_pos = int(np.clip(50 + dy_smooth * manual_scale_multiplier + self.y_offset, 0, 100))
            secondary_pos = int(np.clip(50 + dx_smooth * manual_scale_multiplier + self.x_offset, 0, 100))

        # Fix inversion logic: thrusting should be normal (base_primary_pos), riding should be inverted
        raw_primary_pos = base_primary_pos if self.motion_mode == "thrusting" else 100 - base_primary_pos
        
        # Apply enhanced signal mastering using helper module
        sensitivity = self._get_current_sensitivity()
        enhanced_primary, enhanced_secondary = self.signal_amplifier.enhance_signal(
            raw_primary_pos, secondary_pos, dy_smooth, dx_smooth, 
            sensitivity=sensitivity, apply_smoothing=False  # We'll apply our own smoothing
        )
        
        # Apply additional smoothing to reduce jerkiness
        primary_pos, secondary_pos = self._apply_final_smoothing(enhanced_primary, enhanced_secondary)

        return primary_pos, secondary_pos, dy_smooth, dx_smooth, updated_sparse_features_out
    
    def _apply_final_smoothing(self, primary_pos: int, secondary_pos: int) -> Tuple[int, int]:
        """Apply minimal smoothing for low latency (~50ms total delay)."""
        # First frame: initialize with actual position instead of defaulting to 50
        if self.last_primary_position is None or self.last_secondary_position is None:
            self.last_primary_position = primary_pos
            self.last_secondary_position = secondary_pos
            self.first_valid_position_captured = True
            return primary_pos, secondary_pos

        # Exponential moving average with reduced lag
        # Apply EMA smoothing
        smoothed_primary = int(FINAL_SMOOTHING_ALPHA * primary_pos + (1 - FINAL_SMOOTHING_ALPHA) * self.last_primary_position)
        smoothed_secondary = int(FINAL_SMOOTHING_ALPHA * secondary_pos + (1 - FINAL_SMOOTHING_ALPHA) * self.last_secondary_position)

        # Store position history for minimal additional smoothing
        self.position_history_smooth.append((smoothed_primary, smoothed_secondary))

        # Apply median smoothing only if we have POSITION_HISTORY_SMOOTH_SIZE samples
        if len(self.position_history_smooth) >= POSITION_HISTORY_SMOOTH_SIZE:
            primary_values = [pos[0] for pos in list(self.position_history_smooth)]
            secondary_values = [pos[1] for pos in list(self.position_history_smooth)]

            final_primary = int(np.median(primary_values))
            final_secondary = int(np.median(secondary_values))
        else:
            final_primary = smoothed_primary
            final_secondary = smoothed_secondary

        # Update last positions
        self.last_primary_position = final_primary
        self.last_secondary_position = final_secondary

        return final_primary, final_secondary
    
    def _calculate_flow_in_patch(self, current_patch_gray: np.ndarray, prev_patch_gray: Optional[np.ndarray], 
                                use_sparse: bool = False, prev_features_for_sparse: Optional[np.ndarray] = None) -> Tuple[float, float, Optional[np.ndarray], Optional[np.ndarray]]:
        """Original sparse/dense flow calculation method from tracker.py"""
        if prev_patch_gray is None or current_patch_gray.shape != prev_patch_gray.shape:
            return 0.0, 0.0, None, None
            
        if use_sparse and prev_features_for_sparse is not None:
            # Sparse flow tracking (simplified implementation)
            try:
                import cv2
                lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                new_features, status, error = cv2.calcOpticalFlowPyrLK(prev_patch_gray, current_patch_gray, prev_features_for_sparse, None, **lk_params)
                
                if new_features is not None and status is not None:
                    good_features = new_features[status == 1]
                    good_old_features = prev_features_for_sparse[status == 1]
                    
                    if len(good_features) > 0:
                        motion_vectors = good_features - good_old_features
                        dx_raw = np.median(motion_vectors[:, 0]) if len(motion_vectors) > 0 else 0.0
                        dy_raw = np.median(motion_vectors[:, 1]) if len(motion_vectors) > 0 else 0.0
                        return dx_raw, dy_raw, None, good_features
                        
            except Exception as e:
                self.logger.error(f"Sparse flow calculation error: {e}", exc_info=True)
                
        return 0.0, 0.0, None, None
    
    def _calculate_flow_in_sub_regions(self, patch_gray: np.ndarray, prev_patch_gray: Optional[np.ndarray]) -> Tuple[float, float, float, float, Optional[np.ndarray]]:
        """
        Calculate optical flow using sophisticated sub-region analysis from original tracker.
        For VR videos, uses weighted histogram of flow for robust motion detection.
        """
        # Handle edge cases
        if self.flow_dense is None or prev_patch_gray is None or prev_patch_gray.shape != patch_gray.shape:
            return 0.0, 0.0, 0.0, 0.0, None

        prev_patch_cont = np.ascontiguousarray(prev_patch_gray)
        patch_cont = np.ascontiguousarray(patch_gray)

        # Calculate optical flow for the entire patch
        flow = self.flow_dense.calc(prev_patch_cont, patch_cont, None)
        if flow is None:
            return 0.0, 0.0, 0.0, 0.0, None

        h, w, _ = flow.shape
        is_vr_video = self._is_vr_video()

        # For VR, determine the dominant motion region
        dominant_flow_region = flow
        lower_magnitude = 0.0
        upper_magnitude = 0.0

        if is_vr_video and hasattr(self, 'inversion_detection_split_ratio'):
            split_ratio = getattr(self, 'inversion_detection_split_ratio', 3.0)
            if split_ratio > 1.0:
                lower_region_h = int(h / split_ratio)
                if lower_region_h > 0 and lower_region_h < h:
                    upper_region_flow_vertical = flow[0:h - lower_region_h, :, 1]
                    lower_region_flow_vertical = flow[h - lower_region_h:h, :, 1]
                    upper_magnitude = np.median(np.abs(upper_region_flow_vertical))
                    lower_magnitude = np.median(np.abs(lower_region_flow_vertical))

                    motion_threshold = getattr(self, 'motion_inversion_threshold', 1.5)
                    if lower_magnitude > upper_magnitude * motion_threshold:
                        dominant_flow_region = flow[h - lower_region_h:h, :, :]
                        self.logger.debug("Thrusting pattern dominant. Using lower ROI for flow calculation.")
                    elif upper_magnitude > lower_magnitude * motion_threshold:
                        dominant_flow_region = flow[0:h - lower_region_h, :, :]
                        self.logger.debug("Riding pattern dominant. Using upper ROI for flow calculation.")

        # Perform robust calculation on the selected dominant region
        region_h, region_w, _ = dominant_flow_region.shape
        overall_dx, overall_dy = 0.0, 0.0

        if is_vr_video:
            # VR-SPECIFIC: Magnitude-weighted flow with Gaussian spatial weighting
            # This combines magnitude weighting (to capture small moving objects like hands)
            # with Gaussian spatial weighting (to focus on center of ROI)

            # Calculate magnitude weights (prioritize moving pixels)
            magnitudes = np.sqrt(dominant_flow_region[..., 0]**2 + dominant_flow_region[..., 1]**2)

            # Create 2D Gaussian spatial weights (prioritize center of ROI)
            center_x, sigma_x = region_w / 2, region_w / 4.0
            center_y, sigma_y = region_h / 2, region_h / 4.0

            x_coords = np.arange(region_w)
            y_coords = np.arange(region_h)
            weights_x = np.exp(-((x_coords - center_x) ** 2) / (2 * sigma_x ** 2))
            weights_y = np.exp(-((y_coords - center_y) ** 2) / (2 * sigma_y ** 2))

            # Create 2D spatial weight matrix
            spatial_weights = np.outer(weights_y, weights_x)

            # Combine magnitude and spatial weights
            # This ensures: (1) moving pixels dominate, (2) center pixels are prioritized
            combined_weights = magnitudes * spatial_weights

            total_weight = np.sum(combined_weights)
            if total_weight > 0:
                # Weighted average of flow vectors
                overall_dy = np.sum(dominant_flow_region[..., 1] * combined_weights) / total_weight
                overall_dx = np.sum(dominant_flow_region[..., 0] * combined_weights) / total_weight
            else:
                # Fallback if no motion detected
                overall_dy = np.median(dominant_flow_region[..., 1])
                overall_dx = np.median(dominant_flow_region[..., 0])
        else:
            # 2D VIDEO: Magnitude-weighted average to capture small moving objects
            # This prevents motion dilution when small moving objects (hand, face) are
            # surrounded by large static regions (penis, background)
            magnitudes = np.sqrt(dominant_flow_region[..., 0]**2 + dominant_flow_region[..., 1]**2)
            total_magnitude = np.sum(magnitudes)

            if total_magnitude > 0:
                # Weight each flow vector by its magnitude - large motions dominate
                # This ensures hand/face motion isn't diluted by static background
                overall_dy = np.sum(dominant_flow_region[..., 1] * magnitudes) / total_magnitude
                overall_dx = np.sum(dominant_flow_region[..., 0] * magnitudes) / total_magnitude
            else:
                # Fallback if no motion detected at all
                overall_dy = np.median(dominant_flow_region[..., 1])
                overall_dx = np.median(dominant_flow_region[..., 0])

        return overall_dy, overall_dx, lower_magnitude, upper_magnitude, flow

    def _analyze_roi_motion(self, processed_frame: np.ndarray, current_roi_gray: np.ndarray,
                          prev_roi_gray: Optional[np.ndarray]) -> Tuple[int, int]:
        """Analyze motion within the ROI patch using the original sophisticated method."""
        if prev_roi_gray is None or current_roi_gray.shape != prev_roi_gray.shape:
            return 50, 50
        
        if not self.flow_dense:
            return 50, 50
        
        try:
            # Use the sophisticated sub-regions method from original tracker
            if self.use_sparse_flow:
                # Sparse flow method (simplified for now)
                flow = self.flow_dense.calc(
                    np.ascontiguousarray(prev_roi_gray), 
                    np.ascontiguousarray(current_roi_gray), 
                    None
                )
                if flow is None:
                    return 50, 50
                dx_raw = np.median(flow[..., 0])
                dy_raw = np.median(flow[..., 1])
                lower_mag, upper_mag = 0.0, 0.0
            else:
                # Use sophisticated sub-region analysis
                dy_raw, dx_raw, lower_mag, upper_mag, flow_field = self._calculate_flow_in_sub_regions(
                    current_roi_gray, prev_roi_gray
                )

            is_vr_video = self._is_vr_video()

            # Handle motion mode detection for VR
            if self.enable_inversion_detection and is_vr_video:
                from collections import Counter
                motion_threshold = getattr(self, 'motion_inversion_threshold', 1.5)
                current_dominant_motion = 'undetermined'
                
                if lower_mag > upper_mag * motion_threshold:
                    current_dominant_motion = 'thrusting'
                elif upper_mag > lower_mag * motion_threshold:
                    current_dominant_motion = 'riding'

                self.motion_mode_history.append(current_dominant_motion)
                
                if self.motion_mode_history:
                    most_common_mode, count = Counter(self.motion_mode_history).most_common(1)[0]
                    window_size = getattr(self, 'motion_mode_history_window', 30)
                    if count > window_size // 2 and self.motion_mode != most_common_mode:
                        self.logger.info(f"Motion mode switched: from '{self.motion_mode}' to '{most_common_mode}'.")
                        self.motion_mode = most_common_mode
            else:
                self.motion_mode = 'thrusting'  # Default non-inverting mode

            # Apply smoothing to raw values
            self.primary_flow_history_smooth.append(dy_raw)
            self.secondary_flow_history_smooth.append(dx_raw)
            
            dy_smooth = np.median(self.primary_flow_history_smooth) if self.primary_flow_history_smooth else dy_raw
            dx_smooth = np.median(self.secondary_flow_history_smooth) if self.secondary_flow_history_smooth else dx_raw

            # Calculate positions with proper scaling (EXACT original logic)
            size_factor = self.get_current_penis_size_factor()
            
            if self.adaptive_flow_scale:
                base_primary_pos = self._apply_adaptive_scaling(dy_smooth, "flow_min_primary_adaptive", "flow_max_primary_adaptive", size_factor, True)
                secondary_pos = self._apply_adaptive_scaling(dx_smooth, "flow_min_secondary_adaptive", "flow_max_secondary_adaptive", size_factor, False)
            else:
                effective_amp_factor = self._get_effective_amplification_factor()
                manual_scale_multiplier = (self._get_current_sensitivity() / 10.0) * (1.0 / size_factor) * effective_amp_factor
                base_primary_pos = int(np.clip(50 + dy_smooth * manual_scale_multiplier + self.y_offset, 0, 100))
                secondary_pos = int(np.clip(50 + dx_smooth * manual_scale_multiplier + self.x_offset, 0, 100))

            # Apply inversion logic: thrusting should be normal, riding should be inverted
            primary_pos = base_primary_pos if self.motion_mode == "thrusting" else 100 - base_primary_pos

            return primary_pos, secondary_pos
            
        except Exception as e:
            self.logger.error(f"ROI motion analysis error: {e}", exc_info=True)
            return 50, 50
    
    def get_current_penis_size_factor(self) -> float:
        """Calculate current penis size factor (EXACT original method)."""
        if not self.penis_max_size_history or not self.penis_last_known_box:
            return 1.0
        
        max_hist = max(self.penis_max_size_history)
        if max_hist < 1:
            return 1.0
            
        cur_size = self.penis_last_known_box[2] * self.penis_last_known_box[3]
        return np.clip(cur_size / max_hist, 0.1, 1.5)
    
    def _apply_adaptive_scaling(self, value: float, min_val_attr: str, max_val_attr: str, size_factor: float, is_primary: bool) -> int:
        """Security-compliant adaptive scaling method (avoiding setattr)"""
        # Direct attribute access instead of setattr to comply with security restrictions
        if min_val_attr == "flow_min_primary_adaptive":
            min_h = self.flow_min_primary_adaptive
            self.flow_min_primary_adaptive = min(min_h * 0.995, value * 0.9 if value < -0.1 else value * 1.1)
            min_h = min(self.flow_min_primary_adaptive, -0.2)
        else:  # flow_min_secondary_adaptive
            min_h = self.flow_min_secondary_adaptive
            self.flow_min_secondary_adaptive = min(min_h * 0.995, value * 0.9 if value < -0.1 else value * 1.1)
            min_h = min(self.flow_min_secondary_adaptive, -0.2)
            
        if max_val_attr == "flow_max_primary_adaptive":
            max_h = self.flow_max_primary_adaptive
            self.flow_max_primary_adaptive = max(max_h * 0.995, value * 1.1 if value > 0.1 else value * 0.9)
            max_h = max(self.flow_max_primary_adaptive, 0.2)
        else:  # flow_max_secondary_adaptive
            max_h = self.flow_max_secondary_adaptive
            self.flow_max_secondary_adaptive = max(max_h * 0.995, value * 1.1 if value > 0.1 else value * 0.9)
            max_h = max(self.flow_max_secondary_adaptive, 0.2)
            
        flow_range = max_h - min_h
        if abs(flow_range) < 0.1: 
            flow_range = np.sign(flow_range) * 0.1 if flow_range != 0 else 0.1
        normalized_centered_flow = (2 * (value - min_h) / flow_range) - 1.0 if flow_range != 0 else 0.0
        normalized_centered_flow = np.clip(normalized_centered_flow, -1.0, 1.0)
        effective_amp_factor = self._get_effective_amplification_factor()
        max_deviation = (self._get_current_sensitivity() / 2.5) * effective_amp_factor
        pos_offset = self.y_offset if is_primary else self.x_offset
        return int(np.clip(50 + normalized_centered_flow * max_deviation + pos_offset, 0, 100))

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
        
        # Add tracking indicator
        self._draw_tracking_indicator(processed_frame)
    
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
        """Calculate effective amplification factor with real-time settings updates."""
        # Read from app settings in real-time for control panel responsiveness
        if hasattr(self.app, 'app_settings') and self.app.app_settings:
            base_factor = self.app.app_settings.get('live_tracker_base_amplification', getattr(self, 'base_amplification_factor', 1.0))
            class_multipliers = self.app.app_settings.get('live_tracker_class_amp_multipliers', getattr(self, 'class_specific_amplification_multipliers', {}))
        else:
            base_factor = getattr(self, 'base_amplification_factor', 1.0)
            class_multipliers = getattr(self, 'class_specific_amplification_multipliers', {})
        
        if not class_multipliers:
            # Default multipliers from original constants
            class_multipliers = {
                'face': 1.0, 'hand': 1.0, 'finger': 1.0, 'breast': 1.0,
                'pussy': 1.2, 'ass': 1.2, 'dildo': 1.0
            }
        
        # Apply class-specific multiplier if main interaction class is set
        if self.main_interaction_class and self.main_interaction_class in class_multipliers:
            multiplier = class_multipliers[self.main_interaction_class]
            effective_factor = base_factor * multiplier
            return effective_factor
        
        return base_factor
    
    def _get_current_sensitivity(self) -> float:
        """Get current sensitivity with real-time settings updates."""
        if hasattr(self.app, 'app_settings') and self.app.app_settings:
            return self.app.app_settings.get('live_tracker_sensitivity', getattr(self, 'sensitivity', 10.0))
        else:
            return getattr(self, 'sensitivity', 10.0)
    
    def _get_current_confidence_threshold(self) -> float:
        """Get current confidence threshold with real-time settings updates."""
        if hasattr(self.app, 'app_settings') and self.app.app_settings:
            return self.app.app_settings.get('live_tracker_confidence_threshold', getattr(self, 'confidence_threshold', 0.7))
        else:
            return getattr(self, 'confidence_threshold', 0.7)
    
    def _is_vr_video(self) -> bool:
        """Detect if this is a VR video based on aspect ratio (cached)."""
        # Return cached result if available
        if self._is_vr_video_cached is not None:
            return self._is_vr_video_cached

        # Calculate and cache result
        try:
            if hasattr(self.app, 'get_video_dimensions'):
                width, height = self.app.get_video_dimensions()
                if width and height:
                    aspect_ratio = width / height
                    self._is_vr_video_cached = aspect_ratio >= VR_ASPECT_RATIO_THRESHOLD
                    return self._is_vr_video_cached
        except Exception:
            pass

        self._is_vr_video_cached = False
        return False

    def _update_flow_preset(self, preset: str):
        """Update optical flow preset configuration."""
        try:
            if not self.flow_dense:
                return
                
            preset_upper = preset.upper()
            if preset_upper == 'ULTRAFAST':
                self.flow_dense = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
            elif preset_upper == 'FAST':
                self.flow_dense = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
            elif preset_upper == 'MEDIUM':
                self.flow_dense = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            else:
                self.logger.warning(f"Unknown flow preset: {preset}, using ULTRAFAST")
                self.flow_dense = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
                
            self.logger.info(f"Updated DIS optical flow preset to: {preset_upper}")
        except Exception as e:
            self.logger.error(f"Failed to update flow preset: {e}")

    def update_dis_flow_config(self, preset=None, finest_scale=None):
        """Update DIS optical flow configuration (called by control panel)."""
        if preset:
            self._update_flow_preset(preset)
        
        if finest_scale is not None:
            try:
                # Update finest scale if the flow implementation supports it
                if hasattr(self.flow_dense, 'setFinestScale'):
                    if finest_scale == 0:
                        # Auto mode - let DIS choose
                        pass  
                    else:
                        self.flow_dense.setFinestScale(finest_scale)
                    self.logger.info(f"Updated DIS finest scale to: {finest_scale}")
            except Exception as e:
                self.logger.error(f"Failed to update finest scale: {e}", exc_info=True)
                
    # Methods for real-time setting updates (called by control panel)
    def update_oscillation_sensitivity(self):
        """Update oscillation detection sensitivity (compatibility method)."""
        # Not applicable for YOLO ROI but required for interface compatibility
        pass

    def _update_fps(self):
        """Update FPS calculation based on actual frame processing rate."""
        current_time = time.time()
        self._fps_counter += 1

        # Update FPS every FPS_UPDATE_FRAME_COUNT frames or every second, whichever comes first
        time_diff = current_time - self._fps_last_time
        if self._fps_counter >= FPS_UPDATE_FRAME_COUNT or time_diff >= 1.0:
            if time_diff > 0:
                self.current_fps = self._fps_counter / time_diff
                self._fps_counter = 0
                self._fps_last_time = current_time
        
    def update_oscillation_grid_size(self):
        """Update oscillation grid size (compatibility method).""" 
        # Not applicable for YOLO ROI but required for interface compatibility
        pass