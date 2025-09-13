#!/usr/bin/env python3
"""
Hybrid Intelligence Tracker - Multi-Modal Approach

This tracker combines frame differentiation, optical flow, YOLO detection, and 
oscillation analysis in an intelligent hybrid system. It uses frame differences 
to identify regions of change, applies YOLO for semantic understanding, computes 
selective optical flow only within changed areas, and weights signals based on 
detection priorities (genitals first).

The goal is to create the most accurate and responsive funscript signal by 
leveraging the strengths of each approach while minimizing computational overhead.

Author: VR Funscript AI Generator
Version: 1.0.0
"""

import logging
import time
import numpy as np
import cv2
from collections import deque
from typing import Dict, Any, Optional, List, Tuple, Set
import threading
import traceback
from dataclasses import dataclass

try:
    from ..core.base_tracker import BaseTracker, TrackerMetadata, TrackerResult
    from ..helpers.signal_amplifier import SignalAmplifier
    from ..helpers.visualization import (
        TrackerVisualizationHelper, BoundingBox, PoseKeypoints
    )
except ImportError:
    from tracker_modules.core.base_tracker import BaseTracker, TrackerMetadata, TrackerResult
    from tracker_modules.helpers.signal_amplifier import SignalAmplifier
    from tracker_modules.helpers.visualization import (
        TrackerVisualizationHelper, BoundingBox, PoseKeypoints
    )

import config.constants as constants


@dataclass
class ChangeRegion:
    """Represents a region where frame differences were detected."""
    x: int
    y: int
    width: int
    height: int
    area: int
    intensity: float  # Average difference intensity
    bbox: Tuple[int, int, int, int]  # (x, y, x+w, y+h)


@dataclass
class SemanticRegion:
    """Represents a YOLO-detected region with semantic information."""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    priority: int  # Higher = more important (genitals=10, hands=5, etc.)
    change_overlap: float = 0.0  # Overlap with change regions


@dataclass
class FlowAnalysis:
    """Results from optical flow analysis within a region."""
    region_id: int
    flow_magnitude: float
    flow_direction: np.ndarray  # Average flow vector
    oscillation_strength: float
    confidence: float


class HybridIntelligenceTracker(BaseTracker):
    """
    Multi-modal hybrid tracker combining:
    1. Frame Differentiation - Efficient change detection
    2. YOLO Detection - Semantic understanding 
    3. Selective Optical Flow - Precise motion in changed regions
    4. Oscillation Analysis - Rhythm pattern detection
    5. Intelligent Weighting - Priority-based signal fusion
    """
    
    @property
    def metadata(self) -> TrackerMetadata:
        return TrackerMetadata(
            name="hybrid_intelligence",
            display_name="Hybrid Intelligence Tracker",
            description="Multi-modal approach combining frame diff, optical flow, YOLO, and oscillation detection",
            category="live",
            version="1.0.0",
            author="VR Funscript AI Generator",
            tags=["hybrid", "intelligent", "multi-modal", "frame-diff", "optical-flow", "yolo", "oscillation"],
            requires_roi=False,
            supports_dual_axis=True
        )
    
    def initialize(self, app_instance, **kwargs) -> bool:
        """Initialize the hybrid intelligence tracker."""
        try:
            self.app = app_instance
            self.logger = logging.getLogger(self.__class__.__name__)
            
            # Core tracking state
            self.tracking_active = False
            self.current_fps = 30.0
            self.frame_count = 0
            self.current_oscillation_intensity = 0.0  # Initialize oscillation intensity for debug display
            
            # Initialize funscript connection
            if hasattr(self, 'funscript') and self.funscript:
                pass  # Already have funscript from bridge
            elif hasattr(self.app, 'funscript') and self.app.funscript:
                self.funscript = self.app.funscript
            else:
                from funscript.dual_axis_funscript import DualAxisFunscript
                self.funscript = DualAxisFunscript(logger=self.logger)
                self.logger.info("Created local funscript instance for Hybrid Intelligence")
            
            # Visual settings
            self.show_debug_overlay = kwargs.get('show_debug_overlay', True)
            self.show_change_regions = kwargs.get('show_change_regions', True)
            self.show_yolo_detections = kwargs.get('show_yolo_detections', True)
            self.show_flow_vectors = kwargs.get('show_flow_vectors', True)
            self.show_oscillation_grid = kwargs.get('show_oscillation_grid', True)
            
            # === 1. ENHANCED FRAME DIFFERENTIATION PARAMETERS ===
            self.frame_diff_threshold = kwargs.get('frame_diff_threshold', 12)  # Much more sensitive
            self.min_change_area = kwargs.get('min_change_area', 100)  # Smaller minimum area
            self.gaussian_blur_ksize = kwargs.get('gaussian_blur_ksize', 3)  # Less blurring for detail
            self.morphology_kernel_size = kwargs.get('morphology_kernel_size', 2)  # Smaller morphology
            self.adaptive_threshold_enabled = True  # Enable adaptive thresholding
            self.contact_detection_enabled = True  # Enable contact detection between persons and penis
            
            # === 2. YOLO DETECTION PARAMETERS ===
            self.yolo_model = None
            self.yolo_model_path = None
            self.yolo_base_update_interval = kwargs.get('yolo_base_interval', 3)  # Base: every 3 frames
            self.yolo_confidence_threshold = kwargs.get('yolo_confidence_threshold', 0.2)  # Lower threshold
            
            # Adaptive YOLO frequency based on motion intensity
            self.yolo_adaptive_frequency = kwargs.get('yolo_adaptive_frequency', True)
            self.yolo_motion_history = deque(maxlen=10)  # Track motion over 10 frames
            self.yolo_current_interval = self.yolo_base_update_interval
            self.yolo_high_motion_threshold = kwargs.get('yolo_high_motion_threshold', 0.5)
            self.yolo_low_motion_threshold = kwargs.get('yolo_low_motion_threshold', 0.1)
            
            # Semantic priorities (higher = more important)
            self.class_priorities = {
                'penis': 10, 'locked_penis': 10,
                'pussy': 9, 'vagina': 9,
                'ass': 8, 'anus': 8,
                'hand': 5, 'finger': 5,
                'mouth': 4, 'face': 3,
                'breast': 2, 'body': 1
            }
            
            # === 3. OPTICAL FLOW PARAMETERS ===
            self.flow_dense = None
            self.flow_update_method = kwargs.get('flow_update_method', 'selective')  # 'selective' or 'full'
            self.flow_grid_size = kwargs.get('flow_grid_size', 16)  # Subsampling for performance
            self.flow_window_size = kwargs.get('flow_window_size', 15)  # LK flow window
            
            # === 4. OSCILLATION DETECTION PARAMETERS ===
            self.oscillation_grid_size = kwargs.get('oscillation_grid_size', 10)
            self.oscillation_sensitivity = kwargs.get('oscillation_sensitivity', 1.2)
            self.oscillation_history_max_len = kwargs.get('oscillation_history_max_len', 60)
            
            # === 5. SIGNAL FUSION PARAMETERS ===
            self.signal_fusion_method = kwargs.get('signal_fusion_method', 'weighted_average')
            self.temporal_smoothing_alpha = kwargs.get('temporal_smoothing_alpha', 0.25)
            
            # Initialize components
            self._init_frame_differentiation()
            self._init_yolo_detection()
            self._init_optical_flow()
            self._init_pose_estimation()
            self._init_oscillation_detection()
            self._init_signal_processing()
            
            # State tracking
            self.prev_frame_gray = None
            self.current_frame_gray = None
            self.change_regions: List[ChangeRegion] = []
            self.semantic_regions: List[SemanticRegion] = []
            self.flow_analyses: List[FlowAnalysis] = []
            
            # History tracking for graphical debug window
            self.position_history = deque(maxlen=120)  # 2-4 seconds at 30-60fps
            self.change_regions_history = deque(maxlen=60)  # 1-2 seconds
            self.flow_intensity_history = deque(maxlen=60)  # 1-2 seconds
            
            # Performance monitoring
            self.processing_times = deque(maxlen=30)
            
            # Modern overlay system is now implemented
            self.logger.info("Hybrid Intelligence Tracker initialized with modern overlay system")
            
            # Visualization system configuration
            self.use_external_visualization = False  # Use internal overlays by default
            self.debug_window_enabled = True
            self.logger.info("Debug window enabled for rich tracker visualization")
            
            self.logger.info("Hybrid Intelligence Tracker initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Hybrid Intelligence Tracker: {e}")
            return False
    
    def _init_frame_differentiation(self):
        """Initialize frame differentiation components."""
        # Create morphological kernels
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.morphology_kernel_size, self.morphology_kernel_size)
        )
        
        # Gaussian blur kernel
        self.blur_ksize = (self.gaussian_blur_ksize, self.gaussian_blur_ksize)
        
        self.logger.debug("Frame differentiation initialized")
    
    def _init_yolo_detection(self):
        """Initialize YOLO detection system."""
        try:
            # Get YOLO model path from various sources (similar to yolo_roi tracker)
            yolo_model_path = None
            
            # First: Check app settings for YOLO detection model path
            if hasattr(self.app, 'app_settings') and self.app.app_settings:
                yolo_model_path = self.app.app_settings.get('yolo_det_model_path', None)
                if yolo_model_path:
                    self.logger.info(f"Using YOLO model path from app settings: {yolo_model_path}")
                else:
                    self.logger.debug("No yolo_det_model_path found in app_settings")
            else:
                self.logger.debug("No app_settings available")
            
            # Fallback: Check app instance for various model path attributes
            if not yolo_model_path and hasattr(self.app, 'tracker_model_path'):
                yolo_model_path = self.app.tracker_model_path
                self.logger.debug(f"Found tracker_model_path: {yolo_model_path}")
            elif not yolo_model_path and hasattr(self.app, 'det_model_path'):
                yolo_model_path = self.app.det_model_path
                self.logger.debug(f"Found det_model_path: {yolo_model_path}")
            elif not yolo_model_path and hasattr(self.app, 'yolo_model_path'):
                yolo_model_path = self.app.yolo_model_path
                self.logger.debug(f"Found yolo_model_path: {yolo_model_path}")
            
            # Fallback: Check processor for model path
            if not yolo_model_path and hasattr(self.app, 'processor'):
                if hasattr(self.app.processor, 'tracker_model_path'):
                    yolo_model_path = self.app.processor.tracker_model_path
                    self.logger.debug(f"Found processor.tracker_model_path: {yolo_model_path}")
                elif hasattr(self.app.processor, 'det_model_path'):
                    yolo_model_path = self.app.processor.det_model_path
                    self.logger.debug(f"Found processor.det_model_path: {yolo_model_path}")
            
            self.logger.debug(f"Final yolo_model_path: {yolo_model_path}")
            
            # Load the YOLO model if we found a path
            if yolo_model_path:
                import os
                model_exists = False
                
                # Check for .mlpackage (Core ML - directory) or regular file
                if os.path.isdir(yolo_model_path) and yolo_model_path.endswith('.mlpackage'):
                    model_exists = True
                    self.logger.debug(f"Found Core ML model package (directory): {yolo_model_path}")
                elif os.path.isfile(yolo_model_path):
                    model_exists = True  
                    self.logger.debug(f"Found model file: {yolo_model_path}")
                else:
                    self.logger.warning(f"Model path does not exist: {yolo_model_path}")
                    # Check what actually exists at that location
                    parent_dir = os.path.dirname(yolo_model_path) 
                    if os.path.exists(parent_dir):
                        files = os.listdir(parent_dir)
                        self.logger.debug(f"Files in {parent_dir}: {files[:5]}...")  # Show first 5
                
                if model_exists:
                    try:
                        from ultralytics import YOLO
                        self.yolo_model = YOLO(yolo_model_path, task='detect')
                        self.logger.info(f"YOLO model loaded successfully from: {yolo_model_path}")
                        
                        # Load class names
                        names_attr = getattr(self.yolo_model, 'names', None)
                        if names_attr:
                            if isinstance(names_attr, dict):
                                self.classes = list(names_attr.values())
                            else:
                                self.classes = list(names_attr)
                            self.logger.info(f"Loaded {len(self.classes)} classes: {self.classes[:10]}...")  # Show first 10 classes
                    except Exception as e:
                        self.logger.error(f"Failed to load YOLO model: {e}")
                        self.yolo_model = None
                        self.classes = []
                        return False
                else:
                    self.logger.warning(f"YOLO model not found: {yolo_model_path}")
                    self.yolo_model = None
                    self.classes = []
            else:
                self.logger.warning("No YOLO model path found - object detection will be disabled")
                self.yolo_model = None
                self.classes = []
            
        except Exception as e:
            self.logger.error(f"YOLO initialization failed: {e}")
            self.yolo_model = None
            self.classes = []
        
        # Always initialize these variables regardless of YOLO success/failure
        self.yolo_frame_counter = 0
        self.last_yolo_detections = []
    
    def _init_optical_flow(self):
        """Initialize optical flow components - OPTIMIZED for DIS ULTRAFAST."""
        try:
            # FORCE CPU DIS ULTRAFAST - it's 10-50x faster than GPU Farneback
            self.use_gpu_flow = False
            self.flow_dense = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
            self.gpu_flow = None
            
            self.logger.info("🚀 OPTIMIZED: Using CPU DIS ULTRAFAST (10-50x faster than GPU Farneback)")
            
            # Note: GPU Farneback is intentionally disabled because:
            # - cv2.cuda.FarnebackOpticalFlow is extremely slow (dense algorithm)
            # - CPU DIS ULTRAFAST consistently outperforms GPU Farneback by orders of magnitude
            # - This eliminates the major performance bottleneck we discovered
            
            # LK optical flow parameters for sparse tracking
            self.lk_params = dict(
                winSize=(self.flow_window_size, self.flow_window_size),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            
            # Feature detection parameters
            self.feature_params = dict(
                maxCorners=100,
                qualityLevel=0.3,
                minDistance=7,
                blockSize=7
            )
            
            # GPU memory for optical flow
            if self.use_gpu_flow:
                self.gpu_frame1 = None
                self.gpu_frame2 = None
                self.gpu_flow_result = None
            
        except Exception as e:
            self.logger.error(f"Failed to initialize optical flow: {e}")
            self.flow_dense = None
            self.use_gpu_flow = False
    
    def _init_pose_estimation(self):
        """Initialize YOLO pose estimation for body keypoint tracking."""
        # Initialize all attributes first
        self.pose_model = None
        self.pose_available = False
        self.mp_pose = None  # Keep for compatibility
        self.mp_drawing = None  # Keep for compatibility
        self.pose_landmarks = None
        
        # YOLO pose keypoint indices (COCO format) - complete mapping
        # 0:nose, 1:left_eye, 2:right_eye, 3:left_ear, 4:right_ear,
        # 5:left_shoulder, 6:right_shoulder, 7:left_elbow, 8:right_elbow,
        # 9:left_wrist, 10:right_wrist, 11:left_hip, 12:right_hip,
        # 13:left_knee, 14:right_knee, 15:left_ankle, 16:right_ankle
        self.keypoint_indices = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2, 
            'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6,
            'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10,
            'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14,
            'left_ankle': 15, 'right_ankle': 16
        }
        
        # Penis tracking persistence system
        self.penis_box_history = deque(maxlen=150)  # 5 seconds @ 30fps - longer persistence
        self.penis_size_history = deque(maxlen=900)  # 30 seconds @ 30fps for 95th percentile calculation
        
        # === STAGE 2 DERIVED LOCKED PENIS SYSTEM ===
        # Constants for more stable locking behavior
        self.PENIS_PATIENCE = 300  # Frames to wait before releasing lock (10s @ 30fps) - very long persistence
        self.PENIS_MIN_DETECTIONS = 3  # Minimum detections needed for activation (more confidence required)  
        self.PENIS_DETECTION_WINDOW = 90  # Frames window to evaluate detections (3s @ 30fps) - longer window
        self.PENIS_IOU_THRESHOLD = 0.1  # IoU threshold for tracking continuity
        self.PENIS_CONFIDENCE_THRESHOLD = 0.4  # Minimum confidence for new locks
        
        # Separate thresholds for activation vs deactivation (hysteresis)
        self.PENIS_ACTIVATION_MIN_DETECTIONS = 3  # Need 3+ detections in window to activate
        self.PENIS_DEACTIVATION_MIN_DETECTIONS = 1  # Only deactivate if < 1 detection in window AND patience exceeded
        
        # Height adaptation using 95th percentile over large window (like Stage 2)
        self.PENIS_SIZE_ANALYSIS_WINDOW_SECONDS = 30  # Window for height percentile calculation (larger for stability)
        self.PENIS_SIZE_UPDATE_INTERVAL = 60  # Only update evolved size every N frames (2 sec at 30fps)
        self.PENIS_SIZE_MIN_SAMPLES = 15  # Minimum samples needed for reliable 95th percentile
        
        # THREAD SAFETY: Add lock for penis tracking state
        self._penis_state_lock = threading.RLock()  # Reentrant lock for nested access
        
        # Tracker state (protected by _penis_state_lock)
        self.locked_penis_tracker = {
            'box': None,  # Current locked penis box (x1, y1, x2, y2)
            'confidence': 0.0,  # Confidence of locked penis
            'unseen_frames': 0,  # Frames since last detection
            'total_detections': 0,  # Total detections in current window
            'detection_frames': deque(maxlen=self.PENIS_DETECTION_WINDOW),  # Frame numbers where detected
            'last_seen_timestamp': 0.0,
            'active': False,  # Is tracking currently active
            'established_frame': None,  # Frame when lock was first established
            # Size evolution based on 90th percentile
            'current_size': None,  # (width, height, area)
            'base_size': None,  # Initial established size 
            'evolved_size': None  # 90th percentile evolved size
        }
        
        # Legacy compatibility (protected by _penis_state_lock)
        self.locked_penis_box = None  # Will be set from locked_penis_tracker
        self.locked_penis_last_seen = 0  
        self.locked_penis_persistence_duration = 10.0  
        self.penis_tracker_confidence = 0.0
        self.primary_person_pose_id = None
        self.pose_person_history = {}  # Track multiple people over time
        
        # Anatomical region tracking
        self.anatomical_regions = {
            'face': {'keypoints': ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear'], 'activity': 0.0},
            'breasts': {'center_estimate': None, 'activity': 0.0, 'movement_history': deque(maxlen=10)},
            'navel': {'center_estimate': None, 'activity': 0.0, 'movement_history': deque(maxlen=10)},
            'hands': {'positions': [], 'activity': 0.0},
            'torso': {'center': None, 'stability': 0.0}
        }
        
        try:
            # Try to get YOLO pose model path from app settings
            yolo_pose_model_path = self.app.app_settings.get('yolo_pose_model_path', None)
            self.logger.debug(f"Checking YOLO pose model path: {yolo_pose_model_path}")
            
            if yolo_pose_model_path:
                import os
                if os.path.exists(yolo_pose_model_path):
                    from ultralytics import YOLO
                    self.pose_model = YOLO(yolo_pose_model_path, task='pose')
                    self.pose_available = True
                    self.logger.info(f"YOLO pose estimation initialized from: {yolo_pose_model_path}")
                else:
                    self.logger.info(f"YOLO pose model not found at {yolo_pose_model_path}")
            else:
                self.logger.info("YOLO pose model path not available, pose estimation disabled")
            
        except ImportError:
            self.pose_available = False
            self.logger.warning("Ultralytics not available. Pose estimation disabled.")
        except Exception as e:
            self.pose_available = False
            self.pose_model = None
            self.logger.warning(f"YOLO pose estimation initialization failed: {e}")
    
    def _init_oscillation_detection(self):
        """Initialize oscillation detection system."""
        # Initialize empty history first for safety
        self.oscillation_history: Dict[Tuple[int, int], deque] = {}
        self.oscillation_persistence: Dict[Tuple[int, int], int] = {}
        
        # Grid-based oscillation tracking
        self.oscillation_block_size = constants.YOLO_INPUT_SIZE // self.oscillation_grid_size
        
        # Oscillation analysis state
        self.last_oscillation_peaks = deque(maxlen=10)
        self.oscillation_frequency_estimate = 0.0
        
        self.logger.debug("Oscillation detection initialized")
    
    def _init_signal_processing(self):
        """Initialize signal processing and fusion components."""
        # Enhanced signal amplifier
        self.signal_amplifier = SignalAmplifier(
            history_size=120,  # 4 seconds @ 30fps
            enable_live_amp=True,
            smoothing_alpha=self.temporal_smoothing_alpha,
            logger=self.logger
        )
        
        # Signal history for temporal consistency
        self.primary_signal_history = deque(maxlen=30)
        self.secondary_signal_history = deque(maxlen=30)
        
        # Fusion weights (will be dynamically adjusted)
        self.fusion_weights = {
            'frame_diff': 0.2,
            'yolo_weighted': 0.4,
            'optical_flow': 0.3,
            'oscillation': 0.1
        }
        
        # Final output state
        self.last_primary_position = 50.0
        self.last_secondary_position = 50.0
        
        self.logger.debug("Signal processing initialized")
    
    def start_tracking(self) -> bool:
        """Start the tracking session."""
        try:
            self.tracking_active = True
            self.frame_count = 0
            
            # Reset state
            self.prev_frame_gray = None
            self.current_frame_gray = None
            self.change_regions = []
            self.semantic_regions = []
            self.flow_analyses = []
            
            # Reset signal history
            self.primary_signal_history.clear()
            self.secondary_signal_history.clear()
            
            # Reset oscillation tracking
            self.oscillation_history.clear()
            self.oscillation_persistence.clear()
            
            # Reset penis tracking state (thread-safe)
            with self._penis_state_lock:
                self.penis_box_history.clear()
                self.locked_penis_box = None
                self.locked_penis_last_seen = 0
                # Reset tracked state dictionary
                self.locked_penis_tracker.update({
                    'box': None,
                    'confidence': 0.0,
                    'unseen_frames': 0,
                    'total_detections': 0,
                    'last_seen_timestamp': 0.0,
                    'active': False,
                    'established_frame': None,
                    'current_size': None,
                    'base_size': None,
                    'evolved_size': None
                })
                self.locked_penis_tracker['detection_frames'].clear()
            
            # Reset performance tracking
            self.processing_times.clear()
            
            self.logger.info("Hybrid Intelligence Tracker started")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start tracking: {e}")
            return False
    
    def stop_tracking(self) -> bool:
        """Stop the tracking session and maintain final positions."""
        try:
            self.tracking_active = False
            
            # Important: Do NOT reset positions to 50 - maintain final tracking state
            # This prevents decay when tracking stops at end of video
            
            # Log final positions if available
            primary_pos = getattr(self, 'last_primary_position', 50.0)
            secondary_pos = getattr(self, 'last_secondary_position', 50.0)
            self.logger.info(f"Hybrid Intelligence Tracker stopped at final positions: P={primary_pos:.1f}, S={secondary_pos:.1f}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to stop tracking: {e}")
            return False
    
    def _perform_memory_cleanup(self):
        """Periodic memory cleanup to prevent unbounded growth."""
        try:
            # CRITICAL FIX: Aggressive oscillation history cleanup
            if hasattr(self, 'oscillation_history') and self.oscillation_history:
                max_history = 50  # Reduced from 200 to prevent memory explosion
                current_count = len(self.oscillation_history)
                
                if current_count > max_history:
                    # Smart cleanup: keep only cells with recent activity and good history
                    cells_by_activity = []
                    
                    for cell_key, history in self.oscillation_history.items():
                        if len(history) > 5:  # Only consider cells with some data
                            # Calculate activity score (recent variance + mean motion)
                            recent_history = list(history)[-10:] if len(history) >= 10 else list(history)
                            activity_score = np.std(recent_history) + np.mean(recent_history)
                            cells_by_activity.append((cell_key, activity_score, len(history)))
                    
                    # Sort by activity score (descending) and keep top entries
                    cells_by_activity.sort(key=lambda x: x[1], reverse=True)
                    keys_to_keep = set(cell[0] for cell in cells_by_activity[:max_history])
                    
                    # Remove inactive cells
                    keys_to_remove = [k for k in self.oscillation_history.keys() if k not in keys_to_keep]
                    removed_count = len(keys_to_remove)
                    
                    for key in keys_to_remove:
                        del self.oscillation_history[key]
                    
                    if removed_count > 100:  # Log major cleanups
                        self.logger.info(f"Memory cleanup: removed {removed_count} inactive oscillation cells")
            
            # Limit processing times history (check if exists)
            if hasattr(self, 'processing_times') and self.processing_times:
                if len(self.processing_times) > 100:
                    self.processing_times = self.processing_times[-50:]  # Keep last 50
                    
            # Limit signal histories (check if exists)
            if hasattr(self, 'primary_signal_history') and self.primary_signal_history:
                if len(self.primary_signal_history) > 300:
                    self.primary_signal_history = self.primary_signal_history[-150:]
                    
            if hasattr(self, 'secondary_signal_history') and self.secondary_signal_history:
                if len(self.secondary_signal_history) > 300:
                    self.secondary_signal_history = self.secondary_signal_history[-150:]
                
        except Exception as e:
            self.logger.warning(f"Memory cleanup error: {e}")

    def cleanup(self):
        """Clean up resources."""
        try:
            self.stop_tracking()
            
            # Perform final memory cleanup
            self._perform_memory_cleanup()
            
            # Clear GPU resources  
            if hasattr(self, 'gpu_flow') and self.gpu_flow:
                del self.gpu_flow
            
            # Clear pose model
            if hasattr(self, 'pose_model') and self.pose_model:
                del self.pose_model
            
            if hasattr(self, 'signal_amplifier'):
                # Signal amplifier cleanup is handled internally
                pass
                
            self.logger.info("Hybrid Intelligence Tracker cleaned up")
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
    
    # Modern overlay system methods
    def get_overlay_data(self) -> Dict[str, Any]:
        """
        Get overlay data for frame rendering.
        
        Returns rich visualization data prepared by TrackerVisualizationHelper
        for drawing on the video frame, including detections, regions, flow vectors, 
        and tracking state.
        
        Returns:
            Dict containing all overlay visualization data
        """
        return getattr(self, 'overlay_data', {
            'yolo_boxes': [],
            'poses': [], 
            'change_regions': [],
            'flow_vectors': [],
            'motion_mode': 'hybrid',
            'locked_penis_box': None,
            'contact_info': {},
            'oscillation_grid_active': False,
            'oscillation_sensitivity': 1.0,
            'tracking_active': self.tracking_active
        })
    
    def get_debug_window_data(self) -> Dict[str, Any]:
        """
        Get debug window data for external rendering.
        
        Returns structured debug information organized into metrics,
        progress bars, and status information for display.
        
        Returns:
            Dict containing debug window visualization data
        """
        return getattr(self, 'debug_window_data', {
            'tracker_name': 'Hybrid Intelligence',
            'metrics': {'Status': {'Initializing': 'Please wait...'}},
            'progress_bars': {},
            'show_graphs': False,
            'graphs': None
        })
    
    def process_frame(self, frame: np.ndarray, frame_time_ms: int, 
                     frame_index: Optional[int] = None) -> TrackerResult:
        """
        Process a frame using the hybrid intelligence approach.
        
        Processing Pipeline:
        1. Frame Differentiation - Detect areas of change
        2. YOLO Detection - Identify semantic objects  
        3. Region Prioritization - Weight regions by importance
        4. Selective Optical Flow - Compute flow in priority regions
        5. Oscillation Analysis - Detect rhythmic patterns
        6. Signal Fusion - Combine all signals intelligently
        7. Temporal Smoothing - Apply smoothing and generate actions
        """
        start_time = time.time()
        
        try:
            self.frame_count += 1
            
            # Periodic memory cleanup every 30 frames for performance
            if self.frame_count % 30 == 0:
                self._perform_memory_cleanup()
            
            # Convert to grayscale for processing
            self.current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # === STEP 1: FRAME DIFFERENTIATION ===
            if self.prev_frame_gray is not None:
                self.change_regions = self._detect_frame_changes(
                    self.prev_frame_gray, self.current_frame_gray
                )
                
                # Track motion intensity for adaptive YOLO
                total_motion = sum(region.intensity * region.area for region in self.change_regions) / 1000000.0
                self.yolo_motion_history.append(min(total_motion, 2.0))  # Clamp to reasonable range
                
                # Update graphical debug history
                self.change_regions_history.append(self.change_regions.copy())
                flow_intensity = sum(analysis.flow_magnitude for analysis in self.flow_analyses)
                self.flow_intensity_history.append(flow_intensity)
            else:
                self.change_regions = []
            
            # === STEP 2: YOLO DETECTION ===
            should_run = self._should_run_yolo()
            if should_run:
                self.logger.debug(f"Running YOLO detection (frame {self.frame_count}, interval={self.yolo_current_interval})")
                self.semantic_regions = self._detect_semantic_objects(frame)
                self.logger.debug(f"YOLO detected {len(self.semantic_regions)} semantic regions")
            else:
                # Keep previous detections if not running YOLO this frame
                # self.semantic_regions already contains previous detections
                self.logger.debug(f"Skipping YOLO (frame {self.frame_count} % {self.yolo_current_interval} = {self.frame_count % self.yolo_current_interval}), keeping {len(self.semantic_regions)} previous regions")
            
            # === STEP 2.5: POSE ESTIMATION ===
            pose_data = self._estimate_pose(frame)
            
            # === STEP 3: REGION PRIORITIZATION ===
            priority_regions = self._prioritize_regions(pose_data)
            
            # === STEP 4: SELECTIVE OPTICAL FLOW ===
            if self.prev_frame_gray is not None and priority_regions:
                self.flow_analyses = self._compute_selective_flow(
                    self.prev_frame_gray, self.current_frame_gray, priority_regions
                )
            else:
                self.flow_analyses = []
            
            # === STEP 5: OSCILLATION ANALYSIS ===
            oscillation_signal = self._analyze_oscillation_patterns()
            self.current_oscillation_intensity = oscillation_signal  # Store for debug display
            
            # === STEP 6: SIGNAL FUSION ===
            # Store pose data for debug visualization
            self.last_pose_data = pose_data
            
            primary_pos, secondary_pos = self._fuse_signals(oscillation_signal, pose_data)
            
            # === STEP 7: GENERATE ACTIONS ===
            action_log = self._generate_funscript_actions(
                primary_pos, secondary_pos, frame_time_ms, frame_index
            )
            
            # === STEP 8: PREPARE OVERLAY DATA ===
            # Always prepare overlay data for external visualization
            self._prepare_overlay_data(priority_regions, pose_data)
            
            # Decide whether to draw on frame or return clean frame
            if getattr(self, 'use_external_visualization', False):
                display_frame = frame  # Return clean frame for external overlay
                boxes_count = len(self.overlay_data.get('yolo_boxes', []))
                poses_count = len(self.overlay_data.get('poses', []))
                change_regions_count = len(self.overlay_data.get('change_regions', []))
                flow_vectors_count = len(self.overlay_data.get('flow_vectors', []))
                
                self.logger.debug(f"Overlay prepared: {boxes_count} boxes, {poses_count} poses, {change_regions_count} regions, {flow_vectors_count} flows")
            else:
                # Use internal visualization - ensure overlays are visible
                display_frame = self._create_debug_overlay(frame.copy())
                
                # Debug logging to understand overlay status
                if hasattr(self, 'overlay_data') and self.overlay_data:
                    computation_count = len(self.overlay_data.get('computation_areas', []))
                    disregarded_count = len(self.overlay_data.get('disregarded_areas', []))
                    zones_count = len(self.overlay_data.get('interaction_zones', []))
                    
                    if computation_count > 0 or disregarded_count > 0 or zones_count > 0:
                        self.logger.debug(f"Overlays: {computation_count} computed, {disregarded_count} disregarded, {zones_count} zones")
                else:
                    self.logger.debug("No overlay data prepared")
            
            # Update state for next frame
            self.prev_frame_gray = self.current_frame_gray.copy()
            
            # Performance tracking
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Create result
            debug_info = self._generate_debug_info(processing_time)
            status_msg = f"Hybrid Intelligence: {len(priority_regions)} regions, {len(self.flow_analyses)} flows"
            
            # Update debug window data if enabled
            if self.debug_window_enabled:
                self._update_debug_window_data(processing_time)
            
            return TrackerResult(
                processed_frame=display_frame,
                action_log=action_log,
                debug_info=debug_info,
                status_message=status_msg
            )
            
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            # Return safe fallback
            return TrackerResult(
                processed_frame=frame,
                action_log=[],
                debug_info={"error": str(e)},
                status_message=f"Error: {str(e)}"
            )
    
    def _detect_frame_changes(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> List[ChangeRegion]:
        """🚀 OPTIMIZED: Detect regions of significant change using powerful numpy operations."""
        try:
            # VECTORIZED absolute difference - much faster than cv2.absdiff for grayscale
            diff = np.abs(curr_gray.astype(np.int16) - prev_gray.astype(np.int16)).astype(np.uint8)
            
            # VECTORIZED Gaussian blur - using OpenCV optimized filter
            diff_blurred = cv2.GaussianBlur(diff, self.blur_ksize, 0)
            
            # VECTORIZED thresholding - numpy comparison is extremely fast
            binary_mask = (diff_blurred > self.frame_diff_threshold).astype(np.uint8) * 255
            
            # OPTIMIZED morphological operations - OpenCV uses SIMD instructions
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, self.morph_kernel)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, self.morph_kernel)
            
            # Find contours to identify change regions
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # VECTORIZED contour processing - batch operations where possible
            change_regions = []
            
            if len(contours) > 0:
                # Pre-compute all bounding rects at once for cache efficiency
                bounding_rects = [cv2.boundingRect(contour) for contour in contours]
                areas = [cv2.contourArea(contour) for contour in contours]
                
                # Filter and process valid regions
                for i, (area, (x, y, w, h)) in enumerate(zip(areas, bounding_rects)):
                    if area < self.min_change_area:
                        continue
                    
                    # VECTORIZED intensity calculation - numpy mean is highly optimized
                    roi_diff = diff_blurred[y:y+h, x:x+w]
                    avg_intensity = np.mean(roi_diff)
                    
                    change_region = ChangeRegion(
                        x=x, y=y, width=w, height=h,
                        area=int(area),
                        intensity=float(avg_intensity),
                        bbox=(x, y, x+w, y+h)
                    )
                    
                    change_regions.append(change_region)
            
            return change_regions
            
        except Exception as e:
            self.logger.error(f"Error in optimized frame change detection: {e}")
            # Fallback to simple approach if optimization fails
            diff = cv2.absdiff(prev_gray, curr_gray)
            diff_blurred = cv2.GaussianBlur(diff, self.blur_ksize, 0)
            _, binary_mask = cv2.threshold(diff_blurred, self.frame_diff_threshold, 255, cv2.THRESH_BINARY)
            return []
    
    def _should_run_yolo(self) -> bool:
        """Determine if YOLO detection should run this frame with adaptive frequency."""
        self.yolo_frame_counter += 1
        
        if not self.yolo_adaptive_frequency:
            # Original fixed frequency
            return self.yolo_frame_counter % self.yolo_base_update_interval == 0
        
        # Update adaptive frequency based on recent motion
        if len(self.yolo_motion_history) >= 3:  # Need some history
            avg_motion = np.mean(list(self.yolo_motion_history))
            
            if avg_motion > self.yolo_high_motion_threshold:
                # High motion: Check every frame
                self.yolo_current_interval = 1
            elif avg_motion > self.yolo_low_motion_threshold:
                # Medium motion: Use base interval
                self.yolo_current_interval = self.yolo_base_update_interval
            else:
                # Low motion: Check every 10 frames (save processing)
                self.yolo_current_interval = 10
        
        return self.yolo_frame_counter % self.yolo_current_interval == 0
    
    def _detect_semantic_objects(self, frame: np.ndarray) -> List[SemanticRegion]:
        """Run YOLO detection to identify semantic objects."""
        semantic_regions = []
        
        try:
            if not self.yolo_model:
                self.logger.warning("No YOLO model available for detection - make sure model path is configured")
                return semantic_regions  # No model available
            
            # Run YOLO detection with proper parameters
            device = getattr(constants, 'DEVICE', 'auto')
            self.logger.debug(f"Calling YOLO model with device={device}, conf={self.yolo_confidence_threshold}")
            results = self.yolo_model(frame, device=device, verbose=False, conf=self.yolo_confidence_threshold)
            
            self.logger.debug(f"YOLO returned {len(results)} results")
            for i, result in enumerate(results):
                if hasattr(result, 'boxes') and result.boxes is not None:
                    self.logger.debug(f"Result {i}: {len(result.boxes)} boxes detected")
                    for box in result.boxes:
                        # Extract detection data
                        confidence = float(box.conf.item())
                        # Note: confidence already filtered by model call, but double-check
                        # if confidence < self.yolo_confidence_threshold:
                        #     continue
                        
                        class_id = int(box.cls.item())
                        class_name = self.yolo_model.names.get(class_id, f"class_{class_id}")
                        
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy.cpu().numpy().flatten()
                        
                        # Determine priority
                        priority = self.class_priorities.get(class_name.lower(), 0)
                        
                        semantic_region = SemanticRegion(
                            class_name=class_name,
                            confidence=confidence,
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            priority=priority
                        )
                        
                        semantic_regions.append(semantic_region)
                        
                        # Debug log for important detections
                        if class_name.lower() in ['penis', 'locked_penis', 'pussy', 'butt', 'hand', 'face']:
                            self.logger.debug(f"YOLO detected: {class_name} (conf={confidence:.2f}, priority={priority})")
        
        except Exception as e:
            self.logger.warning(f"YOLO detection failed: {e}")
        
        return semantic_regions
    
    def _estimate_pose(self, frame: np.ndarray) -> Dict[str, Any]:
        """Advanced multi-person pose estimation with penis-centric tracking."""
        pose_data = {
            'primary_person': None,
            'all_persons': [],
            'anatomical_activities': {},
            'penis_association_confidence': 0.0,
            'signal_components': {},
            'debug_info': {}
        }
        
        if not self.pose_available or not self.pose_model:
            return pose_data
            
        try:
            # Run YOLO pose estimation on all detected persons
            pose_results = self.pose_model(frame, verbose=False, conf=0.3)
            
            if not pose_results or pose_results[0].keypoints is None:
                return pose_data
                
            # Process all detected persons
            all_persons = []
            frame_h, frame_w = frame.shape[:2]
            
            for person_idx in range(len(pose_results[0].keypoints.data)):
                person_data = self._process_person_pose(
                    pose_results[0].keypoints.data[person_idx], 
                    person_idx, frame_w, frame_h
                )
                if person_data:
                    all_persons.append(person_data)
            
            pose_data['all_persons'] = all_persons
            
            # Find primary person based on penis proximity and persistence
            primary_person = self._determine_primary_person(all_persons, frame_w, frame_h)
            pose_data['primary_person'] = primary_person
            
            # Detect person-penis contact for all persons
            contact_info = self._detect_person_penis_contact(all_persons, frame_w, frame_h)
            pose_data['contact_info'] = contact_info
            
            if primary_person:
                # Analyze anatomical regions for the primary person
                anatomical_activities = self._analyze_anatomical_regions(primary_person, frame_w, frame_h)
                pose_data['anatomical_activities'] = anatomical_activities
                
                # Calculate comprehensive activity signals
                signal_components = self._calculate_pose_signals(primary_person, anatomical_activities)
                pose_data['signal_components'] = signal_components
                
                # Update penis association confidence
                pose_data['penis_association_confidence'] = self._calculate_penis_association(primary_person)
            
            # Debug information
            pose_data['debug_info'] = {
                'total_persons_detected': len(all_persons),
                'primary_person_id': primary_person['person_id'] if primary_person else None,
                'penis_box_history_size': len(self.penis_box_history),
                'anatomical_regions_active': len([r for r in pose_data.get('anatomical_activities', {}).values() if r.get('activity', 0) > 0.1])
            }
        
        except Exception as e:
            self.logger.warning(f"Pose estimation failed: {e}")
            pose_data['debug_info']['error'] = str(e)
        
        return pose_data
    
    def _process_person_pose(self, keypoints_data, person_idx: int, frame_w: int, frame_h: int) -> Dict[str, Any]:
        """Process a single person's pose keypoints."""
        try:
            keypoints = keypoints_data.cpu().numpy() if hasattr(keypoints_data, 'cpu') else keypoints_data
            
            # Extract keypoint positions with confidence filtering
            person_keypoints = {}
            total_confidence = 0
            valid_keypoints = 0
            
            for name, idx in self.keypoint_indices.items():
                if idx < len(keypoints):
                    x, y, conf = float(keypoints[idx][0]), float(keypoints[idx][1]), float(keypoints[idx][2])
                    
                    # Convert normalized coordinates to pixels if needed
                    if x <= 1.0 and y <= 1.0:
                        x, y = int(x * frame_w), int(y * frame_h)
                    else:
                        x, y = int(x), int(y)
                    
                    if conf > 0.3:  # Minimum confidence threshold
                        person_keypoints[name] = {'x': x, 'y': y, 'confidence': conf}
                        total_confidence += conf
                        valid_keypoints += 1
            
            # Only return person if we have enough valid keypoints
            if valid_keypoints < 8:  # Need at least 8 valid keypoints
                return None
            
            # Calculate person bounding box and center
            xs = [kp['x'] for kp in person_keypoints.values()]
            ys = [kp['y'] for kp in person_keypoints.values()]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            
            return {
                'person_id': person_idx,
                'keypoints': person_keypoints,
                'bbox': bbox,
                'center': center,
                'confidence': total_confidence / valid_keypoints,
                'valid_keypoints': valid_keypoints,
                'raw_keypoints': keypoints
            }
            
        except Exception as e:
            self.logger.warning(f"Error processing person {person_idx}: {e}")
            return None
    
    def _determine_primary_person(self, all_persons: list, frame_w: int, frame_h: int) -> Dict[str, Any]:
        """Determine which person is associated with the penis detection."""
        if not all_persons:
            return None
            
        # Update penis box from recent YOLO detections
        self._update_penis_box_tracking()
        
        if not self.penis_box_history:
            # No penis detected recently, use largest/most central person
            return max(all_persons, key=lambda p: p['confidence'] * p['valid_keypoints'])
        
        # Find person closest to current penis box
        current_penis_box = self.penis_box_history[-1]  # Most recent
        penis_center = (
            (current_penis_box['bbox'][0] + current_penis_box['bbox'][2]) // 2,
            (current_penis_box['bbox'][1] + current_penis_box['bbox'][3]) // 2
        )
        
        best_person = None
        best_distance = float('inf')
        
        for person in all_persons:
            # Calculate distance from person center to penis center
            person_center = person['center']
            distance = np.sqrt((person_center[0] - penis_center[0])**2 + (person_center[1] - penis_center[1])**2)
            
            # Weight by person confidence and keypoint count
            weighted_distance = distance / (person['confidence'] * person['valid_keypoints'] / 17)
            
            if weighted_distance < best_distance:
                best_distance = weighted_distance
                best_person = person
        
        # Update primary person tracking
        if best_person:
            self.primary_person_pose_id = best_person['person_id']
            
        return best_person
    
    def _update_penis_box_tracking(self):
        """Stage 2 derived locked penis tracking with IoU continuity and patience mechanism."""
        # THREAD SAFETY: Protect entire penis tracking state update
        with self._penis_state_lock:
            self._update_penis_box_tracking_internal()
    
    def _update_penis_box_tracking_internal(self):
        """Internal implementation of penis tracking (thread-safe)."""
        # Find current penis candidates
        penis_candidates = []
        target_classes = ['penis', 'locked_penis']  # Focus on actual penis detections
        
        for region in self.semantic_regions:
            if (region.class_name.lower() in target_classes and 
                region.confidence > 0.3 and 
                hasattr(region, 'bbox')):
                penis_candidates.append({
                    'bbox': region.bbox,
                    'confidence': region.confidence,
                    'class_name': region.class_name,
                    'area': (region.bbox[2] - region.bbox[0]) * (region.bbox[3] - region.bbox[1])
                })
        
        tracker = self.locked_penis_tracker
        current_time = time.time()
        selected_penis = None
        
        # === INTERMITTENT DETECTION TRACKING LOGIC ===
        # Always increment unseen_frames (will be reset if detection found)
        tracker['unseen_frames'] += 1
        
        # 1. Process any detected candidates
        if penis_candidates:
            best_candidate = None
            
            # If we have an active lock, try IoU matching first
            if tracker['box'] and tracker['active']:
                best_iou = 0
                for candidate in penis_candidates:
                    iou = TrackerVisualizationHelper.calculate_iou(tracker['box'], candidate['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_candidate = candidate
                
                # If IoU match is good, update existing lock
                if best_iou > self.PENIS_IOU_THRESHOLD:
                    selected_penis = best_candidate
                    tracker['box'] = selected_penis['bbox']
                    tracker['confidence'] = selected_penis['confidence']
                    tracker['unseen_frames'] = 0
                    tracker['last_seen_timestamp'] = current_time
                    tracker['detection_frames'].append(self.frame_count)
                    self.logger.debug(f"Locked penis continued: IoU={best_iou:.3f} conf={selected_penis['confidence']:.2f}")
            
            # If no existing lock or IoU match failed, select best new candidate
            if not selected_penis:
                # Select best candidate (highest confidence * area, but must meet minimum confidence)
                high_conf_candidates = [c for c in penis_candidates if c['confidence'] >= self.PENIS_CONFIDENCE_THRESHOLD]
                
                if high_conf_candidates:
                    best_candidate = max(high_conf_candidates, 
                                       key=lambda x: x['confidence'] * (x['area'] / 10000.0))
                    
                    # Update or establish lock
                    selected_penis = best_candidate
                    tracker['box'] = selected_penis['bbox']
                    tracker['confidence'] = selected_penis['confidence']
                    tracker['unseen_frames'] = 0
                    tracker['last_seen_timestamp'] = current_time
                    tracker['detection_frames'].append(self.frame_count)
                    
                    if not tracker['established_frame']:
                        tracker['established_frame'] = self.frame_count
                        self.logger.info(f"Penis lock candidate found: {selected_penis['class_name']} conf={selected_penis['confidence']:.2f}")
                    else:
                        self.logger.debug(f"Penis lock updated: conf={selected_penis['confidence']:.2f}")
        
        # 2. Determine if lock should be active based on detection history with hysteresis
        recent_detections = len([f for f in tracker['detection_frames'] 
                               if self.frame_count - f <= self.PENIS_DETECTION_WINDOW])
        
        # Hysteresis logic: different thresholds for activation vs deactivation
        if not tracker['active']:
            # ACTIVATION: Require strong evidence (multiple detections)
            if recent_detections >= self.PENIS_ACTIVATION_MIN_DETECTIONS and tracker['box']:
                tracker['active'] = True
                self.logger.info(f"Penis lock ACTIVATED: {recent_detections} detections in {self.PENIS_DETECTION_WINDOW} frame window")
        else:
            # DEACTIVATION: Only deactivate if BOTH conditions met:
            # 1. Very few recent detections AND 2. Patience period exceeded
            if (recent_detections < self.PENIS_DEACTIVATION_MIN_DETECTIONS and 
                tracker['unseen_frames'] > self.PENIS_PATIENCE):
                self.logger.info(f"Penis lock DEACTIVATED: only {recent_detections} detections in {self.PENIS_DETECTION_WINDOW} frames, unseen for {tracker['unseen_frames']} frames")
                tracker['active'] = False
                tracker['box'] = None
                tracker['detection_frames'].clear()
                tracker['established_frame'] = None
        
        # 4. Update legacy compatibility fields
        if tracker['active'] and tracker['box']:
            # Use evolved size box if available, otherwise fallback to raw detection
            display_box = self._get_evolved_penis_box() or tracker['box']
            
            self.locked_penis_box = {
                'bbox': display_box,
                'confidence': tracker['confidence'],
                'timestamp': tracker['last_seen_timestamp']
            }
            self.penis_tracker_confidence = tracker['confidence']
            self.locked_penis_last_seen = tracker['last_seen_timestamp']
            
            # Add to history for continuity - use evolved box for consistent size
            penis_box = {
                'bbox': display_box,
                'confidence': tracker['confidence'],
                'timestamp': current_time,
                'area': (display_box[2] - display_box[0]) * (display_box[3] - display_box[1])
            }
            if not self.penis_box_history or self._is_significantly_different_box(penis_box):
                self.penis_box_history.append(penis_box)
                
                # Track size evolution for 95th percentile calculation - use RAW detection box
                # Important: Use raw detection box here to avoid circular feedback with evolved size
                self._update_penis_size_evolution(tracker['box'])
        else:
            # No active lock - clear legacy fields
            self.locked_penis_box = None
            self.penis_tracker_confidence = 0.0
            
        # Note: Non-YOLO frames are now handled by the unseen_frames increment at the top
        # and the patience mechanism in the main logic above
                        
        # Debug logging for locked penis tracker status
        if self.frame_count % 60 == 0:  # Every 2 seconds at 30fps
            tracker = self.locked_penis_tracker
            recent_detections = len([f for f in tracker['detection_frames'] 
                                   if self.frame_count - f <= self.PENIS_DETECTION_WINDOW])
            
            activation_threshold = self.PENIS_ACTIVATION_MIN_DETECTIONS if not tracker['active'] else self.PENIS_DEACTIVATION_MIN_DETECTIONS
            self.logger.debug(f"Penis Lock: active={tracker['active']}, "
                           f"unseen={tracker['unseen_frames']}/{self.PENIS_PATIENCE}, "
                           f"recent_detections={recent_detections}/{activation_threshold}, "
                           f"window={self.PENIS_DETECTION_WINDOW}f, conf={tracker['confidence']:.2f}")
            
            if penis_candidates:
                self.logger.debug(f"Penis candidates: {[(c['class_name'], f'{c['confidence']:.2f}') for c in penis_candidates]}")
            elif tracker['active']:
                self.logger.debug("Locked penis active but no candidates this frame (intermittent YOLO)")
    
    def _is_significantly_different_box(self, new_box: Dict) -> bool:
        """Check if new penis box is significantly different from recent ones."""
        if not self.penis_box_history:
            return True
            
        recent_box = self.penis_box_history[-1]
        
        # Calculate IoU (Intersection over Union)
        x1 = max(new_box['bbox'][0], recent_box['bbox'][0])
        y1 = max(new_box['bbox'][1], recent_box['bbox'][1])
        x2 = min(new_box['bbox'][2], recent_box['bbox'][2])
        y2 = min(new_box['bbox'][3], recent_box['bbox'][3])
        
        if x1 >= x2 or y1 >= y2:
            return True  # No overlap, definitely different
            
        intersection = (x2 - x1) * (y2 - y1)
        area1 = new_box['area']
        area2 = recent_box['area']
        union = area1 + area2 - intersection
        
        iou = intersection / union if union > 0 else 0
        
        return iou < 0.7  # Consider different if IoU < 70%
    
    def _update_penis_size_evolution(self, bbox: Tuple[float, float, float, float]):
        """Update penis size tracking for 90th percentile evolution."""
        if not bbox:
            return
            
        # Calculate current size metrics
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        
        size_metrics = {
            'width': width,
            'height': height,
            'area': area,
            'timestamp': time.time(),
            'frame': self.frame_count
        }
        
        # Add to size history
        self.penis_size_history.append(size_metrics)
        
        tracker = self.locked_penis_tracker
        tracker['current_size'] = (width, height, area)
        
        # Set base size on first establishment
        if not tracker['base_size'] and tracker['established_frame']:
            tracker['base_size'] = (width, height, area)
            self.logger.info(f"Penis base size established: {width:.1f}x{height:.1f} (area: {area:.0f})")
        
        # Update evolved size based on 95th percentile over 20-second window (like Stage 2)
        # Update less frequently to prevent overly frequent height adaptation
        if (len(self.penis_size_history) >= 30 and  # Need at least 1 second of data
            self.frame_count % self.PENIS_SIZE_UPDATE_INTERVAL == 0):  # Only update every N frames
            result = self._calculate_95th_percentile_size()
            if result:
                new_evolved_size, sample_count = result
                old_height = tracker['evolved_size'][1] if tracker['evolved_size'] else 0
                new_height = new_evolved_size[1]
                tracker['evolved_size'] = new_evolved_size
                self.logger.debug(f"Frame {self.frame_count}: Updated evolved penis height: {old_height:.1f} -> {new_height:.1f} (95th percentile over {self.PENIS_SIZE_ANALYSIS_WINDOW_SECONDS}s, {sample_count} samples)")
    
    def _calculate_95th_percentile_size(self) -> Tuple[float, float, float]:
        """
        Calculate 95th percentile size from penis size history over large window.
        Uses the same stable approach as Stage 2 for maximum consistency.
        """
        if len(self.penis_size_history) < self.PENIS_SIZE_MIN_SAMPLES:
            return None
        
        # Get current time and calculate window cutoff    
        current_time = time.time()
        window_cutoff = current_time - self.PENIS_SIZE_ANALYSIS_WINDOW_SECONDS
        
        # Filter history to only include entries within the time window
        recent_history = [
            s for s in self.penis_size_history 
            if s.get('timestamp', 0) >= window_cutoff
        ]
        
        if len(recent_history) < self.PENIS_SIZE_MIN_SAMPLES:  # Need sufficient samples for reliable percentile
            return None
            
        # Extract size metrics from recent history
        heights = [s['height'] for s in recent_history]
        
        # Calculate 95th percentile height (main focus for penis tracking)
        import numpy as np
        height_95th = np.percentile(heights, 95)
        
        # Use current width and calculate area based on 95th percentile height
        if recent_history:
            current_width = recent_history[-1]['width']  # Use most recent width
            area_95th = current_width * height_95th
            return ((current_width, height_95th, area_95th), len(recent_history))
        
        return None
    
    def _get_evolved_penis_box(self) -> Optional[Tuple[float, float, float, float]]:
        """Get penis box adjusted to 95th percentile size if available."""
        tracker = self.locked_penis_tracker
        
        if not tracker['box'] or not tracker['evolved_size']:
            return tracker['box']  # Return current box if no evolution data
        
        current_box = tracker['box']
        evolved_width, evolved_height, _ = tracker['evolved_size']
        
        # Calculate center of current box
        center_x = (current_box[0] + current_box[2]) / 2
        center_y = (current_box[1] + current_box[3]) / 2
        
        # Create evolved box centered on current position
        half_width = evolved_width / 2
        half_height = evolved_height / 2
        
        evolved_box = (
            center_x - half_width,
            center_y - half_height,
            center_x + half_width,
            center_y + half_height
        )
        
        return evolved_box
    
    def _analyze_anatomical_regions(self, person: Dict, frame_w: int, frame_h: int) -> Dict[str, Dict]:
        """Analyze activity in different anatomical regions."""
        regions = {
            'face': self._analyze_face_activity(person),
            'breasts': self._analyze_breast_activity(person, frame_w, frame_h),
            'navel': self._analyze_navel_activity(person, frame_w, frame_h),
            'hands': self._analyze_hand_activity(person),
            'torso': self._analyze_torso_stability(person)
        }
        
        return regions
    
    def _analyze_face_activity(self, person: Dict) -> Dict:
        """Analyze facial movement and activity."""
        face_keypoints = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear']
        face_points = []
        
        for kp_name in face_keypoints:
            if kp_name in person['keypoints']:
                kp = person['keypoints'][kp_name]
                face_points.append([kp['x'], kp['y']])
        
        if len(face_points) < 3:
            return {'activity': 0.0, 'center': None, 'movement': 0.0}
            
        # Calculate face center
        face_center = np.mean(face_points, axis=0)
        
        # Update face tracking history
        if 'face' not in self.pose_person_history:
            self.pose_person_history['face'] = deque(maxlen=15)
            
        self.pose_person_history['face'].append(face_center)
        
        # Calculate movement based on recent history
        movement = 0.0
        if len(self.pose_person_history['face']) > 5:
            recent_positions = list(self.pose_person_history['face'])[-5:]
            movements = []
            for i in range(1, len(recent_positions)):
                dist = np.linalg.norm(np.array(recent_positions[i]) - np.array(recent_positions[i-1]))
                movements.append(dist)
            movement = np.mean(movements) if movements else 0.0
        
        return {
            'activity': min(movement / 20.0, 1.0),  # Normalize to 0-1
            'center': face_center.tolist(),
            'movement': movement
        }
    
    def _analyze_breast_activity(self, person: Dict, frame_w: int, frame_h: int) -> Dict:
        """Estimate and analyze breast region activity."""
        # Estimate breast center from shoulder and torso keypoints
        shoulders = []
        if 'left_shoulder' in person['keypoints']:
            shoulders.append(person['keypoints']['left_shoulder'])
        if 'right_shoulder' in person['keypoints']:
            shoulders.append(person['keypoints']['right_shoulder'])
            
        if len(shoulders) < 2:
            return {'activity': 0.0, 'center': None, 'movement': 0.0}
            
        # Estimate breast center (below shoulder line, above navel)
        shoulder_center = np.mean([[s['x'], s['y']] for s in shoulders], axis=0)
        
        # Adjust downward for breast region (approximately 15% of torso height)
        torso_height = frame_h * 0.4  # Approximate torso height
        breast_center = [shoulder_center[0], shoulder_center[1] + torso_height * 0.15]
        
        # Track movement
        if 'breasts' not in self.pose_person_history:
            self.pose_person_history['breasts'] = deque(maxlen=10)
            
        self.pose_person_history['breasts'].append(breast_center)
        
        # Calculate activity based on movement
        movement = 0.0
        if len(self.pose_person_history['breasts']) > 3:
            recent = list(self.pose_person_history['breasts'])[-3:]
            movements = []
            for i in range(1, len(recent)):
                dist = np.linalg.norm(np.array(recent[i]) - np.array(recent[i-1]))
                movements.append(dist)
            movement = np.mean(movements) if movements else 0.0
        
        return {
            'activity': min(movement / 15.0, 1.0),
            'center': breast_center,
            'movement': movement
        }
    
    def _analyze_navel_activity(self, person: Dict, frame_w: int, frame_h: int) -> Dict:
        """Estimate and analyze navel region activity."""
        # Estimate navel from hip and shoulder positions
        hips, shoulders = [], []
        
        for side in ['left', 'right']:
            if f'{side}_hip' in person['keypoints']:
                hips.append(person['keypoints'][f'{side}_hip'])
            if f'{side}_shoulder' in person['keypoints']:
                shoulders.append(person['keypoints'][f'{side}_shoulder'])
        
        if len(hips) < 1 or len(shoulders) < 1:
            return {'activity': 0.0, 'center': None, 'movement': 0.0}
            
        hip_center = np.mean([[h['x'], h['y']] for h in hips], axis=0)
        shoulder_center = np.mean([[s['x'], s['y']] for s in shoulders], axis=0)
        
        # Navel is approximately 60% down from shoulders to hips
        navel_center = shoulder_center + 0.6 * (hip_center - shoulder_center)
        
        # Track movement
        if 'navel' not in self.pose_person_history:
            self.pose_person_history['navel'] = deque(maxlen=10)
            
        self.pose_person_history['navel'].append(navel_center)
        
        # Calculate activity
        movement = 0.0
        if len(self.pose_person_history['navel']) > 3:
            recent = list(self.pose_person_history['navel'])[-3:]
            movements = []
            for i in range(1, len(recent)):
                dist = np.linalg.norm(np.array(recent[i]) - np.array(recent[i-1]))
                movements.append(dist)
            movement = np.mean(movements) if movements else 0.0
        
        return {
            'activity': min(movement / 12.0, 1.0),
            'center': navel_center.tolist(),
            'movement': movement
        }
    
    def _analyze_hand_activity(self, person: Dict) -> Dict:
        """Analyze hand positions and movement."""
        hands = []
        for side in ['left', 'right']:
            if f'{side}_wrist' in person['keypoints']:
                wrist = person['keypoints'][f'{side}_wrist']
                hands.append([wrist['x'], wrist['y'], wrist['confidence']])
        
        if not hands:
            return {'activity': 0.0, 'positions': [], 'movement': 0.0}
            
        # Track hand movement
        if 'hands' not in self.pose_person_history:
            self.pose_person_history['hands'] = deque(maxlen=10)
            
        self.pose_person_history['hands'].append(hands)
        
        # Calculate hand activity
        movement = 0.0
        if len(self.pose_person_history['hands']) > 3:
            recent_hands = list(self.pose_person_history['hands'])[-3:]
            all_movements = []
            
            for hand_idx in range(len(hands)):
                hand_movements = []
                for frame_idx in range(1, len(recent_hands)):
                    if hand_idx < len(recent_hands[frame_idx]) and hand_idx < len(recent_hands[frame_idx-1]):
                        curr_pos = np.array(recent_hands[frame_idx][hand_idx][:2])
                        prev_pos = np.array(recent_hands[frame_idx-1][hand_idx][:2])
                        dist = np.linalg.norm(curr_pos - prev_pos)
                        hand_movements.append(dist)
                
                if hand_movements:
                    all_movements.append(np.mean(hand_movements))
            
            movement = np.mean(all_movements) if all_movements else 0.0
        
        return {
            'activity': min(movement / 25.0, 1.0),
            'positions': hands,
            'movement': movement
        }
    
    def _analyze_torso_stability(self, person: Dict) -> Dict:
        """Analyze overall torso stability."""
        torso_points = []
        for kp_name in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']:
            if kp_name in person['keypoints']:
                kp = person['keypoints'][kp_name]
                torso_points.append([kp['x'], kp['y']])
        
        if len(torso_points) < 3:
            return {'stability': 0.0, 'center': None}
            
        torso_center = np.mean(torso_points, axis=0)
        
        # Track torso stability
        if 'torso' not in self.pose_person_history:
            self.pose_person_history['torso'] = deque(maxlen=15)
            
        self.pose_person_history['torso'].append(torso_center)
        
        # Calculate stability (inverse of movement)
        stability = 1.0
        if len(self.pose_person_history['torso']) > 5:
            recent_positions = list(self.pose_person_history['torso'])[-5:]
            movements = []
            for i in range(1, len(recent_positions)):
                dist = np.linalg.norm(np.array(recent_positions[i]) - np.array(recent_positions[i-1]))
                movements.append(dist)
            avg_movement = np.mean(movements) if movements else 0.0
            stability = max(0.0, 1.0 - (avg_movement / 30.0))  # More movement = less stability
        
        return {
            'stability': stability,
            'center': torso_center.tolist()
        }
    
    def _calculate_pose_signals(self, person: Dict, anatomical_activities: Dict) -> Dict:
        """Calculate comprehensive pose-based signals."""
        signals = {
            'face_activity': anatomical_activities.get('face', {}).get('activity', 0.0),
            'breast_activity': anatomical_activities.get('breasts', {}).get('activity', 0.0),
            'navel_activity': anatomical_activities.get('navel', {}).get('activity', 0.0),
            'hand_activity': anatomical_activities.get('hands', {}).get('activity', 0.0),
            'torso_stability': anatomical_activities.get('torso', {}).get('stability', 1.0),
            'overall_body_activity': 0.0
        }
        
        # Calculate overall body activity as weighted combination
        activity_weights = {
            'face_activity': 0.15,
            'breast_activity': 0.25,
            'navel_activity': 0.20,
            'hand_activity': 0.30,
            'torso_stability': 0.10  # Inverse weight since stability is opposite of activity
        }
        
        total_activity = 0.0
        for signal_name, weight in activity_weights.items():
            if signal_name == 'torso_stability':
                total_activity += weight * (1.0 - signals[signal_name])  # Invert stability
            else:
                total_activity += weight * signals[signal_name]
        
        signals['overall_body_activity'] = min(total_activity, 1.0)
        
        return signals
    
    def _calculate_penis_association(self, person: Dict) -> float:
        """Calculate confidence that this person is associated with the penis."""
        if not self.penis_box_history:
            return 0.0
            
        # Calculate average distance between person center and penis boxes
        person_center = person['center']
        distances = []
        
        for penis_box in self.penis_box_history:
            penis_center = (
                (penis_box['bbox'][0] + penis_box['bbox'][2]) // 2,
                (penis_box['bbox'][1] + penis_box['bbox'][3]) // 2
            )
            distance = np.sqrt((person_center[0] - penis_center[0])**2 + (person_center[1] - penis_center[1])**2)
            distances.append(distance)
        
        avg_distance = np.mean(distances)
        
        # Convert distance to confidence (closer = higher confidence)
        # Normalize by frame diagonal
        frame_diagonal = np.sqrt(640**2 + 640**2)  # Assuming 640x640 frame
        normalized_distance = avg_distance / frame_diagonal
        
        confidence = max(0.0, 1.0 - (normalized_distance * 2))  # Scale factor of 2
        
        return confidence
    
    def _detect_person_penis_contact(self, all_persons: list, frame_w: int, frame_h: int) -> Dict[str, Any]:
        """Detect which persons are in contact with the penis."""
        contact_info = {
            'persons_in_contact': [],
            'contact_scores': {},
            'locked_penis_box': None,
            'contact_regions': []
        }
        
        # Use persistent locked penis box if available and recent (thread-safe)
        penis_state = self._get_penis_state()
        penis_box_to_use = None
        
        if self._is_penis_active():
            penis_box_to_use = penis_state['locked_penis_box']
            contact_info['locked_penis_box'] = penis_state['locked_penis_box']
        elif self.penis_box_history:
            # Fallback to most recent detection
            recent_penis_box = self.penis_box_history[-1]
            penis_box_to_use = recent_penis_box
            contact_info['locked_penis_box'] = recent_penis_box
        else:
            return contact_info
        
        if not all_persons:
            return contact_info
        
        # Check each person for contact with penis
        if isinstance(penis_box_to_use, dict) and 'bbox' in penis_box_to_use:
            penis_bbox = penis_box_to_use['bbox']
        else:
            penis_bbox = penis_box_to_use  # Assume it's already a bbox tuple
        penis_x1, penis_y1, penis_x2, penis_y2 = penis_bbox
        
        for person in all_persons:
            person_bbox = person['bbox']
            person_x1, person_y1, person_x2, person_y2 = person_bbox
            
            # Calculate overlap between person and penis
            overlap_x1 = max(penis_x1, person_x1)
            overlap_y1 = max(penis_y1, person_y1)
            overlap_x2 = min(penis_x2, person_x2)
            overlap_y2 = min(penis_y2, person_y2)
            
            if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                # There is overlap
                overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                penis_area = (penis_x2 - penis_x1) * (penis_y2 - penis_y1)
                person_area = (person_x2 - person_x1) * (person_y2 - person_y1)
                
                # Contact score based on overlap relative to smaller bounding box
                min_area = min(penis_area, person_area)
                contact_score = overlap_area / min_area if min_area > 0 else 0
                
                if contact_score > 0.1:  # Minimum 10% overlap for contact
                    contact_info['persons_in_contact'].append(person['person_id'])
                    contact_info['contact_scores'][person['person_id']] = contact_score
                    
                    # Add contact region
                    contact_region = {
                        'person_id': person['person_id'],
                        'overlap_bbox': (overlap_x1, overlap_y1, overlap_x2, overlap_y2),
                        'contact_score': contact_score
                    }
                    contact_info['contact_regions'].append(contact_region)
        
        return contact_info

    def _prioritize_regions(self, pose_data: Dict[str, Any] = None) -> List[Tuple[ChangeRegion, List[SemanticRegion]]]:
        """Combine and prioritize change regions with semantic information."""
        priority_regions = []
        
        for change_region in self.change_regions:
            overlapping_semantics = []
            
            # Find semantic regions that overlap with this change region
            for semantic_region in self.semantic_regions:
                overlap = self._calculate_bbox_overlap(change_region.bbox, semantic_region.bbox)
                if overlap > 0.1:  # At least 10% overlap
                    semantic_region.change_overlap = overlap
                    overlapping_semantics.append(semantic_region)
            
            # Sort by priority (highest first)
            overlapping_semantics.sort(key=lambda x: x.priority, reverse=True)
            
            priority_regions.append((change_region, overlapping_semantics))
        
        # Sort regions by combined priority score
        def priority_score(region_tuple):
            change_region, semantics = region_tuple
            if not semantics:
                return change_region.intensity  # Base on change intensity
            
            # Weighted score: semantic priority * overlap * confidence + change intensity
            max_semantic_score = max(
                sem.priority * sem.change_overlap * sem.confidence
                for sem in semantics
            )
            return max_semantic_score + (change_region.intensity * 0.1)
        
        priority_regions.sort(key=priority_score, reverse=True)
        
        return priority_regions
    
    def _calculate_bbox_overlap(self, bbox1: Tuple[int, int, int, int], 
                               bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate overlap ratio between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
    
    def _compute_selective_flow(self, prev_gray: np.ndarray, curr_gray: np.ndarray,
                               priority_regions: List[Tuple[ChangeRegion, List[SemanticRegion]]]) -> List[FlowAnalysis]:
        """Compute optical flow selectively within priority regions."""
        flow_analyses = []
        
        if not self.flow_dense:
            return flow_analyses
        
        try:
            # Process more regions since we have CPU/GPU headroom
            for i, (change_region, semantics) in enumerate(priority_regions[:10]):  # Process top 10 for better coverage (was 5)
                # Extract ROI
                x, y, w, h = change_region.x, change_region.y, change_region.width, change_region.height
                
                # Ensure bounds are within frame
                frame_h, frame_w = prev_gray.shape
                x = max(0, min(x, frame_w - 1))
                y = max(0, min(y, frame_h - 1))
                w = min(w, frame_w - x)
                h = min(h, frame_h - y)
                
                if w < 10 or h < 10:  # Skip tiny regions
                    continue
                
                prev_roi = prev_gray[y:y+h, x:x+w].copy()  # Ensure contiguous
                curr_roi = curr_gray[y:y+h, x:x+w].copy()  # Ensure contiguous
                
                # 🚀 OPTIMIZED: CPU DIS ULTRAFAST optical flow (10-50x faster than GPU Farneback)
                flow = self.flow_dense.calc(prev_roi, curr_roi, None)
                
                # 🚀 VECTORIZED flow analysis - numpy operations are highly optimized
                # Calculate magnitude using vectorized operations
                magnitude = np.linalg.norm(flow, axis=2)  # Faster than manual sqrt
                avg_magnitude = np.mean(magnitude)
                avg_direction = np.mean(flow, axis=(0, 1))
                
                # Simple oscillation detection within region
                flow_std = np.std(magnitude)
                oscillation_strength = min(flow_std / (avg_magnitude + 1e-6), 2.0)
                
                # Calculate confidence based on semantic information
                confidence = 0.5  # Base confidence
                if semantics:
                    # Boost confidence for high-priority semantic objects
                    max_priority_sem = max(semantics, key=lambda x: x.priority)
                    semantic_boost = (max_priority_sem.priority / 10.0) * max_priority_sem.confidence
                    confidence = min(confidence + semantic_boost, 1.0)
                
                flow_analysis = FlowAnalysis(
                    region_id=i,
                    flow_magnitude=avg_magnitude,
                    flow_direction=avg_direction,
                    oscillation_strength=oscillation_strength,
                    confidence=confidence
                )
                
                flow_analyses.append(flow_analysis)
        
        except Exception as e:
            self.logger.warning(f"Selective optical flow computation failed: {e}")
        
        return flow_analyses
    
    def _analyze_oscillation_patterns(self) -> float:
        """Advanced oscillation analysis using FFT for frequency domain detection."""
        if self.prev_frame_gray is None or self.current_frame_gray is None:
            return 0.0
        
        try:
            # SMART OPTIMIZATION: Grid-based oscillation with movement-focused analysis
            frame_h, frame_w = self.current_frame_gray.shape
            
            # Pre-compute frame difference for movement detection (PERFORMANCE BOOST)
            frame_diff = cv2.absdiff(self.prev_frame_gray, self.current_frame_gray)
            
            # SMART THRESHOLDING: Different thresholds for different zones
            base_movement_threshold = 5  # Base threshold for interaction zones
            global_movement_threshold = 15  # Higher threshold for non-interaction areas
            
            # Get interaction regions for person-penis proximity (SMART LIMITING)
            interaction_regions = self._get_interaction_regions()
            
            oscillation_scores = []
            processed_cells = 0
            skipped_cells = 0
            
            for grid_y in range(0, frame_h - self.oscillation_block_size, self.oscillation_block_size):
                for grid_x in range(0, frame_w - self.oscillation_block_size, self.oscillation_block_size):
                    # MOVEMENT-FOCUSED OPTIMIZATION: Skip areas with no significant movement
                    y1, y2 = grid_y, grid_y + self.oscillation_block_size
                    x1, x2 = grid_x, grid_x + self.oscillation_block_size
                    
                    # Check if this cell has enough movement to warrant FFT analysis
                    cell_movement = np.mean(frame_diff[y1:y2, x1:x2])
                    
                    # INTERACTION-BASED OPTIMIZATION: Different thresholds for different zones
                    cell_center = (grid_x + self.oscillation_block_size // 2, grid_y + self.oscillation_block_size // 2)
                    is_in_interaction_zone = self._is_cell_in_interaction_zone(cell_center, interaction_regions)
                    
                    if is_in_interaction_zone:
                        # Lower threshold for interaction zones - catch subtle oscillations
                        if cell_movement < base_movement_threshold:
                            skipped_cells += 1
                            continue  # Skip FFT for static interaction areas
                    else:
                        # Higher threshold for non-interaction areas - be more selective
                        if cell_movement < global_movement_threshold:
                            skipped_cells += 1
                            continue  # Skip FFT for static non-interaction areas
                    
                    # Extract grid cell for analysis
                    motion_score = cell_movement / 255.0
                    
                    # Track oscillation history for this cell (with cleanup)
                    cell_key = (grid_x, grid_y)
                    if cell_key not in self.oscillation_history:
                        self.oscillation_history[cell_key] = deque(maxlen=self.oscillation_history_max_len)
                    
                    self.oscillation_history[cell_key].append(motion_score)
                    processed_cells += 1
                    
                    # Advanced FFT-based oscillation analysis
                    if len(self.oscillation_history[cell_key]) >= 30:  # Need enough history for FFT
                        history = np.array(list(self.oscillation_history[cell_key]))
                        
                        # Apply windowing to reduce spectral leakage
                        windowed_history = history * np.hanning(len(history))
                        
                        # Compute FFT
                        fft = np.fft.fft(windowed_history)
                        freqs = np.fft.fftfreq(len(history), d=1.0/30.0)  # Assuming 30 fps
                        
                        # Focus on oscillation frequencies (0.5-5 Hz typical for sexual activity)
                        valid_freq_mask = (np.abs(freqs) >= 0.5) & (np.abs(freqs) <= 5.0)
                        
                        if np.any(valid_freq_mask):
                            # Find dominant frequency in the valid range
                            power_spectrum = np.abs(fft[valid_freq_mask])
                            max_power = np.max(power_spectrum)
                            
                            # Calculate oscillation strength based on:
                            # 1. Peak power in frequency domain
                            # 2. Ratio of peak power to mean power (spectral contrast)
                            mean_power = np.mean(power_spectrum)
                            spectral_contrast = max_power / (mean_power + 1e-6)
                            
                            # Oscillation score combines both factors
                            oscillation_score = min((max_power / len(history)) * spectral_contrast, 2.0)
                            oscillation_scores.append(oscillation_score)
                        
                        else:
                            # Fallback to simple std dev if no valid frequencies
                            oscillation_score = min(np.std(history) * motion_score * 2, 1.0)
                            oscillation_scores.append(oscillation_score)
                    
                    elif len(self.oscillation_history[cell_key]) > 10:
                        # Simple fallback for insufficient FFT data
                        history = np.array(list(self.oscillation_history[cell_key]))
                        oscillation_score = min(np.std(history) * motion_score * 2, 1.0)
                        oscillation_scores.append(oscillation_score)
            
            # Log optimization stats occasionally (time-based to avoid frame_count dependency)
            total_cells = processed_cells + skipped_cells
            current_time = time.time()
            if hasattr(self, '_last_optimization_log_time'):
                if current_time - self._last_optimization_log_time > 10.0:  # Every 10 seconds
                    skip_percent = (skipped_cells / total_cells * 100) if total_cells > 0 else 0
                    self.logger.info(f"Smart optimization: skipped {skip_percent:.1f}% of cells, processed {processed_cells}/{total_cells}")
                    self._last_optimization_log_time = current_time
            else:
                self._last_optimization_log_time = current_time
            
            # Return average oscillation across all cells
            return np.mean(oscillation_scores) if oscillation_scores else 0.0
            
        except Exception as e:
            self.logger.warning(f"Advanced oscillation analysis failed, using fallback: {e}")
            return self._analyze_oscillation_fallback()
    
    def _analyze_oscillation_fallback(self) -> float:
        """Simple fallback oscillation analysis when FFT fails."""
        try:
            oscillation_scores = []
            
            for cell_key, history in self.oscillation_history.items():
                if len(history) > 5:
                    history_array = np.array(list(history))
                    oscillation_score = np.std(history_array) * np.mean(history_array)
                    oscillation_scores.append(min(oscillation_score, 1.0))
            
            return np.mean(oscillation_scores) if oscillation_scores else 0.0
            
        except Exception as e:
            self.logger.warning(f"Even fallback oscillation analysis failed: {e}")
            return 0.0
    
    def _get_interaction_regions(self) -> List[Tuple[int, int, int, int]]:
        """Get regions where person interactions with locked penis occur."""
        interaction_regions = []
        
        try:
            # Add locked penis region as primary interaction zone (thread-safe access)
            locked_penis_box = self._get_penis_state('locked_penis_box')
            if locked_penis_box and isinstance(locked_penis_box, dict) and 'bbox' in locked_penis_box:
                x1, y1, x2, y2 = locked_penis_box['bbox']
                # Expand region for interaction zone (50% larger)
                expand = 0.25
                w, h = x2 - x1, y2 - y1
                interaction_regions.append((
                    int(x1 - w * expand),
                    int(y1 - h * expand), 
                    int(x2 + w * expand),
                    int(y2 + h * expand)
                ))
            
            # Add regions around detected body parts that might interact with penis
            if hasattr(self, 'current_detections') and self.current_detections:
                for detection in self.current_detections:
                    class_name = detection.get('class_name', '').lower()
                    # Focus on body parts that commonly interact with penis
                    if class_name in ['hand', 'finger', 'mouth', 'pussy', 'butt', 'breast']:
                        bbox = detection.get('bbox')
                        if bbox:
                            x1, y1, x2, y2 = bbox
                            # Add smaller expansion for body parts
                            expand = 0.15
                            w, h = x2 - x1, y2 - y1
                            interaction_regions.append((
                                int(x1 - w * expand),
                                int(y1 - h * expand),
                                int(x2 + w * expand), 
                                int(y2 + h * expand)
                            ))
            
            # Add pose keypoint regions if available
            if hasattr(self, 'current_pose_data') and self.current_pose_data:
                pose_data = self.current_pose_data
                if pose_data.get('primary_person') and pose_data['primary_person'].get('keypoints'):
                    keypoints = pose_data['primary_person']['keypoints']
                    # Focus on hands, hips, torso keypoints
                    for kp_name in ['left_wrist', 'right_wrist', 'left_hip', 'right_hip']:
                        if kp_name in keypoints and keypoints[kp_name]['confidence'] > 0.3:
                            kp = keypoints[kp_name]
                            x, y = int(kp['x']), int(kp['y'])
                            # Add small region around keypoint
                            size = 50
                            interaction_regions.append((x - size, y - size, x + size, y + size))
        
        except Exception as e:
            self.logger.warning(f"Error getting interaction regions: {e}")
        
        return interaction_regions
    
    def _is_cell_in_interaction_zone(self, cell_center: Tuple[int, int], interaction_regions: List[Tuple[int, int, int, int]]) -> bool:
        """Check if a cell center is within any interaction zone."""
        cell_x, cell_y = cell_center
        
        for x1, y1, x2, y2 in interaction_regions:
            if x1 <= cell_x <= x2 and y1 <= cell_y <= y2:
                return True
        
        return False
    
    # THREAD SAFETY: Helper methods for safe penis state access
    def _get_penis_state(self, key: str = None):
        """Thread-safe getter for penis tracking state."""
        with self._penis_state_lock:
            if key is None:
                # Return copy of entire state to prevent external modifications
                return {
                    'locked_penis_box': self.locked_penis_box,
                    'locked_penis_last_seen': self.locked_penis_last_seen,
                    'penis_tracker_confidence': self.penis_tracker_confidence,
                    'locked_penis_tracker': dict(self.locked_penis_tracker)  # Shallow copy
                }
            elif key == 'locked_penis_box':
                return self.locked_penis_box
            elif key == 'locked_penis_last_seen':
                return self.locked_penis_last_seen
            elif key == 'locked_penis_tracker':
                return dict(self.locked_penis_tracker)  # Shallow copy
            else:
                return getattr(self, key, None)
    
    def _update_penis_state(self, updates: Dict[str, Any]):
        """Thread-safe updater for penis tracking state."""
        with self._penis_state_lock:
            for key, value in updates.items():
                if key == 'locked_penis_box':
                    self.locked_penis_box = value
                elif key == 'locked_penis_last_seen':
                    self.locked_penis_last_seen = value
                elif key == 'penis_tracker_confidence':
                    self.penis_tracker_confidence = value
                elif key.startswith('locked_penis_tracker.'):
                    # Handle nested updates like 'locked_penis_tracker.active'
                    nested_key = key.split('.', 1)[1]
                    self.locked_penis_tracker[nested_key] = value
                else:
                    # Explicit attribute assignment (security-safe)
                    if key == 'primary_person_pose_id':
                        self.primary_person_pose_id = value
                    elif key == 'penis_tracker_confidence':
                        self.penis_tracker_confidence = value
                    # Add other specific attributes as needed
                    else:
                        self.logger.warning(f"Attempted to update unknown penis state key: {key}")
    
    def _is_penis_active(self) -> bool:
        """Thread-safe check if locked penis is currently active."""
        with self._penis_state_lock:
            current_time = time.time()
            return (self.locked_penis_box is not None and 
                    (current_time - self.locked_penis_last_seen) < self.locked_penis_persistence_duration)
    
    def _fuse_signals(self, oscillation_signal: float, pose_data: Dict[str, Any] = None) -> Tuple[float, float]:
        """Intelligently fuse all signal sources to generate primary/secondary positions."""
        
        # Initialize signal components
        frame_diff_signal = 0.0
        yolo_weighted_signal = 0.0
        optical_flow_signal = 0.0
        pose_activity_signal = 0.0
        
        # === 1. FRAME DIFF SIGNAL ===
        if self.change_regions:
            # Weight by both intensity and area
            weighted_changes = []
            for region in self.change_regions:
                area_weight = min(region.area / 10000.0, 1.0)  # Normalize area
                intensity_weight = region.intensity / 255.0   # Normalize intensity
                weighted_changes.append(area_weight * intensity_weight)
            
            frame_diff_signal = min(sum(weighted_changes), 1.0)
        
        # === 2. YOLO WEIGHTED SIGNAL ===
        if self.semantic_regions:
            weighted_detections = []
            for sem_region in self.semantic_regions:
                weight = sem_region.priority * sem_region.confidence
                weighted_detections.append(weight)
            
            if weighted_detections:
                yolo_weighted_signal = min(sum(weighted_detections) / 50.0, 1.0)  # Normalize
        
        # === 3. OPTICAL FLOW SIGNAL ===
        if self.flow_analyses:
            weighted_flows = []
            for flow_analysis in self.flow_analyses:
                flow_contribution = flow_analysis.flow_magnitude * flow_analysis.confidence
                weighted_flows.append(flow_contribution)
            
            if weighted_flows:
                optical_flow_signal = min(np.mean(weighted_flows) * 10.0, 1.0)  # Normalize
        
        # === 4. COMPREHENSIVE POSE SIGNALS ===
        pose_signals = {}
        penis_confidence = 0.0
        
        if pose_data and pose_data.get('primary_person'):
            primary_person = pose_data['primary_person']
            anatomical_activities = pose_data.get('anatomical_activities', {})
            signal_components = pose_data.get('signal_components', {})
            penis_confidence = pose_data.get('penis_association_confidence', 0.0)
            
            # Extract individual anatomical signals
            pose_signals = {
                'face_activity': signal_components.get('face_activity', 0.0),
                'breast_activity': signal_components.get('breast_activity', 0.0),
                'navel_activity': signal_components.get('navel_activity', 0.0),
                'hand_activity': signal_components.get('hand_activity', 0.0),
                'torso_stability': signal_components.get('torso_stability', 1.0),
                'overall_body_activity': signal_components.get('overall_body_activity', 0.0)
            }
        
        # === 5. SOPHISTICATED SIGNAL FUSION ===
        # Calculate weighted signals with penis association confidence
        penis_weight = max(0.3, penis_confidence)  # Minimum 30% weight even without penis detection
        
        # Traditional signals
        base_signals = {
            'frame_diff': frame_diff_signal * self.fusion_weights['frame_diff'],
            'yolo_weighted': yolo_weighted_signal * self.fusion_weights['yolo_weighted'], 
            'optical_flow': optical_flow_signal * self.fusion_weights['optical_flow'],
            'oscillation': oscillation_signal * self.fusion_weights['oscillation']
        }
        
        # Advanced pose-based signals (weighted by penis association)
        pose_contribution = 0.0
        if pose_signals:
            # Weighted combination of anatomical activities
            anatomical_weights = {
                'face_activity': 0.10,
                'breast_activity': 0.30,  # High weight for breast movement
                'navel_activity': 0.25,   # High weight for core movement
                'hand_activity': 0.25,    # Hand-penis proximity important
                'overall_body_activity': 0.10
            }
            
            for signal_name, weight in anatomical_weights.items():
                signal_value = pose_signals.get(signal_name, 0.0)
                pose_contribution += weight * signal_value
            
            # Apply penis association weighting
            pose_contribution *= penis_weight
        
        # Check if we have meaningful activity
        total_base_signal = sum(base_signals.values())
        total_signal_strength = total_base_signal + pose_contribution
        has_active_signals = total_signal_strength > 0.01
        
        if has_active_signals:
            # Combine base signals and pose contributions
            base_fused_signal = total_base_signal
            
            # Apply pose enhancement - stronger boost for high-confidence penis association
            pose_boost_factor = 1.0 + (pose_contribution * penis_confidence * 0.8)  # Up to 80% boost
            
            # Final signal combines base + enhanced pose activity
            fused_signal = (base_fused_signal * pose_boost_factor) + (pose_contribution * 0.3)
            
            # Map to funscript position (0-100)
            primary_pos = fused_signal * 100.0
            
            # For secondary axis, use optical flow direction if available
            secondary_pos = self.last_secondary_position  # Start with current position
            if self.flow_analyses:
                # Use horizontal flow component for secondary axis
                horizontal_flows = [fa.flow_direction[0] for fa in self.flow_analyses if fa.confidence > 0.3]
                if horizontal_flows:
                    avg_horizontal = np.mean(horizontal_flows)
                    secondary_pos = 50.0 + (avg_horizontal * 25.0)  # Scale to ±25 around center
                    secondary_pos = max(0.0, min(100.0, secondary_pos))  # Clamp to valid range
            
            # Apply temporal smoothing only when we have active signals
            alpha = self.temporal_smoothing_alpha
            primary_pos = alpha * primary_pos + (1 - alpha) * self.last_primary_position
            secondary_pos = alpha * secondary_pos + (1 - alpha) * self.last_secondary_position
            
        else:
            # No motion detected - maintain current positions (no decay)
            primary_pos = self.last_primary_position
            secondary_pos = self.last_secondary_position
        
        # Update history
        self.last_primary_position = primary_pos
        self.last_secondary_position = secondary_pos
        
        self.primary_signal_history.append(primary_pos)
        self.secondary_signal_history.append(secondary_pos)
        
        # Update graphical debug history
        self.position_history.append(primary_pos)
        
        return primary_pos, secondary_pos
    
    def _generate_funscript_actions(self, primary_pos: float, secondary_pos: float, 
                                  frame_time_ms: int, frame_index: Optional[int] = None) -> List[Dict]:
        """Generate funscript actions based on computed positions."""
        action_log = []
        
        # Only generate actions when tracking is active
        if not self.tracking_active:
            return action_log
        
        try:
            # Use provided timestamp directly
            timestamp = frame_time_ms
            
            # Apply signal amplification if available
            if hasattr(self, 'signal_amplifier') and self.signal_amplifier:
                # Use optical flow data for signal enhancement
                dy_flow = 0.0
                dx_flow = 0.0
                if self.flow_analyses:
                    # Average the flow directions from all analyses
                    avg_flow = np.mean([fa.flow_direction for fa in self.flow_analyses], axis=0)
                    dy_flow = float(avg_flow[1]) if len(avg_flow) > 1 else 0.0
                    dx_flow = float(avg_flow[0]) if len(avg_flow) > 0 else 0.0
                
                primary_pos, secondary_pos = self.signal_amplifier.enhance_signal(
                    int(primary_pos), int(secondary_pos), dy_flow, dx_flow
                )
            
            # Primary axis action
            action_primary = {
                "at": timestamp,
                "pos": int(np.clip(primary_pos, 0, 100))
            }
            
            # Secondary axis action (for dual-axis support)
            action_secondary = {
                "at": timestamp,
                "secondary_pos": int(np.clip(secondary_pos, 0, 100))
            }
            
            # Add to funscript if available
            if hasattr(self, 'funscript') and self.funscript:
                self.funscript.add_action(timestamp, int(primary_pos))
                if hasattr(self.funscript, 'add_secondary_action'):
                    self.funscript.add_secondary_action(timestamp, int(secondary_pos))
            
            # Return for action log
            action_log.append({**action_primary, **action_secondary})
        
        except Exception as e:
            self.logger.warning(f"Action generation failed: {e}")
        
        return action_log
    
    def _create_debug_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Create visual debug overlay showing all processing stages."""
        if not self.show_debug_overlay:
            return frame
        
        try:
            # === DRAW CHANGE REGIONS (REDUCED CLUTTER) ===
            if self.show_change_regions and len(self.change_regions) <= 15:  # Only show if not too cluttered
                for region in self.change_regions:
                    if region.intensity > 20:  # Only show significant changes
                        cv2.rectangle(frame, (region.x, region.y), 
                                    (region.x + region.width, region.y + region.height),
                                    (0, 180, 180), 1)  # Dimmer teal rectangles
                        
                        # Simplified label - no "??..." confusion
                        if region.intensity > 35:  # Only label high-intensity regions
                            label = f"Δ{region.intensity:.0f}"
                            cv2.putText(frame, label, (region.x, region.y - 3),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 180, 180), 1)
            
            # === DRAW YOLO DETECTIONS (CLEANER) ===
            if self.show_yolo_detections:
                for sem_region in self.semantic_regions:
                    x1, y1, x2, y2 = sem_region.bbox
                    
                    # Highlight only important detections (penis, person, etc.)
                    important_classes = ['penis', 'locked_penis', 'person', 'hand', 'breast']
                    if sem_region.class_name.lower() in important_classes:
                        # Color based on priority (red=high, blue=low)
                        priority_ratio = min(sem_region.priority / 10.0, 1.0)
                        color = (
                            int(255 * (1 - priority_ratio)),  # Blue component
                            int(128 * priority_ratio),         # Green component  
                            int(255 * priority_ratio)          # Red component
                        )
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Clear, readable label with enhanced visibility for target classes
                        label = f"{sem_region.class_name}:{sem_region.confidence:.2f}"
                        label_color = color
                        if sem_region.class_name.lower() in ['penis', 'locked_penis', 'pussy', 'butt']:
                            label_color = (0, 255, 255)  # Bright cyan for target classes
                        cv2.putText(frame, label, (x1, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, label_color, 1)
            
            # === DRAW FLOW VECTORS ===
            if self.show_flow_vectors:
                for i, flow_analysis in enumerate(self.flow_analyses):
                    # Draw flow vector at region center
                    if i < len(self.change_regions):
                        region = self.change_regions[i]
                        center_x = region.x + region.width // 2
                        center_y = region.y + region.height // 2
                        
                        # Scale flow direction for visualization
                        flow_scale = 30.0 * flow_analysis.flow_magnitude
                        end_x = int(center_x + flow_analysis.flow_direction[0] * flow_scale)
                        end_y = int(center_y + flow_analysis.flow_direction[1] * flow_scale)
                        
                        # Color based on confidence
                        conf_color = (
                            0,
                            int(255 * flow_analysis.confidence),
                            int(255 * (1 - flow_analysis.confidence))
                        )
                        
                        cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y),
                                      conf_color, 2)
            
            # === OSCILLATION GRID OVERLAY ===
            if hasattr(self, 'show_oscillation_grid') and getattr(self, 'show_oscillation_grid', True):
                self._draw_oscillation_grid(frame)
            
            # === POSE VISUALIZATION ===
            if hasattr(self, 'last_pose_data') and self.last_pose_data:
                frame = self._draw_pose_visualization(frame, self.last_pose_data)
            
            # === STATUS OVERLAY ===
            # FPS info removed - will be passed to control panel via debug_info
            
            # Show target tracking status when locked box is active
            current_time = time.time()
            if (self.locked_penis_box and 
                (current_time - self.locked_penis_last_seen) < self.locked_penis_persistence_duration):
                time_since_seen = current_time - self.locked_penis_last_seen
                target_text = f"LOCKED TARGET (seen {time_since_seen:.1f}s ago)"
                cv2.putText(frame, target_text, (10, 25),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Debug: Show YOLO detection count for troubleshooting
            if len(self.semantic_regions) > 0:
                yolo_debug_text = f"YOLO: {len(self.semantic_regions)} objects detected"
                cv2.putText(frame, yolo_debug_text, (10, 50),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
            
            # Position text overlay removed - data available via debug_info
            
            # Simplified fusion weights - move to bottom right
            if hasattr(self, '_show_detailed_debug') and self._show_detailed_debug:
                weights_text = f"W: FD={self.fusion_weights['frame_diff']:.1f} YL={self.fusion_weights['yolo_weighted']:.1f} OF={self.fusion_weights['optical_flow']:.1f}"
                text_size_w = cv2.getTextSize(weights_text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
                text_x_w = frame_w - text_size_w[0] - 10
                text_y_w = frame_h - 30
                cv2.putText(frame, weights_text, (text_x_w, text_y_w),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        
        except Exception as e:
            self.logger.warning(f"Debug overlay creation failed: {e}")
        
        return frame
    
    def _draw_oscillation_grid(self, frame: np.ndarray):
        """Draw oscillation detection grid overlay."""
        try:
            frame_h, frame_w = frame.shape[:2]
            
            # Draw grid lines
            for y in range(0, frame_h, self.oscillation_block_size):
                cv2.line(frame, (0, y), (frame_w, y), (100, 100, 100), 1)
            
            for x in range(0, frame_w, self.oscillation_block_size):
                cv2.line(frame, (x, 0), (x, frame_h), (100, 100, 100), 1)
            
            # Draw oscillation intensity in each cell
            for (grid_x, grid_y), history in self.oscillation_history.items():
                if len(history) > 5:  # Only show cells with some history
                    recent_intensity = np.mean(list(history)[-5:])  # Average of last 5 frames
                    
                    if recent_intensity > 0.1:  # Only show active cells
                        # Calculate cell center
                        center_x = grid_x + self.oscillation_block_size // 2
                        center_y = grid_y + self.oscillation_block_size // 2
                        
                        # Color intensity based on oscillation strength
                        intensity_color = int(255 * min(recent_intensity, 1.0))
                        color = (0, intensity_color, intensity_color)  # Yellow-ish
                        
                        # Draw filled circle
                        radius = int(self.oscillation_block_size * 0.3 * recent_intensity)
                        cv2.circle(frame, (center_x, center_y), max(radius, 2), color, -1)
                        
                        # Draw intensity value
                        cv2.putText(frame, f"{recent_intensity:.2f}", 
                                  (center_x - 15, center_y + 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        except Exception as e:
            self.logger.warning(f"Grid overlay drawing failed: {e}")
    
    def _draw_pose_visualization(self, frame: np.ndarray, pose_data: Dict[str, Any]) -> np.ndarray:
        """Draw comprehensive multi-person pose and anatomical activity visualization."""
        if not pose_data or not pose_data.get('all_persons'):
            return frame
        
        try:
            frame_h, frame_w = frame.shape[:2]
            
            # Get contact information
            contact_info = pose_data.get('contact_info', {})
            persons_in_contact = contact_info.get('persons_in_contact', [])
            contact_scores = contact_info.get('contact_scores', {})
            
            # Draw all detected persons with contact highlighting
            primary_person = pose_data.get('primary_person')
            primary_id = primary_person['person_id'] if primary_person else -1
            
            for person in pose_data['all_persons']:
                person_id = person['person_id']
                is_primary = (person_id == primary_id)
                is_in_contact = person_id in persons_in_contact
                keypoints = person.get('raw_keypoints', [])
                
                if len(keypoints) < 17:
                    continue
                
                # Enhanced color coding
                if is_primary and is_in_contact:
                    # Primary person in contact - bright red/orange
                    skeleton_color = (0, 100, 255)  # Bright red-orange
                    joint_color = (0, 50, 255)
                elif is_primary:
                    # Primary person not in contact - standard bright
                    skeleton_color = (245, 117, 66)
                    joint_color = (245, 66, 230)
                elif is_in_contact:
                    # Non-primary person in contact - bright yellow
                    skeleton_color = (0, 255, 255)  # Bright yellow
                    joint_color = (0, 200, 255)
                else:
                    # Non-primary, not in contact - dim
                    skeleton_color = (150, 80, 40)
                    joint_color = (150, 40, 150)
                
                # Draw pose skeleton
                pose_connections = [
                    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
                    (11, 12), (5, 11), (6, 12),               # Torso
                    (11, 13), (13, 15), (12, 14), (14, 16)    # Legs
                ]
                
                for connection in pose_connections:
                    pt1_idx, pt2_idx = connection
                    if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
                        pt1, pt2 = keypoints[pt1_idx], keypoints[pt2_idx]
                        
                        if pt1[2] > 0.5 and pt2[2] > 0.5:
                            x1, y1 = int(pt1[0]), int(pt1[1])
                            x2, y2 = int(pt2[0]), int(pt2[1])
                            thickness = 3 if is_primary else 2
                            cv2.line(frame, (x1, y1), (x2, y2), skeleton_color, thickness)
                
                # Draw keypoints
                for keypoint in keypoints:
                    if keypoint[2] > 0.5:
                        x, y = int(keypoint[0]), int(keypoint[1])
                        radius = 5 if is_primary else 3
                        cv2.circle(frame, (x, y), radius, joint_color, -1)
                
                # Draw enhanced person ID with contact info
                if person.get('center'):
                    center_x, center_y = person['center']
                    
                    # Build person label
                    label_parts = [f"P{person_id}"]
                    if is_primary:
                        label_parts.append("*")
                    if is_in_contact:
                        contact_score = contact_scores.get(person_id, 0)
                        label_parts.append(f"CONTACT({contact_score:.2f})")
                    
                    person_label = " ".join(label_parts)
                    
                    # Color based on contact status
                    label_color = (255, 255, 255)  # Default white
                    if is_primary and is_in_contact:
                        label_color = (0, 255, 255)  # Cyan for primary in contact
                    elif is_in_contact:
                        label_color = (0, 255, 0)    # Green for contact
                    
                    cv2.putText(frame, person_label, (center_x - 30, center_y - 25),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)
            
            # Draw anatomical activity regions for primary person
            if primary_person:
                anatomical_activities = pose_data.get('anatomical_activities', {})
                self._draw_anatomical_regions(frame, anatomical_activities, frame_w, frame_h)
            
            # Draw locked penis box and contact regions
            self._draw_locked_penis_and_contact(frame, contact_info)
            
            # Draw comprehensive activity panel
            self._draw_activity_panel(frame, pose_data, frame_w, frame_h)
        
        except Exception as e:
            self.logger.warning(f"Pose visualization failed: {e}")
        
        return frame
    
    def _draw_anatomical_regions(self, frame: np.ndarray, anatomical_activities: Dict, frame_w: int, frame_h: int):
        """Draw anatomical region centers and activity indicators."""
        try:
            regions = [
                ('face', (255, 255, 0), anatomical_activities.get('face', {})),       # Yellow
                ('breasts', (255, 0, 255), anatomical_activities.get('breasts', {})), # Magenta  
                ('navel', (0, 255, 255), anatomical_activities.get('navel', {})),     # Cyan
                ('hands', (0, 255, 0), anatomical_activities.get('hands', {}))        # Green
            ]
            
            for region_name, color, region_data in regions:
                if not region_data:
                    continue
                    
                center = region_data.get('center')
                activity = region_data.get('activity', 0.0)
                
                if center and len(center) >= 2:
                    x, y = int(center[0]), int(center[1])
                    
                    # Draw region center
                    radius = max(5, int(15 * activity))  # Size based on activity
                    cv2.circle(frame, (x, y), radius, color, 2)
                    
                    # Draw activity text
                    label = f"{region_name.title()}: {activity:.2f}"
                    cv2.putText(frame, label, (x + 20, y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        except Exception as e:
            self.logger.warning(f"Anatomical region drawing failed: {e}")
    
    def _draw_locked_penis_and_contact(self, frame: np.ndarray, contact_info: Dict):
        """Draw locked penis box, contact regions, and tracking history."""
        try:
            # Draw penis box history with fading effect
            if hasattr(self, 'penis_box_history') and self.penis_box_history:
                for i, penis_box in enumerate(self.penis_box_history[-5:]):  # Last 5 boxes
                    bbox = penis_box['bbox']
                    confidence = penis_box['confidence']
                    
                    # Fade older boxes
                    age_factor = (i + 1) / 5.0
                    alpha = int(255 * age_factor * confidence)
                    color = (alpha//2, alpha//2, alpha)  # Fading purple
                    
                    thickness = 1
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                                 (int(bbox[2]), int(bbox[3])), color, thickness)
            
            # Draw prominent LOCKED PENIS box with enhanced visibility
            locked_penis_box = contact_info.get('locked_penis_box')
            if locked_penis_box:
                bbox = locked_penis_box['bbox']
                confidence = locked_penis_box['confidence']
                
                # Thick bright cyan box for locked penis - EXTRA VISIBILITY
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                
                # Double border for maximum visibility
                cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), (0, 255, 255), 6)  # Outer thick cyan
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)  # Inner white
                
                # Large corner indicators for locked status
                corner_size = 20
                cv2.line(frame, (x1-5, y1-5), (x1 + corner_size, y1-5), (0, 255, 255), 8)
                cv2.line(frame, (x1-5, y1-5), (x1-5, y1 + corner_size), (0, 255, 255), 8)
                cv2.line(frame, (x2+5, y1-5), (x2 - corner_size, y1-5), (0, 255, 255), 8)
                cv2.line(frame, (x2+5, y1-5), (x2+5, y1 + corner_size), (0, 255, 255), 8)
                cv2.line(frame, (x1-5, y2+5), (x1 + corner_size, y2+5), (0, 255, 255), 8)
                cv2.line(frame, (x1-5, y2+5), (x1-5, y2 - corner_size), (0, 255, 255), 8)
                cv2.line(frame, (x2+5, y2+5), (x2 - corner_size, y2+5), (0, 255, 255), 8)
                cv2.line(frame, (x2+5, y2+5), (x2+5, y2 - corner_size), (0, 255, 255), 8)
                
                # Large, prominent locked target label
                label_text = f"*** LOCKED TARGET ({confidence:.2f}) ***"
                cv2.putText(frame, label_text, (x1, y1 - 15),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)
                cv2.putText(frame, label_text, (x1, y1 - 15),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)  # White outline
                
                # Draw contact regions with bright highlights
                contact_regions = contact_info.get('contact_regions', [])
                for i, contact_region in enumerate(contact_regions):
                    contact_bbox = contact_region['overlap_bbox']
                    contact_score = contact_region['contact_score']
                    person_id = contact_region['person_id']
                    
                    # Bright contact region highlight
                    cx1, cy1, cx2, cy2 = contact_bbox
                    contact_color = (0, int(255 * contact_score), 255)  # Yellow-cyan based on score
                    
                    # Draw contact region with pulsing effect
                    thickness = max(2, int(6 * contact_score))
                    cv2.rectangle(frame, (int(cx1), int(cy1)), (int(cx2), int(cy2)), 
                                 contact_color, thickness)
                    
                    # Contact label
                    contact_label = f"CONTACT P{person_id}: {contact_score:.2f}"
                    cv2.putText(frame, contact_label, (int(cx1), int(cy1) - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, contact_color, 1)
        
        except Exception as e:
            self.logger.error(f"Locked penis visualization failed: {e}")
            # Show error on screen for debugging
            cv2.putText(frame, f"Locked Penis Viz Error: {str(e)[:50]}", (10, 100),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    def _draw_activity_panel(self, frame: np.ndarray, pose_data: Dict, frame_w: int, frame_h: int):
        """Draw comprehensive activity status panel."""
        try:
            panel_x = frame_w - 250
            panel_y = 10
            panel_w = 240
            panel_h = 200
            
            # Semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), 
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Title
            cv2.putText(frame, "POSE INTELLIGENCE", (panel_x + 5, panel_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            y_offset = panel_y + 40
            line_height = 16
            
            # Person detection info
            total_persons = pose_data.get('debug_info', {}).get('total_persons_detected', 0)
            primary_id = pose_data.get('debug_info', {}).get('primary_person_id', 'None')
            cv2.putText(frame, f"Persons: {total_persons} (Primary: {primary_id})", 
                       (panel_x + 5, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += line_height
            
            # Penis association and contact
            penis_conf = pose_data.get('penis_association_confidence', 0.0)
            penis_boxes = len(getattr(self, 'penis_box_history', []))
            contact_info = pose_data.get('contact_info', {})
            persons_in_contact = len(contact_info.get('persons_in_contact', []))
            
            cv2.putText(frame, f"Penis Assoc: {penis_conf:.2f} ({penis_boxes} boxes)", 
                       (panel_x + 5, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            y_offset += line_height
            
            cv2.putText(frame, f"Contact: {persons_in_contact} persons", 
                       (panel_x + 5, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            y_offset += line_height + 5
            
            # Anatomical activities
            anatomical = pose_data.get('anatomical_activities', {})
            activities = [
                ('Face', anatomical.get('face', {}).get('activity', 0.0), (255, 255, 0)),
                ('Breast', anatomical.get('breasts', {}).get('activity', 0.0), (255, 0, 255)),
                ('Navel', anatomical.get('navel', {}).get('activity', 0.0), (0, 255, 255)),
                ('Hands', anatomical.get('hands', {}).get('activity', 0.0), (0, 255, 0))
            ]
            
            for name, activity, color in activities:
                # Activity bar
                bar_width = int(100 * activity)
                if bar_width > 0:
                    cv2.rectangle(frame, (panel_x + 60, y_offset - 5), 
                                 (panel_x + 60 + bar_width, y_offset + 5), color, -1)
                
                cv2.putText(frame, f"{name}: {activity:.2f}", 
                           (panel_x + 5, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                y_offset += line_height
            
            # Overall signals
            signals = pose_data.get('signal_components', {})
            overall_activity = signals.get('overall_body_activity', 0.0)
            torso_stability = signals.get('torso_stability', 1.0)
            
            y_offset += 5
            cv2.putText(frame, f"Overall Activity: {overall_activity:.2f}", 
                       (panel_x + 5, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += line_height
            cv2.putText(frame, f"Torso Stability: {torso_stability:.2f}", 
                       (panel_x + 5, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        except Exception as e:
            self.logger.warning(f"Activity panel drawing failed: {e}")
    
    def _prepare_overlay_data(self, priority_regions: List, pose_data: Dict[str, Any]):
        """
        Prepare overlay data in standardized format for VideoDisplayUI.
        This replaces the internal _create_debug_overlay method.
        """
        # Convert semantic regions to bounding boxes
        self.logger.debug(f"Converting {len(self.semantic_regions)} semantic regions to boxes")
        boxes = TrackerVisualizationHelper.convert_semantic_regions_to_boxes(
            self.semantic_regions
        )
        self.logger.debug(f"Converted to {len(boxes)} bounding boxes")
        
        # Add locked penis box if available
        if self.locked_penis_box:
            # locked_penis_box is a dict with 'bbox' key
            if isinstance(self.locked_penis_box, dict) and 'bbox' in self.locked_penis_box:
                bbox = self.locked_penis_box['bbox']
                confidence = self.locked_penis_box.get('confidence', self.penis_tracker_confidence)
                locked_box = BoundingBox(
                    x1=bbox[0],
                    y1=bbox[1],
                    x2=bbox[2],
                    y2=bbox[3],
                    class_name="locked_penis",
                    confidence=confidence,
                    color_override=(0, 255, 255),  # Bright cyan
                    thickness_override=3.0,
                    label_suffix="*** LOCKED ***"
                )
                boxes.append(locked_box)
            else:
                self.logger.warning(f"Invalid locked_penis_box format: {type(self.locked_penis_box)}")
        
        # Convert pose data to keypoints
        pose_keypoints = []
        if pose_data:
            self.logger.debug(f"Pose data keys: {list(pose_data.keys())}")
            primary_person = pose_data.get('primary_person')
            all_persons = pose_data.get('all_persons', [])
            self.logger.debug(f"Primary person: {primary_person is not None}")
            self.logger.debug(f"All persons: {len(all_persons)} items")
            
            pose_keypoints = TrackerVisualizationHelper.convert_pose_data_to_keypoints(pose_data)
            self.logger.debug(f"Converted pose keypoints: {len(pose_keypoints)}")
            
            # Debug the actual keypoint data when detected
            if pose_keypoints:
                for i, pose in enumerate(pose_keypoints):
                    keypoints = pose.keypoints if hasattr(pose, 'keypoints') else []
                    high_conf = len([kp for kp in keypoints if len(kp) >= 3 and kp[2] > 0.5])
                    self.logger.debug(f"Pose {i}: {len(keypoints)} keypoints, {high_conf} high confidence")
        
        # Prepare contact info
        contact_info = pose_data.get('contact_info', {}) if pose_data else {}
        
        # Determine motion mode (if applicable)
        motion_mode = None
        if hasattr(self, 'motion_mode'):
            motion_mode = self.motion_mode
        
        # Use locked penis box data (already contains evolved size)
        locked_penis_box_data = self.locked_penis_box
        
        # Prepare change regions data for contact-aware visualization
        change_regions_data = []
        self.logger.debug(f"Processing {len(self.change_regions)} change regions")
        
        # Convert change regions to standard format first
        for i, region in enumerate(self.change_regions):
            self.logger.debug(f"Change region {i}: type={type(region)}")
            self.logger.debug(f"  Has bbox: {hasattr(region, 'bbox')}")
            self.logger.debug(f"  Has x,y: {hasattr(region, 'x')}, {hasattr(region, 'y')}")
            if hasattr(region, 'x'):
                self.logger.debug(f"  Values: x={region.x}, y={region.y}, w={getattr(region, 'width', 'N/A')}, h={getattr(region, 'height', 'N/A')}")
            
            # Handle ChangeRegion objects with x,y,width,height format
            if hasattr(region, 'bbox'):
                bbox = region.bbox
            elif hasattr(region, 'x') and hasattr(region, 'y'):
                bbox = (region.x, region.y, region.x + region.width, region.y + region.height)
            else:
                continue
                
            intensity = getattr(region, 'intensity', 1.0)
            change_regions_data.append({
                'bbox': bbox,
                'intensity': intensity,
                'type': 'change_region'
            })
        
        # Apply contact-aware prioritization if we have pose data and locked penis
        if pose_keypoints and locked_penis_box_data:
            self.logger.debug("Applying contact-aware visualization to change regions")
            
            # Analyze skeleton-penis contact
            poses_for_analysis = [pose.to_dict() for pose in pose_keypoints]
            contact_analysis = TrackerVisualizationHelper.analyze_skeleton_penis_contact(
                poses_for_analysis, locked_penis_box_data
            )
            
            # Apply contact-aware colors to regions
            change_regions_data = TrackerVisualizationHelper.apply_contact_aware_colors(
                change_regions_data, contact_analysis, poses_for_analysis
            )
            
            self.logger.debug(f"Contact analysis: {contact_analysis}")
            high_priority_count = len([r for r in change_regions_data if r.get('contact_priority', 0) >= 3])
            self.logger.debug(f"High-priority contact regions: {high_priority_count}/{len(change_regions_data)}")
        
        # Prepare flow vectors data for visualization  
        flow_vectors_data = []
        self.logger.debug(f"Processing {len(self.flow_analyses)} flow analyses")
        for i, analysis in enumerate(self.flow_analyses):
            self.logger.debug(f"Flow analysis {i}: type={type(analysis)}")
            self.logger.debug(f"  Has flow_magnitude: {hasattr(analysis, 'flow_magnitude')}")
            self.logger.debug(f"  Has flow_direction: {hasattr(analysis, 'flow_direction')}")
            if hasattr(analysis, 'flow_magnitude'):
                self.logger.debug(f"  Magnitude: {analysis.flow_magnitude}")
            if hasattr(analysis, 'flow_direction'):
                self.logger.debug(f"  Direction: {analysis.flow_direction} (type: {type(analysis.flow_direction)})")
            # Handle FlowAnalysis objects
            if hasattr(analysis, 'flow_magnitude') and hasattr(analysis, 'flow_direction'):
                # Create synthetic flow vector from magnitude and direction
                import math
                magnitude = analysis.flow_magnitude
                direction = analysis.flow_direction  # in radians
                
                # Ensure both magnitude and direction are scalars (handle numpy arrays)
                if hasattr(magnitude, 'shape') and magnitude.shape == ():
                    # 0-dimensional numpy array
                    magnitude = magnitude.item()
                elif hasattr(magnitude, '__len__') and len(magnitude) == 1:
                    magnitude = float(magnitude[0])
                elif hasattr(magnitude, 'size') and magnitude.size == 1:
                    # Single element numpy array
                    magnitude = float(magnitude.flat[0])
                elif not isinstance(magnitude, (int, float)):
                    # Multi-element array - take mean or first element
                    if hasattr(magnitude, '__len__'):
                        magnitude = float(magnitude[0]) if len(magnitude) > 0 else 0.0
                    else:
                        magnitude = float(magnitude)
                    
                if hasattr(direction, 'shape') and direction.shape == ():
                    # 0-dimensional numpy array
                    direction = direction.item()
                elif hasattr(direction, '__len__') and len(direction) == 1:
                    direction = float(direction[0])
                elif hasattr(direction, 'size') and direction.size == 1:
                    # Single element numpy array
                    direction = float(direction.flat[0])
                elif not isinstance(direction, (int, float)):
                    # Multi-element array - take mean or first element
                    if hasattr(direction, '__len__'):
                        direction = float(direction[0]) if len(direction) > 0 else 0.0
                    else:
                        direction = float(direction)
                
                # Use region center as start point if available
                if hasattr(analysis, 'region_id'):
                    # Find the corresponding change region for start point
                    start_point = (320, 320)  # Default center
                    for region in self.change_regions:
                        if hasattr(region, 'x') and hasattr(region, 'y'):
                            start_point = (region.x + region.width//2, region.y + region.height//2)
                            break
                else:
                    start_point = (320, 320)
                
                # Calculate end point from magnitude and direction
                end_x = start_point[0] + magnitude * math.cos(direction) * 20  # Scale for visibility
                end_y = start_point[1] + magnitude * math.sin(direction) * 20
                
                flow_vectors_data.append({
                    'start_point': start_point,
                    'end_point': (end_x, end_y),
                    'magnitude': magnitude,
                    'type': 'optical_flow'
                })

        self.overlay_data = TrackerVisualizationHelper.prepare_overlay_data(
            yolo_boxes=boxes,
            poses=pose_keypoints,
            motion_mode=motion_mode,
            locked_penis_box=locked_penis_box_data,
            contact_info=contact_info,
            change_regions=change_regions_data,
            flow_vectors=flow_vectors_data,
            # Additional hybrid tracker data
            oscillation_grid_active=self.show_oscillation_grid,
            oscillation_sensitivity=self.oscillation_sensitivity,
            frame_count=self.frame_count
        )
    
    def _update_debug_window_data(self, processing_time: float):
        """Update debug window data for external rendering."""
        fps_estimate = 1.0 / processing_time if processing_time > 0 else 0
        
        # Get amplification factor from signal amplifier for use in metrics and progress bars
        amplification_factor = 1.0
        if hasattr(self, 'signal_amplifier') and self.signal_amplifier:
            # Use live amp status as amplification factor indicator
            stats = self.signal_amplifier.get_statistics()
            amplification_factor = 2.0 if stats.get('live_amp_enabled', False) else 1.0
        
        # Organize metrics into collapsible sections
        metrics = {
            # Performance Section
            "Performance": {
                "Frame Count": self.frame_count,
                "Processing FPS": f"{fps_estimate:.1f}",
                "YOLO Interval": self.yolo_current_interval,
            },
            # Detection Section
            "Detection": {
                "Objects": len(self.semantic_regions),
                "Motion Regions": len(self.change_regions), 
                "Flow Analyses": len(self.flow_analyses),
            },
            # Tracking Section
            "Tracking": {
                "Primary Position": f"{self.last_primary_position:.1f}",
                "Secondary Position": f"{self.last_secondary_position:.1f}",
                "Oscillation Sensitivity": f"{self.oscillation_sensitivity:.1f}",
                "Dynamic Amplification": f"{amplification_factor:.2f}x",
            }
        }
        
        # Add pose & penis tracking (compact, non-duplicate format)
        if hasattr(self, 'last_pose_data') and self.last_pose_data:
            pose_debug = self.last_pose_data.get('debug_info', {})
            persons_count = pose_debug.get('total_persons_detected', 0)
            primary_id = pose_debug.get('primary_person_id', 'None')
            # Get penis state thread-safely
            penis_state = self._get_penis_state()
            penis_status = "Locked" if penis_state['locked_penis_box'] else "Inactive" 
            penis_hist = len(getattr(self, 'penis_box_history', []))
            size_hist = len(getattr(self, 'penis_size_history', []))
            
            # Add size evolution info (thread-safe access)
            evolved_info = ""
            tracker = penis_state['locked_penis_tracker']
            if tracker['evolved_size']:
                base_area = tracker['base_size'][2] if tracker['base_size'] else 0
                evolved_area = tracker['evolved_size'][2]
                growth_pct = ((evolved_area - base_area) / base_area * 100) if base_area > 0 else 0
                evolved_info = f", growth:{growth_pct:+.0f}%"
            
            # Add pose & penis data to tracking section
            metrics["Tracking"].update({
                "Detected Persons": persons_count,
                "Primary Person ID": primary_id,
                "Penis Status": penis_status,
                "Penis History": f"{penis_hist} boxes, {size_hist} sizes{evolved_info}"
            })
        
        # Create relevant progress bars for visual feedback
        oscillation_scaled = min(1.0, getattr(self, 'current_oscillation_intensity', 0.0) * 20.0)
        
        progress_bars = {
            "Penis Confidence": min(1.0, self.last_pose_data.get('penis_association_confidence', 0.0)) if hasattr(self, 'last_pose_data') and self.last_pose_data else 0.0,
            "Oscillation Intensity": oscillation_scaled,
            "Processing Load": min(1.0, fps_estimate / 60.0) if fps_estimate > 0 else 0.0,
            "Dynamic Amplification": min(1.0, (amplification_factor - 1.0) / 9.0) if amplification_factor > 1.0 else 0.0,  # Scale to 0-1 (1x-10x range)
        }

        self.debug_window_data = TrackerVisualizationHelper.create_debug_window_data(
            tracker_name="Hybrid Intelligence",
            metrics=metrics,
            show_graphs=False,  # Disable useless graphs
            graphs=None,  # No graphs needed
            progress_bars=progress_bars
        )
    
    def _generate_debug_info(self, processing_time: float) -> Dict[str, Any]:
        """Generate debug information for UI display."""
        # Calculate FPS for control panel display
        fps_estimate = 0
        if self.processing_times and len(self.processing_times) > 5:
            avg_time = np.mean(list(self.processing_times))
            fps_estimate = 1.0 / avg_time if avg_time > 0 else 0
        
        debug_info = {
            "hybrid_intelligence_tracker": {
                "processing_time_ms": processing_time * 1000,
                "fps_estimate": fps_estimate,  # For control panel display
                "frame_count": self.frame_count,
                "tracking_active": self.tracking_active,
                "change_regions_detected": len(self.change_regions),
                "semantic_objects_detected": len(self.semantic_regions),
                "flow_analyses_computed": len(self.flow_analyses),
                "primary_position": self.last_primary_position,
                "secondary_position": self.last_secondary_position,
                "fusion_weights": self.fusion_weights.copy(),
                "yolo_update_interval": self.yolo_current_interval,
                "oscillation_grid_size": self.oscillation_grid_size,
                "pose_estimation_active": self.pose_model is not None
            }
        }
        
        # Add comprehensive pose debug info if available
        if hasattr(self, 'last_pose_data') and self.last_pose_data:
            pose_debug = self.last_pose_data.get('debug_info', {})
            anatomical = self.last_pose_data.get('anatomical_activities', {})
            signals = self.last_pose_data.get('signal_components', {})
            # Get penis state for thread-safe access
            penis_state = self._get_penis_state()
            
            debug_info["hybrid_intelligence_tracker"].update({
                "total_persons_detected": pose_debug.get('total_persons_detected', 0),
                "primary_person_id": pose_debug.get('primary_person_id'),
                "penis_association_confidence": self.last_pose_data.get('penis_association_confidence', 0.0),
                "penis_box_history_size": len(getattr(self, 'penis_box_history', [])),
                "locked_penis_active": self._is_penis_active(),
                "locked_penis_age": time.time() - penis_state['locked_penis_last_seen'] if penis_state['locked_penis_box'] else 0,
                "anatomical_activities": {
                    "face_activity": anatomical.get('face', {}).get('activity', 0.0),
                    "breast_activity": anatomical.get('breasts', {}).get('activity', 0.0),
                    "navel_activity": anatomical.get('navel', {}).get('activity', 0.0),
                    "hand_activity": anatomical.get('hands', {}).get('activity', 0.0),
                    "torso_stability": anatomical.get('torso', {}).get('stability', 1.0)
                },
                "pose_signal_components": signals,
                "anatomical_regions_active": pose_debug.get('anatomical_regions_active', 0)
            })
        
        return debug_info
    
    # ========================================
    # ENHANCED VISUALIZATION SYSTEM
    # ========================================
    
    def _prepare_overlay_data(self, priority_regions: List, pose_data: Dict[str, Any]):
        """
        Prepare comprehensive overlay data for external visualization.
        
        This provides clear visual indicators for:
        - Computation areas vs disregarded areas
        - Centroids vs actual action points
        - Flow vectors showing movement
        - Contact regions and interaction zones
        """
        self.overlay_data = {
            'computation_areas': [],
            'disregarded_areas': [],
            'action_points': [],
            'centroids': [],
            'flow_vectors': [],
            'contact_regions': [],
            'interaction_zones': [],
            'locked_penis_state': None,
            'oscillation_grid': [],
            'signal_source_indicators': []
        }
        
        try:
            # === COMPUTATION vs DISREGARDED AREAS ===
            if hasattr(self, 'change_regions'):
                for region in self.change_regions:
                    area_data = {
                        'bbox': (region.x, region.y, region.x + region.width, region.y + region.height),
                        'intensity': region.intensity,
                        'area': region.area,
                        'type': 'change_region'
                    }
                    
                    # Smart classification based on interaction zones and thresholds
                    region_center = (region.x + region.width // 2, region.y + region.height // 2)
                    interaction_regions = self._get_interaction_regions()
                    is_in_interaction_zone = self._is_cell_in_interaction_zone(region_center, interaction_regions)
                    
                    # Use smart thresholds matching oscillation detection
                    threshold = 5 if is_in_interaction_zone else 15
                    
                    if region.intensity > threshold:
                        area_data['status'] = 'computed'
                        area_data['color'] = (0, 255, 0, 80)  # Green overlay for computed areas
                        area_data['zone_type'] = 'interaction' if is_in_interaction_zone else 'global'
                        self.overlay_data['computation_areas'].append(area_data)
                    else:
                        area_data['status'] = 'disregarded'  
                        area_data['color'] = (128, 128, 128, 40)  # Gray overlay for disregarded
                        area_data['zone_type'] = 'interaction' if is_in_interaction_zone else 'global'
                        self.overlay_data['disregarded_areas'].append(area_data)
            
            # === INTERACTION ZONES ===
            interaction_regions = self._get_interaction_regions()
            for i, (x1, y1, x2, y2) in enumerate(interaction_regions):
                zone_data = {
                    'bbox': (x1, y1, x2, y2),
                    'type': 'interaction_zone',
                    'color': (255, 255, 0, 60),  # Yellow overlay for interaction zones
                    'priority': 'high'
                }
                self.overlay_data['interaction_zones'].append(zone_data)
            
            # === ACTION POINTS vs CENTROIDS ===
            # Check if flow analyses exist and have the expected structure
            if hasattr(self, 'flow_analyses') and self.flow_analyses:
                for flow_analysis in self.flow_analyses:
                    try:
                        # Check if flow_analysis has the expected attributes
                        if hasattr(flow_analysis, 'flow_direction') and hasattr(flow_analysis, 'region_id'):
                            # FlowAnalysis structure: region_id, flow_magnitude, flow_direction, oscillation_strength, confidence
                            # flow_direction is np.ndarray with flow vector
                            
                            # We need to determine center position from the region_id or other means
                            # For now, use a fallback approach since the exact structure isn't clear
                            
                            # Extract flow vector components
                            if hasattr(flow_analysis.flow_direction, '__len__') and len(flow_analysis.flow_direction) >= 2:
                                flow_x = flow_analysis.flow_direction[0] 
                                flow_y = flow_analysis.flow_direction[1]
                            else:
                                flow_x, flow_y = 0, 0
                            
                            # For now, skip detailed centroid/action point visualization until we can
                            # properly determine the region centers. Just add the flow magnitude info.
                            
                            # Add basic flow info for debugging
                            flow_info = {
                                'region_id': flow_analysis.region_id,
                                'magnitude': flow_analysis.flow_magnitude,
                                'direction': (flow_x, flow_y),
                                'confidence': flow_analysis.confidence,
                                'type': 'flow_analysis'
                            }
                            # Store in a separate category until we can properly visualize
                            if 'flow_info' not in self.overlay_data:
                                self.overlay_data['flow_info'] = []
                            self.overlay_data['flow_info'].append(flow_info)
                        
                    except Exception as e:
                        self.logger.debug(f"Skipping flow_analysis due to attribute issue: {e}")
                        continue
            
            # === LOCKED PENIS STATE ===
            penis_state = self._get_penis_state()
            if penis_state['locked_penis_box']:
                locked_penis_data = {
                    'bbox': penis_state['locked_penis_box'].get('bbox') if isinstance(penis_state['locked_penis_box'], dict) else penis_state['locked_penis_box'],
                    'confidence': penis_state.get('penis_tracker_confidence', 0.0),
                    'active': self._is_penis_active(),
                    'color': (0, 255, 0) if self._is_penis_active() else (255, 165, 0),  # Green if active, orange if inactive
                    'thickness': 3
                }
                self.overlay_data['locked_penis_state'] = locked_penis_data
            
            # === OSCILLATION ANALYSIS GRID ===
            if hasattr(self, 'oscillation_history') and self.oscillation_history:
                for (grid_x, grid_y), history in list(self.oscillation_history.items())[:50]:  # Limit for performance
                    if len(history) > 5:
                        recent_activity = np.std(list(history)[-10:]) if len(history) >= 10 else np.std(list(history))
                        grid_cell = {
                            'position': (grid_x, grid_y),
                            'size': getattr(self, 'oscillation_block_size', 32),
                            'activity': recent_activity,
                            'color': (255, int(255 * (1 - min(recent_activity, 1.0))), 0, 120),  # Heat map color
                            'computed': recent_activity > 0.1  # Show if this cell was actively computed
                        }
                        self.overlay_data['oscillation_grid'].append(grid_cell)
            
            # === SIGNAL SOURCE INDICATORS ===
            # Show what's contributing to the final funscript signal
            signal_sources = []
            
            # Frame differentiation contribution
            if hasattr(self, 'change_regions') and self.change_regions:
                total_change = sum(region.intensity * region.area for region in self.change_regions) / 1000000.0
                signal_sources.append({
                    'type': 'frame_diff',
                    'strength': min(total_change, 1.0),
                    'color': (0, 255, 0),
                    'label': f'Frame Diff: {total_change:.2f}'
                })
            
            # Optical flow contribution 
            if hasattr(self, 'flow_analyses') and self.flow_analyses:
                try:
                    # Safely extract flow magnitudes
                    flow_magnitudes = []
                    for f in self.flow_analyses:
                        if hasattr(f, 'flow_magnitude'):
                            flow_magnitudes.append(f.flow_magnitude)
                    
                    if flow_magnitudes:
                        flow_strength = np.mean(flow_magnitudes)
                        signal_sources.append({
                            'type': 'optical_flow', 
                            'strength': min(flow_strength * 10, 1.0),
                            'color': (0, 255, 255),
                            'label': f'Optical Flow: {flow_strength:.2f}'
                        })
                except Exception as e:
                    self.logger.debug(f"Error processing flow strength: {e}")
                    # Add placeholder
                    signal_sources.append({
                        'type': 'optical_flow', 
                        'strength': 0.0,
                        'color': (0, 255, 255),
                        'label': 'Optical Flow: N/A'
                    })
            
            # Oscillation contribution
            oscillation_strength = getattr(self, 'current_oscillation_intensity', 0.0)
            signal_sources.append({
                'type': 'oscillation',
                'strength': min(oscillation_strength * 20, 1.0),
                'color': (255, 0, 255),
                'label': f'Oscillation: {oscillation_strength:.3f}'
            })
            
            self.overlay_data['signal_source_indicators'] = signal_sources
            
        except Exception as e:
            self.logger.warning(f"Error preparing overlay data: {e}")
            # Ensure overlay_data exists even on error
            if not hasattr(self, 'overlay_data'):
                self.overlay_data = {}
    
    def _create_debug_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Create internal debug overlay on frame for fallback visualization.
        
        This method renders overlays directly on the frame when external
        visualization is not available.
        """
        try:
            # Use TrackerVisualizationHelper for consistent rendering
            from tracker_modules.helpers.visualization import TrackerVisualizationHelper
            
            # Ensure overlay_data exists
            if not hasattr(self, 'overlay_data'):
                return frame
                
            overlay_frame = frame.copy()
            
            # === RENDER COMPUTATION vs DISREGARDED AREAS ===
            for area in self.overlay_data.get('computation_areas', []):
                TrackerVisualizationHelper.draw_filled_rectangle(
                    overlay_frame, area['bbox'], area['color']
                )
                # Add label
                cv2.putText(overlay_frame, "COMPUTED", 
                          (int(area['bbox'][0]), int(area['bbox'][1]) - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            
            for area in self.overlay_data.get('disregarded_areas', []):
                TrackerVisualizationHelper.draw_filled_rectangle(
                    overlay_frame, area['bbox'], area['color']
                )
            
            # === RENDER INTERACTION ZONES ===
            for zone in self.overlay_data.get('interaction_zones', []):
                TrackerVisualizationHelper.draw_rectangle(
                    overlay_frame, zone['bbox'], zone['color'][:3], thickness=2
                )
                cv2.putText(overlay_frame, "INTERACTION ZONE",
                          (int(zone['bbox'][0]), int(zone['bbox'][1]) - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # === RENDER CENTROIDS vs ACTION POINTS ===
            # Draw centroids first (larger, behind)
            for centroid in self.overlay_data.get('centroids', []):
                cv2.circle(overlay_frame, 
                          (int(centroid['position'][0]), int(centroid['position'][1])),
                          centroid['size'], centroid['color'], -1)
                cv2.putText(overlay_frame, "C", 
                          (int(centroid['position'][0]) - 3, int(centroid['position'][1]) + 3),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Draw action points (smaller, in front)
            for action in self.overlay_data.get('action_points', []):
                cv2.circle(overlay_frame, 
                          (int(action['position'][0]), int(action['position'][1])),
                          action['size'], action['color'], -1)
                cv2.putText(overlay_frame, "A", 
                          (int(action['position'][0]) - 3, int(action['position'][1]) + 3),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
            
            # === RENDER FLOW VECTORS ===
            for vector in self.overlay_data.get('flow_vectors', []):
                cv2.arrowedLine(overlay_frame,
                               (int(vector['start'][0]), int(vector['start'][1])),
                               (int(vector['end'][0]), int(vector['end'][1])),
                               vector['color'], vector['thickness'])
            
            # === RENDER LOCKED PENIS STATE ===
            penis_state = self.overlay_data.get('locked_penis_state')
            if penis_state:
                bbox = penis_state['bbox']
                color = penis_state['color']
                thickness = penis_state['thickness']
                
                TrackerVisualizationHelper.draw_rectangle(
                    overlay_frame, bbox, color, thickness
                )
                
                status = "ACTIVE" if penis_state['active'] else "INACTIVE"
                cv2.putText(overlay_frame, f"PENIS {status}",
                          (int(bbox[0]), int(bbox[1]) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # === RENDER POSE SKELETONS ===
            for pose in self.overlay_data.get('poses', []):
                # Draw skeleton connections
                pose_connections = [
                    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
                    (11, 12), (5, 11), (6, 12),               # Torso
                    (11, 13), (13, 15), (12, 14), (14, 16),   # Legs
                    (1, 2), (2, 3), (3, 4), (1, 5), (2, 6)    # Head/neck
                ]
                
                keypoints = pose.get('keypoints', [])
                person_id = pose.get('person_id', 0)
                is_primary = pose.get('is_primary', False)
                is_in_contact = pose.get('is_in_contact', False)
                
                # Color based on status
                if is_primary and is_in_contact:
                    skeleton_color = (0, 255, 0)  # Green for primary+contact
                    joint_color = (0, 200, 0)
                elif is_primary:
                    skeleton_color = (255, 255, 0)  # Yellow for primary
                    joint_color = (255, 200, 0)  
                elif is_in_contact:
                    skeleton_color = (0, 255, 255)  # Cyan for contact
                    joint_color = (0, 200, 255)
                else:
                    skeleton_color = (150, 150, 150)  # Gray for others
                    joint_color = (100, 100, 100)
                
                # Draw connections
                thickness = 3 if is_primary else 2
                for connection in pose_connections:
                    idx1, idx2 = connection
                    if idx1 < len(keypoints) and idx2 < len(keypoints):
                        kp1, kp2 = keypoints[idx1], keypoints[idx2]
                        if kp1[2] > 0.3 and kp2[2] > 0.3:  # Confidence threshold
                            cv2.line(overlay_frame, 
                                   (int(kp1[0]), int(kp1[1])),
                                   (int(kp2[0]), int(kp2[1])),
                                   skeleton_color, thickness)
                
                # Draw keypoints
                for i, (x, y, conf) in enumerate(keypoints):
                    if conf > 0.3:  # Only draw confident keypoints
                        cv2.circle(overlay_frame, (int(x), int(y)), 4, joint_color, -1)
                        cv2.circle(overlay_frame, (int(x), int(y)), 4, (255, 255, 255), 1)

            # === RENDER LEGEND ===
            self._draw_visualization_legend(overlay_frame)
            
            return overlay_frame
            
        except Exception as e:
            self.logger.warning(f"Error creating debug overlay: {e}")
            return frame
    
    def _draw_visualization_legend(self, frame: np.ndarray):
        """Draw a legend explaining the visualization elements."""
        try:
            # Legend background
            legend_x, legend_y = 10, frame.shape[0] - 150
            legend_w, legend_h = 200, 140
            
            # Semi-transparent background
            cv2.rectangle(frame, (legend_x, legend_y), (legend_x + legend_w, legend_y + legend_h),
                         (0, 0, 0), -1)
            cv2.rectangle(frame, (legend_x, legend_y), (legend_x + legend_w, legend_y + legend_h),
                         (255, 255, 255), 1)
            
            # Title
            cv2.putText(frame, "VISUAL LEGEND", (legend_x + 5, legend_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Legend items
            legend_items = [
                ("C = Centroid", (255, 0, 255)),
                ("A = Action Point", (0, 255, 255)), 
                ("→ = Flow Vector", (0, 255, 255)),
                ("Green = Computed", (0, 255, 0)),
                ("Gray = Disregarded", (128, 128, 128)),
                ("Yellow = Interaction", (255, 255, 0)),
            ]
            
            y_offset = legend_y + 35
            for item, color in legend_items:
                cv2.putText(frame, item, (legend_x + 5, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                y_offset += 16
                
        except Exception as e:
            self.logger.warning(f"Error drawing legend: {e}")


# Registration function for the tracker system
def create_tracker() -> HybridIntelligenceTracker:
    """Factory function to create tracker instance."""
    return HybridIntelligenceTracker()