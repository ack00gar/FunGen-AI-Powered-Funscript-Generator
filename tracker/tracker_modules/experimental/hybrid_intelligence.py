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
from concurrent.futures import ThreadPoolExecutor
import traceback
from dataclasses import dataclass

try:
    from scipy import signal as scipy_signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from ..core.base_tracker import BaseTracker, TrackerMetadata, TrackerResult
    from ..helpers.signal_amplifier import SignalAmplifier
    from ..helpers.visualization import (
        TrackerVisualizationHelper, BoundingBox, PoseKeypoints
    )
except ImportError:
    from tracker.tracker_modules.core.base_tracker import BaseTracker, TrackerMetadata, TrackerResult
    from tracker.tracker_modules.helpers.signal_amplifier import SignalAmplifier
    from tracker.tracker_modules.helpers.visualization import (
        TrackerVisualizationHelper, BoundingBox, PoseKeypoints
    )

import config.constants as constants


class ButterworthFilter:
    """Low-pass Butterworth filter for smooth signal filtering without phase lag."""
    
    def __init__(self, cutoff_freq: float = 2.0, fs: float = 30.0, order: int = 2):
        """
        Initialize Butterworth filter.
        
        Args:
            cutoff_freq: Cutoff frequency in Hz
            fs: Sampling frequency in Hz (frame rate)
            order: Filter order (higher = steeper rolloff)
        """
        self.cutoff_freq = cutoff_freq
        self.fs = fs
        self.order = order
        
        if SCIPY_AVAILABLE:
            # Design the Butterworth filter
            nyquist = fs / 2
            normalized_cutoff = cutoff_freq / nyquist
            self.b, self.a = scipy_signal.butter(order, normalized_cutoff, btype='low', analog=False)
            
            # Initialize filter state for continuous filtering
            self.zi_primary = scipy_signal.lfilter_zi(self.b, self.a)
            self.zi_secondary = scipy_signal.lfilter_zi(self.b, self.a)
        else:
            # Fallback to simple EMA if scipy not available
            self.alpha = min(cutoff_freq / (fs / 2), 0.5)  # Convert to EMA alpha
            self.last_primary = None
            self.last_secondary = None
    
    def filter(self, primary_pos: float, secondary_pos: float) -> Tuple[float, float]:
        """Apply Butterworth filter to positions."""
        if SCIPY_AVAILABLE:
            # Use proper Butterworth filter
            filtered_primary, self.zi_primary = scipy_signal.lfilter(
                self.b, self.a, [primary_pos], zi=self.zi_primary
            )
            filtered_secondary, self.zi_secondary = scipy_signal.lfilter(
                self.b, self.a, [secondary_pos], zi=self.zi_secondary
            )
            return float(filtered_primary[0]), float(filtered_secondary[0])
        else:
            # Fallback to EMA
            if self.last_primary is None:
                self.last_primary = primary_pos
                self.last_secondary = secondary_pos
            
            self.last_primary = self.alpha * primary_pos + (1 - self.alpha) * self.last_primary
            self.last_secondary = self.alpha * secondary_pos + (1 - self.alpha) * self.last_secondary
            
            return self.last_primary, self.last_secondary
    
    def reset(self):
        """Reset filter state."""
        if SCIPY_AVAILABLE:
            self.zi_primary = scipy_signal.lfilter_zi(self.b, self.a)
            self.zi_secondary = scipy_signal.lfilter_zi(self.b, self.a)
        else:
            self.last_primary = None
            self.last_secondary = None


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
            
            # Performance settings
            self.processing_timeout = kwargs.get('processing_timeout', 0.5)  # 500ms default timeout
            
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
            
            # Adaptive YOLO frequency - simplified
            self.yolo_motion_history = deque(maxlen=5)  # Reduced history
            self.yolo_current_interval = self.yolo_base_update_interval
            
            # Thread pool for parallel processing  
            self.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="HI")
            
            # Memory pool for ROI operations
            self.roi_pool = {}
            
            # VR POV optimized priorities (using actual YOLO class names from constants.py)
            self.class_priorities = {
                # CRITICAL genital regions (highest priority for VR POV)
                'penis': 15, 'locked_penis': 15,  # HIGHEST priority
                'glans': 14,                       # Glans is critical for depth detection
                'pussy': 12,                       # Very high priority for penetration
                'butt': 10, 'anus': 10,          # High priority for anal scenes
                
                # Contact regions (important for interaction)
                'hand': 8,                         # Hand contact important
                'face': 6,                         # Face for oral scenes
                'breast': 4,                       # Moderate priority
                
                # Support regions (lower priority)
                'navel': 3,                        # Core movement indicator
                'foot': 2,                         # Footjob scenes
                'hips center': 2,                  # Hip movement
                
                # Generic classes (lowest priority)
                'person': 1, 'body': 1            # Fallback detections
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
            
            # Butterworth filter for superior temporal smoothing (replaces EMA)
            self.butterworth_filter = ButterworthFilter(
                cutoff_freq=2.0,  # 2 Hz cutoff - smooth but responsive
                fs=30.0,          # 30 fps assumed frame rate
                order=2           # 2nd order filter
            )
            
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
            
            # History tracking for debug
            self.position_history = deque(maxlen=60)  # Reduced history
            
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
        self.PENIS_SIZE_ANALYSIS_WINDOW_SECONDS = 20  # Window for height percentile calculation
        self.PENIS_SIZE_UPDATE_INTERVAL = 30  # Only update evolved size every N frames (1 sec at 30fps)
        self.PENIS_SIZE_MIN_SAMPLES = 15  # Minimum samples needed for reliable 95th percentile
        
        # THREAD SAFETY: Add lock for penis tracking state
        self._penis_state_lock = threading.RLock()  # Reentrant lock for nested access
        
        # Tracker state (protected by _penis_state_lock)
        self.locked_penis_tracker = {
            'box': None,  # Current locked penis box (x1, y1, x2, y2)
            'center': None, # Last known center point
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
            smoothing_alpha=0.3,  # SignalAmplifier uses internal EMA for different purpose
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
        
        # State for signal persistence and decay
        self.signal_state = {
            'frame_diff': 0.0,
            'yolo_weighted': 0.0,
            'optical_flow': 0.0,
            'oscillation': 0.0,
            'pose_activity': 0.0
        }
        self.signal_decay_rate = 0.85  # Decay to 85% of previous value per frame
        
        self.droi_state = 'DISCOVERY'
        self.current_droi_box = None # (x1, y1, x2, y2)
        self.target_droi_box = None
        self.DROI_UPDATE_INTERVAL = 300 # frames (10 seconds @ 30fps)
        self.droi_last_update_frame = 0

        # Position Identification
        self.current_sexual_position = 'discovery'
        self.interaction_history = deque(maxlen=150) # 5 seconds @ 30fps
        self.POSITION_UPDATE_INTERVAL = 150 # 5 seconds
        self.position_last_update_frame = 0
        self.DROI_EXPANSION_FACTORS = {
            'hand': {'w': 0.15, 'h': 0.25},
            'face': {'w': 0.15, 'h': 0.25},
            'pussy': {'w': 1.0, 'h_up': 2.0, 'h_down': 0.5},
            'butt': {'w': 1.0, 'h_up': 2.0, 'h_down': 0.5},
            'discovery': {'w': 0.5, 'h': 0.5}
        }

        # Thrust Detection
        self.thrust_detection_patch = None
        self.is_male_thrusting = False

        # Final output state - VR POV semantics: 100=no action, 0=full insertion
        self.last_primary_position = 100.0  # Start at no action (VR POV default)
        self.last_secondary_position = 50.0   # Secondary centered
        
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
            
            # Reset Butterworth filter state
            if hasattr(self, 'butterworth_filter'):
                self.butterworth_filter.reset()
            
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
            primary_pos = getattr(self, 'last_primary_position', 100.0)  # VR POV default
            secondary_pos = getattr(self, 'last_secondary_position', 50.0)
            self.logger.info(f"Hybrid Intelligence Tracker stopped at final positions: P={primary_pos:.1f}, S={secondary_pos:.1f}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to stop tracking: {e}")
            return False
    
    def _perform_memory_cleanup(self):
        """Lean memory cleanup."""
        try:
            # Aggressive oscillation history cleanup
            if hasattr(self, 'oscillation_history') and len(self.oscillation_history) > 30:
                # Keep only top 20 most active cells
                active_cells = [(k, np.mean(list(v)[-3:]) if v else 0) for k, v in self.oscillation_history.items()]
                active_cells.sort(key=lambda x: x[1], reverse=True)
                self.oscillation_history = {k: self.oscillation_history[k] for k, _ in active_cells[:20]}
                
            # Trim histories aggressively
            if hasattr(self, 'processing_times') and len(self.processing_times) > 30:
                self.processing_times = self.processing_times[-15:]
            if hasattr(self, 'primary_signal_history') and len(self.primary_signal_history) > 100:
                self.primary_signal_history = self.primary_signal_history[-50:]
            if hasattr(self, 'secondary_signal_history') and len(self.secondary_signal_history) > 100:
                self.secondary_signal_history = self.secondary_signal_history[-50:]
                
        except Exception as e:
            self.logger.warning(f"Memory cleanup error: {e}")

    def cleanup(self):
        """Clean up resources."""
        try:
            self.stop_tracking()
            
            # Cleanup resources
            self._perform_memory_cleanup()
            if hasattr(self, 'executor'): self.executor.shutdown(wait=False)
            if hasattr(self, 'gpu_flow') and self.gpu_flow: del self.gpu_flow
            
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
        Process a frame using the DROI-based hybrid intelligence approach.
        """
        start_time = time.time()
        
        try:
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                self._perform_memory_cleanup()
            
            self.current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # === STAGE 1: PARALLEL DETECTION & ANALYSIS ===
            futures = []
            pose_data = {}
            
            if self.prev_frame_gray is not None:
                # Launch all expensive tasks in parallel
                if self._should_run_yolo():
                    futures.append(self.executor.submit(self._detect_semantic_objects, frame))
                futures.append(self.executor.submit(self._estimate_pose, frame))

                # Collect results with progressive timeout handling
                for future in futures:
                    try:
                        result = future.result(timeout=self.processing_timeout)
                        if isinstance(result, list) and result and isinstance(result[0], SemanticRegion):
                            self.semantic_regions = result
                        elif isinstance(result, dict):
                            pose_data = result
                    except TimeoutError:
                        self.logger.warning(f"Frame {self.frame_count}: Processing timeout, using previous results")
                        # Cancel the future to free resources
                        future.cancel()
                        # Continue with previous frame's data
                        continue
            else:
                # First frame case
                self.semantic_regions = self._detect_semantic_objects(frame)
                pose_data = self._estimate_pose(frame)

            # === STAGE 2: DROI DEFINITION & MASKED ANALYSIS ===
            # Periodically update the DROI, otherwise keep it locked for stable analysis
            if (self.frame_count - self.droi_last_update_frame) > self.DROI_UPDATE_INTERVAL:
                self._update_dynamic_roi()
                self.droi_last_update_frame = self.frame_count

            if (self.frame_count - self.position_last_update_frame) > self.POSITION_UPDATE_INTERVAL:
                self._update_sexual_position()
                self.position_last_update_frame = self.frame_count

            self._smoothly_update_droi()
            droi_mask = self._create_droi_mask(self.current_frame_gray.shape)
            oscillation_signal = 0.0 # Initialize here

            if droi_mask is not None and self.prev_frame_gray is not None:
                masked_prev_gray = cv2.bitwise_and(self.prev_frame_gray, self.prev_frame_gray, mask=droi_mask)
                masked_current_gray = cv2.bitwise_and(self.current_frame_gray, self.current_frame_gray, mask=droi_mask)
                
                self.flow_analyses = self._compute_selective_flow(masked_prev_gray, masked_current_gray)
                oscillation_signal = self._analyze_oscillation_patterns() # This now uses flow, not change regions
            else:
                self.flow_analyses = []

            self.current_oscillation_intensity = oscillation_signal
            
            # === STAGE 3: SIGNAL FUSION & ACTION GENERATION ===
            self.last_pose_data = pose_data
            primary_pos, secondary_pos = self._fuse_signals(oscillation_signal, pose_data)
            action_log = self._generate_funscript_actions(primary_pos, secondary_pos, frame_time_ms, frame_index)
            
            # === STAGE 4: VISUALIZATION ===
            self._prepare_overlay_data(self.change_regions, pose_data)
            
            display_frame = self._create_debug_overlay(frame) if not getattr(self, 'use_external_visualization', False) else frame
            
            self.prev_frame_gray = self.current_frame_gray
            
            # Performance tracking & final result
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            debug_info = self._generate_debug_info(processing_time)
            status_msg = f"DROI: {self.droi_state} ({len(self.change_regions)} motion)"
            
            if self.debug_window_enabled:
                self._update_debug_window_data(processing_time)
            
            return TrackerResult(
                processed_frame=display_frame, action_log=action_log,
                debug_info=debug_info, status_message=status_msg
            )
            
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}\n{traceback.format_exc()}")
            return TrackerResult(
                processed_frame=frame, action_log=[],
                debug_info={"error": str(e)}, status_message=f"Error: {str(e)}"
            )
    
    def _detect_frame_changes_fast(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> List[ChangeRegion]:
        """Fast frame difference using minimal operations."""
        try:
            # Vectorized operations only
            diff = np.abs(curr_gray.astype(np.int16) - prev_gray.astype(np.int16)).astype(np.uint8)
            diff_blurred = cv2.GaussianBlur(diff, (5, 5), 0)
            binary_mask = (diff_blurred > self.frame_diff_threshold).astype(np.uint8) * 255
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, self.morph_kernel)
            
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            regions = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 50:  # Filter small regions
                    x, y, w, h = cv2.boundingRect(contour)
                    intensity = np.mean(diff[y:y+h, x:x+w])
                    regions.append(ChangeRegion(
                        x=x, y=y, width=w, height=h,
                        area=int(area),
                        intensity=float(intensity),
                        bbox=(x, y, x+w, y+h)
                    ))
                    
            return sorted(regions, key=lambda r: r.intensity * r.area, reverse=True)[:10]  # Top 10 only
            
        except Exception as e:
            self.logger.warning(f"Fast frame diff failed: {e}")
            return []
    
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
        """Adaptive YOLO frequency based on motion."""
        self.yolo_frame_counter += 1
        
        # Simplified adaptive logic
        if len(self.yolo_motion_history) >= 3:
            avg_motion = np.mean(list(self.yolo_motion_history))
            self.yolo_current_interval = 1 if avg_motion > 0.3 else (3 if avg_motion > 0.1 else 8)
        
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
        """Enhanced penis tracking using Stage 2 insights with IoU continuity and conceptual box evolution."""
        # Find current penis candidates with enhanced filtering
        penis_candidates = []
        target_classes = ['penis', 'locked_penis', 'glans']  # Include glans for better tracking
        
        for region in self.semantic_regions:
            if (region.class_name.lower() in target_classes and 
                region.confidence > 0.2 and  # Lower threshold for better detection
                hasattr(region, 'bbox')):
                # Calculate area and aspect ratio for better filtering
                width = region.bbox[2] - region.bbox[0]
                height = region.bbox[3] - region.bbox[1]
                area = width * height
                aspect_ratio = width / height if height > 0 else 0
                
                # Filter out obviously invalid detections
                if area > 100 and 0.3 < aspect_ratio < 5.0:  # Reasonable size and shape
                    penis_candidates.append({
                        'bbox': region.bbox,
                        'confidence': region.confidence,
                        'class_name': region.class_name,
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'center': ((region.bbox[0] + region.bbox[2]) / 2, (region.bbox[1] + region.bbox[3]) / 2)
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
                    tracker['center'] = ((selected_penis['bbox'][0] + selected_penis['bbox'][2]) / 2, (selected_penis['bbox'][1] + selected_penis['bbox'][3]) / 2)
                    tracker['confidence'] = selected_penis['confidence']
                    tracker['unseen_frames'] = 0
                    tracker['last_seen_timestamp'] = current_time
                    tracker['detection_frames'].append(self.frame_count)
                    
                    if not tracker['established_frame']:
                        tracker['established_frame'] = self.frame_count
        
        # 2. Enhanced Stage 2-style state management with conceptual box evolution
        recent_detections = len([f for f in tracker['detection_frames'] 
                               if self.frame_count - f <= self.PENIS_DETECTION_WINDOW])
        
        # Update conceptual box and size evolution tracking
        if selected_penis and tracker['box']:
            self._update_penis_conceptual_box(tracker, selected_penis)
        
        # Enhanced hysteresis logic with Stage 2 insights
        if not tracker['active']:
            # ACTIVATION: Require strong evidence with size consistency
            if (recent_detections >= self.PENIS_ACTIVATION_MIN_DETECTIONS and 
                tracker['box'] and 
                self._validate_penis_size_consistency(tracker)):
                tracker['active'] = True
                tracker['locked_frame'] = self.frame_count
                # Initialize conceptual box parameters
                if 'max_height' not in tracker:
                    tracker['max_height'] = tracker['box'][3] - tracker['box'][1]
                self.logger.info(f"Penis lock ACTIVATED: {recent_detections} detections, size validated")
        else:
            # DEACTIVATION: Enhanced logic with size tracking
            size_inconsistent = not self._validate_penis_size_consistency(tracker)
            if ((recent_detections < self.PENIS_DEACTIVATION_MIN_DETECTIONS and 
                 tracker['unseen_frames'] > self.PENIS_PATIENCE) or 
                (size_inconsistent and tracker['unseen_frames'] > 5)):
                self.logger.info(f"Penis lock DEACTIVATED: detections={recent_detections}, unseen={tracker['unseen_frames']}, size_issue={size_inconsistent}")
                tracker['active'] = False
                tracker['locked_frame'] = None
                # Preserve box/center for continuity but reset size tracking
                tracker.pop('max_height', None)
                tracker.pop('max_penetration_height', None)
        
        # 4. Update legacy compatibility fields with enhanced conceptual box
        if tracker['active']:
            # Use evolved conceptual box if available, otherwise fallback to raw detection
            display_box = self._get_evolved_penis_box() or tracker['box']
            
            # Update distance calculation reference points
            if display_box:
                tracker['penis_base_y'] = display_box[3]  # Bottom of conceptual box
                tracker['penis_center'] = (
                    (display_box[0] + display_box[2]) / 2,
                    (display_box[1] + display_box[3]) / 2
                )
            
            if display_box:
                self.locked_penis_box = {
                    'bbox': display_box,
                    'confidence': tracker['confidence'],
                    'timestamp': tracker['last_seen_timestamp']
                }
                self.penis_tracker_confidence = tracker['confidence']
                self.locked_penis_last_seen = tracker['last_seen_timestamp']
                
                # Add to history for continuity - use evolved box for consistent size
                penis_box_for_history = {
                    'bbox': display_box,
                    'confidence': tracker['confidence'],
                    'timestamp': current_time,
                    'area': (display_box[2] - display_box[0]) * (display_box[3] - display_box[1])
                }
                if not self.penis_box_history or self._is_significantly_different_box(penis_box_for_history):
                    self.penis_box_history.append(penis_box_for_history)
                    
                # Track size evolution for 95th percentile calculation - use RAW detection box
                if tracker['box']:
                    self._update_penis_size_evolution(tracker['box'])
            else:
                # Case where lock is active but box is gone, clear legacy fields
                self.locked_penis_box = None
                self.penis_tracker_confidence = 0.0
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
                candidates_info = [(c['class_name'], f"{c['confidence']:.2f}") for c in penis_candidates]
                self.logger.debug(f"Penis candidates: {candidates_info}")
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
        """
        Get penis box adjusted to 95th percentile size.
        Crucially, maintains this size at the last known position even if detection is temporarily lost.
        """
        tracker = self.locked_penis_tracker
        
        # We need a last known center and an evolved size to proceed
        if not tracker['center'] or not tracker['evolved_size']:
            return tracker['box']  # Fallback to raw box if we haven't learned anything yet

        evolved_width, evolved_height, _ = tracker['evolved_size']
        center_x, center_y = tracker['center']
        
        # Create the stable, evolved box centered on the last known position
        half_width = evolved_width / 2
        half_height = evolved_height / 2
        
        evolved_box = (
            center_x - half_width,
            center_y - half_height,
            center_x + half_width,
            center_y + half_height
        )
        
        return evolved_box
    
    def _update_penis_conceptual_box(self, tracker: Dict, selected_penis: Dict):
        """Update conceptual box with Stage 2-style evolution tracking."""
        if not tracker.get('box') or not selected_penis:
            return
        
        current_height = selected_penis['bbox'][3] - selected_penis['bbox'][1]
        
        # Update max height tracking (similar to Stage 2's max_height)
        if 'max_height' not in tracker:
            tracker['max_height'] = current_height
        else:
            tracker['max_height'] = max(tracker['max_height'], current_height)
        
        # Track penetration depth variations (for distance calculation)
        if 'max_penetration_height' not in tracker:
            tracker['max_penetration_height'] = current_height
        else:
            # Update max penetration height with some decay to adapt to scene changes
            tracker['max_penetration_height'] = max(
                tracker['max_penetration_height'] * 0.99,  # Slight decay
                current_height
            )
        
        # Update conceptual full stroke box (like Stage 2)
        x1, _, x2, y2_raw = selected_penis['bbox']
        conceptual_full_box = (x1, y2_raw - tracker['max_height'], x2, y2_raw)
        tracker['conceptual_box'] = conceptual_full_box
    
    def _validate_penis_size_consistency(self, tracker: Dict) -> bool:
        """Validate penis detection size consistency using Stage 2 insights."""
        if not tracker.get('box'):
            return False
        
        current_box = tracker['box']
        current_width = current_box[2] - current_box[0]
        current_height = current_box[3] - current_box[1]
        current_area = current_width * current_height
        
        # Check against historical max if available
        if 'max_height' in tracker:
            max_height = tracker['max_height']
            # Current height shouldn't be more than 3x the historical max
            if current_height > max_height * 3.0:
                return False
            # Current height shouldn't be less than 0.2x the historical max
            if current_height < max_height * 0.2:
                return False
        
        # Basic sanity checks
        if current_area < 50:  # Too small
            return False
        if current_area > 50000:  # Unrealistically large
            return False
        
        # Aspect ratio check
        aspect_ratio = current_width / current_height if current_height > 0 else 0
        if aspect_ratio < 0.2 or aspect_ratio > 10.0:  # Unrealistic shape
            return False
        
        return True
    
    def _calculate_normalized_distance_to_penis(self, contact_box: Tuple[float, float, float, float], 
                                               contact_class: str) -> float:
        """Calculate normalized distance using Stage 2's optimal reference points."""
        tracker = self.locked_penis_tracker
        
        if not tracker.get('active') or not tracker.get('conceptual_box'):
            return 50.0  # Default middle position
        
        # Use conceptual box like Stage 2
        penis_box = tracker['conceptual_box']
        penis_base_y = penis_box[3]  # Bottom of conceptual box
        max_height = tracker.get('max_height', 100.0)
        
        # Use Stage 2's optimal reference points for each interaction type
        if contact_class == 'face':
            contact_y_ref = contact_box[3]  # Bottom of face
        elif contact_class == 'hand':
            contact_y_ref = (contact_box[1] + contact_box[3]) / 2  # Center Y of hand
        elif contact_class in ['pussy', 'vagina']:
            contact_y_ref = (contact_box[1] + contact_box[3]) / 2  # Center Y of pussy
        elif contact_class in ['butt', 'anus']:
            contact_y_ref = (9 * contact_box[3] + contact_box[1]) / 10  # Mostly bottom
        else:  # breast, foot, etc.
            contact_y_ref = (contact_box[1] + contact_box[3]) / 2  # Center
        
        # Calculate distance from contact to penis base
        distance_to_base = abs(penis_base_y - contact_y_ref)
        
        # Normalize using max height as reference
        normalized_distance = min(100.0, (distance_to_base / max_height) * 100.0)
        
        return 100.0 - normalized_distance  # Invert for VR POV (100=close, 0=far)
    
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
    
    def _compute_selective_flow(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> List[FlowAnalysis]:
        """Compute optical flow for the entire masked DROI."""
        if not self.flow_dense or prev_gray.shape[0] < 12 or prev_gray.shape[1] < 12:
            return []

        try:
            flow = self.flow_dense.calc(prev_gray, curr_gray, None)
            
            magnitude = np.linalg.norm(flow, axis=2)
            avg_magnitude = np.mean(magnitude)
            avg_direction = np.mean(flow, axis=(0, 1))
            
            flow_std = np.std(magnitude)
            oscillation_strength = min(flow_std / (avg_magnitude + 1e-6), 2.0)

            analysis = FlowAnalysis(
                region_id=0,
                flow_magnitude=avg_magnitude,
                flow_direction=avg_direction,
                oscillation_strength=oscillation_strength,
                confidence=1.0 # Confidence is 1.0 as it's our only source
            )
            
            return [analysis]
        
        except Exception as e:
            self.logger.warning(f"Optical flow computation failed: {e}")
            return []

    def _analyze_oscillation_patterns(self) -> float:
        """Analyzes oscillation based on the global flow analysis."""
        if not self.flow_analyses:
            return 0.0
        
        # Use the oscillation strength calculated in the main flow analysis
        # This is simpler and more directly tied to the signal source
        oscillation_strength = self.flow_analyses[0].oscillation_strength
        return min(oscillation_strength * self.oscillation_sensitivity, 1.0)
    
    
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

    def _update_sexual_position(self):
        """Analyzes interaction history to deduce the current sexual position."""
        if not self.interaction_history:
            self.current_sexual_position = 'discovery'
            return

        from collections import Counter
        interaction_counts = Counter(self.interaction_history)
        
        # Find the most common interaction
        most_common_interaction, count = interaction_counts.most_common(1)[0]

        # Set position based on the dominant interaction type
        if count > len(self.interaction_history) * 0.6: # Needs to be dominant
            if most_common_interaction in ['pussy', 'butt']:
                # Simple for now, can be refined with orientation later
                self.current_sexual_position = 'missionary' # or doggy
            elif most_common_interaction == 'hand':
                self.current_sexual_position = 'handjob'
            elif most_common_interaction == 'face':
                self.current_sexual_position = 'blowjob'
            else:
                self.current_sexual_position = 'discovery'
        else:
            self.current_sexual_position = 'discovery'
        
        self.logger.info(f"Updated sexual position to: {self.current_sexual_position}")

    def _analyze_thrust_patch(self) -> bool:
        """Analyzes optical flow in the thrust patch to detect male thrusting."""
        if self.thrust_detection_patch is None or self.prev_frame_gray is None:
            return False

        x1, y1, x2, y2 = map(int, self.thrust_detection_patch)
        h, w = self.current_frame_gray.shape
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return False

        prev_roi = self.prev_frame_gray[y1:y2, x1:x2]
        curr_roi = self.current_frame_gray[y1:y2, x1:x2]

        flow = self.flow_dense.calc(prev_roi, curr_roi, None)
        magnitude = np.linalg.norm(flow, axis=2)
        avg_magnitude = np.mean(magnitude)

        # If average motion magnitude in this patch is high, assume male is thrusting
        is_thrusting = avg_magnitude > 2.5 # Threshold needs tuning
        self.is_male_thrusting = is_thrusting
        return is_thrusting

    def _smoothly_update_droi(self):
        """Interpolates the current DROI towards the target DROI for smooth transitions."""
        if self.target_droi_box is None:
            return

        if self.current_droi_box is None:
            self.current_droi_box = self.target_droi_box
            return

        # Simple linear interpolation
        alpha = 0.05 # 5% movement per frame
        new_box = [
            (1 - alpha) * self.current_droi_box[i] + alpha * self.target_droi_box[i]
            for i in range(4)
        ]
        self.current_droi_box = tuple(new_box)

    def _update_dynamic_roi(self):
        """
        STATE-DRIVEN DROI CALCULATION
        Sets the target_droi_box and thrust_detection_patch based on the identified sexual position.
        """
        is_penis_locked = self._is_penis_active() and self.penis_tracker_confidence > 0.5
        penis_box = self._get_evolved_penis_box()

        if not is_penis_locked or not penis_box:
            self.droi_state = 'DISCOVERY'
            self.target_droi_box = None
            self.thrust_detection_patch = None
            return

        px1, py1, px2, py2 = penis_box
        pw, ph = px2 - px1, py2 - py1

        # Define the thrust patch relative to the penis box base
        thrust_patch_height = ph * 0.5
        self.thrust_detection_patch = (px1, py2, px2, py2 + thrust_patch_height)

        # Find best intersecting interactive region
        interactive_regions = [r for r in self.semantic_regions if r.class_name.lower() in ['pussy', 'butt', 'hand', 'face']]
        best_intersecting_region = None
        max_iou = 0.0
        if interactive_regions:
            for region in interactive_regions:
                iou = TrackerVisualizationHelper.calculate_iou(penis_box, region.bbox)
                if iou > max_iou:
                    max_iou = iou
                    best_intersecting_region = region
        
        if max_iou > 0.01 and best_intersecting_region:
            new_droi_state = 'LOCKED'
            self.interaction_history.append(best_intersecting_region.class_name.lower())
            
            # Use the CURRENT interaction for sizing, not the historical position
            current_interaction_type = best_intersecting_region.class_name.lower()
            exp_factors = self.DROI_EXPANSION_FACTORS.get(current_interaction_type, self.DROI_EXPANSION_FACTORS['discovery'])

            if current_interaction_type in ['pussy', 'butt']:
                # Anchor DROI to penis base and expand upwards
                droi_w = pw * (1 + exp_factors['w'])
                droi_h = ph * (1 + exp_factors['h_up'] + exp_factors['h_down'])
                center_x = (px1 + px2) / 2
                new_x1 = center_x - droi_w / 2
                new_x2 = center_x + droi_w / 2
                # Anchor at base and extend up
                new_y2 = py2 + ph * exp_factors['h_down']
                new_y1 = new_y2 - droi_h
                new_target_box = (new_x1, new_y1, new_x2, new_y2)
            else: # handjob, blowjob
                # DROI is just the penis box itself
                new_target_box = penis_box

        else:
            new_droi_state = 'PROXIMITY'
            new_target_box = penis_box # Default to penis box in proximity

        if new_droi_state != self.droi_state:
            self.logger.info(f"DROI state changed: {self.droi_state} -> {new_droi_state}")
            self.droi_state = new_droi_state

        if new_target_box != self.target_droi_box:
            self.logger.info(f"DROI target updated. New Box: ({new_target_box[0]:.0f}, {new_target_box[1]:.0f}, {new_target_box[2]:.0f}, {new_target_box[3]:.0f}))")
            self.target_droi_box = new_target_box

    def _create_droi_mask(self, frame_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Creates a binary mask from the current DROI box."""
        if self.current_droi_box is None:
            return None
        
        mask = np.zeros(frame_shape, dtype=np.uint8)
        x1, y1, x2, y2 = self.current_droi_box
        
        # Clip coordinates to be within frame dimensions
        h, w = frame_shape
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        
        if x1 < x2 and y1 < y2:
            mask[y1:y2, x1:x2] = 255
        
        return mask
    
    def _fuse_signals(self, oscillation_signal: float, pose_data: Dict[str, Any] = None) -> Tuple[float, float]:
        """STATE-DRIVEN SIGNAL SELECTION for a clean, context-aware signal."""
        
        primary_pos = self.last_primary_position
        secondary_pos = self.last_secondary_position
        fused_signal = 0.0

        # === STATE-BASED SIGNAL SELECTION ===
        if self.droi_state == 'LOCKED':
            # In LOCKED state, we trust ONLY the vertical optical flow within the DROI.
            if self.flow_analyses:
                # Extract vertical component of flow (y-axis)
                vertical_flows = [fa.flow_direction[1] for fa in self.flow_analyses if fa.confidence > 0.4]
                if vertical_flows:
                    # Average the vertical flow and scale it to represent funscript motion
                    # A positive flow (downward motion) should decrease the funscript position (move down)
                    fused_signal = np.mean(vertical_flows) * 1.5 # Amplification factor

        elif self.droi_state == 'PROXIMITY':
            # In PROXIMITY, we use optical flow magnitude from the partner's body motion.
            if self.flow_analyses:
                flow_magnitudes = [fa.flow_magnitude for fa in self.flow_analyses if fa.confidence > 0.3]
                if flow_magnitudes:
                    fused_signal = np.mean(flow_magnitudes) * 0.5 # Use magnitude, with some scaling

        elif self.droi_state == 'DISCOVERY':
            # In DISCOVERY, we also rely on optical flow magnitude to find the action.
            if self.flow_analyses:
                flow_magnitudes = [fa.flow_magnitude for fa in self.flow_analyses if fa.confidence > 0.2]
                if flow_magnitudes:
                    fused_signal = np.mean(flow_magnitudes) * 0.5 # Use magnitude, with some scaling

        # === POSITION CALCULATION ===
        if fused_signal != 0.0:
            # Invert signal if male is detected to be thrusting
            if self.is_male_thrusting:
                fused_signal *= -1

            # Context-Aware Amplification
            droi_h = self.current_droi_box[3] - self.current_droi_box[1] if self.current_droi_box else frame.shape[0]
            # Smaller DROI implies smaller movements, so we need more amplification.
            amplification_factor = max(1.0, 150.0 / droi_h) # Inverse relationship
            fused_signal *= amplification_factor

            # Map the fused signal to the funscript position range
            position_offset = fused_signal * 50.0
            primary_pos = 50.0 - position_offset
        else:
            # No signal, decay towards the neutral position
            primary_pos = self.last_primary_position + (50.0 - self.last_primary_position) * 0.02

        # For secondary axis, always use horizontal flow if available, regardless of state
        if self.flow_analyses:
            horizontal_flows = [fa.flow_direction[0] for fa in self.flow_analyses if fa.confidence > 0.3]
            if horizontal_flows:
                avg_horizontal = np.mean(horizontal_flows)
                secondary_pos = 50.0 + (avg_horizontal * 25.0)
        else:
            secondary_pos = self.last_secondary_position + (50.0 - self.last_secondary_position) * 0.02

        # Apply final smoothing and update history
        primary_pos, secondary_pos = self.butterworth_filter.filter(primary_pos, secondary_pos)
        
        self.last_primary_position = np.clip(primary_pos, 0, 100)
        self.last_secondary_position = np.clip(secondary_pos, 0, 100)
        
        self.primary_signal_history.append(self.last_primary_position)
        self.secondary_signal_history.append(self.last_secondary_position)
        self.position_history.append(self.last_primary_position)
        
        return self.last_primary_position, self.last_secondary_position
    
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
                # Improved pose connections with anatomical facial structure
                pose_connections = [
                    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
                    (11, 12), (5, 11), (6, 12),               # Torso
                    (11, 13), (13, 15), (12, 14), (14, 16),   # Legs
                    # Facial connections:
                    (0, 1),  # Nose to left eye
                    (0, 2),  # Nose to right eye
                    (1, 3),  # Left eye to left ear
                    (2, 4),  # Right eye to right ear
                    (3, 5),  # Left ear to left shoulder
                    (4, 6)   # Right ear to right shoulder
                ]
                
                for connection in pose_connections:
                    pt1_idx, pt2_idx = connection
                    if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
                        pt1, pt2 = keypoints[pt1_idx], keypoints[pt2_idx]
                        
                        # Check confidence AND that keypoints are not at (0,0)
                        if (pt1[2] > 0.5 and pt2[2] > 0.5 and
                            (pt1[0] != 0 or pt1[1] != 0) and
                            (pt2[0] != 0 or pt2[1] != 0)):
                            x1, y1 = int(pt1[0]), int(pt1[1])
                            x2, y2 = int(pt2[0]), int(pt2[1])
                            thickness = 3 if is_primary else 2
                            cv2.line(frame, (x1, y1), (x2, y2), skeleton_color, thickness)
                
                # Draw keypoints
                for keypoint in keypoints:
                    # Only draw confident keypoints that are not at (0,0)
                    if keypoint[2] > 0.5 and (keypoint[0] != 0 or keypoint[1] != 0):
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
        try:
            # Convert semantic regions to bounding boxes
            self.logger.debug(f"Converting {len(self.semantic_regions)} semantic regions to boxes")
            boxes = TrackerVisualizationHelper.convert_semantic_regions_to_boxes(
                self.semantic_regions
            )
            self.logger.debug(f"Converted to {len(boxes)} bounding boxes")
        except Exception as e:
            self.logger.error(f"Error converting semantic regions to boxes: {e}")
            boxes = []
        
        # Add locked penis box if available
        if self.locked_penis_box:
            # locked_penis_box is a dict with 'bbox' key
            if isinstance(self.locked_penis_box, dict) and 'bbox' in self.locked_penis_box:
                bbox = self.locked_penis_box['bbox']
                confidence = self.locked_penis_box.get('confidence', self.penis_tracker_confidence)
                
                # Ensure bbox has exactly 4 values to prevent unpacking errors
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
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
                    self.logger.warning(f"Invalid locked_penis_box bbox format: {bbox} (type: {type(bbox)}, len: {len(bbox) if hasattr(bbox, '__len__') else 'N/A'})")
            else:
                self.logger.warning(f"Invalid locked_penis_box format: {type(self.locked_penis_box)}")
        
        # Convert pose data to keypoints
        pose_keypoints = []
        if pose_data:
            try:
                self.logger.debug(f"Pose data keys: {list(pose_data.keys())}")
                primary_person = pose_data.get('primary_person')
                all_persons = pose_data.get('all_persons', [])
                self.logger.debug(f"Primary person: {primary_person is not None}")
                self.logger.debug(f"All persons: {len(all_persons)} items")
                
                pose_keypoints = TrackerVisualizationHelper.convert_pose_data_to_keypoints(pose_data)
                self.logger.debug(f"Converted pose keypoints: {len(pose_keypoints)}")
            except Exception as e:
                self.logger.error(f"Error converting pose data to keypoints: {e}")
                pose_keypoints = []
            
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
                # Ensure start_point has exactly 2 elements
                if isinstance(start_point, (list, tuple)) and len(start_point) >= 2:
                    end_x = start_point[0] + magnitude * math.cos(direction) * 20  # Scale for visibility
                    end_y = start_point[1] + magnitude * math.sin(direction) * 20
                else:
                    self.logger.warning(f"Invalid start_point format: {start_point}, skipping flow vector")
                    continue
                
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
            # DROI Visualization
            droi_box=self.current_droi_box,
            droi_state=self.droi_state,
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
    
    
    def _create_debug_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Create internal debug overlay on frame for fallback visualization.
        
        This method renders overlays directly on the frame when external
        visualization is not available.
        """
        try:
            # Use TrackerVisualizationHelper for consistent rendering
            from tracker.tracker_modules.helpers.visualization import TrackerVisualizationHelper
            
            # Ensure overlay_data exists
            if not hasattr(self, 'overlay_data'):
                return frame
                
            overlay_frame = frame.copy()
            
            # === RENDER YOLO BOXES ===
            for box in self.overlay_data.get('yolo_boxes', []):
                # Get color (RGB to BGR for OpenCV)
                color = box.get('color_override') or (128, 128, 128)
                bgr_color = (color[2], color[1], color[0])  # RGB to BGR

                # Get bounding box coordinates
                bbox = box.get('bbox', [0, 0, 0, 0])
                x1, y1, x2, y2 = map(int, bbox)

                # Draw bounding box
                thickness = int(box.get('thickness_override', 2))
                cv2.rectangle(overlay_frame,
                            (x1, y1),
                            (x2, y2),
                            bgr_color, thickness)

                # Draw label with class name and confidence
                class_name = box.get('class_name', 'unknown')
                confidence = box.get('confidence', 0.0)
                label = f"{class_name}:{confidence:.2f}"

                label_suffix = box.get('label_suffix')
                if label_suffix:
                    label += f" {label_suffix}"

                label_color = (0, 255, 255) if class_name.lower() in ['penis', 'locked_penis', 'pussy', 'butt'] else bgr_color
                cv2.putText(overlay_frame, label,
                          (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, label_color, 1)

            # === RENDER DROI STATE ===
            droi_box = self.overlay_data.get('droi_box')
            droi_state = self.overlay_data.get('droi_state', 'N/A')
            if droi_box:
                state_colors = {
                    'LOCKED': (0, 255, 0), # Green
                    'PROXIMITY': (0, 255, 255), # Yellow
                    'DISCOVERY': (255, 0, 0) # Blue
                }
                color = state_colors.get(droi_state, (255, 255, 255))
                x1, y1, x2, y2 = map(int, droi_box)
                
                # Draw DROI box
                cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw DROI state label
                label = f"DROI: {droi_state}"
                cv2.putText(overlay_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
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
                # Check for both 'start_point'/'end_point' and 'start'/'end' formats
                start_key = 'start_point' if 'start_point' in vector else 'start'
                end_key = 'end_point' if 'end_point' in vector else 'end'
                
                if start_key in vector and end_key in vector:
                    start_point = vector[start_key]
                    end_point = vector[end_key]
                    color = vector.get('color', (0, 255, 255))
                    thickness = vector.get('thickness', 2)
                    
                    cv2.arrowedLine(overlay_frame,
                                   (int(start_point[0]), int(start_point[1])),
                                   (int(end_point[0]), int(end_point[1])),
                                   color, thickness)
            
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
                # Draw skeleton connections with anatomical facial structure
                pose_connections = [
                    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
                    (11, 12), (5, 11), (6, 12),               # Torso
                    (11, 13), (13, 15), (12, 14), (14, 16),   # Legs
                    # Anatomically correct facial connections:
                    (0, 1),  # Nose to left eye
                    (0, 2),  # Nose to right eye
                    (1, 3),  # Left eye to left ear
                    (2, 4),  # Right eye to right ear
                    (3, 5),  # Left ear to left shoulder
                    (4, 6)   # Right ear to right shoulder
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
                        # Check confidence AND that keypoints are not at (0,0)
                        if (kp1[2] > 0.3 and kp2[2] > 0.3 and 
                            (kp1[0] != 0 or kp1[1] != 0) and 
                            (kp2[0] != 0 or kp2[1] != 0)):
                            cv2.line(overlay_frame, 
                                   (int(kp1[0]), int(kp1[1])),
                                   (int(kp2[0]), int(kp2[1])),
                                   skeleton_color, thickness)
                
                # Draw keypoints
                for i, (x, y, conf) in enumerate(keypoints):
                    # Only draw confident keypoints that are not at (0,0)
                    if conf > 0.3 and (x != 0 or y != 0):
                        cv2.circle(overlay_frame, (int(x), int(y)), 4, joint_color, -1)
                        cv2.circle(overlay_frame, (int(x), int(y)), 4, (255, 255, 255), 1)

            return overlay_frame
            
        except Exception as e:
            self.logger.warning(f"Error creating debug overlay: {e}")
            return frame
    


# Registration function for the tracker system
def create_tracker() -> HybridIntelligenceTracker:
    """Factory function to create tracker instance."""
    return HybridIntelligenceTracker()