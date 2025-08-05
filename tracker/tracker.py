import cv2
from collections import Counter, deque
import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Any
from ultralytics import YOLO
import logging
import os

from funscript import DualAxisFunscript
from config import constants
from config.constants_colors import RGBColors
from config.element_group_colors import AppGUIColors
from application.utils import ModelPool


class ROITracker:
    def __init__(self,
                 app_logic_instance: Optional[Any],
                 tracker_model_path: str,
                 pose_model_path: Optional[str] = None,
                 confidence_threshold: float = constants.DEFAULT_TRACKER_CONFIDENCE_THRESHOLD,
                 roi_padding: int = constants.DEFAULT_TRACKER_ROI_PADDING,
                 roi_update_interval: int = constants.DEFAULT_ROI_UPDATE_INTERVAL,
                 roi_smoothing_factor: float = constants.DEFAULT_ROI_SMOOTHING_FACTOR,
                 dis_flow_preset: str = constants.DEFAULT_DIS_FLOW_PRESET,
                 dis_finest_scale: Optional[int] = constants.DEFAULT_DIS_FINEST_SCALE,
                 target_size_preprocess: Tuple[int, int] = (constants.YOLO_INPUT_SIZE, constants.YOLO_INPUT_SIZE),
                 flow_history_window_smooth: int = constants.DEFAULT_FLOW_HISTORY_SMOOTHING_WINDOW,
                 adaptive_flow_scale: bool = True,
                 use_sparse_flow: bool = False,
                 max_frames_for_roi_persistence: int = constants.DEFAULT_ROI_PERSISTENCE_FRAMES,
                 base_amplification_factor: float = constants.DEFAULT_LIVE_TRACKER_BASE_AMPLIFICATION,
                 class_specific_amplification_multipliers: Optional[Dict[str, float]] = None,
                 logger: Optional[logging.Logger] = None,
                 inversion_detection_split_ratio: float = constants.INVERSION_DETECTION_SPLIT_RATIO,
                 video_type_override: Optional[str] = None,
                 load_models_on_init: bool = True
                 ):
        self.app = app_logic_instance # Can be None if instantiated by Stage 3
        self.video_type_override = video_type_override

        if logger:
            self.logger = logger
        elif self.app and hasattr(self.app, 'logger'):
            self.logger = self.app.logger
        else:
            self.logger = logging.getLogger('ROITracker_fallback')
            if not self.logger.handlers:
                self.logger.addHandler(logging.NullHandler())
            self.logger.warning("No external logger provided to ROITracker, using fallback NullHandler.",
                                extra={'status_message': False})

        self.tracking_mode: str = "YOLO_ROI"
        self.user_roi_fixed: Optional[Tuple[int, int, int, int]] = None
        self.user_roi_initial_point_relative: Optional[Tuple[float, float]] = None
        self.user_roi_tracked_point_relative: Optional[Tuple[float, float]] = None
        self.prev_gray_user_roi_patch: Optional[np.ndarray] = None
        self.user_roi_current_flow_vector: Tuple[float, float] = (0.0, 0.0)

        # --- Oscillation Area Selection ---
        self.oscillation_area_fixed: Optional[Tuple[int, int, int, int]] = None
        self.oscillation_area_initial_point_relative: Optional[Tuple[float, float]] = None
        self.oscillation_area_tracked_point_relative: Optional[Tuple[float, float]] = None
        self.prev_gray_oscillation_area_patch: Optional[np.ndarray] = None
        self.oscillation_ema_alpha: float = 0.3

        # --- Static grid storage for oscillation area ---
        self.oscillation_grid_blocks: List[Tuple[int, int, int, int]] = []  # List of (x, y, w, h) for each grid block
        # --- Always-available set of active grid blocks ---
        self.oscillation_active_block_positions: set = set()

        self.enable_user_roi_sub_tracking: bool = True
        self.user_roi_tracking_box_size: Tuple[int, int] = (5, 5)

        # Store paths
        self.det_model_path = tracker_model_path
        self.pose_model_path = pose_model_path

        # Initialize ModelPool for efficient memory management
        self.model_pool = ModelPool(max_gpu_memory_ratio=0.8, logger=self.logger)
        
        # Legacy model holders (kept for compatibility but will be None)
        self.yolo: Optional[YOLO] = None
        self.yolo_pose: Optional[YOLO] = None
        self.classes = []

        # Load class names if detection model is available
        if load_models_on_init and self.det_model_path and os.path.exists(self.det_model_path):
            self._load_class_names()

        self.confidence_threshold = confidence_threshold
        self.roi: Optional[Tuple[int, int, int, int]] = None
        self.roi_padding = roi_padding
        self.roi_update_interval = roi_update_interval
        self.roi_smoothing_factor = max(0.0, min(1.0, roi_smoothing_factor))
        self.internal_frame_counter: int = 0
        self.max_frames_for_roi_persistence = max_frames_for_roi_persistence
        self.frames_since_target_lost: int = 0

        self.base_amplification_factor = base_amplification_factor
        self.class_specific_amplification_multipliers = class_specific_amplification_multipliers if class_specific_amplification_multipliers is not None else constants.DEFAULT_CLASS_AMP_MULTIPLIERS
        self.logger.info(f"Base Amplification: {self.base_amplification_factor}x")
        self.logger.info(f"Class Specific Amp Multipliers: {self.class_specific_amplification_multipliers}")

        # Track video source information for user feedback
        self._last_video_source_status: Optional[Dict[str, Any]] = None

        # Ensure these are initialized regardless of self.app
        self.output_delay_frames: int = self.app.tracker.output_delay_frames if self.app and hasattr(self.app, 'tracker') else 0
        self.current_video_fps_for_delay: float = self.app.tracker.current_video_fps_for_delay if self.app and hasattr(self.app, 'tracker') else 30.0
        # Sensitivity and offsets should also be settable or taken from defaults if no app
        self.y_offset = self.app.tracker.y_offset if self.app and hasattr(self.app, 'tracker') else constants.DEFAULT_LIVE_TRACKER_Y_OFFSET
        self.x_offset = self.app.tracker.x_offset if self.app and hasattr(self.app, 'tracker') else constants.DEFAULT_LIVE_TRACKER_X_OFFSET
        self.sensitivity = self.app.tracker.sensitivity if self.app and hasattr(self.app, 'tracker') else constants.DEFAULT_LIVE_TRACKER_SENSITIVITY

        self.dis_flow_preset = dis_flow_preset
        self.dis_finest_scale = dis_finest_scale
        dis_preset_map = {
            "ULTRAFAST": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
            "FAST": cv2.DISOPTICAL_FLOW_PRESET_FAST,
            "MEDIUM": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
        }
        try:
            selected_preset_cv = dis_preset_map.get(self.dis_flow_preset.upper(), cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            self.flow_dense = cv2.DISOpticalFlow_create(selected_preset_cv)
            if self.dis_finest_scale is not None:
                self.flow_dense.setFinestScale(self.dis_finest_scale)
        except AttributeError:
            self.logger.warning("cv2.DISOpticalFlow_create not found or preset invalid. Optical flow might not work.")
            self.flow_dense = None

        self.prev_gray_main_roi: Optional[np.ndarray] = None
        self.funscript = DualAxisFunscript(logger=self.logger)
        self.tracking_active: bool = False
        self.start_time_tracking: float = 0
        self.target_size_preprocess = target_size_preprocess
        self.penis_max_size_history: List[float] = []
        self.penis_size_history_window: int = 300
        self.penis_last_known_box: Optional[Tuple[int, int, int, int]] = None
        self.primary_flow_history_smooth: List[float] = []
        self.secondary_flow_history_smooth: List[float] = []
        self.flow_history_window_smooth = flow_history_window_smooth
        self.adaptive_flow_scale = adaptive_flow_scale
        self.flow_min_primary_adaptive: float = -1.0
        self.flow_max_primary_adaptive: float = 1.0
        self.flow_min_secondary_adaptive: float = -1.0
        self.flow_max_secondary_adaptive: float = 1.0
        self.use_sparse_flow = use_sparse_flow
        self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.prev_features_main_roi: Optional[np.ndarray] = None
        self.main_interaction_class: Optional[str] = None
        self.CLASS_PRIORITY = {"pussy": 0, "anus": 0, "butt": 1, "face": 2, "hand": 3, "breast": 4, "foot": 5}
        # CLASS_COLORS is now imported from constants_colors.py
        self.INTERACTION_CLASSES = ["pussy", "anus", "butt", "face", "hand", "breast", "foot"]
        self.class_history: List[Optional[str]] = []
        self.class_stability_window: int = 10
        self.last_interaction_time: float = 0
        self.show_roi: bool = True
        self.show_flow: bool = True
        self.show_all_boxes: bool = True
        self.show_tracking_points: bool = True
        self.show_masks: bool = False
        self.show_stats: bool = False

        # Properties for thrust vs. ride detection
        self.enable_inversion_detection: bool = True  # Master switch for this feature
        # The ratio to split the ROI for inversion detection.
        # e.g., 3.0 means the lower 1/3 is compared against the upper 2/3.
        self.inversion_detection_split_ratio = inversion_detection_split_ratio
        self.motion_mode: str = 'undetermined'  # Can be 'thrusting', 'riding', or 'undetermined'
        self.motion_mode_history: List[str] = []
        self.motion_mode_history_window: int = 30  # Buffer size, e.g., 1s at 30fps
        self.motion_inversion_threshold: float = 1.2  # Motion in one region must be 20% greater than the other to trigger a change

        self.oscillation_cell_persistence = {}  # Tracks active cells over time
        self.OSCILLATION_PERSISTENCE_FRAMES = 4  # A cell stays active for 4 frames without motion

        # --- Attributes for Oscillation Detector ---
        self.oscillation_grid_size: int = self.app.app_settings.get("oscillation_detector_grid_size", 20) if self.app else 20
        self.oscillation_block_size: int = constants.YOLO_INPUT_SIZE // self.oscillation_grid_size
        self.oscillation_history_seconds: float = 2.0
        self.oscillation_history: Dict[Tuple[int, int], deque] = {}
        self.prev_gray_oscillation: Optional[np.ndarray] = None
        # --- Attributes for live amplification and smoothing ---
        self.live_amp_enabled = self.app.app_settings.get("live_oscillation_dynamic_amp_enabled", True) if self.app else True
        # --- Initialize deque with a default maxlen. It will be resized in start_tracking. ---
        self.oscillation_position_history = deque(maxlen=120) # Default to 4 seconds @ 30fps
        self.oscillation_last_known_pos: float = 50.0
        self.oscillation_last_known_secondary_pos = 50.0
        self.oscillation_last_active_time = 0
        self.oscillation_hold_duration_ms = 250  # Hold for 250ms before decaying
        self.oscillation_ema_alpha: float = 0.3 # Smoothing factor for the final signal
        self.oscillation_history_max_len: int = 60

        # --- Oscillation sensitivity control ---
        self.oscillation_sensitivity: float = self.app.app_settings.get("oscillation_detector_sensitivity", 1.0) if self.app else 1.0

        self.last_frame_time_sec_fps: Optional[float] = None
        self.current_fps: float = 0.0
        self.current_effective_amp_factor: float = self.base_amplification_factor
        self.stats_display: List[str] = []
        self.logger.info(
            f"Tracker fully initialized (ROI Persistence: {self.max_frames_for_roi_persistence} frames, ROI Smoothing: {self.roi_smoothing_factor}). App instance {'provided' if self.app else 'not provided (e.g. S3 mode)'}.")

    def _is_vr_video(self) -> bool:
        """Determines if the video is VR, using the override if available."""
        if self.video_type_override:
            return self.video_type_override == 'VR'
        if self.app and hasattr(self.app, 'processor') and self.app.processor:
            return self.app.processor.determined_video_type == 'VR'
        return False

    def _load_class_names(self):
        """Load class names from detection model for compatibility."""
        if self.det_model_path and os.path.exists(self.det_model_path):
            try:
                with self.model_pool.get_model(self.det_model_path, 'detect') as model:
                    self.classes = model.names
                self.logger.info(f"Loaded class names from detection model: {self.det_model_path}")
            except Exception as e:
                self.logger.error(f"Could not load class names from {self.det_model_path}: {e}")
                self.classes = []
        else:
            self.logger.warning("Detection model path not set or file does not exist.")
            self.classes = []

    def _load_models(self):
        """Legacy method - now just loads class names for compatibility."""
        self.logger.warning("_load_models() is deprecated. Models are now loaded on-demand via ModelPool.")
        self._load_class_names()

    def unload_detection_model(self):
        """Unloads the detection model to free up memory."""
        # Legacy compatibility - ModelPool handles this automatically
        self.yolo = None
        self.logger.info("Tracker: Detection model reference cleared.")

    def unload_pose_model(self):
        """Unloads the pose model to free up memory."""
        # Legacy compatibility - ModelPool handles this automatically
        self.yolo_pose = None
        self.logger.info("Tracker: Pose model reference cleared.")
    
    def unload_all_models(self):
        """Unload all models and clear GPU memory."""
        self.model_pool.clear_all_models()
        self.yolo = None
        self.yolo_pose = None
        self.logger.info("Tracker: All models unloaded and GPU memory cleared.")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory usage statistics from the model pool."""
        return self.model_pool.get_memory_stats()

    def histogram_calculate_flow_in_sub_regions(self, patch_gray: np.ndarray, prev_patch_gray: Optional[np.ndarray]) \
            -> Tuple[float, float, float, float, Optional[np.ndarray]]:
        """
        Calculates optical flow. For VR, it first identifies a dominant motion
        region (upper for riding, lower for thrusting) and then runs a robust
        Histogram of Flow calculation only on that sub-region.
        """
        # 1. Handle edge cases
        if self.flow_dense is None or prev_patch_gray is None or prev_patch_gray.shape != patch_gray.shape:
            return 0.0, 0.0, 0.0, 0.0, None

        prev_patch_cont = np.ascontiguousarray(prev_patch_gray)
        patch_cont = np.ascontiguousarray(patch_gray)

        # 2. Calculate optical flow for the entire patch once
        flow = self.flow_dense.calc(prev_patch_cont, patch_cont, None)
        if flow is None:
            return 0.0, 0.0, 0.0, 0.0, None

        h, w, _ = flow.shape
        # is_vr_video = self.app and hasattr(self.app, 'processor') and self.app.processor.determined_video_type == 'VR'
        is_vr_video = self._is_vr_video()

        # 3. For VR, determine the dominant motion region BEFORE main calculation
        dominant_flow_region = flow  # Default to the full ROI
        lower_magnitude = 0.0
        upper_magnitude = 0.0

        if is_vr_video and self.inversion_detection_split_ratio > 1.0:
            lower_region_h = int(h / self.inversion_detection_split_ratio)
            if lower_region_h > 0 and lower_region_h < h:
                # Calculate magnitudes to decide the dominant region
                upper_region_flow_vertical = flow[0:h - lower_region_h, :, 1]
                lower_region_flow_vertical = flow[h - lower_region_h:h, :, 1]
                upper_magnitude = np.median(np.abs(upper_region_flow_vertical))
                lower_magnitude = np.median(np.abs(lower_region_flow_vertical))

                # Select the dominant part of the 'flow' array for the main calculation
                if lower_magnitude > upper_magnitude * self.motion_inversion_threshold:
                    dominant_flow_region = flow[h - lower_region_h:h, :, :]
                    self.logger.debug("Thrusting pattern dominant. Using lower ROI for flow calculation.")
                elif upper_magnitude > lower_magnitude * self.motion_inversion_threshold:
                    dominant_flow_region = flow[0:h - lower_region_h, :, :]
                    self.logger.debug("Riding pattern dominant. Using upper ROI for flow calculation.")

        # 4. Perform robust calculation on the selected dominant region
        region_h, region_w, _ = dominant_flow_region.shape
        overall_dx, overall_dy = 0.0, 0.0

        if is_vr_video:
            # --- VR-SPECIFIC: Histogram of Flow on the DOMINANT region ---
            block_size = 8
            weighted_flows = []

            # Create a weight map based on the width of the dominant region
            center_x = region_w / 2
            sigma = region_w / 4.0
            x_indices = np.arange(region_w)
            weights_x = np.exp(-((x_indices - center_x) ** 2) / (2 * sigma ** 2))

            for y_start in range(0, region_h, block_size):
                for x_start in range(0, region_w, block_size):
                    block_flow = dominant_flow_region[y_start:y_start + block_size, x_start:x_start + block_size]
                    if block_flow.size < 2:
                        continue

                    dx_vals, dy_vals = block_flow[..., 0].flatten(), block_flow[..., 1].flatten()

                    # Histogram calculation to find the most common motion in the block
                    flow_range = [[-15, 15], [-15, 15]]
                    bins = 30
                    hist, x_edges, y_edges = np.histogram2d(dx_vals, dy_vals, bins=bins, range=flow_range)
                    max_idx = np.unravel_index(np.argmax(hist), hist.shape)

                    mode_dx = (x_edges[max_idx[0]] + x_edges[max_idx[0] + 1]) / 2
                    mode_dy = (y_edges[max_idx[1]] + y_edges[max_idx[1] + 1]) / 2

                    block_weight = np.mean(weights_x[x_start:x_start + block_size])
                    weighted_flows.append({'dx': mode_dx, 'dy': mode_dy, 'weight': block_weight})

            if weighted_flows:
                total_weight = sum(f['weight'] for f in weighted_flows)
                if total_weight > 0:
                    overall_dx = sum(f['dx'] * f['weight'] for f in weighted_flows) / total_weight
                    overall_dy = sum(f['dy'] * f['weight'] for f in weighted_flows) / total_weight
        else:
            # --- 2D VIDEO: Use the simple median on the full ROI (dominant_flow_region is 'flow') ---
            overall_dy = np.median(dominant_flow_region[..., 1])
            overall_dx = np.median(dominant_flow_region[..., 0])

        # 5. Return the calculated values. Magnitudes are returned for context/logging.
        return overall_dy, overall_dx, lower_magnitude, upper_magnitude, flow

    def _calculate_flow_in_sub_regions(self, patch_gray: np.ndarray, prev_patch_gray: Optional[np.ndarray]) \
            -> Tuple[float, float, float, float, Optional[np.ndarray]]:
        """
        Calculates optical flow. For VR, it identifies a dominant motion
        region (upper/lower), then applies a 2D Gaussian weighted Histogram of Flow
        calculation on that sub-region to find the most common motion vector.
        """
        # 1. Handle edge cases
        if self.flow_dense is None or prev_patch_gray is None or prev_patch_gray.shape != patch_gray.shape:
            return 0.0, 0.0, 0.0, 0.0, None

        prev_patch_cont = np.ascontiguousarray(prev_patch_gray)
        patch_cont = np.ascontiguousarray(patch_gray)

        # 2. Calculate optical flow for the entire patch once
        flow = self.flow_dense.calc(prev_patch_cont, patch_cont, None)
        if flow is None:
            return 0.0, 0.0, 0.0, 0.0, None

        h, w, _ = flow.shape
        # is_vr_video = self.app and hasattr(self.app, 'processor') and self.app.processor.determined_video_type == 'VR'
        is_vr_video = self._is_vr_video()

        # 3. For VR, determine the dominant motion region BEFORE main calculation
        dominant_flow_region = flow
        lower_magnitude = 0.0
        upper_magnitude = 0.0

        if is_vr_video and self.inversion_detection_split_ratio > 1.0:
            lower_region_h = int(h / self.inversion_detection_split_ratio)
            if lower_region_h > 0 and lower_region_h < h:
                upper_region_flow_vertical = flow[0:h - lower_region_h, :, 1]
                lower_region_flow_vertical = flow[h - lower_region_h:h, :, 1]
                upper_magnitude = np.median(np.abs(upper_region_flow_vertical))
                lower_magnitude = np.median(np.abs(lower_region_flow_vertical))

                if lower_magnitude > upper_magnitude * self.motion_inversion_threshold:
                    dominant_flow_region = flow[h - lower_region_h:h, :, :]
                    self.logger.debug("Thrusting pattern dominant. Using lower ROI for flow calculation.")
                elif upper_magnitude > lower_magnitude * self.motion_inversion_threshold:
                    dominant_flow_region = flow[0:h - lower_region_h, :, :]
                    self.logger.debug("Riding pattern dominant. Using upper ROI for flow calculation.")

        # 4. Perform robust calculation on the selected dominant region
        region_h, region_w, _ = dominant_flow_region.shape
        overall_dx, overall_dy = 0.0, 0.0

        if is_vr_video:
            # --- VR-SPECIFIC: 2D Weighted Histogram of Flow on the DOMINANT region ---
            block_size = 8
            weighted_flows = []

            center_x, sigma_x = region_w / 2, region_w / 4.0
            weights_x = np.exp(-((np.arange(region_w) - center_x) ** 2) / (2 * sigma_x ** 2))

            center_y, sigma_y = region_h / 2, region_h / 4.0
            weights_y = np.exp(-((np.arange(region_h) - center_y) ** 2) / (2 * sigma_y ** 2))

            for y in range(0, region_h, block_size):
                for x in range(0, region_w, block_size):
                    block_flow = dominant_flow_region[y:y + block_size, x:x + block_size]
                    if block_flow.size < 2:
                        continue

                    # --- Histogram Calculation to find the Mode (most common motion) ---
                    dx_vals, dy_vals = block_flow[..., 0].flatten(), block_flow[..., 1].flatten()

                    flow_range = [[-20, 20], [-20, 20]]  # Range of expected pixel movements per frame
                    bins = 40  # Discretization level (higher is more precise but needs more data)

                    hist, x_edges, y_edges = np.histogram2d(dx_vals, dy_vals, bins=bins, range=flow_range)

                    # Find the bin with the highest count
                    max_idx = np.unravel_index(np.argmax(hist), hist.shape)

                    # Get the center of that bin as the representative motion vector for the block
                    mode_dx = (x_edges[max_idx[0]] + x_edges[max_idx[0] + 1]) / 2
                    mode_dy = (y_edges[max_idx[1]] + y_edges[max_idx[1] + 1]) / 2

                    # Get the combined 2D weight for this block
                    mean_x_weight = np.mean(weights_x[x:x + block_size])
                    mean_y_weight = np.mean(weights_y[y:y + block_size])
                    block_weight = mean_x_weight * mean_y_weight

                    weighted_flows.append({'dx': mode_dx, 'dy': mode_dy, 'weight': block_weight})

            if weighted_flows:
                # The mode-finding is robust, so we can proceed directly to the final weighted average
                total_weight = sum(f['weight'] for f in weighted_flows)
                if total_weight > 0:
                    overall_dx = sum(f['dx'] * f['weight'] for f in weighted_flows) / total_weight
                    overall_dy = sum(f['dy'] * f['weight'] for f in weighted_flows) / total_weight
        else:
            # --- 2D VIDEO: Use simple median on the full ROI for performance ---
            overall_dy = np.median(dominant_flow_region[..., 1])
            overall_dx = np.median(dominant_flow_region[..., 0])

        # 5. Return the calculated values
        return overall_dy, overall_dx, lower_magnitude, upper_magnitude, flow

    def set_tracking_mode(self, mode: str):
        if mode in ["YOLO_ROI", "USER_FIXED_ROI", "OSCILLATION_DETECTOR"]:
            if self.tracking_mode != mode:
                self.tracking_mode = mode
                self.logger.info(f"Tracker mode set to: {self.tracking_mode}")
                # DO NOT STOP TRACKING. This allows for seamless transitions.
                # Instead, clear the state relevant to the new mode.
                if mode == "YOLO_ROI":
                    # Also clear YOLO-specific state for a clean transition
                    self.prev_gray_main_roi, self.prev_features_main_roi = None, None
                    self.roi = None
                elif mode == "USER_FIXED_ROI":
                    # When switching to user ROI, invalidate any existing YOLO ROI
                    self.roi = None
                    self.penis_last_known_box = None
                    self.main_interaction_class = None
                    # Clear flow history to prevent using old data in the new fixed ROI
                    self.primary_flow_history_smooth.clear()
                    self.secondary_flow_history_smooth.clear()
        else:
            self.logger.warning(f"Attempted to set invalid tracking mode: {mode}")

    def reconfigure_for_chapter(self, chapter):  # video_segment.VideoSegment
        """Reconfigures the tracker using ROI data from a chapter."""
        if chapter.user_roi_fixed and chapter.user_roi_initial_point_relative:
            self.set_tracking_mode("USER_FIXED_ROI")
            self.user_roi_fixed = chapter.user_roi_fixed
            self.user_roi_initial_point_relative = chapter.user_roi_initial_point_relative

            # Reset state for the new scene
            self.user_roi_tracked_point_relative = chapter.user_roi_initial_point_relative
            self.primary_flow_history_smooth.clear()
            self.secondary_flow_history_smooth.clear()
            self.prev_gray_user_roi_patch = None
            self.logger.info(f"Tracker reconfigured for chapter {chapter.unique_id[:8]}.")
        # else:
        #     # If the chapter has no ROI, switch to a safe, inactive state
        #     self.set_tracking_mode("YOLO_ROI")  # A mode that does nothing without a target
        #     self.roi = None
        #     if self.tracking_active:
        #         self.stop_tracking()  # Stop generating actions in gaps/unconfigured chapters
        #     self.logger.info(f"Chapter {chapter.unique_id[:8]} has no ROI. Tracker is inactive.")

    def set_user_defined_roi_and_point(self,
                                       roi_abs_coords: Tuple[int, int, int, int],
                                       point_abs_coords_in_frame: Tuple[int, int],
                                       current_frame_for_patch: np.ndarray):
        self.user_roi_fixed = roi_abs_coords
        rx, ry, rw, rh = roi_abs_coords
        point_x_frame, point_y_frame = point_abs_coords_in_frame

        if not (rx <= point_x_frame < rx + rw and ry <= point_y_frame < ry + rh):
            self.logger.warning(f"Selected point ({point_x_frame},{point_y_frame}) is outside defined ROI. Clamping.")
            clamped_point_x_frame = max(rx, min(point_x_frame, rx + rw - 1))
            clamped_point_y_frame = max(ry, min(point_y_frame, ry + rh - 1))
            self.user_roi_initial_point_relative = (
                float(clamped_point_x_frame - rx),
                float(clamped_point_y_frame - ry)
            )
        else:
            self.user_roi_initial_point_relative = (float(point_x_frame - rx), float(point_y_frame - ry))
        self.user_roi_tracked_point_relative = self.user_roi_initial_point_relative
        self.logger.info(f"User defined ROI set to: {self.user_roi_fixed}")
        self.logger.info(f"User initial/tracked point (relative to ROI): {self.user_roi_initial_point_relative}")

        if current_frame_for_patch is not None and current_frame_for_patch.size > 0:
            frame_gray = cv2.cvtColor(current_frame_for_patch, cv2.COLOR_BGR2GRAY)
            urx, ury, urw, urh = self.user_roi_fixed
            urx_c, ury_c = max(0, urx), max(0, ury)
            urw_c = min(urw, frame_gray.shape[1] - urx_c)
            urh_c = min(urh, frame_gray.shape[0] - ury_c)
            if urw_c > 0 and urh_c > 0:
                patch_slice = frame_gray[ury_c: ury_c + urh_c, urx_c: urx_c + urw_c]
                self.prev_gray_user_roi_patch = np.ascontiguousarray(patch_slice)
                self.logger.info(
                    f"Initial gray patch for User ROI captured, shape: {self.prev_gray_user_roi_patch.shape}")
            else:
                self.logger.warning("User defined ROI resulted in zero-size patch. Patch not set.")
                self.prev_gray_user_roi_patch = None
        else:
            self.logger.warning("Frame for patch not provided or empty during User ROI setup.")
            self.prev_gray_user_roi_patch = None
        self.primary_flow_history_smooth.clear()
        self.secondary_flow_history_smooth.clear()
        self.user_roi_current_flow_vector = (0.0, 0.0)

    def clear_user_defined_roi_and_point(self):
        self.user_roi_fixed = None
        self.user_roi_initial_point_relative = None
        self.user_roi_tracked_point_relative = None
        self.prev_gray_user_roi_patch = None
        self.user_roi_current_flow_vector = (0.0, 0.0)
        self.logger.info("User defined ROI and point cleared.")

    def set_oscillation_area_and_point(self,
                                       area_abs_coords: Tuple[int, int, int, int],
                                       point_abs_coords_in_frame: Tuple[int, int],
                                       current_frame_for_patch: np.ndarray):
        """Sets the oscillation detection area and initial tracking point."""
        self.oscillation_area_fixed = area_abs_coords
        ax, ay, aw, ah = area_abs_coords
        point_x_frame, point_y_frame = point_abs_coords_in_frame

        if not (ax <= point_x_frame < ax + aw and ay <= point_y_frame < ay + ah):
            self.logger.warning(f"Selected point ({point_x_frame},{point_y_frame}) is outside defined oscillation area. Clamping.")
            clamped_point_x_frame = max(ax, min(point_x_frame, ax + aw - 1))
            clamped_point_y_frame = max(ay, min(point_y_frame, ay + ah - 1))
            self.oscillation_area_initial_point_relative = (
                float(clamped_point_x_frame - ax),
                float(clamped_point_y_frame - ay)
            )
        else:
            self.oscillation_area_initial_point_relative = (float(point_x_frame - ax), float(point_y_frame - ay))

        self.oscillation_area_tracked_point_relative = self.oscillation_area_initial_point_relative
        self.logger.info(f"Oscillation area set to: {self.oscillation_area_fixed}")
        self.logger.info(f"Oscillation initial/tracked point (relative to area): {self.oscillation_area_initial_point_relative}")

        # Calculate and store the static grid layout
        self._calculate_oscillation_grid_layout()

        if current_frame_for_patch is not None and current_frame_for_patch.size > 0:
            frame_gray = cv2.cvtColor(current_frame_for_patch, cv2.COLOR_BGR2GRAY)
            ax_c, ay_c = max(0, ax), max(0, ay)
            aw_c = min(aw, frame_gray.shape[1] - ax_c)
            ah_c = min(ah, frame_gray.shape[0] - ay_c)
            if aw_c > 0 and ah_c > 0:
                patch_slice = frame_gray[ay_c: ay_c + ah_c, ax_c: ax_c + aw_c]
                self.prev_gray_oscillation_area_patch = np.ascontiguousarray(patch_slice)
                self.logger.info(f"Initial gray patch for oscillation area captured, shape: {self.prev_gray_oscillation_area_patch.shape}")
            else:
                self.logger.warning("Oscillation area resulted in zero-size patch. Patch not set.")
                self.prev_gray_oscillation_area_patch = None
        else:
            self.logger.warning("Frame for patch not provided or empty during oscillation area setup.")
            self.prev_gray_oscillation_area_patch = None

    def _calculate_oscillation_grid_layout(self):
        """Calculates and stores the static grid layout for the oscillation area."""
        if not self.oscillation_area_fixed:
            return

        ax, ay, aw, ah = self.oscillation_area_fixed
        ax_c, ay_c = max(0, ax), max(0, ay)
        aw_c = min(aw, 640 - ax_c)  # Assuming 640x640 frame
        ah_c = min(ah, 640 - ay_c)

        if aw_c <= 0 or ah_c <= 0:
            return

        # Calculate grid layout similar to the analysis
        effective_grid_size = min(self.oscillation_grid_size, min(ah_c, aw_c) // 8)
        adjusted_block_size = min(ah_c // effective_grid_size, aw_c // effective_grid_size)
        if adjusted_block_size < 8:
            adjusted_block_size = 8

        max_blocks_h = ah_c // adjusted_block_size
        max_blocks_w = aw_c // adjusted_block_size

        # Store the grid blocks
        self.oscillation_grid_blocks = []
        for r in range(max_blocks_h):
            for c in range(max_blocks_w):
                x1 = ax_c + c * adjusted_block_size
                y1 = ay_c + r * adjusted_block_size
                x2 = x1 + adjusted_block_size
                y2 = y1 + adjusted_block_size
                self.oscillation_grid_blocks.append((x1, y1, x2 - x1, y2 - y1))

        self.logger.info(f"Calculated static grid layout: {len(self.oscillation_grid_blocks)} blocks")

    def clear_oscillation_area_and_point(self):
        """Clears the oscillation detection area and tracking point."""
        self.oscillation_area_fixed = None
        self.oscillation_area_initial_point_relative = None
        self.oscillation_area_tracked_point_relative = None
        self.prev_gray_oscillation_area_patch = None
        self.oscillation_grid_blocks = []
        self.logger.info("Oscillation area and point cleared.")

    def _get_effective_amplification_factor(self) -> float:
        # main_interaction_class is set by YOLO_ROI mode or by Stage 3 processor
        if self.main_interaction_class and self.main_interaction_class in self.class_specific_amplification_multipliers:
            multiplier = self.class_specific_amplification_multipliers[self.main_interaction_class]
            effective_factor = self.base_amplification_factor * multiplier
            return effective_factor
        return self.base_amplification_factor

    def _update_fps(self):
        current_time_sec = time.time()
        if self.last_frame_time_sec_fps is not None:
            delta_time = current_time_sec - self.last_frame_time_sec_fps
            if delta_time > 0.001:
                self.current_fps = 1.0 / delta_time
        self.last_frame_time_sec_fps = current_time_sec

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if h == 0 or w == 0: return frame.copy()
        target_w, target_h = self.target_size_preprocess
        if (w, h) == (target_w, target_h): return frame.copy()
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        if new_w <= 0 or new_h <= 0: return frame.copy()
        frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if (new_w, new_h) == (target_w, target_h): return frame_resized
        delta_w, delta_h = target_w - new_w, target_h - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        return cv2.copyMakeBorder(frame_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=RGBColors.BLACK)

    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        detections = []
        
        # Check if detection model is available
        if not self.det_model_path or not os.path.exists(self.det_model_path):
            return detections
        
        # Determine discarded classes based on self.app context if available
        discarded_classes_runtime = []
        if self.app and hasattr(self.app, 'discarded_tracking_classes'):
            discarded_classes_runtime = self.app.discarded_tracking_classes

        # Use ModelPool for efficient model management
        try:
            with self.model_pool.get_model(self.det_model_path, 'detect') as model:
                results = model(frame, device=constants.DEVICE, verbose=False, conf=self.confidence_threshold)
                
                for result in results:
                    for box in result.boxes:
                        conf = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = self.classes[class_id] if class_id < len(self.classes) else f"class_{class_id}"
                        
                        if class_name in discarded_classes_runtime:
                            continue
                            
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        detections.append({
                            "box": (x1, y1, x2 - x1, y2 - y1),
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": conf
                        })
        except Exception as e:
            self.logger.error(f"Object detection failed: {e}")
            
        return detections

    def _update_penis_tracking(self, penis_box_xywh: Tuple[int, int, int, int]):
        self.penis_last_known_box = penis_box_xywh
        penis_size = penis_box_xywh[2] * penis_box_xywh[3]
        self.penis_max_size_history.append(penis_size)
        if len(self.penis_max_size_history) > self.penis_size_history_window:
            self.penis_max_size_history.pop(0)

    def _find_interacting_objects(self, penis_box_xywh: Tuple[int, int, int, int], all_detections: List[Dict]) -> List[Dict]:
        if not penis_box_xywh or not all_detections: return []
        px, py, pw, ph = penis_box_xywh
        pcx, pcy = px + pw // 2, py + ph // 2
        interacting = []
        for obj in all_detections:
            if obj["class_name"].lower() in self.INTERACTION_CLASSES:
                ox, oy, ow, oh = obj["box"]
                ocx, ocy = ox + ow // 2, oy + oh // 2
                dist = np.sqrt((ocx - pcx) ** 2 + (ocy - pcy) ** 2)
                max_dist = (np.sqrt(pw ** 2 + ph ** 2) / 2 + np.sqrt(ow ** 2 + oh ** 2) / 2)  * 0.85
                if dist < max_dist: interacting.append(obj)
        return interacting

    def update_main_interaction_class(self, current_best_interaction_class_name: Optional[str]):
        self.class_history.append(current_best_interaction_class_name)
        if len(self.class_history) > self.class_stability_window: self.class_history.pop(0)
        if not self.class_history:
            self.main_interaction_class = None
            return
        counts = {}
        for cls_name in self.class_history:
            if cls_name: counts[cls_name] = counts.get(cls_name, 0) + 1
        if not counts:
            self.main_interaction_class = None
            return
        sorted_cand = sorted(counts.items(), key=lambda it: (self.CLASS_PRIORITY.get(it[0], 99), -it[1]))
        stable_cls = None
        if sorted_cand and sorted_cand[0][1] >= self.class_stability_window // 2 + 1:
            stable_cls = sorted_cand[0][0]
        if self.main_interaction_class != stable_cls:
            self.main_interaction_class = stable_cls
            if stable_cls:
                self.last_interaction_time = time.time()
                self.logger.info(
                    f"Interaction: {stable_cls} (Effective Amp: {self._get_effective_amplification_factor():.2f}x)")
        if self.main_interaction_class and (time.time() - self.last_interaction_time > 3.0):
            self.logger.info(f"Interaction {self.main_interaction_class} timed out. Reverting to base amp.")
            self.main_interaction_class = None
        self.current_effective_amp_factor = self._get_effective_amplification_factor()

    def _calculate_combined_roi(self, frame_shape: Tuple[int, int], penis_box_xywh: Tuple[int, int, int, int], interacting_objects: List[Dict]) -> Tuple[int, int, int, int]:
        entities = [penis_box_xywh] + [obj["box"] for obj in interacting_objects]
        min_x, min_y = min(e[0] for e in entities), min(e[1] for e in entities)
        max_x_coord, max_y_coord = max(e[0] + e[2] for e in entities), max(e[1] + e[3] for e in entities)
        rx1, ry1 = max(0, min_x - self.roi_padding), max(0, min_y - self.roi_padding)
        rx2, ry2 = min(frame_shape[1], max_x_coord + self.roi_padding), min(frame_shape[0], max_y_coord + self.roi_padding)
        rw, rh = rx2 - rx1, ry2 - ry1
        min_w, min_h = 128, 128
        if rw < min_w:
            deficit = min_w - rw
            rx1 = max(0, rx1 - deficit // 2)
            rw = min_w
        if rh < min_h:
            deficit = min_h - rh
            ry1 = max(0, ry1 - deficit // 2)
            rh = min_h
        if rx1 + rw > frame_shape[1]: rx1 = frame_shape[1] - rw
        if ry1 + rh > frame_shape[0]: ry1 = frame_shape[0] - rh
        return int(rx1), int(ry1), int(rw), int(rh)

    def _smooth_roi_transition(self, candidate_roi_xywh: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        old_roi_weight = self.roi_smoothing_factor
        new_roi_weight = 1.0 - old_roi_weight
        if self.roi is None: return candidate_roi_xywh
        x1, y1, w1, h1 = self.roi
        x2, y2, w2, h2 = candidate_roi_xywh
        nx = int(x1 * old_roi_weight + x2 * new_roi_weight)
        ny = int(y1 * old_roi_weight + y2 * new_roi_weight)
        nw = int(w1 * old_roi_weight + w2 * new_roi_weight)
        nh = int(h1 * old_roi_weight + h2 * new_roi_weight)
        return nx, ny, nw, nh

    def update_dis_flow_config(self, preset: Optional[str] = None, finest_scale: Optional[int] = None):
        """
        Updates the DIS optical flow configuration and re-initializes the flow object.
        This method is called from the UI when tracker settings are changed.
        """
        if preset is not None and preset.upper() != self.dis_flow_preset.upper():
            self.dis_flow_preset = preset.upper()
            self.logger.info(f"DIS Optical Flow preset updated to: {self.dis_flow_preset}")

        if finest_scale is not None and finest_scale != self.dis_finest_scale:
            # A value of 0 from the UI means 'auto', which we can represent as None internally
            self.dis_finest_scale = finest_scale if finest_scale > 0 else None
            self.logger.info(f"DIS Optical Flow finest scale updated to: {self.dis_finest_scale}")

        dis_preset_map = {
            "ULTRAFAST": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
            "FAST": cv2.DISOPTICAL_FLOW_PRESET_FAST,
            "MEDIUM": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
        }

        try:
            selected_preset_cv = dis_preset_map.get(self.dis_flow_preset.upper(), cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            # Re-create the dense flow object with the new settings
            self.flow_dense = cv2.DISOpticalFlow_create(selected_preset_cv)
            if self.dis_finest_scale is not None:
                self.flow_dense.setFinestScale(self.dis_finest_scale)
            self.logger.info("Successfully re-initialized DIS Optical Flow object with new configuration.")
        except AttributeError:
            self.logger.warning("cv2.DISOpticalFlow_create not found or preset invalid while updating config. Optical flow may not work.")
            self.flow_dense = None
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while updating DIS flow config: {e}")
            self.flow_dense = None

    def _calculate_flow_in_patch(self, patch_gray: np.ndarray, prev_patch_gray: Optional[np.ndarray], use_sparse: bool = False, prev_features_for_sparse: Optional[np.ndarray] = None) \
            -> Tuple[float, float, Optional[np.ndarray], Optional[np.ndarray]]:
        dx, dy = 0.0, 0.0
        flow_vis = None
        updated_sparse = prev_features_for_sparse
        if prev_patch_gray is None or prev_patch_gray.shape != patch_gray.shape or patch_gray.size == 0:
            if use_sparse and patch_gray.size > 0:
                updated_sparse = cv2.goodFeaturesToTrack(patch_gray, mask=None, **self.feature_params)
            return dx, dy, flow_vis, updated_sparse
        prev_cont = np.ascontiguousarray(prev_patch_gray)
        curr_cont = np.ascontiguousarray(patch_gray)

        if use_sparse and self.feature_params:
            if prev_features_for_sparse is not None and len(prev_features_for_sparse) > 0:
                next_feat, status, _ = cv2.calcOpticalFlowPyrLK(prev_cont, curr_cont, prev_features_for_sparse, None, **self.lk_params)
                good_prev = prev_features_for_sparse[status == 1]
                good_next = next_feat[status == 1] if next_feat is not None else np.array([]) # Ensure good_next is an array

                if len(good_prev) > 0 and len(good_next) > 0 and good_next.ndim == 2 and good_next.shape[1] == 2: # Check shape
                    dx, dy = np.median(good_next[:, 0] - good_prev[:, 0]), np.median(good_next[:, 1] - good_prev[:, 1])
                    updated_sparse = good_next.reshape(-1, 1, 2)
                else:
                    updated_sparse = cv2.goodFeaturesToTrack(curr_cont, mask=None, **self.feature_params) if curr_cont.size > 0 else None
            else:
                updated_sparse = cv2.goodFeaturesToTrack(curr_cont, mask=None, **self.feature_params) if curr_cont.size > 0 else None
        elif self.flow_dense:
            flow = self.flow_dense.calc(prev_cont, curr_cont, None)
            if flow is not None:
                dx, dy, flow_vis = np.median(flow[..., 0]), np.median(flow[..., 1]), flow
        return dx, dy, flow_vis, updated_sparse

    def _apply_adaptive_scaling(self, value: float, min_val_attr: str, max_val_attr: str, size_factor: float, is_primary: bool) -> int:
        min_h, max_h = getattr(self, min_val_attr), getattr(self, max_val_attr)
        setattr(self, min_val_attr, min(min_h * 0.995, value * 0.9 if value < -0.1 else value * 1.1))
        setattr(self, max_val_attr, max(max_h * 0.995, value * 1.1 if value > 0.1 else value * 0.9))
        min_h, max_h = min(getattr(self, min_val_attr), -0.2), max(getattr(self, max_val_attr), 0.2)
        flow_range = max_h - min_h
        if abs(flow_range) < 0.1: flow_range = np.sign(flow_range) * 0.1 if flow_range != 0 else 0.1
        normalized_centered_flow = (2 * (value - min_h) / flow_range) - 1.0 if flow_range != 0 else 0.0
        normalized_centered_flow = np.clip(normalized_centered_flow, -1.0, 1.0)
        effective_amp_factor = self._get_effective_amplification_factor()
        max_deviation = (self.sensitivity / 2.5) * effective_amp_factor
        pos_offset = self.y_offset if is_primary else self.x_offset
        return int(np.clip(50 + normalized_centered_flow * max_deviation + pos_offset, 0, 100))

    def get_current_penis_size_factor(self) -> float:
        if not self.penis_max_size_history or not self.penis_last_known_box: return 1.0
        max_hist = max(self.penis_max_size_history)
        if max_hist < 1: return 1.0
        cur_size = self.penis_last_known_box[2] * self.penis_last_known_box[3]
        return np.clip(cur_size / max_hist, 0.1, 1.5)

    def process_main_roi_content(self, processed_frame_draw_target: np.ndarray,
                                 current_roi_patch_gray: np.ndarray,
                                 prev_roi_patch_gray: Optional[np.ndarray],
                                 prev_sparse_features: Optional[np.ndarray]) \
            -> Tuple[int, int, float, float, Optional[np.ndarray]]:

        updated_sparse_features_out = None
        dy_raw, dx_raw, lower_mag, upper_mag = 0.0, 0.0, 0.0, 0.0
        flow_field_for_vis = None  # Initialize here to ensure it's always defined

        if self.use_sparse_flow:
            dx_raw, dy_raw, _, updated_sparse_features_out = self._calculate_flow_in_patch(
                current_roi_patch_gray, prev_roi_patch_gray, use_sparse=True,
                prev_features_for_sparse=prev_sparse_features)
        else:
            # Use our sub-region analysis method for dense flow
            dy_raw, dx_raw, lower_mag, upper_mag, flow_field_for_vis = self._calculate_flow_in_sub_regions(current_roi_patch_gray, prev_roi_patch_gray)

        # is_vr_video = self.app and hasattr(self.app, 'processor') and self.app.processor.determined_video_type == 'VR'
        is_vr_video = self._is_vr_video()

        if self.enable_inversion_detection and is_vr_video:
            # This logic now ONLY runs for VR videos.
            current_dominant_motion = 'undetermined'
            if lower_mag > upper_mag * self.motion_inversion_threshold:
                current_dominant_motion = 'thrusting'
            elif upper_mag > lower_mag * self.motion_inversion_threshold:
                current_dominant_motion = 'riding'

            self.motion_mode_history.append(current_dominant_motion)
            if len(self.motion_mode_history) > self.motion_mode_history_window:
                self.motion_mode_history.pop(0)

            if self.motion_mode_history:
                most_common_mode, count = Counter(self.motion_mode_history).most_common(1)[0]
                # Solidly switch mode if a new one is dominant in the history window.
                if count > self.motion_mode_history_window // 2 and self.motion_mode != most_common_mode:
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
            manual_scale_multiplier = (self.sensitivity / 10.0) * (1.0 / size_factor) * effective_amp_factor
            base_primary_pos = int(np.clip(50 + dy_smooth * manual_scale_multiplier + self.y_offset, 0, 100))
            secondary_pos = int(np.clip(50 + dx_smooth * manual_scale_multiplier + self.x_offset, 0, 100))

        # Fix inversion logic: thrusting should be normal (base_primary_pos), riding should be inverted
        primary_pos = base_primary_pos if self.motion_mode == "thrusting" else 100 - base_primary_pos

        # Visualization logic (only if self.app is present, for live mode)
        if self.app and self.roi and self.show_flow:
            rx, ry, rw, rh = self.roi
            if ry + rh <= processed_frame_draw_target.shape[0] and rx + rw <= processed_frame_draw_target.shape[1]:
                roi_display_patch = processed_frame_draw_target[ry:ry + rh, rx:rx + rw]
                if roi_display_patch.size > 0:
                    if flow_field_for_vis is not None and self.flow_dense:
                        try:
                            if flow_field_for_vis.shape[-1] == 2:
                                mag, ang = cv2.cartToPolar(flow_field_for_vis[..., 0], flow_field_for_vis[..., 1])
                                hsv = np.zeros_like(roi_display_patch)
                                hsv[..., 1] = 255
                                hsv[..., 0] = ang * 180 / np.pi / 2
                                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                                vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                                processed_frame_draw_target[ry:ry + rh, rx:rx + rw] = cv2.addWeighted(roi_display_patch, 0.5, vis.astype( roi_display_patch.dtype), 0.5, 0)
                        except cv2.error as e:
                            self.logger.error(f"Flow vis error: {e}")
                    if self.use_sparse_flow and updated_sparse_features_out is not None and self.show_tracking_points:
                        for pt in updated_sparse_features_out:
                            x, y = pt.ravel()
                            if 0 <= x < rw and 0 <= y < rh:
                                cv2.circle(roi_display_patch, (int(x), int(y)), 2, RGBColors.CYAN, -1)
                    cx, cy = rw // 2, rh // 2
                    arrow_end_x = np.clip(int(cx + dx_smooth * 5), 0, rw - 1)
                    arrow_end_y = np.clip(int(cy + dy_smooth * 5), 0, rh - 1)
                    cv2.arrowedLine(roi_display_patch, (cx, cy), (arrow_end_x, arrow_end_y), RGBColors.BLUE, 1)

        return primary_pos, secondary_pos, dy_smooth, dx_smooth, updated_sparse_features_out

    def process_frame(self, frame: np.ndarray, frame_time_ms: int, frame_index: Optional[int] = None,
                          min_write_frame_id: Optional[int] = None) -> Tuple[np.ndarray, Optional[List[Dict]]]:
        if self.tracking_mode == "OSCILLATION_DETECTOR":
            return self.process_frame_for_oscillation(frame, frame_time_ms, frame_index)

        self._update_fps()
        processed_frame = self.preprocess_frame(frame)
        current_frame_gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        final_primary_pos, final_secondary_pos = 50, 50
        action_log_list = []
        detected_objects_this_frame: List[Dict] = []  # Initialize for YOLO_ROI mode

        self.current_effective_amp_factor = self._get_effective_amplification_factor()

        if self.tracking_mode == "YOLO_ROI":
            run_detection_this_frame = (
                (self.internal_frame_counter % self.roi_update_interval == 0)
                or (self.roi is None)
                or (not self.penis_last_known_box
                    and self.frames_since_target_lost < self.max_frames_for_roi_persistence
                    and self.internal_frame_counter % max(1, self.roi_update_interval // 3) == 0)
            )

            self.stats_display = [
                f"T-FPS:{self.current_fps:.1f} T(ms):{frame_time_ms} Amp:{self.current_effective_amp_factor:.2f}x"]
            if frame_index is not None: self.stats_display.append(f"FIdx:{frame_index}")
            if self.main_interaction_class: self.stats_display.append(f"Interact: {self.main_interaction_class}")
            # Add this new line for our mode status
            if self.enable_inversion_detection:
                self.stats_display.append(f"Mode: {self.motion_mode}")

            if run_detection_this_frame:
                detected_objects_this_frame = self.detect_objects(processed_frame)
                penis_boxes = [obj["box"] for obj in detected_objects_this_frame if
                               obj["class_name"].lower() == "penis"]
                if penis_boxes:
                    self.frames_since_target_lost = 0
                    self._update_penis_tracking(penis_boxes[0])
                    interacting_objs = self._find_interacting_objects(self.penis_last_known_box, detected_objects_this_frame)
                    current_best_interaction_name = None
                    if interacting_objs:
                        interacting_objs.sort(key=lambda x: self.CLASS_PRIORITY.get(x["class_name"].lower(), 99))
                        current_best_interaction_name = interacting_objs[0]["class_name"].lower()
                    self.update_main_interaction_class(current_best_interaction_name)
                    combined_roi_candidate = self._calculate_combined_roi(processed_frame.shape[:2], self.penis_last_known_box, interacting_objs)

                    # Apply new VR-specific ROI width limits
                    if self._is_vr_video() and self.penis_last_known_box:
                        penis_w = self.penis_last_known_box[2]
                        rx, ry, rw, rh = combined_roi_candidate
                        new_rw = 0

                        # Check main_interaction_class which is more stable than the instantaneous best name
                        self.logger.info(f"Main interaction class: {self.main_interaction_class}")
                        if self.main_interaction_class in ["face", "hand"]:
                            new_rw = penis_w
                        else:
                            new_rw = min(rw, penis_w * 2)

                        if new_rw > 0:
                            original_center_x = rx + rw / 2
                            new_rx = int(original_center_x - new_rw / 2)

                            frame_width = processed_frame.shape[1]
                            final_rw = int(min(new_rw, frame_width))
                            final_rx = max(0, min(new_rx, frame_width - final_rw))

                            combined_roi_candidate = (final_rx, ry, final_rw, rh)

                    self.roi = self._smooth_roi_transition(combined_roi_candidate)
                else:
                    if self.penis_last_known_box: self.logger.info("Primary target (penis) lost in detection cycle.")
                    self.penis_last_known_box = None
                    self.update_main_interaction_class(None)

            if not self.penis_last_known_box and self.roi is not None:
                self.frames_since_target_lost += 1
                if self.frames_since_target_lost > self.max_frames_for_roi_persistence:
                    self.logger.info(f"ROI persistence timeout. Clearing ROI.")
                    self.roi = None
                    self.prev_gray_main_roi = None
                    self.prev_features_main_roi = None
                    self.primary_flow_history_smooth.clear()
                    self.secondary_flow_history_smooth.clear()
                    self.frames_since_target_lost = 0

            self.stats_display = [
                f"T-FPS:{self.current_fps:.1f} T(ms):{frame_time_ms} Amp:{self.current_effective_amp_factor:.2f}x"]
            if frame_index is not None: self.stats_display.append(f"FIdx:{frame_index}")
            if self.main_interaction_class: self.stats_display.append(f"Interact: {self.main_interaction_class}")

            if self.roi and self.tracking_active and self.roi[2] > 0 and self.roi[3] > 0:
                rx, ry, rw, rh = self.roi
                main_roi_patch_gray = current_frame_gray[ry:min(ry + rh, current_frame_gray.shape[0]), rx:min(rx + rw, current_frame_gray.shape[1])]
                if main_roi_patch_gray.size > 0:
                    # process_main_roi_content returns updated_sparse_features, which we store in self.prev_features_main_roi
                    final_primary_pos, final_secondary_pos, _, _, self.prev_features_main_roi = \
                        self.process_main_roi_content(processed_frame, main_roi_patch_gray, self.prev_gray_main_roi, self.prev_features_main_roi)
                    self.prev_gray_main_roi = main_roi_patch_gray.copy()
                else:
                    self.prev_gray_main_roi = None
            else:
                self.prev_gray_main_roi = None

        elif self.tracking_mode == "USER_FIXED_ROI":
            self.current_effective_amp_factor = self._get_effective_amplification_factor()
            self.stats_display = [
                f"UserROI FPS:{self.current_fps:.1f} T(ms):{frame_time_ms} Amp:{self.current_effective_amp_factor:.2f}x"]
            if frame_index is not None: self.stats_display.append(f"FIdx:{frame_index}")

            if self.user_roi_fixed and self.tracking_active:
                urx, ury, urw, urh = self.user_roi_fixed
                urx_c, ury_c = max(0, urx), max(0, ury)
                urw_c, urh_c = min(urw, current_frame_gray.shape[1] - urx_c), min(urh, current_frame_gray.shape[0] - ury_c)

                if urw_c > 0 and urh_c > 0:
                    current_user_roi_patch_gray = current_frame_gray[ury_c: ury_c + urh_c, urx_c: urx_c + urw_c]
                    dy_raw, dx_raw = 0.0, 0.0

                    if self.enable_user_roi_sub_tracking and self.prev_gray_user_roi_patch is not None and self.user_roi_tracked_point_relative and self.flow_dense:
                        if self.prev_gray_user_roi_patch.shape == current_user_roi_patch_gray.shape:
                            flow = self.flow_dense.calc(np.ascontiguousarray(self.prev_gray_user_roi_patch), np.ascontiguousarray(current_user_roi_patch_gray), None)

                            if flow is not None:
                                track_w, track_h = self.user_roi_tracking_box_size
                                box_center_x, box_center_y = self.user_roi_tracked_point_relative
                                box_x1 = int(box_center_x - track_w / 2)
                                box_y1 = int(box_center_y - track_h / 2)
                                box_x2 = box_x1 + track_w
                                box_y2 = box_y1 + track_h
                                patch_h, patch_w = current_user_roi_patch_gray.shape
                                box_x1_c, box_y1_c = max(0, box_x1), max(0, box_y1)
                                box_x2_c, box_y2_c = min(patch_w, box_x2), min(patch_h, box_y2)

                                if box_x2_c > box_x1_c and box_y2_c > box_y1_c:
                                    sub_flow = flow[box_y1_c:box_y2_c, box_x1_c:box_x2_c]
                                    if sub_flow.size > 0:
                                        dx_raw = np.median(sub_flow[..., 0])
                                        dy_raw = np.median(sub_flow[..., 1])

                        self.primary_flow_history_smooth.append(dy_raw)
                        self.secondary_flow_history_smooth.append(dx_raw)

                        if len(self.primary_flow_history_smooth) > self.flow_history_window_smooth:
                            self.primary_flow_history_smooth.pop(0)
                        if len(self.secondary_flow_history_smooth) > self.flow_history_window_smooth:
                            self.secondary_flow_history_smooth.pop(0)

                        dy_smooth = np.median(
                            self.primary_flow_history_smooth) if self.primary_flow_history_smooth else dy_raw
                        dx_smooth = np.median(
                            self.secondary_flow_history_smooth) if self.secondary_flow_history_smooth else dx_raw

                        size_factor = self.get_current_penis_size_factor()
                        if self.adaptive_flow_scale:
                            final_primary_pos = self._apply_adaptive_scaling(dy_smooth, "flow_min_primary_adaptive", "flow_max_primary_adaptive", size_factor, True)
                            final_secondary_pos = self._apply_adaptive_scaling(dx_smooth, "flow_min_secondary_adaptive", "flow_max_secondary_adaptive", size_factor, False)
                        else:
                            effective_amp_factor = self._get_effective_amplification_factor()
                            manual_scale_multiplier = (self.sensitivity / 10.0) * effective_amp_factor
                            final_primary_pos = int(
                                np.clip(50 + dy_smooth * manual_scale_multiplier + self.y_offset, 0, 100))
                            final_secondary_pos = int(
                                np.clip(50 + dx_smooth * manual_scale_multiplier + self.x_offset, 0, 100))

                        self.user_roi_current_flow_vector = (dx_smooth, dy_smooth)

                        if self.user_roi_tracked_point_relative:
                            prev_x_rel, prev_y_rel = self.user_roi_tracked_point_relative
                            new_x_rel = prev_x_rel + dx_smooth
                            new_y_rel = prev_y_rel + dy_smooth
                            self.user_roi_tracked_point_relative = (max(0.0, min(new_x_rel, float(urw_c))), max(0.0, min(new_y_rel, float(urh_c))))
                    else:
                        # Fallback to calculating flow on the whole User ROI without calling YOLO-specific methods.
                        dx_raw, dy_raw, _, _ = self._calculate_flow_in_patch(
                            current_user_roi_patch_gray,
                            self.prev_gray_user_roi_patch,
                            use_sparse=self.use_sparse_flow,
                            prev_features_for_sparse=None
                        )

                        # Smooth the calculated raw dx/dy values
                        self.primary_flow_history_smooth.append(dy_raw)
                        self.secondary_flow_history_smooth.append(dx_raw)

                        if len(self.primary_flow_history_smooth) > self.flow_history_window_smooth:
                            self.primary_flow_history_smooth.pop(0)
                        if len(self.secondary_flow_history_smooth) > self.flow_history_window_smooth:
                            self.secondary_flow_history_smooth.pop(0)

                        dy_smooth = np.median(self.primary_flow_history_smooth) if self.primary_flow_history_smooth else dy_raw
                        dx_smooth = np.median(self.secondary_flow_history_smooth) if self.secondary_flow_history_smooth else dx_raw

                        # Apply scaling and generate final position
                        size_factor = 1.0  # No object detection in this mode
                        if self.adaptive_flow_scale:
                            final_primary_pos = self._apply_adaptive_scaling(dy_smooth, "flow_min_primary_adaptive", "flow_max_primary_adaptive", size_factor, True)
                            final_secondary_pos = self._apply_adaptive_scaling(dx_smooth, "flow_min_secondary_adaptive", "flow_max_secondary_adaptive", size_factor, False)
                        else:
                            effective_amp_factor = self._get_effective_amplification_factor()
                            manual_scale_multiplier = (self.sensitivity / 10.0) * effective_amp_factor
                            final_primary_pos = int(np.clip(50 + dy_smooth * manual_scale_multiplier + self.y_offset, 0, 100))
                            final_secondary_pos = int(np.clip(50 + dx_smooth * manual_scale_multiplier + self.x_offset, 0, 100))

                        # Update state
                        self.user_roi_current_flow_vector = (dx_smooth, dy_smooth)
                        if self.user_roi_tracked_point_relative:
                            prev_x_rel, prev_y_rel = self.user_roi_tracked_point_relative
                            new_x_rel = prev_x_rel + dx_smooth
                            new_y_rel = prev_y_rel + dy_smooth
                            self.user_roi_tracked_point_relative = (max(0.0, min(new_x_rel, float(urw_c))), max(0.0, min(new_y_rel, float(urh_c))))

                    self.prev_gray_user_roi_patch = np.ascontiguousarray(current_user_roi_patch_gray)

                else:  # User ROI has no size
                    self.prev_gray_user_roi_patch = None
                    final_primary_pos, final_secondary_pos = 50, 50
                    self.user_roi_current_flow_vector = (0.0, 0.0)
            else:  # No User ROI is set or tracking is inactive
                self.prev_gray_user_roi_patch = None
                final_primary_pos, final_secondary_pos = 50, 50
                self.user_roi_current_flow_vector = (0.0, 0.0)

        if self.app and self.tracking_active and \
                (min_write_frame_id is None or (frame_index is not None and frame_index >= min_write_frame_id)):

            # --- Automatic Lag Compensation ---
            # Calculate the inherent delay from the smoothing window. A window of size N has a lag of (N-1)/2 frames.
            # A window size of 1 means no smoothing and no delay.
            automatic_smoothing_delay_frames = (self.flow_history_window_smooth - 1) / 2.0 if self.flow_history_window_smooth > 1 else 0.0

            # Combine the automatic compensation with the user's manual delay setting.
            total_delay_frames = self.output_delay_frames + automatic_smoothing_delay_frames

            # Convert the total frame delay to milliseconds.
            delay_ms = (total_delay_frames / self.current_video_fps_for_delay) * 1000.0 if self.current_video_fps_for_delay > 0 else 0.0

            # Adjust the timestamp to compensate for the total delay.
            adjusted_frame_time_ms = frame_time_ms - delay_ms
            current_tracking_axis_mode = self.app.tracking_axis_mode

            current_single_axis_output = self.app.single_axis_output_target
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

            is_file_processing_context = frame_index is not None

            self.funscript.add_action(
                timestamp_ms=int(round(adjusted_frame_time_ms)),
                primary_pos=primary_to_write,
                secondary_pos=secondary_to_write,
                is_from_live_tracker=(not is_file_processing_context)
            )

            self.funscript.add_action(int(round(adjusted_frame_time_ms)), primary_to_write, secondary_to_write)
            action_log_list.append({
                "at": int(round(adjusted_frame_time_ms)), "pos": primary_to_write, "secondary_pos": secondary_to_write,
                "raw_ud_pos_computed": final_primary_pos, "raw_lr_pos_computed": final_secondary_pos,
                "mode": current_tracking_axis_mode,
                "target": current_single_axis_output if current_tracking_axis_mode != "both" else "N/A",
                "raw_at": frame_time_ms, "delay_applied_ms": delay_ms,
                "roi_main": self.roi if self.tracking_mode == "YOLO_ROI" else self.user_roi_fixed,
                "amp": self.current_effective_amp_factor
            })

        if self.show_masks and detected_objects_this_frame and self.tracking_mode == "YOLO_ROI":
            self.draw_detections(processed_frame, detected_objects_this_frame)

        if self.tracking_mode == "YOLO_ROI" and self.show_roi and self.roi:
            rx, ry, rw, rh = self.roi
            color = self.get_class_color(
                self.main_interaction_class or ("penis" if self.penis_last_known_box else "persisting"))
            cv2.rectangle(processed_frame, (rx, ry), (rx + rw, ry + rh), color, 1)
            status_text = self.main_interaction_class or ('P' if self.penis_last_known_box else 'Lost...')
            cv2.putText(processed_frame, f"ROI:{status_text}", (rx, ry - 2), cv2.FONT_HERSHEY_PLAIN, 0.7, color, 1)
            if not self.penis_last_known_box:
                cv2.putText(processed_frame,
                            f"Lost: {self.frames_since_target_lost}/{self.max_frames_for_roi_persistence}",
                            (rx, ry + rh + 10), cv2.FONT_HERSHEY_PLAIN, 0.6, RGBColors.BLUE, 1)

        elif self.tracking_mode == "USER_FIXED_ROI" and self.show_roi and self.user_roi_fixed:
            urx, ury, urw, urh = self.user_roi_fixed
            urx_c, ury_c = max(0, urx), max(0, ury)
            urw_c, urh_c = min(urw, processed_frame.shape[1] - urx_c), min(urh, processed_frame.shape[0] - ury_c)
            cv2.rectangle(processed_frame, (urx_c, ury_c), (urx_c + urw_c, ury_c + urh_c), RGBColors.CYAN, 2)

            if self.user_roi_tracked_point_relative:
                point_x_abs = urx_c + int(self.user_roi_tracked_point_relative[0])
                point_y_abs = ury_c + int(self.user_roi_tracked_point_relative[1])
                cv2.circle(processed_frame, (point_x_abs, point_y_abs), 3, RGBColors.GREEN, -1)

        if self.show_stats:
            for i, stat_text in enumerate(self.stats_display):
                cv2.putText(processed_frame, stat_text, (5, 15 + i * 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, RGBColors.TEAL, 1)

        self.internal_frame_counter += 1
        return processed_frame, action_log_list if action_log_list else None

    def get_class_color(self, class_name: Optional[str]) -> Tuple[int, int, int]:
        return RGBColors.CLASS_COLORS.get(class_name.lower() if class_name else "", RGBColors.GREY_LIGHT)

    def draw_detections(self, frame: np.ndarray, detected_objects: List[Dict]):
        for obj in detected_objects:
            x, y, w, h = obj["box"]
            cn = obj["class_name"]
            cf = obj["confidence"]
            clr = self.get_class_color(cn)
            cv2.rectangle(frame, (x, y), (x + w, y + h), clr, 1)
            cv2.putText(frame, f"{cn} {cf:.1f}", (x, y - 2), cv2.FONT_HERSHEY_PLAIN, 0.7, clr, 1)

    def get_current_value(self, axis: str = 'primary') -> int:
        return self.funscript.get_latest_value(axis)

    def start_tracking(self):
        self.tracking_active = True
        self.start_time_tracking = time.time() * 1000

        # Check and report video source status when tracking starts
        self._check_and_report_video_source_status()
        if self.tracking_mode == "OSCILLATION_DETECTOR":
            # Update grid size and sensitivity from settings before starting
            self.update_oscillation_grid_size()
            self.update_oscillation_sensitivity()

            fps = self.app.processor.fps if self.app and self.app.processor and self.app.processor.fps > 0 else 30.0

            # --- Dynamically resize history buffer for live amplification when tracking starts ---
            amp_window_ms = self.app.app_settings.get("live_oscillation_amp_window_ms", 4000)
            new_maxlen = int(fps * (amp_window_ms / 1000.0))
            if not hasattr(self, 'oscillation_position_history') or self.oscillation_position_history.maxlen != new_maxlen:
                self.oscillation_position_history = deque(maxlen=new_maxlen)
                self.logger.info(f"Live amplification history buffer resized to {new_maxlen} frames.")

            self.oscillation_history_max_len = int(self.oscillation_history_seconds * fps)
            self.oscillation_history.clear()
            self.prev_gray_oscillation = None
            self.oscillation_funscript_pos = 50
            self.logger.info(f"Oscillation detector started. History size set to {self.oscillation_history_max_len} frames.")

        self.internal_frame_counter = 0 # Reset for both live and S3 context if S3 reuses this
        self.flow_min_primary_adaptive, self.flow_max_primary_adaptive = -1.0, 1.0
        self.flow_min_secondary_adaptive, self.flow_max_secondary_adaptive = -1.0, 1.0
        for lst in [self.primary_flow_history_smooth, self.secondary_flow_history_smooth, self.class_history]:
            lst.clear()
        # self.current_effective_amp_factor = self._get_effective_amplification_factor() # Done per frame/segment start

        # Dynamically set the motion history window to match the video's FPS (~1 second buffer)
        if self.app and hasattr(self.app, 'processor') and self.app.processor.fps > 0:
            new_window_size = int(round(self.app.processor.fps))
            self.motion_mode_history_window = new_window_size
            self.logger.info(f"Motion history window set to video FPS: {new_window_size} frames.")
        else:
            # Fallback to the default if FPS is not available
            self.motion_mode_history_window = 30
            self.logger.info(f"Falling back to default motion history window: {self.motion_mode_history_window} frames.")

        if self.tracking_mode == "YOLO_ROI": # Also applies to S3 like processing
            self.frames_since_target_lost = 0
            self.penis_max_size_history.clear()
            self.prev_gray_main_roi, self.prev_features_main_roi = None, None
            self.penis_last_known_box, self.main_interaction_class = None, None # main_interaction_class set by live or S3 segment
            self.last_interaction_time = 0
            self.roi = None
            self.logger.info(f"Tracking state re-initialized (mode: {self.tracking_mode}).")
        elif self.tracking_mode == "USER_FIXED_ROI":
            if self.prev_gray_user_roi_patch is None and self.user_roi_fixed:
                self.logger.warning("User ROI tracking started, but initial patch not set. Flow may be delayed.")
            if self.user_roi_initial_point_relative:
                self.user_roi_tracked_point_relative = self.user_roi_initial_point_relative
            self.user_roi_current_flow_vector = (0.0, 0.0)
            self.logger.info("User Defined ROI Tracking started.")

    def stop_tracking(self):
        self.tracking_active = False
        if self.tracking_mode == "YOLO_ROI":  # Also for S3-like
            self.prev_gray_main_roi, self.prev_features_main_roi = None, None

        # Reset motion mode to undetermined when stopping.
        self.motion_mode = 'undetermined'
        self.motion_mode_history.clear()

        self.logger.info(f"Tracking stopped (mode: {self.tracking_mode}). Motion state reset to 'undetermined'.")

    def update_oscillation_grid_size(self):
        """Updates the oscillation grid size from app settings."""
        if self.app:
            new_grid_size = self.app.app_settings.get("oscillation_detector_grid_size", 20)
            if new_grid_size != self.oscillation_grid_size:
                self.oscillation_grid_size = new_grid_size
                self.oscillation_block_size = constants.YOLO_INPUT_SIZE // self.oscillation_grid_size
                self.logger.info(f"Oscillation grid size updated to {self.oscillation_grid_size} (block size: {self.oscillation_block_size}x{self.oscillation_block_size})")

    def update_oscillation_sensitivity(self):
        """Updates the oscillation sensitivity from app settings."""
        if self.app:
            new_sensitivity = self.app.app_settings.get("oscillation_detector_sensitivity", 1.0)
            if new_sensitivity != self.oscillation_sensitivity:
                self.oscillation_sensitivity = new_sensitivity
                self.logger.info(f"Oscillation sensitivity updated to {self.oscillation_sensitivity}")

    def reset(self, reason: Optional[str] = None):
        self.stop_tracking()  # Explicitly set tracking_active to False.
        self.clear_user_defined_roi_and_point()
        self.set_tracking_mode("YOLO_ROI")  # Default mode on full reset

        # Clear all relevant state variables to ensure a clean slate
        self.internal_frame_counter = 0
        self.frames_since_target_lost = 0
        self.roi = None
        self.prev_gray_main_roi = None
        self.prev_features_main_roi = None
        self.penis_last_known_box = None
        self.main_interaction_class = None
        self.penis_max_size_history.clear()
        self.primary_flow_history_smooth.clear()
        self.secondary_flow_history_smooth.clear()
        self.class_history.clear()
        self.last_interaction_time = 0
        self.motion_mode_history.clear()
        self.motion_mode = 'undetermined'

        self.oscillation_history.clear()
        self.prev_gray_oscillation = None
        self.oscillation_funscript_pos = 50

        if self.funscript:  # This funscript is for live tracking
            if reason not in ["seek", "project_load_preserve_actions", "stop_preserve_funscript"]:
                self.funscript.clear()
                self.logger.info(f"Live tracker Funscript cleared (reason: {reason}).")
            else:
                self.logger.info(f"Live tracker Funscript preserved (reason: {reason}).")

        self.logger.info(f"Tracker reset complete (reason: {reason}). Tracking is now inactive.")

        # Clear video source status on reset
        self._last_video_source_status = None

    def _check_and_report_video_source_status(self) -> None:
        """
        Checks the current video source status and reports changes to the user.
        This helps users understand when preprocessed videos are being used.
        """
        if not self.app or not hasattr(self.app, 'processor'):
            return

        try:
            current_status = self.app.processor.get_preprocessed_video_status()

            # Only report if status has changed
            if current_status != self._last_video_source_status:
                if current_status.get("using_preprocessed", False):
                    if current_status.get("valid", False):
                        self.logger.info(f"Live tracking using preprocessed video: {os.path.basename(current_status.get('path', 'unknown'))} "
                                       f"({current_status.get('frame_count', 0)} frames)", extra={'status_message': True})
                    else:
                        self.logger.warning("Live tracking: preprocessed video detected but invalid, using original video",
                                          extra={'status_message': True})
                else:
                    if current_status.get("exists", False):
                        self.logger.info("Live tracking using original video (preprocessed video available but not used)",
                                       extra={'status_message': True})
                    else:
                        self.logger.info("Live tracking using original video (no preprocessed video available)",
                                       extra={'status_message': True})

                self._last_video_source_status = current_status.copy()

        except Exception as e:
            self.logger.error(f"Error checking video source status: {e}")

    def get_video_source_info(self) -> Dict[str, Any]:
        """
        Returns information about the current video source being used for tracking.

        Returns:
            Dictionary with video source information
        """
        info = {
            "using_preprocessed": False,
            "preprocessed_available": False,
            "preprocessed_valid": False,
            "source_description": "Unknown"
        }

        if self.app and hasattr(self.app, 'processor'):
            try:
                status = self.app.processor.get_preprocessed_video_status()
                info["using_preprocessed"] = status.get("using_preprocessed", False)
                info["preprocessed_available"] = status.get("exists", False)
                info["preprocessed_valid"] = status.get("valid", False)

                if info["using_preprocessed"]:
                    info["source_description"] = f"Preprocessed video ({status.get('frame_count', 0)} frames)"
                elif info["preprocessed_available"]:
                    if info["preprocessed_valid"]:
                        info["source_description"] = "Original video (preprocessed available but not used)"
                    else:
                        info["source_description"] = "Original video (preprocessed invalid)"
                else:
                    info["source_description"] = "Original video (no preprocessed available)"

            except Exception as e:
                self.logger.error(f"Error getting video source info: {e}")
                info["source_description"] = "Error retrieving source info"

        return info

    def process_frame_for_oscillation(self, frame: np.ndarray, frame_time_ms: int, frame_index: Optional[int] = None) -> Tuple[np.ndarray, Optional[List[Dict]]]:
        """
        [V9 - Advanced Filtering] Implements global motion cancellation, advanced oscillation scoring,
        and a VR-specific focus on the central third of the frame.
        """
        processed_frame = self.preprocess_frame(frame)
        current_gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        action_log_list = []

        if self.prev_gray_oscillation is None or self.prev_gray_oscillation.shape != current_gray.shape:
            self.prev_gray_oscillation = current_gray
            return processed_frame, None

        if not self.flow_dense:
            self.logger.warning("Dense optical flow not available for oscillation detection.")
            return processed_frame, None

        # --- Step 1: Calculate Global Optical Flow & Global Motion Vector ---
        flow = self.flow_dense.calc(self.prev_gray_oscillation, current_gray, None)
        if flow is None:
            self.prev_gray_oscillation = current_gray
            return processed_frame, None

        # Calculate Global Motion to cancel out camera pans/shakes
        global_dx = np.median(flow[..., 0])
        global_dy = np.median(flow[..., 1])

        # --- Step 2: Identify Active Cells & Apply VR Focus ---
        min_motion_threshold = 15
        min_cell_activation_pixels = (self.oscillation_block_size**2) * 0.05

        frame_diff = cv2.absdiff(current_gray, self.prev_gray_oscillation)
        _, motion_mask = cv2.threshold(frame_diff, min_motion_threshold, 255, cv2.THRESH_BINARY)

        # Check if the video is VR to apply the focus rule
        is_vr = self._is_vr_video()
        vr_central_third_start = self.oscillation_grid_size // 3
        vr_central_third_end = 2 * self.oscillation_grid_size // 3

        newly_active_cells = set()
        for r in range(self.oscillation_grid_size):
            for c in range(self.oscillation_grid_size):
                # If VR, skip cells outside the central third
                if is_vr and (c < vr_central_third_start or c > vr_central_third_end):
                    continue

                y_start, x_start = r * self.oscillation_block_size, c * self.oscillation_block_size
                mask_roi = motion_mask[y_start:y_start + self.oscillation_block_size, x_start:x_start + self.oscillation_block_size]
                if cv2.countNonZero(mask_roi) > min_cell_activation_pixels:
                    newly_active_cells.add((r, c))

        # Update persistence counters
        # Add newly active cells or reset their timers
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
        max_magnitude = 0.0
        for r, c in persistent_active_cells:
            y_start = r * self.oscillation_block_size
            x_start = c * self.oscillation_block_size

            # Sample the pre-computed flow field for this cell's ROI
            flow_patch = flow[y_start:y_start + self.oscillation_block_size, x_start:x_start + self.oscillation_block_size]

            if flow_patch.size > 0:
                # Subtract global motion to get true local motion
                local_dx = np.median(flow_patch[..., 0]) - global_dx
                local_dy = np.median(flow_patch[..., 1]) - global_dy

                mag = np.sqrt(local_dx**2 + local_dy**2)
                block_motions.append({'dx': local_dx, 'dy': local_dy, 'mag': mag, 'pos': (r, c)})
                if (r, c) not in self.oscillation_history:
                    self.oscillation_history[(r, c)] = deque(maxlen=self.oscillation_history_max_len)
                self.oscillation_history[(r, c)].append({'dx': local_dx, 'dy': local_dy, 'mag': mag})

        # --- Step 4: Advanced Oscillation Scoring ---
        active_blocks = []
        if block_motions:
            candidate_blocks = []
            for motion in block_motions:
                history = self.oscillation_history.get(motion['pos'])
                # Only consider blocks with some history and current motion
                if history and len(history) > 10 and motion['mag'] > 0.2:

                    # 1. Get stats from history
                    recent_dy = [h['dy'] for h in history]
                    mean_mag = np.mean([h['mag'] for h in history])

                    # 2. Calculate a "frequency score" (higher is better)
                    # This proxy for frequency counts direction changes
                    zero_crossings = np.sum(np.diff(np.sign(recent_dy)) != 0)
                    frequency_score = (zero_crossings / len(recent_dy)) * 10.0

                    # 3. Calculate "variance score" (higher is better)
                    # A high standard deviation means the motion isn't linear
                    variance_score = np.std(recent_dy)

                    # 4. Combine into a final oscillation score
                    # This rewards blocks that are strong, frequent, and non-linear
                    oscillation_score = mean_mag * (1 + frequency_score) * (1 + variance_score)

                    if oscillation_score > 0.5: # Filter out low-scoring blocks
                        candidate_blocks.append({'pos': motion['pos'], 'score': oscillation_score, 'dy': motion['dy'], 'dx': motion['dx']})

            if candidate_blocks:
                max_score = max(b['score'] for b in candidate_blocks)
                # Take any block that has at least 40% of the max score.
                active_blocks = [b for b in candidate_blocks if b['score'] >= max_score * 0.4]

        # Step 5: Signal Generation
        if active_blocks:
            total_weight = sum(b['score'] for b in active_blocks)
            if total_weight > 0:
                # Calculate the raw, unsmoothed position for this frame
                final_dy = sum(b['dy'] * b['score'] for b in active_blocks) / total_weight
                final_dx = sum(b['dx'] * b['score'] for b in active_blocks) / total_weight

                max_deviation = 49 * self.oscillation_sensitivity
                new_raw_primary_pos = 50 + np.clip(final_dy * -10, -max_deviation, max_deviation)
                new_raw_secondary_pos = 50 + np.clip(final_dx * 10, -max_deviation, max_deviation)

                # --- Apply EMA Smoothing Filter ---
                alpha = self.oscillation_ema_alpha
                self.oscillation_last_known_pos = self.oscillation_last_known_pos * (1 - alpha) + new_raw_primary_pos * alpha
                self.oscillation_last_known_secondary_pos = self.oscillation_last_known_secondary_pos * (1 - alpha) + new_raw_secondary_pos * alpha

                self.oscillation_last_active_time = frame_time_ms
        else:
            # This part handles the decay when no motion is detected and has its own smoothing
            time_since_last_active = frame_time_ms - self.oscillation_last_active_time
            if time_since_last_active > self.oscillation_hold_duration_ms:
                # Decay towards center after hold duration expires.
                decay_factor = 0.95
                self.oscillation_last_known_pos = self.oscillation_last_known_pos * decay_factor + 50 * (1 - decay_factor)
                self.oscillation_last_known_secondary_pos = self.oscillation_last_known_secondary_pos * decay_factor + 50 * (1 - decay_factor)

        # The final funscript position is now based on the smoothed value
        self.oscillation_funscript_pos = int(round(self.oscillation_last_known_pos))
        self.oscillation_funscript_secondary_pos = int(round(self.oscillation_last_known_secondary_pos))

        # Step 6: Action Logging
        if self.tracking_active:
            # This block now correctly saves the generated signal
            current_tracking_axis_mode = self.app.tracking_axis_mode
            current_single_axis_output = self.app.single_axis_output_target
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

            self.funscript.add_action(timestamp_ms=frame_time_ms, primary_pos=primary_to_write,
                                      secondary_pos=secondary_to_write)
            action_log_list.append({"at": frame_time_ms, "pos": primary_to_write, "secondary_pos": secondary_to_write})

        # Step 7: Visualization
        active_block_positions = {b['pos'] for b in active_blocks}
        for r,c in self.oscillation_cell_persistence.keys():
            x1, y1 = c * self.oscillation_block_size, r * self.oscillation_block_size
            color = (0, 255, 0) if (r, c) in active_block_positions else (180, 100, 100)
            cv2.rectangle(processed_frame, (x1, y1), (x1 + self.oscillation_block_size, y1 + self.oscillation_block_size), color, 1)

        self.prev_gray_oscillation = current_gray
        return processed_frame, action_log_list if action_log_list else None
    
    def cleanup(self):
        """Explicit cleanup method for resource management."""
        try:
            # Clean up ModelPool if it exists
            if hasattr(self, 'model_pool') and self.model_pool is not None:
                self.model_pool.cleanup()
                self.logger.debug("ROITracker: ModelPool cleaned up")
            
            # Clear OpenCV objects that might hold memory
            self.prev_gray_main_roi = None
            self.prev_gray_user_roi_patch = None
            self.prev_gray_oscillation_area_patch = None
            self.prev_gray_oscillation = None
            
            # Clear optical flow objects
            self.flow_dense = None
            self.flow_sparse_features = None
            
            # Clear large data structures
            self.oscillation_cell_persistence.clear()
            self.oscillation_active_block_positions.clear()
            
            self.logger.debug("ROITracker: Resources cleaned up")
            
        except Exception as e:
            self.logger.warning(f"ROITracker cleanup error: {e}")
    
    def __del__(self):
        """Destructor to ensure resource cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass  # Avoid errors during destruction
