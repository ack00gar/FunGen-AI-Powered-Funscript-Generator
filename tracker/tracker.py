import cv2
import logging
import os
import numpy as np
import time
import math
from collections import Counter, deque
from typing import List, Dict, Tuple, Optional, Any
from ultralytics import YOLO
from funscript import DualAxisFunscript
from config import constants
from config.constants_colors import RGBColors


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
        self.app = app_logic_instance  # Can be None if instantiated by Stage 3
        self.video_type_override = video_type_override

        if logger:
            self.logger = logger
        elif self.app and hasattr(self.app, 'logger'):
            self.logger = self.app.logger
        else:
            self.logger = logging.getLogger('ROITracker_fallback')
            if not self.logger.handlers:
                self.logger.addHandler(logging.NullHandler())
            self.logger.warning("No external logger provided to ROITracker, using fallback NullHandler.", extra={'status_message': False})

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
        self.oscillation_ema_alpha: float = 0.5

        # --- Static grid storage for oscillation area ---
        self.oscillation_grid_blocks: List[Tuple[int, int, int, int]] = []  # List of (x, y, w, h) for each grid block
        # --- Always-available set of active grid blocks ---
        self.oscillation_active_block_positions: set = set()

        # --- Per-mode ROI caches to preserve user selections across mode switches ---
        # Cache for USER_FIXED_ROI
        self._cache_user_roi: Dict[str, Optional[Tuple]] = {
            'roi': None,
            'initial_rel': None,
            'tracked_rel': None,
        }
        # Cache for OSCILLATION_DETECTOR
        self._cache_oscillation: Dict[str, Optional[Tuple]] = {
            'area': None,
            'initial_rel': None,
            'tracked_rel': None,
        }

        self.enable_user_roi_sub_tracking: bool = True
        self.user_roi_tracking_box_size: Tuple[int, int] = (5, 5)

        # Store paths
        self.det_model_path = tracker_model_path
        self.pose_model_path = pose_model_path

        # Direct YOLO usage with simple caching
        self._cached_detection_model: Optional[YOLO] = None
        self._cached_model_path: Optional[str] = None

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
        self.logger.debug(f"Base Amplification: {self.base_amplification_factor}x")
        self.logger.debug(f"Class Specific Amp Multipliers: {self.class_specific_amplification_multipliers}")

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
            selected_preset_cv = dis_preset_map.get(self.dis_flow_preset.upper(), cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)

            # General-purpose dense flow (used in non-oscillation paths)
            self.flow_dense = cv2.DISOpticalFlow_create(selected_preset_cv)
            if self.dis_finest_scale is not None:
                self.flow_dense.setFinestScale(self.dis_finest_scale)

            # Dedicated dense flow object for oscillation detector to avoid cross-mode side effects
            self.flow_dense_osc = cv2.DISOpticalFlow_create(selected_preset_cv)
            if self.dis_finest_scale is not None:
                self.flow_dense_osc.setFinestScale(self.dis_finest_scale)
        except AttributeError:
            self.logger.debug("cv2.DISOpticalFlow_create not found or preset invalid. Optical flow might not work.")
            self.flow_dense = None
            self.flow_dense_osc = None

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
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
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
        # Whether to draw oscillation grid/cell overlays
        self.show_masks: bool = bool(self.app.app_settings.get("oscillation_show_overlay", False)) if self.app else False
        # Whether to draw static grid blocks for oscillation area (UI-controlled)
        self.show_grid_blocks: bool = bool(self.app.app_settings.get("oscillation_show_grid_blocks", False)) if self.app else False
        self.show_stats: bool = False

        # --- Preallocated buffers for memory reuse ---
        self._preprocess_buffer: Optional[np.ndarray] = None  # BGR target size buffer
        self._resize_tmp: Optional[np.ndarray] = None          # temp buffer for resized content
        self._resize_tmp_shape: Optional[Tuple[int, int]] = None
        self._gray_full_buffer: Optional[np.ndarray] = None    # Gray buffer for full frame (target size)
        self._gray_roi_buffer: Optional[np.ndarray] = None     # Gray buffer for ROI-sized crops
        self._prev_gray_osc_buffer: Optional[np.ndarray] = None

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
        self.oscillation_position_history = deque(maxlen=120)  # Default to 4 seconds @ 30fps
        self.oscillation_last_known_pos: float = 50.0
        self.oscillation_last_known_secondary_pos = 50.0
        self.oscillation_last_active_time = 0
        self.oscillation_hold_duration_ms = 200  # Hold for 200ms before decaying
        self.oscillation_ema_alpha: float = 0.3  # Smoothing factor for the final signal (c9e6fbd original)
        self.oscillation_history_max_len: int = 60
        self.oscillation_funscript_pos: int = 50  # Initialize legacy funscript positions
        self.oscillation_funscript_secondary_pos: int = 50

        # --- Oscillation sensitivity control ---
        self.oscillation_sensitivity: float = self.app.app_settings.get("oscillation_detector_sensitivity", 1.0) if self.app else 1.0

        self.last_frame_time_sec_fps: Optional[float] = None
        self.current_fps: float = 0.0
        self.current_effective_amp_factor: float = self.base_amplification_factor
        self.stats_display: List[str] = []
        self.logger.info(f"Tracker fully initialized (ROI Persistence: {self.max_frames_for_roi_persistence} frames, ROI Smoothing: {self.roi_smoothing_factor}). App instance {'provided' if self.app else 'not provided (e.g. S3 mode)'}.")

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
                # Ensure cached model is available to read names
                if self._cached_detection_model is None or self._cached_model_path != self.det_model_path:
                    self._cached_detection_model = YOLO(self.det_model_path, task='detect')
                    self._cached_model_path = self.det_model_path
                names_attr = getattr(self._cached_detection_model, 'names', None)
                if names_attr:
                    if isinstance(names_attr, dict):
                        try:
                            self.classes = [names_attr[k] for k in sorted(names_attr.keys(), key=lambda x: int(x))]
                        except Exception:
                            self.classes = list(names_attr.values())
                    elif isinstance(names_attr, (list, tuple)):
                        self.classes = list(names_attr)
                self.logger.info(f"Loaded class names from detection model: {self.det_model_path}")
            except Exception as e:
                self.logger.error(f"Could not load class names from {self.det_model_path}: {e}")
                self.classes = []
        else:
            self.logger.warning("Detection model path not set or file does not exist.")
            self.classes = []

    def _load_models(self):
        """Legacy method - now just loads class names for compatibility."""
        self.logger.warning("_load_models() is deprecated. Models are now loaded on-demand via direct YOLO caching.")
        self._load_class_names()

    def unload_detection_model(self):
        """Unloads the detection model to free up memory."""
        self._cached_detection_model = None
        self._cached_model_path = None
        self.yolo = None
        self.logger.info("Tracker: Detection model reference cleared.")

    def unload_pose_model(self):
        """Unloads the pose model to free up memory."""
        self.yolo_pose = None
        self.logger.info("Tracker: Pose model reference cleared.")

    def unload_all_models(self):
        """Unload all models and clear references."""
        self._cached_detection_model = None
        self._cached_model_path = None
        self.yolo = None
        self.yolo_pose = None
        self.logger.info("Tracker: All models unloaded and GPU memory cleared.")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory usage statistics (placeholder since pool removed)."""
        return {"pool_removed": True}

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
        if mode in ["YOLO_ROI", "USER_FIXED_ROI", "OSCILLATION_DETECTOR", "OSCILLATION_DETECTOR_LEGACY", "OSCILLATION_DETECTOR_EXPERIMENTAL_2"]:
            if self.tracking_mode != mode:
                previous_mode = self.tracking_mode
                # Before switching, update caches from current state
                self._update_roi_caches_from_current()
                self.tracking_mode = mode
                self.logger.info(f"Tracker mode changed: {previous_mode} -> {self.tracking_mode}")
                # Clear ALL drawn overlays when switching modes (ROI rectangles, oscillation area, YOLO ROI box)
                self.clear_all_drawn_overlays()
                # After clearing, restore ROI from cache for the target mode (if any)
                self._restore_roi_for_current_mode()
        else:
            self.logger.warning(f"Attempted to set invalid tracking mode: {mode}")

    def clear_all_drawn_overlays(self) -> None:
        """Clears any visuals drawn on the video (ROI rectangles, oscillation area, YOLO ROI box).
        Also resets flags so UI does not re-render stale overlays.
        """
        self.logger.debug("clear_all_drawn_overlays: invoked")
        # Clear manual user ROI and point
        if self.user_roi_fixed is not None or self.user_roi_initial_point_relative is not None:
            self.logger.info(f"Clearing user ROI: {self.user_roi_fixed}, point_rel={self.user_roi_initial_point_relative}")
        # Use silent=True so caches are preserved across mode switches
        self.clear_user_defined_roi_and_point(silent=True)

        # Clear oscillation visualization
        if hasattr(self, 'clear_oscillation_area_and_point') and getattr(self, 'oscillation_area_fixed', None) is not None:
            self.logger.info(f"Clearing oscillation area: {self.oscillation_area_fixed} with {len(getattr(self, 'oscillation_grid_blocks', []))} blocks")
            # Use silent=True so caches are preserved across mode switches
            self.clear_oscillation_area_and_point(silent=True)

        # Clear YOLO ROI box and related state
        if self.roi is not None:
            self.logger.info(f"Clearing YOLO ROI: {self.roi}")
        self.roi = None
        self.prev_gray_main_roi = None
        self.prev_features_main_roi = None
        self.penis_last_known_box = None
        self.main_interaction_class = None

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

    def set_user_defined_roi_and_point(self, roi_abs_coords: Tuple[int, int, int, int], point_abs_coords_in_frame: Tuple[int, int], current_frame_for_patch: Optional[np.ndarray]):
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
                self.logger.info(f"Initial gray patch for User ROI captured, shape: {self.prev_gray_user_roi_patch.shape}")
            else:
                self.logger.warning("User defined ROI resulted in zero-size patch. Patch not set.")
                self.prev_gray_user_roi_patch = None
        else:
            self.logger.warning("Frame for patch not provided or empty during User ROI setup.")
            self.prev_gray_user_roi_patch = None
        self.primary_flow_history_smooth.clear()
        self.secondary_flow_history_smooth.clear()
        self.user_roi_current_flow_vector = (0.0, 0.0)

        # Update cache for USER_FIXED_ROI mode
        self._cache_user_roi['roi'] = self.user_roi_fixed
        self._cache_user_roi['initial_rel'] = self.user_roi_initial_point_relative
        self._cache_user_roi['tracked_rel'] = self.user_roi_tracked_point_relative

    def clear_user_defined_roi_and_point(self, silent=False):
        self.user_roi_fixed = None
        self.user_roi_initial_point_relative = None
        self.user_roi_tracked_point_relative = None
        self.prev_gray_user_roi_patch = None
        self.user_roi_current_flow_vector = (0.0, 0.0)
        # Clear cache on explicit user action (silent=False)
        if not silent:
            self._cache_user_roi = {'roi': None, 'initial_rel': None, 'tracked_rel': None}
            self.logger.info("User defined ROI and point cleared.")

    def set_oscillation_area_and_point(self, area_abs_coords: Tuple[int, int, int, int], point_abs_coords_in_frame: Tuple[int, int], current_frame_for_patch: Optional[np.ndarray]):
        """Sets the oscillation detection area and initial tracking point."""
        self.oscillation_area_fixed = area_abs_coords
        ax, ay, aw, ah = area_abs_coords
        point_x_frame, point_y_frame = point_abs_coords_in_frame
        # Set the fixed oscillation area
        self.oscillation_area_fixed = area_abs_coords
        # Calculate relative point within the area
        area_x, area_y, area_w, area_h = area_abs_coords
        point_x, point_y = point_abs_coords_in_frame
        # Clamp point to area bounds
        rel_x = max(0, min(area_w - 1, point_x - area_x))
        rel_y = max(0, min(area_h - 1, point_y - area_y))
        self.oscillation_area_initial_point_relative = (rel_x, rel_y)

        # Recalculate the grid blocks for the new area
        self._calculate_oscillation_grid_layout()
        # Reset tracked point and block positions
        self.oscillation_area_tracked_point_relative = self.oscillation_area_initial_point_relative
        self.oscillation_active_block_positions = set()
        # Optionally log
        if hasattr(self, 'logger') and self.logger:
            self.logger.info(f"Oscillation area set: {area_abs_coords}, point: {point_abs_coords_in_frame}")
        if not (ax <= point_x_frame < ax + aw and ay <= point_y_frame < ay + ah):
            self.logger.warning(
                f"Selected point ({point_x_frame},{point_y_frame}) is outside defined oscillation area. Clamping.")
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

        # Reset oscillation detector state when (re)setting area to avoid stale state affecting performance
        self.prev_gray_oscillation = None
        if hasattr(self, 'oscillation_history') and self.oscillation_history is not None:
            self.oscillation_history.clear()
        if hasattr(self, 'oscillation_cell_persistence') and self.oscillation_cell_persistence is not None:
            self.oscillation_cell_persistence.clear()
        if hasattr(self, 'oscillation_active_block_positions') and self.oscillation_active_block_positions is not None:
            self.oscillation_active_block_positions.clear()

        # Update cache for OSCILLATION_DETECTOR and OSCILLATION_DETECTOR_LEGACY modes
        self._cache_oscillation['area'] = self.oscillation_area_fixed
        self._cache_oscillation['initial_rel'] = self.oscillation_area_initial_point_relative
        self._cache_oscillation['tracked_rel'] = self.oscillation_area_tracked_point_relative

    def _calculate_oscillation_grid_layout(self):
        """Calculates and stores the static grid layout for the oscillation area."""
        if not self.oscillation_area_fixed:
            return

        ax, ay, aw, ah = self.oscillation_area_fixed
        ax_c, ay_c = max(0, ax), max(0, ay)
        target_h = constants.DEFAULT_OSCILLATION_PROCESSING_TARGET_HEIGHT
        target_w = self.target_size_preprocess[0] if hasattr(self, 'target_size_preprocess') and self.target_size_preprocess else constants.YOLO_INPUT_SIZE
        aw_c = min(aw, target_w - ax_c)
        ah_c = min(ah, target_h - ay_c)

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

    def clear_oscillation_area_and_point(self, silent: bool = False):
        """Clears the oscillation detection area and tracking point.
        If silent=True, preserve cache (used on mode switch)."""
        self.oscillation_area_fixed = None
        self.oscillation_area_initial_point_relative = None
        self.oscillation_area_tracked_point_relative = None
        self.prev_gray_oscillation_area_patch = None
        self.oscillation_grid_blocks = []

        # Also reset oscillation detector state to prevent persistent history from impacting full-frame runs
        self.prev_gray_oscillation = None
        if hasattr(self, 'oscillation_history') and self.oscillation_history is not None:
            self.oscillation_history.clear()
        if hasattr(self, 'oscillation_cell_persistence') and self.oscillation_cell_persistence is not None:
            self.oscillation_cell_persistence.clear()
        if hasattr(self, 'oscillation_active_block_positions') and self.oscillation_active_block_positions is not None:
            self.oscillation_active_block_positions.clear()
        if not silent:
            self._cache_oscillation = {'area': None, 'initial_rel': None, 'tracked_rel': None}
            self.logger.info("Oscillation area and point cleared.")

    def _update_roi_caches_from_current(self) -> None:
        """Capture current ROI state into per-mode caches before switching modes."""
        if self.user_roi_fixed is not None or self.user_roi_initial_point_relative is not None:
            self._cache_user_roi['roi'] = self.user_roi_fixed
            self._cache_user_roi['initial_rel'] = self.user_roi_initial_point_relative
            self._cache_user_roi['tracked_rel'] = self.user_roi_tracked_point_relative
        if self.oscillation_area_fixed is not None or self.oscillation_area_initial_point_relative is not None:
            self._cache_oscillation['area'] = self.oscillation_area_fixed
            self._cache_oscillation['initial_rel'] = self.oscillation_area_initial_point_relative
            self._cache_oscillation['tracked_rel'] = self.oscillation_area_tracked_point_relative

    def _restore_roi_for_current_mode(self) -> None:
        """Restore cached ROI state for the active mode after switching."""
        if self.tracking_mode == "USER_FIXED_ROI":
            cached_roi = self._cache_user_roi.get('roi')
            if cached_roi:
                self.user_roi_fixed = cached_roi
                self.user_roi_initial_point_relative = self._cache_user_roi.get('initial_rel')
                self.user_roi_tracked_point_relative = self._cache_user_roi.get('tracked_rel') or self.user_roi_initial_point_relative
                # Defer patch recreation to processing step; ensure histories reset
                self.prev_gray_user_roi_patch = None
                self.primary_flow_history_smooth.clear()
                self.secondary_flow_history_smooth.clear()
        elif self.tracking_mode in ["OSCILLATION_DETECTOR", "OSCILLATION_DETECTOR_LEGACY", "OSCILLATION_DETECTOR_EXPERIMENTAL_2"]:
            cached_area = self._cache_oscillation.get('area')
            if cached_area:
                self.oscillation_area_fixed = cached_area
                self.oscillation_area_initial_point_relative = self._cache_oscillation.get('initial_rel')
                self.oscillation_area_tracked_point_relative = self._cache_oscillation.get('tracked_rel') or self.oscillation_area_initial_point_relative
                # Recalculate grid layout for restored area; defer patches to processing
                self._calculate_oscillation_grid_layout()

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
        """
        Aspect-preserving resize with letterbox to target size in a single pass.

        Rationale: Replaced the previous resize + copyMakeBorder sequence with a
        one-pass affine scale-and-center (cv2.warpAffine). This removes an extra
        intermediate image write, reducing memory bandwidth per frame while
        producing the same letterboxed result.
        """
        h, w = frame.shape[:2]
        target_w, target_h = self.target_size_preprocess
        if h <= 1 or w <= 1 or (w, h) == (target_w, target_h):
            return frame.copy()
        # Single-pass: scale and center via affine transform
        scale = min(target_w / float(w), target_h / float(h))
        if scale <= 0.0:
            return frame.copy()
        trans_x = (target_w - scale * w) * 0.5
        trans_y = (target_h - scale * h) * 0.5
        affine_trans = np.array([[scale, 0.0, trans_x], [0.0, scale, trans_y]], dtype=np.float32)
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        # Single render into the final target canvas (no separate border staging)
        return cv2.warpAffine(frame, affine_trans, (target_w, target_h), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=RGBColors.BLACK)

    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        detections = []

        # Check if detection model is available
        if not self.det_model_path or not os.path.exists(self.det_model_path):
            return detections

        # Determine discarded classes based on self.app context if available
        discarded_classes_runtime = []
        if self.app and hasattr(self.app, 'discarded_tracking_classes'):
            discarded_classes_runtime = self.app.discarded_tracking_classes

        # Direct YOLO model caching
        try:
            if self._cached_detection_model is None or self._cached_model_path != self.det_model_path:
                self.logger.info(f"Loading {self.det_model_path} for detection...")
                self._cached_detection_model = YOLO(self.det_model_path, task='detect')
                self._cached_model_path = self.det_model_path
                # Names
                try:
                    names_attr = getattr(self._cached_detection_model, 'names', None)
                    if names_attr:
                        if isinstance(names_attr, dict):
                            # Convert dict to ordered list by index if possible
                            try:
                                self.classes = [names_attr[k] for k in sorted(names_attr.keys(), key=lambda x: int(x))]
                            except Exception:
                                self.classes = list(names_attr.values())
                        elif isinstance(names_attr, (list, tuple)):
                            self.classes = list(names_attr)
                except Exception:
                    pass

            results = self._cached_detection_model(frame, device=constants.DEVICE, verbose=False, conf=self.confidence_threshold)

            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = None
                    if self.classes and 0 <= class_id < len(self.classes):
                        class_name = self.classes[class_id]
                    else:
                        # Try names in result
                        rn = getattr(result, 'names', None)
                        if isinstance(rn, dict) and class_id in rn:
                            class_name = rn[class_id]
                        else:
                            class_name = f"class_{class_id}"

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

    def _find_interacting_objects(self, penis_box_xywh: Tuple[int, int, int, int], all_detections: List[Dict]) -> List[
        Dict]:
        if not penis_box_xywh or not all_detections: return []
        px, py, pw, ph = penis_box_xywh
        pcx, pcy = px + pw // 2, py + ph // 2
        interacting = []
        for obj in all_detections:
            if obj["class_name"].lower() in self.INTERACTION_CLASSES:
                ox, oy, ow, oh = obj["box"]
                ocx, ocy = ox + ow // 2, oy + oh // 2
                dist = np.sqrt((ocx - pcx) ** 2 + (ocy - pcy) ** 2)
                max_dist = (np.sqrt(pw ** 2 + ph ** 2) / 2 + np.sqrt(ow ** 2 + oh ** 2) / 2) * 0.85
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
            self.logger.debug(f"DIS Optical Flow preset updated to: {self.dis_flow_preset}")

        if finest_scale is not None and finest_scale != self.dis_finest_scale:
            # A value of 0 from the UI means 'auto', which we can represent as None internally
            self.dis_finest_scale = finest_scale if finest_scale > 0 else None
            self.logger.debug(f"DIS Optical Flow finest scale updated to: {self.dis_finest_scale}")

        dis_preset_map = {
            "ULTRAFAST": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
            "FAST": cv2.DISOPTICAL_FLOW_PRESET_FAST,
            "MEDIUM": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
        }

        try:
            selected_preset_cv = dis_preset_map.get(self.dis_flow_preset.upper(), cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
            # Re-create the dense flow object with the new settings
            self.flow_dense = cv2.DISOpticalFlow_create(selected_preset_cv)
            if self.dis_finest_scale is not None:
                self.flow_dense.setFinestScale(self.dis_finest_scale)
            self.logger.debug("Successfully re-initialized DIS Optical Flow object with new configuration.")
        except AttributeError:
            self.logger.debug("cv2.DISOpticalFlow_create not found or preset invalid while updating config. Optical flow may not work.")
            self.flow_dense = None
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while updating DIS flow config: {e}")
            self.flow_dense = None

    def _calculate_flow_in_patch(self, patch_gray: np.ndarray, prev_patch_gray: Optional[np.ndarray], use_sparse: bool = False, prev_features_for_sparse: Optional[np.ndarray] = None) -> Tuple[float, float, Optional[np.ndarray], Optional[np.ndarray]]:
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
                good_next = next_feat[status == 1] if next_feat is not None else np.array(
                    [])  # Ensure good_next is an array

                if len(good_prev) > 0 and len(good_next) > 0 and good_next.ndim == 2 and good_next.shape[
                    1] == 2:  # Check shape
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

    def process_main_roi_content(self, processed_frame_draw_target: np.ndarray, current_roi_patch_gray: np.ndarray, prev_roi_patch_gray: Optional[np.ndarray], prev_sparse_features: Optional[np.ndarray]) -> Tuple[int, int, float, float, Optional[np.ndarray]]:

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
                                processed_frame_draw_target[ry:ry + rh, rx:rx + rw] = cv2.addWeighted(roi_display_patch, 0.5, vis.astype(
                                        roi_display_patch.dtype), 0.5, 0)
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

    def process_frame(self, frame: np.ndarray, frame_time_ms: int, frame_index: Optional[int] = None, min_write_frame_id: Optional[int] = None) -> Tuple[np.ndarray, Optional[List[Dict]]]:
        if self.tracking_mode == "OSCILLATION_DETECTOR":
            return self.process_frame_for_oscillation(frame, frame_time_ms, frame_index)
        elif self.tracking_mode == "OSCILLATION_DETECTOR_LEGACY":
            return self.process_frame_for_oscillation_legacy(frame, frame_time_ms, frame_index)
        elif self.tracking_mode == "OSCILLATION_DETECTOR_EXPERIMENTAL_2":
            return self.process_frame_for_oscillation_experimental_2(frame, frame_time_ms, frame_index)

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
                penis_boxes = [obj["box"] for obj in detected_objects_this_frame if obj["class_name"].lower() == "penis"]
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
                        self.logger.debug(f"Main interaction class: {self.main_interaction_class}")
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

            self.stats_display = [f"T-FPS:{self.current_fps:.1f} T(ms):{frame_time_ms} Amp:{self.current_effective_amp_factor:.2f}x"]
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

                        dy_smooth = np.median(
                            self.primary_flow_history_smooth) if self.primary_flow_history_smooth else dy_raw
                        dx_smooth = np.median(
                            self.secondary_flow_history_smooth) if self.secondary_flow_history_smooth else dx_raw

                        # Apply scaling and generate final position
                        size_factor = 1.0  # No object detection in this mode
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

        if self.app and self.tracking_active and (min_write_frame_id is None or (frame_index is not None and frame_index >= min_write_frame_id)):

            # Determine which axis we will write for this frame
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

            # --- Automatic Lag Compensation ---
            # Calculate the inherent delay from the smoothing window. A window of size N has a lag of (N-1)/2 frames.
            # A window size of 1 means no smoothing and no delay.
            automatic_smoothing_delay_frames = (self.flow_history_window_smooth - 1) / 2.0 if self.flow_history_window_smooth > 1 else 0.0

            # Combine the automatic compensation with the user's manual delay setting.
            total_delay_frames = self.output_delay_frames + automatic_smoothing_delay_frames

            # Convert the total frame delay to milliseconds.
            base_delay_ms = (total_delay_frames / self.current_video_fps_for_delay) * 1000.0 if self.current_video_fps_for_delay > 0 else 0.0

            # Immediate-visibility safeguard after a clear: for the first live point on the axis we are writing, bypass delay
            primary_empty = (len(self.funscript.primary_actions) == 0)
            secondary_empty = (len(self.funscript.secondary_actions) == 0)

            effective_delay_ms = base_delay_ms
            if (primary_to_write is not None and primary_empty) or (secondary_to_write is not None and secondary_empty):
                effective_delay_ms = 0.0

            # Adjust the timestamp with the effective delay
            adjusted_frame_time_ms = frame_time_ms - effective_delay_ms

            is_file_processing_context = frame_index is not None

            self.funscript.add_action(
                timestamp_ms=int(round(adjusted_frame_time_ms)),
                primary_pos=primary_to_write,
                secondary_pos=secondary_to_write,
                is_from_live_tracker=(not is_file_processing_context)
            )
            action_log_list.append({
                "at": int(round(adjusted_frame_time_ms)), "pos": primary_to_write, "secondary_pos": secondary_to_write,
                "raw_ud_pos_computed": final_primary_pos, "raw_lr_pos_computed": final_secondary_pos,
                "mode": current_tracking_axis_mode,
                "target": current_single_axis_output if current_tracking_axis_mode != "both" else "N/A",
                "raw_at": frame_time_ms, "delay_applied_ms": effective_delay_ms,
                "roi_main": self.roi if self.tracking_mode == "YOLO_ROI" else self.user_roi_fixed,
                "amp": self.current_effective_amp_factor
            })

        if self.show_masks and detected_objects_this_frame and self.tracking_mode == "YOLO_ROI":
            self.draw_detections(processed_frame, detected_objects_this_frame)

        # YOLO ROI rectangle overlay
        if self.tracking_mode == "YOLO_ROI" and self.show_roi and self.roi:
            rx, ry, rw, rh = self.roi
            color = self.get_class_color(
                self.main_interaction_class or ("penis" if self.penis_last_known_box else "persisting"))
            cv2.rectangle(processed_frame, (rx, ry), (rx + rw, ry + rh), color, 1)
            status_text = self.main_interaction_class or ('P' if self.penis_last_known_box else 'Lost...')
            cv2.putText(processed_frame, f"ROI:{status_text}", (rx, ry - 2), cv2.FONT_HERSHEY_PLAIN, 0.7, color, 1)
            if not self.penis_last_known_box:
                cv2.putText(processed_frame, f"Lost: {self.frames_since_target_lost}/{self.max_frames_for_roi_persistence}", (rx, ry + rh + 10), cv2.FONT_HERSHEY_PLAIN, 0.6, RGBColors.BLUE, 1)

        # User-defined ROI rectangle overlay
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
        if self.tracking_mode in ["OSCILLATION_DETECTOR", "OSCILLATION_DETECTOR_LEGACY", "OSCILLATION_DETECTOR_EXPERIMENTAL_2"]:
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
            # Force-recreate optical flow engine to eliminate internal state carry-over
            try:
                self.logger.debug(f"Oscillation start: recreating DIS flow (preset={self.dis_flow_preset}, finest_scale={self.dis_finest_scale})")
                self.update_dis_flow_config()
            except Exception as e:
                self.logger.warning(f"Oscillation start: DIS flow recreate failed: {e}")
            self.oscillation_funscript_pos = 50

            self.logger.info(f"Oscillation detector started. History size set to {self.oscillation_history_max_len} frames.")

        # Dynamically set the motion history window to match the video's FPS (~1 second buffer)
        if self.app and hasattr(self.app, 'processor') and self.app.processor.fps > 0:
            new_window_size = int(round(self.app.processor.fps))
            self.motion_mode_history_window = new_window_size
            self.logger.info(f"Motion history window set to video FPS: {new_window_size} frames.")
        else:
            # Fallback to the default if FPS is not available
            self.motion_mode_history_window = 30
            self.logger.info(
                f"Falling back to default motion history window: {self.motion_mode_history_window} frames.")

        if self.tracking_mode == "YOLO_ROI":  # Also applies to S3 like processing
            self.frames_since_target_lost = 0
            self.penis_max_size_history.clear()
            self.prev_gray_main_roi, self.prev_features_main_roi = None, None
            self.penis_last_known_box, self.main_interaction_class = None, None  # main_interaction_class set by live or S3 segment
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
                self.logger.debug(f"Oscillation sensitivity updated to {self.oscillation_sensitivity}")

    def reset(self, reason: Optional[str] = None):
        self.stop_tracking()  # Explicitly set tracking_active to False.
        preserve_user_roi = (reason == "stop_preserve_funscript")
        if not preserve_user_roi:
            # Full reset clears any user-defined ROI/point and returns to default mode
            self.clear_user_defined_roi_and_point(silent=True)  # Silent to avoid duplicate messages from clear_all_drawn_overlays
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
                        self.logger.info(f"Live tracking using preprocessed video: {os.path.basename(current_status.get('path', 'unknown'))} ({current_status.get('frame_count', 0)} frames)", extra={'status_message': True})
                    else:
                        self.logger.warning("Live tracking: preprocessed video detected but invalid, using original video", extra={'status_message': True})
                else:
                    if current_status.get("exists", False):
                        self.logger.info(
                            "Live tracking using original video (preprocessed video available but not used)", extra={'status_message': True})
                    else:
                        self.logger.info("Live tracking using original video (no preprocessed video available)", extra={'status_message': True})

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

    def process_frame_for_oscillation(self, frame: np.ndarray, frame_time_ms: int, frame_index: Optional[int] = None) -> \
    Tuple[np.ndarray, Optional[List[Dict]]]:
        """
        [V9 - Advanced Filtering] Implements global motion cancellation, advanced oscillation scoring,
        and a VR-specific focus on the central third of the frame.
        """

        self._update_fps()

        try:
            target_height = getattr(self.app.app_settings, 'get', lambda k, d=None: d)('oscillation_processing_target_height', constants.DEFAULT_OSCILLATION_PROCESSING_TARGET_HEIGHT)
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
            # Preprocess entire frame to expected working size first
            processed_frame = self.preprocess_frame(frame)
            # Crop to area first, then convert only the crop to grayscale (avoid full-frame cvtColor)
            processed_frame_area = processed_frame[ay:ay+ah, ax:ax+aw]
            # Reuse/allocate ROI gray buffer
            if self._gray_roi_buffer is None or self._gray_roi_buffer.shape[:2] != (processed_frame_area.shape[0], processed_frame_area.shape[1]):
                self._gray_roi_buffer = np.empty((processed_frame_area.shape[0], processed_frame_area.shape[1]), dtype=np.uint8)
            cv2.cvtColor(processed_frame_area, cv2.COLOR_BGR2GRAY, dst=self._gray_roi_buffer)
            current_gray = self._gray_roi_buffer
        else:
            processed_frame = self.preprocess_frame(frame)
            # Full-frame path: compute grayscale once for full frame
            target_h, target_w = processed_frame.shape[0], processed_frame.shape[1]
            if self._gray_full_buffer is None or self._gray_full_buffer.shape[:2] != (target_h, target_w):
                self._gray_full_buffer = np.empty((target_h, target_w), dtype=np.uint8)
            cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY, dst=self._gray_full_buffer)
            current_gray = self._gray_full_buffer
            processed_frame_area = processed_frame
            ax, ay = 0, 0
            aw, ah = processed_frame.shape[1], processed_frame.shape[0]
        action_log_list = []
        active_blocks = getattr(self, 'oscillation_active_block_positions', set())
        is_camera_motion = getattr(self, 'is_camera_motion', False)

        # Compute dynamic grid based on current analysis image size (used in vis and scoring)
        img_h, img_w = current_gray.shape[:2]
        grid_size = max(1, int(self.oscillation_grid_size))
        local_block_size = max(8, min(img_h // grid_size, img_w // grid_size))
        if local_block_size <= 0:
            local_block_size = 8
        num_rows = max(1, img_h // local_block_size)
        num_cols = max(1, img_w // local_block_size)
        min_cell_activation_pixels = (local_block_size * local_block_size) * 0.05

        # --- Visualization: optional static grid blocks overlay (independent of show_masks) ---
        if getattr(self, 'show_grid_blocks', False):
            # Draw a static grid over the current analysis area (ROI or full frame)
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

        # --- Detection logic: operate only on area ---
        if self.prev_gray_oscillation is None or self.prev_gray_oscillation.shape != current_gray.shape:
            self.prev_gray_oscillation = current_gray.copy()
            return processed_frame, None

        if not hasattr(self, 'flow_dense_osc') or not self.flow_dense_osc:
            self.logger.warning("Dense optical flow not available for oscillation detection.")
            return processed_frame, None

        # --- Step 1: Calculate Global Optical Flow & Global Motion Vector ---
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

        # Check if the video is VR to apply the focus rule
        is_vr = self._is_vr_video()

        # Apply VR central-third focus only for full-frame scans. When an ROI is set,
        # scan the full ROI width to avoid off-center ROI being ignored.
        apply_vr_central_focus = is_vr and not use_oscillation_area
        # Use effective grid size for current image to compute central thirds
        eff_cols = max(1, min(self.oscillation_grid_size, current_gray.shape[1] // self.oscillation_block_size))
        vr_central_third_start = eff_cols // 3
        vr_central_third_end = 2 * eff_cols // 3

        newly_active_cells = set()
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
                if (r, c) not in self.oscillation_history:
                    self.oscillation_history[(r, c)] = deque(maxlen=self.oscillation_history_max_len)
                self.oscillation_history[(r, c)].append({'dx': local_dx, 'dy': local_dy, 'mag': mag})

        # --- Step 4 & 5 Combined: Adaptive Motion Calculation ---
        final_dy, final_dx = 0.0, 0.0
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

                    if oscillation_score > 0.5:  # Filter out low-scoring blocks
                        candidate_blocks.append({**motion, 'score': oscillation_score})

            if candidate_blocks:
                max_score = max(b['score'] for b in candidate_blocks)
                # Take any block that has at least 40% of the max score.
                active_blocks = [b for b in candidate_blocks if b['score'] >= max_score * 0.4]

        # --- ADAPTIVE LOGIC PATH ---

        # Knob 3: The number of active cells at or below which we use the "Follow the Leader" logic.
        SPARSITY_THRESHOLD = 2

        if 0 < len(active_blocks) <= SPARSITY_THRESHOLD:
            # --- Sparse Motion Path ("Follow the Leader") ---
            # Ideal for localized action like handjobs/blowjobs.
            # Find the single block with the highest raw motion magnitude (most energy).
            if hasattr(self, 'logger'): self.logger.debug(f"Sparse motion detected ({len(active_blocks)} blocks). Following the leader.")

            leader_block = max(active_blocks, key=lambda b: b['mag'])
            final_dy = leader_block['dy']
            final_dx = leader_block['dx']

        elif len(active_blocks) > SPARSITY_THRESHOLD:
            # --- Dense Motion Path (Weighted Average) ---
            # Ideal for full-body motion. Uses the original democratic logic.
            if hasattr(self, 'logger'): self.logger.debug(f"Dense motion detected ({len(active_blocks)} blocks). Using weighted average.")

            total_weight = sum(b['score'] for b in active_blocks)
            if total_weight > 0:
                # Calculate the weighted average velocity for this frame
                final_dy = sum(b['dy'] * b['score'] for b in active_blocks) / total_weight
                final_dx = sum(b['dx'] * b['score'] for b in active_blocks) / total_weight

        # If no blocks are active, final_dy and final_dx remain 0.0, and the position holds.

        # --- Enhanced signal processing with legacy improvements ---
        # Check for simple amplification mode setting
        use_simple_amplification = getattr(self.app.app_settings, 'get', lambda k, d: d)('oscillation_use_simple_amplification', False) if self.app else False
        enable_decay = getattr(self.app.app_settings, 'get', lambda k, d: d)('oscillation_enable_decay', True) if self.app else True
        
        if abs(final_dy) > 0.01 or abs(final_dx) > 0.01:
            # Update last active time for decay mechanism
            self.oscillation_last_active_time = frame_time_ms
            
            if use_simple_amplification:
                # Legacy-style simple amplification
                max_deviation = 49 * self.oscillation_sensitivity
                new_raw_primary_pos = 50 + np.clip(final_dy * -10, -max_deviation, max_deviation)
                new_raw_secondary_pos = 50 + np.clip(final_dx * 10, -max_deviation, max_deviation)
                
                # Apply EMA smoothing
                alpha = self.oscillation_ema_alpha
                self.oscillation_last_known_pos = self.oscillation_last_known_pos * (1 - alpha) + new_raw_primary_pos * alpha
                self.oscillation_last_known_secondary_pos = self.oscillation_last_known_secondary_pos * (1 - alpha) + new_raw_secondary_pos * alpha
            else:
                # Current dynamic scaling approach
                base_sensitivity_scaler = 2.5  # 1.5
                intensity_exponent = 0.7

                dynamic_scaler_y = base_sensitivity_scaler * (abs(final_dy) ** intensity_exponent) if abs(final_dy) > 0.1 else base_sensitivity_scaler
                dynamic_scaler_x = base_sensitivity_scaler * (abs(final_dx) ** intensity_exponent) if abs(final_dx) > 0.1 else base_sensitivity_scaler

                primary_pos_change = -final_dy * dynamic_scaler_y
                secondary_pos_change = final_dx * dynamic_scaler_x

                new_primary_pos = self.oscillation_last_known_pos + primary_pos_change
                new_secondary_pos = self.oscillation_last_known_secondary_pos + secondary_pos_change

                alpha = self.oscillation_ema_alpha
                self.oscillation_last_known_pos = (self.oscillation_last_known_pos * (1 - alpha)) + (new_primary_pos * alpha)
                self.oscillation_last_known_secondary_pos = (self.oscillation_last_known_secondary_pos * (1 - alpha)) + (new_secondary_pos * alpha)
            
            # Clip to valid range
            self.oscillation_last_known_pos = np.clip(self.oscillation_last_known_pos, 0, 100)
            self.oscillation_last_known_secondary_pos = np.clip(self.oscillation_last_known_secondary_pos, 0, 100)
        
        elif enable_decay:
            # Legacy-style decay mechanism when no motion is detected
            hold_duration_ms = getattr(self.app.app_settings, 'get', lambda k, d: d)('oscillation_hold_duration_ms', 250) if self.app else 250
            decay_factor = getattr(self.app.app_settings, 'get', lambda k, d: d)('oscillation_decay_factor', 0.95) if self.app else 0.95
            
            time_since_last_active = frame_time_ms - self.oscillation_last_active_time
            if time_since_last_active > hold_duration_ms:
                # Decay towards center after hold duration expires
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

            self.funscript.add_action(timestamp_ms=frame_time_ms, primary_pos=primary_to_write, secondary_pos=secondary_to_write)
            action_log_list.append({"at": frame_time_ms, "pos": primary_to_write, "secondary_pos": secondary_to_write})

        # Step 7: Visualization
        # Draw oscillation grid overlay if enabled
        if self.show_masks:
            active_block_positions = {b['pos'] for b in active_blocks}
            for r,c in list(self.oscillation_cell_persistence.keys()):
                x1, y1 = c * local_block_size + ax, r * local_block_size + ay
                color = (0, 255, 0) if (r, c) in active_block_positions else (180, 100, 100)
                cv2.rectangle(processed_frame, (x1, y1), (x1 + local_block_size, y1 + local_block_size), color, 1)

        # Keep a reusable prev gray buffer
        if self._prev_gray_osc_buffer is None or self._prev_gray_osc_buffer.shape != current_gray.shape:
            self._prev_gray_osc_buffer = np.empty_like(current_gray)
        np.copyto(self._prev_gray_osc_buffer, current_gray)
        self.prev_gray_oscillation = self._prev_gray_osc_buffer

        # Lightweight instrumentation (debug level, once per second)
        if self.logger and self.logger.isEnabledFor(logging.DEBUG):
            try:
                now_sec = time.time()
                last = getattr(self, '_osc_instr_last_log_sec', 0.0)
                if now_sec - last >= 1.0:
                    self._osc_instr_last_log_sec = now_sec
                    cur_rows = max(1, current_gray.shape[0] // max(1, local_block_size))
                    cur_cols = max(1, current_gray.shape[1] // max(1, local_block_size))
                    self.logger.debug(
                        f"OSC perf: area={use_oscillation_area} img={current_gray.shape} grid={cur_rows}x{cur_cols} "
                        f"preset={self.dis_flow_preset} finest={self.dis_finest_scale} active_cells={len(self.oscillation_cell_persistence)} "
                        f"fps={self.current_fps:.1f}")
            except Exception:
                pass
        return processed_frame, action_log_list if action_log_list else None

    def process_frame_for_oscillation_experimental_2(self, frame: np.ndarray, frame_time_ms: int, frame_index: Optional[int] = None) -> \
    Tuple[np.ndarray, Optional[List[Dict]]]:
        """
        [Experimental 2] Hybrid oscillation detector combining experimental timing precision 
        with legacy amplification and signal conditioning. Best of both worlds approach:
        
        From Experimental:
        - Zero-crossing analysis for precise peak/valley detection
        - Adaptive motion logic (sparse/dense paths) 
        - Global motion cancellation
        - Advanced frequency/variance scoring
        
        From Legacy:
        - Live dynamic amplification with percentile normalization
        - Amplitude-aware scaling with natural response
        - Cohesion analysis for spatial consistency
        - Proven frequency weighting (2.5Hz optimal)
        """

        self._update_fps()

        try:
            target_height = getattr(self.app.app_settings, 'get', lambda k, d=None: d)('oscillation_processing_target_height', constants.DEFAULT_OSCILLATION_PROCESSING_TARGET_HEIGHT)
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
            processed_frame = self.preprocess_frame(frame)
            processed_frame_area = processed_frame[ay:ay+ah, ax:ax+aw]
            if self._gray_roi_buffer is None or self._gray_roi_buffer.shape[:2] != (processed_frame_area.shape[0], processed_frame_area.shape[1]):
                self._gray_roi_buffer = np.empty((processed_frame_area.shape[0], processed_frame_area.shape[1]), dtype=np.uint8)
            cv2.cvtColor(processed_frame_area, cv2.COLOR_BGR2GRAY, dst=self._gray_roi_buffer)
            current_gray = self._gray_roi_buffer
        else:
            processed_frame = self.preprocess_frame(frame)
            target_h, target_w = processed_frame.shape[0], processed_frame.shape[1]
            if self._gray_full_buffer is None or self._gray_full_buffer.shape[:2] != (target_h, target_w):
                self._gray_full_buffer = np.empty((target_h, target_w), dtype=np.uint8)
            cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY, dst=self._gray_full_buffer)
            current_gray = self._gray_full_buffer
            processed_frame_area = processed_frame
            ax, ay = 0, 0
            aw, ah = processed_frame.shape[1], processed_frame.shape[0]

        action_log_list = []
        active_blocks = getattr(self, 'oscillation_active_block_positions', set())
        is_camera_motion = getattr(self, 'is_camera_motion', False)

        # Compute dynamic grid based on current analysis image size
        img_h, img_w = current_gray.shape[:2]
        grid_size = max(1, int(self.oscillation_grid_size))
        local_block_size = max(8, min(img_h // grid_size, img_w // grid_size))
        if local_block_size <= 0:
            local_block_size = 8
        num_rows = max(1, img_h // local_block_size)
        num_cols = max(1, img_w // local_block_size)
        min_cell_activation_pixels = (local_block_size * local_block_size) * 0.05

        # --- Visualization: optional static grid blocks overlay ---
        if getattr(self, 'show_grid_blocks', False):
            start_x, start_y = ax, ay
            end_x, end_y = ax + aw, ay + ah
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
        active_blocks = []

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
                active_blocks = [b for b in candidate_blocks if b['score'] > max_score * 0.6]

        # --- Step 5: ADAPTIVE MOTION CALCULATION (FROM EXPERIMENTAL) ---
        SPARSITY_THRESHOLD = 2

        if 0 < len(active_blocks) <= SPARSITY_THRESHOLD:
            # Sparse Motion Path ("Follow the Leader")
            if hasattr(self, 'logger'): 
                self.logger.debug(f"Experimental 2: Sparse motion detected ({len(active_blocks)} blocks). Following the leader.")
            leader_block = max(active_blocks, key=lambda b: b['mag'])
            final_dy = leader_block['dy']
            final_dx = leader_block['dx']

        elif len(active_blocks) > SPARSITY_THRESHOLD:
            # Dense Motion Path (Weighted Average)
            if hasattr(self, 'logger'): 
                self.logger.debug(f"Experimental 2: Dense motion detected ({len(active_blocks)} blocks). Using weighted average.")
            total_weight = sum(b['score'] for b in active_blocks)
            if total_weight > 0:
                final_dy = sum(b['dy'] * b['score'] for b in active_blocks) / total_weight
                final_dx = sum(b['dx'] * b['score'] for b in active_blocks) / total_weight

        # --- Step 6: LEGACY SIGNAL CONDITIONING ---
        if active_blocks:
            # Calculate magnitude for amplitude-aware scaling
            final_mag = np.sqrt(final_dy ** 2 + final_dx ** 2)
            
            # LEGACY: Amplitude-aware scaling with natural response
            amplitude_scaler = np.clip(final_mag / 4.0, 0.7, 1.5)
            max_deviation = 49 * self.oscillation_sensitivity * amplitude_scaler
            scaled_dy = np.clip(final_dy * -10, -max_deviation, max_deviation)
            scaled_dx = np.clip(final_dx * 10, -max_deviation, max_deviation)
            new_raw_primary_pos = np.clip(50 + scaled_dy, 0, 100)
            new_raw_secondary_pos = np.clip(50 + scaled_dx, 0, 100)
        else:
            # Decay towards center when no motion
            new_raw_primary_pos = self.oscillation_last_known_pos * 0.95 + 50 * 0.05
            new_raw_secondary_pos = self.oscillation_last_known_secondary_pos * 0.95 + 50 * 0.05

        # --- Step 7: LEGACY LIVE DYNAMIC AMPLIFICATION ---
        self.oscillation_position_history.append(new_raw_primary_pos)
        final_primary_pos = new_raw_primary_pos
        final_secondary_pos = new_raw_secondary_pos

        # LEGACY: Live dynamic amplification with percentile normalization (anti-plateau)
        if self.live_amp_enabled and len(self.oscillation_position_history) > self.oscillation_position_history.maxlen * 0.5:
            p10 = np.percentile(self.oscillation_position_history, 10)
            p90 = np.percentile(self.oscillation_position_history, 90)
            effective_range = p90 - p10

            if effective_range > 15:  # Auto-adapts to prevent plateau
                normalized_pos = (new_raw_primary_pos - p10) / effective_range
                final_primary_pos = np.clip(normalized_pos * 100, 0, 100)
                
                # Apply same normalization to secondary axis
                normalized_secondary = (new_raw_secondary_pos - 50) / max(1, effective_range) + 0.5
                final_secondary_pos = np.clip(normalized_secondary * 100, 0, 100)

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
            active_block_positions = {b['pos'] for b in active_blocks}
            for r,c in list(self.oscillation_cell_persistence.keys()):
                x1, y1 = c * local_block_size + ax, r * local_block_size + ay
                color = (0, 255, 0) if (r, c) in active_block_positions else (180, 100, 100)
                cv2.rectangle(processed_frame, (x1, y1), (x1 + local_block_size, y1 + local_block_size), color, 1)

        # Keep reusable prev gray buffer
        if self._prev_gray_osc_buffer is None or self._prev_gray_osc_buffer.shape != current_gray.shape:
            self._prev_gray_osc_buffer = np.empty_like(current_gray)
        np.copyto(self._prev_gray_osc_buffer, current_gray)
        self.prev_gray_oscillation = self._prev_gray_osc_buffer

        # Debug instrumentation
        if self.logger and self.logger.isEnabledFor(logging.DEBUG):
            try:
                now_sec = time.time()
                last = getattr(self, '_osc_instr_last_log_sec', 0.0)
                if now_sec - last >= 1.0:
                    self._osc_instr_last_log_sec = now_sec
                    cur_rows = max(1, current_gray.shape[0] // max(1, local_block_size))
                    cur_cols = max(1, current_gray.shape[1] // max(1, local_block_size))
                    self.logger.debug(
                        f"OSC EXP2 perf: area={use_oscillation_area} img={current_gray.shape} grid={cur_rows}x{cur_cols} "
                        f"active_cells={len(self.oscillation_cell_persistence)} fps={self.current_fps:.1f} "
                        f"live_amp={self.live_amp_enabled}")
            except Exception:
                pass

        return processed_frame, action_log_list if action_log_list else None

    def process_frame_for_oscillation_legacy(self, frame: np.ndarray, frame_time_ms: int, frame_index: Optional[int] = None) -> \
    Tuple[np.ndarray, Optional[List[Dict]]]:
        """
        [V3] Legacy oscillation detector from commit c9e6fbd. Processes a frame to detect rhythmic oscillations. 
        Culls black bars, uses a configurable grid, and employs cohesion, frequency weighting, and EMA smoothing 
        for a natural signal with live dynamic amplification.
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

        # --- Detect active video area to ignore black bars ---
        active_area_rect = (0, 0, processed_frame.shape[1], processed_frame.shape[0])
        if np.any(current_gray):
            rows = np.any(current_gray, axis=1)
            cols = np.any(current_gray, axis=0)
            if np.any(rows) and np.any(cols):
                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]
                active_area_rect = (x_min, y_min, x_max, y_max)

        prev_gray_cont = np.ascontiguousarray(self.prev_gray_oscillation)
        current_gray_cont = np.ascontiguousarray(current_gray)
        flow = self.flow_dense.calc(prev_gray_cont, current_gray_cont, None)

        if flow is None:
            self.prev_gray_oscillation = current_gray
            return processed_frame, None

        # Analyze flow in grid blocks
        block_motions = []
        ax_min, ay_min, ax_max, ay_max = active_area_rect
        for r in range(self.oscillation_grid_size):
            for c in range(self.oscillation_grid_size):
                y_start, x_start = r * self.oscillation_block_size, c * self.oscillation_block_size

                # --- Skip blocks in black bar areas ---
                block_center_x = x_start + self.oscillation_block_size / 2
                block_center_y = y_start + self.oscillation_block_size / 2

                if not (ax_min < block_center_x < ax_max and ay_min < block_center_y < ay_max):
                    continue  # Skip blocks in padded areas

                block_flow = flow[y_start:y_start + self.oscillation_block_size,
                             x_start:x_start + self.oscillation_block_size]

                if block_flow.size > 0:
                    dx, dy = np.median(block_flow[..., 0]), np.median(block_flow[..., 1])
                    mag = np.sqrt(dx ** 2 + dy ** 2)
                    block_motions.append({'dx': dx, 'dy': dy, 'mag': mag, 'pos': (r, c)})

                    block_pos = (r, c)
                    if block_pos not in self.oscillation_history:
                        self.oscillation_history[block_pos] = deque(maxlen=self.oscillation_history_max_len)
                    self.oscillation_history[block_pos].append({'dx': dx, 'dy': dy, 'mag': mag})

        is_camera_motion = False
        median_dx, median_dy = (np.median([m['dx'] for m in block_motions]),
                                np.median([m['dy'] for m in block_motions])) if block_motions else (0, 0)
        if np.sqrt(median_dx ** 2 + median_dy ** 2) > 1.0:
            coherent_blocks = 0
            vec_median = np.array([median_dx, median_dy])
            for motion in block_motions:
                vec_block = np.array([motion['dx'], motion['dy']])
                if np.linalg.norm(vec_block) > 0.5:
                    cosine_sim = np.dot(vec_block, vec_median) / (np.linalg.norm(vec_block) * np.linalg.norm(vec_median) + 1e-6)
                    if cosine_sim > 0.8:
                        coherent_blocks += 1
            if block_motions and (coherent_blocks / len(block_motions)) > 0.85:
                is_camera_motion = True

        active_blocks = []
        if not is_camera_motion:
            # --- Cohesion Analysis to find natural motion clusters ---
            candidate_blocks = []
            for pos, history in self.oscillation_history.items():
                if len(history) < self.oscillation_history_max_len * 0.8: continue  # c9e6fbd original: 80%
                mags, dys, dxs = [h['mag'] for h in history], [h['dy'] for h in history], [h['dx'] for h in history]
                mean_mag, std_dev_dy = np.mean(mags), np.std(dys)
                if mean_mag < 0.5 or (mean_mag > 0 and std_dev_dy / mean_mag < 0.5):  # c9e6fbd original thresholds
                    continue  # --- Filter non-oscillating motion

                smoothed_dys = np.convolve(dys, np.ones(5) / 5, mode='valid')
                if len(smoothed_dys) < 2: continue
                freq = (len(np.where(np.diff(np.sign(smoothed_dys)))[0]) / 2) / self.oscillation_history_seconds

                # --- Adaptive Frequency Weighting (bell curve centered at 2.5Hz) ---
                if 0.5 <= freq <= 7.0:  # c9e6fbd original frequency range
                    freq_weight = np.exp(-((freq - 2.5) ** 2) / (2 * (1.5 ** 2)))  # Gaussian weight
                    score = mean_mag * freq * freq_weight
                    candidate_blocks.append({'pos': pos, 'score': score, 'dy': history[-1]['dy'], 'dx': history[-1]['dx'], 'mag': history[-1]['mag']})

            if candidate_blocks:
                candidate_pos = {b['pos'] for b in candidate_blocks}
                for block in candidate_blocks:
                    r, c = block['pos']
                    cohesion_boost = 1.0
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0: continue
                            if (r + dr, c + dc) in candidate_pos:
                                cohesion_boost += 0.2  # Boost score by 20% for each active neighbor
                    block['score'] *= cohesion_boost

                max_score = max(b['score'] for b in candidate_blocks)
                active_blocks = [b for b in candidate_blocks if b['score'] > max_score * 0.6]  # c9e6fbd original: 60% threshold

        if active_blocks:
            total_weight = sum(b['score'] for b in active_blocks)
            final_dy = sum(b['dy'] * b['score'] for b in active_blocks) / total_weight
            final_dx = sum(b['dx'] * b['score'] for b in active_blocks) / total_weight  # Add dx calculation
            final_mag = sum(b['mag'] * b['score'] for b in active_blocks) / total_weight

            amplitude_scaler = np.clip(final_mag / 4.0, 0.7, 1.5)
            max_deviation = 50 * amplitude_scaler
            scaled_dy = np.clip(final_dy * -10, -max_deviation, max_deviation)
            scaled_dx = np.clip(final_dx * 10, -max_deviation, max_deviation)  # Add dx scaling
            new_raw_primary_pos = np.clip(50 + scaled_dy, 0, 100)
            new_raw_secondary_pos = np.clip(50 + scaled_dx, 0, 100)  # Add secondary calculation
        else:
            new_raw_primary_pos = self.oscillation_last_known_pos * 0.95 + 50 * 0.05
            new_raw_secondary_pos = self.oscillation_last_known_secondary_pos * 0.95 + 50 * 0.05  # Add secondary decay

        # --- Live Dynamic Amplification Stage ---
        self.oscillation_position_history.append(new_raw_primary_pos)
        final_primary_pos = new_raw_primary_pos
        final_secondary_pos = new_raw_secondary_pos  # No amplification for secondary in c9e6fbd

        if self.live_amp_enabled and len(self.oscillation_position_history) > self.oscillation_position_history.maxlen * 0.5:
            p10 = np.percentile(self.oscillation_position_history, 10)
            p90 = np.percentile(self.oscillation_position_history, 90)
            effective_range = p90 - p10

            if effective_range > 15: # c9e6fbd original amplification threshold
                normalized_pos = (new_raw_primary_pos - p10) / effective_range
                final_primary_pos = np.clip(normalized_pos * 100, 0, 100)

        # Apply EMA smoothing to the final calculated positions
        self.oscillation_last_known_pos = self.oscillation_last_known_pos * (1 - self.oscillation_ema_alpha) + final_primary_pos * self.oscillation_ema_alpha
        self.oscillation_last_known_secondary_pos = self.oscillation_last_known_secondary_pos * (1 - self.oscillation_ema_alpha) + final_secondary_pos * self.oscillation_ema_alpha
        self.oscillation_funscript_pos = int(round(self.oscillation_last_known_pos))
        self.oscillation_funscript_secondary_pos = int(round(self.oscillation_last_known_secondary_pos))

        if self.tracking_active:
            # Use the same action logging logic as the current oscillation detector
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

        # --- Visualization only for active video area cells ---
        active_block_positions = {b['pos'] for b in active_blocks}
        for motion in block_motions:  # Loop through visible blocks only
            r, c = motion['pos']
            x1, y1 = c * self.oscillation_block_size, r * self.oscillation_block_size
            x2, y2 = x1 + self.oscillation_block_size, y1 + self.oscillation_block_size
            color = (100, 100, 100)
            if is_camera_motion:
                color = (0, 165, 255)
            elif (r, c) in active_block_positions:
                color = (0, 255, 0)
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 1)

        self.prev_gray_oscillation = current_gray
        return processed_frame, action_log_list if action_log_list else None

    def cleanup(self):
        """Explicit cleanup method for resource management."""
        try:
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
