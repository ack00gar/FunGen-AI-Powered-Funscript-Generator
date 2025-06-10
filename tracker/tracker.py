import cv2
from collections import Counter
import numpy as np
import time
import platform
from typing import List, Dict, Tuple, Optional, Any
from ultralytics import YOLO
import logging

from funscript.dual_axis_funscript import DualAxisFunscript
from config import constants


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
                 inversion_detection_split_ratio: float = constants.INVERSION_DETECTION_SPLIT_RATIO
                 ):
        self.app = app_logic_instance # Can be None if instantiated by Stage 3

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

        # Object Detection Model (YOLO)
        # For Stage 3, this might not be strictly needed if ROI comes from ATR,
        # but ROITracker init expects it. Can be a path to a dummy/small model if not used.
        self.det_model_path = tracker_model_path # Store path
        try:
            self.yolo = YOLO(tracker_model_path, task='detect')
            self.classes = self.yolo.names
            self.logger.info(f"Object detection model loaded from {tracker_model_path}")
            self.logger.info(f"Available classes: {self.classes}")
        except Exception as e:
            self.logger.error(f"Could not load YOLO object model from {tracker_model_path}: {e}")
            self.yolo = None
            self.classes = []

        # Pose Estimation Model (YOLO Pose)
        self.pose_model_path = pose_model_path # Store path
        if pose_model_path:
            try:
                self.yolo_pose = YOLO(pose_model_path)
                self.logger.info(f"Pose estimation model loaded from {pose_model_path}")
            except Exception as e:
                self.logger.warning(f"Could not load YOLO pose model from {pose_model_path}: {e}")
                self.yolo_pose = None
        else:
            self.yolo_pose = None

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

        # Ensure these are initialized regardless of self.app
        self.output_delay_frames: int = self.app.tracker.output_delay_frames if self.app and hasattr(self.app, 'tracker') else 0
        self.current_video_fps_for_delay: float = self.app.tracker.current_video_fps_for_delay if self.app and hasattr(self.app, 'tracker') else 30.0
        # Sensitivity and offsets should also be settable or taken from defaults if no app
        self.y_offset = self.app.tracker.y_offset if self.app and hasattr(self.app,
                                                                          'tracker') else constants.DEFAULT_LIVE_TRACKER_Y_OFFSET
        self.x_offset = self.app.tracker.x_offset if self.app and hasattr(self.app,
                                                                          'tracker') else constants.DEFAULT_LIVE_TRACKER_X_OFFSET
        self.sensitivity = self.app.tracker.sensitivity if self.app and hasattr(self.app,
                                                                                'tracker') else constants.DEFAULT_LIVE_TRACKER_SENSITIVITY

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
        self.funscript = DualAxisFunscript(logger=self.logger) # For live tracking
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
        self.main_interaction_class: Optional[str] = None # This will be set by Stage 3 processor per segment
        self.CLASS_PRIORITY = {"pussy": 0, "anus": 0, "butt": 1, "face": 2, "hand": 3, "breast": 4, "foot": 5}
        self.CLASS_COLORS = {
            "pussy": (255, 0, 255), "anus": (255, 0, 255), "butt": (255, 165, 0), "face": (0, 255, 255),
            "hand": (0, 0, 255), "breast": (255, 192, 203), "foot": (139, 69, 19), "penis": (0, 255, 0)
        }
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

        self.last_frame_time_sec_fps: Optional[float] = None
        self.current_fps: float = 0.0
        self.current_effective_amp_factor: float = self.base_amplification_factor # Initial value
        self.stats_display: List[str] = []
        self.logger.info(
            f"Tracker fully initialized (ROI Persistence: {self.max_frames_for_roi_persistence} frames, ROI Smoothing: {self.roi_smoothing_factor}). App instance {'provided' if self.app else 'not provided (e.g. S3 mode)'}.")

    def _calculate_flow_in_sub_regions(self, patch_gray: np.ndarray, prev_patch_gray: Optional[np.ndarray]) \
            -> Tuple[float, float, float, float, Optional[np.ndarray]]:
        """
        Calculates optical flow for the entire patch. It returns the overall
        median flow and the motion magnitude in the lower nth of the patch (for thrusting)
        versus the upper rest of the patch (for riding). The split is controlled by
        `self.inversion_detection_split_ratio`.

        Returns:
            A tuple of (overall_dy, overall_dx, lower_region_magnitude, upper_region_magnitude, flow_field)
        """
        # 1. Handle edge cases where flow cannot be calculated
        if self.flow_dense is None or prev_patch_gray is None or prev_patch_gray.shape != patch_gray.shape:
            return 0.0, 0.0, 0.0, 0.0, None

        # Ensure arrays are C-continuous before passing to OpenCV
        prev_patch_cont = np.ascontiguousarray(prev_patch_gray)
        patch_cont = np.ascontiguousarray(patch_gray)

        # 2. Calculate optical flow for the entire patch ONCE
        flow = self.flow_dense.calc(prev_patch_cont, patch_cont, None)
        if flow is None:
            return 0.0, 0.0, 0.0, 0.0, None

        # 3. Get the overall median flow vector (this is our main signal)
        overall_dy = np.median(flow[..., 1])
        overall_dx = np.median(flow[..., 0])

        # 4. Analyze the lower nth vs the upper rest of the flow field for motion magnitude
        h, _, _ = flow.shape
        lower_magnitude = 0.0
        upper_magnitude = 0.0

        # Ensure the ratio is valid to prevent division by zero or illogical splits.
        if self.inversion_detection_split_ratio > 1.0:
            # Calculate the height of the lower region based on the ratio.
            # e.g., if ratio is 3, lower_region_h is one-third of the total height.
            lower_region_h = int(h / self.inversion_detection_split_ratio)

            # Check if the calculated height is a meaningful split of the patch
            if lower_region_h > 0 and lower_region_h < h:
                # The upper region is the "rest" of the patch, from the top to the start of the lower region.
                upper_region_flow_vertical = flow[0:h - lower_region_h, :, 1]
                # The lower region is the bottom 'nth' of the patch.
                lower_region_flow_vertical = flow[h - lower_region_h:h, :, 1]

                # Calculate magnitude as the median of the absolute vertical movements in each region.
                upper_magnitude = np.median(np.abs(upper_region_flow_vertical))
                lower_magnitude = np.median(np.abs(lower_region_flow_vertical))

        return overall_dy, overall_dx, lower_magnitude, upper_magnitude, flow


    def set_tracking_mode(self, mode: str):
        if mode in ["YOLO_ROI", "USER_FIXED_ROI"]:
            if self.tracking_mode != mode:
                self.tracking_mode = mode
                self.logger.info(f"Tracker mode set to: {self.tracking_mode}")
                if self.tracking_active:
                    self.stop_tracking()
                if mode == "YOLO_ROI":
                    self.clear_user_defined_roi_and_point()
                elif mode == "USER_FIXED_ROI":
                    self.roi = None
                    self.penis_last_known_box = None
                    self.main_interaction_class = None
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
            self.prev_gray_user_roi_patch = None  # CRITICAL: This forces recapture on the first frame of the chapter
            self.logger.info(f"Tracker reconfigured for chapter {chapter.unique_id[:8]}.")
        else:
            # If the chapter has no ROI, switch to a safe, inactive state
            self.set_tracking_mode("YOLO_ROI")  # A mode that does nothing without a target
            self.roi = None
            if self.tracking_active:
                self.stop_tracking()  # Stop generating actions in gaps/unconfigured chapters
            self.logger.info(f"Chapter {chapter.unique_id[:8]} has no ROI. Tracker is inactive.")

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
        return cv2.copyMakeBorder(frame_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        detections = []
        if not self.yolo: return detections
        # Determine discarded classes based on self.app context if available
        discarded_classes_runtime = []
        if self.app and hasattr(self.app, 'discarded_tracking_classes'):
            discarded_classes_runtime = self.app.discarded_tracking_classes

        results = self.yolo(frame, device=constants.DEVICE,  verbose=False, conf=self.confidence_threshold)
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.classes[class_id]
                if class_name in discarded_classes_runtime: # Use runtime list
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append({
                    "box": (x1, y1, x2 - x1, y2 - y1),
                    "class_id": class_id, "class_name": class_name, "confidence": conf
                })
        return detections

    def _update_penis_tracking(self, penis_box_xywh: Tuple[int, int, int, int]):
        self.penis_last_known_box = penis_box_xywh
        penis_size = penis_box_xywh[2] * penis_box_xywh[3]
        self.penis_max_size_history.append(penis_size)
        if len(self.penis_max_size_history) > self.penis_size_history_window:
            self.penis_max_size_history.pop(0)

    def _find_interacting_objects(self, penis_box_xywh: Tuple[int, int, int, int],
                                  all_detections: List[Dict]) -> List[Dict]:
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

    def _calculate_combined_roi(self, frame_shape: Tuple[int, int], penis_box_xywh: Tuple[int, int, int, int],
                                interacting_objects: List[Dict]) -> Tuple[int, int, int, int]:
        entities = [penis_box_xywh] + [obj["box"] for obj in interacting_objects]
        min_x, min_y = min(e[0] for e in entities), min(e[1] for e in entities)
        max_x_coord, max_y_coord = max(e[0] + e[2] for e in entities), max(e[1] + e[3] for e in entities)
        rx1, ry1 = max(0, min_x - self.roi_padding), max(0, min_y - self.roi_padding)
        rx2, ry2 = min(frame_shape[1], max_x_coord + self.roi_padding), min(frame_shape[0],
                                                                            max_y_coord + self.roi_padding)
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

    def _calculate_flow_in_patch(self, patch_gray: np.ndarray, prev_patch_gray: Optional[np.ndarray],
                                 use_sparse: bool = False, prev_features_for_sparse: Optional[np.ndarray] = None) \
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
                next_feat, status, _ = cv2.calcOpticalFlowPyrLK(prev_cont, curr_cont, prev_features_for_sparse, None,
                                                                **self.lk_params)
                good_prev = prev_features_for_sparse[status == 1]
                good_next = next_feat[status == 1] if next_feat is not None else np.array([]) # Ensure good_next is an array

                if len(good_prev) > 0 and len(good_next) > 0 and good_next.ndim == 2 and good_next.shape[1] == 2: # Check shape
                    dx, dy = np.median(good_next[:, 0] - good_prev[:, 0]), np.median(good_next[:, 1] - good_prev[:, 1])
                    updated_sparse = good_next.reshape(-1, 1, 2)
                else:
                    updated_sparse = cv2.goodFeaturesToTrack(curr_cont, mask=None,
                                                             **self.feature_params) if curr_cont.size > 0 else None
            else:
                updated_sparse = cv2.goodFeaturesToTrack(curr_cont, mask=None,
                                                         **self.feature_params) if curr_cont.size > 0 else None
        elif self.flow_dense:
            flow = self.flow_dense.calc(prev_cont, curr_cont, None)
            if flow is not None:
                dx, dy, flow_vis = np.median(flow[..., 0]), np.median(flow[..., 1]), flow
        return dx, dy, flow_vis, updated_sparse

    def _apply_adaptive_scaling(self, value: float, min_val_attr: str, max_val_attr: str, size_factor: float,
                                is_primary: bool) -> int:
        min_h, max_h = getattr(self, min_val_attr), getattr(self, max_val_attr)
        setattr(self, min_val_attr, min(min_h * 0.995, value * 0.9 if value < -0.1 else value * 1.1))
        setattr(self, max_val_attr, max(max_h * 0.995, value * 1.1 if value > 0.1 else value * 0.9))
        min_h, max_h = min(getattr(self, min_val_attr), -0.2), max(getattr(self, max_val_attr), 0.2)
        flow_range = max_h - min_h
        if abs(flow_range) < 0.1: flow_range = np.sign(flow_range) * 0.1 if flow_range != 0 else 0.1
        normalized_centered_flow = (2 * (value - min_h) / flow_range) - 1.0 if flow_range != 0 else 0.0
        normalized_centered_flow = np.clip(normalized_centered_flow, -1.0, 1.0)
        effective_amp_factor = self._get_effective_amplification_factor()
        max_deviation = (self.sensitivity / 2.0) * effective_amp_factor
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
                prev_features_for_sparse=prev_sparse_features
            )
        else:
            # Use our sub-region analysis method for dense flow
            # Ensure _calculate_flow_in_sub_regions returns flow_field_for_vis if you intend to use it
            dy_raw, dx_raw, lower_mag, upper_mag, flow_field_for_vis = self._calculate_flow_in_sub_regions(
                current_roi_patch_gray, prev_roi_patch_gray
            )

        # Check if the feature is enabled AND if the video processor has identified the video as VR.

        is_vr_video = self.app and hasattr(self.app, 'processor') and self.app.processor.determined_video_type == 'VR'

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

            if self.motion_mode_history:  # Check if history is not empty before processing
                most_common = Counter(self.motion_mode_history).most_common(1)[0]
                if most_common[1] > self.motion_mode_history_window // 2:
                    if self.motion_mode != most_common[0]:
                        self.motion_mode = most_common[0]
                        self.logger.info(f"Motion mode changed to: {self.motion_mode}")
        else:
            # If the feature is disabled or the video is 2D, ensure we are in the default, non-inverting state.
            self.motion_mode = 'thrusting'  # Default non-inverting mode

        # Smooth the raw dx/dy values from the overall flow
        self.primary_flow_history_smooth.append(dy_raw)
        self.secondary_flow_history_smooth.append(dx_raw)  # FIX: Append dx_raw here

        if len(self.primary_flow_history_smooth) > self.flow_history_window_smooth:
            self.primary_flow_history_smooth.pop(0)
        if len(self.secondary_flow_history_smooth) > self.flow_history_window_smooth:
            self.secondary_flow_history_smooth.pop(0)

        dy_smooth = np.median(self.primary_flow_history_smooth) if self.primary_flow_history_smooth else dy_raw
        dx_smooth = np.median(self.secondary_flow_history_smooth) if self.secondary_flow_history_smooth else dx_raw

        # Calculate the base positions before potential inversion
        size_factor = self.get_current_penis_size_factor()
        if self.adaptive_flow_scale:
            base_primary_pos = self._apply_adaptive_scaling(dy_smooth, "flow_min_primary_adaptive",
                                                            "flow_max_primary_adaptive", size_factor, True)
            secondary_pos = self._apply_adaptive_scaling(dx_smooth, "flow_min_secondary_adaptive",
                                                         "flow_max_secondary_adaptive", size_factor, False)
        else:
            effective_amp_factor = self._get_effective_amplification_factor()
            manual_scale_multiplier = (self.sensitivity / 10.0) * (1.0 / size_factor) * effective_amp_factor
            base_primary_pos = int(np.clip(50 + dy_smooth * manual_scale_multiplier + self.y_offset, 0, 100))
            secondary_pos = int(np.clip(50 + dx_smooth * manual_scale_multiplier + self.x_offset, 0, 100))

        # Apply inversion based on the detected mode (which is now correctly handled for non-VR video)
        primary_pos = 100 - base_primary_pos
        if self.motion_mode == 'thrusting':
            #primary_pos = 100 - base_primary_pos
            primary_pos = base_primary_pos

        # Visualization logic (only if self.app is present, for live mode)
        if self.app and self.roi and self.show_flow:  # Removed redundant processed_frame_draw_target is not None
            rx, ry, rw, rh = self.roi
            # Ensure ROI is within bounds of the frame meant for drawing
            if ry + rh <= processed_frame_draw_target.shape[0] and rx + rw <= processed_frame_draw_target.shape[1]:
                roi_display_patch = processed_frame_draw_target[ry:ry + rh, rx:rx + rw]
                if roi_display_patch.size > 0:  # Check if patch is not empty
                    if flow_field_for_vis is not None and self.flow_dense:
                        try:
                            # Ensure flow_field_for_vis has the correct shape for cv2.cartToPolar
                            # It should be a 2-channel array (dx, dy)
                            if flow_field_for_vis.shape[-1] == 2:
                                mag, ang = cv2.cartToPolar(flow_field_for_vis[..., 0], flow_field_for_vis[..., 1])
                                hsv = np.zeros_like(roi_display_patch)
                                hsv[..., 1] = 255
                                hsv[..., 0] = ang * 180 / np.pi / 2
                                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                                vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                                # Ensure vis and roi_display_patch have same data type and shape for addWeighted
                                processed_frame_draw_target[ry:ry + rh, rx:rx + rw] = cv2.addWeighted(roi_display_patch,
                                                                                                     0.5, vis.astype(roi_display_patch.dtype), 0.5, 0)
                            else:
                                self.logger.warning(
                                    f"flow_field_for_vis has unexpected shape: {flow_field_for_vis.shape}. Expected (H, W, 2).")

                        except cv2.error as e:
                            self.logger.error(f"Flow vis error: {e}")
                    if self.use_sparse_flow and updated_sparse_features_out is not None and self.show_tracking_points:
                        # Draw sparse features relative to the ROI patch
                        for pt in updated_sparse_features_out:
                            x, y = pt.ravel()
                            # Check if point is within the bounds of the roi_display_patch before drawing
                            if 0 <= x < rw and 0 <= y < rh:
                                cv2.circle(roi_display_patch, (int(x), int(y)), 2, (0, 255, 255), -1)
                            else:
                                self.logger.warning(f"Sparse feature point ({x}, {y}) out of bounds for ROI patch.")

                    # Draw arrow relative to the ROI patch
                    cx, cy = rw // 2, rh // 2
                    # Ensure the arrow endpoints are within the patch bounds
                    arrow_end_x = int(cx + dx_smooth * 5)
                    arrow_end_y = int(cy + dy_smooth * 5)
                    # Clip arrow coordinates to stay within roi_display_patch
                    arrow_end_x = np.clip(arrow_end_x, 0, rw - 1)
                    arrow_end_y = np.clip(arrow_end_y, 0, rh - 1)
                    cv2.arrowedLine(roi_display_patch, (cx, cy), (arrow_end_x, arrow_end_y), (0, 0, 255), 1)
            else:
                self.logger.warning(
                    f"ROI {self.roi} out of bounds for drawing on frame shape {processed_frame_draw_target.shape}")

        return primary_pos, secondary_pos, dy_smooth, dx_smooth, updated_sparse_features_out


    def process_frame(self, frame: np.ndarray, frame_time_ms: int, frame_index: Optional[int] = None,
                      min_write_frame_id: Optional[int] = None) \
            -> Tuple[np.ndarray, Optional[List[Dict]]]:
        self._update_fps()
        processed_frame = self.preprocess_frame(frame)
        current_frame_gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        final_primary_pos, final_secondary_pos = 50, 50
        action_log_list = []
        detected_objects_this_frame: List[Dict] = [] # Initialize for YOLO_ROI mode

        self.current_effective_amp_factor = self._get_effective_amplification_factor()

        if self.tracking_mode == "YOLO_ROI":
            run_detection_this_frame = (self.internal_frame_counter % self.roi_update_interval == 0) or \
                                       (self.roi is None) or \
                                       (not self.penis_last_known_box and self.frames_since_target_lost < self.max_frames_for_roi_persistence and \
                                        self.internal_frame_counter % max(1, self.roi_update_interval // 3) == 0)

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
                main_roi_patch_gray = current_frame_gray[ry:min(ry + rh, current_frame_gray.shape[0]),
                                      rx:min(rx + rw, current_frame_gray.shape[1])]
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
                urw_c, urh_c = min(urw, current_frame_gray.shape[1] - urx_c), min(urh,
                                                                                  current_frame_gray.shape[0] - ury_c)

                if urw_c > 0 and urh_c > 0:
                    current_user_roi_patch_gray = current_frame_gray[ury_c: ury_c + urh_c, urx_c: urx_c + urw_c]

                    # We must temporarily set self.roi to the user's fixed ROI so that visualization
                    # inside process_main_roi_content works correctly (e.g., drawing the flow arrow).
                    original_yolo_roi = self.roi
                    self.roi = (urx_c, ury_c, urw_c, urh_c)

                    # This call now handles everything: sparse/dense flow, smoothing, adaptive scaling, and inversion.
                    final_primary_pos, final_secondary_pos, dy_smooth, dx_smooth, _ = \
                        self.process_main_roi_content(
                            processed_frame,
                            current_user_roi_patch_gray,
                            self.prev_gray_user_roi_patch,
                            None  # This mode doesn't use/manage sparse features
                        )

                    # Restore the original YOLO_ROI to not disrupt its logic.
                    self.roi = original_yolo_roi

                    self.user_roi_current_flow_vector = (dx_smooth, dy_smooth)

                    if self.user_roi_tracked_point_relative:
                        prev_x_rel, prev_y_rel = self.user_roi_tracked_point_relative

                        # Add the smoothed flow vector to the previous position
                        # dx_smooth is horizontal motion, dy_smooth is vertical motion
                        new_x_rel = prev_x_rel + dx_smooth
                        new_y_rel = prev_y_rel + dy_smooth

                        # Clamp the new coordinates to ensure the point stays within the ROI boundaries
                        clamped_x_rel = max(0.0, min(new_x_rel, float(urw_c)))
                        clamped_y_rel = max(0.0, min(new_y_rel, float(urh_c)))

                        # Update the state with the new, clamped position for the next frame
                        self.user_roi_tracked_point_relative = (clamped_x_rel, clamped_y_rel)

                    # Update the previous patch for the next frame's calculation.
                    self.prev_gray_user_roi_patch = np.ascontiguousarray(current_user_roi_patch_gray)

                else:
                    self.prev_gray_user_roi_patch = None
                    final_primary_pos, final_secondary_pos = 50, 50
                    self.user_roi_current_flow_vector = (0.0, 0.0)
            else:
                self.prev_gray_user_roi_patch = None
                final_primary_pos, final_secondary_pos = 50, 50
                self.user_roi_current_flow_vector = (0.0, 0.0)

        if self.app and self.tracking_active and \
           (min_write_frame_id is None or (frame_index is not None and frame_index >= min_write_frame_id)):
            delay_ms = (self.output_delay_frames / self.current_video_fps_for_delay) * 1000.0 if self.current_video_fps_for_delay > 0 else 0.0
            adjusted_frame_time_ms = frame_time_ms - delay_ms
            current_tracking_axis_mode = self.app.tracking_axis_mode
            current_single_axis_output = self.app.single_axis_output_target
            primary_to_write, secondary_to_write = None, None

            if current_tracking_axis_mode == "both":
                primary_to_write, secondary_to_write = final_primary_pos, final_secondary_pos
            elif current_tracking_axis_mode == "vertical":
                if current_single_axis_output == "primary": primary_to_write = final_primary_pos
                else: secondary_to_write = final_primary_pos
            elif current_tracking_axis_mode == "horizontal":
                if current_single_axis_output == "primary": primary_to_write = final_secondary_pos
                else: secondary_to_write = final_secondary_pos

            # Determine if this is a file processing context (like "track in chapter") vs. pure live tracking.
            # The presence of `frame_index` is a strong indicator of file processing.
            # When `is_from_live_tracker` is False, the `max_history` limit in DualAxisFunscript is bypassed.
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
                "mode": current_tracking_axis_mode, "target": current_single_axis_output if current_tracking_axis_mode != "both" else "N/A",
                "raw_at": frame_time_ms, "delay_applied_ms": delay_ms,
                "roi_main": self.roi if self.tracking_mode == "YOLO_ROI" else self.user_roi_fixed,
                "amp": self.current_effective_amp_factor
            })

        if self.show_masks and detected_objects_this_frame and self.tracking_mode == "YOLO_ROI":
            self.draw_detections(processed_frame, detected_objects_this_frame)

        if self.tracking_mode == "YOLO_ROI" and self.show_roi and self.roi:
            rx, ry, rw, rh = self.roi
            color = self.get_class_color(self.main_interaction_class or ("penis" if self.penis_last_known_box else "persisting"))
            cv2.rectangle(processed_frame, (rx, ry), (rx + rw, ry + rh), color, 1)
            status_text = self.main_interaction_class or ('P' if self.penis_last_known_box else 'Lost...')
            cv2.putText(processed_frame, f"ROI:{status_text}", (rx, ry - 2), cv2.FONT_HERSHEY_PLAIN, 0.7, color, 1)
            if not self.penis_last_known_box:
                cv2.putText(processed_frame, f"Lost: {self.frames_since_target_lost}/{self.max_frames_for_roi_persistence}",
                            (rx, ry + rh + 10), cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 0, 255), 1)

            # Add motion mode indicator text for VR videos
            is_vr_video = self.app and hasattr(self.app, 'processor') and self.app.processor.determined_video_type == 'VR'
            if self.enable_inversion_detection and is_vr_video:
                mode_text = self.motion_mode.capitalize()
                mode_color = (0, 255, 0) if self.motion_mode == 'thrusting' else (255, 100, 255) if self.motion_mode == 'riding' else (255, 255, 0)
                cv2.putText(processed_frame, mode_text, (rx + 5, ry + rh - 5), cv2.FONT_HERSHEY_PLAIN, 0.8, mode_color, 1)

        # The logic is simplified to only draw the box, as the motion arrow is now
        # handled by the shared process_main_roi_content function.
        elif self.tracking_mode == "USER_FIXED_ROI" and self.show_roi and self.user_roi_fixed:
            urx, ury, urw, urh = self.user_roi_fixed
            urx_c, ury_c = max(0, urx), max(0, ury)
            urw_c, urh_c = min(urw, processed_frame.shape[1] - urx_c), min(urh, processed_frame.shape[0] - ury_c)
            cv2.rectangle(processed_frame, (urx_c, ury_c), (urx_c + urw_c, ury_c + urh_c), (0, 255, 255), 2)

            # Draw the tracked point to visualize its movement
            if self.user_roi_tracked_point_relative:
                point_x_abs = urx_c + int(self.user_roi_tracked_point_relative[0])
                point_y_abs = ury_c + int(self.user_roi_tracked_point_relative[1])
                cv2.circle(processed_frame, (point_x_abs, point_y_abs), 3, (0, 255, 0), -1)

            # Add motion mode indicator text for VR videos
            is_vr_video = self.app and hasattr(self.app, 'processor') and self.app.processor.determined_video_type == 'VR'
            if self.enable_inversion_detection and is_vr_video:
                mode_text = self.motion_mode.capitalize()
                mode_color = (0, 255, 0) if self.motion_mode == 'thrusting' else (255, 100, 255) if self.motion_mode == 'riding' else (255, 255, 0)
                cv2.putText(processed_frame, mode_text, (urx_c + 5, ury_c + urh_c - 5), cv2.FONT_HERSHEY_PLAIN, 0.8, mode_color, 1)


        if self.show_stats:
            for i, stat_text in enumerate(self.stats_display):
                cv2.putText(processed_frame, stat_text, (5, 15 + i * 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 220), 1)

        self.internal_frame_counter += 1
        return processed_frame, action_log_list if action_log_list else None

    # process_frame_for_stage3 is REMOVED from ROITracker.
    # Its logic will be adapted into the new stage_3_optical_flow_processor.py

    def get_class_color(self, class_name: Optional[str]) -> Tuple[int, int, int]:
        return self.CLASS_COLORS.get(class_name.lower() if class_name else "", (180, 180, 180))

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
        if self.tracking_mode == "YOLO_ROI": # Also for S3-like
            self.prev_gray_main_roi, self.prev_features_main_roi = None, None
        self.logger.info(f"Tracking stopped (mode: {self.tracking_mode}).")

    def reset(self, reason: Optional[str] = None):
        self.clear_user_defined_roi_and_point()
        self.set_tracking_mode("YOLO_ROI") # Default mode on full reset

        # self.roi is cleared by start_tracking if mode is YOLO_ROI
        self.start_tracking() # Re-initializes states based on the current mode (now YOLO_ROI)

        if self.funscript: # This funscript is for live tracking
            if reason != "seek" and reason != "project_load_preserve_actions":
                self.funscript.clear()
                self.logger.info(f"Live tracker Funscript cleared (reason: {reason}).")
            else:
                self.logger.info(f"Live tracker Funscript preserved (reason: {reason}).")
        self.logger.info(f"Tracker reset complete (reason: {reason}). Tracking is now {self.tracking_active}.")