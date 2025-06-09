import os
import time
import logging
import numpy as np
import cv2 # Ensure OpenCV is available
from typing import Optional, List, Dict, Any, Tuple

from funscript.dual_axis_funscript import DualAxisFunscript
from video.video_processor import VideoProcessor
from tracker.tracker import ROITracker # ROITracker will be instantiated here
from detection.cd.stage_2_cd import FrameObject, ATRSegment
from config.constants import *

# Helper to avoid NameError if constants are not directly in tracker anymore


class Stage3OpticalFlowProcessor:
    def __init__(self,
                 video_path: str,
                 atr_segments_list: List[ATRSegment],
                 s2_frame_objects_map: Dict[int, FrameObject],
                 tracker_config: Dict[str, Any],
                 common_app_config: Dict[str, Any],
                 progress_callback: callable,
                 stop_event: Any, # threading.Event or multiprocessing.Event
                 parent_logger: logging.Logger):

        self.video_path = video_path
        self.atr_segments = atr_segments_list
        self.s2_frame_objects_map = s2_frame_objects_map
        self.tracker_config = tracker_config
        self.common_app_config = common_app_config
        self.progress_callback = progress_callback
        self.stop_event = stop_event
        self.logger = parent_logger.getChild("S3_OF_Processor")

        self.video_processor: Optional[VideoProcessor] = None
        self.roi_tracker_instance: Optional[ROITracker] = None
        self.funscript = DualAxisFunscript(logger=self.logger)

        self.current_fps = 0.0
        self.last_frame_time_sec_fps: Optional[float] = None

    def _update_fps(self):
        current_time_sec = time.time()
        if self.last_frame_time_sec_fps is not None:
            delta_time = current_time_sec - self.last_frame_time_sec_fps
            if delta_time > 0.001: # avoid division by zero or too small dt
                self.current_fps = 1.0 / delta_time
        self.last_frame_time_sec_fps = current_time_sec


    def _initialize_dependencies(self) -> bool:
        # Initialize VideoProcessor
        # Create a dummy app instance for VideoProcessor if it expects one for logger
        class VPAppProxy:
            pass
        vp_app_proxy = VPAppProxy()
        vp_app_proxy.logger = self.logger.getChild("VideoProcessor_S3")
        vp_app_proxy.hardware_acceleration_method = self.common_app_config.get("hardware_acceleration_method", "none")

        # Retrieve the list from the config and add it to the proxy
        vp_app_proxy.available_ffmpeg_hwaccels = self.common_app_config.get("available_ffmpeg_hwaccels", [])

        self.video_processor = VideoProcessor(
            app_instance=vp_app_proxy, # Provides logger
            tracker=None, # Tracker not needed by VP for frame fetching
            yolo_input_size=self.common_app_config.get('yolo_input_size', 640),
            video_type=self.common_app_config.get('video_type', 'auto'),
            vr_input_format=self.common_app_config.get('vr_input_format', 'he'),
            vr_fov=self.common_app_config.get('vr_fov', 190),
            vr_pitch=self.common_app_config.get('vr_pitch', 0)
        )
        if not self.video_processor.open_video(self.video_path):
            self.logger.error(f"S3 OF: VideoProcessor could not open video: {self.video_path}")
            return False

        # Initialize ROITracker instance for S3 processing
        # ROITracker's __init__ will need to handle app_logic_instance=None
        # by taking all necessary configs directly.
        try:
            self.roi_tracker_instance = ROITracker(
                app_logic_instance=None, # Explicitly None for S3
                tracker_model_path=self.common_app_config.get('yolo_det_model_path', ''), # Not used for S3 flow, but needed by init
                pose_model_path=self.common_app_config.get('yolo_pose_model_path', ''),   # Not used for S3 flow
                confidence_threshold=self.tracker_config.get('confidence_threshold', 0.4),
                roi_padding=self.tracker_config.get('roi_padding', 30),
                roi_update_interval=self.tracker_config.get('roi_update_interval', DEFAULT_ROI_UPDATE_INTERVAL),
                roi_smoothing_factor=self.tracker_config.get('roi_smoothing_factor', DEFAULT_ROI_SMOOTHING_FACTOR),
                dis_flow_preset=self.tracker_config.get('dis_flow_preset', "ULTRAFAST"),
                target_size_preprocess=self.tracker_config.get('target_size_preprocess', (640,640)),
                flow_history_window_smooth=self.tracker_config.get('flow_history_window_smooth', 3),
                adaptive_flow_scale=self.tracker_config.get('adaptive_flow_scale', True),
                use_sparse_flow=self.tracker_config.get('use_sparse_flow', False), # S3 typically uses dense
                max_frames_for_roi_persistence=self.tracker_config.get('max_frames_for_roi_persistence', DEFAULT_ROI_PERSISTENCE_FRAMES), # Not really used in S3 like in live
                base_amplification_factor=self.tracker_config.get('base_amplification_factor', DEFAULT_BASE_AMPLIFICATION),
                class_specific_amplification_multipliers=self.tracker_config.get('class_specific_amplification_multipliers', None),
                logger=self.logger.getChild("ROITracker_S3")
            )
            # Set parameters that might not be in __init__ or need override for S3
            self.roi_tracker_instance.y_offset = self.tracker_config.get('y_offset', DEFAULT_Y_OFFSET)
            self.roi_tracker_instance.x_offset = self.tracker_config.get('x_offset', DEFAULT_X_OFFSET)
            self.roi_tracker_instance.sensitivity = self.tracker_config.get('sensitivity', DEFAULT_SENSITIVITY)
            self.roi_tracker_instance.output_delay_frames = self.common_app_config.get('output_delay_frames', 0)
            self.roi_tracker_instance.current_video_fps_for_delay = self.common_app_config.get('video_fps', 30.0)
            self.roi_tracker_instance.tracking_mode = "YOLO_ROI" # S3 operates in a mode analogous to YOLO_ROI for ROI definition
            self.roi_tracker_instance.show_roi = self.common_app_config.get('s3_show_roi_debug', False) # For debug frames

        except Exception as e:
            self.logger.error(f"S3 OF: Failed to initialize ROITracker: {e}", exc_info=True)
            return False
        return True

    def process_segments(self) -> Dict[str, List[Dict[str, Any]]]:
        if not self._initialize_dependencies():
            return {"error": "Failed to initialize S3 OF dependencies."}

        s3_start_time = time.time()
        total_frames_processed_s3 = 0
        estimated_total_frames_s3 = sum(
            (seg.end_frame_id - max(0, seg.start_frame_id - self.common_app_config.get('num_warmup_frames_s3', 10)) + 1)
            for seg in self.atr_segments
        )
        if estimated_total_frames_s3 == 0 and self.atr_segments: # Handle case where segments might be empty or start=end
             estimated_total_frames_s3 = len(self.atr_segments) # At least 1 frame per segment for ETA

        # Make sure tracker is "started" for S3 processing context
        self.roi_tracker_instance.start_tracking() # Resets flow history, internal counters etc.

        for seg_idx, segment_obj in enumerate(self.atr_segments):
            if self.stop_event.is_set():
                self.logger.info("S3 OF: Stop event detected during segment processing.")
                return {"error": "S3 OF processing stopped by user."}

            # Skip "Not Relevant" or "Close Up" segments
            if segment_obj.major_position in ["Not Relevant", "Close Up"]:
                self.logger.info(f"S3 OF: Skipping segment {seg_idx + 1}/{len(self.atr_segments)} because its position is '{segment_obj.major_position}'.")
                continue


            segment_name_for_progress = f"{segment_obj.major_position} (F{segment_obj.start_frame_id}-{segment_obj.end_frame_id})"
            self.logger.info(f"S3 OF: Processing segment {seg_idx + 1}/{len(self.atr_segments)}: {segment_name_for_progress}")

            # Reset tracker's internal state for each new segment (it's done by start_tracking, but good to be explicit for flow logic)
            self.roi_tracker_instance.internal_frame_counter = 0 # Reset for ROI update interval logic within this segment
            self.roi_tracker_instance.prev_gray_main_roi = None
            self.roi_tracker_instance.prev_features_main_roi = None
            self.roi_tracker_instance.roi = None # Crucial: ROI must be redefined for each segment
            self.roi_tracker_instance.primary_flow_history_smooth.clear()
            self.roi_tracker_instance.secondary_flow_history_smooth.clear()


            # Set main interaction class for amplification for this segment
            self.roi_tracker_instance.main_interaction_class = segment_obj.major_position

            num_warmup_frames = self.common_app_config.get('num_warmup_frames_s3', 10)
            actual_processing_start_frame = max(0, segment_obj.start_frame_id - num_warmup_frames)
            actual_processing_end_frame = segment_obj.end_frame_id

            num_frames_in_actual_segment_for_progress = segment_obj.end_frame_id - segment_obj.start_frame_id + 1

            for frame_id_to_process in range(actual_processing_start_frame, actual_processing_end_frame + 1):
                if self.stop_event.is_set():
                    self.logger.info("S3 OF: Stop event detected during frame processing.")
                    return {"error": "S3 OF processing stopped by user."}

                self._update_fps() # Update FPS calculation for this module
                time_elapsed_s3 = time.time() - s3_start_time
                eta_s3 = ((estimated_total_frames_s3 - total_frames_processed_s3) / self.current_fps) if self.current_fps > 0 and total_frames_processed_s3 < estimated_total_frames_s3 else 0.0


                current_frame_image = self.video_processor._get_specific_frame(frame_id_to_process)
                if current_frame_image is None:
                    self.logger.warning(f"S3 OF: Could not retrieve frame {frame_id_to_process}. Skipping.")
                    continue

                processed_frame_for_tracker = self.roi_tracker_instance.preprocess_frame(current_frame_image)
                current_frame_gray = cv2.cvtColor(processed_frame_for_tracker, cv2.COLOR_BGR2GRAY)

                frame_time_ms = int(round((frame_id_to_process / self.common_app_config.get('video_fps', 30.0)) * 1000.0))
                frame_obj_s2 = self.s2_frame_objects_map.get(frame_id_to_process)

                if not frame_obj_s2:
                    self.logger.warning(f"S3 OF: S2 FrameObject not found for frame ID {frame_id_to_process}. ATR context might be limited for ROI.")
                    # Create a minimal dummy FrameObject if ROITracker absolutely needs one
                    # For S3, ROI logic relies heavily on atr_locked_penis_state from FrameObject
                    class MinimalFrameObject:
                        def __init__(self, fid):
                            self.frame_id = fid
                            self.atr_locked_penis_state = ATRLockedPenisState() # Default empty state
                            self.atr_detected_contact_boxes = []
                    frame_obj_s2 = MinimalFrameObject(frame_id_to_process)


                # --- ROI Definition Logic (adapted from ROITracker.process_frame_for_stage3) ---
                run_roi_definition_this_frame = False
                if self.roi_tracker_instance.roi is None:
                    run_roi_definition_this_frame = True
                elif self.roi_tracker_instance.roi_update_interval > 0 and \
                     (self.roi_tracker_instance.internal_frame_counter % self.roi_tracker_instance.roi_update_interval == 0):
                    run_roi_definition_this_frame = True

                if run_roi_definition_this_frame:
                    candidate_roi_xywh: Optional[Tuple[int, int, int, int]] = None
                    if frame_obj_s2.atr_locked_penis_state.active and frame_obj_s2.atr_locked_penis_state.box:
                        lp_box_coords_xyxy = frame_obj_s2.atr_locked_penis_state.box
                        lp_x1, lp_y1, lp_x2, lp_y2 = lp_box_coords_xyxy
                        current_penis_box_for_roi_calc = (lp_x1, lp_y1, lp_x2 - lp_x1, lp_y2 - lp_y1)

                        interacting_objects_for_roi_calc = []
                        # Determine relevant classes based on segment_obj.major_position (already in tracker.main_interaction_class)
                        # This logic needs to be here or passed to _calculate_combined_roi
                        # For now, _calculate_combined_roi takes all interacting objects
                        relevant_classes_for_pos = [] # Define based on segment_obj.major_position
                        # (Code from ROITracker.process_frame_for_stage3 for relevant_classes_for_pos)
                        if segment_obj.major_position == "Cowgirl / Missionary": relevant_classes_for_pos = ["pussy"]
                        elif segment_obj.major_position == "Rev. Cowgirl / Doggy": relevant_classes_for_pos = ["butt"]
                        elif segment_obj.major_position == "Handjob / Blowjob": relevant_classes_for_pos = ["face", "hand"]
                        elif segment_obj.major_position == "Boobjob": relevant_classes_for_pos = ["breast", "hand"]
                        elif segment_obj.major_position == "Footjob": relevant_classes_for_pos = ["foot"]


                        for contact_dict in frame_obj_s2.atr_detected_contact_boxes:
                            box_rec = contact_dict.get("box_rec")
                            if box_rec and contact_dict.get("class_name") in relevant_classes_for_pos:
                                interacting_objects_for_roi_calc.append({
                                    "box": (box_rec.x1, box_rec.y1, box_rec.width, box_rec.height),
                                    "class_name": box_rec.class_name
                                })

                        if current_penis_box_for_roi_calc[2] > 0 and current_penis_box_for_roi_calc[3] > 0:
                            candidate_roi_xywh = self.roi_tracker_instance._calculate_combined_roi(
                                processed_frame_for_tracker.shape[:2],
                                current_penis_box_for_roi_calc,
                                interacting_objects_for_roi_calc
                            )
                            if segment_obj.major_position == "Handjob / Blowjob" and candidate_roi_xywh:
                                narrow_factor = self.common_app_config.get("roi_narrow_factor_hjbj", DEFAULT_ROI_NARROW_FACTOR_HJBJ)
                                min_roi_dim = self.common_app_config.get("min_roi_dim_hjbj", DEFAULT_MIN_ROI_DIM_HJBJ)
                                rx, ry, rw, rh = candidate_roi_xywh
                                rw_new = max(min_roi_dim, int(rw * narrow_factor))
                                rh_new = max(min_roi_dim, int(rh * narrow_factor))
                                rx_new = rx + (rw - rw_new) // 2
                                ry_new = ry + (rh - rh_new) // 2
                                rx_new = max(0, min(rx_new, processed_frame_for_tracker.shape[1] - rw_new))
                                ry_new = max(0, min(ry_new, processed_frame_for_tracker.shape[0] - rh_new))
                                candidate_roi_xywh = (rx_new, ry_new, rw_new, rh_new)
                    self.roi_tracker_instance.roi = self.roi_tracker_instance._smooth_roi_transition(candidate_roi_xywh) if candidate_roi_xywh else None


                # --- Optical Flow Processing ---
                final_primary_pos, final_secondary_pos = 50, 50
                if self.roi_tracker_instance.roi and self.roi_tracker_instance.roi[2] > 0 and self.roi_tracker_instance.roi[3] > 0:
                    rx, ry, rw, rh = self.roi_tracker_instance.roi
                    # Ensure ROI coords are within frame dimensions before slicing
                    rx_c, ry_c = max(0, rx), max(0, ry)
                    rw_c = min(rw, current_frame_gray.shape[1] - rx_c)
                    rh_c = min(rh, current_frame_gray.shape[0] - ry_c)

                    if rw_c > 0 and rh_c > 0:
                        main_roi_patch_gray = current_frame_gray[ry_c: ry_c + rh_c, rx_c: rx_c + rw_c]
                        if main_roi_patch_gray.size > 0:
                            # process_main_roi_content updates tracker's prev_features_main_roi internally
                            final_primary_pos, final_secondary_pos, _, _, _ = \
                                self.roi_tracker_instance.process_main_roi_content(
                                    processed_frame_for_tracker, # For drawing debug visuals if enabled
                                    main_roi_patch_gray,
                                    self.roi_tracker_instance.prev_gray_main_roi,
                                    self.roi_tracker_instance.prev_features_main_roi
                                )
                            self.roi_tracker_instance.prev_gray_main_roi = main_roi_patch_gray.copy()
                        else: self.roi_tracker_instance.prev_gray_main_roi = None
                    else: self.roi_tracker_instance.prev_gray_main_roi = None
                else: self.roi_tracker_instance.prev_gray_main_roi = None


                # --- Funscript Writing ---
                can_write_action_s3 = (segment_obj.start_frame_id <= frame_id_to_process <= segment_obj.end_frame_id)
                if can_write_action_s3:
                    delay_ms = (self.roi_tracker_instance.output_delay_frames / self.roi_tracker_instance.current_video_fps_for_delay) * 1000.0 \
                        if self.roi_tracker_instance.current_video_fps_for_delay > 0 else 0.0
                    adjusted_frame_time_ms = frame_time_ms - delay_ms
                    final_adjusted_time_ms = max(0, int(round(adjusted_frame_time_ms)))

                    # Get axis mode from common_app_config
                    tracking_axis_mode = self.common_app_config.get("tracking_axis_mode", "both")
                    single_axis_target = self.common_app_config.get("single_axis_output_target", "primary")
                    primary_to_write, secondary_to_write = None, None

                    if tracking_axis_mode == "both":
                        primary_to_write, secondary_to_write = final_primary_pos, final_secondary_pos
                    elif tracking_axis_mode == "vertical":
                        if single_axis_target == "primary": primary_to_write = final_primary_pos
                        else: secondary_to_write = final_primary_pos
                    elif tracking_axis_mode == "horizontal":
                        if single_axis_target == "primary": primary_to_write = final_secondary_pos
                        else: secondary_to_write = final_secondary_pos

                    self.funscript.add_action(final_adjusted_time_ms, primary_to_write, secondary_to_write, is_from_live_tracker=False)

                self.roi_tracker_instance.internal_frame_counter += 1
                total_frames_processed_s3 +=1

                # Update progress
                if segment_obj.start_frame_id <= frame_id_to_process <= segment_obj.end_frame_id:
                    processed_in_seg_for_progress = frame_id_to_process - segment_obj.start_frame_id + 1
                    if processed_in_seg_for_progress % 10 == 0 or processed_in_seg_for_progress == num_frames_in_actual_segment_for_progress:
                        self.progress_callback(
                            seg_idx + 1, len(self.atr_segments), segment_name_for_progress,
                            processed_in_seg_for_progress, num_frames_in_actual_segment_for_progress,
                            self.current_fps, time_elapsed_s3, eta_s3
                        )
            # Final progress for the segment
            self.progress_callback(
                seg_idx + 1, len(self.atr_segments), segment_name_for_progress,
                num_frames_in_actual_segment_for_progress, num_frames_in_actual_segment_for_progress,
                self.current_fps, time_elapsed_s3, eta_s3
            )


        if self.video_processor:
            self.video_processor.reset(close_video=True)

        self.logger.info(f"S3 OF: Processing complete. Generated {len(self.funscript.primary_actions)} primary actions.")
        return {
            "primary_actions": list(self.funscript.primary_actions),
            "secondary_actions": list(self.funscript.secondary_actions)
        }


def perform_stage3_analysis(video_path: str,
                              atr_segments_list: List[ATRSegment],
                              s2_frame_objects_map: Dict[int, FrameObject],
                              tracker_config: Dict[str, Any],
                              common_app_config: Dict[str, Any],
                              progress_callback: callable,
                              stop_event: Any, # threading.Event or multiprocessing.Event
                              parent_logger: logging.Logger
                             ) -> Dict[str, Any]:
    """
    Main entry point for Stage 3 Optical Flow processing.
    """
    processor = Stage3OpticalFlowProcessor(
        video_path, atr_segments_list, s2_frame_objects_map,
        tracker_config, common_app_config,
        progress_callback, stop_event, parent_logger
    )
    results = processor.process_segments()
    return results
