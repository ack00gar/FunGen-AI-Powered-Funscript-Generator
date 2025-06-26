import time
import logging
import cv2
from typing import Optional, List, Dict, Any, Tuple

from funscript.dual_axis_funscript import DualAxisFunscript
from video.video_processor import VideoProcessor
from tracker.tracker import ROITracker
from detection.cd.stage_2_cd import FrameObject, ATRSegment, ATRLockedPenisState
from config import constants

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

        self.determined_video_type = self.video_processor.determined_video_type

        # Initialize ROITracker instance for S3 processing
        # ROITracker's __init__ will need to handle app_logic_instance=None
        # by taking all necessary configs directly.
        try:
            self.roi_tracker_instance = ROITracker(
                app_logic_instance=None, # Explicitly None for S3
                tracker_model_path=self.common_app_config.get('yolo_det_model_path', ''), # Not used for S3 flow, but needed by init
                pose_model_path=self.common_app_config.get('yolo_pose_model_path', ''),   # Not used for S3 flow
                confidence_threshold=self.tracker_config.get('confidence_threshold', 0.4),
                roi_padding=self.tracker_config.get('roi_padding', 20),
                roi_update_interval=self.tracker_config.get('roi_update_interval',
                                                            constants.DEFAULT_ROI_UPDATE_INTERVAL),
                roi_smoothing_factor=self.tracker_config.get('roi_smoothing_factor',
                                                             constants.DEFAULT_ROI_SMOOTHING_FACTOR),
                base_amplification_factor=self.tracker_config.get('base_amplification_factor',
                                                                  constants.DEFAULT_LIVE_TRACKER_BASE_AMPLIFICATION),

                dis_flow_preset=self.tracker_config.get('dis_flow_preset', "ULTRAFAST"),
                target_size_preprocess=self.tracker_config.get('target_size_preprocess', (640,640)),
                flow_history_window_smooth=self.tracker_config.get('flow_history_window_smooth', 3),
                adaptive_flow_scale=self.tracker_config.get('adaptive_flow_scale', True),
                use_sparse_flow=self.tracker_config.get('use_sparse_flow', False), # S3 typically uses dense
                max_frames_for_roi_persistence=self.tracker_config.get('max_frames_for_roi_persistence', constants.DEFAULT_ROI_PERSISTENCE_FRAMES), # Not really used in S3 like in live
                class_specific_amplification_multipliers=self.tracker_config.get('class_specific_amplification_multipliers', None),
                logger=self.logger.getChild("ROITracker_S3"),
                video_type_override=self.determined_video_type
            )
            # Set parameters that might not be in __init__ or need override for S3
            self.roi_tracker_instance.y_offset = self.tracker_config.get('y_offset',
                                                                         constants.DEFAULT_LIVE_TRACKER_Y_OFFSET)
            self.roi_tracker_instance.x_offset = self.tracker_config.get('x_offset',
                                                                         constants.DEFAULT_LIVE_TRACKER_X_OFFSET)
            self.roi_tracker_instance.sensitivity = self.tracker_config.get('sensitivity',
                                                                            constants.DEFAULT_LIVE_TRACKER_SENSITIVITY)
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

        relevant_segments = [
            seg for seg in self.atr_segments
            if seg.major_position not in ["Not Relevant", "Close Up"]
        ]

        if not relevant_segments:
            self.logger.info("S3 OF: No relevant segments to process.")
            if self.video_processor:
                self.video_processor.reset(close_video=True)
            return {"primary_actions": [], "secondary_actions": []}

        estimated_total_frames_s3 = sum(
            (seg.end_frame_id - max(0, seg.start_frame_id - self.common_app_config.get('num_warmup_frames_s3', 10)) + 1)
            for seg in relevant_segments
        )
        if estimated_total_frames_s3 == 0 and relevant_segments:
            estimated_total_frames_s3 = len(relevant_segments)

        self.roi_tracker_instance.start_tracking()

        relevant_seg_count = len(relevant_segments)
        processed_relevant_count = 0

        for original_idx, segment_obj in enumerate(self.atr_segments):
            if self.stop_event.is_set():
                self.logger.info("S3 OF: Stop event detected during segment processing.")
                break  # Exit the loop cleanly

            if segment_obj.major_position in ["Not Relevant", "Close Up"]:
                continue

            processed_relevant_count += 1
            segment_name_for_progress = f"{segment_obj.major_position} (F{segment_obj.start_frame_id}-{segment_obj.end_frame_id})"
            self.logger.info(
                f"S3 OF: Processing segment {processed_relevant_count}/{relevant_seg_count}: {segment_name_for_progress}")

            # Reset tracker's internal state for each new segment
            self.roi_tracker_instance.internal_frame_counter = 0
            self.roi_tracker_instance.prev_gray_main_roi = None
            self.roi_tracker_instance.prev_features_main_roi = None
            self.roi_tracker_instance.roi = None
            self.roi_tracker_instance.primary_flow_history_smooth.clear()
            self.roi_tracker_instance.secondary_flow_history_smooth.clear()
            self.roi_tracker_instance.main_interaction_class = segment_obj.major_position

            num_warmup_frames = self.common_app_config.get('num_warmup_frames_s3', 10)
            actual_processing_start_frame = max(0, segment_obj.start_frame_id - num_warmup_frames)
            actual_processing_end_frame = segment_obj.end_frame_id

            # Calculate the number of frames to read for the efficient streaming method.
            num_frames_to_read = actual_processing_end_frame - actual_processing_start_frame + 1
            if num_frames_to_read <= 0:
                continue

            # Use the efficient `stream_frames_for_segment` generator.
            # This starts ONE FFmpeg process for the entire segment and yields frames.
            frame_stream = self.video_processor.stream_frames_for_segment(
                start_frame_abs_idx=actual_processing_start_frame,
                num_frames_to_read=num_frames_to_read,
                stop_event=self.stop_event
            )

            num_frames_in_actual_segment_for_progress = segment_obj.end_frame_id - segment_obj.start_frame_id + 1

            for frame_id_to_process, current_frame_image in frame_stream:
                if self.stop_event.is_set():
                    self.logger.info("S3 OF: Stop event detected during frame streaming.")
                    break  # Exit this inner loop

                self._update_fps()
                time_elapsed_s3 = time.time() - s3_start_time
                average_fps_s3 = total_frames_processed_s3 / time_elapsed_s3 if time_elapsed_s3 > 1 else 0.0
                remaining_frames_s3 = estimated_total_frames_s3 - total_frames_processed_s3
                eta_s3 = remaining_frames_s3 / average_fps_s3 if average_fps_s3 > 0 and remaining_frames_s3 > 0 else 0.0

                if current_frame_image is None:
                    self.logger.warning(f"S3 OF: Stream yielded a None frame for ID {frame_id_to_process}. Skipping.")
                    continue

                processed_frame_for_tracker = self.roi_tracker_instance.preprocess_frame(current_frame_image)
                current_frame_gray = cv2.cvtColor(processed_frame_for_tracker, cv2.COLOR_BGR2GRAY)
                frame_time_ms = int(
                    round((frame_id_to_process / self.common_app_config.get('video_fps', 30.0)) * 1000.0))
                frame_obj_s2 = self.s2_frame_objects_map.get(frame_id_to_process)

                if not frame_obj_s2:
                    self.logger.warning(
                        f"S3 OF: S2 FrameObject not found for frame ID {frame_id_to_process}. ATR context might be limited for ROI.")

                    class MinimalFrameObject:
                        def __init__(self, fid):
                            self.frame_id = fid
                            self.atr_locked_penis_state = ATRLockedPenisState()
                            self.atr_detected_contact_boxes = []

                    frame_obj_s2 = MinimalFrameObject(frame_id_to_process)

                # --- ROI Definition Logic (adapted from ROITracker.process_frame_for_stage3) ---
                run_roi_definition_this_frame = False
                if self.roi_tracker_instance.roi is None:
                    run_roi_definition_this_frame = True
                elif self.roi_tracker_instance.roi_update_interval > 0 and \
                        (
                                self.roi_tracker_instance.internal_frame_counter % self.roi_tracker_instance.roi_update_interval == 0):
                    run_roi_definition_this_frame = True

                if run_roi_definition_this_frame:
                    candidate_roi_xywh: Optional[Tuple[int, int, int, int]] = None
                    if frame_obj_s2.atr_locked_penis_state.active and frame_obj_s2.atr_locked_penis_state.box:
                        lp_box_coords_xyxy = frame_obj_s2.atr_locked_penis_state.box
                        lp_x1, lp_y1, lp_x2, lp_y2 = lp_box_coords_xyxy
                        current_penis_box_for_roi_calc = (lp_x1, lp_y1, lp_x2 - lp_x1, lp_y2 - lp_y1)
                        interacting_objects_for_roi_calc = []
                        relevant_classes_for_pos = []
                        if segment_obj.major_position == "Cowgirl / Missionary":
                            relevant_classes_for_pos = ["pussy"]
                        elif segment_obj.major_position == "Rev. Cowgirl / Doggy":
                            relevant_classes_for_pos = ["butt"]
                        elif segment_obj.major_position == "Handjob / Blowjob":
                            relevant_classes_for_pos = ["face", "hand"]
                        elif segment_obj.major_position == "Boobjob":
                            relevant_classes_for_pos = ["breast", "hand"]
                        elif segment_obj.major_position == "Footjob":
                            relevant_classes_for_pos = ["foot"]
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
                            if self.determined_video_type == 'VR' and candidate_roi_xywh:
                                penis_w = current_penis_box_for_roi_calc[2]
                                rx, ry, rw, rh = candidate_roi_xywh
                                new_rw = 0

                                if segment_obj.major_position in ["Handjob", "Blowjob", "Handjob / Blowjob"]:
                                    # For HJ/BJ, lock width to the penis box width
                                    new_rw = penis_w
                                else:
                                    # For other positions, limit to 2x penis box width
                                    new_rw = min(rw, penis_w * 2)

                                if new_rw > 0:
                                    # Recenter the new, narrower ROI
                                    original_center_x = rx + rw / 2
                                    new_rx = int(original_center_x - new_rw / 2)

                                    # Ensure the new ROI stays within frame boundaries
                                    frame_width = processed_frame_for_tracker.shape[1]
                                    final_rw = int(min(new_rw, frame_width))
                                    final_rx = max(0, min(new_rx, frame_width - final_rw))

                                    candidate_roi_xywh = (final_rx, ry, final_rw, rh)

                    if candidate_roi_xywh:
                        self.roi_tracker_instance.roi = self.roi_tracker_instance._smooth_roi_transition(
                            candidate_roi_xywh)

                # --- Optical Flow Processing ---
                final_primary_pos, final_secondary_pos = 50, 50
                if self.roi_tracker_instance.roi and self.roi_tracker_instance.roi[2] > 0 and \
                        self.roi_tracker_instance.roi[3] > 0:
                    rx, ry, rw, rh = self.roi_tracker_instance.roi
                    rx_c, ry_c = max(0, rx), max(0, ry)
                    rw_c = min(rw, current_frame_gray.shape[1] - rx_c)
                    rh_c = min(rh, current_frame_gray.shape[0] - ry_c)
                    if rw_c > 0 and rh_c > 0:
                        main_roi_patch_gray = current_frame_gray[ry_c: ry_c + rh_c, rx_c: rx_c + rw_c]
                        if main_roi_patch_gray.size > 0:
                            final_primary_pos, final_secondary_pos, _, _, _ = \
                                self.roi_tracker_instance.process_main_roi_content(
                                    processed_frame_for_tracker,
                                    main_roi_patch_gray,
                                    self.roi_tracker_instance.prev_gray_main_roi,
                                    self.roi_tracker_instance.prev_features_main_roi
                                )
                            # Update the FrameObject with the determined motion mode
                            if frame_obj_s2:
                                frame_obj_s2.motion_mode = self.roi_tracker_instance.motion_mode

                            self.roi_tracker_instance.prev_gray_main_roi = main_roi_patch_gray.copy()
                        else:
                            self.roi_tracker_instance.prev_gray_main_roi = None
                    else:
                        self.roi_tracker_instance.prev_gray_main_roi = None
                else:
                    self.roi_tracker_instance.prev_gray_main_roi = None

                # --- Funscript Writing ---
                can_write_action_s3 = (segment_obj.start_frame_id <= frame_id_to_process <= segment_obj.end_frame_id)
                if can_write_action_s3:
                    # --- Lag Compensation (manual + automatic) ---
                    # Calculate inherent delay from the smoothing window. A window of N has a lag of (N-1)/2 frames.
                    smoothing_window = self.roi_tracker_instance.flow_history_window_smooth
                    automatic_smoothing_delay_frames = (smoothing_window - 1) / 2.0 if smoothing_window > 1 else 0.0

                    # Combine automatic compensation with the user's manual delay setting.
                    total_delay_frames = self.roi_tracker_instance.output_delay_frames + automatic_smoothing_delay_frames

                    # Convert the total frame delay to milliseconds.
                    delay_ms = (total_delay_frames / self.roi_tracker_instance.current_video_fps_for_delay) * 1000.0 \
                        if self.roi_tracker_instance.current_video_fps_for_delay > 0 else 0.0

                    adjusted_frame_time_ms = frame_time_ms - delay_ms
                    final_adjusted_time_ms = max(0, int(round(adjusted_frame_time_ms)))
                    tracking_axis_mode = self.common_app_config.get("tracking_axis_mode", "both")
                    single_axis_target = self.common_app_config.get("single_axis_output_target", "primary")
                    primary_to_write, secondary_to_write = None, None
                    if tracking_axis_mode == "both":
                        primary_to_write, secondary_to_write = final_primary_pos, final_secondary_pos
                    elif tracking_axis_mode == "vertical":
                        if single_axis_target == "primary":
                            primary_to_write = final_primary_pos
                        else:
                            secondary_to_write = final_primary_pos
                    elif tracking_axis_mode == "horizontal":
                        if single_axis_target == "primary":
                            primary_to_write = final_secondary_pos
                        else:
                            secondary_to_write = final_secondary_pos
                    self.funscript.add_action(final_adjusted_time_ms, primary_to_write, secondary_to_write,
                                              is_from_live_tracker=False)

                self.roi_tracker_instance.internal_frame_counter += 1
                total_frames_processed_s3 += 1

                if segment_obj.start_frame_id <= frame_id_to_process <= segment_obj.end_frame_id:
                    processed_in_seg_for_progress = frame_id_to_process - segment_obj.start_frame_id + 1
                    if processed_in_seg_for_progress % 10 == 0 or processed_in_seg_for_progress == num_frames_in_actual_segment_for_progress:
                        self.progress_callback(
                            processed_relevant_count, relevant_seg_count, segment_name_for_progress,
                            processed_in_seg_for_progress, num_frames_in_actual_segment_for_progress,
                            total_frames_processed_s3, estimated_total_frames_s3,
                            self.current_fps, time_elapsed_s3, eta_s3,
                            original_idx + 1
                        )

            # Final progress update for the segment
            self.progress_callback(
                processed_relevant_count, relevant_seg_count, segment_name_for_progress,
                num_frames_in_actual_segment_for_progress, num_frames_in_actual_segment_for_progress,
                total_frames_processed_s3, estimated_total_frames_s3,
                self.current_fps, time_elapsed_s3, eta_s3,
                original_idx + 1
            )

        if self.video_processor:
            # The reset call in the new loop will handle the ffmpeg process termination.
            self.video_processor.reset(close_video=True)

        self.logger.info(
            f"S3 OF: Processing complete. Generated {len(self.funscript.primary_actions)} primary actions.")
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
