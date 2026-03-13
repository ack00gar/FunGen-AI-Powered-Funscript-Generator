"""Stage execution methods for the processing pipeline.

Extracted from AppStageProcessor to reduce file size.
Contains _execute_stage1_logic, _execute_stage2_logic,
_execute_stage3_optical_flow_module, _execute_stage3_mixed_module,
and their helpers.

Used as a mixin — all methods operate on the AppStageProcessor instance.
"""
import os
import gc
import pickle
from typing import Optional, List, Dict, Any, Tuple

from config import constants
from config.constants import ChapterSource, ChapterSegmentType
from application.utils import VideoSegment

import detection.cd.stage_1_cd as stage1_module
import detection.cd.stage_2_cd as stage2_module
import detection.cd.stage_3_of_processor as stage3_module
import detection.cd.stage_3_mixed_processor as stage3_mixed_module


class StageExecutorMixin:
    """Mixin with methods that launch and manage individual stage executions."""

    def _filter_segments_for_range(self, all_segments: List[Any], range_is_active: bool,
                                   start_frame: Optional[int], end_frame: Optional[int]) -> List[Any]:
        if not range_is_active:
            return all_segments
        if start_frame is None:
            self.logger.warning(
                "Segment filtering called for active range but start_frame is None. Returning all segments.")
            return all_segments

        effective_end_frame = end_frame
        if effective_end_frame is None or effective_end_frame == -1:
            if self.app.processor and self.app.processor.total_frames > 0:
                effective_end_frame = self.app.processor.total_frames - 1
            else:
                return [seg for seg in all_segments if seg.end_frame_id >= start_frame]

        filtered_segments = [
            seg for seg in all_segments
            if max(seg.start_frame_id, start_frame) <= min(seg.end_frame_id, effective_end_frame)
        ]
        self.logger.info(f"Found {len(filtered_segments)} segments overlapping with the selected range.")
        return filtered_segments

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------

    def _execute_stage1_logic(self, frame_range: Optional[Tuple[Optional[int], Optional[int]]] = None,
                              output_path: Optional[str] = None,
                              num_producers_override: Optional[int] = None,
                              num_consumers_override: Optional[int] = None,
                              is_autotune_run: bool = False) -> Dict[str, Any]:
        self.gui_event_queue.put(("stage1_status_update", "Running S1...", "Initializing S1..."))
        fm = self.app.file_manager
        self.stage1_frame_queue_size = 0
        self.stage1_result_queue_size = 0

        logger_config_for_stage1 = {
            'main_logger': self.logger,
            'log_file': self.app.app_log_file_path,
            'log_level': self.logger.level
        }
        try:
            if not stage1_module:
                self.gui_event_queue.put(("stage1_status_update", "Error - S1 Module not loaded.", "Error"))
                return {"success": False, "max_fps": 0.0}

            preprocessed_video_path = fm.get_output_path_for_file(fm.video_path, "_preprocessed.mp4")

            num_producers = num_producers_override if num_producers_override is not None else self.num_producers_stage1
            num_consumers = num_consumers_override if num_consumers_override is not None else self.num_consumers_stage1

            result_path, max_fps = stage1_module.perform_yolo_analysis(
                video_path_arg=fm.video_path,
                yolo_model_path_arg=self.app.yolo_det_model_path,
                yolo_pose_model_path_arg=self.app.yolo_pose_model_path,
                confidence_threshold=self.app.tracker.confidence_threshold,
                progress_callback=self.on_stage1_progress,
                stop_event_external=self.stop_stage_event,
                num_producers_arg=num_producers,
                num_consumers_arg=num_consumers,
                hwaccel_method_arg=self.app.hardware_acceleration_method,
                hwaccel_avail_list_arg=self.app.available_ffmpeg_hwaccels,
                video_type_arg=self.app.processor.video_type_setting if self.app.processor else "auto",
                vr_input_format_arg=self.app.processor.vr_input_format if self.app.processor else "he",
                vr_fov_arg=self.app.processor.vr_fov if self.app.processor else 190,
                vr_pitch_arg=self.app.processor.vr_pitch if self.app.processor else 0,
                yolo_input_size_arg=self.app.yolo_input_size,
                app_logger_config_arg=logger_config_for_stage1,
                gui_event_queue_arg=self.gui_event_queue,
                frame_range_arg=frame_range,
                output_filename_override=output_path,
                save_preprocessed_video_arg=self.save_preprocessed_video,
                preprocessed_video_path_arg=preprocessed_video_path if self.save_preprocessed_video else None,
                is_autotune_run_arg=is_autotune_run
            )
            if self.stop_stage_event.is_set():
                self.gui_event_queue.put(("stage1_status_update", "S1 Aborted by user.", "Aborted"))
                self.gui_event_queue.put(
                    ("stage1_progress_update", 0.0, {"message": "Aborted", "current": 0, "total": 1}))
                return {"success": False, "max_fps": 0.0}
            if result_path and os.path.exists(result_path):
                fm.stage1_output_msgpack_path = result_path
                if self.save_preprocessed_video and os.path.exists(preprocessed_video_path):
                    fm.preprocessed_video_path = preprocessed_video_path
                    self.logger.info(f"Preprocessed video saved: {os.path.basename(preprocessed_video_path)}")
                final_msg = f"S1 Completed. Output: {os.path.basename(result_path)}"
                self.gui_event_queue.put(("stage1_status_update", final_msg, "Done"))
                self.gui_event_queue.put(("stage1_progress_update", 1.0, {"message": "Done", "current": 1, "total": 1}))
                self.app.project_manager.project_dirty = True
                return {"success": True, "max_fps": max_fps,
                        "preprocessed_video_path": preprocessed_video_path if self.save_preprocessed_video else None}
            self.gui_event_queue.put(("stage1_status_update", "S1 Failed (no output file).", "Failed"))
            return {"success": False, "max_fps": 0.0, "preprocessed_video_path": None}
        except Exception as e:
            self.logger.error(f"Stage 1 execution error in AppLogic: {e}", exc_info=True,
                              extra={'status_message': True})
            self.gui_event_queue.put(("stage1_status_update", f"S1 Error - {str(e)}", "Error"))
            return {"success": False, "max_fps": 0.0, "preprocessed_video_path": None}

    # ------------------------------------------------------------------
    # Stage 2 — loading existing data
    # ------------------------------------------------------------------

    def _load_existing_stage2_data(self, stage2_data_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Load existing Stage 2 data from database and overlay files."""
        try:
            file_paths = stage2_data_info.get('file_paths', {})
            db_path = file_paths.get('database')
            overlay_path = file_paths.get('overlay_msgpack')

            self.logger.info(f"DEBUG _load_existing_stage2_data: db_path={db_path}, overlay_path={overlay_path}")

            loaded_data = {
                "video_segments": [],
                "segments_objects": [],
                "overlay_data": None,
                "frame_objects_map": {},
                "all_s2_frame_objects_list": []
            }

            # Load segments and frame data from database
            if db_path and os.path.exists(db_path):
                try:
                    import sqlite3
                    with sqlite3.connect(db_path) as conn:
                        conn.row_factory = sqlite3.Row
                        cursor = conn.cursor()

                        segment_tables = ['atr_segments', 'segments']
                        for table_name in segment_tables:
                            try:
                                from application.utils.video_segment import VideoSegment as VS
                                cursor.execute(f"SELECT * FROM {table_name}")
                                for segment_row in cursor:
                                    _seg = pickle.loads(segment_row['segment_data'])
                                    segment = VS(
                                        start_frame_id=_seg.start_frame_id,
                                        end_frame_id=_seg.end_frame_id,
                                        class_id=getattr(_seg, 'class_id', None),
                                        class_name=getattr(_seg, 'class_name', 'unknown'),
                                        segment_type=ChapterSegmentType.POSITION.value,
                                        position_short_name=_seg.position_short_name,
                                        position_long_name=_seg.position_long_name,
                                        source=ChapterSource.STAGE2.value
                                    )
                                    loaded_data["video_segments"].append(segment)
                                    loaded_data["segments_objects"].append(segment)
                                break
                            except Exception:
                                continue

                        try:
                            from detection.cd.stage_2_sqlite_storage import Stage2SQLiteStorage
                            storage = Stage2SQLiteStorage(db_path, self.logger)
                            min_frame, max_frame = storage.get_frame_range()
                            if min_frame is not None and max_frame is not None:
                                frame_objects_dict = storage.get_frame_objects_range(min_frame, max_frame)
                                loaded_data["frame_objects_map"] = frame_objects_dict
                                loaded_data["all_s2_frame_objects_list"] = list(frame_objects_dict.values())
                                self.logger.info(f"Loaded {len(frame_objects_dict)} frame objects from database")
                            storage.close()
                        except Exception as fe:
                            self.logger.warning(f"Failed to load frame objects from database: {fe}")

                        self.app.s2_sqlite_db_path = db_path
                        loaded_data["sqlite_db_path"] = db_path

                except Exception as e:
                    self.logger.warning(f"Failed to load segments from database: {e}")

            # Load overlay data if available
            if overlay_path and os.path.exists(overlay_path):
                try:
                    import msgpack
                    with open(overlay_path, 'rb') as f:
                        overlay_data = msgpack.unpack(f, raw=False)
                        loaded_data["overlay_data"] = overlay_data
                except Exception as e:
                    self.logger.warning(f"Failed to load overlay data: {e}")

            has_segments = bool(loaded_data["video_segments"])
            has_frame_objects = stage2_data_info.get('frame_objects_available', False)

            if has_segments or has_frame_objects:
                self.logger.info(f"Loaded existing Stage 2 data: {len(loaded_data['video_segments'])} segments")

                if not has_segments and has_frame_objects and loaded_data.get("overlay_data"):
                    self._reconstruct_segments_from_overlay(loaded_data, stage2_data_info)

                # Create funscript object from loaded segments
                self._create_funscript_from_loaded_data(loaded_data)
                return loaded_data
            else:
                self.logger.warning("No usable Stage 2 data found in existing assets")
                return None

        except Exception as e:
            self.logger.error(f"Error loading existing Stage 2 data: {e}")
            return None

    def _reconstruct_segments_from_overlay(self, loaded_data: Dict, stage2_data_info: Dict):
        """Reconstruct video segments from overlay data when segments aren't in DB."""
        self.logger.info("Reconstructing video segments from Stage 2 overlay data")
        try:
            frame_objects = self._reconstruct_frame_objects_from_overlay(loaded_data["overlay_data"])
            if frame_objects:
                from detection.cd.stage_2_cd import _aggregate_segments

                fps = 30.0
                if self.app and hasattr(self.app, 'processor') and self.app.processor:
                    video_info = getattr(self.app.processor, 'video_info', {})
                    fps = video_info.get('fps', 30.0)

                min_segment_duration_frames = int(fps * 1.0)
                segments = _aggregate_segments(frame_objects, fps, min_segment_duration_frames, self.logger)

                from application.utils.video_segment import VideoSegment as VS
                for segment in segments:
                    segment_dict = segment.to_dict()
                    video_segment = VS(
                        start_frame_id=segment_dict['start_frame_id'],
                        end_frame_id=segment_dict['end_frame_id'],
                        class_id=1,
                        class_name=segment_dict['class_name'],
                        segment_type=ChapterSegmentType.POSITION.value,
                        position_short_name=segment_dict['position_short_name'],
                        position_long_name=segment_dict['position_long_name'],
                        duration=segment_dict['duration'],
                        source=ChapterSource.STAGE3.value
                    )
                    loaded_data["video_segments"].append(video_segment)
                    loaded_data["segments_objects"].append(segment)

                frame_objects_map = {fo.frame_id: fo for fo in frame_objects}
                loaded_data["frame_objects_map"] = frame_objects_map
                loaded_data["all_s2_frame_objects_list"] = frame_objects
                self.logger.info(f"Reconstructed {len(segments)} segments and {len(frame_objects)} frame objects from overlay data")
            else:
                self.logger.warning("Failed to reconstruct frame objects from overlay data")
        except Exception as e:
            self.logger.warning(f"Failed to reconstruct segments from overlay: {e}")
            self.logger.info("Falling back to single full-video segment")
            from application.utils.video_segment import VideoSegment as VS
            estimated_frame_count = stage2_data_info.get('estimated_frame_count', 17982)
            fallback_segment = VS(
                start_frame_id=0, end_frame_id=estimated_frame_count - 1,
                class_id=1, class_name="mixed", segment_type="SexAct",
                position_short_name="Mixed", position_long_name="Mixed Content"
            )
            loaded_data["video_segments"].append(fallback_segment)

    def _create_funscript_from_loaded_data(self, loaded_data: Dict):
        """Create funscript object from loaded segments for consistency."""
        try:
            from funscript.multi_axis_funscript import MultiAxisFunscript
            fps = 30.0
            if self.app and hasattr(self.app, 'processor') and self.app.processor:
                video_info = getattr(self.app.processor, 'video_info', {})
                fps = video_info.get('fps', 30.0)

            funscript_obj = MultiAxisFunscript(fps=fps)

            if loaded_data["video_segments"]:
                funscript_obj.set_chapters_from_segments(loaded_data["video_segments"], fps)
                self.logger.info(f"Created funscript with {len(funscript_obj.chapters)} chapters from loaded segments")

            if loaded_data.get("all_s2_frame_objects_list"):
                actions_generated = 0
                for frame_obj in loaded_data["all_s2_frame_objects_list"]:
                    if hasattr(frame_obj, 'atr_funscript_distance') and hasattr(frame_obj, 'frame_id'):
                        try:
                            timestamp_ms = int((frame_obj.frame_id / fps) * 1000)
                            pos_0_100 = max(0, min(100, int(frame_obj.atr_funscript_distance)))
                            funscript_obj.add_action(timestamp_ms, pos_0_100)
                            actions_generated += 1
                        except (ValueError, TypeError, AttributeError):
                            continue
                if actions_generated > 0:
                    self.logger.info(f"Regenerated {actions_generated} funscript actions from cached frame objects for mixed mode")
                else:
                    self.logger.warning("No valid funscript actions could be regenerated from cached frame objects")

            loaded_data["funscript"] = funscript_obj
        except Exception as e:
            self.logger.warning(f"Failed to create funscript from loaded data: {e}")

    def _process_stage2_results_direct(self, packaged_data: Dict[str, Any], s2_overlay_path: Optional[str] = None):
        """Process Stage 2 results directly (CLI mode / cached data)."""
        fs_proc = self.app.funscript_processor
        results_dict = packaged_data.get("results_dict", {})
        funscript_obj = results_dict.get("funscript")

        if funscript_obj:
            primary_actions = funscript_obj.primary_actions
            secondary_actions = funscript_obj.secondary_actions

            # Reuse shared helper from StageGuiEventsMixin
            self._apply_actions_to_timelines(primary_actions, secondary_actions, fs_proc, "Stage 2")

            self.logger.info("Updating chapters with Stage 2 analysis results.")
            fs_proc.video_chapters.clear()

            if hasattr(funscript_obj, 'chapters') and funscript_obj.chapters:
                fps = (self.app.processor.video_info.get('fps', 30.0)
                       if self.app.processor and self.app.processor.video_info else 30.0)
                for chapter in funscript_obj.chapters:
                    start_frame = int((chapter.get('start', 0) / 1000.0) * fps)
                    end_frame = int((chapter.get('end', 0) / 1000.0) * fps)
                    from application.utils.video_segment import VideoSegment as VS
                    video_segment = VS(
                        start_frame_id=start_frame,
                        end_frame_id=end_frame,
                        class_id=chapter.get('class_id'),
                        class_name=chapter.get('name', 'Unknown'),
                        segment_type="SexAct",
                        position_short_name=chapter.get('position_short', chapter.get('name', '')),
                        position_long_name=chapter.get('position_long', chapter.get('description', chapter.get('name', 'Unknown'))),
                        source="stage2_funscript"
                    )
                    fs_proc.video_chapters.append(video_segment)
                self.logger.info(f"Extracted {len(funscript_obj.chapters)} chapters from funscript object")

        self.stage2_status_text = "S2 Completed. Results Processed."
        self.app.project_manager.project_dirty = True
        self.logger.info("Processed Stage 2 results directly.")

    def _reconstruct_frame_objects_from_overlay(self, overlay_data):
        """Reconstruct minimal frame objects from Stage 2 overlay data."""
        try:
            from detection.cd.data_structures import FrameObject

            frame_objects = []
            for frame_dict in overlay_data:
                if isinstance(frame_dict, dict) and 'frame_id' in frame_dict:
                    frame_obj = FrameObject(
                        frame_id=frame_dict.get('frame_id', 0),
                        yolo_input_size=640
                    )
                    frame_obj.atr_assigned_position = frame_dict.get('atr_assigned_position', 'unknown')
                    frame_obj.motion_mode = frame_dict.get('motion_mode', 'unknown')
                    frame_obj.active_interaction_track_id = frame_dict.get('active_interaction_track_id', 0)
                    frame_objects.append(frame_obj)

            self.logger.debug(f"Reconstructed {len(frame_objects)} frame objects from overlay data")
            return frame_objects
        except Exception as e:
            self.logger.warning(f"Error reconstructing frame objects from overlay: {e}")
            return []

    # ------------------------------------------------------------------
    # Stage 2 — execution
    # ------------------------------------------------------------------

    def _execute_stage2_logic(self, s2_overlay_output_path: Optional[str],
                              generate_funscript_actions: bool = True,
                              is_ranged_data_source: bool = False) -> Dict[str, Any]:
        self.gui_event_queue.put(("stage2_status_update", "Checking existing S2...", "Validating"))
        fm = self.app.file_manager

        # Check if we can skip Stage 2 by reusing existing assets
        if not self.force_rerun_stage2_segmentation:
            from application.utils.stage_output_validator import can_skip_stage2_for_stage3
            output_folder = os.path.dirname(fm.get_output_path_for_file(fm.video_path, "_dummy.tmp"))
            project_db_path = getattr(self.app, 's2_sqlite_db_path', None)
            can_skip, stage2_data = can_skip_stage2_for_stage3(fm.video_path, False, output_folder, self.logger,
                                                                project_db_path)
            if can_skip:
                self.logger.info("Stage 2 assets found and validated - skipping Stage 2 processing")
                self.gui_event_queue.put(("stage2_status_update", "Reusing existing S2...", "Loading cached results"))
                existing_data = self._load_existing_stage2_data(stage2_data)
                if existing_data:
                    self.gui_event_queue.put(
                        ("stage2_dual_progress", (6, 6, "Loaded from cache"), (1, 1, "Complete")))
                    self.gui_event_queue.put(("stage2_status_update", "S2 Complete (Cached)", "Loaded from cache"))
                    if s2_overlay_output_path and os.path.exists(s2_overlay_output_path):
                        self.gui_event_queue.put(("load_s2_overlay", s2_overlay_output_path, None))

                    packaged_data = {
                        "results_dict": existing_data,
                        "was_ranged": False,
                        "range_frames": (0, -1)
                    }
                    try:
                        self._process_stage2_results_direct(packaged_data, s2_overlay_output_path)
                    except Exception as e:
                        self.logger.warning(f"Failed to process Stage 2 results directly: {e}")
                        self.gui_event_queue.put(
                            ("stage2_results_success", packaged_data, s2_overlay_output_path))

                    self.logger.info("DEBUG: Returning early with cached data")
                    return {
                        "success": True, "data": existing_data,
                        "skipped": True, "skip_reason": "Existing Stage 2 assets validated and reused"
                    }
                else:
                    self.logger.warning("Failed to load existing Stage 2 data - will reprocess")
            else:
                self.logger.debug("Stage 2 assets not suitable for reuse - processing from scratch")
        else:
            self.logger.debug("Stage 2 force rerun enabled - processing from scratch")

        self.gui_event_queue.put(("stage2_status_update", "Running S2...", "Initializing S2..."))
        initial_total_main_steps = getattr(stage2_module, 'ATR_PASS_COUNT', self.S2_TOTAL_MAIN_STEPS_FALLBACK)
        if not generate_funscript_actions:
            initial_total_main_steps = getattr(stage2_module, 'ATR_PASS_COUNT_SEGMENTATION_ONLY', 3)
            self.gui_event_queue.put(
                ("stage2_status_update", "Running S2 (Segmentation)...", "Initializing S2 Seg..."))

        self.gui_event_queue.put(
            ("stage2_dual_progress", (1, initial_total_main_steps, "Initializing..."), (0, 1, "Starting")))

        try:
            if not stage2_module:
                msg = "Error - S2 Module not loaded."
                self.gui_event_queue.put(("stage2_status_update", msg, "Error"))
                return {"success": False, "error": msg}
            if not fm.stage1_output_msgpack_path:
                msg = "Error - S1 output missing for S2."
                self.gui_event_queue.put(("stage2_status_update", msg, "Error"))
                return {"success": False, "error": msg}

            preprocessed_video_path_for_s2 = None
            if self.app.file_manager.preprocessed_video_path and os.path.exists(
                    self.app.file_manager.preprocessed_video_path):
                preprocessed_video_path_for_s2 = self.app.file_manager.preprocessed_video_path
            else:
                preprocessed_video_path_for_s2 = fm.get_output_path_for_file(fm.video_path, "_preprocessed.mp4")
                if not os.path.exists(preprocessed_video_path_for_s2):
                    self.logger.warning(
                        "Optical flow recovery may fail: Preprocessed video from Stage 1 not found.")
                    preprocessed_video_path_for_s2 = None

            range_is_active, range_start_frame, range_end_frame = self.app.funscript_processor.get_effective_scripting_range()

            self.logger.info("Using Stage 2 implementation")
            stage2_results = stage2_module.perform_contact_analysis(
                video_path_arg=fm.video_path,
                msgpack_file_path_arg=fm.stage1_output_msgpack_path,
                preprocessed_video_path_arg=preprocessed_video_path_for_s2,
                progress_callback=self.on_stage2_progress,
                stop_event=self.stop_stage_event,
                app=self.app,
                ml_model_dir_path_arg=self.app.pose_model_artifacts_dir,
                output_overlay_msgpack_path=s2_overlay_output_path,
                parent_logger_arg=self.logger,
                yolo_input_size_arg=self.app.yolo_input_size,
                video_type_arg=self.app.processor.video_type_setting if self.app.processor else "auto",
                vr_input_format_arg=self.app.processor.vr_input_format if self.app.processor else "he",
                vr_fov_arg=self.app.processor.vr_fov if self.app.processor else 190,
                vr_pitch_arg=self.app.processor.vr_pitch if self.app.processor else 0,
                vr_vertical_third_filter_arg=self.app_settings.get("vr_filter_stage2", True),
                enable_of_debug_prints=self.app_settings.get("debug_prints_stage2", False),
                discarded_classes_runtime_arg=self.app.discarded_tracking_classes,
                scripting_range_active_arg=range_is_active,
                scripting_range_start_frame_arg=range_start_frame,
                scripting_range_end_frame_arg=range_end_frame,
                is_ranged_data_source=is_ranged_data_source,
                generate_funscript_actions_arg=generate_funscript_actions,
                output_folder_path=os.path.dirname(fm.get_output_path_for_file(fm.video_path, "_dummy.tmp"))
            )
            if self.stop_stage_event.is_set():
                msg = "S2 Aborted by user."
                self.gui_event_queue.put(("stage2_status_update", msg, "Aborted"))
                current_main_step = int(self.stage2_main_progress_value * initial_total_main_steps)
                self.gui_event_queue.put(("stage2_dual_progress",
                                          (current_main_step, initial_total_main_steps, "Aborted"),
                                          (0, 1, "Aborted")))
                return {"success": False, "error": msg}

            if stage2_results and "error" not in stage2_results:
                sqlite_db_path = stage2_results.get("sqlite_db_path")
                if sqlite_db_path:
                    self.app.s2_sqlite_db_path = sqlite_db_path
                    self.logger.info(f"Stage 2 database path saved: {sqlite_db_path}")

                if generate_funscript_actions:
                    packaged_data = {
                        "results_dict": stage2_results,
                        "was_ranged": range_is_active,
                        "range_frames": (range_start_frame, range_end_frame)
                    }
                    self.gui_event_queue.put(("stage2_results_success", packaged_data, s2_overlay_output_path))
                    status_msg = "S2 Completed. Results Processed."
                else:
                    status_msg = "S2 Segmentation Completed."
                self.gui_event_queue.put(("stage2_status_update", status_msg, "Done"))
                self.gui_event_queue.put(("stage2_dual_progress",
                                          (initial_total_main_steps, initial_total_main_steps,
                                           "Completed" if generate_funscript_actions else "Segmentation Done"),
                                          (1, 1, "Done")))
                self.app.project_manager.project_dirty = True
                return {"success": True, "data": stage2_results}
            error_msg = stage2_results.get("error",
                                           "Unknown S2 failure") if stage2_results else "S2 returned None."
            self.gui_event_queue.put(("stage2_status_update", f"S2 Failed: {error_msg}", "Failed"))
            return {"success": False, "error": error_msg}
        except Exception as e:
            self.logger.error(f"Stage 2 execution error in AppLogic: {e}", exc_info=True,
                              extra={'status_message': True})
            error_msg = f"S2 Exception: {str(e)}"
            self.gui_event_queue.put(("stage2_status_update", error_msg, "Error"))
            return {"success": False, "error": error_msg}

    # ------------------------------------------------------------------
    # Stage 3 — optical flow
    # ------------------------------------------------------------------

    def _execute_stage3_optical_flow_module(self, segments_objects: List[Any],
                                            preprocessed_video_path: Optional[str]) -> bool:
        """Wrapper to call the Stage 3 OF module."""
        fs_proc = self.app.funscript_processor

        if not self.app.file_manager.video_path:
            self.logger.error("Stage 3: Video path not available.")
            self.gui_event_queue.put(("stage3_status_update", "Error: Video path missing", "Error"))
            return False

        if not stage3_module:
            self.logger.error("Stage 3: Optical Flow processing module (stage3_module) not loaded.")
            self.gui_event_queue.put(("stage3_status_update", "Error: S3 Module missing", "Error"))
            return False

        tracker_config_s3 = {
            "confidence_threshold": self.app_settings.get('tracker_confidence_threshold', 0.4),
            "roi_padding": self.app_settings.get('tracker_roi_padding', 20),
            "roi_update_interval": self.app_settings.get('s3_roi_update_interval',
                                                          constants.DEFAULT_ROI_UPDATE_INTERVAL),
            "roi_smoothing_factor": self.app_settings.get('tracker_roi_smoothing_factor',
                                                           constants.DEFAULT_ROI_SMOOTHING_FACTOR),
            "dis_flow_preset": self.app_settings.get('tracker_dis_flow_preset', "ULTRAFAST"),
            "target_size_preprocess": getattr(self.app.tracker, 'target_size_preprocess',
                                              (640, 640)) if self.app.tracker else (640, 640),
            "flow_history_window_smooth": self.app_settings.get('tracker_flow_history_window_smooth', 3),
            "adaptive_flow_scale": self.app_settings.get('tracker_adaptive_flow_scale', True),
            "use_sparse_flow": self.app_settings.get('tracker_use_sparse_flow', False),
            "base_amplification_factor": self.app_settings.get('tracker_base_amplification',
                                                                constants.DEFAULT_LIVE_TRACKER_BASE_AMPLIFICATION),
            "class_specific_amplification_multipliers": self.app_settings.get(
                'tracker_class_specific_multipliers', constants.DEFAULT_CLASS_AMP_MULTIPLIERS),
            "y_offset": self.app_settings.get('tracker_y_offset', constants.DEFAULT_LIVE_TRACKER_Y_OFFSET),
            "x_offset": self.app_settings.get('tracker_x_offset', constants.DEFAULT_LIVE_TRACKER_X_OFFSET),
            "sensitivity": self.app_settings.get('tracker_sensitivity',
                                                  constants.DEFAULT_LIVE_TRACKER_SENSITIVITY),
            "oscillation_grid_size": self.app_settings.get('oscillation_detector_grid_size', 20),
            "oscillation_sensitivity": self.app_settings.get('oscillation_detector_sensitivity', 1.0)
        }

        video_fps_s3 = self._resolve_video_fps()

        common_app_config_s3 = {
            "yolo_det_model_path": self.app.yolo_det_model_path,
            "yolo_pose_model_path": self.app.yolo_pose_model_path,
            "yolo_input_size": self.app.yolo_input_size,
            "video_fps": video_fps_s3,
            "output_delay_frames": getattr(self.app.tracker, 'output_delay_frames',
                                           0) if self.app.tracker else 0,
            "num_warmup_frames_s3": self.app_settings.get('s3_num_warmup_frames',
                                                           10 + (getattr(self.app.tracker, 'output_delay_frames',
                                                                         0) if self.app.tracker else 0)),
            "roi_narrow_factor_hjbj": self.app_settings.get("roi_narrow_factor_hjbj",
                                                             constants.DEFAULT_ROI_NARROW_FACTOR_HJBJ),
            "min_roi_dim_hjbj": self.app_settings.get("min_roi_dim_hjbj", constants.DEFAULT_MIN_ROI_DIM_HJBJ),
            "tracking_axis_mode": self.app.tracking_axis_mode,
            "single_axis_output_target": self.app.single_axis_output_target,
            "s3_show_roi_debug": self.app_settings.get("s3_show_roi_debug", False),
            "hardware_acceleration_method": self.app.hardware_acceleration_method,
            "available_ffmpeg_hwaccels": self.app.available_ffmpeg_hwaccels,
            "video_type": self.app.processor.video_type_setting if self.app.processor else "auto",
            "vr_input_format": self.app.processor.vr_input_format if self.app.processor else "he",
            "vr_fov": self.app.processor.vr_fov if self.app.processor else 190,
            "vr_pitch": self.app.processor.vr_pitch if self.app.processor else 0,
            "s3_chunk_size": self.app.app_settings.get("s3_chunk_size", 1000),
            "s3_overlap_size": self.app.app_settings.get("s3_overlap_size", 30)
        }

        sqlite_db_path = getattr(self.app, 's2_sqlite_db_path', None)

        s3_results = stage3_module.perform_stage3_analysis(
            video_path=self.app.file_manager.video_path,
            preprocessed_video_path_arg=preprocessed_video_path,
            atr_segments_list=segments_objects,
            s2_frame_objects_map=self.app.s2_frame_objects_map_for_s3,
            tracker_config=tracker_config_s3,
            common_app_config=common_app_config_s3,
            progress_callback=self.on_stage3_progress,
            stop_event=self.stop_stage_event,
            parent_logger=self.logger,
            sqlite_db_path=sqlite_db_path
        )

        # Clear Stage 2 memory map immediately after Stage 3 finishes
        if hasattr(self.app, 's2_frame_objects_map_for_s3') and self.app.s2_frame_objects_map_for_s3:
            map_size = len(self.app.s2_frame_objects_map_for_s3)
            self.app.s2_frame_objects_map_for_s3 = None
            self.logger.info(
                f"[Memory] Cleared Stage 2 data map ({map_size} frames) early to reduce memory pressure")
            gc.collect()

        if self.stop_stage_event.is_set():
            return False

        if s3_results and "error" not in s3_results:
            funscript_obj = s3_results.get("funscript")
            if funscript_obj:
                final_s3_primary_actions = funscript_obj.primary_actions
                final_s3_secondary_actions = funscript_obj.secondary_actions
            else:
                final_s3_primary_actions = s3_results.get("primary_actions", [])
                final_s3_secondary_actions = s3_results.get("secondary_actions", [])

            self.logger.info(
                f"Stage 3 Optical Flow generated {len(final_s3_primary_actions)} primary and "
                f"{len(final_s3_secondary_actions)} secondary actions.")

            range_is_active, range_start_f, range_end_f_effective = fs_proc.get_effective_scripting_range()
            op_desc_s3 = "Stage 3 Opt.Flow"
            video_total_frames_s3 = self.app.processor.total_frames if self.app.processor else 0
            video_duration_ms_s3 = fs_proc.frame_to_ms(
                video_total_frames_s3 - 1) if video_total_frames_s3 > 0 else 0

            if range_is_active:
                start_ms = fs_proc.frame_to_ms(range_start_f if range_start_f is not None else 0)
                end_ms = (fs_proc.frame_to_ms(range_end_f_effective)
                          if range_end_f_effective is not None else video_duration_ms_s3)
                op_desc_s3_range = (
                    f"{op_desc_s3} (Range F{range_start_f or 'Start'}-"
                    f"{range_end_f_effective if range_end_f_effective is not None else 'End'})")
                if final_s3_primary_actions:
                    fs_proc.clear_actions_in_range_and_inject_new(1, final_s3_primary_actions, start_ms, end_ms,
                                                                  op_desc_s3_range + " (T1)")
                else:
                    self.logger.warning("No primary actions from Stage 3 - Timeline 1 range unchanged")
                if final_s3_secondary_actions:
                    fs_proc.clear_actions_in_range_and_inject_new(2, final_s3_secondary_actions, start_ms,
                                                                  end_ms, op_desc_s3_range + " (T2)")
                else:
                    self.logger.info("No secondary actions from Stage 3 - Timeline 2 range unchanged")
            else:
                if final_s3_primary_actions:
                    fs_proc.clear_timeline_history_and_set_new_baseline(1, final_s3_primary_actions,
                                                                        op_desc_s3 + " (T1)")
                else:
                    self.logger.warning("No primary actions from Stage 3 - Timeline 1 unchanged")
                if final_s3_secondary_actions:
                    fs_proc.clear_timeline_history_and_set_new_baseline(2, final_s3_secondary_actions,
                                                                        op_desc_s3 + " (T2)")
                else:
                    self.logger.info("No secondary actions from Stage 3 - Timeline 2 unchanged")

            self.gui_event_queue.put(("stage3_status_update", "Stage 3 Completed.", "Done"))
            self.app.project_manager.project_dirty = True

            if "video_segments" in s3_results:
                fs_proc.video_chapters.clear()
                for seg_data in s3_results["video_segments"]:
                    fs_proc.video_chapters.append(VideoSegment.from_dict(seg_data))
                self.app.app_state_ui.heatmap_dirty = True
                self.app.app_state_ui.funscript_preview_dirty = True
            return s3_results
        else:
            error_msg = s3_results.get("error",
                                        "Unknown S3 failure") if s3_results else "S3 returned None."
            self.logger.error(f"Stage 3 execution failed: {error_msg}")
            self.gui_event_queue.put(("stage3_status_update", f"S3 Failed: {error_msg}", "Failed"))
            return None

    def _resolve_video_fps(self) -> float:
        """Resolve the video FPS from multiple sources with priority ordering."""
        video_fps = 0.0
        fps_source = "none"

        if self.app.processor and hasattr(self.app.processor, 'fps') and self.app.processor.fps > 0:
            video_fps = self.app.processor.fps
            fps_source = "processor.fps"
        elif self.app.processor and self.app.processor.video_info:
            video_fps = self.app.processor.video_info.get('fps', 0.0)
            fps_source = "processor.video_info"
        elif (self.app.project_manager.current_project_data and
              self.app.project_manager.current_project_data.get('video_info')):
            video_fps = self.app.project_manager.current_project_data['video_info'].get('fps', 0.0)
            fps_source = "project_data"

        if video_fps <= 0:
            video_fps = 30.0
            self.logger.warning(
                f"Video FPS not available (source: {fps_source}), using fallback 30.0 FPS. "
                "This may cause incorrect timestamps for 60fps videos!")
        else:
            self.logger.debug(f"Using video FPS: {video_fps} (source: {fps_source})")
        return video_fps

    # ------------------------------------------------------------------
    # Stage 3 — mixed module
    # ------------------------------------------------------------------

    def _execute_stage3_mixed_module(self, segments_objects: List[Any],
                                     preprocessed_video_path: Optional[str],
                                     s2_output_data: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Execute Mixed Stage 3 processing using stage_3_mixed_processor if available."""
        if stage3_mixed_module is None:
            self.logger.error("Stage 3 Mixed module not available.")
            return None
        fs_proc = self.app.funscript_processor
        fm = self.app.file_manager
        if not fm or not fm.video_path:
            self.logger.error("Stage 3 Mixed: Video path not available.")
            return None
        common_app_config = {
            "yolo_det_model_path": self.app.yolo_det_model_path,
            "yolo_pose_model_path": self.app.yolo_pose_model_path,
            "yolo_input_size": self.app.yolo_input_size,
            "video_fps": (self.app.processor.video_info.get('fps', 30.0)
                          if self.app.processor and self.app.processor.video_info else 30.0),
        }
        try:
            self.logger.info(f"Stage 2 data available for mixed mode: {s2_output_data is not None}")
            stage2_funscript = s2_output_data.get("funscript") if s2_output_data else None
            self.logger.info(f"Stage 2 funscript available: {stage2_funscript is not None}")
            if stage2_funscript:
                self.logger.info(
                    f"Stage 2 funscript has primary_actions: {hasattr(stage2_funscript, 'primary_actions')}")
                if hasattr(stage2_funscript, 'primary_actions'):
                    self.logger.info(
                        f"Stage 2 funscript primary_actions count: {len(stage2_funscript.primary_actions)}")

            results = stage3_mixed_module.perform_mixed_stage_analysis(
                video_path=fm.video_path,
                preprocessed_video_path_arg=preprocessed_video_path,
                atr_segments_list=segments_objects,
                s2_frame_objects_map=self.app.s2_frame_objects_map_for_s3 or {},
                tracker_config={},
                common_app_config=common_app_config,
                progress_callback=self.on_stage3_progress,
                stop_event=self.stop_stage_event,
                parent_logger=self.logger,
                sqlite_db_path=getattr(self.app, 's2_sqlite_db_path', None),
                stage2_funscript=stage2_funscript,
            )
            return results
        except Exception as e:
            self.logger.error(f"Stage 3 Mixed execution failed: {e}", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Stage 3 — modular offline tracker (process_stage API)
    # ------------------------------------------------------------------

    def _execute_stage3_modular_tracker(self, tracker_name: str,
                                        segments_objects: List[Any],
                                        preprocessed_video_path: Optional[str]) -> Optional[Dict[str, Any]]:
        """Execute Stage 3 via a modular offline tracker's process_stage() method."""
        from tracker.tracker_modules import create_tracker
        from tracker.tracker_modules.core.base_offline_tracker import OfflineProcessingStage

        fm = self.app.file_manager
        if not fm or not fm.video_path:
            self.logger.error("Stage 3 modular: Video path not available.")
            return None

        tracker = create_tracker(tracker_name)
        if tracker is None:
            self.logger.error(f"Stage 3 modular: Could not create tracker '{tracker_name}'")
            return None

        if not tracker.initialize(self.app):
            self.logger.error(f"Stage 3 modular: Tracker '{tracker_name}' initialization failed")
            return None

        # Set stop event so tracker can check for cancellation
        if hasattr(tracker, 'set_stop_event'):
            tracker.set_stop_event(self.stop_stage_event)

        # Build input_files for the tracker
        s2_overlay_path = getattr(fm, 'stage2_output_msgpack_path', None)
        if not s2_overlay_path:
            # Try to derive from video path
            try:
                s2_overlay_path = fm.get_output_path_for_file(fm.video_path, "_stage2_overlay.msgpack")
            except Exception:
                pass

        input_files = {}
        if s2_overlay_path and os.path.exists(s2_overlay_path):
            input_files['stage2'] = s2_overlay_path
            input_files['stage2_output'] = s2_overlay_path
        # Pass SQLite path for trackers that can use it (e.g. Guided Flow)
        s2_sqlite_path = getattr(self.app, 's2_sqlite_db_path', None)
        if s2_sqlite_path and os.path.exists(s2_sqlite_path):
            input_files['stage2_sqlite'] = s2_sqlite_path
        if preprocessed_video_path and os.path.exists(preprocessed_video_path):
            input_files['preprocessed_video'] = preprocessed_video_path

        try:
            result = tracker.process_stage(
                stage=OfflineProcessingStage.STAGE_3,
                video_path=fm.video_path,
                input_files=input_files,
                progress_callback=self.on_stage3_progress,
            )

            if result and result.success and result.output_data:
                return result.output_data
            else:
                error_msg = getattr(result, 'error_message', 'Unknown error') if result else 'No result'
                self.logger.error(f"Stage 3 modular tracker failed: {error_msg}")
                return None
        except Exception as e:
            self.logger.error(f"Stage 3 modular tracker execution failed: {e}", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Hybrid tracker — handles Stage 1 + 2 internally
    # ------------------------------------------------------------------

    def _execute_hybrid_tracker(self, tracker_name: str) -> Dict[str, Any]:
        """Execute a hybrid tracker that handles both Stage 1 and Stage 2 internally."""
        from tracker.tracker_modules import create_tracker
        from tracker.tracker_modules.core.base_offline_tracker import OfflineProcessingStage

        fm = self.app.file_manager
        if not fm or not fm.video_path:
            return {"success": False, "error": "Video path not available"}

        tracker = create_tracker(tracker_name)
        if tracker is None:
            return {"success": False, "error": f"Could not create tracker '{tracker_name}'"}

        if not tracker.initialize(self.app):
            return {"success": False, "error": f"Tracker '{tracker_name}' initialization failed"}

        # Set stop event
        if hasattr(tracker, 'set_stop_event'):
            tracker.set_stop_event(self.stop_stage_event)

        output_directory = os.path.dirname(fm.get_output_path_for_file(fm.video_path, "_dummy.tmp"))
        preprocessed_path = fm.get_output_path_for_file(fm.video_path, "_preprocessed.mp4")

        save_preprocessed = getattr(self, 'save_preprocessed_video', False)
        hwaccel = getattr(self.app, 'hardware_acceleration_method', 'auto')

        self.gui_event_queue.put(("stage2_status_update", "Running Hybrid...", "Initializing"))

        is_headless = not getattr(self.app, 'gui_instance', None)

        def progress_wrapper(info):
            """Route hybrid tracker progress to the Stage 2 UI (or CLI stdout)."""
            if isinstance(info, dict):
                stage = info.get('stage', '')
                task = info.get('task', '')
                pct = info.get('percentage', 0)
                self.stage2_main_progress_value = pct / 100.0
                self.stage2_main_progress_label = f"{stage}: {task}" if stage else task
                self.stage2_status_text = f"{stage}: {task}"
                self.gui_event_queue.put(("stage2_status_update", f"{task}", f"{pct}%"))

                if is_headless:
                    import sys
                    bar_w = 40
                    filled = int(pct / 100.0 * bar_w)
                    bar = '\u2588' * filled + '-' * (bar_w - filled)
                    sys.stdout.write(f"\rHybrid: |{bar}| {pct:>3}% | {stage}: {task}   ")
                    sys.stdout.flush()
                    if pct >= 100:
                        sys.stdout.write("\n")

        try:
            result = tracker.process_stage(
                stage=OfflineProcessingStage.STAGE_2,
                video_path=fm.video_path,
                output_directory=output_directory,
                progress_callback=progress_wrapper,
                save_preprocessed_video=save_preprocessed,
                preprocessed_video_path=preprocessed_path,
                hwaccel_method=hwaccel,
            )

            if self.stop_stage_event.is_set():
                # Clean up incomplete preprocessed file on abort
                if os.path.exists(preprocessed_path) and not save_preprocessed:
                    try:
                        os.remove(preprocessed_path)
                    except OSError:
                        pass
                return {"success": False, "error": "Aborted by user"}

            if result and result.success and result.output_data:
                funscript = result.output_data.get('funscript', {})
                chapters = result.output_data.get('chapters', [])

                # Build Stage 2-compatible output data with funscript actions
                output_data = {
                    'funscript': funscript,
                    'chapters': chapters,
                    'video_segments': [
                        {
                            'label': ch.get('position', 'Unknown'),
                            'start_frame': ch.get('start_frame', 0),
                            'end_frame': ch.get('end_frame', 0),
                        }
                        for ch in chapters
                    ],
                }

                # Manage preprocessed video file
                if save_preprocessed and os.path.exists(preprocessed_path):
                    fm.preprocessed_video_path = preprocessed_path
                    self.logger.info(f"Preprocessed video saved: {os.path.basename(preprocessed_path)}")
                elif not save_preprocessed and os.path.exists(preprocessed_path):
                    try:
                        os.remove(preprocessed_path)
                        self.logger.info("Deleted preprocessed video (save_preprocessed_video=off)")
                    except OSError as del_err:
                        self.logger.warning(f"Could not delete preprocessed video: {del_err}")

                self.gui_event_queue.put(("stage2_status_update", "Hybrid Complete", "Done"))
                return {"success": True, "data": output_data}
            else:
                error_msg = getattr(result, 'error_message', 'Unknown error') if result else 'No result'
                return {"success": False, "error": error_msg}

        except Exception as e:
            self.logger.error(f"Hybrid tracker execution failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _has_modular_stage3(self, tracker_name: str) -> bool:
        """Check if the named tracker has its own process_stage method (new offline API)."""
        from tracker.tracker_modules import tracker_registry
        tracker_class = tracker_registry.get_tracker(tracker_name)
        if tracker_class is None:
            return False
        # Check if it's an offline tracker with process_stage (not the legacy perform_stage3_analysis)
        return (hasattr(tracker_class, 'process_stage')
                and hasattr(tracker_class, 'processing_stages'))

    # ------------------------------------------------------------------
    # Artifact validation
    # ------------------------------------------------------------------

    def _validate_preprocessed_artifacts(self, msgpack_path: str, video_path: str) -> bool:
        """Validates that preprocessed artifacts are complete and consistent."""
        try:
            from detection.cd.stage_1_cd import (_validate_preprocessed_file_completeness,
                                                  _validate_preprocessed_video_completeness)

            if not self.app.processor or not self.app.processor.video_info:
                self.logger.warning("Cannot validate preprocessed artifacts: video info not available")
                return False

            expected_frames = self.app.processor.video_info.get('total_frames', 0)
            expected_fps = self.app.processor.video_info.get('fps', 30.0)

            if expected_frames <= 0:
                self.logger.warning("Cannot validate preprocessed artifacts: invalid frame count")
                return False

            if not _validate_preprocessed_file_completeness(msgpack_path, expected_frames, self.logger):
                self.logger.warning(
                    f"Preprocessed msgpack validation failed: {os.path.basename(msgpack_path)}")
                return False

            if os.path.exists(video_path):
                if not _validate_preprocessed_video_completeness(video_path, expected_frames, expected_fps,
                                                                  self.logger):
                    self.logger.warning(
                        f"Preprocessed video validation failed: {os.path.basename(video_path)}")
                    return False

            self.logger.info("Preprocessed artifacts validation passed")
            return True
        except Exception as e:
            self.logger.error(f"Error validating preprocessed artifacts: {e}")
            return False
