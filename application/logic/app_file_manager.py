import os
import json
import orjson
import msgpack
import time
from typing import List, Optional, Dict, Tuple

from application.utils.video_segment import VideoSegment
from config.constants import PROJECT_FILE_EXTENSION, AUTOSAVE_FILE, DEFAULT_CHAPTER_FPS, APP_VERSION, FUNSCRIPT_METADATA_VERSION


class AppFileManager:
    def __init__(self, app_logic_instance):
        self.app = app_logic_instance
        self.logger = self.app.logger
        self.app_settings = self.app.app_settings

        self.video_path: str = ""
        self.funscript_path: str = ""
        self.loaded_funscript_path: str = ""
        self.stage1_output_msgpack_path: Optional[str] = None
        self.stage2_output_msgpack_path: Optional[str] = None
        self.last_dropped_files: Optional[List[str]] = None

    def _set_yolo_model_path_callback(self, filepath: str, model_type: str):
        """Callback for setting YOLO model paths from file dialogs."""
        if model_type == "detection":
            self.app.yolo_detection_model_path_setting = filepath
            self.app.yolo_det_model_path = filepath
            if self.app.tracker:
                self.app.tracker.det_model_path = filepath
                self.app.save_app_settings()
            self.logger.info(f"Stage 1 YOLO Detection model path set: {os.path.basename(filepath)}",
                             extra={'status_message': True})
        elif model_type == "pose":
            self.app.yolo_pose_model_path_setting = filepath
            self.app.yolo_pose_model_path = filepath
            if self.app.tracker:
                self.app.tracker.pose_model_path = filepath
                self.app.save_app_settings()
            self.logger.info(f"YOLO Pose model path set: {os.path.basename(filepath)}", extra={'status_message': True})
        self.app.project_manager.project_dirty = True
        self.app.energy_saver.reset_activity_timer()

    def get_output_path_for_file(self, video_path: str, file_suffix: str) -> str:
        """
        Generates a full, absolute path for an output file within a video-specific subfolder
        inside the main configured output directory.
        """
        if not video_path:
            self.logger.error("Cannot get output path: video_path is empty.")
            return f"error_no_video_path{file_suffix}"

        output_folder_base = self.app.app_settings.get("output_folder_path", "output")
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        video_specific_output_dir = os.path.join(output_folder_base, video_basename)

        try:
            os.makedirs(video_specific_output_dir, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Could not create output directory '{video_specific_output_dir}': {e}")
            video_specific_output_dir = output_folder_base
            os.makedirs(video_specific_output_dir, exist_ok=True)

        final_filename = video_basename + file_suffix
        return os.path.join(video_specific_output_dir, final_filename)

    def _parse_funscript_file(self, funscript_file_path: str) -> Tuple[Optional[List[Dict]], Optional[str], Optional[List[Dict]], Optional[float]]:
        """ Parses a funscript file using the high-performance orjson library. """
        try:
            with open(funscript_file_path, 'rb') as f:  # <-- Read in binary mode 'rb'
                data = orjson.loads(f.read())  # <-- Use orjson.loads

            actions_data = data.get("actions", [])
            if not isinstance(actions_data, list):
                return None, f"Invalid format: 'actions' is not a list in {os.path.basename(funscript_file_path)}.", None, None

            valid_actions = []
            for action in actions_data:
                if isinstance(action, dict) and "at" in action and "pos" in action:
                    try:
                        action["at"] = int(action["at"])
                        action["pos"] = int(action["pos"])
                        action["pos"] = min(max(action["pos"], 0), 100)
                        valid_actions.append(action)
                    except (ValueError, TypeError):  # orjson might raise TypeError
                        self.logger.warning(f"Skipping action with invalid at/pos types: {action}",
                                            extra={'status_message': False})
                else:
                    self.logger.warning(f"Skipping invalid action format: {action}", extra={'status_message': False})

            parsed_actions = sorted(valid_actions, key=lambda x: x["at"]) if valid_actions else []

            chapters_list_of_dicts = []
            chapters_fps_from_file: Optional[float] = None
            if "metadata" in data and isinstance(data["metadata"], dict):
                metadata = data["metadata"]
                if "chapters_fps" in metadata and isinstance(metadata["chapters_fps"], (int, float)):
                    chapters_fps_from_file = float(metadata["chapters_fps"])
                if "chapters" in metadata and isinstance(metadata["chapters"], list):
                    for chap_data_item in metadata["chapters"]:
                        if isinstance(chap_data_item,
                                      dict) and "name" in chap_data_item and "startTime" in chap_data_item and "endTime" in chap_data_item:
                            chapters_list_of_dicts.append(chap_data_item)
                        else:
                            self.logger.warning(f"Skipping malformed chapter data in Funscript: {chap_data_item}",
                                                extra={'status_message': True})
                    if chapters_list_of_dicts:
                        self.logger.info(
                            f"Found {len(chapters_list_of_dicts)} chapter entries in metadata of {os.path.basename(funscript_file_path)}.")

            return parsed_actions, None, chapters_list_of_dicts, chapters_fps_from_file
        except FileNotFoundError:
            return None, f"File not found: {os.path.basename(funscript_file_path)}", None, None
        except orjson.JSONDecodeError:  # <-- Catch the specific orjson exception
            return None, f"Error decoding JSON from {os.path.basename(funscript_file_path)}.", None, None
        except Exception as e:
            self.logger.error(f"Unexpected error loading funscript '{funscript_file_path}': {e}", exc_info=True,
                              extra={'status_message': True})
            return None, f"Error loading funscript: {str(e)}", None, None

    def save_raw_funscripts_after_generation(self, video_path: str):
        if not self.app.funscript_processor: return
        if not video_path: return

        primary_actions = self.app.funscript_processor.get_actions('primary')
        secondary_actions = self.app.funscript_processor.get_actions('secondary')
        chapters = self.app.funscript_processor.video_chapters
        self.logger.info("Saving raw (pre-post-processing) funscript backup to output folder...")

        if primary_actions:
            primary_path = self.get_output_path_for_file(video_path, "_t1_raw.funscript")
            self._save_funscript_file(primary_path, primary_actions, chapters)
        if secondary_actions:
            secondary_path = self.get_output_path_for_file(video_path, "_t2_raw.funscript")
            self._save_funscript_file(secondary_path, secondary_actions, None)



    def load_funscript_to_timeline(self, funscript_file_path: str, timeline_num: int = 1):
        actions, error_msg, chapters_as_dicts, chapters_fps_from_file = self._parse_funscript_file(funscript_file_path)
        funscript_processor = self.app.funscript_processor

        if error_msg:
            self.logger.error(error_msg, extra={'status_message': True})
            return

        if actions is None:  # Should be caught by error_msg, but as a safeguard
            self.logger.error(f"Failed to parse actions from {os.path.basename(funscript_file_path)}.",
                              extra={'status_message': True})
            return

        if not (self.app.processor and self.app.processor.tracker and self.app.processor.tracker.funscript):
            self.logger.warning(f"Cannot load to Timeline {timeline_num}: Tracker or Funscript object not available.",
                                extra={'status_message': True})
            return

        desc = f"Load T{timeline_num}: {os.path.basename(funscript_file_path)}"
        funscript_processor.clear_timeline_history_and_set_new_baseline(timeline_num, actions, desc)

        if timeline_num == 1:
            self.loaded_funscript_path = funscript_file_path  # T1's own loaded script
            self.funscript_path = funscript_file_path  # Project associated script (if T1)
            self.logger.info(
                f"Loaded {len(actions)} actions to Timeline 1 from {os.path.basename(funscript_file_path)}",
                extra={'status_message': True})

            # Load chapters only when loading to T1 and if video is present for FPS context
            if chapters_as_dicts:
                funscript_processor.video_chapters.clear()
                fps_for_conversion = DEFAULT_CHAPTER_FPS
                if chapters_fps_from_file and chapters_fps_from_file > 0:
                    fps_for_conversion = chapters_fps_from_file
                elif self.app.processor and self.app.processor.video_info and self.app.processor.fps > 0:
                    fps_for_conversion = self.app.processor.fps

                if fps_for_conversion <= 0:
                    self.logger.error(
                        f"Cannot convert chapter timecodes: FPS for conversion is invalid ({fps_for_conversion:.2f}). Chapters will not be loaded.",
                        extra={'status_message': True})
                else:
                    for chap_data in chapters_as_dicts:
                        try:
                            segment = VideoSegment.from_funscript_chapter_dict(chap_data, fps_for_conversion)
                            funscript_processor.video_chapters.append(segment)
                        except Exception as e:
                            self.logger.error(f"Error creating VideoSegment from Funscript chapter: {e}",
                                              extra={'status_message': True})
                    if funscript_processor.video_chapters:
                        funscript_processor.video_chapters.sort(key=lambda s: s.start_frame_id)
                        self.logger.info(
                            f"Loaded {len(funscript_processor.video_chapters)} chapters from {os.path.basename(funscript_file_path)} using FPS {fps_for_conversion:.2f}.")
            self.app.app_state_ui.heatmap_dirty = True
            self.app.app_state_ui.funscript_preview_dirty = True

        elif timeline_num == 2:
            self.logger.info(
                f"Loaded {len(actions)} actions to Timeline 2 from {os.path.basename(funscript_file_path)}",
                extra={'status_message': True})

        self.app.project_manager.project_dirty = True
        self.app.energy_saver.reset_activity_timer()

    def _get_funscript_data(self, filepath: str) -> Optional[Dict]:
        """Safely reads and returns the entire parsed dictionary from a funscript file."""
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, 'rb') as f:
                data = orjson.loads(f.read())
            return data
        except Exception as e:
            self.logger.warning(f"Could not parse funscript data from file: {filepath}. Error: {e}")
            return None

    def _save_funscript_file(self, filepath: str, actions: List[Dict], chapters: Optional[List[VideoSegment]] = None):
        """
        A centralized, high-performance method to save a single funscript file.
        This is the single source of truth for funscript saving.
        """
        if not actions:
            self.logger.info(f"No actions to save to {os.path.basename(filepath)}.", extra={'status_message': True})
            return

        # --- Backup logic before saving ---
        if os.path.exists(filepath):
            try:
                # Create a unique backup filename with a Unix timestamp
                backup_path = f"{filepath}.{int(time.time())}.bak"
                os.rename(filepath, backup_path)
                self.logger.info(f"Created backup of existing file: {os.path.basename(backup_path)}")
            except Exception as e:
                self.logger.error(f"Failed to create backup for {os.path.basename(filepath)}: {e}")
                # We can decide whether to proceed with the overwrite or not.
                # For safety, let's proceed but the user is warned.

        funscript_data = {
            "version": "1.0",
            "author": f"FunGen beta {APP_VERSION}",
            "inverted": False,
            "range": 100,
            "actions": sorted(actions, key=lambda x: x["at"]),  # Ensure sorted
            "metadata": {"chapters": []}  # Default empty metadata
        }

        # Add chapter data to metadata if chapters are provided
        if chapters:
            current_fps = DEFAULT_CHAPTER_FPS
            if self.app.processor and self.app.processor.video_info and self.app.processor.fps > 0:
                current_fps = self.app.processor.fps
            else:
                self.logger.warning(
                    f"Video FPS not available for saving chapters in timecode format. Using default FPS: {DEFAULT_CHAPTER_FPS}. Timecodes may be inaccurate.",
                    extra={'status_message': True})

            funscript_data["metadata"] = {
                "version": f"{FUNSCRIPT_METADATA_VERSION}",
                "chapters_fps": current_fps,
                "chapters": [chapter.to_funscript_chapter_dict(current_fps) for chapter in chapters]
            }

        try:
            # Use orjson for high-performance writing
            with open(filepath, 'wb') as f:
                f.write(orjson.dumps(funscript_data))
            self.logger.info(f"Funscript saved to {os.path.basename(filepath)}",
                             extra={'status_message': True})
        except Exception as e:
            self.logger.error(f"Error saving funscript to '{filepath}': {e}",
                              extra={'status_message': True})

    def save_funscript_from_timeline(self, filepath: str, timeline_num: int):
        funscript_processor = self.app.funscript_processor
        actions = funscript_processor.get_actions('primary' if timeline_num == 1 else 'secondary')

        # Chapters are only saved for timeline 1
        chapters = funscript_processor.video_chapters if timeline_num == 1 else None

        # Call the centralized saving method
        self._save_funscript_file(filepath, actions, chapters)

        if timeline_num == 1:
            self.funscript_path = filepath
            self.loaded_funscript_path = filepath

        self.app.energy_saver.reset_activity_timer()

    def save_funscripts_for_batch(self, video_path: str):
        """
        Automatically saves funscripts next to the video file using the centralized saver.
        This now correctly includes all metadata.
        """
        if not self.app.funscript_processor:
            self.app.logger.error("Funscript processor not available for saving.")
            return

        base, _ = os.path.splitext(video_path)
        primary_actions = self.app.funscript_processor.get_actions('primary')
        secondary_actions = self.app.funscript_processor.get_actions('secondary')
        chapters = self.app.funscript_processor.video_chapters
        save_next_to_video = self.app.app_settings.get("autosave_final_funscript_to_video_location", True)

        if primary_actions:
            if save_next_to_video:
                base, _ = os.path.splitext(video_path)
                primary_path = f"{base}_t1.funscript"
            else:
                primary_path = self.get_output_path_for_file(video_path, "_t1.funscript")
            self._save_funscript_file(primary_path, primary_actions, chapters)

        if secondary_actions:
            if save_next_to_video:
                base, _ = os.path.splitext(video_path)
                secondary_path = f"{base}_t2.funscript"
            else:
                secondary_path = self.get_output_path_for_file(video_path, "_t2.funscript")
            self._save_funscript_file(secondary_path, secondary_actions, None)

    def handle_video_file_load(self, file_path: str, is_project_load=False):
        self.video_path = file_path
        funscript_processor = self.app.funscript_processor
        stage_processor = self.app.stage_processor

        if not is_project_load:
            self.funscript_path = ""
            self.loaded_funscript_path = ""
            stage_processor.reset_stage1_status()
            stage_processor.reset_stage2_status()
            self.clear_stage2_overlay_data()
            funscript_processor.video_chapters.clear()
            self.app.app_state_ui.heatmap_dirty = True
            self.app.app_state_ui.funscript_preview_dirty = True

            potential_s1_path = self.get_output_path_for_file(self.video_path, ".msgpack")
            if os.path.exists(potential_s1_path):
                self.stage1_output_msgpack_path = potential_s1_path
                self.app.stage_processor.stage1_status_text = f"Found: {os.path.basename(potential_s1_path)}"
                self.app.stage_processor.stage1_progress_value = 1.0

            potential_s2_overlay = self.get_output_path_for_file(self.video_path, "_stage2_overlay.msgpack")
            if os.path.exists(potential_s2_overlay):
                self.load_stage2_overlay_data(potential_s2_overlay)


            if not self.loaded_funscript_path:
                funscript_processor.clear_timeline_history_and_set_new_baseline(1, [], "New Video (T1 Cleared)")
                funscript_processor.clear_timeline_history_and_set_new_baseline(2, [], "New Video (T2 Cleared)")

        if self.app.processor:
            if self.app.processor.open_video(file_path, from_project_load=is_project_load):
                if not is_project_load:
                    path_in_output = self.get_output_path_for_file(file_path, ".funscript")
                    path_next_to_video = os.path.splitext(file_path)[0] + ".funscript"

                    funscript_to_load = None
                    if os.path.exists(path_in_output):
                        funscript_to_load = path_in_output
                    elif os.path.exists(path_next_to_video):
                        funscript_to_load = path_next_to_video

                    if funscript_to_load:
                        self.load_funscript_to_timeline(funscript_to_load, timeline_num=1)

    def close_video_action(self, clear_funscript_unconditionally=False):
        if self.app.processor:
            if self.app.processor.is_processing: self.app.processor.stop_processing()
            self.app.processor.reset(close_video=True)  # Resets video info in processor

        self.video_path = ""
        self.app.stage_processor.reset_stage1_status()
        self.app.stage_processor.reset_stage2_status()
        self.app.funscript_processor.video_chapters.clear()
        self.clear_stage2_overlay_data()

        # If funscript was loaded from a file (not generated) and we are not clearing unconditionally, keep T1.
        # Otherwise, clear T1. Always clear T2.
        if clear_funscript_unconditionally or not self.loaded_funscript_path:  # loaded_funscript_path is for T1
            if self.app.processor and self.app.processor.tracker and self.app.processor.tracker.funscript:
                self.app.funscript_processor.clear_timeline_history_and_set_new_baseline(1, [],
                                                                                         "Video Closed (T1 Cleared)")
            self.funscript_path = ""  # Project association
            self.loaded_funscript_path = ""  # T1 specific

        # Always clear T2 on video close unless a specific logic dictates otherwise
        if self.app.processor and self.app.processor.tracker and self.app.processor.tracker.funscript:
            self.app.funscript_processor.clear_timeline_history_and_set_new_baseline(2, [], "Video Closed (T2 Cleared)")

        self.app.funscript_processor.update_funscript_stats_for_timeline(1, "Video Closed")
        self.app.funscript_processor.update_funscript_stats_for_timeline(2, "Video Closed")

        self.logger.info("Video closed.", extra={'status_message': True})
        self.app.energy_saver.reset_activity_timer()
        self.app.app_state_ui.heatmap_dirty = True
        self.app.app_state_ui.funscript_preview_dirty = True
        self.app.project_manager.project_dirty = True

    def load_stage2_overlay_data(self, filepath: str):
        self.clear_stage2_overlay_data()  # Clear previous before loading new
        stage_processor = self.app.stage_processor
        try:
            with open(filepath, 'rb') as f:
                packed_data = f.read()
            loaded_data = msgpack.unpackb(packed_data, raw=False)

            if isinstance(loaded_data, list):
                stage_processor.stage2_overlay_data = loaded_data
                # Create a map for quick lookup by frame_id
                stage_processor.stage2_overlay_data_map = {
                    frame_data.get("frame_id", -1): frame_data
                    for frame_data in stage_processor.stage2_overlay_data if isinstance(frame_data, dict)
                }
                self.stage2_output_msgpack_path = filepath  # Store path of loaded overlay

                if loaded_data:
                    stage_processor.stage2_status_text = f"Overlay loaded: {os.path.basename(filepath)}"
                    # stage_processor.stage2_progress_value = 1.0 # No, this is for S2 run
                    # stage_processor.stage2_progress_label = f"{len(stage_processor.stage2_overlay_data)} frames"
                    self.logger.info(
                        f"Loaded Stage 2 overlay: {os.path.basename(filepath)} ({len(stage_processor.stage2_overlay_data)} frames)",
                        extra={'status_message': True})
                    self.app.app_state_ui.show_stage2_overlay = True
                else:
                    stage_processor.stage2_status_text = f"Overlay file empty: {os.path.basename(filepath)}"
                    self.app.app_state_ui.show_stage2_overlay = False
                    self.logger.warning(f"Stage 2 overlay file is empty: {os.path.basename(filepath)}",
                                        extra={'status_message': True})

                self.app.project_manager.project_dirty = True
                self.app.energy_saver.reset_activity_timer()
            else:
                stage_processor.stage2_status_text = "Error: Overlay not list format"
                self.app.app_state_ui.show_stage2_overlay = False
                self.logger.error("Stage 2 overlay data is not in expected list format.",
                                  extra={'status_message': True})
        except Exception as e:
            stage_processor.stage2_status_text = "Error loading overlay"
            self.app.app_state_ui.show_stage2_overlay = False
            self.logger.error(f"Error loading Stage 2 overlay msgpack '{filepath}': {e}",
                              extra={'status_message': True})

    def clear_stage2_overlay_data(self):
        stage_processor = self.app.stage_processor
        stage_processor.stage2_overlay_data = None
        stage_processor.stage2_overlay_data_map = None
        self.stage2_output_msgpack_path = None  # Clear path if data is cleared

    def open_video_from_path(self, file_path: str) -> bool:
        """
        Opens a video file, updates the application state, and returns success.
        This is the central method for loading a video.
        """
        if not file_path or not os.path.exists(file_path):
            self.app.logger.error(f"Video file not found: {file_path}")
            return False

        self.app.logger.info(f"Opening video: {os.path.basename(file_path)}", extra={'status_message': True})

        # Reset relevant states before loading a new video
        self.close_video_action(clear_funscript_unconditionally=False)

        # Call the core video opening logic in the VideoProcessor
        success = self.app.processor.open_video(file_path)

        if success:
            self.video_path = file_path
            self.app.project_manager.project_dirty = True
            # Reset UI states for the new video
            self.app.app_state_ui.reset_video_zoom_pan()
            self.app.app_state_ui.force_timeline_pan_to_current_frame = True
            self.app.funscript_processor.update_funscript_stats_for_timeline(1, "Video Loaded")
            self.app.funscript_processor.update_funscript_stats_for_timeline(2, "Video Loaded")
        else:
            self.video_path = ""
            self.app.logger.error(f"Failed to open video file: {os.path.basename(file_path)}",
                                  extra={'status_message': True})

        return success

    def _scan_folder_for_videos(self, folder_path: str) -> List[str]:
        """Recursively scans a folder for video files."""
        video_files = []
        valid_extensions = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
        self.app.logger.info(f"Scanning folder: {folder_path}")
        for root, _, files in os.walk(folder_path):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_extensions:
                    video_files.append(os.path.join(root, file))
        return sorted(video_files)

    def handle_drop_event(self, paths: List[str]):
        """
        Handles dropped files or folders. If a single video is found, it's opened directly.
        If multiple videos are found, it triggers the batch processing confirmation.
        """
        if not paths:
            return

        videos_for_batch = []
        other_files = []
        valid_video_extensions = {".mp4", ".mkv", ".mov", ".avi", ".webm"}

        # Categorize all dropped paths
        for path in paths:
            if os.path.isdir(path):
                # If a directory is dropped, scan it and its subfolders for videos
                self.app.logger.info(f"Scanning dropped folder for videos: {path}")
                videos_for_batch.extend(self._scan_folder_for_videos(path))
            elif os.path.splitext(path)[1].lower() in valid_video_extensions:
                # If a video file is dropped, add it to the list for processing
                videos_for_batch.append(path)
            else:
                # Keep track of other non-video file types
                other_files.append(path)

        # Ensure the list of videos is unique and sorted
        unique_videos = sorted(list(set(videos_for_batch)))

        if len(unique_videos) > 1:
            # If more than one video was found, start the batch processing workflow
            self.app.start_batch_processing(unique_videos)
        elif len(unique_videos) == 1:
            # If exactly one video was found, open it directly without batch confirmation
            self.app.logger.info(f"Single video dropped. Opening directly: {os.path.basename(unique_videos[0])}")
            self.open_video_from_path(unique_videos[0])
        elif other_files:
            # If no videos were found, fall back to the original logic for handling other file types
            self.app.logger.info("No videos found for batching, handling as single file drop.")
            path = other_files[0]
            ext = os.path.splitext(path)[1].lower()
            if ext == '.funscript':
                self.load_funscript_to_timeline(path, 1)
            elif ext == PROJECT_FILE_EXTENSION:
                self.app.project_manager.load_project(path)
            elif ext == '.msgpack':
                self.load_stage2_overlay_data(path)
            else:
                self.last_dropped_files = other_files
                self.app.logger.warning(f"Unrecognized file type dropped: {os.path.basename(path)}",
                                        extra={'status_message': True})


    def update_settings_from_app(self):
        """Called by AppLogic to reflect loaded settings or project data."""
        # Model paths are handled by AppLogic's _apply_loaded_settings directly
        # Stage output paths are mostly managed by project load/save and stage runs
        pass

    def save_settings_to_app(self):
        """Called by AppLogic when app settings are saved."""
        # Model paths are handled by AppLogic's save_app_settings directly
        pass

    def save_final_funscripts(self, video_path: str, chapters: Optional[List[Dict]] = None):
        """
        Saves the final (potentially post-processed) funscripts.
        Adheres to the 'autosave_final_funscript_to_video_location' setting.
        """
        if not self.app.funscript_processor:
            self.logger.error("Funscript processor not available for saving final funscripts.")
            return

        save_next_to_video = self.app.app_settings.get("autosave_final_funscript_to_video_location", True)
        if self.app.is_batch_processing_active:
            save_next_to_video = self.app.batch_copy_funscript_to_video_location


        primary_actions = self.app.funscript_processor.get_actions('primary')
        secondary_actions = self.app.funscript_processor.get_actions('secondary')

        chapters_to_save = []
        if chapters is not None:
            chapters_to_save = [VideoSegment.from_dict(chap_data) for chap_data in chapters if
                                isinstance(chap_data, dict)]
        else:
            chapters_to_save = self.app.funscript_processor.video_chapters

        # Always save to the output directory
        if primary_actions:
            path_in_output = self.get_output_path_for_file(video_path, ".funscript")
            self._save_funscript_file(path_in_output, primary_actions, chapters_to_save)

        # 1. Start with the global setting as the default.
        generate_roll = self.app.app_settings.get("generate_roll_file", True)
        # 2. If in batch mode, override with the specific choice made for that batch.
        if self.app.is_batch_processing_active:
            generate_roll = self.app.batch_generate_roll_file

        # --- Main funscript saving ---
        if primary_actions:
            path_in_output = self.get_output_path_for_file(video_path, ".funscript")
            self._save_funscript_file(path_in_output, primary_actions, chapters_to_save)

        # --- Roll funscript saving now respects the final 'generate_roll' value ---
        if secondary_actions and generate_roll:
            path_in_output_t2 = self.get_output_path_for_file(video_path, ".roll.funscript")
            self._save_funscript_file(path_in_output_t2, secondary_actions, None)

        # Additionally, save next to the video if the setting is enabled
        if save_next_to_video:
            self.logger.info("Also saving a copy of the final funscript next to the video file.")
            base, _ = os.path.splitext(video_path)
            if primary_actions:
                path_next_to_vid = f"{base}.funscript"
                self._save_funscript_file(path_next_to_vid, primary_actions, chapters_to_save)
            if secondary_actions and generate_roll:
                path_next_to_vid_t2 = f"{base}.roll.funscript"
                self._save_funscript_file(path_next_to_vid_t2, secondary_actions, None)


    