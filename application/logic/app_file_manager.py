import os
import json
import orjson
import msgpack
import time
from typing import List, Optional, Dict, Tuple, Any

from application.utils import VideoSegment, check_write_access
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
        self.preprocessed_video_path: Optional[str] = None
        self.last_dropped_files: Optional[List[str]] = None

    def _set_yolo_model_path_callback(self, filepath: str, model_type: str):
        """Callback for setting YOLO model paths from file dialogs."""
        if model_type == "detection":
            # Use the settings manager to set and persist the new path immediately.
            self.app.app_settings.set("yolo_det_model_path", filepath)

            # Also update the live application state.
            self.app.yolo_detection_model_path_setting = filepath
            self.app.yolo_det_model_path = filepath
            if self.app.tracker:
                self.app.tracker.det_model_path = filepath

            self.logger.info(f"Stage 1 YOLO Detection model path set: {os.path.basename(filepath)}",
                             extra={'status_message': True})
        elif model_type == "pose":
            # Use the settings manager to set and persist the new path immediately.
            self.app.app_settings.set("yolo_pose_model_path", filepath)

            # Also update the live application state.
            self.app.yolo_pose_model_path_setting = filepath
            self.app.yolo_pose_model_path = filepath
            if self.app.tracker:
                self.app.tracker.pose_model_path = filepath

            self.logger.info(f"YOLO Pose model path set: {os.path.basename(filepath)}", extra={'status_message': True})

        # Mark the project as dirty because this setting can also be saved per-project.
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
        # Ensure the returned path is always absolute
        return os.path.abspath(os.path.join(video_specific_output_dir, final_filename))

    def _parse_funscript_file(self, funscript_file_path: str) -> Tuple[Optional[List[Dict]], Optional[str], Optional[List[Dict]], Optional[float]]:
        """ Parses a funscript file using the high-performance orjson library. """
        try:
            with open(funscript_file_path, 'rb') as f:
                data = orjson.loads(f.read())

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

    def save_raw_funscripts_next_to_video(self, video_path: str):
        """Save raw funscripts next to the video file with .raw.funscript extension."""
        if not self.app.funscript_processor: return
        if not video_path: return

        primary_actions = self.app.funscript_processor.get_actions('primary')
        secondary_actions = self.app.funscript_processor.get_actions('secondary')
        chapters = self.app.funscript_processor.video_chapters
        
        # Check if copy to video location is enabled
        save_next_to_video = self.app.app_settings.get("autosave_final_funscript_to_video_location", True)
        if self.app.is_batch_processing_active:
            save_next_to_video = self.app.batch_copy_funscript_to_video_location
        
        if not save_next_to_video:
            self.logger.info("Copy to video location is disabled. Raw funscripts saved only to output folder.")
            return
        
        self.logger.info("Saving raw funscripts next to video file with .raw.funscript extension...")

        if primary_actions:
            base, _ = os.path.splitext(video_path)
            primary_path = f"{base}.raw.funscript"
            self._save_funscript_file(primary_path, primary_actions, chapters)
            self.logger.info(f"Raw primary funscript saved: {os.path.basename(primary_path)}")
        
        # Determine roll generation setting
        generate_roll = self.app.app_settings.get("generate_roll_file", True)
        if self.app.is_batch_processing_active:
            generate_roll = self.app.batch_generate_roll_file
            
        if secondary_actions and generate_roll:
            base, _ = os.path.splitext(video_path)
            secondary_path = f"{base}.raw.roll.funscript"
            self._save_funscript_file(secondary_path, secondary_actions, None)
            self.logger.info(f"Raw secondary funscript saved: {os.path.basename(secondary_path)}")

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
                        # Sync loaded chapters to funscript object
                        funscript_processor._sync_chapters_to_funscript()
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
        base, _ = os.path.splitext(filepath)
        if base.endswith(".roll"):
            base = base[:-5]
        path_next_to_vid, _ = os.path.splitext(self.video_path)
        if os.path.exists(filepath):
            if not base == path_next_to_vid:
                try:
                    check_write_access(filepath)
                    # Create a unique backup filename with a Unix timestamp
                    backup_path = f"{filepath}.{int(time.time())}.bak"
                    os.rename(filepath, backup_path)
                    self.logger.info(f"Created backup of existing file: {os.path.basename(backup_path)}")
                except Exception as e:
                    self.logger.error(f"Failed to create backup for {os.path.basename(filepath)}: {e}")
                    # We can decide whether to proceed with the overwrite or not.
                    # For safety, let's proceed but the user is warned.

        sanitized_actions = [ {'at': int(action['at']), 'pos': int(action['pos'])} for action in actions]

        metadata = {
            "version": f"{FUNSCRIPT_METADATA_VERSION}",
            "chapters": []
        }

        # Add chapter data to the metadata dictionary if chapters are provided
        if chapters:
            current_fps = DEFAULT_CHAPTER_FPS
            if self.app.processor and self.app.processor.video_info and self.app.processor.fps > 0:
                current_fps = self.app.processor.fps
            else:
                self.logger.warning(
                    f"Video FPS not available for saving chapters in timecode format. Using default FPS: {DEFAULT_CHAPTER_FPS}. Timecodes may be inaccurate.",
                    extra={'status_message': True})

            metadata["chapters_fps"] = current_fps
            metadata["chapters"] = [chapter.to_funscript_chapter_dict(current_fps) for chapter in chapters]

        # Construct the final funscript data object
        funscript_data = {
            "version": "1.0",
            "author": f"FunGen beta {APP_VERSION}",
            "inverted": False,
            "range": 100,
            "actions": sorted(sanitized_actions, key=lambda x: x["at"]),
            "metadata": metadata
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

    def import_funscript_to_timeline(self, timeline_num: int):
        """Trigger file dialog to import funscript to specified timeline."""
        if hasattr(self.app, 'gui_instance') and self.app.gui_instance and self.app.gui_instance.file_dialog:
            self.app.gui_instance.file_dialog.show(
                is_save=False,
                title=f"Import Funscript to Timeline {timeline_num}",
                extension_filter="Funscript Files (*.funscript),*.funscript",
                callback=lambda filepath: self.load_funscript_to_timeline(filepath, timeline_num)
            )

    def export_funscript_from_timeline(self, timeline_num: int):
        """Trigger file dialog to export funscript from specified timeline.

        Mirrors import_funscript_to_timeline() for API consistency.
        Centralizes all export file dialog logic in one place.

        Args:
            timeline_num: Timeline number to export (1 for primary, 2 for secondary)
        """
        import os

        if not self.app.gui_instance or not self.app.gui_instance.file_dialog:
            self.logger.warning("File dialog not available", extra={"status_message": True})
            return

        output_folder_base = self.app.app_settings.get("output_folder_path", "output")
        initial_path = output_folder_base

        # Set initial filename based on timeline number
        if timeline_num == 1:
            initial_filename = "timeline1.funscript"
        else:
            initial_filename = "timeline2.funscript"

        if self.video_path:
            video_basename = os.path.splitext(os.path.basename(self.video_path))[0]
            initial_path = os.path.join(output_folder_base, video_basename)
            if timeline_num == 1:
                initial_filename = f"{video_basename}.funscript"
            else:
                initial_filename = f"{video_basename}_t2.funscript"

        if not os.path.isdir(initial_path):
            os.makedirs(initial_path, exist_ok=True)

        self.app.gui_instance.file_dialog.show(
            is_save=True,
            title=f"Export Funscript from Timeline {timeline_num}",
            extension_filter="Funscript Files (*.funscript),*.funscript",
            callback=lambda filepath: self.save_funscript_from_timeline(filepath, timeline_num),
            initial_path=initial_path,
            initial_filename=initial_filename
        )

    def import_stage2_overlay_data(self):
        """Trigger file dialog to import stage 2 overlay data."""
        if hasattr(self.app, 'gui_instance') and self.app.gui_instance and self.app.gui_instance.file_dialog:
            self.app.gui_instance.file_dialog.show(
                is_save=False,
                title="Import Stage 2 Overlay Data",
                extension_filter="MessagePack Files (*.msgpack),*.msgpack",
                callback=lambda filepath: self.load_stage2_overlay_data(filepath)
            )

    def open_video_dialog(self):
        """Trigger file dialog to open video file."""
        if hasattr(self.app, 'gui_instance') and self.app.gui_instance and self.app.gui_instance.file_dialog:
            self.app.gui_instance.file_dialog.show(
                is_save=False,
                title="Open Video",
                extension_filter="Video Files (*.mp4;*.avi;*.mov;*.mkv;*.wmv;*.flv;*.webm),*.mp4;*.avi;*.mov;*.mkv;*.wmv;*.flv;*.webm",
                callback=lambda filepath: self.open_video_from_path(filepath)
            )

    def save_funscripts_for_batch(self, video_path: str):
        """
        Automatically saves funscripts next to the video file using the centralized saver.
        This now correctly includes all metadata.
        For remote videos (HTTP URLs), funscripts are always saved to the output directory.
        """
        if not self.app.funscript_processor:
            self.app.logger.error("Funscript processor not available for saving.")
            return

        primary_actions = self.app.funscript_processor.get_actions('primary')
        secondary_actions = self.app.funscript_processor.get_actions('secondary')
        chapters = self.app.funscript_processor.video_chapters
        save_next_to_video = self.app.app_settings.get("autosave_final_funscript_to_video_location", True)

        # Check if video is remote (HTTP/HTTPS URL)
        is_remote = video_path and video_path.startswith(('http://', 'https://'))

        if primary_actions:
            if save_next_to_video and not is_remote:
                # Save next to video file (only for local files)
                base, _ = os.path.splitext(video_path)
                primary_path = f"{base}_t1.funscript"
            else:
                # Save to output directory (for remote videos or when configured)
                primary_path = self.get_output_path_for_file(video_path, "_t1.funscript")

            self._save_funscript_file(primary_path, primary_actions, chapters)
            self.logger.info(f"💾 Saved primary funscript to: {primary_path}")

        if secondary_actions:
            if save_next_to_video and not is_remote:
                # Save next to video file (only for local files)
                base, _ = os.path.splitext(video_path)
                secondary_path = f"{base}_t2.funscript"
            else:
                # Save to output directory (for remote videos or when configured)
                secondary_path = self.get_output_path_for_file(video_path, "_t2.funscript")

            self._save_funscript_file(secondary_path, secondary_actions, None)
            self.logger.info(f"💾 Saved secondary funscript to: {secondary_path}")

    def handle_video_file_load(self, file_path: str, is_project_load=False):
        # If this is a direct video load, first check if an associated project exists.
        # If it does, load that project instead. The project load will handle opening the video.
        if not is_project_load:
            potential_project_path = self.get_output_path_for_file(file_path, PROJECT_FILE_EXTENSION)
            if os.path.exists(potential_project_path):
                self.logger.info(f"Found existing project file for this video. Loading project: {os.path.basename(potential_project_path)}")
                # The load_project method will internally call this function again,
                # but with is_project_load=True, so this block won't re-run.
                self.app.project_manager.load_project(potential_project_path)
                return # End this function call here.

        # If we are here, it's either a project load, or a direct video load with no existing project file.
        self.video_path = file_path
        funscript_processor = self.app.funscript_processor
        stage_processor = self.app.stage_processor

        # This check now runs for ALL video loads, ensuring the app is always aware of the preprocessed file.
        potential_preprocessed_path = self.get_output_path_for_file(self.video_path, "_preprocessed.mp4")
        if os.path.exists(potential_preprocessed_path):
            # Validate the preprocessed file before using it
            preprocessed_status = self._get_preprocessed_file_status(potential_preprocessed_path)

            if preprocessed_status["valid"]:
                self.preprocessed_video_path = potential_preprocessed_path
                self.logger.info(f"Found valid preprocessed video: {os.path.basename(potential_preprocessed_path)} "
                               f"({preprocessed_status['frame_count']}/{preprocessed_status['expected_frames']} frames)")
            else:
                self.preprocessed_video_path = None
                #self.logger.warning(f"Found invalid preprocessed video: {os.path.basename(potential_preprocessed_path)} "
                #                  f"({preprocessed_status['frame_count']}/{preprocessed_status['expected_frames']} frames) - "
                #                  f"will be cleaned up automatically")
        else:
            self.preprocessed_video_path = None

        if not is_project_load:
            # This block is for a direct video load where no project was found. This is a "new project".
            self.app.reset_project_state(for_new_project=True)
            self.video_path = file_path # reset_project_state clears the path, so set it again.

            potential_s1_path = self.get_output_path_for_file(self.video_path, ".msgpack")
            if os.path.exists(potential_s1_path):
                self.stage1_output_msgpack_path = potential_s1_path
                self.app.stage_processor.stage1_status_text = f"Found: {os.path.basename(potential_s1_path)}"
                self.app.stage_processor.stage1_progress_value = 1.0

            potential_s2_overlay = self.get_output_path_for_file(self.video_path, "_stage2_overlay.msgpack")
            if os.path.exists(potential_s2_overlay):
                self.load_stage2_overlay_data(potential_s2_overlay)
            
            # Auto-load Stage 3 mixed debug data if it exists
            potential_s3_mixed_debug = self.get_output_path_for_file(self.video_path, "_stage3_mixed_debug.msgpack")
            if os.path.exists(potential_s3_mixed_debug):
                self.load_stage3_mixed_debug_data(potential_s3_mixed_debug)

        # This part runs for both project loads and new projects.
        if self.app.processor:
            if self.app.processor.open_video(file_path, from_project_load=is_project_load):
                # If it was a new project, we can still try to auto-load an adjacent funscript.
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

    def close_video_action(self, clear_funscript_unconditionally=False, skip_tracker_reset=False):
        if self.app.processor:
            if self.app.processor.is_processing:
                self.app.processor.stop_processing()
            try:
                self.app.processor.reset(close_video=True, skip_tracker_reset=skip_tracker_reset)  # Resets video info in processor
            except TypeError:
                self.app.processor.reset(close_video=True)

        self.video_path = ""
        self.preprocessed_video_path = None
        self.app.stage_processor.reset_stage_status(stages=("stage1", "stage2", "stage3"))
        self.app.funscript_processor.video_chapters.clear()
        self.clear_stage2_overlay_data()
        # Also clear any mixed debug artifacts
        if hasattr(self.app, 'stage3_mixed_debug_data'):
            self.app.stage3_mixed_debug_data = None
        if hasattr(self.app, 'stage3_mixed_debug_frame_map'):
            self.app.stage3_mixed_debug_frame_map = None

        # Clear audio waveform data
        self.app.audio_waveform_data = None
        self.app.app_state_ui.show_audio_waveform = False

        # If funscript was loaded from a file (not generated) and we are not clearing unconditionally, keep T1.
        # Otherwise, clear T1. Always clear T2.
        if clear_funscript_unconditionally or not self.loaded_funscript_path:  # loaded_funscript_path is for T1
            if self.app.processor and self.app.processor.tracker and self.app.processor.tracker.funscript:
                self.app.funscript_processor.clear_timeline_history_and_set_new_baseline(1, [], "Video Closed (T1 Cleared)")
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
        """Load Stage 2 overlay data (supports legacy list format and new dict with frames/segments/metadata)."""
        self.clear_stage2_overlay_data()  # Clear previous before loading new
        stage_processor = self.app.stage_processor
        try:
            with open(filepath, 'rb') as f:
                packed_data = f.read()
            loaded_data = msgpack.unpackb(packed_data, raw=False)

            overlay_frames: list = []
            overlay_segments: list = []

            # New format: { "frames": [...], "segments": [...], "metadata": {...} }
            if isinstance(loaded_data, dict) and ("frames" in loaded_data or "segments" in loaded_data):
                overlay_frames = loaded_data.get("frames", []) or []
                overlay_segments = loaded_data.get("segments", []) or []
            # Legacy format: list of frame dicts
            elif isinstance(loaded_data, list):
                overlay_frames = loaded_data
            else:
                stage_processor.stage2_status_text = "Error: Unsupported overlay format"
                self.app.app_state_ui.show_stage2_overlay = False
                self.logger.error("Stage 2 overlay data is not in expected dict/list format.")
                return

            # Set overlay frames into processor structures
            stage_processor.stage2_overlay_data = overlay_frames
            stage_processor.stage2_overlay_data_map = {
                frame_data.get("frame_id", -1): frame_data
                for frame_data in overlay_frames if isinstance(frame_data, dict)
            }
            self.stage2_output_msgpack_path = filepath

            # Load segments if present
            if overlay_segments:
                try:
                    self.app.stage2_segments = [VideoSegment.from_dict(seg) if isinstance(seg, dict) else seg for seg in overlay_segments]
                except Exception as seg_e:
                    self.logger.warning(f"Failed to parse overlay segments: {seg_e}")

            if overlay_frames:
                stage_processor.stage2_status_text = f"Overlay loaded: {os.path.basename(filepath)}"
                self.logger.info(
                    f"Loaded Stage 2 overlay: {os.path.basename(filepath)} ({len(overlay_frames)} frames)",
                    extra={'status_message': True})
                self.app.app_state_ui.show_stage2_overlay = True
            else:
                stage_processor.stage2_status_text = f"Overlay file empty: {os.path.basename(filepath)}"
                self.app.app_state_ui.show_stage2_overlay = False
                self.logger.warning(f"Stage 2 overlay file is empty: {os.path.basename(filepath)}", extra={'status_message': True})

            self.app.project_manager.project_dirty = True
            self.app.energy_saver.reset_activity_timer()
        except Exception as e:
            stage_processor.stage2_status_text = "Error loading overlay"
            self.app.app_state_ui.show_stage2_overlay = False
            self.logger.error(f"Error loading Stage 2 overlay msgpack '{filepath}': {e}", extra={'status_message': True})

    def clear_stage2_overlay_data(self):
        stage_processor = self.app.stage_processor
        stage_processor.stage2_overlay_data = None
        stage_processor.stage2_overlay_data_map = None
        self.stage2_output_msgpack_path = None  # Clear path if data is cleared

    def load_stage3_mixed_debug_data(self, filepath: str):
        """Load Stage 3 mixed debug msgpack for overlay display during video playback."""
        self.clear_stage3_mixed_debug_data()
        try:
            with open(filepath, 'rb') as f:
                packed_data = f.read()
            loaded_data = msgpack.unpackb(packed_data, raw=False)
            
            if isinstance(loaded_data, dict) and 'frame_data' in loaded_data:
                # Store the loaded debug data
                self.app.stage3_mixed_debug_data = loaded_data
                self.app.stage3_mixed_debug_frame_map = {}
                
                # Create frame map for quick lookup
                for frame_id_str, frame_debug in loaded_data['frame_data'].items():
                    try:
                        frame_id = int(frame_id_str)
                        self.app.stage3_mixed_debug_frame_map[frame_id] = frame_debug
                    except (ValueError, TypeError):
                        continue
                
                frame_count = len(self.app.stage3_mixed_debug_frame_map)
                self.logger.info(f"Loaded Stage 3 mixed debug data: {os.path.basename(filepath)} ({frame_count} frames)")
                return True
            else:
                self.logger.error("Stage 3 mixed debug data is not in expected format.")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading Stage 3 mixed debug msgpack '{filepath}': {e}")
            return False
    
    def clear_stage3_mixed_debug_data(self):
        """Clear Stage 3 mixed debug data."""
        if hasattr(self.app, 'stage3_mixed_debug_data'):
            self.app.stage3_mixed_debug_data = None
        if hasattr(self.app, 'stage3_mixed_debug_frame_map'):
            self.app.stage3_mixed_debug_frame_map = None

    def open_video_from_path(self, file_path: str) -> bool:
        """
        Opens a video file, updates the application state, and returns success.
        This is the central method for loading a video.
        """
        # Check if file exists (skip check for HTTP URLs - FFmpeg will handle those)
        is_remote = file_path and file_path.startswith(('http://', 'https://'))
        if not file_path or (not is_remote and not os.path.exists(file_path)):
            self.app.logger.error(f"Video file not found: {file_path}")
            return False

        self.app.logger.info(f"Opening video: {os.path.basename(file_path)}", extra={'status_message': True})

        # Reset relevant states before loading a new video
        self.close_video_action(clear_funscript_unconditionally=True)

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
            self.app.logger.error(f"Failed to open video file: {os.path.basename(file_path)}", extra={'status_message': True})

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
        Handles dropped files/folders. Scans for videos and prepares them for the
        new enhanced batch processing dialog.
        """
        if not paths:
            return

        from video import VideoProcessor # Local import to avoid circular dependency
        from application.classes import ImGuiFileDialog

        videos_to_process = []
        valid_video_extensions = {".mp4", ".mkv", ".mov", ".avi", ".webm"}

        # Categorize all dropped paths
        for path in paths:
            if os.path.isdir(path):
                # If a directory is dropped, scan it and its subfolders for videos
                self.app.logger.info(f"Scanning dropped folder for videos: {path}")
                videos_to_process.extend(self._scan_folder_for_videos(path))
            elif os.path.splitext(path)[1].lower() in valid_video_extensions:
                # If a video file is dropped, add it to the list for processing
                videos_to_process.append(path)
            # else:
            #     # Keep track of other non-video file types
            #     other_files.append(path)

        unique_videos = sorted(list(set(videos_to_process)))

        if not unique_videos:
            self.app.logger.info("No video files found in dropped items.")
            return

        if len(unique_videos) == 1:
            self.app.logger.info(f"Single video dropped. Opening directly: {os.path.basename(unique_videos[0])}")
            self.open_video_from_path(unique_videos[0])
            return

        # elif other_files:
        #     # If no videos were found, fall back to the original logic for handling other file types
        #     self.app.logger.info("No videos found for batching, handling as single file drop.")
        #     path = other_files[0]
        #     ext = os.path.splitext(path)[1].lower()
        #     if ext == '.funscript':
        #         self.load_funscript_to_timeline(path, 1)
        #     elif ext == PROJECT_FILE_EXTENSION:
        #         self.app.project_manager.load_project(path)
        #     elif ext == '.msgpack':
        #         self.load_stage2_overlay_data(path)
        #     else:
        #         self.last_dropped_files = other_files
        #         self.app.logger.warning(f"Unrecognized file type dropped: {os.path.basename(path)}", extra={'status_message': True})

        self.app.logger.info(f"Found {len(unique_videos)} videos. Preparing batch dialog...")
        gui = self.app.gui_instance
        if not gui:
            self.app.logger.error("GUI instance not available to show batch dialog.")
            return

        gui.batch_videos_data.clear()
        for video_path in unique_videos:
            gui.batch_videos_data.append({
                "path": video_path,
                "selected": False,
                "funscript_status": ImGuiFileDialog.get_funscript_status(video_path, self.app.logger),
                "detected_format": VideoProcessor.get_video_type_heuristic(video_path),
                "override_format_idx": 0, # Index for 'Auto'
            })

        gui.last_overwrite_mode_ui = -1
        self.app.show_batch_confirmation_dialog = True

    def update_settings_from_app(self):
        """Called by AppLogic to reflect loaded settings or project data."""
        # Model paths are handled by AppLogic's _apply_loaded_settings directly
        # Stage output paths are mostly managed by project load/save and stage runs
        pass

    def save_settings_to_app(self):
        """Called by AppLogic when app settings are saved."""
        # Model paths are handled by AppLogic's save_app_settings directly
        pass

    def save_final_funscripts(self, video_path: str, chapters: Optional[List[Dict]] = None) -> List[str]:
        """
        Saves the final (potentially post-processed) funscripts.
        Adheres to the 'autosave_final_funscript_to_video_location' setting.
        Returns a list of the full paths of the saved files.
        """
        if not self.app.funscript_processor:
            self.logger.error("Funscript processor not available for saving final funscripts.")
            return []

        saved_paths = []
        save_next_to_video = self.app.app_settings.get("autosave_final_funscript_to_video_location", True)
        if self.app.is_batch_processing_active:
            save_next_to_video = self.app.batch_copy_funscript_to_video_location


        primary_actions = self.app.funscript_processor.get_actions('primary')
        secondary_actions = self.app.funscript_processor.get_actions('secondary')

        chapters_to_save = []
        if chapters is not None:
            chapters_to_save = [VideoSegment.from_dict(chap_data) for chap_data in chapters if isinstance(chap_data, dict)]
        else:
            chapters_to_save = self.app.funscript_processor.video_chapters

        # Determine roll generation setting
        generate_roll = self.app.app_settings.get("generate_roll_file", True)
        if self.app.is_batch_processing_active:
            generate_roll = self.app.batch_generate_roll_file

        # --- Main funscript saving ---
        if primary_actions:
            path_in_output = self.get_output_path_for_file(video_path, ".funscript")
            self._save_funscript_file(path_in_output, primary_actions, chapters_to_save)
            saved_paths.append(path_in_output)

            if save_next_to_video:
                self.logger.info("Also saving a copy of the final funscript next to the video file.")
                base, _ = os.path.splitext(video_path)
                path_next_to_vid = f"{base}.funscript"
                self._save_funscript_file(path_next_to_vid, primary_actions, chapters_to_save)
                # Do not add this to saved_paths to avoid double copying

        # --- Roll funscript saving ---
        if secondary_actions and generate_roll:
            path_in_output_t2 = self.get_output_path_for_file(video_path, ".roll.funscript")
            self._save_funscript_file(path_in_output_t2, secondary_actions, None)
            saved_paths.append(path_in_output_t2)

            if save_next_to_video:
                base, _ = os.path.splitext(video_path)
                path_next_to_vid_t2 = f"{base}.roll.funscript"
                self._save_funscript_file(path_next_to_vid_t2, secondary_actions, None)
                # Do not add this to saved_paths to avoid double copying

        return saved_paths

    def _get_preprocessed_file_status(self, preprocessed_path: str) -> Dict[str, Any]:
        """
        Gets the status of a preprocessed file, including validation.

        Args:
            preprocessed_path: Path to the preprocessed file

        Returns:
            Dictionary with status information
        """
        status = {
            "exists": False,
            "valid": False,
            "frame_count": 0,
            "expected_frames": 0,
            "file_size": 0
        }

        try:
            if not os.path.exists(preprocessed_path):
                return status

            status["exists"] = True
            status["file_size"] = os.path.getsize(preprocessed_path)

            # Get expected frame count from current video
            if self.app.processor and self.app.processor.video_info:
                expected_frames = self.app.processor.video_info.get("total_frames", 0)
                expected_fps = self.app.processor.video_info.get("fps", 30.0)
                status["expected_frames"] = expected_frames

                if expected_frames > 0:
                    # Import validation function
                    from detection.cd.stage_1_cd import _validate_preprocessed_video_completeness

                    # Validate the preprocessed video
                    status["valid"] = _validate_preprocessed_video_completeness(
                        preprocessed_path, expected_frames, expected_fps, self.logger
                    )

                    # Get actual frame count
                    if self.app.processor:
                        preprocessed_info = self.app.processor._get_video_info(preprocessed_path)
                        if preprocessed_info:
                            status["frame_count"] = preprocessed_info.get("total_frames", 0)

        except Exception as e:
            self.logger.error(f"Error getting preprocessed file status: {e}")

        return status

    def get_preprocessed_status_summary(self) -> str:
        """
        Returns a human-readable summary of the preprocessed video status.

        Returns:
            Status summary string
        """
        if not self.video_path:
            return "No video loaded"

        preprocessed_path = self.get_output_path_for_file(self.video_path, "_preprocessed.mp4")
        status = self._get_preprocessed_file_status(preprocessed_path)

        if not status["exists"]:
            return "No preprocessed video available"
        elif not status["valid"]:
            return f"Invalid preprocessed video ({status['frame_count']}/{status['expected_frames']} frames)"
        else:
            size_mb = status["file_size"] / (1024 * 1024)
            return f"Valid preprocessed video ({status['frame_count']} frames, {size_mb:.1f} MB)"
