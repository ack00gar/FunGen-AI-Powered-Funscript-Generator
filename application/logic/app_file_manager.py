import os
import glob as glob_module
import orjson
import msgpack
import time
from typing import List, Optional, Dict, Tuple, Any

from application.utils import VideoSegment, check_write_access
from config.constants import PROJECT_FILE_EXTENSION, AUTOSAVE_FILE, DEFAULT_CHAPTER_FPS, APP_VERSION, APP_NAME, FUNSCRIPT_METADATA_VERSION
from funscript.axis_registry import (
    file_suffix_for_axis, axis_from_file_suffix, axis_from_tcode, tcode_for_axis,
    all_known_suffixes, AXIS_FILE_SUFFIX, FunscriptAxis
)

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

    def _resolve_axis_name_for_timeline(self, timeline_num: int) -> str:
        """Get the semantic axis name for a timeline number from the funscript object."""
        if self.app.processor and self.app.processor.tracker and self.app.processor.tracker.funscript:
            return self.app.processor.tracker.funscript.get_axis_for_timeline(timeline_num)
        # Fallback defaults
        defaults = {1: "stroke", 2: "roll"}
        return defaults.get(timeline_num, f"axis_{timeline_num}")

    def _resolve_tracker_info(self) -> Tuple[Optional[str], Optional[str]]:
        """Resolve the tracker display name and version via priority chain.

        Returns (display_name, version) tuple.
        """
        # 1. Active tracker instance
        try:
            if self.app.processor and self.app.processor.tracker:
                md = self.app.processor.tracker.metadata
                return md.display_name, getattr(md, 'version', None)
        except (AttributeError, TypeError):
            pass

        # 2/3. Batch tracker name or GUI-selected tracker — resolve via TrackerDiscovery
        from config.tracker_discovery import DynamicTrackerDiscovery
        discovery = DynamicTrackerDiscovery()

        internal_name = getattr(self.app, 'batch_tracker_name', None)
        if not internal_name:
            state_ui = getattr(self.app, 'app_state_ui', None)
            if state_ui:
                internal_name = getattr(state_ui, 'selected_tracker_name', None)

        if internal_name:
            info = discovery.get_tracker_info(internal_name)
            if info:
                return info.display_name, getattr(info, 'version', None)

        return None, None

    def _resolve_tracker_display_name(self) -> Optional[str]:
        """Resolve the human-readable tracker name via priority chain."""
        name, _ = self._resolve_tracker_info()
        return name

    def _build_generation_metadata(self, multi_axis: bool = False) -> dict:
        """Return FunGen generation metadata for funscript exports."""
        from datetime import datetime, timezone
        meta = {
            "creator_software": APP_NAME,
            "creator_software_version": APP_VERSION,
            "creation_date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "funscript_type": "multi-axis" if multi_axis else "single-axis",
        }
        tracker_name, tracker_version = self._resolve_tracker_info()
        if tracker_name:
            meta["tracker_name"] = tracker_name
        if tracker_version:
            meta["tracker_version"] = tracker_version

        # Verbose metadata (git info, video info)
        if self.app_settings.get("metadata_verbose", True):
            from application.utils.logger import _GIT_INFO
            if _GIT_INFO and "@" in _GIT_INFO:
                branch, commit_hash = _GIT_INFO.split("@", 1)
                if branch and branch != "unknown":
                    meta["git_branch"] = branch
                if commit_hash and commit_hash != "unknown":
                    meta["git_commit_hash"] = commit_hash

            # Video info
            if self.video_path:
                meta["original_video_filename"] = os.path.basename(self.video_path)
                try:
                    fsize = os.path.getsize(self.video_path)
                    if fsize > 0:
                        meta["original_video_file_size_bytes"] = fsize
                except OSError:
                    pass
            try:
                video_info = self.app.processor.video_info
                if video_info:
                    if video_info.get("duration"):
                        meta["original_video_duration_seconds"] = video_info["duration"]
                    if video_info.get("total_frames"):
                        meta["original_video_total_frames"] = video_info["total_frames"]
                    if video_info.get("fps"):
                        meta["original_video_fps"] = video_info["fps"]
                    w, h = video_info.get("width"), video_info.get("height")
                    if w and h:
                        meta["original_video_resolution"] = f"{w}x{h}"
            except (AttributeError, TypeError):
                pass

        if self.app_settings.get("performance_metadata", False):
            # Performance / pipeline settings
            meta["hardware_acceleration"] = getattr(self.app, "hardware_acceleration_method", "none")
            meta["vr_unwarp_method"] = self.app_settings.get("vr_unwarp_method", "auto")
            meta["num_producers_stage1"] = self.app_settings.get("num_producers_stage1", 1)
            meta["num_consumers_stage1"] = self.app_settings.get("num_consumers_stage1", 1)
            meta["num_workers_stage2_of"] = self.app_settings.get("num_workers_stage2_of", 1)

            # Per-stage and total processing time (clock wall-time)
            try:
                sp = self.app.stage_processor
                stage_times = {}
                total_seconds = 0
                for key, attr in (("stage1", "stage1_final_elapsed_time_str"),
                                  ("stage2", "stage2_final_elapsed_time_str"),
                                  ("stage3", "stage3_final_elapsed_time_str")):
                    val = getattr(sp, attr, "")
                    if val:
                        stage_times[key] = val
                        parts = val.split(":")
                        if len(parts) == 3:
                            total_seconds += int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                if stage_times:
                    meta["processing_time_per_stage"] = stage_times
                    meta["processing_time_total_seconds"] = total_seconds
            except (AttributeError, TypeError, ValueError):
                pass

        # Optional creator identity
        creator_id = self.app_settings.get("metadata_creator_identity", "")
        if creator_id:
            meta["creator_identity"] = creator_id

        return meta

    def _get_funscript_path_for_axis(self, video_path: str, axis_name: str) -> str:
        """Return the OFS-convention path for a given axis next to the video file.

        E.g. axis_name='stroke' -> '/path/to/video.funscript'
             axis_name='roll'   -> '/path/to/video.roll.funscript'
             axis_name='pitch'  -> '/path/to/video.pitch.funscript'
        """
        suffix = file_suffix_for_axis(axis_name)
        base, _ = os.path.splitext(video_path)
        return f"{base}{suffix}.funscript"

    def discover_axis_funscripts(self, video_path: str) -> Dict[str, str]:
        """Scan for funscript files associated with a video using OFS and legacy naming.

        Returns a dict mapping axis_name -> filepath for discovered files.
        Searches both OFS naming (video.roll.funscript) and legacy _tN naming.
        """
        if not video_path or not os.path.exists(video_path):
            return {}

        base, _ = os.path.splitext(video_path)
        discovered: Dict[str, str] = {}

        # 1. Check OFS-style naming: basename.{suffix}.funscript
        for suffix in all_known_suffixes():
            # suffix is like '.roll', '.pitch', etc.
            candidate = f"{base}{suffix}.funscript"
            if os.path.exists(candidate):
                axis = axis_from_file_suffix(suffix)
                if axis:
                    discovered[axis.value] = candidate

        # 2. Check primary (no suffix): basename.funscript
        primary_path = f"{base}.funscript"
        if os.path.exists(primary_path):
            discovered.setdefault("stroke", primary_path)

        # 3. Check legacy _tN naming: basename_t1.funscript, basename_t2.funscript, ...
        legacy_axis_map = {1: "stroke", 2: "roll"}
        for pattern_file in glob_module.glob(f"{base}_t*.funscript"):
            fname = os.path.basename(pattern_file)
            # Extract the N from _tN.funscript
            prefix = os.path.basename(base) + "_t"
            if fname.startswith(prefix) and fname.endswith(".funscript"):
                num_str = fname[len(prefix):-len(".funscript")]
                try:
                    tl_num = int(num_str)
                    axis_name = legacy_axis_map.get(tl_num, f"axis_{tl_num}")
                    # Don't overwrite OFS-discovered files (OFS naming takes priority)
                    discovered.setdefault(axis_name, pattern_file)
                except ValueError:
                    pass

        return discovered

    def _parse_funscript_file(self, funscript_file_path: str) -> Tuple[Optional[List[Dict]], Optional[str], Optional[List[Dict]], Optional[float]]:
        """ Parses a funscript file using the high-performance orjson library.

        Returns (actions, error_msg, chapters_list, chapters_fps).
        Also parses embedded ``axes`` array (unified multi-axis format) and
        stores the results on ``self._last_parsed_embedded_axes`` so callers
        (e.g. load_funscript_to_timeline) can pick them up.
        """
        self._last_parsed_embedded_axes: List[Dict] = []  # Reset each parse
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

            # Parse embedded axes (unified multi-axis format)
            if 'axes' in data and isinstance(data['axes'], list):
                self._last_parsed_embedded_axes = self._parse_embedded_axes(data['axes'])
                if self._last_parsed_embedded_axes:
                    self.logger.info(
                        f"Found {len(self._last_parsed_embedded_axes)} embedded axes in {os.path.basename(funscript_file_path)}.")

            return parsed_actions, None, chapters_list_of_dicts, chapters_fps_from_file
        except FileNotFoundError:
            return None, f"File not found: {os.path.basename(funscript_file_path)}", None, None
        except orjson.JSONDecodeError:  # <-- Catch the specific orjson exception
            return None, f"Error decoding JSON from {os.path.basename(funscript_file_path)}.", None, None
        except Exception as e:
            self.logger.error(f"Unexpected error loading funscript '{funscript_file_path}': {e}", exc_info=True,
                              extra={'status_message': True})
            return None, f"Error loading funscript: {str(e)}", None, None

    def _resolve_axis_name(self, axis_ids: List[str], name: Optional[str]) -> str:
        """Resolve an axis identity from TCode IDs and/or a human name.

        Tries TCode lookup first (e.g. 'R1' -> 'roll'), falls back to the
        provided ``name``, and finally to the raw TCode ID.
        """
        for tcode_id in axis_ids:
            fa = axis_from_tcode(tcode_id)
            if fa is not None:
                return fa.value
        if name:
            return name
        return axis_ids[0] if axis_ids else "unknown"

    @staticmethod
    def _scale_axis_actions(actions: List[Dict]) -> List[Dict]:
        """Detect 0-9999 range and scale to 0-100. Returns a new list."""
        if not actions:
            return []
        max_pos = max(a.get("pos", 0) for a in actions)
        needs_scale = max_pos > 100
        scaled = []
        for a in actions:
            if not isinstance(a, dict) or "at" not in a or "pos" not in a:
                continue
            ts = int(a["at"])
            pos = int(a["pos"])
            if needs_scale:
                pos = round(pos * 100 / 9999)
            pos = min(max(pos, 0), 100)
            scaled.append({"at": ts, "pos": pos})
        return sorted(scaled, key=lambda x: x["at"])

    def _parse_embedded_axes(self, axes_list: list) -> List[Dict]:
        """Parse the ``axes`` array from a unified multi-axis funscript.

        Returns a list of dicts: ``[{"name": str, "actions": list}, ...]``
        """
        result = []
        for entry in axes_list:
            if not isinstance(entry, dict):
                continue
            axis_ids = entry.get("axes", [])
            axis_name_hint = entry.get("name")
            raw_actions = entry.get("actions", [])
            if not raw_actions:
                continue
            resolved_name = self._resolve_axis_name(axis_ids, axis_name_hint)
            scaled_actions = self._scale_axis_actions(raw_actions)
            if scaled_actions:
                result.append({"name": resolved_name, "actions": scaled_actions})
        return result

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
        """Save raw funscripts next to the video file using OFS naming conventions.

        Uses .raw.funscript extension by default, or .funscript if
        the 'export_raw_as_funscript' setting is enabled.
        Saves all active axes (T1, T2, and any T3+ additional axes).
        """
        if not self.app.funscript_processor: return
        if not video_path: return

        chapters = self.app.funscript_processor.video_chapters

        # Check if copy to video location is enabled
        save_next_to_video = self.app.app_settings.get("autosave_final_funscript_to_video_location", True)
        if self.app.is_batch_processing_active:
            save_next_to_video = self.app.batch_copy_funscript_to_video_location

        if not save_next_to_video:
            self.logger.info("Copy to video location is disabled. Raw funscripts saved only to output folder.")
            return

        skip_raw_prefix = self.app.app_settings.get("export_raw_as_funscript", False)
        raw_infix = "" if skip_raw_prefix else ".raw"
        base, _ = os.path.splitext(video_path)

        # Build list of (timeline_num, axis_internal_name, axis_semantic_name) to save
        timelines_to_save = [(1, 'primary'), (2, 'secondary')]
        funscript_obj = None
        if self.app.processor and self.app.processor.tracker and self.app.processor.tracker.funscript:
            funscript_obj = self.app.processor.tracker.funscript
            for axis_name in funscript_obj.additional_axes:
                # Find the timeline number for this additional axis
                tl_num = None
                for tn, an in funscript_obj._axis_assignments.items():
                    if an == axis_name or f"axis_{tn}" == axis_name:
                        tl_num = tn
                        break
                if tl_num is None:
                    # Derive from axis name convention
                    if axis_name.startswith("axis_"):
                        try:
                            tl_num = int(axis_name.split("_")[1])
                        except (IndexError, ValueError):
                            continue
                    else:
                        continue
                timelines_to_save.append((tl_num, axis_name))

        # Determine roll generation setting
        generate_roll = self.app.app_settings.get("generate_roll_file", True)
        if self.app.is_batch_processing_active:
            generate_roll = self.app.batch_generate_roll_file

        self.logger.info(f"Saving raw funscripts next to video file ({len(timelines_to_save)} axes)...")

        for tl_num, internal_axis in timelines_to_save:
            actions = self.app.funscript_processor.get_actions(internal_axis)
            if not actions:
                continue
            # Skip secondary axis if roll generation is disabled
            if tl_num == 2 and not generate_roll:
                continue

            axis_name = self._resolve_axis_name_for_timeline(tl_num)
            suffix = file_suffix_for_axis(axis_name)
            path = f"{base}{raw_infix}{suffix}.funscript"
            chaps = chapters if tl_num == 1 else None
            self._save_funscript_file(path, actions, chaps)
            self.logger.info(f"Raw funscript saved (T{tl_num}/{axis_name}): {os.path.basename(path)}")

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

        # Load embedded axes from unified format into the funscript object
        if timeline_num == 1 and self._last_parsed_embedded_axes:
            funscript_obj = self.app.processor.tracker.funscript if (
                self.app.processor and self.app.processor.tracker
            ) else None
            if not funscript_obj and hasattr(self.app, 'multi_axis_funscript'):
                funscript_obj = self.app.multi_axis_funscript
            if funscript_obj:
                next_tl = 2  # Start assigning from T2 if not already used
                for embedded in self._last_parsed_embedded_axes:
                    axis_name = embedded["name"]
                    axis_actions = embedded["actions"]
                    # Skip stroke (already loaded as root actions)
                    if axis_name == "stroke":
                        continue
                    # roll goes to T2
                    if axis_name == "roll":
                        funscript_processor.clear_timeline_history_and_set_new_baseline(
                            2, axis_actions,
                            f"Embedded axis: {axis_name}")
                        funscript_obj.assign_axis(2, axis_name)
                        self.logger.info(f"Loaded embedded axis '{axis_name}' to T2 ({len(axis_actions)} actions)")
                        continue
                    # T3+ for other axes
                    next_tl = max(next_tl + 1, 3)
                    internal_axis = f"axis_{next_tl}"
                    funscript_obj.ensure_axis(internal_axis)
                    funscript_obj.additional_axes[internal_axis] = axis_actions
                    funscript_obj.assign_axis(next_tl, axis_name)
                    funscript_obj._invalidate_cache(internal_axis)
                    self.logger.info(
                        f"Loaded embedded axis '{axis_name}' to T{next_tl} ({len(axis_actions)} actions)")
                    next_tl += 1

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

        metadata.update(self._build_generation_metadata())

        # Include project metadata if available
        project_metadata = {}
        if hasattr(self.app, 'project_manager') and hasattr(self.app.project_manager, 'get_metadata'):
            project_metadata = self.app.project_manager.get_metadata() or {}

        # Construct the final funscript data object
        funscript_data = {
            "version": "1.0",
            "author": project_metadata.get("creator", f"FunGen beta {APP_VERSION}"),
            "inverted": False,
            "range": 100,
            "actions": sorted(sanitized_actions, key=lambda x: x["at"]),
            "metadata": metadata
        }

        # Add optional metadata fields if present
        if project_metadata.get("title"):
            funscript_data["title"] = project_metadata["title"]
        if project_metadata.get("description"):
            funscript_data["description"] = project_metadata["description"]
        if project_metadata.get("tags"):
            funscript_data["tags"] = [t.strip() for t in project_metadata["tags"].split(",") if t.strip()]
        if project_metadata.get("performers"):
            funscript_data["performers"] = [p.strip() for p in project_metadata["performers"].split(",") if p.strip()]
        if project_metadata.get("script_url"):
            funscript_data["script_url"] = project_metadata["script_url"]
        if project_metadata.get("video_url"):
            funscript_data["video_url"] = project_metadata["video_url"]
        if project_metadata.get("license"):
            funscript_data["license"] = project_metadata["license"]
        if project_metadata.get("notes"):
            funscript_data["notes"] = project_metadata["notes"]

        try:
            # Use orjson for high-performance writing
            with open(filepath, 'wb') as f:
                f.write(orjson.dumps(funscript_data))
            self.logger.info(f"Funscript saved to {os.path.basename(filepath)}",
                             extra={'status_message': True})
        except Exception as e:
            self.logger.error(f"Error saving funscript to '{filepath}': {e}",
                              extra={'status_message': True})

    def _save_funscript_file_unified(self, filepath: str, funscript_obj, chapters: Optional[List] = None):
        """Save all axes into a single .funscript file with embedded axes array.

        The root ``actions`` key holds the primary/stroke axis. Additional axes
        are stored in an ``axes`` array following the community multi-axis spec.
        """
        primary_actions = funscript_obj.get_axis_actions('primary')
        if not primary_actions:
            self.logger.info(f"No primary actions to save to {os.path.basename(filepath)}.",
                             extra={'status_message': True})
            return

        sanitized_primary = [{'at': int(a['at']), 'pos': int(a['pos'])} for a in primary_actions]

        metadata = {"version": f"{FUNSCRIPT_METADATA_VERSION}", "chapters": []}
        if chapters:
            current_fps = DEFAULT_CHAPTER_FPS
            if self.app.processor and self.app.processor.video_info and self.app.processor.fps > 0:
                current_fps = self.app.processor.fps
            metadata["chapters_fps"] = current_fps
            metadata["chapters"] = [ch.to_funscript_chapter_dict(current_fps) for ch in chapters]

        metadata.update(self._build_generation_metadata(multi_axis=True))

        funscript_data = {
            "version": "1.0",
            "author": f"FunGen beta {APP_VERSION}",
            "inverted": False,
            "range": 100,
            "actions": sorted(sanitized_primary, key=lambda x: x["at"]),
            "metadata": metadata,
            "axes": []
        }

        # Build embedded axes from assignments (skip T1/primary)
        assignments = funscript_obj.get_axis_assignments()
        for tl_num in sorted(assignments.keys()):
            if tl_num <= 1:
                continue
            axis_name = assignments[tl_num]
            # Determine which internal key holds this axis's data
            if tl_num == 2:
                actions = funscript_obj.get_axis_actions('secondary')
            else:
                actions = funscript_obj.get_axis_actions(f'axis_{tl_num}')
                if not actions:
                    actions = funscript_obj.get_axis_actions(axis_name)
            if not actions:
                continue
            tcode = tcode_for_axis(axis_name)
            axis_entry = {
                "axes": [tcode] if tcode else [axis_name],
                "name": axis_name,
                "actions": [{'at': int(a['at']), 'pos': int(a['pos'])} for a in actions]
            }
            funscript_data["axes"].append(axis_entry)

        try:
            with open(filepath, 'wb') as f:
                f.write(orjson.dumps(funscript_data))
            axis_count = 1 + len(funscript_data["axes"])
            self.logger.info(
                f"Unified funscript saved ({axis_count} axes) to {os.path.basename(filepath)}",
                extra={'status_message': True})
        except Exception as e:
            self.logger.error(f"Error saving unified funscript to '{filepath}': {e}",
                              extra={'status_message': True})

    def save_funscript_from_timeline(self, filepath: str, timeline_num: int):
        funscript_processor = self.app.funscript_processor
        if timeline_num == 1:
            axis = 'primary'
        elif timeline_num == 2:
            axis = 'secondary'
        else:
            axis = f'axis_{timeline_num}'
        actions = funscript_processor.get_actions(axis)

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

        # Resolve axis name for OFS-style filename
        axis_name = self._resolve_axis_name_for_timeline(timeline_num)
        suffix = file_suffix_for_axis(axis_name)

        if timeline_num == 1:
            initial_filename = "timeline1.funscript"
        else:
            initial_filename = f"timeline{timeline_num}{suffix}.funscript"

        if self.video_path:
            video_basename = os.path.splitext(os.path.basename(self.video_path))[0]
            initial_path = os.path.join(output_folder_base, video_basename)
            initial_filename = f"{video_basename}{suffix}.funscript"

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

    def export_heatmap_png(self, timeline_num: int = 1):
        """Export heatmap PNG image for the specified timeline.

        Follows the same dialog pattern as export_funscript_from_timeline().
        """
        if not self.app.gui_instance or not self.app.gui_instance.file_dialog:
            self.logger.warning("File dialog not available", extra={"status_message": True})
            return

        # Get actions
        fs_proc = self.app.funscript_processor
        if not fs_proc:
            self.logger.warning("No funscript data to export heatmap.", extra={"status_message": True})
            return

        axis = "primary" if timeline_num == 1 else "secondary"
        funscript_obj = fs_proc.get_funscript_obj() if hasattr(fs_proc, 'get_funscript_obj') else None
        if not funscript_obj:
            self.logger.warning("No funscript loaded.", extra={"status_message": True})
            return

        actions = funscript_obj.get_axis_actions(axis)
        if not actions:
            self.logger.warning("No actions on this timeline.", extra={"status_message": True})
            return

        # Duration
        duration_ms = actions[-1]['at'] if actions else 0
        if self.app.processor and self.app.processor.video_info:
            fps = self.app.processor.fps
            total_frames = self.app.processor.video_info.get('total_frames', 0)
            if fps > 0 and total_frames > 0:
                duration_ms = max(duration_ms, (total_frames / fps) * 1000.0)

        output_folder = self.app.app_settings.get("output_folder_path", "output")
        initial_filename = "heatmap.png"
        if self.video_path:
            video_basename = os.path.splitext(os.path.basename(self.video_path))[0]
            initial_filename = f"{video_basename}_heatmap.png"
            output_folder = os.path.join(output_folder, video_basename)

        if not os.path.isdir(output_folder):
            os.makedirs(output_folder, exist_ok=True)

        def _do_export(filepath):
            try:
                from funscript.heatmap_export import HeatmapExporter
                exporter = HeatmapExporter()
                exporter.export_png(filepath, actions, duration_ms)
                self.logger.info(f"Heatmap exported to {os.path.basename(filepath)}",
                                 extra={"status_message": True})
            except Exception as e:
                self.logger.error(f"Heatmap export failed: {e}", extra={"status_message": True})

        self.app.gui_instance.file_dialog.show(
            is_save=True,
            title="Export Heatmap PNG",
            extension_filter="PNG Images (*.png),*.png",
            callback=_do_export,
            initial_path=output_folder,
            initial_filename=initial_filename,
        )

    def export_unified_funscript(self):
        """Export all axes as a single multi-axis funscript file."""
        if not self.app.gui_instance or not self.app.gui_instance.file_dialog:
            self.logger.warning("File dialog not available", extra={"status_message": True})
            return

        funscript_obj = None
        if self.app.tracker and hasattr(self.app.tracker, 'funscript'):
            funscript_obj = self.app.tracker.funscript
        if not funscript_obj:
            self.logger.warning("No funscript data available to export.", extra={"status_message": True})
            return

        output_folder = self.app.app_settings.get("output_folder_path", "output")
        initial_path = output_folder
        initial_filename = "unified.funscript"

        if self.video_path:
            video_basename = os.path.splitext(os.path.basename(self.video_path))[0]
            initial_path = os.path.join(output_folder, video_basename)
            initial_filename = f"{video_basename}.funscript"

        if not os.path.isdir(initial_path):
            os.makedirs(initial_path, exist_ok=True)

        chapters = self.app.funscript_processor.video_chapters if self.app.funscript_processor else None

        self.app.gui_instance.file_dialog.show(
            is_save=True,
            title="Export Multi-Axis Funscript (Single File)",
            extension_filter="Funscript Files (*.funscript),*.funscript",
            callback=lambda filepath: self._save_funscript_file_unified(filepath, funscript_obj, chapters),
            initial_path=initial_path,
            initial_filename=initial_filename
        )

    def export_all_axes_ofs(self):
        """Export all axes as separate OFS-named funscript files (auto-save, no dialog)."""
        funscript_obj = None
        if self.app.tracker and hasattr(self.app.tracker, 'funscript'):
            funscript_obj = self.app.tracker.funscript
        if not funscript_obj:
            self.logger.warning("No funscript data available to export.", extra={"status_message": True})
            return

        if not self.video_path:
            self.logger.warning("No video loaded. Open a video first.", extra={"status_message": True})
            return

        chapters = self.app.funscript_processor.video_chapters if self.app.funscript_processor else None
        assignments = funscript_obj.get_axis_assignments()
        saved_count = 0

        for tl_num in sorted(assignments.keys()):
            axis_name = assignments[tl_num]
            if tl_num == 1:
                actions = funscript_obj.get_axis_actions('primary')
            elif tl_num == 2:
                actions = funscript_obj.get_axis_actions('secondary')
            else:
                actions = funscript_obj.get_axis_actions(f'axis_{tl_num}')
                if not actions:
                    actions = funscript_obj.get_axis_actions(axis_name)
            if not actions:
                continue

            path = self._get_funscript_path_for_axis(self.video_path, axis_name)
            chaps = chapters if tl_num == 1 else None
            self._save_funscript_file(path, actions, chaps)
            self.logger.info(f"Exported {axis_name} (T{tl_num}): {os.path.basename(path)}")
            saved_count += 1

        if saved_count:
            self.logger.info(f"Exported {saved_count} funscript file(s) next to video.",
                             extra={"status_message": True})
        else:
            self.logger.warning("No axes with data to export.", extra={"status_message": True})

    def import_unified_funscript(self):
        """Import a multi-axis funscript file (loads primary to T1, embedded axes to T2+)."""
        if not self.app.gui_instance or not self.app.gui_instance.file_dialog:
            return
        self.app.gui_instance.file_dialog.show(
            is_save=False,
            title="Import Multi-Axis Funscript",
            extension_filter="Funscript Files (*.funscript),*.funscript",
            callback=lambda filepath: self.load_funscript_to_timeline(filepath, 1)
        )

    def import_all_axes_ofs(self):
        """Discover and import all OFS-named funscript files for the current video."""
        if not self.video_path:
            self.logger.warning("No video loaded. Open a video first.", extra={"status_message": True})
            return

        discovered = self.discover_axis_funscripts(self.video_path)
        if not discovered:
            self.logger.warning("No funscript files found next to the video.", extra={"status_message": True})
            return

        # Determine axis→timeline mapping
        axis_to_tl = {"stroke": 1, "roll": 2}
        next_tl = 3
        for axis_name in discovered:
            if axis_name not in axis_to_tl:
                axis_to_tl[axis_name] = next_tl
                next_tl += 1

        loaded_count = 0
        for axis_name, filepath in discovered.items():
            tl_num = axis_to_tl.get(axis_name, next_tl)
            self.load_funscript_to_timeline(filepath, tl_num)
            self.logger.info(f"Imported {axis_name} from {os.path.basename(filepath)} to T{tl_num}")
            loaded_count += 1

        self.logger.info(f"Imported {loaded_count} funscript file(s).", extra={"status_message": True})

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
        Automatically saves funscripts for all active axes using OFS naming.
        For remote videos (HTTP URLs), funscripts are always saved to the output directory.
        """
        if not self.app.funscript_processor:
            self.app.logger.error("Funscript processor not available for saving.")
            return

        chapters = self.app.funscript_processor.video_chapters
        save_next_to_video = self.app.app_settings.get("autosave_final_funscript_to_video_location", True)
        is_remote = video_path and video_path.startswith(('http://', 'https://'))

        # Collect all axes to save: (timeline_num, internal_axis_name)
        axes_to_save = [(1, 'primary'), (2, 'secondary')]
        funscript_obj = None
        if self.app.processor and self.app.processor.tracker and self.app.processor.tracker.funscript:
            funscript_obj = self.app.processor.tracker.funscript
            for axis_name in funscript_obj.additional_axes:
                tl_num = funscript_obj.get_timeline_for_axis(axis_name)
                if tl_num is None and axis_name.startswith("axis_"):
                    try:
                        tl_num = int(axis_name.split("_")[1])
                    except (IndexError, ValueError):
                        continue
                if tl_num is not None:
                    axes_to_save.append((tl_num, axis_name))

        saved_count = 0
        for tl_num, internal_axis in axes_to_save:
            actions = self.app.funscript_processor.get_actions(internal_axis)
            if not actions:
                continue

            axis_name = self._resolve_axis_name_for_timeline(tl_num)
            suffix = file_suffix_for_axis(axis_name)

            if save_next_to_video and not is_remote:
                path = self._get_funscript_path_for_axis(video_path, axis_name)
            else:
                path = self.get_output_path_for_file(video_path, f"{suffix}.funscript")

            chaps = chapters if tl_num == 1 else None
            self._save_funscript_file(path, actions, chaps)
            self.logger.info(f"Saved {axis_name} funscript (T{tl_num}): {os.path.basename(path)}")
            saved_count += 1

        if saved_count:
            self.logger.info(f"Batch save complete: {saved_count} funscript(s) saved.")

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
                self.logger.warning(f"Found invalid preprocessed video: {os.path.basename(potential_preprocessed_path)} "
                                  f"({preprocessed_status['frame_count']}/{preprocessed_status['expected_frames']} frames) - "
                                  f"not using, will re-encode if needed")
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
                # Notify audio player of new video
                if self.app._audio_player:
                    has_audio = self.app.processor.video_info.get("has_audio", False)
                    fps = self.app.processor.fps
                    self.app.logger.debug(f"Audio: set_video has_audio={has_audio} fps={fps}")
                    self.app._audio_player.set_video(file_path, has_audio, fps)

                # If it was a new project, auto-discover and load adjacent funscripts.
                if not is_project_load:
                    # Discover all axis funscripts next to the video
                    discovered = self.discover_axis_funscripts(file_path)
                    # Also check the output folder for the primary funscript
                    path_in_output = self.get_output_path_for_file(file_path, ".funscript")
                    if os.path.exists(path_in_output):
                        discovered.setdefault("stroke", path_in_output)

                    if discovered:
                        funscript_obj = None
                        if self.app.processor and self.app.processor.tracker and self.app.processor.tracker.funscript:
                            funscript_obj = self.app.processor.tracker.funscript

                        # Map axis names to timeline numbers for loading
                        axis_to_timeline = {"stroke": 1, "roll": 2}
                        next_timeline = 3

                        for axis_name, fpath in discovered.items():
                            tl_num = axis_to_timeline.get(axis_name)
                            if tl_num is None:
                                tl_num = next_timeline
                                next_timeline += 1
                                # Ensure the axis exists on the funscript object
                                if funscript_obj:
                                    funscript_obj.ensure_axis(f"axis_{tl_num}")

                            self.load_funscript_to_timeline(fpath, timeline_num=tl_num)
                            # Update axis assignment on the funscript object
                            if funscript_obj:
                                funscript_obj.assign_axis(tl_num, axis_name)
                        self.logger.info(f"Auto-discovered {len(discovered)} axis funscript(s) next to video.")

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

        # Stop audio playback
        if self.app._audio_player:
            self.app._audio_player.stop()

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

        self.logger.debug("Video closed.", extra={'status_message': True})
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

            # Notify audio player of new video
            if self.app._audio_player:
                has_audio = self.app.processor.video_info.get("has_audio", False)
                fps = self.app.processor.fps
                self.app.logger.info(f"Audio: set_video has_audio={has_audio} fps={fps}")
                self.app._audio_player.set_video(file_path, has_audio, fps)
            else:
                self.app.logger.debug("Audio: _audio_player is None, skipping set_video")
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

        export_format = self.app.app_settings.get("funscript_export_format", "separate")

        # --- Separate files (OFS) mode ---
        if export_format in ("separate", "both"):
            if primary_actions:
                path_in_output = self.get_output_path_for_file(video_path, ".funscript")
                self._save_funscript_file(path_in_output, primary_actions, chapters_to_save)
                saved_paths.append(path_in_output)

                if save_next_to_video:
                    self.logger.info("Also saving a copy of the final funscript next to the video file.")
                    base, _ = os.path.splitext(video_path)
                    path_next_to_vid = f"{base}.funscript"
                    self._save_funscript_file(path_next_to_vid, primary_actions, chapters_to_save)

            if secondary_actions and generate_roll:
                path_in_output_t2 = self.get_output_path_for_file(video_path, ".roll.funscript")
                self._save_funscript_file(path_in_output_t2, secondary_actions, None)
                saved_paths.append(path_in_output_t2)

                if save_next_to_video:
                    base, _ = os.path.splitext(video_path)
                    path_next_to_vid_t2 = f"{base}.roll.funscript"
                    self._save_funscript_file(path_next_to_vid_t2, secondary_actions, None)

        # --- Unified (embedded axes) mode ---
        if export_format in ("unified", "both"):
            funscript_obj = None
            if self.app.processor and self.app.processor.tracker:
                funscript_obj = getattr(self.app.processor.tracker, 'funscript', None)
            if not funscript_obj and hasattr(self.app, 'multi_axis_funscript'):
                funscript_obj = self.app.multi_axis_funscript
            if funscript_obj and primary_actions:
                suffix = ".unified.funscript" if export_format == "both" else ".funscript"
                path_unified = self.get_output_path_for_file(video_path, suffix)
                self._save_funscript_file_unified(path_unified, funscript_obj, chapters_to_save)
                saved_paths.append(path_unified)

                if save_next_to_video:
                    base, _ = os.path.splitext(video_path)
                    path_unified_vid = f"{base}{suffix}"
                    self._save_funscript_file_unified(path_unified_vid, funscript_obj, chapters_to_save)

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
