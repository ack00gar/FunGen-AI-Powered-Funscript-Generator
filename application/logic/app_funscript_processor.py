import os
import copy
import logging
from typing import List, Dict, Optional, Tuple

from application.utils.video_segment import VideoSegment
from funscript.dual_axis_funscript import DualAxisFunscript
from config.constants import POSITION_INFO_MAPPING_CONST
from application.utils.time_format import _format_time


class AppFunscriptProcessor:
    def __init__(self, app_logic):
        self.app = app_logic
        self.logger = self.app.logger if hasattr(self.app, 'logger') else logging.getLogger(
            "AppFunscriptProcessor_fallback")

        # Chapters and Scripting Range
        self.video_chapters: List[VideoSegment] = []
        self.chapter_bar_height = 20

        # These would be managed and potentially loaded by ProjectManager
        self.selected_chapter_for_scripting: Optional[VideoSegment] = None
        self.scripting_range_active: bool = False
        self.scripting_start_frame: int = 0
        self.scripting_end_frame: int = 0

        # Funscript Attributes (stats are per timeline)
        self.funscript_stats_t1: Dict = self._get_default_funscript_stats()
        self.funscript_stats_t2: Dict = self._get_default_funscript_stats()

        # Selection state for operations (indices for the currently active_axis_for_processing)
        self.current_selection_indices: List[int] = []

        # Funscript Operations Parameters
        self.selected_axis_for_processing: str = 'primary'  # 'primary' or 'secondary'
        self.operation_target_mode: str = 'apply_to_scripting_range'  # or 'apply_to_selected_points'
        self.sg_window_length_input: int = 5
        self.sg_polyorder_input: int = 2
        self.rdp_epsilon_input: float = 8.0
        self.amplify_factor_input: float = 1.1
        self.amplify_center_input: int = 50

        # Clipboard
        self.clipboard_actions_data: List[Dict] = []

    def get_chapter_at_frame(self, frame_index: int) -> Optional[VideoSegment]:
        """
        Efficiently finds the chapter that contains the given frame index.
        Returns None if the frame is not within any chapter (i.e., in a gap).
        Assumes chapters are sorted by start_frame_id.
        """
        # This is a simple linear scan. For a huge number of chapters,
        # a binary search (bisect_right) would be more efficient.
        # For typical use cases, this is fast enough and simpler.
        for chapter in self.video_chapters:
            if chapter.start_frame_id <= frame_index <= chapter.end_frame_id:
                return chapter
        return None

    def get_funscript_obj(self) -> Optional[DualAxisFunscript]:
        if self.app.processor and self.app.processor.tracker and self.app.processor.tracker.funscript:
            return self.app.processor.tracker.funscript
        self.logger.warning("Funscript object not available.")
        return None

    def get_actions(self, axis: str) -> List[dict]:
        funscript_obj = self.get_funscript_obj()
        if funscript_obj:
            if axis == 'primary':
                return funscript_obj.primary_actions
            elif axis == 'secondary':
                return funscript_obj.secondary_actions
        return []

    def _get_default_funscript_stats(self) -> Dict:
        return {
            "source_type": "N/A", "path": "N/A", "num_points": 0,
            "duration_scripted_s": 0.0, "avg_speed_pos_per_s": 0.0,
            "avg_intensity_percent": 0.0, "min_pos": -1, "max_pos": -1,
            "avg_interval_ms": 0.0, "min_interval_ms": -1, "max_interval_ms": -1,
            "total_travel_dist": 0, "num_strokes": 0
        }

    def _get_target_funscript_object_and_axis(self, timeline_num: int) -> Tuple[Optional[object], Optional[str]]:
        """Returns the funscript object and axis name ('primary' or 'secondary')."""
        if self.app.processor and self.app.processor.tracker and self.app.processor.tracker.funscript:
            funscript_obj = self.app.processor.tracker.funscript
            if timeline_num == 1:
                return funscript_obj, 'primary'
            elif timeline_num == 2:
                return funscript_obj, 'secondary'
        return None, None

    def _ensure_undo_managers_linked(self):
        """Ensures undo managers are created and linked if they weren't at init."""
        if not (self.app.processor and self.app.processor.tracker and self.app.processor.tracker.funscript):
            # self.logger.warning("Cannot ensure undo manager linking: Funscript object not available.")
            return

        funscript_obj = self.app.processor.tracker.funscript
        if self.app.undo_manager_t1 is None:
            from application.classes.undo_redo_manager import UndoRedoManager  # Corrected import path
            self.app.undo_manager_t1 = UndoRedoManager(max_history=50)
            self.logger.info("UndoManager T1 created dynamically.")
        if self.app.undo_manager_t1._actions_list_reference is not funscript_obj.primary_actions:
            self.app.undo_manager_t1.set_actions_reference(funscript_obj.primary_actions)
            # self.logger.debug("UndoManager T1 re-linked to primary_actions.")

        if self.app.undo_manager_t2 is None:
            from application.classes.undo_redo_manager import UndoRedoManager  # Corrected import path
            self.app.undo_manager_t2 = UndoRedoManager(max_history=50)
            self.logger.info("UndoManager T2 created dynamically.")
        if self.app.undo_manager_t2._actions_list_reference is not funscript_obj.secondary_actions:
            self.app.undo_manager_t2.set_actions_reference(funscript_obj.secondary_actions)
            # self.logger.debug("UndoManager T2 re-linked to secondary_actions.")

    def _get_undo_manager(self, timeline_num: int) -> Optional[object]:  # Actually UndoRedoManager
        self._ensure_undo_managers_linked()
        if timeline_num == 1: return self.app.undo_manager_t1
        if timeline_num == 2: return self.app.undo_manager_t2
        self.logger.warning(f"Requested undo manager for invalid timeline_num: {timeline_num}")
        return None

    def _get_current_fps(self) -> float:  # Duplicated for internal use, could centralize in app_logic
        fps = 30.0
        if self.app.processor and hasattr(self.app.processor, 'video_info') and \
                self.app.processor.video_info and self.app.processor.video_info.get('fps', 0) > 0:
            fps = self.app.processor.video_info['fps']
        elif self.app.processor and hasattr(self.app.processor, 'fps') and self.app.processor.fps > 0:  # Fallback
            fps = self.app.processor.fps
        return fps

    def _check_chapter_overlap(self, start_frame: int, end_frame: int,
                               existing_chapter_id: Optional[str] = None) -> bool:
        """Checks if the given frame range overlaps with any existing chapters.
           Overlap is defined as sharing one or more frames. [s,e] includes s and e.
        """
        for chapter in self.video_chapters:
            if existing_chapter_id and chapter.unique_id == existing_chapter_id:
                continue  # Skip self when checking for an update

            # Overlap if max(start_frame, chapter.start_frame_id) <= min(end_frame, chapter.end_frame_id)
            if max(start_frame, chapter.start_frame_id) <= min(end_frame, chapter.end_frame_id):
                self.logger.warning(
                    f"Overlap detected: Proposed [{start_frame}-{end_frame}] with existing '{chapter.unique_id}' [{chapter.start_frame_id}-{chapter.end_frame_id}]")
                return True
        return False

    def create_new_chapter_from_data(self, data: Dict,
                                     return_chapter_object: bool = False):  # Added return_chapter_object
        self.logger.info(f"Attempting to create new chapter with data: {data}")
        new_chapter = None  # Initialize
        try:
            start_frame = int(data.get("start_frame_str", "0"))
            end_frame = int(data.get("end_frame_str", "0"))

            if start_frame < 0 or end_frame < start_frame:
                self.logger.error(f"Invalid frame range for new chapter: Start={start_frame}, End={end_frame}")
                return None if return_chapter_object else None  # Explicitly return None

            if self._check_chapter_overlap(start_frame, end_frame):
                self.logger.error("New chapter overlaps with an existing chapter.")
                return None if return_chapter_object else None  # Explicitly return None

            pos_short_key = data.get("position_short_name_key")
            pos_info = POSITION_INFO_MAPPING_CONST.get(pos_short_key, {})
            pos_short_name = pos_info.get("short_name", pos_short_key if pos_short_key else "N/A")
            pos_long_name = pos_info.get("long_name", "Unknown Position")

            derived_class_name = pos_short_key if pos_short_key else "DefaultChapterType"

            new_chapter = VideoSegment(
                start_frame_id=start_frame,
                end_frame_id=end_frame,
                class_id=None,
                class_name=derived_class_name,
                segment_type=data.get("segment_type", "default"),
                position_short_name=pos_short_name,
                position_long_name=pos_long_name,
                source=data.get("source", "manual"),
                color=None
            )
            self.video_chapters.append(new_chapter)
            self.video_chapters.sort(key=lambda c: c.start_frame_id)

            self.app.project_manager.project_dirty = True
            self.app.app_state_ui.heatmap_dirty = True
            self.app.app_state_ui.funscript_preview_dirty = True
            self.logger.info(
                f"Successfully created new chapter: {new_chapter.unique_id} ({new_chapter.position_short_name}) with color {new_chapter.color}")
            self.app.set_status_message(f"Chapter '{new_chapter.position_short_name}' created.")
            # TODO: Add undo action
        except ValueError:
            self.logger.error("Invalid frame number format for new chapter.")
            self.app.set_status_message("Error: Frame numbers must be integers.", level=logging.ERROR)
        except Exception as e:
            self.logger.error(f"Error creating new chapter: {e}", exc_info=True)
            self.app.set_status_message("Error creating chapter.", level=logging.ERROR)

        return new_chapter if return_chapter_object else None

    def update_chapter_from_data(self, chapter_id: str, new_data: Dict):
        self.logger.info(f"Attempting to update chapter {chapter_id} with data: {new_data}")
        chapter_to_update = next((ch for ch in self.video_chapters if ch.unique_id == chapter_id), None)

        if not chapter_to_update:
            self.logger.error(f"Chapter ID {chapter_id} not found for update.")
            self.app.set_status_message("Error: Chapter not found.", level=logging.ERROR)
            return

        try:
            start_frame = int(new_data.get("start_frame_str", str(chapter_to_update.start_frame_id)))
            end_frame = int(new_data.get("end_frame_str", str(chapter_to_update.end_frame_id)))

            if start_frame < 0 or end_frame < start_frame:
                self.logger.error(f"Invalid frame range for chapter update: Start={start_frame}, End={end_frame}")
                return

            if self._check_chapter_overlap(start_frame, end_frame, chapter_id):
                self.logger.error("Updated chapter overlaps with another chapter.")
                return

            chapter_to_update.start_frame_id = start_frame
            chapter_to_update.end_frame_id = end_frame

            pos_short_key = new_data.get("position_short_name_key")
            pos_info = POSITION_INFO_MAPPING_CONST.get(pos_short_key, {})
            chapter_to_update.position_short_name = pos_info.get("short_name",
                                                                 pos_short_key if pos_short_key else chapter_to_update.position_short_name)
            chapter_to_update.position_long_name = pos_info.get("long_name", chapter_to_update.position_long_name)

            # Update class_name based on the new position key
            chapter_to_update.class_name = pos_short_key if pos_short_key else "DefaultChapterType"

            chapter_to_update.segment_type = new_data.get("segment_type", chapter_to_update.segment_type)
            chapter_to_update.source = new_data.get("source", chapter_to_update.source)

            # Force color re-evaluation by calling a helper or reconstructing part of __init__'s color logic
            # This is a simplified re-evaluation. A dedicated method in VideoSegment would be cleaner.
            current_color_before_update = chapter_to_update.color
            temp_segment_for_color = VideoSegment(
                start_frame_id=chapter_to_update.start_frame_id,
                end_frame_id=chapter_to_update.end_frame_id,
                class_id=chapter_to_update.class_id,
                class_name=chapter_to_update.class_name,
                segment_type=chapter_to_update.segment_type,
                position_short_name=chapter_to_update.position_short_name,
                position_long_name=chapter_to_update.position_long_name,
                color=None
            )
            chapter_to_update.color = temp_segment_for_color.color
            if chapter_to_update.color is None:
                chapter_to_update.color = current_color_before_update if current_color_before_update else (0.5, 0.5,
                                                                                                           0.5, 0.7)

            self.video_chapters.sort(key=lambda c: c.start_frame_id)

            self.app.project_manager.project_dirty = True
            self.app.app_state_ui.heatmap_dirty = True
            self.app.app_state_ui.funscript_preview_dirty = True
            self.logger.info(f"Successfully updated chapter: {chapter_id}")
            self.app.set_status_message(f"Chapter '{chapter_to_update.position_short_name}' updated.")
            # TODO: Add undo action using original_data_for_undo
        except ValueError:
            self.logger.error("Invalid frame number format for chapter update.")
            self.app.set_status_message("Error: Frame numbers must be integers.", level=logging.ERROR)
        except Exception as e:
            self.logger.error(f"Error updating chapter {chapter_id}: {e}", exc_info=True)
            self.app.set_status_message("Error updating chapter.", level=logging.ERROR)

    def _record_timeline_action(self, timeline_num: int, action_description: str):
        undo_manager = self._get_undo_manager(timeline_num)
        if undo_manager:
            try:
                # Ensure reference is current (already handled by _ensure_undo_managers_linked)
                undo_manager.record_state_before_action(action_description)
                self.logger.debug(f"UndoRec: T{timeline_num} - '{action_description}'")
            except Exception as e:
                self.logger.error(f"Error recording undo for T{timeline_num} ('{action_description}'): {e}",
                                  exc_info=True)
        else:
            self.logger.warning(f"Could not record undo for T{timeline_num}: Undo manager not found.")

    def perform_undo_redo(self, timeline_num: int, operation: str):  # operation is 'undo' or 'redo'
        undo_manager = self._get_undo_manager(timeline_num)
        if not undo_manager:
            self.logger.info(f"Cannot {operation} on Timeline {timeline_num}: Manager missing.",
                             extra={'status_message': False})
            return

        action_description = None
        success = False
        if operation == 'undo' and undo_manager.can_undo():
            action_description = undo_manager.undo()
            success = action_description is not None
        elif operation == 'redo' and undo_manager.can_redo():
            action_description = undo_manager.redo()
            success = action_description is not None

        if success:
            target_funscript, axis_name = self._get_target_funscript_object_and_axis(timeline_num)
            if target_funscript and axis_name:  # Update funscript object's internal state like last_timestamp
                actions_list = getattr(target_funscript, f"{axis_name}_actions", [])
                last_ts = actions_list[-1]['at'] if actions_list else 0
                setattr(target_funscript, f"last_timestamp_{axis_name}", last_ts)

            self._finalize_action_and_update_ui(timeline_num, f"{operation.capitalize()}: {action_description}")
            self.logger.info(f"Performed {operation.capitalize()} on Timeline {timeline_num}: {action_description}",
                             extra={'status_message': True})
            self.app.energy_saver.reset_activity_timer()
        else:
            self.logger.info(
                f"Cannot {operation} on Timeline {timeline_num}: No actions in history or operation failed.",
                extra={'status_message': False})

    def _finalize_action_and_update_ui(self, timeline_num: int, change_description: str):
        self.update_funscript_stats_for_timeline(timeline_num, change_description)
        self.app.project_manager.project_dirty = True
        if timeline_num == 1:
            self.app.app_state_ui.heatmap_dirty = True
            self.app.app_state_ui.funscript_preview_dirty = True

        # Clear selection if it was for the timeline/axis that just changed
        current_timeline_num_for_selection = 1 if self.selected_axis_for_processing == 'primary' else 2
        if timeline_num == current_timeline_num_for_selection:
            self.current_selection_indices.clear()

    def clear_timeline_history_and_set_new_baseline(self, timeline_num: int, new_actions: list,
                                                    loaded_from_description: str):
        target_funscript, axis_name = self._get_target_funscript_object_and_axis(timeline_num)
        if not target_funscript or not axis_name:
            self.logger.warning(f"Cannot clear/set baseline for T{timeline_num}: Target funscript or axis not found.")
            return

        self._record_timeline_action(timeline_num, f"Replace T{timeline_num} with: {loaded_from_description}")

        live_actions_list_attr_name = f"{axis_name}_actions"
        live_actions_list = getattr(target_funscript, live_actions_list_attr_name, None)
        if live_actions_list is None:  # Should not happen if target_funscript is valid
            self.logger.error(f"Could not get actions list '{live_actions_list_attr_name}' for T{timeline_num}")
            return

        live_actions_list.clear()
        live_actions_list.extend(copy.deepcopy(new_actions))

        last_ts_attr_name = f"last_timestamp_{axis_name}"
        setattr(target_funscript, last_ts_attr_name, new_actions[-1]['at'] if new_actions else 0)

        # Undo manager's redo stack is auto-cleared by record_state_before_action.
        # For major events, explicitly clear full history.
        if any(kw in loaded_from_description for kw in
               ["New Project", "Project Loaded", "Video Closed", "Stage 1 Pending", "Stage 2"]):
            undo_manager = self._get_undo_manager(timeline_num)
            if undo_manager:
                undo_manager.clear_history()
                # After clearing, the state recorded by _record_timeline_action IS the new baseline's undo.
                self.logger.debug(f"Undo history cleared for T{timeline_num} due to: {loaded_from_description}")

        self._finalize_action_and_update_ui(timeline_num, loaded_from_description)
        self.app.energy_saver.reset_activity_timer()

    def update_funscript_stats_for_timeline(self, timeline_num: int, source_type_description: str = "N/A"):
        stats_dict_to_update = self.funscript_stats_t1 if timeline_num == 1 else self.funscript_stats_t2
        default_app_stats = self._get_default_funscript_stats()

        for key in default_app_stats:
            stats_dict_to_update[key] = default_app_stats[key]

        stats_dict_to_update["source_type"] = source_type_description
        if "Loaded T1" in source_type_description and self.app.file_manager.loaded_funscript_path:
            stats_dict_to_update["path"] = os.path.basename(self.app.file_manager.loaded_funscript_path)
        elif "Loaded T2" in source_type_description:
            try:
                stats_dict_to_update["path"] = source_type_description.split(": ", 1)[1]
            except:
                stats_dict_to_update["path"] = "Loaded Funscript (T2)"
        elif "Stage 2" in source_type_description:
            stats_dict_to_update["path"] = "Generated by Stage 2" + (" (Secondary)" if timeline_num == 2 else "")
        elif "Live Tracker" in source_type_description:
            stats_dict_to_update["path"] = "From Live Tracker"
        # else path remains N/A

        target_funscript, axis_name = self._get_target_funscript_object_and_axis(timeline_num)
        if target_funscript and axis_name:
            core_stats = target_funscript.get_actions_statistics(axis=axis_name)
            for key, value in core_stats.items():
                if key in stats_dict_to_update:
                    stats_dict_to_update[key] = value

        if timeline_num == 1:
            self.app.app_state_ui.heatmap_dirty = True
            self.app.app_state_ui.funscript_preview_dirty = True
        # self.app.project_manager.project_dirty = True # Already set by finalize or calling context

    def set_clipboard_actions(self, actions_data: List[Dict]):
        self.clipboard_actions_data = copy.deepcopy(actions_data)
        self.app.energy_saver.reset_activity_timer()
        if actions_data:
            self.logger.info(f"Copied {len(actions_data)} point(s) to clipboard.", extra={'status_message': True})
        else:
            self.logger.info("Clipboard cleared (no points selected to copy).", extra={'status_message': True})

    def get_clipboard_actions(self) -> List[Dict]:
        return copy.deepcopy(self.clipboard_actions_data)

    def clipboard_has_actions(self) -> bool:
        return bool(self.clipboard_actions_data)

        # In app_funscript_processor.py (AppFunscriptProcessor class)

    def clear_actions_in_range_and_inject_new(self, timeline_num: int,
                                              new_actions_for_range: List[Dict],
                                              range_start_ms: int, range_end_ms: int,  # Actual ms for clearing
                                              operation_description: str):
        target_funscript, axis_name = self._get_target_funscript_object_and_axis(timeline_num)
        if not target_funscript or not axis_name:
            self.logger.warning(f"Cannot clear and inject for T{timeline_num}: Target funscript or axis not found.")
            return

        self._record_timeline_action(timeline_num, operation_description)  # Record state BEFORE modification

        live_actions_list_attr_name = f"{axis_name}_actions"
        # Make a copy of current actions to work with for filtering and merging
        original_actions_copy: List[Dict] = list(
            copy.deepcopy(getattr(target_funscript, live_actions_list_attr_name, [])))

        # 1. Preserve actions outside the specified range
        actions_before_range = [action for action in original_actions_copy if action['at'] < range_start_ms]
        actions_after_range = [action for action in original_actions_copy if action['at'] > range_end_ms]

        # 2. Prepare new actions (ensure they are sorted and deepcopied)
        # Stage 2 should provide actions with absolute timestamps already correct for the range.
        processed_new_actions = sorted(copy.deepcopy(new_actions_for_range), key=lambda x: x['at'])

        # 3. Combine the three parts: before, new (for the range), after
        merged_actions = actions_before_range + processed_new_actions + actions_after_range

        # 4. Sort the final combined list by time and ensure unique timestamps
        merged_actions.sort(key=lambda x: x['at'])

        unique_final_actions = []
        if merged_actions:
            unique_final_actions.append(merged_actions[0])
            for i in range(1, len(merged_actions)):
                # Only add if the timestamp is strictly greater, effectively removing exact duplicates.
                # If timestamps were identical, the one from `processed_new_actions` would typically be kept if
                # it was correctly placed by the sort due to its value.
                if merged_actions[i]['at'] > merged_actions[i - 1]['at']:
                    unique_final_actions.append(merged_actions[i])
                else:  # Timestamps are the same, potentially overwrite previous with this one if different pos.
                    # This simple filter just takes the first one for a given timestamp.
                    # A more sophisticated merge might be needed if identical timestamps with different values are common.
                    # For funscript, unique 'at' is typical.
                    # If `merged_actions[i]` came from `processed_new_actions` and `merged_actions[i-1]`
                    # was an original point at the exact same ms, `processed_new_actions` effectively replaces it here.
                    if unique_final_actions and unique_final_actions[-1]['at'] == merged_actions[i]['at']:
                        unique_final_actions[-1] = merged_actions[
                            i]  # Replace with the later one in sorted list (favors new if at same 'at')
                    # This else block might need refinement if identical 'at' values from 'before'/'after' vs 'new' are a concern.
                    # Safest: build a dict keyed by 'at', favoring new_actions, then sort.
                    # Alternative for unique_final_actions:
                    # temp_dict = {}
                    # for action in actions_before_range: temp_dict[action['at']] = action
                    # for action in processed_new_actions: temp_dict[action['at']] = action # Overwrites if 'at' is same
                    # for action in actions_after_range:
                    #    if action['at'] not in temp_dict: temp_dict[action['at']] = action
                    # unique_final_actions = sorted(list(temp_dict.values()), key=lambda x: x['at'])

        # Update the actual live list in the funscript object
        live_actions_list_ref = getattr(target_funscript, live_actions_list_attr_name)
        live_actions_list_ref.clear()
        live_actions_list_ref.extend(unique_final_actions)

        last_ts_attr_name = f"last_timestamp_{axis_name}"
        new_last_ts = unique_final_actions[-1]['at'] if unique_final_actions else 0
        setattr(target_funscript, last_ts_attr_name, new_last_ts)

        self._finalize_action_and_update_ui(timeline_num, operation_description)
        self.logger.info(
            f"Funscript T{timeline_num}: Range [{range_start_ms}-{range_end_ms}]ms updated. Injected {len(processed_new_actions)} new. Total: {len(unique_final_actions)}.",
            extra={'status_message': True})
        self.app.energy_saver.reset_activity_timer()

    def get_effective_scripting_range(self) -> Tuple[bool, Optional[int], Optional[int]]:
        """
        Returns if the scripting range is active and the effective start and end frames.
        -1 for scripting_end_frame is resolved to the total number of video frames if possible.
        If video info is not available, -1 for end_frame results in None (no upper bound).
        """
        if not self.scripting_range_active:
            return False, None, None

        start_f = self.scripting_start_frame
        end_f = self.scripting_end_frame

        if end_f == -1:
            if self.app.processor and self.app.processor.video_info:
                total_frames = self.app.processor.video_info.get('total_frames', 0)
                if total_frames > 0:
                    end_f = total_frames - 1  # 0-indexed
                else:
                    # No video info with total_frames, so -1 means no upper bound effectively
                    self.app.logger.warning(
                        "Scripting range end is -1, but no video total_frames info. Treating as no upper bound.")
                    end_f = None
            else:
                self.logger.warning(
                    "Scripting range end is -1, but no video processor/info. Treating as no upper bound.")
                end_f = None  # No video info, -1 means no upper bound

        # Basic validation for the range itself
        if start_f is not None and end_f is not None and start_f > end_f:
            self.logger.warning(
                f"Scripting range start_frame {start_f} is after end_frame {end_f}. Effective range will be empty for filtering.")
            pass  # Allow Stage 2 to see the invalid range and produce no points

        return True, start_f, end_f

    def frame_to_ms(self, frame_id: int) -> int:
        fps = self._get_current_fps()
        if fps > 0:
            return int(round((frame_id / fps) * 1000))
        # Try to get from Chapters FPS if video not loaded but chapters are
        if self.video_chapters and hasattr(self.video_chapters[0], 'source_fps') and self.video_chapters[
            0].source_fps > 0:
            return int(round((frame_id / self.video_chapters[0].source_fps) * 1000))
        return 0  # Fallback

    def get_script_end_time_ms(self, axis_name: str) -> int:
        actions_list = self.get_actions(axis_name)
        return actions_list[-1]['at'] if actions_list else 0

    def get_processing_args_for_operation(self) -> Tuple[Optional[int], Optional[int], Optional[List[int]]]:
        """Determines start_time, end_time, or selected_indices for a funscript operation."""
        start_time_ms: Optional[int] = None
        end_time_ms: Optional[int] = None
        selected_indices: Optional[List[int]] = None

        if self.operation_target_mode == 'apply_to_selected_points':
            if self.current_selection_indices:
                selected_indices = list(self.current_selection_indices)
            else:
                return None, None, None
        elif self.operation_target_mode == 'apply_to_scripting_range':
            if self.scripting_range_active:
                start_time_ms = self.frame_to_ms(self.scripting_start_frame)
                if self.scripting_end_frame == -1:  # Means end of video/script
                    if self.app.processor and self.app.processor.video_info and self.app.processor.video_info.get(
                            'duration', 0) > 0:
                        end_time_ms = int(self.app.processor.video_info['duration'] * 1000)
                    else:  # Fallback to script end if no video
                        end_time_ms = self.get_script_end_time_ms(self.selected_axis_for_processing)
                else:
                    end_time_ms = self.frame_to_ms(self.scripting_end_frame)

                if start_time_ms is not None and end_time_ms is not None and start_time_ms > end_time_ms:
                    end_time_ms = start_time_ms  # Ensure end is not before start
            # If not scripting_range_active, it implies full script (None, None, None for time means full)
        return start_time_ms, end_time_ms, selected_indices

    def handle_funscript_operation(self, operation_name: str):
        if not (self.app.processor and self.app.processor.tracker and self.app.processor.tracker.funscript):
            self.logger.info("Funscript processor not ready for operation.", extra={'status_message': True})
            return

        s_time, e_time, sel_idx = self.get_processing_args_for_operation()

        if self.operation_target_mode == 'apply_to_selected_points' and not sel_idx:
            self.logger.info("Operation requires selected points, but none are selected.",
                             extra={'status_message': True})
            return

        timeline_num_map = {'primary': 1, 'secondary': 2}
        timeline_num = timeline_num_map.get(self.selected_axis_for_processing)
        if timeline_num is None:
            self.logger.error("Invalid axis for processing.")
            return

        target_fs_obj, axis = self._get_target_funscript_object_and_axis(timeline_num)
        if not target_fs_obj or not axis:
            self.logger.info(f"Target funscript/axis ({self.selected_axis_for_processing}) not available.",
                             extra={'status_message': True})
            return

        action_desc_map = {
            'clamp_0': f"Clamp to 0 ({axis})", 'clamp_100': f"Clamp to 100 ({axis})",
            'invert': f"Invert values ({axis})", 'clear': f"Clear points ({axis})",
            'amplify': f"Amplify (F:{self.amplify_factor_input:.2f}, C:{self.amplify_center_input}) ({axis})",
            'apply_sg': f"Apply SG (W:{self.sg_window_length_input}, P:{self.sg_polyorder_input}) ({axis})",
            'apply_rdp': f"Apply RDP (Eps:{self.rdp_epsilon_input:.2f}) ({axis})"
        }
        action_desc = action_desc_map.get(operation_name)
        if not action_desc:
            self.logger.info(f"Unknown funscript operation: {operation_name}")
            return

        # Validation for specific ops
        if operation_name == 'apply_sg':
            if self.sg_window_length_input < 3 or self.sg_window_length_input % 2 == 0:
                self.logger.info("SG: Window must be odd & >= 3.", extra={'status_message': True})
                return
            if self.sg_polyorder_input < 1 or self.sg_polyorder_input >= self.sg_window_length_input:
                self.logger.info("SG: Polyorder invalid.", extra={'status_message': True})
                return
        if operation_name == 'apply_rdp' and self.rdp_epsilon_input <= 0:
            self.logger.info("RDP: Epsilon must be > 0.", extra={'status_message': True})
            return

        self._record_timeline_action(timeline_num, action_desc)  # Record state BEFORE

        op_dispatch = {
            'clamp_0': lambda: target_fs_obj.clamp_points_values(axis, 0, s_time, e_time, sel_idx),
            'clamp_100': lambda: target_fs_obj.clamp_points_values(axis, 100, s_time, e_time, sel_idx),
            'invert': lambda: target_fs_obj.invert_points_values(axis, s_time, e_time, sel_idx),
            'clear': lambda: target_fs_obj.clear_points(axis, s_time, e_time, sel_idx),
            'amplify': lambda: target_fs_obj.amplify_points_values(axis, self.amplify_factor_input,
                                                                   self.amplify_center_input, s_time, e_time, sel_idx),
            'apply_sg': lambda: target_fs_obj.apply_savitzky_golay(axis, self.sg_window_length_input,
                                                                   self.sg_polyorder_input, s_time, e_time, sel_idx),
            'apply_rdp': lambda: target_fs_obj.simplify_rdp(axis, self.rdp_epsilon_input, s_time, e_time, sel_idx)
        }
        op_func = op_dispatch.get(operation_name)
        if op_func:
            op_func()
        else:
            self.logger.error(f"Dispatch failed for {operation_name}")
            return

        self._finalize_action_and_update_ui(timeline_num, action_desc)
        self.logger.info(f"Applied: {action_desc}", extra={'status_message': True})
        self.app.energy_saver.reset_activity_timer()

    def get_time_range_ms_from_scripting_frames(self) -> Tuple[Optional[float], Optional[float]]:
        fps = self._get_current_fps()
        if not (fps > 0):
            self.logger.info("Video/FPS info needed for time range calculation.", extra={'status_message': True})
            return None, None

        start_ms = (self.scripting_start_frame / fps) * 1000.0

        eff_end_frame = self.scripting_end_frame
        if eff_end_frame == -1:  # To end of video
            eff_end_frame = (
                    self.app.processor.total_frames - 1) if self.app.processor and self.app.processor.total_frames > 0 else self.scripting_start_frame

        end_ms = (eff_end_frame / fps) * 1000.0
        if end_ms < start_ms:
            self.logger.info("Scripting range end time < start time.", extra={'status_message': True})
            return None, None  # Or (start_ms, start_ms)
        return start_ms, end_ms

    def get_scripting_range_display_text(self) -> Tuple[str, str]:
        """Returns display strings for scripting start and end frames."""
        start_display = str(self.scripting_start_frame)
        end_display = str(self.scripting_end_frame)

        if self.scripting_end_frame == -1:
            if self.app.processor and self.app.processor.total_frames > 0:
                end_display = f"{self.app.processor.total_frames - 1} (Video End)"
            elif self.get_actions('primary'):  # Check primary script if no video
                end_display = f"Script End (T1)"
            else:
                end_display = "End (No Media)"
        return start_display, end_display

    def get_operation_target_range_label(self) -> str:
        if self.operation_target_mode == 'apply_to_selected_points':
            return f"{len(self.current_selection_indices)} Selected Point(s)"
        if self.scripting_range_active:
            start_d, end_d = self.get_scripting_range_display_text()
            return f"Frames: {start_d} to {end_d}"
        return "Full Script"  # Default if not selected points and not scripting range

    def reset_scripting_range(self):
        self.scripting_range_active = False
        self.scripting_start_frame = 0
        self.scripting_end_frame = -1
        self.selected_chapter_for_scripting = None

    def update_project_specific_settings(self, project_data: Dict):
        """Called when a project is loaded to update relevant settings."""
        self.video_chapters = [VideoSegment.from_dict(data) for data in project_data.get("video_chapters", []) if
                               VideoSegment.is_valid_dict(data)]

        self.scripting_range_active = project_data.get("scripting_range_active", False)
        self.scripting_start_frame = project_data.get("scripting_start_frame", 0)
        self.scripting_end_frame = project_data.get("scripting_end_frame", -1)

        selected_chapter_id = project_data.get("selected_chapter_for_scripting_id")
        if selected_chapter_id and self.video_chapters:
            self.selected_chapter_for_scripting = next(
                (ch for ch in self.video_chapters if hasattr(ch, 'unique_id') and ch.unique_id == selected_chapter_id),
                None)
        else:
            self.selected_chapter_for_scripting = None

    def get_project_save_data(self) -> Dict:
        """Returns data from this module to be saved in a project file."""
        chapters_serializable = []
        if self.video_chapters and all(hasattr(ch, 'to_dict') for ch in self.video_chapters):  # Check all have method
            chapters_serializable = [chapter.to_dict() for chapter in self.video_chapters]
        elif self.video_chapters:
            self.logger.warning("Some VideoSegment objects lack to_dict() method. Chapters may not be fully saved.")
            chapters_serializable = [chapter.to_dict() for chapter in self.video_chapters if
                                     hasattr(chapter, 'to_dict')]

        return {
            "video_chapters": chapters_serializable,
            "scripting_range_active": self.scripting_range_active,
            "scripting_start_frame": self.scripting_start_frame,
            "scripting_end_frame": self.scripting_end_frame,
            "selected_chapter_for_scripting_id": self.selected_chapter_for_scripting.unique_id if self.selected_chapter_for_scripting and hasattr(
                self.selected_chapter_for_scripting, 'unique_id') else None,
        }

    # --- Functions for Context Menu Actions ---

    def request_create_new_chapter(self):
        # This would typically open a dialog. For now, just log.
        self.logger.info("Request to create a new chapter received (UI Dialog Needed).")
        # Placeholder: Add a default chapter for demonstration if needed
        # default_start = self.app.processor.current_frame_index if self.app.processor else 0
        # default_end = default_start + (self.app.processor.video_info.get('fps', 30) * 5) # 5 seconds
        # new_chapter = VideoSegment(start_frame_id=default_start, end_frame_id=default_end, ...)
        # self.video_chapters.append(new_chapter)
        # self.app.project_manager.project_dirty = True
        self.app.set_status_message("Create New Chapter: Not fully implemented (needs UI dialog).")

    def request_edit_chapter(self, chapter_to_edit: VideoSegment):
        if not chapter_to_edit:
            self.logger.warning("Request to edit chapter received, but no chapter provided.")
            return
        self.logger.info(f"Request to edit chapter '{chapter_to_edit.unique_id}' received (UI Dialog Needed).")
        self.app.set_status_message(
            f"Edit Chapter {chapter_to_edit.position_short_name}: Not fully implemented (needs UI dialog).")

    def delete_video_chapters_by_ids(self, chapter_ids: List[str]):
        if not chapter_ids:
            self.logger.info("No chapter IDs provided for deletion.")
            return

        initial_count = len(self.video_chapters)
        self.video_chapters = [ch for ch in self.video_chapters if ch.unique_id not in chapter_ids]
        deleted_count = initial_count - len(self.video_chapters)

        if deleted_count > 0:
            self.logger.info(f"Deleted {deleted_count} chapter(s): {chapter_ids}")
            self.app.project_manager.project_dirty = True
            self.app.app_state_ui.heatmap_dirty = True
            self.app.app_state_ui.funscript_preview_dirty = True
            # TODO: Add Undo/Redo record
            self.app.set_status_message(f"Deleted {deleted_count} chapter(s).")
        else:
            self.logger.info(f"No chapters found matching IDs for deletion: {chapter_ids}")
            self.app.set_status_message("No matching chapters found to delete.")

    def clear_script_points_in_selected_chapters(self, selected_chapters: List[VideoSegment]):
        if not selected_chapters:
            self.logger.info("No chapters selected to clear points from.")
            return

        funscript_obj = self.get_funscript_obj()
        if not funscript_obj:
            self.logger.error("Funscript object not found. Cannot clear points.")
            return

        fps = self._get_current_fps()
        if fps == 30.0 and not (
                self.app.processor and hasattr(self.app.processor, 'video_info') and self.app.processor.video_info):
            self.logger.warning(
                f"Valid FPS not found, using default {fps}fps for point clearing. Accuracy may be affected.")

        cleared_any_points = False
        for chapter in selected_chapters:
            start_ms = int(round((chapter.start_frame_id / fps) * 1000.0))
            end_ms = int(round((chapter.end_frame_id / fps) * 1000.0))

            if start_ms >= end_ms:
                self.logger.warning(
                    f"Chapter {chapter.unique_id} has invalid time range for point clearing: {start_ms}ms - {end_ms}ms. Skipping.")
                continue

            self.logger.info(
                f"Clearing script points in chapter '{chapter.unique_id}' (Frames: {chapter.start_frame_id}-{chapter.end_frame_id}, Time: {start_ms}ms-{end_ms}ms)")
            funscript_obj.clear_actions_in_time_range(start_ms, end_ms, axis='both')
            cleared_any_points = True  # Assume it might have cleared something if called

            # TODO: Add Undo/Redo record for point deletion
            # Example:
            # self.app.undo_manager_t1.add_action(
            #     lambda: setattr(funscript_obj, 'primary_actions', [a.copy() for a in actions_before_primary]), # redo
            #     lambda: funscript_obj.clear_actions_in_time_range(start_ms, end_ms, axis='primary') # undo (problematic if multiple calls)
            # )
            # A better undo for point clearing would restore the exact points deleted.

        if cleared_any_points:
            self.app.project_manager.project_dirty = True
            self.app.app_state_ui.heatmap_dirty = True
            self.app.app_state_ui.funscript_preview_dirty = True
            self.update_funscript_stats_for_timeline(1, "Points Cleared in Chapter")
            self.update_funscript_stats_for_timeline(2, "Points Cleared in Chapter")
            self.app.set_status_message(f"Cleared script points in {len(selected_chapters)} chapter(s).")

    def merge_selected_chapters(self, chapter1: VideoSegment, chapter2: VideoSegment,
                                return_chapter_object: bool = False):
        if not chapter1 or not chapter2:
            self.logger.error("Two chapters must be provided for merging.")
            return
        if chapter1.unique_id == chapter2.unique_id:
            self.logger.warning("Cannot merge a chapter with itself.")
            return

        # Ensure chapter1 is the earlier one
        if chapter1.start_frame_id > chapter2.start_frame_id:
            chapter1, chapter2 = chapter2, chapter1  # Ensure chapter1 is earlier

        new_start_frame = chapter1.start_frame_id
        new_end_frame = max(chapter1.end_frame_id, chapter2.end_frame_id)

        # Check for overlap with *other* chapters BEFORE creating the new one
        # Exclude the two chapters being merged from this specific check
        ids_to_ignore_for_overlap_check = {chapter1.unique_id, chapter2.unique_id}
        temp_chapters_for_check = [ch for ch in self.video_chapters if
                                   ch.unique_id not in ids_to_ignore_for_overlap_check]

        for other_ch in temp_chapters_for_check:
            if max(new_start_frame, other_ch.start_frame_id) <= min(new_end_frame, other_ch.end_frame_id):
                self.logger.error(
                    f"Merge failed: Resulting chapter [{new_start_frame}-{new_end_frame}] would overlap with existing chapter '{other_ch.unique_id}' [{other_ch.start_frame_id}-{other_ch.end_frame_id}].")
                self.app.set_status_message("Error: Merge would cause overlap with another chapter.",
                                            level=logging.ERROR)
                return

        merged_pos_short_key = chapter1.position_short_name
        merged_pos_info = POSITION_INFO_MAPPING_CONST.get(merged_pos_short_key, {})
        merged_pos_short_name = merged_pos_info.get("short_name",
                                                    merged_pos_short_key if merged_pos_short_key else "N/A")
        merged_pos_long_name = f"Merged: {chapter1.position_long_name} & {chapter2.position_long_name}"
        if len(merged_pos_long_name) > 70: merged_pos_long_name = merged_pos_long_name[:67] + "..."

        # Derived class_name from position key
        merged_derived_class_name = merged_pos_short_key if merged_pos_short_key else "MergedChapter"

        merged_chapter = VideoSegment(
            start_frame_id=new_start_frame,
            end_frame_id=new_end_frame,
            class_id=chapter1.class_id,
            class_name=merged_derived_class_name,
            segment_type=chapter1.segment_type,
            position_short_name=merged_pos_short_name,
            position_long_name=merged_pos_long_name,
            source="manual_merge",
            color=None
        )
        # Duration will be calculated by VideoSegment if not passed, or we can set it
        merged_chapter.duration = new_end_frame - new_start_frame

        ids_to_delete = {chapter1.unique_id, chapter2.unique_id}
        self.video_chapters = [ch for ch in self.video_chapters if ch.unique_id not in ids_to_delete]
        self.video_chapters.append(merged_chapter)
        self.video_chapters.sort(key=lambda c: c.start_frame_id)

        self.logger.info(
            f"Merged chapters '{chapter1.unique_id}' and '{chapter2.unique_id}' into new chapter '{merged_chapter.unique_id}'.")
        self.app.project_manager.project_dirty = True
        self.app.app_state_ui.heatmap_dirty = True
        self.app.app_state_ui.funscript_preview_dirty = True
        # TODO: Add Undo/Redo record
        self.app.set_status_message("Chapters merged successfully.")
        return merged_chapter if return_chapter_object else None

    def finalize_merge_after_gap_tracking(self, chapter1_id: str, chapter2_id: str):
        self.logger.info(f"Finalizing merge after gap tracking for chapters: {chapter1_id}, {chapter2_id}")

        chapter1 = next((ch for ch in self.video_chapters if ch.unique_id == chapter1_id), None)
        chapter2 = next((ch for ch in self.video_chapters if ch.unique_id == chapter2_id), None)

        if not chapter1 or not chapter2:
            self.logger.error(f"Could not find original chapters for final merge: C1={chapter1_id}, C2={chapter2_id}")
            # Attempt to clean up funscript even if chapters are gone, though unlikely.
            funscript_obj_check = self.get_funscript_obj()
            if funscript_obj_check:
                funscript_obj_check.primary_actions.sort(key=lambda x: x['at'])  # Basic sort at least
            return

        funscript_obj = self.get_funscript_obj()
        if not funscript_obj:
            self.logger.error("Funscript object not available for finalizing merge.")
            return

        # The funscript (e.g., funscript_obj.primary_actions) has ALREADY been modified by the live tracker
        # for the gap range. We just need to ensure it's clean and then update chapters.

        actions_list_ref = funscript_obj.primary_actions

        if actions_list_ref:
            actions_list_ref.sort(key=lambda x: x['at'])
            unique_actions = []
            if actions_list_ref:
                unique_actions.append(actions_list_ref[0])
                for i in range(1, len(actions_list_ref)):
                    if actions_list_ref[i]['at'] > unique_actions[-1]['at']:
                        unique_actions.append(actions_list_ref[i])

            actions_list_ref[:] = unique_actions  # Update in place

            if funscript_obj.primary_actions:
                funscript_obj.last_timestamp_primary = funscript_obj.primary_actions[-1]['at']
            else:
                funscript_obj.last_timestamp_primary = 0
        # Similar logic if secondary axis was tracked for the gap.

        # Now, perform the chapter merge logic (similar to merge_chapters_across_gap or merge_selected_chapters)
        # This will create one new chapter spanning the old C1, the gap, and the old C2.

        # Ensure chapter1 is the earlier one for definition
        if chapter1.start_frame_id > chapter2.start_frame_id:
            chapter1, chapter2 = chapter2, chapter1

        new_merged_start_frame = chapter1.start_frame_id
        new_merged_end_frame = chapter2.end_frame_id

        # Check for overlap with *other* chapters (excluding the two being merged)
        ids_to_ignore_for_overlap_check = {chapter1_id, chapter2_id}
        temp_chapters_for_check = [ch for ch in self.video_chapters if
                                   ch.unique_id not in ids_to_ignore_for_overlap_check]
        for other_ch in temp_chapters_for_check:
            if max(new_merged_start_frame, other_ch.start_frame_id) <= min(new_merged_end_frame, other_ch.end_frame_id):
                self.logger.error(
                    f"Final merge operation aborted: Resulting chapter [{new_merged_start_frame}-{new_merged_end_frame}] "
                    f"would overlap with existing chapter '{other_ch.unique_id}'. Funscript for gap remains.")
                self.app.set_status_message("Error: Merge would cause chapter overlap. Gap tracked.",
                                            level=logging.ERROR)
                # The funscript was already modified by tracking. The undo will handle reverting it.
                # No chapter changes are made in this error case.
                return

        # Use properties from the first chapter for the merged chapter metadata
        merged_pos_short_key = chapter1.position_short_name
        merged_pos_info = POSITION_INFO_MAPPING_CONST.get(merged_pos_short_key, {})
        merged_pos_short_name = merged_pos_info.get("short_name",
                                                    merged_pos_short_key if merged_pos_short_key else "N/A")
        merged_pos_long_name = f"Merged (Gap Tracked): {chapter1.position_long_name} & {chapter2.position_long_name}"
        if len(merged_pos_long_name) > 70: merged_pos_long_name = merged_pos_long_name[:67] + "..."
        merged_derived_class_name = chapter1.class_name  # Or a new derived name like "MergedGapTracked"

        merged_chapter = VideoSegment(
            start_frame_id=new_merged_start_frame,
            end_frame_id=new_merged_end_frame,
            class_id=chapter1.class_id,
            class_name=merged_derived_class_name,
            segment_type=chapter1.segment_type,
            position_short_name=merged_pos_short_name,
            position_long_name=merged_pos_long_name,
            source="manual_gap_track_merge"
            # Color will be auto-assigned by VideoSegment
        )

        # Update chapter list: remove old chapters, add new merged one
        self.video_chapters = [ch for ch in self.video_chapters if ch.unique_id not in [chapter1_id, chapter2_id]]
        self.video_chapters.append(merged_chapter)
        self.video_chapters.sort(key=lambda c: c.start_frame_id)

        self.app.project_manager.project_dirty = True
        self.app.app_state_ui.heatmap_dirty = True
        self.app.app_state_ui.funscript_preview_dirty = True
        self.update_funscript_stats_for_timeline(1, "Gap Tracked & Chapters Merged")
        # self.update_funscript_stats_for_timeline(2, "Gap Tracked & Chapters Merged") # If T2 involved

        self.logger.info(
            f"Successfully finalized tracking of gap and merged into new chapter '{merged_chapter.unique_id}'.")
        self.app.set_status_message("Gap tracked and chapters merged successfully.")
        self.app.energy_saver.reset_activity_timer()

    def merge_chapters_across_gap(self, chapter1: VideoSegment, chapter2: VideoSegment) -> Optional[
        VideoSegment]:  # Add return type hint
        if not chapter1 or not chapter2:
            self.logger.error("Two chapters must be provided for merging across a gap.")
            self.app.set_status_message("Error: Two chapters needed for merge.", level=logging.ERROR)
            return None  # Explicitly return None
        if chapter1.unique_id == chapter2.unique_id:
            self.logger.warning("Cannot merge a chapter with itself (across gap).")
            # Optionally set a status message if desired
            return None  # Explicitly return None

        # Ensure chapter1 is the earlier one, already handled by UI sort before call usually
        if chapter1.start_frame_id > chapter2.start_frame_id:
            chapter1, chapter2 = chapter2, chapter1

        new_start_frame = chapter1.start_frame_id
        new_end_frame = chapter2.end_frame_id  # This is the key difference: always use chapter2's end.

        # Check for overlap with *other* chapters BEFORE creating the new one
        ids_to_ignore_for_overlap_check = {chapter1.unique_id, chapter2.unique_id}
        temp_chapters_for_check = [ch for ch in self.video_chapters if
                                   ch.unique_id not in ids_to_ignore_for_overlap_check]

        for other_ch in temp_chapters_for_check:
            if max(new_start_frame, other_ch.start_frame_id) <= min(new_end_frame, other_ch.end_frame_id):
                self.logger.error(
                    f"Merge across gap failed: Resulting chapter [{new_start_frame}-{new_end_frame}] would overlap with existing chapter '{other_ch.unique_id}' [{other_ch.start_frame_id}-{other_ch.end_frame_id}].")
                self.app.set_status_message("Error: Merge would cause overlap with another chapter.",
                                            level=logging.ERROR)
                return None  # Return None on overlap failure

        # Use properties from the first chapter
        merged_pos_short_key = chapter1.position_short_name
        merged_pos_info = POSITION_INFO_MAPPING_CONST.get(merged_pos_short_key, {})
        merged_pos_short_name = merged_pos_info.get("short_name",
                                                    merged_pos_short_key if merged_pos_short_key else "N/A")
        merged_pos_long_name = f"Filled Gap from {chapter1.position_short_name} to {chapter2.position_short_name}"
        if len(merged_pos_long_name) > 70: merged_pos_long_name = merged_pos_long_name[:67] + "..."

        merged_derived_class_name = merged_pos_short_key if merged_pos_short_key else "GapFilledChapter"

        merged_chapter = VideoSegment(
            start_frame_id=new_start_frame,
            end_frame_id=new_end_frame,
            class_id=chapter1.class_id,
            class_name=merged_derived_class_name,
            segment_type=chapter1.segment_type,
            position_short_name=merged_pos_short_name,
            position_long_name=merged_pos_long_name,
            source="manual_merge_gap_fill",
            color=None
        )
        merged_chapter.duration = new_end_frame - new_start_frame

        ids_to_delete = {chapter1.unique_id, chapter2.unique_id}
        self.video_chapters = [ch for ch in self.video_chapters if ch.unique_id not in ids_to_delete]
        self.video_chapters.append(merged_chapter)
        self.video_chapters.sort(key=lambda c: c.start_frame_id)

        self.logger.info(
            f"Filled gap and merged chapters '{chapter1.unique_id}' and '{chapter2.unique_id}' into new chapter '{merged_chapter.unique_id}'.")
        self.app.project_manager.project_dirty = True
        self.app.app_state_ui.heatmap_dirty = True
        self.app.app_state_ui.funscript_preview_dirty = True
        self.app.set_status_message("Chapters merged across gap successfully.")

        return merged_chapter  # Return the newly created chapter

    def set_scripting_range_from_chapter(self, chapter: VideoSegment):
        if not chapter:
            self.logger.warning("Attempted to set scripting range from None chapter.")
            return

        self.scripting_start_frame = chapter.start_frame_id
        self.scripting_end_frame = chapter.end_frame_id
        self.scripting_range_active = True
        self.selected_chapter_for_scripting = chapter

        self.app.project_manager.project_dirty = True
        current_fps_for_log = self._get_current_fps()
        start_t_str = _format_time(self.app,
                                   chapter.start_frame_id / current_fps_for_log if current_fps_for_log > 0 else 0)
        end_t_str = _format_time(self.app, chapter.end_frame_id / current_fps_for_log if current_fps_for_log > 0 else 0)
        self.logger.info(
            f"Scripting range auto-set to chapter: {chapter.position_short_name} [{start_t_str} - {end_t_str}] (Frames: {self.scripting_start_frame}-{self.scripting_end_frame})",
            extra={'status_message': True}
        )
        if hasattr(self.app, 'energy_saver'):
            self.app.energy_saver.reset_activity_timer()

    def reset_state_for_new_project(self):  # Added from app_logic context
        self.logger.debug("AppFunscriptProcessor resetting state for new project.")
        self.video_chapters = []
        self.selected_chapter_for_scripting = None
        self.scripting_range_active = False
        self.scripting_start_frame = 0
        self.scripting_end_frame = 0
        funscript_obj = self.get_funscript_obj()
        if funscript_obj:
            # funscript_obj.clear() # Clearing funscript usually handled by FileManage.close_video or tracker.reset
            pass
        self.update_funscript_stats_for_timeline(1, "Project Reset")
        self.update_funscript_stats_for_timeline(2, "Project Reset")

    def apply_automatic_post_processing(self, frame_range: Optional[Tuple[int, int]] = None):
        """
        Applies a series of post-processing steps to the funscript(s).
        If frame_range is provided (e.g., from a live tracking session), processing is limited to that range.
        Otherwise, it applies to the full script.
        """
        funscript_obj = self.get_funscript_obj()
        if not funscript_obj:
            self.logger.warning("Post-Processing: Funscript object not available.")
            return

        start_ms: Optional[int] = None
        end_ms: Optional[int] = None
        current_fps = self._get_current_fps()

        if frame_range and current_fps > 0:
            start_frame, end_frame = frame_range
            start_ms = self.frame_to_ms(start_frame)
            end_ms = self.frame_to_ms(end_frame) if end_frame != -1 else None  # None means to the end of the script
            self.logger.info(
                f"--- Starting Post-Processing for range: Frames {start_frame}-{end_frame if end_frame != -1 else 'End'} ({start_ms}ms-{end_ms or 'End'}ms) ---")
        else:
            self.logger.info("--- Starting Post-Processing for full script ---")

        # --- Get Parameters ---
        sg_window = self.app.app_settings.get("auto_post_processing_sg_window", 7)
        sg_polyorder = self.app.app_settings.get("auto_post_processing_sg_polyorder", 3)
        rdp_epsilon = self.app.app_settings.get("auto_post_processing_rdp_epsilon", 1.5)
        clamp_lower_thresh_pri = self.app.app_settings.get("auto_post_processing_clamp_lower_threshold_primary", 10)
        clamp_upper_thresh_pri = self.app.app_settings.get("auto_post_processing_clamp_upper_threshold_primary", 90)
        amp_config = self.app.app_settings.get("auto_post_processing_amplification_config", {})
        default_amp_params = amp_config.get("Default", {"scale_factor": 1.0, "center_value": 50})

        # --- Process Primary Axis ---
        if funscript_obj.primary_actions:
            self.logger.info("Post-Processing: Primary Axis...")
            op_desc = "Auto Post-Process (Primary)" + (f" on range" if frame_range else "")
            self._record_timeline_action(1, op_desc)

            self.logger.info(f"Applying Savitzky-Golay (Primary): Window={sg_window}, Polyorder={sg_polyorder}")
            funscript_obj.apply_savitzky_golay('primary', sg_window, sg_polyorder, start_time_ms=start_ms,
                                               end_time_ms=end_ms)

            self.logger.info(f"Applying RDP Simplification (Primary): Epsilon={rdp_epsilon}")
            funscript_obj.simplify_rdp('primary', rdp_epsilon, start_time_ms=start_ms, end_time_ms=end_ms)

            self.logger.info(
                f"Applying Threshold Clamping (Primary): Lower={clamp_lower_thresh_pri}, Upper={clamp_upper_thresh_pri}")
            funscript_obj.clamp_points_thresholded('primary', clamp_lower_thresh_pri, clamp_upper_thresh_pri,
                                                   start_time_ms=start_ms, end_time_ms=end_ms)

            # Amplification based on chapter type
            if self.video_chapters:
                self.logger.info("Applying chapter-based amplification (Primary)...")
                for chapter in self.video_chapters:
                    chapter_start_ms = self.frame_to_ms(chapter.start_frame_id)
                    chapter_end_ms = self.frame_to_ms(chapter.end_frame_id)

                    # Determine the actual processing range for this chapter by intersecting with the global range
                    effective_start_ms = max(start_ms, chapter_start_ms) if start_ms is not None else chapter_start_ms
                    effective_end_ms = min(end_ms, chapter_end_ms) if end_ms is not None else chapter_end_ms

                    if effective_end_ms <= effective_start_ms:
                        continue  # This chapter is outside the processing range

                    chapter_key = chapter.position_long_name
                    params = amp_config.get(chapter_key, default_amp_params)

                    self.logger.debug(
                        f"Amplifying Primary: Chapter '{chapter_key}' ({effective_start_ms}ms - {effective_end_ms}ms) with Scale={params['scale_factor']}, Center={params['center_value']}")
                    funscript_obj.amplify_points_values(
                        axis='primary',
                        scale_factor=params["scale_factor"],
                        center_value=params["center_value"],
                        start_time_ms=effective_start_ms,
                        end_time_ms=effective_end_ms
                    )
            else:
                self.logger.info(f"No chapters found. Applying default amplification to the processing range.")
                funscript_obj.amplify_points_values(
                    axis='primary',
                    scale_factor=default_amp_params["scale_factor"],
                    center_value=default_amp_params["center_value"],
                    start_time_ms=start_ms,
                    end_time_ms=end_ms
                )
            self._finalize_action_and_update_ui(1, op_desc)

        # --- Process Secondary Axis (SG and RDP only) ---
        if funscript_obj.secondary_actions:
            self.logger.info("Post-Processing: Secondary Axis...")
            op_desc_sec = "Auto Post-Process (Secondary)" + (f" on range" if frame_range else "")
            self._record_timeline_action(2, op_desc_sec)

            self.logger.info(f"Applying Savitzky-Golay (Secondary): Window={sg_window}, Polyorder={sg_polyorder}")
            funscript_obj.apply_savitzky_golay('secondary', sg_window, sg_polyorder, start_time_ms=start_ms,
                                               end_time_ms=end_ms)

            self.logger.info(f"Applying RDP Simplification (Secondary): Epsilon={rdp_epsilon}")
            funscript_obj.simplify_rdp('secondary', rdp_epsilon, start_time_ms=start_ms, end_time_ms=end_ms)

            self._finalize_action_and_update_ui(2, op_desc_sec)

        self.logger.info("--- Post-Processing Finished ---")
        self.app.set_status_message("Post-processing applied.", duration=5.0)
        self.app.energy_saver.reset_activity_timer()
