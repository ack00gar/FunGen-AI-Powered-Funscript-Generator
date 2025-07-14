import imgui
import os
import numpy as np
import math
import glfw
from typing import Optional, List, Dict, Tuple
from bisect import bisect_left, bisect_right

from application.utils.time_format import _format_time


class InteractiveFunscriptTimeline:
    def __init__(self, app_instance, timeline_num: int):
        self.app = app_instance
        self.timeline_num = timeline_num

        self.selected_action_idx = -1
        self.dragging_action_idx = -1
        self.drag_start_action_state: Optional[Dict] = None
        self.drag_undo_recorded = False
        self.context_mouse_pos_screen = (0, 0)
        self.new_point_candidate_at = 0
        self.new_point_candidate_pos = 50

        self.show_sg_settings_popup = False
        self.sg_apply_to_selection = False
        self.show_rdp_settings_popup = False
        self.rdp_apply_to_selection = False

        # --- AMP STATE ---
        self.show_amp_settings_popup = False
        self.amp_apply_to_selection = False
        self.amp_scale_factor = self.app.app_settings.get(f"timeline{self.timeline_num}_amp_default_scale", 1.5)
        self.amp_center_value = self.app.app_settings.get(f"timeline{self.timeline_num}_amp_default_center", 50)

        # --- KEYFRAME STATE ---
        self.show_keyframe_settings_popup = False
        self.keyframe_apply_to_selection = False
        self.keyframe_position_tolerance = self.app.app_settings.get(
            f"timeline{self.timeline_num}_keyframe_default_pos_tol", 10)
        self.keyframe_time_tolerance = self.app.app_settings.get(
            f"timeline{self.timeline_num}_keyframe_default_time_tol", 50)

        # --- SPEED LIMITER STATE ---
        self.show_speed_limiter_popup = False
        self.min_interval = self.app.app_settings.get(f"timeline{self.timeline_num}_min_interval", 60)
        self.vibe_amount = self.app.app_settings.get(f"timeline{self.timeline_num}_vibe_amount", 0)
        self.speed_threshold = self.app.app_settings.get(f"timeline{self.timeline_num}_speed_threshold", 500.0)

        # Get initial defaults from app_settings (AppLogic holds app_settings directly)
        self.sg_window_length = self.app.app_settings.get(f"timeline{self.timeline_num}_sg_default_window", 5)
        self.sg_poly_order = self.app.app_settings.get(f"timeline{self.timeline_num}_sg_default_polyorder", 2)
        self.rdp_epsilon = self.app.app_settings.get(f"timeline{self.timeline_num}_rdp_default_epsilon", 8.0)

        # --- PEAKS STATE ---
        self.show_peaks_settings_popup = False
        self.peaks_apply_to_selection = False
        self.peaks_height = self.app.app_settings.get(f"timeline{self.timeline_num}_peaks_default_height", 0)
        self.peaks_threshold = self.app.app_settings.get(f"timeline{self.timeline_num}_peaks_default_threshold", 0)
        self.peaks_distance = self.app.app_settings.get(f"timeline{self.timeline_num}_peaks_default_distance", 1)
        self.peaks_prominence = self.app.app_settings.get(f"timeline{self.timeline_num}_peaks_default_prominence", 0)
        self.peaks_width = self.app.app_settings.get(f"timeline{self.timeline_num}_peaks_default_width", 0)

        self.multi_selected_action_indices = set()
        self.is_marqueeing = False
        self.marquee_start_screen_pos = None
        self.marquee_end_screen_pos = None

        self.shift_frames_amount = 1

        # Unified interaction flag, replacing is_interacting_with_pan_zoom and app_state.timeline_interaction_active
        self.is_interacting: bool = False
        self.is_panning_active: bool = False
        self.is_zooming_active: bool = False

        self.zoom_action_request: Optional[float] = None

        self.is_previewing: bool = False
        self.preview_actions: Optional[List[Dict]] = None

        # For range selection
        self.selection_anchor_idx: int = -1

        # Stores the index of the point that was right-clicked to open the context menu
        self.context_menu_point_idx: int = -1

    def _get_target_funscript_details(self) -> Tuple[Optional[object], Optional[str]]:
        if self.app.funscript_processor:
            return self.app.funscript_processor._get_target_funscript_object_and_axis(self.timeline_num)
        return None, None

    def _get_actions_list_ref(self) -> Optional[List[Dict]]:
        funscript_instance, axis_name = self._get_target_funscript_details()
        if funscript_instance and axis_name:
            return getattr(funscript_instance, f"{axis_name}_actions", None)
        return None

    def _perform_time_shift(self, frame_delta: int):
        fs_proc = self.app.funscript_processor
        video_fps_for_calc = self.app.processor.fps if self.app.processor and self.app.processor.fps and self.app.processor.fps > 0 else 0
        if video_fps_for_calc <= 0:
            self.app.logger.warning(f"T{self.timeline_num}: Cannot shift time. Video FPS is not available.", extra={'status_message': True})
            return

        time_delta_ms = int(round((frame_delta / video_fps_for_calc) * 1000.0))
        op_desc = f"Shifted All Points by {frame_delta} frames"

        fs_proc._record_timeline_action(self.timeline_num, op_desc)
        if self._call_funscript_method('shift_points_time', 'time shift', time_delta_ms=time_delta_ms):
            fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)
            self.app.logger.info(f"{op_desc} on T{self.timeline_num}.", extra={'status_message': True})

    # --- COPY/PASTE HELPER METHODS ---
    def _get_selected_actions_for_copy(self) -> List[Dict]:
        actions_list_ref = self._get_actions_list_ref()
        if not actions_list_ref: return []
        actions_to_copy_refs = []
        if self.multi_selected_action_indices:
            # Ensure indices are valid and sorted before accessing
            valid_indices = sorted(
                [idx for idx in self.multi_selected_action_indices if 0 <= idx < len(actions_list_ref)])
            for idx in valid_indices: actions_to_copy_refs.append(actions_list_ref[idx])
        elif self.selected_action_idx != -1 and 0 <= self.selected_action_idx < len(actions_list_ref):
            actions_to_copy_refs.append(actions_list_ref[self.selected_action_idx])
        if not actions_to_copy_refs: return []
        actions_to_copy_refs.sort(key=lambda x: x['at'])
        earliest_at = actions_to_copy_refs[0]['at']
        return [{"relative_at": action['at'] - earliest_at, "pos": action['pos']} for action in actions_to_copy_refs]

    def _handle_copy_selection(self):
        actions_to_copy = self._get_selected_actions_for_copy()
        self.app.funscript_processor.set_clipboard_actions(actions_to_copy)

    def _handle_paste_actions(self, paste_at_time_ms: int):
        funscript_instance, axis_name = self._get_target_funscript_details()
        if not funscript_instance or not axis_name: return

        clipboard_data = self.app.funscript_processor.get_clipboard_actions()
        if not clipboard_data:
            self.app.logger.info(f"T{self.timeline_num}: Clipboard empty.")
            return

        op_desc = f"Pasted {len(clipboard_data)} Point(s)"
        self.app.funscript_processor._record_timeline_action(self.timeline_num, op_desc)

        actions_to_add = []
        newly_pasted_timestamps = []
        for action_data in clipboard_data:
            new_at = max(0, int(paste_at_time_ms + action_data['relative_at']))
            new_pos = int(action_data['pos'])
            actions_to_add.append({
                'timestamp_ms': new_at,
                'primary_pos': new_pos if axis_name == 'primary' else None,
                'secondary_pos': new_pos if axis_name == 'secondary' else None
            })
            newly_pasted_timestamps.append(new_at)

        if not actions_to_add: return

        # Use the batch add method for efficiency and correctness
        funscript_instance.add_actions_batch(actions_to_add, is_from_live_tracker=False)

        updated_actions_list_ref = self._get_actions_list_ref()
        if not updated_actions_list_ref: return
        self.multi_selected_action_indices.clear()
        new_selected_indices = set()

        # A simple way to find pasted actions is by their timestamp, though this can be ambiguous.
        # A more robust method would be needed if timestamps could easily collide.
        for ts_pasted in newly_pasted_timestamps:
            for i, act in enumerate(updated_actions_list_ref):
                if act['at'] == ts_pasted:
                    new_selected_indices.add(i)

        self.multi_selected_action_indices = new_selected_indices
        self.selected_action_idx = min(self.multi_selected_action_indices) if self.multi_selected_action_indices else -1

        self.app.funscript_processor._finalize_action_and_update_ui(self.timeline_num, op_desc)
        self.app.logger.info(f"Pasted {len(actions_to_add)} point(s).", extra={'status_message': True})

    def _handle_copy_to_other_timeline(self):
        source_actions_ref = self._get_actions_list_ref()
        if not source_actions_ref or not self.multi_selected_action_indices:
            self.app.logger.warning(f"T{self.timeline_num}: No points selected to copy.", extra={'status_message': True})
            return

        destination_timeline_num = 2 if self.timeline_num == 1 else 1
        dest_funscript_instance, dest_axis_name = self.app.funscript_processor._get_target_funscript_object_and_axis(destination_timeline_num)

        if not dest_funscript_instance or not dest_axis_name:
            self.app.logger.error(f"T{self.timeline_num}: Could not find destination timeline to copy points.",
            extra = {'status_message': True})
            return

        actions_to_copy = [source_actions_ref[idx] for idx in sorted(list(self.multi_selected_action_indices))]
        if not actions_to_copy:
            return

        # Determine the time range of the actions being copied
        min_timestamp = min(action['at'] for action in actions_to_copy)
        max_timestamp = max(action['at'] for action in actions_to_copy)

        actions_to_add = []
        for action in actions_to_copy:
            new_action = {'timestamp_ms': action['at']}
            if dest_axis_name == 'primary':
                new_action['primary_pos'] = action['pos']
                new_action['secondary_pos'] = None
            else:  # secondary
                new_action['primary_pos'] = None
                new_action['secondary_pos'] = action['pos']
            actions_to_add.append(new_action)

        if not actions_to_add:
            return

        op_desc = f"Pasted {len(actions_to_add)} points from T{self.timeline_num} (overwrite)"
        # Record a single undo action for the entire operation on the DESTINATION timeline
        self.app.funscript_processor._record_timeline_action(destination_timeline_num, op_desc)

        # --- Delete existing points in the target range ---
        dest_actions_ref = getattr(dest_funscript_instance, f"{dest_axis_name}_actions", [])
        if dest_actions_ref:
            indices_to_delete = [
                i for i, action in enumerate(dest_actions_ref)
                if min_timestamp <= action['at'] <= max_timestamp
            ]
            if indices_to_delete:
                # This call is now part of the recorded undo action
                dest_funscript_instance.clear_points(axis=dest_axis_name, selected_indices=indices_to_delete)

        # Add the new points
        dest_funscript_instance.add_actions_batch(actions_to_add, is_from_live_tracker=False)

        # Finalize the single undoable action on the DESTINATION timeline
        self.app.funscript_processor._finalize_action_and_update_ui(destination_timeline_num, op_desc)
        self.app.logger.info(
            f"Pasted to T{destination_timeline_num}, overwriting points in range [{min_timestamp}, {max_timestamp}].",
            extra = {'status_message': True})

    # --- Bulk Operations Direct Calls to Funscript Object ---
    def _call_funscript_method(self, method_name: str, error_context: str, **kwargs) -> bool:
        funscript_instance, axis_name = self._get_target_funscript_details()
        if not funscript_instance or not axis_name:
            self.app.logger.warning(f"T{self.timeline_num}: Cannot {error_context}. Funscript object not found.",
            extra = {'status_message': True})
            return False
        try:
            method_to_call = getattr(funscript_instance, method_name)
            method_to_call(axis=axis_name, **kwargs)
            return True
        except Exception as e:
            self.app.logger.error(
                f"T{self.timeline_num} Error in {error_context} ({method_name}): {str(e)}", exc_info=True,
                extra = {'status_message': True})
            return False

    def _perform_sg_filter(self, window_length: int, polyorder: int, selected_indices: Optional[List[int]]):
        return (self._call_funscript_method('apply_savitzky_golay', 'SG filter', window_length=window_length, polyorder=polyorder, selected_indices=selected_indices))

    def _perform_speed_limiter(self, min_interval: int, vibe_amount: int, speed_threshold: float):
        return (self._call_funscript_method('apply_speed_limiter', 'Speed Limiter', min_interval = min_interval, vibe_amount = vibe_amount, speed_threshold = speed_threshold))

    def _perform_rdp_simplification(self, epsilon: float, selected_indices: Optional[List[int]]):
        return (self._call_funscript_method('simplify_rdp', 'RDP simplification', epsilon = epsilon, selected_indices = selected_indices))

    def _perform_peaks_simplification(self, height, threshold, distance, prominence, width, selected_indices: Optional[List[int]]):
        return (self._call_funscript_method('find_peaks_and_valleys', 'Peak finding',
                                           height=height,
                                           threshold=threshold,
                                           distance=distance,
                                           prominence=prominence,
                                           width=width,
                                           selected_indices=selected_indices))

    def _perform_inversion(self, selected_indices: Optional[List[int]]):
        return (self._call_funscript_method('invert_points_values', 'inversion', selected_indices = selected_indices))

    def _perform_clamp(self, clamp_value: int, selected_indices: Optional[List[int]]):
        return (self._call_funscript_method('clamp_points_values', f'clamp to {clamp_value}', clamp_value = clamp_value, selected_indices = selected_indices))

    def _perform_amplify(self, scale_factor: float, center_value: int, selected_indices: Optional[List[int]]):
        return (self._call_funscript_method('amplify_points_values', 'Amplify', scale_factor = scale_factor, center_value = center_value, selected_indices = selected_indices))

    def _perform_resample(self, selected_indices: Optional[List[int]]):
        # We can hardcode the resample rate for now, or add a popup later.
        # 50ms is a good starting point.
        resample_rate = 50
        return (self._call_funscript_method('apply_peak_preserving_resample', 'Resample', resample_rate_ms = resample_rate, selected_indices = selected_indices))

    def _perform_keyframe_simplification(self, position_tolerance: int, time_tolerance_ms: int, selected_indices: Optional[List[int]]):
        return (self._call_funscript_method('simplify_to_keyframes', 'Keyframe Extraction', position_tolerance = position_tolerance, time_tolerance_ms = time_tolerance_ms, selected_indices = selected_indices))

    def set_preview_actions(self, preview_actions: Optional[List[Dict]]):
        self.preview_actions = preview_actions
        self.is_previewing = preview_actions is not None

    def clear_preview(self):
        if self.is_previewing:
            self.is_previewing = False
            self.preview_actions = None

    def _update_preview(self, filter_type: str):
        """Generates and sets the preview data for the active filter."""
        funscript_instance, axis_name = self._get_target_funscript_details()
        if not funscript_instance:
            self.clear_preview()
            return

        apply_to_selection = False
        if filter_type == 'sg':
            apply_to_selection = self.sg_apply_to_selection
        elif filter_type == 'rdp':
            apply_to_selection = self.rdp_apply_to_selection
        elif filter_type == 'amp':
            apply_to_selection = self.amp_apply_to_selection
        elif filter_type == 'keyframe':
            apply_to_selection = self.keyframe_apply_to_selection
        elif filter_type == 'peaks':
            apply_to_selection = self.peaks_apply_to_selection

        indices_to_process = list(self.multi_selected_action_indices) if apply_to_selection else None

        params = {}
        if filter_type == 'sg':
            params = {'window_length': self.sg_window_length, 'polyorder': self.sg_poly_order}
        elif filter_type == 'rdp':
            params = {'epsilon': self.rdp_epsilon}
        elif filter_type == 'amp':
            params = {'scale_factor': self.amp_scale_factor, 'center_value': self.amp_center_value}
        elif filter_type == 'keyframe':
            params = {
                'position_tolerance': self.keyframe_position_tolerance,
                'time_tolerance_ms': self.keyframe_time_tolerance
            }
        elif filter_type == 'peaks':
            params = {
                'height': self.peaks_height,
                'threshold': self.peaks_threshold,
                'distance': self.peaks_distance,
                'prominence': self.peaks_prominence,
                'width': self.peaks_width
            }
        elif filter_type == 'speed_limiter':
            params = {
                'min_interval': self.min_interval,
                'vibe_amount': self.vibe_amount,
                'speed_threshold': self.speed_threshold
            }

        generated_preview_actions = funscript_instance.calculate_filter_preview(
            axis = axis_name,
            filter_type = filter_type,
            filter_params = params,
            selected_indices = indices_to_process)
        self.set_preview_actions(generated_preview_actions)

    def render(self, timeline_y_start_coord: float = 0, timeline_render_height: float = 0):
        app_state = self.app.app_state_ui
        is_floating = app_state.ui_layout_mode == 'floating'

        visibility_flag_name = f"show_funscript_interactive_timeline{'' if self.timeline_num == 1 else '2'}"
        is_visible = getattr(app_state, visibility_flag_name, False)

        if not is_visible:
            return

        # --- Window Creation (Begin) ---
        should_render_content = True
        if is_floating:
            window_title = f"Interactive Timeline {self.timeline_num}"
            imgui.set_next_window_size(app_state.window_width, 180, condition=imgui.APPEARING)
            is_open, new_visibility = imgui.begin(window_title, closable=True, flags=imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_SCROLL_WITH_MOUSE)
            if new_visibility != is_visible:
                setattr(app_state, visibility_flag_name, new_visibility)
                self.app.project_manager.project_dirty = True
            if not is_open:
                should_render_content = False
        else:  # Fixed mode
            if timeline_y_start_coord < 0 or timeline_render_height <= 0:
                return  # Cannot render fixed without coordinates
            imgui.set_next_window_position(0, timeline_y_start_coord)
            imgui.set_next_window_size(app_state.window_width, timeline_render_height)
            imgui.begin(f"Funscript Editor Timeline##Interactive{self.timeline_num}",
                flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_SCROLL_WITH_MOUSE))

        # --- Content Rendering ---
        if should_render_content:
            mouse_pos = imgui.get_mouse_pos()
            io = imgui.get_io()

            fs_proc = self.app.funscript_processor
            target_funscript_instance_for_render, axis_name_for_render = self._get_target_funscript_details()
            actions_list = []
            if target_funscript_instance_for_render and axis_name_for_render:
                actions_list = getattr(target_funscript_instance_for_render, f"{axis_name_for_render}_actions", [])

            shortcuts = self.app.app_settings.get("funscript_editor_shortcuts", {})

            window_id_suffix = f"Timeline{self.timeline_num}"
            allow_editing_timeline = True
            has_actions = bool(actions_list)
            context_popup_id = f"TimelineActionPopup##{window_id_suffix}"

            script_info_text = f"Timeline {self.timeline_num} - "
            if self.timeline_num == 1:
                loaded_path = self.app.file_manager.loaded_funscript_path
                if loaded_path and os.path.exists(loaded_path):
                    script_info_text += f"{os.path.basename(loaded_path)}"
                elif has_actions:
                    script_info_text += "(Edited/Generated)"
                else:
                    script_info_text += "(Empty - Drag .funscript here or Load/Generate)"
            elif self.timeline_num == 2:
                stats_t2 = fs_proc.funscript_stats_t2
                if stats_t2["path"] != "N/A" and stats_t2["source_type"] == "File":
                    script_info_text += f"{os.path.basename(stats_t2['path'])}"
                elif has_actions:
                    script_info_text += "(Secondary Axis - Edited/Generated)"
                else:
                    script_info_text += "(Secondary Axis - Empty)"

            # --- Buttons ---
            # Clear Button
            num_selected_for_clear = len(self.multi_selected_action_indices)
            clear_button_text = f"Clear Sel. ({num_selected_for_clear})" if num_selected_for_clear > 0 and has_actions else "Clear All"
            clear_op_disabled = not (
                    allow_editing_timeline and has_actions and target_funscript_instance_for_render and axis_name_for_render)
            if clear_op_disabled:
                imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
                imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)
            if imgui.button(f"{clear_button_text}##ClearButton{window_id_suffix}"):
                if not clear_op_disabled:
                    indices_to_clear = list(self.multi_selected_action_indices) if num_selected_for_clear > 0 else None
                    op_desc = f"Cleared {len(indices_to_clear) if indices_to_clear else 'All'} Point(s)"
                    fs_proc._record_timeline_action(self.timeline_num, op_desc)
                    target_funscript_instance_for_render.clear_points(axis=axis_name_for_render, selected_indices = indices_to_clear)
                    if indices_to_clear: self.multi_selected_action_indices.clear()
                    self.selected_action_idx = -1
                    self.dragging_action_idx = -1
                    fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)
                    self.app.logger.info(f"{op_desc} on T{self.timeline_num}.", extra={'status_message': True})
            if clear_op_disabled: imgui.pop_style_var(); imgui.internal.pop_item_flag()
            imgui.same_line()

            if imgui.button(f"Unload##Unload{window_id_suffix}"):
                if allow_editing_timeline and has_actions and target_funscript_instance_for_render:
                    op_desc = "Funscript Unloaded via Timeline"
                    fs_proc._record_timeline_action(self.timeline_num, op_desc)
                    target_funscript_instance_for_render.clear_points(axis=axis_name_for_render)
                    self.selected_action_idx = -1
                    self.multi_selected_action_indices.clear()
                    self.dragging_action_idx = -1
                    fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)
                    self.app.logger.info(f"T{self.timeline_num} Unloaded.", extra={'status_message': True})
            imgui.same_line()

            # --- SG Filter Button ---
            sg_disabled_bool = not allow_editing_timeline or not has_actions
            if sg_disabled_bool:
                imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
                imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)
            if imgui.button(f"Smooth (SG)##SGFilter{window_id_suffix}"):
                if not sg_disabled_bool:
                    self.show_sg_settings_popup = True
                    self.sg_apply_to_selection = bool(
                        self.multi_selected_action_indices and len(self.multi_selected_action_indices) >= 2)
            if sg_disabled_bool: imgui.pop_style_var(); imgui.internal.pop_item_flag()
            imgui.same_line()

            # --- RDP Button ---
            rdp_disabled_bool = not allow_editing_timeline or not has_actions
            if rdp_disabled_bool:
                imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
                imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)
            if imgui.button(f"Simplify (RDP)##{window_id_suffix}"):
                if not rdp_disabled_bool:
                    self.show_rdp_settings_popup = True
                    self.rdp_apply_to_selection = bool(
                        self.multi_selected_action_indices and len(self.multi_selected_action_indices) >= 2)
            if rdp_disabled_bool:
                imgui.pop_style_var()
                imgui.internal.pop_item_flag()
            imgui.same_line()

            # --- Peaks Button ---
            peaks_disabled_bool = not allow_editing_timeline or not has_actions
            if peaks_disabled_bool:
                imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
                imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)
            if imgui.button(f"Peaks##{window_id_suffix}"):
                if not peaks_disabled_bool:
                    self.show_peaks_settings_popup = True
                    self.peaks_apply_to_selection = bool(
                        self.multi_selected_action_indices and len(self.multi_selected_action_indices) >= 3)
            if peaks_disabled_bool:
                imgui.pop_style_var()
                imgui.internal.pop_item_flag()
            imgui.same_line()

            # --- Amplify Button ---
            amp_disabled_bool = not allow_editing_timeline or not has_actions
            if amp_disabled_bool:
                imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
                imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)

            if imgui.button(f"Amplify##AmpFilter{window_id_suffix}"):
                if not amp_disabled_bool:
                    self.show_amp_settings_popup = True
                    self.amp_apply_to_selection = bool(
                        self.multi_selected_action_indices and len(self.multi_selected_action_indices) >= 2)

            if amp_disabled_bool: imgui.pop_style_var(); imgui.internal.pop_item_flag()
            imgui.same_line()

            # --- Resample Button ---
            resample_disabled_bool = not allow_editing_timeline or not has_actions
            if resample_disabled_bool:
                imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
                imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)

            if imgui.button(f"Resample##ResampleFilter{window_id_suffix}"):
                if not resample_disabled_bool:
                    indices_to_use = list(self.multi_selected_action_indices) if (
                            self.multi_selected_action_indices and len(self.multi_selected_action_indices) >= 3
                    ) else None

                    op_desc = "Applied Peak-Preserving Resample" + (" to selection" if indices_to_use else "")

                    fs_proc._record_timeline_action(self.timeline_num, op_desc)  # For Undo
                    if self._perform_resample(selected_indices=indices_to_use):
                        fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)
                        self.app.logger.info(f"{op_desc} on T{self.timeline_num}.", extra={'status_message': True})

            if resample_disabled_bool: imgui.pop_style_var(); imgui.internal.pop_item_flag()
            imgui.same_line()

            # --- Keyframe Extraction Button ---
            keyframe_disabled_bool = not allow_editing_timeline or not has_actions
            if keyframe_disabled_bool:
                imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
                imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)

            if imgui.button(f"Keyframes##KeyframeFilter{window_id_suffix}"):
                # This now opens the popup instead of applying directly
                if not keyframe_disabled_bool:
                    self.show_keyframe_settings_popup = True
                    self.keyframe_apply_to_selection = bool(
                        self.multi_selected_action_indices and len(self.multi_selected_action_indices) >= 3)

            if keyframe_disabled_bool: imgui.pop_style_var(); imgui.internal.pop_item_flag()
            imgui.same_line()

            # --- Speed Limiter Button ---
            speed_disabled_bool = not allow_editing_timeline or not has_actions
            if speed_disabled_bool:
                imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
                imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)
            if imgui.button(f"Speed Limiter##HandyBTLimiter{window_id_suffix}"):
                if not speed_disabled_bool:
                    self.show_speed_limiter_popup = True
            if speed_disabled_bool:
                imgui.pop_style_var()
                imgui.internal.pop_item_flag()
            imgui.same_line()

            # Invert Button
            invert_button_prefix = "Invert Sel." if self.multi_selected_action_indices else "Invert All"
            invert_disabled_bool = not allow_editing_timeline or not has_actions
            if invert_disabled_bool:
                imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
                imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)
            if imgui.button(f"{invert_button_prefix}##Invert{window_id_suffix}"):
                if not invert_disabled_bool:
                    indices_to_op = list(
                        self.multi_selected_action_indices) if self.multi_selected_action_indices else None
                    op_desc = "Inverted Selected Points" if indices_to_op else "Inverted Funscript Actions"
                    fs_proc._record_timeline_action(self.timeline_num, op_desc)
                    if self._perform_inversion(selected_indices=indices_to_op):
                        fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)
                        self.app.logger.info(f"{op_desc} on T{self.timeline_num}.", extra={'status_message': True})
            if invert_disabled_bool: imgui.pop_style_var(); imgui.internal.pop_item_flag()
            imgui.same_line()

            # Clamp Buttons
            clamp_button_prefix = "Clamp Sel. to" if self.multi_selected_action_indices else "Clamp All to"
            clamp_disabled_bool = not allow_editing_timeline or not has_actions
            if clamp_disabled_bool:
                imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
                imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)
            if imgui.button(f"{clamp_button_prefix} 0##Clamp0{window_id_suffix}"):
                if not clamp_disabled_bool:
                    indices_to_op = list(
                        self.multi_selected_action_indices) if self.multi_selected_action_indices else None
                    op_desc = "Clamped Selected to 0" if indices_to_op else "Clamped All to 0"
                    fs_proc._record_timeline_action(self.timeline_num, op_desc)
                    if self._perform_clamp(0, selected_indices=indices_to_op):
                        fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)
                        self.app.logger.info(f"{op_desc} on T{self.timeline_num}.", extra={'status_message': True})
            imgui.same_line()
            if imgui.button(f"{clamp_button_prefix} 100##Clamp100{window_id_suffix}"):
                if not clamp_disabled_bool:
                    indices_to_op = list(self.multi_selected_action_indices) if self.multi_selected_action_indices else None
                    op_desc = "Clamped Selected to 100" if indices_to_op else "Clamped All to 100"
                    fs_proc._record_timeline_action(self.timeline_num, op_desc)
                    if self._perform_clamp(100, selected_indices=indices_to_op):
                        fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)
                        self.app.logger.info(f"{op_desc} on T{self.timeline_num}.", extra={'status_message': True})
            if clamp_disabled_bool: imgui.pop_style_var(); imgui.internal.pop_item_flag()
            imgui.same_line()

            # --- Time Shift controls ---
            time_shift_disabled_bool = not allow_editing_timeline or not has_actions or not (
                    self.app.processor and self.app.processor.fps and self.app.processor.fps > 0)
            if time_shift_disabled_bool:
                imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
                imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)

            if imgui.button(f"<<##ShiftLeft{window_id_suffix}"):
                if not time_shift_disabled_bool and self.shift_frames_amount > 0: self._perform_time_shift(-self.shift_frames_amount)
            imgui.same_line()

            imgui.push_item_width(80 * self.app.app_settings.get("global_font_scale", 1.0))
            _, self.shift_frames_amount = imgui.input_int(f"Frames##ShiftAmount{window_id_suffix}", self.shift_frames_amount, 1, 10)
            if self.shift_frames_amount < 0: self.shift_frames_amount = 0
            imgui.pop_item_width()
            imgui.same_line()
            if imgui.button(f">>##ShiftRight{window_id_suffix}"):
                if not time_shift_disabled_bool and self.shift_frames_amount > 0: self._perform_time_shift(self.shift_frames_amount)
            if time_shift_disabled_bool: imgui.pop_style_var(); imgui.internal.pop_item_flag()
            imgui.same_line()

            # --- Zoom Buttons ---
            video_loaded = self.app.processor and self.app.processor.video_info and self.app.processor.total_frames > 0
            # can_manual_pan_zoom = (video_loaded and not self.app.processor.is_processing) or not video_loaded
            zoom_disabled_bool = False  # not allow_editing_timeline or not can_manual_pan_zoom

            if zoom_disabled_bool:
                imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
                imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)

            imgui.text("Zoom:")
            imgui.same_line()
            if imgui.button(f"+##ZoomIn{window_id_suffix}"):
                self.zoom_action_request = 0.85  # Set zoom-in request
            imgui.same_line()
            if imgui.button(f"-##ZoomOut{window_id_suffix}"):
                self.zoom_action_request = 1.15  # Set zoom-out request

            if zoom_disabled_bool:
                imgui.pop_style_var()
                imgui.internal.pop_item_flag()
            imgui.same_line()
            # endregion

            # --- Popup Windows ---
            # region Popups
            # --- Speed Limiter Settings Window ---
            speed_window_title = f"Speed Limiter Settings (Timeline {self.timeline_num})##SpeedLimiterSettingsWindow{window_id_suffix}"
            if self.show_speed_limiter_popup:
                # On first open, generate an initial preview
                if not self.is_previewing:
                    self._update_preview('speed_limiter')

                main_viewport = imgui.get_main_viewport()
                popup_pos_x = main_viewport.pos[0] + (main_viewport.size[0] - 400) * 0.5
                popup_pos_y = main_viewport.pos[1] + (main_viewport.size[1] - 220) * 0.5
                imgui.set_next_window_position(popup_pos_x, popup_pos_y, condition=imgui.APPEARING)
                imgui.set_next_window_size(400, 0, condition=imgui.APPEARING)
                window_expanded, self.show_speed_limiter_popup = imgui.begin(
                    speed_window_title, closable=True,
                    flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)

                if window_expanded:
                    imgui.text(f"Speed Limiter (Timeline {self.timeline_num})")
                    imgui.separator()

                    interval_changed, self.min_interval = imgui.slider_int("Min Interval (ms)##HandyMinIntervalPopup", self.min_interval, 0, 200)
                    if interval_changed: self._update_preview('speed_limiter')
                    if imgui.is_item_hovered(): imgui.set_tooltip("Action points closer than this (in ms) are removed.")

                    #vibe_changed, self.vibe_amount = imgui.slider_int("Vibration Amplitude##HandyVibeAmountPopup",
                    #                                                        self.vibe_amount, 0, 100)
                    #if vibe_changed: self._update_preview('speed_limiter')
                    #if imgui.is_item_hovered(): imgui.set_tooltip("Vibration starting amplitude (1-100).")

                    speed_changed, self.speed_threshold = imgui.slider_float("Speed Threshold##HandySpeedThresholdPopup", self.speed_threshold, 100.0, 1000.0, "%.1f")
                    if speed_changed: self._update_preview('speed_limiter')
                    if imgui.is_item_hovered(): imgui.set_tooltip(
                        "Speed limit for the script. 500 is the default 'Handy' limit.")

                    imgui.separator()

                    if imgui.button(f"Apply##HandyBTApplyPop{window_id_suffix}"):
                        op_desc = f"Applied Speed Limiter (Interval:{self.min_interval}, Vibe:{self.vibe_amount}, Speed:{self.speed_threshold:.1f})"
                        fs_proc._record_timeline_action(self.timeline_num, op_desc)

                        if self._perform_speed_limiter(self.min_interval, self.vibe_amount, self.speed_threshold):
                            fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)
                            self.app.app_settings.set(f"timeline{self.timeline_num}_min_interval", self.min_interval)
                            self.app.app_settings.set(f"timeline{self.timeline_num}_vibe_amount", self.vibe_amount)
                            self.app.app_settings.set(f"timeline{self.timeline_num}_speed_threshold", self.speed_threshold)
                            self.app.logger.info(f"{op_desc} on T{self.timeline_num}.", extra={'status_message': True})

                        self.clear_preview()
                        self.show_speed_limiter_popup = False

                    imgui.same_line()
                    if imgui.button(f"Cancel##HandyBTCancelPop{window_id_suffix}"):
                        self.clear_preview()
                        self.show_speed_limiter_popup = False

                if not self.show_speed_limiter_popup:
                    self.clear_preview()

                if window_expanded:
                    imgui.end()

            # --- SG Settings Window ---
            sg_window_title = f"Savitzky-Golay Filter Settings (Timeline {self.timeline_num})##SGSettingsWindow{window_id_suffix}"
            if self.show_sg_settings_popup:
                # On first open, generate an initial preview
                if not self.is_previewing:
                    self._update_preview('sg')

                main_viewport = imgui.get_main_viewport()
                popup_pos_x = main_viewport.pos[0] + (main_viewport.size[0] - 350) * 0.5
                popup_pos_y = main_viewport.pos[1] + (main_viewport.size[1] - 200) * 0.5
                imgui.set_next_window_position(popup_pos_x, popup_pos_y, condition=imgui.APPEARING)
                imgui.set_next_window_size(350, 0, condition=imgui.APPEARING)
                window_expanded, self.show_sg_settings_popup = imgui.begin(
                    sg_window_title, closable=True,
                    flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)  # Use True for closable to handle 'X'

                if window_expanded:
                    imgui.text(f"Savitzky-Golay Filter (Timeline {self.timeline_num})")
                    imgui.separator()

                    # --- Window Length Slider ---
                    wl_changed, current_wl = imgui.slider_int("Window Length##SGWinPopup", self.sg_window_length, 3, 99)
                    if wl_changed:
                        self.sg_window_length = current_wl if current_wl % 2 != 0 else current_wl + 1
                        if self.sg_window_length < 3: self.sg_window_length = 3
                        self._update_preview('sg')  # CORRECT: Only update preview

                    # --- Polyorder Slider ---
                    max_po = max(1, self.sg_window_length - 1)
                    po_val = min(self.sg_poly_order, max_po)
                    if po_val < 1: po_val = 1
                    po_changed, current_po = imgui.slider_int("Polyorder##SGPolyPopup", po_val, 1, max_po)
                    if po_changed:
                        self.sg_poly_order = current_po
                        self._update_preview('sg')  # CORRECT: Only update preview

                    # --- Apply to Selection Checkbox ---
                    if self.multi_selected_action_indices and len(self.multi_selected_action_indices) >= 2:
                        sel_changed, self.sg_apply_to_selection = imgui.checkbox(
                            f"Apply to {len(self.multi_selected_action_indices)} selected##SGApplyToSel",
                            self.sg_apply_to_selection)
                        if sel_changed:
                            self._update_preview('sg')  # CORRECT: Only update preview
                    else:
                        imgui.text_disabled("Apply to: Full Script")
                        self.sg_apply_to_selection = False

                    # --- Buttons ---
                    if imgui.button(f"Apply##SGApplyPop{window_id_suffix}"):
                        indices_to_use = list(
                            self.multi_selected_action_indices) if self.sg_apply_to_selection else None
                        op_desc = f"Applied SG (W:{self.sg_window_length}, P:{self.sg_poly_order})" + (" to selection" if indices_to_use else "")

                        fs_proc._record_timeline_action(self.timeline_num, op_desc)  # Record for Undo

                        # This calls the original, DESTRUCTIVE method
                        if self._perform_sg_filter(self.sg_window_length, self.sg_poly_order, selected_indices=indices_to_use):
                            fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)
                            self.app.app_settings.set(f"timeline{self.timeline_num}_sg_default_window", self.sg_window_length)
                            self.app.app_settings.set(f"timeline{self.timeline_num}_sg_default_polyorder", self.sg_poly_order)
                            self.app.logger.info(f"{op_desc} on T{self.timeline_num}.", extra={'status_message': True})

                        self.clear_preview()
                        self.show_sg_settings_popup = False

                    imgui.same_line()
                    if imgui.button(f"Cancel##SGCancelPop{window_id_suffix}"):
                        self.clear_preview()
                        self.show_sg_settings_popup = False

                # This handles closing the popup with the 'X' button
                if not self.show_sg_settings_popup:
                    self.clear_preview()

                if window_expanded:
                    imgui.end()

            # --- RDP Settings Window ---
            rdp_window_title = f"RDP Simplification Settings (Timeline {self.timeline_num})##RDPSettingsWindow{window_id_suffix}"
            if self.show_rdp_settings_popup:
                # --- RDP Settings Window ---
                if self.show_rdp_settings_popup:
                    if not self.is_previewing:
                        self._update_preview('rdp')

                main_viewport = imgui.get_main_viewport()
                popup_pos_x = main_viewport.pos[0] + (main_viewport.size[0] - 350) * 0.5
                popup_pos_y = main_viewport.pos[1] + (main_viewport.size[1] - 180) * 0.5
                imgui.set_next_window_position(popup_pos_x, popup_pos_y, condition=imgui.APPEARING)
                imgui.set_next_window_size(350, 0, condition=imgui.APPEARING)
                window_expanded, self.show_rdp_settings_popup = imgui.begin(rdp_window_title, closable=self.show_rdp_settings_popup, flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)

                if window_expanded:
                    imgui.text(f"RDP Simplification (Timeline {self.timeline_num})");
                    imgui.separator()
                    epsilon_changed, self.rdp_epsilon = imgui.slider_float("Epsilon##RDPEpsPopup", self.rdp_epsilon, 0.1, 20.0, "%.1f")
                    if epsilon_changed:
                        self._update_preview('rdp')

                    if self.multi_selected_action_indices and len(self.multi_selected_action_indices) >= 2:
                        sel_changed, self.rdp_apply_to_selection = imgui.checkbox(
                            f"Apply to {len(self.multi_selected_action_indices)} selected##RDPApplyToSel",
                            self.rdp_apply_to_selection)
                        if sel_changed:
                            self._update_preview('rdp')
                    else:
                        imgui.text_disabled("Apply to: Full Script")
                        self.rdp_apply_to_selection = False
                    if imgui.button(f"Apply##RDPApplyPop{window_id_suffix}"):
                        indices_to_use = list(
                            self.multi_selected_action_indices) if self.rdp_apply_to_selection else None
                        op_desc = f"Applied RDP (Epsilon:{self.rdp_epsilon:.1f})" + (
                            " to selection" if indices_to_use else "")
                        fs_proc._record_timeline_action(self.timeline_num, op_desc)
                        if self._perform_rdp_simplification(self.rdp_epsilon, selected_indices=indices_to_use):
                            fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)
                            self.app.app_settings.set(f"timeline{self.timeline_num}_rdp_default_epsilon", self.rdp_epsilon)
                            self.app.logger.info(f"{op_desc} on T{self.timeline_num}.", extra={'status_message': True})
                        self.clear_preview()
                        self.show_rdp_settings_popup = False
                    imgui.same_line()
                    if imgui.button(f"Cancel##RDPCancelPop{window_id_suffix}"):
                        self.clear_preview()
                        self.show_rdp_settings_popup = False
                imgui.end()

            # --- Peaks Settings Window ---
            peaks_window_title = f"Peak Extraction Settings (Timeline {self.timeline_num})##PeaksSettingsWindow{window_id_suffix}"
            if self.show_peaks_settings_popup:
                if not self.is_previewing:
                    self._update_preview('peaks')

                main_viewport = imgui.get_main_viewport()
                popup_pos_x = main_viewport.pos[0] + (main_viewport.size[0] - 450) * 0.5
                popup_pos_y = main_viewport.pos[1] + (main_viewport.size[1] - 300) * 0.5
                imgui.set_next_window_position(popup_pos_x, popup_pos_y, condition=imgui.APPEARING)
                imgui.set_next_window_size(450, 0, condition=imgui.APPEARING)

                window_expanded, self.show_peaks_settings_popup = imgui.begin(peaks_window_title, closable=True, flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)

                if window_expanded:
                    imgui.text("Peak & Valley Extraction (scipy.find_peaks)")
                    imgui.text_wrapped("Filters to only include peaks and valleys. Parameters with 0 are disabled.")
                    imgui.separator()

                    h_changed, self.peaks_height = imgui.slider_int("Height##PeaksHeight", self.peaks_height, 0, 100)
                    if h_changed:
                        self._update_preview('peaks')
                    if imgui.is_item_hovered(): imgui.set_tooltip("Required height of peaks.")

                    t_changed, self.peaks_threshold = imgui.slider_int("Threshold##PeaksThreshold", self.peaks_threshold, 0, 2)
                    if t_changed:
                        self._update_preview('peaks')
                    if imgui.is_item_hovered(): imgui.set_tooltip("Vertical distance between a peak and its neighbors.")

                    d_changed, self.peaks_distance = imgui.slider_int("Distance##PeaksDistance", self.peaks_distance, 1, 50)
                    if d_changed:
                        self._update_preview('peaks')
                    if imgui.is_item_hovered(): imgui.set_tooltip(
                        "Minimum horizontal distance (in # of points) between peaks.")

                    p_changed, self.peaks_prominence = imgui.slider_int("Prominence##PeaksProminence", self.peaks_prominence, 0, 50)
                    if p_changed:
                        self._update_preview('peaks')
                    if imgui.is_item_hovered(): imgui.set_tooltip("The vertical distance between the peak and its lowest contour line.")

                    w_changed, self.peaks_width = imgui.slider_int("Width##PeaksWidth", self.peaks_width,0, 5)
                    if w_changed:
                        self._update_preview('peaks')
                    if imgui.is_item_hovered(): imgui.set_tooltip("Required width of peaks in samples.")

                    imgui.separator()

                    if self.multi_selected_action_indices and len(self.multi_selected_action_indices) >= 3:
                        sel_changed, self.peaks_apply_to_selection = imgui.checkbox(
                            f"Apply to {len(self.multi_selected_action_indices)} selected##PeaksApplyToSel", self.peaks_apply_to_selection)
                        if sel_changed: self._update_preview('peaks')
                    else:
                        imgui.text_disabled("Apply to: Full Script")
                        self.peaks_apply_to_selection = False

                    if imgui.button(f"Apply##PeaksApplyPop{window_id_suffix}"):
                        indices_to_use = list(
                            self.multi_selected_action_indices) if self.peaks_apply_to_selection else None
                        op_desc = f"Applied Peaks Filter" + (" to selection" if indices_to_use else "")

                        fs_proc._record_timeline_action(self.timeline_num, op_desc)
                        if self._perform_peaks_simplification(
                                height=self.peaks_height,
                                threshold=self.peaks_threshold,
                                distance=self.peaks_distance,
                                prominence=self.peaks_prominence,
                                width=self.peaks_width,
                                selected_indices=indices_to_use
                        ):
                            fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)
                            # Save settings
                            self.app.app_settings.set(f"timeline{self.timeline_num}_peaks_default_height",
                                                      self.peaks_height)
                            self.app.app_settings.set(f"timeline{self.timeline_num}_peaks_default_threshold",
                                                      self.peaks_threshold)
                            self.app.app_settings.set(f"timeline{self.timeline_num}_peaks_default_distance",
                                                      self.peaks_distance)
                            self.app.app_settings.set(f"timeline{self.timeline_num}_peaks_default_prominence",
                                                      self.peaks_prominence)
                            self.app.app_settings.set(f"timeline{self.timeline_num}_peaks_default_width",
                                                      self.peaks_width)
                            self.app.logger.info(f"{op_desc} on T{self.timeline_num}.", extra={'status_message': True})

                        self.clear_preview()
                        self.show_peaks_settings_popup = False

                    imgui.same_line()
                    if imgui.button(f"Cancel##PeaksCancelPop{window_id_suffix}"):
                        self.clear_preview()
                        self.show_peaks_settings_popup = False

                if not self.show_peaks_settings_popup:
                    self.clear_preview()

                if window_expanded:
                    imgui.end()

            # --- Amplify Settings Window ---
            amp_window_title = f"Amplify Settings (Timeline {self.timeline_num})##AmpSettingsWindow{window_id_suffix}"
            if self.show_amp_settings_popup:
                if not self.is_previewing:
                    self._update_preview('amp')

                main_viewport = imgui.get_main_viewport()
                popup_pos_x = main_viewport.pos[0] + (main_viewport.size[0] - 350) * 0.5
                popup_pos_y = main_viewport.pos[1] + (main_viewport.size[1] - 200) * 0.5
                imgui.set_next_window_position(popup_pos_x, popup_pos_y, condition=imgui.APPEARING)
                imgui.set_next_window_size(350, 0, condition=imgui.APPEARING)

                window_expanded, self.show_amp_settings_popup = imgui.begin(amp_window_title, closable=True, flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)

                if window_expanded:
                    imgui.text(f"Amplify Values (Timeline {self.timeline_num})")
                    imgui.separator()

                    sf_changed, self.amp_scale_factor = imgui.slider_float("Scale Factor##AmpScalePopup", self.amp_scale_factor, 0.1, 5.0, "%.2f")
                    if sf_changed:
                        self._update_preview('amp')

                    cv_changed, self.amp_center_value = imgui.slider_int("Center Value##AmpCenterPopup", self.amp_center_value, 0, 100)
                    if cv_changed:
                        self._update_preview('amp')

                    if self.multi_selected_action_indices and len(self.multi_selected_action_indices) >= 2:
                        sel_changed, self.amp_apply_to_selection = imgui.checkbox(f"Apply to {len(self.multi_selected_action_indices)} selected##AmpApplyToSel", self.amp_apply_to_selection)
                        if sel_changed:
                            self._update_preview('amp')
                    else:
                        imgui.text_disabled("Apply to: Full Script")
                        self.amp_apply_to_selection = False

                    if imgui.button(f"Apply##AmpApplyPop{window_id_suffix}"):
                        indices_to_use = list(
                            self.multi_selected_action_indices) if self.amp_apply_to_selection else None
                        op_desc = (f"Amplified (S:{self.amp_scale_factor:.2f}, C:{self.amp_center_value})" + (" to selection" if indices_to_use else ""))

                        fs_proc._record_timeline_action(self.timeline_num, op_desc)
                        if self._perform_amplify(self.amp_scale_factor, self.amp_center_value, selected_indices=indices_to_use):
                            fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)
                            self.app.app_settings.set(f"timeline{self.timeline_num}_amp_default_scale", self.amp_scale_factor)
                            self.app.app_settings.set(f"timeline{self.timeline_num}_amp_default_center", self.amp_center_value)
                            self.app.logger.info(f"{op_desc} on T{self.timeline_num}.", extra={'status_message': True})

                        self.clear_preview()
                        self.show_amp_settings_popup = False

                    imgui.same_line()
                    if imgui.button(f"Cancel##AmpCancelPop{window_id_suffix}"):
                        self.clear_preview()
                        self.show_amp_settings_popup = False

                if not self.show_amp_settings_popup:
                    self.clear_preview()

                if window_expanded:
                    imgui.end()

            # --- Keyframe Extraction Settings Window ---
            keyframe_window_title = f"Keyframe Extraction Settings (Timeline {self.timeline_num})##KeyframeSettingsWindow{window_id_suffix}"
            if self.show_keyframe_settings_popup:
                if not self.is_previewing:
                    self._update_preview('keyframe')

                main_viewport = imgui.get_main_viewport()
                popup_pos_x = main_viewport.pos[0] + (main_viewport.size[0] - 350) * 0.5
                popup_pos_y = main_viewport.pos[1] + (main_viewport.size[1] - 220) * 0.5 
                imgui.set_next_window_position(popup_pos_x, popup_pos_y, condition=imgui.APPEARING)
                imgui.set_next_window_size(350, 0, condition=imgui.APPEARING)

                window_expanded, self.show_keyframe_settings_popup = imgui.begin(keyframe_window_title, closable=True, flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)

                if window_expanded:
                    imgui.text(f"Dominant Keyframe Extraction")
                    imgui.separator()

                    imgui.text_wrapped("Filters out minor peaks/valleys based on position and time.")

                    # Slider 1: Position Tolerance
                    pos_tol_changed, self.keyframe_position_tolerance = imgui.slider_int("Position Tolerance##KeyframePosTol", self.keyframe_position_tolerance, 0, 15)
                    if pos_tol_changed: self._update_preview('keyframe')
                    if imgui.is_item_hovered(): imgui.set_tooltip("Minimum change in position to create a new keyframe.")

                    # Slider 2: Time Tolerance
                    time_tol_changed, self.keyframe_time_tolerance = imgui.slider_int("Time Tolerance (ms)##KeyframeTimeTol", self.keyframe_time_tolerance, 0, 100)
                    if time_tol_changed: self._update_preview('keyframe')
                    if imgui.is_item_hovered(): imgui.set_tooltip("Minimum time between two keyframes.")

                    # ... (The rest of the popup: selection checkbox and Apply/Cancel buttons) ...
                    if self.multi_selected_action_indices and len(self.multi_selected_action_indices) >= 3:
                        sel_changed, self.keyframe_apply_to_selection = imgui.checkbox(f"Apply to {len(self.multi_selected_action_indices)} selected##KeyframeApplyToSel", self.keyframe_apply_to_selection)
                        if sel_changed:
                            self._update_preview('keyframe')
                    else:
                        imgui.text_disabled("Apply to: Full Script")
                        self.keyframe_apply_to_selection = False

                    if imgui.button(f"Apply##KeyframeApplyPop{window_id_suffix}"):
                        indices_to_use = list(self.multi_selected_action_indices) if self.keyframe_apply_to_selection else None
                        op_desc = (f"Simplified to Keyframes (P-Tol:{self.keyframe_position_tolerance}, T-Tol:{self.keyframe_time_tolerance}")

                        fs_proc._record_timeline_action(self.timeline_num, op_desc)
                        if self._perform_keyframe_simplification(self.keyframe_position_tolerance, self.keyframe_time_tolerance, selected_indices=indices_to_use):
                            fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)
                            self.app.app_settings.set(f"timeline{self.timeline_num}_keyframe_default_pos_tol", self.keyframe_position_tolerance)
                            self.app.app_settings.set(f"timeline{self.timeline_num}_keyframe_default_time_tol", self.keyframe_time_tolerance)
                            self.app.logger.info(f"{op_desc} on T{self.timeline_num}.", extra={'status_message': True})

                        self.clear_preview()
                        self.show_keyframe_settings_popup = False

                    imgui.same_line()
                    if imgui.button(f"Cancel##KeyframeCancelPop{window_id_suffix}"):
                        self.clear_preview()
                        self.show_keyframe_settings_popup = False

                if not self.show_keyframe_settings_popup:
                    self.clear_preview()

                if window_expanded:
                    imgui.end()

            if not self.show_sg_settings_popup and not self.show_rdp_settings_popup and not self.show_peaks_settings_popup and not self.show_amp_settings_popup and not self.show_keyframe_settings_popup and not self.show_speed_limiter_popup:
                self.clear_preview()

            imgui.text_colored(script_info_text, 0.75, 0.75, 0.75, 0.95)
            # --- (Drag and drop target for T2 remains the same) ---
            if self.timeline_num == 2:
                if imgui.begin_drag_drop_target():
                    payload = imgui.accept_drag_drop_payload("FILES")
                    if payload is not None and self.app.file_manager.last_dropped_files:
                        if self.app.file_manager.last_dropped_files[0].lower().endswith(".funscript"):
                            self.app.file_manager.load_funscript_to_timeline(
                                self.app.file_manager.last_dropped_files[0], timeline_num=2)
                        self.app.file_manager.last_dropped_files = None
                    imgui.end_drag_drop_target()

            # --- (Canvas setup, grid, points, lines drawing remains the same) ---
            draw_list = imgui.get_window_draw_list()
            canvas_abs_pos = imgui.get_cursor_screen_pos()
            canvas_size = imgui.get_content_region_available()

            center_x_marker = canvas_abs_pos[0] + canvas_size[0] / 2.0

            if canvas_size[0] <= 0 or canvas_size[1] <= 0:
                imgui.end()
                if is_floating:
                    imgui.end()
                return

            draw_list.add_rect_filled(canvas_abs_pos[0], canvas_abs_pos[1], canvas_abs_pos[0] + canvas_size[0], canvas_abs_pos[1] + canvas_size[1], imgui.get_color_u32_rgba(0.08, 0.08, 0.1, 1))

            video_loaded = self.app.processor and self.app.processor.video_info and self.app.processor.total_frames > 0
            video_fps_for_calc = self.app.processor.fps if video_loaded and self.app.processor.fps > 0 else 30.0
            effective_total_duration_s, _, _ = self.app.get_effective_video_duration_params()
            effective_total_duration_ms = effective_total_duration_s * 1000.0

            # --- Coordinate Transformation Helpers (Unchanged) ---
            def time_to_x(time_ms: float) -> float:
                if app_state.timeline_zoom_factor_ms_per_px == 0: return canvas_abs_pos[0]
                return canvas_abs_pos[0] + (time_ms - app_state.timeline_pan_offset_ms) / app_state.timeline_zoom_factor_ms_per_px

            def x_to_time(x_pos: float) -> float:
                return (x_pos - canvas_abs_pos[0]) * app_state.timeline_zoom_factor_ms_per_px + app_state.timeline_pan_offset_ms

            def pos_to_y(val: int) -> float:
                if canvas_size[1] == 0:
                    return canvas_abs_pos[1] + canvas_size[1] / 2.0
                return canvas_abs_pos[1] + canvas_size[1] * (1.0 - (val / 100.0))

            def y_to_pos(y_pos: float) -> int:
                if canvas_size[1] == 0: return 50
                val = (1.0 - (y_pos - canvas_abs_pos[1]) / canvas_size[1]) * 100.0
                return min(100, max(0, int(round(val))))

            def time_to_x_vec(time_ms_arr: np.ndarray) -> np.ndarray:
                if app_state.timeline_zoom_factor_ms_per_px == 0:
                    return np.full_like(time_ms_arr, canvas_abs_pos[0], dtype=float)
                return canvas_abs_pos[0] + (time_ms_arr - app_state.timeline_pan_offset_ms) / app_state.timeline_zoom_factor_ms_per_px

            def pos_to_y_vec(val_arr: np.ndarray) -> np.ndarray:
                if canvas_size[1] == 0: return np.full_like(val_arr, canvas_abs_pos[1] + canvas_size[1] / 2.0, dtype=float)
                return canvas_abs_pos[1] + canvas_size[1] * (1.0 - (val_arr / 100.0))

            # endregion

            # --- Deferred Zoom Logic Execution ---
            if self.zoom_action_request is not None:
                # if can_manual_pan_zoom:
                time_at_current_center_ms = x_to_time(center_x_marker)
                scale_factor = self.zoom_action_request
                app_state.timeline_zoom_factor_ms_per_px = max(0.01,
                                                               min(app_state.timeline_zoom_factor_ms_per_px * scale_factor,
                                                                   2000.0))
                app_state.timeline_pan_offset_ms = time_at_current_center_ms - (
                            canvas_size[0] / 2.0) * app_state.timeline_zoom_factor_ms_per_px
                app_state.timeline_interaction_active = True
                self.is_zooming_active = True

                self.zoom_action_request = None  # Reset request

            # Pan/Zoom boundaries
            center_marker_offset_ms = (canvas_size[0] / 2.0) * app_state.timeline_zoom_factor_ms_per_px
            min_pan_allowed = -center_marker_offset_ms
            max_pan_allowed = effective_total_duration_ms - center_marker_offset_ms
            if max_pan_allowed < min_pan_allowed: max_pan_allowed = min_pan_allowed

            # --- Unified Input and State Machine ---
            # region StateMachine
            was_interacting = app_state.timeline_interaction_active
            is_interacting_this_frame = False
            # Reset pan and zoom flags at the start of each frame
            was_panning_active = self.is_panning_active
            was_zooming_active = self.is_zooming_active
            self.is_panning_active = False
            self.is_zooming_active = False

            # Original strict hover check for general timeline interaction
            is_timeline_hovered = (imgui.is_window_hovered()
                and canvas_abs_pos[0] <= mouse_pos[0] < canvas_abs_pos[0] + canvas_size[0] 
                and canvas_abs_pos[1] <= mouse_pos[1] < canvas_abs_pos[1] + canvas_size[1])

            # Relaxed hover check specifically for starting a marquee selection
            # This allows starting a marquee a few pixels outside the strict canvas bounds
            marquee_hover_padding = 5.0 * self.app.app_settings.get("global_font_scale", 1.0)  # Adjust padding as needed
            is_timeline_hovered_for_marquee_start = (imgui.is_window_hovered()
                and canvas_abs_pos[0] - marquee_hover_padding <= mouse_pos[0] < canvas_abs_pos[0] + canvas_size[0] + marquee_hover_padding
                and canvas_abs_pos[1] - marquee_hover_padding <= mouse_pos[1] < canvas_abs_pos[1] + canvas_size[1] + marquee_hover_padding)

            can_manual_pan_zoom = (video_loaded and not self.app.processor.is_processing) or not video_loaded

            # Check all inputs that count as interaction
            if is_timeline_hovered and can_manual_pan_zoom:  # Only allow pan/zoom if strictly hovered over canvas
                # Mouse Pan/Zoom
                is_mouse_panning = (imgui.is_mouse_dragging(glfw.MOUSE_BUTTON_MIDDLE)
                        or (io.key_shift
                        and imgui.is_mouse_dragging(glfw.MOUSE_BUTTON_LEFT)
                        and self.dragging_action_idx == -1
                        and not self.is_marqueeing))

                if is_mouse_panning:
                    app_state.timeline_pan_offset_ms -= io.mouse_delta[0] * app_state.timeline_zoom_factor_ms_per_px
                    is_interacting_this_frame = True
                    self.is_panning_active = True

                if io.mouse_wheel != 0.0:
                    # Capture the time at the current center of the visible timeline BEFORE zoom
                    time_at_current_center_ms = x_to_time(center_x_marker)  # Using the center_x_marker defined earlier

                    scale_factor = 0.85 if io.mouse_wheel > 0 else 1.15
                    app_state.timeline_zoom_factor_ms_per_px = max(0.01, min(app_state.timeline_zoom_factor_ms_per_px * scale_factor, 2000.0))
                    # Calculate new pan offset to keep the *original center time* at the *new center of the screen*
                    app_state.timeline_pan_offset_ms = time_at_current_center_ms - (canvas_size[0] / 2.0) * app_state.timeline_zoom_factor_ms_per_px

                    is_interacting_this_frame = True
                    self.is_zooming_active = True

            # Also count point dragging and marqueeing as interaction
            if self.dragging_action_idx != -1 and imgui.is_mouse_dragging(glfw.MOUSE_BUTTON_LEFT):
                is_interacting_this_frame = True
            if self.is_marqueeing and imgui.is_mouse_dragging(glfw.MOUSE_BUTTON_LEFT):
                is_interacting_this_frame = True

            # --- Keyboard Shortcuts ---
            # region Keyboard
            if imgui.is_window_focused(imgui.FOCUS_ROOT_AND_CHILD_WINDOWS) and allow_editing_timeline:
                # Keyboard Panning
                pan_multiplier = self.app.app_settings.get("timeline_pan_speed_multiplier", 5)
                pan_key_speed_ms = pan_multiplier * app_state.timeline_zoom_factor_ms_per_px
                pan_left_tuple = self.app._map_shortcut_to_glfw_key(
                    shortcuts.get("pan_timeline_left", "ALT+LEFT_ARROW"))

                if pan_left_tuple and (pan_left_tuple[1]['alt'] == io.key_alt
                    and pan_left_tuple[1]['ctrl'] == io.key_ctrl
                    and pan_left_tuple[1]['shift'] == io.key_shift
                    and pan_left_tuple[1]['super'] == io.key_super) \
                    and imgui.is_key_down(pan_left_tuple[0]):
                        app_state.timeline_pan_offset_ms -= pan_key_speed_ms
                        is_interacting_this_frame = True
                        self.is_panning_active = True

                pan_right_tuple = self.app._map_shortcut_to_glfw_key(
                    shortcuts.get("pan_timeline_right", "ALT+RIGHT_ARROW"))

                if pan_right_tuple and (pan_right_tuple[1]['alt'] == io.key_alt
                    and pan_right_tuple[1]['ctrl'] == io.key_ctrl
                    and pan_right_tuple[1]['shift'] == io.key_shift
                    and pan_right_tuple[1]['super'] == io.key_super) \
                    and imgui.is_key_down(pan_right_tuple[0]):
                        app_state.timeline_pan_offset_ms += pan_key_speed_ms
                        is_interacting_this_frame = True
                        self.is_panning_active = True

                # Select All
                select_all_tuple = self.app._map_shortcut_to_glfw_key(shortcuts.get("select_all_points", "CTRL+A"))
                if select_all_tuple and actions_list and imgui.is_key_pressed(select_all_tuple[0]) and all(
                        m == io_m for m, io_m in
                        zip(select_all_tuple[1].values(), [io.key_shift, io.key_ctrl, io.key_alt, io.key_super])):
                    self.multi_selected_action_indices = set(range(len(actions_list)))
                    self.selected_action_idx = min(
                        self.multi_selected_action_indices) if self.multi_selected_action_indices else -1

                # Nudge Position
                pos_nudge_delta = 0
                nudge_up_tuple = self.app._map_shortcut_to_glfw_key(shortcuts.get("nudge_selection_pos_up", "UP_ARROW"))
                if nudge_up_tuple and imgui.is_key_pressed(nudge_up_tuple[0]) and all(m == io_m for m, io_m in zip(nudge_up_tuple[1].values(), [io.key_shift, io.key_ctrl, io.key_alt, io.key_super])):
                    pos_nudge_delta = app_state.snap_to_grid_pos if app_state.snap_to_grid_pos > 0 else 1
                nudge_down_tuple = self.app._map_shortcut_to_glfw_key(
                    shortcuts.get("nudge_selection_pos_down", "DOWN_ARROW"))
                if pos_nudge_delta == 0 and nudge_down_tuple and imgui.is_key_pressed(nudge_down_tuple[0]) and all(m == io_m for m, io_m in zip(nudge_down_tuple[1].values(), [io.key_shift, io.key_ctrl, io.key_alt, io.key_super])):
                    pos_nudge_delta = -(app_state.snap_to_grid_pos if app_state.snap_to_grid_pos > 0 else 1)

                if pos_nudge_delta != 0 and self.multi_selected_action_indices:
                    op_desc = f"Nudged Point(s) Pos by {pos_nudge_delta}"
                    fs_proc._record_timeline_action(self.timeline_num, op_desc)
                    for idx in self.multi_selected_action_indices:
                        if 0 <= idx < len(actions_list): actions_list[idx]["pos"] = min(100, max(0, actions_list[idx]["pos"] + pos_nudge_delta))
                    fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)

                # Nudge Time
                time_nudge_delta_ms = 0
                snap_grid_time_ms_for_nudge = app_state.snap_to_grid_time_ms if app_state.snap_to_grid_time_ms > 0 else (
                    int(1000 / video_fps_for_calc) if video_fps_for_calc > 0 else 20)
                nudge_prev_tuple = self.app._map_shortcut_to_glfw_key(
                    shortcuts.get("nudge_selection_time_prev", "SHIFT+LEFT_ARROW"))
                if nudge_prev_tuple and imgui.is_key_pressed(nudge_prev_tuple[0]) and all(m == io_m for m, io_m in zip(
                        nudge_prev_tuple[1].values(), [io.key_shift, io.key_ctrl, io.key_alt, io.key_super])):
                    time_nudge_delta_ms = -snap_grid_time_ms_for_nudge
                nudge_next_tuple = self.app._map_shortcut_to_glfw_key(
                    shortcuts.get("nudge_selection_time_next", "SHIFT+RIGHT_ARROW"))
                if time_nudge_delta_ms == 0 and nudge_next_tuple and imgui.is_key_pressed(nudge_next_tuple[0]) and all(
                        m == io_m for m, io_m in
                        zip(nudge_next_tuple[1].values(), [io.key_shift, io.key_ctrl, io.key_alt, io.key_super])):
                    time_nudge_delta_ms = snap_grid_time_ms_for_nudge

                if time_nudge_delta_ms != 0 and self.multi_selected_action_indices:
                    op_desc = f"Nudged Point(s) Time by {time_nudge_delta_ms}ms"
                    fs_proc._record_timeline_action(self.timeline_num, op_desc)
                    indices_to_affect = sorted(list(self.multi_selected_action_indices), reverse=(time_nudge_delta_ms < 0))

                    objects_to_move = [actions_list[idx] for idx in indices_to_affect]
                    for i, action_obj in enumerate(objects_to_move):
                        current_idx_in_list = actions_list.index(action_obj)
                        new_at = action_obj["at"] + time_nudge_delta_ms

                        prev_at_limit = -1
                        # Find true previous point NOT in the current selection being moved
                        for k in range(current_idx_in_list - 1, -1, -1):
                            if actions_list[k] not in objects_to_move: prev_at_limit = actions_list[k]["at"] + 1; break
                        if prev_at_limit == -1: prev_at_limit = 0

                        next_at_limit = float('inf')
                        # Find true next point NOT in the current selection being moved
                        for k in range(current_idx_in_list + 1, len(actions_list)):
                            if actions_list[k] not in objects_to_move: next_at_limit = actions_list[k]["at"] - 1; break
                        if video_loaded: next_at_limit = min(next_at_limit, effective_total_duration_ms)

                        action_obj["at"] = int(
                            round(np.clip(float(new_at), float(prev_at_limit), float(next_at_limit))))

                    actions_list.sort(key=lambda a: a["at"])
                    # Re-select moved points robustly
                    self.multi_selected_action_indices = {actions_list.index(obj) for obj in objects_to_move if obj in actions_list}
                    self.selected_action_idx = min(
                        self.multi_selected_action_indices) if self.multi_selected_action_indices else -1
                    fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)

                # Delete Selected
                del_sc_str = shortcuts.get("delete_selected_point", "DELETE")
                del_alt_sc_str = shortcuts.get("delete_selected_point_alt", "BACKSPACE")
                del_key_tuple = self.app._map_shortcut_to_glfw_key(del_sc_str)
                bck_key_tuple = self.app._map_shortcut_to_glfw_key(del_alt_sc_str)
                delete_pressed = False

                if del_key_tuple and imgui.is_key_pressed(del_key_tuple[0]) and all(m == io_m for m, io_m in zip(del_key_tuple[1].values(), [io.key_shift, io.key_ctrl, io.key_alt, io.key_super])):
                    delete_pressed = True

                if (not delete_pressed
                    and bck_key_tuple
                    and imgui.is_key_pressed(bck_key_tuple[0])
                    and all(m == io_m for m, io_m in zip(bck_key_tuple[1].values(),
                    [io.key_shift, io.key_ctrl, io.key_alt, io.key_super]))):
                        delete_pressed = True

                if delete_pressed and self.multi_selected_action_indices:
                    op_desc = f"Deleted {len(self.multi_selected_action_indices)} Selected Point(s) (Key)"
                    fs_proc._record_timeline_action(self.timeline_num, op_desc)
                    target_funscript_instance_for_render.clear_points(axis=axis_name_for_render, selected_indices = list(self.multi_selected_action_indices))
                    self.multi_selected_action_indices.clear()
                    self.selected_action_idx = -1
                    fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)

                # Add Point with Number Keys
                time_at_center_add = x_to_time(center_x_marker)
                snap_time_add_key = app_state.snap_to_grid_time_ms if app_state.snap_to_grid_time_ms > 0 else 1
                snapped_time_add_key = max(0, int(round(time_at_center_add / snap_time_add_key)) * snap_time_add_key)

                for i in range(10):
                    bound_key_str_num = shortcuts.get(f"add_point_{i * 10}", str(i))
                    if not bound_key_str_num: continue
                    glfw_key_num_tuple = self.app._map_shortcut_to_glfw_key(bound_key_str_num)
                    if (glfw_key_num_tuple and imgui.is_key_pressed(glfw_key_num_tuple[0]) and all(m == io_m for m, io_m in  zip(glfw_key_num_tuple[1].values(), [io.key_shift, io.key_ctrl, io.key_alt, io.key_super]))):
                        target_pos_val = i * 10
                        op_desc = "Added Point (Key)"
                        fs_proc._record_timeline_action(self.timeline_num, op_desc)

                        # FIX: Corrected typo from `add_actio` to `add_action`.
                        target_funscript_instance_for_render.add_action(
                            timestamp_ms=snapped_time_add_key,
                            primary_pos=target_pos_val if axis_name_for_render == 'primary' else None,
                            secondary_pos=target_pos_val if axis_name_for_render == 'secondary' else None,
                            is_from_live_tracker=False
                        )

                        # Re-fetch actions and select the new point
                        actions_list = getattr(target_funscript_instance_for_render, f"{axis_name_for_render}_actions",
                                               [])
                        new_idx = next((idx for idx, act in enumerate(actions_list) if
                                        act['at'] == snapped_time_add_key and act['pos'] == target_pos_val), -1)
                        if new_idx != -1:
                            # Set the new point as the only selected point
                            self.selected_action_idx = new_idx
                            self.multi_selected_action_indices = {new_idx}

                        fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)
                        break
            # endregion

            # Update the shared state flag for this frame
            app_state.timeline_interaction_active = is_interacting_this_frame

            # Clip pan offset if any interaction occurred
            if app_state.timeline_interaction_active:
                app_state.timeline_pan_offset_ms = np.clip(app_state.timeline_pan_offset_ms, min_pan_allowed, max_pan_allowed)

            # Detect end of *panning* interaction to seek video
            # Only seek if a pan was active and it just ended, OR if a non-panning interaction just ended
            # AND there's no continuous interaction active right now (like auto-pan for playback)
            # This specifically prevents video seeking on zoom ending.
            if (was_panning_active or was_zooming_active) and not self.is_panning_active and not self.is_zooming_active:
                if video_loaded and not self.app.processor.is_processing:
                    center_x_marker = canvas_abs_pos[0] + canvas_size[0] / 2.0
                    time_at_center_ms = x_to_time(center_x_marker)
                    clamped_time_ms = np.clip(time_at_center_ms, 0, effective_total_duration_ms)
                    target_frame = int(round((clamped_time_ms / 1000.0) * video_fps_for_calc))
                    clamped_frame = np.clip(target_frame, 0, self.app.processor.total_frames - 1)
                    # Only seek if the target frame is different from current
                    if abs(clamped_frame - self.app.processor.current_frame_index) > 0:
                        self.app.processor.seek_video(clamped_frame)
                        self.app.project_manager.project_dirty = True
                        app_state.force_timeline_pan_to_current_frame = True  # This will cause it to pan to the frame if needed

            # Auto-pan logic (for playback or forced sync)
            is_playing = video_loaded and self.app.processor.is_processing
            pan_to_current_frame = video_loaded and not is_playing and app_state.force_timeline_pan_to_current_frame
            if (is_playing or pan_to_current_frame) and not app_state.timeline_interaction_active:  # No manual interaction right now
                    current_video_time_ms = (self.app.processor.current_frame_index / video_fps_for_calc) * 1000.0
                    # Using effective center of screen
                    target_pan_offset = current_video_time_ms - center_marker_offset_ms  
                    app_state.timeline_pan_offset_ms = np.clip(
                        target_pan_offset, min_pan_allowed,
                        max_pan_allowed)
                    if pan_to_current_frame:
                        app_state.force_timeline_pan_to_current_frame = False

            marker_color_fixed = imgui.get_color_u32_rgba(0.9, 0.2, 0.2, 0.9)
            draw_list.add_line(center_x_marker, canvas_abs_pos[1], center_x_marker, canvas_abs_pos[1] + canvas_size[1], marker_color_fixed, 1.5)
            tri_half_base, tri_height = 5.0, 8.0

            draw_list.add_triangle_filled(
                center_x_marker, canvas_abs_pos[1] + tri_height,
                center_x_marker - tri_half_base,
                canvas_abs_pos[1],
                center_x_marker + tri_half_base,
                canvas_abs_pos[1],
                marker_color_fixed)

            time_at_center_ms_display = x_to_time(center_x_marker)
            time_str_display_main = _format_time(self.app, time_at_center_ms_display / 1000.0)
            frame_str_display = ""
            if video_loaded and video_fps_for_calc > 0:
                frame_at_center = int(round((time_at_center_ms_display / 1000.0) * video_fps_for_calc))
                frame_str_display = f" (F: {frame_at_center})"

            full_time_display_str = f"{time_str_display_main}{frame_str_display}"
            draw_list.add_text(center_x_marker + 5, canvas_abs_pos[1] + tri_height + 5,
                               imgui.get_color_u32_rgba(1, 1, 1, 0.7), full_time_display_str)

            # Grid drawing (Horizontal - Position)
            for i in range(5):  # 0, 25, 50, 75, 100
                y_grid_h = canvas_abs_pos[1] + (i / 4.0) * canvas_size[1]
                grid_col = imgui.get_color_u32_rgba(0.2, 0.2, 0.2, 0.8 if i != 2 else 0.9)  # Center line darker
                line_thickness = 1.0 if i != 2 else 1.5
                draw_list.add_line(canvas_abs_pos[0], y_grid_h, canvas_abs_pos[0] + canvas_size[0], y_grid_h, grid_col, line_thickness)
                pos_val = 100 - int((i / 4.0) * 100)
                text_y_offset = -imgui.get_text_line_height() - 2 if i == 4 else (
                    2 if i == 0 else -imgui.get_text_line_height() / 2)
                draw_list.add_text(canvas_abs_pos[0] + 3, y_grid_h + text_y_offset, imgui.get_color_u32_rgba(0.7, 0.7, 0.7, 1), str(pos_val))

            # Grid drawing (Vertical - Time)
            time_per_screen_ms_grid = canvas_size[0] * app_state.timeline_zoom_factor_ms_per_px
            px_per_100ms_grid = 100.0 / app_state.timeline_zoom_factor_ms_per_px if app_state.timeline_zoom_factor_ms_per_px > 0 else 0
            if px_per_100ms_grid > 80:
                time_step_ms_grid = 100
            elif px_per_100ms_grid > 40:
                time_step_ms_grid = 200
            elif px_per_100ms_grid > 20:
                time_step_ms_grid = 500
            elif px_per_100ms_grid > 8:
                time_step_ms_grid = 1000
            elif px_per_100ms_grid > 4:
                time_step_ms_grid = 2000
            elif px_per_100ms_grid > 2:
                time_step_ms_grid = 5000
            elif px_per_100ms_grid > 0.5:
                time_step_ms_grid = 10000
            else: time_step_ms_grid = 60000

            if time_step_ms_grid > 0:
                start_time = app_state.timeline_pan_offset_ms
                end_time = start_time + time_per_screen_ms_grid

                # Find the first grid line within the visible range
                first_grid_time = math.ceil(start_time / time_step_ms_grid) * time_step_ms_grid

                # Loop over grid lines within the visible range
                for t_ms in np.arange(first_grid_time, end_time + time_step_ms_grid, time_step_ms_grid):
                    x_pos = time_to_x(t_ms)

                    # Skip if x position is outside the canvas
                    if not (canvas_abs_pos[0] <= x_pos <= canvas_abs_pos[0] + canvas_size[0]):
                        continue

                    is_major_tick = (t_ms % (time_step_ms_grid * 5)) == 0
                    tick_color = imgui.get_color_u32_rgba(0.2, 0.2, 0.2, 0.9 if is_major_tick else 0.8)
                    tick_thickness = 1.5 if is_major_tick else 1.0

                    # Draw vertical grid line
                    draw_list.add_line(
                        x_pos, canvas_abs_pos[1],
                        x_pos, canvas_abs_pos[1] + canvas_size[1],
                        tick_color, tick_thickness)

                    # Optionally draw time label
                    if not video_loaded or (0 <= t_ms <= effective_total_duration_ms + 1e-4):
                        label = f"{t_ms / 1000.0:.1f}s"
                        label_color = imgui.get_color_u32_rgba(0.7, 0.7, 0.7, 1.0)
                        draw_list.add_text(x_pos + 3, canvas_abs_pos[1] + 3, label_color, label)

            # --- Draw Audio Waveform ---
            if self.app.app_state_ui.show_audio_waveform and self.app.audio_waveform_data is not None:
                    waveform_data = self.app.audio_waveform_data
                    num_samples = len(waveform_data)

                    if num_samples > 1 and effective_total_duration_ms > 0:
                        # Compute visible time range on timeline
                        visible_start_ms = app_state.timeline_pan_offset_ms
                        visible_end_ms = visible_start_ms + (canvas_size[0] * app_state.timeline_zoom_factor_ms_per_px)

                        # Convert visible time range to sample indices
                        start_idx = int(max(0, (visible_start_ms / effective_total_duration_ms) * num_samples))
                        end_idx = int(min(num_samples, (visible_end_ms / effective_total_duration_ms) * num_samples + 2))

                        if end_idx > start_idx:
                            # Slice visible waveform segment
                            visible_indices = np.arange(start_idx, end_idx)
                            visible_times_ms = (visible_indices / (num_samples - 1)) * effective_total_duration_ms
                            visible_amplitudes = waveform_data[start_idx:end_idx]

                            # Calculate coordinates
                            x_coords = time_to_x_vec(visible_times_ms)
                            canvas_center_y = canvas_abs_pos[1] + (canvas_size[1] / 2.0)
                            y_offsets = visible_amplitudes * (canvas_size[1] / 2.0)
                            y_coords_top = canvas_center_y - y_offsets
                            y_coords_bottom = canvas_center_y + y_offsets

                        waveform_color = imgui.get_color_u32_rgba(0.2, 0.35, 0.6, 0.6)
                        points_top = list(zip(x_coords, y_coords_top))
                        points_bottom = list(zip(x_coords, y_coords_bottom))
                        draw_list.add_polyline(points_top, waveform_color, False, 1.0)
                        draw_list.add_polyline(points_bottom, waveform_color, False, 1.0)

            # --- Draw Actions (Points and Lines) ---
            hovered_action_idx_current_timeline = -1
            visible_actions_indices_range = None
            if actions_list:
                action_times = [action["at"] for action in actions_list]
                margin_ms_act = 2000
                search_start_time = app_state.timeline_pan_offset_ms - margin_ms_act
                search_end_time = app_state.timeline_pan_offset_ms + canvas_size[0] * app_state.timeline_zoom_factor_ms_per_px + margin_ms_act
                start_idx = bisect_left(action_times, search_start_time)
                end_idx = bisect_right(action_times, search_end_time)
                if start_idx < end_idx:
                    visible_actions_indices_range = (start_idx, end_idx)

            # --- Draw Lines (Vectorized) ---
            if len(actions_list) > 1 and visible_actions_indices_range:
                s_idx, e_idx = visible_actions_indices_range
                render_e_idx = min(e_idx, len(actions_list) - 1)

                if s_idx < render_e_idx:
                    p1_indices = np.arange(s_idx, render_e_idx)
                    p2_indices = p1_indices + 1

                    p1_ats = np.array([actions_list[i]["at"] for i in p1_indices], dtype=float)
                    p1_poss = np.array([actions_list[i]["pos"] for i in p1_indices], dtype=float)
                    p2_ats = np.array([actions_list[i]["at"] for i in p2_indices], dtype=float)
                    p2_poss = np.array([actions_list[i]["pos"] for i in p2_indices], dtype=float)

                    x1s = time_to_x_vec(p1_ats)
                    y1s = pos_to_y_vec(p1_poss)
                    x2s = time_to_x_vec(p2_ats)
                    y2s = pos_to_y_vec(p2_poss)

                    delta_t_ms_vec = p2_ats - p1_ats
                    delta_pos_vec = np.abs(p2_poss - p1_poss)
                    speeds_vec = np.divide(delta_pos_vec, delta_t_ms_vec / 1000.0, out=np.zeros_like(delta_pos_vec), where=delta_t_ms_vec > 1e-5)

                    for i in range(len(x1s)):
                        x1, y1, x2, y2 = x1s[i], y1s[i], x2s[i], y2s[i]

                        canvas_x, canvas_w = canvas_abs_pos[0], canvas_size[0]
                        canvas_x_end = canvas_x + canvas_w

                        # Skip lines that are completely outside the horizontal canvas bounds
                        if (x1 < canvas_x and x2 < canvas_x) or (x1 > canvas_x_end and x2 > canvas_x_end):
                            continue

                        color = self.app.utility.get_speed_color_from_map(speeds_vec[i])
                        alpha = 0.2 if self.is_previewing else 1.0
                        thickness = 1.0 if self.is_previewing else 2.0

                        final_color = imgui.get_color_u32_rgba(color[0], color[1], color[2], color[3] * alpha)
                        draw_list.add_line(x1, y1, x2, y2, final_color, thickness)

            # --- Draw Points ---
            if actions_list and visible_actions_indices_range:
                s_idx, e_idx = visible_actions_indices_range
                point_indices = np.arange(s_idx, e_idx)
                if point_indices.size > 0:
                    point_ats = np.array([actions_list[i]["at"] for i in point_indices], dtype=float)
                    point_poss = np.array([actions_list[i]["pos"] for i in point_indices], dtype=float)
                    pxs = time_to_x_vec(point_ats)
                    pys = pos_to_y_vec(point_poss)

                    for i_loop, original_list_idx in enumerate(point_indices):
                        px, py = pxs[i_loop], pys[i_loop]
                        dist_sq = (mouse_pos[0] - px) ** 2 + (mouse_pos[1] - py) ** 2
                        is_hovered_pt = dist_sq < (app_state.timeline_point_radius + 4) ** 2
                        is_primary_selected = (original_list_idx == self.selected_action_idx)
                        is_in_multi_selection = (original_list_idx in self.multi_selected_action_indices)
                        is_being_dragged = (original_list_idx == self.dragging_action_idx)
                        point_radius_draw = app_state.timeline_point_radius
                        pt_color_tuple = (0.3, 0.9, 0.3, 1)

                        if is_being_dragged:
                            pt_color_tuple = (1.0, 0.2, 0.2, 1.0)
                            point_radius_draw += 1
                        elif is_primary_selected or is_in_multi_selection:
                            pt_color_tuple = (1.0, 0.0, 0.0, 1.0)
                            if is_in_multi_selection and not is_primary_selected: point_radius_draw += 0.5
                        elif is_hovered_pt and imgui.is_window_hovered() and not self.is_marqueeing:
                            pt_color_tuple = (0.5, 1.0, 0.5, 1.0)
                            if self.dragging_action_idx == -1:
                                hovered_action_idx_current_timeline = original_list_idx

                        point_alpha = 0.3 if self.is_previewing else 1.0
                        final_pt_color = imgui.get_color_u32_rgba(pt_color_tuple[0], pt_color_tuple[1], pt_color_tuple[2], pt_color_tuple[3] * point_alpha)

                        draw_list.add_circle_filled(px, py, point_radius_draw, final_pt_color)

                        if is_primary_selected and not is_being_dragged:
                            draw_list.add_circle(px, py, point_radius_draw + 1, imgui.get_color_u32_rgba(0.6, 0.0, 0.0, 1.0), thickness=1.0)

            # --- Draw Preview Actions on Top ---
            if self.is_previewing and self.preview_actions:
                preview_line_color = imgui.get_color_u32_rgba(1.0, 0.5, 0.0, 0.9)  # Bright Orange
                preview_point_color = imgui.get_color_u32_rgba(1.0, 0.5, 0.0, 1.0)
                preview_point_radius = app_state.timeline_point_radius

                # Draw preview lines and points using the self.preview_actions list
                if len(self.preview_actions) > 1:
                    p_ats = np.array([a['at'] for a in self.preview_actions])
                    p_poss = np.array([a['pos'] for a in self.preview_actions])
                    pxs = time_to_x_vec(p_ats)
                    pys = pos_to_y_vec(p_poss)
                    for i in range(len(pxs) - 1):
                        draw_list.add_line(pxs[i], pys[i], pxs[i + 1], pys[i + 1],
                        preview_line_color, 2.0)

                for action in self.preview_actions:
                    px = time_to_x(action['at'])
                    py = pos_to_y(action['pos'])
                    draw_list.add_circle_filled(px, py, preview_point_radius, preview_point_color)

            # --- Draw Marquee Selection ---
            if self.is_marqueeing and self.marquee_start_screen_pos and self.marquee_end_screen_pos:
                min_x, max_x = min(self.marquee_start_screen_pos[0], self.marquee_end_screen_pos[0]), max(self.marquee_start_screen_pos[0], self.marquee_end_screen_pos[0])
                min_y, max_y = min(self.marquee_start_screen_pos[1], self.marquee_end_screen_pos[1]), max(self.marquee_start_screen_pos[1], self.marquee_end_screen_pos[1])
                draw_list.add_rect_filled(min_x, min_y, max_x, max_y, imgui.get_color_u32_rgba(0.5, 0.5, 1.0, 0.3))
                draw_list.add_rect(min_x, min_y, max_x, max_y, imgui.get_color_u32_rgba(0.8, 0.8, 1.0, 0.7))

            # --- Mouse Interactions ---
            # region Mouse
            # Use the relaxed hover check for starting the marquee
            if imgui.is_mouse_clicked(glfw.MOUSE_BUTTON_LEFT) and not io.key_shift:
                if hovered_action_idx_current_timeline != -1:
                    # ====== START: CALIBRATION HOOK ======
                    if self.app.calibration and self.app.calibration.is_calibration_mode_active:
                        # 1. Notify the calibration manager with the timestamp
                        clicked_action_time_ms = actions_list[hovered_action_idx_current_timeline]['at']
                        self.app.calibration.handle_calibration_point_selection(clicked_action_time_ms)

                        # 2. Update this timeline's UI to select the point
                        self.selected_action_idx = hovered_action_idx_current_timeline
                        self.multi_selected_action_indices = {hovered_action_idx_current_timeline}

                        # 3. Seek the video and focus the timeline on the selected point
                        if video_loaded and not self.app.processor.is_processing and video_fps_for_calc > 0:
                            target_frame_on_click = int(
                                round((clicked_action_time_ms / 1000.0) * video_fps_for_calc))
                            self.app.processor.seek_video(
                                np.clip(target_frame_on_click, 0, self.app.processor.total_frames - 1))
                            app_state.force_timeline_pan_to_current_frame = True

                        # Prevent any other click logic from running for this event
                        imgui.end()
                        if is_floating: imgui.end()
                        return
                    # ====== END: CALIBRATION HOOK ======

                    self.is_marqueeing = False
                    if not io.key_ctrl: self.multi_selected_action_indices.clear()
                    if hovered_action_idx_current_timeline in self.multi_selected_action_indices and io.key_ctrl:
                        self.multi_selected_action_indices.remove(hovered_action_idx_current_timeline)
                    else:
                        self.multi_selected_action_indices.add(hovered_action_idx_current_timeline)
                    self.selected_action_idx = min(
                        self.multi_selected_action_indices) if self.multi_selected_action_indices else -1

                    self.dragging_action_idx = hovered_action_idx_current_timeline
                    if not self.drag_undo_recorded and allow_editing_timeline:
                        fs_proc._record_timeline_action(self.timeline_num, "Start Point Drag")
                        self.drag_undo_recorded = True

                    if video_loaded and not self.app.processor.is_processing and video_fps_for_calc > 0:
                        target_frame_on_click = int(round((actions_list[hovered_action_idx_current_timeline]["at"] / 1000.0) * video_fps_for_calc))
                        self.app.processor.seek_video(
                            np.clip(target_frame_on_click, 0, self.app.processor.total_frames - 1))
                        app_state.force_timeline_pan_to_current_frame = True
                elif is_timeline_hovered_for_marquee_start:  # Use the relaxed hover check here for marquee
                    # Start marquee
                    self.is_marqueeing = True
                    self.marquee_start_screen_pos = mouse_pos
                    self.marquee_end_screen_pos = mouse_pos
                    if not io.key_ctrl:
                        self.multi_selected_action_indices.clear()
                        self.selected_action_idx = -1

            # Context Menu (only when strictly hovered over the canvas, not the padding)
            if imgui.is_mouse_clicked(glfw.MOUSE_BUTTON_RIGHT) and is_timeline_hovered:
                self.context_mouse_pos_screen = mouse_pos
                time_at_click = x_to_time(mouse_pos[0])
                pos_at_click = y_to_pos(mouse_pos[1])
                snap_time = app_state.snap_to_grid_time_ms if app_state.snap_to_grid_time_ms > 0 else 1
                snap_pos = app_state.snap_to_grid_pos if app_state.snap_to_grid_pos > 0 else 1
                self.new_point_candidate_at = max(0, int(round(time_at_click / snap_time)) * snap_time)
                self.new_point_candidate_pos = min(100, max(0, int(round(pos_at_click / snap_pos)) * snap_pos))

                # Store the hovered_action_idx_current_timeline value
                # because it might reset before the popup renders its contents.
                self.context_menu_point_idx = hovered_action_idx_current_timeline

                if self.context_menu_point_idx != -1 and not (
                        self.context_menu_point_idx in self.multi_selected_action_indices):
                    if not io.key_ctrl: self.multi_selected_action_indices.clear()
                    self.multi_selected_action_indices.add(self.context_menu_point_idx)
                    self.selected_action_idx = self.context_menu_point_idx
                imgui.open_popup(context_popup_id)

            # Point Drag
            if self.dragging_action_idx != -1 and imgui.is_mouse_dragging(
                    glfw.MOUSE_BUTTON_LEFT) and allow_editing_timeline and not io.key_shift:
                if 0 <= self.dragging_action_idx < len(actions_list):
                    action_to_drag = actions_list[self.dragging_action_idx]
                    new_time_cand_ms = x_to_time(mouse_pos[0])
                    new_pos_cand = y_to_pos(mouse_pos[1])
                    snap_time = app_state.snap_to_grid_time_ms if app_state.snap_to_grid_time_ms > 0 else 1
                    snap_pos = app_state.snap_to_grid_pos if app_state.snap_to_grid_pos > 0 else 1
                    snapped_new_at = max(0, int(round(new_time_cand_ms / snap_time)) * snap_time)
                    snapped_new_pos = min(100, max(0, int(round(new_pos_cand / snap_pos)) * snap_pos))

                    effective_prev_at_lim = actions_list[self.dragging_action_idx - 1]["at"] + 1 if self.dragging_action_idx > 0 else 0
                    effective_next_at_lim = actions_list[self.dragging_action_idx + 1]["at"] - 1 if self.dragging_action_idx < len(actions_list) - 1 else float('inf')
                    action_to_drag["at"] = int(np.clip(float(snapped_new_at), float(effective_prev_at_lim), float(effective_next_at_lim)))
                    action_to_drag["pos"] = snapped_new_pos

                    self.app.project_manager.project_dirty = True
                    if self.timeline_num == 1:
                        app_state.heatmap_dirty = True
                        app_state.funscript_preview_dirty = True

            # Marquee Drag
            if self.is_marqueeing and imgui.is_mouse_dragging(glfw.MOUSE_BUTTON_LEFT) and not io.key_shift:
                self.marquee_end_screen_pos = mouse_pos

            # Handle Mouse Release for Marquee Selection
            if imgui.is_mouse_released(glfw.MOUSE_BUTTON_LEFT) and self.is_marqueeing:
                self.is_marqueeing = False

                # Ensure all conditions are valid before attempting selection
                if self.marquee_start_screen_pos and self.marquee_end_screen_pos and visible_actions_indices_range and actions_list:
                    start_x, start_y = self.marquee_start_screen_pos
                    end_x, end_y = self.marquee_end_screen_pos

                    min_x, max_x = min(start_x, end_x), max(start_x, end_x)
                    min_y, max_y = min(start_y, end_y), max(start_y, end_y)

                    s_idx_vis, e_idx_vis = visible_actions_indices_range
                    vis_range = range(s_idx_vis, e_idx_vis)

                    # Extract visible action times and positions
                    vis_ats = np.array([actions_list[i]["at"] for i in vis_range], dtype=float)
                    vis_poss = np.array([actions_list[i]["pos"] for i in vis_range], dtype=float)

                    if vis_ats.size > 0:
                        vis_pxs = time_to_x_vec(vis_ats)
                        vis_pys = pos_to_y_vec(vis_poss)

                        in_rect_mask = (
                            (vis_pxs >= min_x) & (vis_pxs <= max_x) &
                            (vis_pys >= min_y) & (vis_pys <= max_y))
                        newly_selected = set(np.array(list(vis_range))[in_rect_mask])
                    else:
                        newly_selected = set()

                    if io.key_ctrl:
                        self.multi_selected_action_indices.symmetric_difference_update(newly_selected)
                    else:
                        self.multi_selected_action_indices = newly_selected

                    self.selected_action_idx = (
                        min(self.multi_selected_action_indices)
                        if self.multi_selected_action_indices else -1)

                # Reset marquee state
                self.marquee_start_screen_pos = None
                self.marquee_end_screen_pos = None

                if self.dragging_action_idx != -1 and allow_editing_timeline:
                    dragged_action_ref = actions_list[self.dragging_action_idx]
                    actions_list.sort(key=lambda a: a["at"])
                    try:
                        new_idx = actions_list.index(dragged_action_ref)
                        self.selected_action_idx = new_idx
                        self.multi_selected_action_indices = {new_idx}
                    except ValueError:
                        self.selected_action_idx = -1
                        self.multi_selected_action_indices.clear()

                    if self.drag_undo_recorded:
                        fs_proc._finalize_action_and_update_ui(self.timeline_num, "Point Dragged")
                        self.drag_undo_recorded = False
                    self.dragging_action_idx = -1
            # endregion

            # --- Context Menu & Tooltip ---
            # region Context/Tooltip
            if imgui.begin_popup(context_popup_id):
                imgui.text(f"Timeline {self.timeline_num} @ Time: {self.new_point_candidate_at}ms, Pos: {self.new_point_candidate_pos}")
                imgui.separator()

                if allow_editing_timeline:
                    if imgui.menu_item(f"Add Point Here##CTXMenuAdd{window_id_suffix}")[0]:
                        op_desc = "Added Point (Menu)"
                        fs_proc._record_timeline_action(self.timeline_num, op_desc)

                        primary_pos, secondary_pos = None, None
                        if axis_name_for_render == 'primary':
                            primary_pos = self.new_point_candidate_pos
                        elif axis_name_for_render == 'secondary':
                            secondary_pos = self.new_point_candidate_pos

                        target_funscript_instance_for_render.add_action(
                            timestamp_ms=self.new_point_candidate_at,
                            primary_pos=primary_pos,
                            secondary_pos=secondary_pos,
                            is_from_live_tracker=False
                        )

                        fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)
                        imgui.close_current_popup()
                else:
                    imgui.menu_item("Add Point Here", enabled=False)

                imgui.separator()
                can_copy_to_other = allow_editing_timeline and bool(self.multi_selected_action_indices)
                copy_dest_t_num = 2 if self.timeline_num == 1 else 1

                if imgui.menu_item(f"Copy Selected to Timeline {copy_dest_t_num}", enabled=can_copy_to_other)[0]:
                    if can_copy_to_other:
                        self._handle_copy_to_other_timeline()
                    imgui.close_current_popup()

                imgui.separator()

                # --- New "Start Selection" / "End Selection" Context Menu Items ---
                # Use self.context_menu_point_idx here
                if self.context_menu_point_idx != -1:
                    # Only show "Start selection" if no anchor is set or if clicking the same anchor point again
                    if self.selection_anchor_idx == -1 or self.selection_anchor_idx == self.context_menu_point_idx:
                        if imgui.menu_item(f"Start Selection Here##CTXMenuStartSel{window_id_suffix}")[0]:
                            self.selection_anchor_idx = self.context_menu_point_idx
                            self.selected_action_idx = self.selection_anchor_idx
                            self.multi_selected_action_indices.clear()
                            self.multi_selected_action_indices.add(self.selection_anchor_idx)
                            self.app.logger.info(f"T{self.timeline_num}: Selection start point set at index {self.selection_anchor_idx}.")
                            imgui.close_current_popup()
                    # Only show "End selection" if an anchor is set AND it's a different point
                    elif self.selection_anchor_idx != -1 and self.selection_anchor_idx != self.context_menu_point_idx:
                        if imgui.menu_item(f"End Selection Here##CTXMenuEndSel{window_id_suffix}")[0]:
                            start_idx = min(self.selection_anchor_idx, self.context_menu_point_idx)
                            end_idx = max(self.selection_anchor_idx, self.context_menu_point_idx)

                            new_selection = set(range(start_idx, end_idx + 1))
                            self.multi_selected_action_indices.update(new_selection)
                            self.selected_action_idx = self.context_menu_point_idx  # Set the last clicked as primary
                            self.selection_anchor_idx = -1  # Reset anchor
                            self.app.logger.info(
                                f"T{self.timeline_num}: Selected points from {start_idx} to {end_idx}.")
                            imgui.close_current_popup()
                else:  # If not hovering over a point when the context menu was opened
                    imgui.menu_item(f"Start Selection Here", enabled=False)
                    imgui.menu_item(f"End Selection Here", enabled=False)
                # --- End new items ---
                imgui.separator()

                can_copy = allow_editing_timeline and (bool(self.multi_selected_action_indices) or self.selected_action_idx != -1)
                if imgui.menu_item(f"Copy Selected##CTXMenuCopy", shortcut=shortcuts.get("copy_selection", "Ctrl+C"), enabled=can_copy)[0]:
                    if can_copy: self._handle_copy_selection()
                    imgui.close_current_popup()
                can_paste = allow_editing_timeline and self.app.funscript_processor.clipboard_has_actions()
                if imgui.menu_item(f"Paste at Cursor##CTXMenuPaste", shortcut=shortcuts.get("paste_selection", "Ctrl+V"), enabled=can_paste)[0]:
                    if can_paste: self._handle_paste_actions(self.new_point_candidate_at)
                    imgui.close_current_popup()
                imgui.separator()
                if imgui.menu_item(f"Cancel##CTXMenuCancel")[0]: imgui.close_current_popup()
                imgui.end_popup()

            if hovered_action_idx_current_timeline != -1 and self.dragging_action_idx == -1 and not imgui.is_popup_open(context_popup_id):
                if 0 <= hovered_action_idx_current_timeline < len(actions_list):
                    action_hovered = actions_list[hovered_action_idx_current_timeline]
                    imgui.begin_tooltip()
                    imgui.text(f"Time: {action_hovered['at']} ms ({action_hovered['at'] / 1000.0:.2f}s) | Pos: {action_hovered['pos']}")
                    if video_loaded and video_fps_for_calc > 0:
                        imgui.text(f"Frame: {int(round((action_hovered['at'] / 1000.0) * video_fps_for_calc))}")
                    if hovered_action_idx_current_timeline > 0:
                        dt = action_hovered['at'] - actions_list[hovered_action_idx_current_timeline - 1]['at']
                        speed = abs(action_hovered['pos'] - actions_list[hovered_action_idx_current_timeline - 1]['pos']) / (dt / 1000.0) if dt > 0 else 0
                        imgui.text(f"In-Speed: {speed:.1f} pos/s")
                    if hovered_action_idx_current_timeline < len(actions_list) - 1:
                        dt = actions_list[hovered_action_idx_current_timeline + 1]['at'] - action_hovered['at']
                        speed = abs(actions_list[hovered_action_idx_current_timeline + 1]['pos'] - action_hovered['pos']) / (dt / 1000.0) if dt > 0 else 0
                        imgui.text(f"Out-Speed: {speed:.1f} pos/s")
                    imgui.end_tooltip()
            # endregion

        # --- Window End ---
        imgui.end()
        if is_floating:
            imgui.end()
