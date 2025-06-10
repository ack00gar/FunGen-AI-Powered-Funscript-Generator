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

        # Get initial defaults from app_settings (AppLogic holds app_settings directly)
        self.sg_window_length = self.app.app_settings.get(f"timeline{self.timeline_num}_sg_default_window", 5)
        self.sg_poly_order = self.app.app_settings.get(f"timeline{self.timeline_num}_sg_default_polyorder", 2)
        self.rdp_epsilon = self.app.app_settings.get(f"timeline{self.timeline_num}_rdp_default_epsilon", 8.0)

        self.multi_selected_action_indices = set()
        self.is_marqueeing = False
        self.marquee_start_screen_pos = None
        self.marquee_end_screen_pos = None

    def _get_target_funscript_details(self) -> Tuple[Optional[object], Optional[str]]:
        if self.app.funscript_processor:
            return self.app.funscript_processor._get_target_funscript_object_and_axis(self.timeline_num)
        return None, None

    def _get_actions_list_ref(self) -> Optional[List[Dict]]:
        funscript_instance, axis_name = self._get_target_funscript_details()
        if funscript_instance and axis_name:
            return getattr(funscript_instance, f"{axis_name}_actions", None)
        return None

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
        num_pasted = 0
        newly_pasted_timestamps = []
        for action_data in clipboard_data:
            new_at = max(0, int(paste_at_time_ms + action_data['relative_at']))
            new_pos = int(action_data['pos'])

            funscript_instance.add_action(
                timestamp_ms=new_at,
                primary_pos=new_pos if axis_name == 'primary' else None,
                secondary_pos=new_pos if axis_name == 'secondary' else None,
                is_from_live_tracker=False
            )
            newly_pasted_timestamps.append(new_at)
            num_pasted += 1

        if num_pasted == 0: return

        updated_actions_list_ref = self._get_actions_list_ref()
        if not updated_actions_list_ref: return
        self.multi_selected_action_indices.clear()
        new_selected_indices = set()

        for ts_pasted in newly_pasted_timestamps:
            for i, act in enumerate(updated_actions_list_ref):
                if act['at'] == ts_pasted:
                    new_selected_indices.add(i)

        self.multi_selected_action_indices = new_selected_indices
        self.selected_action_idx = min(self.multi_selected_action_indices) if self.multi_selected_action_indices else -1

        self.app.funscript_processor._finalize_action_and_update_ui(self.timeline_num, op_desc)
        self.app.logger.info(f"Pasted {num_pasted} point(s).", extra={'status_message': True})

    # --- Bulk Operations Direct Calls to Funscript Object ---
    def _call_funscript_method(self, method_name: str, error_context: str, **kwargs) -> bool:
        funscript_instance, axis_name = self._get_target_funscript_details()
        if not funscript_instance or not axis_name:
            self.app.logger.warning(f"T{self.timeline_num}: Cannot {error_context}. Funscript object not found.",
                                    extra={'status_message': True})
            return False
        try:
            method_to_call = getattr(funscript_instance, method_name)
            method_to_call(axis=axis_name, **kwargs)
            return True
        except Exception as e:
            self.app.logger.error(
                f"T{self.timeline_num} Error in {error_context} ({method_name}): {str(e)}", exc_info=True,
                extra={'status_message': True})
            return False

    def _perform_sg_filter(self, window_length: int, polyorder: int, selected_indices: Optional[List[int]]):
        return self._call_funscript_method('apply_savitzky_golay', 'SG filter',
                                           window_length=window_length, polyorder=polyorder,
                                           selected_indices=selected_indices)

    def _perform_rdp_simplification(self, epsilon: float, selected_indices: Optional[List[int]]):
        return self._call_funscript_method('simplify_rdp', 'RDP simplification',
                                           epsilon=epsilon, selected_indices=selected_indices)

    def _perform_inversion(self, selected_indices: Optional[List[int]]):
        return self._call_funscript_method('invert_points_values', 'inversion',
                                           selected_indices=selected_indices)

    def _perform_clamp(self, clamp_value: int, selected_indices: Optional[List[int]]):
        return self._call_funscript_method('clamp_points_values', f'clamp to {clamp_value}',
                                           clamp_value=clamp_value, selected_indices=selected_indices)

    def render(self, timeline_y_start_coord: float = None, timeline_render_height: float = None):
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
            is_open, new_visibility = imgui.begin(window_title, closable=True,
                                                  flags=imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_SCROLL_WITH_MOUSE)
            if new_visibility != is_visible:
                setattr(app_state, visibility_flag_name, new_visibility)
                self.app.project_manager.project_dirty = True
            if not is_open:
                should_render_content = False
        else:  # Fixed mode
            if timeline_y_start_coord is None or timeline_render_height is None or timeline_render_height <= 0:
                return  # Cannot render fixed without coordinates
            imgui.set_next_window_position(0, timeline_y_start_coord)
            imgui.set_next_window_size(app_state.window_width, timeline_render_height)
            imgui.begin(f"Funscript Editor Timeline##Interactive{self.timeline_num}", flags=(
                    imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_SCROLL_WITH_MOUSE))

        # --- Content Rendering ---
        if should_render_content:
            fs_proc = self.app.funscript_processor
            target_funscript_instance_for_render, axis_name_for_render = self._get_target_funscript_details()
            actions_list = []
            if target_funscript_instance_for_render and axis_name_for_render:
                actions_list = getattr(target_funscript_instance_for_render, f"{axis_name_for_render}_actions", [])

            io = imgui.get_io()
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

            if sg_disabled_bool:
                imgui.pop_style_var()
                imgui.internal.pop_item_flag()

            # --- SG Settings Window ---
            sg_window_title = f"Savitzky-Golay Filter Settings (Timeline {self.timeline_num})##SGSettingsWindow{window_id_suffix}"
            if self.show_sg_settings_popup:
                main_viewport = imgui.get_main_viewport()
                # Center the window; consider making this behavior optional or using set_next_window_appearing_position
                popup_pos_x = main_viewport.pos[0] + (main_viewport.size[0] - 350) * 0.5
                popup_pos_y = main_viewport.pos[1] + (main_viewport.size[1] - 200) * 0.5
                imgui.set_next_window_position(popup_pos_x, popup_pos_y, condition=imgui.APPEARING)
                imgui.set_next_window_size(350, 0, condition=imgui.APPEARING)
                window_expanded, self.show_sg_settings_popup = imgui.begin(
                    sg_window_title, closable=self.show_sg_settings_popup, flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)
                if window_expanded:
                    imgui.text(f"Savitzky-Golay Filter (Timeline {self.timeline_num})")
                    imgui.separator()

                    wl_changed, current_wl = imgui.slider_int("Window Length##SGWinPopup", self.sg_window_length, 3, 99)
                    if wl_changed:
                        self.sg_window_length = current_wl if current_wl % 2 != 0 else current_wl + 1
                        if self.sg_window_length < 3: self.sg_window_length = 3

                    max_po = max(1, self.sg_window_length - 1)
                    po_val = min(self.sg_poly_order, max_po)
                    if po_val < 1: po_val = 1
                    po_changed, current_po = imgui.slider_int("Polyorder##SGPolyPopup", po_val, 1, max_po)
                    if po_changed: self.sg_poly_order = current_po

                    if self.multi_selected_action_indices and len(self.multi_selected_action_indices) >= 2:
                        _, self.sg_apply_to_selection = imgui.checkbox(
                            f"Apply to {len(self.multi_selected_action_indices)} selected##SGApplyToSel",
                            self.sg_apply_to_selection)
                    else:
                        imgui.text_disabled("Apply to: Full Script")
                        self.sg_apply_to_selection = False

                    if imgui.button(f"Apply##SGApplyPop{window_id_suffix}"):
                        indices_to_use = list(self.multi_selected_action_indices) if self.sg_apply_to_selection else None
                        op_desc = f"Applied SG (W:{self.sg_window_length}, P:{self.sg_poly_order})" + (
                            " to selection" if indices_to_use else "")
                        fs_proc._record_timeline_action(self.timeline_num, op_desc)
                        if self._perform_sg_filter(self.sg_window_length, self.sg_poly_order,
                                                   selected_indices=indices_to_use):
                            fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)
                            self.app.app_settings.set(f"timeline{self.timeline_num}_sg_default_window",
                                                      self.sg_window_length)
                            self.app.app_settings.set(f"timeline{self.timeline_num}_sg_default_polyorder",
                                                      self.sg_poly_order)
                            self.app.logger.info(f"{op_desc} on T{self.timeline_num}.", extra={'status_message': True})
                        self.show_sg_settings_popup = False  # Close window
                    imgui.same_line()
                    if imgui.button(f"Cancel##SGCancelPop{window_id_suffix}"):
                        self.show_sg_settings_popup = False  # Close window
                imgui.end()
            imgui.same_line()

            # RDP Button
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

            # --- RDP Settings Window ---
            rdp_window_title = f"RDP Simplification Settings (Timeline {self.timeline_num})##RDPSettingsWindow{window_id_suffix}"
            if self.show_rdp_settings_popup:
                main_viewport = imgui.get_main_viewport()
                popup_pos_x = main_viewport.pos[0] + (main_viewport.size[0] - 350) * 0.5
                popup_pos_y = main_viewport.pos[1] + (main_viewport.size[1] - 180) * 0.5
                imgui.set_next_window_position(popup_pos_x, popup_pos_y, condition=imgui.APPEARING)
                imgui.set_next_window_size(350, 0, condition=imgui.APPEARING)

                window_expanded, self.show_rdp_settings_popup = imgui.begin(
                    rdp_window_title, closable=self.show_rdp_settings_popup, flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)
                if window_expanded:
                    imgui.text(f"RDP Simplification (Timeline {self.timeline_num})")
                    imgui.separator()
                    _, self.rdp_epsilon = imgui.slider_float("Epsilon##RDPEpsPopup", self.rdp_epsilon, 0.1, 20.0, "%.1f")
                    if self.multi_selected_action_indices and len(self.multi_selected_action_indices) >= 2:
                        _, self.rdp_apply_to_selection = imgui.checkbox(
                            f"Apply to {len(self.multi_selected_action_indices)} selected##RDPApplyToSel",
                            self.rdp_apply_to_selection)
                    else:
                        imgui.text_disabled("Apply to: Full Script")
                        self.rdp_apply_to_selection = False

                    if imgui.button(f"Apply##RDPApplyPop{window_id_suffix}"):
                        indices_to_use = list(self.multi_selected_action_indices) if self.rdp_apply_to_selection else None
                        op_desc = f"Applied RDP (Epsilon:{self.rdp_epsilon:.1f})" + (
                            " to selection" if indices_to_use else "")
                        fs_proc._record_timeline_action(self.timeline_num, op_desc)
                        if self._perform_rdp_simplification(self.rdp_epsilon, selected_indices=indices_to_use):
                            fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)
                            self.app.app_settings.set(f"timeline{self.timeline_num}_rdp_default_epsilon", self.rdp_epsilon)
                            self.app.logger.info(f"{op_desc} on T{self.timeline_num}.", extra={'status_message': True})
                        self.show_rdp_settings_popup = False  # Close window
                    imgui.same_line()
                    if imgui.button(f"Cancel##RDPCancelPop{window_id_suffix}"):
                        self.show_rdp_settings_popup = False  # Close window
                imgui.end()
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
                    indices_to_op = list(
                        self.multi_selected_action_indices) if self.multi_selected_action_indices else None
                    op_desc = "Clamped Selected to 100" if indices_to_op else "Clamped All to 100"
                    fs_proc._record_timeline_action(self.timeline_num, op_desc)
                    if self._perform_clamp(100, selected_indices=indices_to_op):
                        fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)
                        self.app.logger.info(f"{op_desc} on T{self.timeline_num}.", extra={'status_message': True})
            if clamp_disabled_bool: imgui.pop_style_var(); imgui.internal.pop_item_flag()
            imgui.same_line()

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
                    target_funscript_instance_for_render.clear_points(axis=axis_name_for_render,
                                                                      selected_indices=indices_to_clear)
                    if indices_to_clear: self.multi_selected_action_indices.clear()
                    self.selected_action_idx = -1
                    self.dragging_action_idx = -1
                    fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)
                    self.app.logger.info(f"{op_desc} on T{self.timeline_num}.", extra={'status_message': True})
            if clear_op_disabled: imgui.pop_style_var(); imgui.internal.pop_item_flag()
            imgui.same_line()

            imgui.text_colored(script_info_text, 0.75, 0.75, 0.75, 0.95)
            # --- (Drag and drop target for T2 remains the same) ---
            if self.timeline_num == 2:
                if imgui.begin_drag_drop_target():
                    payload_accepted = imgui.accept_drag_drop_payload("FILES")
                    if payload_accepted is not None and self.app.file_manager.last_dropped_files:
                        if self.app.file_manager.last_dropped_files[0].lower().endswith(".funscript"):
                            dropped_path_t2 = self.app.file_manager.last_dropped_files[0]
                            self.app.file_manager.load_funscript_to_timeline(dropped_path_t2, timeline_num=2)
                        self.app.file_manager.last_dropped_files = None
                    imgui.end_drag_drop_target()

            # --- (Canvas setup, grid, points, lines drawing remains the same) ---
            draw_list = imgui.get_window_draw_list()
            canvas_abs_pos = imgui.get_cursor_screen_pos()
            canvas_size = imgui.get_content_region_available()

            if canvas_size[0] <= 0 or canvas_size[1] <= 0:
                imgui.end()
                return

            draw_list.add_rect_filled(canvas_abs_pos[0], canvas_abs_pos[1], canvas_abs_pos[0] + canvas_size[0],
                                      canvas_abs_pos[1] + canvas_size[1], imgui.get_color_u32_rgba(0.08, 0.08, 0.1, 1))

            video_loaded = self.app.processor and self.app.processor.video_info and self.app.processor.total_frames > 0
            # Use self.app.processor.fps directly where FPS is needed for frame calculations,
            # and ensure it's > 0 for valid calculation.
            # video_fps is used as a fallback for some calculations if actual fps isn't available,
            # but for frame number display, we should rely on actual positive FPS.
            video_fps_for_calc = self.app.processor.fps if video_loaded and self.app.processor.fps and self.app.processor.fps > 0 else 0

            effective_total_duration_s, _, _ = self.app.get_effective_video_duration_params()
            effective_total_duration_ms = effective_total_duration_s * 1000.0

            # --- NumPy Vectorized Coordinate Transformation Helpers ---
            def time_to_x_vec(time_ms_arr: np.ndarray) -> np.ndarray:
                if app_state.timeline_zoom_factor_ms_per_px == 0:
                    return np.full_like(time_ms_arr, canvas_abs_pos[0], dtype=float)
                return canvas_abs_pos[0] + (
                        time_ms_arr - app_state.timeline_pan_offset_ms) / app_state.timeline_zoom_factor_ms_per_px

            def pos_to_y_vec(val_arr: np.ndarray) -> np.ndarray:
                if canvas_size[1] == 0:
                    return np.full_like(val_arr, canvas_abs_pos[1] + canvas_size[1] / 2.0, dtype=float)
                return canvas_abs_pos[1] + canvas_size[1] * (1.0 - (val_arr / 100.0))

            # --- Original Scalar Coordinate Transformation (still needed for single points) ---
            def time_to_x(time_ms: float) -> float:
                if app_state.timeline_zoom_factor_ms_per_px == 0: return canvas_abs_pos[0]
                return canvas_abs_pos[0] + (
                        time_ms - app_state.timeline_pan_offset_ms) / app_state.timeline_zoom_factor_ms_per_px

            def x_to_time(x_pos: float) -> float:
                return (x_pos - canvas_abs_pos[
                    0]) * app_state.timeline_zoom_factor_ms_per_px + app_state.timeline_pan_offset_ms

            def pos_to_y(val: int) -> float:
                if canvas_size[1] == 0: return canvas_abs_pos[1] + canvas_size[1] / 2.0
                return canvas_abs_pos[1] + canvas_size[1] * (1.0 - (val / 100.0))

            def y_to_pos(y_pos: float) -> int:
                if canvas_size[1] == 0: return 50
                val = (1.0 - (y_pos - canvas_abs_pos[1]) / canvas_size[1]) * 100.0
                return min(100, max(0, int(round(val))))

            current_video_time_ms = (
                                                self.app.processor.current_frame_index / video_fps_for_calc) * 1000.0 if video_loaded and video_fps_for_calc > 0 else 0.0
            center_marker_offset_ms = (canvas_size[0] / 2.0) * app_state.timeline_zoom_factor_ms_per_px
            min_pan_allowed = -center_marker_offset_ms
            max_pan_allowed = effective_total_duration_ms - center_marker_offset_ms
            if max_pan_allowed < min_pan_allowed: max_pan_allowed = min_pan_allowed

            if ((video_loaded and self.app.processor.is_processing) or \
                (
                        video_loaded and not self.app.processor.is_processing and app_state.force_timeline_pan_to_current_frame)) and \
                    not app_state.timeline_interaction_active:
                target_pan_offset = current_video_time_ms - center_marker_offset_ms
                app_state.timeline_pan_offset_ms = np.clip(target_pan_offset, min_pan_allowed, max_pan_allowed)
                app_state.last_synced_frame_index_timeline = self.app.processor.current_frame_index if video_loaded else -1
                if app_state.force_timeline_pan_to_current_frame: self.app.project_manager.project_dirty = True
                app_state.force_timeline_pan_to_current_frame = False

            center_x_marker = canvas_abs_pos[0] + canvas_size[0] / 2.0
            marker_color_fixed = imgui.get_color_u32_rgba(0.9, 0.2, 0.2, 0.9)
            draw_list.add_line(center_x_marker, canvas_abs_pos[1], center_x_marker, canvas_abs_pos[1] + canvas_size[1],
                               marker_color_fixed, 1.5)
            tri_half_base, tri_height = 5.0, 8.0
            draw_list.add_triangle_filled(center_x_marker, canvas_abs_pos[1] + tri_height, center_x_marker - tri_half_base,

                                          canvas_abs_pos[1], center_x_marker + tri_half_base, canvas_abs_pos[1],
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
                draw_list.add_line(canvas_abs_pos[0], y_grid_h, canvas_abs_pos[0] + canvas_size[0], y_grid_h, grid_col,
                                   line_thickness)
                pos_val = 100 - int((i / 4.0) * 100)
                text_y_offset = -imgui.get_text_line_height() - 2 if i == 4 else (
                    2 if i == 0 else -imgui.get_text_line_height() / 2)
                draw_list.add_text(canvas_abs_pos[0] + 3, y_grid_h + text_y_offset,
                                   imgui.get_color_u32_rgba(0.7, 0.7, 0.7, 1), str(pos_val))

            # Grid drawing (Vertical - Time)
            time_per_screen_ms_grid = canvas_size[0] * app_state.timeline_zoom_factor_ms_per_px
            time_step_ms_grid = 1000.0
            if app_state.timeline_zoom_factor_ms_per_px > 0:
                px_per_100ms_grid = 100.0 / app_state.timeline_zoom_factor_ms_per_px
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
                else:
                    time_step_ms_grid = 60000

            start_visible_time_ms_grid, end_visible_time_ms_grid = app_state.timeline_pan_offset_ms, app_state.timeline_pan_offset_ms + time_per_screen_ms_grid
            if time_step_ms_grid > 0:
                first_line_time_ms_grid = math.ceil(start_visible_time_ms_grid / time_step_ms_grid) * time_step_ms_grid
                for t_ms_grid in np.arange(first_line_time_ms_grid, end_visible_time_ms_grid + time_step_ms_grid,
                                           time_step_ms_grid):
                    x_grid = time_to_x(t_ms_grid)
                    if canvas_abs_pos[0] <= x_grid <= canvas_abs_pos[0] + canvas_size[0]:
                        is_major = (t_ms_grid % (time_step_ms_grid * 5)) == 0
                        grid_col_t = imgui.get_color_u32_rgba(0.2, 0.2, 0.2, 0.8 if not is_major else 0.9)
                        line_thick_t = 1.0 if not is_major else 1.5
                        draw_list.add_line(x_grid, canvas_abs_pos[1], x_grid, canvas_abs_pos[1] + canvas_size[1],
                                           grid_col_t, line_thick_t)

                        time_label_txt = f"{t_ms_grid / 1000.0:.1f}s"
                        should_display_label = True
                        if video_loaded and not (0 <= t_ms_grid <= effective_total_duration_ms + 1e-4):
                            should_display_label = False
                        elif not video_loaded and actions_list and not (0 <= t_ms_grid <= actions_list[-1]["at"] + 1e-4):
                            should_display_label = False

                        if should_display_label:
                            draw_list.add_text(x_grid + 3, canvas_abs_pos[1] + 3,
                                               imgui.get_color_u32_rgba(0.7, 0.7, 0.7, 1), time_label_txt)

            # --- Draw Actions (Points and Lines) ---
            hovered_action_idx_current_timeline = -1
            mouse_pos = imgui.get_mouse_pos()
            start_vis_time_act, end_vis_time_act = app_state.timeline_pan_offset_ms, app_state.timeline_pan_offset_ms + \
                                                                                     canvas_size[
                                                                                         0] * app_state.timeline_zoom_factor_ms_per_px
            margin_ms_act = 2000  # Render points slightly outside view for smooth panning

            visible_actions_indices_range = None  # Initialize to None
            if actions_list:
                action_times = [action["at"] for action in actions_list]
                search_start_time = start_vis_time_act - margin_ms_act
                search_end_time = end_vis_time_act + margin_ms_act
                start_idx = bisect_left(action_times, search_start_time)
                end_idx = bisect_right(action_times, search_end_time)
                start_idx = max(0, start_idx)
                # Ensure end_idx is not out of bounds for list length AND for slicing (should be one greater than last index needed)
                end_idx = min(len(actions_list), end_idx)

                # A range is valid if start_idx is less than end_idx.
                # bisect_left/right can return len(list), so end_idx can be len(list)
                # This range is for actions_list[start_idx:end_idx]
                if start_idx < end_idx:  # and end_idx <= len(actions_list) # end_idx is already capped
                    visible_actions_indices_range = (start_idx, end_idx)

            # --- Draw Actions (Points and Lines) using NumPy ---
            if actions_list and visible_actions_indices_range:
                s_idx, e_idx = visible_actions_indices_range

                # FIX: Re-validate indices against the current length of actions_list to prevent
                # a race condition where the list is modified by another thread after the
                # indices were calculated.
                current_list_len = len(actions_list)
                s_idx = min(s_idx, current_list_len)
                e_idx = min(e_idx, current_list_len)

                # --- Lines ---
                # To draw lines between points actions_list[i] and actions_list[i+1],
                # we iterate from s_idx up to e_idx-1 (or len(actions_list)-1 if e_idx is at the end)
                if (e_idx - s_idx) >= 1 and len(actions_list) > 1:
                    p1_ats_list = []
                    p1_poss_list = []
                    p2_ats_list = []
                    p2_poss_list = []

                    # Iterate for the first point of the segment
                    for i in range(s_idx, e_idx):  # Iterate through all potentially visible points
                        if i + 1 < len(actions_list):  # Check if there's a next point to form a line
                            # Further check: if p2 (i+1) is beyond e_idx, it might not be strictly needed for display
                            # but if lines are drawn based on p1 being in visible_actions_indices_range, then it's fine.
                            # The original logic was: range(s_idx, actual_render_e_idx_for_p1) where actual_render_e_idx_for_p1 = min(e_idx, len(actions_list) - 1)
                            # This meant p1 went up to actual_render_e_idx_for_p1 - 1 effectively for list access inside loop.
                            # Let's use a simpler range from s_idx to e_idx-1 for p1, ensuring p2 (i+1) is valid.
                            if i < e_idx - 1:  # p1 is from s_idx to e_idx-2. This ensures p2 (i+1) can be up to e_idx-1.
                                # And both points are within the original [s_idx, e_idx) slice.
                                p1 = actions_list[i]
                                p2 = actions_list[i + 1]  # p2 is actions_list[i+1]
                                p1_ats_list.append(p1["at"])
                                p1_poss_list.append(p1["pos"])
                                p2_ats_list.append(p2["at"])
                                p2_poss_list.append(p2["pos"])
                            # If i == e_idx -1 (last point in visible range), we might still need to draw a line
                            # to actions_list[i+1] if i+1 is valid and potentially just outside e_idx,
                            # for smooth connection when panning.
                            # The bisect_right for end_idx gives the insertion point, so actions_list[e_idx-1] is the last visible.
                            # We need to draw lines up to actions_list[e_idx-1].
                            # So, if point i is in [s_idx, e_idx-1] and point i+1 exists, draw a line.
                            # The loop should ensure p1 is at most e_idx-1, and p2 is at most len(actions_list)-1.
                            # Let's ensure p1 is at most e_idx-1.
                            # And that p1_ats_list will contain points up to e_idx-1 (if they form a pair with i+1)

                    # Simpler: Iterate through indices that will be the *first* point of a line segment.
                    # These indices go from s_idx up to min(e_idx - 1, len(actions_list) - 2).
                    # No, this is also complex. Let's use the visible points range and connect them.
                    # Points are actions_list[s_idx]...actions_list[e_idx-1]
                    # Lines are between (s_idx, s_idx+1), ..., (e_idx-2, e_idx-1)

                    # Correct range for collecting segments:
                    # p1 ranges from index s_idx to (e_idx-1)-1 = e_idx-2
                    # This ensures that p2 (index i+1) is at most e_idx-1
                    # Both points of the segment are within the original visible_actions_indices_range slice
                    # And also within the bounds of the full actions_list

                    # Iterate from the first visible point up to the second to last *visible* point
                    # The points considered are from index s_idx to e_idx-1.
                    # We need to form pairs (actions_list[i], actions_list[i+1])
                    # So, i can go from s_idx to e_idx-2, ensuring i+1 is at most e_idx-1.
                    # All these indices must also be < len(actions_list).

                    # Iterate over the indices of the first point of each segment
                    for i in range(s_idx, e_idx - 1):  # i goes from s_idx to e_idx-2
                        if i + 1 < len(actions_list):  # Ensures actions_list[i+1] is valid
                            # Both actions_list[i] and actions_list[i+1] are within the original [s_idx, e_idx)
                            # or actions_list[i+1] could be the point at e_idx if e_idx < len(actions_list)
                            # No, this is simpler: iterate all points in visible range, and if there's a next one *in the full list*, connect them.
                            # The original `actual_render_e_idx_for_p1` was `min(e_idx, len(actions_list) - 1)`
                            # and loop went `range(s_idx, actual_render_e_idx_for_p1)`
                            # This means `i` (p1) goes up to `min(e_idx, len(actions_list) - 1) - 1`.
                            # And `i+1` (p2) goes up to `min(e_idx, len(actions_list) - 1)`.
                            # This seems correct.

                            p1 = actions_list[i]
                            p2 = actions_list[i + 1]  # If i is the last element of actions_list, i+1 is out of bounds.

                            # We need to ensure that p1 is at index `i` and p2 is at index `i+1`.
                            # `i` should iterate such that `i+1` is still a valid index in `actions_list`.
                            # So `i` goes from `s_idx` up to `len(actions_list)-2`.
                            # And also, we only care about lines where at least one point is visible.
                            # The lines connect points within the s_idx to e_idx-1 range primarily.

                            # Let's draw lines between all consecutive points from actions_list[s_idx] to actions_list[e_idx-1]
                            # This means iterating i from s_idx to e_idx-2
                            # The condition i+1 < len(actions_list) is implicitly handled if e_idx <= len(actions_list)

                            # Corrected loop for line segments
                            # Iterate for the first point of the segment.
                            # p1_idx ranges from s_idx to min(e_idx, len(actions_list) - 1) - 1
                            # This ensures p2_idx = p1_idx + 1 is valid and at most min(e_idx, len(actions_list) - 1)

                            # Iterate through points that can be the first point of a visible segment
                            # Points are from actions_list[s_idx]...actions_list[e_idx-1]
                            # Lines are between point i and point i+1
                            # So iterate i from s_idx to e_idx-2. This ensures i and i+1 are in the range [s_idx, e_idx-1]
                            # And ensure i+1 < len(actions_list)

                            # Iterate for p1 from s_idx up to e_idx-1 (the last actual point in the slice)
                            # If p1 is not the very last point in the entire actions_list, then form a pair with p1+1
                            if i < len(actions_list) - 1:  # if actions_list[i] is not the last point overall
                                # If actions_list[i] is within [s_idx, e_idx-1]
                                # and actions_list[i+1] exists.
                                # We want lines where at least one endpoint is "visible".
                                # The original slice is actions_list[s_idx:e_idx].
                                # Points are actions_list[s_idx], ..., actions_list[e_idx-1].
                                # Lines connect actions_list[j] and actions_list[j+1].
                                # We need j to go from s_idx to e_idx-2.

                                # The points to render are indexed s_idx to e_idx-1
                                # Lines connect point[k] to point[k+1]
                                # So k ranges from s_idx to (e_idx-1) - 1 = e_idx-2
                                # This loop is for the *index within actions_list* for the first point of a line.
                                # We also need to ensure that i+1 is a valid index.
                                # The lines should be between points that are "visible" or connect to a visible point.
                                # Iterate `i` from `s_idx` up to `e_idx - 1`. If `actions_list[i+1]` exists, draw a line.

                                # This should iterate for `p1` from `s_idx` up to `min(e_idx, len(actions_list)-1) -1`
                                # to ensure `p2` at `i+1` is valid.

                                # Let's consider points from s_idx to e_idx-1.
                                # A line is between actions_list[k] and actions_list[k+1].
                                # We need k to be in [s_idx, e_idx-2] for both points to be in the original slice.
                                # Or k can be e_idx-1 if we connect to actions_list[e_idx] (if it exists & is relevant).
                                # The original NumPy approach iterates from s_idx to actual_render_e_idx_for_p1
                                # actual_render_e_idx_for_p1 = min(e_idx, len(actions_list) - 1)
                                # This means i goes up to min(e_idx, len(actions_list)-1) - 1
                                # And i+1 (p2) goes up to min(e_idx, len(actions_list)-1)
                                # This seems correct for ensuring p2 is valid and mostly within the view or just outside.

                                p1 = actions_list[i]
                                # Check for p2's validity for safety, though the loop range should handle it.
                                if i + 1 < len(actions_list):
                                    p2 = actions_list[i + 1]
                                    p1_ats_list.append(p1["at"])
                                    p1_poss_list.append(p1["pos"])
                                    p2_ats_list.append(p2["at"])
                                    p2_poss_list.append(p2["pos"])

                    if p1_ats_list:  # If there are segments to draw
                        p1_ats = np.array(p1_ats_list, dtype=float)
                        p1_poss = np.array(p1_poss_list, dtype=float)
                        p2_ats = np.array(p2_ats_list, dtype=float)
                        p2_poss = np.array(p2_poss_list, dtype=float)

                        x1s = time_to_x_vec(p1_ats)
                        y1s = pos_to_y_vec(p1_poss)
                        x2s = time_to_x_vec(p2_ats)
                        y2s = pos_to_y_vec(p2_poss)

                        delta_t_ms_vec = p2_ats - p1_ats
                        delta_pos_vec = np.abs(p2_poss - p1_poss)

                        speeds_vec = np.full_like(delta_pos_vec, 0.0, dtype=float)
                        mask = delta_t_ms_vec > 1e-5
                        speeds_vec[mask] = delta_pos_vec[mask] / (delta_t_ms_vec[mask] / 1000.0)

                        for i_line in range(len(x1s)):
                            x1, y1, x2, y2 = x1s[i_line], y1s[i_line], x2s[i_line], y2s[i_line]
                            if not ((x1 < canvas_abs_pos[0] and x2 < canvas_abs_pos[0]) or \
                                    (x1 > canvas_abs_pos[0] + canvas_size[0] and x2 > canvas_abs_pos[0] + canvas_size[0])):
                                color_tuple = self.app.utility.get_speed_color_from_map(speeds_vec[i_line])
                                draw_list.add_line(x1, y1, x2, y2, imgui.get_color_u32_rgba(*color_tuple), 2.0)

                # --- Points ---
                # Points to draw are from actions_list[s_idx] to actions_list[e_idx-1]
                point_ats_list = [actions_list[i]["at"] for i in range(s_idx, e_idx)]
                point_poss_list = [actions_list[i]["pos"] for i in range(s_idx, e_idx)]

                if point_ats_list:
                    point_ats = np.array(point_ats_list, dtype=float)
                    point_poss = np.array(point_poss_list, dtype=float)

                    pxs = time_to_x_vec(point_ats)
                    pys = pos_to_y_vec(point_poss)

                    for i_loop, original_list_idx in enumerate(range(s_idx, e_idx)):
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
                            if self.dragging_action_idx == -1: hovered_action_idx_current_timeline = original_list_idx

                        draw_list.add_circle_filled(px, py, point_radius_draw, imgui.get_color_u32_rgba(*pt_color_tuple))
                        if is_primary_selected and not is_being_dragged:
                            draw_list.add_circle(px, py, point_radius_draw + 1,
                                                 imgui.get_color_u32_rgba(0.6, 0.0, 0.0, 1.0), thickness=1.0)

            if self.is_marqueeing and self.marquee_start_screen_pos and self.marquee_end_screen_pos:
                rect_min_x = min(self.marquee_start_screen_pos[0], self.marquee_end_screen_pos[0])
                rect_min_y = min(self.marquee_start_screen_pos[1], self.marquee_end_screen_pos[1])
                rect_max_x = max(self.marquee_start_screen_pos[0], self.marquee_end_screen_pos[0])
                rect_max_y = max(self.marquee_start_screen_pos[1], self.marquee_end_screen_pos[1])
                draw_list.add_rect_filled(rect_min_x, rect_min_y, rect_max_x, rect_max_y,
                                          imgui.get_color_u32_rgba(0.5, 0.5, 1.0, 0.3))
                draw_list.add_rect(rect_min_x, rect_min_y, rect_max_x, rect_max_y,
                                   imgui.get_color_u32_rgba(0.8, 0.8, 1.0, 0.7))

            # --- Mouse Interactions (Pan, Zoom, Click, Drag, Marquee, Context Menu) ---
            is_timeline_hovered_for_interaction = imgui.is_window_hovered() and \
                                                  canvas_abs_pos[0] <= mouse_pos[0] < canvas_abs_pos[0] + canvas_size[0] and \
                                                  canvas_abs_pos[1] <= mouse_pos[1] < canvas_abs_pos[1] + canvas_size[1]

            current_interaction_active_this_frame = False

            # Handle opening context menu on right click
            if is_timeline_hovered_for_interaction and imgui.is_mouse_clicked(glfw.MOUSE_BUTTON_RIGHT):
                self.is_marqueeing = False
                self.context_mouse_pos_screen = mouse_pos
                time_at_click = x_to_time(mouse_pos[0])
                pos_at_click = y_to_pos(mouse_pos[1])
                snap_time = app_state.snap_to_grid_time_ms if app_state.snap_to_grid_time_ms > 0 else 1
                snap_pos = app_state.snap_to_grid_pos if app_state.snap_to_grid_pos > 0 else 1
                self.new_point_candidate_at = max(0, int(round(time_at_click / snap_time)) * snap_time)
                self.new_point_candidate_pos = min(100, max(0, int(round(pos_at_click / snap_pos)) * snap_pos))
                if hovered_action_idx_current_timeline != -1:
                    if not io.key_ctrl:
                        if not (
                                len(self.multi_selected_action_indices) == 1 and hovered_action_idx_current_timeline in self.multi_selected_action_indices):
                            self.multi_selected_action_indices.clear()
                            self.multi_selected_action_indices.add(hovered_action_idx_current_timeline)
                        self.selected_action_idx = hovered_action_idx_current_timeline
                    else:
                        if hovered_action_idx_current_timeline in self.multi_selected_action_indices:
                            self.multi_selected_action_indices.remove(hovered_action_idx_current_timeline)
                            if self.selected_action_idx == hovered_action_idx_current_timeline:
                                self.selected_action_idx = min(
                                    self.multi_selected_action_indices) if self.multi_selected_action_indices else -1
                        else:
                            self.multi_selected_action_indices.add(hovered_action_idx_current_timeline)
                            self.selected_action_idx = hovered_action_idx_current_timeline
                imgui.open_popup(context_popup_id)

            if is_timeline_hovered_for_interaction and not imgui.is_popup_open(context_popup_id):
                can_manual_pan_zoom = (video_loaded and not self.app.processor.is_processing) or not video_loaded

                if can_manual_pan_zoom and (imgui.is_mouse_dragging(glfw.MOUSE_BUTTON_MIDDLE) or \
                                            (io.key_shift and imgui.is_mouse_dragging(
                                                glfw.MOUSE_BUTTON_LEFT) and self.dragging_action_idx == -1 and not self.is_marqueeing)):
                    mouse_delta_x = io.mouse_delta[0]
                    if app_state.timeline_zoom_factor_ms_per_px > 0:
                        app_state.timeline_pan_offset_ms -= mouse_delta_x * app_state.timeline_zoom_factor_ms_per_px
                    current_interaction_active_this_frame = True
                    app_state.timeline_pan_offset_ms = np.clip(app_state.timeline_pan_offset_ms, min_pan_allowed,
                                                               max_pan_allowed)

                if can_manual_pan_zoom and io.mouse_wheel != 0:
                    mouse_x_relative_to_canvas = mouse_pos[0] - canvas_abs_pos[0]
                    time_at_mouse_before_zoom = app_state.timeline_pan_offset_ms + mouse_x_relative_to_canvas * app_state.timeline_zoom_factor_ms_per_px

                    scale_factor = 0.85 if io.mouse_wheel > 0 else 1.15
                    new_zoom_factor = max(0.01, min(app_state.timeline_zoom_factor_ms_per_px * scale_factor, 2000.0))
                    new_center_marker_offset_ms = (canvas_size[0] / 2.0) * new_zoom_factor
                    new_min_pan_allowed = -new_center_marker_offset_ms
                    new_max_pan_allowed = effective_total_duration_ms - new_center_marker_offset_ms
                    if new_max_pan_allowed < new_min_pan_allowed: new_max_pan_allowed = new_min_pan_allowed

                    app_state.timeline_zoom_factor_ms_per_px = new_zoom_factor
                    app_state.timeline_pan_offset_ms = time_at_mouse_before_zoom - mouse_x_relative_to_canvas * app_state.timeline_zoom_factor_ms_per_px
                    current_interaction_active_this_frame = True
                    app_state.timeline_pan_offset_ms = np.clip(app_state.timeline_pan_offset_ms, new_min_pan_allowed,
                                                               new_max_pan_allowed)

                if imgui.is_mouse_clicked(glfw.MOUSE_BUTTON_LEFT) and not io.key_shift:
                    if hovered_action_idx_current_timeline != -1:
                        self.is_marqueeing = False
                        if not io.key_ctrl: self.multi_selected_action_indices.clear()

                        if hovered_action_idx_current_timeline in self.multi_selected_action_indices and io.key_ctrl:
                            self.multi_selected_action_indices.remove(hovered_action_idx_current_timeline)
                            if self.selected_action_idx == hovered_action_idx_current_timeline:
                                self.selected_action_idx = min(
                                    self.multi_selected_action_indices) if self.multi_selected_action_indices else -1
                        else:
                            self.multi_selected_action_indices.add(hovered_action_idx_current_timeline)
                            self.selected_action_idx = hovered_action_idx_current_timeline

                        if self.app.calibration.is_calibration_mode_active and self.timeline_num == 1:
                            if 0 <= self.selected_action_idx < len(actions_list):
                                self.app.calibration.handle_calibration_point_selection(
                                    float(actions_list[self.selected_action_idx]['at']))

                        self.dragging_action_idx = hovered_action_idx_current_timeline
                        if not self.drag_undo_recorded and allow_editing_timeline:
                            fs_proc._record_timeline_action(self.timeline_num, "Start Point Drag")
                            self.drag_undo_recorded = True

                        if video_loaded and not self.app.processor.is_processing and video_fps_for_calc > 0:
                            if 0 <= hovered_action_idx_current_timeline < len(actions_list):
                                target_frame_on_click = int(
                                    round((actions_list[hovered_action_idx_current_timeline][
                                               "at"] / 1000.0) * video_fps_for_calc))
                                self.app.processor.seek_video(np.clip(target_frame_on_click, 0,
                                                                      self.app.processor.total_frames - 1 if self.app.processor.total_frames > 0 else 0))
                                app_state.force_timeline_pan_to_current_frame = True
                                self.app.project_manager.project_dirty = True

                    elif not self.is_marqueeing:
                        self.is_marqueeing = True
                        self.marquee_start_screen_pos = mouse_pos
                        self.marquee_end_screen_pos = mouse_pos
                        if not io.key_ctrl:
                            self.multi_selected_action_indices.clear()
                            self.selected_action_idx = -1
            # Marquee Drag
            if self.is_marqueeing and imgui.is_mouse_dragging(glfw.MOUSE_BUTTON_LEFT) and not io.key_shift:
                self.marquee_end_screen_pos = mouse_pos
                current_interaction_active_this_frame = True
            # Point Drag
            if self.dragging_action_idx != -1 and imgui.is_mouse_dragging(
                    glfw.MOUSE_BUTTON_LEFT) and allow_editing_timeline and not io.key_shift:
                current_interaction_active_this_frame = True
                if 0 <= self.dragging_action_idx < len(actions_list):
                    action_to_drag = actions_list[self.dragging_action_idx]
                    new_time_cand_ms = x_to_time(mouse_pos[0])
                    new_pos_cand = y_to_pos(mouse_pos[1])
                    snap_time = app_state.snap_to_grid_time_ms if app_state.snap_to_grid_time_ms > 0 else 1
                    snap_pos = app_state.snap_to_grid_pos if app_state.snap_to_grid_pos > 0 else 1
                    snapped_new_at = max(0, int(round(new_time_cand_ms / snap_time)) * snap_time)
                    snapped_new_pos = min(100, max(0, int(round(new_pos_cand / snap_pos)) * snap_pos))

                    effective_prev_at_lim = actions_list[self.dragging_action_idx - 1][
                                                "at"] + 1 if self.dragging_action_idx > 0 else 0
                    effective_next_at_lim = actions_list[self.dragging_action_idx + 1][
                                                "at"] - 1 if self.dragging_action_idx < len(actions_list) - 1 else float(
                        'inf')
                    action_to_drag["at"] = int(
                        np.clip(float(snapped_new_at), float(effective_prev_at_lim), float(effective_next_at_lim)))
                    action_to_drag["pos"] = snapped_new_pos

                    self.app.project_manager.project_dirty = True
                    if self.timeline_num == 1:
                        app_state.heatmap_dirty = True
                        app_state.funscript_preview_dirty = True

                    if video_loaded and not self.app.processor.is_processing and video_fps_for_calc > 0:
                        target_frame_drag = int(round((action_to_drag["at"] / 1000.0) * video_fps_for_calc))
                        self.app.processor.seek_video(np.clip(target_frame_drag, 0,
                                                              self.app.processor.total_frames - 1 if self.app.processor.total_frames > 0 else 0))
                        app_state.force_timeline_pan_to_current_frame = True
            # Mouse Release
            if imgui.is_mouse_released(glfw.MOUSE_BUTTON_LEFT):
                current_interaction_active_this_frame = False
                if self.is_marqueeing:
                    self.is_marqueeing = False
                    if self.marquee_start_screen_pos and self.marquee_end_screen_pos and visible_actions_indices_range and actions_list:
                        min_x = min(self.marquee_start_screen_pos[0], self.marquee_end_screen_pos[0])
                        max_x = max(self.marquee_start_screen_pos[0], self.marquee_end_screen_pos[0])
                        min_y = min(self.marquee_start_screen_pos[1], self.marquee_end_screen_pos[1])
                        max_y = max(self.marquee_start_screen_pos[1], self.marquee_end_screen_pos[1])

                        s_idx_vis, e_idx_vis = visible_actions_indices_range
                        vis_ats = np.array([actions_list[i]["at"] for i in range(s_idx_vis, e_idx_vis)], dtype=float)
                        vis_poss = np.array([actions_list[i]["pos"] for i in range(s_idx_vis, e_idx_vis)], dtype=float)

                        if vis_ats.size > 0:
                            vis_pxs = time_to_x_vec(vis_ats)
                            vis_pys = pos_to_y_vec(vis_poss)

                            marqueed_bool_x = (vis_pxs >= min_x) & (vis_pxs <= max_x)
                            marqueed_bool_y = (vis_pys >= min_y) & (vis_pys <= max_y)
                            marqueed_in_rect_mask = marqueed_bool_x & marqueed_bool_y
                            original_indices_in_marquee = np.array(range(s_idx_vis, e_idx_vis))[marqueed_in_rect_mask]
                            newly_marqueed_indices = set(original_indices_in_marquee)
                        else:
                            newly_marqueed_indices = set()

                        if io.key_ctrl:
                            self.multi_selected_action_indices.symmetric_difference_update(newly_marqueed_indices)
                        else:
                            self.multi_selected_action_indices = newly_marqueed_indices
                        self.selected_action_idx = min(
                            self.multi_selected_action_indices) if self.multi_selected_action_indices else -1
                    self.marquee_start_screen_pos = None
                    self.marquee_end_screen_pos = None

                if self.dragging_action_idx != -1 and allow_editing_timeline:
                    actions_list.sort(key=lambda a: a["at"])
                    try:
                        # This section needs careful review for robust re-selection after sort,
                        # especially if multiple points are dragged or 'at' values are not unique.
                        # For now, primary selected index might be min of multi-selection or reset.
                        if self.multi_selected_action_indices:
                            # Attempt to find the new indices of previously selected actions if their 'at' changed
                            # This is complex. A simple approach is to update selected_action_idx to the minimum.
                            # A truly robust solution might involve tagging actions before drag and finding them after.
                            # For now, keep it simple or re-evaluate the primary selected action from multi_selected_action_indices
                            if self.dragging_action_idx in self.multi_selected_action_indices:  # Dragged point was in selection
                                # Try to find its new position if possible, or just update based on current multi_selected_action_indices
                                # A placeholder: re-select the minimum if the set is not empty
                                self.selected_action_idx = min(
                                    self.multi_selected_action_indices) if self.multi_selected_action_indices else -1
                            # If a single point was dragged (not part of a multi-select initially)
                            # its index might have changed. It needs to be found again.
                        elif self.selected_action_idx == self.dragging_action_idx:  # A single point was dragged
                            # Find the action that was being dragged. Its 'at' value is now action_to_drag['at']
                            # This part is tricky if action_to_drag reference isn't available or 'at' isn't unique.
                            # For simplicity, if it was singly selected and dragged, it should remain selected.
                            # Try to find its new index.
                            # This requires storing the object or a unique ID.
                            # For now, this might lead to selected_action_idx becoming stale or pointing to a different action
                            # if the sort changed its original index and it wasn't part of multi-selection.
                            # A simple robust fix is to clear selected_action_idx if it's not in multi_selected_action_indices
                            # or re-find it if that's feasible.
                            # For now, this is a known area for potential refinement.
                            pass  # Needs robust re-selection of self.selected_action_idx

                    except ValueError:  # If .index() fails or min() on empty set
                        self.selected_action_idx = -1

                    if self.drag_undo_recorded:
                        fs_proc._finalize_action_and_update_ui(self.timeline_num, "Point Dragged")
                        self.drag_undo_recorded = False
                    self.dragging_action_idx = -1

                    if target_funscript_instance_for_render and axis_name_for_render:
                        setattr(target_funscript_instance_for_render, f"last_timestamp_{axis_name_for_render}",
                                actions_list[-1]['at'] if actions_list else 0)
                else:
                    self.drag_undo_recorded = False

            if current_interaction_active_this_frame and not app_state.timeline_interaction_active:
                app_state.timeline_interaction_active = True
            elif not current_interaction_active_this_frame and app_state.timeline_interaction_active and \
                    self.dragging_action_idx == -1 and not self.is_marqueeing:
                app_state.timeline_interaction_active = False
                if video_loaded and not self.app.processor.is_processing and video_fps_for_calc > 0:
                    time_at_timeline_center_ms_sync = x_to_time(center_x_marker)
                    video_total_duration_ms_sync = ((
                                                                self.app.processor.total_frames / video_fps_for_calc) * 1000.0) if self.app.processor.total_frames > 0 else 0.0
                    clamped_time_at_center_sync = np.clip(time_at_timeline_center_ms_sync, 0,
                                                          video_total_duration_ms_sync if video_total_duration_ms_sync > 0 else float(
                                                              'inf'))
                    target_frame_sync = int(round((clamped_time_at_center_sync / 1000.0) * video_fps_for_calc))
                    target_frame_sync = np.clip(target_frame_sync, 0,
                                                self.app.processor.total_frames - 1 if self.app.processor.total_frames > 0 else 0)
                    if abs(target_frame_sync - self.app.processor.current_frame_index) > 0:
                        self.app.processor.seek_video(target_frame_sync)
                        self.app.project_manager.project_dirty = True

            default_shortcuts = self.app.app_settings.get_default_settings().get("funscript_editor_shortcuts", {})
            shortcuts = self.app.app_settings.get("funscript_editor_shortcuts", default_shortcuts)
            io = imgui.get_io()

            if imgui.is_window_focused(imgui.FOCUS_ROOT_AND_CHILD_WINDOWS) and allow_editing_timeline:
                select_all_sc_str = shortcuts.get("select_all_points", "CTRL+A")
                select_all_tuple = self.app._map_shortcut_to_glfw_key(select_all_sc_str)
                if select_all_tuple and actions_list:
                    key_code_sa, mods_sa = select_all_tuple
                    if (mods_sa['shift'] == io.key_shift and mods_sa['ctrl'] == io.key_ctrl and \
                            mods_sa['alt'] == io.key_alt and mods_sa['super'] == io.key_super and \
                            imgui.is_key_pressed(key_code_sa)):
                        self.multi_selected_action_indices = set(range(len(actions_list)))
                        self.selected_action_idx = min(
                            self.multi_selected_action_indices) if self.multi_selected_action_indices else -1
                        self.app.logger.info(f"T{self.timeline_num}: Selected all {len(actions_list)} points.",
                                             extra={'status_message': True})

                nudge_pos_up_sc_str = shortcuts.get("nudge_selection_pos_up", "UP_ARROW")
                nudge_pos_down_sc_str = shortcuts.get("nudge_selection_pos_down", "DOWN_ARROW")
                nudge_pos_up_tuple = self.app._map_shortcut_to_glfw_key(nudge_pos_up_sc_str)
                nudge_pos_down_tuple = self.app._map_shortcut_to_glfw_key(nudge_pos_down_sc_str)

                pos_nudge_delta = 0
                if nudge_pos_up_tuple:
                    key, mods = nudge_pos_up_tuple
                    if (mods['shift'] == io.key_shift and mods['ctrl'] == io.key_ctrl and mods['alt'] == io.key_alt and
                            mods['super'] == io.key_super and imgui.is_key_pressed(key)):
                        pos_nudge_delta = app_state.snap_to_grid_pos if app_state.snap_to_grid_pos > 0 else 1
                if pos_nudge_delta == 0 and nudge_pos_down_tuple:
                    key, mods = nudge_pos_down_tuple
                    if (mods['shift'] == io.key_shift and mods['ctrl'] == io.key_ctrl and mods['alt'] == io.key_alt and
                            mods['super'] == io.key_super and imgui.is_key_pressed(key)):
                        pos_nudge_delta = -(app_state.snap_to_grid_pos if app_state.snap_to_grid_pos > 0 else 1)
                if pos_nudge_delta != 0 and (self.multi_selected_action_indices or self.selected_action_idx != -1):
                    op_desc = f"Nudged Point(s) Pos by {pos_nudge_delta}"
                    fs_proc._record_timeline_action(self.timeline_num, op_desc)
                    indices_to_affect = list(
                        self.multi_selected_action_indices) if self.multi_selected_action_indices else (
                        [self.selected_action_idx] if self.selected_action_idx != -1 else [])
                    for idx in indices_to_affect:
                        if 0 <= idx < len(actions_list): actions_list[idx]["pos"] = min(100, max(0, actions_list[idx][
                            "pos"] + pos_nudge_delta))
                    fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)

                nudge_time_prev_sc_str = shortcuts.get("nudge_selection_time_prev", "SHIFT+LEFT_ARROW")
                nudge_time_next_sc_str = shortcuts.get("nudge_selection_time_next", "SHIFT+RIGHT_ARROW")
                nudge_time_prev_tuple = self.app._map_shortcut_to_glfw_key(nudge_time_prev_sc_str)
                nudge_time_next_tuple = self.app._map_shortcut_to_glfw_key(nudge_time_next_sc_str)

                time_nudge_delta_ms = 0
                snap_grid_time_ms_for_nudge = app_state.snap_to_grid_time_ms if app_state.snap_to_grid_time_ms > 0 else \
                    (int(1000 / video_fps_for_calc) if video_fps_for_calc > 0 else 20)

                if nudge_time_prev_tuple:
                    key, mods = nudge_time_prev_tuple
                    if (mods['shift'] == io.key_shift and mods['ctrl'] == io.key_ctrl and mods['alt'] == io.key_alt and
                            mods['super'] == io.key_super and imgui.is_key_pressed(key)):
                        time_nudge_delta_ms = -snap_grid_time_ms_for_nudge
                if time_nudge_delta_ms == 0 and nudge_time_next_tuple:
                    key, mods = nudge_time_next_tuple
                    if (mods['shift'] == io.key_shift and mods['ctrl'] == io.key_ctrl and mods['alt'] == io.key_alt and
                            mods['super'] == io.key_super and imgui.is_key_pressed(key)):
                        time_nudge_delta_ms = snap_grid_time_ms_for_nudge

                if time_nudge_delta_ms != 0 and (self.multi_selected_action_indices or self.selected_action_idx != -1):
                    op_desc = f"Nudged Point(s) Time by {time_nudge_delta_ms}ms"
                    fs_proc._record_timeline_action(self.timeline_num, op_desc)
                    objects_to_move = []
                    # Sort indices to process movements consistently, especially for multi-select
                    indices_to_affect = sorted(list(
                        self.multi_selected_action_indices) if self.multi_selected_action_indices else \
                                                   ([self.selected_action_idx] if self.selected_action_idx != -1 else []),
                                               reverse=(time_nudge_delta_ms < 0)
                                               # Process from right to left if nudging left, and vice-versa
                                               )

                    current_selected_actions_data = []  # Store 'at' and original index
                    for idx in indices_to_affect:
                        if 0 <= idx < len(actions_list):
                            objects_to_move.append(actions_list[idx])
                            current_selected_actions_data.append(
                                {'obj': actions_list[idx], 'original_at': actions_list[idx]['at']})

                    for action_data_to_move in current_selected_actions_data:
                        action_obj = action_data_to_move['obj']
                        # Find current index in potentially modified list (if prior nudges in multi-select caused re-sorts, though actions_list isn't sorted mid-loop here)
                        current_idx_in_list = -1
                        try:
                            current_idx_in_list = actions_list.index(action_obj)  # This relies on object identity
                        except ValueError:
                            continue  # Should not happen if objects_to_move came from actions_list

                        new_at = action_obj["at"] + time_nudge_delta_ms

                        prev_at_limit = -1
                        # Find true previous point NOT in the current selection being moved
                        for k in range(current_idx_in_list - 1, -1, -1):
                            if actions_list[k] not in objects_to_move:
                                prev_at_limit = actions_list[k]["at"] + 1
                                break
                        if prev_at_limit == -1: prev_at_limit = 0  # No preceding point not in selection

                        next_at_limit = float('inf')
                        # Find true next point NOT in the current selection being moved
                        for k in range(current_idx_in_list + 1, len(actions_list)):
                            if actions_list[k] not in objects_to_move:
                                next_at_limit = actions_list[k]["at"] - 1
                                break

                        if video_loaded: next_at_limit = min(next_at_limit, effective_total_duration_ms)

                        action_obj["at"] = int(round(np.clip(float(new_at), float(prev_at_limit), float(next_at_limit))))

                    actions_list.sort(key=lambda a: a["at"])

                    # Re-select moved points
                    self.multi_selected_action_indices.clear()
                    new_primary_selected_idx = -1
                    # Find new indices of the objects that were moved
                    for moved_action_data in current_selected_actions_data:
                        try:
                            new_idx = actions_list.index(moved_action_data['obj'])  # Find by object identity after sort
                            self.multi_selected_action_indices.add(new_idx)
                            if new_primary_selected_idx == -1 or new_idx < new_primary_selected_idx:  # Keep smallest index as primary
                                new_primary_selected_idx = new_idx
                        except ValueError:
                            pass  # Should not happen

                    self.selected_action_idx = new_primary_selected_idx if self.multi_selected_action_indices else -1

                    fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)
                    if video_loaded and not self.app.processor.is_processing and self.selected_action_idx != -1 and video_fps_for_calc > 0:
                        target_frame_nudge = int(
                            round((actions_list[self.selected_action_idx]["at"] / 1000.0) * video_fps_for_calc))
                        self.app.processor.seek_video(np.clip(target_frame_nudge, 0,
                                                              self.app.processor.total_frames - 1 if self.app.processor.total_frames > 0 else 0))
                        app_state.force_timeline_pan_to_current_frame = True

                del_sc_str = shortcuts.get("delete_selected_point", "DELETE")
                del_alt_sc_str = shortcuts.get("delete_selected_point_alt", "BACKSPACE")
                del_key_tuple = self.app._map_shortcut_to_glfw_key(del_sc_str)
                bck_key_tuple = self.app._map_shortcut_to_glfw_key(del_alt_sc_str)
                delete_pressed = False
                if del_key_tuple:
                    key, mods = del_key_tuple
                    if (mods['shift'] == io.key_shift and mods['ctrl'] == io.key_ctrl and mods['alt'] == io.key_alt and
                            mods['super'] == io.key_super and imgui.is_key_pressed(key)):
                        delete_pressed = True
                if not delete_pressed and bck_key_tuple:
                    key, mods = bck_key_tuple
                    if (mods['shift'] == io.key_shift and mods['ctrl'] == io.key_ctrl and mods['alt'] == io.key_alt and
                            mods['super'] == io.key_super and imgui.is_key_pressed(key) and \
                            (not del_key_tuple or key != del_key_tuple[0])):
                        delete_pressed = True

                if delete_pressed and (self.multi_selected_action_indices or self.selected_action_idx != -1):
                    op_desc = ""
                    deleted_count = 0
                    indices_to_remove = []
                    if self.multi_selected_action_indices:
                        op_desc = f"Deleted {len(self.multi_selected_action_indices)} Selected Point(s) (Key)"
                        deleted_count = len(self.multi_selected_action_indices)
                        indices_to_remove = list(self.multi_selected_action_indices)
                    elif self.selected_action_idx != -1:
                        op_desc = "Deleted Point (Key)"
                        deleted_count = 1
                        indices_to_remove = [self.selected_action_idx]

                    if deleted_count > 0 and target_funscript_instance_for_render and axis_name_for_render:
                        fs_proc._record_timeline_action(self.timeline_num, op_desc)
                        target_funscript_instance_for_render.clear_points(axis=axis_name_for_render,
                                                                          selected_indices=indices_to_remove)
                        if self.multi_selected_action_indices: self.multi_selected_action_indices.clear()
                        self.selected_action_idx = -1
                        self.dragging_action_idx = -1
                        fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)

                time_at_center_add = x_to_time(center_x_marker)
                snap_time_add_key = app_state.snap_to_grid_time_ms if app_state.snap_to_grid_time_ms > 0 else 1
                snapped_time_add_key = max(0, int(round(time_at_center_add / snap_time_add_key)) * snap_time_add_key)

                for i in range(10):
                    setting_key_name = f"add_point_{i * 10}"
                    bound_key_str_num = shortcuts.get(setting_key_name, str(i))
                    if not bound_key_str_num: continue

                    glfw_key_num_tuple = self.app._map_shortcut_to_glfw_key(bound_key_str_num)
                    if glfw_key_num_tuple:
                        key_code_num, mods_num = glfw_key_num_tuple
                        if (mods_num['shift'] == io.key_shift and mods_num['ctrl'] == io.key_ctrl and \
                                mods_num['alt'] == io.key_alt and mods_num['super'] == io.key_super and \
                                imgui.is_key_pressed(key_code_num)):

                            target_pos_val = i * 10
                            if self.selected_action_idx != -1 and 0 <= self.selected_action_idx < len(actions_list):
                                op_desc = f"Set Point Pos to {target_pos_val} (Key)"
                                fs_proc._record_timeline_action(self.timeline_num, op_desc)
                                actions_list[self.selected_action_idx]["pos"] = target_pos_val
                                fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)
                                self.app.logger.info(
                                    f"Set T{self.timeline_num} point {self.selected_action_idx} to pos {target_pos_val}.",
                                    extra={'status_message': True})
                            elif target_funscript_instance_for_render and axis_name_for_render:
                                op_desc = "Added Point (Key)"
                                fs_proc._record_timeline_action(self.timeline_num, op_desc)
                                target_funscript_instance_for_render.add_action(
                                    timestamp_ms=snapped_time_add_key,
                                    primary_pos=target_pos_val if axis_name_for_render == 'primary' else None,
                                    secondary_pos=target_pos_val if axis_name_for_render == 'secondary' else None,
                                    is_from_live_tracker=False
                                )
                                actions_list = getattr(target_funscript_instance_for_render,
                                                       f"{axis_name_for_render}_actions", [])
                                new_idx = -1
                                for idx_act, act in enumerate(actions_list):
                                    if act['at'] == snapped_time_add_key and act['pos'] == target_pos_val:
                                        new_idx = idx_act
                                        break
                                if new_idx != -1:
                                    self.selected_action_idx = new_idx
                                    self.multi_selected_action_indices = {new_idx}
                                else:
                                    self.selected_action_idx = -1
                                    self.multi_selected_action_indices.clear()
                                fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)
                                if video_loaded and not self.app.processor.is_processing and video_fps_for_calc > 0:
                                    target_frame_key_add = int(round((snapped_time_add_key / 1000.0) * video_fps_for_calc))
                                    self.app.processor.seek_video(np.clip(target_frame_key_add, 0,
                                                                          self.app.processor.total_frames - 1 if self.app.processor.total_frames > 0 else 0))
                                    app_state.force_timeline_pan_to_current_frame = True
                            break
            if imgui.begin_popup(context_popup_id):
                if not allow_editing_timeline: imgui.text_disabled("Editing disabled.")
                imgui.separator()
                imgui.text(f"Timeline {self.timeline_num} - Add @ (snapped):")
                imgui.text(f"Time (ms): {self.new_point_candidate_at}, Pos: {self.new_point_candidate_pos}")
                imgui.separator()

                if allow_editing_timeline:
                    if imgui.menu_item(f"Add Point Here##CTXMenuAdd{window_id_suffix}")[0]:
                        if target_funscript_instance_for_render and axis_name_for_render:
                            op_desc = "Added Point (Menu)"
                            fs_proc._record_timeline_action(self.timeline_num, op_desc)
                            target_funscript_instance_for_render.add_action(
                                timestamp_ms=self.new_point_candidate_at,
                                primary_pos=self.new_point_candidate_pos if axis_name_for_render == 'primary' else None,
                                secondary_pos=self.new_point_candidate_pos if axis_name_for_render == 'secondary' else None,
                                is_from_live_tracker=False)
                            actions_list = getattr(target_funscript_instance_for_render, f"{axis_name_for_render}_actions",
                                                   [])
                            new_idx_ctx = -1
                            for idx_act, act in enumerate(actions_list):
                                if act['at'] == self.new_point_candidate_at and act['pos'] == self.new_point_candidate_pos:
                                    new_idx_ctx = idx_act
                                    break
                            if new_idx_ctx != -1:
                                self.selected_action_idx = new_idx_ctx
                                self.multi_selected_action_indices = {new_idx_ctx}
                                self.app.logger.info(
                                    f"Added point to T{self.timeline_num} at {self.new_point_candidate_at}ms, pos {self.new_point_candidate_pos} (Menu).",
                                    extra={'status_message': True})
                            else:
                                self.selected_action_idx = -1
                                self.multi_selected_action_indices.clear()
                                self.app.logger.warning(
                                    f"Point at {self.new_point_candidate_at}ms may have been merged or not added as expected.")
                            fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)
                        imgui.close_current_popup()
                else:
                    imgui.menu_item(f"Add Point Here##CTXMenuAdd{window_id_suffix}", enabled=False)
                imgui.separator()
                can_copy = allow_editing_timeline and (
                        bool(self.multi_selected_action_indices) or self.selected_action_idx != -1)
                if imgui.menu_item(f"Copy Selected##CTXMenuCopy{window_id_suffix}",
                                   shortcut=shortcuts.get("copy_selection", "Ctrl+C"), enabled=can_copy)[0]:
                    if can_copy: self._handle_copy_selection()
                    imgui.close_current_popup()
                can_paste = allow_editing_timeline and self.app.funscript_processor.clipboard_has_actions()
                if imgui.menu_item(f"Paste at Cursor##CTXMenuPaste{window_id_suffix}",
                                   shortcut=shortcuts.get("paste_selection", "Ctrl+V"), enabled=can_paste)[0]:
                    if can_paste: self._handle_paste_actions(self.new_point_candidate_at)
                    imgui.close_current_popup()
                imgui.separator()

                if imgui.menu_item(f"Cancel##CTXMenuCancel{window_id_suffix}")[0]:
                    imgui.close_current_popup()
                imgui.end_popup()

            if hovered_action_idx_current_timeline != -1 and self.dragging_action_idx == -1 and not imgui.is_popup_open(
                    context_popup_id):
                if 0 <= hovered_action_idx_current_timeline < len(actions_list):
                    action_hovered = actions_list[hovered_action_idx_current_timeline]
                    imgui.begin_tooltip()
                    imgui.text(
                        f"T{self.timeline_num} - Time: {action_hovered['at']} ms ({action_hovered['at'] / 1000.0:.2f}s)")
                    imgui.text(f"Pos: {action_hovered['pos']}")
                    if video_loaded and video_fps_for_calc > 0:
                        frame_of_point = int(round((action_hovered['at'] / 1000.0) * video_fps_for_calc))
                        imgui.text(f"Frame: {frame_of_point}")
                    if hovered_action_idx_current_timeline > 0:
                        p_prev = actions_list[hovered_action_idx_current_timeline - 1]
                        dt_prev = action_hovered['at'] - p_prev['at']
                        dp_prev = abs(action_hovered['pos'] - p_prev['pos'])
                        speed_prev = dp_prev / (dt_prev / 1000.0) if dt_prev > 0 else 0
                        imgui.text(f"In-Speed: {speed_prev:.1f} pos/s")
                    if hovered_action_idx_current_timeline < len(actions_list) - 1:
                        p_next = actions_list[hovered_action_idx_current_timeline + 1]
                        dt_next = p_next['at'] - action_hovered['at']
                        dp_next = abs(p_next['pos'] - action_hovered['pos'])
                        speed_next = dp_next / (dt_next / 1000.0) if dt_next > 0 else 0
                        imgui.text(f"Out-Speed: {speed_next:.1f} pos/s")
                    imgui.end_tooltip()
            imgui.end()
