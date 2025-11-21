import imgui
import os
import numpy as np
import math
import time
import glfw
import copy
from typing import Optional, List, Dict, Tuple, Set
from bisect import bisect_left, bisect_right

# Imports from your application structure
from .plugin_ui_manager import PluginUIManager, PluginUIState
from .plugin_ui_renderer import PluginUIRenderer
from .plugin_preview_renderer import PluginPreviewRenderer
from application.utils import _format_time
from config.element_group_colors import TimelineColors

class TimelineTransformer:
    """
    Handles coordinate transformations between Time/Value space and Screen Pixel space.
    Optimized with vectorization support.
    """
    def __init__(self, pos: Tuple[float, float], size: Tuple[float, float], 
                 pan_ms: float, zoom_ms_px: float):
        self.x_offset = pos[0]
        self.y_offset = pos[1]
        self.width = size[0]
        self.height = size[1]
        self.pan_ms = pan_ms
        self.zoom = max(0.001, zoom_ms_px) # Prevent div by zero
        
        # Calculate visible time range
        self.visible_start_ms = pan_ms
        self.visible_end_ms = pan_ms + (self.width * self.zoom)

    def time_to_x(self, t_ms: float) -> float:
        return self.x_offset + (t_ms - self.pan_ms) / self.zoom

    def val_to_y(self, val: float) -> float:
        # Funscript 0-100 mapping: 0 is usually bottom, 100 is top
        # UI Coords: Y increases downwards. 
        # So Val 100 -> y_offset (top), Val 0 -> y_offset + height (bottom)
        return self.y_offset + self.height * (1.0 - (val / 100.0))

    def x_to_time(self, x: float) -> float:
        return (x - self.x_offset) * self.zoom + self.pan_ms

    def y_to_val(self, y: float) -> int:
        if self.height == 0: return 0
        val = (1.0 - (y - self.y_offset) / self.height) * 100.0
        return max(0, min(100, int(round(val))))

    # Vectorized versions for numpy arrays (Rendering path)
    def vec_time_to_x(self, times: np.ndarray) -> np.ndarray:
        return self.x_offset + (times - self.pan_ms) / self.zoom

    def vec_val_to_y(self, vals: np.ndarray) -> np.ndarray:
        return self.y_offset + self.height * (1.0 - (vals / 100.0))


class InteractiveFunscriptTimeline:
    def __init__(self, app_instance, timeline_num: int):
        self.app = app_instance
        self.timeline_num = timeline_num
        self.logger = getattr(app_instance, 'logger', None)

        # --- Selection & Interaction State ---
        self.selected_action_idx: int = -1
        self.multi_selected_action_indices: Set[int] = set()
        
        self.dragging_action_idx: int = -1
        self.drag_start_pos: Optional[Tuple[float, float]] = None
        self.is_dragging_active: bool = False  # True only after exceeding drag threshold
        self.drag_undo_recorded: bool = False
        
        self.is_marqueeing: bool = False
        self.marquee_start: Optional[Tuple[float, float]] = None
        self.marquee_end: Optional[Tuple[float, float]] = None
        
        self.range_selecting: bool = False
        self.range_start_time: float = 0
        self.range_end_time: float = 0
        
        self.context_menu_target_idx: int = -1
        self.selection_anchor_idx: int = -1 # For Shift+Click range selection logic if needed

        # --- Plugin System Integration ---
        self.plugin_manager = PluginUIManager(logger=self.logger)
        self.plugin_renderer = PluginUIRenderer(self.plugin_manager, logger=self.logger)
        self.plugin_preview_renderer = PluginPreviewRenderer(logger=self.logger)
        
        # Connect components
        self.plugin_manager.preview_renderer = self.plugin_preview_renderer
        self.plugin_renderer.set_timeline_reference(self)
        self.plugin_manager.initialize()

        # --- Visualization State ---
        self.preview_actions: Optional[List[Dict]] = None
        self.is_previewing: bool = False
        self.ultimate_autotune_preview_actions: Optional[List[Dict]] = None
        
        # Settings
        self.shift_frames_amount = 1
        self.show_ultimate_autotune_preview = self.app.app_settings.get(
            f"timeline{self.timeline_num}_show_ultimate_preview", True)
        self._ultimate_preview_dirty = True

    # ==================================================================================
    # CORE DATA HELPERS
    # ==================================================================================

    def _get_target_funscript_details(self) -> Tuple[Optional[object], Optional[str]]:
        """Get the target funscript object and axis for this timeline"""
        if self.app.funscript_processor:
            return self.app.funscript_processor._get_target_funscript_object_and_axis(self.timeline_num)
        return None, None

    def _get_actions(self) -> List[Dict]:
        fs, axis = self._get_target_funscript_details()
        if fs and axis:
            return getattr(fs, f"{axis}_actions", [])
        return []

    def invalidate_cache(self):
        """Forces updates on next frame"""
        self._ultimate_preview_dirty = True

    def invalidate_ultimate_preview(self):
        self._ultimate_preview_dirty = True

    # ==================================================================================
    # MAIN RENDER LOOP
    # ==================================================================================

    def render(self, y_pos: float = 0, height: float = 0, view_mode: str = 'expert'):
        app_state = self.app.app_state_ui
        visibility_attr = f"show_funscript_interactive_timeline{'' if self.timeline_num == 1 else '2'}"
        
        if not getattr(app_state, visibility_attr, False):
            return

        # 1. Window Configuration
        is_floating = app_state.ui_layout_mode == "floating"
        flags = (imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_SCROLL_WITH_MOUSE)
        
        if not is_floating:
            # Fixed Layout
            if height <= 0: return
            imgui.set_next_window_position(0, y_pos)
            imgui.set_next_window_size(app_state.window_width, height)
            flags |= (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)
            if not imgui.begin(f"##TimelineFixed{self.timeline_num}", True, flags):
                imgui.end()
                return
        else:
            # Floating Window
            imgui.set_next_window_size(app_state.window_width, 180, condition=imgui.APPEARING)
            is_open, visible = imgui.begin(f"Interactive Timeline {self.timeline_num}", True, flags)
            setattr(app_state, visibility_attr, visible)
            if not is_open:
                imgui.end()
                return

        # 2. Render Toolbar (Buttons)
        self._render_toolbar(view_mode)

        # 3. Prepare Canvas
        draw_list = imgui.get_window_draw_list()
        canvas_pos = imgui.get_cursor_screen_pos()
        canvas_size = imgui.get_content_region_available()
        
        if canvas_size[0] < 1 or canvas_size[1] < 1:
            imgui.end()
            return

        # 4. Setup Coordinate Transformer
        zoom = getattr(app_state, 'timeline_zoom_factor_ms_per_px', 1.0)
        pan = getattr(app_state, 'timeline_pan_offset_ms', 0.0)
        tf = TimelineTransformer(canvas_pos, canvas_size, pan, zoom)

        # 5. Handle User Input (Mouse & Keyboard)
        self._handle_input(app_state, tf)

        # 6. Render Visual Layers
        self._draw_background_grid(draw_list, tf)
        self._draw_audio_waveform(draw_list, tf)
        
        # Data Layers
        main_actions = self._get_actions()
        
        # 6a. Update & Draw Ultimate Preview (if enabled)
        self._update_ultimate_autotune_preview()
        if self.ultimate_autotune_preview_actions:
             self._draw_curve(draw_list, tf, self.ultimate_autotune_preview_actions, 
                              color_override=TimelineColors.ULTIMATE_AUTOTUNE_PREVIEW, 
                              force_lines_only=True, alpha=0.7)

        # 6b. Draw Active Plugin Preview (if any)
        if self.is_previewing and self.preview_actions:
             self._draw_curve(draw_list, tf, self.preview_actions, is_preview=True)

        # 6c. Draw Main Script
        self._draw_curve(draw_list, tf, main_actions, is_preview=False)

        # 6d. Plugin Overlay Renderers (New System)
        if self.plugin_preview_renderer:
            self.plugin_preview_renderer.render_preview_overlay(
                draw_list, canvas_pos[0], canvas_pos[1], canvas_size[0], canvas_size[1],
                int(tf.visible_start_ms), int(tf.visible_end_ms), None
            )

        # 6e. UI Overlays (Selection Box, Playhead, Text)
        self._draw_ui_overlays(draw_list, tf)
        
        # 7. Render Plugin Windows (Popups)
        self.plugin_renderer.render_plugin_windows(self.timeline_num, f"TL{self.timeline_num}")

        # 7b. Check for and execute pending plugin apply requests
        self._check_and_apply_pending_plugins()

        # 8. Handle Auto-Scroll/Sync
        self._handle_sync_logic(app_state, tf)

        imgui.end()

    # ==================================================================================
    # INPUT HANDLING
    # ==================================================================================

    def _handle_input(self, app_state, tf: TimelineTransformer):
        io = imgui.get_io()
        mouse_pos = imgui.get_mouse_pos()
        
        # Check bounds
        is_hovered = (tf.x_offset <= mouse_pos[0] <= tf.x_offset + tf.width and 
                      tf.y_offset <= mouse_pos[1] <= tf.y_offset + tf.height)
        is_focused = imgui.is_window_focused(imgui.FOCUS_ROOT_AND_CHILD_WINDOWS)

        # --- Keyboard Shortcuts (Global / Focused) ---
        if is_focused:
            self._handle_keyboard_shortcuts(app_state, io)

        # --- Navigation (Zoom/Pan) ---
        if is_hovered:
            # Wheel Zoom
            if io.mouse_wheel != 0:
                scale = 0.85 if io.mouse_wheel > 0 else 1.15
                center_ms = tf.x_to_time(mouse_pos[0]) # Zoom towards mouse cursor
                
                new_zoom = max(0.01, min(2000.0, tf.zoom * scale))
                # Adjust pan to keep mouse point stationary
                mouse_offset_px = mouse_pos[0] - tf.x_offset
                new_pan = center_ms - (mouse_offset_px * new_zoom)
                
                app_state.timeline_zoom_factor_ms_per_px = new_zoom
                app_state.timeline_pan_offset_ms = new_pan
                app_state.timeline_interaction_active = True

            # Middle Drag Pan
            if imgui.is_mouse_dragging(glfw.MOUSE_BUTTON_MIDDLE):
                delta_x = io.mouse_delta[0]
                app_state.timeline_pan_offset_ms -= delta_x * tf.zoom
                app_state.timeline_interaction_active = True

        # --- Action Interaction ---
        actions = self._get_actions()
        
        # Left Click
        if is_hovered and imgui.is_mouse_clicked(glfw.MOUSE_BUTTON_LEFT):
            hit_idx = self._hit_test_point(mouse_pos, actions, tf)
            
            if io.key_alt:
                # Alt + Drag = Range Select
                self.range_selecting = True
                self.range_start_time = tf.x_to_time(mouse_pos[0])
                self.range_end_time = self.range_start_time
                if not io.key_ctrl: self.multi_selected_action_indices.clear()
            
            elif hit_idx != -1:
                # Point Clicked
                self.dragging_action_idx = hit_idx
                self.drag_start_pos = mouse_pos
                self.is_dragging_active = False # Wait for drag threshold
                self.drag_undo_recorded = False
                
                # Selection Logic
                if not io.key_ctrl:
                    # If clicking an unselected point, clear others. 
                    # If clicking a selected point, keep selection (might be starting a multi-drag)
                    if hit_idx not in self.multi_selected_action_indices:
                        self.multi_selected_action_indices.clear()
                        self.multi_selected_action_indices.add(hit_idx)
                else:
                    # Toggle selection
                    if hit_idx in self.multi_selected_action_indices:
                        self.multi_selected_action_indices.remove(hit_idx)
                    else:
                        self.multi_selected_action_indices.add(hit_idx)
                
                self.selected_action_idx = hit_idx
                self._seek_video(actions[hit_idx]['at']) # Jump video to point

            else:
                # Empty Space Click -> Marquee or Deselect
                if not io.key_ctrl:
                    self.multi_selected_action_indices.clear()
                    self.selected_action_idx = -1
                
                self.is_marqueeing = True
                self.marquee_start = mouse_pos
                self.marquee_end = mouse_pos

        # --- Dragging Processing ---
        if imgui.is_mouse_dragging(glfw.MOUSE_BUTTON_LEFT):
            
            if self.dragging_action_idx != -1:
                # Threshold check (prevent jitter on simple clicks)
                if not self.is_dragging_active:
                    dist = math.hypot(mouse_pos[0] - self.drag_start_pos[0], mouse_pos[1] - self.drag_start_pos[1])
                    if dist > 5: self.is_dragging_active = True
                
                if self.is_dragging_active:
                    app_state.timeline_interaction_active = True
                    self._update_drag(mouse_pos, tf)
            
            elif self.is_marqueeing:
                self.marquee_end = mouse_pos
                app_state.timeline_interaction_active = True
                
            elif self.range_selecting:
                self.range_end_time = tf.x_to_time(mouse_pos[0])
                app_state.timeline_interaction_active = True

        # --- Mouse Release ---
        if imgui.is_mouse_released(glfw.MOUSE_BUTTON_LEFT):
            if self.is_marqueeing:
                self._finalize_marquee(tf, actions, io.key_ctrl)
            elif self.range_selecting:
                self._finalize_range_select(actions, io.key_ctrl)
            elif self.is_dragging_active:
                self._finalize_drag()

            # Reset States
            self.is_marqueeing = False
            self.range_selecting = False
            self.dragging_action_idx = -1
            self.is_dragging_active = False

            # Clear interaction flag to allow auto-scroll to resume
            app_state.timeline_interaction_active = False

        # Also clear interaction flag when middle mouse is released (after panning)
        if imgui.is_mouse_released(glfw.MOUSE_BUTTON_MIDDLE):
            app_state.timeline_interaction_active = False

        # --- Context Menu ---
        if is_hovered and imgui.is_mouse_clicked(glfw.MOUSE_BUTTON_RIGHT):
            hit_idx = self._hit_test_point(mouse_pos, actions, tf)
            self.context_menu_target_idx = hit_idx
            
            # Auto-select target if not already selected
            if hit_idx != -1 and hit_idx not in self.multi_selected_action_indices:
                self.multi_selected_action_indices = {hit_idx}
                self.selected_action_idx = hit_idx
            
            # Store coords for "Add Point Here"
            self.new_point_candidate = (tf.x_to_time(mouse_pos[0]), tf.y_to_val(mouse_pos[1]))
            imgui.open_popup(f"TimelineContext{self.timeline_num}")

        self._render_context_menu(tf)

    def _handle_keyboard_shortcuts(self, app_state, io):
        shortcuts = self.app.app_settings.get("funscript_editor_shortcuts", {})
        
        # Helper to map shortcuts (for single-press actions)
        def check_shortcut(name, default):
            key_str = shortcuts.get(name, default)
            tuple_key = self.app._map_shortcut_to_glfw_key(key_str)
            if not tuple_key: return False
            key_code, mods = tuple_key

            pressed = imgui.is_key_pressed(key_code)
            # Check modifiers
            match = (mods["ctrl"] == io.key_ctrl and
                     mods["alt"] == io.key_alt and
                     mods["shift"] == io.key_shift)
            return pressed and match

        # Helper for persistent/held key actions (like panning)
        def check_key_held(name, default):
            key_str = shortcuts.get(name, default)
            tuple_key = self.app._map_shortcut_to_glfw_key(key_str)
            if not tuple_key: return False
            key_code, mods = tuple_key

            held = imgui.is_key_down(key_code)
            # Check modifiers
            match = (mods["ctrl"] == io.key_ctrl and
                     mods["alt"] == io.key_alt and
                     mods["shift"] == io.key_shift)
            return held and match

        # 1. Pan Left/Right (Arrow keys) - persistent while held
        pan_speed = self.app.app_settings.get("timeline_pan_speed_multiplier", 5) * app_state.timeline_zoom_factor_ms_per_px
        if check_key_held("pan_timeline_left", "ALT+LEFT_ARROW"):
            app_state.timeline_pan_offset_ms -= pan_speed
        if check_key_held("pan_timeline_right", "ALT+RIGHT_ARROW"):
            app_state.timeline_pan_offset_ms += pan_speed

        # 2. Select All (Ctrl+A)
        if check_shortcut("select_all_points", "CTRL+A"):
            actions = self._get_actions()
            self.multi_selected_action_indices = set(range(len(actions)))

        # 3. Delete (Delete/Backspace)
        if check_shortcut("delete_selected_point", "DELETE") or check_shortcut("delete_selected_point_alt", "BACKSPACE"):
            self._delete_selected()

        # 4. Copy/Paste
        if check_shortcut("copy_selection", "CTRL+C"):
            self._handle_copy_selection()
        if check_shortcut("paste_selection", "CTRL+V"):
            # Paste at current mouse location or center if mouse not hovered
            t = getattr(self, 'new_point_candidate', (app_state.timeline_pan_offset_ms, 50))[0]
            self._handle_paste_actions(t)

        # 5. Nudge Selection (Arrows)
        nudge_val = 0
        if check_shortcut("nudge_selection_pos_up", "UP_ARROW"): nudge_val = 1
        if check_shortcut("nudge_selection_pos_down", "DOWN_ARROW"): nudge_val = -1
        
        if nudge_val != 0 and self.multi_selected_action_indices:
            self._nudge_selection_value(nudge_val)

        # 6. Nudge Time (Shift+Arrows)
        nudge_t = 0
        snap_t = app_state.snap_to_grid_time_ms if app_state.snap_to_grid_time_ms > 0 else 20
        if check_shortcut("nudge_selection_time_prev", "SHIFT+LEFT_ARROW"): nudge_t = -snap_t
        if check_shortcut("nudge_selection_time_next", "SHIFT+RIGHT_ARROW"): nudge_t = snap_t

        if nudge_t != 0 and self.multi_selected_action_indices:
            self._nudge_selection_time(nudge_t)

    def _hit_test_point(self, mouse_pos, actions, tf: TimelineTransformer) -> int:
        """Optimized hit testing using binary search."""
        if not actions: return -1
        
        tol_px = 8.0 # Pixel radius tolerance
        tol_ms = tol_px * tf.zoom
        
        t_mouse = tf.x_to_time(mouse_pos[0])
        
        # Only search points near the mouse timestamp
        start_idx = bisect_left([a['at'] for a in actions], t_mouse - tol_ms)
        end_idx = bisect_right([a['at'] for a in actions], t_mouse + tol_ms)
        
        best_dist = float('inf')
        best_idx = -1
        
        for i in range(start_idx, end_idx):
            if i >= len(actions): break
            act = actions[i]
            px = tf.time_to_x(act['at'])
            py = tf.val_to_y(act['pos'])
            
            dist = (px - mouse_pos[0])**2 + (py - mouse_pos[1])**2
            if dist < tol_px**2 and dist < best_dist:
                best_dist = dist
                best_idx = i
                
        return best_idx

    # ==================================================================================
    # LOGIC: DRAG / MODIFY / CLIPBOARD
    # ==================================================================================

    def _update_drag(self, mouse_pos, tf: TimelineTransformer):
        actions = self._get_actions()
        if self.dragging_action_idx < 0 or self.dragging_action_idx >= len(actions): return
        
        # Record Undo State (Once per drag)
        if not self.drag_undo_recorded:
            self.app.funscript_processor._record_timeline_action(self.timeline_num, "Drag Point")
            self.drag_undo_recorded = True

        # Calculate New Values
        t_raw = tf.x_to_time(mouse_pos[0])
        v_raw = tf.y_to_val(mouse_pos[1])
        
        # Snapping
        snap_t = self.app.app_state_ui.snap_to_grid_time_ms
        snap_v = self.app.app_state_ui.snap_to_grid_pos
        if snap_t > 0: t_raw = round(t_raw / snap_t) * snap_t
        if snap_v > 0: v_raw = round(v_raw / snap_v) * snap_v
        
        # Constraints: Cannot drag past neighbors
        idx = self.dragging_action_idx
        prev_limit = actions[idx - 1]['at'] + 1 if idx > 0 else 0
        next_limit = actions[idx + 1]['at'] - 1 if idx < len(actions) - 1 else float('inf')
        
        new_t = int(max(prev_limit, min(next_limit, t_raw)))
        new_v = int(max(0, min(100, v_raw)))
        
        # Apply
        actions[idx]['at'] = new_t
        actions[idx]['pos'] = new_v
        
        # Update state
        self.invalidate_cache()
        self.app.project_manager.project_dirty = True

    def _finalize_drag(self):
        if self.drag_undo_recorded:
             self.app.funscript_processor._finalize_action_and_update_ui(self.timeline_num, "Drag Point")

    def _finalize_marquee(self, tf, actions, append: bool):
        if not self.marquee_start or not self.marquee_end: return

        # Check if this was a simple click (not a drag)
        dx = abs(self.marquee_end[0] - self.marquee_start[0])
        dy = abs(self.marquee_end[1] - self.marquee_start[1])
        is_simple_click = (dx < 5 and dy < 5)  # Threshold: less than 5 pixels = click

        if is_simple_click:
            # Single click on empty space -> seek video to clicked time
            click_time = tf.x_to_time(self.marquee_start[0])
            self._seek_video(click_time)
            return

        # Get marquee rect
        x1, x2 = sorted([self.marquee_start[0], self.marquee_end[0]])
        y1, y2 = sorted([self.marquee_start[1], self.marquee_end[1]])

        t_start = tf.x_to_time(x1)
        t_end = tf.x_to_time(x2)

        # Optimize: Binary search time bounds
        s_idx = bisect_left([a['at'] for a in actions], t_start)
        e_idx = bisect_right([a['at'] for a in actions], t_end)

        new_selection = set()
        for i in range(s_idx, e_idx):
            act = actions[i]
            py = tf.val_to_y(act['pos'])
            if y1 <= py <= y2:
                new_selection.add(i)

        if append:
            self.multi_selected_action_indices.update(new_selection)
        else:
            self.multi_selected_action_indices = new_selection

    def _finalize_range_select(self, actions, append: bool):
        t1, t2 = sorted([self.range_start_time, self.range_end_time])
        
        s_idx = bisect_left([a['at'] for a in actions], t1)
        e_idx = bisect_right([a['at'] for a in actions], t2)
        
        new_set = set(range(s_idx, e_idx))
        if append:
            self.multi_selected_action_indices.update(new_set)
        else:
            self.multi_selected_action_indices = new_set

    def _seek_video(self, time_ms: float):
        if self.app.processor and self.app.processor.video_info:
            fps = self.app.processor.fps
            if fps > 0:
                frame = int(round((time_ms / 1000.0) * fps))
                self.app.processor.seek_video(frame)
                self.app.app_state_ui.force_timeline_pan_to_current_frame = True

    # --- Nudge Helpers ---
    def _nudge_selection_value(self, delta: int):
        actions = self._get_actions()
        if not actions: return
        
        snap = self.app.app_state_ui.snap_to_grid_pos
        actual_delta = delta * (snap if snap > 0 else 1)
        
        self.app.funscript_processor._record_timeline_action(self.timeline_num, "Nudge Value")
        for idx in self.multi_selected_action_indices:
            if idx < len(actions):
                actions[idx]['pos'] = max(0, min(100, actions[idx]['pos'] + actual_delta))
        self.app.funscript_processor._finalize_action_and_update_ui(self.timeline_num, "Nudge Value")
        self.invalidate_cache()

    def _nudge_selection_time(self, delta_ms: int):
        actions = self._get_actions()
        if not actions: return

        self.app.funscript_processor._record_timeline_action(self.timeline_num, "Nudge Time")

        # Sort indices to avoid collision logic issues
        indices = sorted(list(self.multi_selected_action_indices), reverse=(delta_ms > 0))

        for idx in indices:
            if idx < len(actions):
                # Logic similar to drag constraint
                prev_limit = actions[idx - 1]['at'] + 1 if idx > 0 else 0
                next_limit = actions[idx + 1]['at'] - 1 if idx < len(actions) - 1 else float('inf')

                new_at = actions[idx]['at'] + delta_ms
                actions[idx]['at'] = int(max(prev_limit, min(next_limit, new_at)))

        self.app.funscript_processor._finalize_action_and_update_ui(self.timeline_num, "Nudge Time")
        self.invalidate_cache()

    def _nudge_all_time(self, frames: int):
        """Nudge ALL points by a number of frames (not just selection)"""
        actions = self._get_actions()
        if not actions: return

        processor = self.app.processor
        if not processor or not processor.video_info: return

        fps = processor.fps
        if fps <= 0: return

        # Convert frames to milliseconds
        delta_ms = int((frames / fps) * 1000.0)

        self.app.funscript_processor._record_timeline_action(self.timeline_num, "Nudge All Points")

        # Nudge all points by the same amount
        for action in actions:
            action['at'] = max(0, action['at'] + delta_ms)

        self.app.funscript_processor._finalize_action_and_update_ui(self.timeline_num, "Nudge All Points")
        self.invalidate_cache()

    # --- Clipboard & Timeline Ops ---
    def _handle_copy_selection(self):
        actions = self._get_actions()
        if not self.multi_selected_action_indices: return
        
        indices = sorted(list(self.multi_selected_action_indices))
        selection = [actions[i] for i in indices]
        
        if not selection: return
        
        # Normalize to relative time (0 start)
        base_time = selection[0]['at']
        clipboard_data = [{'relative_at': a['at'] - base_time, 'pos': a['pos']} for a in selection]
        
        self.app.funscript_processor.set_clipboard_actions(clipboard_data)
        self.logger.info(f"Copied {len(clipboard_data)} points.")

    def _handle_paste_actions(self, paste_at_ms: float):
        clip = self.app.funscript_processor.get_clipboard_actions()
        if not clip: return

        fs, axis = self._get_target_funscript_details()
        if not fs: return
        
        self.app.funscript_processor._record_timeline_action(self.timeline_num, "Paste")
        
        new_actions = []
        for item in clip:
            t = int(paste_at_ms + item['relative_at'])
            v = int(item['pos'])
            new_actions.append({
                'timestamp_ms': t,
                'primary_pos': v if axis=='primary' else None,
                'secondary_pos': v if axis=='secondary' else None
            })
            
        fs.add_actions_batch(new_actions, is_from_live_tracker=False)
        self.app.funscript_processor._finalize_action_and_update_ui(self.timeline_num, "Paste")
        self.invalidate_cache()

    def _handle_swap_timeline(self):
        other_num = 2 if self.timeline_num == 1 else 1
        self.app.funscript_processor.swap_timelines(self.timeline_num, other_num)

    def _handle_copy_to_other(self):
        actions = self._get_actions()
        if not self.multi_selected_action_indices: return
        
        other_num = 2 if self.timeline_num == 1 else 1
        fs_other, axis_other = self.app.funscript_processor._get_target_funscript_object_and_axis(other_num)
        
        if not fs_other: return
        
        indices = sorted(list(self.multi_selected_action_indices))
        points_to_copy = [actions[i] for i in indices]
        
        # Convert format
        batch = []
        for p in points_to_copy:
            batch.append({
                'timestamp_ms': p['at'],
                'primary_pos': p['pos'] if axis_other == 'primary' else None,
                'secondary_pos': p['pos'] if axis_other == 'secondary' else None
            })
            
        self.app.funscript_processor._record_timeline_action(other_num, f"Copy from T{self.timeline_num}")
        fs_other.add_actions_batch(batch, is_from_live_tracker=False)
        self.app.funscript_processor._finalize_action_and_update_ui(other_num, f"Copy from T{self.timeline_num}")

    # --- Selection Filters ---
    def _filter_selection(self, mode: str):
        """Filter selection: 'top', 'bottom', 'mid'"""
        actions = self._get_actions()
        if len(self.multi_selected_action_indices) < 3: return
        
        indices = sorted(list(self.multi_selected_action_indices))
        subset = [actions[i] for i in indices]
        
        # Simple peak detection logic within selection
        keep_indices = set()
        
        for k, idx in enumerate(indices):
            current = actions[idx]['pos']
            # Check neighbors within the selection list, not global list
            prev_val = subset[k-1]['pos'] if k > 0 else -1
            next_val = subset[k+1]['pos'] if k < len(subset)-1 else -1
            
            is_peak = (current > prev_val) and (current >= next_val)
            is_valley = (current < prev_val) and (current <= next_val)
            
            if mode == 'top' and is_peak: keep_indices.add(idx)
            elif mode == 'bottom' and is_valley: keep_indices.add(idx)
            elif mode == 'mid' and not is_peak and not is_valley: keep_indices.add(idx)
            
        self.multi_selected_action_indices = keep_indices

    # ==================================================================================
    # VISUAL DRAWING
    # ==================================================================================

    def _draw_background_grid(self, dl, tf: TimelineTransformer):
        # 1. Background
        dl.add_rect_filled(tf.x_offset, tf.y_offset, tf.x_offset + tf.width, tf.y_offset + tf.height, 
                           imgui.get_color_u32_rgba(*TimelineColors.CANVAS_BACKGROUND))
        
        # 2. Horizontal Lines (0, 25, 50, 75, 100)
        for val in [0, 25, 50, 75, 100]:
            y = tf.val_to_y(val)
            col = TimelineColors.GRID_MAJOR_LINES if val == 50 else TimelineColors.GRID_LINES
            thick = 1.5 if val == 50 else 1.0
            dl.add_line(tf.x_offset, y, tf.x_offset + tf.width, y, imgui.get_color_u32_rgba(*col), thick)

            # Position labels
            label_text = str(val)
            text_size = imgui.calc_text_size(label_text)

            if val == 100 or val == 75:
                # Place below the line
                label_y = y + 2
            elif val == 50:
                # Center on the middle bar with background
                label_y = y - text_size[1] / 2
                # Draw background rectangle for readability
                padding = 2
                dl.add_rect_filled(
                    tf.x_offset + 2 - padding,
                    label_y - padding,
                    tf.x_offset + 2 + text_size[0] + padding,
                    label_y + text_size[1] + padding,
                    imgui.get_color_u32_rgba(*TimelineColors.CANVAS_BACKGROUND)
                )
            else:
                # 0 and 25: above the line
                label_y = y - 12

            dl.add_text(tf.x_offset + 2, label_y, imgui.get_color_u32_rgba(*TimelineColors.GRID_LABELS), label_text)

        # 3. Vertical Lines (Adaptive Time Steps)
        pixels_per_sec = 1000.0 / tf.zoom
        # Determine grid interval based on visual density
        if pixels_per_sec > 200: step_ms = 100
        elif pixels_per_sec > 50: step_ms = 1000
        elif pixels_per_sec > 10: step_ms = 5000
        else: step_ms = 30000

        # Snap start time to step
        start_ms = (tf.visible_start_ms // step_ms) * step_ms
        curr_ms = start_ms
        
        while curr_ms <= tf.visible_end_ms:
            x = tf.time_to_x(curr_ms)
            if x >= tf.x_offset:
                is_major = (curr_ms % (step_ms * 5) == 0)
                col = TimelineColors.GRID_MAJOR_LINES if is_major else TimelineColors.GRID_LINES
                dl.add_line(x, tf.y_offset, x, tf.y_offset + tf.height, imgui.get_color_u32_rgba(*col))
                # Only show time labels for non-negative times
                if is_major and curr_ms >= 0:
                     dl.add_text(x + 3, tf.y_offset + tf.height - 15, imgui.get_color_u32_rgba(*TimelineColors.GRID_LABELS), f"{curr_ms/1000:.1f}s")
            curr_ms += step_ms

    def _draw_audio_waveform(self, dl, tf: TimelineTransformer):
        if not self.app.app_state_ui.show_audio_waveform or self.app.audio_waveform_data is None: return
        
        data = self.app.audio_waveform_data
        total_frames = self.app.processor.total_frames
        fps = self.app.processor.fps
        if total_frames <= 0 or fps <= 0: return
        
        duration_ms = (total_frames / fps) * 1000.0
        
        # Map visible range to data indices
        idx_start = int((tf.visible_start_ms / duration_ms) * len(data))
        idx_end = int((tf.visible_end_ms / duration_ms) * len(data))
        
        idx_start = max(0, idx_start)
        idx_end = min(len(data), idx_end)
        
        if idx_end <= idx_start: return

        # Decimate for performance (Max 1 sample per pixel)
        step = max(1, (idx_end - idx_start) // int(tf.width))
        subset = data[idx_start:idx_end:step]
        
        # Coordinates
        times = np.linspace(tf.visible_start_ms, tf.visible_end_ms, len(subset))
        xs = tf.vec_time_to_x(times)
        
        center_y = tf.y_offset + tf.height / 2
        # Scaling amplitude to timeline height
        ys_top = center_y - (subset * tf.height / 2)
        ys_bot = center_y + (subset * tf.height / 2)
        
        col = imgui.get_color_u32_rgba(*TimelineColors.AUDIO_WAVEFORM)
        
        # LOD: Lines vs Polylines
        if step > 10:
            for i in range(len(xs)):
                dl.add_line(xs[i], ys_top[i], xs[i], ys_bot[i], col)
        else:
            pts_top = list(zip(xs, ys_top))
            pts_bot = list(zip(xs, ys_bot))
            dl.add_polyline(pts_top, col, False, 1.0)
            dl.add_polyline(pts_bot, col, False, 1.0)

    def _draw_curve(self, dl, tf: TimelineTransformer, actions: List[Dict], 
                    is_preview=False, color_override=None, force_lines_only=False, alpha=1.0):
        if not actions or len(actions) < 2: return

        # 1. Culling: Identify visible slice
        margin_ms = tf.zoom * 100 
        s_idx = bisect_left([a['at'] for a in actions], tf.visible_start_ms - margin_ms)
        e_idx = bisect_right([a['at'] for a in actions], tf.visible_end_ms + margin_ms)
        
        s_idx = max(0, s_idx - 1)
        e_idx = min(len(actions), e_idx + 1)
        
        if e_idx - s_idx < 2: return

        visible_actions = actions[s_idx:e_idx]
        
        # 2. Vectorized Transform
        ats = np.array([a['at'] for a in visible_actions], dtype=np.float32)
        poss = np.array([a['pos'] for a in visible_actions], dtype=np.float32)
        
        xs = tf.vec_time_to_x(ats)
        ys = tf.vec_val_to_y(poss)
        
        # 3. LOD Decision
        points_on_screen = len(xs)
        pixels_per_point = tf.width / points_on_screen if points_on_screen > 0 else 0
        
        # -- LOD A: Density Envelope (Massive Zoom Out) --
        if pixels_per_point < 2 and not is_preview and len(visible_actions) > 2000:
            # Optimization: Draw simple vertical bars representing min/max in horizontal chunks
            col = color_override or TimelineColors.AUDIO_WAVEFORM # Reuse waveform color for density
            col_u32 = imgui.get_color_u32_rgba(col[0], col[1], col[2], 0.5 * alpha)
            
            # Draw simplified polyline for shape
            pts = list(zip(xs, ys))
            dl.add_polyline(pts, col_u32, False, 1.0)
            return

        # -- LOD B: Lines Only --
        base_col = color_override or (TimelineColors.PREVIEW_LINES if is_preview else (0.8, 0.8, 0.8, 1.0))
        col_u32 = imgui.get_color_u32_rgba(base_col[0], base_col[1], base_col[2], base_col[3] * alpha)
        thick = 1.5 if is_preview else 2.0
        
        pts = list(zip(xs, ys))
        dl.add_polyline(pts, col_u32, False, thick)

        # -- LOD C: Points (Zoomed In) --
        # Draw points if space permits OR if they are selected/dragged (always draw interactive points)
        should_draw_points = (pixels_per_point > 5) or (not force_lines_only)
        
        if should_draw_points and not force_lines_only:
            radius = self.app.app_state_ui.timeline_point_radius
            
            for i in range(len(visible_actions)):
                real_idx = s_idx + i
                
                # Check interaction state
                is_sel = real_idx in self.multi_selected_action_indices
                is_drag = (real_idx == self.dragging_action_idx)
                is_interactive = is_sel or is_drag
                
                # Skip drawing non-selected points if zoomed out too far
                if not is_interactive and pixels_per_point < 5:
                    continue

                px, py = xs[i], ys[i]
                
                # Colors
                if is_drag:
                    c_tuple = TimelineColors.POINT_DRAGGING
                    r = radius + 2
                elif is_sel:
                    c_tuple = TimelineColors.POINT_SELECTED
                    r = radius + 1
                else:
                    c_tuple = TimelineColors.POINT_DEFAULT if not is_preview else TimelineColors.PREVIEW_POINTS
                    r = radius

                dl.add_circle_filled(px, py, r, imgui.get_color_u32_rgba(c_tuple[0], c_tuple[1], c_tuple[2], c_tuple[3] * alpha))
                
                if is_sel:
                    dl.add_circle(px, py, r+1, imgui.get_color_u32_rgba(*TimelineColors.SELECTED_POINT_BORDER))

    def _draw_ui_overlays(self, dl, tf: TimelineTransformer):
        # 1. Playhead (Center)
        center_x = tf.x_offset + (tf.width / 2)
        dl.add_line(center_x, tf.y_offset, center_x, tf.y_offset + tf.height, 
                    imgui.get_color_u32_rgba(*TimelineColors.CENTER_MARKER), 2.0)
        
        # Playhead Time Info
        time_ms = tf.x_to_time(center_x)
        txt = _format_time(self.app, time_ms/1000.0)
        dl.add_text(center_x + 6, tf.y_offset + 6, imgui.get_color_u32_rgba(*TimelineColors.TIME_DISPLAY_TEXT), txt)
        
        # 2. Marquee Box
        if self.is_marqueeing and self.marquee_start and self.marquee_end:
            p1 = self.marquee_start
            p2 = self.marquee_end
            x_min, x_max = min(p1[0], p2[0]), max(p1[0], p2[0])
            y_min, y_max = min(p1[1], p2[1]), max(p1[1], p2[1])
            
            dl.add_rect_filled(x_min, y_min, x_max, y_max, imgui.get_color_u32_rgba(*TimelineColors.MARQUEE_SELECTION_FILL))
            dl.add_rect(x_min, y_min, x_max, y_max, imgui.get_color_u32_rgba(*TimelineColors.MARQUEE_SELECTION_BORDER))

        # 3. Range Selection Highlight
        if self.range_selecting:
            t1, t2 = sorted([self.range_start_time, self.range_end_time])
            x1 = tf.time_to_x(t1)
            x2 = tf.time_to_x(t2)
            dl.add_rect_filled(x1, tf.y_offset, x2, tf.y_offset + tf.height, imgui.get_color_u32_rgba(0.0, 0.7, 1.0, 0.2))
            dl.add_line(x1, tf.y_offset, x1, tf.y_offset+tf.height, imgui.get_color_u32_rgba(0.0, 0.7, 1.0, 0.5))
            dl.add_line(x2, tf.y_offset, x2, tf.y_offset+tf.height, imgui.get_color_u32_rgba(0.0, 0.7, 1.0, 0.5))

    # ==================================================================================
    # TOOLBAR & MENUS
    # ==================================================================================

    def _render_toolbar(self, view_mode):
        if view_mode != 'expert': return
        
        # Standard Buttons - Clear selected points
        num_selected = len(self.multi_selected_action_indices) if self.multi_selected_action_indices else 0
        clear_label = f"Clear ({num_selected})##{self.timeline_num}" if num_selected > 0 else f"Clear##{self.timeline_num}"

        if imgui.button(clear_label):
            self._delete_selected()

        if imgui.is_item_hovered():
            tooltip = f"Delete {num_selected} selected points" if num_selected > 0 else "Delete selected points (none selected)"
            imgui.set_tooltip(tooltip)

        imgui.same_line()

        # Clear All button
        if imgui.button(f"Clear All##{self.timeline_num}"):
            self._clear_all_points()
        if imgui.is_item_hovered(): imgui.set_tooltip("Delete ALL points on this timeline (Ctrl+Z to undo)")
        
        imgui.same_line()
        
        # Plugin System Buttons
        self.plugin_renderer.render_plugin_buttons(self.timeline_num, view_mode)
        
        imgui.same_line()
        imgui.text("|")
        imgui.same_line()
        
        # Nudge All Buttons
        if imgui.button(f"<<##{self.timeline_num}"):
            self._nudge_all_time(-1)
        if imgui.is_item_hovered(): imgui.set_tooltip("Nudge all points left by 1 frame")
        imgui.same_line()

        if imgui.button(f">>##{self.timeline_num}"):
            self._nudge_all_time(1)
        if imgui.is_item_hovered(): imgui.set_tooltip("Nudge all points right by 1 frame")
        imgui.same_line()

        imgui.text("|")
        imgui.same_line()

        # View Controls
        if imgui.button(f"+##ZIn{self.timeline_num}"):
            self.app.app_state_ui.timeline_zoom_factor_ms_per_px *= 0.8
        imgui.same_line()
        if imgui.button(f"-##ZOut{self.timeline_num}"):
             self.app.app_state_ui.timeline_zoom_factor_ms_per_px *= 1.2

        imgui.same_line()
        changed, self.show_ultimate_autotune_preview = imgui.checkbox("Ult. Preview", self.show_ultimate_autotune_preview)
        if changed:
            self.app.app_settings.set(f"timeline{self.timeline_num}_show_ultimate_preview", self.show_ultimate_autotune_preview)
            self.invalidate_ultimate_preview()

        # Timeline Status Text
        imgui.same_line()
        imgui.text("|")
        imgui.same_line()
        status_text = self._get_timeline_status_text()
        imgui.text(status_text)

    def _render_context_menu(self, tf):
        if imgui.begin_popup(f"TimelineContext{self.timeline_num}"):
            # Add Point
            if imgui.menu_item("Add Point Here")[0]:
                t, v = getattr(self, 'new_point_candidate', (0, 0))
                self._add_point(t, v)
                imgui.close_current_popup()
            
            imgui.separator()
            
            if imgui.menu_item("Delete Selected")[0]:
                self._delete_selected()
                imgui.close_current_popup()
            
            if imgui.menu_item("Select All")[0]:
                actions = self._get_actions()
                self.multi_selected_action_indices = set(range(len(actions)))
                imgui.close_current_popup()

            imgui.separator()
            
            # Selection Filters
            if imgui.begin_menu("Filters"):
                if imgui.menu_item("Keep Top Points")[0]: self._filter_selection('top')
                if imgui.menu_item("Keep Bottom Points")[0]: self._filter_selection('bottom')
                if imgui.menu_item("Keep Mid Points")[0]: self._filter_selection('mid')
                imgui.end_menu()

            # Timeline Ops
            imgui.separator()
            other_num = 2 if self.timeline_num == 1 else 1
            
            if imgui.menu_item("Copy Selected to Clipboard")[0]:
                self._handle_copy_selection()
                imgui.close_current_popup()
            
            if imgui.menu_item("Paste from Clipboard")[0]:
                t, v = getattr(self, 'new_point_candidate', (0, 0))
                self._handle_paste_actions(t)
                imgui.close_current_popup()
                
            if imgui.menu_item(f"Copy Selection to T{other_num}")[0]:
                self._handle_copy_to_other()
                imgui.close_current_popup()
                
            if imgui.menu_item(f"Swap with T{other_num}")[0]:
                self._handle_swap_timeline()
                imgui.close_current_popup()
                
            imgui.separator()
            
            # Allow plugins to inject menu items via plugin_renderer
            self._render_plugin_selection_menu()
            
            imgui.end_popup()
            
    def _render_plugin_selection_menu(self):
        # Helper to render selection-specific plugin actions
        if len(self.multi_selected_action_indices) < 2: return
        if imgui.begin_menu("Plugins (Selection)"):
             fs, axis = self._get_target_funscript_details()
             self.app.funscript_processor # Pass context if needed
             # Render simplified plugin list from manager
             available = self.plugin_renderer.plugin_manager.get_available_plugins()
             for p_name in available:
                 if imgui.menu_item(p_name)[0]:
                     # Trigger plugin context with selection
                     self.plugin_renderer.plugin_manager.set_plugin_state(p_name, PluginUIState.OPEN)
                     # Set context to selection mode (implementation specific to your plugin manager)
             imgui.end_menu()

    # ==================================================================================
    # DATA MODIFICATION HELPERS
    # ==================================================================================

    def _add_point(self, t, v):
        fs, axis = self._get_target_funscript_details()
        if not fs: return
        
        snap_t = self.app.app_state_ui.snap_to_grid_time_ms
        snap_v = self.app.app_state_ui.snap_to_grid_pos
        
        t = int(round(t / snap_t) * snap_t) if snap_t > 0 else int(t)
        v = int(round(v / snap_v) * snap_v) if snap_v > 0 else int(v)
        
        self.app.funscript_processor._record_timeline_action(self.timeline_num, "Add Point")
        fs.add_action(t, v if axis=='primary' else None, v if axis=='secondary' else None)
        self.app.funscript_processor._finalize_action_and_update_ui(self.timeline_num, "Add Point")
        self.invalidate_cache()

    def _delete_selected(self):
        if not self.multi_selected_action_indices:
            return

        fs, axis = self._get_target_funscript_details()
        if not fs:
            self.logger.error(f"Could not get funscript details for timeline {self.timeline_num}")
            return

        self.app.funscript_processor._record_timeline_action(self.timeline_num, "Delete Points")
        fs.clear_points(axis=axis, selected_indices=list(self.multi_selected_action_indices))
        self.multi_selected_action_indices.clear()
        self.selected_action_idx = -1
        self.app.funscript_processor._finalize_action_and_update_ui(self.timeline_num, "Delete Points")
        self.invalidate_cache()

    def _clear_all_points(self):
        """Delete all points on this timeline (undoable)."""
        fs, axis = self._get_target_funscript_details()
        if not fs:
            return

        # Get current point count
        actions = self._get_actions()
        num_points = len(actions) if actions else 0

        if num_points == 0:
            return

        # Record for undo
        self.app.funscript_processor._record_timeline_action(self.timeline_num, "Clear All Points")

        # Select all points then delete them
        all_indices = list(range(num_points))
        fs.clear_points(axis=axis, selected_indices=all_indices)

        # Clear selection
        self.multi_selected_action_indices.clear()
        self.selected_action_idx = -1

        # Finalize
        self.app.funscript_processor._finalize_action_and_update_ui(self.timeline_num, "Clear All Points")
        self.invalidate_cache()
        self.invalidate_ultimate_preview()

    def _get_timeline_status_text(self) -> str:
        """Generate status text showing timeline info (filename, axis, status)."""
        fs, axis = self._get_target_funscript_details()

        # Timeline number
        parts = [f"Timeline {self.timeline_num}"]

        # Axis name
        if axis:
            axis_display = axis.capitalize()
            parts.append(axis_display)

        # Get filename if available
        if self.app and hasattr(self.app, 'processor') and self.app.processor:
            video_path = getattr(self.app.processor, 'video_path', None)
            if video_path:
                import os
                filename = os.path.basename(video_path)
                # Truncate if too long
                if len(filename) > 30:
                    filename = filename[:27] + "..."
                parts.append(filename)

        # Status indicators
        if fs:
            actions = self._get_actions()
            num_points = len(actions) if actions else 0
            parts.append(f"{num_points} pts")

            # Check if generated or loaded
            if hasattr(fs, 'metadata') and fs.metadata:
                if fs.metadata.get('generated'):
                    parts.append("Generated")

        return " | ".join(parts)

    # ==================================================================================
    # MISC / UTILS
    # ==================================================================================
    
    def _update_ultimate_autotune_preview(self):
        if not self.show_ultimate_autotune_preview:
            self.ultimate_autotune_preview_actions = None
            return

        if not self._ultimate_preview_dirty: return

        # Generate preview via plugin system
        from funscript.plugins.base_plugin import plugin_registry
        plugin = plugin_registry.get_plugin('Ultimate Autotune')
        if plugin:
            fs, axis = self._get_target_funscript_details()
            if fs:
                # Create temp copy for non-destructive preview
                import copy
                temp = copy.deepcopy(fs)
                res = plugin.transform(temp, axis)
                if res:
                    self.ultimate_autotune_preview_actions = res.primary_actions if axis == 'primary' else res.secondary_actions
        
        self._ultimate_preview_dirty = False

    def _check_and_apply_pending_plugins(self):
        """Check for plugins with apply_requested flag and execute them."""
        # Get list of plugins that have been requested to apply
        apply_requests = self.plugin_renderer.plugin_manager.check_and_handle_apply_requests()

        if not apply_requests:
            return

        # Execute each requested plugin
        for plugin_name in apply_requests:
            self.logger.info(f"Executing pending plugin apply request: {plugin_name} on timeline {self.timeline_num}")

            # Get the plugin context to access parameters and settings
            context = self.plugin_renderer.plugin_manager.plugin_contexts.get(plugin_name)
            if not context:
                self.logger.error(f"No context found for plugin {plugin_name}")
                continue

            # Get target funscript and axis
            fs, axis = self._get_target_funscript_details()
            if not fs:
                self.logger.error(f"Could not get target funscript for {plugin_name}")
                continue

            # Get plugin instance from registry
            from funscript.plugins.base_plugin import plugin_registry
            plugin_instance = plugin_registry.get_plugin(plugin_name)
            if not plugin_instance:
                self.logger.error(f"Could not find plugin instance for {plugin_name}")
                continue

            # Prepare parameters - use context parameters
            params = dict(context.parameters) if context.parameters else {}

            # Handle selection if apply_to_selection is enabled
            selected_indices = None
            if context.apply_to_selection and self.multi_selected_action_indices:
                selected_indices = list(self.multi_selected_action_indices)
                params['selected_indices'] = selected_indices

            # Record undo action
            self.app.funscript_processor._record_timeline_action(
                self.timeline_num,
                f"Apply {plugin_name}"
            )

            # Apply the plugin transformation
            try:
                result = plugin_instance.transform(fs, axis, **params)

                # Plugins may return the modified funscript or None (for in-place modifications)
                # Both are valid - what matters is that the transformation was applied
                self.logger.info(f"Successfully applied {plugin_name} to timeline {self.timeline_num}")

                # Finalize and update UI
                self.app.funscript_processor._finalize_action_and_update_ui(
                    self.timeline_num,
                    f"Apply {plugin_name}"
                )

                # Invalidate caches
                self.invalidate_cache()
                self.invalidate_ultimate_preview()

                # Close the plugin window and clear its preview
                self.plugin_renderer.plugin_manager.set_plugin_state(
                    plugin_name,
                    PluginUIState.CLOSED
                )

                # Clear the preview for this plugin
                context.preview_actions = None

                # If this was the active preview, clear it from the renderer
                if self.plugin_renderer.plugin_manager.active_preview_plugin == plugin_name:
                    self.plugin_renderer.plugin_manager.active_preview_plugin = None
                    if self.plugin_preview_renderer:
                        self.plugin_preview_renderer.clear_preview(plugin_name)
            except Exception as e:
                self.logger.error(f"Error applying plugin {plugin_name}: {e}", exc_info=True)

    def _handle_sync_logic(self, app_state, tf):
        """Auto-scrolls timeline during playback."""
        processor = self.app.processor
        if not processor or not processor.video_info: return

        # Check if video is playing - use is_playing attribute if available
        is_playing = False
        if hasattr(processor, 'is_playing'):
            is_playing = processor.is_playing
        elif hasattr(processor, 'is_processing'):
            # Fallback: check if processing and not paused
            pause_event = getattr(processor, "pause_event", None)
            if pause_event is not None:
                is_playing = processor.is_processing and not pause_event.is_set()
            else:
                is_playing = processor.is_processing

        forced = app_state.force_timeline_pan_to_current_frame

        # DEBUG: Uncomment to see sync state
        if self.timeline_num == 1 and (is_playing or forced):
            print(f"[TL Sync] is_playing={is_playing}, forced={forced}, interaction_active={app_state.timeline_interaction_active}, current_frame={processor.current_frame_index}")

        # Auto-scroll during playback (ignore interaction flag when playing)
        # Only respect interaction flag when forced sync (manual seeking while paused)
        should_sync = is_playing or (forced and not app_state.timeline_interaction_active)

        if should_sync:
            current_ms = (processor.current_frame_index / processor.fps) * 1000.0

            # Center the playhead
            center_offset = (tf.width * tf.zoom) / 2
            target_pan = current_ms - center_offset

            app_state.timeline_pan_offset_ms = target_pan

            if forced: app_state.force_timeline_pan_to_current_frame = False
