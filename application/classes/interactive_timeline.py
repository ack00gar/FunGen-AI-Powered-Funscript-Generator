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
from application.utils.feature_detection import is_feature_available as _is_feature_available
from application.utils.timeline_constants import EXTRA_TIMELINE_RANGE
from config.element_group_colors import TimelineColors
from funscript.axis_registry import FunscriptAxis, AXIS_TCODE
from application.utils.heatmap_utils import HeatmapColorMapper
from application.utils.timeline_modes import TimelineMode, TimelineInteractionState
from application.utils.bpm_analyzer import BPMOverlayConfig, TapTempo
from application.classes.bookmark_manager import BookmarkManager
from application.classes.recording_capture import RecordingCapture

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
        self.nudge_chapter_only = False  # When True, << >> only affect points in selected chapter
        self._container_mode = False  # Set by render() when inside scrollable container

        # --- OFS-Inspired Features ---

        # Heatmap coloring (Phase 1.1)
        self._show_heatmap_coloring = False
        self._heatmap_mapper = HeatmapColorMapper(max_speed=400.0)

        # Speed limit visualization (Phase 1.3)
        self._show_speed_warnings = False
        self._speed_limit_threshold = 400.0

        # Bookmarks (Phase 1.6)
        self._bookmark_manager = BookmarkManager()
        self._bookmark_rename_id: Optional[str] = None
        self._bookmark_rename_buf: str = ""

        # Timeline Mode State Machine (Phase 2.1)
        self._mode = TimelineMode.SELECT
        self._interaction_state = TimelineInteractionState.IDLE

        # Alternating Mode (Phase 2.2)
        self._alt_next_is_top = True
        self._alt_top_value = 95
        self._alt_bottom_value = 5

        # Recording Mode (Phase 3.2)
        self._recording_capture: Optional[RecordingCapture] = None
        self._recording_rdp_epsilon = 2.0

        # BPM/Tempo Overlay (Phase 3.3)
        self._bpm_config: Optional[BPMOverlayConfig] = None
        self._tap_tempo = TapTempo()

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
            return fs.get_axis_actions(axis)
        return []

    def _get_cached_timestamps(self) -> list:
        """Return the funscript's cached timestamp list for this timeline's axis."""
        fs, axis = self._get_target_funscript_details()
        if fs and axis:
            return fs._get_timestamps_for_axis(axis)
        return []

    def invalidate_cache(self):
        """Forces updates on next frame"""
        self._ultimate_preview_dirty = True

    def invalidate_ultimate_preview(self):
        self._ultimate_preview_dirty = True

    # ==================================================================================
    # MAIN RENDER LOOP
    # ==================================================================================

    def render(self, y_pos: float = 0, height: float = 0, view_mode: str = 'expert',
               container_mode: bool = False):
        app_state = self.app.app_state_ui
        visibility_attr = f"show_funscript_interactive_timeline{'' if self.timeline_num == 1 else str(self.timeline_num)}"

        if not getattr(app_state, visibility_attr, False):
            return

        self._container_mode = container_mode

        # 1. Window Configuration
        is_floating = app_state.ui_layout_mode == "floating"
        # NO_BRING_TO_FRONT_ON_FOCUS prevents the timeline from stealing
        # z-order when clicked — dialog/plugin windows stay on top.
        flags = (imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_SCROLL_WITH_MOUSE |
                 imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS)

        if container_mode:
            # Render as a child region inside a scrollable container
            if height <= 0: return
            child_flags = imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_SCROLL_WITH_MOUSE
            if not imgui.begin_child(f"##TimelineChild{self.timeline_num}", 0, height, border=True, flags=child_flags):
                imgui.end_child()
                return
        elif not is_floating:
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
            axis_label = self._get_axis_label()
            window_title = f"T{self.timeline_num}: {axis_label}" if axis_label else f"Interactive Timeline {self.timeline_num}"
            is_open, visible = imgui.begin(window_title, True, flags)
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
            imgui.end_child() if self._container_mode else imgui.end()
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

        # 6-pre. Speed limit overlay (behind curve)
        if self._show_speed_warnings and main_actions:
            self._draw_speed_limit_overlay(draw_list, tf, main_actions)

        # 6-pre2. BPM/Tempo grid overlay
        if self._bpm_config and _is_feature_available("patreon_features"):
            self._draw_bpm_grid(draw_list, tf)

        # 6a. Update & Draw Ultimate Preview (if enabled)
        self._update_ultimate_autotune_preview()
        if self.ultimate_autotune_preview_actions:
             self._draw_curve(draw_list, tf, self.ultimate_autotune_preview_actions,
                              color_override=TimelineColors.ULTIMATE_AUTOTUNE_PREVIEW,
                              force_lines_only=True, alpha=0.7)

        # 6b. Draw Active Plugin Preview (if any)
        if self.is_previewing and self.preview_actions:
             self._draw_curve(draw_list, tf, self.preview_actions, is_preview=True)

        # 6c. Draw Main Script (heatmap or standard)
        if self._show_heatmap_coloring and main_actions and len(main_actions) >= 2:
            self._draw_curve_heatmap(draw_list, tf, main_actions)
        else:
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

        # 9. Draw Active/Read-only State Border
        self._draw_state_border(draw_list, canvas_pos, canvas_size, app_state)

        imgui.end_child() if self._container_mode else imgui.end()

    # ==================================================================================
    # INPUT HANDLING
    # ==================================================================================

    def _handle_input(self, app_state, tf: TimelineTransformer):
        io = imgui.get_io()
        mouse_pos = imgui.get_mouse_pos()

        # Check canvas bounds AND that no other window/popup/dialog is on top.
        # is_window_hovered() returns False when a popup, dialog, or overlapping
        # window is above this one — preventing click-through to the canvas.
        in_canvas = (tf.x_offset <= mouse_pos[0] <= tf.x_offset + tf.width and
                     tf.y_offset <= mouse_pos[1] <= tf.y_offset + tf.height)
        is_hovered = in_canvas and imgui.is_window_hovered()
        is_focused = imgui.is_window_focused(imgui.FOCUS_ROOT_AND_CHILD_WINDOWS)

        # Update active timeline ONLY on explicit user interaction (click)
        # This prevents the last-rendered timeline from stealing focus on startup
        if is_hovered and imgui.is_mouse_clicked(0):  # Left click
            app_state.active_timeline_num = self.timeline_num

        # --- Keyboard Shortcuts (Global / Focused) ---
        if is_focused:
            self._handle_keyboard_shortcuts(app_state, io)

        # --- Navigation (Zoom/Pan) ---
        if is_hovered:
            # Wheel Zoom
            if io.mouse_wheel != 0:
                scale = 0.85 if io.mouse_wheel > 0 else 1.15
                # Zoom centered on playhead (center of timeline) to keep funscript position stable
                playhead_x = tf.x_offset + tf.width / 2
                playhead_time_ms = tf.x_to_time(playhead_x)

                new_zoom = max(0.01, min(2000.0, tf.zoom * scale))
                # Adjust pan to keep playhead centered on the same time point
                center_offset_px = tf.width / 2
                new_pan = playhead_time_ms - (center_offset_px * new_zoom)

                app_state.timeline_zoom_factor_ms_per_px = new_zoom
                app_state.timeline_pan_offset_ms = new_pan
                app_state.timeline_interaction_active = True

            # Middle Drag Pan
            if imgui.is_mouse_dragging(glfw.MOUSE_BUTTON_MIDDLE):
                delta_x = io.mouse_delta[0]
                app_state.timeline_pan_offset_ms -= delta_x * tf.zoom
                app_state.timeline_interaction_active = True

        # --- Mode-specific input dispatch ---
        if self._mode == TimelineMode.ALTERNATING:
            self._handle_alternating_mode_input(app_state, tf, mouse_pos, is_hovered, io)
        elif self._mode == TimelineMode.RECORDING:
            self._handle_recording_mode_input(app_state, tf, mouse_pos, is_hovered, io)
        elif self._mode == TimelineMode.INJECTION and _is_feature_available("patreon_features"):
            self._handle_injection_mode_input(app_state, tf, mouse_pos, is_hovered, io)

        # --- Action Interaction (SELECT mode or fallback) ---
        actions = self._get_actions()

        # Left Click
        if is_hovered and imgui.is_mouse_clicked(glfw.MOUSE_BUTTON_LEFT) and self._mode == TimelineMode.SELECT:
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
            # Seek video to the current playhead position (center of timeline)
            center_time_ms = tf.x_to_time(tf.x_offset + tf.width / 2)
            self._seek_video(center_time_ms)

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
        self._render_bookmark_rename_popup()

    def _render_bookmark_rename_popup(self):
        """Render popup for renaming a bookmark."""
        if self._bookmark_rename_id is None:
            return

        if not imgui.is_popup_open("Rename Bookmark##popup"):
            imgui.open_popup("Rename Bookmark##popup")

        if imgui.begin_popup_modal("Rename Bookmark##popup", flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
            imgui.text("Enter bookmark name:")
            changed, self._bookmark_rename_buf = imgui.input_text(
                "##bm_rename", self._bookmark_rename_buf, 128,
                imgui.INPUT_TEXT_ENTER_RETURNS_TRUE
            )

            if changed or imgui.button("OK", 80, 0):
                self._bookmark_manager.rename(self._bookmark_rename_id, self._bookmark_rename_buf)
                self._bookmark_rename_id = None
                self._bookmark_rename_buf = ""
                imgui.close_current_popup()

            imgui.same_line()
            if imgui.button("Cancel", 80, 0):
                self._bookmark_rename_id = None
                self._bookmark_rename_buf = ""
                imgui.close_current_popup()

            imgui.end_popup()

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
            # Paste at current playhead position (video time), not stale mouse position
            paste_time_ms = 0
            if self.app.processor and self.app.processor.fps > 0:
                paste_time_ms = (self.app.processor.current_frame_index / self.app.processor.fps) * 1000.0
            self._handle_paste_actions(paste_time_ms)

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

        # 7. Bookmark at playhead (B key)
        if check_shortcut("add_bookmark", "B"):
            center_x = (app_state.window_width or 800) / 2
            playhead_time = getattr(app_state, 'timeline_pan_offset_ms', 0) + (center_x * getattr(app_state, 'timeline_zoom_factor_ms_per_px', 1.0))
            self._bookmark_manager.add(playhead_time)

    def _hit_test_point(self, mouse_pos, actions, tf: TimelineTransformer) -> int:
        """Optimized hit testing using binary search."""
        if not actions: return -1

        tol_px = 8.0 # Pixel radius tolerance
        tol_ms = tol_px * tf.zoom

        t_mouse = tf.x_to_time(mouse_pos[0])

        # Use funscript's cached timestamp list instead of rebuilding per call
        timestamps = self._get_cached_timestamps()
        start_idx = bisect_left(timestamps, t_mouse - tol_ms)
        end_idx = bisect_right(timestamps, t_mouse + tol_ms)
        
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

        # Optimize: Binary search time bounds using cached timestamps
        timestamps = self._get_cached_timestamps()
        if not timestamps or len(timestamps) != len(actions):
            timestamps = [a['at'] for a in actions]
        s_idx = bisect_left(timestamps, t_start)
        e_idx = bisect_right(timestamps, t_end)

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

        timestamps = self._get_cached_timestamps()
        if not timestamps or len(timestamps) != len(actions):
            timestamps = [a['at'] for a in actions]
        s_idx = bisect_left(timestamps, t1)
        e_idx = bisect_right(timestamps, t2)
        
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

    def _nudge_chapter_time(self, frames: int):
        """Nudge points within the selected chapter(s) by a number of frames"""
        # Get selected chapters from video_navigation_ui
        selected_chapters = []
        if self.app.gui_instance and hasattr(self.app.gui_instance, 'video_navigation_ui'):
            nav_ui = self.app.gui_instance.video_navigation_ui
            if nav_ui and hasattr(nav_ui, 'context_selected_chapters'):
                selected_chapters = nav_ui.context_selected_chapters

        if not selected_chapters:
            if self.logger:
                self.logger.info("No chapter selected for nudging", extra={'status_message': True})
            return

        actions = self._get_actions()
        if not actions: return

        processor = self.app.processor
        if not processor or not processor.video_info: return

        fps = processor.fps
        if fps <= 0: return

        # Convert frames to milliseconds
        delta_ms = int((frames / fps) * 1000.0)

        self.app.funscript_processor._record_timeline_action(self.timeline_num, "Nudge Chapter Points")

        # Process each selected chapter
        total_nudged = 0
        for chapter in selected_chapters:
            # Get chapter time range
            start_ms = int(round((chapter.start_frame_id / fps) * 1000.0))
            end_ms = int(round((chapter.end_frame_id / fps) * 1000.0))

            # Find points within chapter using binary search on cached timestamps
            action_timestamps = self._get_cached_timestamps()
            if not action_timestamps or len(action_timestamps) != len(actions):
                action_timestamps = [a['at'] for a in actions]
            start_idx = bisect_left(action_timestamps, start_ms)
            end_idx = bisect_right(action_timestamps, end_ms)

            # Nudge only points within the chapter
            for i in range(start_idx, end_idx):
                actions[i]['at'] = max(0, actions[i]['at'] + delta_ms)
                total_nudged += 1

        if total_nudged == 0:
            if self.logger:
                self.logger.info("No points found in selected chapter(s)", extra={'status_message': True})
            return

        self.app.funscript_processor._finalize_action_and_update_ui(self.timeline_num, "Nudge Chapter Points")
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

    def _handle_swap_timeline(self, target_num=None):
        if target_num is None:
            target_num = 2 if self.timeline_num == 1 else 1
        self.app.funscript_processor.swap_timelines(self.timeline_num, target_num)

    def _handle_copy_to_other(self, target_num=None):
        actions = self._get_actions()
        if not self.multi_selected_action_indices: return

        other_num = target_num if target_num is not None else (2 if self.timeline_num == 1 else 1)
        fs_other, axis_other = self.app.funscript_processor._get_target_funscript_object_and_axis(other_num)

        if not fs_other: return

        indices = sorted(list(self.multi_selected_action_indices))
        points_to_copy = [actions[i] for i in indices]

        self.app.funscript_processor._record_timeline_action(other_num, f"Copy from T{self.timeline_num}")

        if axis_other in ('primary', 'secondary'):
            # Use batch add for built-in axes
            batch = []
            for p in points_to_copy:
                batch.append({
                    'timestamp_ms': p['at'],
                    'primary_pos': p['pos'] if axis_other == 'primary' else None,
                    'secondary_pos': p['pos'] if axis_other == 'secondary' else None
                })
            fs_other.add_actions_batch(batch, is_from_live_tracker=False)
        else:
            # Use add_action_to_axis for additional axes
            for p in points_to_copy:
                fs_other.add_action_to_axis(axis_other, p['at'], p['pos'])

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

            if val == 100:
                # Place below the line
                label_y = y + 2
            elif val == 25 or val == 50 or val == 75:
                # Center on the line with background for readability
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
                # 0: above the line
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
        data = self.app.get_waveform_data()
        if not self.app.app_state_ui.show_audio_waveform or data is None: return
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

        # 1. Culling: Identify visible slice using cached timestamps
        margin_ms = tf.zoom * 100
        # For main curves, prefer the funscript's cached timestamp list (avoids O(n) rebuild)
        if not is_preview and not color_override:
            timestamps = self._get_cached_timestamps()
            if not timestamps or len(timestamps) != len(actions):
                timestamps = [a['at'] for a in actions]
        else:
            timestamps = [a['at'] for a in actions]
        s_idx = bisect_left(timestamps, tf.visible_start_ms - margin_ms)
        e_idx = bisect_right(timestamps, tf.visible_end_ms + margin_ms)
        
        s_idx = max(0, s_idx - 1)
        e_idx = min(len(actions), e_idx + 1)
        
        if e_idx - s_idx < 2: return

        visible_actions = actions[s_idx:e_idx]
        
        # 2. Vectorized Transform
        ats = np.array([a['at'] for a in visible_actions], dtype=np.float32)
        poss = np.array([a['pos'] for a in visible_actions], dtype=np.float32)
        
        xs = tf.vec_time_to_x(ats)
        ys = tf.vec_val_to_y(poss)

        # CLAMP COORDINATES: Fix invisible lines when zoomed in on sparse data
        # ImGui rendering can glitch if coordinates exceed +/- 32k (integer overflow in vertex buffer)
        # We clamp x coordinates to a safe range slightly outside the viewport
        safe_min_x = tf.x_offset - 5000
        safe_max_x = tf.x_offset + tf.width + 5000
        xs = np.clip(xs, safe_min_x, safe_max_x)

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

    # ==================================================================================
    # OFS-INSPIRED DRAWING METHODS
    # ==================================================================================

    def _draw_curve_heatmap(self, dl, tf: TimelineTransformer, actions: List[Dict]):
        """Draw the main curve with per-segment heatmap coloring (Phase 1.1)."""
        if not actions or len(actions) < 2:
            return

        # Culling
        margin_ms = tf.zoom * 100
        timestamps = self._get_cached_timestamps()
        if not timestamps or len(timestamps) != len(actions):
            timestamps = [a['at'] for a in actions]
        s_idx = bisect_left(timestamps, tf.visible_start_ms - margin_ms)
        e_idx = bisect_right(timestamps, tf.visible_end_ms + margin_ms)
        s_idx = max(0, s_idx - 1)
        e_idx = min(len(actions), e_idx + 1)
        if e_idx - s_idx < 2:
            return

        visible_actions = actions[s_idx:e_idx]

        # Vectorized transform
        ats = np.array([a['at'] for a in visible_actions], dtype=np.float32)
        poss = np.array([a['pos'] for a in visible_actions], dtype=np.float32)
        xs = tf.vec_time_to_x(ats)
        ys = tf.vec_val_to_y(poss)

        # Clamp coordinates
        safe_min_x = tf.x_offset - 5000
        safe_max_x = tf.x_offset + tf.width + 5000
        xs = np.clip(xs, safe_min_x, safe_max_x)

        # Compute per-segment speeds
        speeds = HeatmapColorMapper.compute_segment_speeds(visible_actions)
        colors = self._heatmap_mapper.speeds_to_colors_rgba(speeds)

        # Draw per-segment colored lines
        for i in range(len(speeds)):
            c = colors[i]
            col_u32 = imgui.get_color_u32_rgba(float(c[0]), float(c[1]), float(c[2]), float(c[3]))
            dl.add_line(float(xs[i]), float(ys[i]), float(xs[i + 1]), float(ys[i + 1]), col_u32, 2.0)

        # Draw points (same logic as standard _draw_curve)
        radius = self.app.app_state_ui.timeline_point_radius
        pixels_per_point = tf.width / max(1, len(xs))
        if pixels_per_point > 5:
            for i in range(len(visible_actions)):
                real_idx = s_idx + i
                is_sel = real_idx in self.multi_selected_action_indices
                is_drag = (real_idx == self.dragging_action_idx)

                if is_drag:
                    c_tuple = TimelineColors.POINT_DRAGGING
                    r = radius + 2
                elif is_sel:
                    c_tuple = TimelineColors.POINT_SELECTED
                    r = radius + 1
                else:
                    c_tuple = TimelineColors.POINT_DEFAULT
                    r = radius

                dl.add_circle_filled(float(xs[i]), float(ys[i]), r,
                                     imgui.get_color_u32_rgba(*c_tuple))
                if is_sel:
                    dl.add_circle(float(xs[i]), float(ys[i]), r + 1,
                                  imgui.get_color_u32_rgba(*TimelineColors.SELECTED_POINT_BORDER))

    def _draw_speed_limit_overlay(self, dl, tf: TimelineTransformer, actions: List[Dict]):
        """Draw red semi-transparent bands for speed limit violations (Phase 1.3)."""
        if not actions or len(actions) < 2:
            return

        # Culling
        margin_ms = tf.zoom * 100
        timestamps = self._get_cached_timestamps()
        if not timestamps or len(timestamps) != len(actions):
            timestamps = [a['at'] for a in actions]
        s_idx = bisect_left(timestamps, tf.visible_start_ms - margin_ms)
        e_idx = bisect_right(timestamps, tf.visible_end_ms + margin_ms)
        s_idx = max(0, s_idx - 1)
        e_idx = min(len(actions), e_idx + 1)
        if e_idx - s_idx < 2:
            return

        visible_actions = actions[s_idx:e_idx]
        speeds = HeatmapColorMapper.compute_segment_speeds(visible_actions)
        threshold = self._speed_limit_threshold

        ats = np.array([a['at'] for a in visible_actions], dtype=np.float32)
        xs = tf.vec_time_to_x(ats)
        xs = np.clip(xs, tf.x_offset - 100, tf.x_offset + tf.width + 100)

        violation_col = imgui.get_color_u32_rgba(0.9, 0.1, 0.1, 0.15)
        for i in range(len(speeds)):
            if speeds[i] > threshold:
                x1 = float(xs[i])
                x2 = float(xs[i + 1])
                dl.add_rect_filled(x1, tf.y_offset, x2, tf.y_offset + tf.height, violation_col)

    def _draw_bpm_grid(self, dl, tf: TimelineTransformer):
        """Draw BPM beat grid lines on the timeline (Phase 3.3)."""
        cfg = self._bpm_config
        if not cfg or cfg.bpm <= 0:
            return

        interval_ms = cfg.beat_interval_ms
        if interval_ms <= 0:
            return

        # Calculate visible beat positions
        start_beat = int((tf.visible_start_ms - cfg.offset_ms) / interval_ms)
        end_beat = int((tf.visible_end_ms - cfg.offset_ms) / interval_ms) + 1

        # Base beat interval (quarter note, ignoring subdivision)
        base_interval = 60000.0 / cfg.bpm

        beat_col = imgui.get_color_u32_rgba(0.6, 0.3, 0.8, 0.3)
        downbeat_col = imgui.get_color_u32_rgba(0.7, 0.3, 0.9, 0.6)

        for beat_num in range(start_beat, end_beat + 1):
            t_ms = cfg.offset_ms + beat_num * interval_ms
            if t_ms < tf.visible_start_ms or t_ms > tf.visible_end_ms:
                continue
            x = tf.time_to_x(t_ms)

            # Check if this is a downbeat (on the main quarter-note grid)
            is_downbeat = (abs((t_ms - cfg.offset_ms) % base_interval) < 1.0) if cfg.subdivision > 1 else True
            col = downbeat_col if is_downbeat else beat_col
            thick = 1.5 if is_downbeat else 0.8

            dl.add_line(x, tf.y_offset, x, tf.y_offset + tf.height, col, thick)

    def _draw_bookmarks(self, dl, tf: TimelineTransformer):
        """Draw bookmark markers on the timeline (Phase 1.6)."""
        visible = self._bookmark_manager.get_in_range(tf.visible_start_ms, tf.visible_end_ms)
        if not visible:
            return

        for bm in visible:
            x = tf.time_to_x(bm.time_ms)
            col = imgui.get_color_u32_rgba(*bm.color)

            # Vertical line
            dl.add_line(x, tf.y_offset, x, tf.y_offset + tf.height, col, 1.5)

            # Triangle marker at top
            tri_size = 6
            dl.add_triangle_filled(
                x, tf.y_offset,
                x - tri_size, tf.y_offset - tri_size,
                x + tri_size, tf.y_offset - tri_size,
                col
            )

            # Label
            if bm.name:
                dl.add_text(x + 4, tf.y_offset + 2, col, bm.name[:20])

    # ==================================================================================
    # MODE-SPECIFIC INPUT HANDLERS
    # ==================================================================================

    def _handle_alternating_mode_input(self, app_state, tf: TimelineTransformer,
                                        mouse_pos, is_hovered, io):
        """Handle input for alternating mode (Phase 2.2).

        Left-click places a point at click X, with Y alternating between
        top and bottom values. Inspects last action to auto-determine direction.
        """
        if not is_hovered:
            return

        if imgui.is_mouse_clicked(0) and not io.key_ctrl:
            click_time = tf.x_to_time(mouse_pos[0])

            # Auto-determine next direction from last action
            actions = self._get_actions()
            if actions:
                # Find nearest previous action
                nearest_idx = bisect_left([a['at'] for a in actions], click_time)
                if nearest_idx > 0:
                    prev_pos = actions[nearest_idx - 1]['pos']
                    mid = (self._alt_top_value + self._alt_bottom_value) / 2
                    self._alt_next_is_top = prev_pos < mid

            # Place point
            val = self._alt_top_value if self._alt_next_is_top else self._alt_bottom_value
            self._add_point(click_time, val)
            self._alt_next_is_top = not self._alt_next_is_top

    def _handle_recording_mode_input(self, app_state, tf: TimelineTransformer,
                                      mouse_pos, is_hovered, io):
        """Handle input for recording mode (Phase 3.2).

        While recording: map mouse Y in canvas to 0-100, capture each frame.
        """
        if not self._recording_capture:
            return

        if self._recording_capture.is_recording:
            # Map mouse Y to 0-100
            if is_hovered and tf.height > 0:
                normalized_y = 1.0 - ((mouse_pos[1] - tf.y_offset) / tf.height)
                pos = max(0, min(100, normalized_y * 100.0))

                # Get current video time
                processor = self.app.processor
                if processor and processor.fps > 0:
                    current_ms = (processor.current_frame_index / processor.fps) * 1000.0
                    self._recording_capture.capture_frame(current_ms, pos)
                    # Show raw samples as live preview on timeline
                    self.preview_actions = self._recording_capture._samples
                    self.is_previewing = True

    def _handle_injection_mode_input(self, app_state, tf: TimelineTransformer,
                                      mouse_pos, is_hovered, io):
        """Handle input for injection mode (Phase 2.3).

        Click on a segment to inject intermediate points into it.
        """
        if not is_hovered:
            return

        actions = self._get_actions()
        if not actions or len(actions) < 2:
            return

        click_time = tf.x_to_time(mouse_pos[0])

        # Highlight segment under cursor (visual feedback)
        timestamps = [a['at'] for a in actions]
        idx = bisect_left(timestamps, click_time)
        if idx <= 0 or idx >= len(actions):
            return

        if imgui.is_mouse_clicked(0) and not io.key_ctrl:
            # Inject points into this segment
            a0 = actions[idx - 1]
            a1 = actions[idx]
            dt = a1['at'] - a0['at']
            if dt < 40:
                return  # Segment too short

            # Record undo
            self._record_timeline_action()

            # Generate interpolated points
            num_injections = max(1, int(dt / 100)) - 1
            new_actions = list(actions)  # Copy
            insert_pos = idx
            for j in range(1, num_injections + 1):
                t_frac = j / (num_injections + 1)
                t_ms = a0['at'] + dt * t_frac
                # Cosine interpolation
                t2 = (1.0 - math.cos(t_frac * math.pi)) / 2.0
                pos = a0['pos'] + (a1['pos'] - a0['pos']) * t2
                new_actions.insert(insert_pos, {
                    'at': int(round(t_ms)),
                    'pos': max(0, min(100, int(round(pos)))),
                })
                insert_pos += 1

            new_actions.sort(key=lambda a: a['at'])

            # Apply
            fs, axis = self._get_target_funscript_details()
            if fs and axis:
                fs.set_axis_actions(axis, new_actions)
                self._finalize_action_and_update_ui()

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

        # 4. Bookmarks
        self._draw_bookmarks(dl, tf)

        # 5. Recording indicator
        if self._recording_capture and self._recording_capture.is_recording:
            rec_col = imgui.get_color_u32_rgba(0.9, 0.1, 0.1, 1.0)
            dl.add_circle_filled(tf.x_offset + 12, tf.y_offset + 12, 5, rec_col)
            dl.add_text(tf.x_offset + 20, tf.y_offset + 5,
                        imgui.get_color_u32_rgba(0.9, 0.1, 0.1, 1.0), "REC")

    def _draw_state_border(self, dl, canvas_pos, canvas_size, app_state):
        """
        Draw a colored border indicating timeline state:
        - Green: Active and editable (shortcuts will work)
        - Red: Active but read-only (during playback, text input, etc.)
        - Gray: Inactive (another timeline is active)
        """
        is_active = app_state.active_timeline_num == self.timeline_num

        if not is_active:
            # Gray border for inactive timeline
            border_color = imgui.get_color_u32_rgba(0.4, 0.4, 0.4, 0.6)
        else:
            # Check if editable or read-only
            is_read_only = self._is_timeline_read_only(app_state)
            if is_read_only:
                # Red border for active but read-only
                border_color = imgui.get_color_u32_rgba(0.9, 0.2, 0.2, 0.8)
            else:
                # Green border for active and editable
                border_color = imgui.get_color_u32_rgba(0.2, 0.8, 0.2, 0.8)

        # Draw border around canvas area
        x1, y1 = canvas_pos[0], canvas_pos[1]
        x2, y2 = x1 + canvas_size[0], y1 + canvas_size[1]
        border_thickness = 2.0 if is_active else 1.0
        dl.add_rect(x1, y1, x2, y2, border_color, 0.0, 0, border_thickness)

    def _is_timeline_read_only(self, app_state) -> bool:
        """Check if timeline is in read-only mode (shortcuts blocked)."""
        # Video is playing
        if self.app.processor and getattr(self.app.processor, 'is_playing', False):
            return True

        # Text input is active
        io = imgui.get_io()
        if io.want_text_input:
            return True

        # Shortcut recording in progress
        if self.app.shortcut_manager and self.app.shortcut_manager.is_recording_shortcut_for:
            return True

        # Live tracking is active
        if self.app.processor and getattr(self.app.processor, 'is_processing', False):
            return True

        return False

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
            if self.nudge_chapter_only:
                self._nudge_chapter_time(-1)
            else:
                self._nudge_all_time(-1)
        if imgui.is_item_hovered():
            tip = "Nudge points in selected chapter left by 1 frame" if self.nudge_chapter_only else "Nudge all points left by 1 frame"
            imgui.set_tooltip(tip)
        imgui.same_line()

        if imgui.button(f">>##{self.timeline_num}"):
            if self.nudge_chapter_only:
                self._nudge_chapter_time(1)
            else:
                self._nudge_all_time(1)
        if imgui.is_item_hovered():
            tip = "Nudge points in selected chapter right by 1 frame" if self.nudge_chapter_only else "Nudge all points right by 1 frame"
            imgui.set_tooltip(tip)
        imgui.same_line()

        # Chapter-only nudge checkbox
        _, self.nudge_chapter_only = imgui.checkbox(f"Ch.##{self.timeline_num}", self.nudge_chapter_only)
        if imgui.is_item_hovered():
            imgui.set_tooltip("When checked, << and >> only affect points in the selected chapter")
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

        # --- OFS-Inspired Visualization Toggles ---
        imgui.same_line()
        imgui.text("|")
        imgui.same_line()

        # Heatmap toggle
        _, self._show_heatmap_coloring = imgui.checkbox(f"Heat##{self.timeline_num}", self._show_heatmap_coloring)
        if imgui.is_item_hovered():
            imgui.set_tooltip("OFS-style heatmap coloring (speed-based segment colors)")
        imgui.same_line()

        # Speed warnings toggle
        _, self._show_speed_warnings = imgui.checkbox(f"Spd##{self.timeline_num}", self._show_speed_warnings)
        if imgui.is_item_hovered():
            imgui.set_tooltip(f"Highlight speed limit violations (>{self._speed_limit_threshold:.0f} u/s)")

        # --- Mode Selector ---
        imgui.same_line()
        imgui.text("|")
        imgui.same_line()

        mode_labels = ["Select", "Alternating"]
        # Add patreon-exclusive modes
        _is_patreon = _is_feature_available("patreon_features")
        if _is_patreon:
            mode_labels.extend(["Injection", "Recording"])

        mode_map = [TimelineMode.SELECT, TimelineMode.ALTERNATING]
        if _is_patreon:
            mode_map.extend([TimelineMode.INJECTION, TimelineMode.RECORDING])

        current_mode_idx = 0
        for i, m in enumerate(mode_map):
            if m == self._mode:
                current_mode_idx = i
                break

        imgui.push_item_width(90)
        changed_mode, new_mode_idx = imgui.combo(f"Mode##{self.timeline_num}", current_mode_idx, mode_labels)
        imgui.pop_item_width()
        if changed_mode:
            self._mode = mode_map[new_mode_idx]
        if imgui.is_item_hovered():
            imgui.set_tooltip("Timeline editing mode")

        # Mode-specific toolbar additions
        if self._mode == TimelineMode.ALTERNATING:
            imgui.same_line()
            imgui.push_item_width(50)
            _, self._alt_top_value = imgui.slider_int(f"Top##{self.timeline_num}", self._alt_top_value, 50, 100)
            imgui.same_line()
            _, self._alt_bottom_value = imgui.slider_int(f"Bot##{self.timeline_num}", self._alt_bottom_value, 0, 50)
            imgui.pop_item_width()

        elif self._mode == TimelineMode.RECORDING and _is_patreon:
            imgui.same_line()
            if self._recording_capture and self._recording_capture.is_recording:
                if imgui.button(f"Stop Rec##{self.timeline_num}"):
                    self.is_previewing = False
                    self.preview_actions = None
                    simplified = self._recording_capture.stop_recording(self._recording_rdp_epsilon)
                    if simplified:
                        self._record_timeline_action()
                        actions = self._get_actions()
                        actions.extend(simplified)
                        actions.sort(key=lambda a: a['at'])
                        fs, axis = self._get_target_funscript_details()
                        if fs and axis:
                            fs.set_axis_actions(axis, actions)
                            self._finalize_action_and_update_ui()
            else:
                if imgui.button(f"Record##{self.timeline_num}"):
                    self._recording_capture = RecordingCapture()
                    self._recording_capture.start_recording()
                    self.is_previewing = True
                    self.preview_actions = self._recording_capture._samples
            imgui.same_line()
            imgui.push_item_width(60)
            _, self._recording_rdp_epsilon = imgui.slider_float(
                f"RDP##{self.timeline_num}", self._recording_rdp_epsilon, 0.5, 10.0, "%.1f")
            imgui.pop_item_width()
            if imgui.is_item_hovered():
                imgui.set_tooltip("RDP simplification (higher = fewer points)")

        # BPM controls (Patreon exclusive)
        if _is_patreon:
            imgui.same_line()
            imgui.text("|")
            imgui.same_line()
            has_bpm = self._bpm_config is not None
            _, has_bpm = imgui.checkbox(f"BPM##{self.timeline_num}", has_bpm)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Show BPM beat grid overlay")
            if has_bpm and self._bpm_config is None:
                self._bpm_config = BPMOverlayConfig()
            elif not has_bpm:
                self._bpm_config = None

            if self._bpm_config:
                imgui.same_line()
                imgui.push_item_width(55)
                _, self._bpm_config.bpm = imgui.drag_float(
                    f"##bpm_val{self.timeline_num}", self._bpm_config.bpm, 0.5, 30.0, 300.0, "%.0f")
                imgui.pop_item_width()
                if imgui.is_item_hovered():
                    imgui.set_tooltip("BPM value (drag to adjust)")
                imgui.same_line()
                if imgui.button(f"Tap##{self.timeline_num}"):
                    bpm = self._tap_tempo.tap()
                    if bpm:
                        self._bpm_config.bpm = round(bpm, 1)
                if imgui.is_item_hovered():
                    imgui.set_tooltip(f"Tap tempo ({self._tap_tempo.tap_count} taps)")

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

            # Build list of visible target timelines
            _copy_swap_targets = []
            _copy_swap_targets.append(other_num)  # Always include the default other (T1<->T2)
            app_state = self.app.app_state_ui
            if _is_feature_available("patreon_features"):
                for t_num in EXTRA_TIMELINE_RANGE:
                    if t_num == self.timeline_num:
                        continue
                    vis_attr = f"show_funscript_interactive_timeline{t_num}"
                    if getattr(app_state, vis_attr, False):
                        _copy_swap_targets.append(t_num)

            if imgui.begin_menu("Copy Selection to..."):
                for t_num in _copy_swap_targets:
                    if imgui.menu_item(self._tl_label(t_num))[0]:
                        self._handle_copy_to_other(t_num)
                        imgui.close_current_popup()
                imgui.end_menu()
            if imgui.begin_menu("Swap with..."):
                for t_num in _copy_swap_targets:
                    if imgui.menu_item(self._tl_label(t_num))[0]:
                        self._handle_swap_timeline(t_num)
                        imgui.close_current_popup()
                imgui.end_menu()
                
            imgui.separator()

            # Axis Assignment submenu
            if imgui.begin_menu("Assign Axis"):
                current_axis = self._get_axis_label()
                for fa in FunscriptAxis:
                    is_selected = (fa.value == current_axis)
                    tcode = AXIS_TCODE.get(fa, "")
                    label = f"{fa.value.capitalize()} ({tcode})" if tcode else fa.value.capitalize()
                    if imgui.menu_item(label, selected=is_selected)[0]:
                        self._set_axis_assignment(fa.value)
                        imgui.close_current_popup()
                imgui.end_menu()

            # --- Bookmarks ---
            imgui.separator()
            if imgui.menu_item("Add Bookmark Here", "B")[0]:
                t, _ = getattr(self, 'new_point_candidate', (0, 0))
                self._bookmark_manager.add(t)
                imgui.close_current_popup()

            if imgui.begin_menu("Go to Bookmark"):
                for bm in self._bookmark_manager.bookmarks:
                    time_str = _format_time(self.app, bm.time_ms / 1000.0)
                    label = f"{bm.name or 'Bookmark'} ({time_str})"
                    if imgui.menu_item(label)[0]:
                        self._seek_video(bm.time_ms)
                        imgui.close_current_popup()
                if not self._bookmark_manager.bookmarks:
                    imgui.menu_item("(no bookmarks)", enabled=False)
                imgui.end_menu()

            if imgui.begin_menu("Rename Bookmark"):
                for bm in self._bookmark_manager.bookmarks:
                    time_str = _format_time(self.app, bm.time_ms / 1000.0)
                    label = f"{bm.name or 'Bookmark'} ({time_str})##{bm.id}"
                    if imgui.menu_item(label)[0]:
                        self._bookmark_rename_id = bm.id
                        self._bookmark_rename_buf = bm.name
                        imgui.close_current_popup()
                if not self._bookmark_manager.bookmarks:
                    imgui.menu_item("(no bookmarks)", enabled=False)
                imgui.end_menu()

            if imgui.begin_menu("Delete Bookmark"):
                for bm in list(self._bookmark_manager.bookmarks):
                    time_str = _format_time(self.app, bm.time_ms / 1000.0)
                    label = f"{bm.name or 'Bookmark'} ({time_str})##{bm.id}_del"
                    if imgui.menu_item(label)[0]:
                        self._bookmark_manager.remove(bm.id)
                        imgui.close_current_popup()
                if not self._bookmark_manager.bookmarks:
                    imgui.menu_item("(no bookmarks)", enabled=False)
                imgui.end_menu()

            if self._bookmark_manager.bookmarks:
                if imgui.menu_item("Clear All Bookmarks")[0]:
                    self._bookmark_manager.clear()
                    imgui.close_current_popup()

            # --- Pattern Library (Patreon) ---
            if _is_feature_available("patreon_features"):
                imgui.separator()
                if self.multi_selected_action_indices and len(self.multi_selected_action_indices) >= 2:
                    if imgui.menu_item("Save Selection as Pattern")[0]:
                        actions = self._get_actions()
                        sel_actions = [actions[i] for i in sorted(self.multi_selected_action_indices)
                                       if i < len(actions)]
                        if len(sel_actions) >= 2:
                            pattern_lib = getattr(self.app, 'pattern_library', None)
                            if pattern_lib:
                                pattern_lib.save_pattern(f"pattern_{int(time.time())}", sel_actions)
                        imgui.close_current_popup()

                if imgui.begin_menu("Apply Pattern"):
                    pattern_lib = getattr(self.app, 'pattern_library', None)
                    if pattern_lib:
                        for p_name in pattern_lib.list_patterns():
                            if imgui.menu_item(p_name)[0]:
                                pattern = pattern_lib.load_pattern(p_name)
                                if pattern:
                                    t, _ = getattr(self, 'new_point_candidate', (0, 0))
                                    new_actions = pattern_lib.apply_pattern(pattern, t)
                                    if new_actions:
                                        self._record_timeline_action()
                                        actions = self._get_actions()
                                        actions.extend(new_actions)
                                        actions.sort(key=lambda a: a['at'])
                                        fs, axis = self._get_target_funscript_details()
                                        if fs and axis:
                                            fs.set_axis_actions(axis, actions)
                                            self._finalize_action_and_update_ui()
                                imgui.close_current_popup()
                    else:
                        imgui.menu_item("(library not loaded)", enabled=False)
                    imgui.end_menu()

                # Generate Axis submenu
                if imgui.begin_menu("Generate Axis"):
                    for axis_name in ['roll', 'pitch', 'twist', 'sway', 'surge']:
                        if imgui.menu_item(axis_name.capitalize())[0]:
                            self._trigger_multi_axis_generation(axis_name)
                            imgui.close_current_popup()
                    imgui.end_menu()

            # Allow plugins to inject menu items via plugin_renderer
            self._render_plugin_selection_menu()

            imgui.end_popup()
            
    def _render_plugin_selection_menu(self):
        imgui.separator()
        if imgui.begin_menu("Run Plugin"):
             fs, axis = self._get_target_funscript_details()
             self.app.funscript_processor # Pass context if needed
             # Render simplified plugin list from manager
             available = self.plugin_renderer.plugin_manager.get_available_plugins()
             for p_name in available:
                 if imgui.menu_item(p_name)[0]:
                     # Trigger plugin context with selection
                     self.plugin_renderer.plugin_manager.set_plugin_state(p_name, PluginUIState.OPEN)
                     # Enable apply_to_selection since this was triggered from selection menu
                     context = self.plugin_renderer.plugin_manager.plugin_contexts.get(p_name)
                     if context:
                         context.apply_to_selection = True
                         self.logger.info(f"Auto-enabled 'apply to selection' for {p_name} (triggered from context menu)")
             imgui.end_menu()

    # ==================================================================================
    # DATA MODIFICATION HELPERS
    # ==================================================================================

    def _record_timeline_action(self):
        """Convenience wrapper for undo recording."""
        self.app.funscript_processor._record_timeline_action(self.timeline_num, "Edit")

    def _finalize_action_and_update_ui(self):
        """Convenience wrapper for undo finalization."""
        self.app.funscript_processor._finalize_action_and_update_ui(self.timeline_num, "Edit")
        self.invalidate_cache()

    def _trigger_multi_axis_generation(self, target_axis: str):
        """Trigger multi-axis generation for the given axis via the plugin system."""
        try:
            fs, _ = self._get_target_funscript_details()
            if not fs:
                return
            self._record_timeline_action()
            fs.apply_plugin("Multi-Axis Generator", axis='primary',
                           target_axis=target_axis, generation_mode='heuristic')
            self._finalize_action_and_update_ui()
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Multi-axis generation failed: {e}")

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

    def _set_axis_assignment(self, axis_name: str):
        """Assign a semantic axis name to this timeline."""
        funscript_obj = None
        # TrackerManager.funscript is always available after app init
        if self.app and self.app.tracker and hasattr(self.app.tracker, 'funscript'):
            funscript_obj = self.app.tracker.funscript
        if funscript_obj and hasattr(funscript_obj, 'assign_axis'):
            funscript_obj.assign_axis(self.timeline_num, axis_name)
            self.app.project_manager.project_dirty = True
            self.app.logger.info(f"T{self.timeline_num} assigned to axis: {axis_name}")

    def _get_axis_label(self) -> str:
        """Return the semantic axis name for this timeline (e.g. 'stroke', 'roll', 'pitch')."""
        funscript_obj = None
        # TrackerManager.funscript is always available after app init
        if self.app and self.app.tracker and hasattr(self.app.tracker, 'funscript'):
            funscript_obj = self.app.tracker.funscript
        if funscript_obj and hasattr(funscript_obj, 'get_axis_for_timeline'):
            return funscript_obj.get_axis_for_timeline(self.timeline_num)
        defaults = {1: "stroke", 2: "roll"}
        return defaults.get(self.timeline_num, f"axis_{self.timeline_num}")

    def _axis_label_for(self, t_num: int) -> str:
        """Return axis label for any timeline number."""
        if self.app and self.app.tracker and hasattr(self.app.tracker, 'funscript'):
            funscript_obj = self.app.tracker.funscript
            if hasattr(funscript_obj, 'get_axis_for_timeline'):
                return funscript_obj.get_axis_for_timeline(t_num)
        defaults = {1: "stroke", 2: "roll"}
        return defaults.get(t_num, f"axis_{t_num}")

    def _tl_label(self, t_num: int) -> str:
        """Short timeline label with axis, e.g. 'T1 (stroke)'."""
        axis = self._axis_label_for(t_num)
        return f"T{t_num} ({axis})" if axis else f"T{t_num}"

    def _get_timeline_status_text(self) -> str:
        """Generate status text showing timeline info (filename, axis, status)."""
        fs, axis = self._get_target_funscript_details()

        # Timeline number with semantic axis name
        axis_label = self._get_axis_label()
        parts = [f"T{self.timeline_num}: {axis_label}"]

        # Axis name (internal)
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
                # Create temp lightweight object for non-destructive preview
                # copy.deepcopy fails on RLock objects in the full Funscript instance
                from funscript.multi_axis_funscript import MultiAxisFunscript
                temp = MultiAxisFunscript()
                # Manually copy only the necessary data lists
                import copy
                temp.primary_actions = copy.deepcopy(fs.primary_actions)
                temp.secondary_actions = copy.deepcopy(fs.secondary_actions)

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
        if getattr(processor, 'live_capture_active', False):
            # Live capture pauses the processor but still updates current_frame_index
            is_playing = True
        elif hasattr(processor, 'is_playing'):
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
        # if self.timeline_num == 1 and (is_playing or forced):
        #     print(f"[TL Sync] is_playing={is_playing}, forced={forced}, interaction_active={app_state.timeline_interaction_active}, current_frame={processor.current_frame_index}")

        # Auto-scroll during playback (ignore interaction flag when playing)
        # Only respect interaction flag when forced sync (manual seeking while paused)

        # CRITICAL: Do not consume the forced sync flag if a seek is still in progress.
        # The processor frame index might be stale (pre-seek), causing us to sync to the WRONG time
        # and then turn off the flag, effectively cancelling the jump visual.
        seek_in_progress = getattr(processor, 'seek_in_progress', False)

        should_sync = is_playing or (forced and not app_state.timeline_interaction_active)

        if should_sync:
            # Guard against division by zero when fps not yet available (e.g., loading from Stash WebView)
            if not processor.fps or processor.fps <= 0:
                return

            # If seeking, we might want to wait, but if we sync, we sync to current reported frame
            current_ms = (processor.current_frame_index / processor.fps) * 1000.0

            # Center the playhead
            center_offset = (tf.width * tf.zoom) / 2
            target_pan = current_ms - center_offset

            app_state.timeline_pan_offset_ms = target_pan

            # Only clear the forced flag if we are NOT waiting for a seek to complete
            if forced and not seek_in_progress:
                app_state.force_timeline_pan_to_current_frame = False
