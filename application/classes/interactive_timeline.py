import imgui
from application.utils.imgui_helpers import center_next_window_pivot
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
from application.utils.bpm_analyzer import BPMOverlayConfig, TapTempo, SUBDIVISION_LABELS, SUBDIVISION_VALUES
from application.classes.bookmark_manager import BookmarkManager
from application.classes.recording_capture import RecordingCapture
from common.frame_utils import ms_to_frame, frame_to_ms
from application.classes.timeline_ops import (
    add_point, clear_all_points,
    copy_selection, copy_to_other,
    paste_actions, paste_actions_exact, paste_actions_replace,
    delete_selected, equalize_selection, filter_selection,
    isolate_selection,
    begin_live_drag, end_live_drag, live_range_extend, live_rdp_simplify,
    nudge_all_time, nudge_chapter_time,
    nudge_selection_time, nudge_selection_value,
    repeat_last_stroke,
    select_points_in_chapter, select_relative_to_playhead,
    snap_to_playhead,
)
from application.classes.timeline.drawing_mixin import DrawingMixin

class TimelineTransformer:
    """
    Handles coordinate transformations between Time/Value space and Screen Pixel space.
    Optimized with vectorization support.
    """
    # Vertical padding keeps points at 0/100 fully visible inside the canvas
    POINT_PADDING = 6.0

    def __init__(self, pos: Tuple[float, float], size: Tuple[float, float],
                 pan_ms: float, zoom_ms_px: float):
        self.x_offset = pos[0]
        self.y_offset = pos[1]
        self.width = size[0]
        self.height = size[1]
        self.pan_ms = pan_ms
        self.zoom = max(0.001, zoom_ms_px) # Prevent div by zero

        # Usable vertical range after padding
        self._pad = self.POINT_PADDING
        self._usable_h = max(1.0, self.height - 2 * self._pad)

        # Calculate visible time range
        self.visible_start_ms = pan_ms
        self.visible_end_ms = pan_ms + (self.width * self.zoom)

    def time_to_x(self, t_ms: float) -> float:
        return self.x_offset + (t_ms - self.pan_ms) / self.zoom

    def val_to_y(self, val: float) -> float:
        return self.y_offset + self._pad + self._usable_h * (1.0 - (val / 100.0))

    def x_to_time(self, x: float) -> float:
        return (x - self.x_offset) * self.zoom + self.pan_ms

    def y_to_val(self, y: float) -> int:
        if self._usable_h <= 0: return 0
        val = (1.0 - (y - self.y_offset - self._pad) / self._usable_h) * 100.0
        return max(0, min(100, int(round(val))))

    # Vectorized versions for numpy arrays (Rendering path)
    def vec_time_to_x(self, times: np.ndarray) -> np.ndarray:
        return self.x_offset + (times - self.pan_ms) / self.zoom

    def vec_val_to_y(self, vals: np.ndarray) -> np.ndarray:
        return self.y_offset + self._pad + self._usable_h * (1.0 - (vals / 100.0))


class InteractiveFunscriptTimeline(DrawingMixin):
    def __init__(self, app_instance, timeline_num: int):
        self.app = app_instance
        self.timeline_num = timeline_num
        self.logger = getattr(app_instance, 'logger', None)
        # Cache per-instance attribute names so render() doesn't f-string them.
        self._visibility_attr = (
            "show_funscript_interactive_timeline"
            if timeline_num == 1
            else f"show_funscript_interactive_timeline{timeline_num}"
        )

        # --- Selection & Interaction State ---
        self.selected_action_idx: int = -1
        self.multi_selected_action_indices: Set[Tuple[int, int]] = set()  # (at, pos) identity tuples
        # Cache of the resolved indices list; see _resolve_selected_indices.
        self._resolve_selected_cache: Optional[Tuple[Tuple[int, int, int], List[int]]] = None

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

        self.is_hovered: bool = False  # Set each frame; read by status strip hints
        self._alt_arrow_panning: bool = False  # Track Alt+Arrow pan for seek-on-release
        self._hovered_point_idx: int = -1  # For hover tooltip stats

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
        # Reference funscript comparison overlay
        self.reference_overlay_actions: Optional[List[Dict]] = None
        self.reference_overlay_name: str = ""
        self._reference_metrics: Optional[Dict] = None
        self._reference_metrics_dirty: bool = False
        self._reference_peak_matches: Optional[list] = None  # Cached peak match data
        self._reference_unmatched_main: Optional[list] = None
        self._reference_unmatched_ref: Optional[list] = None
        self._reference_problem_sections: Optional[list] = None

        # Settings
        self.shift_frames_amount = 1
        self.nudge_chapter_only = False  # When True, << >> only affect points in selected chapter
        self._container_mode = False  # Set by render() when inside scrollable container

        # --- Visualization Features ---

        # Heatmap coloring - speed-based segment colors enabled by default
        self._show_heatmap_coloring = True
        _settings = getattr(self.app, 'app_settings', None)
        # Catmull-rom spline between points. Toggle: View menu or C.
        self._show_smooth_curve = bool(
            _settings.get("timeline_smooth_curve", True)) if _settings else True
        self._smooth_samples_per_segment = 12
        _max_speed = float(_settings.get("heatmap_max_speed", 400.0)) if _settings else 400.0
        _highlight = bool(_settings.get("heatmap_highlight_overspeed", True)) if _settings else True
        self._heatmap_mapper = HeatmapColorMapper(max_speed=_max_speed,
                                                   highlight_overspeed=_highlight)
        self._heatmap_speeds_cache: Optional[np.ndarray] = None   # full-script speeds
        self._heatmap_colors_cache: Optional[np.ndarray] = None   # full-script u32 colors
        self._heatmap_cache_np_id: int = 0  # id() of numpy arrays when cache was built

        # Speed limit visualization
        self._show_speed_warnings = False
        self._speed_limit_threshold = 400.0

        # Selection bracket loop playback (A-B style). When armed, the playback
        # tick wraps from the last selected action's time back to the first.
        self._selection_loop_armed: bool = False

        # Live-drag tool state (popup-driven). Opened from the Rendering menu;
        # _live_drag_state holds {op_key, baseline, axis} during a drag so each
        # tick re-applies from the original snapshot, not the previous tick.
        self._live_tool_open: Optional[str] = None  # 'range' | 'rdp' | None
        self._live_tool_value: float = 0.0
        self._live_drag_state: Optional[dict] = None

        # Bookmarks. Mutations dirty the project so autosave + exit-save persist them.
        self._bookmark_manager = BookmarkManager(on_change=self._mark_project_dirty)
        self._bookmark_rename_id: Optional[str] = None
        self._bookmark_rename_buf: str = ""

        # Timeline Mode State Machine
        self._mode = TimelineMode.SELECT
        self._interaction_state = TimelineInteractionState.IDLE

        # Alternating Mode
        self._alt_next_is_top = True
        self._alt_top_value = 95
        self._alt_bottom_value = 5

        # Waveform draw cache, skip recomputation when viewport unchanged
        self._waveform_cache_key: Optional[tuple] = None  # (start_ms, end_ms, width)
        self._waveform_cache_xs = None
        self._waveform_cache_ys_top = None
        self._waveform_cache_ys_bot = None
        self._waveform_cache_step: int = 0

        # Pan-seek throttle for continuous video updates during middle-mouse drag
        self._last_pan_seek_time: float = 0.0
        self._is_modifier_panning: bool = False

        # Recording Mode
        self._recording_capture: Optional[RecordingCapture] = None
        self._recording_rdp_epsilon = 2.0

        # Controller live-scripting (gamepad input for recording mode)
        self._gamepad_input = None           # Lazy-init GamepadInput
        self._gamepad_connected: bool = False
        self._controller_device_preview: bool = False
        self._controller_input_delay_ms: int = 0
        self._calibration = None             # CalibrationRoutine when active
        self._show_controller_settings: bool = False

        # BPM/Tempo Overlay
        self._bpm_config: Optional[BPMOverlayConfig] = None
        self._tap_tempo = TapTempo()

    # ==================================================================================
    # CORE DATA HELPERS
    # ==================================================================================

    def _mark_project_dirty(self) -> None:
        """Mark the project as having unsaved changes (autosave + exit-save gate on this)."""
        pm = getattr(self.app, 'project_manager', None)
        if pm is not None:
            pm.project_dirty = True

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

    def _get_cached_numpy_arrays(self):
        """Return cached (ats_np, poss_np) float32 arrays for this timeline's axis."""
        fs, axis = self._get_target_funscript_details()
        if fs and axis:
            return fs._get_numpy_arrays_for_axis(axis)
        return None, None

    def _action_key(self, action: dict) -> Tuple[int, int]:
        """Return the identity tuple for a funscript action."""
        return (action['at'], action['pos'])

    def _resolve_selected_indices(self) -> List[int]:
        """Convert selection tuples to current valid indices in the action list."""
        actions = self._get_actions()
        sel_set = self.multi_selected_action_indices
        if not actions or not sel_set:
            return []
        # Result cache keyed on (actions_len, sel_len, sel_set_id). Hits every
        # call during a single render frame, and typically for long stretches
        # until the user edits either side.
        cache_key = (len(actions), len(sel_set), id(sel_set))
        cached = self._resolve_selected_cache
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        indices = [i for i, a in enumerate(actions)
                   if (a['at'], a['pos']) in sel_set]
        # Already in index order; no sort needed.
        self._resolve_selected_cache = (cache_key, indices)
        return indices

    def _invalidate_selection_cache(self) -> None:
        """Clear the resolve-indices cache. Call when the selection set mutates."""
        self._resolve_selected_cache = None

    def invalidate_cache(self):
        """Forces updates on next frame"""
        self._heatmap_speeds_cache = None
        self._heatmap_colors_cache = None
        if self.reference_overlay_actions:
            self._reference_metrics_dirty = True
            self._reference_metrics_dirty_time = time.monotonic()

    # ==================================================================================
    # MAIN RENDER LOOP
    # ==================================================================================

    def render(self, y_pos: float = 0, height: float = 0, container_mode: bool = False,
               **kwargs):
        app_state = self.app.app_state_ui
        visibility_attr = self._visibility_attr

        if not getattr(app_state, visibility_attr, False):
            return

        self._container_mode = container_mode

        # Selection-loop tick: enforced once per render frame (cheap, no-op
        # when not armed). Wraps playhead at end of selection back to start.
        self._tick_selection_loop()

        # 1. Window Configuration
        is_floating = app_state.ui_layout_mode == "floating"
        # NO_BRING_TO_FRONT_ON_FOCUS prevents the timeline from stealing
        # z-order when clicked, dialog/plugin windows stay on top.
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
            window_title = f"FunGen: Funscript {self.timeline_num}: {axis_label}" if axis_label else f"FunGen: Funscript {self.timeline_num}"
            is_open, visible = imgui.begin(window_title, True, flags)
            setattr(app_state, visibility_attr, visible)
            if not is_open:
                imgui.end()
                return

        # 2. Render Toolbar (Buttons)
        self._render_toolbar()

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

        # Empty state hint
        if not main_actions:
            hint = "Click to add points  |  0-9 keys set position  |  M snaps to playhead"
            hint_size = imgui.calc_text_size(hint)
            hx = canvas_pos[0] + (canvas_size[0] - hint_size[0]) * 0.5
            hy = canvas_pos[1] + canvas_size[1] * 0.3
            draw_list.add_text(hx, hy, imgui.get_color_u32_rgba(*TimelineColors.EMPTY_HINT), hint)

        # 6-pre. Speed limit overlay (behind curve)
        if self._show_speed_warnings and main_actions:
            self._draw_speed_limit_overlay(draw_list, tf, main_actions)

        # 6-ch. Chapter highlight overlay
        self._draw_chapter_highlight_overlay(draw_list, tf)

        # 6-pre2. BPM/Tempo grid overlay
        if self._bpm_config:
            self._draw_bpm_grid(draw_list, tf)

        # 6-ref. Draw Reference Funscript Overlay (if loaded)
        if self.reference_overlay_actions:
            self._draw_curve(draw_list, tf, self.reference_overlay_actions,
                             color_override=TimelineColors.REFERENCE_OVERLAY,
                             force_lines_only=True, alpha=0.8)
            self._draw_reference_peak_markers(draw_list, tf)
            self._draw_reference_problem_bands(draw_list, tf)


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

        # 6f. Reference comparison metrics (rendered in info_graphs_ui, not on canvas)

        # 7. Render Plugin Windows (Popups)
        self.plugin_renderer.render_plugin_windows(self.timeline_num, f"TL{self.timeline_num}")

        # 7b. Check for and execute pending plugin apply requests
        self._check_and_apply_pending_plugins()

        # 8. Handle Auto-Scroll/Sync
        self._handle_sync_logic(app_state, tf)

        # 9. Draw Active/Read-only State Border
        self._draw_state_border(draw_list, canvas_pos, canvas_size, app_state)

        imgui.end_child() if self._container_mode else imgui.end()

        # Live-drag tool popup (rendered after the main timeline so it floats
        # above without z-order tricks).
        self._render_live_tool_popup()

    # ==================================================================================
    # INPUT HANDLING
    # ==================================================================================

    def _handle_input(self, app_state, tf: TimelineTransformer):
        io = imgui.get_io()
        mouse_pos = imgui.get_mouse_pos()

        # Check canvas bounds AND that no other window/popup/dialog is on top.
        # is_window_hovered() returns False when a popup, dialog, or overlapping
        # window is above this one, preventing click-through to the canvas.
        in_canvas = (tf.x_offset <= mouse_pos[0] <= tf.x_offset + tf.width and
                     tf.y_offset <= mouse_pos[1] <= tf.y_offset + tf.height)
        is_hovered = in_canvas and imgui.is_window_hovered()
        self.is_hovered = is_hovered
        is_focused = imgui.is_window_focused(imgui.FOCUS_ROOT_AND_CHILD_WINDOWS)

        # Update active timeline ONLY on explicit user interaction (click)
        # This prevents the last-rendered timeline from stealing focus on startup
        if is_hovered and imgui.is_mouse_clicked(0):  # Left click
            app_state.active_timeline_num = self.timeline_num

        # --- Keyboard Shortcuts (active timeline, not dependent on imgui window focus) ---
        is_active_timeline = (app_state.active_timeline_num == self.timeline_num)
        if is_active_timeline and not io.want_text_input:
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
                # Wheel events have no release; if we set timeline_interaction_active
                # here it never clears and blocks subsequent forced syncs.

            # Double middle-click clears selection (fast deselect gesture).
            if imgui.is_mouse_double_clicked(glfw.MOUSE_BUTTON_MIDDLE):
                if self.multi_selected_action_indices:
                    self.multi_selected_action_indices.clear()
                    self.selected_action_idx = -1

            # Middle-drag pan: seek only when the target is in the nav cache,
            # otherwise defer to mouse-release to avoid decode stutter.
            if imgui.is_mouse_dragging(glfw.MOUSE_BUTTON_MIDDLE):
                delta_x = io.mouse_delta[0]
                app_state.timeline_pan_offset_ms -= delta_x * tf.zoom
                app_state.timeline_interaction_active = True
                center_ms = tf.x_to_time(tf.x_offset + tf.width / 2) - delta_x * tf.zoom
                self._seek_if_cached(center_ms)

        # --- Mode-specific input dispatch ---
        if self._mode == TimelineMode.ALTERNATING:
            self._handle_alternating_mode_input(app_state, tf, mouse_pos, is_hovered, io)
        elif self._mode == TimelineMode.RECORDING:
            self._handle_recording_mode_input(app_state, tf, mouse_pos, is_hovered, io)
        elif self._mode == TimelineMode.INJECTION and _is_feature_available("patreon_features"):
            self._handle_injection_mode_input(app_state, tf, mouse_pos, is_hovered, io)

        # --- Action Interaction (SELECT mode or fallback) ---
        actions = self._get_actions()

        # --- Hover Tooltip ---
        if is_hovered and not self.is_dragging_active and not self.is_marqueeing and actions:
            hover_idx = self._hit_test_point(mouse_pos, actions, tf)
            self._hovered_point_idx = hover_idx
            if hover_idx != -1:
                act = actions[hover_idx]
                if hover_idx > 0:
                    prev = actions[hover_idx - 1]
                    dt = act['at'] - prev['at']
                    pos_delta = act['pos'] - prev['pos']
                    abs_delta = abs(pos_delta)
                    speed = abs_delta / (dt / 1000.0) if dt > 0 else 0.0
                    arrow = "UP" if pos_delta > 0 else ("DN" if pos_delta < 0 else "--")
                    imgui.set_tooltip(
                        f"{prev['pos']} -> {act['pos']} = {abs_delta} {arrow}\n"
                        f"Interval: {dt:.0f} ms\n"
                        f"Speed: {speed:.0f} units/s"
                    )
                else:
                    imgui.set_tooltip(f"{act['pos']} @{act['at']:.0f}ms (first point)")
        else:
            self._hovered_point_idx = -1

        # Double-click: always seek to time (regardless of point hit)
        if is_hovered and imgui.is_mouse_double_clicked(glfw.MOUSE_BUTTON_LEFT) and self._mode == TimelineMode.SELECT:
            hit_idx = self._hit_test_point(mouse_pos, actions, tf)
            if hit_idx != -1:
                # Double-click on point: select and seek
                self.multi_selected_action_indices = {self._action_key(actions[hit_idx])}
                self.selected_action_idx = hit_idx
                self._seek_video(actions[hit_idx]['at'])
            else:
                # Double-click on empty space: seek to clicked time
                self._seek_video(tf.x_to_time(mouse_pos[0]))
            # Cancel any pending single-click actions
            self.range_selecting = False
            self.is_marqueeing = False

        # Left Click (single, handled after double-click check)
        elif is_hovered and imgui.is_mouse_clicked(glfw.MOUSE_BUTTON_LEFT) and self._mode == TimelineMode.SELECT:
            hit_idx = self._hit_test_point(mouse_pos, actions, tf)

            if self._is_pan_modifier_held(io):
                # Pan modifier + Drag = Pan (trackpad alternative to middle-mouse)
                self._is_modifier_panning = True

            elif self._is_create_modifier_held(io) and hit_idx == -1:
                # Create-point modifier + click on empty: add new point at cursor
                t = tf.x_to_time(mouse_pos[0])
                v = tf.y_to_val(mouse_pos[1])
                add_point(self, t, v)

            elif hit_idx != -1:
                # Point Clicked, select only (no seek on single click)
                self.dragging_action_idx = hit_idx
                self.drag_start_pos = mouse_pos
                self.is_dragging_active = False  # Wait for drag threshold
                self.drag_undo_recorded = False

                # Multi-select toggle: create-modifier or marquee-modifier on a point
                if self._is_create_modifier_held(io) or self._is_marquee_modifier_held(io):
                    key = self._action_key(actions[hit_idx])
                    if key in self.multi_selected_action_indices:
                        self.multi_selected_action_indices.discard(key)
                    else:
                        self.multi_selected_action_indices.add(key)
                else:
                    # Plain click: select only this point (keep if already in multi-sel for drag)
                    key = self._action_key(actions[hit_idx])
                    if key not in self.multi_selected_action_indices:
                        self.multi_selected_action_indices.clear()
                        self.multi_selected_action_indices.add(key)

                self.selected_action_idx = hit_idx

            elif self._is_marquee_modifier_held(io):
                # Marquee modifier + drag on empty: box select (2D rectangle)
                self.is_marqueeing = True
                self.marquee_start = mouse_pos
                self.marquee_end = mouse_pos

            else:
                # Click on empty space: start range selection (time-based, full height)
                self.multi_selected_action_indices.clear()
                self.selected_action_idx = -1
                self.range_selecting = True
                self.range_start_time = tf.x_to_time(mouse_pos[0])
                self.range_end_time = self.range_start_time

        # --- Dragging Processing ---
        if imgui.is_mouse_dragging(glfw.MOUSE_BUTTON_LEFT):

            if self._is_modifier_panning:
                delta_x = io.mouse_delta[0]
                app_state.timeline_pan_offset_ms -= delta_x * tf.zoom
                app_state.timeline_interaction_active = True

            elif self.dragging_action_idx != -1:
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
                # Edge autoscroll: when marquee drags within 3% of canvas edge,
                # pan timeline so the selection can extend off-screen.
                edge_pct = 0.03
                edge_px = max(8.0, tf.width * edge_pct)
                pan_speed = max(8.0, tf.zoom * 4.0)
                if mouse_pos[0] < tf.x_offset + edge_px:
                    app_state.timeline_pan_offset_ms -= pan_speed
                elif mouse_pos[0] > tf.x_offset + tf.width - edge_px:
                    app_state.timeline_pan_offset_ms += pan_speed
                
            elif self.range_selecting:
                self.range_end_time = tf.x_to_time(mouse_pos[0])
                app_state.timeline_interaction_active = True

        # --- Mouse Release ---
        if imgui.is_mouse_released(glfw.MOUSE_BUTTON_LEFT):
            if self._is_modifier_panning:
                self._is_modifier_panning = False
                app_state.timeline_interaction_active = False
                self._seek_pan_center_with_sync(app_state)
            else:
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
            self._seek_pan_center_with_sync(app_state)

        # --- Context Menu ---
        if is_hovered and imgui.is_mouse_clicked(glfw.MOUSE_BUTTON_RIGHT):
            hit_idx = self._hit_test_point(mouse_pos, actions, tf)
            self.context_menu_target_idx = hit_idx
            
            # Auto-select target if not already selected
            if hit_idx != -1:
                key = self._action_key(actions[hit_idx])
                if key not in self.multi_selected_action_indices:
                    self.multi_selected_action_indices = {key}
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

        center_next_window_pivot()
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
        if not self.app.shortcut_manager.should_handle_shortcuts():
            return
        shortcuts = self.app.app_settings.get("funscript_editor_shortcuts", {})
        
        # Helper to map shortcuts (for single-press actions). Pass repeat=True
        # to fire continuously while the key is held (matches the standard
        # editor behaviour for nudge / step actions).
        def check_shortcut(name, default, repeat=False):
            key_str = shortcuts.get(name, default)
            tuple_key = self.app._map_shortcut_to_glfw_key(key_str)
            if not tuple_key: return False
            key_code, mods = tuple_key

            try:
                pressed = imgui.is_key_pressed(key_code, repeat)
            except TypeError:
                pressed = imgui.is_key_pressed(key_code)
            match = (mods["ctrl"] == io.key_ctrl and
                     mods["alt"] == io.key_alt and
                     mods["shift"] == io.key_shift and
                     mods["super"] == io.key_super)
            return pressed and match

        # Helper for persistent/held key actions (like panning)
        def check_key_held(name, default):
            key_str = shortcuts.get(name, default)
            tuple_key = self.app._map_shortcut_to_glfw_key(key_str)
            if not tuple_key: return False
            key_code, mods = tuple_key

            held = imgui.is_key_down(key_code)
            match = (mods["ctrl"] == io.key_ctrl and
                     mods["alt"] == io.key_alt and
                     mods["shift"] == io.key_shift and
                     mods["super"] == io.key_super)
            return held and match

        # 1. Pan Left/Right (Arrow keys) - persistent while held, seek on release
        pan_speed = self.app.app_settings.config.ui.timeline_pan_speed_multiplier * app_state.timeline_zoom_factor_ms_per_px
        panning_now = False
        if check_key_held("pan_timeline_left", "ALT+LEFT_ARROW"):
            app_state.timeline_pan_offset_ms -= pan_speed
            panning_now = True
        if check_key_held("pan_timeline_right", "ALT+RIGHT_ARROW"):
            app_state.timeline_pan_offset_ms += pan_speed
            panning_now = True

        if panning_now:
            self._alt_arrow_panning = True
        elif self._alt_arrow_panning:
            # Just released, seek to center of visible timeline
            self._alt_arrow_panning = False
            if self.app.processor and self.app.processor.fps > 0:
                # Estimate visible width from window width (timeline spans most of it)
                tl_width_px = max(200, app_state.window_width - 50)
                visible_width_ms = tl_width_px * app_state.timeline_zoom_factor_ms_per_px
                center_ms = app_state.timeline_pan_offset_ms + visible_width_ms / 2
                target_frame = max(0, ms_to_frame(center_ms, self.app.processor.fps))
                if self.app.processor.total_frames > 0:
                    target_frame = min(target_frame, self.app.processor.total_frames - 1)
                self.app.event_handlers.seek_video_with_sync(target_frame)

        # 2. Select All / Deselect All
        if check_shortcut("select_all_points", "CTRL+A"):
            actions = self._get_actions()
            self.multi_selected_action_indices = {self._action_key(a) for a in actions}
        if check_shortcut("deselect_all_points", "SUPER+D"):
            self.multi_selected_action_indices.clear()

        # 3. Delete (Delete/Backspace)
        if check_shortcut("delete_selected_point", "DELETE") or check_shortcut("delete_selected_point_alt", "BACKSPACE"):
            if not self.multi_selected_action_indices:
                nearest = self._find_nearest_point_index()
                if nearest is not None and nearest >= 0:
                    actions = self._get_actions()
                    if 0 <= nearest < len(actions):
                        self.multi_selected_action_indices = {self._action_key(actions[nearest])}
            delete_selected(self)

        # 4. Copy/Paste
        if check_shortcut("copy_selection", "CTRL+C"):
            copy_selection(self)

        def _playhead_paste_ms():
            proc = self.app.processor
            if proc and proc.fps > 0:
                ph = getattr(proc, 'playhead_override_ms', None)
                if ph is None:
                    ph = frame_to_ms(proc.current_frame_index, proc.fps)
                return ph
            return 0

        if check_shortcut("paste_selection", "CTRL+V"):
            paste_actions(self, _playhead_paste_ms())
        if check_shortcut("paste_selection_replace", "CTRL+SHIFT+V"):
            paste_actions_replace(self, _playhead_paste_ms())
        if check_shortcut("paste_selection_exact", "CTRL+ALT+V"):
            paste_actions_exact(self)

        # 5. Nudge Selection (Arrows)
        nudge_val = 0
        if check_shortcut("nudge_selection_pos_up", "SHIFT+UP_ARROW", repeat=True): nudge_val = 1
        if check_shortcut("nudge_selection_pos_down", "SHIFT+DOWN_ARROW", repeat=True): nudge_val = -1
        
        if nudge_val != 0:
            if not self.multi_selected_action_indices:
                nearest = self._find_nearest_point_index()
                if nearest is not None and nearest >= 0:
                    actions = self._get_actions()
                    if 0 <= nearest < len(actions):
                        self.multi_selected_action_indices = {self._action_key(actions[nearest])}
            if self.multi_selected_action_indices:
                nudge_selection_value(self, nudge_val)

        # 6. Nudge Time (Shift+Arrows)
        nudge_t = 0
        cfg = self._bpm_config
        if cfg and cfg.snap_to_beat and cfg.bpm > 0:
            snap_t = cfg.beat_interval_ms
        else:
            snap_t = app_state.snap_to_grid_time_ms if app_state.snap_to_grid_time_ms > 0 else 20
        if check_shortcut("nudge_selection_time_prev", "SHIFT+LEFT_ARROW", repeat=True): nudge_t = -snap_t
        if check_shortcut("nudge_selection_time_next", "SHIFT+RIGHT_ARROW", repeat=True): nudge_t = snap_t

        if nudge_t != 0:
            if not self.multi_selected_action_indices:
                nearest = self._find_nearest_point_index()
                if nearest is not None and nearest >= 0:
                    actions = self._get_actions()
                    if 0 <= nearest < len(actions):
                        self.multi_selected_action_indices = {self._action_key(actions[nearest])}
            if self.multi_selected_action_indices:
                nudge_selection_time(self, nudge_t)

        # 6b. Snap nearest to playhead - handled by global shortcut handler

        # 6c. Select all left / right of playhead
        if check_shortcut("select_left_of_playhead", "CTRL+ALT+LEFT_ARROW"):
            select_relative_to_playhead(self, before=True)
        if check_shortcut("select_right_of_playhead", "CTRL+ALT+RIGHT_ARROW"):
            select_relative_to_playhead(self, before=False)

        # 6d. Repeat last stroke at playhead (Home)
        if check_shortcut("repeat_last_stroke", "HOME"):
            repeat_last_stroke(self)

        # 6e. Selection-aware ops (SHIFT+ convention)
        if check_shortcut("equalize_selection", "SHIFT+E"):
            equalize_selection(self)
        if check_shortcut("isolate_selection", "SHIFT+R"):
            isolate_selection(self)
        if check_shortcut("filter_selection_top", "SHIFT+T"):
            filter_selection(self, 'top')
        if check_shortcut("filter_selection_bottom", "SHIFT+B"):
            filter_selection(self, 'bottom')
        if check_shortcut("filter_selection_mid", "SHIFT+M"):
            filter_selection(self, 'mid')
        if check_shortcut("toggle_selection_loop", "\\"):
            self._toggle_selection_loop()

        # 7. Bookmark at playhead (B key)
        if check_shortcut("add_bookmark", "B"):
            proc = self.app.processor
            if proc and proc.fps > 0:
                playhead_time = getattr(proc, 'playhead_override_ms', None)
                if playhead_time is None:
                    playhead_time = frame_to_ms(proc.current_frame_index, proc.fps)
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
            self.drag_undo_recorded = True
            # Capture original values for unified undo
            idx = self.dragging_action_idx
            self._drag_old_at = actions[idx]['at']
            self._drag_old_pos = actions[idx]['pos']

        # Calculate New Values
        t_raw = tf.x_to_time(mouse_pos[0])
        v_raw = tf.y_to_val(mouse_pos[1])
        
        # Snapping
        t_raw = self._snap_time(t_raw)
        snap_v = self.app.app_state_ui.snap_to_grid_pos
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

        # O(1) in-place cache patch; heatmap flush deferred to _finalize_drag.
        fs, axis = self._get_target_funscript_details()
        if fs:
            ax = axis or 'primary'
            patched = fs._patch_cache_entry(ax, idx, new_t, new_v) if ax in ('primary', 'secondary') else False
            if not patched:
                fs._invalidate_cache(axis or 'both')
                self.invalidate_cache()
        self.app.project_manager.project_dirty = True

    def _finalize_drag(self):
        if self.drag_undo_recorded:
            # Flush heatmap caches now that the point has settled.
            self.invalidate_cache()
            self.app.funscript_processor._post_mutation_refresh(self.timeline_num, "Drag Point")
            # New unified undo: capture final position
            actions = self._get_actions()
            idx = self.dragging_action_idx
            if actions and 0 <= idx < len(actions):
                from application.classes.undo_manager import MovePointCmd
                self.app.undo_manager.push_done(MovePointCmd(
                    self.timeline_num, idx,
                    getattr(self, '_drag_old_at', 0), getattr(self, '_drag_old_pos', 0),
                    actions[idx]['at'], actions[idx]['pos']))

    def _finalize_marquee(self, tf, actions, append: bool):
        if not self.marquee_start or not self.marquee_end: return

        # Ignore very small drags (accidental)
        dx = abs(self.marquee_end[0] - self.marquee_start[0])
        dy = abs(self.marquee_end[1] - self.marquee_start[1])
        if dx < 5 and dy < 5:
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
                new_selection.add(self._action_key(act))

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
        
        new_set = {self._action_key(actions[i]) for i in range(s_idx, e_idx)}
        if append:
            self.multi_selected_action_indices.update(new_set)
        else:
            self.multi_selected_action_indices = new_set

    def _seek_video(self, time_ms: float):
        if self.app.processor and self.app.processor.video_info:
            fps = self.app.processor.fps
            if fps > 0:
                frame = ms_to_frame(time_ms, fps)
                self.app.processor.seek_video(frame)
                self.app.app_state_ui.force_timeline_pan_to_current_frame = True

    def _is_modifier_held(self, io, setting_key: str, default: str) -> bool:
        """Check if a configured modifier key is held."""
        mod = self.app.app_settings.get(setting_key, default)
        if mod == "SHIFT": return io.key_shift
        if mod == "ALT": return io.key_alt
        if mod == "CTRL": return io.key_ctrl
        if mod == "SUPER": return io.key_super
        return False

    def _is_pan_modifier_held(self, io) -> bool:
        return self._is_modifier_held(io, "timeline_pan_drag_modifier", "ALT")

    def _is_create_modifier_held(self, io) -> bool:
        return self._is_modifier_held(io, "timeline_create_point_modifier", "SHIFT")

    def _is_marquee_modifier_held(self, io) -> bool:
        return self._is_modifier_held(io, "timeline_marquee_modifier", "CTRL")

    def _seek_video_no_pan(self, time_ms: float):
        """Drag-scrub seek: keyframe-fast so embedded mpv stays responsive."""
        if self.app.processor and self.app.processor.video_info:
            fps = self.app.processor.fps
            if fps > 0:
                frame = ms_to_frame(time_ms, fps)
                self.app.processor.seek_video(frame, accurate=False)

    def _seek_if_cached(self, time_ms: float):
        """Seek only when the target frame is in the nav cache. Used during
        middle-drag pan to keep the video in sync without decode stutter;
        cache-miss seeks are deferred to mouse-release."""
        proc = self.app.processor
        if not proc or proc.fps <= 0:
            return
        nav = getattr(proc, '_nav_cache', None)
        if nav is None:
            return
        frame = ms_to_frame(time_ms, proc.fps)
        if nav.contains(frame):
            proc.seek_video(frame)

    def _seek_pan_center_with_sync(self, app_state):
        """Same path as alt+arrow pan-release: single seek_video_with_sync."""
        proc = self.app.processor
        if not proc or proc.fps <= 0:
            return
        tl_width_px = max(200, app_state.window_width - 50)
        visible_width_ms = tl_width_px * app_state.timeline_zoom_factor_ms_per_px
        center_ms = app_state.timeline_pan_offset_ms + visible_width_ms / 2
        target_frame = max(0, ms_to_frame(center_ms, proc.fps))
        if proc.total_frames > 0:
            target_frame = min(target_frame, proc.total_frames - 1)
        self.app.event_handlers.seek_video_with_sync(target_frame)

    def _playhead_ms(self) -> Optional[int]:
        proc = self.app.processor
        if not proc or proc.fps <= 0:
            return None
        ms = getattr(proc, 'playhead_override_ms', None)
        if ms is None:
            ms = frame_to_ms(proc.current_frame_index, proc.fps)
        return ms

    def _find_nearest_point_index(self) -> Optional[int]:
        """Find the index of the action nearest to the current playhead, without moving it."""
        from bisect import bisect_left

        actions = self._get_actions()
        if not actions:
            return None

        processor = self.app.processor
        if not processor or processor.fps <= 0:
            return None
        playhead_ms = frame_to_ms(processor.current_frame_index, processor.fps)

        timestamps = [a['at'] for a in actions]
        idx = bisect_left(timestamps, playhead_ms)

        best_idx = None
        best_dist = float('inf')
        for candidate in (idx - 1, idx):
            if 0 <= candidate < len(actions):
                dist = abs(actions[candidate]['at'] - playhead_ms)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = candidate
        return best_idx

    def _handle_swap_timeline(self, target_num=None):
        if target_num is None:
            target_num = 2 if self.timeline_num == 1 else 1
        self.app.funscript_processor.swap_timelines(self.timeline_num, target_num)

    # ==================================================================================
    # VISUAL DRAWING
    # ==================================================================================

    def _load_reference_funscript(self):
        """Open file dialog to load a reference funscript for comparison overlay."""
        gi = getattr(self.app, "gui_instance", None)
        if not gi:
            return

        def _on_reference_selected(path):
            if not path or not os.path.isfile(path):
                return
            # File I/O + json parse off the UI thread; dirty flag triggers
            # next-frame recompute via the existing render-time path.
            import threading
            def _worker():
                try:
                    import json
                    with open(path, 'rb') as f:
                        raw = f.read()
                    for enc in ('utf-8', 'utf-8-sig', 'latin-1'):
                        try:
                            text = raw.decode(enc)
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        text = raw.decode('latin-1')
                    data = json.loads(text)
                    actions = sorted(data.get('actions', []), key=lambda a: a['at'])
                    if len(actions) < 2:
                        self.app.logger.warning("Reference funscript has fewer than 2 actions")
                        return
                    self.reference_overlay_actions = actions
                    self.reference_overlay_name = os.path.basename(path)
                    self._reference_metrics_dirty = True
                    self.app.logger.info(f"Loaded reference funscript: {self.reference_overlay_name} ({len(actions)} actions)")
                except Exception as e:
                    self.app.logger.error(f"Failed to load reference funscript: {e}")
            threading.Thread(target=_worker, daemon=True, name="ReferenceFunscriptLoad").start()

        gi.file_dialog.show(
            title="Load Reference Funscript",
            is_save=False,
            callback=_on_reference_selected,
            extension_filter=".funscript",
        )

    def _clear_reference_overlay(self):
        """Clear the reference funscript overlay and all cached data."""
        self.reference_overlay_actions = None
        self.reference_overlay_name = ""
        self._reference_metrics = None
        self._reference_metrics_dirty = False
        self._reference_peak_matches = None
        self._reference_unmatched_main = None
        self._reference_unmatched_ref = None
        self._reference_problem_sections = None

    def _recompute_reference_data(self):
        """Recompute peak matches and metrics for the reference overlay."""
        from application.utils.funscript_comparison import (
            detect_peaks, match_peaks, classify_match, compute_comparison_metrics,
            detect_problem_sections
        )
        main_actions = self._get_actions()
        ref_actions = self.reference_overlay_actions
        if not main_actions or not ref_actions or len(main_actions) < 2 or len(ref_actions) < 2:
            self._reference_metrics = None
            self._reference_peak_matches = None
            self._reference_unmatched_main = None
            self._reference_unmatched_ref = None
            self._reference_problem_sections = None
            return

        fps = 30.0
        proc = self.app.processor
        if proc and proc.fps and proc.fps > 0:
            fps = proc.fps

        # Peak matching
        main_peaks = detect_peaks(main_actions)
        ref_peaks = detect_peaks(ref_actions)
        matched, unmatched_main, unmatched_ref = match_peaks(main_peaks, ref_peaks)

        # Classify each match
        self._reference_peak_matches = [
            (mp, rp, offset, classify_match(offset, abs(mp['pos'] - rp['pos']), fps))
            for mp, rp, offset in matched
        ]
        self._reference_unmatched_main = unmatched_main
        self._reference_unmatched_ref = unmatched_ref

        # Build chapters list for per-chapter stats
        chapters_data = None
        video_chapters = getattr(self.app.funscript_processor, 'video_chapters', None)
        if video_chapters:
            chapters_data = []
            for ch in video_chapters:
                s_frame = ch.start_frame_id
                e_frame = ch.end_frame_id
                s_ms = (s_frame / fps) * 1000.0 if fps > 0 else 0
                e_ms = (e_frame / fps) * 1000.0 if fps > 0 else 0
                chapters_data.append({
                    'start_ms': s_ms,
                    'end_ms': e_ms,
                    'name': ch.position_short_name or ch.position_long_name or 'unknown',
                })

        # Aggregate metrics (with per-chapter if chapters exist)
        # Pass pre-computed peak data to avoid redundant detect_peaks + match_peaks
        self._reference_metrics = compute_comparison_metrics(
            main_actions, ref_actions, fps, chapters=chapters_data,
            peak_data=(matched, unmatched_main, unmatched_ref)
        )

        # Problem section detection
        self._reference_problem_sections = detect_problem_sections(
            main_actions, ref_actions, fps
        )

        self._reference_metrics_dirty = False

    def _create_chapters_from_problem_sections(self):
        """Create chapters from detected problem sections for easy review."""
        if not self._reference_problem_sections:
            return

        fps = 30.0
        proc = self.app.processor
        if proc and proc.fps and proc.fps > 0:
            fps = proc.fps

        created = 0
        for i, sec in enumerate(self._reference_problem_sections):
            start_frame = ms_to_frame(sec['start_ms'], fps)
            end_frame = ms_to_frame(sec['end_ms'], fps)
            self.app.funscript_processor.create_new_chapter_from_data(
                data={
                    'start_frame_str': str(start_frame),
                    'end_frame_str': str(end_frame),
                    'position_short_name_key': 'NR',
                    'segment_type': 'default',
                    'source': 'reference_comparison',
                },
            )
            created += 1

        self.app.logger.info(f"Created {created} chapters from problem sections (MAE > threshold)")

    def _handle_alternating_mode_input(self, app_state, tf: TimelineTransformer,
                                        mouse_pos, is_hovered, io):
        """Handle input for alternating mode.

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
                nearest_idx = bisect_left(self._get_cached_timestamps() or [a['at'] for a in actions], click_time)
                if nearest_idx > 0:
                    prev_pos = actions[nearest_idx - 1]['pos']
                    mid = (self._alt_top_value + self._alt_bottom_value) / 2
                    self._alt_next_is_top = prev_pos < mid

            # Place point
            val = self._alt_top_value if self._alt_next_is_top else self._alt_bottom_value
            add_point(self, click_time, val)
            self._alt_next_is_top = not self._alt_next_is_top

    def _handle_recording_mode_input(self, app_state, tf: TimelineTransformer,
                                      mouse_pos, is_hovered, io):
        """Handle input for recording mode.

        While recording: map mouse Y (or gamepad stick) in canvas to 0-100,
        capture each frame.
        """
        if not self._recording_capture:
            return

        if self._recording_capture.is_recording:
            processor = self.app.processor
            if not processor or processor.fps <= 0:
                return
            current_ms = getattr(processor, 'playhead_override_ms', None)
            if current_ms is None:
                current_ms = frame_to_ms(processor.current_frame_index, processor.fps)

            # --- Gamepad input (priority when connected) ---
            if self._gamepad_input and self._gamepad_connected:
                state = self._gamepad_input.poll()
                if state is not None:
                    self._recording_capture.capture_frame(current_ms, state.primary)
                    self.preview_actions = self._recording_capture._samples
                    self.is_previewing = True

                    # Optional device preview
                    if self._controller_device_preview:
                        self._send_device_preview(state.primary, state.secondary)

                    # B button = stop recording
                    if state.button_b:
                        self._stop_recording_and_merge()
                    return  # Gamepad takes priority

            # --- Mouse input (original behaviour) ---
            if is_hovered and tf.height > 0:
                normalized_y = 1.0 - ((mouse_pos[1] - tf.y_offset) / tf.height)
                pos = max(0, min(100, normalized_y * 100.0))
                self._recording_capture.capture_frame(current_ms, pos)
                self.preview_actions = self._recording_capture._samples
                self.is_previewing = True

    # ------------------------------------------------------------------
    # Controller / gamepad helpers (recording mode)
    # ------------------------------------------------------------------

    def _init_gamepad(self):
        """Lazy-init gamepad input and detect controllers."""
        from application.live_scripting.gamepad_input import GamepadInput
        self._gamepad_input = GamepadInput()
        self._gamepad_input.center_mode = self.app.app_settings.config.recording.gamepad_center_mode
        gamepads = self._gamepad_input.detect_gamepads()
        if gamepads:
            self._gamepad_input.active_joystick_id = gamepads[0].joystick_id
            self._gamepad_connected = True

    def _stop_recording_and_merge(self):
        """Stop recording, RDP-simplify, merge into timeline."""
        self.is_previewing = False
        self.preview_actions = None
        if not self._recording_capture:
            return
        simplified = self._recording_capture.stop_recording(self._recording_rdp_epsilon)
        if simplified:
            actions_before = list(self._get_actions() or [])
            actions = self._get_actions()
            merged = list(actions) + simplified
            merged.sort(key=lambda a: a['at'])
            fs, axis = self._get_target_funscript_details()
            if fs and axis:
                fs.set_axis_actions(axis, merged)
                self._post_mutation_refresh()

                actions_after = list(self._get_actions() or [])
                from application.classes.undo_manager import BulkReplaceCmd
                self.app.undo_manager.push_done(BulkReplaceCmd(self.timeline_num, actions_before, actions_after, f"Recording Inject (T{self.timeline_num})"))

    def _send_device_preview(self, primary: float, secondary: float):
        """Send position to device_manager for haptic feedback (optional)."""
        dm = self._get_device_manager()
        if dm and hasattr(dm, 'update_position'):
            dm.update_position(primary, secondary)

    def _get_device_manager(self):
        """Safely get device_manager if device_control addon is loaded."""
        try:
            return self.app.device_manager
        except AttributeError:
            return None

    def _start_calibration(self):
        """Begin input-delay calibration routine."""
        from application.live_scripting.gamepad_input import CalibrationRoutine
        if self._gamepad_input:
            self._calibration = CalibrationRoutine(self._gamepad_input)
            self._calibration.start()

    def _handle_injection_mode_input(self, app_state, tf: TimelineTransformer,
                                      mouse_pos, is_hovered, io):
        """Handle input for injection mode.

        Click on a segment to inject intermediate points into it.
        """
        if not is_hovered:
            return

        actions = self._get_actions()
        if not actions or len(actions) < 2:
            return

        center_x = tf.x_offset + (tf.width / 2)
        click_time = tf.x_to_time(center_x)

        # Highlight segment under cursor (visual feedback)
        timestamps = self._get_cached_timestamps() or [a['at'] for a in actions]
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

            actions_before = list(self._get_actions() or [])

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
                self._post_mutation_refresh()

                actions_after = list(self._get_actions() or [])
                from application.classes.undo_manager import BulkReplaceCmd
                self.app.undo_manager.push_done(BulkReplaceCmd(self.timeline_num, actions_before, actions_after, f"Interpolate (T{self.timeline_num})"))

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

    def _render_toolbar(self):
        # Hide this timeline
        if imgui.button(f"Hide##hide_{self.timeline_num}"):
            setattr(self.app.app_state_ui, self._visibility_attr, False)
        if imgui.is_item_hovered():
            imgui.set_tooltip(f"Hide timeline {self.timeline_num} (toggle from View menu)")
        imgui.same_line()

        # Delete / Clear button - adaptive label based on selection
        num_selected = len(self.multi_selected_action_indices) if self.multi_selected_action_indices else 0
        if num_selected > 0:
            del_label = f"Delete ({num_selected})##{self.timeline_num}"
            if imgui.button(del_label):
                delete_selected(self)
            if imgui.is_item_hovered():
                imgui.set_tooltip(f"Delete {num_selected} selected points (Ctrl+Z to undo)")
        else:
            del_label = f"Clear Funscript##{self.timeline_num}"
            if imgui.button(del_label):
                clear_all_points(self)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Delete ALL actions on this funscript (Ctrl+Z to undo)")
        
        imgui.same_line()
        imgui.text("|")
        imgui.same_line()

        # Autotune one-click button
        if imgui.button(f"Autotune##{self.timeline_num}"):
            context = self.plugin_renderer.plugin_manager.plugin_contexts.get("Ultimate Autotune")
            if context:
                context.apply_requested = True
                if self.multi_selected_action_indices:
                    context.apply_to_selection = True
        if imgui.is_item_hovered():
            imgui.set_tooltip("Apply Ultimate Autotune (selection or full timeline)")

        # Plugin / Pipeline buttons removed from the per-timeline toolbar.
        # Plugins are accessible from the right-side Plugins block and the
        # Pipeline window is openable from the Plugins block header.

        # Keep the toolbar on one row: separator + continue with Nudge block.
        imgui.same_line()
        imgui.text("|")
        imgui.same_line()

        # Nudge All Buttons
        if imgui.button(f"<<##{self.timeline_num}"):
            if self.nudge_chapter_only:
                nudge_chapter_time(self, -1)
            else:
                nudge_all_time(self, -1)
        if imgui.is_item_hovered():
            tip = "Nudge points in selected chapter left by 1 frame" if self.nudge_chapter_only else "Nudge all points left by 1 frame"
            imgui.set_tooltip(tip)
        imgui.same_line()

        if imgui.button(f">>##{self.timeline_num}"):
            if self.nudge_chapter_only:
                nudge_chapter_time(self, 1)
            else:
                nudge_all_time(self, 1)
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

        # --- Visualization Toggles ---
        imgui.same_line()
        imgui.text("|")
        imgui.same_line()

        # Heatmap toggle
        _, self._show_heatmap_coloring = imgui.checkbox(f"Heat##{self.timeline_num}", self._show_heatmap_coloring)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Speed-based heatmap coloring for line segments")
        imgui.same_line()

        # Speed warnings toggle
        _, self._show_speed_warnings = imgui.checkbox(f"Spd##{self.timeline_num}", self._show_speed_warnings)
        if imgui.is_item_hovered():
            imgui.set_tooltip(f"Highlight speed limit violations (>{self._speed_limit_threshold:.0f} u/s)")

        # Timeline row height lives in the main toolbar's TIMELINE section
        # now (toolbar_ui._render_edit_section), removed from here to keep
        # per-timeline toolbars minimal.

        # --- Mode Selector ---
        imgui.same_line()
        imgui.text("|")
        imgui.same_line()

        _MODE_DESCRIPTIONS = {
            TimelineMode.SELECT: (
                "Select Mode\n"
                "Click: select point | Double-click: seek video\n"
                "Drag empty: range select | Ctrl+drag: box select\n"
                "Shift+click empty: create point | Shift/Ctrl+click: multi-select"
            ),
            TimelineMode.ALTERNATING: (
                "Alternating Mode\n"
                "Click to place alternating high/low points.\n"
                "Top/Bottom values adjustable via sliders.\n"
                "Great for quickly creating rhythmic stroke patterns."
            ),
            TimelineMode.INJECTION: (
                "Injection Mode (Supporter)\n"
                "Right-click a segment to inject intermediate points.\n"
                "Supports linear, cosine, and cubic interpolation."
            ),
            TimelineMode.RECORDING: (
                "Recording Mode (Supporter)\n"
                "Draw funscripts by moving your mouse while video plays.\n"
                "Points auto-simplified with RDP. Press Record to start."
            ),
        }

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
            imgui.set_tooltip(_MODE_DESCRIPTIONS.get(self._mode, "Timeline editing mode"))

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
                    self._stop_recording_and_merge()
            else:
                if imgui.button(f"Record##{self.timeline_num}"):
                    self._recording_capture = RecordingCapture()
                    self._recording_capture.input_delay_ms = self._controller_input_delay_ms
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

            # --- Controller indicator / detect ---
            imgui.same_line()
            imgui.text("|")
            imgui.same_line()
            if self._gamepad_connected:
                imgui.text_colored("Gamepad", 0.2, 1.0, 0.4, 1.0)
                if imgui.is_item_hovered():
                    name = getattr(self._gamepad_input, '_detected_name', 'Controller')
                    imgui.set_tooltip(f"Connected: {name}")
            else:
                if imgui.small_button(f"Gamepad##{self.timeline_num}"):
                    self._init_gamepad()
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Detect gamepad controller")

            # Expand/collapse controller settings
            if self._gamepad_connected:
                imgui.same_line()
                if imgui.small_button(f"...##{self.timeline_num}_gp"):
                    self._show_controller_settings = not self._show_controller_settings

            # --- Controller settings panel (collapsible) ---
            if self._show_controller_settings and self._gamepad_connected:
                self._draw_controller_settings()

        # BPM controls (core feature)
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
            cfg = self._bpm_config
            imgui.same_line()
            # BPM value with +/- buttons
            imgui.push_item_width(120)
            _, cfg.bpm = imgui.input_float(
                f"##bpm_val{self.timeline_num}", cfg.bpm, 1.0, 5.0, "%.3f")
            imgui.pop_item_width()
            cfg.bpm = max(1.0, min(999.0, cfg.bpm))
            if imgui.is_item_hovered():
                imgui.set_tooltip("BPM value")
            # Offset
            imgui.same_line()
            imgui.push_item_width(100)
            offset_s = cfg.offset_ms / 1000.0
            _, offset_s = imgui.input_float(
                f"##bpm_off{self.timeline_num}", offset_s, 0.001, 0.01, "%.3f")
            cfg.offset_ms = offset_s * 1000.0
            imgui.pop_item_width()
            if imgui.is_item_hovered():
                imgui.set_tooltip("Beat offset (seconds)")
            # Subdivision dropdown
            imgui.same_line()
            imgui.push_item_width(105)
            _, cfg.subdivision = imgui.combo(
                f"##bpm_sub{self.timeline_num}", cfg.subdivision, SUBDIVISION_LABELS)
            imgui.pop_item_width()
            if imgui.is_item_hovered():
                imgui.set_tooltip("Beat subdivision")
            # Snap to beat
            imgui.same_line()
            _, cfg.snap_to_beat = imgui.checkbox(f"Snap##{self.timeline_num}", cfg.snap_to_beat)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Snap points to beat grid")
            # Tap tempo
            imgui.same_line()
            if imgui.button(f"Tap##{self.timeline_num}"):
                bpm = self._tap_tempo.tap()
                if bpm:
                    cfg.bpm = round(bpm, 3)
            if imgui.is_item_hovered():
                imgui.set_tooltip(f"Tap tempo ({self._tap_tempo.tap_count} taps)")

        # FPS Override controls (core feature)
        app_state = self.app.app_state_ui
        imgui.same_line()
        imgui.text("|")
        imgui.same_line()
        _, app_state.fps_override_enabled = imgui.checkbox(
            f"FPS##{self.timeline_num}", app_state.fps_override_enabled)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Snap to custom frame grid")
        if app_state.fps_override_enabled:
            imgui.same_line()
            imgui.push_item_width(60)
            _, app_state.fps_override_value = imgui.input_float(
                f"##fps_val{self.timeline_num}", app_state.fps_override_value, 0, 0, "%.3f")
            app_state.fps_override_value = max(1.0, min(240.0, app_state.fps_override_value))
            imgui.pop_item_width()
            if imgui.is_item_hovered():
                imgui.set_tooltip("Override FPS for snap grid (e.g. 59.940)")
            # Dynamically update snap grid (keep float precision for fractional FPS)
            app_state.snap_to_grid_time_ms = 1000.0 / app_state.fps_override_value

        # Timeline Status Text
        imgui.same_line()
        imgui.text("|")
        imgui.same_line()
        status_text = self._get_timeline_status_text()
        imgui.text(status_text)

    def _render_context_menu(self, tf):
        if imgui.begin_popup(f"TimelineContext{self.timeline_num}"):
            has_sel = bool(self.multi_selected_action_indices)
            n_sel = len(self.multi_selected_action_indices) if has_sel else 0

            if has_sel:
                imgui.text_disabled(f"{n_sel} point{'s' if n_sel != 1 else ''} selected")
                imgui.separator()
                if imgui.menu_item("Delete Selected", "Del")[0]:
                    delete_selected(self)
                    imgui.close_current_popup()
                if imgui.menu_item("Copy Selection", "Ctrl+C")[0]:
                    copy_selection(self)
                    imgui.close_current_popup()
                if imgui.begin_menu("Keep Only..."):
                    if imgui.menu_item("Top Points")[0]: filter_selection(self, 'top')
                    if imgui.menu_item("Bottom Points")[0]: filter_selection(self, 'bottom')
                    if imgui.menu_item("Mid Points")[0]: filter_selection(self, 'mid')
                    imgui.end_menu()
                imgui.separator()
                self._render_quickfix_flat(cursor_only=False)
            else:
                if imgui.menu_item("Add Point Here")[0]:
                    t, v = getattr(self, 'new_point_candidate', (0, 0))
                    add_point(self, t, v)
                    imgui.close_current_popup()
                if imgui.begin_menu("Paste from Clipboard"):
                    if imgui.menu_item("At Cursor (relative)", "Ctrl+V")[0]:
                        t, v = getattr(self, 'new_point_candidate', (0, 0))
                        paste_actions(self, t)
                        imgui.close_current_popup()
                    if imgui.menu_item("At Cursor + Replace Interval", "Ctrl+Shift+V")[0]:
                        t, v = getattr(self, 'new_point_candidate', (0, 0))
                        paste_actions_replace(self, t)
                        imgui.close_current_popup()
                    if imgui.menu_item("At Original Timestamps", "Ctrl+Alt+V")[0]:
                        paste_actions_exact(self)
                        imgui.close_current_popup()
                    imgui.end_menu()
                if imgui.menu_item("Select All", "Ctrl+A")[0]:
                    actions = self._get_actions()
                    self.multi_selected_action_indices = {self._action_key(a) for a in actions}
                    imgui.close_current_popup()
                imgui.separator()
                self._render_quickfix_flat(cursor_only=True)

            imgui.separator()

            # Rendering settings (heatmap, sync line, etc.)
            if imgui.begin_menu("Rendering..."):
                _hl = bool(self._heatmap_mapper.highlight_overspeed)
                changed_hl, _hl = imgui.checkbox("Highlight overspeed segments", _hl)
                if changed_hl:
                    self._heatmap_mapper.highlight_overspeed = _hl
                    self.app.app_settings.config.heatmap.highlight_overspeed = _hl
                    self._heatmap_speeds_cache = None
                    self._heatmap_colors_cache = None
                imgui.text("Max speed (units/s):")
                imgui.same_line()
                imgui.push_item_width(120)
                _ms = float(self._heatmap_mapper.max_speed)
                changed_ms, _ms = imgui.slider_float("##hm_max_speed",
                                                      _ms, 50.0, 1000.0, "%.0f")
                imgui.pop_item_width()
                if changed_ms:
                    self._heatmap_mapper.max_speed = max(1.0, _ms)
                    self.app.app_settings.config.heatmap.max_speed = _ms
                    self._heatmap_speeds_cache = None
                    self._heatmap_colors_cache = None
                imgui.separator()
                _sync = self.app.app_settings.config.ui.timeline_show_video_sync_line
                changed_sync, _sync = imgui.checkbox(
                    "Show video sync line (red, exact frame time)", _sync)
                if changed_sync:
                    self.app.app_settings.config.ui.timeline_show_video_sync_line = _sync
                imgui.end_menu()

            # Live tools (drag a slider, see preview, single undo on release)
            if imgui.begin_menu("Live Tools..."):
                if imgui.menu_item("Range Extender (live drag)")[0]:
                    self._open_live_tool('range')
                if imgui.menu_item("RDP Simplify (live drag)")[0]:
                    self._open_live_tool('rdp')
                imgui.end_menu()

            # Timeline ops (cross-timeline copy / swap / axis)
            if imgui.begin_menu("Timeline Ops"):
                other_num = 2 if self.timeline_num == 1 else 1
                _copy_swap_targets = [other_num]
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
                            copy_to_other(self, t_num)
                            imgui.close_current_popup()
                    imgui.end_menu()
                if imgui.begin_menu("Swap with..."):
                    for t_num in _copy_swap_targets:
                        if imgui.menu_item(self._tl_label(t_num))[0]:
                            self._handle_swap_timeline(t_num)
                            imgui.close_current_popup()
                    imgui.end_menu()
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
                imgui.end_menu()

            # Reference comparison
            if imgui.begin_menu("Reference"):
                if imgui.menu_item("Load Reference Funscript...")[0]:
                    self._load_reference_funscript()
                    imgui.close_current_popup()
                if self.reference_overlay_actions:
                    if imgui.menu_item("Clear Reference Overlay")[0]:
                        self._clear_reference_overlay()
                        imgui.close_current_popup()
                    if self._reference_problem_sections:
                        n = len(self._reference_problem_sections)
                        if imgui.menu_item(f"Create Chapters from {n} Problem Section{'s' if n != 1 else ''}")[0]:
                            self._create_chapters_from_problem_sections()
                            imgui.close_current_popup()
                imgui.end_menu()

            # Bookmarks (collapsed under one submenu)
            if imgui.begin_menu("Bookmarks"):
                if imgui.menu_item("Add Bookmark Here", "B")[0]:
                    t, _ = getattr(self, 'new_point_candidate', (0, 0))
                    self._bookmark_manager.add(t)
                    imgui.close_current_popup()
                if imgui.begin_menu("Go to..."):
                    for bm in self._bookmark_manager.bookmarks:
                        time_str = _format_time(self.app, bm.time_ms / 1000.0)
                        label = f"{bm.name or 'Bookmark'} ({time_str})"
                        if imgui.menu_item(label)[0]:
                            self._seek_video(bm.time_ms)
                            imgui.close_current_popup()
                    if not self._bookmark_manager.bookmarks:
                        imgui.menu_item("(no bookmarks)", enabled=False)
                    imgui.end_menu()
                if imgui.begin_menu("Rename..."):
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
                if imgui.begin_menu("Delete..."):
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
                    if imgui.menu_item("Clear All")[0]:
                        self._bookmark_manager.clear()
                        imgui.close_current_popup()
                imgui.end_menu()

            # Patreon-only: patterns + multi-axis generation
            if _is_feature_available("patreon_features"):
                if imgui.begin_menu("Patterns"):
                    if has_sel and n_sel >= 2:
                        if imgui.menu_item("Save Selection as Pattern")[0]:
                            actions = self._get_actions()
                            sel_actions = [actions[i] for i in self._resolve_selected_indices()]
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
                                            actions_before = list(self._get_actions() or [])
                                            actions = self._get_actions()
                                            merged = list(actions) + new_actions
                                            merged.sort(key=lambda a: a['at'])
                                            fs, axis = self._get_target_funscript_details()
                                            if fs and axis:
                                                fs.set_axis_actions(axis, merged)
                                                self._post_mutation_refresh()
                                                actions_after = list(self._get_actions() or [])
                                                from application.classes.undo_manager import BulkReplaceCmd
                                                self.app.undo_manager.push_done(BulkReplaceCmd(self.timeline_num, actions_before, actions_after, f"Apply Pattern (T{self.timeline_num})"))
                                    imgui.close_current_popup()
                        else:
                            imgui.menu_item("(library not loaded)", enabled=False)
                        imgui.end_menu()
                    if imgui.begin_menu("Generate Axis"):
                        for axis_name in ['roll', 'pitch', 'twist', 'sway', 'surge']:
                            if imgui.menu_item(axis_name.capitalize())[0]:
                                self._trigger_multi_axis_generation(axis_name)
                                imgui.close_current_popup()
                        imgui.end_menu()
                    imgui.end_menu()

            # Other (non-Quickfix) plugin categories stay under "Run Plugin"
            self._render_plugin_selection_menu(skip_categories={"Quickfix Tools"})

            imgui.end_popup()

    def _activate_plugin(self, plugin_name: str, apply_to_selection: bool):
        """Open or directly-apply a plugin from the context menu."""
        pm = self.plugin_renderer.plugin_manager
        ui_data = pm.get_plugin_ui_data(plugin_name)
        if not ui_data:
            return
        ctx = pm.plugin_contexts.get(plugin_name)
        if ctx:
            ctx.apply_to_selection = apply_to_selection
        # If the plugin opts into direct-apply (no required params), run it now.
        if self.plugin_renderer._should_apply_directly(plugin_name, ui_data):
            if ctx:
                ctx.apply_requested = True
        else:
            pm.set_plugin_state(plugin_name, PluginUIState.OPEN)
        self.logger.info(
            f"Plugin invoked: {plugin_name} (sel={apply_to_selection}) on T{self.timeline_num}")

    def _render_quickfix_flat(self, cursor_only: bool):
        """Surface Quickfix Tools at the top level of the context menu instead
        of forcing the user through Run Plugin -> Quickfix Tools."""
        pm = self.plugin_renderer.plugin_manager
        names = []
        for p_name in pm.get_available_plugins():
            ctx = pm.plugin_contexts.get(p_name)
            if not ctx or not ctx.plugin_instance:
                continue
            cat = getattr(ctx.plugin_instance, 'category', 'General')
            if cat != 'Quickfix Tools':
                continue
            requires_cursor = bool(getattr(ctx.plugin_instance, 'requires_cursor', False))
            if cursor_only and not requires_cursor:
                continue
            if not cursor_only and requires_cursor:
                continue
            names.append(p_name)

        if not names:
            return

        header = "Quickfix at Cursor" if cursor_only else "Quickfix on Selection"
        imgui.text_disabled(header)
        for n in sorted(names):
            if imgui.menu_item(n)[0]:
                self._activate_plugin(n, apply_to_selection=not cursor_only)
                imgui.close_current_popup()
            ui_data = pm.get_plugin_ui_data(n)
            if ui_data and ui_data.get('description') and imgui.is_item_hovered():
                imgui.set_tooltip(ui_data['description'])
            
    def _render_plugin_selection_menu(self, skip_categories: Optional[set] = None):
        skip_categories = skip_categories or set()
        imgui.separator()
        if imgui.begin_menu("Run Plugin"):
             fs, axis = self._get_target_funscript_details()
             pm = self.plugin_renderer.plugin_manager
             available = pm.get_available_plugins()

             # Group plugins by category
             _CAT_ORDER = ["Autotune", "Quickfix Tools", "Transform", "Smoothing",
                           "Timing & Generation", "General"]
             categorized = {c: [] for c in _CAT_ORDER}
             for p_name in available:
                 ctx = pm.plugin_contexts.get(p_name)
                 cat = "General"
                 if ctx and ctx.plugin_instance:
                     cat = getattr(ctx.plugin_instance, 'category', 'General')
                 if cat in skip_categories:
                     continue
                 if cat not in categorized:
                     categorized[cat] = []
                 categorized[cat].append(p_name)

             def _activate(name):
                 pm.set_plugin_state(name, PluginUIState.OPEN)
                 ctx2 = pm.plugin_contexts.get(name)
                 if ctx2:
                     ctx2.apply_to_selection = True
                     self.logger.info(f"Auto-enabled 'apply to selection' for {name} (triggered from context menu)")

             for cat in _CAT_ORDER:
                 names = categorized.get(cat, [])
                 if not names:
                     continue
                 if imgui.begin_menu(cat):
                     if cat == "Quickfix Tools":
                         # Separate selection vs cursor tools
                         sel_names = []
                         cur_names = []
                         for n in sorted(names):
                             ctx = pm.plugin_contexts.get(n)
                             if ctx and ctx.plugin_instance and getattr(ctx.plugin_instance, 'requires_cursor', False):
                                 cur_names.append(n)
                             else:
                                 sel_names.append(n)
                         for n in sel_names:
                             if imgui.menu_item(n)[0]:
                                 _activate(n)
                         if cur_names:
                             imgui.separator()
                             imgui.text_disabled("Position playhead first")
                             for n in cur_names:
                                 if imgui.menu_item(n)[0]:
                                     _activate(n)
                     else:
                         for n in sorted(names):
                             if imgui.menu_item(n)[0]:
                                 _activate(n)
                     imgui.end_menu()

             # Render any extra categories not in the predefined order
             for cat, names in categorized.items():
                 if cat not in _CAT_ORDER and names:
                     if imgui.begin_menu(cat):
                         for n in sorted(names):
                             if imgui.menu_item(n)[0]:
                                 _activate(n)
                         imgui.end_menu()

             imgui.end_menu()

    # ==================================================================================
    # DATA MODIFICATION HELPERS
    # ==================================================================================

    def _post_mutation_refresh(self):
        """Convenience wrapper for undo finalization."""
        self.app.funscript_processor._post_mutation_refresh(self.timeline_num, "Edit")
        self.invalidate_cache()

    def _trigger_multi_axis_generation(self, target_axis: str):
        """Trigger multi-axis generation for the given axis via the plugin system."""
        try:
            fs, _ = self._get_target_funscript_details()
            if not fs:
                return

            actions_before = list(self._get_actions() or [])

            fs.apply_plugin("Multi-Axis Generator", axis='primary',
                           target_axis=target_axis, generation_mode='heuristic')

            # The plugin stores generated data under the semantic axis name
            # (e.g. 'roll') in additional_axes.  Route it to the positional
            # axis that the timeline system actually reads.
            generated = fs.get_axis_actions(target_axis)
            if generated:
                tl_num = fs.get_timeline_for_axis(target_axis) if hasattr(fs, 'get_timeline_for_axis') else None
                if tl_num is not None:
                    # Map timeline number to positional axis name
                    if tl_num == 1:
                        pos_axis = 'primary'
                    elif tl_num == 2:
                        pos_axis = 'secondary'
                    else:
                        pos_axis = f'axis_{tl_num}'
                    # Only copy if semantic and positional names differ
                    if pos_axis != target_axis:
                        fs.set_axis_actions(pos_axis, generated)

            self._post_mutation_refresh()

            actions_after = list(self._get_actions() or [])
            from application.classes.undo_manager import BulkReplaceCmd
            self.app.undo_manager.push_done(BulkReplaceCmd(self.timeline_num, actions_before, actions_after, f"Multi-Axis Generation (T{self.timeline_num})"))
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Multi-axis generation failed: {e}")

    def _snap_time(self, t_ms):
        """Snap time to beat grid (if BPM snap enabled) or normal grid."""
        cfg = self._bpm_config
        if cfg and cfg.snap_to_beat and cfg.bpm > 0:
            interval = cfg.beat_interval_ms
            return cfg.offset_ms + round((t_ms - cfg.offset_ms) / interval) * interval
        snap_t = self.app.app_state_ui.snap_to_grid_time_ms
        return round(t_ms / snap_t) * snap_t if snap_t > 0 else t_ms

    def _toggle_selection_loop(self):
        self._selection_loop_armed = not self._selection_loop_armed
        state = "armed" if self._selection_loop_armed else "off"
        self.app.notify(f"Selection loop {state} (T{self.timeline_num})", "info", 1.5)

    def _open_live_tool(self, op: str):
        self._live_tool_open = op
        self._live_tool_value = 10.0 if op == 'range' else 1.0
        begin_live_drag(self, 'range_extend' if op == 'range' else 'rdp')

    def _render_live_tool_popup(self):
        if not self._live_tool_open:
            return
        op = self._live_tool_open
        title = "Range Extender" if op == 'range' else "RDP Simplify"
        imgui.set_next_window_size(360, 0, condition=imgui.APPEARING)
        opened, visible = imgui.begin(f"{title}##LiveTool_T{self.timeline_num}",
                                       closable=True,
                                       flags=imgui.WINDOW_NO_COLLAPSE)
        if not visible:
            self._close_live_tool()
            imgui.end()
            return
        if opened:
            sel_count = len(self.multi_selected_action_indices)
            if op == 'range':
                imgui.text(f"Targeting: {'selection' if sel_count else 'all'} ({sel_count} pts)")
                imgui.push_item_width(-1)
                changed, val = imgui.slider_int("##rng_amt", int(self._live_tool_value), -50, 50, "%d %%")
                imgui.pop_item_width()
                if changed:
                    self._live_tool_value = float(val)
                    live_range_extend(self, int(val))
                if imgui.button("Apply & Close"):
                    self._close_live_tool()
                imgui.same_line()
                if imgui.button("Cancel"):
                    self._cancel_live_tool()
            else:
                imgui.text(f"Targeting: {'selection' if sel_count else 'all'} ({sel_count} pts)")
                imgui.push_item_width(-1)
                changed, val = imgui.slider_float("##rdp_eps", float(self._live_tool_value), 0.1, 25.0, "eps=%.2f")
                imgui.pop_item_width()
                if changed:
                    self._live_tool_value = float(val)
                    live_rdp_simplify(self, float(val))
                if imgui.button("Apply & Close"):
                    self._close_live_tool()
                imgui.same_line()
                if imgui.button("Cancel"):
                    self._cancel_live_tool()
        imgui.end()

    def _close_live_tool(self):
        end_live_drag(self)
        self._live_tool_open = None

    def _cancel_live_tool(self):
        # Pop the coalesced live-drag entry and undo it to restore baseline.
        from application.classes.undo_manager import LiveDragCmd
        if self.app.undo_manager.match_top(LiveDragCmd):
            self.app.undo_manager.pop_top(self.app)
        self._close_live_tool()

    def _tick_selection_loop(self):
        """Wrap playhead from last selected to first when loop is armed."""
        if not self._selection_loop_armed:
            return
        if not self.multi_selected_action_indices:
            return
        proc = self.app.processor
        if not proc or not proc.fps or proc.fps <= 0:
            return
        # Honor only when actually playing, paused/scrubbing should not loop.
        if not getattr(proc, 'is_processing', False):
            return
        if hasattr(proc, 'pause_event') and proc.pause_event.is_set():
            return
        actions = self._get_actions()
        if not actions:
            return
        resolved = sorted(self._resolve_selected_indices())
        if len(resolved) < 2:
            return
        t_start = actions[resolved[0]]['at']
        t_end = actions[resolved[-1]]['at']
        if t_end <= t_start:
            return
        cur_ms = (proc.current_frame_index / proc.fps) * 1000.0
        if cur_ms >= t_end:
            target_frame = max(0, int((t_start / 1000.0) * proc.fps))
            self.app.event_handlers.seek_video_with_sync(target_frame)

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
                selected_indices = self._resolve_selected_indices()
                params['selected_indices'] = selected_indices

            # Auto-inject current_time_ms for cursor-dependent plugins
            if getattr(plugin_instance, 'requires_cursor', False):
                fps = self.app.processor.video_info.get('fps', 30) if self.app.processor and self.app.processor.video_info else 30
                frame_idx = getattr(self.app, 'current_frame_index', 0)
                params['current_time_ms'] = frame_to_ms(frame_idx, fps)

            # Capture actions before plugin for unified undo
            actions_before = list(fs.get_axis_actions(axis) or [])

            # Apply the plugin transformation
            try:
                result = plugin_instance.transform(fs, axis, **params)

                self.logger.info(f"Successfully applied {plugin_name} to timeline {self.timeline_num}")
                self.app.notify(f"Applied {plugin_name}", "success")

                # Finalize and update UI
                self.app.funscript_processor._post_mutation_refresh(
                    self.timeline_num,
                    f"Apply {plugin_name}"
                )

                # Invalidate caches
                self.invalidate_cache()

                # Unified undo
                actions_after = list(fs.get_axis_actions(axis) or [])
                from application.classes.undo_manager import BulkReplaceCmd
                self.app.undo_manager.push_done(BulkReplaceCmd(
                    self.timeline_num, actions_before, actions_after,
                    f"Apply {plugin_name} (T{self.timeline_num})"))
        
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
        mpv = getattr(self.app, '_mpv_controller', None)
        if mpv and mpv.is_active:
            # mpv review mode, processor is stopped but mpv drives current_frame_index
            is_playing = mpv.is_playing
        elif getattr(processor, 'live_capture_active', False):
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
        seek_in_progress = False

        should_sync = is_playing or (forced and not app_state.timeline_interaction_active)

        if should_sync:
            # Guard against division by zero when fps not yet available (e.g., loading from Stash WebView)
            if not processor.fps or processor.fps <= 0:
                return

            # Use exact action time if we just jumped to a point, otherwise compute from frame
            current_ms = getattr(processor, 'playhead_override_ms', None)
            if current_ms is None:
                current_ms = frame_to_ms(processor.current_frame_index, processor.fps)

            # Center the playhead
            center_offset = (tf.width * tf.zoom) / 2
            target_pan = current_ms - center_offset

            app_state.timeline_pan_offset_ms = target_pan

            # Only clear the forced flag if we are NOT waiting for a seek to complete
            if forced and not seek_in_progress:
                app_state.force_timeline_pan_to_current_frame = False
