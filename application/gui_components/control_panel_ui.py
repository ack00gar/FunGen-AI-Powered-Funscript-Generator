import imgui
import os
import config
from config.constants_colors import CurrentTheme
from application.utils import get_icon_texture_manager, primary_button_style, destructive_button_style
from application.utils.feature_detection import is_feature_available as _is_feature_available
from application.utils.imgui_helpers import DisabledScope as _DisabledScope
from application.utils.section_card import section_card

# Import dynamic tracker discovery
try:
    from .dynamic_tracker_ui import DynamicTrackerUI
    from config.tracker_discovery import get_tracker_discovery, TrackerCategory
except ImportError:
    DynamicTrackerUI = None
    TrackerCategory = None

# Import mixin sub-modules
from .cp_simple_mode_ui import SimpleModeMixin
from .cp_post_processing_ui import PostProcessingMixin
from .cp_advanced_settings_ui import AdvancedSettingsMixin
from .cp_execution_ui import ExecutionMixin
from .cp_tracker_settings_ui import TrackerSettingsMixin
from .cp_device_control_ui import DeviceControlMixin
from .cp_streamer_ui import StreamerMixin

def _tooltip_if_hovered(text):
    if imgui.is_item_hovered():
        imgui.set_tooltip(text)

def _readonly_input(label_id, value, width=-1):
    if width is not None and width >= 0:
        imgui.push_item_width(width)
    imgui.input_text(label_id, value or "Not set", 256, flags=imgui.INPUT_TEXT_READ_ONLY)
    if width is not None and width >= 0:
        imgui.pop_item_width()


class ControlPanelUI(
    SimpleModeMixin,
    PostProcessingMixin,
    AdvancedSettingsMixin,
    ExecutionMixin,
    TrackerSettingsMixin,
    DeviceControlMixin,
    StreamerMixin,
):
    def __init__(self, app):
        self.app = app
        self.timeline_editor1 = None
        self.timeline_editor2 = None
        self.ControlPanelColors = config.ControlPanelColors
        self.GeneralColors = config.GeneralColors

        # PERFORMANCE OPTIMIZATIONS: Smart rendering and caching
        self._last_tab_hash = None  # Track tab changes
        self._cached_tab_content = {}  # Cache expensive tab rendering
        self._widget_visibility_cache = {}  # Cache widget visibility states
        self._update_throttle_counter = 0  # Throttle expensive updates
        self._heavy_operation_frame_skip = 0  # Skip frames during heavy ops
        self.constants = config.constants
        self.AI_modelExtensionsFilter = self.constants.AI_MODEL_EXTENSIONS_FILTER
        self.AI_modelTooltipExtensions = self.constants.AI_MODEL_TOOLTIP_EXTENSIONS

        # Initialize dynamic tracker UI helper
        self.tracker_ui = None
        self._try_reinitialize_tracker_ui()

        # Initialize device control attributes (supporter feature)
        self.device_manager = None
        self.param_manager = None
        self._device_control_initialized = False
        self._first_frame_rendered = False
        self.video_playback_bridge = None  # Video playback bridge for live control
        self.live_tracker_bridge = None    # Live tracker bridge for real-time control
        self.device_list = []  # List of discovered devices
        self._available_osr_ports = []
        self._osr_scan_performed = False

        # Device video integration (observer pattern)
        self.device_video_integration = None
        self.device_video_bridge = None
        self.device_bridge_thread = None

        # Buttplug device discovery UI state
        self._discovered_buttplug_devices = []
        self._buttplug_discovery_performed = False

        # Streamer attributes (supporter feature)
        self._native_sync_manager = None
        self._prev_client_count = 0
        self._native_sync_status_cache = None
        self._native_sync_status_time = 0

        # Active sidebar section (replaces tab bar)
        self._active_section = "run"

        # Advanced tab search
        self._advanced_search_query = ""

        # Post-Processing tab state
        self._pp_timeline_choice = 0
        self._pp_scope_choice = 0

        # Tier 2/3 Simple Mode state
        self._auto_recommended_tracker = None
        self._auto_recommendation_reason = None
        self._simple_mode_post_processing_applied = False
        self._user_manually_picked_tracker = False

        # Settings profiles state
        self._profile_name_input = ""
        self._profile_list_cache = None
        self._profile_list_cache_time = 0
        self._selected_profile_idx = 0

        # Progressive disclosure toggle for Advanced tab (default True in expert mode)
        self._show_all_advanced_settings = True

    # ------- Helpers -------

    def _try_reinitialize_tracker_ui(self):
        """Try to initialize or reinitialize the dynamic tracker UI."""
        if self.tracker_ui is not None:
            return  # Already initialized

        try:
            if DynamicTrackerUI:
                self.tracker_ui = DynamicTrackerUI()
                if hasattr(self.app, 'logger'):
                    self.app.logger.debug("Dynamic tracker UI initialized successfully")
            else:
                if hasattr(self.app, 'logger'):
                    self.app.logger.warning("DynamicTrackerUI class not available (import failed)")
        except Exception as e:
            if hasattr(self.app, 'logger'):
                self.app.logger.error(f"Failed to initialize dynamic tracker UI: {e}")
            self.tracker_ui = None

    def _is_tracker_category(self, tracker_name: str, category) -> bool:
        """Check if tracker belongs to specific category using dynamic discovery."""
        from config.tracker_discovery import get_tracker_discovery
        discovery = get_tracker_discovery()
        tracker_info = discovery.get_tracker_info(tracker_name)
        return tracker_info and tracker_info.category == category

    def _is_live_tracker(self, tracker_name: str) -> bool:
        """Check if tracker is a live tracker (LIVE or LIVE_INTERVENTION)."""
        from config.tracker_discovery import get_tracker_discovery, TrackerCategory
        discovery = get_tracker_discovery()
        tracker_info = discovery.get_tracker_info(tracker_name)
        return tracker_info and tracker_info.category in [TrackerCategory.LIVE, TrackerCategory.LIVE_INTERVENTION]

    def _is_offline_tracker(self, tracker_name: str) -> bool:
        """Check if tracker is an offline tracker."""
        from config.tracker_discovery import TrackerCategory
        return self._is_tracker_category(tracker_name, TrackerCategory.OFFLINE)
    def _is_stage2_tracker(self, tracker_name: str) -> bool:
        """Check if tracker is a 2-stage offline tracker."""
        if not self.tracker_ui:
            # Try to reinitialize if it failed during __init__
            self._try_reinitialize_tracker_ui()

        if self.tracker_ui:
            return self.tracker_ui.is_stage2_tracker(tracker_name)

        # If still failing, log error but don't crash
        if hasattr(self.app, 'logger'):
            self.app.logger.warning(f"Dynamic tracker UI not available, cannot check if '{tracker_name}' is stage2 tracker")
        return False

    def _is_stage3_tracker(self, tracker_name: str) -> bool:
        """Check if tracker is a 3-stage offline tracker."""
        if not self.tracker_ui:
            self._try_reinitialize_tracker_ui()

        if self.tracker_ui:
            return self.tracker_ui.is_stage3_tracker(tracker_name)

        if hasattr(self.app, 'logger'):
            self.app.logger.warning(f"Dynamic tracker UI not available, cannot check if '{tracker_name}' is stage3 tracker")
        return False

    def _is_mixed_stage3_tracker(self, tracker_name: str) -> bool:
        """Check if tracker is a mixed 3-stage offline tracker."""
        if not self.tracker_ui:
            self._try_reinitialize_tracker_ui()

        if self.tracker_ui:
            return self.tracker_ui.is_mixed_stage3_tracker(tracker_name)

        if hasattr(self.app, 'logger'):
            self.app.logger.warning(f"Dynamic tracker UI not available, cannot check if '{tracker_name}' is mixed stage3 tracker")
        return False

    def _get_tracker_lists_for_ui(self, simple_mode=False):
        """Get tracker lists for UI combo boxes using dynamic discovery."""
        try:
            if simple_mode:
                # Simple mode: only live trackers
                display_names, internal_names = self.tracker_ui.get_simple_mode_trackers()
            else:
                # Full mode: all trackers
                display_names, internal_names = self.tracker_ui.get_gui_display_list()

            # Return display names, internal names, and internal names for tooltip generation
            return display_names, internal_names, internal_names

        except Exception as e:
            if hasattr(self.app, 'logger'):
                self.app.logger.warning(f"Dynamic tracker discovery failed: {e}")

            # Return empty lists on failure
            return [], [], []


    def _generate_combined_tooltip(self, tracker_names):
        """Generate combined tooltip for discovered trackers."""
        if not tracker_names:
            return "No trackers available. Please check your tracker_modules installation."

        return self.tracker_ui.get_combined_tooltip(tracker_names)

    def _help_tooltip(self, text):
        if imgui.is_item_hovered():
            imgui.set_tooltip(text)

    def _section_header(self, text, help_text=None):
        imgui.spacing()
        imgui.push_style_color(imgui.COLOR_TEXT, *self.ControlPanelColors.SECTION_HEADER)
        imgui.text(text)
        imgui.pop_style_color()
        if help_text:
            _tooltip_if_hovered(help_text)
        imgui.separator()

    def _status_indicator(self, text, status, help_text=None):
        c = self.ControlPanelColors
        icon_mgr = get_icon_texture_manager()

        # Set color and get emoji texture based on status
        if status == "ready":
            color, icon_text = c.STATUS_READY, "[OK]"
            icon_texture, _, _ = icon_mgr.get_icon_texture('check.png')
        elif status == "warning":
            color, icon_text = c.STATUS_WARNING, "[!]"
            icon_texture, _, _ = icon_mgr.get_icon_texture('warning.png')
        elif status == "error":
            color, icon_text = c.STATUS_ERROR, "[X]"
            icon_texture, _, _ = icon_mgr.get_icon_texture('error.png')
        else:
            color, icon_text = c.STATUS_INFO, "[i]"
            icon_texture = None

        # Display icon (emoji image if available, fallback to text)
        if icon_texture:
            icon_size = imgui.get_text_line_height()
            imgui.image(icon_texture, icon_size, icon_size)
            imgui.same_line(spacing=4)
        else:
            imgui.push_style_color(imgui.COLOR_TEXT, *color)
            imgui.text(icon_text)
            imgui.pop_style_color()
            imgui.same_line(spacing=4)

        # Display status text
        imgui.push_style_color(imgui.COLOR_TEXT, *color)
        imgui.text(text)
        imgui.pop_style_color()

        if help_text:
            _tooltip_if_hovered(help_text)

    # ------- Vertical Sidebar Navigation -------

    _SIDEBAR_WIDTH = 40
    _SIDEBAR_CORE_SECTIONS = [
        ("run", "R", "Run"),
        ("post_processing", "P", "Post-Processing"),
        ("advanced", "A", "Advanced"),
    ]
    _SIDEBAR_SUPPORTER_SECTIONS = [
        ("device_control", "D", "Device Control", "_feat_device"),
        ("native_sync", "S", "Streamer", "_feat_streamer"),
        ("supporter_batch", "B", "Patreon Exclusive", "_feat_supporter"),
    ]
    # Map section keys to icon asset filenames for sidebar PNG icons
    _SIDEBAR_ICON_MAP = {
        "run": "sidebar-run.png",
        "post_processing": "sidebar-postproc.png",
        "advanced": "sidebar-advanced.png",
        "device_control": "sidebar-device.png",
        "native_sync": "sidebar-stream.png",
        "supporter_batch": "sidebar-batch.png",
    }

    def _render_sidebar(self, total_h):
        """Render vertical icon sidebar for section navigation.

        Returns the active section key.
        """
        from config.element_group_colors import SidebarColors

        sidebar_w = self._SIDEBAR_WIDTH
        draw_list = imgui.get_window_draw_list()

        imgui.begin_child("##Sidebar", width=sidebar_w, height=total_h, border=False)

        # Draw sidebar background
        pos = imgui.get_window_position()
        size = (sidebar_w, total_h)
        bg_u32 = imgui.get_color_u32_rgba(*SidebarColors.BG)
        draw_list.add_rect_filled(pos[0], pos[1], pos[0] + size[0], pos[1] + size[1], bg_u32)

        btn_size = 36
        active_section = self._active_section

        # Core sections
        for key, icon, tooltip in self._SIDEBAR_CORE_SECTIONS:
            is_active = (active_section == key)
            self._render_sidebar_entry(draw_list, key, icon, tooltip, is_active,
                                       available=True, btn_size=btn_size, sidebar_w=sidebar_w)

        # Separator
        imgui.spacing()
        sep_pos = imgui.get_cursor_screen_pos()
        sep_color = imgui.get_color_u32_rgba(0.3, 0.3, 0.3, 0.5)
        draw_list.add_line(sep_pos[0] + 6, sep_pos[1],
                           sep_pos[0] + sidebar_w - 6, sep_pos[1], sep_color)
        imgui.spacing()

        # Supporter sections
        for key, icon, tooltip, feat_attr in self._SIDEBAR_SUPPORTER_SECTIONS:
            available = getattr(self, feat_attr, False)
            is_active = (active_section == key)
            self._render_sidebar_entry(draw_list, key, icon, tooltip, is_active,
                                       available=available, btn_size=btn_size, sidebar_w=sidebar_w)

        imgui.end_child()
        return self._active_section

    def _render_sidebar_entry(self, draw_list, key, icon, tooltip, is_active,
                               available, btn_size, sidebar_w):
        """Render a single sidebar navigation entry."""
        from config.element_group_colors import SidebarColors

        cursor = imgui.get_cursor_screen_pos()
        pad_x = (sidebar_w - btn_size) * 0.5

        # Background highlight for active entry
        if is_active:
            active_bg = imgui.get_color_u32_rgba(
                SidebarColors.ACTIVE_ACCENT[0],
                SidebarColors.ACTIVE_ACCENT[1],
                SidebarColors.ACTIVE_ACCENT[2], 0.2)
            draw_list.add_rect_filled(
                cursor[0], cursor[1],
                cursor[0] + sidebar_w, cursor[1] + btn_size,
                active_bg, 4.0)
            # Left accent bar
            accent_u32 = imgui.get_color_u32_rgba(*SidebarColors.ACTIVE_ACCENT)
            draw_list.add_rect_filled(
                cursor[0], cursor[1] + 4,
                cursor[0] + 3, cursor[1] + btn_size - 4,
                accent_u32, 2.0)

        # Hover highlight
        hover_region = (cursor[0], cursor[1], cursor[0] + sidebar_w, cursor[1] + btn_size)

        # Alpha for locked features
        alpha = 1.0 if available else SidebarColors.LOCKED_ALPHA
        if alpha < 1.0:
            imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * alpha)

        # Invisible button for click detection
        imgui.set_cursor_screen_pos((cursor[0] + pad_x, cursor[1]))
        if imgui.invisible_button(f"##SB_{key}", btn_size, btn_size):
            self._active_section = key

        # Check hover state BEFORE drawing so bg goes behind text
        is_hovered = imgui.is_item_hovered()
        if is_hovered:
            if available:
                imgui.set_tooltip(tooltip)
            else:
                imgui.set_tooltip(f"{tooltip}\n(Available for supporters)")
            # Hover bg (drawn before text so text stays on top)
            hover_bg = imgui.get_color_u32_rgba(*SidebarColors.HOVER_BG)
            draw_list.add_rect_filled(
                hover_region[0], hover_region[1],
                hover_region[2], hover_region[3],
                hover_bg, 4.0)

        # Draw icon (on top of hover bg)
        icon_tex = None
        icon_path = self._SIDEBAR_ICON_MAP.get(key)
        if icon_path:
            icon_mgr = get_icon_texture_manager()
            icon_tex, _, _ = icon_mgr.get_icon_texture(icon_path)

        if icon_tex:
            icon_sz = 16
            ix = cursor[0] + (sidebar_w - icon_sz) * 0.5
            iy = cursor[1] + (btn_size - icon_sz) * 0.5
            draw_list.add_image(icon_tex, (ix, iy), (ix + icon_sz, iy + icon_sz))
        else:
            # Fallback: draw letter text centered
            text_size = imgui.calc_text_size(icon)
            text_x = cursor[0] + (sidebar_w - text_size[0]) * 0.5
            text_y = cursor[1] + (btn_size - text_size[1]) * 0.5
            text_color = imgui.get_color_u32_rgba(0.9, 0.9, 0.9, alpha)
            draw_list.add_text(text_x, text_y, text_color, icon)

        if alpha < 1.0:
            imgui.pop_style_var()

    # ------- Workflow Breadcrumb -------

    def _get_workflow_stage(self):
        """Derive current workflow stage from app state (1-5)."""
        app = self.app
        proc = app.processor
        stage_proc = app.stage_processor

        if not proc or not proc.is_video_open():
            return 1  # Load
        if stage_proc.full_analysis_active:
            return 3  # Track
        if app.funscript_processor and hasattr(app, 'multi_axis_funscript'):
            fs = app.multi_axis_funscript
            if fs and fs.get_axis_actions("primary"):
                return 4  # Edit (has results)
        return 2  # Configure (video loaded, no results)

    def _render_workflow_breadcrumb(self):
        """Render 5-stage horizontal workflow indicator at top of control panel."""
        from config.element_group_colors import WorkflowColors

        current_stage = self._get_workflow_stage()
        stages = ["Load", "Configure", "Track", "Edit", "Export"]
        draw_list = imgui.get_window_draw_list()
        cursor = imgui.get_cursor_screen_pos()
        avail_w = imgui.get_content_region_available_width()

        bar_h = 24
        stage_w = avail_w / len(stages)

        for i, name in enumerate(stages):
            stage_num = i + 1
            x = cursor[0] + i * stage_w
            y = cursor[1]

            # Determine color
            if stage_num < current_stage:
                color = WorkflowColors.DONE
            elif stage_num == current_stage:
                color = WorkflowColors.ACTIVE
            else:
                color = WorkflowColors.FUTURE

            color_u32 = imgui.get_color_u32_rgba(*color)

            # Draw pill background for active stage
            if stage_num == current_stage:
                bg_u32 = imgui.get_color_u32_rgba(color[0], color[1], color[2], 0.15)
                draw_list.add_rect_filled(x + 2, y + 2, x + stage_w - 2, y + bar_h - 2,
                                          bg_u32, 4.0)

            # Draw text centered in stage area
            text_size = imgui.calc_text_size(name)
            text_x = x + (stage_w - text_size[0]) * 0.5
            text_y = y + (bar_h - text_size[1]) * 0.5
            draw_list.add_text(text_x, text_y, color_u32, name)

            # Draw connector triangle between stages
            if i < len(stages) - 1:
                conn_x = x + stage_w - 3
                conn_y = y + bar_h * 0.5
                tri_size = 4
                conn_color = imgui.get_color_u32_rgba(*WorkflowColors.CONNECTOR)
                draw_list.add_triangle_filled(
                    conn_x, conn_y - tri_size,
                    conn_x + tri_size, conn_y,
                    conn_x, conn_y + tri_size,
                    conn_color
                )

        # Advance cursor past the breadcrumb bar
        imgui.dummy(avail_w, bar_h)
        imgui.spacing()

    # ------- Pinned Action Bar -------

    def _render_pinned_action_bar(self):
        """Render contextual primary action button pinned at bottom of control panel."""
        app = self.app
        proc = app.processor
        stage_proc = app.stage_processor
        events = app.event_handlers

        bar_h = 50
        avail = imgui.get_content_region_available()

        imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + max(0, avail[1] - bar_h))

        imgui.separator()
        imgui.spacing()

        video_loaded = proc and proc.is_video_open()
        is_offline_active = stage_proc.full_analysis_active
        is_live_active = proc and getattr(proc, 'is_processing', False)

        if not video_loaded:
            # No video loaded — show Load Video
            with primary_button_style():
                if imgui.button("Load Video##PinnedAction", width=-1, height=32):
                    if hasattr(events, 'trigger_open_video_dialog'):
                        events.trigger_open_video_dialog()
                    elif hasattr(app, 'file_manager') and hasattr(app.file_manager, 'open_video_dialog'):
                        app.file_manager.open_video_dialog()
        elif is_offline_active:
            # Offline analysis active — show progress and Stop
            progress = getattr(stage_proc, 'overall_progress', 0.0)
            imgui.progress_bar(progress, (-1, 18),
                               f"{int(progress * 100)}%")
            with destructive_button_style():
                if imgui.button("Stop Analysis##PinnedAction", width=-1, height=32):
                    stage_proc.request_stop()
        elif is_live_active:
            # Live tracking active — show Pause/Resume and Stop
            is_paused = proc.pause_event.is_set() if hasattr(proc, 'pause_event') else False
            avail_w = imgui.get_content_region_available()[0]
            btn_w = (avail_w - imgui.get_style().item_spacing[0]) / 2
            if is_paused:
                with primary_button_style():
                    if imgui.button("Resume##PinnedAction", width=btn_w, height=32):
                        proc.start_processing()
                        if app.tracker and not app.tracker.tracking_active:
                            app.tracker.start_tracking()
            else:
                if imgui.button("Pause##PinnedAction", width=btn_w, height=32):
                    proc.pause_processing()
            imgui.same_line()
            with destructive_button_style():
                if imgui.button("Stop Tracking##PinnedAction", width=btn_w, height=32):
                    events.handle_abort_process_click()
        else:
            # Has results or ready to start
            has_results = False
            if hasattr(app, 'multi_axis_funscript'):
                fs = app.multi_axis_funscript
                if fs and fs.get_axis_actions("primary"):
                    has_results = True

            if has_results:
                with primary_button_style():
                    if imgui.button("Export Funscript##PinnedAction", width=-1, height=32):
                        if hasattr(events, 'trigger_save_funscript_dialog'):
                            events.trigger_save_funscript_dialog()
                        elif hasattr(app, 'file_manager'):
                            app.file_manager.save_funscript_dialog()
            else:
                selected_mode = app.app_state_ui.selected_tracker_name
                fs_proc = app.funscript_processor
                range_active = fs_proc.scripting_range_active if fs_proc else False
                if self._is_live_tracker(selected_mode):
                    label = ("Start Live AI Tracking (Range)##PinnedAction" if range_active
                             else "Start Live AI Tracking##PinnedAction")
                    with primary_button_style():
                        if imgui.button(label, width=-1, height=32):
                            self._start_live_tracking()
                elif self._is_offline_tracker(selected_mode):
                    label = ("Start AI Analysis (Range)##PinnedAction" if range_active
                             else "Start Full AI Analysis##PinnedAction")
                    with primary_button_style():
                        if imgui.button(label, width=-1, height=32):
                            events.handle_start_ai_cv_analysis()
                else:
                    with primary_button_style():
                        imgui.button("Select a Tracker##PinnedAction", width=-1, height=32)

    # ------- Main render -------

    def render(self, control_panel_w=None, available_height=None):
        app = self.app
        app_state = app.app_state_ui
        calibration_mgr = app.calibration

        # Cache feature detection flags for this frame
        self._feat_supporter = _is_feature_available("patreon_features")
        self._feat_device = _is_feature_available("device_control")
        self._feat_streamer = _is_feature_available("streamer")

        if calibration_mgr.is_calibration_mode_active:
            self._render_calibration_window(calibration_mgr, app_state)
            return

        is_simple_mode = (getattr(app_state, "ui_view_mode", "expert") == "simple")
        if is_simple_mode:
            self._render_simple_mode_ui()
            return

        floating = (app_state.ui_layout_mode == "floating")
        if floating:
            if not getattr(app_state, "show_control_panel_window", True):
                return
            is_open, new_vis = imgui.begin("Control Panel##ControlPanelFloating", closable=True)
            if new_vis != app_state.show_control_panel_window:
                app_state.show_control_panel_window = new_vis
            if not is_open:
                imgui.end()
                return
        else:
            flags = imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE
            imgui.begin("Control Panel##MainControlPanel", flags=flags)

        # --- Sidebar + Content layout ---
        avail = imgui.get_content_region_available()
        total_h = avail[1]
        sidebar_w = self._SIDEBAR_WIDTH
        content_w = max(50, avail[0] - sidebar_w - 4)

        # Left: Vertical sidebar
        self._render_sidebar(total_h)

        imgui.same_line(spacing=4)

        # Right: Breadcrumb + Content + Action bar
        imgui.begin_child("##RightPanel", width=content_w, height=total_h, border=False)

        # Workflow breadcrumb bar
        self._render_workflow_breadcrumb()

        # Split: scrollable content + pinned action bar
        action_bar_h = 56
        right_avail = imgui.get_content_region_available()
        content_h = max(50, right_avail[1] - action_bar_h)

        tab_selected = self._active_section
        imgui.begin_child("TabContentRegion", width=0, height=content_h, border=False)
        if tab_selected == "run":
            self._render_run_control_tab()
        elif tab_selected == "post_processing":
            self._render_post_processing_tab()
        elif tab_selected == "advanced":
            self._render_advanced_tab()
        elif tab_selected == "device_control":
            if self._feat_device:
                self._render_device_control_tab()
            else:
                self._render_locked_feature_placeholder(
                    "Device Control",
                    "Control hardware devices in real-time during playback. "
                    "Supports OSR, Buttplug, and other haptic devices.")
        elif tab_selected == "native_sync":
            if self._feat_streamer:
                self._render_native_sync_tab()
            else:
                self._render_locked_feature_placeholder(
                    "Streamer",
                    "Stream video with synchronized funscript to browsers and VR headsets. "
                    "Built-in web server with HereSphere integration.")
        elif tab_selected == "supporter_batch":
            self._render_supporter_batch_tab()
        imgui.end_child()

        # Pinned action bar at bottom
        self._render_pinned_action_bar()

        imgui.end_child()  # ##RightPanel
        imgui.end()

    def _render_locked_feature_placeholder(self, feature_name, description):
        """Render a placeholder card for locked supporter features."""
        with section_card(f"{feature_name}##Locked", tier="secondary", open_by_default=True) as _:
            imgui.spacing()
            imgui.push_style_color(imgui.COLOR_TEXT, 0.7, 0.7, 0.7, 1.0)
            imgui.text_wrapped(description)
            imgui.pop_style_color()
            imgui.spacing()
            imgui.push_style_color(imgui.COLOR_TEXT, 0.9, 0.75, 0.3, 1.0)
            imgui.text("Available for supporters")
            imgui.pop_style_color()
            imgui.spacing()

    def _render_supporter_batch_tab(self):
        """Render the Patreon Exclusive tab (supporter feature)."""
        if not self._feat_supporter:
            self._render_locked_feature_placeholder(
                "Patreon Exclusive (monthly ko-fi supporter)",
                "Process multiple videos in sequence with a batch queue. "
                "Set up watched folders for automatic processing.")
            return
        # Version info (top of tab, consistent with other supporter modules)
        try:
            import patreon_features
            version = getattr(patreon_features, '__version__', 'unknown')
            imgui.text_colored(f"Patreon Exclusive v{version}", 0.5, 0.5, 0.5, 1.0)
            imgui.spacing()
        except Exception:
            pass
        try:
            from patreon_features.batch.batch_ui import render_batch_panel
            render_batch_panel(self.app)
        except Exception as e:
            imgui.text_colored("Patreon module error", 0.9, 0.3, 0.3, 1.0)
            imgui.text_wrapped(str(e))

    # ------- Tab orchestrators (call into mixins) -------

    def _render_run_control_tab(self):
        app = self.app
        app_state = app.app_state_ui
        stage_proc = app.stage_processor
        fs_proc = app.funscript_processor
        events = app.event_handlers
        # TrackerMode removed - using dynamic discovery system

        # Ensure this is always defined before any conditional UI blocks use it
        processor = app.processor
        disable_combo = (
            stage_proc.full_analysis_active
            or app.is_setting_user_roi_mode
            or (processor and processor.is_processing and not processor.pause_event.is_set())
        )

        # Use dynamic tracker discovery for full mode
        modes_display_full, modes_enum, discovered_trackers_full = self._get_tracker_lists_for_ui(simple_mode=False)

        # Early access tracker gating — annotate gated trackers
        _early_access_set = set()
        _supporter_available = self._feat_supporter
        try:
            from patreon_features.tracker_gating.experimental_gate import is_tracker_early_access, get_early_access_message
            _early_access_set = {name for name in modes_enum if is_tracker_early_access(name)}
        except ImportError:
            pass

        # Build display list with early access annotations
        modes_display_gated = []
        for i, (display, internal) in enumerate(zip(modes_display_full, modes_enum)):
            if internal in _early_access_set and not _supporter_available:
                modes_display_gated.append(f"[Early Access] {display}")
            else:
                modes_display_gated.append(display)

        with section_card("Choose Analysis Method##SimpleAnalysisMethod", tier="primary",
                          accent_color=self.ControlPanelColors.ACTIVE_PROGRESS) as open_:
            if open_:
                modes_display = modes_display_gated

                processor = app.processor
                disable_combo = (
                    stage_proc.full_analysis_active
                    or app.is_setting_user_roi_mode
                    or (processor and processor.is_processing and not processor.pause_event.is_set())
                )
                with _DisabledScope(disable_combo):
                    try:
                        cur_idx = modes_enum.index(app_state.selected_tracker_name)
                    except ValueError:
                        cur_idx = 0
                        app_state.selected_tracker_name = modes_enum[cur_idx]

                    clicked, new_idx = imgui.combo("##TrackerModeCombo", cur_idx, modes_display)
                    self._help_tooltip(self._generate_combined_tooltip(discovered_trackers_full))

                if clicked and new_idx != cur_idx:
                    new_mode = modes_enum[new_idx]
                    # Block selection of early access trackers when patreon_features not available
                    if new_mode in _early_access_set and not _supporter_available:
                        pass  # Selection blocked — tracker is early access only
                    else:
                        # Clear all overlays when switching to a different mode
                        if app_state.selected_tracker_name != new_mode:
                            if hasattr(app, 'logger') and app.logger:
                                app.logger.info(f"UI(RunTab): Mode change requested {app_state.selected_tracker_name} -> {new_mode}. Clearing overlays.")
                            if hasattr(app, 'clear_all_overlays_and_ui_drawings'):
                                app.clear_all_overlays_and_ui_drawings()
                        app_state.selected_tracker_name = new_mode
                        # Persist user choice (store tracker name directly)
                        if hasattr(app, 'app_settings') and hasattr(app.app_settings, 'set'):
                            app.app_settings.set("selected_tracker_name", new_mode)

                        # Set tracker mode using dynamic discovery
                        tr = app.tracker
                        if tr:
                            tr.set_tracking_mode(new_mode)

                # Axis mode combo (merged from Tracking card)
                imgui.spacing()
                imgui.text("Tracking Axes:")
                self._render_tracking_axes_mode(stage_proc)

                proc = app.processor
                video_loaded = proc and proc.is_video_open()
                processing_active = stage_proc.full_analysis_active
                disable_after = (not video_loaded) or processing_active

                with _DisabledScope(disable_after):
                    self._render_execution_progress_display()

        mode = app_state.selected_tracker_name
        if mode and (self._is_offline_tracker(mode) or self._is_live_tracker(mode)):
            if app_state.show_advanced_options:
                with section_card("Analysis Options##RunControlAnalysisOptions",
                                  tier="secondary") as open_:
                    if open_:
                        imgui.text("Analysis Range")
                        self._render_range_selection(stage_proc, fs_proc, events)

                        if self._is_offline_tracker(mode):
                            imgui.text("Stage Reruns:")
                            with _DisabledScope(disable_combo):
                                _, stage_proc.force_rerun_stage1 = imgui.checkbox(
                                    "Force Re-run Stage 1##ForceRerunS1",
                                    stage_proc.force_rerun_stage1,
                                )
                                _tooltip_if_hovered(
                                    "Re-run YOLO object detection even if cached results exist.\n"
                                    "Use when the detection model has been updated."
                                )
                                imgui.same_line()
                                _, stage_proc.force_rerun_stage2_segmentation = imgui.checkbox(
                                    "Force Re-run Stage 2##ForceRerunS2",
                                    stage_proc.force_rerun_stage2_segmentation,
                                )
                                _tooltip_if_hovered(
                                    "Re-run contact analysis and segmentation even if cached results exist.\n"
                                    "Use when you want to regenerate chapters and signals from scratch."
                                )
                                if not hasattr(stage_proc, "save_preprocessed_video"):
                                    stage_proc.save_preprocessed_video = app.app_settings.get("save_preprocessed_video", False)
                                changed, new_val = imgui.checkbox("Save/Reuse Preprocessed Video##SavePreprocessedVideo", stage_proc.save_preprocessed_video)
                                if changed:
                                    stage_proc.save_preprocessed_video = new_val
                                    app.app_settings.set("save_preprocessed_video", new_val)
                                    if new_val:
                                        stage_proc.num_producers_stage1 = 1
                                        app.app_settings.set("num_producers_stage1", 1)
                                _tooltip_if_hovered(
                                    "Saves a preprocessed (resized/unwarped) video for faster re-runs.\n"
                                    "This enables Optical Flow recovery in Stage 2 and is RECOMMENDED for Stage 3 speed.\n"
                                    "Forces the number of Producer threads to 1."
                                )

                            # Database Retention Option
                            with _DisabledScope(disable_combo):
                                retain_database = self.app.app_settings.get("retain_stage2_database", True)
                                changed_db, new_db_val = imgui.checkbox("Keep Stage 2 Database##RetainStage2Database", retain_database)
                                if changed_db:
                                    self.app.app_settings.set("retain_stage2_database", new_db_val)
                            if imgui.is_item_hovered():
                                imgui.set_tooltip(
                                    "Keep the Stage 2 database file after processing completes.\n"
                                    "Disable to save disk space (database is automatically deleted).\n"
                                    "Note: Database is always kept during 3-stage pipelines until Stage 3 completes."
                                )

        proc = app.processor
        video_loaded = proc and proc.is_video_open()
        processing_active = stage_proc.full_analysis_active
        disable_after = (not video_loaded) or processing_active

        self._render_start_stop_buttons(stage_proc, fs_proc, events)

        self._render_interactive_refinement_controls()

        chapters = getattr(app.funscript_processor, "video_chapters", [])
        if chapters:
            # Clear All Chapters button (DESTRUCTIVE - deletes all chapters)
            with destructive_button_style():
                if imgui.button("Clear All Chapters", width=-1):
                    imgui.open_popup("ConfirmClearChapters")
            opened, _ = imgui.begin_popup_modal("ConfirmClearChapters")
            if opened:
                w = imgui.get_window_width()
                text = "Are you sure you want to clear all chapters? This cannot be undone."
                tw = imgui.calc_text_size(text)[0]
                imgui.set_cursor_pos_x((w - tw) * 0.5)
                imgui.text(text)
                imgui.spacing()
                bw, cw = 150, 100
                total = bw + cw + imgui.get_style().item_spacing[0]
                imgui.set_cursor_pos_x((w - total) * 0.5)
                # Confirm button (DESTRUCTIVE - irreversible action)
                with destructive_button_style():
                    if imgui.button("Yes, clear all", width=bw):
                        app.funscript_processor.video_chapters.clear()
                        app.project_manager.project_dirty = True
                        imgui.close_current_popup()
                imgui.same_line()
                if imgui.button("Cancel", width=cw):
                    imgui.close_current_popup()
                imgui.end_popup()

        if disable_after and imgui.is_item_hovered():
            imgui.set_tooltip("Requires a video to be loaded and no other process to be active.")

    def _render_configuration_tab(self):
        app = self.app
        app_state = app.app_state_ui
        tmode = app_state.selected_tracker_name

        imgui.text("Configure settings for the selected mode.")
        imgui.spacing()

        # AI Models & Inference moved to Tools > AI Models dialog

        adv = app.app_state_ui.show_advanced_options
        if self._is_live_tracker(tmode) and adv:
            self._render_live_tracker_settings()

        # TEMPORARILY DISABLE SECTIONS WITH HARDCODED TRACKERMODE REFERENCES
        # TODO: Replace with dynamic discovery logic

        # Class filtering for advanced users
        if (self._is_live_tracker(tmode) or self._is_offline_tracker(tmode)) and adv:
            if imgui.collapsing_header("Class Filtering##ConfigClassFilterHeader")[0]:
                self._render_class_filtering_content()

        # Oscillation detector settings for oscillation trackers
        from config.tracker_discovery import get_tracker_discovery
        discovery = get_tracker_discovery()
        tracker_info = discovery.get_tracker_info(tmode)
        if tracker_info and 'oscillation' in tracker_info.display_name.lower():
            if imgui.collapsing_header("Oscillation Detector Settings##ConfigOscillationDetector", flags=0)[0]:
                self._render_oscillation_detector_settings()

        # Stage 3 specific settings (temporarily disabled - needs proper stage detection)
        # if tmode == "stage3_optical_flow":
        #     if imgui.collapsing_header("Stage 3 Oscillation Detector Mode##ConfigStage3OD", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
        #         self._render_stage3_oscillation_detector_mode_settings()

        # Check if configuration is available for this tracker
        has_config = self._is_live_tracker(tmode) or self._is_offline_tracker(tmode)
        if not has_config:
            imgui.text_disabled("No configuration available for this mode.")

    def _render_settings_tab(self):
        app = self.app
        app_state = app.app_state_ui

        imgui.text("Global application settings. Saved in settings.json.")
        imgui.spacing()

        if imgui.collapsing_header(
            "Interface & Performance##SettingsMenuPerfInterface",
            flags=0,
        )[0]:
            self._render_settings_interface_perf()

        if imgui.collapsing_header(
            "File & Output##SettingsMenuOutput", flags=0
        )[0]:
            self._render_settings_file_output()

        if app_state.show_advanced_options:
            if imgui.collapsing_header("Logging & Autosave##SettingsMenuLogging")[0]:
                self._render_settings_logging_autosave()
        imgui.spacing()

        # Reset All Settings button (DESTRUCTIVE - resets all settings)
        with destructive_button_style():
            if imgui.button("Reset All Settings to Default##ResetAllSettingsButton", width=-1):
                imgui.open_popup("Confirm Reset##ResetSettingsPopup")

        if imgui.begin_popup_modal(
            "Confirm Reset##ResetSettingsPopup", True, imgui.WINDOW_ALWAYS_AUTO_RESIZE
        )[0]:
            imgui.text(
                "This will reset all application settings to their defaults.\n"
                "Your projects will not be affected.\n"
                "This action cannot be undone."
            )

            avail_w = imgui.get_content_region_available_width()
            pw = (avail_w - imgui.get_style().item_spacing[0]) / 2.0

            # Confirm Reset button (DESTRUCTIVE - irreversible action)
            with destructive_button_style():
                if imgui.button("Confirm Reset", width=pw):
                    app.app_settings.reset_to_defaults()
                    app.logger.info("All settings have been reset to default.", extra={"status_message": True})
                    imgui.close_current_popup()

            imgui.same_line()
            if imgui.button("Cancel", width=pw):
                imgui.close_current_popup()
            imgui.end_popup()
