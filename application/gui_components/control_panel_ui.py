import imgui
import config
from config.constants_colors import CurrentTheme
from config.element_group_colors import SidebarColors as _SidebarColors
from application.utils import get_icon_texture_manager, primary_button_style, destructive_button_style
from application.utils.feature_detection import is_feature_available as _is_feature_available
from application.utils.imgui_helpers import DisabledScope as _DisabledScope, tooltip_if_hovered as _tooltip_if_hovered
from application.utils.section_card import section_card

# Import dynamic tracker discovery
try:
    from .dynamic_tracker_ui import DynamicTrackerUI
    from config.tracker_discovery import get_tracker_discovery, TrackerCategory
except ImportError:
    DynamicTrackerUI = None
    TrackerCategory = None

# Early access tracker gating (optional add-on)
try:
    from patreon_features.tracker_gating.experimental_gate import is_tracker_early_access as _is_tracker_early_access, get_early_access_message as _get_early_access_message
    _HAS_EARLY_ACCESS_GATE = True
except ImportError:
    _HAS_EARLY_ACCESS_GATE = False
    _is_tracker_early_access = None
    _get_early_access_message = None

# Import mixin sub-modules
from .cp_post_processing_ui import PostProcessingMixin
from .cp_ai_models_ui import AIModelsMixin
from .cp_execution_ui import ExecutionMixin
from .cp_tracker_settings_ui import TrackerSettingsMixin
from .cp_device_control_ui import DeviceControlMixin
from .cp_streamer_ui import StreamerMixin
from .cp_batch_ui import BatchMixin
from .cp_metadata_ui import MetadataEditorMixin

def _readonly_input(label_id, value, width=-1):
    if width is not None and width >= 0:
        imgui.push_item_width(width)
    imgui.input_text(label_id, value or "Not set", 256, flags=imgui.INPUT_TEXT_READ_ONLY)
    if width is not None and width >= 0:
        imgui.pop_item_width()


class ControlPanelUI(
    PostProcessingMixin,
    AIModelsMixin,
    ExecutionMixin,
    TrackerSettingsMixin,
    DeviceControlMixin,
    StreamerMixin,
    BatchMixin,
    MetadataEditorMixin,
):
    def __init__(self, app):
        self.app = app
        self.timeline_editor1 = None
        self.timeline_editor2 = None
        self.ControlPanelColors = config.ControlPanelColors
        self.constants = config.constants
        self.AI_modelExtensionsFilter = self.constants.AI_MODEL_EXTENSIONS_FILTER
        self.AI_modelTooltipExtensions = self.constants.AI_MODEL_TOOLTIP_EXTENSIONS

        # Initialize dynamic tracker UI helper
        self.tracker_ui = None
        self._try_reinitialize_tracker_ui()

        # Initialize device control attributes (add-on feature)
        self.device_manager = None
        self.param_manager = None
        self._device_control_initialized = False
        self._device_control_init_error = None   # stores last init exception for UI display
        self._streamer_init_error = None          # stores last streamer init exception for UI display
        self._first_frame_rendered = False
        self.video_playback_bridge = None  # Video playback bridge for live control
        self.live_tracker_bridge = None    # Live tracker bridge for real-time control
        self._available_osr_ports = []
        self._osr_scan_performed = False

        # Device video integration (observer pattern)
        self.device_video_integration = None
        self.device_video_bridge = None
        self.device_bridge_thread = None

        # Buttplug device discovery UI state
        self._discovered_buttplug_devices = []
        self._buttplug_discovery_performed = False

        # OSSM BLE device discovery UI state
        self._discovered_ossm_devices = []
        self._ossm_scan_performed = False

        # Unified axis configuration panel state
        self._axis_details_expanded = {}  # keyed by "{device_id}_{channel}"

        # Streamer attributes (add-on feature)
        self._native_sync_manager = None
        self._prev_client_count = 0
        self._native_sync_status_cache = None
        self._native_sync_status_time = 0

        # Active sidebar section (replaces tab bar)
        self._active_section = "run"

        # Tracker filter row toggle (ephemeral, resets each session)
        self._tracker_filter_open = False

        # Cached tracker lists (invalidated when filter settings change)
        self._cached_tracker_hidden_folders = None
        self._cached_tracker_lists = None       # (modes_display_full, modes_enum, discovered)
        self._cached_tracker_tooltip = None
        self._cached_tracker_gated = None
        self._cached_tracker_supporter_flag = None

        # Batch/Capture state (from BatchMixin)
        self._init_batch_state()

    # ------- Helpers -------

    @staticmethod
    def _render_panel_label(text):
        """Render a dim uppercase centered label at the top of the panel (toolbar-style)."""
        label = text.upper()
        text_size = imgui.calc_text_size(label)
        avail_w = imgui.get_content_region_available_width()
        x_offset = (avail_w - text_size[0]) * 0.5
        if x_offset > 0:
            imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + x_offset)
        imgui.text_colored(label, 0.45, 0.45, 0.50, 0.7)
        imgui.spacing()

    def _render_addon_version_label(self, module_name, display_name):
        """Render a dim version label for an addon module."""
        try:
            mod = __import__(module_name)
            version = getattr(mod, '__version__', 'unknown')
            imgui.text_colored(f"{display_name} v{version}", 0.5, 0.5, 0.5, 1.0)
            imgui.spacing()
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Failed to read {module_name} version: {e}")

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
        """Check if tracker is a live tracker (LIVE, LIVE_INTERVENTION, or COMMUNITY)."""
        from config.tracker_discovery import get_tracker_discovery, TrackerCategory
        discovery = get_tracker_discovery()
        tracker_info = discovery.get_tracker_info(tracker_name)
        return tracker_info and tracker_info.category in [
            TrackerCategory.LIVE, TrackerCategory.LIVE_INTERVENTION, TrackerCategory.COMMUNITY]

    def _is_offline_tracker(self, tracker_name: str) -> bool:
        """Check if tracker is an offline tracker."""
        from config.tracker_discovery import TrackerCategory
        return self._is_tracker_category(tracker_name, TrackerCategory.OFFLINE)
    def _check_tracker_ui(self, method_name: str, tracker_name: str) -> bool:
        """Dispatch a tracker-query method on tracker_ui, with lazy init."""
        if not self.tracker_ui:
            self._try_reinitialize_tracker_ui()
        if self.tracker_ui:
            return getattr(self.tracker_ui, method_name)(tracker_name)
        if hasattr(self.app, 'logger'):
            self.app.logger.warning(f"Dynamic tracker UI not available for {method_name}('{tracker_name}')")
        return False

    def _is_stage2_tracker(self, tracker_name: str) -> bool:
        return self._check_tracker_ui('is_stage2_tracker', tracker_name)

    def _is_stage3_tracker(self, tracker_name: str) -> bool:
        return self._check_tracker_ui('is_stage3_tracker', tracker_name)

    def _is_mixed_stage3_tracker(self, tracker_name: str) -> bool:
        return self._check_tracker_ui('is_mixed_stage3_tracker', tracker_name)

    def _is_hybrid_tracker(self, tracker_name: str) -> bool:
        return self._check_tracker_ui('is_hybrid_tracker', tracker_name)

    def _tracker_produces_funscript(self, tracker_name: str) -> bool:
        """Check if any of the tracker's stages produce a funscript."""
        info = self.tracker_ui.discovery.get_tracker_info(tracker_name) if self.tracker_ui else None
        if info and info.stages:
            return any(s.produces_funscript for s in info.stages)
        # Live trackers always produce funscript; default True for safety
        return True

    def _get_tracker_display_name(self, tracker_name: str) -> str:
        return self.tracker_ui.get_tracker_display_name(tracker_name)

    def _get_tracker_property(self, tracker_name: str, prop: str, default=None):
        """Get a property value from a tracker's metadata."""
        info = self.tracker_ui.discovery.get_tracker_info(tracker_name) if self.tracker_ui else None
        if info and info.properties:
            return info.properties.get(prop, default)
        return default

    def _get_tracker_lists_for_ui(self, hidden_folders=None):
        """Get tracker lists for UI combo boxes using dynamic discovery."""
        try:
            if hidden_folders:
                # Full mode with category filtering
                display_names, internal_names = self.tracker_ui.get_gui_display_list_filtered(hidden_folders)
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
        ("run", "R", "Analysis"),
        ("post", "P", "Post-Processing"),
        ("metadata", "M", "Metadata"),
    ]
    _SIDEBAR_ADDON_SECTIONS = [
        ("device_control", "D", "Device Control", "_feat_device"),
        ("native_sync", "S", "Streamer", "_feat_streamer"),
        ("supporter_batch", "B", "Batch Processing", "_feat_supporter"),
    ]
    # Map section keys to icon asset filenames for sidebar PNG icons
    _SIDEBAR_ICON_MAP = {
        "run": "sidebar-run.png",
        "post": "sidebar-postproc.png",
        "device_control": "sidebar-device.png",
        "native_sync": "sidebar-stream.png",
        "supporter_batch": "sidebar-batch.png",
        "metadata": "sidebar-metadata.png",
    }

    def _render_sidebar(self, total_h):
        """Render vertical icon sidebar for section navigation.

        Returns the active section key.
        """
        SidebarColors = _SidebarColors

        sidebar_w = self._SIDEBAR_WIDTH
        draw_list = imgui.get_window_draw_list()

        imgui.begin_child("##Sidebar", width=sidebar_w, height=total_h, border=False)

        # Top padding to align first sidebar icon with toolbar icons
        _SIDEBAR_TOP_PAD = 6
        imgui.dummy(0, _SIDEBAR_TOP_PAD)

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

        # Add-on sections
        for key, icon, tooltip, feat_attr in self._SIDEBAR_ADDON_SECTIONS:
            available = getattr(self, feat_attr, False)
            is_active = (active_section == key)
            self._render_sidebar_entry(draw_list, key, icon, tooltip, is_active,
                                       available=available, btn_size=btn_size, sidebar_w=sidebar_w)

        imgui.end_child()
        return self._active_section

    # Sidebar feature key → feature_detection name (for locked tooltip)
    _SIDEBAR_FEATURE_MAP = {
        "device_control": "device_control",
        "native_sync": "streamer",
        "supporter_batch": "patreon_features",
    }

    def _render_sidebar_entry(self, draw_list, key, icon, tooltip, is_active,
                               available, btn_size, sidebar_w):
        """Render a single sidebar navigation entry."""
        SidebarColors = _SidebarColors

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
        if imgui.invisible_button(f"##SB_{key}", btn_size, btn_size) and available:
            self._active_section = key

        # Check hover state BEFORE drawing so bg goes behind text
        is_hovered = imgui.is_item_hovered()
        if is_hovered:
            if available:
                imgui.set_tooltip(tooltip)
            else:
                # Show FeatureInfo description alongside locked message
                feat_name = self._SIDEBAR_FEATURE_MAP.get(key)
                desc = ""
                if feat_name:
                    from application.utils.feature_detection import get_feature_detector
                    info = get_feature_detector().get_feature_info(feat_name)
                    if info:
                        desc = f"\n{info.description}"
                imgui.set_tooltip(f"{tooltip}{desc}\n(Add-on available at ko-fi.com/k00gar)")
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

    # ------- Pinned Action Bar -------

    def _render_pinned_action_bar(self):
        """Render contextual primary action button pinned at bottom of control panel."""
        app = self.app
        proc = app.processor
        stage_proc = app.stage_processor
        events = app.event_handlers

        # Pin action bar to bottom of visible window area.
        content_max_y = imgui.get_window_content_region_max()[1]
        bar_h = 46  # separator + spacing + button (32) + margin
        target_y = content_max_y - bar_h
        imgui.set_cursor_pos_y(max(imgui.get_cursor_pos_y(), target_y))

        imgui.separator()
        imgui.spacing()

        video_loaded = proc and proc.is_video_open()
        is_offline_active = stage_proc.full_analysis_active
        is_live_active = (proc and getattr(proc, 'is_processing', False)
                         and getattr(proc, 'enable_tracker_processing', False))
        is_playback_active = (proc and getattr(proc, 'is_processing', False)
                              and not getattr(proc, 'enable_tracker_processing', False))

        if not video_loaded:
            # No video loaded — show Load Video
            with primary_button_style():
                if imgui.button("Load Video##PinnedAction", width=-1, height=32):
                    if hasattr(events, 'trigger_open_video_dialog'):
                        events.trigger_open_video_dialog()
                    elif hasattr(app, 'file_manager') and hasattr(app.file_manager, 'open_video_dialog'):
                        app.file_manager.open_video_dialog()
        elif is_offline_active:
            # Offline analysis active — Stop button only (progress shown in execution display above)
            with destructive_button_style():
                if imgui.button("Stop Analysis##PinnedAction", width=-1, height=32):
                    events.handle_abort_process_click()
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
        elif is_playback_active:
            # Simple video playback — show Pause/Resume button
            is_paused = hasattr(proc, 'pause_event') and proc.pause_event.is_set()
            if is_paused:
                with primary_button_style():
                    if imgui.button("Resume Playback##PinnedAction", width=-1, height=32):
                        events.handle_playback_control("play_pause")
            else:
                if imgui.button("Pause Playback##PinnedAction", width=-1, height=32):
                    events.handle_playback_control("play_pause")
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
        # Cache feature detection flags for this frame
        self._feat_supporter = _is_feature_available("patreon_features")
        self._feat_device = _is_feature_available("device_control")
        self._feat_streamer = _is_feature_available("streamer")

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
            self._render_panel_label("AI ANALYSIS")

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

        # Split: scrollable content + pinned action bar
        action_bar_h = 56
        right_avail = imgui.get_content_region_available()
        content_h = max(50, right_avail[1] - action_bar_h)

        tab_selected = self._active_section
        imgui.begin_child("TabContentRegion", width=0, height=content_h, border=False)
        if tab_selected == "run":
            self._render_run_control_tab()
        elif tab_selected == "post":
            self._render_post_processing_tab()
        elif tab_selected == "device_control":
            if self._feat_device:
                self._render_device_control_tab()
            else:
                self._render_device_control_preview()
        elif tab_selected == "native_sync":
            if self._feat_streamer:
                self._render_native_sync_tab()
            else:
                self._render_streamer_preview()
        elif tab_selected == "metadata":
            self._render_metadata_tab()
        elif tab_selected == "supporter_batch":
            self._render_supporter_batch_tab()
        imgui.end_child()

        # Pinned action bar at bottom
        self._render_pinned_action_bar()

        imgui.end_child()  # ##RightPanel
        imgui.end()

    def _render_locked_feature_placeholder(self, feature_name, description):
        """Render a placeholder card for locked add-on features."""
        with section_card(f"{feature_name}##Locked", tier="secondary", open_by_default=True) as _:
            imgui.spacing()
            imgui.push_style_color(imgui.COLOR_TEXT, 0.7, 0.7, 0.7, 1.0)
            imgui.text_wrapped(description)
            imgui.pop_style_color()
            imgui.spacing()
            imgui.push_style_color(imgui.COLOR_TEXT, 0.9, 0.75, 0.3, 1.0)
            imgui.text("Add-on available at ko-fi.com/k00gar")
            imgui.pop_style_color()
            imgui.spacing()

    def _render_addon_promo_banner(self, feature_name, description):
        """Gold add-on banner with feature description and ko-fi link."""
        imgui.spacing()
        # Gold accent bar
        draw_list = imgui.get_window_draw_list()
        cursor = imgui.get_cursor_screen_pos()
        avail_w = imgui.get_content_region_available_width()
        bar_color = imgui.get_color_u32_rgba(0.9, 0.75, 0.3, 0.8)
        draw_list.add_rect_filled(cursor[0], cursor[1], cursor[0] + avail_w, cursor[1] + 3, bar_color, 1.0)
        imgui.dummy(0, 6)

        imgui.push_style_color(imgui.COLOR_TEXT, 0.9, 0.75, 0.3, 1.0)
        imgui.text(f"{feature_name} — Add-on at ko-fi.com/k00gar")
        imgui.pop_style_color()
        imgui.push_style_color(imgui.COLOR_TEXT, 0.7, 0.7, 0.7, 1.0)
        imgui.text_wrapped(description)
        imgui.pop_style_color()
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

    def _render_device_control_preview(self):
        """Render a preview of the Device Control add-on."""
        self._render_addon_promo_banner(
            "Device Control",
            "Control hardware devices in real-time during playback. "
            "Supports OSR2/OSR6, Buttplug.io, Handy, and OSSM devices."
        )
        with _DisabledScope(True):
            imgui.text_colored("Device: Not Connected", 0.7, 0.3, 0.3, 1.0)
            imgui.separator()
            imgui.text("Connect a Device:")
            imgui.spacing()

            # OSR2/OSR6
            with section_card("OSR2/OSR6 (USB)##PreviewOSR", tier="primary") as is_open:
                if is_open:
                    imgui.text("Serial Port:")
                    imgui.push_item_width(150)
                    imgui.combo("##PreviewOSRPort", 0, ["Select port..."])
                    imgui.pop_item_width()
                    imgui.same_line()
                    imgui.button("Scan Ports")

            # Buttplug.io
            with section_card("Buttplug.io (Universal)##PreviewButtplug", tier="primary") as is_open:
                if is_open:
                    imgui.text("Server Address:")
                    imgui.push_item_width(200)
                    imgui.input_text("##PreviewBPAddr", "ws://127.0.0.1:12345", 256, flags=imgui.INPUT_TEXT_READ_ONLY)
                    imgui.pop_item_width()
                    imgui.button("Connect##PreviewBPConnect")

            # Handy
            with section_card("Handy (Direct)##PreviewHandy", tier="primary") as is_open:
                if is_open:
                    imgui.text("Connection Key:")
                    imgui.push_item_width(200)
                    imgui.input_text("##PreviewHandyKey", "", 256, flags=imgui.INPUT_TEXT_READ_ONLY)
                    imgui.pop_item_width()
                    imgui.button("Connect##PreviewHandyConnect")

            # OSSM
            with section_card("OSSM (Bluetooth)##PreviewOSSM", tier="primary") as is_open:
                if is_open:
                    imgui.button("Scan for OSSM Devices")

            imgui.separator()
            # Axis Configuration
            with section_card("Axis Configuration##PreviewAxisConfig", tier="secondary",
                              open_by_default=False) as is_open:
                if is_open:
                    imgui.text_colored("Connect a device to configure axes", 0.5, 0.5, 0.5, 1.0)

            # Live Control Integration
            with section_card("Live Control Integration##PreviewLiveControl", tier="secondary",
                              open_by_default=False) as is_open:
                if is_open:
                    imgui.text_colored("Enables real-time device control during video playback", 0.5, 0.5, 0.5, 1.0)

            # Device Playback
            with section_card("Device Playback##PreviewPlayback", tier="secondary",
                              open_by_default=False) as is_open:
                if is_open:
                    imgui.text_colored("Play funscripts on connected devices", 0.5, 0.5, 0.5, 1.0)

    def _render_streamer_preview(self):
        """Render a preview of the Streamer add-on."""
        self._render_addon_promo_banner(
            "Video Streamer",
            "Stream video to browsers and VR headsets with frame-perfect synchronization. "
            "Built-in web server with HereSphere, XBVR, and Stash integration."
        )
        with _DisabledScope(True):
            # Server Control
            with section_card("Server Control##PreviewSyncControl", tier="primary") as is_open:
                if is_open:
                    imgui.push_text_wrap_pos(imgui.get_content_region_available_width())
                    imgui.text_colored(
                        "Stream video to browsers/VR headsets with frame-perfect synchronization. "
                        "Supports zoom/pan controls, speed modes, and interactive device control.",
                        0.7, 0.7, 0.7
                    )
                    imgui.pop_text_wrap_pos()
                    imgui.spacing()
                    imgui.button("Start Streaming Server", width=-1)

            # Display Options
            with section_card("Display Options##PreviewSyncDisplay", tier="secondary",
                              open_by_default=False) as is_open:
                if is_open:
                    imgui.checkbox("Auto-hide Video Feed while streaming##PreviewAutoHide", False)

            # Rolling Autotune
            with section_card("Rolling Autotune (Live Tracking)##PreviewRollingAT", tier="secondary",
                              open_by_default=False) as is_open:
                if is_open:
                    imgui.checkbox("Enable Rolling Autotune##PreviewRAT", False)
                    imgui.text("Interval (seconds):")
                    imgui.push_item_width(100)
                    imgui.slider_float("##PreviewRATInterval", 10.0, 5.0, 60.0, "%.0f")
                    imgui.pop_item_width()

            # XBVR Integration
            with section_card("XBVR Integration##PreviewXBVR", tier="primary") as is_open:
                if is_open:
                    imgui.push_text_wrap_pos(imgui.get_content_region_available_width())
                    imgui.text_colored(
                        "Browse your XBVR library in the VR viewer with scene thumbnails and funscript availability.",
                        0.7, 0.7, 0.7
                    )
                    imgui.pop_text_wrap_pos()
                    imgui.spacing()
                    imgui.text("XBVR Host/IP:")
                    imgui.push_item_width(200)
                    imgui.input_text("##PreviewXBVRHost", "localhost", 256, flags=imgui.INPUT_TEXT_READ_ONLY)
                    imgui.pop_item_width()
                    imgui.text("XBVR Port:")
                    imgui.push_item_width(100)
                    imgui.input_text("##PreviewXBVRPort", "9999", 256, flags=imgui.INPUT_TEXT_READ_ONLY)
                    imgui.pop_item_width()

            # Stash Integration
            with section_card("Stash Integration##PreviewStash", tier="primary") as is_open:
                if is_open:
                    imgui.push_text_wrap_pos(imgui.get_content_region_available_width())
                    imgui.text_colored(
                        "Browse and load videos from your Stash library directly in the VR viewer.",
                        0.7, 0.7, 0.7
                    )
                    imgui.pop_text_wrap_pos()
                    imgui.spacing()
                    imgui.text("Stash Host/IP:")
                    imgui.push_item_width(200)
                    imgui.input_text("##PreviewStashHost", "localhost", 256, flags=imgui.INPUT_TEXT_READ_ONLY)
                    imgui.pop_item_width()
                    imgui.text("Stash Port:")
                    imgui.push_item_width(100)
                    imgui.input_text("##PreviewStashPort", "9999", 256, flags=imgui.INPUT_TEXT_READ_ONLY)
                    imgui.pop_item_width()

            # Requirements
            with section_card("Requirements##PreviewSyncReqs", tier="secondary") as is_open:
                if is_open:
                    imgui.bullet_text("Ports 8080 (HTTP) and 8765 (WebSocket) available")
                    imgui.bullet_text("Browser with HTML5 video support")
                    imgui.bullet_text("Video can be loaded before or after starting the server")

            # Features
            with section_card("Features##PreviewSyncFeatures", tier="secondary",
                              open_by_default=False) as is_open:
                if is_open:
                    imgui.bullet_text("Native hardware H.265/AV1 decode")
                    imgui.bullet_text("Zoom/Pan controls (+/- and WASD keys)")
                    imgui.bullet_text("Speed modes (Real Time / Slo Mo)")
                    imgui.bullet_text("Real-time FPS and resolution stats")
                    imgui.bullet_text("Interactive device control")
                    imgui.bullet_text("Funscript visualization graph")

    # ------- Tab orchestrators (call into mixins) -------

    def _render_run_control_tab(self):
        app = self.app
        app_state = app.app_state_ui
        stage_proc = app.stage_processor
        fs_proc = app.funscript_processor
        events = app.event_handlers
        # Build set of hidden tracker folders from settings
        _settings = app.app_settings
        _hidden_folders = set()
        if not _settings.get("tracker_show_legacy", False):
            _hidden_folders.add("legacy")
        if not _settings.get("tracker_show_experimental", True):
            _hidden_folders.add("experimental")
        if not _settings.get("tracker_show_community", True):
            _hidden_folders.add("community")

        _supporter_available = self._feat_supporter

        # Use cached tracker lists — invalidate when filter settings or supporter status change
        _hidden_key = frozenset(_hidden_folders)
        if (self._cached_tracker_lists is None
                or self._cached_tracker_hidden_folders != _hidden_key
                or self._cached_tracker_supporter_flag != _supporter_available):
            # Recompute tracker lists
            modes_display_full, modes_enum, discovered_trackers_full = self._get_tracker_lists_for_ui(
                hidden_folders=_hidden_folders
            )

            # Early access tracker gating — annotate gated trackers
            _early_access_set = set()
            if _HAS_EARLY_ACCESS_GATE:
                _early_access_set = {name for name in modes_enum if _is_tracker_early_access(name)}

            # Build display list with early access annotations
            modes_display_gated = []
            for i, (display, internal) in enumerate(zip(modes_display_full, modes_enum)):
                if internal in _early_access_set and not _supporter_available:
                    modes_display_gated.append(f"[Early Access] {display}")
                else:
                    modes_display_gated.append(display)

            self._cached_tracker_lists = (modes_display_full, modes_enum, discovered_trackers_full)
            self._cached_tracker_gated = modes_display_gated
            self._cached_tracker_early_access = _early_access_set
            self._cached_tracker_tooltip = self._generate_combined_tooltip(discovered_trackers_full)
            self._cached_tracker_hidden_folders = _hidden_key
            self._cached_tracker_supporter_flag = _supporter_available

        modes_display_full, modes_enum, discovered_trackers_full = self._cached_tracker_lists
        modes_display_gated = self._cached_tracker_gated
        _early_access_set = getattr(self, '_cached_tracker_early_access', set())

        processor = app.processor
        disable_combo = (
            stage_proc.full_analysis_active
            or app.is_setting_user_roi_mode
            or (processor and processor.is_processing
                and getattr(processor, 'enable_tracker_processing', False)
                and not processor.pause_event.is_set())
        )

        with section_card("Analysis Method##RunControlAnalysisMethod",
                          tier="primary") as open_:
            if open_:
                modes_display = modes_display_gated

                # --- Tracker combo + settings icon button on same line ---
                style = imgui.get_style()
                icon_size = imgui.get_frame_height() - style.frame_padding[1] * 2
                btn_w = icon_size + style.frame_padding[0] * 2
                combo_width = imgui.get_content_region_available_width() - btn_w - style.item_spacing.x

                with _DisabledScope(disable_combo):
                    try:
                        cur_idx = modes_enum.index(app_state.selected_tracker_name)
                    except ValueError:
                        cur_idx = 0
                        app_state.selected_tracker_name = modes_enum[cur_idx]

                    imgui.set_next_item_width(combo_width)
                    clicked, new_idx = imgui.combo("##TrackerModeCombo", cur_idx, modes_display)
                    self._help_tooltip(self._cached_tracker_tooltip)

                imgui.same_line()
                icon_mgr = get_icon_texture_manager()
                settings_tex, _, _ = icon_mgr.get_icon_texture('settings.png')
                if settings_tex and imgui.image_button(settings_tex, icon_size, icon_size):
                    self._tracker_filter_open = not self._tracker_filter_open
                elif not settings_tex and imgui.button("...##TrkFilter"):
                    self._tracker_filter_open = not self._tracker_filter_open
                _tooltip_if_hovered("Filter tracker categories")

                # --- Collapsible filter row ---
                _chk_legacy = _settings.get("tracker_show_legacy", False)
                _chk_exp = _settings.get("tracker_show_experimental", True)
                _chk_comm = _settings.get("tracker_show_community", True)
                if self._tracker_filter_open:
                    imgui.text_disabled("Show:")
                    imgui.same_line()
                    ch_l, nv_l = imgui.checkbox("Legacy##TrkFilterLeg", _chk_legacy)
                    if ch_l:
                        _settings.set("tracker_show_legacy", nv_l)
                    imgui.same_line()
                    ch_e, nv_e = imgui.checkbox("Exp.##TrkFilterExp", _chk_exp)
                    if ch_e:
                        _settings.set("tracker_show_experimental", nv_e)
                    imgui.same_line()
                    ch_c, nv_c = imgui.checkbox("Comm.##TrkFilterComm", _chk_comm)
                    if ch_c:
                        _settings.set("tracker_show_community", nv_c)

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
                        # Toast notification for tracker switch
                        display = self._get_tracker_display_name(new_mode) if self.tracker_ui else new_mode
                        app.notify(f"Switched to {display}", "success")

                # Tracker info line: version + description
                tracker_info = self.tracker_ui.discovery.get_tracker_info(app_state.selected_tracker_name) if self.tracker_ui else None
                tracker_version = self._get_tracker_property(app_state.selected_tracker_name, "version", None)
                if not tracker_version and tracker_info:
                    tracker_version = tracker_info.version

                info_parts = []
                if tracker_version:
                    info_parts.append(f"v{tracker_version}")
                if tracker_info and tracker_info.description:
                    info_parts.append(tracker_info.description)
                if info_parts:
                    imgui.push_style_color(imgui.COLOR_TEXT, 0.5, 0.5, 0.6, 1.0)
                    imgui.text_wrapped(" - ".join(info_parts))
                    imgui.pop_style_color()

                # Axis mode (only show if tracker produces funscript)
                if self._tracker_produces_funscript(app_state.selected_tracker_name):
                    imgui.spacing()
                    imgui.separator()
                    imgui.spacing()
                    imgui.text("Tracking Axes")
                    self._render_tracking_axes_mode(stage_proc)

                # Clear All Chapters (inside the card, only when chapters exist)
                chapters = getattr(app.funscript_processor, "video_chapters", [])
                if chapters:
                    imgui.spacing()
                    imgui.separator()
                    imgui.spacing()
                    with destructive_button_style():
                        if imgui.button("Clear All Chapters##RunTab", width=-1):
                            imgui.open_popup("Clear All Chapters?###ConfirmClearChaptersRun")
                    imgui.set_next_window_size(380, 0)
                    opened, _ = imgui.begin_popup_modal("Clear All Chapters?###ConfirmClearChaptersRun")
                    if opened:
                        imgui.text_wrapped("Are you sure you want to clear all chapters?\nThis cannot be undone.")
                        imgui.spacing()
                        w = imgui.get_content_region_available()[0]
                        bw, cw = 150, 100
                        total = bw + cw + imgui.get_style().item_spacing[0]
                        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + (w - total) * 0.5)
                        with destructive_button_style():
                            if imgui.button("Yes, clear all##Run", width=bw):
                                app.funscript_processor.video_chapters.clear()
                                app.project_manager.project_dirty = True
                                app.notify("All chapters cleared", "info", 2.0)
                                imgui.close_current_popup()
                        imgui.same_line()
                        if imgui.button("Cancel##Run", width=cw):
                            imgui.close_current_popup()
                        imgui.end_popup()

        mode = app_state.selected_tracker_name
        # During batch processing, use the batch tracker name for progress display
        if app.is_batch_processing_active and hasattr(app, 'batch_tracker_name') and app.batch_tracker_name:
            mode = app.batch_tracker_name

        # Batch progress card (always visible during batch processing)
        if app.is_batch_processing_active and getattr(app, 'batch_video_paths', None):
            imgui.spacing()
            total = len(app.batch_video_paths)
            current = getattr(app, 'current_batch_video_index', 0) + 1
            if current < 1:
                current = 1
            with section_card(f"Batch Processing ({current}/{total})##BatchProgress", tier="secondary") as batch_open:
                if batch_open:
                    import os as _os
                    video_name = ""
                    idx = getattr(app, 'current_batch_video_index', 0)
                    if 0 <= idx < total:
                        video_name = _os.path.basename(app.batch_video_paths[idx].get("path", ""))
                    if video_name:
                        imgui.text_wrapped(video_name)
                    progress = current / max(1, total)
                    imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, 0.3, 0.65, 1.0, 1.0)
                    imgui.progress_bar(progress, size=(-1, 0), overlay=f"{current}/{total}")
                    imgui.pop_style_color()

        # ROI controls for live trackers that require user intervention (e.g., User ROI)
        if self._is_live_tracker(mode):
            tracker_info = app.tracker.get_tracker_info() if app.tracker else None
            if tracker_info and getattr(tracker_info, 'requires_intervention', False):
                self._render_user_roi_controls_for_run_tab()

        # Progress card (only during analysis)
        if self._is_offline_tracker(mode) and stage_proc.full_analysis_active:
            imgui.spacing()
            with section_card("Progress##RunControlProgress", tier="secondary") as progress_open:
                if progress_open:
                    self._render_execution_progress_display()

        # Force re-run option for offline trackers (only when idle)
        if self._is_offline_tracker(mode) and not stage_proc.full_analysis_active:
            _, stage_proc.force_rerun_stage1 = imgui.checkbox(
                "Force re-run (ignore cache)##FRCard", stage_proc.force_rerun_stage1)

        # Analysis range
        tracker_supports_range = self._get_tracker_property(mode, "supports_range", True)
        if mode and tracker_supports_range and not stage_proc.full_analysis_active:
            with section_card("Analysis Range##RunControlAnalysisRange",
                              tier="primary", open_by_default=False) as open_:
                if open_:
                    self._render_range_selection(stage_proc, fs_proc, events)

    def _render_post_processing_tab(self):
        """Render the Post-Processing sidebar section content."""
        app = self.app
        app_state = app.app_state_ui

        # Plugin Pipeline
        with section_card("Plugin Pipeline##PostProcPipeline", tier="primary") as pp_open:
            if pp_open:
                self._render_post_analysis_section(app, app_state)

        # Chapters
        chapters = getattr(app.funscript_processor, "video_chapters", [])
        if chapters:
            with section_card("Chapters##PostProcChapters", tier="primary") as ch_open:
                if ch_open:
                    with destructive_button_style():
                        if imgui.button("Clear All Chapters", width=-1):
                            imgui.open_popup("Clear All Chapters?###ConfirmClearChapters")
                    imgui.set_next_window_size(380, 0)
                    opened, _ = imgui.begin_popup_modal("Clear All Chapters?###ConfirmClearChapters")
                    if opened:
                        imgui.text_wrapped("Are you sure you want to clear all chapters?\nThis cannot be undone.")
                        imgui.spacing()
                        w = imgui.get_content_region_available()[0]
                        bw, cw = 150, 100
                        total = bw + cw + imgui.get_style().item_spacing[0]
                        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + (w - total) * 0.5)
                        with destructive_button_style():
                            if imgui.button("Yes, clear all", width=bw):
                                app.funscript_processor.video_chapters.clear()
                                app.project_manager.project_dirty = True
                                imgui.close_current_popup()
                        imgui.same_line()
                        if imgui.button("Cancel", width=cw):
                            imgui.close_current_popup()
                        imgui.end_popup()

    def _render_post_analysis_section(self, app, app_state):
        """Render the Post-Analysis section: auto-apply toggle + pipeline flow visualization."""
        settings = app.app_settings

        # --- Auto-apply checkbox ---
        auto_on = settings.get("auto_apply_post_processing", True)
        changed, auto_on = imgui.checkbox("Auto-apply post-processing", auto_on)
        if changed:
            settings.set("auto_apply_post_processing", auto_on)
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Automatically run the pipeline below after analysis completes.\n"
                "Full video: applies to entire result.\n"
                "Range analysis: applies to analyzed range only."
            )

        imgui.spacing()

        # --- Pipeline flow visualization ---
        gui = getattr(app, 'gui_instance', None)
        pipeline = gui.plugin_pipeline_ui.pipeline if gui and hasattr(gui, 'plugin_pipeline_ui') else None

        dl = imgui.get_window_draw_list()
        avail_w = imgui.get_content_region_available_width()
        cursor = imgui.get_cursor_screen_position()

        step_h = 22
        arrow_h = 12
        box_rounding = 4.0
        pad_x = 4

        # Collect labels: raw + enabled pipeline steps + final
        flow_labels = ["Raw script"]
        if pipeline and pipeline.steps:
            for s in pipeline.steps:
                if s.enabled:
                    flow_labels.append(s.plugin_name)
        flow_labels.append("Final script")

        # Colors
        col_raw = imgui.get_color_u32_rgba(0.45, 0.55, 0.70, 1.0)
        col_step = imgui.get_color_u32_rgba(0.30, 0.55, 0.80, 1.0)
        col_final = imgui.get_color_u32_rgba(0.25, 0.75, 0.40, 1.0)
        col_bg = imgui.get_color_u32_rgba(0.15, 0.15, 0.18, 0.6)
        col_arrow = imgui.get_color_u32_rgba(0.4, 0.4, 0.45, 0.7)
        col_text = imgui.get_color_u32_rgba(0.85, 0.85, 0.88, 1.0)
        col_disabled = imgui.get_color_u32_rgba(0.4, 0.4, 0.4, 0.6)

        y = cursor.y
        bx = cursor.x + pad_x
        bw = avail_w - pad_x * 2

        for i, label in enumerate(flow_labels):
            is_first = (i == 0)
            is_last = (i == len(flow_labels) - 1)
            is_empty_pipeline = (len(flow_labels) == 2)  # only raw + final, no steps

            # Box color
            if is_first:
                bc = col_raw
            elif is_last:
                bc = col_final
            else:
                bc = col_step

            # Draw box
            dl.add_rect_filled(bx, y, bx + bw, y + step_h, col_bg, box_rounding)
            dl.add_rect(bx, y, bx + bw, y + step_h, bc, box_rounding, thickness=1.5)

            # Label text centered
            ts = imgui.calc_text_size(label)
            tx = bx + (bw - ts.x) * 0.5
            ty = y + (step_h - ts.y) * 0.5
            dl.add_text(tx, ty, col_text if not is_empty_pipeline or is_first or is_last else col_disabled, label)

            y += step_h

            # Arrow between boxes (not after the last)
            if not is_last:
                mid_x = bx + bw * 0.5
                arrow_top = y + 2
                arrow_bot = y + arrow_h - 2
                dl.add_line(mid_x, arrow_top, mid_x, arrow_bot, col_arrow, 1.5)
                # Arrowhead
                dl.add_triangle_filled(
                    mid_x - 4, arrow_bot - 3,
                    mid_x + 4, arrow_bot - 3,
                    mid_x, arrow_bot + 1,
                    col_arrow
                )
                y += arrow_h

        total_h = y - cursor.y
        imgui.dummy(avail_w, total_h + 4)

        # Action buttons row: Edit | Preview | Apply
        has_steps = pipeline and any(s.enabled for s in pipeline.steps)
        btn_w = (avail_w - imgui.get_style().item_spacing[0] * 2) / 3

        if imgui.button("Edit##PostAnalysis", width=btn_w):
            app_state.show_plugin_pipeline = True
        if imgui.is_item_hovered():
            imgui.set_tooltip("Open the Plugin Pipeline builder")

        imgui.same_line()

        if not has_steps:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.4)
        if imgui.button("Preview##PostAnalysis", width=btn_w) and has_steps:
            pipeline_ui = gui.plugin_pipeline_ui if gui else None
            if pipeline_ui:
                pipeline_ui._preview_pipeline()
        if not has_steps:
            imgui.pop_style_var()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Preview the pipeline result on the timeline" if has_steps
                              else "Add steps to the pipeline first")

        imgui.same_line()

        if not has_steps:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.4)
        if imgui.button("Apply##PostAnalysis", width=btn_w) and has_steps:
            pipeline_ui = gui.plugin_pipeline_ui if gui else None
            if pipeline_ui:
                pipeline_ui._clear_preview()
                pipeline_ui._apply_pipeline()
        if not has_steps:
            imgui.pop_style_var()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Apply the pipeline to the funscript (Ctrl+Z to undo)" if has_steps
                              else "Add steps to the pipeline first")

