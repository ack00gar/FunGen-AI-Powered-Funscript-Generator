import imgui
import threading
from application.utils.component_perf_monitor import ComponentPerformanceMonitor
from application.utils.system_monitor import SystemMonitor
from application.utils.section_card import section_card as _section_card
from application.utils.timeline_constants import EXTRA_TIMELINE_RANGE
from application.gui_components.ig_panels import InfoGraphsMixin
from application.gui_components.settings_renderer import SettingsRenderer


class InfoGraphsUI(InfoGraphsMixin):

    def __init__(self, app):
        self.app = app
        # Debounced video rendering for View Pitch slider
        self.video_render_timer = None
        self.timer_lock = threading.Lock()
        self.last_pitch_value = None
        # Track slider interaction state
        self.pitch_slider_is_dragging = False
        self.pitch_slider_was_dragging = False

        # System performance monitor - start with less frequent updates,
        # will optimize based on tab visibility
        self.system_monitor = SystemMonitor(update_interval=3.0, history_size=60)
        self.system_monitor.start()

        # Performance monitoring for different components
        self.perf_monitor = ComponentPerformanceMonitor(
            "Performance Panel", history_size=60
        )
        self.video_info_perf = ComponentPerformanceMonitor(
            "Video Info", history_size=30
        )
        self.video_settings_perf = ComponentPerformanceMonitor(
            "Video Settings", history_size=30
        )
        self.funscript_info_perf = ComponentPerformanceMonitor(
            "Funscript Info", history_size=30
        )
        self.ui_performance_perf = ComponentPerformanceMonitor(
            "UI Performance", history_size=30
        )

        # Track tab visibility for optimization
        self._last_performance_tab_active = False

        # Disk I/O zero value delay tracking
        self._last_non_zero_read_rate = 0.0
        self._last_non_zero_write_rate = 0.0
        self._read_zero_start_time = None
        self._write_zero_start_time = None
        self._zero_delay_duration = 3.0  # 3 seconds

        # Quality scoring state (auto-computed in Info tab)
        self._quality_reports = {}          # {timeline_num: QualityReport}
        self._quality_last_action_counts = {}  # {timeline_num: int} for change detection
        self._quality_last_funscript_id = None  # detect funscript object swap
        self._quality_last_compute_time = {}   # {timeline_num: float} monotonic time

        # Cached action statistics (avoid O(N) recomputation every frame)
        self._stats_cache = {}               # {timeline_num: stats dict}
        self._stats_last_action_counts = {}  # {timeline_num: int} for change detection
        self._stats_last_funscript_id = None # detect funscript object swap

        # Per-frame cache for system_monitor.get_stats() (avoid multiple calls per frame)
        self._cached_system_stats = None
        self._cached_system_stats_frame = -1
        self._render_frame_counter = 0

        # Shared settings renderer (owns Settings tab)
        self.settings_renderer = SettingsRenderer(app)

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

    def _get_system_stats(self):
        """Get system monitor stats, cached per frame to avoid redundant calls."""
        if self._cached_system_stats is None or self._cached_system_stats_frame != self._render_frame_counter:
            self._cached_system_stats = self.system_monitor.get_stats()
            self._cached_system_stats_frame = self._render_frame_counter
        return self._cached_system_stats

    def cleanup(self):
        """Clean up any pending timers."""
        self._cancel_video_render_timer()
        self.system_monitor.stop()
        self.settings_renderer.cleanup()

    def render(self):
        self._render_frame_counter += 1
        app_state = self.app.app_state_ui
        window_title = "Info & Graphs##InfoGraphsFloating"

        # Determine flags based on layout mode
        if app_state.ui_layout_mode == "floating":
            if not getattr(app_state, "show_info_graphs_window", True):
                return
            is_open, new_visibility = imgui.begin(window_title, closable=True)
            if new_visibility != app_state.show_info_graphs_window:
                app_state.show_info_graphs_window = new_visibility
            if not is_open:
                imgui.end()
                return
        else:  # Fixed mode
            imgui.begin(
                "Graphs##RightGraphsContainer",
                flags=imgui.WINDOW_NO_TITLE_BAR
                | imgui.WINDOW_NO_MOVE
                | imgui.WINDOW_NO_COLLAPSE,
            )
            self._render_panel_label("EXPERT")

        self._render_tabbed_content()

        imgui.end()

    def _render_tabbed_content(self):
        tab_selected = None
        if imgui.begin_tab_bar("InfoGraphsTabs"):
            if imgui.begin_tab_item("Info")[0]:
                tab_selected = "info"
                imgui.end_tab_item()
            if imgui.begin_tab_item("Video")[0]:
                tab_selected = "video"
                imgui.end_tab_item()
            if imgui.begin_tab_item("Settings")[0]:
                tab_selected = "settings"
                imgui.end_tab_item()
            if imgui.begin_tab_item("Tracker")[0]:
                tab_selected = "tracker"
                imgui.end_tab_item()
            if imgui.begin_tab_item("Undo")[0]:
                tab_selected = "undo"
                imgui.end_tab_item()
            if imgui.begin_tab_item("Performance")[0]:
                tab_selected = "performance"
                imgui.end_tab_item()
            imgui.end_tab_bar()

        # Update performance tab visibility tracking
        performance_tab_now_active = tab_selected == "performance"
        if performance_tab_now_active != self._last_performance_tab_active:
            self._last_performance_tab_active = performance_tab_now_active
            if hasattr(self, "system_monitor"):
                if performance_tab_now_active:
                    self.system_monitor.set_update_interval(1.0)
                else:
                    self.system_monitor.set_update_interval(3.0)

        avail = imgui.get_content_region_available()
        imgui.begin_child(
            "InfoGraphsTabContent", width=0, height=avail[1], border=False
        )
        if tab_selected == "info":
            imgui.spacing()

            # Funscript (expanded by default)
            with _section_card("Funscript##FunscriptParentSection", tier="primary") as fs_open:
                if fs_open:
                    self._render_funscript_info_section(1)
                    if self.app.app_state_ui.show_funscript_interactive_timeline2:
                        self._render_funscript_info_section(2)
                    for tl_num in EXTRA_TIMELINE_RANGE:
                        vis_attr = f"show_funscript_interactive_timeline{tl_num}"
                        if getattr(self.app.app_state_ui, vis_attr, False):
                            self._render_funscript_info_section(tl_num)

            # Funscript Comparison (only visible when a reference is loaded)
            if self._has_any_reference_loaded():
                with _section_card("Funscript Comparison##RefCompSection", tier="primary") as ref_open:
                    if ref_open:
                        self._render_reference_comparison_standalone()

            # Segment Statistics
            with _section_card("Segment Statistics##SegStatSection", tier="primary",
                               ) as seg_open:
                if seg_open:
                    self._render_segment_statistics()

        elif tab_selected == "video":
            imgui.spacing()

            # Header (same style as Tracker tab)
            processor = self.app.processor
            if processor and processor.video_info:
                video_type = processor.determined_video_type or "2D"
                imgui.text_colored(f"{video_type} Video", 0.4, 0.8, 1.0, 1.0)
                if video_type == "VR":
                    imgui.text_wrapped("Recommended unwarp: CPU (v360). Recommended pitch: -21 for typical POV.")
                imgui.spacing()
            else:
                imgui.text_disabled("No video loaded.")
                imgui.spacing()

            with _section_card("Video Settings##VideoSettingsSection", tier="primary") as vs_open:
                if vs_open:
                    self._render_content_video_settings()

            with _section_card("Video Information##VideoInfoSection", tier="primary",
                               open_by_default=False) as vi_open:
                if vi_open:
                    self._render_content_video_info()

        elif tab_selected == "settings":
            imgui.spacing()
            self.settings_renderer.render()

        elif tab_selected == "tracker":
            imgui.spacing()
            self.settings_renderer.render_tracker_tab()

        elif tab_selected == "undo":
            imgui.spacing()
            self._render_content_undo_redo_history()

        elif tab_selected == "performance":
            imgui.spacing()

            with _section_card("System Monitor##SystemMonitorSection",
                               tier="primary") as sys_open:
                if sys_open:
                    self._render_content_performance()

            with _section_card("System Report##SystemReportSection", tier="primary",
                               open_by_default=False) as report_open:
                if report_open:
                    self._render_system_report_section()

            # Developer details (behind Show Advanced Options)
            if self.app.app_state_ui.show_advanced_options:
                if performance_tab_now_active:
                    self.perf_monitor.render_info(show_detailed=True)
                    imgui.separator()

                with _section_card("Video Pipeline##PipelineTimingSection",
                                   tier="primary", open_by_default=False) as pipe_open:
                    if pipe_open:
                        self._render_content_pipeline_timing()

                with _section_card("Disk I/O##DiskIOSection", tier="primary",
                                   open_by_default=False) as disk_open:
                    if disk_open:
                        self._render_disk_io_section()

                with _section_card("UI Performance##UIPerformanceSection", tier="primary",
                                   open_by_default=False) as ui_open:
                    if ui_open:
                        self._render_content_ui_performance()

        # Always check memory alerts
        if hasattr(self, "system_monitor"):
            self._check_memory_alerts(self._get_system_stats())

        imgui.end_child()
