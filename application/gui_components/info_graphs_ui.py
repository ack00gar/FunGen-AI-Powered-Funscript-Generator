import imgui
import os
import threading
import numpy as np
import time
from collections import deque
from application.utils import _format_time
from application.utils.system_monitor import SystemMonitor


class ComponentPerformanceMonitor:
    """Efficient performance monitoring for UI components with memory management."""

    def __init__(self, component_name: str, history_size: int = 60):
        self.component_name = component_name
        self.history = deque(maxlen=history_size)  # Efficient O(1) append/pop
        self.start_time = None

    def start_timing(self):
        """Start timing a render cycle."""
        self.start_time = time.perf_counter()

    def end_timing(self):
        """End timing and record the measurement."""
        if self.start_time is not None:
            render_time_ms = (time.perf_counter() - self.start_time) * 1000
            self.history.append(render_time_ms)
            self.start_time = None
            return render_time_ms
        return 0.0

    def get_stats(self):
        """Get performance statistics."""
        if not self.history:
            return {"current": 0.0, "avg": 0.0, "max": 0.0, "min": 0.0, "count": 0}

        history_list = list(self.history)
        return {
            "current": history_list[-1],
            "avg": sum(history_list) / len(history_list),
            "max": max(history_list),
            "min": min(history_list),
            "count": len(history_list),
        }

    def get_status_info(self):
        """Get color-coded status information."""
        stats = self.get_stats()
        current = stats["current"]

        if current < 1.0:
            return "[EXCELLENT]", (0.0, 1.0, 0.0, 1.0)  # Bright green
        elif current < 5.0:
            return "[VERY GOOD]", (0.2, 0.8, 0.2, 1.0)  # Green
        elif current < 16.67:
            return "[GOOD]", (0.4, 0.8, 0.4, 1.0)  # Light green
        elif current < 33.33:
            return "[OK]", (1.0, 0.8, 0.2, 1.0)  # Yellow
        elif current < 50.0:
            return "[SLOW]", (1.0, 0.5, 0.0, 1.0)  # Orange
        else:
            return "[VERY SLOW]", (1.0, 0.2, 0.2, 1.0)  # Red

    def render_info(self, show_detailed=True):
        """Render performance information in imgui."""
        stats = self.get_stats()
        status_text, status_color = self.get_status_info()

        if show_detailed:
            imgui.text_colored(
                f"{self.component_name} Performance {status_text}", *status_color
            )
            imgui.text(
                f"Current: {stats['current']:.2f}ms | "
                f"Avg: {stats['avg']:.2f}ms | "
                f"Max: {stats['max']:.2f}ms | "
                f"Min: {stats['min']:.2f}ms"
            )

            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    f"{self.component_name} Render Performance:\n"
                    f"Current frame: {stats['current']:.2f}ms\n"
                    f"Average ({stats['count']} frames): {stats['avg']:.2f}ms\n"
                    f"Maximum: {stats['max']:.2f}ms\n"
                    f"Minimum: {stats['min']:.2f}ms\n\n"
                    "Performance Targets:\n"
                    "< 1ms: Excellent\n"
                    "< 5ms: Very Good\n"
                    "< 16.67ms: Good (60 FPS)\n"
                    "< 33.33ms: OK (30 FPS)\n"
                    "> 33.33ms: Needs optimization"
                )
        else:
            imgui.text_colored(
                f"{self.component_name}: {stats['current']:.1f}ms {status_text}",
                *status_color,
            )


class InfoGraphsUI:
    # Constants
    PITCH_SLIDER_DELAY_MS = 1000
    PITCH_SLIDER_MIN = -40
    PITCH_SLIDER_MAX = 40

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

    def _apply_video_render(self, new_pitch):
        """Execute video rendering with the new pitch value after delay"""
        processor = self.app.processor
        if processor:
            processor.set_active_vr_parameters(pitch=new_pitch)
            processor.reapply_video_settings()

    def _schedule_video_render(self, new_pitch):
        """Schedule video rendering with delay, canceling any existing timer"""
        with self.timer_lock:
            if self.video_render_timer:
                self.video_render_timer.cancel()

            self.video_render_timer = threading.Timer(
                self.PITCH_SLIDER_DELAY_MS / 1000.0,
                self._apply_video_render,
                args=[new_pitch],
            )
            self.video_render_timer.daemon = True
            self.video_render_timer.start()

    def _cancel_video_render_timer(self):
        """Cancel any pending video render timer"""
        with self.timer_lock:
            if self.video_render_timer:
                self.video_render_timer.cancel()
                self.video_render_timer = None

    def _handle_mouse_release(self):
        """Handle mouse release - cancel timer and render immediately"""
        self.pitch_slider_is_dragging = False
        self.pitch_slider_was_dragging = False
        self._cancel_video_render_timer()

        # Execute video rendering immediately with final value
        if self.last_pitch_value is not None:
            self._apply_video_render(self.last_pitch_value)

    def cleanup(self):
        """Clean up any pending timers"""
        self._cancel_video_render_timer()
        self.system_monitor.stop()

    def render(self):
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

        if app_state.ui_view_mode == "simple":
            self._render_simple_view_content()
        else:
            self._render_tabbed_content()

        imgui.end()

    def _render_simple_view_content(self):
        """Renders only the video information for Simple Mode."""
        imgui.begin_child("SimpleInfoChild", border=False)
        imgui.spacing()
        self._render_content_video_info()
        imgui.end_child()

    def _render_tabbed_content(self):
        tab_selected = None
        if imgui.begin_tab_bar("InfoGraphsTabs"):
            if imgui.begin_tab_item("Video")[0]:
                tab_selected = "video"
                imgui.end_tab_item()
            if imgui.begin_tab_item("Funscript")[0]:
                tab_selected = "funscript"
                imgui.end_tab_item()
            if imgui.begin_tab_item("History")[0]:
                tab_selected = "history"
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
        if tab_selected == "video":
            imgui.spacing()
            if imgui.collapsing_header(
                "Video Information##VideoInfoSection",
                flags=imgui.TREE_NODE_DEFAULT_OPEN,
            )[0]:
                self._render_content_video_info()
            imgui.separator()
            if imgui.collapsing_header(
                "Video Settings##VideoSettingsSection",
                flags=imgui.TREE_NODE_DEFAULT_OPEN,
            )[0]:
                self._render_content_video_settings()
        elif tab_selected == "funscript":
            imgui.spacing()
            if imgui.collapsing_header(
                "Funscript Info (Timeline 1)##FSInfoT1Section",
                flags=imgui.TREE_NODE_DEFAULT_OPEN,
            )[0]:
                self._render_content_funscript_info(1)
            imgui.separator()
            if self.app.app_state_ui.show_funscript_interactive_timeline2:
                if imgui.collapsing_header(
                    "Funscript Info (Timeline 2)##FSInfoT2Section",
                    flags=imgui.TREE_NODE_DEFAULT_OPEN,
                )[0]:
                    self._render_content_funscript_info(2)
            else:
                imgui.text_disabled(
                    "Enable Interactive Timeline 2 to see its stats."
                )
        elif tab_selected == "history":
            imgui.spacing()
            if imgui.collapsing_header(
                "Undo-Redo History##UndoRedoSection",
                flags=imgui.TREE_NODE_DEFAULT_OPEN,
            )[0]:
                self._render_content_undo_redo_history()
        elif tab_selected == "performance":
            imgui.spacing()
            # OPTIMIZATION: Only render performance data if tab is actually visible and focused
            if performance_tab_now_active:
                # Performance Panel Performance at the top
                self.perf_monitor.render_info(show_detailed=True)
            imgui.separator()
            if imgui.collapsing_header(
                "System Monitor##PerformanceSection",
                flags=imgui.TREE_NODE_DEFAULT_OPEN,
            )[0]:
                self._render_content_performance()
            imgui.separator()
            # --- Disk I/O Section (independent, collapsible) ---
            if imgui.collapsing_header(
                "Disk I/O##DiskIOSection", flags=imgui.TREE_NODE_DEFAULT_OPEN
            )[0]:
                self._render_disk_io_section()
            imgui.separator()
            if imgui.collapsing_header(
                "UI Performance##UIPerformanceSection",
                flags=imgui.TREE_NODE_DEFAULT_OPEN,
            )[0]:
                self._render_content_ui_performance()

        # Always check memory alerts
        if hasattr(self, "system_monitor"):
            stats = self.system_monitor.get_stats()
            self._check_memory_alerts(stats)

        imgui.end_child()

    def _get_k_resolution_label(self, width, height):
        if width <= 0 or height <= 0:
            return ""
        if (1280, 720) == (width, height):
            return " (HD)"
        if (1920, 1080) == (width, height):
            return " (Full HD)"
        if (2560, 1440) == (width, height):
            return " (QHD/2.5K)"
        if (3840, 2160) == (width, height):
            return " (4K UHD)"
        if width >= 7600:
            return " (8K)"
        if width >= 6600:
            return " (7K)"
        if width >= 5600:
            return " (6K)"
        if width >= 5000:
            return " (5K)"
        if width >= 3800:
            return " (4K)"
        return ""

    def _render_content_video_info(self):
        self.video_info_perf.start_timing()
        file_mgr = self.app.file_manager

        imgui.columns(2, "video_info_stats", border=False)
        imgui.set_column_width(0, 120 * imgui.get_io().font_global_scale)

        if self.app.processor and self.app.processor.video_info:
            path = (
                os.path.dirname(file_mgr.video_path)
                if file_mgr.video_path
                else "N/A (Drag & Drop Video)"
            )
            filename = self.app.processor.video_info.get("filename", "N/A")

            info = self.app.processor.video_info
            width, height = info.get("width", 0), info.get("height", 0)

            imgui.text("Path:")
            imgui.next_column()
            imgui.text_wrapped(path)
            imgui.next_column()

            imgui.text("File:")
            imgui.next_column()
            imgui.text_wrapped(filename)
            imgui.next_column()

            imgui.text("Resolution:")
            imgui.next_column()
            imgui.text(
                f"{width}x{height}{self._get_k_resolution_label(width, height)}"
            )
            imgui.next_column()

            imgui.text("Duration:")
            imgui.next_column()
            imgui.text(f"{_format_time(self.app, info.get('duration', 0.0))}")
            imgui.next_column()

            imgui.text("Total Frames:")
            imgui.next_column()
            imgui.text(f"{info.get('total_frames', 0)}")
            imgui.next_column()

            imgui.text("Frame Rate:")
            imgui.next_column()
            fps_text = f"{info.get('fps', 0):.3f}"
            fps_mode = " (VFR)" if info.get("is_vfr", False) else " (CFR)"
            imgui.text(fps_text + fps_mode)
            imgui.next_column()

            imgui.text("Size:")
            imgui.next_column()
            size_bytes = info.get("file_size", 0)
            if size_bytes > 0:
                if size_bytes > 1024 * 1024 * 1024:
                    size_str = f"{size_bytes / (1024**3):.2f} GB"
                else:
                    size_str = f"{size_bytes / (1024**2):.2f} MB"
            else:
                size_str = "N/A"
            imgui.text(size_str)
            imgui.next_column()

            imgui.text("Bitrate:")
            imgui.next_column()
            bitrate_bps = info.get("bitrate", 0)
            if bitrate_bps > 0:
                bitrate_mbps = bitrate_bps / 1_000_000
                bitrate_str = f"{bitrate_mbps:.2f} Mbit/s"
            else:
                bitrate_str = "N/A"
            imgui.text(bitrate_str)
            imgui.next_column()

            imgui.text("Bit Depth:")
            imgui.next_column()
            imgui.text(f"{info.get('bit_depth', 'N/A')} bit")
            imgui.next_column()
            imgui.text("Detected Type:")
            imgui.next_column()
            imgui.text(f"{self.app.processor.determined_video_type or 'N/A'}")
            imgui.next_column()

            imgui.text("Active Source:")
            imgui.next_column()
            processor = self.app.processor
            if (
                hasattr(processor, "_active_video_source_path")
                and processor._active_video_source_path != processor.video_path
            ):
                imgui.text("Preprocessed File")
                if imgui.is_item_hovered():
                    imgui.set_tooltip(
                        f"Using: {os.path.basename(processor._active_video_source_path)}\n"
                        "All filtering/de-warping is pre-applied."
                    )
            else:
                imgui.text("Original File")
                if imgui.is_item_hovered():
                    imgui.set_tooltip(
                        f"Using: {os.path.basename(processor.video_path)}\n"
                        "Filters are applied on-the-fly."
                    )
            imgui.next_column()
        else:
            imgui.text("Status:")
            imgui.next_column()
            imgui.text("Video details not loaded.")
            imgui.next_column()
        imgui.columns(1)
        imgui.spacing()

        self.video_info_perf.end_timing()

    def _render_content_video_settings(self):
        self.video_settings_perf.start_timing()
        processor = self.app.processor
        if not processor:
            imgui.text("VideoProcessor not initialized.")
            self.video_settings_perf.end_timing()
            return

        imgui.text("Hardware Acceleration")
        hw_accel_options = self.app.available_ffmpeg_hwaccels
        hw_accel_display = [
            name.replace("_", " ").title()
            if name not in ["auto", "none"]
            else ("Auto Detect" if name == "auto" else "None (CPU Only)")
            for name in hw_accel_options
        ]

        try:
            current_hw_idx = hw_accel_options.index(
                self.app.hardware_acceleration_method
            )
        except ValueError:
            current_hw_idx = 0

        changed, new_idx = imgui.combo(
            "Method##HWAccel", current_hw_idx, hw_accel_display
        )
        if changed:
            self.app.hardware_acceleration_method = hw_accel_options[new_idx]
            self.app.app_settings.set(
                "hardware_acceleration_method",
                self.app.hardware_acceleration_method,
            )

            if processor.is_video_open():
                processor.reapply_video_settings()

        imgui.separator()
        video_types = ["auto", "2D", "VR"]
        current_type_idx = (
            video_types.index(processor.video_type_setting)
            if processor.video_type_setting in video_types
            else 0
        )
        changed, new_idx = imgui.combo(
            "Video Type##vidType", current_type_idx, video_types
        )
        if changed:
            processor.set_active_video_type_setting(video_types[new_idx])
            processor.reapply_video_settings()

        if processor.is_vr_active_or_potential():
            imgui.separator()
            imgui.text("VR Settings")
            vr_fmt_disp = [
                "Equirectangular (SBS)",
                "Fisheye (SBS)",
                "Equirectangular (TB)",
                "Fisheye (TB)",
                "Equirectangular (Mono)",
                "Fisheye (Mono)",
            ]
            vr_fmt_val = ["he_sbs", "fisheye_sbs", "he_tb", "fisheye_tb", "he", "fisheye"]
            current_vr_idx = (
                vr_fmt_val.index(processor.vr_input_format)
                if processor.vr_input_format in vr_fmt_val
                else 0
            )
            changed, new_idx = imgui.combo(
                "Input Format##vrFmt", current_vr_idx, vr_fmt_disp
            )
            if changed:
                processor.set_active_vr_parameters(input_format=vr_fmt_val[new_idx])
                processor.reapply_video_settings()

            # Track slider dragging state
            is_slider_hovered = imgui.is_item_hovered()
            is_mouse_down = imgui.is_mouse_down(0)  # Left mouse button
            is_mouse_released = imgui.is_mouse_released(0)  # Left mouse button released

            if is_slider_hovered and is_mouse_down:
                self.pitch_slider_is_dragging = True
                self.pitch_slider_was_dragging = True
            elif (is_mouse_released or not is_mouse_down) and self.pitch_slider_was_dragging:
                self._handle_mouse_release()

            changed_pitch, new_pitch = imgui.slider_int(
                "View Pitch##vrPitch",
                processor.vr_pitch,
                self.PITCH_SLIDER_MIN,
                self.PITCH_SLIDER_MAX,
            )
            if changed_pitch:
                processor.vr_pitch = new_pitch
                self.last_pitch_value = new_pitch
                self._schedule_video_render(new_pitch)

        self.video_settings_perf.end_timing()

    def _render_content_funscript_info(self, timeline_num):
        self.funscript_info_perf.start_timing()
        fs_proc = self.app.funscript_processor
        stats = (
            fs_proc.funscript_stats_t1
            if timeline_num == 1
            else fs_proc.funscript_stats_t2
        )
        source_text = stats.get("source_type", "N/A")

        if source_text == "File" and stats.get("path", "N/A") != "N/A":
            source_text = f"File: {os.path.basename(stats['path'])}"
        elif stats.get("path", "N/A") != "N/A":
            source_text = stats["path"]

        imgui.text_wrapped(f"Source: {source_text}")
        imgui.separator()

        imgui.columns(2, f"fs_stats_{timeline_num}", border=False)
        imgui.set_column_width(0, 180 * imgui.get_io().font_global_scale)

        def stat_row(label, value):
            imgui.text(label)
            imgui.next_column()
            imgui.text(str(value))
            imgui.next_column()

        stat_row("Points:", stats.get("num_points", 0))
        stat_row("Duration (s):", f"{stats.get('duration_scripted_s', 0.0):.2f}")
        stat_row("Total Travel:", stats.get("total_travel_dist", 0))
        stat_row("Strokes:", stats.get("num_strokes", 0))
        imgui.separator()
        imgui.next_column()
        imgui.separator()
        imgui.next_column()
        stat_row("Avg Speed (pos/s):", f"{stats.get('avg_speed_pos_per_s', 0.0):.2f}")
        stat_row("Avg Intensity (%):", f"{stats.get('avg_intensity_percent', 0.0):.1f}")
        imgui.separator()
        imgui.next_column()
        imgui.separator()
        imgui.next_column()
        stat_row(
            "Position Range:",
            f"{stats.get('min_pos', 'N/A')} - {stats.get('max_pos', 'N/A')}",
        )
        imgui.separator()
        imgui.next_column()
        imgui.separator()
        imgui.next_column()
        stat_row(
            "Min/Max Interval (ms):",
            f"{stats.get('min_interval_ms', 'N/A')} / {stats.get('max_interval_ms', 'N/A')}",
        )
        stat_row("Avg Interval (ms):", f"{stats.get('avg_interval_ms', 0.0):.2f}")

        imgui.columns(1)
        imgui.spacing()

        self.funscript_info_perf.end_timing()

    def _render_content_undo_redo_history(self):
        fs_proc = self.app.funscript_processor
        imgui.begin_child("UndoRedoChild", height=150, border=True)

        def render_history_for_timeline(num):
            manager = fs_proc._get_undo_manager(num)
            if not manager:
                return

            imgui.text(f"T{num} Undo History:")
            imgui.next_column()
            imgui.text(f"T{num} Redo History:")
            imgui.next_column()

            undo_history = manager.get_undo_history_for_display()
            redo_history = manager.get_redo_history_for_display()

            if undo_history:
                for i, desc in enumerate(undo_history):
                    imgui.text(f"  {i}: {desc}")
            else:
                imgui.text_disabled("  (empty)")

            imgui.next_column()

            if redo_history:
                for i, desc in enumerate(redo_history):
                    imgui.text(f"  {i}: {desc}")
            else:
                imgui.text_disabled("  (empty)")
            imgui.next_column()

        imgui.columns(2, "UndoRedoColumnsT1")
        render_history_for_timeline(1)
        imgui.columns(1)

        if self.app.app_state_ui.show_funscript_interactive_timeline2:
            imgui.separator()
            imgui.columns(2, "UndoRedoColumnsT2")
            render_history_for_timeline(2)
            imgui.columns(1)

        imgui.end_child()

    def _render_content_performance(self):
        self.perf_monitor.start_timing()

        stats = self.system_monitor.get_stats()

        self._check_memory_alerts(stats)

        available_width = imgui.get_content_region_available_width()

        def render_graph(
            label,
            data,
            overlay_text,
            scale_min=0,
            scale_max=100,
            height=60,
            color=None,
        ):
            np_data = np.array(data, dtype=np.float32) if data else np.array([], dtype=np.float32)
            current_value = data[-1] if data else 0.0
            if color is None:
                if current_value < 50:
                    color = (0.2, 0.8, 0.2, 0.8)
                elif current_value < 80:
                    color = (1.0, 0.8, 0.2, 0.8)
                else:
                    color = (1.0, 0.2, 0.2, 0.8)

            imgui.push_style_color(imgui.COLOR_PLOT_LINES, *color)
            imgui.plot_lines(
                f"##{label}",
                np_data,
                overlay_text=overlay_text,
                scale_min=scale_min,
                scale_max=scale_max,
                graph_size=(available_width, height),
            )
            imgui.pop_style_color()

        def render_core_bars(per_core_usage):
            if not per_core_usage:
                return

            bar_width = max(8, (available_width - 20) / len(per_core_usage))
            bar_height = 30
            spacing = 2

            total_width = (bar_width + spacing) * len(per_core_usage) - spacing
            start_x = imgui.get_cursor_screen_pos()[0]

            for i, core_load in enumerate(per_core_usage):
                bar_x = start_x + i * (bar_width + spacing)
                bar_y = imgui.get_cursor_screen_pos()[1]

                if core_load < 50:
                    color = imgui.get_color_u32_rgba(0.2, 0.8, 0.2, 1.0)
                elif core_load < 80:
                    color = imgui.get_color_u32_rgba(1.0, 0.8, 0.2, 1.0)
                else:
                    color = imgui.get_color_u32_rgba(1.0, 0.2, 0.2, 1.0)

                bg_color = imgui.get_color_u32_rgba(0.2, 0.2, 0.2, 0.8)
                imgui.get_window_draw_list().add_rect_filled(
                    bar_x, bar_y, bar_x + bar_width, bar_y + bar_height, bg_color
                )

                usage_height = (core_load / 100.0) * bar_height
                imgui.get_window_draw_list().add_rect_filled(
                    bar_x,
                    bar_y + bar_height - usage_height,
                    bar_x + bar_width,
                    bar_y + bar_height,
                    color,
                )

                text = f"C{i}"
                text_size = imgui.calc_text_size(text)
                text_x = bar_x + (bar_width - text_size[0]) / 2
                text_y = bar_y + bar_height + 2
                imgui.get_window_draw_list().add_text(
                    text_x,
                    text_y,
                    imgui.get_color_u32_rgba(0.8, 0.8, 0.8, 1.0),
                    text,
                )

            imgui.dummy(total_width, bar_height + 20)

        # CPU
        physical_cores = stats.get("cpu_physical_cores", stats["cpu_core_count"])
        logical_cores = stats["cpu_core_count"]
        core_info = (
            f"({physical_cores}P/{logical_cores}L cores)"
            if physical_cores != logical_cores
            else f"({logical_cores} cores)"
        )

        cpu_load = stats["cpu_load"]
        current_cpu = cpu_load[-1] if cpu_load else 0

        if current_cpu < 50:
            header_color = (0.2, 0.8, 0.2, 1.0)
            cpu_status = "[OK]"
        elif current_cpu < 80:
            header_color = (1.0, 0.8, 0.2, 1.0)
            cpu_status = "[HIGH]"
        else:
            header_color = (1.0, 0.2, 0.2, 1.0)
            cpu_status = "[CRIT]"

        imgui.text_colored(f"CPU {core_info} {cpu_status}", *header_color)
        render_graph("cpu_load", cpu_load, f"{current_cpu:.1f}%", height=50)

        # CPU freq (MHz to GHz) if available
        cpu_freq = stats.get("cpu_freq", 0)
        if cpu_freq > 0:
            freq_ghz = cpu_freq / 1000.0
            imgui.same_line()
            imgui.text_colored(f" @ {freq_ghz:.1f}GHz", 0.7, 0.7, 0.7, 1.0)

        # CPU temp if available
        cpu_temp = stats.get("cpu_temp", None)
        if cpu_temp is not None:
            imgui.same_line()
            imgui.text_colored(f" | {cpu_temp:.0f}°C", 0.7, 0.7, 0.7, 1.0)

        imgui.spacing()

        per_core_usage = stats.get("cpu_per_core", [])
        if per_core_usage:
            imgui.text("Per-Core Usage:")
            render_core_bars(per_core_usage)
        else:
            imgui.text_disabled("Per-core data not available")

        imgui.separator()
        imgui.spacing()

        # RAM
        ram_percent = stats["ram_usage_percent"]
        ram_gb = stats["ram_usage_gb"]
        ram_total = stats.get("ram_total_gb", 0)
        current_ram = ram_percent[-1] if ram_percent else 0

        if current_ram < 60:
            ram_color = (0.2, 0.8, 0.2, 1.0)
        elif current_ram < 85:
            ram_color = (1.0, 0.8, 0.2, 1.0)
        else:
            ram_color = (1.0, 0.2, 0.2, 1.0)

        ram_status = (
            "[OK]" if current_ram < 60 else "[HIGH]" if current_ram < 85 else "[CRIT]"
        )
        imgui.text_colored(f"Memory (RAM) {ram_status}", *ram_color)
        last_ram_gb = ram_gb[-1] if ram_gb else 0.0
        render_graph(
            "ram_usage",
            ram_percent,
            f"{current_ram:.1f}% ({last_ram_gb:.1f}/{ram_total:.1f}GB)",
            height=55,
        )

        swap_percent = stats.get("swap_usage_percent", 0.0)
        swap_gb = stats.get("swap_usage_gb", 0.0)
        if swap_percent > 0:
            swap_color = (
                (0.2, 0.8, 0.2, 1.0)
                if swap_percent < 50
                else (1.0, 0.8, 0.2, 1.0)
                if swap_percent < 80
                else (1.0, 0.2, 0.2, 1.0)
            )
            imgui.text_colored(
                f"Swap: {swap_percent:.1f}% ({swap_gb:.1f}GB)", *swap_color
            )
        else:
            imgui.text_colored("Swap: Not in use", 0.2, 0.8, 0.2, 1.0)

        imgui.separator()

        # GPU
        if stats.get("gpu_available", False):
            gpu_name = stats.get("gpu_name", "Unknown GPU")
            if len(gpu_name) > 35:
                gpu_name = gpu_name[:32] + "..."

            gpu_load = stats.get("gpu_load", [])
            current_gpu = gpu_load[-1] if gpu_load else 0.0

            if current_gpu < 50:
                gpu_color = (0.2, 0.8, 0.2, 1.0)
            elif current_gpu < 80:
                gpu_color = (1.0, 0.8, 0.2, 1.0)
            else:
                gpu_color = (1.0, 0.2, 0.2, 1.0)

            gpu_status = (
                "[OK]" if current_gpu < 50 else "[HIGH]" if current_gpu < 80 else "[CRIT]"
            )
            header = f"GPU - {gpu_name} {gpu_status}"

            # Append temp if available
            gpu_temp = stats.get("gpu_temp", None)
            if gpu_temp is not None:
                header += f" | {gpu_temp:.0f}°C"

            imgui.text_colored(header, *gpu_color)

            if any(load > 0 for load in gpu_load):
                render_graph("gpu_load", gpu_load, f"{current_gpu:.1f}%", height=55)

                gpu_mem = stats.get("gpu_mem_usage_percent", [])
                current_gpu_mem = gpu_mem[-1] if gpu_mem else 0.0

                # Best effort: get memory used/total from gpu_info if present
                mem_overlay = f"{current_gpu_mem:.1f}%"
                gpu_info = stats.get("gpu_info", None)
                try:
                    if (
                        isinstance(gpu_info, dict)
                        and "gpu" in gpu_info
                        and gpu_info["gpu"]
                    ):
                        fb = gpu_info["gpu"][0].get("fb_memory_usage", {})
                        used = fb.get("used", 0.0)
                        total = fb.get("total", 0.0)

                        # pynvml.smi may provide MiB values or strings; try to coerce
                        def _to_mib(x):
                            if isinstance(x, (int, float)):
                                return float(x)
                            s = str(x)
                            for ch in ["MiB", "MB", "GiB", "GB"]:
                                s = s.replace(ch, "")
                            try:
                                return float(s.strip())
                            except Exception:
                                return None

                        used_mib = _to_mib(used)
                        total_mib = _to_mib(total)
                        if used_mib is not None and total_mib is not None and total_mib > 0:
                            used_gb = used_mib / 1024.0
                            total_gb = total_mib / 1024.0
                            mem_overlay = f"{current_gpu_mem:.1f}% ({used_gb:.1f}/{total_gb:.1f}GB)"
                except Exception:
                    pass

                render_graph("gpu_mem", gpu_mem, mem_overlay, height=45)
            else:
                if stats.get("os") == "Darwin" and "Apple" in gpu_name:
                    imgui.text_colored(
                        "[INFO] GPU detected but metrics may require admin access",
                        0.6,
                        0.8,
                        1.0,
                        1.0,
                    )
                    imgui.text_colored(
                        "      Run with 'sudo' to enable GPU monitoring",
                        0.5,
                        0.6,
                        0.7,
                        1.0,
                    )
                else:
                    imgui.text_colored(
                        "[INFO] GPU monitoring not available", 0.7, 0.7, 0.7, 1.0
                    )
                imgui.spacing()

            imgui.spacing()

        self.perf_monitor.end_timing()

    def _render_disk_io_section(self):
        """Render the independent Disk I/O section."""
        stats = self.system_monitor.get_stats()
        
        available_width = imgui.get_content_region_available_width()

        def render_graph(
            label,
            data,
            overlay_text,
            scale_min=0,
            scale_max=100,
            height=60,
            color=None,
        ):
            np_data = np.array(data, dtype=np.float32) if data else np.array([], dtype=np.float32)
            current_value = data[-1] if data else 0.0
            if color is None:
                if current_value < 50:
                    color = (0.2, 0.8, 0.2, 0.8)
                elif current_value < 80:
                    color = (1.0, 0.8, 0.2, 0.8)
                else:
                    color = (1.0, 0.2, 0.2, 0.8)

            imgui.push_style_color(imgui.COLOR_PLOT_LINES, *color)
            imgui.plot_lines(
                f"##{label}",
                np_data,
                overlay_text=overlay_text,
                scale_min=scale_min,
                scale_max=scale_max,
                graph_size=(available_width, height),
            )
            imgui.pop_style_color()

        disk_read_mb_s = stats.get("disk_read_mb_s", [])
        disk_write_mb_s = stats.get("disk_write_mb_s", [])
        current_read_rate = disk_read_mb_s[-1] if disk_read_mb_s else 0.0
        current_write_rate = disk_write_mb_s[-1] if disk_write_mb_s else 0.0
        current_time = time.time()

        # Handle 3-second delay for zero values
        # Read rate logic
        if current_read_rate > 0:
            self._last_non_zero_read_rate = current_read_rate
            self._read_zero_start_time = None
            read_rate = current_read_rate
        else:
            if self._read_zero_start_time is None:
                self._read_zero_start_time = current_time
                read_rate = self._last_non_zero_read_rate
            elif current_time - self._read_zero_start_time >= self._zero_delay_duration:
                read_rate = 0.0
            else:
                read_rate = self._last_non_zero_read_rate

        # Write rate logic
        if current_write_rate > 0:
            self._last_non_zero_write_rate = current_write_rate
            self._write_zero_start_time = None
            write_rate = current_write_rate
        else:
            if self._write_zero_start_time is None:
                self._write_zero_start_time = current_time
                write_rate = self._last_non_zero_write_rate
            elif current_time - self._write_zero_start_time >= self._zero_delay_duration:
                write_rate = 0.0
            else:
                write_rate = self._last_non_zero_write_rate

        # Header status colors based on individual read/write activity
        def get_io_color_and_status(rate):
            if rate < 10:
                return (0.2, 0.8, 0.2, 1.0)
            elif rate < 100:
                return (1.0, 0.8, 0.2, 1.0)
            else:
                return (1.0, 0.4, 0.4, 1.0)  # Lighter red for better readability
        
        read_color = get_io_color_and_status(read_rate)
        write_color = get_io_color_and_status(write_rate)
        
        imgui.text_colored(f"           Read {read_rate:.2f} MB/s", *read_color)
        imgui.same_line()
        imgui.text_colored(" | ", 1.0, 1.0, 1.0, 1.0)
        imgui.same_line()
        imgui.text_colored(f"Write {write_rate:.2f} MB/s", *write_color)

        # Read MB/s graph
        render_graph(
            "disk_read_mb_s",
            disk_read_mb_s,
            f"Read {read_rate:.2f} MB/s",
            scale_min=0,
            scale_max=max(1.0, max(disk_read_mb_s) if disk_read_mb_s else 1.0),
            height=45,
            color=(0.4, 0.8, 1.0, 0.9),
        )

        # Write MB/s graph
        render_graph(
            "disk_write_mb_s",
            disk_write_mb_s,
            f"Write {write_rate:.2f} MB/s",
            scale_min=0,
            scale_max=max(1.0, max(disk_write_mb_s) if disk_write_mb_s else 1.0),
            height=45,
            color=(1.0, 0.6, 0.2, 0.9),
        )

        imgui.spacing()

    def _check_memory_alerts(self, stats):
        """Check memory usage and trigger alerts if thresholds are exceeded."""
        if not hasattr(self, "_last_alert_time"):
            self._last_alert_time = {"ram": 0}

        current_time = time.time()
        alert_cooldown = 300  # 5 minutes

        ram_percent = stats.get("ram_usage_percent", [])
        if ram_percent:
            current_ram = ram_percent[-1]
            if current_ram >= 90 and (current_time - self._last_alert_time["ram"]) > alert_cooldown:
                ram_gb = stats.get("ram_usage_gb", [])
                ram_gb_val = ram_gb[-1] if ram_gb else 0.0
                ram_total = stats.get("ram_total_gb", 0.0)
                self.app.logger.warning(
                    f"[CRITICAL] HIGH MEMORY USAGE: {current_ram:.1f}% "
                    f"({ram_gb_val:.1f}/{ram_total:.1f} GB) - "
                    "Consider closing unnecessary applications or upgrading RAM.",
                    extra={"status_message": True},
                )
                self._last_alert_time["ram"] = current_time
            elif current_ram >= 85 and (current_time - self._last_alert_time["ram"]) > alert_cooldown:
                self.app.logger.warning(
                    f"[WARNING] Memory usage is high: {current_ram:.1f}% - Monitor for potential issues.",
                    extra={"status_message": True},
                )
                self._last_alert_time["ram"] = current_time

    def _render_content_ui_performance(self):
        """Render UI performance information with clean, organized layout."""
        self.ui_performance_perf.start_timing()
        app = self.app
        gui = app.gui_instance if hasattr(app, "gui_instance") else None
        if not gui:
            imgui.text_disabled("Performance data not available.")
            self.ui_performance_perf.end_timing()
            return

        imgui.text("UI Component Performance")
        imgui.separator()
        imgui.spacing()

        current_stats = (
            list(gui.component_render_times.items())
            if hasattr(gui, "component_render_times")
            else []
        )
        current_total = 0

        if current_stats:
            current_total = sum(t for _, t in current_stats)
            imgui.text_colored("Current Frame:", 0.8, 0.9, 1.0, 1.0)

            if current_total < 16.67:
                status_color = (0.2, 0.8, 0.2, 1.0)
                status_text = "[SMOOTH]"
                fps_target = "60+ FPS"
            elif current_total < 33.33:
                status_color = (1.0, 0.8, 0.2, 1.0)
                status_text = "[GOOD]"
                fps_target = "30-60 FPS"
            else:
                status_color = (1.0, 0.2, 0.2, 1.0)
                status_text = "[SLOW]"
                fps_target = "<30 FPS"

            imgui.same_line()
            imgui.text_colored(
                f" {current_total:.1f}ms {status_text} ({fps_target})", *status_color
            )
        else:
            imgui.text_disabled("No current frame data available")

        # Enhanced breakdown with visual indicators  
        if current_stats:
            imgui.spacing()
            imgui.text("Component Breakdown - All Components:")
            imgui.spacing()
            
            # Show ALL components with columns for better organization
            imgui.columns(3, "AllComponentColumns", True)
            imgui.text("Component")
            imgui.next_column()
            imgui.text("Time (ms)")
            imgui.next_column() 
            imgui.text("% of Total")
            imgui.next_column()
            imgui.separator()
            
            # Show all components sorted by time (most expensive first)
            for component, time_ms in current_stats:
                percentage = (time_ms / current_total) * 100 if current_total > 0 else 0
                
                # Color code the component name based on impact
                if time_ms > 5.0:
                    imgui.text_colored(component, 1.0, 0.2, 0.2, 1.0)  # Red - high impact
                elif time_ms > 1.0:
                    imgui.text_colored(component, 1.0, 0.8, 0.2, 1.0)  # Yellow - moderate impact
                else:
                    imgui.text_colored(component, 0.2, 0.8, 0.2, 1.0)  # Green - low impact
                
                imgui.next_column()
                imgui.text(f"{time_ms:.3f}")
                imgui.next_column()
                imgui.text(f"{percentage:.1f}%")
                imgui.next_column()
            
            imgui.columns(1)
            
            # Optional: Visual bars for top components
            imgui.spacing()
            if len(current_stats) > 5 and imgui.collapsing_header("Visual Performance Bars##PerfBars")[0]:
                imgui.text("Top Performance Impact:")
                # Show top 5 with visual bars
                for component, time_ms in current_stats[:5]:
                    percentage = (time_ms / current_total) * 100 if current_total > 0 else 0
                    
                    # Color code based on performance impact
                    if time_ms > 5.0:
                        color = (1.0, 0.2, 0.2, 1.0)  # Red
                    elif time_ms > 1.0:
                        color = (1.0, 0.8, 0.2, 1.0)  # Yellow
                    else:
                        color = (0.2, 0.8, 0.2, 1.0)  # Green
                    
                    imgui.text(f"{component}: {time_ms:.2f}ms ({percentage:.1f}%)")
                    
                    # Progress bar visualization
                    bar_width = min(percentage / 100.0, 1.0)
                    imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, *color)
                    imgui.progress_bar(bar_width, size=(300, 0))
                    imgui.pop_style_color()

        # Extended Performance Monitoring Section
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        
        if imgui.collapsing_header("Extended Performance Monitoring##ExtendedPerf")[0]:
            gui = app.gui_instance if hasattr(app, "gui_instance") else None
            if gui:
                imgui.columns(2, "ExtendedPerfColumns", True)
                
                # Video Decoding Performance
                imgui.text_colored("Video Decoding:", 0.8, 0.9, 1.0, 1.0)
                imgui.next_column()
                if hasattr(gui, 'video_decode_times') and gui.video_decode_times:
                    avg_decode = sum(gui.video_decode_times) / len(gui.video_decode_times)
                    recent_decode = gui.video_decode_times[-1] if gui.video_decode_times else 0
                    imgui.text(f"Recent: {recent_decode:.2f}ms | Avg: {avg_decode:.2f}ms")
                else:
                    imgui.text_disabled("No decode data available")
                imgui.next_column()
                
                # GPU Memory Usage
                imgui.text_colored("GPU Memory:", 0.8, 0.9, 1.0, 1.0)
                imgui.next_column()
                if hasattr(gui, 'gpu_memory_usage') and gui.gpu_memory_usage > 0:
                    if gui.gpu_memory_usage < 50:
                        color = (0.2, 0.8, 0.2, 1.0)  # Green
                    elif gui.gpu_memory_usage < 80:
                        color = (1.0, 0.8, 0.2, 1.0)  # Yellow
                    else:
                        color = (1.0, 0.2, 0.2, 1.0)  # Red
                    imgui.text_colored(f"{gui.gpu_memory_usage:.1f}% used", *color)
                else:
                    imgui.text_disabled("GPU monitoring unavailable")
                imgui.next_column()
                
                # Disk I/O Performance
                imgui.text_colored("Disk I/O:", 0.8, 0.9, 1.0, 1.0)
                imgui.next_column()
                if hasattr(gui, 'disk_io_times') and gui.disk_io_times:
                    recent_io = gui.disk_io_times[-1] if gui.disk_io_times else 0
                    avg_io = sum(gui.disk_io_times) / len(gui.disk_io_times)
                    if avg_io < 5.0:
                        color = (0.2, 0.8, 0.2, 1.0)
                    elif avg_io < 20.0:
                        color = (1.0, 0.8, 0.2, 1.0)
                    else:
                        color = (1.0, 0.2, 0.2, 1.0)
                    imgui.text_colored(f"Recent: {recent_io:.2f}ms | Avg: {avg_io:.2f}ms", *color)
                else:
                    imgui.text_disabled("No I/O data available")
                imgui.next_column()
                
                # Network Operations
                imgui.text_colored("Network Ops:", 0.8, 0.9, 1.0, 1.0)
                imgui.next_column()
                if hasattr(gui, 'network_operation_times') and gui.network_operation_times:
                    recent_net = gui.network_operation_times[-1] if gui.network_operation_times else 0
                    avg_net = sum(gui.network_operation_times) / len(gui.network_operation_times)
                    if avg_net < 100.0:
                        color = (0.2, 0.8, 0.2, 1.0)
                    elif avg_net < 500.0:
                        color = (1.0, 0.8, 0.2, 1.0)
                    else:
                        color = (1.0, 0.2, 0.2, 1.0)
                    imgui.text_colored(f"Recent: {recent_net:.1f}ms | Avg: {avg_net:.1f}ms", *color)
                else:
                    imgui.text_disabled("No network data available")
                imgui.next_column()
                
                imgui.columns(1)
                
                # Performance Budget Analysis
                imgui.spacing()
                imgui.text_colored("Frame Budget Analysis (60fps = 16.67ms):", 0.9, 0.9, 0.3, 1.0)
                if current_total > 0:
                    budget_used = (current_total / 16.67) * 100
                    remaining = max(0, 16.67 - current_total)
                    
                    if budget_used < 60:
                        budget_color = (0.2, 0.8, 0.2, 1.0)
                        status = "PLENTY OF HEADROOM"
                    elif budget_used < 90:
                        budget_color = (1.0, 0.8, 0.2, 1.0)
                        status = "GOOD PERFORMANCE"
                    else:
                        budget_color = (1.0, 0.2, 0.2, 1.0)
                        status = "FRAME BUDGET EXCEEDED"
                    
                    imgui.text(f"Budget used: ")
                    imgui.same_line()
                    imgui.text_colored(f"{budget_used:.1f}% ({status})", *budget_color)
                    imgui.text(f"Remaining budget: {remaining:.2f}ms")
                    
                    # Visual budget bar
                    imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, *budget_color)
                    imgui.progress_bar(min(budget_used / 100.0, 1.0), size=(300, 0), overlay=f"{budget_used:.1f}%")
                    imgui.pop_style_color()

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Frontend: Always read from queue for continuous data
        current_accumulated_times = {}
        current_frame_count = 0
        
        # Get the most recent data from the queue
        if hasattr(gui, "_frontend_perf_queue") and gui._frontend_perf_queue:
            latest_data = gui._frontend_perf_queue[-1]  # Get most recent entry
            current_accumulated_times = latest_data.get('accumulated_times', {})
            current_frame_count = latest_data.get('frame_count', 0)

        if current_accumulated_times and current_frame_count > 0:
            imgui.text_colored("Average Performance:", 0.8, 0.9, 1.0, 1.0)
            imgui.same_line()
            imgui.text(f"({current_frame_count} frames tracked)")

            avg_stats = list(current_accumulated_times.items())
            avg_total = (
                sum(total_time / current_frame_count for _, total_time in avg_stats)
                if current_frame_count > 0
                else 0
            )

            avg_stats_for_expensive = list(current_accumulated_times.items())
            if current_frame_count > 0:
                avg_stats_for_expensive.sort(
                    key=lambda x: x[1] / current_frame_count, reverse=True
                )
            else:
                avg_stats_for_expensive.sort(key=lambda x: x[0].lower())

            if avg_total < 16.67:
                avg_status_color = (0.2, 0.8, 0.2, 1.0)
                avg_status_text = "[EXCELLENT]"
            elif avg_total < 33.33:
                avg_status_color = (1.0, 0.8, 0.2, 1.0)
                avg_status_text = "[GOOD]"
            else:
                avg_status_color = (1.0, 0.2, 0.2, 1.0)
                avg_status_text = "[NEEDS OPTIMIZATION]"

            imgui.text_colored(
                f"Overall: {avg_total:.1f}ms {avg_status_text}", *avg_status_color
            )

            imgui.spacing()

            imgui.text("Most Expensive Components (avg):")
            for i, (component, total_time) in enumerate(
                avg_stats_for_expensive[:3]
            ):
                avg_time = (
                    total_time / current_frame_count if current_frame_count > 0 else 0.0
                )
                time_color = (
                    (0.0, 1.0, 0.0, 1.0)
                    if avg_time < 5.0
                    else (1.0, 0.8, 0.0, 1.0)
                    if avg_time < 16.67
                    else (1.0, 0.2, 0.2, 1.0)
                )
                imgui.text_colored(
                    f"  {i+1}. {component}: {avg_time:.2f}ms", *time_color
                )

            if len(avg_stats_for_expensive) > 3:
                imgui.text_disabled(
                    f"  ... and {len(avg_stats_for_expensive) - 3} more components"
                )
        else:
            imgui.text_disabled("No historical data available")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        if (
            current_stats
            and len(current_stats) > 0
            and current_accumulated_times
            and current_frame_count > 0
        ):
            imgui.text_colored("All Components:", 0.8, 0.9, 1.0, 1.0)

            if not hasattr(self, "_perf_sort_mode"):
                self._perf_sort_mode = 0

            sort_modes = ["by Current", "by Average", "A-Z"]
            sort_label = sort_modes[self._perf_sort_mode]
            if imgui.small_button(f"Sort: {sort_label}"):
                self._perf_sort_mode = (self._perf_sort_mode + 1) % 3

            all_component_names = set()
            current_dict = dict(current_stats)
            avg_dict = {}

            for component, _ in current_stats:
                all_component_names.add(component)

            if current_frame_count > 0:
                for component, total_time in current_accumulated_times.items():
                    all_component_names.add(component)
                    avg_dict[component] = total_time / current_frame_count
            else:
                for component in current_accumulated_times.keys():
                    all_component_names.add(component)
                    avg_dict[component] = 0.0

            complete_stats = []
            for component in all_component_names:
                current_time = current_dict.get(component, 0.0)
                avg_time = avg_dict.get(component, 0.0)
                complete_stats.append((component, current_time, avg_time))

            if self._perf_sort_mode == 0:
                complete_stats.sort(key=lambda x: x[1], reverse=True)
            elif self._perf_sort_mode == 1:
                complete_stats.sort(key=lambda x: x[2], reverse=True)
            else:
                complete_stats.sort(key=lambda x: x[0].lower())

            imgui.spacing()
            imgui.columns(4, "complete_perf_table", border=False)
            imgui.set_column_width(0, 180)
            imgui.set_column_width(1, 70)
            imgui.set_column_width(2, 70)
            imgui.set_column_width(3, 70)
            imgui.text("Component")
            imgui.next_column()
            imgui.text("Time (ms)")
            imgui.next_column()
            imgui.text("Avg (ms)")
            imgui.next_column()
            imgui.text("% of Total")
            imgui.next_column()
            imgui.separator()

            for component, current_time, avg_time in complete_stats:
                percentage = (
                    (current_time / current_total) * 100
                    if current_total > 0 and current_time > 0
                    else 0.0
                )
                imgui.text(component)
                imgui.next_column()

                if current_time > 0:
                    time_color = (
                        (0.0, 1.0, 0.0, 1.0)
                        if current_time < 16.67
                        else (1.0, 0.8, 0.0, 1.0)
                        if current_time < 33.33
                        else (1.0, 0.2, 0.2, 1.0)
                    )
                    imgui.text_colored(f"{current_time:.2f}", *time_color)
                else:
                    imgui.text_disabled("0.00")
                imgui.next_column()

                if avg_time > 0:
                    avg_color = (
                        (0.0, 1.0, 0.0, 1.0)
                        if avg_time < 5.0
                        else (1.0, 0.8, 0.0, 1.0)
                        if avg_time < 16.67
                        else (1.0, 0.2, 0.2, 1.0)
                    )
                    imgui.text_colored(f"{avg_time:.2f}", *avg_color)
                else:
                    imgui.text_disabled("0.00")
                imgui.next_column()

                if percentage > 0:
                    imgui.text(f"{percentage:.1f}%")
                else:
                    imgui.text_disabled("-")
                imgui.next_column()

            imgui.columns(1)

        elif current_stats and len(current_stats) > 0:
            imgui.text_colored(
                "All Components (current frame only):", 0.8, 0.9, 1.0, 1.0
            )
            imgui.text_disabled("Historical data not yet available")

            imgui.spacing()

            imgui.columns(3, "current_only_table", border=False)
            imgui.set_column_width(0, 200)
            imgui.set_column_width(1, 80)
            imgui.text("Component")
            imgui.next_column()
            imgui.text("Time (ms)")
            imgui.next_column()
            imgui.text("% of Total")
            imgui.next_column()
            imgui.separator()

            for component, render_time in current_stats:
                percentage = (render_time / current_total) * 100 if current_total > 0 else 0
                imgui.text(component)
                imgui.next_column()
                time_color = (
                    (0.0, 1.0, 0.0, 1.0)
                    if render_time < 16.67
                    else (1.0, 0.8, 0.0, 1.0)
                    if render_time < 33.33
                    else (1.0, 0.2, 0.2, 1.0)
                )
                imgui.text_colored(f"{render_time:.2f}", *time_color)
                imgui.next_column()
                imgui.text(f"{percentage:.1f}%")
                imgui.next_column()

            imgui.columns(1)

        imgui.spacing()
        imgui.separator()

        if hasattr(gui, "last_perf_log_time") and hasattr(gui, "perf_log_interval"):
            time_since_log = time.time() - gui.last_perf_log_time
            next_log_in = gui.perf_log_interval - time_since_log

            imgui.text_disabled(f"Next debug log in: {next_log_in:.1f}s")
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Performance data is logged to console every few seconds.\n\n"
                    "Color coding:\n"
                    "Green: Excellent performance\n"
                    "Yellow: Acceptable performance\n"
                    "Red: Needs optimization"
                )

        self.ui_performance_perf.end_timing()