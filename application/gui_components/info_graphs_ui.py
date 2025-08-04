import imgui
import os
import threading
from application.utils.time_format import _format_time


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

    def _apply_video_render(self, new_pitch):
        """Execute video rendering with the new pitch value after delay"""
        processor = self.app.processor
        if processor:
            processor.set_active_vr_parameters(pitch=new_pitch)
            processor.reapply_video_settings()

    def _schedule_video_render(self, new_pitch):
        """Schedule video rendering with delay, canceling any existing timer"""
        with self.timer_lock:
            # Cancel existing timer if it exists
            if self.video_render_timer:
                self.video_render_timer.cancel()
            
            # Create new timer for delay
            self.video_render_timer = threading.Timer(
                self.PITCH_SLIDER_DELAY_MS / 1000.0, 
                self._apply_video_render, 
                args=[new_pitch]
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

    def render(self):
        app_state = self.app.app_state_ui
        window_title = "Info & Graphs##InfoGraphsFloating"

        # Determine flags based on layout mode
        if app_state.ui_layout_mode == 'floating':
            if not getattr(app_state, 'show_info_graphs_window', True):
                return
            is_open, new_visibility = imgui.begin(window_title, closable=True)
            if new_visibility != app_state.show_info_graphs_window:
                app_state.show_info_graphs_window = new_visibility
            if not is_open:
                imgui.end()
                return
        else:  # Fixed mode
            imgui.begin("Graphs##RightGraphsContainer", flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE)

        if app_state.ui_view_mode == 'simple':
            self._render_simple_view_content()
        else:
            self._render_tabbed_content()

        imgui.end()

    def _render_simple_view_content(self):
        """Renders only the video information for Simple Mode."""
        imgui.begin_child("SimpleInfoChild", border=False)
        imgui.spacing()
        # We don't need the collapsing header in this focused view
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

        avail = imgui.get_content_region_available()
        imgui.begin_child("InfoGraphsTabContent", width=0, height=avail[1], border=False)
        if tab_selected == "video":
            imgui.spacing()
            if imgui.collapsing_header("Video Information##VideoInfoSection", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                self._render_content_video_info()
            imgui.separator()
            if imgui.collapsing_header("Video Settings##VideoSettingsSection", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                self._render_content_video_settings()
        elif tab_selected == "funscript":
            imgui.spacing()
            if imgui.collapsing_header("Funscript Info (Timeline 1)##FSInfoT1Section", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                self._render_content_funscript_info(1)
            imgui.separator()
            if self.app.app_state_ui.show_funscript_interactive_timeline2:
                if imgui.collapsing_header("Funscript Info (Timeline 2)##FSInfoT2Section", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                    self._render_content_funscript_info(2)
            else:
                imgui.text_disabled("Enable Interactive Timeline 2 to see its stats.")
        elif tab_selected == "history":
            imgui.spacing()
            if imgui.collapsing_header("Undo-Redo History##UndoRedoSection", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                self._render_content_undo_redo_history()
        elif tab_selected == "performance":
            imgui.spacing()
            if imgui.collapsing_header("Performance Monitor##PerformanceSection", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                self._render_content_performance()
        imgui.end_child()

    def _get_k_resolution_label(self, width, height):
        if width <= 0 or height <= 0: return ""
        if (1280, 720) == (width, height): return " (HD)"
        if (1920, 1080) == (width, height): return " (Full HD)"
        if (2560, 1440) == (width, height): return " (QHD/2.5K)"
        if (3840, 2160) == (width, height): return " (4K UHD)"
        # Next, apply general checks for VR resolutions based on width, from highest to lowest
        if width >= 7600: return " (8K)"
        if width >= 6600: return " (7K)"
        if width >= 5600: return " (6K)"
        if width >= 5000: return " (5K)"
        if width >= 3800: return " (4K)"
        return ""

    def _render_content_video_info(self):
        file_mgr = self.app.file_manager

        imgui.columns(2, "video_info_stats", border=False)
        imgui.set_column_width(0, 120 * imgui.get_io().font_global_scale)

        if self.app.processor and self.app.processor.video_info:
            path = (os.path.dirname(file_mgr.video_path) if file_mgr.video_path else "N/A (Drag & Drop Video)")
            filename =  self.app.processor.video_info.get('filename', "N/A")

            info = self.app.processor.video_info
            width, height = info.get('width', 0), info.get('height', 0)

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
            imgui.text(f"{width}x{height}{self._get_k_resolution_label(width, height)}")
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
            # Display with more precision and add VFR/CFR indicator
            fps_text = f"{info.get('fps', 0):.3f}"
            fps_mode = " (VFR)" if info.get('is_vfr', False) else " (CFR)"
            imgui.text(fps_text + fps_mode)
            imgui.next_column()

            imgui.text("Size:")
            imgui.next_column()
            # Format file size from bytes to MB or GB
            size_bytes = info.get('file_size', 0)
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
            # Format bitrate from bps to Mbit/s
            bitrate_bps = info.get('bitrate', 0)
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

            # --- Section to Display Active Source ---
            imgui.text("Active Source:")
            imgui.next_column()
            processor = self.app.processor
            if hasattr(processor, '_active_video_source_path') and processor._active_video_source_path != processor.video_path:
                imgui.text("Preprocessed File")
                if imgui.is_item_hovered():
                    imgui.set_tooltip(f"Using: {os.path.basename(processor._active_video_source_path)}\nAll filtering/de-warping is pre-applied.")
            else:
                imgui.text("Original File")
                if imgui.is_item_hovered():
                    imgui.set_tooltip(f"Using: {os.path.basename(processor.video_path)}\nFilters are applied on-the-fly.")
            imgui.next_column()
        else:
            imgui.text("Status:")
            imgui.next_column()
            imgui.text("Video details not loaded.")
            imgui.next_column()
        imgui.columns(1)
        imgui.spacing()

    def _render_content_video_settings(self):
        processor = self.app.processor
        if not processor:
            imgui.text("VideoProcessor not initialized.")
            return

        imgui.text("Hardware Acceleration")
        hw_accel_options = self.app.available_ffmpeg_hwaccels
        hw_accel_display = [name.replace("_", " ").title() if name not in ["auto", "none"] else (
            "Auto Detect" if name == "auto" else "None (CPU Only)") for name in hw_accel_options]

        try:
            current_hw_idx = hw_accel_options.index(self.app.hardware_acceleration_method)
        except ValueError:
            current_hw_idx = 0

        changed, new_idx = imgui.combo("Method##HWAccel", current_hw_idx, hw_accel_display)
        if changed:
            self.app.hardware_acceleration_method = hw_accel_options[new_idx]
            self.app.app_settings.set("hardware_acceleration_method", self.app.hardware_acceleration_method)

            if processor.is_video_open():
                processor.reapply_video_settings()

        imgui.separator()
        video_types = ["auto", "2D", "VR"]
        current_type_idx = video_types.index(
            processor.video_type_setting) if processor.video_type_setting in video_types else 0
        changed, new_idx = imgui.combo("Video Type##vidType", current_type_idx, video_types)
        if changed:
            processor.set_active_video_type_setting(video_types[new_idx])
            processor.reapply_video_settings()

        if processor.is_vr_active_or_potential():
            imgui.separator()
            imgui.text("VR Settings")
            vr_fmt_disp = ["Equirectangular (SBS)", "Fisheye (SBS)", "Equirectangular (TB)", "Fisheye (TB)",
                           "Equirectangular (Mono)", "Fisheye (Mono)"]
            vr_fmt_val = ["he_sbs", "fisheye_sbs", "he_tb", "fisheye_tb", "he", "fisheye"]
            current_vr_idx = vr_fmt_val.index(
                processor.vr_input_format) if processor.vr_input_format in vr_fmt_val else 0
            changed, new_idx = imgui.combo("Input Format##vrFmt", current_vr_idx, vr_fmt_disp)
            if changed:
                processor.set_active_vr_parameters(input_format=vr_fmt_val[new_idx])
                processor.reapply_video_settings()

            # Track slider dragging state
            is_slider_hovered = imgui.is_item_hovered()
            is_mouse_down = imgui.is_mouse_down(0)  # Left mouse button
            is_mouse_released = imgui.is_mouse_released(0)  # Left mouse button released
            
            # Update dragging state
            if is_slider_hovered and is_mouse_down:
                self.pitch_slider_is_dragging = True
                self.pitch_slider_was_dragging = True
            elif (is_mouse_released or not is_mouse_down) and self.pitch_slider_was_dragging:
                self._handle_mouse_release()

            changed_pitch, new_pitch = imgui.slider_int("View Pitch##vrPitch", processor.vr_pitch, self.PITCH_SLIDER_MIN, self.PITCH_SLIDER_MAX)
            if changed_pitch:
                # Update the processor value immediately so slider shows the new value
                processor.vr_pitch = new_pitch
                
                # Update the tracked value
                self.last_pitch_value = new_pitch
                
                # Schedule debounced video rendering (only the expensive video processing)
                self._schedule_video_render(new_pitch)

    def _render_content_funscript_info(self, timeline_num):
        fs_proc = self.app.funscript_processor
        stats = fs_proc.funscript_stats_t1 if timeline_num == 1 else fs_proc.funscript_stats_t2
        source_text = stats.get("source_type", "N/A")

        if source_text == "File" and stats.get("path", "N/A") != "N/A":
            source_text = f"File: {os.path.basename(stats['path'])}"
        elif stats.get("path", "N/A") != "N/A":
            source_text = stats['path']

        imgui.text_wrapped(f"Source: {source_text}")
        imgui.separator()

        imgui.columns(2, f"fs_stats_{timeline_num}", border=False)
        imgui.set_column_width(0, 180 * imgui.get_io().font_global_scale)

        def stat_row(label, value):
            imgui.text(label);
            imgui.next_column();
            imgui.text(str(value));
            imgui.next_column()

        stat_row("Points:", stats.get('num_points', 0))
        stat_row("Duration (s):", f"{stats.get('duration_scripted_s', 0.0):.2f}")
        stat_row("Total Travel:", stats.get('total_travel_dist', 0))
        stat_row("Strokes:", stats.get('num_strokes', 0))
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
        stat_row("Position Range:", f"{stats.get('min_pos', 'N/A')} - {stats.get('max_pos', 'N/A')}")
        imgui.separator()
        imgui.next_column()
        imgui.separator()
        imgui.next_column()
        stat_row("Min/Max Interval (ms):",
                 f"{stats.get('min_interval_ms', 'N/A')} / {stats.get('max_interval_ms', 'N/A')}")
        stat_row("Avg Interval (ms):", f"{stats.get('avg_interval_ms', 0.0):.2f}")

        imgui.columns(1)
        imgui.spacing()

    def _render_content_undo_redo_history(self):
        fs_proc = self.app.funscript_processor
        imgui.begin_child("UndoRedoChild", height=150, border=True)

        def render_history_for_timeline(num):
            manager = fs_proc._get_undo_manager(num)
            if not manager: return

            imgui.text(f"T{num} Undo History:");
            imgui.next_column()
            imgui.text(f"T{num} Redo History:");
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
        """Render performance information with sorting and total time."""
        app = self.app
        gui = app.gui_instance if hasattr(app, 'gui_instance') else None
        if not gui:
            imgui.text_disabled("Performance data not available.")
            return

        # Sorting state
        if not hasattr(self, '_perf_sort_mode'):
            self._perf_sort_mode = 0  # 0: slowest→fastest, 1: fastest→slowest, 2: alphabetical
        sort_modes = ["Slowest→Fastest", "Fastest→Slowest", "A→Z"]
        if imgui.button(f"Sort: {sort_modes[self._perf_sort_mode]}"):
            self._perf_sort_mode = (self._perf_sort_mode + 1) % 3
        if imgui.is_item_hovered():
            imgui.set_tooltip("Click to cycle sort order for the list below.")
        imgui.spacing()

        # Prepare data
        stats = list(gui.component_render_times.items())
        if self._perf_sort_mode == 0:
            stats.sort(key=lambda x: x[1], reverse=True)  # slowest→fastest
        elif self._perf_sort_mode == 1:
            stats.sort(key=lambda x: x[1])  # fastest→slowest
        else:
            stats.sort(key=lambda x: x[0].lower())  # alphabetical

        # Display stats
        imgui.text("Frame Render Times (ms):")
        total = 0.0
        for component, render_time in stats:
            color = (0.0, 1.0, 0.0, 1.0) if render_time < 16.67 else (1.0, 1.0, 0.0, 1.0)
            imgui.text_colored(f"  {component}: {render_time:.2f}", *color)
            total += render_time
        if not stats:
            imgui.text_disabled("  No render data available")
        imgui.separator()
        imgui.text(f"Total: {total:.2f} ms")
        imgui.spacing()

        # Show average performance if available
        if gui.perf_accumulated_times and gui.perf_frame_count > 0:
            imgui.text("Average Times (ms):")
            avg_stats = list(gui.perf_accumulated_times.items())
            if self._perf_sort_mode == 0:
                avg_stats.sort(key=lambda x: x[1]/gui.perf_frame_count, reverse=True)
            elif self._perf_sort_mode == 1:
                avg_stats.sort(key=lambda x: x[1]/gui.perf_frame_count)
            else:
                avg_stats.sort(key=lambda x: x[0].lower())
            avg_total = 0.0
            for component, total_time in avg_stats:
                avg_time = total_time / gui.perf_frame_count
                color = (0.0, 1.0, 0.0, 1.0) if avg_time < 16.67 else (1.0, 1.0, 0.0, 1.0)
                imgui.text_colored(f"  {component}: {avg_time:.2f}", *color)
                avg_total += avg_time
            if not avg_stats:
                imgui.text_disabled("  No average data available")
            imgui.separator()
            imgui.text(f"Total: {avg_total:.2f} ms")
            imgui.spacing()

        # Show frame count and timing info
        if gui.perf_frame_count > 0:
            imgui.text(f"Frames tracked: {gui.perf_frame_count}")
            import time
            time_since_log = time.time() - gui.last_perf_log_time
            imgui.text(f"Next log in: {gui.perf_log_interval - time_since_log:.1f}s")
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Shows real-time frame render times for GUI components.\n"
                "Green: < 16.67ms (60+ FPS)\n"
                "Yellow: ≥ 16.67ms (< 60 FPS)\n"
                "Data is logged every 5 seconds to debug output."
            )
