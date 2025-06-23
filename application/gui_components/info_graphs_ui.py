import imgui
import os
from application.utils.time_format import _format_time


class InfoGraphsUI:
    def __init__(self, app):
        self.app = app

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
            imgui.begin("Graphs##RightGraphsContainer",
                        flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE)

        # Render tabbed content, which is now the main view
        self._render_tabbed_content()

        imgui.end()

    def _render_tabbed_content(self):
        if imgui.begin_tab_bar("InfoGraphsTabs"):
            # --- Video Tab ---
            if imgui.begin_tab_item("Video")[0]:
                imgui.spacing()
                if imgui.collapsing_header("Video Information##VideoInfoSection", flags=imgui.TREE_NODE_DEFAULT_OPEN)[
                    0]:
                    self._render_content_video_info()
                imgui.separator()
                if imgui.collapsing_header("Video Settings##VideoSettingsSection", flags=imgui.TREE_NODE_DEFAULT_OPEN)[
                    0]:
                    self._render_content_video_settings()
                imgui.end_tab_item()

            # --- Funscript Tab ---
            if imgui.begin_tab_item("Funscript")[0]:
                imgui.spacing()
                if imgui.collapsing_header("Funscript Info (Timeline 1)##FSInfoT1Section",
                                           flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                    self._render_content_funscript_info(1)
                imgui.separator()
                if self.app.app_state_ui.show_funscript_interactive_timeline2:
                    if imgui.collapsing_header("Funscript Info (Timeline 2)##FSInfoT2Section",
                                               flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                        self._render_content_funscript_info(2)
                else:
                    imgui.text_disabled("Enable Interactive Timeline 2 to see its stats.")
                imgui.end_tab_item()

            # --- History Tab ---
            if imgui.begin_tab_item("History")[0]:
                imgui.spacing()
                if imgui.collapsing_header("Undo-Redo History##UndoRedoSection", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                    self._render_content_undo_redo_history()
                imgui.end_tab_item()

            imgui.end_tab_bar()

    def _get_k_resolution_label(self, width, height):
        if width <= 0 or height <= 0: return ""
        if (1280, 720) == (width, height): return " (HD)"
        if (1920, 1080) == (width, height): return " (Full HD)"
        if (2560, 1440) == (width, height): return " (QHD/2.5K)"
        if (3840, 2160) == (width, height): return " (4K UHD)"
        if width >= 7000: return " (8K)"
        if width >= 3800: return " (4K)"
        return ""

    def _render_content_video_info(self):
        file_mgr = self.app.file_manager
        imgui.text("Video Path:")
        imgui.text_wrapped(file_mgr.video_path if file_mgr.video_path else "N/A (Drag & Drop Video)")
        imgui.separator()

        imgui.columns(2, "video_info_stats", border=False)
        imgui.set_column_width(0, 120 * imgui.get_io().font_global_scale)

        if self.app.processor and self.app.processor.video_info:
            info = self.app.processor.video_info
            width, height = info.get('width', 0), info.get('height', 0)
            imgui.text("Resolution:")
            imgui.next_column()
            imgui.text(f"{width}x{height}{self._get_k_resolution_label(width, height)}")
            imgui.next_column()
            imgui.text("FPS:")
            imgui.next_column()
            imgui.text(f"{info.get('fps', 0):.2f}")
            imgui.next_column()
            imgui.text("Total Frames:")
            imgui.next_column();
            imgui.text(f"{info.get('total_frames', 0)}")
            imgui.next_column()
            imgui.text("Duration:")
            imgui.next_column()
            imgui.text(f"{_format_time(self.app, info.get('duration', 0.0))}")
            imgui.next_column()
            imgui.text("Detected Type:")
            imgui.next_column()
            imgui.text(f"{self.app.processor.determined_video_type or 'N/A'}")
            imgui.next_column()
        else:
            imgui.text("Status:");
            imgui.next_column();
            imgui.text("Video details not loaded.");
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

            changed_pitch, new_pitch = imgui.slider_int("View Pitch##vrPitch", processor.vr_pitch, -40, 40)
            if changed_pitch:
                processor.set_active_vr_parameters(pitch=new_pitch)
                processor.reapply_video_settings()

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
