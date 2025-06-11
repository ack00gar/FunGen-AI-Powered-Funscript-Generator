# info_graphs_ui.py
import imgui
import os  # For os.path.basename
from application.utils.time_format import _format_time


class InfoGraphsUI:
    def __init__(self, app):
        self.app = app

    def render(self):
        app_state = self.app.app_state_ui

        sections = {
            "show_video_info_section": ("Video Information", self._render_content_video_info),
            "show_video_settings_section": ("Video Settings", self._render_content_video_settings),
            "show_funscript_info_t1_section": ("Funscript Info (Timeline 1)",
                                               lambda: self._render_content_funscript_info(1)),
            "show_funscript_info_t2_section": ("Funscript Info (Timeline 2)",
                                               lambda: self._render_content_funscript_info(2)),
            "show_undo_redo_history_section": ("Undo-Redo History", self._render_content_undo_redo_history),
        }

        if app_state.ui_layout_mode == 'floating':
            for flag_name, (title, content_func) in sections.items():
                is_visible = getattr(app_state, flag_name, False)
                if flag_name == "show_funscript_info_t2_section" and not app_state.show_funscript_interactive_timeline2:
                    is_visible = False
                if is_visible:
                    is_open, new_visibility = imgui.begin(title, closable=True, flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)
                    setattr(app_state, flag_name, new_visibility)
                    if is_open:
                        content_func()
                    imgui.end()
        else:  # fixed mode
            imgui.begin("Graphs##RightGraphsContainer",
                        flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE)

            for flag_name, (title, content_func) in sections.items():
                is_visible = getattr(app_state, flag_name, False)
                if flag_name == "show_funscript_info_t2_section" and not app_state.show_funscript_interactive_timeline2:
                    continue

                if is_visible:
                    header_title = f"{title}##{title.replace(' ', '')}Fixed"
                    default_open_flag = imgui.TREE_NODE_DEFAULT_OPEN if flag_name != "show_undo_redo_history_section" else 0
                    expanded, new_visibility = imgui.collapsing_header(header_title, is_visible, flags=default_open_flag)
                    setattr(app_state, flag_name, new_visibility)
                    if expanded:
                        content_func()
            imgui.end()

    def _get_k_resolution_label(self, width, height):
        if width <= 0 or height <= 0:
            return ""

        # Prioritize common named resolutions based on exact width and height
        if (width == 1280 and height == 720): return " (HD)"
        if (width == 1920 and height == 1080): return " (Full HD)"
        if (width == 2560 and height == 1440): return " (QHD/2.5K)"
        if (width == 3840 and height == 2160): return " (4K UHD)"
        if (width == 4096 and height == 2160): return " (4K)"
        if (width == 5120 and height == 2880): return " (5K)"
        if (width == 7680 and height == 4320): return " (8K UHD)"

        # General K approximation based on width, if no exact standard above matched
        # These are more approximate and can be adjusted
        if width >= 7000:
            return " (8K)"
        elif width >= 6000:
            return " (6K)"
        elif width >= 5000:
            return " (5K)"
        elif width >= 3800:
            return " (4K)"
        elif width >= 3400:
            return " (3.5K/UWQHD)"
        elif width >= 2500:
            return " (2.5K/QHD)"
        elif width >= 1900:
            return " (2K)"
        return ""

    def _render_content_video_info(self):
        file_mgr = self.app.file_manager
        imgui.text("Video Path:")
        imgui.text_wrapped(file_mgr.video_path if file_mgr.video_path else "N/A (Drag & Drop Video)")
        imgui.separator()

        # Offset for the value column for the statistics.
        # Adjusted to be wide enough for labels like "Detected Type:".
        value_column_offset = 130

        if self.app.processor and self.app.processor.video_info:
            info = self.app.processor.video_info
            width = info.get('width', 0)
            height = info.get('height', 0)
            k_label = self._get_k_resolution_label(width, height)

            imgui.text("Resolution:")
            imgui.same_line(value_column_offset)
            imgui.text(f"{width}x{height}{k_label}")

            imgui.text("FPS:")
            imgui.same_line(value_column_offset)
            imgui.text(f"{info.get('fps', 0):.2f}")

            imgui.text("Total Frames:")
            imgui.same_line(value_column_offset)
            imgui.text(f"{info.get('total_frames', 0)}")

            imgui.text("Duration:")
            imgui.same_line(value_column_offset)
            imgui.text(f"{_format_time(self.app, info.get('duration', 0.0))}")

            imgui.text("Detected Type:")
            imgui.same_line(value_column_offset)
            imgui.text(f"{self.app.processor.determined_video_type or 'N/A'}")
        else:
            # If processor or video_info is not available, display a general status.
            # The path (or "N/A") is already displayed above the separator.
            imgui.text("Status:")
            imgui.same_line(value_column_offset)
            imgui.text("Video details not loaded.")

        imgui.spacing()

    def _render_content_video_settings(self):
        if not self.app.processor:
            imgui.text("VideoProcessor not initialized.")
            return

        imgui.separator()
        imgui.text("Hardware Acceleration")

        # Get available methods from AppLogic
        hw_accel_options = self.app.available_ffmpeg_hwaccels

        # Create display names (e.g., capitalize, replace underscores)
        hw_accel_display = []
        for name in hw_accel_options:
            if name == "auto":
                hw_accel_display.append("Auto Detect")
            elif name == "none":
                hw_accel_display.append("None (CPU Only)")
            else:
                hw_accel_display.append(name.replace("_", " ").title())

        current_method_val = self.app.hardware_acceleration_method
        try:
            current_hw_idx = hw_accel_options.index(current_method_val)
        except ValueError:
            self.app.logger.warning(f"Current HW accel method '{current_method_val}' "
                                    f"not in available options {hw_accel_options}. Defaulting to 'auto'.")
            try:
                current_hw_idx = hw_accel_options.index("auto")
                self.app.hardware_acceleration_method = "auto"
            except ValueError:
                current_hw_idx = 0
                if hw_accel_options:
                    self.app.hardware_acceleration_method = hw_accel_options[0]
                else:
                    self.app.hardware_acceleration_method = "none"

        changed_hw_accel, new_hw_idx = imgui.combo("Method##HWAccelMethod", current_hw_idx, hw_accel_display)
        if changed_hw_accel:
            new_method_selected = hw_accel_options[new_hw_idx]
            if self.app.hardware_acceleration_method != new_method_selected:
                self.app.hardware_acceleration_method = new_method_selected
                self.app.app_settings.set("hardware_acceleration_method", new_method_selected)
                self.app.logger.info(f"Hardware acceleration set to: {new_method_selected}. Applying setting...",
                                     extra={'status_message': True})
                self.app.energy_saver.reset_activity_timer()

                # If a video is loaded, attempt to reapply settings to make the change active.
                # This allows the VideoProcessor to pick up the new HW accel setting for subsequent FFmpeg calls.
                if self.app.processor and self.app.processor.video_path:
                    self.app.processor.reapply_video_settings()
                else:
                    self.app.logger.info("No video loaded. HW Accel setting will apply when a video is opened.")
        imgui.separator()
        video_types = ["auto", "2D", "VR"]
        current_type_idx = video_types.index(
            self.app.processor.video_type_setting) if self.app.processor.video_type_setting in video_types else 0
        changed_type, new_type_idx = imgui.combo("Video Type##vidType", current_type_idx, video_types)
        if changed_type:
            self.app.processor.set_active_video_type_setting(video_types[new_type_idx])
            self.app.processor.reapply_video_settings()
        res_opts_vals = [320, 640, 1280]
        res_opts_disp = [str(r) for r in res_opts_vals]
        curr_res_idx = res_opts_vals.index(
            self.app.processor.yolo_input_size) if self.app.processor.yolo_input_size in res_opts_vals else (
            res_opts_vals.index(640) if 640 in res_opts_vals else 0)
        changed_res, new_res_idx = imgui.combo("Render Res (px)##yoloSize", curr_res_idx, res_opts_disp)
        if changed_res:
            self.app.processor.set_active_yolo_input_size(res_opts_vals[new_res_idx])
            self.app.processor.reapply_video_settings()

        if self.app.processor.is_vr_active_or_potential():
            imgui.separator()
            imgui.text("VR Settings")

            vr_fmt_disp = [
                "Equirectangular (SBS Left Eye)",
                "Fisheye (SBS Left Eye)",
                "Equirectangular (TB Top Eye)",
                "Fisheye (TB Top Eye)",
                "Equirectangular (Mono)",
                "Fisheye (Mono)",
            ]
            vr_fmt_val = [
                "he_sbs",
                "fisheye_sbs",
                "he_tb",  # Suffix '_tb' indicates Top-Bottom, top eye will be used
                "fisheye_tb",
                "he",
                "fisheye"
            ]

            curr_vr_fmt_idx = vr_fmt_val.index(
                self.app.processor.vr_input_format) if self.app.processor.vr_input_format in vr_fmt_val else 0
            changed_vr_fmt, new_vr_fmt_idx = imgui.combo("Input Format##vrFmt", curr_vr_fmt_idx, vr_fmt_disp)
            if changed_vr_fmt:
                new_format = vr_fmt_val[new_vr_fmt_idx]
                if self.app.processor.vr_input_format != new_format:
                    self.app.processor.set_active_vr_parameters(input_format=new_format)
                    self.app.processor.reapply_video_settings()

            fov_opts = [180, 190, 200, 210, 220]  # FOV options
            # Ensure current FOV is in options, default if not
            curr_fov_idx = fov_opts.index(self.app.processor.vr_fov) if self.app.processor.vr_fov in fov_opts else (
                fov_opts.index(190) if 190 in fov_opts else 0)
            changed_fov, temp_new_fov_idx = imgui.slider_int("Input FOV##vrFov", curr_fov_idx, 0, len(fov_opts) - 1,
                                                             format=f"{fov_opts[curr_fov_idx]}°")
            new_fov_idx = temp_new_fov_idx

            if changed_fov:
                self.app.processor.set_active_vr_parameters(fov=fov_opts[temp_new_fov_idx])
                self.app.processor.reapply_video_settings()

            changed_pitch, new_pitch = imgui.slider_int("View Pitch##vrPitch", self.app.processor.vr_pitch, -40, 40)
            if changed_pitch:
                self.app.processor.set_active_vr_parameters(pitch=new_pitch)
                self.app.processor.reapply_video_settings()
            imgui.separator()

        if imgui.button("Force Reload Stream##applyVidSettings"):
            self.app.processor.reapply_video_settings()
        imgui.spacing()

    def _render_content_funscript_info(self, timeline_num):
        fs_proc = self.app.funscript_processor
        stats = fs_proc.funscript_stats_t1 if timeline_num == 1 else fs_proc.funscript_stats_t2
        source_text = stats["source_type"]
        if stats["source_type"] == "File" and stats["path"] != "N/A":
            source_text = f"File: {os.path.basename(stats['path'])}"
        elif stats["path"] != "N/A" and stats["source_type"] != "File":
            source_text = stats["path"]
        imgui.text_wrapped(f"Source: {source_text}")
        imgui.separator()
        col_width1 = 180*imgui.get_io().font_global_scale
        imgui.text("Number of Points:")
        imgui.same_line(col_width1)
        imgui.text(f"{stats['num_points']}")
        imgui.text("Scripted Duration (s):")
        imgui.same_line(col_width1)
        imgui.text(f"{stats['duration_scripted_s']:.2f} s")
        imgui.text("Total Travel Distance:")
        imgui.same_line(col_width1)
        imgui.text(f"{stats['total_travel_dist']}")
        imgui.text("Number of Strokes:")
        imgui.same_line(col_width1)
        imgui.text(f"{stats['num_strokes']}")
        imgui.separator()
        imgui.text("Average Speed (pos/s):")
        imgui.same_line(col_width1)
        imgui.text(f"{stats['avg_speed_pos_per_s']:.2f}")
        imgui.text("Avg. Intensity (%):")
        imgui.same_line(col_width1)
        imgui.text(f"{stats['avg_intensity_percent']:.1f} %")
        imgui.separator()
        imgui.text("Position Range (0-100):")
        imgui.same_line(col_width1)
        imgui.text(f"{stats['min_pos']} - {stats['max_pos']}")
        imgui.separator()
        min_interval_display = f"{stats['min_interval_ms']}" if stats['min_interval_ms'] != -1 else "N/A"
        max_interval_display = f"{stats['max_interval_ms']}" if stats['max_interval_ms'] != -1 else "N/A"
        imgui.text("Min Interval (ms):")
        imgui.same_line(col_width1)
        imgui.text(min_interval_display)
        imgui.text("Max Interval (ms):")
        imgui.same_line(col_width1)
        imgui.text(max_interval_display)
        imgui.text("Avg Interval (ms):")
        imgui.same_line(col_width1)
        imgui.text(f"{stats['avg_interval_ms']:.2f}")
        imgui.spacing()

    def _render_content_undo_redo_history(self):
        fs_proc = self.app.funscript_processor
        app_state = self.app.app_state_ui
        imgui.begin_child("UndoRedoChild", height=150, border=True)

        # Timeline 1 History
        imgui.columns(2, "UndoRedoColumnsT1")
        imgui.text("T1 Undo History:")
        manager_t1 = fs_proc._get_undo_manager(1)
        if manager_t1:
            undo_history_t1 = manager_t1.get_undo_history_for_display()
            if undo_history_t1:
                for i, desc in enumerate(undo_history_t1):
                    if imgui.selectable(f"T1U {i}: {desc}", selected=False,
                                        flags=imgui.SELECTABLE_DONT_CLOSE_POPUPS): pass
            else:
                imgui.text_disabled("No undo history for T1")
        else:
            imgui.text_disabled("T1 Undo Manager N/A")

        imgui.next_column()
        imgui.text("T1 Redo History:")
        if manager_t1:
            redo_history_t1 = manager_t1.get_redo_history_for_display()
            if redo_history_t1:
                for i, desc in enumerate(redo_history_t1):
                    if imgui.selectable(f"T1R {i}: {desc}", selected=False,
                                        flags=imgui.SELECTABLE_DONT_CLOSE_POPUPS): pass
            else:
                imgui.text_disabled("No redo history for T1")
        else:
            imgui.text_disabled("T1 Redo Manager N/A")
        imgui.columns(1)  # Reset columns

        if app_state.show_funscript_interactive_timeline2:
            imgui.separator()
            imgui.columns(2, "UndoRedoColumnsT2")
            imgui.text("T2 Undo History:")
            manager_t2 = fs_proc._get_undo_manager(2)
            if manager_t2:
                undo_history_t2 = manager_t2.get_undo_history_for_display()
                if undo_history_t2:
                    for i, desc in enumerate(undo_history_t2):
                        if imgui.selectable(f"T2U {i}: {desc}", selected=False,
                                            flags=imgui.SELECTABLE_DONT_CLOSE_POPUPS): pass
                else:
                    imgui.text_disabled("No undo history for T2")
            else:
                imgui.text_disabled("T2 Undo Manager N/A")

            imgui.next_column()
            imgui.text("T2 Redo History:")
            if manager_t2:
                redo_history_t2 = manager_t2.get_redo_history_for_display()
                if redo_history_t2:
                    for i, desc in enumerate(redo_history_t2):
                        if imgui.selectable(f"T2R {i}: {desc}", selected=False,
                                            flags=imgui.SELECTABLE_DONT_CLOSE_POPUPS): pass
                else:
                    imgui.text_disabled("No redo history for T2")
            else:
                imgui.text_disabled("T2 Redo Manager N/A")
            imgui.columns(1)  # Reset columns

        imgui.end_child()

    # --- Main Render Method ---

    def render(self):
        app_state = self.app.app_state_ui

        sections = {
            "show_video_info_section": ("Video Information", self._render_content_video_info),
            "show_video_settings_section": ("Video Settings", self._render_content_video_settings),
            "show_funscript_info_t1_section": ("Funscript Info (Timeline 1)",
                                               lambda: self._render_content_funscript_info(1)),
            "show_funscript_info_t2_section": ("Funscript Info (Timeline 2)",
                                               lambda: self._render_content_funscript_info(2)),
            "show_undo_redo_history_section": ("Undo-Redo History", self._render_content_undo_redo_history),
        }

        if app_state.ui_layout_mode == 'floating':
            for flag_name, (title, content_func) in sections.items():
                is_visible = getattr(app_state, flag_name, False)
                if flag_name == "show_funscript_info_t2_section" and not app_state.show_funscript_interactive_timeline2:
                    is_visible = False  # Enforce dependency
                if is_visible:
                    is_open, new_visibility = imgui.begin(title, closable=True, flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)
                    setattr(app_state, flag_name, new_visibility)
                    if is_open:
                        content_func()
                    imgui.end()
        else:  # fixed mode
            imgui.begin("Graphs##RightGraphsContainer",
                        flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE)

            for flag_name, (title, content_func) in sections.items():
                is_visible = getattr(app_state, flag_name, False)
                if flag_name == "show_funscript_info_t2_section" and not app_state.show_funscript_interactive_timeline2:
                    continue  # Don't even show the header if T2 is not active

                if is_visible:
                    header_title = f"{title}##{title.replace(' ', '')}Fixed"
                    default_open_flag = imgui.TREE_NODE_DEFAULT_OPEN if flag_name != "show_undo_redo_history_section" else 0
                    expanded, new_visibility = imgui.collapsing_header(header_title, is_visible, flags=default_open_flag)
                    setattr(app_state, flag_name, new_visibility)
                    if expanded:
                        content_func()
            imgui.end()
