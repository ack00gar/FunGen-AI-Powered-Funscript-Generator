"""Video settings controls mixin for InfoGraphsUI."""
import imgui
import threading


# Constants
PITCH_SLIDER_DELAY_MS = 1000
PITCH_SLIDER_MIN = -40
PITCH_SLIDER_MAX = 40


class VideoSettingsMixin:

    def _apply_video_render(self, new_pitch):
        """Execute video rendering with the new pitch value after delay."""
        processor = self.app.processor
        if processor:
            processor.set_active_vr_parameters(pitch=new_pitch)
            processor.reapply_video_settings()

    def _schedule_video_render(self, new_pitch):
        """Schedule video rendering with delay, canceling any existing timer."""
        with self.timer_lock:
            if self.video_render_timer:
                self.video_render_timer.cancel()

            self.video_render_timer = threading.Timer(
                PITCH_SLIDER_DELAY_MS / 1000.0,
                self._apply_video_render,
                args=[new_pitch],
            )
            self.video_render_timer.daemon = True
            self.video_render_timer.start()

    def _cancel_video_render_timer(self):
        """Cancel any pending video render timer."""
        with self.timer_lock:
            if self.video_render_timer:
                self.video_render_timer.cancel()
                self.video_render_timer = None

    def _handle_mouse_release(self):
        """Handle mouse release - cancel timer and render immediately."""
        self.pitch_slider_is_dragging = False
        self.pitch_slider_was_dragging = False
        self._cancel_video_render_timer()

        # Execute video rendering immediately with final value
        if self.last_pitch_value is not None:
            self._apply_video_render(self.last_pitch_value)

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
                # Run reapply in background thread to avoid blocking UI
                threading.Thread(
                    target=processor.reapply_video_settings,
                    daemon=True,
                    name='HWAccelReapply'
                ).start()

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
            # Run reapply in background thread to avoid blocking UI
            threading.Thread(
                target=processor.reapply_video_settings,
                daemon=True,
                name='VideoTypeReapply'
            ).start()

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
                # Run reapply in background thread to avoid blocking UI
                threading.Thread(
                    target=processor.reapply_video_settings,
                    daemon=True,
                    name='VRFormatReapply'
                ).start()

            # VR Unwarp Method dropdown
            imgui.spacing()
            unwarp_method_disp = ["Auto (Metal/OpenGL)", "GPU Metal", "GPU OpenGL", "CPU (v360)", "None (Crop Only)"]
            unwarp_method_val = ["auto", "metal", "opengl", "v360", "none"]

            # Get current unwarp method
            current_unwarp_method = getattr(processor, 'vr_unwarp_method_override', 'auto')
            try:
                current_unwarp_idx = unwarp_method_val.index(current_unwarp_method)
            except ValueError:
                current_unwarp_idx = 0

            changed_unwarp, new_unwarp_idx = imgui.combo(
                "Unwarp Method##vrUnwarp", current_unwarp_idx, unwarp_method_disp
            )
            if changed_unwarp:
                processor.vr_unwarp_method_override = unwarp_method_val[new_unwarp_idx]
                # Save to settings
                self.app.app_settings.set("vr_unwarp_method", unwarp_method_val[new_unwarp_idx])
                # Run reapply in background thread to avoid blocking UI
                threading.Thread(
                    target=processor.reapply_video_settings,
                    daemon=True,
                    name='UnwarpMethodReapply'
                ).start()

            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Unwarp Method:\n"
                    "Auto: Automatically choose Metal or OpenGL\n"
                    "GPU Metal: Use Metal shader (macOS)\n"
                    "GPU OpenGL: Use OpenGL shader (cross-platform)\n"
                    "CPU (v360): Use FFmpeg v360 filter (slower)\n"
                    "None (Crop Only): Skip unwarping, just crop one panel.\n"
                    "  Best performance for User ROI / OF-only trackers.\n"
                    "  YOLO detection may not work on warped image."
                )

            # Panel selector — shown when unwarp is "None (Crop Only)" and format is stereo
            vr_fmt = getattr(processor, 'vr_input_format', '')
            is_sbs = '_sbs' in vr_fmt or '_lr' in vr_fmt or '_rl' in vr_fmt
            is_tb = '_tb' in vr_fmt
            if current_unwarp_method == 'none' and (is_sbs or is_tb):
                if is_sbs:
                    panel_disp = ["Left", "Right"]
                else:
                    panel_disp = ["Top", "Bottom"]
                panel_val = ["first", "second"]
                current_panel = self.app.app_settings.get("vr_crop_panel", "first")
                try:
                    current_panel_idx = panel_val.index(current_panel)
                except ValueError:
                    current_panel_idx = 0

                changed_panel, new_panel_idx = imgui.combo(
                    "Crop Panel##vrCropPanel", current_panel_idx, panel_disp
                )
                if changed_panel:
                    self.app.app_settings.set("vr_crop_panel", panel_val[new_panel_idx])
                    processor.vr_crop_panel = panel_val[new_panel_idx]
                    threading.Thread(
                        target=processor.reapply_video_settings,
                        daemon=True,
                        name='CropPanelReapply'
                    ).start()
                if imgui.is_item_hovered():
                    label = "left/right" if is_sbs else "top/bottom"
                    imgui.set_tooltip(f"Select which {label} panel to crop from the stereo VR frame.")

            # View Pitch slider — hidden when unwarp is "None (Crop Only)" since pitch has no effect
            if current_unwarp_method != 'none':
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
                    PITCH_SLIDER_MIN,
                    PITCH_SLIDER_MAX,
                )
                if changed_pitch:
                    processor.vr_pitch = new_pitch
                    self.last_pitch_value = new_pitch
                    self._schedule_video_render(new_pitch)

        self.video_settings_perf.end_timing()
