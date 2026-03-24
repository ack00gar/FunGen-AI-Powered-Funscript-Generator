"""Video settings controls mixin for InfoGraphsUI."""
import imgui
import threading
from application.utils.imgui_layout_helpers import (
    begin_settings_columns, end_settings_columns, row_label, row_end, row_separator,
)
from application.utils.section_card import section_card


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
            imgui.text_disabled("VideoProcessor not initialized.")
            self.video_settings_perf.end_timing()
            return

        begin_settings_columns("video_general_cols")

        # HW Acceleration
        row_label("HW Acceleration", "FFmpeg hardware acceleration method.\nRequires video reload to take effect.")
        hw_accel_options = self.app.available_ffmpeg_hwaccels
        hw_accel_display = [
            name.replace("_", " ").title()
            if name not in ["auto", "none"]
            else ("Auto Detect" if name == "auto" else "None (CPU Only)")
            for name in hw_accel_options
        ]
        try:
            current_hw_idx = hw_accel_options.index(self.app.hardware_acceleration_method)
        except ValueError:
            current_hw_idx = 0

        imgui.push_item_width(-1)
        changed, new_idx = imgui.combo("##HWAccelVid", current_hw_idx, hw_accel_display)
        imgui.pop_item_width()
        if changed:
            self.app.hardware_acceleration_method = hw_accel_options[new_idx]
            self.app.app_settings.set("hardware_acceleration_method", self.app.hardware_acceleration_method)
            self.app.notify(f"HW acceleration: {hw_accel_display[new_idx]}", "info", 2.0)
            if processor.is_video_open():
                threading.Thread(target=processor.reapply_video_settings, daemon=True, name='HWAccelReapply').start()
        row_end()

        # Video Type
        row_label("Video Type", "Auto-detect, force 2D, or force VR mode.")
        video_types = ["auto", "2D", "VR"]
        current_type_idx = video_types.index(processor.video_type_setting) if processor.video_type_setting in video_types else 0
        imgui.push_item_width(-1)
        changed, new_idx = imgui.combo("##vidType", current_type_idx, video_types)
        imgui.pop_item_width()
        if changed:
            processor.set_active_video_type_setting(video_types[new_idx])
            threading.Thread(target=processor.reapply_video_settings, daemon=True, name='VideoTypeReapply').start()
        row_end()

        end_settings_columns()

        # --- VR Settings (conditional) ---
        if processor.is_vr_active_or_potential():
            imgui.spacing()
            with section_card("VR Settings##VRSettingsCard", tier="secondary") as vr_open:
                if vr_open:
                    self._render_vr_settings(processor)

        # Reset button
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        if imgui.button("Reset Video Settings##ResetVideoDefaults", width=-1):
            self.app.hardware_acceleration_method = "auto"
            self.app.app_settings.set("hardware_acceleration_method", "auto")
            processor.set_active_video_type_setting("auto")
            if hasattr(processor, 'vr_unwarp_method_override'):
                processor.vr_unwarp_method_override = "auto"
                self.app.app_settings.set("vr_unwarp_method", "auto")
            processor.vr_pitch = 0
            self.app.app_settings.set("vr_crop_panel", "first")
            if processor.is_video_open():
                threading.Thread(target=processor.reapply_video_settings, daemon=True, name='ResetReapply').start()
            self.app.notify("Video settings reset to defaults", "info", 2.0)

        self.video_settings_perf.end_timing()

    def _render_vr_settings(self, processor):
        """Render VR-specific settings in harmonized layout."""
        begin_settings_columns("vr_cols")

        # Input Format
        row_label("Input Format", "The stereoscopic layout of the VR video file.")
        vr_fmt_disp = [
            "Equirectangular (SBS)", "Fisheye (SBS)",
            "Equirectangular (TB)", "Fisheye (TB)",
            "Equirectangular (Mono)", "Fisheye (Mono)",
        ]
        vr_fmt_val = ["he_sbs", "fisheye_sbs", "he_tb", "fisheye_tb", "he", "fisheye"]
        current_vr_idx = vr_fmt_val.index(processor.vr_input_format) if processor.vr_input_format in vr_fmt_val else 0
        imgui.push_item_width(-1)
        changed, new_idx = imgui.combo("##vrFmt", current_vr_idx, vr_fmt_disp)
        imgui.pop_item_width()
        if changed:
            processor.set_active_vr_parameters(input_format=vr_fmt_val[new_idx])
            threading.Thread(target=processor.reapply_video_settings, daemon=True, name='VRFormatReapply').start()
        row_end()

        # Unwarp Method
        show_advanced = self.app.app_state_ui.show_advanced_options
        if show_advanced:
            row_label("Unwarp Method",
                      "Auto: choose GPU backend automatically\n"
                      "GPU Metal: Metal shader (macOS)\n"
                      "GPU OpenGL: OpenGL shader (cross-platform)\n"
                      "CPU (v360): FFmpeg v360 filter (best quality)\n"
                      "None (Crop Only): skip unwarping, just crop one panel")
            unwarp_disp = ["Auto (GPU)", "GPU Metal", "GPU OpenGL", "CPU (v360)", "None (Crop Only)"]
            unwarp_val = ["auto", "metal", "opengl", "v360", "none"]
        else:
            row_label("Unwarp Method",
                      "Auto: GPU unwarping (fast)\n"
                      "CPU (v360): FFmpeg v360 filter (best quality, slower)\n"
                      "None (Crop Only): skip unwarping, just crop")
            unwarp_disp = ["Auto (GPU)", "CPU (v360)", "None (Crop Only)"]
            unwarp_val = ["auto", "v360", "none"]
        current_unwarp = getattr(processor, 'vr_unwarp_method_override', 'auto')
        # Map advanced values to simple list when not in advanced mode
        if not show_advanced and current_unwarp in ("metal", "opengl"):
            current_unwarp = "auto"
        try:
            current_unwarp_idx = unwarp_val.index(current_unwarp)
        except ValueError:
            current_unwarp_idx = 0
        imgui.push_item_width(-1)
        changed, new_idx = imgui.combo("##vrUnwarp", current_unwarp_idx, unwarp_disp)
        imgui.pop_item_width()
        if changed:
            processor.vr_unwarp_method_override = unwarp_val[new_idx]
            self.app.app_settings.set("vr_unwarp_method", unwarp_val[new_idx])
            threading.Thread(target=processor.reapply_video_settings, daemon=True, name='UnwarpReapply').start()
        row_end()

        # Crop Panel (conditional — only when unwarp=none and stereo)
        vr_fmt = getattr(processor, 'vr_input_format', '')
        is_sbs = '_sbs' in vr_fmt or '_lr' in vr_fmt or '_rl' in vr_fmt
        is_tb = '_tb' in vr_fmt
        if current_unwarp == 'none' and (is_sbs or is_tb):
            panel_label = "left/right" if is_sbs else "top/bottom"
            row_label("Crop Panel", f"Select which {panel_label} panel to crop from the stereo frame.")
            panel_disp = ["Left", "Right"] if is_sbs else ["Top", "Bottom"]
            panel_val = ["first", "second"]
            current_panel = self.app.app_settings.get("vr_crop_panel", "first")
            try:
                current_panel_idx = panel_val.index(current_panel)
            except ValueError:
                current_panel_idx = 0
            imgui.push_item_width(-1)
            changed, new_idx = imgui.combo("##vrCropPanel", current_panel_idx, panel_disp)
            imgui.pop_item_width()
            if changed:
                self.app.app_settings.set("vr_crop_panel", panel_val[new_idx])
                processor.vr_crop_panel = panel_val[new_idx]
                threading.Thread(target=processor.reapply_video_settings, daemon=True, name='CropPanelReapply').start()
            row_end()

        # View Pitch (hidden when unwarp=none)
        if current_unwarp != 'none':
            row_label("View Pitch", "Vertical viewing angle offset for VR projection.\nDrag to adjust, release to apply.")

            # Track slider dragging state
            is_slider_hovered = imgui.is_item_hovered()
            is_mouse_down = imgui.is_mouse_down(0)
            is_mouse_released = imgui.is_mouse_released(0)

            if is_slider_hovered and is_mouse_down:
                self.pitch_slider_is_dragging = True
                self.pitch_slider_was_dragging = True
            elif (is_mouse_released or not is_mouse_down) and self.pitch_slider_was_dragging:
                self._handle_mouse_release()

            imgui.push_item_width(-1)
            changed_pitch, new_pitch = imgui.slider_int(
                "##vrPitch", processor.vr_pitch, PITCH_SLIDER_MIN, PITCH_SLIDER_MAX, format="%d deg"
            )
            imgui.pop_item_width()
            if changed_pitch:
                processor.vr_pitch = new_pitch
                self.last_pitch_value = new_pitch
                self._schedule_video_render(new_pitch)
            row_end()

        end_settings_columns()
