"""First-run wizard — 6-step guided setup shown on initial launch."""

import imgui
import os
import sys
import platform
import threading
import logging
import time

from application.utils.logo_texture import get_logo_texture_manager
from application.utils.system_scaling import get_system_scaling_info

logger = logging.getLogger(__name__)

# Scale options: (label, value)
_SCALE_OPTIONS = [
    ("100%", 1.0),
    ("125%", 1.25),
    ("150%", 1.5),
    ("200%", 2.0),
]


class FirstRunWizard:
    """Full-window 6-step first-run wizard overlay."""

    STEP_WELCOME = 0
    STEP_SCALE = 1
    STEP_MODE = 2
    STEP_OUTPUT = 3
    STEP_MODELS = 4
    STEP_SUPPORT = 5
    NUM_STEPS = 6

    _STEP_TITLES = [
        "Welcome",
        "Display Scale",
        "Interface Mode",
        "Output Folder",
        "AI Models",
        "Support & Finish",
    ]

    def __init__(self, app):
        self.app = app
        self._step = self.STEP_WELCOME

        # Step 2 — scale
        scaling_factor, self._detected_dpi, self._platform_name = get_system_scaling_info()
        self._detected_scale_pct = int(round(scaling_factor * 100))
        # Pre-select closest scale option
        current_scale = app.app_settings.get("global_font_scale", 1.0)
        self._selected_scale_idx = 0
        best_dist = 999
        for i, (_, val) in enumerate(_SCALE_OPTIONS):
            d = abs(val - current_scale)
            if d < best_dist:
                best_dist = d
                self._selected_scale_idx = i

        # Step 3 — mode
        self._selected_mode = app.app_settings.get("ui_view_mode", "simple")

        # Step 4 — output folder
        self._output_folder = app.app_settings.get("output_folder_path", "output")

        # Step 5 — model download
        self._download_thread = None
        self._download_started = False
        self._download_done = False
        self._download_failed = False


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self) -> bool:
        """Render wizard overlay. Returns True when wizard is complete."""
        viewport = imgui.get_main_viewport()
        vw, vh = viewport.size
        vx, vy = viewport.pos

        # Full-window child
        imgui.set_next_window_position(vx, vy)
        imgui.set_next_window_size(vw, vh)
        flags = (
            imgui.WINDOW_NO_TITLE_BAR
            | imgui.WINDOW_NO_RESIZE
            | imgui.WINDOW_NO_MOVE
            | imgui.WINDOW_NO_SCROLLBAR
            | imgui.WINDOW_NO_COLLAPSE
            | imgui.WINDOW_NO_SAVED_SETTINGS
        )
        imgui.begin("##FirstRunWizard", flags=flags)

        # --- Step dots ---
        self._render_step_dots(vw)

        # --- Centered content area ---
        max_content_w = 600
        pad_x = max((vw - max_content_w) * 0.5, 20)
        imgui.set_cursor_pos_x(pad_x)
        imgui.begin_child("##WizardContent", width=max_content_w, height=vh - 120,
                          border=False, flags=imgui.WINDOW_NO_SCROLLBAR)

        if self._step == self.STEP_WELCOME:
            self._render_step_welcome()
        elif self._step == self.STEP_SCALE:
            self._render_step_scale()
        elif self._step == self.STEP_MODE:
            self._render_step_mode()
        elif self._step == self.STEP_OUTPUT:
            self._render_step_output()
        elif self._step == self.STEP_MODELS:
            self._render_step_models()
        elif self._step == self.STEP_SUPPORT:
            self._render_step_support()

        imgui.end_child()

        # --- Nav buttons ---
        done = self._render_nav_buttons(vw, vh)

        imgui.end()
        return done

    # ------------------------------------------------------------------
    # Step dots
    # ------------------------------------------------------------------

    def _render_step_dots(self, window_w):
        draw_list = imgui.get_window_draw_list()
        cx = imgui.get_cursor_screen_pos()[0] + window_w * 0.5
        cy = imgui.get_cursor_screen_pos()[1] + 12
        radius = 5.0
        spacing = 18.0
        total_w = spacing * (self.NUM_STEPS - 1)
        start_x = cx - total_w * 0.5

        for i in range(self.NUM_STEPS):
            x = start_x + i * spacing
            if i == self._step:
                col = imgui.get_color_u32_rgba(0.4, 0.7, 1.0, 1.0)
                draw_list.add_circle_filled(x, cy, radius, col)
            else:
                col = imgui.get_color_u32_rgba(0.5, 0.5, 0.5, 0.6)
                draw_list.add_circle(x, cy, radius, col, 12, 1.5)

        imgui.dummy(0, 30)

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _render_step_welcome(self):
        imgui.dummy(0, 40)
        # Logo
        logo_mgr = get_logo_texture_manager()
        tex_id = logo_mgr.get_texture_id()
        if tex_id is not None:
            lw, lh = logo_mgr.get_dimensions()
            display_h = 100
            display_w = int(lw * (display_h / lh)) if lh > 0 else 100
            avail_w = imgui.get_content_region_available_width()
            imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + (avail_w - display_w) * 0.5)
            imgui.image(tex_id, display_w, display_h)

        imgui.dummy(0, 20)
        self._center_text("Welcome to FunGen", large=True)
        imgui.dummy(0, 10)
        self._center_text("AI-powered funscript generation from video")
        imgui.dummy(0, 20)
        self._center_text("This wizard will help you configure the basics.")

    def _render_step_scale(self):
        imgui.dummy(0, 20)
        self._center_text("Display Scale", large=True)
        imgui.dummy(0, 10)

        imgui.text_wrapped(
            f"Detected: {self._detected_scale_pct}% ({self._detected_dpi:.0f} DPI) on {self._platform_name}"
        )
        imgui.dummy(0, 15)

        avail_w = imgui.get_content_region_available_width()
        card_w = (avail_w - 30) / 4  # 4 cards with gaps

        for i, (label, scale_val) in enumerate(_SCALE_OPTIONS):
            if i > 0:
                imgui.same_line(spacing=10)

            selected = (i == self._selected_scale_idx)

            # Card styling
            if selected:
                imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, 0.2, 0.4, 0.7, 0.8)
                imgui.push_style_color(imgui.COLOR_BORDER, 0.4, 0.7, 1.0, 1.0)
            else:
                imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, 0.15, 0.15, 0.15, 0.8)
                imgui.push_style_color(imgui.COLOR_BORDER, 0.3, 0.3, 0.3, 0.5)

            imgui.push_style_var(imgui.STYLE_CHILD_ROUNDING, 6.0)
            imgui.begin_child(f"##scale_{i}", width=card_w, height=80, border=True)

            imgui.dummy(0, 8)
            # Center the label
            text_w = imgui.calc_text_size(label)[0]
            imgui.set_cursor_pos_x((card_w - text_w) * 0.5)
            imgui.text(label)

            # Sample text at that scale
            sample = "Abc"
            imgui.set_cursor_pos_x(8)
            old_scale = imgui.get_io().font_global_scale
            imgui.get_io().font_global_scale = scale_val
            imgui.text(sample)
            imgui.get_io().font_global_scale = old_scale

            imgui.end_child()
            imgui.pop_style_var()
            imgui.pop_style_color(2)

            # Handle click
            if imgui.is_item_clicked():
                self._selected_scale_idx = i
                # Live preview
                imgui.get_io().font_global_scale = scale_val
                self.app.app_settings.set("global_font_scale", scale_val)

    def _render_step_mode(self):
        imgui.dummy(0, 20)
        self._center_text("Interface Mode", large=True)
        imgui.dummy(0, 15)

        avail_w = imgui.get_content_region_available_width()
        card_w = (avail_w - 15) / 2

        modes = [
            ("Simple", "simple",
             "Streamlined interface, automatic settings.\nBest for getting started."),
            ("Expert", "expert",
             "Full control over all parameters.\nMulti-axis, advanced tracking, plugins."),
        ]

        for i, (title, key, desc) in enumerate(modes):
            if i > 0:
                imgui.same_line(spacing=15)

            selected = (self._selected_mode == key)
            if selected:
                imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, 0.2, 0.4, 0.7, 0.8)
                imgui.push_style_color(imgui.COLOR_BORDER, 0.4, 0.7, 1.0, 1.0)
            else:
                imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, 0.15, 0.15, 0.15, 0.8)
                imgui.push_style_color(imgui.COLOR_BORDER, 0.3, 0.3, 0.3, 0.5)

            imgui.push_style_var(imgui.STYLE_CHILD_ROUNDING, 6.0)
            imgui.begin_child(f"##mode_{key}", width=card_w, height=120, border=True)

            imgui.dummy(0, 12)
            text_w = imgui.calc_text_size(title)[0]
            imgui.set_cursor_pos_x((card_w - text_w) * 0.5)
            imgui.text(title)
            imgui.dummy(0, 6)
            imgui.set_cursor_pos_x(12)
            imgui.push_text_wrap_pos(card_w - 12)
            imgui.text_wrapped(desc)
            imgui.pop_text_wrap_pos()

            imgui.end_child()
            imgui.pop_style_var()
            imgui.pop_style_color(2)

            if imgui.is_item_clicked():
                self._selected_mode = key
                self.app.app_settings.set("ui_view_mode", key)

    def _render_step_output(self):
        imgui.dummy(0, 20)
        self._center_text("Output Folder", large=True)
        imgui.dummy(0, 15)

        imgui.text_wrapped("Exported funscripts and project files will be saved here.")
        imgui.dummy(0, 10)

        imgui.text("Current path:")
        imgui.same_line()
        imgui.text_colored(self._output_folder, 0.6, 0.8, 1.0, 1.0)

        imgui.dummy(0, 10)
        if imgui.button("Browse...", width=120):
            self._open_folder_dialog()

    def _render_step_models(self):
        imgui.dummy(0, 20)
        self._center_text("AI Models", large=True)
        imgui.dummy(0, 15)

        imgui.text_wrapped(
            "FunGen needs AI models for pose detection (~50 MB). "
            "They will be downloaded automatically."
        )
        imgui.dummy(0, 8)

        # Platform info
        is_mac_arm = platform.system() == "Darwin" and platform.machine() == "arm64"
        if is_mac_arm:
            opt_note = "CoreML acceleration available"
        elif sys.platform.startswith("win"):
            opt_note = "CUDA/DirectML acceleration if GPU available"
        else:
            opt_note = "CPU inference (GPU optional)"
        imgui.text(f"Your system: {self._platform_name} — {opt_note}")

        imgui.dummy(0, 15)

        # Auto-start download on entering this step
        if not self._download_started:
            self._download_started = True
            self._start_model_download()

        # Progress
        progress = self.app.first_run_progress / 100.0
        status = self.app.first_run_status_message or "Preparing..."
        imgui.text(status)
        imgui.progress_bar(progress, size=(-1, 0), overlay=f"{self.app.first_run_progress:.0f}%")

        # Check completion
        status_lower = status.lower()
        if "complete" in status_lower:
            self._download_done = True
            self._download_failed = False
        elif "failed" in status_lower or "error" in status_lower:
            self._download_done = True
            self._download_failed = True

        if self._download_failed:
            imgui.dummy(0, 8)
            imgui.push_style_color(imgui.COLOR_TEXT, 0.9, 0.4, 0.3, 1.0)
            imgui.text_wrapped(
                "Download failed. You can retry later via AI menu > Download Models."
            )
            imgui.pop_style_color()

    def _render_step_support(self):
        imgui.dummy(0, 20)
        self._center_text("FunGen is free and open source", large=True)
        imgui.dummy(0, 15)

        imgui.text_wrapped(
            "Unlock premium add-ons with a one-time or monthly Ko-fi purchase:"
        )
        imgui.dummy(0, 4)
        imgui.bullet_text("Device Control (real-time hardware sync)")
        imgui.bullet_text("Streaming (stream video to browser with funscript support)")
        imgui.bullet_text("Patreon Features (monthly Ko-fi subscription)")
        imgui.dummy(0, 10)
        imgui.text("ko-fi.com/k00gar")

    # ------------------------------------------------------------------
    # Nav buttons
    # ------------------------------------------------------------------

    def _render_nav_buttons(self, window_w, window_h) -> bool:
        """Render Back/Next/Get Started. Returns True when wizard is done."""
        done = False
        btn_w = 120
        btn_h = 30

        # Position at bottom
        imgui.set_cursor_pos_y(window_h - btn_h - 20)

        if self._step == self.STEP_WELCOME:
            # Only Next
            imgui.set_cursor_pos_x(window_w - btn_w - 30)
            if imgui.button("Next >", width=btn_w, height=btn_h):
                self._step += 1

        elif self._step == self.STEP_SUPPORT:
            # Back + Get Started
            imgui.set_cursor_pos_x(30)
            if imgui.button("< Back", width=btn_w, height=btn_h):
                self._step -= 1
            imgui.same_line(spacing=window_w - 2 * btn_w - 80)
            if imgui.button("Get Started", width=btn_w, height=btn_h):
                self._finish()
                done = True

        elif self._step == self.STEP_MODELS:
            # Back (disabled during download) + Next (enabled after done)
            downloading = self._download_started and not self._download_done
            imgui.set_cursor_pos_x(30)
            if downloading:
                imgui.push_style_var(imgui.STYLE_ALPHA, 0.4)
                imgui.button("< Back", width=btn_w, height=btn_h)
                imgui.pop_style_var()
            else:
                if imgui.button("< Back", width=btn_w, height=btn_h):
                    self._step -= 1

            imgui.same_line(spacing=window_w - 2 * btn_w - 80)
            if self._download_done:
                if imgui.button("Next >", width=btn_w, height=btn_h):
                    self._step += 1
            else:
                imgui.push_style_var(imgui.STYLE_ALPHA, 0.4)
                imgui.button("Next >", width=btn_w, height=btn_h)
                imgui.pop_style_var()
        else:
            # Back + Next
            imgui.set_cursor_pos_x(30)
            if imgui.button("< Back", width=btn_w, height=btn_h):
                self._step -= 1
            imgui.same_line(spacing=window_w - 2 * btn_w - 80)
            if imgui.button("Next >", width=btn_w, height=btn_h):
                self._step += 1

        return done

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _center_text(self, text, large=False):
        """Render centered text. If large, just uses default font (no scaling)."""
        avail_w = imgui.get_content_region_available_width()
        text_w = imgui.calc_text_size(text)[0]
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + (avail_w - text_w) * 0.5)
        imgui.text(text)

    def _open_folder_dialog(self):
        """Open native folder dialog via the app's ImGuiFileDialog."""
        try:
            fd = self.app.gui.file_dialog if hasattr(self.app, "gui") else None
            if fd is not None:
                fd.open(
                    title="Select Output Folder",
                    default_path=self._output_folder,
                    is_folder_dialog=True,
                    callback=self._on_folder_selected,
                )
            else:
                logger.warning("File dialog not available during wizard")
        except Exception as e:
            logger.warning(f"Could not open folder dialog: {e}")

    def _on_folder_selected(self, path):
        if path:
            self._output_folder = path
            self.app.app_settings.set("output_folder_path", path)

    def _start_model_download(self):
        """Kick off model download using existing app_logic method."""
        self.app.trigger_first_run_setup()

    def _finish(self):
        """Mark first run complete, persist all settings."""
        self.app.app_settings.set("is_first_run_complete", True)
        self.app.app_settings.is_first_run = False
        self.app.app_settings.save_settings()
        logger.info("First-run wizard completed")

