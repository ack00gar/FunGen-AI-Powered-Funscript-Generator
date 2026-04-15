"""Subtitle Translation tab UI mixin for ControlPanelUI.

Panel tab shows: status, settings, generate/import/export actions, progress.
Editing (subtitle list, waveform, context menus) lives in the floating editor window.
"""
import imgui
from application.utils.imgui_helpers import center_next_window_pivot
import logging
import math
import time

from application.utils import primary_button_style, destructive_button_style
from application.utils.imgui_helpers import DisabledScope as _DisabledScope
from application.utils.section_card import section_card

_logger = logging.getLogger(__name__)


class SubtitleMixin:
    """Mixin providing Subtitle Translation sidebar tab."""

    def _ensure_subtitle_tool(self):
        """Lazily create the SubtitleToolUI instance. Returns True if ready."""
        if not hasattr(self, '_subtitle_tool'):
            self._subtitle_tool = None
            self._subtitle_init_error = None

        if self._subtitle_tool is not None:
            return True

        if self._subtitle_init_error is not None:
            return False

        try:
            from application.gui_components.subtitle_tool_ui import SubtitleToolUI
            self._subtitle_tool = SubtitleToolUI(self.app)
            self._subtitle_tool.is_open = False  # Floating window starts closed
            # Sync any existing subtitle track
            if getattr(self.app, 'subtitle_track', None):
                self._subtitle_tool.track = self.app.subtitle_track
                if len(self.app.subtitle_track) > 0:
                    self._subtitle_tool.state = self._subtitle_tool.STATE_EDITING
            return True
        except Exception as e:
            _logger.error(f"Subtitle tool init failed: {e}", exc_info=True)
            self._subtitle_init_error = e
            return False

    # ------------------------------------------------------------------ #
    #  Main tab render
    # ------------------------------------------------------------------ #

    def _render_subtitle_tab(self):
        """Render the subtitle translation control panel."""
        if not self._ensure_subtitle_tool():
            self._render_subtitle_error()
            return

        self._render_addon_version_label("subtitle_translation", "Subtitle Translation")
        tool = self._subtitle_tool

        # Poll generation thread (same logic as tool's render_content)
        if tool._gen_thread and not tool._gen_thread.is_alive():
            tool._gen_thread = None
            if tool._gen_error:
                self.app.notify(f"Failed: {tool._gen_error}", "error", 5.0)
                tool.state = tool.STATE_EMPTY
                tool._gen_error = None
            elif tool.track and len(tool.track) > 0:
                tool.state = tool.STATE_EDITING
                self.app.subtitle_track = tool.track
                self.app.notify(f"{len(tool.track)} subtitles generated", "success", 3.0)
            else:
                tool.state = tool.STATE_EMPTY
                self.app.notify("No speech detected", "info", 3.0)

        if tool.state == tool.STATE_GENERATING:
            self._render_subtitle_progress(tool)
        elif tool.state == tool.STATE_EDITING:
            self._render_subtitle_status(tool)
        else:
            self._render_subtitle_actions(tool)

    # ------------------------------------------------------------------ #
    #  Empty / Actions
    # ------------------------------------------------------------------ #

    def _render_subtitle_actions(self, tool):
        """Render generate/import controls when no subtitles are loaded."""
        video_ok = self.app.processor and self.app.processor.is_video_open()

        with section_card("Generate##SubGen", tier="primary") as o:
            if o:
                # Language + quality settings
                imgui.text("Language:")
                imgui.same_line(spacing=8)
                imgui.push_item_width(-1)
                _, tool._lang_idx = imgui.combo("##sub_lang", tool._lang_idx,
                    ["Auto", "Japanese", "Spanish", "English"])
                imgui.pop_item_width()

                imgui.text("Quality:")
                imgui.same_line(spacing=14)
                imgui.push_item_width(-1)
                _, tool._quality_idx = imgui.combo("##sub_qual", tool._quality_idx,
                    tool._quality_labels)
                imgui.pop_item_width()

                imgui.spacing()
                with _DisabledScope(not video_ok):
                    with primary_button_style():
                        if imgui.button("Generate Subtitles##sg", width=-1, height=30):
                            tool._start_generate()

                if not video_ok:
                    imgui.text_disabled("Open a video first")

        imgui.spacing()
        with section_card("Import##SubImp", tier="secondary") as o:
            if o:
                if imgui.button("Import SRT##si", width=-1, height=26):
                    tool._do_import_dialog()

    # ------------------------------------------------------------------ #
    #  Generating / Progress
    # ------------------------------------------------------------------ #

    def _render_subtitle_progress(self, tool):
        """Render generation progress with step indicators."""
        steps = [
            (0.00, 0.10, "Extracting audio"),
            (0.10, 0.25, "Voice detection"),
            (0.25, 0.70, "Transcribing"),
            (0.70, 0.95, "Translating"),
            (0.95, 1.00, "Finishing"),
        ]
        p = tool.progress

        with section_card("Generating##SubGenProg", tier="secondary") as o:
            if o:
                for start, end, name in steps:
                    if p >= end:
                        imgui.text_colored(f"  {name}", 0.3, 0.8, 0.3, 1.0)
                    elif p >= start:
                        step_p = (p - start) / (end - start)
                        imgui.text(f"  {name}...")
                        imgui.same_line()
                        w = imgui.get_content_region_available()[0]
                        imgui.push_item_width(w)
                        if step_p < 0.02:
                            pulse = (math.sin(time.monotonic() * 3) + 1) * 0.3 + 0.1
                            imgui.progress_bar(pulse, (0, 12), "")
                        else:
                            imgui.progress_bar(step_p, (0, 12), f"{int(step_p*100)}%")
                        imgui.pop_item_width()
                    else:
                        imgui.text_disabled(f"  {name}")

                imgui.spacing()
                imgui.progress_bar(p, (-1, 4), "")
                imgui.spacing()
                if tool.progress_text:
                    imgui.text_disabled(tool.progress_text)
                imgui.spacing()
                with destructive_button_style():
                    if imgui.button("Cancel##sc", width=-1, height=26):
                        tool.state = tool.STATE_EMPTY

    # ------------------------------------------------------------------ #
    #  Editing / Status
    # ------------------------------------------------------------------ #

    def _render_subtitle_status(self, tool):
        """Render subtitle status summary and action buttons."""
        track = tool.track
        if not track:
            tool.state = tool.STATE_EMPTY
            return

        # Status summary
        with section_card("Subtitles##SubStatus", tier="primary") as o:
            if o:
                speech_count = sum(1 for s in track if s.is_speech)
                translated_count = sum(1 for s in track if s.text_translated)
                if track.segments:
                    last_ms = max(s.end_ms for s in track if s.is_speech)
                    dur_min = last_ms / 60000 if last_ms > 0 else 1
                    density = speech_count / dur_min
                else:
                    dur_min = 0
                    density = 0

                imgui.text(f"{speech_count} subtitles")
                imgui.same_line(spacing=8)
                imgui.text_disabled(f"({density:.1f}/min)")
                if translated_count < speech_count:
                    imgui.text_colored(f"{translated_count}/{speech_count} translated",
                                       1.0, 0.85, 0.2, 1.0)
                else:
                    imgui.text_colored(f"All translated", 0.3, 0.8, 0.3, 1.0)

                if track.source_language:
                    imgui.text_disabled(f"{track.source_language} \u2192 {track.target_language or 'en'}")

        imgui.spacing()

        # Open editor button
        with section_card("Editor##SubEdit", tier="secondary") as o:
            if o:
                editor_open = tool.is_open
                label = "Close Editor" if editor_open else "Open Editor"
                with primary_button_style():
                    if imgui.button(f"{label}##sub_editor", width=-1, height=28):
                        tool.is_open = not tool.is_open

        imgui.spacing()

        # Export / actions
        with section_card("Actions##SubActions", tier="secondary") as o:
            if o:
                bw = imgui.get_content_region_available_width()
                half = (bw - 4) / 2
                if imgui.button("Export SRT##se", width=half, height=26):
                    tool._do_export()
                imgui.same_line(spacing=4)
                if imgui.button("Bilingual##seb", width=half, height=26):
                    tool._do_export(bilingual=True)

                imgui.spacing()
                if imgui.button("Import SRT##si2", width=half, height=26):
                    tool._do_import_dialog()
                imgui.same_line(spacing=4)
                if imgui.button("Regenerate##sregen", width=half, height=26):
                    tool._start_generate()

                imgui.spacing()
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.5, 0.12, 0.12, 1.0)
                if imgui.button("Clear All##subclear", width=-1, height=26):
                    imgui.open_popup("Clear Subtitles?##SubClearCP")
                imgui.pop_style_color()

                # Clear confirmation
                center_next_window_pivot()
                if imgui.begin_popup_modal("Clear Subtitles?##SubClearCP",
                                            flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
                    imgui.text("Remove all subtitles?")
                    imgui.spacing()
                    with destructive_button_style():
                        if imgui.button("Yes, clear##subclrcp", width=120):
                            tool.track = None
                            self.app.subtitle_track = None
                            tool._selected = -1
                            tool.state = tool.STATE_EMPTY
                            tool.is_open = False
                            self.app.notify("Subtitles cleared", "info", 2.0)
                            imgui.close_current_popup()
                    imgui.same_line()
                    if imgui.button("Cancel##subclrcp", width=80):
                        imgui.close_current_popup()
                    imgui.end_popup()

    # ------------------------------------------------------------------ #
    #  Error / Preview
    # ------------------------------------------------------------------ #

    def _render_subtitle_error(self):
        """Render subtitle initialization error with retry."""
        err = getattr(self, '_subtitle_init_error', None)
        if err is not None:
            imgui.text_colored(f"Failed: {err}", 1.0, 0.4, 0.4, 1.0)
            if isinstance(err, (ImportError, ModuleNotFoundError)):
                imgui.spacing()
                imgui.text_colored("Subtitle addon not found.", 1.0, 0.85, 0.2, 1.0)
            else:
                imgui.text_colored("Check logs.", 1.0, 0.5, 0.0, 1.0)
        else:
            imgui.text("Initializing...")
        imgui.spacing()
        if imgui.button("Retry##subtitle_retry"):
            self._subtitle_init_error = None

    def _render_subtitle_preview(self):
        """Render promo banner when addon is not installed."""
        self._render_addon_promo_banner(
            "Subtitle Translation",
            "Generate subtitles from video audio using local AI models. "
            "Japanese, Spanish, and English with contextual LLM translation. "
            "All processing runs locally \u2014 no API keys required."
        )
        with _DisabledScope(True):
            imgui.text_disabled("Language: Japanese")
            imgui.text_disabled("Quality: Best (7B Uncensored)")
            imgui.spacing()
            imgui.button("Generate Subtitles", width=-1, height=30)
            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            imgui.text_disabled("No subtitles loaded")
