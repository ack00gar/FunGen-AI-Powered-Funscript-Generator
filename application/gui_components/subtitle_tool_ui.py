"""
Subtitle Tool Window — subtitle list editor with batch operations.

Timing visualization is in the dedicated InteractiveSubtitleTimeline (bottom panel).
This window handles: generation, import/export, text editing, batch ops.

Layout:
  [Toolbar: Export | Import | Regen | Clear]
  [Data grid: # Start | End | Dur | Translation — double-click to edit]
"""

import imgui
from application.utils.imgui_helpers import center_next_window_pivot
import logging
import os
import queue
import threading
from typing import Optional

from application.utils import primary_button_style, destructive_button_style
from application.utils.imgui_helpers import DisabledScope as _DisabledScope
from application.utils.section_card import section_card
from common.frame_utils import frame_to_ms

logger = logging.getLogger(__name__)


class SubtitleToolUI:
    STATE_EMPTY = "empty"
    STATE_GENERATING = "generating"
    STATE_EDITING = "editing"

    def __init__(self, app):
        self.app = app
        self.is_open = True
        self.state = self.STATE_EMPTY
        self.track = None

        # Generation (thread-safe: _progress_q is the only cross-thread channel)
        self.progress = 0.0
        self.progress_text = ""
        self._gen_thread = None
        self._gen_error = None
        self._cancel_requested = False
        self._progress_q = queue.Queue()

        # Editor
        self._selected = -1
        self._editing = -1
        self._edit_field = None    # 'start', 'end', 'text', or None
        self._edit_buf = ""
        self._last_playing = -1
        self._scroll_needed = False

        # Multi-selection (set of indices, _selected is the anchor/primary)
        self._multi_sel = set()

        # Shift-all dialog
        self._shift_dialog_open = False
        self._shift_ms_buf = "0"

        # Export dialog
        self._export_dialog_open = False
        self._export_format_idx = 0   # 0=English, 1=Original, 2=Bilingual
        self._export_enforce_std = False

        # Settings (stored in UI, applied on generate)
        self._lang_idx = 0  # 0=Auto, 1=ja, 2=es, 3=en
        self._lang_codes = [None, "ja", "es", "en"]
        self._quality_idx = 1  # 0=medium, 1=best
        self._quality_labels = ["Good (3B LLM, ~2GB)", "Best (7B Uncensored, ~4GB)"]
        self._quality_map = [
            "medium",
            "large",
        ]

    def render(self):
        """Render as a floating tool window."""
        if not self.is_open:
            return

        imgui.set_next_window_size(720, 560, imgui.FIRST_USE_EVER)
        expanded, opened = imgui.begin("Subtitles##SubTool", closable=True,
                                        flags=imgui.WINDOW_NO_COLLAPSE)
        if not opened:
            self.is_open = False
            imgui.end()
            return
        if not expanded:
            imgui.end()
            return

        self.render_content()
        imgui.end()

    def render_content(self):
        """Render subtitle UI content (usable in both floating window and embedded panel)."""
        # Drain progress queue (thread-safe: only main thread reads these)
        while not self._progress_q.empty():
            try:
                self.progress, self.progress_text = self._progress_q.get_nowait()
            except queue.Empty:
                break

        if self.state == self.STATE_EMPTY:
            self._render_empty()
        elif self.state == self.STATE_GENERATING:
            # Minimal progress in floating window — sidebar shows full progress
            imgui.text(f"Generating... {int(self.progress * 100)}%")
            if self.progress_text:
                imgui.text_disabled(self.progress_text)
        elif self.state == self.STATE_EDITING:
            self._render_editor()

        # Check generation thread completion (safe: thread ref only set/cleared from main thread)
        t = self._gen_thread
        if t is not None and not t.is_alive():
            self._gen_thread = None
            if self._gen_error:
                self.app.notify(f"Failed: {self._gen_error}", "error", 5.0)
                self.state = self.STATE_EMPTY
                self._gen_error = None
            elif self.track and len(self.track) > 0:
                self.state = self.STATE_EDITING
                self.app.subtitle_track = self.track
                # Auto-show the subtitle timeline at the bottom
                if hasattr(self.app, 'app_state_ui'):
                    self.app.app_state_ui.show_subtitle_timeline = True
                self.app.notify(f"{len(self.track)} subtitles generated", "success", 3.0)
            else:
                self.state = self.STATE_EMPTY
                self.app.notify("No speech detected", "info", 3.0)

    # ==============================================================
    #  Empty State - clean two-action screen
    # ==============================================================

    def _render_empty(self):
        video_ok = self.app.processor and self.app.processor.is_video_open()

        with section_card("Subtitles##SubMain", tier="primary") as o:
            if o:
                with primary_button_style():
                    if imgui.button("Import SRT##si", width=-1, height=30):
                        self._do_import_dialog()

                imgui.spacing()
                imgui.separator()
                imgui.spacing()

                # Settings row
                imgui.push_item_width(100)
                _, self._lang_idx = imgui.combo("##lang", self._lang_idx,
                    ["Auto", "Japanese", "Spanish", "English"])
                imgui.pop_item_width()
                imgui.same_line()
                imgui.push_item_width(170)
                _, self._quality_idx = imgui.combo("##qual", self._quality_idx,
                    self._quality_labels)
                imgui.pop_item_width()

                imgui.spacing()
                with _DisabledScope(not video_ok):
                    with primary_button_style():
                        if imgui.button("Generate Subtitles##sg", width=-1, height=30):
                            self._start_generate()
                    if not video_ok and imgui.is_item_hovered(imgui.HOVERED_FLAGS_ALLOW_WHEN_DISABLED):
                        imgui.set_tooltip("Open a video first")

    # ==============================================================
    #  Generating State - step progress
    # ==============================================================

    def _render_generating(self):
        steps = [
            (0.00, 0.10, "Extracting audio"),
            (0.10, 0.25, "Voice detection"),
            (0.25, 0.70, "Transcribing"),
            (0.70, 0.95, "Translating"),
            (0.95, 1.00, "Finishing"),
        ]
        p = self.progress

        with section_card("Generating##SubGen", tier="secondary") as o:
            if o:
                for start, end, name in steps:
                    if p >= end:
                        imgui.text_colored(f"  \u2713 {name}", 0.3, 0.8, 0.3, 1.0)
                    elif p >= start:
                        step_p = (p - start) / (end - start)
                        pct = int(step_p * 100)
                        imgui.text(f"  {name}... {pct}%")
                        imgui.same_line()
                        w = imgui.get_content_region_available()[0]
                        if w > 20:
                            imgui.push_item_width(w)
                            imgui.progress_bar(step_p, (0, 6), "")
                            imgui.pop_item_width()
                    else:
                        imgui.text_disabled(f"  {name}")

                imgui.spacing()
                imgui.progress_bar(p, (-1, 4), "")
                imgui.spacing()
                if self.progress_text:
                    imgui.text(self.progress_text)
                imgui.spacing()
                with destructive_button_style():
                    if imgui.button("Cancel##sc", width=-1, height=26):
                        self._cancel_requested = True
                        self.state = self.STATE_EMPTY

    # ==============================================================
    #  Editor - waveform + list, minimal chrome
    # ==============================================================

    def _render_editor(self):
        if not self.track:
            self.state = self.STATE_EMPTY
            return

        # Compact toolbar
        self._render_toolbar()

        # Subtitle list (fills remaining space)
        avail = imgui.get_content_region_available()
        imgui.begin_child("##slist", width=0, height=max(80, avail[1]), border=False)
        self._render_list()

        # Keyboard — handle inside the list child window for proper focus detection
        io = imgui.get_io()
        if (imgui.is_window_focused(imgui.FOCUS_ROOT_AND_CHILD_WINDOWS)
                and not io.want_text_input
                and not imgui.is_any_item_active()
                and self._editing < 0):
            self._handle_keys()
        imgui.end_child()

    def _render_toolbar(self):
        bw = 80
        if imgui.button("Export...", width=bw):
            self._export_dialog_open = True
        imgui.same_line()
        if imgui.button("Import", width=bw):
            self._do_import_dialog()
        imgui.same_line()
        if imgui.button("Regen", width=bw):
            self._start_generate()
        imgui.same_line()
        imgui.push_style_color(imgui.COLOR_BUTTON, 0.5, 0.12, 0.12, 1.0)
        if imgui.button("Clear", width=bw):
            imgui.open_popup("Clear Subtitles?##SubClearConfirm")
        imgui.pop_style_color()

        # Clear confirmation popup
        center_next_window_pivot()
        if imgui.begin_popup_modal("Clear Subtitles?##SubClearConfirm", flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
            imgui.text("Remove all subtitles? This cannot be undone.")
            imgui.spacing()
            with destructive_button_style():
                if imgui.button("Yes, clear all##subclr", width=120):
                    self.track = None
                    self.app.subtitle_track = None
                    self._selected = -1
                    self.state = self.STATE_EMPTY
                    self.app.notify("Subtitles cleared", "info", 2.0)
                    imgui.close_current_popup()
            imgui.same_line()
            if imgui.button("Cancel##subclr", width=80):
                imgui.close_current_popup()
            imgui.end_popup()
        imgui.same_line(spacing=12)
        # Density info
        if self.track and len(self.track) > 0:
            speech_segs = [s for s in self.track if s.is_speech]
            if speech_segs:
                last_ms = max(s.end_ms for s in speech_segs)
                dur_min = last_ms / 60000 if last_ms > 0 else 1
                density = len(self.track) / dur_min
                imgui.text_disabled(f"{len(self.track)} subs ({density:.0f}/min)")
            else:
                imgui.text_disabled(f"{len(self.track)} subs")

        # Second row: selection info + actions
        n_sel = len(self._multi_sel)
        if n_sel > 1:
            # Multi-selection actions
            imgui.text_disabled(f"  {n_sel} selected")
            imgui.same_line(spacing=8)
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.5, 0.12, 0.12, 1.0)
            if imgui.small_button("Delete selected"):
                self._delete_selected()
            imgui.pop_style_color()
            imgui.same_line(spacing=6)
            if imgui.small_button("Re-translate selected"):
                self._retranslate_selected()
            imgui.same_line(spacing=6)
            if imgui.small_button("Shift All..."):
                self._shift_dialog_open = True
                self._shift_ms_buf = "0"
            imgui.same_line(spacing=8)
            imgui.text_disabled("Ctrl+Click: toggle | Shift+Click: range | Esc: deselect")
        elif 0 <= self._selected < len(self.track):
            seg = self.track[self._selected]
            imgui.text_disabled(f"  #{seg.index+1}  {_t(seg.start_ms)} -> {_t(seg.end_ms)}  ({seg.duration_ms/1000:.1f}s)")
            imgui.same_line(spacing=12)
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.2, 0.4, 0.6, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.25, 0.5, 0.7, 1.0)
            if imgui.small_button("Snap (N)"):
                self._snap_nearest_to_playhead()
            imgui.pop_style_color(2)
            imgui.same_line(spacing=4)
            if imgui.small_button("Shift All..."):
                self._shift_dialog_open = True
                self._shift_ms_buf = "0"
            imgui.same_line(spacing=8)
            imgui.text_disabled("Arrows: nudge | N: snap | S/M/A: split/merge/add | Del | Enter: edit | Dbl-click: edit timing")
        else:
            imgui.text_disabled("  Click a subtitle or Up/Down to navigate | Shift+Click: range select")

        # Dialogs
        if self._shift_dialog_open:
            self._render_shift_dialog()
        if self._export_dialog_open:
            self._render_export_dialog()

    def _render_list(self):
        if not self.track:
            return

        # Current playback position
        cms = -1
        if self.app.processor and self.app.processor.fps > 0:
            cms = int(self.app.processor.current_frame_index / self.app.processor.fps * 1000)

        playing = -1
        for i, seg in enumerate(self.track):
            if seg.is_speech and seg.start_ms <= cms <= seg.end_ms:
                playing = i
                break

        if playing >= 0 and playing != self._last_playing:
            self._last_playing = playing
            self._scroll_needed = True

        # Also scroll when selection changes (e.g., from waveform click or keyboard)
        if not hasattr(self, '_last_selected_scroll'):
            self._last_selected_scroll = -1
        if self._selected >= 0 and self._selected != self._last_selected_scroll:
            self._last_selected_scroll = self._selected
            self._scroll_needed = True

        _cjk_ok = getattr(getattr(self.app, 'gui_instance', None), '_cjk_font_loaded', False)

        # Column headers
        imgui.columns(4, "##subcols", border=True)
        imgui.set_column_width(0, 80)   # Start
        imgui.set_column_width(1, 80)   # End
        imgui.set_column_width(2, 45)   # Dur
        # Col 3 = Translation (fills remaining)
        imgui.text_disabled("#  Start")
        imgui.next_column()
        imgui.text_disabled("End")
        imgui.next_column()
        imgui.text_disabled("Dur")
        imgui.next_column()
        imgui.text_disabled("Translation")
        imgui.next_column()
        imgui.separator()

        for i, seg in enumerate(self.track):
            if not seg.is_speech:
                continue

            is_sel = (i == self._selected or i in self._multi_sel)
            is_play = (i == playing)
            is_edit = (i == self._editing)

            imgui.push_id(f"s{i}")

            # Row highlight (blue for playing, subtle for multi-selected)
            _row_colors = False
            if is_play:
                imgui.push_style_color(imgui.COLOR_HEADER, 0.15, 0.30, 0.55, 0.9)
                imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, 0.20, 0.35, 0.60, 0.9)
                _row_colors = True
            elif i in self._multi_sel and i != self._selected:
                imgui.push_style_color(imgui.COLOR_HEADER, 0.12, 0.18, 0.30, 0.7)
                imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, 0.15, 0.22, 0.35, 0.8)
                _row_colors = True

            # --- Column 1: Start time ---
            if is_edit and self._edit_field == 'start':
                imgui.push_item_width(-1)
                changed, self._edit_buf = imgui.input_text("##es", self._edit_buf, 32,
                    imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
                imgui.pop_item_width()
                if changed:
                    ms = _parse_time(self._edit_buf)
                    if ms is not None and ms < seg.end_ms:
                        self.track._push_undo()
                        seg.start_ms = ms
                    self._editing = -1
                    self._edit_field = None
                if imgui.is_key_pressed(imgui.KEY_ESCAPE):
                    self._editing = -1
                    self._edit_field = None
            else:
                clicked, _ = imgui.selectable(f"{seg.index+1:2d} {_t(seg.start_ms)}",
                    is_sel or is_play, imgui.SELECTABLE_SPAN_ALL_COLUMNS)
                if clicked:
                    io_ref = imgui.get_io()
                    if io_ref.key_shift and self._selected >= 0:
                        lo, hi = sorted([self._selected, i])
                        self._multi_sel = set(range(lo, hi + 1))
                    elif io_ref.key_ctrl or io_ref.key_super:
                        if i in self._multi_sel:
                            self._multi_sel.discard(i)
                        else:
                            self._multi_sel.add(i)
                    else:
                        self._multi_sel.clear()
                    self._selected = i
                    self._seek(seg)
                # Double-click start time → edit timing
                if imgui.is_item_hovered() and imgui.is_mouse_double_clicked(0):
                    self._editing = i
                    self._edit_field = 'start'
                    self._edit_buf = _t(seg.start_ms)

                # Right-click context menu
                if imgui.is_item_hovered() and imgui.is_mouse_clicked(1):
                    self._selected = i
                    imgui.open_popup(f"SubCtx##{i}")

                self._render_context_menu(i, seg)

            imgui.next_column()

            # --- Column 2: End time ---
            if is_edit and self._edit_field == 'end':
                imgui.push_item_width(-1)
                changed, self._edit_buf = imgui.input_text("##ee", self._edit_buf, 32,
                    imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
                imgui.pop_item_width()
                if changed:
                    ms = _parse_time(self._edit_buf)
                    if ms is not None and ms > seg.start_ms:
                        self.track._push_undo()
                        seg.end_ms = ms
                    self._editing = -1
                    self._edit_field = None
                if imgui.is_key_pressed(imgui.KEY_ESCAPE):
                    self._editing = -1
                    self._edit_field = None
            else:
                imgui.text(_t(seg.end_ms))
                if imgui.is_item_hovered() and imgui.is_mouse_double_clicked(0):
                    self._editing = i
                    self._edit_field = 'end'
                    self._edit_buf = _t(seg.end_ms)
            imgui.next_column()

            # --- Column 3: Duration + confidence dot ---
            dur_s = seg.duration_ms / 1000
            cr, cg, ccb = _confidence_rgb(seg.confidence)
            # Confidence dot (filled circle via bullet)
            dl = imgui.get_window_draw_list()
            cx, cy = imgui.get_cursor_screen_pos()
            dl.add_circle_filled(cx + 4, cy + 7, 3, imgui.get_color_u32_rgba(cr, cg, ccb, 0.9))
            imgui.dummy(10, 0)
            imgui.same_line(spacing=0)
            if dur_s < 0.5:
                imgui.text_colored(f"{dur_s:.1f}", 1.0, 0.4, 0.4, 1.0)
            else:
                imgui.text_disabled(f"{dur_s:.1f}")
            imgui.next_column()

            # --- Column 4: Translation text ---
            if is_edit and self._edit_field == 'text':
                imgui.push_item_width(-1)
                changed, self._edit_buf = imgui.input_text("##et", self._edit_buf, 512,
                    imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
                imgui.pop_item_width()
                # Char count + reading speed hint
                n_chars = len(self._edit_buf)
                cps = n_chars / max(0.1, dur_s)  # chars per second
                if cps > 25:
                    imgui.same_line()
                    imgui.text_colored(f"{n_chars}ch {cps:.0f}cps", 1.0, 0.4, 0.2, 1.0)
                elif n_chars > 50:
                    imgui.same_line()
                    imgui.text_colored(f"{n_chars}ch", 0.9, 0.8, 0.3, 1.0)
                if changed:
                    self.track._push_undo()
                    seg.text_translated = self._edit_buf
                    self._editing = -1
                    self._edit_field = None
                if imgui.is_key_pressed(imgui.KEY_ESCAPE):
                    self._editing = -1
                    self._edit_field = None
            else:
                en = seg.text_translated or seg.text_original or ""
                if not _cjk_ok and en and any(ord(c) > 0x3000 for c in en) and not any(c.isascii() and c.isalpha() for c in en):
                    imgui.text_disabled("[CJK font missing]")
                else:
                    imgui.text(en[:80] if en else "---")
                if imgui.is_item_hovered() and imgui.is_mouse_double_clicked(0):
                    self._editing = i
                    self._edit_field = 'text'
                    self._edit_buf = seg.text_translated or seg.text_original or ""
            imgui.next_column()

            if _row_colors:
                imgui.pop_style_color(2)

            # Auto-scroll
            if (is_play or is_sel) and self._scroll_needed:
                imgui.set_scroll_here_y(0.3)
                self._scroll_needed = False

            imgui.pop_id()

        imgui.columns(1)

    def _render_context_menu(self, i, seg):
        """Right-click context menu for subtitle list rows."""
        if imgui.begin_popup(f"SubCtx##{i}"):
            if imgui.menu_item("Edit text (Enter)")[0]:
                self._editing = i
                self._edit_field = 'text'
                self._edit_buf = seg.text_translated or seg.text_original or ""
            if imgui.menu_item("Snap to playhead (N)")[0]:
                self._snap_nearest_to_playhead()
            if imgui.menu_item("Split at playhead (Shift+S)")[0]:
                if self.app.processor and self.app.processor.fps > 0:
                    ph = int(self.app.processor.current_frame_index / self.app.processor.fps * 1000)
                    if seg.start_ms < ph < seg.end_ms:
                        self.track.split_segment(i, ph)
                        self.app.notify("Split subtitle", "info", 1.5)
            if i < len(self.track) - 1:
                if imgui.menu_item("Merge with next (Shift+M)")[0]:
                    self.track.merge_segments(i, i + 1)
                    self.app.notify("Merged subtitles", "info", 1.5)
            imgui.separator()
            if imgui.menu_item("Add subtitle here")[0]:
                ph_ms = 0
                if self.app.processor and self.app.processor.fps > 0:
                    ph_ms = int(self.app.processor.current_frame_index / self.app.processor.fps * 1000)
                self.track.add_segment(ph_ms, ph_ms + 2000, text_translated="(new subtitle)")
                self.app.notify("Added subtitle", "info", 1.5)
            imgui.separator()
            _t_ref = self._gen_thread
            re_disabled = (_t_ref is not None and _t_ref.is_alive()) or not seg.text_original
            if re_disabled:
                imgui.push_style_var(imgui.STYLE_ALPHA, 0.4)
            if imgui.menu_item("Re-translate")[0] and not re_disabled:
                self._retranslate_segment(i)
            if re_disabled:
                imgui.pop_style_var()
            imgui.separator()
            imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 0.3, 0.3, 1.0)
            if imgui.menu_item("Delete (Del)")[0]:
                self.track.remove_segment(seg.index)
                self._selected = max(-1, min(self._selected, len(self.track) - 1))
                self.app.notify("Deleted subtitle", "info", 1.5)
            imgui.pop_style_color()
            imgui.end_popup()

    # ==============================================================
    #  Keyboard
    # ==============================================================

    def _handle_keys(self):
        if self._editing >= 0 and self._edit_field is not None:
            return  # Input field handles keys

        io = imgui.get_io()
        ctrl = io.key_ctrl or io.key_super
        alt = io.key_alt

        # Escape = clear multi-selection
        if imgui.is_key_pressed(imgui.KEY_ESCAPE):
            if self._multi_sel:
                self._multi_sel.clear()
                return

        # Undo/Redo (Ctrl+Z / Ctrl+Shift+Z)
        if ctrl and imgui.is_key_pressed(ord('Z')):
            if self.track:
                (self.track.redo if io.key_shift else self.track.undo)()
            return

        # Select all (Ctrl+A)
        if ctrl and imgui.is_key_pressed(ord('A')) and self.track:
            self._multi_sel = set(range(len(self.track)))
            return

        if not self.track or self._selected < 0 or self._selected >= len(self.track):
            # No selection: only navigation
            if self.track and imgui.is_key_pressed(imgui.KEY_DOWN_ARROW) and len(self.track) > 0:
                self._selected = 0
                self._multi_sel.clear()
                self._seek(self.track[0])
            return

        seg = self.track[self._selected]

        # Enter = edit selected subtitle text
        if imgui.is_key_pressed(imgui.KEY_ENTER):
            self._editing = self._selected
            self._edit_field = 'text'
            self._edit_buf = seg.text_translated or seg.text_original or ""
            return

        # Navigation (Up/Down) — clears multi-sel unless Shift held
        if imgui.is_key_pressed(imgui.KEY_UP_ARROW) and not io.key_shift:
            self._selected = max(0, self._selected - 1)
            self._multi_sel.clear()
            self._seek(self.track[self._selected])
        elif imgui.is_key_pressed(imgui.KEY_DOWN_ARROW) and not io.key_shift:
            self._selected = min(len(self.track) - 1, self._selected + 1)
            self._multi_sel.clear()
            self._seek(self.track[self._selected])

        # Timing nudge — Shift = 1 frame (fps-aware), Alt = 100ms
        elif io.key_shift and imgui.is_key_pressed(imgui.KEY_LEFT_ARROW):
            frame_ms = _frame_ms(self.app)
            seg.start_ms = max(0, seg.start_ms - frame_ms)
            seg.end_ms = max(seg.start_ms + 100, seg.end_ms - frame_ms)
        elif io.key_shift and imgui.is_key_pressed(imgui.KEY_RIGHT_ARROW):
            frame_ms = _frame_ms(self.app)
            seg.start_ms += frame_ms
            seg.end_ms += frame_ms
        elif alt and imgui.is_key_pressed(imgui.KEY_LEFT_ARROW):
            seg.start_ms = max(0, seg.start_ms - 100)
            seg.end_ms = max(seg.start_ms + 100, seg.end_ms - 100)
        elif alt and imgui.is_key_pressed(imgui.KEY_RIGHT_ARROW):
            seg.start_ms += 100
            seg.end_ms += 100

        # Subtitle-specific shortcuts (Shift+key to avoid conflicts with global shortcuts)
        # Snap nearest edge to playhead (N - no conflict, not used globally)
        elif imgui.is_key_pressed(ord('N')) and not ctrl and not alt:
            self._snap_nearest_to_playhead()
            self.app.notify("Snapped to playhead", "info", 1.0)
        # Split at playhead (Shift+S)
        elif io.key_shift and imgui.is_key_pressed(ord('S')) and not ctrl:
            if self.app.processor and self.app.processor.fps > 0:
                ph = int(self.app.processor.current_frame_index / self.app.processor.fps * 1000)
                if seg.start_ms < ph < seg.end_ms:
                    self.track.split_segment(self._selected, ph)
                    self.app.notify("Split", "info", 1.0)
        # Merge with next (Shift+M)
        elif io.key_shift and imgui.is_key_pressed(ord('M')) and not ctrl:
            if self._selected < len(self.track) - 1:
                self.track.merge_segments(self._selected, self._selected + 1)
                self.app.notify("Merged", "info", 1.0)
        # Add new subtitle at playhead (Shift+A)
        elif io.key_shift and imgui.is_key_pressed(ord('A')) and not ctrl:
            if self.app.processor and self.app.processor.fps > 0:
                ph = int(self.app.processor.current_frame_index / self.app.processor.fps * 1000)
                self.track.add_segment(ph, ph + 2000, text_translated="(new)")
                self.app.notify("Added subtitle", "info", 1.5)
        # Delete (multi-select aware)
        elif imgui.is_key_pressed(imgui.KEY_DELETE):
            if len(self._multi_sel) > 1:
                self._delete_selected()
            else:
                self.track.remove_segment(seg.index)
                self._selected = max(-1, min(self._selected, len(self.track) - 1))
                self.app.notify("Deleted", "info", 1.0)

    # ==============================================================
    #  Actions
    # ==============================================================

    def _snap_nearest_to_playhead(self):
        """Snap the nearest subtitle edge (start or end) to the current playhead."""
        if not self.track or not self.app.processor or self.app.processor.fps <= 0:
            return

        playhead_ms = int(self.app.processor.current_frame_index / self.app.processor.fps * 1000)

        # Find the nearest edge (start or end) across all segments
        best_seg = None
        best_edge = None  # 'start' or 'end'
        best_dist = float('inf')

        for seg in self.track:
            if not seg.is_speech:
                continue
            for edge, val in [('start', seg.start_ms), ('end', seg.end_ms)]:
                dist = abs(val - playhead_ms)
                if dist < best_dist:
                    best_dist = dist
                    best_seg = seg
                    best_edge = edge

        if best_seg is None or best_dist > 30000:  # Max 30s snap range
            return

        # Push undo before modifying
        self.track._push_undo()

        if best_edge == 'start':
            best_seg.start_ms = playhead_ms
            if best_seg.end_ms <= best_seg.start_ms:
                best_seg.end_ms = best_seg.start_ms + 500
        else:
            best_seg.end_ms = playhead_ms
            if best_seg.end_ms <= best_seg.start_ms:
                best_seg.start_ms = max(0, best_seg.end_ms - 500)

        # Select the snapped segment
        self._selected = best_seg.index

    def _delete_selected(self):
        """Delete all multi-selected subtitles."""
        if not self._multi_sel or not self.track:
            return
        n = len(self._multi_sel)
        self.track._push_undo()
        # Delete by index descending so indices don't shift
        for idx in sorted(self._multi_sel, reverse=True):
            if 0 <= idx < len(self.track):
                self.track.segments.pop(idx)
        self.track._sort_and_reindex()
        self._multi_sel.clear()
        self._selected = max(-1, min(self._selected, len(self.track) - 1))
        self.app.notify(f"Deleted {n} subtitles", "info", 1.5)

    def _retranslate_selected(self):
        """Re-translate all multi-selected subtitles in the background."""
        t = self._gen_thread
        if t is not None and t.is_alive():
            return
        indices = sorted(self._multi_sel)
        if not indices:
            return
        originals = []
        for idx in indices:
            if 0 <= idx < len(self.track):
                seg = self.track[idx]
                if seg.text_original:
                    originals.append((idx, seg.text_original))
        if not originals:
            self.app.notify("No original text to re-translate", "warning", 2.0)
            return

        n = len(originals)
        self.app.notify(f"Re-translating {n} subtitles...", "info", 2.0)

        def _run():
            try:
                translator = self._get_translator()
                texts = [t for _, t in originals]
                results = translator.translate_batch(texts)
                self.track._push_undo()
                changed = 0
                for (idx, _), new_text in zip(originals, results):
                    if new_text and 0 <= idx < len(self.track):
                        self.track[idx].text_translated = new_text
                        changed += 1
                self.app.notify(f"Re-translated {changed} subtitles", "success", 2.0)
            except Exception as e:
                logger.error(f"Batch re-translate failed: {e}", exc_info=True)
                self.app.notify(f"Re-translate failed: {e}", "error", 4.0)

        self._gen_thread = threading.Thread(target=_run, daemon=True, name="SubBatchRetrans")
        self._gen_thread.start()

    def _render_shift_dialog(self):
        """Modal dialog for shifting all subtitle timings by an offset."""
        imgui.open_popup("Shift All Timings##ShiftDlg")
        center_next_window_pivot()
        if imgui.begin_popup_modal("Shift All Timings##ShiftDlg",
                                    flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
            imgui.text("Shift all subtitles by milliseconds:")
            imgui.text_disabled("Positive = later, negative = earlier")
            imgui.spacing()

            imgui.push_item_width(150)
            _, self._shift_ms_buf = imgui.input_text("##shiftms", self._shift_ms_buf, 32)
            imgui.pop_item_width()
            imgui.same_line()
            imgui.text("ms")

            # Quick presets
            imgui.spacing()
            for label, val in [("-2s", -2000), ("-500ms", -500), ("+500ms", 500), ("+2s", 2000)]:
                if imgui.small_button(label):
                    self._shift_ms_buf = str(val)
                imgui.same_line()
            imgui.new_line()

            imgui.spacing()
            if imgui.button("Apply##shift", width=100):
                try:
                    offset = int(self._shift_ms_buf)
                    if offset != 0 and self.track:
                        self.track.shift_all(offset)
                        self.app.notify(f"Shifted all by {offset:+d}ms", "success", 2.0)
                except ValueError:
                    self.app.notify("Invalid offset value", "error", 2.0)
                self._shift_dialog_open = False
                imgui.close_current_popup()
            imgui.same_line()
            if imgui.button("Cancel##shift", width=80):
                self._shift_dialog_open = False
                imgui.close_current_popup()
            imgui.end_popup()
        else:
            self._shift_dialog_open = False

    def _render_export_dialog(self):
        """Export options dialog with format choice, validation, and preview."""
        imgui.open_popup("Export Subtitles##ExportDlg")
        center_next_window_pivot()
        if imgui.begin_popup_modal("Export Subtitles##ExportDlg",
                                    flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
            # Format choice
            imgui.text("Format:")
            formats = ["English (translated)", "Original (source)", "Bilingual (both)"]
            _, self._export_format_idx = imgui.combo("##expfmt", self._export_format_idx, formats)

            imgui.spacing()

            # Enforce standards
            _, self._export_enforce_std = imgui.checkbox(
                "Enforce subtitle standards (50 chars, 2 lines)", self._export_enforce_std)

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            # Validation info
            if self.track:
                total = len(self.track)
                translated = sum(1 for s in self.track if s.text_translated)
                untranslated = total - translated
                short = sum(1 for s in self.track if s.duration_ms < 500)
                low_conf = sum(1 for s in self.track if s.confidence < 0.4)

                imgui.text(f"Subtitles: {total}")
                if untranslated > 0 and self._export_format_idx == 0:
                    imgui.text_colored(f"Untranslated: {untranslated}", 1.0, 0.6, 0.2, 1.0)
                if short > 0:
                    imgui.text_colored(f"Very short (<0.5s): {short}", 0.9, 0.7, 0.3, 1.0)
                if low_conf > 0:
                    imgui.text_colored(f"Low confidence: {low_conf}", 0.9, 0.5, 0.3, 1.0)

                # Preview first 3 subtitles
                imgui.spacing()
                imgui.text_disabled("Preview:")
                lang_key = ['translated', 'original', 'bilingual'][self._export_format_idx]
                for seg in list(self.track)[:3]:
                    if not seg.is_speech:
                        continue
                    if lang_key == 'translated':
                        txt = seg.text_translated or seg.text_original
                    elif lang_key == 'original':
                        txt = seg.text_original
                    else:
                        txt = f"{seg.text_translated}\n  {seg.text_original}" if seg.text_translated else seg.text_original
                    imgui.text_disabled(f"  {_t(seg.start_ms)}  {(txt or '---')[:50]}")

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            if imgui.button("Export##expgo", width=120):
                lang_key = ['translated', 'original', 'bilingual'][self._export_format_idx]
                if self._export_enforce_std and self.track:
                    self.track.enforce_subtitle_standards()
                self._do_export_with_format(lang_key)
                self._export_dialog_open = False
                imgui.close_current_popup()
            imgui.same_line()
            if imgui.button("Cancel##expcan", width=80):
                self._export_dialog_open = False
                imgui.close_current_popup()
            imgui.end_popup()
        else:
            self._export_dialog_open = False

    def _do_export_with_format(self, lang_key: str):
        """Export using the project's file dialog with the chosen format."""
        if not self.track:
            return
        fd = getattr(getattr(self.app, 'gui_instance', None), 'file_dialog', None)
        if not fd:
            self.app.notify("File dialog not available", "error", 3.0)
            return

        d, n = "", "subtitles.srt"
        if self.app.processor and self.app.processor.video_path:
            vp = self.app.processor.video_path
            d = os.path.dirname(vp)
            base = os.path.splitext(os.path.basename(vp))[0]
            suffix = {'translated': 'en', 'original': 'ja', 'bilingual': 'bilingual'}.get(lang_key, 'en')
            n = f"{base}.{suffix}.srt"

        def _on_export(path):
            try:
                self.track.export_srt(path, lang_key)
                self.app.notify(f"Exported {os.path.basename(path)}", "success", 3.0)
            except Exception as e:
                self.app.notify(f"Export failed: {e}", "error", 5.0)

        fd.show(
            title="Export Subtitles",
            is_save=True,
            extension_filter="SRT Files (*.srt),*.srt",
            callback=_on_export,
            initial_path=d or None,
            initial_filename=n,
        )

    def _get_translator(self):
        """Get or create a cached Translator instance (avoids reloading 7B model each time)."""
        llm_size = self._quality_map[self._quality_idx]
        track = self.track or getattr(self.app, 'subtitle_track', None)
        source_lang = (track.source_language if track else None) or "ja"
        # Cache key: (source_lang, llm_size)
        cache_key = (source_lang, llm_size)
        if not hasattr(self, '_translator_cache_key') or self._translator_cache_key != cache_key:
            from subtitle_translation.pipeline import enable_offline_mode
            enable_offline_mode()
            from subtitle_translation.translator import Translator
            self._cached_translator = Translator(
                source_lang=source_lang, target_lang='en', llm_size=llm_size)
            self._translator_cache_key = cache_key
        return self._cached_translator

    def _retranslate_segment(self, seg_index: int):
        """Re-translate a single subtitle segment in the background."""
        t = self._gen_thread
        if t is not None and t.is_alive():
            self.app.notify("Translation already in progress", "warning", 2.0)
            return
        track = self.track or getattr(self.app, 'subtitle_track', None)
        if not track or seg_index < 0 or seg_index >= len(track):
            self.app.notify("No subtitle to re-translate", "warning", 2.0)
            return
        seg = track[seg_index]
        if not seg.text_original:
            self.app.notify("No original text to re-translate", "warning", 2.0)
            return

        old_text = seg.text_translated or ""
        self.app.notify(f"Re-translating #{seg_index+1}...", "info", 1.0)

        def _run():
            try:
                translator = self._get_translator()
                results = translator.translate_batch([seg.text_original])
                if results and results[0] and results[0] != seg.text_original:
                    track._push_undo()
                    seg.text_translated = results[0]
                    logger.info(f"Re-translated #{seg_index}: '{old_text}' -> '{results[0]}'")
                    self.app.notify(f"#{seg_index+1}: {results[0][:40]}", "success", 2.5)
                else:
                    self.app.notify("Re-translate returned same text", "warning", 2.0)
            except Exception as e:
                logger.error(f"Re-translate failed: {e}", exc_info=True)
                self.app.notify(f"Re-translate failed: {e}", "error", 4.0)

        self._gen_thread = threading.Thread(target=_run, daemon=True, name="SubRetrans")
        self._gen_thread.start()

    def _seek(self, seg):
        if self.app.processor and self.app.processor.fps > 0:
            self.app.processor.seek_video(int(seg.start_ms * self.app.processor.fps / 1000))

    def _start_generate(self):
        t = self._gen_thread
        if t is not None and t.is_alive():
            return
        self.state = self.STATE_GENERATING
        self.progress = 0.0
        self.progress_text = ""
        self._gen_error = None
        self._cancel_requested = False
        # Drain any stale progress from previous run
        while not self._progress_q.empty():
            try:
                self._progress_q.get_nowait()
            except queue.Empty:
                break

        lang = self._lang_codes[self._lang_idx]
        llm_size = self._quality_map[self._quality_idx]

        def _run():
            try:
                from subtitle_translation.model_downloader import check_dependencies_installed, download_models
                if not check_dependencies_installed():
                    self._on_progress(0.0, "Installing dependencies...")
                    if not download_models(progress_callback=self._on_progress):
                        self._gen_error = "Install failed"
                        return

                # Go offline AFTER deps are confirmed but BEFORE model loading
                from subtitle_translation.pipeline import enable_offline_mode, SubtitlePipeline
                enable_offline_mode()
                self.track = SubtitlePipeline(
                    llm_size=llm_size, app=self.app
                ).generate(
                    video_path=self.app.processor.video_path,
                    language=lang,
                    progress_callback=self._on_progress,
                )
            except InterruptedError:
                logger.info("Subtitle generation cancelled by user")
            except Exception as e:
                self._gen_error = str(e)
                logger.error(f"Generate failed: {e}", exc_info=True)

        self._gen_thread = threading.Thread(target=_run, daemon=True, name="SubGen")
        self._gen_thread.start()

    def _on_progress(self, p, text):
        """Thread-safe progress callback — enqueues for main thread to read."""
        if self._cancel_requested:
            raise InterruptedError("Cancelled by user")
        self._progress_q.put((p, text))

    def _do_import_dialog(self):
        fd = getattr(getattr(self.app, 'gui_instance', None), 'file_dialog', None)
        if fd:
            initial_dir = ""
            if self.app.processor and self.app.processor.video_path:
                initial_dir = os.path.dirname(self.app.processor.video_path)
            fd.show(
                title="Import Subtitle File",
                is_save=False,
                extension_filter="Subtitle Files (*.srt *.vtt),*.srt;*.vtt",
                callback=self._do_import,
                initial_path=initial_dir or None,
            )
        else:
            self.app.notify("File dialog not available", "error", 3.0)

    def _do_import(self, path):
        try:
            from subtitle_translation.srt_importer import import_srt
            vms = None
            if self.app.processor and self.app.processor.video_info:
                fps = self.app.processor.fps or 30
                vms = frame_to_ms(self.app.processor.total_frames or 0, fps) if fps > 0 else None
            self.track = import_srt(path, vms)
            if self.track and len(self.track) > 0:
                self.state = self.STATE_EDITING
                self.app.subtitle_track = self.track
                if hasattr(self.app, 'app_state_ui'):
                    self.app.app_state_ui.show_subtitle_timeline = True
                self.app.notify(f"Imported {len(self.track)} subtitles", "success", 3.0)
            else:
                self.app.notify("No subtitles found", "warning", 3.0)
        except Exception as e:
            self.app.notify(f"Import failed: {e}", "error", 5.0)

def _t(ms):
    """Format ms as MM:SS.ss"""
    m, s = divmod(ms / 1000, 60)
    return f"{int(m):02d}:{s:05.2f}"


def _frame_ms(app) -> int:
    """Get duration of 1 video frame in ms (fps-aware). Falls back to 33ms (~30fps)."""
    if app.processor and app.processor.fps > 0:
        return max(1, int(1000 / app.processor.fps))
    return 33


def _confidence_rgb(confidence: float):
    """Map confidence (0-1) to RGB: green=high, yellow=medium, red=low."""
    c = max(0.0, min(1.0, confidence))
    if c >= 0.7:
        return (0.15, 0.40, 0.60)   # blue-green (good)
    elif c >= 0.4:
        return (0.45, 0.40, 0.15)   # amber (medium)
    else:
        return (0.55, 0.20, 0.15)   # red (low)


def _parse_time(text: str):
    """Parse time string to ms. Accepts MM:SS.ss, SS.ss, or just SS. Returns int ms or None."""
    text = text.strip()
    try:
        if ':' in text:
            parts = text.split(':')
            m = int(parts[0])
            s = float(parts[1])
            return int((m * 60 + s) * 1000)
        else:
            return int(float(text) * 1000)
    except (ValueError, IndexError):
        return None
