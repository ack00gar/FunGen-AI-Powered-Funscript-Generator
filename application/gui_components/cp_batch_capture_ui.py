"""Live Capture UI mixin for ControlPanelUI, sibling of BatchMixin.

Split out of cp_batch_ui.py to keep the batch file focused on batch-queue /
watched-folder rendering. All live-capture state (_capture_*) continues to
live on the parent ControlPanelUI instance via _init_batch_state in BatchMixin;
this mixin only adds render + async-helper methods.
"""
import imgui
import logging
import threading
from typing import List

from application.utils import primary_button_style, destructive_button_style
from application.utils.imgui_helpers import tooltip_if_hovered as _tooltip_if_hovered
from config.element_group_colors import ControlPanelColors as _CPColors

logger = logging.getLogger(__name__)

_SOURCE_LABELS = ["Screen", "Window", "Stream URL"]
_CAPTURE_WIDTH = 640
_CAPTURE_HEIGHT = 640


class BatchCaptureMixin:

    def _get_capture_manager(self):
        if self._capture_manager is not None:
            return self._capture_manager
        with self._batch_init_lock:
            if self._capture_manager is None:
                try:
                    from application.live_capture.capture_manager import CaptureManager
                    self._capture_manager = CaptureManager(width=_CAPTURE_WIDTH, height=_CAPTURE_HEIGHT)
                except ImportError:
                    return None
        return self._capture_manager

    def _load_capture_tracker_list(self):
        try:
            from config.tracker_discovery import get_tracker_discovery
            discovery = get_tracker_discovery()
            display_names, internal_names = discovery.get_gui_display_list()
            self._capture_tracker_display_names = list(display_names)
            self._capture_tracker_names = list(internal_names)
        except Exception as e:
            logger.warning(f"Failed to load tracker list: {e}")
            self._capture_tracker_display_names = ["(no trackers found)"]
            self._capture_tracker_names = [""]
        self._capture_trackers_loaded = True

    def _sync_capture_tracker_selection(self):
        if not self._capture_tracker_names:
            return
        current = getattr(self.app.app_state_ui, 'selected_tracker_name', '')
        if current and current in self._capture_tracker_names:
            self._capture_tracker_idx = self._capture_tracker_names.index(current)

    def _refresh_windows_async(self):
        if self._windows_loading:
            return
        self._windows_loading = True

        def _do_refresh():
            try:
                from application.live_capture.window_list import get_available_windows
                self._cached_windows = get_available_windows()
            except Exception as e:
                logger.warning(f"Failed to list windows: {e}")
                self._cached_windows = []
            self._windows_loading = False
            self._windows_loaded = True

        threading.Thread(target=_do_refresh, daemon=True, name="WindowListRefresh").start()

    def _validate_url_async(self, url: str):
        self._url_validating = True
        self._url_validation_result = None

        def _do_validate():
            try:
                from application.live_capture.capture_sources import StreamCaptureSource
                ok, msg = StreamCaptureSource.validate_url(url)
                self._url_validation_result = msg if not ok else "Valid"
            except Exception as e:
                self._url_validation_result = str(e)
            self._url_validating = False

        threading.Thread(target=_do_validate, daemon=True).start()

    def _render_capture_panel(self):
        manager = self._get_capture_manager()
        if manager is None:
            imgui.text_colored("Live capture module not available", *_CPColors.ERROR_TEXT)
            return

        if not self._capture_trackers_loaded:
            self._load_capture_tracker_list()
            self._sync_capture_tracker_selection()

        status = manager.get_status()
        is_capturing = manager.is_capturing

        imgui.text("Source:")
        imgui.same_line()
        for i, label in enumerate(_SOURCE_LABELS):
            if i > 0:
                imgui.same_line()
            if imgui.radio_button(label, self._capture_source_type == i):
                self._capture_source_type = i
                if i == 1 and not self._windows_loaded and not self._windows_loading:
                    self._refresh_windows_async()

        imgui.spacing()

        if self._capture_source_type == 0:
            self._render_capture_screen_controls()
        elif self._capture_source_type == 1:
            self._render_capture_window_controls()
        elif self._capture_source_type == 2:
            self._render_capture_stream_controls()

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        if not is_capturing:
            self._render_capture_tracker_options()
            imgui.spacing()
            imgui.separator()

        if is_capturing:
            with destructive_button_style():
                if imgui.button("Stop Capture", width=-1):
                    manager.stop()
            if imgui.is_item_hovered():
                imgui.set_tooltip("Stop capturing and close the source. If 'Save funscript on stop' is on, writes out now.")
        else:
            with primary_button_style():
                if imgui.button("Start Capture", width=-1):
                    self._start_capture(manager)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Begin capturing from the selected source and feed frames to the chosen tracker.")

        imgui.spacing()
        if status["state"] == "CAPTURING":
            imgui.text_colored(f"Capturing at {status['fps']:.1f} FPS", *_CPColors.SUCCESS_TEXT)
            imgui.text(f"Frames: {status['frames_captured']}")
            if status.get("tracker_active"):
                imgui.same_line()
                imgui.text_colored("  Tracking", *_CPColors.STATUS_INFO)
        elif status["state"] == "ERROR":
            imgui.text_colored("Error", *_CPColors.ERROR_TEXT)
            imgui.text_wrapped(status["error"])
        else:
            imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.LABEL_TEXT)
            imgui.text("Idle")
            imgui.pop_style_color()

        last_save = status.get("last_save_path", "")
        if last_save:
            imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.LABEL_TEXT)
            imgui.text("Saved:")
            imgui.pop_style_color()
            imgui.same_line()
            imgui.text(last_save)
            if imgui.is_item_hovered():
                imgui.set_tooltip(last_save)

    def _render_capture_screen_controls(self):
        imgui.text("Captures your primary screen.")

    def _render_capture_window_controls(self):
        imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.WARNING_TEXT)
        imgui.text_wrapped(
            "Note: GPU-accelerated apps (Chrome, Firefox) may show a black/white screen. "
            "Use Screen capture instead.")
        imgui.pop_style_color()
        imgui.spacing()

        if not self._windows_loaded and not self._windows_loading:
            self._refresh_windows_async()

        if self._windows_loading:
            imgui.text("Scanning windows...")
            return

        if imgui.small_button("Refresh##WindowRefresh"):
            self._refresh_windows_async()
            return

        if not self._cached_windows:
            imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.LABEL_TEXT)
            imgui.text("No windows found.")
            imgui.pop_style_color()
            return

        changed, self._capture_window_filter = imgui.input_text(
            "Filter##WindowFilter", self._capture_window_filter, 256)

        if self._capture_window_filter:
            filt = self._capture_window_filter.lower()
            filtered = [(i, w) for i, w in enumerate(self._cached_windows)
                        if filt in w.display_label.lower()]
        else:
            filtered = list(enumerate(self._cached_windows))

        window_labels = [w.display_label for _, w in filtered]
        if not window_labels:
            imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.LABEL_TEXT)
            imgui.text("No matching windows")
            imgui.pop_style_color()
            return

        display_idx = 0
        for di, (orig_idx, _) in enumerate(filtered):
            if orig_idx == self._capture_selected_window:
                display_idx = di
                break

        clicked, new_display_idx = imgui.combo("##WindowSelect", display_idx, window_labels)
        if clicked and 0 <= new_display_idx < len(filtered):
            self._capture_selected_window = filtered[new_display_idx][0]

    def _render_capture_stream_controls(self):
        changed, self._capture_stream_url = imgui.input_text(
            "URL##StreamURL", self._capture_stream_url, 1024)

        imgui.same_line()
        if self._url_validating:
            imgui.text("Testing...")
        else:
            if imgui.small_button("Test URL"):
                if self._capture_stream_url.strip():
                    self._validate_url_async(self._capture_stream_url.strip())

        if self._url_validation_result:
            if self._url_validation_result == "Valid":
                imgui.text_colored("URL is valid", *_CPColors.SUCCESS_TEXT)
            else:
                imgui.text_colored(self._url_validation_result, *_CPColors.ERROR_TEXT)

    def _render_capture_tracker_options(self):
        if self._capture_tracker_display_names:
            imgui.text("Tracker:")
            imgui.same_line()
            imgui.push_item_width(imgui.get_content_region_available()[0])
            changed, self._capture_tracker_idx = imgui.combo(
                "##CaptureTracker", self._capture_tracker_idx, self._capture_tracker_display_names)
            imgui.pop_item_width()
        else:
            imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.LABEL_TEXT)
            imgui.text("No trackers available")
            imgui.pop_style_color()

        if self._feat_device:
            _, self._capture_connect_device = imgui.checkbox(
                "Connect Device##CaptureDevice", self._capture_connect_device)
            _tooltip_if_hovered("Send tracker output to connected device in real-time")

        _, self._capture_save_funscript = imgui.checkbox(
            "Save funscript on stop##CaptureSave", self._capture_save_funscript)

    def _start_capture(self, manager):
        w, h = _CAPTURE_WIDTH, _CAPTURE_HEIGHT
        manager.width = w
        manager.height = h

        tracker_name = ""
        names = self._capture_tracker_names
        if names and 0 <= self._capture_tracker_idx < len(names):
            tracker_name = names[self._capture_tracker_idx]

        try:
            if self._capture_source_type == 0:
                from application.live_capture.capture_sources import ScreenCaptureSource
                source = ScreenCaptureSource(screen_index=0, width=w, height=h)
            elif self._capture_source_type == 1:
                from application.live_capture.capture_sources import WindowCaptureSource
                if self._cached_windows and 0 <= self._capture_selected_window < len(self._cached_windows):
                    win = self._cached_windows[self._capture_selected_window]
                    source = WindowCaptureSource(
                        window_title=win.title or win.app_name,
                        window_id=win.window_id, width=w, height=h)
                else:
                    logger.warning("No window selected")
                    return
            elif self._capture_source_type == 2:
                from application.live_capture.capture_sources import StreamCaptureSource
                if not self._capture_stream_url.strip():
                    logger.warning("No stream URL provided")
                    return
                source = StreamCaptureSource(url=self._capture_stream_url.strip(), width=w, height=h)
            else:
                return

            processor = getattr(self.app, 'processor', None)
            if processor and hasattr(processor, 'pause_processing') and processor.is_processing:
                processor.pause_processing()
                logger.info("Video playback paused for live capture")

            manager.start(source, self.app,
                          tracker_name=tracker_name,
                          device_control=self._capture_connect_device,
                          save_funscript=self._capture_save_funscript)
        except Exception as e:
            logger.error(f"Failed to start capture: {e}")
