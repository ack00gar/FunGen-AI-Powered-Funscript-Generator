"""Batch Processing & Live Capture tab UI mixin for ControlPanelUI.

All GUI rendering for the Patreon Exclusive tab lives here.
The addon modules (patreon_features/) provide only backend logic.
When the addon is missing, a grayed-out preview is shown instead.
"""

import os
import imgui
import logging
import threading
from typing import Optional, List

from application.utils import primary_button_style, destructive_button_style
from application.utils.imgui_helpers import DisabledScope as _DisabledScope, tooltip_if_hovered as _tooltip_if_hovered
from application.utils.section_card import section_card as _section_card
from config.element_group_colors import ControlPanelColors as _CPColors

logger = logging.getLogger(__name__)

# ── Module-level state (batch) ──────────────────────────────────────────────
_watched_folder_processor = None
_batch_queue = None
_batch_worker = None

# ── Module-level state (live capture) ───────────────────────────────────────
_capture_manager = None

_cached_windows: List = []
_windows_loading = False
_windows_loaded = False

_url_validation_result: Optional[str] = None
_url_validating = False

_source_type = 0  # 0=Screen, 1=Window, 2=Stream
_selected_window = 0
_stream_url = ""
_window_filter = ""

_selected_tracker_idx = 0
_tracker_names: List[str] = []
_tracker_display_names: List[str] = []
_trackers_loaded = False
_connect_device = False
_save_funscript = True

_SOURCE_LABELS = ["Screen", "Window", "Stream URL"]
_CAPTURE_WIDTH = 640
_CAPTURE_HEIGHT = 640


# ── Lazy-init helpers ───────────────────────────────────────────────────────

def _get_batch_components(app):
    """Import batch logic from addon, or return (None, None, None) for preview mode."""
    global _watched_folder_processor, _batch_queue, _batch_worker

    try:
        if _batch_queue is None:
            from patreon_features.batch.batch_queue import BatchQueue
            _batch_queue = BatchQueue()

        if _watched_folder_processor is None:
            from patreon_features.batch.watched_folder import WatchedFolderProcessor

            def on_new_video(path):
                _batch_queue.add(path)

            _watched_folder_processor = WatchedFolderProcessor(on_new_video=on_new_video)

        if _batch_worker is None:
            from patreon_features.batch.batch_worker import BatchWorker
            _batch_worker = BatchWorker(app, _batch_queue)

        return _watched_folder_processor, _batch_queue, _batch_worker
    except ImportError:
        return None, None, None


def _get_capture_manager():
    """Import capture manager from addon, or return None."""
    global _capture_manager
    if _capture_manager is None:
        try:
            from patreon_features.live_capture.capture_manager import CaptureManager
            _capture_manager = CaptureManager(width=_CAPTURE_WIDTH, height=_CAPTURE_HEIGHT)
        except ImportError:
            return None
    return _capture_manager


def _load_tracker_list():
    """Load available trackers from the tracker discovery system."""
    global _tracker_names, _tracker_display_names, _trackers_loaded, _selected_tracker_idx

    try:
        from config.tracker_discovery import get_tracker_discovery
        discovery = get_tracker_discovery()
        display_names, internal_names = discovery.get_gui_display_list()
        _tracker_display_names = list(display_names)
        _tracker_names = list(internal_names)
        _trackers_loaded = True
    except Exception as e:
        logger.warning(f"Failed to load tracker list: {e}")
        _tracker_display_names = ["(no trackers found)"]
        _tracker_names = [""]
        _trackers_loaded = True


def _sync_tracker_selection(app):
    """Sync the tracker combo with the app's current tracker selection."""
    global _selected_tracker_idx
    if not _tracker_names:
        return
    current = getattr(app.app_state_ui, 'selected_tracker_name', '')
    if current and current in _tracker_names:
        _selected_tracker_idx = _tracker_names.index(current)


def _refresh_windows_async():
    """Refresh window list in a background thread."""
    global _windows_loading, _windows_loaded, _cached_windows

    if _windows_loading:
        return

    _windows_loading = True

    def _do_refresh():
        global _cached_windows, _windows_loading, _windows_loaded
        try:
            from patreon_features.live_capture.window_list import get_available_windows
            result = get_available_windows()
            _cached_windows = result
        except Exception as e:
            logger.warning(f"Failed to list windows: {e}")
            _cached_windows = []
        _windows_loading = False
        _windows_loaded = True

    threading.Thread(target=_do_refresh, daemon=True, name="WindowListRefresh").start()


def _validate_url_async(url: str):
    """Validate a stream URL in a background thread."""
    global _url_validation_result, _url_validating

    def _do_validate():
        global _url_validation_result, _url_validating
        try:
            from patreon_features.live_capture.capture_sources import StreamCaptureSource
            ok, msg = StreamCaptureSource.validate_url(url)
            _url_validation_result = msg if not ok else "Valid"
        except Exception as e:
            _url_validation_result = str(e)
        _url_validating = False

    _url_validating = True
    _url_validation_result = None
    threading.Thread(target=_do_validate, daemon=True).start()


def _open_folder_dialog(app, watcher):
    """Open the app's folder dialog to pick a watch folder."""
    def _on_folder_selected(folder_path):
        app_state = app.app_state_ui
        app_state._batch_watch_path = folder_path
        app.app_settings.set("batch_watch_path", folder_path)

    gi = getattr(app, "gui_instance", None)
    if gi and hasattr(gi, "file_dialog"):
        initial = app.app_settings.get("batch_watch_path", "")
        gi.file_dialog.show(
            title="Select Watch Folder",
            callback=_on_folder_selected,
            is_folder_dialog=True,
            initial_path=initial if initial and os.path.isdir(initial) else None,
        )


# ── Mixin class ─────────────────────────────────────────────────────────────

class BatchMixin:
    """Mixin providing Batch Processing & Live Capture tab rendering methods."""

    def _render_supporter_batch_tab(self):
        """Render the Patreon Exclusive tab (supporter feature).

        When addon is available -> full interactive UI.
        When addon is missing -> grayed-out preview with promo banner.
        """
        if not self._feat_supporter:
            self._render_batch_preview()
            return

        # Version info
        self._render_addon_version_label("patreon_features", "Patreon Exclusive")

        watcher, queue, worker = _get_batch_components(self.app)
        if watcher is None:
            imgui.text_colored("Batch module could not be loaded", *_CPColors.ERROR_TEXT)
            return

        # --- Watched Folder Section ---
        with _section_card("Watched Folder##BatchWatchedFolder", tier="primary") as is_open:
            if is_open:
                self._render_watched_folder_section(watcher, worker)

        # --- Queue Section ---
        with _section_card("Batch Queue##BatchQueue", tier="primary") as is_open:
            if is_open:
                self._render_queue_section(queue, worker)

        # --- Live Capture Section ---
        with _section_card("Live Capture##LiveCapture", tier="primary") as is_open:
            if is_open:
                self._render_capture_panel()

        imgui.spacing()
        self._render_patreon_exclusive_extras()

    # ── Watched Folder ──────────────────────────────────────────────────

    def _render_watched_folder_section(self, watcher, worker):
        """Render watched folder controls."""
        app = self.app
        if watcher.is_watching:
            imgui.text_colored("Watching:", *_CPColors.SUCCESS_TEXT)
            imgui.same_line()
            watch_display = watcher.watch_path or ""
            imgui.text(watch_display)
            if imgui.is_item_hovered() and watcher.watch_path:
                imgui.set_tooltip(watcher.watch_path)

            with destructive_button_style():
                if imgui.button("Stop Watching", width=-1):
                    watcher.stop_watching()
                    if worker.is_running:
                        worker.stop()
        else:
            if not hasattr(app_state := app.app_state_ui, '_batch_watch_path'):
                app_state._batch_watch_path = app.app_settings.get("batch_watch_path", "")

            if imgui.button("Browse...##WatchFolderBrowse"):
                _open_folder_dialog(app, watcher)
            imgui.same_line()
            path_display = app_state._batch_watch_path or "(no folder selected)"
            imgui.text(path_display)
            if imgui.is_item_hovered() and app_state._batch_watch_path:
                imgui.set_tooltip(app_state._batch_watch_path)

            if not hasattr(app_state, '_batch_watch_recursive'):
                app_state._batch_watch_recursive = app.app_settings.get("batch_watch_recursive", False)
            _, app_state._batch_watch_recursive = imgui.checkbox(
                "Include subfolders", app_state._batch_watch_recursive)

            can_start = bool(app_state._batch_watch_path) and os.path.isdir(app_state._batch_watch_path)
            with _DisabledScope(not can_start):
                with primary_button_style():
                    if imgui.button("Start Watching", width=-1):
                        app.app_settings.set("batch_watch_path", app_state._batch_watch_path)
                        app.app_settings.set("batch_watch_recursive", app_state._batch_watch_recursive)
                        watcher.start_watching(app_state._batch_watch_path, app_state._batch_watch_recursive)
                        if not worker.is_running:
                            worker.start()

            imgui.text_wrapped("Watches while the app is running. For headless mode use: --watch FOLDER")

        imgui.spacing()
        tracker_name = getattr(app.app_state_ui, 'selected_tracker_name', '(none)')
        imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.LABEL_TEXT)
        imgui.text(f"Tracker: {tracker_name}")
        imgui.pop_style_color()
        imgui.text_wrapped("Uses the same tracker and settings as the Run tab.")
        imgui.spacing()

    # ── Batch Queue ─────────────────────────────────────────────────────

    def _render_queue_section(self, queue, worker):
        """Render batch queue status and controls."""
        status = queue.get_status_summary()

        # Worker status
        if worker.is_running:
            imgui.text_colored("Worker: Running", *_CPColors.SUCCESS_TEXT)
        else:
            imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.LABEL_TEXT)
            imgui.text("Worker: Stopped")
            imgui.pop_style_color()
        imgui.same_line()

        if worker.is_running:
            if imgui.small_button("Stop Worker##BatchWorkerStop"):
                worker.stop()
        else:
            if imgui.small_button("Start Worker##BatchWorkerStart"):
                worker.start()

        imgui.spacing()

        imgui.text(f"Total: {status['total']}  ")
        imgui.same_line()
        imgui.text_colored(f"Done: {status['completed']}", *_CPColors.SUCCESS_TEXT)
        imgui.same_line()
        if status['failed'] > 0:
            imgui.text_colored(f"Failed: {status['failed']}", *_CPColors.ERROR_TEXT)
            imgui.same_line()
        imgui.text(f"Queued: {status['queued']}")

        est = status['estimated_remaining_s']
        if est > 0:
            minutes = int(est // 60)
            seconds = int(est % 60)
            imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.LABEL_TEXT)
            imgui.text(f"Estimated remaining: {minutes}m {seconds}s")
            imgui.pop_style_color()

        # Pause/Resume
        if queue.is_paused:
            with primary_button_style():
                if imgui.button("Resume", width=100):
                    queue.resume()
        else:
            if imgui.button("Pause", width=100):
                queue.pause()

        imgui.same_line()
        with destructive_button_style():
            if imgui.button("Clear Queue", width=100):
                queue.clear()

        # Item list
        items = queue.items
        if items:
            imgui.spacing()
            avail = imgui.get_content_region_available()
            list_height = min(avail[1] - 20, max(80, len(items) * 22))
            imgui.begin_child("##BatchQueueItems", width=0, height=list_height, border=True)
            for i, item in enumerate(items):
                status_icon = {
                    "QUEUED": "[.]", "PROCESSING": "[>]", "COMPLETED": "[+]",
                    "FAILED": "[X]", "SKIPPED": "[-]"
                }.get(item.status.name, "[?]")

                color = {
                    "QUEUED": _CPColors.LABEL_TEXT,
                    "PROCESSING": _CPColors.STATUS_INFO,
                    "COMPLETED": _CPColors.SUCCESS_TEXT,
                    "FAILED": _CPColors.ERROR_TEXT,
                    "SKIPPED": _CPColors.LABEL_TEXT,
                }.get(item.status.name, _CPColors.LABEL_TEXT)

                imgui.text_colored(status_icon, *color)
                imgui.same_line()
                imgui.text(os.path.basename(item.video_path))
                if imgui.is_item_hovered():
                    tooltip = item.video_path
                    if item.error_message:
                        tooltip += f"\nError: {item.error_message}"
                    imgui.set_tooltip(tooltip)
            imgui.end_child()

        imgui.spacing()

    # ── Live Capture ────────────────────────────────────────────────────

    def _render_capture_panel(self):
        """Render the live capture panel."""
        global _source_type

        manager = _get_capture_manager()
        if manager is None:
            imgui.text_colored("Live capture module not available", *_CPColors.ERROR_TEXT)
            return

        if not _trackers_loaded:
            _load_tracker_list()
            _sync_tracker_selection(self.app)

        status = manager.get_status()
        is_capturing = manager.is_capturing

        # Source type selector
        imgui.text("Source:")
        imgui.same_line()
        for i, label in enumerate(_SOURCE_LABELS):
            if i > 0:
                imgui.same_line()
            if imgui.radio_button(label, _source_type == i):
                _source_type = i
                if i == 1 and not _windows_loaded and not _windows_loading:
                    _refresh_windows_async()

        imgui.spacing()

        if _source_type == 0:
            self._render_capture_screen_controls()
        elif _source_type == 1:
            self._render_capture_window_controls()
        elif _source_type == 2:
            self._render_capture_stream_controls()

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        if not is_capturing:
            self._render_capture_tracker_options()
            imgui.spacing()
            imgui.separator()

        # Start/Stop
        if is_capturing:
            with destructive_button_style():
                if imgui.button("Stop Capture", width=-1):
                    manager.stop()
        else:
            with primary_button_style():
                if imgui.button("Start Capture", width=-1):
                    self._start_capture(manager)

        # Status
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
        global _selected_window, _window_filter

        imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.WARNING_TEXT)
        imgui.text_wrapped(
            "Note: GPU-accelerated apps (Chrome, Firefox) may show a black/white screen. "
            "Use Screen capture instead.")
        imgui.pop_style_color()
        imgui.spacing()

        if not _windows_loaded and not _windows_loading:
            _refresh_windows_async()

        if _windows_loading:
            imgui.text("Scanning windows...")
            return

        if imgui.small_button("Refresh##WindowRefresh"):
            _refresh_windows_async()
            return

        if not _cached_windows:
            imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.LABEL_TEXT)
            imgui.text("No windows found.")
            imgui.pop_style_color()
            return

        changed, _window_filter = imgui.input_text("Filter##WindowFilter", _window_filter, 256)

        if _window_filter:
            filt = _window_filter.lower()
            filtered = [(i, w) for i, w in enumerate(_cached_windows)
                        if filt in w.display_label.lower()]
        else:
            filtered = list(enumerate(_cached_windows))

        window_labels = [w.display_label for _, w in filtered]
        if not window_labels:
            imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.LABEL_TEXT)
            imgui.text("No matching windows")
            imgui.pop_style_color()
            return

        display_idx = 0
        for di, (orig_idx, _) in enumerate(filtered):
            if orig_idx == _selected_window:
                display_idx = di
                break

        clicked, new_display_idx = imgui.combo("##WindowSelect", display_idx, window_labels)
        if clicked and 0 <= new_display_idx < len(filtered):
            _selected_window = filtered[new_display_idx][0]

    def _render_capture_stream_controls(self):
        global _stream_url

        changed, _stream_url = imgui.input_text("URL##StreamURL", _stream_url, 1024)

        imgui.same_line()
        if _url_validating:
            imgui.text("Testing...")
        else:
            if imgui.small_button("Test URL"):
                if _stream_url.strip():
                    _validate_url_async(_stream_url.strip())

        if _url_validation_result:
            if _url_validation_result == "Valid":
                imgui.text_colored("URL is valid", *_CPColors.SUCCESS_TEXT)
            else:
                imgui.text_colored(_url_validation_result, *_CPColors.ERROR_TEXT)

    def _render_capture_tracker_options(self):
        global _selected_tracker_idx, _connect_device, _save_funscript

        if _tracker_display_names:
            imgui.text("Tracker:")
            imgui.same_line()
            avail_w = imgui.get_content_region_available()[0]
            imgui.push_item_width(avail_w)
            changed, _selected_tracker_idx = imgui.combo(
                "##CaptureTracker", _selected_tracker_idx, _tracker_display_names)
            imgui.pop_item_width()
        else:
            imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.LABEL_TEXT)
            imgui.text("No trackers available")
            imgui.pop_style_color()

        from application.utils.feature_detection import is_feature_available
        if is_feature_available("device_control"):
            _, _connect_device = imgui.checkbox("Connect Device##CaptureDevice", _connect_device)
            _tooltip_if_hovered("Send tracker output to connected device in real-time")

        _, _save_funscript = imgui.checkbox("Save funscript on stop##CaptureSave", _save_funscript)

    def _start_capture(self, manager):
        """Create the appropriate capture source and start capturing."""
        w, h = _CAPTURE_WIDTH, _CAPTURE_HEIGHT
        manager.width = w
        manager.height = h

        tracker_name = ""
        if _tracker_names and 0 <= _selected_tracker_idx < len(_tracker_names):
            tracker_name = _tracker_names[_selected_tracker_idx]

        try:
            if _source_type == 0:
                from patreon_features.live_capture.capture_sources import ScreenCaptureSource
                source = ScreenCaptureSource(screen_index=0, width=w, height=h)
            elif _source_type == 1:
                from patreon_features.live_capture.capture_sources import WindowCaptureSource
                if _cached_windows and 0 <= _selected_window < len(_cached_windows):
                    win = _cached_windows[_selected_window]
                    source = WindowCaptureSource(
                        window_title=win.title or win.app_name,
                        window_id=win.window_id, width=w, height=h)
                else:
                    logger.warning("No window selected")
                    return
            elif _source_type == 2:
                from patreon_features.live_capture.capture_sources import StreamCaptureSource
                if not _stream_url.strip():
                    logger.warning("No stream URL provided")
                    return
                source = StreamCaptureSource(url=_stream_url.strip(), width=w, height=h)
            else:
                return

            processor = getattr(self.app, 'processor', None)
            if processor and hasattr(processor, 'pause_processing') and processor.is_processing:
                processor.pause_processing()
                logger.info("Video playback paused for live capture")

            manager.start(source, self.app,
                          tracker_name=tracker_name,
                          device_control=_connect_device,
                          save_funscript=_save_funscript)
        except Exception as e:
            logger.error(f"Failed to start capture: {e}")

    # ── Preview (addon not installed) ───────────────────────────────────

    def _render_batch_preview(self):
        """Render grayed-out preview of batch tab when addon is missing."""
        self._render_addon_promo_banner(
            "Patreon Exclusive",
            "Process multiple videos in sequence with a batch queue. "
            "Set up watched folders for automatic processing. "
            "Live capture from screen, window, or stream."
        )
        with _DisabledScope(True):
            # Watched Folder preview
            with _section_card("Watched Folder##PreviewWatchedFolder", tier="primary") as is_open:
                if is_open:
                    imgui.button("Browse...##PreviewWatchBrowse")
                    imgui.same_line()
                    imgui.text("(no folder selected)")
                    imgui.checkbox("Include subfolders##PreviewRecursive", False)
                    imgui.button("Start Watching##PreviewStartWatch", width=-1)
                    imgui.spacing()
                    imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.LABEL_TEXT)
                    imgui.text("Tracker: (select in Run tab)")
                    imgui.pop_style_color()
                    imgui.text_wrapped("Watches while the app is running. For headless mode use: --watch FOLDER")

            # Batch Queue preview
            with _section_card("Batch Queue##PreviewBatchQueue", tier="primary") as is_open:
                if is_open:
                    imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.LABEL_TEXT)
                    imgui.text("Worker: Stopped")
                    imgui.pop_style_color()
                    imgui.same_line()
                    imgui.small_button("Start Worker##PreviewWorkerStart")
                    imgui.spacing()
                    imgui.text("Total: 0  ")
                    imgui.same_line()
                    imgui.text_colored("Done: 0", *_CPColors.SUCCESS_TEXT)
                    imgui.same_line()
                    imgui.text("Queued: 0")
                    imgui.button("Pause##PreviewPause", width=100)
                    imgui.same_line()
                    imgui.button("Clear Queue##PreviewClear", width=100)

            # Live Capture preview
            with _section_card("Live Capture##PreviewLiveCapture", tier="primary") as is_open:
                if is_open:
                    imgui.text("Source:")
                    imgui.same_line()
                    imgui.radio_button("Screen##PreviewScreen", True)
                    imgui.same_line()
                    imgui.radio_button("Window##PreviewWindow", False)
                    imgui.same_line()
                    imgui.radio_button("Stream URL##PreviewStream", False)
                    imgui.spacing()
                    imgui.text("Captures your primary screen.")
                    imgui.spacing()
                    imgui.separator()
                    imgui.spacing()
                    imgui.text("Tracker:")
                    imgui.same_line()
                    imgui.push_item_width(imgui.get_content_region_available()[0])
                    imgui.combo("##PreviewCaptureTracker", 0, ["(select tracker)"])
                    imgui.pop_item_width()
                    imgui.checkbox("Save funscript on stop##PreviewCaptureSave", True)
                    imgui.spacing()
                    imgui.button("Start Capture##PreviewStartCapture", width=-1)
                    imgui.spacing()
                    imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.LABEL_TEXT)
                    imgui.text("Idle")
                    imgui.pop_style_color()

        # Show extras section even in preview
        self._render_patreon_exclusive_extras()

    # ── Patreon Extras (always shown) ───────────────────────────────────

    def _render_patreon_exclusive_extras(self):
        """Render additional Patreon-exclusive feature sections."""
        # --- Recording Mode ---
        with _section_card("Recording Mode##patreon_rec", open_by_default=False) as _open:
            if _open:
                imgui.text_wrapped(
                    "Draw funscripts by moving your mouse while video plays. "
                    "Points are auto-simplified with RDP."
                )
                imgui.spacing()
                imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.HINT_TEXT)
                imgui.text("Activate via timeline toolbar (REC mode)")
                imgui.pop_style_color()

        # --- Dynamic Injection ---
        with _section_card("Dynamic Injection##patreon_inj", open_by_default=False) as _open:
            if _open:
                imgui.text_wrapped(
                    "Add intermediate points to smooth out segments. "
                    "Supports linear, cosine, and cubic interpolation."
                )
                imgui.spacing()
                imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.HINT_TEXT)
                imgui.text("Right-click a segment in Injection mode on the timeline")
                imgui.pop_style_color()

        # --- Pattern Library ---
        with _section_card("Pattern Library##patreon_pat", open_by_default=False) as _open:
            if _open:
                imgui.text_wrapped(
                    "Save and reuse movement patterns (soft bounces, sine waves, custom motions). "
                    "Apply with speed and amplitude scaling."
                )
                imgui.spacing()
                pattern_lib = getattr(self.app, 'pattern_library', None)
                if pattern_lib:
                    patterns = pattern_lib.list_patterns()
                    if patterns:
                        imgui.text(f"Saved patterns: {len(patterns)}")
                        for p_name in patterns:
                            imgui.bullet_text(p_name)
                    else:
                        imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.LABEL_TEXT)
                        imgui.text("No patterns saved yet")
                        imgui.pop_style_color()
                    imgui.spacing()
                    imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.HINT_TEXT)
                    imgui.text("Select points on timeline > right-click > Save Selection as Pattern")
                    imgui.pop_style_color()
                else:
                    imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.LABEL_TEXT)
                    imgui.text("Pattern library not loaded")
                    imgui.pop_style_color()

        # --- Multi-Axis Generation ---
        with _section_card("Multi-Axis Generation##patreon_ma", open_by_default=False) as _open:
            if _open:
                imgui.text_wrapped(
                    "Auto-generate Roll, Pitch, Twist, Sway, and Surge axes from your stroke axis. "
                    "Heuristic mode derives motion from stroke data. Video-aware mode uses body pose keypoints."
                )
                imgui.spacing()
                imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.HINT_TEXT)
                imgui.text("Right-click timeline > Generate Axis > choose axis")
                imgui.pop_style_color()

        # --- Live Device Preview ---
        with _section_card("Live Device Preview##patreon_ldp", open_by_default=False) as _open:
            if _open:
                imgui.text_wrapped(
                    "Feel cursor position on your device while editing, without playing the video. "
                    "Rate-limited to avoid flooding device commands."
                )
                imgui.spacing()
                imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.HINT_TEXT)
                imgui.text("Enable in Device Control tab > Live Control Integration")
                imgui.pop_style_color()
