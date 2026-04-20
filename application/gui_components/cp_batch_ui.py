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


_QUEUE_STATUS_ICONS = {
    "QUEUED": "[.]",
    "PROCESSING": "[>]",
    "COMPLETED": "[+]",
    "FAILED": "[X]",
    "SKIPPED": "[-]",
}


def _queue_status_color_map():
    return {
        "QUEUED": _CPColors.LABEL_TEXT,
        "PROCESSING": _CPColors.STATUS_INFO,
        "COMPLETED": _CPColors.SUCCESS_TEXT,
        "FAILED": _CPColors.ERROR_TEXT,
        "SKIPPED": _CPColors.LABEL_TEXT,
    }


class BatchMixin:
    """Mixin providing Batch Processing & Live Capture tab rendering methods."""

    def _init_batch_state(self):
        """Initialize all batch/capture state. Called from ControlPanelUI.__init__."""
        # Backend singletons (lazy-init)
        self._batch_watcher = None
        self._batch_queue = None
        self._batch_worker = None
        self._capture_manager = None
        self._batch_init_lock = threading.Lock()

        # Live capture UI state
        self._capture_source_type = 0  # 0=Screen, 1=Window, 2=Stream
        self._capture_selected_window = 0
        self._capture_stream_url = ""
        self._capture_window_filter = ""
        self._cached_windows: List = []
        self._windows_loading = False
        self._windows_loaded = False
        self._url_validation_result: Optional[str] = None
        self._url_validating = False

        # Tracker selection for capture
        self._capture_tracker_idx = 0
        self._capture_tracker_names: List[str] = []
        self._capture_tracker_display_names: List[str] = []
        self._capture_trackers_loaded = False
        self._capture_connect_device = False
        self._capture_save_funscript = True

    # ── Lazy-init helpers ──────────────────────────────────────────────

    def _get_batch_components(self):
        """Import batch logic from addon, or return (None, None, None)."""
        if self._batch_queue is not None and self._batch_watcher is not None and self._batch_worker is not None:
            return self._batch_watcher, self._batch_queue, self._batch_worker

        with self._batch_init_lock:
            try:
                if self._batch_queue is None:
                    from application.batch.batch_queue import BatchQueue
                    self._batch_queue = BatchQueue()

                if self._batch_watcher is None:
                    from application.batch.watched_folder import WatchedFolderProcessor
                    queue = self._batch_queue
                    self._batch_watcher = WatchedFolderProcessor(on_new_video=lambda path: queue.add(path))

                if self._batch_worker is None:
                    from application.batch.batch_worker import BatchWorker
                    max_parallel = int(self.app.app_settings.get("batch_max_parallel_items", 1) or 1)
                    self._batch_worker = BatchWorker(self.app, self._batch_queue, max_parallel=max_parallel)

                return self._batch_watcher, self._batch_queue, self._batch_worker
            except ImportError:
                return None, None, None

    def _open_batch_folder_dialog(self):
        """Open the app's folder dialog to pick a watch folder."""
        app = self.app
        def _on_folder_selected(folder_path):
            app.app_state_ui._batch_watch_path = folder_path
            app.app_settings.config.batch.watch_path = folder_path

        gi = getattr(app, "gui_instance", None)
        if gi and hasattr(gi, "file_dialog"):
            initial = app.app_settings.config.batch.watch_path
            gi.file_dialog.show(
                title="Select Watch Folder",
                callback=_on_folder_selected,
                is_folder_dialog=True,
                initial_path=initial if initial and os.path.isdir(initial) else None,
            )

    def _render_supporter_batch_tab(self):
        """Render the Batch Processing tab (add-on holders).

        When addon is available -> full interactive UI.
        When addon is missing -> grayed-out preview with promo banner.
        """
        if not self._feat_supporter:
            self._render_batch_preview()
            return

        # Version info
        self._render_addon_version_label("patreon_features", "Batch Processing")

        watcher, queue, worker = self._get_batch_components()
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
            _batch_cfg = app.app_settings.config.batch
            if not hasattr(app_state := app.app_state_ui, '_batch_watch_path'):
                app_state._batch_watch_path = _batch_cfg.watch_path

            if imgui.button("Browse...##WatchFolderBrowse"):
                self._open_batch_folder_dialog()
            imgui.same_line()
            path_display = app_state._batch_watch_path or "(no folder selected)"
            imgui.text(path_display)
            if imgui.is_item_hovered() and app_state._batch_watch_path:
                imgui.set_tooltip(app_state._batch_watch_path)

            if not hasattr(app_state, '_batch_watch_recursive'):
                app_state._batch_watch_recursive = _batch_cfg.watch_recursive
            _, app_state._batch_watch_recursive = imgui.checkbox(
                "Include subfolders", app_state._batch_watch_recursive)

            can_start = bool(app_state._batch_watch_path) and os.path.isdir(app_state._batch_watch_path)
            with _DisabledScope(not can_start):
                with primary_button_style():
                    if imgui.button("Start Watching", width=-1):
                        _batch_cfg.watch_path = app_state._batch_watch_path
                        _batch_cfg.watch_recursive = app_state._batch_watch_recursive
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
            color_map = _queue_status_color_map()
            for i, item in enumerate(items):
                status_name = item.status.name
                status_icon = _QUEUE_STATUS_ICONS.get(status_name, "[?]")
                color = color_map.get(status_name, _CPColors.LABEL_TEXT)

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

    # ── Preview (addon not installed) ───────────────────────────────────

    def _render_batch_preview(self):
        """Render grayed-out preview of batch tab when addon is missing."""
        self._render_addon_promo_banner(
            "Batch Processing",
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

