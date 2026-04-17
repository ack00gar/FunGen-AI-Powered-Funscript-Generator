"""
Batch Worker

Background worker that polls the BatchQueue and processes items one at a time,
reusing the same tracker and settings as the Run tab. Mirrors the processing
logic in app_logic._run_batch_processing_thread() but operates on queued items
from the watched folder instead of a pre-built batch list.
"""

import os
import time
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class BatchWorker:
    """Background queue processor for watched-folder batch items."""

    def __init__(self, app, queue):
        """
        Args:
            app: The FunGen ApplicationLogic instance.
            queue: A BatchQueue instance to pull items from.
        """
        self.app = app
        self.queue = queue
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        """Start the worker thread."""
        if self.is_running:
            logger.warning("BatchWorker already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="BatchWorker"
        )
        self._thread.start()
        logger.info("BatchWorker started")

    def stop(self):
        """Signal the worker to stop (does not interrupt current item)."""
        self._stop_event.set()
        logger.info("BatchWorker stop requested")

    def _run(self):
        """Main worker loop: poll queue, process items."""
        while not self._stop_event.is_set():
            # Respect queue pause
            if not self.queue.wait_if_paused(timeout=1.0):
                continue
            if self._stop_event.is_set():
                break

            # Find next QUEUED item
            next_idx = self._find_next_queued()
            if next_idx is None:
                # Nothing to do — poll again after a short sleep
                self._stop_event.wait(timeout=2.0)
                continue

            # Don't process if app is already busy with batch or analysis
            if self._app_is_busy():
                self._stop_event.wait(timeout=3.0)
                continue

            # Process the item
            self._process_item(next_idx)

        logger.info("BatchWorker stopped")

    def _find_next_queued(self) -> Optional[int]:
        """Find the index of the next QUEUED item."""
        from application.batch.batch_queue import BatchItemStatus
        items = self.queue.items
        for i, item in enumerate(items):
            if item.status == BatchItemStatus.QUEUED:
                return i
        return None

    def _app_is_busy(self) -> bool:
        """Check if the app is currently busy with other processing."""
        app = self.app
        if getattr(app, 'is_batch_processing_active', False):
            return True
        stage_proc = getattr(app, 'stage_processor', None)
        if stage_proc and getattr(stage_proc, 'full_analysis_active', False):
            return True
        processor = getattr(app, 'processor', None)
        if processor and getattr(processor, 'is_processing', False):
            return True
        return False

    def _process_item(self, idx: int):
        """Process a single queued item using the Run tab's tracker and settings."""
        items = self.queue.items
        if idx >= len(items):
            return

        item = items[idx]
        video_path = item.video_path
        video_basename = os.path.basename(video_path)

        logger.info(f"BatchWorker: Processing {video_basename}")
        self.queue.mark_processing(idx)

        try:
            # Get tracker name from the Run tab's current selection
            tracker_name = getattr(self.app.app_state_ui, 'selected_tracker_name', '')
            if not tracker_name:
                raise RuntimeError("No tracker selected in Run tab")

            # Resolve tracker info
            from config.tracker_discovery import get_tracker_discovery, TrackerCategory
            discovery = get_tracker_discovery()
            tracker_info = discovery.get_tracker_info(tracker_name)
            if not tracker_info:
                raise RuntimeError(f"Unknown tracker: {tracker_name}")

            # User-intervention trackers (draw-a-box flows like User ROI)
            # cannot run unattended in the watched folder worker.
            if tracker_info.requires_intervention or not tracker_info.supports_batch:
                raise RuntimeError(
                    f"Tracker '{tracker_info.display_name}' requires user intervention and cannot run in batch mode")

            # TOOL trackers (Oscillation, Chapter Maker, etc.) dispatch by the
            # base class they inherit, not by the UI-grouping category.
            runtime_category = discovery.get_runtime_category(tracker_name)

            # Open the video
            open_success = self.app.file_manager.open_video_from_path(video_path)
            if not open_success:
                raise RuntimeError(f"Failed to open video: {video_path}")

            # Give the video time to load
            time.sleep(1.0)
            if self._stop_event.is_set():
                return

            selected_mode = tracker_info.internal_name

            if runtime_category == TrackerCategory.OFFLINE:
                self._process_offline(selected_mode, video_basename)
            elif runtime_category == TrackerCategory.LIVE:
                self._process_live(selected_mode, video_basename)
            else:
                raise RuntimeError(
                    f"Unsupported tracker category for batch: {tracker_info.category}")

            self.queue.mark_completed(idx)
            logger.info(f"BatchWorker: Completed {video_basename}")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"BatchWorker: Failed {video_basename}: {error_msg}", exc_info=True)
            self.queue.mark_failed(idx, error_msg)

    def _process_offline(self, selected_mode: str, video_basename: str):
        """Process using an offline (stage-based) tracker."""
        app = self.app

        # Set processing speed to MAX_SPEED
        from config.constants import ProcessingSpeedMode
        original_speed = app.app_state_ui.selected_processing_speed_mode
        app.app_state_ui.selected_processing_speed_mode = ProcessingSpeedMode.MAX_SPEED

        try:
            app.single_video_analysis_complete_event.clear()
            app.save_and_reset_complete_event.clear()
            app.stage_processor.start_full_analysis(processing_mode=selected_mode)

            # Block until analysis completes (with periodic stop checks)
            while not app.single_video_analysis_complete_event.wait(timeout=2.0):
                if self._stop_event.is_set():
                    logger.info("BatchWorker: Stop requested during offline analysis")
                    return

            # For CLI mode, load results (mirroring batch thread logic)
            if not app.gui_instance:
                results_package = app.stage_processor.last_analysis_result
                if results_package and "results_dict" in results_package:
                    result_script = results_package["results_dict"].get("funscript")
                    if result_script:
                        app.funscript_processor.clear_timeline_history_and_set_new_baseline(
                            1, result_script.primary_actions, "Stage 2 (BatchWorker)")
                        app.funscript_processor.clear_timeline_history_and_set_new_baseline(
                            2, result_script.secondary_actions, "Stage 2 (BatchWorker)")
                app.on_offline_analysis_completed({"video_path": app.file_manager.video_path})

            # Wait for save/reset to complete
            app.save_and_reset_complete_event.wait(timeout=120)

        finally:
            app.app_state_ui.selected_processing_speed_mode = original_speed

    def _process_live(self, selected_mode: str, video_basename: str):
        """Process using a live (real-time) tracker."""
        app = self.app

        from config.constants import ProcessingSpeedMode
        original_speed = app.app_state_ui.selected_processing_speed_mode
        app.app_state_ui.selected_processing_speed_mode = ProcessingSpeedMode.MAX_SPEED

        try:
            app.tracker.set_tracking_mode(selected_mode)
            app.tracker.start_tracking()
            app.processor.set_tracker_processing_enabled(True)

            # Process the entire video
            app.processor.start_processing(start_frame=0, end_frame=-1)

            # Block until processing thread finishes
            proc_thread = getattr(app.processor, 'processing_thread', None)
            while proc_thread and proc_thread.is_alive():
                proc_thread.join(timeout=2.0)
                if self._stop_event.is_set():
                    logger.info("BatchWorker: Stop requested during live processing")
                    return

            # Post-processing and saving (same as batch thread)
            app.on_processing_stopped(was_scripting_session=True)

        finally:
            app.app_state_ui.selected_processing_speed_mode = original_speed
