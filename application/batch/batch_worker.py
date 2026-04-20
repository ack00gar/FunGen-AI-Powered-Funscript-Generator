"""
Batch Worker

Background worker that polls the BatchQueue and processes items.

Two modes:
- Sequential (max_parallel=1, default): one item at a time in-process, reusing
  the app's tracker/processor/stage_processor. Live trackers require this mode.
- Parallel (max_parallel>1): offline batch-compatible items are launched as
  independent `main.py` subprocesses with a concurrency cap. Live / intervention
  trackers still go through the sequential path.
"""

import os
import sys
import subprocess
import time
import logging
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    """Locate the FunGen repo root (where main.py lives)."""
    return Path(__file__).resolve().parents[2]


class BatchWorker:
    """Background queue processor for watched-folder batch items."""

    def __init__(self, app, queue, max_parallel: int = 1):
        """
        Args:
            app: The FunGen ApplicationLogic instance.
            queue: A BatchQueue instance to pull items from.
            max_parallel: Concurrency cap. 1 = sequential (default, unchanged
                behavior). >1 = spawn up to N `main.py` subprocesses in parallel
                for offline batch-compatible items.
        """
        self.app = app
        self.queue = queue
        self.max_parallel = max(1, int(max_parallel))
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._inflight: Dict[int, subprocess.Popen] = {}

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
        """Main worker loop. Dispatches to sequential or parallel path."""
        if self.max_parallel > 1:
            self._run_parallel()
        else:
            self._run_sequential()
        logger.info("BatchWorker stopped")

    def _run_sequential(self):
        """Poll queue, process one item at a time in-process."""
        while not self._stop_event.is_set():
            if not self.queue.wait_if_paused(timeout=1.0):
                continue
            if self._stop_event.is_set():
                break

            next_idx = self._find_next_queued()
            if next_idx is None:
                self._stop_event.wait(timeout=2.0)
                continue

            if self._app_is_busy():
                self._stop_event.wait(timeout=3.0)
                continue

            self._process_item(next_idx)

    def _run_parallel(self):
        """Launch up to max_parallel `main.py` subprocesses concurrently.

        Each queued item is eligible if its tracker is offline and
        batch-compatible. Live / intervention trackers are skipped here (they
        require the live app state and cannot be driven from CLI unattended).
        """
        logger.info(f"BatchWorker: parallel mode, max_parallel={self.max_parallel}")
        while not self._stop_event.is_set():
            if not self.queue.wait_if_paused(timeout=1.0):
                continue
            if self._stop_event.is_set():
                break

            self._reap_finished_subprocesses()

            # Fill up to max_parallel as long as the app isn't running its own
            # in-process analysis (Run tab). If the app is busy we stay idle.
            if not self._app_is_busy():
                while len(self._inflight) < self.max_parallel:
                    if self._stop_event.is_set():
                        break
                    idx = self._find_next_eligible_for_subprocess()
                    if idx is None:
                        break
                    if not self._start_subprocess(idx):
                        break

            if not self._inflight:
                self._stop_event.wait(timeout=1.5)
            else:
                self._stop_event.wait(timeout=0.5)

        self._terminate_all_inflight()

    def _find_next_eligible_for_subprocess(self) -> Optional[int]:
        """Next QUEUED item whose tracker can run unattended via CLI."""
        from application.batch.batch_queue import BatchItemStatus
        from config.tracker_discovery import get_tracker_discovery
        tracker_name = getattr(self.app.app_state_ui, 'selected_tracker_name', '')
        if not tracker_name:
            return None
        info = get_tracker_discovery().get_tracker_info(tracker_name)
        if not info or info.requires_intervention or not info.supports_batch or not info.cli_aliases:
            return None
        for i, item in enumerate(self.queue.items):
            if item.status == BatchItemStatus.QUEUED and i not in self._inflight:
                return i
        return None

    def _start_subprocess(self, idx: int) -> bool:
        """Launch `main.py <video> --mode <cli_alias> --quiet` as a subprocess.

        Returns True if the process was started, False otherwise. On success
        the item is marked PROCESSING and the Popen is tracked in _inflight.
        """
        from application.batch.batch_queue import BatchItemStatus
        from config.tracker_discovery import get_tracker_discovery

        item = self.queue.items[idx]
        if item.status != BatchItemStatus.QUEUED:
            return False

        tracker_name = getattr(self.app.app_state_ui, 'selected_tracker_name', '')
        info = get_tracker_discovery().get_tracker_info(tracker_name)
        if not info or not info.cli_aliases:
            self.queue.mark_failed(idx, f"Unknown tracker: {tracker_name}")
            return False
        cli_alias = info.cli_aliases[0]

        repo = _repo_root()
        main_py = repo / "main.py"
        if not main_py.exists():
            self.queue.mark_failed(idx, f"main.py not found at {main_py}")
            return False

        # --overwrite: sequential batch always re-processes queued items;
        # mirror that here so parallel mode does not silently skip videos that
        # already have a sibling .funscript.
        cmd = [sys.executable, str(main_py), item.video_path,
               "--mode", cli_alias, "--quiet", "--overwrite"]
        env = os.environ.copy()
        # Signal to the subprocess how many siblings are running concurrently
        # so its Stage 1 pool can down-scale (see app_stage_executor.py).
        env["FUNGEN_BATCH_PARALLEL"] = str(self.max_parallel)

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(repo),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception as e:
            self.queue.mark_failed(idx, f"subprocess launch failed: {e}")
            return False

        self._inflight[idx] = proc
        self.queue.mark_processing(idx)
        logger.info(f"BatchWorker: launched subprocess pid={proc.pid} for {os.path.basename(item.video_path)}")
        return True

    def _reap_finished_subprocesses(self):
        """Poll inflight subprocesses; mark completed or failed based on exit code."""
        done: list[Tuple[int, subprocess.Popen]] = []
        for idx, proc in list(self._inflight.items()):
            rc = proc.poll()
            if rc is not None:
                done.append((idx, proc))
        for idx, proc in done:
            del self._inflight[idx]
            if proc.returncode == 0:
                self.queue.mark_completed(idx)
                logger.info(f"BatchWorker: subprocess pid={proc.pid} completed idx={idx}")
            else:
                self.queue.mark_failed(idx, f"subprocess exit code {proc.returncode}")
                logger.warning(f"BatchWorker: subprocess pid={proc.pid} failed idx={idx} rc={proc.returncode}")

    def _terminate_all_inflight(self):
        """Terminate inflight subprocesses cleanly; mark their items as failed."""
        if not self._inflight:
            return
        logger.info(f"BatchWorker: terminating {len(self._inflight)} inflight subprocess(es)")
        for proc in self._inflight.values():
            try:
                proc.terminate()
            except Exception:
                pass
        deadline = time.time() + 5.0
        for idx, proc in list(self._inflight.items()):
            timeout = max(0.0, deadline - time.time())
            try:
                proc.wait(timeout=timeout)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            self.queue.mark_failed(idx, "interrupted")
        self._inflight.clear()

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
