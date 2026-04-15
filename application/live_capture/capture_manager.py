"""
Capture Manager

Reads raw BGR24 frames from a capture source and feeds them directly
into FunGen's video display (app.processor.current_frame) so the
captured content appears in the main video viewer.

When a tracker_name is provided, creates a dedicated TrackerManager that
processes each frame in real-time (producing overlays, accumulating
funscript actions, and optionally driving connected devices).
"""

import os
import json
import time
import logging
import threading
import numpy as np
from datetime import datetime
from typing import Optional
from enum import Enum, auto

logger = logging.getLogger(__name__)


class CaptureState(Enum):
    IDLE = auto()
    CAPTURING = auto()
    ERROR = auto()


class CaptureManager:
    """Reads frames from a CaptureSource and pushes them into the video display."""

    def __init__(self, width: int = 640, height: int = 640, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps

        self._app = None
        self._source = None
        self._state = CaptureState.IDLE
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Tracker integration
        self._tracker_manager = None
        self._original_tracker = None  # Saved app tracker during capture
        self._tracker_name: str = ""
        self._device_control_enabled: bool = False
        self._save_funscript: bool = True

        # Stats
        self._frames_captured: int = 0
        self._capture_start_time: float = 0.0
        self._last_fps: float = 0.0
        self._error_message: str = ""
        self._last_save_path: str = ""

    @property
    def state(self) -> CaptureState:
        return self._state

    @property
    def frames_captured(self) -> int:
        return self._frames_captured

    @property
    def current_fps(self) -> float:
        return self._last_fps

    @property
    def error_message(self) -> str:
        return self._error_message

    @property
    def is_capturing(self) -> bool:
        return self._state == CaptureState.CAPTURING

    @property
    def last_save_path(self) -> str:
        return self._last_save_path

    def start(self, source, app, tracker_name: str = "",
              device_control: bool = False, save_funscript: bool = True) -> bool:
        """Start capturing from the given source.

        Args:
            source: A CaptureSource instance.
            app: The FunGen app instance (needs app.processor with
                 current_frame, current_frame_index, frame_lock).
            tracker_name: Internal name of the tracker to run on frames.
                          Empty string means display-only (no tracking).
            device_control: Whether to enable live device control.
            save_funscript: Whether to save funscript when capture stops.
        """
        if self._state == CaptureState.CAPTURING:
            logger.warning("Capture already running")
            return False

        self._app = app
        self._source = source
        self._tracker_name = tracker_name
        self._device_control_enabled = device_control
        self._save_funscript = save_funscript
        self._stop_event.clear()
        self._frames_captured = 0
        self._error_message = ""
        self._last_save_path = ""
        self._capture_start_time = time.time()

        # Create a dedicated TrackerManager for live capture if tracker specified
        self._tracker_manager = None
        if tracker_name:
            try:
                self._init_tracker(app, tracker_name, device_control)
            except Exception as e:
                self._error_message = f"Failed to init tracker: {e}"
                self._state = CaptureState.ERROR
                logger.error(self._error_message, exc_info=True)
                return False

        try:
            self._source.start()
        except Exception as e:
            self._error_message = f"Failed to start capture: {e}"
            self._state = CaptureState.ERROR
            logger.error(self._error_message)
            self._cleanup_tracker()
            return False

        self._state = CaptureState.CAPTURING
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="CaptureManager",
        )
        self._capture_thread.start()
        tracker_info = f" + tracker '{tracker_name}'" if tracker_name else ""
        logger.info(f"Capture started: {self.width}x{self.height} @ {self.fps}fps{tracker_info}")
        return True

    def stop(self):
        """Stop capturing and save funscript if tracking was active."""
        if self._state == CaptureState.IDLE:
            return

        self._stop_event.set()

        if self._source:
            self._source.stop()

        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=10)

        # Stop tracker and save funscript
        if self._tracker_manager:
            try:
                self._stop_tracker_and_save()
            except Exception as e:
                logger.error(f"Error saving capture funscript: {e}", exc_info=True)
            self._cleanup_tracker()

        # Clear live capture flag and resume video playback
        if self._app:
            processor = getattr(self._app, 'processor', None)
            if processor:
                processor.live_capture_active = False
            if processor and hasattr(processor, 'pause_event') and processor.is_processing:
                if processor.pause_event.is_set():
                    processor.pause_event.clear()
                    logger.info("Video playback resumed after capture stopped")

        self._state = CaptureState.IDLE
        logger.info(f"Capture stopped. Frames: {self._frames_captured}")

    def get_status(self) -> dict:
        """Get current capture status for UI display."""
        elapsed = time.time() - self._capture_start_time if self._capture_start_time else 0
        return {
            "state": self._state.name,
            "frames_captured": self._frames_captured,
            "fps": self._last_fps,
            "elapsed_s": elapsed,
            "error": self._error_message,
            "tracker_active": self._tracker_manager is not None and getattr(
                self._tracker_manager, 'tracking_active', False),
            "last_save_path": self._last_save_path,
        }

    def _init_tracker(self, app, tracker_name: str, device_control: bool):
        """Create a dedicated TrackerManager for live capture."""
        from tracker.tracker_manager import create_tracker_manager

        model_path = getattr(app, 'yolo_detection_model_path_setting', '') or ''
        self._tracker_manager = create_tracker_manager(app, model_path)

        if not self._tracker_manager.set_tracking_mode(tracker_name):
            raise RuntimeError(f"Failed to set tracking mode: {tracker_name}")

        # Enable device control if requested and device_manager exists
        if device_control and getattr(app, 'device_manager', None):
            self._tracker_manager.live_device_control_enabled = True
            logger.info("Live device control enabled for capture")

        self._tracker_manager.start_tracking()

        # Wire capture tracker to app's processor so timeline displays live actions
        processor = getattr(app, 'processor', None)
        if processor:
            self._original_tracker = getattr(processor, 'tracker', None)
            processor.tracker = self._tracker_manager
            processor.live_capture_active = True
            logger.info("Capture tracker wired to app processor (timeline will show live actions)")

        logger.info(f"Capture tracker initialized: {tracker_name}")

    def _cleanup_tracker(self):
        """Clean up the tracker manager.

        Stops tracking but preserves the funscript on the app's processor
        so the timeline continues to display the captured actions.
        Does NOT call tracker.cleanup() — that would wipe the funscript.
        """
        if self._tracker_manager:
            try:
                if self._tracker_manager.tracking_active:
                    self._tracker_manager.stop_tracking()
                # Clean up tracker internals (model, detector) without wiping funscript
                self._tracker_manager._cleanup_current_tracker()
            except Exception as e:
                logger.warning(f"Tracker cleanup error: {e}")
            # Don't set self._tracker_manager = None — it's still referenced
            # by app.processor.tracker, keeping the funscript on the timeline

    def _stop_tracker_and_save(self):
        """Stop tracking and save the accumulated funscript."""
        tm = self._tracker_manager
        if not tm:
            return

        if tm.tracking_active:
            tm.stop_tracking()

        if not self._save_funscript:
            return

        funscript = tm.funscript
        if not funscript:
            return

        primary_actions = funscript.primary_actions
        secondary_actions = funscript.secondary_actions

        if not primary_actions:
            logger.info("No funscript actions recorded during capture")
            return

        # Build output path
        output_dir = self._get_output_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.funscript"
        output_path = os.path.join(output_dir, filename)

        # Build funscript data
        funscript_data = {
            "actions": primary_actions,
            "metadata": {
                "creator": "FunGen Live Capture",
                "type": "basic",
                "range": 100,
                "tracker": self._tracker_name,
                "duration_s": time.time() - self._capture_start_time,
                "frames": self._frames_captured,
            }
        }

        try:
            os.makedirs(output_dir, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(funscript_data, f, indent=2)
            self._last_save_path = output_path
            logger.info(f"Saved capture funscript ({len(primary_actions)} actions) to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save capture funscript: {e}")

        # Also save secondary axis if present
        if secondary_actions:
            secondary_path = os.path.join(output_dir, f"capture_{timestamp}.roll.funscript")
            secondary_data = {
                "actions": secondary_actions,
                "metadata": {
                    "creator": "FunGen Live Capture",
                    "type": "roll",
                    "range": 100,
                    "tracker": self._tracker_name,
                }
            }
            try:
                with open(secondary_path, 'w') as f:
                    json.dump(secondary_data, f, indent=2)
                logger.info(f"Saved secondary axis ({len(secondary_actions)} actions) to: {secondary_path}")
            except Exception as e:
                logger.error(f"Failed to save secondary funscript: {e}")

    def _get_output_dir(self) -> str:
        """Get the output directory for capture funscripts."""
        if self._app:
            output_base = self._app.app_settings.get("output_folder_path", "")
            if output_base and os.path.isdir(output_base):
                capture_dir = os.path.join(output_base, "live_captures")
                return capture_dir
        # Fallback to a captures directory next to the app
        return os.path.join(os.getcwd(), "output", "live_captures")

    def _capture_loop(self):
        """Read frames from source and push into app.processor.current_frame."""
        frame_size = self.width * self.height * 3  # BGR24
        fps_counter = 0
        fps_timer = time.time()

        processor = getattr(self._app, 'processor', None)
        if processor is None:
            self._error_message = "No video processor available"
            self._state = CaptureState.ERROR
            return

        frame_lock = getattr(processor, 'frame_lock', None)
        if frame_lock is None:
            self._error_message = "Video processor has no frame_lock"
            self._state = CaptureState.ERROR
            return

        tracker = self._tracker_manager

        while not self._stop_event.is_set():
            if not self._source.is_running:
                if not self._stop_event.is_set():
                    self._error_message = "Capture source stopped unexpectedly"
                    self._state = CaptureState.ERROR
                break

            try:
                raw = self._source._process.stdout.read(frame_size)
            except Exception as e:
                if not self._stop_event.is_set():
                    self._error_message = f"Read error: {e}"
                    self._state = CaptureState.ERROR
                break

            if not raw or len(raw) < frame_size:
                if self._stop_event.is_set():
                    break
                # Try to get process exit info for better diagnostics
                proc = self._source._process
                exit_code = proc.poll() if proc else None
                if exit_code is not None:
                    self._error_message = f"Capture process exited (code {exit_code})"
                else:
                    self._error_message = "Capture source returned incomplete frame"
                self._state = CaptureState.ERROR
                break

            # Convert raw bytes to numpy
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                self.height, self.width, 3
            )

            # Run tracker on frame if active
            display_frame = frame
            if tracker and tracker.tracking_active:
                try:
                    timestamp_ms = int(self._frames_captured * (1000.0 / self.fps))
                    processed, action_log = tracker.process_frame(
                        frame.copy(), timestamp_ms, self._frames_captured)
                    if processed is not None:
                        display_frame = processed
                except Exception as e:
                    # Don't crash the capture loop on tracker errors
                    if self._frames_captured % 300 == 0:  # Log every ~10s at 30fps
                        logger.warning(f"Tracker frame error: {e}")

            with frame_lock:
                processor.current_frame = display_frame
                processor.current_frame_index = self._frames_captured

            self._frames_captured += 1
            fps_counter += 1

            # Update FPS every second
            now = time.time()
            if now - fps_timer >= 1.0:
                self._last_fps = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now
