"""
Watched Folder Processor

Cross-platform filesystem monitoring using watchdog. Automatically queues new
video files for processing when they appear in a watched directory.
"""

import os
import time
import logging
import threading
from typing import Optional, Callable, Set

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}


class WatchedFolderProcessor:
    """Monitors a folder for new video files and triggers processing."""

    def __init__(self, on_new_video: Optional[Callable[[str], None]] = None):
        self._observer = None
        self._watch_path: Optional[str] = None
        self._recursive: bool = False
        self._on_new_video = on_new_video
        self._is_watching: bool = False
        self._known_files: Set[str] = set()
        # Files seen but still being written; promoted to _known_files once
        # size has been stable for STABLE_S. Avoids handing the queue a
        # half-transferred file (gh#kaoszwerg).
        self._pending: Set[str] = set()
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    @property
    def is_watching(self) -> bool:
        return self._is_watching

    @property
    def watch_path(self) -> Optional[str]:
        return self._watch_path

    def start_watching(self, path: str, recursive: bool = False):
        """Start monitoring a folder for new video files."""
        if self._is_watching:
            self.stop_watching()

        if not os.path.isdir(path):
            logger.error(f"Watch path is not a directory: {path}")
            return

        self._watch_path = path
        self._recursive = recursive
        self._stop_event.clear()

        # Snapshot existing files so we only process new ones
        self._known_files = self._scan_existing_videos(path, recursive)
        self._pending.clear()
        logger.info(f"Found {len(self._known_files)} existing video(s) in watch folder")

        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler

            class _VideoHandler(FileSystemEventHandler):
                def __init__(self, processor):
                    self._processor = processor

                def on_created(self, event):
                    if event.is_directory:
                        return
                    self._processor._handle_file_event(event.src_path)

                def on_moved(self, event):
                    if event.is_directory:
                        return
                    self._processor._handle_file_event(event.dest_path)

            self._observer = Observer()
            handler = _VideoHandler(self)
            self._observer.schedule(handler, path, recursive=recursive)
            self._observer.start()
            self._is_watching = True
            logger.info(f"Watching folder: {path} (recursive={recursive})")

        except ImportError:
            logger.error("watchdog package not installed. Install with: pip install watchdog")
            self._is_watching = False
        except Exception as e:
            logger.error(f"Failed to start folder watcher: {e}")
            self._is_watching = False

    def stop_watching(self):
        """Stop monitoring the folder."""
        # Signal in-flight stability waiters before tearing down the observer
        # so they exit on their next poll instead of running to MAX_S.
        self._stop_event.set()
        if self._observer:
            try:
                self._observer.stop()
                self._observer.join(timeout=5)
            except Exception as e:
                logger.warning(f"Error stopping folder watcher: {e}")
            self._observer = None

        self._is_watching = False
        self._watch_path = None
        logger.info("Stopped folder watching")

    def _handle_file_event(self, filepath: str):
        """File event entry. Spawns a stability waiter so a transfer that's
        still in progress doesn't get queued mid-write."""
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in VIDEO_EXTENSIONS:
            return
        abs_path = os.path.abspath(filepath)
        with self._lock:
            if abs_path in self._known_files or abs_path in self._pending:
                return
            self._pending.add(abs_path)
        logger.info(f"New video detected (waiting for stable size): {os.path.basename(filepath)}")
        threading.Thread(
            target=self._wait_stable, args=(abs_path,),
            daemon=True, name="WatchStable").start()

    def _wait_stable(self, abs_path: str,
                     poll_s: float = 2.0,
                     stable_s: float = 5.0,
                     max_s: float = 600.0):
        """Promote `abs_path` to known + fire callback once size has been
        unchanged for `stable_s` and the file opens for read. Exits early on
        stop_event, vanished file, or `max_s` deadline."""
        last_size = -1
        last_change = time.time()
        deadline = last_change + max_s
        try:
            while not self._stop_event.is_set() and time.time() < deadline:
                try:
                    size = os.path.getsize(abs_path)
                except OSError:
                    logger.warning(f"Watched file vanished mid-transfer: {abs_path}")
                    return
                if size != last_size:
                    last_size = size
                    last_change = time.time()
                elif size > 0 and (time.time() - last_change) >= stable_s:
                    try:
                        with open(abs_path, "rb") as f:
                            f.read(1)
                    except OSError:
                        # Writer still holds an exclusive lock; keep waiting.
                        last_change = time.time()
                    else:
                        with self._lock:
                            self._known_files.add(abs_path)
                        logger.info(f"Watched file stable, queueing: {os.path.basename(abs_path)}")
                        if self._on_new_video:
                            self._on_new_video(abs_path)
                        return
                self._stop_event.wait(poll_s)
            if not self._stop_event.is_set():
                logger.warning(f"Gave up waiting for stable size: {abs_path}")
        finally:
            with self._lock:
                self._pending.discard(abs_path)

    def _scan_existing_videos(self, path: str, recursive: bool) -> Set[str]:
        """Scan existing video files in the watch path."""
        files = set()
        if recursive:
            for root, _, filenames in os.walk(path):
                for f in filenames:
                    if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS:
                        files.add(os.path.abspath(os.path.join(root, f)))
        else:
            for f in os.listdir(path):
                if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS:
                    files.add(os.path.abspath(os.path.join(path, f)))
        return files
