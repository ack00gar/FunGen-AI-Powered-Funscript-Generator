"""
Watched Folder Processor

Cross-platform filesystem monitoring using watchdog. Automatically queues new
video files for processing when they appear in a watched directory.
"""

import os
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

        # Snapshot existing files so we only process new ones
        self._known_files = self._scan_existing_videos(path, recursive)
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
        """Handle a new/moved file event."""
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in VIDEO_EXTENSIONS:
            return

        with self._lock:
            abs_path = os.path.abspath(filepath)
            if abs_path in self._known_files:
                return
            self._known_files.add(abs_path)

        logger.info(f"New video detected: {os.path.basename(filepath)}")
        if self._on_new_video:
            self._on_new_video(abs_path)

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
