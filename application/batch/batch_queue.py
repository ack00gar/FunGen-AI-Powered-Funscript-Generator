"""
Batch Queue with Pause/Resume

Priority queue wrapper around FunGen's existing batch processing that adds:
- Priority ordering
- Pause/resume via threading.Event
- Per-item status tracking
- Estimated completion time
"""

import time
import logging
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum, auto

logger = logging.getLogger(__name__)


class BatchItemStatus(Enum):
    QUEUED = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass
class BatchItem:
    """A single item in the batch queue."""
    video_path: str
    priority: int = 0  # Lower = higher priority
    tracker_name: Optional[str] = None
    status: BatchItemStatus = BatchItemStatus.QUEUED
    error_message: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    override_format: str = "Auto (Heuristic)"

    @property
    def processing_time(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return time.time() - self.start_time
        return 0.0


class BatchQueue:
    """Priority batch queue with pause/resume support."""

    def __init__(self):
        self._items: List[BatchItem] = []
        self._lock = threading.Lock()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused initially
        self._processing_times: List[float] = []
        self._current_index: int = -1

    @property
    def is_paused(self) -> bool:
        return not self._pause_event.is_set()

    @property
    def items(self) -> List[BatchItem]:
        with self._lock:
            return list(self._items)

    @property
    def current_item(self) -> Optional[BatchItem]:
        with self._lock:
            if 0 <= self._current_index < len(self._items):
                return self._items[self._current_index]
            return None

    @property
    def total_count(self) -> int:
        return len(self._items)

    @property
    def completed_count(self) -> int:
        return sum(1 for item in self._items
                   if item.status in (BatchItemStatus.COMPLETED, BatchItemStatus.SKIPPED))

    @property
    def failed_count(self) -> int:
        return sum(1 for item in self._items if item.status == BatchItemStatus.FAILED)

    def add(self, video_path: str, priority: int = 0,
            tracker_name: Optional[str] = None) -> BatchItem:
        """Add a video to the batch queue."""
        item = BatchItem(
            video_path=video_path,
            priority=priority,
            tracker_name=tracker_name,
        )
        with self._lock:
            self._items.append(item)
            # Sort by priority (lower = higher priority)
            self._items.sort(key=lambda x: x.priority)
        logger.info(f"Added to batch queue: {video_path} (priority={priority})")
        return item

    def add_multiple(self, video_paths: List[str], priority: int = 0,
                     tracker_name: Optional[str] = None):
        """Add multiple videos to the batch queue."""
        with self._lock:
            for path in video_paths:
                item = BatchItem(
                    video_path=path,
                    priority=priority,
                    tracker_name=tracker_name,
                )
                self._items.append(item)
            self._items.sort(key=lambda x: x.priority)
        logger.info(f"Added {len(video_paths)} videos to batch queue")

    def clear(self):
        """Clear all queued items (does not stop processing)."""
        with self._lock:
            self._items.clear()
            self._current_index = -1
            self._processing_times.clear()

    def pause(self):
        """Pause batch processing."""
        self._pause_event.clear()
        logger.info("Batch processing paused")

    def resume(self):
        """Resume batch processing."""
        self._pause_event.set()
        logger.info("Batch processing resumed")

    def wait_if_paused(self, timeout: float = 1.0) -> bool:
        """Wait if paused. Returns True if not paused, False if still paused after timeout."""
        return self._pause_event.wait(timeout=timeout)

    def mark_processing(self, index: int):
        """Mark an item as currently processing."""
        with self._lock:
            if 0 <= index < len(self._items):
                self._items[index].status = BatchItemStatus.PROCESSING
                self._items[index].start_time = time.time()
                self._current_index = index

    def mark_completed(self, index: int):
        """Mark an item as completed."""
        with self._lock:
            if 0 <= index < len(self._items):
                item = self._items[index]
                item.status = BatchItemStatus.COMPLETED
                item.end_time = time.time()
                if item.start_time:
                    self._processing_times.append(item.end_time - item.start_time)

    def mark_failed(self, index: int, error: str = ""):
        """Mark an item as failed."""
        with self._lock:
            if 0 <= index < len(self._items):
                self._items[index].status = BatchItemStatus.FAILED
                self._items[index].end_time = time.time()
                self._items[index].error_message = error

    def mark_skipped(self, index: int):
        """Mark an item as skipped."""
        with self._lock:
            if 0 <= index < len(self._items):
                self._items[index].status = BatchItemStatus.SKIPPED
                self._items[index].end_time = time.time()

    def estimated_remaining_time(self) -> float:
        """Estimate remaining processing time in seconds."""
        if not self._processing_times:
            return 0.0
        avg_time = sum(self._processing_times) / len(self._processing_times)
        remaining = sum(1 for item in self._items if item.status == BatchItemStatus.QUEUED)
        return avg_time * remaining

    def to_batch_video_paths(self) -> List[Dict]:
        """Convert queue to the format expected by app_logic batch processing."""
        return [
            {"path": item.video_path, "override_format": item.override_format}
            for item in self._items
            if item.status == BatchItemStatus.QUEUED
        ]

    def get_status_summary(self) -> Dict:
        """Get a summary of queue status."""
        return {
            "total": self.total_count,
            "completed": self.completed_count,
            "failed": self.failed_count,
            "queued": sum(1 for i in self._items if i.status == BatchItemStatus.QUEUED),
            "processing": sum(1 for i in self._items if i.status == BatchItemStatus.PROCESSING),
            "paused": self.is_paused,
            "estimated_remaining_s": self.estimated_remaining_time(),
        }
