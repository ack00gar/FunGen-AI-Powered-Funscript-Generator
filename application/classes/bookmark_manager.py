"""Bookmark manager for named navigation markers on the timeline.

Bookmarks are time-stamped markers with optional names and colors,
allowing quick navigation to important points in the script.
"""
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class Bookmark:
    time_ms: float
    name: str = ""
    color: tuple = (0.9, 0.7, 0.2, 1.0)  # Default: gold
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


class BookmarkManager:
    """Manages a list of timeline bookmarks with CRUD operations."""

    def __init__(self):
        self._bookmarks: List[Bookmark] = []

    @property
    def bookmarks(self) -> List[Bookmark]:
        return self._bookmarks

    def add(self, time_ms: float, name: str = "", color: tuple = None) -> Bookmark:
        """Add a new bookmark at the given time."""
        bm = Bookmark(
            time_ms=time_ms,
            name=name,
            color=color or (0.9, 0.7, 0.2, 1.0),
        )
        self._bookmarks.append(bm)
        self._bookmarks.sort(key=lambda b: b.time_ms)
        return bm

    def remove(self, bookmark_id: str) -> bool:
        """Remove a bookmark by ID. Returns True if found and removed."""
        for i, bm in enumerate(self._bookmarks):
            if bm.id == bookmark_id:
                self._bookmarks.pop(i)
                return True
        return False

    def rename(self, bookmark_id: str, new_name: str) -> bool:
        """Rename a bookmark by ID."""
        for bm in self._bookmarks:
            if bm.id == bookmark_id:
                bm.name = new_name
                return True
        return False

    def get_in_range(self, start_ms: float, end_ms: float) -> List[Bookmark]:
        """Return bookmarks within the given time range."""
        return [bm for bm in self._bookmarks
                if start_ms <= bm.time_ms <= end_ms]

    def get_nearest(self, time_ms: float, direction: int = 0) -> Optional[Bookmark]:
        """Get nearest bookmark to given time.

        Args:
            time_ms: Reference time
            direction: 0=closest, 1=next, -1=previous
        """
        if not self._bookmarks:
            return None

        if direction == 0:
            return min(self._bookmarks, key=lambda b: abs(b.time_ms - time_ms))
        elif direction == 1:
            candidates = [b for b in self._bookmarks if b.time_ms > time_ms]
            return candidates[0] if candidates else None
        else:
            candidates = [b for b in self._bookmarks if b.time_ms < time_ms]
            return candidates[-1] if candidates else None

    def clear(self):
        """Remove all bookmarks."""
        self._bookmarks.clear()

    def to_dict(self) -> List[Dict]:
        """Serialize bookmarks to a list of dicts for project save."""
        return [
            {
                'time_ms': bm.time_ms,
                'name': bm.name,
                'color': list(bm.color),
                'id': bm.id,
            }
            for bm in self._bookmarks
        ]

    @classmethod
    def from_dict(cls, data: List[Dict]) -> 'BookmarkManager':
        """Deserialize bookmarks from project data."""
        mgr = cls()
        if not data:
            return mgr
        for d in data:
            bm = Bookmark(
                time_ms=d.get('time_ms', 0),
                name=d.get('name', ''),
                color=tuple(d.get('color', (0.9, 0.7, 0.2, 1.0))),
                id=d.get('id', str(uuid.uuid4())[:8]),
            )
            mgr._bookmarks.append(bm)
        mgr._bookmarks.sort(key=lambda b: b.time_ms)
        return mgr
