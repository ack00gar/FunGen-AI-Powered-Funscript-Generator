from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ChapterBarInteractionState:
    is_resizing_chapter: bool = False
    resize_chapter_id: Optional[str] = None
    resize_edge: Optional[str] = None
    resize_preview_frame: Optional[int] = None
    resize_old_start: Optional[int] = None
    resize_old_end: Optional[int] = None

    is_dragging_chapter_range: bool = False
    drag_start_frame: int = 0
    drag_current_frame: int = 0

    context_menu_opened_at_frame: Optional[int] = None
    context_menu_opened_this_frame: bool = False

    last_chapter_count: int = -1
