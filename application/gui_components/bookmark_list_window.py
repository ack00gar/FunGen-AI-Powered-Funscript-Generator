"""Bookmark List Window — table view of all bookmarks with CRUD actions.

Follows the same pattern as ChapterListWindow: standalone imgui window
opened from Edit > Bookmark List menu item.
"""
import imgui
from typing import Optional

from common.frame_utils import frame_to_ms


class BookmarkListWindow:
    """Standalone window for managing timeline bookmarks."""

    def __init__(self, app, gui):
        self.app = app
        self.gui = gui
        self._rename_id: Optional[str] = None
        self._rename_buf: str = ""
        self._add_name_buf: str = ""
        self._show_add_row: bool = False

    # ----- helpers -----

    def _get_all_bookmark_managers(self):
        """Return list of (timeline_num, bookmark_manager) tuples."""
        managers = []
        gui = self.gui
        if gui:
            tl1 = getattr(gui, 'timeline_editor1', None)
            if tl1:
                bm = getattr(tl1, '_bookmark_manager', None)
                if bm:
                    managers.append((1, bm))
            tl2 = getattr(gui, 'timeline_editor2', None)
            if tl2:
                bm = getattr(tl2, '_bookmark_manager', None)
                if bm:
                    managers.append((2, bm))
            extras = getattr(gui, '_extra_timeline_editors', {})
            for tl_num, editor in sorted(extras.items()):
                bm = getattr(editor, '_bookmark_manager', None)
                if bm:
                    managers.append((tl_num, bm))
        return managers

    def _collect_all_bookmarks(self):
        """Collect bookmarks from all timelines into a flat sorted list."""
        items = []
        for tl_num, mgr in self._get_all_bookmark_managers():
            for bm in mgr.bookmarks:
                items.append((tl_num, mgr, bm))
        items.sort(key=lambda x: x[2].time_ms)
        return items

    def _format_time(self, time_ms):
        secs = time_ms / 1000.0
        m, s = divmod(secs, 60)
        return f"{int(m)}:{s:05.2f}"

    def _seek(self, time_ms):
        proc = self.app.processor
        if proc and proc.is_video_open() and proc.fps > 0:
            frame_idx = int((time_ms / 1000.0) * proc.fps)
            frame_idx = max(0, min(frame_idx, proc.total_frames - 1))
            self.app.event_handlers.seek_video_with_sync(frame_idx)

    # ----- render -----

    def render(self):
        app_state = self.app.app_state_ui
        if not getattr(app_state, 'show_bookmark_list_window', False):
            return

        window_flags = imgui.WINDOW_NO_COLLAPSE
        imgui.set_next_window_size(550, 350, condition=imgui.FIRST_USE_EVER)

        is_open, app_state.show_bookmark_list_window = imgui.begin(
            "Bookmark List##BookmarkListWindow",
            closable=True,
            flags=window_flags
        )

        if not is_open:
            imgui.end()
            return

        all_bookmarks = self._collect_all_bookmarks()

        # --- Action buttons ---
        if imgui.button("Add Bookmark at Playhead"):
            proc = self.app.processor
            if proc and proc.is_video_open() and proc.fps > 0:
                current_time_ms = frame_to_ms(proc.current_frame_index, proc.fps)
                managers = self._get_all_bookmark_managers()
                if managers:
                    _, mgr = managers[0]  # Add to timeline 1
                    mgr.add(current_time_ms)

        imgui.same_line()
        if all_bookmarks:
            if imgui.button("Clear All"):
                for _, mgr in self._get_all_bookmark_managers():
                    mgr.clear()
                all_bookmarks = []

        imgui.separator()

        # --- Table ---
        if not all_bookmarks:
            imgui.text_colored("No bookmarks. Press B on timeline or click 'Add Bookmark at Playhead'.",
                               0.5, 0.5, 0.5, 1.0)
        else:
            table_flags = (imgui.TABLE_BORDERS | imgui.TABLE_RESIZABLE |
                           imgui.TABLE_SIZING_STRETCH_PROP | imgui.TABLE_ROW_BACKGROUND)
            if imgui.begin_table("##BookmarkTable", 5, table_flags):
                imgui.table_setup_column("Time", imgui.TABLE_COLUMN_WIDTH_FIXED, 80)
                imgui.table_setup_column("Name", imgui.TABLE_COLUMN_WIDTH_STRETCH)
                imgui.table_setup_column("Timeline", imgui.TABLE_COLUMN_WIDTH_FIXED, 55)
                imgui.table_setup_column("Color", imgui.TABLE_COLUMN_WIDTH_FIXED, 30)
                imgui.table_setup_column("Actions", imgui.TABLE_COLUMN_WIDTH_FIXED, 140)
                imgui.table_headers_row()

                delete_target = None

                for tl_num, mgr, bm in all_bookmarks:
                    imgui.table_next_row()

                    # Time
                    imgui.table_next_column()
                    if imgui.selectable(
                        f"{self._format_time(bm.time_ms)}##{bm.id}",
                        False
                    )[0]:
                        self._seek(bm.time_ms)

                    # Name (inline rename)
                    imgui.table_next_column()
                    if self._rename_id == bm.id:
                        imgui.push_item_width(-1)
                        enter, self._rename_buf = imgui.input_text(
                            f"##rename_{bm.id}", self._rename_buf, 128,
                            imgui.INPUT_TEXT_ENTER_RETURNS_TRUE
                        )
                        imgui.pop_item_width()
                        if enter:
                            mgr.rename(bm.id, self._rename_buf)
                            self._rename_id = None
                    else:
                        imgui.text(bm.name or "(unnamed)")

                    # Timeline
                    imgui.table_next_column()
                    imgui.text(f"T{tl_num}")

                    # Color swatch
                    imgui.table_next_column()
                    dl = imgui.get_window_draw_list()
                    cx, cy = imgui.get_cursor_screen_pos()
                    col_u32 = imgui.get_color_u32_rgba(*bm.color)
                    dl.add_rect_filled(cx, cy, cx + 16, cy + 14, col_u32, 2.0)
                    imgui.dummy(16, 14)

                    # Actions
                    imgui.table_next_column()
                    if imgui.small_button(f"Go##{bm.id}"):
                        self._seek(bm.time_ms)
                    imgui.same_line()
                    if imgui.small_button(f"Rename##{bm.id}"):
                        self._rename_id = bm.id
                        self._rename_buf = bm.name
                    imgui.same_line()
                    if imgui.small_button(f"Delete##{bm.id}"):
                        delete_target = (mgr, bm.id)

                imgui.end_table()

                # Deferred delete to avoid mutation during iteration
                if delete_target:
                    delete_target[0].remove(delete_target[1])

        # --- Rename popup cancel on Escape ---
        if self._rename_id is not None and imgui.is_key_pressed(imgui.KEY_ESCAPE):
            self._rename_id = None

        imgui.end()
