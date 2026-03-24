import imgui
from typing import Optional

from application.utils import _format_time, VideoSegment, get_icon_texture_manager, primary_button_style, destructive_button_style
from application.utils.imgui_helpers import DisabledScope
from config.constants import POSITION_INFO_MAPPING, DEFAULT_CHAPTER_FPS
from config.element_group_colors import VideoNavigationColors
from config.constants_colors import CurrentTheme



class ChapterListWindow:
    def __init__(self, app, nav_ui):
        self.app = app
        self.nav_ui = nav_ui
        self.list_context_selected_chapters = []

        # Thumbnail cache for chapter previews
        from application.classes.chapter_thumbnail_cache import ChapterThumbnailCache
        self.thumbnail_cache = ChapterThumbnailCache(app, thumbnail_height=60)

    def render(self):
        app_state = self.app.app_state_ui
        if not hasattr(app_state, 'show_chapter_list_window') or not app_state.show_chapter_list_window:
            return

        window_flags = imgui.WINDOW_NO_COLLAPSE
        imgui.set_next_window_size(850, 400, condition=imgui.APPEARING)

        is_open, app_state.show_chapter_list_window = imgui.begin(
            "Chapter List##ChapterListWindow",
            closable=True,
            flags=window_flags
        )

        if is_open:
            fs_proc = self.app.funscript_processor
            if not fs_proc:
                imgui.text("Funscript processor not available.")
                imgui.end()
                return

            # --- RENDER ACTION BUTTONS ---
            num_selected = len(self.list_context_selected_chapters)

            # --- Merge Button ---
            can_merge = num_selected == 2
            with DisabledScope(not can_merge):
                with primary_button_style():
                    if imgui.button("Merge Selected"):
                        if can_merge:
                            chaps_to_merge = sorted(self.list_context_selected_chapters, key=lambda c: c.start_frame_id)
                            fs_proc.merge_selected_chapters(chaps_to_merge[0], chaps_to_merge[1])
                            self.list_context_selected_chapters.clear()
            if imgui.is_item_hovered():
                imgui.set_tooltip("Select exactly two chapters to merge.")

            imgui.same_line()

            # --- Track Gap & Merge Button ---
            can_track_gap, gap_c1, gap_c2, _, _ = self._get_gap_info(fs_proc)
            with DisabledScope(not can_track_gap):
                with primary_button_style():
                    if imgui.button("Track Gap & Merge"):
                        if can_track_gap:
                            self._handle_track_gap_and_merge(fs_proc, gap_c1, gap_c2)
                            self.list_context_selected_chapters.clear()
            if imgui.is_item_hovered():
                imgui.set_tooltip("Select two chapters with a frame gap between them to track the gap and merge.")

            imgui.same_line()

            # --- Create Chapter in Gap & Track Button ---
            can_create_in_gap, create_c1, _, gap_start, gap_end = self._get_gap_info(fs_proc)
            with DisabledScope(not can_create_in_gap):
                with primary_button_style():
                    if imgui.button("Create Chapter in Gap & Track"):
                        if can_create_in_gap:
                            self._handle_create_chapter_in_gap(fs_proc, create_c1, gap_start, gap_end)
                            self.list_context_selected_chapters.clear()
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Select two chapters with a gap to create a new chapter within that gap and start tracking.")

            imgui.separator()

            # --- RENDER TABLE ---
            if not fs_proc.video_chapters:
                imgui.text("No chapters loaded.")
                imgui.end()
                return

            table_flags = (imgui.TABLE_BORDERS |
                           imgui.TABLE_RESIZABLE |
                           imgui.TABLE_SIZING_STRETCH_PROP)

            if imgui.begin_table("ChapterListTable", 9, flags=table_flags):
                imgui.table_setup_column("##Select", init_width_or_weight=0.15)
                imgui.table_setup_column("#", init_width_or_weight=0.15)
                imgui.table_setup_column("Preview", init_width_or_weight=0.8)
                imgui.table_setup_column("Color", init_width_or_weight=0.25)
                imgui.table_setup_column("Position", init_width_or_weight=1.0)
                imgui.table_setup_column("Start", init_width_or_weight=0.9)
                imgui.table_setup_column("End", init_width_or_weight=0.9)
                imgui.table_setup_column("Duration", init_width_or_weight=0.7)
                imgui.table_setup_column("Actions", init_width_or_weight=0.6)
                imgui.table_headers_row()

                # Get FPS once for time calculation
                fps = self.app.processor.fps if self.app.processor and self.app.processor.fps > 0 else DEFAULT_CHAPTER_FPS
                sorted_chapters = sorted(fs_proc.video_chapters, key=lambda c: c.start_frame_id)
                chapters_to_remove_from_selection = []

                for i, chapter in enumerate(sorted_chapters):
                    imgui.table_next_row()

                    # Selection Checkbox
                    imgui.table_next_column()
                    imgui.push_id(f"select_{chapter.unique_id}")
                    is_selected = chapter in self.list_context_selected_chapters
                    changed, new_val = imgui.checkbox("", is_selected)
                    if changed:
                        if new_val:
                            if chapter not in self.list_context_selected_chapters:
                                self.list_context_selected_chapters.append(chapter)
                        else:
                            if chapter in self.list_context_selected_chapters:
                                self.list_context_selected_chapters.remove(chapter)
                        self.list_context_selected_chapters.sort(key=lambda c: c.start_frame_id)
                    imgui.pop_id()

                    # Chapter Number
                    imgui.table_next_column()
                    imgui.text(str(i + 1))

                    # Thumbnail Preview
                    imgui.table_next_column()
                    thumbnail_data = self.thumbnail_cache.get_thumbnail(chapter)
                    if thumbnail_data:
                        texture_id, thumb_width, thumb_height = thumbnail_data
                        # Draw thumbnail with slight padding
                        imgui.image(texture_id, thumb_width, thumb_height)
                        if imgui.is_item_hovered():
                            imgui.set_tooltip(f"Preview from start frame {chapter.start_frame_id}")
                    else:
                        # Placeholder if thumbnail failed to load
                        imgui.text_disabled("(no preview)")

                    # Color
                    imgui.table_next_column()
                    draw_list = imgui.get_window_draw_list()
                    cursor_pos = imgui.get_cursor_screen_pos()
                    swatch_start = (cursor_pos[0] + 2, cursor_pos[1] + 2)
                    swatch_end = (cursor_pos[0] + imgui.get_column_width() - 2, swatch_start[1] + 16)
                    color_tuple = chapter.color if isinstance(chapter.color, (tuple, list)) else (*CurrentTheme.GRAY_MEDIUM[:3], 0.7)  # Using GRAY_MEDIUM with 0.7 alpha
                    color_u32 = imgui.get_color_u32_rgba(*color_tuple)
                    draw_list.add_rect_filled(swatch_start[0], swatch_start[1], swatch_end[0], swatch_end[1], color_u32, rounding=3.0)

                    # Position Column
                    imgui.table_next_column()
                    imgui.text(chapter.position_long_name)
                    if imgui.is_item_hovered():
                        imgui.set_tooltip(f"ID: {chapter.unique_id}\nType: {chapter.segment_type}\nSource: {chapter.source}")

                    # Start Time / Frame
                    imgui.table_next_column()
                    start_time_s = chapter.start_frame_id / fps
                    imgui.text(f"{_format_time(self.app, start_time_s)} ({chapter.start_frame_id})")

                    # End Time / Frame
                    imgui.table_next_column()
                    end_time_s = chapter.end_frame_id / fps
                    imgui.text(f"{_format_time(self.app, end_time_s)} ({chapter.end_frame_id})")

                    # Duration
                    imgui.table_next_column()
                    duration_frames = chapter.end_frame_id - chapter.start_frame_id + 1
                    duration_s = duration_frames / fps
                    duration_str = f"{_format_time(self.app, duration_s)} ({duration_frames})"
                    imgui.text(duration_str)

                    # Actions
                    imgui.table_next_column()
                    imgui.push_id(f"actions_{chapter.unique_id}")

                    # Get icon textures
                    icon_mgr = get_icon_texture_manager()
                    edit_tex, _, _ = icon_mgr.get_icon_texture('edit.png')
                    trash_tex, _, _ = icon_mgr.get_icon_texture('trash.png')
                    btn_size = imgui.get_frame_height()

                    # Edit button with icon (SECONDARY - default styling)
                    if edit_tex:
                        if imgui.image_button(edit_tex, btn_size, btn_size):
                            self._open_edit_dialog(chapter)
                        if imgui.is_item_hovered():
                            imgui.set_tooltip("Edit chapter")
                    else:
                        if imgui.button("Edit"):
                            self._open_edit_dialog(chapter)
                        if imgui.is_item_hovered():
                            imgui.set_tooltip("Edit chapter")

                    imgui.same_line()

                    # Delete button with icon (DESTRUCTIVE - dangerous action)
                    with destructive_button_style():
                        if trash_tex:
                            if imgui.image_button(trash_tex, btn_size, btn_size):
                                fs_proc.delete_video_chapters_by_ids([chapter.unique_id])
                                if chapter in self.list_context_selected_chapters:
                                    chapters_to_remove_from_selection.append(chapter)
                            if imgui.is_item_hovered():
                                imgui.set_tooltip("Delete chapter")
                        else:
                            if imgui.button("Delete"):
                                fs_proc.delete_video_chapters_by_ids([chapter.unique_id])
                                if chapter in self.list_context_selected_chapters:
                                    chapters_to_remove_from_selection.append(chapter)
                            if imgui.is_item_hovered():
                                imgui.set_tooltip("Delete chapter")

                    imgui.pop_id()

                if chapters_to_remove_from_selection:
                    for chap in chapters_to_remove_from_selection:
                        self.list_context_selected_chapters.remove(chap)

                imgui.end_table()
        imgui.end()

    def _get_gap_info(self, fs_proc):
        if len(self.list_context_selected_chapters) != 2:
            return False, None, None, 0, 0

        chapters = sorted(self.list_context_selected_chapters, key=lambda c: c.start_frame_id)
        c1, c2 = chapters[0], chapters[1]

        gap_start = c1.end_frame_id + 1
        gap_end = c2.start_frame_id - 1

        if gap_end >= gap_start:
            return True, c1, c2, gap_start, gap_end
        return False, None, None, 0, 0

    def _handle_track_gap_and_merge(self, fs_proc, c1, c2):
        self.app.logger.info(f"UI Action: Initiating track gap then merge between {c1.unique_id} and {c2.unique_id}")
        gap_start = c1.end_frame_id + 1
        gap_end = c2.start_frame_id - 1

        self.app.set_pending_action_after_tracking(
            action_type='finalize_gap_merge_after_tracking',
            chapter1_id=c1.unique_id,
            chapter2_id=c2.unique_id
        )
        fs_proc.scripting_start_frame = gap_start
        fs_proc.scripting_end_frame = gap_end
        fs_proc.scripting_range_active = True
        fs_proc.selected_chapter_for_scripting = None
        self.app.project_manager.project_dirty = True

        self._start_live_tracking(on_error_clear_pending_action=True)

    def _handle_create_chapter_in_gap(self, fs_proc, c1, gap_start, gap_end):
        self.app.logger.info(f"UI Action: Creating new chapter in gap after {c1.unique_id}")
        gap_chapter_data = {
            "start_frame_str": str(gap_start),
            "end_frame_str": str(gap_end),
            "segment_type": c1.segment_type,
            "position_short_name_key": c1.position_short_name,
            "source": "manual_gap_fill_track"
        }
        new_chapter = fs_proc.create_new_chapter_from_data(gap_chapter_data, return_chapter_object=True)
        if new_chapter:
            fs_proc.set_scripting_range_from_chapter(new_chapter)
            self._start_live_tracking()
        else:
            self.app.logger.error("Failed to create new chapter in gap.")

    def _open_edit_dialog(self, chapter):
        self.nav_ui.chapter_to_edit_id = chapter.unique_id
        self.nav_ui.chapter_edit_data = {
            "start_frame_str": str(chapter.start_frame_id),
            "end_frame_str": str(chapter.end_frame_id),
            "segment_type": chapter.segment_type,
            "position_short_name_key": chapter.position_short_name,
            "source": chapter.source
        }
        try:
            self.nav_ui.selected_position_idx_in_dialog = self.nav_ui.position_short_name_keys.index(
                chapter.position_short_name)
        except (ValueError, IndexError):
            self.nav_ui.selected_position_idx_in_dialog = 0

        # Initialize timecode fields
        self.nav_ui._update_timecode_from_frame("start")
        self.nav_ui._update_timecode_from_frame("end")

        self.nav_ui.show_edit_chapter_dialog = True
