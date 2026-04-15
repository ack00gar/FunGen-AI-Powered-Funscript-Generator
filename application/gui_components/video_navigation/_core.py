import imgui
from typing import Optional

from application.utils import _format_time, VideoSegment, get_icon_texture_manager, primary_button_style, destructive_button_style
from config.constants import POSITION_INFO_MAPPING, DEFAULT_CHAPTER_FPS
from config.element_group_colors import VideoNavigationColors
from config.constants_colors import CurrentTheme



class VideoNavigationCoreMixin:
    """Mixin fragment for VideoNavigationUI."""

    def __init__(self, app, gui_instance):
        self.app = app
        self.gui_instance = gui_instance
        self.chapter_tooltip_segment = None
        self.context_selected_chapters = []
        self.chapter_bar_popup_id = "ChapterBarContextPopup_Main"

        # State for dialogs/windows
        self.show_create_chapter_dialog = False
        self.show_edit_chapter_dialog = False

        # Timecode display fields (synced with frame fields)
        self.chapter_start_timecode = "00:00:00.000"
        self.chapter_end_timecode = "00:00:00.000"

        # Prepare data for dialogs - dynamically from ChapterTypeManager
        self._update_chapter_type_lists()

        default_pos_key = self.position_short_name_keys[0] if self.position_short_name_keys else "N/A"

        # Import enums for segment type and source
        from config.constants import ChapterSegmentType, ChapterSource

        self.chapter_edit_data = {
            "start_frame_str": "0",
            "end_frame_str": "0",
            "segment_type": ChapterSegmentType.get_default().value,
            "position_short_name_key": default_pos_key,
            "source": ChapterSource.get_default().value
        }
        self.chapter_to_edit_id: Optional[str] = None

        # Dropdown indices for segment type and source
        self.selected_segment_type_idx = 0
        self.selected_source_idx = 0
        
        # Chapter creation drag state
        self.is_dragging_chapter_range = False
        self.drag_start_frame = 0
        self.drag_current_frame = 0
        
        # Chapter resizing state
        self.is_resizing_chapter = False
        self.resize_chapter_id = None
        self.resize_edge = None  # 'left' or 'right'
        self.resize_original_start = 0
        self.resize_original_end = 0

        # Chapter edge drag preview state (similar to hover navigation preview)
        self.resize_preview_frame = None
        self.resize_preview_data = None

        # Store frame position when context menu opens for chapter split
        self.context_menu_opened_at_frame = None

        # Track if context menu was opened this frame to prevent create dialog from opening
        self.context_menu_opened_this_frame = False

        try:
            self.selected_position_idx_in_dialog = self.position_short_name_keys.index(
                self.chapter_edit_data["position_short_name_key"])
        except (ValueError, IndexError):
            self.selected_position_idx_in_dialog = 0


    def _start_live_tracking(self, success_info: Optional[str] = None, on_error_clear_pending_action: bool = False) -> None:
        """Centralized starter for live tracking across UI entry points.

        - Calls `event_handlers.handle_start_live_tracker_click()` if available
        - Logs a provided success message
        - Optionally clears pending action on error when requested
        """
        try:
            handler = getattr(self.app.event_handlers, 'handle_start_live_tracker_click', None)
            if callable(handler):
                handler()
                if success_info:
                    self.app.logger.info(success_info)
            else:
                self.app.logger.error("handle_start_live_tracker_click not found in event_handlers.")
                if on_error_clear_pending_action and hasattr(self.app, 'clear_pending_action_after_tracking'):
                    self.app.clear_pending_action_after_tracking()
        except Exception as exc:
            self.app.logger.error(f"Failed to start live tracking: {exc}")
            if on_error_clear_pending_action and hasattr(self.app, 'clear_pending_action_after_tracking'):
                self.app.clear_pending_action_after_tracking()


    def _get_current_fps(self) -> float:
        fps = DEFAULT_CHAPTER_FPS
        if self.app.processor:
            if hasattr(self.app.processor,
                       'video_info') and self.app.processor.video_info and self.app.processor.video_info.get('fps', 0) > 0:
                fps = self.app.processor.video_info['fps']
            elif hasattr(self.app.processor, 'fps') and self.app.processor.fps > 0:
                fps = self.app.processor.fps
        return fps


    def _update_chapter_type_lists(self):
        """Update chapter type lists from ChapterTypeManager (includes custom types)."""
        from application.classes.chapter_type_manager import get_chapter_type_manager

        type_manager = get_chapter_type_manager()
        if type_manager:
            all_types = type_manager.get_all_chapter_types()
        else:
            # Fallback to built-in types if manager not initialized yet
            all_types = POSITION_INFO_MAPPING

        self.position_short_name_keys = list(all_types.keys())
        self.position_display_names = [
            f"{all_types[key]['short_name']} ({all_types[key]['long_name']})"
            for key in self.position_short_name_keys
        ] if self.position_short_name_keys else ["N/A"]


    def render(self, nav_content_width=None):
        app_state = self.app.app_state_ui
        is_floating = app_state.ui_layout_mode == 'floating'

        should_render = True
        if is_floating:
            if not getattr(app_state, 'show_video_navigation_window', True):
                return
            is_open, new_visibility = imgui.begin("Video Navigation", closable=True)
            if new_visibility != app_state.show_video_navigation_window:
                app_state.show_video_navigation_window = new_visibility
                self.app.project_manager.project_dirty = True
            if not is_open:
                should_render = False
        else:
            imgui.begin("Video Navigation##CenterNav",
                        flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_COLLAPSE)

        if should_render:
            actual_content_width = imgui.get_content_region_available()[0]
            fs_proc = self.app.funscript_processor

            eff_duration_s, _, _ = self.app.get_effective_video_duration_params()

            section_left_x, section_top_screen_y = imgui.get_cursor_screen_pos()
            playhead_top_y = None

            if app_state.show_funscript_timeline:
                imgui.push_item_width(actual_content_width)
                # Funscript preview internally offsets cursor by +20 before the image.
                playhead_top_y = section_top_screen_y + 20
                self._render_funscript_timeline_preview(eff_duration_s, app_state.funscript_preview_draw_height)
                imgui.pop_item_width()

            total_frames_for_bars = 0
            if self.app.processor and self.app.processor.video_info and self.app.processor.video_info.get(
                    'total_frames', 0) > 0:
                total_frames_for_bars = self.app.processor.video_info.get('total_frames', 0)
            elif self.app.file_manager.video_path:
                if self.app.processor and hasattr(self.app.processor, 'total_frames') and self.app.processor.total_frames > 0:
                    total_frames_for_bars = self.app.processor.total_frames

            chapter_bar_h = fs_proc.chapter_bar_height if hasattr(fs_proc, 'chapter_bar_height') else 20
            if playhead_top_y is None:
                playhead_top_y = imgui.get_cursor_screen_pos()[1]
            self._render_chapter_bar(fs_proc, total_frames_for_bars, actual_content_width, chapter_bar_h)
            imgui.spacing()

            if app_state.show_heatmap:
                self._render_funscript_heatmap_preview(eff_duration_s, actual_content_width, app_state.timeline_heatmap_height)
                imgui.spacing()

            # Unified playhead line across funscript preview + chapter bar + heatmap.
            proc = self.app.processor
            if (proc and proc.video_info and proc.current_frame_index >= 0
                    and total_frames_for_bars > 0 and eff_duration_s > 0):
                fps = proc.fps if proc.fps and proc.fps > 0 else 30.0
                current_time_s = proc.current_frame_index / fps
                norm = max(0.0, min(1.0, current_time_s / eff_duration_s))
                marker_x = section_left_x + norm * actual_content_width
                playhead_bottom_y = imgui.get_cursor_screen_pos()[1] - 2
                if playhead_bottom_y > playhead_top_y:
                    dl = imgui.get_window_draw_list()
                    col = imgui.get_color_u32_rgba(1.0, 0.15, 0.15, 1.0)
                    dl.add_line(marker_x, playhead_top_y, marker_x, playhead_bottom_y, col, thickness=1.5)
            if self.chapter_tooltip_segment and total_frames_for_bars > 0:
                self._render_chapter_tooltip()

            self._render_chapter_context_menu()
            if self.show_create_chapter_dialog: self._render_create_chapter_window()
            if self.show_edit_chapter_dialog: self._render_edit_chapter_window()

        # --- Timeline Visibility Toggles as Small Buttons ---
        # Ensure full_width_nav attribute exists (user-controlled via View > Layout menu)
        if not hasattr(app_state, 'full_width_nav'):
            app_state.full_width_nav = False

        imgui.end()


