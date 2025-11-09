import imgui
import logging
from typing import Optional

from application.utils import _format_time, VideoSegment, get_icon_texture_manager, primary_button_style, destructive_button_style
from config.constants import POSITION_INFO_MAPPING, DEFAULT_CHAPTER_FPS
from config.element_group_colors import VideoNavigationColors
from config.constants_colors import CurrentTheme


class VideoNavigationUI:
    def __init__(self, app, gui_instance):
        self.app = app
        self.gui_instance = gui_instance
        self.chapter_tooltip_segment = None
        self.context_selected_chapters = []
        self.chapter_bar_popup_id = "ChapterBarContextPopup_Main"

        # State for dialogs/windows
        self.show_create_chapter_dialog = False
        self.show_edit_chapter_dialog = False

        # Prepare data for dialogs
        self.position_short_name_keys = list(POSITION_INFO_MAPPING.keys())
        # IMPROVEMENT: Make display names more user-friendly by removing the internal key.
        self.position_display_names = [
            f"{POSITION_INFO_MAPPING[key]['short_name']} ({POSITION_INFO_MAPPING[key]['long_name']})"
            for key in self.position_short_name_keys
        ] if self.position_short_name_keys else ["N/A"]

        default_pos_key = self.position_short_name_keys[0] if self.position_short_name_keys else "N/A"

        self.chapter_edit_data = {
            "start_frame_str": "0",
            "end_frame_str": "0",
            "segment_type": "SexAct",
            "position_short_name_key": default_pos_key,
            "source": "manual"
        }
        self.chapter_to_edit_id: Optional[str] = None
        
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
            if app_state.show_funscript_timeline:
                imgui.push_item_width(actual_content_width)
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
            self._render_chapter_bar(fs_proc, total_frames_for_bars, actual_content_width, chapter_bar_h)
            imgui.spacing()

            if app_state.show_heatmap:
                self._render_funscript_heatmap_preview(eff_duration_s, actual_content_width, app_state.timeline_heatmap_height)
                imgui.spacing()
            if self.chapter_tooltip_segment and total_frames_for_bars > 0:
                self._render_chapter_tooltip()

            self._render_chapter_context_menu()
            if self.show_create_chapter_dialog: self._render_create_chapter_window()
            if self.show_edit_chapter_dialog: self._render_edit_chapter_window()

        # --- Timeline Visibility Toggles as Small Buttons ---
        def render_timeline_toggle(label, visible_attr, tooltip):
            pushed = False
            if not getattr(app_state, visible_attr):
                imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)
                pushed = True
            if imgui.button(label):
                setattr(app_state, visible_attr, not getattr(app_state, visible_attr))
                self.app.project_manager.project_dirty = True
            if imgui.is_item_hovered():
                imgui.set_tooltip(tooltip)
            if pushed:
                imgui.pop_style_var()

        use_small_font = hasattr(self.gui_instance, 'small_font') and self.gui_instance.small_font and getattr(self.gui_instance.small_font, 'is_loaded', lambda: True)()
        if use_small_font:
            imgui.push_font(self.gui_instance.small_font)

        # Left-aligned: T1, T2, Options dropdown
        render_timeline_toggle("T1", "show_funscript_interactive_timeline", "Toggle Timeline 1")
        imgui.same_line(spacing=4)
        render_timeline_toggle("T2", "show_funscript_interactive_timeline2", "Toggle Timeline 2 (T)")
        imgui.same_line(spacing=4)

        # Options dropdown button
        if imgui.button("Options"):
            imgui.open_popup("VideoNavOptions")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Show video navigation options")

        # Options dropdown menu
        if imgui.begin_popup("VideoNavOptions"):
            # Ensure full_width_nav attribute exists
            if not hasattr(app_state, 'full_width_nav'):
                app_state.full_width_nav = False

            # Preview checkbox
            changed, new_val = imgui.checkbox("Show Funscript Preview (P)", app_state.show_funscript_timeline)
            if changed:
                app_state.show_funscript_timeline = new_val
                self.app.project_manager.project_dirty = True

            # Heatmap checkbox
            changed, new_val = imgui.checkbox("Show Heatmap (H)", app_state.show_heatmap)
            if changed:
                app_state.show_heatmap = new_val
                self.app.project_manager.project_dirty = True

            # Full Width Nav checkbox
            changed, new_val = imgui.checkbox("Full Width Navigation", app_state.full_width_nav)
            if changed:
                app_state.full_width_nav = new_val
                self.app.project_manager.project_dirty = True

            imgui.end_popup()

        if use_small_font:
            imgui.pop_font()

        imgui.end()

    def _render_chapter_bar(self, fs_proc, total_video_frames: int, bar_width: float, bar_height: float):
        # Reset flag at start of each frame to prevent stale state
        self.context_menu_opened_this_frame = False

        ###########################################################################################
        # TEMPORARY: Assign random colors to chapters for visual distinction when all chapters have the same position_short_name (e.g., 'NR').
        # Remove this logic once position detection is implemented for "Scene Detection without AI analysis"

        # AI Analysis = fixed colors per position
        # Scene Detection without AI analysis = returns all 'NR', will then assign random colors
        ###########################################################################################

        if hasattr(self, '_last_chapter_count'):
            last_count = self._last_chapter_count
        else:
            last_count = -1
        current_count = len(fs_proc.video_chapters)
        if current_count > 0 and current_count != last_count:
            # If all chapters have the same position_short_name, assign random colors for visual distinction
            unique_short_names = set(seg.position_short_name for seg in fs_proc.video_chapters)
            if len(unique_short_names) == 1:
                VideoSegment.assign_random_colors_to_segments(fs_proc.video_chapters)
            else:
                VideoSegment.assign_colors_to_segments(fs_proc.video_chapters)
        self._last_chapter_count = current_count

        # END OF TEMPORARY COLOR SELECTION LOGIC
        ###########################################################################################

        style = imgui.get_style()  # Get style for frame_padding
        draw_list = imgui.get_window_draw_list()
        cursor_screen_pos = imgui.get_cursor_screen_pos()

        # bar_start_x and bar_width define the full extent of the chapter bar background
        bar_start_x = cursor_screen_pos[0]
        bar_start_y = cursor_screen_pos[1]
        # bar_width is nav_content_width

        bg_col = imgui.get_color_u32_rgba(*VideoNavigationColors.BACKGROUND)
        # Draw the background for the chapter bar using full bar_width
        draw_list.add_rect_filled(bar_start_x, bar_start_y, bar_start_x + bar_width, bar_start_y + bar_height, bg_col)

        if total_video_frames <= 0:
            imgui.dummy(bar_width, bar_height)
            imgui.set_cursor_screen_pos((bar_start_x, bar_start_y + bar_height))
            imgui.spacing()
            return

        # For marker alignment with slider track, calculate effective start_x and width
        effective_marker_area_start_x = bar_start_x + style.frame_padding[0]
        effective_marker_area_width = bar_width - (style.frame_padding[0] * 2)

        self.chapter_tooltip_segment = None
        action_on_segment_this_frame = False

        for segment_idx, segment in enumerate(fs_proc.video_chapters):
            start_x_norm = segment.start_frame_id / total_video_frames
            end_x_norm = segment.end_frame_id / total_video_frames
            seg_start_x = bar_start_x + start_x_norm * bar_width
            seg_end_x = bar_start_x + end_x_norm * bar_width
            seg_width = max(1, seg_end_x - seg_start_x)

            if segment.user_roi_fixed:
                icon_pos_x = seg_start_x + 3
                icon_pos_y = bar_start_y + (bar_height - imgui.get_text_line_height()) / 2
                icon_color = imgui.get_color_u32_rgba(*VideoNavigationColors.ICON)  # Bright Yellow
                # Using a simple character as an icon. A texture could be used for a nicer look.
                draw_list.add_text(icon_pos_x, icon_pos_y, icon_color, "[R]")

            segment_color_tuple = segment.color
            if not (isinstance(segment_color_tuple, (tuple, list)) and len(segment_color_tuple) in [3, 4]):
                self.app.logger.warning(
                    f"Segment {segment.unique_id} ('{segment.class_name if hasattr(segment, 'class_name') else 'N/A'}') has invalid color {segment_color_tuple}, using default gray.")
                segment_color_tuple = (*CurrentTheme.GRAY_MEDIUM[:3], 0.7)  # Using GRAY_MEDIUM with 0.7 alpha
            seg_color = imgui.get_color_u32_rgba(*segment_color_tuple)

            is_selected_for_scripting = (fs_proc.scripting_range_active
                and fs_proc.selected_chapter_for_scripting
                and fs_proc.selected_chapter_for_scripting.unique_id == segment.unique_id
                and fs_proc.scripting_start_frame == segment.start_frame_id
                and fs_proc.scripting_end_frame == segment.end_frame_id)

            draw_list.add_rect_filled(seg_start_x, bar_start_y, seg_start_x + seg_width, bar_start_y + bar_height, seg_color)

            is_context_selected_primary = False
            is_context_selected_secondary = False
            if len(self.context_selected_chapters) > 0 and self.context_selected_chapters[
                0].unique_id == segment.unique_id:
                is_context_selected_primary = True
            if len(self.context_selected_chapters) > 1 and self.context_selected_chapters[
                1].unique_id == segment.unique_id:
                is_context_selected_secondary = True

            if is_selected_for_scripting:
                scripting_border_col = imgui.get_color_u32_rgba(*VideoNavigationColors.SCRIPTING_BORDER)
                draw_list.add_rect(seg_start_x + 0.5, bar_start_y + 0.5, seg_start_x + seg_width - 0.5, bar_start_y + bar_height - 0.5, scripting_border_col, thickness=1.0, rounding=0.0)

            if is_context_selected_primary:
                border_col_sel1 = imgui.get_color_u32_rgba(*VideoNavigationColors.SELECTION_PRIMARY)
                draw_list.add_rect(seg_start_x - 1, bar_start_y - 1, seg_start_x + seg_width + 1, bar_start_y + bar_height + 1, border_col_sel1, thickness=2.0, rounding=1.0)

            if is_context_selected_secondary:
                border_col_sel2 = imgui.get_color_u32_rgba(*VideoNavigationColors.SELECTION_SECONDARY)
                draw_list.add_rect(seg_start_x - 2, bar_start_y - 2, seg_start_x + seg_width + 2, bar_start_y + bar_height + 2, border_col_sel2, thickness=1.5, rounding=1.0)

            text_to_draw = f"{segment.position_short_name}"
            text_width = imgui.calc_text_size(text_to_draw)[0]
            if text_width < seg_width - 8:
                text_pos_x = seg_start_x + (seg_width - text_width) / 2
                text_pos_y = bar_start_y + (bar_height - imgui.get_text_line_height()) / 2
                valid_color_for_lum = segment_color_tuple if isinstance(segment_color_tuple, tuple) and len(
                    segment_color_tuple) >= 3 else CurrentTheme.GRAY_MEDIUM[:3]  # Using GRAY_MEDIUM
                lum = 0.2100 * valid_color_for_lum[0] + 0.587 * valid_color_for_lum[1] + 0.114 * valid_color_for_lum[2]
                text_color = imgui.get_color_u32_rgba(*VideoNavigationColors.TEXT_BLACK) if lum > 0.6 else imgui.get_color_u32_rgba(*VideoNavigationColors.TEXT_WHITE)
                draw_list.add_text(text_pos_x, text_pos_y, text_color, text_to_draw)

            # Expand clickable area to include selection borders (which extend 2px beyond segment)
            # to ensure clicks on borders are properly detected
            border_expansion = 3  # Slightly larger than the largest border offset (2px)
            expanded_start_x = seg_start_x - border_expansion
            expanded_start_y = bar_start_y - border_expansion
            expanded_width = seg_width + (border_expansion * 2)
            expanded_height = bar_height + (border_expansion * 2)

            imgui.set_cursor_screen_pos((expanded_start_x, expanded_start_y))
            button_id = f"chapter_bar_segment_btn_{segment.unique_id}"

            imgui.invisible_button(button_id, expanded_width, expanded_height)

            if imgui.is_item_hovered():
                self.chapter_tooltip_segment = segment

                if imgui.is_mouse_double_clicked(0):
                    action_on_segment_this_frame = True
                    if self.app.processor:
                        io = imgui.get_io()
                        if io.key_alt:
                            self.app.processor.seek_video(segment.end_frame_id)
                        else:
                            self.app.processor.seek_video(segment.start_frame_id)
                        # Ensure timeline synchronization after seeking
                        self.app.app_state_ui.force_timeline_pan_to_current_frame = True
                elif imgui.is_item_clicked(0):
                    # Check if this is a resize anchor click first
                    io = imgui.get_io()
                    mouse_pos = io.mouse_pos
                    edge_tolerance = 8  # Same as resize logic
                    
                    # Calculate edge positions for this segment
                    start_x_norm = segment.start_frame_id / total_video_frames
                    end_x_norm = segment.end_frame_id / total_video_frames
                    seg_start_x = bar_start_x + start_x_norm * bar_width
                    seg_end_x = bar_start_x + end_x_norm * bar_width
                    
                    # Check if click is near an edge of a selected chapter
                    is_anchor_click = False
                    if segment in self.context_selected_chapters:
                        near_left_edge = abs(mouse_pos[0] - seg_start_x) <= edge_tolerance
                        near_right_edge = abs(mouse_pos[0] - seg_end_x) <= edge_tolerance
                        if near_left_edge or near_right_edge:
                            # This is an anchor click - start resize mode
                            self.is_resizing_chapter = True
                            self.resize_chapter_id = segment.unique_id
                            self.resize_edge = 'left' if near_left_edge else 'right'
                            action_on_segment_this_frame = True
                            is_anchor_click = True
                    
                    if not is_anchor_click:
                        # Handle normal chapter selection
                        action_on_segment_this_frame = True
                        is_shift_held = io.key_shift
                        if is_shift_held:
                            if segment in self.context_selected_chapters:
                                self.context_selected_chapters.remove(segment)
                            elif len(self.context_selected_chapters) < 2:
                                self.context_selected_chapters.append(segment)
                        else:
                            if segment in self.context_selected_chapters and len(self.context_selected_chapters) == 1 and \
                                    self.context_selected_chapters[0].unique_id == segment.unique_id:
                                self.context_selected_chapters.clear()
                            else:
                                self.context_selected_chapters.clear()
                                self.context_selected_chapters.append(segment)

                        unique_sel = []
                        seen_ids = set()
                        for s_item in self.context_selected_chapters:
                            if s_item.unique_id not in seen_ids:
                                unique_sel.append(s_item)
                                seen_ids.add(s_item.unique_id)
                        self.context_selected_chapters = unique_sel
                        if self.context_selected_chapters:
                            self.context_selected_chapters.sort(key=lambda s: s.start_frame_id)

                        if hasattr(self.app.event_handlers, 'handle_chapter_bar_segment_click'):
                            self.app.event_handlers.handle_chapter_bar_segment_click(segment, is_selected_for_scripting)

                elif imgui.is_item_clicked(1):
                    action_on_segment_this_frame = True
                    if segment not in self.context_selected_chapters:
                        self.context_selected_chapters.clear()
                        self.context_selected_chapters.append(segment)
                    # Store current frame position for chapter split operation
                    self.context_menu_opened_at_frame = self.app.processor.current_frame_index if self.app.processor else None
                    # Set flag to prevent create dialog from opening in the same frame
                    self.context_menu_opened_this_frame = True
                    self.app.logger.debug(
                        f"Right clicked on chapter {segment.unique_id} at frame {self.context_menu_opened_at_frame}. Current selection: {[s.unique_id for s in self.context_selected_chapters]}. Opening context menu: {self.chapter_bar_popup_id}")
                    imgui.open_popup(self.chapter_bar_popup_id)

        # Smart chapter resizing - check for edge hover and handle resize drags
        io = imgui.get_io()
        mouse_pos = io.mouse_pos
        edge_tolerance = 8  # pixels near edge to trigger resize
        
        # Handle ongoing resize
        if self.is_resizing_chapter and imgui.is_mouse_dragging(0):
            # Find the chapter being resized
            resize_chapter = None
            for chapter in fs_proc.video_chapters:
                if chapter.unique_id == self.resize_chapter_id:
                    resize_chapter = chapter
                    break
            
            if resize_chapter:
                # Calculate new frame position
                dragged_x_on_bar = mouse_pos[0] - bar_start_x
                norm_drag_pos = max(0, min(1, dragged_x_on_bar / bar_width))
                new_frame = int(norm_drag_pos * total_video_frames)
                
                if self.resize_edge == 'left':
                    # Resizing left edge (start)
                    new_start = new_frame
                    new_end = resize_chapter.end_frame_id
                    # Constrain to prevent invalid range
                    new_start = min(new_start, new_end - 1)  # At least 1 frame duration
                    # Constrain to prevent overlap with previous chapter
                    chapters_sorted = sorted(fs_proc.video_chapters, key=lambda c: c.start_frame_id)
                    for i, chapter in enumerate(chapters_sorted):
                        if chapter.unique_id == self.resize_chapter_id and i > 0:
                            prev_chapter = chapters_sorted[i - 1]
                            new_start = max(new_start, prev_chapter.end_frame_id + 1)
                            break
                    new_start = max(0, new_start)  # Don't go below 0
                    
                    # Update chapter
                    resize_chapter.start_frame_id = new_start
                else:
                    # Resizing right edge (end)
                    new_start = resize_chapter.start_frame_id
                    new_end = new_frame
                    # Constrain to prevent invalid range
                    new_end = max(new_end, new_start + 1)  # At least 1 frame duration
                    # Constrain to prevent overlap with next chapter
                    chapters_sorted = sorted(fs_proc.video_chapters, key=lambda c: c.start_frame_id)
                    for i, chapter in enumerate(chapters_sorted):
                        if chapter.unique_id == self.resize_chapter_id and i < len(chapters_sorted) - 1:
                            next_chapter = chapters_sorted[i + 1]
                            new_end = min(new_end, next_chapter.start_frame_id - 1)
                            break
                    
                    # Update chapter
                    resize_chapter.end_frame_id = new_end
        
        # End resize on mouse release
        elif self.is_resizing_chapter and imgui.is_mouse_released(0):
            self.is_resizing_chapter = False
            self.resize_chapter_id = None
            self.resize_edge = None
            action_on_segment_this_frame = True  # Prevent other interactions
            self.app.logger.info("Chapter resized", extra={'status_message': True})
        
        # Check for resize initiation and draw resize handles when hovering near edges
        elif not self.is_resizing_chapter and not action_on_segment_this_frame:
            # Only show resize handles for selected chapters
            if len(self.context_selected_chapters) > 0:
                # For each selected chapter, check for edge proximity
                closest_distance = float('inf')
                closest_segment = None
                closest_edge_type = None
                closest_edge_x = None
                
                for selected_chapter in self.context_selected_chapters:
                    start_x_norm = selected_chapter.start_frame_id / total_video_frames
                    end_x_norm = selected_chapter.end_frame_id / total_video_frames
                    seg_start_x = bar_start_x + start_x_norm * bar_width
                    seg_end_x = bar_start_x + end_x_norm * bar_width
                    
                    # Check if mouse is within vertical bounds
                    if bar_start_y <= mouse_pos[1] <= bar_start_y + bar_height:
                        # Check distance to left edge
                        left_distance = abs(mouse_pos[0] - seg_start_x)
                        if left_distance <= edge_tolerance and left_distance < closest_distance:
                            closest_distance = left_distance
                            closest_segment = selected_chapter
                            closest_edge_type = 'left'
                            closest_edge_x = seg_start_x
                        
                        # Check distance to right edge
                        right_distance = abs(mouse_pos[0] - seg_end_x)
                        if right_distance <= edge_tolerance and right_distance < closest_distance:
                            closest_distance = right_distance
                            closest_segment = selected_chapter
                            closest_edge_type = 'right'
                            closest_edge_x = seg_end_x
                
                # Draw anchor point for closest edge of selected chapter
                if closest_segment and closest_edge_type:
                    handle_color = imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 0.9)  # White
                    border_color = imgui.get_color_u32_rgba(0.3, 0.3, 0.3, 1.0)  # Dark border
                    radius = 3.5
                    center_y = bar_start_y + bar_height / 2
                    
                    # Draw filled circle
                    draw_list.add_circle_filled(closest_edge_x, center_y, radius, handle_color)
                    # Draw border circle
                    draw_list.add_circle(closest_edge_x, center_y, radius, border_color, thickness=1.5)
                    
                    imgui.set_mouse_cursor(imgui.MOUSE_CURSOR_RESIZE_EW)
                    if imgui.is_mouse_clicked(0):
                        self.is_resizing_chapter = True
                        self.resize_chapter_id = closest_segment.unique_id
                        self.resize_edge = closest_edge_type
                        action_on_segment_this_frame = True

        if self.app.processor and self.app.processor.video_info and self.app.processor.current_frame_index >= 0 and total_video_frames > 0:
            current_norm_pos = self.app.processor.current_frame_index / total_video_frames
            # marker_x = bar_start_x + current_norm_pos * bar_width
            marker_x = effective_marker_area_start_x + current_norm_pos * effective_marker_area_width
            marker_col = imgui.get_color_u32_rgba(*VideoNavigationColors.MARKER)
            draw_list.add_line(marker_x, bar_start_y, marker_x, bar_start_y + bar_height, marker_col, thickness=2.0)

        # Note: io and mouse_pos already defined above for resize logic
        full_bar_rect_min = (bar_start_x, bar_start_y)
        full_bar_rect_max = (bar_start_x + bar_width, bar_start_y + bar_height)

        is_mouse_over_bar = full_bar_rect_min[0] <= mouse_pos[0] <= full_bar_rect_max[0] and full_bar_rect_min[1] <= mouse_pos[1] <= full_bar_rect_max[1]

        # Prevent create dialog if context menu or any other dialog is already open
        is_any_popup_open = imgui.is_popup_open(self.chapter_bar_popup_id, imgui.POPUP_ANY_POPUP_ID)

        # Additional check: prevent create dialog if context menu was just opened this frame
        # This is needed because is_popup_open returns False immediately after open_popup is called
        if is_mouse_over_bar and imgui.is_mouse_clicked(1) and not action_on_segment_this_frame and not is_any_popup_open and not self.context_menu_opened_this_frame:

            clicked_x_on_bar = mouse_pos[0] - bar_start_x
            norm_click_pos = clicked_x_on_bar / bar_width
            clicked_frame_id = int(norm_click_pos * total_video_frames)
            self.app.logger.info(
                f"Right-clicked on empty chapter bar space at frame: {clicked_frame_id}. Triggering create dialog.")

            chapters_sorted = sorted(fs_proc.video_chapters, key=lambda c: c.start_frame_id)
            prev_ch = None
            for ch_idx, ch in enumerate(chapters_sorted):
                if ch.end_frame_id < clicked_frame_id:
                    if prev_ch is None or ch.end_frame_id > prev_ch.end_frame_id:
                        prev_ch = ch
                else:
                    break

            next_ch = None
            for ch_idx in range(len(chapters_sorted) - 1, -1, -1):
                ch = chapters_sorted[ch_idx]
                if ch.start_frame_id > clicked_frame_id:
                    if next_ch is None or ch.start_frame_id < next_ch.start_frame_id:
                        next_ch = ch
                else:
                    break

            fps = self._get_current_fps()
            default_duration_frames = int(fps * 5)

            start_f = clicked_frame_id
            end_f = clicked_frame_id + default_duration_frames - 1

            if prev_ch is not None:
                start_f = prev_ch.end_frame_id + 1
            if next_ch is not None:
                end_f = next_ch.start_frame_id - 1
            if prev_ch is not None and next_ch is None:
                end_f = start_f + default_duration_frames - 1
            elif prev_ch is None and next_ch is not None:
                start_f = end_f - default_duration_frames + 1

            if start_f > end_f:
                start_f = clicked_frame_id
                end_f = clicked_frame_id

            start_f = max(0, start_f)
            end_f = min(total_video_frames - 1, end_f)
            start_f = min(start_f, end_f)  # Ensure start is not after end
            end_f = max(start_f, end_f)  # Ensure end is not before start

            default_pos_key = self.position_short_name_keys[0] if self.position_short_name_keys else "N/A"
            self.chapter_edit_data = {
                "start_frame_str": str(start_f),
                "end_frame_str": str(end_f),
                "segment_type": "SexAct",
                "position_short_name_key": default_pos_key,
                "source": "manual_bar_rclick"
            }
            try:
                self.selected_position_idx_in_dialog = self.position_short_name_keys.index(default_pos_key)
            except (ValueError, IndexError):
                self.selected_position_idx_in_dialog = 0

            self.show_create_chapter_dialog = True
            self.context_selected_chapters.clear()

        # Enhanced UX: Click in gap between chapters to fill it
        if is_mouse_over_bar and not action_on_segment_this_frame and imgui.is_mouse_clicked(0):
            clicked_x_on_bar = mouse_pos[0] - bar_start_x
            norm_click_pos = clicked_x_on_bar / bar_width
            clicked_frame = int(norm_click_pos * total_video_frames)

            # Check if click is in a gap between two chapters
            gap_detected = False
            if fs_proc.video_chapters:
                chapters_sorted = sorted(fs_proc.video_chapters, key=lambda c: c.start_frame_id)

                # Find if clicked frame is in a gap
                for i in range(len(chapters_sorted) - 1):
                    current_chapter = chapters_sorted[i]
                    next_chapter = chapters_sorted[i + 1]

                    gap_start = current_chapter.end_frame_id + 1
                    gap_end = next_chapter.start_frame_id - 1

                    # Check if click is within this gap
                    if gap_start <= clicked_frame <= gap_end:
                        gap_detected = True

                        # Create chapter to fill the gap
                        default_pos_key = self.position_short_name_keys[0] if self.position_short_name_keys else "N/A"
                        chapter_data = {
                            "start_frame_str": str(gap_start),
                            "end_frame_str": str(gap_end),
                            "segment_type": "SexAct",
                            "position_short_name_key": default_pos_key,
                            "source": "gap_fill_click"
                        }

                        if self.app.funscript_processor:
                            self.app.funscript_processor.create_new_chapter_from_data(chapter_data)
                            self.app.logger.info(f"Created chapter to fill gap ({gap_end - gap_start + 1} frames)", extra={'status_message': True})
                        break

            # If no gap detected, start normal drag to create chapter
            if not gap_detected:
                # Start dragging
                self.drag_start_frame = clicked_frame
                self.is_dragging_chapter_range = True
                self.app.logger.info("Drag started - creating chapter range", extra={'status_message': True})

        # Handle ongoing drag (separate check)
        if is_mouse_over_bar and not action_on_segment_this_frame:
            if self.is_dragging_chapter_range and imgui.is_mouse_dragging(0):
                # Update drag end position
                dragged_x_on_bar = mouse_pos[0] - bar_start_x
                norm_drag_pos = max(0, min(1, dragged_x_on_bar / bar_width))
                self.drag_current_frame = int(norm_drag_pos * total_video_frames)
                
                # Draw drag preview
                start_frame = min(self.drag_start_frame, self.drag_current_frame)
                end_frame = max(self.drag_start_frame, self.drag_current_frame)
                
                start_x = bar_start_x + (start_frame / total_video_frames) * bar_width
                end_x = bar_start_x + (end_frame / total_video_frames) * bar_width
                preview_width = max(2, end_x - start_x)
                
                # Draw semi-transparent preview rectangle
                preview_color = imgui.get_color_u32_rgba(0.2, 0.8, 0.2, 0.4)  # Green with transparency
                draw_list.add_rect_filled(start_x, bar_start_y, start_x + preview_width, bar_start_y + bar_height, preview_color)
                
                # Draw border
                border_color = imgui.get_color_u32_rgba(0.2, 0.8, 0.2, 0.8)
                draw_list.add_rect(start_x, bar_start_y, start_x + preview_width, bar_start_y + bar_height, border_color, thickness=2.0)

            if self.is_dragging_chapter_range and imgui.is_mouse_released(0):
                # Finish drag and create chapter
                start_frame = min(self.drag_start_frame, self.drag_current_frame)
                end_frame = max(self.drag_start_frame, self.drag_current_frame)
                
                if end_frame - start_frame >= 1:  # Minimum 1 frame difference
                    default_pos_key = self.position_short_name_keys[0] if self.position_short_name_keys else "N/A"
                    chapter_data = {
                        "start_frame_str": str(start_frame),
                        "end_frame_str": str(end_frame),
                        "segment_type": "SexAct", 
                        "position_short_name_key": default_pos_key,
                        "source": "drag_create"
                    }
                    
                    if self.app.funscript_processor:
                        # Chapter creation will auto-adjust for overlaps
                        self.app.funscript_processor.create_new_chapter_from_data(chapter_data)
                else:
                    self.app.logger.info("Drag too small - chapter not created", extra={'status_message': True})
                
                self.is_dragging_chapter_range = False

        # Reset drag if mouse leaves bar area
        if self.is_dragging_chapter_range and not is_mouse_over_bar:
            self.is_dragging_chapter_range = False

        # Chapter Deletion - Keyboard shortcuts (DELETE/BACKSPACE for selected chapters)
        if len(self.context_selected_chapters) > 0:
            shortcuts = self.app.app_settings.get("funscript_editor_shortcuts", {})
            io = imgui.get_io()
            
            # Delete Selected Chapter
            del_sc_str = shortcuts.get("delete_selected_chapter", "DELETE")
            del_alt_sc_str = shortcuts.get("delete_selected_chapter_alt", "BACKSPACE")
            del_key_tuple = self.app._map_shortcut_to_glfw_key(del_sc_str)
            bck_key_tuple = self.app._map_shortcut_to_glfw_key(del_alt_sc_str)
            delete_pressed = False

            if del_key_tuple and (
                imgui.is_key_pressed(del_key_tuple[0]) and
                all(m == io_m for m, io_m in
                    zip(del_key_tuple[1].values(), [io.key_shift, io.key_ctrl, io.key_alt, io.key_super]))
            ):
                delete_pressed = True

            if (not delete_pressed and bck_key_tuple and (
                imgui.is_key_pressed(bck_key_tuple[0]) and
                all(m == io_m for m, io_m in
                    zip(bck_key_tuple[1].values(), [io.key_shift, io.key_ctrl, io.key_alt, io.key_super]))
            )):
                delete_pressed = True

            if delete_pressed and self.context_selected_chapters:
                # Record undo action before deletion
                chapter_names = [ch.position_short_name for ch in self.context_selected_chapters]
                op_desc = f"Deleted {len(self.context_selected_chapters)} Selected Chapter(s) (Key): {', '.join(chapter_names)}"
                
                # Delete the chapters
                ch_ids = [ch.unique_id for ch in self.context_selected_chapters]
                fs_proc.delete_video_chapters_by_ids(ch_ids)
                self.context_selected_chapters.clear()
                
                self.app.logger.info(f"Deleted {len(ch_ids)} chapters via keyboard shortcut", extra={'status_message': True})

        imgui.set_cursor_screen_pos((bar_start_x, bar_start_y + bar_height))
        imgui.spacing()

    def _render_chapter_context_menu(self):
        fs_proc = self.app.funscript_processor
        if not fs_proc: return

        if imgui.begin_popup(self.chapter_bar_popup_id):
            num_selected = len(self.context_selected_chapters)
            can_select_one = num_selected == 1

            if imgui.menu_item("Seek to Beginning of Chapter", enabled=can_select_one)[0]:
                if can_select_one:
                    selected_chapter = self.context_selected_chapters[0]
                    if self.app.processor:
                        self.app.processor.seek_video(selected_chapter.start_frame_id)
                        # Ensure timeline synchronization after seeking
                        self.app.app_state_ui.force_timeline_pan_to_current_frame = True

            if imgui.menu_item("Seek to End of Chapter", enabled=can_select_one)[0]:
                if can_select_one:
                    selected_chapter = self.context_selected_chapters[0]
                    if self.app.processor:
                        self.app.processor.seek_video(selected_chapter.end_frame_id)
                        # Ensure timeline synchronization after seeking
                        self.app.app_state_ui.force_timeline_pan_to_current_frame = True

            imgui.separator()

            can_edit = num_selected == 1
            if imgui.menu_item("Edit Chapter", enabled=can_edit)[0]:
                if can_edit and self.context_selected_chapters:
                    chapter_obj_to_edit = self.context_selected_chapters[0]
                    self.chapter_to_edit_id = chapter_obj_to_edit.unique_id
                    self.chapter_edit_data = {
                        "start_frame_str": str(chapter_obj_to_edit.start_frame_id),
                        "end_frame_str": str(chapter_obj_to_edit.end_frame_id),
                        "segment_type": chapter_obj_to_edit.segment_type,
                        "position_short_name_key": chapter_obj_to_edit.position_short_name,
                        "source": chapter_obj_to_edit.source
                    }
                    try:
                        self.selected_position_idx_in_dialog = self.position_short_name_keys.index(
                            chapter_obj_to_edit.position_short_name)
                    except (ValueError, IndexError):
                        self.selected_position_idx_in_dialog = 0
                    self.show_edit_chapter_dialog = True

            # --- Menu item for setting chapter-specific ROI ---
            if imgui.menu_item("Set ROI & Point for this Chapter", enabled=can_edit)[0]:
                if can_edit:
                    selected_chapter = self.context_selected_chapters[0]
                    # Tell AppLogic which chapter we're setting the ROI for
                    self.app.chapter_id_for_roi_setting = selected_chapter.unique_id
                    # Enter the global ROI selection mode, which will now use the chapter_id
                    self.app.enter_set_user_roi_mode()
                    # Seek to the start of the chapter to make it easy for the user
                    if self.app.processor:
                        self.app.processor.seek_video(selected_chapter.start_frame_id)
                        # Ensure timeline synchronization after seeking
                        self.app.app_state_ui.force_timeline_pan_to_current_frame = True

            can_delete = num_selected > 0
            delete_label = f"Delete Selected Chapter(s) ({num_selected})" if num_selected > 0 else "Delete Selected Chapter(s)"
            shortcuts = self.app.app_settings.get("funscript_editor_shortcuts", {})
            delete_shortcut = shortcuts.get("delete_selected_chapter", "Delete")
            if imgui.menu_item(delete_label, shortcut=delete_shortcut, enabled=can_delete)[0]:
                if can_delete and self.context_selected_chapters:
                    ch_ids = [ch.unique_id for ch in self.context_selected_chapters]
                    fs_proc.delete_video_chapters_by_ids(ch_ids)
                    self.context_selected_chapters.clear()

            can_delete_points = num_selected > 0
            delete_points_label = f"Delete Points in Chapter(s) ({num_selected})" if num_selected > 0 else "Delete Points in Chapter(s)"
            if imgui.menu_item(delete_points_label, enabled=can_delete_points)[0]:
                if can_delete_points and self.context_selected_chapters:
                    fs_proc.clear_script_points_in_selected_chapters(self.context_selected_chapters)

            imgui.separator()

            can_select_points = num_selected > 0
            select_points_label = f"Select Points in {num_selected} Chapter{'s' if num_selected != 1 else ''}" if num_selected > 0 else "Select Points in Chapter(s)"
            if imgui.begin_menu(select_points_label, enabled=can_select_points):
                if imgui.menu_item("On Timeline 1")[0]:
                    if hasattr(self.app.funscript_processor, 'select_points_in_chapters'):
                        self.app.funscript_processor.select_points_in_chapters(self.context_selected_chapters, target_timeline='primary')
                # Disable Timeline 2 option if it's not visible
                timeline2_visible = self.app.app_state_ui.show_funscript_interactive_timeline2
                if imgui.menu_item("On Timeline 2", enabled=timeline2_visible)[0]:
                    if hasattr(self.app.funscript_processor, 'select_points_in_chapters'):
                        self.app.funscript_processor.select_points_in_chapters(self.context_selected_chapters, target_timeline='secondary')

                if imgui.menu_item("On Both Timelines")[0]:
                    if hasattr(self.app.funscript_processor, 'select_points_in_chapters'):
                        self.app.funscript_processor.select_points_in_chapters(self.context_selected_chapters, target_timeline='both')
                imgui.end_menu()

            # --- Apply Plugin to Chapter submenu ---
            can_apply_plugin = num_selected > 0  
            plugin_label = f"Apply Plugin to {num_selected} Chapter{'s' if num_selected != 1 else ''}" if num_selected > 0 else "Apply Plugin to Chapter(s)"
            if imgui.begin_menu(plugin_label, enabled=can_apply_plugin):
                if can_apply_plugin and self.context_selected_chapters:
                    # Timeline selection submenus with plugin lists
                    if imgui.begin_menu("Timeline 1 (Primary)"):
                        self._render_chapter_plugin_menu('primary')
                        imgui.end_menu()
                    
                    # Only show Timeline 2 if it's visible
                    timeline2_visible = self.app.app_state_ui.show_funscript_interactive_timeline2
                    if imgui.begin_menu("Timeline 2 (Secondary)", enabled=timeline2_visible):
                        self._render_chapter_plugin_menu('secondary')
                        imgui.end_menu()
                imgui.end_menu()

            # --- Analysis submenu ---
            can_analyze = num_selected == 1
            if imgui.begin_menu("Chapter Analysis", enabled=can_analyze):
                if can_analyze and self.context_selected_chapters:
                    self._render_dynamic_chapter_analysis_menu()
                imgui.end_menu()

            imgui.separator()

            can_standard_merge = num_selected == 2
            if imgui.menu_item("Merge Selected Chapters", enabled=can_standard_merge)[0]:
                if can_standard_merge and len(self.context_selected_chapters) == 2:
                    chaps_to_merge = sorted(self.context_selected_chapters, key=lambda c: c.start_frame_id)
                    if hasattr(fs_proc, 'merge_selected_chapters'):
                        fs_proc.merge_selected_chapters(chaps_to_merge[0], chaps_to_merge[1])
                        self.context_selected_chapters.clear()
                    else:
                        self.app.logger.warning("FunscriptProcessor needs 'merge_selected_chapters' method.")

            can_fill_gap_merge = False
            gap_fill_c1, gap_fill_c2 = None, None
            if len(self.context_selected_chapters) == 2:
                temp_chaps_fill_gap = sorted(self.context_selected_chapters, key=lambda c: c.start_frame_id)
                c1_fg_check, c2_fg_check = temp_chaps_fill_gap[0], temp_chaps_fill_gap[1]
                if c1_fg_check.end_frame_id < c2_fg_check.start_frame_id - 1:
                    can_fill_gap_merge = True
                    gap_fill_c1, gap_fill_c2 = c1_fg_check, c2_fg_check

            if imgui.menu_item("Track Gap & Merge Chapters", enabled=can_fill_gap_merge)[0]:  # Renamed
                if gap_fill_c1 and gap_fill_c2:  # gap_fill_c1 and c2 are sorted chapters defining the gap
                    self.app.logger.info(
                        f"UI Action: Initiating track gap then merge between {gap_fill_c1.unique_id} and {gap_fill_c2.unique_id}")

                    gap_start_frame = gap_fill_c1.end_frame_id + 1
                    gap_end_frame = gap_fill_c2.start_frame_id - 1

                    if gap_end_frame < gap_start_frame:
                        self.app.logger.warning("No actual gap to track. Merging directly (if possible).")
                        if hasattr(fs_proc, 'merge_selected_chapters'):  # Standard merge
                            merged_chapter = fs_proc.merge_selected_chapters(gap_fill_c1, gap_fill_c2, return_chapter_object=True)
                            if merged_chapter:
                                self.context_selected_chapters = [merged_chapter]
                            else:
                                self.context_selected_chapters.clear()
                        imgui.close_current_popup()  # Close context menu
                        # No further action needed if no gap
                        return  # Exit this handler early

                    # Record current funscript state for potential UNDO of the whole operation (tracking + merge)
                    fs_proc._record_timeline_action(1, f"Prepare for Gap Track & Merge: {gap_fill_c1.unique_id[:4]}+{gap_fill_c2.unique_id[:4]}")
                    # If secondary axis is also involved, record for it too.

                    # Set up AppLogic state for post-tracking action
                    self.app.set_pending_action_after_tracking(
                        action_type='finalize_gap_merge_after_tracking',
                        chapter1_id=gap_fill_c1.unique_id,
                        chapter2_id=gap_fill_c2.unique_id
                        # gap_start_frame and gap_end_frame are implicitly handled by the script range now
                    )

                    # Set scripting range to ONLY THE GAP
                    fs_proc.scripting_start_frame = gap_start_frame
                    fs_proc.scripting_end_frame = gap_end_frame
                    fs_proc.scripting_range_active = True
                    fs_proc.selected_chapter_for_scripting = None  # It's not an existing chapter being scripted
                    self.app.project_manager.project_dirty = True

                    # Start the tracker for the gap
                    self._start_live_tracking(
                        success_info=(
                            f"Tracker started for gap between {gap_fill_c1.position_short_name} and {gap_fill_c2.position_short_name} "
                            f"(Frames: {gap_start_frame}-{gap_end_frame})"
                        ),
                        on_error_clear_pending_action=True
                    )

                    self.context_selected_chapters.clear()  # Clear selection as process is underway
                    imgui.close_current_popup()

            can_bridge_gap_and_track = False
            bridge_ch1, bridge_ch2 = None, None
            actual_gap_start, actual_gap_end = 0, 0
            if len(self.context_selected_chapters) == 2:
                temp_chaps_bridge_gap = sorted(self.context_selected_chapters, key=lambda c: c.start_frame_id)
                c1_bg_check, c2_bg_check = temp_chaps_bridge_gap[0], temp_chaps_bridge_gap[1]
                if c1_bg_check.end_frame_id < c2_bg_check.start_frame_id - 1:
                    current_actual_gap_start = c1_bg_check.end_frame_id + 1
                    current_actual_gap_end = c2_bg_check.start_frame_id - 1
                    if current_actual_gap_end >= current_actual_gap_start:
                        can_bridge_gap_and_track = True
                        bridge_ch1, bridge_ch2 = c1_bg_check, c2_bg_check
                        actual_gap_start, actual_gap_end = current_actual_gap_start, current_actual_gap_end

            if imgui.menu_item("Create Chapter in Gap & Track", enabled=can_bridge_gap_and_track)[0]:
                if bridge_ch1 and bridge_ch2:
                    self.app.logger.info(
                        f"UI Action: Creating new chapter in gap between {bridge_ch1.unique_id} and {bridge_ch2.unique_id}")
                    gap_chapter_data = {
                        "start_frame_str": str(actual_gap_start),
                        "end_frame_str": str(actual_gap_end),
                        "segment_type": bridge_ch1.segment_type,
                        "position_short_name_key": bridge_ch1.position_short_name,
                        "source": "manual_gap_fill_track"
                    }
                    new_gap_chapter = fs_proc.create_new_chapter_from_data(gap_chapter_data, return_chapter_object=True)
                    if new_gap_chapter:
                        self.context_selected_chapters = [new_gap_chapter]
                        if hasattr(self.app.funscript_processor, 'set_scripting_range_from_chapter'):
                            self.app.funscript_processor.set_scripting_range_from_chapter(new_gap_chapter)
                            self._start_live_tracking(
                                success_info=f"Tracker started for new gap chapter: {new_gap_chapter.unique_id}"
                            )
                        else:
                            self.app.logger.error("set_scripting_range_from_chapter not found in funscript_processor.")
                    else:
                        self.app.logger.error(
                            "Failed to create new chapter in gap, or it was not returned by create_new_chapter_from_data.")
                        self.context_selected_chapters.clear()
            imgui.separator()

            can_start_tracker = num_selected == 1
            if imgui.menu_item("Start Tracker in Chapter", enabled=can_start_tracker)[0]:
                if can_start_tracker and len(self.context_selected_chapters) == 1:
                    selected_chapter = self.context_selected_chapters[0]
                    self.app.logger.info(
                        f"UI Action: Setting range and starting tracker for chapter {selected_chapter.unique_id}")
                    if hasattr(self.app.funscript_processor, 'set_scripting_range_from_chapter'):
                        self.app.funscript_processor.set_scripting_range_from_chapter(selected_chapter)
                        self._start_live_tracking(
                            success_info=f"Tracker started for chapter: {selected_chapter.position_short_name}"
                        )
                    else:
                        self.app.logger.error("set_scripting_range_from_chapter not found in funscript_processor.")
            # --- Split Chapter ---
            can_split = False
            # Use the frame position when the context menu was opened, not the current frame
            # This ensures the split happens where the user right-clicked, even if they seek afterwards
            split_frame = self.context_menu_opened_at_frame
            split_pos_key = None
            if num_selected == 1 and self.context_selected_chapters:
                chapter = self.context_selected_chapters[0]
                if split_frame is not None and chapter.start_frame_id < split_frame < chapter.end_frame_id:
                    # Find the key in POSITION_INFO_MAPPING whose short_name matches the chapter's position_short_name
                    from config.constants import POSITION_INFO_MAPPING
                    for key, info in POSITION_INFO_MAPPING.items():
                        if info.get("short_name") == chapter.position_short_name:
                            split_pos_key = key
                            break
                    else:
                        split_pos_key = chapter.position_short_name
                    can_split = True
            if imgui.menu_item("Split Chapter", enabled=can_split)[0]:
                if can_split and split_frame is not None and split_pos_key is not None:
                    original_end_frame = chapter.end_frame_id
                    fs_proc.update_chapter_from_data(
                        chapter.unique_id,
                        {
                            "start_frame_str": str(chapter.start_frame_id),
                            "end_frame_str": str(split_frame),
                            "position_short_name_key": split_pos_key
                        }
                    )
                    fs_proc.create_new_chapter_from_data(
                        {
                            "start_frame_str": str(split_frame + 1),
                            "end_frame_str": str(original_end_frame),
                            "position_short_name_key": split_pos_key,
                            "segment_type": chapter.segment_type,
                            "source": chapter.source
                        }
                    )
                    self.context_selected_chapters.clear()
                    imgui.close_current_popup()
                    imgui.end_popup()
                    return
            imgui.end_popup()

    def _render_create_chapter_window(self):
        if not self.show_create_chapter_dialog:
            return
        window_flags = imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_COLLAPSE
        io = imgui.get_io()
        if io.display_size[0] > 0 and io.display_size[1] > 0:
            main_viewport = imgui.get_main_viewport()
            center_x = main_viewport.pos[0] + main_viewport.size[0] * 0.5
            center_y = main_viewport.pos[1] + main_viewport.size[1] * 0.5
            imgui.set_next_window_position(center_x, center_y, imgui.APPEARING, 0.5, 0.5)

        is_not_collapsed, self.show_create_chapter_dialog = imgui.begin(
            "Create New Chapter##CreateWindow",
            closable=True,
            flags=window_flags
        )

        if is_not_collapsed and self.show_create_chapter_dialog:
            imgui.text("Create New Chapter Details")
            imgui.separator()
            imgui.push_item_width(200)
            _, self.chapter_edit_data["start_frame_str"] = imgui.input_text("Start Frame##CreateWin", self.chapter_edit_data.get("start_frame_str", "0"), 64)
            _, self.chapter_edit_data["end_frame_str"] = imgui.input_text("End Frame##CreateWin", self.chapter_edit_data.get("end_frame_str", "0"), 64)
            _, self.chapter_edit_data["segment_type"] = imgui.input_text("Segment Type##CreateWin", self.chapter_edit_data.get("segment_type", "SexAct"), 128)
            clicked_pos, self.selected_position_idx_in_dialog = imgui.combo("Position##CreateWin", self.selected_position_idx_in_dialog, self.position_display_names)
            if clicked_pos and self.position_short_name_keys and 0 <= self.selected_position_idx_in_dialog < len(
                    self.position_short_name_keys):
                self.chapter_edit_data["position_short_name_key"] = self.position_short_name_keys[
                    self.selected_position_idx_in_dialog]
            current_selected_key = self.chapter_edit_data.get("position_short_name_key")
            long_name_display = POSITION_INFO_MAPPING.get(current_selected_key, {}).get("long_name", "N/A") if current_selected_key else "N/A"
            imgui.text_disabled(f"Long Name (auto): {long_name_display}")
            _, self.chapter_edit_data["source"] = imgui.input_text("Source##CreateWin", self.chapter_edit_data.get("source", "manual"), 64)

            if imgui.button("Set Range##ChapterCreateSetRangeWinBtn"):
                self._set_chapter_range_by_selection()
            if imgui.is_item_hovered():
                imgui.set_tooltip("Set chapter range from timeline selection")

            # New enhanced chapter creation buttons
            imgui.same_line()
            if imgui.button("Set Start##ChapterCreateSetStartBtn"):
                self._set_chapter_start_to_current_frame()
            if imgui.is_item_hovered():
                imgui.set_tooltip("Set chapter start to current frame")

            imgui.same_line()
            if imgui.button("Set End##ChapterCreateSetEndBtn"):
                self._set_chapter_end_to_current_frame()
            if imgui.is_item_hovered():
                imgui.set_tooltip("Set chapter end to current frame")
            
            # Show current frame info for reference
            current_frame = self._get_current_frame()
            imgui.text_disabled(f"Current frame: {current_frame}")

            imgui.pop_item_width()
            imgui.separator()

            # Get icon texture manager
            icon_mgr = get_icon_texture_manager()
            plus_circle_tex, _, _ = icon_mgr.get_icon_texture('plus-circle.png')
            btn_size = imgui.get_frame_height()

            # Create button with icon and PRIMARY styling (positive action)
            with primary_button_style():
                clicked = False
                if plus_circle_tex:
                    if imgui.image_button(plus_circle_tex, btn_size, btn_size):
                        clicked = True
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("Create new chapter")
                    imgui.same_line()
                    imgui.text("Create")
                else:
                    clicked = imgui.button("Create##ChapterCreateWinBtn")
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("Create new chapter")

                if clicked and self.app.funscript_processor:
                    self.app.funscript_processor.create_new_chapter_from_data(self.chapter_edit_data.copy())
                    self.show_create_chapter_dialog = False
            imgui.same_line()
            if imgui.button("Cancel##ChapterCreateWinCancelBtn"):
                self.show_create_chapter_dialog = False
            if imgui.is_item_hovered():
                imgui.set_tooltip("Cancel chapter creation")
        imgui.end()

    def _render_edit_chapter_window(self):
        if not self.show_edit_chapter_dialog or not self.chapter_to_edit_id:
            if not self.show_edit_chapter_dialog: self.chapter_to_edit_id = None
            return
        window_flags = imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_COLLAPSE
        io = imgui.get_io()
        if io.display_size[0] > 0 and io.display_size[1] > 0:
            main_viewport = imgui.get_main_viewport()
            center_x = main_viewport.pos[0] + main_viewport.size[0] * 0.5
            center_y = main_viewport.pos[1] + main_viewport.size[1] * 0.5
            imgui.set_next_window_position(center_x, center_y, imgui.APPEARING, 0.5, 0.5)

        is_not_collapsed, self.show_edit_chapter_dialog = imgui.begin(
            f"Edit Chapter: {self.chapter_to_edit_id[:8]}...##EditChapterWindow",
            closable=True,
            flags=window_flags
        )
        if not self.show_edit_chapter_dialog:
            self.chapter_to_edit_id = None

        if is_not_collapsed and self.show_edit_chapter_dialog:
            imgui.text(f"Editing Chapter ID: {self.chapter_to_edit_id}")
            imgui.separator()
            imgui.push_item_width(200)
            _, self.chapter_edit_data["start_frame_str"] = imgui.input_text("Start Frame##EditWin", self.chapter_edit_data.get("start_frame_str", "0"), 64)
            _, self.chapter_edit_data["end_frame_str"] = imgui.input_text("End Frame##EditWin", self.chapter_edit_data.get("end_frame_str", "0"), 64)
            _, self.chapter_edit_data["segment_type"] = imgui.input_text("Segment Type##EditWin", self.chapter_edit_data.get("segment_type", ""), 128)
            current_pos_key_for_edit = self.chapter_edit_data.get("position_short_name_key")
            try:
                if self.position_short_name_keys and current_pos_key_for_edit in self.position_short_name_keys:
                    self.selected_position_idx_in_dialog = self.position_short_name_keys.index(current_pos_key_for_edit)
                elif self.position_short_name_keys:  # Default to first if current is invalid but list exists
                    self.selected_position_idx_in_dialog = 0
                    self.chapter_edit_data["position_short_name_key"] = self.position_short_name_keys[0]
                else:  # No positions available
                    self.selected_position_idx_in_dialog = 0
            except ValueError:  # Should not happen if above logic is correct, but as a fallback
                self.selected_position_idx_in_dialog = 0
                if self.position_short_name_keys: self.chapter_edit_data["position_short_name_key"] = self.position_short_name_keys[0]

            clicked_pos_edit, self.selected_position_idx_in_dialog = imgui.combo("Position##EditWin", self.selected_position_idx_in_dialog, self.position_display_names)
            if clicked_pos_edit and self.position_short_name_keys and 0 <= self.selected_position_idx_in_dialog < len(
                    self.position_short_name_keys):
                self.chapter_edit_data["position_short_name_key"] = self.position_short_name_keys[
                    self.selected_position_idx_in_dialog]
            pos_key_edit_display = self.chapter_edit_data.get("position_short_name_key")
            long_name_display_edit = POSITION_INFO_MAPPING.get(pos_key_edit_display, {}).get("long_name", "N/A") if pos_key_edit_display else "N/A"
            imgui.text_disabled(f"Long Name (auto): {long_name_display_edit}")
            _, self.chapter_edit_data["source"] = imgui.input_text("Source##EditWin", self.chapter_edit_data.get("source", ""), 64)

            if imgui.button("Set Range##ChapterUpdateSetRangeWinBtn"):
                self._set_chapter_range_by_selection()
            if imgui.is_item_hovered():
                imgui.set_tooltip("Set chapter range from timeline selection")

            # Enhanced chapter editing buttons
            imgui.same_line()
            if imgui.button("Set Start##ChapterEditSetStartBtn"):
                self._set_chapter_start_to_current_frame()
            if imgui.is_item_hovered():
                imgui.set_tooltip("Set chapter start to current frame")

            imgui.same_line()
            if imgui.button("Set End##ChapterEditSetEndBtn"):
                self._set_chapter_end_to_current_frame()
            if imgui.is_item_hovered():
                imgui.set_tooltip("Set chapter end to current frame")
            
            # Show current frame info for reference
            current_frame = self._get_current_frame()
            imgui.text_disabled(f"Current frame: {current_frame}")

            imgui.pop_item_width()
            imgui.separator()

            # Get icon texture manager
            icon_mgr = get_icon_texture_manager()
            save_tex, _, _ = icon_mgr.get_icon_texture('save-as.png')
            btn_size = imgui.get_frame_height()

            # Save button with icon and PRIMARY styling (positive action)
            with primary_button_style():
                clicked = False
                if save_tex:
                    if imgui.image_button(save_tex, btn_size, btn_size):
                        clicked = True
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("Save chapter changes")
                    imgui.same_line()
                    imgui.text("Save")
                else:
                    clicked = imgui.button("Save##ChapterEditWinBtn")
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("Save chapter changes")

                if clicked and self.app.funscript_processor and self.chapter_to_edit_id:
                    self.app.funscript_processor.update_chapter_from_data(self.chapter_to_edit_id, self.chapter_edit_data.copy())
                    self.show_edit_chapter_dialog = False
                    self.chapter_to_edit_id = None
            imgui.same_line()
            if imgui.button("Cancel##ChapterEditWinCancelBtn"):
                self.show_edit_chapter_dialog = False
                self.chapter_to_edit_id = None
            if imgui.is_item_hovered():
                imgui.set_tooltip("Cancel chapter editing")
        imgui.end()
        if not self.show_edit_chapter_dialog:
            self.chapter_to_edit_id = None

    def _set_chapter_range_by_selection(self):
        selected_idxs = []
        t1_selected_idxs = self.gui_instance.timeline_editor1.multi_selected_action_indices
        t2_selected_idxs = self.gui_instance.timeline_editor2.multi_selected_action_indices
        fs = self.app.processor.tracker.funscript
        fs_actions = []
        # Take selection from either, primary if both
        if len(t1_selected_idxs) >= 2:
            selected_idxs = t1_selected_idxs
            fs_actions = fs.primary_actions
        elif len(t2_selected_idxs) >= 2:
            selected_idxs = t2_selected_idxs
            fs_actions = fs.secondary_actions

        if len(selected_idxs) < 2:
            return

        v_info = self.app.processor.video_info
        start_action_ms = fs_actions[min(selected_idxs)]['at']
        end_action_ms = fs_actions[max(selected_idxs)]['at']

        start_frame = VideoSegment.ms_to_frame_idx(ms=start_action_ms, total_frames=v_info['total_frames'], fps=v_info['fps'])
        end_frame = VideoSegment.ms_to_frame_idx(ms=end_action_ms, total_frames=v_info['total_frames'], fps=v_info['fps'])

        self.chapter_edit_data["start_frame_str"] = str(start_frame)
        self.chapter_edit_data["end_frame_str"] = str(end_frame)
    
    def _get_current_frame(self) -> int:
        """Get the current video frame position."""
        if self.app.processor and hasattr(self.app.processor, 'current_frame_index'):
            return max(0, self.app.processor.current_frame_index)
        return 0
    
    def _set_chapter_start_to_current_frame(self):
        """Set the chapter start frame to the current video frame."""
        current_frame = self._get_current_frame()
        self.chapter_edit_data["start_frame_str"] = str(current_frame)
        self.app.logger.info(f"Chapter start set to frame {current_frame}", extra={'status_message': True})
    
    def _set_chapter_end_to_current_frame(self):
        """Set the chapter end frame to the current video frame."""
        current_frame = self._get_current_frame()
        self.chapter_edit_data["end_frame_str"] = str(current_frame)
        self.app.logger.info(f"Chapter end set to frame {current_frame}", extra={'status_message': True})

    def _render_funscript_timeline_preview(self, total_duration_s: float, graph_height: int):
        self.gui_instance.render_funscript_timeline_preview(total_duration_s, graph_height)

    def _render_funscript_heatmap_preview(self, total_video_duration_s: float, bar_width_float: float, bar_height_float: float):
        # bar_width_float here is nav_content_width
        self.gui_instance.render_funscript_heatmap_preview(total_video_duration_s, bar_width_float, bar_height_float)

    def _render_chapter_tooltip(self):
        # Make sure chapter_tooltip_segment is valid before trying to access its attributes
        if not self.chapter_tooltip_segment or not hasattr(self.chapter_tooltip_segment, 'class_name'):
            return

        imgui.begin_tooltip()
        segment = self.chapter_tooltip_segment

        fs_proc = self.app.funscript_processor
        chapter_number_str = "N/A"
        if fs_proc and fs_proc.video_chapters:
            sorted_chapters = sorted(fs_proc.video_chapters, key=lambda c: c.start_frame_id)
            try:
                chapter_index = sorted_chapters.index(segment)
                chapter_number_str = str(chapter_index + 1)
            except ValueError:
                # Fallback to ID search if object identity fails
                for i, chap in enumerate(sorted_chapters):
                    if chap.unique_id == segment.unique_id:
                        chapter_number_str = str(i + 1)
                        break

        imgui.text(f"Chapter #{chapter_number_str}: {segment.position_short_name} ({segment.segment_type})")
        imgui.text(f"Pos:  {segment.position_long_name}")
        imgui.text(f"Source: {segment.source}")
        imgui.text(f"Frames: {segment.start_frame_id} - {segment.end_frame_id}")

        fps_tt = self._get_current_fps()
        start_t_tt = segment.start_frame_id / fps_tt if fps_tt > 0 else 0
        end_t_tt = segment.end_frame_id / fps_tt if fps_tt > 0 else 0
        imgui.text(f"Time: {_format_time(self.app, start_t_tt)} - {_format_time(self.app, end_t_tt)}")
        imgui.end_tooltip()

    def _render_chapter_plugin_menu(self, target_timeline: str):
        """Render plugin selection menu for chapter-based plugin application.
        
        Args:
            target_timeline: 'primary' or 'secondary' - which timeline to target
        """
        if not self.context_selected_chapters:
            imgui.text_disabled("No chapters selected")
            return
            
        # Get the appropriate timeline instance
        timeline_instance = None
        timeline_num = None
        
        if target_timeline == 'primary':
            timeline_instance = self.gui_instance.timeline_editor1
            timeline_num = 1
        elif target_timeline == 'secondary':
            timeline_instance = self.gui_instance.timeline_editor2
            timeline_num = 2
        else:
            imgui.text_disabled("Invalid timeline")
            return
            
        # Get funscript and axis using the same method as timeline
        target_funscript, axis_name = self.app.funscript_processor._get_target_funscript_object_and_axis(timeline_num)
            
        if not timeline_instance or not target_funscript:
            imgui.text_disabled("Timeline not available")
            return
            
        # Get available plugins from the timeline's plugin renderer
        if not hasattr(timeline_instance, 'plugin_renderer') or not timeline_instance.plugin_renderer:
            imgui.text_disabled("Plugin system not available")
            return
            
        plugin_manager = timeline_instance.plugin_renderer.plugin_manager
        available_plugins = plugin_manager.get_available_plugins()
        
        if not available_plugins:
            imgui.text_disabled("No plugins available")
            return
            
        chapter_count = len(self.context_selected_chapters)
        chapter_text = "chapter" if chapter_count == 1 else f"{chapter_count} chapters"
        
        # Render plugin menu items
        for plugin_name in sorted(available_plugins):
            ui_data = plugin_manager.get_plugin_ui_data(plugin_name)
            if not ui_data or not ui_data['available']:
                continue
                
            display_name = ui_data['display_name']
            if imgui.menu_item(f"{display_name}")[0]:
                # Apply plugin to chapter(s) using the same logic as timeline selection
                self._apply_plugin_to_chapters(
                    plugin_name, 
                    target_timeline, 
                    timeline_instance,
                    plugin_manager
                )
                imgui.close_current_popup()
                
    def _apply_plugin_to_chapters(self, plugin_name: str, target_timeline: str, 
                                 timeline_instance, plugin_manager):
        """Apply a plugin to all points within selected chapter(s).
        
        This method:
        1. Selects all points within the chapter time ranges on the target timeline
        2. Uses the exact same PluginUIRenderer code paths as timeline selection menu
        """
        try:
            # Step 1: Select all points in the chapters on the target timeline
            if hasattr(self.app.funscript_processor, 'select_points_in_chapters'):
                # Use the existing method to select points in chapters
                self.app.funscript_processor.select_points_in_chapters(
                    self.context_selected_chapters, 
                    target_timeline=target_timeline
                )
            else:
                self.app.logger.error("select_points_in_chapters method not available")
                return
                
            # Step 2: Get the selected indices from the timeline
            selected_indices = list(timeline_instance.multi_selected_action_indices) if timeline_instance.multi_selected_action_indices else []
            
            if len(selected_indices) < 2:
                chapter_names = [ch.position_short_name for ch in self.context_selected_chapters]
                self.app.logger.warning(f"No points found in chapter(s) {chapter_names} on {target_timeline} timeline")
                return
                
            # Step 3: Apply plugin using EXACT same logic as timeline selection menu
            ui_data = plugin_manager.get_plugin_ui_data(plugin_name)
            
            if ui_data and timeline_instance.plugin_renderer._should_apply_directly(plugin_name, ui_data):
                # Direct application (like Invert, Ultimate Autotune)
                context = plugin_manager.plugin_contexts.get(plugin_name)
                if context:
                    context.apply_requested = True
                    # Force apply_to_selection for chapter context  
                    if hasattr(context, 'apply_to_selection'):
                        context.apply_to_selection = True
                    
                    chapter_names = [ch.position_short_name for ch in self.context_selected_chapters]
                    chapter_text = chapter_names[0] if len(chapter_names) == 1 else f"{len(chapter_names)} chapters"
                    self.app.logger.info(
                        f"Applied {ui_data['display_name']} to {len(selected_indices)} points in {chapter_text} on {target_timeline} timeline",
                        extra={"status_message": True}
                    )
            else:
                # Open configuration window (like other filters)
                from application.classes.plugin_ui_manager import PluginUIState
                plugin_manager.set_plugin_state(plugin_name, PluginUIState.OPEN)
                
                # Force apply_to_selection for chapter context
                context = plugin_manager.plugin_contexts.get(plugin_name)
                if context and hasattr(context, 'apply_to_selection'):
                    context.apply_to_selection = True
                    
                chapter_names = [ch.position_short_name for ch in self.context_selected_chapters]
                chapter_text = chapter_names[0] if len(chapter_names) == 1 else f"{len(chapter_names)} chapters"
                self.app.logger.info(
                    f"Opened {ui_data['display_name']} configuration for {len(selected_indices)} points in {chapter_text} on {target_timeline} timeline",
                    extra={"status_message": True}
                )
            
            # Clear chapter selection
            self.context_selected_chapters.clear()
            
        except Exception as e:
            self.app.logger.error(f"Error applying plugin {plugin_name} to chapters: {e}")
            import traceback
            traceback.print_exc()

    def _render_dynamic_chapter_analysis_menu(self):
        """Render dynamic chapter analysis menu with all available trackers."""
        if not self.context_selected_chapters:
            imgui.text_disabled("No chapter selected")
            return
            
        selected_chapter = self.context_selected_chapters[0]
        
        try:
            from config.tracker_discovery import get_tracker_discovery, TrackerCategory
            discovery = get_tracker_discovery()
            
            # Get all tracker categories
            live_trackers = discovery.get_trackers_by_category(TrackerCategory.LIVE)
            live_intervention_trackers = discovery.get_trackers_by_category(TrackerCategory.LIVE_INTERVENTION)
            offline_trackers = discovery.get_trackers_by_category(TrackerCategory.OFFLINE)
            
            # Group trackers by category for better organization
            tracker_groups = [
                ("Live Trackers", live_trackers),
                ("Live Intervention Trackers", live_intervention_trackers), 
                ("Offline Trackers", offline_trackers)
            ]
            
            tracker_found = False
            
            for group_name, trackers in tracker_groups:
                if not trackers:
                    continue
                    
                if tracker_found:  # Add separator between groups
                    imgui.separator()
                    
                # Render group header (non-clickable)
                imgui.text_colored(group_name, 0.7, 0.7, 0.7, 1.0)
                
                for tracker in trackers:
                    display_name = getattr(tracker, 'display_name', tracker.internal_name)
                    
                    # Only show the tracker name, not the description
                    menu_text = display_name
                    
                    if imgui.menu_item(menu_text)[0]:
                        self._apply_tracker_to_chapter(tracker, selected_chapter)
                        imgui.close_current_popup()
                
                tracker_found = True
            
            if not tracker_found:
                imgui.text_disabled("No trackers available")
                
        except Exception as e:
            self.app.logger.error(f"Error loading trackers for chapter analysis: {e}")
            imgui.text_colored("Error loading trackers", 1.0, 0.3, 0.3, 1.0)
            
    def _apply_tracker_to_chapter(self, tracker, chapter):
        """Apply a specific tracker to a chapter."""
        try:
            # Set the selected tracker
            self.app.app_state_ui.selected_tracker_name = tracker.internal_name
            
            # Set scripting range to the selected chapter
            if hasattr(self.app.funscript_processor, 'set_scripting_range_from_chapter'):
                self.app.funscript_processor.set_scripting_range_from_chapter(chapter)
                
                # Start tracking with descriptive message
                display_name = getattr(tracker, 'display_name', tracker.internal_name)
                self._start_live_tracking(
                    success_info=f"Started {display_name} analysis for chapter: {chapter.position_short_name}"
                )
                
                self.context_selected_chapters.clear()
            else:
                self.app.logger.error("set_scripting_range_from_chapter not found in funscript_processor.")
                
        except Exception as e:
            self.app.logger.error(f"Error applying tracker {tracker.internal_name} to chapter: {e}")


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
            if not can_merge:
                imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
                imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)

            # Merge Selected button (PRIMARY - positive action)
            with primary_button_style():
                if imgui.button("Merge Selected"):
                    if can_merge:
                        chaps_to_merge = sorted(self.list_context_selected_chapters, key=lambda c: c.start_frame_id)
                        fs_proc.merge_selected_chapters(chaps_to_merge[0], chaps_to_merge[1])
                        self.list_context_selected_chapters.clear()

            if not can_merge:
                imgui.pop_style_var()
                imgui.internal.pop_item_flag()
            if imgui.is_item_hovered():
                imgui.set_tooltip("Select exactly two chapters to merge.")

            imgui.same_line()

            # --- Track Gap & Merge Button ---
            can_track_gap, gap_c1, gap_c2, _, _ = self._get_gap_info(fs_proc)
            if not can_track_gap:
                imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
                imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)

            # Track Gap & Merge button (PRIMARY - positive action)
            with primary_button_style():
                if imgui.button("Track Gap & Merge"):
                    if can_track_gap:
                        self._handle_track_gap_and_merge(fs_proc, gap_c1, gap_c2)
                        self.list_context_selected_chapters.clear()

            if not can_track_gap:
                imgui.pop_style_var()
                imgui.internal.pop_item_flag()
            if imgui.is_item_hovered():
                imgui.set_tooltip("Select two chapters with a frame gap between them to track the gap and merge.")

            imgui.same_line()

            # --- Create Chapter in Gap & Track Button ---
            can_create_in_gap, create_c1, _, gap_start, gap_end = self._get_gap_info(fs_proc)
            if not can_create_in_gap:
                imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
                imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)

            # Create Chapter in Gap & Track button (PRIMARY - positive action)
            with primary_button_style():
                if imgui.button("Create Chapter in Gap & Track"):
                    if can_create_in_gap:
                        self._handle_create_chapter_in_gap(fs_proc, create_c1, gap_start, gap_end)
                        self.list_context_selected_chapters.clear()

            if not can_create_in_gap:
                imgui.pop_style_var()
                imgui.internal.pop_item_flag()
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

            if imgui.begin_table("ChapterListTable", 8, flags=table_flags):
                imgui.table_setup_column("##Select", init_width_or_weight=0.15)
                imgui.table_setup_column("#", init_width_or_weight=0.15)
                imgui.table_setup_column("Preview", init_width_or_weight=0.8)
                imgui.table_setup_column("Color", init_width_or_weight=0.25)
                imgui.table_setup_column("Position", init_width_or_weight=1.0)
                imgui.table_setup_column("Start", init_width_or_weight=0.9)
                imgui.table_setup_column("End", init_width_or_weight=0.9)
                imgui.table_setup_column("Actions", init_width_or_weight=0.6)
                imgui.table_headers_row()

                # Get FPS once for time calculation
                fps = self.app.processor.fps if self.app.processor and self.app.processor.fps > 0 else DEFAULT_CHAPTER_FPS
                sorted_chapters = sorted(fs_proc.video_chapters, key=lambda c: c.start_frame_id)
                chapters_to_remove_from_selection = []

                for i, chapter in enumerate(list(sorted_chapters)):
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

        fs_proc._record_timeline_action(1, f"Prepare for Gap Track & Merge: {c1.unique_id[:4]}+{c2.unique_id[:4]}")
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
        self.nav_ui.show_edit_chapter_dialog = True
