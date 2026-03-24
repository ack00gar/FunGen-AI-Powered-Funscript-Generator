import imgui
from typing import Optional

from application.utils import _format_time, VideoSegment, get_icon_texture_manager, primary_button_style, destructive_button_style
from config.constants import POSITION_INFO_MAPPING, DEFAULT_CHAPTER_FPS
from config.element_group_colors import VideoNavigationColors
from config.constants_colors import CurrentTheme



class ChapterBarMixin:
    """Mixin fragment for VideoNavigationUI."""

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

        # Empty state hint when no chapters exist (non-blocking — mouse interaction still runs)
        if not fs_proc.video_chapters:
            hint = "No chapters - use I/O keys, drag on bar, or run chapter detection"
            hint_size = imgui.calc_text_size(hint)
            hint_x = bar_start_x + (bar_width - hint_size[0]) * 0.5
            hint_y = bar_start_y + (bar_height - hint_size[1]) * 0.5
            draw_list.add_text(hint_x, hint_y,
                               imgui.get_color_u32_rgba(0.5, 0.5, 0.55, 0.6), hint)

        # For marker alignment with slider track, calculate effective start_x and width
        effective_marker_area_start_x = bar_start_x + style.frame_padding[0]
        effective_marker_area_width = bar_width - (style.frame_padding[0] * 2)

        self.chapter_tooltip_segment = None
        action_on_segment_this_frame = False

        # Pre-compute constants outside the segment loop
        text_line_height = imgui.get_text_line_height()
        icon_color_u32 = imgui.get_color_u32_rgba(*VideoNavigationColors.ICON)
        scripting_border_u32 = imgui.get_color_u32_rgba(*VideoNavigationColors.SCRIPTING_BORDER)
        sel_primary_u32 = imgui.get_color_u32_rgba(*VideoNavigationColors.SELECTION_PRIMARY)
        sel_secondary_u32 = imgui.get_color_u32_rgba(*VideoNavigationColors.SELECTION_SECONDARY)
        text_black_u32 = imgui.get_color_u32_rgba(*VideoNavigationColors.TEXT_BLACK)
        text_white_u32 = imgui.get_color_u32_rgba(*VideoNavigationColors.TEXT_WHITE)
        default_gray_color = (*CurrentTheme.GRAY_MEDIUM[:3], 0.7)

        # Pre-compute selection lookup sets for O(1) membership tests
        sel_primary_id = self.context_selected_chapters[0].unique_id if len(self.context_selected_chapters) > 0 else None
        sel_secondary_id = self.context_selected_chapters[1].unique_id if len(self.context_selected_chapters) > 1 else None

        # Pre-compute scripting chapter id
        scripting_chapter_id = None
        if fs_proc.scripting_range_active and fs_proc.selected_chapter_for_scripting:
            scripting_chapter_id = fs_proc.selected_chapter_for_scripting.unique_id
        scripting_start = fs_proc.scripting_start_frame
        scripting_end = fs_proc.scripting_end_frame

        inv_total_frames = 1.0 / total_video_frames

        for segment_idx, segment in enumerate(fs_proc.video_chapters):
            start_x_norm = segment.start_frame_id * inv_total_frames
            end_x_norm = segment.end_frame_id * inv_total_frames
            seg_start_x = bar_start_x + start_x_norm * bar_width
            seg_end_x = bar_start_x + end_x_norm * bar_width
            seg_width = max(1, seg_end_x - seg_start_x)

            if segment.user_roi_fixed:
                icon_pos_x = seg_start_x + 3
                icon_pos_y = bar_start_y + (bar_height - text_line_height) / 2
                draw_list.add_text(icon_pos_x, icon_pos_y, icon_color_u32, "[R]")

            segment_color_tuple = segment.color
            if not (isinstance(segment_color_tuple, (tuple, list)) and len(segment_color_tuple) in [3, 4]):
                self.app.logger.warning(
                    f"Segment {segment.unique_id} ('{segment.class_name if hasattr(segment, 'class_name') else 'N/A'}') has invalid color {segment_color_tuple}, using default gray.")
                segment_color_tuple = default_gray_color
            seg_color = imgui.get_color_u32_rgba(*segment_color_tuple)

            seg_uid = segment.unique_id
            is_selected_for_scripting = (scripting_chapter_id is not None
                and scripting_chapter_id == seg_uid
                and scripting_start == segment.start_frame_id
                and scripting_end == segment.end_frame_id)

            draw_list.add_rect_filled(seg_start_x, bar_start_y, seg_start_x + seg_width, bar_start_y + bar_height, seg_color)

            is_context_selected_primary = (sel_primary_id == seg_uid)
            is_context_selected_secondary = (sel_secondary_id == seg_uid)

            if is_selected_for_scripting:
                draw_list.add_rect(seg_start_x + 0.5, bar_start_y + 0.5, seg_start_x + seg_width - 0.5, bar_start_y + bar_height - 0.5, scripting_border_u32, thickness=1.0, rounding=0.0)

            if is_context_selected_primary:
                draw_list.add_rect(seg_start_x - 1, bar_start_y - 1, seg_start_x + seg_width + 1, bar_start_y + bar_height + 1, sel_primary_u32, thickness=2.0, rounding=1.0)

            if is_context_selected_secondary:
                draw_list.add_rect(seg_start_x - 2, bar_start_y - 2, seg_start_x + seg_width + 2, bar_start_y + bar_height + 2, sel_secondary_u32, thickness=1.5, rounding=1.0)

            text_to_draw = segment.position_short_name
            text_width = imgui.calc_text_size(text_to_draw)[0]
            if text_width < seg_width - 8:
                text_pos_x = seg_start_x + (seg_width - text_width) / 2
                text_pos_y = bar_start_y + (bar_height - text_line_height) / 2
                valid_color_for_lum = segment_color_tuple if isinstance(segment_color_tuple, tuple) and len(
                    segment_color_tuple) >= 3 else CurrentTheme.GRAY_MEDIUM[:3]
                lum = 0.2100 * valid_color_for_lum[0] + 0.587 * valid_color_for_lum[1] + 0.114 * valid_color_for_lum[2]
                text_color = text_black_u32 if lum > 0.6 else text_white_u32
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
                chapters_sorted = sorted(fs_proc.video_chapters, key=lambda c: c.start_frame_id)

                if self.resize_edge == 'left':
                    # Resizing left edge (start)
                    new_start = new_frame
                    new_end = resize_chapter.end_frame_id
                    # Constrain to prevent invalid range
                    new_start = min(new_start, new_end - 1)  # At least 1 frame duration
                    # Constrain to prevent overlap with previous chapter
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
                    for i, chapter in enumerate(chapters_sorted):
                        if chapter.unique_id == self.resize_chapter_id and i < len(chapters_sorted) - 1:
                            next_chapter = chapters_sorted[i + 1]
                            new_end = min(new_end, next_chapter.start_frame_id - 1)
                            break

                    # Update chapter
                    resize_chapter.end_frame_id = new_end

                # Show preview tooltip with video frame at new boundary position
                # Update preview frame if it changed
                if self.resize_preview_frame != new_frame:
                    self.resize_preview_frame = new_frame
                    self.resize_preview_data = None  # Clear old data, will be fetched immediately

                    # Async fetch frame for preview (instant loading)
                    import threading
                    def fetch_resize_preview():
                        try:
                            if self.app.processor:
                                frame_data = self.app.processor.get_thumbnail_frame(new_frame, use_gpu_unwarp=False)
                                if frame_data is not None:
                                    self.resize_preview_data = {
                                        'frame': new_frame,
                                        'frame_data': frame_data,
                                        'edge': self.resize_edge
                                    }
                        except Exception as e:
                            self.app.logger.warning(f"Failed to fetch resize preview frame: {e}")

                    threading.Thread(target=fetch_resize_preview, daemon=True).start()

                # Render preview tooltip (always show during resize drag)
                self._render_resize_preview_tooltip(new_frame, self.resize_edge)

        # End resize on mouse release
        elif self.is_resizing_chapter and imgui.is_mouse_released(0):
            self.is_resizing_chapter = False
            self.resize_chapter_id = None
            self.resize_edge = None
            self.resize_preview_frame = None
            self.resize_preview_data = None
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
            # Frame-based normalization matching segment rendering (no padding offset)
            current_norm_pos = self.app.processor.current_frame_index / total_video_frames
            marker_x = bar_start_x + current_norm_pos * bar_width
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

            # Initialize timecode fields
            self._update_timecode_from_frame("start")
            self._update_timecode_from_frame("end")

            self.show_create_chapter_dialog = True
            self.context_selected_chapters.clear()

        # Enhanced UX: Click in gap between chapters to fill it
        # Click behavior:
        # - If no chapters exist: do nothing (require drag to create first chapter)
        # - If click is in a gap between chapters: fill the gap automatically
        # - Otherwise: require drag to create chapter (don't create on simple click)
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

            # Only start drag mode if gap was not filled
            # This allows creating chapters via drag-and-drop in any empty space
            if not gap_detected:
                # Start dragging for manual chapter creation
                self.drag_start_frame = clicked_frame
                self.drag_current_frame = clicked_frame  # Initialize to start position
                self.is_dragging_chapter_range = True

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
                
                if end_frame - start_frame >= 1:  # Minimum 1 frame difference (prevents click-only creation)
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
                        self.app.logger.info(f"Created chapter via drag ({end_frame - start_frame + 1} frames)", extra={'status_message': True})
                # If drag was too small (< 1 frame), silently ignore - no chapter created
                
                self.is_dragging_chapter_range = False

        # Reset drag if mouse leaves bar area
        if self.is_dragging_chapter_range and not is_mouse_over_bar:
            self.is_dragging_chapter_range = False

        # Chapter keyboard shortcuts (when chapters are context-selected)
        if len(self.context_selected_chapters) > 0:
            shortcuts = self.app.app_settings.get("funscript_editor_shortcuts", {})
            io = imgui.get_io()

            # Check Select Points in Chapter (E key) - must come before delete checks
            sel_pts_sc_str = shortcuts.get("select_points_in_chapter", "E")
            sel_pts_key_tuple = self.app._map_shortcut_to_glfw_key(sel_pts_sc_str)
            if sel_pts_key_tuple and (
                imgui.is_key_pressed(sel_pts_key_tuple[0]) and
                sel_pts_key_tuple[1]['ctrl'] == io.key_ctrl and
                sel_pts_key_tuple[1]['alt'] == io.key_alt and
                sel_pts_key_tuple[1]['shift'] == io.key_shift and
                sel_pts_key_tuple[1]['super'] == io.key_super
            ):
                # Dispatch to the active timeline editor
                active_tl = getattr(self.app.app_state_ui, 'active_timeline_num', 1)
                editor = self.gui_instance.timeline_editor1 if active_tl == 1 else self.gui_instance.timeline_editor2
                if editor:
                    editor.select_points_in_chapter()

            # Check Delete Points in Chapter FIRST (SHIFT+DELETE or SHIFT+BACKSPACE) - must come before regular delete
            del_points_sc_str = shortcuts.get("delete_points_in_chapter", "SHIFT+DELETE")
            del_points_alt_sc_str = shortcuts.get("delete_points_in_chapter_alt", "SHIFT+BACKSPACE")
            del_points_key_tuple = self.app._map_shortcut_to_glfw_key(del_points_sc_str)
            del_points_alt_key_tuple = self.app._map_shortcut_to_glfw_key(del_points_alt_sc_str)
            delete_points_pressed = False

            if del_points_key_tuple and (
                imgui.is_key_pressed(del_points_key_tuple[0]) and
                del_points_key_tuple[1]['ctrl'] == io.key_ctrl and
                del_points_key_tuple[1]['alt'] == io.key_alt and
                del_points_key_tuple[1]['shift'] == io.key_shift and
                del_points_key_tuple[1]['super'] == io.key_super
            ):
                delete_points_pressed = True

            if (not delete_points_pressed and del_points_alt_key_tuple and (
                imgui.is_key_pressed(del_points_alt_key_tuple[0]) and
                del_points_alt_key_tuple[1]['ctrl'] == io.key_ctrl and
                del_points_alt_key_tuple[1]['alt'] == io.key_alt and
                del_points_alt_key_tuple[1]['shift'] == io.key_shift and
                del_points_alt_key_tuple[1]['super'] == io.key_super
            )):
                delete_points_pressed = True

            if delete_points_pressed and self.context_selected_chapters:
                # Delete points in the selected chapters
                fs_proc.clear_script_points_in_selected_chapters(self.context_selected_chapters)
                self.app.logger.info(f"Deleted points in {len(self.context_selected_chapters)} chapter(s) via keyboard shortcut", extra={'status_message': True})
            else:
                # Only check regular delete if delete points wasn't pressed
                # Delete Selected Chapter (DELETE or BACKSPACE without modifiers)
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

            # === CHAPTER OPERATIONS ===
            imgui.text_disabled("Chapter Operations")
            imgui.separator()

            # --- Navigation ---
            # Seek to Beginning
            if imgui.menu_item("Seek to Beginning", enabled=can_select_one)[0]:
                if can_select_one:
                    selected_chapter = self.context_selected_chapters[0]
                    if self.app.processor:
                        self.app.processor.seek_video(selected_chapter.start_frame_id)
                        self.app.app_state_ui.force_timeline_pan_to_current_frame = True

            # Seek to End
            if imgui.menu_item("Seek to End", enabled=can_select_one)[0]:
                if can_select_one:
                    selected_chapter = self.context_selected_chapters[0]
                    if self.app.processor:
                        self.app.processor.seek_video(selected_chapter.end_frame_id)
                        self.app.app_state_ui.force_timeline_pan_to_current_frame = True

            imgui.separator()

            # --- Selection & Points ---
            shortcuts = self.app.app_settings.get("funscript_editor_shortcuts", {})
            import platform as _platform

            can_any = num_selected > 0

            # Select Points in Chapter(s)
            select_pts_shortcut = shortcuts.get("select_points_in_chapter", "E")
            select_pts_label = f"Select Points in Chapter{'s' if num_selected > 1 else ''}"
            if imgui.menu_item(select_pts_label, shortcut=select_pts_shortcut, enabled=can_any)[0]:
                if can_any and self.context_selected_chapters:
                    active_tl = getattr(self.app.app_state_ui, 'active_timeline_num', 1)
                    editor = self.gui_instance.timeline_editor1 if active_tl == 1 else self.gui_instance.timeline_editor2
                    if editor:
                        editor.select_points_in_chapter()

            # Delete Points in Chapter(s)
            if _platform.system() == "Darwin":
                delete_points_shortcut = shortcuts.get("delete_points_in_chapter_alt", "Shift+Backspace")
            else:
                delete_points_shortcut = shortcuts.get("delete_points_in_chapter", "Shift+Delete")
            delete_points_label = f"Delete Points in Chapter{'s' if num_selected > 1 else ''}"
            if imgui.menu_item(delete_points_label, shortcut=delete_points_shortcut, enabled=can_any)[0]:
                if can_any and self.context_selected_chapters:
                    fs_proc.clear_script_points_in_selected_chapters(self.context_selected_chapters)

            imgui.separator()

            # --- Editing ---
            can_edit = num_selected == 1
            if imgui.begin_menu("Change Type", enabled=can_select_one):
                if can_select_one and self.context_selected_chapters:
                    self._render_quick_type_change_menu()
                imgui.end_menu()

            if imgui.menu_item("Edit Details...", enabled=can_edit)[0]:
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

                    # Initialize timecode fields
                    self._update_timecode_from_frame("start")
                    self._update_timecode_from_frame("end")

                    self.show_edit_chapter_dialog = True

            # Split Chapter at Cursor
            can_split = False
            split_frame = self.context_menu_opened_at_frame
            split_pos_key = None
            if num_selected == 1 and self.context_selected_chapters:
                chapter = self.context_selected_chapters[0]
                if split_frame is not None and chapter.start_frame_id < split_frame < chapter.end_frame_id:
                    from config.constants import POSITION_INFO_MAPPING
                    for key, info in POSITION_INFO_MAPPING.items():
                        if info.get("short_name") == chapter.position_short_name:
                            split_pos_key = key
                            break
                    else:
                        split_pos_key = chapter.position_short_name
                    can_split = True
            if imgui.menu_item("Split Chapter at Cursor", enabled=can_split)[0]:
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

            imgui.separator()

            # --- Tracking & Merge ---
            if imgui.menu_item("Start Tracker in Chapter", enabled=can_select_one)[0]:
                if can_select_one and len(self.context_selected_chapters) == 1:
                    selected_chapter = self.context_selected_chapters[0]
                    if hasattr(fs_proc, 'set_scripting_range_from_chapter'):
                        fs_proc.set_scripting_range_from_chapter(selected_chapter)
                        self._start_live_tracking(
                            success_info=f"Tracker started for chapter: {selected_chapter.position_short_name}"
                        )

            can_standard_merge = num_selected == 2
            if imgui.menu_item("Merge Chapters (2)", enabled=can_standard_merge)[0]:
                if can_standard_merge and len(self.context_selected_chapters) == 2:
                    chaps_to_merge = sorted(self.context_selected_chapters, key=lambda c: c.start_frame_id)
                    if hasattr(fs_proc, 'merge_selected_chapters'):
                        fs_proc.merge_selected_chapters(chaps_to_merge[0], chaps_to_merge[1])
                        self.context_selected_chapters.clear()

            imgui.separator()

            # --- Delete Chapter (destructive, last) ---
            can_delete = num_selected > 0
            if _platform.system() == "Darwin":
                delete_shortcut = shortcuts.get("delete_selected_chapter_alt", "Backspace")
            else:
                delete_shortcut = shortcuts.get("delete_selected_chapter", "Delete")
            delete_label = f"Delete Chapter{'s' if num_selected > 1 else ''} ({num_selected})" if num_selected > 1 else "Delete Chapter"
            if imgui.menu_item(delete_label, shortcut=delete_shortcut, enabled=can_delete)[0]:
                if can_delete and self.context_selected_chapters:
                    ch_ids = [ch.unique_id for ch in self.context_selected_chapters]
                    fs_proc.delete_video_chapters_by_ids(ch_ids)
                    self.context_selected_chapters.clear()

            imgui.separator()

            # === ADVANCED OPERATIONS (Submenu) ===
            if imgui.begin_menu("Advanced"):
                # Set ROI
                if imgui.menu_item("Set ROI & Point", enabled=can_edit)[0]:
                    if can_edit:
                        selected_chapter = self.context_selected_chapters[0]
                        self.app.chapter_id_for_roi_setting = selected_chapter.unique_id
                        self.app.enter_set_user_roi_mode()
                        if self.app.processor:
                            self.app.processor.seek_video(selected_chapter.start_frame_id)
                            self.app.app_state_ui.force_timeline_pan_to_current_frame = True

                # Apply Plugin
                can_apply_plugin = num_selected > 0
                plugin_label = f"Apply Plugin ({num_selected})" if num_selected > 1 else "Apply Plugin"
                if imgui.begin_menu(plugin_label, enabled=can_apply_plugin):
                    if can_apply_plugin and self.context_selected_chapters:
                        if imgui.begin_menu("Timeline 1 (Primary)"):
                            self._render_chapter_plugin_menu('primary')
                            imgui.end_menu()
                        timeline2_visible = self.app.app_state_ui.show_funscript_interactive_timeline2
                        if imgui.begin_menu("Timeline 2 (Secondary)", enabled=timeline2_visible):
                            self._render_chapter_plugin_menu('secondary')
                            imgui.end_menu()
                    imgui.end_menu()

                # Chapter Analysis
                can_analyze = num_selected == 1
                if imgui.begin_menu("Chapter Analysis", enabled=can_analyze):
                    if can_analyze and self.context_selected_chapters:
                        self._render_dynamic_chapter_analysis_menu()
                    imgui.end_menu()

                imgui.separator()

                # Gap Operations
                can_fill_gap_merge = False
                gap_fill_c1, gap_fill_c2 = None, None
                if len(self.context_selected_chapters) == 2:
                    temp_chaps_fill_gap = sorted(self.context_selected_chapters, key=lambda c: c.start_frame_id)
                    c1_fg_check, c2_fg_check = temp_chaps_fill_gap[0], temp_chaps_fill_gap[1]
                    if c1_fg_check.end_frame_id < c2_fg_check.start_frame_id - 1:
                        can_fill_gap_merge = True
                        gap_fill_c1, gap_fill_c2 = c1_fg_check, c2_fg_check

                if imgui.menu_item("Track Gap & Merge", enabled=can_fill_gap_merge)[0]:
                    if gap_fill_c1 and gap_fill_c2:
                        self.app.logger.info(
                            f"UI Action: Initiating track gap then merge between {gap_fill_c1.unique_id} and {gap_fill_c2.unique_id}")

                        gap_start_frame = gap_fill_c1.end_frame_id + 1
                        gap_end_frame = gap_fill_c2.start_frame_id - 1

                        if gap_end_frame < gap_start_frame:
                            self.app.logger.warning("No actual gap to track. Merging directly (if possible).")
                            if hasattr(fs_proc, 'merge_selected_chapters'):
                                merged_chapter = fs_proc.merge_selected_chapters(gap_fill_c1, gap_fill_c2, return_chapter_object=True)
                                if merged_chapter:
                                    self.context_selected_chapters = [merged_chapter]
                                else:
                                    self.context_selected_chapters.clear()
                            imgui.close_current_popup()
                            return

                        fs_proc._record_timeline_action(1, f"Prepare for Gap Track & Merge: {gap_fill_c1.unique_id[:4]}+{gap_fill_c2.unique_id[:4]}")
                        self.app.set_pending_action_after_tracking(
                            action_type='finalize_gap_merge_after_tracking',
                            chapter1_id=gap_fill_c1.unique_id,
                            chapter2_id=gap_fill_c2.unique_id
                        )

                        fs_proc.scripting_start_frame = gap_start_frame
                        fs_proc.scripting_end_frame = gap_end_frame
                        fs_proc.scripting_range_active = True
                        fs_proc.selected_chapter_for_scripting = None
                        self.app.project_manager.project_dirty = True

                        self._start_live_tracking(
                            success_info=(
                                f"Tracker started for gap between {gap_fill_c1.position_short_name} and {gap_fill_c2.position_short_name} "
                                f"(Frames: {gap_start_frame}-{gap_end_frame})"
                            ),
                            on_error_clear_pending_action=True
                        )

                        self.context_selected_chapters.clear()
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
                        from config.constants import ChapterSource
                        self.app.logger.info(
                            f"UI Action: Creating new chapter in gap between {bridge_ch1.unique_id} and {bridge_ch2.unique_id}")
                        gap_chapter_data = {
                            "start_frame_str": str(actual_gap_start),
                            "end_frame_str": str(actual_gap_end),
                            "segment_type": bridge_ch1.segment_type,
                            "position_short_name_key": bridge_ch1.position_short_name,
                            "source": ChapterSource.MANUAL_GAP_FILL.value
                        }
                        new_gap_chapter = fs_proc.create_new_chapter_from_data(gap_chapter_data, return_chapter_object=True)
                        if new_gap_chapter:
                            self.context_selected_chapters = [new_gap_chapter]
                            if hasattr(fs_proc, 'set_scripting_range_from_chapter'):
                                fs_proc.set_scripting_range_from_chapter(new_gap_chapter)
                                self._start_live_tracking(
                                    success_info=f"Tracker started for new gap chapter: {new_gap_chapter.unique_id}"
                                )
                        else:
                            self.app.logger.error(
                                "Failed to create new chapter in gap.")
                            self.context_selected_chapters.clear()

                imgui.end_menu()  # End Advanced

            imgui.end_popup()


    def _render_quick_type_change_menu(self):
        """Render quick chapter type change menu without opening full edit dialog."""
        if not self.context_selected_chapters:
            return

        selected_chapter = self.context_selected_chapters[0]
        fs_proc = self.app.funscript_processor

        # Get chapter type manager for custom types
        from application.classes.chapter_type_manager import get_chapter_type_manager
        type_mgr = get_chapter_type_manager()

        # Get all available types (built-in + custom) organized by category
        from config.constants import POSITION_INFO_MAPPING

        # Organize by simplified categories: Position and Not Relevant
        position_types = []
        not_relevant_types = []

        # Built-in types from POSITION_INFO_MAPPING
        for key, info in POSITION_INFO_MAPPING.items():
            short_name = info.get("short_name", key)
            long_name = info.get("long_name", short_name)
            category = info.get("category", "Position")

            if category == "Position":
                position_types.append((short_name, long_name))
            else:  # Not Relevant category
                not_relevant_types.append((short_name, long_name))

        # Add custom types if available (organized by their category)
        if type_mgr:
            all_custom_types = type_mgr.custom_types  # Only custom, not built-in
            for short_name, info in all_custom_types.items():
                long_name = info.get("long_name", short_name)
                category = info.get("category", "Position")

                if category == "Position":
                    position_types.append((short_name, long_name))
                else:  # Not Relevant
                    not_relevant_types.append((short_name, long_name))

        # Render organized menu
        current_type = selected_chapter.position_short_name

        # Position category (scripted content)
        if position_types:
            if imgui.begin_menu("Position (Scripted)"):
                for short_name, long_name in sorted(position_types, key=lambda x: x[1]):
                    is_current = short_name == current_type
                    if imgui.menu_item(long_name, selected=is_current)[0] and not is_current:
                        self._change_chapter_type(selected_chapter, short_name)
                imgui.end_menu()

        # Not Relevant category (non-scripted content)
        if not_relevant_types:
            if imgui.begin_menu("Not Relevant (Non-scripted)"):
                for short_name, long_name in sorted(not_relevant_types, key=lambda x: x[1]):
                    is_current = short_name == current_type
                    if imgui.menu_item(long_name, selected=is_current)[0] and not is_current:
                        self._change_chapter_type(selected_chapter, short_name)
                imgui.end_menu()


    def _change_chapter_type(self, chapter, new_type_short_name):
        """Change a chapter's type without opening edit dialog."""
        fs_proc = self.app.funscript_processor
        if not fs_proc:
            return

        fs_proc.update_chapter_from_data(
            chapter.unique_id,
            {
                "start_frame_str": str(chapter.start_frame_id),
                "end_frame_str": str(chapter.end_frame_id),
                "position_short_name_key": new_type_short_name
            }
        )

        # Track usage in chapter type manager
        from application.classes.chapter_type_manager import get_chapter_type_manager
        type_mgr = get_chapter_type_manager()
        if type_mgr:
            type_mgr.increment_usage(new_type_short_name)

        self.app.logger.info(f"Changed chapter type to {new_type_short_name}", extra={'status_message': True})
        self.app.project_manager.project_dirty = True

