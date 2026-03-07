import imgui
from typing import Optional

from application.utils import _format_time, VideoSegment, get_icon_texture_manager, primary_button_style, destructive_button_style
from application.utils.imgui_helpers import center_next_window
from config.constants import POSITION_INFO_MAPPING, DEFAULT_CHAPTER_FPS
from config.element_group_colors import VideoNavigationColors
from config.constants_colors import CurrentTheme



class ChapterEditorMixin:
    """Mixin fragment for VideoNavigationUI."""

    def _render_create_chapter_window(self):
        if not self.show_create_chapter_dialog:
            return
        window_flags = imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_COLLAPSE
        center_next_window(450)

        is_not_collapsed, self.show_create_chapter_dialog = imgui.begin(
            "Create New Chapter##CreateWindow",
            closable=True,
            flags=window_flags
        )

        if is_not_collapsed and self.show_create_chapter_dialog:
            imgui.text("Create New Chapter Details")
            imgui.separator()

            imgui.push_item_width(200)

            # Frame input fields
            frame_changed_start, new_start_frame = imgui.input_text("Start Frame##CreateWinFrame", self.chapter_edit_data.get("start_frame_str", "0"), 64)
            if frame_changed_start:
                self.chapter_edit_data["start_frame_str"] = new_start_frame
                # Update timecode when frame changes
                self._update_timecode_from_frame("start")

            # Timecode display (synced)
            imgui.same_line()
            timecode_changed_start, new_start_timecode = imgui.input_text("Start Time##CreateWinTime", self.chapter_start_timecode, 64)
            if timecode_changed_start:
                self.chapter_start_timecode = new_start_timecode
                # Update frame when timecode changes
                self._update_frame_from_timecode("start", new_start_timecode)

            # End frame and timecode
            frame_changed_end, new_end_frame = imgui.input_text("End Frame##CreateWinFrame", self.chapter_edit_data.get("end_frame_str", "0"), 64)
            if frame_changed_end:
                self.chapter_edit_data["end_frame_str"] = new_end_frame
                # Update timecode when frame changes
                self._update_timecode_from_frame("end")

            imgui.same_line()
            timecode_changed_end, new_end_timecode = imgui.input_text("End Time##CreateWinTime", self.chapter_end_timecode, 64)
            if timecode_changed_end:
                self.chapter_end_timecode = new_end_timecode
                # Update frame when timecode changes
                self._update_frame_from_timecode("end", new_end_timecode)

            imgui.text_disabled("Timecode format: HH:MM:SS or MM:SS or SS")

            # Segment Type dropdown (instead of free text)
            from config.constants import ChapterSegmentType
            segment_type_values = ChapterSegmentType.get_all_values()
            current_segment_type = self.chapter_edit_data.get("segment_type", ChapterSegmentType.get_default().value)
            try:
                self.selected_segment_type_idx = segment_type_values.index(current_segment_type)
            except ValueError:
                self.selected_segment_type_idx = 0

            clicked_segment_type, self.selected_segment_type_idx = imgui.combo(
                "Category##CreateWin",
                self.selected_segment_type_idx,
                segment_type_values
            )
            if clicked_segment_type:
                self.chapter_edit_data["segment_type"] = segment_type_values[self.selected_segment_type_idx]

            # Position dropdown
            clicked_pos, self.selected_position_idx_in_dialog = imgui.combo("Position##CreateWin", self.selected_position_idx_in_dialog, self.position_display_names)
            if clicked_pos and self.position_short_name_keys and 0 <= self.selected_position_idx_in_dialog < len(
                    self.position_short_name_keys):
                self.chapter_edit_data["position_short_name_key"] = self.position_short_name_keys[
                    self.selected_position_idx_in_dialog]
            current_selected_key = self.chapter_edit_data.get("position_short_name_key")
            long_name_display = POSITION_INFO_MAPPING.get(current_selected_key, {}).get("long_name", "N/A") if current_selected_key else "N/A"
            imgui.text_disabled(f"Long Name (auto): {long_name_display}")

            # Source is auto-set based on creation method, so we show it as read-only info
            current_source = self.chapter_edit_data.get("source", "manual")
            imgui.text_disabled(f"Source: {current_source}")

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
        center_next_window(450)

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

            # Frame input fields
            frame_changed_start, new_start_frame = imgui.input_text("Start Frame##EditWinFrame", self.chapter_edit_data.get("start_frame_str", "0"), 64)
            if frame_changed_start:
                self.chapter_edit_data["start_frame_str"] = new_start_frame
                # Update timecode when frame changes
                self._update_timecode_from_frame("start")

            # Timecode display (synced)
            imgui.same_line()
            timecode_changed_start, new_start_timecode = imgui.input_text("Start Time##EditWinTime", self.chapter_start_timecode, 64)
            if timecode_changed_start:
                self.chapter_start_timecode = new_start_timecode
                # Update frame when timecode changes
                self._update_frame_from_timecode("start", new_start_timecode)

            # End frame and timecode
            frame_changed_end, new_end_frame = imgui.input_text("End Frame##EditWinFrame", self.chapter_edit_data.get("end_frame_str", "0"), 64)
            if frame_changed_end:
                self.chapter_edit_data["end_frame_str"] = new_end_frame
                # Update timecode when frame changes
                self._update_timecode_from_frame("end")

            imgui.same_line()
            timecode_changed_end, new_end_timecode = imgui.input_text("End Time##EditWinTime", self.chapter_end_timecode, 64)
            if timecode_changed_end:
                self.chapter_end_timecode = new_end_timecode
                # Update frame when timecode changes
                self._update_frame_from_timecode("end", new_end_timecode)

            imgui.text_disabled("Timecode format: HH:MM:SS or MM:SS or SS")

            # Category dropdown (Position or Not Relevant only)
            # Determine category from position_short_name in POSITION_INFO_MAPPING
            from config.constants import ChapterSegmentType, POSITION_INFO_MAPPING
            category_options = ChapterSegmentType.get_user_category_options()

            # Get category from position_short_name (reliable)
            current_pos_short_name = self.chapter_edit_data.get("position_short_name_key", "")
            position_info = POSITION_INFO_MAPPING.get(current_pos_short_name, {})
            current_category = position_info.get('category', 'Position')  # Default to Position

            try:
                self.selected_segment_type_idx = category_options.index(current_category)
            except ValueError:
                self.selected_segment_type_idx = 0

            clicked_segment_type, self.selected_segment_type_idx = imgui.combo(
                "Category##EditWin",
                self.selected_segment_type_idx,
                category_options
            )
            if clicked_segment_type:
                self.chapter_edit_data["segment_type"] = category_options[self.selected_segment_type_idx]

            # Position dropdown
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

            # Source dropdown (instead of free text)
            from config.constants import ChapterSource
            source_values = ChapterSource.get_all_values()
            current_source = self.chapter_edit_data.get("source", ChapterSource.get_default().value)
            try:
                self.selected_source_idx = source_values.index(current_source)
            except ValueError:
                self.selected_source_idx = 0

            clicked_source, self.selected_source_idx = imgui.combo(
                "Source##EditWin",
                self.selected_source_idx,
                source_values
            )
            if clicked_source:
                self.chapter_edit_data["source"] = source_values[self.selected_source_idx]

            # Show source type indicator with icon
            icon_mgr = self.app.icon_manager if hasattr(self.app, 'icon_manager') else None
            if icon_mgr:
                if ChapterSource.is_ai_generated(current_source):
                    robot_tex = icon_mgr.get_texture('robot.png')
                    if robot_tex:
                        imgui.image(robot_tex, 16, 16)
                        imgui.same_line()
                    imgui.text_disabled("AI Generated")
                elif ChapterSource.is_user_created(current_source):
                    user_tex = icon_mgr.get_texture('user.png')
                    if user_tex:
                        imgui.image(user_tex, 16, 16)
                        imgui.same_line()
                    imgui.text_disabled("User Created")
                else:
                    download_tex = icon_mgr.get_texture('download.png')
                    if download_tex:
                        imgui.image(download_tex, 16, 16)
                        imgui.same_line()
                    imgui.text_disabled("Imported/Other")
            else:
                # Fallback to text only
                if ChapterSource.is_ai_generated(current_source):
                    imgui.text_disabled("AI Generated")
                elif ChapterSource.is_user_created(current_source):
                    imgui.text_disabled("User Created")
                else:
                    imgui.text_disabled("Imported/Other")

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
        self._update_timecode_from_frame("start")
        self._update_timecode_from_frame("end")
    

    def _get_current_frame(self) -> int:
        """Get the current video frame position."""
        if self.app.processor and hasattr(self.app.processor, 'current_frame_index'):
            return max(0, self.app.processor.current_frame_index)
        return 0
    

    def _set_chapter_start_to_current_frame(self):
        """Set the chapter start frame to the current video frame."""
        current_frame = self._get_current_frame()
        self.chapter_edit_data["start_frame_str"] = str(current_frame)
        self._update_timecode_from_frame("start")
        self.app.logger.info(f"Chapter start set to frame {current_frame}", extra={'status_message': True})
    

    def _set_chapter_end_to_current_frame(self):
        """Set the chapter end frame to the current video frame."""
        current_frame = self._get_current_frame()
        self.chapter_edit_data["end_frame_str"] = str(current_frame)
        self._update_timecode_from_frame("end")
        self.app.logger.info(f"Chapter end set to frame {current_frame}", extra={'status_message': True})


    def _update_timecode_from_frame(self, field: str):
        """Update timecode display when frame number changes."""
        from application.utils.video_segment import VideoSegment

        fps = self.app.processor.fps if self.app.processor and self.app.processor.fps > 0 else 30.0

        try:
            if field == "start":
                frame_str = self.chapter_edit_data.get("start_frame_str", "0")
                frame = int(frame_str) if frame_str.isdigit() else 0
                self.chapter_start_timecode = VideoSegment._frames_to_timecode(frame, fps)
            elif field == "end":
                frame_str = self.chapter_edit_data.get("end_frame_str", "0")
                frame = int(frame_str) if frame_str.isdigit() else 0
                self.chapter_end_timecode = VideoSegment._frames_to_timecode(frame, fps)
        except (ValueError, AttributeError):
            pass  # Ignore invalid input during typing


    def _update_frame_from_timecode(self, field: str, timecode: str):
        """Update frame number when timecode changes."""
        from application.utils.video_segment import VideoSegment

        fps = self.app.processor.fps if self.app.processor and self.app.processor.fps > 0 else 30.0

        try:
            frame = VideoSegment.parse_time_input_to_frames(timecode, fps)
            if frame >= 0:
                if field == "start":
                    self.chapter_edit_data["start_frame_str"] = str(frame)
                elif field == "end":
                    self.chapter_edit_data["end_frame_str"] = str(frame)
        except (ValueError, AttributeError):
            pass  # Ignore invalid input during typing

