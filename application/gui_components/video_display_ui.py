import imgui
from typing import Optional, Tuple


import config.constants as constants


class VideoDisplayUI:
    def __init__(self, app, gui_instance):
        self.app = app
        self.gui_instance = gui_instance
        self._video_display_rect_min = (0, 0)
        self._video_display_rect_max = (0, 0)
        self._actual_video_image_rect_on_screen = {'min_x': 0, 'min_y': 0, 'max_x': 0, 'max_y': 0, 'w': 0, 'h': 0}

        # ROI Drawing state for User Defined ROI
        self.is_drawing_user_roi: bool = False
        self.user_roi_draw_start_screen_pos: tuple = (0, 0)  # In ImGui screen space
        self.user_roi_draw_current_screen_pos: tuple = (0, 0)  # In ImGui screen space
        self.drawn_user_roi_video_coords: tuple | None = None  # (x,y,w,h) in original video frame pixel space (e.g. 640x640)
        self.waiting_for_point_click: bool = False

    def _update_actual_video_image_rect(self, display_w, display_h, cursor_x_offset, cursor_y_offset):
        win_pos_x, win_pos_y = imgui.get_window_position()
        content_region_min_x, content_region_min_y = imgui.get_window_content_region_min()
        self._actual_video_image_rect_on_screen['min_x'] = win_pos_x + content_region_min_x + cursor_x_offset
        self._actual_video_image_rect_on_screen['min_y'] = win_pos_y + content_region_min_y + cursor_y_offset
        self._actual_video_image_rect_on_screen['w'] = display_w
        self._actual_video_image_rect_on_screen['h'] = display_h
        self._actual_video_image_rect_on_screen['max_x'] = self._actual_video_image_rect_on_screen['min_x'] + display_w
        self._actual_video_image_rect_on_screen['max_y'] = self._actual_video_image_rect_on_screen['min_y'] + display_h

    def _screen_to_video_coords(self, screen_x: float, screen_y: float) -> tuple | None:
        """Converts absolute screen coordinates to video buffer coordinates, accounting for pan and zoom."""
        app_state = self.app.app_state_ui

        img_rect = self._actual_video_image_rect_on_screen
        if img_rect['w'] <= 0 or img_rect['h'] <= 0:
            return None

        # Mouse position relative to the displayed video image's top-left corner
        mouse_rel_img_x = screen_x - img_rect['min_x']
        mouse_rel_img_y = screen_y - img_rect['min_y']

        # Normalized position on the *visible part* of the texture
        if img_rect['w'] == 0 or img_rect['h'] == 0: return None  # Avoid division by zero
        norm_visible_x = mouse_rel_img_x / img_rect['w']
        norm_visible_y = mouse_rel_img_y / img_rect['h']

        if not (0 <= norm_visible_x <= 1 and 0 <= norm_visible_y <= 1):  # Click outside displayed image
            return None

        # Account for pan and zoom to find normalized position on the *full* texture
        uv_pan_x, uv_pan_y = app_state.video_pan_normalized
        uv_disp_w_tex = 1.0 / app_state.video_zoom_factor
        uv_disp_h_tex = 1.0 / app_state.video_zoom_factor

        tex_norm_x = uv_pan_x + norm_visible_x * uv_disp_w_tex
        tex_norm_y = uv_pan_y + norm_visible_y * uv_disp_h_tex

        if not (0 <= tex_norm_x <= 1 and 0 <= tex_norm_y <= 1):  # Point is outside the full texture due to pan/zoom
            return None

        video_buffer_w, video_buffer_h = self.app.yolo_input_size, self.app.yolo_input_size  # Assume tracker works on this size
        if self.app.processor and self.app.processor.current_frame is not None:
            h, w = self.app.processor.current_frame.shape[:2]
            video_buffer_w, video_buffer_h = w, h

        video_x = int(tex_norm_x * video_buffer_w)
        video_y = int(tex_norm_y * video_buffer_h)

        return video_x, video_y

    def _video_to_screen_coords(self, video_x: int, video_y: int) -> tuple | None:
        """Converts video buffer coordinates to absolute screen coordinates, accounting for pan and zoom."""
        app_state = self.app.app_state_ui
        img_rect = self._actual_video_image_rect_on_screen

        video_buffer_w, video_buffer_h = self.app.yolo_input_size, self.app.yolo_input_size
        if self.app.processor and self.app.processor.current_frame is not None:
            h, w = self.app.processor.current_frame.shape[:2]
            video_buffer_w, video_buffer_h = w, h

        if video_buffer_w <= 0 or video_buffer_h <= 0 or img_rect['w'] <= 0 or img_rect['h'] <= 0:
            return None

        # Normalized position on the *full* texture
        tex_norm_x = video_x / video_buffer_w
        tex_norm_y = video_y / video_buffer_h

        # Account for pan and zoom to find normalized position on the *visible part* of the texture
        uv_pan_x, uv_pan_y = app_state.video_pan_normalized
        uv_disp_w_tex = 1.0 / app_state.video_zoom_factor
        uv_disp_h_tex = 1.0 / app_state.video_zoom_factor

        if uv_disp_w_tex == 0 or uv_disp_h_tex == 0: return None  # Avoid division by zero

        norm_visible_x = (tex_norm_x - uv_pan_x) / uv_disp_w_tex
        norm_visible_y = (tex_norm_y - uv_pan_y) / uv_disp_h_tex

        # If the video point is outside the current view due to pan/zoom, don't draw it
        if not (0 <= norm_visible_x <= 1 and 0 <= norm_visible_y <= 1):
            return None

        # Position relative to the displayed video image's top-left corner
        mouse_rel_img_x = norm_visible_x * img_rect['w']
        mouse_rel_img_y = norm_visible_y * img_rect['h']

        # Absolute screen coordinates
        screen_x = img_rect['min_x'] + mouse_rel_img_x
        screen_y = img_rect['min_y'] + mouse_rel_img_y

        return screen_x, screen_y

    def _render_playback_controls_overlay(self):
        """Renders playback controls as an overlay on the video."""
        style = imgui.get_style()
        event_handlers = self.app.event_handlers
        stage_proc = self.app.stage_processor
        file_mgr = self.app.file_manager

        controls_disabled = stage_proc.full_analysis_active or not file_mgr.video_path

        ICON_JUMP_START, ICON_PREV_FRAME, ICON_PLAY, ICON_PAUSE, ICON_STOP, ICON_NEXT_FRAME, ICON_JUMP_END = "|<", "<<", ">", "||", "[]", ">>", ">|"
        button_h_ref = imgui.get_frame_height()
        pb_icon_w, pb_play_w, pb_stop_w, pb_btn_spacing = button_h_ref * 1.5, button_h_ref * 1.7, button_h_ref * 1.5, 4.0

        total_controls_width = (pb_icon_w * 4) + pb_play_w + pb_stop_w + (pb_btn_spacing * 5)

        img_rect = self._actual_video_image_rect_on_screen
        if img_rect['w'] <= 0 or img_rect['h'] <= 0:
            return

        overlay_x = img_rect['min_x'] + (img_rect['w'] - total_controls_width) / 2
        overlay_y = img_rect['max_y'] - button_h_ref - style.item_spacing[1] * 2
        overlay_y = max(img_rect['min_y'] + style.item_spacing[1], overlay_y)
        overlay_x = max(img_rect['min_x'] + style.item_spacing[1], overlay_x)
        imgui.set_cursor_screen_pos((overlay_x, overlay_y))

        if controls_disabled:
            imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
            imgui.push_style_var(imgui.STYLE_ALPHA, style.alpha * 0.5)

        imgui.begin_group()
        if imgui.button(ICON_JUMP_START + "##VidOverStart", width=pb_icon_w): event_handlers.handle_playback_control(
            "jump_start")
        imgui.same_line(spacing=pb_btn_spacing)
        if imgui.button(ICON_PREV_FRAME + "##VidOverPrev", width=pb_icon_w): event_handlers.handle_playback_control(
            "prev_frame")
        imgui.same_line(spacing=pb_btn_spacing)
        play_pause_icon = ICON_PAUSE if self.app.processor and self.app.processor.is_processing else ICON_PLAY
        if imgui.button(play_pause_icon + "##VidOverPlayPause",
                        width=pb_play_w): event_handlers.handle_playback_control("play_pause")
        imgui.same_line(spacing=2)
        if imgui.button(ICON_STOP + "##VidOverStop", width=pb_stop_w): event_handlers.handle_playback_control("stop")
        imgui.same_line(spacing=pb_btn_spacing)
        if imgui.button(ICON_NEXT_FRAME + "##VidOverNext", width=pb_icon_w): event_handlers.handle_playback_control(
            "next_frame")
        imgui.same_line(spacing=pb_btn_spacing)
        if imgui.button(ICON_JUMP_END + "##VidOverEnd", width=pb_icon_w): event_handlers.handle_playback_control(
            "jump_end")
        imgui.end_group()

        if controls_disabled:
            imgui.pop_style_var()
            imgui.internal.pop_item_flag()

    def _render_fps_slider_controls(self, app_state, button_h_ref, slider_w, reset_button_w, native_fps_val,
                                    reset_button_text):
        # This method is moved from VideoNavigationUI
        imgui.begin_group()
        imgui.set_next_item_width(slider_w)

        current_target_fps_setting = self.app.processor.target_fps if self.app.processor else 30.0
        fps_value_for_slider_display = current_target_fps_setting
        if self.app.processor and self.app.processor.is_processing and self.app.processor.actual_fps > 0.1:
            fps_value_for_slider_display = self.app.processor.actual_fps
        elif self.app.tracker and self.app.tracker.current_fps > 0.1 and not (
                self.app.processor and self.app.processor.is_processing):
            fps_value_for_slider_display = self.app.tracker.current_fps

        changed_fps, new_target_value_from_drag = imgui.slider_float(
            "##TargetFPSTinyOverlay", fps_value_for_slider_display,
            app_state.fps_slider_min_val, app_state.fps_slider_max_val, "%.0f FPS"
        )
        if changed_fps:
            if self.app.processor:
                self.app.processor.set_target_fps(new_target_value_from_drag)

        rect_min, rect_max = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        tracker_fps_mark = self.app.tracker.current_fps if self.app.tracker else 0.0
        processor_fps_mark = self.app.processor.actual_fps if self.app.processor else 0.0
        self.gui_instance._draw_fps_marks_on_slider(
            imgui.get_window_draw_list(), rect_min, rect_max,
            current_target_fps_setting, tracker_fps_mark, processor_fps_mark
        )

        imgui.same_line()
        if native_fps_val > 0:
            if imgui.button(reset_button_text + "##FPSResetTinyOverlay", width=reset_button_w):
                if self.app.processor:
                    self.app.processor.set_target_fps(native_fps_val)
        else:
            imgui.dummy(reset_button_w, button_h_ref)
        imgui.end_group()

    def _render_fps_controls_overlay(self):
        """Manages positioning and parameters for FPS controls overlay."""
        style = imgui.get_style()
        event_handlers = self.app.event_handlers
        app_state = self.app.app_state_ui
        stage_proc = self.app.stage_processor
        file_mgr = self.app.file_manager

        controls_disabled = stage_proc.full_analysis_active or not file_mgr.video_path

        button_h_ref = imgui.get_frame_height()
        tiny_fps_slider_w = 75.0
        native_fps_str, native_fps_val_for_reset = event_handlers.get_native_fps_info_for_button()
        tiny_fps_reset_button_text = f"R{native_fps_str}"
        tiny_fps_reset_button_w = max(button_h_ref * 1.5,
                                      imgui.calc_text_size(tiny_fps_reset_button_text)[0] + style.frame_padding[
                                          0] * 2 + 5)

        total_fps_controls_width = tiny_fps_slider_w + style.item_spacing[0] + tiny_fps_reset_button_w

        img_rect = self._actual_video_image_rect_on_screen
        if img_rect['w'] <= 0 or img_rect['h'] <= 0: return

        overlay_x = img_rect['max_x'] - total_fps_controls_width - style.item_spacing[1]
        overlay_y = img_rect['max_y'] - button_h_ref - style.item_spacing[1] * 2
        overlay_y = max(img_rect['min_y'] + style.item_spacing[1], overlay_y)
        overlay_x = max(img_rect['min_x'] + style.item_spacing[1], overlay_x)
        imgui.set_cursor_screen_pos((overlay_x, overlay_y))

        if controls_disabled:
            imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
            imgui.push_style_var(imgui.STYLE_ALPHA, style.alpha * 0.5)
        self._render_fps_slider_controls(app_state, button_h_ref, tiny_fps_slider_w, tiny_fps_reset_button_w,
                                         native_fps_val_for_reset, tiny_fps_reset_button_text)
        if controls_disabled:
            imgui.pop_style_var()
            imgui.internal.pop_item_flag()

    def _render_pose_skeleton(self, draw_list, pose_data: dict, is_dominant: bool):
        """Draws the skeleton, highlighting the dominant pose."""
        keypoints = pose_data.get("keypoints", [])
        if not isinstance(keypoints, list) or len(keypoints) < 17: return

        # --- Color based on whether this is the dominant pose ---
        if is_dominant:
            limb_color = imgui.get_color_u32_rgba(0.1, 1.0, 0.1, 0.95)  # Bright Green
            kpt_color = imgui.get_color_u32_rgba(1.0, 0.5, 0.1, 1.0)  # Bright Orange
            thickness = 2
        else:
            limb_color = imgui.get_color_u32_rgba(0.2, 0.6, 1, 0.4)  # Muted Cyan
            kpt_color = imgui.get_color_u32_rgba(0.9, 0.2, 0.2, 0.5)  # Muted Red
            thickness = 1

        skeleton = [[0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 11], [6, 12], [11, 12], [5, 7], [7, 9], [6, 8], [8, 10],
                    [11, 13], [13, 15], [12, 14], [14, 16]]

        for conn in skeleton:
            idx1, idx2 = conn
            if not (idx1 < len(keypoints) and idx2 < len(keypoints)): continue
            kp1, kp2 = keypoints[idx1], keypoints[idx2]
            if kp1[2] > 0.5 and kp2[2] > 0.5:
                p1_screen = self._video_to_screen_coords(int(kp1[0]), int(kp1[1]))
                p2_screen = self._video_to_screen_coords(int(kp2[0]), int(kp2[1]))
                if p1_screen and p2_screen:
                    draw_list.add_line(p1_screen[0], p1_screen[1], p2_screen[0], p2_screen[1], limb_color,
                                       thickness=thickness)

        for kp in keypoints:
            if kp[2] > 0.5:
                p_screen = self._video_to_screen_coords(int(kp[0]), int(kp[1]))
                if p_screen:
                    draw_list.add_circle_filled(p_screen[0], p_screen[1], 3.0, kpt_color)

    def _render_motion_mode_overlay(self, draw_list, motion_mode: Optional[str], interaction_class: Optional[str],
                                    roi_video_coords: Tuple[int, int, int, int]):
        """Renders the motion mode text (Thrusting, Riding, etc.) as an ImGui overlay."""
        if not motion_mode or motion_mode == 'undetermined':
            return

        mode_color = imgui.get_color_u32_rgba(1, 1, 0, 0.9)
        mode_text = "Undetermined"

        if motion_mode == 'thrusting':
            mode_text = "Thrusting"
            mode_color = imgui.get_color_u32_rgba(0.1, 1.0, 0.1, 0.95)
        elif motion_mode == 'riding':
            mode_color = imgui.get_color_u32_rgba(1.0, 0.4, 1.0, 0.95)
            if interaction_class == 'face':
                mode_text = "Blowing"
            elif interaction_class == 'hand':
                mode_text = "Stroking"
            else:
                mode_text = "Riding"

        if mode_text == "Undetermined":
            return

        # Position text under the anchor box instead of the corner of the ROI
        box_x, box_y, box_w, box_h = roi_video_coords
        # Anchor point in video coordinates: bottom-center of the box
        anchor_vid_x = box_x + box_w / 2
        anchor_vid_y = box_y + box_h

        anchor_screen_pos = self._video_to_screen_coords(int(anchor_vid_x), int(anchor_vid_y))

        if anchor_screen_pos:
            text_size = imgui.calc_text_size(mode_text)
            # Center the text horizontally under the anchor point
            text_pos_x = anchor_screen_pos[0] - (text_size[0] / 2)
            # Position it vertically below the anchor point, with padding
            text_pos_y = anchor_screen_pos[1] + 5  # 5 pixels of padding below the box

            img_rect = self._actual_video_image_rect_on_screen
            if (text_pos_x + text_size[0]) < img_rect['max_x'] and text_pos_y < img_rect['max_y']:
                draw_list.add_text(text_pos_x, text_pos_y, mode_color, mode_text)

    def render(self):
        app_state = self.app.app_state_ui
        is_floating = app_state.ui_layout_mode == 'floating'

        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (0, 0))

        should_render_content = False
        if is_floating:
            # For floating mode, this is a standard, toggleable window.
            # If it's not set to be visible, don't render anything.
            if not app_state.show_video_display_window:
                imgui.pop_style_var()
                return

            # Begin the window. The second return value `new_visibility` will be False if the user clicks the 'x'.
            is_expanded, new_visibility = imgui.begin("Video Display", closable=True,
                                                      flags=imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_COLLAPSE)

            # Update our state based on the window's visibility (i.e., if the user closed it).
            if new_visibility != app_state.show_video_display_window:
                app_state.show_video_display_window = new_visibility
                self.app.project_manager.project_dirty = True

            # We should only render the content if the window is visible and not collapsed.
            if new_visibility and is_expanded:
                should_render_content = True
        else:
            # For fixed mode, it's a static panel that's always present.
            imgui.begin("Video Display##CenterVideo",
                        flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS)
            should_render_content = True

        if should_render_content:
            # The original content of the render method, which draws the video frame
            stage_proc = self.app.stage_processor

            current_frame_for_texture = None
            if self.app.processor and self.app.processor.current_frame is not None:
                with self.app.processor.frame_lock:
                    if self.app.processor.current_frame is not None:
                        current_frame_for_texture = self.app.processor.current_frame.copy()

            video_frame_available = current_frame_for_texture is not None

            if video_frame_available:
                self.gui_instance.update_texture(self.gui_instance.frame_texture_id, current_frame_for_texture)
                available_w_video, available_h_video = imgui.get_content_region_available()

                if available_w_video > 0 and available_h_video > 0:
                    display_w, display_h, cursor_x_offset, cursor_y_offset = app_state.calculate_video_display_dimensions(
                        available_w_video, available_h_video)
                    if display_w > 0 and display_h > 0:
                        self._update_actual_video_image_rect(display_w, display_h, cursor_x_offset, cursor_y_offset)

                        win_content_x, win_content_y = imgui.get_cursor_pos()
                        imgui.set_cursor_pos((win_content_x + cursor_x_offset, win_content_y + cursor_y_offset))

                        uv0_x, uv0_y, uv1_x, uv1_y = app_state.get_video_uv_coords()
                        imgui.image(self.gui_instance.frame_texture_id, display_w, display_h, (uv0_x, uv0_y),
                                    (uv1_x, uv1_y))

                        # Store the item rect for overlay positioning, AFTER imgui.image
                        self._video_display_rect_min = imgui.get_item_rect_min()
                        self._video_display_rect_max = imgui.get_item_rect_max()

                        #--- User Defined ROI Drawing/Selection Logic ---
                        io = imgui.get_io()
                        #  Check hover based on the actual image rect stored by _update_actual_video_image_rect
                        is_hovering_actual_video_image = imgui.is_mouse_hovering_rect(
                            self._actual_video_image_rect_on_screen['min_x'],
                            self._actual_video_image_rect_on_screen['min_y'],
                            self._actual_video_image_rect_on_screen['max_x'],
                            self._actual_video_image_rect_on_screen['max_y']
                        )

                        if self.app.is_setting_user_roi_mode:
                            draw_list = imgui.get_window_draw_list()
                            mouse_screen_x, mouse_screen_y = io.mouse_pos

                            if is_hovering_actual_video_image:
                                if not self.waiting_for_point_click: # ROI Drawing phase
                                    if io.mouse_down[0] and not self.is_drawing_user_roi: # Left mouse button down
                                        self.is_drawing_user_roi = True
                                        self.user_roi_draw_start_screen_pos = (mouse_screen_x, mouse_screen_y)
                                        self.user_roi_draw_current_screen_pos = (mouse_screen_x, mouse_screen_y)
                                        self.drawn_user_roi_video_coords = None
                                        self.app.energy_saver.reset_activity_timer()

                                    if self.is_drawing_user_roi:
                                        self.user_roi_draw_current_screen_pos = (mouse_screen_x, mouse_screen_y)
                                        draw_list.add_rect(
                                            min(self.user_roi_draw_start_screen_pos[0],
                                                self.user_roi_draw_current_screen_pos[0]),
                                            min(self.user_roi_draw_start_screen_pos[1],
                                                self.user_roi_draw_current_screen_pos[1]),
                                            max(self.user_roi_draw_start_screen_pos[0],
                                                self.user_roi_draw_current_screen_pos[0]),
                                            max(self.user_roi_draw_start_screen_pos[1],
                                                self.user_roi_draw_current_screen_pos[1]),
                                            imgui.get_color_u32_rgba(1, 1, 0, 0.7), thickness=2
                                        )

                                    if not io.mouse_down[0] and self.is_drawing_user_roi: # Mouse released
                                        self.is_drawing_user_roi = False
                                        start_vid_coords = self._screen_to_video_coords(
                                            *self.user_roi_draw_start_screen_pos)
                                        end_vid_coords = self._screen_to_video_coords(
                                            *self.user_roi_draw_current_screen_pos)

                                        if start_vid_coords and end_vid_coords:
                                            vx1, vy1 = start_vid_coords
                                            vx2, vy2 = end_vid_coords
                                            roi_x, roi_y = min(vx1, vx2), min(vy1, vy2)
                                            roi_w, roi_h = abs(vx2 - vx1), abs(vy2 - vy1)

                                            if roi_w > 5 and roi_h > 5: # Minimum ROI size
                                                self.drawn_user_roi_video_coords = (roi_x, roi_y, roi_w, roi_h)
                                                self.waiting_for_point_click = True
                                                self.app.logger.info("ROI drawn. Click a point inside the ROI.",
                                                                     extra={'status_message': True, 'duration': 5.0})
                                            else:
                                                self.app.logger.info("Drawn ROI is too small. Please redraw.",
                                                                     extra={'status_message': True})
                                                self.drawn_user_roi_video_coords = None
                                        else:
                                            self.app.logger.warning(
                                                "Could not convert ROI screen coordinates to video coordinates (likely drawn outside video area).")
                                            self.drawn_user_roi_video_coords = None

                                elif self.waiting_for_point_click and self.drawn_user_roi_video_coords: # Point selection phase
                                    if imgui.is_mouse_clicked(0): # Left click
                                        self.app.energy_saver.reset_activity_timer()
                                        point_vid_coords = self._screen_to_video_coords(mouse_screen_x, mouse_screen_y)
                                        if point_vid_coords:
                                            roi_x, roi_y, roi_w, roi_h = self.drawn_user_roi_video_coords
                                            pt_x, pt_y = point_vid_coords
                                            if roi_x <= pt_x < roi_x + roi_w and roi_y <= pt_y < roi_y + roi_h:
                                                self.app.user_roi_and_point_set(self.drawn_user_roi_video_coords,
                                                                                point_vid_coords)
                                                self.waiting_for_point_click = False
                                                self.drawn_user_roi_video_coords = None
                                            else:
                                                self.app.logger.info(
                                                    "Clicked point is outside the drawn ROI. Please click inside.",
                                                    extra={'status_message': True})
                                        else:
                                            self.app.logger.info("Point click was outside the video content area.",
                                                                 extra={'status_message': True})
                            elif self.is_drawing_user_roi and not io.mouse_down[0]: # Mouse released outside hovered area while drawing
                                self.is_drawing_user_roi = False
                                self.app.logger.info("ROI drawing cancelled (mouse released outside video).",
                                                     extra={'status_message': True})

                        # Visualization of active User Fixed ROI (even when not setting)
                        if self.app.tracker and self.app.tracker.tracking_mode == "USER_FIXED_ROI" and \
                                self.app.tracker.user_roi_fixed and not self.app.is_setting_user_roi_mode:
                            draw_list = imgui.get_window_draw_list()
                            urx_vid, ury_vid, urw_vid, urh_vid = self.app.tracker.user_roi_fixed

                            roi_start_screen = self._video_to_screen_coords(urx_vid, ury_vid)
                            roi_end_screen = self._video_to_screen_coords(urx_vid + urw_vid, ury_vid + urh_vid)

                            if roi_start_screen and roi_end_screen:
                                draw_list.add_rect(roi_start_screen[0],roi_start_screen[1],roi_end_screen[0],roi_end_screen[1],imgui.get_color_u32_rgba(0.1,0.9,0.9,0.8),thickness=2)
                            if self.app.tracker.user_roi_tracked_point_relative: # UPDATED TO USE TRACKED POINT
                                abs_tracked_x_vid = self.app.tracker.user_roi_fixed[0] + int(self.app.tracker.user_roi_tracked_point_relative[0])
                                abs_tracked_y_vid = self.app.tracker.user_roi_fixed[1] + int(self.app.tracker.user_roi_tracked_point_relative[1])
                                point_screen_coords = self._video_to_screen_coords(abs_tracked_x_vid,abs_tracked_y_vid)
                                if point_screen_coords:
                                    draw_list.add_circle_filled(point_screen_coords[0], point_screen_coords[1], 5, imgui.get_color_u32_rgba(0.2, 1, 0.2, 0.9)) # Green moving dot
                                    if self.app.tracker.show_flow:
                                        dx_flow_vid, dy_flow_vid = self.app.tracker.user_roi_current_flow_vector
                                        flow_end_vid_x, flow_end_vid_y = abs_tracked_x_vid + int(dx_flow_vid * 10), abs_tracked_y_vid+int(dy_flow_vid*10)
                                        flow_end_screen_coords = self._video_to_screen_coords(flow_end_vid_x,flow_end_vid_y)
                                        if flow_end_screen_coords:
                                            draw_list.add_line(point_screen_coords[0], point_screen_coords[1], flow_end_screen_coords[0], flow_end_screen_coords[1], imgui.get_color_u32_rgba(1, 0.2, 0.2, 0.9), thickness=2)
                        self._handle_video_mouse_interaction(app_state)

                        if app_state.show_stage2_overlay and stage_proc.stage2_overlay_data_map and self.app.processor and \
                                self.app.processor.current_frame_index >= 0:
                            self._render_stage2_overlay(stage_proc, app_state)

                        self._render_playback_controls_overlay()
                        self._render_fps_controls_overlay()
                        self._render_video_zoom_pan_controls(app_state)

            else:
                self._render_drop_video_prompt()

        imgui.end()
        imgui.pop_style_var()

    def _handle_video_mouse_interaction(self, app_state):
        if not (self.app.processor and self.app.processor.current_frame is not None): return

        img_rect = self._actual_video_image_rect_on_screen
        is_hovering_video = imgui.is_mouse_hovering_rect(img_rect['min_x'], img_rect['min_y'], img_rect['max_x'],
                                                         img_rect['max_y'])

        if not is_hovering_video: return
        # If in ROI selection mode, these interactions should be disabled or handled differently.
        # For now, let's disable them if is_setting_user_roi_mode is active to prevent conflict.
        if self.app.is_setting_user_roi_mode:
            return

        io = imgui.get_io()
        if io.mouse_wheel != 0.0: #Basic check, no item active/hovered needed for wheel on background
           #Check if an ImGui window is focused and wants scroll, if so, don't zoom video
           #This is a bit tricky as is_window_hovered(HOVERED_ANY_WINDOW) and is_window_focused(FOCUSED_ANY_WINDOW)
           #might still allow scroll if mouse is over non-interactive part of a window.
           #A simpler check for now: if no specific item is active that would take wheel input.
            if not imgui.is_any_item_active() and not imgui.is_any_item_focused():
                mouse_screen_x, mouse_screen_y = io.mouse_pos
                view_width_on_screen = img_rect['w']
                view_height_on_screen = img_rect['h']
                if view_width_on_screen > 0 and view_height_on_screen > 0:
                    relative_mouse_x_in_view = (mouse_screen_x - img_rect['min_x']) / view_width_on_screen
                    relative_mouse_y_in_view = (mouse_screen_y - img_rect['min_y']) / view_height_on_screen
                    zoom_speed = 1.1
                    factor = zoom_speed if io.mouse_wheel > 0.0 else 1.0 / zoom_speed
                    app_state.adjust_video_zoom(factor, mouse_pos_normalized=(relative_mouse_x_in_view,
                                                                              relative_mouse_y_in_view))
                    self.app.energy_saver.reset_activity_timer()

        if app_state.video_zoom_factor > 1.0 and imgui.is_mouse_dragging(0) and not imgui.is_any_item_active():
            # Dragging with left mouse button
            delta_x_screen, delta_y_screen = io.mouse_delta
            view_width_on_screen = img_rect['w']
            view_height_on_screen = img_rect['h']
            if view_width_on_screen > 0 and view_height_on_screen > 0:
                pan_dx_norm_view = -delta_x_screen / view_width_on_screen
                pan_dy_norm_view = -delta_y_screen / view_height_on_screen
                app_state.pan_video_normalized_delta(pan_dx_norm_view, pan_dy_norm_view)
                self.app.energy_saver.reset_activity_timer()

    def _render_stage2_overlay(self, stage_proc, app_state):
        frame_overlay_data = stage_proc.stage2_overlay_data_map.get(self.app.processor.current_frame_index)
        if not frame_overlay_data: return

        draw_list = imgui.get_window_draw_list()
        img_rect = self._actual_video_image_rect_on_screen
        draw_list.push_clip_rect(img_rect['min_x'], img_rect['min_y'], img_rect['max_x'], img_rect['max_y'], True)

        dominant_pose_id = frame_overlay_data.get("dominant_pose_id")
        active_track_id = frame_overlay_data.get("active_interaction_track_id")
        is_occluded = frame_overlay_data.get("is_occluded", False)

        for pose in frame_overlay_data.get("poses", []):
            is_dominant = (pose.get("id") == dominant_pose_id)
            self._render_pose_skeleton(draw_list, pose, is_dominant)

        for box in frame_overlay_data.get("yolo_boxes", []):
            if not box or "bbox" not in box: continue

            p1 = self._video_to_screen_coords(box["bbox"][0], box["bbox"][1])
            p2 = self._video_to_screen_coords(box["bbox"][2], box["bbox"][3])

            if p1 and p2:
                is_active_interactor = (box.get("track_id") is not None and box.get("track_id") == active_track_id)
                is_locked_penis = (box.get("class_name") == "locked_penis")

                is_inferred_relative = (box.get("status") == constants.STATUS_INFERRED_RELATIVE)

                # --- NEW HIERARCHICAL HIGHLIGHTING LOGIC ---
                if is_active_interactor:
                    color = (1.0, 1.0, 0.0, 1.0)  # Bright Yellow for ACTIVE interactor
                    thickness = 3.0
                elif is_locked_penis:
                    color = (0.1, 1.0, 0.1, 0.95)  # Bright Green for LOCKED PENIS
                    thickness = 2.0
                elif is_inferred_relative:
                    color = (0.8, 0.4, 1.0, 0.85) # A distinct purple for inferred boxes
                    thickness = 1.0 # Thinner to indicate lower certainty
                else:
                    color, thickness, _ = self.app.utility.get_box_style(box)

                color_u32 = imgui.get_color_u32_rgba(*color)
                draw_list.add_rect(p1[0], p1[1], p2[0], p2[1], color_u32, thickness=thickness, rounding=2.0)

                label = f'{box.get("class_name", "?")}'
                if is_active_interactor:
                    label += " (ACTIVE)"
                elif is_inferred_relative:
                    label += " (Inferred)"
                draw_list.add_text(p1[0] + 3, p1[1] + 3, imgui.get_color_u32_rgba(1, 1, 1, 1), label)

        if is_occluded:
            draw_list.add_text(img_rect['min_x'] + 10, img_rect['max_y'] - 20, imgui.get_color_u32_rgba(1, 0.6, 0, 0.9),
                               "OCCLUSION (FALLBACK)")

        motion_mode = frame_overlay_data.get("motion_mode")
        is_vr_video = self.app.processor and self.app.processor.determined_video_type == 'VR'

        if motion_mode and is_vr_video:
            roi_to_use = None
            locked_penis_box = next(
                (b for b in frame_overlay_data.get("yolo_boxes", []) if b.get("class_name") == "locked_penis"), None)
            if locked_penis_box and "bbox" in locked_penis_box:
                x1, y1, x2, y2 = locked_penis_box["bbox"]
                roi_to_use = (x1, y1, x2 - x1, y2 - y1)

            if roi_to_use:
                interaction_class_proxy = None
                position = frame_overlay_data.get("atr_assigned_position")
                if position:
                    if "Blowjob" in position:
                        interaction_class_proxy = "face"
                    elif "Handjob" in position:
                        interaction_class_proxy = "hand"

                self._render_motion_mode_overlay(
                    draw_list=draw_list,
                    motion_mode=motion_mode,
                    interaction_class=interaction_class_proxy,
                    roi_video_coords=roi_to_use
                )

        draw_list.pop_clip_rect()

    def _render_video_zoom_pan_controls(self, app_state):
        style = imgui.get_style()
        button_h_ref = imgui.get_frame_height()
        img_rect = self._actual_video_image_rect_on_screen
        if img_rect['w'] <= 0 or img_rect['h'] <= 0: return
        num_control_lines = 1
        pan_buttons_active = app_state.video_zoom_factor > 1.0
        if pan_buttons_active: num_control_lines = 2
        group_height = (button_h_ref * num_control_lines) + (
                    style.item_spacing[1] * (num_control_lines - 1 if num_control_lines > 1 else 0))
        overlay_ctrl_y = img_rect['max_y'] - group_height - (style.item_spacing[1] * 2)
        overlay_ctrl_x = img_rect['min_x'] + style.item_spacing[1]
        overlay_ctrl_y = max(img_rect['min_y'] + style.item_spacing[1], overlay_ctrl_y)
        overlay_ctrl_x = max(img_rect['min_x'] + style.item_spacing[1], overlay_ctrl_x)
        imgui.set_cursor_screen_pos((overlay_ctrl_x, overlay_ctrl_y))

        imgui.begin_group()

        if pan_buttons_active:
            # Pan Arrows Block (Left, Right, Up, Down on one line)
            if imgui.arrow_button("##VidOverPanLeft", imgui.DIRECTION_LEFT):
                app_state.pan_video_normalized_delta(-app_state.video_pan_step, 0)
            imgui.same_line()
            if imgui.arrow_button("##VidOverPanRight", imgui.DIRECTION_RIGHT):
                app_state.pan_video_normalized_delta(app_state.video_pan_step, 0)
            imgui.same_line()
            if imgui.arrow_button("##VidOverPanUp", imgui.DIRECTION_UP):
                app_state.pan_video_normalized_delta(0, -app_state.video_pan_step)
            imgui.same_line()
            if imgui.arrow_button("##VidOverPanDown", imgui.DIRECTION_DOWN):
                app_state.pan_video_normalized_delta(0, app_state.video_pan_step)

        # Zoom Settings Block (Z-In, Z-Out, Rst, Text on one line)
        if imgui.button("Z-In##VidOverZoomIn"):
            app_state.adjust_video_zoom(1.2)
        imgui.same_line()
        if imgui.button("Z-Out##VidOverZoomOut"):
            app_state.adjust_video_zoom(1 / 1.2)
        imgui.same_line()
        if imgui.button("Rst##VidOverZoomReset"):
            app_state.reset_video_zoom_pan()
        imgui.same_line()
        imgui.text(f"{app_state.video_zoom_factor:.1f}x")

        imgui.end_group()

    def _render_drop_video_prompt(self):
        cursor_start_pos = imgui.get_cursor_pos()
        win_size = imgui.get_window_size()
        text_to_display = "Drag and drop a video file here."
        text_size = imgui.calc_text_size(text_to_display)
        if win_size[0] > text_size[0] and win_size[1] > text_size[1]:  # Check if window is large enough for text
            imgui.set_cursor_pos(((win_size[0] - text_size[0]) * 0.5 + cursor_start_pos[0],
                                  (win_size[1] - text_size[1]) * 0.5 + cursor_start_pos[1]))
        imgui.text(text_to_display)
