import imgui
from typing import Optional, Tuple


import config.constants as constants
from config.element_group_colors import VideoDisplayColors


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

        # Oscillation Area Drawing state
        self.is_drawing_oscillation_area: bool = False
        self.oscillation_area_draw_start_screen_pos: tuple = (0, 0)  # In ImGui screen space
        self.oscillation_area_draw_current_screen_pos: tuple = (0, 0)  # In ImGui screen space
        self.drawn_oscillation_area_video_coords: tuple | None = None  # (x,y,w,h) in original video frame pixel space
        self.waiting_for_oscillation_point_click: bool = False

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

        # Check if live tracking is running  
        is_live_tracking_running = (self.app.processor and
                                    self.app.processor.is_processing and
                                    self.app.processor.enable_tracker_processing)
        
        controls_disabled = stage_proc.full_analysis_active or is_live_tracking_running or not file_mgr.video_path

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
        if imgui.button(ICON_JUMP_START + "##VidOverStart", width=pb_icon_w): event_handlers.handle_playback_control("jump_start")
        imgui.same_line(spacing=pb_btn_spacing)
        if imgui.button(ICON_PREV_FRAME + "##VidOverPrev", width=pb_icon_w): event_handlers.handle_playback_control("prev_frame")
        imgui.same_line(spacing=pb_btn_spacing)
        # The condition now checks for the "not paused" state to accurately reflect playing vs. paused.
        is_playing = self.app.processor and self.app.processor.is_processing and not self.app.processor.pause_event.is_set()
        play_pause_icon = ICON_PAUSE if is_playing else ICON_PLAY

        #play_pause_icon = ICON_PAUSE if self.app.processor and self.app.processor.is_processing else ICON_PLAY
        if imgui.button(play_pause_icon + "##VidOverPlayPause", width=pb_play_w): event_handlers.handle_playback_control("play_pause")
        imgui.same_line(spacing=2)
        if imgui.button(ICON_STOP + "##VidOverStop", width=pb_stop_w): event_handlers.handle_playback_control("stop")
        imgui.same_line(spacing=pb_btn_spacing)
        if imgui.button(ICON_NEXT_FRAME + "##VidOverNext", width=pb_icon_w): event_handlers.handle_playback_control("next_frame")
        imgui.same_line(spacing=pb_btn_spacing)
        if imgui.button(ICON_JUMP_END + "##VidOverEnd", width=pb_icon_w): event_handlers.handle_playback_control("jump_end")
        imgui.end_group()

        if controls_disabled:
            imgui.pop_style_var()
            imgui.internal.pop_item_flag()

    

    def _render_pose_skeleton(self, draw_list, pose_data: dict, is_dominant: bool):
        """Draws the skeleton, highlighting the dominant pose."""
        keypoints = pose_data.get("keypoints", [])
        if not isinstance(keypoints, list) or len(keypoints) < 17: return

        # --- Color based on whether this is the dominant pose ---
        if is_dominant:
            limb_color = imgui.get_color_u32_rgba(*VideoDisplayColors.DOMINANT_LIMB)  # Bright Green
            kpt_color = imgui.get_color_u32_rgba(*VideoDisplayColors.DOMINANT_KEYPOINT)  # Bright Orange
            thickness = 2
        else:
            limb_color = imgui.get_color_u32_rgba(*VideoDisplayColors.MUTED_LIMB)  # Muted Cyan
            kpt_color = imgui.get_color_u32_rgba(*VideoDisplayColors.MUTED_KEYPOINT)  # Muted Red
            thickness = 1

        skeleton = [[0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 11], [6, 12], [11, 12], [5, 7], [7, 9], [6, 8], [8, 10], [11, 13], [13, 15], [12, 14], [14, 16]]

        for conn in skeleton:
            idx1, idx2 = conn
            if not (idx1 < len(keypoints) and idx2 < len(keypoints)): continue
            kp1, kp2 = keypoints[idx1], keypoints[idx2]
            if kp1[2] > 0.5 and kp2[2] > 0.5:
                p1_screen = self._video_to_screen_coords(int(kp1[0]), int(kp1[1]))
                p2_screen = self._video_to_screen_coords(int(kp2[0]), int(kp2[1]))
                if p1_screen and p2_screen:
                    draw_list.add_line(p1_screen[0], p1_screen[1], p2_screen[0], p2_screen[1], limb_color, thickness=thickness)

        for kp in keypoints:
            if kp[2] > 0.5:
                p_screen = self._video_to_screen_coords(int(kp[0]), int(kp[1]))
                if p_screen:
                    draw_list.add_circle_filled(p_screen[0], p_screen[1], 3.0, kpt_color)

    def _render_motion_mode_overlay(self, draw_list, motion_mode: Optional[str], interaction_class: Optional[str], roi_video_coords: Tuple[int, int, int, int]):
        """Renders the motion mode text (Thrusting, Riding, etc.) as an ImGui overlay."""
        if not motion_mode or motion_mode == 'undetermined':
            return

        mode_color = imgui.get_color_u32_rgba(*VideoDisplayColors.MOTION_UNDETERMINED)
        mode_text = "Undetermined"

        if motion_mode == 'thrusting':
            mode_text = "Thrusting"
            mode_color = imgui.get_color_u32_rgba(*VideoDisplayColors.MOTION_THRUSTING)
        elif motion_mode == 'riding':
            mode_color = imgui.get_color_u32_rgba(*VideoDisplayColors.MOTION_RIDING)
            if interaction_class == 'face':
                mode_text = "Blowing"
            elif interaction_class == 'hand':
                mode_text = "Stroking"
            else:
                mode_text = "Riding"

        if mode_text == "Undetermined":
            return

        # Anchor point in video coordinates: top-left of the box
        box_x, box_y, _, _ = roi_video_coords
        anchor_vid_x = box_x
        anchor_vid_y = box_y

        anchor_screen_pos = self._video_to_screen_coords(int(anchor_vid_x), int(anchor_vid_y))

        if anchor_screen_pos:
            # Position text inside the top-left corner with padding
            text_pos_x = anchor_screen_pos[0] + 5  # 5 pixels of padding from the left
            text_pos_y = anchor_screen_pos[1] + 5  # 5 pixels of padding from the top

            img_rect = self._actual_video_image_rect_on_screen
            text_size = imgui.calc_text_size(mode_text) # Calculate text size to check bounds
            if (text_pos_x + text_size[0]) < img_rect['max_x'] and (text_pos_y + text_size[1]) < img_rect['max_y']:
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
            is_expanded, new_visibility = imgui.begin("Video Display", closable=True, flags=imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_COLLAPSE)

            # Update our state based on the window's visibility (i.e., if the user closed it).
            if new_visibility != app_state.show_video_display_window:
                app_state.show_video_display_window = new_visibility
                self.app.project_manager.project_dirty = True

            # We should only render the content if the window is visible and not collapsed.
            if new_visibility and is_expanded:
                should_render_content = True
        else:
            # For fixed mode, it's a static panel that's always present.
            imgui.begin("Video Display##CenterVideo", flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS)
            should_render_content = True

        if should_render_content:
            stage_proc = self.app.stage_processor

            # If video feed is disabled, just show the drop prompt if no video is loaded,
            # otherwise show a blank placeholder and skip all expensive processing.
            if not app_state.show_video_feed:
                if not (self.app.processor and self.app.processor.current_frame is not None):
                    self._render_drop_video_prompt()
                # Otherwise, do nothing, leaving the panel blank.
            else:
                # --- Original logic when video feed is enabled ---
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
                        display_w, display_h, cursor_x_offset, cursor_y_offset = app_state.calculate_video_display_dimensions(available_w_video, available_h_video)
                        if display_w > 0 and display_h > 0:
                            self._update_actual_video_image_rect(display_w, display_h, cursor_x_offset, cursor_y_offset)

                            win_content_x, win_content_y = imgui.get_cursor_pos()
                            imgui.set_cursor_pos((win_content_x + cursor_x_offset, win_content_y + cursor_y_offset))

                            uv0_x, uv0_y, uv1_x, uv1_y = app_state.get_video_uv_coords()
                            imgui.image(self.gui_instance.frame_texture_id, display_w, display_h, (uv0_x, uv0_y), (uv1_x, uv1_y))

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
                                                imgui.get_color_u32_rgba(*VideoDisplayColors.ROI_DRAWING), thickness=2
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
                                                    self.app.logger.info("ROI drawn. Click a point inside the ROI.", extra={'status_message': True, 'duration': 5.0})
                                                else:
                                                    self.app.logger.info("Drawn ROI is too small. Please redraw.", extra={'status_message': True})
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
                                                    self.app.user_roi_and_point_set(self.drawn_user_roi_video_coords, point_vid_coords)
                                                    self.waiting_for_point_click = False
                                                    self.drawn_user_roi_video_coords = None
                                                else:
                                                    self.app.logger.info(
                                                        "Clicked point is outside the drawn ROI. Please click inside.",
                                                        extra={'status_message': True})
                                            else:
                                                self.app.logger.info("Point click was outside the video content area.", extra={'status_message': True})
                                elif self.is_drawing_user_roi and not io.mouse_down[0]: # Mouse released outside hovered area while drawing
                                    self.is_drawing_user_roi = False
                                    self.app.logger.info("ROI drawing cancelled (mouse released outside video).", extra={'status_message': True})

                            # --- Oscillation Area Drawing/Selection Logic ---
                            if self.app.is_setting_oscillation_area_mode:
                                draw_list = imgui.get_window_draw_list()
                                mouse_screen_x, mouse_screen_y = io.mouse_pos

                                if is_hovering_actual_video_image:
                                    if not self.waiting_for_oscillation_point_click: # Area Drawing phase
                                        if io.mouse_down[0] and not self.is_drawing_oscillation_area: # Left mouse button down
                                            self.is_drawing_oscillation_area = True
                                            self.oscillation_area_draw_start_screen_pos = (mouse_screen_x, mouse_screen_y)
                                            self.oscillation_area_draw_current_screen_pos = (mouse_screen_x, mouse_screen_y)
                                            self.drawn_oscillation_area_video_coords = None
                                            self.app.energy_saver.reset_activity_timer()

                                        if self.is_drawing_oscillation_area:
                                            self.oscillation_area_draw_current_screen_pos = (mouse_screen_x, mouse_screen_y)
                                            draw_list.add_rect(
                                                min(self.oscillation_area_draw_start_screen_pos[0],
                                                    self.oscillation_area_draw_current_screen_pos[0]),
                                                min(self.oscillation_area_draw_start_screen_pos[1],
                                                    self.oscillation_area_draw_current_screen_pos[1]),
                                                max(self.oscillation_area_draw_start_screen_pos[0],
                                                    self.oscillation_area_draw_current_screen_pos[0]),
                                                max(self.oscillation_area_draw_start_screen_pos[1],
                                                    self.oscillation_area_draw_current_screen_pos[1]),
                                                imgui.get_color_u32_rgba(0, 255, 255, 255), thickness=2  # Cyan color
                                            )

                                        if not io.mouse_down[0] and self.is_drawing_oscillation_area: # Mouse released
                                            self.is_drawing_oscillation_area = False
                                            start_vid_coords = self._screen_to_video_coords(
                                                *self.oscillation_area_draw_start_screen_pos)
                                            end_vid_coords = self._screen_to_video_coords(
                                                *self.oscillation_area_draw_current_screen_pos)

                                            if start_vid_coords and end_vid_coords:
                                                vx1, vy1 = start_vid_coords
                                                vx2, vy2 = end_vid_coords
                                                area_x, area_y = min(vx1, vx2), min(vy1, vy2)
                                                area_w, area_h = abs(vx2 - vx1), abs(vy2 - vy1)

                                            if area_w > 5 and area_h > 5: # Minimum area size
                                                self.drawn_oscillation_area_video_coords = (area_x, area_y, area_w, area_h)
                                                self.waiting_for_oscillation_point_click = True
                                                self.app.logger.info("Oscillation area drawn. Setting tracking point to center.", extra={'status_message': True, 'duration': 5.0})
                                                if hasattr(self.app, 'tracker') and self.app.tracker:
                                                    current_frame = None
                                                    if self.app.processor and self.app.processor.current_frame is not None:
                                                        current_frame = self.app.processor.current_frame.copy()
                                                    center_x = area_x + area_w // 2
                                                    center_y = area_y + area_h // 2
                                                    point_vid_coords = (center_x, center_y)
                                                    self.app.tracker.set_oscillation_area_and_point(
                                                        (area_x, area_y, area_w, area_h),
                                                        point_vid_coords,
                                                        current_frame
                                                    )
                                                # --- FULLY RESET DRAWING STATE AND EXIT MODE ---
                                                self.waiting_for_oscillation_point_click = False
                                                self.drawn_oscillation_area_video_coords = None
                                                self.is_drawing_oscillation_area = False
                                                self.oscillation_area_draw_start_screen_pos = (0, 0)
                                                self.oscillation_area_draw_current_screen_pos = (0, 0)
                                                self.app.is_setting_oscillation_area_mode = False
                                            else:
                                                self.app.logger.info("Drawn oscillation area is too small. Please redraw.", extra={'status_message': True})
                                                self.drawn_oscillation_area_video_coords = None
                                        else:
                                            self.app.logger.warning(
                                                "Could not convert oscillation area screen coordinates to video coordinates (likely drawn outside video area).")
                                            self.drawn_oscillation_area_video_coords = None

                                elif self.waiting_for_oscillation_point_click and self.drawn_oscillation_area_video_coords: # Point selection phase
                                    # Use center point of the area as the tracking point
                                    area_x, area_y, area_w, area_h = self.drawn_oscillation_area_video_coords
                                    center_x = area_x + area_w // 2
                                    center_y = area_y + area_h // 2
                                    point_vid_coords = (center_x, center_y)
                                    
                                    # Set the oscillation area immediately without requiring point click
                                    if hasattr(self.app, 'tracker') and self.app.tracker:
                                        current_frame = None
                                        if self.app.processor and self.app.processor.current_frame is not None:
                                            current_frame = self.app.processor.current_frame.copy()
                                        self.app.tracker.set_oscillation_area_and_point(
                                            self.drawn_oscillation_area_video_coords,
                                            point_vid_coords,
                                            current_frame
                                        )
                                    self.waiting_for_oscillation_point_click = False
                                    self.drawn_oscillation_area_video_coords = None
                                    # Clear drawing state to prevent showing both rectangles
                                    self.is_drawing_oscillation_area = False
                                    self.oscillation_area_draw_start_screen_pos = (0, 0)
                                    self.oscillation_area_draw_current_screen_pos = (0, 0)
                            elif self.is_drawing_oscillation_area and not io.mouse_down[0]: # Mouse released outside hovered area while drawing
                                self.is_drawing_oscillation_area = False
                                self.app.logger.info("Oscillation area drawing cancelled (mouse released outside video).", extra={'status_message': True})

                            # Visualization of active Oscillation Area (even when not setting)
                            if (
                                self.app.tracker and 
                                self.app.tracker.oscillation_area_fixed is not None and 
                                self.app.tracker.oscillation_grid_blocks and
                                not self.app.is_setting_oscillation_area_mode
                            ):
                                draw_list = imgui.get_window_draw_list()
                                ax_vid, ay_vid, aw_vid, ah_vid = self.app.tracker.oscillation_area_fixed

                                area_start_screen = self._video_to_screen_coords(ax_vid, ay_vid)
                                area_end_screen = self._video_to_screen_coords(ax_vid + aw_vid, ay_vid + ah_vid)

                                if area_start_screen and area_end_screen:
                                    # Draw blue outline for the area
                                    draw_list.add_rect(area_start_screen[0], area_start_screen[1], area_end_screen[0], area_end_screen[1], imgui.get_color_u32_rgba(0, 128, 255, 255), thickness=2)
                                    # Add label
                                    draw_list.add_text(area_start_screen[0], area_start_screen[1] - 15, imgui.get_color_u32_rgba(0, 255, 255, 255), "Oscillation Area")

                                    # Draw the grid blocks
                                    # Get active block positions from tracker if available
                                    active_block_positions = set()
                                    if hasattr(self.app.tracker, 'oscillation_active_block_positions'):
                                        active_block_positions = set(self.app.tracker.oscillation_active_block_positions)
                                    elif hasattr(self.app.tracker, 'last_active_block_positions'):
                                        active_block_positions = set(self.app.tracker.last_active_block_positions)
                                    # Fallback: try to get from tracker attribute if exposed
                                    elif hasattr(self.app.tracker, 'get_active_block_positions'):
                                        active_block_positions = set(self.app.tracker.get_active_block_positions())
                                    # If not available, just draw all as grey

                                # Try to infer grid dimensions
                                grid_blocks = self.app.tracker.oscillation_grid_blocks
                                num_blocks = len(grid_blocks)
                                # Try to get max_blocks_w from tracker if available
                                max_blocks_w = getattr(self.app.tracker, 'oscillation_max_blocks_w', 0)
                                if max_blocks_w <= 0:
                                    # Fallback: estimate as square
                                    max_blocks_w = int(num_blocks ** 0.5) if num_blocks > 0 else 1
                                for i, (x, y, w, h) in enumerate(grid_blocks):
                                    grid_start = self._video_to_screen_coords(x, y)
                                    grid_end = self._video_to_screen_coords(x + w, y + h)
                                    if grid_start and grid_end:
                                        # Compute (r, c) for this block
                                        r = i // max_blocks_w
                                        c = i % max_blocks_w
                                        color = (0, 0, 0, 0.3)  # Faded grey
                                        if (r, c) in self.app.tracker.oscillation_active_block_positions:
                                            color = (0, 255, 0, 255)  # Green for active
                                        #draw_list.add_rect(grid_start[0], grid_start[1], grid_end[0], grid_end[1], imgui.get_color_u32_rgba(*color), thickness=1)

                            # Visualization of active User Fixed ROI (even when not setting)
                            if self.app.tracker and self.app.tracker.tracking_mode == "USER_FIXED_ROI" and \
                                    self.app.tracker.user_roi_fixed and not self.app.is_setting_user_roi_mode:
                                draw_list = imgui.get_window_draw_list()
                                urx_vid, ury_vid, urw_vid, urh_vid = self.app.tracker.user_roi_fixed

                                roi_start_screen = self._video_to_screen_coords(urx_vid, ury_vid)
                                roi_end_screen = self._video_to_screen_coords(urx_vid + urw_vid, ury_vid + urh_vid)

                                if roi_start_screen and roi_end_screen:
                                    draw_list.add_rect(roi_start_screen[0],roi_start_screen[1],roi_end_screen[0],roi_end_screen[1],imgui.get_color_u32_rgba(*VideoDisplayColors.ROI_BORDER),thickness=2)
                                if self.app.tracker.user_roi_tracked_point_relative: # UPDATED TO USE TRACKED POINT
                                    abs_tracked_x_vid = self.app.tracker.user_roi_fixed[0] + int(self.app.tracker.user_roi_tracked_point_relative[0])
                                    abs_tracked_y_vid = self.app.tracker.user_roi_fixed[1] + int(self.app.tracker.user_roi_tracked_point_relative[1])
                                    point_screen_coords = self._video_to_screen_coords(abs_tracked_x_vid,abs_tracked_y_vid)
                                    if point_screen_coords:
                                        draw_list.add_circle_filled(point_screen_coords[0], point_screen_coords[1], 5, imgui.get_color_u32_rgba(*VideoDisplayColors.TRACKING_POINT)) # Green moving dot
                                        if self.app.tracker.show_flow:
                                            dx_flow_vid, dy_flow_vid = self.app.tracker.user_roi_current_flow_vector
                                            flow_end_vid_x, flow_end_vid_y = abs_tracked_x_vid + int(dx_flow_vid * 10), abs_tracked_y_vid+int(dy_flow_vid*10)
                                            flow_end_screen_coords = self._video_to_screen_coords(flow_end_vid_x,flow_end_vid_y)
                                            if flow_end_screen_coords:
                                                draw_list.add_line(point_screen_coords[0], point_screen_coords[1], flow_end_screen_coords[0], flow_end_screen_coords[1], imgui.get_color_u32_rgba(*VideoDisplayColors.FLOW_VECTOR), thickness=2)
                            self._handle_video_mouse_interaction(app_state)

                            if app_state.show_stage2_overlay and stage_proc.stage2_overlay_data_map and self.app.processor and \
                                    self.app.processor.current_frame_index >= 0:
                                self._render_stage2_overlay(stage_proc, app_state)

                            # Only show live tracker info if the Stage 2 overlay isn't active
                            if self.app.tracker and self.app.tracker.tracking_active and not (app_state.show_stage2_overlay and stage_proc.stage2_overlay_data_map):
                                draw_list = imgui.get_window_draw_list()
                                img_rect = self._actual_video_image_rect_on_screen
                                # Clip rendering to the video display area
                                draw_list.push_clip_rect(img_rect['min_x'], img_rect['min_y'], img_rect['max_x'], img_rect['max_y'], True)
                                self._render_live_tracker_overlay(draw_list)
                                draw_list.pop_clip_rect()

                            self._render_playback_controls_overlay()
                            self._render_video_zoom_pan_controls(app_state)

                            # --- Overlay: Top-right buttons for L/R Dial and Gauge ---
                            def render_overlay_toggle(label, visible_attr):
                                pushed = False
                                if not getattr(app_state, visible_attr, False):
                                    imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)
                                    pushed = True
                                if imgui.button(label):
                                    setattr(app_state, visible_attr, not getattr(app_state, visible_attr))
                                    self.app.project_manager.project_dirty = True
                                if pushed:
                                    imgui.pop_style_var()

                            # Position at top right of video frame
                            btn_labels = ["Gauge T1", "Gauge T2", "Dial T2"]
                            btn_attrs = ["show_gauge_window_timeline1", "show_gauge_window_timeline2", "show_lr_dial_graph"]
                            btn_widths = [imgui.calc_text_size(lbl)[0] + imgui.get_style().frame_padding[0]*2 + imgui.get_style().item_spacing[0] for lbl in btn_labels]
                            total_btn_width = sum(btn_widths)
                            # btn_height = imgui.get_frame_height()
                            img_rect = self._actual_video_image_rect_on_screen
                            top_right_x = img_rect['max_x'] - total_btn_width + 4
                            top_right_y = img_rect['min_y'] + 8
                            imgui.set_cursor_screen_pos((top_right_x, top_right_y))
                            use_small_font = hasattr(self.gui_instance, 'small_font') and self.gui_instance.small_font and getattr(self.gui_instance.small_font, 'is_loaded', lambda: True)()
                            if use_small_font:
                                imgui.push_font(self.gui_instance.small_font)
                            for i, (lbl, attr) in enumerate(zip(btn_labels, btn_attrs)):
                                render_overlay_toggle(lbl, attr)
                                if i < len(btn_labels) - 1:
                                    imgui.same_line(spacing=4)
                            if use_small_font:
                                imgui.pop_font()

                # --- Interactive Refinement Overlay and Click Handling ---
                if self.app.app_state_ui.interactive_refinement_mode_enabled:
                    # 1. Render the bounding boxes so the user can see what to click.
                    # We reuse the existing stage 2 overlay logic for this.
                    if self.app.stage_processor.stage2_overlay_data_map:
                        self._render_stage2_overlay(self.app.stage_processor, self.app.app_state_ui)

                    # 2. Handle the mouse click for the "hint".
                    io = imgui.get_io()
                    is_hovering_video = imgui.is_mouse_hovering_rect(
                        self._actual_video_image_rect_on_screen['min_x'], self._actual_video_image_rect_on_screen['min_y'],
                        self._actual_video_image_rect_on_screen['max_x'], self._actual_video_image_rect_on_screen['max_y'])

                    if is_hovering_video and imgui.is_mouse_clicked(
                            0) and not self.app.stage_processor.refinement_analysis_active:
                        mouse_x, mouse_y = io.mouse_pos
                        current_frame_idx = self.app.processor.current_frame_index

                        # Find the chapter at the current frame
                        chapter = self.app.funscript_processor.get_chapter_at_frame(current_frame_idx)
                        if not chapter:
                            self.app.logger.info("Cannot refine: Please click within a chapter boundary.", extra={'status_message': True})
                            return

                        # Find which bounding box was clicked
                        overlay_data = self.app.stage_processor.stage2_overlay_data_map.get(current_frame_idx)
                        if overlay_data and "yolo_boxes" in overlay_data:
                            for box in overlay_data["yolo_boxes"]:
                                p1 = self._video_to_screen_coords(box["bbox"][0], box["bbox"][1])
                                p2 = self._video_to_screen_coords(box["bbox"][2], box["bbox"][3])
                                if p1 and p2 and p1[0] <= mouse_x <= p2[0] and p1[1] <= mouse_y <= p2[1]:
                                    clicked_track_id = box.get("track_id")
                                    if clicked_track_id is not None:
                                        self.app.logger.info(f"Hint received! Refining chapter '{chapter.position_short_name}' "f"to follow object with track_id: {clicked_track_id}", extra={'status_message': True})
                                        # Trigger the backend process
                                        self.app.event_handlers.handle_interactive_refinement_click(chapter, clicked_track_id)
                                        break  # Stop after finding the first clicked box
                if not video_frame_available:
                    self._render_drop_video_prompt()

        imgui.end()
        imgui.pop_style_var()

    def _handle_video_mouse_interaction(self, app_state):
        if not (self.app.processor and self.app.processor.current_frame is not None): return

        img_rect = self._actual_video_image_rect_on_screen
        is_hovering_video = imgui.is_mouse_hovering_rect(img_rect['min_x'], img_rect['min_y'], img_rect['max_x'], img_rect['max_y'])

        if not is_hovering_video: return
        # If in ROI selection mode, these interactions should be disabled or handled differently.
        # For now, let's disable them if is_setting_user_roi_mode is active to prevent conflict.
        if self.app.is_setting_user_roi_mode or self.app.is_setting_oscillation_area_mode:
            return

        io = imgui.get_io()
        if io.mouse_wheel != 0.0:
            # Prevent zoom if any ImGui window is hovered, unless it's this specific video window.
            # This stops the video from zooming when scrolling over other windows like the file dialog.
            is_video_window_hovered = imgui.is_window_hovered(
                imgui.HOVERED_ROOT_WINDOW | imgui.HOVERED_CHILD_WINDOWS
            )
            if is_video_window_hovered and not imgui.is_any_item_active():
                mouse_screen_x, mouse_screen_y = io.mouse_pos
                view_width_on_screen = img_rect['w']
                view_height_on_screen = img_rect['h']
                if view_width_on_screen > 0 and view_height_on_screen > 0:
                    relative_mouse_x_in_view = (mouse_screen_x - img_rect['min_x']) / view_width_on_screen
                    relative_mouse_y_in_view = (mouse_screen_y - img_rect['min_y']) / view_height_on_screen
                    zoom_speed = 1.1
                    factor = zoom_speed if io.mouse_wheel > 0.0 else 1.0 / zoom_speed
                    app_state.adjust_video_zoom(factor, mouse_pos_normalized=(relative_mouse_x_in_view, relative_mouse_y_in_view))
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

    def _render_live_tracker_overlay(self, draw_list):
        """Renders overlays specific to the live tracker, like motion mode."""
        tracker = self.app.tracker

        # Ensure the tracker is active and has a defined ROI to anchor the text
        if not tracker or not tracker.tracking_active or not tracker.roi:
            return

        # Check if the video is VR, as this feature is VR-specific
        is_vr_video = tracker._is_vr_video()

        if tracker.enable_inversion_detection and is_vr_video:
            # Get the necessary data from the live tracker instance
            interaction_class = tracker.main_interaction_class
            roi_video_coords = tracker.roi
            motion_mode = tracker.motion_mode

            # Call the existing motion mode rendering function with live data
            self._render_motion_mode_overlay(
                draw_list=draw_list,
                motion_mode=motion_mode,
                interaction_class=interaction_class,
                roi_video_coords=roi_video_coords
            )

    def _render_stage2_overlay(self, stage_proc, app_state):
        frame_overlay_data = stage_proc.stage2_overlay_data_map.get(self.app.processor.current_frame_index)
        if not frame_overlay_data: return

        current_chapter = self.app.funscript_processor.get_chapter_at_frame(self.app.processor.current_frame_index)

        draw_list = imgui.get_window_draw_list()
        img_rect = self._actual_video_image_rect_on_screen
        draw_list.push_clip_rect(img_rect['min_x'], img_rect['min_y'], img_rect['max_x'], img_rect['max_y'], True)

        dominant_pose_id = frame_overlay_data.get("dominant_pose_id")
        active_track_id = frame_overlay_data.get("active_interaction_track_id")
        is_occluded = frame_overlay_data.get("is_occluded", False)
        # Get the list of aligned fallback candidate IDs for this frame
        aligned_fallback_ids = set(frame_overlay_data.get("atr_aligned_fallback_candidate_ids", []))


        for pose in frame_overlay_data.get("poses", []):
            is_dominant = (pose.get("id") == dominant_pose_id)
            self._render_pose_skeleton(draw_list, pose, is_dominant)

        for box in frame_overlay_data.get("yolo_boxes", []):
            if not box or "bbox" not in box: continue

            p1 = self._video_to_screen_coords(box["bbox"][0], box["bbox"][1])
            p2 = self._video_to_screen_coords(box["bbox"][2], box["bbox"][3])

            if p1 and p2:
                track_id = box.get("track_id")
                is_active_interactor = (track_id is not None and track_id == active_track_id)
                is_locked_penis = (box.get("class_name") == "locked_penis")
                is_inferred_status = (box.get("status") == constants.STATUS_INFERRED_RELATIVE or box.get("status") == constants.STATUS_POSE_INFERRED)
                is_of_recovered = (box.get("status") == constants.STATUS_OF_RECOVERED or box.get("status") == constants.STATUS_OF_RECOVERED)

                # Check if this box is an aligned fallback candidate
                is_aligned_candidate = (track_id is not None and track_id in aligned_fallback_ids)

                is_refined_track = False
                if current_chapter and current_chapter.refined_track_id is not None:
                    if track_id == current_chapter.refined_track_id:
                        is_refined_track = True

                # --- HIERARCHICAL HIGHLIGHTING LOGIC ---
                if is_refined_track:
                    color = VideoDisplayColors.PERSISTENT_REFINED_TRACK  # Bright Cyan for the persistent refined track
                    thickness = 3.0
                elif is_active_interactor:
                    color = VideoDisplayColors.ACTIVE_INTERACTOR  # Bright Yellow for the ACTIVE interactor
                    thickness = 3.0
                elif is_locked_penis:
                    color = VideoDisplayColors.LOCKED_PENIS  # Bright Green for LOCKED PENIS
                    thickness = 2.0
                    # If it's a locked penis and has a visible part, draw the solid fill first.
                    if "visible_bbox" in box and box["visible_bbox"]:
                        vis_bbox = box["visible_bbox"]
                        p1_vis = self._video_to_screen_coords(vis_bbox[0], vis_bbox[1])
                        p2_vis = self._video_to_screen_coords(vis_bbox[2], vis_bbox[3])
                        if p1_vis and p2_vis:
                            # Use a semi-transparent fill of the same base color
                            fill_color = VideoDisplayColors.FILL_COLOR
                            fill_color_u32 = imgui.get_color_u32_rgba(*fill_color)
                            draw_list.add_rect_filled(p1_vis[0], p1_vis[1], p2_vis[0], p2_vis[1], fill_color_u32, rounding=2.0)
                elif is_aligned_candidate:
                    color = VideoDisplayColors.ALIGNED_FALLBACK  # Orange for ALIGNED FALLBACK candidates
                    thickness = 1.5
                elif is_inferred_status:
                    color = VideoDisplayColors.INFERRED_BOX # A distinct purple for inferred boxes
                    thickness = 1.0
                else:
                    color, thickness, _ = self.app.utility.get_box_style(box)

                color_u32 = imgui.get_color_u32_rgba(*color)
                draw_list.add_rect(p1[0], p1[1], p2[0], p2[1], color_u32, thickness=thickness, rounding=2.0)

                track_id_str = f" (id: {track_id})" if track_id is not None else ""
                label = f'{box.get("class_name", "?")}{track_id_str}'

                if is_active_interactor:
                    label += " (ACTIVE)"
                elif is_aligned_candidate:
                    label += " (Aligned)"
                elif is_inferred_status:
                    label += " (Inferred)"

                if is_of_recovered:
                    label += " [OF]"

                draw_list.add_text(p1[0] + 3, p1[1] + 3, imgui.get_color_u32_rgba(*VideoDisplayColors.BOX_LABEL), label)

        if is_occluded:
            draw_list.add_text(img_rect['min_x'] + 10, img_rect['max_y'] - 30, imgui.get_color_u32_rgba(*VideoDisplayColors.OCCLUSION_WARNING), "OCCLUSION (FALLBACK)")

        motion_mode = frame_overlay_data.get("motion_mode")
        is_vr_video = self.app.processor and self.app.processor.determined_video_type == 'VR'

        if motion_mode and is_vr_video:
            roi_to_use = None
            locked_penis_box = next((b for b in frame_overlay_data.get("yolo_boxes", []) if b.get("class_name") == "locked_penis"), None)
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
        group_height = (button_h_ref * num_control_lines) + (style.item_spacing[1] * (num_control_lines - 1 if num_control_lines > 1 else 0))
        overlay_ctrl_y = img_rect['min_y'] - group_height - (style.item_spacing[1] * 2)
        overlay_ctrl_x = img_rect['min_x'] + style.item_spacing[1]
        overlay_ctrl_y = max(img_rect['min_y'] + style.item_spacing[1], overlay_ctrl_y)
        overlay_ctrl_x = max(img_rect['min_x'] + style.item_spacing[1], overlay_ctrl_x)
        imgui.set_cursor_screen_pos((overlay_ctrl_x, overlay_ctrl_y))

        imgui.begin_group()

        # Zoom Settings Block (Z-In, Z-Out, Rst, Text on one line)
        if imgui.button("Z-In##VidOverZoomIn"):
            app_state.adjust_video_zoom(1.2)
        imgui.same_line(spacing=4)
        if imgui.button("Z-Out##VidOverZoomOut"):
            app_state.adjust_video_zoom(1 / 1.2)
        imgui.same_line(spacing=4)
        if imgui.button("Rst##VidOverZoomReset"):
            app_state.reset_video_zoom_pan()
        imgui.same_line(spacing=4)
        imgui.text(f"{app_state.video_zoom_factor:.1f}x")

        if pan_buttons_active:
            # Pan Arrows Block (Left, Right, Up, Down on one line)
            if imgui.arrow_button("##VidOverPanLeft", imgui.DIRECTION_LEFT):
                app_state.pan_video_normalized_delta(-app_state.video_pan_step, 0)
            imgui.same_line(spacing=4)
            if imgui.arrow_button("##VidOverPanRight", imgui.DIRECTION_RIGHT):
                app_state.pan_video_normalized_delta(app_state.video_pan_step, 0)
            imgui.same_line(spacing=4)
            if imgui.arrow_button("##VidOverPanUp", imgui.DIRECTION_UP):
                app_state.pan_video_normalized_delta(0, -app_state.video_pan_step)
            imgui.same_line(spacing=4)
            if imgui.arrow_button("##VidOverPanDown", imgui.DIRECTION_DOWN):
                app_state.pan_video_normalized_delta(0, app_state.video_pan_step)

        imgui.end_group()

    def _render_drop_video_prompt(self):
        cursor_start_pos = imgui.get_cursor_pos()
        win_size = imgui.get_window_size()
        text_to_display = "Drag and drop one or more video files here."
        text_size = imgui.calc_text_size(text_to_display)
        if win_size[0] > text_size[0] and win_size[1] > text_size[1]:  # Check if window is large enough for text
            imgui.set_cursor_pos(((win_size[0] - text_size[0]) * 0.5 + cursor_start_pos[0], (win_size[1] - text_size[1]) * 0.5 + cursor_start_pos[1]))
        imgui.text(text_to_display)
