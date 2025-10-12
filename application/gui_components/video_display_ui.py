import imgui
from typing import Optional, Tuple

import config.constants as constants
from config.element_group_colors import VideoDisplayColors
from application.utils import get_logo_texture_manager, get_icon_texture_manager


class VideoDisplayUI:
    def __init__(self, app, gui_instance):
        self.app = app
        self.gui_instance = gui_instance
        self._video_display_rect_min = (0, 0)
        self._video_display_rect_max = (0, 0)
        self._actual_video_image_rect_on_screen = {'min_x': 0, 'min_y': 0, 'max_x': 0, 'max_y': 0, 'w': 0, 'h': 0}
        
        # PERFORMANCE OPTIMIZATIONS: Video display caching and smart rendering
        self._last_frame_texture_id = None  # Track texture changes
        self._cached_overlay_data = None  # Cache overlay rendering data
        self._overlay_dirty = True  # Flag for overlay re-rendering
        self._last_overlay_hash = None  # Detect overlay changes
        self._render_quality_mode = "auto"  # auto/high/medium/low
        self._frame_skip_counter = 0  # Skip expensive operations during load

        # Video texture update optimization (dirty flag)
        self._last_uploaded_frame_index = None  # Track which frame is currently in GPU texture
        self._texture_update_count = 0  # Count actual texture updates
        self._texture_skip_count = 0  # Count skipped updates (cache hits)
        self._last_perf_log_time = 0  # For periodic performance logging

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
        
        # Handy device control state
        self.handy_streaming_active = False
        self.handy_preparing = False
        self.handy_last_funscript_path = None
        self.saved_processing_speed_mode = None  # Store original speed mode when Handy starts
        
        # Fullscreen process state
        self._fullscreen_process = None

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

        # Get icon texture manager for playback controls
        icon_mgr = get_icon_texture_manager()

        # Button sizing
        button_h_ref = imgui.get_frame_height()
        pb_icon_w, pb_play_w, pb_stop_w, pb_btn_spacing = button_h_ref, button_h_ref, button_h_ref, 4.0

        total_controls_width = (pb_icon_w * 7) + (pb_btn_spacing * 6)

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

        # Jump Start button
        jump_start_tex, _, _ = icon_mgr.get_icon_texture('jump-start.png')
        if jump_start_tex and imgui.image_button(jump_start_tex, pb_icon_w, pb_icon_w):
            event_handlers.handle_playback_control("jump_start")
        elif not jump_start_tex and imgui.button("|<##VidOverStart", width=pb_icon_w):
            event_handlers.handle_playback_control("jump_start")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Jump to Start (HOME)")

        imgui.same_line(spacing=pb_btn_spacing)

        # Previous Frame button
        prev_frame_tex, _, _ = icon_mgr.get_icon_texture('prev-frame.png')
        if prev_frame_tex and imgui.image_button(prev_frame_tex, pb_icon_w, pb_icon_w):
            event_handlers.handle_playback_control("prev_frame")
        elif not prev_frame_tex and imgui.button("<<##VidOverPrev", width=pb_icon_w):
            event_handlers.handle_playback_control("prev_frame")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Previous Frame (LEFT ARROW)")

        imgui.same_line(spacing=pb_btn_spacing)

        # Play/Pause button (dynamic based on state)
        is_playing = self.app.processor and self.app.processor.is_processing and not self.app.processor.pause_event.is_set()
        play_pause_icon_name = 'pause.png' if is_playing else 'play.png'
        play_pause_fallback = "||" if is_playing else ">"

        play_pause_tex, _, _ = icon_mgr.get_icon_texture(play_pause_icon_name)
        if play_pause_tex and imgui.image_button(play_pause_tex, pb_play_w, pb_icon_w):
            event_handlers.handle_playback_control("play_pause")
        elif not play_pause_tex and imgui.button(f"{play_pause_fallback}##VidOverPlayPause", width=pb_play_w):
            event_handlers.handle_playback_control("play_pause")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Toggle Play/Pause (SPACE)")

        imgui.same_line(spacing=2)

        # Stop button
        stop_tex, _, _ = icon_mgr.get_icon_texture('stop.png')
        if stop_tex and imgui.image_button(stop_tex, pb_stop_w, pb_icon_w):
            event_handlers.handle_playback_control("stop")
        elif not stop_tex and imgui.button("[]##VidOverStop", width=pb_stop_w):
            event_handlers.handle_playback_control("stop")

        imgui.same_line(spacing=pb_btn_spacing)

        # Next Frame button
        next_frame_tex, _, _ = icon_mgr.get_icon_texture('next-frame.png')
        if next_frame_tex and imgui.image_button(next_frame_tex, pb_icon_w, pb_icon_w):
            event_handlers.handle_playback_control("next_frame")
        elif not next_frame_tex and imgui.button(">>##VidOverNext", width=pb_icon_w):
            event_handlers.handle_playback_control("next_frame")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Next Frame (RIGHT ARROW)")

        imgui.same_line(spacing=pb_btn_spacing)

        # Jump End button
        jump_end_tex, _, _ = icon_mgr.get_icon_texture('jump-end.png')
        if jump_end_tex and imgui.image_button(jump_end_tex, pb_icon_w, pb_icon_w):
            event_handlers.handle_playback_control("jump_end")
        elif not jump_end_tex and imgui.button(">|##VidOverEnd", width=pb_icon_w):
            event_handlers.handle_playback_control("jump_end")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Jump to End (END)")

        # Add Handy control button inline with playback controls
        self._render_handy_control_button_inline(pb_btn_spacing, button_h_ref, controls_disabled)
        
        # Add Fullscreen button inline with playback controls
        self._render_fullscreen_button_inline(pb_btn_spacing, button_h_ref, controls_disabled)
        
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
                current_frame_index = getattr(self.app.processor, 'current_frame_index', None)

                # PERFORMANCE: Check if frame changed before copying/uploading to GPU
                frame_changed = (current_frame_index != self._last_uploaded_frame_index)

                if self.app.processor and self.app.processor.current_frame is not None:
                    with self.app.processor.frame_lock:
                        if self.app.processor.current_frame is not None:
                            # Check if current_frame is actually a frame (numpy array) and not just a frame number (int)
                            if hasattr(self.app.processor.current_frame, 'copy'):
                                # Only copy frame if it changed
                                if frame_changed:
                                    current_frame_for_texture = self.app.processor.current_frame.copy()
                                else:
                                    # Frame hasn't changed - skip expensive copy
                                    self._texture_skip_count += 1
                            # else: current_frame is just an int (frame number), no image to display

                video_frame_available = current_frame_for_texture is not None or (not frame_changed and self._last_uploaded_frame_index is not None)

                # Upload new frame to GPU if we copied one
                if current_frame_for_texture is not None:
                    self.gui_instance.update_texture(self.gui_instance.frame_texture_id, current_frame_for_texture)
                    self._last_uploaded_frame_index = current_frame_index
                    self._texture_update_count += 1

                # Render video (either new frame or reuse existing texture)
                if video_frame_available:
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

                                # Keep the just-drawn ROI visible while waiting for the user to click the point
                                if self.waiting_for_point_click and self.drawn_user_roi_video_coords:
                                    img_rect = self._actual_video_image_rect_on_screen
                                    draw_list.push_clip_rect(img_rect['min_x'], img_rect['min_y'], img_rect['max_x'], img_rect['max_y'], True)
                                    rx_vid, ry_vid, rw_vid, rh_vid = self.drawn_user_roi_video_coords
                                    roi_start_screen = self._video_to_screen_coords(rx_vid, ry_vid)
                                    roi_end_screen = self._video_to_screen_coords(rx_vid + rw_vid, ry_vid + rh_vid)
                                    if roi_start_screen and roi_end_screen:
                                        draw_list.add_rect(
                                            roi_start_screen[0], roi_start_screen[1],
                                            roi_end_screen[0], roi_end_screen[1],
                                            imgui.get_color_u32_rgba(*VideoDisplayColors.ROI_BORDER),
                                            thickness=2
                                        )
                                    draw_list.pop_clip_rect()

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
                                        # Only warn on conversion failure during mouse release, handled above.

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

                            # Visualization of active Oscillation Area (ROI outline)
                            # Rule: If ROI toggle is ON => always show. If ROI toggle is OFF => show only when not actively tracking (paused/stopped).
                            if self.app.tracker and self.app.tracker.oscillation_area_fixed is not None and not self.app.is_setting_oscillation_area_mode:
                                tracker = self.app.tracker
                                proc = getattr(self.app, 'processor', None)
                                is_paused = bool(proc and hasattr(proc, 'pause_event') and proc.pause_event.is_set())
                                is_actively_tracking = bool(getattr(tracker, 'tracking_active', False)) and not is_paused
                                show_toggle_on = bool(getattr(tracker, 'show_roi', True))
                                allow_outline = show_toggle_on or (not show_toggle_on and not is_actively_tracking)
                                if allow_outline:
                                    draw_list = imgui.get_window_draw_list()
                                    ax_vid, ay_vid, aw_vid, ah_vid = tracker.oscillation_area_fixed
                                    area_start_screen = self._video_to_screen_coords(ax_vid, ay_vid)
                                    area_end_screen = self._video_to_screen_coords(ax_vid + aw_vid, ay_vid + ah_vid)
                                    if area_start_screen and area_end_screen:
                                        draw_list.add_rect(area_start_screen[0], area_start_screen[1], area_end_screen[0], area_end_screen[1], imgui.get_color_u32_rgba(0, 128, 255, 255), thickness=2)
                                        draw_list.add_text(area_start_screen[0], area_start_screen[1] - 15, imgui.get_color_u32_rgba(0, 255, 255, 255), "Oscillation Area")

                                    # Do not draw grid blocks in overlay

                                # Do not draw the block grid outline here. Grid visualization is handled in-frame or elsewhere.

                            # Legacy User Fixed ROI visualization removed - ModularTrackerBridge doesn't use this mode
                            self._handle_video_mouse_interaction(app_state)

                            if app_state.show_stage2_overlay and stage_proc.stage2_overlay_data_map and self.app.processor and \
                                    self.app.processor.current_frame_index >= 0:
                                self._render_stage2_overlay(stage_proc, app_state)

                            # Mixed mode debug overlay (shows when in mixed mode and debug data is available)
                            if (app_state.selected_tracker_name and "mixed" in app_state.selected_tracker_name.lower() and 
                                ((hasattr(self.app, 'stage3_mixed_debug_frame_map') and self.app.stage3_mixed_debug_frame_map) or 
                                 (hasattr(self.app, 'mixed_stage_processor') and self.app.mixed_stage_processor))):
                                draw_list = imgui.get_window_draw_list()
                                img_rect = self._actual_video_image_rect_on_screen
                                draw_list.push_clip_rect(img_rect['min_x'], img_rect['min_y'], img_rect['max_x'], img_rect['max_y'], True)
                                self._render_mixed_mode_debug_overlay(draw_list)
                                draw_list.pop_clip_rect()

                            # Only show live tracker info if the Stage 2 overlay isn't active
                            if self.app.tracker and self.app.tracker.tracking_active and not (app_state.show_stage2_overlay and stage_proc.stage2_overlay_data_map):
                                draw_list = imgui.get_window_draw_list()
                                img_rect = self._actual_video_image_rect_on_screen
                                # Clip rendering to the video display area
                                draw_list.push_clip_rect(img_rect['min_x'], img_rect['min_y'], img_rect['max_x'], img_rect['max_y'], True)
                                self._render_live_tracker_overlay(draw_list)
                                draw_list.pop_clip_rect()

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
                            btn_labels = ["Gauge T1", "Gauge T2", "3D Simulator"]
                            btn_attrs = ["show_gauge_window_timeline1", "show_gauge_window_timeline2", "show_simulator_3d"]
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

                            # --- Render Component Overlays (if enabled) ---
                            self._render_component_overlays(app_state)

                            # --- Render Playback Controls (AFTER all overlays so they're clickable) ---
                            # Note: draw_list operations render on top of ImGui widgets, so buttons must be last
                            self._render_playback_controls_overlay()
                            self._render_video_zoom_pan_controls(app_state)

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
                        else:
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

        # Get icon texture manager for zoom controls
        icon_mgr = get_icon_texture_manager()
        zoom_btn_size = button_h_ref

        # Zoom In button
        zoom_in_tex, _, _ = icon_mgr.get_icon_texture('zoom-in.png')
        if zoom_in_tex and imgui.image_button(zoom_in_tex, zoom_btn_size, zoom_btn_size):
            app_state.adjust_video_zoom(1.2)
        elif not zoom_in_tex and imgui.button("Z-In##VidOverZoomIn"):
            app_state.adjust_video_zoom(1.2)

        imgui.same_line(spacing=4)

        # Zoom Out button
        zoom_out_tex, _, _ = icon_mgr.get_icon_texture('zoom-out.png')
        if zoom_out_tex and imgui.image_button(zoom_out_tex, zoom_btn_size, zoom_btn_size):
            app_state.adjust_video_zoom(1 / 1.2)
        elif not zoom_out_tex and imgui.button("Z-Out##VidOverZoomOut"):
            app_state.adjust_video_zoom(1 / 1.2)

        imgui.same_line(spacing=4)

        # Reset button (counterclockwise arrow icon)
        reset_tex, _, _ = icon_mgr.get_icon_texture('reset.png')
        if reset_tex and imgui.image_button(reset_tex, zoom_btn_size, zoom_btn_size):
            app_state.reset_video_zoom_pan()
        elif not reset_tex and imgui.button("Rst##VidOverZoomReset"):
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
        """Render logo and drop prompt when no video is loaded."""
        cursor_start_pos = imgui.get_cursor_pos()
        win_size = imgui.get_window_size()

        # Load logo texture
        logo_manager = get_logo_texture_manager()
        logo_texture = logo_manager.get_texture_id()
        logo_width, logo_height = logo_manager.get_dimensions()

        # Calculate sizes and positions for centered layout
        text_to_display = "Drag and drop one or more video files here."
        text_size = imgui.calc_text_size(text_to_display)

        if logo_texture and logo_width > 0 and logo_height > 0:
            # Scale logo to reasonable size (max 200px while maintaining aspect ratio)
            max_logo_size = 200
            if logo_width > logo_height:
                display_logo_w = min(logo_width, max_logo_size)
                display_logo_h = int(logo_height * (display_logo_w / logo_width))
            else:
                display_logo_h = min(logo_height, max_logo_size)
                display_logo_w = int(logo_width * (display_logo_h / logo_height))

            # Calculate total height (logo + spacing + text)
            spacing = 20
            total_height = display_logo_h + spacing + text_size[1]

            # Center vertically
            start_y = (win_size[1] - total_height) * 0.5 + cursor_start_pos[1]

            # Draw logo centered horizontally
            logo_x = (win_size[0] - display_logo_w) * 0.5 + cursor_start_pos[0]
            imgui.set_cursor_pos((logo_x, start_y))

            # Draw logo with slight transparency
            imgui.image(logo_texture, display_logo_w, display_logo_h, tint_color=(1.0, 1.0, 1.0, 0.6))

            # Draw text below logo
            text_y = start_y + display_logo_h + spacing
            text_x = (win_size[0] - text_size[0]) * 0.5 + cursor_start_pos[0]
            imgui.set_cursor_pos((text_x, text_y))
            imgui.text_colored(text_to_display, 0.7, 0.7, 0.7, 1.0)  # Slightly dimmed text
        else:
            # Fallback to text-only if logo fails to load
            if win_size[0] > text_size[0] and win_size[1] > text_size[1]:
                imgui.set_cursor_pos(((win_size[0] - text_size[0]) * 0.5 + cursor_start_pos[0], (win_size[1] - text_size[1]) * 0.5 + cursor_start_pos[1]))
            imgui.text(text_to_display)

    def _render_mixed_mode_debug_overlay(self, draw_list):
        """
        Render debug overlay for Mixed Stage 3 mode.
        Shows current processing state, ROI info, and signal source.
        """
        debug_info = None
        
        # Check if we have debug data loaded from msgpack (during video playback)
        if (hasattr(self.app, 'stage3_mixed_debug_frame_map') and 
            self.app.stage3_mixed_debug_frame_map and
            self.app.processor and hasattr(self.app.processor, 'current_frame_index')):
            
            current_frame = self.app.processor.current_frame_index
            debug_info = self.app.stage3_mixed_debug_frame_map.get(current_frame, {})
            
        # Fallback to live processor debug info (during processing)
        elif (hasattr(self.app, 'mixed_stage_processor') and self.app.mixed_stage_processor):
            debug_info = self.app.mixed_stage_processor.get_debug_info()
        
        if not debug_info:
            return
        
        # Position overlay text in top-left corner of video area
        img_rect = self._actual_video_image_rect_on_screen
        overlay_x = img_rect['min_x'] + 10
        overlay_y = img_rect['min_y'] + 10
        
        # Background for text
        text_bg_color = (0, 0, 0, 180)  # Semi-transparent black
        text_color = (255, 255, 255, 255)  # White text
        
        # Build debug text
        debug_lines = [
            f"Mixed Stage 3 Debug",
            f"Chapter: {debug_info.get('current_chapter_type', 'Unknown')}",
            f"Signal: {debug_info.get('signal_source', 'Unknown')}",
            f"Live Tracker: {'Active' if debug_info.get('live_tracker_active', False) else 'Inactive'}",
        ]
        
        # Add ROI info if available
        roi = debug_info.get('current_roi')
        if roi:
            debug_lines.append(f"ROI: ({roi[0]}, {roi[1]}) - ({roi[2]}, {roi[3]})")
            # Add ROI update info
            roi_updated = debug_info.get('roi_updated', False)
            roi_counter = debug_info.get('roi_update_counter', 0)
            debug_lines.append(f"ROI Updated: {roi_updated} (Frame #{roi_counter})")
        
        # Add oscillation details if live tracker is active
        if debug_info.get('live_tracker_active', False):
            intensity = debug_info.get('oscillation_intensity', 0.0)
            debug_lines.append(f"Oscillation: {intensity:.2f}")
            
            # Add oscillation position if available
            osc_pos = debug_info.get('oscillation_pos')
            if osc_pos is not None:
                debug_lines.append(f"Osc Pos: {osc_pos}/100")
            
            # Add EMA alpha setting
            ema_alpha = debug_info.get('ema_alpha')
            if ema_alpha is not None:
                debug_lines.append(f"EMA Alpha: {ema_alpha:.2f}")
            
            # Add last known position for debugging smoothing
            last_known = debug_info.get('oscillation_last_known')
            if last_known is not None:
                debug_lines.append(f"Last Known: {last_known:.1f}")
        
        # Add frame ID for debugging
        frame_id = debug_info.get('frame_id')
        if frame_id is not None:
            debug_lines.append(f"Frame: {frame_id}")
        
        # Render each line
        line_height = 16
        for i, line in enumerate(debug_lines):
            text_y = overlay_y + (i * line_height)
            
            # Calculate text size for background
            text_size = imgui.calc_text_size(line)
            
            # Draw background rectangle
            draw_list.add_rect_filled(
                overlay_x - 5, text_y - 2,
                overlay_x + text_size.x + 5, text_y + text_size.y + 2,
                imgui.get_color_u32_rgba(*text_bg_color)
            )
            
            # Draw text
            draw_list.add_text(
                overlay_x, text_y,
                imgui.get_color_u32_rgba(*text_color),
                line
            )
        
        # Render ROI box if available
        roi = debug_info.get('current_roi')
        if roi:
            p1 = self._video_to_screen_coords(roi[0], roi[1])
            p2 = self._video_to_screen_coords(roi[2], roi[3])
            
            if p1 and p2:
                # ROI box color based on chapter type
                chapter_type = debug_info.get('current_chapter_type', 'Other')
                if chapter_type in ['BJ', 'HJ']:
                    roi_color = (0, 255, 0, 255)  # Green for BJ/HJ (ROI tracking active)
                else:
                    roi_color = (255, 255, 0, 255)  # Yellow for other (Stage 2 signal)
                
                # Draw ROI rectangle
                draw_list.add_rect(
                    p1[0], p1[1], p2[0], p2[1],
                    imgui.get_color_u32_rgba(*roi_color),
                    thickness=2.0
                )
                
                # Add ROI label
                roi_label = f"ROI ({chapter_type})"
                draw_list.add_text(
                    p1[0], p1[1] - 20,
                    imgui.get_color_u32_rgba(*roi_color),
                    roi_label
                )
    
    def _render_handy_control_button_inline(self, spacing: float, button_height: float, controls_disabled: bool):
        """Render Handy device control button inline with playback controls."""
        # Check if Handy devices are available
        if not self._is_handy_available():
            return
            
        style = imgui.get_style()
        button_width = button_height * 2.8  # Smaller width for inline display
        
        # Add spacing and render inline with other controls
        imgui.same_line(spacing=spacing)
        
        # Determine button state and text
        if self.handy_preparing:
            # Show preparing state
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.8, 0.8, 0.2, 1.0)  # Yellow
            button_text = "Preparing"  # Loading text
            button_enabled = False
        elif self.handy_streaming_active:
            # Show stop streaming button
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.8, 0.2, 0.2, 1.0)  # Red
            button_text = "Stop"  # Stop text
            button_enabled = True
        else:
            # Show start streaming button
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.2, 0.8, 0.2, 1.0)  # Green
            button_text = "Handy"  # Handy button text
            button_enabled = True
            
        # Disable button if no funscript actions available
        has_funscript = self._has_funscript_actions()
        if not has_funscript and not self.handy_streaming_active:
            button_enabled = False
            button_text = "No Funscript"  # No funscript available
            
        # Apply disabled styling if controls are disabled or button is disabled
        if controls_disabled or not button_enabled:
            if not controls_disabled:  # Only add if not already applied
                imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
                imgui.push_style_var(imgui.STYLE_ALPHA, style.alpha * 0.5)
            
        # Render the button
        button_clicked = imgui.button(f"{button_text}##HandyControl", width=button_width)
        
        if button_clicked:
            print(f"DEBUG: Handy button clicked - enabled: {button_enabled}, controls_disabled: {controls_disabled}")
            if button_enabled and not controls_disabled:
                print(f"DEBUG: Button action triggered - streaming_active: {self.handy_streaming_active}")
                if self.handy_streaming_active:
                    print("DEBUG: Stopping Handy streaming")
                    self._stop_handy_streaming()
                else:
                    print("DEBUG: Starting Handy streaming")
                    self._start_handy_streaming()
            else:
                print("DEBUG: Button click ignored - button disabled or controls disabled")
                
        # Clean up disabled styling if we added it
        if not controls_disabled and not button_enabled:
            imgui.pop_style_var()
            imgui.internal.pop_item_flag()
            
        imgui.pop_style_color()
        
        # Show tooltip with additional info
        if imgui.is_item_hovered():
            imgui.begin_tooltip()
            if self.handy_preparing:
                imgui.text("Uploading funscript and setting up HSSP streaming...")
            elif self.handy_streaming_active:
                imgui.text("Handy streaming active. Click to stop.")
            elif not has_funscript:
                imgui.text("No funscript actions available. Create a timeline first.")
            else:
                imgui.text("Start Handy streaming with current funscript")
                imgui.text("Will upload to Handy servers and sync with video position")
            imgui.end_tooltip()
    
    def _render_fullscreen_button_inline(self, spacing: float, button_height: float, controls_disabled: bool):
        """Render fullscreen button inline with playback controls."""
        style = imgui.get_style()
        button_width = button_height  # Square button for consistency with other playback controls
        
        # Add spacing and render inline with other controls
        imgui.same_line(spacing=spacing)
        
        # Check if live tracking is active (when FS button should be enabled)
        is_live_tracking = (self.app.processor and
                           self.app.processor.is_processing and
                           self.app.processor.enable_tracker_processing)
        
        # If parent controls are disabled due to live tracking, temporarily enable just for FS button
        parent_disabled_override = False
        if controls_disabled and is_live_tracking:
            # Pop the parent disabled state temporarily
            imgui.internal.pop_item_flag()
            imgui.pop_style_var()
            parent_disabled_override = True
        
        # Determine if video is loaded and ready for fullscreen
        video_loaded = (hasattr(self.app, 'processor') and 
                       self.app.processor and 
                       hasattr(self.app.processor, 'video_path') and 
                       self.app.processor.video_path)
        
        # Check if fullscreen is currently active
        has_fullscreen_process = (hasattr(self, '_fullscreen_process') and 
                                 self._fullscreen_process and 
                                 self._fullscreen_process.poll() is None)
        
        # live tracking status already checked above
        
        # Fullscreen button should be enabled during live tracking!
        # That's exactly when users want to see fullscreen with audio
        # Override the parent controls_disabled for fullscreen specifically
        fullscreen_button_disabled = not video_loaded
        
        # Important: Ignore parent controls_disabled during live tracking
        # The fullscreen feature is most useful during live tracking!
        if is_live_tracking and video_loaded:
            fullscreen_button_disabled = False
        
        # Get icon texture manager
        icon_mgr = get_icon_texture_manager()

        # Determine button state, styling, and icon
        if has_fullscreen_process:
            # Exit fullscreen mode
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.8, 0.2, 0.2, 1.0)  # Red
            button_icon_name = 'fullscreen-exit.png'
            button_text_fallback = "Exit FS"
            button_enabled = True
        else:
            # Enter fullscreen mode - highlight during live tracking
            if is_live_tracking:
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.2, 0.8, 0.2, 1.0)  # Green (live tracking)
                button_text_fallback = "FS Live"
            else:
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.2, 0.6, 0.8, 1.0)  # Blue (normal)
                button_text_fallback = "FS"
            button_icon_name = 'fullscreen.png'
            button_enabled = video_loaded

        # Apply disabled styling if needed (but NOT during live tracking)
        if fullscreen_button_disabled:
            imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
            imgui.push_style_var(imgui.STYLE_ALPHA, style.alpha * 0.5)

        # Render the button with icon or fallback text
        fs_tex, _, _ = icon_mgr.get_icon_texture(button_icon_name)
        if fs_tex:
            button_clicked = imgui.image_button(fs_tex, button_width, button_width)
        else:
            button_clicked = imgui.button(f"{button_text_fallback}##FullscreenControl", width=button_width)
        
        # Use our fullscreen-specific logic, not parent controls_disabled
        if button_clicked and not fullscreen_button_disabled:
            self._handle_fullscreen_button_click()
        
        # Clean up disabled styling if we added it
        if fullscreen_button_disabled:
            imgui.pop_style_var()
            imgui.internal.pop_item_flag()
        
        imgui.pop_style_color()
        
        # Show tooltip
        if imgui.is_item_hovered():
            imgui.begin_tooltip()
            if not video_loaded:
                imgui.text("No video loaded")
            elif has_fullscreen_process:
                imgui.text("Exit fullscreen mode")
                imgui.text("Press Q in fullscreen window to quit")
            elif is_live_tracking:
                imgui.text("Fullscreen during live tracking")
                imgui.text("Perfect sync + audio + controls")
            else:
                imgui.text("Enter fullscreen mode")
                imgui.text("High-quality display with audio")
            imgui.end_tooltip()
        
        # Restore parent disabled state if we overrode it
        if parent_disabled_override:
            # Re-apply parent disabled styling for subsequent buttons
            imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
            imgui.push_style_var(imgui.STYLE_ALPHA, style.alpha * 0.5)
    
    def _handle_fullscreen_button_click(self):
        """Handle fullscreen button click - use simple synchronized ffplay approach."""
        try:
            # Check if we already have a fullscreen process running
            if hasattr(self, '_fullscreen_process') and self._fullscreen_process:
                if self._fullscreen_process.poll() is None:  # Still running
                    # Stop existing fullscreen
                    self._fullscreen_process.terminate()
                    self._fullscreen_process = None
                    if hasattr(self.app, 'logger'):
                        self.app.logger.info("Fullscreen mode disabled")
                    return
            
            # Start new fullscreen with synchronized ffplay
            if not hasattr(self.app.processor, 'video_path') or not self.app.processor.video_path:
                if hasattr(self.app, 'logger'):
                    self.app.logger.warning("No video loaded for fullscreen")
                return
            
            # Get current timestamp for perfect sync
            current_time = self._get_current_playback_time()
            
            # Build ffplay command with high quality and audio
            ffplay_cmd = [
                'ffplay',
                '-fs',                              # Fullscreen
                '-autoexit',                        # Auto-exit when done
                '-ss', str(current_time),           # Start at current time for sync
                '-vf', self._get_fullscreen_video_filter(),  # High quality filter
                self.app.processor.video_path       # Video file
            ]
            
            # Start synchronized ffplay process
            import subprocess
            import sys
            creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            
            self._fullscreen_process = subprocess.Popen(
                ffplay_cmd,
                stderr=subprocess.DEVNULL,  # Hide ffplay output
                creationflags=creation_flags
            )
            
            if hasattr(self.app, 'logger'):
                self.app.logger.info(f"✅ Fullscreen started with ffplay (sync time: {current_time:.2f}s)")
                self.app.logger.info("🎮 Controls: Space=pause, ←→=seek±10s, ↑↓=seek±1min, Q=quit")
                self.app.logger.info("🔊 Audio: Full quality audio included")
                    
        except Exception as e:
            if hasattr(self.app, 'logger'):
                self.app.logger.error(f"Error starting fullscreen: {e}")
            import traceback
            if hasattr(self.app, 'logger'):
                self.app.logger.error(traceback.format_exc())
    
    def _initialize_dual_frame_processing(self):
        """Initialize dual-frame processing for fullscreen display."""
        try:
            # Initialize dual-frame processor if needed
            if not hasattr(self.app.processor, 'dual_frame_processor'):
                from video.dual_frame_processor import DualFrameProcessor
                self.app.processor.dual_frame_processor = DualFrameProcessor(self.app.processor)
            
            # Enable dual-frame mode
            if not self.app.processor.dual_frame_processor.dual_frame_enabled:
                self.app.processor.dual_frame_processor.enable_dual_frame_mode()
                
        except Exception as e:
            if hasattr(self.app, 'logger'):
                self.app.logger.error(f"Error initializing dual-frame processing: {e}")
    
    def _get_current_fullscreen_frame(self) -> Optional[object]:
        """Get current frame for fullscreen display."""
        # This would get the latest fullscreen frame from dual-frame processor
        if (hasattr(self.app.processor, 'dual_frame_processor') and
            self.app.processor.dual_frame_processor.dual_frame_enabled):
            return self.app.processor.dual_frame_processor.get_latest_fullscreen_frame()
        return None
    
    def _handle_fullscreen_play_pause(self):
        """Handle play/pause from fullscreen controls."""
        if hasattr(self.app.processor, 'is_processing'):
            if self.app.processor.is_processing:
                self.app.processor.pause_processing()
            else:
                self.app.processor.resume_processing()
    
    def _handle_fullscreen_stop(self):
        """Handle stop from fullscreen controls."""
        if hasattr(self.app.processor, 'stop_processing'):
            self.app.processor.stop_processing()
    
    def _handle_fullscreen_exit(self):
        """Handle fullscreen exit."""
        if hasattr(self.app, 'fullscreen_manager'):
            self.app.fullscreen_manager.stop_fullscreen_display()
    
    def _get_current_playback_time(self) -> float:
        """Get current playback time in seconds for synchronization."""
        try:
            # Try to get current time from processor
            if hasattr(self.app.processor, 'get_current_time'):
                return self.app.processor.get_current_time()
            
            # Fallback: calculate from current frame index
            if (hasattr(self.app.processor, 'current_frame_index') and 
                hasattr(self.app.processor, 'video_info') and
                self.app.processor.video_info):
                
                fps = self.app.processor.video_info.get('fps', 30.0)
                current_frame = self.app.processor.current_frame_index or 0
                return current_frame / fps
            
            # Default to start of video
            return 0.0
            
        except Exception as e:
            if hasattr(self.app, 'logger'):
                self.app.logger.warning(f"Could not get current playback time: {e}")
            return 0.0
    
    def _get_fullscreen_video_filter(self) -> str:
        """Get appropriate video filter for fullscreen display."""
        try:
            # For VR videos, use simple crop for left eye view (full resolution)
            if (hasattr(self.app.processor, 'is_vr_active_or_potential') and 
                self.app.processor.is_vr_active_or_potential()):
                return "crop=iw/2:ih:0:0"  # Left eye crop only, preserve full resolution
            
            # For regular videos, no scaling - preserve original resolution
            return "null"  # No filtering, full original resolution
            
        except Exception as e:
            if hasattr(self.app, 'logger'):
                self.app.logger.warning(f"Could not determine video filter: {e}")
            return "null"  # Safe default - no filtering
    
    def _is_handy_available(self):
        """Check if Handy devices are connected and device control is enabled."""
        try:
            # Check if device control is enabled
            if not hasattr(self.app, 'app_settings'):
                return False
                
            device_control_enabled = self.app.app_settings.get("device_control_video_playback", False)
            if not device_control_enabled:
                return False
                
            # Check if device manager exists and has Handy devices
            if not hasattr(self.app, 'device_manager') or not self.app.device_manager:
                return False
                
            device_manager = self.app.device_manager
            if not device_manager.is_connected():
                return False
                
            # Check for connected Handy devices
            for device_id, backend in device_manager.connected_devices.items():
                device_info = backend.get_device_info()
                if device_info and "handy" in device_info.name.lower():
                    return True
                    
            return False
            
        except Exception:
            return False
    
    def _has_funscript_actions(self):
        """Check if funscript actions are available for streaming."""
        try:
            # Check if funscript processor exists
            if not hasattr(self.app, 'funscript_processor') or not self.app.funscript_processor:
                return False
                
            fs_proc = self.app.funscript_processor
            
            # Get the DualAxisFunscript object
            funscript_obj = fs_proc.get_funscript_obj()
            if not funscript_obj:
                return False
                
            # Check primary axis actions only (Handy uses primary axis)
            primary_actions = fs_proc.get_actions('primary')
            return len(primary_actions) > 0 if primary_actions else False
            
        except Exception as e:
            return False
    
    def _start_handy_streaming(self):
        """Start Handy streaming with current funscript and video position."""
        print("DEBUG: _start_handy_streaming() called")
        
        # Force video to real-time speed for proper Handy synchronization
        if hasattr(self.app, 'app_state_ui') and hasattr(self.app.app_state_ui, 'selected_processing_speed_mode'):
            # Save current speed mode to restore later
            self.saved_processing_speed_mode = self.app.app_state_ui.selected_processing_speed_mode
            # Force to real-time speed
            self.app.app_state_ui.selected_processing_speed_mode = constants.ProcessingSpeedMode.REALTIME
            print(f"DEBUG: Forced video speed to REALTIME (was {self.saved_processing_speed_mode.value})")
        
        import threading
        import asyncio
        
        def start_streaming_async():
            print("DEBUG: start_streaming_async() thread started")
            loop = None
            try:
                # Set preparing state
                self.handy_preparing = True
                print("DEBUG: handy_preparing set to True")
                
                # Get current video position with multiple fallback methods
                current_time_ms = 0.0
                current_frame = 0
                fps = 0.0
                
                if hasattr(self.app, 'processor') and self.app.processor:
                    # Method 1: Direct frame index and FPS
                    if hasattr(self.app.processor, 'current_frame_index'):
                        current_frame = self.app.processor.current_frame_index
                        
                    if hasattr(self.app.processor, 'fps'):
                        fps = self.app.processor.fps
                        
                    # Method 2: Try video_info if available
                    if hasattr(self.app.processor, 'video_info') and self.app.processor.video_info:
                        if fps <= 0 and 'fps' in self.app.processor.video_info:
                            fps = self.app.processor.video_info['fps']
                            
                    # Method 3: Try get_current_frame_timestamp_ms if available
                    if hasattr(self.app.processor, 'get_current_frame_timestamp_ms'):
                        try:
                            timestamp_ms = self.app.processor.get_current_frame_timestamp_ms()
                            if timestamp_ms > 0:
                                current_time_ms = timestamp_ms
                        except:
                            pass
                    
                    # Calculate from frame and FPS if timestamp method didn't work
                    if current_time_ms == 0.0 and fps > 0:
                        current_time_ms = (current_frame / fps) * 1000.0
                
                print(f"DEBUG: Current video position: {current_time_ms}ms (frame {current_frame}, fps {fps})")
                
                # Extract funscript from current position onwards (your suggested approach!)
                # This creates a new funscript where the current video position becomes time 0
                print(f"DEBUG: Creating time-extracted funscript starting from {current_time_ms}ms")
                
                # Get funscript actions using the same method as detection
                print("DEBUG: Getting funscript actions")
                if not hasattr(self.app, 'funscript_processor') or not self.app.funscript_processor:
                    print("DEBUG: No funscript processor found")
                    self.handy_preparing = False
                    return
                    
                fs_proc = self.app.funscript_processor
                funscript_obj = fs_proc.get_funscript_obj()
                if not funscript_obj:
                    print("DEBUG: No funscript object found")
                    self.handy_preparing = False
                    return
                
                primary_actions = fs_proc.get_actions('primary')
                secondary_actions = fs_proc.get_actions('secondary')
                
                print(f"DEBUG: Retrieved {len(primary_actions)} primary actions, {len(secondary_actions)} secondary actions")
                
                if not primary_actions:
                    print("DEBUG: No primary actions available")
                    self.handy_preparing = False
                    return
                
                # Create and save temporary funscript file
                import tempfile
                import json
                import os
                
                print("DEBUG: Creating time-extracted funscript for Handy")
                
                # Extract actions from current video position onwards
                extracted_primary_actions = []
                for action in primary_actions:
                    action_time = action.get('at', 0)
                    if action_time >= current_time_ms:
                        # Adjust timestamp to start from 0 (current video position becomes time 0)
                        adjusted_action = {
                            'at': int(action_time - current_time_ms),  # Integer timestamps for Handy compatibility
                            'pos': int(action.get('pos', 0))  # Integer positions for Handy compatibility
                        }
                        extracted_primary_actions.append(adjusted_action)
                
                # Do the same for secondary actions if present
                extracted_secondary_actions = []
                if secondary_actions:
                    for action in secondary_actions:
                        action_time = action.get('at', 0)
                        if action_time >= current_time_ms:
                            adjusted_action = {
                                'at': int(action_time - current_time_ms),  # Integer timestamps for Handy compatibility
                                'pos': int(action.get('pos', 0))  # Integer positions for Handy compatibility
                            }
                            extracted_secondary_actions.append(adjusted_action)
                
                # Ensure funscript always starts at time=0 for HSSP compatibility
                if extracted_primary_actions and extracted_primary_actions[0]['at'] > 0:
                    # Interpolate the position at current_time_ms for time=0 baseline
                    baseline_pos = 50  # Default if no data
                    
                    # Find the actions before and after current_time_ms for interpolation
                    prev_action = None
                    next_action = None
                    
                    for i, action in enumerate(primary_actions):
                        action_time = action.get('at', 0)
                        if action_time <= current_time_ms:
                            prev_action = action
                        elif action_time > current_time_ms and next_action is None:
                            next_action = action
                            break
                    
                    # Interpolate position at current_time_ms
                    if prev_action and next_action:
                        # Linear interpolation between two actions
                        t1, p1 = prev_action['at'], prev_action['pos']
                        t2, p2 = next_action['at'], next_action['pos']
                        
                        # Calculate interpolation factor (0 to 1)
                        if t2 > t1:
                            factor = (current_time_ms - t1) / (t2 - t1)
                            baseline_pos = p1 + (p2 - p1) * factor
                        else:
                            baseline_pos = p1
                    elif prev_action:
                        # Use last known position if no next action
                        baseline_pos = prev_action.get('pos', 50)
                    elif next_action:
                        # Use next position if no previous action
                        baseline_pos = next_action.get('pos', 50)
                    
                    # Insert interpolated baseline action at time=0
                    baseline_action = {'at': 0, 'pos': int(baseline_pos)}
                    extracted_primary_actions.insert(0, baseline_action)
                    print(f"DEBUG: Added interpolated baseline action at time=0: {baseline_action}")
                
                # Ensure minimum of 2 actions for HSSP compatibility
                if len(extracted_primary_actions) < 2:
                    # Add a hold action at 1000ms with same position
                    if extracted_primary_actions:
                        last_pos = extracted_primary_actions[-1]['pos']
                    else:
                        last_pos = 50  # Default middle position
                        extracted_primary_actions.append({'at': 0, 'pos': last_pos})
                    
                    hold_action = {'at': 1000, 'pos': last_pos}
                    extracted_primary_actions.append(hold_action)
                    print(f"DEBUG: Added hold action for HSSP minimum requirement: {hold_action}")
                
                print(f"DEBUG: Extracted {len(extracted_primary_actions)} primary actions starting from time 0")
                print(f"DEBUG: Original video time {current_time_ms}ms now maps to funscript time 0ms")
                
                if not extracted_primary_actions:
                    print("DEBUG: No actions found after current video position - video may be at end")
                    self.handy_preparing = False
                    return
                
                # Show sample extracted actions for debugging
                if extracted_primary_actions:
                    print(f"DEBUG: First extracted action: {extracted_primary_actions[0]}")
                    print(f"DEBUG: Last extracted action: {extracted_primary_actions[-1]}")
                    
                    # Get extracted funscript duration
                    last_action_time = max(action.get('at', 0) for action in extracted_primary_actions)
                    first_action_time = min(action.get('at', 0) for action in extracted_primary_actions)
                    funscript_duration_ms = last_action_time - first_action_time
                    
                    print(f"DEBUG: First action: at={primary_actions[0].get('at')}ms, pos={primary_actions[0].get('pos')}")
                    if len(primary_actions) > 1:
                        print(f"DEBUG: Last action: at={last_action_time}ms")
                    print(f"DEBUG: Funscript duration: {funscript_duration_ms}ms ({funscript_duration_ms/1000:.1f}s)")
                    print(f"DEBUG: Start time: {current_time_ms}ms")
                    
                    # Check if start time is within funscript range
                    if current_time_ms > last_action_time:
                        print(f"WARNING: Start time ({current_time_ms}ms) is AFTER last action ({last_action_time}ms)")
                        print("This might cause HSSP play to fail - no actions to play")
                    elif current_time_ms < first_action_time:
                        print(f"WARNING: Start time ({current_time_ms}ms) is BEFORE first action ({first_action_time}ms)")
                    else:
                        print(f"INFO: Start time is within funscript range ({first_action_time}ms - {last_action_time}ms)")
                    
                    # Find actions around the current video position
                    nearby_actions = [a for a in primary_actions if abs(a.get('at', 0) - current_time_ms) < 5000]
                    print(f"DEBUG: Actions within 5s of current position ({current_time_ms}ms): {len(nearby_actions)}")
                    if nearby_actions:
                        for i, action in enumerate(nearby_actions[:3]):
                            print(f"DEBUG: Nearby action {i+1}: at={action.get('at')}ms, pos={action.get('pos')}")
                
                # Use extracted actions that start from time 0 (current video position)
                funscript_data = {
                    "actions": extracted_primary_actions,
                    "inverted": False,
                    "range": 90,
                    "version": "1.0"
                }
                
                # Save to temporary file
                temp_dir = tempfile.gettempdir()
                temp_filename = f"handy_stream_{int(current_time_ms)}.funscript"
                temp_path = os.path.join(temp_dir, temp_filename)
                
                print(f"DEBUG: Saving funscript to: {temp_path}")
                with open(temp_path, 'w') as f:
                    json.dump(funscript_data, f, indent=2)
                
                self.handy_last_funscript_path = temp_path
                print("DEBUG: Funscript file saved successfully")
                
                # Start async workflow
                print("DEBUG: Creating new event loop")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                device_manager = self.app.device_manager
                print(f"DEBUG: Device manager available: {device_manager is not None}")
                
                # Prepare Handy devices with extracted (time-shifted) actions
                print("DEBUG: Calling prepare_handy_for_video_playback with extracted actions")
                prepare_success = loop.run_until_complete(
                    device_manager.prepare_handy_for_video_playback(extracted_primary_actions, extracted_secondary_actions)
                )
                print(f"DEBUG: Prepare result: {prepare_success}")
                
                if not prepare_success:
                    print("DEBUG: Prepare failed, returning")
                    self.handy_preparing = False
                    return
                
                # Wait a moment for upload to complete
                print("DEBUG: Waiting 2 seconds for upload to complete")
                loop.run_until_complete(asyncio.sleep(2))
                
                # Start synchronized playback from time 0 (since our funscript now starts at 0)
                print(f"DEBUG: Starting video sync at 0ms (extracted funscript starts from current video position {current_time_ms}ms)")
                start_success = loop.run_until_complete(
                    device_manager.start_handy_video_sync(0.0)  # Always start from 0 with extracted funscript
                )
                print(f"DEBUG: Video sync start result: {start_success}")
                
                if start_success:
                    self.handy_streaming_active = True
                    print("DEBUG: Handy streaming activated successfully!")
                    
                    # Auto-start video playback for real-time sync
                    print("DEBUG: Auto-starting video playback for Handy sync")
                    try:
                        # Check if video is not currently playing
                        is_currently_playing = (self.app.processor and 
                                              self.app.processor.is_processing and 
                                              not self.app.processor.pause_event.is_set())
                        
                        if not is_currently_playing:
                            print("DEBUG: Video not playing, starting playback")
                            # Start video playback using the same method as the play button
                            if hasattr(self.app, 'event_handlers'):
                                self.app.event_handlers.handle_playback_control("play_pause")
                                print("DEBUG: Video playback started via event handler")
                            else:
                                print("DEBUG: No event handlers available")
                        else:
                            print("DEBUG: Video already playing")
                            
                    except Exception as playback_error:
                        print(f"DEBUG: Failed to auto-start video playback: {playback_error}")
                    
                    if hasattr(self.app, 'logger'):
                        self.app.logger.info(f"Handy streaming started at {current_time_ms:.1f}ms")
                else:
                    print("DEBUG: Video sync start failed")
                
                self.handy_preparing = False
                print("DEBUG: handy_preparing set to False, streaming setup complete")
                
            except Exception as e:
                print(f"DEBUG: Exception in streaming setup: {e}")
                import traceback
                traceback.print_exc()
                self.handy_preparing = False
                if hasattr(self.app, 'logger'):
                    self.app.logger.error(f"Failed to start Handy streaming: {e}")
            finally:
                if loop is not None:
                    print("DEBUG: Closing event loop")
                    loop.close()
                else:
                    print("DEBUG: No loop to close")
        
        # Start in background thread
        print("DEBUG: Creating background thread for Handy streaming")
        thread = threading.Thread(target=start_streaming_async, name="HandyStreamStart", daemon=True)
        thread.start()
        print("DEBUG: Background thread started")
    
    def _stop_handy_streaming(self):
        """Stop Handy streaming and clean up."""
        print("DEBUG: _stop_handy_streaming() called")
        
        # Restore original video speed mode
        if (self.saved_processing_speed_mode is not None and 
            hasattr(self.app, 'app_state_ui') and 
            hasattr(self.app.app_state_ui, 'selected_processing_speed_mode')):
            self.app.app_state_ui.selected_processing_speed_mode = self.saved_processing_speed_mode
            print(f"DEBUG: Restored video speed to {self.saved_processing_speed_mode.value}")
            self.saved_processing_speed_mode = None
        
        try:
            # Stop Handy device streaming
            if hasattr(self.app, 'device_manager') and self.app.device_manager:
                print("DEBUG: Stopping Handy device streaming")
                self.app.device_manager.stop_handy_streaming()
            else:
                print("DEBUG: No device manager available for stopping")
            
            # Stop video playback
            print("DEBUG: Stopping video playback")
            try:
                if hasattr(self.app, 'processor') and self.app.processor:
                    is_currently_playing = (self.app.processor.is_processing and 
                                          not self.app.processor.pause_event.is_set())
                    
                    if is_currently_playing:
                        print("DEBUG: Video is playing, pausing it")
                        if hasattr(self.app, 'event_handlers'):
                            self.app.event_handlers.handle_playback_control("play_pause")
                            print("DEBUG: Video playback paused via event handler")
                        else:
                            print("DEBUG: No event handlers available for stopping video")
                    else:
                        print("DEBUG: Video was not playing")
                else:
                    print("DEBUG: No video processor available")
                    
            except Exception as video_stop_error:
                print(f"DEBUG: Failed to stop video playback: {video_stop_error}")
            
            # Update streaming state
            self.handy_streaming_active = False
            print("DEBUG: handy_streaming_active set to False")
            
            # Clean up temporary funscript file
            if self.handy_last_funscript_path:
                try:
                    import os
                    if os.path.exists(self.handy_last_funscript_path):
                        os.remove(self.handy_last_funscript_path)
                        print(f"DEBUG: Removed temporary funscript file: {self.handy_last_funscript_path}")
                    else:
                        print(f"DEBUG: Temporary funscript file not found: {self.handy_last_funscript_path}")
                except Exception as cleanup_error:
                    print(f"DEBUG: Failed to clean up funscript file: {cleanup_error}")
                self.handy_last_funscript_path = None
            
            if hasattr(self.app, 'logger'):
                self.app.logger.info("Handy streaming stopped")
                
            print("DEBUG: Handy streaming stop complete")
                
        except Exception as e:
            if hasattr(self.app, 'logger'):
                self.app.logger.error(f"Error stopping Handy streaming: {e}")
    
    def _resync_handy_after_seek(self):
        """Resynchronize Handy device after video seek."""
        print("DEBUG: _resync_handy_after_seek() called")
        
        if not self.handy_streaming_active:
            print("DEBUG: Handy streaming not active, skipping resync")
            return
        
        try:
            # Stop current streaming first
            print("DEBUG: Stopping current Handy streaming for resync")
            if hasattr(self.app, 'device_manager') and self.app.device_manager:
                # Just stop the playback, not the entire streaming session
                import asyncio
                
                def stop_and_resync():
                    loop = None
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        # Stop current HSSP playback
                        print("DEBUG: Stopping HSSP playback for resync")
                        loop.run_until_complete(self.app.device_manager.stop_handy_playback())
                        
                        # Wait a brief moment for stop to complete
                        loop.run_until_complete(asyncio.sleep(0.5))
                        
                        # Restart with new position
                        print("DEBUG: Restarting Handy streaming from new position")
                        self._start_handy_streaming()
                        
                    except Exception as e:
                        print(f"DEBUG: Error during resync: {e}")
                        if hasattr(self.app, 'logger'):
                            self.app.logger.error(f"Failed to resync Handy after seek: {e}")
                    finally:
                        if loop is not None:
                            loop.close()
                
                # Run resync in background thread
                import threading
                thread = threading.Thread(target=stop_and_resync, name="HandyResync", daemon=True)
                thread.start()
                print("DEBUG: Resync thread started")
                
        except Exception as e:
            print(f"DEBUG: Exception in Handy resync: {e}")
            if hasattr(self.app, 'logger'):
                self.app.logger.error(f"Failed to resync Handy after seek: {e}")

    def _render_component_overlays(self, app_state):
        """Render component overlays (gauges, movement bar, 3D simulator) on video display."""
        # Check if any overlay modes are enabled
        gauge_overlay = self.app.app_settings.get('gauge_overlay_mode', False)
        movement_bar_overlay = self.app.app_settings.get('movement_bar_overlay_mode', False)
        simulator_3d_overlay = self.app.app_settings.get('simulator_3d_overlay_mode', False)

        if not (gauge_overlay or movement_bar_overlay or simulator_3d_overlay):
            return

        # Get video display rect for positioning
        img_rect = self._actual_video_image_rect_on_screen
        if not img_rect:
            return

        video_min_x = img_rect['min_x']
        video_min_y = img_rect['min_y']
        video_max_x = img_rect['max_x']
        video_max_y = img_rect['max_y']
        video_width = video_max_x - video_min_x
        video_height = video_max_y - video_min_y

        # Render gauges overlay (bottom-left and bottom-center)
        if gauge_overlay:
            if app_state.show_gauge_window_timeline1:
                self._render_gauge_overlay(app_state, "timeline1", video_min_x, video_min_y, video_max_x, video_max_y)
            if app_state.show_gauge_window_timeline2:
                self._render_gauge_overlay(app_state, "timeline2", video_min_x, video_min_y, video_max_x, video_max_y)

        # Render movement bar overlay (bottom-right)
        if movement_bar_overlay and app_state.show_lr_dial_graph:
            self._render_movement_bar_overlay(app_state, video_min_x, video_min_y, video_max_x, video_max_y)

        # Render 3D simulator overlay (top-left)
        if simulator_3d_overlay and app_state.show_simulator_3d:
            self._render_simulator_3d_overlay(app_state, video_min_x, video_min_y, video_max_x, video_max_y)

    def _render_movement_bar_overlay(self, app_state, video_min_x, video_min_y, video_max_x, video_max_y):
        """Render movement bar as overlay on video display (bottom-right)."""
        overlay_width = 180
        overlay_height = 220
        padding = 10

        # Position at bottom-right of video (default position)
        overlay_x = video_max_x - overlay_width - padding
        overlay_y = video_max_y - overlay_height - padding

        imgui.set_next_window_position(overlay_x, overlay_y, condition=imgui.ONCE)
        imgui.set_next_window_size(overlay_width, overlay_height, condition=imgui.ONCE)

        # Fully transparent background, no border
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, 0.0, 0.0, 0.0, 0.0)
        imgui.push_style_color(imgui.COLOR_BORDER, 0.0, 0.0, 0.0, 0.0)
        imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 0.0)

        window_flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_SCROLLBAR |
                       imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_SAVED_SETTINGS)

        imgui.begin("Movement Bar (Overlay)##MovementBarOverlay", flags=window_flags)

        # Draw movement bar content
        self._draw_movement_bar_content(app_state)

        # Clamp window position to video display area
        current_pos = imgui.get_window_position()
        current_size = imgui.get_window_size()

        # Calculate clamped position
        min_x = max(video_min_x, min(current_pos[0], video_max_x - current_size[0]))
        min_y = max(video_min_y, min(current_pos[1], video_max_y - current_size[1]))

        # Apply clamped position if different
        if (min_x, min_y) != (current_pos[0], current_pos[1]):
            imgui.set_window_position(min_x, min_y)

        imgui.end()
        imgui.pop_style_var(1)
        imgui.pop_style_color(2)

    def _draw_movement_bar_content(self, app_state):
        """Draw the actual movement bar visualization."""
        import numpy as np

        draw_list = imgui.get_window_draw_list()
        content_start_pos = imgui.get_cursor_screen_pos()
        content_avail_w, content_avail_h = imgui.get_content_region_available()

        padding = 15
        canvas_origin_x = content_start_pos[0] + padding
        canvas_origin_y = content_start_pos[1] + padding
        drawable_width = content_avail_w - 2 * padding
        drawable_height = content_avail_h - 2 * padding

        if drawable_width < 80 or drawable_height < 120:
            imgui.text("Too small")
            return

        # Get current funscript values
        up_down_position = getattr(app_state, 'gauge_value_t1', 50)
        roll_angle = getattr(app_state, 'lr_dial_value', 50)

        # Calculate center point for rotation
        center_x = canvas_origin_x + drawable_width / 2
        center_y = canvas_origin_y + drawable_height / 2

        # Convert roll_angle to rotation (-30° to +30°)
        roll_degrees = -((roll_angle / 100.0) - 0.5) * 60.0
        roll_radians = np.radians(roll_degrees)
        cos_r, sin_r = np.cos(roll_radians), np.sin(roll_radians)

        # Bar dimensions
        bar_width = min(50, drawable_width * 0.25)
        bar_height = drawable_height * 0.75

        # Bar corners (unrotated, centered)
        half_w = bar_width / 2
        half_h = bar_height / 2
        corners = [
            (-half_w, -half_h),  # Top-left
            (half_w, -half_h),   # Top-right
            (half_w, half_h),    # Bottom-right
            (-half_w, half_h)    # Bottom-left
        ]

        # Rotate corners
        rotated = [(x * cos_r - y * sin_r + center_x, x * sin_r + y * cos_r + center_y) for x, y in corners]

        # Draw bar background
        p1, p2, p3, p4 = rotated
        bar_bg_color = imgui.get_color_u32_rgba(0.2, 0.2, 0.2, 0.9)
        draw_list.add_triangle_filled(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], bar_bg_color)
        draw_list.add_triangle_filled(p1[0], p1[1], p3[0], p3[1], p4[0], p4[1], bar_bg_color)

        # Calculate fill height based on up_down_position
        fill_ratio = up_down_position / 100.0
        fill_height = bar_height * fill_ratio

        # Fill region corners (from bottom up to fill_height, with 2px padding)
        fill_y_start = half_h - fill_height
        fill_corners_local = [
            (-half_w + 2, fill_y_start),  # Top-left of fill
            (half_w - 2, fill_y_start),   # Top-right of fill
            (half_w - 2, half_h - 2),     # Bottom-right
            (-half_w + 2, half_h - 2)     # Bottom-left
        ]

        # Only draw fill if there's something to fill
        if fill_height > 0:
            rotated_fill = [(x * cos_r - y * sin_r + center_x, x * sin_r + y * cos_r + center_y) for x, y in fill_corners_local]

            # Color based on position: red (down) to green (up)
            if fill_ratio < 0.5:
                # Bottom half: red to yellow
                t = fill_ratio * 2  # 0 to 1
                r, g, b = 0.8, 0.2 + t * 0.6, 0.0
            else:
                # Top half: yellow to green
                t = (fill_ratio - 0.5) * 2  # 0 to 1
                r, g, b = 0.8 - t * 0.6, 0.8, t * 0.2

            fill_color = imgui.get_color_u32_rgba(r, g, b, 0.8)

            # Draw filled portion
            f1, f2, f3, f4 = rotated_fill
            draw_list.add_triangle_filled(f1[0], f1[1], f2[0], f2[1], f3[0], f3[1], fill_color)
            draw_list.add_triangle_filled(f1[0], f1[1], f3[0], f3[1], f4[0], f4[1], fill_color)

        # Draw bar border
        bar_border_color = imgui.get_color_u32_rgba(0.5, 0.5, 0.5, 1.0)
        for i in range(len(rotated)):
            p_cur = rotated[i]
            p_next = rotated[(i + 1) % len(rotated)]
            draw_list.add_line(p_cur[0], p_cur[1], p_next[0], p_next[1], bar_border_color, thickness=2)

    def _render_gauge_overlay(self, app_state, timeline_name, video_min_x, video_min_y, video_max_x, video_max_y):
        """Render gauge as overlay on video display."""
        # Gauge dimensions
        overlay_width = 120
        overlay_height = 250
        padding = 10

        # Position based on timeline (T1 = bottom-left, T2 = bottom-center)
        if timeline_name == "timeline1":
            overlay_x = video_min_x + padding
            overlay_y = video_max_y - overlay_height - padding
            window_id = "Gauge T1 (Overlay)##GaugeT1Overlay"
            gauge_value = getattr(app_state, 'gauge_value_t1', 50)
        else:
            # Center-left (offset from timeline1)
            overlay_x = video_min_x + padding + overlay_width + 20
            overlay_y = video_max_y - overlay_height - padding
            window_id = "Gauge T2 (Overlay)##GaugeT2Overlay"
            gauge_value = getattr(app_state, 'gauge_value_t2', 50)

        imgui.set_next_window_position(overlay_x, overlay_y, condition=imgui.ONCE)
        imgui.set_next_window_size(overlay_width, overlay_height, condition=imgui.ONCE)

        # Fully transparent background, no border
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, 0.0, 0.0, 0.0, 0.0)
        imgui.push_style_color(imgui.COLOR_BORDER, 0.0, 0.0, 0.0, 0.0)
        imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 0.0)

        window_flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_SCROLLBAR |
                       imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_SAVED_SETTINGS)

        imgui.begin(window_id, flags=window_flags)

        # Draw gauge content
        self._draw_gauge_content(gauge_value)

        # Clamp window position to video display area
        current_pos = imgui.get_window_position()
        current_size = imgui.get_window_size()

        # Calculate clamped position
        min_x = max(video_min_x, min(current_pos[0], video_max_x - current_size[0]))
        min_y = max(video_min_y, min(current_pos[1], video_max_y - current_size[1]))

        # Apply clamped position if different
        if (min_x, min_y) != (current_pos[0], current_pos[1]):
            imgui.set_window_position(min_x, min_y)

        imgui.end()
        imgui.pop_style_var(1)
        imgui.pop_style_color(2)

    def _draw_gauge_content(self, gauge_value):
        """Draw the actual gauge visualization."""
        from config.element_group_colors import GaugeColors

        draw_list = imgui.get_window_draw_list()
        content_start_pos = imgui.get_cursor_screen_pos()
        content_avail_w, content_avail_h = imgui.get_content_region_available()

        padding = 10
        canvas_origin_x = content_start_pos[0] + padding
        canvas_origin_y = content_start_pos[1] + padding
        drawable_width = content_avail_w - 2 * padding
        drawable_height = content_avail_h - 2 * padding

        if drawable_width < 30 or drawable_height < 100:
            imgui.text("Too small")
            return

        # Vertical gauge bar
        bar_x = canvas_origin_x + drawable_width / 2 - 15
        bar_y = canvas_origin_y
        bar_width = 30
        bar_height = drawable_height

        # Background
        bg_color = imgui.get_color_u32_rgba(*GaugeColors.BACKGROUND)
        draw_list.add_rect_filled(bar_x, bar_y, bar_x + bar_width, bar_y + bar_height, bg_color, rounding=5)

        # Fill based on gauge value (bottom to top) - use BAR_GREEN
        fill_height = bar_height * (gauge_value / 100.0)
        fill_y = bar_y + bar_height - fill_height

        fill_color = imgui.get_color_u32_rgba(*GaugeColors.BAR_GREEN)
        if fill_height > 0:
            draw_list.add_rect_filled(bar_x, fill_y, bar_x + bar_width, bar_y + bar_height, fill_color, rounding=5)

        # Outline
        outline_color = imgui.get_color_u32_rgba(*GaugeColors.BORDER)
        draw_list.add_rect(bar_x, bar_y, bar_x + bar_width, bar_y + bar_height, outline_color, rounding=5, thickness=2)

        # Center line (50% mark) - use TEXT color
        center_y = bar_y + bar_height / 2
        center_line_color = imgui.get_color_u32_rgba(*GaugeColors.TEXT)
        draw_list.add_line(bar_x, center_y, bar_x + bar_width, center_y, center_line_color, thickness=2)

        # Value text
        text = f"{int(gauge_value)}"
        text_size = imgui.calc_text_size(text)
        text_x = bar_x + bar_width / 2 - text_size[0] / 2
        text_y = bar_y + bar_height + 5
        text_color = imgui.get_color_u32_rgba(*GaugeColors.VALUE_TEXT)
        draw_list.add_text(text_x, text_y, text_color, text)

    def _render_simulator_3d_overlay(self, app_state, video_min_x, video_min_y, video_max_x, video_max_y):
        """Render 3D simulator as overlay on video display (bottom-right, half size)."""
        video_width = video_max_x - video_min_x
        video_height = video_max_y - video_min_y

        # Size: exactly half of video width and height
        overlay_width = int(video_width / 2)
        overlay_height = int(video_height / 2)

        # Position at bottom-right of video (aligned to corner)
        overlay_x = video_max_x - overlay_width
        overlay_y = video_max_y - overlay_height

        imgui.set_next_window_position(overlay_x, overlay_y, condition=imgui.ALWAYS)
        imgui.set_next_window_size(overlay_width, overlay_height, condition=imgui.ALWAYS)

        # Fully transparent background, no border
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, 0.0, 0.0, 0.0, 0.0)
        imgui.push_style_color(imgui.COLOR_BORDER, 0.0, 0.0, 0.0, 0.0)
        imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 0.0)

        window_flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_SCROLLBAR |
                       imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_SAVED_SETTINGS |
                       imgui.WINDOW_NO_RESIZE)

        imgui.begin("3D Simulator (Overlay)##Simulator3DOverlay", flags=window_flags)

        # Get current window content size for rendering
        content_w, content_h = imgui.get_content_region_available()

        # Get simulator instance and render 3D content
        if hasattr(self.gui_instance, 'simulator_3d_window_ui'):
            simulator = self.gui_instance.simulator_3d_window_ui
            # Render the 3D content with overlay window size
            if content_w > 50 and content_h > 50:  # Minimum size check
                simulator.render_3d_content(width=int(content_w), height=int(content_h))
            else:
                imgui.text("Window too small")
        else:
            imgui.text("3D Simulator Unavailable")

        imgui.end()
        imgui.pop_style_var(1)
        imgui.pop_style_color(2)
