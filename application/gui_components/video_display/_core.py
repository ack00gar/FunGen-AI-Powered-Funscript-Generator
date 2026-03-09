import imgui
import logging
from typing import Optional, Tuple

import config.constants as constants
from config.element_group_colors import VideoDisplayColors
from application.utils import get_logo_texture_manager, get_icon_texture_manager
from application.utils.imgui_helpers import DisabledScope as _DisabledScope
from application.utils.feature_detection import is_feature_available as _is_feature_available

# Module-level logger for Handy debug output (disabled by default)
_handy_debug_logger = logging.getLogger(__name__ + '.handy')



class VideoDisplayCoreMixin:
    """Mixin fragment for VideoDisplayUI."""

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
        
        # Video controls overlay auto-hide
        self._controls_last_activity_time = 0.0
        self._controls_last_mouse_pos = (0.0, 0.0)
        self._CONTROLS_HIDE_TIMEOUT = 3.0


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


    def update_frame_texture_if_needed(self):
        """Upload the current video frame to GPU if it changed.

        Call this every frame so the texture stays fresh even when the
        normal video panel is not rendered (e.g. during fullscreen).
        """
        current_frame_index = getattr(self.app.processor, 'current_frame_index', None)
        frame_changed = (current_frame_index != self._last_uploaded_frame_index)

        if not frame_changed:
            return

        if self.app.processor and self.app.processor.current_frame is not None:
            with self.app.processor.frame_lock:
                if self.app.processor.current_frame is not None and hasattr(self.app.processor.current_frame, 'copy'):
                    current_frame_for_texture = self.app.processor.current_frame.copy()
                    self.gui_instance.update_texture(self.gui_instance.frame_texture_id, current_frame_for_texture)
                    self._last_uploaded_frame_index = current_frame_index
                    self._texture_update_count += 1
        else:
            # No frame available — invalidate so next video's frame 0 uploads fresh
            self._last_uploaded_frame_index = None


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

            # If video feed is disabled, show logo + button to reactivate (never show drop text)
            if not app_state.show_video_feed:
                self._render_reactivate_feed_button()
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
                else:
                    # No frame available (video closed/switching) — invalidate texture cache
                    # so the next video's frame 0 is always uploaded fresh
                    self._last_uploaded_frame_index = None

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

                            # Show "Seeking..." indicator when video is seeking
                            if self.app.processor and self.app.processor.seek_in_progress:
                                draw_list = imgui.get_window_draw_list()
                                # Draw semi-transparent overlay
                                overlay_color = imgui.get_color_u32_rgba(0.0, 0.0, 0.0, 0.5)
                                draw_list.add_rect_filled(
                                    self._video_display_rect_min[0],
                                    self._video_display_rect_min[1],
                                    self._video_display_rect_max[0],
                                    self._video_display_rect_max[1],
                                    overlay_color
                                )

                                center_x = (self._video_display_rect_min[0] + self._video_display_rect_max[0]) / 2
                                center_y = (self._video_display_rect_min[1] + self._video_display_rect_max[1]) / 2

                                # Draw "Seeking..." text in center
                                text = "Seeking..."
                                text_size = imgui.calc_text_size(text)
                                text_x = center_x - text_size.x / 2
                                text_y = center_y - text_size.y / 2 - 30  # Move up to make room for progress bar
                                text_color = imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0)
                                draw_list.add_text(text_x, text_y, text_color, text)

                                # Draw progress bar if we're creating frame buffer
                                if self.app.processor.frame_buffer_progress > 0:
                                    # Progress text
                                    progress_text = f"Creating frames buffer: {self.app.processor.frame_buffer_current}/{self.app.processor.frame_buffer_total}"
                                    progress_text_size = imgui.calc_text_size(progress_text)
                                    progress_text_x = center_x - progress_text_size.x / 2
                                    progress_text_y = center_y - 5
                                    draw_list.add_text(progress_text_x, progress_text_y, text_color, progress_text)

                                    # Progress bar
                                    bar_width = 300
                                    bar_height = 20
                                    bar_x = center_x - bar_width / 2
                                    bar_y = center_y + 15

                                    # Background
                                    bg_color = imgui.get_color_u32_rgba(0.2, 0.2, 0.2, 0.8)
                                    draw_list.add_rect_filled(bar_x, bar_y, bar_x + bar_width, bar_y + bar_height, bg_color)

                                    # Foreground (progress)
                                    progress = self.app.processor.frame_buffer_progress
                                    fg_color = imgui.get_color_u32_rgba(0.2, 0.6, 1.0, 0.9)
                                    draw_list.add_rect_filled(bar_x, bar_y, bar_x + bar_width * progress, bar_y + bar_height, fg_color)

                                    # Border
                                    border_color = imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 0.8)
                                    draw_list.add_rect(bar_x, bar_y, bar_x + bar_width, bar_y + bar_height, border_color, thickness=2)

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

                            # Visualization of active User ROI (outline + tracked point)
                            # Rule: If ROI toggle is ON => always show. If ROI toggle is OFF => show only when not actively tracking (paused/stopped).
                            # Only show when the current tracker actually uses ROI (requires_intervention)
                            _tracker_info = self.app.tracker.get_tracker_info() if self.app.tracker else None
                            _is_roi_tracker = _tracker_info and getattr(_tracker_info, 'requires_intervention', False)
                            if _is_roi_tracker and getattr(self.app.tracker, 'user_roi_fixed', None) is not None and not self.app.is_setting_user_roi_mode:
                                tracker = self.app.tracker
                                proc = getattr(self.app, 'processor', None)
                                is_paused = bool(proc and hasattr(proc, 'pause_event') and proc.pause_event.is_set())
                                is_actively_tracking = bool(getattr(tracker, 'tracking_active', False)) and not is_paused
                                show_toggle_on = bool(getattr(tracker, 'show_roi', True))
                                allow_outline = show_toggle_on or (not show_toggle_on and not is_actively_tracking)
                                if allow_outline:
                                    draw_list = imgui.get_window_draw_list()
                                    urx_vid, ury_vid, urw_vid, urh_vid = tracker.user_roi_fixed
                                    roi_start_screen = self._video_to_screen_coords(urx_vid, ury_vid)
                                    roi_end_screen = self._video_to_screen_coords(urx_vid + urw_vid, ury_vid + urh_vid)
                                    if roi_start_screen and roi_end_screen:
                                        draw_list.add_rect(
                                            roi_start_screen[0], roi_start_screen[1],
                                            roi_end_screen[0], roi_end_screen[1],
                                            imgui.get_color_u32_rgba(*VideoDisplayColors.ROI_BORDER),
                                            thickness=2
                                        )
                                        draw_list.add_text(roi_start_screen[0], roi_start_screen[1] - 15,
                                                           imgui.get_color_u32_rgba(0, 255, 255, 255), "User ROI")
                                    # Draw tracked point
                                    if getattr(tracker, 'user_roi_tracked_point_relative', None) is not None:
                                        rel_x, rel_y = tracker.user_roi_tracked_point_relative
                                        pt_abs_x = urx_vid + rel_x
                                        pt_abs_y = ury_vid + rel_y
                                        pt_screen = self._video_to_screen_coords(pt_abs_x, pt_abs_y)
                                        if pt_screen:
                                            draw_list.add_circle_filled(pt_screen[0], pt_screen[1], 5,
                                                                        imgui.get_color_u32_rgba(0, 0.5, 1.0, 1.0))
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

                            # --- Render Component Overlays (if enabled) ---
                            self._render_component_overlays(app_state)

                            # Video control overlays (zoom/pan buttons + playback controls)
                            if app_state.show_video_controls_overlay:
                                self._render_video_controls_with_autohide(app_state)

                            # Handy sync overlay (top-right pill when connected)
                            self._render_handy_sync_overlay()

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


    def _render_reactivate_feed_button(self):
        """Renders logo and button to re-activate the video feed."""
        cursor_start_pos = imgui.get_cursor_pos()
        win_size = imgui.get_window_size()

        # Load logo texture
        logo_manager = get_logo_texture_manager()
        logo_texture = logo_manager.get_texture_id()
        logo_width, logo_height = logo_manager.get_dimensions()

        button_text = "Show Video Feed"
        button_size = imgui.calc_text_size(button_text)
        button_width = button_size[0] + imgui.get_style().frame_padding[0] * 2
        button_height = button_size[1] + imgui.get_style().frame_padding[1] * 2

        if logo_texture and logo_width > 0 and logo_height > 0:
            # Scale logo to reasonable size (max 200px while maintaining aspect ratio)
            max_logo_size = 200
            if logo_width > logo_height:
                display_logo_w = min(logo_width, max_logo_size)
                display_logo_h = int(logo_height * (display_logo_w / logo_width))
            else:
                display_logo_h = min(logo_height, max_logo_size)
                display_logo_w = int(logo_width * (display_logo_h / logo_height))

            # Calculate total height (logo + spacing + button)
            spacing = 20
            total_height = display_logo_h + spacing + button_height

            # Center vertically
            start_y = (win_size[1] - total_height) * 0.5 + cursor_start_pos[1]

            # Draw logo centered horizontally
            logo_x = (win_size[0] - display_logo_w) * 0.5 + cursor_start_pos[0]
            imgui.set_cursor_pos((logo_x, start_y))

            # Draw logo with slight transparency
            imgui.image(logo_texture, display_logo_w, display_logo_h, tint_color=(1.0, 1.0, 1.0, 0.6))

            # Draw button below logo
            button_y = start_y + display_logo_h + spacing
            button_x = (win_size[0] - button_width) * 0.5 + cursor_start_pos[0]
            imgui.set_cursor_pos((button_x, button_y))
        else:
            # Fallback to button-only if logo fails to load
            button_x = (win_size[0] - button_width) / 2 + cursor_start_pos[0]
            button_y = (win_size[1] - button_height) / 2 + cursor_start_pos[1]
            imgui.set_cursor_pos((button_x, button_y))

        if imgui.button(button_text):
            self.app.app_state_ui.show_video_feed = True


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

