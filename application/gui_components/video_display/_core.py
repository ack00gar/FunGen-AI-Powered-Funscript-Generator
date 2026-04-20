import imgui
import logging
from typing import Optional, Tuple

import OpenGL.GL as gl

import config.constants as constants
from config.element_group_colors import VideoDisplayColors
from application.utils import get_logo_texture_manager, get_icon_texture_manager
from application.utils.imgui_helpers import DisabledScope as _DisabledScope
from application.utils.feature_detection import is_feature_available as _is_feature_available
from config.constants_colors import CurrentTheme

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
        self._last_uploaded_frame_version = -1  # Track which frame version is in GPU texture
        self._last_uploaded_frame_index = None  # Track which frame index is in GPU texture
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
        """Converts absolute screen coordinates to video buffer coordinates, accounting for pan, zoom, and content UV cropping."""
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

        # Map through processing content UV rect (640x640 padded space -> screen)
        c_left, c_top, c_right, c_bottom = app_state.get_processing_content_uv_rect()
        c_w = c_right - c_left
        c_h = c_bottom - c_top

        # Pan/zoom in content-relative space, then map to full texture UV
        uv_span_x = c_w / app_state.video_zoom_factor
        uv_span_y = c_h / app_state.video_zoom_factor

        tex_norm_x = c_left + app_state.video_pan_normalized[0] * c_w + norm_visible_x * uv_span_x
        tex_norm_y = c_top + app_state.video_pan_normalized[1] * c_h + norm_visible_y * uv_span_y

        if not (0 <= tex_norm_x <= 1 and 0 <= tex_norm_y <= 1):  # Point is outside the full texture due to pan/zoom
            return None

        # Always use processing frame dimensions (yolo_input_size) for overlay coordinate space
        video_buffer_w = self.app.yolo_input_size
        video_buffer_h = self.app.yolo_input_size

        video_x = int(tex_norm_x * video_buffer_w)
        video_y = int(tex_norm_y * video_buffer_h)

        return video_x, video_y


    def _video_to_screen_coords(self, video_x: int, video_y: int) -> tuple | None:
        """Converts video buffer coordinates to absolute screen coordinates, accounting for pan, zoom, and content UV cropping."""
        app_state = self.app.app_state_ui
        img_rect = self._actual_video_image_rect_on_screen

        # Always use processing frame dimensions (yolo_input_size) for overlay coordinate space
        video_buffer_w = self.app.yolo_input_size
        video_buffer_h = self.app.yolo_input_size

        if video_buffer_w <= 0 or video_buffer_h <= 0 or img_rect['w'] <= 0 or img_rect['h'] <= 0:
            return None

        # Normalized position on the *full* texture
        tex_norm_x = video_x / video_buffer_w
        tex_norm_y = video_y / video_buffer_h

        # Map through processing content UV rect (reverse: screen -> 640x640 padded space)
        c_left, c_top, c_right, c_bottom = app_state.get_processing_content_uv_rect()
        c_w = c_right - c_left
        c_h = c_bottom - c_top

        if c_w <= 0 or c_h <= 0: return None

        uv_span_x = c_w / app_state.video_zoom_factor
        uv_span_y = c_h / app_state.video_zoom_factor

        if uv_span_x == 0 or uv_span_y == 0: return None

        # Convert texture-space position to content-relative visible position
        # tex_norm = c_left + pan * c_w + norm_visible * uv_span  =>  solve for norm_visible
        norm_visible_x = (tex_norm_x - c_left - app_state.video_pan_normalized[0] * c_w) / uv_span_x
        norm_visible_y = (tex_norm_y - c_top - app_state.video_pan_normalized[1] * c_h) / uv_span_y

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

        Uses _frame_version (incremented on every current_frame assignment)
        instead of current_frame_index to detect changes.  This avoids a
        race where seek_video() updates the index immediately but the
        background worker hasn't delivered the new frame yet - the old
        frame would be uploaded and the new one silently dropped.
        """
        if not self.app.processor:
            return

        frame_version = getattr(self.app.processor, '_frame_version', 0)
        if frame_version == self._last_uploaded_frame_version:
            return

        if self.app.processor.current_frame is not None:
            # Snapshot ref under lock; upload outside so GL doesn't starve decoder.
            with self.app.processor.frame_lock:
                current_frame = self.app.processor.current_frame
                current_index = getattr(self.app.processor, 'current_frame_index', None)
            if current_frame is not None and hasattr(current_frame, 'shape'):
                self.gui_instance.update_texture(self.gui_instance.frame_texture_id, current_frame)
                self._last_uploaded_frame_version = frame_version
                self._last_uploaded_frame_index = current_index
                self._texture_update_count += 1
        else:
            # No frame available - invalidate so next video's frame 0 uploads fresh
            self._last_uploaded_frame_version = -1
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
            is_expanded, new_visibility = imgui.begin("FunGen: Video Display", closable=True, flags=imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_COLLAPSE)

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
                current_frame_index = getattr(self.app.processor, 'current_frame_index', None)
                frame_version = getattr(self.app.processor, '_frame_version', 0)

                # PERFORMANCE: Check if frame data changed before copying/uploading to GPU
                frame_changed = (frame_version != self._last_uploaded_frame_version)
                uploaded_this_frame = False

                if self.app.processor and self.app.processor.current_frame is not None:
                    # Snapshot ref under lock; upload outside (avoid decoder stall).
                    with self.app.processor.frame_lock:
                        current_frame = self.app.processor.current_frame
                        # shape check: current_frame can be an int sentinel, not a ndarray.
                        if current_frame is None or not hasattr(current_frame, 'shape'):
                            current_frame = None
                    if current_frame is not None:
                        if frame_changed:
                            self.gui_instance.update_texture(
                                self.gui_instance.frame_texture_id, current_frame)
                            self._last_uploaded_frame_version = frame_version
                            self._last_uploaded_frame_index = current_frame_index
                            self._texture_update_count += 1
                            uploaded_this_frame = True
                        else:
                            self._texture_skip_count += 1
                else:
                    # No frame available (video closed/switching) - invalidate texture cache
                    # so the next video's frame 0 is always uploaded fresh
                    self._last_uploaded_frame_version = -1
                    self._last_uploaded_frame_index = None

                video_frame_available = uploaded_this_frame or (not frame_changed and self._last_uploaded_frame_index is not None)

                proc = self.app.processor
                mpv_display = getattr(self.gui_instance, 'mpv_display', None)

                from application.gui_components.video_display.display_route import (
                    compute_display_route, fit_rect_to_panel,
                )
                route = compute_display_route(self.app)

                if video_frame_available or route.source != 'blank':
                    available_w_video, available_h_video = imgui.get_content_region_available()

                    if available_w_video > 0 and available_h_video > 0:
                        zoom = float(getattr(app_state, 'video_zoom_factor', 1.0) or 1.0)
                        if route.fill_panel:
                            display_w = int(available_w_video)
                            display_h = int(available_h_video)
                            cursor_x_offset = 0.0
                            cursor_y_offset = 0.0
                            self._cached_fit_size = None
                        else:
                            dw, dh, off_x, off_y = fit_rect_to_panel(
                                route.content_aspect,
                                available_w_video, available_h_video, zoom)
                            display_w, display_h = int(dw), int(dh)
                            cursor_x_offset, cursor_y_offset = off_x, off_y

                        if route.source in ('mpv_shader', 'mpv_direct'):
                            self._render_mpv_to_fbo(proc, mpv_display)
                        if route.source == 'mpv_shader':
                            self._render_shader_dewarp(
                                proc, app_state,
                                target_w=display_w, target_h=display_h,
                                locked=route.shader_locked)

                        if display_w > 0 and display_h > 0:
                            self._update_actual_video_image_rect(display_w, display_h, cursor_x_offset, cursor_y_offset)

                            win_content_x, win_content_y = imgui.get_cursor_pos()
                            imgui.set_cursor_pos((win_content_x + cursor_x_offset, win_content_y + cursor_y_offset))

                            uv0_x, uv0_y, uv1_x, uv1_y = route.uv
                            if route.texture_id > 0:
                                imgui.image(route.texture_id,
                                            display_w, display_h,
                                            (uv0_x, uv0_y), (uv1_x, uv1_y))
                            else:
                                imgui.dummy(display_w, display_h)

                            self._video_display_rect_min = imgui.get_item_rect_min()
                            self._video_display_rect_max = imgui.get_item_rect_max()

                            if (route.source == 'mpv_shader'
                                    and not route.shader_locked
                                    and imgui.is_item_hovered()
                                    and imgui.is_mouse_dragging(0, 2.0)):
                                drag = imgui.get_mouse_drag_delta(0, 2.0)
                                imgui.reset_mouse_drag_delta(0)
                                _SENS = 0.20
                                fmt_lc = (getattr(proc, 'vr_input_format', '') or '').lower()
                                fov_src = float(getattr(proc, 'vr_fov', 0) or 190)
                                if 'fisheye' in fmt_lc:
                                    yaw_lim = max(50.0, (fov_src - 90.0) * 0.5 + 45.0)
                                else:
                                    yaw_lim = 90.0
                                new_yaw = app_state.vr_pan_yaw + float(drag.x) * _SENS
                                new_pitch = app_state.vr_pan_pitch + float(drag.y) * _SENS
                                app_state.vr_pan_yaw = max(-yaw_lim, min(yaw_lim, new_yaw))
                                app_state.vr_pan_pitch = max(-80.0, min(80.0, new_pitch))

                            if route.overlay_status:
                                self._render_status_overlay(
                                    route.overlay_status,
                                    route.status_busy,
                                    self._video_display_rect_min,
                                    self._video_display_rect_max)

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


    def _render_mpv_to_fbo(self, proc, mpv_display):
        if mpv_display is None:
            return
        cap = getattr(proc, '_DISPLAY_CHAIN_MAX', 1024)
        info = getattr(proc, 'video_info', None) or {}
        src_w = int(info.get('width', 0))
        src_h = int(info.get('height', 0))
        if src_w > 0 and src_h > 0:
            if src_w >= src_h:
                fbo_w = min(src_w, cap)
                fbo_h = max(64, int(round(fbo_w * src_h / src_w)))
            else:
                fbo_h = min(src_h, cap)
                fbo_w = max(64, int(round(fbo_h * src_w / src_h)))
        else:
            fbo_w = min(int(getattr(proc, '_display_frame_w', 640) or 640), cap)
            fbo_h = min(int(getattr(proc, '_display_frame_h', 640) or 640), cap)
        fbo_w = max(64, fbo_w)
        fbo_h = max(64, fbo_h)

        render_to_fbo = getattr(mpv_display, 'render_to_fbo', None)
        if render_to_fbo is None:
            buf = mpv_display.render_to_buffer(fbo_w, fbo_h)
            if buf is not None:
                self.gui_instance.update_texture(
                    self.gui_instance.mpv_display_texture_id, buf)
            return

        self.gui_instance.resize_mpv_display_target(fbo_w, fbo_h)
        try:
            gl.glDisable(gl.GL_BLEND); gl.glDisable(gl.GL_SCISSOR_TEST)
            gl.glDisable(gl.GL_DEPTH_TEST); gl.glDisable(gl.GL_CULL_FACE)
            gl.glDisable(gl.GL_STENCIL_TEST)
            gl.glColorMask(True, True, True, True); gl.glDepthMask(True)
            gl.glUseProgram(0); gl.glBindVertexArray(0)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
            gl.glActiveTexture(gl.GL_TEXTURE0); gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)
            gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 4)
        except Exception:
            pass
        try:
            render_to_fbo(self.gui_instance.mpv_display_fbo, fbo_w, fbo_h)
        finally:
            try:
                gl.glBindVertexArray(0); gl.glUseProgram(0)
                gl.glDisable(gl.GL_SCISSOR_TEST); gl.glDisable(gl.GL_DEPTH_TEST)
                gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)
            except Exception:
                pass

    def _render_shader_dewarp(self, proc, app_state,
                              target_w: int = 0, target_h: int = 0,
                              locked: bool = False):
        shader = getattr(self.gui_instance, 'vr_dewarp_shader', None)
        if shader is None or not shader.is_ready:
            return
        from video import vr_panel
        if target_w <= 0 or target_h <= 0:
            avail_w, avail_h = imgui.get_content_region_available()
            target_w = int(avail_w)
            target_h = int(avail_h)
        if target_w <= 0 or target_h <= 0:
            return
        # Pick the adaptive-quality spec for this frame.
        from video.vr_render_quality import VRRenderQualityMonitor
        mon = getattr(self.gui_instance, '_vr_quality_monitor', None)
        if mon is None:
            mon = VRRenderQualityMonitor()
            self.gui_instance._vr_quality_monitor = mon
        try:
            mode = self.app.app_settings.config.vr_display.quality_mode
        except Exception:
            mode = 'auto'
        try:
            want_supersample = self.app.app_settings.config.vr_display.shader_supersample
        except Exception:
            want_supersample = True
        spec = mon.current_spec(mode=mode)
        ss_factor = float(spec.supersample_factor) if want_supersample else 1.0
        # Log level changes so users can see adaptive quality working.
        _last_level = getattr(self.gui_instance, '_vr_quality_last_level', None)
        if _last_level != spec.name:
            self.app.logger.info(
                f"VR shader quality: {spec.name} (ss={ss_factor:.2f}x "
                f"bicubic={'on' if spec.use_bicubic else 'off'} "
                f"aniso={int(spec.aniso_level)}x ema={mon.ema_ms:.1f}ms)")
            self.gui_instance._vr_quality_last_level = spec.name
        cap = 4096 if ss_factor > 1.0 else 2048
        dw_w = max(64, min(int(round(target_w * ss_factor)), cap))
        dw_h = max(64, min(int(round(target_h * ss_factor)), cap))
        self.gui_instance.resize_vr_dewarp_target(dw_w, dw_h)
        # Apply anisotropic filtering to the input (mpv display) texture.
        # Cache the GL_MAX_TEXTURE_MAX_ANISOTROPY cap on first use (glGetFloatv
        # can force a pipeline sync every frame otherwise) and skip the
        # texture parameter update if the effective level has not changed --
        # glTexParameterf is cheap but redundant calls still traverse the GL
        # driver.
        try:
            import OpenGL.GL.EXT.texture_filter_anisotropic as _aniso_ext
            aniso_cap = getattr(self.gui_instance, "_aniso_cap", None)
            if aniso_cap is None:
                aniso_cap = float(gl.glGetFloatv(
                    _aniso_ext.GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT))
                self.gui_instance._aniso_cap = aniso_cap
            aniso = min(float(spec.aniso_level), aniso_cap)
            last_aniso = getattr(self.gui_instance, "_last_applied_aniso", None)
            if last_aniso != aniso:
                gl.glBindTexture(gl.GL_TEXTURE_2D,
                                  self.gui_instance.mpv_display_texture_id)
                gl.glTexParameterf(gl.GL_TEXTURE_2D,
                                    _aniso_ext.GL_TEXTURE_MAX_ANISOTROPY_EXT,
                                    aniso)
                gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
                self.gui_instance._last_applied_aniso = aniso
        except Exception:
            pass
        # Apply mpv scaler choice (runtime property change is cheap).
        mpv_display = getattr(self.gui_instance, 'mpv_display', None)
        if mpv_display is not None:
            try:
                cur_scale = getattr(mpv_display, '_current_scale', None)
                if cur_scale != spec.mpv_scale:
                    player = getattr(mpv_display, '_player', None) \
                             or getattr(mpv_display, 'player', None)
                    if player is not None:
                        try:
                            player.scale = spec.mpv_scale
                        except Exception:
                            pass
                    mpv_display._current_scale = spec.mpv_scale
            except Exception:
                pass

        fmt = (getattr(proc, 'vr_input_format', '') or '').lower()
        eye = vr_panel.read_setting(self.app.app_settings,
                                    default=vr_panel.EYE_LEFT)
        stereo, use_right_eye = vr_panel.shader_params(fmt, eye)
        if locked:
            yaw = 0.0
            pitch = float(getattr(proc, 'vr_pitch', 0) or 0.0)
            output_fov = 90.0
        else:
            yaw = float(getattr(app_state, 'vr_pan_yaw', 0.0))
            pitch = float(getattr(app_state, 'vr_pan_pitch', 0.0))
            output_fov = float(getattr(app_state, 'vr_pan_output_fov_deg', 90.0))
        projection = "fisheye" if 'fisheye' in fmt else "equirect"
        fisheye_fov = float(getattr(proc, 'vr_fov', 0) or 190) or 190.0
        output_projection = 'sg' if locked else 'flat'
        try:
            out_scale = self.app.app_settings.config.vr_display.shader_sg_scale
        except Exception:
            out_scale = 1.840
        try:
            gl.glBindTexture(gl.GL_TEXTURE_2D,
                              self.gui_instance.mpv_display_texture_id)
            gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        except Exception:
            pass
        in_w = int(getattr(self.gui_instance, 'mpv_display_w', 0) or 0)
        in_h = int(getattr(self.gui_instance, 'mpv_display_h', 0) or 0)
        shader.render_pass(
            input_texture_id=self.gui_instance.mpv_display_texture_id,
            output_fbo=self.gui_instance.vr_dewarp_fbo,
            width=self.gui_instance.vr_dewarp_w,
            height=self.gui_instance.vr_dewarp_h,
            params={
                "fisheye_fov_deg": fisheye_fov,
                "output_fov_deg": output_fov,
                "yaw_deg": yaw,
                "pitch_deg": pitch,
                "stereo_format": stereo,
                "use_right_eye": use_right_eye,
                "projection": projection,
                "output_projection": output_projection,
                "output_scale": out_scale,
                "use_bicubic": spec.use_bicubic,
                "input_tex_w": in_w,
                "input_tex_h": in_h,
            },
        )
        # Feed timing back into the monitor so auto mode can adapt next frame.
        try:
            mon.record_pass_ms(getattr(shader, '_last_render_ms', None))
        except Exception:
            pass
        try:
            gl.glBindTexture(gl.GL_TEXTURE_2D,
                              self.gui_instance.vr_dewarp_texture_id)
            gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        except Exception:
            pass

    def _render_status_overlay(self, text, busy, img_min, img_max):
        if not text or img_min is None or img_max is None:
            return
        draw_list = imgui.get_window_draw_list()
        x0, y0 = float(img_min[0]), float(img_min[1])
        x1, y1 = float(img_max[0]), float(img_max[1])
        cx = (x0 + x1) * 0.5
        cy = (y0 + y1) * 0.5
        draw_list.add_rect_filled(
            x0, y0, x1, y1,
            imgui.get_color_u32_rgba(0, 0, 0, 0.35))
        tx_size = imgui.calc_text_size(text)
        tx_color = imgui.get_color_u32_rgba(1, 1, 1, 0.95)
        draw_list.add_text(cx - tx_size.x * 0.5,
                           cy - tx_size.y * 0.5,
                           tx_color, text)
        if busy:
            import math as _m
            import time as _t
            t = _t.monotonic() * 2.0
            r = 14.0
            cy_spin = cy + tx_size.y + r + 12.0
            for i in range(8):
                ang = (i / 8.0) * _m.pi * 2.0 + t
                px = cx + _m.cos(ang) * r
                py = cy_spin + _m.sin(ang) * r
                alpha = 0.15 + 0.7 * ((i / 8.0 + (t % 1.0)) % 1.0)
                draw_list.add_circle_filled(
                    px, py, 2.5,
                    imgui.get_color_u32_rgba(1, 1, 1, alpha))

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
            imgui.text_colored(text_to_display, *CurrentTheme.DESCRIPTION_TEXT)  # Slightly dimmed text
        else:
            # Fallback to text-only if logo fails to load
            if win_size[0] > text_size[0] and win_size[1] > text_size[1]:
                imgui.set_cursor_pos(((win_size[0] - text_size[0]) * 0.5 + cursor_start_pos[0], (win_size[1] - text_size[1]) * 0.5 + cursor_start_pos[1]))
            imgui.text(text_to_display)

