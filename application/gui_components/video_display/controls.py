import sys
import time

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



class VideoControlsMixin:
    """Mixin fragment for VideoDisplayUI."""

    def _render_video_controls_with_autohide(self, app_state):
        """Wrapper that handles auto-hide logic for video overlay controls."""
        img_rect = self._actual_video_image_rect_on_screen
        if img_rect['w'] <= 0 or img_rect['h'] <= 0:
            return

        io = imgui.get_io()
        mouse_x, mouse_y = io.mouse_pos
        now = time.monotonic()

        # Detect mouse movement
        if (mouse_x, mouse_y) != self._controls_last_mouse_pos:
            self._controls_last_mouse_pos = (mouse_x, mouse_y)
            # Reset timer if mouse is hovering over the video area
            is_hovering_video = (img_rect['min_x'] <= mouse_x <= img_rect['max_x'] and
                                 img_rect['min_y'] <= mouse_y <= img_rect['max_y'])
            if is_hovering_video:
                self._controls_last_activity_time = now

        # Show controls if within timeout
        elapsed = now - self._controls_last_activity_time
        if elapsed > self._CONTROLS_HIDE_TIMEOUT:
            return

        # Fade alpha: full opacity for first 2s, then fade over the last 1s
        fade_start = self._CONTROLS_HIDE_TIMEOUT - 1.0
        if elapsed > fade_start:
            alpha = 1.0 - (elapsed - fade_start) / 1.0
        else:
            alpha = 1.0

        self._render_video_zoom_pan_controls(app_state, alpha)
        self._render_playback_controls_overlay(alpha)


    def _render_playback_controls_overlay(self, alpha=1.0):
        """Renders playback controls as a floating overlay window on the video."""
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

        # Button sizing — slightly larger for modern feel
        button_h_ref = imgui.get_frame_height()
        pb_icon_w, pb_play_w, pb_stop_w = button_h_ref, button_h_ref, button_h_ref
        pb_btn_spacing = 6.0  # More breathing room between buttons
        group_separator = 14.0  # Visual gap between transport group and extras

        # Account for handy + fullscreen buttons in width calculation
        total_controls_width = (pb_icon_w * 6) + (pb_btn_spacing * 5) + group_separator + (button_h_ref * 2.8) + button_h_ref

        img_rect = self._actual_video_image_rect_on_screen
        if img_rect['w'] <= 0 or img_rect['h'] <= 0:
            return

        padding = 8.0
        overlay_x = img_rect['min_x'] + (img_rect['w'] - total_controls_width) / 2 - padding
        overlay_y = img_rect['max_y'] - button_h_ref - padding * 4
        # Clamp both edges so the overlay stays within the video panel.
        # Without the right clamp the rightmost buttons can overlap the adjacent
        # panel, which captures their mouse events and makes them unresponsive.
        overlay_x = max(img_rect['min_x'], overlay_x)
        overlay_x = min(overlay_x, max(img_rect['min_x'],
                                       img_rect['max_x'] - total_controls_width - 2 * padding))
        overlay_y = max(img_rect['min_y'], overlay_y)

        flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE |
                 imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_SAVED_SETTINGS |
                 imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_FOCUS_ON_APPEARING |
                 imgui.WINDOW_NO_NAV | imgui.WINDOW_NO_BACKGROUND)

        imgui.set_next_window_position(overlay_x, overlay_y)
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (padding, padding))

        imgui.begin("##VideoPlaybackOverlay", flags=flags)

        # Pill-shaped semi-transparent background
        draw_list = imgui.get_window_draw_list()
        win_pos = imgui.get_window_position()
        win_size = imgui.get_window_size()
        pill_rounding = win_size[1] / 2.0  # Full pill shape
        draw_list.add_rect_filled(
            win_pos[0], win_pos[1],
            win_pos[0] + win_size[0], win_pos[1] + win_size[1],
            imgui.get_color_u32_rgba(0.08, 0.08, 0.08, 0.6 * alpha),
            rounding=pill_rounding
        )

        # Keep controls visible while hovering the overlay itself
        if imgui.is_window_hovered(imgui.HOVERED_ALLOW_WHEN_BLOCKED_BY_ACTIVE_ITEM):
            self._controls_last_activity_time = time.monotonic()

        imgui.push_style_var(imgui.STYLE_ALPHA, alpha)

        with _DisabledScope(controls_disabled):
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
            _mpv = getattr(self.app, '_mpv_controller', None)
            if _mpv and _mpv.is_active:
                is_playing = _mpv.is_playing
            else:
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

            imgui.same_line(spacing=pb_btn_spacing)

            # Stop button
            stop_tex, _, _ = icon_mgr.get_icon_texture('stop.png')
            if stop_tex and imgui.image_button(stop_tex, pb_stop_w, pb_icon_w):
                event_handlers.handle_playback_control("stop")
            elif not stop_tex and imgui.button("[]##VidOverStop", width=pb_stop_w):
                event_handlers.handle_playback_control("stop")
            if imgui.is_item_hovered():
                imgui.set_tooltip("Stop Playback")

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

        # Separator before extras group (Handy + Fullscreen)
        imgui.same_line(spacing=group_separator)
        # Subtle vertical divider
        sep_x = imgui.get_cursor_screen_pos()[0] - group_separator / 2
        sep_y_top = win_pos[1] + padding + 2
        sep_y_bot = win_pos[1] + win_size[1] - padding - 2
        draw_list.add_line(sep_x, sep_y_top, sep_x, sep_y_bot,
                           imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 0.15 * alpha), 1.0)
        imgui.same_line(spacing=0)

        # Handy and Fullscreen buttons manage their own disabled state
        self._render_handy_control_button_inline(pb_btn_spacing, button_h_ref, controls_disabled)
        self._render_fullscreen_button_inline(pb_btn_spacing, button_h_ref, controls_disabled)

        imgui.pop_style_var()  # STYLE_ALPHA
        imgui.end()
        imgui.pop_style_var()  # STYLE_WINDOW_PADDING


    def _render_video_zoom_pan_controls(self, app_state, alpha=1.0):
        button_h_ref = imgui.get_frame_height()
        img_rect = self._actual_video_image_rect_on_screen
        if img_rect['w'] <= 0 or img_rect['h'] <= 0: return

        padding = 8.0
        overlay_x = img_rect['min_x'] + padding
        overlay_y = img_rect['min_y'] + padding

        flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE |
                 imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_SAVED_SETTINGS |
                 imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_FOCUS_ON_APPEARING |
                 imgui.WINDOW_NO_NAV | imgui.WINDOW_NO_BACKGROUND)

        imgui.set_next_window_position(overlay_x, overlay_y)
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (padding, padding))

        imgui.begin("##VideoZoomPanOverlay", flags=flags)

        # Pill-shaped semi-transparent background
        draw_list = imgui.get_window_draw_list()
        win_pos = imgui.get_window_position()
        win_size = imgui.get_window_size()
        pill_rounding = min(win_size[1] / 2.0, 12.0)
        draw_list.add_rect_filled(
            win_pos[0], win_pos[1],
            win_pos[0] + win_size[0], win_pos[1] + win_size[1],
            imgui.get_color_u32_rgba(0.08, 0.08, 0.08, 0.6 * alpha),
            rounding=pill_rounding
        )

        # Keep controls visible while hovering the overlay itself
        if imgui.is_window_hovered(imgui.HOVERED_ALLOW_WHEN_BLOCKED_BY_ACTIVE_ITEM):
            self._controls_last_activity_time = time.monotonic()

        imgui.push_style_var(imgui.STYLE_ALPHA, alpha)

        # Get icon texture manager for zoom controls
        icon_mgr = get_icon_texture_manager()
        zoom_btn_size = button_h_ref

        # Zoom In button
        zoom_in_tex, _, _ = icon_mgr.get_icon_texture('zoom-in.png')
        if zoom_in_tex and imgui.image_button(zoom_in_tex, zoom_btn_size, zoom_btn_size):
            app_state.adjust_video_zoom(1.2)
        elif not zoom_in_tex and imgui.button("Z-In##VidOverZoomIn"):
            app_state.adjust_video_zoom(1.2)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Zoom In Video")

        imgui.same_line(spacing=4)

        # Zoom Out button
        zoom_out_tex, _, _ = icon_mgr.get_icon_texture('zoom-out.png')
        if zoom_out_tex and imgui.image_button(zoom_out_tex, zoom_btn_size, zoom_btn_size):
            app_state.adjust_video_zoom(1 / 1.2)
        elif not zoom_out_tex and imgui.button("Z-Out##VidOverZoomOut"):
            app_state.adjust_video_zoom(1 / 1.2)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Zoom Out Video")

        imgui.same_line(spacing=4)

        # Reset button (counterclockwise arrow icon)
        reset_tex, _, _ = icon_mgr.get_icon_texture('reset.png')
        if reset_tex and imgui.image_button(reset_tex, zoom_btn_size, zoom_btn_size):
            app_state.reset_video_zoom_pan()
        elif not reset_tex and imgui.button("Rst##VidOverZoomReset"):
            app_state.reset_video_zoom_pan()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Reset Zoom and Pan (R)")

        imgui.same_line(spacing=4)
        imgui.text(f"{app_state.video_zoom_factor:.1f}x")

        pan_buttons_active = app_state.video_zoom_factor > 1.0
        if pan_buttons_active:
            # Pan Arrows Block (Left, Right, Up, Down on one line)
            if imgui.arrow_button("##VidOverPanLeft", imgui.DIRECTION_LEFT):
                app_state.pan_video_normalized_delta(-app_state.video_pan_step, 0)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Pan Video Left")
            imgui.same_line(spacing=4)
            if imgui.arrow_button("##VidOverPanRight", imgui.DIRECTION_RIGHT):
                app_state.pan_video_normalized_delta(app_state.video_pan_step, 0)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Pan Video Right")
            imgui.same_line(spacing=4)
            if imgui.arrow_button("##VidOverPanUp", imgui.DIRECTION_UP):
                app_state.pan_video_normalized_delta(0, -app_state.video_pan_step)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Pan Video Up")
            imgui.same_line(spacing=4)
            if imgui.arrow_button("##VidOverPanDown", imgui.DIRECTION_DOWN):
                app_state.pan_video_normalized_delta(0, app_state.video_pan_step)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Pan Video Down")

        imgui.pop_style_var()  # STYLE_ALPHA
        imgui.end()
        imgui.pop_style_var()  # STYLE_WINDOW_PADDING


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
        handy_disabled = controls_disabled or not button_enabled

        with _DisabledScope(handy_disabled):
            button_clicked = imgui.button(f"{button_text}##HandyControl", width=button_width)

        if button_clicked and not handy_disabled:
            if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(f" Handy button clicked - enabled: {button_enabled}, controls_disabled: {controls_disabled}")
            if self.handy_streaming_active:
                if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" Stopping Handy streaming")
                self._stop_handy_streaming()
            else:
                if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" Starting Handy streaming")
                self._start_handy_streaming()
            
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
        """Render the fullscreen button inline with playback controls (mpv, supporter exclusive)."""
        button_width = button_height
        imgui.same_line(spacing=spacing)

        mpv = getattr(self.app, '_mpv_controller', None)
        mpv_missing = getattr(self.app, '_mpv_binary_missing', False)
        video_path = self.app.file_manager.video_path if self.app.file_manager else None
        video_loaded = bool(video_path)
        is_fs_active = mpv is not None and mpv.is_active
        is_supporter = _is_feature_available("patreon_features")

        button_disabled = not video_loaded or not is_supporter or (is_supporter and mpv is None)

        icon_mgr = get_icon_texture_manager()
        button_icon_name = 'fullscreen-exit.png' if is_fs_active else 'fullscreen.png'
        button_text_fallback = "Exit FS" if is_fs_active else "FS"

        # Active highlight when in fullscreen
        from config.element_group_colors import ToolbarColors
        if is_fs_active:
            imgui.push_style_color(imgui.COLOR_BUTTON, *ToolbarColors.ACTIVE_BUTTON)
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *ToolbarColors.ACTIVE_BUTTON_HOVERED)
            imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *ToolbarColors.ACTIVE_BUTTON_PRESSED)

        button_clicked = False
        with _DisabledScope(button_disabled):
            fs_tex, _, _ = icon_mgr.get_icon_texture(button_icon_name)
            if fs_tex:
                button_clicked = imgui.image_button(fs_tex, button_width, button_width)
            else:
                button_clicked = imgui.button(f"{button_text_fallback}##FullscreenControl", width=button_width)

        # Capture item rect BEFORE pop_style_color so the last item is still the button
        item_min = imgui.get_item_rect_min()
        item_max = imgui.get_item_rect_max()

        if is_fs_active:
            imgui.pop_style_color(3)

        if button_clicked and not button_disabled:
            if is_fs_active:
                mpv.stop()
            elif mpv is not None:
                processor = self.app.processor
                start_frame = processor.current_frame_index if processor else 0
                mpv.start(video_path, start_frame=start_frame, fullscreen=True)

        # Use is_mouse_hovering_rect so tooltip fires even when button is disabled
        if imgui.is_mouse_hovering_rect(item_min[0], item_min[1], item_max[0], item_max[1]):
            imgui.begin_tooltip()
            if not is_supporter:
                imgui.text("Fullscreen playback (mpv)")
                imgui.text_disabled("Patreon Exclusive - add-on supporters")
            elif mpv is None and mpv_missing:
                imgui.text("mpv not installed")
                if sys.platform == "darwin":
                    imgui.text_disabled("Install: brew install mpv")
                elif sys.platform.startswith("linux"):
                    imgui.text_disabled("Install: sudo apt install mpv")
                else:
                    imgui.text_disabled("Install: winget install mpv")
            elif mpv is None:
                imgui.text("Fullscreen unavailable")
                imgui.text_disabled("mpv could not be initialised - check logs")
            elif not video_loaded:
                imgui.text("No video loaded")
            elif is_fs_active:
                imgui.text("Exit Fullscreen (F11)")
            else:
                imgui.text("Enter Fullscreen (F11)")
            imgui.end_tooltip()

