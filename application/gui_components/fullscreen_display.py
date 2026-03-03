"""
Native Fullscreen Display

Borderless fullscreen on the main GLFW window. Renders the existing
frame_texture_id at full resolution with auto-hiding playback controls.
Audio, live tracking, and device sync continue unchanged.

Native-resolution display: a background thread (OpenCV VideoCapture) decodes
the current frame at a capped resolution (≤ _DISPLAY_MAX_DIM on the long edge)
into a dedicated GL texture so the fullscreen view is crisp.

VR content: the native-decode thread is skipped for VR videos (SBS/TB/fisheye
etc.).  For VR, the GPU-unwarped single-eye frame that already lives in
frame_texture_id is the correct view to display — showing the raw equirectangular
source would expose both panels and would need an unwarp step we cannot
perform efficiently here at full resolution.
"""

import time
import threading

import cv2
import glfw
import imgui
from OpenGL import GL as gl

from application.utils import get_icon_texture_manager

# Maximum pixel count on the long edge for the native-resolution display frame.
# Keeps memory and GPU texture upload manageable (8K → 1920px etc.).
_DISPLAY_MAX_DIM = 1920

# VR format strings that are NOT plain monoscopic 2D
_VR_FORMATS = {'he_sbs', 'he_tb', 'fisheye180', 'fisheye200', 'equirect180',
               'sbs', 'tb', 'half_sbs', 'half_tb'}


def _is_vr_format(vr_input_format: str) -> bool:
    """Return True when the format string indicates stereoscopic / spherical content."""
    if not vr_input_format:
        return False
    fmt = vr_input_format.lower().strip()
    return fmt in _VR_FORMATS or (fmt != 'mono' and fmt != 'auto' and fmt != '')


class NativeFullscreenManager:
    """GLFW borderless fullscreen using the main window."""

    __slots__ = (
        "app", "gui", "is_active",
        "_saved_pos", "_saved_size",
        "_show_controls", "_last_mouse_move_time", "_controls_timeout",
        "_last_mouse_pos", "_transition_time",
        # Native-resolution display state
        "_display_thread", "_display_stop",
        "_display_frame_lock", "_pending_display_frame",
        "_fullscreen_tex_id", "_fullscreen_tex_w", "_fullscreen_tex_h",
        "_display_is_vr",   # True when VR path is active (skip native decode)
    )

    def __init__(self, app_instance, gui_instance):
        self.app = app_instance
        self.gui = gui_instance
        self.is_active = False

        # Saved window state for restore
        self._saved_pos = (0, 0)
        self._saved_size = (1280, 720)

        # Auto-hiding controls
        self._show_controls = True
        self._last_mouse_move_time = 0.0
        self._controls_timeout = 3.0
        self._last_mouse_pos = (0.0, 0.0)

        # Debounce: ignore exit keys for a short period after entering
        self._transition_time = 0.0

        # Native-resolution background decoder
        self._display_thread = None
        self._display_stop = None
        self._display_frame_lock = threading.Lock()
        self._pending_display_frame = None   # (frame_w, frame_h, rgb_bytes)
        self._fullscreen_tex_id = 0
        self._fullscreen_tex_w = 0
        self._fullscreen_tex_h = 0
        self._display_is_vr = False

    # ------------------------------------------------------------------ public

    def enter_fullscreen(self):
        """Save window state, go exclusive fullscreen on current monitor."""
        window = self.gui.window
        if not window or self.is_active:
            return

        # Save current windowed state
        self._saved_pos = glfw.get_window_pos(window)
        self._saved_size = glfw.get_window_size(window)

        # Get the monitor the window is currently on
        monitor = self._get_current_monitor(window)
        mode = glfw.get_video_mode(monitor)

        # True exclusive fullscreen via GLFW — hides menu bar and dock on macOS
        glfw.set_window_monitor(
            window, monitor, 0, 0,
            mode.size.width, mode.size.height, mode.refresh_rate
        )

        self.is_active = True
        now = time.time()
        self._last_mouse_move_time = now
        self._transition_time = now
        self._last_mouse_pos = imgui.get_io().mouse_pos

        # Decide whether this is VR content. For VR we skip the native-decode
        # thread entirely and fall back to the GPU-unwarped frame_texture_id.
        proc = self.app.processor
        vr_fmt = getattr(proc, 'vr_input_format', 'mono') if proc else 'mono'
        self._display_is_vr = _is_vr_format(vr_fmt)

        if not self._display_is_vr:
            # Create a dedicated GL texture for the native-resolution display
            self._fullscreen_tex_id = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._fullscreen_tex_id)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            self._fullscreen_tex_w = 0
            self._fullscreen_tex_h = 0

            # Start background thread that decodes frames at capped resolution
            self._display_stop = threading.Event()
            self._display_thread = threading.Thread(
                target=self._display_decode_loop,
                daemon=True,
                name="FSDisplayDecode",
            )
            self._display_thread.start()

        self.app.logger.info("Entered fullscreen", extra={'status_message': True})

    def exit_fullscreen(self):
        """Restore windowed mode."""
        window = self.gui.window
        if not window or not self.is_active:
            return

        self.is_active = False
        self._transition_time = time.time()

        # Stop the native-resolution decoder thread
        if self._display_stop is not None:
            self._display_stop.set()
        if self._display_thread is not None:
            self._display_thread.join(timeout=2.0)
            self._display_thread = None
        self._display_stop = None
        with self._display_frame_lock:
            self._pending_display_frame = None

        # Delete the fullscreen GL texture
        if self._fullscreen_tex_id:
            try:
                gl.glDeleteTextures([self._fullscreen_tex_id])
            except Exception:
                pass
            self._fullscreen_tex_id = 0
            self._fullscreen_tex_w = 0
            self._fullscreen_tex_h = 0

        self._display_is_vr = False

        # Return to windowed mode — passing None as monitor restores windowed
        sx, sy = self._saved_pos
        sw, sh = self._saved_size
        glfw.set_window_monitor(window, None, sx, sy, sw, sh, 0)

        # Reset the controls overlay so it is visible immediately on return and
        # the position rect is recalculated on the very first windowed frame.
        if hasattr(self.gui, 'video_display_ui'):
            vdu = self.gui.video_display_ui
            vdu._controls_last_activity_time = time.monotonic()
            vdu._actual_video_image_rect_on_screen = {
                'min_x': 0, 'min_y': 0, 'max_x': 0, 'max_y': 0, 'w': 0, 'h': 0,
            }

        self.app.logger.info("Exited fullscreen", extra={'status_message': True})

    def toggle(self):
        # Debounce: ignore rapid toggle calls (e.g. key repeat)
        if time.time() - self._transition_time < 0.3:
            return
        if self.is_active:
            self.exit_fullscreen()
        else:
            self.enter_fullscreen()

    def render(self):
        """Render the fullscreen UI. Called instead of normal UI when active."""
        now = time.time()

        # Debounce: skip exit key detection for 0.3s after entering fullscreen
        # This prevents glfw.get_key() from detecting a still-held F11 key
        if now - self._transition_time > 0.3:
            window = self.gui.window
            if window:
                if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
                    self.exit_fullscreen()
                    return

        # Mouse tracking for auto-hide
        self._handle_mouse_for_controls()

        io = imgui.get_io()
        display_w, display_h = io.display_size

        # Full-window panel: no padding, no decoration, pure black background
        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(display_w, display_h)
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (0, 0))
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, 0.0, 0.0, 0.0, 1.0)

        flags = (
            imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE |
            imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE |
            imgui.WINDOW_NO_SCROLLBAR
        )

        imgui.begin("##FullscreenVideo", True, flags)

        self._render_fullscreen_video(display_w, display_h)

        # Controls rendered inside the fullscreen window
        elapsed = now - self._last_mouse_move_time
        if elapsed < self._controls_timeout:
            fade_start = self._controls_timeout - 1.0
            alpha = 1.0 - (elapsed - fade_start) / 1.0 if elapsed > fade_start else 1.0
            self._render_controls(display_w, display_h, alpha)

        imgui.end()
        imgui.pop_style_color()  # COLOR_WINDOW_BACKGROUND
        imgui.pop_style_var()    # STYLE_WINDOW_PADDING

    # ------------------------------------------------------------------ private

    def _get_current_monitor(self, window):
        """Return the monitor that contains the center of the window."""
        wx, wy = glfw.get_window_pos(window)
        ww, wh = glfw.get_window_size(window)
        cx, cy = wx + ww // 2, wy + wh // 2

        monitors = glfw.get_monitors()
        for mon in monitors:
            mx, my = glfw.get_monitor_pos(mon)
            mode = glfw.get_video_mode(mon)
            if mx <= cx < mx + mode.size.width and my <= cy < my + mode.size.height:
                return mon

        return glfw.get_primary_monitor()

    def _handle_mouse_for_controls(self):
        """Show controls on mouse move, hide after timeout."""
        io = imgui.get_io()
        mx, my = io.mouse_pos
        now = time.time()

        dx = abs(mx - self._last_mouse_pos[0])
        dy = abs(my - self._last_mouse_pos[1])

        if dx > 2 or dy > 2 or imgui.is_mouse_clicked(0) or imgui.is_mouse_clicked(1):
            self._last_mouse_move_time = now

        self._last_mouse_pos = (mx, my)

    # ---------------------------------------------------------------- display

    def _display_decode_loop(self):
        """Background thread: decode current frame capped at _DISPLAY_MAX_DIM px.

        Only runs for non-VR content.  Polls VideoProcessor.current_frame_index
        and decodes that frame via OpenCV VideoCapture.  Sequential reads are
        fast (no seek needed); random seeks only happen on jumps.  Decoded
        frames are resized to fit within _DISPLAY_MAX_DIM and placed into
        _pending_display_frame for the render thread to upload to GL.
        """
        proc = self.app.processor
        if not proc or not proc.video_path:
            return

        try:
            cap = cv2.VideoCapture(proc.video_path)
        except Exception as exc:
            self.app.logger.warning(f"FSDisplayDecode: could not open VideoCapture: {exc}")
            return

        if not cap.isOpened():
            self.app.logger.warning("FSDisplayDecode: VideoCapture failed to open video")
            return

        # Determine decode dimensions: cap the long edge at _DISPLAY_MAX_DIM
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        long_edge = max(src_w, src_h, 1)
        if long_edge > _DISPLAY_MAX_DIM:
            scale = _DISPLAY_MAX_DIM / long_edge
            dec_w = max(1, int(src_w * scale))
            dec_h = max(1, int(src_h * scale))
            needs_resize = True
        else:
            dec_w, dec_h = src_w, src_h
            needs_resize = False

        self.app.logger.info(
            f"FSDisplayDecode: {src_w}x{src_h} source"
            f"{f' → {dec_w}x{dec_h}' if needs_resize else ' (native)'}"
        )

        last_decoded_index = -1

        try:
            while not self._display_stop.is_set():
                current_index = getattr(proc, 'current_frame_index', 0)

                if current_index == last_decoded_index:
                    time.sleep(0.005)  # poll at ~200 Hz
                    continue

                # Seek only when non-sequential (avoid slow key-frame seeks in
                # the common sequential-playback case).
                if current_index != last_decoded_index + 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_index)

                ret, frame = cap.read()
                if ret and frame is not None:
                    if needs_resize:
                        frame = cv2.resize(frame, (dec_w, dec_h),
                                           interpolation=cv2.INTER_AREA)
                    # Convert BGR → RGB once here; keeps the render thread lean
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    fh, fw = rgb.shape[:2]
                    payload = rgb.tobytes()
                    with self._display_frame_lock:
                        self._pending_display_frame = (fw, fh, payload)
                    last_decoded_index = current_index
                else:
                    time.sleep(0.005)
        finally:
            cap.release()

    def _upload_pending_display_frame(self):
        """Upload a pending native-res frame to the fullscreen GL texture.

        Must be called from the render (main) thread.
        """
        with self._display_frame_lock:
            pending = self._pending_display_frame
            self._pending_display_frame = None

        if pending is None or not self._fullscreen_tex_id:
            return

        fw, fh, payload = pending
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._fullscreen_tex_id)
        if fw != self._fullscreen_tex_w or fh != self._fullscreen_tex_h:
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D, 0, gl.GL_RGB, fw, fh, 0,
                gl.GL_RGB, gl.GL_UNSIGNED_BYTE, payload,
            )
            self._fullscreen_tex_w = fw
            self._fullscreen_tex_h = fh
        else:
            gl.glTexSubImage2D(
                gl.GL_TEXTURE_2D, 0, 0, 0, fw, fh,
                gl.GL_RGB, gl.GL_UNSIGNED_BYTE, payload,
            )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    # ----------------------------------------------------------------- render

    def _render_fullscreen_video(self, display_w, display_h):
        """Render the video texture centred with correct aspect ratio."""
        proc = self.app.processor
        app_state = self.app.app_state_ui

        # Upload any new frame produced by the background decode thread
        if not self._display_is_vr:
            self._upload_pending_display_frame()

        # ------------------------------------------------------------------
        # Choose texture and frame dimensions
        # ------------------------------------------------------------------
        # Priority:
        #   1. Native (capped) decode texture — 2D content only
        #   2. GPU-unwarped frame_texture_id  — VR, or 2D before first frame
        # For VR we always use frame_texture_id because it already contains the
        # correctly unwarped single-eye view at whatever resolution the GPU
        # unwarp produced.
        # ------------------------------------------------------------------

        if not self._display_is_vr and self._fullscreen_tex_id and self._fullscreen_tex_w > 0:
            # --- Native-decode path (2D content) ---
            texture_id = self._fullscreen_tex_id
            frame_w = self._fullscreen_tex_w
            frame_h = self._fullscreen_tex_h
            uv0 = (0.0, 0.0)
            uv1 = (1.0, 1.0)

        else:
            # --- GPU-unwarped / fallback path ---
            texture_id = self.gui.frame_texture_id
            if not texture_id or not (proc and proc.current_frame is not None):
                center_x = display_w / 2
                center_y = display_h / 2
                text = "No video loaded"
                text_size = imgui.calc_text_size(text)
                imgui.set_cursor_pos((center_x - text_size.x / 2,
                                      center_y - text_size.y / 2))
                imgui.text_colored(text, 0.5, 0.5, 0.5, 1.0)
                return

            if self._display_is_vr:
                # VR: use the actual unwarped-frame dimensions for aspect ratio.
                # current_frame is already the single-eye unwarped view.
                frame = proc.current_frame
                frame_h, frame_w = frame.shape[:2]
            else:
                # 2D: first frame not decoded yet — use native video_info dims
                # so the letterbox is correct even for the preview frame.
                if proc.video_info:
                    frame_w = proc.video_info.get('width', 0)
                    frame_h = proc.video_info.get('height', 0)
                if not frame_w or not frame_h:
                    frame = proc.current_frame
                    frame_h, frame_w = frame.shape[:2]

            uv0_x, uv0_y, uv1_x, uv1_y = app_state.get_video_uv_coords()
            uv0 = (uv0_x, uv0_y)
            uv1 = (uv1_x, uv1_y)

        if frame_w <= 0 or frame_h <= 0:
            return

        # Fit frame onto screen maintaining aspect ratio (letter/pillar-box) --
        frame_aspect = frame_w / frame_h
        screen_aspect = display_w / display_h

        if frame_aspect > screen_aspect:
            vid_w = display_w
            vid_h = display_w / frame_aspect
        else:
            vid_h = display_h
            vid_w = display_h * frame_aspect

        offset_x = (display_w - vid_w) / 2
        offset_y = (display_h - vid_h) / 2

        imgui.set_cursor_pos((offset_x, offset_y))
        imgui.image(texture_id, vid_w, vid_h, uv0, uv1)

        # Resolution info overlay (top-right corner)
        if self._display_is_vr:
            src_label = "VR (unwarped)"
        elif self._fullscreen_tex_w > 0:
            src_label = "native" if (self._fullscreen_tex_w == frame_w) else "scaled"
        else:
            src_label = "preview"
        res_text = f"{frame_w}x{frame_h} ({src_label}) -> {int(vid_w)}x{int(vid_h)}"
        text_size = imgui.calc_text_size(res_text)
        draw_list = imgui.get_window_draw_list()
        win_pos = imgui.get_window_position()
        tx = win_pos[0] + display_w - text_size.x - 10
        ty = win_pos[1] + 10
        draw_list.add_rect_filled(
            tx - 4, ty - 2, tx + text_size.x + 4, ty + text_size.y + 2,
            imgui.get_color_u32_rgba(0.0, 0.0, 0.0, 0.5), rounding=3.0,
        )
        draw_list.add_text(tx, ty, imgui.get_color_u32_rgba(0.7, 0.7, 0.7, 0.8), res_text)

    def _render_controls(self, display_w, display_h, alpha=1.0):
        """Semi-transparent control bar at the bottom of the fullscreen window."""
        event_handlers = self.app.event_handlers
        icon_mgr = get_icon_texture_manager()

        btn_size = 32
        spacing = 8
        padding = 10.0
        # 8 buttons total
        bar_w = (btn_size * 8) + (spacing * 7) + spacing
        bar_x = (display_w - bar_w) / 2
        bar_y = display_h - btn_size - padding * 4

        # Semi-transparent background behind controls
        draw_list = imgui.get_window_draw_list()
        win_pos = imgui.get_window_position()
        abs_x = win_pos[0] + bar_x
        abs_y = win_pos[1] + bar_y
        draw_list.add_rect_filled(
            abs_x - padding, abs_y - padding,
            abs_x + bar_w + padding, abs_y + btn_size + padding,
            imgui.get_color_u32_rgba(0.0, 0.0, 0.0, 0.6 * alpha),
            rounding=8.0
        )

        imgui.set_cursor_pos((bar_x, bar_y))
        imgui.push_style_var(imgui.STYLE_ALPHA, alpha)
        imgui.begin_group()

        is_playing = (self.app.processor and self.app.processor.is_processing
                      and not self.app.processor.pause_event.is_set())

        # Jump Start
        tex, _, _ = icon_mgr.get_icon_texture('jump-start.png')
        if tex and imgui.image_button(tex, btn_size, btn_size):
            event_handlers.handle_playback_control("jump_start")
        elif not tex and imgui.button("|<##FSStart", width=btn_size):
            event_handlers.handle_playback_control("jump_start")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Jump to Start")
        imgui.same_line(spacing=spacing)

        # Prev Frame
        tex, _, _ = icon_mgr.get_icon_texture('prev-frame.png')
        if tex and imgui.image_button(tex, btn_size, btn_size):
            event_handlers.handle_playback_control("prev_frame")
        elif not tex and imgui.button("<<##FSPrev", width=btn_size):
            event_handlers.handle_playback_control("prev_frame")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Previous Frame")
        imgui.same_line(spacing=spacing)

        # Play/Pause
        pp_icon = 'pause.png' if is_playing else 'play.png'
        pp_fallback = "||" if is_playing else ">"
        tex, _, _ = icon_mgr.get_icon_texture(pp_icon)
        if tex and imgui.image_button(tex, btn_size, btn_size):
            event_handlers.handle_playback_control("play_pause")
        elif not tex and imgui.button(f"{pp_fallback}##FSPlayPause", width=btn_size):
            event_handlers.handle_playback_control("play_pause")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Play / Pause (SPACE)")
        imgui.same_line(spacing=spacing)

        # Stop
        tex, _, _ = icon_mgr.get_icon_texture('stop.png')
        if tex and imgui.image_button(tex, btn_size, btn_size):
            event_handlers.handle_playback_control("stop")
        elif not tex and imgui.button("[]##FSStop", width=btn_size):
            event_handlers.handle_playback_control("stop")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Stop Playback")
        imgui.same_line(spacing=spacing)

        # Next Frame
        tex, _, _ = icon_mgr.get_icon_texture('next-frame.png')
        if tex and imgui.image_button(tex, btn_size, btn_size):
            event_handlers.handle_playback_control("next_frame")
        elif not tex and imgui.button(">>##FSNext", width=btn_size):
            event_handlers.handle_playback_control("next_frame")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Next Frame")
        imgui.same_line(spacing=spacing)

        # Jump End
        tex, _, _ = icon_mgr.get_icon_texture('jump-end.png')
        if tex and imgui.image_button(tex, btn_size, btn_size):
            event_handlers.handle_playback_control("jump_end")
        elif not tex and imgui.button(">|##FSEnd", width=btn_size):
            event_handlers.handle_playback_control("jump_end")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Jump to End")
        imgui.same_line(spacing=spacing * 2)

        # Exit Fullscreen
        imgui.push_style_color(imgui.COLOR_BUTTON, 0.8, 0.2, 0.2, 0.8)
        tex, _, _ = icon_mgr.get_icon_texture('fullscreen-exit.png')
        if tex and imgui.image_button(tex, btn_size, btn_size):
            self.exit_fullscreen()
        elif not tex and imgui.button("X##FSExit", width=btn_size):
            self.exit_fullscreen()
        imgui.pop_style_color()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Exit Fullscreen (ESC / F11)")

        imgui.end_group()
        imgui.pop_style_var()  # STYLE_ALPHA
