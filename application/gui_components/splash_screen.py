"""
Animated splash screen for application startup.
Inspired by the HTML VR viewer's splash screen design.
"""
import imgui
import OpenGL.GL as gl
import glfw
from imgui.integrations.glfw import GlfwRenderer
from PIL import Image
import numpy as np
import os
import time
import math
import threading


class SplashScreen:
    """Animated splash screen with logo, title, and loading animation."""

    def __init__(self, app_logic):
        self.app = app_logic
        self.active = True
        self.start_time = time.time()
        self.fade_out_start = None
        self.fade_out_duration = 0.5  # Fade out over 0.5 seconds

        # Animation parameters
        self.logo_float_speed = 2.0  # Float animation speed
        self.logo_float_amplitude = 10.0  # Pixels to float up/down
        self.title_glow_speed = 1.5  # Glow animation speed
        self.progress_speed = 0.3  # Progress bar animation speed

        # Display settings
        self.display_duration = 2.0  # Show splash for 2 seconds minimum
        self.logo_texture = None
        self.logo_size = (200, 200)  # Logo display size

        # Status messages
        self.status_messages = [
            "Initializing...",
            "Loading AI models...",
            "Preparing workspace...",
            "Ready!"
        ]
        self.current_status_index = 0
        self.last_status_update = time.time()
        self.status_update_interval = 0.5  # Change status every 0.5 seconds

    def load_logo_texture(self):
        """Load the logo texture for display."""
        try:
            # Get logo path (same as used by 3D simulator)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            logo_path = os.path.join(script_dir, '..', '..', 'assets', 'branding', 'logo.png')

            if not os.path.exists(logo_path):
                self.app.logger.warning(f"Splash screen logo not found: {logo_path}")
                return

            # Load with PIL
            img = Image.open(logo_path)
            img = img.convert("RGBA")
            img_data = np.array(img, dtype=np.uint8)

            # Create OpenGL texture
            self.logo_texture = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.logo_texture)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, img.width, img.height,
                          0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, img_data)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

            self.app.logger.debug(f"Splash screen logo loaded from {logo_path}")

        except Exception as e:
            self.app.logger.warning(f"Failed to load splash screen logo: {e}")

    def should_close(self):
        """Check if splash screen should close."""
        if self.fade_out_start is not None:
            # Fade out in progress
            elapsed = time.time() - self.fade_out_start
            return elapsed >= self.fade_out_duration

        # Auto-close after display duration
        elapsed = time.time() - self.start_time
        if elapsed >= self.display_duration:
            if self.fade_out_start is None:
                self.fade_out_start = time.time()

        return False

    def get_alpha(self):
        """Get current alpha value for fade in/out."""
        elapsed = time.time() - self.start_time

        # Fade in over first 0.3 seconds
        if elapsed < 0.3:
            return elapsed / 0.3

        # Fade out
        if self.fade_out_start is not None:
            fade_elapsed = time.time() - self.fade_out_start
            return max(0.0, 1.0 - (fade_elapsed / self.fade_out_duration))

        return 1.0

    def update_status(self):
        """Update the status message based on time."""
        current_time = time.time()
        if current_time - self.last_status_update >= self.status_update_interval:
            self.current_status_index = min(
                self.current_status_index + 1,
                len(self.status_messages) - 1
            )
            self.last_status_update = current_time

    def render(self, window_width, window_height):
        """Render the splash screen as a full-screen modal."""
        if not self.active:
            return

        # Check if we should close
        if self.should_close():
            self.active = False
            return

        # Update status message
        self.update_status()

        # Get alpha for fade in/out
        alpha = self.get_alpha()
        if alpha <= 0:
            self.active = False
            return

        # Get current time for animations
        current_time = time.time() - self.start_time

        # Calculate animation values
        # Logo float: sine wave oscillation
        logo_float_offset = math.sin(current_time * self.logo_float_speed) * self.logo_float_amplitude

        # Title glow: pulsing opacity
        title_glow = 0.7 + 0.3 * math.sin(current_time * self.title_glow_speed)

        # Progress bar: continuous animation
        progress = (current_time * self.progress_speed) % 1.0

        # Full-screen dark overlay
        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(window_width, window_height)
        imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 0.0)
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (0.0, 0.0))

        # Dark background with gradient effect (using dark blue/purple colors)
        bg_color = (0.04 * alpha, 0.04 * alpha, 0.06 * alpha, 0.95 * alpha)
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, *bg_color)
        imgui.push_style_color(imgui.COLOR_BORDER, 0.0, 0.0, 0.0, 0.0)

        window_flags = (
            imgui.WINDOW_NO_TITLE_BAR |
            imgui.WINDOW_NO_RESIZE |
            imgui.WINDOW_NO_MOVE |
            imgui.WINDOW_NO_SCROLLBAR |
            imgui.WINDOW_NO_SCROLL_WITH_MOUSE |
            imgui.WINDOW_NO_COLLAPSE |
            imgui.WINDOW_NO_NAV
        )

        imgui.begin("##SplashScreen", flags=window_flags)

        # Center content vertically
        content_height = 400  # Approximate total content height
        start_y = (window_height - content_height) / 2

        imgui.set_cursor_pos_y(start_y + logo_float_offset)

        # Draw logo if available
        if self.logo_texture is not None:
            logo_x = (window_width - self.logo_size[0]) / 2
            imgui.set_cursor_pos_x(logo_x)

            # Add subtle drop shadow effect (draw logo slightly offset in darker color first)
            shadow_offset = 5
            imgui.set_cursor_pos((logo_x + shadow_offset, imgui.get_cursor_pos_y() + shadow_offset))
            imgui.image(self.logo_texture, self.logo_size[0], self.logo_size[1],
                       tint_color=(0, 0, 0, 0.5 * alpha))

            # Draw actual logo
            imgui.set_cursor_pos((logo_x, start_y + logo_float_offset))
            imgui.image(self.logo_texture, self.logo_size[0], self.logo_size[1],
                       tint_color=(1, 1, 1, alpha))

        # Spacing
        imgui.dummy(1, 30)

        # Title: "FUNGEN" with gradient-like effect (cyan to purple)
        title_text = "FUNGEN"
        title_font_size = imgui.get_font_size() * 3.0

        # Calculate text width for centering
        # Approximate width (ImGui doesn't support exact multi-colored text width calc easily)
        char_width = title_font_size * 0.6  # Approximate
        title_width = len(title_text) * char_width
        title_x = (window_width - title_width) / 2

        imgui.set_cursor_pos_x(title_x)

        # Draw title with glow effect (multiple passes)
        draw_list = imgui.get_window_draw_list()
        cursor_pos = imgui.get_cursor_screen_pos()

        # Glow layers (multiple passes with increasing size and decreasing opacity)
        glow_color = imgui.get_color_u32_rgba(0.0, 0.83, 1.0, 0.3 * title_glow * alpha)
        for i in range(3):
            glow_offset = (3 - i) * 2
            draw_list.add_text(
                cursor_pos[0] - glow_offset, cursor_pos[1],
                glow_color, title_text
            )
            draw_list.add_text(
                cursor_pos[0] + glow_offset, cursor_pos[1],
                glow_color, title_text
            )

        # Main title text (gradient from cyan to purple - using cyan for now)
        title_color = imgui.get_color_u32_rgba(0.0, 0.83, 1.0, title_glow * alpha)

        # Scale up font for title (use text with custom size)
        imgui.push_style_color(imgui.COLOR_TEXT, 0.0, 0.83, 1.0, title_glow * alpha)

        # Since ImGui doesn't easily support per-character colors, use a single color
        # We'll use a cyan-to-purple blend color
        blend_color = (0.3, 0.6, 1.0, title_glow * alpha)
        imgui.push_style_color(imgui.COLOR_TEXT, *blend_color)

        imgui.set_cursor_pos_x(title_x)

        # Draw large title text
        for char in title_text:
            imgui.text(char)
            imgui.same_line()

        imgui.pop_style_color(2)

        imgui.new_line()

        # Spacing
        imgui.dummy(1, 20)

        # Loading bar
        bar_width = 400
        bar_height = 6
        bar_x = (window_width - bar_width) / 2

        imgui.set_cursor_pos_x(bar_x)

        # Draw loading bar background
        cursor_screen_pos = imgui.get_cursor_screen_pos()
        draw_list = imgui.get_window_draw_list()

        # Background bar (dark)
        bg_bar_color = imgui.get_color_u32_rgba(0.2, 0.2, 0.3, 0.5 * alpha)
        draw_list.add_rect_filled(
            cursor_screen_pos[0], cursor_screen_pos[1],
            cursor_screen_pos[0] + bar_width, cursor_screen_pos[1] + bar_height,
            bg_bar_color, rounding=3.0
        )

        # Progress bar (gradient from cyan to purple)
        progress_width = bar_width * progress
        progress_color = imgui.get_color_u32_rgba(0.0, 0.83, 1.0, alpha)
        draw_list.add_rect_filled(
            cursor_screen_pos[0], cursor_screen_pos[1],
            cursor_screen_pos[0] + progress_width, cursor_screen_pos[1] + bar_height,
            progress_color, rounding=3.0
        )

        # Animated shine effect on progress bar
        shine_width = 50
        shine_x = progress_width - shine_width if progress_width > shine_width else 0
        if progress_width > 0:
            shine_color = imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 0.3 * alpha)
            draw_list.add_rect_filled(
                cursor_screen_pos[0] + shine_x, cursor_screen_pos[1],
                cursor_screen_pos[0] + min(progress_width, shine_x + shine_width),
                cursor_screen_pos[1] + bar_height,
                shine_color, rounding=3.0
            )

        imgui.dummy(1, bar_height)

        # Spacing
        imgui.dummy(1, 20)

        # Status message
        status_text = self.status_messages[self.current_status_index]
        status_width = imgui.calc_text_size(status_text)[0]
        status_x = (window_width - status_width) / 2

        imgui.set_cursor_pos_x(status_x)

        # Pulsing status text
        status_alpha = 0.6 + 0.4 * math.sin(current_time * 2.0)
        imgui.push_style_color(imgui.COLOR_TEXT, 0.7, 0.7, 0.8, status_alpha * alpha)
        imgui.text(status_text)
        imgui.pop_style_color()

        # Spacing
        imgui.dummy(1, 30)

        # "Click anywhere to continue" hint (after 1 second)
        if current_time > 1.0:
            hint_text = "Click anywhere to continue..."
            hint_width = imgui.calc_text_size(hint_text)[0]
            hint_x = (window_width - hint_width) / 2

            imgui.set_cursor_pos_x(hint_x)

            hint_alpha = 0.4 + 0.2 * math.sin(current_time * 3.0)
            imgui.push_style_color(imgui.COLOR_TEXT, 0.5, 0.5, 0.6, hint_alpha * alpha)
            imgui.text(hint_text)
            imgui.pop_style_color()

            # Check for click to dismiss
            if imgui.is_mouse_clicked(0):
                if self.fade_out_start is None:
                    self.fade_out_start = time.time()

        imgui.end()
        imgui.pop_style_color(2)
        imgui.pop_style_var(2)

    def cleanup(self):
        """Clean up resources."""
        if self.logo_texture is not None:
            gl.glDeleteTextures([self.logo_texture])
            self.logo_texture = None


class StandaloneSplashWindow:
    """
    Standalone splash window for early startup (before main GUI window).
    Runs in a separate thread to display during ApplicationLogic initialization.
    """

    def __init__(self):
        self.window = None
        self.impl = None
        self.splash_screen = None
        self.running = False
        self.thread = None
        self.status_message = "Initializing..."
        self.status_lock = threading.Lock()
        self.logo_texture = None
        self.emoji_textures = {}  # Store emoji textures

    def _init_window(self):
        """Initialize a minimal GLFW window for the splash screen."""
        if not glfw.init():
            return False

        # Window hints for a borderless, non-resizable splash window
        glfw.window_hint(glfw.DECORATED, glfw.FALSE)  # No title bar or borders
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        glfw.window_hint(glfw.TRANSPARENT_FRAMEBUFFER, glfw.TRUE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

        # Get primary monitor for FULLSCREEN effect
        monitor = glfw.get_primary_monitor()
        if monitor:
            mode = glfw.get_video_mode(monitor)
            splash_width = mode.size.width
            splash_height = mode.size.height
        else:
            # Fallback if no monitor detected
            splash_width = 1920
            splash_height = 1080

        # Create FULLSCREEN borderless window for maximum dramatic effect
        self.window = glfw.create_window(splash_width, splash_height, "FunGen", None, None)
        if not self.window:
            glfw.terminate()
            return False

        # Position at top-left corner (0, 0) for fullscreen coverage
        glfw.set_window_pos(self.window, 0, 0)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)  # Enable vsync

        # Initialize ImGui
        imgui.create_context()
        self.impl = GlfwRenderer(self.window)

        # Load logo texture after OpenGL context is created
        self._load_logo_texture()
        self._load_emoji_textures()

        return True

    def _load_logo_texture(self):
        """Load the logo texture for display."""
        try:
            import os
            # Get logo path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            logo_path = os.path.join(script_dir, '..', '..', 'assets', 'branding', 'logo.png')

            if not os.path.exists(logo_path):
                return

            # Load with PIL
            img = Image.open(logo_path)
            img = img.convert("RGBA")
            img_data = np.array(img, dtype=np.uint8)

            # Create OpenGL texture
            self.logo_texture = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.logo_texture)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, img.width, img.height,
                          0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, img_data)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        except Exception as e:
            print(f"Failed to load splash screen logo: {e}")

    def _load_emoji_textures(self):
        """Load emoji textures for display in laser circles (only loads available emojis)."""
        try:
            import os
            from config.constants import SPLASH_EMOJI_URLS

            script_dir = os.path.dirname(os.path.abspath(__file__))
            assets_dir = os.path.join(script_dir, '..', '..', 'assets')

            # Build emoji name mapping from SPLASH_EMOJI_URLS
            # Keys will be the filename without extension
            for filename in SPLASH_EMOJI_URLS.keys():
                emoji_path = os.path.join(assets_dir, filename)
                if not os.path.exists(emoji_path):
                    continue

                # Use filename without extension as the name key
                name = os.path.splitext(filename)[0]

                # Load with PIL
                img = Image.open(emoji_path)
                img = img.convert("RGBA")
                img_data = np.array(img, dtype=np.uint8)

                # Create OpenGL texture
                texture_id = gl.glGenTextures(1)
                gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
                gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, img.width, img.height,
                              0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, img_data)
                gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

                self.emoji_textures[name] = texture_id

            # Note: Using print() here since StandaloneSplashWindow runs before app logger is available
            # Emoji loading is non-critical, so we silently continue if none are available

        except Exception as e:
            print(f"Failed to load emoji textures: {e}")


    def _render_loop(self):
        """Main render loop for the splash window."""
        try:
            while self.running and not glfw.window_should_close(self.window):
                glfw.poll_events()
                if self.impl:
                    self.impl.process_inputs()

                gl.glClearColor(0.0, 0.0, 0.0, 0.0)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT)

                imgui.new_frame()

                # Render splash screen content
                self._render_splash_content()

                imgui.render()
                if self.impl:
                    self.impl.render(imgui.get_draw_data())

                glfw.swap_buffers(self.window)

        except Exception as e:
            print(f"Splash window error: {e}")
        finally:
            self._cleanup()

    def _render_splash_content(self):
        """Render the splash screen content (minimalist: just logo)."""
        # Get window size
        window_width, window_height = glfw.get_window_size(self.window)

        # Full-screen window with transparent background
        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(window_width, window_height)
        imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 0.0)
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (0.0, 0.0))

        # Fully transparent background
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, 0.0, 0.0, 0.0, 0.0)
        imgui.push_style_color(imgui.COLOR_BORDER, 0.0, 0.0, 0.0, 0.0)

        window_flags = (
            imgui.WINDOW_NO_TITLE_BAR |
            imgui.WINDOW_NO_RESIZE |
            imgui.WINDOW_NO_MOVE |
            imgui.WINDOW_NO_SCROLLBAR |
            imgui.WINDOW_NO_SCROLL_WITH_MOUSE |
            imgui.WINDOW_NO_COLLAPSE |
            imgui.WINDOW_NO_NAV
        )

        imgui.begin("##StandaloneSplash", flags=window_flags)

        # Current time for animations
        current_time = time.time()

        # Center logo vertically and horizontally
        if self.logo_texture is not None:
            logo_size = 250  # Logo fills most of the window
            logo_x = (window_width - logo_size) / 2
            logo_y = (window_height - logo_size) / 2

            # Gentle floating animation
            float_offset = math.sin(current_time * 2.0) * 8.0

            # Fade in animation (first 0.3 seconds)
            fade_alpha = min(1.0, current_time / 0.3) if current_time < 0.3 else 1.0

            imgui.set_cursor_pos((logo_x, logo_y + float_offset))
            imgui.image(self.logo_texture, logo_size, logo_size, tint_color=(1, 1, 1, fade_alpha))

            # Draw lasers AFTER logo (so they appear IN FRONT)
            if current_time > 0.3:  # Start lasers after fade-in
                self._render_laser_eyes(logo_x, logo_y + float_offset, logo_size, current_time - 0.3)

            # "Loading FunGen..." text below the logo
            if current_time > 0.3:  # Show after fade-in
                loading_text = "Loading FunGen..."

                # Use draw list for manual positioning
                text_size = imgui.calc_text_size(loading_text)
                text_x = (window_width - text_size[0]) / 2
                text_y = logo_y + logo_size + float_offset + 30  # Below logo with spacing

                # Pulsing animation
                text_pulse = 0.6 + 0.4 * math.sin((current_time - 0.3) * 3.0)

                # Get draw list for manual text rendering
                draw_list_text = imgui.get_window_draw_list()

                # Glow effect (multiple layers)
                for i in range(3, 0, -1):
                    glow_offset = i * 2
                    glow_alpha = (0.3 * fade_alpha * text_pulse) / (i * 1.5)
                    glow_color = imgui.get_color_u32_rgba(1.0, 0.3, 0.0, glow_alpha)
                    draw_list_text.add_text(text_x - glow_offset, text_y, glow_color, loading_text)
                    draw_list_text.add_text(text_x + glow_offset, text_y, glow_color, loading_text)
                    draw_list_text.add_text(text_x, text_y - glow_offset, glow_color, loading_text)
                    draw_list_text.add_text(text_x, text_y + glow_offset, glow_color, loading_text)

                # Main text (bright white/yellow)
                text_color = imgui.get_color_u32_rgba(1.0, 0.95, 0.8, 0.9 * fade_alpha * text_pulse)
                draw_list_text.add_text(text_x, text_y, text_color, loading_text)

        imgui.end()
        imgui.pop_style_color(2)
        imgui.pop_style_var(2)

    def _render_laser_eyes(self, logo_x, logo_y, logo_size, laser_time):
        """Render epic RED SCANNING CONES from the logo's eyes."""
        draw_list = imgui.get_window_draw_list()
        window_width, window_height = glfw.get_window_size(self.window)

        # Eye positions (approximate - adjust these based on your logo)
        logo_center_x = logo_x + logo_size / 2
        logo_center_y = logo_y + logo_size / 2

        # Eyes positioned at the actual eye location on the logo
        eye_y = logo_center_y - logo_size * 0.05  # Slightly above center
        left_eye_x = logo_center_x - logo_size * 0.12  # Closer to center
        right_eye_x = logo_center_x + logo_size * 0.12  # Closer to center

        # MULTI-PATTERN SCANNING: Alternate between 3 different scanning patterns
        # Each pattern runs for 15 seconds, then switches to the next
        pattern_duration = 15.0
        pattern_index = int(laser_time / pattern_duration) % 3
        pattern_time = laser_time % pattern_duration
        t = pattern_time / pattern_duration  # Normalized time (0 to 1)

        screen_center_x = window_width / 2
        screen_center_y = window_height / 2

        if pattern_index == 0:
            # PATTERN 1: SPIRAL - Outward and back
            # 3 rotations outward (0 to 0.6), then 2 rotations back (0.6 to 1.0)
            if t < 0.6:
                # Spiraling outward (3 rotations)
                progress = t / 0.6
                angle = progress * 3 * 2 * math.pi
                radius_factor = progress
            else:
                # Spiraling back inward (2 rotations)
                progress = (t - 0.6) / 0.4
                angle = (1.0 - progress) * 2 * 2 * math.pi + 3 * 2 * math.pi
                radius_factor = 1.0 - progress

            # Calculate position
            max_radius_x = window_width * 0.45
            max_radius_y = window_height * 0.45
            target_x_center = screen_center_x + math.cos(angle) * max_radius_x * radius_factor
            target_y = screen_center_y + math.sin(angle) * max_radius_y * radius_factor

        elif pattern_index == 1:
            # PATTERN 2: FIGURE-8 (Lemniscate of Gerono)
            angle = t * 2 * math.pi * 2  # 2 full loops
            scale_x = window_width * 0.35
            scale_y = window_height * 0.35
            target_x_center = screen_center_x + math.cos(angle) * scale_x
            target_y = screen_center_y + math.sin(angle) * math.cos(angle) * scale_y

        else:
            # PATTERN 3: HORIZONTAL SWEEP with vertical oscillation
            # Sweep left to right and back
            if t < 0.5:
                h_progress = t / 0.5
            else:
                h_progress = 1.0 - (t - 0.5) / 0.5

            # Horizontal position
            target_x_center = window_width * 0.1 + h_progress * window_width * 0.8

            # Vertical oscillation (3 waves during the sweep)
            v_wave = math.sin(h_progress * 3 * 2 * math.pi)
            target_y = screen_center_y + v_wave * window_height * 0.3

        # Add subtle horizontal drift
        horizontal_drift = math.sin(laser_time * 0.4) * 40

        # Pulsing intensity for dramatic effect
        pulse = 0.8 + 0.2 * math.sin(laser_time * 10.0)

        # COLOR CYCLING: Smooth transition through red → orange → yellow → blue → back
        # Complete cycle every 20 seconds
        color_cycle_time = laser_time / 20.0
        color_phase = color_cycle_time % 1.0  # 0 to 1

        # Calculate RGB color based on phase
        if color_phase < 0.25:
            # Red to Orange (0 to 0.25)
            t = color_phase / 0.25
            laser_r = 1.0
            laser_g = 0.5 * t
            laser_b = 0.0
        elif color_phase < 0.5:
            # Orange to Yellow (0.25 to 0.5)
            t = (color_phase - 0.25) / 0.25
            laser_r = 1.0
            laser_g = 0.5 + 0.5 * t
            laser_b = 0.0
        elif color_phase < 0.75:
            # Yellow to Blue (0.5 to 0.75)
            t = (color_phase - 0.5) / 0.25
            laser_r = 1.0 - 1.0 * t
            laser_g = 1.0 - 0.5 * t
            laser_b = 1.0 * t
        else:
            # Blue to Red (0.75 to 1.0)
            t = (color_phase - 0.75) / 0.25
            laser_r = 0.0 + 1.0 * t
            laser_g = 0.5 - 0.5 * t
            laser_b = 1.0 - 1.0 * t

        # Calculate 3D depth effect - distance from screen center determines perceived depth
        screen_center_x = window_width / 2
        screen_center_y = window_height / 2
        dx_center = target_x_center - screen_center_x
        dy_center = target_y - screen_center_y
        dist_from_center = math.sqrt(dx_center*dx_center + dy_center*dy_center)
        max_distance = math.sqrt(screen_center_x**2 + screen_center_y**2)
        normalized_center_dist = dist_from_center / max_distance  # 0 (center) to 1 (corners)

        # 3D depth: closer objects (center) have MORE eye separation, distant (edges) have LESS
        # This creates stereoscopic depth perception
        min_eye_separation = 15  # Pixels at edges (far away)
        max_eye_separation = 80  # Pixels at center (close)
        eye_separation = max_eye_separation - (max_eye_separation - min_eye_separation) * normalized_center_dist

        # Cone width also scales with depth - bigger when closer (center)
        cone_width_multiplier = 0.45 - 0.2 * normalized_center_dist  # 0.45 at center, 0.25 at edges

        # Draw scanning cone for each eye
        def draw_scanning_cone(eye_x, eye_y, target_x, target_y, eye_offset=0, show_emoji=True):
            # Calculate cone spread angle (wider as it gets farther from eye, and when closer to screen center)
            distance = math.sqrt((target_x - eye_x)**2 + (target_y - eye_y)**2)
            # Ensure minimum distance to prevent degenerate triangles at start
            distance = max(distance, 50)
            cone_width = distance * cone_width_multiplier  # Use dynamic multiplier for 3D effect

            # Calculate perpendicular vector for cone edges
            dx = target_x - eye_x
            dy = target_y - eye_y
            length = math.sqrt(dx*dx + dy*dy)
            if length > 0:
                dx /= length
                dy /= length

            # Perpendicular vector
            perp_x = -dy
            perp_y = dx

            # Cone edge points
            left_edge_x = target_x + perp_x * cone_width
            left_edge_y = target_y + perp_y * cone_width
            right_edge_x = target_x - perp_x * cone_width
            right_edge_y = target_y - perp_y * cone_width

            # Draw multiple layers for glow effect
            # Outer glow (most transparent, widest)
            for i in range(4, 0, -1):
                spread = i * 0.25
                glow_left_x = target_x + perp_x * cone_width * (1 + spread * 0.3)
                glow_left_y = target_y + perp_y * cone_width * (1 + spread * 0.3)
                glow_right_x = target_x - perp_x * cone_width * (1 + spread * 0.3)
                glow_right_y = target_y - perp_y * cone_width * (1 + spread * 0.3)

                alpha = (0.15 * pulse) / i
                glow_color = imgui.get_color_u32_rgba(laser_r, laser_g * 0.5, laser_b, alpha)

                # Draw filled triangle for cone
                draw_list.add_triangle_filled(
                    eye_x, eye_y,
                    glow_left_x, glow_left_y,
                    glow_right_x, glow_right_y,
                    glow_color
                )

            # Core cone (bright, using dynamic color)
            core_alpha = 0.5 * pulse
            core_color = imgui.get_color_u32_rgba(laser_r, laser_g, laser_b, core_alpha)
            draw_list.add_triangle_filled(
                eye_x, eye_y,
                left_edge_x, left_edge_y,
                right_edge_x, right_edge_y,
                core_color
            )

            # Circle size MATCHES the cone width at the endpoint (not independent!)
            circle_radius = cone_width

            # Draw scan target circle at endpoint (slightly transparent)
            # Multiple rings for glow
            for i in range(3, 0, -1):
                ring_alpha = (0.3 * pulse) / i
                ring_color = imgui.get_color_u32_rgba(laser_r, laser_g, laser_b, ring_alpha)
                draw_list.add_circle_filled(target_x, target_y, circle_radius * (1 + i * 0.15), ring_color)

            # Solid center circle (bright)
            center_alpha = 0.7 * pulse
            center_color = imgui.get_color_u32_rgba(laser_r * 0.9, laser_g * 0.9, laser_b * 0.9, center_alpha)
            draw_list.add_circle_filled(target_x, target_y, circle_radius, center_color)

            # Draw emoji inside the circle (alternates every 3 seconds)
            # Only draw emoji if this eye is active
            if show_emoji and self.emoji_textures:
                emoji_cycle = (int(laser_time / 3.0) + eye_offset) % len(self.emoji_textures)
                emoji_names = list(self.emoji_textures.keys())
                if emoji_cycle < len(emoji_names):
                    emoji_name = emoji_names[emoji_cycle]
                    emoji_texture = self.emoji_textures[emoji_name]

                    # Scale emoji to fit inside circle (60% of radius)
                    emoji_size = circle_radius * 1.2
                    emoji_x = target_x - emoji_size / 2
                    emoji_y = target_y - emoji_size / 2

                    # Draw emoji with pulsing alpha
                    emoji_alpha = 0.9 * pulse
                    draw_list.add_image(emoji_texture, (emoji_x, emoji_y),
                                       (emoji_x + emoji_size, emoji_y + emoji_size),
                                       col=imgui.get_color_u32_rgba(1, 1, 1, emoji_alpha))

            # Pulsing ring around scan point
            ring_pulse = math.sin(laser_time * 8.0) * 0.5 + 0.5
            ring_radius = circle_radius * (1.5 + ring_pulse * 0.3)
            ring_alpha = 0.4 * pulse * (1 - ring_pulse)
            ring_color = imgui.get_color_u32_rgba(laser_r, laser_g, laser_b, ring_alpha)
            draw_list.add_circle(target_x, target_y, ring_radius, ring_color, thickness=3.0)

        # Calculate synchronized target positions with 3D depth-based eye separation
        # Both eyes point to same general area, but separation varies for depth effect
        left_target_x = target_x_center + horizontal_drift - eye_separation
        left_target_y = target_y

        right_target_x = target_x_center + horizontal_drift + eye_separation
        right_target_y = target_y

        # Draw bright glow at eye positions (source of the laser)
        eye_glow_radius = 8
        for i in range(3, 0, -1):
            glow_alpha = (0.4 * pulse) / i
            glow_color = imgui.get_color_u32_rgba(laser_r, laser_g, laser_b, glow_alpha)
            draw_list.add_circle_filled(left_eye_x, eye_y, eye_glow_radius * i, glow_color)
            draw_list.add_circle_filled(right_eye_x, eye_y, eye_glow_radius * i, glow_color)

        # Bright core at eyes (slightly desaturated for better effect)
        eye_core_color = imgui.get_color_u32_rgba(
            laser_r * 0.9 + 0.1,
            laser_g * 0.9 + 0.1,
            laser_b * 0.9 + 0.1,
            0.9 * pulse
        )
        draw_list.add_circle_filled(left_eye_x, eye_y, eye_glow_radius * 0.5, eye_core_color)
        draw_list.add_circle_filled(right_eye_x, eye_y, eye_glow_radius * 0.5, eye_core_color)

        # Draw both scanning cones
        # Only the last drawn circle (right eye) shows the emoji
        draw_scanning_cone(left_eye_x, eye_y, left_target_x, left_target_y, eye_offset=0, show_emoji=False)
        draw_scanning_cone(right_eye_x, eye_y, right_target_x, right_target_y, eye_offset=0, show_emoji=True)

        # Add horizontal scan line across entire screen
        scan_line_alpha = 0.3 * pulse
        scan_line_color = imgui.get_color_u32_rgba(laser_r, laser_g, laser_b, scan_line_alpha)
        draw_list.add_line(0, target_y, window_width, target_y, scan_line_color, 2.0)

    def _render_spinner(self, window_width, window_height, current_time):
        """Render an animated loading spinner."""
        spinner_radius = 30
        spinner_thickness = 4
        num_segments = 30

        # Center position
        center_x = window_width / 2
        center_y = imgui.get_cursor_screen_pos()[1] + spinner_radius

        draw_list = imgui.get_window_draw_list()

        # Rotating arc
        rotation = current_time * 3.0  # Rotation speed
        arc_length = math.pi * 1.5  # 270 degrees

        for i in range(num_segments):
            angle = rotation + (i / num_segments) * (2 * math.pi)
            next_angle = rotation + ((i + 1) / num_segments) * (2 * math.pi)

            # Fade effect based on position
            fade = (i / num_segments)

            # Only draw the arc portion
            if fade > 0.25:  # Skip first 25% for arc effect
                x1 = center_x + math.cos(angle) * spinner_radius
                y1 = center_y + math.sin(angle) * spinner_radius
                x2 = center_x + math.cos(next_angle) * spinner_radius
                y2 = center_y + math.sin(next_angle) * spinner_radius

                alpha = fade * 0.8
                color = imgui.get_color_u32_rgba(0.0, 0.83, 1.0, alpha)

                draw_list.add_line(x1, y1, x2, y2, color, spinner_thickness)

        imgui.dummy(1, spinner_radius * 2)

    def _cleanup(self):
        """Clean up resources."""
        # Clean up logo texture
        if self.logo_texture is not None:
            try:
                gl.glDeleteTextures([self.logo_texture])
            except:
                pass
            self.logo_texture = None

        # Clean up emoji textures
        for emoji_texture in self.emoji_textures.values():
            try:
                gl.glDeleteTextures([emoji_texture])
            except:
                pass
        self.emoji_textures.clear()

        if self.impl:
            try:
                self.impl.shutdown()
            except:
                pass
            self.impl = None

        # Destroy ImGui context to avoid conflicts with main window
        try:
            ctx = imgui.get_current_context()
            if ctx is not None:
                imgui.destroy_context(ctx)
        except:
            pass

        if self.window:
            try:
                glfw.destroy_window(self.window)
            except:
                pass
            self.window = None

        # CRITICAL: Reset GLFW window hints to defaults
        # The splash window set DECORATED=FALSE and TRANSPARENT=TRUE
        # These hints persist and will affect the main window!
        try:
            glfw.default_window_hints()
        except:
            pass

        # Don't call glfw.terminate() here - let the main window re-init GLFW
        # The main application will terminate GLFW on final shutdown

    def start(self):
        """Start the splash window in the current thread."""
        self.running = True
        if not self._init_window():
            print("Failed to initialize splash window")
            return False

        self._render_loop()
        return True

    def stop(self):
        """Stop the splash window."""
        self.running = False

    def set_status(self, message):
        """Update the status message (thread-safe)."""
        with self.status_lock:
            self.status_message = message


def show_splash_during_init(init_function, *args, **kwargs):
    """
    Show splash window while running an initialization function.

    Args:
        init_function: Function to run during splash display
        *args, **kwargs: Arguments to pass to init_function

    Returns:
        Result of init_function
    """
    # Note: GLFW must run on the main thread on macOS, so we'll run
    # the initialization in a separate thread instead

    splash = StandaloneSplashWindow()
    result_container = {"result": None, "exception": None}

    def run_init():
        try:
            result_container["result"] = init_function(*args, **kwargs)
        except Exception as e:
            result_container["exception"] = e
        finally:
            splash.stop()

    # Start initialization in a separate thread
    init_thread = threading.Thread(target=run_init, daemon=False)
    init_thread.start()

    # Run splash on main thread (required for macOS)
    splash.start()

    # Wait for initialization to complete
    init_thread.join()

    # Check for exceptions
    if result_container["exception"]:
        raise result_container["exception"]

    return result_container["result"]
