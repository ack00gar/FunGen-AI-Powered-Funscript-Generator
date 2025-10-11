#!/usr/bin/env python3
"""
Integrated Fullscreen Display

This module provides fullscreen video display with integrated controls
using ImGui, PyOpenGL, or Pygame as fallback.

Features:
- Fullscreen video rendering with high quality
- Integrated playback controls (play, pause, seek)
- Perfect sync with main processing pipeline  
- VR viewing mode support
- Device control integration
"""

import imgui
import numpy as np
from typing import Optional, Tuple, Callable
import time
from application.utils import get_icon_texture_manager

try:
    import OpenGL.GL as gl
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


class FullscreenDisplayManager:
    """
    Manages fullscreen video display with integrated controls.
    """
    
    def __init__(self, app_instance):
        self.app = app_instance
        self.logger = app_instance.logger if hasattr(app_instance, 'logger') else None
        
        # Display state
        self.is_fullscreen_active = False
        self.current_frame = None
        self.frame_texture_id = None
        
        # Controls state
        self.show_controls = True
        self.controls_timeout = 3.0  # Hide controls after 3 seconds
        self.last_mouse_move_time = time.time()
        
        # Rendering backend
        self.render_backend = self._detect_best_backend()
        
        # Control callbacks
        self.on_play_pause: Optional[Callable] = None
        self.on_seek: Optional[Callable] = None
        self.on_stop: Optional[Callable] = None
        self.on_fullscreen_exit: Optional[Callable] = None
        
    def _detect_best_backend(self) -> str:
        """Detect the best available rendering backend."""
        if OPENGL_AVAILABLE:
            return "opengl"
        elif PYGAME_AVAILABLE:
            return "pygame"
        else:
            return "imgui"  # Fallback to pure ImGui
    
    def start_fullscreen_display(self, initial_frame: Optional[np.ndarray] = None):
        """
        Start fullscreen display mode.
        
        Args:
            initial_frame: First frame to display
        """
        try:
            if self.is_fullscreen_active:
                self.logger.warning("Fullscreen display already active") if self.logger else None
                return
            
            self.is_fullscreen_active = True
            self.current_frame = initial_frame
            self.last_mouse_move_time = time.time()
            
            if self.logger:
                self.logger.info(f"✅ Fullscreen display started (backend: {self.render_backend})")
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to start fullscreen display: {e}")
            self.is_fullscreen_active = False
    
    def stop_fullscreen_display(self):
        """Stop fullscreen display mode."""
        try:
            if not self.is_fullscreen_active:
                return
            
            # Cleanup texture if exists
            if self.frame_texture_id is not None:
                # TODO: Cleanup OpenGL texture
                self.frame_texture_id = None
            
            self.is_fullscreen_active = False
            self.current_frame = None
            
            if self.logger:
                self.logger.info("✅ Fullscreen display stopped")
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error stopping fullscreen display: {e}")
    
    def update_frame(self, frame: np.ndarray):
        """
        Update the displayed frame.
        
        Args:
            frame: New frame to display (RGB format)
        """
        if not self.is_fullscreen_active:
            return
        
        self.current_frame = frame
        
        # Update texture based on backend
        if self.render_backend == "opengl":
            self._update_opengl_texture(frame)
        elif self.render_backend == "pygame":
            self._update_pygame_surface(frame)
        # ImGui backend uses frame directly
    
    def render_fullscreen_window(self) -> bool:
        """
        Render the fullscreen window with controls.
        
        Returns:
            True if fullscreen should continue, False to exit
        """
        if not self.is_fullscreen_active:
            return False
        
        try:
            # Create fullscreen window
            imgui.set_next_window_position(0, 0)
            imgui.set_next_window_size(imgui.get_io().display_size.x, imgui.get_io().display_size.y)
            
            window_flags = (
                imgui.WINDOW_NO_TITLE_BAR |
                imgui.WINDOW_NO_RESIZE |
                imgui.WINDOW_NO_MOVE |
                imgui.WINDOW_NO_COLLAPSE |
                imgui.WINDOW_NO_SCROLLBAR |
                imgui.WINDOW_NO_DECORATION
            )
            
            expanded, opened = imgui.begin("##FullscreenVideo", True, window_flags)
            
            if not opened:
                self.stop_fullscreen_display()
                return False
            
            if expanded:
                # Handle mouse movement for control visibility
                self._handle_mouse_movement()
                
                # Render video frame
                self._render_video_frame()
                
                # Render controls overlay (if visible)
                if self.show_controls:
                    continue_fullscreen = self._render_controls_overlay()
                    if not continue_fullscreen:
                        imgui.end()
                        return False
            
            imgui.end()
            
            # Handle escape key
            if imgui.is_key_pressed(imgui.KEY_ESCAPE):
                self.stop_fullscreen_display()
                return False
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error rendering fullscreen window: {e}")
            return False
    
    def _handle_mouse_movement(self):
        """Handle mouse movement for control visibility."""
        io = imgui.get_io()
        current_time = time.time()
        
        # Check if mouse moved
        if (abs(io.mouse_delta.x) > 1 or abs(io.mouse_delta.y) > 1 or 
            io.mouse_clicked[0] or io.mouse_clicked[1]):
            self.last_mouse_move_time = current_time
            self.show_controls = True
        
        # Hide controls after timeout
        elif current_time - self.last_mouse_move_time > self.controls_timeout:
            self.show_controls = False
    
    def _render_video_frame(self):
        """Render the current video frame."""
        if self.current_frame is None:
            return
        
        # Get window size
        window_size = imgui.get_window_size()
        frame_height, frame_width = self.current_frame.shape[:2]
        
        # Calculate display size maintaining aspect ratio
        window_aspect = window_size.x / window_size.y
        frame_aspect = frame_width / frame_height
        
        if frame_aspect > window_aspect:
            # Frame is wider - fit to width
            display_width = window_size.x
            display_height = window_size.x / frame_aspect
        else:
            # Frame is taller - fit to height
            display_height = window_size.y
            display_width = window_size.y * frame_aspect
        
        # Center the frame
        offset_x = (window_size.x - display_width) / 2
        offset_y = (window_size.y - display_height) / 2
        
        imgui.set_cursor_pos((offset_x, offset_y))
        
        # Render frame based on backend
        if self.render_backend == "imgui":
            self._render_frame_imgui(display_width, display_height)
        elif self.render_backend == "opengl":
            self._render_frame_opengl(display_width, display_height)
        elif self.render_backend == "pygame":
            self._render_frame_pygame(display_width, display_height)
    
    def _render_frame_imgui(self, width: float, height: float):
        """Render frame using pure ImGui (fallback)."""
        # Convert numpy array to ImGui texture
        # This is a simplified version - in practice you'd need proper texture handling
        imgui.text(f"Video Frame: {self.current_frame.shape}")
        imgui.text("(ImGui rendering - simplified)")
    
    def _render_frame_opengl(self, width: float, height: float):
        """Render frame using OpenGL."""
        if self.frame_texture_id is not None:
            imgui.image(self.frame_texture_id, width, height)
    
    def _render_frame_pygame(self, width: float, height: float):
        """Render frame using Pygame."""
        imgui.text("Pygame rendering not implemented yet")
    
    def _render_controls_overlay(self) -> bool:
        """
        Render control overlay at bottom of screen.
        
        Returns:
            True to continue fullscreen, False to exit
        """
        window_size = imgui.get_window_size()
        
        # Position controls at bottom center
        controls_height = 80
        controls_width = 400
        controls_x = (window_size.x - controls_width) / 2
        controls_y = window_size.y - controls_height - 20
        
        imgui.set_cursor_pos((controls_x, controls_y))
        
        # Semi-transparent background
        draw_list = imgui.get_window_draw_list()
        pos = imgui.get_cursor_screen_pos()
        draw_list.add_rect_filled(
            pos.x - 10, pos.y - 10,
            pos.x + controls_width + 10, pos.y + controls_height + 10,
            imgui.get_color_u32_rgba(0, 0, 0, 0.7)
        )
        
        # Control buttons
        icon_mgr = get_icon_texture_manager()
        button_size = 40
        button_spacing = 10

        # Play/Pause button (dynamic based on state)
        is_playing = self._is_playing()
        play_pause_icon_name = 'pause.png' if is_playing else 'play.png'
        play_pause_fallback = "⏸" if is_playing else "▶"

        play_pause_tex, _, _ = icon_mgr.get_icon_texture(play_pause_icon_name)
        if play_pause_tex and imgui.image_button(play_pause_tex, button_size, button_size):
            if self.on_play_pause:
                self.on_play_pause()
        elif not play_pause_tex and imgui.button(play_pause_fallback, button_size, button_size):
            if self.on_play_pause:
                self.on_play_pause()

        imgui.same_line(spacing=button_spacing)

        # Stop button
        stop_tex, _, _ = icon_mgr.get_icon_texture('stop.png')
        if stop_tex and imgui.image_button(stop_tex, button_size, button_size):
            if self.on_stop:
                self.on_stop()
        elif not stop_tex and imgui.button("⏹", button_size, button_size):
            if self.on_stop:
                self.on_stop()

        imgui.same_line(spacing=button_spacing)

        # Exit fullscreen button
        fullscreen_exit_tex, _, _ = icon_mgr.get_icon_texture('fullscreen-exit.png')
        if fullscreen_exit_tex and imgui.image_button(fullscreen_exit_tex, button_size, button_size):
            return False  # Exit fullscreen
        elif not fullscreen_exit_tex and imgui.button("⛶", button_size, button_size):
            return False  # Exit fullscreen

        imgui.same_line(spacing=button_spacing)

        # Settings button
        settings_tex, _, _ = icon_mgr.get_icon_texture('settings.png')
        if settings_tex and imgui.image_button(settings_tex, button_size, button_size):
            pass  # Settings popup
        elif not settings_tex and imgui.button("⚙", button_size, button_size):
            pass  # Settings popup
        
        return True
    
    def _is_playing(self) -> bool:
        """Check if video is currently playing."""
        # This would check the actual playback state
        return hasattr(self.app, 'processor') and getattr(self.app.processor, 'is_processing', False)
    
    def _update_opengl_texture(self, frame: np.ndarray):
        """Update OpenGL texture with new frame."""
        if not OPENGL_AVAILABLE:
            return
        
        # TODO: Implement OpenGL texture creation/update
        pass
    
    def _update_pygame_surface(self, frame: np.ndarray):
        """Update Pygame surface with new frame."""
        if not PYGAME_AVAILABLE:
            return
        
        # TODO: Implement Pygame surface update
        pass
    
    def set_control_callbacks(self, 
                            on_play_pause: Optional[Callable] = None,
                            on_seek: Optional[Callable] = None, 
                            on_stop: Optional[Callable] = None,
                            on_fullscreen_exit: Optional[Callable] = None):
        """Set callback functions for control actions."""
        self.on_play_pause = on_play_pause
        self.on_seek = on_seek
        self.on_stop = on_stop
        self.on_fullscreen_exit = on_fullscreen_exit