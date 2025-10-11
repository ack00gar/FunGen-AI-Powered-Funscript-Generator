"""
UI Icon Texture Utility

Provides centralized UI icon texture loading and management for FunGen GUI.
Used for:
- Playback controls (play, pause, stop, etc.)
- Zoom controls (zoom in, zoom out)
- Fullscreen controls
- Settings and other UI buttons
"""

import cv2
import numpy as np
import OpenGL.GL as gl
import os
import logging

logger = logging.getLogger(__name__)


class IconTextureManager:
    """Manages UI icon texture loading and caching for OpenGL rendering."""

    def __init__(self):
        self._icon_cache = {}  # {icon_name: (texture_id, width, height)}
        self._assets_dir = os.path.join(
            os.path.dirname(__file__), '..', '..', 'assets'
        )

    def load_icon_texture(self, icon_name):
        """
        Load an icon from assets/ui/icons/ as OpenGL texture.

        Args:
            icon_name: Icon filename without path (e.g., 'play.png', 'pause.png')

        Returns:
            tuple: (texture_id, width, height), or (None, 0, 0) if loading failed
        """
        # Check cache first
        if icon_name in self._icon_cache:
            return self._icon_cache[icon_name]

        try:
            # Construct icon path
            icon_path = os.path.join(self._assets_dir, 'ui', 'icons', icon_name)

            if not os.path.exists(icon_path):
                logger.debug(f"Icon file not found: {icon_path}")
                return (None, 0, 0)

            # Load icon with cv2
            icon_img = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)

            if icon_img is None:
                logger.warning(f"Failed to load icon image: {icon_path}")
                return (None, 0, 0)

            # Convert BGR(A) to RGB(A)
            if len(icon_img.shape) == 3 and icon_img.shape[2] == 4:  # Has alpha channel
                icon_rgb = cv2.cvtColor(icon_img, cv2.COLOR_BGRA2RGBA)
            elif len(icon_img.shape) == 3:  # RGB only
                icon_rgb = cv2.cvtColor(icon_img, cv2.COLOR_BGR2RGB)
                # Add alpha channel (fully opaque)
                alpha = np.full((icon_rgb.shape[0], icon_rgb.shape[1], 1), 255, dtype=np.uint8)
                icon_rgb = np.concatenate([icon_rgb, alpha], axis=2)
            else:
                # Grayscale - convert to RGBA
                icon_rgb = cv2.cvtColor(icon_img, cv2.COLOR_GRAY2RGB)
                alpha = np.full((icon_rgb.shape[0], icon_rgb.shape[1], 1), 255, dtype=np.uint8)
                icon_rgb = np.concatenate([icon_rgb, alpha], axis=2)

            height, width = icon_rgb.shape[:2]

            # Create OpenGL texture
            texture_id = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)

            # Set texture parameters
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

            # Upload texture data
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D,
                0,
                gl.GL_RGBA,
                width,
                height,
                0,
                gl.GL_RGBA,
                gl.GL_UNSIGNED_BYTE,
                icon_rgb
            )

            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

            # Cache the result
            self._icon_cache[icon_name] = (texture_id, width, height)
            logger.debug(f"Loaded icon texture: {icon_name} ({width}x{height})")

            return (texture_id, width, height)

        except Exception as e:
            logger.error(f"Error loading icon texture '{icon_name}': {e}")
            return (None, 0, 0)

    def get_icon_texture(self, icon_name):
        """
        Get icon texture (loads if not already cached).

        Args:
            icon_name: Icon filename without path (e.g., 'play.png', 'pause.png')

        Returns:
            tuple: (texture_id, width, height), or (None, 0, 0) if loading failed
        """
        if icon_name not in self._icon_cache:
            return self.load_icon_texture(icon_name)
        return self._icon_cache[icon_name]

    def preload_icons(self, icon_list):
        """
        Preload multiple icons to avoid loading delays during runtime.

        Args:
            icon_list: List of icon filenames to preload
        """
        for icon_name in icon_list:
            self.load_icon_texture(icon_name)

    def cleanup(self):
        """Free all OpenGL texture resources."""
        for icon_name, (texture_id, _, _) in self._icon_cache.items():
            if texture_id is not None:
                gl.glDeleteTextures([texture_id])
        self._icon_cache.clear()


# Global singleton instance
_icon_manager = None


def get_icon_texture_manager():
    """
    Get global IconTextureManager singleton.

    Returns:
        IconTextureManager: Global icon texture manager instance
    """
    global _icon_manager
    if _icon_manager is None:
        _icon_manager = IconTextureManager()
    return _icon_manager


# Convenience function for common playback icons
def preload_playback_icons():
    """Preload all playback control icons for immediate availability."""
    manager = get_icon_texture_manager()
    playback_icons = [
        'jump-start.png',
        'prev-frame.png',
        'play.png',
        'pause.png',
        'stop.png',
        'next-frame.png',
        'jump-end.png',
        'zoom-in.png',
        'zoom-out.png',
        'fullscreen.png',
        'fullscreen-exit.png',
        'settings.png',
    ]
    manager.preload_icons(playback_icons)
    logger.debug(f"Preloaded {len(playback_icons)} playback control icons")
