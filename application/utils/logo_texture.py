"""
Logo Texture Utility

Provides centralized logo texture loading and management for FunGen GUI.
Used for:
- Video feed placeholder when no video is loaded
- Error/warning dialogs
- About dialog
- Splash screen
"""

import cv2
import numpy as np
import OpenGL.GL as gl
import os


class LogoTextureManager:
    """Manages logo texture loading and caching for OpenGL rendering."""

    def __init__(self):
        self.logo_texture_id = None
        self.logo_width = 0
        self.logo_height = 0
        self._logo_loaded = False

    def load_logo_texture(self):
        """
        Load logo.png from assets folder as OpenGL texture.

        Returns:
            int: OpenGL texture ID, or None if loading failed
        """
        if self._logo_loaded and self.logo_texture_id is not None:
            return self.logo_texture_id

        try:
            # Find logo.png in assets folder
            logo_path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'logo.png')

            if not os.path.exists(logo_path):
                print(f"Logo file not found: {logo_path}")
                return None

            # Load logo with cv2
            logo_img = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)

            if logo_img is None:
                print(f"Failed to load logo image: {logo_path}")
                return None

            # Convert BGR(A) to RGB(A)
            if logo_img.shape[2] == 4:  # Has alpha channel
                logo_rgb = cv2.cvtColor(logo_img, cv2.COLOR_BGRA2RGBA)
            else:
                logo_rgb = cv2.cvtColor(logo_img, cv2.COLOR_BGR2RGB)
                # Add alpha channel (fully opaque)
                alpha = np.full((logo_rgb.shape[0], logo_rgb.shape[1], 1), 255, dtype=np.uint8)
                logo_rgb = np.concatenate([logo_rgb, alpha], axis=2)

            # Note: No vertical flip needed for imgui.image - it handles texture coordinates correctly

            self.logo_height, self.logo_width = logo_rgb.shape[:2]

            # Create OpenGL texture
            self.logo_texture_id = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.logo_texture_id)

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
                self.logo_width,
                self.logo_height,
                0,
                gl.GL_RGBA,
                gl.GL_UNSIGNED_BYTE,
                logo_rgb
            )

            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

            self._logo_loaded = True
            return self.logo_texture_id

        except Exception as e:
            print(f"Error loading logo texture: {e}")
            return None

    def get_texture_id(self):
        """
        Get logo texture ID (loads if not already loaded).

        Returns:
            int: OpenGL texture ID, or None if loading failed
        """
        if not self._logo_loaded:
            return self.load_logo_texture()
        return self.logo_texture_id

    def get_dimensions(self):
        """
        Get logo dimensions.

        Returns:
            tuple: (width, height) in pixels, or (0, 0) if not loaded
        """
        return (self.logo_width, self.logo_height)

    def cleanup(self):
        """Free OpenGL texture resources."""
        if self.logo_texture_id is not None:
            gl.glDeleteTextures([self.logo_texture_id])
            self.logo_texture_id = None
            self._logo_loaded = False


# Global singleton instance
_logo_manager = None


def get_logo_texture_manager():
    """
    Get global LogoTextureManager singleton.

    Returns:
        LogoTextureManager: Global logo texture manager instance
    """
    global _logo_manager
    if _logo_manager is None:
        _logo_manager = LogoTextureManager()
    return _logo_manager
