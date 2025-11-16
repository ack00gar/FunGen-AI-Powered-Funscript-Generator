"""
Chapter Thumbnail Cache System

Extracts and caches thumbnail images from video chapters for display in the UI.
Thumbnails are extracted from the middle frame of each chapter and stored in memory
with OpenGL textures for fast rendering.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Dict
import OpenGL.GL as gl
from pathlib import Path


class ChapterThumbnailCache:
    """
    Manages thumbnail extraction and caching for video chapters.

    Features:
    - Lazy loading: Thumbnails are only extracted when needed
    - Memory-efficient: Thumbnails are downscaled to a reasonable size
    - OpenGL texture caching: Ready for immediate ImGui rendering
    - Automatic cleanup: Textures are cleaned up when cache is cleared
    """

    def __init__(self, app, thumbnail_height=60):
        """
        Initialize the thumbnail cache.

        Args:
            app: Application instance (for video access and logging)
            thumbnail_height: Target height for thumbnails in pixels
        """
        self.app = app
        self.logger = logging.getLogger("ChapterThumbnailCache")
        self.thumbnail_height = thumbnail_height

        # Cache structure: {chapter_unique_id: (texture_id, width, height)}
        self._texture_cache: Dict[str, Tuple[int, int, int]] = {}

        # Track video path to invalidate cache on video change
        self._current_video_path = None

    def get_thumbnail(self, chapter) -> Optional[Tuple[int, int, int]]:
        """
        Get thumbnail texture for a chapter.

        Args:
            chapter: VideoSegment chapter object

        Returns:
            Tuple of (texture_id, width, height) or None if extraction failed
        """
        # Check if video path changed (invalidate cache)
        video_path = self.app.file_manager.video_path if self.app.file_manager else None
        if video_path != self._current_video_path:
            self.clear_cache()
            self._current_video_path = video_path

        # Return cached thumbnail if available
        if chapter.unique_id in self._texture_cache:
            return self._texture_cache[chapter.unique_id]

        # Extract and cache new thumbnail
        return self._extract_and_cache_thumbnail(chapter)

    def _extract_and_cache_thumbnail(self, chapter) -> Optional[Tuple[int, int, int]]:
        """Extract thumbnail from video and cache it."""
        try:
            # Get video path
            video_path = self.app.file_manager.video_path if self.app.file_manager else None
            if not video_path or not Path(video_path).exists():
                self.logger.debug(f"Video path not available for thumbnail extraction")
                return None

            # Extract first frame of chapter
            start_frame = chapter.start_frame_id

            # Extract frame using OpenCV
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.warning(f"Could not open video for thumbnail: {video_path}")
                return None

            # Get total frame count to validate
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if start_frame >= total_frames:
                self.logger.debug(f"Frame {start_frame} is beyond video length ({total_frames} frames)")
                cap.release()
                return None

            # Try to seek to the frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Verify we're at or near the requested frame
            actual_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Read the frame
            ret, frame = cap.read()

            # If seeking failed, try reading sequentially from a nearby keyframe
            if not ret or frame is None:
                # Try seeking a bit earlier and reading forward
                seek_frame = max(0, start_frame - 10)
                cap.set(cv2.CAP_PROP_POS_FRAMES, seek_frame)

                # Read frames until we reach the target or fail
                for _ in range(min(20, start_frame - seek_frame + 5)):
                    ret, frame = cap.read()
                    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    if ret and frame is not None and current_frame >= start_frame:
                        break

            cap.release()

            if not ret or frame is None:
                self.logger.debug(f"Could not read frame {start_frame} for chapter thumbnail (total frames: {total_frames})")
                return None

            # For VR videos, crop to show only left panel (left eye view)
            if (hasattr(self.app, 'processor') and self.app.processor and
                hasattr(self.app.processor, 'is_vr_active_or_potential') and
                self.app.processor.is_vr_active_or_potential()):
                # Crop left half of the frame (left eye panel)
                orig_height, orig_width = frame.shape[:2]
                frame = frame[:, :orig_width // 2]

            # Resize thumbnail to target height while preserving aspect ratio
            orig_height, orig_width = frame.shape[:2]
            aspect_ratio = orig_width / orig_height
            thumbnail_width = int(self.thumbnail_height * aspect_ratio)

            thumbnail = cv2.resize(frame, (thumbnail_width, self.thumbnail_height),
                                 interpolation=cv2.INTER_AREA)

            # Convert BGR to RGBA for OpenGL
            thumbnail_rgba = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGBA)

            # Create OpenGL texture
            texture_id = self._create_gl_texture(thumbnail_rgba)

            if texture_id is None:
                return None

            # Cache the texture
            self._texture_cache[chapter.unique_id] = (texture_id, thumbnail_width, self.thumbnail_height)

            self.logger.debug(f"Cached thumbnail for chapter {chapter.unique_id[:8]}...")

            return (texture_id, thumbnail_width, self.thumbnail_height)

        except Exception as e:
            self.logger.warning(f"Failed to extract thumbnail for chapter {chapter.unique_id}: {e}")
            return None

    def _create_gl_texture(self, image_rgba: np.ndarray) -> Optional[int]:
        """
        Create an OpenGL texture from an RGBA image.

        Args:
            image_rgba: NumPy array in RGBA format

        Returns:
            OpenGL texture ID or None on failure
        """
        try:
            height, width = image_rgba.shape[:2]

            # Generate texture
            texture_id = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)

            # Set texture parameters
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

            # Upload texture data
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D, 0, gl.GL_RGBA,
                width, height, 0,
                gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, image_rgba
            )

            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

            return texture_id

        except Exception as e:
            self.logger.error(f"Failed to create OpenGL texture: {e}")
            return None

    def clear_cache(self):
        """Clear all cached thumbnails and free OpenGL textures."""
        try:
            for chapter_id, (texture_id, _, _) in self._texture_cache.items():
                if texture_id > 0:
                    gl.glDeleteTextures([texture_id])
        except Exception as e:
            self.logger.warning(f"Error cleaning up textures: {e}")

        self._texture_cache.clear()
        self.logger.debug("Thumbnail cache cleared")

    def preload_thumbnails(self, chapters):
        """
        Preload thumbnails for a list of chapters in the background.

        This can be called to warm up the cache before displaying the chapter list.

        Args:
            chapters: List of chapter objects to preload
        """
        for chapter in chapters:
            if chapter.unique_id not in self._texture_cache:
                self._extract_and_cache_thumbnail(chapter)

    def __del__(self):
        """Cleanup OpenGL textures on deletion."""
        self.clear_cache()
