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

            # Extract first frame of chapter via PyAV (in-process libav).
            start_frame = chapter.start_frame_id
            import av

            container = None
            frame = None
            total_frames = 0
            try:
                container = av.open(video_path)
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                fps = float(stream.average_rate or stream.guessed_rate or 30.0)
                time_base = stream.time_base
                total_frames = int(stream.frames or 0)
                if total_frames and start_frame >= total_frames:
                    self.logger.debug(
                        f"Frame {start_frame} is beyond video length ({total_frames} frames)")
                    return None
                target_pts = int(start_frame / fps / time_base)
                container.seek(target_pts, backward=True, any_frame=False, stream=stream)
                for packet in container.demux(stream):
                    for avframe in packet.decode():
                        frame = avframe.to_ndarray(format="bgr24")
                        break
                    if frame is not None:
                        break
            except Exception as e:
                self.logger.debug(f"PyAV chapter thumbnail extraction failed: {e}")
                return None
            finally:
                if container is not None:
                    try: container.close()
                    except Exception: pass

            if frame is None:
                self.logger.debug(
                    f"Could not read frame {start_frame} for chapter thumbnail "
                    f"(total frames: {total_frames})")
                return None

            # For VR videos, crop to show only one panel (left/right/top eye view)
            if (hasattr(self.app, 'processor') and self.app.processor and
                hasattr(self.app.processor, 'is_vr_active_or_potential') and
                self.app.processor.is_vr_active_or_potential()):
                # Determine VR format to know which panel to crop
                vr_format = getattr(self.app.processor, 'vr_input_format', '').lower()
                is_tb = '_tb' in vr_format
                is_rl = '_rl' in vr_format  # Right-left format (crop right panel)

                orig_height, orig_width = frame.shape[:2]

                if is_tb:
                    # Top-bottom format: crop to top half (top eye panel)
                    frame = frame[:orig_height // 2, :]
                elif is_rl:
                    # Right-left format (RL): crop to right half (right eye panel)
                    frame = frame[:, orig_width // 2:]
                else:
                    # Side-by-side format (SBS/LR): crop to left half (left eye panel)
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
