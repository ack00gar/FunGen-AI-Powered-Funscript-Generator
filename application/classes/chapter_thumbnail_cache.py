"""
Chapter Thumbnail Cache System

Extracts and caches thumbnail images from video chapters for display in the UI.
Thumbnails are extracted from the middle frame of each chapter and stored in memory
with OpenGL textures for fast rendering.
"""

import cv2
import numpy as np
import logging
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Set, Tuple, Dict
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
    - Async decode: ffmpeg/ffprobe + cv2 on worker pool; GL upload on UI thread.
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

        # _generation bumps on clear so stale worker results are dropped.
        self._current_video_path = None
        self._generation = 0

        # 2 workers saturate a single SSD on ffmpeg subprocess spawn cost.
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="ChapterThumb"
        )
        self._pending: Set[str] = set()
        self._pending_lock = threading.Lock()
        self._completed: "queue.Queue" = queue.Queue()

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

        # Drain any completed decodes -> create GL textures here (UI thread).
        self._drain_completed()

        # Return cached thumbnail if available
        if chapter.unique_id in self._texture_cache:
            return self._texture_cache[chapter.unique_id]

        # Schedule async decode if not already in flight
        if video_path:
            with self._pending_lock:
                already = chapter.unique_id in self._pending
                if not already:
                    self._pending.add(chapter.unique_id)
            if not already:
                gen = self._generation
                self._executor.submit(
                    self._decode_worker, chapter, video_path, gen,
                )
        return None

    def _drain_completed(self) -> None:
        """Drain decoded buffers, upload as GL textures. UI-thread only."""
        MAX_PER_FRAME = 4
        for _ in range(MAX_PER_FRAME):
            try:
                cid, rgba, w, h, gen = self._completed.get_nowait()
            except queue.Empty:
                return
            with self._pending_lock:
                self._pending.discard(cid)
            if gen != self._generation or cid in self._texture_cache:
                continue
            tex = self._create_gl_texture(rgba)
            if tex is not None:
                self._texture_cache[cid] = (tex, w, h)

    def _decode_worker(self, chapter, video_path: str, generation: int) -> None:
        try:
            result = self._extract_thumbnail_buffer(chapter, video_path)
            if result is not None:
                rgba, w, h = result
                self._completed.put((chapter.unique_id, rgba, w, h, generation))
                return  # drain clears _pending on success
        except Exception as e:
            self.logger.debug(f"thumbnail worker failed: {e}")
        with self._pending_lock:
            self._pending.discard(chapter.unique_id)

    def _extract_thumbnail_buffer(
        self, chapter, video_path: str,
    ) -> Optional[Tuple[np.ndarray, int, int]]:
        """Off-thread: ffprobe + ffmpeg + cv2 + resize. No GL calls."""
        if not video_path or not Path(video_path).exists():
            return None

        start_frame = chapter.start_frame_id
        import subprocess
        from video.ffmpeg_helpers import find_ffmpeg, subprocess_flags
        from video.frame_source.probe import probe as _probe

        info = _probe(video_path)
        if info is None:
            return None
        fps = info.fps if info.fps > 0 else 30.0
        total_frames = info.total_frames
        if total_frames and start_frame >= total_frames:
            return None

        seek_time = start_frame / fps
        cmd = [
            find_ffmpeg(), '-hide_banner', '-loglevel', 'error', '-nostats',
            '-ss', f'{seek_time:.6f}',
            '-i', video_path,
            '-frames:v', '1',
            '-f', 'image2pipe', '-c:v', 'bmp', '-',
        ]
        frame = None
        try:
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=10.0, creationflags=subprocess_flags(),
            )
            if result.returncode == 0 and result.stdout:
                buf = np.frombuffer(result.stdout, dtype=np.uint8)
                frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except (OSError, subprocess.TimeoutExpired):
            return None

        if frame is None or frame.size == 0:
            return None

        if (hasattr(self.app, 'processor') and self.app.processor and
            hasattr(self.app.processor, 'is_vr_active_or_potential') and
            self.app.processor.is_vr_active_or_potential()):
            from video import vr_panel
            vr_format = getattr(self.app.processor, 'vr_input_format', '')
            eye = vr_panel.read_setting(
                getattr(self.app, 'app_settings', None),
                default=vr_panel.EYE_LEFT)
            if eye == vr_panel.EYE_FULL:
                eye = vr_panel.EYE_LEFT
            orig_height, orig_width = frame.shape[:2]
            region = vr_panel.resolve_eye(vr_format, eye)
            if not region.is_full():
                x, y, w, h = region.pixel_rect(orig_width, orig_height)
                frame = frame[y:y + h, x:x + w]

        orig_height, orig_width = frame.shape[:2]
        aspect_ratio = orig_width / orig_height
        thumbnail_width = int(self.thumbnail_height * aspect_ratio)

        thumbnail = cv2.resize(
            frame, (thumbnail_width, self.thumbnail_height),
            interpolation=cv2.INTER_AREA)
        thumbnail_rgba = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGBA)
        return (thumbnail_rgba, thumbnail_width, self.thumbnail_height)

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
        """Free GL textures and invalidate in-flight worker results."""
        try:
            for chapter_id, (texture_id, _, _) in self._texture_cache.items():
                if texture_id > 0:
                    gl.glDeleteTextures([texture_id])
        except Exception as e:
            self.logger.warning(f"texture cleanup: {e}")

        self._texture_cache.clear()
        self._generation += 1
        try:
            while True:
                self._completed.get_nowait()
        except queue.Empty:
            pass
        with self._pending_lock:
            self._pending.clear()
        self.logger.debug("Thumbnail cache cleared")

    def preload_thumbnails(self, chapters):
        """Schedule async decode for a batch of chapters."""
        video_path = self.app.file_manager.video_path if self.app.file_manager else None
        if not video_path:
            return
        gen = self._generation
        for chapter in chapters:
            if chapter.unique_id in self._texture_cache:
                continue
            with self._pending_lock:
                if chapter.unique_id in self._pending:
                    continue
                self._pending.add(chapter.unique_id)
            self._executor.submit(
                self._decode_worker, chapter, video_path, gen,
            )

    def __del__(self):
        """Cleanup OpenGL textures on deletion."""
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass
        self.clear_cache()
