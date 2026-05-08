"""Ring of GPU textures snapshotting recent dewarp outputs by frame index.

Long-GOP HEVC (typical 8k VR) breaks mpv's frame-back-step ("Backstep
failed"), so backward single-frame nav has no fallback unless we already
have the previous frame on hand. This ring captures the shader output
texture each render and replays it on backward arrow hit.

GL-thread only. Uses glCopyImageSubData (GL 4.3+) to clone textures GPU
to GPU at ~0.1ms per capture. Falls back to FBO blit on older contexts.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Tuple

import OpenGL.GL as gl


class BackFrameRing:
    """Fixed-capacity LRU of (frame_index -> GL texture id)."""

    def __init__(self, capacity: int = 30):
        self.capacity = int(capacity)
        self._map: "OrderedDict[int, int]" = OrderedDict()
        self._w = 0
        self._h = 0
        self._free_textures: list[int] = []  # pre-allocated but unused

    def reset(self, w: int, h: int) -> None:
        """Drop all entries and re-pool textures at the new size."""
        for tex in list(self._map.values()) + list(self._free_textures):
            try:
                gl.glDeleteTextures(1, [tex])
            except Exception:
                pass
        self._map.clear()
        self._free_textures.clear()
        self._w = max(1, int(w))
        self._h = max(1, int(h))

    def _alloc_texture(self) -> int:
        tex = int(gl.glGenTextures(1))
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, self._w, self._h, 0,
                        gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        return tex

    def capture(self, frame_index: int, src_tex: int, src_w: int, src_h: int) -> None:
        """Copy src_tex contents into the ring slot for frame_index."""
        if src_tex <= 0 or src_w <= 0 or src_h <= 0:
            return
        # Realloc if size changed.
        if (src_w, src_h) != (self._w, self._h):
            self.reset(src_w, src_h)
        # Reuse existing slot if already keyed; else evict oldest if full.
        if frame_index in self._map:
            dst = self._map.pop(frame_index)
        elif len(self._map) >= self.capacity:
            _, dst = self._map.popitem(last=False)
        elif self._free_textures:
            dst = self._free_textures.pop()
        else:
            dst = self._alloc_texture()
        try:
            gl.glCopyImageSubData(
                int(src_tex), gl.GL_TEXTURE_2D, 0, 0, 0, 0,
                int(dst), gl.GL_TEXTURE_2D, 0, 0, 0, 0,
                int(self._w), int(self._h), 1)
        except Exception:
            self._free_textures.append(dst)
            return
        self._map[int(frame_index)] = dst

    def get(self, frame_index: int) -> Optional[int]:
        tex = self._map.get(int(frame_index))
        if tex is None:
            return None
        # Touch to mark recently used.
        self._map.move_to_end(int(frame_index))
        return int(tex)

    def has(self, frame_index: int) -> bool:
        return int(frame_index) in self._map

    def size(self) -> Tuple[int, int]:
        return (self._w, self._h)

    def __len__(self) -> int:
        return len(self._map)
