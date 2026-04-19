"""Shortcut string → (GLFW key code, modifier dict) parser with a hot-path cache.

Called per-frame from the GUI event loop, so the cache matters: without it
every frame re-tokenizes `"CTRL+SHIFT+A"` into parts. Cache entries can be
None to memoize "unparseable" results and skip the work on subsequent calls.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple


_MODIFIER_NAMES = ("CTRL", "ALT", "SHIFT", "SUPER")


class ShortcutMapper:
    """Parses shortcut strings against a ShortcutManager's keycode table."""

    __slots__ = ("shortcut_manager", "_cache")

    def __init__(self, shortcut_manager) -> None:
        self.shortcut_manager = shortcut_manager
        self._cache: Dict[str, Optional[Tuple[int, Dict[str, bool]]]] = {}

    def map(self, shortcut_string: str) -> Optional[Tuple[int, Dict[str, bool]]]:
        """Parse a shortcut string to (glfw_key_code, modifier_dict), or None."""
        if not shortcut_string:
            return None

        cached = self._cache.get(shortcut_string, _MISS)
        if cached is not _MISS:
            return cached

        parts = shortcut_string.upper().split('+')
        modifiers = {'ctrl': False, 'alt': False, 'shift': False, 'super': False}
        main_key_str = None

        for part in parts:
            part = part.strip()
            if part == "CTRL":
                modifiers['ctrl'] = True
            elif part == "ALT":
                modifiers['alt'] = True
            elif part == "SHIFT":
                modifiers['shift'] = True
            elif part == "SUPER":
                modifiers['super'] = True
            else:
                if main_key_str is not None:
                    self._cache[shortcut_string] = None
                    return None
                main_key_str = part

        if main_key_str is None:
            self._cache[shortcut_string] = None
            return None

        if not self.shortcut_manager:
            return None

        glfw_key_code = self.shortcut_manager.name_to_glfw_key(main_key_str)
        if glfw_key_code is None:
            self._cache[shortcut_string] = None
            return None

        result = (glfw_key_code, modifiers)
        self._cache[shortcut_string] = result
        return result

    def invalidate(self) -> None:
        """Clear the cache — call when shortcut bindings change."""
        self._cache.clear()


_MISS = object()  # sentinel for dict.get to distinguish "cached None" from "not cached"
