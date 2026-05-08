"""macOS dock icon override for non-bundled Python launches.

When FunGen runs via .venv/bin/python (not a .app bundle), macOS shows
the Python framework's default rocket icon. NSApplication.setApplicationIconImage_
overrides it for the running process. Must be called after NSApplication
exists; in practice, after the first GLFW window is created (or right
before, since GLFW lazily inits NSApp the first time create_window runs).

The NSImage must be retained for the lifetime of the process; we cache it
in a module global so Python's GC doesn't free it out from under AppKit.
"""
from __future__ import annotations

import platform
from typing import Optional

_retained_image = None
_applied = False


def apply(icon_path: str, logger=None) -> bool:
    global _retained_image, _applied
    if platform.system() != "Darwin":
        return False
    if _applied:
        return True
    try:
        from AppKit import NSApplication, NSImage
    except Exception as e:
        if logger:
            logger.debug(f"AppKit unavailable, skipping dock icon: {e}")
        return False
    try:
        img = NSImage.alloc().initWithContentsOfFile_(icon_path)
        if img is None or not img.isValid():
            if logger:
                logger.debug(f"Dock icon image invalid: {icon_path}")
            return False
        app = NSApplication.sharedApplication()
        # Regular activation policy = show in Dock + Cmd-Tab.
        try:
            app.setActivationPolicy_(0)
        except Exception:
            pass
        app.setApplicationIconImage_(img)
        _retained_image = img
        _applied = True
        if logger:
            logger.info(f"macOS dock icon set from {icon_path}")
        return True
    except Exception as e:
        if logger:
            logger.debug(f"Dock icon set failed: {e}")
        return False
