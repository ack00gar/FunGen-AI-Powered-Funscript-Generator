"""Shared helpers for locating ffmpeg / ffprobe and spawning them cleanly.

Centralizes the pattern v0.8.0 inlined at every subprocess call site:
  creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
plus the binary-location fallback chain used in application/live_capture/
and the old offline-tracker code.

Keep this module dependency-free (stdlib only) so every subprocess caller
can import it without pulling heavy deps.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Optional


_FFMPEG_FALLBACKS = ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg")
_FFPROBE_FALLBACKS = ("/opt/homebrew/bin/ffprobe", "/usr/local/bin/ffprobe", "/usr/bin/ffprobe")


def find_ffmpeg() -> str:
    """Locate the ffmpeg binary. Returns the bare name 'ffmpeg' as a last
    resort so subprocess still tries PATH at call time."""
    path = shutil.which("ffmpeg")
    if path:
        return path
    for candidate in _FFMPEG_FALLBACKS:
        if os.path.isfile(candidate):
            return candidate
    return "ffmpeg"


def find_ffprobe() -> str:
    """Locate the ffprobe binary. Same fallback strategy as find_ffmpeg()."""
    path = shutil.which("ffprobe")
    if path:
        return path
    for candidate in _FFPROBE_FALLBACKS:
        if os.path.isfile(candidate):
            return candidate
    return "ffprobe"


def subprocess_flags() -> int:
    """Return Windows CREATE_NO_WINDOW so child processes don't pop a console
    window. On non-Windows, 0 (no flags)."""
    if sys.platform == "win32":
        return getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return 0


def run(cmd: list, *, timeout: Optional[float] = None,
        check: bool = False, capture_output: bool = True) -> subprocess.CompletedProcess:
    """Thin wrapper around subprocess.run with the Windows creation flag
    applied and stderr captured by default. Use this for one-shot invocations
    (ffprobe, ffmpeg screenshots, clip remux). For persistent pipes use
    subprocess.Popen directly with subprocess_flags()."""
    kwargs = {
        "creationflags": subprocess_flags(),
        "check": check,
    }
    if capture_output:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    if timeout is not None:
        kwargs["timeout"] = timeout
    return subprocess.run(cmd, **kwargs)
