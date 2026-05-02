"""Shim around the python-mpv import that fixes libmpv discovery.

The installed python-mpv (jaseg/python-mpv) resolves libmpv via
``ctypes.util.find_library('mpv')`` at module import time. On Apple
Silicon this often returns ``/usr/local/lib/libmpv.dylib`` (the Intel
Homebrew prefix) while the arm64 build actually lives at
``/opt/homebrew/lib/libmpv.dylib``. The same pattern bites other
platforms when libmpv is installed outside the default search path.

We patch ``find_library`` before importing ``mpv`` so the right dylib
is loaded on the first try. Consumers should import from this module,
not directly from ``mpv``, e.g.:

    from video.mpv_loader import mpv

so the patch is applied before the real import runs.

No fallback via ffmpeg subprocess here: that lives in the Sprint D
preview/display modules, which check ``mpv_available`` below before
choosing the libmpv path.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import glob
import os
import platform
import shutil
import sys

_CANDIDATES = {
    "Darwin": [
        "/opt/homebrew/lib/libmpv.dylib",
        "/opt/homebrew/lib/libmpv.2.dylib",
        "/usr/local/lib/libmpv.dylib",
        "/usr/local/lib/libmpv.2.dylib",
    ],
    "Linux": [
        "/usr/lib/x86_64-linux-gnu/libmpv.so.2",
        "/usr/lib/x86_64-linux-gnu/libmpv.so.1",
        "/usr/lib64/libmpv.so.2",
        "/usr/lib64/libmpv.so.1",
        "/usr/local/lib/libmpv.so.2",
        "/usr/local/lib/libmpv.so.1",
    ],
    "Windows": [],
}


def _windows_candidates() -> list[str]:
    """Locate libmpv-2.dll on Windows by following mpv.exe, scanning PATH for
    the dll itself, and checking typical winget / installer drop-points.
    python-mpv's default find_library only searches PATH for the bare DLL
    name, which fails when the dll is split from mpv.exe."""
    paths: list[str] = []
    mpv_exe = shutil.which("mpv") or shutil.which("mpv.exe")
    if mpv_exe:
        mpv_dir = os.path.dirname(mpv_exe)
        for name in ("libmpv-2.dll", "mpv-2.dll", "mpv-1.dll"):
            paths.append(os.path.join(mpv_dir, name))
    # Scan every PATH entry for the dll directly; covers third-party mpv
    # bundles where the user dropped libmpv-2.dll in a different folder.
    for d in (os.environ.get("PATH") or "").split(os.pathsep):
        if not d:
            continue
        for name in ("libmpv-2.dll", "mpv-2.dll", "mpv-1.dll"):
            paths.append(os.path.join(d, name))
    local_app = os.environ.get("LOCALAPPDATA") or ""
    if local_app:
        for pattern in (
            os.path.join(local_app, "Microsoft", "WinGet", "Packages",
                         "shinchiro.mpv*", "mpv", "libmpv-2.dll"),
            os.path.join(local_app, "Programs", "mpv", "libmpv-2.dll"),
        ):
            paths.extend(glob.glob(pattern))
    program_files = os.environ.get("ProgramFiles") or r"C:\Program Files"
    paths.append(os.path.join(program_files, "mpv", "libmpv-2.dll"))
    return paths


if platform.system() == "Windows":
    _CANDIDATES["Windows"] = _windows_candidates()


def _first_existing(paths):
    for p in paths:
        if os.path.isfile(p):
            return p
    return None


def _patch_find_library() -> None:
    override = _first_existing(_CANDIDATES.get(platform.system(), []))
    if override is None:
        return
    _orig = ctypes.util.find_library

    def _patched(name):
        if name == "mpv":
            return override
        return _orig(name)

    ctypes.util.find_library = _patched


mpv = None
mpv_available = False
mpv_load_error: str = ""

try:
    _patch_find_library()
    import mpv as _mpv
    mpv = _mpv
    mpv_available = True
except Exception as e:
    mpv_load_error = f"{type(e).__name__}: {e}"
    mpv = None
    mpv_available = False


__all__ = ["mpv", "mpv_available", "mpv_load_error"]
