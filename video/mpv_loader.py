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
    # FunGen install root: lets users drop libmpv-2.dll next to the app or
    # under <root>/lib/. shinchiro player package (what winget installs) does
    # not ship libmpv-2.dll; the dev SDK does and gets dropped here.
    try:
        from common.paths import APP_ROOT
        for d in (APP_ROOT, APP_ROOT / "lib"):
            for name in ("libmpv-2.dll", "mpv-2.dll", "mpv-1.dll"):
                paths.append(str(d / name))
    except Exception:
        pass
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
            # Recursive: shinchiro.mpv unpacks to a versioned mpv-x86_64-* subdir
            # whose name changes per release, so a fixed glob misses it.
            os.path.join(local_app, "Microsoft", "WinGet", "Packages",
                         "shinchiro.mpv*", "**", "libmpv-2.dll"),
            os.path.join(local_app, "Programs", "mpv", "libmpv-2.dll"),
        ):
            paths.extend(glob.glob(pattern, recursive=True))
    program_files = os.environ.get("ProgramFiles") or r"C:\Program Files"
    paths.append(os.path.join(program_files, "mpv", "libmpv-2.dll"))

    # Registry: HKLM/HKCU App Paths is set by some mpv installers and survives
    # the PATH-not-refreshed-yet trap right after a winget install.
    try:
        import winreg
        for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            for sub in (r"Software\Microsoft\Windows\CurrentVersion\App Paths\mpv.exe",):
                try:
                    with winreg.OpenKey(hive, sub) as k:
                        val, _ = winreg.QueryValueEx(k, "Path")
                        if val:
                            for name in ("libmpv-2.dll", "mpv-2.dll", "mpv-1.dll"):
                                paths.append(os.path.join(val, name))
                except OSError:
                    pass
    except ImportError:
        pass

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


def _try_fetch_libmpv_windows() -> bool:
    """Last-resort autofetch when first import fails on Windows.

    The winget shinchiro.mpv package ships mpv.exe but no libmpv-2.dll,
    so users with that install combo land here even after a clean
    install of FunGen if they skipped install.py. Pulls the dev SDK,
    extracts libmpv-2.dll into <ROOT>/lib/. Stdlib only; tar.exe ships
    with Windows 10 1803+ and reads the BCJ2-filtered .7z."""
    if platform.system() != "Windows":
        return False
    try:
        from common.paths import APP_ROOT
    except Exception:
        return False
    target = APP_ROOT / "lib" / "libmpv-2.dll"
    if target.is_file():
        return True
    import json as _json
    import re as _re
    import subprocess as _sp
    import tempfile as _tempfile
    import urllib.request as _urlreq
    api = "https://api.github.com/repos/shinchiro/mpv-winbuild-cmake/releases/latest"
    try:
        with _urlreq.urlopen(api, timeout=15) as r:
            release = _json.load(r)
    except Exception:
        return False
    asset = next((a for a in release.get("assets", [])
                  if _re.match(r"^mpv-dev-x86_64-\d", a.get("name", ""))
                  and a["name"].endswith(".7z")), None)
    if not asset:
        return False
    archive = os.path.join(_tempfile.gettempdir(), asset["name"])
    try:
        _urlreq.urlretrieve(asset["browser_download_url"], archive)
    except Exception:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = _sp.run(["tar", "-xf", archive, "-C", str(target.parent),
                     "libmpv-2.dll"],
                    capture_output=True, text=True, timeout=120)
    except Exception:
        return False
    finally:
        try:
            os.remove(archive)
        except OSError:
            pass
    return r.returncode == 0 and target.is_file() and target.stat().st_size > 50_000_000


mpv = None
mpv_available = False
mpv_load_error: str = ""

try:
    _patch_find_library()
    import mpv as _mpv
    mpv = _mpv
    mpv_available = True
except Exception as e:
    # Windows + winget shinchiro.mpv is the most common path that lands
    # here: mpv.exe is on PATH but libmpv-2.dll is missing. Try one
    # autofetch + reload before giving up.
    if platform.system() == "Windows" and _try_fetch_libmpv_windows():
        # Re-scan candidates so the patched find_library picks up the
        # newly downloaded dll, then retry the import.
        _CANDIDATES["Windows"] = _windows_candidates()
        try:
            _patch_find_library()
            import mpv as _mpv  # type: ignore
            mpv = _mpv
            mpv_available = True
            mpv_load_error = ""
        except Exception as e2:
            mpv_load_error = f"{type(e2).__name__}: {e2}"
            mpv = None
            mpv_available = False
    else:
        mpv_load_error = f"{type(e).__name__}: {e}"
        mpv = None
        mpv_available = False


__all__ = ["mpv", "mpv_available", "mpv_load_error"]
