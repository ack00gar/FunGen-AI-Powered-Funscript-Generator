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


_preload_error: str = ""
# Hold the add_dll_directory handle for the process lifetime. If GC'd, the
# directory is removed from Windows DLL search path and python-mpv's bare
# CDLL("libmpv-2.dll") loses its way back to the file.
_dll_dir_handles: list = []


def _patch_find_library() -> None:
    global _preload_error
    override = _first_existing(_CANDIDATES.get(platform.system(), []))
    if override is None:
        return
    if platform.system() == "Windows":
        try:
            h = os.add_dll_directory(os.path.dirname(override))  # type: ignore[attr-defined]
            _dll_dir_handles.append(h)
        except (AttributeError, OSError):
            pass
        # Pre-load the DLL ourselves so python-mpv reuses the already-loaded
        # library on import and we capture the real Windows error if it
        # fails (Defender block, missing dep, bad arch, file lock). Without
        # this, python-mpv's generic "Cannot find" OSError masks the cause.
        try:
            ctypes.CDLL(override)
            _preload_error = ""
        except OSError as e:
            _preload_error = f"CDLL({override!r}) failed: {e}"
    _orig = ctypes.util.find_library

    def _patched(name):
        # python-mpv tries several names in order: "mpv", "libmpv-2",
        # "libmpv-1", "libmpv". Intercept all of them so a relative
        # "libmpv-2.dll" from Windows PATH search never reaches CDLL.
        if name and name.lower() in (
            "mpv", "libmpv", "libmpv-1", "libmpv-2", "mpv-1", "mpv-2"
        ):
            return override
        return _orig(name)

    ctypes.util.find_library = _patched


_autofetch_diag: list = []


def _try_fetch_libmpv_windows() -> bool:
    """Last-resort autofetch when first import fails on Windows.

    The winget shinchiro.mpv package ships mpv.exe but no libmpv-2.dll,
    so users with that install combo land here even after a clean
    install of FunGen if they skipped install.py. Pulls the dev SDK,
    extracts libmpv-2.dll into <ROOT>/lib/. Stdlib only; tar.exe ships
    with Windows 10 1803+ and reads the BCJ2-filtered .7z.

    Each step appends to _autofetch_diag so the final mpv_load_error
    can tell the user exactly which step failed.
    """
    _autofetch_diag.clear()
    if platform.system() != "Windows":
        _autofetch_diag.append("not on Windows")
        return False
    try:
        from common.paths import APP_ROOT
    except Exception as e:
        _autofetch_diag.append(f"APP_ROOT resolve failed: {e}")
        return False
    target = APP_ROOT / "lib" / "libmpv-2.dll"
    if target.is_file():
        _autofetch_diag.append(f"already present at {target}")
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
    except Exception as e:
        _autofetch_diag.append(f"GitHub API fetch failed (rate limit / network / firewall): {e}")
        return False
    asset = next((a for a in release.get("assets", [])
                  if _re.match(r"^mpv-dev-x86_64-\d", a.get("name", ""))
                  and a["name"].endswith(".7z")), None)
    if not asset:
        _autofetch_diag.append("no mpv-dev-x86_64 .7z asset found in latest release")
        return False
    _autofetch_diag.append(f"downloading {asset['name']} ({asset.get('size', 0)//1024//1024} MB)")
    archive = os.path.join(_tempfile.gettempdir(), asset["name"])
    try:
        _urlreq.urlretrieve(asset["browser_download_url"], archive)
    except Exception as e:
        _autofetch_diag.append(f"download failed: {e}")
        return False
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        _autofetch_diag.append(f"cannot create {target.parent}: {e}")
        try:
            os.remove(archive)
        except OSError:
            pass
        return False
    try:
        r = _sp.run(["tar", "-xf", archive, "-C", str(target.parent),
                     "libmpv-2.dll"],
                    capture_output=True, text=True, timeout=120)
    except Exception as e:
        _autofetch_diag.append(f"tar extraction failed: {e}")
        try:
            os.remove(archive)
        except OSError:
            pass
        return False
    finally:
        try:
            os.remove(archive)
        except OSError:
            pass
    if r.returncode != 0:
        _autofetch_diag.append(f"tar returned {r.returncode}: {r.stderr.strip()[:200]}")
        return False
    if not target.is_file():
        _autofetch_diag.append("tar succeeded but libmpv-2.dll was not extracted")
        return False
    sz = target.stat().st_size
    if sz < 50_000_000:
        _autofetch_diag.append(f"extracted file too small ({sz} bytes), likely corrupt")
        return False
    _autofetch_diag.append(f"installed libmpv-2.dll ({sz//1024//1024} MB)")
    return True


def _manual_install_hint() -> str:
    return (
        "libmpv-2.dll not found. To install manually:\n"
        "  1. Download mpv-dev-x86_64-vN-*.7z from "
        "https://github.com/shinchiro/mpv-winbuild-cmake/releases/latest\n"
        "  2. Extract libmpv-2.dll from the archive\n"
        "  3. Place it next to FunGen's main.py, or in <FunGen>/lib/, "
        "or anywhere on your %PATH%\n"
        "  4. Restart FunGen"
    )


mpv = None
mpv_available = False
mpv_load_error: str = ""

def _format_error(outer: str) -> str:
    parts = [outer]
    if _preload_error:
        parts.append(f"underlying load error: {_preload_error}")
    if _autofetch_diag:
        parts.append("autofetch steps: " + " -> ".join(_autofetch_diag))
    parts.append(_manual_install_hint())
    return "\n".join(parts)


try:
    _patch_find_library()
    import mpv as _mpv
    mpv = _mpv
    mpv_available = True
except Exception as e:
    # Windows + winget shinchiro.mpv is the most common path that lands
    # here: mpv.exe is on PATH but libmpv-2.dll is missing. Try one
    # autofetch + reload before giving up.
    _first_err = f"{type(e).__name__}: {e}"
    if platform.system() == "Windows":
        fetched = _try_fetch_libmpv_windows()
        if fetched:
            # Re-scan candidates so the patched find_library picks up the
            # newly downloaded dll, then retry the import.
            _CANDIDATES["Windows"] = _windows_candidates()
            # Drop the failed partial module from sys.modules so the second
            # import actually re-runs the body. Without this the cached
            # failed module is returned and patch_find_library has no effect.
            import sys as _sys
            _sys.modules.pop("mpv", None)
            try:
                _patch_find_library()
                import mpv as _mpv  # type: ignore
                mpv = _mpv
                mpv_available = True
                mpv_load_error = ""
            except Exception as e2:
                mpv_load_error = _format_error(f"{type(e2).__name__}: {e2}")
                mpv = None
                mpv_available = False
        else:
            mpv_load_error = _format_error(_first_err)
            mpv = None
            mpv_available = False
    else:
        mpv_load_error = _first_err
        mpv = None
        mpv_available = False


__all__ = ["mpv", "mpv_available", "mpv_load_error"]
