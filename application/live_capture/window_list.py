"""
Window Enumeration

Platform-specific window listing for the "select what to capture" UI.
macOS: Quartz CGWindowListCopyWindowInfo or AppleScript fallback
Windows: ctypes user32.dll EnumWindows
Linux: wmctrl or xdotool
"""

import sys
import logging
import subprocess
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class WindowInfo:
    """Information about a visible window."""
    title: str
    pid: int
    app_name: str
    window_id: int

    @property
    def display_label(self) -> str:
        """Human-readable label for UI dropdown."""
        if self.title and self.app_name:
            return f"{self.app_name} - {self.title}"
        return self.app_name or self.title or f"Window {self.window_id}"


@dataclass
class ScreenInfo:
    """Information about a display/monitor."""
    index: int
    name: str
    width: int
    height: int
    x: int = 0
    y: int = 0

    @property
    def display_label(self) -> str:
        return f"{self.name} ({self.width}x{self.height})"


def get_available_windows() -> List[WindowInfo]:
    """Return list of visible windows on the current platform."""
    if sys.platform == "darwin":
        return _get_windows_macos()
    elif sys.platform == "win32":
        return _get_windows_windows()
    else:
        return _get_windows_linux()


def get_available_screens() -> List[ScreenInfo]:
    """Return list of available screens/monitors."""
    if sys.platform == "darwin":
        return _get_screens_macos()
    elif sys.platform == "win32":
        return _get_screens_windows()
    else:
        return _get_screens_linux()


# ── macOS ──────────────────────────────────────────────────────────────

def _get_windows_macos() -> List[WindowInfo]:
    """Get visible windows on macOS using Quartz."""
    windows = []
    try:
        from Quartz import (
            CGWindowListCopyWindowInfo,
            kCGWindowListOptionOnScreenOnly,
            kCGWindowListExcludeDesktopElements,
            kCGNullWindowID,
        )
        window_list = CGWindowListCopyWindowInfo(
            kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements,
            kCGNullWindowID,
        )
        for win in window_list:
            owner = win.get("kCGWindowOwnerName", "")
            name = win.get("kCGWindowName", "")
            pid = win.get("kCGWindowOwnerPID", 0)
            wid = win.get("kCGWindowNumber", 0)
            # Skip windows with no name and system UI elements
            if not owner or owner in ("Window Server", "Dock", "SystemUIServer"):
                continue
            windows.append(WindowInfo(title=name, pid=pid, app_name=owner, window_id=wid))
    except ImportError:
        # Fallback to AppleScript
        windows = _get_windows_macos_applescript()
    return windows


def _get_windows_macos_applescript() -> List[WindowInfo]:
    """Fallback: list visible apps via AppleScript."""
    windows = []
    try:
        # Use a loop to emit one "name|pid" per line for reliable parsing
        script = (
            'tell application "System Events"\n'
            '  set appList to ""\n'
            '  repeat with p in (every process whose visible is true)\n'
            '    set appList to appList & name of p & "|" & unix id of p & "\n"\n'
            '  end repeat\n'
            '  return appList\n'
            'end tell'
        )
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                parts = line.split("|", 1)
                if len(parts) == 2:
                    name = parts[0].strip()
                    try:
                        pid = int(parts[1].strip())
                    except ValueError:
                        pid = 0
                    if name:
                        windows.append(WindowInfo(title="", pid=pid, app_name=name, window_id=0))
    except Exception as e:
        logger.warning(f"AppleScript window listing failed: {e}")
    return windows


def _get_screens_macos() -> List[ScreenInfo]:
    """Get available screens on macOS."""
    screens = []
    try:
        from Quartz import CGDisplayBounds, CGMainDisplayID
        from CoreGraphics import CGGetActiveDisplayList
        max_displays = 16
        active_displays, count = CGGetActiveDisplayList(max_displays, None, None)
        for i, display_id in enumerate(active_displays[:count]):
            bounds = CGDisplayBounds(display_id)
            is_main = display_id == CGMainDisplayID()
            name = f"Display {i + 1}" + (" (Main)" if is_main else "")
            screens.append(ScreenInfo(
                index=i, name=name,
                width=int(bounds.size.width), height=int(bounds.size.height),
                x=int(bounds.origin.x), y=int(bounds.origin.y),
            ))
    except (ImportError, Exception) as e:
        logger.debug(f"Quartz screen listing unavailable: {e}")
        # Fallback: assume a single primary display
        screens.append(ScreenInfo(index=0, name="Primary Display", width=1920, height=1080))
    return screens


# ── Windows ────────────────────────────────────────────────────────────

def _get_windows_windows() -> List[WindowInfo]:
    """Get visible windows on Windows using ctypes."""
    windows = []
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32

        EnumWindows = user32.EnumWindows
        GetWindowTextW = user32.GetWindowTextW
        GetWindowTextLengthW = user32.GetWindowTextLengthW
        IsWindowVisible = user32.IsWindowVisible
        GetWindowThreadProcessId = user32.GetWindowThreadProcessId

        WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def enum_callback(hwnd, _lparam):
            if not IsWindowVisible(hwnd):
                return True
            length = GetWindowTextLengthW(hwnd)
            if length == 0:
                return True
            buf = ctypes.create_unicode_buffer(length + 1)
            GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value
            pid = wintypes.DWORD()
            GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            windows.append(WindowInfo(
                title=title, pid=pid.value, app_name="", window_id=hwnd,
            ))
            return True

        EnumWindows(WNDENUMPROC(enum_callback), 0)
    except Exception as e:
        logger.warning(f"Windows window listing failed: {e}")
    return windows


def _get_screens_windows() -> List[ScreenInfo]:
    """Get available screens on Windows."""
    screens = []
    try:
        import ctypes
        user32 = ctypes.windll.user32

        def monitor_enum_proc(hmonitor, hdc, lprect, lparam):
            import ctypes
            from ctypes import wintypes
            rect = lprect.contents
            w = rect.right - rect.left
            h = rect.bottom - rect.top
            idx = len(screens)
            screens.append(ScreenInfo(
                index=idx, name=f"Monitor {idx + 1}",
                width=w, height=h, x=rect.left, y=rect.top,
            ))
            return True

        MONITORENUMPROC = ctypes.WINFUNCTYPE(
            ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
            ctypes.POINTER(ctypes.wintypes.RECT), ctypes.c_double
        )
        user32.EnumDisplayMonitors(None, None, MONITORENUMPROC(monitor_enum_proc), 0)
    except Exception as e:
        logger.debug(f"Windows screen listing failed: {e}")
        screens.append(ScreenInfo(index=0, name="Primary Display", width=1920, height=1080))
    return screens


# ── Linux ──────────────────────────────────────────────────────────────

def _get_windows_linux() -> List[WindowInfo]:
    """Get visible windows on Linux via wmctrl or xdotool."""
    windows = []
    # Try wmctrl first
    try:
        result = subprocess.run(["wmctrl", "-l", "-p"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                parts = line.split(None, 4)
                if len(parts) >= 5:
                    wid = int(parts[0], 16)
                    pid = int(parts[2])
                    title = parts[4]
                    windows.append(WindowInfo(title=title, pid=pid, app_name="", window_id=wid))
            return windows
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.debug(f"wmctrl failed: {e}")

    # Fallback to xdotool
    try:
        result = subprocess.run(
            ["xdotool", "search", "--onlyvisible", "--name", ""],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for wid_str in result.stdout.strip().splitlines():
                try:
                    wid = int(wid_str)
                    name_result = subprocess.run(
                        ["xdotool", "getwindowname", str(wid)],
                        capture_output=True, text=True, timeout=2,
                    )
                    title = name_result.stdout.strip() if name_result.returncode == 0 else ""
                    pid_result = subprocess.run(
                        ["xdotool", "getwindowpid", str(wid)],
                        capture_output=True, text=True, timeout=2,
                    )
                    pid = int(pid_result.stdout.strip()) if pid_result.returncode == 0 else 0
                    if title:
                        windows.append(WindowInfo(title=title, pid=pid, app_name="", window_id=wid))
                except (ValueError, Exception):
                    continue
    except FileNotFoundError:
        logger.warning("Neither wmctrl nor xdotool found. Install one for window listing.")
    except Exception as e:
        logger.debug(f"xdotool failed: {e}")

    return windows


def _get_screens_linux() -> List[ScreenInfo]:
    """Get available screens on Linux via xrandr."""
    screens = []
    try:
        result = subprocess.run(["xrandr", "--query"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            import re
            idx = 0
            for line in result.stdout.splitlines():
                match = re.match(r"(\S+)\s+connected\s+(?:primary\s+)?(\d+)x(\d+)\+(\d+)\+(\d+)", line)
                if match:
                    name = match.group(1)
                    screens.append(ScreenInfo(
                        index=idx, name=name,
                        width=int(match.group(2)), height=int(match.group(3)),
                        x=int(match.group(4)), y=int(match.group(5)),
                    ))
                    idx += 1
    except Exception as e:
        logger.debug(f"xrandr failed: {e}")

    if not screens:
        screens.append(ScreenInfo(index=0, name="Primary Display", width=1920, height=1080))
    return screens
