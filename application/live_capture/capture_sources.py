"""
Capture Sources

Three ffmpeg-based capture backends:
- ScreenCaptureSource: full screen or region
- WindowCaptureSource: specific app window
- StreamCaptureSource: URL via yt-dlp + ffmpeg

All sources produce raw BGR24 frames on stdout for pipeline consumption.
"""

import sys
import shutil
import logging
import subprocess
import shlex
import threading
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List

logger = logging.getLogger(__name__)

_CREATION_FLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

# Cached macOS screen device index (avfoundation lists cameras before screens)
_macos_screen_device_index: Optional[int] = None


def _get_macos_screen_device_index() -> int:
    """Find the avfoundation device index for 'Capture screen 0'.

    avfoundation lists video devices in order: cameras first, then screens.
    We need to find the index of the first screen capture device.
    """
    global _macos_screen_device_index
    if _macos_screen_device_index is not None:
        return _macos_screen_device_index

    try:
        result = subprocess.run(
            [_find_ffmpeg(), "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True, text=True, timeout=5,
        )
        # Parse stderr for "[X] Capture screen" lines
        for line in result.stderr.splitlines():
            if "Capture screen" in line:
                # Format: "[AVFoundation ...] [2] Capture screen 0"
                bracket_start = line.rfind("[", 0, line.index("Capture screen"))
                if bracket_start >= 0:
                    bracket_end = line.index("]", bracket_start)
                    idx = int(line[bracket_start + 1:bracket_end])
                    _macos_screen_device_index = idx
                    return idx
    except Exception as e:
        logger.debug(f"Failed to detect avfoundation screen index: {e}")

    # Fallback: guess index 2 (common: 0=webcam, 1=phone, 2=screen)
    _macos_screen_device_index = 2
    return 2


def _find_ffmpeg() -> str:
    """Locate ffmpeg binary."""
    path = shutil.which("ffmpeg")
    if path:
        return path
    # Common fallback locations
    for candidate in ("/usr/local/bin/ffmpeg", "/opt/homebrew/bin/ffmpeg"):
        import os
        if os.path.isfile(candidate):
            return candidate
    return "ffmpeg"  # Hope it's on PATH


def _find_ytdlp() -> Optional[str]:
    """Locate yt-dlp binary, return None if not found."""
    return shutil.which("yt-dlp")


def _drain_stderr(proc: subprocess.Popen, label: str = ""):
    """Drain stderr from a subprocess in a background thread to prevent pipe deadlock.

    On Windows, subprocess stderr pipes have limited buffers (~64KB).
    If stderr is never read, the buffer fills and the process blocks,
    causing the stdout pipe to also stop producing data.
    """
    def _reader():
        try:
            while True:
                line = proc.stderr.readline()
                if not line:
                    break
                # Log at debug level to avoid noise (ffmpeg/yt-dlp are verbose)
                decoded = line.decode(errors='replace').rstrip()
                if decoded:
                    logger.debug(f"[{label}] {decoded}")
        except Exception:
            pass  # Process already terminated
    t = threading.Thread(target=_reader, daemon=True, name=f"stderr-drain-{label}")
    t.start()


class CaptureSource(ABC):
    """Base class for all capture sources."""

    def __init__(self, width: int = 640, height: int = 640):
        self.width = width
        self.height = height
        self._process: Optional[subprocess.Popen] = None
        self._pipe1_process: Optional[subprocess.Popen] = None

    @abstractmethod
    def start(self) -> subprocess.Popen:
        """Start capturing. Returns the process whose stdout emits raw BGR24 frames."""

    def stop(self):
        """Stop capturing and clean up processes."""
        for proc in (self._process, self._pipe1_process):
            if proc and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        self._process = None
        self._pipe1_process = None

    def get_frame_size(self) -> Tuple[int, int]:
        """Return (width, height) of output frames."""
        return (self.width, self.height)

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def frame_bytes(self) -> int:
        """Bytes per raw BGR24 frame."""
        return self.width * self.height * 3


class ScreenCaptureSource(CaptureSource):
    """Capture full screen or a region.

    macOS:  ffmpeg -f avfoundation -i "<screen_index>:" ...
    Windows: ffmpeg -f gdigrab -i desktop ...
    Linux:  ffmpeg -f x11grab -i :0.0+X,Y ...
    """

    def __init__(self, screen_index: int = 0, region: Optional[Tuple[int, int, int, int]] = None,
                 width: int = 640, height: int = 640, fps: int = 30):
        super().__init__(width, height)
        self.screen_index = screen_index
        self.region = region  # (x, y, w, h) or None for full screen
        self.fps = fps

    def start(self) -> subprocess.Popen:
        self.stop()
        ffmpeg = _find_ffmpeg()
        cmd = self._build_command(ffmpeg)
        logger.info(f"ScreenCapture CMD: {' '.join(shlex.quote(str(c)) for c in cmd)}")

        self._process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=-1, creationflags=_CREATION_FLAGS,
        )
        _drain_stderr(self._process, "screen-ffmpeg")
        return self._process

    def _build_command(self, ffmpeg: str) -> List[str]:
        vf_scale = f"scale={self.width}:{self.height}"

        if sys.platform == "darwin":
            # avfoundation device indices: cameras first, then screens.
            # self.screen_index is a logical screen number (0, 1, ...);
            # we offset by the first screen device index.
            device_idx = _get_macos_screen_device_index() + self.screen_index
            cmd = [ffmpeg, "-f", "avfoundation", "-framerate", str(self.fps),
                   "-i", f"{device_idx}:"]
            if self.region:
                x, y, w, h = self.region
                cmd.extend(["-vf", f"crop={w}:{h}:{x}:{y},{vf_scale}"])
            else:
                cmd.extend(["-vf", vf_scale])

        elif sys.platform == "win32":
            cmd = [ffmpeg, "-f", "gdigrab", "-framerate", str(self.fps)]
            if self.region:
                x, y, w, h = self.region
                cmd.extend(["-offset_x", str(x), "-offset_y", str(y),
                            "-video_size", f"{w}x{h}"])
            cmd.extend(["-i", "desktop", "-vf", vf_scale])

        else:  # Linux
            display = ":0.0"
            if self.region:
                x, y, w, h = self.region
                cmd = [ffmpeg, "-f", "x11grab", "-framerate", str(self.fps),
                       "-video_size", f"{w}x{h}", "-i", f"{display}+{x},{y}",
                       "-vf", vf_scale]
            else:
                cmd = [ffmpeg, "-f", "x11grab", "-framerate", str(self.fps),
                       "-i", display, "-vf", vf_scale]

        cmd.extend(["-f", "rawvideo", "-pix_fmt", "bgr24", "-an", "pipe:1"])
        return cmd


class WindowCaptureSource(CaptureSource):
    """Capture a specific application window.

    macOS:  uses avfoundation or screencapture
    Windows: gdigrab with -i title="<title>"
    Linux:  x11grab with xdotool geometry
    """

    def __init__(self, window_title: str = "", window_id: int = 0,
                 width: int = 640, height: int = 640, fps: int = 30):
        super().__init__(width, height)
        self.window_title = window_title
        self.window_id = window_id
        self.fps = fps

    def start(self) -> subprocess.Popen:
        self.stop()
        ffmpeg = _find_ffmpeg()
        cmd = self._build_command(ffmpeg)
        logger.info(f"WindowCapture CMD: {' '.join(shlex.quote(str(c)) for c in cmd)}")

        self._process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=-1, creationflags=_CREATION_FLAGS,
        )
        _drain_stderr(self._process, "window-ffmpeg")
        return self._process

    def _build_command(self, ffmpeg: str) -> List[str]:
        vf_scale = f"scale={self.width}:{self.height}"

        if sys.platform == "darwin":
            # macOS avfoundation can only capture full screens, not individual
            # windows. Capture the primary screen (same as screen capture).
            device_idx = _get_macos_screen_device_index()
            cmd = [ffmpeg, "-f", "avfoundation", "-framerate", str(self.fps),
                   "-capture_cursor", "0", "-i", f"{device_idx}:",
                   "-vf", vf_scale]

        elif sys.platform == "win32":
            # gdigrab supports -i title="Window Title"
            cmd = [ffmpeg, "-f", "gdigrab", "-framerate", str(self.fps),
                   "-i", f"title={self.window_title}",
                   "-vf", vf_scale]

        else:  # Linux
            # Use xdotool to get window geometry, then x11grab
            geom = self._get_linux_window_geometry()
            if geom:
                x, y, w, h = geom
                cmd = [ffmpeg, "-f", "x11grab", "-framerate", str(self.fps),
                       "-video_size", f"{w}x{h}", "-i", f":0.0+{x},{y}",
                       "-vf", vf_scale]
            else:
                # Fallback to full screen capture
                cmd = [ffmpeg, "-f", "x11grab", "-framerate", str(self.fps),
                       "-i", ":0.0", "-vf", vf_scale]

        cmd.extend(["-f", "rawvideo", "-pix_fmt", "bgr24", "-an", "pipe:1"])
        return cmd

    def _get_linux_window_geometry(self) -> Optional[Tuple[int, int, int, int]]:
        """Get window geometry on Linux using xdotool."""
        try:
            wid = self.window_id
            if not wid:
                return None
            result = subprocess.run(
                ["xdotool", "getwindowgeometry", "--shell", str(wid)],
                capture_output=True, text=True, timeout=3,
            )
            if result.returncode != 0:
                return None
            vals = {}
            for line in result.stdout.strip().splitlines():
                key, _, val = line.partition("=")
                vals[key] = int(val)
            return (vals.get("X", 0), vals.get("Y", 0),
                    vals.get("WIDTH", 640), vals.get("HEIGHT", 480))
        except Exception as e:
            logger.debug(f"xdotool geometry failed: {e}")
            return None


class StreamCaptureSource(CaptureSource):
    """Capture from a stream URL (Twitch, YouTube, etc.) via yt-dlp + ffmpeg.

    Falls back to direct ffmpeg if yt-dlp is not installed.
    """

    def __init__(self, url: str, width: int = 640, height: int = 640, fps: int = 30):
        super().__init__(width, height)
        self.url = url
        self.fps = fps

    def start(self) -> subprocess.Popen:
        self.stop()
        ffmpeg = _find_ffmpeg()
        ytdlp = _find_ytdlp()

        if ytdlp:
            return self._start_ytdlp_pipeline(ytdlp, ffmpeg)
        else:
            return self._start_direct_ffmpeg(ffmpeg)

    def _start_ytdlp_pipeline(self, ytdlp: str, ffmpeg: str) -> subprocess.Popen:
        """Use yt-dlp to extract stream, pipe to ffmpeg for frame output."""
        cmd1 = [ytdlp, "-f", "best", "-o", "-", self.url]
        cmd2 = [
            ffmpeg, "-i", "pipe:0",
            "-vf", f"scale={self.width}:{self.height}",
            "-r", str(self.fps),
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-an", "pipe:1",
        ]

        logger.info(f"StreamCapture (yt-dlp): {' '.join(shlex.quote(str(c)) for c in cmd1)} | "
                     f"{' '.join(shlex.quote(str(c)) for c in cmd2)}")

        self._pipe1_process = subprocess.Popen(
            cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=-1, creationflags=_CREATION_FLAGS,
        )
        _drain_stderr(self._pipe1_process, "stream-ytdlp")
        self._process = subprocess.Popen(
            cmd2, stdin=self._pipe1_process.stdout, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=-1, creationflags=_CREATION_FLAGS,
        )
        _drain_stderr(self._process, "stream-ffmpeg")
        # Allow pipe1 to receive SIGPIPE if process exits
        self._pipe1_process.stdout.close()
        return self._process

    def _start_direct_ffmpeg(self, ffmpeg: str) -> subprocess.Popen:
        """Direct ffmpeg for simple URLs (no yt-dlp)."""
        cmd = [
            ffmpeg,
            "-reconnect", "1", "-reconnect_streamed", "1",
            "-reconnect_delay_max", "5",
            "-i", self.url,
            "-vf", f"scale={self.width}:{self.height}",
            "-r", str(self.fps),
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-an", "pipe:1",
        ]
        logger.info(f"StreamCapture (direct): {' '.join(shlex.quote(str(c)) for c in cmd)}")

        self._process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=-1, creationflags=_CREATION_FLAGS,
        )
        _drain_stderr(self._process, "stream-direct-ffmpeg")
        return self._process

    @staticmethod
    def validate_url(url: str) -> Tuple[bool, str]:
        """Test if a URL is actually downloadable by fetching a few frames.

        yt-dlp --simulate only checks URL parseability, not whether the
        full yt-dlp | ffmpeg pipeline actually produces frames.  We run
        the real pipeline for a few seconds and check for output.
        """
        ffmpeg = _find_ffmpeg()
        ytdlp = _find_ytdlp()

        if not ytdlp:
            return (False, "yt-dlp not installed")

        try:
            # Actually test the full pipeline: yt-dlp -> ffmpeg -> read frames
            cmd1 = [ytdlp, "-f", "best", "-o", "-", url]
            cmd2 = [
                ffmpeg, "-i", "pipe:0",
                "-vf", "scale=64:64",
                "-frames:v", "3",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-an", "pipe:1",
            ]

            p1 = subprocess.Popen(
                cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                creationflags=_CREATION_FLAGS,
            )
            p2 = subprocess.Popen(
                cmd2, stdin=p1.stdout, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, creationflags=_CREATION_FLAGS,
            )
            p1.stdout.close()

            # Read a few frames (64x64x3 = 12288 bytes per frame)
            try:
                out, err = p2.communicate(timeout=20)
            except subprocess.TimeoutExpired:
                p2.kill()
                p1.kill()
                p2.wait()
                p1.wait()
                return (False, "Stream timed out - could not fetch frames within 20s")

            p1.wait(timeout=5)

            frame_size = 64 * 64 * 3
            if len(out) >= frame_size:
                return (True, "URL is valid (frames received)")

            # No frames - check yt-dlp stderr for clues
            p1_err = ""
            try:
                p1_err = p1.stderr.read().decode(errors='replace')[:300]
            except Exception:
                pass
            p2_err = err.decode(errors='replace')[:300] if err else ""
            detail = p1_err or p2_err or "No frames received"
            return (False, f"Stream failed: {detail}")

        except FileNotFoundError as e:
            return (False, f"Missing dependency: {e}")
        except Exception as e:
            return (False, str(e)[:200])
