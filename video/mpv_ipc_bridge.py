"""
MpvIPCBridge — FunGen-ready cross-platform IPC control for mpv.

Supports:
- Windows Named Pipe
- Unix domain socket

Keeps full functionality:
- play/pause/seek
- position callbacks
- synchronous get_property
- latency tracking
- fullscreen, no_video, start_ms, extra_args
"""

import json
import os
import shutil
import socket
import subprocess
import threading
import time
import logging
import sys
from typing import Optional, Callable

logger = logging.getLogger(__name__)

_DEFAULT_SOCK = r"\\.\pipe\fungen_mpv" if sys.platform == "win32" else "/tmp/fungen_mpv.sock"

def _find_mpv_binary() -> str:
    found = shutil.which("mpv")
    if found:
        return found
    # Example for macOS Homebrew, fallback
    homebrew_path = "/opt/homebrew/bin/mpv"
    if os.path.isfile(homebrew_path):
        return homebrew_path
    return "mpv"

class MpvIPCBridge:
    """FunGen-ready cross-platform mpv IPC bridge."""

    def __init__(
        self,
        video_path: str,
        mpv_binary: Optional[str] = None,
        sock_path: str = _DEFAULT_SOCK,
        fullscreen: bool = False,
        no_video: bool = False,
        start_ms: float = 0.0,
        poll_hz: int = 30,
        extra_args: Optional[list] = None,
    ):
        self.video_path = video_path
        self.mpv_binary = mpv_binary or _find_mpv_binary()
        self.sock_path = sock_path
        self.fullscreen = fullscreen
        self.no_video = no_video
        self.start_ms = start_ms
        self.poll_hz = poll_hz
        self.extra_args = extra_args or []

        self._proc: Optional[subprocess.Popen] = None
        self._sock = None  # socket (Unix) or file object (Windows pipe)
        self._sock_lock = threading.Lock()
        self._req_id = 0

        self._position_ms: float = 0.0
        self._duration_ms: float = 0.0
        self._is_playing: bool = False
        self._state_lock = threading.RLock()

        self._poller_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._position_callbacks: list[Callable] = []

        self.last_poll_latency_ms: float = 0.0
        self.last_seek_latency_ms: float = 0.0

    # -------------------------------
    # Lifecycle
    # -------------------------------

    def start(self) -> bool:
        """Launch mpv and connect to IPC."""
        if sys.platform != "win32" and os.path.exists(self.sock_path):
            os.unlink(self.sock_path)

        cmd = [
            self.mpv_binary,
            self.video_path,
            f"--input-ipc-server={self.sock_path}",
            "--really-quiet",
            "--keep-open=yes",
            "--pause=yes",
        ]
        if self.fullscreen:
            cmd.append("--fullscreen")
        if self.no_video:
            cmd.append("--vo=null")
        if self.start_ms > 0:
            cmd.append(f"--start={self.start_ms / 1000.0:.3f}")
        cmd.extend(self.extra_args)

        logger.info(f"Launching mpv: {' '.join(cmd)}")
        self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Wait for IPC availability and connect (unified retry loop)
        deadline = time.monotonic() + 5.0
        connected = False
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                logger.error(f"mpv exited early (code {self._proc.returncode})")
                return False
            if sys.platform == "win32":
                try:
                    import win32file
                    import msvcrt
                    handle = win32file.CreateFile(
                        self.sock_path,
                        win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                        0, None,
                        win32file.OPEN_EXISTING,
                        0, None
                    )
                    self._sock = os.fdopen(
                        msvcrt.open_osfhandle(handle.Detach(), os.O_RDWR | os.O_BINARY),
                        'r+b', buffering=0
                    )
                    connected = True
                    break
                except Exception:
                    time.sleep(0.05)
            else:
                if not os.path.exists(self.sock_path):
                    time.sleep(0.05)
                    continue
                try:
                    self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    self._sock.connect(self.sock_path)
                    self._sock.settimeout(0.5)
                    connected = True
                    break
                except Exception:
                    time.sleep(0.05)

        if not connected:
            logger.error("mpv IPC socket/pipe did not appear within 5s")
            return False

        # Observe required properties
        self._send_cmd(["observe_property", 1, "time-pos"])
        self._send_cmd(["observe_property", 2, "pause"])
        self._send_cmd(["observe_property", 3, "duration"])

        # Start event loop
        self._stop_event.clear()
        self._poller_thread = threading.Thread(target=self._event_loop, name="mpv-ipc-poller", daemon=True)
        self._poller_thread.start()

        # Get duration synchronously
        dur = self._get_property("duration")
        if dur is not None:
            with self._state_lock:
                self._duration_ms = dur * 1000.0

        logger.info(f"MpvIPCBridge connected (duration {self._duration_ms:.0f}ms)")
        return True

    def stop(self):
        """Stop mpv and close IPC."""
        self._stop_event.set()
        if self._poller_thread and self._poller_thread.is_alive():
            self._poller_thread.join(timeout=2.0)

        try:
            self._send_cmd(["quit"])
        except Exception:
            pass

        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

        if sys.platform != "win32" and os.path.exists(self.sock_path):
            os.unlink(self.sock_path)

        logger.info("MpvIPCBridge stopped")

    # -------------------------------
    # Playback
    # -------------------------------

    def play(self):
        self._send_cmd(["set_property", "pause", False])
        with self._state_lock:
            self._is_playing = True

    def pause(self):
        self._send_cmd(["set_property", "pause", True])
        with self._state_lock:
            self._is_playing = False

    def seek(self, position_ms: float):
        t0 = time.monotonic()
        self._send_cmd(["seek", position_ms / 1000.0, "absolute"])
        target = position_ms
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            with self._state_lock:
                if abs(self._position_ms - target) < 200:
                    self.last_seek_latency_ms = (time.monotonic() - t0) * 1000.0
                    return
            time.sleep(0.005)
        self.last_seek_latency_ms = (time.monotonic() - t0) * 1000.0

    # -------------------------------
    # State
    # -------------------------------

    def get_position_ms(self) -> float:
        with self._state_lock:
            return self._position_ms

    def get_duration_ms(self) -> float:
        with self._state_lock:
            return self._duration_ms

    @property
    def is_playing(self) -> bool:
        with self._state_lock:
            return self._is_playing

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # -------------------------------
    # Callbacks
    # -------------------------------

    def add_position_callback(self, cb: Callable):
        self._position_callbacks.append(cb)

    def remove_position_callback(self, cb: Callable):
        try:
            self._position_callbacks.remove(cb)
        except ValueError:
            pass

    # -------------------------------
    # Internal
    # -------------------------------

    def _next_req_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _send_cmd(self, cmd: list) -> dict:
        req = {"command": cmd, "request_id": self._next_req_id()}
        payload = (json.dumps(req) + "\n").encode()
        with self._sock_lock:
            if self._sock is None:
                return {}
            try:
                if sys.platform == "win32":
                    self._sock.write(payload)
                    self._sock.flush()
                else:
                    self._sock.sendall(payload)
            except Exception:
                return {}
        return {}

    def _get_property(self, name: str):
        req_id = self._next_req_id()
        req = {"command": ["get_property", name], "request_id": req_id}
        payload = (json.dumps(req) + "\n").encode()
        with self._sock_lock:
            if self._sock is None:
                return None
            try:
                buf = b""
                deadline = time.monotonic() + 1.0
                if sys.platform == "win32":
                    self._sock.write(payload)
                    self._sock.flush()
                    while time.monotonic() < deadline:
                        chunk = self._sock.read(4096)
                        if not chunk:
                            break
                        buf += chunk
                        for line in buf.split(b"\n"):
                            if not line.strip():
                                continue
                            try:
                                obj = json.loads(line)
                                if obj.get("request_id") == req_id:
                                    return obj.get("data")
                            except json.JSONDecodeError:
                                pass
                        if b"\n" in buf:
                            buf = buf.split(b"\n")[-1]
                else:
                    self._sock.sendall(payload)
                    while time.monotonic() < deadline:
                        try:
                            chunk = self._sock.recv(4096)
                            if not chunk:
                                break
                            buf += chunk
                            for line in buf.split(b"\n"):
                                if not line.strip():
                                    continue
                                try:
                                    obj = json.loads(line)
                                    if obj.get("request_id") == req_id:
                                        return obj.get("data")
                                except json.JSONDecodeError:
                                    pass
                            if b"\n" in buf:
                                buf = buf.split(b"\n")[-1]
                        except socket.timeout:
                            break
            except Exception:
                return None
        return None

    def _event_loop(self):
        buf = b""
        while not self._stop_event.is_set():
            if self._sock is None:
                break
            try:
                if sys.platform == "win32":
                    chunk = self._sock.read(4096)
                else:
                    chunk = self._sock.recv(4096)
                if not chunk:
                    break
                buf += chunk
            except (OSError, socket.timeout):
                continue

            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                t_poll = time.monotonic()
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                event = obj.get("event")
                if event == "property-change":
                    prop_id = obj.get("id")
                    data = obj.get("data")
                    if prop_id == 1 and data is not None:  # time-pos
                        self.last_poll_latency_ms = (time.monotonic() - t_poll) * 1000.0
                        pos_ms = data * 1000.0
                        with self._state_lock:
                            self._position_ms = pos_ms
                        for cb in self._position_callbacks:
                            try:
                                cb(pos_ms, self._duration_ms)
                            except Exception:
                                pass
                    elif prop_id == 2 and data is not None:  # pause
                        with self._state_lock:
                            self._is_playing = not data
                    elif prop_id == 3 and data is not None:  # duration
                        with self._state_lock:
                            self._duration_ms = data * 1000.0
