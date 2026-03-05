"""
MpvIPCBridge — controls mpv via IPC (Unix socket or Windows named pipe).

Launches mpv with --input-ipc-server, then communicates over the socket/pipe
to drive playback and poll position for FunGen timeline sync.

Protocol: newline-delimited JSON (MPV IPC v2 / JSON-IPC).
"""

import json
import os
import sys
import shutil
import socket
import subprocess
import threading
import time
import logging
from typing import Optional, Callable

logger = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform == "win32"
_DEFAULT_SOCK = r"\\.\pipe\fungen_mpv" if _IS_WINDOWS else "/tmp/fungen_mpv.sock"


def _find_mpv_binary() -> str:
    """Locate the mpv binary: PATH first, then known Homebrew location."""
    found = shutil.which("mpv")
    if found:
        return found
    homebrew_path = "/opt/homebrew/bin/mpv"
    if os.path.isfile(homebrew_path):
        return homebrew_path
    return "mpv"  # last-resort: let subprocess raise a clear error


class MpvIPCBridge:
    """
    Thin wrapper around mpv's JSON IPC socket.

    Usage:
        bridge = MpvIPCBridge(video_path)
        bridge.start()
        pos = bridge.get_position_ms()
        bridge.seek(5000)
        bridge.stop()

    Position callbacks are fired from a background event thread whenever
    mpv pushes a time-pos property-change event.
    Callback signature: callback(position_ms: float, duration_ms: float)
    """

    def __init__(
        self,
        video_path: str,
        mpv_binary: Optional[str] = None,
        sock_path: str = _DEFAULT_SOCK,
        fullscreen: bool = False,
        no_video: bool = False,          # headless: audio + IPC only
        start_ms: float = 0.0,
        poll_hz: int = 30,
        extra_args: Optional[list] = None,  # extra mpv CLI args appended to command
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
        self._sock: Optional[socket.socket] = None
        self._pipe = None  # Windows named pipe file handle
        self._sock_lock = threading.Lock()
        self._req_id = 0

        self._position_ms: float = 0.0
        self._duration_ms: float = 0.0
        self._is_playing: bool = False
        self._state_lock = threading.RLock()

        self._poller_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._position_callbacks: list[Callable] = []

        # Latency tracking
        self.last_poll_latency_ms: float = 0.0
        self.last_seek_latency_ms: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Launch mpv and connect to IPC socket/pipe. Returns True on success."""
        # Clean up stale socket (Unix only)
        if not _IS_WINDOWS and os.path.exists(self.sock_path):
            os.unlink(self.sock_path)

        cmd = [
            self.mpv_binary,
            self.video_path,
            f"--input-ipc-server={self.sock_path}",
            "--really-quiet",           # minimal terminal output
            "--keep-open=yes",          # don't close on EOF
            "--pause=yes",              # start paused so we control timing
        ]
        if self.fullscreen:
            cmd.append("--fullscreen")
        if self.no_video:
            # --vo=null: decode video without opening a display window.
            # Do NOT use --vid=no (leaves nothing to play if no audio track).
            cmd.append("--vo=null")
        if self.start_ms > 0:
            cmd.append(f"--start={self.start_ms / 1000.0:.3f}")
        cmd.extend(self.extra_args)

        logger.info(f"Launching mpv: {' '.join(cmd)}")
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait for IPC endpoint to appear (up to 5s)
        deadline = time.monotonic() + 5.0
        while True:
            if time.monotonic() > deadline:
                logger.error("mpv IPC endpoint did not appear within 5s")
                return False
            if self._proc.poll() is not None:
                logger.error(f"mpv exited early (code {self._proc.returncode})")
                return False
            if self._ipc_endpoint_ready():
                break
            time.sleep(0.05)

        # Connect
        if not self._connect():
            return False

        # Observe time-pos and pause state via push events
        self._send_cmd(["observe_property", 1, "time-pos"])
        self._send_cmd(["observe_property", 2, "pause"])
        self._send_cmd(["observe_property", 3, "duration"])

        # Start event loop thread (reads events pushed by mpv)
        self._stop_event.clear()
        self._poller_thread = threading.Thread(
            target=self._event_loop, name="mpv-ipc-poller", daemon=True
        )
        self._poller_thread.start()

        # Get duration
        dur = self._get_property("duration")
        if dur is not None:
            with self._state_lock:
                self._duration_ms = dur * 1000.0

        logger.info(f"MpvIPCBridge connected (duration {self._duration_ms:.0f}ms)")
        return True

    def _ipc_endpoint_ready(self) -> bool:
        """Check if the IPC endpoint (socket or named pipe) is available."""
        if _IS_WINDOWS:
            # On Windows, try to open the named pipe briefly
            try:
                h = open(self.sock_path, "r+b", buffering=0)
                h.close()
                return True
            except OSError:
                return False
        else:
            return os.path.exists(self.sock_path)

    def _connect(self) -> bool:
        """Establish IPC connection (socket on Unix, named pipe on Windows)."""
        if _IS_WINDOWS:
            try:
                self._pipe = open(self.sock_path, "r+b", buffering=0)
                logger.info(f"Connected to mpv named pipe: {self.sock_path}")
                return True
            except OSError as e:
                logger.error(f"Failed to connect to mpv named pipe: {e}")
                return False
        else:
            self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                self._sock.connect(self.sock_path)
                self._sock.settimeout(0.5)
                return True
            except OSError as e:
                logger.error(f"Failed to connect to mpv socket: {e}")
                return False

    def stop(self):
        """Stop mpv and close socket/pipe."""
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

        if self._pipe:
            try:
                self._pipe.close()
            except Exception:
                pass
            self._pipe = None

        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

        # Clean up Unix socket file (no-op on Windows)
        if not _IS_WINDOWS and os.path.exists(self.sock_path):
            os.unlink(self.sock_path)

        logger.info("MpvIPCBridge stopped")

    # ------------------------------------------------------------------
    # Playback control
    # ------------------------------------------------------------------

    def play(self):
        self._send_cmd(["set_property", "pause", False])
        with self._state_lock:
            self._is_playing = True

    def pause(self):
        self._send_cmd(["set_property", "pause", True])
        with self._state_lock:
            self._is_playing = False

    def seek(self, position_ms: float):
        """Seek to absolute position (milliseconds). Measures response latency."""
        t0 = time.monotonic()
        self._send_cmd(["seek", position_ms / 1000.0, "absolute"])
        # Latency = time until time-pos update is received (event loop)
        target = position_ms
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            with self._state_lock:
                if abs(self._position_ms - target) < 200:  # within 200ms
                    self.last_seek_latency_ms = (time.monotonic() - t0) * 1000.0
                    return
            time.sleep(0.005)
        self.last_seek_latency_ms = (time.monotonic() - t0) * 1000.0

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def add_position_callback(self, cb: Callable):
        """Register callback(position_ms, duration_ms) called on every position update."""
        self._position_callbacks.append(cb)

    def remove_position_callback(self, cb: Callable):
        """Remove a previously registered callback."""
        try:
            self._position_callbacks.remove(cb)
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # Internal — transport abstraction
    # ------------------------------------------------------------------

    def _next_req_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _send_raw(self, data: bytes):
        """Send raw bytes over the IPC transport."""
        if _IS_WINDOWS:
            if self._pipe is None:
                return
            self._pipe.write(data)
            self._pipe.flush()
        else:
            if self._sock is None:
                return
            self._sock.sendall(data)

    def _recv_raw(self, bufsize: int = 4096) -> bytes:
        """Receive raw bytes from the IPC transport. May raise OSError/timeout."""
        if _IS_WINDOWS:
            if self._pipe is None:
                return b""
            return self._pipe.read(bufsize) or b""
        else:
            if self._sock is None:
                return b""
            return self._sock.recv(bufsize)

    def _send_cmd(self, cmd: list) -> dict:
        req = {"command": cmd, "request_id": self._next_req_id()}
        payload = json.dumps(req) + "\n"
        with self._sock_lock:
            try:
                self._send_raw(payload.encode())
            except OSError:
                pass
        return {}

    def _get_property(self, name: str):
        """Synchronous property get (used at startup before event loop)."""
        req_id = self._next_req_id()
        req = {"command": ["get_property", name], "request_id": req_id}
        payload = json.dumps(req) + "\n"
        with self._sock_lock:
            try:
                self._send_raw(payload.encode())
                buf = b""
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline:
                    try:
                        chunk = self._recv_raw(4096)
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
                    except (socket.timeout, OSError):
                        break
            except OSError:
                pass
        return None

    def _event_loop(self):
        """Background thread: reads push events from mpv and updates state."""
        buf = b""
        while not self._stop_event.is_set():
            try:
                chunk = self._recv_raw(4096)
                if not chunk:
                    if _IS_WINDOWS:
                        # On Windows, pipe read returns empty on EOF
                        time.sleep(0.01)
                        continue
                    break
                buf += chunk
            except socket.timeout:
                continue
            except OSError:
                break

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
