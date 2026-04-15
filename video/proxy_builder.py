"""Proxy (edit-time transcode) builder.

For VR sources, run a single ffmpeg invocation that v360-unwarps to a flat
1920x1080 HEVC file with AAC 160k audio. The proxy is written next to the
original with a fixed filename suffix and a sidecar JSON that tracks the
source's mtime + frame count so stale proxies can be detected.

Frame timing is preserved verbatim (fps_mode=passthrough + copyts) so
funscript timestamps remain valid when the editor swaps its active source
to the proxy. If the post-encode frame count drifts, the proxy is deleted.

This module is pure: no imgui, no app singleton. A ProxyBuilder instance
runs an encode with progress/cancel callbacks. UI dialogs call into it.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

PROXY_SUFFIX = ".fungen-proxy.mp4"
SIDECAR_SUFFIX = ".fungen-proxy.json"
REGISTRY_PATH = os.path.join(
    os.path.expanduser("~"), ".fungen", "proxies.json",
)
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
# All-I-frame (keyint=1) encode. Every frame is a keyframe so arrow-nav and
# scrub are instant regardless of direction. Larger file (~2-3x a GOP encode)
# but this is an edit proxy, not an archival copy. Matches the community
# "Iframer" convention scripters already expect.
VIDEO_BITRATE = "15M"
AUDIO_BITRATE = "160k"

# v360 presets keyed by our internal vr_input_format string. Only the
# projections we actually handle at read-time are mirrored here; anything
# else falls back to "he:in_stereo=0" (mono equirect) which is a reasonable
# default and still produces a usable preview.
_V360_CHAIN = {
    "he_sbs":   "v360=he:in_stereo=sbs:output=flat:w={w}:h={h}:v_fov=90:h_fov=90",
    "he_tb":    "v360=he:in_stereo=tb:output=flat:w={w}:h={h}:v_fov=90:h_fov=90",
    "fisheye_sbs": "v360=fisheye:ih_fov=190:iv_fov=190:in_stereo=sbs:output=flat:w={w}:h={h}:v_fov=100:h_fov=100",
    "fisheye_tb":  "v360=fisheye:ih_fov=190:iv_fov=190:in_stereo=tb:output=flat:w={w}:h={h}:v_fov=100:h_fov=100",
}


# ----------------------------------------------------------------- helpers

def proxy_path_for(source_path: str) -> str:
    """Return the conventional proxy path *next to* the source video.

    For cases where the proxy is stored elsewhere (output folder, custom
    folder), callers should use ``resolve_proxy_target_path`` instead. The
    sidecar next to the source always records the actual proxy location,
    so discovery on re-open still works.
    """
    base, _ = os.path.splitext(source_path)
    return base + PROXY_SUFFIX


def resolve_proxy_target_path(source_path: str,
                               mode: str,
                               output_folder: str = "",
                               custom_folder: str = "") -> str:
    """Decide where to write the proxy, based on the user's setting.

    Modes:
      - "next_to_source": alongside the source file.
      - "output_folder": ``<output_folder>/<basename>.fungen-proxy.mp4``.
      - "custom": ``<custom_folder>/<basename>.fungen-proxy.mp4``.
    Falls back to next-to-source on an unknown mode or a missing path.
    """
    base_noext = os.path.splitext(os.path.basename(source_path))[0]
    filename = base_noext + PROXY_SUFFIX
    if mode == "output_folder" and output_folder:
        os.makedirs(output_folder, exist_ok=True)
        return os.path.join(output_folder, filename)
    if mode == "custom" and custom_folder:
        os.makedirs(custom_folder, exist_ok=True)
        return os.path.join(custom_folder, filename)
    # default: next to source
    return proxy_path_for(source_path)


def sidecar_path_for(source_path: str) -> str:
    base, _ = os.path.splitext(source_path)
    return base + SIDECAR_SUFFIX


def is_proxy_filename(path: str) -> bool:
    return path.endswith(PROXY_SUFFIX)


def read_sidecar(source_path: str) -> Optional[dict]:
    p = sidecar_path_for(source_path)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def write_sidecar(source_path: str, proxy_path: str, nb_frames: int,
                  fps: float, preset: str) -> None:
    data = {
        "source_path": source_path,
        "source_mtime": os.path.getmtime(source_path),
        "source_size": os.path.getsize(source_path),
        "source_nb_frames": nb_frames,
        "source_fps": fps,
        "proxy_path": proxy_path,
        "proxy_nb_frames": nb_frames,
        "preset": preset,
        "created_at": time.time(),
    }
    try:
        with open(sidecar_path_for(source_path), "w") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass


def _load_registry() -> list:
    try:
        with open(REGISTRY_PATH, "r") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _save_registry(entries: list) -> None:
    try:
        os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
        with open(REGISTRY_PATH, "w") as f:
            json.dump(entries, f, indent=2)
    except OSError:
        pass


def registry_add(source_path: str, proxy_path: str) -> None:
    """Record a completed proxy so the UI can list all known proxies globally."""
    entries = _load_registry()
    entries = [e for e in entries if e.get("proxy_path") != proxy_path]
    entries.append({
        "source_path": source_path,
        "proxy_path": proxy_path,
        "created_at": time.time(),
    })
    _save_registry(entries)


def registry_remove(proxy_path: str) -> None:
    entries = _load_registry()
    entries = [e for e in entries if e.get("proxy_path") != proxy_path]
    _save_registry(entries)


def registry_list() -> list:
    """Return registry entries, filtering out ones whose proxy file is gone.
    Also refreshes sizes from disk."""
    entries = _load_registry()
    live = []
    for e in entries:
        pp = e.get("proxy_path", "")
        if not pp or not os.path.exists(pp):
            continue
        try:
            e["proxy_size_bytes"] = os.path.getsize(pp)
        except OSError:
            e["proxy_size_bytes"] = 0
        e["source_exists"] = os.path.exists(e.get("source_path", ""))
        live.append(e)
    if len(live) != len(entries):
        _save_registry(live)
    return live


def delete_proxy(proxy_path: str) -> bool:
    """Delete a proxy file + its sidecar + registry entry."""
    import send2trash
    ok = True
    for p in (proxy_path, proxy_path.replace(PROXY_SUFFIX, SIDECAR_SUFFIX)):
        if os.path.exists(p):
            try:
                send2trash.send2trash(p)
            except Exception:
                ok = False
    registry_remove(proxy_path)
    return ok


def is_valid_proxy(source_path: str) -> bool:
    """True iff a matching proxy + sidecar exist AND the source hasn't been
    modified since the proxy was built. The proxy itself may live anywhere
    (next to source, in the output folder, or in a custom dir); the sidecar
    stores the real ``proxy_path``."""
    sc = read_sidecar(source_path)
    if not sc:
        return False
    pp = sc.get("proxy_path") or proxy_path_for(source_path)
    if not os.path.exists(pp):
        return False
    try:
        cur_mtime = os.path.getmtime(source_path)
        cur_size = os.path.getsize(source_path)
    except OSError:
        return False
    if abs(cur_mtime - sc.get("source_mtime", 0)) > 1.0:
        return False
    if cur_size != sc.get("source_size", -1):
        return False
    return True


def proxy_path_from_sidecar(source_path: str) -> Optional[str]:
    """Return the recorded proxy path for a source (from its sidecar), or
    None if no valid proxy is registered."""
    if not is_valid_proxy(source_path):
        return None
    sc = read_sidecar(source_path) or {}
    return sc.get("proxy_path") or proxy_path_for(source_path)


# ----------------------------------------------- encoder / hwaccel detection

_ENCODER_PROBE_CACHE: Optional[str] = None


def detect_hevc_encoder(logger: Optional[logging.Logger] = None) -> str:
    """Pick the best HEVC encoder available on this host. Probes once and
    caches. Always succeeds: libx265 is the CPU fallback."""
    global _ENCODER_PROBE_CACHE
    if _ENCODER_PROBE_CACHE is not None:
        return _ENCODER_PROBE_CACHE

    candidates_by_platform = {
        "Darwin": ["hevc_videotoolbox", "libx265"],
        "Windows": ["hevc_nvenc", "hevc_qsv", "hevc_amf", "libx265"],
        "Linux": ["hevc_nvenc", "hevc_vaapi", "hevc_qsv", "libx265"],
    }
    preferred = candidates_by_platform.get(platform.system(), ["libx265"])

    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except (OSError, subprocess.TimeoutExpired) as e:
        if logger:
            logger.warning(f"ffmpeg -encoders probe failed: {e}; defaulting to libx265")
        _ENCODER_PROBE_CACHE = "libx265"
        return _ENCODER_PROBE_CACHE

    available = {line.strip().split()[1] for line in out.splitlines()
                 if line.strip().startswith("V") and len(line.strip().split()) > 1}
    for enc in preferred:
        if enc in available:
            _ENCODER_PROBE_CACHE = enc
            if logger:
                logger.info(f"Proxy encoder selected: {enc}")
            return enc
    _ENCODER_PROBE_CACHE = "libx265"
    return _ENCODER_PROBE_CACHE


def _decode_hwaccel_args() -> list:
    """Decode hwaccel args for the source read.

    We deliberately run the decode on CPU. The output filter chain uses
    CPU-side filters (v360, split, fps, scale, format), so any hwaccel that
    leaves frames in GPU memory (e.g. cuda with the default
    ``hwaccel_output_format``) would force a silent GPU->CPU download, and
    some builds (notably Windows + nvenc) fail that round-trip with
    ``Could not open encoder before EOF``. CPU decode is cheap enough for
    a one-shot proxy re-encode and keeps the path portable.
    """
    return []


# ----------------------------------------------------------------- job type

@dataclass
class ProxyJob:
    source_path: str
    vr_input_format: str          # 'he_sbs' etc; must match VideoProcessor
    duration_s: float
    source_nb_frames: int
    source_fps: float
    target_path: str = ""
    # Optional callbacks: (fraction[0..1], out_time_s, speed_x, eta_s)
    progress_cb: Optional[Callable[[float, float, float, float], None]] = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    # Live-preview JPEG: if set, ffmpeg writes a 2fps/480-wide JPEG here
    # (overwriting the same file). Consumer polls and uploads to a GL
    # texture. Left None = no preview output.
    preview_path: Optional[str] = None

    def __post_init__(self):
        if not self.target_path:
            self.target_path = proxy_path_for(self.source_path)


# ----------------------------------------------------------------- builder

class ProxyBuilder:
    """Runs one ffmpeg encode synchronously on the calling thread. Call from
    a background thread; the caller's thread returns only after success,
    cancel, or error."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("ProxyBuilder")

    def build_command(self, job: ProxyJob, encoder: str) -> list:
        # VR source -> v360 unwarp; 2D source -> plain scale with letterbox.
        if job.vr_input_format and job.vr_input_format != "2d":
            vf_core = _V360_CHAIN.get(
                job.vr_input_format,
                _V360_CHAIN["he_sbs"],
            ).format(w=TARGET_WIDTH, h=TARGET_HEIGHT)
        else:
            vf_core = (f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}"
                       f":force_original_aspect_ratio=decrease,"
                       f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}"
                       f":(ow-iw)/2:(oh-ih)/2:black")

        # Decode stays on CPU regardless of target encoder (see
        # _decode_hwaccel_args docstring). The encoder itself runs on GPU
        # when available (videotoolbox / nvenc / qsv / vaapi / amf).
        decode_args = _decode_hwaccel_args()

        # All-I-frame args. `-g 1` forces a keyframe every frame on most
        # encoders; some HW encoders also need explicit keyint params.
        iframe_args = ["-g", "1", "-keyint_min", "1"]
        if encoder == "libx265":
            iframe_args += ["-x265-params", "keyint=1:min-keyint=1:no-open-gop=1"]
        elif encoder == "hevc_nvenc":
            iframe_args += ["-no-scenecut", "1"]
        elif encoder == "hevc_videotoolbox":
            # videotoolbox honors -g; no extra switches needed.
            pass
        elif encoder == "hevc_qsv":
            iframe_args += ["-look_ahead", "0"]
        elif encoder in ("hevc_vaapi", "hevc_amf"):
            pass  # -g 1 is sufficient

        # Build the main video filter chain. If a preview JPEG is requested,
        # split the filtered output so the preview uses the already-unwarped
        # frames (no double v360 cost).
        if job.preview_path:
            # 1 preview frame every 7 seconds; scaled to 480 wide.
            filter_complex = (
                f"[0:v]{vf_core},split=2[vout][vprev];"
                f"[vprev]fps=1/7,scale=480:-2[preview]"
            )
            video_in = ["-filter_complex", filter_complex, "-map", "[vout]",
                        "-map", "0:a?"]
            preview_out = ["-map", "[preview]",
                           "-q:v", "5", "-update", "1",
                           "-f", "image2", job.preview_path]
        else:
            video_in = ["-filter:v", vf_core]
            preview_out = []

        cmd = [
            "ffmpeg", "-hide_banner", "-nostats", "-loglevel", "error",
            "-y",
            *decode_args,
            "-i", job.source_path,
            # Global output: progress pipe applies to the whole run.
            "-progress", "pipe:1",
            *video_in,
            # --- Main output (proxy MP4) -----------------------------------
            # All flags below apply only to the next output file.
            "-c:v", encoder,
            "-b:v", VIDEO_BITRATE,
            "-pix_fmt", "yuv420p",
            *iframe_args,
            "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-ac", "2",
            "-movflags", "+faststart",
            # passthrough + copyts preserve frame timing for funscript parity.
            # Scoped here (before the file) so they DON'T leak into the
            # preview output, whose fps=1/7 filter emits sparse timestamps
            # that break image2 with -copyts.
            "-fps_mode", "passthrough", "-copyts",
            # Explicit muxer: filename ends in ".partial" so ffmpeg can't
            # infer the format. Always MP4.
            "-f", "mp4",
            job.target_path + ".partial",
            # --- Preview output (sparse JPEGs) -----------------------------
            *preview_out,
        ]
        return cmd

    def encode(self, job: ProxyJob) -> bool:
        """Run the encode. Returns True on success. Writes sidecar on success,
        cleans up .partial on failure or cancel."""
        encoder = detect_hevc_encoder(self.logger)
        cmd = self.build_command(job, encoder)
        self.logger.info(f"Proxy encode start: {os.path.basename(job.source_path)} "
                         f"-> {os.path.basename(job.target_path)} ({encoder})")
        self.logger.debug("ffmpeg: " + " ".join(cmd))

        partial_path = job.target_path + ".partial"
        if os.path.exists(partial_path):
            try: os.remove(partial_path)
            except OSError: pass

        creation_flags = (subprocess.CREATE_NO_WINDOW
                          if sys.platform == "win32" else 0)
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                creationflags=creation_flags,
            )
        except FileNotFoundError:
            self.logger.error("ffmpeg executable not found on PATH")
            return False

        self._parse_progress(proc, job)

        # Ensure we've waited on the process regardless of how parse_progress exited.
        if proc.poll() is None:
            if job.cancel_event.is_set():
                self._kill(proc)
            else:
                proc.wait(timeout=10)

        ok = (proc.returncode == 0 and not job.cancel_event.is_set())
        stderr_tail = ""
        if proc.stderr is not None:
            try:
                stderr_tail = proc.stderr.read()[-1000:]
            except Exception:
                pass

        if not ok:
            if job.cancel_event.is_set():
                self.logger.info("Proxy encode canceled")
            else:
                self.logger.error(f"Proxy encode failed (rc={proc.returncode})"
                                  + (f": {stderr_tail.strip()}" if stderr_tail else ""))
            self._cleanup_partial(partial_path)
            return False

        # Verify frame parity.
        proxy_frames = self._probe_frame_count(partial_path)
        if (proxy_frames is not None and job.source_nb_frames > 0
                and abs(proxy_frames - job.source_nb_frames) > 5):
            self.logger.error(f"Proxy frame count drift: source={job.source_nb_frames} "
                              f"proxy={proxy_frames}; discarding")
            self._cleanup_partial(partial_path)
            return False

        # Rename .partial -> final.
        try:
            if os.path.exists(job.target_path):
                os.remove(job.target_path)
            shutil.move(partial_path, job.target_path)
        except OSError as e:
            self.logger.error(f"Proxy rename failed: {e}")
            self._cleanup_partial(partial_path)
            return False

        write_sidecar(job.source_path, job.target_path,
                      nb_frames=(proxy_frames or job.source_nb_frames),
                      fps=job.source_fps, preset="flatten_vr_1080p_iframe")
        registry_add(job.source_path, job.target_path)
        self.logger.info(f"Proxy ready: {job.target_path}")
        return True

    # ------------------------------------------------------------- internals

    _PROGRESS_RE = re.compile(r"^([a-zA-Z_]+)=(.+)$")

    def _parse_progress(self, proc: subprocess.Popen, job: ProxyJob) -> None:
        """Consume ffmpeg's `-progress pipe:1` stream until the process exits
        or cancel is requested."""
        out_time_ms = 0
        speed = 0.0
        last_cb_ts = 0.0
        assert proc.stdout is not None
        for raw in proc.stdout:
            if job.cancel_event.is_set():
                self._kill(proc)
                return
            line = raw.strip()
            if not line:
                continue
            m = self._PROGRESS_RE.match(line)
            if not m:
                continue
            key, val = m.group(1), m.group(2).strip()
            if key == "out_time_ms":
                try: out_time_ms = int(val)
                except ValueError: pass
            elif key == "speed":
                try: speed = float(val.rstrip("x").strip() or "0")
                except ValueError: speed = 0.0
            elif key == "progress":
                if val == "end":
                    return
            now = time.time()
            if now - last_cb_ts >= 0.25 and job.progress_cb:
                last_cb_ts = now
                out_time_s = out_time_ms / 1_000_000.0
                frac = 0.0
                eta_s = 0.0
                if job.duration_s > 0:
                    frac = max(0.0, min(1.0, out_time_s / job.duration_s))
                    remaining = max(0.0, job.duration_s - out_time_s)
                    eta_s = remaining / speed if speed > 0 else 0.0
                try:
                    job.progress_cb(frac, out_time_s, speed, eta_s)
                except Exception as e:
                    self.logger.debug(f"progress_cb error: {e}")

    def _kill(self, proc: subprocess.Popen) -> None:
        try:
            proc.terminate()
            try: proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:
            pass

    def _cleanup_partial(self, path: str) -> None:
        if os.path.exists(path):
            try: os.remove(path)
            except OSError: pass

    def _probe_frame_count(self, path: str) -> Optional[int]:
        """Best-effort frame count via ffprobe. Returns None if unavailable."""
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error",
                 "-select_streams", "v:0",
                 "-count_packets",
                 "-show_entries", "stream=nb_read_packets",
                 "-of", "default=nokey=1:noprint_wrappers=1",
                 path],
                capture_output=True, text=True, timeout=60,
            )
            if out.returncode != 0:
                return None
            s = out.stdout.strip()
            return int(s) if s.isdigit() else None
        except (OSError, subprocess.TimeoutExpired, ValueError):
            return None


# ---------------------------------------------------- suggest-on-open helper

def should_suggest_proxy(video_info: dict, determined_video_type: str,
                         min_size_gb: float = 1.5,
                         min_2d_height: int = 2160) -> bool:
    """Decide whether the open_video hook should pop the suggestion dialog.

    VR: any source past the size threshold. 2D: source with height >= 4K
    (default 2160) past the size threshold. Smaller 2D is fine as-is.
    """
    size = int(video_info.get("file_size") or video_info.get("file_size_bytes") or 0)
    if size < min_size_gb * (1024 ** 3):
        return False
    if determined_video_type == "VR":
        return True
    height = int(video_info.get("height", 0) or 0)
    return height >= min_2d_height
