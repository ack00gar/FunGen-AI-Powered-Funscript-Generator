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
# 2D sources -> 1920x1080 (standard editor 16:9).
# VR sources, after v360 unwarp -> 1080x1080 (square, matches per-eye
# viewport; no horizontal FOV stretch, no wasted letterbox pixels).
TARGET_2D_W, TARGET_2D_H = 1920, 1080
TARGET_VR_W, TARGET_VR_H = 1080, 1080
# All-I-frame (keyint=1) encode. Every frame is a keyframe so arrow-nav and
# scrub are instant regardless of direction. Larger file (~2-3x a GOP encode)
# but this is an edit proxy, not an archival copy. Matches the community
# "Iframer" convention scripters already expect.
VIDEO_BITRATE = "15M"
AUDIO_BITRATE = "160k"


def _target_dims_for(job) -> tuple:
    """(width, height) for the proxy, based on source kind."""
    if job.vr_input_format and job.vr_input_format != "2d":
        return TARGET_VR_W, TARGET_VR_H
    return TARGET_2D_W, TARGET_2D_H

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

    def _filter_chain_str(self, job: ProxyJob) -> str:
        """Filter chain applied inside our PyAV Graph (NOT passed to ffmpeg)."""
        w, h = _target_dims_for(job)
        if job.vr_input_format and job.vr_input_format != "2d":
            return _V360_CHAIN.get(
                job.vr_input_format, _V360_CHAIN["he_sbs"],
            ).format(w=w, h=h) + ",format=bgr24"
        return (f"scale={w}:{h}"
                f":force_original_aspect_ratio=decrease,"
                f"pad={w}:{h}"
                f":(ow-iw)/2:(oh-ih)/2:black,format=bgr24")

    def build_encoder_command(self, job: ProxyJob, encoder: str,
                              width: int, height: int, fps: float) -> list:
        """ffmpeg command for the ENCODE-ONLY subprocess.

        Video comes in as raw BGR24 on stdin (pipe:0). Audio is read
        directly from the source file by ffmpeg (second input). This
        mirrors the Stage-1 FFmpegEncoder pattern which avoids all the
        filter_complex / hwaccel pitfalls we hit on Windows + nvenc.
        """
        # All-I-frame: every frame is a keyframe. -bf 0 is mandatory for
        # nvenc (defaults to 3 B-frames, which rejects GOP<B+1) and safe
        # for every other encoder since B-frames can't exist at GOP=1.
        iframe_args = ["-g", "1", "-keyint_min", "1", "-bf", "0"]
        if encoder == "libx265":
            iframe_args += ["-x265-params",
                            "keyint=1:min-keyint=1:no-open-gop=1:bframes=0"]
        elif encoder == "hevc_nvenc":
            iframe_args += ["-no-scenecut", "1"]
        elif encoder == "hevc_qsv":
            iframe_args += ["-look_ahead", "0"]

        return [
            "ffmpeg", "-hide_banner", "-nostats", "-loglevel", "warning",
            "-y",
            # Input 0: rawvideo from our stdin pipe.
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", f"{fps:.6f}",
            "-i", "pipe:0",
            # Input 1: source file (for audio only).
            "-i", job.source_path,
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c:v", encoder,
            "-b:v", VIDEO_BITRATE,
            "-pix_fmt", "yuv420p",
            *iframe_args,
            "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-ac", "2",
            "-movflags", "+faststart",
            "-f", "mp4",
            job.target_path + ".partial",
        ]

    def _open_decode_graph(self, job: ProxyJob):
        """Open the source with PyAV and build the unwarp/scale filter graph.

        Returns (container, stream, graph, fps, nb_frames) or raises.
        """
        import av
        container = av.open(job.source_path)
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        fps = float(stream.average_rate or stream.guessed_rate or job.source_fps or 30.0)
        nb_frames = int(stream.frames or 0)
        if nb_frames <= 0 and job.source_nb_frames > 0:
            nb_frames = job.source_nb_frames

        graph = av.filter.Graph()
        src = graph.add_buffer(template=stream)
        chain = self._filter_chain_str(job).strip().strip(",")
        prev = src
        for spec in chain.split(","):
            spec = spec.strip()
            if not spec:
                continue
            name, _, args = spec.partition("=")
            node = graph.add(name.strip(), args.strip())
            prev.link_to(node)
            prev = node
        sink = graph.add("buffersink")
        prev.link_to(sink)
        graph.configure()
        return container, stream, graph, fps, nb_frames

    def encode(self, job: ProxyJob) -> bool:
        """Stage 1-style encode: PyAV decode + v360 in Python, raw BGR piped
        to an ffmpeg encoder subprocess, audio copied from the source file.
        Preview JPEGs are written directly by this method (no side output
        from ffmpeg). Safe on any platform where PyAV can open the source
        and where ffmpeg can run the target encoder."""
        import av
        import cv2
        encoder = detect_hevc_encoder(self.logger)

        # Stage 1: open source + build filter graph up-front so we fail
        # fast with a clear Python exception if filters are misconfigured
        # (instead of silently through an ffmpeg stderr tail).
        try:
            container, stream, graph, fps, nb_frames = self._open_decode_graph(job)
        except Exception as e:
            self.logger.error(f"PyAV decode setup failed: {e}", exc_info=True)
            return False

        partial_path = job.target_path + ".partial"
        if os.path.exists(partial_path):
            try: os.remove(partial_path)
            except OSError: pass

        w, h = _target_dims_for(job)
        cmd = self.build_encoder_command(job, encoder, w, h, fps)
        self.logger.info(f"Proxy encode start: {os.path.basename(job.source_path)} "
                         f"-> {os.path.basename(job.target_path)} ({encoder})")
        self.logger.info("ffmpeg cmd: " + " ".join(cmd))

        creation_flags = (subprocess.CREATE_NO_WINDOW
                          if sys.platform == "win32" else 0)
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=False,
                creationflags=creation_flags,
            )
        except FileNotFoundError:
            self.logger.error("ffmpeg executable not found on PATH")
            try: container.close()
            except Exception: pass
            return False

        # Drain ffmpeg stderr in a background thread so its pipe buffer
        # never fills up and blocks the encoder.
        stderr_lines: list = []
        def _drain_stderr():
            if proc.stderr is None:
                return
            try:
                for raw in iter(proc.stderr.readline, b""):
                    line = raw.decode("utf-8", errors="replace").rstrip()
                    if line:
                        stderr_lines.append(line)
            except Exception:
                pass
        stderr_thread = threading.Thread(
            target=_drain_stderr, daemon=True, name="ProxyFFmpegStderr")
        stderr_thread.start()

        # Feed loop.
        frames_written = 0
        last_preview_frame = -10 ** 9
        PREVIEW_EVERY = max(1, int(fps * 7))  # ~1 JPEG every 7 seconds
        t_start = time.time()
        write_failed = False

        try:
            for packet in container.demux(stream):
                if job.cancel_event.is_set():
                    break
                for frame in packet.decode():
                    if job.cancel_event.is_set():
                        break
                    graph.push(frame)
                    while True:
                        try:
                            out = graph.pull()
                        except (av.BlockingIOError, av.FFmpegError):
                            break
                        arr = out.to_ndarray(format="bgr24")
                        try:
                            proc.stdin.write(arr.tobytes())
                        except (BrokenPipeError, OSError):
                            write_failed = True
                            break
                        frames_written += 1

                        # Preview: save a downscaled JPEG periodically.
                        if (job.preview_path and
                                frames_written - last_preview_frame >= PREVIEW_EVERY):
                            try:
                                h, w = arr.shape[:2]
                                nh = max(1, int(h * 480 / max(1, w)))
                                small = cv2.resize(arr, (480, nh))
                                cv2.imwrite(job.preview_path, small,
                                            [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                            except Exception:
                                pass
                            last_preview_frame = frames_written

                        # Progress callback (cheap, every ~30 frames).
                        if job.progress_cb and frames_written % 30 == 0:
                            elapsed = max(1e-3, time.time() - t_start)
                            out_time_s = frames_written / fps if fps > 0 else 0.0
                            src_seconds_done = out_time_s
                            speed = src_seconds_done / elapsed
                            frac = (frames_written / nb_frames
                                    if nb_frames > 0 else 0.0)
                            remain_src = max(0.0,
                                             (nb_frames - frames_written) / fps) if fps > 0 else 0.0
                            eta = remain_src / speed if speed > 0 else 0.0
                            try:
                                job.progress_cb(
                                    max(0.0, min(1.0, frac)),
                                    out_time_s, speed, eta)
                            except Exception:
                                pass
                    if write_failed:
                        break
                if write_failed:
                    break

            if not job.cancel_event.is_set() and not write_failed:
                # Flush the filter graph for any tail frames buffered inside.
                try:
                    graph.push(None)
                    while True:
                        try:
                            out = graph.pull()
                        except (av.BlockingIOError, av.FFmpegError, av.EOFError):
                            break
                        arr = out.to_ndarray(format="bgr24")
                        try:
                            proc.stdin.write(arr.tobytes())
                            frames_written += 1
                        except (BrokenPipeError, OSError):
                            write_failed = True
                            break
                except Exception:
                    pass
        finally:
            try: container.close()
            except Exception: pass
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except Exception:
                pass

        # Wait for ffmpeg to finish muxing/flushing.
        if job.cancel_event.is_set():
            self._kill(proc)
        else:
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self._kill(proc)

        stderr_thread.join(timeout=2.0)
        ok = (proc.returncode == 0 and not job.cancel_event.is_set()
              and not write_failed)

        if not ok:
            if job.cancel_event.is_set():
                self.logger.info("Proxy encode canceled")
            else:
                self.logger.error(f"Proxy encode failed (rc={proc.returncode})")
                for line in stderr_lines[-80:]:
                    self.logger.error(f"  ffmpeg: {line}")
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
