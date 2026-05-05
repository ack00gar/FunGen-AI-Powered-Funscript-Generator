"""Proxy (edit-time transcode) builder.

Runs a single ffmpeg invocation that v360-unwarps VR sources to a flat
1080x1080 HEVC file (or downscales 2D sources to 1920x1080 with letterbox)
with AAC 160k audio. The proxy is written with a fixed filename suffix and
a sidecar JSON that tracks the source's mtime + frame count so stale
proxies can be detected.

Frame timing is preserved verbatim so funscript timestamps remain valid
when the editor swaps its active source to the proxy. If the post-encode
frame count drifts by more than 5 frames, the proxy is discarded.

Single-ffmpeg design: one ffmpeg process handles decode, v360/scale, and
encode end-to-end. libavfilter errors surface as stderr text rather than
in-process C exceptions, and ffmpeg's own progress reporter drives the UI.

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

from video.ffmpeg_helpers import find_ffmpeg, find_ffprobe, subprocess_flags

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

# Default FOV / pitch used when ProxyJob doesn't carry explicit values.
# Matches v0.8.0's VideoProcessor defaults (190 deg lenses, -21 deg pitch).
_DEFAULT_VR_FOV = 190
_DEFAULT_VR_PITCH = -21


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


def _test_encode_works(encoder: str, ffmpeg: str, logger: Optional[logging.Logger] = None) -> bool:
    """Run a 1-frame encode against /dev/null to confirm the encoder loads at
    runtime. Catches "compiled in but driver missing" cases like hevc_nvenc
    on AMD/Intel boxes (nvenc shows in -encoders but fails when nvcuda.dll
    is absent)."""
    try:
        rc = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "color=black:s=256x256:d=0.1:r=1",
             "-c:v", encoder, "-frames:v", "1", "-f", "null", "-"],
            capture_output=True, text=True, timeout=10,
            creationflags=subprocess_flags(),
        ).returncode
        if rc != 0 and logger:
            logger.debug(f"Encoder {encoder} failed runtime probe (rc={rc})")
        return rc == 0
    except (OSError, subprocess.TimeoutExpired) as e:
        if logger:
            logger.debug(f"Encoder {encoder} probe error: {e}")
        return False


def detect_hevc_encoder(logger: Optional[logging.Logger] = None) -> str:
    """Pick the best HEVC encoder that actually works on this host. Probes
    by test-encoding 1 frame; first candidate that succeeds wins. Cached
    per session. libx265 is the CPU fallback and assumed to always work."""
    global _ENCODER_PROBE_CACHE
    if _ENCODER_PROBE_CACHE is not None:
        return _ENCODER_PROBE_CACHE

    candidates_by_platform = {
        "Darwin": ["hevc_videotoolbox", "libx265"],
        "Windows": ["hevc_nvenc", "hevc_qsv", "hevc_amf", "libx265"],
        "Linux": ["hevc_nvenc", "hevc_vaapi", "hevc_qsv", "libx265"],
    }
    preferred = candidates_by_platform.get(platform.system(), ["libx265"])
    ffmpeg = find_ffmpeg()

    # First narrow to encoders compiled in (cheap), then runtime-probe each.
    try:
        out = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=5,
            creationflags=subprocess_flags(),
        ).stdout
    except (OSError, subprocess.TimeoutExpired) as e:
        if logger:
            logger.warning(f"ffmpeg -encoders probe failed: {e}; defaulting to libx265")
        _ENCODER_PROBE_CACHE = "libx265"
        return _ENCODER_PROBE_CACHE

    available = {line.strip().split()[1] for line in out.splitlines()
                 if line.strip().startswith("V") and len(line.strip().split()) > 1}
    for enc in preferred:
        if enc not in available:
            continue
        if enc == "libx265" or _test_encode_works(enc, ffmpeg, logger):
            _ENCODER_PROBE_CACHE = enc
            if logger:
                logger.info(f"Proxy encoder selected: {enc}")
            return enc
        if logger:
            logger.info(f"Encoder {enc} compiled in but failed runtime probe; trying next")
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
    # Source dimensions. Required for VR so the crop filter knows how to
    # split SBS/TB panels. 0 triggers a fallback ffprobe at encode time.
    source_width: int = 0
    source_height: int = 0
    # VR dewarp params mirrored from VideoProcessor. 0 means "use default".
    vr_fov: int = 0
    vr_pitch: float = 0.0
    target_path: str = ""
    # Optional callbacks: (fraction[0..1], out_time_s, speed_x, eta_s)
    progress_cb: Optional[Callable[[float, float, float, float], None]] = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    # Live-preview JPEG: if set, ffmpeg writes a downscaled JPEG here
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

    def _video_filter_str(self, job: ProxyJob) -> str:
        """The ffmpeg -vf chain: v360 dewarp for VR, scale+pad for 2D.

        Mirrors v0.8.0's ``_get_vr_video_filters`` (video/_vp_ffmpeg_builders.py)
        verbatim so the proxy's baked-in dewarp looks identical to the
        runtime preview the user sees when scrubbing the raw source:
        crop one stereo panel, then v360 with in_stereo=0 output=sg.
        """
        out_w, out_h = _target_dims_for(job)
        fmt = (job.vr_input_format or "").lower()
        is_vr = bool(fmt) and fmt != "2d"

        if not is_vr:
            # 2D: scale-and-pad to the target frame.
            return (f"scale={out_w}:{out_h}"
                    f":force_original_aspect_ratio=decrease,"
                    f"pad={out_w}:{out_h}"
                    f":(ow-iw)/2:(oh-ih)/2:black")

        parts = []
        ow = int(job.source_width or 0)
        oh = int(job.source_height or 0)
        # Pre-crop to one stereo panel. Matches v0.8.0's logic exactly:
        # SBS / LR -> left half, RL -> right half, TB -> top half.
        if ow > 0 and oh > 0:
            if "_sbs" in fmt or "_lr" in fmt:
                parts.append(f"crop={ow // 2}:{oh}:0:0")
            elif "_rl" in fmt:
                parts.append(f"crop={ow // 2}:{oh}:{ow // 2}:0")
            elif "_tb" in fmt:
                parts.append(f"crop={ow}:{oh // 2}:0:0")

        base_fmt = (fmt.replace("_sbs", "")
                       .replace("_tb", "")
                       .replace("_lr", "")
                       .replace("_rl", "")) or "he"
        vr_fov = job.vr_fov if job.vr_fov and job.vr_fov > 0 else _DEFAULT_VR_FOV
        vr_pitch = job.vr_pitch if job.vr_pitch else _DEFAULT_VR_PITCH
        parts.append(
            f"v360={base_fmt}:in_stereo=0:output=sg:"
            f"iv_fov={vr_fov}:ih_fov={vr_fov}:"
            f"d_fov={vr_fov}:"
            f"v_fov=90:h_fov=90:"
            f"pitch={vr_pitch}:yaw=0:roll=0:"
            f"w={out_w}:h={out_h}:interp=linear"
        )
        return ",".join(parts)

    def _build_ffmpeg_command(self, job: ProxyJob, encoder: str,
                              partial_path: str) -> list:
        """Single-process ffmpeg command: decode + v360/scale + encode.

        Progress lines are emitted on stdout via ``-progress pipe:1``.
        The optional preview JPEG is a second output driven by a filter_complex
        split so no extra processes are needed.
        """
        # Near-all-I-frame: GOP=2 with B-frames disabled gives an I-P-I-P
        # pattern. Worst-case backward scrub is ~1 extra frame decode, and
        # the config is accepted by every encoder build we care about. GOP=1
        # is rejected by some nvenc releases ("GOP Length should be > B + 1").
        iframe_args = ["-g", "2", "-keyint_min", "2", "-bf", "0"]
        if encoder == "libx265":
            iframe_args += ["-x265-params",
                            "keyint=2:min-keyint=2:no-open-gop=1:bframes=0"]
        elif encoder == "hevc_nvenc":
            iframe_args += ["-no-scenecut", "1"]
        elif encoder == "hevc_qsv":
            iframe_args += ["-look_ahead", "0"]

        vf = self._video_filter_str(job)

        # v360 in libavfilter is single-threaded by default; bump filter pools
        # so VR dewarp doesn't bottleneck on one core during proxy build.
        import os as _os
        _ft = max(1, min(8, (_os.cpu_count() or 4) // 2))
        cmd = [
            find_ffmpeg(),
            "-hide_banner", "-nostats", "-loglevel", "warning",
            "-filter_threads", str(_ft),
            "-filter_complex_threads", str(_ft),
            "-y",
            "-i", job.source_path,
            "-progress", "pipe:1",
        ]

        if job.preview_path:
            # filter_complex: split the filtered video into the main encode
            # output plus a low-fps downscaled JPEG written to a side file.
            # The preview file is overwritten in place (-update 1) so the UI
            # just polls its mtime.
            filter_complex = (
                f"[0:v]{vf},split=2[main][prev];"
                f"[prev]fps=1/7,scale=480:-2[preview]"
            )
            cmd += [
                "-filter_complex", filter_complex,
                "-map", "[main]",
                "-c:v", encoder,
                "-b:v", VIDEO_BITRATE,
                "-pix_fmt", "yuv420p",
                *iframe_args,
                "-map", "0:a:0?",
                "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-ac", "2",
                "-movflags", "+faststart",
                "-f", "mp4",
                partial_path,
                "-map", "[preview]",
                "-q:v", "3", "-update", "1",
                "-f", "image2",
                job.preview_path,
            ]
        else:
            cmd += [
                "-vf", vf,
                "-c:v", encoder,
                "-b:v", VIDEO_BITRATE,
                "-pix_fmt", "yuv420p",
                *iframe_args,
                "-map", "0:v:0",
                "-map", "0:a:0?",
                "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-ac", "2",
                "-movflags", "+faststart",
                "-f", "mp4",
                partial_path,
            ]
        return cmd

    def encode(self, job: ProxyJob) -> bool:
        """Run ffmpeg end-to-end: single process does decode, v360/scale,
        encode, and audio re-encode. Progress arrives via ``-progress pipe:1``
        on stdout; stderr is drained to a log buffer for error reporting.
        """
        # Fallback: if the caller didn't supply source dims (older integration
        # path), probe now. Only needed for VR so the crop knows panel sizes.
        if (job.vr_input_format and job.vr_input_format != "2d"
                and (not job.source_width or not job.source_height)):
            from video.frame_source.probe import probe as _probe
            p = _probe(job.source_path)
            if p is not None and p.width > 0 and p.height > 0:
                job.source_width = p.width
                job.source_height = p.height

        encoder = detect_hevc_encoder(self.logger)

        partial_path = job.target_path + ".partial"
        if os.path.exists(partial_path):
            try: os.remove(partial_path)
            except OSError: pass

        cmd = self._build_ffmpeg_command(job, encoder, partial_path)
        self.logger.info(f"Proxy encode start: {os.path.basename(job.source_path)} "
                         f"-> {os.path.basename(job.target_path)} ({encoder})")
        self.logger.info("ffmpeg cmd: " + " ".join(cmd))

        nb_frames = int(job.source_nb_frames or 0)
        fps = float(job.source_fps or 30.0)
        if fps <= 0:
            fps = 30.0

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                creationflags=subprocess_flags(),
            )
        except FileNotFoundError:
            self.logger.error("ffmpeg executable not found on PATH")
            return False

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

        # Progress reader. ffmpeg emits a block every ~0.5 s terminated by
        # `progress=continue` (or `progress=end` at the tail). Fields are
        # one per line. We only read the ones we need; the rest are ignored.
        t_start = time.time()
        last_frame = 0
        last_out_time_s = 0.0
        canceled = False
        try:
            if proc.stdout is None:
                raise IOError("ffmpeg progress pipe is None")
            for raw in iter(proc.stdout.readline, b""):
                if job.cancel_event.is_set():
                    canceled = True
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                key, _, val = line.partition("=")
                if key == "frame":
                    try: last_frame = int(val)
                    except ValueError: pass
                elif key == "out_time_ms":
                    try: last_out_time_s = int(val) / 1_000_000.0
                    except ValueError: pass
                elif key == "progress":
                    # One block complete, fire callback.
                    if job.progress_cb:
                        elapsed = max(1e-3, time.time() - t_start)
                        speed = last_out_time_s / elapsed
                        frac = (last_frame / nb_frames
                                if nb_frames > 0 else 0.0)
                        remain_src = max(0.0,
                                         (nb_frames - last_frame) / fps) if fps > 0 and nb_frames > 0 else 0.0
                        eta = remain_src / speed if speed > 0 else 0.0
                        try:
                            job.progress_cb(
                                max(0.0, min(1.0, frac)),
                                last_out_time_s, speed, eta)
                        except Exception:
                            pass
                    if val == "end":
                        break
        except Exception as e:
            self.logger.debug(f"Progress reader ended: {e}")

        if canceled or job.cancel_event.is_set():
            self._kill(proc)
        else:
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self._kill(proc)

        stderr_thread.join(timeout=2.0)
        for pipe in (proc.stdout, proc.stderr):
            if pipe is not None:
                try: pipe.close()
                except Exception: pass
        ok = (proc.returncode == 0
              and not job.cancel_event.is_set()
              and not canceled)

        if not ok:
            if job.cancel_event.is_set() or canceled:
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
                [find_ffprobe(), "-v", "error",
                 "-select_streams", "v:0",
                 "-count_packets",
                 "-show_entries", "stream=nb_read_packets",
                 "-of", "default=nokey=1:noprint_wrappers=1",
                 path],
                capture_output=True, text=True, timeout=60,
                creationflags=subprocess_flags(),
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
