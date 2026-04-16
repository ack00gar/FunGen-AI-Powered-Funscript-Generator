"""
PyAV-based thumbnail extractor for frame-accurate random access.

Replaces the former subprocess-ffmpeg implementation with a persistent
libav container + filter graph so each ``get_frame`` call is an in-process
seek + decode rather than a fresh ffmpeg spawn (~5-15s cold start on heavy
formats like 8K 10-bit VR). Same output byte format (BGR24 ndarray) and
same filter chain (crop + v360 for VR, scale+pad for 2D), so downstream
consumers (timeline tooltips, scrub previews, subprocess-backend fallback)
are unaffected.
"""

import logging
from threading import Lock
from typing import Optional, Tuple

import av
import av.filter
import numpy as np


class ThumbnailExtractor:
    """
    Frame-accurate thumbnail extractor using a persistent PyAV decoder.

    Video properties (fps, dimensions, etc.) are passed from the video
    processor at init time — no redundant probe calls.
    """

    def __init__(self, video_path: str, fps: float, total_frames: int,
                 output_size: int = 320, vr_input_format: Optional[str] = None,
                 vr_fov: int = 190, vr_pitch: float = 0.0,
                 display_dimensions: Optional[Tuple[int, int]] = None,
                 logger: Optional[logging.Logger] = None):
        self.video_path = video_path
        self.logger = logger or logging.getLogger(__name__)
        self.output_size = output_size
        self.fps = fps
        self.total_frames = total_frames
        self.vr_input_format = vr_input_format
        self.vr_fov = vr_fov
        self.vr_pitch = vr_pitch

        # HD display dimensions (width, height) — when set, 2D output is non-square
        self._display_w, self._display_h = (
            display_dimensions if display_dimensions else (output_size, output_size)
        )

        self.lock = Lock()

        vr_fmt = (vr_input_format or '').lower()
        self.is_sbs_left = '_sbs' in vr_fmt or '_lr' in vr_fmt
        self.is_sbs_right = '_rl' in vr_fmt
        self.is_tb = '_tb' in vr_fmt
        self.is_vr = vr_input_format is not None

        # Persistent decode state
        self._container: Optional[av.container.InputContainer] = None
        self._stream = None
        self._time_base = None
        self._graph: Optional[av.filter.Graph] = None
        self.is_open = False

        if fps <= 0 or total_frames <= 0:
            return

        try:
            self._open_container()
        except Exception as e:
            self.logger.warning(f"ThumbnailExtractor: open failed: {e}")
            self._close_container()
            return

        if self.is_open:
            self.logger.debug(
                f"ThumbnailExtractor (PyAV): {fps:.2f} FPS, "
                f"{total_frames} frames, output={self._display_w}x{self._display_h}"
            )

    # ------------------------------------------------------------------ open

    def _open_container(self) -> None:
        self._container = av.open(self.video_path)
        try:
            self._stream = self._container.streams.video[0]
        except (IndexError, AttributeError):
            raise RuntimeError("no video stream")
        self._stream.thread_type = "AUTO"
        self._time_base = self._stream.time_base
        self._graph = self._build_filter_graph()
        self.is_open = True

    def _build_filter_graph(self) -> av.filter.Graph:
        g = av.filter.Graph()
        src = g.add_buffer(template=self._stream)
        prev = src

        for spec in self._filter_specs():
            name, _, args = spec.partition("=")
            node = g.add(name.strip(), args.strip())
            prev.link_to(node)
            prev = node

        fmt = g.add("format", "bgr24")
        prev.link_to(fmt)
        sink = g.add("buffersink")
        fmt.link_to(sink)
        g.configure()
        return g

    def _filter_specs(self) -> list:
        specs = []
        if self.is_vr:
            ow = self._stream.width
            oh = self._stream.height
            if self.is_sbs_left:
                specs.append(f"crop={ow // 2}:{oh}:0:0")
            elif self.is_sbs_right:
                specs.append(f"crop={ow // 2}:{oh}:{ow // 2}:0")
            elif self.is_tb:
                specs.append(f"crop={ow}:{oh // 2}:0:0")
            base_fmt = (self.vr_input_format or '').replace(
                '_sbs', '').replace('_tb', '').replace('_lr', '').replace('_rl', '')
            # Guard against vr_fov=0 leaking in. libavfilter rejects v360 with
            # iv_fov/ih_fov/d_fov all zero. See video_processor._build_pyav_filter_chain.
            vr_fov = self.vr_fov if self.vr_fov and self.vr_fov > 0 else 190
            specs.append(
                f"v360={base_fmt}:in_stereo=0:output=sg:"
                f"iv_fov={vr_fov}:ih_fov={vr_fov}:"
                f"d_fov={vr_fov}:"
                f"v_fov=90:h_fov=90:"
                f"pitch={self.vr_pitch}:yaw=0:roll=0:"
                f"w={self.output_size}:h={self.output_size}:interp=linear"
            )
        else:
            if self._display_w != self.output_size or self._display_h != self.output_size:
                specs.append(f"scale={self._display_w}:{self._display_h}")
            else:
                specs.append(
                    f"scale={self.output_size}:{self.output_size}"
                    f":force_original_aspect_ratio=decrease"
                )
                specs.append(
                    f"pad={self.output_size}:{self.output_size}"
                    f":(ow-iw)/2:(oh-ih)/2:black"
                )
        return specs

    # --------------------------------------------------------------- get_frame

    def get_frame(self, frame_index: int, **_kwargs) -> Optional[np.ndarray]:
        """
        Extract a single frame at the specified index using the persistent
        PyAV decoder. Returns a BGR24 ndarray (display_h x display_w x 3) or None.
        """
        if not self.is_open:
            return None
        if frame_index < 0 or frame_index >= self.total_frames:
            return None

        with self.lock:
            if self._container is None or self._stream is None or self._graph is None:
                return None
            try:
                target_pts = int(frame_index / self.fps / self._time_base)
                self._container.seek(
                    target_pts, backward=True, any_frame=False,
                    stream=self._stream,
                )
            except Exception as e:
                self.logger.debug(f"ThumbnailExtractor seek({frame_index}) failed: {e}")
                return None

            # Decode until a filtered frame is ready. We don't drain to exact
            # target — the thumbnail use case accepts the GOP-keyframe-near-target
            # approximation; the subprocess predecessor did the same by default.
            try:
                for packet in self._container.demux(self._stream):
                    for frame in packet.decode():
                        self._graph.push(frame)
                        while True:
                            try:
                                out_frame = self._graph.pull()
                            except (av.BlockingIOError, av.FFmpegError):
                                break
                            arr = out_frame.to_ndarray(format="bgr24")
                            expected = (self._display_h, self._display_w, 3)
                            if arr.shape != expected:
                                self.logger.debug(
                                    f"ThumbnailExtractor shape mismatch {arr.shape} vs {expected}")
                                return None
                            return arr
            except Exception as e:
                self.logger.debug(
                    f"ThumbnailExtractor decode at frame {frame_index} failed: {e}")
                return None
        return None

    # ---------------------------------------------------------------- close

    def close(self):
        with self.lock:
            self._close_container()

    def _close_container(self):
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                pass
        self._container = None
        self._stream = None
        self._graph = None
        self.is_open = False

    def __del__(self):
        try: self.close()
        except Exception: pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
