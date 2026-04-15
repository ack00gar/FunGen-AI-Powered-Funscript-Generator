"""
Audio playback engine for FunGen.

Decodes audio from the video file using PyAV (libav* in-process — no
subprocess), feeds raw PCM to the OS audio system via sounddevice.

The video pipeline runs independently; AudioVideoSync (observer of
VideoProcessor) drives start/stop/seek on this player.

Tempo for slow-motion is handled by libavfilter's ``atempo`` filter chain
(same algorithm and same 0.5-2.0 per-instance bounds as ffmpeg's CLI).
The output sample rate matches the OS device's native rate so
``RawOutputStream`` doesn't have to resample (silence on macOS CoreAudio if
mismatched).
"""

import logging
import sys
import threading
import time
from typing import Optional

import av
import av.audio.resampler
import numpy as np

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError):
    SOUNDDEVICE_AVAILABLE = False

logger = logging.getLogger(__name__)

# Audio format constants
CHANNELS = 2
SAMPLE_WIDTH = 2  # 16-bit = 2 bytes
FRAME_BYTES = CHANNELS * SAMPLE_WIDTH  # bytes per audio frame
BLOCK_SIZE = 1024  # frames per sounddevice callback
BUFFER_MAX_BYTES = 2 * 1024 * 1024  # 2 MB max buffer before trimming


def _get_device_sample_rate() -> int:
    """Query the default output device's native sample rate."""
    if not SOUNDDEVICE_AVAILABLE:
        return 48000
    try:
        info = sd.query_devices(kind='output')
        return int(info['default_samplerate'])
    except Exception:
        return 48000


class AudioPlayer:
    """Plays audio from a video file via PyAV + sounddevice."""

    def __init__(self):
        self._video_path: Optional[str] = None
        self._has_audio: bool = False
        self._video_fps: float = 30.0
        self._sample_rate: int = _get_device_sample_rate()

        # Decode state (PyAV)
        self._container: Optional[av.container.InputContainer] = None
        self._audio_stream = None
        self._filter_graph: Optional[av.filter.Graph] = None
        self._resampler: Optional[av.audio.resampler.AudioResampler] = None

        # Sounddevice playback state
        self._stream: "sd.RawOutputStream | None" = None
        self._reader_thread: Optional[threading.Thread] = None

        self._buffer = bytearray()
        self._buf_lock = threading.Lock()
        self._buf_read_pos = 0

        self._is_paused = False
        self._is_stopped = True

        # Volume / mute
        self._volume: float = 1.0
        self._muted: bool = False

        # Scrub support
        self._scrub_timer: Optional[threading.Timer] = None

        self._lock = threading.Lock()  # guards start/stop/seek

        logger.debug(f"AudioPlayer init: device sample_rate={self._sample_rate}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_video(self, path: str, has_audio: bool, fps: float):
        """Called when a new video is opened."""
        self.stop()
        self._video_path = path
        self._has_audio = has_audio
        self._video_fps = fps if fps > 0 else 30.0

    def start(self, position_ms: float, tempo: float = 1.0):
        """Start (or restart) audio playback from *position_ms*."""
        if not SOUNDDEVICE_AVAILABLE or not self._has_audio or not self._video_path:
            return

        with self._lock:
            self._stop_internal()
            self._start_internal(position_ms, tempo)

    def pause(self):
        self._is_paused = True

    def resume(self):
        self._is_paused = False

    def stop(self):
        with self._lock:
            self._stop_internal()

    def seek(self, position_ms: float, tempo: float = 1.0):
        """Seek to a new position (stop + start)."""
        self.start(position_ms, tempo)

    def scrub(self, position_ms: float, duration_ms: float = 100):
        """Play a short audio burst for frame-stepping scrub."""
        if not SOUNDDEVICE_AVAILABLE or not self._has_audio or not self._video_path:
            return

        with self._lock:
            self._cancel_scrub_timer()
            self._stop_internal()
            self._start_internal(position_ms, tempo=1.0)

            t = threading.Timer(duration_ms / 1000.0, self._scrub_timeout)
            t.daemon = True
            t.start()
            self._scrub_timer = t

    def set_volume(self, vol: float):
        self._volume = max(0.0, min(1.0, vol))

    def set_mute(self, muted: bool):
        self._muted = muted

    def cleanup(self):
        """Full teardown for app shutdown."""
        self.stop()
        self._video_path = None

    @property
    def has_audio(self) -> bool:
        return self._has_audio

    # ------------------------------------------------------------------
    # Internals — decode pipeline (PyAV)
    # ------------------------------------------------------------------

    def _start_internal(self, position_ms: float, tempo: float):
        """Open container, build filter graph + resampler, start reader and
        sounddevice stream (caller must hold self._lock)."""
        start_sec = max(0.0, position_ms / 1000.0)
        sr = self._sample_rate

        try:
            self._container = av.open(self._video_path)
        except Exception as e:
            logger.error(f"Failed to open audio container: {e}")
            self._container = None
            return

        try:
            self._audio_stream = self._container.streams.audio[0]
        except (IndexError, AttributeError):
            logger.warning("No audio stream in video; aborting audio playback")
            self._close_pyav()
            return

        self._audio_stream.thread_type = "AUTO"

        # Optional atempo filter chain for slow-mo / fast-forward.
        atempo_args = self._build_atempo_chain(tempo)
        if atempo_args:
            try:
                self._filter_graph = self._build_filter_graph(self._audio_stream, atempo_args)
            except Exception as e:
                logger.warning(f"Audio atempo graph failed ({e}); playing at 1.0x")
                self._filter_graph = None
        else:
            self._filter_graph = None

        # Resampler converts whatever the codec produces to interleaved s16
        # stereo at the device's native rate.
        try:
            self._resampler = av.audio.resampler.AudioResampler(
                format='s16', layout='stereo', rate=sr,
            )
        except Exception as e:
            logger.error(f"AudioResampler init failed: {e}")
            self._close_pyav()
            return

        # Seek to the requested position. Audio seeks are cheap.
        if start_sec > 0.001:
            try:
                target_pts = int(start_sec / self._audio_stream.time_base)
                self._container.seek(
                    target_pts, backward=True, any_frame=False,
                    stream=self._audio_stream,
                )
            except Exception as e:
                logger.warning(f"Audio seek to {start_sec:.3f}s failed: {e}")

        # Reset playback state
        self._buffer = bytearray()
        self._buf_read_pos = 0
        self._is_paused = False
        self._is_stopped = False

        self._reader_thread = threading.Thread(
            target=self._reader_func, daemon=True, name="AudioReader",
        )
        self._reader_thread.start()

        # Pre-buffer: wait for enough decoded data before opening the output
        # stream. Track wall time so we can compensate for the delay.
        prebuffer_start = time.monotonic()
        deadline = prebuffer_start + 0.5
        min_bytes = BLOCK_SIZE * FRAME_BYTES * 4
        while time.monotonic() < deadline:
            with self._buf_lock:
                if len(self._buffer) >= min_bytes:
                    break
            time.sleep(0.01)

        # Skip ahead in the buffer by how long pre-buffer took, so audio
        # rejoins the now-current video position.
        elapsed_ms = (time.monotonic() - prebuffer_start) * 1000.0
        if elapsed_ms > 10.0:
            skip_bytes = int(elapsed_ms * sr * FRAME_BYTES / 1000.0)
            skip_bytes -= skip_bytes % FRAME_BYTES
            with self._buf_lock:
                max_skip = max(0, len(self._buffer) - BLOCK_SIZE * FRAME_BYTES)
                self._buf_read_pos = min(skip_bytes, max_skip)

        try:
            self._stream = sd.RawOutputStream(
                samplerate=sr,
                blocksize=BLOCK_SIZE,
                channels=CHANNELS,
                dtype='int16',
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception as e:
            logger.error(f"Failed to open audio stream: {e}")
            self._close_pyav()
            self._stream = None

    def _stop_internal(self):
        """Stop everything (caller must hold self._lock)."""
        self._is_stopped = True
        self._cancel_scrub_timer()

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        self._close_pyav()

        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
            self._reader_thread = None

        self._buffer = bytearray()
        self._buf_read_pos = 0

    def _close_pyav(self):
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                pass
            self._container = None
        self._audio_stream = None
        self._filter_graph = None
        self._resampler = None

    def _build_filter_graph(self, audio_stream, atempo_chain: str) -> av.filter.Graph:
        g = av.filter.Graph()
        src = g.add_abuffer(template=audio_stream)
        prev = src
        for spec in atempo_chain.split(","):
            spec = spec.strip()
            if not spec: continue
            name, _, args = spec.partition("=")
            node = g.add(name.strip(), args.strip())
            prev.link_to(node)
            prev = node
        sink = g.add("abuffersink")
        prev.link_to(sink)
        g.configure()
        return g

    def _emit_pcm(self, frame, pcm_acc: bytearray) -> None:
        """Resample one decoded/filtered frame to s16 stereo @ device rate
        and append the PCM bytes to ``pcm_acc``."""
        if self._resampler is None:
            return
        for resampled in self._resampler.resample(frame):
            arr = resampled.to_ndarray()
            # AudioResampler with format=s16, layout=stereo yields shape
            # (1, N*2) in interleaved form. tobytes() gives ready-to-play PCM.
            pcm_acc.extend(arr.tobytes())

    def _reader_func(self):
        """Background thread: demux + decode + filter + resample, feeding the
        ring buffer the audio_callback drains.

        Applies backpressure when the unread buffer exceeds ~1s of audio so we
        don't burn CPU decoding far ahead of playback — important when the
        video decoder (in the same process now that we're PyAV-in-process)
        is already saturating cores on heavy formats like 8K HEVC.
        """
        if self._container is None or self._audio_stream is None:
            return
        graph = self._filter_graph
        # Backpressure threshold: ~1 second of audio buffered ahead is plenty
        # for jitter absorption; beyond that, sleep until the consumer drains.
        target_buffered = self._sample_rate * FRAME_BYTES
        try:
            for packet in self._container.demux(self._audio_stream):
                if self._is_stopped:
                    break
                # Wait if we're already running well ahead of playback.
                while not self._is_stopped:
                    with self._buf_lock:
                        unread = len(self._buffer) - self._buf_read_pos
                    if unread < target_buffered:
                        break
                    time.sleep(0.02)
                if self._is_stopped:
                    break
                pcm_acc = bytearray()
                for frame in packet.decode():
                    if graph is not None:
                        graph.push(frame)
                        while True:
                            try:
                                out_frame = graph.pull()
                            except (av.BlockingIOError, av.FFmpegError):
                                break
                            self._emit_pcm(out_frame, pcm_acc)
                    else:
                        self._emit_pcm(frame, pcm_acc)
                if not pcm_acc:
                    continue
                with self._buf_lock:
                    self._buffer.extend(pcm_acc)
                    if self._buf_read_pos > BUFFER_MAX_BYTES:
                        del self._buffer[:self._buf_read_pos]
                        self._buf_read_pos = 0
        except Exception as e:
            logger.debug(f"Audio reader stopped: {e}")

    # ------------------------------------------------------------------
    # Sounddevice callback
    # ------------------------------------------------------------------

    def _audio_callback(self, outdata, frames, time_info, status):
        """Sounddevice callback — fills output buffer from our ring buffer."""
        needed = frames * FRAME_BYTES

        if self._is_paused or self._muted or self._is_stopped:
            outdata[:] = b'\x00' * needed
            return

        with self._buf_lock:
            available = len(self._buffer) - self._buf_read_pos
            if available >= needed:
                chunk = bytes(self._buffer[self._buf_read_pos:self._buf_read_pos + needed])
                self._buf_read_pos += needed
            else:
                chunk = bytes(self._buffer[self._buf_read_pos:self._buf_read_pos + available])
                chunk += b'\x00' * (needed - available)
                self._buf_read_pos += available

        if self._volume < 0.99:
            samples = np.frombuffer(chunk, dtype=np.int16).copy()
            samples = (samples * self._volume).astype(np.int16)
            outdata[:] = samples.tobytes()
        else:
            outdata[:] = chunk

    def _scrub_timeout(self):
        with self._lock:
            self._stop_internal()

    def _cancel_scrub_timer(self):
        if self._scrub_timer is not None:
            self._scrub_timer.cancel()
            self._scrub_timer = None

    @staticmethod
    def _build_atempo_chain(tempo: float) -> Optional[str]:
        """Build an atempo filter chain string for libavfilter. Each atempo
        instance is bounded to 0.5-2.0; values outside that range are chained.
        Returns None when tempo is ~1.0 (no filter needed)."""
        if tempo <= 0 or abs(tempo - 1.0) < 0.01:
            return None

        filters = []
        remaining = tempo

        if remaining < 0.5:
            while remaining < 0.5:
                filters.append('atempo=0.5')
                remaining /= 0.5
            if abs(remaining - 1.0) > 0.01:
                filters.append(f'atempo={remaining:.4f}')
        elif remaining > 2.0:
            while remaining > 2.0:
                filters.append('atempo=2.0')
                remaining /= 2.0
            if abs(remaining - 1.0) > 0.01:
                filters.append(f'atempo={remaining:.4f}')
        else:
            filters.append(f'atempo={remaining:.4f}')

        return ','.join(filters) if filters else None
