"""
Audio playback engine for FunGen.

Uses a separate FFmpeg subprocess to decode audio to raw PCM,
fed to the OS audio system via sounddevice.RawOutputStream.

The video FFmpeg pipe is never modified — audio runs independently
and is synced via the AudioVideoSync observer.

IMPORTANT: We use the output device's native sample rate for both
FFmpeg decoding and the sounddevice stream. RawOutputStream with int16
does not resample on macOS CoreAudio, so a mismatch = silence.
"""

import subprocess
import sys
import threading
import logging
import time

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
    """Plays audio from a video file via FFmpeg + sounddevice."""

    def __init__(self):
        self._video_path: str | None = None
        self._has_audio: bool = False
        self._video_fps: float = 30.0
        self._sample_rate: int = _get_device_sample_rate()

        # Playback state
        self._ffmpeg_proc: subprocess.Popen | None = None
        self._stream: "sd.RawOutputStream | None" = None
        self._reader_thread: threading.Thread | None = None

        self._buffer = bytearray()
        self._buf_lock = threading.Lock()
        self._buf_read_pos = 0

        self._is_paused = False
        self._is_stopped = True

        # Volume / mute
        self._volume: float = 1.0
        self._muted: bool = False

        # Scrub support
        self._scrub_timer: threading.Timer | None = None

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
    # Internals
    # ------------------------------------------------------------------

    def _start_internal(self, position_ms: float, tempo: float):
        """Spawn FFmpeg + sounddevice (caller must hold self._lock)."""
        start_sec = max(0.0, position_ms / 1000.0)
        sr = self._sample_rate

        cmd = ['ffmpeg', '-hide_banner', '-nostats', '-loglevel', 'error']
        if start_sec > 0.001:
            cmd.extend(['-ss', f'{start_sec:.3f}'])
        cmd.extend(['-i', self._video_path])
        cmd.extend(['-vn', '-sn'])

        # Tempo filter for slow-motion
        af = self._build_atempo_filter(tempo)
        if af:
            cmd.extend(['-af', af])

        cmd.extend(['-ac', str(CHANNELS), '-ar', str(sr),
                     '-c:a', 'pcm_s16le', '-f', 's16le', 'pipe:1'])

        try:
            creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            self._ffmpeg_proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                bufsize=-1, creationflags=creation_flags,
            )
        except Exception as e:
            logger.error(f"Failed to start audio FFmpeg: {e}")
            self._ffmpeg_proc = None
            return

        # Reset buffer
        self._buffer = bytearray()
        self._buf_read_pos = 0
        self._is_paused = False
        self._is_stopped = False

        # Reader thread
        self._reader_thread = threading.Thread(
            target=self._reader_func, daemon=True, name="AudioReader"
        )
        self._reader_thread.start()

        # Pre-buffer: wait for enough data before starting the output stream.
        # Track wall-clock time so we can compensate for the delay below.
        prebuffer_start = time.monotonic()
        deadline = prebuffer_start + 0.5
        min_bytes = BLOCK_SIZE * FRAME_BYTES * 4
        while time.monotonic() < deadline:
            with self._buf_lock:
                if len(self._buffer) >= min_bytes:
                    break
            time.sleep(0.01)

        # Compensate for pre-buffer delay: video has been advancing while we
        # waited, so skip the equivalent amount of audio from the buffer so
        # that playback begins at the current video position, not the position
        # that was passed to start().
        elapsed_ms = (time.monotonic() - prebuffer_start) * 1000.0
        if elapsed_ms > 10.0:
            skip_bytes = int(elapsed_ms * sr * FRAME_BYTES / 1000.0)
            skip_bytes -= skip_bytes % FRAME_BYTES  # align to audio-frame boundary
            with self._buf_lock:
                # Keep at least one block available so the callback doesn't underrun
                max_skip = max(0, len(self._buffer) - BLOCK_SIZE * FRAME_BYTES)
                self._buf_read_pos = min(skip_bytes, max_skip)

        # Open sounddevice output at device native sample rate
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
            self._kill_ffmpeg()
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

        self._kill_ffmpeg()

        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
            self._reader_thread = None

        self._buffer = bytearray()
        self._buf_read_pos = 0

    def _kill_ffmpeg(self):
        proc = self._ffmpeg_proc
        if proc is not None:
            try:
                proc.kill()
                proc.wait(timeout=2.0)
            except Exception:
                pass
            self._ffmpeg_proc = None

    def _reader_func(self):
        """Background thread: read FFmpeg stdout into buffer."""
        proc = self._ffmpeg_proc
        if proc is None or proc.stdout is None:
            return

        sr = self._sample_rate
        chunk_size = sr * FRAME_BYTES // 4  # ~250ms of audio per read

        try:
            while not self._is_stopped:
                data = proc.stdout.read(chunk_size)
                if not data:
                    break
                with self._buf_lock:
                    self._buffer.extend(data)
                    # Trim consumed data to bound memory usage
                    if self._buf_read_pos > BUFFER_MAX_BYTES:
                        del self._buffer[:self._buf_read_pos]
                        self._buf_read_pos = 0
        except Exception:
            pass

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
                # Underrun — pad with silence
                chunk = bytes(self._buffer[self._buf_read_pos:self._buf_read_pos + available])
                chunk += b'\x00' * (needed - available)
                self._buf_read_pos += available

        # Apply volume scaling (app-level volume within OS mixer)
        if self._volume < 0.99:
            samples = np.frombuffer(chunk, dtype=np.int16).copy()
            samples = (samples * self._volume).astype(np.int16)
            outdata[:] = samples.tobytes()
        else:
            outdata[:] = chunk

    def _scrub_timeout(self):
        """Called by scrub timer to auto-stop."""
        with self._lock:
            self._stop_internal()

    def _cancel_scrub_timer(self):
        if self._scrub_timer is not None:
            self._scrub_timer.cancel()
            self._scrub_timer = None

    @staticmethod
    def _build_atempo_filter(tempo: float) -> str | None:
        """Build FFmpeg atempo filter chain for the given tempo factor.

        FFmpeg atempo supports 0.5-2.0 per instance, so values outside
        that range are chained (e.g. 0.333x -> atempo=0.5,atempo=0.667).
        Returns None if tempo is ~1.0 (no filter needed).
        """
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
