"""
AudioVideoSync — observer that keeps AudioPlayer in sync with VideoProcessor.

Registers with VideoProcessor's playback-state and seek callbacks
(the same observer pattern used by device_control and streamer).

IMPORTANT: The playback-state callback fires on EVERY decoded frame,
not just on transitions. We track state locally so that audio is only
started/stopped on actual transitions.
"""

import logging

from config.constants import ProcessingSpeedMode
from video.audio_player import AudioPlayer, SOUNDDEVICE_AVAILABLE

logger = logging.getLogger(__name__)


class AudioVideoSync:
    """Bridges VideoProcessor events to AudioPlayer."""

    def __init__(self, video_processor, audio_player: AudioPlayer, app_instance):
        self._vp = video_processor
        self._ap = audio_player
        self._app = app_instance
        self._registered = False

        # State tracking to detect transitions (callback fires every frame)
        self._audio_running = False
        self._current_tempo = 1.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Register callbacks with VideoProcessor."""
        if not SOUNDDEVICE_AVAILABLE or self._registered:
            return
        self._vp.register_playback_state_callback(self._on_playback_state_change)
        self._vp.register_seek_callback(self._on_video_seek)
        self._registered = True
        logger.info("AudioVideoSync started")

    def stop(self):
        """Unregister callbacks and stop audio."""
        if self._registered:
            self._vp.unregister_playback_state_callback(self._on_playback_state_change)
            self._vp.unregister_seek_callback(self._on_video_seek)
            self._registered = False
        self._ap.stop()
        self._audio_running = False
        logger.info("AudioVideoSync stopped")

    # ------------------------------------------------------------------
    # Settings bridge
    # ------------------------------------------------------------------

    def update_settings(self, volume: float, muted: bool):
        """Forward UI volume/mute changes to the audio player."""
        self._ap.set_volume(volume)
        self._ap.set_mute(muted)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _is_streamer_active(self) -> bool:
        """Return True if the streamer module is active (audio handled by browser)."""
        return getattr(self._app, '_streamer_active', False)

    def _on_playback_state_change(self, is_playing: bool, current_time_ms: float):
        """Called by VideoProcessor on every frame and on pause/stop.

        We only act on transitions:
        - not playing → playing: start audio
        - playing → not playing: stop audio
        - speed mode changed while playing: restart with new tempo
        - streamer active: no audio (browser handles its own audio)
        """
        # Streamer handles its own audio in the browser — suppress FunGen audio
        if self._is_streamer_active():
            if self._audio_running:
                self._ap.stop()
                self._audio_running = False
            return

        tempo = self._get_tempo_for_mode()

        if is_playing and tempo > 0:
            if not self._audio_running:
                # Transition: start audio
                self._ap.start(current_time_ms, tempo)
                self._audio_running = True
                self._current_tempo = tempo
            elif abs(tempo - self._current_tempo) > 0.01:
                # Speed mode changed while playing — restart with new tempo
                self._ap.seek(current_time_ms, tempo)
                self._current_tempo = tempo
            # else: already running at correct tempo — do nothing

        elif is_playing and tempo <= 0:
            # MAX_SPEED — no audio
            if self._audio_running:
                self._ap.stop()
                self._audio_running = False

        else:
            # Not playing (paused or stopped)
            if self._audio_running:
                self._ap.stop()
                self._audio_running = False

    def _on_video_seek(self, frame_index: int):
        """Called by VideoProcessor on explicit seek (timeline click, arrow nav)."""
        if self._is_streamer_active():
            return

        fps = self._vp.fps if self._vp.fps > 0 else 30.0
        position_ms = (frame_index / fps) * 1000.0

        if self._audio_running:
            # Continuous playback — restart audio at new position
            tempo = self._get_tempo_for_mode()
            if tempo > 0:
                self._ap.seek(position_ms, tempo)
        else:
            # Paused / stopped — frame stepping, play short scrub burst
            self._ap.scrub(position_ms, duration_ms=100)

    def _get_tempo_for_mode(self) -> float:
        """Return audio tempo factor for the current speed mode.

        - REALTIME → 1.0
        - SLOW_MOTION → 10.0 / native_fps  (typically ~0.33 for 30fps)
        - MAX_SPEED → 0 (meaning: don't play audio)
        """
        mode = self._app.app_state_ui.selected_processing_speed_mode

        if mode == ProcessingSpeedMode.REALTIME:
            return 1.0
        elif mode == ProcessingSpeedMode.SLOW_MOTION:
            fps = self._vp.fps if self._vp.fps > 0 else 30.0
            return 10.0 / fps
        else:
            # MAX_SPEED
            return 0.0
