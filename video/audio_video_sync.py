"""Bridges VideoProcessor playback/seek events to AudioPlayer."""

import logging

from config.constants import ProcessingSpeedMode
from video.audio_player import AudioPlayer, SOUNDDEVICE_AVAILABLE

logger = logging.getLogger(__name__)


class AudioVideoSync:

    def __init__(self, video_processor, audio_player: AudioPlayer, app_instance):
        self._vp = video_processor
        self._ap = audio_player
        self._app = app_instance
        self._registered = False

        # The playback-state callback fires every frame, so track state and
        # only act on transitions.
        self._audio_running = False
        self._current_tempo = 1.0

    def start(self):
        if not SOUNDDEVICE_AVAILABLE or self._registered:
            return
        self._vp.register_playback_state_callback(self._on_playback_state_change)
        self._vp.register_seek_callback(self._on_video_seek)
        self._registered = True
        logger.debug("AudioVideoSync started")

    def stop(self):
        if self._registered:
            self._vp.unregister_playback_state_callback(self._on_playback_state_change)
            self._vp.unregister_seek_callback(self._on_video_seek)
            self._registered = False
        self._ap.stop()
        self._audio_running = False
        logger.info("AudioVideoSync stopped")

    def update_settings(self, volume: float, muted: bool):
        self._ap.set_volume(volume)
        self._ap.set_mute(muted)

    def _is_streamer_active(self) -> bool:
        return getattr(self._app, '_streamer_active', False)

    def _mpv_owns_audio(self) -> bool:
        # libmpv plays audio itself unless the tracker is running; the tracker
        # pulls frames at target_delay and mpv's audio would drift.
        gui = getattr(self._app, 'gui_instance', None) if self._app else None
        disp = getattr(gui, 'mpv_display', None) if gui else None
        if disp is None or not getattr(disp, 'is_loaded', False):
            return False
        if not getattr(disp, 'with_audio', False):
            return False
        tracker = getattr(self._app, 'tracker', None) if self._app else None
        if tracker is not None and getattr(tracker, 'tracking_active', False):
            return False
        return True

    def _on_playback_state_change(self, is_playing: bool, current_time_ms: float):
        if self._is_streamer_active() or self._mpv_owns_audio():
            if self._audio_running:
                self._ap.stop()
                self._audio_running = False
            return

        tempo = self._get_tempo_for_mode()

        if is_playing and tempo > 0:
            if not self._audio_running:
                self._ap.start(current_time_ms, tempo)
                self._audio_running = True
                self._current_tempo = tempo
            elif abs(tempo - self._current_tempo) > 0.01:
                self._ap.seek(current_time_ms, tempo)
                self._current_tempo = tempo

        elif is_playing and tempo <= 0:
            if self._audio_running:
                self._ap.stop()
                self._audio_running = False

        else:
            if self._audio_running:
                self._ap.stop()
                self._audio_running = False

    def _on_video_seek(self, frame_index: int):
        if self._is_streamer_active():
            return

        fps = self._vp.fps if self._vp.fps > 0 else 30.0
        position_ms = (frame_index / fps) * 1000.0

        if self._audio_running:
            tempo = self._get_tempo_for_mode()
            if tempo > 0:
                self._ap.seek(position_ms, tempo)

    def _get_tempo_for_mode(self) -> float:
        mode = self._app.app_state_ui.selected_processing_speed_mode

        if mode == ProcessingSpeedMode.REALTIME:
            return 1.0
        elif mode == ProcessingSpeedMode.SLOW_MOTION:
            fps = self._vp.fps if self._vp.fps > 0 else 30.0
            return 10.0 / fps
        else:
            # MAX_SPEED: suppress audio.
            return 0.0
