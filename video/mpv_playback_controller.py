"""
MpvPlaybackController — high-level review-mode playback via mpv IPC.

Wraps MpvIPCBridge with app-level integration:
- Stops FFmpeg audio sync (mpv handles audio natively via CoreAudio/VideoToolbox)
- Updates processor.current_frame_index from IPC position callbacks
- Fires processor playback-state callbacks so DeviceControlVideoIntegration stays in sync
- Routes playback control actions from handle_playback_control

For VR content, a panel-crop is applied so mpv shows the correct eye (left/top).
No v360 dewarping — VideoToolbox hardware decode stays fully on GPU.
"""

import logging

from video.mpv_ipc_bridge import MpvIPCBridge

logger = logging.getLogger(__name__)


class MpvPlaybackController:
    """
    Review-mode playback engine backed by mpv.

    Mutually exclusive with FFmpeg-pipe analysis mode: processor stays
    stopped while mpv is active.  The processor's current_frame_index is
    updated every time mpv pushes a time-pos event, so the timeline,
    scrubber, and device-sync code all continue to work without changes.
    """

    def __init__(self, app):
        self._app = app
        self._bridge: MpvIPCBridge | None = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        return self._bridge is not None and self._bridge.is_alive()

    @property
    def is_playing(self) -> bool:
        return self._bridge is not None and self._bridge.is_playing

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, video_path: str, start_frame: int = 0, fullscreen: bool = False) -> bool:
        """
        Launch mpv for review playback.

        For VR (SBS/TB) videos, a panel-crop filter is added so mpv shows
        the correct eye at the right aspect ratio. Hardware decode (VideoToolbox)
        is unaffected by a simple crop filter.
        """
        if self._bridge is not None:
            self.stop()

        processor = self._app.processor
        fps = processor.fps if processor and processor.fps > 0 else 30.0
        start_ms = (start_frame / fps) * 1000.0

        extra_args = self._vr_crop_args(processor)
        if extra_args:
            logger.info(
                f"MpvPlaybackController: VR crop applied "
                f"({getattr(processor, 'vr_input_format', '?')}): {extra_args}"
            )

        self._bridge = MpvIPCBridge(
            video_path, start_ms=start_ms, fullscreen=fullscreen, extra_args=extra_args
        )
        if not self._bridge.start():
            self._bridge = None
            logger.error("MpvPlaybackController: mpv failed to start")
            return False

        # Disable FFmpeg audio sync — mpv handles audio natively
        if self._app._audio_sync:
            self._app._audio_sync.stop()

        self._bridge.add_position_callback(self._on_position)
        self._bridge.play()
        logger.info("MpvPlaybackController: review mode started")
        return True

    def stop(self):
        """Stop mpv and restore FFmpeg audio sync."""
        if self._bridge:
            self._bridge.stop()
            self._bridge = None

        if self._app._audio_sync:
            try:
                self._app._audio_sync.start()
            except Exception:
                pass  # already running is fine

        logger.info("MpvPlaybackController: review mode stopped")

    # ------------------------------------------------------------------
    # Playback controls
    # ------------------------------------------------------------------

    def play(self):
        if self._bridge:
            self._bridge.play()

    def pause(self):
        if self._bridge:
            self._bridge.pause()

    def seek(self, frame_index: int):
        """Seek to a frame index (converts to ms for mpv)."""
        if not self._bridge:
            return
        processor = self._app.processor
        fps = processor.fps if processor and processor.fps > 0 else 30.0
        self._bridge.seek((frame_index / fps) * 1000.0)

    def handle_action(self, action_name: str):
        """Route a playback control action (called from handle_playback_control)."""
        if action_name == "play_pause":
            if self.is_playing:
                self.pause()
            else:
                self.play()
        elif action_name == "stop":
            self.stop()
        elif action_name == "jump_start":
            self.seek(0)
        elif action_name == "jump_end":
            processor = self._app.processor
            if processor:
                self.seek(max(0, processor.total_frames - 1))
        elif action_name in ("prev_frame", "next_frame"):
            processor = self._app.processor
            if processor:
                delta = -1 if action_name == "prev_frame" else 1
                self.seek(max(0, processor.current_frame_index + delta))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _vr_crop_args(self, processor) -> list:
        """
        Return mpv --vf crop arg to show the correct eye/panel for VR content.

        SBS / LR (_sbs, _lr): left eye at x=0  — crop left half
        RL        (_rl):       left eye at x=iw/2 — crop right half as "left"
        TB        (_tb):       top eye — crop top half
        Mono / 2D:             no crop

        Panel override from app_settings['vr_panel_selection']:
          'left' (default) | 'right' | 'full' (no crop)
        """
        if not processor or processor.determined_video_type != 'VR':
            return []

        fmt = (processor.vr_input_format or '').lower()
        app_settings = getattr(self._app, 'app_settings', None)
        panel = app_settings.get('vr_panel_selection', 'left') if app_settings else 'left'

        if fmt.endswith(('_sbs', '_lr')):
            if panel == 'full':
                return []
            crop = 'crop=iw/2:ih:iw/2:0' if panel == 'right' else 'crop=iw/2:ih:0:0'
        elif fmt.endswith('_rl'):
            if panel == 'full':
                return []
            crop = 'crop=iw/2:ih:0:0' if panel == 'right' else 'crop=iw/2:ih:iw/2:0'
        elif fmt.endswith('_tb'):
            crop = 'crop=iw:ih/2:0:0'  # always top, no panel override
        else:
            return []

        return [f"--vf={crop}"]

    def _on_position(self, pos_ms: float, dur_ms: float):
        """
        IPC callback — runs on the mpv-ipc-poller background thread.

        Updates processor.current_frame_index so the timeline, scrubber,
        and device-sync code all read the correct position.
        Uses round() so pausing snaps to the nearest frame boundary.
        """
        processor = self._app.processor
        if not processor or processor.fps <= 0:
            return

        frame_index = round(pos_ms / 1000.0 * processor.fps)
        frame_index = max(0, min(frame_index, processor.total_frames - 1))

        processor.current_frame_index = frame_index  # GIL-safe plain int assignment

        try:
            processor._notify_playback_state_callbacks(self.is_playing, pos_ms)
        except Exception:
            pass
