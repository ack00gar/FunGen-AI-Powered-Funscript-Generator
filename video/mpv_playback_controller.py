"""Review-mode playback via a standalone mpv subprocess (IPC)."""

import logging

from video.mpv_ipc_bridge import MpvIPCBridge

logger = logging.getLogger(__name__)


class MpvPlaybackController:

    def __init__(self, app):
        self._app = app
        self._bridge: MpvIPCBridge | None = None

    @property
    def is_active(self) -> bool:
        return self._bridge is not None and self._bridge.is_alive()

    @property
    def is_playing(self) -> bool:
        return self._bridge is not None and self._bridge.is_playing

    def start(self, video_path: str, start_frame: int = 0, fullscreen: bool = False) -> bool:
        if self._bridge is not None:
            self.stop()

        processor = getattr(self._app, 'processor', None)
        fps = processor.fps if processor and processor.fps > 0 else 30.0
        start_ms = (start_frame / fps) * 1000.0

        extra_args = self._vr_crop_args(processor)
        if extra_args:
            logger.info(
                f"VR crop applied ({getattr(processor, 'vr_input_format', '?')}): {extra_args}"
            )

        self._bridge = MpvIPCBridge(
            video_path, start_ms=start_ms, fullscreen=fullscreen, extra_args=extra_args
        )
        if not self._bridge.start():
            self._bridge = None
            logger.error("mpv failed to start")
            return False

        # Mute the embedded display so we don't get two audio streams.
        self._embedded_was_muted = False
        gui = getattr(self._app, 'gui_instance', None)
        embedded = getattr(gui, 'mpv_display', None) if gui else None
        if embedded is not None and getattr(embedded, 'is_loaded', False):
            try:
                embedded.pause()
                if embedded._player is not None:
                    self._embedded_was_muted = bool(getattr(embedded._player, 'mute', False))
                    embedded._player.mute = True
            except Exception as e:
                logger.debug(f"embedded mute failed: {e}")

        audio_sync = getattr(self._app, '_audio_sync', None)
        if audio_sync:
            audio_sync.stop()

        self._bridge.add_position_callback(self._on_position)
        self._bridge.play()
        logger.info("review mode started")
        return True

    def stop(self):
        if self._bridge:
            self._bridge.stop()
            self._bridge = None

        gui = getattr(self._app, 'gui_instance', None)
        embedded = getattr(gui, 'mpv_display', None) if gui else None
        if embedded is not None and getattr(embedded, 'is_loaded', False):
            try:
                if embedded._player is not None:
                    embedded._player.mute = bool(getattr(self, '_embedded_was_muted', False))
            except Exception as e:
                logger.debug(f"embedded unmute failed: {e}")

        processor = getattr(self._app, 'processor', None)
        if processor and hasattr(processor, 'seek_video'):
            processor.seek_video(processor.current_frame_index)

        audio_sync = getattr(self._app, '_audio_sync', None)
        if audio_sync:
            try:
                audio_sync.start()
            except Exception:
                pass

        logger.info("review mode stopped")

    def poll_external_exit(self) -> bool:
        # Returns True when the fullscreen subprocess has exited and we cleaned up.
        if self._bridge is None:
            return False
        if self._bridge.is_alive():
            return False
        logger.info("fullscreen mpv exited externally")
        self.stop()
        return True

    def play(self):
        if self._bridge:
            self._bridge.play()

    def pause(self):
        if self._bridge:
            self._bridge.pause()

    def seek(self, frame_index: int):
        if not self._bridge:
            return
        processor = getattr(self._app, 'processor', None)
        fps = processor.fps if processor and processor.fps > 0 else 30.0
        if processor:
            processor.current_frame_index = max(0, frame_index)
        self._bridge.seek((frame_index / fps) * 1000.0)

    def handle_action(self, action_name: str):
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
            processor = getattr(self._app, 'processor', None)
            if processor:
                self.seek(max(0, processor.total_frames - 1))
        elif action_name in ("prev_frame", "next_frame"):
            processor = getattr(self._app, 'processor', None)
            if processor:
                delta = -1 if action_name == "prev_frame" else 1
                self.seek(max(0, processor.current_frame_index + delta))

    def _vr_crop_args(self, processor) -> list:
        """Build --vf crop= args for a standalone mpv subprocess."""
        if not processor or processor.determined_video_type != 'VR':
            return []
        from video import vr_panel
        app_settings = getattr(self._app, 'app_settings', None)
        eye = vr_panel.read_setting(app_settings, default=vr_panel.EYE_LEFT)
        if eye == vr_panel.EYE_FULL:
            return []
        region = vr_panel.resolve_eye(processor.vr_input_format, eye)
        if region.is_full():
            return []
        def _frac(val: float, axis: str) -> str:
            if val == 0.0:
                return "0"
            if val == 0.5:
                return f"{axis}/2"
            return axis
        crop = (f"crop={_frac(region.w, 'iw')}:{_frac(region.h, 'ih')}"
                f":{_frac(region.x, 'iw')}:{_frac(region.y, 'ih')}")
        return [f"--vf={crop}"]

    def _on_position(self, pos_ms: float, dur_ms: float):
        processor = getattr(self._app, 'processor', None)
        if not processor or processor.fps <= 0:
            return

        frame_index = round(pos_ms / 1000.0 * processor.fps)
        frame_index = max(0, min(frame_index, processor.total_frames - 1))

        processor.current_frame_index = frame_index

        try:
            processor._notify_playback_state_callbacks(self.is_playing, pos_ms)
        except Exception:
            pass
