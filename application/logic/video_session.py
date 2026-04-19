"""Video session: open/close orchestration across every subsystem that cares.

Was scattered across AppFileManager.open_video_from_path and .close_video_action.
Those entry points still exist as thin delegators so external callers don't change.
"""
from __future__ import annotations

import glob
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from application.logic.app_logic import ApplicationLogic


def _is_feature_available(name: str) -> bool:
    from application.utils.feature_detection import is_feature_available
    return is_feature_available(name)


class VideoSession:
    __slots__ = ("app",)

    def __init__(self, app: "ApplicationLogic") -> None:
        self.app = app

    # ---- OPEN ----
    def open(self, file_path: str) -> bool:
        """Open a video file; returns True on success. Resets prior state first."""
        app = self.app
        fm = app.file_manager

        is_remote = bool(file_path) and file_path.startswith(('http://', 'https://'))
        if not file_path or (not is_remote and not os.path.exists(file_path)):
            app.logger.error(f"Video file not found: {file_path}")
            return False

        app.logger.info(f"Loading video: {os.path.basename(file_path)}...", extra={'status_message': True})
        app.notify(f"Loading {os.path.basename(file_path)}...", "info", 2.0)

        # Fresh state
        self.close(clear_funscript_unconditionally=True)

        # Transparent proxy auto-load: if a registered Foo.fungen-proxy.mp4
        # exists for the source, open the proxy instead.
        if not is_remote:
            try:
                from video.proxy_builder import is_proxy_filename, proxy_path_from_sidecar
                if not is_proxy_filename(file_path):
                    proxy = proxy_path_from_sidecar(file_path)
                    if proxy:
                        app.logger.info(
                            f"Using existing proxy: {os.path.basename(proxy)}",
                            extra={'status_message': True})
                        fm._original_source_for_proxy = file_path
                        file_path = proxy
            except Exception:
                pass

        # Hand off to the VideoProcessor.
        success = app.processor.open_video(file_path)

        if success:
            fm.video_path = file_path
            app.project_manager.project_dirty = True
            self._reset_ui_for_new_video()
            app.funscript_processor.update_funscript_stats_for_timeline(1, "Video Loaded")
            app.funscript_processor.update_funscript_stats_for_timeline(2, "Video Loaded")

            if _is_feature_available("subtitle_translation"):
                app.subtitle_track = None
                self._auto_load_subtitles(file_path)

            if app._audio_player:
                has_audio = app.processor.video_info.get("has_audio", False)
                fps = app.processor.fps
                app.logger.info(f"Audio: set_video has_audio={has_audio} fps={fps}")
                app._audio_player.set_video(file_path, has_audio, fps)
            else:
                app.logger.debug("Audio: _audio_player is None, skipping set_video")
        else:
            fm.video_path = ""
            app.logger.error(f"Failed to open video file: {os.path.basename(file_path)}",
                             extra={'status_message': True})
        return success

    def _reset_ui_for_new_video(self) -> None:
        ui = self.app.app_state_ui
        ui.reset_video_zoom_pan()
        # Park the timeline at t=0; force_sync alone is gated by
        # timeline_interaction_active which can be stale from a prior file.
        ui.timeline_pan_offset_ms = 0.0
        ui.timeline_interaction_active = False
        ui.force_timeline_pan_to_current_frame = True

    def _auto_load_subtitles(self, video_path: str) -> None:
        """Load a .srt next to the video if present; prefer .en.srt / .bilingual.srt."""
        app = self.app
        try:
            base = os.path.splitext(video_path)[0]
            candidates = glob.glob(f"{base}*.srt")
            if not candidates:
                return
            srt_path = candidates[0]
            for c in candidates:
                if '.en.' in c or '.bilingual.' in c:
                    srt_path = c
                    break

            from subtitle_translation.srt_importer import import_srt
            track = import_srt(srt_path)
            if track and len(track) > 0:
                app.subtitle_track = track
                app.logger.info(
                    f"Auto-loaded {len(track)} subtitles from {os.path.basename(srt_path)}")
                gui = getattr(app, 'gui_instance', None)
                cp = gui.control_panel_ui if gui else None
                tool = getattr(cp, '_subtitle_tool', None) if cp else None
                if tool:
                    tool.track = track
                    tool.state = tool.STATE_EDITING
        except Exception as e:
            app.logger.debug(f"No subtitles auto-loaded: {e}")

    # ---- CLOSE ----
    def close(self, clear_funscript_unconditionally: bool = False,
              skip_tracker_reset: bool = False) -> None:
        """Tear down every subsystem attached to the current video."""
        app = self.app
        fm = app.file_manager

        # Stop + reset processor
        if app.processor:
            if app.processor.is_processing:
                app.processor.stop_processing()
            try:
                app.processor.reset(close_video=True, skip_tracker_reset=skip_tracker_reset)
            except TypeError:
                app.processor.reset(close_video=True)

        # Tracker state
        if app.tracker:
            app.tracker.user_roi_fixed = None
            app.tracker.user_roi_initial_point_relative = None
            app.tracker.user_roi_tracked_point_relative = None
            app.tracker.user_roi_current_flow_vector = None
            app.tracker.cleanup()

        # File paths + stage overlays
        fm.video_path = ""
        fm.preprocessed_video_path = None
        app.stage_processor.reset_stage_status(stages=("stage1", "stage2", "stage3"))
        app.funscript_processor.video_chapters.clear()
        fm.clear_stage2_overlay_data()
        if hasattr(app, 'stage3_mixed_debug_data'):
            app.stage3_mixed_debug_data = None
        if hasattr(app, 'stage3_mixed_debug_frame_map'):
            app.stage3_mixed_debug_frame_map = None

        # Audio
        if app._audio_player:
            app._audio_player.stop()
        app.audio_waveform_data = None
        app.app_state_ui.show_audio_waveform = False

        # Funscripts: T1 cleared only when asked; T2 always cleared.
        keep_t1 = not clear_funscript_unconditionally and bool(fm.loaded_funscript_path)
        if not keep_t1:
            if app.processor and app.processor.tracker and app.processor.tracker.funscript:
                app.funscript_processor.clear_timeline_history_and_set_new_baseline(
                    1, [], "Video Closed (T1 Cleared)")
            fm.funscript_path = ""
            fm.loaded_funscript_path = ""

        if app.processor and app.processor.tracker and app.processor.tracker.funscript:
            app.funscript_processor.clear_timeline_history_and_set_new_baseline(
                2, [], "Video Closed (T2 Cleared)")

        app.funscript_processor.update_funscript_stats_for_timeline(1, "Video Closed")
        app.funscript_processor.update_funscript_stats_for_timeline(2, "Video Closed")

        fm.logger.debug("Video closed.", extra={'status_message': True})
        app.energy_saver.reset_activity_timer()
        app.app_state_ui.heatmap_dirty = True
        app.app_state_ui.funscript_preview_dirty = True
        app.project_manager.project_dirty = True
