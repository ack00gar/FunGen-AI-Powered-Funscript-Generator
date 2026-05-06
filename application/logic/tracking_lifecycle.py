"""End-of-tracking lifecycle hooks.

Handles the transitions that happen when a tracking session (live or
offline analysis) finishes: save raw funscript, run the post-analysis
pipeline (CLI preset / GUI steps / default autotune), save the final
funscript, save the project, signal the batch loop, etc.

Also owns the small "pending action after tracking" state used to defer
operations that need to wait for an in-progress tracking pass to end.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

from config.constants import PROJECT_FILE_EXTENSION

if TYPE_CHECKING:
    from application.logic.app_logic import ApplicationLogic


class TrackingLifecycleController:
    """End-of-tracking / end-of-analysis coordinator."""

    __slots__ = ("app",)

    def __init__(self, app: "ApplicationLogic") -> None:
        self.app = app

    # ---- Pending action ----
    def set_pending_action(self, action_type: str, **kwargs) -> None:
        """Defer an action until the in-progress tracking pass ends."""
        app = self.app
        app.pending_action_after_tracking = {"type": action_type, "data": kwargs}
        app.logger.info(f"Pending action set after tracking: {action_type} with data {kwargs}")

    def clear_pending_action(self) -> None:
        app = self.app
        if app.pending_action_after_tracking:
            app.logger.info(
                f"Cleared pending action: {app.pending_action_after_tracking.get('type')}")
        app.pending_action_after_tracking = None

    # ---- Post-analysis pipeline ----
    def run_post_analysis_pipeline(self, frame_range=None) -> bool:
        """Run post-analysis processing after tracking completes.

        Priority: CLI --pipeline preset > batch autotune flag > per-axis
        preset assignments > single GUI pipeline > default Ultimate Autotune.
        Returns True if any processing was applied.
        """
        app = self.app

        # CLI pipeline preset takes priority
        pipeline_preset = getattr(app, 'batch_pipeline_preset', None)
        if pipeline_preset:
            app.logger.info(f"Applying CLI pipeline preset '{pipeline_preset}' after analysis.")
            from application.classes.plugin_pipeline import PluginPipeline
            pipeline = PluginPipeline(app)
            if pipeline.load_preset(pipeline_preset):
                funscript_obj = app.funscript_processor.get_funscript_obj()
                if funscript_obj:
                    success, errors = pipeline.run_with_target(funscript_obj)
                    for err in errors:
                        app.logger.warning(f"Pipeline: {err}")
                    return success
            else:
                app.logger.warning(f"Pipeline preset '{pipeline_preset}' not found.")
            return False

        # Batch mode: use batch autotune flag
        if app.is_batch_processing_active:
            if app.batch_apply_ultimate_autotune:
                app.logger.info("Applying Ultimate Autotune for batch processing.")
                app.trigger_ultimate_autotune_with_defaults(timeline_num=1)
                return True
            return False

        # Interactive mode: check auto_apply_post_processing setting
        if not app.app_settings.config.funscript.auto_apply_post_processing:
            app.logger.info("Auto post-processing disabled, skipping.")
            return False

        funscript_obj = app.funscript_processor.get_funscript_obj()
        if not funscript_obj:
            return False

        # Per-axis preset assignments (e.g. {"T1": "Full Enhancement", "T2": "Light Polish"})
        assignments = app.app_settings.config.plugin_pipeline.auto_assignments
        if assignments:
            from application.classes.plugin_pipeline import PluginPipeline, timeline_label_to_axis
            any_applied = False
            for axis_label, preset_name in assignments.items():
                if not preset_name:
                    continue
                pipeline = PluginPipeline(app)
                if pipeline.load_preset(preset_name):
                    axis_name = timeline_label_to_axis(axis_label, funscript_obj)
                    app.logger.info(f"Auto pipeline: '{preset_name}' on {axis_label} ({axis_name})")
                    success, errors = pipeline.run(funscript_obj, axis=axis_name)
                    for err in errors:
                        app.logger.warning(f"Pipeline ({axis_label}): {err}")
                    any_applied = any_applied or success
                else:
                    app.logger.warning(
                        f"Auto pipeline: preset '{preset_name}' not found for {axis_label}")
            if any_applied:
                return True

        # Single GUI pipeline with target_axis
        gui = getattr(app, 'gui_instance', None)
        if gui and hasattr(gui, 'plugin_pipeline_ui'):
            pipeline = gui.plugin_pipeline_ui.pipeline
            enabled_steps = [s for s in pipeline.steps if s.enabled]
            if enabled_steps:
                app.logger.info(
                    f"Running pipeline ({len(enabled_steps)} steps, target: {pipeline.target_axis}) "
                    f"after analysis.")
                success, errors = pipeline.run_with_target(funscript_obj)
                for err in errors:
                    app.logger.warning(f"Pipeline: {err}")
                return success

        # Fallback: run Ultimate Autotune as default post-processing
        app.logger.info("Running default Ultimate Autotune after analysis.")
        app.trigger_ultimate_autotune_with_defaults(timeline_num=1)
        return True

    # ---- Completion hooks ----
    def on_offline_analysis_completed(self, payload: Dict) -> None:
        """Finalize a completed offline analysis run: save raw, post-process,
        save final, save project, signal batch loop."""
        app = self.app
        video_path = payload.get("video_path")

        if not video_path:
            app.logger.warning("Completion event is missing its video path. Cannot save funscripts.")
            app.save_and_reset_complete_event.set()
            return

        # Chapter list is owned by funscript_processor (populated by the
        # stage2_results_success event).
        chapters_for_save = app.funscript_processor.video_chapters

        # 1. Save raw funscript before post-processing touches it.
        app.logger.info("Offline analysis completed. Saving raw funscript before post-processing.")
        app.file_manager.save_raw_funscripts_after_generation(video_path)

        # 2. Post-processing.
        any_processing_applied = self.run_post_analysis_pipeline()

        action_count = len(app.funscript_processor.get_actions('primary') or [])
        app.notify(f"Analysis complete - {action_count} points generated", "success")

        if any_processing_applied:
            app.logger.info("Saving final (post-processed) funscripts...")
            app.file_manager.save_final_funscripts(video_path, chapters=chapters_for_save)
        else:
            app.logger.info(
                "No post-processing was applied. Saving raw funscript with .raw.funscript "
                "extension to video location.")
            app.file_manager.save_raw_funscripts_next_to_video(video_path)

        # 5. Save the project file.
        app.logger.info("Saving project file for completed video...")
        project_filepath = app.file_manager.get_output_path_for_file(video_path, PROJECT_FILE_EXTENSION)
        app.project_manager.save_project(project_filepath)

        # Signal any batch waiter (legacy batch_processor or watched-folder
        # BatchWorker). Single-video runs have no waiter; the set is a no-op.
        app.save_and_reset_complete_event.set()

        # CLI + batch: no GUI to drive project reset, so do it here.
        if not app.gui_instance and app.is_batch_processing_active:
            app.logger.info("CLI Mode: Resetting project state for next video in batch.")
            app.reset_project_state(for_new_project=False)

    def on_processing_stopped(self,
                              was_scripting_session: bool = False,
                              scripted_frame_range: Optional[Tuple[int, int]] = None) -> None:
        """Called when video processing (tracking / playback) stops.

        Handles post-processing for live tracking sessions and dispatches
        any pending-action follow-ups (e.g. gap-merge finalization).
        """
        app = self.app
        app.logger.debug(
            f"on_processing_stopped triggered. Was scripting: {was_scripting_session}, "
            f"Range: {scripted_frame_range}")

        # Handle pending actions first.
        if app.pending_action_after_tracking:
            action_info = app.pending_action_after_tracking
            self.clear_pending_action()
            app.logger.info(f"Processing pending action: {action_info['type']}")
            action_type = action_info['type']
            action_data = action_info['data']
            if action_type == 'finalize_gap_merge_after_tracking':
                chapter1_id = action_data.get('chapter1_id')
                chapter2_id = action_data.get('chapter2_id')
                if not all([chapter1_id, chapter2_id]):
                    app.logger.error(
                        f"Missing data for finalize_gap_merge_after_tracking: {action_data}")
                    return
                if hasattr(app.funscript_processor, 'finalize_merge_after_gap_tracking'):
                    app.funscript_processor.finalize_merge_after_gap_tracking(chapter1_id, chapter2_id)
                else:
                    app.logger.error(
                        "FunscriptProcessor missing finalize_merge_after_gap_tracking method.")
            else:
                app.logger.warning(f"Unknown pending action type: {action_type}")

        # Live scripting session: save the raw script first, then post-process.
        if was_scripting_session:
            video_path = app.file_manager.video_path
            if not video_path:
                app.logger.warning("Live session ended, but no video path is available to save the raw funscript.")
                app.save_and_reset_complete_event.set()
                return

            # Batch fires on_processing_stopped twice (EOS + post-join);
            # skip the second spawn so daemons don't race the same file.
            existing = getattr(app, '_post_live_thread', None)
            if existing is not None and existing.is_alive():
                app.logger.debug("Post-live save already running, skipping duplicate.")
                return

            # Heavy work (file I/O + Ultimate Autotune) runs on a daemon
            # thread so the Stop Tracking click does not block the UI.
            import threading as _threading
            scripted_range = scripted_frame_range

            def _post_live_work() -> None:
                try:
                    app.file_manager.save_raw_funscripts_after_generation(video_path)
                    self._invalidate_all_timeline_caches()
                    any_processing_applied = self.run_post_analysis_pipeline(frame_range=scripted_range)
                    if any_processing_applied:
                        chapters_for_save = app.funscript_processor.video_chapters
                        app.file_manager.save_final_funscripts(video_path, chapters=chapters_for_save)
                    else:
                        app.file_manager.save_raw_funscripts_next_to_video(video_path)
                    self._invalidate_all_timeline_caches()
                except Exception as e:
                    app.logger.error(f"Live post-processing failed: {e}", exc_info=True)
                finally:
                    # Unblock any batch waiter before the next open_video.
                    app.save_and_reset_complete_event.set()

            app.logger.info("Live session ended, saving funscript and running post-processing...")
            t = _threading.Thread(
                target=_post_live_work, name="LivePostProcess", daemon=True)
            app._post_live_thread = t
            t.start()

    def _invalidate_all_timeline_caches(self) -> None:
        """Invalidate every timeline editor's cached action arrays so the
        next render reflects live-tracking data committed during the run.
        Earlier code read app.interactive_timeline1/2 which are never
        assigned; timelines live on gui_instance.timeline_editor1/2."""
        gui = getattr(self.app, 'gui_instance', None)
        if gui is None:
            return
        for attr in ('timeline_editor1', 'timeline_editor2'):
            tl = getattr(gui, attr, None)
            if tl is not None and hasattr(tl, 'invalidate_cache'):
                tl.invalidate_cache()
                self.app.logger.debug(f"{attr} cache invalidated after live session completion")
        for t_num, editor in getattr(gui, '_extra_timeline_editors', {}).items():
            if hasattr(editor, 'invalidate_cache'):
                editor.invalidate_cache()
                self.app.logger.debug(f"Timeline {t_num} cache invalidated after live session completion")
