"""Ultimate Autotune trigger for ApplicationLogic.

The legacy "Stage 1 performance autotuner" (scan-best-P/C-combination thread
and its GUI) has been removed; this module now only holds the non-interactive
Ultimate Autotune helper used by batch processing, the CLI, and the default
post-analysis hook.
"""


class AppAutotuner:
    """Thin facade kept so AppLogic can call ``self.autotuner.<method>``.
    New code can call the Ultimate Autotune plugin directly."""

    def __init__(self, app_logic):
        self.app = app_logic

    def trigger_ultimate_autotune_with_defaults(self, timeline_num: int):
        """Non-interactively run the Ultimate Autotune pipeline with default
        parameters on the given timeline."""
        self.app.logger.info(
            f"Triggering default Ultimate Autotune for Timeline {timeline_num}...")
        fs_proc = self.app.funscript_processor
        funscript_instance, axis_name = fs_proc._get_target_funscript_object_and_axis(timeline_num)

        if not funscript_instance or not axis_name:
            self.app.logger.error(
                f"Ultimate Autotune (auto): Could not find target funscript for T{timeline_num}.")
            return

        params = fs_proc.get_default_ultimate_autotune_params()
        op_desc = "Auto-Applied Ultimate Autotune"
        actions_before = list(funscript_instance.get_axis_actions(axis_name) or [])

        try:
            from funscript.plugins.base_plugin import plugin_registry
            from funscript.plugins import ultimate_autotune_plugin  # noqa: F401 (registration side-effect)
            ultimate_plugin = plugin_registry.get_plugin('Ultimate Autotune')

            if ultimate_plugin:
                result = ultimate_plugin.transform(funscript_instance, axis_name, **params)
                if result:
                    fs_proc._post_mutation_refresh(timeline_num, op_desc)
                    self.app.logger.info(
                        "Default Ultimate Autotune applied successfully.",
                        extra={'status_message': True, 'duration': 5.0})
                    actions_after = list(funscript_instance.get_axis_actions(axis_name) or [])
                    from application.classes.undo_manager import BulkReplaceCmd
                    self.app.undo_manager.push_done(BulkReplaceCmd(
                        timeline_num, actions_before, actions_after,
                        f"Ultimate Autotune (T{timeline_num})"))
                else:
                    self.app.logger.warning(
                        "Default Ultimate Autotune failed to produce a result.",
                        extra={'status_message': True})
            else:
                self.app.logger.error(
                    "Ultimate Autotune plugin not available.",
                    extra={'status_message': True})
        except Exception as e:
            self.app.logger.error(
                f"Error applying Ultimate Autotune: {e}",
                extra={'status_message': True})
