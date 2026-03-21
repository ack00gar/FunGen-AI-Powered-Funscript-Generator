"""Tracker Settings UI mixin for ControlPanelUI."""
import imgui
from application.utils import primary_button_style, destructive_button_style
from application.utils.imgui_helpers import DisabledScope as _DisabledScope, tooltip_if_hovered as _tooltip_if_hovered
from application.utils.section_card import section_card as _section_card
from config.element_group_colors import ControlPanelColors as _CPColors


class TrackerSettingsMixin:
    """Mixin providing tracker settings rendering methods."""

    # ---- Dynamic dispatch (new) ----

    def _get_current_tracker_instance(self):
        """Return the active tracker module instance, or None."""
        tr = getattr(self.app, 'tracker', None)
        if tr and hasattr(tr, '_current_tracker'):
            return tr._current_tracker
        return None

    def _render_tracker_dynamic_settings(self):
        """Dispatch to tracker-provided settings UI."""
        tracker_instance = self._get_current_tracker_instance()
        if not tracker_instance:
            imgui.text_disabled("Tracker not initialized.")
            return False

        try:
            # Path A: Direct custom render
            if tracker_instance.render_settings_ui():
                return True
        except Exception as exc:
            imgui.text_colored("Settings UI error: %s" % exc, 1.0, 0.3, 0.3, 1.0)
            return False

        # Path B: Schema auto-render
        schema = tracker_instance.get_settings_schema()
        if schema and schema.get('properties'):
            from application.utils.schema_settings_renderer import render_schema_settings
            return render_schema_settings(schema, self.app.app_settings, tracker_instance)

        return False

    def _render_tracker_debug_panel(self):
        """Render debug panel only if the tracker actually provides content."""
        tracker_instance = self._get_current_tracker_instance()
        if not tracker_instance:
            return
        # Only show the header if the tracker overrides render_debug_ui
        # (base class returns False, so skip trackers that don't override it)
        method = getattr(tracker_instance, 'render_debug_ui', None)
        if method is None:
            return
        from tracker.tracker_modules.core.base_tracker import BaseTracker
        if method.__func__ is BaseTracker.render_debug_ui:
            return
        if imgui.collapsing_header("Tracker Debug##TrackerDebugPanel")[0]:
            tracker_instance.render_debug_ui()

    # ---- Setting helpers ----

    @staticmethod
    def _setting_slider_float(settings, target, label, key, attr, lo, hi,
                              fmt="%.2f", tooltip=None, clamp_min=None):
        """Render a slider_float bound to a setting key and target attribute."""
        cur = settings.get(key)
        ch, nv = imgui.slider_float(label, cur, lo, hi, fmt)
        if tooltip:
            _tooltip_if_hovered(tooltip)
        if ch:
            if clamp_min is not None:
                nv = max(clamp_min, nv)
            if nv != cur:
                settings.set(key, nv)
                setattr(target, attr, nv)

    @staticmethod
    def _setting_input_int(settings, target, label, key, attr, min_val=0, tooltip=None):
        """Render an input_int bound to a setting key and target attribute."""
        cur = settings.get(key)
        ch, nv = imgui.input_int(label, cur)
        if tooltip:
            _tooltip_if_hovered(tooltip)
        if ch:
            v = max(min_val, nv)
            if v != cur:
                settings.set(key, v)
                setattr(target, attr, v)

    @staticmethod
    def _setting_checkbox(settings, target, label, key, attr):
        """Render a checkbox bound to a setting key and target attribute."""
        cur = settings.get(key)
        ch, nv = imgui.checkbox(label, cur)
        if ch:
            settings.set(key, nv)
            setattr(target, attr, nv)

    def _render_tracking_axes_mode(self, stage_proc):
        """Renders UI elements for tracking axis mode."""
        # Get tracker info from discovery (works for both live and offline trackers)
        tracker_info = None
        selected_name = getattr(self.app.app_state_ui, 'selected_tracker_name', None)
        if selected_name and self.tracker_ui:
            tracker_info = self.tracker_ui.discovery.get_tracker_info(selected_name)

        is_dual_axis = tracker_info.supports_dual_axis if tracker_info else True
        primary_name = (tracker_info.primary_axis if tracker_info else "stroke").capitalize()
        # Use user's default secondary axis if set, otherwise tracker's declared axis
        user_secondary = self.app.app_settings.get("default_secondary_axis") if hasattr(self.app, 'app_settings') else None
        if user_secondary and tracker_info and tracker_info.supports_dual_axis:
            secondary_name = user_secondary.capitalize()
        else:
            secondary_name = (tracker_info.secondary_axis if tracker_info else "roll").capitalize()

        # Show which axes this tracker outputs
        if tracker_info:
            if is_dual_axis:
                imgui.text_colored(f"T1: {primary_name}  |  T2: {secondary_name}", 0.6, 0.8, 1.0, 1.0)
            else:
                imgui.text_colored(f"T1: {primary_name}  (single axis)", 0.6, 0.8, 1.0, 1.0)
            _tooltip_if_hovered(
                "Axis assignments set by the tracker.\n"
                "Override in Advanced Settings > Axis Assignments."
            )

        if is_dual_axis:
            axis_modes = [
                f"Both ({primary_name} + {secondary_name})",
                f"{primary_name} Only",
                f"{secondary_name} Only",
            ]
        else:
            axis_modes = [f"{primary_name} Only"]

        current_axis_mode_idx = 0
        if is_dual_axis:
            if self.app.tracking_axis_mode == "vertical":
                current_axis_mode_idx = 1
            elif self.app.tracking_axis_mode == "horizontal":
                current_axis_mode_idx = 2

        processor = self.app.processor
        disable_axis_controls = (
            stage_proc.full_analysis_active
            or self.app.is_setting_user_roi_mode
            or (processor and processor.is_processing
                and getattr(processor, 'enable_tracker_processing', False)
                and not processor.pause_event.is_set())
        )
        with _DisabledScope(disable_axis_controls or not is_dual_axis):
            imgui.set_next_item_width(-1)
            axis_mode_changed, new_axis_mode_idx = imgui.combo("##TrackingAxisModeComboGlobal", current_axis_mode_idx, axis_modes)
            if axis_mode_changed and is_dual_axis:
                old_mode = self.app.tracking_axis_mode
                if new_axis_mode_idx == 0:
                    self.app.tracking_axis_mode = "both"
                elif new_axis_mode_idx == 1:
                    self.app.tracking_axis_mode = "vertical"
                else:
                    self.app.tracking_axis_mode = "horizontal"
                if old_mode != self.app.tracking_axis_mode:
                    self.app.project_manager.project_dirty = True
                    self.app.logger.info(f"Tracking axis mode set to: {self.app.tracking_axis_mode}", extra={'status_message': True})
                    self.app.app_settings.set("tracking_axis_mode", self.app.tracking_axis_mode)
                    self.app.energy_saver.reset_activity_timer()

            if is_dual_axis and self.app.tracking_axis_mode != "both":
                imgui.text("Output Single Axis To:")
                output_targets = [f"Timeline 1 ({primary_name})", f"Timeline 2 ({secondary_name})"]
                current_output_target_idx = 1 if self.app.single_axis_output_target == "secondary" else 0

                imgui.set_next_item_width(-1)
                output_target_changed, new_output_target_idx = imgui.combo("##SingleAxisOutputComboGlobal", current_output_target_idx, output_targets)
                if output_target_changed:
                    old_target = self.app.single_axis_output_target
                    self.app.single_axis_output_target = "secondary" if new_output_target_idx == 1 else "primary"
                    if old_target != self.app.single_axis_output_target:
                        self.app.project_manager.project_dirty = True
                        self.app.logger.info(f"Single axis output target set to: {self.app.single_axis_output_target}", extra={'status_message': True})
                        self.app.app_settings.set("single_axis_output_target", self.app.single_axis_output_target)
                        self.app.energy_saver.reset_activity_timer()

    def _render_class_filtering_content(self):
        app = self.app
        classes = app.get_available_tracking_classes()
        if not classes:
            imgui.text_disabled("No classes available (model not loaded or no classes defined).")
            return

        imgui.text_wrapped("Select classes to DISCARD from tracking and analysis.")
        discarded = set(app.discarded_tracking_classes)
        changed_any = False
        num_cols = 3
        if imgui.begin_table("ClassFilterTable", num_cols, flags=imgui.TABLE_SIZING_STRETCH_SAME):
            col = 0
            for cls in classes:
                if col == 0:
                    imgui.table_next_row()
                imgui.table_set_column_index(col)
                is_discarded = (cls in discarded)
                imgui.push_id("discard_cls_%s" % cls)
                clicked, new_val = imgui.checkbox(" %s" % cls, is_discarded)
                imgui.pop_id()
                if clicked:
                    changed_any = True
                    if new_val:
                        discarded.add(cls)
                    else:
                        discarded.discard(cls)
                col = (col + 1) % num_cols
            imgui.end_table()

        if changed_any:
            new_list = sorted(list(discarded))
            if new_list != app.discarded_tracking_classes:
                app.discarded_tracking_classes = new_list
                app.app_settings.set("discarded_tracking_classes", new_list)
                app.project_manager.project_dirty = True
                app.logger.info("Discarded classes updated: %s" % new_list, extra={"status_message": True})
                app.energy_saver.reset_activity_timer()

        imgui.spacing()
        if imgui.button(
            "Clear All Discards##ClearDiscardFilters",
            width=imgui.get_content_region_available_width(),
        ):
            if app.discarded_tracking_classes:
                app.discarded_tracking_classes.clear()
                app.app_settings.set("discarded_tracking_classes", [])
                app.project_manager.project_dirty = True
                app.logger.info("All class discard filters cleared.", extra={"status_message": True})
                app.energy_saver.reset_activity_timer()
        _tooltip_if_hovered("Unchecks all classes, enabling all classes for tracking/analysis.")
