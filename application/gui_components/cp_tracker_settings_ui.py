"""Tracker Settings UI mixin for ControlPanelUI."""
import imgui
from application.utils import primary_button_style, destructive_button_style
from application.utils.imgui_helpers import DisabledScope as _DisabledScope


def _tooltip_if_hovered(text):
    if imgui.is_item_hovered():
        imgui.set_tooltip(text)


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

    # ---- Legacy per-tracker renders (kept as fallback) ----

    def _render_live_tracker_settings(self):
        app = self.app
        tr = app.tracker
        if not tr:
            imgui.text_disabled("Tracker not initialized.")
            return

        settings = app.app_settings

        imgui.indent()
        if imgui.collapsing_header("Detection & ROI Definition##ROIDetectionTrackerMenu")[0]:
            cur_conf = settings.get("live_tracker_confidence_threshold")
            ch, new_conf = imgui.slider_float("Obj. Confidence##ROIConfTrackerMenu", cur_conf, 0.1, 0.95, "%.2f")
            if imgui.is_item_hovered():
                imgui.set_tooltip("Minimum confidence for object detection (higher = fewer false positives, lower = more detections)")
            if ch and new_conf != cur_conf:
                settings.set("live_tracker_confidence_threshold", new_conf)
                tr.confidence_threshold = new_conf

            cur_pad = settings.get("live_tracker_roi_padding")
            ch, new_pad = imgui.input_int("ROI Padding##ROIPadTrackerMenu", cur_pad)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Pixels to expand the region of interest beyond detected object (larger = more context)")
            if ch:
                v = max(0, new_pad)
                if v != cur_pad:
                    settings.set("live_tracker_roi_padding", v)
                    tr.roi_padding = v

            cur_int = settings.get("live_tracker_roi_update_interval")
            ch, new_int = imgui.input_int("ROI Update Interval (frames)##ROIIntervalTrackerMenu", cur_int)
            if imgui.is_item_hovered():
                imgui.set_tooltip("How often to run object detection (higher = better performance, lower = more responsive tracking)")
            if ch:
                v = max(1, new_int)
                if v != cur_int:
                    settings.set("live_tracker_roi_update_interval", v)
                    tr.roi_update_interval = v

            cur_sm = settings.get("live_tracker_roi_smoothing_factor")
            ch, new_sm = imgui.slider_float("ROI Smoothing Factor##ROISmoothTrackerMenu", cur_sm, 0.0, 1.0, "%.2f")
            if imgui.is_item_hovered():
                imgui.set_tooltip("Smooths ROI position changes between frames (0=instant changes, 1=maximum smoothing)")
            if ch and new_sm != cur_sm:
                settings.set("live_tracker_roi_smoothing_factor", new_sm)
                tr.roi_smoothing_factor = new_sm

            cur_persist = settings.get("live_tracker_roi_persistence_frames")
            ch, new_pf = imgui.input_int("ROI Persistence (frames)##ROIPersistTrackerMenu", cur_persist)
            if imgui.is_item_hovered():
                imgui.set_tooltip("How many frames to keep tracking after losing detection (0=stop immediately, higher=keep tracking longer)")
            if ch:
                v = max(0, new_pf)
                if v != cur_persist:
                    settings.set("live_tracker_roi_persistence_frames", v)
                    tr.max_frames_for_roi_persistence = v

        if imgui.collapsing_header("Optical Flow##ROIFlowTrackerMenu")[0]:
            cur_sparse = settings.get("live_tracker_use_sparse_flow")
            ch, new_sparse = imgui.checkbox("Use Sparse Optical Flow##ROISparseFlowTrackerMenu", cur_sparse)
            if ch:
                settings.set("live_tracker_use_sparse_flow", new_sparse)
                tr.use_sparse_flow = new_sparse

            imgui.text("DIS Dense Flow Settings:")
            with _DisabledScope(cur_sparse):
                presets = ["ULTRAFAST", "FAST", "MEDIUM"]
                cur_p = settings.get("live_tracker_dis_flow_preset").upper()
                try:
                    p_idx = presets.index(cur_p)
                except ValueError:
                    p_idx = 0
                ch, nidx = imgui.combo("DIS Preset##ROIDISPresetTrackerMenu", p_idx, presets)
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Optical flow quality preset (ULTRAFAST=best performance, MEDIUM=best quality)")
                if ch:
                    nv = presets[nidx]
                    if nv != cur_p:
                        settings.set("live_tracker_dis_flow_preset", nv)
                        tr.update_dis_flow_config(preset=nv)

                cur_scale = settings.get("live_tracker_dis_finest_scale")
                ch, new_scale = imgui.input_int("DIS Finest Scale (0-10, 0=auto)##ROIDISFineScaleTrackerMenu", cur_scale)
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Optical flow scale detail level (0=auto, lower=more detail but slower)")
                if ch and new_scale != cur_scale:
                    settings.set("live_tracker_dis_finest_scale", new_scale)
                    tr.update_dis_flow_config(finest_scale=new_scale)

            if imgui.collapsing_header("Output Signal Generation##ROISignalTrackerMenu")[0]:
                cur_sens = settings.get("live_tracker_sensitivity")
                ch, ns = imgui.slider_float("Output Sensitivity##ROISensTrackerMenu", cur_sens, 0.0, 100.0, "%.1f")
                if imgui.is_item_hovered():
                    imgui.set_tooltip("How responsive the output is to motion changes (higher = more sensitive to small movements)")
                if ch and ns != cur_sens:
                    settings.set("live_tracker_sensitivity", ns)
                    tr.sensitivity = ns

                cur_amp = settings.get("live_tracker_base_amplification")
                ch, na = imgui.slider_float("Base Amplification##ROIBaseAmpTrackerMenu", cur_amp, 0.1, 5.0, "%.2f")
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Multiplier for output range (higher = more movement, lower = gentler motion)")
                if ch:
                    v = max(0.1, na)
                    if v != cur_amp:
                        settings.set("live_tracker_base_amplification", v)
                        tr.base_amplification_factor = v

                imgui.text("Class-Specific Amplification Multipliers:")
                cur = settings.get("live_tracker_class_amp_multipliers", {})
                changed = False

                face = cur.get("face", 1.0)
                ch, nv = imgui.slider_float("Face Amp. Mult.##ROIFaceAmpTrackerMenu", face, 0.1, 5.0, "%.2f")
                if ch:
                    cur["face"] = max(0.1, nv)
                    changed = True

                hand = cur.get("hand", 1.0)
                ch, nv = imgui.slider_float("Hand Amp. Mult.##ROIHandAmpTrackerMenu", hand, 0.1, 5.0, "%.2f")
                if ch:
                    cur["hand"] = max(0.1, nv)
                    changed = True

                if changed:
                    settings.set("live_tracker_class_amp_multipliers", cur)
                    tr.class_specific_amplification_multipliers = cur

            cur_smooth = settings.get("live_tracker_flow_smoothing_window")
            ch, nv = imgui.input_int("Flow Smoothing Window##ROIFlowSmoothWinTrackerMenu", cur_smooth)
            if ch:
                v = max(1, nv)
                if v != cur_smooth:
                    settings.set("live_tracker_flow_smoothing_window", v)
                    tr.flow_history_window_smooth = v

            imgui.text("Output Delay (frames):")
            cur_delay = settings.get("funscript_output_delay_frames")
            ch, nd = imgui.slider_int("##OutputDelayFrames", cur_delay, 0, 20)
            if ch and nd != cur_delay:
                settings.set("funscript_output_delay_frames", nd)
                app.calibration.funscript_output_delay_frames = nd
                app.calibration.update_tracker_delay_params()

        imgui.unindent()

    def _render_tracking_axes_mode(self, stage_proc):
        """Renders UI elements for tracking axis mode."""
        axis_modes = ["Both Axes (Up/Down + Left/Right)", "Up/Down Only (Vertical)", "Left/Right Only (Horizontal)"]
        current_axis_mode_idx = 0
        if self.app.tracking_axis_mode == "vertical":
            current_axis_mode_idx = 1
        elif self.app.tracking_axis_mode == "horizontal":
            current_axis_mode_idx = 2

        processor = self.app.processor
        disable_axis_controls = (
            stage_proc.full_analysis_active
            or self.app.is_setting_user_roi_mode
            or (processor and processor.is_processing and not processor.pause_event.is_set())
        )
        with _DisabledScope(disable_axis_controls):
            imgui.set_next_item_width(-1)
            axis_mode_changed, new_axis_mode_idx = imgui.combo("##TrackingAxisModeComboGlobal", current_axis_mode_idx, axis_modes)
            if axis_mode_changed:
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
                    self.app.app_settings.set("tracking_axis_mode", self.app.tracking_axis_mode) # Auto-save
                    self.app.energy_saver.reset_activity_timer()

            if self.app.tracking_axis_mode != "both":
                imgui.text("Output Single Axis To:")
                output_targets = ["Timeline 1 (Primary)", "Timeline 2 (Secondary)"]
                current_output_target_idx = 1 if self.app.single_axis_output_target == "secondary" else 0

                imgui.set_next_item_width(-1)
                output_target_changed, new_output_target_idx = imgui.combo("##SingleAxisOutputComboGlobal", current_output_target_idx, output_targets)
                if output_target_changed:
                    old_target = self.app.single_axis_output_target
                    self.app.single_axis_output_target = "secondary" if new_output_target_idx == 1 else "primary"
                    if old_target != self.app.single_axis_output_target:
                        self.app.project_manager.project_dirty = True
                        self.app.logger.info(f"Single axis output target set to: {self.app.single_axis_output_target}", extra={'status_message': True})
                        self.app.app_settings.set("single_axis_output_target", self.app.single_axis_output_target) # Auto-save
                        self.app.energy_saver.reset_activity_timer()

    def _render_oscillation_detector_settings(self):
        app = self.app
        settings = app.app_settings

        imgui.text("Analysis Grid Size")
        _tooltip_if_hovered(
            "Finer grids (higher numbers) are more precise but use more CPU.\n"
            "8=Very Coarse\n"
            "20=Balanced\n"
            "40=Fine\n"
            "80=Very Fine"
        )

        cur_grid = settings.get("oscillation_detector_grid_size", 20)
        imgui.push_item_width(200)
        ch, nv = imgui.slider_int("##GridSize", cur_grid, 8, 80)
        if ch:
            valid = [8, 10, 16, 20, 32, 40, 64, 80]
            closest = min(valid, key=lambda x: abs(x - nv))
            if closest != cur_grid:
                settings.set("oscillation_detector_grid_size", closest)
                tr = app.tracker
                if tr:
                    tr.update_oscillation_grid_size()
        imgui.same_line()
        if imgui.button("Reset##ResetGridSize"):
            default_grid = 20
            if cur_grid != default_grid:
                settings.set("oscillation_detector_grid_size", default_grid)
                tr = app.tracker
                if tr:
                    tr.update_oscillation_grid_size()
        imgui.pop_item_width()

        imgui.text("Detection Sensitivity")
        _tooltip_if_hovered(
            "Adjusts how sensitive the oscillation detector is to motion.\n"
            "Lower values = less sensitive, Higher values = more sensitive"
        )

        cur_sens = settings.get("oscillation_detector_sensitivity", 1.0)
        imgui.push_item_width(200)
        ch, nv = imgui.slider_float("##Sensitivity", cur_sens, 0.1, 3.0, "%.2f")
        if ch and nv != cur_sens:
            settings.set("oscillation_detector_sensitivity", nv)
            tr = app.tracker
            if tr:
                tr.update_oscillation_sensitivity()
        imgui.same_line()
        if imgui.button("Reset##ResetSensitivity"):
            default_sens = 1.0
            if cur_sens != default_sens:
                settings.set("oscillation_detector_sensitivity", default_sens)
                tr = app.tracker
                if tr:
                    tr.update_oscillation_sensitivity()
        imgui.pop_item_width()

        imgui.text("Oscillation Area Selection")
        _tooltip_if_hovered("Select a specific area for oscillation detection instead of the full frame.")

        tr = app.tracker
        has_area = tr and tr.oscillation_area_fixed
        btn_count = 2 if has_area else 1
        avail_w = imgui.get_content_region_available_width()
        btn_w = (
            (avail_w - imgui.get_style().item_spacing.x * (btn_count - 1)) / btn_count
            if btn_count > 1
            else -1
        )

        set_text = "Cancel Set Oscillation Area" if app.is_setting_oscillation_area_mode else "Set Oscillation Area"
        # Set Oscillation Area button - PRIMARY when starting, DESTRUCTIVE when canceling
        if app.is_setting_oscillation_area_mode:
            with destructive_button_style():
                if imgui.button("%s##SetOscillationArea" % set_text, width=btn_w):
                    app.exit_set_oscillation_area_mode()
        else:
            with primary_button_style():
                if imgui.button("%s##SetOscillationArea" % set_text, width=btn_w):
                    app.enter_set_oscillation_area_mode()

        if has_area:
            imgui.same_line()
            # Clear Oscillation Area button (DESTRUCTIVE - clears user data)
            with destructive_button_style():
                if imgui.button("Clear Oscillation Area##ClearOscillationArea", width=btn_w):
                    tr.clear_oscillation_area_and_point()
                if hasattr(app, "is_setting_oscillation_area_mode"):
                    app.is_setting_oscillation_area_mode = False
                gi = getattr(app, "gui_instance", None)
                if gi and hasattr(gi, "video_display_ui"):
                    v = gi.video_display_ui
                    v.is_drawing_oscillation_area = False
                    v.drawn_oscillation_area_video_coords = None
                    v.waiting_for_oscillation_point_click = False
                    v.oscillation_area_draw_start_screen_pos = (0, 0)
                    v.oscillation_area_draw_current_screen_pos = (0, 0)
                app.logger.info("Oscillation area cleared.", extra={"status_message": True})
        # Overlays
        imgui.text("Overlays")
        _tooltip_if_hovered("Visualization layers for the Oscillation Detector.")
        cur_overlay = settings.get("oscillation_show_overlay", getattr(tr, "show_masks", False))
        ch, nv_overlay = imgui.checkbox("Show Oscillation Overlay##OscShowOverlay", cur_overlay)
        if ch and nv_overlay != cur_overlay:
            settings.set("oscillation_show_overlay", nv_overlay)
            if hasattr(tr, "show_masks"):
                tr.show_masks = nv_overlay
        # Default ROI rectangle to enabled on first launch (True)
        cur_roi_overlay = settings.get("oscillation_show_roi_overlay", True)
        has_osc_area = bool(tr and getattr(tr, "oscillation_area_fixed", None))
        with _DisabledScope(not has_osc_area):
            ch, nv_roi_overlay = imgui.checkbox("Show ROI Rectangle##OscShowROIOverlay", cur_roi_overlay)
        if has_osc_area and ch and nv_roi_overlay != cur_roi_overlay:
            settings.set("oscillation_show_roi_overlay", nv_roi_overlay)
            if hasattr(tr, "show_roi"):
                tr.show_roi = nv_roi_overlay
        # Static grid blocks toggle (processed-frame grid visualization)
        cur_grid_blocks = settings.get("oscillation_show_grid_blocks", False)
        ch, nv_grid_blocks = imgui.checkbox("Show Static Grid Blocks##OscShowGridBlocks", cur_grid_blocks)
        if ch and nv_grid_blocks != cur_grid_blocks:
            settings.set("oscillation_show_grid_blocks", nv_grid_blocks)
            if hasattr(tr, "show_grid_blocks"):
                tr.show_grid_blocks = nv_grid_blocks

        imgui.text("Live Signal Amplification")
        _tooltip_if_hovered("Stretches the live signal to use the full 0-100 range based on recent motion.")

        en = settings.get("live_oscillation_dynamic_amp_enabled", True)
        ch, nv = imgui.checkbox("Enable Dynamic Amplification##EnableLiveAmp", en)
        if ch and nv != en:
            settings.set("live_oscillation_dynamic_amp_enabled", nv)

        # Legacy improvements settings
        imgui.separator()
        imgui.text("Signal Processing Improvements")

        # Simple amplification mode
        cur_simple_amp = settings.get("oscillation_use_simple_amplification", False)
        ch, nv_simple = imgui.checkbox("Use Simple Amplification##UseSimpleAmp", cur_simple_amp)
        if ch and nv_simple != cur_simple_amp:
            settings.set("oscillation_use_simple_amplification", nv_simple)
        _tooltip_if_hovered("Use legacy-style fixed multipliers (dy*-10, dx*10) instead of dynamic scaling")

        # Decay mechanism
        cur_decay = settings.get("oscillation_enable_decay", True)
        ch, nv_decay = imgui.checkbox("Enable Decay Mechanism##EnableDecay", cur_decay)
        if ch and nv_decay != cur_decay:
            settings.set("oscillation_enable_decay", nv_decay)
        _tooltip_if_hovered("Gradually return to center when no motion is detected")

        if cur_decay:
            # Hold duration
            imgui.text("Hold Duration (ms)")
            cur_hold = settings.get("oscillation_hold_duration_ms", 250)
            imgui.push_item_width(150)
            ch, nv_hold = imgui.slider_int("##HoldDuration", cur_hold, 50, 1000)
            if ch and nv_hold != cur_hold:
                settings.set("oscillation_hold_duration_ms", nv_hold)
            imgui.pop_item_width()
            _tooltip_if_hovered("How long to hold position before starting decay")

            # Decay factor
            imgui.text("Decay Factor")
            cur_decay_factor = settings.get("oscillation_decay_factor", 0.95)
            imgui.push_item_width(150)
            ch, nv_decay_factor = imgui.slider_float("##DecayFactor", cur_decay_factor, 0.85, 0.99, "%.3f")
            if ch and nv_decay_factor != cur_decay_factor:
                settings.set("oscillation_decay_factor", nv_decay_factor)
            imgui.pop_item_width()
            _tooltip_if_hovered("How quickly to decay towards center (0.95 = slow, 0.85 = fast)")

        imgui.new_line()
        imgui.text_ansi_colored("Note: Detection Sensitivity and Dynamic\nAmplification are currently not yet working.", 0.25, 0.88, 0.82)

        # TODO: Move values to constants
        if settings.get("live_oscillation_dynamic_amp_enabled", True):
            imgui.text("Analysis Window (ms)")
            cur_ms = settings.get("live_oscillation_amp_window_ms", 4000)
            imgui.push_item_width(200)
            ch, nv = imgui.slider_int("##LiveAmpWindow", cur_ms, 1000, 10000)
            if ch and nv != cur_ms:
                settings.set("live_oscillation_amp_window_ms", nv)
            imgui.same_line()
            if imgui.button("Reset##ResetAmpWindow"):
                default_ms = 4000
                if cur_ms != default_ms:
                    settings.set("live_oscillation_amp_window_ms", default_ms)
            imgui.pop_item_width()

    def _render_stage3_oscillation_detector_mode_settings(self):
        """Render UI for selecting oscillation detector mode in Stage 3"""
        app = self.app
        settings = app.app_settings

        imgui.text("Stage 3 Oscillation Detector Mode")
        _tooltip_if_hovered(
            "Choose which oscillation detector algorithm to use in Stage 3:\n\n"
            "Current: Uses the experimental oscillation detector with\n"
            "  adaptive motion detection and dynamic scaling\n\n"
            "Legacy: Uses the legacy oscillation detector from commit f5ae40f\n"
            "  with fixed amplification and explicit decay mechanisms\n\n"
            "Hybrid: Combines benefits from both approaches (future feature)"
        )

        current_mode = settings.get("stage3_oscillation_detector_mode", "current")
        mode_options = ["current", "legacy", "hybrid"]
        mode_display = ["Current (Experimental)", "Legacy (f5ae40f)", "Hybrid (Coming Soon)"]

        try:
            current_idx = mode_options.index(current_mode)
        except ValueError:
            current_idx = 0

        imgui.push_item_width(200)

        # Disable hybrid for now
        with _DisabledScope(current_idx == 2):  # hybrid not implemented yet
            clicked, new_idx = imgui.combo("##Stage3ODMode", current_idx, mode_display)

        if clicked and new_idx != current_idx and new_idx != 2:  # Don't allow selecting hybrid
            new_mode = mode_options[new_idx]
            settings.set("stage3_oscillation_detector_mode", new_mode)
            app.logger.info(f"Stage 3 Oscillation Detector mode set to: {new_mode}", extra={"status_message": True})

        imgui.pop_item_width()

        # Show current selection info
        if current_mode == "current":
            imgui.text_ansi_colored("Using experimental oscillation detector", 0.0, 0.8, 0.0)
        elif current_mode == "legacy":
            imgui.text_ansi_colored("Using legacy oscillation detector (f5ae40f)", 0.0, 0.6, 0.8)
        else:
            imgui.text_ansi_colored("Hybrid mode (not yet implemented)", 0.8, 0.6, 0.0)

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
