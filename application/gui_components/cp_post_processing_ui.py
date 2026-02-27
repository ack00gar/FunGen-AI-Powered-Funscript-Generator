"""Post-Processing tab UI mixin for ControlPanelUI."""
import imgui
from application.utils import primary_button_style, destructive_button_style
from application.utils.imgui_helpers import DisabledScope as _DisabledScope
from application.utils.section_card import section_card
from funscript.quality_validator import FunscriptQualityValidator, IssueSeverity


def _tooltip_if_hovered(text):
    if imgui.is_item_hovered():
        imgui.set_tooltip(text)


class PostProcessingMixin:
    """Mixin providing Post-Processing tab rendering methods."""

    def _render_post_processing_tab(self):
        app = self.app
        fs_proc = app.funscript_processor

        # Get plugin manager from timeline
        plugin_manager = None
        if self.timeline_editor1 and hasattr(self.timeline_editor1, 'plugin_manager'):
            plugin_manager = self.timeline_editor1.plugin_manager

        if not plugin_manager:
            imgui.text_disabled("Plugin system not initialized")
            return

        # Timeline and scope selection
        imgui.text("Apply to:")
        imgui.spacing()

        # Build dynamic timeline list from funscript object's axis assignments
        timeline_labels = ["Timeline 1 (stroke)", "Timeline 2 (roll)"]
        funscript_obj = None
        if hasattr(app, 'multi_axis_funscript'):
            funscript_obj = app.multi_axis_funscript
        if not funscript_obj and hasattr(fs_proc, 'get_funscript_obj'):
            funscript_obj = fs_proc.get_funscript_obj()
        if funscript_obj and hasattr(funscript_obj, '_axis_assignments'):
            # Override T1/T2 defaults from actual assignments
            for tl_num in (1, 2):
                axis_name = funscript_obj._axis_assignments.get(tl_num)
                if axis_name:
                    timeline_labels[tl_num - 1] = f"Timeline {tl_num} ({axis_name})"
            # Add T3+ from assignments
            for tl_num in sorted(funscript_obj._axis_assignments.keys()):
                if tl_num >= 3:
                    axis_name = funscript_obj._axis_assignments[tl_num]
                    timeline_labels.append(f"Timeline {tl_num} ({axis_name})")

        # Timeline selection
        timeline_choice = getattr(self, '_pp_timeline_choice', 0)
        if timeline_choice >= len(timeline_labels):
            timeline_choice = 0
        imgui.push_item_width(200)
        _, timeline_choice = imgui.combo("Timeline##PostProcTimeline", timeline_choice, timeline_labels)
        self._pp_timeline_choice = timeline_choice
        imgui.pop_item_width()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Select which timeline to apply processing to")

        # Scope selection
        scope_choice = getattr(self, '_pp_scope_choice', 0)
        imgui.push_item_width(200)
        _, scope_choice = imgui.combo("Scope##PostProcScope", scope_choice, ["Full Script", "Selection Only"])
        self._pp_scope_choice = scope_choice
        imgui.pop_item_width()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Apply to entire script or selected points only")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Get all plugins
        all_plugins = plugin_manager.get_available_plugins()
        if not all_plugins:
            imgui.text_disabled("No plugins available")
            return

        # Prioritize Ultimate Autotune plugin first
        ultimate_autotune = None
        other_plugins = []
        for plugin_name in all_plugins:
            if 'ultimate' in plugin_name.lower() and 'autotune' in plugin_name.lower():
                ultimate_autotune = plugin_name
            else:
                other_plugins.append(plugin_name)

        # Sort other plugins alphabetically
        other_plugins.sort()

        # Render plugins in order: Ultimate Autotune first, then others
        plugins_to_render = ([ultimate_autotune] if ultimate_autotune else []) + other_plugins

        for plugin_name in plugins_to_render:
            ui_data = plugin_manager.get_plugin_ui_data(plugin_name)
            if not ui_data or not ui_data['available']:
                continue

            # Render plugin section
            self._render_plugin_section(plugin_name, ui_data, plugin_manager, fs_proc, timeline_choice, scope_choice)

        # Script Quality Validator section
        self._render_quality_validator_section(timeline_choice)

    def _render_quality_validator_section(self, timeline_choice):
        """Render the Script Quality validator card (Phase 1.4)."""
        with section_card("Script Quality##QualityValidator", tier="secondary",
                          open_by_default=False) as _open:
            if not _open:
                return

            imgui.push_style_color(imgui.COLOR_TEXT, 0.7, 0.7, 0.7, 1.0)
            imgui.text_wrapped("Check your script for speed limit violations, dead zones, and other issues.")
            imgui.pop_style_color()
            imgui.spacing()

            # Run Quality Check button
            if imgui.button("Run Quality Check##QV_Run"):
                self._run_quality_check(timeline_choice)

            # Display results if available
            report = getattr(self, '_quality_report', None)
            if report:
                imgui.spacing()

                # Score bar
                score = report.score
                if score >= 80:
                    score_color = (0.2, 0.8, 0.2, 1.0)
                elif score >= 50:
                    score_color = (0.9, 0.7, 0.1, 1.0)
                else:
                    score_color = (0.9, 0.2, 0.2, 1.0)

                imgui.push_style_color(imgui.COLOR_TEXT, *score_color)
                imgui.text(f"Quality Score: {score}/100")
                imgui.pop_style_color()

                # Progress bar
                imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, *score_color)
                imgui.progress_bar(score / 100.0, (-1, 0), f"{score}%")
                imgui.pop_style_color()

                # Stats
                stats = report.stats
                if stats:
                    imgui.text(f"Actions: {stats.get('action_count', 0)}")
                    imgui.same_line()
                    imgui.text(f"Avg speed: {stats.get('avg_speed', 0):.0f} u/s")
                    imgui.same_line()
                    imgui.text(f"Max: {stats.get('max_speed', 0):.0f} u/s")

                imgui.spacing()

                # Issue list
                if report.issues:
                    imgui.text(f"Issues: {report.error_count} errors, {report.warning_count} warnings, {report.info_count} info")
                    imgui.spacing()

                    # Scrollable issue list
                    if imgui.begin_child("##QualityIssues", 0, 150, border=True):
                        for issue in report.issues:
                            if issue.severity == IssueSeverity.ERROR:
                                icon, color = "[E]", (0.9, 0.2, 0.2, 1.0)
                            elif issue.severity == IssueSeverity.WARNING:
                                icon, color = "[W]", (0.9, 0.7, 0.1, 1.0)
                            else:
                                icon, color = "[i]", (0.5, 0.5, 0.5, 1.0)

                            imgui.push_style_color(imgui.COLOR_TEXT, *color)
                            imgui.text(icon)
                            imgui.pop_style_color()
                            imgui.same_line()
                            imgui.text_wrapped(issue.message)
                    imgui.end_child()
                else:
                    imgui.text_colored("No issues found!", 0.2, 0.8, 0.2, 1.0)

    def _run_quality_check(self, timeline_choice):
        """Execute the quality validator on the selected timeline."""
        try:
            app = self.app
            fs_proc = app.funscript_processor

            # Get actions for the selected timeline
            if timeline_choice == 0:
                axis = "primary"
            elif timeline_choice == 1:
                axis = "secondary"
            else:
                axis = "primary"

            funscript_obj = None
            if hasattr(fs_proc, 'get_funscript_obj'):
                funscript_obj = fs_proc.get_funscript_obj()
            if not funscript_obj:
                return

            actions = funscript_obj.get_axis_actions(axis)
            if not actions:
                return

            # Get video duration
            duration_ms = 0
            if app.processor and app.processor.video_info:
                fps = app.processor.fps
                total_frames = app.processor.video_info.get('total_frames', 0)
                if fps > 0 and total_frames > 0:
                    duration_ms = (total_frames / fps) * 1000.0

            validator = FunscriptQualityValidator()
            self._quality_report = validator.validate(actions, duration_ms)

        except Exception as e:
            if hasattr(self.app, 'logger'):
                self.app.logger.error(f"Quality check failed: {e}")

    def _render_plugin_section(self, plugin_name, ui_data, plugin_manager, fs_proc, timeline_choice, scope_choice):
        """Render a collapsible section for a plugin with its parameters and apply button."""
        display_name = ui_data.get('display_name', plugin_name)
        description = ui_data.get('description', '')

        # Section card for this plugin (collapsed by default)
        with section_card(f"{display_name}##Plugin_{plugin_name}", tier="secondary",
                          open_by_default=False) as _plugin_open:
            if not _plugin_open:
                return
            if description:
                imgui.push_style_color(imgui.COLOR_TEXT, 0.7, 0.7, 0.7, 1.0)
                imgui.text_wrapped(description)
                imgui.pop_style_color()
                imgui.spacing()

            # Get plugin context and parameters
            context = plugin_manager.plugin_contexts.get(plugin_name)
            if not context:
                imgui.text_disabled("Plugin context not available")
                return

            plugin_instance = context.plugin_instance
            if not plugin_instance or not hasattr(plugin_instance, 'parameters_schema'):
                imgui.text_disabled("No parameters available")
            else:
                # Render parameters
                schema = plugin_instance.parameters_schema
                params = context.parameters

                for param_name, param_info in schema.items():
                    param_type = param_info.get('type')
                    param_label = param_info.get('label', param_name)
                    param_desc = param_info.get('description', '')

                    # Get constraints from schema
                    constraints = param_info.get('constraints', {})
                    param_min = constraints.get('min', 0)
                    param_max = constraints.get('max', 100)
                    default_value = param_info.get('default')

                    current_value = params.get(param_name, default_value)

                    # Skip internal parameters that shouldn't be shown in UI
                    if param_name in ['start_time_ms', 'end_time_ms', 'selected_indices']:
                        continue

                    # Render different UI elements based on parameter type
                    # Compare against Python type objects, not strings
                    if param_type == float or param_type == 'float':
                        if current_value is None:
                            current_value = default_value if default_value is not None else 0.0
                        imgui.push_item_width(200)
                        _, new_value = imgui.slider_float(
                            f"{param_label}##PP_{plugin_name}_{param_name}",
                            float(current_value),
                            float(param_min),
                            float(param_max),
                            "%.2f"
                        )
                        imgui.pop_item_width()
                        params[param_name] = new_value
                    elif param_type == int or param_type == 'int':
                        if current_value is None:
                            current_value = default_value if default_value is not None else 0
                        imgui.push_item_width(200)
                        _, new_value = imgui.slider_int(
                            f"{param_label}##PP_{plugin_name}_{param_name}",
                            int(current_value),
                            int(param_min),
                            int(param_max)
                        )
                        imgui.pop_item_width()
                        params[param_name] = new_value
                    elif param_type == bool or param_type == 'bool':
                        if current_value is None:
                            current_value = default_value if default_value is not None else False
                        _, new_value = imgui.checkbox(
                            f"{param_label}##PP_{plugin_name}_{param_name}",
                            bool(current_value)
                        )
                        params[param_name] = new_value
                    elif param_type == str or param_type == 'str' or param_type == 'choice':
                        choices = constraints.get('choices', [])
                        if choices:
                            try:
                                current_idx = choices.index(current_value) if current_value in choices else 0
                            except (ValueError, TypeError):
                                current_idx = 0
                            imgui.push_item_width(200)
                            _, new_idx = imgui.combo(
                                f"{param_label}##PP_{plugin_name}_{param_name}",
                                current_idx,
                                choices
                            )
                            imgui.pop_item_width()
                            params[param_name] = choices[new_idx]

                    if param_desc and imgui.is_item_hovered():
                        imgui.set_tooltip(param_desc)

            imgui.spacing()

            # Reset to default button
            if imgui.button(f"Reset to Default##PP_{plugin_name}_Reset"):
                default_params = plugin_manager._get_default_parameters(plugin_instance)
                context.parameters = default_params.copy()

            # Apply button (PRIMARY styling)
            imgui.same_line()
            with primary_button_style():
                if imgui.button(f"Apply##PP_{plugin_name}_Apply"):
                    self._apply_plugin(plugin_name, context.parameters, timeline_choice, scope_choice, fs_proc)

            imgui.spacing()

    def _apply_plugin(self, plugin_name, parameters, timeline_choice, scope_choice, fs_proc):
        """Apply a plugin with the given parameters."""
        try:
            # Determine axis from timeline choice index
            if timeline_choice == 0:
                axis = "primary"
            elif timeline_choice == 1:
                axis = "secondary"
            else:
                # T3+: resolve axis name from funscript assignments
                axis = "primary"  # fallback
                funscript_obj = fs_proc.get_funscript_obj() if hasattr(fs_proc, 'get_funscript_obj') else None
                if funscript_obj and hasattr(funscript_obj, 'additional_axes'):
                    sorted_axes = sorted(funscript_obj.additional_axes.keys())
                    extra_idx = timeline_choice - 2
                    if 0 <= extra_idx < len(sorted_axes):
                        axis = sorted_axes[extra_idx]

            # Get the funscript object
            funscript_obj = fs_proc.get_funscript_obj()
            if not funscript_obj:
                self.app.logger.warning("No funscript loaded", extra={"status_message": True})
                return

            # Determine selection scope
            selected_indices = None
            if scope_choice == 1:  # Selection Only
                # Get selected indices from the appropriate timeline
                timeline = self.timeline_editor1 if timeline_choice == 0 else self.timeline_editor2
                if timeline and hasattr(timeline, 'multi_selected_action_indices'):
                    selected_indices = timeline.multi_selected_action_indices.copy() if timeline.multi_selected_action_indices else None

            # Apply the plugin
            plugin_params = parameters.copy()
            plugin_params['axis'] = axis
            if selected_indices:
                plugin_params['selected_indices'] = selected_indices

            funscript_obj.apply_plugin(plugin_name, **plugin_params)
            self.app.logger.info(f"Applied {plugin_name} to {axis}", extra={"status_message": True})

        except Exception as e:
            self.app.logger.error(f"Failed to apply plugin {plugin_name}: {e}", extra={"status_message": True})

    def _render_post_processing_profile_row(self, long_name, profile_params, config_copy):
        changed_cfg = False
        imgui.push_id("profile_%s" % long_name)
        is_open = imgui.tree_node(long_name)

        if is_open:
            imgui.columns(2, "profile_settings", border=False)

            imgui.text("Amplification")

            imgui.text("Scale")
            imgui.next_column()
            imgui.push_item_width(-1)
            val = profile_params.get("scale_factor", 1.0)
            ch, nv = imgui.slider_float("##scale", val, 0.1, 5.0, "%.2f")
            if ch:
                profile_params["scale_factor"] = nv
                changed_cfg = True
            imgui.pop_item_width()
            imgui.next_column()

            imgui.text("Center")
            imgui.next_column()
            imgui.push_item_width(-1)
            val = profile_params.get("center_value", 50)
            ch, nv = imgui.slider_int("##amp_center", val, 0, 100)
            if ch:
                profile_params["center_value"] = nv
                changed_cfg = True
            imgui.pop_item_width()
            imgui.next_column()

            clamp_low = profile_params.get("clamp_lower", 10)
            clamp_high = profile_params.get("clamp_upper", 90)

            imgui.text("Clamp Low")
            imgui.next_column()
            imgui.push_item_width(-1)
            ch_l, nv_l = imgui.slider_int("##clamp_low", clamp_low, 0, 100)
            if ch_l:
                clamp_low = min(nv_l, clamp_high)
                profile_params["clamp_lower"] = clamp_low
                changed_cfg = True
            imgui.pop_item_width()
            imgui.next_column()

            imgui.text("Clamp High")
            imgui.next_column()
            imgui.push_item_width(-1)
            ch_h, nv_h = imgui.slider_int("##clamp_high", clamp_high, 0, 100)
            if ch_h:
                clamp_high = max(nv_h, clamp_low)
                profile_params["clamp_upper"] = clamp_high
                changed_cfg = True
            imgui.pop_item_width()
            imgui.next_column()

            imgui.columns(1)
            imgui.spacing()
            imgui.columns(2, "profile_settings_2", border=False)

            imgui.text("Smoothing (SG Filter)")

            imgui.text("Window")
            imgui.next_column()
            imgui.push_item_width(-1)
            sg_win = profile_params.get("sg_window", 7)
            ch, nv = imgui.slider_int("##sg_win", sg_win, 3, 99)
            if ch:
                nv = max(3, nv + 1 if nv % 2 == 0 else nv)
                if nv != sg_win:
                    profile_params["sg_window"] = nv
                    changed_cfg = True
            imgui.pop_item_width()
            imgui.next_column()

            imgui.text("Polyorder")
            imgui.next_column()
            imgui.push_item_width(-1)
            sg_poly = profile_params.get("sg_polyorder", 3)
            max_poly = max(1, profile_params.get("sg_window", 7) - 1)
            cur_poly = min(sg_poly, max_poly)
            ch, nv = imgui.slider_int("##sg_poly", cur_poly, 1, max_poly)
            if ch and nv != sg_poly:
                profile_params["sg_polyorder"] = nv
                changed_cfg = True
            imgui.pop_item_width()
            imgui.next_column()

            imgui.text("Simplification (RDP)")

            imgui.text("Epsilon")
            imgui.next_column()
            imgui.push_item_width(-1)
            rdp_eps = profile_params.get("rdp_epsilon", 1.0)
            ch, nv = imgui.slider_float("##rdp_eps", rdp_eps, 0.1, 20.0, "%.2f")
            if ch and nv != rdp_eps:
                profile_params["rdp_epsilon"] = nv
                changed_cfg = True
            imgui.pop_item_width()
            imgui.next_column()

            out_min = profile_params.get("output_min", 0)
            out_max = profile_params.get("output_max", 100)

            imgui.text("Output Min")
            imgui.next_column()
            imgui.push_item_width(-1)
            ch, nv = imgui.slider_int("##out_min", out_min, 0, 100)
            if ch:
                out_min = min(nv, out_max)
                profile_params["output_min"] = out_min
                changed_cfg = True
            imgui.pop_item_width()
            imgui.next_column()

            imgui.text("Output Max")
            imgui.next_column()
            imgui.push_item_width(-1)
            ch, nv = imgui.slider_int("##out_max", out_max, 0, 100)
            if ch:
                out_max = max(nv, out_min)
                profile_params["output_max"] = out_max
                changed_cfg = True
            imgui.pop_item_width()
            imgui.next_column()

            imgui.columns(1)
            imgui.tree_pop()

        if changed_cfg:
            config_copy[long_name] = profile_params
        imgui.pop_id()
        return changed_cfg

    def _render_automatic_post_processing_new(self, fs_proc):
        app = self.app
        sp = app.stage_processor
        proc = app.processor

        proc_tools_disabled = sp.full_analysis_active or (proc and proc.is_processing) or app.is_setting_user_roi_mode
        with _DisabledScope(proc_tools_disabled):
            enabled = app.app_settings.get("enable_auto_post_processing", False)
            ch, nv = imgui.checkbox("Enable Automatic Post-Processing on Completion", enabled)
            if ch and nv != enabled:
                app.app_settings.set("enable_auto_post_processing", nv)
                app.project_manager.project_dirty = True
                app.logger.info("Automatic post-processing on completion %s." % ("enabled" if nv else "disabled"), extra={"status_message": True})
            _tooltip_if_hovered("If checked, the profiles below will be applied automatically\nafter an offline analysis or live tracking session finishes.")

            # Run post-processing button (PRIMARY - positive action)
            with primary_button_style():
                if imgui.button("Run Post-Processing Now##RunAutoPostProcessButton", width=-1):
                    if hasattr(fs_proc, "apply_automatic_post_processing"):
                        fs_proc.apply_automatic_post_processing()

            use_chapter = app.app_settings.get("auto_processing_use_chapter_profiles", True)
            ch, nv = imgui.checkbox("Apply Per-Chapter Settings (if available)", use_chapter)
            if ch and nv != use_chapter:
                app.app_settings.set("auto_processing_use_chapter_profiles", nv)
            _tooltip_if_hovered("If checked, applies specific profiles below to each chapter.\nIf unchecked, applies only the 'Default' profile to the entire script.")

            config = app.app_settings.get("auto_post_processing_amplification_config", {})
            config_copy = config.copy()
            master_changed = False

            if app.app_settings.get("auto_processing_use_chapter_profiles", True):
                imgui.text("Per-Position Processing Profiles")
                all_pos = ["Default"] + sorted(
                    list({info["long_name"] for info in self.constants.POSITION_INFO_MAPPING.values()})
                )
                default_profile = self.constants.DEFAULT_AUTO_POST_AMP_CONFIG.get("Default", {})
                for name in all_pos:
                    if not name:
                        continue
                    params = config_copy.get(name, default_profile).copy()
                    if self._render_post_processing_profile_row(name, params, config_copy):
                        master_changed = True
            else:
                imgui.text("Default Processing Profile (applies to all)")
                name = "Default"
                default_profile = self.constants.DEFAULT_AUTO_POST_AMP_CONFIG.get(name, {})
                params = config_copy.get(name, default_profile).copy()
                if self._render_post_processing_profile_row(name, params, config_copy):
                    master_changed = True

            if master_changed:
                app.app_settings.set("auto_post_processing_amplification_config", config_copy)
                app.project_manager.project_dirty = True

            # Reset All Profiles button (DESTRUCTIVE - resets to defaults)
            with destructive_button_style():
                if imgui.button("Reset All Profiles to Defaults##ResetAutoPostProcessing", width=-1):
                    app.app_settings.set(
                        "auto_post_processing_amplification_config",
                        self.constants.DEFAULT_AUTO_POST_AMP_CONFIG,
                    )
                    app.project_manager.project_dirty = True
                    app.logger.info("All post-processing profiles reset to defaults.", extra={"status_message": True})

            imgui.text("Final Smoothing Pass")
            en = app.app_settings.get("auto_post_proc_final_rdp_enabled", False)
            ch, nv = imgui.checkbox("Run Final RDP Pass to Seam Chapters", en)
            if ch and nv != en:
                app.app_settings.set("auto_post_proc_final_rdp_enabled", nv)
                app.project_manager.project_dirty = True
            _tooltip_if_hovered(
                "After all other processing, run one final simplification pass\n"
                "on the entire script. This can help smooth out the joints\n"
                "between chapters that used different processing settings."
            )

            if app.app_settings.get("auto_post_proc_final_rdp_enabled", False):
                imgui.same_line()
                imgui.push_item_width(120)
                cur_eps = app.app_settings.get("auto_post_proc_final_rdp_epsilon", 10.0)
                ch, nv = imgui.slider_float("Epsilon##FinalRDPEpsilon", cur_eps, 0.1, 20.0, "%.2f")
                if ch and nv != cur_eps:
                    app.app_settings.set("auto_post_proc_final_rdp_epsilon", nv)
                    app.project_manager.project_dirty = True
                imgui.pop_item_width()

        # Disabled tooltip
        if proc_tools_disabled and imgui.is_item_hovered():
            imgui.set_tooltip("Disabled while another process is active.")

    # ------- Calibration -------

    def _render_latency_calibration(self, calibration_mgr):
        col = self.ControlPanelColors.STATUS_WARNING
        imgui.text_ansi_colored("--- LATENCY CALIBRATION MODE ---", *col)
        if not calibration_mgr.calibration_reference_point_selected:
            imgui.text_wrapped("1. Start the live tracker for 10s of action then pause it.")
            imgui.text_wrapped("   Select a clear action point on Timeline 1.")
        else:
            imgui.text_wrapped("1. Point at %.0fms selected." % calibration_mgr.calibration_timeline_point_ms)
            imgui.text_wrapped("2. Now, use video controls (seek, frame step) to find the")
            imgui.text_wrapped("   EXACT visual moment corresponding to the selected point.")
            imgui.text_wrapped("3. Press 'Confirm Visual Match' below.")
        # Confirm Visual Match button (PRIMARY - positive action)
        with primary_button_style():
            if imgui.button("Confirm Visual Match##ConfirmCalibration", width=-1):
                if calibration_mgr.calibration_reference_point_selected:
                    calibration_mgr.confirm_latency_calibration()
                else:
                    self.app.logger.info("Please select a reference point on Timeline 1 first.", extra={"status_message": True})
        _tooltip_if_hovered("Confirm the current video frame matches the selected timeline point.\nThis calculates and applies the latency offset.")
        # Cancel Calibration button (DESTRUCTIVE - cancels process)
        with destructive_button_style():
            if imgui.button("Cancel Calibration##CancelCalibration", width=-1):
                calibration_mgr.is_calibration_mode_active = False
                calibration_mgr.calibration_reference_point_selected = False
                self.app.logger.info("Latency calibration cancelled.", extra={"status_message": True})
                self.app.energy_saver.reset_activity_timer()
        _tooltip_if_hovered("Exit calibration mode without applying changes.")

    # ------- Range selection -------

    def _render_range_selection(self, stage_proc, fs_proc, event_handlers):
        app = self.app
        disabled = stage_proc.full_analysis_active or (app.processor and app.processor.is_processing) or app.is_setting_user_roi_mode

        with _DisabledScope(disabled):
            ch, new_active = imgui.checkbox("Enable Range Processing", fs_proc.scripting_range_active)
            if ch:
                event_handlers.handle_scripting_range_active_toggle(new_active)
            _tooltip_if_hovered(
                "Restrict processing to a specific frame range or chapter.\n"
                "Enable the checkbox and set frames, or select a chapter."
            )

            if fs_proc.scripting_range_active:
                imgui.text("Set Frames Range Manually (-1 = End):")
                imgui.push_item_width(imgui.get_content_region_available()[0] * 0.4)
                ch, nv = imgui.input_int(
                    "Start##SR_InputStart",
                    fs_proc.scripting_start_frame,
                    flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE,
                )
                if ch:
                    event_handlers.handle_scripting_start_frame_input(nv)
                imgui.same_line()
                imgui.text(" ")
                imgui.same_line()
                ch, nv = imgui.input_int(
                    "End (-1)##SR_InputEnd",
                    fs_proc.scripting_end_frame,
                    flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE,
                )
                if ch:
                    event_handlers.handle_scripting_end_frame_input(nv)
                imgui.pop_item_width()

                start_disp, end_disp = fs_proc.get_scripting_range_display_text()
                imgui.text("Active Range: Frames: %s to %s" % (start_disp, end_disp))
                sel_ch = fs_proc.selected_chapter_for_scripting
                if sel_ch:
                    imgui.text("Chapter: %s (%s)" % (sel_ch.class_name, sel_ch.segment_type))
                if imgui.button("Clear Range Selection##ClearRangeButton"):
                    event_handlers.clear_scripting_range_selection()
                _tooltip_if_hovered("Reset frame range and deselect chapter.")
        if disabled and imgui.is_item_hovered():
            imgui.set_tooltip("Disabled while another process is active.")



    # ------- Post-processing manual tools -------

    def _render_funscript_processing_tools(self, fs_proc, event_handlers):
        app = self.app
        sp = app.stage_processor
        proc = app.processor
        disabled = sp.full_analysis_active or (proc and proc.is_processing) or app.is_setting_user_roi_mode

        with _DisabledScope(disabled):
            axis_opts = ["Primary Axis", "Secondary Axis"]
            cur_idx = 0 if fs_proc.selected_axis_for_processing == "primary" else 1
            ch, nidx = imgui.combo("Target Axis##ProcAxis", cur_idx, axis_opts)
            if ch and nidx != cur_idx:
                event_handlers.set_selected_axis_for_processing("primary" if nidx == 0 else "secondary")
            # imgui.separator()

            imgui.text("Apply To:")
            range_label = fs_proc.get_operation_target_range_label()
            if imgui.radio_button(
                "%s##OpTargetRange" % range_label,
                fs_proc.operation_target_mode == "apply_to_scripting_range",
            ):
                fs_proc.operation_target_mode = "apply_to_scripting_range"
            imgui.same_line()
            if imgui.radio_button(
                "Selected Points##OpTargetSelect",
                fs_proc.operation_target_mode == "apply_to_selected_points",
            ):
                fs_proc.operation_target_mode = "apply_to_selected_points"

            def prep_op():
                if fs_proc.operation_target_mode == "apply_to_selected_points":
                    editor = (
                        self.timeline_editor1
                        if fs_proc.selected_axis_for_processing == "primary"
                        else self.timeline_editor2
                    )
                    fs_proc.current_selection_indices = list(
                        editor.multi_selected_action_indices
                    ) if editor else []
                    if not fs_proc.current_selection_indices:
                        app.logger.info("No points selected for operation.", extra={"status_message": True})

            imgui.text("Points operations")
            if imgui.button("Clamp to 0##Clamp0"):
                prep_op()
                fs_proc.handle_funscript_operation("clamp_0")
            imgui.same_line()
            if imgui.button("Clamp to 100##Clamp100"):
                prep_op()
                fs_proc.handle_funscript_operation("clamp_100")
            imgui.same_line()
            if imgui.button("Invert##InvertPoints"):
                prep_op()
                fs_proc.handle_funscript_operation("invert")
            imgui.same_line()
            # Clear button (DESTRUCTIVE - deletes all points)
            with destructive_button_style():
                if imgui.button("Clear##ClearPoints"):
                    prep_op()
                    fs_proc.handle_funscript_operation("clear")

            imgui.text("Amplify Values")
            ch, nv = imgui.slider_float("Factor##AmplifyFactor", fs_proc.amplify_factor_input, 0.1, 3.0, "%.2f")
            if ch:
                fs_proc.amplify_factor_input = nv
            ch, nv = imgui.slider_int("Center##AmplifyCenter", fs_proc.amplify_center_input, 0, 100)
            if ch:
                fs_proc.amplify_center_input = nv
            # Apply button (PRIMARY - positive action)
            with primary_button_style():
                if imgui.button("Apply Amplify##ApplyAmplify"):
                    prep_op()
                    fs_proc.handle_funscript_operation("amplify")

            imgui.text("Savitzky-Golay Filter")
            ch, nv = imgui.slider_int("Window Length##SGWin", fs_proc.sg_window_length_input, 3, 99)
            if ch:
                event_handlers.update_sg_window_length(nv)
            max_po = max(1, fs_proc.sg_window_length_input - 1)
            po_val = min(fs_proc.sg_polyorder_input, max_po)
            ch, nv = imgui.slider_int("Polyorder##SGPoly", po_val, 1, max_po)
            if ch:
                fs_proc.sg_polyorder_input = nv
            # Apply button (PRIMARY - positive action)
            with primary_button_style():
                if imgui.button("Apply Savitzky-Golay##ApplySG"):
                    prep_op()
                    fs_proc.handle_funscript_operation("apply_sg")

            imgui.text("RDP Simplification")
            ch, nv = imgui.slider_float("Epsilon##RDPEps", fs_proc.rdp_epsilon_input, 0.01, 20.0, "%.2f")
            if imgui.is_item_hovered():
                imgui.set_tooltip("Curve simplification strength (lower = more detail, higher = smoother/fewer points)")
            if ch:
                fs_proc.rdp_epsilon_input = nv
            # Apply button (PRIMARY - positive action)
            with primary_button_style():
                if imgui.button("Apply RDP##ApplyRDP"):
                    prep_op()
                    fs_proc.handle_funscript_operation("apply_rdp")

            imgui.text("Dynamic Amplification")
            if not hasattr(fs_proc, "dynamic_amp_window_ms_input"):
                fs_proc.dynamic_amp_window_ms_input = 4000
            ch, nv = imgui.slider_int("Window (ms)##DynAmpWin", fs_proc.dynamic_amp_window_ms_input, 500, 10000)
            if ch:
                fs_proc.dynamic_amp_window_ms_input = nv
            _tooltip_if_hovered("The size of the 'before/after' window in milliseconds to consider for amplification.")

            # Apply button (PRIMARY - positive action)
            with primary_button_style():
                if imgui.button("Apply Dynamic Amplify##ApplyDynAmp"):
                    prep_op()
                    fs_proc.handle_funscript_operation("apply_dynamic_amp")

        # If disabled, show a tooltip on hover (outside the disabled scope)
        if disabled and imgui.is_item_hovered():
            imgui.set_tooltip("Disabled while another process is active.")
