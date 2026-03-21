"""Post-Processing tab UI mixin for ControlPanelUI."""
import imgui
from collections import OrderedDict
from application.utils import primary_button_style, destructive_button_style
from application.utils.imgui_helpers import DisabledScope as _DisabledScope, tooltip_if_hovered as _tooltip_if_hovered
from application.utils.section_card import section_card

# Category display order — General is a catch-all at the end
_CATEGORY_ORDER = ["Autotune", "Quickfix Tools", "Transform", "Smoothing", "Timing & Generation", "General"]


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

        # Get all plugins and group by category
        all_plugins = plugin_manager.get_available_plugins()
        if not all_plugins:
            imgui.text_disabled("No plugins available")
            return

        # Build category -> [plugin_name] mapping
        categorized = OrderedDict()
        for cat in _CATEGORY_ORDER:
            categorized[cat] = []

        for plugin_name in all_plugins:
            ctx = plugin_manager.plugin_contexts.get(plugin_name)
            if not ctx or not ctx.plugin_instance:
                continue
            cat = getattr(ctx.plugin_instance, 'category', 'General')
            if cat not in categorized:
                categorized[cat] = []
            categorized[cat].append(plugin_name)

        # Sort within each category (Ultimate Autotune stays first in Autotune)
        for cat, names in categorized.items():
            if cat == "Autotune":
                # Put Ultimate Autotune first
                ua = [n for n in names if 'ultimate' in n.lower() and 'autotune' in n.lower()]
                rest = sorted([n for n in names if n not in ua])
                categorized[cat] = ua + rest
            else:
                categorized[cat] = sorted(names)

        # Render each non-empty category
        for cat, plugin_names in categorized.items():
            if not plugin_names:
                continue

            open_default = (cat == "Autotune")
            with section_card(f"{cat}##PluginCat_{cat}", tier="primary",
                              open_by_default=open_default) as cat_open:
                if not cat_open:
                    continue

                def _render_plugins(names):
                    for pn in names:
                        ui_data = plugin_manager.get_plugin_ui_data(pn)
                        if ui_data and ui_data['available']:
                            self._render_plugin_section(pn, ui_data, plugin_manager, fs_proc, timeline_choice, scope_choice)

                if cat == "Quickfix Tools":
                    # Separate selection vs cursor tools
                    cur_tools = [pn for pn in plugin_names
                                 if (ctx := plugin_manager.plugin_contexts.get(pn))
                                 and ctx.plugin_instance and getattr(ctx.plugin_instance, 'requires_cursor', False)]
                    sel_tools = [pn for pn in plugin_names if pn not in cur_tools]
                    _render_plugins(sel_tools)
                    if cur_tools:
                        imgui.spacing()
                        imgui.separator()
                        imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 0.8, 0.3, 1.0)
                        imgui.text("Cursor-based (position playhead first)")
                        imgui.pop_style_color()
                        imgui.spacing()
                        _render_plugins(cur_tools)
                else:
                    _render_plugins(plugin_names)

    def _render_plugin_section(self, plugin_name, ui_data, plugin_manager, fs_proc, timeline_choice, scope_choice):
        """Render a collapsible section for a plugin with its parameters and apply button."""
        display_name = ui_data.get('display_name', plugin_name)
        description = ui_data.get('description', '')

        # Check if plugin requires cursor
        context_obj = plugin_manager.plugin_contexts.get(plugin_name)
        needs_cursor = (context_obj and context_obj.plugin_instance and
                        getattr(context_obj.plugin_instance, 'requires_cursor', False))

        # Collapsing header for this plugin (not section_card — we're already inside a category card)
        _plugin_open, _ = imgui.collapsing_header(f"{display_name}##Plugin_{plugin_name}", flags=0)
        if not _plugin_open:
            return
        imgui.indent(6)
        # Show cursor info label for cursor-dependent plugins
        if needs_cursor:
            imgui.push_style_color(imgui.COLOR_TEXT, 0.4, 0.8, 1.0, 1.0)
            imgui.text("Position your video playhead first")
            imgui.pop_style_color()
            imgui.spacing()
        if description:
            imgui.push_style_color(imgui.COLOR_TEXT, 0.7, 0.7, 0.7, 1.0)
            imgui.text_wrapped(description)
            imgui.pop_style_color()
            imgui.spacing()

        # Get plugin context and parameters
        context = plugin_manager.plugin_contexts.get(plugin_name)
        if not context:
            imgui.text_disabled("Plugin context not available")
            imgui.unindent(6)
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

                # Normalize type (schemas may use type objects or strings)
                _type_map = {float: 'float', 'float': 'float', int: 'int', 'int': 'int',
                             bool: 'bool', 'bool': 'bool', str: 'str', 'str': 'str', 'choice': 'choice'}
                norm_type = _type_map.get(param_type)
                uid = f"##PP_{plugin_name}_{param_name}"

                if norm_type == 'float':
                    if current_value is None:
                        current_value = default_value or 0.0
                    imgui.push_item_width(200)
                    _, new_value = imgui.slider_float(
                        f"{param_label}{uid}", float(current_value),
                        float(param_min), float(param_max), "%.2f")
                    imgui.pop_item_width()
                    params[param_name] = new_value
                elif norm_type == 'int':
                    if current_value is None:
                        current_value = default_value or 0
                    imgui.push_item_width(200)
                    _, new_value = imgui.slider_int(
                        f"{param_label}{uid}", int(current_value),
                        int(param_min), int(param_max))
                    imgui.pop_item_width()
                    params[param_name] = new_value
                elif norm_type == 'bool':
                    if current_value is None:
                        current_value = default_value or False
                    _, new_value = imgui.checkbox(f"{param_label}{uid}", bool(current_value))
                    params[param_name] = new_value
                elif norm_type in ('str', 'choice'):
                    choices = constraints.get('choices', [])
                    if choices:
                        try:
                            current_idx = choices.index(current_value) if current_value in choices else 0
                        except (ValueError, TypeError):
                            current_idx = 0
                        imgui.push_item_width(200)
                        _, new_idx = imgui.combo(f"{param_label}{uid}", current_idx, choices)
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

        imgui.unindent(6)
        imgui.spacing()

    def _apply_plugin(self, plugin_name, parameters, timeline_choice, scope_choice, fs_proc):
        """Apply a plugin with the given parameters."""
        try:
            # Determine axis from timeline choice index via axis assignments
            _default_axes = {0: "primary", 1: "secondary"}
            axis = _default_axes.get(timeline_choice)
            if axis is None:
                # T3+: resolve from funscript axis assignments (timeline_num = choice + 1)
                axis = "primary"  # fallback
                funscript_obj = fs_proc.get_funscript_obj() if hasattr(fs_proc, 'get_funscript_obj') else None
                if funscript_obj and hasattr(funscript_obj, '_axis_assignments'):
                    tl_num = timeline_choice + 1
                    axis = funscript_obj._axis_assignments.get(tl_num, "primary")

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

            # Auto-inject current_time_ms for cursor-dependent plugins
            plugin_manager = None
            if self.timeline_editor1 and hasattr(self.timeline_editor1, 'plugin_manager'):
                plugin_manager = self.timeline_editor1.plugin_manager
            if plugin_manager:
                ctx = plugin_manager.plugin_contexts.get(plugin_name)
                if ctx and ctx.plugin_instance and getattr(ctx.plugin_instance, 'requires_cursor', False):
                    app = self.app
                    fps = getattr(app, 'fps', None) or 30.0
                    frame_idx = getattr(app, 'current_frame_index', 0)
                    plugin_params['current_time_ms'] = int((frame_idx / fps) * 1000)

            funscript_obj.apply_plugin(plugin_name, **plugin_params)
            self.app.logger.info(f"Applied {plugin_name} to {axis}", extra={"status_message": True})

        except Exception as e:
            self.app.logger.error(f"Failed to apply plugin {plugin_name}: {e}", extra={"status_message": True})

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

