"""
Plugin Pipeline UI — ImGui window for building and running ordered plugin chains.

Provides:
- Add plugin dropdown (categorized)
- Ordered step list with enable/disable, remove, reorder
- Per-step parameter editing (collapsible)
- Preset load/save
- Preview and Apply buttons
"""

import copy
import logging
from collections import OrderedDict
from typing import Dict, Any, Optional, Tuple

try:
    import imgui
except ImportError:
    imgui = None

from application.classes.plugin_pipeline import PluginPipeline, timeline_label_to_axis

_CATEGORY_ORDER = ["Autotune", "Quickfix Tools", "Transform", "Smoothing", "Timing & Generation", "General"]


class PluginPipelineUI:
    """ImGui window that renders and manages a PluginPipeline."""

    def __init__(self, app_instance, logger: Optional[logging.Logger] = None):
        self.app = app_instance
        self.logger = logger or logging.getLogger('PluginPipelineUI')
        self.pipeline = PluginPipeline(app_instance, logger=self.logger)
        self._save_name_buf = ""
        self._last_errors: list = []
        self._previewing = False

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def render(self):
        """Render the pipeline window. Called each frame."""
        app_state = self.app.app_state_ui
        if not getattr(app_state, 'show_plugin_pipeline', False):
            return

        imgui.set_next_window_size(460, 420, condition=imgui.ONCE)
        visible, opened = imgui.begin("Plugin Pipeline", closable=True)

        if getattr(app_state, 'show_plugin_pipeline') != opened:
            app_state.show_plugin_pipeline = opened

        if visible:
            self._render_content()

        imgui.end()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_plugin_manager(self):
        """Get the PluginUIManager from the first timeline editor."""
        gui = getattr(self.app, 'gui_instance', None)
        if gui and hasattr(gui, 'timeline_editor1'):
            return gui.timeline_editor1.plugin_renderer.plugin_manager
        return None

    def _render_content(self):
        # Top bar: add plugin | save/load presets
        self._render_top_bar()

        imgui.separator()

        # Step list (scrollable middle area)
        self._render_step_list()

        imgui.separator()

        # Bottom: preview / apply / clear
        self._render_bottom_bar()

        # Auto pipeline assignments per axis
        self._render_auto_assignments()

        # Error display
        if self._last_errors:
            imgui.spacing()
            imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 0.4, 0.4, 1.0)
            for err in self._last_errors:
                imgui.text_wrapped(err)
            imgui.pop_style_color()

    # ---- Top bar ----

    def _render_top_bar(self):
        plugin_mgr = self._get_plugin_manager()
        if not plugin_mgr:
            imgui.text("Plugin system not initialized")
            return

        # Add Plugin dropdown
        if imgui.button("+ Add Plugin"):
            imgui.open_popup("##PipelineAddPlugin")

        if imgui.begin_popup("##PipelineAddPlugin"):
            categorized = self._categorize_plugins(plugin_mgr)
            for cat, names in categorized.items():
                if not names:
                    continue
                if imgui.begin_menu(cat):
                    for pn in names:
                        if imgui.menu_item(pn)[0]:
                            defaults = self._get_defaults_for(plugin_mgr, pn)
                            self.pipeline.add_step(pn, defaults)
                    imgui.end_menu()
            imgui.end_popup()

        imgui.same_line()
        imgui.text("|")
        imgui.same_line()

        # Save preset
        if imgui.button("Save Preset"):
            imgui.open_popup("##PipelineSavePreset")

        if imgui.begin_popup("##PipelineSavePreset"):
            imgui.text("Preset name:")
            changed, self._save_name_buf = imgui.input_text("##PresetName", self._save_name_buf, 128)
            if imgui.button("Save##Confirm") and self._save_name_buf.strip():
                self.pipeline.save_preset(self._save_name_buf.strip())
                self._save_name_buf = ""
                imgui.close_current_popup()
            imgui.end_popup()

        imgui.same_line()

        # Load preset
        presets = self.pipeline.get_available_presets()
        if imgui.button("Load Preset"):
            imgui.open_popup("##PipelineLoadPreset")

        if imgui.begin_popup("##PipelineLoadPreset"):
            for name in sorted(presets.keys()):
                builtin = self.pipeline.is_builtin_preset(name)
                label = f"{name}  (built-in)" if builtin else name
                if imgui.menu_item(label)[0]:
                    self.pipeline.load_preset(name)
            imgui.end_popup()

    # ---- Step list ----

    def _render_step_list(self):
        steps = self.pipeline.steps
        if not steps:
            imgui.spacing()
            imgui.text_colored("No steps. Add a plugin above.", 0.5, 0.5, 0.5, 1.0)
            imgui.spacing()
            return

        plugin_mgr = self._get_plugin_manager()
        remove_idx = None
        move_from = None
        move_to = None

        for i, step in enumerate(steps):
            imgui.push_id(f"step_{i}")

            # Enable/disable checkbox
            changed, step.enabled = imgui.checkbox(f"##en", step.enabled)

            imgui.same_line()

            # Step label (collapsible header for params)
            header_label = f"{i + 1}. {step.plugin_name}"
            if not step.enabled:
                imgui.push_style_color(imgui.COLOR_TEXT, 0.5, 0.5, 0.5, 1.0)

            header_open = imgui.tree_node(header_label)

            if not step.enabled:
                imgui.pop_style_color()

            # Move / remove buttons (right-aligned, always 3 slots for alignment)
            imgui.same_line(imgui.get_window_width() - 90)

            can_up = i > 0
            can_down = i < len(steps) - 1

            if not can_up:
                imgui.push_style_var(imgui.STYLE_ALPHA, 0.3)
            if imgui.small_button("^") and can_up:
                move_from = i
                move_to = i - 1
            if not can_up:
                imgui.pop_style_var()

            imgui.same_line()

            if not can_down:
                imgui.push_style_var(imgui.STYLE_ALPHA, 0.3)
            if imgui.small_button("v") and can_down:
                move_from = i
                move_to = i + 1
            if not can_down:
                imgui.pop_style_var()

            imgui.same_line()
            if imgui.small_button("x"):
                remove_idx = i

            # Parameter editing (inside collapsible)
            if header_open:
                if plugin_mgr:
                    self._render_step_params(step, plugin_mgr, i)
                imgui.tree_pop()

            imgui.pop_id()

        # Apply deferred mutations
        if remove_idx is not None:
            self.pipeline.remove_step(remove_idx)
        if move_from is not None:
            self.pipeline.move_step(move_from, move_to)

    def _render_step_params(self, step, plugin_mgr, step_idx: int):
        """Render parameter controls for a pipeline step."""
        ctx = plugin_mgr.plugin_contexts.get(step.plugin_name)
        if not ctx or not ctx.plugin_instance:
            imgui.text("Plugin not available")
            return

        schema = ctx.plugin_instance.parameters_schema
        for param_name, param_info in schema.items():
            # Skip list-type internal params
            if param_info.get('type') is list:
                continue

            current_value = step.params.get(param_name, param_info.get('default'))
            control_id = f"{param_name}##PL{step_idx}"
            display_name = param_name.replace('_', ' ').title()

            changed, new_value = self._render_param(control_id, display_name, param_info, current_value)
            if changed:
                step.params[param_name] = new_value

            if imgui.is_item_hovered() and param_info.get('description'):
                imgui.set_tooltip(param_info['description'])

    # ---- Bottom bar ----

    def _get_axis_choices(self):
        """Build axis choices from the current funscript (T1, T2, T3..., All)."""
        choices = ["T1", "T2"]
        processor = getattr(self.app, 'processor', None)
        if processor and processor.tracker and processor.tracker.funscript:
            fs = processor.tracker.funscript
            if hasattr(fs, 'get_all_axis_names'):
                all_axes = fs.get_all_axis_names()
                for i, name in enumerate(all_axes[2:], start=3):
                    choices.append(f"T{i}")
        choices.append("All")
        return choices

    def _render_bottom_bar(self):
        avail_w = imgui.get_content_region_available_width()
        spacing = imgui.get_style().item_spacing[0]

        # Row: [Edit] [Preview (T?)] [Apply (T?)]
        # Buttons share width equally
        btn_count = 3
        btn_w = max(60, (avail_w - spacing * (btn_count - 1)) / btn_count)

        target_label = self.pipeline.target_axis

        if imgui.button("Edit", width=btn_w):
            imgui.open_popup("##PipelineEditMenu")
        if imgui.begin_popup("##PipelineEditMenu"):
            # Axis selector inside edit menu
            axis_choices = self._get_axis_choices()
            cur_idx = axis_choices.index(self.pipeline.target_axis) if self.pipeline.target_axis in axis_choices else 0
            imgui.text("Target Axis")
            imgui.set_next_item_width(120)
            changed, new_idx = imgui.combo("##PipelineAxis", cur_idx, axis_choices)
            if changed:
                self.pipeline.target_axis = axis_choices[new_idx]
            imgui.separator()
            if imgui.menu_item("Clear All Steps")[0]:
                self.pipeline.clear()
                self._last_errors.clear()
            imgui.end_popup()

        imgui.same_line()

        if imgui.button(f"Preview ({target_label})", width=btn_w):
            self._preview_pipeline()
        if imgui.is_item_hovered():
            imgui.set_tooltip(f"Preview pipeline result on {target_label}")

        imgui.same_line()

        if self._previewing:
            if imgui.button("Cancel Preview", width=btn_w):
                self._clear_preview()
        else:
            if imgui.button(f"Apply ({target_label})", width=btn_w):
                self._clear_preview()
                self._apply_pipeline()
            if imgui.is_item_hovered():
                imgui.set_tooltip(f"Apply pipeline to {target_label} (Ctrl+Z to undo)")

    def _render_auto_assignments(self):
        """Render per-axis preset assignment for auto post-processing."""
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        imgui.push_style_color(imgui.COLOR_TEXT, 0.6, 0.7, 0.8, 1.0)
        imgui.text("Auto Post-Processing")
        imgui.pop_style_color()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Assign a preset to each axis.\nThese run automatically after tracking completes.")

        presets = self.pipeline.get_available_presets()
        preset_names = ["--"] + sorted(presets.keys())
        assignments = self.app.app_settings.get("auto_pipeline_assignments", {})
        changed = False

        axis_labels = self._get_axis_choices()
        axis_labels = [a for a in axis_labels if a != "All"]

        label_w = 24
        for axis_label in axis_labels:
            current_preset = assignments.get(axis_label, "")
            cur_idx = preset_names.index(current_preset) if current_preset in preset_names else 0

            imgui.text(axis_label)
            imgui.same_line(label_w + imgui.get_style().item_spacing[0])
            imgui.set_next_item_width(-1)
            ch, new_idx = imgui.combo(f"##AutoAssign_{axis_label}", cur_idx, preset_names)
            if ch:
                new_preset = preset_names[new_idx] if new_idx > 0 else ""
                if new_preset:
                    assignments[axis_label] = new_preset
                elif axis_label in assignments:
                    del assignments[axis_label]
                changed = True

        if changed:
            self.app.app_settings.set("auto_pipeline_assignments", assignments)

    def _get_timeline_editor(self):
        """Get the active timeline editor (T1)."""
        gui = getattr(self.app, 'gui_instance', None)
        if gui and hasattr(gui, 'timeline_editor1'):
            return gui.timeline_editor1
        return None

    def _get_timeline_editor_for_axis(self, target: str):
        """Get timeline editor for a given target label."""
        gui = getattr(self.app, 'gui_instance', None)
        if not gui:
            return None
        if target == "T2" and hasattr(gui, 'timeline_editor2'):
            return gui.timeline_editor2
        if target.startswith("T") and target[1:].isdigit():
            tnum = int(target[1:])
            if tnum >= 3 and hasattr(gui, '_extra_timeline_editors'):
                return gui._extra_timeline_editors.get(tnum)
        # Default to T1 (also for "All")
        return getattr(gui, 'timeline_editor1', None)

    def _preview_pipeline(self):
        """Run pipeline on a copy and show preview overlay on the timeline."""
        import copy as _copy
        self._last_errors.clear()
        processor = getattr(self.app, 'processor', None)
        if not processor or not processor.tracker or not processor.tracker.funscript:
            self._last_errors.append("No funscript loaded")
            return

        funscript_obj = processor.tracker.funscript
        target = self.pipeline.target_axis
        # Determine preview axis (for "All", preview T1)
        preview_axis = timeline_label_to_axis("T1" if target == "All" else target, funscript_obj)

        original_actions = list(funscript_obj.get_axis_actions(preview_axis) or [])
        if not original_actions:
            self._last_errors.append(f"No actions on {target}")
            return

        # Run pipeline on a deep copy
        preview_funscript = _copy.deepcopy(funscript_obj)
        success, errors = self.pipeline.run_with_target(preview_funscript)
        if errors:
            self._last_errors = errors
            return

        transformed_actions = list(preview_funscript.get_axis_actions(preview_axis) or [])

        # Build preview data for the timeline renderer
        preview_points = []
        for action in transformed_actions:
            preview_points.append({
                'at': action['at'],
                'pos': action['pos'],
                'is_modified': True,
                'is_selected': True,
            })

        preview_data = {
            'preview_points': preview_points,
            'style': 'default',
            'plugin_name': 'Pipeline',
        }

        # Send to the appropriate timeline's preview renderer
        editor = self._get_timeline_editor_for_axis(target)
        if editor and editor.plugin_preview_renderer:
            editor.plugin_preview_renderer.set_preview_data('Pipeline', preview_data)
            self._previewing = True
        else:
            self._last_errors.append("Timeline preview not available")

    def _clear_preview(self):
        """Clear the preview overlay from the timeline."""
        if self._previewing:
            editor = self._get_timeline_editor()
            if editor and editor.plugin_preview_renderer:
                editor.plugin_preview_renderer.clear_preview('Pipeline')
            self._previewing = False

    def _apply_pipeline(self):
        """Apply the pipeline to the targeted timeline(s)."""
        self._last_errors.clear()
        processor = getattr(self.app, 'processor', None)
        if not processor or not processor.tracker or not processor.tracker.funscript:
            self._last_errors.append("No funscript loaded")
            return

        funscript_obj = processor.tracker.funscript
        target = self.pipeline.target_axis
        fs_proc = self.app.funscript_processor

        # Determine which timelines are affected for undo
        if target == "All":
            affected = list(range(1, (funscript_obj.num_axes if hasattr(funscript_obj, 'num_axes') else 2) + 1))
        else:
            tnum = int(target[1:]) if target.startswith("T") and target[1:].isdigit() else 1
            affected = [tnum]

        # Capture before for undo
        before_map = {}
        for tnum in affected:
            axis_name = timeline_label_to_axis(f"T{tnum}", funscript_obj)
            before_map[tnum] = list(funscript_obj.get_axis_actions(axis_name) or [])

        success, errors = self.pipeline.run_with_target(funscript_obj)

        if errors:
            self._last_errors = errors

        # Refresh UI for all affected timelines
        if fs_proc:
            for tnum in affected:
                fs_proc._post_mutation_refresh(tnum, "Plugin Pipeline")

        if success:
            label = f"Plugin Pipeline ({target})"
            self.app.logger.info(f"Pipeline applied to {target}", extra={'status_message': True})
            # Undo for each affected timeline
            from application.classes.undo_manager import BulkReplaceCmd
            for tnum in affected:
                axis_name = timeline_label_to_axis(f"T{tnum}", funscript_obj)
                actions_after = list(funscript_obj.get_axis_actions(axis_name) or [])
                self.app.undo_manager.push_done(BulkReplaceCmd(
                    tnum, before_map[tnum], actions_after, label))

    # ---- Parameter rendering (mirrors plugin_ui_renderer patterns) ----

    @staticmethod
    def _render_param(control_id: str, display_name: str,
                      param_info: Dict[str, Any], current_value: Any) -> Tuple[bool, Any]:
        """Render a single parameter control."""
        param_type = param_info['type']
        constraints = param_info.get('constraints', {})

        if param_type == int:
            min_v = constraints.get('min', 0)
            max_v = constraints.get('max', 100)
            val = int(current_value) if current_value is not None else min_v
            if 'choices' in constraints:
                choices = constraints['choices']
                idx = choices.index(val) if val in choices else 0
                changed, new_idx = imgui.combo(f"{display_name}##{control_id}", idx,
                                               [str(c) for c in choices])
                return changed, choices[new_idx] if changed else val
            changed, new_val = imgui.slider_int(f"{display_name}##{control_id}", val, min_v, max_v)
            return changed, new_val

        elif param_type == float:
            min_v = constraints.get('min', 0.0)
            max_v = constraints.get('max', 1.0)
            val = float(current_value) if current_value is not None else min_v
            changed, new_val = imgui.slider_float(f"{display_name}##{control_id}", val, min_v, max_v)
            return changed, new_val

        elif param_type == bool:
            val = bool(current_value) if current_value is not None else False
            changed, new_val = imgui.checkbox(f"{display_name}##{control_id}", val)
            return changed, new_val

        elif param_type == str:
            val = str(current_value) if current_value is not None else ""
            if 'choices' in constraints:
                choices = constraints['choices']
                idx = choices.index(val) if val in choices else 0
                changed, new_idx = imgui.combo(f"{display_name}##{control_id}", idx, choices)
                return changed, choices[new_idx] if changed else val
            changed, new_val = imgui.input_text(f"{display_name}##{control_id}", val, 256)
            return changed, new_val

        else:
            imgui.text(f"{display_name}: {current_value}")
            return False, current_value

    # ---- Helpers ----

    @staticmethod
    def _categorize_plugins(plugin_mgr) -> OrderedDict:
        categorized = OrderedDict()
        for cat in _CATEGORY_ORDER:
            categorized[cat] = []
        for pn, ctx in plugin_mgr.plugin_contexts.items():
            cat = getattr(ctx.plugin_instance, 'category', 'General')
            if cat not in categorized:
                categorized[cat] = []
            categorized[cat].append(pn)
        return categorized

    @staticmethod
    def _get_defaults_for(plugin_mgr, plugin_name: str) -> dict:
        ctx = plugin_mgr.plugin_contexts.get(plugin_name)
        if ctx and ctx.plugin_instance:
            schema = ctx.plugin_instance.parameters_schema
            return {k: v.get('default') for k, v in schema.items() if 'default' in v and v.get('type') is not list}
        return {}
