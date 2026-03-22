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

from application.classes.plugin_pipeline import PluginPipeline

_CATEGORY_ORDER = ["Autotune", "Quickfix Tools", "Transform", "Smoothing", "Timing & Generation", "General"]


class PluginPipelineUI:
    """ImGui window that renders and manages a PluginPipeline."""

    def __init__(self, app_instance, logger: Optional[logging.Logger] = None):
        self.app = app_instance
        self.logger = logger or logging.getLogger('PluginPipelineUI')
        self.pipeline = PluginPipeline(app_instance, logger=self.logger)
        self._save_name_buf = ""
        self._last_errors: list = []

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
        # Top bar: add plugin + preset controls
        self._render_top_bar()

        imgui.separator()

        # Step list
        self._render_step_list()

        imgui.separator()

        # Bottom: apply / preview
        self._render_bottom_bar()

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

        # Preset load
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

        imgui.same_line()

        # Save preset
        if imgui.button("Save"):
            imgui.open_popup("##PipelineSavePreset")

        if imgui.begin_popup("##PipelineSavePreset"):
            imgui.text("Preset name:")
            changed, self._save_name_buf = imgui.input_text("##PresetName", self._save_name_buf, 128)
            if imgui.button("Save##Confirm") and self._save_name_buf.strip():
                self.pipeline.save_preset(self._save_name_buf.strip())
                self._save_name_buf = ""
                imgui.close_current_popup()
            imgui.end_popup()

    # ---- Step list ----

    def _render_step_list(self):
        steps = self.pipeline.steps
        if not steps:
            imgui.text_colored("No steps — add a plugin above.", 0.5, 0.5, 0.5, 1.0)
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

    def _render_bottom_bar(self):
        if imgui.button("Apply Pipeline", width=140):
            self._apply_pipeline()

        imgui.same_line()

        if imgui.button("Clear All", width=100):
            self.pipeline.clear()
            self._last_errors.clear()

    def _apply_pipeline(self):
        """Apply the pipeline to the active timeline's funscript."""
        self._last_errors.clear()
        processor = getattr(self.app, 'processor', None)
        if not processor or not processor.tracker or not processor.tracker.funscript:
            self._last_errors.append("No funscript loaded")
            return

        funscript_obj = processor.tracker.funscript

        # Record undo
        if hasattr(processor, '_record_timeline_action'):
            processor._record_timeline_action(1, "Plugin Pipeline")

        success, errors = self.pipeline.run(funscript_obj, axis='primary')

        if errors:
            self._last_errors = errors

        # Refresh UI
        if hasattr(processor, '_finalize_action_and_update_ui'):
            processor._finalize_action_and_update_ui(1, "Plugin Pipeline")

        if success:
            self.app.logger.info("Pipeline applied successfully", extra={'status_message': True})

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
