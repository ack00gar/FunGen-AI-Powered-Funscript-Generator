"""
Test GUI component class structure without requiring a display.

These tests verify the STRUCTURE of GUI classes -- inheritance, method
presence, attribute initialisation -- without rendering anything.
No glfw, OpenGL, or imgui context is needed.

The tests are architecture-agnostic: they work whether the classes use a
monolithic design or a mixin-based decomposition. The key requirement is
that all expected methods are reachable on the class (directly defined or
inherited via mixins).

Strategy:
  - ControlPanelUI: Use AST analysis on all relevant source files (main
    file + any mixin files) to collect the full method set.
  - GUI: Same AST approach across main file + mixin files.
"""

import ast
import os
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_GUI_COMPONENTS_DIR = _PROJECT_ROOT / "application" / "gui_components"


# ---------------------------------------------------------------------------
# AST-based class inspector
# ---------------------------------------------------------------------------

def _ast_get_class_methods(filepath: str, class_name: str):
    """Parse a Python file with AST and return the set of method names
    defined on the given class."""
    source = Path(filepath).read_text()
    tree = ast.parse(source, filename=filepath)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            methods = set()
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.add(item.name)
            return methods

    return set()


def _ast_get_all_classes_and_methods(filepath: str):
    """Parse a Python file and return {class_name: set(method_names)}."""
    source = Path(filepath).read_text()
    tree = ast.parse(source, filename=filepath)
    result = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = set()
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.add(item.name)
            result[node.name] = methods
    return result


def _ast_get_base_classes(filepath: str, class_name: str):
    """Return the list of base class names for a given class."""
    source = Path(filepath).read_text()
    tree = ast.parse(source, filename=filepath)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            bases = []
            for b in node.bases:
                if isinstance(b, ast.Name):
                    bases.append(b.id)
                elif isinstance(b, ast.Attribute):
                    bases.append(b.attr)
            return bases
    return []


def _collect_all_methods_for_class(main_file: str, class_name: str, search_dir: str):
    """Collect ALL methods available on a class, including those from
    mixin base classes defined in the same directory.

    Returns the union of methods defined directly on the class plus
    methods from any base classes found in sibling files.
    """
    # Get direct methods and base class names
    direct_methods = _ast_get_class_methods(main_file, class_name)
    bases = _ast_get_base_classes(main_file, class_name)

    all_methods = set(direct_methods)

    # For each base class, search all .py files in the directory
    if bases:
        for py_file in Path(search_dir).glob("*.py"):
            classes = _ast_get_all_classes_and_methods(str(py_file))
            for base in bases:
                if base in classes:
                    all_methods |= classes[base]

    return all_methods


# ---------------------------------------------------------------------------
# Pre-compute method sets for both classes (done once at module level)
# ---------------------------------------------------------------------------

_CP_FILE = str(_GUI_COMPONENTS_DIR / "control_panel_ui.py")
_GUI_FILE = str(_GUI_COMPONENTS_DIR / "app_gui.py")
_SEARCH_DIR = str(_GUI_COMPONENTS_DIR)

_CP_ALL_METHODS = _collect_all_methods_for_class(_CP_FILE, "ControlPanelUI", _SEARCH_DIR)
_GUI_ALL_METHODS = _collect_all_methods_for_class(_GUI_FILE, "GUI", _SEARCH_DIR)


# ---------------------------------------------------------------------------
# ControlPanelUI tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.gui
class TestControlPanelUIStructure:
    """Test ControlPanelUI class structure."""

    def test_control_panel_ui_has_render_method(self):
        """ControlPanelUI must have a render() entry point."""
        assert "render" in _CP_ALL_METHODS, "ControlPanelUI is missing render()"

    def test_control_panel_ui_has_init(self):
        """ControlPanelUI must define __init__."""
        assert "__init__" in _CP_ALL_METHODS, "ControlPanelUI is missing __init__"

    def test_control_panel_ui_has_all_render_tab_methods(self):
        """ControlPanelUI must have render methods for each major tab."""
        expected = [
            "render",
            "_render_run_control_tab",
            "_render_configuration_tab",
            "_render_settings_tab",
            "_render_post_processing_tab",
            "_render_advanced_tab",
            "_render_device_control_tab",
        ]
        for method_name in expected:
            assert method_name in _CP_ALL_METHODS, (
                f"ControlPanelUI is missing method: {method_name}"
            )

    def test_control_panel_ui_has_settings_sub_render_methods(self):
        """ControlPanelUI should have settings sub-section renderers."""
        expected = [
            "_render_ai_model_settings",
            "_render_settings_interface_perf",
            "_render_settings_file_output",
            "_render_settings_logging_autosave",
        ]
        for method_name in expected:
            assert method_name in _CP_ALL_METHODS, (
                f"ControlPanelUI is missing settings sub-method: {method_name}"
            )

    def test_control_panel_ui_has_device_control_methods(self):
        """ControlPanelUI should have all device control rendering methods."""
        expected = [
            "_render_device_control_tab",
            "_render_device_control_content",
            "_render_compact_connection_status",
            "_render_quick_controls",
            "_render_connection_status_section",
            "_render_device_types_section",
            "_render_osr_controls",
            "_render_handy_controls",
            "_render_buttplug_controls",
            "_render_device_settings_section",
        ]
        for method_name in expected:
            assert method_name in _CP_ALL_METHODS, (
                f"ControlPanelUI is missing device control method: {method_name}"
            )

    def test_control_panel_ui_has_streamer_methods(self):
        """ControlPanelUI should have native sync / streamer methods."""
        expected = [
            "_render_native_sync_tab",
            "_start_native_sync",
            "_stop_native_sync",
        ]
        for method_name in expected:
            assert method_name in _CP_ALL_METHODS, (
                f"ControlPanelUI is missing streamer method: {method_name}"
            )

    def test_control_panel_ui_has_execution_methods(self):
        """ControlPanelUI should have execution/run control methods."""
        expected = [
            "_render_execution_progress_display",
            "_render_start_stop_buttons",
            "_render_stage_progress_ui",
            "_render_calibration_window",
        ]
        for method_name in expected:
            assert method_name in _CP_ALL_METHODS, (
                f"ControlPanelUI is missing execution method: {method_name}"
            )

    def test_control_panel_ui_has_tracker_settings_methods(self):
        """ControlPanelUI should have tracker settings methods."""
        expected = [
            "_render_tracking_axes_mode",
            "_render_oscillation_detector_settings",
            "_render_live_tracker_settings",
            "_render_class_filtering_content",
        ]
        for method_name in expected:
            assert method_name in _CP_ALL_METHODS, (
                f"ControlPanelUI is missing tracker settings method: {method_name}"
            )

    def test_control_panel_ui_has_simple_mode_methods(self):
        """ControlPanelUI should have simple mode methods."""
        expected = [
            "_render_simple_mode_ui",
            "_render_simple_mode_tracker_selection",
            "_render_tracker_card",
            "_render_simple_progress_display",
        ]
        for method_name in expected:
            assert method_name in _CP_ALL_METHODS, (
                f"ControlPanelUI is missing simple mode method: {method_name}"
            )

    def test_control_panel_ui_has_post_processing_methods(self):
        """ControlPanelUI should have post-processing methods."""
        expected = [
            "_render_post_processing_tab",
            "_render_plugin_section",
            "_apply_plugin",
            "_render_funscript_processing_tools",
        ]
        for method_name in expected:
            assert method_name in _CP_ALL_METHODS, (
                f"ControlPanelUI is missing post-processing method: {method_name}"
            )

    def test_control_panel_method_count(self):
        """ControlPanelUI should have a significant number of methods (>50)."""
        non_dunder = [m for m in _CP_ALL_METHODS if not m.startswith("__")]
        assert len(non_dunder) > 50, (
            f"ControlPanelUI has only {len(non_dunder)} methods, expected > 50"
        )

    def test_control_panel_no_duplicate_definitions_in_source(self):
        """No single source file should have duplicate method defs for same class."""
        source = Path(_CP_FILE).read_text()
        tree = ast.parse(source, filename=_CP_FILE)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "ControlPanelUI":
                method_names = [
                    item.name for item in node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                duplicates = [
                    n for n in set(method_names) if method_names.count(n) > 1
                ]
                assert len(duplicates) == 0, (
                    f"ControlPanelUI has duplicate method defs: {duplicates}"
                )


# ---------------------------------------------------------------------------
# GUI class tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.gui
class TestGUIClassStructure:
    """Test GUI class structure via AST analysis (no display needed)."""

    def test_gui_class_has_init(self):
        """GUI should define __init__."""
        assert "__init__" in _GUI_ALL_METHODS, "GUI is missing __init__"

    def test_gui_class_has_render_gui_method(self):
        """GUI must have a render_gui() entry point."""
        assert "render_gui" in _GUI_ALL_METHODS, "GUI is missing render_gui()"

    def test_gui_class_has_global_shortcuts_handler(self):
        """GUI should have _handle_global_shortcuts and individual shortcut handlers."""
        expected = [
            "_handle_global_shortcuts",
            "_handle_set_chapter_start_shortcut",
            "_handle_set_chapter_end_shortcut",
            "_handle_save_project_shortcut",
            "_handle_open_project_shortcut",
            "_handle_jump_to_start_shortcut",
            "_handle_jump_to_end_shortcut",
            "_handle_zoom_in_timeline_shortcut",
            "_handle_zoom_out_timeline_shortcut",
        ]
        for method_name in expected:
            assert method_name in _GUI_ALL_METHODS, (
                f"GUI is missing shortcut handler: {method_name}"
            )

    def test_gui_class_has_toggle_shortcut_handlers(self):
        """GUI should have toggle shortcut handlers for UI elements."""
        toggle_methods = [
            "_handle_toggle_video_display_shortcut",
            "_handle_toggle_timeline2_shortcut",
            "_handle_toggle_gauge_window_shortcut",
            "_handle_toggle_3d_simulator_shortcut",
            "_handle_toggle_movement_bar_shortcut",
            "_handle_toggle_chapter_list_shortcut",
            "_handle_toggle_heatmap_shortcut",
            "_handle_toggle_funscript_preview_shortcut",
            "_handle_toggle_video_feed_shortcut",
            "_handle_toggle_waveform_shortcut",
            "_handle_reset_timeline_view_shortcut",
        ]
        for method_name in toggle_methods:
            assert method_name in _GUI_ALL_METHODS, (
                f"GUI is missing toggle shortcut handler: {method_name}"
            )

    def test_gui_class_has_preview_methods(self):
        """GUI should have preview generation methods."""
        expected = [
            "_preview_generation_worker",
            "_process_preview_results",
            "_generate_funscript_preview_data",
            "_generate_heatmap_data",
        ]
        for method_name in expected:
            assert method_name in _GUI_ALL_METHODS, (
                f"GUI is missing preview method: {method_name}"
            )

    def test_gui_class_has_dialog_methods(self):
        """GUI should have dialog rendering methods."""
        expected = [
            "_render_batch_confirmation_dialog",
            "_render_ai_models_dialog",
            "_render_status_message",
            "_render_error_popup",
            "_render_all_popups",
            "show_error_popup",
        ]
        for method_name in expected:
            assert method_name in _GUI_ALL_METHODS, (
                f"GUI is missing dialog method: {method_name}"
            )

    def test_gui_class_has_core_methods(self):
        """GUI should have core infrastructure methods."""
        expected = [
            "init_glfw",
            "run",
            "cleanup",
            "update_texture",
            "handle_drop",
            "handle_window_close",
        ]
        for method_name in expected:
            assert method_name in _GUI_ALL_METHODS, (
                f"GUI is missing core method: {method_name}"
            )

    def test_gui_class_method_count(self):
        """GUI should have a substantial number of methods (>40)."""
        non_dunder = [m for m in _GUI_ALL_METHODS if not m.startswith("__")]
        assert len(non_dunder) > 40, (
            f"GUI has only {len(non_dunder)} methods, expected > 40"
        )

    def test_gui_class_no_duplicate_definitions_in_source(self):
        """No duplicate method definitions in the main GUI source file."""
        source = Path(_GUI_FILE).read_text()
        tree = ast.parse(source, filename=_GUI_FILE)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "GUI":
                method_names = [
                    item.name for item in node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                duplicates = [
                    n for n in set(method_names) if method_names.count(n) > 1
                ]
                assert len(duplicates) == 0, (
                    f"GUI has duplicate method defs: {duplicates}"
                )


# ---------------------------------------------------------------------------
# Cross-component consistency tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.gui
class TestGUIComponentConsistency:
    """Tests verifying consistency across GUI components."""

    def test_control_panel_render_method_count(self):
        """ControlPanelUI should have many _render_* methods."""
        render_methods = [m for m in _CP_ALL_METHODS if m.startswith("_render_")]
        assert len(render_methods) > 15, (
            f"Expected > 15 _render_* methods on ControlPanelUI, found {len(render_methods)}"
        )

    def test_gui_render_and_handle_method_counts(self):
        """GUI should have many _render_* and _handle_* methods."""
        render_methods = [m for m in _GUI_ALL_METHODS if m.startswith("_render_")]
        handle_methods = [m for m in _GUI_ALL_METHODS if m.startswith("_handle_")]
        total = len(render_methods) + len(handle_methods)
        assert total > 20, (
            f"Expected > 20 _render_/_handle_* methods on GUI, found {total}"
        )

    def test_mixin_files_exist_if_used(self):
        """If ControlPanelUI inherits from mixins, those files must exist."""
        bases = _ast_get_base_classes(_CP_FILE, "ControlPanelUI")
        if bases:  # Mixin architecture
            # Verify each mixin can be found in some file
            found_bases = set()
            for py_file in Path(_SEARCH_DIR).glob("*.py"):
                classes = _ast_get_all_classes_and_methods(str(py_file))
                found_bases |= set(classes.keys())
            for base in bases:
                assert base in found_bases, (
                    f"Mixin base class '{base}' not found in any file under {_SEARCH_DIR}"
                )

    def test_no_method_name_conflicts_across_mixins(self):
        """If using mixins, no two mixins should define the same method."""
        bases = _ast_get_base_classes(_CP_FILE, "ControlPanelUI")
        if not bases:
            pytest.skip("Not using mixin architecture")

        mixin_methods = {}  # method_name -> [mixin_names]
        for py_file in Path(_SEARCH_DIR).glob("*.py"):
            classes = _ast_get_all_classes_and_methods(str(py_file))
            for base in bases:
                if base in classes:
                    for method in classes[base]:
                        if method.startswith("__"):
                            continue
                        mixin_methods.setdefault(method, []).append(base)

        conflicts = {m: sources for m, sources in mixin_methods.items() if len(sources) > 1}
        assert len(conflicts) == 0, (
            f"Method name conflicts across mixins: {conflicts}"
        )
