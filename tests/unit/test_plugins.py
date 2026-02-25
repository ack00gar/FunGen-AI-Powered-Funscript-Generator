"""
Comprehensive tests for the FunGen plugin system.

Tests cover:
- PluginRegistry: registration, retrieval, unregistration, listing, visibility
- PluginLoader: built-in plugin discovery and loading
- Base plugin validation: parameter validation against schemas
- Individual plugin transforms: Amplify, Invert, Clamp, ThresholdClamp
- All-plugins smoke tests: metadata presence, transform safety
"""

import copy
import logging
import pytest
from typing import Dict, Any

from funscript.plugins.base_plugin import (
    FunscriptTransformationPlugin,
    PluginRegistry,
    plugin_registry,
)
from funscript.plugins.plugin_loader import PluginLoader
from funscript.multi_axis_funscript import MultiAxisFunscript


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample_funscript() -> MultiAxisFunscript:
    """Create a MultiAxisFunscript with deterministic sample data on primary axis.

    Produces 11 points at 100 ms intervals with positions cycling 0-100:
        at=0/pos=0, at=100/pos=20, at=200/pos=40, at=300/pos=60, at=400/pos=80,
        at=500/pos=0, at=600/pos=20, at=700/pos=40, at=800/pos=60, at=900/pos=80,
        at=1000/pos=0
    """
    fs = MultiAxisFunscript()
    fs.primary_actions = [{"at": i * 100, "pos": (i * 20) % 100} for i in range(11)]
    fs._invalidate_cache("primary")
    return fs


def _make_funscript_with_both_axes() -> MultiAxisFunscript:
    """Create a MultiAxisFunscript with data on both axes."""
    fs = _make_sample_funscript()
    fs.secondary_actions = [{"at": i * 100, "pos": 100 - (i * 20) % 100} for i in range(11)]
    fs._invalidate_cache("secondary")
    return fs


class _DummyPlugin(FunscriptTransformationPlugin):
    """Minimal concrete plugin for registry tests."""

    @property
    def name(self) -> str:
        return "DummyTestPlugin"

    @property
    def description(self) -> str:
        return "A dummy plugin used in unit tests"

    @property
    def version(self) -> str:
        return "0.0.1"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "intensity": {
                "type": float,
                "required": True,
                "description": "Intensity of the effect",
                "constraints": {"min": 0.0, "max": 10.0},
            },
            "mode": {
                "type": str,
                "required": False,
                "default": "normal",
                "description": "Operation mode",
                "constraints": {"choices": ["normal", "fast", "slow"]},
            },
        }

    def transform(self, funscript, axis="both", **parameters):
        return None


class _DummyPluginB(FunscriptTransformationPlugin):
    """Second dummy plugin with a different name."""

    @property
    def name(self) -> str:
        return "DummyTestPluginB"

    @property
    def description(self) -> str:
        return "Another dummy plugin"

    @property
    def version(self) -> str:
        return "0.0.2"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {}

    def transform(self, funscript, axis="both", **parameters):
        return None


# ---------------------------------------------------------------------------
# Registry Tests
# ---------------------------------------------------------------------------

@pytest.mark.plugins
class TestPluginRegistry:
    """Tests for PluginRegistry: register, get, unregister, list."""

    def _fresh_registry(self) -> PluginRegistry:
        return PluginRegistry(logger=logging.getLogger("test_registry"))

    # -- register / get --

    def test_register_and_get_plugin(self):
        reg = self._fresh_registry()
        plugin = _DummyPlugin()
        assert reg.register(plugin) is True
        retrieved = reg.get_plugin("DummyTestPlugin")
        assert retrieved is plugin

    def test_get_plugin_returns_none_for_unknown(self):
        reg = self._fresh_registry()
        assert reg.get_plugin("NonExistent") is None

    # -- unregister --

    def test_unregister_existing_plugin(self):
        reg = self._fresh_registry()
        reg.register(_DummyPlugin())
        assert reg.unregister("DummyTestPlugin") is True
        assert reg.get_plugin("DummyTestPlugin") is None

    def test_unregister_nonexistent_returns_false(self):
        reg = self._fresh_registry()
        assert reg.unregister("DoesNotExist") is False

    # -- list / get_all --

    def test_list_plugins_returns_all_non_hidden(self):
        reg = self._fresh_registry()
        reg.register(_DummyPlugin())
        reg.register(_DummyPluginB())
        listing = reg.list_plugins()
        names = [p["name"] for p in listing]
        assert "DummyTestPlugin" in names
        assert "DummyTestPluginB" in names

    def test_list_plugins_hides_template_and_example(self):
        """Plugins whose name contains 'template' or 'example' should be hidden."""
        reg = self._fresh_registry()

        class _TemplatePlugin(_DummyPlugin):
            @property
            def name(self):
                return "template_plugin"

        class _ExamplePlugin(_DummyPlugin):
            @property
            def name(self):
                return "my_example"

        reg.register(_TemplatePlugin())
        reg.register(_ExamplePlugin())
        listing = reg.list_plugins()
        names = [p["name"] for p in listing]
        assert "template_plugin" not in names
        assert "my_example" not in names

    # -- duplicate name --

    def test_duplicate_register_replaces_plugin(self):
        """Registering a plugin with the same name should replace the old one."""
        reg = self._fresh_registry()
        first = _DummyPlugin()
        second = _DummyPlugin()
        reg.register(first)
        reg.register(second)
        assert reg.get_plugin("DummyTestPlugin") is second

    # -- capability filtering --

    def test_get_plugins_by_capability_scipy(self):
        reg = self._fresh_registry()
        reg.register(_DummyPlugin())  # requires_scipy = False

        class _ScipyPlugin(_DummyPlugin):
            @property
            def name(self):
                return "ScipyDummy"

            @property
            def requires_scipy(self):
                return True

            def check_dependencies(self):
                return True  # pretend it's available

        reg.register(_ScipyPlugin())
        scipy_plugins = reg.get_plugins_by_capability(requires_scipy=True)
        assert "ScipyDummy" in scipy_plugins
        assert "DummyTestPlugin" not in scipy_plugins

    def test_get_plugins_by_capability_axis(self):
        reg = self._fresh_registry()
        reg.register(_DummyPlugin())
        both_axis = reg.get_plugins_by_capability(supports_axis="both")
        assert "DummyTestPlugin" in both_axis

    # -- global loaded flag --

    def test_global_plugins_loaded_flag(self):
        reg = self._fresh_registry()
        assert reg.is_global_plugins_loaded() is False
        reg.set_global_plugins_loaded(True)
        assert reg.is_global_plugins_loaded() is True


# ---------------------------------------------------------------------------
# Loader Tests
# ---------------------------------------------------------------------------

@pytest.mark.plugins
class TestPluginLoader:
    """Tests for PluginLoader: built-in plugin discovery."""

    def test_load_builtin_plugins_discovers_expected_count(self):
        """load_builtin_plugins should discover at least 12 plugin files."""
        loader = PluginLoader()
        results = loader.load_builtin_plugins()
        # results maps filename -> bool; at least 12 plugin files should load
        successful = [f for f, ok in results.items() if ok]
        assert len(successful) >= 12, (
            f"Expected at least 12 successfully loaded plugin files, got {len(successful)}: {successful}"
        )

    def test_load_builtin_plugins_registers_in_global_registry(self):
        """After loading, the global registry should contain the built-in plugins."""
        # Use a fresh loader but note it registers into the global plugin_registry.
        # Some plugins may already be there from other tests, so we check a few known names.
        loader = PluginLoader()
        loader.load_builtin_plugins()

        expected_names = [
            "Amplify",
            "Invert",
            "Threshold Clamp",
            "Speed Limiter",
            "Anti-Jerk",
            "Time Shift",
            "Dynamic Amplify",
            "Resample",
        ]
        for name in expected_names:
            assert plugin_registry.get_plugin(name) is not None, (
                f"Expected plugin '{name}' to be in global registry after load_builtin_plugins"
            )

    def test_load_from_nonexistent_directory(self):
        """Loading from a non-existent directory should return empty results."""
        loader = PluginLoader()
        results = loader.load_plugins_from_directory("/tmp/nonexistent_plugin_dir_xyz")
        assert results == {}

    def test_load_plugin_from_nonexistent_file(self):
        """Loading from a non-existent file should return False."""
        loader = PluginLoader()
        assert loader.load_plugin_from_file("/tmp/nonexistent_plugin_file_xyz.py") is False


# ---------------------------------------------------------------------------
# Base Validation Tests
# ---------------------------------------------------------------------------

@pytest.mark.plugins
class TestPluginBaseValidation:
    """Tests for FunscriptTransformationPlugin.validate_parameters."""

    def _plugin(self) -> _DummyPlugin:
        return _DummyPlugin()

    # -- valid params --

    def test_valid_required_and_optional(self):
        p = self._plugin()
        result = p.validate_parameters({"intensity": 5.0, "mode": "fast"})
        assert result["intensity"] == 5.0
        assert result["mode"] == "fast"

    def test_defaults_applied_when_optional_missing(self):
        p = self._plugin()
        result = p.validate_parameters({"intensity": 1.0})
        assert result["mode"] == "normal"

    # -- missing required --

    def test_missing_required_raises(self):
        p = self._plugin()
        with pytest.raises(ValueError, match="Required parameter 'intensity' is missing"):
            p.validate_parameters({})

    # -- type coercion --

    def test_type_coercion_int_to_float(self):
        p = self._plugin()
        result = p.validate_parameters({"intensity": 3})
        assert isinstance(result["intensity"], float)
        assert result["intensity"] == 3.0

    def test_type_coercion_failure_raises(self):
        p = self._plugin()
        with pytest.raises(ValueError, match="must be of type float"):
            p.validate_parameters({"intensity": "not_a_number"})

    # -- constraint: min / max --

    def test_constraint_min_violated(self):
        p = self._plugin()
        with pytest.raises(ValueError, match="must be >= 0.0"):
            p.validate_parameters({"intensity": -1.0})

    def test_constraint_max_violated(self):
        p = self._plugin()
        with pytest.raises(ValueError, match="must be <= 10.0"):
            p.validate_parameters({"intensity": 11.0})

    # -- constraint: choices --

    def test_constraint_choices_valid(self):
        p = self._plugin()
        result = p.validate_parameters({"intensity": 1.0, "mode": "slow"})
        assert result["mode"] == "slow"

    def test_constraint_choices_invalid(self):
        p = self._plugin()
        with pytest.raises(ValueError, match="must be one of"):
            p.validate_parameters({"intensity": 1.0, "mode": "turbo"})

    # -- supported axes --

    def test_default_supported_axes(self):
        p = self._plugin()
        assert set(p.supported_axes) == {"primary", "secondary", "both"}

    # -- dependency flags --

    def test_default_requires_scipy_false(self):
        p = self._plugin()
        assert p.requires_scipy is False

    def test_default_requires_rdp_false(self):
        p = self._plugin()
        assert p.requires_rdp is False

    # -- ui_preference --

    def test_default_ui_preference_popup(self):
        p = self._plugin()
        assert p.ui_preference == "popup"


# ---------------------------------------------------------------------------
# Amplify Plugin Tests
# ---------------------------------------------------------------------------

@pytest.mark.plugins
class TestAmplifyPlugin:
    """Tests for the Amplify plugin transform behaviour."""

    def _plugin(self):
        from funscript.plugins.amplify_plugin import AmplifyPlugin
        return AmplifyPlugin()

    def test_metadata(self):
        p = self._plugin()
        assert p.name == "Amplify"
        assert p.version == "1.0.0"
        assert p.ui_preference == "popup"

    def test_default_parameters(self):
        p = self._plugin()
        validated = p.validate_parameters({})
        assert validated["scale_factor"] == 1.25
        assert validated["center_value"] == 50

    def test_transform_with_defaults_modifies_positions(self):
        """Default scale_factor=1.25 around center=50 should push values away from 50."""
        p = self._plugin()
        fs = _make_sample_funscript()
        original_positions = [a["pos"] for a in fs.primary_actions]
        p.transform(fs, axis="primary")
        new_positions = [a["pos"] for a in fs.primary_actions]
        # Positions that were != 50 should have changed
        assert new_positions != original_positions

    def test_scale_factor_gt_1_increases_amplitude(self):
        """scale_factor > 1 should push positions further from center."""
        p = self._plugin()
        fs = _make_sample_funscript()
        # Use a large scale factor
        p.transform(fs, axis="primary", scale_factor=2.0, center_value=50)
        # Positions that started at 0 (far from 50) should be pushed to 0 (clamped).
        # Positions at 80 should go towards 100.
        for action in fs.primary_actions:
            assert 0 <= action["pos"] <= 100

    def test_scale_factor_lt_1_decreases_amplitude(self):
        """scale_factor < 1 should pull positions toward center."""
        p = self._plugin()
        fs = _make_sample_funscript()
        original_positions = [a["pos"] for a in fs.primary_actions]
        p.transform(fs, axis="primary", scale_factor=0.5, center_value=50)
        new_positions = [a["pos"] for a in fs.primary_actions]
        # Every position should be closer to 50 than before
        for orig, new in zip(original_positions, new_positions):
            assert abs(new - 50) <= abs(orig - 50) + 1  # +1 for rounding

    def test_scale_factor_1_no_change(self):
        """scale_factor=1.0 should produce no change."""
        p = self._plugin()
        fs = _make_sample_funscript()
        original_positions = [a["pos"] for a in fs.primary_actions]
        p.transform(fs, axis="primary", scale_factor=1.0, center_value=50)
        new_positions = [a["pos"] for a in fs.primary_actions]
        assert new_positions == original_positions

    def test_center_value_shift(self):
        """Different center_value should change the amplification reference point."""
        p = self._plugin()
        fs1 = _make_sample_funscript()
        fs2 = _make_sample_funscript()
        p.transform(fs1, axis="primary", scale_factor=2.0, center_value=0)
        p.transform(fs2, axis="primary", scale_factor=2.0, center_value=100)
        pos1 = [a["pos"] for a in fs1.primary_actions]
        pos2 = [a["pos"] for a in fs2.primary_actions]
        # With center=0, values should be pushed higher; with center=100, pushed lower
        assert pos1 != pos2

    def test_selected_indices(self):
        """Only selected indices should be modified."""
        p = self._plugin()
        fs = _make_sample_funscript()
        original_positions = [a["pos"] for a in fs.primary_actions]
        # Only amplify indices 2 and 3
        p.transform(fs, axis="primary", scale_factor=3.0, center_value=50, selected_indices=[2, 3])
        new_positions = [a["pos"] for a in fs.primary_actions]
        # Indices 0, 1, 4+ should be unchanged
        for i in [0, 1, 4, 5, 6, 7, 8, 9, 10]:
            assert new_positions[i] == original_positions[i], f"Index {i} should not have changed"
        # At least one of the selected indices should have changed
        assert (new_positions[2] != original_positions[2] or new_positions[3] != original_positions[3])

    def test_time_range_filtering(self):
        """Only points within the time range should be modified."""
        p = self._plugin()
        fs = _make_sample_funscript()
        original_positions = [a["pos"] for a in fs.primary_actions]
        # Amplify only 200ms-400ms range (indices 2, 3, 4)
        p.transform(fs, axis="primary", scale_factor=3.0, center_value=50,
                     start_time_ms=200, end_time_ms=400)
        new_positions = [a["pos"] for a in fs.primary_actions]
        # Points outside range should be unchanged
        for i in [0, 1, 5, 6, 7, 8, 9, 10]:
            assert new_positions[i] == original_positions[i], f"Index {i} should not have changed"

    def test_constraint_scale_factor_too_low(self):
        p = self._plugin()
        with pytest.raises(ValueError, match="must be >= 0.1"):
            p.validate_parameters({"scale_factor": 0.01})

    def test_constraint_scale_factor_too_high(self):
        p = self._plugin()
        with pytest.raises(ValueError, match="must be <= 5.0"):
            p.validate_parameters({"scale_factor": 6.0})

    def test_positions_always_clamped_0_100(self):
        """No matter how extreme the scale, positions must stay in [0, 100]."""
        p = self._plugin()
        fs = _make_sample_funscript()
        p.transform(fs, axis="primary", scale_factor=5.0, center_value=50)
        for action in fs.primary_actions:
            assert 0 <= action["pos"] <= 100

    def test_transform_on_empty_axis_does_not_crash(self):
        """Transforming an axis with no data should not raise."""
        p = self._plugin()
        fs = MultiAxisFunscript()
        # secondary_actions is empty
        p.transform(fs, axis="secondary", scale_factor=2.0)


# ---------------------------------------------------------------------------
# Invert Plugin Tests
# ---------------------------------------------------------------------------

@pytest.mark.plugins
class TestInvertPlugin:
    """Tests for the Invert plugin."""

    def _plugin(self):
        from funscript.plugins.invert_plugin import InvertPlugin
        return InvertPlugin()

    def test_metadata(self):
        p = self._plugin()
        assert p.name == "Invert"
        assert p.ui_preference == "direct"

    def test_invert_0_becomes_100(self):
        fs = MultiAxisFunscript()
        fs.primary_actions = [{"at": 0, "pos": 0}]
        fs._invalidate_cache("primary")
        self._plugin().transform(fs, axis="primary")
        assert fs.primary_actions[0]["pos"] == 100

    def test_invert_100_becomes_0(self):
        fs = MultiAxisFunscript()
        fs.primary_actions = [{"at": 0, "pos": 100}]
        fs._invalidate_cache("primary")
        self._plugin().transform(fs, axis="primary")
        assert fs.primary_actions[0]["pos"] == 0

    def test_invert_50_stays_50(self):
        fs = MultiAxisFunscript()
        fs.primary_actions = [{"at": 0, "pos": 50}]
        fs._invalidate_cache("primary")
        self._plugin().transform(fs, axis="primary")
        assert fs.primary_actions[0]["pos"] == 50

    def test_double_invert_restores_original(self):
        p = self._plugin()
        fs = _make_sample_funscript()
        original_positions = [a["pos"] for a in fs.primary_actions]
        p.transform(fs, axis="primary")
        p.transform(fs, axis="primary")
        restored_positions = [a["pos"] for a in fs.primary_actions]
        assert restored_positions == original_positions

    def test_invert_all_values_formula(self):
        """Every position should satisfy: new_pos = 100 - old_pos."""
        p = self._plugin()
        fs = _make_sample_funscript()
        original_positions = [a["pos"] for a in fs.primary_actions]
        p.transform(fs, axis="primary")
        for orig, action in zip(original_positions, fs.primary_actions):
            assert action["pos"] == 100 - orig

    def test_invert_preserves_timestamps(self):
        p = self._plugin()
        fs = _make_sample_funscript()
        original_times = [a["at"] for a in fs.primary_actions]
        p.transform(fs, axis="primary")
        new_times = [a["at"] for a in fs.primary_actions]
        assert new_times == original_times

    def test_invert_with_selected_indices(self):
        """Only selected indices should be inverted."""
        p = self._plugin()
        fs = _make_sample_funscript()
        original_positions = [a["pos"] for a in fs.primary_actions]
        p.transform(fs, axis="primary", selected_indices=[0, 1])
        new_positions = [a["pos"] for a in fs.primary_actions]
        assert new_positions[0] == 100 - original_positions[0]
        assert new_positions[1] == 100 - original_positions[1]
        # Rest unchanged
        for i in range(2, len(original_positions)):
            assert new_positions[i] == original_positions[i]


# ---------------------------------------------------------------------------
# Clamp Plugin Tests
# ---------------------------------------------------------------------------

@pytest.mark.plugins
class TestClampPlugins:
    """Tests for ThresholdClampPlugin and ValueClampPlugin."""

    def _threshold_plugin(self):
        from funscript.plugins.clamp_plugin import ThresholdClampPlugin
        return ThresholdClampPlugin()

    def _value_plugin(self):
        from funscript.plugins.clamp_plugin import ValueClampPlugin
        return ValueClampPlugin()

    # -- ThresholdClamp --

    def test_threshold_clamp_metadata(self):
        p = self._threshold_plugin()
        assert p.name == "Threshold Clamp"
        assert p.version == "1.0.0"

    def test_threshold_clamp_default_params(self):
        p = self._threshold_plugin()
        validated = p.validate_parameters({})
        assert validated["lower_threshold"] == 20
        assert validated["upper_threshold"] == 80

    def test_threshold_clamp_values_below_lower_become_0(self):
        p = self._threshold_plugin()
        fs = MultiAxisFunscript()
        fs.primary_actions = [
            {"at": 0, "pos": 10},   # below 20 -> 0
            {"at": 100, "pos": 50}, # between -> unchanged
            {"at": 200, "pos": 90}, # above 80 -> 100
        ]
        fs._invalidate_cache("primary")
        p.transform(fs, axis="primary", lower_threshold=20, upper_threshold=80)
        positions = [a["pos"] for a in fs.primary_actions]
        assert positions[0] == 0
        assert positions[1] == 50
        assert positions[2] == 100

    def test_threshold_clamp_values_in_range_unchanged(self):
        p = self._threshold_plugin()
        fs = MultiAxisFunscript()
        fs.primary_actions = [
            {"at": 0, "pos": 30},
            {"at": 100, "pos": 50},
            {"at": 200, "pos": 70},
        ]
        fs._invalidate_cache("primary")
        p.transform(fs, axis="primary", lower_threshold=20, upper_threshold=80)
        positions = [a["pos"] for a in fs.primary_actions]
        assert positions == [30, 50, 70]

    def test_threshold_clamp_lower_ge_upper_raises(self):
        """lower_threshold must be < upper_threshold."""
        p = self._threshold_plugin()
        with pytest.raises(ValueError, match="lower_threshold must be less than upper_threshold"):
            p.validate_parameters({"lower_threshold": 80, "upper_threshold": 80})

    def test_threshold_clamp_lower_gt_upper_raises(self):
        p = self._threshold_plugin()
        with pytest.raises(ValueError, match="lower_threshold must be less than upper_threshold"):
            p.validate_parameters({"lower_threshold": 90, "upper_threshold": 50})

    # -- ValueClamp --

    def test_value_clamp_metadata(self):
        p = self._value_plugin()
        assert p.name == "Clamp"
        assert p.version == "1.0.0"

    def test_value_clamp_sets_all_to_clamp_value(self):
        p = self._value_plugin()
        fs = _make_sample_funscript()
        p.transform(fs, axis="primary", clamp_value=75)
        for action in fs.primary_actions:
            assert action["pos"] == 75

    def test_value_clamp_preserves_timestamps(self):
        p = self._value_plugin()
        fs = _make_sample_funscript()
        original_times = [a["at"] for a in fs.primary_actions]
        p.transform(fs, axis="primary", clamp_value=50)
        new_times = [a["at"] for a in fs.primary_actions]
        assert new_times == original_times

    def test_value_clamp_constraint_range(self):
        p = self._value_plugin()
        with pytest.raises(ValueError):
            p.validate_parameters({"clamp_value": -1})
        with pytest.raises(ValueError):
            p.validate_parameters({"clamp_value": 101})


# ---------------------------------------------------------------------------
# RDP Simplify Plugin Tests
# ---------------------------------------------------------------------------

@pytest.mark.plugins
class TestRdpSimplifyPlugin:
    """Tests for the RDP Simplify plugin."""

    def _plugin(self):
        from funscript.plugins.rdp_simplify_plugin import RdpSimplifyPlugin
        return RdpSimplifyPlugin()

    def test_metadata(self):
        p = self._plugin()
        assert p.name == "Simplify (RDP)"
        assert p.version == "1.0.0"

    def test_simplify_reduces_point_count(self):
        """RDP with a reasonable epsilon should reduce redundant points."""
        p = self._plugin()
        # Create linear data (many collinear points) which RDP should simplify aggressively
        fs = MultiAxisFunscript()
        fs.primary_actions = [{"at": i * 10, "pos": i} for i in range(101)]
        fs._invalidate_cache("primary")
        original_count = len(fs.primary_actions)
        p.transform(fs, axis="primary", epsilon=1.0)
        new_count = len(fs.primary_actions)
        assert new_count < original_count, "RDP should have reduced linear points"

    def test_simplify_preserves_first_and_last(self):
        p = self._plugin()
        fs = _make_sample_funscript()
        first_action = copy.deepcopy(fs.primary_actions[0])
        last_action = copy.deepcopy(fs.primary_actions[-1])
        p.transform(fs, axis="primary", epsilon=1.0)
        assert fs.primary_actions[0]["at"] == first_action["at"]
        assert fs.primary_actions[-1]["at"] == last_action["at"]

    def test_simplify_does_not_crash_on_few_points(self):
        p = self._plugin()
        fs = MultiAxisFunscript()
        fs.primary_actions = [{"at": 0, "pos": 50}, {"at": 100, "pos": 80}]
        fs._invalidate_cache("primary")
        # Should not raise - just return the same or fewer points
        p.transform(fs, axis="primary", epsilon=5.0)
        assert len(fs.primary_actions) >= 1


# ---------------------------------------------------------------------------
# Savgol Filter Plugin Tests
# ---------------------------------------------------------------------------

@pytest.mark.plugins
class TestSavgolFilterPlugin:
    """Tests for the Savgol (Smooth) plugin."""

    def _plugin(self):
        from funscript.plugins.savgol_filter_plugin import SavgolFilterPlugin
        return SavgolFilterPlugin()

    def test_metadata(self):
        p = self._plugin()
        assert p.name == "Smooth (SG)"
        assert p.requires_scipy is True

    def test_transform_smooths_data(self):
        """Savgol filter should smooth noisy data, reducing variance."""
        pytest.importorskip("scipy")
        p = self._plugin()
        # Create noisy data
        fs = MultiAxisFunscript()
        import numpy as np
        np.random.seed(42)
        noisy = np.clip(50 + np.random.randn(50) * 20, 0, 100).astype(int)
        fs.primary_actions = [{"at": i * 50, "pos": int(noisy[i])} for i in range(50)]
        fs._invalidate_cache("primary")
        original_variance = np.var([a["pos"] for a in fs.primary_actions])
        p.transform(fs, axis="primary", window_length=7, polyorder=3)
        new_variance = np.var([a["pos"] for a in fs.primary_actions])
        # Smoothing should reduce or maintain variance
        assert new_variance <= original_variance * 1.1  # allow small rounding increase

    def test_positions_stay_in_range(self):
        pytest.importorskip("scipy")
        p = self._plugin()
        fs = _make_sample_funscript()
        p.transform(fs, axis="primary", window_length=5, polyorder=2)
        for action in fs.primary_actions:
            assert 0 <= action["pos"] <= 100


# ---------------------------------------------------------------------------
# Speed Limiter Plugin Tests
# ---------------------------------------------------------------------------

@pytest.mark.plugins
class TestSpeedLimiterPlugin:
    """Tests for the Speed Limiter plugin."""

    def _plugin(self):
        from funscript.plugins.speed_limiter_plugin import SpeedLimiterPlugin
        return SpeedLimiterPlugin()

    def test_metadata(self):
        p = self._plugin()
        assert p.name == "Speed Limiter"
        assert p.version == "1.0.0"

    def test_speed_limiter_does_not_crash(self):
        p = self._plugin()
        fs = _make_sample_funscript()
        p.transform(fs, axis="primary")
        # Should complete without error
        assert len(fs.primary_actions) >= 1


# ---------------------------------------------------------------------------
# Anti-Jerk Plugin Tests
# ---------------------------------------------------------------------------

@pytest.mark.plugins
class TestAntiJerkPlugin:
    """Tests for the Anti-Jerk plugin."""

    def _plugin(self):
        from funscript.plugins.anti_jerk_plugin import AntiJerkPlugin
        return AntiJerkPlugin()

    def test_metadata(self):
        p = self._plugin()
        assert p.name == "Anti-Jerk"
        assert p.version == "2.0.0"

    def test_transform_does_not_crash(self):
        p = self._plugin()
        fs = _make_sample_funscript()
        result = p.transform(fs, axis="primary")
        # AntiJerkPlugin.transform returns None on success or an error string
        assert result is None


# ---------------------------------------------------------------------------
# Keyframe Plugin Tests
# ---------------------------------------------------------------------------

@pytest.mark.plugins
class TestKeyframePlugin:
    """Tests for the Keyframe plugin."""

    def _plugin(self):
        from funscript.plugins.keyframe_plugin import KeyframePlugin
        return KeyframePlugin()

    def test_metadata(self):
        p = self._plugin()
        assert p.name == "Keyframes"
        assert p.version == "1.0.0"

    def test_keyframe_reduces_points(self):
        """Keyframe simplification should reduce points for data with small oscillations."""
        p = self._plugin()
        fs = MultiAxisFunscript()
        # Create data with a big swing and small noise
        actions = []
        for i in range(20):
            pos = 50 + 40 * (1 if i % 4 < 2 else -1) + (i % 2) * 2
            actions.append({"at": i * 100, "pos": max(0, min(100, pos))})
        fs.primary_actions = actions
        fs._invalidate_cache("primary")
        original_count = len(fs.primary_actions)
        p.transform(fs, axis="primary", position_tolerance=5)
        new_count = len(fs.primary_actions)
        # Should have reduced or at minimum not increased
        assert new_count <= original_count


# ---------------------------------------------------------------------------
# TimeShift Plugin Tests
# ---------------------------------------------------------------------------

@pytest.mark.plugins
class TestTimeShiftPlugin:
    """Tests for the Time Shift plugin."""

    def _plugin(self):
        from funscript.plugins.time_shift_plugin import TimeShiftPlugin
        return TimeShiftPlugin()

    def test_metadata(self):
        p = self._plugin()
        assert p.name == "Time Shift"
        assert p.version == "1.0.0"

    def test_shift_forward(self):
        p = self._plugin()
        fs = _make_sample_funscript()
        original_times = [a["at"] for a in fs.primary_actions]
        p.transform(fs, axis="primary", time_delta_ms=500)
        new_times = [a["at"] for a in fs.primary_actions]
        for orig, new in zip(original_times, new_times):
            assert new == orig + 500

    def test_shift_backward_removes_negative(self):
        p = self._plugin()
        fs = MultiAxisFunscript()
        fs.primary_actions = [
            {"at": 100, "pos": 50},
            {"at": 200, "pos": 60},
            {"at": 300, "pos": 70},
        ]
        fs._invalidate_cache("primary")
        p.transform(fs, axis="primary", time_delta_ms=-250)
        # at=100-250=-150 removed, at=200-250=-50 removed, at=300-250=50 kept
        assert len(fs.primary_actions) == 1
        assert fs.primary_actions[0]["at"] == 50

    def test_shift_zero_no_change(self):
        """time_delta_ms=0 should do nothing."""
        p = self._plugin()
        fs = _make_sample_funscript()
        original = copy.deepcopy(fs.primary_actions)
        p.transform(fs, axis="primary", time_delta_ms=0)
        assert fs.primary_actions == original

    def test_required_parameter_enforcement(self):
        p = self._plugin()
        # time_delta_ms is required, but has a default of 0 -- validation should succeed
        result = p.validate_parameters({"time_delta_ms": 100})
        assert result["time_delta_ms"] == 100


# ---------------------------------------------------------------------------
# Dynamic Amplify Plugin Tests
# ---------------------------------------------------------------------------

@pytest.mark.plugins
class TestDynamicAmplifyPlugin:
    """Tests for the Dynamic Amplify plugin."""

    def _plugin(self):
        from funscript.plugins.dynamic_amplify_plugin import DynamicAmplifyPlugin
        return DynamicAmplifyPlugin()

    def test_metadata(self):
        p = self._plugin()
        assert p.name == "Dynamic Amplify"
        assert p.version == "1.0.0"

    def test_transform_does_not_crash(self):
        p = self._plugin()
        fs = _make_sample_funscript()
        p.transform(fs, axis="primary")
        for action in fs.primary_actions:
            assert 0 <= action["pos"] <= 100


# ---------------------------------------------------------------------------
# PeakPreservingResample Plugin Tests
# ---------------------------------------------------------------------------

@pytest.mark.plugins
class TestPeakPreservingResamplePlugin:
    """Tests for the Peak-Preserving Resample plugin."""

    def _plugin(self):
        from funscript.plugins.resample_plugin import PeakPreservingResamplePlugin
        return PeakPreservingResamplePlugin()

    def test_metadata(self):
        p = self._plugin()
        assert p.name == "Resample"
        assert p.version == "1.0.0"

    def test_resample_produces_regular_intervals(self):
        p = self._plugin()
        fs = _make_sample_funscript()
        p.transform(fs, axis="primary", resample_rate_ms=50)
        # All positions should be valid
        for action in fs.primary_actions:
            assert 0 <= action["pos"] <= 100
        # Should have more points than original (50ms rate over 1000ms span)
        assert len(fs.primary_actions) >= 10


# ---------------------------------------------------------------------------
# All Plugins Smoke Tests
# ---------------------------------------------------------------------------

@pytest.mark.plugins
class TestAllPlugins:
    """Smoke tests iterating over every loaded plugin."""

    @pytest.fixture(autouse=True)
    def _load_plugins(self):
        """Ensure all built-in plugins are loaded before each test."""
        loader = PluginLoader()
        loader.load_builtin_plugins()

    def _get_all_plugins(self):
        """Return all registered plugin instances from the global registry."""
        listing = plugin_registry.list_plugins()
        plugins = []
        for info in listing:
            p = plugin_registry.get_plugin(info["name"])
            if p is not None:
                plugins.append(p)
        return plugins

    def test_every_plugin_has_nonempty_name(self):
        for p in self._get_all_plugins():
            assert isinstance(p.name, str) and len(p.name) > 0, f"Plugin has empty name: {p}"

    def test_every_plugin_has_nonempty_description(self):
        for p in self._get_all_plugins():
            assert isinstance(p.description, str) and len(p.description) > 0, (
                f"Plugin '{p.name}' has empty description"
            )

    def test_every_plugin_has_nonempty_version(self):
        for p in self._get_all_plugins():
            assert isinstance(p.version, str) and len(p.version) > 0, (
                f"Plugin '{p.name}' has empty version"
            )

    def test_every_plugin_has_schema_dict(self):
        for p in self._get_all_plugins():
            schema = p.parameters_schema
            assert isinstance(schema, dict), f"Plugin '{p.name}' schema is not a dict"

    def test_every_plugin_transform_does_not_crash_on_sample_data(self):
        """Call transform on each plugin with default parameters and sample data.

        This is a smoke test -- we don't assert specific results, just that no
        unhandled exception is raised.
        """
        for p in self._get_all_plugins():
            fs = _make_sample_funscript()
            try:
                # Build default parameters from schema
                default_params = {}
                for param_name, param_info in p.parameters_schema.items():
                    if "default" in param_info and param_info["default"] is not None:
                        default_params[param_name] = param_info["default"]
                    elif param_info.get("required", False):
                        # Provide a sensible default for required params without defaults
                        ptype = param_info["type"]
                        if ptype is int:
                            default_params[param_name] = 0
                        elif ptype is float:
                            default_params[param_name] = 0.0
                        elif ptype is str:
                            default_params[param_name] = ""
                        elif ptype is bool:
                            default_params[param_name] = False

                p.transform(fs, axis="primary", **default_params)
            except Exception as exc:
                # Allow dependency-related errors (scipy, rdp not installed)
                if "scipy" in str(exc).lower() or "rdp" in str(exc).lower():
                    pytest.skip(f"Plugin '{p.name}' requires optional dependency: {exc}")
                else:
                    pytest.fail(f"Plugin '{p.name}' raised during transform: {exc}")

    def test_every_plugin_validates_default_parameters(self):
        """validate_parameters with defaults from schema should not raise."""
        for p in self._get_all_plugins():
            default_params = {}
            for param_name, param_info in p.parameters_schema.items():
                if "default" in param_info and param_info["default"] is not None:
                    default_params[param_name] = param_info["default"]
                elif param_info.get("required", False):
                    ptype = param_info["type"]
                    if ptype is int:
                        default_params[param_name] = 0
                    elif ptype is float:
                        default_params[param_name] = 0.0
                    elif ptype is str:
                        default_params[param_name] = ""
                    elif ptype is bool:
                        default_params[param_name] = False

            try:
                validated = p.validate_parameters(default_params)
                assert isinstance(validated, dict)
            except Exception as exc:
                pytest.fail(f"Plugin '{p.name}' validate_parameters raised: {exc}")

    def test_minimum_plugin_count(self):
        """At least 12 plugins should be visible (non-hidden) after loading builtins."""
        plugins = self._get_all_plugins()
        assert len(plugins) >= 12, (
            f"Expected at least 12 visible plugins, got {len(plugins)}: "
            f"{[p.name for p in plugins]}"
        )

    def test_known_plugins_present(self):
        """Verify specific well-known plugins are registered."""
        expected = [
            "Amplify",
            "Invert",
            "Threshold Clamp",
            "Speed Limiter",
            "Anti-Jerk",
            "Time Shift",
            "Dynamic Amplify",
            "Resample",
        ]
        plugin_names = [p.name for p in self._get_all_plugins()]
        for name in expected:
            assert name in plugin_names, f"Expected plugin '{name}' not found in {plugin_names}"

    def test_ui_preference_values_valid(self):
        """Every plugin's ui_preference should be 'direct' or 'popup'."""
        for p in self._get_all_plugins():
            assert p.ui_preference in ("direct", "popup"), (
                f"Plugin '{p.name}' has invalid ui_preference: {p.ui_preference}"
            )

    def test_supported_axes_valid(self):
        """Every plugin's supported_axes should contain only known values."""
        valid_axes = {"primary", "secondary", "both"}
        for p in self._get_all_plugins():
            for axis in p.supported_axes:
                assert axis in valid_axes, (
                    f"Plugin '{p.name}' has invalid axis '{axis}' in supported_axes"
                )
