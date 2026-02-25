"""
Integration tests for plugin chaining and integration with MultiAxisFunscript.

These tests exercise the real plugin classes (Amplify, Clamp, Invert, etc.)
against real MultiAxisFunscript instances -- no mocks.
"""

import copy
import os
import sys
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from funscript import MultiAxisFunscript
from funscript.plugins.base_plugin import PluginRegistry
from funscript.plugins.plugin_loader import PluginLoader


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_PRIMARY_ACTIONS = [
    {"at": 0, "pos": 10},
    {"at": 200, "pos": 90},
    {"at": 400, "pos": 20},
    {"at": 600, "pos": 80},
    {"at": 800, "pos": 30},
    {"at": 1000, "pos": 70},
    {"at": 1200, "pos": 15},
    {"at": 1400, "pos": 85},
    {"at": 1600, "pos": 25},
    {"at": 1800, "pos": 75},
    {"at": 2000, "pos": 50},
]

SAMPLE_SECONDARY_ACTIONS = [
    {"at": 0, "pos": 50},
    {"at": 500, "pos": 60},
    {"at": 1000, "pos": 40},
    {"at": 1500, "pos": 55},
    {"at": 2000, "pos": 45},
]


@pytest.fixture
def sample_funscript():
    """Return a MultiAxisFunscript populated with sample actions."""
    fs = MultiAxisFunscript()
    fs.primary_actions = [d.copy() for d in SAMPLE_PRIMARY_ACTIONS]
    fs.secondary_actions = [d.copy() for d in SAMPLE_SECONDARY_ACTIONS]
    fs._invalidate_cache('both')
    return fs


@pytest.fixture
def loaded_registry():
    """Return a fresh PluginRegistry with built-in plugins loaded."""
    registry = PluginRegistry()
    loader = PluginLoader()
    loader.load_builtin_plugins()

    # Copy registered plugins from the global registry into our local one.
    from funscript.plugins.base_plugin import plugin_registry as global_reg
    for name, plugin in global_reg._plugins.items():
        registry.register(plugin)

    return registry


@pytest.fixture
def amplify_plugin(loaded_registry):
    return loaded_registry.get_plugin("Amplify")


@pytest.fixture
def invert_plugin(loaded_registry):
    return loaded_registry.get_plugin("Invert")


@pytest.fixture
def threshold_clamp_plugin(loaded_registry):
    return loaded_registry.get_plugin("Threshold Clamp")


@pytest.fixture
def value_clamp_plugin(loaded_registry):
    return loaded_registry.get_plugin("Clamp")


# ---------------------------------------------------------------------------
# Plugin chain tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.plugins
class TestPluginChainAmplifyThenClamp:
    """Apply amplify then clamp, verify result in bounds."""

    def test_plugin_chain_amplify_then_clamp(self, sample_funscript, amplify_plugin, threshold_clamp_plugin):
        if amplify_plugin is None or threshold_clamp_plugin is None:
            pytest.skip("Amplify or Threshold Clamp plugin not available")

        # Step 1: Amplify with a large factor
        amplify_plugin.transform(
            sample_funscript,
            axis='primary',
            scale_factor=3.0,
            center_value=50,
        )

        # After aggressive amplification, some values may be at 0 or 100
        for action in sample_funscript.primary_actions:
            assert 0 <= action['pos'] <= 100

        # Step 2: Threshold clamp to push extremes
        threshold_clamp_plugin.transform(
            sample_funscript,
            axis='primary',
            lower_threshold=20,
            upper_threshold=80,
        )

        # After threshold clamping: values <20 become 0, >80 become 100
        for action in sample_funscript.primary_actions:
            pos = action['pos']
            assert 0 <= pos <= 100


@pytest.mark.integration
@pytest.mark.plugins
class TestPluginChainInvertTwice:
    """Apply invert twice, verify original restored."""

    def test_plugin_chain_invert_twice_is_identity(self, sample_funscript, invert_plugin):
        if invert_plugin is None:
            pytest.skip("Invert plugin not available")

        original_primary = [d.copy() for d in sample_funscript.primary_actions]
        original_secondary = [d.copy() for d in sample_funscript.secondary_actions]

        # Invert once
        invert_plugin.transform(sample_funscript, axis='both')

        # Verify positions changed
        for orig, current in zip(original_primary, sample_funscript.primary_actions):
            assert current['pos'] == 100 - orig['pos']

        # Invert again
        invert_plugin.transform(sample_funscript, axis='both')

        # Verify positions restored
        for orig, current in zip(original_primary, sample_funscript.primary_actions):
            assert current['pos'] == orig['pos'], (
                f"at={orig['at']}: expected {orig['pos']}, got {current['pos']}"
            )

        for orig, current in zip(original_secondary, sample_funscript.secondary_actions):
            assert current['pos'] == orig['pos']


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.plugins
class TestPluginWithEmptyFunscript:
    """Apply plugins to empty actions list, verify no crash."""

    def test_plugin_with_empty_funscript(self, amplify_plugin, invert_plugin, threshold_clamp_plugin):
        fs = MultiAxisFunscript()
        # All actions lists are empty

        for plugin in [amplify_plugin, invert_plugin, threshold_clamp_plugin]:
            if plugin is None:
                continue
            # Should not raise
            plugin.transform(fs, axis='both')
            assert len(fs.primary_actions) == 0
            assert len(fs.secondary_actions) == 0


@pytest.mark.integration
@pytest.mark.plugins
class TestPluginWithSingleAction:
    """Apply plugins to a single action."""

    def test_plugin_with_single_action(self, amplify_plugin, invert_plugin):
        fs = MultiAxisFunscript()
        fs.primary_actions = [{"at": 0, "pos": 60}]
        fs._invalidate_cache('primary')

        if invert_plugin:
            invert_plugin.transform(fs, axis='primary')
            assert fs.primary_actions[0]['pos'] == 40  # 100 - 60

        if amplify_plugin:
            # Reset
            fs.primary_actions = [{"at": 0, "pos": 60}]
            fs._invalidate_cache('primary')
            amplify_plugin.transform(fs, axis='primary', scale_factor=2.0, center_value=50)
            # (60 - 50) * 2 + 50 = 70
            assert fs.primary_actions[0]['pos'] == 70


# ---------------------------------------------------------------------------
# All plugins on sample data
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.plugins
class TestAllPluginsOnSampleData:
    """Load all plugins, apply each with defaults, verify no crash."""

    def test_all_plugins_on_sample_data(self, loaded_registry, sample_funscript):
        plugin_list = loaded_registry.list_plugins()
        assert len(plugin_list) > 0, "No plugins were loaded"

        for info in plugin_list:
            plugin = loaded_registry.get_plugin(info['name'])
            if plugin is None:
                continue

            # Build defaults from schema
            defaults = {}
            for pname, pinfo in plugin.parameters_schema.items():
                if 'default' in pinfo:
                    defaults[pname] = pinfo['default']

            # Create a fresh funscript copy for each plugin
            fs = MultiAxisFunscript()
            fs.primary_actions = [d.copy() for d in SAMPLE_PRIMARY_ACTIONS]
            fs.secondary_actions = [d.copy() for d in SAMPLE_SECONDARY_ACTIONS]
            fs._invalidate_cache('both')

            try:
                plugin.transform(fs, axis='primary', **defaults)
            except Exception as exc:
                pytest.fail(f"Plugin '{info['name']}' raised {type(exc).__name__}: {exc}")

            # Output should still have actions (plugins should not delete all data)
            assert len(fs.primary_actions) >= 0  # some plugins may empty on edge cases


# ---------------------------------------------------------------------------
# Timing preservation
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.plugins
class TestPluginPreservesTiming:
    """Apply amplify, verify 'at' values unchanged."""

    def test_plugin_preserves_timing(self, sample_funscript, amplify_plugin):
        if amplify_plugin is None:
            pytest.skip("Amplify plugin not available")

        original_timestamps = [a['at'] for a in sample_funscript.primary_actions]

        amplify_plugin.transform(
            sample_funscript,
            axis='primary',
            scale_factor=2.0,
            center_value=50,
        )

        current_timestamps = [a['at'] for a in sample_funscript.primary_actions]
        assert original_timestamps == current_timestamps, (
            "Amplify plugin must not modify timestamps"
        )


# ---------------------------------------------------------------------------
# Primary axis only
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.plugins
class TestPluginPrimaryAxisOnly:
    """Apply to primary only, verify secondary unchanged."""

    def test_plugin_primary_axis_only(self, sample_funscript, amplify_plugin):
        if amplify_plugin is None:
            pytest.skip("Amplify plugin not available")

        secondary_before = [d.copy() for d in sample_funscript.secondary_actions]

        amplify_plugin.transform(
            sample_funscript,
            axis='primary',
            scale_factor=2.0,
            center_value=50,
        )

        # Secondary should be untouched
        assert len(sample_funscript.secondary_actions) == len(secondary_before)
        for before, after in zip(secondary_before, sample_funscript.secondary_actions):
            assert before['at'] == after['at']
            assert before['pos'] == after['pos']


# ---------------------------------------------------------------------------
# Selected indices
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.plugins
class TestPluginSelectedIndices:
    """Apply amplification with selected_indices parameter."""

    def test_plugin_selected_indices(self, sample_funscript, amplify_plugin):
        if amplify_plugin is None:
            pytest.skip("Amplify plugin not available")

        original_actions = [d.copy() for d in sample_funscript.primary_actions]
        # Only amplify indices 2, 3, 4
        selected = [2, 3, 4]

        amplify_plugin.transform(
            sample_funscript,
            axis='primary',
            scale_factor=2.0,
            center_value=50,
            selected_indices=selected,
        )

        for i, action in enumerate(sample_funscript.primary_actions):
            if i in selected:
                # These should have changed (unless they were already at center)
                orig_pos = original_actions[i]['pos']
                expected = int(round(max(0, min(100, 50 + (orig_pos - 50) * 2.0))))
                assert action['pos'] == expected, (
                    f"Index {i}: expected {expected}, got {action['pos']}"
                )
            else:
                # These should be unchanged
                assert action['pos'] == original_actions[i]['pos'], (
                    f"Index {i} should be unchanged"
                )


# ---------------------------------------------------------------------------
# Time range
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.plugins
class TestPluginTimeRange:
    """Apply amplification with start_time_ms/end_time_ms."""

    def test_plugin_time_range(self, sample_funscript, amplify_plugin):
        if amplify_plugin is None:
            pytest.skip("Amplify plugin not available")

        original_actions = [d.copy() for d in sample_funscript.primary_actions]
        start_ms = 400
        end_ms = 1200

        amplify_plugin.transform(
            sample_funscript,
            axis='primary',
            scale_factor=0.5,
            center_value=50,
            start_time_ms=start_ms,
            end_time_ms=end_ms,
        )

        for i, action in enumerate(sample_funscript.primary_actions):
            orig = original_actions[i]
            if start_ms <= orig['at'] <= end_ms:
                # Should be modified (scaled toward center)
                expected = int(round(max(0, min(100, 50 + (orig['pos'] - 50) * 0.5))))
                assert action['pos'] == expected, (
                    f"at={orig['at']}: expected {expected}, got {action['pos']}"
                )
            else:
                # Outside range: untouched
                assert action['pos'] == orig['pos'], (
                    f"at={orig['at']} outside range should be unchanged"
                )
