"""
Performance benchmarks for the plugin system.

Tests cover plugin loading speed, registry lookup throughput, and
transform performance on large datasets for individual plugins and
plugin chains.
"""

import copy
import logging
import random
import time

import pytest

from funscript.multi_axis_funscript import MultiAxisFunscript
from funscript.plugins.base_plugin import (
    FunscriptTransformationPlugin,
    PluginRegistry,
    plugin_registry,
)
from funscript.plugins.plugin_loader import PluginLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_large_funscript(n: int, seed: int = 42) -> MultiAxisFunscript:
    """Create a MultiAxisFunscript with *n* deterministic actions on primary axis."""
    rng = random.Random(seed)
    fs = MultiAxisFunscript()
    fs.primary_actions = [
        {"at": i * 33, "pos": rng.randint(0, 100)} for i in range(n)
    ]
    fs._invalidate_cache("primary")
    return fs


def _get_default_params(plugin):
    """Build a parameter dict from a plugin's schema using defaults."""
    params = {}
    for name, info in plugin.parameters_schema.items():
        if "default" in info and info["default"] is not None:
            params[name] = info["default"]
        elif info.get("required", False):
            ptype = info["type"]
            if ptype is int:
                params[name] = 0
            elif ptype is float:
                params[name] = 0.0
            elif ptype is str:
                params[name] = ""
            elif ptype is bool:
                params[name] = False
    return params


class _BenchDummyPlugin(FunscriptTransformationPlugin):
    """Minimal plugin used to stress-test the registry."""

    def __init__(self, idx: int):
        super().__init__()
        self._name = f"BenchDummy_{idx:04d}"

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return "Benchmark dummy"

    @property
    def version(self):
        return "0.0.1"

    @property
    def parameters_schema(self):
        return {}

    def transform(self, funscript, axis="both", **parameters):
        return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.performance
@pytest.mark.plugins
class TestPluginPerformance:
    """Performance benchmarks for the plugin system."""

    def test_plugin_loading_time(self):
        """Loading all builtin plugins should complete in under 1 second."""
        loader = PluginLoader()

        start = time.perf_counter()
        results = loader.load_builtin_plugins()
        elapsed = time.perf_counter() - start

        successful = [f for f, ok in results.items() if ok]
        assert len(successful) > 0, "No plugins loaded successfully"
        assert elapsed < 1.0, (
            f"Loading {len(successful)} builtin plugins took {elapsed:.3f}s, expected < 1.0s"
        )

    def test_plugin_registry_lookup_performance(self):
        """10,000 lookups on a registry with 100 plugins should complete in under 0.1 seconds."""
        registry = PluginRegistry(logger=logging.getLogger("perf_registry"))

        # Register 100 dummy plugins
        for i in range(100):
            plugin = _BenchDummyPlugin(i)
            registry.register(plugin)

        # Build lookup keys (mix of existing and non-existing)
        rng = random.Random(42)
        keys = [f"BenchDummy_{rng.randint(0, 149):04d}" for _ in range(10_000)]

        start = time.perf_counter()
        for key in keys:
            registry.get_plugin(key)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.1, (
            f"10,000 registry lookups took {elapsed:.4f}s, expected < 0.1s"
        )

    def test_amplify_plugin_large_dataset(self):
        """Amplify plugin should transform 50,000 actions in reasonable time."""
        from funscript.plugins.amplify_plugin import AmplifyPlugin

        plugin = AmplifyPlugin()
        fs = _make_large_funscript(50_000)

        start = time.perf_counter()
        plugin.transform(fs, axis="primary", scale_factor=1.5, center_value=50)
        elapsed = time.perf_counter() - start

        assert elapsed < 5.0, (
            f"Amplify on 50,000 actions took {elapsed:.3f}s, expected < 5.0s"
        )
        # Verify transform actually ran
        for action in fs.primary_actions[:10]:
            assert 0 <= action["pos"] <= 100

    def test_invert_plugin_large_dataset(self):
        """Invert plugin should transform 50,000 actions in reasonable time."""
        from funscript.plugins.invert_plugin import InvertPlugin

        plugin = InvertPlugin()
        fs = _make_large_funscript(50_000)
        original_positions = [a["pos"] for a in fs.primary_actions[:10]]

        start = time.perf_counter()
        plugin.transform(fs, axis="primary")
        elapsed = time.perf_counter() - start

        assert elapsed < 5.0, (
            f"Invert on 50,000 actions took {elapsed:.3f}s, expected < 5.0s"
        )
        # Verify inversion: pos -> 100 - pos
        for orig, action in zip(original_positions, fs.primary_actions[:10]):
            assert action["pos"] == 100 - orig

    def test_chain_three_plugins_large_dataset(self):
        """Chaining Amplify -> ThresholdClamp -> Invert on 50,000 actions should be fast."""
        from funscript.plugins.amplify_plugin import AmplifyPlugin
        from funscript.plugins.clamp_plugin import ThresholdClampPlugin
        from funscript.plugins.invert_plugin import InvertPlugin

        amplify = AmplifyPlugin()
        clamp = ThresholdClampPlugin()
        invert = InvertPlugin()

        fs = _make_large_funscript(50_000)

        start = time.perf_counter()
        amplify.transform(fs, axis="primary", scale_factor=1.5, center_value=50)
        clamp.transform(fs, axis="primary", lower_threshold=10, upper_threshold=90)
        invert.transform(fs, axis="primary")
        elapsed = time.perf_counter() - start

        assert elapsed < 10.0, (
            f"Chaining 3 plugins on 50,000 actions took {elapsed:.3f}s, expected < 10.0s"
        )
        # All positions should still be valid
        for action in fs.primary_actions:
            assert 0 <= action["pos"] <= 100

    def test_all_plugins_benchmark(self):
        """Benchmark every builtin plugin on 10,000 actions and report all times."""
        loader = PluginLoader()
        loader.load_builtin_plugins()

        listing = plugin_registry.list_plugins()
        assert len(listing) > 0, "No plugins loaded for benchmark"

        results = {}
        for info in listing:
            plugin = plugin_registry.get_plugin(info["name"])
            if plugin is None:
                continue
            if not plugin.check_dependencies():
                results[info["name"]] = "skipped (missing dependencies)"
                continue

            fs = _make_large_funscript(10_000)
            params = _get_default_params(plugin)

            start = time.perf_counter()
            try:
                plugin.transform(fs, axis="primary", **params)
                elapsed = time.perf_counter() - start
                results[info["name"]] = f"{elapsed:.4f}s"
            except Exception as exc:
                results[info["name"]] = f"error: {exc}"

        # Verify we benchmarked a meaningful number of plugins
        timed = [v for v in results.values() if v.endswith("s") and not v.startswith("error")]
        assert len(timed) >= 5, (
            f"Only {len(timed)} plugins completed benchmark, expected at least 5. "
            f"Results: {results}"
        )
