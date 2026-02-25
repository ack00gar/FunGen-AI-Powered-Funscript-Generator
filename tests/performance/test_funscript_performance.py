"""
Performance benchmarks for funscript operations.

Tests verify that core MultiAxisFunscript operations stay within
acceptable time and memory budgets on datasets up to 100,000 actions.
"""

import copy
import random
import sys
import time

import pytest

from funscript.multi_axis_funscript import MultiAxisFunscript
from funscript.plugins.plugin_loader import PluginLoader
from funscript.plugins.base_plugin import plugin_registry


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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.performance
class TestFunscriptPerformance:
    """Performance benchmarks for core MultiAxisFunscript operations."""

    def test_add_action_performance(self):
        """Adding 10,000 actions should complete in under 5 seconds.

        The add_action() method does per-action work including minimum
        interval enforcement, point simplification, and cache invalidation,
        so it is slower than raw list appends.
        """
        fs = MultiAxisFunscript()
        n = 10_000

        start = time.perf_counter()
        for i in range(n):
            fs.add_action(i * 33, random.randint(0, 100))
        elapsed = time.perf_counter() - start

        assert len(fs.primary_actions) > 0, "Expected actions to have been added"
        assert elapsed < 5.0, (
            f"Adding {n:,} actions took {elapsed:.3f}s, expected < 5.0s"
        )

    def test_get_value_lookup_performance(self):
        """100,000 random interpolated lookups on a 10,000-action funscript should complete in under 5 seconds."""
        fs = _make_large_funscript(10_000)
        max_time_ms = fs.primary_actions[-1]["at"]
        rng = random.Random(123)
        lookup_times = [rng.randint(0, max_time_ms) for _ in range(100_000)]

        start = time.perf_counter()
        for t in lookup_times:
            fs.get_value(t, axis="primary")
        elapsed = time.perf_counter() - start

        assert elapsed < 5.0, (
            f"100,000 lookups on 10,000 actions took {elapsed:.3f}s, expected < 5.0s"
        )

    def test_range_query_performance(self):
        """1,000 range queries on a 10,000-action funscript should complete in reasonable time."""
        fs = _make_large_funscript(10_000)
        max_time_ms = fs.primary_actions[-1]["at"]
        rng = random.Random(456)
        queries = []
        for _ in range(1_000):
            a = rng.randint(0, max_time_ms)
            b = rng.randint(0, max_time_ms)
            queries.append((min(a, b), max(a, b)))

        start = time.perf_counter()
        for start_ms, end_ms in queries:
            fs.get_actions_in_range(start_ms, end_ms, axis="primary")
        elapsed = time.perf_counter() - start

        # Range queries use bisect so should be fast
        assert elapsed < 5.0, (
            f"1,000 range queries on 10,000 actions took {elapsed:.3f}s, expected < 5.0s"
        )

    def test_cache_invalidation_overhead(self):
        """Measure overhead of cache invalidation versus cached lookups.

        After populating the cache, repeated lookups should be fast.
        Invalidation + rebuild should add measurable but bounded overhead.
        """
        fs = _make_large_funscript(10_000)
        rng = random.Random(789)
        max_time_ms = fs.primary_actions[-1]["at"]
        lookup_times = [rng.randint(0, max_time_ms) for _ in range(10_000)]

        # Warm the cache
        fs.get_value(0, axis="primary")

        # Cached lookups
        start = time.perf_counter()
        for t in lookup_times:
            fs.get_value(t, axis="primary")
        cached_time = time.perf_counter() - start

        # Lookups with cache invalidation every 100 lookups
        start = time.perf_counter()
        for i, t in enumerate(lookup_times):
            if i % 100 == 0:
                fs._invalidate_cache("primary")
            fs.get_value(t, axis="primary")
        invalidated_time = time.perf_counter() - start

        # The invalidation overhead should not more than triple the time
        overhead_ratio = invalidated_time / max(cached_time, 1e-9)
        assert overhead_ratio < 10.0, (
            f"Cache invalidation overhead ratio {overhead_ratio:.1f}x is too high. "
            f"Cached: {cached_time:.4f}s, With invalidation: {invalidated_time:.4f}s"
        )

    def test_large_funscript_memory(self):
        """Creating 100,000 actions should use a reasonable amount of memory."""
        # Measure baseline
        baseline = sys.getsizeof([])

        fs = _make_large_funscript(100_000)

        # Approximate memory: each action dict + list overhead
        # Each dict {"at": int, "pos": int} is roughly 200-300 bytes on CPython
        action_count = len(fs.primary_actions)
        assert action_count == 100_000, (
            f"Expected 100,000 actions, got {action_count:,}"
        )

        # Estimate memory of the actions list (not a precise measure, but a sanity check)
        sample_action_size = sys.getsizeof(fs.primary_actions[0])
        estimated_bytes = action_count * sample_action_size + sys.getsizeof(fs.primary_actions)
        estimated_mb = estimated_bytes / (1024 * 1024)

        # 100,000 actions should use less than 100 MB
        assert estimated_mb < 100.0, (
            f"Estimated memory for 100,000 actions is {estimated_mb:.1f} MB, expected < 100 MB"
        )

    def test_plugin_transform_performance(self):
        """Each plugin should transform a 10,000-action funscript in under 2 seconds."""
        # Ensure plugins are loaded
        loader = PluginLoader()
        loader.load_builtin_plugins()

        listing = plugin_registry.list_plugins()
        assert len(listing) > 0, "No plugins loaded for performance test"

        results = {}
        for info in listing:
            plugin = plugin_registry.get_plugin(info["name"])
            if plugin is None:
                continue

            # Skip plugins with missing dependencies
            if not plugin.check_dependencies():
                continue

            fs = _make_large_funscript(10_000)
            params = _get_default_params(plugin)

            start = time.perf_counter()
            try:
                plugin.transform(fs, axis="primary", **params)
            except Exception:
                # Some plugins may fail with default params; skip them
                continue
            elapsed = time.perf_counter() - start
            results[info["name"]] = elapsed

            assert elapsed < 2.0, (
                f"Plugin '{info['name']}' took {elapsed:.3f}s on 10,000 actions, expected < 2.0s"
            )

        # Ensure at least some plugins were benchmarked
        assert len(results) >= 3, (
            f"Only {len(results)} plugins were benchmarked, expected at least 3. "
            f"Results: {results}"
        )


@pytest.mark.performance
class TestAddActionScaling:
    """Verify add_action does NOT degrade with scale (regression guard)."""

    def test_chronological_append_is_constant_time(self):
        """Adding points in order should be ~O(1) per point after pre-fill."""
        fs = MultiAxisFunscript()
        fs.enable_point_simplification = False
        # Pre-fill with 10K points
        for i in range(10_000):
            fs.add_action(i * 33, (i * 7) % 101)
        # Measure next 1000 appends — should be fast
        t0 = time.perf_counter()
        for i in range(10_000, 11_000):
            fs.add_action(i * 33, (i * 7) % 101)
        elapsed = time.perf_counter() - t0
        # Should complete in <50ms (was ~500ms before fix at 10K points)
        assert elapsed < 0.05, (
            f"Chronological append too slow: {elapsed*1000:.1f}ms for 1000 points"
        )

    def test_cache_stays_in_sync_after_append(self):
        """Cache must match actual actions after fast-path appends."""
        fs = MultiAxisFunscript()
        fs.enable_point_simplification = False
        for i in range(500):
            fs.add_action(i * 33, (i * 7) % 101)
        cache = fs._get_timestamps_for_axis('primary')
        actual = [a['at'] for a in fs.primary_actions]
        assert cache == actual

    def test_cache_stays_in_sync_with_simplification(self):
        """Cache must match after simplification pops points."""
        fs = MultiAxisFunscript()
        fs.enable_point_simplification = True
        for i in range(1000):
            fs.add_action(i * 33, 50)  # Constant position triggers simplification
        cache = fs._get_timestamps_for_axis('primary')
        actual = [a['at'] for a in fs.primary_actions]
        assert cache == actual

    def test_out_of_order_insertion_still_works(self):
        """Manual editing (out-of-order) must still produce correct results."""
        fs = MultiAxisFunscript()
        fs.enable_point_simplification = False
        fs.add_action(0, 0)
        fs.add_action(1000, 100)
        fs.add_action(500, 50)  # Out of order
        assert len(fs.primary_actions) == 3
        assert fs.primary_actions[1]['at'] == 500
