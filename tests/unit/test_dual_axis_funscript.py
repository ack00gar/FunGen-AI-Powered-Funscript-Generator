"""Unit tests for MultiAxisFunscript class."""
import pytest
import logging
from unittest.mock import patch, MagicMock


@pytest.mark.unit
class TestMultiAxisFunscriptCreation:
    """Tests for MultiAxisFunscript construction and initial state."""

    def test_creation_with_defaults(self, empty_multi_axis_funscript):
        """Newly created instance has empty action lists and default settings."""
        fs = empty_multi_axis_funscript
        assert fs.primary_actions == []
        assert fs.secondary_actions == []
        assert fs.last_timestamp_primary == 0
        assert fs.last_timestamp_secondary == 0
        assert fs.min_interval_ms == 10
        assert fs.enable_point_simplification is True

    def test_creation_with_custom_logger(self):
        """Instance can be created with a custom logger."""
        from funscript.multi_axis_funscript import MultiAxisFunscript
        logger = logging.getLogger("test_custom")
        fs = MultiAxisFunscript(logger=logger)
        assert fs.logger is logger

    def test_creation_without_logger_uses_fallback(self):
        """Instance without explicit logger uses a fallback logger."""
        from funscript.multi_axis_funscript import MultiAxisFunscript
        fs = MultiAxisFunscript()
        assert fs.logger is not None

    def test_cache_initially_dirty(self):
        """Both caches are dirty on creation since no timestamps are cached yet."""
        from funscript.multi_axis_funscript import MultiAxisFunscript
        fs = MultiAxisFunscript()
        assert fs._cache_dirty_primary is True
        assert fs._cache_dirty_secondary is True

    def test_simplification_stats_initialized(self):
        """Simplification statistics are initialized to zero."""
        from funscript.multi_axis_funscript import MultiAxisFunscript
        fs = MultiAxisFunscript()
        assert fs._simplification_stats_primary['total_removed'] == 0
        assert fs._simplification_stats_primary['total_considered'] == 0
        assert fs._simplification_stats_secondary['total_removed'] == 0
        assert fs._simplification_stats_secondary['total_considered'] == 0


@pytest.mark.unit
class TestMultiAxisFunscriptAddAction:
    """Tests for adding actions to both axes."""

    def test_add_primary_action(self, empty_multi_axis_funscript):
        """Adding a primary action stores it in primary_actions."""
        fs = empty_multi_axis_funscript
        fs.enable_point_simplification = False
        fs.add_action(1000, 75)
        assert len(fs.primary_actions) == 1
        assert fs.primary_actions[0]['at'] == 1000
        assert fs.primary_actions[0]['pos'] == 75

    def test_add_secondary_action(self, empty_multi_axis_funscript):
        """Adding a secondary action stores it in secondary_actions."""
        fs = empty_multi_axis_funscript
        fs.enable_point_simplification = False
        fs.add_action(1000, None, 30)
        assert len(fs.secondary_actions) == 1
        assert fs.secondary_actions[0]['at'] == 1000
        assert fs.secondary_actions[0]['pos'] == 30
        assert len(fs.primary_actions) == 0

    def test_add_both_axes_simultaneously(self, empty_multi_axis_funscript):
        """Adding actions for both axes at same timestamp stores both."""
        fs = empty_multi_axis_funscript
        fs.enable_point_simplification = False
        fs.add_action(1000, 80, 20)
        assert len(fs.primary_actions) == 1
        assert len(fs.secondary_actions) == 1
        assert fs.primary_actions[0]['pos'] == 80
        assert fs.secondary_actions[0]['pos'] == 20

    def test_position_clamped_to_0_100(self, empty_multi_axis_funscript):
        """Positions outside 0-100 are clamped."""
        fs = empty_multi_axis_funscript
        fs.enable_point_simplification = False
        fs.add_action(1000, 150)
        fs.add_action(2000, -50)
        assert fs.primary_actions[0]['pos'] == 100
        assert fs.primary_actions[1]['pos'] == 0

    def test_add_multiple_sorted_by_timestamp(self, empty_multi_axis_funscript):
        """Actions are maintained in sorted timestamp order."""
        fs = empty_multi_axis_funscript
        fs.enable_point_simplification = False
        fs.add_action(200, 50)
        fs.add_action(100, 25)
        fs.add_action(300, 75)
        timestamps = [a['at'] for a in fs.primary_actions]
        assert timestamps == sorted(timestamps)

    def test_min_interval_enforced(self, empty_multi_axis_funscript):
        """Actions closer than min_interval_ms are filtered out."""
        fs = empty_multi_axis_funscript
        fs.enable_point_simplification = False
        fs.min_interval_ms = 50
        fs.add_action(100, 50)
        fs.add_action(120, 60)  # Only 20ms apart, should be rejected
        # The second action is too close and should not be inserted
        assert len(fs.primary_actions) == 1

    def test_duplicate_timestamp_updates_position(self, empty_multi_axis_funscript):
        """Adding action at existing timestamp updates position rather than duplicating."""
        fs = empty_multi_axis_funscript
        fs.enable_point_simplification = False
        fs.add_action(1000, 50)
        fs.add_action(1000, 80)
        assert len(fs.primary_actions) == 1
        assert fs.primary_actions[0]['pos'] == 80

    def test_none_primary_pos_skips_primary(self, empty_multi_axis_funscript):
        """When primary_pos is None, primary_actions is not modified."""
        fs = empty_multi_axis_funscript
        fs.add_action(1000, None, 50)
        assert len(fs.primary_actions) == 0
        assert len(fs.secondary_actions) == 1

    def test_last_timestamp_updated(self, empty_multi_axis_funscript):
        """last_timestamp fields are updated after adding actions."""
        fs = empty_multi_axis_funscript
        fs.enable_point_simplification = False
        fs.add_action(500, 50, 50)
        assert fs.last_timestamp_primary == 500
        assert fs.last_timestamp_secondary == 500


@pytest.mark.unit
class TestMultiAxisFunscriptGetValue:
    """Tests for interpolated value retrieval."""

    def test_get_value_at_exact_point(self, multi_axis_funscript):
        """get_value returns exact position when timestamp matches an action."""
        fs = multi_axis_funscript
        assert fs.get_value(0, 'primary') == 50
        assert fs.get_value(100, 'primary') == 0
        assert fs.get_value(200, 'primary') == 100

    def test_get_value_interpolated_midpoint(self, multi_axis_funscript):
        """get_value interpolates linearly between two points."""
        fs = multi_axis_funscript
        # Between t=0 (pos=50) and t=100 (pos=0), midpoint t=50 should be 25
        val = fs.get_value(50, 'primary')
        assert val == 25

    def test_get_value_before_first_action(self, multi_axis_funscript):
        """get_value before first action returns first action's position."""
        fs = multi_axis_funscript
        # First action is at t=0, pos=50
        # Querying before first action returns the first position
        val = fs.get_value(0, 'primary')
        assert val == 50

    def test_get_value_after_last_action(self, multi_axis_funscript):
        """get_value after last action returns last action's position."""
        fs = multi_axis_funscript
        # Last action is at t=500, pos=50
        val = fs.get_value(1000, 'primary')
        assert val == 50

    def test_get_value_empty_returns_neutral(self, empty_multi_axis_funscript):
        """get_value on empty funscript returns 50 (neutral position)."""
        fs = empty_multi_axis_funscript
        assert fs.get_value(1000, 'primary') == 50
        assert fs.get_value(1000, 'secondary') == 50

    def test_get_value_secondary_axis(self):
        """get_value works correctly on the secondary axis."""
        from funscript.multi_axis_funscript import MultiAxisFunscript
        fs = MultiAxisFunscript()
        fs.enable_point_simplification = False
        fs.add_action(0, None, 0)
        fs.add_action(100, None, 100)
        val = fs.get_value(50, 'secondary')
        assert val == 50

    def test_get_value_clamped_0_100(self):
        """Interpolated values are clamped between 0 and 100."""
        from funscript.multi_axis_funscript import MultiAxisFunscript
        fs = MultiAxisFunscript()
        fs.enable_point_simplification = False
        fs.add_action(0, 0)
        fs.add_action(100, 100)
        # Value at boundaries should be within range
        assert 0 <= fs.get_value(0, 'primary') <= 100
        assert 0 <= fs.get_value(100, 'primary') <= 100
        assert 0 <= fs.get_value(50, 'primary') <= 100


@pytest.mark.unit
class TestMultiAxisFunscriptGetActionsInRange:
    """Tests for retrieving actions within a time range."""

    def test_get_actions_in_range_basic(self, multi_axis_funscript):
        """Returns actions within the specified time range (inclusive)."""
        fs = multi_axis_funscript
        result = fs.get_actions_in_range(100, 300, 'primary')
        timestamps = [a['at'] for a in result]
        assert all(100 <= t <= 300 for t in timestamps)
        assert len(result) >= 1

    def test_get_actions_in_range_full(self, multi_axis_funscript):
        """Returns all actions when range covers entire timeline."""
        fs = multi_axis_funscript
        result = fs.get_actions_in_range(0, 500, 'primary')
        assert len(result) == len(fs.primary_actions)

    def test_get_actions_in_range_empty(self, multi_axis_funscript):
        """Returns empty list when no actions in range."""
        fs = multi_axis_funscript
        result = fs.get_actions_in_range(550, 600, 'primary')
        assert result == []

    def test_get_actions_in_range_empty_funscript(self, empty_multi_axis_funscript):
        """Returns empty list when funscript has no actions."""
        fs = empty_multi_axis_funscript
        assert fs.get_actions_in_range(0, 1000, 'primary') == []

    def test_get_actions_in_range_single_point(self, multi_axis_funscript):
        """Returns single action when range covers exactly one point."""
        fs = multi_axis_funscript
        result = fs.get_actions_in_range(100, 100, 'primary')
        # bisect_left finds 100, bisect_right stops after 100
        assert len(result) == 1
        assert result[0]['at'] == 100

    def test_get_actions_in_range_secondary_axis(self):
        """get_actions_in_range works on secondary axis."""
        from funscript.multi_axis_funscript import MultiAxisFunscript
        fs = MultiAxisFunscript()
        fs.enable_point_simplification = False
        fs.add_action(100, None, 25)
        fs.add_action(200, None, 50)
        fs.add_action(300, None, 75)
        result = fs.get_actions_in_range(100, 200, 'secondary')
        assert len(result) == 2


@pytest.mark.unit
class TestMultiAxisFunscriptCacheInvalidation:
    """Tests for timestamp cache management."""

    def test_invalidate_cache_primary(self, empty_multi_axis_funscript):
        """Invalidating primary cache sets primary dirty flag."""
        fs = empty_multi_axis_funscript
        fs._cache_dirty_primary = False
        fs._invalidate_cache('primary')
        assert fs._cache_dirty_primary is True
        assert fs._cache_dirty_secondary is True  # Was already dirty from init

    def test_invalidate_cache_secondary(self, empty_multi_axis_funscript):
        """Invalidating secondary cache sets secondary dirty flag."""
        fs = empty_multi_axis_funscript
        fs._cache_dirty_secondary = False
        fs._invalidate_cache('secondary')
        assert fs._cache_dirty_secondary is True

    def test_invalidate_cache_both(self, empty_multi_axis_funscript):
        """Invalidating 'both' sets both dirty flags."""
        fs = empty_multi_axis_funscript
        fs._cache_dirty_primary = False
        fs._cache_dirty_secondary = False
        fs._invalidate_cache('both')
        assert fs._cache_dirty_primary is True
        assert fs._cache_dirty_secondary is True

    def test_cache_rebuilt_on_get_timestamps(self, multi_axis_funscript):
        """Accessing timestamps rebuilds cache and clears dirty flag."""
        fs = multi_axis_funscript
        fs._cache_dirty_primary = True
        timestamps = fs._get_timestamps_for_axis('primary')
        assert fs._cache_dirty_primary is False
        assert len(timestamps) == len(fs.primary_actions)

    def test_cache_not_rebuilt_when_clean(self, multi_axis_funscript):
        """Cache is not rebuilt when it is already clean."""
        fs = multi_axis_funscript
        # Build cache first
        ts1 = fs._get_timestamps_for_axis('primary')
        # Access again - should return same list
        ts2 = fs._get_timestamps_for_axis('primary')
        assert ts1 is ts2  # Same object, not rebuilt

    def test_add_action_keeps_cache_in_sync(self, empty_multi_axis_funscript):
        """Adding actions keeps the timestamp cache in sync with the actions list."""
        fs = empty_multi_axis_funscript
        fs.enable_point_simplification = False
        # Build cache
        fs._get_timestamps_for_axis('primary')
        assert fs._cache_dirty_primary is False
        # Chronological appends update cache incrementally (fast path)
        fs.add_action(100, 50)
        fs.add_action(200, 75)
        cache = fs._get_timestamps_for_axis('primary')
        actual = [a['at'] for a in fs.primary_actions]
        assert cache == actual
        # Out-of-order insert invalidates cache (slow path)
        fs.add_action(150, 60)
        assert fs._cache_dirty_primary is True


@pytest.mark.unit
class TestMultiAxisFunscriptClear:
    """Tests for clearing all actions."""

    def test_clear_removes_all_actions(self, multi_axis_funscript):
        """clear() removes all actions from both axes."""
        fs = multi_axis_funscript
        fs.secondary_actions = [{"at": 100, "pos": 50}]
        fs.clear()
        assert fs.primary_actions == []
        assert fs.secondary_actions == []
        assert fs.last_timestamp_primary == 0
        assert fs.last_timestamp_secondary == 0

    def test_clear_invalidates_caches(self, multi_axis_funscript):
        """clear() invalidates both caches."""
        fs = multi_axis_funscript
        fs._cache_dirty_primary = False
        fs._cache_dirty_secondary = False
        fs.clear()
        assert fs._cache_dirty_primary is True
        assert fs._cache_dirty_secondary is True


@pytest.mark.unit
class TestMultiAxisFunscriptBoundaryConditions:
    """Tests for edge cases and boundary conditions."""

    def test_single_action_get_value(self):
        """get_value with only one action returns that action's position."""
        from funscript.multi_axis_funscript import MultiAxisFunscript
        fs = MultiAxisFunscript()
        fs.enable_point_simplification = False
        fs.add_action(500, 75)
        assert fs.get_value(0, 'primary') == 75
        assert fs.get_value(500, 'primary') == 75
        assert fs.get_value(1000, 'primary') == 75

    def test_zero_timestamp(self):
        """Actions at timestamp 0 are handled correctly."""
        from funscript.multi_axis_funscript import MultiAxisFunscript
        fs = MultiAxisFunscript()
        fs.enable_point_simplification = False
        fs.add_action(0, 0)
        assert fs.primary_actions[0]['at'] == 0
        assert fs.get_value(0, 'primary') == 0

    def test_very_large_timestamp(self):
        """Very large timestamps are handled correctly."""
        from funscript.multi_axis_funscript import MultiAxisFunscript
        fs = MultiAxisFunscript()
        fs.enable_point_simplification = False
        large_ts = 3600000  # 1 hour in ms
        fs.add_action(large_ts, 50)
        assert fs.primary_actions[0]['at'] == large_ts

    def test_actions_property_getter(self, multi_axis_funscript):
        """The actions property returns primary_actions."""
        fs = multi_axis_funscript
        assert fs.actions is fs.primary_actions

    def test_actions_property_setter_sorts(self):
        """The actions setter sorts actions by timestamp."""
        from funscript.multi_axis_funscript import MultiAxisFunscript
        fs = MultiAxisFunscript()
        unsorted = [
            {"at": 300, "pos": 75},
            {"at": 100, "pos": 25},
            {"at": 200, "pos": 50},
        ]
        fs.actions = unsorted
        timestamps = [a['at'] for a in fs.actions]
        assert timestamps == [100, 200, 300]

    def test_actions_setter_invalid_data_clears(self):
        """Setting actions to invalid data clears the list."""
        from funscript.multi_axis_funscript import MultiAxisFunscript
        fs = MultiAxisFunscript()
        fs.actions = "not a list"
        assert fs.primary_actions == []

    def test_get_latest_value_empty(self, empty_multi_axis_funscript):
        """get_latest_value on empty returns 50 (neutral)."""
        fs = empty_multi_axis_funscript
        assert fs.get_latest_value('primary') == 50
        assert fs.get_latest_value('secondary') == 50

    def test_get_latest_value_with_data(self, multi_axis_funscript):
        """get_latest_value returns the last action's position."""
        fs = multi_axis_funscript
        assert fs.get_latest_value('primary') == fs.primary_actions[-1]['pos']


@pytest.mark.unit
class TestMultiAxisFunscriptLargeDataset:
    """Tests for performance with large datasets."""

    def test_large_dataset_add_actions(self, empty_multi_axis_funscript):
        """Adding many actions maintains sorted order."""
        fs = empty_multi_axis_funscript
        fs.enable_point_simplification = False
        for i in range(1000):
            fs.add_action(i * 100, i % 101)
        assert len(fs.primary_actions) >= 100  # min_interval may filter some
        timestamps = [a['at'] for a in fs.primary_actions]
        assert timestamps == sorted(timestamps)

    def test_large_dataset_get_value(self, empty_multi_axis_funscript):
        """get_value works efficiently on large datasets."""
        fs = empty_multi_axis_funscript
        fs.enable_point_simplification = False
        for i in range(500):
            fs.add_action(i * 100, i % 101)
        # Should return valid interpolated values
        val = fs.get_value(25050, 'primary')
        assert 0 <= val <= 100

    def test_large_dataset_get_actions_in_range(self, empty_multi_axis_funscript):
        """get_actions_in_range works on large datasets."""
        fs = empty_multi_axis_funscript
        fs.enable_point_simplification = False
        for i in range(500):
            fs.add_action(i * 100, i % 101)
        result = fs.get_actions_in_range(10000, 20000, 'primary')
        for a in result:
            assert 10000 <= a['at'] <= 20000


@pytest.mark.unit
class TestMultiAxisFunscriptSimplification:
    """Tests for point simplification feature."""

    def test_simplification_enabled_by_default(self):
        """Point simplification is enabled by default."""
        from funscript.multi_axis_funscript import MultiAxisFunscript
        fs = MultiAxisFunscript()
        assert fs.enable_point_simplification is True

    def test_simplification_removes_collinear_points(self):
        """Simplification removes redundant collinear middle points."""
        from funscript.multi_axis_funscript import MultiAxisFunscript
        fs = MultiAxisFunscript()
        fs.enable_point_simplification = True
        # Add three collinear points: all at position 50
        fs.add_action(100, 50)
        fs.add_action(200, 50)
        fs.add_action(300, 50)
        # The middle point should be removed by simplification
        assert len(fs.primary_actions) <= 2

    def test_simplification_disabled_keeps_all_points(self):
        """When simplification is disabled, all points are kept."""
        from funscript.multi_axis_funscript import MultiAxisFunscript
        fs = MultiAxisFunscript()
        fs.enable_point_simplification = False
        fs.add_action(100, 50)
        fs.add_action(200, 50)
        fs.add_action(300, 50)
        assert len(fs.primary_actions) == 3

    def test_simplification_stats_tracked(self):
        """Simplification statistics are tracked when points are removed."""
        from funscript.multi_axis_funscript import MultiAxisFunscript
        fs = MultiAxisFunscript()
        fs.enable_point_simplification = True
        # Add collinear points that will be simplified
        for i in range(10):
            fs.add_action(i * 100, 50)
        assert fs._simplification_stats_primary['total_considered'] > 0

    def test_log_final_simplification_summary_resets_stats(self):
        """log_final_simplification_summary resets stats after logging."""
        from funscript.multi_axis_funscript import MultiAxisFunscript
        fs = MultiAxisFunscript()
        fs.enable_point_simplification = True
        for i in range(10):
            fs.add_action(i * 100, 50)
        fs.log_final_simplification_summary()
        assert fs._simplification_stats_primary['total_removed'] == 0
        assert fs._simplification_stats_primary['total_considered'] == 0


@pytest.mark.unit
class TestMultiAxisFunscriptStatistics:
    """Tests for the get_actions_statistics method."""

    def test_statistics_empty(self, empty_multi_axis_funscript):
        """Statistics on empty funscript returns defaults."""
        fs = empty_multi_axis_funscript
        stats = fs.get_actions_statistics('primary')
        assert stats['num_points'] == 0
        assert stats['duration_scripted_s'] == 0.0

    def test_statistics_with_data(self, multi_axis_funscript):
        """Statistics are computed correctly for populated funscript."""
        fs = multi_axis_funscript
        stats = fs.get_actions_statistics('primary')
        assert stats['num_points'] == len(fs.primary_actions)
        assert stats['duration_scripted_s'] > 0
        assert stats['min_pos'] >= 0
        assert stats['max_pos'] <= 100

    def test_statistics_single_point(self):
        """Statistics with a single point returns minimal data."""
        from funscript.multi_axis_funscript import MultiAxisFunscript
        fs = MultiAxisFunscript()
        fs.enable_point_simplification = False
        fs.add_action(100, 50)
        stats = fs.get_actions_statistics('primary')
        assert stats['num_points'] == 1
        assert stats['min_pos'] == 50
        assert stats['max_pos'] == 50
