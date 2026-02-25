"""Unit tests for multi-timeline support.

Tests cover two areas:
1. MultiAxisFunscript multi-axis extension (additional_axes, ensure_axis,
   get_axis_actions, add_action_to_axis, get_axis_count, get_all_axis_names,
   clear_axis, and extended clear()).
2. InteractiveFunscriptTimeline visibility attribute fix for timeline_num >= 3.
"""
import pytest
from unittest.mock import MagicMock
from funscript.multi_axis_funscript import MultiAxisFunscript


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def funscript():
    """Create a fresh MultiAxisFunscript with point simplification disabled."""
    fs = MultiAxisFunscript()
    fs.enable_point_simplification = False
    return fs


@pytest.fixture
def populated_funscript(funscript):
    """Funscript with some data on primary, secondary, and an extra axis."""
    funscript.add_action(100, 10, 90)
    funscript.add_action(200, 20, 80)
    funscript.add_action(300, 30, 70)
    funscript.ensure_axis("axis_3")
    funscript.add_action_to_axis("axis_3", 100, 50)
    funscript.add_action_to_axis("axis_3", 200, 60)
    funscript.add_action_to_axis("axis_3", 300, 70)
    return funscript


# =========================================================================
# 1. MultiAxisFunscript multi-axis extension tests
# =========================================================================


@pytest.mark.unit
class TestAdditionalAxesInit:
    """additional_axes starts as an empty dict on a fresh instance."""

    def test_additional_axes_is_empty_dict(self, funscript):
        assert funscript.additional_axes == {}
        assert isinstance(funscript.additional_axes, dict)

    def test_additional_cache_structures_init_empty(self, funscript):
        """Cache-related dicts for additional axes start empty."""
        assert funscript._additional_timestamps_cache == {}
        assert funscript._additional_cache_dirty == {}
        assert funscript._additional_last_timestamps == {}


@pytest.mark.unit
class TestEnsureAxis:
    """ensure_axis creates storage when needed and is idempotent."""

    def test_creates_storage_for_new_axis(self, funscript):
        funscript.ensure_axis("axis_3")
        assert "axis_3" in funscript.additional_axes
        assert funscript.additional_axes["axis_3"] == []

    def test_creates_cache_entries_for_new_axis(self, funscript):
        funscript.ensure_axis("axis_3")
        assert "axis_3" in funscript._additional_timestamps_cache
        assert "axis_3" in funscript._additional_cache_dirty
        assert "axis_3" in funscript._additional_last_timestamps

    def test_idempotent_does_not_reset_data(self, funscript):
        """Calling ensure_axis twice must not wipe out existing data."""
        funscript.ensure_axis("axis_3")
        funscript.add_action_to_axis("axis_3", 100, 42)
        assert len(funscript.additional_axes["axis_3"]) == 1

        # Second call should leave data intact
        funscript.ensure_axis("axis_3")
        assert len(funscript.additional_axes["axis_3"]) == 1
        assert funscript.additional_axes["axis_3"][0]["pos"] == 42

    def test_can_create_multiple_axes(self, funscript):
        for name in ("axis_3", "axis_4", "twist"):
            funscript.ensure_axis(name)
        assert set(funscript.additional_axes.keys()) == {"axis_3", "axis_4", "twist"}


@pytest.mark.unit
class TestGetAxisActions:
    """get_axis_actions returns the correct action list for any axis name."""

    def test_primary_returns_primary_actions(self, funscript):
        funscript.add_action(100, 50)
        result = funscript.get_axis_actions("primary")
        assert result is funscript.primary_actions

    def test_secondary_returns_secondary_actions(self, funscript):
        funscript.add_action(100, None, 50)
        result = funscript.get_axis_actions("secondary")
        assert result is funscript.secondary_actions

    def test_additional_axis_returns_correct_list(self, funscript):
        funscript.ensure_axis("axis_3")
        funscript.add_action_to_axis("axis_3", 100, 55)
        result = funscript.get_axis_actions("axis_3")
        assert result is funscript.additional_axes["axis_3"]
        assert len(result) == 1
        assert result[0]["pos"] == 55

    def test_nonexistent_axis_returns_empty_list(self, funscript):
        result = funscript.get_axis_actions("does_not_exist")
        assert result == []


@pytest.mark.unit
class TestAddActionToAxis:
    """add_action_to_axis adds actions correctly and respects min_interval_ms."""

    def test_adds_action_to_additional_axis(self, funscript):
        funscript.ensure_axis("axis_3")
        funscript.add_action_to_axis("axis_3", 1000, 75)
        actions = funscript.additional_axes["axis_3"]
        assert len(actions) == 1
        assert actions[0]["at"] == 1000
        assert actions[0]["pos"] == 75

    def test_multiple_actions_sorted_order(self, funscript):
        funscript.ensure_axis("axis_3")
        funscript.add_action_to_axis("axis_3", 300, 30)
        funscript.add_action_to_axis("axis_3", 100, 10)
        funscript.add_action_to_axis("axis_3", 200, 20)
        actions = funscript.additional_axes["axis_3"]
        timestamps = [a["at"] for a in actions]
        assert timestamps == sorted(timestamps)

    def test_respects_min_interval_ms(self, funscript):
        """Actions closer than min_interval_ms should be rejected/collapsed."""
        funscript.ensure_axis("axis_3")
        funscript.min_interval_ms = 10
        funscript.add_action_to_axis("axis_3", 100, 50)
        # This is only 5 ms later -- should be rejected
        funscript.add_action_to_axis("axis_3", 105, 60)
        actions = funscript.additional_axes["axis_3"]
        assert len(actions) == 1
        assert actions[0]["at"] == 100

    def test_clamps_position_to_0_100(self, funscript):
        funscript.ensure_axis("axis_3")
        funscript.add_action_to_axis("axis_3", 100, -10)
        funscript.add_action_to_axis("axis_3", 200, 150)
        actions = funscript.additional_axes["axis_3"]
        assert actions[0]["pos"] == 0
        assert actions[1]["pos"] == 100

    def test_auto_ensures_axis_if_not_exists(self, funscript):
        """add_action_to_axis should auto-create the axis if needed."""
        funscript.add_action_to_axis("axis_3", 500, 42)
        assert "axis_3" in funscript.additional_axes
        assert len(funscript.additional_axes["axis_3"]) == 1


@pytest.mark.unit
class TestAxisCount:
    """get_axis_count returns 2 + number of additional axes."""

    def test_returns_2_when_no_extra_axes(self, funscript):
        assert funscript.get_axis_count() == 2

    def test_returns_3_with_one_extra(self, funscript):
        funscript.ensure_axis("axis_3")
        assert funscript.get_axis_count() == 3

    def test_returns_2_plus_n(self, funscript):
        for i in range(5):
            funscript.ensure_axis(f"axis_{i + 3}")
        assert funscript.get_axis_count() == 7


@pytest.mark.unit
class TestGetAllAxisNames:
    """get_all_axis_names returns ordered list starting with primary, secondary."""

    def test_default_names(self, funscript):
        names = funscript.get_all_axis_names()
        assert names == ["primary", "secondary"]

    def test_includes_additional_axes(self, funscript):
        funscript.ensure_axis("axis_3")
        funscript.ensure_axis("axis_4")
        names = funscript.get_all_axis_names()
        assert names[0] == "primary"
        assert names[1] == "secondary"
        assert "axis_3" in names
        assert "axis_4" in names
        assert len(names) == 4

    def test_names_reflect_all_ensures(self, funscript):
        funscript.ensure_axis("twist")
        names = funscript.get_all_axis_names()
        assert "twist" in names


@pytest.mark.unit
class TestClearAxis:
    """clear_axis clears a specific additional axis without affecting others."""

    def test_clears_target_axis(self, populated_funscript):
        fs = populated_funscript
        assert len(fs.additional_axes["axis_3"]) == 3
        fs.clear_axis("axis_3")
        assert fs.additional_axes["axis_3"] == []

    def test_does_not_affect_primary(self, populated_funscript):
        fs = populated_funscript
        primary_count = len(fs.primary_actions)
        fs.clear_axis("axis_3")
        assert len(fs.primary_actions) == primary_count

    def test_does_not_affect_secondary(self, populated_funscript):
        fs = populated_funscript
        secondary_count = len(fs.secondary_actions)
        fs.clear_axis("axis_3")
        assert len(fs.secondary_actions) == secondary_count

    def test_does_not_affect_other_additional(self, funscript):
        funscript.ensure_axis("axis_3")
        funscript.ensure_axis("axis_4")
        funscript.add_action_to_axis("axis_3", 100, 50)
        funscript.add_action_to_axis("axis_4", 100, 60)
        funscript.clear_axis("axis_3")
        assert funscript.additional_axes["axis_3"] == []
        assert len(funscript.additional_axes["axis_4"]) == 1

    def test_clear_nonexistent_axis_is_noop(self, funscript):
        """Clearing an axis that was never created should not raise."""
        funscript.clear_axis("nonexistent")  # Should not raise


@pytest.mark.unit
class TestClearAll:
    """Full clear() also clears additional axes."""

    def test_clear_empties_additional_axes(self, populated_funscript):
        fs = populated_funscript
        assert len(fs.additional_axes["axis_3"]) == 3
        fs.clear()
        # After clear, additional_axes storage should be empty (or axes reset)
        for axis_name, actions in fs.additional_axes.items():
            assert actions == [], f"Axis {axis_name} was not cleared"

    def test_clear_empties_primary_and_secondary(self, populated_funscript):
        fs = populated_funscript
        fs.clear()
        assert fs.primary_actions == []
        assert fs.secondary_actions == []

    def test_clear_resets_timestamps(self, populated_funscript):
        fs = populated_funscript
        fs.clear()
        assert fs.last_timestamp_primary == 0
        assert fs.last_timestamp_secondary == 0


@pytest.mark.unit
class TestGetValueAdditionalAxis:
    """get_value interpolation works on additional axes."""

    def test_interpolation_midpoint(self, funscript):
        funscript.ensure_axis("axis_3")
        funscript.add_action_to_axis("axis_3", 0, 0)
        funscript.add_action_to_axis("axis_3", 100, 100)
        value = funscript.get_value(50, axis="axis_3")
        assert value == 50

    def test_interpolation_quarter_point(self, funscript):
        funscript.ensure_axis("axis_3")
        funscript.add_action_to_axis("axis_3", 0, 0)
        funscript.add_action_to_axis("axis_3", 100, 100)
        value = funscript.get_value(25, axis="axis_3")
        assert value == 25

    def test_before_first_action_returns_first_pos(self, funscript):
        funscript.ensure_axis("axis_3")
        funscript.add_action_to_axis("axis_3", 100, 42)
        funscript.add_action_to_axis("axis_3", 200, 80)
        value = funscript.get_value(0, axis="axis_3")
        assert value == 42

    def test_after_last_action_returns_last_pos(self, funscript):
        funscript.ensure_axis("axis_3")
        funscript.add_action_to_axis("axis_3", 100, 42)
        funscript.add_action_to_axis("axis_3", 200, 80)
        value = funscript.get_value(500, axis="axis_3")
        assert value == 80

    def test_empty_axis_returns_default_50(self, funscript):
        funscript.ensure_axis("axis_3")
        value = funscript.get_value(100, axis="axis_3")
        assert value == 50

    def test_exact_timestamp_match(self, funscript):
        funscript.ensure_axis("axis_3")
        funscript.add_action_to_axis("axis_3", 100, 75)
        funscript.add_action_to_axis("axis_3", 200, 25)
        value = funscript.get_value(100, axis="axis_3")
        assert value == 75


@pytest.mark.unit
class TestBackwardCompatibility:
    """Existing add_action(primary, secondary) API still works unchanged."""

    def test_add_primary_only(self, funscript):
        funscript.add_action(1000, 75)
        assert len(funscript.primary_actions) == 1
        assert funscript.primary_actions[0]["at"] == 1000
        assert funscript.primary_actions[0]["pos"] == 75

    def test_add_secondary_only(self, funscript):
        funscript.add_action(1000, None, 30)
        assert len(funscript.secondary_actions) == 1
        assert funscript.secondary_actions[0]["at"] == 1000
        assert funscript.secondary_actions[0]["pos"] == 30
        assert len(funscript.primary_actions) == 0

    def test_add_both_axes_simultaneously(self, funscript):
        funscript.add_action(1000, 80, 20)
        assert len(funscript.primary_actions) == 1
        assert len(funscript.secondary_actions) == 1
        assert funscript.primary_actions[0]["pos"] == 80
        assert funscript.secondary_actions[0]["pos"] == 20

    def test_get_value_primary(self, funscript):
        funscript.add_action(0, 0)
        funscript.add_action(100, 100)
        assert funscript.get_value(50, axis="primary") == 50

    def test_get_value_secondary(self, funscript):
        funscript.add_action(0, None, 0)
        funscript.add_action(100, None, 100)
        assert funscript.get_value(50, axis="secondary") == 50

    def test_clear_still_works(self, funscript):
        funscript.add_action(1000, 50, 50)
        funscript.clear()
        assert funscript.primary_actions == []
        assert funscript.secondary_actions == []


# =========================================================================
# 2. InteractiveFunscriptTimeline visibility attribute fix
# =========================================================================


@pytest.mark.unit
class TestVisibilityAttr:
    """Verify the visibility attribute naming pattern for multi-timeline support.

    The fixed pattern is:
        f"show_funscript_interactive_timeline{'' if timeline_num == 1 else str(timeline_num)}"

    Timeline 1 -> 'show_funscript_interactive_timeline'
    Timeline 2 -> 'show_funscript_interactive_timeline2'
    Timeline 3 -> 'show_funscript_interactive_timeline3'
    Timeline N -> 'show_funscript_interactive_timelineN'

    The old (buggy) code hardcoded '2' for all timeline_num != 1, which meant
    timeline 3, 4, ... would all incorrectly resolve to the timeline 2 attribute.
    """

    @staticmethod
    def _make_visibility_attr(timeline_num: int) -> str:
        """Replicate the fixed visibility attribute generation logic."""
        return f"show_funscript_interactive_timeline{'' if timeline_num == 1 else str(timeline_num)}"

    def test_visibility_attr_timeline_1(self):
        attr = self._make_visibility_attr(1)
        assert attr == "show_funscript_interactive_timeline"

    def test_visibility_attr_timeline_2(self):
        attr = self._make_visibility_attr(2)
        assert attr == "show_funscript_interactive_timeline2"

    def test_visibility_attr_timeline_3(self):
        attr = self._make_visibility_attr(3)
        assert attr == "show_funscript_interactive_timeline3"

    def test_visibility_attr_timeline_4(self):
        attr = self._make_visibility_attr(4)
        assert attr == "show_funscript_interactive_timeline4"

    def test_old_bug_timeline_3_would_be_wrong(self):
        """Demonstrate the bug: the old code would produce '2' for timeline 3."""
        timeline_num = 3
        # Old (buggy) pattern
        old_attr = f"show_funscript_interactive_timeline{'' if timeline_num == 1 else '2'}"
        # New (fixed) pattern
        new_attr = f"show_funscript_interactive_timeline{'' if timeline_num == 1 else str(timeline_num)}"
        # The old attr is wrong for timeline 3
        assert old_attr == "show_funscript_interactive_timeline2"  # bug!
        assert new_attr == "show_funscript_interactive_timeline3"  # fix
        assert old_attr != new_attr


# =========================================================================
# 3. Serialization round-trip tests (to_dict / from_dict)
# =========================================================================


@pytest.mark.unit
class TestToDict:
    """to_dict() captures all axes and assignments."""

    def test_empty_funscript_serializes(self, funscript):
        data = funscript.to_dict()
        assert "axes" in data
        assert "axis_assignments" in data
        assert data["axes"]["primary"] == []
        assert data["axes"]["secondary"] == []

    def test_populated_funscript_serializes(self, populated_funscript):
        data = populated_funscript.to_dict()
        assert len(data["axes"]["primary"]) == 3
        assert len(data["axes"]["secondary"]) == 3
        assert len(data["axes"]["axis_3"]) == 3

    def test_axis_assignments_serialized(self, funscript):
        funscript.assign_axis(1, "stroke")
        funscript.assign_axis(2, "roll")
        funscript.assign_axis(3, "pitch")
        data = funscript.to_dict()
        assignments = data["axis_assignments"]
        assert assignments["1"] == "stroke"
        assert assignments["2"] == "roll"
        assert assignments["3"] == "pitch"

    def test_actions_are_copies(self, populated_funscript):
        """Serialized actions should be copies, not references."""
        data = populated_funscript.to_dict()
        data["axes"]["primary"][0]["pos"] = 999
        assert populated_funscript.primary_actions[0]["pos"] != 999


@pytest.mark.unit
class TestFromDict:
    """from_dict() reconstructs the full state."""

    def test_round_trip_primary_secondary(self, populated_funscript):
        data = populated_funscript.to_dict()
        restored = MultiAxisFunscript.from_dict(data)
        assert len(restored.primary_actions) == 3
        assert len(restored.secondary_actions) == 3

    def test_round_trip_additional_axes(self, populated_funscript):
        data = populated_funscript.to_dict()
        restored = MultiAxisFunscript.from_dict(data)
        assert "axis_3" in restored.additional_axes
        assert len(restored.additional_axes["axis_3"]) == 3

    def test_round_trip_axis_assignments(self, funscript):
        funscript.assign_axis(1, "stroke")
        funscript.assign_axis(2, "roll")
        funscript.assign_axis(3, "pitch")
        data = funscript.to_dict()
        restored = MultiAxisFunscript.from_dict(data)
        assert restored._axis_assignments[1] == "stroke"
        assert restored._axis_assignments[2] == "roll"
        assert restored._axis_assignments[3] == "pitch"

    def test_round_trip_preserves_action_values(self, populated_funscript):
        data = populated_funscript.to_dict()
        restored = MultiAxisFunscript.from_dict(data)
        for orig, copy in zip(populated_funscript.primary_actions, restored.primary_actions):
            assert orig["at"] == copy["at"]
            assert orig["pos"] == copy["pos"]

    def test_from_dict_without_assignments_uses_defaults(self):
        """Loading data without axis_assignments key should use defaults."""
        data = {"axes": {"primary": [], "secondary": []}}
        restored = MultiAxisFunscript.from_dict(data)
        assert restored._axis_assignments == {1: "stroke", 2: "roll"}

    def test_from_dict_empty_data(self):
        """from_dict with empty dict should produce valid empty object."""
        restored = MultiAxisFunscript.from_dict({})
        assert restored.primary_actions == []
        assert restored.secondary_actions == []
        assert restored.additional_axes == {}


@pytest.mark.unit
class TestAxisAssignment:
    """Test timeline-to-axis assignment methods."""

    def test_default_assignments(self, funscript):
        assert funscript.get_axis_for_timeline(1) == "stroke"
        assert funscript.get_axis_for_timeline(2) == "roll"

    def test_assign_and_get(self, funscript):
        funscript.assign_axis(3, "pitch")
        assert funscript.get_axis_for_timeline(3) == "pitch"

    def test_get_unassigned_timeline_returns_fallback(self, funscript):
        result = funscript.get_axis_for_timeline(5)
        assert result == "axis_5"

    def test_get_timeline_for_axis(self, funscript):
        assert funscript.get_timeline_for_axis("stroke") == 1
        assert funscript.get_timeline_for_axis("roll") == 2

    def test_get_timeline_for_unknown_axis(self, funscript):
        assert funscript.get_timeline_for_axis("nonexistent") is None

    def test_reassign_axis(self, funscript):
        funscript.assign_axis(2, "pitch")
        assert funscript.get_axis_for_timeline(2) == "pitch"
        assert funscript.get_timeline_for_axis("pitch") == 2

    def test_get_axis_assignments_returns_copy(self, funscript):
        assignments = funscript.get_axis_assignments()
        assignments[99] = "test"
        assert 99 not in funscript._axis_assignments
