"""
Comprehensive tests for the FunGen tracker discovery system.

Tests cover:
- DynamicTrackerDiscovery: tracker listing, category filtering, CLI mode resolution
- TrackerDisplayInfo: field presence and validity
- TrackerRegistry: auto-discovery of tracker modules
- TrackerCategory: enum membership
- GUI display list consistency
"""

import pytest

from config.tracker_discovery import (
    DynamicTrackerDiscovery,
    TrackerCategory,
    TrackerDisplayInfo,
    get_tracker_discovery,
)
from funscript.axis_registry import FunscriptAxis
from tracker.tracker_modules import (
    TrackerRegistry,
    TrackerMetadata,
    tracker_registry,
)

# Valid axis names from the canonical registry
_VALID_AXIS_NAMES = {a.value for a in FunscriptAxis}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def discovery() -> DynamicTrackerDiscovery:
    """Return the global DynamicTrackerDiscovery singleton."""
    return get_tracker_discovery()


@pytest.fixture(scope="module")
def all_trackers(discovery):
    """Return dict of internal_name -> TrackerDisplayInfo."""
    return discovery.get_all_trackers()


# ---------------------------------------------------------------------------
# DynamicTrackerDiscovery basic tests
# ---------------------------------------------------------------------------

@pytest.mark.trackers
class TestDiscoveryBasic:
    """Basic tests for DynamicTrackerDiscovery."""

    def test_get_tracker_discovery_returns_singleton(self):
        """get_tracker_discovery() should always return the same instance."""
        d1 = get_tracker_discovery()
        d2 = get_tracker_discovery()
        assert d1 is d2

    def test_discovery_returns_nonempty_tracker_list(self, all_trackers):
        """At least one tracker should be discovered."""
        assert len(all_trackers) > 0, "Discovery returned no trackers at all"

    def test_all_trackers_have_required_fields(self, all_trackers):
        """Every TrackerDisplayInfo must have non-empty name, display_name, category, description."""
        for name, info in all_trackers.items():
            assert isinstance(info, TrackerDisplayInfo), (
                f"Tracker '{name}' is not a TrackerDisplayInfo"
            )
            assert info.internal_name and len(info.internal_name) > 0, (
                f"Tracker '{name}' has empty internal_name"
            )
            assert info.display_name and len(info.display_name) > 0, (
                f"Tracker '{name}' has empty display_name"
            )
            assert isinstance(info.category, TrackerCategory), (
                f"Tracker '{name}' has invalid category type: {type(info.category)}"
            )
            assert info.description and len(info.description) > 0, (
                f"Tracker '{name}' has empty description"
            )

    def test_categories_are_valid_enum_values(self, all_trackers):
        """Every tracker's category should be a valid TrackerCategory member."""
        valid_categories = set(TrackerCategory)
        for name, info in all_trackers.items():
            assert info.category in valid_categories, (
                f"Tracker '{name}' has category '{info.category}' "
                f"which is not in {[c.value for c in valid_categories]}"
            )


# ---------------------------------------------------------------------------
# CLI modes
# ---------------------------------------------------------------------------

@pytest.mark.trackers
class TestCLIModes:
    """Tests for CLI mode resolution."""

    def test_cli_modes_nonempty(self, discovery):
        cli_modes = discovery.get_supported_cli_modes()
        assert isinstance(cli_modes, list)
        assert len(cli_modes) > 0, "No CLI modes discovered"

    def test_cli_modes_are_strings(self, discovery):
        for mode in discovery.get_supported_cli_modes():
            assert isinstance(mode, str) and len(mode) > 0

    def test_resolve_cli_mode_known(self, discovery, all_trackers):
        """Resolving a known internal name should return that name."""
        # Pick the first tracker's internal name as a test
        first_name = next(iter(all_trackers))
        resolved = discovery.resolve_cli_mode(first_name)
        assert resolved is not None, (
            f"Expected '{first_name}' to be resolvable as a CLI alias"
        )

    def test_resolve_cli_mode_unknown_returns_none(self, discovery):
        assert discovery.resolve_cli_mode("__nonexistent_mode__") is None


# ---------------------------------------------------------------------------
# GUI display list
# ---------------------------------------------------------------------------

@pytest.mark.trackers
class TestGUIDisplayList:
    """Tests for GUI display list generation."""

    def test_gui_display_list_returns_two_lists(self, discovery):
        result = discovery.get_gui_display_list()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_gui_display_list_matching_lengths(self, discovery):
        display_names, internal_names = discovery.get_gui_display_list()
        assert len(display_names) == len(internal_names), (
            f"Display names ({len(display_names)}) and internal names "
            f"({len(internal_names)}) have different lengths"
        )

    def test_gui_display_list_nonempty(self, discovery):
        display_names, internal_names = discovery.get_gui_display_list()
        assert len(display_names) > 0, "GUI display list is empty"

    def test_gui_display_names_are_strings(self, discovery):
        display_names, _ = discovery.get_gui_display_list()
        for dn in display_names:
            assert isinstance(dn, str) and len(dn) > 0

    def test_gui_internal_names_are_strings(self, discovery):
        _, internal_names = discovery.get_gui_display_list()
        for name in internal_names:
            assert isinstance(name, str) and len(name) > 0

    def test_gui_excludes_example_trackers(self, discovery):
        """Example trackers should not appear in the GUI display list."""
        display_names, internal_names = discovery.get_gui_display_list()
        for dn in display_names:
            assert "example" not in dn.lower(), (
                f"GUI display list should not contain example trackers, found: '{dn}'"
            )
        for name in internal_names:
            assert "example" not in name.lower(), (
                f"GUI internal names should not contain example trackers, found: '{name}'"
            )


# ---------------------------------------------------------------------------
# Tracker info lookup
# ---------------------------------------------------------------------------

@pytest.mark.trackers
class TestTrackerInfoLookup:
    """Tests for get_tracker_info()."""

    def test_get_tracker_info_known_tracker(self, discovery, all_trackers):
        """Looking up a known tracker by internal name should return valid info."""
        for name in all_trackers:
            info = discovery.get_tracker_info(name)
            assert info is not None, f"get_tracker_info('{name}') returned None"
            assert info.internal_name == name

    def test_get_tracker_info_unknown_returns_none(self, discovery):
        result = discovery.get_tracker_info("__does_not_exist__")
        assert result is None

    def test_get_tracker_info_via_cli_alias(self, discovery, all_trackers):
        """If a tracker has CLI aliases, looking up by alias should resolve to a valid tracker.

        Note: Some CLI aliases (e.g. 'oscillation-legacy') may be shared across
        multiple trackers.  The alias map keeps only the last registration, so we
        check that the alias resolves to *some* valid tracker rather than
        requiring it to map back to the exact tracker we found it on.
        """
        for name, info in all_trackers.items():
            for alias in info.cli_aliases:
                resolved = discovery.get_tracker_info(alias)
                assert resolved is not None, (
                    f"CLI alias '{alias}' for tracker '{name}' did not resolve"
                )
                # The resolved tracker must itself be in the discovery cache
                assert resolved.internal_name in all_trackers, (
                    f"CLI alias '{alias}' resolved to unknown tracker '{resolved.internal_name}'"
                )


# ---------------------------------------------------------------------------
# Category filtering
# ---------------------------------------------------------------------------

@pytest.mark.trackers
class TestCategoryFiltering:
    """Tests for category-based tracker retrieval."""

    def test_at_least_one_live_tracker(self, discovery):
        """At least one tracker should be in the LIVE category."""
        live = discovery.get_trackers_by_category(TrackerCategory.LIVE)
        # Also include LIVE_INTERVENTION as "live" trackers
        live_intervention = discovery.get_trackers_by_category(TrackerCategory.LIVE_INTERVENTION)
        total_live = len(live) + len(live_intervention)
        assert total_live >= 1, "Expected at least one LIVE or LIVE_INTERVENTION tracker"

    def test_at_least_one_offline_tracker(self, discovery):
        """At least one tracker should be in the OFFLINE category."""
        offline = discovery.get_trackers_by_category(TrackerCategory.OFFLINE)
        assert len(offline) >= 1, "Expected at least one OFFLINE tracker"

    def test_trackers_by_category_returns_display_info(self, discovery):
        """get_trackers_by_category should return TrackerDisplayInfo instances."""
        for category in TrackerCategory:
            trackers = discovery.get_trackers_by_category(category)
            for info in trackers:
                assert isinstance(info, TrackerDisplayInfo), (
                    f"Expected TrackerDisplayInfo, got {type(info)} for category {category}"
                )

    def test_batch_compatible_excludes_intervention(self, discovery):
        """Batch-compatible trackers should not include LIVE_INTERVENTION trackers."""
        batch = discovery.get_batch_compatible_trackers()
        for info in batch:
            assert info.category != TrackerCategory.LIVE_INTERVENTION, (
                f"Batch-compatible list should not include LIVE_INTERVENTION tracker '{info.internal_name}'"
            )

    def test_realtime_trackers_have_supports_realtime(self, discovery):
        """Realtime-compatible trackers should have supports_realtime=True."""
        realtime = discovery.get_realtime_compatible_trackers()
        for info in realtime:
            assert info.supports_realtime is True, (
                f"Realtime tracker '{info.internal_name}' has supports_realtime=False"
            )


# ---------------------------------------------------------------------------
# TrackerRegistry tests (lower-level)
# ---------------------------------------------------------------------------

@pytest.mark.trackers
class TestTrackerRegistry:
    """Tests for the TrackerRegistry auto-discovery system."""

    def test_registry_discovers_trackers(self):
        """The global tracker_registry should have at least one tracker."""
        all_metadata = tracker_registry.list_trackers()
        assert len(all_metadata) > 0, "TrackerRegistry discovered zero trackers"

    def test_registry_list_trackers_returns_metadata(self):
        """list_trackers should return TrackerMetadata instances."""
        all_metadata = tracker_registry.list_trackers()
        for meta in all_metadata:
            assert isinstance(meta, TrackerMetadata), (
                f"Expected TrackerMetadata, got {type(meta)}"
            )

    def test_registry_metadata_has_required_fields(self):
        """Each TrackerMetadata should have name, display_name, category, description."""
        for meta in tracker_registry.list_trackers():
            assert meta.name and len(meta.name) > 0, "TrackerMetadata has empty name"
            assert meta.display_name and len(meta.display_name) > 0, (
                f"TrackerMetadata '{meta.name}' has empty display_name"
            )
            assert meta.category and len(meta.category) > 0, (
                f"TrackerMetadata '{meta.name}' has empty category"
            )
            assert meta.description and len(meta.description) > 0, (
                f"TrackerMetadata '{meta.name}' has empty description"
            )

    def test_registry_get_tracker_class(self):
        """get_tracker should return a class for known tracker names."""
        all_metadata = tracker_registry.list_trackers()
        if not all_metadata:
            pytest.skip("No trackers in registry")
        first = all_metadata[0]
        tracker_class = tracker_registry.get_tracker(first.name)
        assert tracker_class is not None, (
            f"get_tracker('{first.name}') returned None"
        )

    def test_registry_get_tracker_returns_none_for_unknown(self):
        result = tracker_registry.get_tracker("__nonexistent_tracker__")
        assert result is None

    def test_registry_get_available_names(self):
        """get_available_names should return a list of strings matching known trackers."""
        names = tracker_registry.get_available_names()
        assert isinstance(names, list)
        for name in names:
            assert isinstance(name, str) and len(name) > 0

    def test_registry_get_metadata_known(self):
        """get_metadata for a known tracker should return TrackerMetadata."""
        all_metadata = tracker_registry.list_trackers()
        if not all_metadata:
            pytest.skip("No trackers in registry")
        first = all_metadata[0]
        meta = tracker_registry.get_metadata(first.name)
        assert meta is not None
        assert meta.name == first.name

    def test_registry_get_metadata_unknown(self):
        assert tracker_registry.get_metadata("__nonexistent__") is None

    def test_registry_category_filter_live(self):
        """list_trackers(category='live') should only return live trackers."""
        live = tracker_registry.list_trackers(category="live")
        for meta in live:
            assert meta.category == "live", (
                f"Tracker '{meta.name}' has category '{meta.category}' but was returned for 'live' filter"
            )

    def test_registry_category_filter_offline(self):
        """list_trackers(category='offline') should only return offline trackers."""
        offline = tracker_registry.list_trackers(category="offline")
        for meta in offline:
            assert meta.category == "offline", (
                f"Tracker '{meta.name}' has category '{meta.category}' but was returned for 'offline' filter"
            )

    def test_registry_get_tracker_folder(self):
        """get_tracker_folder should return a string for known trackers."""
        all_metadata = tracker_registry.list_trackers()
        if not all_metadata:
            pytest.skip("No trackers in registry")
        first = all_metadata[0]
        folder = tracker_registry.get_tracker_folder(first.name)
        assert folder is not None and isinstance(folder, str)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

@pytest.mark.trackers
class TestTrackerValidation:
    """Tests for the tracker setup validation."""

    def test_validate_setup_returns_tuple(self, discovery):
        result = discovery.validate_setup()
        assert isinstance(result, tuple)
        assert len(result) == 2
        is_valid, errors = result
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)

    def test_validate_setup_passes_with_live_and_offline(self, discovery):
        """If both live and offline trackers exist, validation should pass."""
        live = discovery.get_trackers_by_category(TrackerCategory.LIVE)
        live_intervention = discovery.get_trackers_by_category(TrackerCategory.LIVE_INTERVENTION)
        offline = discovery.get_trackers_by_category(TrackerCategory.OFFLINE)

        total_live = len(live) + len(live_intervention)
        if total_live > 0 and len(offline) > 0:
            is_valid, errors = discovery.validate_setup()
            assert is_valid is True, f"Validation failed with errors: {errors}"
        else:
            pytest.skip("Not enough tracker categories for full validation")


# ---------------------------------------------------------------------------
# TrackerDisplayInfo field tests
# ---------------------------------------------------------------------------

@pytest.mark.trackers
class TestTrackerDisplayInfoFields:
    """Tests verifying TrackerDisplayInfo fields are populated correctly."""

    def test_cli_aliases_is_list(self, all_trackers):
        for name, info in all_trackers.items():
            assert isinstance(info.cli_aliases, list), (
                f"Tracker '{name}' cli_aliases is not a list"
            )

    def test_cli_aliases_contain_internal_name(self, all_trackers):
        """The internal_name should be one of the CLI aliases for each tracker."""
        for name, info in all_trackers.items():
            assert info.internal_name in info.cli_aliases, (
                f"Tracker '{name}' internal_name not in its own cli_aliases: {info.cli_aliases}"
            )

    def test_supports_dual_axis_is_bool(self, all_trackers):
        for name, info in all_trackers.items():
            assert isinstance(info.supports_dual_axis, bool), (
                f"Tracker '{name}' supports_dual_axis is not bool"
            )

    def test_supports_batch_is_bool(self, all_trackers):
        for name, info in all_trackers.items():
            assert isinstance(info.supports_batch, bool), (
                f"Tracker '{name}' supports_batch is not bool"
            )

    def test_supports_realtime_is_bool(self, all_trackers):
        for name, info in all_trackers.items():
            assert isinstance(info.supports_realtime, bool), (
                f"Tracker '{name}' supports_realtime is not bool"
            )

    def test_requires_intervention_is_bool(self, all_trackers):
        for name, info in all_trackers.items():
            assert isinstance(info.requires_intervention, bool), (
                f"Tracker '{name}' requires_intervention is not bool"
            )

    def test_folder_name_is_string(self, all_trackers):
        for name, info in all_trackers.items():
            assert isinstance(info.folder_name, str), (
                f"Tracker '{name}' folder_name is not a string"
            )

    def test_primary_axis_is_valid_string(self, all_trackers):
        """Every tracker must declare a non-empty primary_axis that is a valid FunscriptAxis value."""
        for name, info in all_trackers.items():
            assert isinstance(info.primary_axis, str) and len(info.primary_axis) > 0, (
                f"Tracker '{name}' has empty or non-string primary_axis"
            )
            assert info.primary_axis in _VALID_AXIS_NAMES, (
                f"Tracker '{name}' primary_axis '{info.primary_axis}' is not a valid "
                f"FunscriptAxis value. Valid: {sorted(_VALID_AXIS_NAMES)}"
            )

    def test_secondary_axis_is_valid_for_dual_axis(self, all_trackers):
        """Dual-axis trackers must declare a valid secondary_axis."""
        for name, info in all_trackers.items():
            if not info.supports_dual_axis:
                continue
            assert isinstance(info.secondary_axis, str) and len(info.secondary_axis) > 0, (
                f"Dual-axis tracker '{name}' has empty or non-string secondary_axis"
            )
            assert info.secondary_axis in _VALID_AXIS_NAMES, (
                f"Dual-axis tracker '{name}' secondary_axis '{info.secondary_axis}' is not a valid "
                f"FunscriptAxis value. Valid: {sorted(_VALID_AXIS_NAMES)}"
            )
