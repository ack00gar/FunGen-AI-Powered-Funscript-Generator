"""
Tests for OD Exp 3 auto-discovery and tracker creation.

Tests cover:
- TrackerRegistry auto-discovery of oscillation_experimental_3
- GUI display list inclusion
- Tracker instantiation via tracker_registry.create_tracker
- Metadata correctness (name, version, tags, category, display_name)
"""

import pytest
pytest.importorskip("patreon_features")

from config.tracker_discovery import (
    DynamicTrackerDiscovery,
    TrackerCategory,
    TrackerDisplayInfo,
    get_tracker_discovery,
)
from tracker.tracker_modules import (
    TrackerRegistry,
    TrackerMetadata,
    tracker_registry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def discovery() -> DynamicTrackerDiscovery:
    """Return the global DynamicTrackerDiscovery singleton."""
    return get_tracker_discovery()


# ---------------------------------------------------------------------------
# Exp 3 Tracker Discovery & Creation
# ---------------------------------------------------------------------------

@pytest.mark.trackers
class TestExp3TrackerDiscovery:
    def test_exp3_tracker_auto_discovered(self):
        """TrackerRegistry finds oscillation_experimental_3."""
        all_metadata = tracker_registry.list_trackers()
        names = [m.name for m in all_metadata]
        assert "oscillation_experimental_3" in names, (
            f"oscillation_experimental_3 not found in registry. Found: {names}"
        )

    def test_exp3_in_gui_display_list(self):
        """Appears in tracker selection combo."""
        discovery = get_tracker_discovery()
        display_names, internal_names = discovery.get_gui_display_list()
        assert "oscillation_experimental_3" in internal_names, (
            f"oscillation_experimental_3 not in GUI internal names. Found: {internal_names}"
        )

    def test_exp3_tracker_creation(self):
        """tracker_registry.create_tracker('oscillation_experimental_3') succeeds."""
        # get_tracker returns the class
        tracker_class = tracker_registry.get_tracker("oscillation_experimental_3")
        assert tracker_class is not None, "get_tracker returned None for oscillation_experimental_3"
        # Instantiate it
        tracker = tracker_class()
        assert tracker is not None
        assert tracker.metadata.name == "oscillation_experimental_3"
        assert tracker.metadata.display_name == "Oscillation Detector (Experimental 3)"
        assert tracker.metadata.category == "live"
        assert tracker.metadata.supports_dual_axis is True

    def test_exp3_metadata_correct(self):
        """Metadata fields are correct."""
        meta = tracker_registry.get_metadata("oscillation_experimental_3")
        assert meta is not None
        assert meta.name == "oscillation_experimental_3"
        assert meta.version == "1.0.0"
        assert "endpoint" in meta.tags
        assert "camera-compensation" in meta.tags
