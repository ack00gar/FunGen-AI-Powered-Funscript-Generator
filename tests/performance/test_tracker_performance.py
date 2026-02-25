"""
Performance benchmarks for tracker discovery and registry operations.

Tests verify that tracker discovery, metadata access, and registry
lookups stay within acceptable time budgets.
"""

import time

import pytest

from config.tracker_discovery import (
    DynamicTrackerDiscovery,
    TrackerCategory,
    get_tracker_discovery,
)
from tracker.tracker_modules import (
    TrackerRegistry,
    TrackerMetadata,
    tracker_registry,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.performance
@pytest.mark.trackers
class TestTrackerPerformance:
    """Performance benchmarks for tracker operations."""

    def test_tracker_discovery_time(self):
        """Discovering all trackers should complete in under 0.5 seconds.

        This measures the time to instantiate a fresh DynamicTrackerDiscovery
        and retrieve the full tracker map, which involves scanning the
        tracker_modules package.
        """
        start = time.perf_counter()
        discovery = get_tracker_discovery()
        all_trackers = discovery.get_all_trackers()
        elapsed = time.perf_counter() - start

        assert len(all_trackers) > 0, "Tracker discovery found no trackers"
        assert elapsed < 0.5, (
            f"Tracker discovery took {elapsed:.3f}s, expected < 0.5s. "
            f"Found {len(all_trackers)} trackers."
        )

    def test_tracker_metadata_access_time(self):
        """Accessing metadata for all discovered trackers should be fast.

        Iterates over every tracker in the registry and reads its
        metadata fields (name, display_name, category, description).
        This should be nearly instantaneous since metadata is precomputed.
        """
        all_metadata = tracker_registry.list_trackers()
        assert len(all_metadata) > 0, "No trackers in registry for metadata test"

        start = time.perf_counter()
        for meta in all_metadata:
            # Access all core metadata fields to ensure they are cached / fast
            _ = meta.name
            _ = meta.display_name
            _ = meta.category
            _ = meta.description
        elapsed = time.perf_counter() - start

        assert elapsed < 0.1, (
            f"Accessing metadata for {len(all_metadata)} trackers took "
            f"{elapsed:.4f}s, expected < 0.1s"
        )

    def test_tracker_registry_size(self):
        """Verify that the expected number of trackers are registered.

        The registry should contain at least 5 tracker modules (live and
        offline combined). This is a sanity check that auto-discovery is
        working correctly.
        """
        all_metadata = tracker_registry.list_trackers()
        names = [m.name for m in all_metadata]

        assert len(all_metadata) >= 5, (
            f"Expected at least 5 registered trackers, found {len(all_metadata)}: {names}"
        )

        # Also verify through the discovery layer
        discovery = get_tracker_discovery()
        all_trackers = discovery.get_all_trackers()
        assert len(all_trackers) >= 5, (
            f"Expected at least 5 discovered trackers, found {len(all_trackers)}: "
            f"{list(all_trackers.keys())}"
        )
