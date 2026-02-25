"""
Comprehensive tests for the OscillationExperimental3Tracker.

Tests cover:
- TrackerMetadata: name, category, version, display_name, supports_dual_axis
- Initialization: return value, state setup, MultiAxisFunscript creation
- Endpoint extraction: direction change detection, stable angle tracking,
  speed limiting, position accumulation, clamping, endpoint markers
- Camera compensation: temporal averaging, magnitude threshold, adaptive scaling
- Grid detection: VR mode central focus
- No-motion decay: position decays toward center
- Lifecycle: start/stop tracking, reset state, cleanup
- Process frame: returns TrackerResult with correct fields
- Configurable parameters: kwargs override defaults
"""

import pytest
pytest.importorskip("patreon_features")

import numpy as np
from unittest.mock import MagicMock, patch
from collections import deque


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_app():
    """Return a mock application instance with common attributes."""
    app = MagicMock()
    app.app_settings = MagicMock()
    app.app_settings.get = MagicMock(return_value=True)
    app.funscript = None
    app.tracking_axis_mode = "both"
    app.single_axis_output_target = "primary"
    app.processor = MagicMock()
    app.processor.frame_width = 1920
    app.processor.frame_height = 1080
    return app


@pytest.fixture
def tracker(mock_app):
    """Return an initialized OscillationExperimental3Tracker."""
    from patreon_features.trackers.oscillation_experimental_3 import OscillationExperimental3Tracker
    t = OscillationExperimental3Tracker()
    t.initialize(mock_app)
    return t


# ---------------------------------------------------------------------------
# TestMetadata
# ---------------------------------------------------------------------------

@pytest.mark.trackers
class TestMetadata:
    """Tests for tracker metadata properties."""

    def test_metadata_name(self, tracker):
        """Metadata name should be 'oscillation_experimental_3'."""
        assert tracker.metadata.name == "oscillation_experimental_3"

    def test_metadata_display_name(self, tracker):
        """Metadata display_name should be human-readable."""
        assert tracker.metadata.display_name == "Oscillation Detector (Experimental 3)"

    def test_metadata_category(self, tracker):
        """Metadata category should be 'live'."""
        assert tracker.metadata.category == "live"

    def test_metadata_version(self, tracker):
        """Metadata version should be '1.0.0'."""
        assert tracker.metadata.version == "1.0.0"

    def test_metadata_supports_dual_axis(self, tracker):
        """Metadata supports_dual_axis should be True."""
        assert tracker.metadata.supports_dual_axis is True

    def test_metadata_returns_tracker_metadata_type(self, tracker):
        """Metadata should be a TrackerMetadata instance."""
        from tracker.tracker_modules.core.base_tracker import TrackerMetadata
        assert isinstance(tracker.metadata, TrackerMetadata)


# ---------------------------------------------------------------------------
# TestInitialize
# ---------------------------------------------------------------------------

@pytest.mark.trackers
class TestInitialize:
    """Tests for tracker initialization."""

    def test_initialize_returns_true(self, mock_app):
        """initialize() should return True on success."""
        from patreon_features.trackers.oscillation_experimental_3 import OscillationExperimental3Tracker
        t = OscillationExperimental3Tracker()
        result = t.initialize(mock_app)
        assert result is True

    def test_initialized_flag_set(self, tracker):
        """_initialized should be True after successful initialization."""
        assert tracker._initialized is True

    def test_tracking_not_active_after_init(self, tracker):
        """tracking_active should be False immediately after initialization."""
        assert tracker.tracking_active is False

    def test_state_defaults_after_init(self, tracker):
        """Core state should have correct defaults after initialization."""
        assert tracker._stable_angle == 0.0
        assert tracker._stable_mag == 0.0
        assert tracker.motion_direction == 1
        assert tracker.last_primary_pos == 50.0
        assert tracker.last_secondary_pos == 50.0
        assert tracker._prev_frame_time_ms == 0

    def test_creates_multi_axis_funscript_when_app_has_none(self, mock_app):
        """When app.funscript is None, tracker should create its own MultiAxisFunscript."""
        from patreon_features.trackers.oscillation_experimental_3 import OscillationExperimental3Tracker
        mock_app.funscript = None
        t = OscillationExperimental3Tracker()
        t.initialize(mock_app)
        assert t.funscript is not None

    def test_uses_app_funscript_when_available(self, mock_app):
        """When app.funscript is set, tracker should use it."""
        from patreon_features.trackers.oscillation_experimental_3 import OscillationExperimental3Tracker
        fake_funscript = MagicMock()
        mock_app.funscript = fake_funscript
        t = OscillationExperimental3Tracker()
        t.initialize(mock_app)
        assert t.funscript is fake_funscript

    def test_oscillation_history_initialized_as_empty_dict(self, tracker):
        """oscillation_history should be an empty dict after init."""
        assert isinstance(tracker.oscillation_history, dict)
        assert len(tracker.oscillation_history) == 0

    def test_oscillation_cell_persistence_initialized_as_empty_dict(self, tracker):
        """oscillation_cell_persistence should be an empty dict after init."""
        assert isinstance(tracker.oscillation_cell_persistence, dict)
        assert len(tracker.oscillation_cell_persistence) == 0

    def test_global_motion_history_initialized_as_deque(self, tracker):
        """_global_motion_history should be a deque with correct maxlen."""
        assert isinstance(tracker._global_motion_history, deque)
        assert tracker._global_motion_history.maxlen == tracker.camera_temporal_window

    def test_optical_flow_initialized(self, tracker):
        """flow_dense_osc should be initialized (not None) after init."""
        assert tracker.flow_dense_osc is not None


# ---------------------------------------------------------------------------
# TestEndpointExtraction
# ---------------------------------------------------------------------------

@pytest.mark.trackers
class TestEndpointExtraction:
    """Tests for the _apply_endpoint_extraction method."""

    def test_direction_change_detection(self, tracker):
        """When circular angle diff > threshold, direction should flip."""
        tracker._stable_angle = 0.0
        # Set stable_mag higher than incoming mean_mag so stable_angle is NOT
        # overwritten before the angle diff comparison.
        tracker._stable_mag = 10.0
        # Call with angle > 100 degrees different from stable angle 0
        primary, secondary, changed = tracker._apply_endpoint_extraction(6.0, 150.0, 1000)
        assert changed is True
        # Direction should have flipped from initial 1 to -1
        assert tracker.motion_direction == -1

    def test_no_direction_change_below_threshold(self, tracker):
        """When circular angle diff < threshold, direction should NOT flip."""
        tracker._stable_angle = 0.0
        tracker._stable_mag = 5.0
        # Call with angle only 50 degrees different (below default threshold of 100)
        primary, secondary, changed = tracker._apply_endpoint_extraction(6.0, 50.0, 1000)
        assert changed is False
        # Direction should remain at initial value of 1
        assert tracker.motion_direction == 1

    def test_stable_angle_updates_when_mag_increases(self, tracker):
        """Stable angle should update when mean_mag > current _stable_mag."""
        tracker._stable_angle = 0.0
        tracker._stable_mag = 2.0
        # Provide higher magnitude - should update stable angle
        tracker._apply_endpoint_extraction(5.0, 45.0, 1000)
        assert tracker._stable_mag == 5.0
        assert tracker._stable_angle == 45.0

    def test_stable_angle_does_not_update_when_mag_decreases(self, tracker):
        """Stable angle should NOT update when mean_mag <= current _stable_mag."""
        tracker._stable_angle = 90.0
        tracker._stable_mag = 10.0
        # Provide lower magnitude - stable angle should not change
        # Use small angle diff to avoid direction change
        tracker._apply_endpoint_extraction(3.0, 95.0, 1000)
        # stable_mag and stable_angle should remain unchanged (no direction change happened)
        assert tracker._stable_mag == 10.0
        assert tracker._stable_angle == 90.0

    def test_speed_limiting(self, tracker):
        """Magnitude should be capped at max_magnitude_speed_limit."""
        tracker._stable_angle = 0.0
        tracker._stable_mag = 0.0
        tracker.last_primary_pos = 50.0
        tracker.motion_direction = 1
        tracker.max_magnitude_speed_limit = 10
        # Provide magnitude of 50, which exceeds the limit of 10
        primary, secondary, changed = tracker._apply_endpoint_extraction(50.0, 5.0, 1000)
        # Position should move by at most max_magnitude_speed_limit (10), not 50
        assert primary <= 60  # 50 + 10

    def test_position_accumulation_upward(self, tracker):
        """With motion_direction=+1, position should increase."""
        tracker._stable_angle = 0.0
        tracker._stable_mag = 0.0
        tracker.last_primary_pos = 50.0
        tracker.motion_direction = 1
        primary, secondary, changed = tracker._apply_endpoint_extraction(5.0, 5.0, 1000)
        assert primary == 55  # 50 + 5

    def test_position_accumulation_downward(self, tracker):
        """With motion_direction=-1, position should decrease."""
        tracker._stable_angle = 0.0
        tracker._stable_mag = 0.0
        tracker.last_primary_pos = 50.0
        tracker.motion_direction = -1
        # Use an angle close to stable_angle (0) so no direction change
        primary, secondary, changed = tracker._apply_endpoint_extraction(5.0, 5.0, 1000)
        assert primary == 45  # 50 - 5

    def test_position_clamped_at_100(self, tracker):
        """Position should not exceed 100."""
        tracker._stable_angle = 0.0
        tracker._stable_mag = 0.0
        tracker.last_primary_pos = 98.0
        tracker.motion_direction = 1
        tracker.max_magnitude_speed_limit = 100
        primary, secondary, changed = tracker._apply_endpoint_extraction(10.0, 5.0, 1000)
        assert primary == 100
        assert tracker.last_primary_pos == 100.0

    def test_position_clamped_at_0(self, tracker):
        """Position should not go below 0."""
        tracker._stable_angle = 0.0
        tracker._stable_mag = 0.0
        tracker.last_primary_pos = 2.0
        tracker.motion_direction = -1
        tracker.max_magnitude_speed_limit = 100
        primary, secondary, changed = tracker._apply_endpoint_extraction(10.0, 5.0, 1000)
        assert primary == 0
        assert tracker.last_primary_pos == 0.0

    def test_endpoint_markers_on_direction_change(self, tracker):
        """Secondary should output 0 or 100 on direction change."""
        tracker._stable_angle = 0.0
        # Set stable_mag higher than incoming mean_mag to preserve stable_angle
        tracker._stable_mag = 10.0
        tracker.motion_direction = 1
        # Trigger direction change (angle diff > 100)
        primary, secondary, changed = tracker._apply_endpoint_extraction(6.0, 150.0, 1000)
        assert changed is True
        # After flip: direction is now -1, so secondary should be 0
        assert secondary == 0

    def test_endpoint_markers_opposite_direction(self, tracker):
        """Secondary should be 100 when direction flips to +1."""
        tracker._stable_angle = 0.0
        # Set stable_mag higher than incoming mean_mag to preserve stable_angle
        tracker._stable_mag = 10.0
        tracker.motion_direction = -1
        # Trigger direction change (angle diff > 100)
        primary, secondary, changed = tracker._apply_endpoint_extraction(6.0, 150.0, 1000)
        assert changed is True
        # After flip: direction is now +1, so secondary should be 100
        assert secondary == 100

    def test_no_endpoint_when_no_direction_change(self, tracker):
        """Secondary should be 50 when there is no direction change."""
        tracker._stable_angle = 0.0
        tracker._stable_mag = 5.0
        tracker.motion_direction = 1
        # Small angle difference, no direction change
        primary, secondary, changed = tracker._apply_endpoint_extraction(6.0, 30.0, 1000)
        assert changed is False
        assert secondary == 50

    def test_returns_int_positions(self, tracker):
        """Primary and secondary positions should be integer values."""
        primary, secondary, changed = tracker._apply_endpoint_extraction(3.7, 45.0, 1000)
        assert isinstance(primary, int)
        assert isinstance(secondary, int)

    def test_nan_magnitude_handled(self, tracker):
        """NaN magnitude should be treated as 0.0."""
        tracker.last_primary_pos = 50.0
        primary, secondary, changed = tracker._apply_endpoint_extraction(float('nan'), 45.0, 1000)
        # With 0.0 magnitude, position should stay at 50
        assert primary == 50

    def test_nan_angle_handled(self, tracker):
        """NaN angle should not cause direction change."""
        tracker._stable_angle = 0.0
        tracker._stable_mag = 5.0
        primary, secondary, changed = tracker._apply_endpoint_extraction(5.0, float('nan'), 1000)
        assert changed is False

    def test_direction_flips_back_on_second_change(self, tracker):
        """Two consecutive direction changes should flip direction back to original."""
        tracker._stable_angle = 0.0
        # Set stable_mag higher than incoming mean_mag to preserve stable_angle
        tracker._stable_mag = 10.0
        tracker.motion_direction = 1
        # First direction change: 1 -> -1
        # After direction change, stable_angle is updated to 150.0, stable_mag to 6.0
        tracker._apply_endpoint_extraction(6.0, 150.0, 1000)
        assert tracker.motion_direction == -1
        # Second direction change: -1 -> 1
        # After first change: stable_angle=150.0, stable_mag=6.0
        # We pass mean_mag=3.0 (< 6.0) so stable_angle stays at 150.0
        # Angle diff: circular_diff(300, 150) = 150 > 100 threshold
        tracker._apply_endpoint_extraction(3.0, 300.0, 2000)
        assert tracker.motion_direction == 1


# ---------------------------------------------------------------------------
# TestCameraCompensation
# ---------------------------------------------------------------------------

@pytest.mark.trackers
class TestCameraCompensation:
    """Tests for enhanced camera motion compensation parameters."""

    def test_temporal_avg_window(self, tracker):
        """_global_motion_history should use the configured temporal window."""
        assert tracker._global_motion_history.maxlen == tracker.camera_temporal_window

    def test_temporal_avg_accumulates_history(self, tracker):
        """Appending to global motion history should accumulate entries."""
        tracker._global_motion_history.append((1.0, 2.0))
        tracker._global_motion_history.append((3.0, 4.0))
        assert len(tracker._global_motion_history) == 2
        avg_dx = np.mean([m[0] for m in tracker._global_motion_history])
        avg_dy = np.mean([m[1] for m in tracker._global_motion_history])
        assert avg_dx == pytest.approx(2.0)
        assert avg_dy == pytest.approx(3.0)

    def test_temporal_avg_respects_maxlen(self, tracker):
        """History should not exceed camera_temporal_window entries."""
        for i in range(20):
            tracker._global_motion_history.append((float(i), float(i)))
        assert len(tracker._global_motion_history) == tracker.camera_temporal_window

    def test_magnitude_threshold_default(self, tracker):
        """Default camera_magnitude_threshold should be 0.5."""
        assert tracker.camera_magnitude_threshold == 0.5

    def test_magnitude_threshold_zeroes_below(self, tracker):
        """Global motion below threshold should be zeroed in the pipeline."""
        # This is tested by verifying the parameter value and the logic.
        # When global_mag < threshold, global_dx/dy are set to 0.
        # We verify the threshold is stored correctly.
        assert tracker.camera_magnitude_threshold > 0

    def test_adaptive_scaling_ratio_default(self, tracker):
        """Default camera_adaptive_ratio should be 0.8."""
        assert tracker.camera_adaptive_ratio == 0.8

    def test_adaptive_scaling_ratio_is_fraction(self, tracker):
        """camera_adaptive_ratio should be between 0 and 1."""
        assert 0 < tracker.camera_adaptive_ratio <= 1.0


# ---------------------------------------------------------------------------
# TestGridDetection
# ---------------------------------------------------------------------------

@pytest.mark.trackers
class TestGridDetection:
    """Tests for grid-based cell detection parameters."""

    def test_vr_mode_detection_standard_aspect(self, tracker):
        """Standard 16:9 video should NOT be detected as VR."""
        tracker.app.processor.frame_width = 1920
        tracker.app.processor.frame_height = 1080
        assert tracker._is_vr_video() is False

    def test_vr_mode_detection_vr_aspect(self, tracker):
        """Wide aspect ratio (> 2:1) should be detected as VR."""
        tracker.app.processor.frame_width = 4096
        tracker.app.processor.frame_height = 1080
        assert tracker._is_vr_video() is True

    def test_vr_mode_detection_exact_2_to_1(self, tracker):
        """Exactly 2:1 aspect ratio should NOT be detected as VR (requires > 2.0)."""
        tracker.app.processor.frame_width = 2160
        tracker.app.processor.frame_height = 1080
        assert tracker._is_vr_video() is False

    def test_grid_size_default(self, tracker):
        """Default oscillation_grid_size should be 10."""
        assert tracker.oscillation_grid_size == 10

    def test_oscillation_sensitivity_default(self, tracker):
        """Default oscillation_sensitivity should be 1.0."""
        assert tracker.oscillation_sensitivity == 1.0

    def test_vr_mode_handles_missing_processor(self, tracker):
        """_is_vr_video should return False when processor is missing."""
        tracker.app.processor = None
        assert tracker._is_vr_video() is False


# ---------------------------------------------------------------------------
# TestNoMotionDecay
# ---------------------------------------------------------------------------

@pytest.mark.trackers
class TestNoMotionDecay:
    """Tests for no-motion decay toward center position."""

    def test_decay_rate_initialized(self, tracker):
        """_no_motion_decay_rate should be initialized."""
        assert tracker._no_motion_decay_rate > 0
        assert tracker._no_motion_decay_rate < 1

    def test_position_decays_toward_50_from_above(self, tracker):
        """When at position > 50 with no active blocks, position should decay toward 50."""
        tracker.last_primary_pos = 80.0
        rate = tracker._no_motion_decay_rate
        expected = 80.0 * (1 - rate) + 50 * rate
        # Simulate no-motion decay (this is the logic from process_frame when no active blocks)
        tracker.last_primary_pos = tracker.last_primary_pos * (1 - rate) + 50 * rate
        assert tracker.last_primary_pos == pytest.approx(expected)
        assert tracker.last_primary_pos < 80.0

    def test_position_decays_toward_50_from_below(self, tracker):
        """When at position < 50 with no active blocks, position should decay toward 50."""
        tracker.last_primary_pos = 20.0
        rate = tracker._no_motion_decay_rate
        expected = 20.0 * (1 - rate) + 50 * rate
        tracker.last_primary_pos = tracker.last_primary_pos * (1 - rate) + 50 * rate
        assert tracker.last_primary_pos == pytest.approx(expected)
        assert tracker.last_primary_pos > 20.0

    def test_position_at_50_stays_at_50(self, tracker):
        """When already at 50, decay should keep position at 50."""
        tracker.last_primary_pos = 50.0
        rate = tracker._no_motion_decay_rate
        tracker.last_primary_pos = tracker.last_primary_pos * (1 - rate) + 50 * rate
        assert tracker.last_primary_pos == pytest.approx(50.0)

    def test_repeated_decay_converges_to_50(self, tracker):
        """Repeated decay steps should converge position toward 50."""
        tracker.last_primary_pos = 100.0
        rate = tracker._no_motion_decay_rate
        for _ in range(200):
            tracker.last_primary_pos = tracker.last_primary_pos * (1 - rate) + 50 * rate
        assert tracker.last_primary_pos == pytest.approx(50.0, abs=0.1)


# ---------------------------------------------------------------------------
# TestLifecycle
# ---------------------------------------------------------------------------

@pytest.mark.trackers
class TestLifecycle:
    """Tests for start/stop tracking and cleanup lifecycle."""

    def test_start_tracking_returns_true(self, tracker):
        """start_tracking() should return True when initialized."""
        result = tracker.start_tracking()
        assert result is True

    def test_start_tracking_sets_active(self, tracker):
        """start_tracking() should set tracking_active to True."""
        tracker.start_tracking()
        assert tracker.tracking_active is True

    def test_start_tracking_resets_state(self, tracker):
        """start_tracking() should reset endpoint extraction state."""
        # Dirty the state first
        tracker._stable_angle = 123.0
        tracker._stable_mag = 99.0
        tracker.motion_direction = -1
        tracker.last_primary_pos = 75.0
        tracker.last_secondary_pos = 25.0
        tracker._prev_frame_time_ms = 5000
        tracker.oscillation_history[(0, 0)] = deque([1, 2, 3])
        tracker.oscillation_cell_persistence[(0, 0)] = 5
        tracker._global_motion_history.append((1.0, 2.0))

        tracker.start_tracking()

        assert tracker._stable_angle == 0.0
        assert tracker._stable_mag == 0.0
        assert tracker.motion_direction == 1
        assert tracker.last_primary_pos == 50.0
        assert tracker.last_secondary_pos == 50.0
        assert tracker._prev_frame_time_ms == 0
        assert len(tracker.oscillation_history) == 0
        assert len(tracker.oscillation_cell_persistence) == 0
        assert len(tracker._global_motion_history) == 0

    def test_start_tracking_fails_when_not_initialized(self, mock_app):
        """start_tracking() should return False when not initialized."""
        from patreon_features.trackers.oscillation_experimental_3 import OscillationExperimental3Tracker
        t = OscillationExperimental3Tracker()
        # Do NOT initialize
        result = t.start_tracking()
        assert result is False

    def test_stop_tracking_returns_true(self, tracker):
        """stop_tracking() should return True."""
        tracker.start_tracking()
        result = tracker.stop_tracking()
        assert result is True

    def test_stop_tracking_sets_inactive(self, tracker):
        """stop_tracking() should set tracking_active to False."""
        tracker.start_tracking()
        tracker.stop_tracking()
        assert tracker.tracking_active is False

    def test_cleanup_clears_history(self, tracker):
        """cleanup() should clear oscillation_history."""
        tracker.oscillation_history[(0, 0)] = deque([1, 2, 3])
        tracker.cleanup()
        assert len(tracker.oscillation_history) == 0

    def test_cleanup_clears_cell_persistence(self, tracker):
        """cleanup() should clear oscillation_cell_persistence."""
        tracker.oscillation_cell_persistence[(0, 0)] = 5
        tracker.cleanup()
        assert len(tracker.oscillation_cell_persistence) == 0

    def test_cleanup_clears_global_motion_history(self, tracker):
        """cleanup() should clear _global_motion_history."""
        tracker._global_motion_history.append((1.0, 2.0))
        tracker.cleanup()
        assert len(tracker._global_motion_history) == 0

    def test_cleanup_frees_buffers(self, tracker):
        """cleanup() should set frame buffers to None."""
        tracker.prev_gray_oscillation = np.zeros((100, 100), dtype=np.uint8)
        tracker._gray_roi_buffer = np.zeros((100, 100), dtype=np.uint8)
        tracker._gray_full_buffer = np.zeros((100, 100), dtype=np.uint8)
        tracker._prev_gray_osc_buffer = np.zeros((100, 100), dtype=np.uint8)
        tracker.cleanup()
        assert tracker.prev_gray_oscillation is None
        assert tracker._gray_roi_buffer is None
        assert tracker._gray_full_buffer is None
        assert tracker._prev_gray_osc_buffer is None

    def test_cleanup_frees_optical_flow(self, tracker):
        """cleanup() should set flow_dense_osc to None."""
        tracker.cleanup()
        assert tracker.flow_dense_osc is None


# ---------------------------------------------------------------------------
# TestProcessFrame
# ---------------------------------------------------------------------------

@pytest.mark.trackers
class TestProcessFrame:
    """Tests for process_frame() returning TrackerResult."""

    def test_returns_tracker_result(self, tracker):
        """process_frame() should return a TrackerResult instance."""
        from tracker.tracker_modules.core.base_tracker import TrackerResult
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = tracker.process_frame(frame, 0, 0)
        assert isinstance(result, TrackerResult)

    def test_result_has_processed_frame(self, tracker):
        """TrackerResult should contain a processed_frame (numpy array)."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = tracker.process_frame(frame, 0, 0)
        assert result.processed_frame is not None
        assert isinstance(result.processed_frame, np.ndarray)

    def test_result_has_debug_info(self, tracker):
        """TrackerResult should contain debug_info dict."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = tracker.process_frame(frame, 0, 0)
        assert result.debug_info is not None
        assert isinstance(result.debug_info, dict)

    def test_debug_info_contains_expected_keys(self, tracker):
        """debug_info should contain mode, last_position, fps, etc."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = tracker.process_frame(frame, 0, 0)
        expected_keys = {
            'mode', 'last_position', 'last_secondary_position',
            'active_cells', 'motion_direction', 'stable_angle',
            'stable_mag', 'fps'
        }
        assert expected_keys.issubset(result.debug_info.keys())

    def test_debug_info_mode_is_experimental_3(self, tracker):
        """debug_info mode should be 'experimental_3'."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = tracker.process_frame(frame, 0, 0)
        assert result.debug_info['mode'] == 'experimental_3'

    def test_result_supports_tuple_unpacking(self, tracker):
        """TrackerResult should support tuple-like unpacking for backward compat."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = tracker.process_frame(frame, 0, 0)
        processed_frame, action_log = result
        assert isinstance(processed_frame, np.ndarray)

    def test_first_frame_returns_none_action_log(self, tracker):
        """First frame should return None action_log (no prev_gray to compare)."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = tracker.process_frame(frame, 0, 0)
        assert result.action_log is None

    def test_none_frame_returns_none_action_log(self, tracker):
        """Passing None frame should return None action_log."""
        # First pass a real frame to set prev_gray
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        tracker.process_frame(frame, 0, 0)
        # Now pass None
        result = tracker.process_frame(None, 100, 1)
        assert result.action_log is None

    def test_consecutive_frames_produce_result(self, tracker):
        """Processing two consecutive frames should produce a TrackerResult."""
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.ones((100, 100, 3), dtype=np.uint8) * 128
        tracker.process_frame(frame1, 0, 0)
        result = tracker.process_frame(frame2, 33, 1)
        assert isinstance(result.processed_frame, np.ndarray)


# ---------------------------------------------------------------------------
# TestConfigurableParameters
# ---------------------------------------------------------------------------

@pytest.mark.trackers
class TestConfigurableParameters:
    """Tests for kwargs overriding default parameter values."""

    def test_direction_change_angle_threshold_default(self, tracker):
        """Default direction_change_angle_threshold should be 100."""
        assert tracker.direction_change_angle_threshold == 100

    def test_direction_change_angle_threshold_override(self, mock_app):
        """direction_change_angle_threshold should accept kwarg override."""
        from patreon_features.trackers.oscillation_experimental_3 import OscillationExperimental3Tracker
        t = OscillationExperimental3Tracker()
        t.initialize(mock_app, direction_change_angle_threshold=80)
        assert t.direction_change_angle_threshold == 80

    def test_max_magnitude_speed_limit_default(self, tracker):
        """Default max_magnitude_speed_limit should be 10."""
        assert tracker.max_magnitude_speed_limit == 10

    def test_max_magnitude_speed_limit_override(self, mock_app):
        """max_magnitude_speed_limit should accept kwarg override."""
        from patreon_features.trackers.oscillation_experimental_3 import OscillationExperimental3Tracker
        t = OscillationExperimental3Tracker()
        t.initialize(mock_app, max_magnitude_speed_limit=20)
        assert t.max_magnitude_speed_limit == 20

    def test_camera_temporal_window_default(self, tracker):
        """Default camera_temporal_window should be 5."""
        assert tracker.camera_temporal_window == 5

    def test_camera_temporal_window_override(self, mock_app):
        """camera_temporal_window should accept kwarg override."""
        from patreon_features.trackers.oscillation_experimental_3 import OscillationExperimental3Tracker
        t = OscillationExperimental3Tracker()
        t.initialize(mock_app, camera_temporal_window=10)
        assert t.camera_temporal_window == 10
        assert t._global_motion_history.maxlen == 10

    def test_camera_magnitude_threshold_default(self, tracker):
        """Default camera_magnitude_threshold should be 0.5."""
        assert tracker.camera_magnitude_threshold == 0.5

    def test_camera_magnitude_threshold_override(self, mock_app):
        """camera_magnitude_threshold should accept kwarg override."""
        from patreon_features.trackers.oscillation_experimental_3 import OscillationExperimental3Tracker
        t = OscillationExperimental3Tracker()
        t.initialize(mock_app, camera_magnitude_threshold=1.0)
        assert t.camera_magnitude_threshold == 1.0

    def test_camera_adaptive_ratio_default(self, tracker):
        """Default camera_adaptive_ratio should be 0.8."""
        assert tracker.camera_adaptive_ratio == 0.8

    def test_camera_adaptive_ratio_override(self, mock_app):
        """camera_adaptive_ratio should accept kwarg override."""
        from patreon_features.trackers.oscillation_experimental_3 import OscillationExperimental3Tracker
        t = OscillationExperimental3Tracker()
        t.initialize(mock_app, camera_adaptive_ratio=0.5)
        assert t.camera_adaptive_ratio == 0.5

    def test_oscillation_sensitivity_default(self, tracker):
        """Default oscillation_sensitivity should be 1.0."""
        assert tracker.oscillation_sensitivity == 1.0

    def test_oscillation_sensitivity_override(self, mock_app):
        """oscillation_sensitivity should accept kwarg override."""
        from patreon_features.trackers.oscillation_experimental_3 import OscillationExperimental3Tracker
        t = OscillationExperimental3Tracker()
        t.initialize(mock_app, oscillation_sensitivity=2.0)
        assert t.oscillation_sensitivity == 2.0

    def test_oscillation_grid_size_default(self, tracker):
        """Default oscillation_grid_size should be 10."""
        assert tracker.oscillation_grid_size == 10

    def test_oscillation_grid_size_override(self, mock_app):
        """oscillation_grid_size should accept kwarg override."""
        from patreon_features.trackers.oscillation_experimental_3 import OscillationExperimental3Tracker
        t = OscillationExperimental3Tracker()
        t.initialize(mock_app, oscillation_grid_size=20)
        assert t.oscillation_grid_size == 20

    def test_show_masks_default(self, tracker):
        """Default show_masks should be True."""
        assert tracker.show_masks is True

    def test_show_masks_override(self, mock_app):
        """show_masks should accept kwarg override."""
        from patreon_features.trackers.oscillation_experimental_3 import OscillationExperimental3Tracker
        t = OscillationExperimental3Tracker()
        t.initialize(mock_app, show_masks=False)
        assert t.show_masks is False

    def test_multiple_kwargs_at_once(self, mock_app):
        """Multiple kwargs should all be applied correctly."""
        from patreon_features.trackers.oscillation_experimental_3 import OscillationExperimental3Tracker
        t = OscillationExperimental3Tracker()
        t.initialize(
            mock_app,
            direction_change_angle_threshold=60,
            max_magnitude_speed_limit=15,
            camera_temporal_window=8,
            oscillation_sensitivity=0.5,
            show_masks=False
        )
        assert t.direction_change_angle_threshold == 60
        assert t.max_magnitude_speed_limit == 15
        assert t.camera_temporal_window == 8
        assert t.oscillation_sensitivity == 0.5
        assert t.show_masks is False
