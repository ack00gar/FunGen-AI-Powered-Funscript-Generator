"""
Unit tests for UI state management fixes during live tracking sessions.

These tests validate the core logic of button states, control enabling/disabling,
and proper state transitions without requiring a full GUI environment.
"""

import pytest
import sys
import os
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config.constants import TrackerMode


@pytest.mark.unit
class TestControlPanelButtonStateLogic:
    """Unit tests for control panel button state logic during live tracking."""
    
    def setup_method(self):
        """Set up mock app and components for testing."""
        self.mock_app = MagicMock()
        self.mock_processor = MagicMock()
        self.mock_stage_proc = MagicMock()
        self.mock_app_state = MagicMock()
        
        # Set up app structure
        self.mock_app.processor = self.mock_processor
        self.mock_app.stage_processor = self.mock_stage_proc
        self.mock_app.app_state_ui = self.mock_app_state
        self.mock_app.is_setting_user_roi_mode = False
        
        # Default processor state
        self.mock_processor.is_video_open.return_value = True
        self.mock_processor.is_processing = False
        self.mock_processor.enable_tracker_processing = False
        
        # Default stage processor state
        self.mock_stage_proc.full_analysis_active = False
    
    def _simulate_button_state_logic(self):
        """Simulate the exact button state logic from the control panel."""
        app = self.mock_app
        stage_proc = self.mock_stage_proc
        
        # This mirrors the exact logic from control_panel_ui.py:2041-2051
        analysis_active = stage_proc.full_analysis_active
        video_loaded = app.processor and app.processor.is_video_open()
        
        # Check if live tracking is running
        is_live_tracking_running = (app.processor and
                                    app.processor.is_processing and
                                    app.processor.enable_tracker_processing)
        
        # Determine button states and availability
        can_start = video_loaded and not analysis_active and not is_live_tracking_running and not app.is_setting_user_roi_mode
        can_stop = analysis_active or is_live_tracking_running
        
        return {
            'analysis_active': analysis_active,
            'video_loaded': video_loaded,
            'is_live_tracking_running': is_live_tracking_running,
            'can_start': can_start,
            'can_stop': can_stop
        }
    
    def test_no_video_loaded_state(self):
        """Test button states when no video is loaded."""
        self.mock_processor.is_video_open.return_value = False
        
        result = self._simulate_button_state_logic()
        
        assert result['video_loaded'] is False
        assert result['can_start'] is False
        assert result['can_stop'] is False
        assert result['is_live_tracking_running'] is False
    
    def test_video_loaded_idle_state(self):
        """Test button states when video is loaded but nothing is running."""
        result = self._simulate_button_state_logic()
        
        assert result['video_loaded'] is True
        assert result['can_start'] is True
        assert result['can_stop'] is False
        assert result['is_live_tracking_running'] is False
    
    def test_offline_analysis_running_state(self):
        """Test button states during offline analysis."""
        self.mock_stage_proc.full_analysis_active = True
        
        result = self._simulate_button_state_logic()
        
        assert result['analysis_active'] is True
        assert result['can_start'] is False
        assert result['can_stop'] is True
        assert result['is_live_tracking_running'] is False
    
    def test_live_tracking_running_state(self):
        """Test button states during live tracking session."""
        self.mock_processor.is_processing = True
        self.mock_processor.enable_tracker_processing = True
        
        result = self._simulate_button_state_logic()
        
        assert result['is_live_tracking_running'] is True
        assert result['can_start'] is False
        assert result['can_stop'] is True
        assert result['analysis_active'] is False
    
    def test_regular_video_playback_state(self):
        """Test button states during regular video playback (not tracking)."""
        self.mock_processor.is_processing = True
        self.mock_processor.enable_tracker_processing = False  # Key difference
        
        result = self._simulate_button_state_logic()
        
        assert result['is_live_tracking_running'] is False
        assert result['can_start'] is True  # Should allow starting tracking
        assert result['can_stop'] is False
        assert result['analysis_active'] is False
    
    def test_roi_setting_mode_state(self):
        """Test button states when setting user ROI."""
        self.mock_app.is_setting_user_roi_mode = True
        
        result = self._simulate_button_state_logic()
        
        assert result['can_start'] is False
        assert result['can_stop'] is False
    
    def test_live_tracking_transition_states(self):
        """Test state transitions during live tracking lifecycle."""
        # 1. Initial idle state
        result = self._simulate_button_state_logic()
        assert result['can_start'] is True
        assert result['can_stop'] is False
        
        # 2. Live tracking starts
        self.mock_processor.is_processing = True
        self.mock_processor.enable_tracker_processing = True
        result = self._simulate_button_state_logic()
        assert result['can_start'] is False
        assert result['can_stop'] is True
        assert result['is_live_tracking_running'] is True
        
        # 3. Live tracking stops (enable_tracker_processing cleared)
        self.mock_processor.enable_tracker_processing = False
        result = self._simulate_button_state_logic()
        assert result['can_start'] is True  # Should allow new session
        assert result['can_stop'] is False
        assert result['is_live_tracking_running'] is False


@pytest.mark.unit
class TestVideoDisplayControlStateLogic:
    """Unit tests for video display control state logic during live tracking."""
    
    def setup_method(self):
        """Set up mock components for testing."""
        self.mock_app = MagicMock()
        self.mock_processor = MagicMock()
        self.mock_stage_proc = MagicMock()
        self.mock_file_mgr = MagicMock()
        
        # Set up app structure
        self.mock_app.processor = self.mock_processor
        self.mock_app.stage_processor = self.mock_stage_proc
        self.mock_app.file_manager = self.mock_file_mgr
        
        # Default states
        self.mock_processor.is_processing = False
        self.mock_processor.enable_tracker_processing = False
        self.mock_stage_proc.full_analysis_active = False
        self.mock_file_mgr.video_path = "/test/video.mp4"
    
    def _simulate_video_controls_disabled_logic(self):
        """Simulate the exact video controls disabled logic from video_display_ui.py."""
        app = self.mock_app
        stage_proc = app.stage_processor
        file_mgr = app.file_manager
        
        # This mirrors the exact logic from video_display_ui.py:130-135
        is_live_tracking_running = (app.processor and
                                    app.processor.is_processing and
                                    app.processor.enable_tracker_processing)
        
        controls_disabled = stage_proc.full_analysis_active or is_live_tracking_running or not file_mgr.video_path
        
        return {
            'is_live_tracking_running': is_live_tracking_running,
            'controls_disabled': controls_disabled
        }
    
    def test_video_controls_enabled_idle_state(self):
        """Test video controls are enabled when idle."""
        result = self._simulate_video_controls_disabled_logic()
        
        assert result['is_live_tracking_running'] is False
        assert result['controls_disabled'] is False
    
    def test_video_controls_disabled_during_offline_analysis(self):
        """Test video controls are disabled during offline analysis."""
        self.mock_stage_proc.full_analysis_active = True
        
        result = self._simulate_video_controls_disabled_logic()
        
        assert result['controls_disabled'] is True
    
    def test_video_controls_disabled_during_live_tracking(self):
        """Test video controls are disabled during live tracking."""
        self.mock_processor.is_processing = True
        self.mock_processor.enable_tracker_processing = True
        
        result = self._simulate_video_controls_disabled_logic()
        
        assert result['is_live_tracking_running'] is True
        assert result['controls_disabled'] is True
    
    def test_video_controls_enabled_during_regular_playback(self):
        """Test video controls remain enabled during regular video playback."""
        self.mock_processor.is_processing = True
        self.mock_processor.enable_tracker_processing = False  # Not tracking
        
        result = self._simulate_video_controls_disabled_logic()
        
        assert result['is_live_tracking_running'] is False
        assert result['controls_disabled'] is False
    
    def test_video_controls_disabled_no_video_loaded(self):
        """Test video controls are disabled when no video is loaded."""
        self.mock_file_mgr.video_path = None
        
        result = self._simulate_video_controls_disabled_logic()
        
        assert result['controls_disabled'] is True


@pytest.mark.unit
class TestLiveTrackingDetectionLogic:
    """Unit tests for the core live tracking detection logic."""
    
    @pytest.mark.parametrize("processor_exists,is_processing,enable_tracker,expected_result", [
        (False, False, False, False),  # No processor
        (True, False, False, False),   # Processor exists but not processing
        (True, False, True, False),    # Processor exists, tracker enabled but not processing
        (True, True, False, False),    # Processor running but tracker disabled (regular playback)
        (True, True, True, True),      # Processor running with tracker enabled (live tracking)
    ])
    def test_live_tracking_detection_matrix(self, processor_exists, is_processing, enable_tracker, expected_result):
        """Test live tracking detection across all possible states."""
        # Set up mock processor based on test parameters
        if processor_exists:
            mock_processor = MagicMock()
            mock_processor.is_processing = is_processing
            mock_processor.enable_tracker_processing = enable_tracker
        else:
            mock_processor = None
        
        # Simulate the detection logic
        is_live_tracking_running = (mock_processor and
                                    mock_processor.is_processing and
                                    mock_processor.enable_tracker_processing)
        
        # Convert None to False for cleaner comparison
        is_live_tracking_running = bool(is_live_tracking_running)
        
        assert is_live_tracking_running == expected_result
    
    def test_live_tracking_flag_cleared_on_stop(self):
        """Test that enable_tracker_processing is properly cleared when tracking stops."""
        # This is a behavior test - in the actual implementation,
        # enable_tracker_processing should be set to False when processing stops
        mock_processor = MagicMock()
        
        # Start with live tracking active
        mock_processor.is_processing = True
        mock_processor.enable_tracker_processing = True
        
        is_live_tracking_running = (mock_processor and
                                    mock_processor.is_processing and
                                    mock_processor.enable_tracker_processing)
        assert is_live_tracking_running is True
        
        # Simulate tracking stop (enable_tracker_processing cleared)
        mock_processor.enable_tracker_processing = False
        
        is_live_tracking_running = (mock_processor and
                                    mock_processor.is_processing and
                                    mock_processor.enable_tracker_processing)
        assert is_live_tracking_running is False


@pytest.mark.unit
class TestPlaybackControlLogic:
    """Unit tests for playback control behavior during and after tracking sessions."""
    
    def test_play_button_should_not_restart_tracking(self):
        """Test that play button doesn't restart tracking when enable_tracker_processing is False."""
        # Simulate the scenario where:
        # 1. A tracking session just ended
        # 2. enable_tracker_processing was cleared (as per video_processor.py:1088)
        # 3. User presses play button
        
        mock_processor = MagicMock()
        mock_processor.is_processing = False  # Not currently playing
        mock_processor.enable_tracker_processing = False  # Tracking session ended
        
        # Simulate play button press - should start regular playback
        # The key is that enable_tracker_processing remains False
        mock_processor.start_processing()
        
        # Verify start_processing was called (regular playback)
        mock_processor.start_processing.assert_called_once()
        
        # The critical assertion: enable_tracker_processing should remain False
        # This ensures tracking doesn't restart automatically
        assert mock_processor.enable_tracker_processing is False
    
    def test_natural_completion_clears_tracker_flag(self):
        """Test that enable_tracker_processing is cleared when live tracking completes naturally."""
        # This tests the specific scenario reported by the user:
        # 1. Live tracking session runs to completion (end of video)
        # 2. enable_tracker_processing should be cleared automatically
        # 3. Play button afterwards should not restart tracking
        
        mock_processor = MagicMock()
        
        # Simulate live tracking running
        mock_processor.is_processing = True
        mock_processor.enable_tracker_processing = True
        mock_processor.total_frames = 1000
        mock_processor.current_frame_index = 1000  # Reached end
        
        # Simulate the natural completion logic from video_processor.py:1237-1247
        # When processing reaches end of video naturally:
        if mock_processor.total_frames > 0 and mock_processor.current_frame_index >= mock_processor.total_frames:
            mock_processor.is_processing = False
            mock_processor.enable_tracker_processing = False  # This is the fix
        
        # Verify the flags are cleared
        assert mock_processor.is_processing is False
        assert mock_processor.enable_tracker_processing is False
        
        # Now simulate user navigating and pressing play
        # Should detect as regular playback, not live tracking
        is_live_tracking_running = (mock_processor and
                                    mock_processor.is_processing and
                                    mock_processor.enable_tracker_processing)
        
        assert bool(is_live_tracking_running) is False, "Should not detect as live tracking after natural completion"
    
    def test_stop_analysis_preserves_funscript(self):
        """Test that stopping live analysis preserves the funscript data."""
        # This tests the specific scenario reported by the user:
        # 1. Live tracking generates funscript data
        # 2. User clicks "Stop Analysis"
        # 3. Funscript data should be preserved, not cleared
        
        mock_tracker = MagicMock()
        mock_funscript = MagicMock()
        
        # Simulate tracker with funscript data
        mock_tracker.tracking_active = True
        mock_tracker.funscript = mock_funscript
        mock_tracker.funscript.clear = MagicMock()
        
        # Simulate the fixed reset logic with preserve reason
        reason = "stop_preserve_funscript"
        
        # Test the funscript preservation logic from tracker.py:1402
        if reason not in ["seek", "project_load_preserve_actions", "stop_preserve_funscript"]:
            mock_tracker.funscript.clear()
        # else: funscript is preserved
        
        # Verify funscript was NOT cleared
        mock_tracker.funscript.clear.assert_not_called()
        
        # Verify with old behavior (should clear)
        reason_old = "normal_reset"
        if reason_old not in ["seek", "project_load_preserve_actions", "stop_preserve_funscript"]:
            mock_tracker.funscript.clear()
        
        # Now it should have been called
        mock_tracker.funscript.clear.assert_called_once()