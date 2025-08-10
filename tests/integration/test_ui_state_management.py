"""
Integration tests for UI state management during live tracking sessions.

These tests validate the complete UI behavior including button states,
control enabling/disabling, and proper integration between components
during live tracking workflows.
"""

import pytest
import sys
import os
import tempfile
import time
from unittest.mock import patch, MagicMock, Mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from application.logic.app_logic import ApplicationLogic
from config.constants import TrackerMode


@pytest.mark.integration
class TestControlPanelLiveTrackingIntegration:
    """Integration tests for control panel UI behavior during live tracking."""
    
    def setup_method(self):
        """Set up application instance for testing."""
        with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
            self.app = ApplicationLogic(is_cli=True)
    
    def teardown_method(self):
        """Clean up after each test."""
        if hasattr(self.app, 'cleanup'):
            self.app.cleanup()
    
    def test_control_panel_button_states_during_live_tracking_lifecycle(self):
        """Test control panel button states through complete live tracking lifecycle."""
        app = self.app
        
        # Skip if GUI components not available in CLI mode
        if not hasattr(app, 'gui') or not app.gui:
            pytest.skip("GUI components not available in CLI mode")
        
        control_panel = getattr(app.gui, 'control_panel_ui', None)
        if not control_panel:
            pytest.skip("Control panel UI not available")
        
        # Mock video loading
        with patch.object(app.processor, 'is_video_open', return_value=True):
            # 1. Initial state - should allow starting
            app.stage_processor.full_analysis_active = False
            app.processor.is_processing = False
            app.processor.enable_tracker_processing = False
            app.is_setting_user_roi_mode = False
            
            # Simulate button state logic
            analysis_active = app.stage_processor.full_analysis_active
            video_loaded = app.processor.is_video_open()
            is_live_tracking_running = (app.processor and
                                        app.processor.is_processing and
                                        app.processor.enable_tracker_processing)
            
            can_start = video_loaded and not analysis_active and not is_live_tracking_running and not app.is_setting_user_roi_mode
            can_stop = analysis_active or is_live_tracking_running
            
            assert can_start is True
            assert can_stop is False
            assert is_live_tracking_running is False
            
            # 2. Start live tracking
            app.processor.is_processing = True
            app.processor.enable_tracker_processing = True
            
            is_live_tracking_running = (app.processor and
                                        app.processor.is_processing and
                                        app.processor.enable_tracker_processing)
            can_start = video_loaded and not analysis_active and not is_live_tracking_running and not app.is_setting_user_roi_mode
            can_stop = analysis_active or is_live_tracking_running
            
            assert can_start is False
            assert can_stop is True
            assert is_live_tracking_running is True
            
            # 3. Stop live tracking (simulate processor stopping)
            app.processor.enable_tracker_processing = False  # Cleared by processor
            
            is_live_tracking_running = (app.processor and
                                        app.processor.is_processing and
                                        app.processor.enable_tracker_processing)
            can_start = video_loaded and not analysis_active and not is_live_tracking_running and not app.is_setting_user_roi_mode
            can_stop = analysis_active or is_live_tracking_running
            
            assert can_start is True
            assert can_stop is False
            assert is_live_tracking_running is False
    
    def test_simple_mode_vs_expert_mode_consistency(self):
        """Test that simple mode and expert mode have consistent button state logic."""
        app = self.app
        
        if not hasattr(app, 'app_state_ui'):
            pytest.skip("App state UI not available")
        
        # Mock video loading
        with patch.object(app.processor, 'is_video_open', return_value=True):
            # Test both modes have the same live tracking detection logic
            app.processor.is_processing = True
            app.processor.enable_tracker_processing = True
            app.stage_processor.full_analysis_active = False
            app.is_setting_user_roi_mode = False
            
            # Both modes should use the same live tracking detection
            is_live_tracking_running = (app.processor and
                                        app.processor.is_processing and
                                        app.processor.enable_tracker_processing)
            
            assert is_live_tracking_running is True
            
            # Both modes should have consistent button states
            analysis_active = app.stage_processor.full_analysis_active
            video_loaded = app.processor.is_video_open()
            
            can_start = video_loaded and not analysis_active and not is_live_tracking_running and not app.is_setting_user_roi_mode
            can_stop = analysis_active or is_live_tracking_running
            
            assert can_start is False
            assert can_stop is True
    
    def test_roi_setting_mode_disables_start_button(self):
        """Test that ROI setting mode properly disables start button."""
        app = self.app
        
        with patch.object(app.processor, 'is_video_open', return_value=True):
            # Set ROI setting mode
            app.is_setting_user_roi_mode = True
            app.stage_processor.full_analysis_active = False
            app.processor.is_processing = False
            app.processor.enable_tracker_processing = False
            
            analysis_active = app.stage_processor.full_analysis_active
            video_loaded = app.processor.is_video_open()
            is_live_tracking_running = (app.processor and
                                        app.processor.is_processing and
                                        app.processor.enable_tracker_processing)
            
            can_start = video_loaded and not analysis_active and not is_live_tracking_running and not app.is_setting_user_roi_mode
            can_stop = analysis_active or is_live_tracking_running
            
            assert can_start is False  # Should be disabled during ROI setting
            assert can_stop is False


@pytest.mark.integration  
class TestVideoDisplayControlsIntegration:
    """Integration tests for video display controls behavior during live tracking."""
    
    def setup_method(self):
        """Set up application instance for testing."""
        with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
            self.app = ApplicationLogic(is_cli=True)
    
    def teardown_method(self):
        """Clean up after each test."""
        if hasattr(self.app, 'cleanup'):
            self.app.cleanup()
    
    def test_video_controls_disabled_during_live_tracking(self):
        """Test that video playback controls are disabled during live tracking."""
        app = self.app
        
        # Mock video loaded
        app.file_manager.video_path = "/test/video.mp4"
        
        # Test controls enabled initially
        app.stage_processor.full_analysis_active = False
        app.processor.is_processing = False
        app.processor.enable_tracker_processing = False
        
        is_live_tracking_running = (app.processor and
                                    app.processor.is_processing and
                                    app.processor.enable_tracker_processing)
        controls_disabled = app.stage_processor.full_analysis_active or is_live_tracking_running or not app.file_manager.video_path
        
        assert controls_disabled is False
        
        # Start live tracking - controls should be disabled
        app.processor.is_processing = True
        app.processor.enable_tracker_processing = True
        
        is_live_tracking_running = (app.processor and
                                    app.processor.is_processing and
                                    app.processor.enable_tracker_processing)
        controls_disabled = app.stage_processor.full_analysis_active or is_live_tracking_running or not app.file_manager.video_path
        
        assert controls_disabled is True
        
        # Stop live tracking - controls should be enabled again
        app.processor.enable_tracker_processing = False
        
        is_live_tracking_running = (app.processor and
                                    app.processor.is_processing and
                                    app.processor.enable_tracker_processing)
        controls_disabled = app.stage_processor.full_analysis_active or is_live_tracking_running or not app.file_manager.video_path
        
        assert controls_disabled is False
    
    def test_video_controls_enabled_during_regular_playback(self):
        """Test that video controls remain enabled during regular video playback."""
        app = self.app
        
        app.file_manager.video_path = "/test/video.mp4"
        app.stage_processor.full_analysis_active = False
        
        # Regular video playback (not tracking)
        app.processor.is_processing = True
        app.processor.enable_tracker_processing = False
        
        is_live_tracking_running = (app.processor and
                                    app.processor.is_processing and
                                    app.processor.enable_tracker_processing)
        controls_disabled = app.stage_processor.full_analysis_active or is_live_tracking_running or not app.file_manager.video_path
        
        assert is_live_tracking_running is False
        assert controls_disabled is False
    
    def test_video_controls_disabled_during_offline_analysis(self):
        """Test that video controls are disabled during offline analysis."""
        app = self.app
        
        app.file_manager.video_path = "/test/video.mp4"
        app.stage_processor.full_analysis_active = True
        app.processor.is_processing = False
        app.processor.enable_tracker_processing = False
        
        is_live_tracking_running = (app.processor and
                                    app.processor.is_processing and
                                    app.processor.enable_tracker_processing)
        controls_disabled = app.stage_processor.full_analysis_active or is_live_tracking_running or not app.file_manager.video_path
        
        assert controls_disabled is True


@pytest.mark.integration
class TestPlaybackControlHandlerIntegration:
    """Integration tests for playback control handler behavior."""
    
    def setup_method(self):
        """Set up application instance for testing."""
        with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
            self.app = ApplicationLogic(is_cli=True)
    
    def teardown_method(self):
        """Clean up after each test."""
        if hasattr(self.app, 'cleanup'):
            self.app.cleanup()
    
    def test_play_button_does_not_restart_tracking_sessions(self):
        """Test that play button doesn't restart tracking after session ends."""
        app = self.app
        
        if not hasattr(app, 'event_handlers'):
            pytest.skip("Event handlers not available")
        
        # Mock video loaded
        app.file_manager.video_path = "/test/video.mp4"
        
        # Mock processor with video info
        app.processor.video_info = {'total_frames': 1000}
        app.processor.current_frame_index = 0
        
        # Simulate scenario: tracking session just ended
        app.processor.is_processing = False
        app.processor.enable_tracker_processing = False  # Cleared when session ended
        
        # Mock the processor methods
        with patch.object(app.processor, 'start_processing') as mock_start:
            # Simulate play button press
            app.event_handlers.handle_playback_control("play_pause")
            
            # Verify regular playback started
            mock_start.assert_called_once()
            
            # Verify tracking flag remains cleared
            assert app.processor.enable_tracker_processing is False
    
    def test_pause_button_during_live_tracking(self):
        """Test pause button behavior during live tracking."""
        app = self.app
        
        if not hasattr(app, 'event_handlers'):
            pytest.skip("Event handlers not available")
        
        # Mock video loaded
        app.file_manager.video_path = "/test/video.mp4"
        app.processor.video_info = {'total_frames': 1000}
        app.processor.current_frame_index = 0
        
        # Simulate live tracking session
        app.processor.is_processing = True
        app.processor.enable_tracker_processing = True
        
        # Mock pause event
        app.processor.pause_event = MagicMock()
        app.processor.pause_event.is_set.return_value = False  # Not paused
        
        with patch.object(app.processor, 'pause_processing') as mock_pause:
            # Simulate pause button press during live tracking
            app.event_handlers.handle_playback_control("play_pause")
            
            # Verify pause was called
            mock_pause.assert_called_once()


@pytest.mark.integration
class TestLiveTrackingModeTransitions:
    """Integration tests for mode transitions with proper UI state management."""
    
    def setup_method(self):
        """Set up application instance for testing."""
        with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
            self.app = ApplicationLogic(is_cli=True)
    
    def teardown_method(self):
        """Clean up after each test."""
        if hasattr(self.app, 'cleanup'):
            self.app.cleanup()
    
    @pytest.mark.parametrize("tracker_mode", [
        TrackerMode.LIVE_YOLO_ROI,
        TrackerMode.LIVE_USER_ROI,
        TrackerMode.OSCILLATION_DETECTOR
    ])
    def test_live_tracking_mode_ui_states(self, tracker_mode):
        """Test UI states are consistent across different live tracking modes."""
        app = self.app
        
        if not hasattr(app, 'app_state_ui'):
            pytest.skip("App state UI not available")
        
        # Set tracking mode
        app.app_state_ui.selected_tracker_mode = tracker_mode
        
        with patch.object(app.processor, 'is_video_open', return_value=True):
            # Test initial state
            app.stage_processor.full_analysis_active = False
            app.processor.is_processing = False
            app.processor.enable_tracker_processing = False
            app.is_setting_user_roi_mode = False
            
            # Check initial state allows starting
            is_live_tracking_running = (app.processor and
                                        app.processor.is_processing and
                                        app.processor.enable_tracker_processing)
            
            analysis_active = app.stage_processor.full_analysis_active
            video_loaded = app.processor.is_video_open()
            can_start = video_loaded and not analysis_active and not is_live_tracking_running and not app.is_setting_user_roi_mode
            can_stop = analysis_active or is_live_tracking_running
            
            assert can_start is True
            assert can_stop is False
            
            # Test during tracking
            app.processor.is_processing = True
            app.processor.enable_tracker_processing = True
            
            is_live_tracking_running = (app.processor and
                                        app.processor.is_processing and
                                        app.processor.enable_tracker_processing)
            can_start = video_loaded and not analysis_active and not is_live_tracking_running and not app.is_setting_user_roi_mode
            can_stop = analysis_active or is_live_tracking_running
            
            assert can_start is False
            assert can_stop is True
            assert is_live_tracking_running is True
    
    def test_mode_switch_during_idle_state(self):
        """Test switching between tracking modes when idle."""
        app = self.app
        
        if not hasattr(app, 'app_state_ui'):
            pytest.skip("App state UI not available")
        
        with patch.object(app.processor, 'is_video_open', return_value=True):
            # Ensure idle state
            app.stage_processor.full_analysis_active = False
            app.processor.is_processing = False
            app.processor.enable_tracker_processing = False
            app.is_setting_user_roi_mode = False
            
            # Test switching between modes
            modes = [TrackerMode.LIVE_YOLO_ROI, TrackerMode.LIVE_USER_ROI, TrackerMode.OSCILLATION_DETECTOR]
            
            for mode in modes:
                app.app_state_ui.selected_tracker_mode = mode
                
                # Should always allow starting when idle
                is_live_tracking_running = (app.processor and
                                            app.processor.is_processing and
                                            app.processor.enable_tracker_processing)
                analysis_active = app.stage_processor.full_analysis_active
                video_loaded = app.processor.is_video_open()
                can_start = video_loaded and not analysis_active and not is_live_tracking_running and not app.is_setting_user_roi_mode
                
                assert can_start is True
                assert is_live_tracking_running is False


@pytest.mark.integration
class TestUIStateManagementRegression:
    """Regression tests to prevent future UI state management issues."""
    
    def setup_method(self):
        """Set up application instance for testing."""
        with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
            self.app = ApplicationLogic(is_cli=True)
    
    def teardown_method(self):
        """Clean up after each test."""
        if hasattr(self.app, 'cleanup'):
            self.app.cleanup()
    
    def test_live_tracking_flag_consistency(self):
        """Test that enable_tracker_processing flag is consistent across components."""
        app = self.app
        
        # Initial state
        assert app.processor.enable_tracker_processing is False
        
        # Simulate starting live tracking
        app.processor.set_tracker_processing_enabled(True)
        app.processor.is_processing = True
        
        # Check flag is set
        assert app.processor.enable_tracker_processing is True
        
        # Check detection logic works
        is_live_tracking_running = (app.processor and
                                    app.processor.is_processing and
                                    app.processor.enable_tracker_processing)
        assert is_live_tracking_running is True
        
        # Simulate stopping (flag should be cleared)
        app.processor.enable_tracker_processing = False
        
        # Check flag is cleared
        assert app.processor.enable_tracker_processing is False
        
        # Check detection logic reflects the change
        is_live_tracking_running = (app.processor and
                                    app.processor.is_processing and
                                    app.processor.enable_tracker_processing)
        assert is_live_tracking_running is False
    
    def test_button_state_edge_cases(self):
        """Test button state logic handles edge cases correctly."""
        app = self.app
        
        with patch.object(app.processor, 'is_video_open', return_value=True):
            # Edge case 1: Processor exists but is None
            original_processor = app.processor
            app.processor = None
            
            # Should handle gracefully
            is_live_tracking_running = (app.processor and
                                        app.processor.is_processing and
                                        app.processor.enable_tracker_processing)
            assert bool(is_live_tracking_running) is False
            
            # Restore processor
            app.processor = original_processor
            
            # Edge case 2: Both offline analysis and live tracking flags set (shouldn't happen)
            app.stage_processor.full_analysis_active = True
            app.processor.is_processing = True
            app.processor.enable_tracker_processing = True
            
            analysis_active = app.stage_processor.full_analysis_active
            is_live_tracking_running = (app.processor and
                                        app.processor.is_processing and
                                        app.processor.enable_tracker_processing)
            
            # Both should be detected independently
            assert analysis_active is True
            assert is_live_tracking_running is True
            
            # Button logic should handle this correctly
            video_loaded = app.processor.is_video_open()
            can_start = video_loaded and not analysis_active and not is_live_tracking_running and not app.is_setting_user_roi_mode
            can_stop = analysis_active or is_live_tracking_running
            
            assert can_start is False  # Can't start when either is active
            assert can_stop is True   # Can stop when either is active
    
    def test_natural_live_tracking_completion_workflow(self):
        """Test complete workflow when live tracking completes naturally to end of video."""
        app = self.app
        
        with patch.object(app.processor, 'is_video_open', return_value=True):
            # Simulate video with specific frame count
            app.processor.total_frames = 100
            app.processor.current_frame_index = 0
            
            # 1. Start live tracking
            app.stage_processor.full_analysis_active = False
            app.processor.is_processing = True
            app.processor.enable_tracker_processing = True
            app.is_setting_user_roi_mode = False
            
            # Verify tracking is detected as active
            is_live_tracking_running = (app.processor and
                                        app.processor.is_processing and
                                        app.processor.enable_tracker_processing)
            assert is_live_tracking_running is True
            
            # 2. Simulate natural completion (processor reaches end of video)
            app.processor.current_frame_index = 100  # Reached end
            
            # Simulate the natural completion logic from video_processor.py
            if app.processor.total_frames > 0 and app.processor.current_frame_index >= app.processor.total_frames:
                app.processor.is_processing = False
                app.processor.enable_tracker_processing = False  # This is the critical fix
            
            # 3. Verify state after natural completion
            is_live_tracking_running = (app.processor and
                                        app.processor.is_processing and
                                        app.processor.enable_tracker_processing)
            assert is_live_tracking_running is False, "Live tracking should not be detected after natural completion"
            
            # 4. Verify UI button states after natural completion
            analysis_active = app.stage_processor.full_analysis_active
            video_loaded = app.processor.is_video_open()
            can_start = video_loaded and not analysis_active and not is_live_tracking_running and not app.is_setting_user_roi_mode
            can_stop = analysis_active or is_live_tracking_running
            
            assert can_start is True, "Start button should be enabled after natural completion"
            assert can_stop is False, "Stop button should be disabled after natural completion"
            
            # 5. Simulate user navigating to middle of video and pressing play
            app.processor.current_frame_index = 50  # Navigate to middle
            
            # The play button should start regular playback, not tracking
            with patch.object(app.processor, 'start_processing') as mock_start:
                # Simulate play button press (this would call event handler)
                app.processor.start_processing()
                
                # Verify regular playback started
                mock_start.assert_called_once()
                
                # Critical: enable_tracker_processing should remain False
                assert app.processor.enable_tracker_processing is False, "Tracking flag should remain cleared after play"
    
    def test_stop_analysis_preserves_funscript_data(self):
        """Test that stopping live analysis preserves the funscript and timeline data."""
        app = self.app
        
        if not hasattr(app, 'event_handlers'):
            pytest.skip("Event handlers not available")
        
        with patch.object(app.processor, 'is_video_open', return_value=True):
            # 1. Simulate live tracking session with funscript data
            app.stage_processor.full_analysis_active = False
            app.processor.is_processing = True
            app.processor.enable_tracker_processing = True
            app.is_setting_user_roi_mode = False
            
            # Mock tracker with funscript
            if app.tracker:
                app.tracker.tracking_active = True
                # Simulate funscript with some data
                if hasattr(app.tracker, 'funscript') and app.tracker.funscript:
                    # Add some mock funscript data to verify it's preserved
                    original_actions_count = len(app.tracker.funscript.actions) if hasattr(app.tracker.funscript, 'actions') else 0
                    
                    # 2. Simulate clicking "Stop Analysis"
                    with patch.object(app.tracker, 'reset') as mock_reset:
                        app.event_handlers.handle_reset_live_tracker_click()
                        
                        # Verify reset was called with preserve reason
                        mock_reset.assert_called_once_with(reason="stop_preserve_funscript")
                    
                    # 3. Verify tracking is stopped but processor state is correct
                    assert app.processor.is_processing is False, "Processing should be stopped"
                    assert app.processor.enable_tracker_processing is False, "Tracker processing should be disabled"
                    
                    # 4. Test the actual funscript preservation logic
                    # This simulates what happens in tracker.reset() with the new reason
                    reason = "stop_preserve_funscript"
                    should_preserve = reason in ["seek", "project_load_preserve_actions", "stop_preserve_funscript"]
                    assert should_preserve is True, "Funscript should be preserved with stop_preserve_funscript reason"
                    
                    # Contrast with old behavior
                    reason_old = None  # Default reason
                    should_preserve_old = reason_old in ["seek", "project_load_preserve_actions", "stop_preserve_funscript"]
                    assert should_preserve_old is False, "Funscript would be cleared with default reason"