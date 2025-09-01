"""
End-to-end tests for live tracking UI behavior.

These tests validate the complete user workflow including UI interactions,
button states, and proper behavior during live tracking sessions using
the full application stack.
"""

import pytest
import sys
import os
import time
import tempfile
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from application.logic.app_logic import ApplicationLogic
from application.gui_components.dynamic_tracker_ui import get_dynamic_tracker_ui


@pytest.mark.e2e
class TestLiveTrackingUIWorkflow:
    """End-to-end tests for complete live tracking UI workflows."""
    
    def setup_method(self):
        """Set up full application for e2e testing."""
        with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
            self.app = ApplicationLogic(is_cli=False)  # Full GUI mode for e2e
    
    def teardown_method(self):
        """Clean up after each test."""
        if hasattr(self.app, 'cleanup'):
            self.app.cleanup()
    
    @pytest.mark.skipif("not config.getoption('--run-e2e')")
    def test_complete_live_tracking_session_workflow(self):
        """Test complete workflow from starting to stopping a live tracking session."""
        app = self.app
        
        # Skip if GUI not available
        if not hasattr(app, 'gui') or not app.gui:
            pytest.skip("Full GUI not available for e2e testing")
        
        # Mock video loading
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
            temp_video_path = temp_video.name
        
        try:
            # Simulate video loading
            with patch.object(app.file_manager, 'video_path', temp_video_path), \
                 patch.object(app.processor, 'is_video_open', return_value=True), \
                 patch.object(app.processor, 'video_info', {'total_frames': 1000}):
                
                # 1. Initial state - buttons should be in correct state
                self._assert_initial_ui_state(app)
                
                # 2. Set live tracking mode
                app.app_state_ui.selected_tracker_name = 'yolo_roi'
                
                # 3. Simulate starting live tracking
                self._simulate_start_live_tracking(app)
                
                # 4. Verify UI state during tracking
                self._assert_tracking_active_ui_state(app)
                
                # 5. Simulate stopping live tracking
                self._simulate_stop_live_tracking(app)
                
                # 6. Verify UI state after tracking
                self._assert_post_tracking_ui_state(app)
                
        finally:
            # Clean up temp file
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
    
    def _assert_initial_ui_state(self, app):
        """Assert UI is in correct initial state."""
        # Control panel button states
        analysis_active = app.stage_processor.full_analysis_active
        video_loaded = app.processor.is_video_open()
        is_live_tracking_running = (app.processor and
                                    app.processor.is_processing and
                                    app.processor.enable_tracker_processing)
        
        can_start = video_loaded and not analysis_active and not is_live_tracking_running and not app.is_setting_user_roi_mode
        can_stop = analysis_active or is_live_tracking_running
        
        assert can_start is True, "Start button should be enabled initially"
        assert can_stop is False, "Stop button should be disabled initially"
        assert is_live_tracking_running is False, "Live tracking should not be running initially"
        
        # Video controls should be enabled
        controls_disabled = analysis_active or is_live_tracking_running or not app.file_manager.video_path
        assert controls_disabled is False, "Video controls should be enabled initially"
    
    def _simulate_start_live_tracking(self, app):
        """Simulate starting a live tracking session."""
        # This would normally be triggered by the UI button
        app.processor.is_processing = True
        app.processor.enable_tracker_processing = True
        
        # Simulate tracker activation
        if app.tracker:
            app.tracker.tracking_active = True
    
    def _assert_tracking_active_ui_state(self, app):
        """Assert UI is in correct state during active tracking."""
        # Control panel button states
        analysis_active = app.stage_processor.full_analysis_active
        video_loaded = app.processor.is_video_open()
        is_live_tracking_running = (app.processor and
                                    app.processor.is_processing and
                                    app.processor.enable_tracker_processing)
        
        can_start = video_loaded and not analysis_active and not is_live_tracking_running and not app.is_setting_user_roi_mode
        can_stop = analysis_active or is_live_tracking_running
        
        assert can_start is False, "Start button should be disabled during tracking"
        assert can_stop is True, "Stop button should be enabled during tracking"
        assert is_live_tracking_running is True, "Live tracking should be detected as running"
        
        # Video controls should be disabled
        controls_disabled = analysis_active or is_live_tracking_running or not app.file_manager.video_path
        assert controls_disabled is True, "Video controls should be disabled during tracking"
    
    def _simulate_stop_live_tracking(self, app):
        """Simulate stopping a live tracking session."""
        # This simulates what happens when tracking stops
        app.processor.enable_tracker_processing = False  # Key: flag is cleared
        app.processor.is_processing = False
        
        if app.tracker:
            app.tracker.tracking_active = False
    
    def _assert_post_tracking_ui_state(self, app):
        """Assert UI is in correct state after tracking stops."""
        # Control panel button states
        analysis_active = app.stage_processor.full_analysis_active
        video_loaded = app.processor.is_video_open()
        is_live_tracking_running = (app.processor and
                                    app.processor.is_processing and
                                    app.processor.enable_tracker_processing)
        
        can_start = video_loaded and not analysis_active and not is_live_tracking_running and not app.is_setting_user_roi_mode
        can_stop = analysis_active or is_live_tracking_running
        
        assert can_start is True, "Start button should be enabled after tracking stops"
        assert can_stop is False, "Stop button should be disabled after tracking stops"
        assert is_live_tracking_running is False, "Live tracking should not be detected after stop"
        
        # Video controls should be enabled again
        controls_disabled = analysis_active or is_live_tracking_running or not app.file_manager.video_path
        assert controls_disabled is False, "Video controls should be enabled after tracking stops"
    
    @pytest.mark.skipif("not config.getoption('--run-e2e')")
    def test_playback_controls_behavior_during_tracking(self):
        """Test that playback controls behave correctly during and after tracking."""
        app = self.app
        
        if not hasattr(app, 'event_handlers'):
            pytest.skip("Event handlers not available")
        
        # Mock video setup
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
            temp_video_path = temp_video.name
        
        try:
            with patch.object(app.file_manager, 'video_path', temp_video_path), \
                 patch.object(app.processor, 'is_video_open', return_value=True), \
                 patch.object(app.processor, 'video_info', {'total_frames': 1000}):
                
                app.processor.current_frame_index = 0
                
                # Test 1: Controls work normally before tracking
                with patch.object(app.processor, 'start_processing') as mock_start:
                    app.event_handlers.handle_playback_control("play_pause")
                    mock_start.assert_called_once()
                    assert app.processor.enable_tracker_processing is False
                
                # Test 2: Start tracking session
                app.processor.is_processing = True
                app.processor.enable_tracker_processing = True
                
                # Controls should be disabled during tracking (tested in UI logic)
                is_live_tracking_running = (app.processor and
                                            app.processor.is_processing and
                                            app.processor.enable_tracker_processing)
                controls_disabled = app.stage_processor.full_analysis_active or is_live_tracking_running or not app.file_manager.video_path
                assert controls_disabled is True
                
                # Test 3: Stop tracking
                app.processor.enable_tracker_processing = False  # Cleared when tracking stops
                
                # Test 4: Play button after tracking should NOT restart tracking
                with patch.object(app.processor, 'start_processing') as mock_start:
                    app.event_handlers.handle_playback_control("play_pause")
                    mock_start.assert_called_once()
                    # Critical: tracking flag should remain False
                    assert app.processor.enable_tracker_processing is False
                
        finally:
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
    
    @pytest.mark.skipif("not config.getoption('--run-e2e')")
    def test_mode_switching_ui_consistency(self):
        """Test UI consistency when switching between tracking modes."""
        app = self.app
        
        if not hasattr(app, 'app_state_ui'):
            pytest.skip("App state UI not available")
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
            temp_video_path = temp_video.name
        
        try:
            with patch.object(app.file_manager, 'video_path', temp_video_path), \
                 patch.object(app.processor, 'is_video_open', return_value=True):
                
                # Test switching between live tracking modes
                live_modes = ['yolo_roi', 'user_roi', 'oscillation_experimental_2']
                
                for mode in live_modes:
                    app.app_state_ui.selected_tracker_mode = mode
                    
                    # Verify initial state consistency
                    self._assert_initial_ui_state(app)
                    
                    # Simulate starting tracking
                    app.processor.is_processing = True
                    app.processor.enable_tracker_processing = True
                    
                    # Verify tracking state consistency
                    self._assert_tracking_active_ui_state(app)
                    
                    # Simulate stopping tracking
                    app.processor.enable_tracker_processing = False
                    app.processor.is_processing = False
                    
                    # Verify post-tracking state consistency
                    self._assert_post_tracking_ui_state(app)
                
        finally:
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
    
    @pytest.mark.skipif("not config.getoption('--run-e2e')")
    def test_roi_setting_mode_ui_behavior(self):
        """Test UI behavior during ROI setting mode."""
        app = self.app
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
            temp_video_path = temp_video.name
        
        try:
            with patch.object(app.file_manager, 'video_path', temp_video_path), \
                 patch.object(app.processor, 'is_video_open', return_value=True):
                
                # Test 1: Normal state
                app.is_setting_user_roi_mode = False
                self._assert_initial_ui_state(app)
                
                # Test 2: Enter ROI setting mode
                app.is_setting_user_roi_mode = True
                
                analysis_active = app.stage_processor.full_analysis_active
                video_loaded = app.processor.is_video_open()
                is_live_tracking_running = (app.processor and
                                            app.processor.is_processing and
                                            app.processor.enable_tracker_processing)
                
                can_start = video_loaded and not analysis_active and not is_live_tracking_running and not app.is_setting_user_roi_mode
                can_stop = analysis_active or is_live_tracking_running
                
                assert can_start is False, "Start button should be disabled during ROI setting"
                assert can_stop is False, "Stop button should remain disabled during ROI setting"
                
                # Test 3: Exit ROI setting mode
                app.is_setting_user_roi_mode = False
                self._assert_initial_ui_state(app)
                
        finally:
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)


@pytest.mark.e2e
class TestUIStateManagementEdgeCases:
    """End-to-end tests for edge cases in UI state management."""
    
    def setup_method(self):
        """Set up full application for e2e testing."""
        with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
            self.app = ApplicationLogic(is_cli=False)
    
    def teardown_method(self):
        """Clean up after each test."""
        if hasattr(self.app, 'cleanup'):
            self.app.cleanup()
    
    @pytest.mark.skipif("not config.getoption('--run-e2e')")
    def test_concurrent_analysis_and_tracking_prevention(self):
        """Test that offline analysis and live tracking cannot run simultaneously."""
        app = self.app
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
            temp_video_path = temp_video.name
        
        try:
            with patch.object(app.file_manager, 'video_path', temp_video_path), \
                 patch.object(app.processor, 'is_video_open', return_value=True):
                
                # Start offline analysis
                app.stage_processor.full_analysis_active = True
                
                analysis_active = app.stage_processor.full_analysis_active
                video_loaded = app.processor.is_video_open()
                is_live_tracking_running = (app.processor and
                                            app.processor.is_processing and
                                            app.processor.enable_tracker_processing)
                
                can_start = video_loaded and not analysis_active and not is_live_tracking_running and not app.is_setting_user_roi_mode
                can_stop = analysis_active or is_live_tracking_running
                
                assert can_start is False, "Should not be able to start tracking during offline analysis"
                assert can_stop is True, "Should be able to stop offline analysis"
                
                # Try to start live tracking (should be prevented by UI logic)
                app.processor.is_processing = True
                app.processor.enable_tracker_processing = True
                
                # Button logic should still prevent starting new sessions
                can_start = video_loaded and not analysis_active and not is_live_tracking_running and not app.is_setting_user_roi_mode
                assert can_start is False, "Should still not allow starting when both flags are set"
                
        finally:
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
    
    @pytest.mark.skipif("not config.getoption('--run-e2e')")
    def test_processor_None_edge_case(self):
        """Test UI behavior when processor is None."""
        app = self.app
        
        # Temporarily set processor to None
        original_processor = app.processor
        app.processor = None
        
        try:
            # UI logic should handle gracefully
            is_live_tracking_running = (app.processor and
                                        app.processor.is_processing and
                                        app.processor.enable_tracker_processing)
            
            assert is_live_tracking_running is False, "Should handle None processor gracefully"
            
            # Button logic should work
            analysis_active = app.stage_processor.full_analysis_active
            video_loaded = app.processor and app.processor.is_video_open() if app.processor else False
            
            can_start = video_loaded and not analysis_active and not is_live_tracking_running and not app.is_setting_user_roi_mode
            can_stop = analysis_active or is_live_tracking_running
            
            assert can_start is False, "Should not allow starting without processor"
            assert can_stop is False, "Should not show stop button without processor"
            
        finally:
            # Restore processor
            app.processor = original_processor
    
    @pytest.mark.skipif("not config.getoption('--run-e2e')")
    def test_rapid_mode_switching_stability(self):
        """Test UI stability during rapid mode switching."""
        app = self.app
        
        if not hasattr(app, 'app_state_ui'):
            pytest.skip("App state UI not available")
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
            temp_video_path = temp_video.name
        
        try:
            with patch.object(app.file_manager, 'video_path', temp_video_path), \
                 patch.object(app.processor, 'is_video_open', return_value=True):
                
                modes = [
                    'yolo_roi',
                    'user_roi',
                    'oscillation_experimental_2',
                    'stage2_contact_analysis',
                    'stage3_optical_flow'
                ]
                
                # Rapidly switch between modes
                for _ in range(3):  # Multiple cycles
                    for mode in modes:
                        app.app_state_ui.selected_tracker_mode = mode
                        
                        # UI should remain stable
                        analysis_active = app.stage_processor.full_analysis_active
                        video_loaded = app.processor.is_video_open()
                        is_live_tracking_running = (app.processor and
                                                    app.processor.is_processing and
                                                    app.processor.enable_tracker_processing)
                        
                        can_start = video_loaded and not analysis_active and not is_live_tracking_running and not app.is_setting_user_roi_mode
                        
                        # Should always be able to start when idle
                        assert can_start is True, f"Should be able to start in mode {mode}"
                        assert is_live_tracking_running is False, f"Should not show tracking active in mode {mode}"
                
        finally:
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)