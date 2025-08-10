import pytest
from unittest.mock import patch, MagicMock
from application.logic.app_logic import ApplicationLogic

@pytest.mark.integration
def test_oscillation_detector_mode():
    """Test Live - Oscillation Detector mode."""
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
        app = ApplicationLogic(is_cli=True)
        
        # Test tracker mode configuration
        if hasattr(app, 'app_state_ui'):
            # Test oscillation detector mode exists
            from config.app_state import TrackerMode
            assert hasattr(TrackerMode, 'OSCILLATION_DETECTOR')
        else:
            # Basic functionality test
            assert app.tracker is not None
            assert hasattr(app.tracker, 'start_tracking')

@pytest.mark.integration
def test_live_yolo_roi_mode():
    """Test Live - YOLO auto ROI mode."""
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
        app = ApplicationLogic(is_cli=True)
        
        # Test YOLO ROI functionality exists
        if hasattr(app, 'app_state_ui'):
            from config.app_state import TrackerMode
            assert hasattr(TrackerMode, 'LIVE_YOLO_ROI')
        
        # Test ROI-related functionality
        assert app.tracker is not None
        # ROI functionality would need actual implementation to test

@pytest.mark.integration
def test_live_user_roi_mode():
    """Test Live - User manual ROI mode."""
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
        app = ApplicationLogic(is_cli=True)
        
        # Test manual ROI functionality
        if hasattr(app, 'app_state_ui'):
            from config.app_state import TrackerMode
            assert hasattr(TrackerMode, 'LIVE_USER_ROI')
        
        # Test ROI setting functionality
        if hasattr(app, 'is_setting_user_roi_mode'):
            assert isinstance(app.is_setting_user_roi_mode, bool)

@pytest.mark.integration
def test_tracker_initialization():
    """Test tracker component initialization and capabilities."""
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
        app = ApplicationLogic(is_cli=True)
        
        # Test tracker exists and is properly initialized
        assert app.tracker is not None
        assert hasattr(app.tracker, 'start_tracking')
        assert hasattr(app.tracker, 'confidence_threshold')
        
        # Test video processor integration
        assert app.processor is not None
        assert hasattr(app.processor, 'tracker')
        assert app.processor.tracker == app.tracker
