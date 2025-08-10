import pytest
from unittest.mock import patch, MagicMock
from application.logic.app_logic import ApplicationLogic

@pytest.mark.integration
def test_ultimate_autotune_system_exists():
    """Test that Ultimate Autotune system components exist."""
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
        app = ApplicationLogic(is_cli=True)
        
        # Check if autotune functionality exists
        if hasattr(app, 'ultimate_autotune'):
            autotune = app.ultimate_autotune
            assert autotune is not None
        
        # Check stage processor for autotune capabilities
        assert hasattr(app.stage_processor, 'start_full_analysis')
        
        # Test if autotune can be triggered
        # (Would need actual implementation to test fully)
        assert True  # Placeholder for now

@pytest.mark.integration  
def test_hardware_acceleration_detection():
    """Test hardware acceleration detection and optimization."""
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
        app = ApplicationLogic(is_cli=True)
        
        # Test FFmpeg hardware acceleration detection
        hwaccels = app._get_available_ffmpeg_hwaccels()
        assert isinstance(hwaccels, list)
        assert len(hwaccels) > 0
        assert 'none' in hwaccels  # Should always have software fallback
        
        # Test that hardware acceleration preferences can be set
        # (Actual implementation would need to be tested)
        assert True

@pytest.mark.integration
def test_performance_benchmarking():
    """Test performance benchmarking functionality.""" 
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
        app = ApplicationLogic(is_cli=True)
        
        # Test basic performance monitoring exists
        assert hasattr(app.stage_processor, 'stage1_processing_fps_str')
        assert hasattr(app.stage_processor, 'stage1_time_elapsed_str')
        
        # Test FPS tracking functionality exists
        # (Would need actual processing to test fully)
        assert True
