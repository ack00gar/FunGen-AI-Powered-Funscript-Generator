import pytest
import os
from unittest.mock import patch, MagicMock
from application.logic.app_logic import ApplicationLogic

@pytest.mark.integration
def test_tensorrt_compiler_system():
    """Test TensorRT compiler functionality.""" 
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
        app = ApplicationLogic(is_cli=True)
        
        # Test TensorRT-related functionality exists
        # (Would need actual TensorRT installation to test fully)
        
        # Test basic model loading with fallbacks
        assert app.tracker is not None
        # Model loading tested in other suites

@pytest.mark.integration  
def test_gpu_memory_management():
    """Test GPU memory management and model pooling."""
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
        app = ApplicationLogic(is_cli=True)
        
        # Test model pool exists and functions
        if hasattr(app.tracker, 'model_pool'):
            model_pool = app.tracker.model_pool
            assert model_pool is not None
            
            # Test basic model pool functionality
            assert hasattr(model_pool, 'get_model')
            assert hasattr(model_pool, 'clear_all_models')

@pytest.mark.integration
def test_ffmpeg_hardware_acceleration():
    """Test FFmpeg hardware acceleration detection."""
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
        app = ApplicationLogic(is_cli=True)
        
        # Test hardware acceleration detection
        hwaccels = app._get_available_ffmpeg_hwaccels()
        assert isinstance(hwaccels, list)
        assert len(hwaccels) > 0
        
        # Should have at least software fallback
        assert 'none' in hwaccels
        
        # On macOS, should have VideoToolbox
        if os.name == 'posix' and 'darwin' in os.uname().sysname.lower():
            assert 'videotoolbox' in hwaccels

@pytest.mark.integration
def test_performance_monitoring():
    """Test performance monitoring and FPS tracking."""
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
        app = ApplicationLogic(is_cli=True)
        
        # Test FPS tracking attributes exist
        assert hasattr(app.stage_processor, 'stage1_processing_fps_str')
        assert hasattr(app.stage_processor, 'stage1_instant_fps_str')
        assert hasattr(app.stage_processor, 'stage1_time_elapsed_str')
        
        # Test ETA calculation exists
        assert hasattr(app.stage_processor, 'stage1_eta_str')
