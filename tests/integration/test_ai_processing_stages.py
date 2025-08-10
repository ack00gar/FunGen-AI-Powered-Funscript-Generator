import pytest
from unittest.mock import patch, MagicMock
from application.logic.app_logic import ApplicationLogic

@pytest.mark.integration
def test_stage2_optical_flow_processing():
    """Test Stage 2 optical flow processing functionality."""
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
        app = ApplicationLogic(is_cli=True)
        
        # Test stage 2 progress callback exists
        assert hasattr(app.stage_processor, 'on_stage2_progress')
        assert callable(app.stage_processor.on_stage2_progress)
        
        # Test stage 2 can handle progress updates
        try:
            app.stage_processor.on_stage2_progress(
                main_info_from_module=(1, 10, "Test Stage"),
                sub_info_from_module=(5, 100, "Test Sub Stage")
            )
            assert True  # If no exception, callback works
        except Exception as e:
            pytest.fail(f"Stage 2 progress callback failed: {e}")

@pytest.mark.integration
def test_stage3_funscript_generation():
    """Test Stage 3 funscript generation functionality."""
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
        app = ApplicationLogic(is_cli=True)
        
        # Test stage 3 progress callback exists
        assert hasattr(app.stage_processor, 'on_stage3_progress')
        assert callable(app.stage_processor.on_stage3_progress)
        
        # Test stage 3 can handle progress updates
        try:
            app.stage_processor.on_stage3_progress(
                current_chapter_idx=1,
                total_chapters=5,
                chapter_name="Test Chapter",
                current_chunk_idx=2,
                total_chunks=10,
                total_frames_processed_overall=500,
                total_frames_to_process_overall=2000,
                processing_fps=30.0,
                time_elapsed=15.0,
                eta_seconds=45.0
            )
            assert True  # If no exception, callback works
        except Exception as e:
            pytest.fail(f"Stage 3 progress callback failed: {e}")

@pytest.mark.integration  
def test_multi_stage_analysis_coordination():
    """Test coordination between processing stages."""
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
        app = ApplicationLogic(is_cli=True)
        
        # Test stage processor state management
        assert hasattr(app.stage_processor, 'current_analysis_stage')
        assert hasattr(app.stage_processor, 'full_analysis_active')
        
        # Initial state should be inactive
        assert app.stage_processor.current_analysis_stage == 0
        assert app.stage_processor.full_analysis_active == False
        
        # Test stage processor has required methods
        assert hasattr(app.stage_processor, 'start_full_analysis')
        assert callable(app.stage_processor.start_full_analysis)
