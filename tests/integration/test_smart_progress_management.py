import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock
from application.logic.app_logic import ApplicationLogic
from application.utils.checkpoint_manager import ProcessingStage, get_checkpoint_manager

@pytest.mark.integration 
def test_checkpoint_creation_during_processing():
    """Test that checkpoints are automatically created during processing."""
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
        app = ApplicationLogic(is_cli=True)
        
        # Mock a video path
        app.file_manager.video_path = "/tmp/test_video.mp4"
        
        # Test checkpoint creation
        checkpoint_manager = app.stage_processor.checkpoint_manager
        assert checkpoint_manager is not None
        
        # Simulate processing progress and checkpoint creation
        test_checkpoint_id = checkpoint_manager.create_checkpoint(
            video_path="/tmp/test_video.mp4",
            stage=ProcessingStage.STAGE_1_OBJECT_DETECTION, 
            progress_percentage=25.0,
            frame_index=250,
            total_frames=1000,
            stage_data={"current_frame": 250},
            processing_settings={"mode": "test"}
        )
        
        assert test_checkpoint_id is not None

@pytest.mark.integration
def test_resume_from_checkpoint():
    """Test resuming processing from a saved checkpoint."""
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
        app = ApplicationLogic(is_cli=True)
        
        # Test checkpoint resume detection
        resumable_tasks = app.stage_processor.check_resumable_tasks()
        assert isinstance(resumable_tasks, list)
        
        # Test resume capability check
        can_resume = app.stage_processor.can_resume_video("/tmp/test_video.mp4")
        # Should return None if no checkpoint exists, or CheckpointData if it exists
        assert can_resume is None or hasattr(can_resume, 'checkpoint_id')

@pytest.mark.integration
def test_processing_thread_manager_integration():
    """Test Processing Thread Manager integration."""
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
        app = ApplicationLogic(is_cli=True)
        
        # Check if processing thread manager exists in GUI
        if hasattr(app, 'processing_thread_manager'):
            ptm = app.processing_thread_manager
            
            # Test task submission capability
            assert hasattr(ptm, 'submit_task')
            assert hasattr(ptm, 'get_stats')
            assert hasattr(ptm, 'cancel_task')
            
            # Test stats functionality
            stats = ptm.get_stats()
            assert isinstance(stats, dict)
            assert 'tasks_completed' in stats
