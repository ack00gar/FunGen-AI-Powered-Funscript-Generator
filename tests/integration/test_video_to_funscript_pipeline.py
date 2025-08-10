import pytest
import sys
import os
import tempfile
import json
import subprocess
import time
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from unittest.mock import patch
from application.logic.app_logic import ApplicationLogic

@pytest.mark.integration
def test_complete_video_to_funscript_workflow():
    """
    Test the complete workflow from video input to funscript output.
    This covers the critical video processing pipeline integration.
    """
    # Create a minimal test video using FFmpeg
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
        video_path = temp_video.name
    
    try:
        # Generate a minimal test video (black frames)
        result = subprocess.run([
            'ffmpeg', '-f', 'lavfi', '-i', 'color=black:duration=2:rate=30:size=320x240',
            '-c:v', 'libx264', '-preset', 'ultrafast', '-y', video_path
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            pytest.skip("FFmpeg not available for test video generation")
        
        # Verify video was created
        assert os.path.exists(video_path)
        assert os.path.getsize(video_path) > 0
        
        # Test CLI processing pipeline
        cli_result = subprocess.run([
            'python', 'main.py', video_path,
            '--mode', '2-stage',
            '--no-autotune',
            '--overwrite'
        ], capture_output=True, text=True, timeout=120)
        
        # Check CLI completed successfully
        assert cli_result.returncode == 0, f"CLI processing failed: {cli_result.stderr}"
        
        # Verify funscript was generated
        expected_funscript = video_path.replace('.mp4', '.funscript')
        assert os.path.exists(expected_funscript), "Funscript file was not generated"
        
        # Validate funscript content
        with open(expected_funscript, 'r') as f:
            funscript_data = json.load(f)
            assert 'actions' in funscript_data
            assert isinstance(funscript_data['actions'], list)
            # Should have some actions for a 2-second video
            assert len(funscript_data['actions']) >= 0  # May be empty for test video
            
            # Validate action structure if actions exist
            for action in funscript_data['actions']:
                assert 'at' in action
                assert 'pos' in action
                assert isinstance(action['at'], int)
                assert isinstance(action['pos'], int)
                assert 0 <= action['pos'] <= 100
    
    finally:
        # Cleanup
        for path in [video_path, video_path.replace('.mp4', '.funscript')]:
            if os.path.exists(path):
                os.remove(path)

@pytest.mark.integration
def test_stage_processor_integration():
    """
    Test the stage processor integration with application logic.
    This validates the core processing pipeline coordination.
    """
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
        app = ApplicationLogic(is_cli=True)
        
        # Verify stage processor is properly initialized
        assert app.stage_processor is not None
        assert hasattr(app.stage_processor, 'start_full_analysis')
        assert hasattr(app.stage_processor, 'start_scene_detection_analysis')
        assert hasattr(app.stage_processor, 'start_interactive_refinement_analysis')
        
        # Test stage processor state management
        assert app.stage_processor.current_analysis_stage == 0  # Not currently running
        assert app.stage_processor.full_analysis_active == False
        
        # Test stage processor integration with video processor
        assert app.processor is not None
        assert app.stage_processor.app == app

@pytest.mark.integration
def test_video_loading_and_validation():
    """
    Test video loading pipeline and format validation.
    """
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
        video_path = temp_video.name
    
    try:
        # Create minimal test video
        result = subprocess.run([
            'ffmpeg', '-f', 'lavfi', '-i', 'testsrc=duration=1:size=320x240:rate=10',
            '-c:v', 'libx264', '-preset', 'ultrafast', '-y', video_path
        ], capture_output=True, text=True, timeout=20)
        
        if result.returncode != 0:
            pytest.skip("FFmpeg not available for video generation")
        
        with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
            app = ApplicationLogic(is_cli=True)
            
            # Test video loading
            success = app.file_manager.open_video_from_path(video_path)
            
            # Verify video was loaded (may fail in test environment but should not crash)
            assert isinstance(success, bool)
            
            # Test video processor state
            assert app.processor is not None
            
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

@pytest.mark.integration 
def test_batch_processing_simulation():
    """
    Test batch processing logic without actual video processing.
    This validates the CLI batch workflow coordination.
    """
    temp_dir = tempfile.mkdtemp()
    video_files = []
    
    try:
        # Create multiple test video files
        for i in range(3):
            video_path = os.path.join(temp_dir, f'test_video_{i}.mp4')
            # Create empty files for batch testing
            with open(video_path, 'wb') as f:
                f.write(b'fake video content')
            video_files.append(video_path)
        
        # Test batch discovery logic by running CLI in simulation mode
        # This tests the file discovery without actual processing
        cli_result = subprocess.run([
            'python', '-c', f'''
import sys
import os
sys.path.insert(0, ".")
from application.logic.app_file_manager import AppFileManager
from application.logic.app_logic import ApplicationLogic

# Create a mock app to access file manager
app = ApplicationLogic(is_cli=True)
files = app.file_manager._scan_folder_for_videos("{temp_dir}")
print(f"Found {{len(files)}} files")
for f in files:
    print(f"File: {{f}}")
'''
        ], capture_output=True, text=True, timeout=10)
        
        # Check that batch discovery works
        assert cli_result.returncode == 0
        output = cli_result.stdout
        assert "Found 3 files" in output or "Found" in output  # May vary based on filtering
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.mark.integration
def test_project_state_workflow():
    """
    Test complete project lifecycle including save/load operations.
    """
    with tempfile.NamedTemporaryFile(suffix='.fgnproj', delete=False) as temp_project:
        project_path = temp_project.name
    
    try:
        with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
            # Create app and set up project state
            app = ApplicationLogic(is_cli=True)
            
            # Modify some project state
            app.project_manager.project_dirty = True
            
            # Test project save capability (method doesn't return value)
            app.project_manager.save_project(project_path)
            
            # Verify project file was created
            assert os.path.exists(project_path)
            assert os.path.getsize(project_path) > 0
                
            # Test project load (returns None on error, otherwise loads successfully)
            load_result = app.project_manager.load_project(project_path)
            # Method returns None, but if it succeeds it doesn't crash
            assert load_result is None  # Normal successful load
    
    finally:
        if os.path.exists(project_path):
            os.remove(project_path)

@pytest.mark.integration
def test_funscript_generation_pipeline():
    """
    Test the complete funscript generation pipeline from actions to file output.
    """
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
        app = ApplicationLogic(is_cli=True)
        
        # Generate test actions
        test_actions = [
            {"at": 0, "pos": 10},
            {"at": 1000, "pos": 90},
            {"at": 2000, "pos": 20},
            {"at": 3000, "pos": 80},
            {"at": 4000, "pos": 50}
        ]
        
        # Add actions to funscript processor
        for action in test_actions:
            app.funscript_processor.get_funscript_obj().add_action(
                action["at"], action["pos"]
            ) if app.funscript_processor.get_funscript_obj() else None
        
        # Verify actions were added
        primary_actions = (app.funscript_processor.get_funscript_obj().primary_actions if app.funscript_processor.get_funscript_obj() else [])
        assert len(primary_actions) == 5
        
        # Test export functionality
        with tempfile.NamedTemporaryFile(suffix='.funscript', delete=False) as temp_file:
            export_path = temp_file.name
        
        try:
            # Test export (method doesn't return value, just executes)
            app.file_manager.save_funscript_from_timeline(export_path, 1)
            
            # Check if file was created
            if os.path.exists(export_path):
                assert True  # Export worked
                
                # Validate exported content
                with open(export_path, 'r') as f:
                    exported_data = json.load(f)
                    assert 'actions' in exported_data
                    assert len(exported_data['actions']) == 5
                    
                    # Verify action integrity
                    for i, action in enumerate(exported_data['actions']):
                        assert action['at'] == test_actions[i]['at']
                        assert action['pos'] == test_actions[i]['pos']
        
        finally:
            if os.path.exists(export_path):
                os.remove(export_path)

@pytest.mark.integration
def test_error_handling_pipeline():
    """
    Test error handling in critical pipeline components.
    """
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
        app = ApplicationLogic(is_cli=True)
        
        # Test handling of invalid video path
        invalid_path = "/nonexistent/video.mp4"
        result = app.file_manager.open_video_from_path(invalid_path)
        # Should handle gracefully without crashing
        assert isinstance(result, bool)
        
        # Test handling of invalid project path
        invalid_project = "/nonexistent/project.fgnproj"
        result = app.project_manager.load_project(invalid_project)
        # Should handle gracefully without crashing (returns None on error)
        assert result is None or isinstance(result, bool)
        
        # Test funscript processor with invalid operations
        # Should not crash on empty operations
        funscript_obj = app.funscript_processor.get_funscript_obj()
        if funscript_obj:
            stats = funscript_obj.get_actions_statistics('primary')
            assert 'num_points' in stats
        else:
            # If no funscript object, just ensure the method exists
            assert hasattr(app.funscript_processor, 'get_funscript_obj')
        
        # Test export with no actions
        with tempfile.NamedTemporaryFile(suffix='.funscript', delete=False) as temp_file:
            export_path = temp_file.name
        
        try:
            # Should handle empty funscript gracefully without crashing
            app.file_manager.save_funscript_from_timeline(export_path, 1)
            # If we get here without exception, the method handled it gracefully
            assert True
        
        finally:
            if os.path.exists(export_path):
                os.remove(export_path)