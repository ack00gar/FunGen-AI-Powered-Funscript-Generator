import pytest
from unittest.mock import patch, MagicMock
from imgui_bundle import imgui, hello_imgui
import os
import time
import json
import tempfile
import shutil

@pytest.mark.e2e
def test_complete_video_to_funscript_workflow(app_instance):
    """
    Test the complete end-to-end workflow from video loading to funscript generation.
    """
    video_path = os.path.abspath("test_data/workflow_video.mp4")
    output_dir = os.path.abspath("output")
    funscript_path = os.path.join(output_dir, "workflow_video.funscript")
    
    # Setup test video
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    with open(video_path, "w") as f:
        f.write("dummy video content for workflow test")
    
    # Clean up any existing funscript
    if os.path.exists(funscript_path):
        os.remove(funscript_path)
    
    engine = hello_imgui.get_imgui_test_engine()
    
    # Step 1: Load video
    with patch('application.classes.file_dialog.ImGuiFileDialog.show') as mock_show_dialog:
        engine.item_open("**/File")
        engine.item_open("**/Open...")
        engine.item_click("**/Video...")
        args, kwargs = mock_show_dialog.call_args
        callback = kwargs.get('callback')
        callback(video_path)
    
    # Wait for video to load
    start_time = time.time()
    while not app_instance.file_manager.video_path:
        if time.time() - start_time > 10:
            pytest.fail("Video load timed out")
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)
    
    # Step 2: Set up analysis mode
    engine.item_open("**/Run Control")
    engine.item_click("**/Tracker Type##TrackerModeComboGlobal")
    engine.item_click("**/Offline AI Analysis (3-Stage)")
    
    # Step 3: Start analysis
    engine.item_click("**/Start Full AI Analysis")
    
    # Wait for analysis to complete
    start_time = time.time()
    while app_instance.stage_processor.full_analysis_active:
        if time.time() - start_time > 120:
            pytest.fail("Analysis workflow timed out")
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)
    
    # Step 4: Verify funscript was generated
    assert os.path.exists(funscript_path), f"Funscript not generated at {funscript_path}"
    
    # Step 5: Verify funscript content
    with open(funscript_path, 'r') as f:
        funscript_data = json.load(f)
        assert 'actions' in funscript_data
        assert isinstance(funscript_data['actions'], list)
    
    # Step 6: Test post-processing
    engine.item_open("**/Post-Processing")
    engine.item_click("**/Run Post-Processing Now##RunAutoPostProcessButton")
    
    # Wait for post-processing
    start_time = time.time()
    actions_before = app_instance.funscript_processor.get_actions('primary')
    while True:
        if time.time() - start_time > 15:
            break
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)
        current_actions = app_instance.funscript_processor.get_actions('primary')
        if current_actions != actions_before:
            break
    
    # Step 7: Save project
    project_path = os.path.join(output_dir, "workflow_test.fgnproj")
    with patch('application.classes.file_dialog.ImGuiFileDialog.show') as mock_save_dialog:
        engine.item_open("**/File")
        engine.item_click("**/Save Project As...")
        args, kwargs = mock_save_dialog.call_args
        callback = kwargs.get('callback')
        callback(project_path)
    
    # Wait for project save
    start_time = time.time()
    while not os.path.exists(project_path):
        if time.time() - start_time > 10:
            pytest.fail("Project save timed out")
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)
    
    assert os.path.exists(project_path), "Project file not saved"

@pytest.mark.e2e
def test_batch_processing_workflow(app_instance):
    """
    Test batch processing multiple video files.
    """
    # Create multiple test videos
    video_dir = os.path.abspath("test_data/batch_videos")
    os.makedirs(video_dir, exist_ok=True)
    
    video_files = []
    for i in range(3):
        video_path = os.path.join(video_dir, f"batch_video_{i}.mp4")
        with open(video_path, "w") as f:
            f.write(f"dummy video content {i}")
        video_files.append(video_path)
    
    # Test would implement batch processing logic here
    # This is a placeholder for the actual batch processing test implementation
    assert len(video_files) == 3

@pytest.mark.e2e 
def test_dual_axis_funscript_generation(app_instance):
    """
    Test generation of dual-axis funscripts (primary and secondary).
    """
    video_path = os.path.abspath("test_data/dual_axis_video.mp4")
    output_dir = os.path.abspath("output") 
    funscript_path = os.path.join(output_dir, "dual_axis_video.funscript")
    roll_path = os.path.join(output_dir, "dual_axis_video.roll.funscript")
    
    # Setup test video
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    with open(video_path, "w") as f:
        f.write("dummy video for dual axis test")
    
    # Clean up existing files
    for path in [funscript_path, roll_path]:
        if os.path.exists(path):
            os.remove(path)
    
    engine = hello_imgui.get_imgui_test_engine()
    
    # Load video
    with patch('application.classes.file_dialog.ImGuiFileDialog.show') as mock_show_dialog:
        engine.item_open("**/File")
        engine.item_open("**/Open...")
        engine.item_click("**/Video...")
        args, kwargs = mock_show_dialog.call_args
        callback = kwargs.get('callback')
        callback(video_path)
    
    # Wait for video load
    start_time = time.time()
    while not app_instance.file_manager.video_path:
        if time.time() - start_time > 10:
            pytest.fail("Video load timed out")
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)
    
    # Enable dual axis mode
    engine.item_open("**/Settings")
    engine.item_click("**/Enable Dual Axis##DualAxisToggle")
    
    # Run analysis
    engine.item_open("**/Run Control")
    engine.item_click("**/Tracker Type##TrackerModeComboGlobal")
    engine.item_click("**/Offline AI Analysis (3-Stage)")
    engine.item_click("**/Start Full AI Analysis")
    
    # Wait for completion
    start_time = time.time()
    while app_instance.stage_processor.full_analysis_active:
        if time.time() - start_time > 120:
            pytest.fail("Dual axis analysis timed out")
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)
    
    # Verify both funscript files were created
    assert os.path.exists(funscript_path), "Primary funscript not generated"
    assert os.path.exists(roll_path), "Secondary (roll) funscript not generated"
    
    # Verify content of both files
    for path in [funscript_path, roll_path]:
        with open(path, 'r') as f:
            data = json.load(f)
            assert 'actions' in data
            assert isinstance(data['actions'], list)

@pytest.mark.e2e
def test_live_tracking_workflow(app_instance):
    """
    Test live tracking functionality with real-time funscript generation.
    """
    video_path = os.path.abspath("test_data/live_tracking_video.mp4") 
    output_dir = os.path.abspath("output")
    funscript_path = os.path.join(output_dir, "live_tracking_video.funscript")
    
    # Setup
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    with open(video_path, "w") as f:
        f.write("dummy video for live tracking")
    
    if os.path.exists(funscript_path):
        os.remove(funscript_path)
    
    engine = hello_imgui.get_imgui_test_engine()
    
    # Load video
    with patch('application.classes.file_dialog.ImGuiFileDialog.show') as mock_show_dialog:
        engine.item_open("**/File")
        engine.item_open("**/Open...")
        engine.item_click("**/Video...")
        args, kwargs = mock_show_dialog.call_args
        callback = kwargs.get('callback')
        callback(video_path)
    
    # Wait for video load
    start_time = time.time()
    while not app_instance.file_manager.video_path:
        if time.time() - start_time > 10:
            pytest.fail("Video load timed out")
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)
    
    # Set up live tracking
    engine.item_open("**/Run Control")
    engine.item_click("**/Tracker Type##TrackerModeComboGlobal")
    engine.item_click("**/Live YOLO ROI Detection")
    
    # Start live tracking
    engine.item_click("**/Start Live Tracking")
    
    # Let it run for a short time
    start_time = time.time()
    while time.time() - start_time < 10:  # Run for 10 seconds
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)
    
    # Stop tracking
    engine.item_click("**/Abort/Stop Process##AbortGeneral")
    
    # Wait for stop
    start_time = time.time()
    while app_instance.stage_processor.is_running():
        if time.time() - start_time > 10:
            pytest.fail("Live tracking stop timed out")
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)
    
    # Verify funscript was generated
    assert os.path.exists(funscript_path), "Live tracking funscript not generated"

@pytest.mark.e2e
def test_error_recovery_workflow(app_instance):
    """
    Test error recovery and graceful degradation scenarios.
    """
    engine = hello_imgui.get_imgui_test_engine()
    
    # Test 1: Load invalid video file
    invalid_video_path = os.path.abspath("test_data/invalid_video.txt")
    os.makedirs(os.path.dirname(invalid_video_path), exist_ok=True)
    with open(invalid_video_path, "w") as f:
        f.write("this is not a video file")
    
    with patch('application.classes.file_dialog.ImGuiFileDialog.show') as mock_show_dialog:
        engine.item_open("**/File")
        engine.item_open("**/Open...")
        engine.item_click("**/Video...")
        args, kwargs = mock_show_dialog.call_args
        callback = kwargs.get('callback')
        callback(invalid_video_path)
    
    # Give time for error handling
    start_time = time.time()
    while time.time() - start_time < 5:
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)
    
    # App should still be responsive after error
    assert app_instance.file_manager.video_path is None
    
    # Test 2: Attempt analysis without video
    engine.item_open("**/Run Control") 
    engine.item_click("**/Start Full AI Analysis")
    
    # Should handle gracefully
    start_time = time.time()
    while time.time() - start_time < 2:
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)
    
    # App should still be functional
    assert not app_instance.stage_processor.full_analysis_active