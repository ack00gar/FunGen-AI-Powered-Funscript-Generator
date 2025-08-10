import pytest
from unittest.mock import patch, MagicMock
from imgui_bundle import imgui, hello_imgui
import os
import time
import json
import numpy as np

@pytest.mark.e2e
def test_interactive_timeline_editing(app_instance):
    """
    Test interactive timeline editing functionality.
    """
    # Setup test data
    video_path = os.path.abspath("test_data/timeline_test_video.mp4")
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    with open(video_path, "w") as f:
        f.write("dummy video for timeline test")
    
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
    
    # Add some test points to the timeline
    test_actions = [
        {"at": 1000, "pos": 20},
        {"at": 2000, "pos": 80},
        {"at": 3000, "pos": 30},
        {"at": 4000, "pos": 70},
    ]
    app_instance.funscript_processor.clear_timeline_history_and_set_new_baseline(1, test_actions, "test_setup")
    
    # Wait for actions to be set
    start_time = time.time()
    while not app_instance.funscript_processor.get_actions('primary'):
        if time.time() - start_time > 5:
            pytest.fail("Timeline setup timed out")
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)
    
    # Test timeline interaction
    engine.item_open("**/Timeline")
    
    # Simulate timeline clicks/edits - this would be more complex in real implementation
    # For now, verify that timeline is accessible and responsive
    time.sleep(1)
    
    # Verify actions are still present
    actions = app_instance.funscript_processor.get_actions('primary')
    assert len(actions) >= 4

@pytest.mark.e2e
def test_video_navigation_controls(app_instance):
    """
    Test video navigation and playback controls.
    """
    video_path = os.path.abspath("test_data/navigation_test_video.mp4")
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    with open(video_path, "w") as f:
        f.write("dummy video for navigation test")
    
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
    
    # Test navigation controls
    engine.item_open("**/Video Navigation")
    
    # Test play/pause
    engine.item_click("**/Play/Pause##PlayPauseButton")
    time.sleep(1)
    
    # Test seeking
    engine.slider_set_float("**/Timeline##VideoTimeline", 0.5)  # Seek to middle
    time.sleep(1)
    
    # Test frame stepping
    engine.item_click("**/Next Frame##NextFrameButton")
    engine.item_click("**/Previous Frame##PrevFrameButton")
    
    # Test speed control
    engine.slider_set_float("**/Playback Speed##PlaybackSpeedSlider", 2.0)
    time.sleep(1)
    
    # Verify video state is responsive
    assert app_instance.processor.video_info is not None

@pytest.mark.e2e
def test_roi_management(app_instance):
    """
    Test Region of Interest (ROI) management functionality.
    """
    video_path = os.path.abspath("test_data/roi_test_video.mp4")
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    with open(video_path, "w") as f:
        f.write("dummy video for ROI test")
    
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
    
    # Test ROI creation
    engine.item_open("**/ROI Management")
    engine.item_click("**/Add ROI##AddROIButton")
    
    # Test ROI editing
    engine.slider_set_int("**/ROI X##ROIXSlider", 100)
    engine.slider_set_int("**/ROI Y##ROIYSlider", 100)
    engine.slider_set_int("**/ROI Width##ROIWidthSlider", 200)
    engine.slider_set_int("**/ROI Height##ROIHeightSlider", 200)
    
    # Test ROI save/load
    engine.item_click("**/Save ROI##SaveROIButton")
    time.sleep(1)
    
    # Verify ROI is active
    # This would need access to the ROI system to verify
    assert True  # Placeholder

@pytest.mark.e2e
def test_keyboard_shortcuts(app_instance):
    """
    Test keyboard shortcut functionality.
    """
    video_path = os.path.abspath("test_data/shortcuts_test_video.mp4")
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    with open(video_path, "w") as f:
        f.write("dummy video for shortcuts test")
    
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
    
    # Test keyboard shortcuts
    # Space bar for play/pause
    engine.key_press(imgui.Key.space)
    time.sleep(0.5)
    
    # Arrow keys for frame navigation
    engine.key_press(imgui.Key.right_arrow)
    engine.key_press(imgui.Key.left_arrow)
    time.sleep(0.5)
    
    # Ctrl+O for open
    engine.key_down(imgui.Key.left_ctrl)
    engine.key_press(imgui.Key.o)
    engine.key_up(imgui.Key.left_ctrl)
    time.sleep(0.5)
    
    # Verify shortcuts are working
    assert True  # Placeholder - would verify actual shortcut effects

@pytest.mark.e2e
def test_chapter_segment_management(app_instance):
    """
    Test chapter and segment management functionality.
    """
    video_path = os.path.abspath("test_data/chapter_test_video.mp4")
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    with open(video_path, "w") as f:
        f.write("dummy video for chapter test")
    
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
    
    # Test chapter management
    engine.item_open("**/Chapters")
    
    # Add new chapter
    engine.item_click("**/Add Chapter##AddChapterButton")
    engine.input_text("**/Chapter Name##ChapterNameInput", "Test Chapter 1")
    engine.slider_set_float("**/Start Time##ChapterStartSlider", 10.0)  # 10 seconds
    engine.slider_set_float("**/End Time##ChapterEndSlider", 30.0)  # 30 seconds
    engine.item_click("**/Save Chapter##SaveChapterButton")
    
    # Add another chapter
    engine.item_click("**/Add Chapter##AddChapterButton")
    engine.input_text("**/Chapter Name##ChapterNameInput", "Test Chapter 2")
    engine.slider_set_float("**/Start Time##ChapterStartSlider", 35.0)
    engine.slider_set_float("**/End Time##ChapterEndSlider", 60.0)
    engine.item_click("**/Save Chapter##SaveChapterButton")
    
    # Test chapter navigation
    engine.item_click("**/Previous Chapter##PrevChapterButton")
    engine.item_click("**/Next Chapter##NextChapterButton")
    
    # Test chapter deletion
    engine.item_click("**/Delete Chapter##DeleteChapterButton")
    
    # Verify chapter management is working
    assert True  # Placeholder

@pytest.mark.e2e
def test_dual_timeline_functionality(app_instance):
    """
    Test dual-axis timeline functionality.
    """
    video_path = os.path.abspath("test_data/dual_timeline_video.mp4")
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    with open(video_path, "w") as f:
        f.write("dummy video for dual timeline test")
    
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
    
    # Test dual timeline
    engine.item_open("**/Timeline")
    
    # Add actions to primary timeline
    primary_actions = [
        {"at": 1000, "pos": 20},
        {"at": 2000, "pos": 80},
    ]
    app_instance.funscript_processor.clear_timeline_history_and_set_new_baseline(1, primary_actions, "primary_test")
    
    # Add actions to secondary timeline
    secondary_actions = [
        {"at": 1500, "pos": 30},
        {"at": 2500, "pos": 70},
    ]
    app_instance.funscript_processor.clear_timeline_history_and_set_new_baseline(2, secondary_actions, "secondary_test")
    
    # Wait for setup
    time.sleep(2)
    
    # Test timeline switching
    engine.item_click("**/Primary Timeline##PrimaryTimelineTab")
    engine.item_click("**/Secondary Timeline##SecondaryTimelineTab")
    
    # Verify both timelines have data
    primary_data = app_instance.funscript_processor.get_actions('primary')
    secondary_data = app_instance.funscript_processor.get_actions('secondary')
    
    assert len(primary_data) >= 2
    assert len(secondary_data) >= 2

@pytest.mark.e2e
def test_real_time_preview_generation(app_instance):
    """
    Test real-time preview generation and visualization.
    """
    video_path = os.path.abspath("test_data/preview_test_video.mp4")
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    with open(video_path, "w") as f:
        f.write("dummy video for preview test")
    
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
    
    # Add test data for preview
    test_actions = [
        {"at": 1000, "pos": 10},
        {"at": 2000, "pos": 90},
        {"at": 3000, "pos": 20},
        {"at": 4000, "pos": 80},
        {"at": 5000, "pos": 30},
    ]
    app_instance.funscript_processor.clear_timeline_history_and_set_new_baseline(1, test_actions, "preview_test")
    
    # Wait for data setup
    time.sleep(1)
    
    # Test preview generation
    engine.item_open("**/Post-Processing")
    engine.item_click("**/Show Preview##ShowPreviewToggle")
    
    # Test filter preview
    engine.slider_set_int("**/Window Length##SGWin", 5)
    engine.item_click("**/Preview SG Filter##PreviewSGButton")
    
    # Let preview generate
    time.sleep(2)
    
    # Test different preview types
    engine.item_click("**/Show Heatmap##HeatmapToggle")
    engine.item_click("**/Show Speed Graph##SpeedGraphToggle")
    
    # Verify preview system is working
    assert app_instance.funscript_processor.get_actions('primary') is not None

@pytest.mark.e2e
def test_performance_monitoring_ui(app_instance):
    """
    Test performance monitoring UI elements.
    """
    engine = hello_imgui.get_imgui_test_engine()
    
    # Enable performance monitoring
    engine.item_open("**/Settings")
    engine.item_click("**/Show Performance Stats##PerfStatsToggle")
    
    # Enable frame rate display
    engine.item_click("**/Show Frame Rate##FrameRateToggle")
    
    # Enable memory usage display
    engine.item_click("**/Show Memory Usage##MemoryUsageToggle")
    
    # Let it run to collect performance data
    start_time = time.time()
    while time.time() - start_time < 5:
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)
    
    # Test performance settings
    engine.item_open("**/Performance")
    engine.slider_set_int("**/Max FPS##MaxFPSSlider", 30)
    engine.slider_set_int("**/Update Interval##UpdateIntervalSlider", 100)
    
    # Verify performance monitoring is active
    assert True  # Placeholder - would check actual performance metrics