
import pytest
from unittest.mock import patch
from imgui_bundle import imgui, hello_imgui
import os
import time
import json

@pytest.mark.e2e
def test_2_stage_analysis(app_instance):
    """
    Test case for the 2-stage analysis mode.
    """
    video_path = os.path.abspath("test_data/sample_video.mp4")
    output_dir = os.path.abspath("output")
    funscript_path = os.path.join(output_dir, "sample_video.funscript")

    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    with open(video_path, "w") as f:
        f.write("dummy video content")

    if os.path.exists(funscript_path):
        os.remove(funscript_path)

    engine = hello_imgui.get_imgui_test_engine()

    with patch('application.classes.file_dialog.ImGuiFileDialog.show') as mock_show_dialog:
        engine.item_open("**/File")
        engine.item_open("**/Open...")
        engine.item_click("**/Video...")
        args, kwargs = mock_show_dialog.call_args
        callback = kwargs.get('callback')
        callback(video_path)
    time.sleep(2)

    engine.item_open("**/Run Control")
    engine.item_click("**/Tracker Type##TrackerModeComboGlobal")
    engine.item_click("**/Offline AI Analysis (2-Stage)")

    engine.item_click("**/Start Full AI Analysis")

    start_time = time.time()
    while app_instance.stage_processor.full_analysis_active:
        if time.time() - start_time > 120:
            pytest.fail("2-Stage analysis timed out")
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)

    assert os.path.exists(funscript_path), f"Funscript file not found at {funscript_path}"

@pytest.mark.e2e
def test_oscillation_detector_analysis(app_instance):
    """
    Test case for the Oscillation Detector analysis mode.
    """
    video_path = os.path.abspath("test_data/sample_video.mp4")
    output_dir = os.path.abspath("output")
    funscript_path = os.path.join(output_dir, "sample_video.funscript")

    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    with open(video_path, "w") as f:
        f.write("dummy video content")

    if os.path.exists(funscript_path):
        os.remove(funscript_path)

    engine = hello_imgui.get_imgui_test_engine()

    with patch('application.classes.file_dialog.ImGuiFileDialog.show') as mock_show_dialog:
        engine.item_open("**/File")
        engine.item_open("**/Open...")
        engine.item_click("**/Video...")
        args, kwargs = mock_show_dialog.call_args
        callback = kwargs.get('callback')
        callback(video_path)
    
    start_time = time.time()
    while not app_instance.file_manager.video_path:
        if time.time() - start_time > 10:
            pytest.fail("Video load timed out")
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)

    engine.item_open("**/Run Control")
    engine.item_click("**/Tracker Type##TrackerModeComboGlobal")
    engine.item_click("**/Oscillation Detector")

    engine.item_click("**/Start Live Tracking")
    
    start_time = time.time()
    while time.time() - start_time < 10: # Let it run for 10 seconds
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)
        
    engine.item_click("**/Abort/Stop Process##AbortGeneral")
    
    start_time = time.time()
    while app_instance.stage_processor.is_running():
        if time.time() - start_time > 10:
            pytest.fail("Abort process timed out")
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)

    assert os.path.exists(funscript_path), f"Funscript file not found at {funscript_path}"
    with open(funscript_path, "r") as f:
        data = json.load(f)
        assert len(data["actions"]) > 0
