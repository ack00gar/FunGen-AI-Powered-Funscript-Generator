import pytest
from unittest.mock import patch
from imgui_bundle import imgui, hello_imgui
import main
import os
import time

@pytest.mark.e2e
def test_load_video(app_instance):
    """
    Test case for loading a video file through the GUI.
    """
    # Path to a dummy video file for testing
    video_path = os.path.abspath("test_data/sample_video.mp4")

    # Ensure the dummy video file exists
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    with open(video_path, "w") as f:
        f.write("dummy video content")

    # Use the test engine to interact with the GUI
    engine = hello_imgui.get_imgui_test_engine()

    # 1. Open the "File" menu
    engine.item_open("**/File")

    # 2. Open the "Open..." sub-menu
    engine.item_open("**/Open...")

    # 3. Click the "Video..." menu item
    with patch('application.classes.file_dialog.ImGuiFileDialog.show') as mock_show_dialog:
        engine.item_click("**/Video...")

        # 4. Simulate the file dialog returning our test video path
        args, kwargs = mock_show_dialog.call_args
        callback = kwargs.get('callback')
        assert callback is not None, "Callback function not passed to file dialog"
        callback(video_path)

    # 5. Assert that the video was loaded successfully
    start_time = time.time()
    while not app_instance.file_manager.video_path:
        if time.time() - start_time > 10:
            pytest.fail("Video load timed out")
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)
        
    assert app_instance.file_manager.video_path == video_path
    assert app_instance.processor.video_info is not None
    assert app_instance.processor.video_info['total_frames'] > 0

@pytest.mark.e2e
def test_load_video_with_associated_funscript(app_instance):
    """
    Test case for loading a video that has an associated funscript file.
    """
    video_path = os.path.abspath("test_data/sample_with_funscript.mp4")
    funscript_path = os.path.abspath("test_data/sample_with_funscript.funscript")

    # Create dummy files
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    with open(video_path, "w") as f:
        f.write("dummy video")
    with open(funscript_path, "w") as f:
        f.write('{"actions": [{"at": 100, "pos": 50}]}')

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
        
    assert app_instance.file_manager.video_path == video_path
    assert len(app_instance.funscript_processor.get_actions('primary')) > 0