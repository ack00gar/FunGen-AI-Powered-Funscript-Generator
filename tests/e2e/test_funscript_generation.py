
import pytest
from unittest.mock import patch
from imgui_bundle import imgui, hello_imgui
import main
import os
import time

@pytest.mark.e2e
def test_funscript_generation(app_instance):
    """
    Test case for generating a funscript file.
    """
    # Path to a dummy video file for testing
    video_path = os.path.abspath("test_data/sample_video.mp4")
    output_dir = os.path.abspath("output")
    funscript_path = os.path.join(output_dir, "sample_video.funscript")

    # Ensure the dummy video file exists
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    with open(video_path, "w") as f:
        f.write("dummy video content")

    # Clean up any previous funscript file
    if os.path.exists(funscript_path):
        os.remove(funscript_path)

    # Use the test engine to interact with the GUI
    engine = hello_imgui.get_imgui_test_engine()

    # 1. Load the video
    with patch('application.classes.file_dialog.ImGuiFileDialog.show') as mock_show_dialog:
        engine.item_open("**/File")
        engine.item_open("**/Open...")
        engine.item_click("**/Video...")
        args, kwargs = mock_show_dialog.call_args
        callback = kwargs.get('callback')
        callback(video_path)

    # Wait for the video to load
    start_time = time.time()
    while not app_instance.file_manager.video_path:
        if time.time() - start_time > 10:
            pytest.fail("Video load timed out")
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)

    # 2. Select the 3-stage analysis mode
    engine.item_open("**/Run Control") # Ensure the tab is selected
    engine.item_click("**/Tracker Type##TrackerModeComboGlobal")
    engine.item_click("**/Offline AI Analysis (3-Stage)")

    # 3. Start the analysis
    engine.item_click("**/Start Full AI Analysis")

    # 4. Wait for the analysis to complete
    start_time = time.time()
    while app_instance.stage_processor.full_analysis_active:
        if time.time() - start_time > 120: # 2 minute timeout
            pytest.fail("Funscript generation timed out")
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)

    # 5. Assert that the funscript file was created
    assert os.path.exists(funscript_path), f"Funscript file not found at {funscript_path}"
