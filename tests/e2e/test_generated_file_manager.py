
import pytest
from unittest.mock import patch
from imgui_bundle import imgui, hello_imgui
import os
import time

@pytest.mark.e2e
def test_generated_file_manager_tracking(app_instance):
    """
    Test if the Generated File Manager correctly tracks a newly created funscript.
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
    engine.item_click("**/Offline AI Analysis (3-Stage)")
    engine.item_click("**/Start Full AI Analysis")

    start_time = time.time()
    while app_instance.stage_processor.full_analysis_active:
        time.sleep(1)
        if time.time() - start_time > 120:
            pytest.fail("Funscript generation timed out")

    # 1. Open the generated file manager
    engine.item_open("**/Window")
    engine.item_click("**/Generated File Manager")
    time.sleep(1)

    # 2. Assert that the newly generated file is listed
    # This requires a way to access the state of the file manager window.
    # For now, we'll check the underlying data model.
    tracked_files = app_instance.generated_file_manager.get_tracked_files()
    assert any(f["funscript_path"] == funscript_path for f in tracked_files)
