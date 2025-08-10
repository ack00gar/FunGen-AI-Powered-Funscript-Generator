
import pytest
from unittest.mock import patch
from imgui_bundle import imgui, hello_imgui
import os
import time

@pytest.mark.e2e
def test_new_project(app_instance):
    """
    Tests creating a new project to ensure the state is reset.
    """
    engine = hello_imgui.get_imgui_test_engine()

    # 1. Load a video to create a dirty state
    video_path = os.path.abspath("test_data/sample_video.mp4")
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
        
    assert app_instance.file_manager.video_path is not None
    app_instance.project_manager.project_dirty = True

    # 2. Click "New Project"
    engine.item_open("**/File")
    engine.item_click("**/New Project")
    start_time = time.time()
    while app_instance.project_manager.project_dirty:
        if time.time() - start_time > 5:
            pytest.fail("New project state change timed out")
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)

    # 3. Assert that the state has been reset
    assert app_instance.file_manager.video_path is None
    assert app_instance.project_manager.project_dirty is False
    assert not app_instance.funscript_processor.get_actions('primary')

@pytest.mark.e2e
def test_save_and_load_project(app_instance):
    """
    Tests saving the project to a file and then loading it back.
    """
    engine = hello_imgui.get_imgui_test_engine()
    project_path = os.path.abspath("output/test_project.fgnproj")
    video_path = os.path.abspath("test_data/sample_video.mp4")

    # Clean up previous test artifacts
    if os.path.exists(project_path):
        os.remove(project_path)

    # 1. Load a video and change a setting to make the project dirty
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
        
    app_instance.app_state_ui.ui_view_mode = 'simple' # Change a setting
    app_instance.project_manager.project_dirty = True

    # 2. Save the project
    with patch('application.classes.file_dialog.ImGuiFileDialog.show') as mock_save_dialog:
        engine.item_open("**/File")
        engine.item_click("**/Save Project As...")
        args, kwargs = mock_save_dialog.call_args
        callback = kwargs.get('callback')
        callback(project_path)
    
    start_time = time.time()
    while not os.path.exists(project_path):
        if time.time() - start_time > 5:
            pytest.fail("Project save timed out")
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)
        
    assert os.path.exists(project_path)

    # 3. Reset the application state
    engine.item_open("**/File")
    engine.item_click("**/New Project")
    start_time = time.time()
    while app_instance.project_manager.project_dirty:
        if time.time() - start_time > 5:
            pytest.fail("New project state change timed out")
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)
        
    assert app_instance.file_manager.video_path is None

    # 4. Load the project back
    with patch('application.classes.file_dialog.ImGuiFileDialog.show') as mock_load_dialog:
        engine.item_open("**/File")
        engine.item_open("**/Open...")
        engine.item_click("**/Project...")
        args, kwargs = mock_load_dialog.call_args
        callback = kwargs.get('callback')
        callback(project_path)
        
    start_time = time.time()
    while not app_instance.file_manager.video_path:
        if time.time() - start_time > 10:
            pytest.fail("Project load timed out")
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)

    # 5. Assert the state was restored
    assert app_instance.file_manager.video_path == video_path
    assert app_instance.app_state_ui.ui_view_mode == 'simple'
