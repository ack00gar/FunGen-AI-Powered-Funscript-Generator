
import pytest
from unittest.mock import patch
from imgui_bundle import imgui, hello_imgui
import os
import time
import json

@pytest.fixture
def dummy_funscript_file():
    """Creates a dummy funscript file for testing and cleans it up afterward."""
    funscript_path = os.path.abspath("test_data/dummy.funscript")
    funscript_data = {
        "actions": [
            {"at": 100, "pos": 10},
            {"at": 500, "pos": 90},
            {"at": 1000, "pos": 20},
        ]
    }
    os.makedirs(os.path.dirname(funscript_path), exist_ok=True)
    with open(funscript_path, "w") as f:
        json.dump(funscript_data, f)
    
    yield funscript_path
    
    if os.path.exists(funscript_path):
        os.remove(funscript_path)

@pytest.mark.e2e
def test_import_funscript_timeline1(app_instance, dummy_funscript_file):
    """
    Tests importing a funscript file into Timeline 1.
    """
    engine = hello_imgui.get_imgui_test_engine()

    # 1. Reset the state
    engine.item_open("**/File")
    engine.item_click("**/New Project")
    start_time = time.time()
    while app_instance.project_manager.project_dirty:
        if time.time() - start_time > 5:
            pytest.fail("New project state change timed out")
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)

    # 2. Import the funscript
    with patch('application.classes.file_dialog.ImGuiFileDialog.show') as mock_dialog:
        engine.item_open("**/File")
        engine.item_open("**/Import...")
        engine.item_click("**/Funscript to Timeline 1...")
        args, kwargs = mock_dialog.call_args
        callback = kwargs.get('callback')
        callback(dummy_funscript_file)
    
    start_time = time.time()
    while not app_instance.funscript_processor.funscript_dual_axis_obj.primary_actions:
        if time.time() - start_time > 5:
            pytest.fail("Funscript import timed out")
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)

    # 3. Assert that the actions were loaded into the primary timeline
    actions = app_instance.funscript_processor.funscript_dual_axis_obj.primary_actions
    assert len(actions) == 3
    assert actions[0]["pos"] == 10
    assert actions[1]["pos"] == 90
    assert actions[2]["pos"] == 20

@pytest.mark.e2e
def test_import_funscript_timeline2(app_instance, dummy_funscript_file):
    """
    Tests importing a funscript file into Timeline 2.
    """
    engine = hello_imgui.get_imgui_test_engine()

    # 1. Reset the state
    engine.item_open("**/File")
    engine.item_click("**/New Project")
    start_time = time.time()
    while app_instance.project_manager.project_dirty:
        if time.time() - start_time > 5:
            pytest.fail("New project state change timed out")
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)

    # 2. Import the funscript
    with patch('application.classes.file_dialog.ImGuiFileDialog.show') as mock_dialog:
        engine.item_open("**/File")
        engine.item_open("**/Import...")
        engine.item_click("**/Funscript to Timeline 2...")
        args, kwargs = mock_dialog.call_args
        callback = kwargs.get('callback')
        callback(dummy_funscript_file)
    
    start_time = time.time()
    while not app_instance.funscript_processor.funscript_dual_axis_obj.secondary_actions:
        if time.time() - start_time > 5:
            pytest.fail("Funscript import timed out")
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)

    # 3. Assert that the actions were loaded into the secondary timeline
    actions = app_instance.funscript_processor.funscript_dual_axis_obj.secondary_actions
    assert len(actions) == 3
    assert actions[0]["pos"] == 10

@pytest.mark.e2e
def test_export_funscript_timeline1(app_instance, dummy_funscript_file):
    """
    Tests exporting a funscript file from Timeline 1.
    """
    engine = hello_imgui.get_imgui_test_engine()
    export_path = os.path.abspath("output/exported_t1.funscript")

    if os.path.exists(export_path):
        os.remove(export_path)

    # 1. Load a funscript into Timeline 1 using the available API
    with open(dummy_funscript_file, 'r') as f:
        funscript_data = json.load(f)
        actions = funscript_data.get('actions', [])
        app_instance.funscript_processor.clear_timeline_history_and_set_new_baseline(1, actions, "test_load")
    
    start_time = time.time()
    while not app_instance.funscript_processor.funscript_dual_axis_obj.primary_actions:
        if time.time() - start_time > 5:
            pytest.fail("Funscript load timed out")
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)

    # 2. Export the funscript
    with patch('application.classes.file_dialog.ImGuiFileDialog.show') as mock_dialog:
        engine.item_open("**/File")
        engine.item_open("**/Export...")
        engine.item_click("**/Funscript from Timeline 1...")
        args, kwargs = mock_dialog.call_args
        callback = kwargs.get('callback')
        callback(export_path)
    
    start_time = time.time()
    while not os.path.exists(export_path):
        if time.time() - start_time > 5:
            pytest.fail("Funscript export timed out")
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)

    # 3. Assert the file was created and has the correct content
    assert os.path.exists(export_path)
    with open(export_path, "r") as f:
        data = json.load(f)
        assert len(data["actions"]) == 3
        assert data["actions"][1]["pos"] == 90

@pytest.mark.e2e
def test_export_funscript_timeline2(app_instance, dummy_funscript_file):
    """
    Tests exporting a funscript file from Timeline 2.
    """
    engine = hello_imgui.get_imgui_test_engine()
    export_path = os.path.abspath("output/exported_t2.funscript")

    if os.path.exists(export_path):
        os.remove(export_path)

    # 1. Load a funscript into Timeline 2 using the available API
    with open(dummy_funscript_file, 'r') as f:
        funscript_data = json.load(f)
        actions = funscript_data.get('actions', [])
        app_instance.funscript_processor.clear_timeline_history_and_set_new_baseline(2, actions, "test_load")
    
    start_time = time.time()
    while not app_instance.funscript_processor.funscript_dual_axis_obj.secondary_actions:
        if time.time() - start_time > 5:
            pytest.fail("Funscript load timed out")
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)

    # 2. Export the funscript
    with patch('application.classes.file_dialog.ImGuiFileDialog.show') as mock_dialog:
        engine.item_open("**/File")
        engine.item_open("**/Export...")
        engine.item_click("**/Funscript from Timeline 2...")
        args, kwargs = mock_dialog.call_args
        callback = kwargs.get('callback')
        callback(export_path)
    
    start_time = time.time()
    while not os.path.exists(export_path):
        if time.time() - start_time > 5:
            pytest.fail("Funscript export timed out")
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)

    # 3. Assert the file was created and has the correct content
    assert os.path.exists(export_path)
    with open(export_path, "r") as f:
        data = json.load(f)
        assert len(data["actions"]) == 3
        assert data["actions"][1]["pos"] == 90
