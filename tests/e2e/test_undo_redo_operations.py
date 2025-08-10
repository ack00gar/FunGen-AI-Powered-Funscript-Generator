
import pytest
from unittest.mock import patch
from imgui_bundle import imgui, hello_imgui
import os
import time
import json

@pytest.fixture
def funscript_for_undo_redo(app_instance):
    """Prepares a funscript in the app for undo/redo tests."""
    actions = [
        {"at": 100, "pos": 20},
        {"at": 500, "pos": 80},
        {"at": 1000, "pos": 30},
    ]
    app_instance.funscript_processor.clear_timeline_history_and_set_new_baseline(1, actions, "test_setup")
    time.sleep(1)
    # Perform an action to create an undo point
    engine = hello_imgui.get_imgui_test_engine()
    engine.item_open("**/Post-Processing")
    engine.slider_set_float("**/Factor##AmplifyFactor", 1.5)
    engine.item_click("**/Apply Amplify##ApplyAmplify")
    time.sleep(1)
    return app_instance.funscript_processor.get_actions('primary')

@pytest.mark.e2e
def test_undo_operation(app_instance, funscript_for_undo_redo):
    """
    Tests the undo functionality.
    """
    actions_after_amplify = funscript_for_undo_redo
    engine = hello_imgui.get_imgui_test_engine()

    # 1. Trigger Undo
    engine.item_open("**/Edit")
    engine.item_click("**/Undo")
    time.sleep(1)

    # 2. Assert that the state has been reverted
    actions_after_undo = app_instance.funscript_processor.get_actions('primary')
    assert len(actions_after_undo) == 3
    assert actions_after_undo[0]["pos"] == 20 # Back to original value
    assert actions_after_undo[1]["pos"] == 80

@pytest.mark.e2e
def test_redo_operation(app_instance, funscript_for_undo_redo):
    """
    Tests the redo functionality.
    """
    actions_after_amplify = funscript_for_undo_redo
    engine = hello_imgui.get_imgui_test_engine()

    # 1. Trigger Undo first
    engine.item_open("**/Edit")
    engine.item_click("**/Undo")
    time.sleep(1)

    # 2. Trigger Redo
    engine.item_open("**/Edit")
    engine.item_click("**/Redo")
    time.sleep(1)

    # 3. Assert that the state has been restored to after the amplification
    actions_after_redo = app_instance.funscript_processor.get_actions('primary')
    assert actions_after_redo == actions_after_amplify
