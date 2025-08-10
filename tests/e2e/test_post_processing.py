
import pytest
from unittest.mock import patch
from imgui_bundle import imgui, hello_imgui
import os
import time
import json

@pytest.fixture
def funscript_for_processing(app_instance):
    """Prepares a funscript in the app for post-processing tests."""
    actions = [
        {"at": 100, "pos": 20},
        {"at": 500, "pos": 80},
        {"at": 1000, "pos": 30},
    ]
    app_instance.funscript_processor.clear_timeline_history_and_set_new_baseline(1, actions, "test_setup")
    start_time = time.time()
    while not app_instance.funscript_processor.get_actions('primary'):
        if time.time() - start_time > 5:
            pytest.fail("Funscript setup timed out")
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)
    return actions

@pytest.mark.e2e
def test_amplify_processing(app_instance, funscript_for_processing):
    """
    Tests the amplify post-processing tool.
    """
    engine = hello_imgui.get_imgui_test_engine()
    actions_before = app_instance.funscript_processor.get_actions('primary')

    # 1. Go to the Post-Processing tab
    engine.item_open("**/Post-Processing")

    # 2. Set the amplify factor and center
    engine.slider_set_float("**/Factor##AmplifyFactor", 2.0)
    engine.slider_set_int("**/Center##AmplifyCenter", 50)

    # 3. Apply the amplification
    engine.item_click("**/Apply Amplify##ApplyAmplify")
    
    start_time = time.time()
    while app_instance.funscript_processor.get_actions('primary') == actions_before:
        if time.time() - start_time > 5:
            pytest.fail("Amplify processing timed out")
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)

    # 4. Assert that the actions have been amplified
    actions = app_instance.funscript_processor.get_actions('primary')
    assert actions[0]["pos"] < 20 # Should be moved away from the center
    assert actions[1]["pos"] > 80 # Should be moved away from the center

@pytest.mark.e2e
def test_savitzky_golay_processing(app_instance, funscript_for_processing):
    """
    Tests the Savitzky-Golay filter post-processing tool.
    """
    engine = hello_imgui.get_imgui_test_engine()
    actions_before = app_instance.funscript_processor.get_actions('primary')

    engine.item_open("**/Post-Processing")
    engine.slider_set_int("**/Window Length##SGWin", 5)
    engine.slider_set_int("**/Polyorder##SGPoly", 2)
    engine.item_click("**/Apply Savitzky-Golay##ApplySG")
    
    start_time = time.time()
    while app_instance.funscript_processor.get_actions('primary') == actions_before:
        if time.time() - start_time > 5:
            pytest.fail("Savitzky-Golay processing timed out")
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)

    actions_after = app_instance.funscript_processor.get_actions('primary')
    assert actions_before != actions_after # The filter should have modified the actions

@pytest.mark.e2e
def test_rdp_simplification_processing(app_instance, funscript_for_processing):
    """
    Tests the RDP simplification post-processing tool.
    """
    engine = hello_imgui.get_imgui_test_engine()

    # Add more points for a better test of RDP
    actions = funscript_for_processing + [
        {"at": 1200, "pos": 32},
        {"at": 1500, "pos": 28},
    ]
    app_instance.funscript_processor.clear_timeline_history_and_set_new_baseline(1, actions, "rdp_setup")
    start_time = time.time()
    while len(app_instance.funscript_processor.get_actions('primary')) != len(actions):
        if time.time() - start_time > 5:
            pytest.fail("RDP setup timed out")
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)
    actions_before = app_instance.funscript_processor.get_actions('primary')

    engine.item_open("**/Post-Processing")
    engine.slider_set_float("**/Epsilon##RDPEps", 5.0)
    engine.item_click("**/Apply RDP##ApplyRDP")
    
    start_time = time.time()
    while app_instance.funscript_processor.get_actions('primary') == actions_before:
        if time.time() - start_time > 5:
            pytest.fail("RDP processing timed out")
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)

    actions_after = app_instance.funscript_processor.get_actions('primary')
    assert len(actions_after) < len(actions) # RDP should have removed some points

@pytest.mark.e2e
def test_automatic_post_processing(app_instance, funscript_for_processing):
    """
    Tests the automatic post-processing feature.
    """
    engine = hello_imgui.get_imgui_test_engine()
    actions_before = app_instance.funscript_processor.get_actions('primary')

    engine.item_open("**/Post-Processing")
    engine.item_click("**/Enable Automatic Post-Processing on Completion")
    engine.item_click("**/Run Post-Processing Now##RunAutoPostProcessButton")
    
    start_time = time.time()
    while app_instance.funscript_processor.get_actions('primary') == actions_before:
        if time.time() - start_time > 5:
            pytest.fail("Automatic post-processing timed out")
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)

    actions_after = app_instance.funscript_processor.get_actions('primary')
    assert actions_before != actions_after
