
import pytest
from unittest.mock import patch
from imgui_bundle import imgui, hello_imgui
import os
import time
import json

@pytest.mark.e2e
def test_settings_persistence(app_instance):
    """
    Tests if settings are saved on exit and reloaded on start.
    """
    engine = hello_imgui.get_imgui_test_engine()

    # 1. Change a setting
    engine.item_open("**/Settings")
    engine.item_click("**/Theme##ThemeSelector")
    engine.item_click("**/Classic") # Change to a non-default theme
    time.sleep(1)

    # 2. Simulate shutdown and restart by re-initializing the app fixture
    # (This is a simplified simulation for testing purposes)
    app_instance.settings_manager.save_settings()
    app_instance.settings_manager.load_settings()

    # 3. Assert that the setting was persisted
    assert app_instance.settings_manager.get_setting("theme") == "Classic"

    # 4. Reset to default for other tests
    engine.item_click("**/Theme##ThemeSelector")
    engine.item_click("**/Dark")
    time.sleep(1)
    app_instance.settings_manager.save_settings()
