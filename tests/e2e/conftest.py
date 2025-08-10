
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pytest
import threading
import time
from unittest.mock import patch, MagicMock
from application.logic.app_logic import ApplicationLogic
from application.gui_components.app_gui import GUI
from imgui_bundle import hello_imgui

@pytest.fixture(scope="session")
def app_instance():
    """
    A pytest fixture that sets up and tears down the application for testing.
    """
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup', return_value=None), \
         patch('application.gui_components.app_gui.GUI.run') as mock_run, \
         patch('application.gui_components.app_gui.GUI.init_glfw', return_value=True), \
         patch('imgui_bundle.hello_imgui.get_imgui_test_engine') as mock_engine:
        
        # Create a mock test engine with all the methods used in tests
        mock_test_engine = MagicMock()
        mock_test_engine.item_open.return_value = None
        mock_test_engine.item_click.return_value = None
        mock_test_engine.slider_set_float.return_value = None
        mock_test_engine.slider_set_int.return_value = None
        mock_test_engine.input_text.return_value = None
        mock_test_engine.key_press.return_value = None
        mock_test_engine.key_down.return_value = None
        mock_test_engine.key_up.return_value = None
        mock_engine.return_value = mock_test_engine
        
        core_app = ApplicationLogic(is_cli=False)
        gui = GUI(app_logic=core_app)
        core_app.gui_instance = gui
        
        # Mock the GUI initialization and add is_initialized property
        gui.window = MagicMock()  # Mock the window object
        gui.impl = MagicMock()    # Mock the renderer
        
        # Add is_initialized as a property that returns True
        type(gui).is_initialized = property(lambda self: True)
        
        # Add run_one_frame method for tests
        gui.run_one_frame = MagicMock(return_value=None)
        
        mock_run.return_value = None

        try:
            yield core_app
        finally:
            # Clean shutdown
            pass
