import pytest
from unittest.mock import patch, MagicMock
from application.logic.app_logic import ApplicationLogic
from config.constants import TrackerMode

# Mock AppGui since it may not be available in test environment
class MockAppGui:
    def __init__(self):
        pass

@pytest.mark.integration
def test_oscillation_settings_update():
    """Test that updating oscillation settings propagates to the tracker."""
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
        app = ApplicationLogic(is_cli=False) # is_cli=False to init gui_instance
        app.app_state_ui.selected_tracker_mode = TrackerMode.OSCILLATION_DETECTOR

        # Mock the tracker and stage_processor
        app.tracker = MagicMock()
        app.stage_processor = MagicMock()

        # Get the control panel UI
        control_panel = app.gui_instance.control_panel_ui

        # Simulate changing the sensitivity
        new_sensitivity = 2.5
        with patch('imgui.slider_float', return_value=(True, new_sensitivity)):
            with patch.object(app.app_settings, 'set') as mock_settings_set:
                control_panel._render_oscillation_detector_settings()

                # Verify that the setting was changed
                mock_settings_set.assert_any_call("oscillation_detector_sensitivity", new_sensitivity)

                # Verify that the tracker's update method was called
                app.tracker.update_oscillation_sensitivity.assert_called_once()

                # Verify that the stage_processor's update method was called
                app.stage_processor.update_tracker_config.assert_called_once()

        # Reset mocks
        app.tracker.reset_mock()
        app.stage_processor.reset_mock()

        # Simulate changing the grid size
        new_grid_size = 40
        with patch('imgui.slider_int', return_value=(True, new_grid_size)):
            with patch.object(app.app_settings, 'set') as mock_settings_set:
                control_panel._render_oscillation_detector_settings()

                # Verify that the setting was changed
                mock_settings_set.assert_any_call("oscillation_detector_grid_size", new_grid_size)

                # Verify that the tracker's update method was called
                app.tracker.update_oscillation_grid_size.assert_called_once()

                # Verify that the stage_processor's update method was called
                app.stage_processor.update_tracker_config.assert_called_once()
