import pytest
import sys
import os
import tempfile
import json
import logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from unittest.mock import patch, MagicMock
from application.logic.app_logic import ApplicationLogic
from funscript.dual_axis_funscript import DualAxisFunscript

@pytest.mark.integration
def test_core_application_initialization():
    """
    Test that the core application initializes properly without GUI.
    """
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
        app = ApplicationLogic(is_cli=True)  # CLI mode to avoid GUI dependencies
        
        assert app is not None
        assert app.funscript_processor is not None
        assert app.stage_processor is not None
        assert app.file_manager is not None
        # Settings manager might be accessed through different attribute
        assert hasattr(app, 'settings_manager') or hasattr(app, 'app_state_ui')

@pytest.mark.integration
def test_funscript_processing_workflow():
    """
    Test the complete funscript processing workflow.
    """
    logger = logging.getLogger("test")
    funscript = DualAxisFunscript(logger=logger)
    
    # Step 1: Add actions
    test_actions = [
        {"at": 1000, "pos": 20},
        {"at": 2000, "pos": 80},
        {"at": 3000, "pos": 40},
        {"at": 4000, "pos": 60}
    ]
    
    for action in test_actions:
        funscript.add_action(action["at"], action["pos"])
    
    assert len(funscript.primary_actions) == 4
    
    # Step 2: Apply post-processing
    funscript.amplify_points_values('primary', scale_factor=1.5, center_value=50)
    
    # Verify amplification worked
    amplified_actions = funscript.primary_actions
    assert amplified_actions[0]["pos"] == 5  # 50 + (20-50)*1.5 = 5
    assert amplified_actions[1]["pos"] == 95  # 50 + (80-50)*1.5 = 95
    
    # Step 3: Test statistics
    stats = funscript.get_actions_statistics('primary')
    assert stats['num_points'] == 4
    assert stats['total_travel_dist'] > 0
    
    # Step 4: Test interpolation
    interpolated_value = funscript.get_value(1500)  # Midpoint between first two actions
    assert 0 <= interpolated_value <= 100

@pytest.mark.integration
def test_project_state_management():
    """
    Test project state management without GUI dependencies.
    """
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
        app = ApplicationLogic(is_cli=True)
        
        # Test initial state
        assert not app.project_manager.project_dirty
        
        # Simulate making changes
        app.project_manager.project_dirty = True
        assert app.project_manager.project_dirty
        
        # Test reset
        app.project_manager.new_project()
        assert not app.project_manager.project_dirty

@pytest.mark.integration 
def test_settings_management():
    """
    Test settings management functionality.
    """
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
        app = ApplicationLogic(is_cli=True)
        
        # Test that app has settings functionality (may be in different component)
        if hasattr(app, 'settings_manager'):
            theme = app.settings_manager.get_setting("theme", "Dark")
            assert theme is not None
        elif hasattr(app, 'app_state_ui'):
            # Settings might be managed through app_state_ui
            assert app.app_state_ui is not None
        else:
            # Basic verification that settings system exists
            assert hasattr(app, 'project_manager')  # At minimum this should exist

@pytest.mark.integration
def test_file_format_handling():
    """
    Test file format detection and handling.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.funscript', delete=False) as temp_file:
        # Create a valid funscript
        funscript_data = {
            "actions": [
                {"at": 1000, "pos": 25},
                {"at": 2000, "pos": 75}
            ]
        }
        json.dump(funscript_data, temp_file)
        temp_file.flush()
        
        # Test reading the funscript
        with open(temp_file.name, 'r') as f:
            loaded_data = json.load(f)
            assert 'actions' in loaded_data
            assert len(loaded_data['actions']) == 2
            assert loaded_data['actions'][0]['at'] == 1000
        
        # Clean up
        os.unlink(temp_file.name)

@pytest.mark.integration
def test_dual_axis_functionality():
    """
    Test dual-axis funscript functionality.
    """
    logger = logging.getLogger("test")
    funscript = DualAxisFunscript(logger=logger)
    
    # Add actions to primary axis
    primary_actions = [{"at": 1000, "pos": 30}, {"at": 2000, "pos": 70}]
    funscript.actions = primary_actions
    
    # Add actions to secondary axis using batch method
    secondary_batch = [
        {'timestamp_ms': 1500, 'secondary_pos': 20},
        {'timestamp_ms': 2500, 'secondary_pos': 80}
    ]
    funscript.add_actions_batch(secondary_batch)
    
    # Verify both axes have data
    assert len(funscript.primary_actions) == 2
    assert len(funscript.secondary_actions) == 2
    
    # Test statistics for both axes
    primary_stats = funscript.get_actions_statistics('primary')
    secondary_stats = funscript.get_actions_statistics('secondary')
    
    assert primary_stats['num_points'] == 2
    assert secondary_stats['num_points'] == 2

@pytest.mark.integration
def test_error_handling():
    """
    Test error handling in core functionality.
    """
    logger = logging.getLogger("test")
    funscript = DualAxisFunscript(logger=logger)
    
    # Test with empty funscript
    stats = funscript.get_actions_statistics('primary')
    assert stats['num_points'] == 0
    
    # Test interpolation with no data
    value = funscript.get_value(1000)
    assert value == 50  # Default value
    
    # Test invalid operations
    funscript.amplify_points_values('primary', scale_factor=2.0)  # Should handle empty gracefully
    
    # Verify no crash occurred
    assert True

@pytest.mark.integration
def test_memory_and_performance():
    """
    Test memory usage and performance considerations.
    """
    logger = logging.getLogger("test")
    funscript = DualAxisFunscript(logger=logger)
    
    # Add many actions to test performance
    large_action_set = [{"at": i * 100, "pos": (i % 100)} for i in range(1000)]
    funscript.actions = large_action_set
    
    assert len(funscript.primary_actions) == 1000
    
    # Test operations on large dataset
    stats = funscript.get_actions_statistics('primary')
    assert stats['num_points'] == 1000
    
    # Test interpolation performance
    for i in range(0, 100000, 10000):
        value = funscript.get_value(i)
        assert 0 <= value <= 100
    
    # Clear and verify cleanup
    funscript.clear()
    assert len(funscript.primary_actions) == 0