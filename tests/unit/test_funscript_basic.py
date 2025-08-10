import pytest
import sys
import os
import logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from funscript.dual_axis_funscript import DualAxisFunscript

@pytest.mark.unit
def test_dual_axis_funscript_creation():
    """Test basic creation and functionality of DualAxisFunscript."""
    logger = logging.getLogger("test")
    funscript = DualAxisFunscript(logger=logger)
    
    assert funscript is not None
    assert len(funscript.primary_actions) == 0
    assert len(funscript.secondary_actions) == 0

@pytest.mark.unit
def test_add_actions():
    """Test adding actions to funscript."""
    logger = logging.getLogger("test")
    funscript = DualAxisFunscript(logger=logger)
    
    # Add single action
    funscript.add_action(1000, 50)  # Default is primary axis
    assert len(funscript.primary_actions) == 1
    assert funscript.primary_actions[0]['at'] == 1000
    assert funscript.primary_actions[0]['pos'] == 50
    
    # Add batch actions
    batch_actions = [
        {'timestamp_ms': 2000, 'primary_pos': 80},
        {'timestamp_ms': 3000, 'primary_pos': 20},
    ]
    funscript.add_actions_batch(batch_actions)
    
    assert len(funscript.primary_actions) == 3
    assert funscript.primary_actions[-1]['at'] == 3000

@pytest.mark.unit
def test_interpolation():
    """Test interpolation functionality."""
    logger = logging.getLogger("test")
    funscript = DualAxisFunscript(logger=logger)
    
    # Add test actions
    actions = [
        {"at": 1000, "pos": 0},
        {"at": 2000, "pos": 100},
        {"at": 3000, "pos": 50},
    ]
    funscript.actions = actions
    
    # Test interpolation
    assert funscript.get_value(1000) == 0
    assert funscript.get_value(2000) == 100
    assert funscript.get_value(1500) == 50  # Midpoint interpolation
    assert funscript.get_value(500) == 0   # Before first action
    assert funscript.get_value(4000) == 50  # After last action

@pytest.mark.unit
def test_statistics():
    """Test statistics calculation."""
    logger = logging.getLogger("test")
    funscript = DualAxisFunscript(logger=logger)
    
    # Add test actions
    actions = [
        {"at": 0, "pos": 10},
        {"at": 1000, "pos": 90},
        {"at": 2000, "pos": 20},
        {"at": 3000, "pos": 80},
    ]
    funscript.actions = actions
    
    stats = funscript.get_actions_statistics('primary')
    
    assert stats['num_points'] == 4
    assert stats['duration_scripted_s'] == 3.0
    assert stats['min_pos'] == 10
    assert stats['max_pos'] == 90
    assert stats['total_travel_dist'] > 0
    assert stats['num_strokes'] > 0

@pytest.mark.unit 
def test_clear_functionality():
    """Test clearing actions."""
    logger = logging.getLogger("test")
    funscript = DualAxisFunscript(logger=logger)
    
    # Add actions
    actions = [{"at": 1000, "pos": 50}, {"at": 2000, "pos": 80}]
    funscript.actions = actions
    
    assert len(funscript.primary_actions) == 2
    
    # Clear all
    funscript.clear()
    
    assert len(funscript.primary_actions) == 0
    assert len(funscript.secondary_actions) == 0

@pytest.mark.unit
def test_basic_processing():
    """Test basic post-processing functionality."""
    logger = logging.getLogger("test")
    funscript = DualAxisFunscript(logger=logger)
    
    # Add test actions
    actions = [
        {"at": 1000, "pos": 40},
        {"at": 2000, "pos": 60},
        {"at": 3000, "pos": 30},
    ]
    funscript.actions = actions
    
    # Test basic functionality
    assert len(funscript.primary_actions) == 3
    
    # Test invert operation
    funscript.invert_points_values('primary')
    inverted_actions = funscript.primary_actions
    
    # Inverted values: 100 - original
    assert inverted_actions[0]['pos'] == 60  # 100 - 40
    assert inverted_actions[1]['pos'] == 40  # 100 - 60
    assert inverted_actions[2]['pos'] == 70  # 100 - 30