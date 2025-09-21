#!/usr/bin/env python3
"""
Test the simplified anti-jerk plugin.
"""

import json
import sys
import os

# Add the project root to the path so we can import the plugin
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from funscript.plugins.anti_jerk_plugin import AntiJerkPlugin

class MockDualAxisFunscript:
    """Mock funscript for testing"""
    def __init__(self, actions):
        self.primary_actions = [{'at': action['at'], 'pos': action['pos']} for action in actions]
        self.secondary_actions = []

def test_simplified_plugin():
    """Test the simplified anti-jerk plugin"""
    
    # Create the example funscript data
    example_actions = [
        {"at": 16, "pos": 50},
        {"at": 716, "pos": 81},
        {"at": 1084, "pos": 2},
        {"at": 1334, "pos": 94},
        {"at": 1551, "pos": 4},
        {"at": 1751, "pos": 100},
        {"at": 1951, "pos": 5},
        {"at": 2118, "pos": 95},
        {"at": 2268, "pos": 0},
        {"at": 2452, "pos": 90},
        {"at": 2619, "pos": 0},
        {"at": 2786, "pos": 98},
        {"at": 2952, "pos": 5},
        {"at": 3086, "pos": 94},
        {"at": 3236, "pos": 0},
        {"at": 3403, "pos": 100},
        {"at": 3553, "pos": 5},
        {"at": 3720, "pos": 99},
        {"at": 3853, "pos": 5},
        {"at": 4003, "pos": 100},
        {"at": 4187, "pos": 2},
        {"at": 4354, "pos": 99},
        {"at": 4521, "pos": 6},
        {"at": 4921, "pos": 95},
        {"at": 5155, "pos": 4},
        {"at": 5355, "pos": 54},
        {"at": 5555, "pos": 41},
        {"at": 5855, "pos": 98},
        {"at": 6089, "pos": 35},
        {"at": 6156, "pos": 35},
        {"at": 6356, "pos": 100},
        {"at": 6690, "pos": 0},
        {"at": 6990, "pos": 100},
        {"at": 7190, "pos": 0},
        {"at": 7440, "pos": 100},
        {"at": 7624, "pos": 0},
        {"at": 7824, "pos": 69},  # Should be removed
        {"at": 7874, "pos": 64},  # Should be removed
        {"at": 7991, "pos": 94},
        {"at": 8158, "pos": 8},
        {"at": 8458, "pos": 100},
        {"at": 8725, "pos": 1},
        {"at": 8975, "pos": 100},
        {"at": 9192, "pos": 0},
        {"at": 9392, "pos": 100},
        {"at": 9659, "pos": 0},
        {"at": 9926, "pos": 100},
        {"at": 10076, "pos": 0},
        {"at": 10393, "pos": 99},
        {"at": 10593, "pos": 26},
        {"at": 10727, "pos": 99},
        {"at": 10894, "pos": 96},
        {"at": 11094, "pos": 19},
        {"at": 11261, "pos": 92},
        {"at": 11494, "pos": 0},
        {"at": 11744, "pos": 100},
        {"at": 11995, "pos": 0}
    ]
    
    print("=== SIMPLIFIED ANTI-JERK PLUGIN TEST ===")
    print(f"Original actions: {len(example_actions)}")
    
    # Create mock funscript
    mock_funscript = MockDualAxisFunscript(example_actions)
    
    # Create and test the plugin
    plugin = AntiJerkPlugin()
    
    # Check plugin info
    print(f"Plugin name: {plugin.name}")
    print(f"Plugin version: {plugin.version}")
    print(f"Plugin description: {plugin.description}")
    print()
    
    # Check dependencies
    if not plugin.check_dependencies():
        print("❌ Dependencies not available")
        return
    
    print("✅ Dependencies available")
    print()
    
    # Test with default parameters
    print("Applying plugin with default parameters...")
    error = plugin.transform(mock_funscript, axis='primary')
    
    if error:
        print(f"❌ Plugin failed: {error}")
        return
    
    print("✅ Plugin applied successfully")
    print()
    
    # Check results
    print("=== RESULTS ===")
    print(f"Original points: {len(example_actions)}")
    print(f"Cleaned points:  {len(mock_funscript.primary_actions)}")
    print(f"Points removed:  {len(example_actions) - len(mock_funscript.primary_actions)}")
    print(f"Reduction:       {((len(example_actions) - len(mock_funscript.primary_actions)) / len(example_actions) * 100):.1f}%")
    
    # Check the specific sequence we care about
    print()
    print("=== VERIFICATION: Target sequence (7624→7824→7874→7991) ===")
    target_times = [7624, 7824, 7874, 7991]
    
    print("Original sequence:")
    for time in target_times:
        for action in example_actions:
            if action['at'] == time:
                print(f"  {time}ms: pos={action['pos']}")
                break
    
    print("\nCleaned sequence:")
    remaining_times = []
    for action in mock_funscript.primary_actions:
        if action['at'] in target_times:
            remaining_times.append(action['at'])
            print(f"  {action['at']}ms: pos={action['pos']}")
    
    # Verify the result
    if 7824 not in remaining_times and 7874 not in remaining_times:
        print("\n✅ SUCCESS: Jerky points at 7824ms and 7874ms were removed!")
    else:
        print("\n❌ ISSUE: Jerky points were not removed as expected")
        print(f"Remaining times: {remaining_times}")
    
    # Test with different parameters
    print("\n=== TESTING WITH CUSTOM PARAMETERS ===")
    
    # Reset the funscript
    mock_funscript2 = MockDualAxisFunscript(example_actions)
    
    # Apply with stricter settings
    error = plugin.transform(mock_funscript2, axis='primary', 
                           jerk_threshold=15.0,  # Stricter threshold
                           min_main_movement=40.0,  # Lower minimum movement
                           deviation_threshold=10.0)  # Stricter deviation
    
    if error:
        print(f"❌ Custom parameters failed: {error}")
    else:
        print("✅ Custom parameters applied successfully")
        print(f"Points with custom settings: {len(mock_funscript2.primary_actions)}")
        print(f"Additional reduction: {len(mock_funscript.primary_actions) - len(mock_funscript2.primary_actions)} points")
    
    # Save results for inspection
    cleaned_funscript = {
        "version": "1.0",
        "author": "Simplified Anti-Jerk Plugin v2.0.0",
        "inverted": False,
        "range": 100,
        "actions": mock_funscript.primary_actions
    }
    
    with open('simplified_anti_jerk_result.funscript', 'w') as f:
        json.dump(cleaned_funscript, f, indent=2)
    
    print(f"\nCleaned funscript saved as: simplified_anti_jerk_result.funscript")

if __name__ == "__main__":
    test_simplified_plugin()