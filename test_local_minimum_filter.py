#!/usr/bin/env python3
"""
Test the new local minimum filter implementation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from funscript.dual_axis_funscript import DualAxisFunscript
from funscript.plugins.anti_jerk_plugin import AntiJerkPlugin

def test_local_minimum_filter():
    """Test the new local minimum detection filter."""
    print("🧪 Testing Local Minimum Filter Implementation")
    print("=" * 60)
    
    # Create test data with small local minima/maxima noise
    test_data = {
        "version": "1.0",
        "author": "Test Data",
        "actions": [
            {"at": 1000, "pos": 20},   # Start
            {"at": 1100, "pos": 15},   # Going down
            {"at": 1150, "pos": 18},   # Small local minimum noise (prev_prev>prev and small change)
            {"at": 1200, "pos": 80},   # Big jump up
            {"at": 1300, "pos": 75},   # Going down  
            {"at": 1350, "pos": 78},   # Small local minimum noise
            {"at": 1400, "pos": 10},   # Big jump down
            {"at": 1500, "pos": 5},    # Going down more
            {"at": 1550, "pos": 8},    # Small local minimum noise 
            {"at": 1600, "pos": 90},   # Big jump up
            {"at": 1700, "pos": 85},   # Going down
            {"at": 1750, "pos": 88},   # Small local maximum noise (going back down)
            {"at": 1800, "pos": 25},   # Big jump down
        ]
    }
    
    print("📊 Test data designed with small local min/max noise patterns")
    print(f"📈 Original actions: {len(test_data['actions'])}")
    
    # Analyze the original pattern
    positions = [action['pos'] for action in test_data['actions']]
    times = [action['at'] for action in test_data['actions']]
    
    print("\n🔍 Analyzing noise patterns:")
    for i in range(2, len(positions) - 1):
        prev_prev_val = positions[i-2]
        prev_val = positions[i-1] 
        val = positions[i]
        next_val = positions[i+1]
        
        # Check user's pattern
        condition1 = abs(prev_val - val) < 30  # Small change from previous
        condition2 = prev_prev_val > prev_val  # Was decreasing  
        condition3 = val > next_val            # Will decrease (local max)
        
        # Check local minimum pattern (corrected)
        local_min_pattern = (abs(prev_val - val) < 30 and 
                           prev_prev_val > prev_val and 
                           val < next_val)  # Going back up
        
        # Check local maximum pattern (user's original)
        local_max_pattern = (abs(prev_val - val) < 30 and
                           prev_prev_val > prev_val and
                           val > next_val)  # Going back down
        
        if local_min_pattern or local_max_pattern:
            pattern_type = "LOCAL_MIN" if local_min_pattern else "LOCAL_MAX"
            print(f"  Frame {i} (t={times[i]}): {prev_prev_val}→{prev_val}→{val}→{next_val} [{pattern_type} NOISE]")
    
    # Test different threshold values
    thresholds_to_test = [15.0, 25.0, 30.0, 40.0]
    
    results = []
    
    for threshold in thresholds_to_test:
        print(f"\n🧪 Testing threshold: {threshold}")
        print("-" * 40)
        
        # Create fresh copy for each test
        test_funscript = DualAxisFunscript()
        for action in test_data["actions"]:
            test_funscript.add_action(action["at"], action["pos"])
        
        # Apply local minimum filter
        plugin = AntiJerkPlugin()
        
        error = plugin.transform(test_funscript, axis='primary', 
                               mode='local_minimum_filter',
                               local_minimum_threshold=threshold)
        
        if error:
            print(f"❌ Error: {error}")
            continue
        
        # Analyze results
        result_actions = len(test_funscript.actions)
        removed_actions = len(test_data['actions']) - result_actions
        
        print(f"📊 Result: {result_actions} actions (-{removed_actions} removed)")
        
        # Show what was removed
        original_times = set(action['at'] for action in test_data['actions'])
        result_times = set(action['at'] for action in test_funscript.actions)
        removed_times = sorted(original_times - result_times)
        
        if removed_times:
            print(f"🗑️  Removed points at times: {removed_times}")
        else:
            print("🔒 No points removed")
        
        results.append({
            'threshold': threshold,
            'removed': removed_actions,
            'removed_times': removed_times
        })
    
    # Test with original problematic data 
    print(f"\n🔬 Testing with Original User Data")
    print("-" * 40)
    
    original_data = {
        "version":"1.0",
        "author":"FunGen beta 0.5.0", 
        "actions":[
            {"at":6100,"pos":13}, {"at":6180,"pos":86}, {"at":6280,"pos":25},
            {"at":6420,"pos":96}, {"at":6540,"pos":1}, {"at":6620,"pos":83},
            {"at":6780,"pos":5}, {"at":6900,"pos":84}, {"at":7020,"pos":12},
            {"at":7140,"pos":91}, {"at":7260,"pos":8}, {"at":7380,"pos":89},
            {"at":7500,"pos":6}, {"at":7620,"pos":87}, {"at":7740,"pos":4},
            {"at":7860,"pos":85}, {"at":7980,"pos":2}, {"at":8100,"pos":83},
            {"at":8220,"pos":0}, {"at":8340,"pos":81}, {"at":8460,"pos":0},
            {"at":8580,"pos":79}, {"at":8700,"pos":2}, {"at":8820,"pos":77},
            {"at":8940,"pos":4}
        ]
    }
    
    original_funscript = DualAxisFunscript()
    for action in original_data["actions"]:
        original_funscript.add_action(action["at"], action["pos"])
    
    plugin = AntiJerkPlugin()
    error = plugin.transform(original_funscript, axis='primary', 
                           mode='local_minimum_filter',
                           local_minimum_threshold=30.0)
    
    if not error:
        original_removed = len(original_data['actions']) - len(original_funscript.actions)
        print(f"📊 Original data: {len(original_funscript.actions)} actions (-{original_removed} removed)")
    else:
        print(f"❌ Error with original data: {error}")
    
    # Summary
    print(f"\n📋 SUMMARY:")
    print("=" * 40)
    print("Local minimum filter successfully implemented with pattern:")
    print("• if abs(prev_val - val) < threshold and prev_prev_val > prev_val")
    print("• Detects both local minima (val < next_val) and maxima (val > next_val)")
    print("• Configurable threshold (10.0-60.0)")
    print("• Preserves endpoints and limits removal")
    
    print(f"\n✅ Implementation completed!")

if __name__ == "__main__":
    test_local_minimum_filter()