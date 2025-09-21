#!/usr/bin/env python3
"""
Test the recursive cleaning on the original problematic funscript data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from funscript.dual_axis_funscript import DualAxisFunscript
from funscript.plugins.anti_jerk_plugin import AntiJerkPlugin

def test_recursive_on_original():
    """Test recursive cleaning on the original problematic data."""
    print("🧪 Testing Recursive Cleaning on Original Problematic Data")
    print("=" * 70)
    
    # Original problematic data from the user
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
    
    positions = [action['pos'] for action in original_data['actions']]
    times = [action['at'] for action in original_data['actions']]
    
    print(f"📊 Original: {len(positions)} actions")
    print(f"📈 Positions: {positions}")
    
    # Analyze original extreme jumps
    movements = np.abs(np.diff(positions))
    extreme_jumps = np.sum(movements > 70)
    extreme_pct = extreme_jumps / len(movements) * 100
    print(f"📈 Extreme jumps (>70): {extreme_jumps}/{len(movements)} ({extreme_pct:.1f}%)")
    
    # Test different thresholds
    thresholds = [30, 40, 50, 60]
    
    for threshold in thresholds:
        print(f"\n🔧 Testing threshold: {threshold}")
        print("-" * 50)
        
        # Create fresh funscript
        test_funscript = DualAxisFunscript()
        for action in original_data['actions']:
            test_funscript.add_action(action['at'], action['pos'])
        
        # Apply recursive cleaning
        plugin = AntiJerkPlugin()
        error = plugin.transform(test_funscript, axis='primary', 
                               mode='local_minimum_filter',
                               local_minimum_threshold=threshold)
        
        if error:
            print(f"❌ Error: {error}")
            continue
        
        # Analyze results
        result_positions = [action['pos'] for action in test_funscript.actions]
        result_times = [action['at'] for action in test_funscript.actions]
        
        removed_count = len(positions) - len(result_positions)
        print(f"📊 Result: {len(result_positions)} actions (-{removed_count} removed)")
        print(f"📈 Positions: {result_positions}")
        
        if len(result_positions) > 1:
            result_movements = np.abs(np.diff(result_positions))
            result_extreme = np.sum(result_movements > 70)
            result_extreme_pct = result_extreme / len(result_movements) * 100 if len(result_movements) > 0 else 0
            improvement = extreme_jumps - result_extreme
            improvement_pct = improvement / extreme_jumps * 100 if extreme_jumps > 0 else 0
            
            print(f"📈 Extreme jumps: {result_extreme}/{len(result_movements)} ({result_extreme_pct:.1f}%)")
            print(f"🎯 Improvement: -{improvement} jumps ({improvement_pct:.1f}% reduction)")
        
        # Show time gaps for very aggressive removal
        if removed_count > 15:
            print(f"⚠️  Very aggressive removal - check time gaps:")
            if len(result_times) > 1:
                time_gaps = np.diff(result_times)
                max_gap = max(time_gaps)
                avg_gap = np.mean(time_gaps)
                print(f"   Max time gap: {max_gap}ms, Avg gap: {avg_gap:.1f}ms")
    
    # Compare with other methods
    print(f"\n📊 COMPARISON WITH OTHER METHODS:")
    print("=" * 50)
    
    methods = [
        ('Line-Fitting Outlier', 'line_fitting_outlier', {}),
        ('Recursive Cleaning (30)', 'local_minimum_filter', {'local_minimum_threshold': 30.0}),
        ('Recursive Cleaning (50)', 'local_minimum_filter', {'local_minimum_threshold': 50.0})
    ]
    
    for method_name, mode, params in methods:
        test_funscript = DualAxisFunscript()
        for action in original_data['actions']:
            test_funscript.add_action(action['at'], action['pos'])
        
        plugin = AntiJerkPlugin()
        error = plugin.transform(test_funscript, axis='primary', mode=mode, **params)
        
        if not error:
            result_count = len(test_funscript.actions)
            removed = len(positions) - result_count
            removal_pct = removed / len(positions) * 100
            print(f"{method_name:<25}: {result_count:2d} actions (-{removed:2d}, {removal_pct:4.1f}%)")
        else:
            print(f"{method_name:<25}: Error - {error}")
    
    print(f"\n✅ Recursive cleaning analysis complete!")

if __name__ == "__main__":
    test_recursive_on_original()