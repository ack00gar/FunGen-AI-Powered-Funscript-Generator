#!/usr/bin/env python3
"""
Test the completed Anti-Jerk plugin with line-fitting outlier detection mode.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from funscript.dual_axis_funscript import DualAxisFunscript
from funscript.plugins.anti_jerk_plugin import AntiJerkPlugin

def test_line_fitting_outlier_mode():
    """Test the line-fitting outlier detection mode."""
    print("🧪 Testing Anti-Jerk Plugin with Line-Fitting Outlier Detection")
    print("=" * 70)
    
    # Create the problematic funscript data from the original conversation
    problematic_data = {
        "version": "1.0",
        "author": "FunGen beta 0.5.0",
        "actions": [
            {"at": 6100, "pos": 13},
            {"at": 6180, "pos": 86},   # +73 in 80ms - extreme jump
            {"at": 6280, "pos": 25},   # -61 in 100ms 
            {"at": 6420, "pos": 96},   # +71 in 140ms
            {"at": 6540, "pos": 1},    # -95 in 120ms - extreme jump
            {"at": 6620, "pos": 83},   # +82 in 80ms - extreme jump
            {"at": 6780, "pos": 5},    # -78 in 160ms
            {"at": 6900, "pos": 84},   # +79 in 120ms
            {"at": 7020, "pos": 12},   # -72 in 120ms
            {"at": 7140, "pos": 91},   # +79 in 120ms
            {"at": 7260, "pos": 8},    # -83 in 120ms
            {"at": 7380, "pos": 89},   # +81 in 120ms
            {"at": 7500, "pos": 6},    # -83 in 120ms
            {"at": 7620, "pos": 87},   # +81 in 120ms
            {"at": 7740, "pos": 4},    # -83 in 120ms
            {"at": 7860, "pos": 85},   # +81 in 120ms
            {"at": 7980, "pos": 2},    # -83 in 120ms
            {"at": 8100, "pos": 83},   # +81 in 120ms
            {"at": 8220, "pos": 0},    # -83 in 120ms
            {"at": 8340, "pos": 81},   # +81 in 120ms
            {"at": 8460, "pos": 0},    # -81 in 120ms
            {"at": 8580, "pos": 79},   # +79 in 120ms
            {"at": 8700, "pos": 2},    # -77 in 120ms
            {"at": 8820, "pos": 77},   # +75 in 120ms
            {"at": 8940, "pos": 4}     # -73 in 120ms
        ]
    }
    
    # Create DualAxisFunscript
    funscript = DualAxisFunscript()
    for action in problematic_data["actions"]:
        funscript.add_action(action["at"], action["pos"])
    
    print(f"📊 Original funscript: {len(funscript.actions)} actions")
    
    # Analyze original data characteristics
    times = np.array([action['at'] for action in funscript.actions])
    positions = np.array([action['pos'] for action in funscript.actions])
    movements = np.abs(np.diff(positions))
    extreme_jumps = np.sum(movements > 70)
    extreme_jumps_pct = extreme_jumps / len(movements) * 100
    
    print(f"📈 Original extreme jumps (>70): {extreme_jumps}/{len(movements)} ({extreme_jumps_pct:.1f}%)")
    print(f"📏 Position range: {positions.min()}-{positions.max()}")
    print(f"⏱️  Time range: {times[0]}-{times[-1]}ms")
    print()
    
    # Test different modes
    modes_to_test = [
        'auto',
        'line_fitting_outlier',
        'intermediate_insertion',
        'sparse'
    ]
    
    for mode in modes_to_test:
        print(f"🔬 Testing mode: {mode}")
        print("-" * 50)
        
        # Create fresh copy for each test
        test_funscript = DualAxisFunscript()
        for action in problematic_data["actions"]:
            test_funscript.add_action(action["at"], action["pos"])
        
        # Create and test plugin
        plugin = AntiJerkPlugin()
        
        # Use different parameters for each mode
        if mode == 'line_fitting_outlier':
            kwargs = {
                'mode': mode,
                'outlier_threshold': 20.0,
                'max_line_distance': 4,
                'outlier_removal_confidence': 60.0
            }
        elif mode == 'intermediate_insertion':
            kwargs = {
                'mode': mode,
                'insertion_jerk_threshold': 70.0,
                'insertion_time_threshold': 250.0,
                'transition_style': 'ease_in_out'
            }
        else:
            kwargs = {'mode': mode}
        
        # Apply plugin
        error = plugin.transform(test_funscript, axis='primary', **kwargs)
        
        if error:
            print(f"❌ Error: {error}")
            continue
        
        # Analyze results
        result_times = np.array([action['at'] for action in test_funscript.actions])
        result_positions = np.array([action['pos'] for action in test_funscript.actions])
        result_movements = np.abs(np.diff(result_positions))
        result_extreme_jumps = np.sum(result_movements > 70)
        result_extreme_jumps_pct = result_extreme_jumps / len(result_movements) * 100 if len(result_movements) > 0 else 0
        
        actions_changed = len(test_funscript.actions) - len(funscript.actions)
        extreme_jump_reduction = extreme_jumps - result_extreme_jumps
        reduction_pct = (extreme_jump_reduction / extreme_jumps * 100) if extreme_jumps > 0 else 0
        
        print(f"📊 Result: {len(test_funscript.actions)} actions ({actions_changed:+d})")
        print(f"📈 Extreme jumps: {result_extreme_jumps}/{len(result_movements)} ({result_extreme_jumps_pct:.1f}%)")
        print(f"🎯 Reduction: {extreme_jump_reduction} jumps ({reduction_pct:.1f}%)")
        print()
    
    print("✅ Anti-Jerk Plugin testing completed!")

if __name__ == "__main__":
    test_line_fitting_outlier_mode()