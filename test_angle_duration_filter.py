#!/usr/bin/env python3
"""
Test the new angle/duration outlier detection filter.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from funscript.dual_axis_funscript import DualAxisFunscript
from funscript.plugins.anti_jerk_plugin import AntiJerkPlugin

def test_angle_duration_filter():
    """Test the angle/duration outlier detection filter."""
    print("🧪 Testing Angle/Duration Outlier Detection Filter")
    print("=" * 65)
    
    # Create test data with angular/temporal outliers
    test_cases = [
        {
            'name': 'Angular Outlier - Sudden Direction Change',
            'data': [
                {"at": 1000, "pos": 20},   # Start
                {"at": 1100, "pos": 30},   # Moving up consistently  
                {"at": 1200, "pos": 40},   # Moving up consistently
                {"at": 1250, "pos": 90},   # OUTLIER: Sudden huge jump up (wrong direction)
                {"at": 1300, "pos": 50},   # Back to expected trajectory
                {"at": 1400, "pos": 60},   # Continuing expected trajectory
                {"at": 1500, "pos": 70}    # End
            ],
            'description': 'Point at t=1250 has extreme angular deviation from expected path'
        },
        {
            'name': 'Duration Outlier - Timing Anomaly',
            'data': [
                {"at": 2000, "pos": 50},   # Start
                {"at": 2100, "pos": 60},   # Normal timing (100ms)
                {"at": 2200, "pos": 70},   # Normal timing (100ms)
                {"at": 2220, "pos": 75},   # OUTLIER: Too close in time (20ms)
                {"at": 2300, "pos": 80},   # Back to normal timing
                {"at": 2400, "pos": 90}    # Normal timing
            ],
            'description': 'Point at t=2220 has timing anomaly (too close to previous point)'
        },
        {
            'name': 'Combined Outlier - Both Angle and Duration',
            'data': [
                {"at": 3000, "pos": 30},   # Start
                {"at": 3100, "pos": 40},   # Normal progression
                {"at": 3200, "pos": 50},   # Normal progression  
                {"at": 3210, "pos": 10},   # OUTLIER: Wrong direction + wrong timing
                {"at": 3300, "pos": 60},   # Back to expected
                {"at": 3400, "pos": 70}    # Normal end
            ],
            'description': 'Point at t=3210 deviates in both angle and timing'
        },
        {
            'name': 'Original Problematic Data (Subset)',
            'data': [
                {"at": 6100, "pos": 13},
                {"at": 6180, "pos": 86},   # +73 in 80ms
                {"at": 6280, "pos": 25},   # -61 in 100ms
                {"at": 6420, "pos": 96},   # +71 in 140ms
                {"at": 6540, "pos": 1},    # -95 in 120ms
                {"at": 6620, "pos": 83},   # +82 in 80ms
                {"at": 6780, "pos": 5}     # -78 in 160ms
            ],
            'description': 'Real problematic data with extreme oscillations'
        }
    ]
    
    # Test different parameter configurations
    configs = [
        {
            'name': 'Sensitive (Small Window)',
            'window_size': 3,
            'angle_threshold': 30.0,
            'duration_weight': 1.0
        },
        {
            'name': 'Balanced (Default)',
            'window_size': 5,
            'angle_threshold': 45.0, 
            'duration_weight': 1.0
        },
        {
            'name': 'Conservative (Large Window)',
            'window_size': 7,
            'angle_threshold': 60.0,
            'duration_weight': 0.5
        }
    ]
    
    for test_case in test_cases:
        print(f"\n📊 TEST CASE: {test_case['name']}")
        print("=" * 65)
        print(f"📝 {test_case['description']}")
        
        original_positions = [action['pos'] for action in test_case['data']]
        original_times = [action['at'] for action in test_case['data']]
        
        print(f"📈 Original: {len(original_positions)} points")
        print(f"    Times: {original_times}")
        print(f"    Positions: {original_positions}")
        
        # Test with different configurations
        for config in configs:
            print(f"\n🔧 Config: {config['name']}")
            print(f"   Window: {config['window_size']}, Angle: {config['angle_threshold']}°, Duration: {config['duration_weight']}")
            
            # Create fresh funscript
            test_funscript = DualAxisFunscript()
            for action in test_case['data']:
                test_funscript.add_action(action['at'], action['pos'])
            
            # Apply angle/duration filter
            plugin = AntiJerkPlugin()
            error = plugin.transform(test_funscript, axis='primary',
                                   mode='angle_duration_filter',
                                   angle_window_size=config['window_size'],
                                   angle_deviation_threshold=config['angle_threshold'],
                                   duration_weight=config['duration_weight'])
            
            if error:
                print(f"   ❌ Error: {error}")
                continue
            
            # Analyze results
            result_positions = [action['pos'] for action in test_funscript.actions]
            result_times = [action['at'] for action in test_funscript.actions]
            
            removed_count = len(original_positions) - len(result_positions)
            print(f"   📊 Result: {len(result_positions)} points (-{removed_count} removed)")
            
            if removed_count > 0:
                # Show what was removed
                original_time_set = set(original_times)
                result_time_set = set(result_times)
                removed_times = sorted(original_time_set - result_time_set)
                print(f"   🗑️  Removed times: {removed_times}")
                
                # Show the filtered data
                print(f"   📈 Filtered positions: {result_positions}")
            else:
                print(f"   🔒 No outliers detected")
    
    # Comprehensive comparison with other methods
    print(f"\n📊 COMPREHENSIVE COMPARISON")
    print("=" * 65)
    
    # Use the original problematic data
    original_data = [
        {"at": 6100, "pos": 13}, {"at": 6180, "pos": 86}, {"at": 6280, "pos": 25},
        {"at": 6420, "pos": 96}, {"at": 6540, "pos": 1}, {"at": 6620, "pos": 83},
        {"at": 6780, "pos": 5}, {"at": 6900, "pos": 84}, {"at": 7020, "pos": 12},
        {"at": 7140, "pos": 91}, {"at": 7260, "pos": 8}, {"at": 7380, "pos": 89},
        {"at": 7500, "pos": 6}, {"at": 7620, "pos": 87}, {"at": 7740, "pos": 4},
        {"at": 7860, "pos": 85}, {"at": 7980, "pos": 2}, {"at": 8100, "pos": 83},
        {"at": 8220, "pos": 0}, {"at": 8340, "pos": 81}, {"at": 8460, "pos": 0},
        {"at": 8580, "pos": 79}, {"at": 8700, "pos": 2}, {"at": 8820, "pos": 77},
        {"at": 8940, "pos": 4}
    ]
    
    methods_to_compare = [
        ('Original Data', None, {}),
        ('Angle/Duration (Sensitive)', 'angle_duration_filter', {'angle_window_size': 3, 'angle_deviation_threshold': 30.0}),
        ('Angle/Duration (Balanced)', 'angle_duration_filter', {'angle_window_size': 5, 'angle_deviation_threshold': 45.0}),
        ('Angle/Duration (Conservative)', 'angle_duration_filter', {'angle_window_size': 7, 'angle_deviation_threshold': 60.0}),
        ('Line-Fitting Outlier', 'line_fitting_outlier', {}),
        ('Local Minimum Filter', 'local_minimum_filter', {}),
        ('Auto Detection', 'auto', {})
    ]
    
    print(f"{'Method':<25} {'Points':<8} {'Removed':<8} {'%Removed':<10}")
    print("-" * 65)
    
    original_count = len(original_data)
    
    for method_name, mode, params in methods_to_compare:
        if mode is None:
            # Original data
            result_count = original_count
            removed = 0
        else:
            # Apply filter
            test_funscript = DualAxisFunscript()
            for action in original_data:
                test_funscript.add_action(action['at'], action['pos'])
            
            plugin = AntiJerkPlugin()
            error = plugin.transform(test_funscript, axis='primary', mode=mode, **params)
            
            if error:
                print(f"{method_name:<25} {'ERROR':<8} {'':<8} {'':<10}")
                continue
            
            result_count = len(test_funscript.actions)
            removed = original_count - result_count
        
        removal_pct = (removed / original_count * 100) if original_count > 0 else 0
        print(f"{method_name:<25} {result_count:<8} {removed:<8} {removal_pct:<10.1f}")
    
    print(f"\n✅ Angle/Duration filter testing complete!")
    print("\n💡 KEY INSIGHTS:")
    print("• Angle/Duration analysis provides a new perspective on outlier detection")
    print("• Combines directional movement analysis with timing considerations")
    print("• Configurable window size allows for local vs global trajectory analysis")
    print("• Duration weighting balances spatial vs temporal outlier importance")

if __name__ == "__main__":
    test_angle_duration_filter()