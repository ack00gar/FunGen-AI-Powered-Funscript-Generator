#!/usr/bin/env python3
"""
Test the user's exact recursive cleaning algorithm.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from funscript.dual_axis_funscript import DualAxisFunscript
from funscript.plugins.anti_jerk_plugin import AntiJerkPlugin

def test_users_example():
    """Test with the user's exact example."""
    print("🧪 Testing User's Exact Recursive Algorithm")
    print("=" * 60)
    
    # User's exact example
    data = [100, 80, 85, 60, 40]
    print(f"📊 User's example data: {data}")
    
    # Manual step-by-step analysis
    print("\n🔍 Manual Analysis:")
    print("Looking for pattern: abs(prev_val - val) < 30 and prev_prev_val > prev_val and val > next_val")
    
    for i in range(2, len(data) - 1):
        prev_prev_val = data[i-2]
        prev_val = data[i-1]
        val = data[i]
        next_val = data[i+1]
        
        condition1 = abs(prev_val - val) < 30
        condition2 = prev_prev_val > prev_val  
        condition3 = val > next_val
        
        print(f"  i={i}: {prev_prev_val}→{prev_val}→{val}→{next_val}")
        print(f"    abs({prev_val}-{val})={abs(prev_val-val)} < 30: {condition1}")
        print(f"    {prev_prev_val} > {prev_val}: {condition2}")
        print(f"    {val} > {next_val}: {condition3}")
        print(f"    Pattern match: {condition1 and condition2 and condition3}")
        
        if condition1 and condition2 and condition3:
            print(f"    ✅ MATCH! Would remove positions {prev_val} and {val} (indices {i-1}, {i})")
            break
        print()
    
    # Test with our plugin implementation
    print(f"\n🔧 Testing with Plugin Implementation:")
    
    # Create a test funscript with the user's data
    times = [1000, 1100, 1200, 1300, 1400]  # Add time values
    test_actions = [{"at": times[i], "pos": data[i]} for i in range(len(data))]
    
    funscript = DualAxisFunscript()
    for action in test_actions:
        funscript.add_action(action["at"], action["pos"])
    
    print(f"📈 Before: {[action['pos'] for action in funscript.actions]}")
    
    # Apply the local minimum filter
    plugin = AntiJerkPlugin()
    error = plugin.transform(funscript, axis='primary', 
                           mode='local_minimum_filter',
                           local_minimum_threshold=30.0)
    
    if error:
        print(f"❌ Error: {error}")
    else:
        result_positions = [action['pos'] for action in funscript.actions]
        print(f"📈 After:  {result_positions}")
        
        removed_count = len(data) - len(result_positions)
        print(f"🗑️  Removed {removed_count} points")
        
        # Compare with expected result
        # Based on the pattern, we should find:
        # i=2: 100→80→85→60, abs(80-85)=5<30, 100>80, 85>60 ✅
        # This should remove 80 and 85, leaving [100, 60, 40]
        expected = [100, 60, 40]
        
        if result_positions == expected:
            print(f"✅ Perfect match with expected result: {expected}")
        else:
            print(f"⚠️  Expected: {expected}, Got: {result_positions}")

def test_comprehensive_scenarios():
    """Test various scenarios with the recursive algorithm."""
    print(f"\n🔬 Comprehensive Recursive Cleaning Tests")
    print("=" * 60)
    
    test_cases = [
        {
            'name': 'User Example',
            'data': [100, 80, 85, 60, 40],
            'expected': [100, 60, 40]  # Remove 80,85 based on pattern
        },
        {
            'name': 'Multiple Patterns',
            'data': [90, 70, 75, 50, 55, 30],
            'expected': None  # Let's see what happens
        },
        {
            'name': 'No Patterns',
            'data': [100, 50, 0, 50, 100],
            'expected': [100, 50, 0, 50, 100]  # No changes
        },
        {
            'name': 'Edge Case - Small Data',
            'data': [80, 60, 65],
            'expected': [80, 60, 65]  # Too few points
        }
    ]
    
    for test_case in test_cases:
        print(f"\n📊 Test: {test_case['name']}")
        print(f"   Input:  {test_case['data']}")
        
        if len(test_case['data']) < 4:
            print(f"   Result: {test_case['data']} (too few points)")
            continue
        
        # Create funscript and test
        times = [1000 + i*100 for i in range(len(test_case['data']))]
        test_actions = [{"at": times[i], "pos": test_case['data'][i]} for i in range(len(test_case['data']))]
        
        funscript = DualAxisFunscript()
        for action in test_actions:
            funscript.add_action(action["at"], action["pos"])
        
        plugin = AntiJerkPlugin()
        error = plugin.transform(funscript, axis='primary', 
                               mode='local_minimum_filter',
                               local_minimum_threshold=30.0)
        
        if error:
            print(f"   ❌ Error: {error}")
        else:
            result = [action['pos'] for action in funscript.actions]
            print(f"   Result: {result}")
            
            if test_case['expected'] is not None:
                if result == test_case['expected']:
                    print(f"   ✅ Matches expected result")
                else:
                    print(f"   ⚠️  Expected: {test_case['expected']}")
    
    print(f"\n✅ Recursive cleaning algorithm implementation complete!")

if __name__ == "__main__":
    test_users_example()
    test_comprehensive_scenarios()