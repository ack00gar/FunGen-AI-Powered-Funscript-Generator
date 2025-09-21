#!/usr/bin/env python3
"""
Test the Discord user's recursive cleaning algorithm on the provided example funscript.
This algorithm removes points based on specific oscillation patterns.
"""

import json
import copy

def clean_points(points):
    """
    Recursively removes points where:
    abs(prev_val - val) < 30 and prev_prev_val > prev_val and val > next_val
    
    This targets specific oscillation patterns where there's a small movement
    followed by a reversal pattern.
    """
    print(f"Processing {len(points)} points...")
    
    for i in range(2, len(points) - 1):
        prev_prev_val = points[i-2]['pos']
        prev_val = points[i-1]['pos']
        val = points[i]['pos']
        next_val = points[i+1]['pos']

        if abs(prev_val - val) < 30 and prev_prev_val > prev_val and val > next_val:
            # Remove val and prev_val
            print(f"  Removing points at indices {i-1} and {i}: "
                  f"prev_prev={prev_prev_val}, prev={prev_val}, val={val}, next={next_val}")
            new_points = points[:i-1] + points[i+1:]
            # Recursively process the shortened list
            return clean_points(new_points)

    # No matches found; return the cleaned list
    print(f"  No more matches found. Final count: {len(points)} points")
    return points

def test_discord_algorithm():
    """Test the Discord recursive cleaning algorithm"""
    
    # Example funscript from the user
    example_funscript = {
        "version": "1.0",
        "author": "FunGen beta 0.5.0",
        "inverted": False,
        "range": 100,
        "actions": [
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
            {"at": 7824, "pos": 69},
            {"at": 7874, "pos": 64},
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
        ],
        "metadata": {
            "version": "0.2.0",
            "chapters": [{"name": "BJ", "startTime": "00:00:00.000", "endTime": "00:00:12.012"}],
            "chapters_fps": 59.94005994005994
        }
    }
    
    print("=== DISCORD RECURSIVE CLEANING ALGORITHM TEST ===")
    print(f"Original funscript: {len(example_funscript['actions'])} points")
    print()
    
    # Make a copy for processing
    cleaned_funscript = copy.deepcopy(example_funscript)
    
    # Apply the recursive cleaning algorithm
    print("Applying recursive cleaning algorithm...")
    cleaned_actions = clean_points(cleaned_funscript['actions'])
    cleaned_funscript['actions'] = cleaned_actions
    
    print()
    print("=== RESULTS ===")
    print(f"Original points: {len(example_funscript['actions'])}")
    print(f"Cleaned points:  {len(cleaned_actions)}")
    print(f"Points removed:  {len(example_funscript['actions']) - len(cleaned_actions)}")
    print(f"Reduction:       {((len(example_funscript['actions']) - len(cleaned_actions)) / len(example_funscript['actions']) * 100):.1f}%")
    
    print()
    print("=== DETAILED ANALYSIS ===")
    
    # Show the first few original vs cleaned actions for comparison
    print("First 10 original actions:")
    for i, action in enumerate(example_funscript['actions'][:10]):
        print(f"  {i+1:2d}: at={action['at']:5d}, pos={action['pos']:3d}")
    
    print()
    print("First 10 cleaned actions:")
    for i, action in enumerate(cleaned_actions[:10]):
        print(f"  {i+1:2d}: at={action['at']:5d}, pos={action['pos']:3d}")
    
    # Save the cleaned version
    with open('discord_cleaned_example.funscript', 'w') as f:
        json.dump(cleaned_funscript, f, indent=2)
    
    print()
    print("Cleaned funscript saved as: discord_cleaned_example.funscript")
    
    # Analysis of movement patterns
    print()
    print("=== MOVEMENT PATTERN ANALYSIS ===")
    
    def analyze_patterns(actions, label):
        print(f"\n{label} analysis:")
        positions = [action['pos'] for action in actions]
        
        # Calculate position differences
        diffs = []
        for i in range(1, len(positions)):
            diff = abs(positions[i] - positions[i-1])
            diffs.append(diff)
        
        if diffs:
            avg_diff = sum(diffs) / len(diffs)
            max_diff = max(diffs)
            min_diff = min(diffs)
            
            # Count small movements (< 30 as per algorithm threshold)
            small_movements = sum(1 for d in diffs if d < 30)
            
            print(f"  Average position change: {avg_diff:.1f}")
            print(f"  Max position change: {max_diff}")
            print(f"  Min position change: {min_diff}")
            print(f"  Small movements (< 30): {small_movements} / {len(diffs)} ({small_movements/len(diffs)*100:.1f}%)")
    
    analyze_patterns(example_funscript['actions'], "Original")
    analyze_patterns(cleaned_actions, "Cleaned")
    
    return cleaned_funscript

if __name__ == "__main__":
    test_discord_algorithm()