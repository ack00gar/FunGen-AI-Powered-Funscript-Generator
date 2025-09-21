#!/usr/bin/env python3
"""
Detailed analysis of why the Discord algorithm didn't trigger on the example funscript.
Let's examine each potential match condition.
"""

import json

def analyze_discord_pattern_matching():
    """Analyze why the Discord algorithm didn't trigger"""
    
    # Example funscript actions
    actions = [
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
    ]
    
    print("=== DISCORD ALGORITHM PATTERN ANALYSIS ===")
    print("Condition: abs(prev_val - val) < 30 AND prev_prev_val > prev_val AND val > next_val")
    print()
    
    matches_found = 0
    near_matches = 0
    
    print("Analyzing each 4-point window:")
    print("Index | prev_prev | prev | val | next | diff<30 | pp>p | v>n | Match")
    print("-" * 75)
    
    for i in range(2, len(actions) - 1):
        prev_prev_val = actions[i-2]['pos']
        prev_val = actions[i-1]['pos']
        val = actions[i]['pos']
        next_val = actions[i+1]['pos']
        
        # Check each condition
        diff_condition = abs(prev_val - val) < 30
        trend_condition = prev_prev_val > prev_val
        reversal_condition = val > next_val
        
        all_match = diff_condition and trend_condition and reversal_condition
        
        if all_match:
            matches_found += 1
            match_str = "YES"
        elif diff_condition or trend_condition or reversal_condition:
            near_matches += 1
            match_str = "partial"
        else:
            match_str = "no"
        
        print(f"{i:5d} | {prev_prev_val:9d} | {prev_val:4d} | {val:3d} | {next_val:4d} | "
              f"{str(diff_condition):7s} | {str(trend_condition):5s} | {str(reversal_condition):5s} | {match_str}")
    
    print()
    print(f"Total exact matches found: {matches_found}")
    print(f"Partial matches (met some conditions): {near_matches}")
    
    print()
    print("=== CONDITION BREAKDOWN ===")
    
    # Analyze each condition separately
    small_diffs = 0
    downward_trends = 0
    reversals = 0
    
    for i in range(2, len(actions) - 1):
        prev_prev_val = actions[i-2]['pos']
        prev_val = actions[i-1]['pos']
        val = actions[i]['pos']
        next_val = actions[i+1]['pos']
        
        if abs(prev_val - val) < 30:
            small_diffs += 1
        if prev_prev_val > prev_val:
            downward_trends += 1
        if val > next_val:
            reversals += 1
    
    total_windows = len(actions) - 3
    print(f"Windows with small differences (< 30): {small_diffs}/{total_windows} ({small_diffs/total_windows*100:.1f}%)")
    print(f"Windows with downward trends: {downward_trends}/{total_windows} ({downward_trends/total_windows*100:.1f}%)")
    print(f"Windows with reversals: {reversals}/{total_windows} ({reversals/total_windows*100:.1f}%)")
    
    print()
    print("=== ALGORITHM ASSESSMENT ===")
    if matches_found == 0:
        print("❌ No matches found - algorithm ineffective on this data")
        print("   Reasons:")
        print("   - Your funscript has mostly large movements (>30 position difference)")
        print("   - The pattern 'prev_prev > prev AND val > next' is very specific")
        print("   - This algorithm targets micro-oscillations, not large movements")
    else:
        print(f"✅ Found {matches_found} potential cleaning targets")
    
    # Let's try a modified version with looser constraints
    print()
    print("=== TESTING MODIFIED ALGORITHM ===")
    print("Modified condition: abs(prev_val - val) < 50 AND prev_prev_val > prev_val AND val > next_val")
    
    modified_matches = 0
    for i in range(2, len(actions) - 1):
        prev_prev_val = actions[i-2]['pos']
        prev_val = actions[i-1]['pos']
        val = actions[i]['pos']
        next_val = actions[i+1]['pos']
        
        if abs(prev_val - val) < 50 and prev_prev_val > prev_val and val > next_val:
            modified_matches += 1
            print(f"  Match at index {i}: {prev_prev_val} > {prev_val} -> {val} > {next_val} (diff: {abs(prev_val - val)})")
    
    print(f"Modified algorithm matches: {modified_matches}")

if __name__ == "__main__":
    analyze_discord_pattern_matching()