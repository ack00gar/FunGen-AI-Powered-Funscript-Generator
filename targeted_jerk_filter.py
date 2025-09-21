#!/usr/bin/env python3
"""
Targeted jerk filter for the specific pattern user identified.
Removes intermediate points that create jerkiness in main movements.
"""

import json
import copy

def remove_intermediate_jerks(actions, jerk_threshold=20, min_main_movement=50):
    """
    Remove intermediate points that create jerkiness.
    
    Pattern: A -> B -> C -> D where:
    - A to D is a significant movement (main motion)
    - B and C create small oscillations that interrupt the main flow
    - Remove B and C to get smooth A -> D transition
    
    Args:
        actions: List of funscript actions
        jerk_threshold: Maximum oscillation size to consider as "jerk"
        min_main_movement: Minimum A->D movement to consider processing
    """
    if len(actions) < 4:
        return actions
    
    cleaned_actions = []
    i = 0
    
    while i < len(actions):
        # Always keep the current point
        cleaned_actions.append(actions[i])
        
        # Look for 4-point jerkiness pattern: current -> next1 -> next2 -> next3
        if i <= len(actions) - 4:
            current = actions[i]
            next1 = actions[i + 1]
            next2 = actions[i + 2] 
            next3 = actions[i + 3]
            
            # Calculate movements
            main_movement = abs(next3['pos'] - current['pos'])
            intermediate_osc = abs(next2['pos'] - next1['pos'])
            
            # Check if this fits the jerkiness pattern
            if (main_movement >= min_main_movement and 
                intermediate_osc <= jerk_threshold):
                
                # Additional check: are the intermediates creating deviation from direct path?
                # Calculate expected positions if moving directly from current to next3
                time_span = next3['at'] - current['at']
                if time_span > 0:
                    position_span = next3['pos'] - current['pos']
                    
                    # Expected positions at intermediate times
                    t1_ratio = (next1['at'] - current['at']) / time_span
                    t2_ratio = (next2['at'] - current['at']) / time_span
                    
                    expected_pos1 = current['pos'] + position_span * t1_ratio
                    expected_pos2 = current['pos'] + position_span * t2_ratio
                    
                    deviation1 = abs(next1['pos'] - expected_pos1)
                    deviation2 = abs(next2['pos'] - expected_pos2)
                    
                    # If intermediates deviate significantly from direct path, remove them
                    if deviation1 > 15 or deviation2 > 15:
                        print(f"Removing jerky intermediates:")
                        print(f"  Keep: {current['at']}ms pos={current['pos']}")
                        print(f"  Remove: {next1['at']}ms pos={next1['pos']} (deviation: {deviation1:.1f})")
                        print(f"  Remove: {next2['at']}ms pos={next2['pos']} (deviation: {deviation2:.1f})")
                        print(f"  Keep: {next3['at']}ms pos={next3['pos']}")
                        print(f"  Main movement: {current['pos']} -> {next3['pos']} ({main_movement})")
                        print()
                        
                        # Skip the intermediate points
                        i += 3  # Will add next3 in next iteration
                        continue
        
        i += 1
    
    return cleaned_actions

def test_targeted_filter():
    """Test the targeted jerk filter"""
    
    # Full funscript
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
    }
    
    print("=== TARGETED JERK FILTER TEST ===")
    print(f"Original funscript: {len(example_funscript['actions'])} points")
    print()
    
    # Apply the filter
    cleaned_funscript = copy.deepcopy(example_funscript)
    cleaned_actions = remove_intermediate_jerks(cleaned_funscript['actions'])
    cleaned_funscript['actions'] = cleaned_actions
    
    print("=== RESULTS ===")
    print(f"Original points: {len(example_funscript['actions'])}")
    print(f"Cleaned points:  {len(cleaned_actions)}")
    print(f"Points removed:  {len(example_funscript['actions']) - len(cleaned_actions)}")
    print(f"Reduction:       {((len(example_funscript['actions']) - len(cleaned_actions)) / len(example_funscript['actions']) * 100):.1f}%")
    
    # Check if the specific sequence was handled correctly
    print("\n=== VERIFICATION: User's specific sequence ===")
    target_times = [7624, 7824, 7874, 7991]
    
    print("Original sequence:")
    for time in target_times:
        for action in example_funscript['actions']:
            if action['at'] == time:
                print(f"  {time}ms: pos={action['pos']}")
                break
    
    print("\nCleaned sequence:")
    remaining_times = []
    for action in cleaned_actions:
        if action['at'] in target_times:
            remaining_times.append(action['at'])
            print(f"  {action['at']}ms: pos={action['pos']}")
    
    # Verify the result
    if 7824 not in remaining_times and 7874 not in remaining_times:
        print("\n✅ SUCCESS: Jerky points at 7824ms and 7874ms were removed!")
    else:
        print("\n❌ ISSUE: Jerky points were not removed as expected")
    
    # Save the result
    with open('targeted_jerk_cleaned.funscript', 'w') as f:
        json.dump(cleaned_funscript, f, indent=2)
    
    print(f"\nCleaned funscript saved as: targeted_jerk_cleaned.funscript")

if __name__ == "__main__":
    test_targeted_filter()