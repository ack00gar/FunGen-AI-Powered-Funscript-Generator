#!/usr/bin/env python3
"""
Smart jerkiness removal algorithm based on user's specific requirements.
Targets intermediate points that create jerkiness in the main motion flow.
"""

import json
import copy

def remove_jerky_intermediates(actions, min_main_movement=60, max_intermediate_deviation=30):
    """
    Remove jerky intermediate points that interrupt main movements.
    
    The algorithm identifies sequences where:
    1. There's a significant main movement (> min_main_movement)
    2. Intermediate points deviate from the direct path by < max_intermediate_deviation
    3. These intermediate points create jerkiness rather than smooth motion
    
    Args:
        actions: List of funscript actions [{"at": timestamp, "pos": position}, ...]
        min_main_movement: Minimum movement to consider as "main movement"
        max_intermediate_deviation: Maximum deviation for intermediate points to be considered jerky
    """
    if len(actions) < 3:
        return actions
    
    cleaned_actions = []
    i = 0
    
    while i < len(actions):
        current_action = actions[i]
        cleaned_actions.append(current_action)
        
        # Look ahead for potential jerkiness pattern
        if i < len(actions) - 2:
            # Find the next significant movement
            start_pos = current_action['pos']
            
            # Look for end of main movement (skip potential intermediate jerky points)
            j = i + 1
            potential_intermediates = []
            
            while j < len(actions):
                candidate_pos = actions[j]['pos']
                main_movement = abs(candidate_pos - start_pos)
                
                # If this could be the end of a main movement
                if main_movement >= min_main_movement:
                    # Check if there are jerky intermediates
                    if potential_intermediates:
                        # Calculate direct movement without intermediates
                        direct_movement = abs(candidate_pos - start_pos)
                        
                        # Check if intermediates are creating jerkiness
                        should_remove_intermediates = True
                        
                        for intermediate in potential_intermediates:
                            # Calculate how much the intermediate deviates from direct path
                            intermediate_pos = intermediate['pos']
                            
                            # Simple deviation check: is the intermediate creating unnecessary movement?
                            # If the movement start->intermediate->end has much more total distance
                            # than the direct start->end movement, it's jerky
                            
                            movement_via_intermediate = (abs(intermediate_pos - start_pos) + 
                                                       abs(candidate_pos - intermediate_pos))
                            direct_movement_distance = abs(candidate_pos - start_pos)
                            
                            # If going through intermediate adds significant extra movement, it's jerky
                            extra_movement = movement_via_intermediate - direct_movement_distance
                            
                            # Also check if intermediate is a small deviation from the main trend
                            if extra_movement <= max_intermediate_deviation:
                                should_remove_intermediates = False
                                break
                        
                        if should_remove_intermediates:
                            print(f"Removing jerky intermediates between {current_action['at']}ms (pos={start_pos}) "
                                  f"and {actions[j]['at']}ms (pos={candidate_pos})")
                            for intermediate in potential_intermediates:
                                print(f"  - Removing {intermediate['at']}ms (pos={intermediate['pos']})")
                            
                            # Skip the intermediates
                            i = j
                            break
                    
                    # No jerky intermediates found, continue normally
                    i += 1
                    break
                else:
                    # This might be an intermediate point
                    potential_intermediates.append(actions[j])
                    j += 1
            else:
                # Reached end of actions
                i += 1
        else:
            i += 1
    
    return cleaned_actions

def detect_specific_jerkiness_pattern(actions):
    """
    Detect the specific pattern the user mentioned:
    7624: pos=0 -> 7824: pos=69 -> 7874: pos=64 -> 7991: pos=94
    
    Pattern: Large movement interrupted by small oscillations
    """
    cleaned_actions = []
    i = 0
    
    while i < len(actions) - 3:
        # Get 4-point window
        p1, p2, p3, p4 = actions[i:i+4]
        
        # Check if this matches the jerkiness pattern
        main_movement = abs(p4['pos'] - p1['pos'])  # Overall movement
        oscillation1 = abs(p2['pos'] - p1['pos'])   # First intermediate
        oscillation2 = abs(p3['pos'] - p2['pos'])   # Small oscillation
        final_movement = abs(p4['pos'] - p3['pos']) # Final movement
        
        # Pattern detection:
        # 1. Main movement is significant (> 50)
        # 2. There are intermediate points that create small oscillations
        # 3. The intermediates don't contribute to the main direction
        
        if (main_movement > 50 and 
            oscillation2 < 20 and  # Small oscillation between intermediates
            oscillation1 > 30):    # But first intermediate is significant
            
            # Check if intermediates are deviating from main trend
            main_direction = 1 if p4['pos'] > p1['pos'] else -1
            
            # If intermediate points are not following main direction smoothly
            intermediate_contributes = False
            
            # Simple check: if removing intermediates results in cleaner motion
            direct_slope = (p4['pos'] - p1['pos']) / max(1, p4['at'] - p1['at'])
            
            # Check if intermediates follow this slope reasonably
            expected_p2_pos = p1['pos'] + direct_slope * (p2['at'] - p1['at'])
            expected_p3_pos = p1['pos'] + direct_slope * (p3['at'] - p1['at'])
            
            p2_deviation = abs(p2['pos'] - expected_p2_pos)
            p3_deviation = abs(p3['pos'] - expected_p3_pos)
            
            if p2_deviation > 20 or p3_deviation > 20:
                print(f"Jerky pattern detected:")
                print(f"  {p1['at']}ms: pos={p1['pos']}")
                print(f"  {p2['at']}ms: pos={p2['pos']} (remove - deviation {p2_deviation:.1f})")
                print(f"  {p3['at']}ms: pos={p3['pos']} (remove - deviation {p3_deviation:.1f})")
                print(f"  {p4['at']}ms: pos={p4['pos']}")
                
                # Keep p1 and p4, skip p2 and p3
                cleaned_actions.append(p1)
                cleaned_actions.append(p4)
                i += 4
                continue
        
        # No pattern detected, keep current point and move forward
        cleaned_actions.append(actions[i])
        i += 1
    
    # Add remaining actions
    while i < len(actions):
        cleaned_actions.append(actions[i])
        i += 1
    
    return cleaned_actions

def test_smart_jerk_removal():
    """Test the smart jerk removal algorithm"""
    
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
    
    print("=== SMART JERK REMOVAL ALGORITHM TEST ===")
    print(f"Original funscript: {len(example_funscript['actions'])} points")
    print()
    
    # Test on specific sequence first
    print("=== TESTING ON USER'S SPECIFIC EXAMPLE ===")
    test_sequence = [
        {"at": 7624, "pos": 0},
        {"at": 7824, "pos": 69},  # Should be removed
        {"at": 7874, "pos": 64},  # Should be removed
        {"at": 7991, "pos": 94}
    ]
    
    print("Original sequence:")
    for action in test_sequence:
        print(f"  {action['at']}ms: pos={action['pos']}")
    
    cleaned_sequence = detect_specific_jerkiness_pattern(test_sequence)
    print(f"\nCleaned sequence ({len(cleaned_sequence)} points):")
    for action in cleaned_sequence:
        print(f"  {action['at']}ms: pos={action['pos']}")
    
    # Test on full funscript
    print("\n=== TESTING ON FULL FUNSCRIPT ===")
    cleaned_funscript = copy.deepcopy(example_funscript)
    cleaned_actions = detect_specific_jerkiness_pattern(cleaned_funscript['actions'])
    cleaned_funscript['actions'] = cleaned_actions
    
    print(f"Original points: {len(example_funscript['actions'])}")
    print(f"Cleaned points:  {len(cleaned_actions)}")
    print(f"Points removed:  {len(example_funscript['actions']) - len(cleaned_actions)}")
    print(f"Reduction:       {((len(example_funscript['actions']) - len(cleaned_actions)) / len(example_funscript['actions']) * 100):.1f}%")
    
    # Save results
    with open('smart_jerk_cleaned.funscript', 'w') as f:
        json.dump(cleaned_funscript, f, indent=2)
    
    print(f"\nCleaned funscript saved as: smart_jerk_cleaned.funscript")
    
    return cleaned_funscript

if __name__ == "__main__":
    test_smart_jerk_removal()