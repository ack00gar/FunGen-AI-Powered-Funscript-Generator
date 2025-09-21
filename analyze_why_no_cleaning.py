#!/usr/bin/env python3
"""
Analyze why the recursive cleaning didn't work on the original data.
"""

def analyze_original_data():
    """Analyze the original data to see why recursive cleaning didn't work."""
    print("🔍 ANALYZING WHY RECURSIVE CLEANING DIDN'T WORK")
    print("=" * 60)
    
    # Original data
    positions = [13, 86, 25, 96, 1, 83, 5, 84, 12, 91, 8, 89, 6, 87, 4, 85, 2, 83, 0, 81, 0, 79, 2, 77, 4]
    
    print(f"📊 Data: {positions}")
    print(f"📈 Length: {len(positions)} points")
    print()
    
    print("🔍 Checking each position for the pattern:")
    print("Pattern: abs(prev_val - val) < threshold AND prev_prev_val > prev_val AND val > next_val")
    print()
    
    threshold = 30
    matches_found = 0
    
    for i in range(2, len(positions) - 1):
        prev_prev_val = positions[i-2]
        prev_val = positions[i-1]
        val = positions[i]
        next_val = positions[i+1]
        
        condition1 = abs(prev_val - val) < threshold
        condition2 = prev_prev_val > prev_val  
        condition3 = val > next_val
        
        print(f"i={i:2d}: {prev_prev_val:2d}→{prev_val:2d}→{val:2d}→{next_val:2d}")
        print(f"      abs({prev_val}-{val})={abs(prev_val-val):2d} < {threshold}: {condition1}")
        print(f"      {prev_prev_val:2d} > {prev_val:2d}: {condition2}")
        print(f"      {val:2d} > {next_val:2d}: {condition3}")
        print(f"      ALL: {condition1 and condition2 and condition3}")
        
        if condition1 and condition2 and condition3:
            matches_found += 1
            print(f"      ✅ MATCH! Would remove {prev_val} and {val}")
        
        print()
    
    print(f"🎯 ANALYSIS RESULTS:")
    print(f"   Total matches found: {matches_found}")
    
    if matches_found == 0:
        print("   ❌ No matches found because:")
        print("   • Most changes are >30 (extreme jumps >70)")
        print("   • The oscillation pattern doesn't match the specific condition")
        print("   • val > next_val is rare in this oscillating data")
        
        print("\n🔍 Let's analyze the actual pattern in this data:")
        
        # Check what patterns actually exist
        large_changes = 0
        oscillations = 0
        
        for i in range(1, len(positions) - 1):
            prev_val = positions[i-1]
            val = positions[i]
            next_val = positions[i+1]
            
            change_in = abs(val - prev_val)
            change_out = abs(next_val - val)
            
            if change_in > 70 or change_out > 70:
                large_changes += 1
            
            # Check for oscillation (up then down, or down then up)
            if (prev_val < val > next_val) or (prev_val > val < next_val):
                oscillations += 1
        
        print(f"   📈 Large changes (>70): {large_changes}/{len(positions)-2}")
        print(f"   📈 Oscillations: {oscillations}/{len(positions)-2}")
        print()
        print("   💡 This data needs a different approach:")
        print("   • Line-fitting outlier detection (works on extreme jumps)")
        print("   • Intermediate insertion (adds smoothing points)")
        print("   • Sparse smoothing (adaptive smoothing)")
    
    # Show why line-fitting works better
    print(f"\n🎯 WHY LINE-FITTING OUTLIER DETECTION WORKS BETTER:")
    print("   • Looks at trajectory lines across multiple points")
    print("   • Identifies points that deviate from expected paths")
    print("   • Handles extreme jumps (>70) effectively")
    print("   • Works with sparse, irregular data")
    
    print(f"\n🎯 WHEN TO USE RECURSIVE CLEANING:")
    print("   • Small local variations (changes <30)")
    print("   • Dense data with minor noise")
    print("   • When you have clear 'bump' patterns")
    print("   • Example: [100, 80, 85, 60, 40] - small 85 'bump'")

if __name__ == "__main__":
    analyze_original_data()