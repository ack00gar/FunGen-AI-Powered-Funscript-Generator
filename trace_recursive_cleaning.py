#!/usr/bin/env python3
"""
Detailed trace of the recursive cleaning algorithm.
"""

def trace_clean_points(points, threshold=30, depth=0):
    """
    Trace version of the recursive cleaning algorithm.
    """
    indent = "  " * depth
    print(f"{indent}🔍 Processing: {points} (depth {depth})")
    
    if len(points) < 4:
        print(f"{indent}   ⚠️  Too few points, returning: {points}")
        return points
    
    for i in range(2, len(points) - 1):
        prev_prev_val = points[i-2]
        prev_val = points[i-1]
        val = points[i]
        next_val = points[i+1]
        
        print(f"{indent}   Checking i={i}: {prev_prev_val}→{prev_val}→{val}→{next_val}")
        
        condition1 = abs(prev_val - val) < threshold
        condition2 = prev_prev_val > prev_val  
        condition3 = val > next_val
        
        print(f"{indent}      abs({prev_val}-{val})={abs(prev_val-val)} < {threshold}: {condition1}")
        print(f"{indent}      {prev_prev_val} > {prev_val}: {condition2}")
        print(f"{indent}      {val} > {next_val}: {condition3}")
        
        if condition1 and condition2 and condition3:
            print(f"{indent}   ✅ PATTERN MATCH! Removing {prev_val} and {val} (indices {i-1}, {i})")
            new_points = points[:i-1] + points[i+1:]
            print(f"{indent}   📝 New points: {new_points}")
            print(f"{indent}   🔄 Recursing...")
            return trace_clean_points(new_points, threshold, depth + 1)
        else:
            print(f"{indent}      ❌ No match")
    
    print(f"{indent}✅ No more matches found, returning: {points}")
    return points

def main():
    print("🔍 DETAILED TRACE OF RECURSIVE CLEANING ALGORITHM")
    print("=" * 70)
    
    test_cases = [
        {
            'name': 'User Example',
            'data': [100, 80, 85, 60, 40]
        },
        {
            'name': 'Multiple Patterns',
            'data': [90, 70, 75, 50, 55, 30]
        }
    ]
    
    for test_case in test_cases:
        print(f"\n📊 TRACE: {test_case['name']}")
        print("=" * 50)
        result = trace_clean_points(test_case['data'])
        print(f"\n🎯 FINAL RESULT: {test_case['data']} → {result}")
        removed = len(test_case['data']) - len(result)
        print(f"📊 Removed {removed} points")

if __name__ == "__main__":
    main()