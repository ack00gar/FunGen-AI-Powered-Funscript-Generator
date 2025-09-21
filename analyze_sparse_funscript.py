#!/usr/bin/env python3
"""
Analyze the sparse funscript data to understand the pattern and improve the Anti-Jerk filter.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_funscript_data():
    """Analyze the provided funscript data."""
    
    # The funscript data from the user
    funscript_data = {
        "version": "1.0",
        "author": "FunGen beta 0.5.0", 
        "inverted": False,
        "range": 100,
        "actions": [
            {"at": 16, "pos": 50}, {"at": 716, "pos": 81}, {"at": 1084, "pos": 2}, {"at": 1334, "pos": 94},
            {"at": 1551, "pos": 4}, {"at": 1751, "pos": 100}, {"at": 1951, "pos": 5}, {"at": 2118, "pos": 95},
            {"at": 2268, "pos": 0}, {"at": 2452, "pos": 90}, {"at": 2619, "pos": 0}, {"at": 2786, "pos": 98},
            {"at": 2952, "pos": 5}, {"at": 3086, "pos": 94}, {"at": 3236, "pos": 0}, {"at": 3403, "pos": 100},
            {"at": 3553, "pos": 5}, {"at": 3720, "pos": 99}, {"at": 3853, "pos": 5}, {"at": 4003, "pos": 100},
            {"at": 4187, "pos": 2}, {"at": 4354, "pos": 99}, {"at": 4521, "pos": 6}, {"at": 4921, "pos": 95},
            {"at": 5155, "pos": 4}, {"at": 5355, "pos": 54}, {"at": 5555, "pos": 41}, {"at": 5855, "pos": 98},
            {"at": 6089, "pos": 35}, {"at": 6156, "pos": 35}, {"at": 6356, "pos": 100}, {"at": 6690, "pos": 0},
            {"at": 6990, "pos": 100}, {"at": 7190, "pos": 0}, {"at": 7440, "pos": 100}, {"at": 7624, "pos": 0},
            {"at": 7824, "pos": 69}, {"at": 7874, "pos": 64}, {"at": 7991, "pos": 94}, {"at": 8158, "pos": 8},
            {"at": 8458, "pos": 100}, {"at": 8725, "pos": 1}, {"at": 8975, "pos": 100}, {"at": 9192, "pos": 0},
            {"at": 9392, "pos": 100}, {"at": 9659, "pos": 0}, {"at": 9926, "pos": 100}, {"at": 10076, "pos": 0},
            {"at": 10393, "pos": 99}, {"at": 10593, "pos": 26}, {"at": 10727, "pos": 99}, {"at": 10894, "pos": 96},
            {"at": 11094, "pos": 19}, {"at": 11261, "pos": 92}, {"at": 11494, "pos": 0}, {"at": 11744, "pos": 100},
            {"at": 11995, "pos": 0}
        ]
    }
    
    # Extract data
    times = np.array([action["at"] for action in funscript_data["actions"]])
    positions = np.array([action["pos"] for action in funscript_data["actions"]])
    
    print(f"📊 FUNSCRIPT ANALYSIS")
    print(f"{'='*50}")
    print(f"Total duration: {times[-1]/1000:.1f} seconds")
    print(f"Total actions: {len(times)}")
    print(f"Average interval: {np.mean(np.diff(times)):.1f}ms")
    print(f"Position range: {positions.min()}-{positions.max()}")
    print(f"Position std dev: {positions.std():.1f}")
    
    # Calculate movement characteristics
    movements = np.abs(np.diff(positions))
    print(f"\n📈 MOVEMENT ANALYSIS")
    print(f"Average movement: {movements.mean():.1f}")
    print(f"Max movement: {movements.max()}")
    print(f"Large jumps (>50): {np.sum(movements > 50)}/{len(movements)} ({np.sum(movements > 50)/len(movements)*100:.1f}%)")
    print(f"Extreme jumps (>80): {np.sum(movements > 80)}/{len(movements)} ({np.sum(movements > 80)/len(movements)*100:.1f}%)")
    
    # Time intervals analysis
    intervals = np.diff(times)
    print(f"\n⏱️ TIMING ANALYSIS")
    print(f"Min interval: {intervals.min()}ms")
    print(f"Max interval: {intervals.max()}ms") 
    print(f"Interval std dev: {intervals.std():.1f}ms")
    print(f"Short intervals (<100ms): {np.sum(intervals < 100)}/{len(intervals)}")
    print(f"Long intervals (>400ms): {np.sum(intervals > 400)}/{len(intervals)}")
    
    # Pattern detection
    print(f"\n🔍 PATTERN ANALYSIS")
    
    # Check for alternating pattern
    alternating_count = 0
    for i in range(1, len(positions)-1):
        prev_pos = positions[i-1]
        curr_pos = positions[i]
        next_pos = positions[i+1]
        
        # Check if current position is between extremes and next jumps to opposite
        if (prev_pos < 20 and curr_pos < 20 and next_pos > 80) or \
           (prev_pos > 80 and curr_pos > 80 and next_pos < 20):
            alternating_count += 1
    
    print(f"Extreme alternations: {alternating_count}/{len(positions)-2} ({alternating_count/(len(positions)-2)*100:.1f}%)")
    
    # Check for consecutive similar values
    consecutive_high = 0
    consecutive_low = 0
    for i in range(len(positions)-1):
        if positions[i] > 80 and positions[i+1] > 80:
            consecutive_high += 1
        elif positions[i] < 20 and positions[i+1] < 20:
            consecutive_low += 1
    
    print(f"Consecutive highs (>80): {consecutive_high}")
    print(f"Consecutive lows (<20): {consecutive_low}")
    
    # Identify problematic regions
    print(f"\n⚠️ PROBLEMATIC REGIONS")
    problematic_indices = []
    
    for i in range(len(movements)):
        movement = movements[i]
        time_gap = intervals[i]
        
        # Identify jerky movements: large position change in short time
        if movement > 70 and time_gap < 200:  # >70 position change in <200ms
            problematic_indices.append(i)
            time_sec = times[i] / 1000
            print(f"  Time {time_sec:.1f}s: {positions[i]} → {positions[i+1]} ({movement} change in {time_gap}ms)")
    
    print(f"Total problematic transitions: {len(problematic_indices)}")
    
    # Create visualization
    create_analysis_plot(times, positions, movements, intervals, problematic_indices)
    
    return times, positions, movements, intervals, problematic_indices

def create_analysis_plot(times, positions, movements, intervals, problematic_indices):
    """Create analysis visualization."""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        times_sec = times / 1000
        
        # Plot 1: Position over time
        ax1.plot(times_sec, positions, 'b-o', markersize=4, linewidth=2)
        ax1.set_title('Funscript Positions Over Time')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Position (0-100)')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-5, 105)
        
        # Highlight problematic regions
        for idx in problematic_indices:
            ax1.axvspan(times_sec[idx], times_sec[idx+1], alpha=0.3, color='red')
        
        # Plot 2: Movement magnitude
        movement_times = times_sec[1:]  # One less point than positions
        ax2.plot(movement_times, movements, 'r-o', markersize=3, linewidth=1)
        ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Large jump threshold')
        ax2.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Extreme jump threshold')
        ax2.set_title('Position Changes Between Actions')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Position Change')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Time intervals
        ax3.plot(movement_times, intervals, 'g-o', markersize=3, linewidth=1)
        ax3.axhline(y=100, color='orange', linestyle='--', alpha=0.7, label='Short interval')
        ax3.axhline(y=400, color='red', linestyle='--', alpha=0.7, label='Long interval')
        ax3.set_title('Time Intervals Between Actions')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Interval (ms)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: Movement vs Time relationship
        ax4.scatter(intervals, movements, alpha=0.6, s=50)
        ax4.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Jerk threshold')
        ax4.axvline(x=200, color='red', linestyle='--', alpha=0.7, label='Short time threshold')
        ax4.set_title('Movement Size vs Time Interval')
        ax4.set_xlabel('Time Interval (ms)')
        ax4.set_ylabel('Position Change')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Add text box with summary
        summary_text = f"""DATA SUMMARY:
        • {len(times)} actions over {times[-1]/1000:.1f}s
        • {np.sum(movements > 50)} large jumps (>50)
        • {np.sum(movements > 80)} extreme jumps (>80)
        • {len(problematic_indices)} jerky transitions"""
        
        ax4.text(0.02, 0.98, summary_text, transform=ax4.transAxes, 
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('sparse_funscript_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n📊 Analysis plot saved as 'sparse_funscript_analysis.png'")
        
    except Exception as e:
        print(f"Could not create analysis plot: {e}")

if __name__ == "__main__":
    analyze_funscript_data()