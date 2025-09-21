#!/usr/bin/env python3
"""
Slope-based analysis approach for identifying meaningful vs non-meaningful points.

This analyzes:
- Slope before each point
- Slope after each point  
- Duration of slope trends
- Direction changes and their significance
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any


class SlopeBasedPointClassifier:
    """
    Classifies funscript points as meaningful or non-meaningful based on slope analysis.
    
    Key concepts:
    - Meaningful points: Start/end of consistent trends, direction changes, extrema
    - Non-meaningful points: Small oscillations, noise, redundant intermediate points
    """
    
    def __init__(self):
        pass
    
    def analyze_slopes(self, times: np.ndarray, positions: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Analyze slopes before and after each point.
        
        Args:
            times: Time values in ms
            positions: Position values 0-100
            
        Returns:
            Dictionary with slope analysis data
        """
        n = len(positions)
        
        # Calculate slopes (position change per ms)
        slopes_before = np.zeros(n)
        slopes_after = np.zeros(n)
        
        # Slope before each point (except first)
        for i in range(1, n):
            dt = times[i] - times[i-1]
            dp = positions[i] - positions[i-1]
            slopes_before[i] = dp / dt if dt > 0 else 0
        
        # Slope after each point (except last)
        for i in range(n-1):
            dt = times[i+1] - times[i]
            dp = positions[i+1] - positions[i]
            slopes_after[i] = dp / dt if dt > 0 else 0
        
        # Calculate slope changes (acceleration-like metric)
        slope_changes = np.abs(slopes_after - slopes_before)
        
        # Calculate slope directions
        directions_before = np.sign(slopes_before)
        directions_after = np.sign(slopes_after)
        direction_changes = directions_before != directions_after
        
        return {
            'slopes_before': slopes_before,
            'slopes_after': slopes_after,
            'slope_changes': slope_changes,
            'directions_before': directions_before,
            'directions_after': directions_after,
            'direction_changes': direction_changes
        }
    
    def analyze_trend_durations(self, times: np.ndarray, positions: np.ndarray, 
                               slope_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Analyze how long slope trends last before and after each point.
        
        Args:
            times: Time values
            positions: Position values
            slope_data: Output from analyze_slopes()
            
        Returns:
            Dictionary with trend duration data
        """
        n = len(positions)
        directions_before = slope_data['directions_before']
        directions_after = slope_data['directions_after']
        
        # Duration of trend before each point
        trend_duration_before = np.zeros(n)
        trend_duration_after = np.zeros(n)
        
        # Analyze backward trends
        for i in range(n):
            if i == 0:
                continue
                
            current_direction = directions_before[i]
            if current_direction == 0:  # No slope
                continue
                
            # Look backward for how long this trend continues
            duration = 0
            for j in range(i-1, -1, -1):
                if directions_before[j] == current_direction:
                    duration += times[j+1] - times[j] if j+1 < len(times) else 0
                else:
                    break
            trend_duration_before[i] = duration
        
        # Analyze forward trends
        for i in range(n):
            if i == n-1:
                continue
                
            current_direction = directions_after[i]
            if current_direction == 0:  # No slope
                continue
                
            # Look forward for how long this trend continues
            duration = 0
            for j in range(i+1, n):
                if j < len(directions_after) and directions_after[j] == current_direction:
                    duration += times[j] - times[j-1] if j > 0 else 0
                else:
                    break
            trend_duration_after[i] = duration
        
        return {
            'trend_duration_before': trend_duration_before,
            'trend_duration_after': trend_duration_after
        }
    
    def classify_points(self, times: np.ndarray, positions: np.ndarray, 
                       slope_data: Dict[str, np.ndarray], 
                       duration_data: Dict[str, np.ndarray],
                       params: Dict[str, float] = None) -> Dict[str, np.ndarray]:
        """
        Classify each point as meaningful or non-meaningful based on slope analysis.
        
        Args:
            times: Time values
            positions: Position values
            slope_data: Slope analysis data
            duration_data: Trend duration data
            params: Classification parameters
            
        Returns:
            Dictionary with classification results
        """
        if params is None:
            params = {
                'min_slope_change': 0.05,  # Minimum slope change to be significant
                'min_trend_duration': 200,  # Minimum trend duration (ms) to be meaningful
                'direction_change_weight': 2.0,  # Weight for direction changes
                'extrema_weight': 1.5,  # Weight for local extrema
                'consistency_threshold': 0.7  # Threshold for trend consistency
            }
        
        n = len(positions)
        
        # Initialize scores
        meaningfulness_scores = np.zeros(n)
        
        # 1. Direction change analysis
        direction_changes = slope_data['direction_changes']
        meaningfulness_scores += direction_changes.astype(float) * params['direction_change_weight']
        
        # 2. Significant slope changes
        slope_changes = slope_data['slope_changes']
        significant_slope_changes = slope_changes > params['min_slope_change']
        meaningfulness_scores += significant_slope_changes.astype(float)
        
        # 3. Trend duration analysis
        trend_before = duration_data['trend_duration_before']
        trend_after = duration_data['trend_duration_after']
        
        # Points that start or end long trends are more meaningful
        long_trend_before = trend_before > params['min_trend_duration']
        long_trend_after = trend_after > params['min_trend_duration']
        meaningfulness_scores += (long_trend_before | long_trend_after).astype(float)
        
        # 4. Local extrema detection
        local_maxima = np.zeros(n, dtype=bool)
        local_minima = np.zeros(n, dtype=bool)
        
        for i in range(1, n-1):
            if positions[i] > positions[i-1] and positions[i] > positions[i+1]:
                local_maxima[i] = True
            elif positions[i] < positions[i-1] and positions[i] < positions[i+1]:
                local_minima[i] = True
        
        extrema_mask = local_maxima | local_minima
        meaningfulness_scores += extrema_mask.astype(float) * params['extrema_weight']
        
        # 5. Absolute extrema (always meaningful)
        abs_min_idx = np.argmin(positions)
        abs_max_idx = np.argmax(positions)
        meaningfulness_scores[abs_min_idx] += 3.0
        meaningfulness_scores[abs_max_idx] += 3.0
        
        # 6. First and last points (always meaningful)
        meaningfulness_scores[0] += 2.0
        meaningfulness_scores[-1] += 2.0
        
        # Classify based on scores
        meaningful_threshold = np.percentile(meaningfulness_scores, 60)  # Top 40% are meaningful
        meaningful_points = meaningfulness_scores >= meaningful_threshold
        
        # Force certain points to be meaningful
        meaningful_points[0] = True  # First point
        meaningful_points[-1] = True  # Last point
        meaningful_points[abs_min_idx] = True  # Absolute minimum
        meaningful_points[abs_max_idx] = True  # Absolute maximum
        
        return {
            'meaningfulness_scores': meaningfulness_scores,
            'meaningful_points': meaningful_points,
            'non_meaningful_points': ~meaningful_points,
            'local_maxima': local_maxima,
            'local_minima': local_minima,
            'direction_changes': direction_changes,
            'significant_slope_changes': significant_slope_changes
        }
    
    def get_point_analysis_summary(self, idx: int, times: np.ndarray, positions: np.ndarray,
                                  slope_data: Dict[str, np.ndarray],
                                  duration_data: Dict[str, np.ndarray],
                                  classification: Dict[str, np.ndarray]) -> str:
        """
        Get a human-readable summary of why a point was classified as meaningful/non-meaningful.
        
        Args:
            idx: Point index
            times: Time values
            positions: Position values
            slope_data: Slope analysis data
            duration_data: Duration analysis data
            classification: Classification results
            
        Returns:
            Human-readable analysis summary
        """
        if idx < 0 or idx >= len(positions):
            return "Invalid index"
        
        summary = f"Point {idx} at t={times[idx]}ms, pos={positions[idx]}\n"
        summary += f"Score: {classification['meaningfulness_scores'][idx]:.2f}\n"
        summary += f"Classification: {'MEANINGFUL' if classification['meaningful_points'][idx] else 'NON-MEANINGFUL'}\n"
        
        # Slope information
        slope_before = slope_data['slopes_before'][idx]
        slope_after = slope_data['slopes_after'][idx]
        summary += f"Slope before: {slope_before:.3f}, after: {slope_after:.3f}\n"
        
        # Direction information
        dir_before = slope_data['directions_before'][idx]
        dir_after = slope_data['directions_after'][idx]
        summary += f"Direction before: {dir_before}, after: {dir_after}\n"
        
        if slope_data['direction_changes'][idx]:
            summary += "⭐ DIRECTION CHANGE detected\n"
        
        # Trend duration
        trend_before = duration_data['trend_duration_before'][idx]
        trend_after = duration_data['trend_duration_after'][idx]
        summary += f"Trend duration before: {trend_before:.0f}ms, after: {trend_after:.0f}ms\n"
        
        # Special properties
        if classification['local_maxima'][idx]:
            summary += "🔺 LOCAL MAXIMUM\n"
        if classification['local_minima'][idx]:
            summary += "🔻 LOCAL MINIMUM\n"
        if idx == np.argmin(positions):
            summary += "🏔️ ABSOLUTE MINIMUM\n"
        if idx == np.argmax(positions):
            summary += "🏔️ ABSOLUTE MAXIMUM\n"
        
        return summary


def test_slope_analysis():
    """Test the slope-based point classification on the sparse funscript data."""
    # Load the problematic funscript data
    funscript_actions = [
        {"at": 16, "pos": 50}, {"at": 716, "pos": 81}, {"at": 1084, "pos": 2}, {"at": 1334, "pos": 94},
        {"at": 1551, "pos": 4}, {"at": 1751, "pos": 100}, {"at": 1951, "pos": 5}, {"at": 2118, "pos": 95},
        {"at": 2268, "pos": 0}, {"at": 2452, "pos": 90}, {"at": 2619, "pos": 0}, {"at": 2786, "pos": 98},
        {"at": 2952, "pos": 5}, {"at": 3086, "pos": 94}, {"at": 3236, "pos": 0}, {"at": 3403, "pos": 100},
        {"at": 3553, "pos": 5}, {"at": 3720, "pos": 99}, {"at": 3853, "pos": 5}, {"at": 4003, "pos": 100}
    ]
    
    times = np.array([action["at"] for action in funscript_actions])
    positions = np.array([action["pos"] for action in funscript_actions])
    
    print("🔍 SLOPE-BASED POINT ANALYSIS")
    print("=" * 50)
    
    classifier = SlopeBasedPointClassifier()
    
    # Perform analysis
    slope_data = classifier.analyze_slopes(times, positions)
    duration_data = classifier.analyze_trend_durations(times, positions, slope_data)
    classification = classifier.classify_points(times, positions, slope_data, duration_data)
    
    meaningful_count = np.sum(classification['meaningful_points'])
    total_count = len(positions)
    
    print(f"📊 RESULTS SUMMARY:")
    print(f"Total points: {total_count}")
    print(f"Meaningful points: {meaningful_count} ({meaningful_count/total_count*100:.1f}%)")
    print(f"Non-meaningful points: {total_count - meaningful_count} ({(total_count-meaningful_count)/total_count*100:.1f}%)")
    
    print(f"\n🎯 MEANINGFUL POINTS:")
    for i, is_meaningful in enumerate(classification['meaningful_points']):
        if is_meaningful:
            print(f"  Point {i}: t={times[i]}ms, pos={positions[i]}, score={classification['meaningfulness_scores'][i]:.2f}")
    
    print(f"\n❌ NON-MEANINGFUL POINTS (candidates for removal/smoothing):")
    for i, is_meaningful in enumerate(classification['meaningful_points']):
        if not is_meaningful:
            print(f"  Point {i}: t={times[i]}ms, pos={positions[i]}, score={classification['meaningfulness_scores'][i]:.2f}")
    
    # Detailed analysis of a few key points
    print(f"\n🔬 DETAILED ANALYSIS OF KEY POINTS:")
    key_indices = [5, 8, 12, 15]  # Some problematic transitions
    for idx in key_indices:
        if idx < len(positions):
            print(f"\n{classifier.get_point_analysis_summary(idx, times, positions, slope_data, duration_data, classification)}")
    
    # Create visualization
    create_slope_analysis_plot(times, positions, slope_data, duration_data, classification)
    
    return classification


def create_slope_analysis_plot(times, positions, slope_data, duration_data, classification):
    """Create visualization of the slope analysis."""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        times_sec = times / 1000.0
        
        # Plot 1: Position with meaningful/non-meaningful classification
        meaningful_mask = classification['meaningful_points']
        non_meaningful_mask = ~meaningful_mask
        
        ax1.plot(times_sec, positions, 'k-', alpha=0.3, linewidth=1, label='All points')
        ax1.scatter(times_sec[meaningful_mask], positions[meaningful_mask], 
                   c='green', s=50, marker='o', label='Meaningful', zorder=5)
        ax1.scatter(times_sec[non_meaningful_mask], positions[non_meaningful_mask], 
                   c='red', s=30, marker='x', label='Non-meaningful', zorder=5)
        
        ax1.set_title('Point Classification: Meaningful vs Non-Meaningful')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Position')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Slopes before and after
        ax2.plot(times_sec, slope_data['slopes_before'], 'b-', label='Slope before', alpha=0.7)
        ax2.plot(times_sec, slope_data['slopes_after'], 'r-', label='Slope after', alpha=0.7)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Mark direction changes
        direction_changes = slope_data['direction_changes']
        ax2.scatter(times_sec[direction_changes], slope_data['slopes_after'][direction_changes], 
                   c='orange', s=100, marker='*', label='Direction changes', zorder=5)
        
        ax2.set_title('Slope Analysis: Before vs After Each Point')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Slope (pos/ms)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Trend durations
        ax3.bar(times_sec, duration_data['trend_duration_before'], width=0.1, alpha=0.6, 
               label='Duration before', color='blue')
        ax3.bar(times_sec, -duration_data['trend_duration_after'], width=0.1, alpha=0.6, 
               label='Duration after', color='red')
        
        ax3.axhline(y=200, color='orange', linestyle='--', alpha=0.7, label='Significance threshold')
        ax3.axhline(y=-200, color='orange', linestyle='--', alpha=0.7)
        
        ax3.set_title('Trend Duration Analysis')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Trend Duration (ms)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Meaningfulness scores
        scores = classification['meaningfulness_scores']
        colors = ['green' if m else 'red' for m in meaningful_mask]
        
        bars = ax4.bar(range(len(scores)), scores, color=colors, alpha=0.7)
        ax4.axhline(y=np.percentile(scores, 60), color='orange', linestyle='--', 
                   alpha=0.7, label='Meaningful threshold')
        
        ax4.set_title('Meaningfulness Scores')
        ax4.set_xlabel('Point Index')
        ax4.set_ylabel('Score')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('slope_analysis_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n📊 Slope analysis plot saved as 'slope_analysis_results.png'")
        
    except Exception as e:
        print(f"Could not create slope analysis plot: {e}")


if __name__ == "__main__":
    test_slope_analysis()