#!/usr/bin/env python3
"""
Line-Fitting Outlier Detection for Funscript Anti-Jerk Filtering

This approach draws lines from every point to the next few points to identify
outliers that deviate significantly from expected trajectories and marks them
for removal or smoothing.

Key concept: If we draw a line from point A to point C, and point B lies
significantly off this line, then B might be an outlier that creates jerkiness.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass


@dataclass
class OutlierAnalysis:
    """Analysis of a point's outlier characteristics."""
    point_index: int
    time: float
    position: float
    max_line_deviation: float
    avg_line_deviation: float
    outlier_score: float
    is_outlier: bool
    contributing_lines: List[Dict[str, Any]]
    removal_confidence: float


class LineFittingOutlierDetector:
    """
    Detect outliers by analyzing how much points deviate from lines drawn
    between other points in their neighborhood.
    """
    
    def __init__(self, 
                 max_line_distance: int = 5,
                 outlier_threshold: float = 15.0,
                 min_contributing_lines: int = 2):
        self.max_line_distance = max_line_distance  # Look ahead up to N points
        self.outlier_threshold = outlier_threshold   # Deviation threshold for outliers
        self.min_contributing_lines = min_contributing_lines  # Min lines to analyze a point
    
    def detect_outliers(self, times: np.ndarray, positions: np.ndarray) -> Dict[str, Any]:
        """
        Detect outlier points using line-fitting analysis.
        
        Args:
            times: Time values
            positions: Position values
            
        Returns:
            Outlier analysis results
        """
        if len(positions) < 4:
            return self._create_minimal_result(times, positions)
        
        # Step 1: Generate all possible lines between points
        lines = self._generate_trajectory_lines(times, positions)
        
        # Step 2: For each point, analyze how it deviates from relevant lines
        point_analyses = []
        for i in range(len(positions)):
            analysis = self._analyze_point_outlier_status(i, times, positions, lines)
            point_analyses.append(analysis)
        
        # Step 3: Identify outliers based on deviation analysis
        outliers = self._identify_outliers(point_analyses)
        
        # Step 4: Create filtered signal
        filtered_result = self._create_filtered_signal(times, positions, outliers)
        
        return {
            'point_analyses': point_analyses,
            'outlier_indices': outliers,
            'trajectory_lines': lines,
            'filtered_times': filtered_result['times'],
            'filtered_positions': filtered_result['positions'],
            'statistics': self._calculate_statistics(point_analyses, times, positions)
        }
    
    def _generate_trajectory_lines(self, times: np.ndarray, positions: np.ndarray) -> List[Dict[str, Any]]:
        """
        Generate all trajectory lines between points within the specified distance.
        
        Args:
            times: Time values
            positions: Position values
            
        Returns:
            List of line definitions
        """
        lines = []
        n = len(positions)
        
        for start_idx in range(n):
            for end_idx in range(start_idx + 2, min(n, start_idx + self.max_line_distance + 1)):
                # Create line from start_idx to end_idx
                start_time = times[start_idx]
                start_pos = positions[start_idx]
                end_time = times[end_idx]
                end_pos = positions[end_idx]
                
                # Calculate line parameters
                if end_time != start_time:
                    slope = (end_pos - start_pos) / (end_time - start_time)
                    intercept = start_pos - slope * start_time
                else:
                    slope = 0
                    intercept = start_pos
                
                line = {
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'start_time': start_time,
                    'start_pos': start_pos,
                    'end_time': end_time,
                    'end_pos': end_pos,
                    'slope': slope,
                    'intercept': intercept,
                    'length': abs(end_pos - start_pos),
                    'duration': end_time - start_time,
                    'intermediate_points': list(range(start_idx + 1, end_idx))
                }
                
                lines.append(line)
        
        return lines
    
    def _analyze_point_outlier_status(self, point_idx: int, times: np.ndarray, 
                                    positions: np.ndarray, lines: List[Dict[str, Any]]) -> OutlierAnalysis:
        """
        Analyze whether a point is an outlier based on how it deviates from trajectory lines.
        
        Args:
            point_idx: Index of point to analyze
            times: Time values
            positions: Position values
            lines: List of trajectory lines
            
        Returns:
            Outlier analysis for the point
        """
        point_time = times[point_idx]
        point_pos = positions[point_idx]
        
        # Find all lines that should pass near this point (intermediate points)
        relevant_lines = []
        deviations = []
        
        for line in lines:
            if point_idx in line['intermediate_points']:
                # This point lies between the line's endpoints
                # Calculate expected position on the line
                expected_pos = line['slope'] * point_time + line['intercept']
                deviation = abs(point_pos - expected_pos)
                
                relevant_lines.append({
                    'line': line,
                    'expected_pos': expected_pos,
                    'deviation': deviation
                })
                deviations.append(deviation)
        
        # Calculate outlier metrics
        if deviations:
            max_deviation = max(deviations)
            avg_deviation = np.mean(deviations)
            
            # Calculate outlier score (higher = more likely to be outlier)
            outlier_score = avg_deviation
            
            # Boost score if consistently deviates from multiple lines
            if len(deviations) >= 2:
                consistency_penalty = np.std(deviations)  # Lower std = more consistent deviation
                outlier_score += (avg_deviation * (1 + 1/max(1, consistency_penalty)))
            
            # Determine if this is an outlier
            is_outlier = (outlier_score > self.outlier_threshold and 
                         len(relevant_lines) >= self.min_contributing_lines)
            
            # Calculate removal confidence
            removal_confidence = min(100.0, (outlier_score / self.outlier_threshold) * 100)
            
        else:
            max_deviation = 0.0
            avg_deviation = 0.0
            outlier_score = 0.0
            is_outlier = False
            removal_confidence = 0.0
        
        return OutlierAnalysis(
            point_index=point_idx,
            time=point_time,
            position=point_pos,
            max_line_deviation=max_deviation,
            avg_line_deviation=avg_deviation,
            outlier_score=outlier_score,
            is_outlier=is_outlier,
            contributing_lines=relevant_lines,
            removal_confidence=removal_confidence
        )
    
    def _identify_outliers(self, point_analyses: List[OutlierAnalysis]) -> List[int]:
        """
        Identify final list of outlier indices with safety checks.
        
        Args:
            point_analyses: List of point analyses
            
        Returns:
            List of outlier indices
        """
        outliers = []
        
        for analysis in point_analyses:
            # Basic outlier detection
            if analysis.is_outlier:
                # Safety checks
                
                # Never remove first or last point
                if analysis.point_index == 0 or analysis.point_index == len(point_analyses) - 1:
                    continue
                
                # Don't remove if very high confidence is needed but not met
                if analysis.removal_confidence < 60.0:  # Require at least 60% confidence
                    continue
                
                # Don't remove global extrema
                all_positions = [p.position for p in point_analyses]
                if (analysis.position == min(all_positions) or 
                    analysis.position == max(all_positions)):
                    continue
                
                outliers.append(analysis.point_index)
        
        # Additional safety: don't remove more than 30% of points
        max_removals = len(point_analyses) // 3
        if len(outliers) > max_removals:
            # Keep only the most confident outliers
            outlier_analyses = [point_analyses[i] for i in outliers]
            outlier_analyses.sort(key=lambda x: x.removal_confidence, reverse=True)
            outliers = [a.point_index for a in outlier_analyses[:max_removals]]
        
        return sorted(outliers)
    
    def _create_filtered_signal(self, times: np.ndarray, positions: np.ndarray, 
                              outliers: List[int]) -> Dict[str, np.ndarray]:
        """
        Create filtered signal with outliers removed.
        
        Args:
            times: Original time values
            positions: Original position values
            outliers: Indices of outliers to remove
            
        Returns:
            Filtered signal
        """
        if not outliers:
            return {'times': times.copy(), 'positions': positions.copy()}
        
        # Create mask for points to keep
        keep_mask = np.ones(len(times), dtype=bool)
        keep_mask[outliers] = False
        
        filtered_times = times[keep_mask]
        filtered_positions = positions[keep_mask]
        
        return {'times': filtered_times, 'positions': filtered_positions}
    
    def _calculate_statistics(self, point_analyses: List[OutlierAnalysis], 
                            times: np.ndarray, positions: np.ndarray) -> Dict[str, Any]:
        """Calculate statistics about the outlier detection."""
        total_points = len(point_analyses)
        outlier_count = sum(1 for p in point_analyses if p.is_outlier)
        
        if point_analyses:
            avg_outlier_score = np.mean([p.outlier_score for p in point_analyses])
            max_outlier_score = max(p.outlier_score for p in point_analyses)
            avg_deviation = np.mean([p.avg_line_deviation for p in point_analyses if p.avg_line_deviation > 0])
        else:
            avg_outlier_score = 0.0
            max_outlier_score = 0.0
            avg_deviation = 0.0
        
        return {
            'total_points': total_points,
            'outliers_detected': outlier_count,
            'outlier_percentage': (outlier_count / total_points * 100) if total_points > 0 else 0,
            'avg_outlier_score': avg_outlier_score,
            'max_outlier_score': max_outlier_score,
            'avg_line_deviation': avg_deviation
        }
    
    def _create_minimal_result(self, times: np.ndarray, positions: np.ndarray) -> Dict[str, Any]:
        """Create minimal result for very short signals."""
        n = len(positions)
        point_analyses = []
        
        for i in range(n):
            analysis = OutlierAnalysis(
                point_index=i,
                time=times[i],
                position=positions[i],
                max_line_deviation=0.0,
                avg_line_deviation=0.0,
                outlier_score=0.0,
                is_outlier=False,
                contributing_lines=[],
                removal_confidence=0.0
            )
            point_analyses.append(analysis)
        
        return {
            'point_analyses': point_analyses,
            'outlier_indices': [],
            'trajectory_lines': [],
            'filtered_times': times.copy(),
            'filtered_positions': positions.copy(),
            'statistics': {
                'total_points': n,
                'outliers_detected': 0,
                'outlier_percentage': 0.0,
                'avg_outlier_score': 0.0,
                'max_outlier_score': 0.0,
                'avg_line_deviation': 0.0
            }
        }


def test_line_fitting_outlier_detection():
    """Test the line-fitting outlier detection on the sparse funscript data."""
    print("📐 LINE-FITTING OUTLIER DETECTION")
    print("=" * 50)
    
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
    
    print(f"📊 ORIGINAL SIGNAL:")
    print(f"Total points: {len(positions)}")
    original_movements = np.abs(np.diff(positions))
    extreme_jumps = np.sum(original_movements > 70)
    print(f"Extreme jumps (>70): {extreme_jumps}")
    
    # Test different threshold levels
    threshold_levels = [10.0, 15.0, 20.0, 25.0, 30.0]
    results = {}
    
    for threshold in threshold_levels:
        print(f"\\n🔧 Testing with outlier threshold = {threshold}")
        
        detector = LineFittingOutlierDetector(
            max_line_distance=4,  # Look ahead up to 4 points
            outlier_threshold=threshold,
            min_contributing_lines=2
        )
        
        result = detector.detect_outliers(times, positions)
        results[threshold] = result
        
        stats = result['statistics']
        outliers = result['outlier_indices']
        
        print(f"   Outliers detected: {len(outliers)}/{stats['total_points']} ({stats['outlier_percentage']:.1f}%)")
        print(f"   Avg outlier score: {stats['avg_outlier_score']:.1f}")
        print(f"   Avg line deviation: {stats['avg_line_deviation']:.1f}")
        
        if outliers:
            print(f"   Outlier points: {outliers}")
            for idx in outliers[:3]:  # Show first 3
                analysis = result['point_analyses'][idx]
                print(f"     Point {idx}: pos={analysis.position}, score={analysis.outlier_score:.1f}, confidence={analysis.removal_confidence:.1f}%")
        
        # Calculate improvement if outliers are removed
        if len(result['filtered_positions']) > 1:
            filtered_movements = np.abs(np.diff(result['filtered_positions']))
            filtered_extreme_jumps = np.sum(filtered_movements > 70)
            improvement = ((extreme_jumps - filtered_extreme_jumps) / extreme_jumps * 100) if extreme_jumps > 0 else 0
            print(f"   Extreme jumps after filtering: {filtered_extreme_jumps} ({improvement:.1f}% reduction)")
    
    # Detailed analysis of best threshold
    best_threshold = 20.0  # Middle ground
    print(f"\\n🔍 DETAILED ANALYSIS (threshold = {best_threshold}):")
    best_result = results[best_threshold]
    
    print(f"\\n📏 TRAJECTORY LINES GENERATED:")
    lines = best_result['trajectory_lines']
    print(f"Total lines: {len(lines)}")
    for i, line in enumerate(lines[:10]):  # Show first 10
        print(f"  Line {i}: Point {line['start_idx']} → Point {line['end_idx']} "
              f"(covers {line['intermediate_points']})")
    
    print(f"\\n🎯 POINT-BY-POINT ANALYSIS:")
    for analysis in best_result['point_analyses']:
        if analysis.outlier_score > 5.0:  # Show points with some deviation
            print(f"  Point {analysis.point_index}: pos={analysis.position}, "
                  f"score={analysis.outlier_score:.1f}, "
                  f"avg_dev={analysis.avg_line_deviation:.1f}, "
                  f"outlier={'YES' if analysis.is_outlier else 'NO'}")
    
    # Create visualization
    create_line_fitting_plot(times, positions, results)
    
    print(f"\\n📊 Line-fitting analysis plot saved as 'line_fitting_results.png'")
    
    return results


def create_line_fitting_plot(times, positions, results):
    """Create visualization of the line-fitting outlier detection."""
    try:
        num_thresholds = len(results)
        fig, axes = plt.subplots(num_thresholds + 1, 1, figsize=(16, (num_thresholds + 1) * 3))
        
        if num_thresholds == 1:
            axes = [axes]
        
        times_sec = times / 1000.0
        
        # Plot original signal with trajectory lines
        axes[0].plot(times_sec, positions, 'b-o', markersize=6, linewidth=2, 
                    alpha=0.8, label='Original Signal')
        
        # Draw some trajectory lines to show the concept
        if results:
            sample_result = list(results.values())[0]
            lines = sample_result['trajectory_lines']
            
            # Draw every 3rd line to avoid clutter
            for i, line in enumerate(lines[::3]):
                start_time = line['start_time'] / 1000.0
                end_time = line['end_time'] / 1000.0
                start_pos = line['start_pos']
                end_pos = line['end_pos']
                
                axes[0].plot([start_time, end_time], [start_pos, end_pos], 
                           'r--', alpha=0.3, linewidth=1)
        
        axes[0].set_title('Original Signal with Sample Trajectory Lines', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Position (0-100)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[0].set_ylim(-5, 105)
        
        # Plot each threshold level
        colors = ['green', 'blue', 'purple', 'orange', 'brown']
        
        for idx, (threshold, result) in enumerate(results.items()):
            ax = axes[idx + 1]
            
            # Plot original signal
            ax.plot(times_sec, positions, 'k-', linewidth=1, alpha=0.4, label='Original')
            ax.scatter(times_sec, positions, c='gray', s=30, alpha=0.5)
            
            # Highlight outliers
            outliers = result['outlier_indices']
            if outliers:
                ax.scatter(times_sec[outliers], positions[outliers],
                          c='red', s=80, marker='X', alpha=0.8,
                          label=f'Outliers ({len(outliers)})', zorder=5)
            
            # Plot filtered signal
            filtered_times_sec = result['filtered_times'] / 1000.0
            ax.plot(filtered_times_sec, result['filtered_positions'],
                   color=colors[idx % len(colors)], linewidth=3, alpha=0.8,
                   label='Filtered signal', zorder=3)
            
            # Show outlier scores as text
            for analysis in result['point_analyses']:
                if analysis.outlier_score > threshold * 0.5:  # Show significant scores
                    ax.annotate(f'{analysis.outlier_score:.0f}', 
                              (analysis.time / 1000.0, analysis.position),
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=8, alpha=0.7)
            
            stats = result['statistics']
            title = f"Threshold = {threshold} | Outliers: {stats['outliers_detected']} "
            title += f"({stats['outlier_percentage']:.1f}%) | Avg Score: {stats['avg_outlier_score']:.1f}"
            
            ax.set_title(title, fontsize=12)
            ax.set_ylabel('Position')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_ylim(-5, 105)
        
        # Set xlabel on last plot
        axes[-1].set_xlabel('Time (seconds)')
        
        plt.tight_layout()
        plt.savefig('line_fitting_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Could not create line-fitting plot: {e}")


if __name__ == "__main__":
    test_line_fitting_outlier_detection()