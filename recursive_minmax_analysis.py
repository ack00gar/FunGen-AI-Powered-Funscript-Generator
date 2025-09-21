#!/usr/bin/env python3
"""
Recursive Min/Max Analysis for Intelligent Point Classification

This approach recursively identifies absolute min/max points within segments,
classifies intermediate points by their structural importance, and determines
what can be removed vs what should be preserved.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum


class PointType(Enum):
    """Classification of point types in the signal hierarchy."""
    ABSOLUTE_EXTREMUM = "absolute_extremum"      # Global min/max
    LOCAL_EXTREMUM = "local_extremum"            # Segment min/max
    STRUCTURAL_POINT = "structural_point"        # Important for signal shape
    TRANSITION_POINT = "transition_point"        # Between extrema
    REDUNDANT_POINT = "redundant_point"          # Can be removed/smoothed
    NOISE_POINT = "noise_point"                  # Should be removed


@dataclass
class PointAnalysis:
    """Analysis data for a single point."""
    index: int
    time: float
    position: float
    point_type: PointType
    importance_score: float
    segment_level: int
    is_removable: bool
    removal_reason: str
    parent_segment: Optional['SegmentAnalysis'] = None


@dataclass
class SegmentAnalysis:
    """Analysis data for a signal segment."""
    start_idx: int
    end_idx: int
    min_idx: int
    max_idx: int
    segment_level: int
    importance_score: float
    subsegments: List['SegmentAnalysis']
    points: List[PointAnalysis]


class RecursiveMinMaxAnalyzer:
    """
    Recursively analyze signal segments to identify hierarchical structure
    and classify points by their structural importance.
    """
    
    def __init__(self, min_segment_size: int = 3, max_recursion_depth: int = 8):
        self.min_segment_size = min_segment_size
        self.max_recursion_depth = max_recursion_depth
    
    def analyze_signal(self, times: np.ndarray, positions: np.ndarray) -> Dict[str, Any]:
        """
        Perform complete recursive analysis of the signal.
        
        Args:
            times: Time values
            positions: Position values
            
        Returns:
            Complete analysis results
        """
        if len(positions) < self.min_segment_size:
            return self._create_minimal_analysis(times, positions)
        
        # Start recursive analysis with the entire signal
        root_segment = self._analyze_segment(times, positions, 0, len(positions) - 1, 0)
        
        # Classify all points based on hierarchical analysis
        point_classifications = self._classify_all_points(root_segment, times, positions)
        
        # Determine removable points
        removable_analysis = self._determine_removable_points(point_classifications, times, positions)
        
        return {
            'root_segment': root_segment,
            'point_classifications': point_classifications,
            'removable_points': removable_analysis['removable_indices'],
            'removal_reasons': removable_analysis['removal_reasons'],
            'keep_points': removable_analysis['keep_indices'],
            'analysis_summary': self._create_analysis_summary(point_classifications, times, positions)
        }
    
    def _analyze_segment(self, times: np.ndarray, positions: np.ndarray, 
                        start_idx: int, end_idx: int, level: int) -> SegmentAnalysis:
        """
        Recursively analyze a signal segment.
        
        Args:
            times: Time values
            positions: Position values  
            start_idx: Segment start index
            end_idx: Segment end index
            level: Recursion level
            
        Returns:
            Segment analysis
        """
        segment_positions = positions[start_idx:end_idx + 1]
        
        # Find absolute min and max within this segment
        local_min_idx = np.argmin(segment_positions)
        local_max_idx = np.argmax(segment_positions)
        
        # Convert to global indices
        global_min_idx = start_idx + local_min_idx
        global_max_idx = start_idx + local_max_idx
        
        # Calculate segment importance based on range and duration
        position_range = np.ptp(segment_positions)
        time_duration = times[end_idx] - times[start_idx]
        importance_score = position_range * (1 + np.log(time_duration + 1))
        
        segment = SegmentAnalysis(
            start_idx=start_idx,
            end_idx=end_idx,
            min_idx=global_min_idx,
            max_idx=global_max_idx,
            segment_level=level,
            importance_score=importance_score,
            subsegments=[],
            points=[]
        )
        
        # Recursively analyze subsegments if conditions are met
        if (level < self.max_recursion_depth and 
            end_idx - start_idx >= self.min_segment_size * 2):
            
            # Create subsegments between key points
            subsegment_boundaries = self._determine_subsegment_boundaries(
                start_idx, end_idx, global_min_idx, global_max_idx
            )
            
            for sub_start, sub_end in subsegment_boundaries:
                if sub_end - sub_start >= self.min_segment_size:
                    subsegment = self._analyze_segment(times, positions, sub_start, sub_end, level + 1)
                    segment.subsegments.append(subsegment)
        
        return segment
    
    def _determine_subsegment_boundaries(self, start_idx: int, end_idx: int, 
                                       min_idx: int, max_idx: int) -> List[Tuple[int, int]]:
        """
        Determine how to divide a segment into subsegments based on extrema.
        
        Args:
            start_idx: Segment start
            end_idx: Segment end
            min_idx: Global minimum index
            max_idx: Global maximum index
            
        Returns:
            List of (start, end) tuples for subsegments
        """
        boundaries = []
        
        # Order the extrema by position
        extrema = sorted([min_idx, max_idx])
        first_extremum = extrema[0]
        second_extremum = extrema[1]
        
        # Create segments: start->first_extremum, first->second, second->end
        if first_extremum > start_idx + 1:
            boundaries.append((start_idx, first_extremum))
        
        if second_extremum > first_extremum + 1:
            boundaries.append((first_extremum, second_extremum))
        
        if end_idx > second_extremum + 1:
            boundaries.append((second_extremum, end_idx))
        
        return boundaries
    
    def _classify_all_points(self, root_segment: SegmentAnalysis, 
                           times: np.ndarray, positions: np.ndarray) -> List[PointAnalysis]:
        """
        Classify all points based on the hierarchical segment analysis.
        
        Args:
            root_segment: Root segment analysis
            times: Time values
            positions: Position values
            
        Returns:
            List of point classifications
        """
        n_points = len(positions)
        classifications = []
        
        # Initialize all points
        for i in range(n_points):
            classifications.append(PointAnalysis(
                index=i,
                time=times[i],
                position=positions[i],
                point_type=PointType.TRANSITION_POINT,  # Default
                importance_score=0.0,
                segment_level=999,  # Will be updated
                is_removable=True,  # Will be updated
                removal_reason="Not analyzed"
            ))
        
        # Recursively classify points based on segment hierarchy
        self._classify_points_in_segment(root_segment, classifications, times, positions)
        
        # Post-process classifications
        self._refine_point_classifications(classifications, times, positions)
        
        return classifications
    
    def _classify_points_in_segment(self, segment: SegmentAnalysis, 
                                  classifications: List[PointAnalysis],
                                  times: np.ndarray, positions: np.ndarray):
        """
        Recursively classify points within a segment.
        
        Args:
            segment: Segment to analyze
            classifications: Point classifications to update
            times: Time values
            positions: Position values
        """
        # Classify extrema in this segment
        min_point = classifications[segment.min_idx]
        max_point = classifications[segment.max_idx]
        
        # Update classification if this is a higher-level (more important) segment
        if segment.segment_level < min_point.segment_level:
            min_point.segment_level = segment.segment_level
            max_point.segment_level = segment.segment_level
            
            # Determine point type based on level
            if segment.segment_level == 0:
                min_point.point_type = PointType.ABSOLUTE_EXTREMUM
                max_point.point_type = PointType.ABSOLUTE_EXTREMUM
            else:
                min_point.point_type = PointType.LOCAL_EXTREMUM
                max_point.point_type = PointType.LOCAL_EXTREMUM
            
            # Calculate importance scores
            min_point.importance_score = segment.importance_score + 100 * (10 - segment.segment_level)
            max_point.importance_score = segment.importance_score + 100 * (10 - segment.segment_level)
            
            # These points are never removable
            min_point.is_removable = False
            max_point.is_removable = False
            min_point.removal_reason = "Extremum point"
            max_point.removal_reason = "Extremum point"
        
        # Analyze intermediate points in this segment
        self._analyze_intermediate_points_in_segment(segment, classifications, times, positions)
        
        # Recursively process subsegments
        for subsegment in segment.subsegments:
            self._classify_points_in_segment(subsegment, classifications, times, positions)
    
    def _analyze_intermediate_points_in_segment(self, segment: SegmentAnalysis,
                                              classifications: List[PointAnalysis],
                                              times: np.ndarray, positions: np.ndarray):
        """
        Analyze intermediate points between extrema in a segment.
        
        Args:
            segment: Current segment
            classifications: Point classifications to update
            times: Time values
            positions: Position values
        """
        for i in range(segment.start_idx, segment.end_idx + 1):
            if i == segment.min_idx or i == segment.max_idx:
                continue  # Skip extrema
            
            point = classifications[i]
            
            # Analyze the point's role in the segment
            point_analysis = self._analyze_point_role(i, segment, times, positions)
            
            # Update classification if this provides more insight
            if point_analysis['importance_score'] > point.importance_score:
                point.importance_score = point_analysis['importance_score']
                point.point_type = point_analysis['point_type']
                point.segment_level = min(point.segment_level, segment.segment_level + 1)
                
                # Determine if removable
                point.is_removable = point_analysis['is_removable']
                point.removal_reason = point_analysis['removal_reason']
    
    def _analyze_point_role(self, point_idx: int, segment: SegmentAnalysis,
                          times: np.ndarray, positions: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the specific role of a point within its segment.
        
        Args:
            point_idx: Index of point to analyze
            segment: Containing segment
            times: Time values
            positions: Position values
            
        Returns:
            Analysis of point's role
        """
        # Get point's position relative to segment extrema
        point_pos = positions[point_idx]
        min_pos = positions[segment.min_idx]
        max_pos = positions[segment.max_idx]
        
        # Normalize position within segment range
        if max_pos != min_pos:
            normalized_pos = (point_pos - min_pos) / (max_pos - min_pos)
        else:
            normalized_pos = 0.5
        
        # Analyze trend consistency
        trend_consistency = self._calculate_trend_consistency(point_idx, segment, positions)
        
        # Analyze contribution to signal shape
        shape_contribution = self._calculate_shape_contribution(point_idx, segment, positions)
        
        # Calculate overall importance
        importance_score = (trend_consistency * 50 + 
                          shape_contribution * 30 + 
                          abs(0.5 - normalized_pos) * 20)  # Distance from middle
        
        # Determine point type and removability
        if importance_score > 80:
            point_type = PointType.STRUCTURAL_POINT
            is_removable = False
            removal_reason = "Structurally important"
        elif importance_score > 50:
            point_type = PointType.TRANSITION_POINT
            is_removable = False
            removal_reason = "Important transition"
        elif importance_score > 20:
            point_type = PointType.REDUNDANT_POINT
            is_removable = True
            removal_reason = "Redundant - can be interpolated"
        else:
            point_type = PointType.NOISE_POINT
            is_removable = True
            removal_reason = "Likely noise - should be removed"
        
        return {
            'importance_score': importance_score,
            'point_type': point_type,
            'is_removable': is_removable,
            'removal_reason': removal_reason,
            'trend_consistency': trend_consistency,
            'shape_contribution': shape_contribution
        }
    
    def _calculate_trend_consistency(self, point_idx: int, segment: SegmentAnalysis, 
                                   positions: np.ndarray) -> float:
        """Calculate how consistent a point is with the overall trend."""
        if point_idx <= segment.start_idx or point_idx >= segment.end_idx:
            return 0.0
        
        # Look at slope before and after point
        prev_pos = positions[point_idx - 1]
        curr_pos = positions[point_idx]
        next_pos = positions[point_idx + 1]
        
        slope_before = curr_pos - prev_pos
        slope_after = next_pos - curr_pos
        
        # Check if slopes are consistent (same direction)
        if slope_before * slope_after >= 0:
            return 100.0  # Consistent trend
        else:
            # Direction change - check if it's significant
            change_magnitude = abs(slope_before) + abs(slope_after)
            if change_magnitude > 20:  # Significant direction change
                return 100.0  # Important inflection point
            else:
                return 20.0   # Minor oscillation
    
    def _calculate_shape_contribution(self, point_idx: int, segment: SegmentAnalysis,
                                    positions: np.ndarray) -> float:
        """Calculate how much a point contributes to the overall signal shape."""
        if point_idx <= segment.start_idx + 1 or point_idx >= segment.end_idx - 1:
            return 100.0  # Boundary points are important
        
        # Calculate how much the signal would change if this point was interpolated
        prev_pos = positions[point_idx - 1]
        curr_pos = positions[point_idx]
        next_pos = positions[point_idx + 1]
        
        # Linear interpolation
        interpolated_pos = (prev_pos + next_pos) / 2
        deviation = abs(curr_pos - interpolated_pos)
        
        # Higher deviation means more shape contribution
        return min(100.0, deviation * 5)  # Scale factor
    
    def _refine_point_classifications(self, classifications: List[PointAnalysis],
                                    times: np.ndarray, positions: np.ndarray):
        """Refine point classifications with additional rules."""
        n_points = len(classifications)
        
        # Always preserve first and last points
        if n_points > 0:
            classifications[0].is_removable = False
            classifications[0].removal_reason = "First point"
            classifications[0].point_type = PointType.STRUCTURAL_POINT
            
            classifications[-1].is_removable = False
            classifications[-1].removal_reason = "Last point"
            classifications[-1].point_type = PointType.STRUCTURAL_POINT
        
        # Look for consecutive removable points and preserve some for smoothness
        removable_runs = self._find_consecutive_removable_runs(classifications)
        for start, end in removable_runs:
            if end - start > 3:  # Long run of removable points
                # Preserve one point in the middle for smoothness
                mid_idx = start + (end - start) // 2
                classifications[mid_idx].is_removable = False
                classifications[mid_idx].removal_reason = "Preserved for smoothness"
                classifications[mid_idx].point_type = PointType.TRANSITION_POINT
    
    def _find_consecutive_removable_runs(self, classifications: List[PointAnalysis]) -> List[Tuple[int, int]]:
        """Find runs of consecutive removable points."""
        runs = []
        start = None
        
        for i, point in enumerate(classifications):
            if point.is_removable:
                if start is None:
                    start = i
            else:
                if start is not None:
                    runs.append((start, i))
                    start = None
        
        # Handle run at the end
        if start is not None:
            runs.append((start, len(classifications)))
        
        return runs
    
    def _determine_removable_points(self, classifications: List[PointAnalysis],
                                  times: np.ndarray, positions: np.ndarray) -> Dict[str, Any]:
        """Determine final list of removable points and reasons."""
        removable_indices = []
        removal_reasons = {}
        keep_indices = []
        
        for point in classifications:
            if point.is_removable:
                removable_indices.append(point.index)
                removal_reasons[point.index] = point.removal_reason
            else:
                keep_indices.append(point.index)
        
        return {
            'removable_indices': removable_indices,
            'removal_reasons': removal_reasons,
            'keep_indices': keep_indices
        }
    
    def _create_analysis_summary(self, classifications: List[PointAnalysis],
                               times: np.ndarray, positions: np.ndarray) -> Dict[str, Any]:
        """Create a summary of the analysis results."""
        total_points = len(classifications)
        removable_count = sum(1 for p in classifications if p.is_removable)
        
        point_type_counts = {}
        for point_type in PointType:
            count = sum(1 for p in classifications if p.point_type == point_type)
            point_type_counts[point_type.value] = count
        
        return {
            'total_points': total_points,
            'removable_points': removable_count,
            'keep_points': total_points - removable_count,
            'removal_percentage': (removable_count / total_points * 100) if total_points > 0 else 0,
            'point_type_distribution': point_type_counts,
            'analysis_depth': max((p.segment_level for p in classifications), default=0)
        }
    
    def _create_minimal_analysis(self, times: np.ndarray, positions: np.ndarray) -> Dict[str, Any]:
        """Create minimal analysis for very short signals."""
        n_points = len(positions)
        classifications = []
        
        for i in range(n_points):
            classifications.append(PointAnalysis(
                index=i,
                time=times[i],
                position=positions[i],
                point_type=PointType.STRUCTURAL_POINT,
                importance_score=100.0,
                segment_level=0,
                is_removable=False,
                removal_reason="Signal too short"
            ))
        
        return {
            'root_segment': None,
            'point_classifications': classifications,
            'removable_points': [],
            'removal_reasons': {},
            'keep_points': list(range(n_points)),
            'analysis_summary': {
                'total_points': n_points,
                'removable_points': 0,
                'keep_points': n_points,
                'removal_percentage': 0.0,
                'point_type_distribution': {'structural_point': n_points},
                'analysis_depth': 0
            }
        }


def test_recursive_analysis():
    """Test the recursive min/max analysis on the sparse funscript data."""
    print("🔍 RECURSIVE MIN/MAX ANALYSIS")
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
    
    # Perform recursive analysis
    analyzer = RecursiveMinMaxAnalyzer(min_segment_size=3, max_recursion_depth=6)
    analysis = analyzer.analyze_signal(times, positions)
    
    # Print results
    print(f"📊 ANALYSIS SUMMARY:")
    summary = analysis['analysis_summary']
    print(f"Total points: {summary['total_points']}")
    print(f"Keep points: {summary['keep_points']} ({100 - summary['removal_percentage']:.1f}%)")
    print(f"Removable points: {summary['removable_points']} ({summary['removal_percentage']:.1f}%)")
    print(f"Analysis depth: {summary['analysis_depth']} levels")
    
    print(f"\n📋 POINT TYPE DISTRIBUTION:")
    for point_type, count in summary['point_type_distribution'].items():
        print(f"  {point_type}: {count}")
    
    print(f"\n🎯 POINTS TO KEEP:")
    classifications = analysis['point_classifications']
    for i in analysis['keep_points']:
        point = classifications[i]
        print(f"  Point {i}: t={point.time}ms, pos={point.position}, type={point.point_type.value}, "
              f"score={point.importance_score:.1f}, level={point.segment_level}")
    
    print(f"\n❌ POINTS TO REMOVE:")
    for i in analysis['removable_points']:
        point = classifications[i]
        reason = analysis['removal_reasons'][i]
        print(f"  Point {i}: t={point.time}ms, pos={point.position}, reason='{reason}', "
              f"type={point.point_type.value}")
    
    # Create visualization
    create_recursive_analysis_plot(times, positions, analysis)
    
    print(f"\n📊 Recursive analysis plot saved as 'recursive_analysis_results.png'")
    
    return analysis


def create_recursive_analysis_plot(times, positions, analysis):
    """Create visualization of the recursive analysis."""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        times_sec = times / 1000.0
        classifications = analysis['point_classifications']
        
        # Plot 1: Original signal with point classifications
        colors = {
            PointType.ABSOLUTE_EXTREMUM: 'red',
            PointType.LOCAL_EXTREMUM: 'orange', 
            PointType.STRUCTURAL_POINT: 'green',
            PointType.TRANSITION_POINT: 'blue',
            PointType.REDUNDANT_POINT: 'gray',
            PointType.NOISE_POINT: 'black'
        }
        
        # Plot signal
        ax1.plot(times_sec, positions, 'k-', alpha=0.3, linewidth=1)
        
        # Plot points by type
        for point_type in PointType:
            type_indices = [p.index for p in classifications if p.point_type == point_type]
            if type_indices:
                ax1.scatter(times_sec[type_indices], positions[type_indices],
                          c=colors[point_type], s=60, alpha=0.8, 
                          label=f'{point_type.value} ({len(type_indices)})', zorder=5)
        
        ax1.set_title('Recursive Analysis: Point Classifications')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Position')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Keep vs Remove
        keep_indices = analysis['keep_points']
        remove_indices = analysis['removable_points']
        
        ax2.plot(times_sec, positions, 'k-', alpha=0.3, linewidth=1)
        ax2.scatter(times_sec[keep_indices], positions[keep_indices],
                   c='green', s=60, alpha=0.8, label=f'Keep ({len(keep_indices)})', zorder=5)
        ax2.scatter(times_sec[remove_indices], positions[remove_indices],
                   c='red', s=40, marker='x', alpha=0.8, label=f'Remove ({len(remove_indices)})', zorder=5)
        
        ax2.set_title('Keep vs Remove Decision')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Position')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Importance scores
        importance_scores = [p.importance_score for p in classifications]
        ax3.bar(range(len(importance_scores)), importance_scores, 
               color=['green' if not p.is_removable else 'red' for p in classifications],
               alpha=0.7)
        ax3.set_title('Point Importance Scores')
        ax3.set_xlabel('Point Index')
        ax3.set_ylabel('Importance Score')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Segment levels
        segment_levels = [p.segment_level for p in classifications]
        ax4.bar(range(len(segment_levels)), segment_levels,
               color=['green' if not p.is_removable else 'red' for p in classifications],
               alpha=0.7)
        ax4.set_title('Hierarchical Segment Levels')
        ax4.set_xlabel('Point Index')
        ax4.set_ylabel('Segment Level (0=root)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('recursive_analysis_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Could not create recursive analysis plot: {e}")


if __name__ == "__main__":
    test_recursive_analysis()