#!/usr/bin/env python3
"""
Aggressive Point Pruning Algorithm

This approach is more aggressive about identifying redundant points by:
1. Finding true extrema (global and local)
2. Identifying intermediate points that don't contribute meaningfully to the signal shape
3. Using interpolation error analysis to determine if points can be safely removed
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from scipy.interpolate import interp1d


@dataclass
class PointClassification:
    """Classification of a single point."""
    index: int
    time: float
    position: float
    is_global_extremum: bool
    is_local_extremum: bool
    is_inflection_point: bool
    interpolation_error: float
    structural_importance: float
    is_removable: bool
    removal_reason: str


class AggressivePointPruner:
    """
    Aggressively identify removable points while preserving signal structure.
    """
    
    def __init__(self, interpolation_tolerance: float = 5.0, 
                 min_structural_importance: float = 20.0):
        self.interpolation_tolerance = interpolation_tolerance
        self.min_structural_importance = min_structural_importance
    
    def analyze_and_prune(self, times: np.ndarray, positions: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the signal and identify which points can be safely removed.
        
        Args:
            times: Time values
            positions: Position values
            
        Returns:
            Analysis results with removable points identified
        """
        if len(positions) < 4:
            return self._create_minimal_result(times, positions)
        
        # Step 1: Find global extrema
        global_min_idx = np.argmin(positions)
        global_max_idx = np.argmax(positions)
        
        # Step 2: Find local extrema using a more aggressive approach
        local_extrema = self._find_local_extrema(positions)
        
        # Step 3: Find inflection points (direction changes)
        inflection_points = self._find_inflection_points(positions)
        
        # Step 4: Analyze each point's structural importance
        point_classifications = []
        for i in range(len(positions)):
            classification = self._classify_point(
                i, times, positions, global_min_idx, global_max_idx,
                local_extrema, inflection_points
            )
            point_classifications.append(classification)
        
        # Step 5: Apply interpolation analysis to determine removability
        self._apply_interpolation_analysis(point_classifications, times, positions)
        
        # Step 6: Apply safety rules to prevent over-pruning
        self._apply_safety_rules(point_classifications)
        
        # Step 7: Create final results
        return self._create_results(point_classifications, times, positions)
    
    def _find_local_extrema(self, positions: np.ndarray, window_size: int = 3) -> List[int]:
        """Find local extrema using a sliding window approach."""
        extrema = []
        n = len(positions)
        
        for i in range(window_size, n - window_size):
            window = positions[i - window_size:i + window_size + 1]
            center_value = positions[i]
            
            # Check if this is a local maximum
            if center_value == np.max(window) and center_value > np.mean(window):
                extrema.append(i)
            # Check if this is a local minimum
            elif center_value == np.min(window) and center_value < np.mean(window):
                extrema.append(i)
        
        return extrema
    
    def _find_inflection_points(self, positions: np.ndarray) -> List[int]:
        """Find inflection points where the signal changes direction significantly."""
        inflection_points = []
        n = len(positions)
        
        if n < 3:
            return inflection_points
        
        for i in range(1, n - 1):
            prev_pos = positions[i - 1]
            curr_pos = positions[i]
            next_pos = positions[i + 1]
            
            # Calculate slopes
            slope_before = curr_pos - prev_pos
            slope_after = next_pos - curr_pos
            
            # Check for direction change
            if slope_before * slope_after < 0:  # Different signs = direction change
                # Only consider significant direction changes
                change_magnitude = abs(slope_before) + abs(slope_after)
                if change_magnitude > 10:  # Threshold for significance
                    inflection_points.append(i)
        
        return inflection_points
    
    def _classify_point(self, idx: int, times: np.ndarray, positions: np.ndarray,
                       global_min_idx: int, global_max_idx: int,
                       local_extrema: List[int], inflection_points: List[int]) -> PointClassification:
        """Classify a single point."""
        
        # Basic classification
        is_global_extremum = idx in [global_min_idx, global_max_idx]
        is_local_extremum = idx in local_extrema
        is_inflection_point = idx in inflection_points
        
        # Calculate structural importance
        structural_importance = 0.0
        
        if is_global_extremum:
            structural_importance += 100.0
        elif is_local_extremum:
            structural_importance += 60.0
        elif is_inflection_point:
            structural_importance += 40.0
        
        # Add importance based on position
        if idx == 0 or idx == len(positions) - 1:
            structural_importance += 50.0  # Endpoints
        
        # Add importance based on deviation from neighbors
        if 0 < idx < len(positions) - 1:
            neighbor_deviation = self._calculate_neighbor_deviation(idx, positions)
            structural_importance += neighbor_deviation
        
        # Initial removability assessment
        is_removable = (not is_global_extremum and 
                       structural_importance < self.min_structural_importance)
        
        removal_reason = ""
        if is_global_extremum:
            removal_reason = "Global extremum - never removable"
        elif is_local_extremum:
            removal_reason = "Local extremum - structurally important"
        elif is_inflection_point:
            removal_reason = "Inflection point - direction change"
        elif idx == 0 or idx == len(positions) - 1:
            removal_reason = "Endpoint - never removable"
        elif structural_importance < self.min_structural_importance:
            removal_reason = "Low structural importance - candidate for removal"
        else:
            removal_reason = "Structurally important"
        
        return PointClassification(
            index=idx,
            time=times[idx],
            position=positions[idx],
            is_global_extremum=is_global_extremum,
            is_local_extremum=is_local_extremum,
            is_inflection_point=is_inflection_point,
            interpolation_error=0.0,  # Will be calculated later
            structural_importance=structural_importance,
            is_removable=is_removable,
            removal_reason=removal_reason
        )
    
    def _calculate_neighbor_deviation(self, idx: int, positions: np.ndarray) -> float:
        """Calculate how much a point deviates from its neighbors."""
        if idx <= 0 or idx >= len(positions) - 1:
            return 0.0
        
        prev_pos = positions[idx - 1]
        curr_pos = positions[idx]
        next_pos = positions[idx + 1]
        
        # Expected position if linearly interpolated
        expected_pos = (prev_pos + next_pos) / 2
        deviation = abs(curr_pos - expected_pos)
        
        return min(deviation * 2, 50.0)  # Cap at 50
    
    def _apply_interpolation_analysis(self, classifications: List[PointClassification],
                                    times: np.ndarray, positions: np.ndarray):
        """Apply interpolation analysis to refine removability decisions."""
        n = len(classifications)
        
        for i, point in enumerate(classifications):
            if point.is_global_extremum or i == 0 or i == n - 1:
                point.interpolation_error = 0.0
                continue
            
            # Calculate interpolation error if this point were removed
            error = self._calculate_interpolation_error(i, times, positions)
            point.interpolation_error = error
            
            # Update removability based on interpolation error
            if error > self.interpolation_tolerance:
                point.is_removable = False
                point.removal_reason = f"High interpolation error ({error:.1f})"
            elif error <= self.interpolation_tolerance and point.structural_importance < 30:
                point.is_removable = True
                point.removal_reason = f"Low interpolation error ({error:.1f}) - safe to remove"
    
    def _calculate_interpolation_error(self, remove_idx: int, times: np.ndarray, 
                                     positions: np.ndarray) -> float:
        """Calculate the interpolation error if a point were removed."""
        n = len(positions)
        
        # Create arrays without the point to be removed
        keep_indices = [i for i in range(n) if i != remove_idx]
        keep_times = times[keep_indices]
        keep_positions = positions[keep_indices]
        
        if len(keep_times) < 2:
            return float('inf')  # Can't interpolate
        
        try:
            # Create interpolation function
            interp_func = interp1d(keep_times, keep_positions, kind='linear', 
                                 bounds_error=False, fill_value='extrapolate')
            
            # Calculate interpolated value at the removed point's time
            interpolated_value = interp_func(times[remove_idx])
            
            # Calculate error
            actual_value = positions[remove_idx]
            error = abs(actual_value - interpolated_value)
            
            return error
            
        except Exception:
            return float('inf')  # Interpolation failed
    
    def _apply_safety_rules(self, classifications: List[PointClassification]):
        """Apply safety rules to prevent over-pruning."""
        n = len(classifications)
        
        # Rule 1: Never remove more than 50% of points
        removable_count = sum(1 for p in classifications if p.is_removable)
        if removable_count > n // 2:
            # Keep the most structurally important removable points
            removable_points = [p for p in classifications if p.is_removable]
            removable_points.sort(key=lambda p: p.structural_importance, reverse=True)
            
            points_to_keep = removable_count - n // 2
            for i in range(points_to_keep):
                point = removable_points[i]
                point.is_removable = False
                point.removal_reason = "Kept for safety (50% rule)"
        
        # Rule 2: Don't create long gaps without points
        self._prevent_long_gaps(classifications)
        
        # Rule 3: Preserve pattern regularity
        self._preserve_pattern_regularity(classifications)
    
    def _prevent_long_gaps(self, classifications: List[PointClassification]):
        """Prevent creating long gaps by removing too many consecutive points."""
        n = len(classifications)
        max_consecutive_removals = 3
        
        consecutive_removals = 0
        for i, point in enumerate(classifications):
            if point.is_removable:
                consecutive_removals += 1
                if consecutive_removals > max_consecutive_removals:
                    # Keep this point to break the consecutive removal chain
                    point.is_removable = False
                    point.removal_reason = "Kept to prevent long gap"
                    consecutive_removals = 0
            else:
                consecutive_removals = 0
    
    def _preserve_pattern_regularity(self, classifications: List[PointClassification]):
        """Preserve some regularity in the pattern."""
        # If we have alternating high-low pattern, preserve some intermediate points
        positions = [p.position for p in classifications]
        
        # Detect if we have a regular alternating pattern
        alternations = 0
        for i in range(len(positions) - 2):
            if ((positions[i] < positions[i+1] > positions[i+2]) or
                (positions[i] > positions[i+1] < positions[i+2])):
                alternations += 1
        
        alternation_ratio = alternations / max(1, len(positions) - 2)
        
        # If highly alternating, be more conservative about removal
        if alternation_ratio > 0.6:  # More than 60% alternating
            for point in classifications:
                if point.is_removable and point.structural_importance > 15:
                    point.is_removable = False
                    point.removal_reason = "Preserved for pattern regularity"
    
    def _create_results(self, classifications: List[PointClassification],
                       times: np.ndarray, positions: np.ndarray) -> Dict[str, Any]:
        """Create final results dictionary."""
        removable_indices = [p.index for p in classifications if p.is_removable]
        keep_indices = [p.index for p in classifications if not p.is_removable]
        
        # Create pruned signal
        if removable_indices:
            pruned_times = times[keep_indices]
            pruned_positions = positions[keep_indices]
        else:
            pruned_times = times.copy()
            pruned_positions = positions.copy()
        
        # Calculate statistics
        total_points = len(classifications)
        removed_count = len(removable_indices)
        removal_percentage = (removed_count / total_points * 100) if total_points > 0 else 0
        
        return {
            'classifications': classifications,
            'removable_indices': removable_indices,
            'keep_indices': keep_indices,
            'pruned_times': pruned_times,
            'pruned_positions': pruned_positions,
            'statistics': {
                'total_points': total_points,
                'removed_points': removed_count,
                'kept_points': len(keep_indices),
                'removal_percentage': removal_percentage
            }
        }
    
    def _create_minimal_result(self, times: np.ndarray, positions: np.ndarray) -> Dict[str, Any]:
        """Create minimal result for very short signals."""
        n = len(positions)
        classifications = []
        
        for i in range(n):
            classifications.append(PointClassification(
                index=i,
                time=times[i],
                position=positions[i],
                is_global_extremum=(i == 0 or i == n-1),
                is_local_extremum=False,
                is_inflection_point=False,
                interpolation_error=0.0,
                structural_importance=100.0,
                is_removable=False,
                removal_reason="Signal too short"
            ))
        
        return {
            'classifications': classifications,
            'removable_indices': [],
            'keep_indices': list(range(n)),
            'pruned_times': times.copy(),
            'pruned_positions': positions.copy(),
            'statistics': {
                'total_points': n,
                'removed_points': 0,
                'kept_points': n,
                'removal_percentage': 0.0
            }
        }


def test_aggressive_pruning():
    """Test the aggressive point pruning on the sparse funscript data."""
    print("🗡️ AGGRESSIVE POINT PRUNING ANALYSIS")
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
    
    # Test different tolerance levels
    tolerance_levels = [3.0, 5.0, 10.0, 15.0]
    
    print(f"📊 ORIGINAL SIGNAL:")
    print(f"Total points: {len(positions)}")
    original_movements = np.abs(np.diff(positions))
    extreme_jumps = np.sum(original_movements > 70)
    print(f"Extreme jumps (>70): {extreme_jumps}")
    
    results = {}
    
    for tolerance in tolerance_levels:
        print(f"\n🔧 Testing with interpolation tolerance = {tolerance}")
        
        pruner = AggressivePointPruner(
            interpolation_tolerance=tolerance,
            min_structural_importance=20.0
        )
        
        result = pruner.analyze_and_prune(times, positions)
        results[tolerance] = result
        
        stats = result['statistics']
        print(f"   Removed: {stats['removed_points']}/{stats['total_points']} ({stats['removal_percentage']:.1f}%)")
        print(f"   Kept: {stats['kept_points']} points")
        
        # Calculate extreme jumps in pruned signal
        if len(result['pruned_positions']) > 1:
            pruned_movements = np.abs(np.diff(result['pruned_positions']))
            pruned_extreme_jumps = np.sum(pruned_movements > 70)
            print(f"   Extreme jumps after pruning: {pruned_extreme_jumps}")
        
        # Show which points would be removed
        if result['removable_indices']:
            print(f"   Points to remove: {result['removable_indices']}")
            for idx in result['removable_indices'][:5]:  # Show first 5
                point = result['classifications'][idx]
                print(f"     Point {idx}: pos={point.position}, reason='{point.removal_reason}'")
    
    # Create visualization
    create_aggressive_pruning_plot(times, positions, results)
    
    print(f"\n📊 Aggressive pruning plot saved as 'aggressive_pruning_results.png'")
    
    return results


def create_aggressive_pruning_plot(times, positions, results):
    """Create visualization of the aggressive pruning analysis."""
    try:
        num_tolerances = len(results)
        fig, axes = plt.subplots(num_tolerances + 1, 1, figsize=(16, (num_tolerances + 1) * 3))
        
        if num_tolerances == 1:
            axes = [axes]
        
        times_sec = times / 1000.0
        
        # Plot original signal
        axes[0].plot(times_sec, positions, 'r-o', markersize=6, linewidth=2, 
                    alpha=0.8, label='Original Signal')
        axes[0].set_title('Original Sparse Signal', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Position (0-100)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[0].set_ylim(-5, 105)
        
        # Highlight extreme jumps
        movements = np.abs(np.diff(positions))
        intervals = np.diff(times)
        for i, movement in enumerate(movements):
            if movement > 70:
                axes[0].axvspan(times_sec[i], times_sec[i+1], alpha=0.3, color='red')
        
        # Plot each tolerance level
        colors = ['blue', 'green', 'purple', 'orange']
        
        for idx, (tolerance, result) in enumerate(results.items()):
            ax = axes[idx + 1]
            
            # Plot original points
            ax.plot(times_sec, positions, 'r-', linewidth=1, alpha=0.4, label='Original')
            ax.scatter(times_sec, positions, c='red', s=30, alpha=0.5)
            
            # Plot kept points
            keep_indices = result['keep_indices']
            ax.scatter(times_sec[keep_indices], positions[keep_indices],
                      c=colors[idx % len(colors)], s=60, alpha=0.8, 
                      label=f'Kept ({len(keep_indices)})', zorder=5)
            
            # Plot removed points
            remove_indices = result['removable_indices']
            if remove_indices:
                ax.scatter(times_sec[remove_indices], positions[remove_indices],
                          c='black', s=40, marker='x', alpha=0.8,
                          label=f'Removed ({len(remove_indices)})', zorder=5)
            
            # Plot pruned signal path
            pruned_times_sec = result['pruned_times'] / 1000.0
            ax.plot(pruned_times_sec, result['pruned_positions'],
                   color=colors[idx % len(colors)], linewidth=3, alpha=0.7,
                   label='Pruned signal', zorder=3)
            
            stats = result['statistics']
            title = f"Tolerance = {tolerance} | Removed {stats['removal_percentage']:.1f}% | "
            title += f"({stats['kept_points']} points kept)"
            
            ax.set_title(title, fontsize=12)
            ax.set_ylabel('Position')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_ylim(-5, 105)
        
        # Set xlabel on last plot
        axes[-1].set_xlabel('Time (seconds)')
        
        plt.tight_layout()
        plt.savefig('aggressive_pruning_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Could not create aggressive pruning plot: {e}")


if __name__ == "__main__":
    test_aggressive_pruning()