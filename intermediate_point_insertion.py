#!/usr/bin/env python3
"""
Intelligent Intermediate Point Insertion for Anti-Jerk Filtering

Instead of removing/smoothing existing points (which are all meaningful in sparse data),
this approach adds intermediate points to create smoother transitions while preserving 
the original extreme points.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d
from typing import List, Tuple, Dict, Any


class IntermediatePointInserter:
    """
    Inserts intermediate points to create smoother transitions between extreme movements.
    
    Key strategy:
    1. Keep all original points (they're all meaningful)
    2. Add intermediate points for jerky transitions
    3. Use intelligent interpolation to maintain the intended movement character
    """
    
    def __init__(self):
        pass
    
    def analyze_transition_jerkiness(self, times: np.ndarray, positions: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Analyze which transitions are jerky and need intermediate points.
        
        Args:
            times: Time values in ms
            positions: Position values 0-100
            
        Returns:
            Dictionary with jerkiness analysis
        """
        n = len(positions)
        
        if n < 2:
            return {'jerky_transitions': np.array([], dtype=bool)}
        
        # Calculate transition characteristics
        time_intervals = np.diff(times)
        position_changes = np.abs(np.diff(positions))
        
        # Calculate "jerkiness" - large position change in short time
        jerkiness_scores = np.zeros(n-1)
        for i in range(n-1):
            pos_change = position_changes[i]
            time_interval = time_intervals[i]
            
            # Jerkiness score: position change rate with penalty for extreme jumps
            if time_interval > 0:
                change_rate = pos_change / (time_interval / 1000.0)  # pos/second
                
                # Extra penalty for very large jumps
                extreme_penalty = 1.0
                if pos_change > 70:
                    extreme_penalty = 2.0
                elif pos_change > 90:
                    extreme_penalty = 3.0
                
                jerkiness_scores[i] = change_rate * extreme_penalty
        
        # Determine jerky transitions
        jerkiness_threshold = np.percentile(jerkiness_scores, 70) if len(jerkiness_scores) > 0 else 0
        jerky_transitions = jerkiness_scores > jerkiness_threshold
        
        # Force certain conditions to be jerky
        for i in range(n-1):
            # Large position change in short time
            if position_changes[i] > 70 and time_intervals[i] < 250:
                jerky_transitions[i] = True
        
        return {
            'jerkiness_scores': jerkiness_scores,
            'jerky_transitions': jerky_transitions,
            'position_changes': position_changes,
            'time_intervals': time_intervals,
            'jerkiness_threshold': jerkiness_threshold
        }
    
    def calculate_intermediate_points(self, start_time: float, end_time: float,
                                    start_pos: float, end_pos: float,
                                    transition_style: str = 'smooth_curve',
                                    num_points: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate intermediate points for a transition.
        
        Args:
            start_time: Start time in ms
            end_time: End time in ms
            start_pos: Start position
            end_pos: End position
            transition_style: Style of transition ('smooth_curve', 'ease_in_out', 'linear')
            num_points: Number of intermediate points (auto if None)
            
        Returns:
            Tuple of (intermediate_times, intermediate_positions)
        """
        duration = end_time - start_time
        pos_change = abs(end_pos - start_pos)
        
        # Determine number of intermediate points based on duration and position change
        if num_points is None:
            # More points for longer durations and larger position changes
            base_points = max(1, int(duration / 100))  # 1 point per 100ms
            change_points = max(1, int(pos_change / 30))  # 1 point per 30 position units
            num_points = min(base_points + change_points, 8)  # Cap at 8 intermediate points
        
        if num_points <= 0:
            return np.array([]), np.array([])
        
        # Generate intermediate times
        intermediate_times = np.linspace(start_time, end_time, num_points + 2)[1:-1]  # Exclude endpoints
        
        # Generate intermediate positions based on style
        if transition_style == 'linear':
            # Simple linear interpolation
            t_normalized = (intermediate_times - start_time) / duration
            intermediate_positions = start_pos + (end_pos - start_pos) * t_normalized
            
        elif transition_style == 'ease_in_out':
            # Ease-in-out curve (smooth acceleration/deceleration)
            t_normalized = (intermediate_times - start_time) / duration
            # Use smoothstep function: 3t² - 2t³
            smooth_t = 3 * t_normalized**2 - 2 * t_normalized**3
            intermediate_positions = start_pos + (end_pos - start_pos) * smooth_t
            
        elif transition_style == 'smooth_curve':
            # Use cubic spline with control points for natural movement
            # Add control points that create a more natural movement curve
            control_times = np.array([start_time, start_time + duration*0.3, 
                                    start_time + duration*0.7, end_time])
            
            # Control positions create a slight overshoot for more natural feel
            direction = 1 if end_pos > start_pos else -1
            overshoot_amount = min(abs(end_pos - start_pos) * 0.1, 5)  # Max 5 unit overshoot
            
            control_positions = np.array([
                start_pos,
                start_pos + (end_pos - start_pos) * 0.4 + direction * overshoot_amount * 0.5,
                start_pos + (end_pos - start_pos) * 0.8 + direction * overshoot_amount,
                end_pos
            ])
            
            # Clamp control positions to valid range
            control_positions = np.clip(control_positions, 0, 100)
            
            try:
                spline = CubicSpline(control_times, control_positions)
                intermediate_positions = spline(intermediate_times)
                # Clamp to valid range
                intermediate_positions = np.clip(intermediate_positions, 0, 100)
            except Exception:
                # Fall back to ease-in-out if spline fails
                t_normalized = (intermediate_times - start_time) / duration
                smooth_t = 3 * t_normalized**2 - 2 * t_normalized**3
                intermediate_positions = start_pos + (end_pos - start_pos) * smooth_t
        
        return intermediate_times, intermediate_positions
    
    def insert_intermediate_points(self, times: np.ndarray, positions: np.ndarray,
                                 jerk_threshold: float = 70.0,
                                 time_threshold: float = 250.0,
                                 transition_style: str = 'smooth_curve') -> Tuple[np.ndarray, np.ndarray]:
        """
        Insert intermediate points to reduce jerkiness.
        
        Args:
            times: Original time values
            positions: Original position values
            jerk_threshold: Position change threshold for adding points
            time_threshold: Time threshold for fast transitions
            transition_style: Style for intermediate points
            
        Returns:
            Tuple of (new_times, new_positions) with intermediate points added
        """
        if len(times) < 2:
            return times.copy(), positions.copy()
        
        # Analyze jerkiness
        analysis = self.analyze_transition_jerkiness(times, positions)
        
        new_times = []
        new_positions = []
        
        for i in range(len(times)):
            # Always add the original point
            new_times.append(times[i])
            new_positions.append(positions[i])
            
            # Check if we need to add intermediate points after this point
            if i < len(times) - 1:
                current_time = times[i]
                next_time = times[i + 1]
                current_pos = positions[i]
                next_pos = positions[i + 1]
                
                pos_change = abs(next_pos - current_pos)
                time_interval = next_time - current_time
                
                # Decide if this transition needs intermediate points
                needs_intermediate = False
                
                # Large position change in short time
                if pos_change > jerk_threshold and time_interval < time_threshold:
                    needs_intermediate = True
                
                # Very large position change regardless of time
                elif pos_change > 85:
                    needs_intermediate = True
                
                # Fast transitions
                elif time_interval < 150 and pos_change > 40:
                    needs_intermediate = True
                
                if needs_intermediate:
                    # Calculate intermediate points
                    inter_times, inter_positions = self.calculate_intermediate_points(
                        current_time, next_time, current_pos, next_pos, transition_style
                    )
                    
                    # Add intermediate points
                    new_times.extend(inter_times)
                    new_positions.extend(inter_positions)
        
        # Convert to arrays and sort by time
        new_times = np.array(new_times)
        new_positions = np.array(new_positions)
        
        # Sort by time to maintain chronological order
        sort_indices = np.argsort(new_times)
        new_times = new_times[sort_indices]
        new_positions = new_positions[sort_indices]
        
        return new_times, new_positions
    
    def get_insertion_summary(self, original_times: np.ndarray, original_positions: np.ndarray,
                            new_times: np.ndarray, new_positions: np.ndarray) -> str:
        """
        Get a summary of the intermediate point insertion.
        
        Args:
            original_times: Original time values
            original_positions: Original position values
            new_times: New time values with intermediate points
            new_positions: New position values with intermediate points
            
        Returns:
            Summary string
        """
        original_count = len(original_times)
        new_count = len(new_times)
        added_count = new_count - original_count
        
        # Calculate improvement metrics
        original_movements = np.abs(np.diff(original_positions))
        new_movements = np.abs(np.diff(new_positions))
        
        original_extreme_jumps = np.sum(original_movements > 70)
        new_extreme_jumps = np.sum(new_movements > 70)
        
        original_avg_movement = np.mean(original_movements) if len(original_movements) > 0 else 0
        new_avg_movement = np.mean(new_movements) if len(new_movements) > 0 else 0
        
        summary = f"INTERMEDIATE POINT INSERTION SUMMARY:\n"
        summary += f"Original points: {original_count}\n"
        summary += f"New points: {new_count} (+{added_count} intermediate points)\n"
        summary += f"Extreme jumps: {original_extreme_jumps} → {new_extreme_jumps} "
        summary += f"({((original_extreme_jumps - new_extreme_jumps) / original_extreme_jumps * 100):.1f}% reduction)\n"
        summary += f"Average movement: {original_avg_movement:.1f} → {new_avg_movement:.1f}\n"
        
        return summary


def test_intermediate_point_insertion():
    """Test the intermediate point insertion approach."""
    print("🛠️ TESTING INTERMEDIATE POINT INSERTION APPROACH")
    print("=" * 60)
    
    # Load the problematic sparse funscript data
    funscript_actions = [
        {"at": 16, "pos": 50}, {"at": 716, "pos": 81}, {"at": 1084, "pos": 2}, {"at": 1334, "pos": 94},
        {"at": 1551, "pos": 4}, {"at": 1751, "pos": 100}, {"at": 1951, "pos": 5}, {"at": 2118, "pos": 95},
        {"at": 2268, "pos": 0}, {"at": 2452, "pos": 90}, {"at": 2619, "pos": 0}, {"at": 2786, "pos": 98},
        {"at": 2952, "pos": 5}, {"at": 3086, "pos": 94}, {"at": 3236, "pos": 0}, {"at": 3403, "pos": 100},
        {"at": 3553, "pos": 5}, {"at": 3720, "pos": 99}, {"at": 3853, "pos": 5}, {"at": 4003, "pos": 100}
    ]
    
    times = np.array([action["at"] for action in funscript_actions])
    positions = np.array([action["pos"] for action in funscript_actions])
    
    inserter = IntermediatePointInserter()
    
    # Analyze original jerkiness
    analysis = inserter.analyze_transition_jerkiness(times, positions)
    jerky_count = np.sum(analysis['jerky_transitions'])
    
    print(f"📊 ORIGINAL DATA ANALYSIS:")
    print(f"Total transitions: {len(analysis['jerky_transitions'])}")
    print(f"Jerky transitions: {jerky_count} ({jerky_count/len(analysis['jerky_transitions'])*100:.1f}%)")
    print(f"Jerkiness threshold: {analysis['jerkiness_threshold']:.1f}")
    
    # Test different transition styles
    styles = ['linear', 'ease_in_out', 'smooth_curve']
    results = {}
    
    for style in styles:
        print(f"\n🔧 Testing {style} transition style...")
        
        new_times, new_positions = inserter.insert_intermediate_points(
            times, positions, 
            jerk_threshold=70.0,
            time_threshold=250.0,
            transition_style=style
        )
        
        summary = inserter.get_insertion_summary(times, positions, new_times, new_positions)
        print(summary)
        
        results[style] = {
            'times': new_times,
            'positions': new_positions,
            'added_points': len(new_times) - len(times)
        }
    
    # Create visualization
    create_insertion_comparison_plot(times, positions, results)
    
    print(f"\n📊 Comparison plot saved as 'intermediate_insertion_results.png'")
    
    # Recommend best approach
    best_style = min(results.keys(), key=lambda k: len(results[k]['times']))
    print(f"\n🏆 RECOMMENDATION:")
    print(f"Best style: {best_style}")
    print(f"This approach adds {results[best_style]['added_points']} intermediate points")
    print(f"to create smoother transitions while preserving all original meaningful points.")


def create_insertion_comparison_plot(original_times, original_positions, results):
    """Create comparison visualization for intermediate point insertion."""
    try:
        num_styles = len(results)
        fig, axes = plt.subplots(num_styles + 1, 1, figsize=(15, (num_styles + 1) * 3))
        
        if num_styles == 0:
            return
        
        if num_styles == 1:
            axes = [axes]
        
        original_times_sec = original_times / 1000.0
        
        # Plot original data
        axes[0].plot(original_times_sec, original_positions, 'r-o', markersize=6, linewidth=2, 
                    label='Original (Jerky)', alpha=0.8)
        axes[0].set_title('Original Sparse Data with Extreme Jumps', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Position (0-100)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[0].set_ylim(-5, 105)
        
        # Highlight jerky transitions
        movements = np.abs(np.diff(original_positions))
        intervals = np.diff(original_times)
        for i, (movement, interval) in enumerate(zip(movements, intervals)):
            if movement > 70 and interval < 250:
                axes[0].axvspan(original_times_sec[i], original_times_sec[i+1], 
                              alpha=0.3, color='red', label='Jerky transition' if i == 0 else '')
        
        # Plot each style
        colors = ['blue', 'green', 'purple']
        
        for idx, (style, result) in enumerate(results.items()):
            ax = axes[idx + 1]
            
            new_times_sec = result['times'] / 1000.0
            
            # Plot original points
            ax.plot(original_times_sec, original_positions, 'ro', markersize=6, 
                   alpha=0.6, label='Original points')
            
            # Plot with intermediate points
            ax.plot(new_times_sec, result['positions'], 
                   color=colors[idx % len(colors)], linewidth=2, marker='.',
                   markersize=4, label=f'{style.replace("_", " ").title()} (+{result["added_points"]} points)')
            
            # Highlight added intermediate points
            original_time_set = set(original_times)
            intermediate_mask = ~np.isin(result['times'], list(original_time_set))
            if np.any(intermediate_mask):
                ax.scatter(new_times_sec[intermediate_mask], result['positions'][intermediate_mask],
                         color='orange', s=20, marker='x', alpha=0.8, 
                         label='Added intermediate points')
            
            ax.set_title(f'{style.replace("_", " ").title()} Transition Style', fontsize=12)
            ax.set_ylabel('Position')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_ylim(-5, 105)
        
        # Set xlabel on last plot
        axes[-1].set_xlabel('Time (seconds)')
        
        plt.tight_layout()
        plt.savefig('intermediate_insertion_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Could not create insertion comparison plot: {e}")


if __name__ == "__main__":
    test_intermediate_point_insertion()