#!/usr/bin/env python3
"""
Test the enhanced Anti-Jerk plugin with intermediate point insertion mode.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from funscript.dual_axis_funscript import DualAxisFunscript
from funscript.plugins.anti_jerk_plugin import AntiJerkPlugin


def create_test_funscript():
    """Create the problematic sparse funscript."""
    funscript_actions = [
        {"at": 16, "pos": 50}, {"at": 716, "pos": 81}, {"at": 1084, "pos": 2}, {"at": 1334, "pos": 94},
        {"at": 1551, "pos": 4}, {"at": 1751, "pos": 100}, {"at": 1951, "pos": 5}, {"at": 2118, "pos": 95},
        {"at": 2268, "pos": 0}, {"at": 2452, "pos": 90}, {"at": 2619, "pos": 0}, {"at": 2786, "pos": 98},
        {"at": 2952, "pos": 5}, {"at": 3086, "pos": 94}, {"at": 3236, "pos": 0}, {"at": 3403, "pos": 100},
        {"at": 3553, "pos": 5}, {"at": 3720, "pos": 99}, {"at": 3853, "pos": 5}, {"at": 4003, "pos": 100}
    ]
    
    funscript = DualAxisFunscript()
    for action in funscript_actions:
        funscript.add_action(action["at"], action["pos"])
    
    return funscript


def test_enhanced_anti_jerk_plugin():
    """Test the enhanced anti-jerk plugin with all modes."""
    print("🚀 TESTING ENHANCED ANTI-JERK PLUGIN")
    print("=" * 50)
    
    # Create test funscript
    original_funscript = create_test_funscript()
    
    # Extract original data
    original_times = np.array([action['at'] for action in original_funscript.primary_actions])
    original_positions = np.array([action['pos'] for action in original_funscript.primary_actions])
    
    original_movements = np.abs(np.diff(original_positions))
    original_extreme_jumps = np.sum(original_movements > 70)
    
    print(f"📊 Original: {len(original_funscript.primary_actions)} actions, {original_extreme_jumps} extreme jumps")
    
    # Create plugin
    plugin = AntiJerkPlugin()
    
    if not plugin.check_dependencies():
        print("❌ Plugin dependencies not available!")
        return
    
    # Test configurations
    test_configs = [
        {
            'name': 'Auto Mode (Should select intermediate insertion)',
            'mode': 'auto'
        },
        {
            'name': 'Intermediate Insertion - Linear',
            'mode': 'intermediate_insertion',
            'transition_style': 'linear',
            'insertion_jerk_threshold': 70.0,
            'insertion_time_threshold': 250.0
        },
        {
            'name': 'Intermediate Insertion - Ease In/Out',
            'mode': 'intermediate_insertion',
            'transition_style': 'ease_in_out',
            'insertion_jerk_threshold': 70.0,
            'insertion_time_threshold': 250.0
        },
        {
            'name': 'Intermediate Insertion - Smooth Curve',
            'mode': 'intermediate_insertion',
            'transition_style': 'smooth_curve',
            'insertion_jerk_threshold': 60.0,  # More aggressive
            'insertion_time_threshold': 300.0
        },
        {
            'name': 'Sparse Mode (Original)',
            'mode': 'sparse',
            'extreme_jump_threshold': 70.0,
            'interpolation_smoothness': 0.5
        }
    ]
    
    results = {}
    
    for config in test_configs:
        print(f"\n🔧 Testing {config['name']}...")
        
        # Create a copy of the funscript for testing
        test_funscript = DualAxisFunscript()
        for action in original_funscript.primary_actions:
            test_funscript.add_action(action['at'], action['pos'])
        
        # Apply the filter
        error = plugin.transform(test_funscript, 'primary', **{k: v for k, v in config.items() if k != 'name'})
        
        if error:
            print(f"❌ Filter failed: {error}")
            continue
        
        # Extract processed data
        processed_times = np.array([action['at'] for action in test_funscript.primary_actions])
        processed_positions = np.array([action['pos'] for action in test_funscript.primary_actions])
        
        # Calculate improvements
        processed_movements = np.abs(np.diff(processed_positions))
        extreme_jumps_processed = np.sum(processed_movements > 70)
        
        # Calculate metrics
        extreme_jump_reduction = (original_extreme_jumps - extreme_jumps_processed) / original_extreme_jumps * 100 if original_extreme_jumps > 0 else 0
        added_points = len(processed_times) - len(original_times)
        
        results[config['name']] = {
            'processed_times': processed_times,
            'processed_positions': processed_positions,
            'extreme_jump_reduction': extreme_jump_reduction,
            'extreme_jumps_final': extreme_jumps_processed,
            'added_points': added_points
        }
        
        print(f"   ✅ Actions: {len(original_times)} → {len(processed_times)} (+{added_points})")
        print(f"   ✅ Extreme jumps: {original_extreme_jumps} → {extreme_jumps_processed} ({extreme_jump_reduction:.1f}% reduction)")
    
    # Create visualization
    create_enhanced_comparison_plot(original_times, original_positions, results)
    
    print(f"\n🎉 Enhanced anti-jerk test completed!")
    print(f"📊 Comparison plot saved as 'enhanced_anti_jerk_results.png'")
    
    # Find best configuration
    if results:
        best_config = max(results.items(), key=lambda x: x[1]['extreme_jump_reduction'])
        print(f"\n🏆 Best configuration: {best_config[0]}")
        print(f"   📈 {best_config[1]['extreme_jump_reduction']:.1f}% extreme jump reduction")
        print(f"   📊 Added {best_config[1]['added_points']} intermediate points")


def create_enhanced_comparison_plot(original_times, original_positions, results):
    """Create enhanced comparison visualization."""
    try:
        num_results = len(results)
        fig_height = max(10, (num_results + 1) * 2.5)
        fig, axes = plt.subplots(num_results + 1, 1, figsize=(16, fig_height))
        
        if num_results == 0:
            return
        
        if num_results == 1:
            axes = [axes]
        
        times_sec = original_times / 1000.0
        
        # Plot original data
        axes[0].plot(times_sec, original_positions, 'r-o', markersize=6, linewidth=2, 
                    alpha=0.8, label='Original Sparse Data')
        axes[0].set_title('Original Sparse Extreme Data (The Problem)', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Position (0-100)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[0].set_ylim(-5, 105)
        
        # Highlight extreme jumps
        movements = np.abs(np.diff(original_positions))
        intervals = np.diff(original_times)
        for i, (movement, interval) in enumerate(zip(movements, intervals)):
            if movement > 70:
                axes[0].axvspan(times_sec[i], times_sec[i+1], alpha=0.3, color='red')
        
        # Add text annotation
        axes[0].text(0.02, 0.98, f'{np.sum(movements > 70)} extreme jumps (>70 position change)', 
                    transform=axes[0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        # Plot each processed version
        colors = ['blue', 'green', 'purple', 'orange', 'brown']
        
        for idx, (name, result) in enumerate(results.items()):
            ax = axes[idx + 1]
            
            result_times_sec = result['processed_times'] / 1000.0
            
            # Plot original points as reference
            ax.plot(times_sec, original_positions, 'r-', linewidth=1, alpha=0.4, label='Original path')
            ax.scatter(times_sec, original_positions, c='red', s=40, alpha=0.7, 
                      label=f'Original points ({len(original_times)})', zorder=5)
            
            # Plot processed result
            ax.plot(result_times_sec, result['processed_positions'], 
                   color=colors[idx % len(colors)], linewidth=2, alpha=0.8,
                   label=f'Processed path')
            
            # Highlight added intermediate points if any
            if result['added_points'] > 0:
                # Find points that weren't in original
                original_time_set = set(original_times)
                added_mask = ~np.isin(result['processed_times'], list(original_time_set))
                if np.any(added_mask):
                    ax.scatter(result_times_sec[added_mask], result['processed_positions'][added_mask],
                             c='orange', s=20, marker='x', alpha=0.8, 
                             label=f'+{result["added_points"]} intermediate points', zorder=4)
            
            # Title with metrics
            title = f"{name}\n"
            title += f"Extreme jumps: {result['extreme_jumps_final']} "
            title += f"({result['extreme_jump_reduction']:.1f}% reduction), "
            title += f"Points: {len(original_times)} → {len(result['processed_times'])}"
            
            ax.set_title(title, fontsize=11)
            ax.set_ylabel('Position')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            ax.set_ylim(-5, 105)
        
        # Set xlabel on last plot
        axes[-1].set_xlabel('Time (seconds)')
        
        plt.tight_layout()
        plt.savefig('enhanced_anti_jerk_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Could not create enhanced comparison plot: {e}")


if __name__ == "__main__":
    test_enhanced_anti_jerk_plugin()