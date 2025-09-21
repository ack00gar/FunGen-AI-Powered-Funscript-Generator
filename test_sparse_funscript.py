#!/usr/bin/env python3
"""
Test the enhanced Anti-Jerk filter on the actual sparse funscript data provided by the user.
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from funscript.dual_axis_funscript import DualAxisFunscript
from funscript.plugins.anti_jerk_plugin import AntiJerkPlugin


def load_test_funscript():
    """Load the actual funscript data provided by the user."""
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
    
    # Create DualAxisFunscript object
    funscript = DualAxisFunscript()
    for action in funscript_data["actions"]:
        funscript.add_action(action["at"], action["pos"])
    
    return funscript


def test_sparse_anti_jerk_filter():
    """Test the enhanced anti-jerk filter on sparse extreme data."""
    print("🧪 Testing Enhanced Anti-Jerk Filter on Sparse Extreme Data")
    print("=" * 65)
    
    # Load the test funscript
    original_funscript = load_test_funscript()
    
    print(f"📊 Original funscript: {len(original_funscript.primary_actions)} actions")
    
    # Extract original data for analysis
    original_times = np.array([action['at'] for action in original_funscript.primary_actions])
    original_positions = np.array([action['pos'] for action in original_funscript.primary_actions])
    
    # Calculate original characteristics
    original_movements = np.abs(np.diff(original_positions))
    extreme_jumps_original = np.sum(original_movements > 70)
    avg_movement_original = np.mean(original_movements)
    
    print(f"📈 Original characteristics:")
    print(f"   • Average movement: {avg_movement_original:.1f}")
    print(f"   • Extreme jumps (>70): {extreme_jumps_original}/{len(original_movements)} ({extreme_jumps_original/len(original_movements)*100:.1f}%)")
    
    # Create plugin
    plugin = AntiJerkPlugin()
    
    if not plugin.check_dependencies():
        print("❌ Plugin dependencies not available!")
        return
    
    # Test different configurations
    test_configs = [
        {
            'name': 'Auto Mode (Default)',
            'mode': 'auto',
            'extreme_jump_threshold': 70.0,
            'transition_time_threshold': 200.0,
            'interpolation_smoothness': 0.3
        },
        {
            'name': 'Sparse Mode - Conservative',
            'mode': 'sparse',
            'extreme_jump_threshold': 80.0,
            'transition_time_threshold': 180.0,
            'interpolation_smoothness': 0.2
        },
        {
            'name': 'Sparse Mode - Aggressive',
            'mode': 'sparse',
            'extreme_jump_threshold': 60.0,
            'transition_time_threshold': 250.0,
            'interpolation_smoothness': 0.5
        },
        {
            'name': 'Dense Mode (Original)',
            'mode': 'dense',
            'smoothing_strength': 2.0,
            'peak_sensitivity': 0.15
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
        avg_movement_processed = np.mean(processed_movements)
        
        # Calculate metrics
        extreme_jump_reduction = (extreme_jumps_original - extreme_jumps_processed) / extreme_jumps_original * 100 if extreme_jumps_original > 0 else 0
        avg_movement_reduction = (avg_movement_original - avg_movement_processed) / avg_movement_original * 100
        
        # Calculate position preservation (how much the overall pattern is preserved)
        position_changes = np.mean(np.abs(original_positions - processed_positions))
        pattern_preservation = max(0, 100 - (position_changes / np.ptp(original_positions) * 100))
        
        results[config['name']] = {
            'processed_positions': processed_positions,
            'extreme_jump_reduction': extreme_jump_reduction,
            'avg_movement_reduction': avg_movement_reduction,
            'pattern_preservation': pattern_preservation,
            'extreme_jumps_count': extreme_jumps_processed
        }
        
        print(f"   ✅ Extreme jump reduction: {extreme_jump_reduction:.1f}% ({extreme_jumps_original} → {extreme_jumps_processed})")
        print(f"   ✅ Average movement reduction: {avg_movement_reduction:.1f}%")
        print(f"   ✅ Pattern preservation: {pattern_preservation:.1f}%")
    
    # Create visualization
    create_sparse_comparison_plot(original_times, original_positions, results)
    
    print(f"\n🎉 Sparse funscript test completed!")
    print(f"📊 Comparison plot saved as 'sparse_funscript_results.png'")
    
    # Find best configuration
    if results:
        best_config = max(results.items(), 
                         key=lambda x: x[1]['extreme_jump_reduction'] + x[1]['pattern_preservation'])
        print(f"\n🏆 Best configuration: {best_config[0]}")
        print(f"   📈 {best_config[1]['extreme_jump_reduction']:.1f}% extreme jump reduction")
        print(f"   🎯 {best_config[1]['pattern_preservation']:.1f}% pattern preservation")


def create_sparse_comparison_plot(times, original_positions, results):
    """Create comparison visualization for sparse data processing."""
    try:
        num_results = len(results)
        fig_height = max(8, num_results * 2)
        fig, axes = plt.subplots(num_results + 1, 1, figsize=(15, fig_height))
        
        if num_results == 0:
            return
        
        if num_results == 1:
            axes = [axes]
        
        times_sec = times / 1000.0
        
        # Plot original data
        axes[0].plot(times_sec, original_positions, 'r-o', markersize=4, linewidth=2, alpha=0.8, label='Original (Jerky)')
        axes[0].set_title('Original Sparse Extreme Data', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Position (0-100)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[0].set_ylim(-5, 105)
        
        # Highlight extreme jumps in original
        movements = np.abs(np.diff(original_positions))
        for i, movement in enumerate(movements):
            if movement > 70:
                axes[0].axvspan(times_sec[i], times_sec[i+1], alpha=0.2, color='red')
        
        # Plot each processed version
        colors = ['blue', 'green', 'purple', 'orange', 'brown']
        
        for idx, (name, result) in enumerate(results.items()):
            ax = axes[idx + 1]
            
            # Plot comparison
            ax.plot(times_sec, original_positions, 'r-', linewidth=1, alpha=0.4, label='Original')
            ax.plot(times_sec, result['processed_positions'], 
                   color=colors[idx % len(colors)], linewidth=2, marker='o', markersize=3,
                   label=f'Processed ({name})')
            
            # Add metrics to title
            title = f"{name}\n"
            title += f"Extreme jumps ↓{result['extreme_jump_reduction']:.1f}%, "
            title += f"Pattern preservation {result['pattern_preservation']:.1f}%"
            
            ax.set_title(title, fontsize=12)
            ax.set_ylabel('Position')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_ylim(-5, 105)
            
            # Highlight remaining extreme jumps
            processed_movements = np.abs(np.diff(result['processed_positions']))
            for i, movement in enumerate(processed_movements):
                if movement > 70:
                    ax.axvspan(times_sec[i], times_sec[i+1], alpha=0.2, color='orange')
        
        # Set xlabel on last plot
        axes[-1].set_xlabel('Time (seconds)')
        
        plt.tight_layout()
        plt.savefig('sparse_funscript_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Could not create comparison plot: {e}")


if __name__ == "__main__":
    test_sparse_anti_jerk_filter()