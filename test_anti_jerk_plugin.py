#!/usr/bin/env python3
"""
Test script for the Anti-Jerk Filter Plugin.
Creates sample data similar to the jerky signal shown in the image and tests the filter.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from funscript.dual_axis_funscript import DualAxisFunscript
from funscript.plugins.anti_jerk_plugin import AntiJerkPlugin


def create_jerky_test_signal(duration_ms=30000, base_frequency=0.5, jerk_frequency=8.0):
    """
    Create a test signal with base movement plus jerky oscillations.
    Similar to the pattern shown in the user's image.
    """
    # Generate time points (every 50ms for smooth signal)
    times = np.arange(0, duration_ms, 50)
    
    # Base signal: slow sinusoidal movement with some variation
    t_seconds = times / 1000.0
    base_signal = 50 + 25 * np.sin(2 * np.pi * base_frequency * t_seconds)
    
    # Add some variation to make it more realistic
    base_signal += 10 * np.sin(2 * np.pi * base_frequency * 2.3 * t_seconds)
    
    # Add jerky high-frequency oscillations (the noise we want to remove)
    jerk_amplitude = 8.0  # Amplitude of jerky movements
    jerk_signal = jerk_amplitude * np.sin(2 * np.pi * jerk_frequency * t_seconds)
    
    # Add random small variations to make it more realistic
    random_jerk = np.random.normal(0, 2, len(times))
    
    # Combine signals
    jerky_positions = base_signal + jerk_signal + random_jerk
    
    # Clamp to valid range
    jerky_positions = np.clip(jerky_positions, 0, 100)
    
    return times, jerky_positions


def test_anti_jerk_filter():
    """Test the anti-jerk filter with sample data."""
    print("Testing Anti-Jerk Filter Plugin...")
    
    # Create jerky test signal
    times, jerky_positions = create_jerky_test_signal()
    
    print(f"Generated test signal with {len(times)} points over {times[-1]/1000:.1f} seconds")
    print(f"Position range: {jerky_positions.min():.1f} - {jerky_positions.max():.1f}")
    
    # Create funscript with jerky data
    funscript = DualAxisFunscript()
    for time, pos in zip(times, jerky_positions):
        funscript.add_action(int(time), int(round(pos)))
    
    print(f"Created funscript with {len(funscript.primary_actions)} actions")
    
    # Create and test the plugin
    plugin = AntiJerkPlugin()
    
    if not plugin.check_dependencies():
        print("❌ Plugin dependencies not available!")
        return
    
    print("✅ Plugin dependencies available")
    
    # Test with different parameter sets
    test_configs = [
        {
            'name': 'Light smoothing',
            'smoothing_strength': 1.5,
            'peak_sensitivity': 0.2,
            'jerk_threshold': 20.0
        },
        {
            'name': 'Medium smoothing', 
            'smoothing_strength': 2.5,
            'peak_sensitivity': 0.15,
            'jerk_threshold': 15.0
        },
        {
            'name': 'Strong smoothing',
            'smoothing_strength': 4.0,
            'peak_sensitivity': 0.1,
            'jerk_threshold': 10.0
        }
    ]
    
    results = {}
    
    for config in test_configs:
        print(f"\nTesting {config['name']}...")
        
        # Create a copy of the funscript for testing
        test_funscript = DualAxisFunscript()
        for action in funscript.primary_actions:
            test_funscript.add_action(action['at'], action['pos'])
        
        # Apply the filter
        error = plugin.transform(
            test_funscript, 
            axis='primary',
            **{k: v for k, v in config.items() if k != 'name'}
        )
        
        if error:
            print(f"❌ Filter failed: {error}")
            continue
        
        # Extract smoothed data
        smoothed_times = np.array([action['at'] for action in test_funscript.primary_actions])
        smoothed_positions = np.array([action['pos'] for action in test_funscript.primary_actions])
        
        # Calculate metrics
        original_jerk = calculate_jerk_metric(times, jerky_positions)
        smoothed_jerk = calculate_jerk_metric(smoothed_times, smoothed_positions)
        jerk_reduction = (original_jerk - smoothed_jerk) / original_jerk * 100
        
        # Calculate peak preservation
        peaks_preserved = calculate_peak_preservation(jerky_positions, smoothed_positions)
        
        results[config['name']] = {
            'smoothed_positions': smoothed_positions,
            'jerk_reduction': jerk_reduction,
            'peaks_preserved': peaks_preserved
        }
        
        print(f"  ✅ Jerk reduction: {jerk_reduction:.1f}%")
        print(f"  ✅ Peak preservation: {peaks_preserved:.1f}%")
    
    # Create visualization
    create_comparison_plot(times, jerky_positions, results)
    
    print(f"\n🎉 Anti-jerk filter test completed!")
    print(f"📊 Comparison plot saved as 'anti_jerk_comparison.png'")


def calculate_jerk_metric(times, positions):
    """Calculate a simple jerk metric (third derivative magnitude)."""
    if len(positions) < 4:
        return 0.0
    
    # Calculate successive differences
    dt = np.diff(times) / 1000.0  # Convert to seconds
    dt = np.maximum(dt, 0.001)   # Avoid division by zero
    
    # First derivative (velocity)
    velocity = np.diff(positions) / dt
    
    # Second derivative (acceleration)
    dt2 = (dt[:-1] + dt[1:]) / 2
    acceleration = np.diff(velocity) / dt2
    
    # Third derivative (jerk)
    dt3 = (dt2[:-1] + dt2[1:]) / 2
    jerk = np.diff(acceleration) / dt3
    
    # Return RMS jerk as metric
    return np.sqrt(np.mean(jerk**2)) if len(jerk) > 0 else 0.0


def calculate_peak_preservation(original, smoothed):
    """Calculate how well peaks are preserved."""
    if len(original) != len(smoothed):
        return 0.0
    
    # Find peaks in original signal
    from scipy.signal import find_peaks
    
    original_peaks, _ = find_peaks(original, prominence=5.0)
    original_valleys, _ = find_peaks(-original, prominence=5.0)
    
    if len(original_peaks) == 0 and len(original_valleys) == 0:
        return 100.0  # No peaks to preserve
    
    # Check how well peaks are preserved
    peak_differences = []
    
    for peak_idx in original_peaks:
        original_val = original[peak_idx]
        smoothed_val = smoothed[peak_idx]
        peak_differences.append(abs(original_val - smoothed_val))
    
    for valley_idx in original_valleys:
        original_val = original[valley_idx]
        smoothed_val = smoothed[valley_idx]
        peak_differences.append(abs(original_val - smoothed_val))
    
    if len(peak_differences) == 0:
        return 100.0
    
    avg_difference = np.mean(peak_differences)
    signal_range = np.ptp(original)
    
    # Return preservation percentage
    preservation = max(0, 100 - (avg_difference / signal_range * 100))
    return preservation


def create_comparison_plot(times, original_positions, results):
    """Create a comparison plot showing original vs smoothed signals."""
    try:
        plt.figure(figsize=(15, 10))
        
        # Convert times to seconds for plotting
        times_sec = times / 1000.0
        
        # Plot original signal
        plt.subplot(2, 2, 1)
        plt.plot(times_sec, original_positions, 'g-', linewidth=1, alpha=0.8, label='Original (Jerky)')
        plt.title('Original Jerky Signal')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Position')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot each smoothed version
        subplot_idx = 2
        for name, result in results.items():
            plt.subplot(2, 2, subplot_idx)
            plt.plot(times_sec, original_positions, 'g-', linewidth=1, alpha=0.4, label='Original')
            plt.plot(times_sec, result['smoothed_positions'], 'b-', linewidth=2, label='Smoothed')
            plt.title(f'{name}\nJerk↓ {result["jerk_reduction"]:.1f}%, Peaks {result["peaks_preserved"]:.1f}%')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Position')
            plt.grid(True, alpha=0.3)
            plt.legend()
            subplot_idx += 1
        
        # Overall comparison
        plt.subplot(2, 2, 4)
        plt.plot(times_sec, original_positions, 'g-', linewidth=1, alpha=0.6, label='Original')
        
        colors = ['blue', 'red', 'purple']
        for i, (name, result) in enumerate(results.items()):
            plt.plot(times_sec, result['smoothed_positions'], 
                    color=colors[i], linewidth=2, alpha=0.8, label=name)
        
        plt.title('All Smoothing Levels Comparison')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Position')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('anti_jerk_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Could not create plot: {e}")


if __name__ == "__main__":
    test_anti_jerk_filter()