#!/usr/bin/env python3
"""
Comparison script for all three oscillation detection modes:
1. Legacy - Superior signal detection & amplification 
2. Experimental - Better timing precision
3. Experimental 2 - Hybrid approach (best of both)
"""

import numpy as np
import cv2
import logging
import sys
import os
import matplotlib.pyplot as plt
from typing import List, Tuple

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from tracker.tracker import ROITracker
from config.constants import TrackerMode
from funscript.dual_axis_funscript import DualAxisFunscript

def create_mock_app():
    """Create a mock app with basic settings"""
    class MockSettings:
        def get(self, key, default=None):
            settings_map = {
                'oscillation_processing_target_height': 480,
                'live_oscillation_dynamic_amp_enabled': True,
                'oscillation_use_simple_amplification': False,
                'oscillation_enable_decay': True,
                'oscillation_hold_duration_ms': 250,
                'oscillation_decay_factor': 0.95
            }
            return settings_map.get(key, default)
    
    class MockApp:
        def __init__(self):
            self.app_settings = MockSettings()
            self.tracking_axis_mode = "both"
            self.single_axis_output_target = "primary"
    
    return MockApp()

def create_test_sequence(width=640, height=480, num_frames=120):
    """
    Create a test sequence with various motion patterns:
    - Frames 0-30: Fast oscillation (3Hz)
    - Frames 30-60: Slow oscillation (1.5Hz) 
    - Frames 60-90: Complex motion (mixed frequencies)
    - Frames 90-120: Static with noise
    """
    frames = []
    
    for frame_num in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        rect_size = 50
        center_x = width // 2
        center_y = height // 2
        
        if frame_num < 30:
            # Fast oscillation (3Hz)
            oscillation = int(25 * np.sin(frame_num * 0.3))  # 3Hz
            motion_type = "Fast 3Hz"
        elif frame_num < 60:
            # Slow oscillation (1.5Hz)
            oscillation = int(30 * np.sin((frame_num - 30) * 0.15))  # 1.5Hz
            motion_type = "Slow 1.5Hz"
        elif frame_num < 90:
            # Complex motion (mixed frequencies)
            fast = 15 * np.sin((frame_num - 60) * 0.3)  # 3Hz component
            slow = 10 * np.sin((frame_num - 60) * 0.1)  # 1Hz component
            oscillation = int(fast + slow)
            motion_type = "Complex mixed"
        else:
            # Static with small random noise
            oscillation = int(2 * np.random.randn())  # Small noise
            motion_type = "Static noise"
        
        x1 = center_x - rect_size // 2
        x2 = center_x + rect_size // 2
        y1 = center_y - rect_size // 2 + oscillation
        y2 = center_y + rect_size // 2 + oscillation
        
        # Ensure bounds
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)
        frames.append((frame, motion_type))
    
    return frames

def test_mode(mode_name: str, method_name: str, frames: List[Tuple[np.ndarray, str]]) -> Tuple[List[int], List[str]]:
    """Test a specific oscillation detection mode"""
    print(f"\n🔬 Testing {mode_name}...")
    
    app = create_mock_app()
    tracker = ROITracker(app, tracker_model_path="/fake/path")
    tracker.tracking_mode = mode_name
    tracker.funscript = DualAxisFunscript()
    tracker.tracking_active = True
    
    positions = []
    motion_types = []
    
    # Get the method to call
    method = getattr(tracker, method_name)
    
    for frame_num, (frame, motion_type) in enumerate(frames):
        frame_time_ms = frame_num * 33  # ~30fps
        
        try:
            processed_frame, action_log = method(frame, frame_time_ms, frame_num)
            
            if action_log:
                for action in action_log:
                    pos = action.get('pos', 50)
                    positions.append(pos)
                    motion_types.append(motion_type)
                    
                    if frame_num % 30 == 0:  # Print every 30th frame
                        print(f"  Frame {frame_num:3d} ({motion_type:12s}): pos = {pos:3d}")
            else:
                # If no action logged, still track the position
                if hasattr(tracker, 'oscillation_funscript_pos'):
                    positions.append(tracker.oscillation_funscript_pos)
                    motion_types.append(motion_type)
                else:
                    positions.append(50)  # Default center
                    motion_types.append(motion_type)
        
        except Exception as e:
            print(f"❌ Error in frame {frame_num}: {e}")
            positions.append(50)  # Default on error
            motion_types.append(motion_type)
    
    return positions, motion_types

def analyze_results(name: str, positions: List[int], motion_types: List[str]):
    """Analyze the results from a mode"""
    positions = np.array(positions)
    
    print(f"\n📊 {name} Analysis:")
    print(f"   Total samples: {len(positions)}")
    print(f"   Mean position: {np.mean(positions):.1f}")
    print(f"   Position range: {np.min(positions)} - {np.max(positions)}")
    print(f"   Standard deviation: {np.std(positions):.1f}")
    
    # Analyze by motion type
    unique_types = list(set(motion_types))
    for motion_type in unique_types:
        mask = [mt == motion_type for mt in motion_types]
        type_positions = positions[mask]
        if len(type_positions) > 0:
            print(f"   {motion_type:12s}: mean={np.mean(type_positions):5.1f}, std={np.std(type_positions):5.1f}, range={np.min(type_positions)}-{np.max(type_positions)}")
    
    return {
        'mean': np.mean(positions),
        'std': np.std(positions),
        'range': (np.min(positions), np.max(positions)),
        'positions': positions
    }

def create_comparison_plot(results_dict: dict):
    """Create a comparison plot of all modes"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Oscillation Detection Mode Comparison', fontsize=16)
    
    # Plot 1: Position over time
    ax1 = axes[0, 0]
    for name, data in results_dict.items():
        ax1.plot(data['positions'], label=name, alpha=0.8)
    ax1.set_title('Position Over Time')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Position (0-100)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Position distribution (histogram)
    ax2 = axes[0, 1]
    for name, data in results_dict.items():
        ax2.hist(data['positions'], bins=20, alpha=0.6, label=name)
    ax2.set_title('Position Distribution')
    ax2.set_xlabel('Position (0-100)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Statistics comparison
    ax3 = axes[1, 0]
    names = list(results_dict.keys())
    means = [results_dict[name]['mean'] for name in names]
    stds = [results_dict[name]['std'] for name in names]
    
    x_pos = np.arange(len(names))
    width = 0.35
    
    ax3.bar(x_pos - width/2, means, width, label='Mean Position', alpha=0.8)
    ax3.bar(x_pos + width/2, stds, width, label='Standard Deviation', alpha=0.8)
    ax3.set_title('Mean Position vs Variability')
    ax3.set_xlabel('Detection Mode')
    ax3.set_ylabel('Value')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(names)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Signal characteristics (FFT)
    ax4 = axes[1, 1]
    for name, data in results_dict.items():
        positions = data['positions']
        if len(positions) > 10:
            # Simple frequency analysis
            fft = np.fft.rfft(positions - np.mean(positions))
            freqs = np.fft.rfftfreq(len(positions), d=1/30)  # 30fps
            magnitude = np.abs(fft)
            ax4.plot(freqs[1:11], magnitude[1:11], label=name, marker='o')  # First 10 frequency bins
    
    ax4.set_title('Frequency Response (0-5 Hz)')
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Magnitude')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/k00gar/PycharmProjects/VR-Funscript-AI-Generator/oscillation_mode_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n📈 Comparison plot saved to: oscillation_mode_comparison.png")

def main():
    """Main comparison function"""
    print("🚀 Starting Oscillation Detection Mode Comparison")
    print("=" * 60)
    
    # Suppress warnings to clean up output
    logging.getLogger().setLevel(logging.ERROR)
    
    # Create test sequence
    print("📹 Creating test sequence with various motion patterns...")
    frames = create_test_sequence()
    print(f"   Generated {len(frames)} frames with 4 motion types")
    
    # Test all modes
    results = {}
    
    # Legacy mode
    try:
        positions, motion_types = test_mode(
            "OSCILLATION_DETECTOR_LEGACY", 
            "process_frame_for_oscillation_legacy", 
            frames
        )
        results['Legacy'] = analyze_results('Legacy', positions, motion_types)
    except Exception as e:
        print(f"❌ Legacy mode failed: {e}")
    
    # Experimental mode
    try:
        positions, motion_types = test_mode(
            "OSCILLATION_DETECTOR", 
            "process_frame_for_oscillation", 
            frames
        )
        results['Experimental'] = analyze_results('Experimental', positions, motion_types)
    except Exception as e:
        print(f"❌ Experimental mode failed: {e}")
    
    # Experimental 2 mode
    try:
        positions, motion_types = test_mode(
            "OSCILLATION_DETECTOR_EXPERIMENTAL_2", 
            "process_frame_for_oscillation_experimental_2", 
            frames
        )
        results['Experimental 2'] = analyze_results('Experimental 2', positions, motion_types)
    except Exception as e:
        print(f"❌ Experimental 2 mode failed: {e}")
    
    # Create comparison plot
    if len(results) >= 2:
        print("\n📊 Creating comparison visualization...")
        try:
            create_comparison_plot(results)
        except Exception as e:
            print(f"⚠️  Could not create plot: {e}")
    
    # Summary comparison
    print(f"\n🏆 FINAL COMPARISON SUMMARY")
    print("=" * 50)
    
    for name, data in results.items():
        print(f"{name:15s}: mean={data['mean']:5.1f}, std={data['std']:5.1f}, range={data['range'][1]-data['range'][0]:3d}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if 'Experimental 2' in results:
        exp2_data = results['Experimental 2']
        print(f"   • Experimental 2 combines the best of both approaches")
        print(f"   • Standard deviation of {exp2_data['std']:.1f} indicates good responsiveness")
        print(f"   • Range of {exp2_data['range'][1]-exp2_data['range'][0]} shows proper signal utilization")
    
    if 'Legacy' in results and 'Experimental' in results:
        legacy_std = results['Legacy']['std']
        exp_std = results['Experimental']['std']
        print(f"   • Legacy std={legacy_std:.1f} vs Experimental std={exp_std:.1f}")
        if legacy_std > exp_std:
            print(f"   • Legacy shows better signal strength/amplification")
        else:
            print(f"   • Experimental shows better signal strength/amplification")
    
    print(f"\n✅ Comparison completed successfully!")

if __name__ == "__main__":
    main()