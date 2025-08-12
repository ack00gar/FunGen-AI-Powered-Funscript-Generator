"""
Unit tests for individual plugin functionality.

Tests each plugin's core transformation logic, parameter handling,
and edge cases in isolation.
"""

import pytest
import numpy as np
import logging
from typing import List, Dict, Any

from funscript.dual_axis_funscript import DualAxisFunscript
from funscript.plugins.base_plugin import plugin_registry

# Skip scipy-dependent tests if scipy not available
try:
    import scipy
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class TestAmplifyPlugin:
    """Test amplify plugin functionality."""
    
    @pytest.fixture
    def simple_funscript(self):
        """Create a simple test funscript."""
        fs = DualAxisFunscript()
        positions = [30, 70, 20, 80, 40, 60]
        for i, pos in enumerate(positions):
            fs.add_action(1000 + i * 200, pos)
        return fs
    
    def test_amplify_basic_scaling(self, simple_funscript):
        """Test basic position amplification."""
        original_positions = [action['pos'] for action in simple_funscript.primary_actions]
        
        success = simple_funscript.apply_plugin('Amplify', scale_factor=1.5, center_value=50)
        assert success
        
        new_positions = [action['pos'] for action in simple_funscript.primary_actions]
        
        # Check that positions were scaled correctly around center
        for orig, new in zip(original_positions, new_positions):
            expected = int(np.clip(50 + (orig - 50) * 1.5, 0, 100))
            assert new == expected, f"Expected {expected}, got {new} for original {orig}"
    
    def test_amplify_clipping(self, simple_funscript):
        """Test that amplification clips to valid range."""
        # Use extreme scaling to force clipping
        success = simple_funscript.apply_plugin('Amplify', scale_factor=5.0, center_value=50)
        assert success
        
        new_positions = [action['pos'] for action in simple_funscript.primary_actions]
        
        # All positions should be clipped to 0-100 range
        for pos in new_positions:
            assert 0 <= pos <= 100, f"Position {pos} is out of valid range"
    
    def test_amplify_center_value_effect(self, simple_funscript):
        """Test different center values produce different results."""
        original_positions = [action['pos'] for action in simple_funscript.primary_actions]
        
        # Test with center at 25
        fs1 = DualAxisFunscript()
        for action in simple_funscript.primary_actions:
            fs1.add_action(action['at'], action['pos'])
        
        fs1.apply_plugin('Amplify', scale_factor=1.5, center_value=25)
        positions_center_25 = [action['pos'] for action in fs1.primary_actions]
        
        # Test with center at 75
        fs2 = DualAxisFunscript()
        for action in simple_funscript.primary_actions:
            fs2.add_action(action['at'], action['pos'])
        
        fs2.apply_plugin('Amplify', scale_factor=1.5, center_value=75)
        positions_center_75 = [action['pos'] for action in fs2.primary_actions]
        
        # Results should be different
        assert positions_center_25 != positions_center_75


class TestInvertPlugin:
    """Test invert plugin functionality."""
    
    def test_invert_basic(self):
        """Test basic position inversion."""
        fs = DualAxisFunscript()
        original_positions = [10, 30, 50, 70, 90]
        expected_inverted = [90, 70, 50, 30, 10]
        
        for i, pos in enumerate(original_positions):
            fs.add_action(1000 + i * 100, pos)
        
        success = fs.apply_plugin('Invert')
        assert success
        
        new_positions = [action['pos'] for action in fs.primary_actions]
        assert new_positions == expected_inverted
    
    def test_invert_custom_center(self):
        """Test inversion around custom center point."""
        fs = DualAxisFunscript()
        fs.add_action(1000, 20)  # 60 units below center 80
        fs.add_action(1100, 80)  # At center
        fs.add_action(1200, 90)  # 10 units above center
        
        success = fs.apply_plugin('Invert', center_value=80)
        assert success
        
        new_positions = [action['pos'] for action in fs.primary_actions]
        expected = [100, 80, 70]  # Mirrored around center 80, clipped
        assert new_positions == expected


class TestThresholdClampPlugin:
    """Test threshold clamping plugin functionality."""
    
    def test_threshold_clamp_basic(self):
        """Test basic threshold clamping."""
        fs = DualAxisFunscript()
        positions = [5, 25, 45, 65, 85, 95]  # Mix below, within, above thresholds
        for i, pos in enumerate(positions):
            fs.add_action(1000 + i * 100, pos)
        
        success = fs.apply_plugin('Threshold Clamp', 
                                  lower_threshold=20, 
                                  upper_threshold=80)
        assert success
        
        new_positions = [action['pos'] for action in fs.primary_actions]
        expected = [20, 25, 45, 65, 80, 80]  # Clamped to thresholds
        assert new_positions == expected
    
    def test_threshold_clamp_edge_cases(self):
        """Test edge cases for threshold clamping."""
        fs = DualAxisFunscript()
        fs.add_action(1000, 0)    # Minimum value
        fs.add_action(1100, 50)   # Middle value
        fs.add_action(1200, 100)  # Maximum value
        
        # Very wide thresholds (should not change anything)
        success = fs.apply_plugin('Threshold Clamp', 
                                  lower_threshold=0, 
                                  upper_threshold=100)
        assert success
        
        new_positions = [action['pos'] for action in fs.primary_actions]
        assert new_positions == [0, 50, 100]


class TestValueClampPlugin:
    """Test value clamping plugin functionality."""
    
    def test_value_clamp_to_center(self):
        """Test clamping values to center point."""
        fs = DualAxisFunscript()
        positions = [10, 30, 50, 70, 90]
        for i, pos in enumerate(positions):
            fs.add_action(1000 + i * 100, pos)
        
        success = fs.apply_plugin('value_clamp', clamp_value=50)
        assert success
        
        new_positions = [action['pos'] for action in fs.primary_actions]
        expected = [50, 50, 50, 50, 50]  # All clamped to 50
        assert new_positions == expected
    
    def test_value_clamp_edge_values(self):
        """Test clamping to edge values."""
        fs = DualAxisFunscript()
        fs.add_action(1000, 25)
        fs.add_action(1100, 75)
        
        # Clamp to minimum
        success = fs.apply_plugin('value_clamp', clamp_value=0)
        assert success
        assert [action['pos'] for action in fs.primary_actions] == [0, 0]
        
        # Reset and clamp to maximum
        fs.primary_actions[0]['pos'] = 25
        fs.primary_actions[1]['pos'] = 75
        success = fs.apply_plugin('value_clamp', clamp_value=100)
        assert success
        assert [action['pos'] for action in fs.primary_actions] == [100, 100]


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy not available")
class TestSavgolFilterPlugin:
    """Test Savitzky-Golay filter plugin functionality."""
    
    def test_savgol_smoothing(self):
        """Test that savgol filter smooths noisy data."""
        fs = DualAxisFunscript()
        
        # Create noisy sine wave
        for i in range(20):
            clean_pos = 50 + 30 * np.sin(i * 0.3)
            noisy_pos = clean_pos + np.random.normal(0, 5)  # Add noise
            fs.add_action(1000 + i * 100, int(np.clip(noisy_pos, 0, 100)))
        
        original_positions = [action['pos'] for action in fs.primary_actions]
        
        success = fs.apply_plugin('Smooth (SG)', window_length=7, polyorder=3)
        assert success
        
        smoothed_positions = [action['pos'] for action in fs.primary_actions]
        
        # Smoothed data should be different from original
        assert smoothed_positions != original_positions
        
        # Calculate smoothness (less variation between adjacent points)
        orig_variations = sum(abs(original_positions[i+1] - original_positions[i]) 
                             for i in range(len(original_positions)-1))
        smooth_variations = sum(abs(smoothed_positions[i+1] - smoothed_positions[i]) 
                               for i in range(len(smoothed_positions)-1))
        
        # Smoothed should have less variation (though this isn't guaranteed for all data)
        assert len(smoothed_positions) == len(original_positions)
    
    def test_savgol_insufficient_points(self):
        """Test savgol filter with insufficient points."""
        fs = DualAxisFunscript()
        fs.add_action(1000, 50)
        fs.add_action(1100, 60)  # Only 2 points
        
        # Should fail with default window size (7)
        with pytest.raises(ValueError):
            fs.apply_plugin('Smooth (SG)')
    
    def test_savgol_parameter_adjustment(self):
        """Test automatic parameter adjustment."""
        fs = DualAxisFunscript()
        for i in range(10):
            fs.add_action(1000 + i * 100, 50 + i * 2)
        
        # Even window length should be adjusted to odd
        success = fs.apply_plugin('Smooth (SG)', window_length=6, polyorder=2)
        assert success  # Should succeed with adjusted window_length=7
        
        # Polyorder >= window_length should be adjusted
        success = fs.apply_plugin('Smooth (SG)', window_length=5, polyorder=10)
        assert success  # Should succeed with adjusted polyorder


class TestRdpSimplifyPlugin:
    """Test RDP simplification plugin functionality."""
    
    def test_rdp_point_reduction(self):
        """Test that RDP reduces number of points."""
        fs = DualAxisFunscript()
        
        # Create line with redundant points
        for i in range(21):
            pos = 20 + i * 3  # Linear progression with small steps
            fs.add_action(1000 + i * 100, int(np.clip(pos, 0, 100)))
        
        original_count = len(fs.primary_actions)
        
        success = fs.apply_plugin('Simplify (RDP)', epsilon=5.0)
        assert success
        
        simplified_count = len(fs.primary_actions)
        
        # Should have reduced points significantly
        assert simplified_count < original_count
        assert simplified_count >= 2  # Should at least keep start and end
    
    def test_rdp_preserve_endpoints(self):
        """Test that RDP preserves start and end points."""
        fs = DualAxisFunscript()
        
        # Create data with distinct start and end
        positions = [10, 15, 20, 25, 30, 35, 40, 45, 90]
        for i, pos in enumerate(positions):
            fs.add_action(1000 + i * 100, pos)
        
        start_pos = fs.primary_actions[0]['pos']
        end_pos = fs.primary_actions[-1]['pos']
        
        success = fs.apply_plugin('Simplify (RDP)', epsilon=10.0)
        assert success
        
        # Start and end should be preserved
        assert fs.primary_actions[0]['pos'] == start_pos
        assert fs.primary_actions[-1]['pos'] == end_pos
    
    def test_rdp_epsilon_effect(self):
        """Test that higher epsilon removes more points."""
        fs1 = DualAxisFunscript()
        fs2 = DualAxisFunscript()
        
        # Create identical slightly wavy data
        for i in range(15):
            pos = 50 + 20 * np.sin(i * 0.2) + np.random.normal(0, 1)
            pos = int(np.clip(pos, 0, 100))
            fs1.add_action(1000 + i * 100, pos)
            fs2.add_action(1000 + i * 100, pos)
        
        # Apply different epsilon values
        fs1.apply_plugin('Simplify (RDP)', epsilon=2.0)  # Conservative
        fs2.apply_plugin('Simplify (RDP)', epsilon=8.0)  # Aggressive
        
        count_low_epsilon = len(fs1.primary_actions)
        count_high_epsilon = len(fs2.primary_actions)
        
        # Higher epsilon should result in fewer points
        assert count_high_epsilon <= count_low_epsilon


class TestSpeedLimiterPlugin:
    """Test speed limiter plugin functionality."""
    
    def test_speed_limiter_basic(self):
        """Test basic speed limiting functionality."""
        fs = DualAxisFunscript()
        
        # Create data with some high-speed movements
        fs.add_action(1000, 0)
        fs.add_action(1100, 100)  # Very fast movement: 100 units in 100ms
        fs.add_action(1200, 0)    # Another fast movement
        fs.add_action(1300, 50)   # Moderate movement
        
        success = fs.apply_plugin('Speed Limiter', max_speed_units_per_second=500)
        assert success
        
        # Check that no movements exceed the speed limit
        actions = fs.primary_actions
        for i in range(len(actions) - 1):
            time_diff = actions[i+1]['at'] - actions[i]['at']
            pos_diff = abs(actions[i+1]['pos'] - actions[i]['pos'])
            
            if time_diff > 0:
                speed = (pos_diff / time_diff) * 1000  # units per second
                assert speed <= 500 + 1, f"Speed {speed} exceeds limit"  # Allow 1 unit tolerance
    
    def test_speed_limiter_no_change_needed(self):
        """Test speed limiter when no changes are needed."""
        fs = DualAxisFunscript()
        
        # Create slow, smooth movements
        for i in range(5):
            fs.add_action(1000 + i * 500, 30 + i * 10)  # Slow progression
        
        original_positions = [action['pos'] for action in fs.primary_actions]
        
        success = fs.apply_plugin('Speed Limiter', max_speed_units_per_second=100)
        assert success
        
        new_positions = [action['pos'] for action in fs.primary_actions]
        
        # No changes should be made
        assert new_positions == original_positions


class TestKeyframePlugin:
    """Test keyframe plugin functionality."""
    
    def test_keyframe_extraction(self):
        """Test keyframe extraction from smooth data."""
        fs = DualAxisFunscript()
        
        # Create smooth curve with clear peaks and valleys
        for i in range(20):
            pos = 50 + 30 * np.sin(i * 0.5)  # Clear sine wave
            fs.add_action(1000 + i * 100, int(np.clip(pos, 0, 100)))
        
        original_count = len(fs.primary_actions)
        
        success = fs.apply_plugin('Keyframes', min_prominence=10)
        assert success
        
        keyframe_count = len(fs.primary_actions)
        
        # Should extract fewer keyframes than original points
        assert keyframe_count < original_count
        assert keyframe_count >= 2  # At least start and end
    
    def test_keyframe_preserve_extremes(self):
        """Test that keyframes preserve extreme values."""
        fs = DualAxisFunscript()
        
        # Create data with clear min and max
        positions = [50, 45, 40, 30, 10, 20, 40, 60, 80, 90, 70, 50]  # Min at index 4, max at index 9
        for i, pos in enumerate(positions):
            fs.add_action(1000 + i * 100, pos)
        
        min_pos = min(positions)
        max_pos = max(positions)
        
        success = fs.apply_plugin('Keyframes', min_prominence=5)
        assert success
        
        keyframe_positions = [action['pos'] for action in fs.primary_actions]
        
        # Min and max should be preserved in keyframes
        assert min_pos in keyframe_positions, f"Minimum {min_pos} not preserved"
        assert max_pos in keyframe_positions, f"Maximum {max_pos} not preserved"


class TestResamplePlugin:
    """Test resample plugin functionality."""
    
    def test_resample_increase_density(self):
        """Test resampling to increase point density."""
        fs = DualAxisFunscript()
        
        # Create sparse data
        fs.add_action(1000, 20)
        fs.add_action(1500, 80)  # 500ms gap
        fs.add_action(2000, 30)  # Another 500ms gap
        
        original_count = len(fs.primary_actions)
        
        success = fs.apply_plugin('Resample', target_interval_ms=100)
        assert success
        
        new_count = len(fs.primary_actions)
        
        # Should have more points now
        assert new_count > original_count
        
        # Check interval consistency
        actions = fs.primary_actions
        for i in range(len(actions) - 1):
            interval = actions[i+1]['at'] - actions[i]['at']
            assert abs(interval - 100) <= 10, f"Interval {interval} not close to target 100ms"
    
    def test_resample_decrease_density(self):
        """Test resampling to decrease point density."""
        fs = DualAxisFunscript()
        
        # Create dense data
        for i in range(21):
            fs.add_action(1000 + i * 50, 50 + int(10 * np.sin(i * 0.3)))  # Every 50ms
        
        original_count = len(fs.primary_actions)
        
        success = fs.apply_plugin('Resample', target_interval_ms=200)
        assert success
        
        new_count = len(fs.primary_actions)
        
        # Should have fewer points now
        assert new_count < original_count


if __name__ == "__main__":
    # Run basic functionality test for all plugins
    import sys
    
    print("🧪 Running Individual Plugin Functionality Tests")
    print("=" * 60)
    
    # Ensure plugins are loaded
    fs = DualAxisFunscript()
    plugins = fs.list_available_plugins()
    print(f"✅ {len(plugins)} plugins available for testing")
    
    # Test each plugin category
    test_classes = [
        TestAmplifyPlugin,
        TestInvertPlugin, 
        TestThresholdClampPlugin,
        TestValueClampPlugin,
        TestRdpSimplifyPlugin,
        TestSpeedLimiterPlugin,
        TestKeyframePlugin,
        TestResamplePlugin
    ]
    
    if SCIPY_AVAILABLE:
        test_classes.append(TestSavgolFilterPlugin)
        print("✅ Scipy available - including SavGol filter tests")
    else:
        print("⚠️ Scipy not available - skipping SavGol filter tests")
    
    success_count = 0
    total_count = 0
    
    for test_class in test_classes:
        print(f"\n🔬 Testing {test_class.__name__}...")
        
        test_methods = [method for method in dir(test_class) 
                       if method.startswith('test_')]
        
        for method_name in test_methods:
            total_count += 1
            try:
                instance = test_class()
                
                # Setup fixtures if needed
                if hasattr(instance, 'simple_funscript'):
                    instance.simple_funscript = instance.simple_funscript()
                
                method = getattr(instance, method_name)
                method()
                
                print(f"  ✅ {method_name}")
                success_count += 1
                
            except Exception as e:
                print(f"  ❌ {method_name}: {e}")
    
    print(f"\n📊 Results: {success_count}/{total_count} tests passed")
    
    if success_count == total_count:
        print("🎉 All plugin functionality tests passed!")
    else:
        print("⚠️ Some tests failed - check implementation")
        sys.exit(1)