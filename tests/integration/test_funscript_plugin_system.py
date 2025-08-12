"""
Comprehensive integration tests for the funscript plugin system.

Tests all aspects of plugin loading, discovery, application, and integration
with the timeline and UI components.
"""

import pytest
import numpy as np
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

from funscript.dual_axis_funscript import DualAxisFunscript
from funscript.plugins.base_plugin import FunscriptTransformationPlugin, plugin_registry
from funscript.plugins.plugin_loader import plugin_loader


class TestPluginSystemIntegration:
    """Test the complete plugin system integration."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup fresh plugin registry for each test."""
        # Clear plugin registry for clean tests
        plugin_registry._plugins.clear()
        plugin_registry._global_plugins_loaded = False
        
    def test_plugin_discovery_and_loading(self):
        """Test that all plugins are discovered and loaded correctly."""
        funscript = DualAxisFunscript()
        
        # Load plugins
        available_plugins = funscript.list_available_plugins()
        
        # Should find both built-in and user plugins
        assert len(available_plugins) >= 10, "Should discover at least 10 plugins"
        
        # Check for expected built-in plugins
        expected_plugins = {
            'Smooth (SG)', 'Simplify (RDP)', 'Speed Limiter', 'sg_autotune',
            'Amplify', 'Threshold Clamp', 'value_clamp', 'Invert', 
            'Keyframes', 'Resample'
        }
        
        found_plugins = {plugin['name'] for plugin in available_plugins}
        missing_plugins = expected_plugins - found_plugins
        
        assert len(missing_plugins) == 0, f"Missing plugins: {missing_plugins}"
        
        # Check plugin metadata structure
        for plugin in available_plugins:
            assert 'name' in plugin
            assert 'description' in plugin
            assert 'version' in plugin
            assert 'parameters_schema' in plugin
            assert isinstance(plugin['parameters_schema'], dict)
    
    def test_plugin_parameter_validation(self):
        """Test plugin parameter validation works correctly."""
        funscript = DualAxisFunscript()
        funscript.add_action(1000, 50)
        
        # Test valid parameters
        success = funscript.apply_plugin('Amplify', scale_factor=1.5, center_value=50)
        assert success, "Should succeed with valid parameters"
        
        # Test invalid parameters
        with pytest.raises(ValueError):
            funscript.apply_plugin('Amplify', scale_factor=-1.0)  # Below minimum
        
        with pytest.raises(ValueError):
            funscript.apply_plugin('Amplify', scale_factor=10.0)  # Above maximum
    
    def test_plugin_application_full_timeline(self):
        """Test plugins apply to full timeline by default."""
        funscript = DualAxisFunscript()
        original_positions = [20, 80, 30, 70, 40]
        
        for i, pos in enumerate(original_positions):
            funscript.add_action(1000 + i * 200, pos)
        
        # Test amplify applies to all points
        success = funscript.apply_plugin('Amplify', scale_factor=1.5, center_value=50)
        assert success
        
        final_positions = [action['pos'] for action in funscript.primary_actions]
        
        # All positions should be modified
        assert final_positions != original_positions, "All positions should be modified"
        assert len(final_positions) == len(original_positions), "Should preserve all points"
    
    def test_plugin_application_with_selection(self):
        """Test plugins work correctly with selected indices."""
        funscript = DualAxisFunscript()
        original_positions = [20, 80, 30, 70, 40]
        
        for i, pos in enumerate(original_positions):
            funscript.add_action(1000 + i * 200, pos)
        
        # Apply to selected indices only
        success = funscript.apply_plugin('Amplify', 
                                       scale_factor=2.0, 
                                       center_value=50,
                                       selected_indices=[0, 2, 4])  # First, third, fifth
        assert success
        
        final_positions = [action['pos'] for action in funscript.primary_actions]
        
        # Only selected positions should be modified
        assert final_positions[1] == original_positions[1], "Unselected point should be unchanged"
        assert final_positions[3] == original_positions[3], "Unselected point should be unchanged"
        assert final_positions[0] != original_positions[0], "Selected point should be changed"
        assert final_positions[2] != original_positions[2], "Selected point should be changed"
        assert final_positions[4] != original_positions[4], "Selected point should be changed"
    
    def test_plugin_preview_system(self):
        """Test plugin preview generation works correctly."""
        funscript = DualAxisFunscript()
        for i in range(5):
            funscript.add_action(1000 + i * 200, 30 + i * 10)
        
        # Test preview for amplify plugin
        preview = funscript.get_plugin_preview('Amplify', scale_factor=1.5, center_value=50)
        
        assert 'error' not in preview, "Preview should not contain errors"
        assert 'filter_type' in preview, "Preview should contain filter type"
        assert preview['filter_type'] == 'Amplification'
        
        # Should contain axis-specific information
        if 'primary_axis' in preview:
            axis_info = preview['primary_axis']
            assert 'total_points' in axis_info
            assert axis_info['total_points'] == 5
            assert 'points_affected' in axis_info
    
    def test_all_plugins_functional(self):
        """Test that all discovered plugins can be applied without errors."""
        funscript = DualAxisFunscript()
        
        # Create substantial test data for all plugins to work with
        for i in range(20):
            pos = 50 + 30 * np.sin(i * 0.3) + np.random.normal(0, 2)
            pos = int(np.clip(pos, 0, 100))
            funscript.add_action(1000 + i * 100, pos)
        
        available_plugins = funscript.list_available_plugins()
        
        successful_plugins = []
        failed_plugins = []
        
        for plugin_info in available_plugins:
            plugin_name = plugin_info['name']
            
            # Skip template/example plugins
            if 'template' in plugin_name or plugin_name in ['advanced_template', 'simple_scale']:
                continue
            
            # Create fresh copy for each plugin test
            test_funscript = DualAxisFunscript()
            for i in range(20):
                pos = 50 + 30 * np.sin(i * 0.3)
                pos = int(np.clip(pos, 0, 100))
                test_funscript.add_action(1000 + i * 100, pos)
            
            try:
                # Get default parameters
                default_params = {}
                schema = plugin_info.get('parameters_schema', {})
                for param_name, param_info in schema.items():
                    if 'default' in param_info and param_info['default'] is not None:
                        default_params[param_name] = param_info['default']
                
                success = test_funscript.apply_plugin(plugin_name, **default_params)
                
                if success:
                    successful_plugins.append(plugin_name)
                else:
                    failed_plugins.append(f"{plugin_name}: returned False")
                    
            except Exception as e:
                failed_plugins.append(f"{plugin_name}: {str(e)}")
        
        print(f"\n✅ Successful plugins ({len(successful_plugins)}): {successful_plugins}")
        if failed_plugins:
            print(f"❌ Failed plugins ({len(failed_plugins)}): {failed_plugins}")
        
        # Most plugins should work
        success_rate = len(successful_plugins) / (len(successful_plugins) + len(failed_plugins))
        assert success_rate >= 0.8, f"Plugin success rate too low: {success_rate:.2f}"
    
    def test_plugin_error_handling(self):
        """Test plugin error handling for insufficient data."""
        funscript = DualAxisFunscript()
        funscript.add_action(1000, 50)  # Only one point
        
        # Plugins requiring multiple points should fail gracefully
        plugins_requiring_multiple_points = ['Smooth (SG)', 'Simplify (RDP)', 'Keyframes']
        
        for plugin_name in plugins_requiring_multiple_points:
            try:
                success = funscript.apply_plugin(plugin_name)
                # Should either fail (return False) or raise appropriate error
                if success:
                    # If it claims success, data should be unchanged (since can't process 1 point)
                    assert len(funscript.primary_actions) == 1
                    assert funscript.primary_actions[0]['pos'] == 50
            except (ValueError, RuntimeError) as e:
                # Expected - plugin should raise appropriate error
                assert "insufficient" in str(e).lower() or "not enough" in str(e).lower()
    
    def test_rdp_performance_optimization(self):
        """Test that RDP uses the fast numpy implementation."""
        funscript = DualAxisFunscript()
        
        # Generate larger dataset for performance testing
        for i in range(200):
            pos = 50 + 40 * np.sin(i * 0.1) + np.random.normal(0, 3)
            pos = int(np.clip(pos, 0, 100))
            funscript.add_action(1000 + i * 50, pos)
        
        original_count = len(funscript.primary_actions)
        
        start_time = time.time()
        success = funscript.apply_plugin('Simplify (RDP)', epsilon=3.0)
        end_time = time.time()
        
        assert success, "RDP should succeed"
        
        processing_time = end_time - start_time
        final_count = len(funscript.primary_actions)
        
        # Should be very fast (< 100ms for 200 points)
        assert processing_time < 0.1, f"RDP too slow: {processing_time:.3f}s for {original_count} points"
        
        # Should reduce points significantly
        reduction_pct = ((original_count - final_count) / original_count) * 100
        assert reduction_pct > 10, f"RDP should reduce points significantly, got {reduction_pct:.1f}%"
        
        # Calculate points per second
        points_per_second = original_count / processing_time
        assert points_per_second > 10000, f"RDP performance too low: {points_per_second:.0f} points/sec"
    
    def test_plugin_cache_invalidation(self):
        """Test that plugins properly invalidate funscript caches."""
        funscript = DualAxisFunscript()
        
        # Add test data
        for i in range(5):
            funscript.add_action(1000 + i * 200, 50 + i * 10)
        
        # Trigger cache population
        _ = funscript._get_timestamps_for_axis('primary')
        
        # Apply plugin
        success = funscript.apply_plugin('Amplify', scale_factor=1.5)
        assert success
        
        # Cache should be properly invalidated and repopulated
        timestamps = funscript._get_timestamps_for_axis('primary')
        assert len(timestamps) == 5, "Cache should contain correct number of timestamps"
    
    def test_plugin_axis_handling(self):
        """Test plugins handle different axis configurations correctly."""
        funscript = DualAxisFunscript()
        
        # Add data to primary axis only
        for i in range(5):
            funscript.add_action(1000 + i * 200, 30 + i * 10)
        
        # Test axis='primary'
        success = funscript.apply_plugin('Amplify', axis='primary', scale_factor=1.2)
        assert success
        
        # Test axis='secondary' (should work even with no secondary data)
        success = funscript.apply_plugin('Amplify', axis='secondary', scale_factor=1.2)
        assert success  # Should not fail, just do nothing for empty secondary
        
        # Test axis='both'
        success = funscript.apply_plugin('Amplify', axis='both', scale_factor=1.2)
        assert success


class TestPluginLoaderSystem:
    """Test the plugin loader functionality."""
    
    def test_builtin_plugin_loading(self):
        """Test that built-in plugins load correctly."""
        loader = plugin_loader
        
        results = loader.load_builtin_plugins()
        
        # Should load multiple plugins successfully
        assert len(results) >= 8, "Should load at least 8 built-in plugins"
        
        # All should succeed (True values)
        successful_loads = sum(1 for success in results.values() if success)
        assert successful_loads == len(results), "All built-in plugins should load successfully"
    
    def test_user_plugin_loading(self):
        """Test that user plugins load correctly."""
        loader = plugin_loader
        
        results = loader.load_user_plugins()
        
        # Should at least attempt to load user plugins directory
        # May be empty, but should not error
        assert isinstance(results, dict), "Should return results dictionary"
    
    def test_plugin_registry_functionality(self):
        """Test the plugin registry works correctly."""
        from funscript.plugins.base_plugin import PluginRegistry
        
        registry = PluginRegistry()
        
        # Test empty registry
        assert len(registry.list_plugins()) == 0
        
        # Create mock plugin for testing
        class MockPlugin(FunscriptTransformationPlugin):
            @property
            def name(self): return "test_plugin"
            @property
            def description(self): return "Test plugin"
            @property
            def version(self): return "1.0.0"
            @property
            def parameters_schema(self): return {}
            def transform(self, funscript, axis='both', **parameters): return None
        
        mock_plugin = MockPlugin()
        
        # Test registration
        success = registry.register(mock_plugin)
        assert success, "Plugin registration should succeed"
        
        # Test retrieval
        retrieved = registry.get_plugin("test_plugin")
        assert retrieved is not None, "Should retrieve registered plugin"
        assert retrieved.name == "test_plugin"
        
        # Test listing
        plugins = registry.list_plugins()
        assert len(plugins) == 1
        assert plugins[0]['name'] == "test_plugin"
        
        # Test unregistration
        success = registry.unregister("test_plugin")
        assert success, "Plugin unregistration should succeed"
        
        assert len(registry.list_plugins()) == 0, "Registry should be empty after unregistration"


class TestPluginPerformance:
    """Test plugin performance characteristics."""
    
    @pytest.fixture
    def large_funscript(self):
        """Create a large funscript for performance testing."""
        funscript = DualAxisFunscript()
        
        # Generate 1000 points with realistic data
        for i in range(1000):
            # Sinusoidal base with noise
            pos = 50 + 30 * np.sin(i * 0.05) + 20 * np.sin(i * 0.15) + np.random.normal(0, 3)
            pos = int(np.clip(pos, 0, 100))
            funscript.add_action(1000 + i * 50, pos)
        
        return funscript
    
    def test_plugin_loading_performance(self):
        """Test that plugin loading is fast."""
        # Clear registry
        plugin_registry._plugins.clear()
        plugin_registry._global_plugins_loaded = False
        
        start_time = time.time()
        
        funscript = DualAxisFunscript()
        plugins = funscript.list_available_plugins()
        
        end_time = time.time()
        
        loading_time = end_time - start_time
        
        assert loading_time < 1.0, f"Plugin loading too slow: {loading_time:.3f}s"
        assert len(plugins) > 5, "Should load multiple plugins"
    
    def test_fast_plugins_performance(self, large_funscript):
        """Test performance of fast plugins on large datasets."""
        fast_plugins = {
            'Amplify': {'scale_factor': 1.2},
            'Invert': {},
            'Threshold Clamp': {'lower_threshold': 20, 'upper_threshold': 80},
            'value_clamp': {'clamp_value': 50}
        }
        
        for plugin_name, params in fast_plugins.items():
            test_funscript = DualAxisFunscript()
            # Copy data
            for action in large_funscript.primary_actions:
                test_funscript.add_action(action['at'], action['pos'])
            
            start_time = time.time()
            success = test_funscript.apply_plugin(plugin_name, **params)
            end_time = time.time()
            
            processing_time = end_time - start_time
            points_per_second = 1000 / processing_time
            
            assert success, f"{plugin_name} should succeed"
            assert processing_time < 0.1, f"{plugin_name} too slow: {processing_time:.3f}s"
            assert points_per_second > 50000, f"{plugin_name} performance: {points_per_second:.0f} pts/sec"
    
    def test_rdp_performance_scaling(self):
        """Test RDP performance scales well with data size."""
        data_sizes = [100, 500, 1000]
        
        for size in data_sizes:
            funscript = DualAxisFunscript()
            
            # Generate test data
            for i in range(size):
                pos = 50 + 30 * np.sin(i * 0.1) + np.random.normal(0, 2)
                pos = int(np.clip(pos, 0, 100))
                funscript.add_action(1000 + i * 50, pos)
            
            start_time = time.time()
            success = funscript.apply_plugin('Simplify (RDP)', epsilon=3.0)
            end_time = time.time()
            
            processing_time = end_time - start_time
            points_per_second = size / processing_time
            
            assert success, f"RDP should succeed for {size} points"
            assert points_per_second > 10000, f"RDP performance: {points_per_second:.0f} pts/sec for {size} points"


if __name__ == "__main__":
    # Run basic functionality test
    import sys
    
    print("🧪 Running Plugin System Integration Tests")
    print("=" * 60)
    
    try:
        # Test plugin discovery
        funscript = DualAxisFunscript()
        plugins = funscript.list_available_plugins()
        print(f"✅ Discovered {len(plugins)} plugins")
        
        # Test a few key plugins
        test_plugins = ['Amplify', 'Invert', 'Simplify (RDP)']
        
        for plugin_name in test_plugins:
            # Create test data
            test_fs = DualAxisFunscript()
            for i in range(10):
                test_fs.add_action(1000 + i*100, 30 + i*5)
            
            success = test_fs.apply_plugin(plugin_name)
            if success:
                print(f"✅ {plugin_name}: Working")
            else:
                print(f"❌ {plugin_name}: Failed")
        
        print("\n🎉 Plugin system integration test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Plugin system test failed: {e}")
        sys.exit(1)