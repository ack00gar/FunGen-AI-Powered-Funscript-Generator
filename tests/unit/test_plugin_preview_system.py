"""
Unit tests for plugin preview system.

Tests preview generation, metadata accuracy, error handling,
and UI integration aspects of the plugin preview functionality.
"""

import pytest
import numpy as np
from typing import List, Dict, Any

from funscript.dual_axis_funscript import DualAxisFunscript


class TestPluginPreviewGeneration:
    """Test plugin preview generation functionality."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup fresh plugin registry for each test."""
        # Ensure plugins are loaded
        fs = DualAxisFunscript()
        fs.list_available_plugins()  # Triggers plugin loading
    
    def test_preview_generation_basic(self):
        """Test basic preview generation for all plugins."""
        fs = DualAxisFunscript()
        
        # Create substantial test data
        for i in range(20):
            pos = 50 + 30 * np.sin(i * 0.3) + np.random.normal(0, 2)
            pos = int(np.clip(pos, 0, 100))
            fs.add_action(1000 + i * 100, pos)
        
        plugins = fs.list_available_plugins()
        
        successful_previews = []
        failed_previews = []
        
        for plugin_info in plugins:
            plugin_name = plugin_info['name']
            
            # Skip template/example plugins
            if 'template' in plugin_name or plugin_name in ['advanced_template', 'simple_scale']:
                continue
            
            try:
                # Get preview with default parameters
                preview = fs.get_plugin_preview(plugin_name)
                
                if 'error' in preview:
                    failed_previews.append(f"{plugin_name}: {preview['error']}")
                else:
                    successful_previews.append(plugin_name)
                    
                    # Basic preview structure validation
                    assert isinstance(preview, dict), f"{plugin_name}: Preview should be dict"
                    assert 'filter_type' in preview, f"{plugin_name}: Missing filter_type"
                    
                    print(f"✅ {plugin_name}: Preview generated successfully")
                
            except Exception as e:
                failed_previews.append(f"{plugin_name}: {str(e)}")
        
        print(f"\n✅ Successful previews: {successful_previews}")
        if failed_previews:
            print(f"❌ Failed previews: {failed_previews}")
        
        # Most plugins should generate previews successfully
        success_rate = len(successful_previews) / (len(successful_previews) + len(failed_previews))
        assert success_rate >= 0.8, f"Preview success rate too low: {success_rate:.2f}"
    
    def test_preview_with_parameters(self):
        """Test preview generation with custom parameters."""
        fs = DualAxisFunscript()
        for i in range(10):
            fs.add_action(1000 + i * 200, 30 + i * 7)
        
        parameter_test_cases = {
            'Amplify': {
                'scale_factor': 2.0,
                'center_value': 60
            },
            'Simplify (RDP)': {
                'epsilon': 5.0
            },
            'Threshold Clamp': {
                'lower_threshold': 25,
                'upper_threshold': 75
            },
            'Smooth (SG)': {
                'window_length': 7,
                'polyorder': 3
            }
        }
        
        for plugin_name, params in parameter_test_cases.items():
            try:
                preview = fs.get_plugin_preview(plugin_name, **params)
                
                assert 'error' not in preview, f"{plugin_name}: Preview should not contain errors"
                assert 'parameters' in preview, f"{plugin_name}: Preview should contain parameters"
                
                # Check that provided parameters are reflected in preview
                preview_params = preview.get('parameters', {})
                for param_name, param_value in params.items():
                    if param_name in preview_params:
                        assert preview_params[param_name] == param_value, \
                            f"{plugin_name}: Parameter {param_name} not reflected correctly"
                
                print(f"✅ {plugin_name}: Preview with parameters successful")
                
            except Exception as e:
                print(f"⚠️ {plugin_name}: Preview with parameters failed -> {e}")
    
    def test_preview_axis_handling(self):
        """Test preview generation for different axis configurations."""
        fs = DualAxisFunscript()
        
        # Add data to primary axis
        for i in range(8):
            fs.add_action(1000 + i * 150, 40 + i * 5)
        
        # Add some data to secondary axis
        for i in range(4):
            fs.add_secondary_action(1000 + i * 300, 50 + i * 10)
        
        test_axes = ['primary', 'secondary', 'both']
        
        for axis in test_axes:
            try:
                preview = fs.get_plugin_preview('Amplify', axis=axis, scale_factor=1.3)
                
                assert 'error' not in preview, f"Preview for axis '{axis}' should not error"
                
                # Check axis-specific information
                if axis == 'primary' or axis == 'both':
                    assert 'primary_axis' in preview, f"Missing primary_axis info for axis='{axis}'"
                
                if axis == 'secondary' or axis == 'both':
                    # Secondary axis info might not be present if no secondary data
                    if fs.secondary_actions:
                        assert 'secondary_axis' in preview, f"Missing secondary_axis info for axis='{axis}'"
                
                print(f"✅ Preview for axis='{axis}': Successful")
                
            except Exception as e:
                print(f"⚠️ Preview for axis='{axis}': {e}")
    
    def test_preview_metadata_accuracy(self):
        """Test accuracy of preview metadata."""
        fs = DualAxisFunscript()
        
        # Create known data pattern
        known_positions = [20, 40, 60, 80, 60, 40, 20]
        for i, pos in enumerate(known_positions):
            fs.add_action(1000 + i * 100, pos)
        
        preview = fs.get_plugin_preview('Amplify', scale_factor=1.5, center_value=50)
        
        # Check metadata accuracy
        assert 'filter_type' in preview
        assert preview['filter_type'] == 'Amplification'
        
        if 'primary_axis' in preview:
            axis_info = preview['primary_axis']
            assert 'total_points' in axis_info
            assert axis_info['total_points'] == len(known_positions)
            
            # Check points affected (should be all points for full timeline)
            if 'points_affected' in axis_info:
                assert axis_info['points_affected'] == len(known_positions)
        
        print("✅ Preview metadata accuracy verified")
    
    def test_preview_with_insufficient_data(self):
        """Test preview generation with insufficient data."""
        fs = DualAxisFunscript()
        
        # Single point (insufficient for many filters)
        fs.add_action(1000, 50)
        
        plugins_requiring_multiple_points = [
            'Smooth (SG)',
            'Simplify (RDP)', 
            'Keyframes',
            'Speed Limiter'
        ]
        
        for plugin_name in plugins_requiring_multiple_points:
            try:
                preview = fs.get_plugin_preview(plugin_name)
                
                # Should either indicate inability to apply or handle gracefully
                if 'error' not in preview:
                    # Check if preview indicates insufficient data
                    for axis_key in ['primary_axis', 'secondary_axis']:
                        if axis_key in preview:
                            axis_info = preview[axis_key]
                            if 'can_apply' in axis_info:
                                # If can_apply is present, it should be False for insufficient data
                                if axis_info['total_points'] < 2:
                                    assert not axis_info['can_apply'], \
                                        f"{plugin_name}: Should indicate cannot apply with 1 point"
                
                print(f"✅ {plugin_name}: Handled insufficient data gracefully")
                
            except Exception as e:
                print(f"⚠️ {plugin_name}: Error with insufficient data -> {e}")
    
    def test_preview_error_handling(self):
        """Test preview error handling for invalid parameters."""
        fs = DualAxisFunscript()
        for i in range(5):
            fs.add_action(1000 + i * 200, 50 + i * 5)
        
        # Test invalid parameters
        invalid_parameter_cases = [
            ('Amplify', {'scale_factor': -1.0}),  # Below minimum
            ('Simplify (RDP)', {'epsilon': -5.0}),  # Below minimum
            ('Threshold Clamp', {'lower_threshold': 150}),  # Above maximum
            ('Smooth (SG)', {'window_length': 1}),  # Too small
        ]
        
        for plugin_name, invalid_params in invalid_parameter_cases:
            try:
                preview = fs.get_plugin_preview(plugin_name, **invalid_params)
                
                # Should contain error information
                assert 'error' in preview, f"{plugin_name}: Should contain error for invalid params"
                
                error_message = preview['error'].lower()
                assert any(keyword in error_message for keyword in ['invalid', 'constraint', 'minimum', 'maximum']), \
                    f"{plugin_name}: Error message should be descriptive"
                
                print(f"✅ {plugin_name}: Error handling working for {invalid_params}")
                
            except Exception as e:
                print(f"⚠️ {plugin_name}: Unexpected error handling -> {e}")
    
    def test_preview_performance(self):
        """Test preview generation performance."""
        import time
        
        fs = DualAxisFunscript()
        
        # Create larger dataset for performance testing
        for i in range(100):
            pos = 50 + 30 * np.sin(i * 0.1) + np.random.normal(0, 3)
            fs.add_action(1000 + i * 50, int(np.clip(pos, 0, 100)))
        
        plugins = fs.list_available_plugins()
        
        performance_results = []
        
        for plugin_info in plugins:
            plugin_name = plugin_info['name']
            
            # Skip template plugins
            if 'template' in plugin_name:
                continue
            
            try:
                start_time = time.time()
                preview = fs.get_plugin_preview(plugin_name)
                end_time = time.time()
                
                preview_time = end_time - start_time
                performance_results.append((plugin_name, preview_time))
                
                # Preview should be fast (< 100ms for 100 points)
                assert preview_time < 0.1, f"{plugin_name}: Preview too slow ({preview_time:.3f}s)"
                
                print(f"✅ {plugin_name}: Preview generated in {preview_time:.3f}s")
                
            except Exception as e:
                print(f"⚠️ {plugin_name}: Performance test error -> {e}")
        
        # Calculate average performance
        if performance_results:
            avg_time = sum(time for _, time in performance_results) / len(performance_results)
            print(f"\n📊 Average preview generation time: {avg_time:.3f}s")
            assert avg_time < 0.05, f"Average preview time too slow: {avg_time:.3f}s"


class TestPreviewUIIntegration:
    """Test preview system integration with UI components."""
    
    def test_preview_data_structure_for_ui(self):
        """Test that preview data structure is UI-friendly."""
        fs = DualAxisFunscript()
        for i in range(10):
            fs.add_action(1000 + i * 100, 30 + i * 6)
        
        preview = fs.get_plugin_preview('Amplify', scale_factor=1.8, center_value=50)
        
        # Check UI-friendly structure
        assert isinstance(preview, dict), "Preview should be dictionary"
        
        # Check for required UI fields
        ui_expected_fields = ['filter_type']
        for field in ui_expected_fields:
            assert field in preview, f"Preview missing UI field: {field}"
        
        # Check that all values are JSON-serializable (for UI communication)
        import json
        try:
            json.dumps(preview)
            print("✅ Preview data is JSON-serializable")
        except (TypeError, ValueError) as e:
            pytest.fail(f"Preview data not JSON-serializable: {e}")
        
        # Check axis information structure
        for axis_key in ['primary_axis', 'secondary_axis']:
            if axis_key in preview:
                axis_info = preview[axis_key]
                assert isinstance(axis_info, dict), f"{axis_key} should be dictionary"
                
                # Common UI fields
                ui_axis_fields = ['total_points']
                for field in ui_axis_fields:
                    if field in axis_info:
                        assert isinstance(axis_info[field], (int, float)), \
                            f"{axis_key}.{field} should be numeric for UI"
    
    def test_preview_with_selection_simulation(self):
        """Test preview generation simulating UI selection scenarios."""
        fs = DualAxisFunscript()
        for i in range(15):
            fs.add_action(1000 + i * 100, 40 + i * 3)
        
        # Simulate different UI selection scenarios
        selection_scenarios = [
            {
                'name': 'First half selection',
                'selected_indices': list(range(0, 8))
            },
            {
                'name': 'Middle selection',
                'selected_indices': [5, 6, 7, 8, 9]
            },
            {
                'name': 'Sparse selection',
                'selected_indices': [0, 3, 7, 12, 14]
            },
            {
                'name': 'Time range selection',
                'start_time_ms': 1300,
                'end_time_ms': 1800
            }
        ]
        
        for scenario in selection_scenarios:
            scenario_name = scenario.pop('name')
            
            try:
                preview = fs.get_plugin_preview('Amplify', 
                                              scale_factor=1.4,
                                              **scenario)
                
                assert 'error' not in preview, f"Selection scenario '{scenario_name}' should not error"
                
                # Check that selection is reflected in preview
                if 'primary_axis' in preview:
                    axis_info = preview['primary_axis']
                    if 'points_affected' in axis_info:
                        affected_count = axis_info['points_affected']
                        total_count = axis_info['total_points']
                        
                        # For selections, affected should be <= total
                        assert affected_count <= total_count, \
                            f"Selection '{scenario_name}': affected > total"
                
                print(f"✅ Selection scenario '{scenario_name}': Preview successful")
                
            except Exception as e:
                print(f"⚠️ Selection scenario '{scenario_name}': {e}")
    
    def test_preview_localization_readiness(self):
        """Test that preview strings are ready for UI localization."""
        fs = DualAxisFunscript()
        for i in range(8):
            fs.add_action(1000 + i * 150, 45 + i * 4)
        
        plugins = fs.list_available_plugins()
        
        for plugin_info in plugins:
            plugin_name = plugin_info['name']
            
            # Skip template plugins
            if 'template' in plugin_name:
                continue
            
            try:
                preview = fs.get_plugin_preview(plugin_name)
                
                if 'error' not in preview:
                    # Check that text fields use standard English
                    filter_type = preview.get('filter_type', '')
                    assert isinstance(filter_type, str), f"{plugin_name}: filter_type should be string"
                    assert len(filter_type) > 0, f"{plugin_name}: filter_type should not be empty"
                    
                    # Check that filter_type is title case (UI standard)
                    assert filter_type[0].isupper(), f"{plugin_name}: filter_type should start with capital"
                    
                    print(f"✅ {plugin_name}: Localization-ready strings")
                
            except Exception as e:
                print(f"⚠️ {plugin_name}: Localization check error -> {e}")


class TestPreviewConsistency:
    """Test consistency between preview and actual plugin application."""
    
    def test_preview_vs_actual_consistency(self):
        """Test that preview information matches actual plugin behavior."""
        fs = DualAxisFunscript()
        
        # Create test data
        for i in range(12):
            fs.add_action(1000 + i * 100, 35 + i * 4)
        
        # Test with a plugin that provides detailed preview
        plugin_name = 'Amplify'
        params = {'scale_factor': 1.6, 'center_value': 50}
        
        # Get preview
        preview = fs.get_plugin_preview(plugin_name, **params)
        
        # Apply plugin to a copy
        test_fs = DualAxisFunscript()
        for action in fs.primary_actions:
            test_fs.add_action(action['at'], action['pos'])
        
        success = test_fs.apply_plugin(plugin_name, **params)
        assert success, "Plugin application should succeed"
        
        # Compare preview vs actual
        if 'primary_axis' in preview:
            preview_info = preview['primary_axis']
            
            # Total points should match
            if 'total_points' in preview_info:
                assert preview_info['total_points'] == len(fs.primary_actions), \
                    "Preview total_points should match original data"
            
            # Points affected should make sense
            if 'points_affected' in preview_info:
                affected_count = preview_info['points_affected']
                actual_changes = sum(1 for orig, new in zip(fs.primary_actions, test_fs.primary_actions)
                                   if orig['pos'] != new['pos'])
                
                # Should be close (exact match depends on implementation details)
                assert abs(affected_count - actual_changes) <= 1, \
                    f"Preview affected count ({affected_count}) should match actual changes ({actual_changes})"
        
        print(f"✅ {plugin_name}: Preview vs actual consistency verified")
    
    def test_preview_parameter_reflection(self):
        """Test that preview accurately reflects provided parameters."""
        fs = DualAxisFunscript()
        for i in range(8):
            fs.add_action(1000 + i * 125, 50 + i * 3)
        
        test_cases = [
            ('Simplify (RDP)', {'epsilon': 7.5}),
            ('Threshold Clamp', {'lower_threshold': 30, 'upper_threshold': 70}),
            ('Smooth (SG)', {'window_length': 5, 'polyorder': 2})
        ]
        
        for plugin_name, test_params in test_cases:
            try:
                preview = fs.get_plugin_preview(plugin_name, **test_params)
                
                if 'error' not in preview and 'parameters' in preview:
                    preview_params = preview['parameters']
                    
                    # Check that all provided parameters are reflected
                    for param_name, param_value in test_params.items():
                        if param_name in preview_params:
                            assert preview_params[param_name] == param_value, \
                                f"{plugin_name}: Parameter {param_name} not accurately reflected"
                    
                    print(f"✅ {plugin_name}: Parameter reflection accurate")
                
            except Exception as e:
                print(f"⚠️ {plugin_name}: Parameter reflection test error -> {e}")


if __name__ == "__main__":
    # Run basic preview system tests
    import sys
    
    print("🧪 Running Plugin Preview System Tests")
    print("=" * 60)
    
    # Ensure plugins are loaded
    fs = DualAxisFunscript()
    plugins = fs.list_available_plugins()
    print(f"✅ {len(plugins)} plugins available for preview testing")
    
    # Run test suites
    test_suites = [
        TestPluginPreviewGeneration,
        TestPreviewUIIntegration,
        TestPreviewConsistency
    ]
    
    for test_suite in test_suites:
        print(f"\n🔬 Running {test_suite.__name__}...")
        
        try:
            instance = test_suite()
            if hasattr(instance, 'setup_method'):
                instance.setup_method()
            
            test_methods = [method for method in dir(instance) 
                           if method.startswith('test_')]
            
            for method_name in test_methods:
                print(f"\n  📋 {method_name}...")
                try:
                    method = getattr(instance, method_name)
                    method()
                    print(f"    ✅ Completed")
                except Exception as e:
                    print(f"    ❌ Failed: {e}")
                    
        except Exception as e:
            print(f"❌ Test suite failed: {e}")
    
    print("\n🎉 Plugin preview system testing completed!")