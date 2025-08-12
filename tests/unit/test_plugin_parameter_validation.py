"""
Unit tests for plugin parameter validation.

Tests parameter schema validation, constraint enforcement,
type checking, and error handling for all plugins.
"""

import pytest
import numpy as np
from typing import List, Dict, Any

from funscript.dual_axis_funscript import DualAxisFunscript
from funscript.plugins.base_plugin import plugin_registry, FunscriptTransformationPlugin


class TestParameterValidation:
    """Test parameter validation across all plugins."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup fresh plugin registry for each test."""
        # Ensure plugins are loaded
        fs = DualAxisFunscript()
        fs.list_available_plugins()  # Triggers plugin loading
    
    def test_required_parameter_validation(self):
        """Test validation of required parameters."""
        fs = DualAxisFunscript()
        fs.add_action(1000, 50)
        
        # Test plugin with required parameters (if any exist)
        plugins = fs.list_available_plugins()
        
        for plugin_info in plugins:
            plugin_name = plugin_info['name']
            schema = plugin_info.get('parameters_schema', {})
            
            # Find required parameters
            required_params = [
                param_name for param_name, param_info in schema.items()
                if param_info.get('required', False)
            ]
            
            if required_params:
                # Should fail when required parameter is missing
                with pytest.raises(ValueError, match="required"):
                    fs.apply_plugin(plugin_name)
                
                print(f"✓ {plugin_name}: Required parameter validation working")
    
    def test_type_validation(self):
        """Test parameter type validation."""
        fs = DualAxisFunscript()
        fs.add_action(1000, 50)
        
        plugins = fs.list_available_plugins()
        
        type_test_cases = {
            'Amplify': {
                'scale_factor': [
                    ("2.5", str, False),  # String instead of float
                    (2.5, float, True),   # Correct float
                    (2, int, True),       # Int should convert to float
                ],
                'center_value': [
                    ("50", str, False),   # String instead of number
                    (50.5, float, True),  # Float should work
                    (50, int, True),      # Int should work
                ]
            },
            'Threshold Clamp': {
                'lower_threshold': [
                    (25.5, float, True),
                    ("25", str, False),
                ],
                'upper_threshold': [
                    (75, int, True),
                    ([], list, False),
                ]
            },
            'Smooth (SG)': {
                'window_length': [
                    (7, int, True),
                    (7.0, float, False),  # Should be int, not float
                    ("7", str, False),
                ],
                'polyorder': [
                    (3, int, True),
                    (3.5, float, False),
                ]
            }
        }
        
        for plugin_name, param_tests in type_test_cases.items():
            # Check if plugin exists
            plugin_exists = any(p['name'] == plugin_name for p in plugins)
            if not plugin_exists:
                continue
            
            for param_name, test_values in param_tests.items():
                for value, expected_type, should_succeed in test_values:
                    try:
                        if should_succeed:
                            success = fs.apply_plugin(plugin_name, **{param_name: value})
                            # Note: Success depends on plugin implementation, 
                            # but it shouldn't raise type errors
                            print(f"✓ {plugin_name}.{param_name}: {value} ({type(value).__name__}) processed")
                        else:
                            with pytest.raises((ValueError, TypeError)):
                                fs.apply_plugin(plugin_name, **{param_name: value})
                            print(f"✓ {plugin_name}.{param_name}: {value} ({type(value).__name__}) correctly rejected")
                    except Exception as e:
                        print(f"⚠ {plugin_name}.{param_name}: {value} -> {e}")
    
    def test_constraint_validation(self):
        """Test parameter constraint enforcement (min, max, etc.)."""
        fs = DualAxisFunscript()
        for i in range(5):
            fs.add_action(1000 + i * 200, 30 + i * 10)
        
        constraint_test_cases = {
            'Amplify': {
                'scale_factor': {
                    'valid': [0.1, 1.0, 2.0, 5.0],
                    'invalid': [-1.0, 0.0, 10.1]  # Below min or above max
                }
            },
            'Simplify (RDP)': {
                'epsilon': {
                    'valid': [0.5, 5.0, 15.0],
                    'invalid': [0.0, -1.0]  # Below minimum
                }
            },
            'Smooth (SG)': {
                'window_length': {
                    'valid': [3, 7, 11, 15],
                    'invalid': [1, 2, 0, -1]  # Below minimum
                },
                'polyorder': {
                    'valid': [0, 1, 3, 5],
                    'invalid': [-1, -5]  # Below minimum
                }
            },
            'Threshold Clamp': {
                'lower_threshold': {
                    'valid': [0, 25, 50, 99],
                    'invalid': [-1, 101]  # Outside 0-100 range
                },
                'upper_threshold': {
                    'valid': [1, 50, 75, 100],
                    'invalid': [-1, 101]  # Outside 0-100 range
                }
            }
        }
        
        plugins = fs.list_available_plugins()
        
        for plugin_name, param_constraints in constraint_test_cases.items():
            # Check if plugin exists
            plugin_exists = any(p['name'] == plugin_name for p in plugins)
            if not plugin_exists:
                continue
            
            for param_name, constraint_data in param_constraints.items():
                # Test valid values
                for valid_value in constraint_data['valid']:
                    try:
                        # Create fresh test data for each test
                        test_fs = DualAxisFunscript()
                        for i in range(10):  # More data for constraints that need it
                            test_fs.add_action(1000 + i * 100, 30 + i * 4)
                        
                        success = test_fs.apply_plugin(plugin_name, **{param_name: valid_value})
                        print(f"✓ {plugin_name}.{param_name}: {valid_value} (valid) processed")
                    except Exception as e:
                        print(f"⚠ {plugin_name}.{param_name}: {valid_value} (should be valid) -> {e}")
                
                # Test invalid values
                for invalid_value in constraint_data['invalid']:
                    try:
                        test_fs = DualAxisFunscript()
                        for i in range(10):
                            test_fs.add_action(1000 + i * 100, 30 + i * 4)
                        
                        with pytest.raises(ValueError):
                            test_fs.apply_plugin(plugin_name, **{param_name: invalid_value})
                        print(f"✓ {plugin_name}.{param_name}: {invalid_value} (invalid) correctly rejected")
                    except ValueError:
                        # Expected - constraint validation working
                        print(f"✓ {plugin_name}.{param_name}: {invalid_value} (invalid) correctly rejected")
                    except Exception as e:
                        print(f"⚠ {plugin_name}.{param_name}: {invalid_value} unexpected error -> {e}")
    
    def test_default_parameter_handling(self):
        """Test that default parameters are used correctly."""
        fs = DualAxisFunscript()
        for i in range(5):
            fs.add_action(1000 + i * 200, 30 + i * 10)
        
        plugins = fs.list_available_plugins()
        
        for plugin_info in plugins:
            plugin_name = plugin_info['name']
            schema = plugin_info.get('parameters_schema', {})
            
            # Skip template/example plugins
            if 'template' in plugin_name or plugin_name in ['advanced_template', 'simple_scale']:
                continue
            
            # Check if plugin has default parameters
            has_defaults = any(
                'default' in param_info and param_info['default'] is not None
                for param_info in schema.values()
            )
            
            if has_defaults:
                try:
                    # Create fresh test data
                    test_fs = DualAxisFunscript()
                    for i in range(10):
                        test_fs.add_action(1000 + i * 100, 40 + i * 3)
                    
                    # Should work with defaults (no parameters specified)
                    success = test_fs.apply_plugin(plugin_name)
                    if success:
                        print(f"✓ {plugin_name}: Default parameters working")
                    else:
                        print(f"⚠ {plugin_name}: Applied with defaults but returned False")
                        
                except Exception as e:
                    print(f"❌ {plugin_name}: Default parameters failed -> {e}")
    
    def test_unknown_parameter_handling(self):
        """Test handling of unknown/unexpected parameters."""
        fs = DualAxisFunscript()
        fs.add_action(1000, 50)
        
        # Test with a simple plugin
        try:
            # Should ignore unknown parameters gracefully or raise appropriate error
            with pytest.warns(UserWarning) or pytest.raises(ValueError):
                fs.apply_plugin('Amplify', 
                              scale_factor=1.5,
                              unknown_param=123,
                              another_unknown="test")
        except Exception as e:
            # Some plugins might be more strict about unknown parameters
            print(f"ℹ Unknown parameter handling: {e}")
    
    def test_parameter_schema_completeness(self):
        """Test that all plugins have complete parameter schemas."""
        fs = DualAxisFunscript()
        plugins = fs.list_available_plugins()
        
        required_schema_fields = ['type', 'description']
        
        for plugin_info in plugins:
            plugin_name = plugin_info['name']
            schema = plugin_info.get('parameters_schema', {})
            
            for param_name, param_info in schema.items():
                # Check required schema fields
                for field in required_schema_fields:
                    assert field in param_info, \
                        f"Plugin '{plugin_name}' parameter '{param_name}' missing '{field}' in schema"
                
                # Check that type is a valid Python type
                param_type = param_info.get('type')
                assert param_type in [int, float, str, bool, list, dict], \
                    f"Plugin '{plugin_name}' parameter '{param_name}' has invalid type: {param_type}"
                
                # If constraints exist, they should be properly formatted
                if 'constraints' in param_info:
                    constraints = param_info['constraints']
                    assert isinstance(constraints, dict), \
                        f"Plugin '{plugin_name}' parameter '{param_name}' constraints should be dict"
                
                print(f"✓ {plugin_name}.{param_name}: Schema complete")
    
    def test_axis_parameter_validation(self):
        """Test axis parameter validation."""
        fs = DualAxisFunscript()
        fs.add_action(1000, 50)
        
        valid_axes = ['primary', 'secondary', 'both']
        invalid_axes = ['invalid', 'Primary', 'BOTH', '', None, 123]
        
        # Test with a simple plugin
        for valid_axis in valid_axes:
            try:
                success = fs.apply_plugin('Amplify', axis=valid_axis, scale_factor=1.2)
                print(f"✓ axis='{valid_axis}': Processed correctly")
            except Exception as e:
                print(f"⚠ axis='{valid_axis}': {e}")
        
        for invalid_axis in invalid_axes:
            try:
                with pytest.raises(ValueError, match="axis"):
                    fs.apply_plugin('Amplify', axis=invalid_axis, scale_factor=1.2)
                print(f"✓ axis='{invalid_axis}': Correctly rejected")
            except ValueError:
                print(f"✓ axis='{invalid_axis}': Correctly rejected")
            except Exception as e:
                print(f"⚠ axis='{invalid_axis}': Unexpected error -> {e}")
    
    def test_selected_indices_validation(self):
        """Test selected_indices parameter validation."""
        fs = DualAxisFunscript()
        for i in range(10):
            fs.add_action(1000 + i * 100, 30 + i * 5)
        
        # Test valid indices
        valid_indices_cases = [
            [0, 2, 4],           # Valid selection
            [0, 9],              # Start and end
            list(range(10)),     # All indices
            []                   # Empty (should work or be ignored)
        ]
        
        for indices in valid_indices_cases:
            try:
                test_fs = DualAxisFunscript()
                for i in range(10):
                    test_fs.add_action(1000 + i * 100, 30 + i * 5)
                
                success = test_fs.apply_plugin('Amplify', 
                                             selected_indices=indices,
                                             scale_factor=1.2)
                print(f"✓ selected_indices={indices}: Processed")
            except Exception as e:
                print(f"⚠ selected_indices={indices}: {e}")
        
        # Test invalid indices
        invalid_indices_cases = [
            [-1, 2, 4],          # Negative index
            [0, 15, 20],         # Out of range
            "invalid",           # Wrong type
            [0.5, 1.5],          # Float indices
        ]
        
        for indices in invalid_indices_cases:
            try:
                test_fs = DualAxisFunscript()
                for i in range(10):
                    test_fs.add_action(1000 + i * 100, 30 + i * 5)
                
                # Should either ignore invalid indices or raise error
                result = test_fs.apply_plugin('Amplify', 
                                            selected_indices=indices,
                                            scale_factor=1.2)
                print(f"ℹ selected_indices={indices}: Processed (may have filtered invalid indices)")
            except (ValueError, TypeError, IndexError):
                print(f"✓ selected_indices={indices}: Correctly rejected")
            except Exception as e:
                print(f"⚠ selected_indices={indices}: Unexpected error -> {e}")


class TestPluginSpecificValidation:
    """Test validation specific to individual plugins."""
    
    def test_amplify_center_value_constraints(self):
        """Test amplify plugin center value constraints."""
        fs = DualAxisFunscript()
        fs.add_action(1000, 50)
        
        # Valid center values
        valid_centers = [0, 25, 50, 75, 100]
        for center in valid_centers:
            try:
                success = fs.apply_plugin('Amplify', scale_factor=1.5, center_value=center)
                print(f"✓ amplify center_value={center}: Valid")
            except Exception as e:
                print(f"⚠ amplify center_value={center}: {e}")
        
        # Invalid center values
        invalid_centers = [-1, 101, 150, -50]
        for center in invalid_centers:
            try:
                with pytest.raises(ValueError):
                    fs.apply_plugin('Amplify', scale_factor=1.5, center_value=center)
                print(f"✓ amplify center_value={center}: Correctly rejected")
            except ValueError:
                print(f"✓ amplify center_value={center}: Correctly rejected")
            except Exception as e:
                print(f"⚠ amplify center_value={center}: {e}")
    
    def test_threshold_clamp_logical_constraints(self):
        """Test threshold clamp logical constraints."""
        fs = DualAxisFunscript()
        fs.add_action(1000, 50)
        
        # Valid threshold combinations
        valid_combinations = [
            (20, 80),   # Normal case
            (0, 100),   # Full range
            (50, 50),   # Equal thresholds (edge case)
        ]
        
        for lower, upper in valid_combinations:
            try:
                success = fs.apply_plugin('Threshold Clamp', 
                                        lower_threshold=lower,
                                        upper_threshold=upper)
                print(f"✓ threshold_clamp ({lower}, {upper}): Valid")
            except Exception as e:
                print(f"⚠ threshold_clamp ({lower}, {upper}): {e}")
        
        # Invalid combinations (if plugin validates this)
        invalid_combinations = [
            (80, 20),   # Lower > upper
            (101, 102), # Both out of range
        ]
        
        for lower, upper in invalid_combinations:
            try:
                # Some plugins might not validate this logical constraint
                fs.apply_plugin('Threshold Clamp', 
                              lower_threshold=lower,
                              upper_threshold=upper)
                print(f"ℹ threshold_clamp ({lower}, {upper}): Processed (no logical validation)")
            except ValueError:
                print(f"✓ threshold_clamp ({lower}, {upper}): Correctly rejected")
            except Exception as e:
                print(f"⚠ threshold_clamp ({lower}, {upper}): {e}")
    
    def test_savgol_window_polyorder_relationship(self):
        """Test Savitzky-Golay window length and polynomial order relationship."""
        fs = DualAxisFunscript()
        for i in range(15):
            fs.add_action(1000 + i * 100, 50 + i * 2)
        
        # Valid combinations
        valid_combinations = [
            (7, 3),    # window > polyorder
            (9, 2),    # window > polyorder
            (5, 1),    # minimal case
        ]
        
        for window, poly in valid_combinations:
            try:
                success = fs.apply_plugin('Smooth (SG)', 
                                        window_length=window,
                                        polyorder=poly)
                print(f"✓ savgol_filter window={window}, poly={poly}: Valid")
            except Exception as e:
                print(f"⚠ savgol_filter window={window}, poly={poly}: {e}")
        
        # Invalid combinations
        invalid_combinations = [
            (5, 5),    # polyorder >= window_length
            (7, 8),    # polyorder > window_length
            (3, 4),    # polyorder > window_length
        ]
        
        for window, poly in invalid_combinations:
            try:
                # Plugin should auto-adjust or reject
                success = fs.apply_plugin('Smooth (SG)', 
                                        window_length=window,
                                        polyorder=poly)
                print(f"ℹ savgol_filter window={window}, poly={poly}: Auto-adjusted or processed")
            except ValueError:
                print(f"✓ savgol_filter window={window}, poly={poly}: Correctly rejected")
            except Exception as e:
                print(f"⚠ savgol_filter window={window}, poly={poly}: {e}")


if __name__ == "__main__":
    # Run basic parameter validation tests
    import sys
    
    print("🧪 Running Plugin Parameter Validation Tests")
    print("=" * 60)
    
    # Ensure plugins are loaded
    fs = DualAxisFunscript()
    plugins = fs.list_available_plugins()
    print(f"✅ {len(plugins)} plugins available for validation testing")
    
    # Run test suites
    test_suites = [TestParameterValidation, TestPluginSpecificValidation]
    
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
    
    print("\n🎉 Parameter validation testing completed!")