# Funscript Plugin Development Guide

## 🚀 Quick Start

1. Copy `template_plugin.py` or `advanced_template_plugin.py`
2. Rename the file and class
3. Modify the plugin properties and logic
4. Save the file - it appears automatically in the UI!

## 📁 Plugin System Architecture

```
funscript/
├── plugins/           # Built-in plugins (don't modify)
│   ├── base_plugin.py
│   ├── plugin_loader.py
│   └── *.py           # Built-in plugin implementations
└── user_plugins/      # Your custom plugins go here
    ├── template_plugin.py
    ├── advanced_template_plugin.py
    └── your_plugin.py  # Create new plugins here
```

## 🔧 Plugin Structure

Every plugin must inherit from `FunscriptTransformationPlugin` and implement these properties and methods:

### Required Properties
```python
@property
def name(self) -> str:
    return "unique_plugin_name"  # Must be unique!

@property  
def description(self) -> str:
    return "What your plugin does"

@property
def version(self) -> str:
    return "1.0.0"

@property
def parameters_schema(self) -> Dict[str, Any]:
    # Define your parameters here
    return {}
```

### Required Methods
```python
def transform(self, funscript, axis: str = 'both', **parameters):
    # Your transformation logic here
    pass

def get_preview(self, funscript, axis: str = 'both', **parameters):
    # Preview information for users
    return {"filter_type": "Your Plugin"}
```

## 📋 Parameter Schema Reference

### Parameter Types
```python
'parameter_name': {
    'type': float,        # int, float, bool, str, list
    'required': False,    # True if mandatory
    'default': 1.0,      # Default value
    'description': 'What this parameter does',
    'constraints': {      # Optional validation
        'min': 0.1,
        'max': 5.0,
        'choices': ['option1', 'option2']  # For string types
    }
}
```

### Special Parameters
- `selected_indices`: Automatically passed when user selects points
- `start_time_ms`, `end_time_ms`: For time-based ranges

## 🎨 UI Integration

### Button Names
Plugin names are automatically converted to friendly button names:
- `my_awesome_filter` → "My Awesome Filter"
- `ramp_generator` → "Ramp Generator" 

### Popup vs Direct Buttons
- **Direct buttons**: Plugins with only optional parameters
- **Popup buttons**: Plugins with required parameters or complex options

### Parameter UI Types
- `int`/`float`: Slider input with min/max
- `bool`: Checkbox
- `str` with choices: Dropdown
- `str` without choices: Text input

## 🔍 Preview System

Your `get_preview()` method should return useful information:

```python
def get_preview(self, funscript, axis='both', **parameters):
    return {
        "filter_type": "My Filter",
        "parameters": validated_params,
        "primary_axis": {
            "total_points": 100,
            "points_affected": 50,
            "effect": "Increases amplitude by 25%"
        }
    }
```

## 💡 Development Examples

### Simple Transformation
```python
class SimpleOffsetPlugin(FunscriptTransformationPlugin):
    @property
    def name(self) -> str:
        return "simple_offset"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            'offset': {
                'type': int,
                'required': False, 
                'default': 10,
                'description': 'Amount to offset positions',
                'constraints': {'min': -50, 'max': 50}
            }
        }
    
    def transform(self, funscript, axis='both', **parameters):
        validated = self.validate_parameters(parameters)
        offset = validated['offset']
        
        axes = ['primary', 'secondary'] if axis == 'both' else [axis]
        for ax in axes:
            actions = getattr(funscript, f"{ax}_actions", [])
            for action in actions:
                action['pos'] = np.clip(action['pos'] + offset, 0, 100)
            funscript._invalidate_cache(ax)
        return None
```

### Selection-Aware Plugin
```python
def _get_indices_to_process(self, actions_list, params):
    selected = params.get('selected_indices')
    if selected:
        return [i for i in selected if 0 <= i < len(actions_list)]
    return list(range(len(actions_list)))
```

### Math-Based Transformation
```python
def _apply_sine_wave(self, actions_list, indices, frequency):
    for i, idx in enumerate(indices):
        action = actions_list[idx]
        wave_offset = 20 * math.sin(2 * math.pi * frequency * i / len(indices))
        action['pos'] = np.clip(action['pos'] + wave_offset, 0, 100)
```

## 🛠️ Best Practices

### Parameter Validation
Always validate parameters:
```python
def transform(self, funscript, axis='both', **parameters):
    validated_params = self.validate_parameters(parameters)
    # Now use validated_params instead of parameters
```

### Error Handling
```python
try:
    # Your transformation logic
    pass
except Exception as e:
    self.logger.error(f"Error in {self.name}: {e}")
    raise
```

### Cache Invalidation
Always invalidate cache after modifying actions:
```python
funscript._invalidate_cache(axis)  # or 'both'
```

### Logging
Use the built-in logger:
```python
self.logger.info(f"Applied {self.name} to {len(indices)} points")
self.logger.warning("No points to process")
self.logger.error(f"Error: {e}")
```

## 📊 Advanced Features

### Dependency Management
```python
@property
def requires_scipy(self) -> bool:
    return True  # Plugin needs scipy

@property 
def requires_rdp(self) -> bool:
    return True  # Plugin needs rdp library
```

### Multiple Transformations
```python
@property
def parameters_schema(self):
    return {
        'mode': {
            'type': str,
            'choices': ['amplify', 'smooth', 'quantize'],
            'default': 'amplify'
        }
    }

def transform(self, funscript, axis='both', **parameters):
    mode = parameters['mode']
    if mode == 'amplify':
        self._apply_amplification(...)
    elif mode == 'smooth':
        self._apply_smoothing(...)
    # etc.
```

## 🔍 Testing Your Plugin

1. Save your plugin file in `funscript/user_plugins/`
2. Restart the application (plugins load once at startup)
3. Look for your plugin button in the timeline interface
4. Test with different parameter values
5. Check the preview functionality
6. Test with selected points vs full script

## ❗ Common Pitfalls

1. **Forgetting cache invalidation** - Always call `funscript._invalidate_cache(axis)`
2. **Not validating parameters** - Use `self.validate_parameters()`  
3. **Ignoring selected indices** - Support user selections when provided
4. **Non-unique names** - Plugin names must be unique across all plugins
5. **Modifying axis incorrectly** - Handle 'both', 'primary', 'secondary' properly

## 📚 Real-World Examples

Check the built-in plugins for inspiration:
- `amplify_plugin.py` - Position scaling around center
- `savgol_filter_plugin.py` - Scientific smoothing with scipy
- `clamp_plugin.py` - Multiple plugin classes in one file
- `speed_limiter_plugin.py` - Complex algorithm implementation

## 🎯 Plugin Ideas

- **Pattern generators**: Ramps, stairs, waves
- **Advanced filters**: Kalman, Butterworth, custom smoothing
- **Position mappers**: Logarithmic, exponential scaling  
- **Rhythm processors**: Beat detection, tempo sync
- **Statistical**: Outlier removal, normalization
- **Creative**: Randomization, artistic effects

## 🚀 Publishing Your Plugin

1. Test thoroughly with different scenarios
2. Add comprehensive parameter descriptions
3. Implement useful preview information
4. Include error handling for edge cases
5. Share in the community forums!

---

Happy plugin development! 🎉

The plugin system automatically handles UI generation, parameter validation, preview rendering, and integration with the timeline. Focus on your transformation logic and let the system handle the rest!