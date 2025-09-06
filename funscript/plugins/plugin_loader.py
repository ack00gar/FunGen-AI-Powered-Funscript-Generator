"""
Plugin loader system for funscript transformations.

This module provides functionality to automatically discover and load
funscript transformation plugins, both built-in and user-defined.
"""

import os
import sys
import importlib
import importlib.util
import inspect
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Type

from .base_plugin import FunscriptTransformationPlugin, plugin_registry


class PluginLoader:
    """
    Loads and manages funscript transformation plugins.
    
    Can load plugins from:
    - Built-in plugins in the plugins directory
    - User-defined plugins in specified directories
    - Individual plugin files
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger('PluginLoader')
        self.loaded_modules = {}
    
    def load_builtin_plugins(self) -> Dict[str, bool]:
        """
        Load all built-in plugins from the plugins directory using auto-discovery.
        
        Returns:
            Dictionary mapping plugin names to load success status
        """
        plugins_dir = Path(__file__).parent
        return self.load_plugins_from_directory(str(plugins_dir), recursive=False)
    
    def load_plugins_from_directory(self, directory: str, recursive: bool = False) -> Dict[str, bool]:
        """
        Load all plugins from a directory.
        
        Args:
            directory: Path to directory containing plugin files
            recursive: If True, search subdirectories as well
            
        Returns:
            Dictionary mapping plugin file names to load success status
        """
        results = {}
        directory_path = Path(directory)
        
        if not directory_path.exists() or not directory_path.is_dir():
            self.logger.warning(f"Plugin directory does not exist: {directory}")
            return results
        
        pattern = "**/*.py" if recursive else "*.py"
        
        # Collect all plugin files first
        plugin_files = []
        for plugin_file in directory_path.glob(pattern):
            if plugin_file.name.startswith('_'):  # Skip private files
                continue
            if plugin_file.name in ['base_plugin.py', 'plugin_loader.py']:  # Skip infrastructure files
                continue
            plugin_files.append(plugin_file)
        
        # Log summary of what we're loading
        if plugin_files:
            self.logger.info(f"Loading {len(plugin_files)} plugins from: {directory}")
        
        # Load each plugin
        for plugin_file in plugin_files:
            success = self.load_plugin_from_file(plugin_file)
            results[plugin_file.name] = success
        
        return results
    
    def load_plugin_from_file(self, file_path: str) -> bool:
        """
        Load a single plugin from a file.
        
        Args:
            file_path: Path to the plugin file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            self.logger.error(f"Plugin file does not exist: {file_path}")
            return False
        
        if not file_path.suffix == '.py':
            self.logger.warning(f"Skipping non-Python file: {file_path}")
            return False
        
        module_name = file_path.stem
        
        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                self.logger.error(f"Could not create module spec for: {file_path}")
                return False
            
            module = importlib.util.module_from_spec(spec)
            
            # Store module to prevent garbage collection
            self.loaded_modules[module_name] = module
            
            # Execute the module
            spec.loader.exec_module(module)
            
            # Find plugin classes in the module
            plugin_classes = self._find_plugin_classes(module)
            
            if not plugin_classes:
                self.logger.warning(f"No plugin classes found in: {file_path}")
                return False
            
            # Register found plugins
            success_count = 0
            for plugin_class in plugin_classes:
                if self._register_plugin_class(plugin_class, file_path):
                    success_count += 1
            
            if success_count > 0:
                self.logger.debug(f"Loaded {success_count} plugin(s) from: {file_path}")
                return True
            else:
                self.logger.warning(f"Failed to register any plugins from: {file_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading plugin from {file_path}: {e}")
            return False
    
    def _find_plugin_classes(self, module) -> List[Type[FunscriptTransformationPlugin]]:
        """Find all plugin classes in a module."""
        plugin_classes = []
        
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (obj is not FunscriptTransformationPlugin and 
                issubclass(obj, FunscriptTransformationPlugin) and
                not inspect.isabstract(obj)):
                plugin_classes.append(obj)
        
        return plugin_classes
    
    def _register_plugin_class(self, plugin_class: Type[FunscriptTransformationPlugin], 
                              file_path: Path) -> bool:
        """Register a plugin class with the registry."""
        try:
            # Create plugin instance
            plugin_instance = plugin_class()
            
            # Register with the global registry
            success = plugin_registry.register(plugin_instance)
            
            if success:
                self.logger.debug(f"Registered plugin '{plugin_instance.name}' from {file_path}")
            else:
                self.logger.warning(
                    f"Failed to register plugin '{plugin_instance.name}' from {file_path} "
                    "(dependencies may be missing)"
                )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error creating instance of {plugin_class.__name__}: {e}")
            return False
    
    def load_user_plugins(self, user_plugins_dir: Optional[str] = None) -> Dict[str, bool]:
        """
        Load user-defined plugins from a specified directory.
        
        Args:
            user_plugins_dir: Directory containing user plugins. If None, 
                            looks for 'user_plugins' directory relative to funscript module.
                            
        Returns:
            Dictionary mapping plugin file names to load success status
        """
        if user_plugins_dir is None:
            # Default to user_plugins directory in the funscript module
            funscript_dir = Path(__file__).parent.parent
            user_plugins_dir = funscript_dir / 'user_plugins'
        
        user_plugins_path = Path(user_plugins_dir)
        
        if not user_plugins_path.exists():
            self.logger.debug(f"User plugins directory does not exist: {user_plugins_path}")
            self.logger.debug("Create this directory and add your custom plugins there")
            return {}
        
        self.logger.debug(f"Loading user plugins from: {user_plugins_path}")
        return self.load_plugins_from_directory(str(user_plugins_path), recursive=True)
    
    def reload_plugin(self, plugin_name: str, file_path: str) -> bool:
        """
        Reload a specific plugin (useful for development).
        
        Args:
            plugin_name: Name of the plugin to reload
            file_path: Path to the plugin file
            
        Returns:
            True if reloaded successfully, False otherwise
        """
        # Unregister the existing plugin
        plugin_registry.unregister(plugin_name)
        
        # Remove from loaded modules if present
        module_name = Path(file_path).stem
        if module_name in self.loaded_modules:
            del self.loaded_modules[module_name]
        
        # Load the plugin again
        return self.load_plugin_from_file(file_path)
    
    def create_user_plugins_directory(self, base_path: Optional[str] = None) -> Path:
        """
        Create a user plugins directory with a template plugin file.
        
        Args:
            base_path: Base directory where to create user_plugins. If None,
                      uses the funscript module directory.
                      
        Returns:
            Path to the created user_plugins directory
        """
        if base_path is None:
            base_path = Path(__file__).parent.parent
        
        user_plugins_dir = Path(base_path) / 'user_plugins'
        user_plugins_dir.mkdir(exist_ok=True)
        
        # Create __init__.py
        init_file = user_plugins_dir / '__init__.py'
        if not init_file.exists():
            init_file.write_text('# User-defined funscript plugins\n')
        
        # Create template plugin
        template_file = user_plugins_dir / 'template_plugin.py'
        if not template_file.exists():
            template_content = self._get_template_plugin_content()
            template_file.write_text(template_content)
        
        self.logger.debug(f"Created user plugins directory: {user_plugins_dir}")
        return user_plugins_dir
    
    def _get_template_plugin_content(self) -> str:
        """Generate template plugin file content."""
        return '''"""
Template plugin for custom funscript transformations.

Copy this file and modify it to create your own custom transformation plugins.
"""

import numpy as np
from typing import Dict, Any, List, Optional

from funscript.plugins.base_plugin import FunscriptTransformationPlugin


class TemplatePlugin(FunscriptTransformationPlugin):
    """
    Template plugin for custom transformations.
    
    Replace this with your own transformation logic.
    """
    
    @property
    def name(self) -> str:
        return "template_plugin"
    
    @property
    def description(self) -> str:
        return "A template plugin for custom transformations"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            'example_parameter': {
                'type': float,
                'required': True,
                'description': 'An example parameter',
                'constraints': {'min': 0.0, 'max': 10.0}
            },
            'optional_parameter': {
                'type': str,
                'required': False,
                'default': 'default_value',
                'description': 'An optional parameter'
            }
        }
    
    def transform(self, funscript, axis: str = 'both', **parameters) -> Optional['DualAxisFunscript']:
        """Apply your custom transformation logic here."""
        # Validate parameters
        validated_params = self.validate_parameters(parameters)
        
        # Validate axis
        if axis not in self.supported_axes:
            raise ValueError(f"Unsupported axis '{axis}'. Must be one of {self.supported_axes}")
        
        # Determine which axes to process
        axes_to_process = []
        if axis == 'both':
            axes_to_process = ['primary', 'secondary']
        else:
            axes_to_process = [axis]
        
        for current_axis in axes_to_process:
            self._apply_transformation_to_axis(funscript, current_axis, validated_params)
        
        return None  # Modifies in-place
    
    def _apply_transformation_to_axis(self, funscript, axis: str, params: Dict[str, Any]):
        """Apply transformation to a single axis."""
        actions_list = funscript.primary_actions if axis == 'primary' else funscript.secondary_actions
        
        if not actions_list:
            self.logger.warning(f"No actions found for {axis} axis")
            return
        
        # TODO: Implement your transformation logic here
        # Example: multiply all positions by a factor
        example_parameter = params['example_parameter']
        
        for action in actions_list:
            # Modify action positions based on your logic
            new_pos = action['pos'] * example_parameter / 10.0
            action['pos'] = int(np.clip(new_pos, 0, 100))
        
        # Invalidate cache after modification
        funscript._invalidate_cache(axis)
        
        self.logger.debug(f"Applied template transformation to {axis} axis with parameter {example_parameter}")
    
    def get_preview(self, funscript, axis: str = 'both', **parameters) -> Dict[str, Any]:
        """Generate a preview of the transformation effect."""
        try:
            validated_params = self.validate_parameters(parameters)
        except ValueError as e:
            return {"error": str(e)}
        
        return {
            "filter_type": "Template Plugin",
            "parameters": validated_params,
            "description": "This is a template plugin preview"
        }
'''


# Global plugin loader instance
plugin_loader = PluginLoader()