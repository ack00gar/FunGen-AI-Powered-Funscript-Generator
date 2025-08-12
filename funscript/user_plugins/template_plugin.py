"""
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
        
        self.logger.info(f"Applied template transformation to {axis} axis with parameter {example_parameter}")
    
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
