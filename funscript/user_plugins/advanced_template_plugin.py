"""
Advanced template plugin for custom funscript transformations.

This is a comprehensive example showing different types of transformations,
parameter handling, and preview generation. Copy and modify for your own plugins.

QUICK START:
1. Copy this file with a new name (e.g., my_custom_plugin.py)
2. Change the class name and plugin name
3. Modify the parameters_schema for your needs
4. Implement your transformation logic in _apply_transformation_to_axis
5. Update the get_preview method to provide useful feedback
6. Save the file - it will automatically appear in the timeline UI!
"""

import numpy as np
import math
from typing import Dict, Any, List, Optional

from funscript.plugins.base_plugin import FunscriptTransformationPlugin


class AdvancedTemplatePlugin(FunscriptTransformationPlugin):
    """
    Advanced template plugin demonstrating various transformation techniques.
    
    This plugin shows examples of:
    - Multiple parameter types and validation
    - Selection-based vs full-script processing  
    - Comprehensive preview generation
    - Different transformation approaches
    """
    
    @property
    def name(self) -> str:
        return "advanced_template"
    
    @property
    def description(self) -> str:
        return "Advanced template showing various transformation techniques"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            # Different parameter types with validation
            'transformation_type': {
                'type': str,
                'required': False,
                'default': 'sine_wave',
                'description': 'Type of transformation to apply',
                'constraints': {'choices': ['sine_wave', 'linear_scale', 'position_shift', 'invert_smooth']}
            },
            'intensity': {
                'type': float,
                'required': False,
                'default': 1.0,
                'description': 'Intensity of the transformation (0.1-3.0)',
                'constraints': {'min': 0.1, 'max': 3.0}
            },
            'preserve_extremes': {
                'type': bool,
                'required': False,
                'default': True,
                'description': 'Preserve original peak and valley positions'
            },
            'frequency_hz': {
                'type': float,
                'required': False,
                'default': 0.5,
                'description': 'Frequency for wave-based transformations (Hz)',
                'constraints': {'min': 0.1, 'max': 2.0}
            },
            # Selection support
            'selected_indices': {
                'type': list,
                'required': False,
                'default': None,
                'description': 'Specific action indices to transform (overrides full script)'
            }
        }
    
    def transform(self, funscript, axis: str = 'both', **parameters) -> Optional['DualAxisFunscript']:
        """Apply the transformation based on selected type and parameters."""
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
        """Apply transformation to a single axis using the selected method."""
        actions_list = funscript.primary_actions if axis == 'primary' else funscript.secondary_actions
        
        if not actions_list:
            self.logger.warning(f"No actions found for {axis} axis")
            return
        
        # Determine which indices to process
        indices_to_process = self._get_indices_to_process(actions_list, params)
        
        if not indices_to_process:
            self.logger.warning(f"No points to process for {axis} axis")
            return
        
        transformation_type = params['transformation_type']
        intensity = params['intensity']
        
        # Apply the selected transformation
        if transformation_type == 'sine_wave':
            self._apply_sine_wave_modulation(actions_list, indices_to_process, params)
        elif transformation_type == 'linear_scale':
            self._apply_linear_scaling(actions_list, indices_to_process, params)
        elif transformation_type == 'position_shift':
            self._apply_position_shift(actions_list, indices_to_process, params)
        elif transformation_type == 'invert_smooth':
            self._apply_smooth_inversion(actions_list, indices_to_process, params)
        
        # Invalidate cache after modification
        funscript._invalidate_cache(axis)
        
        self.logger.info(
            f"Applied {transformation_type} transformation to {axis} axis: "
            f"{len(indices_to_process)} points affected (intensity={intensity})"
        )
    
    def _get_indices_to_process(self, actions_list: List[Dict], params: Dict[str, Any]) -> List[int]:
        """Determine which action indices should be processed."""
        selected_indices = params.get('selected_indices')
        
        if selected_indices is not None and len(selected_indices) > 0:
            # Use selected indices
            return sorted([
                i for i in selected_indices 
                if 0 <= i < len(actions_list)
            ])
        else:
            # Process entire list
            return list(range(len(actions_list)))
    
    def _apply_sine_wave_modulation(self, actions_list: List[Dict], indices: List[int], params: Dict[str, Any]):
        """Apply sine wave modulation to positions."""
        intensity = params['intensity']
        frequency_hz = params['frequency_hz']
        preserve_extremes = params['preserve_extremes']
        
        if not indices:
            return
        
        # Calculate time span for frequency
        start_time = actions_list[indices[0]]['at']
        end_time = actions_list[indices[-1]]['at']
        time_span_sec = (end_time - start_time) / 1000.0
        
        if time_span_sec <= 0:
            return
        
        # Find extremes if preservation is enabled
        original_positions = np.array([actions_list[i]['pos'] for i in indices])
        min_pos = np.min(original_positions)
        max_pos = np.max(original_positions)
        
        for idx_pos, list_idx in enumerate(indices):
            action = actions_list[list_idx]
            original_pos = action['pos']
            
            # Calculate sine wave offset
            time_offset_sec = (action['at'] - start_time) / 1000.0
            wave_phase = 2 * math.pi * frequency_hz * time_offset_sec
            wave_offset = intensity * 10 * math.sin(wave_phase)
            
            # Apply modulation
            new_pos = original_pos + wave_offset
            
            # Preserve extremes if enabled
            if preserve_extremes:
                if original_pos == min_pos or original_pos == max_pos:
                    continue  # Don't modify extremes
            
            # Clamp to valid range
            action['pos'] = int(round(np.clip(new_pos, 0, 100)))
    
    def _apply_linear_scaling(self, actions_list: List[Dict], indices: List[int], params: Dict[str, Any]):
        """Apply linear scaling around center point."""
        intensity = params['intensity']
        
        if not indices:
            return
        
        # Calculate center point from selection
        positions = np.array([actions_list[i]['pos'] for i in indices])
        center_pos = np.mean(positions)
        
        for list_idx in indices:
            action = actions_list[list_idx]
            original_pos = action['pos']
            
            # Scale around center
            new_pos = center_pos + (original_pos - center_pos) * intensity
            action['pos'] = int(round(np.clip(new_pos, 0, 100)))
    
    def _apply_position_shift(self, actions_list: List[Dict], indices: List[int], params: Dict[str, Any]):
        """Apply position shifting based on intensity."""
        intensity = params['intensity']
        
        # Convert intensity to shift amount (-20 to +20)
        shift_amount = (intensity - 1.0) * 20
        
        for list_idx in indices:
            action = actions_list[list_idx]
            new_pos = action['pos'] + shift_amount
            action['pos'] = int(round(np.clip(new_pos, 0, 100)))
    
    def _apply_smooth_inversion(self, actions_list: List[Dict], indices: List[int], params: Dict[str, Any]):
        """Apply smooth inversion with intensity control."""
        intensity = params['intensity']
        
        for list_idx in indices:
            action = actions_list[list_idx]
            original_pos = action['pos']
            
            # Partial inversion based on intensity
            inverted_pos = 100 - original_pos
            new_pos = original_pos + intensity * (inverted_pos - original_pos)
            action['pos'] = int(round(np.clip(new_pos, 0, 100)))
    
    def get_preview(self, funscript, axis: str = 'both', **parameters) -> Dict[str, Any]:
        """Generate a comprehensive preview of the transformation effect."""
        try:
            validated_params = self.validate_parameters(parameters)
        except ValueError as e:
            return {"error": str(e)}
        
        preview_info = {
            "filter_type": "Advanced Template",
            "parameters": validated_params,
            "transformation_type": validated_params['transformation_type']
        }
        
        # Determine which axes would be affected
        if axis == 'both':
            axes_to_check = ['primary', 'secondary']
        else:
            axes_to_check = [axis]
        
        for current_axis in axes_to_check:
            actions_list = funscript.primary_actions if current_axis == 'primary' else funscript.secondary_actions
            if not actions_list:
                continue
            
            indices_to_process = self._get_indices_to_process(actions_list, validated_params)
            
            if indices_to_process:
                # Calculate preview statistics
                positions = np.array([actions_list[i]['pos'] for i in indices_to_process])
                
                axis_info = {
                    "total_points": len(actions_list),
                    "points_affected": len(indices_to_process),
                    "original_range": f"{np.min(positions)}-{np.max(positions)}",
                    "original_mean": round(np.mean(positions), 1),
                    "transformation_type": validated_params['transformation_type'],
                    "intensity": validated_params['intensity']
                }
                
                # Add transformation-specific preview info
                transformation_type = validated_params['transformation_type']
                if transformation_type == 'sine_wave':
                    axis_info["frequency_hz"] = validated_params['frequency_hz']
                    axis_info["effect"] = f"Sine wave modulation at {validated_params['frequency_hz']}Hz"
                elif transformation_type == 'linear_scale':
                    if validated_params['intensity'] > 1:
                        axis_info["effect"] = f"Amplify range by {validated_params['intensity']:.1f}x"
                    else:
                        axis_info["effect"] = f"Reduce range by {validated_params['intensity']:.1f}x"
                elif transformation_type == 'position_shift':
                    shift = (validated_params['intensity'] - 1.0) * 20
                    axis_info["effect"] = f"Shift positions by {shift:+.1f}"
                elif transformation_type == 'invert_smooth':
                    axis_info["effect"] = f"Partial inversion at {validated_params['intensity']*100:.0f}%"
                
                preview_info[f"{current_axis}_axis"] = axis_info
            else:
                preview_info[f"{current_axis}_axis"] = {
                    "total_points": len(actions_list),
                    "points_affected": 0,
                    "can_apply": False
                }
        
        return preview_info


# Additional example: Simple numeric transformation plugin
class SimpleScalePlugin(FunscriptTransformationPlugin):
    """Simple plugin that scales all positions by a factor."""
    
    @property
    def name(self) -> str:
        return "simple_scale"
    
    @property
    def description(self) -> str:
        return "Scale all positions by a factor around center point"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            'scale_factor': {
                'type': float,
                'required': False,
                'default': 1.5,
                'description': 'Scale factor for positions',
                'constraints': {'min': 0.1, 'max': 3.0}
            },
            'center_point': {
                'type': int,
                'required': False,
                'default': 50,
                'description': 'Center point for scaling (0-100)',
                'constraints': {'min': 0, 'max': 100}
            }
        }
    
    def transform(self, funscript, axis: str = 'both', **parameters) -> Optional['DualAxisFunscript']:
        """Scale positions around center point."""
        validated_params = self.validate_parameters(parameters)
        
        axes_to_process = ['primary', 'secondary'] if axis == 'both' else [axis]
        
        for current_axis in axes_to_process:
            actions_list = funscript.primary_actions if current_axis == 'primary' else funscript.secondary_actions
            
            if not actions_list:
                continue
            
            scale_factor = validated_params['scale_factor']
            center_point = validated_params['center_point']
            
            for action in actions_list:
                original_pos = action['pos']
                new_pos = center_point + (original_pos - center_point) * scale_factor
                action['pos'] = int(round(np.clip(new_pos, 0, 100)))
            
            funscript._invalidate_cache(current_axis)
        
        return None
    
    def get_preview(self, funscript, axis: str = 'both', **parameters) -> Dict[str, Any]:
        """Simple preview for scaling effect."""
        try:
            validated_params = self.validate_parameters(parameters)
        except ValueError as e:
            return {"error": str(e)}
        
        scale_factor = validated_params['scale_factor']
        effect = "Amplify" if scale_factor > 1.0 else "Reduce"
        
        return {
            "filter_type": "Simple Scale",
            "scale_factor": scale_factor,
            "center_point": validated_params['center_point'],
            "effect": f"{effect} movement range by {abs(scale_factor - 1.0)*100:.0f}%"
        }