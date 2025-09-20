"""
Position inversion plugin for funscript transformations.

This plugin inverts position values (0 becomes 100, 100 becomes 0, etc.)
while preserving timing information.
"""

import numpy as np
from typing import Dict, Any, List, Optional

try:
    from .base_plugin import FunscriptTransformationPlugin
except ImportError:
    from funscript.plugins.base_plugin import FunscriptTransformationPlugin


class InvertPlugin(FunscriptTransformationPlugin):
    """
    Position inversion plugin.
    
    Inverts position values so that high positions become low and vice versa.
    Useful for creating opposite movement patterns.
    """
    
    @property
    def name(self) -> str:
        return "Invert"
    
    @property
    def description(self) -> str:
        return "Inverts position values (0↔100, preserving timing)"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def ui_preference(self) -> str:
        """Invert is a simple one-click operation."""
        return 'direct'
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            'start_time_ms': {
                'type': int,
                'required': False,
                'default': None,
                'description': 'Start time for inversion range (None for full range)'
            },
            'end_time_ms': {
                'type': int,
                'required': False,
                'default': None,
                'description': 'End time for inversion range (None for full range)'
            },
            'selected_indices': {
                'type': list,
                'required': False,
                'default': None,
                'description': 'Specific action indices to invert (overrides time range)'
            }
        }
    
    def transform(self, funscript, axis: str = 'both', **parameters) -> Optional['DualAxisFunscript']:
        """Apply inversion to the specified axis."""
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
            self._apply_inversion_to_axis(funscript, current_axis, validated_params)
        
        return None  # Modifies in-place
    
    def _apply_inversion_to_axis(self, funscript, axis: str, params: Dict[str, Any]):
        """Apply inversion to a single axis."""
        actions_list = funscript.primary_actions if axis == 'primary' else funscript.secondary_actions
        
        if not actions_list:
            self.logger.warning(f"No actions found for {axis} axis")
            return
        
        # Determine which indices to invert
        indices_to_invert = self._get_indices_to_invert(actions_list, params)
        
        if not indices_to_invert:
            self.logger.debug(f"No points to invert for {axis} axis")
            return
        
        # OPTIMIZED: Apply inversion using vectorized operations for large datasets
        if len(indices_to_invert) > 1000:
            # Vectorized inversion for large datasets
            indices_array = np.array(indices_to_invert)
            positions = np.array([actions_list[i]['pos'] for i in indices_to_invert])
            inverted_positions = 100 - positions
            
            # Bulk update
            for idx, new_pos in zip(indices_array, inverted_positions):
                actions_list[idx]['pos'] = int(new_pos)
        else:
            # Original method for smaller datasets
            for idx in indices_to_invert:
                original_pos = actions_list[idx]['pos']
                inverted_pos = 100 - original_pos
                actions_list[idx]['pos'] = inverted_pos
        
        # Invalidate cache
        funscript._invalidate_cache(axis)
        
        self.logger.info(
            f"Applied inversion to {axis} axis: "
            f"{len(indices_to_invert)} points inverted"
        )
    
    def _get_indices_to_invert(self, actions_list: List[Dict], params: Dict[str, Any]) -> List[int]:
        """Determine which action indices should be inverted."""
        selected_indices = params.get('selected_indices')
        start_time_ms = params.get('start_time_ms')
        end_time_ms = params.get('end_time_ms')
        
        if selected_indices is not None and len(selected_indices) > 0:
            # Use selected indices
            indices_to_invert = sorted([
                i for i in selected_indices 
                if 0 <= i < len(actions_list)
            ])
            return indices_to_invert
        
        elif start_time_ms is not None and end_time_ms is not None:
            # OPTIMIZED: Use vectorized time range filtering for large datasets
            if len(actions_list) > 10000:
                timestamps = np.array([action['at'] for action in actions_list])
                time_mask = (timestamps >= start_time_ms) & (timestamps <= end_time_ms)
                indices_to_invert = np.where(time_mask)[0].tolist()
            else:
                indices_to_invert = []
                for i, action in enumerate(actions_list):
                    if start_time_ms <= action['at'] <= end_time_ms:
                        indices_to_invert.append(i)
            return indices_to_invert
        
        else:
            # Invert entire list
            return list(range(len(actions_list)))
    
    def get_preview(self, funscript, axis: str = 'both', **parameters) -> Dict[str, Any]:
        """Generate a preview of the inversion effect."""
        try:
            validated_params = self.validate_parameters(parameters)
        except ValueError as e:
            return {"error": str(e)}
        
        preview_info = {
            "filter_type": "Position Inversion",
            "parameters": validated_params,
            "transformation": "position = 100 - position",
            "description": "Inverts all positions: high becomes low, low becomes high"
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
            
            indices_to_invert = self._get_indices_to_invert(actions_list, validated_params)
            
            if indices_to_invert:
                # Calculate preview statistics
                positions = np.array([actions_list[i]['pos'] for i in indices_to_invert])
                inverted_positions = 100 - positions
                
                # Calculate movement characteristics
                original_mean = np.mean(positions)
                inverted_mean = np.mean(inverted_positions)
                
                # Find extreme values
                original_min, original_max = np.min(positions), np.max(positions)
                inverted_min, inverted_max = np.min(inverted_positions), np.max(inverted_positions)
                
                axis_info = {
                    "total_points": len(actions_list),
                    "points_to_invert": len(indices_to_invert),
                    "original_mean_position": round(original_mean, 1),
                    "inverted_mean_position": round(inverted_mean, 1),
                    "original_range": f"{original_min}-{original_max}",
                    "inverted_range": f"{inverted_min}-{inverted_max}",
                    "effect": "High positions become low, low positions become high"
                }
                
                preview_info[f"{current_axis}_axis"] = axis_info
            else:
                preview_info[f"{current_axis}_axis"] = {
                    "total_points": len(actions_list),
                    "points_to_invert": 0,
                    "can_apply": False
                }
        
        return preview_info