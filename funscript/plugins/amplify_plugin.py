"""
Amplitude amplification plugin for funscript transformations.

This plugin amplifies position values around a center point, making movements
more or less extreme while preserving the overall pattern.
"""

import numpy as np
from typing import Dict, Any, List, Optional

try:
    from .base_plugin import FunscriptTransformationPlugin
except ImportError:
    from funscript.plugins.base_plugin import FunscriptTransformationPlugin


class AmplifyPlugin(FunscriptTransformationPlugin):
    """
    Amplitude amplification plugin.
    
    Amplifies or reduces position values around a center point, making
    movements more or less extreme while preserving timing and patterns.
    """
    
    @property
    def name(self) -> str:
        return "Amplify"
    
    @property
    def description(self) -> str:
        return "Amplifies position values around a center point"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            'scale_factor': {
                'type': float,
                'required': False,
                'default': 1.25,
                'description': 'Amplification factor (>1 increases, <1 decreases amplitude)',
                'constraints': {'min': 0.1, 'max': 5.0}
            },
            'center_value': {
                'type': int,
                'required': False,
                'default': 50,
                'description': 'Center point for amplification (0-100)',
                'constraints': {'min': 0, 'max': 100}
            },
            'start_time_ms': {
                'type': int,
                'required': False,
                'default': None,
                'description': 'Start time for amplification range (None for full range)'
            },
            'end_time_ms': {
                'type': int,
                'required': False,
                'default': None,
                'description': 'End time for amplification range (None for full range)'
            },
            'selected_indices': {
                'type': list,
                'required': False,
                'default': None,
                'description': 'Specific action indices to amplify (overrides time range)'
            }
        }
    
    def transform(self, funscript, axis: str = 'both', **parameters) -> Optional['DualAxisFunscript']:
        """Apply amplification to the specified axis."""
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
            self._apply_amplification_to_axis(funscript, current_axis, validated_params)
        
        return None  # Modifies in-place
    
    def _apply_amplification_to_axis(self, funscript, axis: str, params: Dict[str, Any]):
        """Apply amplification to a single axis."""
        actions_list = funscript.primary_actions if axis == 'primary' else funscript.secondary_actions
        
        if not actions_list:
            self.logger.warning(f"No actions found for {axis} axis")
            return
        
        # Determine which indices to amplify
        indices_to_amplify = self._get_indices_to_amplify(actions_list, params)
        
        if not indices_to_amplify:
            self.logger.warning(f"No points to amplify for {axis} axis")
            return
        
        scale_factor = params['scale_factor']
        center_value = params['center_value']
        
        # OPTIMIZED: Use numpy array indexing for bulk operations
        if len(indices_to_amplify) > 1000:  # Use optimized path for large datasets
            # Convert indices to numpy array for vectorized indexing
            indices_array = np.array(indices_to_amplify)
            
            # Extract all positions at once using advanced indexing
            positions = np.array([actions_list[i]['pos'] for i in indices_to_amplify])
            
            # Vectorized amplification (same as before)
            amplified_positions = center_value + (positions - center_value) * scale_factor
            clamped_positions = np.clip(np.round(amplified_positions), 0, 100).astype(int)
            
            # OPTIMIZED: Bulk update using numpy operations
            changed_mask = clamped_positions != positions
            affected_count = np.sum(changed_mask)
            
            # Update all positions in batch - much faster for large datasets
            changed_indices = indices_array[changed_mask]
            changed_positions = clamped_positions[changed_mask]
            
            for idx, new_pos in zip(changed_indices, changed_positions):
                actions_list[idx]['pos'] = int(new_pos)
        else:
            # Original path for smaller datasets
            positions = np.array([actions_list[i]['pos'] for i in indices_to_amplify])
            amplified_positions = center_value + (positions - center_value) * scale_factor
            clamped_positions = np.clip(np.round(amplified_positions), 0, 100).astype(int)
            
            changed_mask = clamped_positions != positions
            affected_count = np.sum(changed_mask)
            
            for i, idx in enumerate(indices_to_amplify):
                if changed_mask[i]:
                    actions_list[idx]['pos'] = int(clamped_positions[i])
        
        # Invalidate cache
        funscript._invalidate_cache(axis)
        
        self.logger.info(
            f"Applied amplification to {axis} axis: "
            f"{affected_count}/{len(indices_to_amplify)} points modified "
            f"(scale={scale_factor:.2f}, center={center_value})"
        )
    
    def _get_indices_to_amplify(self, actions_list: List[Dict], params: Dict[str, Any]) -> List[int]:
        """Determine which action indices should be amplified."""
        selected_indices = params.get('selected_indices')
        start_time_ms = params.get('start_time_ms')
        end_time_ms = params.get('end_time_ms')
        
        if selected_indices is not None and len(selected_indices) > 0:
            # Use selected indices
            indices_to_amplify = sorted([
                i for i in selected_indices 
                if 0 <= i < len(actions_list)
            ])
            return indices_to_amplify
        
        elif start_time_ms is not None and end_time_ms is not None:
            # OPTIMIZED: Use vectorized time range filtering for large datasets
            if len(actions_list) > 10000:
                # Extract timestamps as numpy array
                timestamps = np.array([action['at'] for action in actions_list])
                # Use boolean indexing to find matching indices
                time_mask = (timestamps >= start_time_ms) & (timestamps <= end_time_ms)
                indices_to_amplify = np.where(time_mask)[0].tolist()
            else:
                # Original method for smaller datasets
                indices_to_amplify = []
                for i, action in enumerate(actions_list):
                    if start_time_ms <= action['at'] <= end_time_ms:
                        indices_to_amplify.append(i)
            return indices_to_amplify
        
        else:
            # Amplify entire list
            return list(range(len(actions_list)))
    
    def get_preview(self, funscript, axis: str = 'both', **parameters) -> Dict[str, Any]:
        """Generate a preview of the amplification effect."""
        try:
            validated_params = self.validate_parameters(parameters)
        except ValueError as e:
            return {"error": str(e)}
        
        preview_info = {
            "filter_type": "Amplification",
            "parameters": validated_params
        }
        
        scale_factor = validated_params['scale_factor']
        center_value = validated_params['center_value']
        
        # Determine which axes would be affected
        if axis == 'both':
            axes_to_check = ['primary', 'secondary']
        else:
            axes_to_check = [axis]
        
        for current_axis in axes_to_check:
            actions_list = funscript.primary_actions if current_axis == 'primary' else funscript.secondary_actions
            if not actions_list:
                continue
            
            indices_to_amplify = self._get_indices_to_amplify(actions_list, validated_params)
            
            if indices_to_amplify:
                # Calculate preview statistics
                positions = np.array([actions_list[i]['pos'] for i in indices_to_amplify])
                
                # Simulate amplification
                amplified_positions = center_value + (positions - center_value) * scale_factor
                clamped_positions = np.clip(amplified_positions, 0, 100)
                
                # Calculate statistics
                original_range = np.max(positions) - np.min(positions)
                new_range = np.max(clamped_positions) - np.min(clamped_positions)
                range_change_pct = ((new_range - original_range) / max(original_range, 1)) * 100
                
                # Count how many points would be clamped
                clamped_count = np.sum((amplified_positions < 0) | (amplified_positions > 100))
                
                axis_info = {
                    "total_points": len(actions_list),
                    "points_affected": len(indices_to_amplify),
                    "original_range": int(original_range),
                    "new_range": int(new_range),
                    "range_change_percent": round(range_change_pct, 1),
                    "points_clamped": int(clamped_count),
                    "scale_factor": scale_factor,
                    "center_value": center_value
                }
                
                if scale_factor > 1:
                    axis_info["effect"] = "Increases amplitude"
                elif scale_factor < 1:
                    axis_info["effect"] = "Decreases amplitude"
                else:
                    axis_info["effect"] = "No change"
                
                preview_info[f"{current_axis}_axis"] = axis_info
            else:
                preview_info[f"{current_axis}_axis"] = {
                    "total_points": len(actions_list),
                    "points_affected": 0,
                    "can_apply": False
                }
        
        return preview_info