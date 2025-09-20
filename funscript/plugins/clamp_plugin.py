"""
Position clamping plugins for funscript transformations.

These plugins provide various clamping operations to constrain position
values to specific ranges or values.
"""

import numpy as np
from typing import Dict, Any, List, Optional

try:
    from .base_plugin import FunscriptTransformationPlugin
except ImportError:
    from funscript.plugins.base_plugin import FunscriptTransformationPlugin


class ThresholdClampPlugin(FunscriptTransformationPlugin):
    """
    Threshold-based clamping plugin.
    
    Clamps positions to extremes based on thresholds:
    - Values below lower threshold become 0
    - Values above upper threshold become 100
    """
    
    @property
    def name(self) -> str:
        return "Threshold Clamp"
    
    @property
    def description(self) -> str:
        return "Clamps positions to 0/100 based on thresholds"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            'lower_threshold': {
                'type': int,
                'required': False,
                'default': 20,
                'description': 'Positions below this become 0',
                'constraints': {'min': 0, 'max': 100}
            },
            'upper_threshold': {
                'type': int,
                'required': False,
                'default': 80,
                'description': 'Positions above this become 100',
                'constraints': {'min': 0, 'max': 100}
            },
            'start_time_ms': {
                'type': int,
                'required': False,
                'default': None,
                'description': 'Start time for clamping range (None for full range)'
            },
            'end_time_ms': {
                'type': int,
                'required': False,
                'default': None,
                'description': 'End time for clamping range (None for full range)'
            },
            'selected_indices': {
                'type': list,
                'required': False,
                'default': None,
                'description': 'Specific action indices to clamp (overrides time range)'
            }
        }
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters with additional threshold logic."""
        validated = super().validate_parameters(parameters)
        
        # Ensure lower_threshold <= upper_threshold
        if validated['lower_threshold'] >= validated['upper_threshold']:
            raise ValueError("lower_threshold must be less than upper_threshold")
        
        return validated
    
    def transform(self, funscript, axis: str = 'both', **parameters) -> Optional['DualAxisFunscript']:
        """Apply threshold clamping to the specified axis."""
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
            self._apply_threshold_clamp_to_axis(funscript, current_axis, validated_params)
        
        return None  # Modifies in-place
    
    def _apply_threshold_clamp_to_axis(self, funscript, axis: str, params: Dict[str, Any]):
        """Apply threshold clamping to a single axis."""
        actions_list = funscript.primary_actions if axis == 'primary' else funscript.secondary_actions
        
        if not actions_list:
            self.logger.warning(f"No actions found for {axis} axis")
            return
        
        # Determine which indices to clamp
        indices_to_clamp = self._get_indices_to_process(actions_list, params)
        
        if not indices_to_clamp:
            self.logger.debug(f"No points to clamp for {axis} axis")
            return
        
        lower_thresh = params['lower_threshold']
        upper_thresh = params['upper_threshold']
        
        # OPTIMIZED: Use bulk operations for large datasets
        if len(indices_to_clamp) > 1000:
            # Convert to numpy array for vectorized operations
            indices_array = np.array(indices_to_clamp)
            positions = np.array([actions_list[i]['pos'] for i in indices_to_clamp])
            
            # Vectorized clamping using boolean indexing
            clamped_positions = positions.copy()
            clamped_positions[positions < lower_thresh] = 0
            clamped_positions[positions > upper_thresh] = 100
            
            # Find only changed positions for bulk update
            changed_mask = clamped_positions != positions
            count_changed = np.sum(changed_mask)
            
            # Bulk update using advanced indexing
            if count_changed > 0:
                changed_indices = indices_array[changed_mask]
                changed_values = clamped_positions[changed_mask]
                
                for idx, new_pos in zip(changed_indices, changed_values):
                    actions_list[idx]['pos'] = int(new_pos)
        else:
            # Original path for smaller datasets
            positions = np.array([actions_list[i]['pos'] for i in indices_to_clamp])
            
            clamped_positions = positions.copy()
            clamped_positions[positions < lower_thresh] = 0
            clamped_positions[positions > upper_thresh] = 100
            
            changed_mask = clamped_positions != positions
            count_changed = np.sum(changed_mask)
            
            for i, idx in enumerate(indices_to_clamp):
                if changed_mask[i]:
                    actions_list[idx]['pos'] = int(clamped_positions[i])
        
        # Invalidate cache
        funscript._invalidate_cache(axis)
        
        self.logger.info(
            f"Applied threshold clamping to {axis} axis: "
            f"{count_changed}/{len(indices_to_clamp)} points clamped "
            f"(thresholds: {lower_thresh}-{upper_thresh})"
        )
    
    def _get_indices_to_process(self, actions_list: List[Dict], params: Dict[str, Any]) -> List[int]:
        """Determine which action indices should be processed."""
        selected_indices = params.get('selected_indices')
        start_time_ms = params.get('start_time_ms')
        end_time_ms = params.get('end_time_ms')
        
        if selected_indices is not None and len(selected_indices) > 0:
            # Use selected indices
            indices_to_process = sorted([
                i for i in selected_indices 
                if 0 <= i < len(actions_list)
            ])
            return indices_to_process
        
        elif start_time_ms is not None and end_time_ms is not None:
            # OPTIMIZED: Use vectorized time range filtering for large datasets
            if len(actions_list) > 10000:
                timestamps = np.array([action['at'] for action in actions_list])
                time_mask = (timestamps >= start_time_ms) & (timestamps <= end_time_ms)
                indices_to_process = np.where(time_mask)[0].tolist()
            else:
                indices_to_process = []
                for i, action in enumerate(actions_list):
                    if start_time_ms <= action['at'] <= end_time_ms:
                        indices_to_process.append(i)
            return indices_to_process
        
        else:
            # Process entire list
            return list(range(len(actions_list)))
    
    def get_preview(self, funscript, axis: str = 'both', **parameters) -> Dict[str, Any]:
        """Generate a preview of the threshold clamping effect."""
        try:
            validated_params = self.validate_parameters(parameters)
        except ValueError as e:
            return {"error": str(e)}
        
        preview_info = {
            "filter_type": "Threshold Clamping",
            "parameters": validated_params
        }
        
        lower_thresh = validated_params['lower_threshold']
        upper_thresh = validated_params['upper_threshold']
        
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
                
                # Count affected points
                below_lower = np.sum(positions < lower_thresh)
                above_upper = np.sum(positions > upper_thresh)
                total_affected = below_lower + above_upper
                
                axis_info = {
                    "total_points": len(actions_list),
                    "points_analyzed": len(indices_to_process),
                    "points_below_lower": int(below_lower),
                    "points_above_upper": int(above_upper),
                    "total_points_affected": int(total_affected),
                    "affect_percentage": round((total_affected / len(indices_to_process)) * 100, 1),
                    "lower_threshold": lower_thresh,
                    "upper_threshold": upper_thresh
                }
                
                preview_info[f"{current_axis}_axis"] = axis_info
            else:
                preview_info[f"{current_axis}_axis"] = {
                    "total_points": len(actions_list),
                    "points_analyzed": 0,
                    "can_apply": False
                }
        
        return preview_info


class ValueClampPlugin(FunscriptTransformationPlugin):
    """
    Value-based clamping plugin.
    
    Clamps all positions to a specific value.
    """
    
    @property
    def name(self) -> str:
        return "Clamp"
    
    @property
    def description(self) -> str:
        return "Clamps all positions to a specific value"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            'clamp_value': {
                'type': int,
                'required': False,
                'default': 50,
                'description': 'Value to clamp all positions to',
                'constraints': {'min': 0, 'max': 100}
            },
            'start_time_ms': {
                'type': int,
                'required': False,
                'default': None,
                'description': 'Start time for clamping range (None for full range)'
            },
            'end_time_ms': {
                'type': int,
                'required': False,
                'default': None,
                'description': 'End time for clamping range (None for full range)'
            },
            'selected_indices': {
                'type': list,
                'required': False,
                'default': None,
                'description': 'Specific action indices to clamp (overrides time range)'
            }
        }
    
    def transform(self, funscript, axis: str = 'both', **parameters) -> Optional['DualAxisFunscript']:
        """Apply value clamping to the specified axis."""
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
            self._apply_value_clamp_to_axis(funscript, current_axis, validated_params)
        
        return None  # Modifies in-place
    
    def _apply_value_clamp_to_axis(self, funscript, axis: str, params: Dict[str, Any]):
        """Apply value clamping to a single axis."""
        actions_list = funscript.primary_actions if axis == 'primary' else funscript.secondary_actions
        
        if not actions_list:
            self.logger.warning(f"No actions found for {axis} axis")
            return
        
        # Determine which indices to clamp
        indices_to_clamp = self._get_indices_to_process(actions_list, params)
        
        if not indices_to_clamp:
            self.logger.debug(f"No points to clamp for {axis} axis")
            return
        
        clamp_value = params['clamp_value']
        
        # Vectorized value clamping
        # Extract positions using vectorized indexing
        positions = np.array([actions_list[i]['pos'] for i in indices_to_clamp])
        
        # Batch update only changed positions
        changed_mask = positions != clamp_value
        count_changed = np.sum(changed_mask)
        
        # Update only changed positions
        for i, idx in enumerate(indices_to_clamp):
            if changed_mask[i]:
                actions_list[idx]['pos'] = clamp_value
        
        # Invalidate cache
        funscript._invalidate_cache(axis)
        
        self.logger.info(
            f"Applied value clamping to {axis} axis: "
            f"{count_changed}/{len(indices_to_clamp)} points set to {clamp_value}"
        )
    
    def _get_indices_to_process(self, actions_list: List[Dict], params: Dict[str, Any]) -> List[int]:
        """Determine which action indices should be processed."""
        selected_indices = params.get('selected_indices')
        start_time_ms = params.get('start_time_ms')
        end_time_ms = params.get('end_time_ms')
        
        if selected_indices is not None and len(selected_indices) > 0:
            # Use selected indices
            indices_to_process = sorted([
                i for i in selected_indices 
                if 0 <= i < len(actions_list)
            ])
            return indices_to_process
        
        elif start_time_ms is not None and end_time_ms is not None:
            # OPTIMIZED: Use vectorized time range filtering for large datasets
            if len(actions_list) > 10000:
                timestamps = np.array([action['at'] for action in actions_list])
                time_mask = (timestamps >= start_time_ms) & (timestamps <= end_time_ms)
                indices_to_process = np.where(time_mask)[0].tolist()
            else:
                indices_to_process = []
                for i, action in enumerate(actions_list):
                    if start_time_ms <= action['at'] <= end_time_ms:
                        indices_to_process.append(i)
            return indices_to_process
        
        else:
            # Process entire list
            return list(range(len(actions_list)))
    
    def get_preview(self, funscript, axis: str = 'both', **parameters) -> Dict[str, Any]:
        """Generate a preview of the value clamping effect."""
        try:
            validated_params = self.validate_parameters(parameters)
        except ValueError as e:
            return {"error": str(e)}
        
        preview_info = {
            "filter_type": "Value Clamping",
            "parameters": validated_params
        }
        
        clamp_value = validated_params['clamp_value']
        
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
                
                # Count points that would change
                points_to_change = np.sum(positions != clamp_value)
                
                axis_info = {
                    "total_points": len(actions_list),
                    "points_analyzed": len(indices_to_process),
                    "points_to_change": int(points_to_change),
                    "points_unchanged": len(indices_to_process) - int(points_to_change),
                    "change_percentage": round((points_to_change / len(indices_to_process)) * 100, 1),
                    "clamp_value": clamp_value
                }
                
                preview_info[f"{current_axis}_axis"] = axis_info
            else:
                preview_info[f"{current_axis}_axis"] = {
                    "total_points": len(actions_list),
                    "points_analyzed": 0,
                    "can_apply": False
                }
        
        return preview_info