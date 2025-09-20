"""
Savitzky-Golay smoothing filter plugin for funscript transformations.

This plugin applies Savitzky-Golay filtering to smooth funscript data while
preserving features like peaks and valleys better than simple moving averages.
"""

import numpy as np
from typing import Dict, Any, List, Optional
import copy

try:
    from .base_plugin import FunscriptTransformationPlugin
except ImportError:
    from funscript.plugins.base_plugin import FunscriptTransformationPlugin

try:
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class SavgolFilterPlugin(FunscriptTransformationPlugin):
    """
    Savitzky-Golay filter plugin for funscript smoothing.
    
    Applies a Savitzky-Golay filter to smooth position data while preserving
    important features like peaks and valleys. Requires scipy.
    """
    
    @property
    def name(self) -> str:
        return "Smooth (SG)"
    
    @property
    def description(self) -> str:
        return "Applies Savitzky-Golay smoothing filter to preserve features while reducing noise"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            'window_length': {
                'type': int,
                'required': False,
                'default': 7,
                'description': 'Length of the filter window (must be odd)',
                'constraints': {'min': 3}
            },
            'polyorder': {
                'type': int,
                'required': False,
                'default': 3,
                'description': 'Order of the polynomial used to fit the samples',
                'constraints': {'min': 0}
            },
            'start_time_ms': {
                'type': int,
                'required': False,
                'default': None,
                'description': 'Start time for filtering range (None for full range)'
            },
            'end_time_ms': {
                'type': int,
                'required': False,
                'default': None,
                'description': 'End time for filtering range (None for full range)'
            },
            'selected_indices': {
                'type': list,
                'required': False,
                'default': None,
                'description': 'Specific action indices to filter (overrides time range)'
            }
        }
    
    @property
    def requires_scipy(self) -> bool:
        return True
    
    def _get_action_indices_in_time_range(self, actions_list, start_time_ms, end_time_ms):
        """Helper method to find action indices within time range."""
        if not actions_list:
            return None, None
        
        start_idx = None
        end_idx = None
        
        for i, action in enumerate(actions_list):
            if start_idx is None and action['at'] >= start_time_ms:
                start_idx = i
            if action['at'] <= end_time_ms:
                end_idx = i
        
        return start_idx, end_idx
    
    def transform(self, funscript, axis: str = 'both', **parameters) -> Optional['DualAxisFunscript']:
        """Apply Savitzky-Golay filter to the specified axis."""
        if not SCIPY_AVAILABLE:
            raise RuntimeError("scipy is required for Savitzky-Golay filter but is not available")
        
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
        
        # Track if any axis was actually processed
        any_processed = False
        for current_axis in axes_to_process:
            was_processed = self._apply_savgol_to_axis(funscript, current_axis, validated_params)
            if was_processed:
                any_processed = True
        
        # Fail the transformation if no axis could be processed
        if not any_processed:
            raise ValueError("Insufficient data points for Savitzky-Golay filter on any axis")
        
        return None  # Modifies in-place
    
    def _apply_savgol_to_axis(self, funscript, axis: str, params: Dict[str, Any]) -> bool:
        """Apply Savitzky-Golay filter to a single axis. Returns True if processing was done."""
        actions_list = funscript.primary_actions if axis == 'primary' else funscript.secondary_actions
        
        if not actions_list:
            self.logger.debug(f"No actions found for {axis} axis")
            return False
        
        # Determine which indices to filter
        indices_to_filter = self._get_indices_to_filter(actions_list, params)
        
        if not indices_to_filter:
            self.logger.debug(f"No points to filter for {axis} axis")
            return False
        
        num_points = len(indices_to_filter)
        
        # Validate and adjust parameters
        window_length = params['window_length']
        polyorder = params['polyorder']
        
        # Ensure window_length is odd
        if window_length % 2 == 0:
            window_length += 1
        
        # Ensure polyorder is valid for window_length
        if polyorder >= window_length:
            polyorder = window_length - 1
        
        if polyorder < 0:
            polyorder = 0
        
        if num_points < window_length:
            self.logger.debug(
                f"Not enough points ({num_points}) for Savitzky-Golay filter "
                f"(window: {window_length}) on {axis} axis - skipping"
            )
            return False
        
        # OPTIMIZED: Use bulk operations for large datasets
        if len(indices_to_filter) > 1000:
            # Vectorized extraction and processing for large datasets
            indices_array = np.array(indices_to_filter)
            positions = np.array([actions_list[i]['pos'] for i in indices_to_filter])
            
            try:
                # Apply Savitzky-Golay filter
                smoothed_positions = savgol_filter(positions, window_length, polyorder)
                
                # Vectorized clipping and type conversion
                smoothed_positions_clipped = np.clip(np.round(smoothed_positions), 0, 100).astype(int)
                
                # Bulk update using zip for better performance
                for idx, new_pos in zip(indices_array, smoothed_positions_clipped):
                    actions_list[idx]['pos'] = int(new_pos)
            except Exception as e:
                self.logger.error(f"Error applying Savitzky-Golay filter to {axis} axis: {e}")
                raise
        else:
            # Original method for smaller datasets
            positions = np.array([actions_list[i]['pos'] for i in indices_to_filter])
            
            try:
                # Apply Savitzky-Golay filter
                smoothed_positions = savgol_filter(positions, window_length, polyorder)
                
                # Update positions
                smoothed_positions_clipped = np.clip(np.round(smoothed_positions), 0, 100).astype(int)
                for i, list_idx in enumerate(indices_to_filter):
                    actions_list[list_idx]['pos'] = int(smoothed_positions_clipped[i])
            except Exception as e:
                self.logger.error(f"Error applying Savitzky-Golay filter to {axis} axis: {e}")
                raise
            
        # Invalidate cache
        funscript._invalidate_cache(axis)
        
        self.logger.info(
            f"Applied Savitzky-Golay filter to {axis} axis, "
            f"affecting {len(indices_to_filter)} points "
            f"(window: {window_length}, poly: {polyorder})"
        )
        
        return True  # Successfully processed
    
    def _get_indices_to_filter(self, actions_list: List[Dict], params: Dict[str, Any]) -> List[int]:
        """Determine which action indices should be filtered."""
        selected_indices = params.get('selected_indices')
        start_time_ms = params.get('start_time_ms')
        end_time_ms = params.get('end_time_ms')
        
        if selected_indices is not None and len(selected_indices) > 0:
            # Filter valid indices from the selection
            indices_to_filter = sorted([
                i for i in selected_indices 
                if 0 <= i < len(actions_list)
            ])
            return indices_to_filter
        
        elif start_time_ms is not None and end_time_ms is not None:
            # OPTIMIZED: Use vectorized time range filtering for large datasets
            if len(actions_list) > 10000:
                timestamps = np.array([action['at'] for action in actions_list])
                time_mask = (timestamps >= start_time_ms) & (timestamps <= end_time_ms)
                return np.where(time_mask)[0].tolist()
            else:
                # Use original method for smaller datasets
                start_idx, end_idx = self._get_action_indices_in_time_range(
                    actions_list, start_time_ms, end_time_ms
                )
                
                if start_idx is None or end_idx is None or start_idx > end_idx:
                    return []
                
                return list(range(start_idx, end_idx + 1))
        
        else:
            # Filter entire list
            return list(range(len(actions_list)))
    
    def get_preview(self, funscript, axis: str = 'both', **parameters) -> Dict[str, Any]:
        """Generate a preview of the Savitzky-Golay filter effect."""
        if not SCIPY_AVAILABLE:
            return {"error": "scipy not available"}
        
        try:
            validated_params = self.validate_parameters(parameters)
        except ValueError as e:
            return {"error": str(e)}
        
        preview_info = {
            "filter_type": "Savitzky-Golay",
            "parameters": validated_params,
            "estimated_smoothing": "Medium to High"
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
                
            indices_to_filter = self._get_indices_to_filter(actions_list, validated_params)
            window_length = validated_params['window_length']
            
            # Adjust window_length if needed
            if window_length % 2 == 0:
                window_length += 1
            
            axis_info = {
                "points_affected": len(indices_to_filter),
                "total_points": len(actions_list),
                "effective_window_length": window_length,
                "can_apply": len(indices_to_filter) >= window_length
            }
            
            preview_info[f"{current_axis}_axis"] = axis_info
        
        return preview_info