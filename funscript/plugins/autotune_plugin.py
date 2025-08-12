"""
Auto-tune Savitzky-Golay filter plugin for funscript transformations.

This plugin automatically finds optimal SG filter parameters by minimizing 
saturation (points at extremes) while preserving the overall signal shape.
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


class AutotunePlugin(FunscriptTransformationPlugin):
    """
    Auto-tune Savitzky-Golay filter plugin.
    
    Automatically finds the optimal window size for SG filtering by minimizing
    saturation (points at min/max extremes) while preserving signal characteristics.
    """
    
    @property
    def name(self) -> str:
        return "Autotune SG"
    
    @property
    def description(self) -> str:
        return "Automatically finds optimal Savitzky-Golay filter parameters"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            'saturation_low': {
                'type': int,
                'required': False,
                'default': 1,
                'description': 'Position value considered low saturation',
                'constraints': {'min': 0, 'max': 50}
            },
            'saturation_high': {
                'type': int,
                'required': False,
                'default': 99,
                'description': 'Position value considered high saturation',
                'constraints': {'min': 50, 'max': 100}
            },
            'max_window_size': {
                'type': int,
                'required': False,
                'default': 15,
                'description': 'Maximum window size to test',
                'constraints': {'min': 3, 'max': 51}
            },
            'polyorder': {
                'type': int,
                'required': False,
                'default': 2,
                'description': 'Polynomial order for SG filter',
                'constraints': {'min': 0, 'max': 5}
            },
            'selected_indices': {
                'type': list,
                'required': False,
                'default': None,
                'description': 'Specific action indices to tune (None for full range)'
            }
        }
    
    @property
    def requires_scipy(self) -> bool:
        return True
    
    def transform(self, funscript, axis: str = 'both', **parameters) -> Optional['DualAxisFunscript']:
        """Apply auto-tuned SG filter to the specified axis."""
        if not SCIPY_AVAILABLE:
            raise RuntimeError("scipy is required for auto-tune but is not available")
        
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
        
        results = {}
        for current_axis in axes_to_process:
            result = self._apply_autotune_to_axis(funscript, current_axis, validated_params)
            results[current_axis] = result
        
        # Log combined results
        successful_axes = [ax for ax, result in results.items() if result is not None]
        self.logger.info(f"Auto-tune applied successfully to {len(successful_axes)}/{len(axes_to_process)} axes")
        
        return None  # Modifies in-place
    
    def _apply_autotune_to_axis(self, funscript, axis: str, params: Dict[str, Any]) -> Optional[Dict]:
        """Apply auto-tune to a single axis."""
        actions_list = funscript.primary_actions if axis == 'primary' else funscript.secondary_actions
        
        if not actions_list:
            self.logger.warning(f"No actions found for {axis} axis")
            return None
        
        # Determine indices to process
        indices_to_filter = self._get_indices_to_filter(actions_list, params)
        
        if len(indices_to_filter) < 3:
            self.logger.warning(f"Not enough points for auto-tune on {axis} axis")
            return None
        
        # Extract positions for analysis
        positions = np.array([actions_list[i]['pos'] for i in indices_to_filter])
        num_points = len(positions)
        
        # Find optimal window size
        best_params = self._find_optimal_window_size(positions, params)
        
        if best_params is None:
            self.logger.warning(f"Could not find optimal parameters for {axis} axis")
            return None
        
        # Apply the optimal SG filter
        window_length = best_params['window_length']
        polyorder = params['polyorder']
        
        try:
            # Ensure window_length is odd
            if window_length % 2 == 0:
                window_length += 1
            
            # Ensure polyorder is valid for window_length
            if polyorder >= window_length:
                polyorder = window_length - 1
            
            if polyorder < 0:
                polyorder = 0
            
            # Apply SG filter
            smoothed_positions = savgol_filter(positions, window_length, polyorder)
            
            # Update the actions with smoothed positions
            for i, list_idx in enumerate(indices_to_filter):
                actions_list[list_idx]['pos'] = int(round(np.clip(smoothed_positions[i], 0, 100)))
            
            # Invalidate cache
            funscript._invalidate_cache(axis)
            
            self.logger.info(
                f"Auto-tuned SG filter applied to {axis} axis: "
                f"window={window_length}, poly={polyorder}, "
                f"saturation_improvement={best_params.get('saturation_improvement', 0):.1f}%"
            )
            
            return best_params
            
        except Exception as e:
            self.logger.error(f"Error applying auto-tuned SG filter to {axis} axis: {e}")
            raise
    
    def _get_indices_to_filter(self, actions_list: List[Dict], params: Dict[str, Any]) -> List[int]:
        """Determine which action indices should be processed."""
        selected_indices = params.get('selected_indices')
        
        if selected_indices is not None and len(selected_indices) > 0:
            # Filter valid indices from the selection
            indices_to_filter = sorted([
                i for i in selected_indices 
                if 0 <= i < len(actions_list)
            ])
            return indices_to_filter
        else:
            # Process entire list
            return list(range(len(actions_list)))
    
    def _find_optimal_window_size(self, positions: np.ndarray, params: Dict[str, Any]) -> Optional[Dict]:
        """Find the optimal window size by minimizing saturation."""
        saturation_low = params['saturation_low']
        saturation_high = params['saturation_high']
        max_window_size = params['max_window_size']
        polyorder = params['polyorder']
        
        num_points = len(positions)
        best_window_length = -1
        best_saturation_count = float('inf')
        best_result = None
        
        # Test window sizes from 3 up to max_window_size
        for window_length in range(3, min(max_window_size + 1, num_points), 2):  # Only odd sizes
            if polyorder >= window_length:
                continue  # Skip invalid combinations
            
            try:
                # Apply SG filter with this window size
                smoothed_positions = savgol_filter(positions, window_length, polyorder)
                
                # Count saturation points
                low_saturated = np.sum(smoothed_positions <= saturation_low)
                high_saturated = np.sum(smoothed_positions >= saturation_high)
                total_saturated = low_saturated + high_saturated
                
                # Track the best result
                if total_saturated < best_saturation_count:
                    best_saturation_count = total_saturated
                    best_window_length = window_length
                    
                    # Calculate improvement metrics
                    original_low_saturated = np.sum(positions <= saturation_low)
                    original_high_saturated = np.sum(positions >= saturation_high)
                    original_total_saturated = original_low_saturated + original_high_saturated
                    
                    if original_total_saturated > 0:
                        saturation_improvement = ((original_total_saturated - total_saturated) / 
                                                original_total_saturated) * 100
                    else:
                        saturation_improvement = 0
                    
                    best_result = {
                        'window_length': window_length,
                        'polyorder': polyorder,
                        'saturation_count': total_saturated,
                        'saturation_improvement': saturation_improvement,
                        'low_saturated': low_saturated,
                        'high_saturated': high_saturated
                    }
                
            except Exception as e:
                # Skip this window size if it fails
                self.logger.debug(f"Window size {window_length} failed: {e}")
                continue
        
        if best_window_length == -1:
            return None
        
        return best_result
    
    def get_preview(self, funscript, axis: str = 'both', **parameters) -> Dict[str, Any]:
        """Generate a preview of the auto-tune effect."""
        if not SCIPY_AVAILABLE:
            return {"error": "scipy not available"}
        
        try:
            validated_params = self.validate_parameters(parameters)
        except ValueError as e:
            return {"error": str(e)}
        
        preview_info = {
            "filter_type": "SG Auto-tune",
            "parameters": validated_params
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
            
            if len(indices_to_filter) >= 3:
                positions = np.array([actions_list[i]['pos'] for i in indices_to_filter])
                
                # Find optimal parameters (preview only)
                best_params = self._find_optimal_window_size(positions, validated_params)
                
                axis_info = {
                    "total_points": len(actions_list),
                    "points_to_analyze": len(indices_to_filter),
                    "can_apply": best_params is not None
                }
                
                if best_params:
                    axis_info.update({
                        "optimal_window_length": best_params['window_length'],
                        "estimated_saturation_improvement": best_params['saturation_improvement'],
                        "current_saturation_count": best_params['saturation_count'] + 
                                                  int(best_params['saturation_improvement'] / 100 * 
                                                      len(positions))
                    })
                
                preview_info[f"{current_axis}_axis"] = axis_info
            else:
                preview_info[f"{current_axis}_axis"] = {
                    "total_points": len(actions_list),
                    "points_to_analyze": len(indices_to_filter), 
                    "can_apply": False,
                    "error": "Not enough points"
                }
        
        return preview_info