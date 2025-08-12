"""
Ramer-Douglas-Peucker (RDP) simplification plugin for funscript transformations.

This plugin reduces the number of points in a funscript by removing redundant
points while preserving the overall shape using the RDP algorithm.
"""

import numpy as np
from typing import Dict, Any, List, Optional
import copy

try:
    from .base_plugin import FunscriptTransformationPlugin
except ImportError:
    from funscript.plugins.base_plugin import FunscriptTransformationPlugin

# Note: We use our own optimized numpy implementation instead of the slow rdp library
RDP_AVAILABLE = False  # Force use of fast numpy implementation


class RdpSimplifyPlugin(FunscriptTransformationPlugin):
    """
    RDP (Ramer-Douglas-Peucker) simplification plugin.
    
    Reduces funscript complexity by removing redundant points while preserving
    the overall shape and important features. Can use either the rdp library
    or a built-in numpy implementation.
    """
    
    @property
    def name(self) -> str:
        return "Simplify (RDP)"
    
    @property
    def description(self) -> str:
        return "Simplifies funscript by removing redundant points using RDP algorithm"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            'epsilon': {
                'type': float,
                'required': False,
                'default': 8.0,
                'description': 'Distance tolerance for point removal (higher = more aggressive)',
                'constraints': {'min': 0.1}
            },
            'start_time_ms': {
                'type': int,
                'required': False,
                'default': None,
                'description': 'Start time for simplification range (None for full range)'
            },
            'end_time_ms': {
                'type': int,
                'required': False,
                'default': None,
                'description': 'End time for simplification range (None for full range)'
            },
            'selected_indices': {
                'type': list,
                'required': False,
                'default': None,
                'description': 'Specific action indices to simplify (overrides time range)'
            },
        }
    
    @property
    def requires_rdp(self) -> bool:
        return False  # Uses fast numpy implementation, no external dependencies needed
    
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
    
    def _rdp_numpy_implementation(self, points, epsilon):
        """
        Fast numpy-based RDP implementation using vectorized operations.
        Much faster than the rdp library for typical funscript data.
        """
        if len(points) < 3:
            return points
        
        # Calculate perpendicular distances from all points to the line between first and last
        line_vec = points[-1] - points[0]
        line_length = np.linalg.norm(line_vec)
        
        if line_length == 0:
            return np.vstack((points[0], points[-1]))
        
        # Vectorized distance calculation for better performance
        if len(points) == 2:
            return points
        
        # Get all intermediate points (exclude first and last)
        intermediate_points = points[1:-1]
        
        # Calculate perpendicular distances vectorized
        point_vecs = intermediate_points - points[0]
        cross_products = np.cross(line_vec, point_vecs)
        distances = np.abs(cross_products) / line_length
        
        if len(distances) == 0:
            return np.vstack((points[0], points[-1]))
        
        max_index_in_distances = np.argmax(distances)
        max_distance = distances[max_index_in_distances]
        max_index = max_index_in_distances + 1  # +1 because we skipped first point
        
        if max_distance > epsilon:
            # Recursively simplify left and right segments
            left = self._rdp_numpy_implementation(points[:max_index + 1], epsilon)
            right = self._rdp_numpy_implementation(points[max_index:], epsilon)
            return np.vstack((left[:-1], right))
        else:
            return np.vstack((points[0], points[-1]))
    
    def transform(self, funscript, axis: str = 'both', **parameters) -> Optional['DualAxisFunscript']:
        """Apply RDP simplification to the specified axis."""
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
            self._apply_rdp_to_axis(funscript, current_axis, validated_params)
        
        return None  # Modifies in-place
    
    def _apply_rdp_to_axis(self, funscript, axis: str, params: Dict[str, Any]):
        """Apply RDP simplification to a single axis."""
        actions_list = funscript.primary_actions if axis == 'primary' else funscript.secondary_actions
        
        if not actions_list or len(actions_list) < 2:
            self.logger.debug(f"Not enough points on {axis} axis for RDP simplification")
            return False
        
        # Determine segment to simplify
        segment_info = self._get_segment_to_simplify(actions_list, params)
        
        if len(segment_info['segment']) < 2:
            self.logger.debug(f"Segment for RDP on {axis} axis has < 2 points")
            return False
        
        # Convert to points array for RDP
        points = np.array([
            [action['at'], action['pos']] 
            for action in segment_info['segment']
        ], dtype=np.float64)
        
        epsilon = params['epsilon']
        
        try:
            # Always use the fast numpy implementation
            simplified_points = self._rdp_numpy_implementation(points, epsilon)
            self.logger.debug(f"Using optimized numpy RDP implementation for {axis} axis simplification")
            
            # Convert back to action dictionaries
            simplified_actions = [
                {'at': int(point[0]), 'pos': int(np.clip(point[1], 0, 100))}
                for point in simplified_points
            ]
            
            # Reconstruct the full actions list
            new_actions_list = (
                segment_info['prefix'] + 
                simplified_actions + 
                segment_info['suffix']
            )
            
            # Update the funscript IN-PLACE to preserve list identity for undo manager
            actions_target_list = funscript.primary_actions if axis == 'primary' else funscript.secondary_actions
            actions_target_list[:] = new_actions_list
            
            # Invalidate cache
            funscript._invalidate_cache(axis)
            
            original_count = len(segment_info['segment'])
            simplified_count = len(simplified_actions)
            reduction_pct = ((original_count - simplified_count) / original_count) * 100
            
            self.logger.info(
                f"Applied RDP simplification to {axis} axis: "
                f"{original_count} -> {simplified_count} points "
                f"({reduction_pct:.1f}% reduction, epsilon={epsilon})"
            )
            
        except Exception as e:
            self.logger.error(f"Error applying RDP simplification to {axis} axis: {e}")
            raise
    
    def _get_segment_to_simplify(self, actions_list: List[Dict], params: Dict[str, Any]) -> Dict[str, Any]:
        """Determine which segment of actions to simplify and return prefix/suffix."""
        selected_indices = params.get('selected_indices')
        start_time_ms = params.get('start_time_ms')
        end_time_ms = params.get('end_time_ms')
        
        if selected_indices is not None and len(selected_indices) > 0:
            # Use selected indices
            valid_indices = sorted([
                i for i in selected_indices 
                if 0 <= i < len(actions_list)
            ])
            
            if len(valid_indices) < 2:
                return {
                    'prefix': [],
                    'segment': [],
                    'suffix': [],
                    'start_idx': -1,
                    'end_idx': -1
                }
            
            start_idx, end_idx = valid_indices[0], valid_indices[-1]
            
            return {
                'prefix': actions_list[:start_idx],
                'segment': actions_list[start_idx:end_idx + 1],
                'suffix': actions_list[end_idx + 1:],
                'start_idx': start_idx,
                'end_idx': end_idx
            }
        
        elif start_time_ms is not None and end_time_ms is not None:
            # Use time range
            start_idx, end_idx = self._get_action_indices_in_time_range(
                actions_list, start_time_ms, end_time_ms
            )
            
            if start_idx is None or end_idx is None or (end_idx - start_idx + 1) < 2:
                return {
                    'prefix': [],
                    'segment': [],
                    'suffix': [],
                    'start_idx': -1,
                    'end_idx': -1
                }
            
            return {
                'prefix': actions_list[:start_idx],
                'segment': actions_list[start_idx:end_idx + 1],
                'suffix': actions_list[end_idx + 1:],
                'start_idx': start_idx,
                'end_idx': end_idx
            }
        
        else:
            # Use entire list
            return {
                'prefix': [],
                'segment': list(actions_list),
                'suffix': [],
                'start_idx': 0,
                'end_idx': len(actions_list) - 1
            }
    
    def get_preview(self, funscript, axis: str = 'both', **parameters) -> Dict[str, Any]:
        """Generate a preview of the RDP simplification effect."""
        try:
            validated_params = self.validate_parameters(parameters)
        except ValueError as e:
            return {"error": str(e)}
        
        preview_info = {
            "filter_type": "RDP Simplification",
            "parameters": validated_params,
            "use_library": validated_params.get('use_library', True) and RDP_AVAILABLE
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
            
            segment_info = self._get_segment_to_simplify(actions_list, validated_params)
            segment_length = len(segment_info['segment'])
            
            # Rough estimation of reduction (without actually running RDP)
            epsilon = validated_params['epsilon']
            estimated_reduction = min(80, max(10, epsilon * 5))  # Rough heuristic
            
            axis_info = {
                "total_points": len(actions_list),
                "points_to_simplify": segment_length,
                "estimated_reduction_percent": estimated_reduction,
                "can_apply": segment_length >= 2,
                "epsilon": epsilon
            }
            
            preview_info[f"{current_axis}_axis"] = axis_info
        
        return preview_info