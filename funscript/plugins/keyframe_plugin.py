"""
Keyframe simplification plugin for funscript transformations.

This plugin simplifies funscripts by identifying and preserving only the most
significant peaks and valleys (keyframes) while removing less important points.
"""

import numpy as np
from typing import Dict, Any, List, Optional

try:
    from .base_plugin import FunscriptTransformationPlugin
except ImportError:
    from funscript.plugins.base_plugin import FunscriptTransformationPlugin


class KeyframePlugin(FunscriptTransformationPlugin):
    """
    Keyframe simplification plugin.
    
    Uses iterative global refinement to simplify the script to only the most
    significant peaks and valleys, creating a cleaner, more focused experience.
    """
    
    @property
    def name(self) -> str:
        return "Keyframes"
    
    @property
    def description(self) -> str:
        return "Simplifies to significant peaks and valleys using keyframe analysis"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            'position_tolerance': {
                'type': int,
                'required': False,
                'default': 10,
                'description': 'Minimum position difference for significance',
                'constraints': {'min': 1, 'max': 50}
            },
            'time_tolerance_ms': {
                'type': int,
                'required': False,
                'default': 50,
                'description': 'Minimum time difference between keyframes (ms)',
                'constraints': {'min': 10, 'max': 1000}
            },
            'selected_indices': {
                'type': list,
                'required': False,
                'default': None,
                'description': 'Specific action indices to analyze (None for full range)'
            }
        }
    
    def transform(self, funscript, axis: str = 'both', **parameters) -> Optional['DualAxisFunscript']:
        """Apply keyframe simplification to the specified axis."""
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
            self._apply_keyframe_simplification_to_axis(funscript, current_axis, validated_params)
        
        return None  # Modifies in-place
    
    def _apply_keyframe_simplification_to_axis(self, funscript, axis: str, params: Dict[str, Any]):
        """Apply keyframe simplification to a single axis."""
        actions_list = funscript.primary_actions if axis == 'primary' else funscript.secondary_actions
        
        if not actions_list or len(actions_list) < 3:
            self.logger.warning(f"Not enough points for keyframe simplification on {axis} axis")
            return
        
        # Determine segment to process
        segment_info = self._get_segment_to_process(actions_list, params)
        
        if len(segment_info['segment']) < 3:
            self.logger.warning(f"Segment on {axis} axis has < 3 points for keyframe analysis")
            return
        
        # Find keyframes
        keyframes = self._find_keyframes(segment_info['segment'], params)
        
        if not keyframes:
            self.logger.warning(f"No keyframes found for {axis} axis")
            return
        
        # Reconstruct actions list with keyframes
        new_actions_list = (
            segment_info['prefix'] +
            keyframes +
            segment_info['suffix']
        )
        
        # Update the funscript IN-PLACE to preserve list identity for undo manager
        actions_target_list = funscript.primary_actions if axis == 'primary' else funscript.secondary_actions
        actions_target_list[:] = new_actions_list
        
        # Invalidate cache
        funscript._invalidate_cache(axis)
        
        original_count = len(segment_info['segment'])
        keyframe_count = len(keyframes)
        reduction_pct = ((original_count - keyframe_count) / original_count) * 100
        
        self.logger.info(
            f"Applied keyframe simplification to {axis} axis: "
            f"{original_count} -> {keyframe_count} points "
            f"({reduction_pct:.1f}% reduction)"
        )
    
    def _get_segment_to_process(self, actions_list: List[Dict], params: Dict[str, Any]) -> Dict[str, Any]:
        """Determine which segment of actions to process."""
        selected_indices = params.get('selected_indices')
        
        if selected_indices is not None and len(selected_indices) > 0:
            # Use selected indices
            valid_indices = sorted([
                i for i in selected_indices 
                if 0 <= i < len(actions_list)
            ])
            
            if len(valid_indices) < 3:
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
        else:
            # Use entire list
            return {
                'prefix': [],
                'segment': list(actions_list),
                'suffix': [],
                'start_idx': 0,
                'end_idx': len(actions_list) - 1
            }
    
    def _find_keyframes(self, segment: List[Dict], params: Dict[str, Any]) -> List[Dict]:
        """OPTIMIZED: Find significant keyframes using vectorized numpy operations."""
        position_tolerance = params['position_tolerance']
        time_tolerance_ms = params['time_tolerance_ms']

        if len(segment) < 3:
            return segment

        # Use vectorized algorithm for all dataset sizes (simpler, nearly as fast for small, much faster for large)
        return self._find_keyframes_vectorized(segment, params)
    
    def _find_keyframes_vectorized(self, segment: List[Dict], params: Dict[str, Any]) -> List[Dict]:
        """Vectorized keyframe detection for large datasets."""
        position_tolerance = params['position_tolerance']
        time_tolerance_ms = params['time_tolerance_ms']
        
        # Extract positions and timestamps as numpy arrays
        positions = np.array([action['pos'] for action in segment])
        timestamps = np.array([action['at'] for action in segment])
        
        # VECTORIZED: Find local extrema using numpy operations
        # Create shifted arrays for comparison
        pos_prev = positions[:-2]  # positions[i-1]
        pos_curr = positions[1:-1]  # positions[i]
        pos_next = positions[2:]    # positions[i+1]
        
        # Detect peaks: curr > prev AND curr >= next
        # Detect valleys: curr < prev AND curr <= next
        is_peak = (pos_curr > pos_prev) & (pos_curr >= pos_next)
        is_valley = (pos_curr < pos_prev) & (pos_curr <= pos_next)
        is_extremum = is_peak | is_valley
        
        # Build extrema list (always include first and last)
        extrema_indices = [0]
        extrema_indices.extend(np.where(is_extremum)[0] + 1)  # +1 because we compared shifted arrays
        extrema_indices.append(len(segment) - 1)
        
        # Remove duplicates and sort
        extrema_indices = sorted(set(extrema_indices))
        extrema = [segment[i] for i in extrema_indices]
        
        # VECTORIZED significance calculation and filtering
        if len(extrema) <= 2:
            return extrema
            
        # Convert to numpy arrays for vectorized processing
        ext_positions = np.array([ext['pos'] for ext in extrema])
        ext_timestamps = np.array([ext['at'] for ext in extrema])
        
        # Calculate significance for all internal points at once
        while len(extrema) > 2:
            if len(ext_positions) <= 2:
                break
                
            # Vectorized significance calculation for internal points
            prev_pos = ext_positions[:-2]
            curr_pos = ext_positions[1:-1] 
            next_pos = ext_positions[2:]
            prev_time = ext_timestamps[:-2]
            curr_time = ext_timestamps[1:-1]
            next_time = ext_timestamps[2:]
            
            # Calculate projection-based significance
            durations = next_time.astype(np.float64) - prev_time.astype(np.float64)
            time_deltas = curr_time.astype(np.float64) - prev_time.astype(np.float64)
            progress = np.divide(time_deltas, durations, 
                               out=np.zeros_like(time_deltas), where=durations!=0)
            projected_pos = prev_pos + progress * (next_pos - prev_pos)
            significance_scores = np.abs(curr_pos - projected_pos)
            
            # Find weakest point
            min_idx = np.argmin(significance_scores)
            min_significance = significance_scores[min_idx]
            
            if min_significance < position_tolerance:
                # Remove the weakest point (add 1 because we're looking at internal points)
                remove_idx = min_idx + 1
                extrema.pop(remove_idx)
                ext_positions = np.delete(ext_positions, remove_idx)
                ext_timestamps = np.delete(ext_timestamps, remove_idx)
            else:
                break
                
        return extrema
    
    def _find_keyframes_ultra_fast(self, segment: List[Dict], params: Dict[str, Any]) -> List[Dict]:
        """Ultra-fast approximate keyframe detection for massive datasets."""
        position_tolerance = params['position_tolerance']
        time_tolerance_ms = params['time_tolerance_ms']
        
        # STRATEGY: Use statistical sampling + local maxima detection
        positions = np.array([action['pos'] for action in segment])
        timestamps = np.array([action['at'] for action in segment])
        
        # Step 1: Reduce problem size with uniform sampling
        sample_rate = max(1, len(segment) // 10000)  # Reduce to ~10k points max
        sampled_indices = np.arange(0, len(positions), sample_rate)
        if sampled_indices[-1] != len(positions) - 1:
            sampled_indices = np.append(sampled_indices, len(positions) - 1)
        
        sampled_positions = positions[sampled_indices]
        sampled_timestamps = timestamps[sampled_indices]
        sampled_segment = [segment[i] for i in sampled_indices]
        
        # Step 2: Apply fast local maxima/minima detection
        # Use scipy.signal.find_peaks equivalent with numpy
        pos_diff1 = np.diff(sampled_positions)
        pos_diff2 = np.diff(pos_diff1)
        
        # Find local extrema (where derivative changes sign)
        sign_changes = np.diff(np.sign(pos_diff1))
        extrema_indices = np.where(np.abs(sign_changes) > 0)[0] + 1  # +1 to adjust index
        
        # Always include first and last points
        keyframe_indices = [0]
        keyframe_indices.extend(extrema_indices.tolist())
        keyframe_indices.append(len(sampled_segment) - 1)
        
        # Remove duplicates and sort
        keyframe_indices = sorted(set(keyframe_indices))
        
        # Step 3: Filter by significance using vectorized operations
        if len(keyframe_indices) > 3:
            # Quick significance test - keep points that differ significantly from neighbors
            significance_mask = np.ones(len(keyframe_indices), dtype=bool)
            
            for i in range(1, len(keyframe_indices) - 1):
                idx = keyframe_indices[i]
                prev_idx = keyframe_indices[i-1]
                next_idx = keyframe_indices[i+1]
                
                curr_pos = sampled_positions[idx]
                prev_pos = sampled_positions[prev_idx]
                next_pos = sampled_positions[next_idx]
                
                # Simple significance: minimum distance to neighbors
                significance = min(abs(curr_pos - prev_pos), abs(curr_pos - next_pos))
                if significance < position_tolerance:
                    significance_mask[i] = False
            
            keyframe_indices = [keyframe_indices[i] for i in range(len(keyframe_indices)) if significance_mask[i]]
        
        # Step 4: Map back to original segment
        keyframes = [sampled_segment[i] for i in keyframe_indices]
        
        return keyframes
    
    def _find_keyframes_original(self, segment: List[Dict], params: Dict[str, Any]) -> List[Dict]:
        """Original keyframe detection for smaller datasets."""
        position_tolerance = params['position_tolerance']
        time_tolerance_ms = params['time_tolerance_ms']

        # Pass 1: Find all local extrema (peaks and valleys), including equal-to neighbor cases
        extrema: List[Dict] = [segment[0]]
        for i in range(1, len(segment) - 1):
            p_prev = segment[i - 1]['pos']
            p_curr = segment[i]['pos']
            p_next = segment[i + 1]['pos']
            if (p_curr > p_prev and p_curr >= p_next) or (p_curr < p_prev and p_curr <= p_next):
                extrema.append(segment[i])
        extrema.append(segment[-1])

        # Pass 2: Iteratively remove the least significant extremum (projection-based significance)
        def calc_significance(ext: List[Dict], idx: int) -> float:
            if idx == 0 or idx == len(ext) - 1:
                return float('inf')
            p_prev, p_curr, p_next = ext[idx - 1], ext[idx], ext[idx + 1]
            duration = float(p_next['at'] - p_prev['at'])
            if duration <= 0:
                return float('inf')
            progress = (p_curr['at'] - p_prev['at']) / duration
            projected_pos = p_prev['pos'] + progress * (p_next['pos'] - p_prev['pos'])
            return abs(p_curr['pos'] - projected_pos)

        while len(extrema) > 2:
            min_significance = float('inf')
            weakest_idx = -1
            for i in range(1, len(extrema) - 1):
                s = calc_significance(extrema, i)
                if s < min_significance:
                    min_significance = s
                    weakest_idx = i
            if weakest_idx != -1 and min_significance < position_tolerance:
                extrema.pop(weakest_idx)
            else:
                break

        # Pass 3: Enforce time tolerance by spacing and choosing the stronger candidate
        if time_tolerance_ms > 0 and len(extrema) > 1:
            final_keyframes: List[Dict] = [extrema[0]]
            for i in range(1, len(extrema)):
                if (extrema[i]['at'] - final_keyframes[-1]['at']) >= time_tolerance_ms:
                    final_keyframes.append(extrema[i])
                else:
                    # Choose the one further from neutral (50)
                    if abs(extrema[i]['pos'] - 50) > abs(final_keyframes[-1]['pos'] - 50):
                        final_keyframes[-1] = extrema[i]
        else:
            final_keyframes = extrema

        return final_keyframes
    
    def _calculate_significance(self, extrema: List[Dict], idx: int, position_tolerance: int) -> float:
        """Calculate significance score for an extremum."""
        if idx == 0 or idx == len(extrema) - 1:
            return float('inf')  # Endpoints are always significant
        
        current_pos = extrema[idx]['pos']
        
        # Find the maximum position difference to neighboring extrema
        prev_pos = extrema[idx - 1]['pos']
        next_pos = extrema[idx + 1]['pos']
        
        # Significance is the minimum distance to neighbors
        # Higher values mean more significant peaks/valleys
        significance = min(abs(current_pos - prev_pos), abs(current_pos - next_pos))
        
        return max(significance, 1)  # Ensure positive significance
    
    def _check_time_tolerance(self, scored_extrema: List[tuple], time_tolerance_ms: int) -> bool:
        """Check if current keyframes meet minimum time tolerance."""
        if len(scored_extrema) <= 1:
            return True
        
        for i in range(1, len(scored_extrema)):
            action_curr, _ = scored_extrema[i]
            action_prev, _ = scored_extrema[i - 1]
            
            time_diff = action_curr['at'] - action_prev['at']
            if time_diff < time_tolerance_ms:
                return False
        
        return True
    
    def _recalculate_significance_scores(self, scored_extrema: List[tuple], removed_idx: int, position_tolerance: int):
        """Recalculate significance scores after removing a point."""
        # Convert to list of actions for easier processing
        actions = [action for action, _ in scored_extrema]
        
        # Recalculate scores for points that might be affected
        for i in range(max(1, removed_idx - 1), min(len(actions) - 1, removed_idx + 2)):
            if i < len(scored_extrema):
                action = scored_extrema[i][0]
                new_significance = self._calculate_significance(actions, i, position_tolerance)
                scored_extrema[i] = (action, new_significance)
    
    def get_preview(self, funscript, axis: str = 'both', **parameters) -> Dict[str, Any]:
        """Generate a preview of the keyframe simplification effect."""
        try:
            validated_params = self.validate_parameters(parameters)
        except ValueError as e:
            return {"error": str(e)}
        
        preview_info = {
            "filter_type": "Keyframe Simplification",
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
            
            segment_info = self._get_segment_to_process(actions_list, validated_params)
            segment_length = len(segment_info['segment'])
            
            if segment_length >= 3:
                # Estimate keyframes without actually modifying
                keyframes = self._find_keyframes(segment_info['segment'], validated_params)
                estimated_keyframes = len(keyframes)
                reduction_pct = ((segment_length - estimated_keyframes) / segment_length) * 100
                
                axis_info = {
                    "total_points": len(actions_list),
                    "points_to_analyze": segment_length,
                    "estimated_keyframes": estimated_keyframes,
                    "estimated_reduction_percent": round(reduction_pct, 1),
                    "can_apply": True,
                    "position_tolerance": validated_params['position_tolerance'],
                    "time_tolerance_ms": validated_params['time_tolerance_ms']
                }
            else:
                axis_info = {
                    "total_points": len(actions_list),
                    "points_to_analyze": segment_length,
                    "can_apply": False,
                    "error": "Not enough points for keyframe analysis"
                }
            
            preview_info[f"{current_axis}_axis"] = axis_info
        
        return preview_info