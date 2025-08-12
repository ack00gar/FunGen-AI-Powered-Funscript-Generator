"""
Peak-preserving resampling plugin for funscript transformations.

This plugin resamples funscripts while preserving the timing and intensity
of peaks and valleys, creating smooth sinusoidal transitions between them.
"""

import numpy as np
from typing import Dict, Any, List, Optional
import math

try:
    from .base_plugin import FunscriptTransformationPlugin
except ImportError:
    from funscript.plugins.base_plugin import FunscriptTransformationPlugin


class PeakPreservingResamplePlugin(FunscriptTransformationPlugin):
    """
    Peak-preserving resampling plugin.
    
    Resamples the funscript at regular intervals while preserving the exact
    timing and values of peaks and valleys, with smooth sinusoidal transitions.
    """
    
    @property
    def name(self) -> str:
        return "Resample"
    
    @property
    def description(self) -> str:
        return "Resamples with regular intervals while preserving peak timing"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            'resample_rate_ms': {
                'type': int,
                'required': False,
                'default': 50,
                'description': 'Time interval between resampled points (ms)',
                'constraints': {'min': 10, 'max': 1000}
            },
            'selected_indices': {
                'type': list,
                'required': False,
                'default': None,
                'description': 'Specific action indices to resample (None for full range)'
            }
        }
    
    def transform(self, funscript, axis: str = 'both', **parameters) -> Optional['DualAxisFunscript']:
        """Apply peak-preserving resampling to the specified axis."""
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
            self._apply_resampling_to_axis(funscript, current_axis, validated_params)
        
        return None  # Modifies in-place
    
    def _apply_resampling_to_axis(self, funscript, axis: str, params: Dict[str, Any]):
        """Apply peak-preserving resampling to a single axis."""
        actions_list = funscript.primary_actions if axis == 'primary' else funscript.secondary_actions
        
        if not actions_list or len(actions_list) < 3:
            self.logger.warning(f"Not enough points for resampling on {axis} axis")
            return
        
        # Determine segment to process
        segment_info = self._get_segment_to_process(actions_list, params)
        
        if len(segment_info['segment']) < 3:
            self.logger.warning(f"Segment on {axis} axis has < 3 points for resampling")
            return
        
        # Perform resampling
        resampled_actions = self._resample_with_peak_preservation(
            segment_info['segment'], 
            params['resample_rate_ms']
        )
        
        if not resampled_actions:
            self.logger.warning(f"Resampling failed for {axis} axis")
            return
        
        # Reconstruct actions list
        new_actions_list = (
            segment_info['prefix'] +
            resampled_actions +
            segment_info['suffix']
        )
        
        # Update the funscript IN-PLACE to preserve list identity for undo manager
        actions_target_list = funscript.primary_actions if axis == 'primary' else funscript.secondary_actions
        actions_target_list[:] = new_actions_list
        
        # Invalidate cache
        funscript._invalidate_cache(axis)
        
        original_count = len(segment_info['segment'])
        resampled_count = len(resampled_actions)
        
        self.logger.info(
            f"Applied peak-preserving resampling to {axis} axis: "
            f"{original_count} -> {resampled_count} points "
            f"(rate: {params['resample_rate_ms']}ms)"
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
    
    def _resample_with_peak_preservation(self, segment: List[Dict], resample_rate_ms: int) -> List[Dict]:
        """Resample segment while preserving peaks and valleys (match legacy behavior)."""
        if len(segment) < 3:
            return segment

        # 1) Identify anchors with flat-peak/valley handling
        anchors = self._find_anchors(segment)
        if len(anchors) < 2:
            return segment

        # 2) Generate new points using cosine easing per anchor-interval
        new_actions: List[Dict] = []
        new_actions.append(anchors[0])

        for i in range(len(anchors) - 1):
            p1 = anchors[i]
            p2 = anchors[i + 1]

            t1, pos1 = p1['at'], p1['pos']
            t2, pos2 = p2['at'], p2['pos']

            duration = float(t2 - t1)
            pos_delta = float(pos2 - pos1)

            if duration <= 0:
                # Skip invalid intervals
                continue

            current_time = t1 + resample_rate_ms
            while current_time < t2:
                progress = (current_time - t1) / duration
                eased_progress = (1 - np.cos(progress * np.pi)) / 2.0
                new_pos = pos1 + eased_progress * pos_delta
                new_actions.append({
                    'at': int(current_time),
                    'pos': int(round(np.clip(new_pos, 0, 100)))
                })
                current_time += resample_rate_ms

            # Ensure we include the next anchor exactly once
            if not new_actions or new_actions[-1]['at'] < p2['at']:
                new_actions.append(p2)

        return new_actions
    
    def _find_anchors(self, segment: List[Dict]) -> List[Dict]:
        """Find peaks and valleys (including flat peaks/valleys) to preserve as anchors."""
        anchors: List[Dict] = []

        # Always include first point
        anchors.append(segment[0])

        for i in range(1, len(segment) - 1):
            p_prev = segment[i - 1]['pos']
            p_curr = segment[i]['pos']
            p_next = segment[i + 1]['pos']

            # Strict local extrema
            if p_curr > p_prev and p_curr > p_next:
                anchors.append(segment[i])
            elif p_curr < p_prev and p_curr < p_next:
                anchors.append(segment[i])
            # Flat peak/valley handling (e.g., 80, 90, 90, 80)
            elif p_curr == p_next and p_curr != p_prev:
                j = i
                while j < len(segment) - 1 and segment[j]['pos'] == p_curr:
                    j += 1
                p_after_flat = segment[j]['pos']
                # If this plateau forms a peak/valley vs neighbors, take its middle
                if (p_curr > p_prev and p_curr > p_after_flat) or (p_curr < p_prev and p_curr < p_after_flat):
                    anchor_candidate = segment[(i + j - 1) // 2]
                    if not anchors or anchors[-1] != anchor_candidate:
                        anchors.append(anchor_candidate)

        # Always include last point
        if not anchors or anchors[-1] != segment[-1]:
            anchors.append(segment[-1])

        return anchors
    
    def _interpolate_between_anchors(self, anchors: List[Dict], target_time: int) -> float:
        """Interpolate position at target_time using sinusoidal transitions between anchors."""
        # Find the two anchors that bracket the target time
        left_anchor = None
        right_anchor = None
        
        for i in range(len(anchors) - 1):
            if anchors[i]['at'] <= target_time <= anchors[i + 1]['at']:
                left_anchor = anchors[i]
                right_anchor = anchors[i + 1]
                break
        
        if left_anchor is None or right_anchor is None:
            # Fallback to linear interpolation if bracketing fails
            if target_time <= anchors[0]['at']:
                return anchors[0]['pos']
            elif target_time >= anchors[-1]['at']:
                return anchors[-1]['pos']
            else:
                # Simple linear interpolation between first and last
                return anchors[0]['pos']
        
        if left_anchor['at'] == right_anchor['at']:
            return left_anchor['pos']
        
        # Sinusoidal interpolation between anchors
        time_span = right_anchor['at'] - left_anchor['at']
        time_progress = (target_time - left_anchor['at']) / time_span
        
        # Use sinusoidal easing for smooth transitions
        # This creates a smooth S-curve between the anchor points
        eased_progress = 0.5 * (1 - math.cos(time_progress * math.pi))
        
        position_span = right_anchor['pos'] - left_anchor['pos']
        interpolated_pos = left_anchor['pos'] + (position_span * eased_progress)
        
        return interpolated_pos
    
    def get_preview(self, funscript, axis: str = 'both', **parameters) -> Dict[str, Any]:
        """Generate a preview of the resampling effect."""
        try:
            validated_params = self.validate_parameters(parameters)
        except ValueError as e:
            return {"error": str(e)}
        
        preview_info = {
            "filter_type": "Peak-Preserving Resampling",
            "parameters": validated_params
        }
        
        resample_rate_ms = validated_params['resample_rate_ms']
        
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
            segment = segment_info['segment']
            
            if len(segment) >= 3:
                # Estimate resampling results
                start_time = segment[0]['at']
                end_time = segment[-1]['at']
                duration_ms = end_time - start_time
                
                # Estimate number of regular sample points
                estimated_samples = int(duration_ms / resample_rate_ms) + 1
                
                # Find anchors for estimation
                anchors = self._find_anchors(segment)
                estimated_total_points = len(set(range(start_time, end_time + 1, resample_rate_ms)) | 
                                               set(anchor['at'] for anchor in anchors))
                
                axis_info = {
                    "total_points": len(actions_list),
                    "segment_points": len(segment),
                    "estimated_resampled_points": estimated_total_points,
                    "estimated_anchors": len(anchors),
                    "resample_rate_ms": resample_rate_ms,
                    "segment_duration_ms": duration_ms,
                    "can_apply": True
                }
                
                # Calculate estimated change
                if len(segment) > 0:
                    change_pct = ((estimated_total_points - len(segment)) / len(segment)) * 100
                    axis_info["estimated_point_change_percent"] = round(change_pct, 1)
                    
                    if change_pct > 0:
                        axis_info["effect"] = "Increases point density"
                    elif change_pct < 0:
                        axis_info["effect"] = "Reduces point density"
                    else:
                        axis_info["effect"] = "Maintains similar density"
            else:
                axis_info = {
                    "total_points": len(actions_list),
                    "segment_points": len(segment),
                    "can_apply": False,
                    "error": "Not enough points for resampling"
                }
            
            preview_info[f"{current_axis}_axis"] = axis_info
        
        return preview_info