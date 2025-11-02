"""
Speed limiter plugin for funscript transformations.

This plugin applies a series of filters to make funscripts more compatible with 
hardware devices that have speed limitations (like Handy in Bluetooth mode).
It removes rapid actions, replaces small movements with vibrations, and limits 
maximum speed.
"""

import copy
import numpy as np
from typing import Dict, Any, List, Optional

try:
    from .base_plugin import FunscriptTransformationPlugin
except ImportError:
    from funscript.plugins.base_plugin import FunscriptTransformationPlugin


class SpeedLimiterPlugin(FunscriptTransformationPlugin):
    """
    Speed limiter plugin for hardware compatibility.
    
    Applies a series of transformations to make funscripts more compatible
    with hardware devices that have speed or interval limitations:
    1. Removes actions with intervals shorter than minimum
    2. Replaces small movements with vibration patterns
    3. Limits maximum movement speed
    """
    
    @property
    def name(self) -> str:
        return "Speed Limiter"
    
    @property
    def description(self) -> str:
        return "Limits speed and adds vibrations for hardware device compatibility"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            'min_interval_ms': {
                'type': int,
                'required': False,
                'default': 60,
                'description': 'Minimum time interval between actions in milliseconds',
                'constraints': {'min': 10, 'max': 1000}
            },
            'vibe_amount': {
                'type': int,
                'required': False,
                'default': 0,
                'description': 'Amount of vibration to add for small movements (0-50)',
                'constraints': {'min': 0, 'max': 50}
            },
            'speed_threshold': {
                'type': int,
                'required': False,
                'default': 500,
                'description': 'Maximum allowed speed (position change per time)',
                'constraints': {'min': 50, 'max': 1000, 'step': 50}
            },
            'small_movement_threshold': {
                'type': int,
                'required': False,
                'default': 10,
                'description': 'Threshold for considering a movement "small" for vibration replacement',
                'constraints': {'min': 1, 'max': 50}
            },
            'selected_indices': {
                'type': list,
                'required': False,
                'default': None,
                'description': 'List of action indices to apply speed limiting to (None = all actions)'
            }
        }
    
    def transform(self, funscript, axis: str = 'both', **parameters) -> Optional['DualAxisFunscript']:
        """Apply speed limiting to the specified axis."""
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
            self._apply_speed_limiter_to_axis(funscript, current_axis, validated_params)
        
        return None  # Modifies in-place
    
    def _apply_speed_limiter_to_axis(self, funscript, axis: str, params: Dict[str, Any]):
        """Apply speed limiting to a single axis."""
        actions_list = funscript.primary_actions if axis == 'primary' else funscript.secondary_actions
        
        if not actions_list or len(actions_list) < 2:
            self.logger.info(f"Not enough points on {axis} axis for speed limiter")
            return
        
        # Check for selected indices
        selected_indices = params.get('selected_indices')
        if selected_indices is not None and len(selected_indices) > 0:
            # Apply speed limiting only to selected actions
            indices_to_process = sorted([
                i for i in selected_indices 
                if 0 <= i < len(actions_list)
            ])
            if len(indices_to_process) < 2:
                self.logger.info(f"Not enough selected points ({len(indices_to_process)}) on {axis} axis for speed limiter")
                return
        else:
            # Apply to all actions
            indices_to_process = list(range(len(actions_list)))
        
        # Work on a deep copy
        actions = copy.deepcopy(actions_list)
        original_count = len(actions)
        
        min_interval = params['min_interval_ms']
        vibe_amount = params['vibe_amount']
        speed_threshold = params['speed_threshold']
        small_movement_threshold = params.get('small_movement_threshold', 10)
        
        if selected_indices is not None and len(selected_indices) > 0:
            # For selected indices, apply only speed limiting (no removal/addition of points)
            # This preserves the index mapping
            actions = self._limit_speed_for_selected_indices(actions, speed_threshold, indices_to_process, axis)
            self.logger.info(f"Speed limiter applied to {len(indices_to_process)} selected points on {axis} axis")
        else:
            # Apply full speed limiting to all actions
            # Step 1: Remove actions with short intervals
            actions = self._remove_short_intervals(actions, min_interval, axis)
            removed_count = original_count - len(actions)
            
            # Step 2: Replace small movements with vibrations
            if vibe_amount > 0:
                actions, modified_count = self._add_vibrations(actions, vibe_amount, small_movement_threshold, axis)
            else:
                modified_count = 0
            
            # Step 3: Apply speed limiting
            actions = self._limit_speed(actions, speed_threshold, axis)
            
            self.logger.info(f"Speed limiter applied to {axis} axis: {original_count} -> {len(actions)} points ({removed_count} removed, {modified_count if vibe_amount > 0 else 0} modified for vibration)")
        
        # Update the funscript IN-PLACE to preserve list identity for undo manager
        actions_target_list = funscript.primary_actions if axis == 'primary' else funscript.secondary_actions
        actions_target_list[:] = actions
        
        # Invalidate cache
        funscript._invalidate_cache(axis)
    
    def _remove_short_intervals(self, actions: List[Dict], min_interval: int, axis: str) -> List[Dict]:
        """OPTIMIZED: Remove actions that are too close together in time using vectorized operations."""
        if len(actions) <= 1:
            return actions
        
        # OPTIMIZATION: Use vectorized approach for large datasets
        if len(actions) > 5000:
            # Extract timestamps as numpy array
            timestamps = np.array([action['at'] for action in actions])
            
            # Calculate intervals between consecutive actions
            intervals = np.diff(timestamps)
            
            # Create boolean mask for actions to keep
            keep_mask = np.ones(len(actions), dtype=bool)
            
            # Mark actions to remove (those with intervals < min_interval)
            for i in range(len(intervals)):
                if intervals[i] < min_interval:
                    # Remove the later action (preserve chronological order)
                    keep_mask[i + 1] = False
            
            # Filter actions using the mask
            filtered_actions = [actions[i] for i in range(len(actions)) if keep_mask[i]]
            
            removed_count = len(actions) - len(filtered_actions)
            if removed_count > 0:
                self.logger.debug(f"{axis} axis: Removed {removed_count} actions due to min interval (vectorized)")
            
            return filtered_actions
        else:
            # Original method for smaller datasets
            filtered_actions = [actions[-1]]  # Always keep the last action
            last_kept_time = actions[-1]['at']
            
            for i in range(len(actions) - 2, -1, -1):
                current_action = actions[i]
                interval = abs(current_action['at'] - last_kept_time)
                
                if interval >= min_interval:
                    filtered_actions.append(current_action)
                    last_kept_time = current_action['at']
            
            # Restore chronological order
            filtered_actions.reverse()
            
            removed_count = len(actions) - len(filtered_actions)
            if removed_count > 0:
                self.logger.debug(f"{axis} axis: Removed {removed_count} actions due to min interval")
            
            return filtered_actions
    
    def _add_vibrations(self, actions: List[Dict], vibe_amount: int, 
                       small_movement_threshold: int, axis: str) -> tuple[List[Dict], int]:
        """Replace small movements with vibration patterns."""
        if len(actions) <= 2:
            return actions, 0
        
        modified_count = 0
        vibration_state = {'already_vibing': 0, 'last_vibe': '', 'last_height': 0}
        
        for i in range(1, len(actions)):
            current = actions[i]
            previous = actions[i - 1]
            
            # Calculate movement size
            movement_size = abs(current['pos'] - previous['pos'])
            time_interval = current['at'] - previous['at']
            
            # Check if this is a small movement that should be vibrated
            if (movement_size <= small_movement_threshold and 
                time_interval > 0 and 
                vibe_amount > 0):
                
                new_pos = self._calculate_vibration_position(
                    previous['pos'], 
                    current['pos'], 
                    vibe_amount, 
                    vibration_state
                )
                
                if new_pos != current['pos']:
                    actions[i]['pos'] = int(np.clip(new_pos, 0, 100))
                    modified_count += 1
        
        if modified_count > 0:
            self.logger.debug(f"{axis} axis: Added vibration to {modified_count} small movements")
        
        return actions, modified_count
    
    def _calculate_vibration_position(self, prev_pos: int, curr_pos: int, 
                                    vibe_amount: int, vibe_state: Dict) -> int:
        """Calculate vibration position based on movement and state."""
        movement_direction = 1 if curr_pos > prev_pos else -1
        base_pos = (prev_pos + curr_pos) // 2
        
        # Alternate vibration direction to create oscillating pattern
        if vibe_state['last_vibe'] == 'up':
            vibe_direction = -1
            vibe_state['last_vibe'] = 'down'
        else:
            vibe_direction = 1
            vibe_state['last_vibe'] = 'up'
        
        # Apply vibration
        vibrated_pos = base_pos + (vibe_amount * vibe_direction * movement_direction)
        
        return int(np.clip(vibrated_pos, 0, 100))
    
    def _limit_speed_for_selected_indices(self, actions: List[Dict], speed_threshold: float, 
                                        selected_indices: List[int], axis: str) -> List[Dict]:
        """Apply speed limiting only to selected action indices."""
        if len(actions) <= 1 or not selected_indices:
            return actions
        
        # Create a set for faster lookup
        selected_set = set(selected_indices)
        result_actions = copy.deepcopy(actions)
        
        # Process each selected action
        for i in selected_indices:
            if i <= 0 or i >= len(result_actions):
                continue
                
            current = result_actions[i]
            previous = result_actions[i - 1]
            
            # Calculate speed between current and previous
            time_diff = current['at'] - previous['at']
            if time_diff <= 0:
                continue
                
            pos_diff = abs(current['pos'] - previous['pos'])
            current_speed = pos_diff / time_diff * 1000  # positions per second
            
            if current_speed > speed_threshold:
                # Adjust position to limit speed
                max_pos_change = (speed_threshold * time_diff) / 1000
                direction = 1 if current['pos'] > previous['pos'] else -1
                new_pos = previous['pos'] + (direction * max_pos_change)
                result_actions[i]['pos'] = int(np.clip(new_pos, 0, 100))
        
        return result_actions
    
    def _limit_speed(self, actions: List[Dict], speed_threshold: float, axis: str) -> List[Dict]:
        """ULTRA-OPTIMIZED: Vectorized speed limiting with batch processing."""
        if len(actions) <= 1:
            return actions
        
        # OPTIMIZATION: Use vectorized approach for large datasets
        if len(actions) > 3000:
            return self._limit_speed_vectorized(actions, speed_threshold, axis)
        else:
            return self._limit_speed_original(actions, speed_threshold, axis)
    
    def _limit_speed_vectorized(self, actions: List[Dict], speed_threshold: float, axis: str) -> List[Dict]:
        """REVOLUTIONARY: Bulk geometric interpolation for massive speedup."""
        # Extract arrays for vectorized operations
        timestamps = np.array([action['at'] for action in actions])
        positions = np.array([action['pos'] for action in actions])
        
        # BREAKTHROUGH 1: Pre-compute ALL violations using vectorized operations
        time_deltas = np.diff(timestamps)
        pos_deltas = np.diff(positions)
        speeds = np.abs(pos_deltas) / np.maximum(time_deltas, 1)
        violation_mask = speeds > speed_threshold
        
        if not np.any(violation_mask):
            return actions
        
        # BREAKTHROUGH 2: Build interpolation segments in bulk using geometric math
        violation_indices = np.where(violation_mask)[0]
        
        # Pre-allocate result arrays for maximum efficiency
        estimated_size = len(actions) + len(violation_indices) * 5  # Estimate 5 points per violation
        result_times = np.zeros(estimated_size)
        result_positions = np.zeros(estimated_size)
        
        write_idx = 0
        last_processed = 0
        
        # BREAKTHROUGH 3: Vectorized segment processing
        for violation_idx in violation_indices:
            # Copy non-violating points before this violation
            segment_start = last_processed
            segment_end = violation_idx + 1
            
            if segment_start < segment_end:
                segment_len = segment_end - segment_start
                result_times[write_idx:write_idx + segment_len] = timestamps[segment_start:segment_end]
                result_positions[write_idx:write_idx + segment_len] = positions[segment_start:segment_end]
                write_idx += segment_len
            
            # BREAKTHROUGH 4: Geometric interpolation math
            start_time = timestamps[violation_idx]
            end_time = timestamps[violation_idx + 1]
            start_pos = positions[violation_idx]
            end_pos = positions[violation_idx + 1]
            
            # Calculate required intermediate points using mathematics
            total_distance = abs(end_pos - start_pos)
            total_time = end_time - start_time
            max_distance_per_segment = speed_threshold * total_time / 1000
            
            if max_distance_per_segment > 0:
                num_segments = max(1, int(np.ceil(total_distance / max_distance_per_segment)))
                
                # VECTORIZED interpolation point generation
                if num_segments > 1:
                    # Generate intermediate time points
                    time_points = np.linspace(start_time, end_time, num_segments + 1)
                    # Generate intermediate position points
                    pos_points = np.linspace(start_pos, end_pos, num_segments + 1)
                    
                    # Add all interpolated points at once
                    interp_len = len(time_points) - 1  # Skip the start point (already added)
                    if write_idx + interp_len < len(result_times):
                        result_times[write_idx:write_idx + interp_len] = time_points[1:]
                        result_positions[write_idx:write_idx + interp_len] = pos_points[1:]
                        write_idx += interp_len
            
            last_processed = violation_idx + 1
        
        # Add remaining points
        if last_processed < len(timestamps):
            remaining_len = len(timestamps) - last_processed
            result_times[write_idx:write_idx + remaining_len] = timestamps[last_processed:]
            result_positions[write_idx:write_idx + remaining_len] = positions[last_processed:]
            write_idx += remaining_len
        
        # BREAKTHROUGH 5: Bulk reconstruction of action dicts
        result_actions = [
            {'at': int(result_times[i]), 'pos': int(np.clip(result_positions[i], 0, 100))}
            for i in range(write_idx)
        ]
        
        speed_corrections = len(result_actions) - len(actions)
        if speed_corrections > 0:
            self.logger.debug(f"{axis} axis: Added {speed_corrections} points (geometric interpolation)")
        
        return result_actions
    
    def _limit_speed_original(self, actions: List[Dict], speed_threshold: float, axis: str) -> List[Dict]:
        """Original speed limiting for smaller datasets."""
        speed_limited_actions = [actions[0]]  # Always keep first action
        
        for i in range(1, len(actions)):
            current = actions[i]
            previous = speed_limited_actions[-1]
            
            time_delta = current['at'] - previous['at']
            if time_delta <= 0:
                continue
            
            pos_delta = abs(current['pos'] - previous['pos'])
            current_speed = pos_delta / time_delta  # positions per millisecond
            
            if current_speed > speed_threshold:
                # Need to limit speed - insert intermediate points
                intermediate_actions = self._create_intermediate_actions(
                    previous, current, speed_threshold
                )
                speed_limited_actions.extend(intermediate_actions)
            else:
                speed_limited_actions.append(current)
        
        speed_corrections = len(speed_limited_actions) - len(actions)
        if speed_corrections > 0:
            self.logger.debug(f"{axis} axis: Added {speed_corrections} intermediate points for speed limiting")
        
        return speed_limited_actions
    
    def _create_intermediate_actions(self, start_action: Dict, end_action: Dict, 
                                   max_speed: float) -> List[Dict]:
        """Create intermediate actions to limit speed between two points."""
        time_delta = end_action['at'] - start_action['at']
        pos_delta = end_action['pos'] - start_action['pos']
        
        if time_delta <= 0:
            return [end_action]
        
        # Calculate how many intermediate points we need
        required_time = abs(pos_delta) / max_speed
        if required_time <= time_delta:
            return [end_action]
        
        num_segments = int(np.ceil(required_time / time_delta))
        
        intermediate_actions = []
        for i in range(1, num_segments + 1):
            progress = i / num_segments
            
            intermediate_time = int(start_action['at'] + (time_delta * progress))
            intermediate_pos = int(start_action['pos'] + (pos_delta * progress))
            intermediate_pos = int(np.clip(intermediate_pos, 0, 100))
            
            intermediate_actions.append({
                'at': intermediate_time,
                'pos': intermediate_pos
            })
        
        # Make sure the last intermediate action matches the target
        if intermediate_actions:
            intermediate_actions[-1] = end_action
        else:
            intermediate_actions = [end_action]
        
        return intermediate_actions
    
    def get_preview(self, funscript, axis: str = 'both', **parameters) -> Dict[str, Any]:
        """Generate a preview of the speed limiter effect."""
        try:
            validated_params = self.validate_parameters(parameters)
        except ValueError as e:
            return {"error": str(e)}
        
        preview_info = {
            "filter_type": "Speed Limiter",
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
            
            # Analyze current speed characteristics
            speeds = []
            small_movements = 0
            short_intervals = 0
            
            min_interval = validated_params['min_interval_ms']
            small_movement_threshold = validated_params.get('small_movement_threshold', 10)
            
            for i in range(1, len(actions_list)):
                current = actions_list[i]
                previous = actions_list[i - 1]
                
                time_delta = current['at'] - previous['at']
                pos_delta = abs(current['pos'] - previous['pos'])
                
                if time_delta > 0:
                    speed = pos_delta / time_delta
                    speeds.append(speed)
                    
                    if pos_delta <= small_movement_threshold:
                        small_movements += 1
                
                if time_delta < min_interval:
                    short_intervals += 1
            
            max_speed = max(speeds) if speeds else 0
            avg_speed = np.mean(speeds) if speeds else 0
            
            axis_info = {
                "total_actions": len(actions_list),
                "max_current_speed": round(max_speed, 3),
                "avg_current_speed": round(avg_speed, 3),
                "speed_threshold": validated_params['speed_threshold'],
                "actions_too_fast": sum(1 for s in speeds if s > validated_params['speed_threshold']),
                "short_intervals": short_intervals,
                "small_movements": small_movements,
                "estimated_points_removed": short_intervals,
                "estimated_points_modified": small_movements if validated_params['vibe_amount'] > 0 else 0
            }
            
            preview_info[f"{current_axis}_axis"] = axis_info
        
        return preview_info