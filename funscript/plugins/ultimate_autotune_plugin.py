"""
Ultimate Autotune Plugin

A comprehensive multi-stage enhancement pipeline that combines speed limiting,
resampling, smoothing, amplification, and keyframe simplification to create
highly optimized funscripts.

This plugin implements the "ultimate autotune" algorithm that was previously
hardcoded in the DualAxisFunscript class.
"""

from typing import Dict, Any, List, Optional
import copy
from funscript.plugins.base_plugin import FunscriptTransformationPlugin
from funscript.plugins.resample_plugin import PeakPreservingResamplePlugin
from funscript.plugins.savgol_filter_plugin import SavgolFilterPlugin
from funscript.plugins.amplify_plugin import AmplifyPlugin
from funscript.plugins.keyframe_plugin import KeyframePlugin
from funscript.plugins.anti_jerk_plugin import AntiJerkPlugin


class UltimateAutotunePlugin(FunscriptTransformationPlugin):
    """
    Ultimate Autotune Plugin - Multi-stage funscript enhancement pipeline.

    This plugin applies a sophisticated 8-stage processing pipeline using the plugin system:
    1. High-speed point removal (custom speed limiter - no plugin available)
    2. Peak-preserving resample (50ms) via PeakPreservingResamplePlugin
    3. Savitzky-Golay smoothing (window=11, order=7) via SavgolFilterPlugin
    4. Peak-preserving resample (50ms) via PeakPreservingResamplePlugin
    5. Amplification (scale=1.25, center=50) via AmplifyPlugin
    6. Peak-preserving resample (50ms) via PeakPreservingResamplePlugin
    7. Keyframe simplification (tolerance=10, time=50ms) via KeyframePlugin
    8. Anti-jerk filter (removes intermediate jerky points) via AntiJerkPlugin
    """
    
    @property
    def name(self) -> str:
        return "Ultimate Autotune"
    
    @property
    def description(self) -> str:
        return "Comprehensive 8-stage enhancement pipeline for optimal funscript quality"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def author(self) -> str:
        return "FunGen Team"
    
    @property
    def ui_preference(self) -> str:
        """Apply directly with default parameters - no popup needed."""
        return 'direct'
    
    @property
    def parameters_schema(self) -> Dict[str, Dict[str, Any]]:
        return {
            "speed_threshold": {
                "type": float,
                "default": 1000.0,
                "constraints": {"min": 100.0, "max": 5000.0},
                "description": "Speed threshold for high-speed point removal (units/sec)"
            },
            "resample_rate_ms": {
                "type": int,
                "default": 50,
                "constraints": {"min": 10, "max": 200},
                "description": "Resampling rate in milliseconds"
            },
            "sg_window_length": {
                "type": int,
                "default": 11,
                "constraints": {"min": 5, "max": 51},
                "description": "Savitzky-Golay filter window length (must be odd)"
            },
            "sg_polyorder": {
                "type": int,
                "default": 7,
                "constraints": {"min": 1, "max": 10},
                "description": "Savitzky-Golay filter polynomial order"
            },
            "amplify_scale": {
                "type": float,
                "default": 1.25,
                "constraints": {"min": 0.5, "max": 3.0},
                "description": "Amplification scale factor"
            },
            "amplify_center": {
                "type": int,
                "default": 50,
                "constraints": {"min": 0, "max": 100},
                "description": "Amplification center value"
            },
            "keyframe_position_tolerance": {
                "type": int,
                "default": 10,
                "constraints": {"min": 1, "max": 50},
                "description": "Position tolerance for keyframe simplification"
            },
            "keyframe_time_tolerance_ms": {
                "type": int,
                "default": 50,
                "constraints": {"min": 10, "max": 500},
                "description": "Time tolerance for keyframe simplification (ms)"
            },
            "anti_jerk_threshold": {
                "type": float,
                "default": 20.0,
                "constraints": {"min": 5.0, "max": 40.0},
                "description": "Maximum oscillation size to consider as jerk"
            },
            "anti_jerk_min_main_movement": {
                "type": float,
                "default": 50.0,
                "constraints": {"min": 20.0, "max": 100.0},
                "description": "Minimum main movement for anti-jerk processing"
            },
            "anti_jerk_deviation_threshold": {
                "type": float,
                "default": 15.0,
                "constraints": {"min": 5.0, "max": 30.0},
                "description": "Maximum deviation from direct path to allow"
            },
            "selected_indices": {
                "type": list,
                "required": False,
                "default": None,
                "description": "Specific action indices to process (overrides time range)"
            }
        }
    
    def transform(self, funscript_obj, axis: str = 'both', **parameters) -> Optional['DualAxisFunscript']:
        """
        Apply the ultimate autotune pipeline to the specified axis.
        
        Args:
            funscript_obj: The DualAxisFunscript object to process
            axis: Which axis to process ('primary', 'secondary', or 'both')
            **parameters: Parameter overrides
            
        Returns:
            Modified DualAxisFunscript object, or None if processing failed
        """
        try:
            # Validate parameters
            params = self.validate_parameters(parameters)
            
            # Determine which axes to process
            axes_to_process = []
            if axis == 'both':
                if funscript_obj.primary_actions:
                    axes_to_process.append('primary')
                if funscript_obj.secondary_actions:
                    axes_to_process.append('secondary')
            else:
                axes_to_process = [axis]
            
            self.logger.info(f"Starting Ultimate Autotune pipeline on {axes_to_process} axis/axes...")
            
            for current_axis in axes_to_process:
                # Get reference to the actions list
                actions_list_ref = (funscript_obj.primary_actions if current_axis == 'primary' 
                                  else funscript_obj.secondary_actions)
                
                if not actions_list_ref or len(actions_list_ref) < 2:
                    self.logger.warning(f"Insufficient data for ultimate autotune on {current_axis} axis")
                    continue
                
                initial_count = len(actions_list_ref)
                
                # Determine which indices to process
                selected_indices = params.get('selected_indices')
                if selected_indices and len(selected_indices) >= 2:
                    # Process only selected points
                    indices_to_process = sorted([i for i in selected_indices if 0 <= i < len(actions_list_ref)])
                    if len(indices_to_process) < 2:
                        self.logger.warning(f"Not enough valid indices for ultimate autotune on {current_axis} axis")
                        continue
                    
                    # Create a temporary funscript with only selected actions
                    temp_fs = funscript_obj.__class__(logger=self.logger)
                    selected_actions = [actions_list_ref[i] for i in indices_to_process]
                    
                    if current_axis == 'primary':
                        temp_fs.primary_actions = copy.deepcopy(selected_actions)
                    else:
                        temp_fs.secondary_actions = copy.deepcopy(selected_actions)
                    
                    self.logger.info(f"Processing {len(indices_to_process)} selected points on {current_axis} axis")
                else:
                    # Process entire timeline
                    temp_fs = funscript_obj.__class__(logger=self.logger)
                    if current_axis == 'primary':
                        temp_fs.primary_actions = copy.deepcopy(actions_list_ref)
                    else:
                        temp_fs.secondary_actions = copy.deepcopy(actions_list_ref)
                    indices_to_process = None
                
                # === STEP 1: Custom Speed Limiter (Remove high-speed points) ===
                self.logger.debug(f"Ultimate Autotune ({current_axis}): (1) Removing high-speed points")
                self._apply_custom_speed_limiter(temp_fs, current_axis, params["speed_threshold"])
                
                # === STEP 2: Resample ===
                self.logger.debug(f"Ultimate Autotune ({current_axis}): (2) First resampling")
                resample_plugin = PeakPreservingResamplePlugin()
                resample_plugin.transform(temp_fs, axis=current_axis, resample_rate_ms=params["resample_rate_ms"])
                
                # === STEP 3: Smooth SG ===
                self.logger.debug(f"Ultimate Autotune ({current_axis}): (3) Applying Savitzky-Golay filter")
                savgol_plugin = SavgolFilterPlugin()
                savgol_plugin.transform(temp_fs, axis=current_axis,
                                      window_length=params["sg_window_length"],
                                      polyorder=params["sg_polyorder"])
                
                # === STEP 4: Resample ===
                self.logger.debug(f"Ultimate Autotune ({current_axis}): (4) Second resampling")
                resample_plugin = PeakPreservingResamplePlugin()
                resample_plugin.transform(temp_fs, axis=current_axis, resample_rate_ms=params["resample_rate_ms"])
                
                # === STEP 5: Amplify ===
                self.logger.debug(f"Ultimate Autotune ({current_axis}): (5) Amplifying values")
                amplify_plugin = AmplifyPlugin()
                amplify_plugin.transform(temp_fs, axis=current_axis,
                                       scale_factor=params["amplify_scale"],
                                       center_value=params["amplify_center"])
                
                # === STEP 6: Resample ===
                self.logger.debug(f"Ultimate Autotune ({current_axis}): (6) Third resampling")
                resample_plugin = PeakPreservingResamplePlugin()
                resample_plugin.transform(temp_fs, axis=current_axis, resample_rate_ms=params["resample_rate_ms"])
                
                # === STEP 7: Keyframes ===
                self.logger.debug(f"Ultimate Autotune ({current_axis}): (7) Simplifying to keyframes")
                keyframe_plugin = KeyframePlugin()
                keyframe_plugin.transform(temp_fs, axis=current_axis,
                                        position_tolerance=params["keyframe_position_tolerance"],
                                        time_tolerance_ms=params["keyframe_time_tolerance_ms"])

                # === STEP 8: Anti-Jerk Filter ===
                self.logger.debug(f"Ultimate Autotune ({current_axis}): (8) Applying anti-jerk filter")
                anti_jerk_plugin = AntiJerkPlugin()
                anti_jerk_plugin.transform(temp_fs, axis=current_axis,
                                          jerk_threshold=params["anti_jerk_threshold"],
                                          min_main_movement=params["anti_jerk_min_main_movement"],
                                          deviation_threshold=params["anti_jerk_deviation_threshold"])

                # Get the final processed actions
                final_actions = (temp_fs.primary_actions if current_axis == 'primary' 
                               else temp_fs.secondary_actions)
                
                if indices_to_process:
                    # Replace only the selected points
                    # First, remove the old selected points (in reverse order to maintain indices)
                    for idx in reversed(indices_to_process):
                        del actions_list_ref[idx]
                    
                    # Then insert the new processed points at the right positions
                    # We need to insert them based on their timestamps to maintain order
                    for action in final_actions:
                        # Find the correct position to insert based on timestamp
                        insert_idx = 0
                        for i, existing_action in enumerate(actions_list_ref):
                            if existing_action['at'] > action['at']:
                                insert_idx = i
                                break
                            insert_idx = i + 1
                        actions_list_ref.insert(insert_idx, action)
                    
                    final_count = len(final_actions)
                    self.logger.info(f"Ultimate Autotune completed on {current_axis} axis (selection). "
                                   f"Processed points: {len(indices_to_process)} -> {final_count}")
                else:
                    # Replace entire timeline
                    # Use slice assignment to replace contents while preserving the list object
                    # This is critical for undo system compatibility
                    actions_list_ref[:] = final_actions
                    
                    final_count = len(final_actions)
                    self.logger.info(f"Ultimate Autotune pipeline completed on {current_axis} axis. "
                                   f"Points: {initial_count} -> {final_count}")
                
                # Invalidate cache
                funscript_obj._invalidate_cache(current_axis)
            
            return funscript_obj
            
        except Exception as e:
            self.logger.error(f"Ultimate Autotune pipeline failed: {str(e)}")
            return None
    
    def _apply_custom_speed_limiter(self, funscript_obj, axis: str, speed_threshold: float):
        """Apply custom speed limiting to remove high-speed points."""
        actions = (funscript_obj.primary_actions if axis == 'primary' 
                  else funscript_obj.secondary_actions)
        
        if len(actions) <= 2:
            return
        
        actions_to_keep = [actions[0]]  # Always keep the first point
        
        for i in range(1, len(actions) - 1):
            p_prev, p_curr, p_next = actions[i - 1], actions[i], actions[i + 1]
            
            # Calculate in-speed
            in_dt = p_curr['at'] - p_prev['at']
            in_speed = abs(p_curr['pos'] - p_prev['pos']) / (in_dt / 1000.0) if in_dt > 0 else float('inf')
            
            # Calculate out-speed
            out_dt = p_next['at'] - p_curr['at']
            out_speed = abs(p_next['pos'] - p_curr['pos']) / (out_dt / 1000.0) if out_dt > 0 else float('inf')
            
            # Keep point if either speed is below threshold
            if not (in_speed > speed_threshold and out_speed > speed_threshold):
                actions_to_keep.append(p_curr)
        
        actions_to_keep.append(actions[-1])  # Always keep the last point
        
        # Update the actions list
        if axis == 'primary':
            funscript_obj.primary_actions[:] = actions_to_keep
        else:
            funscript_obj.secondary_actions[:] = actions_to_keep
    


# Register the plugin
def register_plugin():
    """Register this plugin with the plugin system."""
    from funscript.plugins.base_plugin import plugin_registry
    plugin_registry.register(UltimateAutotunePlugin())


# Auto-register when imported
register_plugin()