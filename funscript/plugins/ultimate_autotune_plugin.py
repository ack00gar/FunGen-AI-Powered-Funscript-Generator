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


class UltimateAutotunePlugin(FunscriptTransformationPlugin):
    """
    Ultimate Autotune Plugin - Multi-stage funscript enhancement pipeline.
    
    This plugin applies a sophisticated 7-stage processing pipeline:
    1. High-speed point removal (custom speed limiter)
    2. Peak-preserving resample (50ms)
    3. Savitzky-Golay smoothing (window=11, order=7)
    4. Peak-preserving resample (50ms)
    5. Amplification (scale=1.25, center=50)
    6. Peak-preserving resample (50ms)
    7. Keyframe simplification (tolerance=10, time=50ms)
    """
    
    @property
    def name(self) -> str:
        return "Ultimate Autotune"
    
    @property
    def description(self) -> str:
        return "Comprehensive 7-stage enhancement pipeline for optimal funscript quality"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def author(self) -> str:
        return "FunGen Team"
    
    @property
    def ui_preference(self) -> str:
        """Ultimate Autotune is designed to work perfectly with defaults."""
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
                
                # Create a temporary funscript object to work with
                temp_fs = funscript_obj.__class__(logger=self.logger)
                if current_axis == 'primary':
                    temp_fs.primary_actions = copy.deepcopy(actions_list_ref)
                else:
                    temp_fs.secondary_actions = copy.deepcopy(actions_list_ref)
                
                # === STEP 1: Custom Speed Limiter (Remove high-speed points) ===
                self.logger.debug(f"Ultimate Autotune ({current_axis}): (1) Removing high-speed points")
                self._apply_custom_speed_limiter(temp_fs, current_axis, params["speed_threshold"])
                
                # === STEP 2: Resample ===
                self.logger.debug(f"Ultimate Autotune ({current_axis}): (2) First resampling")
                temp_fs.apply_peak_preserving_resample(current_axis, resample_rate_ms=params["resample_rate_ms"])
                
                # === STEP 3: Smooth SG ===
                self.logger.debug(f"Ultimate Autotune ({current_axis}): (3) Applying Savitzky-Golay filter")
                temp_fs.apply_savitzky_golay(current_axis, 
                                           window_length=params["sg_window_length"], 
                                           polyorder=params["sg_polyorder"])
                
                # === STEP 4: Resample ===
                self.logger.debug(f"Ultimate Autotune ({current_axis}): (4) Second resampling")
                temp_fs.apply_peak_preserving_resample(current_axis, resample_rate_ms=params["resample_rate_ms"])
                
                # === STEP 5: Amplify ===
                self.logger.debug(f"Ultimate Autotune ({current_axis}): (5) Amplifying values")
                temp_fs.amplify_points_values(current_axis, 
                                            scale_factor=params["amplify_scale"], 
                                            center_value=params["amplify_center"])
                
                # === STEP 6: Resample ===
                self.logger.debug(f"Ultimate Autotune ({current_axis}): (6) Third resampling")
                temp_fs.apply_peak_preserving_resample(current_axis, resample_rate_ms=params["resample_rate_ms"])
                
                # === STEP 7: Keyframes ===
                self.logger.debug(f"Ultimate Autotune ({current_axis}): (7) Simplifying to keyframes")
                temp_fs.simplify_to_keyframes(current_axis, 
                                            position_tolerance=params["keyframe_position_tolerance"], 
                                            time_tolerance_ms=params["keyframe_time_tolerance_ms"])
                
                # Get the final processed actions
                final_actions = (temp_fs.primary_actions if current_axis == 'primary' 
                               else temp_fs.secondary_actions)
                
                # Use slice assignment to replace contents while preserving the list object
                # This is critical for undo system compatibility
                actions_list_ref[:] = final_actions
                
                # Invalidate cache
                funscript_obj._invalidate_cache(current_axis)
                
                final_count = len(final_actions)
                self.logger.info(f"Ultimate Autotune pipeline completed on {current_axis} axis. "
                               f"Points: {initial_count} -> {final_count}")
            
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