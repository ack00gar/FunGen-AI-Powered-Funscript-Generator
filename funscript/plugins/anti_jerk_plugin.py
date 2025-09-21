#!/usr/bin/env python3
"""
Anti-Jerk Filter Plugin

Removes intermediate points that create jerkiness in main movements.
Targets specific patterns where small oscillations interrupt smooth motion.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any

try:
    from .base_plugin import FunscriptTransformationPlugin
except ImportError:
    from funscript.plugins.base_plugin import FunscriptTransformationPlugin


class AntiJerkPlugin(FunscriptTransformationPlugin):
    """
    Plugin that removes jerky intermediate points that interrupt main movements.
    
    Detects 4-point sequences where intermediate points deviate from the direct path
    and create unnecessary jerkiness. Removes these intermediates to create smoother motion.
    """
    
    @property
    def name(self) -> str:
        return "Anti-Jerk"
    
    @property
    def description(self) -> str:
        return "Removes intermediate points that create jerkiness in main movements"
    
    @property
    def version(self) -> str:
        return "2.0.0"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            'jerk_threshold': {
                'type': float,
                'required': False,
                'default': 20.0,
                'description': 'Maximum oscillation size to consider as jerk',
                'constraints': {'min': 5.0, 'max': 40.0}
            },
            'min_main_movement': {
                'type': float,
                'required': False,
                'default': 50.0,
                'description': 'Minimum main movement to consider processing',
                'constraints': {'min': 20.0, 'max': 100.0}
            },
            'deviation_threshold': {
                'type': float,
                'required': False,
                'default': 15.0,
                'description': 'Maximum deviation from direct path to allow',
                'constraints': {'min': 5.0, 'max': 30.0}
            }
        }
    
    def check_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        try:
            import numpy as np
            return True
        except ImportError as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"Anti-Jerk plugin dependencies not available: {e}")
            return False
    
    def _remove_intermediate_jerks(self, actions: List[Dict[str, Any]], 
                                  jerk_threshold: float = 20.0, 
                                  min_main_movement: float = 50.0,
                                  deviation_threshold: float = 15.0) -> List[Dict[str, Any]]:
        """
        Remove intermediate points that create jerkiness.
        
        Pattern: A -> B -> C -> D where:
        - A to D is a significant movement (main motion)
        - B and C create small oscillations that interrupt the main flow
        - Remove B and C to get smooth A -> D transition
        
        Args:
            actions: List of funscript actions [{"at": timestamp, "pos": position}, ...]
            jerk_threshold: Maximum oscillation size to consider as "jerk"
            min_main_movement: Minimum A->D movement to consider processing
            deviation_threshold: Maximum deviation from direct path to allow
            
        Returns:
            Cleaned list of actions
        """
        if len(actions) < 4:
            return actions.copy()
        
        cleaned_actions = []
        i = 0
        
        while i < len(actions):
            # Always keep the current point
            cleaned_actions.append(actions[i])
            
            # Look for 4-point jerkiness pattern: current -> next1 -> next2 -> next3
            if i <= len(actions) - 4:
                current = actions[i]
                next1 = actions[i + 1]
                next2 = actions[i + 2] 
                next3 = actions[i + 3]
                
                # Calculate movements
                main_movement = abs(next3['pos'] - current['pos'])
                intermediate_osc = abs(next2['pos'] - next1['pos'])
                
                # Check if this fits the jerkiness pattern
                if (main_movement >= min_main_movement and 
                    intermediate_osc <= jerk_threshold):
                    
                    # Additional check: are the intermediates creating deviation from direct path?
                    # Calculate expected positions if moving directly from current to next3
                    time_span = next3['at'] - current['at']
                    if time_span > 0:
                        position_span = next3['pos'] - current['pos']
                        
                        # Expected positions at intermediate times
                        t1_ratio = (next1['at'] - current['at']) / time_span
                        t2_ratio = (next2['at'] - current['at']) / time_span
                        
                        expected_pos1 = current['pos'] + position_span * t1_ratio
                        expected_pos2 = current['pos'] + position_span * t2_ratio
                        
                        deviation1 = abs(next1['pos'] - expected_pos1)
                        deviation2 = abs(next2['pos'] - expected_pos2)
                        
                        # If intermediates deviate significantly from direct path, remove them
                        if deviation1 > deviation_threshold or deviation2 > deviation_threshold:
                            # Skip the intermediate points
                            i += 3  # Will add next3 in next iteration
                            continue
            
            i += 1
        
        return cleaned_actions
    
    def transform(self, funscript, axis: str = 'primary', **kwargs) -> Optional[str]:
        """
        Apply anti-jerk filtering to the funscript.
        
        Args:
            funscript: DualAxisFunscript object to transform
            axis: Which axis to transform ('primary', 'secondary', or 'both')
            **kwargs: Plugin parameters
            
        Returns:
            Error message if any, None if successful
        """
        try:
            # Extract parameters
            jerk_threshold = kwargs.get('jerk_threshold', 20.0)
            min_main_movement = kwargs.get('min_main_movement', 50.0)
            deviation_threshold = kwargs.get('deviation_threshold', 15.0)
            
            # Validate parameters
            jerk_threshold = max(5.0, min(40.0, jerk_threshold))
            min_main_movement = max(20.0, min(100.0, min_main_movement))
            deviation_threshold = max(5.0, min(30.0, deviation_threshold))
            
            axes_to_process = []
            if axis in ['primary', 'both']:
                axes_to_process.append('primary')
            if axis in ['secondary', 'both']:
                axes_to_process.append('secondary')
            
            for current_axis in axes_to_process:
                actions = getattr(funscript, f'{current_axis}_actions')
                if not actions or len(actions) < 4:
                    continue
                
                # Apply the targeted jerk filter
                cleaned_actions = self._remove_intermediate_jerks(
                    actions, jerk_threshold, min_main_movement, deviation_threshold
                )
                
                # Update the actions
                actions.clear()
                actions.extend(cleaned_actions)
            
            return None
            
        except Exception as e:
            error_msg = f"Anti-jerk filter failed: {str(e)}"
            if hasattr(self, 'logger'):
                self.logger.error(error_msg)
            return error_msg