"""
Signal Amplifier Helper Module

Provides enhanced signal amplification and mastering techniques for tracker modules.
Based on proven algorithms from Oscillation Detector Experimental 2.

This module offers:
- Amplitude-aware scaling with natural response
- Live dynamic amplification with anti-plateau normalization
- Motion magnitude analysis
- Configurable sensitivity and history management
"""

import numpy as np
from collections import deque
from typing import Tuple, Optional, Dict, Any
import logging


class SignalAmplifier:
    """
    Signal amplification and mastering for tracker position data.
    
    This class provides advanced signal processing techniques to enhance
    tracker output, preventing under-amplification and signal plateaus.
    """
    
    def __init__(self, 
                 history_size: int = 120,
                 enable_live_amp: bool = True,
                 smoothing_alpha: float = 0.3,
                 amplitude_divisor: float = 4.0,
                 base_scale_primary: float = -10.0,
                 base_scale_secondary: float = 10.0,
                 max_deviation_multiplier: float = 49.0,
                 plateau_threshold: float = 15.0,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the signal amplifier.
        
        Args:
            history_size: Size of position history for dynamic amplification (frames)
            enable_live_amp: Enable live dynamic amplification
            smoothing_alpha: EMA smoothing factor (0=max smooth, 1=no smooth)
            amplitude_divisor: Divisor for amplitude-aware scaling
            base_scale_primary: Base scaling factor for primary axis
            base_scale_secondary: Base scaling factor for secondary axis
            max_deviation_multiplier: Multiplier for max deviation calculation
            plateau_threshold: Minimum range for anti-plateau activation
            logger: Optional logger instance
        """
        # Core configuration
        self.history_size = history_size
        self.live_amp_enabled = enable_live_amp
        self.smoothing_alpha = smoothing_alpha
        
        # Scaling parameters
        self.amplitude_divisor = amplitude_divisor
        self.base_scale_primary = base_scale_primary
        self.base_scale_secondary = base_scale_secondary
        self.max_deviation_multiplier = max_deviation_multiplier
        self.plateau_threshold = plateau_threshold
        
        # Position history for dynamic amplification
        self.position_history_primary = deque(maxlen=history_size)
        self.position_history_secondary = deque(maxlen=history_size)
        
        # EMA state
        self.last_primary_position = 50
        self.last_secondary_position = 50
        
        # Logger
        self.logger = logger or logging.getLogger(__name__)
    
    def enhance_signal(self,
                      raw_primary: int,
                      raw_secondary: int,
                      dy_flow: float,
                      dx_flow: float,
                      sensitivity: float = 10.0,
                      apply_smoothing: bool = True) -> Tuple[int, int]:
        """
        Apply enhanced signal amplification to raw tracker positions.
        
        This method implements the full signal processing pipeline:
        1. Motion magnitude calculation for amplitude-aware scaling
        2. Enhanced base scaling with larger factors
        3. Sensitivity-aware max deviation
        4. Live dynamic amplification with percentile normalization
        5. Optional EMA smoothing
        
        Args:
            raw_primary: Raw primary axis position (0-100)
            raw_secondary: Raw secondary axis position (0-100)
            dy_flow: Vertical optical flow component
            dx_flow: Horizontal optical flow component
            sensitivity: Sensitivity setting (typically 1-20, default 10)
            apply_smoothing: Whether to apply EMA smoothing
            
        Returns:
            Tuple of (enhanced_primary, enhanced_secondary) positions (0-100)
        """
        try:
            # Calculate motion magnitude for amplitude-aware scaling
            motion_magnitude = np.sqrt(dy_flow**2 + dx_flow**2)
            
            # Amplitude-aware scaling with natural response
            # Scales between 0.7x and 1.5x based on motion intensity
            amplitude_scaler = np.clip(motion_magnitude / self.amplitude_divisor, 0.7, 1.5)
            
            # Calculate max deviation with sensitivity and amplitude scaling
            sensitivity_factor = sensitivity / 10.0  # Normalize to 1.0 at default
            max_deviation = self.max_deviation_multiplier * sensitivity_factor * amplitude_scaler
            
            # Apply enhanced scaling with larger base factors
            # These larger factors provide much stronger signal amplitude
            enhanced_dy = np.clip(dy_flow * self.base_scale_primary, -max_deviation, max_deviation)
            enhanced_dx = np.clip(dx_flow * self.base_scale_secondary, -max_deviation, max_deviation)
            
            # Calculate enhanced positions from center point (50)
            enhanced_primary_raw = np.clip(50 + enhanced_dy, 0, 100)
            enhanced_secondary_raw = np.clip(50 + enhanced_dx, 0, 100)
            
            # Apply live dynamic amplification
            final_primary = self._apply_dynamic_amplification(
                enhanced_primary_raw, self.position_history_primary, is_primary=True
            )
            final_secondary = self._apply_dynamic_amplification(
                enhanced_secondary_raw, self.position_history_secondary, is_primary=False
            )
            
            # Apply optional EMA smoothing
            if apply_smoothing:
                final_primary = self._apply_ema_smoothing(final_primary, is_primary=True)
                final_secondary = self._apply_ema_smoothing(final_secondary, is_primary=False)
            
            return int(final_primary), int(final_secondary)
            
        except Exception as e:
            self.logger.error(f"Signal enhancement error: {e}")
            # Fallback to original values on error
            return raw_primary, raw_secondary
    
    def _apply_dynamic_amplification(self, 
                                    position: float, 
                                    history: deque,
                                    is_primary: bool = True) -> float:
        """
        Apply live dynamic amplification with anti-plateau normalization.
        
        This prevents signal plateaus by normalizing the position range
        based on recent history using percentile analysis.
        
        Args:
            position: Current position value (0-100)
            history: Position history deque
            is_primary: Whether this is the primary axis
            
        Returns:
            Dynamically amplified position
        """
        # Add to history
        history.append(position)
        
        # Skip if not enough history or live amp disabled
        if not self.live_amp_enabled or len(history) < self.history_size * 0.5:
            return position
        
        # Calculate percentiles for normalization
        p10 = np.percentile(history, 10)
        p90 = np.percentile(history, 90)
        effective_range = p90 - p10
        
        # Apply anti-plateau normalization if range is sufficient
        if effective_range > self.plateau_threshold:
            # Normalize position to full 0-100 range based on recent history
            normalized_pos = (position - p10) / effective_range
            amplified_position = np.clip(normalized_pos * 100, 0, 100)
            
            # For secondary axis, apply slightly different normalization
            if not is_primary:
                # Center-biased normalization for secondary
                normalized_secondary = (position - 50) / max(1, effective_range) + 0.5
                amplified_position = np.clip(normalized_secondary * 100, 0, 100)
            
            return amplified_position
        
        return position
    
    def _apply_ema_smoothing(self, position: float, is_primary: bool = True) -> float:
        """
        Apply exponential moving average smoothing.
        
        Args:
            position: Current position
            is_primary: Whether this is the primary axis
            
        Returns:
            Smoothed position
        """
        if is_primary:
            smoothed = (self.smoothing_alpha * position + 
                       (1 - self.smoothing_alpha) * self.last_primary_position)
            self.last_primary_position = smoothed
        else:
            smoothed = (self.smoothing_alpha * position + 
                       (1 - self.smoothing_alpha) * self.last_secondary_position)
            self.last_secondary_position = smoothed
        
        return smoothed
    
    def reset(self):
        """Reset the amplifier state for a new tracking session."""
        self.position_history_primary.clear()
        self.position_history_secondary.clear()
        self.last_primary_position = 50
        self.last_secondary_position = 50
    
    def update_parameters(self, **kwargs):
        """
        Update amplifier parameters dynamically.
        
        Supported parameters:
        - enable_live_amp: bool
        - smoothing_alpha: float (0-1)
        - sensitivity_factor: float
        - plateau_threshold: float
        """
        for key, value in kwargs.items():
            if key == 'enable_live_amp' and isinstance(value, bool):
                self.live_amp_enabled = value
            elif key == 'smoothing_alpha' and 0 <= value <= 1:
                self.smoothing_alpha = value
            elif key == 'plateau_threshold' and value > 0:
                self.plateau_threshold = value
            else:
                self.logger.warning(f"Unknown or invalid parameter: {key}={value}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current amplifier statistics for debugging.
        
        Returns:
            Dictionary containing amplifier statistics
        """
        stats = {
            'live_amp_enabled': self.live_amp_enabled,
            'primary_history_size': len(self.position_history_primary),
            'secondary_history_size': len(self.position_history_secondary),
            'last_primary': self.last_primary_position,
            'last_secondary': self.last_secondary_position,
        }
        
        # Add percentile info if enough history
        if len(self.position_history_primary) > 10:
            stats['primary_p10'] = np.percentile(self.position_history_primary, 10)
            stats['primary_p90'] = np.percentile(self.position_history_primary, 90)
            stats['primary_range'] = stats['primary_p90'] - stats['primary_p10']
        
        if len(self.position_history_secondary) > 10:
            stats['secondary_p10'] = np.percentile(self.position_history_secondary, 10)
            stats['secondary_p90'] = np.percentile(self.position_history_secondary, 90)
            stats['secondary_range'] = stats['secondary_p90'] - stats['secondary_p10']
        
        return stats
    
    def create_lightweight_copy(self) -> 'SignalAmplifier':
        """
        Create a lightweight copy with shared configuration but separate state.
        
        Useful for creating multiple independent amplifiers with same settings.
        
        Returns:
            New SignalAmplifier instance with same configuration
        """
        return SignalAmplifier(
            history_size=self.history_size,
            enable_live_amp=self.live_amp_enabled,
            smoothing_alpha=self.smoothing_alpha,
            amplitude_divisor=self.amplitude_divisor,
            base_scale_primary=self.base_scale_primary,
            base_scale_secondary=self.base_scale_secondary,
            max_deviation_multiplier=self.max_deviation_multiplier,
            plateau_threshold=self.plateau_threshold,
            logger=self.logger
        )