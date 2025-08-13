"""
Stage 2 Signal Enhancer - Frame difference based signal validation and enhancement

This module provides frame difference computation to reinforce or tone down Stage 2 signals
by detecting motion inconsistencies and correcting unwanted strokes or adding missing ones.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Dict, List, Tuple, Any
from collections import deque


class Stage2SignalEnhancer:
    """
    Enhances Stage 2 signals using frame difference and motion analysis.
    
    Uses lightweight frame difference computation to:
    - Remove false strokes (Stage 2 signal changes without corresponding motion)
    - Add missing strokes (significant motion without Stage 2 signal change)
    - Reinforce weak signals during high motion periods
    - Smooth out temporal inconsistencies
    """
    
    def __init__(self, 
                 motion_threshold_low: float = 15.0,
                 motion_threshold_high: float = 35.0,
                 signal_change_threshold: int = 8,
                 history_window: int = 10,
                 enhancement_strength: float = 0.3,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the Stage 2 signal enhancer.
        
        Args:
            motion_threshold_low: Minimum motion magnitude to consider significant
            motion_threshold_high: High motion threshold for missing stroke detection
            signal_change_threshold: Minimum Stage 2 signal change to consider significant
            history_window: Number of frames to keep in motion history
            enhancement_strength: How much to adjust signals (0.0-1.0)
            logger: Optional logger instance
        """
        self.motion_threshold_low = motion_threshold_low
        self.motion_threshold_high = motion_threshold_high  
        self.signal_change_threshold = signal_change_threshold
        self.enhancement_strength = enhancement_strength
        self.logger = logger or logging.getLogger(__name__)
        
        # Motion history tracking
        self.motion_history: deque = deque(maxlen=history_window)
        self.signal_history: deque = deque(maxlen=history_window)
        self.enhanced_signal_history: deque = deque(maxlen=history_window)
        
        # Frame storage for difference computation
        self.prev_frame: Optional[np.ndarray] = None
        self.prev_roi_frame: Optional[np.ndarray] = None
        
        # Motion statistics for adaptive thresholding
        self.motion_stats = {
            'mean': 0.0,
            'std': 1.0,
            'max_recent': 0.0
        }
        
        # Enhancement state
        self.last_enhanced_signal: Optional[int] = None
        self.false_stroke_suppression_active: bool = False
        self.missing_stroke_boost_active: bool = False
        
        self.logger.info("Stage2SignalEnhancer initialized")
    
    def process_frame(self, 
                      frame: np.ndarray,
                      stage2_signal: int,
                      roi: Optional[Tuple[int, int, int, int]] = None) -> int:
        """
        Process a frame and enhance the Stage 2 signal based on motion analysis.
        
        Args:
            frame: Current video frame (BGR format)
            stage2_signal: Original Stage 2 signal (0-100 scale)
            roi: Optional ROI coordinates (x1, y1, x2, y2) to focus motion analysis
            
        Returns:
            Enhanced Stage 2 signal (0-100 scale)
        """
        # Convert to grayscale for motion analysis
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use ROI if provided, otherwise use full frame
        if roi:
            x1, y1, x2, y2 = roi
            current_roi = current_gray[y1:y2, x1:x2]
            analysis_frame = current_roi
        else:
            analysis_frame = current_gray
            current_roi = current_gray
        
        # Initialize on first frame
        if self.prev_frame is None:
            self.prev_frame = current_gray.copy()
            self.prev_roi_frame = current_roi.copy()
            self.signal_history.append(stage2_signal)
            self.enhanced_signal_history.append(stage2_signal)
            self.last_enhanced_signal = stage2_signal
            return stage2_signal
        
        # Calculate motion metrics
        motion_magnitude = self._calculate_motion_magnitude(analysis_frame, self.prev_roi_frame)
        motion_vector = self._calculate_motion_vector(analysis_frame, self.prev_roi_frame)
        
        # Update motion statistics
        self._update_motion_stats(motion_magnitude)
        
        # Store in history
        self.motion_history.append({
            'magnitude': motion_magnitude,
            'vector': motion_vector,
            'adaptive_threshold_low': self._get_adaptive_threshold_low(),
            'adaptive_threshold_high': self._get_adaptive_threshold_high()
        })
        self.signal_history.append(stage2_signal)
        
        # Enhance the signal
        enhanced_signal = self._enhance_signal(stage2_signal, motion_magnitude, motion_vector)
        
        # Store enhanced result
        self.enhanced_signal_history.append(enhanced_signal)
        self.last_enhanced_signal = enhanced_signal
        
        # Update frame storage
        self.prev_frame = current_gray.copy()
        self.prev_roi_frame = current_roi.copy()
        
        # Log significant enhancements
        if abs(enhanced_signal - stage2_signal) > 3:
            self.logger.debug(f"Signal enhanced: {stage2_signal} -> {enhanced_signal} "
                             f"(motion: {motion_magnitude:.2f})")
        
        return enhanced_signal
    
    def _calculate_motion_magnitude(self, current_frame: np.ndarray, prev_frame: np.ndarray) -> float:
        """Calculate motion magnitude using frame difference."""
        if current_frame.shape != prev_frame.shape:
            return 0.0
            
        # Frame difference
        frame_diff = cv2.absdiff(current_frame, prev_frame)
        
        # Calculate magnitude as RMS of differences
        motion_magnitude = np.sqrt(np.mean(frame_diff.astype(np.float32) ** 2))
        
        return motion_magnitude
    
    def _calculate_motion_vector(self, current_frame: np.ndarray, prev_frame: np.ndarray) -> Tuple[float, float]:
        """Calculate dominant motion vector using simple block matching."""
        if current_frame.shape != prev_frame.shape:
            return (0.0, 0.0)
        
        # Simple motion estimation using template matching
        h, w = current_frame.shape
        center_h, center_w = h // 2, w // 2
        
        # Take central region as template
        template_size = min(32, h // 4, w // 4)
        template = prev_frame[center_h - template_size:center_h + template_size,
                             center_w - template_size:center_w + template_size]
        
        if template.size == 0:
            return (0.0, 0.0)
        
        # Search in current frame
        search_range = min(16, h // 8, w // 8)
        search_area = current_frame[center_h - template_size - search_range:center_h + template_size + search_range,
                                  center_w - template_size - search_range:center_w + template_size + search_range]
        
        if search_area.size == 0:
            return (0.0, 0.0)
        
        # Template matching
        result = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)
        
        # Calculate motion vector
        dx = max_loc[0] - search_range
        dy = max_loc[1] - search_range
        
        return (float(dx), float(dy))
    
    def _update_motion_stats(self, motion_magnitude: float):
        """Update motion statistics for adaptive thresholding."""
        if len(self.motion_history) > 0:
            recent_magnitudes = [m['magnitude'] for m in self.motion_history]
            self.motion_stats['mean'] = np.mean(recent_magnitudes)
            self.motion_stats['std'] = max(1.0, np.std(recent_magnitudes))
            self.motion_stats['max_recent'] = max(recent_magnitudes[-5:]) if len(recent_magnitudes) >= 5 else motion_magnitude
    
    def _get_adaptive_threshold_low(self) -> float:
        """Get adaptive low motion threshold."""
        return max(self.motion_threshold_low, 
                   self.motion_stats['mean'] - 0.5 * self.motion_stats['std'])
    
    def _get_adaptive_threshold_high(self) -> float:
        """Get adaptive high motion threshold."""
        return max(self.motion_threshold_high,
                   self.motion_stats['mean'] + 1.5 * self.motion_stats['std'])
    
    def _enhance_signal(self, stage2_signal: int, motion_magnitude: float, motion_vector: Tuple[float, float]) -> int:
        """
        Apply signal enhancement based on motion analysis.
        
        Args:
            stage2_signal: Original Stage 2 signal
            motion_magnitude: Current motion magnitude
            motion_vector: Motion vector (dx, dy)
            
        Returns:
            Enhanced signal value
        """
        if len(self.signal_history) < 2:
            return stage2_signal
        
        # Calculate signal change from previous frame
        prev_signal = self.signal_history[-2]
        signal_change = abs(stage2_signal - prev_signal)
        
        # Get adaptive thresholds
        motion_low = self._get_adaptive_threshold_low()
        motion_high = self._get_adaptive_threshold_high()
        
        enhanced_signal = stage2_signal
        
        # Enhancement logic
        
        # 1. False stroke suppression: Signal changed but no significant motion
        if (signal_change > self.signal_change_threshold and 
            motion_magnitude < motion_low):
            
            # Suppress the signal change
            suppression_factor = 1.0 - self.enhancement_strength
            enhanced_signal = int(prev_signal + (stage2_signal - prev_signal) * suppression_factor)
            
            self.false_stroke_suppression_active = True
            self.logger.debug(f"False stroke suppressed: motion={motion_magnitude:.2f} < {motion_low:.2f}")
        
        # 2. Missing stroke detection: High motion but signal didn't change much
        elif (motion_magnitude > motion_high and 
              signal_change < self.signal_change_threshold):
            
            # Add motion-based signal change
            motion_direction = 1 if motion_vector[1] > 0 else -1  # Use vertical motion
            motion_strength = min(motion_magnitude / motion_high, 2.0)  # Cap at 2x
            
            boost_amount = int(self.enhancement_strength * motion_strength * 20)  # Max boost of ~12
            enhanced_signal = np.clip(stage2_signal + motion_direction * boost_amount, 0, 100)
            
            self.missing_stroke_boost_active = True
            self.logger.debug(f"Missing stroke boost: motion={motion_magnitude:.2f} > {motion_high:.2f}, boost={boost_amount}")
        
        # 3. Signal reinforcement: Both signal and motion agree, amplify slightly
        elif (signal_change > self.signal_change_threshold and 
              motion_magnitude > motion_low):
            
            # Reinforce the signal change
            reinforcement_factor = 1.0 + (self.enhancement_strength * 0.5)  # Max 15% boost
            signal_delta = stage2_signal - prev_signal
            enhanced_delta = int(signal_delta * reinforcement_factor)
            enhanced_signal = np.clip(prev_signal + enhanced_delta, 0, 100)
            
            self.logger.debug(f"Signal reinforced: factor={reinforcement_factor:.2f}")
        
        # 4. Temporal smoothing for rapid oscillations
        if len(self.enhanced_signal_history) >= 3:
            recent_enhanced = list(self.enhanced_signal_history)[-3:]
            if self._is_rapid_oscillation(recent_enhanced):
                # Apply light smoothing
                smoothing_factor = 0.7
                enhanced_signal = int(enhanced_signal * smoothing_factor + 
                                    recent_enhanced[-2] * (1 - smoothing_factor))
                
                self.logger.debug("Applied temporal smoothing for rapid oscillation")
        
        # Reset enhancement flags
        if signal_change <= self.signal_change_threshold:
            self.false_stroke_suppression_active = False
            self.missing_stroke_boost_active = False
        
        return int(np.clip(enhanced_signal, 0, 100))
    
    def _is_rapid_oscillation(self, signal_sequence: List[int]) -> bool:
        """Detect rapid signal oscillations that might need smoothing."""
        if len(signal_sequence) < 3:
            return False
        
        # Check if signal is bouncing back and forth rapidly
        changes = [abs(signal_sequence[i] - signal_sequence[i-1]) for i in range(1, len(signal_sequence))]
        
        # Rapid oscillation if multiple large changes in short sequence
        large_changes = sum(1 for c in changes if c > self.signal_change_threshold)
        
        return large_changes >= 2
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about current enhancement state."""
        current_motion = self.motion_history[-1] if self.motion_history else {}
        
        return {
            'motion_magnitude': current_motion.get('magnitude', 0.0),
            'motion_vector': current_motion.get('vector', (0.0, 0.0)),
            'motion_threshold_low': self._get_adaptive_threshold_low(),
            'motion_threshold_high': self._get_adaptive_threshold_high(),
            'false_stroke_suppression_active': self.false_stroke_suppression_active,
            'missing_stroke_boost_active': self.missing_stroke_boost_active,
            'motion_stats': self.motion_stats.copy(),
            'history_length': len(self.motion_history),
            'last_enhanced_signal': self.last_enhanced_signal
        }
    
    def reset(self):
        """Reset the enhancer state for a new video or processing session."""
        self.motion_history.clear()
        self.signal_history.clear()
        self.enhanced_signal_history.clear()
        self.prev_frame = None
        self.prev_roi_frame = None
        self.motion_stats = {'mean': 0.0, 'std': 1.0, 'max_recent': 0.0}
        self.last_enhanced_signal = None
        self.false_stroke_suppression_active = False
        self.missing_stroke_boost_active = False
        
        self.logger.info("Stage2SignalEnhancer reset")