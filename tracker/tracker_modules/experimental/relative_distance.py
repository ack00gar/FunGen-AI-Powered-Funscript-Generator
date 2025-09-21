#!/usr/bin/env python3
"""
Relative Distance Tracker - Enhanced high-performance tracker with Stage 2 insights.

An optimized high-performance tracker with vectorized operations and intelligent caching.
This tracker focuses on calculating relative distances between detected objects (especially 
contact regions and penis) to generate accurate funscript signals.

Features:
- Stage 2-inspired locked penis tracking with IoU continuity
- Enhanced distance calculation using optimal reference points  
- Vectorized operations for maximum performance
- Intelligent caching to reduce computational overhead
- Advanced contact detection and tracking
- Optimized UX overlays with better visualization

Author: VR Funscript AI Generator
Version: 2.0.0 (Rebuilt with Stage 2 enhancements)
"""

import logging
import time
import numpy as np
import cv2
from collections import deque
from typing import Dict, Any, Optional, List, Tuple
import threading
from concurrent.futures import ThreadPoolExecutor
import traceback
from config.constants_colors import RGBColors

try:
    from ..core.base_tracker import BaseTracker, TrackerMetadata, TrackerResult
    from ..helpers.signal_amplifier import SignalAmplifier
    from ..helpers.visualization import TrackerVisualizationHelper, BoundingBox
except ImportError:
    from tracker.tracker_modules.core.base_tracker import BaseTracker, TrackerMetadata, TrackerResult
    from tracker.tracker_modules.helpers.signal_amplifier import SignalAmplifier
    from tracker.tracker_modules.helpers.visualization import TrackerVisualizationHelper, BoundingBox

import config.constants as constants


class LockedPenisTracker:
    """Persistent locked penis tracking with 95th percentile height calculation."""
    
    def __init__(self, fps: float = 30.0):
        self.box = None
        self.conceptual_box = None
        self.active = False
        self.confidence = 0.0
        self.center = None
        self.last_seen_timestamp = 0.0
        self.unseen_frames = 0
        self.detection_frames = deque(maxlen=30)
        self.established_frame = None
        
        # 95th percentile height tracking over 30-second rolling window
        self.fps = fps
        self.height_window_seconds = 30.0
        self.height_history = deque(maxlen=int(fps * self.height_window_seconds))  # 30 seconds at current FPS
        self.computed_height_95th = 0.0
        self.last_height_update = 0.0
        self.height_update_interval = 1.0  # Update every second
        
        # Enhanced persistence - locked penis should STAY locked
        self.max_height = 0.0
        self.max_penetration_height = 0.0
        self.last_raw_coords = None
        self.persistent_lock = True  # Once locked, stay locked unless manually reset
        
        # Persistence parameters
        self.iou_threshold = 0.3
        self.patience = int(fps * 30)  # 30 seconds persistence at current FPS
        self.min_detections_activate = 3
        self.min_detections_deactivate = 1


class VelocityPredictor:
    """Velocity-based predictive tracking for seamless interpolation during detection gaps."""
    
    def __init__(self, max_history: int = 10):
        self.position_history = deque(maxlen=max_history)  # Store (timestamp, center_x, center_y, box)
        self.velocity_history = deque(maxlen=5)  # Store calculated velocities
        self.last_prediction_timestamp = 0.0
        self.max_prediction_duration = 3.0  # Max 3 seconds of prediction
        self.min_history_for_prediction = 3
        
    def update_position(self, timestamp: float, box: Tuple[float, float, float, float]):
        """Update position history with new detection."""
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        
        self.position_history.append((timestamp, center_x, center_y, box))
        
        # Calculate velocity if we have enough history
        if len(self.position_history) >= 2:
            self._update_velocity()
    
    def _update_velocity(self):
        """Calculate current velocity from position history."""
        if len(self.position_history) < 2:
            return
            
        # Use last 3 positions for velocity calculation
        recent_positions = list(self.position_history)[-3:]
        
        velocities_x = []
        velocities_y = []
        
        for i in range(1, len(recent_positions)):
            dt = recent_positions[i][0] - recent_positions[i-1][0]
            if dt > 0:
                vx = (recent_positions[i][1] - recent_positions[i-1][1]) / dt
                vy = (recent_positions[i][2] - recent_positions[i-1][2]) / dt
                velocities_x.append(vx)
                velocities_y.append(vy)
        
        if velocities_x and velocities_y:
            # Weighted average - recent velocities have more weight
            weights = [0.3, 0.7] if len(velocities_x) == 2 else [0.2, 0.3, 0.5]
            avg_vx = np.average(velocities_x, weights=weights[:len(velocities_x)])
            avg_vy = np.average(velocities_y, weights=weights[:len(velocities_y)])
            
            current_time = self.position_history[-1][0]
            self.velocity_history.append((current_time, avg_vx, avg_vy))
    
    def predict_position(self, target_timestamp: float) -> Optional[Tuple[float, float, float, float]]:
        """Predict box position at target timestamp using velocity extrapolation."""
        if not self.position_history or not self.velocity_history:
            return None
            
        if len(self.position_history) < self.min_history_for_prediction:
            return None
        
        last_position = self.position_history[-1]
        last_velocity = self.velocity_history[-1]
        
        # Check if prediction is within reasonable time bounds
        time_delta = target_timestamp - last_position[0]
        if time_delta < 0 or time_delta > self.max_prediction_duration:
            return None
        
        # Extrapolate position using velocity
        predicted_x = last_position[1] + last_velocity[1] * time_delta
        predicted_y = last_position[2] + last_velocity[2] * time_delta
        
        # Extrapolate box dimensions (assuming constant size)
        last_box = last_position[3]
        box_width = last_box[2] - last_box[0]
        box_height = last_box[3] - last_box[1]
        
        predicted_box = (
            predicted_x - box_width/2,
            predicted_y - box_height/2, 
            predicted_x + box_width/2,
            predicted_y + box_height/2
        )
        
        return predicted_box
    
    def get_velocity_magnitude(self) -> float:
        """Get current velocity magnitude for motion analysis."""
        if not self.velocity_history:
            return 0.0
            
        last_velocity = self.velocity_history[-1]
        return np.sqrt(last_velocity[1]**2 + last_velocity[2]**2)
    
    def is_prediction_reliable(self, target_timestamp: float) -> bool:
        """Check if prediction at target timestamp would be reliable."""
        if not self.position_history:
            return False
            
        last_timestamp = self.position_history[-1][0]
        time_delta = target_timestamp - last_timestamp
        
        # Reliable if: good history, reasonable time gap, consistent velocity
        has_good_history = len(self.position_history) >= self.min_history_for_prediction
        reasonable_gap = 0 <= time_delta <= 2.0  # 2 seconds max for high reliability
        
        if not (has_good_history and reasonable_gap):
            return False
        
        # Check velocity consistency (low variance indicates predictable motion)
        if len(self.velocity_history) >= 3:
            recent_vels = list(self.velocity_history)[-3:]
            vx_vals = [v[1] for v in recent_vels]
            vy_vals = [v[2] for v in recent_vels] 
            
            vx_var = np.var(vx_vals)
            vy_var = np.var(vy_vals)
            
            # Consider reliable if velocity is consistent (low variance)
            return vx_var < 50 and vy_var < 50
        
        return True


class MultiScaleTemporalAnalyzer:
    """Multi-scale temporal analysis for different motion patterns."""
    
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        
        # Short-term analysis (1-3 seconds) - rapid movements, stroking
        self.short_term_window = int(fps * 3)
        self.short_term_positions = deque(maxlen=self.short_term_window)
        self.short_term_velocities = deque(maxlen=self.short_term_window)
        
        # Medium-term analysis (5-10 seconds) - position changes, tempo shifts  
        self.medium_term_window = int(fps * 10)
        self.medium_term_positions = deque(maxlen=self.medium_term_window)
        self.medium_term_patterns = deque(maxlen=20)  # Pattern detection
        
        # Long-term analysis (30+ seconds) - height calibration, scene analysis
        self.long_term_window = int(fps * 60)  # 60 seconds
        self.long_term_stats = deque(maxlen=self.long_term_window)
        self.scene_context = None
        
        # Pattern recognition state
        self.current_stroke_direction = None
        self.stroke_cycle_positions = []
        self.last_peak_time = 0.0
        self.last_valley_time = 0.0
        
    def update(self, timestamp: float, position: float, velocity: float):
        """Update all temporal scales with new data."""
        self.short_term_positions.append((timestamp, position))
        self.short_term_velocities.append((timestamp, velocity))
        
        # Downsample for medium-term (every 3rd frame)
        if len(self.short_term_positions) % 3 == 0:
            self.medium_term_positions.append((timestamp, position))
        
        # Downsample for long-term (every 30th frame) 
        if len(self.short_term_positions) % 30 == 0:
            stats = self._calculate_current_stats()
            self.long_term_stats.append((timestamp, stats))
    
    def _calculate_current_stats(self) -> Dict:
        """Calculate statistical measures for long-term analysis."""
        if not self.short_term_positions:
            return {}
            
        positions = [p[1] for p in self.short_term_positions]
        return {
            'mean_position': np.mean(positions),
            'position_std': np.std(positions),
            'position_range': np.max(positions) - np.min(positions),
            'activity_level': self._calculate_activity_level()
        }
    
    def _calculate_activity_level(self) -> float:
        """Calculate current activity level based on velocity variance."""
        if len(self.short_term_velocities) < 10:
            return 0.0
            
        velocities = [v[1] for v in self.short_term_velocities]
        return np.std(velocities)
    
    def detect_motion_pattern(self) -> Dict[str, Any]:
        """Detect current motion patterns across all time scales."""
        pattern_info = {
            'stroke_detected': False,
            'stroke_frequency': 0.0,
            'motion_type': 'unknown',
            'tempo_stable': True,
            'activity_level': 'low'
        }
        
        if len(self.short_term_positions) < 20:
            return pattern_info
        
        # Short-term pattern detection
        short_analysis = self._analyze_short_term_motion()
        pattern_info.update(short_analysis)
        
        # Medium-term tempo analysis
        if len(self.medium_term_positions) >= 20:
            medium_analysis = self._analyze_medium_term_patterns()
            pattern_info.update(medium_analysis)
        
        return pattern_info
    
    def _analyze_short_term_motion(self) -> Dict[str, Any]:
        """Analyze short-term motion for stroke detection."""
        positions = [p[1] for p in list(self.short_term_positions)[-30:]]  # Last 1 second
        
        if len(positions) < 10:
            return {'stroke_detected': False}
        
        # Find peaks and valleys
        peaks = []
        valleys = []
        
        for i in range(1, len(positions) - 1):
            if positions[i] > positions[i-1] and positions[i] > positions[i+1]:
                peaks.append((i, positions[i]))
            elif positions[i] < positions[i-1] and positions[i] < positions[i+1]:
                valleys.append((i, positions[i]))
        
        # Stroke detection logic
        stroke_detected = len(peaks) >= 1 and len(valleys) >= 1
        motion_range = np.max(positions) - np.min(positions)
        motion_type = 'stroking' if stroke_detected and motion_range > 20 else 'subtle'
        
        return {
            'stroke_detected': stroke_detected,
            'motion_type': motion_type,
            'motion_range': motion_range,
            'peaks_count': len(peaks),
            'valleys_count': len(valleys)
        }
    
    def _analyze_medium_term_patterns(self) -> Dict[str, Any]:
        """Analyze medium-term patterns for tempo and stability."""
        positions = [p[1] for p in list(self.medium_term_positions)[-60:]]  # Last ~6 seconds
        
        if len(positions) < 30:
            return {'tempo_stable': True}
        
        # Calculate tempo stability by looking at variance in stroke intervals
        recent_range = np.max(positions[-30:]) - np.min(positions[-30:])  # Last 3 seconds
        earlier_range = np.max(positions[:30]) - np.min(positions[:30])   # Earlier 3 seconds
        
        tempo_stable = abs(recent_range - earlier_range) < 10  # Stable if range change < 10
        
        # Estimate frequency from zero crossings
        mean_pos = np.mean(positions)
        crossings = 0
        for i in range(1, len(positions)):
            if (positions[i-1] < mean_pos) != (positions[i] < mean_pos):
                crossings += 1
        
        frequency = (crossings / 2) / (len(positions) / self.fps)  # Cycles per second
        
        return {
            'tempo_stable': tempo_stable,
            'stroke_frequency': frequency,
            'tempo_change': abs(recent_range - earlier_range)
        }


class AdvancedSignalFusion:
    """Advanced confidence-weighted signal fusion with temporal consistency and quality assessment."""
    
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.signal_history = deque(maxlen=int(fps * 5))  # 5 seconds of signal history
        self.confidence_history = deque(maxlen=int(fps * 3))  # 3 seconds of confidence history
        self.quality_metrics = {}
        
        # Fusion parameters
        self.min_confidence_threshold = 0.3
        self.temporal_consistency_weight = 0.4
        self.spatial_coherence_weight = 0.3
        self.detection_quality_weight = 0.3
        
    def fuse_signals(self, primary_signal: float, primary_confidence: float, primary_class: str,
                     support_signals: List[Dict], locked_penis_box: Dict) -> Dict[str, Any]:
        """Advanced signal fusion with comprehensive quality assessment."""
        fusion_result = {
            'fused_signal': primary_signal,
            'fusion_confidence': primary_confidence,
            'fusion_quality': 'high',
            'contributors': [primary_class],
            'fusion_weights': {primary_class: 1.0},
            'quality_metrics': {}
        }
        
        if not support_signals:
            return fusion_result
        
        # Calculate quality metrics for all signals
        primary_quality = self._assess_signal_quality(
            primary_signal, primary_confidence, primary_class, locked_penis_box
        )
        
        valid_supports = []
        for support in support_signals:
            if support['confidence'] >= self.min_confidence_threshold:
                support_quality = self._assess_signal_quality(
                    support.get('signal', 0.0), support['confidence'], 
                    support['class_name'], locked_penis_box, support['bbox']
                )
                
                valid_supports.append({
                    **support,
                    'quality': support_quality
                })
        
        if not valid_supports:
            fusion_result['quality_metrics']['primary_quality'] = primary_quality
            return fusion_result
        
        # Advanced weighted fusion
        fusion_weights = self._calculate_fusion_weights(
            primary_quality, primary_confidence, primary_class,
            valid_supports, locked_penis_box
        )
        
        # Apply fusion weights
        weighted_sum = primary_signal * fusion_weights['primary']
        total_weight = fusion_weights['primary']
        contributors = [primary_class]
        
        for i, support in enumerate(valid_supports):
            support_weight = fusion_weights.get(f'support_{i}', 0.0)
            if support_weight > 0:
                weighted_sum += support.get('signal', 0.0) * support_weight
                total_weight += support_weight
                contributors.append(support['class_name'])
        
        # Normalize and apply temporal smoothing
        if total_weight > 0:
            raw_fused_signal = weighted_sum / total_weight
            smoothed_signal = self._apply_temporal_smoothing(raw_fused_signal, primary_signal)
            
            fusion_result.update({
                'fused_signal': smoothed_signal,
                'fusion_confidence': min(1.0, primary_confidence * 1.2),  # Fusion boosts confidence
                'fusion_quality': self._determine_fusion_quality(fusion_weights, valid_supports),
                'contributors': contributors,
                'fusion_weights': fusion_weights,
                'quality_metrics': {
                    'primary_quality': primary_quality,
                    'support_qualities': [s['quality'] for s in valid_supports],
                    'temporal_consistency': self._calculate_temporal_consistency(smoothed_signal),
                    'spatial_coherence': self._calculate_spatial_coherence(contributors, locked_penis_box)
                }
            })
        
        return fusion_result
    
    def _assess_signal_quality(self, signal: float, confidence: float, class_name: str, 
                              locked_penis_box: Dict, contact_box: Optional[Tuple] = None) -> Dict[str, float]:
        """Assess the quality of a signal contribution."""
        quality = {
            'confidence_score': confidence,
            'temporal_consistency': 0.5,  # Will be updated with history
            'spatial_coherence': 0.5,     # Will be updated with spatial analysis
            'detection_quality': confidence,  # Basic detection quality
            'overall_quality': 0.0
        }
        
        # Temporal consistency - how stable is this signal over time
        if len(self.signal_history) >= 10:
            recent_signals = [h for h in list(self.signal_history)[-10:] if h.get('class_name') == class_name]
            if recent_signals:
                signal_variance = np.var([h['signal'] for h in recent_signals])
                quality['temporal_consistency'] = max(0.0, 1.0 - signal_variance / 100.0)  # Lower variance = higher consistency
        
        # Spatial coherence - how well positioned is this contact relative to penis
        if contact_box and locked_penis_box:
            distance_to_penis = self._calculate_box_distance(contact_box, locked_penis_box.get('bbox', (0,0,0,0)))
            # Closer contacts are more coherent (up to a reasonable limit)
            quality['spatial_coherence'] = max(0.0, 1.0 - distance_to_penis / 100.0)
        
        # Detection quality - based on confidence and class-specific factors
        class_reliability = {
            'pussy': 1.0, 'vagina': 1.0,    # Most reliable
            'hand': 0.9,                    # Very reliable
            'face': 0.8,                    # Quite reliable
            'butt': 0.8, 'anus': 0.8,      # Quite reliable
            'navel': 0.6,                   # Moderate reliability
            'breast': 0.5,                  # Lower reliability
            'foot': 0.4                     # Lowest reliability
        }
        
        base_reliability = class_reliability.get(class_name, 0.5)
        quality['detection_quality'] = confidence * base_reliability
        
        # Calculate overall quality
        quality['overall_quality'] = (
            quality['confidence_score'] * 0.3 +
            quality['temporal_consistency'] * 0.3 + 
            quality['spatial_coherence'] * 0.2 +
            quality['detection_quality'] * 0.2
        )
        
        return quality
    
    def _calculate_fusion_weights(self, primary_quality: Dict, primary_confidence: float, primary_class: str,
                                 support_signals: List[Dict], locked_penis_box: Dict) -> Dict[str, float]:
        """Calculate sophisticated fusion weights based on multiple quality factors."""
        weights = {'primary': 1.0}  # Primary always starts with weight 1.0
        
        # Base weights for different support classes when supporting pussy
        base_support_weights = {
            'navel': 0.4,    # Increased from 0.3 - navel is quite reliable for pussy support
            'breast': 0.3,   # Increased from 0.2 - breast can provide good support
            'hand': 0.2,     # Hand can interfere, so lower weight
            'face': 0.1      # Face rarely provides good pussy support
        }
        
        for i, support in enumerate(support_signals):
            class_name = support['class_name']
            base_weight = base_support_weights.get(class_name, 0.1)
            
            # Quality-based adjustment
            quality_factor = support['quality']['overall_quality']
            confidence_factor = support['confidence']
            
            # Temporal consistency bonus
            consistency_bonus = support['quality']['temporal_consistency'] * 0.2
            
            # Spatial coherence bonus  
            coherence_bonus = support['quality']['spatial_coherence'] * 0.2
            
            # Final weight calculation
            final_weight = base_weight * quality_factor * confidence_factor + consistency_bonus + coherence_bonus
            
            # Cap support weights to prevent overwhelming primary
            final_weight = min(final_weight, 0.6)  # Max 60% of primary weight
            
            weights[f'support_{i}'] = final_weight
        
        return weights
    
    def _apply_temporal_smoothing(self, new_signal: float, primary_signal: float) -> float:
        """Apply temporal smoothing to reduce signal jitter."""
        if not self.signal_history:
            return new_signal
        
        # Use last few signals for smoothing
        recent_signals = [h['signal'] for h in list(self.signal_history)[-5:]]
        
        if len(recent_signals) >= 3:
            # Weighted average with more weight on recent signals
            weights = [0.1, 0.2, 0.3, 0.4] if len(recent_signals) == 4 else [0.2, 0.3, 0.5]
            smoothed = np.average(recent_signals[-len(weights):] + [new_signal], 
                                weights=weights + [0.6])  # New signal gets highest weight
            return smoothed
        
        return new_signal
    
    def _determine_fusion_quality(self, weights: Dict, support_signals: List[Dict]) -> str:
        """Determine overall fusion quality."""
        total_support_weight = sum(w for k, w in weights.items() if k.startswith('support_'))
        
        if total_support_weight > 0.5:
            return 'high'  # Strong support signals
        elif total_support_weight > 0.2:
            return 'medium'  # Moderate support
        else:
            return 'low'   # Little to no support
    
    def _calculate_temporal_consistency(self, current_signal: float) -> float:
        """Calculate how consistent the current signal is with recent history."""
        if len(self.signal_history) < 5:
            return 0.5
        
        recent_signals = [h['signal'] for h in list(self.signal_history)[-5:]]
        variance = np.var(recent_signals + [current_signal])
        
        # Lower variance = higher consistency
        return max(0.0, 1.0 - variance / 200.0)
    
    def _calculate_spatial_coherence(self, contributors: List[str], locked_penis_box: Dict) -> float:
        """Calculate spatial coherence of contributing signals."""
        # This is a simplified version - could be enhanced with actual spatial analysis
        coherence_scores = {
            ('pussy', 'navel'): 0.9,    # Very coherent
            ('pussy', 'breast'): 0.7,   # Quite coherent  
            ('pussy', 'hand'): 0.6,     # Moderately coherent
            ('pussy', 'face'): 0.3,     # Less coherent
        }
        
        if len(contributors) <= 1:
            return 1.0
        
        # Calculate average coherence for all contributor pairs
        coherence_sum = 0
        pair_count = 0
        
        for i, c1 in enumerate(contributors):
            for c2 in contributors[i+1:]:
                pair = tuple(sorted([c1, c2]))
                coherence_sum += coherence_scores.get(pair, 0.5)
                pair_count += 1
        
        return coherence_sum / pair_count if pair_count > 0 else 0.5
    
    def _calculate_box_distance(self, box1: Tuple[float, float, float, float], 
                               box2: Tuple[float, float, float, float]) -> float:
        """Calculate distance between box centers."""
        center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
        center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def update_history(self, signal: float, confidence: float, class_name: str, timestamp: float):
        """Update signal history for temporal analysis."""
        self.signal_history.append({
            'timestamp': timestamp,
            'signal': signal,
            'confidence': confidence,
            'class_name': class_name
        })


class SceneContextAnalyzer:
    """Scene context awareness for adaptive processing based on scene type (BJ vs penetration vs HJ)."""
    
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.scene_history = deque(maxlen=int(fps * 10))  # 10 seconds of scene analysis
        self.current_scene_type = 'unknown'
        self.scene_confidence = 0.0
        self.scene_transition_threshold = 0.7
        
        # Scene-specific processing parameters
        self.scene_configs = {
            'bj_scene': {
                'dominant_classes': ['face', 'hand'],
                'reference_points': {'face': 'bottom', 'hand': 'center'},
                'distance_scaling': 1.2,  # BJ scenes often have more subtle motion
                'temporal_smoothing': 0.8,  # More smoothing for mouth-based scenes
                'priority_boost': {'face': 1.3}  # Boost face priority in BJ scenes
            },
            'penetration_scene': {
                'dominant_classes': ['pussy', 'vagina', 'butt', 'anus'],
                'reference_points': {'pussy': 'center', 'butt': 'mostly_bottom'},
                'distance_scaling': 1.0,  # Standard scaling
                'temporal_smoothing': 0.6,  # Less smoothing for more dynamic scenes
                'priority_boost': {'pussy': 1.2, 'butt': 1.1}
            },
            'hj_scene': {
                'dominant_classes': ['hand'],
                'reference_points': {'hand': 'center'},
                'distance_scaling': 0.9,  # HJ scenes can have more exaggerated motion
                'temporal_smoothing': 0.7,  # Moderate smoothing
                'priority_boost': {'hand': 1.4}  # Strong hand priority boost
            },
            'mixed_scene': {
                'dominant_classes': ['pussy', 'hand', 'face'],
                'reference_points': {'pussy': 'center', 'hand': 'center', 'face': 'bottom'},
                'distance_scaling': 1.0,
                'temporal_smoothing': 0.7,
                'priority_boost': {}  # No specific boosts
            }
        }
    
    def analyze_scene_context(self, detected_objects: List[Dict], 
                            locked_penis_box: Optional[Dict] = None) -> Dict[str, Any]:
        """Analyze current scene context based on detected objects and their relationships."""
        context = {
            'scene_type': 'unknown',
            'confidence': 0.0,
            'dominant_interactions': [],
            'scene_config': self.scene_configs['mixed_scene'],  # Default fallback
            'distance_scaling': 1.0,
            'priority_adjustments': {},
            'processing_hints': {}
        }
        
        if not detected_objects:
            return context
        
        # Analyze object presence and confidence
        class_presence = {}
        class_confidences = {}
        
        for obj in detected_objects:
            class_name = obj['class_name']
            confidence = obj['confidence']
            
            if class_name not in class_presence:
                class_presence[class_name] = 0
                class_confidences[class_name] = []
            
            class_presence[class_name] += 1
            class_confidences[class_name].append(confidence)
        
        # Calculate average confidence for each class
        avg_confidences = {}
        for class_name, confidences in class_confidences.items():
            avg_confidences[class_name] = np.mean(confidences)
        
        # Scene type determination logic
        scene_scores = {
            'bj_scene': 0.0,
            'penetration_scene': 0.0,
            'hj_scene': 0.0,
            'mixed_scene': 0.0
        }
        
        # BJ scene indicators
        if 'face' in avg_confidences:
            face_score = avg_confidences['face'] * 1.2  # Face is strong BJ indicator
            if 'hand' in avg_confidences:
                face_score += avg_confidences['hand'] * 0.5  # Hands often assist in BJ
            scene_scores['bj_scene'] = min(1.0, face_score)
        
        # Penetration scene indicators
        pussy_score = avg_confidences.get('pussy', 0) + avg_confidences.get('vagina', 0)
        butt_score = avg_confidences.get('butt', 0) + avg_confidences.get('anus', 0)
        penetration_score = max(pussy_score, butt_score) * 1.3
        scene_scores['penetration_scene'] = min(1.0, penetration_score)
        
        # HJ scene indicators  
        if 'hand' in avg_confidences and avg_confidences['hand'] > 0.6:
            hand_exclusivity = avg_confidences['hand']
            # Reduce score if other interaction types are strongly present
            if pussy_score > 0.5 or avg_confidences.get('face', 0) > 0.5:
                hand_exclusivity *= 0.7
            scene_scores['hj_scene'] = min(1.0, hand_exclusivity * 1.1)
        
        # Mixed scene indicators
        interaction_diversity = len([c for c in ['face', 'hand', 'pussy', 'butt'] 
                                   if avg_confidences.get(c, 0) > 0.3])
        if interaction_diversity >= 2:
            scene_scores['mixed_scene'] = min(1.0, interaction_diversity * 0.4)
        
        # Determine best scene type
        best_scene = max(scene_scores.items(), key=lambda x: x[1])
        scene_type, scene_confidence = best_scene
        
        # Apply temporal smoothing to avoid rapid scene switching
        self.scene_history.append((scene_type, scene_confidence))
        
        if len(self.scene_history) >= 5:
            # Use majority voting over recent history
            recent_scenes = [s[0] for s in list(self.scene_history)[-5:]]
            scene_counts = {}
            for scene in recent_scenes:
                scene_counts[scene] = scene_counts.get(scene, 0) + 1
            
            # Only change scene if new scene has strong majority
            most_common_scene = max(scene_counts.items(), key=lambda x: x[1])
            if most_common_scene[1] >= 3 and most_common_scene[0] != self.current_scene_type:
                if scene_confidence > self.scene_transition_threshold:
                    self.current_scene_type = most_common_scene[0]
                    self.scene_confidence = scene_confidence
            elif self.current_scene_type == 'unknown':
                self.current_scene_type = scene_type
                self.scene_confidence = scene_confidence
        else:
            # Not enough history, use current detection
            if scene_confidence > 0.5:
                self.current_scene_type = scene_type
                self.scene_confidence = scene_confidence
        
        # Apply scene-specific configuration
        scene_config = self.scene_configs.get(self.current_scene_type, self.scene_configs['mixed_scene'])
        
        context.update({
            'scene_type': self.current_scene_type,
            'confidence': self.scene_confidence,
            'dominant_interactions': scene_config['dominant_classes'],
            'scene_config': scene_config,
            'distance_scaling': scene_config['distance_scaling'],
            'priority_adjustments': scene_config.get('priority_boost', {}),
            'processing_hints': {
                'temporal_smoothing': scene_config['temporal_smoothing'],
                'reference_points': scene_config['reference_points']
            }
        })
        
        return context
    
    def apply_scene_aware_adjustments(self, position: float, class_name: str, 
                                    scene_context: Dict) -> float:
        """Apply scene-aware adjustments to position calculations."""
        # Apply distance scaling
        adjusted_position = position * scene_context.get('distance_scaling', 1.0)
        
        # Apply scene-specific reference point adjustments
        scene_type = scene_context.get('scene_type', 'unknown')
        
        if scene_type == 'bj_scene' and class_name == 'face':
            # BJ scenes: face bottom reference typically gives better results
            # Position adjustment for face-specific scene characteristics
            adjusted_position = min(100, adjusted_position * 1.1)
            
        elif scene_type == 'penetration_scene' and class_name in ['pussy', 'vagina']:
            # Penetration scenes: center reference usually optimal
            # No additional adjustment needed as it's already optimized
            pass
            
        elif scene_type == 'hj_scene' and class_name == 'hand':
            # HJ scenes: hand center with slightly reduced scaling for more dynamic range
            adjusted_position = adjusted_position * 0.95
        
        return np.clip(adjusted_position, 0.0, 100.0)


class RelativeDistanceTracker(BaseTracker):
    """
    Enhanced Relative Distance Tracker with Stage 2 insights.
    
    This tracker combines high-performance vectorized operations with intelligent
    locked penis tracking and distance-based funscript generation.
    """
    
    @property
    def metadata(self) -> TrackerMetadata:
        return TrackerMetadata(
            name="relative_distance",
            display_name="Relative Distance Tracker",
            description="Advanced multi-modal tracker with velocity prediction, scene context awareness, and intelligent signal fusion",
            category="live",
            version="3.0.0",
            author="VR Funscript AI Generator",
            tags=["distance", "velocity-prediction", "scene-aware", "signal-fusion", "temporal-analysis", "advanced"],
            requires_roi=False,
            supports_dual_axis=True
        )
    
    def initialize(self, app_instance, **kwargs) -> bool:
        """Initialize the relative distance tracker."""
        try:
            self.app = app_instance
            self.logger = logging.getLogger(self.__class__.__name__)
            
            # Core tracking state
            self.tracking_active = False
            self.frame_count = 0
            self.prev_frame_gray = None  # Fix the missing attribute error
            self.current_frame_gray = None
            
            # Initialize funscript connection
            if hasattr(self, 'funscript') and self.funscript:
                pass  # Already have funscript from bridge
            elif hasattr(self.app, 'funscript') and self.app.funscript:
                self.funscript = self.app.funscript
            else:
                from funscript.dual_axis_funscript import DualAxisFunscript
                self.funscript = DualAxisFunscript(logger=self.logger)
                self.logger.info("Created local funscript instance for Relative Distance")
            
            # Visual settings
            self.show_debug_overlay = kwargs.get('show_debug_overlay', True)
            self.show_contact_regions = kwargs.get('show_contact_regions', True)
            self.show_distance_lines = kwargs.get('show_distance_lines', True)
            
            # YOLO detection setup
            self.yolo_model = None
            self.yolo_model_path = None
            self._init_yolo_detection()
            
            # Enhanced locked penis tracker with 95th percentile height
            self.current_fps = 30.0  # Default, will be updated from video
            self.locked_penis_tracker = LockedPenisTracker(self.current_fps)
            
            # Advanced prediction and analysis systems
            self.velocity_predictor = VelocityPredictor(max_history=15)
            self.temporal_analyzer = MultiScaleTemporalAnalyzer(self.current_fps)
            self.signal_fusion = AdvancedSignalFusion(self.current_fps)
            self.scene_analyzer = SceneContextAnalyzer(self.current_fps)
            
            # Contact tracking with touch detection and interpolation
            self.contact_boxes = []
            self.touching_boxes = {}  # Track which boxes are touching locked penis
            self.interpolated_boxes = {}  # Track interpolated box positions
            self.contact_cache = {}  # Intelligent caching for performance
            self.cache_ttl = 5  # Cache time-to-live in frames
            
            # Temporal smoothing for stable primary contact selection
            self.primary_contact_history = deque(maxlen=10)  # Last 10 frames of primary contacts
            self.primary_stability_threshold = 5  # Require 5 frames to change primary
            self.current_stable_primary = None
            
            # DIS Optical Flow for interpolation
            self.dis_flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
            try:
                if hasattr(self.dis_flow, 'setFinestScale'):
                    self.dis_flow.setFinestScale(5)
            except Exception:
                pass
            
            # Master touching priority hierarchy (only ONE master touching at a time)
            self.master_priorities = {
                'pussy': 100, 'vagina': 100,  # Highest priority - master contact
                'butt': 90, 'anus': 90,       # Second priority
                'face': 80,                    # Third priority
                'hand': 70,                    # Fourth priority  
                'navel': 60,                   # Fifth priority
                'breast': 50,                  # Sixth priority
                'foot': 40,                    # Lowest priority
                
                # Support classes (for signal fusion, not master)
                'penis': 0, 'locked_penis': 0, 'glans': 0,  # Never master touching
                'hips center': 0,              # Support only
            }
            
            # Signal fusion supporters for pussy tracking
            self.pussy_support_classes = {'navel', 'breast'}
            
            # Current master touching state
            self.current_master_contact = None
            self.master_contact_history = deque(maxlen=10)  # Stability tracking
            
            # Backward compatibility alias
            self.class_priorities = self.master_priorities
            
            # Signal processing - only for touching boxes
            self.signal_amplifier = SignalAmplifier(logger=self.logger)
            self.position_history = deque(maxlen=30)
            self.last_primary_position = 50.0
            self.last_secondary_position = 50.0
            self.distance_calculated_from_touch = False  # Flag to indicate if distance is from actual touch
            
            # Enhanced stability tracking - 3 seconds minimum for touching
            self.stable_touching_threshold = int(3.0 * self.current_fps)  # 3 seconds at current FPS
            self.touching_stability = {}  # Track how long each contact has been touching
            
            # Enhanced persistence - 3 seconds for missing detections
            self.persistence_threshold = int(3.0 * self.current_fps)  # 3 seconds at current FPS
            
            # Performance optimization
            self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="RDT")
            
            self.logger.info("Relative Distance Tracker initialized successfully.")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Relative Distance Tracker: {e}")
            return False
    
    def _init_yolo_detection(self):
        """Initialize YOLO model for object detection."""
        try:
            # Get YOLO model path from app settings
            if hasattr(self.app, 'app_settings'):
                self.yolo_model_path = self.app.app_settings.get('yolo_model_path')
            
            if not self.yolo_model_path:
                # Default model path
                self.yolo_model_path = "models/FunGen-12s-mix-1.0.0.mlpackage"
            
            # Import and initialize YOLO model
            from ultralytics import YOLO
            self.yolo_model = YOLO(self.yolo_model_path)
            self.logger.info(f"YOLO model loaded successfully from: {self.yolo_model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize YOLO model: {e}")
            self.yolo_model = None
    
    def start_tracking(self) -> bool:
        """Start the relative distance tracking."""
        try:
            self.tracking_active = True
            self.frame_count = 0
            self.logger.info("Relative Distance Tracker started.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start tracking: {e}")
            return False
    
    def stop_tracking(self) -> bool:
        """Stop the relative distance tracking."""
        try:
            self.tracking_active = False
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
            self.logger.info("Relative Distance Tracker stopped.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to stop tracking: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray, frame_time_ms: int, 
                     frame_index: Optional[int] = None) -> TrackerResult:
        """Process a frame using enhanced relative distance calculation."""
        try:
            self.frame_count += 1
            
            # Convert frame to grayscale for processing
            self.current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect objects using YOLO
            semantic_objects = self._detect_semantic_objects(frame)
            
            # Analyze scene context for adaptive processing
            scene_context = self.scene_analyzer.analyze_scene_context(
                semantic_objects, 
                self.locked_penis_tracker.box if self.locked_penis_tracker.active else None
            )
            
            # Update locked penis tracking with Stage 2 insights
            self._update_locked_penis_tracking(semantic_objects)
            
            # Detect contact regions and calculate distances with scene awareness
            contact_info = self._detect_contact_regions(semantic_objects, scene_context)
            
            # Calculate relative distances and generate position
            primary_pos, secondary_pos = self._calculate_relative_distances(contact_info)
            
            # Generate funscript actions
            action_log = self._generate_funscript_actions(primary_pos, secondary_pos, frame_time_ms)
            
            # Create debug visualization
            display_frame = self._create_debug_overlay(frame.copy(), contact_info)
            
            # Update frame state for next iteration
            self.prev_frame_gray = self.current_frame_gray.copy()
            
            # Prepare debug info
            debug_info = {
                'locked_penis_active': self.locked_penis_tracker.active,
                'locked_penis_confidence': self.locked_penis_tracker.confidence,
                'contact_regions': len(contact_info.get('contacts', [])),
                'primary_position': primary_pos,
                'secondary_position': secondary_pos,
                'cache_size': len(self.contact_cache)
            }
            
            return TrackerResult(
                processed_frame=display_frame,
                action_log=action_log,
                debug_info=debug_info,
                status_message=f"Tracking: {len(semantic_objects)} objects detected"
            )
            
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}\n{traceback.format_exc()}")
            return TrackerResult(
                processed_frame=frame,
                action_log=[],
                debug_info={"error": str(e)},
                status_message=f"Error: {str(e)}"
            )
    
    def _detect_semantic_objects(self, frame: np.ndarray) -> List[Dict]:
        """Detect semantic objects using YOLO with intelligent caching."""
        if not self.yolo_model:
            return []
        
        try:
            # Run YOLO detection
            results = self.yolo_model(frame, verbose=False)
            
            detected_objects = []
            if results and len(results) > 0:
                result = results[0]
                
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        # Extract box data
                        xyxy = boxes.xyxy[i].cpu().numpy()
                        conf = float(boxes.conf[i].cpu().numpy())
                        cls_id = int(boxes.cls[i].cpu().numpy())
                        
                        # Get class name
                        class_name = self.yolo_model.names.get(cls_id, f"class_{cls_id}")
                        
                        # Filter by confidence and relevance
                        if conf > 0.3 and class_name.lower() in self.master_priorities:
                            detected_objects.append({
                                'bbox': tuple(xyxy),
                                'confidence': conf,
                                'class_name': class_name.lower(),
                                'priority': self.master_priorities.get(class_name.lower(), 0)
                            })
            
            return detected_objects
            
        except Exception as e:
            self.logger.error(f"YOLO detection failed: {e}")
            return []
    
    def _update_locked_penis_tracking(self, detected_objects: List[Dict]):
        """Update locked penis tracking using Stage 2-inspired logic."""
        penis_candidates = [obj for obj in detected_objects 
                          if obj['class_name'] in ['penis', 'locked_penis', 'glans']]
        
        tracker = self.locked_penis_tracker
        current_time = time.time()
        selected_penis = None
        
        # Always increment unseen frames (reset if detection found)
        tracker.unseen_frames += 1
        
        if penis_candidates:
            best_candidate = None
            
            # Try IoU matching first if we have an active lock
            if tracker.box and tracker.active:
                best_iou = 0
                for candidate in penis_candidates:
                    iou = self._calculate_iou(tracker.box, candidate['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_candidate = candidate
                
                # If IoU match is good, update existing lock
                if best_iou > tracker.iou_threshold:
                    selected_penis = best_candidate
                    tracker.box = selected_penis['bbox']
                    tracker.confidence = selected_penis['confidence']
                    tracker.unseen_frames = 0
                    tracker.last_seen_timestamp = current_time
                    tracker.detection_frames.append(self.frame_count)
            
            # If no existing lock or IoU match failed, select best new candidate
            if not selected_penis and penis_candidates:
                best_candidate = max(penis_candidates, 
                                   key=lambda x: x['confidence'] * x['priority'])
                
                selected_penis = best_candidate
                tracker.box = selected_penis['bbox']
                tracker.center = ((selected_penis['bbox'][0] + selected_penis['bbox'][2]) / 2,
                                (selected_penis['bbox'][1] + selected_penis['bbox'][3]) / 2)
                tracker.confidence = selected_penis['confidence']
                tracker.unseen_frames = 0
                tracker.last_seen_timestamp = current_time
                tracker.detection_frames.append(self.frame_count)
                
                if not tracker.established_frame:
                    tracker.established_frame = self.frame_count
        
        # Update conceptual box using Stage 2 logic
        if selected_penis:
            self._update_penis_conceptual_box(tracker, selected_penis)
        
        # Determine if lock should be active (hysteresis logic)
        recent_detections = len([f for f in tracker.detection_frames 
                               if self.frame_count - f <= 15])  # 15-frame window
        
        if not tracker.active:
            # Activation: require strong evidence
            if recent_detections >= tracker.min_detections_activate and tracker.box:
                tracker.active = True
                self.logger.debug(f"Penis lock ACTIVATED: {recent_detections} detections")
        else:
            # Deactivation: both conditions must be met
            if (recent_detections < tracker.min_detections_deactivate and 
                tracker.unseen_frames > tracker.patience):
                self.logger.debug(f"Penis lock DEACTIVATED: {recent_detections} detections, unseen {tracker.unseen_frames}")
                tracker.active = False
                tracker.established_frame = None
    
    def _update_penis_conceptual_box(self, tracker: LockedPenisTracker, selected_penis: Dict):
        """Update conceptual box with 95th percentile height calculation."""
        current_height = selected_penis['bbox'][3] - selected_penis['bbox'][1]
        current_time = time.time()
        
        # Add current height to rolling window with timestamp
        tracker.height_history.append({
            'height': current_height,
            'timestamp': current_time,
            'frame': self.frame_count
        })
        
        # Update 95th percentile height every second or when we have enough data
        if (current_time - tracker.last_height_update > tracker.height_update_interval or 
            len(tracker.height_history) >= 30):  # At least 1 second of data at 30fps
            
            self._calculate_95th_percentile_height(tracker)
            tracker.last_height_update = current_time
        
        # Update max height tracking (for fallback)
        tracker.max_height = max(tracker.max_height, current_height)
        
        # Use 95th percentile height for conceptual box if available
        conceptual_height = tracker.computed_height_95th if tracker.computed_height_95th > 0 else tracker.max_height
        
        # Update conceptual full stroke box using 95th percentile height
        x1, _, x2, y2_raw = selected_penis['bbox']
        conceptual_full_box = (x1, y2_raw - conceptual_height, x2, y2_raw)
        tracker.conceptual_box = conceptual_full_box
        tracker.last_raw_coords = selected_penis['bbox']
    
    def _calculate_95th_percentile_height(self, tracker: LockedPenisTracker):
        """Calculate 95th percentile height from 30-second rolling window."""
        current_time = time.time()
        window_start = current_time - tracker.height_window_seconds
        
        # Filter to only include heights from the last 30 seconds
        recent_heights = [
            h['height'] for h in tracker.height_history 
            if h['timestamp'] >= window_start
        ]
        
        if len(recent_heights) >= 10:  # Need at least 10 samples for reliable percentile
            tracker.computed_height_95th = np.percentile(recent_heights, 95)
            self.logger.debug(f"Updated 95th percentile height: {tracker.computed_height_95th:.1f} from {len(recent_heights)} samples")
        else:
            self.logger.debug(f"Not enough height samples ({len(recent_heights)}) for 95th percentile calculation")
    
    def _calculate_iou(self, box1: Tuple[float, float, float, float], 
                      box2: Tuple[float, float, float, float]) -> float:
        """Calculate Intersection over Union (IoU) between two boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x1_inter >= x2_inter or y1_inter >= y2_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Calculate union
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _detect_contact_regions(self, detected_objects: List[Dict], scene_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Detect contact regions and prepare ALL objects for display with touching status."""
        contact_info = {
            'touching_contacts': [],
            'interpolated_contacts': [],
            'locked_penis_box': None,
            'primary_touching_contact': None,
            'contacts': [],  # ALL objects for display
            'all_objects': [],  # ALL detected objects including penis
            'scene_context': scene_context or {}
        }
        
        # Separate penis and non-penis objects
        non_penis_objects = [obj for obj in detected_objects 
                           if obj['class_name'] not in ['penis', 'locked_penis', 'glans']]
        
        # Mark ALL objects with their touching status
        all_display_objects = []
        
        # Add locked penis info
        if self.locked_penis_tracker.active and self.locked_penis_tracker.box:
            penis_box = self.locked_penis_tracker.conceptual_box or self.locked_penis_tracker.box
            contact_info['locked_penis_box'] = {
                'bbox': penis_box,
                'confidence': self.locked_penis_tracker.confidence,
                'active': True,
                'height_95th': self.locked_penis_tracker.computed_height_95th
            }
            
            # Find objects that are ACTUALLY TOUCHING the locked penis
            touching_objects = self._find_touching_objects(detected_objects, penis_box)
            contact_info['touching_contacts'] = touching_objects
            
            # Handle interpolation for flickering boxes
            interpolated_contacts = self._handle_box_interpolation(touching_objects, penis_box)
            contact_info['interpolated_contacts'] = interpolated_contacts
            
            # Create touching set for fast lookup
            touching_boxes = set()
            for obj in touching_objects + interpolated_contacts:
                box_id = self._get_box_id(obj['bbox'])
                touching_boxes.add(box_id)
            
            # Process ALL non-penis objects and mark their touching status
            for obj in non_penis_objects:
                obj_copy = obj.copy()
                box_id = self._get_box_id(obj['bbox'])
                obj_copy['is_touching'] = box_id in touching_boxes
                obj_copy['is_interpolated'] = False  # Will be overridden for interpolated
                obj_copy['priority'] = self.master_priorities.get(obj['class_name'], 0)
                all_display_objects.append(obj_copy)
            
            # Override interpolated status for interpolated contacts
            for interp_obj in interpolated_contacts:
                interp_box_id = self._get_box_id(interp_obj['bbox'])
                for display_obj in all_display_objects:
                    if self._get_box_id(display_obj['bbox']) == interp_box_id:
                        display_obj['is_interpolated'] = True
                        break
            
            # Determine SINGLE master touching contact with priority hierarchy
            all_contacts = touching_objects + interpolated_contacts
            if all_contacts:
                # Select master contact based on strict priority hierarchy
                master_contact = self._select_master_contact(all_contacts)
                
                # Check for pussy support signal fusion
                support_signals = self._get_pussy_support_signals(all_contacts, master_contact)
                
                contact_info['primary_contact'] = master_contact
                contact_info['primary_touching_contact'] = master_contact
                contact_info['support_signals'] = support_signals
        else:
            # When no locked penis, mark all objects as non-touching
            for obj in non_penis_objects:
                obj_copy = obj.copy()
                obj_copy['is_touching'] = False
                obj_copy['is_interpolated'] = False
                all_display_objects.append(obj_copy)
            
            if non_penis_objects:
                contact_info['primary_contact'] = max(non_penis_objects,
                    key=lambda x: x['priority'] * x['confidence'])
        
        # Store all objects for overlay display
        contact_info['contacts'] = all_display_objects
        contact_info['all_objects'] = detected_objects  # Store original objects too
        
        return contact_info
    
    def _select_master_contact(self, all_contacts: List[Dict]) -> Dict:
        """Select SINGLE master touching contact based on strict priority hierarchy."""
        # Filter out contacts that can be master touching
        master_candidates = []
        for contact in all_contacts:
            class_name = contact['class_name']
            priority = self.master_priorities.get(class_name, 0)
            if priority > 0:  # Only classes with priority > 0 can be master
                master_candidates.append((contact, priority))
        
        if not master_candidates:
            # Fallback to highest confidence if no priority classes
            return max(all_contacts, key=lambda x: x['confidence'])
        
        # Sort by priority (highest first), then by confidence within priority
        master_candidates.sort(key=lambda x: (x[1], x[0]['confidence']), reverse=True)
        
        # Get the highest priority contact
        new_master = master_candidates[0][0]
        
        # Apply temporal stability to prevent master jumping
        master_id = self._get_box_id(new_master['bbox'])
        self.master_contact_history.append(master_id)
        
        # If we have a stable master, check if we should keep it
        if self.current_master_contact:
            current_master_id = self._get_box_id(self.current_master_contact['bbox'])
            
            # Check if current master is still in candidates
            current_master_still_valid = any(
                self._get_box_id(contact['bbox']) == current_master_id 
                for contact, _ in master_candidates
            )
            
            # Get current master priority for comparison
            current_priority = self.master_priorities.get(self.current_master_contact['class_name'], 0)
            new_priority = self.master_priorities.get(new_master['class_name'], 0)
            
            if current_master_still_valid:
                # Only change master if new one has SIGNIFICANTLY higher priority
                priority_gap = new_priority - current_priority
                if priority_gap < 20:  # Require 20+ priority difference for immediate change
                    # Check stability history for gradual changes
                    recent_frames = list(self.master_contact_history)[-5:]
                    new_master_votes = sum(1 for frame_id in recent_frames if frame_id == master_id)
                    
                    if new_master_votes < 3:  # Need 3/5 recent frames to change
                        # Keep current master
                        for contact, _ in master_candidates:
                            if self._get_box_id(contact['bbox']) == current_master_id:
                                return contact
        
        # Update master and return new one
        self.current_master_contact = new_master
        return new_master
    
    def _get_pussy_support_signals(self, all_contacts: List[Dict], master_contact: Dict) -> List[Dict]:
        """Get navel/breast support signals when pussy is master."""
        support_signals = []
        
        # Only provide support when pussy is master
        if master_contact['class_name'] not in ['pussy', 'vagina']:
            return support_signals
        
        # Find navel and breast contacts for signal fusion
        for contact in all_contacts:
            if contact['class_name'] in self.pussy_support_classes:
                support_signals.append({
                    'class_name': contact['class_name'],
                    'bbox': contact['bbox'],
                    'confidence': contact['confidence'],
                    'support_type': 'signal_fusion'
                })
        
        return support_signals
    
    def _fuse_pussy_support_signals(self, primary_signal: float, support_signals: List[Dict], 
                                   locked_penis_box: Dict) -> float:
        """Fuse navel/breast support signals with pussy primary signal."""
        if not support_signals:
            return primary_signal
        
        # Calculate support signal contributions
        support_contributions = []
        for support in support_signals:
            # Calculate distance for support signal
            support_distance = self._calculate_normalized_distance_to_penis(
                support['bbox'], 
                support['class_name']
            )
            
            if support_distance is not None:
                # Weight support signals based on class type and confidence
                if support['class_name'] == 'navel':
                    weight = 0.3 * support['confidence']  # Navel gets 30% max weight
                elif support['class_name'] == 'breast':
                    weight = 0.2 * support['confidence']  # Breast gets 20% max weight
                else:
                    weight = 0.1 * support['confidence']  # Other supports get 10% max weight
                
                support_contributions.append((support_distance, weight))
        
        if not support_contributions:
            return primary_signal
        
        # Weighted fusion of primary + support signals
        total_weight = 1.0  # Primary signal always has weight 1.0
        weighted_sum = primary_signal * 1.0
        
        for support_signal, weight in support_contributions:
            weighted_sum += support_signal * weight
            total_weight += weight
        
        # Normalize and clamp to 0-100 range
        fused_signal = weighted_sum / total_weight
        return np.clip(fused_signal, 0, 100)
    
    def _get_stable_primary_contact(self, best_contact: Dict, all_contacts: List[Dict]) -> Dict:
        """Apply temporal smoothing to prevent primary contact jumping."""
        current_best_id = self._get_box_id(best_contact['bbox'])
        
        # Add to history
        self.primary_contact_history.append(current_best_id)
        
        # If we have a current stable primary, check if we should keep it
        if self.current_stable_primary:
            current_stable_id = self._get_box_id(self.current_stable_primary['bbox'])
            
            # Check if current stable primary is still in touching contacts
            current_stable_still_touching = any(
                self._get_box_id(contact['bbox']) == current_stable_id 
                for contact in all_contacts
            )
            
            if current_stable_still_touching:
                # Count how many recent frames want to change
                recent_frames = list(self.primary_contact_history)[-self.primary_stability_threshold:]
                change_votes = sum(1 for frame_id in recent_frames if frame_id != current_stable_id)
                
                # Only change if majority of recent frames want the change
                if change_votes < self.primary_stability_threshold:
                    # Find and return the current stable contact from all_contacts
                    for contact in all_contacts:
                        if self._get_box_id(contact['bbox']) == current_stable_id:
                            return contact
        
        # Either no stable primary or it should change - update to new best
        self.current_stable_primary = best_contact
        return best_contact
    
    def _find_touching_objects(self, detected_objects: List[Dict], penis_box: Tuple[float, float, float, float]) -> List[Dict]:
        """Find objects that are actually touching the locked penis box."""
        touching_objects = []
        
        # Exclude penis objects from contact detection
        contact_candidates = [obj for obj in detected_objects 
                            if obj['class_name'] not in ['penis', 'locked_penis', 'glans']]
        
        for obj in contact_candidates:
            # Check if boxes are touching (IoU > 0 or very close)
            iou = self._calculate_iou(obj['bbox'], penis_box)
            distance = self._calculate_vertical_distance(obj['bbox'], penis_box)
            
            # Track stability of touching
            box_id = self._get_box_id(obj['bbox'])
            if box_id not in self.touching_stability:
                self.touching_stability[box_id] = 0
            
            # Consider touching if IoU > 0 or distance < 10 pixels
            if iou > 0 or distance < 10:
                # Increment stability counter
                self.touching_stability[box_id] += 1
                
                # Only consider stable touching if it's been touching for at least 3 seconds
                if self.touching_stability[box_id] >= self.stable_touching_threshold:
                    obj_copy = obj.copy()
                    obj_copy['touch_iou'] = iou
                    obj_copy['touch_distance'] = distance
                    obj_copy['is_touching'] = True
                    obj_copy['is_interpolated'] = False
                    touching_objects.append(obj_copy)
                    
                    # Update touching boxes tracking
                    self.touching_boxes[box_id] = {
                        'last_seen_frame': self.frame_count,
                        'last_bbox': obj['bbox'],
                        'class_name': obj['class_name'],
                        'touch_history': deque(maxlen=10)
                    }
                    self.touching_boxes[box_id]['touch_history'].append({
                        'frame': self.frame_count,
                        'bbox': obj['bbox'],
                        'iou': iou
                    })
                else:
                    # Not yet stable, but track it for potential future stability
                    self.touching_boxes[box_id] = {
                        'last_seen_frame': self.frame_count,
                        'last_bbox': obj['bbox'],
                        'class_name': obj['class_name'],
                        'touch_history': deque(maxlen=10)
                    }
                    self.touching_boxes[box_id]['touch_history'].append({
                        'frame': self.frame_count,
                        'bbox': obj['bbox'],
                        'iou': iou
                    })
            else:
                # Not touching, reset stability counter if it exists
                if box_id in self.touching_stability:
                    self.touching_stability[box_id] = 0
        
        return touching_objects
    
    def _calculate_vertical_distance(self, box1: Tuple[float, float, float, float], 
                                   box2: Tuple[float, float, float, float]) -> float:
        """Calculate vertical distance between two boxes (ignoring horizontal distance)."""
        # Calculate vertical center-to-center distance
        center1_y = (box1[1] + box1[3]) / 2
        center2_y = (box2[1] + box2[3]) / 2
        
        # Return absolute vertical distance
        return abs(center1_y - center2_y)
    
    def _get_box_id(self, bbox: Tuple[float, float, float, float]) -> str:
        """Generate a simple ID for a bounding box based on its center and size."""
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return f"{int(center_x)}_{int(center_y)}_{int(width)}_{int(height)}"
    
    def _handle_box_interpolation(self, current_touching: List[Dict], penis_box: Tuple[float, float, float, float]) -> List[Dict]:
        """Handle interpolation for flickering boxes using DIS Optical Flow."""
        interpolated_contacts = []
        
        # Check for previously touching boxes that are now missing
        current_frame = self.frame_count
        missing_touch_threshold = int(30 * 5)  # Allow up to 5 seconds of missing detection (150 frames at 30fps)
        persistence_threshold = self.persistence_threshold  # But only persist for 3 seconds
        
        # Get IDs of currently touching boxes
        current_touching_ids = set()
        for touch in current_touching:
            box_id = self._get_box_id(touch['bbox'])
            current_touching_ids.add(box_id)
        
        # Check for class replacement - if a different class is now touching, stop interpolating
        current_touching_classes = set()
        for touch in current_touching:
            current_touching_classes.add(touch['class_name'])
        
        # Check each previously touching box
        boxes_to_remove = []
        for box_id, touch_info in self.touching_boxes.items():
            frames_missing = current_frame - touch_info['last_seen_frame']
            
            # Check if this class has been replaced by a different class detection
            class_replaced = (touch_info['class_name'] in current_touching_classes and 
                            box_id not in current_touching_ids)
            
            # If box was recently touching but now missing, try to interpolate
            # Only interpolate for up to 3 seconds (persistence_threshold)
            if (box_id not in current_touching_ids and 
                frames_missing <= persistence_threshold and 
                frames_missing > 0 and 
                not class_replaced):  # Don't interpolate if class was replaced
                
                # Primary: Use velocity-based prediction for smooth interpolation
                interpolated_box = self._interpolate_box_with_velocity_prediction(touch_info, frames_missing)
                
                # Fallback 1: DIS optical flow if velocity prediction fails
                if not interpolated_box:
                    interpolated_box = self._interpolate_box_with_optical_flow(touch_info, frames_missing)
                
                # Fallback 2: Simple linear interpolation for very short gaps
                if not interpolated_box and frames_missing <= 10:  # Only for short gaps
                    interpolated_box = self._fallback_linear_interpolation(touch_info, frames_missing)
                
                if interpolated_box:
                    # Check if interpolated box is still touching penis
                    iou = self._calculate_iou(interpolated_box['bbox'], penis_box)
                    distance = self._calculate_vertical_distance(interpolated_box['bbox'], penis_box)
                    
                    if iou > 0 or distance < 15:  # Slightly more lenient for interpolated
                        interpolated_box['is_interpolated'] = True
                        interpolated_box['is_touching'] = True
                        interpolated_box['touch_iou'] = iou
                        interpolated_box['touch_distance'] = distance
                        interpolated_box['interpolation_frames'] = frames_missing
                        interpolated_contacts.append(interpolated_box)
                        
                        # Enhanced debug logging for interpolation quality
                        quality = interpolated_box.get('interpolation_quality', 0.0)
                        confidence = interpolated_box.get('confidence', 0.0)
                        self.logger.info(f"✓ Interpolated {touch_info['class_name']} for {frames_missing} frames | Quality: {quality:.2f} | Confidence: {confidence:.2f}")
                    else:
                        # Interpolated box no longer touches - remove from tracking
                        boxes_to_remove.append(box_id)
                        self.logger.debug(f"Stopping interpolation for {touch_info['class_name']}: no longer touching")
            elif frames_missing > missing_touch_threshold:
                # Box has been missing too long - remove from tracking
                boxes_to_remove.append(box_id)
                self.logger.debug(f"Stopping interpolation for {touch_info['class_name']}: exceeded 5 second timeout")
            elif class_replaced:
                # Class was replaced by a new detection - remove old tracking
                boxes_to_remove.append(box_id)
                self.logger.debug(f"Stopping interpolation for {touch_info['class_name']}: replaced by new detection")
        
        # Clean up boxes that are no longer relevant
        for box_id in boxes_to_remove:
            # Remove from touching stability tracking as well
            if box_id in self.touching_stability:
                del self.touching_stability[box_id]
            del self.touching_boxes[box_id]
        
        return interpolated_contacts
    
    def _interpolate_box_with_velocity_prediction(self, touch_info: Dict, frames_missing: int) -> Optional[Dict]:
        """Primary interpolation method using velocity-based prediction."""
        try:
            class_name = touch_info['class_name']
            
            # Update velocity predictor with historical position (if available)
            if 'position_history' in touch_info and touch_info['position_history']:
                for timestamp, bbox in touch_info['position_history']:
                    self.velocity_predictor.update_position(timestamp, bbox)
            
            # Calculate target timestamp for prediction
            last_timestamp = touch_info.get('last_timestamp', time.time())
            current_timestamp = time.time()
            frames_delta = frames_missing
            
            # Use frame-based prediction if timestamp prediction fails
            target_timestamp = last_timestamp + (frames_delta / self.current_fps)
            
            # Attempt velocity prediction
            predicted_box = self.velocity_predictor.predict_position(target_timestamp)
            
            if predicted_box:
                # Check prediction reliability
                is_reliable = self.velocity_predictor.is_prediction_reliable(target_timestamp)
                velocity_magnitude = self.velocity_predictor.get_velocity_magnitude()
                
                # Only use prediction if it's reliable and motion is reasonable
                if is_reliable and velocity_magnitude < 200:  # Reasonable motion threshold
                    predicted_contact = {
                        'bbox': predicted_box,
                        'confidence': touch_info['confidence'] * 0.8,  # Reduce confidence for prediction
                        'class_name': class_name,
                        'is_interpolated': True,
                        'interpolation_method': 'velocity_prediction',
                        'interpolation_frames': frames_missing,
                        'prediction_reliability': 'high' if velocity_magnitude < 100 else 'medium',
                        'velocity_magnitude': velocity_magnitude,
                        'priority': self.master_priorities.get(class_name, 0)
                    }
                    
                    self.logger.debug(f"🎯 Velocity prediction: {class_name} predicted for {frames_missing} frames "
                                    f"(vel: {velocity_magnitude:.1f}, reliability: {predicted_contact['prediction_reliability']})")
                    
                    return predicted_contact
            
        except Exception as e:
            self.logger.debug(f"Velocity prediction failed for {touch_info.get('class_name', 'unknown')}: {e}")
            
        return None
    
    def _interpolate_box_with_optical_flow(self, touch_info: Dict, frames_missing: int) -> Optional[Dict]:
        """Enhanced interpolation using accumulated optical flow with multi-point sampling."""
        if self.prev_frame_gray is None or self.current_frame_gray is None:
            return None
        
        try:
            # Calculate optical flow between previous and current frame
            flow = self.dis_flow.calc(self.prev_frame_gray, self.current_frame_gray, None)
            
            # Get the last known bbox and expand sampling area
            last_bbox = touch_info['last_bbox']
            x1, y1, x2, y2 = last_bbox
            
            # Multi-point sampling for more robust flow estimation
            sample_points = [
                (int((x1 + x2) / 2), int((y1 + y2) / 2)),  # Center
                (int(x1 + (x2-x1)*0.25), int(y1 + (y2-y1)*0.25)),  # Top-left quarter
                (int(x1 + (x2-x1)*0.75), int(y1 + (y2-y1)*0.25)),  # Top-right quarter
                (int(x1 + (x2-x1)*0.25), int(y1 + (y2-y1)*0.75)),  # Bottom-left quarter
                (int(x1 + (x2-x1)*0.75), int(y1 + (y2-y1)*0.75))   # Bottom-right quarter
            ]
            
            # Sample flow at multiple points and average
            valid_flows = []
            for pt_x, pt_y in sample_points:
                if (0 <= pt_y < flow.shape[0] and 0 <= pt_x < flow.shape[1]):
                    dx, dy = flow[pt_y, pt_x]
                    # Filter out extreme outlier flows
                    if abs(dx) < 50 and abs(dy) < 50:  # Reasonable motion bounds
                        valid_flows.append((dx, dy))
            
            if not valid_flows:
                return None
            
            # Average the valid flow vectors
            avg_dx = sum(dx for dx, dy in valid_flows) / len(valid_flows)
            avg_dy = sum(dy for dx, dy in valid_flows) / len(valid_flows)
            
            # Enhanced motion prediction with velocity smoothing
            if 'velocity_history' not in touch_info:
                touch_info['velocity_history'] = deque(maxlen=5)
            
            # Add current velocity to history
            touch_info['velocity_history'].append((avg_dx, avg_dy))
            
            # Use smoothed velocity for better prediction
            if len(touch_info['velocity_history']) >= 2:
                # Weighted average favoring recent velocities
                weights = [0.1, 0.2, 0.3, 0.4]  # Most recent gets highest weight
                smoothed_dx = 0
                smoothed_dy = 0
                total_weight = 0
                
                for i, (vx, vy) in enumerate(list(touch_info['velocity_history'])[-4:]):
                    w = weights[min(i, len(weights)-1)]
                    smoothed_dx += vx * w
                    smoothed_dy += vy * w
                    total_weight += w
                
                if total_weight > 0:
                    smoothed_dx /= total_weight
                    smoothed_dy /= total_weight
                else:
                    smoothed_dx, smoothed_dy = avg_dx, avg_dy
            else:
                smoothed_dx, smoothed_dy = avg_dx, avg_dy
            
            # Exponential decay for longer predictions
            decay_factor = max(0.3, 1.0 - (frames_missing * 0.1))  # Decay confidence over time
            predicted_dx = smoothed_dx * frames_missing * decay_factor
            predicted_dy = smoothed_dy * frames_missing * decay_factor
            
            # Apply bounds checking to prevent boxes from going off-screen
            frame_h, frame_w = self.current_frame_gray.shape
            interpolated_bbox = (
                max(0, min(frame_w - (x2-x1), x1 + predicted_dx)),
                max(0, min(frame_h - (y2-y1), y1 + predicted_dy)),
                max(x2-x1, min(frame_w, x2 + predicted_dx)),
                max(y2-y1, min(frame_h, y2 + predicted_dy))
            )
            
            # Calculate confidence based on flow consistency and frames missing
            flow_consistency = 1.0 - (np.std([dx for dx, dy in valid_flows]) + np.std([dy for dx, dy in valid_flows])) / 20.0
            temporal_confidence = max(0.2, 1.0 - (frames_missing * 0.15))
            interpolation_confidence = max(0.3, flow_consistency * temporal_confidence)
            
            return {
                'bbox': interpolated_bbox,
                'class_name': touch_info['class_name'],
                'confidence': interpolation_confidence,
                'priority': self.master_priorities.get(touch_info['class_name'], 1),
                'motion_vector': (predicted_dx / max(1, frames_missing), predicted_dy / max(1, frames_missing)),
                'interpolation_quality': flow_consistency,
                'frames_interpolated': frames_missing
            }
        
        except Exception as e:
            self.logger.warning(f"Enhanced optical flow interpolation failed: {e}")
            return None
    
    def _fallback_linear_interpolation(self, touch_info: Dict, frames_missing: int) -> Optional[Dict]:
        """Simple fallback interpolation when optical flow fails."""
        try:
            # Get touch history for velocity estimation
            if 'touch_history' not in touch_info or len(touch_info['touch_history']) < 2:
                return None
            
            # Calculate velocity from recent history
            recent_history = list(touch_info['touch_history'])[-2:]
            if len(recent_history) < 2:
                return None
            
            prev_pos = recent_history[0]
            curr_pos = recent_history[1] 
            
            # Simple velocity calculation (pixels per frame)
            frame_diff = curr_pos['frame'] - prev_pos['frame']
            if frame_diff <= 0:
                return None
            
            # Use a simple position hold strategy for short gaps
            last_bbox = touch_info['last_bbox']
            
            return {
                'bbox': last_bbox,  # Hold position
                'class_name': touch_info['class_name'],
                'confidence': max(0.4, 0.8 - (frames_missing * 0.05)),  # Decay confidence
                'priority': self.master_priorities.get(touch_info['class_name'], 1),
                'motion_vector': (0, 0),  # No motion vector for position hold
                'interpolation_quality': 0.5,  # Medium quality for fallback
                'frames_interpolated': frames_missing,
                'fallback_method': 'position_hold'
            }
            
        except Exception as e:
            self.logger.warning(f"Fallback interpolation failed: {e}")
            return None
    
    def _calculate_relative_distances(self, contact_info: Dict[str, Any]) -> Tuple[float, float]:
        """Calculate relative distances ONLY for touching boxes."""
        primary_pos = self.last_primary_position
        secondary_pos = self.last_secondary_position
        
        # Get locked penis info and touching contact
        locked_penis_box = contact_info.get('locked_penis_box')
        primary_touching_contact = contact_info.get('primary_touching_contact')
        
        # Reset distance calculation flag
        self.distance_calculated_from_touch = False
        
        # ONLY calculate distance if we have a touching contact
        if locked_penis_box and locked_penis_box['active'] and primary_touching_contact:
            # Calculate primary signal from master contact
            primary_signal = self._calculate_normalized_distance_to_penis(
                primary_touching_contact['bbox'],
                primary_touching_contact['class_name']
            )
            
            if primary_signal is not None:
                # Apply advanced signal fusion if support signals are available
                support_signals = contact_info.get('support_signals', [])
                if support_signals and primary_touching_contact['class_name'] in ['pussy', 'vagina']:
                    # Prepare support signals for advanced fusion
                    enhanced_support_signals = []
                    for support in support_signals:
                        support_signal = self._calculate_normalized_distance_to_penis(
                            support['bbox'], support['class_name']
                        )
                        if support_signal is not None:
                            enhanced_support_signals.append({
                                **support,
                                'signal': support_signal
                            })
                    
                    # Use advanced signal fusion
                    fusion_result = self.signal_fusion.fuse_signals(
                        primary_signal, 
                        primary_touching_contact['confidence'],
                        primary_touching_contact['class_name'],
                        enhanced_support_signals,
                        locked_penis_box
                    )
                    
                    primary_pos = fusion_result['fused_signal']
                    
                    # Update signal history for temporal analysis
                    self.signal_fusion.update_history(
                        primary_pos, 
                        fusion_result['fusion_confidence'],
                        primary_touching_contact['class_name'],
                        time.time()
                    )
                    
                    # Enhanced logging with fusion quality metrics
                    quality = fusion_result.get('fusion_quality', 'unknown')
                    contributors = ', '.join(fusion_result.get('contributors', [primary_touching_contact['class_name']]))
                    self.logger.debug(f"🔗 Advanced fusion: {primary_signal:.1f} → {primary_pos:.1f} "
                                    f"({quality} quality, {contributors})")
                    
                else:
                    primary_pos = primary_signal
                    # Update history for non-fusion signals too
                    self.signal_fusion.update_history(
                        primary_pos, 
                        primary_touching_contact['confidence'],
                        primary_touching_contact['class_name'],
                        time.time()
                    )
                
                self.distance_calculated_from_touch = True
                
                # Log different messages for actual vs interpolated touches
                if primary_touching_contact.get('is_interpolated', False):
                    interpolation_frames = primary_touching_contact.get('interpolation_frames', 0)
                    self.logger.debug(f"Distance signal: {primary_signal:.1f} from INTERPOLATED {primary_touching_contact['class_name']} (missing {interpolation_frames} frames)")
                else:
                    iou = primary_touching_contact.get('touch_iou', 0)
                    self.logger.debug(f"Distance signal: {primary_signal:.1f} from touching {primary_touching_contact['class_name']} (IoU: {iou:.3f})")
        
        # For secondary axis, use horizontal position of touching contact if available
        if primary_touching_contact and self.distance_calculated_from_touch:
            contact_center_x = (primary_touching_contact['bbox'][0] + primary_touching_contact['bbox'][2]) / 2
            # Normalize to frame width (assuming 640 width for now)
            normalized_x = contact_center_x / 640.0
            secondary_pos = normalized_x * 100.0
        
        # Only update history if we have a valid touch-based calculation
        if self.distance_calculated_from_touch:
            self.position_history.append(primary_pos)
            
            # Update temporal analyzer with motion patterns
            current_velocity = 0.0
            if len(self.position_history) >= 2:
                recent_positions = list(self.position_history)[-2:]
                current_velocity = abs(recent_positions[-1] - recent_positions[-2]) * self.current_fps
            
            self.temporal_analyzer.update(time.time(), primary_pos, current_velocity)
            
            # Get motion pattern analysis for enhanced processing
            motion_pattern = self.temporal_analyzer.detect_motion_pattern()
            
            # Apply pattern-aware adjustments to position
            if motion_pattern.get('stroke_detected', False):
                stroke_frequency = motion_pattern.get('stroke_frequency', 0.0)
                if 0.5 < stroke_frequency < 5.0:  # Reasonable stroke frequency range
                    # Apply slight enhancement for detected strokes
                    stroke_enhancement = min(2.0, stroke_frequency / 2.0)  # Max 2x enhancement
                    primary_pos = np.clip(primary_pos * stroke_enhancement, 0, 100)
                    
                    self.logger.debug(f"🎵 Stroke pattern detected: {stroke_frequency:.1f}Hz, enhanced signal")
            
            # Apply scene-aware adjustments if context is available
            if 'scene_context' in contact_info:
                scene_context = contact_info['scene_context']
                if scene_context and primary_touching_contact:
                    scene_adjusted_pos = self.scene_analyzer.apply_scene_aware_adjustments(
                        primary_pos, 
                        primary_touching_contact['class_name'],
                        scene_context
                    )
                    
                    if scene_adjusted_pos != primary_pos:
                        scene_type = scene_context.get('scene_type', 'unknown')
                        self.logger.debug(f"🎬 Scene adjustment ({scene_type}): {primary_pos:.1f} → {scene_adjusted_pos:.1f}")
                        primary_pos = scene_adjusted_pos
            
            # Apply smoothing only for touch-based positions
            if len(self.position_history) > 1:
                recent_positions = list(self.position_history)[-3:]  # Shorter smoothing for responsiveness
                primary_pos = np.mean(recent_positions)
        else:
            # No touching contact - maintain last known position or decay slowly
            decay_factor = 0.98  # Very slow decay towards neutral
            primary_pos = self.last_primary_position * decay_factor + 50.0 * (1 - decay_factor)
        
        # Clamp values
        primary_pos = np.clip(primary_pos, 0, 100)
        secondary_pos = np.clip(secondary_pos, 0, 100)
        
        # Update last positions
        self.last_primary_position = primary_pos
        self.last_secondary_position = secondary_pos
        
        return primary_pos, secondary_pos
    
    def _calculate_normalized_distance_to_penis(self, contact_box: Tuple[float, float, float, float], 
                                               contact_class: str) -> Optional[float]:
        """Calculate normalized vertical distance using Stage 2's optimal reference points."""
        tracker = self.locked_penis_tracker
        
        if not tracker.active or not (tracker.conceptual_box or tracker.box):
            return None
        
        # Use conceptual box if available, otherwise fallback to raw box
        penis_box = tracker.conceptual_box or tracker.box
        penis_base_y = penis_box[3]  # Bottom of conceptual box
        max_height = max(tracker.max_height, 100.0)  # Avoid division by zero
        
        # Use Stage 2's optimal reference points for each interaction type
        if contact_class == 'face':
            contact_y_ref = contact_box[3]  # Bottom of face
        elif contact_class == 'hand':
            contact_y_ref = (contact_box[1] + contact_box[3]) / 2  # Center Y of hand
        elif contact_class in ['pussy', 'vagina']:
            contact_y_ref = (contact_box[1] + contact_box[3]) / 2  # Center Y of pussy
        elif contact_class in ['butt', 'anus']:
            contact_y_ref = (9 * contact_box[3] + contact_box[1]) / 10  # Mostly bottom
        else:  # breast, foot, etc.
            contact_y_ref = (contact_box[1] + contact_box[3]) / 2  # Center
        
        # Calculate vertical distance from contact to penis base (only vertical, not euclidean)
        vertical_distance = penis_base_y - contact_y_ref  # Positive when contact is above penis
        
        # Normalize using max height as reference  
        normalized_distance = (vertical_distance / max_height) * 100.0
        
        # Convert to 0-100 scale where 100=close, 0=far
        # When contact is at penis base (distance=0), position should be ~50
        # When contact is above penis (negative distance), position increases toward 100
        # When contact is below penis (positive distance), position decreases toward 0
        position = 50.0 - normalized_distance
        
        return np.clip(position, 0.0, 100.0)
    
    def _generate_funscript_actions(self, primary_pos: float, secondary_pos: float,
                                  frame_time_ms: int) -> List[Dict]:
        """Generate funscript actions based on calculated positions."""
        action_log = []
        
        if not self.tracking_active:
            return action_log
        
        try:
            # Create primary action
            action_primary = {
                "at": frame_time_ms,
                "pos": int(np.clip(primary_pos, 0, 100))
            }
            
            # Create secondary action
            action_secondary = {
                "at": frame_time_ms,
                "secondary_pos": int(np.clip(secondary_pos, 0, 100))
            }
            
            # Add to funscript if available
            if hasattr(self, 'funscript') and self.funscript:
                self.funscript.add_action(frame_time_ms, int(primary_pos))
                if hasattr(self.funscript, 'add_secondary_action'):
                    self.funscript.add_secondary_action(frame_time_ms, int(secondary_pos))
            
            action_log.append({**action_primary, **action_secondary})
            
        except Exception as e:
            self.logger.warning(f"Action generation failed: {e}")
        
        return action_log
    
    def _create_debug_overlay(self, frame: np.ndarray, contact_info: Dict[str, Any]) -> np.ndarray:
        """Create professional debug visualization overlay with clear interpolation indicators."""
        if not self.show_debug_overlay:
            return frame
        
        try:
            frame_h, frame_w = frame.shape[:2]
            
            # Clean locked penis visualization - no text overlay
            locked_penis_box = contact_info.get('locked_penis_box')
            if locked_penis_box and locked_penis_box['active']:
                bbox = locked_penis_box['bbox']
                
                # Clean locked penis box - just a clean green rectangle
                cv2.rectangle(frame, 
                            (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]), int(bbox[3])),
                            (0, 255, 0), 3)
            
            # Show ALL detected objects with master/support distinction
            if self.show_contact_regions:
                contacts = contact_info.get('contacts', [])
                master_contact = contact_info.get('primary_contact')
                support_signals = contact_info.get('support_signals', [])
                
                # Create support signal lookup
                support_ids = set()
                for support in support_signals:
                    support_ids.add(self._get_box_id(support['bbox']))
                
                for contact in contacts:
                    bbox = contact['bbox']
                    class_name = contact['class_name']
                    confidence = contact['confidence']
                    is_master = contact == master_contact
                    is_touching = contact.get('is_touching', False)
                    is_interpolated = contact.get('is_interpolated', False)
                    is_support = self._get_box_id(bbox) in support_ids
                    
                    # Get class color from constants (convert RGB to BGR for OpenCV)
                    class_color_rgb = RGBColors.CLASS_COLORS.get(class_name, RGBColors.GREY)
                    class_color = (class_color_rgb[2], class_color_rgb[1], class_color_rgb[0])  # BGR format
                    
                    # Different box styles based on status
                    if is_interpolated:
                        # Interpolated boxes: dashed orange overlay  
                        self._draw_dashed_rectangle(frame, bbox, (0, 165, 255), 2)  # Orange dashed
                        
                        # Draw motion vector if available
                        if 'motion_vector' in contact:
                            dx, dy = contact['motion_vector']
                            center = self._get_box_center(bbox)
                            end_point = (int(center[0] + dx * 10), int(center[1] + dy * 10))
                            cv2.arrowedLine(frame, (int(center[0]), int(center[1])), 
                                          end_point, (0, 165, 255), 2, tipLength=0.3)
                    elif is_master:
                        # MASTER touching: double thick border + bright outline
                        cv2.rectangle(frame, 
                                    (int(bbox[0]), int(bbox[1])), 
                                    (int(bbox[2]), int(bbox[3])),
                                    class_color, 4)  # Extra thick for master
                        
                        # Bright white outline for master
                        cv2.rectangle(frame, 
                                    (int(bbox[0]) - 2, int(bbox[1]) - 2), 
                                    (int(bbox[2]) + 2, int(bbox[3]) + 2),
                                    (255, 255, 255), 2)
                    elif is_support:
                        # Support signal: dashed class color border
                        self._draw_dashed_rectangle(frame, bbox, class_color, 2)
                    elif is_touching:
                        # Other touching: normal thick border
                        cv2.rectangle(frame, 
                                    (int(bbox[0]), int(bbox[1])), 
                                    (int(bbox[2]), int(bbox[3])),
                                    class_color, 2)  # Normal thickness
                    else:
                        # Detected but not touching: dotted corners only
                        self._draw_corner_points(frame, bbox, class_color)
                    
                    # Minimal text overlay - show status for important boxes
                    if is_master:
                        label = f"{class_name.upper()}★"  # Star for master
                        cv2.putText(frame, label, (int(bbox[0]), int(bbox[1]) - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    elif is_support:
                        label = f"{class_name.upper()}+"  # Plus for support
                        cv2.putText(frame, label, (int(bbox[0]), int(bbox[1]) - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, class_color, 1)
                    elif is_interpolated:
                        label = f"{class_name.upper()}~"  # Tilde for interpolated
                        cv2.putText(frame, label, (int(bbox[0]), int(bbox[1]) - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
            
            # Professional distance visualization
            if (self.show_distance_lines and locked_penis_box and 
                locked_penis_box['active'] and contact_info.get('primary_contact')):
                
                penis_center = self._get_box_center(locked_penis_box['bbox'])
                contact = contact_info['primary_contact']
                contact_center = self._get_box_center(contact['bbox'])
                
                # Distance line with different style for interpolated
                line_color = (0, 165, 255) if contact.get('interpolated') else (255, 255, 0)
                line_thickness = 2 if contact.get('interpolated') else 3
                
                if contact.get('interpolated'):
                    # Dashed line for interpolated contacts
                    self._draw_dashed_line(frame, 
                                         (int(penis_center[0]), int(penis_center[1])),
                                         (int(contact_center[0]), int(contact_center[1])),
                                         line_color, line_thickness)
                else:
                    # Solid line for real contacts
                    cv2.line(frame, 
                            (int(penis_center[0]), int(penis_center[1])),
                            (int(contact_center[0]), int(contact_center[1])),
                            line_color, line_thickness)
                
                # Distance line only - no text overlay to keep clean
                pass  # Line is already drawn above
            
            # Professional status panel
            self._draw_status_panel(frame, locked_penis_box, contact_info)
            
            return frame
            
        except Exception as e:
            self.logger.warning(f"Debug overlay creation failed: {e}")
            return frame
    
    def _get_box_center(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Get the center point of a bounding box."""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    
    def _draw_dashed_rectangle(self, frame: np.ndarray, bbox: Tuple[float, float, float, float], 
                             color: Tuple[int, int, int], thickness: int = 2, dash_length: int = 8):
        """Draw a dashed rectangle for interpolated contacts."""
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        # Top edge
        for x in range(x1, x2, dash_length * 2):
            cv2.line(frame, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
        
        # Bottom edge  
        for x in range(x1, x2, dash_length * 2):
            cv2.line(frame, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
        
        # Left edge
        for y in range(y1, y2, dash_length * 2):
            cv2.line(frame, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
        
        # Right edge
        for y in range(y1, y2, dash_length * 2):
            cv2.line(frame, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)
    
    def _draw_dashed_line(self, frame: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int],
                         color: Tuple[int, int, int], thickness: int = 2, dash_length: int = 8):
        """Draw a dashed line for interpolated distance indicators."""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Calculate line length and direction
        dx = x2 - x1
        dy = y2 - y1
        length = int(np.sqrt(dx*dx + dy*dy))
        
        if length == 0:
            return
        
        # Normalize direction
        dx_norm = dx / length
        dy_norm = dy / length
        
        # Draw dashed segments
        for i in range(0, length, dash_length * 2):
            start_x = int(x1 + i * dx_norm)
            start_y = int(y1 + i * dy_norm)
            end_x = int(x1 + min(i + dash_length, length) * dx_norm)
            end_y = int(y1 + min(i + dash_length, length) * dy_norm)
            
            cv2.line(frame, (start_x, start_y), (end_x, end_y), color, thickness)
    
    def _draw_corner_points(self, frame: np.ndarray, bbox: Tuple[float, float, float, float], 
                           color: Tuple[int, int, int], size: int = 8):
        """Draw corner points for detected but non-touching boxes."""
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        # Top-left corner
        cv2.line(frame, (x1, y1), (x1 + size, y1), color, 2)
        cv2.line(frame, (x1, y1), (x1, y1 + size), color, 2)
        
        # Top-right corner
        cv2.line(frame, (x2 - size, y1), (x2, y1), color, 2)
        cv2.line(frame, (x2, y1), (x2, y1 + size), color, 2)
        
        # Bottom-left corner
        cv2.line(frame, (x1, y2 - size), (x1, y2), color, 2)
        cv2.line(frame, (x1, y2), (x1 + size, y2), color, 2)
        
        # Bottom-right corner
        cv2.line(frame, (x2 - size, y2), (x2, y2), color, 2)
        cv2.line(frame, (x2, y2 - size), (x2, y2), color, 2)
    
    def _draw_status_panel(self, frame: np.ndarray, locked_penis_box: Dict, contact_info: Dict):
        """Draw professional status panel with comprehensive tracking information."""
        frame_h, frame_w = frame.shape[:2]
        
        # Status panel background
        panel_width = 300
        panel_height = 180
        panel_x = frame_w - panel_width - 10
        panel_y = 10
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Panel border
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), (100, 100, 100), 2)
        
        # Status items with improved contact statistics
        contacts = contact_info.get('contacts', [])
        touching_count = len([c for c in contacts if c.get('is_touching', False)])
        interpolated_count = len([c for c in contacts if c.get('is_interpolated', False)])
        
        # Get scene context information
        scene_context = contact_info.get('scene_context', {})
        scene_type = scene_context.get('scene_type', 'unknown')
        scene_confidence = scene_context.get('confidence', 0.0)
        
        status_items = [
            f"Frame: {self.frame_count}",
            f"Penis: {'LOCKED' if self.locked_penis_tracker.active else 'SEARCHING'}",
            f"Height 95th: {self.locked_penis_tracker.computed_height_95th:.1f}px",
            f"Scene: {scene_type.upper()} ({scene_confidence:.0%})" if scene_type != 'unknown' else "Scene: ANALYZING",
            f"Total Objects: {len(contacts)}",
            f"Touching: {touching_count}",
            f"Interpolated: {interpolated_count}",
            f"Primary: {contact_info.get('primary_contact', {}).get('class_name', 'None')}",
            f"Position: {self.last_primary_position:.1f}"
        ]
        
        # Draw status text
        text_y = panel_y + 20
        for item in status_items:
            cv2.putText(frame, item, (panel_x + 10, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            text_y += 20
    
    def get_status(self) -> Dict[str, Any]:
        """Get current tracker status."""
        return {
            "tracker": self.metadata.display_name,
            "active": self.tracking_active,
            "initialized": hasattr(self, 'yolo_model'),
            "frame_count": self.frame_count,
            "locked_penis_active": self.locked_penis_tracker.active if hasattr(self, 'locked_penis_tracker') else False,
            "position": self.last_primary_position,
            "cache_size": len(self.contact_cache) if hasattr(self, 'contact_cache') else 0
        }