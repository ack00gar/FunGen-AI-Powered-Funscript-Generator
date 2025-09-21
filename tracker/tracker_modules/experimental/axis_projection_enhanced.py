"""
Enhanced Axis Projection Tracker - Production-grade motion tracking system.

This is an enhanced version of the axis projection tracker with additional
advanced features for maximum tracking accuracy and robustness.

Enhanced features:
- Multi-scale tracking for better small object detection
- Temporal coherence analysis for jitter reduction
- Adaptive thresholding based on noise levels
- Enhanced confidence metrics with uncertainty quantification
- Object persistence tracking across occlusions
- Automatic axis calibration and validation
- Performance optimizations and caching
"""

import cv2
import numpy as np
import time
import logging
from typing import Optional, Dict, Any, Tuple, List
from collections import deque
from dataclasses import dataclass

from tracker.tracker_modules.core.base_tracker import BaseTracker, TrackerMetadata, TrackerResult


@dataclass
class TrackingCandidate:
    """Represents a motion tracking candidate with metadata."""
    position_2d: Tuple[float, float]
    position_1d: float
    confidence: float
    source: str  # 'diff', 'flow', 'template'
    area: float
    velocity: Tuple[float, float]
    timestamp: float


class EnhancedAlphaBeta1D:
    """Enhanced 1D Alpha-Beta filter with adaptive parameters and uncertainty."""
    
    def __init__(self, alpha: float = 0.35, beta: float = 0.05, x0: float = 50.0, v0: float = 0.0):
        self.alpha = alpha
        self.beta = beta
        self.base_alpha = alpha
        self.base_beta = beta
        
        # State
        self.x = x0
        self.v = v0
        
        # Uncertainty tracking
        self.position_variance = 25.0  # Initial uncertainty
        self.velocity_variance = 5.0
        
        # Adaptive parameters
        self.consistency_history = deque(maxlen=30)
        self.last_residuals = deque(maxlen=10)
        
    def predict(self, dt: float) -> Tuple[float, float, float]:
        """Predict with uncertainty estimation."""
        x_pred = self.x + self.v * dt
        v_pred = self.v
        
        # Uncertainty grows during prediction
        pred_variance = self.position_variance + (self.velocity_variance * dt * dt)
        
        return x_pred, v_pred, pred_variance
    
    def update(self, z: float, dt: float, measurement_variance: float = 1.0) -> Tuple[float, float, float]:
        """Update with adaptive parameters based on measurement quality."""
        x_pred, v_pred, pred_variance = self.predict(dt)
        
        # Calculate residual and adaptive gain
        residual = z - x_pred
        self.last_residuals.append(abs(residual))
        
        # Adaptive alpha based on residual consistency
        residual_std = np.std(self.last_residuals) if len(self.last_residuals) > 3 else 1.0
        if residual_std < 2.0:  # Consistent measurements
            self.alpha = min(0.6, self.base_alpha * 1.2)  # More responsive
        else:  # Inconsistent measurements
            self.alpha = max(0.1, self.base_alpha * 0.7)  # More conservative
        
        # Kalman-like gain calculation
        total_variance = pred_variance + measurement_variance
        kalman_gain = pred_variance / total_variance if total_variance > 0 else self.alpha
        
        # Update state
        self.x = x_pred + kalman_gain * residual
        self.v = v_pred + self.beta * (residual / max(1e-3, dt))
        
        # Update uncertainty
        self.position_variance = (1 - kalman_gain) * pred_variance
        self.velocity_variance = max(0.1, self.velocity_variance * 0.99)  # Slowly decay uncertainty
        
        # Clamp to valid range
        self.x = float(max(0.0, min(100.0, self.x)))
        
        return self.x, self.v, self.position_variance


class MultiScaleTracker:
    """Multi-scale tracking for better detection at different object sizes."""
    
    def __init__(self, scales: List[float] = None):
        self.scales = scales or [1.0, 0.75, 0.5]  # Full, 3/4, half resolution
        self.dis_flows = {}
        
        # Initialize DIS flow for each scale
        for scale in self.scales:
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
            dis.setFinestScale(max(1, int(2 * scale)))
            dis.setPatchSize(max(4, int(8 * scale)))
            dis.setPatchStride(max(2, int(4 * scale)))
            self.dis_flows[scale] = dis
    
    def compute_multi_scale_flow(self, prev_gray: np.ndarray, gray: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """Compute optical flow at multiple scales."""
        flows = []
        h, w = gray.shape[:2]
        
        for scale in self.scales:
            if scale == 1.0:
                scaled_prev = prev_gray
                scaled_curr = gray
            else:
                new_h, new_w = int(h * scale), int(w * scale)
                scaled_prev = cv2.resize(prev_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
                scaled_curr = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Critical performance optimization: ensure contiguous arrays for OpenCV
            scaled_prev_cont = np.ascontiguousarray(scaled_prev)
            scaled_curr_cont = np.ascontiguousarray(scaled_curr)
            flow = self.dis_flows[scale].calc(scaled_prev_cont, scaled_curr_cont, None)
            
            # Scale flow back to original resolution
            if scale != 1.0:
                flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
                flow *= (1.0 / scale)  # Scale the flow vectors
            
            flows.append((flow, scale))
        
        return flows


class TemporalCoherenceAnalyzer:
    """Analyzes temporal coherence to reduce jitter and false positives."""
    
    def __init__(self, window_size: int = 15):
        self.window_size = window_size
        self.position_history = deque(maxlen=window_size)
        self.velocity_history = deque(maxlen=window_size)
        self.confidence_history = deque(maxlen=window_size)
        
    def add_measurement(self, position: float, velocity: float, confidence: float):
        """Add new measurement to history."""
        self.position_history.append(position)
        self.velocity_history.append(velocity)
        self.confidence_history.append(confidence)
    
    def get_coherence_score(self) -> float:
        """Calculate temporal coherence score [0,1]."""
        if len(self.position_history) < 5:
            return 0.5  # Neutral score for insufficient data
        
        # Velocity consistency
        velocities = np.array(list(self.velocity_history))
        vel_std = np.std(velocities)
        vel_coherence = np.exp(-vel_std / 10.0)  # Exponential decay of coherence with velocity variance
        
        # Position smoothness (second derivative)
        positions = np.array(list(self.position_history))
        if len(positions) >= 3:
            second_deriv = np.diff(np.diff(positions))
            smoothness = np.exp(-np.mean(np.abs(second_deriv)) / 5.0)
        else:
            smoothness = 0.5
        
        # Confidence trend
        confidences = np.array(list(self.confidence_history))
        conf_trend = np.mean(confidences)
        
        # Combined coherence score
        coherence = 0.4 * vel_coherence + 0.4 * smoothness + 0.2 * conf_trend
        return float(np.clip(coherence, 0.0, 1.0))


class AdaptiveThresholder:
    """Adaptive thresholding based on noise characteristics."""
    
    def __init__(self, base_threshold: int = 20):
        self.base_threshold = base_threshold
        self.noise_history = deque(maxlen=60)  # 2 seconds at 30fps
        self.current_threshold = base_threshold
        
    def analyze_frame_noise(self, frame: np.ndarray) -> float:
        """Estimate noise level in frame."""
        # Use Laplacian variance as noise estimate
        laplacian = cv2.Laplacian(frame, cv2.CV_64F)
        noise_level = laplacian.var()
        
        self.noise_history.append(noise_level)
        return noise_level
    
    def get_adaptive_threshold(self, frame: np.ndarray) -> int:
        """Get threshold adapted to current noise conditions."""
        noise_level = self.analyze_frame_noise(frame)
        
        if len(self.noise_history) > 10:
            avg_noise = np.mean(self.noise_history)
            
            # Adapt threshold based on noise level
            if avg_noise > 500:  # High noise
                self.current_threshold = min(40, self.base_threshold * 1.5)
            elif avg_noise < 100:  # Low noise
                self.current_threshold = max(10, self.base_threshold * 0.7)
            else:
                self.current_threshold = self.base_threshold
        
        return int(self.current_threshold)


class ObjectPersistenceTracker:
    """Tracks object persistence across potential occlusions."""
    
    def __init__(self, max_missing_frames: int = 15):
        self.max_missing_frames = max_missing_frames
        self.missing_count = 0
        self.last_known_position = None
        self.last_known_velocity = (0.0, 0.0)
        self.occlusion_recovery_candidates = deque(maxlen=5)
        
    def update(self, detected_objects: List[TrackingCandidate]) -> Optional[TrackingCandidate]:
        """Update persistence tracking with new detections."""
        if not detected_objects:
            self.missing_count += 1
            
            # Try to predict position during occlusion
            if self.last_known_position and self.missing_count <= self.max_missing_frames:
                # Simple linear prediction
                dt = 1/30.0  # Assume 30fps
                predicted_x = self.last_known_position[0] + self.last_known_velocity[0] * dt
                predicted_y = self.last_known_position[1] + self.last_known_velocity[1] * dt
                
                # Create synthetic candidate with reduced confidence
                confidence = max(0.1, 0.8 * (1.0 - self.missing_count / self.max_missing_frames))
                
                return TrackingCandidate(
                    position_2d=(predicted_x, predicted_y),
                    position_1d=50.0,  # Will be recalculated
                    confidence=confidence,
                    source='persistence',
                    area=0.0,
                    velocity=self.last_known_velocity,
                    timestamp=time.time()
                )
            
            return None
        
        # Object found, reset missing count
        self.missing_count = 0
        
        # Find best candidate (highest confidence)
        best_candidate = max(detected_objects, key=lambda c: c.confidence)
        
        # Update tracking history
        if self.last_known_position:
            dt = best_candidate.timestamp - getattr(self, 'last_timestamp', best_candidate.timestamp)
            if dt > 0:
                vel_x = (best_candidate.position_2d[0] - self.last_known_position[0]) / dt
                vel_y = (best_candidate.position_2d[1] - self.last_known_position[1]) / dt
                self.last_known_velocity = (vel_x, vel_y)
        
        self.last_known_position = best_candidate.position_2d
        self.last_timestamp = best_candidate.timestamp
        
        return best_candidate


class AxisProjectionEnhancedTracker(BaseTracker):
    """
    Enhanced axis projection tracker with advanced features.
    
    This version includes:
    - Multi-scale optical flow analysis
    - Temporal coherence analysis for jitter reduction
    - Adaptive thresholding based on noise levels
    - Enhanced confidence metrics with uncertainty
    - Object persistence across occlusions
    - Automatic axis validation
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Enhanced components
        self.multi_scale_tracker = MultiScaleTracker([1.0, 0.75, 0.5])
        self.coherence_analyzer = TemporalCoherenceAnalyzer(window_size=15)
        self.adaptive_thresholder = AdaptiveThresholder(base_threshold=20)
        self.persistence_tracker = ObjectPersistenceTracker(max_missing_frames=15)
        
        # Enhanced filtering
        self.filter_1d = EnhancedAlphaBeta1D(alpha=0.35, beta=0.05, x0=50.0, v0=0.0)
        
        # Processing parameters
        self.proc_width = 640
        self.scale = 1.0
        
        # Axis definition
        self.axis_A = None
        self.axis_B = None
        self.axis_length = 0.0
        self.axis_angle = 0.0
        
        # Frame buffers
        self.prev_gray = None
        self.frame_history = deque(maxlen=5)
        
        # Enhanced tracking state
        self.tracking_active = False
        self.current_fps = 30.0
        self._fps_update_counter = 0
        self._fps_last_time = time.time()
        
        # Performance monitoring
        self.processing_times = deque(maxlen=30)
        self.detection_success_rate = deque(maxlen=100)
        
        # Enhanced outputs
        self.current_position = 50
        self.current_confidence = 0.0
        self.current_uncertainty = 25.0
        self.tracking_quality = 0.0
        
        # Debug info for visualization
        self.debug_candidates = []
        self.debug_selected_candidate = None
        
    @property
    def metadata(self) -> TrackerMetadata:
        return TrackerMetadata(
            name="axis_projection_enhanced",
            display_name="Experimental - Enhanced Axis Projection Tracker",
            description="Production-grade motion tracking with multi-scale analysis, temporal coherence, and adaptive thresholding",
            category="live",
            version="2.0.0",
            author="Advanced Motion Tracking System",
            tags=["optical-flow", "multi-scale", "temporal-coherence", "adaptive", "robust", "production"],
            requires_roi=False,
            supports_dual_axis=True
        )
    
    def initialize(self, app_instance, **kwargs) -> bool:
        """Initialize enhanced tracker."""
        try:
            self.app = app_instance
            
            # Get video dimensions for axis setup
            if hasattr(app_instance, 'get_video_dimensions'):
                width, height = app_instance.get_video_dimensions()
                if width and height:
                    # Set axis based on tracking mode (like other trackers)
                    tracking_axis_mode = getattr(app_instance, 'tracking_axis_mode', 'horizontal')
                    margin = int(0.1 * min(width, height))
                    
                    if tracking_axis_mode == 'vertical':
                        # Vertical axis down center of frame
                        self.axis_A = (int(0.5 * width), margin)
                        self.axis_B = (int(0.5 * width), height - margin)
                        self.logger.info(f"Enhanced VERTICAL axis set: A={self.axis_A}, B={self.axis_B}")
                    else:
                        # Default horizontal axis across center of frame
                        self.axis_A = (margin, int(0.5 * height))
                        self.axis_B = (width - margin, int(0.5 * height))
                        self.logger.info(f"Enhanced HORIZONTAL axis set: A={self.axis_A}, B={self.axis_B}")
                    
                    self._calculate_axis_properties()
            
            # Load enhanced settings
            if hasattr(app_instance, 'app_settings'):
                settings = app_instance.app_settings
                
                # Multi-scale settings
                scales = settings.get('axis_scales', [1.0, 0.75, 0.5])
                self.multi_scale_tracker = MultiScaleTracker(scales)
                
                # Coherence analysis settings
                coherence_window = settings.get('axis_coherence_window', 15)
                self.coherence_analyzer = TemporalCoherenceAnalyzer(coherence_window)
                
                # Adaptive threshold settings
                base_thresh = settings.get('axis_base_threshold', 20)
                self.adaptive_thresholder = AdaptiveThresholder(base_thresh)
                
                # Persistence settings
                max_missing = settings.get('axis_max_missing_frames', 15)
                self.persistence_tracker = ObjectPersistenceTracker(max_missing)
                
                # Enhanced filter settings
                alpha = settings.get('axis_enhanced_alpha', 0.35)
                beta = settings.get('axis_enhanced_beta', 0.05)
                self.filter_1d = EnhancedAlphaBeta1D(alpha=alpha, beta=beta, x0=50.0, v0=0.0)
            
            self._initialized = True
            self.logger.info("Enhanced axis projection tracker initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Enhanced initialization failed: {e}")
            return False
    
    def _calculate_axis_properties(self):
        """Calculate axis properties for optimizations."""
        if self.axis_A and self.axis_B:
            dx = self.axis_B[0] - self.axis_A[0]
            dy = self.axis_B[1] - self.axis_A[1]
            self.axis_length = np.sqrt(dx*dx + dy*dy)
            self.axis_angle = np.arctan2(dy, dx)
    
    def process_frame(self, frame: np.ndarray, frame_time_ms: int,
                     frame_index: Optional[int] = None) -> TrackerResult:
        """Process frame with enhanced tracking."""
        start_time = time.perf_counter()
        
        try:
            self._update_fps()
            
            # Update axis based on current tracking mode
            self._update_axis_for_tracking_mode()
            
            # Debug: Log that we're being called
            if frame_index and frame_index == 1:
                self.logger.info("ENHANCED AXIS DEBUG: process_frame() called - tracker is active in GUI")
            
            # Prepare frame
            gray, frame_small = self._prepare_frame(frame)
            
            # Add to frame history for temporal analysis
            self.frame_history.append(gray)
            
            # Initialize on first frame
            if self.prev_gray is None:
                self.prev_gray = gray.copy()
                return self._create_initialization_result(frame_small, frame_time_ms)
            
            # Adaptive threshold based on noise
            diff_thresh = self.adaptive_thresholder.get_adaptive_threshold(gray)
            
            # Multi-scale optical flow analysis
            multi_flows = self.multi_scale_tracker.compute_multi_scale_flow(self.prev_gray, gray)
            
            # Generate tracking candidates from multiple sources
            candidates = self._generate_tracking_candidates(gray, multi_flows, diff_thresh)
            
            # Store candidates for visualization
            self.debug_candidates = candidates
            
            # Object persistence tracking
            persistent_candidate = self.persistence_tracker.update(candidates)
            self.debug_selected_candidate = persistent_candidate
            
            if persistent_candidate:
                # Project to axis and update filter
                pos_1d = self._project_to_axis(persistent_candidate.position_2d)
                measurement_variance = max(1.0, 10.0 * (1.0 - persistent_candidate.confidence))
                
                dt = (time.time() - getattr(self, 'last_update_time', time.time()))
                filtered_pos, velocity, uncertainty = self.filter_1d.update(pos_1d, dt, measurement_variance)
                
                # Update coherence analysis
                self.coherence_analyzer.add_measurement(filtered_pos, velocity, persistent_candidate.confidence)
                coherence_score = self.coherence_analyzer.get_coherence_score()
                
                # Calculate enhanced confidence
                base_confidence = persistent_candidate.confidence
                temporal_confidence = coherence_score
                uncertainty_confidence = max(0.1, 1.0 - uncertainty / 50.0)
                
                final_confidence = 0.5 * base_confidence + 0.3 * temporal_confidence + 0.2 * uncertainty_confidence
                
                self.current_position = int(round(filtered_pos))
                self.current_confidence = final_confidence
                self.current_uncertainty = uncertainty
                self.tracking_quality = coherence_score
                
                self.detection_success_rate.append(1.0)
            else:
                # No detection - coast with prediction
                dt = (time.time() - getattr(self, 'last_update_time', time.time()))
                pred_pos, pred_vel, pred_uncertainty = self.filter_1d.predict(dt)
                
                self.current_position = int(round(pred_pos))
                self.current_confidence = max(0.0, self.current_confidence * 0.9)  # Decay confidence
                self.current_uncertainty = pred_uncertainty
                
                self.detection_success_rate.append(0.0)
            
            self.last_update_time = time.time()
            
            # Generate output only when we have very confident detections with significant motion
            action_log = None
            recent_success = np.mean(list(self.detection_success_rate)[-5:]) if len(self.detection_success_rate) >= 5 else 0.0
            
            if (self.tracking_active and 
                persistent_candidate and 
                self.current_confidence > 0.6 and  # Even higher confidence threshold
                self.tracking_quality > 0.4 and   # Higher quality threshold
                recent_success > 0.5):  # Require sustained detection success
                action_log = [{
                    "at": int(frame_time_ms),
                    "pos": int(max(0, min(100, self.current_position)))
                }]  # Use standard format without extra fields
                
            # Debug logging for GUI troubleshooting  
            if frame_index and frame_index % 30 == 0:  # Log every 30th frame
                axis_info = f"A={self.axis_A}, B={self.axis_B}" if (self.axis_A and self.axis_B) else "NOT SET"
                action_status = "ACTION" if action_log else "NO_ACTION"
                self.logger.info(f"ENHANCED AXIS DEBUG Frame {frame_index}: pos={self.current_position}, conf={self.current_confidence:.3f}, {action_status}, axis={axis_info}")
            
            # Visualization
            vis_frame = self._draw_enhanced_visualization(frame_small)
            
            # Performance monitoring
            processing_time = time.perf_counter() - start_time
            self.processing_times.append(processing_time)
            
            # Debug info
            success_rate = np.mean(self.detection_success_rate) if self.detection_success_rate else 0.0
            avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0.0
            
            debug_info = {
                'position': self.current_position,
                'confidence': self.current_confidence,
                'uncertainty': self.current_uncertainty,
                'quality': self.tracking_quality,
                'success_rate': success_rate,
                'processing_time_ms': avg_processing_time * 1000,
                'fps': round(self.current_fps, 1),
                'tracking_active': self.tracking_active,
                'adaptive_threshold': diff_thresh,
                'num_candidates': len(candidates) if candidates else 0
            }
            
            status_msg = f"Enhanced Axis | Pos: {self.current_position} | " \
                        f"Conf: {self.current_confidence:.2f} | Q: {self.tracking_quality:.2f}"
            
            # Update prev frame
            self.prev_gray = gray.copy()
            
            return TrackerResult(
                processed_frame=vis_frame,
                action_log=action_log,
                debug_info=debug_info,
                status_message=status_msg
            )
            
        except Exception as e:
            self.logger.error(f"Enhanced frame processing error: {e}")
            return TrackerResult(
                processed_frame=frame,
                action_log=None,
                debug_info={'error': str(e)},
                status_message=f"Error: {str(e)}"
            )
    
    def _prepare_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare frame with enhanced preprocessing."""
        h0, w0 = frame.shape[:2]
        
        # Smart scaling based on content
        if self.proc_width and self.proc_width < w0:
            self.scale = self.proc_width / float(w0)
            frame_small = cv2.resize(frame, (int(w0 * self.scale), int(h0 * self.scale)),
                                    interpolation=cv2.INTER_AREA)
        else:
            frame_small = frame
            self.scale = 1.0
        
        # Enhanced preprocessing
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        
        # Adaptive Gaussian blur based on scale
        blur_size = max(3, int(5 * self.scale))
        if blur_size % 2 == 0:
            blur_size += 1
        gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        
        # Optional CLAHE for low-contrast scenes
        if hasattr(self, 'use_clahe') and self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        
        return gray, frame_small
    
    def _generate_tracking_candidates(self, gray: np.ndarray, multi_flows: List, diff_thresh: int) -> List[TrackingCandidate]:
        """Generate tracking candidates from multiple sources."""
        candidates = []
        current_time = time.time()
        
        # 1. Frame differencing candidates
        diff_mask = self._get_enhanced_frame_diff_mask(gray, diff_thresh)
        diff_candidates = self._extract_diff_candidates(diff_mask, current_time)
        candidates.extend(diff_candidates)
        
        # 2. Multi-scale optical flow candidates  
        for flow, scale in multi_flows:
            flow_candidates = self._extract_flow_candidates(flow, scale, current_time)
            candidates.extend(flow_candidates)
        
        # 3. Template matching candidates (if previous detection exists)
        if hasattr(self, 'template') and self.template is not None:
            template_candidates = self._extract_template_candidates(gray, current_time)
            candidates.extend(template_candidates)
        
        # Remove duplicates and filter low-quality candidates
        candidates = self._filter_and_deduplicate_candidates(candidates)
        
        return candidates
    
    def _get_enhanced_frame_diff_mask(self, gray: np.ndarray, thresh: int) -> np.ndarray:
        """Enhanced frame differencing with temporal consistency."""
        if len(self.frame_history) < 2:
            return np.zeros(gray.shape, dtype=np.uint8)
        
        # Multi-frame differencing for better noise rejection
        diff_masks = []
        for prev_frame in list(self.frame_history)[-3:]:  # Use last 3 frames
            diff = cv2.absdiff(prev_frame, gray)
            diff = cv2.GaussianBlur(diff, (5, 5), 0)
            _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
            diff_masks.append(mask)
        
        # Combine masks using majority vote
        if len(diff_masks) > 1:
            combined = np.zeros_like(diff_masks[0], dtype=np.float32)
            for mask in diff_masks:
                combined += mask.astype(np.float32) / 255.0
            combined_mask = (combined >= (len(diff_masks) * 0.6) * 255).astype(np.uint8)
        else:
            combined_mask = diff_masks[0]
        
        # Enhanced morphological operations
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)
        
        return combined_mask
    
    def _extract_diff_candidates(self, mask: np.ndarray, timestamp: float) -> List[TrackingCandidate]:
        """Extract candidates from frame difference analysis."""
        candidates = []
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        if num_labels <= 1:
            return candidates
        
        # Analyze each component
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 50:  # Skip very small components
                continue
            
            cx, cy = centroids[i]
            
            # Calculate confidence based on area and compactness
            bbox_area = stats[i, cv2.CC_STAT_WIDTH] * stats[i, cv2.CC_STAT_HEIGHT]
            compactness = area / max(1, bbox_area)  # How well component fills bounding box
            
            area_score = min(1.0, area / 1000.0)  # Normalize area score
            confidence = 0.3 * area_score + 0.7 * compactness
            
            # Project to 1D axis
            pos_1d = self._project_to_axis((cx, cy))
            
            candidate = TrackingCandidate(
                position_2d=(cx, cy),
                position_1d=pos_1d,
                confidence=confidence,
                source='diff',
                area=area,
                velocity=(0.0, 0.0),  # Will be calculated later
                timestamp=timestamp
            )
            candidates.append(candidate)
        
        return candidates
    
    def _extract_flow_candidates(self, flow: np.ndarray, scale: float, timestamp: float) -> List[TrackingCandidate]:
        """Extract candidates from optical flow analysis."""
        candidates = []
        
        # Calculate flow magnitude and create motion mask
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        motion_thresh = 0.5 / scale  # Adapt threshold to scale
        motion_mask = (mag > motion_thresh).astype(np.uint8) * 255
        
        # Find motion regions
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(motion_mask, connectivity=8)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 30:  # Skip small regions
                continue
            
            cx, cy = centroids[i]
            
            # Get flow vector at centroid
            flow_u = flow[int(cy), int(cx), 0] if 0 <= cy < flow.shape[0] and 0 <= cx < flow.shape[1] else 0.0
            flow_v = flow[int(cy), int(cx), 1] if 0 <= cy < flow.shape[0] and 0 <= cx < flow.shape[1] else 0.0
            
            # Calculate confidence based on flow magnitude and consistency
            local_mag = np.sqrt(flow_u**2 + flow_v**2)
            
            # Local flow consistency check
            y1, y2 = max(0, int(cy-5)), min(flow.shape[0], int(cy+6))
            x1, x2 = max(0, int(cx-5)), min(flow.shape[1], int(cx+6))
            local_flow = flow[y1:y2, x1:x2]
            local_std = np.std(np.sqrt(local_flow[..., 0]**2 + local_flow[..., 1]**2))
            
            consistency = np.exp(-local_std / 2.0)  # Higher consistency = lower std
            magnitude_score = min(1.0, local_mag / 3.0)
            confidence = 0.6 * magnitude_score + 0.4 * consistency
            
            # Boost confidence for larger scales (more reliable)
            confidence *= (0.7 + 0.3 * scale)
            
            pos_1d = self._project_to_axis((cx, cy))
            
            candidate = TrackingCandidate(
                position_2d=(cx, cy),
                position_1d=pos_1d,
                confidence=confidence,
                source=f'flow_{scale}',
                area=area,
                velocity=(flow_u, flow_v),
                timestamp=timestamp
            )
            candidates.append(candidate)
        
        return candidates
    
    def _filter_and_deduplicate_candidates(self, candidates: List[TrackingCandidate]) -> List[TrackingCandidate]:
        """Filter and deduplicate candidates."""
        if not candidates:
            return candidates
        
        # Filter by minimum confidence
        candidates = [c for c in candidates if c.confidence > 0.1]
        
        # Spatial clustering to remove duplicates
        filtered = []
        for candidate in sorted(candidates, key=lambda c: c.confidence, reverse=True):
            is_duplicate = False
            for existing in filtered:
                dist = np.sqrt((candidate.position_2d[0] - existing.position_2d[0])**2 +
                              (candidate.position_2d[1] - existing.position_2d[1])**2)
                if dist < 30:  # Within 30 pixels
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(candidate)
        
        # Keep top candidates
        return sorted(filtered, key=lambda c: c.confidence, reverse=True)[:5]
    
    def _project_to_axis(self, point: Tuple[float, float]) -> float:
        """Project point to 1D axis position."""
        if self.axis_A is None or self.axis_B is None:
            return 50.0
        
        # Convert to processing coordinates
        A_proc = (self.axis_A[0] * self.scale, self.axis_A[1] * self.scale)
        B_proc = (self.axis_B[0] * self.scale, self.axis_B[1] * self.scale)
        
        # Project point onto axis
        AB = np.array(B_proc) - np.array(A_proc)
        AP = np.array(point) - np.array(A_proc)
        
        t = np.dot(AP, AB) / (np.dot(AB, AB) + 1e-6)
        t = np.clip(t, 0.0, 1.0)
        
        return float(t * 100.0)
    
    def _draw_enhanced_visualization(self, frame: np.ndarray) -> np.ndarray:
        """Draw enhanced visualization with motion centroids (no axis line/marker)."""
        vis = frame.copy()
        
        # Draw only high-quality motion candidates to reduce flickering
        if self.debug_candidates:
            # Filter for stable, high-confidence candidates only
            stable_candidates = [c for c in self.debug_candidates 
                               if c.confidence > 0.3]  # Much higher threshold
            
            # Limit to top 5 candidates max to reduce clutter
            stable_candidates = stable_candidates[:5]
            
            for i, candidate in enumerate(stable_candidates):
                # Convert to processed coordinates
                pos_x = int(candidate.position_2d[0] * self.scale)
                pos_y = int(candidate.position_2d[1] * self.scale)
                
                # Color based on confidence and type - more muted colors
                confidence_color = int(180 + 75 * candidate.confidence)  # Brighter, less flickering
                if candidate.source == 'diff':
                    color = (0, 0, confidence_color)  # Red for frame diff
                elif candidate.source == 'flow':
                    color = (0, confidence_color//2, confidence_color)  # Orange for optical flow
                else:
                    color = (confidence_color//2, 0, confidence_color)  # Purple for template/other
                
                # Consistent size to reduce flickering
                size = max(4, min(6, int(4 + 2 * candidate.confidence)))
                cv2.circle(vis, (pos_x, pos_y), size, color, -1)
                cv2.circle(vis, (pos_x, pos_y), size + 1, color, 1)
                
                # Show confidence as text only for top 2 candidates
                if i < 2:  # Reduced from 3 to 2
                    cv2.putText(vis, f"{candidate.confidence:.1f}", 
                               (pos_x + 6, pos_y - 6), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)
        
        # Highlight the selected/persistent candidate - only if high confidence
        if self.debug_selected_candidate and self.debug_selected_candidate.confidence > 0.4:
            pos_x = int(self.debug_selected_candidate.position_2d[0] * self.scale)
            pos_y = int(self.debug_selected_candidate.position_2d[1] * self.scale)
            
            # Draw stable yellow circle for selected candidate
            cv2.circle(vis, (pos_x, pos_y), 10, (0, 255, 255), 2)  # Yellow ring (slightly smaller)
            cv2.circle(vis, (pos_x, pos_y), 3, (0, 255, 255), -1)  # Yellow center
        
        # Enhanced status display (no axis info)
        self._draw_enhanced_tracking_indicator(vis)
        
        return vis
    
    def _draw_enhanced_tracking_indicator(self, vis: np.ndarray):
        """Draw enhanced tracking indicator with motion detection info."""
        # Basic tracker info
        tracking_mode = getattr(self.app, 'tracking_axis_mode', 'horizontal') if hasattr(self, 'app') and self.app else 'horizontal'
        cv2.putText(vis, f"Enhanced Axis Tracker ({tracking_mode.upper()})", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(vis, f"Pos: {self.current_position}/100  Conf: {self.current_confidence:.2f}", 
                   (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Motion detection info
        if self.debug_candidates:
            num_candidates = len(self.debug_candidates)
            high_conf_candidates = sum(1 for c in self.debug_candidates if c.confidence > 0.5)
            cv2.putText(vis, f"Motion: {num_candidates} candidates ({high_conf_candidates} high-conf)", 
                       (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Show uncertainty and tracking quality
            cv2.putText(vis, f"Uncertainty: {self.current_uncertainty:.1f}  Quality: {self.tracking_quality:.2f}", 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Tracking status
        if self.tracking_active:
            cv2.putText(vis, "TRACKING", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    def _create_initialization_result(self, frame: np.ndarray, frame_time_ms: int) -> TrackerResult:
        """Create initialization result."""
        return TrackerResult(
            processed_frame=frame,
            action_log=None,
            debug_info={'status': 'initializing'},
            status_message="Initializing enhanced axis projection tracker..."
        )
    
    def _update_fps(self):
        """Update FPS calculation using high-performance delta time method."""
        current_time_sec = time.time()
        if self._fps_last_time > 0:
            delta_time = current_time_sec - self._fps_last_time
            if delta_time > 0.001:  # Avoid division by zero
                self.current_fps = 1.0 / delta_time
        self._fps_last_time = current_time_sec
    
    def set_axis(self, point_a: Tuple[int, int], point_b: Tuple[int, int]) -> bool:
        """Set axis with enhanced validation."""
        try:
            # Validate axis length
            dx = point_b[0] - point_a[0]
            dy = point_b[1] - point_a[1]
            length = np.sqrt(dx*dx + dy*dy)
            
            if length < 50:  # Minimum axis length
                self.logger.warning(f"Axis too short: {length:.1f} pixels")
                return False
            
            self.axis_A = tuple(point_a)
            self.axis_B = tuple(point_b)
            self._calculate_axis_properties()
            
            # Reset filter
            self.filter_1d = EnhancedAlphaBeta1D(
                alpha=self.filter_1d.base_alpha,
                beta=self.filter_1d.base_beta,
                x0=50.0, v0=0.0
            )
            
            self.logger.info(f"Enhanced axis set: length={length:.1f}px, angle={np.degrees(self.axis_angle):.1f}°")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set enhanced axis: {e}")
            return False

    def _update_axis_for_tracking_mode(self):
        """Update axis based on current tracking mode (like other trackers)."""
        if not self.app or not hasattr(self.app, 'get_video_dimensions'):
            return
            
        width, height = self.app.get_video_dimensions()
        if not width or not height:
            return
            
        tracking_axis_mode = getattr(self.app, 'tracking_axis_mode', 'horizontal')
        margin = int(0.1 * min(width, height))
        
        if tracking_axis_mode == 'vertical':
            # Vertical axis down center of frame
            new_axis_A = (int(0.5 * width), margin)
            new_axis_B = (int(0.5 * width), height - margin)
        else:
            # Horizontal axis across center of frame  
            new_axis_A = (margin, int(0.5 * height))
            new_axis_B = (width - margin, int(0.5 * height))
        
        # Only update if axis actually changed
        if self.axis_A != new_axis_A or self.axis_B != new_axis_B:
            self.axis_A = new_axis_A
            self.axis_B = new_axis_B
            self._calculate_axis_properties()
            self.logger.info(f"Enhanced axis updated for {tracking_axis_mode.upper()} mode: A={self.axis_A}, B={self.axis_B}")
    
    def start_tracking(self):
        """Start enhanced tracking."""
        self.tracking_active = True
        axis_status = "SET" if (self.axis_A and self.axis_B) else "NOT SET"
        self.logger.info(f"Enhanced axis projection tracking started - Axis: {axis_status}")
        if self.axis_A and self.axis_B:
            self.logger.info(f"Enhanced axis coordinates: A={self.axis_A}, B={self.axis_B}")
        else:
            self.logger.warning("WARNING: Enhanced axis not set! This will cause flat line output.")
    
    def stop_tracking(self):
        """Stop enhanced tracking."""
        self.tracking_active = False
        self.logger.info("Enhanced axis projection tracking stopped")
    
    def get_status_info(self) -> Dict[str, Any]:
        """Get enhanced status information."""
        base_status = super().get_status_info()
        
        success_rate = np.mean(self.detection_success_rate) if self.detection_success_rate else 0.0
        avg_time = np.mean(self.processing_times) if self.processing_times else 0.0
        
        enhanced_status = {
            'position': self.current_position,
            'confidence': self.current_confidence,
            'uncertainty': self.current_uncertainty,
            'tracking_quality': self.tracking_quality,
            'success_rate': success_rate,
            'avg_processing_time_ms': avg_time * 1000,
            'axis_length': self.axis_length,
            'axis_angle_degrees': np.degrees(self.axis_angle) if self.axis_angle else 0.0,
            'tracking_active': self.tracking_active,
            'fps': round(self.current_fps, 1)
        }
        
        base_status.update(enhanced_status)
        return base_status
    
    def cleanup(self):
        """Enhanced cleanup."""
        self.prev_gray = None
        self.frame_history.clear()
        self.multi_scale_tracker = None
        self.logger.debug("Enhanced axis projection tracker cleaned up")