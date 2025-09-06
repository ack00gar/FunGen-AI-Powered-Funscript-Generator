"""
Axis Projection Tracker - Advanced motion tracking with frame differencing and optical flow.

This tracker combines frame differencing with DIS optical flow to track motion along
a user-defined axis. It projects 2D motion onto a 1D axis and applies alpha-beta
filtering for smooth, responsive tracking.

Key features:
- Frame differencing for motion detection
- DIS optical flow for precise displacement tracking
- Robust median flow estimation with outlier rejection
- Alpha-beta filtering for smooth output
- Confidence-based fusion of multiple motion cues
- Global motion compensation for camera movement
"""

import cv2
import numpy as np
import time
import logging
from typing import Optional, Dict, Any, Tuple, List
from collections import deque

from tracker_modules.core.base_tracker import BaseTracker, TrackerMetadata, TrackerResult


def project_point_to_axis_01(p: Tuple[float, float], A: Tuple[float, float], B: Tuple[float, float]) -> float:
    """
    Project point p onto axis A->B and return normalized position [0,1].
    
    Args:
        p: Point to project (x, y)
        A: Axis start point (x, y)
        B: Axis end point (x, y)
    
    Returns:
        Normalized position on axis [0, 1]
    """
    AB = np.array(B, dtype=np.float32) - np.array(A, dtype=np.float32)
    AP = np.array(p, dtype=np.float32) - np.array(A, dtype=np.float32)
    denom = np.dot(AB, AB) + 1e-6
    t = float(np.dot(AP, AB) / denom)
    t = max(0.0, min(1.0, t))
    return t


def to_0_100(t01: float) -> float:
    """Convert normalized [0,1] to funscript range [0,100]."""
    return float(100.0 * max(0.0, min(1.0, t01)))


def get_frame_diff_mask(prev_gray: np.ndarray, gray: np.ndarray, 
                        thresh: int = 25, k_open: int = 3, k_close: int = 5) -> np.ndarray:
    """
    Compute frame difference mask with morphological cleanup.
    
    Args:
        prev_gray: Previous grayscale frame
        gray: Current grayscale frame
        thresh: Threshold for motion detection
        k_open: Kernel size for opening (noise removal)
        k_close: Kernel size for closing (gap filling)
    
    Returns:
        Binary mask of motion areas
    """
    diff = cv2.absdiff(prev_gray, gray)
    # Slight blur to suppress sensor noise before threshold
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    
    if k_open > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open)))
    if k_close > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close)))
    return mask


def largest_component_centroid(mask: np.ndarray) -> Tuple[Optional[Tuple[float, float]], int, Optional[int]]:
    """
    Find centroid of largest connected component in binary mask.
    
    Args:
        mask: Binary mask
    
    Returns:
        Tuple of (centroid (x,y), area, component_id) or (None, 0, None) if no components
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    if num_labels <= 1:
        return None, 0, None
    
    # Skip label 0 (background), pick largest by area
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + np.argmax(areas)
    area = int(stats[idx, cv2.CC_STAT_AREA])
    cx, cy = centroids[idx]
    return (float(cx), float(cy)), area, idx


def compute_DIS_flow(prev_gray: np.ndarray, gray: np.ndarray, dis) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute DIS optical flow between frames.
    
    Args:
        prev_gray: Previous grayscale frame
        gray: Current grayscale frame
        dis: OpenCV DIS optical flow object
    
    Returns:
        Tuple of (flow, magnitude, angle)
    """
    flow = dis.calc(prev_gray, gray, None)  # HxWx2 float32
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
    return flow, mag, ang


def robust_median_flow(flow: np.ndarray, mask: Optional[np.ndarray] = None,
                       roi_center: Optional[Tuple[float, float]] = None, 
                       roi_radius: float = 60) -> Tuple[Tuple[float, float], float, float, int]:
    """
    Compute robust median flow with outlier rejection.
    
    Args:
        flow: Optical flow field (HxWx2)
        mask: Optional binary mask to limit analysis
        roi_center: Optional ROI center for local analysis
        roi_radius: Radius for ROI analysis
    
    Returns:
        Tuple of (median_flow (u,v), magnitude, inlier_ratio, sample_count)
    """
    u = flow[..., 0]
    v = flow[..., 1]
    
    if mask is not None:
        idx = mask > 0
    else:
        idx = np.ones(u.shape, dtype=bool)
    
    if roi_center is not None:
        h, w = u.shape
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        dx = X - roi_center[0]
        dy = Y - roi_center[1]
        roi_idx = (dx*dx + dy*dy) <= (roi_radius*roi_radius)
        idx = np.logical_and(idx, roi_idx)
    
    if not np.any(idx):
        return (0.0, 0.0), 0.0, 0.0, 0  # no data
    
    u_sel = u[idx].ravel()
    v_sel = v[idx].ravel()
    
    # Median as robust central tendency
    mu = float(np.median(u_sel))
    mv = float(np.median(v_sel))
    
    # Robust spread via MAD (Median Absolute Deviation)
    mad_u = float(np.median(np.abs(u_sel - mu)) + 1e-6)
    mad_v = float(np.median(np.abs(v_sel - mv)) + 1e-6)
    
    # Simple inlier count (within 3*MAD)
    inliers = np.logical_and(np.abs(u_sel - mu) <= 3*mad_u, 
                             np.abs(v_sel - mv) <= 3*mad_v)
    inlier_ratio = float(np.mean(inliers)) if u_sel.size > 0 else 0.0
    
    # Magnitude of median flow
    mag_med = float(np.hypot(mu, mv))
    
    return (mu, mv), mag_med, inlier_ratio, u_sel.size


class AlphaBeta1D:
    """
    1D Alpha-Beta filter for smooth position tracking.
    
    This is a simplified Kalman filter for constant velocity motion.
    """
    
    def __init__(self, alpha: float = 0.35, beta: float = 0.05, x0: float = 50.0, v0: float = 0.0):
        """
        Initialize filter.
        
        Args:
            alpha: Position smoothing factor (0-1, higher = more responsive)
            beta: Velocity smoothing factor (0-1, higher = more responsive)
            x0: Initial position
            v0: Initial velocity
        """
        self.alpha = alpha
        self.beta = beta
        self.x = x0
        self.v = v0
        self.last_time = None
    
    def predict(self, dt: float) -> Tuple[float, float]:
        """Predict next state based on current state and time delta."""
        x_pred = self.x + self.v * dt
        v_pred = self.v
        return x_pred, v_pred
    
    def update(self, z: float, dt: float) -> Tuple[float, float]:
        """
        Update filter with new measurement.
        
        Args:
            z: Measured position
            dt: Time delta since last update
        
        Returns:
            Tuple of (filtered_position, filtered_velocity)
        """
        x_pred, v_pred = self.predict(dt)
        r = z - x_pred  # residual
        self.x = x_pred + self.alpha * r
        self.v = v_pred + self.beta * (r / max(1e-3, dt))
        # Clamp to [0, 100]
        self.x = float(max(0.0, min(100.0, self.x)))
        return self.x, self.v


class AxisProjectionTracker(BaseTracker):
    """
    Advanced motion tracker using frame differencing and optical flow.
    
    This tracker combines multiple motion detection techniques:
    1. Frame differencing for coarse motion detection
    2. DIS optical flow for precise displacement tracking
    3. Robust median flow estimation with outlier rejection
    4. Alpha-beta filtering for smooth, responsive output
    5. Confidence-based fusion of multiple motion cues
    """
    
    def __init__(self, **kwargs):
        """Initialize the axis projection tracker."""
        super().__init__(**kwargs)
        
        # Processing parameters
        self.proc_width = 640  # Width to resize frames for processing
        self.scale = 1.0  # Scale factor from original to processed
        
        # Axis definition (will be set via set_axis or use defaults)
        self.axis_A = None  # Start point (x, y) in original coordinates
        self.axis_B = None  # End point (x, y) in original coordinates
        
        # Frame buffers
        self.prev_gray = None
        self.prev_pos_px = None  # 2D position in processed frame space
        
        # Optical flow
        self.dis = None
        self.flow_preset = cv2.DISOPTICAL_FLOW_PRESET_FAST
        
        # Filtering
        self.filter_1d = AlphaBeta1D(alpha=0.35, beta=0.1, x0=50.0, v0=0.0)
        self.last_time = None
        
        # Motion detection parameters
        self.diff_thresh = 20
        self.flow_thresh = 0.5
        self.roi_radius = 80
        
        # Activity tracking
        self.activity_smooth = 0.0
        self.ema_alpha = 0.2
        
        # Tracking state
        self.tracking_active = False
        self.current_fps = 30.0
        self._fps_update_counter = 0
        self._fps_last_time = time.time()
        
        # Visualization
        self.show_axis = True
        self.show_tracking_point = True
        
        # Output
        self.current_position = 50
        self.current_confidence = 0.0
    
    @property
    def metadata(self) -> TrackerMetadata:
        """Return metadata describing this tracker."""
        return TrackerMetadata(
            name="axis_projection",
            display_name="Axis Projection Tracker",
            description="Advanced motion tracking with frame differencing and optical flow projection onto user-defined axis",
            category="live",
            version="1.0.0",
            author="Motion Tracking System",
            tags=["optical-flow", "frame-diff", "projection", "alpha-beta", "robust"],
            requires_roi=False,
            supports_dual_axis=True
        )
    
    def initialize(self, app_instance, **kwargs) -> bool:
        """Initialize the tracker."""
        try:
            self.app = app_instance
            
            # Initialize DIS optical flow
            self.dis = cv2.DISOpticalFlow_create(self.flow_preset)
            
            # Configure DIS parameters for quality vs speed tradeoff
            self.dis.setFinestScale(2)
            self.dis.setPatchSize(8)
            self.dis.setPatchStride(4)
            self.dis.setGradientDescentIterations(12)
            self.dis.setUseMeanNormalization(True)
            self.dis.setUseSpatialPropagation(True)
            self.dis.setVariationalRefinementIterations(5)
            
            # Get video dimensions if available
            if hasattr(app_instance, 'get_video_dimensions'):
                width, height = app_instance.get_video_dimensions()
                if width and height:
                    # Default axis: horizontal line at center
                    self.axis_A = (int(0.1 * width), int(0.5 * height))
                    self.axis_B = (int(0.9 * width), int(0.5 * height))
                    self.logger.info(f"Default axis set: A={self.axis_A}, B={self.axis_B}")
            
            # Load settings if available
            if hasattr(app_instance, 'app_settings'):
                settings = app_instance.app_settings
                
                # Processing parameters
                self.proc_width = settings.get('axis_proc_width', 640)
                self.diff_thresh = settings.get('axis_diff_thresh', 20)
                self.flow_thresh = settings.get('axis_flow_thresh', 0.5)
                self.roi_radius = settings.get('axis_roi_radius', 80)
                
                # Filter parameters
                alpha = settings.get('axis_filter_alpha', 0.35)
                beta = settings.get('axis_filter_beta', 0.1)
                self.filter_1d = AlphaBeta1D(alpha=alpha, beta=beta, x0=50.0, v0=0.0)
                
                # Visualization
                self.show_axis = settings.get('axis_show_axis', True)
                self.show_tracking_point = settings.get('axis_show_tracking', True)
                
                self.logger.info(f"Axis projection settings loaded: proc_width={self.proc_width}, "
                               f"diff_thresh={self.diff_thresh}, alpha={alpha}, beta={beta}")
            
            # Reset state
            self.prev_gray = None
            self.prev_pos_px = None
            self.last_time = None
            self.activity_smooth = 0.0
            self.current_position = 50
            self.current_confidence = 0.0
            
            self._initialized = True
            self.logger.info("Axis projection tracker initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    def set_axis(self, point_a: Tuple[int, int], point_b: Tuple[int, int]) -> bool:
        """
        Set the projection axis endpoints.
        
        Args:
            point_a: Start point (x, y) in video coordinates
            point_b: End point (x, y) in video coordinates
        
        Returns:
            bool: True if axis was set successfully
        """
        try:
            self.axis_A = tuple(point_a)
            self.axis_B = tuple(point_b)
            self.logger.info(f"Projection axis set: A={self.axis_A}, B={self.axis_B}")
            
            # Reset filter to center position
            self.filter_1d = AlphaBeta1D(
                alpha=self.filter_1d.alpha,
                beta=self.filter_1d.beta,
                x0=50.0,
                v0=0.0
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set axis: {e}")
            return False
    
    def _prepare_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare frame for processing (resize and convert to grayscale).
        
        Args:
            frame: Input BGR frame
        
        Returns:
            Tuple of (grayscale, resized_frame)
        """
        h0, w0 = frame.shape[:2]
        
        if self.proc_width is not None and self.proc_width < w0:
            s = self.proc_width / float(w0)
            frame_small = cv2.resize(frame, (int(w0*s), int(h0*s)), 
                                    interpolation=cv2.INTER_AREA)
            self.scale = s
        else:
            frame_small = frame
            self.scale = 1.0
        
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return gray, frame_small
    
    def _original_to_proc_point(self, p: Tuple[float, float]) -> Tuple[float, float]:
        """Convert point from original to processed coordinates."""
        return (p[0] * self.scale, p[1] * self.scale)
    
    def _proc_to_original_point(self, p: Tuple[float, float]) -> Tuple[float, float]:
        """Convert point from processed to original coordinates."""
        inv = 1.0 / max(1e-6, self.scale)
        return (p[0] * inv, p[1] * inv)
    
    def _project_px_to_0_100(self, p_proc: Tuple[float, float]) -> float:
        """
        Project processed space point to [0,100] along axis.
        
        Args:
            p_proc: Point in processed space
        
        Returns:
            Position on axis [0, 100]
        """
        if self.axis_A is None or self.axis_B is None:
            return 50.0  # Default center if no axis defined
        
        A_proc = self._original_to_proc_point(self.axis_A)
        B_proc = self._original_to_proc_point(self.axis_B)
        t01 = project_point_to_axis_01(p_proc, A_proc, B_proc)
        return to_0_100(t01)
    
    def _update_fps(self):
        """Update FPS tracking."""
        self._fps_update_counter += 1
        if self._fps_update_counter >= 30:
            current_time = time.time()
            if self._fps_last_time > 0:
                self.current_fps = 30.0 / (current_time - self._fps_last_time)
            self._fps_last_time = current_time
            self._fps_update_counter = 0
    
    def process_frame(self, frame: np.ndarray, frame_time_ms: int,
                     frame_index: Optional[int] = None) -> TrackerResult:
        """
        Process a frame using axis projection tracking.
        
        This implementation combines:
        1. Frame differencing for motion detection
        2. DIS optical flow for displacement tracking
        3. Robust median flow with outlier rejection
        4. Projection onto user-defined axis
        5. Alpha-beta filtering for smooth output
        """
        try:
            self._update_fps()
            
            now = time.time()
            gray, frame_small = self._prepare_frame(frame)
            h, w = gray.shape[:2]
            
            # Calculate time delta
            dt = 1/30.0
            if self.last_time is not None:
                dt = max(1e-3, now - self.last_time)
            self.last_time = now
            
            # Initialize on first frame
            if self.prev_gray is None:
                self.prev_gray = gray.copy()
                # Initialize position at center
                self.prev_pos_px = (w * 0.5, h * 0.5)
                x_smoothed, _ = self.filter_1d.update(50.0, dt)
                self.current_position = int(round(x_smoothed))
                self.current_confidence = 0.0
                
                return TrackerResult(
                    processed_frame=frame_small,
                    action_log=None,
                    debug_info={'status': 'initializing'},
                    status_message="Initializing axis projection tracker..."
                )
            
            # 1) Frame differencing mask
            diff_mask = get_frame_diff_mask(self.prev_gray, gray, 
                                           thresh=self.diff_thresh, k_open=3, k_close=7)
            centroid_d, area_d, _ = largest_component_centroid(diff_mask)
            
            # Confidence from diff: area fraction
            area_frac = float(area_d) / float(w * h) if area_d > 0 else 0.0
            conf_d = min(1.0, 5.0 * area_frac)  # boost small areas less
            
            # 2) DIS optical flow
            flow, mag, _ = compute_DIS_flow(self.prev_gray, gray, self.dis)
            
            # Motion mask from flow
            motion_mask_flow = (mag > self.flow_thresh).astype(np.uint8) * 255
            
            # Global motion compensation (subtract median flow over entire frame)
            (mu_glob, mv_glob), mag_med_glob, inlier_ratio_glob, _ = robust_median_flow(
                flow, None, None, 0
            )
            flow[..., 0] -= mu_glob
            flow[..., 1] -= mv_glob
            mag = np.hypot(flow[..., 0], flow[..., 1])
            
            # Combine masks to focus on moving subject
            combined_mask = cv2.bitwise_or(diff_mask, motion_mask_flow)
            
            # 3) Optical flow displacement near previous position
            roi_center = (int(self.prev_pos_px[0]), int(self.prev_pos_px[1]))
            (du, dv), mag_med_local, inlier_ratio_local, sample_count = robust_median_flow(
                flow, combined_mask, roi_center=roi_center, roi_radius=self.roi_radius
            )
            
            pos_of = (self.prev_pos_px[0] + du, self.prev_pos_px[1] + dv)
            conf_of = 0.0
            if sample_count > 100:
                # Confidence improves with magnitude and inlier ratio
                conf_of = float(min(1.0, 0.5*inlier_ratio_local + 0.5*min(1.0, mag_med_local/2.0)))
            
            # 4) Position candidate from frame differencing centroid
            pos_d = None
            if centroid_d is not None:
                pos_d = centroid_d
            
            # 5) Choose/fuse 2D positions -> project to 0..100
            candidates = []
            weights = []
            
            if pos_d is not None and conf_d > 0.02:
                x_d = self._project_px_to_0_100(pos_d)
                candidates.append(x_d)
                weights.append(conf_d)
            
            if conf_of > 0.02:
                x_of = self._project_px_to_0_100(pos_of)
                candidates.append(x_of)
                weights.append(conf_of)
            
            if len(candidates) == 0:
                # No good measurements; coast with prediction
                x_pred, _ = self.filter_1d.predict(dt)
                pos_norm_0_100 = float(max(0.0, min(100.0, x_pred)))
                confidence = 0.0
            else:
                # Weighted average of candidates
                z = float(np.average(np.array(candidates), weights=np.array(weights)))
                # 6) Alpha-beta filter update
                pos_norm_0_100, _ = self.filter_1d.update(z, dt)
                confidence = float(min(1.0, sum(weights) / len(weights)))
            
            # Update position and confidence
            self.current_position = int(round(pos_norm_0_100))
            self.current_confidence = confidence
            
            # Update prev position in pixel space for next iteration
            if conf_of >= conf_d and conf_of > 0.05:
                self.prev_pos_px = pos_of
            elif pos_d is not None:
                self.prev_pos_px = pos_d
            
            # Keep for next frame
            self.prev_gray = gray.copy()
            
            # 7) Activity metric (EMA of average flow magnitude in combined mask)
            if np.count_nonzero(combined_mask) > 0:
                avg_mag = float(np.mean(mag[combined_mask > 0]))
            else:
                avg_mag = 0.0
            self.activity_smooth = (1 - self.ema_alpha) * self.activity_smooth + self.ema_alpha * avg_mag
            
            # Generate funscript action if tracking is active
            action_log = None
            if self.tracking_active:
                action_log = [{
                    "at": frame_time_ms,
                    "pos": self.current_position,
                    "confidence": confidence,
                    "activity": self.activity_smooth
                }]
            
            # Visualization
            vis_frame = self._draw_visualization(frame_small, confidence)
            
            # Debug info
            debug_info = {
                'position': self.current_position,
                'confidence': confidence,
                'activity': self.activity_smooth,
                'fps': round(self.current_fps, 1),
                'tracking_active': self.tracking_active,
                'axis_defined': self.axis_A is not None and self.axis_B is not None,
                'motion_pixels': np.count_nonzero(combined_mask),
                'global_motion': (mu_glob, mv_glob)
            }
            
            status_msg = f"Axis Projection | Pos: {self.current_position} | " \
                        f"Conf: {confidence:.2f} | Activity: {self.activity_smooth:.2f}"
            
            return TrackerResult(
                processed_frame=vis_frame,
                action_log=action_log,
                debug_info=debug_info,
                status_message=status_msg
            )
            
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            return TrackerResult(
                processed_frame=frame,
                action_log=None,
                debug_info={'error': str(e)},
                status_message=f"Error: {str(e)}"
            )
    
    def _draw_visualization(self, frame: np.ndarray, confidence: float) -> np.ndarray:
        """Draw visualization overlays on frame."""
        vis = frame.copy()
        
        # Draw axis if defined
        if self.show_axis and self.axis_A is not None and self.axis_B is not None:
            A_proc = self._original_to_proc_point(self.axis_A)
            B_proc = self._original_to_proc_point(self.axis_B)
            
            # Draw axis endpoints
            cv2.circle(vis, (int(A_proc[0]), int(A_proc[1])), 4, (0, 255, 0), -1)
            cv2.circle(vis, (int(B_proc[0]), int(B_proc[1])), 4, (0, 255, 0), -1)
            
            # Draw axis line
            cv2.line(vis, (int(A_proc[0]), int(A_proc[1])), 
                    (int(B_proc[0]), int(B_proc[1])), (0, 200, 0), 2)
            
            # Draw current position on axis
            t = self.current_position / 100.0
            axis_x = A_proc[0] + t * (B_proc[0] - A_proc[0])
            axis_y = A_proc[1] + t * (B_proc[1] - A_proc[1])
            cv2.circle(vis, (int(axis_x), int(axis_y)), 8, (255, 255, 0), -1)
        
        # Draw current 2D tracking point
        if self.show_tracking_point and self.prev_pos_px is not None:
            cv2.circle(vis, (int(self.prev_pos_px[0]), int(self.prev_pos_px[1])), 
                      6, (0, 0, 255), -1)
            
            # Draw confidence circle
            radius = int(self.roi_radius * self.scale * confidence)
            if radius > 0:
                cv2.circle(vis, (int(self.prev_pos_px[0]), int(self.prev_pos_px[1])),
                          radius, (0, 0, 255), 1)
        
        # Overlay text
        text = f"Pos: {self.current_position}/100 Conf: {confidence:.2f} Act: {self.activity_smooth:.2f}"
        cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (20, 220, 20), 2)
        
        if self.tracking_active:
            cv2.putText(vis, "TRACKING", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 255, 0), 2)
        
        return vis
    
    def start_tracking(self):
        """Start tracking and generating funscript actions."""
        self.tracking_active = True
        self.logger.info("Axis projection tracking started")
    
    def stop_tracking(self):
        """Stop tracking."""
        self.tracking_active = False
        self.logger.info("Axis projection tracking stopped")
    
    def get_status_info(self) -> Dict[str, Any]:
        """Get current status information."""
        base_status = super().get_status_info()
        
        custom_status = {
            'position': self.current_position,
            'confidence': self.current_confidence,
            'activity': self.activity_smooth,
            'axis_defined': self.axis_A is not None and self.axis_B is not None,
            'tracking_active': self.tracking_active,
            'fps': round(self.current_fps, 1),
            'processing_width': self.proc_width,
            'filter_alpha': self.filter_1d.alpha,
            'filter_beta': self.filter_1d.beta
        }
        
        base_status.update(custom_status)
        return base_status
    
    def cleanup(self):
        """Clean up resources."""
        self.prev_gray = None
        self.dis = None
        self.logger.info("Axis projection tracker cleaned up")