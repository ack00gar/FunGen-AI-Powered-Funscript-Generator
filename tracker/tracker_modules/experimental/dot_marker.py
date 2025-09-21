#!/usr/bin/env python3
"""
DOT Marker Live Tracker

A live tracker that follows a manually selected dot/point on screen using color-based tracking.
User clicks on a point to select it, then the tracker follows similar colored pixels.

Port from legacy DOT_TRACKER mode to modular tracker system.
"""

import logging
import time
import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from collections import deque

from tracker.tracker_modules.core.base_tracker import BaseTracker, TrackerMetadata, TrackerResult
from config.constants_colors import RGBColors


class DOTMarkerTracker(BaseTracker):
    """Live tracker that follows a user-selected dot/point using color tracking."""
    
    @property
    def metadata(self) -> TrackerMetadata:
        return TrackerMetadata(
            name="dot_marker",
            display_name="DOT Marker (Manual Point)",
            description="Tracks a manually selected colored dot/point on screen",
            category="live_intervention",  # Requires user to click/select point
            version="1.0.0",
            author="FunGen Team",
            tags=["dot", "color-tracking", "manual", "live"],
            requires_roi=True,  # Requires manual interaction
            supports_dual_axis=True
        )
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger("DOTMarkerTracker")
        
        # DOT Tracker state
        self.dot_smoothed_xy: Optional[Tuple[float, float]] = None
        self.dot_last_detected_xy: Optional[Tuple[int, int]] = None
        self.dot_selected_x: Optional[int] = None  # user-selected column
        self.dot_hsv_sample: Optional[Tuple[int, int, int]] = None  # sampled HSV at selection
        # Boundary rectangle (in processed frame coordinates) within which dot detection is allowed
        self.dot_boundary_rect: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
        # Whether to draw the boundary overlay during dot tracking
        self.show_dot_boundary: bool = True
        
        # Tracking state
        self.current_fps: float = 30.0
        self.show_stats: bool = True
        self.stats_display: List[str] = []
        self.internal_frame_counter: int = 0
        self.tracking_active: bool = False
        
        # FPS calculation
        self._fps_update_counter: int = 0
        self._fps_last_time: float = 0.0
        
    def initialize(self, app_instance, **kwargs) -> bool:
        """Initialize the tracker with application instance."""
        try:
            self.app = app_instance
            self._initialized = True
            self.logger.info("DOT Marker tracker initialized")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize DOT Marker tracker: {e}")
            self._initialized = False
            return False
    
    def start_tracking(self) -> bool:
        """Start DOT tracking."""
        if not self.dot_selected_x or not self.dot_hsv_sample:
            self.logger.warning("DOT Marker: No point selected. Please click on a point first.")
            return False
        
        self.tracking_active = True
        self.logger.info("DOT Marker tracking started")
        return True
    
    def stop_tracking(self) -> bool:
        """Stop DOT tracking."""
        self.tracking_active = False
        self.logger.info("DOT Marker tracking stopped")
        return True
    
    def set_roi(self, roi: Tuple[int, int, int, int]) -> bool:
        """Set boundary rectangle for DOT detection."""
        self.dot_boundary_rect = roi
        self.logger.info(f"DOT boundary set: {roi}")
        return True
    
    def set_dot_initial_point(self, x_abs: int, y_abs: int, frame: Optional[np.ndarray] = None) -> bool:
        """Set the initial dot point by user click with built-in validation.
        
        Expects coordinates in the processed frame space (after preprocess_frame letterboxing).
        If a raw frame is provided, the method will preprocess it for consistent sampling.
        Stores the selected column (x) and samples HSV at a small patch to drive adaptive masking.
        
        Args:
            x_abs: X coordinate in processed frame space
            y_abs: Y coordinate in processed frame space
            frame: Optional current video frame for HSV sampling
            
        Returns:
            bool: True if point was set successfully, False otherwise
        """
        try:
            # Input validation
            if not isinstance(x_abs, (int, float)) or not isinstance(y_abs, (int, float)):
                self.logger.error("DOT point coordinates must be numeric")
                return False
                
            x_abs, y_abs = int(x_abs), int(y_abs)
            
            if x_abs < 0 or y_abs < 0:
                self.logger.error("DOT point coordinates must be positive")
                return False
            
            # Frame preprocessing and validation
            proc = None
            if frame is not None:
                if not isinstance(frame, np.ndarray):
                    self.logger.error("Frame must be a numpy array")
                    return False
                    
                if frame.size == 0:
                    self.logger.warning("Empty frame provided for HSV sampling")
                else:
                    proc = self._preprocess_frame(frame)
                    if proc is None:
                        self.logger.warning("Frame preprocessing failed")
            
            # Coordinate validation against frame size
            if proc is not None:
                ph, pw = proc.shape[:2]
                if x_abs >= pw or y_abs >= ph:
                    self.logger.error(f"DOT point ({x_abs}, {y_abs}) outside frame bounds ({pw}x{ph})")
                    return False
                    
                x = int(np.clip(x_abs, 0, pw - 1))
                y = int(np.clip(y_abs, 0, ph - 1))
                
                # HSV sampling with validation
                try:
                    hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)
                    # Sample a small 5x5 neighborhood median to be robust
                    x1 = max(0, x - 2); x2 = min(pw - 1, x + 2)
                    y1 = max(0, y - 2); y2 = min(ph - 1, y + 2)
                    patch = hsv[y1:y2 + 1, x1:x2 + 1]
                    
                    if patch.size > 0:
                        sh = int(np.median(patch[..., 0]))
                        ss = int(np.median(patch[..., 1]))
                        sv = int(np.median(patch[..., 2]))
                        
                        # Validate HSV values
                        if 0 <= sh <= 179 and 0 <= ss <= 255 and 0 <= sv <= 255:
                            self.dot_hsv_sample = (sh, ss, sv)
                        else:
                            self.logger.warning(f"Invalid HSV values: H={sh}, S={ss}, V={sv}")
                            self.dot_hsv_sample = None
                    else:
                        self.logger.warning("Empty HSV patch for sampling")
                        self.dot_hsv_sample = None
                        
                except cv2.error as e:
                    self.logger.error(f"OpenCV error during HSV sampling: {e}")
                    self.dot_hsv_sample = None
            else:
                self.dot_hsv_sample = None
            
            # Set the point and reset tracking state
            self.dot_selected_x = x_abs
            self.dot_smoothed_xy = None
            self.dot_last_detected_xy = None
            
            self.logger.info(f"DOT initial point set: x={self.dot_selected_x}, hsv={self.dot_hsv_sample}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set DOT initial point: {e}")
            return False
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Basic frame preprocessing - letterbox to square."""
        if frame is None or frame.size == 0:
            return frame
        
        h, w = frame.shape[:2]
        if h == w:
            return frame
        
        # Letterbox to square
        size = max(h, w)
        result = np.zeros((size, size, 3), dtype=frame.dtype)
        
        if h > w:
            # Pad width
            pad = (size - w) // 2
            result[:, pad:pad+w] = frame
        else:
            # Pad height  
            pad = (size - h) // 2
            result[pad:pad+h, :] = frame
        
        return result
    
    def _update_fps(self):
        """Update FPS calculation using high-performance delta time method."""
        current_time_sec = time.time()
        if self._fps_last_time > 0:
            delta_time = current_time_sec - self._fps_last_time
            if delta_time > 0.001:  # Avoid division by zero
                self.current_fps = 1.0 / delta_time
        self._fps_last_time = current_time_sec
    
    def process_frame(self, frame: np.ndarray, frame_time_ms: int, frame_index: Optional[int] = None) -> TrackerResult:
        """Process frame for DOT tracking."""
        self._update_fps()
        processed_frame = self._preprocess_frame(frame)
        
        # Get settings from app
        app = self.app
        get = app.app_settings.get if (app and hasattr(app, 'app_settings')) else (lambda k, d=None: d)
        
        # DOT tracking settings
        smooth_alpha = float(get("dot_smooth_alpha", 0.3))
        hsv_tolerance = int(get("dot_hsv_tolerance", 20))
        min_contour_area = int(get("dot_min_contour_area", 10))
        
        action_log_list: List[Dict] = []
        final_primary_pos, final_secondary_pos = 50, 50  # Default positions
        
        # DOT tracking logic
        if self.dot_selected_x is not None and self.dot_hsv_sample is not None:
            try:
                ph, pw = processed_frame.shape[:2]
                hsv = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2HSV)
                sh, ss, sv = self.dot_hsv_sample
                
                # Create HSV mask with tolerance
                lower = np.array([max(0, sh - hsv_tolerance), max(0, ss - 50), max(0, sv - 50)])
                upper = np.array([min(179, sh + hsv_tolerance), 255, 255])
                mask = cv2.inRange(hsv, lower, upper)
                
                # Apply boundary constraint if set
                if self.dot_boundary_rect:
                    bx, by, bw, bh = self.dot_boundary_rect
                    boundary_mask = np.zeros(mask.shape, dtype=np.uint8)
                    boundary_mask[by:by+bh, bx:bx+bw] = 255
                    mask = cv2.bitwise_and(mask, boundary_mask)
                
                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Find largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(largest_contour) >= min_contour_area:
                        # Get centroid
                        M = cv2.moments(largest_contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            self.dot_last_detected_xy = (cx, cy)
                            
                            # Smooth the position
                            if self.dot_smoothed_xy is None:
                                self.dot_smoothed_xy = (float(cx), float(cy))
                            else:
                                sx, sy = self.dot_smoothed_xy
                                self.dot_smoothed_xy = (
                                    sx * (1.0 - smooth_alpha) + cx * smooth_alpha,
                                    sy * (1.0 - smooth_alpha) + cy * smooth_alpha
                                )
                            
                            # Convert to funscript positions (0-100)
                            sx, sy = self.dot_smoothed_xy
                            final_primary_pos = int(np.clip((sy / ph) * 100, 0, 100))
                            final_secondary_pos = int(np.clip((sx / pw) * 100, 0, 100))
                            
                            # Draw tracking indicator
                            cv2.circle(processed_frame, (int(sx), int(sy)), 5, RGBColors.GREEN, 2)
                            cv2.circle(processed_frame, (cx, cy), 3, RGBColors.YELLOW, -1)
                
                # Draw boundary if enabled
                if self.show_dot_boundary and self.dot_boundary_rect:
                    bx, by, bw, bh = self.dot_boundary_rect
                    cv2.rectangle(processed_frame, (bx, by), (bx + bw, by + bh), RGBColors.CYAN, 1)
                
            except Exception as e:
                self.logger.warning(f"DOT tracking error: {e}")
        
        # Write funscript actions if tracking is active
        can_write_now = False
        if self.app:
            get = (self.app.app_settings.get if hasattr(self.app, 'app_settings') else (lambda k, d=None: d))
            preview_write = bool(get('dot_preview_write_enabled', True))
            can_write_now = bool(self.tracking_active or preview_write)
        
        if can_write_now and self.app and (hasattr(self.app, 'funscript') or hasattr(self.app, 'funscript_processor')):
            # Get axis settings
            current_tracking_axis_mode = getattr(self.app, 'tracking_axis_mode', 'both')
            current_single_axis_output = getattr(self.app, 'single_axis_output_target', 'primary')
            
            primary_to_write, secondary_to_write = None, None
            if current_tracking_axis_mode == "both":
                primary_to_write, secondary_to_write = final_primary_pos, final_secondary_pos
            elif current_tracking_axis_mode == "vertical":
                if current_single_axis_output == "primary":
                    primary_to_write = final_primary_pos
                else:
                    secondary_to_write = final_primary_pos
            elif current_tracking_axis_mode == "horizontal":
                if current_single_axis_output == "primary":
                    primary_to_write = final_secondary_pos
                else:
                    secondary_to_write = final_secondary_pos
            
            # Write actions
            delay_frames = int(get("tracker_delay_frames", 0))
            effective_delay_ms = delay_frames * (1000.0 / max(1.0, self.current_fps))
            
            if primary_to_write is not None and hasattr(self.app, 'funscript'):
                self.app.funscript.add_action(
                    timestamp_ms=frame_time_ms + effective_delay_ms,
                    primary_pos=primary_to_write,
                    secondary_pos=secondary_to_write if secondary_to_write is not None else None
                )
            elif secondary_to_write is not None and hasattr(self.app, 'funscript'):
                # If only secondary axis, still need to add action
                self.app.funscript.add_action(
                    timestamp_ms=frame_time_ms + effective_delay_ms,
                    primary_pos=None,
                    secondary_pos=secondary_to_write
                )
            
            action_log_list.append({
                "tracker_mode": "DOT_MARKER",
                "frame_time_ms": frame_time_ms,
                "delay_applied_ms": effective_delay_ms,
                "roi_main": self.dot_boundary_rect,
                "dot_position": self.dot_smoothed_xy
            })
        
        # Update stats display
        self.stats_display = [
            f"DOT FPS:{self.current_fps:.1f} T(ms):{frame_time_ms}",
            f"Selected X:{self.dot_selected_x} HSV:{self.dot_hsv_sample}",
            f"Detected:{self.dot_last_detected_xy} Smoothed:{self.dot_smoothed_xy}",
            f"Output: P:{final_primary_pos} S:{final_secondary_pos}"
        ]
        
        # Add tracking indicator
        self._draw_tracking_indicator(processed_frame)
        
        self.internal_frame_counter += 1
        
        # Prepare debug info
        debug_info = {
            "dot_selected": self.dot_selected_x is not None,
            "dot_position": self.dot_selected_x,
            "hsv_sample": self.dot_hsv_sample,
            "boundary_rect": self.dot_boundary_rect,
            "last_detected": self.dot_last_detected_xy,
            "smoothed_position": self.dot_smoothed_xy,
            "primary_pos": final_primary_pos,
            "secondary_pos": final_secondary_pos,
            "fps": self.current_fps,
            "tracking_active": self.tracking_active
        }
        
        return TrackerResult(
            processed_frame=processed_frame,
            action_log=action_log_list if action_log_list else None,
            debug_info=debug_info,
            status_message=f"DOT Marker: {len(action_log_list)} actions"
        )
    
    def get_status_info(self) -> Dict[str, Any]:
        """Get current tracker status."""
        return {
            "tracker": self.metadata.display_name,
            "active": self.tracking_active,
            "initialized": self._initialized,
            "dot_selected": self.dot_selected_x is not None,
            "dot_hsv_sample": self.dot_hsv_sample,
            "boundary_rect": self.dot_boundary_rect,
            "last_detected": self.dot_last_detected_xy,
            "smoothed_position": self.dot_smoothed_xy,
            "fps": self.current_fps
        }
    
    def validate_settings(self, settings: Dict[str, Any]) -> bool:
        """Validate DOT marker settings."""
        try:
            # Validate smooth alpha
            if 'dot_smooth_alpha' in settings:
                alpha = float(settings['dot_smooth_alpha'])
                if alpha <= 0 or alpha > 1.0:
                    self.logger.error(f"Invalid smooth_alpha: {alpha}. Must be 0-1")
                    return False
            
            # Validate HSV tolerance
            if 'dot_hsv_tolerance' in settings:
                tolerance = int(settings['dot_hsv_tolerance'])
                if tolerance < 1 or tolerance > 179:
                    self.logger.error(f"Invalid HSV tolerance: {tolerance}. Must be 1-179")
                    return False
            
            # Validate min contour area
            if 'dot_min_contour_area' in settings:
                area = int(settings['dot_min_contour_area'])
                if area < 1 or area > 10000:
                    self.logger.error(f"Invalid min contour area: {area}. Must be 1-10000")
                    return False
            
            return True
        except (ValueError, TypeError) as e:
            self.logger.error(f"Settings validation error: {e}")
            return False
    
    def get_settings_schema(self) -> Dict[str, Any]:
        """Get JSON schema for DOT marker settings."""
        return {
            "type": "object",
            "title": "DOT Marker Settings",
            "properties": {
                "dot_smooth_alpha": {
                    "type": "number",
                    "title": "Position Smoothing",
                    "description": "Position smoothing factor (0-1, higher = more responsive)",
                    "minimum": 0.01,
                    "maximum": 1.0,
                    "default": 0.3
                },
                "dot_hsv_tolerance": {
                    "type": "integer",
                    "title": "Color Tolerance",
                    "description": "HSV color matching tolerance (1-179)",
                    "minimum": 1,
                    "maximum": 179,
                    "default": 20
                },
                "dot_min_contour_area": {
                    "type": "integer",
                    "title": "Min Detection Area",
                    "description": "Minimum contour area for detection (pixels)",
                    "minimum": 1,
                    "maximum": 10000,
                    "default": 10
                },
                "dot_preview_write_enabled": {
                    "type": "boolean",
                    "title": "Enable Preview Mode",
                    "description": "Generate funscript actions in preview mode",
                    "default": True
                }
            },
            "additionalProperties": False
        }
    
    def set_roi(self, roi: Tuple[int, int, int, int]) -> bool:
        """Set boundary rectangle for DOT detection."""
        try:
            if len(roi) != 4:
                self.logger.error("ROI must be (x, y, width, height)")
                return False
            
            x, y, w, h = roi
            if w <= 0 or h <= 0:
                self.logger.error("ROI width and height must be positive")
                return False
            
            self.dot_boundary_rect = roi
            self.logger.info(f"DOT Marker boundary ROI set: {roi}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to set ROI: {e}")
            return False
    
    def handle_mouse_click(self, x: int, y: int, frame: Optional[np.ndarray] = None) -> bool:
        """Handle mouse click to select DOT point.
        
        Args:
            x: X coordinate of click
            y: Y coordinate of click 
            frame: Current frame for HSV sampling
            
        Returns:
            bool: True if point was set successfully
        """
        return self.set_dot_initial_point(x, y, frame)
    
    def is_ready_for_tracking(self) -> bool:
        """Check if tracker is ready to start tracking."""
        return (self.dot_selected_x is not None and 
                self.dot_hsv_sample is not None and
                self._initialized)
    
    def reset_selection(self):
        """Reset DOT selection and tracking state."""
        self.dot_smoothed_xy = None
        self.dot_last_detected_xy = None
        self.dot_selected_x = None
        self.dot_hsv_sample = None
        self.tracking_active = False
        self.logger.info("DOT selection reset")
    
    def cleanup(self):
        """Clean up resources."""
        self.dot_smoothed_xy = None
        self.dot_last_detected_xy = None
        self.dot_selected_x = None
        self.dot_hsv_sample = None
        self.dot_boundary_rect = None
        self.tracking_active = False
        self._initialized = False