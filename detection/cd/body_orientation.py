"""Body orientation extraction from YOLO pose keypoints.

Derives roll, pitch, twist, and sway angles from COCO 17-keypoint
pose data. Used by the multi-axis video-aware generation system.

COCO keypoint indices:
  5,6  = left/right shoulder
  11,12 = left/right hip
"""
import numpy as np
import math
from typing import Dict, Optional


class BodyOrientationExtractor:
    """Extract body orientation angles from pose keypoints.

    Uses shoulder (5,6) and hip (11,12) keypoints to derive:
    - Roll: hip tilt angle (lateral lean)
    - Pitch: forward/backward body lean
    - Twist: differential rotation between shoulder and hip lines
    - Sway: horizontal displacement of hip center vs shoulder center
    """

    # COCO keypoint indices
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_HIP = 11
    RIGHT_HIP = 12

    def __init__(self, confidence_threshold: float = 0.3, ema_alpha: float = 0.3):
        """
        Args:
            confidence_threshold: Minimum keypoint confidence to use
            ema_alpha: EMA smoothing factor (0=heavy smooth, 1=no smooth)
        """
        self._conf_threshold = confidence_threshold
        self._ema_alpha = ema_alpha
        self._prev_angles: Dict[str, float] = {}

    def extract_angles(self, keypoints: np.ndarray) -> Optional[Dict[str, float]]:
        """Extract body orientation angles from pose keypoints.

        Args:
            keypoints: numpy array of shape (17, 3) with [x, y, confidence] per keypoint

        Returns:
            Dict with keys: roll_deg, pitch_deg, twist_deg, sway_px
            or None if insufficient keypoint confidence
        """
        if keypoints is None or keypoints.shape[0] < 13:
            return None

        # Extract keypoints with confidence check
        ls = keypoints[self.LEFT_SHOULDER]   # [x, y, conf]
        rs = keypoints[self.RIGHT_SHOULDER]
        lh = keypoints[self.LEFT_HIP]
        rh = keypoints[self.RIGHT_HIP]

        # Check confidence thresholds
        hip_conf = min(lh[2], rh[2])
        shoulder_conf = min(ls[2], rs[2])

        if hip_conf < self._conf_threshold:
            return None  # Hip keypoints not reliable enough

        result = {}

        # --- Roll: hip tilt angle ---
        # atan2(hip_R.y - hip_L.y, hip_R.x - hip_L.x)
        if hip_conf >= self._conf_threshold:
            roll_rad = math.atan2(rh[1] - lh[1], rh[0] - lh[0])
            result['roll_deg'] = math.degrees(roll_rad)

        # --- Pitch: body lean (shoulder midpoint to hip midpoint angle) ---
        if shoulder_conf >= self._conf_threshold and hip_conf >= self._conf_threshold:
            shoulder_mid = np.array([(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2])
            hip_mid = np.array([(lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2])

            # Angle from vertical (hip to shoulder)
            dx = shoulder_mid[0] - hip_mid[0]
            dy = shoulder_mid[1] - hip_mid[1]
            # In image coords, Y increases downward, so negate dy for standard angle
            pitch_rad = math.atan2(dx, -dy)
            result['pitch_deg'] = math.degrees(pitch_rad)

        # --- Twist: differential rotation between shoulder and hip lines ---
        if shoulder_conf >= self._conf_threshold and hip_conf >= self._conf_threshold:
            shoulder_angle = math.atan2(rs[1] - ls[1], rs[0] - ls[0])
            hip_angle = math.atan2(rh[1] - lh[1], rh[0] - lh[0])
            twist_rad = shoulder_angle - hip_angle
            # Normalize to -pi..pi
            while twist_rad > math.pi:
                twist_rad -= 2 * math.pi
            while twist_rad < -math.pi:
                twist_rad += 2 * math.pi
            result['twist_deg'] = math.degrees(twist_rad)

        # --- Sway: horizontal displacement of hip center relative to shoulder center ---
        if shoulder_conf >= self._conf_threshold and hip_conf >= self._conf_threshold:
            shoulder_cx = (ls[0] + rs[0]) / 2
            hip_cx = (lh[0] + rh[0]) / 2
            result['sway_px'] = hip_cx - shoulder_cx

        # Apply EMA smoothing
        result = self._apply_ema(result)

        return result

    def angle_to_funscript_pos(self, angle: float, center: float = 0.0,
                                range_deg: float = 45.0) -> int:
        """Convert an angle to a 0-100 funscript position.

        Args:
            angle: Angle in degrees
            center: Center angle (maps to pos=50)
            range_deg: Full range in degrees (maps to 0-100)

        Returns:
            Funscript position 0-100
        """
        normalized = (angle - center) / max(0.1, range_deg) + 0.5
        pos = normalized * 100.0
        return max(0, min(100, int(round(pos))))

    def _apply_ema(self, angles: Dict[str, float]) -> Dict[str, float]:
        """Apply exponential moving average smoothing."""
        alpha = self._ema_alpha
        smoothed = {}
        for key, val in angles.items():
            if key in self._prev_angles:
                smoothed[key] = alpha * val + (1 - alpha) * self._prev_angles[key]
            else:
                smoothed[key] = val
        self._prev_angles.update(smoothed)
        return smoothed

    def reset(self):
        """Reset smoothing state."""
        self._prev_angles.clear()
