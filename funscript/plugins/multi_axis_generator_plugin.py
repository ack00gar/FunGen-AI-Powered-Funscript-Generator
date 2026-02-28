"""Multi-axis generation plugin for funscript transformations.

Generates secondary axis data (roll, pitch, twist, sway, surge) from the
primary stroke axis using derivative-based heuristics. Optionally uses
video-aware body orientation data when available.
"""
import numpy as np
from typing import Dict, Any, List, Optional
from scipy.ndimage import uniform_filter1d

try:
    from .base_plugin import FunscriptTransformationPlugin
except ImportError:
    from funscript.plugins.base_plugin import FunscriptTransformationPlugin


class MultiAxisGeneratorPlugin(FunscriptTransformationPlugin):
    """Generates secondary axis funscript data from the primary stroke axis.

    Two modes:
    - **Heuristic**: derivative-based generation (no video data needed, ~80% quality)
    - **Video-aware**: uses YOLO pose keypoint data for body orientation (when available)
    """

    @property
    def name(self) -> str:
        return "Multi-Axis Generator"

    @property
    def description(self) -> str:
        return "Generate roll/pitch/twist/sway/surge axes from the stroke axis"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def category(self) -> str:
        return "Timing & Generation"

    @property
    def requires_scipy(self) -> bool:
        return True

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            'target_axis': {
                'type': str,
                'required': False,
                'default': 'roll',
                'label': 'Target Axis',
                'description': 'Which axis to generate',
                'constraints': {'choices': ['roll', 'pitch', 'twist', 'sway', 'surge']}
            },
            'generation_mode': {
                'type': str,
                'required': False,
                'default': 'heuristic',
                'label': 'Generation Mode',
                'description': 'Heuristic (from stroke data) or video-aware (from pose data)',
                'constraints': {'choices': ['heuristic', 'video_aware']}
            },
            'intensity': {
                'type': float,
                'required': False,
                'default': 0.7,
                'label': 'Intensity',
                'description': 'How pronounced the generated axis motion should be (0=subtle, 1=full)',
                'constraints': {'min': 0.0, 'max': 1.0}
            },
            'phase_offset_ms': {
                'type': int,
                'required': False,
                'default': 0,
                'label': 'Phase Offset (ms)',
                'description': 'Time offset for the generated axis relative to stroke',
                'constraints': {'min': -500, 'max': 500}
            },
            'smoothing': {
                'type': float,
                'required': False,
                'default': 5.0,
                'label': 'Smoothing',
                'description': 'Smoothing window size (higher = smoother)',
                'constraints': {'min': 1.0, 'max': 20.0}
            },
        }

    @property
    def supported_axes(self) -> List[str]:
        return ['primary']  # Always reads from primary

    def transform(self, funscript, axis: str = 'primary', **parameters) -> None:
        """Generate secondary axis data from the primary stroke axis."""
        validated = self.validate_parameters(parameters)
        target_axis = validated['target_axis']
        mode = validated['generation_mode']

        # Get source actions (always primary/stroke)
        source_actions = funscript.get_axis_actions('primary')
        if not source_actions or len(source_actions) < 4:
            self.logger.warning("Not enough primary actions to generate axis data")
            return None

        if mode == 'video_aware':
            generated = self._generate_video_aware(funscript, source_actions, target_axis, validated)
            if generated is None:
                self.logger.info("Video-aware data not available, falling back to heuristic")
                generated = self._generate_heuristic(source_actions, target_axis, validated)
        else:
            generated = self._generate_heuristic(source_actions, target_axis, validated)

        if generated:
            # Store on the appropriate axis
            funscript.set_axis_actions(target_axis, generated)
            self.logger.info(f"Generated {len(generated)} actions for {target_axis} axis")

        return None  # Modifies in-place

    def _generate_heuristic(self, source_actions: List[Dict], target_axis: str,
                            params: Dict) -> List[Dict]:
        """Derivative-based generation from the stroke axis.

        Each axis uses a different relationship to the stroke signal:
        - Roll: velocity direction changes (1st derivative sign)
        - Pitch: acceleration (2nd derivative)
        - Twist: correlated random walk scaled by speed
        - Sway: low-pass filtered stroke with phase offset
        - Surge: envelope follower of stroke amplitude
        """
        intensity = params['intensity']
        phase_offset_ms = params['phase_offset_ms']
        smoothing = params['smoothing']

        ats = np.array([a['at'] for a in source_actions], dtype=np.float64)
        poss = np.array([a['pos'] for a in source_actions], dtype=np.float64)

        # Compute derivatives
        dt = np.diff(ats)
        dt_safe = np.where(dt > 0, dt, 1.0)
        velocity = np.diff(poss) / dt_safe  # 1st derivative
        velocity = np.append(velocity, velocity[-1])  # pad to same length

        if target_axis == 'roll':
            generated = self._gen_roll(poss, velocity, intensity, smoothing)
        elif target_axis == 'pitch':
            generated = self._gen_pitch(poss, velocity, intensity, smoothing)
        elif target_axis == 'twist':
            generated = self._gen_twist(poss, velocity, intensity, smoothing)
        elif target_axis == 'sway':
            generated = self._gen_sway(poss, velocity, intensity, smoothing)
        elif target_axis == 'surge':
            generated = self._gen_surge(poss, velocity, intensity, smoothing)
        else:
            return []

        # Apply phase offset
        result_ats = ats + phase_offset_ms
        result_ats = np.clip(result_ats, 0, ats[-1])

        # Build action list
        result = []
        for i in range(len(generated)):
            pos = max(0, min(100, int(round(generated[i]))))
            result.append({'at': int(round(result_ats[i])), 'pos': pos})

        return result

    def _gen_roll(self, poss, velocity, intensity, smoothing):
        """Roll: velocity direction changes -> smoothed -> centered at 50."""
        # Sign of velocity indicates direction of motion
        direction = np.sign(velocity)
        smoothed = uniform_filter1d(direction, size=max(3, int(smoothing * 2)))
        return 50.0 + smoothed * 40.0 * intensity

    def _gen_pitch(self, poss, velocity, intensity, smoothing):
        """Pitch: acceleration (2nd derivative) -> smoothed -> centered at 50."""
        accel = np.diff(velocity)
        accel = np.append(accel, accel[-1])
        smoothed = uniform_filter1d(accel, size=max(3, int(smoothing * 3)))
        # Normalize to reasonable range
        scale = np.std(smoothed) if np.std(smoothed) > 0 else 1.0
        normalized = smoothed / (scale * 3)
        return 50.0 + np.clip(normalized, -1, 1) * 40.0 * intensity

    def _gen_twist(self, poss, velocity, intensity, smoothing):
        """Twist: correlated random walk scaled by stroke speed."""
        rng = np.random.RandomState(42)  # Reproducible
        speed = np.abs(velocity)
        speed_norm = speed / (np.max(speed) + 1e-6)

        # Random walk with speed-dependent step size
        walk = np.cumsum(rng.randn(len(poss)) * speed_norm * 0.3)
        # Remove drift
        walk = walk - uniform_filter1d(walk, size=max(3, len(walk) // 4))
        smoothed = uniform_filter1d(walk, size=max(3, int(smoothing * 2)))

        # Normalize
        scale = np.std(smoothed) if np.std(smoothed) > 0 else 1.0
        normalized = smoothed / (scale * 2.5)
        return 50.0 + np.clip(normalized, -1, 1) * 35.0 * intensity

    def _gen_sway(self, poss, velocity, intensity, smoothing):
        """Sway: low-pass filtered stroke with phase offset."""
        # Heavy smoothing of the stroke signal
        lp = uniform_filter1d(poss, size=max(3, int(smoothing * 5)))
        # Center around 50
        centered = lp - np.mean(lp)
        scale = np.max(np.abs(centered)) if np.max(np.abs(centered)) > 0 else 1.0
        normalized = centered / scale
        return 50.0 + normalized * 30.0 * intensity

    def _gen_surge(self, poss, velocity, intensity, smoothing):
        """Surge: envelope follower of stroke amplitude."""
        speed = np.abs(velocity)
        # Envelope follower: max filter then smooth
        window = max(3, int(smoothing * 3))
        # Simple envelope: rolling max
        envelope = np.array([np.max(speed[max(0, i - window):i + 1])
                            for i in range(len(speed))])
        smoothed = uniform_filter1d(envelope, size=max(3, int(smoothing * 2)))

        # Normalize
        scale = np.max(smoothed) if np.max(smoothed) > 0 else 1.0
        normalized = smoothed / scale
        return 50.0 + (normalized - 0.5) * 60.0 * intensity

    def _generate_video_aware(self, funscript, source_actions: List[Dict],
                               target_axis: str, params: Dict) -> Optional[List[Dict]]:
        """Video-aware generation using body orientation data from pose keypoints.

        Returns None if video data is not available (caller should fall back to heuristic).
        """
        # Check if body orientation data is available in tracker results
        try:
            tracker = None
            if hasattr(funscript, '_app') and funscript._app:
                app = funscript._app
                if hasattr(app, 'processor') and app.processor:
                    tracker = app.processor.tracker

            if not tracker:
                return None

            # Look for body orientation data in tracker's frame objects
            frame_objects = getattr(tracker, 'frame_objects', None)
            if not frame_objects:
                return None

            # Map target axis to body orientation field
            field_map = {
                'roll': 'body_roll_0_100',
                'pitch': 'body_pitch_0_100',
                'twist': 'body_twist_0_100',
                'sway': 'pos_lr_0_100',
            }

            field_name = field_map.get(target_axis)
            if not field_name:
                return None

            # Extract orientation data from frame objects
            intensity = params['intensity']
            smoothing = params['smoothing']
            phase_offset_ms = params['phase_offset_ms']

            times = []
            values = []
            for fo in frame_objects:
                val = getattr(fo, field_name, None)
                if val is not None and hasattr(fo, 'timestamp_ms'):
                    times.append(fo.timestamp_ms)
                    values.append(val)

            if len(times) < 4:
                return None

            times = np.array(times, dtype=np.float64)
            values = np.array(values, dtype=np.float64)

            # Smooth the raw keypoint data
            smoothed = uniform_filter1d(values, size=max(3, int(smoothing * 2)))

            # Apply intensity scaling around center (50)
            result_vals = 50.0 + (smoothed - 50.0) * intensity

            # Apply phase offset
            result_times = times + phase_offset_ms
            result_times = np.clip(result_times, 0, times[-1])

            result = []
            for i in range(len(result_vals)):
                pos = max(0, min(100, int(round(result_vals[i]))))
                result.append({'at': int(round(result_times[i])), 'pos': pos})

            return result

        except Exception as e:
            self.logger.debug(f"Video-aware generation failed: {e}")
            return None
