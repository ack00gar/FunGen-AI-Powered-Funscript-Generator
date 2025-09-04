#!/usr/bin/env python3
"""
Beat Marker Live Tracker

A live tracker that generates funscript actions based on audio/visual beats.
Supports multiple sources: visual brightness, audio envelope, and metronome.

Port from legacy BEAT_MARKER mode to modular tracker system.
"""

import logging
import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from collections import deque

from tracker_modules.core.base_tracker import BaseTracker, TrackerMetadata, TrackerResult
from config.constants_colors import RGBColors


class BeatMarkerTracker(BaseTracker):
    """Live tracker that generates funscript actions based on beat detection."""
    
    @property
    def metadata(self) -> TrackerMetadata:
        return TrackerMetadata(
            name="beat_marker",
            display_name="Live Beat Marker (Visual/Audio)",
            description="Generates actions from visual brightness changes, audio beats, or metronome",
            category="live",
            version="1.0.0",
            author="FunGen Team",
            tags=["beat", "audio-analysis", "visual-analysis", "metronome", "live"],
            requires_roi=False,
            supports_dual_axis=True
        )
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger("BeatMarkerTracker")
        
        # Beat detection state
        self.beat_brightness_history: deque = deque(maxlen=60)
        self.beat_armed: bool = True
        self.beat_last_tick_time_ms: Optional[float] = None
        self.beat_toggle_high: bool = True
        self.beat_last_output_pos: int = 50
        
        # Audio-specific state
        self.beat_audio_env_history: Optional[deque] = None
        self.beat_audio_novelty_history: Optional[deque] = None
        self.beat_audio_ema: Optional[float] = None
        self.beat_audio_last_novelty: Optional[float] = None
        self.audio_beat_analyzer: Optional[Any] = None
        self._beat_analyzer_logged: bool = False
        
        # Metronome state
        self.beat_next_tick_time_ms: Optional[float] = None
        self.beat_swing_long_next: bool = True
        
        # Tracking state  
        self.current_fps: float = 30.0
        self.show_stats: bool = True
        self.stats_display: List[str] = []
        self.tracking_active: bool = False
        
        # Debug logging throttling
        self.last_audio_env_dbg_log_ms: float = 0
        
    def initialize(self, app_instance, **kwargs) -> bool:
        """Initialize the tracker with application instance."""
        try:
            self.app = app_instance
            self.logger.info("Beat Marker tracker initialized")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize Beat Marker tracker: {e}")
            return False
    
    def start_tracking(self) -> bool:
        """Start beat tracking."""
        self.tracking_active = True
        self.beat_armed = True
        self.logger.info("Beat Marker tracking started")
        return True
    
    def stop_tracking(self) -> bool:
        """Stop beat tracking."""
        self.tracking_active = False
        self.logger.info("Beat Marker tracking stopped")
        return True
    
    def set_roi(self, roi: Tuple[int, int, int, int]) -> bool:
        """Set ROI for visual beat detection."""
        # Beat marker can use ROI for visual source
        return True
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Basic frame preprocessing."""
        if frame is None or frame.size == 0:
            return frame
        return frame  # Beat marker doesn't need complex preprocessing
    
    def _update_fps(self):
        """Update FPS calculation."""
        # Simple FPS estimation - could be improved with timing
        self.current_fps = 30.0  # Default assumption
    
    def process_frame(self, frame: np.ndarray, frame_time_ms: int, frame_index: Optional[int] = None) -> TrackerResult:
        """Process frame for beat detection and action generation."""
        self._update_fps()
        processed_frame = self._preprocess_frame(frame)
        
        # Get settings from app
        app = self.app
        get = app.app_settings.get if (app and hasattr(app, 'app_settings')) else (lambda k, d=None: d)
        
        # Beat detection settings
        source = str(get("beat_source", "visual")).strip().lower()
        bpm = float(get("beat_bpm", 120))
        subdivision = max(1, int(get("beat_subdivision", 1)))
        amp_min = int(get("beat_amp_min", 10))
        amp_max = int(get("beat_amp_max", 90))
        waveform = str(get("beat_waveform", "step")).strip().lower()
        thr_sigma = float(get("beat_threshold_sigma", 2.0))
        hyster_ratio = float(get("beat_hysteresis_ratio", 0.6))
        min_interval_ms = float(get("beat_min_interval_ms", 250))
        swing_pct = float(get("beat_swing_percent", 0.0))  # 0-50
        phase_deg = float(get("beat_phase_deg", 0.0))
        
        # Clamp/sanitize amplitude inputs and handle swapped values
        amp_min = max(0, min(100, amp_min))
        amp_max = max(0, min(100, amp_max))
        if amp_min > amp_max:
            amp_min, amp_max = amp_max, amp_min
        
        # Compute derived beat interval from BPM/subdivision for metronome source only
        if source == "metronome" and bpm > 0:
            beat_period_ms = (60000.0 / bpm) / subdivision
            min_interval_ms = max(min_interval_ms, 0.5 * beat_period_ms)
        
        action_log_list: List[Dict] = []
        
        # Beat detection logic
        signal = 0.0
        x1 = y1 = x2 = y2 = 0  # for visual overlay only
        
        if source == "visual":
            gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            use_user_roi = (bool(get("beat_use_user_roi", False)) and getattr(self, 'user_roi_fixed', None))
            
            if use_user_roi and hasattr(self, 'user_roi_fixed') and self.user_roi_fixed is not None:
                rx, ry, rw, rh = self.user_roi_fixed
                if rw > 2 and rh > 2:
                    x1 = max(0, min(w - 1, rx))
                    y1 = max(0, min(h - 1, ry))
                    x2 = max(x1 + 1, min(w, rx + rw))
                    y2 = max(y1 + 1, min(h, ry + rh))
                    patch = gray[y1:y2, x1:x2]
                else:
                    use_user_roi = False
            
            if not use_user_roi:
                # Use center patch
                cx, cy = w // 2, h // 2
                patch_size = min(w, h) // 8
                x1 = max(0, cx - patch_size)
                y1 = max(0, cy - patch_size)
                x2 = min(w, cx + patch_size)
                y2 = min(h, cy + patch_size)
                patch = gray[y1:y2, x1:x2]
            
            signal = float(np.mean(patch)) if patch.size > 0 else 0.0
            
        elif source == "audio":
            # Try to get audio envelope from VideoProcessor
            if hasattr(self.app, 'processor') and self.app.processor is not None:
                try:
                    signal = float(self.app.processor.get_audio_envelope_value(frame_time_ms))
                except Exception as e:
                    self.logger.warning(f"Audio envelope unavailable: {e}")
                    signal = 0.0
            else:
                signal = 0.0
            
            # Build EMA baseline and novelty for robust click detection
            try:
                if not hasattr(self, 'beat_audio_env_history'):
                    self.beat_audio_env_history = deque(maxlen=400)
                if not hasattr(self, 'beat_audio_novelty_history'):
                    self.beat_audio_novelty_history = deque(maxlen=200)
                
                # EMA smoothing factor
                ema_alpha = float(get('beat_audio_ema_alpha', 0.2))
                ema_alpha = max(0.01, min(0.9, ema_alpha))
                
                # Update EMA
                if self.beat_audio_ema is None:
                    self.beat_audio_ema = float(signal)
                else:
                    self.beat_audio_ema = (1.0 - ema_alpha) * float(self.beat_audio_ema) + ema_alpha * float(signal)
                
                novelty = max(0.0, float(signal) - float(self.beat_audio_ema))
                self.beat_audio_env_history.append(float(signal))
                self.beat_audio_novelty_history.append(float(novelty))
                
            except Exception:
                novelty = max(0.0, float(signal))
        
        # Update signal history for z-score
        if not hasattr(self, 'beat_brightness_history') or self.beat_brightness_history.maxlen is None:
            self.beat_brightness_history = deque(maxlen=60)
        self.beat_brightness_history.append(signal)
        
        # Stats baseline
        avg = float(np.mean(self.beat_brightness_history)) if len(self.beat_brightness_history) > 0 else signal
        std = float(np.std(self.beat_brightness_history)) if len(self.beat_brightness_history) > 1 else 0.0
        z = (signal - avg) / (std + 1e-6)
        
        # Hysteresis + interval gating
        now_ms = frame_time_ms
        last_ms = self.beat_last_tick_time_ms if hasattr(self, 'beat_last_tick_time_ms') else None
        interval_ok = (last_ms is None) or ((now_ms - last_ms) >= min_interval_ms)
        
        triggered = False
        
        if source == "visual":
            if self.beat_armed and z >= thr_sigma and interval_ok:
                triggered = True
                self.beat_armed = False
                self.beat_last_tick_time_ms = now_ms
            # Re-arm condition
            if not self.beat_armed and z <= (thr_sigma * hyster_ratio):
                self.beat_armed = True
                
        elif source == "audio":
            # Simple audio trigger based on z-score
            if self.beat_armed and z >= thr_sigma and interval_ok:
                triggered = True
                self.beat_armed = False
                self.beat_last_tick_time_ms = now_ms
            # Re-arm condition
            if not self.beat_armed and z <= (thr_sigma * hyster_ratio):
                self.beat_armed = True
                
        elif source == "metronome" and bpm > 0:
            # Initialize next tick lazily with phase offset
            if self.beat_next_tick_time_ms is None:
                phase_frac = (phase_deg / 360.0)
                base_interval = (60000.0 / bpm) / subdivision
                phase_offset_ms = phase_frac * base_interval
                self.beat_next_tick_time_ms = now_ms + max(0.0, phase_offset_ms)
                self.beat_swing_long_next = True
            
            # Schedule loop in case a frame is delayed
            while self.beat_next_tick_time_ms is not None and now_ms >= self.beat_next_tick_time_ms:
                if interval_ok:
                    triggered = True
                    self.beat_last_tick_time_ms = now_ms
                
                # compute next interval with swing
                base_interval = (60000.0 / bpm) / subdivision
                s = max(0.0, min(50.0, swing_pct)) / 100.0
                if s > 0.0:
                    if self.beat_swing_long_next:
                        interval = base_interval * (1.0 + s)
                    else:
                        interval = base_interval * (1.0 - s)
                    self.beat_swing_long_next = not self.beat_swing_long_next
                else:
                    interval = base_interval
                    
                # Advance next tick
                self.beat_next_tick_time_ms += interval
        
        # Overlay visual aids
        if self.show_stats and source == "visual":
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), 
                         RGBColors.YELLOW if self.beat_armed else RGBColors.ORANGE, 1)
        
        # On beat: compute output position
        can_write_now = False
        if self.app:
            get = (self.app.app_settings.get if hasattr(self.app, 'app_settings') else (lambda k, d=None: d))
            preview_write = bool(get('beat_preview_write_enabled', True))
            can_write_now = bool(self.tracking_active or preview_write)
        
        if triggered and self.app and can_write_now:
            if waveform == "step":
                # Read current toggle state (defaults to True on first use)
                prev_toggle = self.beat_toggle_high
                pos = amp_max if prev_toggle else amp_min
                # Flip toggle for next beat
                self.beat_toggle_high = not prev_toggle
            else:
                # Fallback to max for non-implemented waveforms
                pos = amp_max
            
            final_primary_pos = pos
            final_secondary_pos = pos
            
            # Remember last output for debugging/overlay
            self.beat_last_output_pos = int(pos)
            
            # Axis selection consistent with other trackers
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
                "tracker_mode": "BEAT_MARKER",
                "frame_time_ms": frame_time_ms,
                "delay_applied_ms": effective_delay_ms,
                "roi_main": None,
                "beat_triggered": True,
                "position": pos,
                "source": source
            })
            
            # Log trigger for visibility
            self.logger.debug(f"Beat Marker TRIGGER: source={source} pos={pos} z={z:.2f}")
        
        # Update stats display
        if source == "audio":
            sig_label = f"E:{signal:.2f}"
        else:
            sig_label = f"B:{signal:.1f}"
            
        self.stats_display = [
            f"Beat FPS:{self.current_fps:.1f} T(ms):{frame_time_ms}",
            f"{sig_label} z:{z:.2f} armed:{self.beat_armed} trig:{triggered}",
            f"Src:{source} WF:{waveform} Amp:[{amp_min},{amp_max}] Last:{self.beat_last_output_pos} Tgl:{self.beat_toggle_high}",
            f"BPM:{bpm:.1f} Sub:{subdivision} Swing:{swing_pct:.1f}% Phase:{phase_deg:.1f}"
        ]
        
        if self.show_stats:
            for i, stat_text in enumerate(self.stats_display):
                cv2.putText(processed_frame, stat_text, (5, 15 + i * 12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, RGBColors.TEAL, 1)
        
        return TrackerResult(
            processed_frame=processed_frame,
            action_log=action_log_list if action_log_list else None,
            status_message=f"Beat Marker: {len(action_log_list)} beats"
        )
    
    def get_status_info(self) -> Dict[str, Any]:
        """Get current tracker status."""
        return {
            "beat_armed": self.beat_armed,
            "last_tick_time": self.beat_last_tick_time_ms,
            "toggle_state": self.beat_toggle_high,
            "last_output": self.beat_last_output_pos,
            "signal_history_length": len(self.beat_brightness_history) if hasattr(self, 'beat_brightness_history') else 0
        }
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'beat_brightness_history'):
            self.beat_brightness_history.clear()
        if hasattr(self, 'beat_audio_env_history') and self.beat_audio_env_history:
            self.beat_audio_env_history.clear()
        if hasattr(self, 'beat_audio_novelty_history') and self.beat_audio_novelty_history:
            self.beat_audio_novelty_history.clear()
        self.audio_beat_analyzer = None