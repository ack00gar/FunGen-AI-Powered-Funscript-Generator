#!/usr/bin/env python3
"""
Cock Hero Beat Tracker -- generate funscript from audio beats and energy.

Designed for PMVs, cock heroes, and music-driven content where the rhythm
drives the stroke pattern.

Pipeline (single offline stage):
  Pass 1: Extract audio, detect raw onsets via spectral flux
  Pass 2: Analyze tempo, identify stable beat grid, remove outliers
  Pass 3: Generate clean alternating up/down actions from the clean grid

Pure signal processing on numpy/scipy. No model downloads.
"""

import os
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from funscript.multi_axis_funscript import MultiAxisFunscript
from video.ffmpeg_helpers import find_ffmpeg, subprocess_flags

try:
    from ..core.base_offline_tracker import (
        BaseOfflineTracker,
        OfflineProcessingResult,
        OfflineProcessingStage,
    )
    from ..core.base_tracker import StageDefinition, TrackerMetadata
except ImportError:
    from tracker.tracker_modules.core.base_offline_tracker import (
        BaseOfflineTracker,
        OfflineProcessingResult,
        OfflineProcessingStage,
    )
    from tracker.tracker_modules.core.base_tracker import StageDefinition, TrackerMetadata


class CockHeroBeatTracker(BaseOfflineTracker):
    """Beat-driven offline tracker for PMV / cock-hero content."""

    SAMPLE_RATE = 22050
    HOP_LENGTH = 512
    N_FFT = 2048
    MAX_DEVICE_SPEED = 600  # units per second; clamps amplitude on tight beats

    def __init__(self):
        super().__init__()
        self.app = None
        self.beat_sensitivity = 0.8
        self.energy_smoothing_s = 0.3
        self.min_action_interval_ms = 80
        self.target_amplitude = 100
        self.min_amplitude = 60
        self.silence_threshold = 0.05

    @property
    def metadata(self) -> TrackerMetadata:
        return TrackerMetadata(
            name="OFFLINE_COCK_HERO_BEAT",
            display_name="Cock Hero Beat Tracker",
            description="Generate funscript from audio beats and energy for PMVs and cock-hero content.",
            category="offline",
            version="1.0.0",
            author="FunGen",
            tags=["offline", "audio", "beats", "pmv", "cock-hero", "batch"],
            requires_roi=False,
            supports_dual_axis=False,
            primary_axis="stroke",
            stages=[
                StageDefinition(
                    stage_number=2,
                    name="Audio Beat Analysis & Funscript Generation",
                    description="Extract audio, detect beats, generate stroke actions",
                    produces_funscript=True,
                    requires_previous=False,
                    output_type="funscript",
                ),
            ],
            properties={
                "produces_funscript_in_stage2": True,
                "supports_batch": True,
                "num_stages": 1,
            },
        )

    @property
    def processing_stages(self) -> List[OfflineProcessingStage]:
        return [OfflineProcessingStage.STAGE_2]

    @property
    def stage_dependencies(self) -> Dict[OfflineProcessingStage, List[OfflineProcessingStage]]:
        return {OfflineProcessingStage.STAGE_2: []}

    def initialize(self, app_instance, **kwargs) -> bool:
        try:
            self.app = app_instance
            self._initialized = True
            self.logger.info("Cock Hero Beat Tracker initialized")
            return True
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}", exc_info=True)
            return False

    def can_resume_from_checkpoint(self, checkpoint_data: Dict[str, Any]) -> bool:
        return False

    def estimate_processing_time(self, stage, video_path, **kwargs) -> float:
        return 15.0

    def process_stage(
        self,
        stage,
        video_path,
        input_data=None,
        input_files=None,
        output_directory=None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        frame_range=None,
        resume_data=None,
        **kwargs,
    ) -> OfflineProcessingResult:
        start_time = time.time()
        try:
            self.processing_active = True
            if not output_directory:
                output_directory = os.path.dirname(video_path)
            os.makedirs(output_directory, exist_ok=True)
            self._load_settings()

            if (
                self.app
                and getattr(self.app, "processor", None)
                and getattr(self.app.processor, "video_info", None)
                and not self.app.processor.video_info.get("has_audio", True)
            ):
                return OfflineProcessingResult(
                    success=False, error_message="Video has no audio track"
                )

            # Pass 1 -- extract audio + raw onset detection
            self.logger.info("Pass 1: audio extraction + raw onset detection")
            if progress_callback:
                progress_callback({"stage": "pass1", "task": "Extracting audio", "percentage": 0})

            audio = self._extract_audio(video_path)
            if audio is None or len(audio) == 0:
                return OfflineProcessingResult(
                    success=False, error_message="Failed to extract audio"
                )

            audio_duration_s = len(audio) / self.SAMPLE_RATE
            self.logger.info(f"Audio: {audio_duration_s:.1f}s, {len(audio)} samples")

            if progress_callback:
                progress_callback({"stage": "pass1", "task": "Detecting onsets", "percentage": 15})

            energy_envelope = self._compute_energy_envelope(audio)
            raw_beats = self._detect_raw_onsets(audio)
            self.logger.info(f"Pass 1: {len(raw_beats)} raw onsets detected")

            if self.stop_event and self.stop_event.is_set():
                return OfflineProcessingResult(
                    success=False, error_message="Processing stopped"
                )

            # Pass 2 -- tempo analysis + pattern cleaning
            self.logger.info("Pass 2: tempo analysis + pattern cleaning")
            if progress_callback:
                progress_callback({"stage": "pass2", "task": "Analyzing tempo", "percentage": 40})

            clean_beats = self._clean_beat_grid(raw_beats, energy_envelope)
            estimated_bpm = 0.0
            if len(clean_beats) > 1:
                estimated_bpm = 60.0 / np.median(np.diff(clean_beats))
            self.logger.info(
                f"Pass 2: {len(clean_beats)} clean beats ({estimated_bpm:.0f} BPM)"
            )

            if progress_callback:
                progress_callback(
                    {
                        "stage": "pass2",
                        "task": f"{len(clean_beats)} beats ({estimated_bpm:.0f} BPM)",
                        "percentage": 60,
                    }
                )

            if self.stop_event and self.stop_event.is_set():
                return OfflineProcessingResult(
                    success=False, error_message="Processing stopped"
                )

            # Pass 3 -- generate clean alternating actions
            self.logger.info("Pass 3: funscript generation")
            if progress_callback:
                progress_callback(
                    {"stage": "pass3", "task": "Generating actions", "percentage": 75}
                )

            actions = self._generate_actions(clean_beats, energy_envelope, audio_duration_s)
            self.logger.info(f"Pass 3: {len(actions)} actions")

            funscript = MultiAxisFunscript(logger=self.logger)
            funscript.set_axis_actions("primary", actions)

            processing_time = time.time() - start_time
            self.logger.info(f"Complete in {processing_time:.1f}s")

            if progress_callback:
                progress_callback({"stage": "complete", "task": "Done", "percentage": 100})

            self.processing_active = False
            return OfflineProcessingResult(
                success=True,
                output_data={"funscript": funscript, "chapters": []},
                performance_metrics={
                    "processing_time_seconds": processing_time,
                    "raw_onsets": len(raw_beats),
                    "clean_beats": len(clean_beats),
                    "actions_generated": len(actions),
                    "audio_duration_s": audio_duration_s,
                    "estimated_bpm": estimated_bpm,
                },
            )
        except Exception as e:
            self.logger.error(f"Cock Hero Beat Tracker error: {e}", exc_info=True)
            self.processing_active = False
            return OfflineProcessingResult(success=False, error_message=str(e))

    # ------------------------------------------------------------------ settings

    def _load_settings(self) -> None:
        if not (self.app and hasattr(self.app, "app_settings")):
            return
        s = self.app.app_settings
        # AppSettings classic dict-style API on main; typed-config on v0.9.0 also
        # supports .get() via the same class for backward compatibility.
        try:
            self.beat_sensitivity = float(s.get("audio_beat_sensitivity", 0.8))
            self.energy_smoothing_s = float(s.get("audio_energy_smoothing", 0.3))
            self.min_action_interval_ms = int(s.get("audio_min_interval_ms", 80))
            self.target_amplitude = int(s.get("audio_target_amplitude", 100))
            self.min_amplitude = int(s.get("audio_min_amplitude", 60))
        except Exception as e:
            self.logger.debug(f"Settings load fell back to defaults: {e}")

    # --------------------------------------------------------- pass 1: audio + onsets

    def _extract_audio(self, video_path: str) -> Optional[np.ndarray]:
        try:
            cmd = [
                find_ffmpeg(),
                "-hide_banner", "-nostats", "-loglevel", "error",
                "-i", video_path,
                "-vn", "-ac", "1", "-ar", str(self.SAMPLE_RATE),
                "-f", "f32le", "pipe:1",
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=300,
                creationflags=subprocess_flags(),
            )
            if result.returncode != 0:
                tail = result.stderr.decode("utf-8", errors="replace")[:200]
                self.logger.error(f"ffmpeg failed: {tail}")
                return None
            audio = np.frombuffer(result.stdout, dtype=np.float32)
            return audio if len(audio) > 0 else None
        except subprocess.TimeoutExpired:
            self.logger.error("ffmpeg timed out")
            return None
        except Exception as e:
            self.logger.error(f"Audio extraction error: {e}")
            return None

    def _compute_energy_envelope(self, audio: np.ndarray) -> np.ndarray:
        n_frames = len(audio) // self.HOP_LENGTH
        if n_frames == 0:
            return np.array([0.0])
        trimmed = audio[: n_frames * self.HOP_LENGTH]
        frames = trimmed.reshape(n_frames, self.HOP_LENGTH)
        rms = np.sqrt(np.mean(frames ** 2, axis=1))
        smooth_hops = max(
            1, int(self.energy_smoothing_s * self.SAMPLE_RATE / self.HOP_LENGTH)
        )
        rms_smooth = uniform_filter1d(rms, size=smooth_hops)
        p99 = np.percentile(rms_smooth, 99)
        if p99 > 0:
            rms_smooth = np.clip(rms_smooth / p99, 0, 1)
        return rms_smooth

    def _detect_raw_onsets(self, audio: np.ndarray) -> np.ndarray:
        """Detect all onsets via spectral flux. Returns onset times in seconds."""
        hop = self.HOP_LENGTH
        n_fft = self.N_FFT
        n_frames = (len(audio) - n_fft) // hop + 1
        if n_frames < 2:
            return np.array([])

        flux = np.zeros(n_frames)
        prev_spec = None
        for i in range(n_frames):
            start = i * hop
            frame = audio[start : start + n_fft]
            if len(frame) < n_fft:
                break
            windowed = frame * np.hanning(n_fft)
            spec = np.abs(np.fft.rfft(windowed))
            if prev_spec is not None:
                flux[i] = np.sum(np.maximum(0, spec - prev_spec))
            prev_spec = spec

        if np.max(flux) == 0:
            return np.array([])

        flux = flux / np.max(flux)
        threshold = np.mean(flux) + self.beat_sensitivity * np.std(flux)
        hop_s = hop / self.SAMPLE_RATE
        min_dist = max(1, int(self.min_action_interval_ms / 1000.0 / hop_s))
        peaks, _ = find_peaks(flux, height=threshold, distance=min_dist)
        return peaks * hop_s

    # ------------------------------------------------- pass 2: tempo + pattern clean

    def _clean_beat_grid(
        self, raw_beats: np.ndarray, energy_envelope: np.ndarray
    ) -> np.ndarray:
        if len(raw_beats) < 3:
            return raw_beats

        hop_s = self.HOP_LENGTH / self.SAMPLE_RATE
        intervals = np.diff(raw_beats)

        # Step 1: segment into tempo-consistent sections
        window = 6
        sections: List[Tuple[int, int, float]] = []
        i = 0
        while i < len(intervals):
            end = min(i + window, len(intervals))
            local = intervals[i:end]
            local_median = float(np.median(local))
            j = end
            while j < len(intervals):
                if abs(intervals[j] - local_median) / local_median < 0.4:
                    j += 1
                    local_median = float(np.median(intervals[i:j]))
                else:
                    break
            sections.append((i, j, local_median))
            i = j if j > end else end
        self.logger.info(f"  Found {len(sections)} tempo sections")

        # Step 2: drop outlier beats inside each section
        keep = np.ones(len(raw_beats), dtype=bool)
        for sec_start, sec_end, sec_median in sections:
            for k in range(sec_start, sec_end):
                interval = intervals[k]
                if interval < sec_median * 0.4:
                    keep[k + 1] = False
                    self.logger.debug(
                        f"  Removing spurious beat at {raw_beats[k+1]:.2f}s "
                        f"(interval {interval*1000:.0f}ms vs median {sec_median*1000:.0f}ms)"
                    )

        clean = raw_beats[keep]
        removed = len(raw_beats) - len(clean)
        if removed > 0:
            self.logger.info(f"  Removed {removed} outlier beats")
        if len(clean) < 2:
            return clean

        # Step 3: fill gaps where beats were missed
        clean_intervals = np.diff(clean)
        filled = [clean[0]]
        for i in range(len(clean_intervals)):
            interval = clean_intervals[i]
            local_start = max(0, i - 4)
            local_end = min(len(clean_intervals), i + 4)
            local_median = float(np.median(clean_intervals[local_start:local_end]))
            if interval > local_median * 1.8:
                n_fill = round(interval / local_median)
                if n_fill >= 2:
                    fill_interval = interval / n_fill
                    for f in range(1, n_fill):
                        fill_time = clean[i] + f * fill_interval
                        energy_idx = min(int(fill_time / hop_s), len(energy_envelope) - 1)
                        if (
                            energy_idx >= 0
                            and energy_envelope[energy_idx] > self.silence_threshold
                        ):
                            filled.append(fill_time)
            filled.append(clean[i + 1])
        filled_arr = np.array(sorted(filled))
        added = len(filled_arr) - len(clean)
        if added > 0:
            self.logger.info(f"  Filled {added} missing beats in gaps")

        if len(filled_arr) < 2:
            return filled_arr

        # Step 4: expand each beat into a beat + midpoint pair (1 cycle)
        expanded: List[float] = []
        for i in range(len(filled_arr)):
            expanded.append(float(filled_arr[i]))
            if i < len(filled_arr) - 1:
                expanded.append((filled_arr[i] + filled_arr[i + 1]) / 2.0)
            elif i > 0:
                expanded.append(filled_arr[i] + (filled_arr[i] - filled_arr[i - 1]) / 2.0)
        self.logger.info(
            f"  Expanded {len(filled_arr)} beats to {len(expanded)} action points"
        )

        # Step 5: quantize timing onto a regular local grid
        expanded_arr = np.array(expanded)
        if len(expanded_arr) > 4:
            quantized = [expanded_arr[0]]
            window_q = 8
            for i in range(1, len(expanded_arr)):
                local_start = max(0, i - window_q)
                local_end = min(len(expanded_arr), i + window_q)
                local_intervals = np.diff(expanded_arr[local_start:local_end])
                if len(local_intervals) > 0:
                    expected_interval = float(np.median(local_intervals))
                    expected_time = quantized[-1] + expected_interval
                    deviation = abs(expanded_arr[i] - expected_time) / expected_interval
                    if deviation < 0.3:
                        quantized.append(expected_time)
                    else:
                        quantized.append(float(expanded_arr[i]))
                else:
                    quantized.append(float(expanded_arr[i]))
            expanded_arr = np.array(quantized)
            self.logger.info("  Quantized timing to regular grid")

        return expanded_arr

    # -------------------------------------------- pass 3: alternating action gen

    def _generate_actions(
        self,
        beat_times: np.ndarray,
        energy_envelope: np.ndarray,
        audio_duration_s: float,
    ) -> List[Dict]:
        if len(beat_times) == 0:
            return []

        hop_s = self.HOP_LENGTH / self.SAMPLE_RATE
        center = 50
        half_target = self.target_amplitude / 2.0
        half_min = self.min_amplitude / 2.0

        # Step 1: raw amplitude per beat (energy-scaled)
        raw_half_amps: List[float] = []
        beat_energies: List[float] = []
        for t in beat_times:
            energy_idx = min(int(t / hop_s), len(energy_envelope) - 1)
            energy = float(energy_envelope[energy_idx]) if energy_idx >= 0 else 0.5
            beat_energies.append(energy)
            raw_half_amps.append(half_min + (half_target - half_min) * energy)

        raw_half_amps_arr = np.array(raw_half_amps)
        beat_energies_arr = np.array(beat_energies)

        # Step 2: speed-limit amplitude (device max-throughput)
        intervals_ms = np.diff(beat_times) * 1000.0
        for i in range(len(intervals_ms)):
            if intervals_ms[i] < 1:
                continue
            max_distance = self.MAX_DEVICE_SPEED * (intervals_ms[i] / 1000.0)
            full_stroke = 2 * raw_half_amps_arr[i]
            if full_stroke > max_distance:
                capped_half = max_distance / 2.0
                raw_half_amps_arr[i] = min(raw_half_amps_arr[i], capped_half)
                if i + 1 < len(raw_half_amps_arr):
                    raw_half_amps_arr[i + 1] = min(raw_half_amps_arr[i + 1], capped_half)

        # Step 3: harmonize within tempo sections (consistent peaks/lows)
        if len(raw_half_amps_arr) > 8:
            raw_half_amps_arr = uniform_filter1d(raw_half_amps_arr, size=16)

        # Step 4: alternating top/bottom action stream
        actions: List[Dict] = []
        stroke_index = 0
        for i, t in enumerate(beat_times):
            if beat_energies_arr[i] < self.silence_threshold:
                continue
            half_amp = raw_half_amps_arr[i]
            pos = center - half_amp if stroke_index % 2 == 0 else center + half_amp
            pos = max(0, min(100, int(round(pos))))
            actions.append({"at": int(round(t * 1000)), "pos": pos})
            stroke_index += 1

        # Deduplicate on timestamp
        seen: Dict[int, Dict] = {}
        for a in actions:
            seen[a["at"]] = a
        return sorted(seen.values(), key=lambda a: a["at"])
