#!/usr/bin/env python3
"""
VR Hybrid Chapter-Aware Tracker — Offline pipeline with per-chapter optimization.

Architecture:
  Pass 1: Sparse YOLO detection (2-4 fps) → lightweight chapter map
  Pass 2: Per-chapter Stage 1 YOLO → ROI optical flow from detections
  Pass 3: Merge chapter funscripts, handle close-up/NR, crossfade boundaries

Stage 1 generates preprocessed video (640p, VR dewarped) + per-frame YOLO
detections in msgpack. Stage 2 (contact analysis) is replaced by DIS optical
flow computed in the penis ROI from Stage 1 detections, with offline
SavGol smoothing + percentile amplitude normalization.

Version: 2.0.0
"""

import logging
import os
import time
import json
import threading
import numpy as np
import cv2
import msgpack
from typing import Dict, Any, Optional, List, Tuple, Callable
from multiprocessing import Event
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import median_filter
from config import constants as config_constants

try:
    from ..core.base_offline_tracker import BaseOfflineTracker, OfflineProcessingResult, OfflineProcessingStage
    from ..core.base_tracker import TrackerMetadata, StageDefinition
except ImportError:
    from tracker.tracker_modules.core.base_offline_tracker import BaseOfflineTracker, OfflineProcessingResult, OfflineProcessingStage
    from tracker.tracker_modules.core.base_tracker import TrackerMetadata, StageDefinition

# Ensure project root is on path for video processor imports
try:
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    MODULES_AVAILABLE = True
except Exception:
    MODULES_AVAILABLE = False

# YOLO detection via unified helper
try:
    from tracker.tracker_modules.helpers.yolo_detection_helper import (
        load_model as _yolo_load_model, run_detection as _yolo_run_detection,
        detection_to_dict as _yolo_det_to_dict,
    )
    YOLO_AVAILABLE = True
except ImportError:
    _yolo_load_model = None
    _yolo_run_detection = None
    _yolo_det_to_dict = None
    YOLO_AVAILABLE = False

# Chapter detection helpers (shared with Chapter Maker)
from tracker.tracker_modules.helpers.chapter_detection import (
    CONTACT_TO_POSITION, CONTACT_PRIORITY, ACTIVE_POSITIONS,
    parse_detections, build_contact_info,
    classify_frame_position, classify_no_penis, classify_segment_spatial,
    build_chapters,
)

# Constants
SPARSE_FPS = 2  # Frames per second for chapter detection
BOUNDARY_CROSSFADE_FRAMES = 15  # Frames to crossfade at chapter boundaries
DEFAULT_CONFIDENCE = 0.25  # Low confidence for sparse pass (false positives are OK)
MAX_PARALLEL_CHAPTERS = 2  # Max chapters to process in parallel

# Per-chapter-type amplitude targets (p5-p95 range in position units)
CHAPTER_TYPE_CONFIG = {
    'Cowgirl / Missionary': {'target_range': 70},
    'Rev. Cowgirl / Doggy': {'target_range': 75},
    'Blowjob':              {'target_range': 55},
    'Handjob':              {'target_range': 50},
    'Boobjob':              {'target_range': 45},
    'Footjob':              {'target_range': 40},
}
FLOW_MEDIAN_WINDOW = 3       # Median filter for raw flow values
SAVGOL_WINDOW = 7            # SavGol smoothing window (~117ms at 60fps)
SAVGOL_POLYORDER = 2         # SavGol polynomial order
ROI_PADDING_FACTOR = 0.3     # Padding around union of penis + contact boxes
FLOW_SCALE = 10.0            # Flow-to-position scaling factor
MIN_PEAK_PROMINENCE = 8      # Minimum prominence for peak detection
MIN_PEAK_DISTANCE_S = 0.25   # Minimum seconds between peaks



class VRHybridChapterTracker(BaseOfflineTracker):
    """
    VR Hybrid chapter-aware offline tracker.

    Pass 1: Sparse YOLO at ~2fps to detect chapters (position types)
    Pass 2: Stage 1 YOLO per chapter (preprocessed video + detections),
            then ROI optical flow from Stage 1 detections (replaces Stage 2)
    Pass 3: Merge results with crossfade at boundaries
    """

    def __init__(self):
        super().__init__()
        self.app = None
        self.yolo_input_size = 640
        self.video_type = "auto"
        self.vr_input_format = "he"
        self.vr_fov = 190
        self.vr_pitch = 0
        self.sparse_fps = SPARSE_FPS
        self.max_parallel = MAX_PARALLEL_CHAPTERS
        self._overlay_frames = []

    @property
    def metadata(self) -> TrackerMetadata:
        return TrackerMetadata(
            name="OFFLINE_VR_HYBRID_CHAPTER",
            display_name="VR Hybrid Chapter-Aware (ROI Flow)",
            description="Chapter detection at 2fps, then YOLO + ROI optical flow per chapter",
            category="offline",
            version="2.0.0",
            author="FunGen",
            tags=["offline", "hybrid", "chapter-aware", "optimized", "batch"],
            requires_roi=False,
            supports_dual_axis=False,
            primary_axis="stroke",
            stages=[
                StageDefinition(
                    stage_number=1,
                    name="Sparse Chapter Detection",
                    description="Lightweight YOLO at 2fps for position classification",
                    produces_funscript=False,
                    requires_previous=False,
                    output_type="analysis"
                ),
                StageDefinition(
                    stage_number=2,
                    name="Per-Chapter Analysis & Funscript",
                    description="Full Stage 1+2 on relevant chapters, parallel processing",
                    produces_funscript=True,
                    requires_previous=True,
                    output_type="funscript"
                )
            ],
            properties={
                "produces_funscript_in_stage2": True,
                "supports_batch": True,
                "supports_range": False,
                "is_hybrid_tracker": True,
                "num_stages": 2
            }
        )

    @property
    def processing_stages(self) -> List[OfflineProcessingStage]:
        return [OfflineProcessingStage.STAGE_2]

    @property
    def stage_dependencies(self) -> Dict[OfflineProcessingStage, List[OfflineProcessingStage]]:
        # We handle Stage 1 internally (sparse + per-chapter dense)
        # so we declare no external dependencies
        return {OfflineProcessingStage.STAGE_2: []}

    def initialize(self, app_instance, **kwargs) -> bool:
        try:
            self.app = app_instance
            if not MODULES_AVAILABLE:
                self.logger.error("Required modules not available")
                return False
            if not YOLO_AVAILABLE:
                self.logger.error("YOLO not available for sparse detection")
                return False
            self._initialized = True
            self.logger.info("VR Hybrid Chapter Tracker initialized")
            return True
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}", exc_info=True)
            return False

    def can_resume_from_checkpoint(self, checkpoint_data: Dict[str, Any]) -> bool:
        return False  # No checkpoint support yet

    def estimate_processing_time(self, stage, video_path, **kwargs) -> float:
        try:
            from video.video_processor import VideoProcessor
            # Rough estimate: sparse pass is ~10% of full, dense pass covers ~60-70% of video
            # So total is roughly 70-80% of full Stage 1+2 time
            return 300.0  # Fallback 5 minutes
        except Exception:
            return 300.0

    def process_stage(self,
                     stage: OfflineProcessingStage,
                     video_path: str,
                     input_data: Optional[Dict[str, Any]] = None,
                     input_files: Optional[Dict[str, str]] = None,
                     output_directory: Optional[str] = None,
                     progress_callback: Optional[Callable] = None,
                     frame_range: Optional[Tuple[int, int]] = None,
                     resume_data: Optional[Dict[str, Any]] = None,
                     **kwargs) -> OfflineProcessingResult:
        """Main processing: sparse chapter detection → per-chapter dense processing → merge."""

        if not self._initialized:
            return OfflineProcessingResult(success=False, error_message="Tracker not initialized")

        try:
            start_time = time.time()
            self.processing_active = True
            self.stop_event = self.stop_event or Event()

            if not output_directory:
                output_directory = os.path.dirname(video_path)
            os.makedirs(output_directory, exist_ok=True)

            video_basename = os.path.splitext(os.path.basename(video_path))[0]

            # Extract preprocessed video settings from kwargs
            self._save_preprocessed_video = kwargs.get('save_preprocessed_video', True)
            self._hwaccel_method = kwargs.get('hwaccel_method', 'auto')
            preprocessed_path = kwargs.get('preprocessed_video_path', None)
            if preprocessed_path is None:
                preprocessed_path = os.path.join(output_directory, f'{video_basename}_preprocessed.mp4')

            # Load settings from app
            self._load_settings()

            # --- PASS 1: Sparse Chapter Detection ---
            self.logger.info("=== Pass 1: Sparse Chapter Detection ===")
            if progress_callback:
                progress_callback({'stage': 'pass1', 'task': 'Sparse chapter detection', 'percentage': 0})

            chapters = self._sparse_chapter_detection(video_path, output_directory, progress_callback,
                                                      preprocessed_path=preprocessed_path)

            if self.stop_event.is_set():
                return OfflineProcessingResult(success=False, error_message="Processing stopped")

            if not chapters:
                self.logger.warning("No chapters detected, falling back to full-video processing")
                chapters = [{'start_frame': 0, 'end_frame': -1, 'position': 'Unknown', 'dense': True}]

            self.logger.info(f"Detected {len(chapters)} chapters:")
            for ch in chapters:
                self.logger.info(f"  [{ch['start_frame']}-{ch['end_frame']}] {ch['position']} "
                               f"({'PROCESS' if ch.get('dense') else 'SKIP/100'})")

            # --- PASS 2: Per-Chapter Dense Processing ---
            self.logger.info("=== Pass 2: Per-Chapter Dense Processing ===")
            if progress_callback:
                progress_callback({'stage': 'pass2', 'task': 'Per-chapter processing', 'percentage': 30})

            chapter_results = self._process_chapters_parallel(
                video_path, chapters, output_directory, progress_callback
            )

            if self.stop_event.is_set():
                return OfflineProcessingResult(success=False, error_message="Processing stopped")

            # --- PASS 3: Merge Results ---
            self.logger.info("=== Pass 3: Merge Chapter Results ===")
            if progress_callback:
                progress_callback({'stage': 'pass3', 'task': 'Merging results', 'percentage': 90})

            funscript = self._merge_chapter_results(chapters, chapter_results, video_path)

            # Save overlay data for debug replay
            overlay_path = None
            if self._overlay_frames:
                overlay_path = os.path.join(output_directory, f'{video_basename}_stage2_overlay.msgpack')
                overlay_data = {
                    'frames': sorted(self._overlay_frames, key=lambda f: f['frame_id']),
                    'segments': [
                        {'start_frame': ch['start_frame'], 'end_frame': ch['end_frame'],
                         'major_position': ch.get('position', 'Unknown'),
                         'position_short_name': ch.get('position', 'Unknown')}
                        for ch in chapters
                    ],
                    'metadata': {'schema': 'v1.1', 'source': 'vr_hybrid_chapter_tracker'},
                }
                try:
                    with open(overlay_path, 'wb') as f:
                        f.write(msgpack.packb(overlay_data, use_bin_type=True))
                    self.logger.info(f"Saved overlay data: {len(self._overlay_frames)} frames")
                except Exception as e:
                    self.logger.warning(f"Failed to save overlay data: {e}")
                    overlay_path = None

            processing_time = time.time() - start_time
            self.logger.info(f"VR Hybrid processing complete in {processing_time:.1f}s")

            if progress_callback:
                progress_callback({'stage': 'complete', 'task': 'Done', 'percentage': 100})

            self.processing_active = False

            return OfflineProcessingResult(
                success=True,
                output_data={
                    'funscript': funscript,
                    'chapters': chapters,
                    'chapter_results': {i: r.get('metrics', {}) for i, r in chapter_results.items()},
                    'overlay_path': overlay_path,
                },
                performance_metrics={
                    'processing_time_seconds': processing_time,
                    'chapters_detected': len(chapters),
                    'chapters_processed': sum(1 for ch in chapters if ch.get('dense')),
                    'chapters_skipped': sum(1 for ch in chapters if not ch.get('dense')),
                }
            )

        except Exception as e:
            self.logger.error(f"VR Hybrid tracker error: {e}", exc_info=True)
            self.processing_active = False
            return OfflineProcessingResult(success=False, error_message=str(e))

    def _load_settings(self):
        """Load relevant settings from app instance."""
        if self.app:
            self.yolo_input_size = getattr(self.app, 'yolo_input_size', 640)
            if hasattr(self.app, 'processor') and self.app.processor:
                self.video_type = getattr(self.app.processor, 'video_type_setting', 'auto')
                self.vr_input_format = getattr(self.app.processor, 'vr_input_format', 'he')
                self.vr_fov = getattr(self.app.processor, 'vr_fov', 190)
                self.vr_pitch = getattr(self.app.processor, 'vr_pitch', 0)

    # -------------------------------------------------------------------------
    # Pass 1: Sparse Chapter Detection
    # -------------------------------------------------------------------------

    def _sparse_chapter_detection(self, video_path: str, output_dir: str,
                                   progress_callback: Optional[Callable],
                                   preprocessed_path: Optional[str] = None) -> List[Dict]:
        """
        Stream ALL frames through v360→640p, creating a full preprocessed video
        with exact frame count. Run YOLO only every Nth frame for chapter detection.

        This single-pass approach avoids per-chapter re-decode/dewarp and ensures
        the preprocessed video has every frame for accurate optical flow later.

        If a preprocessed video already exists at preprocessed_path, encoding is
        skipped and frames are streamed from the existing file instead.

        Also saves per-frame YOLO detections (sparse) to a msgpack file.

        Returns list of chapter dicts:
          {'start_frame': int, 'end_frame': int, 'position': str, 'dense': bool}
        """
        from detection.cd.stage_1_cd import FFmpegEncoder
        from video.video_processor import VideoProcessor

        # Load YOLO model
        yolo_model_path = getattr(self.app, 'yolo_det_model_path', None)
        if not yolo_model_path or not os.path.exists(yolo_model_path):
            self.logger.error(f"YOLO model not found: {yolo_model_path}")
            return []

        model = _yolo_load_model(yolo_model_path)
        self._yolo_model = model

        # Create VP for video info and filter building (streams ALL frames)
        vp = VideoProcessor(
            app_instance=self.app,
            tracker=None,
            yolo_input_size=self.yolo_input_size,
            video_type=self.video_type,
            vr_input_format=self.vr_input_format,
            vr_fov=self.vr_fov,
            vr_pitch=self.vr_pitch,
        )

        if not vp.open_video(video_path):
            self.logger.error(f"VideoProcessor failed to open: {video_path}")
            return []

        fps = vp.fps or 30.0
        total_frames = vp.total_frames
        frame_skip = max(1, int(fps / self.sparse_fps))

        # Force CPU v360 dewarping for consistent output
        vp.vr_unwarp_method_override = 'v360'
        vp.gpu_unwarp_enabled = False
        vp._update_video_parameters()

        # NO select filter — stream ALL frames for the preprocessed video
        self.logger.info(f"Full decode + sparse YOLO: {total_frames} frames @ {fps:.1f}fps, "
                        f"YOLO every {frame_skip}th frame")

        # Check if we can reuse an existing preprocessed video
        if preprocessed_path is None:
            preprocessed_path = os.path.join(output_dir, os.path.splitext(os.path.basename(video_path))[0] + '_preprocessed.mp4')
        self._preprocessed_video_path = preprocessed_path
        self._preprocessed_fps = fps

        reuse_preprocessed = (preprocessed_path and os.path.exists(preprocessed_path)
                              and os.path.getsize(preprocessed_path) > 0)

        encoder = None
        reuse_cap = None
        if reuse_preprocessed:
            # Validate the existing file can be opened
            test_cap = cv2.VideoCapture(preprocessed_path)
            if test_cap.isOpened() and test_cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
                self.logger.info(f"Reusing existing preprocessed video: {preprocessed_path}")
                reuse_cap = test_cap
            else:
                test_cap.release()
                self.logger.warning("Existing preprocessed video is invalid, re-encoding")
                reuse_preprocessed = False

        if not reuse_preprocessed:
            hwaccel = getattr(self, '_hwaccel_method', 'auto')
            encoder = FFmpegEncoder(
                output_file=preprocessed_path,
                width=self.yolo_input_size,
                height=self.yolo_input_size,
                fps=fps,
                ffmpeg_path='ffmpeg',
                hwaccel_method=hwaccel,
            )
            encoder.start()

        frame_positions = {}
        sparse_detections = {}  # frame_id → list of detections
        penis_frames = set()    # frames where penis was detected
        frame_contact_info = {} # frame_id → list of {class, box, conf, norm_y, norm_area}
        penis_class_name = 'penis'
        frames_written = 0
        yolo_processed = 0
        pass1_start = time.time()
        yolo_time_accum = 0.0
        decode_time_accum = 0.0
        timing_samples = 0

        # Choose frame source: reuse preprocessed file or VP pipeline
        if reuse_preprocessed and reuse_cap is not None:
            frame_source = self._frames_from_capture(reuse_cap, total_frames)
        else:
            frame_source = self._frames_from_vp(vp, total_frames, encoder)

        try:
            t_frame_start = time.perf_counter()
            for frame_idx, frame in frame_source:

                if self.stop_event and self.stop_event.is_set():
                    break

                t_frame_end = time.perf_counter()
                decode_ms = (t_frame_end - t_frame_start) * 1000.0
                frames_written += 1

                # Run YOLO only every Nth frame
                if frame_idx % frame_skip != 0:
                    t_frame_start = time.perf_counter()
                    continue

                t_yolo_start = time.perf_counter()
                try:
                    det_objs = _yolo_run_detection(model, frame,
                                                   conf=DEFAULT_CONFIDENCE,
                                                   imgsz=self.yolo_input_size,
                                                   device=config_constants.DEVICE)
                except Exception as e:
                    self.logger.debug(f"YOLO error on frame {frame_idx}: {e}")
                    t_frame_start = time.perf_counter()
                    continue
                yolo_ms = (time.perf_counter() - t_yolo_start) * 1000.0
                decode_time_accum += decode_ms
                yolo_time_accum += yolo_ms
                timing_samples += 1

                # Parse detections
                penis_box = None
                other_boxes = []
                frame_dets = [_yolo_det_to_dict(d) for d in det_objs]

                for d in det_objs:
                    if d.class_name == penis_class_name:
                        if penis_box is None or d.confidence > penis_box['conf']:
                            penis_box = {'box': d.bbox, 'conf': d.confidence}
                    elif d.class_name in CONTACT_TO_POSITION:
                        other_boxes.append({
                            'box': d.bbox,
                            'class': d.class_name,
                            'conf': d.confidence
                        })

                sparse_detections[frame_idx] = frame_dets

                # Store spatial info for contact boxes
                frame_h = frame.shape[0] if frame is not None else 640
                frame_w = frame.shape[1] if frame is not None else 640
                if other_boxes:
                    contacts_info = []
                    for ob in other_boxes:
                        bx1, by1, bx2, by2 = ob['box']
                        contacts_info.append({
                            'class': ob['class'],
                            'box': ob['box'],
                            'conf': ob['conf'],
                            'norm_cy': ((by1 + by2) / 2.0) / frame_h,  # 0=top, 1=bottom
                            'norm_area': ((bx2-bx1) * (by2-by1)) / (frame_h * frame_w),
                        })
                    frame_contact_info[frame_idx] = contacts_info

                # Classify position
                if penis_box:
                    penis_frames.add(frame_idx)
                    position = classify_frame_position(penis_box, other_boxes)
                elif len(other_boxes) >= 2:
                    # No penis visible but multiple contact boxes → likely occluded
                    position = classify_no_penis(other_boxes, frame_h)
                else:
                    position = 'Not Relevant'

                frame_positions[frame_idx] = position
                yolo_processed += 1

                # Collect overlay frame for debug replay
                if frame_dets:
                    self._overlay_frames.append({
                        'frame_id': frame_idx,
                        'yolo_boxes': [
                            {'bbox': d['bbox'], 'class_name': d['class_name'],
                             'confidence': d.get('confidence', 0.0), 'track_id': None, 'status': None}
                            for d in frame_dets
                        ],
                        'poses': [],
                        'dominant_pose_id': None,
                        'active_interaction_track_id': None,
                        'is_occluded': False,
                        'atr_assigned_position': position,
                    })

                if progress_callback and yolo_processed % 50 == 0:
                    pct = min(30, int(30 * frame_idx / max(1, total_frames)))
                    elapsed = time.time() - pass1_start
                    avg_decode = decode_time_accum / max(1, timing_samples)
                    avg_yolo = yolo_time_accum / max(1, timing_samples)
                    progress_callback({'stage': 'pass1',
                                      'task': f'Decode + sparse YOLO ({yolo_processed} det, {frames_written} frames)',
                                      'percentage': pct,
                                      'timing': {'decode_ms': avg_decode, 'yolo_det_ms': avg_yolo},
                                      'time_elapsed': elapsed,
                                      'avg_fps': yolo_processed / max(0.001, elapsed)})

                t_frame_start = time.perf_counter()

        except Exception as e:
            self.logger.error(f"Pass 1 error: {e}", exc_info=True)
        finally:
            if reuse_cap is not None:
                reuse_cap.release()

        # Close encoder
        if encoder is not None:
            encoder.stop()
            if self.stop_event and self.stop_event.is_set():
                # Abort — delete incomplete file
                try:
                    if os.path.exists(preprocessed_path):
                        os.remove(preprocessed_path)
                        self.logger.info("Deleted incomplete preprocessed video (aborted)")
                except OSError:
                    pass

        self.logger.info(f"Pass 1 complete: {frames_written} frames written to preprocessed video, "
                        f"{yolo_processed} YOLO detections, "
                        f"penis in {sum(1 for p in frame_positions.values() if p != 'Not Relevant')}")

        # Save sparse detections to msgpack for use in Pass 2
        sparse_det_path = os.path.join(output_dir, 'sparse_detections.msgpack')
        try:
            with open(sparse_det_path, 'wb') as f:
                f.write(msgpack.packb(sparse_detections, use_bin_type=True))
            self._sparse_detections_path = sparse_det_path
        except Exception as e:
            self.logger.error(f"Failed to save sparse detections: {e}")

        # Store total frames for flow pass
        self._total_frames = frames_written

        # Build chapters from frame positions
        chapters = build_chapters(frame_positions, fps, total_frames, frame_skip,
                                        penis_frames=penis_frames,
                                        frame_contact_info=frame_contact_info)

        # Mark which chapters need dense processing (optical flow)
        for ch in chapters:
            ch['dense'] = ch['position'] in ACTIVE_POSITIONS
            ch['duration_s'] = round((ch['end_frame'] - ch['start_frame']) / fps, 1)

        return chapters

    def _frames_from_vp(self, vp, total_frames, encoder):
        """Yield (frame_idx, frame) from VideoProcessor, encoding each frame."""
        for frame_idx, frame, timing in vp.stream_frames_for_segment(
                0, total_frames, stop_event=self.stop_event):
            if encoder is not None:
                encoder.encode_frame(frame.tobytes())
            yield frame_idx, frame

    def _frames_from_capture(self, cap, total_frames):
        """Yield (frame_idx, frame) from an existing preprocessed video file."""
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame_idx, frame
            frame_idx += 1


    # -------------------------------------------------------------------------
    # Pass 2: Per-Chapter Dense Processing
    # -------------------------------------------------------------------------

    def _process_chapters_parallel(self, video_path: str, chapters: List[Dict],
                                    output_dir: str,
                                    progress_callback: Optional[Callable]) -> Dict[int, Dict]:
        """Process each chapter that needs dense analysis, with parallelization."""

        results = {}
        dense_chapters = [(i, ch) for i, ch in enumerate(chapters) if ch.get('dense')]

        if not dense_chapters:
            self.logger.info("No chapters need dense processing")
            return results

        self.logger.info(f"Processing {len(dense_chapters)} chapters densely "
                        f"(max {self.max_parallel} parallel)")

        # Process chapters — use sequential for now to be safe with YOLO model loading
        # Can switch to parallel with ThreadPoolExecutor if needed
        for idx, (chapter_idx, ch) in enumerate(dense_chapters):
            if self.stop_event and self.stop_event.is_set():
                break

            self.logger.info(f"Processing chapter {idx+1}/{len(dense_chapters)}: "
                           f"{ch['position']} [{ch['start_frame']}-{ch['end_frame']}]")

            if progress_callback:
                base_pct = 30
                range_pct = 60
                pct = base_pct + int(range_pct * idx / max(1, len(dense_chapters)))
                progress_callback({'stage': 'pass2',
                                  'task': f"Chapter {idx+1}/{len(dense_chapters)}: {ch['position']}",
                                  'percentage': pct})

            # Build a sub-progress callback for per-frame updates within each chapter
            def make_chapter_progress(ch_idx, ch_total, base, rng, cb):
                def _sub(frame_frac, extra=None):
                    ch_base = base + int(rng * ch_idx / max(1, ch_total))
                    ch_range = rng / max(1, ch_total)
                    pct = ch_base + int(ch_range * frame_frac)
                    info = {'stage': 'pass2',
                            'task': f"Chapter {ch_idx+1}/{ch_total}: {ch['position']} ({int(frame_frac*100)}%)",
                            'percentage': pct}
                    if extra and isinstance(extra, dict):
                        info.update(extra)
                    cb(info)
                return _sub

            chapter_progress = None
            if progress_callback:
                chapter_progress = make_chapter_progress(idx, len(dense_chapters), 30, 60, progress_callback)

            chapter_result = self._process_single_chapter(
                video_path, ch, chapter_idx, output_dir,
                chapter_progress_callback=chapter_progress,
            )
            results[chapter_idx] = chapter_result

        return results

    def _process_single_chapter(self, video_path: str, chapter: Dict,
                                 chapter_idx: int, output_dir: str,
                                 chapter_progress_callback: Optional[Callable] = None) -> Dict:
        """Run ROI optical flow on a single chapter using the full preprocessed video."""

        start_frame = chapter['start_frame']
        end_frame = chapter['end_frame']

        try:
            # Use the full preprocessed video from Pass 1 (no per-chapter re-encoding)
            preprocessed_path = getattr(self, '_preprocessed_video_path', None)
            if not preprocessed_path or not os.path.exists(preprocessed_path):
                self.logger.error(f"  No preprocessed video available for chapter {chapter_idx}")
                return {'success': False, 'error': 'No preprocessed video'}

            # Load sparse detections from Pass 1
            sparse_det_path = getattr(self, '_sparse_detections_path', None)

            self.logger.info(f"  ROI Flow: frames {start_frame}-{end_frame}")

            flow_result = self._process_chapter_roi_flow(
                video_path, chapter, chapter_idx, output_dir,
                preprocessed_video=preprocessed_path,
                sparse_det_path=sparse_det_path,
                frame_progress_callback=chapter_progress_callback,
            )

            if not flow_result or not flow_result.get('raw_positions'):
                self.logger.error(f"  ROI flow failed for chapter {chapter_idx}")
                return {'success': False, 'error': 'ROI flow produced no data'}

            # --- Post-processing ---
            raw_positions = flow_result['raw_positions']
            fps = flow_result['fps']
            invert = flow_result.get('invert', False)
            chapter_type = chapter.get('position', 'Unknown')

            primary_actions, secondary_actions = self._postprocess_chapter_signal(
                raw_positions, chapter_type, fps, start_frame, invert=invert
            )

            self.logger.info(f"  Chapter {chapter_idx} complete: {len(primary_actions)} actions")

            return {
                'success': True,
                'primary_actions': primary_actions,
                'secondary_actions': secondary_actions,
                'metrics': {
                    'actions': len(primary_actions),
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'raw_frames': len(raw_positions),
                }
            }

        except Exception as e:
            self.logger.error(f"  Chapter {chapter_idx} processing error: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    # -------------------------------------------------------------------------
    # ROI Optical Flow (replaces Stage 2)
    # -------------------------------------------------------------------------

    def _process_chapter_roi_flow(self, video_path: str, chapter: Dict,
                                   chapter_idx: int, output_dir: str,
                                   preprocessed_video: str,
                                   sparse_det_path: Optional[str] = None,
                                   frame_progress_callback: Optional[Callable] = None) -> Optional[Dict]:
        """
        Compute DIS optical flow in penis ROI for a chapter's frame range.

        Reads the full preprocessed video (created in Pass 1, already 640x640).
        Runs YOLO on every frame for per-frame ROI (preprocessed video is already
        the right size for YOLO inference — no resize needed).
        Falls back to sparse detections if YOLO model unavailable.

        Returns dict with 'raw_positions', 'fps', 'invert'.
        """
        start_frame = chapter['start_frame']
        end_frame = chapter['end_frame']

        # Get YOLO model for dense per-frame detection
        yolo_model = getattr(self, '_yolo_model', None)
        use_dense_yolo = yolo_model is not None

        # Load sparse detections as fallback
        frame_boxes = {}
        if not use_dense_yolo and sparse_det_path and os.path.exists(sparse_det_path):
            try:
                with open(sparse_det_path, 'rb') as f:
                    sparse_dets = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
                for frame_idx_str, dets in sparse_dets.items():
                    frame_idx = int(frame_idx_str) if isinstance(frame_idx_str, str) else frame_idx_str
                    if frame_idx < start_frame or frame_idx > end_frame:
                        continue
                    penis_box = None
                    contact_boxes = []
                    best_conf = 0.0
                    for det in dets:
                        cls_name = det.get('class_name', '')
                        bbox = det.get('bbox', [])
                        conf = det.get('confidence', 0.0)
                        if len(bbox) != 4:
                            continue
                        if cls_name == 'penis' and conf > best_conf:
                            penis_box = tuple(bbox)
                            best_conf = conf
                        elif cls_name in CONTACT_TO_POSITION:
                            contact_boxes.append(tuple(bbox))
                    if penis_box:
                        frame_boxes[frame_idx] = {'penis': penis_box, 'contacts': contact_boxes}
            except Exception as e:
                self.logger.warning(f"  Failed to load sparse detections: {e}")

        if use_dense_yolo:
            self.logger.info(f"  Dense YOLO + flow on preprocessed video [{start_frame}-{end_frame}]")
        else:
            self.logger.info(f"  Sparse ROI ({len(frame_boxes)} dets) + flow [{start_frame}-{end_frame}]")

        # Open full preprocessed video and seek to chapter start
        if not preprocessed_video or not os.path.exists(preprocessed_video):
            self.logger.error(f"  Preprocessed video not found: {preprocessed_video}")
            return None

        cap = cv2.VideoCapture(preprocessed_video)
        if not cap.isOpened():
            self.logger.error(f"  Cannot open preprocessed video")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_ms = 1000.0 / fps

        # Seek to chapter start
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        chapter_frames = end_frame - start_frame + 1

        # Init DIS optical flow
        dis = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)

        raw_positions = []
        upper_motion_sum = 0.0
        lower_motion_sum = 0.0
        prev_gray = None
        last_roi = None
        yolo_det_count = 0
        p2_decode_accum = 0.0
        p2_yolo_accum = 0.0
        p2_flow_accum = 0.0
        p2_timing_n = 0

        for i in range(chapter_frames):
            t_decode = time.perf_counter()
            ret, frame = cap.read()
            decode_ms = (time.perf_counter() - t_decode) * 1000.0
            if not ret:
                break

            if self.stop_event and self.stop_event.is_set():
                cap.release()
                return None

            if frame_progress_callback and i % 100 == 0 and chapter_frames > 0:
                avg_d = p2_decode_accum / max(1, p2_timing_n)
                avg_y = p2_yolo_accum / max(1, p2_timing_n)
                avg_f = p2_flow_accum / max(1, p2_timing_n)
                frame_progress_callback(float(i) / chapter_frames,
                                        {'timing': {'decode_ms': avg_d, 'yolo_det_ms': avg_y, 'flow_ms': avg_f}})

            abs_frame = start_frame + i
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]

            p2_decode_accum += decode_ms
            p2_timing_n += 1

            # Get ROI -- dense YOLO on every frame, or sparse fallback
            roi_box = None
            frame_overlay_dets = []
            if use_dense_yolo:
                t_yolo = time.perf_counter()
                roi_box, frame_overlay_dets = self._detect_roi_yolo(frame, yolo_model, h, w)
                p2_yolo_accum += (time.perf_counter() - t_yolo) * 1000.0
                if roi_box is not None:
                    yolo_det_count += 1
                    last_roi = roi_box
                elif last_roi is not None:
                    roi_box = last_roi  # Hold last known ROI

                # Collect overlay for dense frames
                if frame_overlay_dets:
                    has_active = any(d.get('track_id') == 1 for d in frame_overlay_dets)
                    self._overlay_frames.append({
                        'frame_id': abs_frame,
                        'yolo_boxes': frame_overlay_dets,
                        'poses': [],
                        'dominant_pose_id': None,
                        'active_interaction_track_id': 1 if has_active else None,
                        'is_occluded': False,
                        'atr_assigned_position': chapter.get('position'),
                    })
            else:
                roi_box = self._get_roi_for_frame(abs_frame, frame_boxes, last_roi, h, w)
                if roi_box is not None:
                    last_roi = roi_box

            if roi_box is None and last_roi is None:
                # No ROI at all — use center 60% of frame
                margin_x = int(w * 0.2)
                margin_y = int(h * 0.2)
                roi_box = (margin_x, margin_y, w - margin_x, h - margin_y)
                last_roi = roi_box

            if roi_box is None:
                roi_box = last_roi

            if prev_gray is not None and roi_box is not None:
                rx1, ry1, rx2, ry2 = roi_box
                prev_patch = prev_gray[ry1:ry2, rx1:rx2]
                curr_patch = gray[ry1:ry2, rx1:rx2]

                if prev_patch.shape[0] > 4 and prev_patch.shape[1] > 4 and prev_patch.shape == curr_patch.shape:
                    t_flow = time.perf_counter()
                    try:
                        flow = dis.calc(
                            np.ascontiguousarray(prev_patch),
                            np.ascontiguousarray(curr_patch),
                            None
                        )
                    except cv2.error:
                        flow = None
                    p2_flow_accum += (time.perf_counter() - t_flow) * 1000.0

                    if flow is not None:
                        dy, dx = self._magnitude_weighted_flow(flow)

                        # Track upper vs lower motion for inversion detection
                        patch_h = flow.shape[0]
                        split = patch_h // 3
                        if split > 0 and patch_h - split > 0:
                            upper_vy = np.median(np.abs(flow[:patch_h - split, :, 1]))
                            lower_vy = np.median(np.abs(flow[patch_h - split:, :, 1]))
                            upper_motion_sum += upper_vy
                            lower_motion_sum += lower_vy

                        time_ms = int(abs_frame * frame_ms)
                        raw_positions.append((i, time_ms, dy))

            prev_gray = gray

        cap.release()

        # Detect motion inversion
        invert = False
        if upper_motion_sum > 0 and lower_motion_sum > 0:
            motion_ratio = lower_motion_sum / upper_motion_sum
            if motion_ratio > 1.5:
                invert = True
                self.logger.info(f"  Motion inversion detected: lower/upper={motion_ratio:.2f} → inverting")
            else:
                self.logger.info(f"  Standard motion: lower/upper={motion_ratio:.2f}")

        self.logger.info(f"  ROI flow: {len(raw_positions)} samples from {chapter_frames} chapter frames")

        return {
            'raw_positions': raw_positions,
            'fps': fps,
            'invert': invert,
        }

    def _get_roi_for_frame(self, frame_idx: int, frame_boxes: Dict,
                            last_roi: Optional[Tuple], h: int, w: int) -> Optional[Tuple[int, int, int, int]]:
        """Get padded ROI for a frame from Stage 1 detections, with interpolation fallback."""
        if frame_idx in frame_boxes:
            boxes = frame_boxes[frame_idx]
            return self._compute_padded_roi(boxes['penis'], boxes['contacts'], h, w)

        # Interpolate from nearest detected frames
        nearest_before = None
        nearest_after = None
        for offset in range(1, 30):  # Search up to 30 frames in each direction
            if nearest_before is None and (frame_idx - offset) in frame_boxes:
                nearest_before = frame_idx - offset
            if nearest_after is None and (frame_idx + offset) in frame_boxes:
                nearest_after = frame_idx + offset
            if nearest_before is not None and nearest_after is not None:
                break

        # Use nearest detection
        nearest = nearest_before or nearest_after
        if nearest is not None:
            boxes = frame_boxes[nearest]
            return self._compute_padded_roi(boxes['penis'], boxes['contacts'], h, w)

        return last_roi

    def _detect_roi_yolo(self, frame: np.ndarray, yolo_model,
                          h: int, w: int) -> Tuple[Optional[Tuple[int, int, int, int]], List[Dict]]:
        """Run YOLO on a single frame. Returns (padded_roi_or_None, overlay_dets_list)."""
        try:
            det_objs = _yolo_run_detection(yolo_model, frame,
                                           conf=DEFAULT_CONFIDENCE,
                                           imgsz=self.yolo_input_size,
                                           device=config_constants.DEVICE)
        except Exception:
            return None, []

        if not det_objs:
            return None, []

        penis_box = None
        best_conf = 0.0
        contact_boxes = []
        overlay_dets = []

        for d in det_objs:
            overlay_dets.append({
                'bbox': list(d.bbox), 'class_name': d.class_name,
                'confidence': d.confidence, 'track_id': None, 'status': None,
            })

            if d.class_name == 'penis' and d.confidence > best_conf:
                penis_box = d.bbox
                best_conf = d.confidence
            elif d.class_name in CONTACT_TO_POSITION:
                contact_boxes.append(d.bbox)

        if penis_box is None:
            return None, overlay_dets

        # Mark the selected penis as locked_penis for overlay highlighting
        px1, py1, px2, py2 = penis_box
        for det in overlay_dets:
            b = det['bbox']
            if det['class_name'] == 'penis' and b[0] == px1 and b[1] == py1:
                det['class_name'] = 'locked_penis'
                break

        # Find and mark the nearest contact as active interactor
        if contact_boxes:
            pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
            best_dist = float('inf')
            best_contact_bbox = None
            for cb in contact_boxes:
                cx, cy = (cb[0] + cb[2]) / 2, (cb[1] + cb[3]) / 2
                dist = (pcx - cx) ** 2 + (pcy - cy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_contact_bbox = cb
            if best_contact_bbox:
                bx1, by1 = best_contact_bbox[0], best_contact_bbox[1]
                for det in overlay_dets:
                    b = det['bbox']
                    if b[0] == bx1 and b[1] == by1 and det['class_name'] != 'locked_penis':
                        det['track_id'] = 1
                        break

        return self._compute_padded_roi(penis_box, contact_boxes, h, w), overlay_dets

    def _compute_padded_roi(self, penis_box: Tuple, contact_boxes: List[Tuple],
                             h: int, w: int) -> Tuple[int, int, int, int]:
        """Compute padded ROI from union of penis + nearest contact box."""
        px1, py1, px2, py2 = penis_box

        # Find nearest contact box
        if contact_boxes:
            pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
            best_dist = float('inf')
            best_contact = None
            for cb in contact_boxes:
                cx, cy = (cb[0] + cb[2]) / 2, (cb[1] + cb[3]) / 2
                dist = (pcx - cx) ** 2 + (pcy - cy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_contact = cb

            if best_contact:
                # Union of penis + nearest contact
                px1 = min(px1, best_contact[0])
                py1 = min(py1, best_contact[1])
                px2 = max(px2, best_contact[2])
                py2 = max(py2, best_contact[3])

        # Add padding
        roi_w = px2 - px1
        roi_h = py2 - py1
        pad_x = roi_w * ROI_PADDING_FACTOR
        pad_y = roi_h * ROI_PADDING_FACTOR

        rx1 = max(0, int(px1 - pad_x))
        ry1 = max(0, int(py1 - pad_y))
        rx2 = min(w, int(px2 + pad_x))
        ry2 = min(h, int(py2 + pad_y))

        return (rx1, ry1, rx2, ry2)

    def _magnitude_weighted_flow(self, flow: np.ndarray) -> Tuple[float, float]:
        """Extract magnitude-weighted dy/dx from flow field (lifted from YOLO ROI tracker)."""
        magnitudes = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

        # Gaussian spatial weights (prioritize center of ROI)
        region_h, region_w = flow.shape[:2]
        center_x, sigma_x = region_w / 2, region_w / 4.0
        center_y, sigma_y = region_h / 2, region_h / 4.0

        x_coords = np.arange(region_w)
        y_coords = np.arange(region_h)
        weights_x = np.exp(-((x_coords - center_x) ** 2) / (2 * sigma_x ** 2))
        weights_y = np.exp(-((y_coords - center_y) ** 2) / (2 * sigma_y ** 2))
        spatial_weights = np.outer(weights_y, weights_x)

        combined_weights = magnitudes * spatial_weights
        total_weight = np.sum(combined_weights)

        if total_weight > 0:
            overall_dy = np.sum(flow[..., 1] * combined_weights) / total_weight
            overall_dx = np.sum(flow[..., 0] * combined_weights) / total_weight
        else:
            overall_dy = np.median(flow[..., 1])
            overall_dx = np.median(flow[..., 0])

        return overall_dy, overall_dx

    # -------------------------------------------------------------------------
    # Post-processing: smoothing, normalization, keyframe extraction
    # -------------------------------------------------------------------------

    def _postprocess_chapter_signal(self, raw_positions: List[Tuple],
                                     chapter_type: str, fps: float,
                                     start_frame: int,
                                     invert: bool = False) -> Tuple[List[Dict], List[Dict]]:
        """
        Post-process raw flow positions into funscript actions.

        1. Convert flow dy → cumulative position
        2. Median filter raw values
        3. SavGol smoothing
        4. Amplitude normalization (percentile → target range)
        5. Peak/valley keyframe extraction
        """
        if len(raw_positions) < SAVGOL_WINDOW:
            self.logger.warning(f"  Too few positions ({len(raw_positions)}) for post-processing")
            return [], []

        frame_indices = np.array([p[0] for p in raw_positions])
        time_ms_arr = np.array([p[1] for p in raw_positions])
        dy_values = np.array([p[2] for p in raw_positions])

        # 1. Median filter raw flow (removes impulse noise from bad frames)
        if len(dy_values) >= FLOW_MEDIAN_WINDOW:
            dy_filtered = median_filter(dy_values, size=FLOW_MEDIAN_WINDOW)
        else:
            dy_filtered = dy_values.copy()

        # 2. Integrate flow velocity → cumulative position
        # Default: negate (positive flow dy = content moves down = insertion = lower pos)
        # Inverted (male thrusting): don't negate (lower ROI motion dominates)
        sign = 1.0 if invert else -1.0
        cumulative = sign * np.cumsum(dy_filtered)

        # 3. Detrend: remove slow drift using large-window SavGol
        # 4s window preserves stroke-frequency oscillations (0.5-2s cycles)
        # while removing camera drift and body repositioning
        detrend_win = int(3.0 * fps)  # ~3 seconds
        if detrend_win % 2 == 0:
            detrend_win += 1
        detrend_win = min(detrend_win, len(cumulative))
        if detrend_win < 3:
            detrend_win = 3
        if detrend_win % 2 == 0:
            detrend_win -= 1

        drift = savgol_filter(cumulative, window_length=detrend_win, polyorder=2)
        detrended = cumulative - drift

        # 4. Fine SavGol smoothing (short window for noise, preserves timing)
        win = min(SAVGOL_WINDOW, len(detrended))
        if win % 2 == 0:
            win -= 1
        if win >= 3:
            smoothed = savgol_filter(detrended, window_length=win, polyorder=SAVGOL_POLYORDER)
        else:
            smoothed = detrended.copy()

        # 5. Amplitude normalization: rescale to target range centered at 50
        config = CHAPTER_TYPE_CONFIG.get(chapter_type, {'target_range': 65})
        target_range = config['target_range']

        p5 = np.percentile(smoothed, 5)
        p95 = np.percentile(smoothed, 95)
        current_range = p95 - p5

        if current_range > 0.01:
            scale = target_range / current_range
            center = (p5 + p95) / 2.0
            normalized = (smoothed - center) * scale + 50.0
        else:
            normalized = np.full_like(smoothed, 50.0)

        # Clip to 0-100
        normalized = np.clip(normalized, 0, 100)

        self.logger.info(f"  Signal stats: raw_dy range [{dy_values.min():.2f}, {dy_values.max():.2f}], "
                        f"detrended range [{detrended.min():.2f}, {detrended.max():.2f}], "
                        f"normalized range [{normalized.min():.1f}, {normalized.max():.1f}]")

        # 6. Peak/valley detection
        min_distance = max(3, int(MIN_PEAK_DISTANCE_S * fps))

        peaks, _ = find_peaks(normalized, prominence=MIN_PEAK_PROMINENCE, distance=min_distance)
        valleys, _ = find_peaks(-normalized, prominence=MIN_PEAK_PROMINENCE, distance=min_distance)

        # Combine peaks + valleys + start/end → keyframes
        keyframe_indices = set()
        keyframe_indices.update(peaks.tolist())
        keyframe_indices.update(valleys.tolist())
        # Always include first and last frame
        keyframe_indices.add(0)
        keyframe_indices.add(len(normalized) - 1)

        keyframe_indices = sorted(keyframe_indices)

        # Build funscript actions
        frame_ms = 1000.0 / fps
        primary_actions = []
        for ki in keyframe_indices:
            if ki < len(time_ms_arr):
                # Snap to frame boundary
                t = int(round(time_ms_arr[ki] / frame_ms) * frame_ms)
                pos = int(round(normalized[ki]))
                pos = max(0, min(100, pos))
                primary_actions.append({'at': t, 'pos': pos})

        # Deduplicate timestamps
        seen = {}
        for a in primary_actions:
            seen[a['at']] = a
        primary_actions = sorted(seen.values(), key=lambda a: a['at'])

        self.logger.info(f"  Post-process: {len(raw_positions)} raw → {len(primary_actions)} keyframes "
                        f"({len(peaks)} peaks, {len(valleys)} valleys)")

        return primary_actions, []  # No secondary axis for now

    # -------------------------------------------------------------------------
    # Pass 3: Merge Chapter Results
    # -------------------------------------------------------------------------

    def _merge_chapter_results(self, chapters: List[Dict], chapter_results: Dict[int, Dict],
                                video_path: str) -> Any:
        """Merge per-chapter funscripts into a single MultiAxisFunscript."""
        try:
            from funscript.multi_axis_funscript import MultiAxisFunscript
        except ImportError:
            from funscript.multi_axis_funscript import MultiAxisFunscript

        # Get video FPS for time calculations
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        all_primary = []
        all_secondary = []

        for i, ch in enumerate(chapters):
            start_ms = int(ch['start_frame'] / fps * 1000)
            end_ms = int(ch['end_frame'] / fps * 1000)

            if ch.get('dense') and i in chapter_results:
                result = chapter_results[i]
                if result.get('success') and result.get('primary_actions'):
                    actions = result['primary_actions']
                    all_primary.extend(actions)

                    if result.get('secondary_actions'):
                        all_secondary.extend(result['secondary_actions'])
                    continue

            # Non-dense chapters (Close-up, NR) or failed chapters → pos=100
            if ch['position'] in ('Close up', 'Not Relevant') or not ch.get('dense'):
                all_primary.append({'at': start_ms, 'pos': 100})
                all_primary.append({'at': end_ms, 'pos': 100})
            else:
                # Failed dense chapter — interpolate or hold at 50
                all_primary.append({'at': start_ms, 'pos': 50})
                all_primary.append({'at': end_ms, 'pos': 50})

        # Sort by timestamp
        all_primary.sort(key=lambda a: a['at'])
        all_secondary.sort(key=lambda a: a['at'])

        # Remove duplicate timestamps (keep last)
        all_primary = self._deduplicate_actions(all_primary)
        all_secondary = self._deduplicate_actions(all_secondary)

        # Build funscript
        funscript = MultiAxisFunscript(logger=self.logger)
        funscript.set_axis_actions('primary', all_primary)
        if all_secondary:
            funscript.set_axis_actions('secondary', all_secondary)

        # Set chapters metadata
        funscript_chapters = []
        for ch in chapters:
            start_ms = int(ch['start_frame'] / fps * 1000)
            end_ms = int(ch['end_frame'] / fps * 1000)
            funscript_chapters.append({
                'name': ch['position'],
                'start': start_ms,
                'end': end_ms,
                'startTime': start_ms,
                'endTime': end_ms,
            })
        funscript.chapters = funscript_chapters

        self.logger.info(f"Merged funscript: {len(all_primary)} primary, "
                        f"{len(all_secondary)} secondary actions, "
                        f"{len(funscript_chapters)} chapters")

        return funscript

    def _deduplicate_actions(self, actions: List[Dict]) -> List[Dict]:
        """Keep only the last action at each timestamp."""
        if not actions:
            return actions
        seen = {}
        for a in actions:
            seen[a['at']] = a
        return sorted(seen.values(), key=lambda a: a['at'])
