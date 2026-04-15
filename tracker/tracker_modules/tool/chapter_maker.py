#!/usr/bin/env python3
"""
Chapter Maker — Lightweight chapter-only tracker for VR and 2D POV.

Derived from VR Hybrid Chapter-Aware tracker but stripped of all signal
generation. Runs sparse YOLO at 2fps to classify position types, builds
chapter boundaries, and outputs a chapter map. No optical flow, no
preprocessed video, no funscript.

Speed advantage over VR Hybrid:
  - No preprocessed video encoding (saves ~30-40% of Pass 1 time)
  - No dense Pass 2 (optical flow) — this is the biggest saving
  - No Pass 3 merge / funscript generation
  - Result: typically 3-5x faster than full hybrid tracker

All video decode goes through VideoProcessor, which auto-detects
format (VR/2D) and applies dewarping as needed.

Supports:
  - VR content (auto-detect or manual, VideoProcessor handles v360 dewarp)
  - 2D POV content (VideoProcessor handles decode + resize)

Output:
  - Chapter list: [{start_frame, end_frame, start_s, end_s, duration_s, position}]
  - Saved as JSON alongside video

Version: 1.0.0
"""

import json
import os
import time
import cv2
from typing import Dict, Any, Optional, List, Tuple, Callable
from multiprocessing import Event

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
    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    MODULES_AVAILABLE = True
except Exception:
    MODULES_AVAILABLE = False

# YOLO detection via unified helper
try:
    from tracker.tracker_modules.helpers.yolo_detection_helper import (
        load_model as _yolo_load_model,
        run_detection as _yolo_run_detection,
    )
    YOLO_AVAILABLE = True
except ImportError:
    _yolo_load_model = None
    _yolo_run_detection = None
    YOLO_AVAILABLE = False

# Chapter detection helpers (shared with VR Hybrid tracker)
from tracker.tracker_modules.helpers.chapter_detection import (
    parse_detections, build_contact_info,
    classify_frame_position, classify_no_penis,
    build_chapters, ACTIVE_POSITIONS,
)

SPARSE_FPS = 2
DEFAULT_CONFIDENCE = 0.25


class ChapterMaker(BaseOfflineTracker):
    """
    Lightweight chapter-only tracker for VR and 2D POV content.

    Runs sparse YOLO detection at ~2fps to classify position types,
    then builds a chapter map. No signal generation, no optical flow.
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

    @property
    def metadata(self) -> TrackerMetadata:
        return TrackerMetadata(
            name="OFFLINE_CHAPTER_MAKER",
            display_name="Chapter Maker (VR + 2D POV)",
            description="Fast chapter detection at 2fps — VR and 2D POV, no signal output",
            category="tool",
            version="1.0.0",
            author="FunGen",
            tags=["offline", "chapter-detection", "lightweight", "vr", "2d", "pov"],
            requires_roi=False,
            supports_dual_axis=False,
            primary_axis="stroke",
            stages=[
                StageDefinition(
                    stage_number=1,
                    name="Chapter Detection",
                    description="Sparse YOLO at 2fps for position classification",
                    produces_funscript=False,
                    requires_previous=False,
                    output_type="analysis"
                ),
            ],
            properties={
                "produces_funscript_in_stage2": False,
                "supports_batch": True,
                "supports_range": False,
                "is_hybrid_tracker": True,
                "num_stages": 1,
            }
        )

    @property
    def processing_stages(self) -> List[OfflineProcessingStage]:
        return [OfflineProcessingStage.STAGE_1]

    @property
    def stage_dependencies(self) -> Dict[OfflineProcessingStage, List[OfflineProcessingStage]]:
        return {OfflineProcessingStage.STAGE_1: []}

    def initialize(self, app_instance, **kwargs) -> bool:
        try:
            self.app = app_instance
            if not MODULES_AVAILABLE:
                self.logger.error("Required modules not available")
                return False
            if not YOLO_AVAILABLE:
                self.logger.error("YOLO not available")
                return False
            self._initialized = True
            self.logger.info("Chapter Maker initialized")
            return True
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}", exc_info=True)
            return False

    def can_resume_from_checkpoint(self, checkpoint_data: Dict[str, Any]) -> bool:
        return False

    def estimate_processing_time(self, stage, video_path, **kwargs) -> float:
        from video.frame_source.probe import probe
        p = probe(video_path)
        if p is None or p.fps <= 0:
            return 30.0
        duration_s = p.total_frames / p.fps
        return max(10.0, duration_s * 0.008)

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
        """Single-stage chapter detection."""

        if not self._initialized:
            return OfflineProcessingResult(success=False, error_message="Tracker not initialized")

        try:
            start_time = time.time()
            self.processing_active = True
            self.stop_event = self.stop_event or Event()

            if not output_directory:
                output_directory = os.path.dirname(video_path)
            os.makedirs(output_directory, exist_ok=True)

            self._load_settings()

            if progress_callback:
                progress_callback({'stage': 'chapter_detection', 'task': 'Starting', 'percentage': 0})

            # Single path through VideoProcessor (handles VR dewarp, 2D, auto-detect)
            chapters, stats = self._detect_chapters(video_path, progress_callback)

            if self.stop_event.is_set():
                return OfflineProcessingResult(success=False, error_message="Processing stopped")

            fps = stats.get('fps', 30.0)
            if not chapters:
                chapters = [{'start_frame': 0, 'end_frame': stats.get('total_frames', 0) - 1,
                             'position': 'Unknown'}]

            # Add time fields
            for ch in chapters:
                ch['start_s'] = round(ch['start_frame'] / fps, 2)
                ch['end_s'] = round(ch['end_frame'] / fps, 2)
                ch['duration_s'] = round((ch['end_frame'] - ch['start_frame']) / fps, 1)

            # Log results
            self.logger.info(f"Chapter Maker: {len(chapters)} chapters detected")
            for ch in chapters:
                self.logger.info(f"  [{ch['start_s']:.1f}s - {ch['end_s']:.1f}s] "
                               f"{ch['position']} ({ch['duration_s']:.1f}s)")

            # Save chapter JSON
            video_basename = os.path.splitext(os.path.basename(video_path))[0]
            chapters_path = os.path.join(output_directory, f'{video_basename}_chapters.json')
            chapter_output = {
                'video': os.path.basename(video_path),
                'fps': fps,
                'total_frames': stats.get('total_frames', 0),
                'duration_s': stats.get('duration_s', 0.0),
                'chapters': chapters,
            }
            with open(chapters_path, 'w') as f:
                json.dump(chapter_output, f, indent=2)
            self.logger.info(f"Saved chapters to {chapters_path}")

            processing_time = time.time() - start_time

            if progress_callback:
                progress_callback({'stage': 'complete', 'task': 'Done', 'percentage': 100})

            self.processing_active = False

            return OfflineProcessingResult(
                success=True,
                output_data={
                    'chapters': chapters,
                    'chapters_path': chapters_path,
                },
                output_files={'chapters': chapters_path},
                performance_metrics={
                    'processing_time_seconds': round(processing_time, 1),
                    'chapters_detected': len(chapters),
                    'active_chapters': sum(1 for ch in chapters if ch['position'] in ACTIVE_POSITIONS),
                    'yolo_frames': stats.get('yolo_frames', 0),
                    'total_frames': stats.get('total_frames', 0),
                    'speed_factor': round(stats.get('duration_s', 0) / max(0.1, processing_time), 1),
                }
            )

        except Exception as e:
            self.logger.error(f"Chapter Maker error: {e}", exc_info=True)
            self.processing_active = False
            return OfflineProcessingResult(success=False, error_message=str(e))

    # ─── Settings ──────────────────────────────────────────────────────────────

    def _load_settings(self):
        """Load relevant settings from app instance."""
        if self.app:
            self.yolo_input_size = getattr(self.app, 'yolo_input_size', 640)
            if hasattr(self.app, 'processor') and self.app.processor:
                self.video_type = getattr(self.app.processor, 'video_type_setting', 'auto')
                self.vr_input_format = getattr(self.app.processor, 'vr_input_format', 'he')
                self.vr_fov = getattr(self.app.processor, 'vr_fov', 190)
                self.vr_pitch = getattr(self.app.processor, 'vr_pitch', 0)

    # ─── Chapter Detection (via VideoProcessor) ───────────────────────────────

    def _detect_chapters(self, video_path: str,
                         progress_callback: Optional[Callable]) -> Tuple[List[Dict], Dict]:
        """Sparse chapter detection via VideoProcessor (handles VR dewarp + 2D auto)."""

        from video.video_processor import VideoProcessor

        yolo_model_path = getattr(self.app, 'yolo_det_model_path', None)
        if not yolo_model_path or not os.path.exists(yolo_model_path):
            self.logger.error(f"YOLO model not found: {yolo_model_path}")
            return [], {}

        model = _yolo_load_model(yolo_model_path)

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
            return [], {}

        fps = vp.fps or 30.0
        total_frames = vp.total_frames
        frame_skip = max(1, int(fps / self.sparse_fps))

        # Force CPU v360 for consistent output
        vp.vr_unwarp_method_override = 'v360'
        vp._update_video_parameters()

        self.logger.info(f"Chapter detection: {total_frames} frames @ {fps:.1f}fps, "
                        f"YOLO every {frame_skip}th frame")

        frame_positions = {}
        penis_frames = set()
        frame_contact_info = {}
        yolo_count = 0
        t_start = time.time()

        from tracker.tracker_modules.helpers.async_yolo_worker import AsyncYoloWorker

        def _apply_yolo(f_idx, fh, fw, det_objs, _payload):
            nonlocal yolo_count
            penis_box, other_boxes = parse_detections(det_objs)
            if other_boxes:
                frame_contact_info[f_idx] = build_contact_info(
                    other_boxes, self.yolo_input_size)
            if penis_box:
                penis_frames.add(f_idx)
                position = classify_frame_position(penis_box, other_boxes)
            elif len(other_boxes) >= 2:
                position = classify_no_penis(other_boxes, self.yolo_input_size)
            else:
                position = 'Not Relevant'
            frame_positions[f_idx] = position
            yolo_count += 1

        yolo_worker = AsyncYoloWorker(
            model=model,
            conf=DEFAULT_CONFIDENCE,
            imgsz=self.yolo_input_size,
            device=config_constants.DEVICE,
            on_result=_apply_yolo,
            logger=self.logger,
            input_queue_size=8,
            batch_size=4,
        )
        yolo_worker.start()

        try:
            for frame_idx, frame, timing in vp.stream_frames_for_segment(
                    0, total_frames, stop_event=self.stop_event):

                if self.stop_event and self.stop_event.is_set():
                    break

                if frame_idx % frame_skip != 0:
                    continue

                yolo_worker.submit(frame_idx, frame)
                yolo_worker.drain()

                if progress_callback and yolo_count > 0 and yolo_count % 50 == 0:
                    pct = min(90, int(90 * frame_idx / max(1, total_frames)))
                    elapsed = time.time() - t_start
                    avg_fps = frame_idx / max(0.001, elapsed)
                    eta_s = (total_frames - frame_idx) / avg_fps if avg_fps > 0 else 0.0
                    yolo_ms = (yolo_worker.total_inference_ms /
                               max(1, yolo_worker.completed)) if yolo_worker.completed else 0.0
                    progress_callback({
                        'stage': 'chapter_detection',
                        'phase': 'Chapter detection',
                        'task': f'Frame {frame_idx}/{total_frames}',
                        'percentage': pct,
                        'current': frame_idx,
                        'total': total_frames,
                        'avg_fps': avg_fps,
                        'eta_seconds': eta_s,
                        'elapsed_seconds': elapsed,
                        'timing': {'yolo_det_ms': yolo_ms},
                    })

        except Exception as e:
            self.logger.error(f"Detection error: {e}", exc_info=True)
        finally:
            yolo_worker.stop()
            if yolo_worker.dropped:
                self.logger.info(
                    f"YOLO worker dropped {yolo_worker.dropped} submissions "
                    f"(queue full). Completed {yolo_worker.completed}.")

        elapsed = time.time() - t_start
        self.logger.info(f"Sparse detection: {yolo_count} YOLO frames in {elapsed:.1f}s "
                        f"({yolo_count / max(0.001, elapsed):.1f} det/s)")

        chapters = build_chapters(frame_positions, fps, total_frames, frame_skip,
                                  penis_frames=penis_frames,
                                  frame_contact_info=frame_contact_info)
        stats = {
            'fps': fps,
            'total_frames': total_frames,
            'duration_s': total_frames / fps,
            'yolo_frames': yolo_count,
        }
        return chapters, stats
