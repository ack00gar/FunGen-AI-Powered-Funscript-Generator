#!/usr/bin/env python3
"""
Hybrid Chapter-Aware Tracker — Offline pipeline with per-chapter optimization.

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

# Try to import YOLO for sparse detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO = None
    YOLO_AVAILABLE = False

# Constants
SPARSE_FPS = 2  # Frames per second for chapter detection
MIN_CHAPTER_SECONDS = 10.0  # Minimum chapter duration (merge shorter ones into neighbors)
CHAPTER_MERGE_GAP_SECONDS = 15.0  # Merge same-position chapters separated by less than this
BOUNDARY_CROSSFADE_FRAMES = 15  # Frames to crossfade at chapter boundaries
CONTACT_IOU_THRESHOLD = 0.05  # Minimum IoU for contact detection
DEFAULT_CONFIDENCE = 0.25  # Low confidence for sparse pass (false positives are OK)
MAX_PARALLEL_CHAPTERS = 2  # Max chapters to process in parallel

# Position type mapping from contact class
CONTACT_TO_POSITION = {
    'pussy': 'Cowgirl / Missionary',
    'butt': 'Rev. Cowgirl / Doggy',
    'anus': 'Rev. Cowgirl / Doggy',
    'face': 'Blowjob',
    'hand': 'Handjob',
    'breast': 'Boobjob',
    'foot': 'Footjob',
}

# Priority order for contact resolution (higher = preferred)
CONTACT_PRIORITY = {
    'pussy': 10, 'butt': 9, 'anus': 8, 'face': 7,
    'hand': 5, 'breast': 4, 'foot': 3, 'navel': 1,
}

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

# Position types that need dense processing
DENSE_PROCESSING_POSITIONS = {
    'Cowgirl / Missionary', 'Rev. Cowgirl / Doggy', 'Blowjob',
    'Handjob', 'Boobjob', 'Footjob',
}


class HybridChapterTracker(BaseOfflineTracker):
    """
    Hybrid chapter-aware offline tracker.

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

    @property
    def metadata(self) -> TrackerMetadata:
        return TrackerMetadata(
            name="OFFLINE_HYBRID_CHAPTER",
            display_name="Hybrid Chapter-Aware (ROI Flow)",
            description="Chapter detection at 2fps, then YOLO + ROI optical flow per chapter",
            category="offline",
            version="2.0.0",
            author="FunGen",
            tags=["offline", "hybrid", "chapter-aware", "optimized", "batch"],
            requires_roi=False,
            supports_dual_axis=True,
            primary_axis="stroke",
            secondary_axis="roll",
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
            self.logger.info("Hybrid Chapter Tracker initialized")
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

            processing_time = time.time() - start_time
            self.logger.info(f"Hybrid processing complete in {processing_time:.1f}s")

            if progress_callback:
                progress_callback({'stage': 'complete', 'task': 'Done', 'percentage': 100})

            self.processing_active = False

            return OfflineProcessingResult(
                success=True,
                output_data={
                    'funscript': funscript,
                    'chapters': chapters,
                    'chapter_results': {i: r.get('metrics', {}) for i, r in chapter_results.items()},
                },
                performance_metrics={
                    'processing_time_seconds': processing_time,
                    'chapters_detected': len(chapters),
                    'chapters_processed': sum(1 for ch in chapters if ch.get('dense')),
                    'chapters_skipped': sum(1 for ch in chapters if not ch.get('dense')),
                }
            )

        except Exception as e:
            self.logger.error(f"Hybrid tracker error: {e}", exc_info=True)
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

        model = YOLO(yolo_model_path)
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

        # Choose frame source: reuse preprocessed file or VP pipeline
        if reuse_preprocessed and reuse_cap is not None:
            frame_source = self._frames_from_capture(reuse_cap, total_frames)
        else:
            frame_source = self._frames_from_vp(vp, total_frames, encoder)

        try:
            for frame_idx, frame in frame_source:

                if self.stop_event and self.stop_event.is_set():
                    break

                frames_written += 1

                # Run YOLO only every Nth frame
                if frame_idx % frame_skip != 0:
                    continue

                try:
                    results = model(frame, device=config_constants.DEVICE, verbose=False,
                                  conf=DEFAULT_CONFIDENCE, imgsz=self.yolo_input_size)
                except Exception as e:
                    self.logger.debug(f"YOLO error on frame {frame_idx}: {e}")
                    continue

                # Parse detections
                penis_box = None
                other_boxes = []
                frame_dets = []

                if results and len(results) > 0:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        cls_name = model.names.get(cls_id, f"class_{cls_id}")
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()

                        det = {'bbox': [x1, y1, x2, y2], 'class_name': cls_name,
                               'confidence': conf}
                        frame_dets.append(det)

                        if cls_name == penis_class_name:
                            if penis_box is None or conf > penis_box['conf']:
                                penis_box = {'box': (x1, y1, x2, y2), 'conf': conf}
                        elif cls_name in CONTACT_TO_POSITION:
                            other_boxes.append({
                                'box': (x1, y1, x2, y2),
                                'class': cls_name,
                                'conf': conf
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
                    position = self._classify_frame_position(penis_box, other_boxes)
                elif len(other_boxes) >= 2:
                    # No penis visible but multiple contact boxes → likely occluded
                    position = self._classify_no_penis(other_boxes, frame_h)
                else:
                    position = 'Not Relevant'

                frame_positions[frame_idx] = position
                yolo_processed += 1

                if progress_callback and yolo_processed % 50 == 0:
                    pct = min(30, int(30 * frame_idx / max(1, total_frames)))
                    progress_callback({'stage': 'pass1', 'task': f'Decode + sparse YOLO ({yolo_processed} det, {frames_written} frames)',
                                      'percentage': pct})

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
        chapters = self._build_chapters(frame_positions, fps, total_frames, frame_skip,
                                        penis_frames=penis_frames,
                                        frame_contact_info=frame_contact_info)

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

    def _classify_no_penis(self, other_boxes: List[Dict], frame_h: int) -> str:
        """Classify position when penis is not visible but contact boxes are.

        Uses a scoring system that combines:
        - Contact class priority (genital > peripheral)
        - Box area (larger = closer to camera = more relevant)
        - Y-position penalty for face boxes in lower frame (likely not the action zone)
        """
        # Genital/body classes are strong indicators even without penis
        GENITAL_CLASSES = {'pussy', 'butt', 'anus'}

        # Check for any genital boxes first — these are reliable
        genital_boxes = [b for b in other_boxes if b['class'] in GENITAL_CLASSES]
        if genital_boxes:
            best = max(genital_boxes, key=lambda b: (
                CONTACT_PRIORITY.get(b['class'], 0),
                (b['box'][2] - b['box'][0]) * (b['box'][3] - b['box'][1])  # area
            ))
            return CONTACT_TO_POSITION.get(best['class'], 'Not Relevant')

        # No genital boxes — use all contacts but with spatial scoring
        # In VR POV, face in upper half → likely blowjob position
        # Face in lower half → might be misleading (body below camera)
        scored = []
        for b in other_boxes:
            cls = b['class']
            if cls not in CONTACT_TO_POSITION:
                continue
            box = b['box']
            area = (box[2] - box[0]) * (box[3] - box[1])
            y_center = (box[1] + box[3]) / 2.0
            priority = CONTACT_PRIORITY.get(cls, 0)
            # Score: priority * area_factor
            area_factor = area / (frame_h * frame_h) * 100  # normalize
            score = priority * (1.0 + area_factor)
            scored.append((score, cls))

        if scored:
            best_cls = max(scored, key=lambda x: x[0])[1]
            return CONTACT_TO_POSITION.get(best_cls, 'Not Relevant')

        return 'Not Relevant'

    def _classify_frame_position(self, penis_box: Dict, other_boxes: List[Dict]) -> str:
        """Classify a frame's position based on penis-body contact."""
        px1, py1, px2, py2 = penis_box['box']

        # Find contacts with sufficient IoU
        contacts = []
        for other in other_boxes:
            ox1, oy1, ox2, oy2 = other['box']

            # Calculate IoU
            ix1 = max(px1, ox1)
            iy1 = max(py1, oy1)
            ix2 = min(px2, ox2)
            iy2 = min(py2, oy2)

            if ix1 < ix2 and iy1 < iy2:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                area_p = (px2 - px1) * (py2 - py1)
                area_o = (ox2 - ox1) * (oy2 - oy1)
                union = area_p + area_o - intersection
                iou = intersection / union if union > 0 else 0

                if iou > CONTACT_IOU_THRESHOLD:
                    contacts.append(other)
            else:
                # Also check proximity (within 1.5x penis width)
                penis_w = px2 - px1
                penis_h = py2 - py1
                diag = np.sqrt(penis_w**2 + penis_h**2)

                # Center distance
                pcx, pcy = (px1+px2)/2, (py1+py2)/2
                ocx, ocy = (ox1+ox2)/2, (oy1+oy2)/2
                dist = np.sqrt((pcx-ocx)**2 + (pcy-ocy)**2)

                o_diag = np.sqrt((ox2-ox1)**2 + (oy2-oy1)**2)
                if dist < (diag + o_diag) * 0.5:
                    contacts.append(other)

        if not contacts:
            return 'Close up'  # Penis visible but no contact

        # Pick highest priority contact
        best_contact = max(contacts, key=lambda c: CONTACT_PRIORITY.get(c['class'], 0))
        return CONTACT_TO_POSITION.get(best_contact['class'], 'Not Relevant')

    def _classify_segment_spatial(self, start_frame: int, end_frame: int,
                                   frame_contact_info: Dict) -> Optional[str]:
        """Classify a segment's position using spatial statistics of contact boxes.

        Uses VR POV layout knowledge:
        - Partner is always in the center third of the panel horizontally
        - Rev. Cowgirl / Doggy: butt dominant in center, face (if visible) above butt,
          aligned horizontally. Butt is large, prominent.
        - Missionary: pussy in bottom of frame, breast above, face above breast.
          Vertical stack of body parts.
        - Cowgirl: similar to missionary (pussy bottom, face above) but different
          vertical positions (woman is more upright).
        - Blowjob: face dominant, in lower-center. Mouth/face is the largest contact.

        Approach: collect per-class spatial stats across the segment, then score
        position types using spatial layout rules.
        """
        from collections import Counter

        # Collect all contact detections in range
        class_counts = Counter()
        class_y_sum = {}     # class → sum of norm_cy
        class_x_sum = {}     # class → sum of norm_cx
        class_area_sum = {}  # class → sum of norm_area
        total_contacts = 0

        for fid, contacts in frame_contact_info.items():
            if fid < start_frame or fid > end_frame:
                continue
            for c in contacts:
                cls = c['class']
                if cls not in CONTACT_TO_POSITION:
                    continue
                bx1, by1, bx2, by2 = c['box']
                frame_w = 640  # preprocessed is 640x640
                norm_cx = ((bx1 + bx2) / 2.0) / frame_w
                class_counts[cls] += 1
                class_y_sum[cls] = class_y_sum.get(cls, 0.0) + c['norm_cy']
                class_x_sum[cls] = class_x_sum.get(cls, 0.0) + norm_cx
                class_area_sum[cls] = class_area_sum.get(cls, 0.0) + c['norm_area']
                total_contacts += 1

        if total_contacts < 3:
            return None

        # Compute per-class stats
        class_stats = {}
        for cls, count in class_counts.items():
            class_stats[cls] = {
                'count': count,
                'freq': count / total_contacts,
                'mean_y': class_y_sum[cls] / count,      # 0=top, 1=bottom
                'mean_x': class_x_sum[cls] / count,      # 0=left, 1=right
                'mean_area': class_area_sum[cls] / count,
            }

        # Check for spatial layout patterns
        has_pussy = 'pussy' in class_stats
        has_butt = 'butt' in class_stats or 'anus' in class_stats
        has_face = 'face' in class_stats
        has_breast = 'breast' in class_stats
        has_hand = 'hand' in class_stats

        # Genital position is the strongest signal
        if has_pussy:
            pussy_stats = class_stats['pussy']
            # Pussy in lower half → Cowgirl / Missionary
            # (Both have pussy low; we can't reliably distinguish without penis)
            if pussy_stats['mean_y'] > 0.5:
                return 'Cowgirl / Missionary'
            else:
                return 'Cowgirl / Missionary'

        if has_butt:
            butt_cls = 'butt' if 'butt' in class_stats else 'anus'
            butt_stats = class_stats[butt_cls]
            # Butt visible → Rev. Cowgirl / Doggy
            # Butt is large and center-to-upper in frame
            if butt_stats['freq'] > 0.2 or butt_stats['mean_area'] > 0.02:
                return 'Rev. Cowgirl / Doggy'

        # No genital/butt → use face + breast layout
        if has_face and has_breast:
            face_stats = class_stats['face']
            breast_stats = class_stats['breast']
            # Face above breast (face.y < breast.y) → vertical body stack
            # This pattern occurs in missionary/cowgirl (body facing camera)
            if face_stats['mean_y'] < breast_stats['mean_y']:
                # Breast in lower half → more likely missionary
                if breast_stats['mean_y'] > 0.5:
                    return 'Cowgirl / Missionary'
                else:
                    return 'Cowgirl / Missionary'

        if has_face and not has_breast and not has_pussy and not has_butt:
            face_stats = class_stats['face']
            # Face alone, large area, in lower-center → blowjob
            if face_stats['mean_area'] > 0.01 and face_stats['mean_y'] > 0.4:
                return 'Blowjob'

        if has_hand and not has_face:
            return 'Handjob'

        # Fallback: use highest-frequency contact class
        if class_counts:
            best_cls = class_counts.most_common(1)[0][0]
            return CONTACT_TO_POSITION.get(best_cls)

        return None

    def _build_chapters(self, frame_positions: Dict[int, str], fps: float,
                        total_frames: int, frame_skip: int,
                        penis_frames: Optional[set] = None,
                        frame_contact_info: Optional[Dict] = None) -> List[Dict]:
        """Build chapters from sparse frame-level position votes.

        Uses a multi-step approach:
        1. Smooth per-frame positions with a centered voting window
        2. Build raw chapters from smoothed sequence
        3. Merge short chapters into neighbors
        4. Merge same-position chapters within a gap threshold
        5. Absorb short NR gaps between active chapters
        6. Trim chapter boundaries to penis-visible frames (avoids intro/outro noise)
        7. Reclassify chapters using spatial contact features
        """
        if not frame_positions:
            return [{'start_frame': 0, 'end_frame': total_frames - 1,
                     'position': 'Unknown', 'dense': True}]

        # Parameters
        VOTE_WINDOW_S = 20.0       # Centered smoothing window
        MIN_ACTIVE_PCT = 0.20      # 20% non-NR frames → classify as active
        MIN_CHAPTER_S = 20.0       # Minimum chapter duration
        MERGE_GAP_S = 60.0         # Merge same-type within 60s
        NR_ABSORB_MAX_S = 120.0    # Absorb NR gaps < 120s between active chapters

        from collections import Counter

        # Sort frames
        sorted_frames = sorted(frame_positions.items())
        frame_ids = np.array([f[0] for f in sorted_frames])
        frame_times = frame_ids / fps
        positions = [f[1] for f in sorted_frames]
        vote_window_half = VOTE_WINDOW_S / 2.0

        # Step 1: Smooth positions with centered majority vote
        smoothed = []
        for i, (fid, pos) in enumerate(sorted_frames):
            t = fid / fps
            mask = (frame_times >= t - vote_window_half) & (frame_times <= t + vote_window_half)
            window_pos = [positions[j] for j in range(len(positions)) if mask[j]]
            counts = Counter(window_pos)
            total = len(window_pos)
            nr_count = counts.get('Not Relevant', 0) + counts.get('Close up', 0)
            active_pct = 1 - nr_count / total

            if active_pct >= MIN_ACTIVE_PCT:
                non_nr = {k: v for k, v in counts.items() if k not in ('Not Relevant', 'Close up')}
                smoothed.append(max(non_nr, key=non_nr.get) if non_nr else 'Not Relevant')
            else:
                smoothed.append('Not Relevant')

        # Step 2: Build raw chapters from smoothed positions
        raw_chapters = []
        cur_pos = smoothed[0]
        cur_start = sorted_frames[0][0]
        for i in range(1, len(sorted_frames)):
            if smoothed[i] != cur_pos:
                raw_chapters.append({
                    'start_frame': cur_start,
                    'end_frame': sorted_frames[i-1][0],
                    'position': cur_pos,
                })
                cur_pos = smoothed[i]
                cur_start = sorted_frames[i][0]
        raw_chapters.append({
            'start_frame': cur_start,
            'end_frame': total_frames - 1,
            'position': cur_pos,
        })

        # Step 3: Merge short chapters
        min_frames = int(MIN_CHAPTER_S * fps)
        merged = self._merge_short_chapters(raw_chapters, min_frames)

        # Step 4: Merge same-position close together
        merge_gap_frames = int(MERGE_GAP_S * fps)
        merged = self._merge_adjacent_same_position(merged, merge_gap_frames)

        # Step 5: Absorb short NR gaps between active chapters
        nr_absorb_frames = int(NR_ABSORB_MAX_S * fps)
        changed = True
        while changed:
            changed = False
            new_merged = []
            i = 0
            while i < len(merged):
                ch = merged[i]
                dur = ch['end_frame'] - ch['start_frame']
                is_nr = ch['position'] in ('Not Relevant', 'Close up')
                if (is_nr and dur < nr_absorb_frames and
                    i > 0 and i < len(merged) - 1 and
                    new_merged and new_merged[-1]['position'] not in ('Not Relevant', 'Close up') and
                    merged[i+1]['position'] not in ('Not Relevant', 'Close up')):
                    # Absorb into the longer neighbor
                    prev_dur = new_merged[-1]['end_frame'] - new_merged[-1]['start_frame']
                    next_dur = merged[i+1]['end_frame'] - merged[i+1]['start_frame']
                    if prev_dur >= next_dur:
                        new_merged[-1]['end_frame'] = ch['end_frame']
                    else:
                        merged[i+1]['start_frame'] = ch['start_frame']
                    changed = True
                    i += 1
                    continue
                new_merged.append(dict(ch))
                i += 1
            merged = new_merged

        # Final merge of same-type adjacent
        result = [merged[0]] if merged else []
        for ch in merged[1:]:
            if result[-1]['position'] == ch['position']:
                result[-1]['end_frame'] = ch['end_frame']
            else:
                result.append(dict(ch))

        # Step 6: Trim active chapter boundaries to confirmed-active frames
        # A "confirmed active" frame has penis + contact (not just penis = close-up)
        # This avoids including intro/outro where body is visible but no action
        confirmed_active = set()
        for fid, pos in frame_positions.items():
            if pos not in ('Not Relevant', 'Close up'):
                # Only count frames where penis was detected WITH contact
                if fid in (penis_frames or set()):
                    confirmed_active.add(fid)

        if confirmed_active:
            TRIM_TOLERANCE_S = 5.0
            trim_tolerance_frames = int(TRIM_TOLERANCE_S * fps)
            sorted_active = sorted(confirmed_active)

            for ch in result:
                if ch['position'] in ('Not Relevant', 'Close up'):
                    continue
                # Find confirmed-active frames within this chapter
                ch_active = [f for f in sorted_active
                             if ch['start_frame'] <= f <= ch['end_frame']]
                if not ch_active:
                    # No confirmed-active frames — contact-only chapter (e.g. missionary)
                    continue
                # Only trim if confirmed-active frames cover >30% of sparse frames
                # Otherwise this is mostly a contact-only chapter → don't trim
                ch_sparse = [f for f in frame_positions
                             if ch['start_frame'] <= f <= ch['end_frame']]
                if len(ch_active) / max(1, len(ch_sparse)) < 0.30:
                    continue
                # Trim start/end to confirmed action zone ± tolerance
                first_active = ch_active[0]
                last_active = ch_active[-1]
                new_start = max(ch['start_frame'], first_active - trim_tolerance_frames)
                new_end = min(ch['end_frame'], last_active + trim_tolerance_frames)
                if new_start < new_end:
                    if new_start > ch['start_frame'] or new_end < ch['end_frame']:
                        self.logger.info(f"  Trim [{ch['start_frame']}-{ch['end_frame']}] → "
                                        f"[{new_start}-{new_end}]")
                    ch['start_frame'] = new_start
                    ch['end_frame'] = new_end

        # Step 7: Spatial reclassification for chapters with no/few penis frames
        # Uses contact box positions across the segment to refine position type
        if frame_contact_info:
            for ch in result:
                if ch['position'] in ('Not Relevant', 'Close up'):
                    continue
                # Count penis frames in this chapter
                ch_penis_count = sum(1 for f in (penis_frames or set())
                                     if ch['start_frame'] <= f <= ch['end_frame'])
                # Only reclassify if very few penis frames (contact-only detection)
                ch_sparse_count = sum(1 for f in frame_positions
                                      if ch['start_frame'] <= f <= ch['end_frame'])
                if ch_sparse_count > 0 and ch_penis_count / ch_sparse_count < 0.15:
                    new_pos = self._classify_segment_spatial(
                        ch['start_frame'], ch['end_frame'], frame_contact_info)
                    if new_pos and new_pos != ch['position']:
                        self.logger.info(f"  Spatial reclassify [{ch['start_frame']}-{ch['end_frame']}]: "
                                        f"{ch['position']} → {new_pos}")
                        ch['position'] = new_pos

        # Mark dense processing
        for ch in result:
            ch['dense'] = ch['position'] in DENSE_PROCESSING_POSITIONS
            duration_s = (ch['end_frame'] - ch['start_frame']) / fps
            ch['duration_s'] = round(duration_s, 1)

        return result

    def _merge_short_chapters(self, chapters: List[Dict], min_frames: int) -> List[Dict]:
        """Merge chapters shorter than min_frames into their neighbors."""
        if len(chapters) <= 1:
            return chapters

        result = [chapters[0]]
        for ch in chapters[1:]:
            duration = ch['end_frame'] - ch['start_frame']
            if duration < min_frames and result:
                # Merge into previous chapter
                result[-1]['end_frame'] = ch['end_frame']
            else:
                result.append(ch)

        return result

    def _merge_adjacent_same_position(self, chapters: List[Dict], gap_frames: int) -> List[Dict]:
        """Merge chapters with same position that are separated by a short gap."""
        if len(chapters) <= 1:
            return chapters

        result = [chapters[0]]
        for ch in chapters[1:]:
            prev = result[-1]
            gap = ch['start_frame'] - prev['end_frame']
            if prev['position'] == ch['position'] and gap < gap_frames:
                prev['end_frame'] = ch['end_frame']
            else:
                result.append(ch)

        return result

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
                def _sub(frame_frac):
                    ch_base = base + int(rng * ch_idx / max(1, ch_total))
                    ch_range = rng / max(1, ch_total)
                    pct = ch_base + int(ch_range * frame_frac)
                    cb({'stage': 'pass2',
                        'task': f"Chapter {ch_idx+1}/{ch_total}: {ch['position']} ({int(frame_frac*100)}%)",
                        'percentage': pct})
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

        for i in range(chapter_frames):
            ret, frame = cap.read()
            if not ret:
                break

            if self.stop_event and self.stop_event.is_set():
                cap.release()
                return None

            if frame_progress_callback and i % 100 == 0 and chapter_frames > 0:
                frame_progress_callback(float(i) / chapter_frames)

            abs_frame = start_frame + i
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]

            # Get ROI — dense YOLO on every frame, or sparse fallback
            roi_box = None
            if use_dense_yolo:
                roi_box = self._detect_roi_yolo(frame, yolo_model, h, w)
                if roi_box is not None:
                    yolo_det_count += 1
                    last_roi = roi_box
                elif last_roi is not None:
                    roi_box = last_roi  # Hold last known ROI
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
                    flow = dis.calc(
                        np.ascontiguousarray(prev_patch),
                        np.ascontiguousarray(curr_patch),
                        None
                    )

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
                          h: int, w: int) -> Optional[Tuple[int, int, int, int]]:
        """Run YOLO on a single frame and return padded ROI from penis + nearest contact."""
        try:
            results = yolo_model(frame, device=config_constants.DEVICE, verbose=False,
                                 conf=DEFAULT_CONFIDENCE, imgsz=self.yolo_input_size)
        except Exception:
            return None

        if not results or len(results) == 0:
            return None

        penis_box = None
        best_conf = 0.0
        contact_boxes = []

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = yolo_model.names.get(cls_id, f"class_{cls_id}")
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            if cls_name == 'penis' and conf > best_conf:
                penis_box = (x1, y1, x2, y2)
                best_conf = conf
            elif cls_name in CONTACT_TO_POSITION:
                contact_boxes.append((x1, y1, x2, y2))

        if penis_box is None:
            return None

        return self._compute_padded_roi(penis_box, contact_boxes, h, w)

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
