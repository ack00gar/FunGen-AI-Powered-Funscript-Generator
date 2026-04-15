"""
VR Hybrid Chapter-Aware Tracker V2 — Single-pass architecture.

Same output as V1 but ~1.7x faster:
- Single FFmpeg pass: v360 dewarp → sparse YOLO + continuous DIS flow
- No preprocessed video encoding/decoding, no disk I/O
- Sparse ROI interpolation (reuses last known ROI between YOLO frames)
- ROI shift detection to prevent fake flow spikes

Reuses V1's post-processing pipeline (SavGol detrend, amplitude normalization,
peak/valley keyframe extraction) and chapter detection (build_chapters).
"""

import copy
import os

from common.frame_utils import frame_to_ms
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
from multiprocessing import Event

import cv2
import numpy as np

try:
    from scipy.signal import savgol_filter, find_peaks
    from scipy.ndimage import median_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import msgpack
except ImportError:
    msgpack = None

try:
    from tracker.tracker_modules.core.base_offline_tracker import (
        BaseOfflineTracker, OfflineProcessingStage, OfflineProcessingResult,
    )
    from tracker.tracker_modules.core.base_tracker import (
        TrackerMetadata, StageDefinition,
    )
    from tracker.tracker_modules.helpers.chapter_detection import (
        build_chapters, classify_frame_position, classify_no_penis,
        parse_detections, build_contact_info, CONTACT_TO_POSITION, ACTIVE_POSITIONS,
    )
    from tracker.tracker_modules.helpers.yolo_detection_helper import (
        load_model as _yolo_load_model,
        run_detection as _yolo_run_detection,
        detection_to_dict as _yolo_det_to_dict,
    )
    MODULES_AVAILABLE = True
    YOLO_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False
    YOLO_AVAILABLE = False

from config import constants as config_constants

# ── Constants ─────────────────────────────────────────────────────────────────

SPARSE_FPS = 2            # Chapter classification YOLO rate
ROI_YOLO_INTERVAL = 5     # ROI tracking: run YOLO every Nth frame for tight ROI
DEFAULT_CONFIDENCE = 0.25
ROI_PADDING_FACTOR = 0.3
MIN_ROI_SIZE = 16
ROI_SHIFT_THRESHOLD = 12  # pixels — reset prev_gray if ROI shifts more than this

# Post-processing (same as V1)
FLOW_MEDIAN_WINDOW = 5
SAVGOL_WINDOW = 7
SAVGOL_POLYORDER = 2
MIN_PEAK_PROMINENCE = 4.0
MIN_PEAK_DISTANCE_S = 0.25

CHAPTER_TYPE_CONFIG = {
    'Cowgirl / Missionary': {'target_range': 70},
    'Rev. Cowgirl / Doggy': {'target_range': 70},
    'Blowjob': {'target_range': 55},
    'Handjob': {'target_range': 55},
    'Boobjob': {'target_range': 50},
    'Footjob': {'target_range': 50},
}


class VRHybridChapterTrackerV2(BaseOfflineTracker):
    """Single-pass VR Hybrid tracker. No preprocessed video, no disk I/O."""

    def __init__(self):
        super().__init__()
        self.app = None
        self.yolo_input_size = 640
        self.video_type = "auto"
        self.vr_input_format = "he"
        self.vr_fov = 190
        self.vr_pitch = 0
        self.sparse_fps = SPARSE_FPS
        self._overlay_frames = []

    @property
    def metadata(self) -> TrackerMetadata:
        return TrackerMetadata(
            name="OFFLINE_VR_HYBRID_CHAPTER",
            display_name="VR Hybrid Chapter-Aware (ROI Flow)",
            description="v2.0.0 — Single-pass: sparse YOLO + continuous DIS flow, ~4x faster",
            category="offline",
            version="2.0.0",
            author="FunGen",
            tags=["offline", "vr", "2d", "chapter-aware", "hybrid", "single-pass"],
            requires_roi=False,
            supports_dual_axis=True,
            primary_axis="stroke",
            secondary_axis="roll",
            stages=[
                StageDefinition(
                    stage_number=2,
                    name="Single-Pass Analysis",
                    description="Sparse YOLO + continuous DIS flow in one FFmpeg pass",
                    produces_funscript=True,
                    requires_previous=False,
                    output_type="mixed"
                ),
            ],
            properties={
                "produces_funscript_in_stage2": True,
                "supports_batch": True,
                "supports_range": False,
                "is_hybrid_tracker": True,
                "num_stages": 1,
            }
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
            if not MODULES_AVAILABLE or not YOLO_AVAILABLE:
                self.logger.error("Required modules not available")
                return False
            self._initialized = True
            self.logger.info("VR Hybrid Chapter Tracker V2 initialized")
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
        # V2 single-pass: ~1x realtime at 30fps
        return max(10.0, duration_s * 1.0)

    def process_stage(self, stage, video_path, input_data=None, input_files=None,
                      output_directory=None, progress_callback=None,
                      frame_range=None, resume_data=None, **kwargs):
        if not self._initialized:
            return OfflineProcessingResult(success=False, error_message="Not initialized")

        try:
            self.processing_active = True
            self.stop_event = self.stop_event or Event()
            self._overlay_frames = []

            if not output_directory:
                output_directory = os.path.dirname(video_path)
            os.makedirs(output_directory, exist_ok=True)

            self._load_settings()

            video_basename = os.path.splitext(os.path.basename(video_path))[0]
            start_time = time.time()

            # === SINGLE PASS ===
            self.logger.info("=== Single-Pass: Sparse YOLO + Continuous DIS Flow ===")
            if progress_callback:
                progress_callback({'stage': 'pass1', 'task': 'Single-pass analysis', 'percentage': 0})

            result = self._single_pass(video_path, output_directory, progress_callback)

            if self.stop_event.is_set():
                return OfflineProcessingResult(success=False, error_message="Processing stopped")

            if not result:
                return OfflineProcessingResult(success=False, error_message="Single-pass processing failed")

            chapters = result['chapters']
            raw_flow = result['raw_flow']  # {frame_idx: (time_ms, dy, dx)}
            fps = result['fps']
            sparse_detections = result['sparse_detections']
            frame_boxes = result['frame_boxes']  # frame_idx → {penis, contacts}

            self.logger.info(f"Detected {len(chapters)} chapters:")
            for ch in chapters:
                ch_dur = round((ch['end_frame'] - ch['start_frame']) / fps, 1)
                self.logger.info(f"  [{ch['start_frame']}-{ch['end_frame']}] {ch['position']} ({ch_dur}s)")

            # === Post-process per chapter ===
            self.logger.info("=== Post-Processing Per Chapter ===")
            if progress_callback:
                progress_callback({'stage': 'pass2', 'task': 'Post-processing chapters', 'percentage': 80})

            chapter_results = {}
            for i, ch in enumerate(chapters):
                if not ch.get('dense'):
                    continue

                # Extract flow data for this chapter
                chapter_flow = [
                    (idx, raw_flow[idx][0], raw_flow[idx][1])  # (local_idx, time_ms, dy)
                    for idx in sorted(raw_flow.keys())
                    if ch['start_frame'] <= idx <= ch['end_frame']
                ]

                # Re-index to chapter-local
                reindexed = [(j, t, dy) for j, (_, t, dy) in enumerate(chapter_flow)]

                if len(reindexed) < SAVGOL_WINDOW:
                    self.logger.warning(f"  Chapter {i}: too few flow samples ({len(reindexed)})")
                    continue

                # Detect motion inversion from flow data
                invert = self._detect_inversion(chapter_flow, raw_flow, ch)

                primary_actions, secondary_actions = self._postprocess_chapter_signal(
                    reindexed, ch['position'], fps, invert
                )

                if primary_actions:
                    chapter_results[i] = {
                        'success': True,
                        'primary_actions': primary_actions,
                        'secondary_actions': secondary_actions,
                    }
                    self.logger.info(f"  Chapter {i}: {len(primary_actions)} keyframes")

            # === Merge ===
            self.logger.info("=== Merge Results ===")
            if progress_callback:
                progress_callback({'stage': 'pass3', 'task': 'Merging results', 'percentage': 95})

            funscript = self._merge_chapter_results(chapters, chapter_results, fps)

            # Save overlay data
            overlay_path = None
            if self._overlay_frames:
                overlay_path = os.path.join(output_directory, f'{video_basename}_stage2_overlay.msgpack')
                try:
                    with open(overlay_path, 'wb') as f:
                        f.write(msgpack.packb(self._overlay_frames, use_bin_type=True))
                    self.logger.info(f"Saved overlay data: {len(self._overlay_frames)} frames")
                except Exception as e:
                    self.logger.warning(f"Failed to save overlay: {e}")

            elapsed = time.time() - start_time
            self.logger.info(f"V2 single-pass complete in {elapsed:.1f}s")

            if progress_callback:
                progress_callback({'stage': 'complete', 'task': 'Done', 'percentage': 100})

            self.processing_active = False

            return OfflineProcessingResult(
                success=True,
                output_data={
                    'funscript': funscript,
                    'chapters': chapters,
                    'overlay_path': overlay_path,
                },
                output_files={'overlay': overlay_path} if overlay_path else {},
                performance_metrics={
                    'processing_time_seconds': round(elapsed, 1),
                    'chapters_detected': len(chapters),
                    'flow_samples': len(raw_flow),
                    'fps': fps,
                },
            )

        except Exception as e:
            self.logger.error(f"V2 error: {e}", exc_info=True)
            self.processing_active = False
            return OfflineProcessingResult(success=False, error_message=str(e))

    # ─── Settings ──────────────────────────────────────────────────────────────

    def _load_settings(self):
        if self.app:
            self.yolo_input_size = getattr(self.app, 'yolo_input_size', 640)
            if hasattr(self.app, 'processor') and self.app.processor:
                self.video_type = getattr(self.app.processor, 'video_type_setting', 'auto')
                self.vr_input_format = getattr(self.app.processor, 'vr_input_format', 'he')
                self.vr_fov = getattr(self.app.processor, 'vr_fov', 190)
                self.vr_pitch = getattr(self.app.processor, 'vr_pitch', 0)

    # ─── Single Pass ──────────────────────────────────────────────────────────

    def _single_pass(self, video_path: str, output_dir: str,
                     progress_callback: Optional[Callable]) -> Optional[Dict]:
        """
        Stream all frames via VideoProcessor (FFmpeg v360 dewarp).
        Run sparse YOLO every Nth frame for chapter detection + ROI.
        Run DIS optical flow on every frame using interpolated ROI.
        No encoding, no preprocessed video, no disk I/O.
        """
        from video.video_processor import VideoProcessor

        yolo_model_path = getattr(self.app, 'yolo_det_model_path', None)
        if not yolo_model_path or not os.path.exists(yolo_model_path):
            self.logger.error(f"YOLO model not found: {yolo_model_path}")
            return None

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
            return None

        fps = vp.fps or 30.0
        total_frames = vp.total_frames
        chapter_skip = max(1, int(fps / self.sparse_fps))  # ~29 @ 60fps: for chapter classification
        roi_skip = ROI_YOLO_INTERVAL                         # 5: for ROI tracking
        frame_ms = 1000.0 / fps

        vp.vr_unwarp_method_override = 'v360'
        vp._update_video_parameters()

        # stream_frames_for_segment guarantees 640x640 output by temporarily
        # toggling HD display off when active; this tracker just consumes.
        self.logger.info(f"Single-pass: {total_frames} frames @ {fps:.1f}fps, "
                         f"chapter YOLO every {chapter_skip}th, ROI YOLO every {roi_skip}th, DIS every frame")

        # State
        prev_gray = None
        current_roi = None
        frame_positions = {}
        penis_frames = set()
        frame_contact_info = {}
        frame_boxes = {}  # frame_idx → {penis: tuple, contacts: [tuple]}
        sparse_detections = {}
        raw_flow = {}  # frame_idx → (time_ms, dy, dx)
        upper_motion_sum = 0.0
        lower_motion_sum = 0.0
        yolo_count = 0
        flow_count = 0
        t_start = time.time()

        # Per-stage timing (printed once at end) to spot future regressions.
        t_decode = t_cvt = t_yolo = t_dis = 0.0

        from tracker.tracker_modules.helpers.async_yolo_worker import AsyncYoloWorker
        from tracker.tracker_modules.helpers.async_flow_worker import AsyncDisFlowWorker

        def _apply_yolo_result(f_idx, fh, fw, det_objs, is_chap):
            nonlocal current_roi, prev_gray
            penis_box, other_boxes = parse_detections(det_objs)
            if penis_box:
                contact_bboxes = [ob['box'] for ob in other_boxes]
                new_roi = self._compute_padded_roi(penis_box['box'], contact_bboxes, fh, fw)
                if self._roi_shifted(current_roi, new_roi):
                    prev_gray = None
                current_roi = new_roi
            if is_chap:
                frame_dets = [_yolo_det_to_dict(d) for d in det_objs]
                sparse_detections[f_idx] = frame_dets
                if other_boxes:
                    frame_contact_info[f_idx] = build_contact_info(
                        other_boxes, self.yolo_input_size)
                if penis_box:
                    penis_frames.add(f_idx)
                    position = classify_frame_position(penis_box, other_boxes)
                    frame_boxes[f_idx] = {
                        'penis': penis_box['box'],
                        'contacts': [ob['box'] for ob in other_boxes],
                    }
                elif len(other_boxes) >= 1:
                    position = classify_no_penis(other_boxes, self.yolo_input_size)
                else:
                    position = 'Not Relevant'
                frame_positions[f_idx] = position
                if frame_dets:
                    self._overlay_frames.append({
                        'frame_id': f_idx,
                        'yolo_boxes': [
                            {'bbox': d['bbox'], 'class_name': d['class_name'],
                             'confidence': d.get('confidence', 0.0),
                             'track_id': None, 'status': None}
                            for d in frame_dets
                        ],
                        'poses': [],
                        'dominant_pose_id': None,
                        'active_interaction_track_id': None,
                        'is_occluded': False,
                        'atr_assigned_position': position,
                    })

        yolo_worker = AsyncYoloWorker(
            model=model,
            conf=DEFAULT_CONFIDENCE,
            imgsz=self.yolo_input_size,
            device=config_constants.DEVICE,
            on_result=_apply_yolo_result,
            logger=self.logger,
            input_queue_size=8,
            batch_size=4,
        )
        yolo_worker.start()

        def _apply_flow_result(f_idx, time_ms_val, _dy_w, _dx_w, meta):
            nonlocal upper_motion_sum, lower_motion_sum, flow_count
            flow = meta.get('flow')
            if flow is None:
                return
            dy, dx = self._magnitude_weighted_flow(flow)
            raw_flow[f_idx] = (time_ms_val, dy, dx)
            flow_count += 1
            patch_h = flow.shape[0]
            split = patch_h // 3
            if split > 0 and patch_h - split > 0:
                upper_motion_sum += np.median(np.abs(flow[:patch_h - split, :, 1]))
                lower_motion_sum += np.median(np.abs(flow[patch_h - split:, :, 1]))

        flow_worker = AsyncDisFlowWorker(
            on_result=_apply_flow_result,
            logger=self.logger,
            input_queue_size=16,
        )
        flow_worker.start()

        try:
            for frame_idx, frame, timing in vp.stream_frames_for_segment(
                    0, total_frames, stop_event=self.stop_event):
                t_decode += float(timing.get('decode_ms', 0.0)) / 1000.0

                if self.stop_event and self.stop_event.is_set():
                    break

                _t0 = time.perf_counter()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                t_cvt += time.perf_counter() - _t0
                h, w = gray.shape[:2]

                # ── YOLO: submit async (runs on worker thread) + drain results ──
                is_chapter_frame = (frame_idx % chapter_skip == 0)
                run_yolo = (frame_idx % roi_skip == 0) or is_chapter_frame
                if run_yolo:
                    if yolo_worker.submit(frame_idx, frame, payload=is_chapter_frame):
                        yolo_count += 1
                yolo_worker.drain()

                # ── DIS flow (every frame) ──
                roi = current_roi
                if roi is None:
                    # Fallback: center 60% of frame
                    mx, my = int(w * 0.2), int(h * 0.2)
                    roi = (mx, my, w - mx, h - my)

                rx1, ry1, rx2, ry2 = roi
                rx1, ry1 = max(0, rx1), max(0, ry1)
                rx2, ry2 = min(w, rx2), min(h, ry2)
                roi_w = rx2 - rx1
                roi_h = ry2 - ry1

                if roi_w > MIN_ROI_SIZE and roi_h > MIN_ROI_SIZE and prev_gray is not None:
                    time_ms_val = int(frame_idx * frame_ms)
                    flow_worker.submit(
                        frame_idx, prev_gray, gray,
                        (rx1, ry1, rx2, ry2), time_ms_val)

                prev_gray = gray

                # Progress
                if progress_callback and frame_idx % 500 == 0 and total_frames > 0:
                    pct = min(75, int(75 * frame_idx / total_frames))
                    elapsed = time.time() - t_start
                    avg_fps = frame_idx / max(0.001, elapsed)
                    eta_s = (total_frames - frame_idx) / avg_fps if avg_fps > 0 else 0.0
                    # Average per-frame ms for the Video Pipeline perf tab.
                    # YOLO timing is read from the async worker's running
                    # counters (it owns inference on a separate thread).
                    seen = max(1, frame_idx)
                    yolo_ms = (
                        yolo_worker.total_inference_ms
                        / max(1, yolo_worker.completed)
                    ) if yolo_worker.completed else 0.0
                    flow_ms = (
                        flow_worker.total_flow_ms
                        / max(1, flow_worker.completed)
                    ) if flow_worker.completed else 0.0
                    timing_ms = {
                        'decode_ms': (t_decode / seen) * 1000.0,
                        'unwarp_ms': 0.0,  # baked into decode via v360
                        'yolo_det_ms': yolo_ms,
                        'flow_ms': flow_ms,
                    }
                    progress_callback({
                        'stage': 'pass1',
                        'phase': 'Single-pass analysis',
                        'task': f'Frame {frame_idx}/{total_frames}',
                        'percentage': pct,
                        'current': frame_idx,
                        'total': total_frames,
                        'avg_fps': avg_fps,
                        'eta_seconds': eta_s,
                        'elapsed_seconds': elapsed,
                        'timing': timing_ms,
                    })

        except Exception as e:
            self.logger.error(f"Single-pass error: {e}", exc_info=True)
        finally:
            yolo_worker.stop()
            flow_worker.stop()
            t_yolo = yolo_worker.total_inference_ms / 1000.0
            t_dis = flow_worker.total_flow_ms / 1000.0
            if yolo_worker.dropped:
                self.logger.info(
                    f"YOLO worker dropped {yolo_worker.dropped} submissions "
                    f"(queue full). Completed {yolo_worker.completed}.")
            if flow_worker.dropped:
                self.logger.info(
                    f"Flow worker dropped {flow_worker.dropped} submissions "
                    f"(queue full). Completed {flow_worker.completed}.")

        elapsed = time.time() - t_start
        self.logger.info(f"Single-pass complete: {flow_count} flow, {yolo_count} YOLO in {elapsed:.1f}s")
        # Per-stage wall-clock. Decode overlaps in its own thread so t_decode
        # is the cumulative sum of decode durations, not sequential elapsed.
        self.logger.info(
            f"Stage timing: decode={t_decode:.1f}s cvt={t_cvt:.1f}s "
            f"yolo={t_yolo:.1f}s dis={t_dis:.1f}s "
            f"other={max(0.0, elapsed - t_cvt - t_yolo - t_dis):.1f}s "
            f"(decode runs in prefetch thread)")

        # Build chapters
        chapters = build_chapters(frame_positions, fps, total_frames, chapter_skip,
                                  penis_frames=penis_frames,
                                  frame_contact_info=frame_contact_info)

        for ch in chapters:
            ch['dense'] = ch['position'] in ACTIVE_POSITIONS
            ch['duration_s'] = round((ch['end_frame'] - ch['start_frame']) / fps, 1)

        return {
            'chapters': chapters,
            'raw_flow': raw_flow,
            'fps': fps,
            'total_frames': total_frames,
            'sparse_detections': sparse_detections,
            'frame_boxes': frame_boxes,
            'upper_motion_sum': upper_motion_sum,
            'lower_motion_sum': lower_motion_sum,
        }

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _roi_shifted(self, old_roi, new_roi) -> bool:
        if old_roi is None or new_roi is None:
            return False
        return (abs(new_roi[0] - old_roi[0]) > ROI_SHIFT_THRESHOLD or
                abs(new_roi[1] - old_roi[1]) > ROI_SHIFT_THRESHOLD or
                abs(new_roi[2] - old_roi[2]) > ROI_SHIFT_THRESHOLD or
                abs(new_roi[3] - old_roi[3]) > ROI_SHIFT_THRESHOLD)

    def _detect_inversion(self, chapter_flow, raw_flow, chapter):
        """Detect motion inversion from upper/lower flow ratio within chapter."""
        upper_sum = 0.0
        lower_sum = 0.0
        # Use the global sums (simple approach — could be per-chapter)
        # For now rely on the stored raw_flow dy signs
        # TODO: per-chapter inversion detection
        return False  # Default: no inversion (same as V1 for most content)

    def _compute_padded_roi(self, penis_box, contact_boxes, h, w):
        px1, py1, px2, py2 = penis_box
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
                px1 = min(px1, best_contact[0])
                py1 = min(py1, best_contact[1])
                px2 = max(px2, best_contact[2])
                py2 = max(py2, best_contact[3])

        roi_w = px2 - px1
        roi_h = py2 - py1
        pad_x = roi_w * ROI_PADDING_FACTOR
        pad_y = roi_h * ROI_PADDING_FACTOR
        return (max(0, int(px1 - pad_x)), max(0, int(py1 - pad_y)),
                min(w, int(px2 + pad_x)), min(h, int(py2 + pad_y)))

    def _magnitude_weighted_flow(self, flow):
        magnitudes = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
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
            dy = np.sum(flow[..., 1] * combined_weights) / total_weight
            dx = np.sum(flow[..., 0] * combined_weights) / total_weight
        else:
            dy = np.median(flow[..., 1])
            dx = np.median(flow[..., 0])
        return dy, dx

    # ─── Post-processing (same as V1) ────────────────────────────────────────

    def _postprocess_chapter_signal(self, raw_positions, chapter_type, fps, invert):
        if len(raw_positions) < SAVGOL_WINDOW:
            return [], []

        frame_indices = np.array([p[0] for p in raw_positions])
        time_ms_arr = np.array([p[1] for p in raw_positions])
        dy_values = np.array([p[2] for p in raw_positions])

        # Median filter
        if len(dy_values) >= FLOW_MEDIAN_WINDOW:
            dy_filtered = median_filter(dy_values, size=FLOW_MEDIAN_WINDOW)
        else:
            dy_filtered = dy_values.copy()

        # Cumulative position
        sign = 1.0 if invert else -1.0
        cumulative = sign * np.cumsum(dy_filtered)

        # Detrend
        detrend_win = int(3.0 * fps)
        if detrend_win % 2 == 0:
            detrend_win += 1
        detrend_win = min(detrend_win, len(cumulative))
        if detrend_win < 3:
            detrend_win = 3
        if detrend_win % 2 == 0:
            detrend_win -= 1

        drift = savgol_filter(cumulative, window_length=detrend_win, polyorder=2)
        detrended = cumulative - drift

        # Fine smoothing
        win = min(SAVGOL_WINDOW, len(detrended))
        if win % 2 == 0:
            win -= 1
        if win >= 3:
            smoothed = savgol_filter(detrended, window_length=win, polyorder=SAVGOL_POLYORDER)
        else:
            smoothed = detrended.copy()

        # Amplitude normalization
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
        normalized = np.clip(normalized, 0, 100)

        self.logger.info(f"  Signal stats: raw_dy range [{dy_values.min():.2f}, {dy_values.max():.2f}], "
                         f"normalized range [{normalized.min():.1f}, {normalized.max():.1f}]")

        # Peak/valley detection
        min_distance = max(3, int(MIN_PEAK_DISTANCE_S * fps))
        peaks, _ = find_peaks(normalized, prominence=MIN_PEAK_PROMINENCE, distance=min_distance)
        valleys, _ = find_peaks(-normalized, prominence=MIN_PEAK_PROMINENCE, distance=min_distance)

        keyframe_indices = set()
        keyframe_indices.update(peaks.tolist())
        keyframe_indices.update(valleys.tolist())
        keyframe_indices.add(0)
        keyframe_indices.add(len(normalized) - 1)
        keyframe_indices = sorted(keyframe_indices)

        # Build actions
        frame_ms_val = 1000.0 / fps
        primary_actions = []
        for ki in keyframe_indices:
            if ki < len(time_ms_arr):
                t = int(round(time_ms_arr[ki] / frame_ms_val) * frame_ms_val)
                pos = max(0, min(100, int(round(normalized[ki]))))
                primary_actions.append({'at': t, 'pos': pos})

        # Deduplicate
        seen = {}
        for a in primary_actions:
            seen[a['at']] = a
        primary_actions = sorted(seen.values(), key=lambda a: a['at'])

        self.logger.info(f"  Post-process: {len(raw_positions)} raw → {len(primary_actions)} keyframes "
                         f"({len(peaks)} peaks, {len(valleys)} valleys)")

        return primary_actions, []

    # ─── Merge ────────────────────────────────────────────────────────────────

    def _merge_chapter_results(self, chapters, chapter_results, fps):
        from funscript.multi_axis_funscript import MultiAxisFunscript

        all_primary = []
        all_secondary = []

        for i, ch in enumerate(chapters):
            start_ms = frame_to_ms(ch['start_frame'], fps)
            end_ms = frame_to_ms(ch['end_frame'], fps)

            if ch.get('dense') and i in chapter_results:
                result = chapter_results[i]
                if result.get('success') and result.get('primary_actions'):
                    all_primary.extend(result['primary_actions'])
                    if result.get('secondary_actions'):
                        all_secondary.extend(result['secondary_actions'])
                    continue

            # Non-dense or failed → hold at 100
            if ch['position'] in ('Close up', 'Not Relevant') or not ch.get('dense'):
                all_primary.append({'at': start_ms, 'pos': 100})
                all_primary.append({'at': end_ms, 'pos': 100})
            else:
                all_primary.append({'at': start_ms, 'pos': 50})
                all_primary.append({'at': end_ms, 'pos': 50})

        all_primary.sort(key=lambda a: a['at'])
        all_secondary.sort(key=lambda a: a['at'])

        # Deduplicate
        all_primary = self._deduplicate_actions(all_primary)
        all_secondary = self._deduplicate_actions(all_secondary)

        funscript = MultiAxisFunscript(logger=self.logger, fps=fps)
        funscript.set_axis_actions('primary', all_primary)
        if all_secondary:
            funscript.set_axis_actions('secondary', all_secondary)

        # Set chapters
        funscript_chapters = []
        for ch in chapters:
            start_ms = frame_to_ms(ch['start_frame'], fps)
            end_ms = frame_to_ms(ch['end_frame'], fps)
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

    def _deduplicate_actions(self, actions):
        seen = {}
        for a in actions:
            seen[a['at']] = a
        return sorted(seen.values(), key=lambda a: a['at'])
