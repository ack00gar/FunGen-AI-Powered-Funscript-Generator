"""
Shared single-pass feature extraction helper.

Used by both LearnedFlowNPZTracker and FunGenProV2LocalTracker to extract
flow + YOLO features in a single ffmpeg decode pass. This is NOT intellectual
property — it's just feature extraction (optical flow + YOLO detections).

Architecture:
  VR video → ffmpeg v360 → 640×640 bgr24 stream
    Every frame:   crop ROI → grayscale → DIS optical flow → 240-dim features
    Every Nth:     YOLO on full 640×640 → detections for chapter classifier
"""

import logging
import os
import subprocess
import time
import cv2
import numpy as np
from typing import Optional, Callable, Dict, Any, List

from tracker.tracker_modules.helpers.yolo_detection_helper import (
    run_detection as _yolo_run_detection,
)

logger = logging.getLogger("tracker.extraction_helpers")

# Contact classes used for filtering YOLO detections (same as train_position_classifier)
_RELEVANT_CLASSES = {'penis', 'pussy', 'butt', 'anus', 'face', 'hand', 'breast', 'foot'}


def _run_yolo_on_frame_via_helper(yolo_model, frame, yolo_input_size: int = 640) -> list:
    """Run YOLO via unified helper, return legacy normalized-coord dicts.

    Drop-in replacement for train_position_classifier.run_yolo_on_frame,
    producing the same dict keys: class, conf, cx, cy, area, x1, y1, x2, y2.
    """
    from config import constants as config_constants

    try:
        det_objs = _yolo_run_detection(
            yolo_model, frame, conf=0.25, imgsz=yolo_input_size,
            device=config_constants.DEVICE)
    except Exception:
        return []

    h, w = frame.shape[:2]
    dets = []
    for d in det_objs:
        if d.class_name not in _RELEVANT_CLASSES:
            continue
        x1, y1, x2, y2 = d.bbox
        dets.append({
            'class': d.class_name,
            'conf': round(d.confidence, 3),
            'cx': round((x1 + x2) / 2.0 / w, 4),
            'cy': round((y1 + y2) / 2.0 / h, 4),
            'area': round(((x2 - x1) * (y2 - y1)) / (h * w), 5),
            'x1': round(x1 / w, 4),
            'y1': round(y1 / h, 4),
            'x2': round(x2 / w, 4),
            'y2': round(y2 / h, 4),
        })
    return dets


def extract_features_single_pass(
    video_path: str,
    yolo_model=None,
    sparse_fps: float = 2.0,
    progress_callback: Optional[Callable] = None,
    frame_range: Optional[tuple] = None,
    confidence: float = 0.25,
) -> Optional[Dict[str, Any]]:
    """Single-pass extraction: VR video → v360 → flow + sparse YOLO.

    One ffmpeg process streams v360-dewarped 640×640 bgr24 frames.
    In the same loop:
    - Every frame: crop ROI → grayscale → DIS optical flow → 240-dim features
    - Every Nth frame (if yolo_model): YOLO on full 640×640 → chapter detections

    Args:
        video_path: Path to VR source video.
        yolo_model: Loaded YOLO model (ultralytics), or None to skip YOLO.
        sparse_fps: YOLO sample rate (default 2 fps).
        progress_callback: Optional (pct, msg) callback.
        frame_range: Optional (start_frame, end_frame) tuple.
        confidence: YOLO confidence threshold.

    Returns:
        dict with keys:
            flow_features: (N, 240) float16 array
            timestamps_ms: (N,) float64 array
            fps: float — video frame rate
            yolo_detections: list of (timestamp_s, dets_list) or None
        Or None on failure.
    """
    from funscript.learning.feature_extractor_v2 import (
        ROI_X_START, ROI_Y_START, ROI_W, ROI_H, FRAME_SIZE,
        CELL_SIZE, GRID_ROWS, GRID_COLS, N_FEATURES,
        _compute_grid_features,
    )
    from tracker.tracker_modules.offline.train_position_classifier import (
        open_video_processor,
    )

    # Open VideoProcessor for v360 filter + video metadata
    vp = open_video_processor(video_path, yolo_input_size=FRAME_SIZE)
    if vp is None:
        logger.error("Failed to open VideoProcessor for %s", video_path)
        return None

    vf_filter = vp.ffmpeg_filter_string
    video_fps = vp.fps or 60.0
    total_frames = vp.total_frames or 0

    frame_skip = max(1, int(video_fps / sparse_fps))
    do_yolo = yolo_model is not None

    logger.info(
        "Single-pass extraction: %d frames @ %.1f fps%s",
        total_frames, video_fps,
        f", YOLO every {frame_skip}th frame" if do_yolo else "",
    )

    # Build ffmpeg command: stream full 640×640 bgr24 frames
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-loglevel", "error",
        "-i", video_path, "-an", "-sn",
        "-vf", vf_filter,
        "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1",
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={**os.environ, "PATH": f"/opt/homebrew/bin:{os.environ.get('PATH', '')}"},
    )

    frame_bytes = FRAME_SIZE * FRAME_SIZE * 3  # 640 * 640 * 3

    # DIS optical flow setup
    flow_algo = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    row_bounds = [(r * CELL_SIZE, (r + 1) * CELL_SIZE) for r in range(GRID_ROWS)]
    col_bounds = [(c * CELL_SIZE, (c + 1) * CELL_SIZE) for c in range(GRID_COLS)]

    # Pre-allocate output arrays
    est = max(total_frames, 100)
    timestamps_ms = np.empty(est, dtype=np.float64)
    flow_features = np.empty((est, N_FEATURES), dtype=np.float16)

    prev_gray = None
    prev_flow = None
    n_out = 0
    frame_idx = 0
    yolo_detections: List = []
    t0 = time.time()

    try:
        while True:
            data = proc.stdout.read(frame_bytes)
            if len(data) < frame_bytes:
                break

            bgr = np.frombuffer(data, dtype=np.uint8).reshape(
                FRAME_SIZE, FRAME_SIZE, 3)

            # ── DIS flow (every frame) ──
            roi = bgr[ROI_Y_START:ROI_Y_START + ROI_H,
                      ROI_X_START:ROI_X_START + ROI_W]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                flow = flow_algo.calc(prev_gray, gray, None)
                features = _compute_grid_features(
                    flow, prev_flow, row_bounds, col_bounds)

                if n_out >= len(timestamps_ms):
                    new_size = len(timestamps_ms) * 2
                    timestamps_ms = np.resize(timestamps_ms, new_size)
                    flow_features = np.resize(
                        flow_features, (new_size, N_FEATURES))

                timestamps_ms[n_out] = frame_idx * 1000.0 / video_fps
                flow_features[n_out] = features
                n_out += 1
                prev_flow = flow.copy()

            prev_gray = gray

            # ── Sparse YOLO (every Nth frame) ──
            if do_yolo and frame_idx % frame_skip == 0:
                yolo_frame = bgr.copy()
                dets = _run_yolo_on_frame_via_helper(
                    yolo_model, yolo_frame, yolo_input_size=FRAME_SIZE)
                yolo_detections.append((frame_idx / video_fps, dets))

            frame_idx += 1

            if progress_callback and frame_idx % 600 == 0:
                elapsed = time.time() - t0
                fps_rate = frame_idx / elapsed if elapsed > 0 else 0
                pct = frame_idx / total_frames if total_frames > 0 else 0
                eta = (total_frames - frame_idx) / fps_rate if fps_rate > 0 else 0
                yolo_str = f", {len(yolo_detections)} YOLO" if do_yolo else ""
                progress_callback({
                    'stage': 'extraction',
                    'task': f"Flow + YOLO: {frame_idx}/{total_frames} "
                            f"({fps_rate:.0f} fps{yolo_str})",
                    'percentage': round(min(pct, 0.88) * 100, 1),
                    'time_elapsed': round(elapsed, 1),
                    'eta_seconds': round(eta, 1),
                    'fps': round(fps_rate, 1),
                })

    finally:
        proc.stdout.close()
        proc.wait()

    if n_out == 0:
        logger.error("No flow frames extracted from %s", video_path)
        return None

    # Trim to actual size
    timestamps_ms = timestamps_ms[:n_out]
    flow_features = flow_features[:n_out]

    elapsed = time.time() - t0
    fps_actual = frame_idx / elapsed if elapsed > 0 else 0
    logger.info(
        "Single-pass done: %d flow frames, %d YOLO frames, "
        "%d total in %.1fs (%.0f fps)",
        n_out, len(yolo_detections), frame_idx, elapsed, fps_actual,
    )

    return {
        "flow_features": flow_features,
        "timestamps_ms": timestamps_ms,
        "fps": video_fps,
        "yolo_detections": yolo_detections if yolo_detections else None,
    }
