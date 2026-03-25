"""
Unified YOLO Detection Helper for FunGen.

Provides standardized detection/pose functions and a stateful helper class
used by all YOLO integration points: Stage 1 consumers, live trackers
(YOLO ROI, Hybrid Flow), and offline trackers (VR Hybrid).

Design:
- Stateless functions (run_detection, run_pose, load_model) for multiprocessing
  workers where each process loads its own model.
- Stateful YoloDetectionHelper class for live/in-process use where a model
  is loaded once and reused across frames.
- Standardized Detection/PoseResult dataclasses replace ad-hoc dicts everywhere.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Standardized data classes
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Detection:
    """A single YOLO detection result."""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    class_id: int
    class_name: str
    confidence: float


@dataclass(slots=True)
class PoseResult:
    """A single YOLO pose estimation result."""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    keypoints: List[List[float]]  # [[x, y, conf], ...] per joint


# ---------------------------------------------------------------------------
#  Stateless functions (safe for multiprocessing workers)
# ---------------------------------------------------------------------------

def load_model(path: str, task: str = 'detect'):
    """Load a YOLO model. Each process/thread should call this independently.

    Args:
        path: Path to YOLO model file (.pt, .engine, etc.)
        task: 'detect' or 'pose'

    Returns:
        ultralytics.YOLO model instance
    """
    from ultralytics import YOLO
    return YOLO(path, task=task)


def run_detection(
    model,
    frame: np.ndarray,
    conf: float = 0.25,
    imgsz: int = 640,
    device: str = 'auto',
) -> List[Detection]:
    """Run YOLO detection on a single frame.

    Args:
        model: Loaded YOLO model (from load_model or ultralytics.YOLO)
        frame: BGR numpy array
        conf: Confidence threshold
        imgsz: Input image size for inference
        device: Device string ('auto', 'cpu', 'cuda', 'mps', etc.)

    Returns:
        List of Detection objects sorted by confidence (highest first).
    """
    results = model(frame, device=device, verbose=False, conf=conf, imgsz=imgsz)
    return _parse_detections(results, model.names)


def run_pose(
    model,
    frame: np.ndarray,
    conf: float = 0.25,
    imgsz: int = 640,
    device: str = 'auto',
) -> List[PoseResult]:
    """Run YOLO pose estimation on a single frame.

    Args:
        model: Loaded YOLO pose model
        frame: BGR numpy array
        conf: Confidence threshold
        imgsz: Input image size for inference
        device: Device string

    Returns:
        List of PoseResult objects.
    """
    results = model(frame, device=device, verbose=False, conf=conf, imgsz=imgsz)
    return _parse_poses(results)


def _parse_detections(results, model_names: Dict[int, str]) -> List[Detection]:
    """Parse ultralytics Results into Detection list."""
    detections: List[Detection] = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0])
            detections.append(Detection(
                bbox=tuple(box.xyxy[0].tolist()),
                class_id=cls_id,
                class_name=model_names.get(cls_id, 'unknown'),
                confidence=float(box.conf[0]),
            ))
    # Highest confidence first
    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


def _parse_poses(results) -> List[PoseResult]:
    """Parse ultralytics Results into PoseResult list."""
    poses: List[PoseResult] = []
    for r in results:
        if r.keypoints is None or r.boxes is None:
            continue
        for i in range(len(r.boxes)):
            poses.append(PoseResult(
                bbox=tuple(r.boxes.xyxy[i].tolist()),
                keypoints=r.keypoints.data[i].tolist(),
            ))
    return poses


def filter_detections(
    detections: List[Detection],
    include_classes: Optional[Set[str]] = None,
    exclude_classes: Optional[Set[str]] = None,
    min_confidence: float = 0.0,
) -> List[Detection]:
    """Filter a detection list by class name and confidence.

    Args:
        detections: List of Detection objects
        include_classes: If set, only keep these class names
        exclude_classes: If set, remove these class names
        min_confidence: Minimum confidence threshold

    Returns:
        Filtered list (new list, originals unchanged).
    """
    out = detections
    if include_classes is not None:
        out = [d for d in out if d.class_name in include_classes]
    if exclude_classes is not None:
        out = [d for d in out if d.class_name not in exclude_classes]
    if min_confidence > 0:
        out = [d for d in out if d.confidence >= min_confidence]
    return out


def detection_to_dict(det: Detection) -> dict:
    """Convert Detection to the legacy dict format for backward compatibility.

    Returns:
        {'bbox': [x1,y1,x2,y2], 'class_id': int, 'class_name': str,
         'confidence': float, 'class': int}
    """
    return {
        'bbox': list(det.bbox),
        'class_id': det.class_id,
        'class': det.class_id,
        'class_name': det.class_name,
        'confidence': det.confidence,
    }


def pose_to_dict(pose: PoseResult) -> dict:
    """Convert PoseResult to the legacy dict format for backward compatibility.

    Returns:
        {'bbox': [x1,y1,x2,y2], 'keypoints': [[x,y,conf], ...]}
    """
    return {
        'bbox': list(pose.bbox),
        'keypoints': pose.keypoints,
    }


# ---------------------------------------------------------------------------
#  Stateful helper class (for live trackers / in-process batch)
# ---------------------------------------------------------------------------

class YoloDetectionHelper:
    """Manages YOLO model(s) and provides a convenient detection API.

    Use this in live trackers and in-process batch processing where
    the model is loaded once and reused across many frames.

    For multiprocessing workers (Stage 1 consumers), use the stateless
    functions directly (load_model + run_detection + run_pose).
    """

    def __init__(
        self,
        det_model_path: str,
        pose_model_path: Optional[str] = None,
        device: str = 'auto',
        conf: float = 0.25,
        imgsz: int = 640,
        logger_instance: Optional[logging.Logger] = None,
    ):
        self.logger = logger_instance or logger
        self.device = device
        self.conf = conf
        self.imgsz = imgsz

        self.logger.info(f"Loading YOLO detection model: {det_model_path}")
        self._det_model = load_model(det_model_path, task='detect')

        self._pose_model = None
        if pose_model_path:
            self.logger.info(f"Loading YOLO pose model: {pose_model_path}")
            self._pose_model = load_model(pose_model_path, task='pose')

    # -- Detection API --

    def detect(self, frame: np.ndarray, conf: Optional[float] = None) -> List[Detection]:
        """Run detection on a single frame."""
        return run_detection(
            self._det_model, frame,
            conf=conf if conf is not None else self.conf,
            imgsz=self.imgsz, device=self.device,
        )

    def detect_filtered(
        self,
        frame: np.ndarray,
        include_classes: Optional[Set[str]] = None,
        exclude_classes: Optional[Set[str]] = None,
        conf: Optional[float] = None,
    ) -> List[Detection]:
        """Run detection and filter results by class name."""
        dets = self.detect(frame, conf=conf)
        return filter_detections(dets, include_classes, exclude_classes)

    def detect_with_pose(
        self, frame: np.ndarray, conf: Optional[float] = None,
    ) -> Tuple[List[Detection], List[PoseResult]]:
        """Run detection + pose on a single frame.

        Returns (detections, poses). Poses empty if no pose model loaded.
        """
        c = conf if conf is not None else self.conf
        dets = run_detection(self._det_model, frame, conf=c, imgsz=self.imgsz, device=self.device)
        poses = []
        if self._pose_model is not None:
            poses = run_pose(self._pose_model, frame, conf=c, imgsz=self.imgsz, device=self.device)
        return dets, poses

    # -- Properties --

    @property
    def class_names(self) -> Dict[int, str]:
        """Detection model's class name mapping."""
        return dict(self._det_model.names)

    @property
    def det_model(self):
        """Direct access to the detection model (for advanced use)."""
        return self._det_model

    @property
    def pose_model(self):
        """Direct access to the pose model (may be None)."""
        return self._pose_model


# ---------------------------------------------------------------------------
#  Stage 1 parallel analysis (reusable public API)
# ---------------------------------------------------------------------------

def run_parallel_yolo_analysis(
    video_path: str,
    det_model_path: str,
    pose_model_path: Optional[str] = None,
    output_path: Optional[str] = None,
    num_producers: int = 2,
    num_consumers: int = 2,
    yolo_input_size: int = 640,
    confidence_threshold: float = 0.25,
    video_processor_kwargs: Optional[dict] = None,
    progress_callback=None,
    stop_event=None,
    logger_instance: Optional[logging.Logger] = None,
) -> Optional[str]:
    """Run YOLO analysis in multiprocessing mode (Stage 1 pipeline).

    This wraps the Stage 1 producer-consumer pipeline from detection/cd/stage_1_cd.py
    with a clean public API. Any offline tracker can call this to get YOLO
    detection data without going through the orchestrator.

    Args:
        video_path: Path to video file
        det_model_path: Path to YOLO detection model
        pose_model_path: Path to YOLO pose model (optional)
        output_path: Output msgpack file path (optional)
        num_producers: Number of frame producer processes
        num_consumers: Number of YOLO consumer processes
        yolo_input_size: Input size for YOLO inference
        confidence_threshold: Detection confidence threshold
        video_processor_kwargs: Extra kwargs for VideoProcessor (vr_format, etc.)
        progress_callback: Optional callback(current_frame, total_frames)
        stop_event: Optional threading/multiprocessing Event to cancel
        logger_instance: Optional logger

    Returns:
        Path to output msgpack file, or None on failure.
    """
    from multiprocessing import Event as MPEvent

    try:
        from detection.cd.stage_1_cd import perform_yolo_analysis
    except ImportError:
        (logger_instance or logger).error("Cannot import Stage 1 module (detection.cd.stage_1_cd)")
        return None

    vp_kwargs = video_processor_kwargs or {}

    # Build logger config dict (perform_yolo_analysis expects dict, not Logger)
    log = logger_instance or logger
    log_config = {
        'main_logger': log,
        'log_file': None,
        'log_level': log.level,
    }

    result_path, _max_fps = perform_yolo_analysis(
        video_path_arg=video_path,
        yolo_model_path_arg=det_model_path,
        yolo_pose_model_path_arg=pose_model_path or '',
        confidence_threshold=confidence_threshold,
        progress_callback=progress_callback or (lambda *a: None),
        stop_event_external=stop_event or MPEvent(),
        num_producers_arg=num_producers,
        num_consumers_arg=num_consumers,
        video_type_arg=vp_kwargs.get('video_type', 'auto'),
        vr_input_format_arg=vp_kwargs.get('vr_input_format', 'he'),
        vr_fov_arg=vp_kwargs.get('vr_fov', 190),
        vr_pitch_arg=vp_kwargs.get('vr_pitch', 0),
        yolo_input_size_arg=yolo_input_size,
        app_logger_config_arg=log_config,
        output_filename_override=output_path,
    )
    return result_path
