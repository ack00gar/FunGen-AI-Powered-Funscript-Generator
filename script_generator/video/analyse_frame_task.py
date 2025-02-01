from dataclasses import dataclass
from typing import Optional

import numpy as np

from script_generator.tasks.abstract_task import Task


@dataclass
class AnalyzeFrameTask(Task):
    frame_pos: int = -1
    preprocessed_frame: Optional[np.ndarray] = None  # Cropped frame from video stream
    rendered_frame: Optional[np.ndarray] = None  # The final 2D image from OpenGL
    yolo_results = None
    # detections: List[Detection] = field(default_factory=list) # YOLO detection results