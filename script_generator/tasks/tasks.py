import time
from dataclasses import dataclass, field
from threading import Lock
from typing import List, Dict, Optional

import numpy as np


@dataclass
class Task:
    id: int = field(init=False)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    profile: Dict[str, float] = field(default_factory=dict)  # Timing/profiling info
    _id_counter: int = 0
    _id_lock: Lock = Lock()
    _lock: Lock = field(default_factory=Lock, repr=False, init=False)

    def __post_init__(self):
        # Use the class-level lock for thread safety
        with self.__class__._id_lock:
            self.__class__._id_counter += 1
            self.id = self.__class__._id_counter

    def start(self, process_type: str):
        self._update_profile(process_type, "start")

    def end(self, process_type: str):
        self._update_profile(process_type, "end")
        self._calculate_duration(process_type)

    def duration(self, process_type: str, duration: int):
        key = f"{process_type}_duration"
        with self._lock:
            self.profile[key] = duration

    def _update_profile(self, process_type: str, action: str):
        key = f"{process_type}_{action}"
        with self._lock:
            self.profile[key] = time.time()

    def _calculate_duration(self, process_type: str):
        start_key = f"{process_type}_start"
        end_key = f"{process_type}_end"
        duration_key = f"{process_type}_duration"

        with self._lock:
            if start_key in self.profile and end_key in self.profile:
                self.profile[duration_key] = self.profile[end_key] - self.profile[start_key]


@dataclass
class AnalyzeVideoTask(Task):
    tasks: List[Task] = field(default_factory=list)

    def __init__(self):
        super().__init__()
        self.tasks = []
        self._lock = Lock()
        self.profile = {}
        self.start_time = time.time()

    def add_task(self, task: Task) -> Task:
        with self._lock:
            self.tasks.append(task)
        return task

    def get_tasks(self) -> List[Task]:
        with self._lock:
            return list(self.tasks)


@dataclass
class AnalyzeFrameTask(Task):
    frame_pos: int = -1
    preprocessed_frame: Optional[np.ndarray] = None  # Cropped frame from video stream
    rendered_frame: Optional[np.ndarray] = None  # The final 2D image from OpenGL
    yolo_results = None
    # detections: List[Detection] = field(default_factory=list) # YOLO detection results
