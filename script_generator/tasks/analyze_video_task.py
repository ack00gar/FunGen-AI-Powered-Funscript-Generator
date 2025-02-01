import queue
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import List, TYPE_CHECKING

from script_generator.constants import QUEUE_MAXSIZE
from script_generator.tasks.abstract_task import Task

from script_generator.object_detection.post_process_worker import PostProcessWorker
from script_generator.object_detection.yolo_worker import YoloWorker
from script_generator.video.workers.ffmpeg_worker import VideoWorker
from script_generator.video.workers.vr_to_2d_worker import VrTo2DWorker

if TYPE_CHECKING:
    from script_generator.state.app_state import AppState


@dataclass
class AnalyzeVideoTask(Task):
    tasks: List[Task] = field(default_factory=list)

    def __init__(self, state: "AppState", use_open_gl):
        super().__init__()
        self.tasks = []
        self._lock = Lock()
        self.profile = {}
        self.start_time = time.time()
        self.opengl_q = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self.yolo_q = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self.analysis_q = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self.result_q = queue.Queue(maxsize=0)
        self.use_open_gl = use_open_gl
        self.is_stopped = False

        # Create threads
        self.decode_thread = VideoWorker(state=state, output_queue=self.opengl_q if use_open_gl else self.yolo_q)
        self.opengl_thread = VrTo2DWorker(state=state, input_queue=self.opengl_q, output_queue=self.yolo_q) if use_open_gl else None
        self.yolo_thread = YoloWorker(state=state, input_queue=self.yolo_q, output_queue=self.analysis_q)
        self.yolo_analysis_thread = PostProcessWorker(state=state, input_queue=self.analysis_q, output_queue=self.result_q)

        state.analyze_task = self

    def add_task(self, task: Task) -> Task:
        with self._lock:
            self.tasks.append(task)
        return task

    def get_tasks(self) -> List[Task]:
        with self._lock:
            return list(self.tasks)

    def stop(self):
        self.is_stopped = True
        if self.decode_thread:
            self.decode_thread.release()
        if self.opengl_thread and self.use_open_gl:
            self.opengl_thread.stop_process()
        if self.yolo_thread:
            self.yolo_thread.stop_process()
        if self.yolo_analysis_thread:
            self.yolo_analysis_thread.stop_process()