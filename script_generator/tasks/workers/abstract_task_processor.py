import queue
import threading
from typing import Generator, Optional, TYPE_CHECKING
from enum import Enum
from script_generator.debug.logger import log

if TYPE_CHECKING:
    from script_generator.video.analyse_frame_task import AnalyzeFrameTask
    from script_generator.state.app_state import AppState

class AbstractTaskProcessor(threading.Thread):

    process_type = ""

    def __init__(self, state: "AppState", output_queue: queue.Queue, input_queue: Optional[queue.Queue] = None):
        """
        Abstract thread class to handle lifecycle management and task handling boilerplate.

        :param input_queue: Queue to consume tasks from.
        :param output_queue: Queue to produce processed tasks.
        """
        super().__init__()
        self.state = state
        self.input_queue = input_queue
        self.output_queue = output_queue
        self._stop_event = threading.Event()
        self.exception = None  # Store the exception that occurs in the thread

    def log(self, message):
        """
        Unified logging for the thread.
        :param message: Message to log.
        """
        thread_name = threading.current_thread().name
        log.info(f"[{self.__class__.__name__}-{thread_name}] {message}")

    def get_task(self) -> Generator["AnalyzeFrameTask", None, None]:
        """
        Generator for retrieving tasks from the input queue.
        Yields tasks until a sentinel (None) is encountered or the thread is stopped.
        Logs and tracks the time taken to retrieve tasks.
        """
        if self.input_queue is None:
            raise ValueError("Input queue is None. An input queue must be provided to use get_task().")

        while not self._stop_event.is_set():
            try:
                task = self.input_queue.get(timeout=1)

                if task is None:
                    self.input_queue.task_done()  # Remove sentinel
                    self.state.analyze_task.end(self.process_type)
                    self.on_last_item()
                    self.finish_task(None)
                    break
                yield task
            except queue.Empty:
                continue

    def finish_task(self, task):
        """
        Finalizes the task by placing it in the output queue.
        If the queue is full, waits until a spot is available.

        :param task: The task to place in the output queue.
        """
        while not self._stop_event.is_set():
            try:
                self.output_queue.put(task, timeout=1)
                break
            except queue.Full:
                return

    def run(self):
        """
        Main thread entry point. Executes the `task_logic` method.
        Catches any exceptions and stores them for the caller.
        """
        try:
            self.state.analyze_task.start(self.process_type)
            self.task_logic()
        except Exception as e:
            self.exception = e  # Capture the exception
            # Propagate sentinel to the output queue
            self.output_queue.put(None)
            log.error(f"An error occurred during task execution on thread {self.process_type}: {e}")
            # import traceback
            # traceback.print_exc()
        finally:
            self._stop_event.set()

    def task_logic(self):
        """
        Abstract method for setup, task processing, and cleanup.
        Subclasses must override this method.
        """
        raise NotImplementedError("Subclasses must implement task_logic")

    def stop_process(self):
        self.state.analyze_task.end(self.process_type)
        self.on_last_item()
        # Propagate sentinel to the output queue
        self.output_queue.put(None)

    def on_last_item(self):
        return

    def check_exception(self):
        """
        Checks if an exception occurred in the thread and raises it in the calling context.
        """
        if self.exception:
            raise self.exception


class TaskProcessorTypes(Enum):
    VIDEO = "Video processing"
    OPENGL = "3D to 2D"
    METAL = "3D to 2D (MPS)"
    YOLO = "YOLO inference"
    YOLO_ANALYSIS = "YOLO analysis"

    def __str__(self):
        return self.value
