import time

import cv2

from config import YOLO_CONF, YOLO_BATCH_SIZE, YOLO_MODEL

from script_generator.tasks.abstract_task_processor import AbstractTaskProcessor, TaskProcessorTypes


class YoloTaskProcessor(AbstractTaskProcessor):
    process_type = TaskProcessorTypes.YOLO

    # TODO add pose model support
    # if run_pose_model:
    #     yolo_pose_results = pose_model.track(frame, persist=True, conf=YOLO_CONF, verbose=False)

    def task_logic(self):
        batch = []
        tasks = []

        for task in self.get_task():
            if task.rendered_frame is not None:
                batch.append(task.rendered_frame)
                tasks.append(task)

                # If batch is ready, process it
                if len(batch) >= YOLO_BATCH_SIZE:
                    self.process_batch(batch, tasks)
                    batch = []
                    tasks = []

        # Process any remaining tasks in the batch
        if batch:
            self.process_batch(batch, tasks)

    def process_batch(self, frames, tasks):
        start_time = time.time()
        # yolo_results = YOLO_MODEL(frames, conf=YOLO_CONF, verbose=False)
        yolo_results = YOLO_MODEL.track(frames, persist=True, conf=YOLO_CONF, verbose=False)
        avg_time = (time.time() - start_time) / len(tasks)

        for t, result in zip(tasks, yolo_results):
            t.yolo_results = result
            batch_time = time.time() - start_time
            t.duration(str(self.process_type), avg_time)
            self.finish_task(t)
