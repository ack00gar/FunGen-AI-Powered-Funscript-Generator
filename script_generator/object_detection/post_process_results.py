import os
import time

import cv2

from script_generator.constants import CLASS_REVERSE_MATCH, CLASS_COLORS, OUTPUT_PATH
from config import RUN_POSE_MODEL
from script_generator.object_detection.object_detection_result import ObjectDetectionResult
from script_generator.tasks.abstract_task_processor import AbstractTaskProcessor, TaskProcessorTypes
from script_generator.utils.file import write_dataset, get_output_file_path
from script_generator.utils.logger import logger


class YoloAnalysisTaskProcessor(AbstractTaskProcessor):
    process_type = TaskProcessorTypes.YOLO_ANALYSIS
    records = []
    test_result = ObjectDetectionResult()  # Test result object for debugging

    def task_logic(self):
        state = self.state

        for task in self.get_task():
            frame_pos = task.frame_pos
            det_results = task.yolo_results
            frame = task.rendered_frame
            pose_results = None # TODO pose support

            # Skip if no boxes are detected or no tracks are found
            if  det_results.boxes.id is None or (len(det_results.boxes) == 0 and not state.life_display_mode):
                task.rendered_frame = None # Clear memory
                task.yolo_results = None  # Clear memory
                self.finish_task(task)
                continue

            ### DETECTION of BODY PARTS
            # Extract track IDs, boxes, classes, and confidence scores
            track_ids = det_results.boxes.id.cpu().tolist()
            boxes = det_results.boxes.xywh.cpu()
            classes = det_results.boxes.cls.cpu().tolist()
            confs = det_results.boxes.conf.cpu().tolist()

            # Process each detection
            for track_id, cls, conf, box in zip(track_ids, classes, confs, boxes):
                track_id = int(track_id)
                x, y, w, h = box.int().tolist()
                x1 = x - w // 2
                y1 = y - h // 2
                x2 = x + w // 2
                y2 = y + h // 2
                # Create a detection record
                record = [frame_pos, int(cls), round(conf, 1), x1, y1, x2, y2, track_id]
                self.records.append(record)
                if state.life_display_mode:
                    test_box = [[x1, y1, x2, y2], round(conf, 1), int(cls), CLASS_REVERSE_MATCH.get(int(cls), 'unknown'), track_id]
                    self.test_result.add_record(frame_pos, test_box)

                    # print and test the record
                    logger.trace(f"Record : {record}")
                    logger.trace(f"For class id: {int(cls)}, getting: {CLASS_REVERSE_MATCH.get(int(cls), 'unknown')}")
                    logger.trace(f"Test box: {test_box}")

            if RUN_POSE_MODEL:
                ### POSE DETECTION - Hips and wrists
                # Extract track IDs, boxes, classes, and confidence scores
                if len(pose_results[0].boxes) > 0 and pose_results[0].boxes.id is not None:
                    pose_track_ids = pose_results[0].boxes.id.cpu().tolist()

                    # Check if keypoints are detected
                    if pose_results[0].keypoints is not None:
                        # logger.trace("We have keypoints")
                        # pose_keypoints = pose_results[0].keypoints.cpu()
                        # pose_track_ids = pose_results[0].boxes.id.cpu().tolist()
                        # pose_boxes = pose_results[0].boxes.xywh.cpu()
                        # pose_classes = pose_results[0].boxes.cls.cpu().tolist()
                        pose_confs = pose_results[0].boxes.conf.cpu().tolist()

                        pose_keypoints = pose_results[0].keypoints.cpu()
                        pose_keypoints_list = pose_keypoints.xy.cpu().tolist()
                        left_hip = pose_keypoints_list[0][11]
                        right_hip = pose_keypoints_list[0][12]

                        middle_x_frame = frame.shape[1] // 2
                        mid_hips = [middle_x_frame, (int(left_hip[1]) + int(right_hip[1])) // 2]
                        x1 = mid_hips[0] - 5
                        y1 = mid_hips[1] - 5
                        x2 = mid_hips[0] + 5
                        y2 = mid_hips[1] + 5
                        cls = 10  # hips center
                        # logger.trace(f"pose_confs: {pose_confs}")
                        conf = pose_confs[0]

                        record = [frame_pos, 10, round(conf, 1), x1, y1, x2, y2, 0]
                        self.records.append(record)
                        if state.life_display_mode:
                            # Print and test the record
                            logger.debug(f"Record : {record}")
                            logger.debug(f"For class id: {int(cls)}, getting: {CLASS_REVERSE_MATCH.get(int(cls), 'unknown')}")
                            test_box = [[x1, y1, x2, y2], round(conf, 1), int(cls),
                                        CLASS_REVERSE_MATCH.get(int(cls), 'unknown'), 0]
                            logger.debug(f"Test box: {test_box}")
                            self.test_result.add_record(frame_pos, test_box)

            if state.life_display_mode:
                # Display the YOLO results for testing
                # det_results.plot()
                # cv2.imshow("YOLO11", det_results.plot())
                # cv2.waitKey(1)
                # Verify the sorted boxes
                sorted_boxes = self.test_result.get_boxes(frame_pos)
                # logger.debug(f"Sorted boxes : {sorted_boxes}")

                # frame_display = frame.copy()
                frame_display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                for box in sorted_boxes:
                    color = CLASS_COLORS.get(box[3])
                    cv2.rectangle(frame_display, (box[0][0], box[0][1]), (box[0][2], box[0][3]), color, 2)
                    cv2.putText(frame_display, f"{box[4]}: {box[3]}", (box[0][0], box[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.imshow("YOLO11 test boxes Tracking", frame_display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            task.rendered_frame = None # Clear memory
            task.yolo_results = None # Clear memory (yolo results contains a copy of the image)
            self.finish_task(task)
            

    def on_last_item(self):
        self.state.analyze_task.end_time = time.time()

        # Write the detection records to a JSON file
        raw_yolo_path, _ = get_output_file_path(self.state.video_path, "_rawyolo.json")
        write_dataset(raw_yolo_path, self.records)
        