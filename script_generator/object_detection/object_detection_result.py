from script_generator.constants import CLASS_PRIORITY_ORDER


class ObjectDetectionResult:
    def __init__(self):
        """
        Define the Result class to store and manage detection results
        :param image_width: Width of the image/frame.
        """
        self.frame_data = {}  # Dictionary to store data for each frame

    def add_record(self, frame_id, box_record):
        """
        Add a BoxRecord to the frame_data dictionary.
        :param frame_id: The frame ID to which the record belongs.
        :param box_record: The BoxRecord object to add.
        """
        if frame_id in self.frame_data:
            self.frame_data[frame_id].append(box_record)
        else:
            self.frame_data[frame_id] = [box_record]

    def get_boxes(self, frame_id):
        """
        Retrieve and sort bounding boxes for a specific frame.
        :param frame_id: The frame ID to retrieve boxes for.
        :return: A list of sorted bounding boxes.
        """
        itemized_boxes = []
        if frame_id not in self.frame_data:
            return itemized_boxes
        boxes = self.frame_data[frame_id]
        for box, conf, cls, class_name, track_id in boxes:
            itemized_boxes.append((box, conf, cls, class_name, track_id))
        # Sort boxes based on class priority order
        sorted_boxes = sorted(
            itemized_boxes,
            key=lambda x: CLASS_PRIORITY_ORDER.get(x[3], 7)  # Default priority is 7 if class not found
        )
        return sorted_boxes

    def get_all_frame_ids(self):
        """
        Get a list of all frame IDs in the frame_data dictionary.
        :return: A list of frame IDs.
        """
        return list(self.frame_data.keys())
