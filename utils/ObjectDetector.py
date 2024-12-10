from ultralytics import YOLO
import gc

class ObjectDetector:
    def __init__(self, model_name):
        self.model = YOLO(model_name, task='detect')
        self.class_priority_order = {
            "glans": 0, "penis": 1, "breast": 2, "navel": 3, "pussy": 4, "butt": 5, "face": 6
        }
        self.class_default_priority = 7
        class_types = {'face': 0, 'hand': 1, 'penis': 2, 'glans': 3, 'pussy': 4, 'butt': 5,
                       'anus': 6, 'breast': 7, 'navel': 8, 'foot': 9}

        # Define priority order
        class_priority_order = {"glans": 0, "penis": 1, "breast": 2, "navel": 3, "pussy": 4, "butt": 5, "face": 6}
        class_default_priority = 7  # For classes not explicitly listed

    def detect(self, frame):
        results = self.model(frame, verbose=False)
        boxes = []
        for result in results:
            for *box, conf, cls in result.boxes.data.cpu().numpy():
                #print(f"Detected {result.names[int(cls)]} with confidence {conf} and box {box}")
                class_name = self.map_class_type_to_name(result.names[int(cls)], box[0], box[2], frame.shape[1])
                boxes.append((box, conf, cls, class_name))
        # Sort the collected boxes by priority
        sorted_boxes = sorted(
            boxes,
            key=lambda x: self.class_priority_order.get(x[3], self.class_default_priority)  # Sort by class name priority
        )
        del results
        gc.collect()
        return sorted_boxes

    def map_class_type_to_name(self, class_type, x1, x2, image_width):
        if class_type in ['foot', 'hand']:
            # Call it left if it is mainly on the left of the frame, right otherwise
            if (x1 + x2) / 2 < image_width / 2:
                class_name = 'left ' + class_type
            else:
                class_name = 'right ' + class_type
        else:
            class_name = class_type
        return class_name
