class BoxRecord:
    def __init__(self, box, conf, cls, class_name, track_id):
        """
        Define the BoxRecord class to store bounding box information
        :param box: Bounding box coordinates [x1, y1, x2, y2].
        :param conf: Confidence score of the detection.
        :param cls: Class ID of the detected object.
        :param class_name: Class name of the detected object.
        :param track_id: Track ID for object tracking.
        """
        self.box = box
        self.conf = conf
        self.cls = cls
        self.class_name = class_name
        self.track_id = int(track_id)

    def __iter__(self):
        """
        Make the BoxRecord object iterable.
        :return: An iterator over the box, confidence, class, class name, and track ID.
        """
        return iter((self.box, self.conf, self.cls, self.class_name, self.track_id))
