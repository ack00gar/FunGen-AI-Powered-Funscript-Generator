import numpy as np
import cv2

class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.position = None
        self.detected = False
        self.prediction_count = 0
        self.half_height = 20
        self.half_width = 20
        self.touching = False

    def update(self, measurement):
        measured = np.array([[np.float32(measurement[0])],
                             [np.float32(measurement[1])]])
        self.kf.correct(measured)
        self.position = measurement
        self.detected = True
        self.prediction_count = 0

    def predict(self):
        prediction = self.kf.predict()
        self.position = (prediction[0] - self.half_width, prediction[1] - self.half_height,
                         prediction[0] + self.half_width, prediction[1] + self.half_height)
        self.detected = False
        self.prediction_count += 1
        return prediction[0]

