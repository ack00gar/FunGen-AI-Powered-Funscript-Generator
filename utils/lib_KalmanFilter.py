import numpy as np
import cv2

# TODO where is this used?
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
        corrected_state = self.kf.correct(measured)
        #self.position = measurement
        self.position = (corrected_state[0], corrected_state[1])
        self.detected = True
        self.prediction_count = 0
        return float(corrected_state[0])

    def predict(self):
        prediction = self.kf.predict()
        self.position = (prediction[0] - self.half_width, prediction[1] - self.half_height,
                         prediction[0] + self.half_width, prediction[1] + self.half_height)
        self.detected = False
        self.prediction_count += 1
        return prediction[0]

# TODO where is this used?
class KalmanFilter_distance:
    def __init__(self):
        # Initialize a 1D Kalman filter (state: [distance, velocity])
        self.kf = cv2.KalmanFilter(2, 1)

        # Measurement matrix (we only observe distance)
        self.kf.measurementMatrix = np.array([[1, 0]], np.float32)

        # Transition matrix (assume constant velocity model)
        self.kf.transitionMatrix = np.array([[1, 1],
                                             [0, 1]], np.float32)

        # Process noise covariance (tune these values)
        self.kf.processNoiseCov = np.eye(2, dtype=np.float32) * 0.01

        # Measurement noise covariance (tune this value)
        self.kf.measurementNoiseCov = np.array([[1]], np.float32) * 0.1

        # Initial state covariance (tune these values)
        self.kf.errorCovPost = np.eye(2, dtype=np.float32) * 1

        # Initialize state (distance = 100, velocity = 0)
        self.kf.statePost = np.array([[100], [0]], np.float32)

        self.distance = 100  # Current filtered distance
        self.detected = False  # Whether a measurement was detected

    def update(self, measurement):
        """
        Update the Kalman filter with a new distance measurement.
        """
        measured = np.array([[np.float32(measurement)]])
        self.kf.correct(measured)
        self.distance = self.kf.statePost[0, 0]
        self.detected = True

    def predict(self):
        """
        Predict the next distance using the Kalman filter.
        """
        prediction = self.kf.predict()
        self.distance = prediction[0, 0]
        self.detected = False
        return self.distance
