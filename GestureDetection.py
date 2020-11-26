import cv2
import numpy as np
import json


class GestureDetection:
    def __init__(self):
        with open('model_parameters.json') as model_parameters:
            self.model_parameters = json.load(model_parameters)

        # import from JSON file
        self.threshold_min = np.array(self.model_parameters['threshold_min'], dtype="uint8")
        self.threshold_max = np.array(self.model_parameters['threshold_max'], dtype="uint8")
        self.model_input_dim = tuple(self.model_parameters['model_input_dim'])

        self.threshold_kernel = np.ones((10, 10), np.uint8)

        self.model = None

    def thresholding(self, roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Skin detection algorithm
        roi_bin = cv2.inRange(hsv, self.threshold_min, self.threshold_max)
        roi_bin = cv2.morphologyEx(roi_bin, cv2.MORPH_CLOSE, self.threshold_kernel)
        roi_bin = cv2.resize(roi_bin, self.model_input_dim, interpolation=cv2.INTER_AREA)

        return [roi_bin, hsv]

    def predict(self, roi_bin):
        pass
