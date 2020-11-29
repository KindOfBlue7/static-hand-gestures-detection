import cv2
import numpy as np
import json
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils.vis_utils import plot_model


class GestureDetection:
    def __init__(self):
        with open('model_parameters.json') as model_parameters:
            self.model_parameters = json.load(model_parameters)

        # import from JSON file
        self.threshold_min = np.array(self.model_parameters['threshold_min'], dtype="uint8")
        self.threshold_max = np.array(self.model_parameters['threshold_max'], dtype="uint8")
        self.model_input_dim = tuple(self.model_parameters['model_input_dim'])

        self.threshold_kernel = np.ones((5, 5), np.uint8)

        self.model = None

    def thresholding(self, roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Skin detection algorithm
        roi_bin = cv2.resize(hsv, self.model_input_dim, interpolation=cv2.INTER_AREA)
        roi_bin = cv2.inRange(roi_bin, self.threshold_min, self.threshold_max)
        roi_bin = cv2.morphologyEx(roi_bin, cv2.MORPH_CLOSE, self.threshold_kernel)

        return [roi_bin, hsv]

    @staticmethod
    def create_a_model():
        cnn_model = Sequential()

        # Convolution
        cnn_model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 1), activation='relu'))

        # Pooling
        cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

        # Adding a second convolutional layer
        cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
        cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

        # Adding a third convolutional layer
        cnn_model.add(Conv2D(128, (3, 3), activation='relu'))
        cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

        # Adding a fourth convolutional layer
        cnn_model.add(Conv2D(128, (3, 3), activation='relu'))
        cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

        # Step 3 - Flattening
        cnn_model.add(Flatten())

        # Step 4 - Full connection
        cnn_model.add(Dense(units=64, activation='relu'))
        cnn_model.add(Dense(units=1, activation='sigmoid'))

        # Compiling the CNN
        cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return cnn_model

    def predict(self, roi_bin):
        pass
