import cv2
import numpy as np
import json
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint


class GestureDetection:
    def __init__(self):
        with open('model_parameters.json') as model_parameters:
            self.model_parameters = json.load(model_parameters)

        # import from JSON file
        self.threshold_min = np.array(self.model_parameters['threshold_min'], dtype="uint8")
        self.threshold_max = np.array(self.model_parameters['threshold_max'], dtype="uint8")
        self.model_input_dim = tuple(self.model_parameters['model_input_dim'])

        self.threshold_kernel = np.ones((5, 5), np.uint8)

        self.model = keras.models.load_model('best_model.hdf5')

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

    @staticmethod
    def train_model(model):
        batch_size = 50

        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           shear_range=0.2,
                                           zoom_range=0.2)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        training_set = train_datagen.flow_from_directory('image_data/training_data/',
                                                         target_size=(64, 64),
                                                         batch_size=batch_size,
                                                         class_mode='binary',
                                                         color_mode='grayscale')

        test_set = test_datagen.flow_from_directory('image_data/test_data/',
                                                    target_size=(64, 64),
                                                    batch_size=batch_size,
                                                    class_mode='binary',
                                                    color_mode='grayscale')

        filepath = "best_model.hdf5"

        steps_per_epoch = int(np.ceil(400 / batch_size))

        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

        history = model.fit(training_set,
                            steps_per_epoch=steps_per_epoch,
                            epochs=5,
                            validation_data=test_set,
                            validation_steps=2,
                            callbacks=[checkpoint])

    def predict(self, roi_bin):
        img = image.img_to_array(roi_bin)
        img = np.expand_dims(img, axis=0)
        result = self.model.predict(img)
        return result
