# Imports

import GestureDetection
import cv2
import numpy as np
import time
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage


class DataProcessing(QThread):
    camera_feed_sig = pyqtSignal(QImage)
    roi_bin_sig = pyqtSignal(QImage)
    record_end_sig = pyqtSignal(bool)

    # camera feed variables
    cap = cv2.VideoCapture(2)
    img_w = cap.get(3)
    img_h = cap.get(4)

    Detection = GestureDetection.GestureDetection()

    # Defining region of interest parameters

    roi_size = Detection.model_parameters['roi_size']
    roi_pos_start_w = round(img_w / 2) - round(roi_size[0] / 2)
    roi_pos_start_h = round(img_h / 2) - round(roi_size[1] / 2)
    roi_pos_end_w = round(img_w / 2) + round(roi_size[0] / 2)
    roi_pos_end_h = round(img_h / 2) + round(roi_size[1] / 2)
    roi_pos = [roi_pos_start_w, roi_pos_start_h, roi_pos_end_w, roi_pos_end_h]

    toggle_hsv = False
    record_start = False
    record_end = True
    detect = False

    # thickness of a rectangle
    th = 2

    def on_toggle_hsv_sig(self, toggle_hsv):
        self.toggle_hsv = toggle_hsv

    def on_record_start_sig(self, toggle_record):
        if toggle_record and not self.detect:
            self.record_start = True
            self.record_end = False
            print("Record_start signal status: " + str(self.record_start))

    def on_detect_start_sig(self, detect_start):
        self.detect = detect_start

    def on_hsv_min_change_sig(self, threshold_min):
        self.Detection.threshold_min = np.array(threshold_min, dtype='uint8')

    def on_hsv_max_change_sig(self, threshold_max):
        self.Detection.threshold_max = np.array(threshold_max, dtype='uint8')

    def run(self):
        print("Width: %d\nHeight: %d" % (self.img_w, self.img_h))
        rec = -1
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                if self.record_start:
                    rec = 0
                    self.record_start = False

                roi = frame[self.roi_pos[1]:self.roi_pos[3],
                            self.roi_pos[0]:self.roi_pos[2]]

                skin_region, hsv = self.Detection.thresholding(roi)

                h, w = skin_region.shape
                ch = 1
                bytesPerLine = ch * w
                convertToQtFormat = QImage(skin_region.data, w, h, bytesPerLine, QImage.Format_Grayscale8)
                p = convertToQtFormat.scaled(64, 64)
                self.roi_bin_sig.emit(p)

                if not self.record_end:
                    if 0 <= rec < 500:
                        rec += 1

                    img_name = str(rec) + '.png'
                    print("Saving " + img_name + "...")
                    cv2.imwrite('image_data/gesture2/' + img_name, skin_region)

                    time.sleep(0.05)
                    if rec == 500:
                        print("Saving training data ended")
                        self.record_end = True
                        self.record_end_sig.emit(self.record_end)
                        rec = -1

                if self.toggle_hsv:
                    frame = cv2.rectangle(frame, (self.roi_pos[0] - self.th, self.roi_pos[1] - self.th),
                                          (self.roi_pos[2] + self.th, self.roi_pos[3] + self.th), (0, 0, 255), self.th)
                    frame[self.roi_pos[1]:self.roi_pos[3], self.roi_pos[0]:self.roi_pos[2]] = hsv

                h, w, ch = frame.shape
                bytesPerLine = ch * w

                convertToQtFormat = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.camera_feed_sig.emit(p)

        self.cap.release()
