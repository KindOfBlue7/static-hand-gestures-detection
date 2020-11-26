# Imports

import cv2
import numpy as np
import sys
import time
import GestureDetection
from PyQt5 import QtGui
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QCoreApplication
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import (QWidget, QLabel, QApplication, QPushButton,
                             QSlider)

# Control variables

ip = "192.168.1.6:4747"
link = "http://" + ip + "/video"
font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(2)
img_w = cap.get(3)
img_h = cap.get(4)

# Defining region of interest variables

roi_size = [250, 250]
roi_pos_start_w = round(img_w / 2) - round(roi_size[0] / 2)
roi_pos_start_h = round(img_h / 2) - round(roi_size[1] / 2)
roi_pos_end_w = round(img_w / 2) + round(roi_size[0] / 2)
roi_pos_end_h = round(img_h / 2) + round(roi_size[1] / 2)

print("Width: %d\nHeight: %d" % (img_w, img_h))


class Thread(QThread):
    camera_feed_sig = pyqtSignal(QImage)

    record_end_sig = pyqtSignal(bool)

    Detection = GestureDetection.GestureDetection()
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
        print(self.Detection.threshold_min)

    def on_hsv_max_change_sig(self, threshold_max):
        self.Detection.threshold_max = np.array(threshold_max, dtype='uint8')

    def run(self):
        cap = cv2.VideoCapture(2)
        rec = -1
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if self.record_start:
                    rec = 0
                    self.record_start = False

                roi = frame[roi_pos_start_h:roi_pos_end_h,
                            roi_pos_start_w:roi_pos_end_w]

                skin_region, hsv = self.Detection.thresholding(roi)

                if not self.record_end:
                    if 0 <= rec < 500:
                        rec += 1

                    img_name = str(rec) + '.png'
                    print("Saving " + img_name + "...")
                    cv2.imwrite('training_data/test4/' + img_name, skin_region)

                    time.sleep(0.05)
                    if rec == 500:
                        print("Saving training data ended")
                        self.record_end = True
                        self.record_end_sig.emit(self.record_end)
                        rec = -1

                if self.toggle_hsv:
                    frame = cv2.rectangle(frame, (roi_pos_start_w - self.th, roi_pos_start_h - self.th),
                                          (roi_pos_end_w + self.th, roi_pos_end_h + self.th), (0, 0, 255), self.th)
                    frame[roi_pos_start_h:roi_pos_end_h, roi_pos_start_w:roi_pos_end_w] = hsv

                h, w, ch = frame.shape
                bytesPerLine = ch * w

                convertToQtFormat = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.camera_feed_sig.emit(p)

        cap.release()


class App(QWidget):
    toggle_hsv_sig = pyqtSignal(bool)
    record_start_sig = pyqtSignal(bool)
    detect_start_sig = pyqtSignal(bool)
    threshold_min_sig = pyqtSignal(list)
    threshold_max_sig = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.title = 'Hand Gestures Detection'
        self.left = 100
        self.top = 100
        self.width = 640
        self.height = 480
        self.record_end = True
        self.lower = [50, 39, 0]
        self.upper = [255, 255, 255]

        self.camera_feed = QLabel(self)

        self.hsv_label = QLabel(self)

        self.button1 = QPushButton('Detect Hand Gesture', self)
        self.button2 = QPushButton('Record A Gesture', self)
        self.button3 = QPushButton('Toggle HSV for ROI', self)

        self.init_UI()

    def record_end_sig(self, record_end):
        self.record_end = True if record_end else False
        print("Record_end singal status: " + str(self.record_end))

    @pyqtSlot(QImage)
    def set_image(self, image):
        self.camera_feed.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot()
    def button1_on_click(self):
        if self.button1.isChecked():
            self.detect_start_sig.emit(True)
            pass
        else:
            self.detect_start_sig.emit(False)
            pass

    @pyqtSlot()
    def button2_on_click(self):
        print("Button 2 pressed")
        self.record_end = False
        self.record_start_sig.emit(True)

    @pyqtSlot()
    def button3_on_click(self):
        if self.button3.isChecked():
            self.toggle_hsv_sig.emit(True)
        else:
            self.toggle_hsv_sig.emit(False)

    def slider1_on_change(self, value):
        self.lower[0] = value
        self.threshold_min_sig.emit(self.lower)
        self.update_hsv_values()

    def slider2_on_change(self, value):
        self.lower[1] = value
        self.threshold_min_sig.emit(self.lower)
        self.update_hsv_values()

    def slider3_on_change(self, value):
        self.lower[2] = value
        self.threshold_min_sig.emit(self.lower)
        self.update_hsv_values()

    def update_hsv_values(self):
        text = ' Thresholding\n\n'
        text += 'Min'.ljust(18) + 'Max\n'
        text += f'H: {self.lower[0]:<15}H: {self.upper[0]}\n'
        text += f'S: {self.lower[1]:<15}S: {self.upper[1]}\n'
        text += f'V: {self.lower[2]:<15}V: {self.upper[2]}\n'
        self.hsv_label.setText(text)

    def init_UI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setWindowIcon(QIcon('icon.png'))
        self.setFixedWidth(840)
        self.setFixedHeight(360)

        # create a camera feed label

        self.camera_feed.move(0, 0)
        self.camera_feed.resize(640, 360)

        # create a HSV range control label

        self.hsv_label.setGeometry(660, 230, 180, 130)
        self.update_hsv_values()
        self.update_hsv_values()
        self.hsv_label.setFont(QtGui.QFont("Arial", 9, QtGui.QFont.Bold))

        # create buttons

        self.button1.move(660, 20)
        self.button2.move(660, 50)
        self.button3.move(660, 80)
        self.button1.setCheckable(True)
        self.button2.setCheckable(True)
        self.button3.setCheckable(True)
        self.button1.clicked.connect(self.button1_on_click)
        self.button2.clicked.connect(self.button2_on_click)
        self.button3.clicked.connect(self.button3_on_click)

        # create sliders for HSV values        
        slider1 = QSlider(Qt.Horizontal, self)
        slider1.setFocusPolicy(Qt.NoFocus)
        slider1.setTickPosition(QSlider.TicksBelow)
        slider1.setTickInterval(1)
        slider1.setRange(0, 255)
        slider1.move(660, 140)
        slider1.setFixedWidth(150)
        slider1.setValue(self.lower[0])
        slider1.valueChanged.connect(self.slider1_on_change)

        slider2 = QSlider(Qt.Horizontal, self)
        slider2.setFocusPolicy(Qt.NoFocus)
        slider2.setTickPosition(QSlider.TicksBelow)
        slider2.setTickInterval(1)
        slider2.setRange(0, 255)
        slider2.move(660, 170)
        slider2.setFixedWidth(150)
        slider2.setValue(self.lower[1])
        slider2.valueChanged.connect(self.slider2_on_change)

        slider3 = QSlider(Qt.Horizontal, self)
        slider3.setFocusPolicy(Qt.NoFocus)
        slider3.setTickPosition(QSlider.TicksBelow)
        slider3.setTickInterval(1)
        slider3.setRange(0, 255)
        slider3.move(660, 200)
        slider3.setFixedWidth(150)
        slider3.setValue(self.lower[2])
        slider3.valueChanged.connect(self.slider3_on_change)

        th = Thread(self)
        th.camera_feed_sig.connect(self.set_image)
        th.record_end_sig.connect(self.record_end_sig)
        self.toggle_hsv_sig.connect(th.on_toggle_hsv_sig)
        self.record_start_sig.connect(th.on_record_start_sig)
        self.detect_start_sig.connect(th.on_detect_start_sig)
        self.threshold_min_sig.connect(th.on_hsv_min_change_sig)
        self.threshold_max_sig.connect(th.on_hsv_max_change_sig)

        th.start()

        self.show()


if __name__ == '__main__':
    app = QCoreApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
