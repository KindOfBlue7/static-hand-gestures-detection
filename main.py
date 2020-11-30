# Imports

import sys
import DataProcessing
from PyQt5 import QtGui
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QCoreApplication
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import (QWidget, QLabel, QApplication, QPushButton,
                             QSlider)

# Control variables
ip = "192.168.1.6:4747"
link = "http://" + ip + "/video"


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
        self.roi_bin = QLabel(self)
        self.hsv_label = QLabel(self)

        self.button1 = QPushButton('Detect Hand Gesture', self)
        self.button2 = QPushButton('Record A Gesture', self)
        self.button3 = QPushButton('Toggle HSV for ROI', self)
        self.button4 = QPushButton('HSV min', self)
        self.button5 = QPushButton('HSV max', self)

        self.threshold_min_sliders = [self.create_hsv_slider() for _ in range(3)]
        self.threshold_max_sliders = [self.create_hsv_slider() for _ in range(3)]

        self.init_UI()

    def on_record_end_sig(self, record_end):
        self.record_end = True if record_end else False
        self.button2.setChecked(False)
        print("Record_end signal status: " + str(self.record_end))

    def on_gesture_detected_sig(self, gesture):
        print(gesture)

    @pyqtSlot(QImage)
    def set_camera_feed(self, image):
        self.camera_feed.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(QImage)
    def set_roi_bin(self, image):
        self.roi_bin.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot()
    def button1_on_click(self):
        if self.button1.isChecked():
            self.detect_start_sig.emit(True)
            self.roi_bin.show()
        else:
            self.detect_start_sig.emit(False)
            self.roi_bin.hide()

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

    @pyqtSlot()
    def button4_on_click(self):
        if not self.button4.isChecked() and self.threshold_min_sliders[0].isVisible():
            self.button4.setChecked(True)

        if self.button4.isChecked() and not self.threshold_min_sliders[0].isVisible():
            self.button5.setChecked(False)
            for slider in self.threshold_max_sliders:
                slider.hide()
            for slider in self.threshold_min_sliders:
                slider.show()

    @pyqtSlot()
    def button5_on_click(self):
        if not self.button5.isChecked() and self.threshold_max_sliders[0].isVisible():
            self.button5.setChecked(True)

        if self.button5.isChecked() and not self.threshold_max_sliders[0].isVisible():
            self.button4.setChecked(False)
            for slider in self.threshold_min_sliders:
                slider.hide()
            for slider in self.threshold_max_sliders:
                slider.show()

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

    def slider4_on_change(self, value):
        self.upper[0] = value
        self.threshold_max_sig.emit(self.upper)
        self.update_hsv_values()

    def slider5_on_change(self, value):
        self.upper[1] = value
        self.threshold_max_sig.emit(self.upper)
        self.update_hsv_values()

    def slider6_on_change(self, value):
        self.upper[2] = value
        self.threshold_max_sig.emit(self.upper)
        self.update_hsv_values()

    def update_hsv_values(self):
        text = ' Thresholding\n\n'
        text += 'Min'.ljust(18) + 'Max\n'
        text += f'H: {self.lower[0]:<15}H: {self.upper[0]}\n'
        text += f'S: {self.lower[1]:<15}S: {self.upper[1]}\n'
        text += f'V: {self.lower[2]:<15}V: {self.upper[2]}\n'
        self.hsv_label.setText(text)

    def create_hsv_slider(self) -> QSlider:
        slider = QSlider(Qt.Horizontal, self)
        slider.setFocusPolicy(Qt.NoFocus)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(1)
        slider.setRange(0, 255)
        slider.setFixedWidth(150)
        slider.hide()
        return slider

    def init_UI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setWindowIcon(QIcon('icon.png'))
        self.setFixedWidth(840)
        self.setFixedHeight(360)

        # camera feed label

        self.camera_feed.move(0, 0)
        self.camera_feed.resize(640, 360)

        # roi_bin label
        self.roi_bin.move(640-64, 360-64)
        self.roi_bin.resize(64, 64)
        # self.roi_bin.hide()

        # create a HSV range control label

        self.hsv_label.setGeometry(660, 230, 180, 130)
        self.update_hsv_values()
        self.hsv_label.setFont(QtGui.QFont("Arial", 9, QtGui.QFont.Bold))

        # create buttons

        self.button1.move(660, 20)
        self.button2.move(660, 50)
        self.button3.move(660, 80)
        self.button4.move(660, 110)
        self.button5.move(740, 110)
        self.button1.setCheckable(True)
        self.button2.setCheckable(True)
        self.button3.setCheckable(True)
        self.button4.setCheckable(True)
        self.button5.setCheckable(True)
        self.button1.clicked.connect(self.button1_on_click)
        self.button2.clicked.connect(self.button2_on_click)
        self.button3.clicked.connect(self.button3_on_click)
        self.button4.clicked.connect(self.button4_on_click)
        self.button5.clicked.connect(self.button5_on_click)

        # init sliders for HSV values

        self.threshold_min_sliders[0].move(660, 140)
        self.threshold_min_sliders[0].setValue(self.lower[0])
        self.threshold_min_sliders[0].valueChanged.connect(self.slider1_on_change)

        self.threshold_min_sliders[1].move(660, 170)
        self.threshold_min_sliders[1].setValue(self.lower[1])
        self.threshold_min_sliders[1].valueChanged.connect(self.slider2_on_change)

        self.threshold_min_sliders[2].move(660, 200)
        self.threshold_min_sliders[2].setValue(self.lower[2])
        self.threshold_min_sliders[2].valueChanged.connect(self.slider3_on_change)

        self.threshold_max_sliders[0].move(660, 140)
        self.threshold_max_sliders[0].setValue(self.upper[0])
        self.threshold_max_sliders[0].valueChanged.connect(self.slider4_on_change)

        self.threshold_max_sliders[1].move(660, 170)
        self.threshold_max_sliders[1].setValue(self.upper[1])
        self.threshold_max_sliders[1].valueChanged.connect(self.slider5_on_change)

        self.threshold_max_sliders[2].move(660, 200)
        self.threshold_max_sliders[2].setValue(self.upper[2])
        self.threshold_max_sliders[2].valueChanged.connect(self.slider6_on_change)

        th = DataProcessing.DataProcessing(self)
        th.camera_feed_sig.connect(self.set_camera_feed)
        th.roi_bin_sig.connect(self.set_roi_bin)
        th.record_end_sig.connect(self.on_record_end_sig)
        th.gesture_detected_sig.connect(self.on_gesture_detected_sig)
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
