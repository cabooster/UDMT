
import os
import sys

import cv2
from PySide6 import QtWidgets
from PySide6.QtWidgets import QWidget, QGridLayout, QSpacerItem, QSizePolicy
# from PyQt5.QtWidgets import QWidget
from PySide6.QtCore import Qt
from udmt.gui.tabs.xmem.interactive_demo import mask_seg_winclass
# from udmt.gui.widgets import launch_napari
from udmt.gui.widgets import ConfigEditor
from udmt.gui.components import (
    DefaultTab,
    ShuffleSpinBox,
    VideoSelectionWidget,
    _create_grid_layout,
    _create_horizontal_layout,
    _create_label_widget, LogWidget, TqdmLogger
)

# def label_frames(config_path):
#     _ = launch_napari(config_path)
#
#
# refine_labels = label_frames

from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QHBoxLayout,QDialog, QLabel, QProgressBar
from PySide6.QtCore import Qt


def get_video_fps(video_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Unable to open video: {video_path}")
        return None

    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()

    return fps
class LoadingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Set up dialog properties
        self.setWindowTitle("Loading")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)  # Hide help button
        self.setModal(True)
        self.setFixedSize(300, 100)

        # Create layout
        layout = QVBoxLayout(self)

        # Create and add label
        self.label = QLabel("Extracting images, please wait...")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)


        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 0) # set to uncertain mode
        layout.addWidget(self.progress_bar)
class TrackingInitialization(DefaultTab):
    def __init__(self, root, parent, h1_description):
        super(TrackingInitialization, self).__init__(root, parent, h1_description)

        self._set_page()

    @property
    def files(self):
        return self.video_selection_widget.files

    def _set_page(self):
        self.main_layout.addWidget(_create_label_widget("Video Selection", "font:bold"))
        self.video_selection_widget = VideoSelectionWidget(self.root, self)
        self.main_layout.addWidget(self.video_selection_widget)

        self.container_layout = _create_horizontal_layout(margins=(0, 0, 0, 0))


        self.main_layout.addLayout(self.container_layout)

        self.mask_seg_btn = QtWidgets.QPushButton("Step 2. Launch Tracking Initialization GUI")
        self.mask_seg_btn.setMinimumWidth(150)
        self.mask_seg_btn.clicked.connect(self.mask_seg)
        self.main_layout.addWidget(self.mask_seg_btn, alignment=Qt.AlignRight)
        ##############################
        spacer = QSpacerItem(150, 400, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.main_layout.addItem(spacer)
        self.log_widget = LogWidget(self.root, self)
        # self.main_layout.addStretch()
        self.main_layout.addWidget(self.log_widget)
        # self.tqdm_logger = TqdmLogger(self.log_widget.text_log)
        # sys.stdout = self.tqdm_logger

        # self.new_window = None


    # def log_color_by_option(self, choice):
    #     self.root.logger.info(f"Labeled images will by colored by {choice.upper()}")

    def mask_seg(self):

        try:
            video = list(self.files)[0]
            loading_dialog = LoadingDialog()
            loading_dialog.show()
            QApplication.processEvents()
            video_fps = get_video_fps(video)
            if video_fps > 60:
                divide_num = 2
            else:
                divide_num = 1
            self.mask_seg_window = mask_seg_winclass(video, self.root.project_folder, divide_num)
            self.mask_seg_window.show()
            loading_dialog.close()
        except IndexError:
            msg_box = QtWidgets.QMessageBox(self)
            msg_box.setIcon(QtWidgets.QMessageBox.Warning)
            msg_box.setWindowTitle("Error")
            msg_box.setText("No video selected. Please select a video first.")
            msg_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg_box.exec()


