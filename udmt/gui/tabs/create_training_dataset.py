
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMessageBox, QSpacerItem, QSizePolicy
from tqdm import tqdm

from udmt.gui import BASE_DIR
from PySide6 import QtGui

from udmt.gui.tabs.shared_state import shared_state
from udmt.gui.components import (
    DefaultTab,
    ShuffleSpinBox,
    _create_grid_layout,
    _create_label_widget,
    VideoSelectionWidget, LogWidget, TqdmLogger
)
# from pytracking.run_tracker import run_tracking
from udmt.gui.tabs.ST_Net.pytracking.run_tracker import run_tracking
from udmt.gui.tabs.tracking_initialization import get_video_fps
from udmt.gui.tabs.ST_Net.pytracking.utils.convert_to_train_set import create_train_label

import udmt
from udmt.utils.auxiliaryfunctions import (
    get_data_and_metadata_filenames,
    get_training_set_folder,
)


def find_subfolders_with_keyword(parent_folder, keyword):
    # List to store matching subfolder names
    matching_folders = []

    # Iterate through all subfolders in the parent folder
    for folder_name in os.listdir(parent_folder):
        # Build the full path to the folder
        folder_path = os.path.join(parent_folder, folder_name)

        # Check if it's a directory and contains the keyword
        if os.path.isdir(folder_path) and keyword in folder_name:
            matching_folders.append(folder_name)
    return matching_folders[0]
class CreateTrainingDataset(DefaultTab):
    def __init__(self, root, parent, h1_description):
        super(CreateTrainingDataset, self).__init__(root, parent, h1_description)
        ##############################
        self.main_layout.addWidget(_create_label_widget("Video Selection", "font:bold"))
        self.video_selection_widget = VideoSelectionWidget(self.root, self)
        self.main_layout.addWidget(self.video_selection_widget)
        ##############################
        self.main_layout.addWidget(_create_label_widget("Attributes", "font:bold"))
        ##############################
        horizontal_layout = QtWidgets.QHBoxLayout()
        horizontal_layout.setAlignment(Qt.AlignLeft)
        horizontal_layout.setSpacing(20)

        #######################################
        resize_label = _create_label_widget("Resize coefficient")
        horizontal_layout.addWidget(resize_label)

        self.resize_spin = QtWidgets.QDoubleSpinBox()
        self.resize_spin.setMinimum(0.0)
        self.resize_spin.setMaximum(1.0)
        self.resize_spin.setSingleStep(0.1)
        self.resize_spin.setValue(0.8)
        self.resize_spin.setFixedSize(80, 35)
        self.resize_spin.valueChanged.connect(self.log_resize_coefficient)
        self.resize_spin.setToolTip(
            "Set the resize coefficient to scale the original video, used to improve processing speed when the resolution is high.")
        horizontal_layout.addWidget(self.resize_spin)


        downsample_label = QtWidgets.QLabel("Downsample factor")
        horizontal_layout.addWidget(downsample_label)

        self.downsample_spin = QtWidgets.QSpinBox()
        self.downsample_spin.setMinimum(1)
        self.downsample_spin.setMaximum(10)
        self.downsample_spin.setSingleStep(1)
        self.downsample_spin.setValue(1)
        self.downsample_spin.setFixedSize(80, 35)
        self.downsample_spin.valueChanged.connect(self.log_downsample_rate)
        self.downsample_spin.setToolTip(
            "Set the downsample factor to reduce the frame rate of the original video, used to improve processing speed when the frame rate is high.")
        horizontal_layout.addWidget(self.downsample_spin)
        ##############################
        self.main_layout.addLayout(horizontal_layout)


        self.root.downsample_value_.connect(self.sync_downsample_value)
        self.root.resize_value_.connect(self.sync_resize_value)

        self.downsample_spin.valueChanged.connect(self.emit_downsample_value)
        self.resize_spin.valueChanged.connect(self.emit_resize_value)


        # self.model_comparison = False
        self.label_frames_btn = QtWidgets.QPushButton("Step 2. Create Training Dataset")
        self.label_frames_btn.setMinimumWidth(150)
        self.label_frames_btn.clicked.connect(self.create_training_dataset)
        self.label_frames_btn.setToolTip("Click to start creating the training dataset.")
        self.main_layout.addWidget(self.label_frames_btn, alignment=Qt.AlignRight)


        self.toggle_videos_btn = QtWidgets.QPushButton("Show Videos")
        self.toggle_videos_btn.setCheckable(True)
        self.toggle_videos_btn.setChecked(False)
        # self.toggle_videos_btn.setMaximumWidth(120)
        self.toggle_videos_btn.setFixedSize(120, 40)
        self.toggle_videos_btn.toggled.connect(self.toggle_video_display)
        self.toggle_videos_btn.setToolTip("Click to expand the visualization window for creating the training dataset.")

        self.main_layout.addWidget(self.toggle_videos_btn, alignment=Qt.AlignRight)


        self.video_container = QtWidgets.QWidget()
        video_layout = QtWidgets.QHBoxLayout(self.video_container)
        video_layout.setSpacing(10)


        self.video_display_label1 = self.create_video_section("Forward Tracking Visualization")
        self.video_display_label2 = self.create_video_section("Backward Tracking Visualization")

        video_layout.addWidget(self.video_display_label1["container"])
        video_layout.addWidget(self.video_display_label2["container"])


        self.video_container.setVisible(False)
        self.main_layout.addWidget(self.video_container, alignment=Qt.AlignCenter)
        spacer = QSpacerItem(150, 400, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.main_layout.addItem(spacer)
        self.log_widget = LogWidget(self.root, self)
        # self.main_layout.addStretch()
        self.main_layout.addWidget(self.log_widget)
        ####################
        # self.tqdm_logger = TqdmLogger(self.log_widget.text_log)
        # sys.stdout = self.tqdm_logger

    def emit_downsample_value(self, value):

        self.root.downsample_value_.emit(value)

    def emit_resize_value(self, value):

        self.root.resize_value_.emit(value)

    def sync_downsample_value(self, value):

        if self.downsample_spin.value() != value:
            self.downsample_spin.blockSignals(True)
            self.downsample_spin.setValue(value)
            self.downsample_spin.blockSignals(False)

    def sync_resize_value(self, value):
        if self.resize_spin.value() != value:
            self.resize_spin.blockSignals(True)
            self.resize_spin.setValue(value)
            self.resize_spin.blockSignals(False)
    def log_resize_coefficient(self, value):
        print(f"Resize coefficient adjusted to: {value:.1f}")

    def log_downsample_rate(self, value):
        print(f"Downsample rate adjusted to: {value:.1f}")
    def create_video_section(self, title):
        title_label = QtWidgets.QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 14px; font-weight: bold;color: #00bcd4;")

        video_label = QtWidgets.QLabel()
        video_label.setAlignment(Qt.AlignCenter)
        video_label.setFixedSize(350, 350)
        video_label.setStyleSheet("""
                                        QLabel {
                                            border: 2px solid #00bcd4;  
                                            border-radius: 5px;        
                                            background-color: #ffffff;  
                                        }
                                    """)

        container_layout = QtWidgets.QVBoxLayout()
        container_layout.addWidget(title_label)
        container_layout.addWidget(video_label)
        container_layout.setSpacing(3)  # The space between the title and the video
        container_layout.setAlignment(Qt.AlignTop)


        container_widget = QtWidgets.QWidget()
        container_widget.setLayout(container_layout)

        return {"container": container_widget, "video_label": video_label}

    def toggle_video_display(self, checked):
        if checked:
            self.video_container.setVisible(True)
            self.toggle_videos_btn.setText("Hide Videos")
        else:
            self.video_container.setVisible(False)
            self.toggle_videos_btn.setText("Show Videos")

    @property
    def files(self):
        return self.video_selection_widget.files
    def get_reize_coff(self):
        return self.resize_spin.value()

    def extract_frames(self, video_name, output_dir, scale_percent, downsample_rate):
        cap = cv2.VideoCapture(video_name)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_name}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f'The selected video was recorded with {round(fps, 2)} fps')
        success, first_image = cap.read()
        width = int(first_image.shape[1] * scale_percent)
        height = int(first_image.shape[0] * scale_percent)
        dim = (width, height)
        print(f"Frames will be resized to {dim[0]}x{dim[1]} (width x height) based on the scale percentage of {scale_percent * 100:.0f}%.")
        frame_count = 0
        saved_count = 0
        print('Extracting frames....Please wait...')
        with tqdm(total=2000, desc="Extracting frames", unit="frame") as pbar:
            while saved_count < 2000:
                ret, frame = cap.read()
                if not ret:
                    print("End of video or error reading frame.")
                    break


                if frame_count % int(1 / downsample_rate) == 0:
                    resized_frame = cv2.resize(frame, dim)
                    file_name = f"{saved_count:05d}.jpg"
                    file_path = os.path.join(output_dir, file_name)
                    cv2.imwrite(file_path, resized_frame)
                    saved_count += 1
                    pbar.update(1)

                frame_count += 1

        cap.release()
        print(f"Saved {saved_count} frames to {os.path.normpath(output_dir)}")

    def create_training_dataset(self):
        try:
            video_name = list(self.files)[0]
        except IndexError:
            msg_box = QtWidgets.QMessageBox(self)
            msg_box.setIcon(QtWidgets.QMessageBox.Warning)
            msg_box.setWindowTitle("Error")
            msg_box.setText("No video selected. Please select a video first.")
            msg_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg_box.exec()
        try:
            frame_rate = get_video_fps(video_name)
            file_path = Path(video_name)
            file_name_without_extension = file_path.stem
            mask_path = self.root.project_folder + '/tmp/' + file_name_without_extension + '/masks/'
            if not os.path.exists(mask_path):
                QMessageBox.information(
                    self,
                    "Warning",
                    "Please return to 'UDMT - Tracking Initialization' to extract foreground masks from the selected video!",
                    QMessageBox.Ok
                )
            else:
                file_list = [f for f in os.listdir(mask_path) if os.path.isfile(os.path.join(mask_path, f))]
                file_count = len(file_list)


                if file_count > 2000:#2000
                    extract_output_dir = os.path.join(self.root.project_folder, 'training-datasets',
                                              file_name_without_extension, 'img')
                    if os.path.exists(extract_output_dir):
                        existing_files = [f for f in os.listdir(extract_output_dir) if f.endswith('.jpg')]
                        if len(existing_files) < 2000:
                            self.extract_frames(video_name,extract_output_dir,self.resize_spin.value(),self.downsample_spin.value())
                    else:
                        os.makedirs(extract_output_dir)
                        self.extract_frames(video_name, extract_output_dir,self.resize_spin.value(),self.downsample_spin.value())

                    run_tracking_params = {'video_display_widget1': self.video_display_label1["video_label"],
                                           'video_display_widget2': self.video_display_label2["video_label"],
                                           'project_folder': self.root.project_folder,
                                           'video_name':file_name_without_extension,
                                           'model_path': BASE_DIR + '/pretrained/trdimp_net_ep.pth.tar',
                                           'resize_factor':self.resize_spin.value(),
                                           'downsample_factor':self.downsample_spin.value(),
                                           'frame_rate': frame_rate,
                                           'frame_num': 2000,
                                           'search_scale_range': np.arange(1.5, 3, 0.5),# 1.5, 3, 0.5
                                           'target_sz_bias_range': [-0.1, 0, 0.1], # [-0.1, 0, 0.1] [-0.2, -0.1, 0, 0.1, 0.2]
                                           'status_flag': 1, # train_param_iter 1 test_param_iter 2 test 3
                                           'evaluation_metric': [],
                                           'is_concave': self.root.cfg['is_concave']
                                           }
                    print('run_tracking_params:',run_tracking_params)
                    # run_training_process()
                    # play_video_widget = [self.video_display_label1["video_label"],self.video_display_label2["video_label"]]
                    run_tracking(run_tracking_params)
                    param_json_path = run_tracking_params['project_folder'] + '/tmp/' + run_tracking_params['video_name'] + "/evaluation_metric_for_train.json"
                    with open(param_json_path, "r") as file:
                        loaded_results_list = json.load(file)
                    best_param_path_find_str = loaded_results_list[-1]["target_sz"] + '_' + loaded_results_list[-1]["search_scale"]
                    # print('best_param_path_find_str', best_param_path_find_str)
                    results_path = run_tracking_params['project_folder'] + '/tmp/' + run_tracking_params['video_name'] + '/train_set_results'
                    best_param_path = results_path + '/' + find_subfolders_with_keyword(results_path,best_param_path_find_str)
                    print(f'Converting results in {best_param_path}...')
                    create_train_label(run_tracking_params['video_name'],best_param_path,run_tracking_params['project_folder'] + '/training-datasets/' + run_tracking_params['video_name'] +'/label')
                    ######################
                    msg = QtWidgets.QMessageBox()
                    msg.setIcon(QtWidgets.QMessageBox.Information)
                    msg.setText("The training dataset is now created and ready to train.")
                    msg.setInformativeText(
                        "Move 'UDMT - Train Network' for training."
                    )

                    msg.setWindowTitle("Info")
                    msg.setMinimumWidth(900)
                    self.logo_dir = os.path.dirname(os.path.realpath("logo.png")) + os.path.sep
                    self.logo = self.logo_dir + "/assets/logo.png"
                    msg.setWindowIcon(QIcon(self.logo))
                    msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                    msg.exec_()

                else:
                    QMessageBox.information(
                        self,
                        "Warning",
                        "The number of extracted foreground masks is below 2000, which is insufficient for training the dataset. Please return to 'UDMT - Tracking Initialization' to extract more foreground masks from the selected video!",
                        QMessageBox.Ok
                    )
        except Exception as e:
            import traceback

            traceback.print_exc()


            tb_str = traceback.format_exc()


            msg_box = QtWidgets.QMessageBox(self)
            msg_box.setIcon(QtWidgets.QMessageBox.Critical)
            msg_box.setWindowTitle("Unexpected Error")
            msg_box.setText("An unexpected error occurred:\n\n" + tb_str)
            msg_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg_box.exec()

