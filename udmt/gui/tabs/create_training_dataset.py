
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMessageBox
from udmt.gui import BASE_DIR
from PySide6 import QtGui

from udmt.gui.tabs.shared_state import shared_state
from udmt.gui.components import (
    DefaultTab,
    ShuffleSpinBox,
    _create_grid_layout,
    _create_label_widget,
    VideoSelectionWidget
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
        # self.test_spin = QtWidgets.QDoubleSpinBox()
        # self.test_spin.setMinimum(0.0)
        # self.test_spin.setMaximum(1.0)
        # self.test_spin.setSingleStep(0.1)
        # self.test_spin.setValue(shared_state.get_test_spin())  # 初始化为共享值
        # self.test_spin.setFixedSize(80, 35)
        # # 当控件值改变时，更新共享值
        # self.test_spin.valueChanged.connect(self.on_test_spin_changed)
        # # 监听共享状态的更新
        # shared_state.test_spin_changed.connect(self.update_test_spin)
        # horizontal_layout.addWidget(self.test_spin)
        #######################################
        resize_label = _create_label_widget("Resize coefficient")
        # resize_label = QtWidgets.QLabel("Resize Coefficient")
        horizontal_layout.addWidget(resize_label)
        # 创建 QDoubleSpinBox 控件
        self.resize_spin = QtWidgets.QDoubleSpinBox()
        self.resize_spin.setMinimum(0.0)  # 最小值
        self.resize_spin.setMaximum(1.0)  # 最大值
        self.resize_spin.setSingleStep(0.1)  # 步进值
        self.resize_spin.setValue(0.8)  # 默认值
        self.resize_spin.setFixedSize(80, 35)  # 限制控件大小
        self.resize_spin.valueChanged.connect(self.log_resize_coefficient)  # 连接信号到槽函数
        horizontal_layout.addWidget(self.resize_spin)


        downsample_label = QtWidgets.QLabel("Downsample factor")
        horizontal_layout.addWidget(downsample_label)

        self.downsample_spin = QtWidgets.QSpinBox()
        self.downsample_spin.setMinimum(1)  # 最小值
        self.downsample_spin.setMaximum(10)  # 最大值
        self.downsample_spin.setSingleStep(1)  # 步进值
        self.downsample_spin.setValue(1)  # 默认值
        self.downsample_spin.setFixedSize(80, 35)  # 限制控件大小
        self.downsample_spin.valueChanged.connect(self.log_downsample_rate)  # 连接信号到槽函数
        horizontal_layout.addWidget(self.downsample_spin)
        ##############################
        self.main_layout.addLayout(horizontal_layout)


        # self.model_comparison = False
        self.label_frames_btn = QtWidgets.QPushButton("Step 2. Create Training Dataset")
        self.label_frames_btn.setMinimumWidth(150)
        self.label_frames_btn.clicked.connect(self.create_training_dataset)
        self.main_layout.addWidget(self.label_frames_btn, alignment=Qt.AlignRight)

        # 添加控制显示的按钮
        self.toggle_videos_btn = QtWidgets.QPushButton("Show Videos")
        self.toggle_videos_btn.setCheckable(True)  # 按钮可切换
        self.toggle_videos_btn.setChecked(False)  # 默认不显示
        # self.toggle_videos_btn.setMaximumWidth(120)  # 设置最大宽度（可选）
        self.toggle_videos_btn.setFixedSize(120, 40)
        self.toggle_videos_btn.toggled.connect(self.toggle_video_display)

        self.main_layout.addWidget(self.toggle_videos_btn, alignment=Qt.AlignRight)

        # 创建视频显示容器
        self.video_container = QtWidgets.QWidget()
        video_layout = QtWidgets.QHBoxLayout(self.video_container)
        video_layout.setSpacing(10)  # 两个区域之间的间隔

        # 创建两个视频区域
        self.video_display_label1 = self.create_video_section("Forward Tracking Visualization")
        self.video_display_label2 = self.create_video_section("Backward Tracking Visualization")

        video_layout.addWidget(self.video_display_label1["container"])
        video_layout.addWidget(self.video_display_label2["container"])

        # 默认隐藏视频容器
        self.video_container.setVisible(False)
        self.main_layout.addWidget(self.video_container, alignment=Qt.AlignCenter)


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
        video_label.setFixedSize(450, 450)
        video_label.setStyleSheet("""
                                        QLabel {
                                            border: 2px solid #00bcd4;  /* 边框颜色 */
                                            border-radius: 5px;        /* 圆角半径 */
                                            background-color: #ffffff;   /* 背景色 */
                                        }
                                    """)

        # 将标题和视频添加到垂直布局中
        container_layout = QtWidgets.QVBoxLayout()
        container_layout.addWidget(title_label)
        container_layout.addWidget(video_label)
        container_layout.setSpacing(3)  # 标题与视频之间的间距
        container_layout.setAlignment(Qt.AlignTop)

        # 创建容器小部件
        container_widget = QtWidgets.QWidget()
        container_widget.setLayout(container_layout)

        return {"container": container_widget, "video_label": video_label}

    def toggle_video_display(self, checked):
        """控制视频显示和隐藏"""
        if checked:
            self.video_container.setVisible(True)
            self.toggle_videos_btn.setText("Hide Videos")
        else:
            self.video_container.setVisible(False)
            self.toggle_videos_btn.setText("Show Videos")

    def on_test_spin_changed(self, value):
        """当 QDoubleSpinBox 的值改变时更新共享状态"""
        shared_state.set_test_spin(value)
        print('cre class:', value)

    def update_test_spin(self, value):
        """当共享状态更新时同步控件的值"""
        if self.test_spin.value() != value:
            self.test_spin.setValue(value)
    @property
    def files(self):
        return self.video_selection_widget.files
    def get_reize_coff(self):
        return self.resize_spin.value()

    def extract_frames(self, video_name, output_dir, scale_percent, downsample_rate):
        # 打开视频文件
        cap = cv2.VideoCapture(video_name)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_name}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("The selected video was recorded with ", round(fps, 2), "fps")
        success, first_image = cap.read()
        width = int(first_image.shape[1] * scale_percent)
        height = int(first_image.shape[0] * scale_percent)
        dim = (width, height)
        print(f"Frames will be resized to {dim[0]}x{dim[1]} (width x height) based on the scale percentage of {scale_percent * 100:.0f}%.")
        frame_count = 0
        saved_count = 0
        print('Extracting frames....Please wait...')

        while saved_count < 2000:
            ret, frame = cap.read()
            if not ret:
                print("End of video or error reading frame.")
                break

            # 按下采样率保存帧
            if frame_count % int(1 / downsample_rate) == 0:
                resized_frame = cv2.resize(frame, dim)
                file_name = f"{saved_count:05d}.jpg"
                file_path = os.path.join(output_dir, file_name)
                cv2.imwrite(file_path, resized_frame)
                saved_count += 1

            frame_count += 1

        cap.release()
        print(f"Saved {saved_count} frames to {output_dir}")

    def create_training_dataset(self):
        try:
            video_name = list(self.files)[0]
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
                # 获取路径中的文件数量
                file_list = [f for f in os.listdir(mask_path) if os.path.isfile(os.path.join(mask_path, f))]
                file_count = len(file_list)

                # 判断文件数量是否大于 2000
                if file_count > 2000:
                    # 设置保存目录
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
                                           'search_scale_range': np.arange(1.5, 2.5, 0.5),
                                           'target_sz_bias_range': [0, 0.1], #[-0.2, -0.1, 0, 0.1, 0.2]
                                           'status_flag': 1, # train_param_iter 1 test_param_iter 2 test 3
                                           'evaluation_metric': []
                                           }
                    print(run_tracking_params)
                    # run_training_process()
                    # play_video_widget = [self.video_display_label1["video_label"],self.video_display_label2["video_label"]]
                    run_tracking(run_tracking_params)
                    param_json_path = run_tracking_params['project_folder'] + '/tmp/' + run_tracking_params['video_name'] + "/evaluation_metric_for_train.json"
                    with open(param_json_path, "r") as file:
                        loaded_results_list = json.load(file)
                    best_param_path_find_str = loaded_results_list[-1]["target_sz"] + '_' + loaded_results_list[-1]["search_scale"]
                    print('best_param_path_find_str', best_param_path_find_str)
                    results_path = run_tracking_params['project_folder'] + '/tmp/' + run_tracking_params['video_name'] + '/train_set_results'
                    best_param_path = results_path + '/' + find_subfolders_with_keyword(results_path,best_param_path_find_str)
                    print('best_param_path', best_param_path)
                    create_train_label(run_tracking_params['video_name'],best_param_path,run_tracking_params['project_folder'] + '/training-datasets/' + run_tracking_params['video_name'] +'/label')
                    ######################
                    msg = QtWidgets.QMessageBox()
                    msg.setIcon(QtWidgets.QMessageBox.Information)
                    msg.setText("The training dataset is now created and ready to train.")
                    msg.setInformativeText(
                        "Move 'UDMT - Train Network' to evaluate the network."
                    )

                    msg.setWindowTitle("Info")
                    msg.setMinimumWidth(900)
                    self.logo_dir = os.path.dirname(os.path.realpath("logo.png")) + os.path.sep
                    self.logo = self.logo_dir + "/assets/logo.png"
                    msg.setWindowIcon(QIcon(self.logo))
                    msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                    msg.exec_()
                    '''
                    cap = cv2.VideoCapture(video_name)
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        # 转换帧为 Qt 支持的格式
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        display_width = play_video_widget[0].width()
                        display_height = play_video_widget[0].height()
                        resized_frame = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
                        height, width, channel = resized_frame.shape
                        bytes_per_line = 3 * width
                        qt_image = QtGui.QImage(resized_frame.data, width, height, bytes_per_line,QtGui.QImage.Format_RGB888)
                        play_video_widget[0].setPixmap(QtGui.QPixmap.fromImage(qt_image))
                        QtWidgets.QApplication.processEvents()
                    '''

                else:
                    QMessageBox.information(
                        self,
                        "Warning",
                        "The number of extracted foreground masks is below 2000, which is insufficient for training the dataset. Please return to 'UDMT - Tracking Initialization' to extract more foreground masks from the selected video!",
                        QMessageBox.Ok
                    )

        except IndexError:  # 如果 files 为空，捕获 IndexError
            msg_box = QtWidgets.QMessageBox(self)
            msg_box.setIcon(QtWidgets.QMessageBox.Warning)
            msg_box.setWindowTitle("Error")
            msg_box.setText("No video selected. Please select a video first.")
            msg_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg_box.exec()
            # loading_dialog = LoadingDialog()
            # loading_dialog.show()
            # QApplication.processEvents()
            # self.mask_seg_window = mask_seg_winclass(video, self.root.project_folder)
            # self.mask_seg_window.show()
            # loading_dialog.close()