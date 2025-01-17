
import json
import os
from datetime import datetime
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMessageBox

from udmt.gui import BASE_DIR
from udmt.gui.tabs.ST_Net.pytracking.run_tracker import run_tracking
from udmt.gui.tabs.ST_Net.pytracking.utils.filter_and_save import post_process_results, create_tracking_video
from udmt.gui.tabs.shared_state import shared_state
from udmt.gui.tabs.tracking_initialization import get_video_fps
# from udmt.gui.utils import move_to_separate_thread
from udmt.gui.widgets import ConfigEditor
from udmt.gui.components import (
    DefaultTab,
    BodypartListWidget,
    ShuffleSpinBox,
    VideoSelectionWidget,
    _create_grid_layout,
    _create_label_widget,
    _create_horizontal_layout,
    _create_vertical_layout,
)

import udmt
from udmt.utils.auxiliaryfunctions import edit_config


def has_model_file(folder_path, file_extension=".pth.tar"):
    """
    Check if a folder or its subfolders contains a file with the specified extension.

    Args:
        folder_path (str): Path to the folder to check.
        file_extension (str): File extension to look for (default is ".pth.tar").

    Returns:
        bool: True if a file with the specified extension is found, otherwise False.
    """
    # Walk through all subdirectories and files
    for root, _, files in os.walk(folder_path):
        # Check each file
        for file in files:
            if file.endswith(file_extension):
                # print(f"Found file: {os.path.join(root, file)}")
                return True
    return False
def has_jpg_files(folder_path):
    """
    Check if a folder contains any .jpg files.

    Args:
        folder_path (str): Path to the folder to check.

    Returns:
        bool: True if there are .jpg files, False otherwise.
    """
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return False

    # Iterate through items in the folder
    for file in os.listdir(folder_path):
        # Check if the item is a .jpg file
        if file.lower().endswith(".jpg"):  # Case-insensitive check
            print(f"Found .jpg file: {file}")
            return True

    return False

def select_model(video_name, models_folder, model_extension=".pth.tar"):
    """
    Select the appropriate model file based on the video name and models folder.

    Args:
        video_name (str): The name of the video (e.g., "5-mice-1min").
        models_folder (str): Path to the models folder containing subfolders.
        model_extension (str): The file extension for the model files (default is ".pth.tar").

    Returns:
        str: Path to the selected model file, or None if no models are found.
    """
    # Extract the subfolder matching the video name
    target_subfolder = os.path.join(models_folder, video_name)

    # List all subfolders in the models folder
    subfolders = [os.path.join(models_folder, sf) for sf in os.listdir(models_folder) if
                  os.path.isdir(os.path.join(models_folder, sf))]

    # Function to get the last model file in a folder (by sorted file name)
    def get_last_model(folder):
        # List all model files in the folder
        model_files = [f for f in os.listdir(folder) if f.endswith(model_extension)]
        if model_files:
            # Sort model files by name and return the last one
            model_files.sort()
            return os.path.join(folder, model_files[-1])
        return None

    # Check if the target subfolder exists and has models
    if os.path.exists(target_subfolder) and target_subfolder in subfolders:
        last_model = get_last_model(target_subfolder)
        if last_model:
            print(f"Using model from target subfolder: {last_model}")
            return last_model

    # If no target subfolder or models, pick any subfolder with models
    for subfolder in subfolders:
        last_model = get_last_model(subfolder)
        if last_model:
            print(f"Using model from other subfolder: {last_model}")
            return last_model

    print("No models found in any subfolder.")
    return None
def find_latest_folder(folder_path):
    """
    This function finds the subfolder with the latest date in its name
    from the specified folder path. The date is assumed to be the
    last part of the folder name after the final underscore.

    Parameters:
    - folder_path (str): The path to the folder containing subfolders.

    Returns:
    - str: The name of the subfolder with the latest date, or None if no valid subfolder is found.
    """
    # Initialize variables to track the latest folder and its date
    latest_folder = None
    latest_date = None

    # Traverse through all subfolders in the provided folder
    for folder_name in os.listdir(folder_path):
        folder_path_full = os.path.join(folder_path, folder_name)

        # Check if it's a directory
        if os.path.isdir(folder_path_full):
            # Split the folder name by underscores and get the last part (assumed to be the date)
            parts = folder_name.split('_')
            date_str = parts[-1]  # The last part is expected to be the date string

            # Convert the date string into a datetime object
            try:
                folder_date = datetime.strptime(date_str, '%Y%m%d%H%M')

                # Update the latest folder if this folder's date is later
                if latest_date is None or folder_date > latest_date:
                    latest_date = folder_date
                    latest_folder = folder_name
            except ValueError:
                continue  # Skip if the date string cannot be parsed

    # Return the folder with the latest date, or None if no valid folder is found
    return latest_folder
class AnalyzeVideos(DefaultTab):
    def __init__(self, root, parent, h1_description):
        super(AnalyzeVideos, self).__init__(root, parent, h1_description)
        self.main_layout.addWidget(_create_label_widget("Video Selection", "font:bold"))
        self.video_selection_widget = VideoSelectionWidget(self.root, self)
        self.main_layout.addWidget(self.video_selection_widget)
        self._set_page()

    @property
    def files(self):
        return self.video_selection_widget.files

    def _set_page(self):


        # tmp_layout = _create_horizontal_layout()

        self.main_layout.addWidget(_create_label_widget("Attributes", "font:bold"))
        self.layout_attributes = _create_grid_layout(margins=(20, 0, 0, 0))
        self._generate_layout_other_options(self.layout_attributes)
        self.main_layout.addLayout(self.layout_attributes)
        # tmp_layout.addLayout(self.layout_attributes)
        #
        # # Single / Multi animal Only Layouts
        # self.layout_singleanimal = _create_horizontal_layout()
        # self.layout_multianimal = _create_horizontal_layout()
        #
        # if self.root.is_multianimal:
        #     self._generate_layout_multianimal(self.layout_multianimal)
        #     tmp_layout.addLayout(self.layout_multianimal)
        # else:
        #     self._generate_layout_single_animal(self.layout_singleanimal)
        #     tmp_layout.addLayout(self.layout_singleanimal)
        #
        # self.main_layout.addLayout(tmp_layout)
        #
        self.main_layout.addWidget(_create_label_widget(""))
        # self.layout_other_options = _create_vertical_layout()
        # self._generate_layout_other_options(self.layout_other_options)
        # self.main_layout.addLayout(self.layout_other_options)

        self.analyze_videos_btn = QtWidgets.QPushButton("Analyze Videos")
        self.analyze_videos_btn.setMinimumWidth(150)
        self.analyze_videos_btn.clicked.connect(self.analyze_videos)

        # self.edit_config_file_btn = QtWidgets.QPushButton("Edit config.yaml")
        # self.edit_config_file_btn.clicked.connect(self.edit_config_file)

        self.main_layout.addWidget(self.analyze_videos_btn, alignment=Qt.AlignRight)
        # self.main_layout.addWidget(self.edit_config_file_btn, alignment=Qt.AlignRight)
        #######################################
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

    # def _generate_layout_single_animal(self, layout):
    #     # Dynamic bodypart cropping
    #     self.crop_bodyparts = QtWidgets.QCheckBox("Dynamically crop bodyparts")
    #     self.crop_bodyparts.setCheckState(Qt.Unchecked)
    #     self.crop_bodyparts.stateChanged.connect(self.update_crop_choice)
    #
    #     self.dynamic_cropping = False
    #     layout.addWidget(self.crop_bodyparts)

    def _generate_layout_other_options(self, layout):
        #################################
        # self.test_spin = QtWidgets.QDoubleSpinBox()
        # self.test_spin.setMinimum(0.0)
        # self.test_spin.setMaximum(1.0)
        # self.test_spin.setSingleStep(0.1)
        # self.test_spin.setValue(shared_state.get_test_spin())  # 初始化为共享值
        # self.test_spin.setFixedSize(80, 35)
        #
        # # 当控件值改变时，更新共享值
        # self.test_spin.valueChanged.connect(self.on_test_spin_changed)
        #
        # # 监听共享状态的更新
        # shared_state.test_spin_changed.connect(self.update_test_spin)
        #################################

        resize_label = QtWidgets.QLabel("Resize coefficient")
        # resize_label = QtWidgets.QLabel("Resize Coefficient")

        # 创建 QDoubleSpinBox 控件
        self.resize_spin = QtWidgets.QDoubleSpinBox()
        self.resize_spin.setMinimum(0.0)  # 最小值
        self.resize_spin.setMaximum(1.0)  # 最大值
        self.resize_spin.setSingleStep(0.1)  # 步进值
        self.resize_spin.setValue(0.8)  # 默认值
        self.resize_spin.setFixedSize(80, 35)  # 限制控件大小
        # self.resize_spin.valueChanged.connect(self.log_resize_coefficient)  # 连接信号到槽函数


        downsample_label = QtWidgets.QLabel("Downsample factor")
        self.downsample_spin = QtWidgets.QSpinBox()
        self.downsample_spin.setMinimum(1)  # 最小值
        self.downsample_spin.setMaximum(10)  # 最大值
        self.downsample_spin.setSingleStep(1)  # 步进值
        self.downsample_spin.setValue(1)  # 默认值
        self.downsample_spin.setFixedSize(80, 35)  # 限制控件大小
        # self.downsample_spin.valueChanged.connect(self.log_downsample_rate)  # 连接信号到槽函数

        # tmp_layout = _create_horizontal_layout(margins=(0, 0, 0, 0))

        # filter size
        filter_size_label = QtWidgets.QLabel("Filter size (post-processing)")
        self.filter_size_spin = QtWidgets.QSpinBox()
        self.filter_size_spin.setMinimum(2)
        self.filter_size_spin.setMaximum(100)
        self.filter_size_spin.setValue(5)
        self.filter_size_spin.valueChanged.connect(self.log_filter_size)

        # layout.addWidget(self.test_spin, 0, 1)
        layout.addWidget(resize_label, 0, 2)
        layout.addWidget(self.resize_spin, 0, 3)
        layout.addWidget(downsample_label, 0, 4)
        layout.addWidget(self.downsample_spin, 0, 5)
        layout.addWidget(filter_size_label, 0, 6)
        layout.addWidget(self.filter_size_spin, 0, 7)
        # layout.addWidget(snapkeep_num_label, 0, 8)
        # layout.addWidget(self.snapshots, 0, 9)
        # Save results as csv
        # self.save_as_csv = QtWidgets.QCheckBox("Save result(s) as csv")
        # self.save_as_csv.setCheckState(Qt.Unchecked)
        # self.save_as_csv.stateChanged.connect(self.update_csv_choice)
        #
        #
        #
        # self.save_as_nwb = QtWidgets.QCheckBox("Save result(s) as nwb")
        # self.save_as_nwb.setCheckState(Qt.Unchecked)
        # self.save_as_nwb.stateChanged.connect(self.update_nwb_choice)

        # layout.addWidget(self.save_as_csv)

        # Filter predictions
        # self.filter_predictions = QtWidgets.QCheckBox("Filter predictions")
        # self.filter_predictions.setCheckState(Qt.Unchecked)
        # self.filter_predictions.stateChanged.connect(self.update_filter_choice)
        #
        # # layout.addWidget(self.filter_predictions)
        #
        # # Plot Trajectories
        # self.plot_trajectories = QtWidgets.QCheckBox("Plot trajectories")
        # self.plot_trajectories.setCheckState(Qt.Unchecked)
        # self.plot_trajectories.stateChanged.connect(self.update_plot_trajectory_choice)
        #
        # # layout.addWidget(self.plot_trajectories)
        #
        # # Show trajectory plots
        # self.show_trajectory_plots = QtWidgets.QCheckBox("Show trajectory plots")
        # self.show_trajectory_plots.setCheckState(Qt.Unchecked)
        # self.show_trajectory_plots.setEnabled(False)
        # self.show_trajectory_plots.stateChanged.connect(self.update_showfigs_choice)

        # layout.addWidget(self.show_trajectory_plots)

        # layout.addLayout(tmp_layout)


        # self.bodyparts_list_widget = BodypartListWidget(root=self.root, parent=self)
        # layout.addWidget(self.bodyparts_list_widget, Qt.AlignLeft)

    def on_test_spin_changed(self, value):
        """当 QDoubleSpinBox 的值改变时更新共享状态"""
        shared_state.set_test_spin(value)
        print('ana class:',value)

    def update_test_spin(self, value):
        """当共享状态更新时同步控件的值"""
        if self.test_spin.value() != value:
            self.test_spin.setValue(value)
    def _generate_layout_attributes(self, layout):
        # Shuffle
        opt_text = QtWidgets.QLabel("Shuffle")
        self.shuffle = ShuffleSpinBox(root=self.root, parent=self)

        layout.addWidget(opt_text, 0, 0)
        layout.addWidget(self.shuffle, 0, 1)

    def _generate_layout_multianimal(self, layout):
        tmp_layout = QtWidgets.QGridLayout()

        opt_text = QtWidgets.QLabel("Tracking method")
        self.tracker_type_widget = QtWidgets.QComboBox()
        self.tracker_type_widget.addItems(["ellipse", "box", "skeleton"])
        self.tracker_type_widget.currentTextChanged.connect(self.update_tracker_type)
        tmp_layout.addWidget(opt_text, 0, 0)
        tmp_layout.addWidget(self.tracker_type_widget, 0, 1)

        opt_text = QtWidgets.QLabel("Number of animals in videos")
        self.num_animals_in_videos = QtWidgets.QSpinBox()
        self.num_animals_in_videos.setMaximum(100)
        self.num_animals_in_videos.setValue(len(self.root.all_individuals))
        tmp_layout.addWidget(opt_text, 1, 0)
        tmp_layout.addWidget(self.num_animals_in_videos, 1, 1)

        # layout.addLayout(tmp_layout)

        # tmp_layout = QtWidgets.QGridLayout()

        self.calibrate_assembly_checkbox = QtWidgets.QCheckBox("Calibrate assembly")
        self.calibrate_assembly_checkbox.setCheckState(Qt.Unchecked)
        self.calibrate_assembly_checkbox.stateChanged.connect(
            self.update_calibrate_assembly
        )
        tmp_layout.addWidget(self.calibrate_assembly_checkbox, 0, 2)

        self.assemble_with_ID_only_checkbox = QtWidgets.QCheckBox(
            "Assemble with ID only"
        )
        self.assemble_with_ID_only_checkbox.setCheckState(Qt.Unchecked)
        self.assemble_with_ID_only_checkbox.stateChanged.connect(
            self.update_assemble_with_ID_only
        )
        tmp_layout.addWidget(self.assemble_with_ID_only_checkbox, 0, 3)

        self.create_detections_video_checkbox = QtWidgets.QCheckBox(
            "Create video with all detections"
        )
        self.create_detections_video_checkbox.setCheckState(Qt.Unchecked)
        self.create_detections_video_checkbox.stateChanged.connect(
            self.update_create_video_detections
        )
        tmp_layout.addWidget(self.create_detections_video_checkbox, 0, 4)

        layout.addLayout(tmp_layout)
    def log_filter_size(self, value):
       print(f"Filter size set to {value}")
    def update_create_video_detections(self, state):
        s = "ENABLED" if state == Qt.Checked else "DISABLED"
        self.root.logger.info(f"Create video with all detections {s}")

    def update_assemble_with_ID_only(self, state):
        s = "ENABLED" if state == Qt.Checked else "DISABLED"
        self.root.logger.info(f"Assembly with ID only {s}")

    def update_calibrate_assembly(self, state):
        s = "ENABLED" if state == Qt.Checked else "DISABLED"
        self.root.logger.info(f"Assembly calibration {s}")

    def update_tracker_type(self, method):
        self.root.logger.info(f"Using {method.upper()} tracker")

    def update_csv_choice(self, state):
        s = "ENABLED" if state == Qt.Checked else "DISABLED"
        self.root.logger.info(f"Save results as CSV {s}")

    def update_nwb_choice(self, state):
        s = "ENABLED" if state == Qt.Checked else "DISABLED"
        self.root.logger.info(f"Save results as NWB {s}")

    def update_filter_choice(self, state):
        s = "ENABLED" if state == Qt.Checked else "DISABLED"
        self.root.logger.info(f"Filtering predictions {s}")

    def update_showfigs_choice(self, state):
        if state == Qt.Checked:
            self.root.logger.info("Plots will show as pop ups.")
        else:
            self.root.logger.info("Plots will not show up.")

    def update_crop_choice(self, state):
        if state == Qt.Checked:
            self.root.logger.info("Dynamic bodypart cropping ENABLED.")
            self.dynamic_cropping = True
        else:
            self.root.logger.info("Dynamic bodypart cropping DISABLED.")
            self.dynamic_cropping = False

    def update_plot_trajectory_choice(self, state):
        if state == Qt.Checked:
            self.bodyparts_list_widget.show()
            self.bodyparts_list_widget.setEnabled(True)
            self.show_trajectory_plots.setEnabled(True)
            self.root.logger.info("Plot trajectories ENABLED.")

        else:
            self.bodyparts_list_widget.hide()
            self.bodyparts_list_widget.setEnabled(False)
            self.show_trajectory_plots.setEnabled(False)
            self.show_trajectory_plots.setCheckState(Qt.Unchecked)
            self.root.logger.info("Plot trajectories DISABLED.")

    def edit_config_file(self):
        if not self.root.config:
            return
        editor = ConfigEditor(self.root.config)
        editor.show()
    def toggle_video_display(self, checked):
        """控制视频显示和隐藏"""
        if checked:
            self.video_container.setVisible(True)
            self.toggle_videos_btn.setText("Hide Videos")
        else:
            self.video_container.setVisible(False)
            self.toggle_videos_btn.setText("Show Videos")

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
    def extract_whole_frames(self, video_name, output_dir, scale_percent, downsample_rate):
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
        while True:
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
        print(f"Saved whole {saved_count} frames to {output_dir}")


    def analyze_videos(self):
        ###################################
        self.video_display_label1["container"].findChild(QtWidgets.QLabel).setText("Forward Tracking Visualization (Automatic parameter tuning...)")
        self.video_display_label2["container"].findChild(QtWidgets.QLabel).setText("Backward Tracking Visualization (Automatic parameter tuning...)")
        ###################################
        video_name = list(self.files)[0]
        frame_rate = get_video_fps(video_name)
        file_path = Path(video_name)
        file_name_without_extension = file_path.stem
        model_path = self.root.project_folder + '/models'
        if has_model_file(model_path):
            selected_model_path = select_model(file_name_without_extension,model_path)
            mask_path = self.root.project_folder + '/tmp/' + file_name_without_extension + '/masks/'
            if os.path.exists(mask_path):
                extract_output_dir = self.root.project_folder + '/tmp/' + file_name_without_extension + '/extracted-images'
                if not has_jpg_files(extract_output_dir):
                    self.extract_whole_frames(video_name, extract_output_dir, self.resize_spin.value(),
                                        self.downsample_spin.value())
                run_tracking_params = {'video_display_widget1': self.video_display_label1["video_label"],
                                       'video_display_widget2': self.video_display_label2["video_label"],
                                       'project_folder': self.root.project_folder,
                                       'video_name': file_name_without_extension,
                                       'model_path': selected_model_path,
                                       'resize_factor': self.resize_spin.value(),#self.get_reize_coff
                                       'downsample_factor': self.downsample_spin.value(),#self.downsample_spin.value()
                                       'frame_rate': frame_rate,
                                       'frame_num': 500,##########
                                       'search_scale_range': np.arange(2.0, 3.0, 0.5),
                                       'target_sz_bias_range': [0, 0.1],  # [-0.2, -0.1, 0, 0.1, 0.2]
                                       'status_flag': 2,  # train_param_iter 1 test_param_iter 2 test 3
                                       'evaluation_metric': []
                                       }
                print("\033[31mAutomatic parameter tuning begins:\033[0m")
                print(run_tracking_params)
                run_tracking(run_tracking_params)
                ######################################################## best_param_path_find
                param_json_path = run_tracking_params['project_folder'] + '/tmp/' + run_tracking_params[
                    'video_name'] + "/evaluation_metric_for_test.json"
                with open(param_json_path, "r") as file:
                    loaded_results_list = json.load(file)
                best_param_path_find_str = loaded_results_list[-1]["target_sz"] + '_' + loaded_results_list[-1][
                    "search_scale"]
                print('best_param_path_find_str', best_param_path_find_str)
                ###################################################### Tracking settting
                run_tracking_params['frame_num'] = 500 #file_count-2
                run_tracking_params['search_scale_range'] = [float(loaded_results_list[-1]['search_scale'])]
                run_tracking_params['target_sz_bias_range'] = [loaded_results_list[-1]['target_sz_bias']]
                run_tracking_params['status_flag'] = 3
                run_tracking_params['evaluation_metric'] = []
                print("\033[31mTracking begins\033[0m")
                print(run_tracking_params)
                ######################################################
                self.video_display_label1["container"].findChild(QtWidgets.QLabel).setText("Forward Tracking Visualization")
                self.video_display_label2["container"].findChild(QtWidgets.QLabel).setText("Backward Tracking Visualization")
                ############################################################## Tracking
                run_tracking(run_tracking_params)
                result_dir = run_tracking_params['project_folder'] + '/tracking-results/' + run_tracking_params['video_name']
                raw_track_dir = result_dir+ '/'+ find_latest_folder(result_dir)
                dataset_name, result_save_path = post_process_results(run_tracking_params['video_name'],self.filter_size_spin.value(),raw_track_dir,result_dir)
                print(f"Processed raw results from '{raw_track_dir}' and saved the results to '{result_save_path}'.")
                ##################################################################
                video_save_path = create_tracking_video(extract_output_dir, result_save_path, result_dir + '/' + dataset_name + '.mp4',frame_rate)
                print(f"Saved video with track to '{video_save_path}'.")
                ##################################################################
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Information)
                msg.setText(f"Tracking completed successfully! Results are saved in:\n{result_save_path}")
                msg.setInformativeText(
                    "Move 'UDMT - Refine Tracklets' to visualize and optimize the results."
                )
                msg.setWindowTitle("Tracking Complete")
                msg.setMinimumWidth(900)
                self.logo_dir = os.path.dirname(os.path.realpath("logo.png")) + os.path.sep
                self.logo = self.logo_dir + "/assets/logo.png"
                msg.setWindowIcon(QIcon(self.logo))
                msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                msg.exec_()
                ######################
            else:
                QMessageBox.information(
                self,
                "Warning",
                "Please return to 'UDMT - Tracking Initialization' to extract foreground masks from the selected video!",
                QMessageBox.Ok)

        else:
            QMessageBox.information(
                self,
                "Warning",
                "Please complete the model training in 'UDMT - Train network' before proceeding!",
                QMessageBox.Ok
            )

