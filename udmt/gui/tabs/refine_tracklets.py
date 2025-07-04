import json
import os
from pathlib import Path
from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox, QSpacerItem, QSizePolicy

from udmt.gui.tabs.ST_Net.pytracking.utils.filter_and_save import post_process_results
from udmt.gui.tabs.analyze_videos import find_latest_folder, has_jpg_files
from udmt.gui.tabs.udmt_refine_gui.refine_win import TrackletVisualizer
from udmt.gui.widgets import ConfigEditor
from udmt.gui.components import (
    DefaultTab,
    ShuffleSpinBox,
    VideoSelectionWidget,
    _create_grid_layout,
    _create_horizontal_layout,
    _create_label_widget, LogWidget,
)

import udmt
# from udmt.pose_estimation_tensorflow.lib import trackingutils
# from udmt.utils.auxiliaryfunctions import GetScorerName


class RefineTracklets(DefaultTab):
    def __init__(self, root, parent, h1_description):
        super(RefineTracklets, self).__init__(root, parent, h1_description)
        self._set_page()

    @property
    def files(self):
        return self.video_selection_widget.files

    def _set_page(self):
        # TODO: Multi video select.... have to change to single video!
        self.main_layout.addWidget(_create_label_widget("Video Selection", "font:bold"))
        self.video_selection_widget = VideoSelectionWidget(self.root, self)
        self.main_layout.addWidget(self.video_selection_widget)

        # self.main_layout.addWidget(_create_label_widget("Attributes", "font:bold"))
        # self.layout_attributes = _create_horizontal_layout()
        # self._generate_layout_attributes(self.layout_attributes)
        # self.main_layout.addLayout(self.layout_attributes)

        # self.container_layout = _create_horizontal_layout(margins=(0, 0, 0, 0))

        # self.layout_refinement_settings = _create_grid_layout(margins=(20, 0, 0, 0))
        # self._generate_layout_refinement(self.layout_refinement_settings)
        # self.container_layout.addLayout(self.layout_refinement_settings)
        self.main_layout.addWidget(_create_label_widget("Attributes", "font:bold"))
        self.layout_attributes = _create_grid_layout(margins=(20, 0, 0, 0))

        # self.layout_filtering_settings = _create_grid_layout(margins=(20, 0, 0, 0))
        self._generate_layout_filtering(self.layout_attributes)
        self.main_layout.addLayout(self.layout_attributes)

        # self.main_layout.addLayout(self.container_layout)

        # self.stitch_tracklets_btn = QtWidgets.QPushButton("(Re-)run stitching")
        # self.stitch_tracklets_btn.setMinimumWidth(150)
        # self.stitch_tracklets_btn.clicked.connect(self.create_tracks)

        # self.edit_inferencecfg_btn = QtWidgets.QPushButton("Edit inference_cfg.yaml")
        # self.edit_inferencecfg_btn.setMinimumWidth(150)
        # self.edit_inferencecfg_btn.clicked.connect(self.open_inferencecfg_editor)
        #
        # self.filter_tracks_button = QtWidgets.QPushButton("Filter tracks ( + .csv)")
        # self.filter_tracks_button.setMinimumWidth(150)
        # self.filter_tracks_button.clicked.connect(self.filter_tracks)

        self.launch_button = QtWidgets.QPushButton("Launch track visualization GUI")
        self.launch_button.setMinimumWidth(150)
        self.launch_button.clicked.connect(self.refine_tracks)

        # self.merge_button = QtWidgets.QPushButton("Merge dataset")
        # self.merge_button.setMinimumWidth(150)
        # self.merge_button.clicked.connect(self.merge_dataset)
        # self.merge_button.setEnabled(False)

        # self.main_layout.addWidget(self.edit_inferencecfg_btn, alignment=Qt.AlignRight)
        # self.main_layout.addWidget(self.stitch_tracklets_btn, alignment=Qt.AlignRight)
        self.main_layout.addWidget(self.launch_button, alignment=Qt.AlignRight)
        # self.main_layout.addWidget(self.filter_tracks_button, alignment=Qt.AlignRight)
        # self.main_layout.addWidget(self.merge_button, alignment=Qt.AlignRight)

        ##############################
        spacer = QSpacerItem(150, 400, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.main_layout.addItem(spacer)
        self.log_widget = LogWidget(self.root, self)
        # self.main_layout.addStretch()
        self.main_layout.addWidget(self.log_widget)

    def _generate_layout_attributes(self, layout):
        # Shuffle
        shuffle_text = QtWidgets.QLabel("Shuffle")
        self.shuffle = ShuffleSpinBox(root=self.root, parent=self)

        # Num animals
        num_animals_text = QtWidgets.QLabel("Number of animals in video")
        self.num_animals_in_videos = QtWidgets.QSpinBox()
        self.num_animals_in_videos.setValue(len(self.root.all_individuals))
        self.num_animals_in_videos.setMaximum(100)
        self.num_animals_in_videos.valueChanged.connect(self.log_num_animals)

        layout.addWidget(shuffle_text)
        layout.addWidget(self.shuffle)
        layout.addWidget(num_animals_text)
        layout.addWidget(self.num_animals_in_videos)

    def _generate_layout_refinement(self, layout):
        section_title = _create_label_widget(
            "Refinement Settings", "font:bold", (0, 50, 0, 0)
        )

        # Min swap length
        swap_length_label = QtWidgets.QLabel("Min swap length to highlight")
        self.swap_length_widget = QtWidgets.QSpinBox()
        self.swap_length_widget.setValue(2)
        self.swap_length_widget.setMinimumWidth(150)
        self.swap_length_widget.valueChanged.connect(self.log_swap_length)

        # Max gap to fill
        max_gap_label = QtWidgets.QLabel("Max gap of missing data to fill")
        self.max_gap_widget = QtWidgets.QSpinBox()
        self.max_gap_widget.setValue(5)
        self.max_gap_widget.setMinimumWidth(150)
        self.max_gap_widget.valueChanged.connect(self.log_max_gap)

        # Trail length
        trail_length_label = QtWidgets.QLabel("Visualization trail length")
        self.trail_length_widget = QtWidgets.QSpinBox()
        self.trail_length_widget.setValue(20)
        self.trail_length_widget.setMinimumWidth(150)
        self.trail_length_widget.valueChanged.connect(self.log_trail_length)

        layout.addWidget(section_title, 0, 0)
        layout.addWidget(swap_length_label, 1, 0)
        layout.addWidget(self.swap_length_widget, 1, 1)
        layout.addWidget(max_gap_label, 2, 0)
        layout.addWidget(self.max_gap_widget, 2, 1)
        layout.addWidget(trail_length_label, 3, 0)
        layout.addWidget(self.trail_length_widget, 3, 1)

    def _generate_layout_filtering(self, layout):
        # section_title = _create_label_widget("Filtering", "font:bold", (0, 50, 0, 0))

        # Filter type
        filter_label = QtWidgets.QLabel("Filter type (post-processing)")
        self.filter_type_widget = QtWidgets.QComboBox()
        # self.filter_type_widget.setMinimumWidth(150)
        options = ["mean","median"]
        self.filter_type_widget.addItems(options)
        self.filter_type_widget.setToolTip(
            "Select the filter type for trajectory smoothing during post-processing. Options are 'mean' or 'median'.")
        self.filter_type_widget.currentTextChanged.connect(self.log_filter_type)


        # Filter window length
        window_length_label = QtWidgets.QLabel("Filter size (post-processing)")
        self.window_length_widget = QtWidgets.QSpinBox()
        self.window_length_widget.setValue(10)
        self.window_length_widget.setToolTip("Set the filter size for trajectory smoothing during post-processing.")
        # self.window_length_widget.setMinimumWidth(150)
        self.window_length_widget.valueChanged.connect(self.log_window_length)

        # layout.addWidget(section_title, 0, 0)
        layout.addWidget(filter_label, 0, 1)
        layout.addWidget(self.filter_type_widget, 0, 2)
        layout.addWidget(window_length_label, 0, 3)
        layout.addWidget(self.window_length_widget, 0, 4)

    def log_swap_length(self, value):
        self.root.logger.info(f"Swap length set to {value}")

    def log_max_gap(self, value):
        self.root.logger.info(f"Max gap size of missing data to fill set to {value}")

    def log_trail_length(self, value):
        self.root.logger.info(f"Visualization trail length set to {value}")

    def log_filter_type(self, filter_type):
        self.root.logger.info(f"Filter type set to {filter_type.upper()}")

    def log_window_length(self, window_length):
        self.root.logger.info(f"Window length set to {window_length}")

    def log_num_animals(self, num_animals):
        self.root.logger.info(f"Number of animals in video set to {num_animals}")

    def open_inferencecfg_editor(self):
        editor = ConfigEditor(self.root.inference_cfg_path)
        editor.show()

    def create_tracks(self):
        udmt.stitch_tracklets(
            self.root.config,
            self.files,
            videotype=self.video_selection_widget.videotype_widget.currentText(),
            shuffle=self.shuffle.value(),
            n_tracks=self.num_animals_in_videos.value(),
        )

    def filter_tracks(self):
        window_length = self.window_length_widget.value()
        if window_length % 2 != 1:
            raise ValueError("Window length should be odd.")

        videotype = self.video_selection_widget.videotype_widget.currentText()
        udmt.filterpredictions(
            self.root.config,
            self.files,
            videotype=videotype,
            shuffle=self.shuffle.value(),
            filtertype=self.filter_type_widget.currentText(),
            windowlength=self.window_length_widget.value(),
            save_as_csv=True,
        )

    def merge_dataset(self):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setText(
            "Make sure that you have refined all the labels before merging the dataset.If you merge the dataset, you need to re-create the training dataset before you start the training. Are you ready to merge the dataset?"
        )
        msg.setWindowTitle("Warning")
        msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        result = msg.exec_()
        if result == QtWidgets.QMessageBox.Yes:
            udmt.merge_datasets(self.root.config, forceiterate=None)
            self.viz.export_to_training_data()

    def refine_tracks(self):
        video_name = list(self.files)[0]
        file_path = Path(video_name)
        video_name = file_path.stem
        result_dir = self.root.project_folder + '/tracking-results/' + video_name
        raw_track_dir = result_dir + '/' + find_latest_folder(result_dir)
        _, result_save_path = post_process_results(video_name, self.window_length_widget.value(),
                                                raw_track_dir, result_dir, self.filter_type_widget.currentText())

        video_frames_path = self.root.project_folder + '/tmp/' + video_name + '/extracted-images'
        if has_jpg_files(video_frames_path):
            track_path = result_save_path
            print(
                f"Post-processing the files from '{raw_track_dir}' and ready to perform track refinement on the file '{result_save_path}'.")
            json_file_path_time_point = self.root.project_folder + '/tmp/' + video_name + "/cross_timepoints.json"
            with open(json_file_path_time_point, 'r') as f:
                loaded_time_point = json.load(f)
            #print(f"loaded_list: {loaded_list}")
            viz = TrackletVisualizer(video_frames_path, track_path, loaded_time_point)
            viz.show()
        else:
            QMessageBox.information(
                self,
                "Warning",
                "Please return to 'UDMT - Analyze Videos' for tracks!",
                QMessageBox.Ok)

