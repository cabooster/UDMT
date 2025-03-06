
import os
from pathlib import Path

from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMessageBox, QSpacerItem, QSizePolicy

from udmt.gui.components import (
    DefaultTab,
    ShuffleSpinBox,
    _create_grid_layout,
    _create_label_widget,
    VideoSelectionWidget, LogWidget
)
from udmt.gui.widgets import ConfigEditor

import udmt
from udmt.utils import auxiliaryfunctions
from udmt.gui.tabs.ST_Net.ltr.run_training import run_training_process


def is_folder_empty(folder_path):
    """
    Check if a folder contains any files.
    Returns True if the folder has no files, otherwise False.
    """
    # Iterate through items in the folder
    for item in os.listdir(folder_path):
        # Build the full path
        item_path = os.path.join(folder_path, item)

        # Check if the item is a file
        if os.path.isfile(item_path):
            return False  # Found a file, folder is not empty

    return True  # No files found, folder is empty
class TrainNetwork(DefaultTab):
    def __init__(self, root, parent, h1_description):
        super(TrainNetwork, self).__init__(root, parent, h1_description)
        ##############################
        self.main_layout.addWidget(_create_label_widget("Video Selection", "font:bold"))
        self.video_selection_widget = VideoSelectionWidget(self.root, self)
        self.main_layout.addWidget(self.video_selection_widget)
        ##############################

        # use the default pose_cfg file for default values
        # default_pose_cfg_path = os.path.join(
        #     Path(udmt.__file__).parent, "pose_cfg.yaml"
        # )
        # pose_cfg = auxiliaryfunctions.read_plainconfig(default_pose_cfg_path)
        # self.display_iters = str(pose_cfg["display_iters"])
        # self.save_iters = str(pose_cfg["save_iters"])
        # self.max_iters = str(pose_cfg["multi_step"][-1][-1])

        self._set_page()

    def _set_page(self):
        self.main_layout.addWidget(_create_label_widget("Attributes", "font:bold"))
        self.layout_attributes = _create_grid_layout(margins=(20, 0, 0, 0))
        self._generate_layout_attributes(self.layout_attributes)
        self.main_layout.addLayout(self.layout_attributes)

        self.main_layout.addWidget(_create_label_widget(""))  # dummy label

        # self.edit_posecfg_btn = QtWidgets.QPushButton("Edit pose_cfg.yaml")
        # self.edit_posecfg_btn.setMinimumWidth(150)
        # self.edit_posecfg_btn.clicked.connect(self.open_posecfg_editor)

        self.ok_button = QtWidgets.QPushButton("Train Network")
        self.ok_button.setMinimumWidth(150)
        self.ok_button.setToolTip("Click to start training the model.")
        self.ok_button.clicked.connect(self.train_network)

        # self.main_layout.addWidget(self.edit_posecfg_btn, alignment=Qt.AlignRight)
        self.main_layout.addWidget(self.ok_button, alignment=Qt.AlignRight)

        ##############################
        spacer = QSpacerItem(150, 400, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.main_layout.addItem(spacer)
        self.log_widget = LogWidget(self.root, self)
        # self.main_layout.addStretch()
        self.main_layout.addWidget(self.log_widget)



    def _generate_layout_attributes(self, layout):
        # batch size
        batch_size_label = QtWidgets.QLabel("Batch size")
        self.batch_size_spin = QtWidgets.QSpinBox()
        self.batch_size_spin.setMinimum(2)
        self.batch_size_spin.setMaximum(100)
        self.batch_size_spin.setValue(8)
        self.batch_size_spin.setToolTip(
            "Adjust the batch size based on GPU usage. If you encounter a 'CUDA out of memory' error, try reducing the batch size.")
        self.batch_size_spin.valueChanged.connect(self.log_batch_size)
        # num workers
        num_workers_label = QtWidgets.QLabel("Num of workers")
        self.num_workers_spin = QtWidgets.QSpinBox()
        self.num_workers_spin.setMinimum(0)
        self.num_workers_spin.setMaximum(100)
        self.num_workers_spin.setValue(0)
        self.num_workers_spin.valueChanged.connect(self.log_batch_size)
        self.num_workers_spin.setToolTip(
            "Set the number of workers for parallel processing. On Windows, set to 0; on Linux, set to 4, 8, or higher for faster performance.")
        # Max iterations
        max_epoches_label = QtWidgets.QLabel("Maximum epoches")
        self.max_epoches_spin = QtWidgets.QSpinBox()
        self.max_epoches_spin.setMinimum(1)
        self.max_epoches_spin.setMaximum(100)
        self.max_epoches_spin.setValue(20)
        self.max_epoches_spin.setToolTip(
            "Set the maximum number of epochs for training. Typically, 20 epochs are sufficient for convergence to a good result.")
        self.max_epoches_spin.valueChanged.connect(self.log_max_iters)

        # Max number snapshots to keep
        snapkeep_num_label = QtWidgets.QLabel("Number of snapshots to keep")
        self.snapshots = QtWidgets.QSpinBox()
        self.snapshots.setMinimum(1)
        self.snapshots.setMaximum(100)
        self.snapshots.setValue(3)
        self.snapshots.valueChanged.connect(self.log_snapshots)

        layout.addWidget(batch_size_label, 0, 2)
        layout.addWidget(self.batch_size_spin, 0, 3)
        layout.addWidget(num_workers_label, 0, 4)
        layout.addWidget(self.num_workers_spin, 0, 5)
        layout.addWidget(max_epoches_label, 0, 6)
        layout.addWidget(self.max_epoches_spin, 0, 7)
        # layout.addWidget(snapkeep_num_label, 0, 8)
        # layout.addWidget(self.snapshots, 0, 9)


    def log_num_workers(self, value):
        print(f"Number of workers set to {value}")

    def log_batch_size(self, value):
       print(f"Batch size set to {value}")

    def log_max_iters(self, value):
        print(f"Max epoches set to {value}")

    def log_snapshots(self, value):
        print(f"Max epoches to keep set to {value}")

    # def open_posecfg_editor(self):
    #     editor = ConfigEditor(self.root.pose_cfg_path)
    #     editor.show()
    @property
    def files(self):
        return self.video_selection_widget.files

    def train_network(self):

        video_name = list(self.files)[0]
        file_path = Path(video_name)
        file_name_without_extension = file_path.stem
        train_set_path = self.root.project_folder + '/training-datasets/'  + file_name_without_extension + '/label'
        if is_folder_empty(train_set_path):
            QMessageBox.information(
                self,
                "Warning",
                "Please complete the training dataset creation in 'UDMT - Create Training Dataset' before proceeding!",
                QMessageBox.Ok
            )
        else:
            run_training_params = {'project_folder': self.root.project_folder,
                                   'video_name': file_name_without_extension,
                                   'num_workers': self.num_workers_spin.value(),
                                   'epoch_num': self.max_epoches_spin.value(),
                                   'batch_size': self.batch_size_spin.value(),
                                   'max_save_snapshots': 3,
                                   'logger': self.root.logger
                                   }
            print(f'run_training_params: {run_training_params}')
            self.root.logger.info(f'run_training_params: {run_training_params}')
            print('Start training...')
            self.root.logger.info('Start training...')
            run_training_process(run_training_params)
            print("Move to 'UDMT - Analyze Video' for animal tracking.")
            self.root.logger.info("Move to 'UDMT - Analyze Video' for animal tracking.")
            ##############################################
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setText("The network is now trained and ready to track.")
            msg.setInformativeText(
                "Move to 'UDMT - Analyze Video' for animal tracking."
            )

            msg.setWindowTitle("Info")
            msg.setMinimumWidth(900)
            self.logo_dir = os.path.dirname(os.path.realpath("logo.png")) + os.path.sep
            self.logo = self.logo_dir + "/assets/logo.png"
            msg.setWindowIcon(QIcon(self.logo))
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.exec_()
            ##############################################


