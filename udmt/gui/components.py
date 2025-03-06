"""
UDMT (https://cabooster.github.io/UDMT/)
Author: Yixin Li
https://github.com/cabooster/UDMT
Licensed under Non-Profit Open Software License 3.0
"""
import os

from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Qt
from tqdm import tqdm

from udmt.gui.udmt_params import UDMTParams
from udmt.gui.widgets import ConfigEditor


def _create_label_widget(
    text: str,
    style: str = "",
    margins: tuple = (20, 10, 0, 10),
) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    label.setContentsMargins(*margins)
    label.setStyleSheet(style)

    return label


def _create_horizontal_layout(
    alignment=None, spacing: int = 20, margins: tuple = (20, 0, 0, 0)
) -> QtWidgets.QHBoxLayout():
    layout = QtWidgets.QHBoxLayout()
    layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    layout.setSpacing(spacing)
    layout.setContentsMargins(*margins)

    return layout


def _create_vertical_layout(
    alignment=None, spacing: int = 20, margins: tuple = (20, 0, 0, 0)
) -> QtWidgets.QVBoxLayout():
    layout = QtWidgets.QVBoxLayout()
    layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    layout.setSpacing(spacing)
    layout.setContentsMargins(*margins)

    return layout


def _create_grid_layout(
    alignment=None,
    spacing: int = 20,
    margins: tuple = None,
) -> QtWidgets.QGridLayout():
    layout = QtWidgets.QGridLayout()
    layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    layout.setSpacing(spacing)
    if margins:
        layout.setContentsMargins(*margins)

    return layout


class BodypartListWidget(QtWidgets.QListWidget):
    def __init__(
        self,
        root: QtWidgets.QMainWindow,
        parent: QtWidgets.QWidget,
        # all_bodyparts: List
        # NOTE: Is there a case where a specific list should
        # have bodyparts other than the root? I don't think so.
    ):
        super(BodypartListWidget, self).__init__()

        self.root = root
        self.parent = parent
        self.selected_bodyparts = self.root.all_bodyparts

        self.setEnabled(False)
        self.setMaximumWidth(600)
        self.setMaximumHeight(500)
        self.hide()

        self.addItems(self.root.all_bodyparts)
        self.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)

        self.itemSelectionChanged.connect(self.update_selected_bodyparts)

    def update_selected_bodyparts(self):
        self.selected_bodyparts = [item.text() for item in self.selectedItems()]
        self.root.logger.info(f"Selected bodyparts:\n\t{self.selected_bodyparts}")
############################################
class LogWidget(QtWidgets.QWidget):
    def __init__(self, root: QtWidgets.QMainWindow, parent: QtWidgets.QWidget):
        super(LogWidget, self).__init__(parent)

        self.root = root
        self.parent = parent

        self._init_layout()

        self.root.log_message_.connect(self.update_log)

    def _init_layout(self):
        layout = QtWidgets.QVBoxLayout()
        # layout = _create_horizontal_layout()
        log_label = QtWidgets.QLabel("Logs")
        log_label.setStyleSheet("margin-bottom: 0px; padding-bottom: 0px;")
        layout.addWidget(log_label)

        self.text_log = QtWidgets.QTextEdit()
        self.text_log.setReadOnly(True)
        self.text_log.setStyleSheet("background-color: #FFFFFF; color: #000000; margin-top: 0px; padding-top: 0px;")
        #self.text_log.setFixedHeight(100)
        # self.text_log.setMinimumHeight(50)
        # self.text_log.setMinimumWidth(500)
        self.text_log.setFixedSize(1260, 80)
        layout.addWidget(self.text_log)

        self.setLayout(layout)

    def update_log(self, message: str):

        self.text_log.append(message)

    def clear_log(self):

        self.text_log.clear()
        self.root.log("Cleared logs")

class TqdmLogger(QtCore.QObject):


    log_signal = QtCore.Signal(str, bool)  # (message, is_tqdm)

    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        self.log_signal.connect(self._update_gui_log)
        self.last_tqdm_message = ""

    def write(self, message):

        if message.strip():
            is_tqdm = message.startswith("Processing Frames") or "%" in message
            self.log_signal.emit(message, is_tqdm)

    def flush(self):

        pass

    def _update_gui_log(self, message, is_tqdm):

        cursor = self.text_widget.textCursor()

        if is_tqdm:

            self.last_tqdm_message = message
            cursor.movePosition(QtGui.QTextCursor.End)
            cursor.select(QtGui.QTextCursor.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.insertText(self.last_tqdm_message)
        else:

            self.text_widget.append(message)

        self.text_widget.setTextCursor(cursor)
class VideoSelectionWidget(QtWidgets.QWidget):
    def __init__(self, root: QtWidgets.QMainWindow, parent: QtWidgets.QWidget):
        super(VideoSelectionWidget, self).__init__(parent)

        self.root = root
        self.parent = parent

        self._init_layout()

    def _init_layout(self):
        layout = _create_horizontal_layout()

        # # Videotype selection
        # self.videotype_widget = QtWidgets.QComboBox()
        # self.videotype_widget.setMinimumWidth(100)
        # self.videotype_widget.addItems(DLCParams.VIDEOTYPES)
        # self.videotype_widget.setCurrentText(self.root.video_type)
        # self.root.video_type_.connect(self.videotype_widget.setCurrentText)
        # self.videotype_widget.currentTextChanged.connect(self.update_videotype)

        # Select videos
        self.select_video_button = QtWidgets.QPushButton("Step 1. Select videos")
        self.select_video_button.setToolTip("Select a video you want to process.")
        self.select_video_button.setMaximumWidth(200)
        self.select_video_button.clicked.connect(self.select_single_video)###241207
        self.root.video_files_.connect(self._update_video_selection)

        # Number of selected videos text
        self.selected_videos_text = QtWidgets.QLabel("")  # updated when videos are selected

        # Clear video selection
        # self.clear_videos = QtWidgets.QPushButton("Clear selection")
        # self.clear_videos.clicked.connect(self.clear_selected_videos)

        # layout.addWidget(self.videotype_widget)
        layout.addWidget(self.select_video_button)
        layout.addWidget(self.selected_videos_text)
        # layout.addWidget(self.clear_videos, alignment=Qt.AlignRight)

        self.setLayout(layout)

    @property
    def files(self):
        return self.root.video_files

    def update_videotype(self, vtype):
        self.clear_selected_videos()
        self.root.video_type = vtype

    def _update_video_selection(self, videopaths):
        n_videos = len(self.root.video_files)
        if n_videos:
            self.selected_videos_text.setText(f"{self.root.video_files} selected")
            # self.selected_videos_text.setText(f"{n_videos} videos selected")
            # self.select_video_button.setText("Add more videos")
        else:
            self.selected_videos_text.setText("")
            self.select_video_button.setText("Select videos")

    def select_videos(self):
        cwd = os.path.join(self.root.project_folder, 'videos')
        filenames = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select video(s) to analyze",
            cwd,
            f"Videos ({' *.'.join(UDMTParams.VIDEOTYPES)[1:]})",
        )

        if filenames[0]:
            # Qt returns a tuple (list of files, filetype)
            self.root.video_files = filenames[0]

    def select_single_video(self):
        cwd = os.path.join(self.root.project_folder, 'videos')
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select a video to analyze",
            cwd,
            f"Videos ({' *.'.join(UDMTParams.VIDEOTYPES)[1:]})",
        )

        if filename:

            self.root.video_files = [filename]

    def clear_selected_videos(self):
        self.root.video_files = set()
        self.root.logger.info(f"Cleared selected videos")



class TrainingSetSpinBox(QtWidgets.QSpinBox):
    def __init__(self, root, parent):
        super(TrainingSetSpinBox, self).__init__(parent)

        self.root = root
        self.parent = parent

        self.setMaximum(100)
        self.setValue(self.root.trainingset_index)
        self.valueChanged.connect(self.root.update_trainingset)


class ShuffleSpinBox(QtWidgets.QSpinBox):
    def __init__(self, root, parent):
        super(ShuffleSpinBox, self).__init__(parent)

        self.root = root
        self.parent = parent

        self.setMaximum(100)
        self.setValue(self.root.shuffle_value)
        self.valueChanged.connect(self.root.update_shuffle)


class DefaultTab(QtWidgets.QWidget):
    def __init__(
        self,
        root: QtWidgets.QMainWindow,
        parent: QtWidgets.QWidget = None,
        h1_description: str = "",
    ):
        super(DefaultTab, self).__init__(parent)

        self.parent = parent
        self.root = root

        self.h1_description = h1_description

        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setLayout(self.main_layout)

        self._init_default_layout()

    def _init_default_layout(self):
        # Add tab header
        self.main_layout.addWidget(
            _create_label_widget(self.h1_description, "font:bold;", (10, 10, 0, 10))
        )

        # Add separating line
        self.separator = QtWidgets.QFrame()
        self.separator.setFrameShape(QtWidgets.QFrame.HLine)
        self.separator.setFrameShadow(QtWidgets.QFrame.Raised)
        self.separator.setLineWidth(0)
        self.separator.setMidLineWidth(1)
        policy = QtWidgets.QSizePolicy()
        policy.setVerticalPolicy(QtWidgets.QSizePolicy.Policy.Fixed)
        policy.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding)
        self.separator.setSizePolicy(policy)
        self.main_layout.addWidget(self.separator)


class EditYamlButton(QtWidgets.QPushButton):
    def __init__(
        self, button_label: str, filepath: str, parent: QtWidgets.QWidget = None
    ):
        super(EditYamlButton, self).__init__(button_label)
        self.filepath = filepath
        self.parent = parent

        self.clicked.connect(self.open_config)

    def open_config(self):
        editor = ConfigEditor(self.filepath)
        editor.show()


class BrowseFilesButton(QtWidgets.QPushButton):
    def __init__(
        self,
        button_label: str,
        filetype: str = None,
        cwd: str = None,
        single_file: bool = False,
        dialog_text: str = None,
        file_text: str = None,
        parent=None,
    ):
        super(BrowseFilesButton, self).__init__(button_label)
        self.filetype = filetype
        self.single_file_only = single_file
        self.cwd = cwd
        self.parent = parent

        self.dialog_text = dialog_text
        self.file_text = file_text

        self.files = set()

        self.clicked.connect(self.browse_files)

    def browse_files(self):
        # Look for any extension by default
        file_ext = "*"
        if self.filetype:
            # This works both with e.g. .avi and avi
            file_ext = self.filetype.split(".")[-1]

        # Choose multiple files by default
        open_file_func = QtWidgets.QFileDialog.getOpenFileNames
        if self.single_file_only:
            open_file_func = QtWidgets.QFileDialog.getOpenFileName

        cwd = ""
        if self.cwd:
            cwd = self.cwd

        dialog_text = f"Select .{file_ext} files"
        if self.dialog_text:
            dialog_text = self.dialog_text

        file_text = f"Files (*.{file_ext})"
        if self.file_text:
            file_text = self.file_text

        filepaths = open_file_func(self, dialog_text, cwd, file_text)

        if filepaths:
            self.files.update(filepaths[0])
