
import os
import logging
import subprocess
import sys
import webbrowser
from functools import cached_property
from pathlib import Path
from typing import List
import qdarkstyle

# import udmt
from udmt import auxiliaryfunctions, VERSION
from udmt.gui import BASE_DIR, components
# from udmt.gui import BASE_DIR, components, utils
from udmt.gui.tabs import *
from udmt.gui.widgets import StreamReceiver, StreamWriter
from PySide6.QtWidgets import QMenu, QWidget, QMainWindow
from PySide6 import QtCore, QtSvg, QtSvgWidgets
from PySide6.QtGui import QIcon, QAction
from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import Qt


class MainWindow(QMainWindow):
    config_loaded = QtCore.Signal()
    video_type_ = QtCore.Signal(str)
    video_files_ = QtCore.Signal(set)

    def __init__(self, app):
        super(MainWindow, self).__init__()
        self.app = app
        screen_size = app.screens()[0].size()
        self.screen_width = screen_size.width()
        self.screen_height = screen_size.height()

        self.logger = logging.getLogger("GUI")

        self.config = None
        self.loaded = False

        self.shuffle_value = 1
        self.trainingset_index = 0
        self.videotype = "mp4"
        self.files = set()

        self.default_set()

        self._generate_welcome_page()
        self.window_set()
        self.default_set()

        names = ["new_project.png", "open.png", "help.png"]
        self.create_actions(names)
        self.create_menu_bar()
        self.load_settings()
        self._toolbar = None
        self.create_toolbar()

        # Thread-safe Stdout redirector
        self.writer = StreamWriter()
        # sys.stdout = self.writer
        self.receiver = StreamReceiver(self.writer.queue)
        self.receiver.new_text.connect(self.print_to_status_bar)

        self._progress_bar = QtWidgets.QProgressBar()
        self._progress_bar.setMaximum(0)
        self._progress_bar.hide()
        self.status_bar.addPermanentWidget(self._progress_bar)

    def print_to_status_bar(self, text):
        self.status_bar.showMessage(text)
        self.status_bar.repaint()

    @property
    def toolbar(self):
        if self._toolbar is None:
            self._toolbar = self.addToolBar("File")
        return self._toolbar

    @cached_property
    def settings(self):
        return QtCore.QSettings()

    def load_settings(self):
        filenames = self.settings.value("recent_files") or []
        for filename in filenames:
            self.add_recent_filename(filename)

    def save_settings(self):
        recent_files = []
        for action in self.recentfiles_menu.actions()[::-1]:
            recent_files.append(action.text())
        self.settings.setValue("recent_files", recent_files)

    def add_recent_filename(self, filename):
        actions = self.recentfiles_menu.actions()
        filenames = [action.text() for action in actions]
        if filename in filenames:
            return
        action = QAction(filename, self)
        before_action = actions[0] if actions else None
        self.recentfiles_menu.insertAction(before_action, action)

    @property
    def cfg(self):
        try:
            cfg = auxiliaryfunctions.read_config(self.config)
        except TypeError:
            cfg = {}
        return cfg

    @property
    def project_folder(self) -> str:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))
        project_path = os.path.join(base_dir, "udmt_project")
        if not os.path.exists(project_path):
            os.makedirs(project_path)
        # return self.cfg.get("project_path", os.path.dirname(os.path.abspath(__file__)))
        return self.cfg.get("project_path",project_path)

    @property
    def is_multianimal(self) -> bool:
        return bool(self.cfg.get("multianimalproject"))

    @property
    def all_bodyparts(self) -> List:
        if self.is_multianimal:
            return self.cfg.get("multianimalbodyparts")
        else:
            return self.cfg["bodyparts"]

    @property
    def all_individuals(self) -> List:
        if self.is_multianimal:
            return self.cfg.get("individuals")
        else:
            return [""]



    def update_cfg(self, text):
        self.root.config = text
        # self.unsupervised_id_tracking.setEnabled(self.is_transreid_available())

    def update_shuffle(self, value):
        self.shuffle_value = value
        self.logger.info(f"Shuffle set to {self.shuffle_value}")

    @property
    def video_type(self):
        return self.videotype

    @video_type.setter
    def video_type(self, ext):
        self.videotype = ext
        self.video_type_.emit(ext)
        self.logger.info(f"Video type set to {self.video_type}")

    @property
    def video_files(self):
        return self.files

    @video_files.setter
    def video_files(self, video_files):
        self.files = set(video_files)
        self.video_files_.emit(self.files)
        self.logger.info(f"Videos selected to analyze:\n{self.files}")

    def window_set(self):
        self.setWindowTitle("UDMT")
        # Set the initial size of the window
        self.resize(1300, 780)

        # Optionally set a minimum and/or maximum size
        self.setMinimumSize(600, 400)

        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor("#ffffff"))
        self.setPalette(palette)

        icon = os.path.join(BASE_DIR, "assets", "logo.png")
        self.setWindowIcon(QIcon(icon))

        self.status_bar = self.statusBar()
        self.status_bar.setObjectName("Status Bar")
        self.status_bar.showMessage("https://cabooster.github.io/UDMT/")

    def _generate_welcome_page(self):
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.setAlignment(Qt.AlignCenter | Qt.AlignTop)
        self.layout.setSpacing(30)

        title = components._create_label_widget(
            f"Welcome to the UDMT GUI {VERSION}!",
            "font:bold; font-size:18px;",
            margins=(0, 30, 0, 0),
        )
        title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(title)

        image_widget = QtWidgets.QLabel(self)
        image_widget.setAlignment(Qt.AlignCenter)
        image_widget.setContentsMargins(0, 0, 0, 0)

        # SVG file
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        svg_file_path = os.path.join(BASE_DIR, "assets", "logo_transparent.svg")

        # create QtSvgWidgets
        # svg_widget = QtSvgWidgets.QSvgWidget(svg_file_path)
        # svg_widget.setFixedSize(250, 250)
        # svg_widget.setFixedHeight(500)

        # add QtSvgWidgets
        # self.layout.addWidget(svg_widget)

        logo = os.path.join(BASE_DIR, "assets", "logo_transparent.png")
        pixmap = QtGui.QPixmap(logo)
        image_widget.setPixmap(
            pixmap.scaledToHeight(200, QtCore.Qt.SmoothTransformation)
        )
        self.layout.addWidget(image_widget)

        description = "UDMT is an open source tool for unsupervised tracking with deep learning.\nOur project page: https://cabooster.github.io/UDMT\n\n To get started,  create a new project or load an existing one."
        label = components._create_label_widget(
            description,
            "font-size:12px; text-align: center;",
            margins=(0, 0, 0, 0),
        )
        label.setMinimumWidth(400)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(label)

        self.layout_buttons = QtWidgets.QHBoxLayout()
        self.layout_buttons.setAlignment(Qt.AlignCenter | Qt.AlignCenter)
        self.create_project_button = QtWidgets.QPushButton("Create New Project")
        self.create_project_button.setFixedWidth(200)
        self.create_project_button.clicked.connect(self._create_project)

        self.load_project_button = QtWidgets.QPushButton("Load Project")
        self.load_project_button.setFixedWidth(200)
        self.load_project_button.clicked.connect(self._open_project)


        ############## three button in main window
        self.layout_buttons.addWidget(self.create_project_button)
        self.layout_buttons.addWidget(self.load_project_button)
        # self.layout_buttons.addWidget(self.run_superanimal_button)

        self.layout.addLayout(self.layout_buttons)

        widget = QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)

    def default_set(self):
        self.name_default = ""
        self.proj_default = ""
        self.exp_default = ""
        self.loc_default = str(Path.home())

    def create_actions(self, names):
        # Creating action using the first constructor
        self.newAction = QAction(self)
        self.newAction.setText("&New Project...")

        self.newAction.setIcon(
            QIcon(os.path.join(BASE_DIR, "assets", "icons", names[0]))
        )
        self.newAction.setShortcut("Ctrl+N")

        self.newAction.triggered.connect(self._create_project)

        # Creating actions using the second constructor
        self.openAction = QAction("&Open...", self)
        self.openAction.setIcon(
            QIcon(os.path.join(BASE_DIR, "assets", "icons", names[1]))
        )
        self.openAction.setShortcut("Ctrl+O")
        self.openAction.triggered.connect(self._open_project)
        ########################
        self.saveAction = QAction("&Save", self)
        self.exitAction = QAction("&Exit", self)

        self.helpAction = QAction("&Help", self)
        self.helpAction.setIcon(
            QIcon(os.path.join(BASE_DIR, "assets", "icons", names[2]))
        )
        self.helpAction.triggered.connect(self._open_help_url)

        self.aboutAction = QAction("&Learn UDMT", self)
        self.aboutAction.triggered.connect(self._open_about_url)

        # self.check_updates = QAction("&Check for Updates...", self)
        # self.check_updates.triggered.connect(_check_for_updates)

    def create_menu_bar(self):
        menu_bar = self.menuBar()

        # File menu
        self.file_menu = QMenu("&File", self)
        menu_bar.addMenu(self.file_menu)

        self.file_menu.addAction(self.newAction)
        self.file_menu.addAction(self.openAction)

        self.recentfiles_menu = self.file_menu.addMenu("Open Recent")
        self.recentfiles_menu.triggered.connect(
            lambda a: self._update_project_state(a.text(), True)
        )
        self.file_menu.addAction(self.saveAction)
        self.file_menu.addAction(self.exitAction)

        # Help menu
        help_menu = QMenu("&Help", self)
        menu_bar.addMenu(help_menu)
        help_menu.addAction(self.helpAction)
        help_menu.adjustSize()
        # help_menu.addAction(self.check_updates)
        help_menu.addAction(self.aboutAction)

    def update_menu_bar(self):
        self.file_menu.removeAction(self.newAction)
        self.file_menu.removeAction(self.openAction)

    def create_toolbar(self):
        self.toolbar.addAction(self.newAction)
        self.toolbar.addAction(self.openAction)
        self.toolbar.addAction(self.helpAction)

    def remove_action(self):
        self.toolbar.removeAction(self.newAction)
        self.toolbar.removeAction(self.openAction)
        self.toolbar.removeAction(self.helpAction)

    def _update_project_state(self, config, loaded):
        self.config = config
        self.loaded = loaded
        if loaded:
            self.add_recent_filename(self.config)
            self.add_tabs()

    def _create_project(self):
        dlg = ProjectCreator(self)
        dlg.show()

    def _open_project(self):
        open_project = OpenProject(self)
        open_project.load_config(self.project_folder)
        if not open_project.config:
            return

        open_project.loaded = True
        self._update_project_state(
            open_project.config,
            open_project.loaded,
        )

    def _open_help_url(self):

        webbrowser.open("https://cabooster.github.io/UDMT/Tutorial/")

    def _open_about_url(self):

        webbrowser.open("https://cabooster.github.io/UDMT/About/")

    # def _goto_superanimal(self):# model zoo delete
    #     self.tab_widget = QtWidgets.QTabWidget()
    #     self.tab_widget.setContentsMargins(0, 20, 0, 0)
    #     self.modelzoo = ModelZoo(
    #         root=self, parent=None, h1_description="UDMT - Model Zoo"
    #     )
    #     self.tab_widget.addTab(self.modelzoo, "Model Zoo")
    #     self.setCentralWidget(self.tab_widget)

    def load_config(self, config):
        self.config = config
        self.config_loaded.emit()
        print(f'Project "{self.cfg["Task"]}" successfully loaded.')

    def darkmode(self):
        dark_stylesheet = qdarkstyle.load_stylesheet_pyside2()
        self.app.setStyleSheet(dark_stylesheet)

        names = ["new_project2.png", "open2.png", "help2.png"]
        self.remove_action()
        self.create_actions(names)
        self.update_menu_bar()
        self.create_toolbar()

    def lightmode(self):
        from qdarkstyle.light.palette import LightPalette

        style = qdarkstyle.load_stylesheet(palette=LightPalette)
        self.app.setStyleSheet(style)

        names = ["new_project.png", "open.png", "help.png"]
        self.remove_action()
        self.create_actions(names)
        self.create_toolbar()
        self.update_menu_bar()

    def add_tabs(self):

        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setContentsMargins(0, 20, 0, 0)
        self.tracking_initialization = TrackingInitialization(
            root=self, parent=None, h1_description="UDMT - Tracking Initialization"
        )

        self.create_training_dataset = CreateTrainingDataset(
            root=self,
            parent=None,
            h1_description="UDMT - Create Training Dataset",
        )
        self.train_network = TrainNetwork(
            root=self, parent=None, h1_description="UDMT - Train Network"
        )
        self.analyze_videos = AnalyzeVideos(
            root=self, parent=None, h1_description="UDMT - Analyze Videos"
        )

        self.refine_tracklets = RefineTracklets(
            root=self, parent=None, h1_description="UDMT - Visualize Tracks"
        )

        self.tab_widget.addTab(self.tracking_initialization, "Tracking Initialization")
        self.tab_widget.addTab(self.create_training_dataset, "Create training dataset")
        self.tab_widget.addTab(self.train_network, "Train network")
        self.tab_widget.addTab(self.analyze_videos, "Analyze videos")
        self.tab_widget.addTab(self.refine_tracklets, "Visualize Tracks")


        self.setCentralWidget(self.tab_widget)
        self.tab_widget.currentChanged.connect(self.refresh_active_tab)

    def refresh_active_tab(self):
        active_tab = self.tab_widget.currentWidget()
        tab_label = self.tab_widget.tabText(self.tab_widget.currentIndex())

        widget_to_attribute_map = {
            QtWidgets.QSpinBox: "setValue",
            components.ShuffleSpinBox: "setValue",
            components.TrainingSetSpinBox: "setValue",
            QtWidgets.QLineEdit: "setText",
        }

        def _attempt_attribute_update(widget_name, updated_value):
            try:
                widget = getattr(active_tab, widget_name)
                method = getattr(widget, widget_to_attribute_map[type(widget)])
                self.logger.debug(
                    f"Setting {widget_name}={updated_value} in tab '{tab_label}'"
                )
                method(updated_value)
            except AttributeError:
                pass

        _attempt_attribute_update("shuffle", self.shuffle_value)
        _attempt_attribute_update("cfg_line", self.config)


    def closeEvent(self, event):
        print("Exiting...")
        self.receiver.terminate()
        event.accept()
        self.save_settings()
