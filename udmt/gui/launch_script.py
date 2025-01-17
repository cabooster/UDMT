"""
UDMT (https://cabooster.github.io/UDMT/)
Author: Yixin Li
https://github.com/cabooster/UDMT
Licensed under Non-Profit Open Software License 3.0
"""
import sys
import os
import logging

import PySide6.QtWidgets as QtWidgets
import gdown

from udmt.gui import BASE_DIR
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap
from qt_material import apply_stylesheet
def launch_udmt():
    from udmt.version import VERSION

    print(f"Loading UDMT {VERSION}...")
    # Define URLs and corresponding save paths in a dictionary
    files_to_download = {
        'https://zenodo.org/records/14671891/files/trdimp_net_ep.pth.tar?download=1': os.path.join(BASE_DIR,
                                                                                                   "pretrained",
                                                                                                   "trdimp_net_ep.pth.tar"),
        'https://zenodo.org/records/14671891/files/sam_vit_b_01ec64.pth?download=1': os.path.join(BASE_DIR, "tabs",
                                                                                                  "xmem", "sam_model",
                                                                                                  "sam_vit_b_01ec64.pth"),
        'https://zenodo.org/records/14671891/files/XMem.pth?download=1': os.path.join(BASE_DIR, "tabs", "xmem", "saves",
                                                                                      "XMem.pth"),
    }

    # Ensure all directories exist
    for url, save_path in files_to_download.items():
        dir_path = os.path.dirname(save_path)  # Get the directory part of the path
        os.makedirs(dir_path, exist_ok=True)  # Create the directory if it doesn't exist

        # Check if the file already exists
        if not os.path.exists(save_path):
            print(f"File not found at {save_path}. Downloading from {url}...")
            gdown.download(url, save_path, quiet=False,resume=True)
        else:
            print(f"File already exists at {save_path}. Skipping download.")
    ###########################
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QIcon(os.path.join(BASE_DIR, "assets", "logo.png")))
    screen_size = app.screens()[0].size()
    pixmap = QPixmap(os.path.join(BASE_DIR, "assets", "welcome.png")).scaledToWidth(
        int(0.5 * screen_size.width()), Qt.SmoothTransformation
    )
    splash = QtWidgets.QSplashScreen(pixmap)
    splash.show()

    apply_stylesheet(app, theme='default')
    from udmt.gui.window import MainWindow

    window = MainWindow(app)
    window.receiver.start()
    window.showMaximized()
    splash.finish(window)
    sys.exit(app.exec())


if __name__ == "__main__":
    launch_udmt()
