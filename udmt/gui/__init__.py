"""
UDMT (https://cabooster.github.io/UDMT/)
Author: Yixin Li
https://github.com/cabooster/UDMT
Licensed under Non-Profit Open Software License 3.0
"""
import ctypes
import os
import sys

os.environ["QT_API"] = "pyside6"

# PySide6 6.5+ requires libxcb-cursor on Linux. The udmt-sam3 Conda
# environment provides it in $CONDA_PREFIX/lib, which is not in the wheel's
# plugin RUNPATH, so make it visible before Qt loads the xcb platform plugin.
if sys.platform.startswith("linux"):
    xcb_cursor = os.path.join(sys.prefix, "lib", "libxcb-cursor.so.0")
    if os.path.isfile(xcb_cursor):
        ctypes.CDLL(xcb_cursor, mode=ctypes.RTLD_GLOBAL)

import qtpy  # Necessary unused import to properly store the env variable

BASE_DIR = os.path.dirname(__file__)
