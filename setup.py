# -*- coding: utf-8 -*-
#############################################
# File Name: setup.py
# Author: Yixin Li
# Mail: 20185414@stu.neu.edu.cn
# Created Time:  2025-01-10
#############################################
from setuptools import setup, find_packages



with open("README.md", 'r', encoding='utf-8') as f:
    readme = f.read()


setup(
    name="udmt_pip",
    version="0.0.8",
    description=("UDMT: Unsupervised Multi-animal Tracking for Quantitative Ethology"),
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Yixin Li, Xinyang Li",
    author_email="liyixin318@gmail.com",
    url="https://github.com/cabooster/UDMT",
    license="Non-Profit Open Software License 3.0",
    packages=find_packages(),
    install_requires=[ #"Cython==0.29.34",
        "dlclibrary",
        "filterpy>=1.4.4",
        "ruamel.yaml>=0.15.0",
        "imgaug>=0.4.0",
        "imageio-ffmpeg",
        "numba>=0.54",
        'matplotlib',
        "networkx>=2.6",
        "numpy>=1.18.5",
        "pandas",
        "opencv-python",
        "scikit-image>=0.17",
        "scikit-learn>=1.0",
        "scipy>=1.4",
        "statsmodels>=0.11",
        "tables>=3.7.0",
        "tensorpack>=0.11",
        "tf_slim>=1.1.0",
        "tqdm",
        "pyyaml",
        "Pillow>=7.1",
        "visdom",
        "similaritymeasures",
        "torchsummary",
        "jpeg4py",
        "progressbar2",
        "gdown",
        "pycocotools",
        "pyside6<6.3.2",
        "qdarkstyle==3.1",
        "napari-deeplabcut==0.2.0",
        "qt-material",
        "tensorflow==2.11.0",
        "segment-anything==1.0",
    ]
)
