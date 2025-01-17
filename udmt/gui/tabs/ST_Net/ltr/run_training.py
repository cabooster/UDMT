# run的文件
import os
import sys
import argparse
import importlib
import multiprocessing
import cv2 as cv
import torch.backends.cudnn

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

# import ltr.admin.settings as ws_settings
from udmt.gui.tabs.ST_Net.ltr.admin import settings as ws_settings


def run_training(train_module, train_name, cudnn_benchmark=True,run_training_params =None):
    """Run a train scripts in train_settings.
    args:
        train_module: Name of module in the "train_settings/" folder.
        train_name: Name of the train settings file.
        cudnn_benchmark: Use cudnn benchmark or not (default is True).
    """

    # This is needed to avoid strange crashes related to opencv
    cv.setNumThreads(0)

    torch.backends.cudnn.benchmark = cudnn_benchmark

    print('Training:  {}  {}'.format(train_module, train_name))

    settings = ws_settings.Settings(run_training_params)
    settings.module_name = train_module
    settings.script_name = train_name
    settings.project_path = 'ltr/{}/{}'.format(train_module, train_name)

    expr_module = importlib.import_module('ltr.train_settings.{}.{}'.format(train_module, train_name))
    expr_func = getattr(expr_module, 'run')

    expr_func(settings,run_training_params)


def run_training_process(run_training_params):
    multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
    parser.add_argument('--train_module', type=str, default='dimp', help='Name of module in the "train_settings/" folder.')#dimp
    parser.add_argument('--train_name', type=str, default='transformer_dimp', help='Name of the train settings file.') #dimp50 transformer_dimp
    parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='Set cudnn benchmark on (1) or off (0) (default is on).')

    args = parser.parse_args()

    run_training(args.train_module, args.train_name, args.cudnn_benchmark,run_training_params)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    run_training_process()
