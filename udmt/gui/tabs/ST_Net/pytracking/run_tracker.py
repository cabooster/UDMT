# 选择inference的数据集
import os
import sys
import argparse
import random
import time

import numpy as np
import torch

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from ..pytracking.evaluation import get_dataset
from ..pytracking.evaluation.running import run_dataset
from ..pytracking.evaluation import Tracker,Sequence


def read_txt_to_nparray(file_path):
    """
    读取文本文件并存储为 numpy 数组
    :param file_path: txt 文件路径
    :return: numpy 数组
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        # 使用 numpy 读取并解析
        data = np.loadtxt(file, delimiter=',')
    return data
def get_sorted_frame_paths(frames_path, extension=".jpg"):
    """
    获取指定目录下按数字排序的图像路径列表
    :param frames_path: 图像文件夹路径
    :param extension: 图像文件扩展名（默认 ".jpg"）
    :return: 按排序后的完整图像路径列表
    """
    # 获取所有指定扩展名的文件并排序
    frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(extension)]
    frame_list.sort(key=lambda f: int(f[:-len(extension)]))  # 按文件名中的数字排序

    # 构造完整路径
    frames_list = [os.path.join(frames_path, frame) for frame in frame_list]
    return frames_list
def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb', sequence=None, debug=0, threads=0,
                visdom_info=None,search_scale=2.0, target_sz_bias = 0, obj_id=None,gui_param = None):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
        visdom_info: Dict optionally containing 'use_visdom', 'server' and 'port' for Visdom visualization.
    """

    visdom_info = {} if visdom_info is None else visdom_info

    # dataset = get_dataset(dataset_name,obj_id)
    #
    # if sequence is not None:
    #     dataset = [dataset[sequence]]
    if gui_param['status_flag'] == 1:
        frames_path = gui_param['project_folder']+'/training-datasets/'+gui_param['video_name']+'/img'
    else:
        frames_path = gui_param['project_folder'] + '/tmp/' + gui_param['video_name']+'/extracted-images'
    frames_list = get_sorted_frame_paths(frames_path,".jpg")
    init_bbox = read_txt_to_nparray(gui_param['project_folder'] + '/tmp/' + gui_param['video_name']+'/extracted-images/start_pos_array.txt')
    init_bbox = init_bbox * gui_param['resize_factor']
    sequence_obj = Sequence(
        name=gui_param['video_name'],
        frames=frames_list,
        init_bbox_all=init_bbox,
        dataset='',
        ground_truth_rect=np.expand_dims(init_bbox[0], axis=0),
    )

    trackers = [Tracker(tracker_name, tracker_param, run_id)]

    early_stop_flag = run_dataset(sequence_obj, trackers, debug, threads, visdom_info=visdom_info,search_scale = search_scale,target_sz_bias = target_sz_bias,obj_id=obj_id,gui_param = gui_param)
    return early_stop_flag
def setup_seed(seed):
    np.random.seed(seed) # numpy 的设置
    random.seed(seed)  # python random module
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了使得hash随机化，使得实验可以复现
    torch.manual_seed(seed) # 为cpu设置随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed) # 如果使用多GPU为，所有GPU设置随机种子
        torch.backends.cudnn.benchmark = False # 设置为True，会使得cuDNN来衡量自己库里面的多个卷积算法的速度，然后选择其中最快的那个卷积算法。
        torch.backends.cudnn.deterministic = True # 每次返回的卷积算法将是确定的，即默认算法。如果配合上设置 Torch 的随机种子为固定值的话，
                                                    # 应该可以保证每次运行网络的时候相同输入的输出是固定的

def run_tracking(run_tracking_params):
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('--tracker_name', type=str, default='trdimp',help='Name of tracking method.')#trdimp
    parser.add_argument('--tracker_param', type=str, default='trdimp',help='Name of parameter file.')#trdimp
    parser.add_argument('--runid', type=int, default = None, help='The run id.')
    parser.add_argument('--dataset_name', type=str, default = 'got10k_test', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--sequence', type=str, default = '5-mice-full', help='Sequence number or name.')# 17-bbnc-fish 1-rat-2-mice-2-full 12-bbnc-fish 17-bbnc-fish 3-white-mice-30hz 5-white-mice-60hz 5-bbnc-mice-compress 5-mice-ori-speed
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--use_visdom', type=bool, default=True, help='Flag to enable visdom.')
    parser.add_argument('--visdom_server', type=str, default=None, help='Server for visdom.')
    parser.add_argument('--visdom_port', type=int, default=None, help='Port for visdom.')

    args = parser.parse_args()
    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence
    setup_seed(1024)

    iteration_time = 0
    start_time = time.time()
    for target_sz_bias in run_tracking_params['target_sz_bias_range']: # -15,0,5
        print('target_sz_bias percentage:', target_sz_bias)
        early_stop_flag_list = []
        for search_scale in run_tracking_params['search_scale_range']:
            print('search_scale:', search_scale)
            early_stop_flag = run_tracker(args.tracker_name, args.tracker_param, args.runid, args.dataset_name, seq_name, args.debug,
                            args.threads, {'use_visdom': args.use_visdom, 'server': args.visdom_server, 'port': args.visdom_port}, search_scale = search_scale, target_sz_bias = target_sz_bias, obj_id=0,gui_param = run_tracking_params)
            # print('early_stop_flag:', early_stop_flag)
            early_stop_flag_list.append(early_stop_flag)
        # print('target_sz_bias:', target_sz_bias,'early_stop_flag_list:', early_stop_flag_list)
        if early_stop_flag_list.count(False) > 0:
            iteration_time += 1
        print('iteration_time:', iteration_time)
        # if iteration_time == 3:
        #     break
    used_time = time.time() - start_time
    print(f"Time used: {used_time:.2f} seconds")
    '''
    for search_scale in np.arange(1.5, 3, 0.5):
        run_tracker(args.tracker_name, args.tracker_param, args.runid, args.dataset_name, seq_name, args.debug,
                        args.threads, {'use_visdom': args.use_visdom, 'server': args.visdom_server, 'port': args.visdom_port}, search_scale = search_scale, target_sz_bias = -15, obj_id=0)
    '''
    '''
    run_tracker(args.tracker_name, args.tracker_param, args.runid, args.dataset_name, seq_name, args.debug,
                        args.threads, {'use_visdom': args.use_visdom, 'server': args.visdom_server, 'port': args.visdom_port}, search_scale = 1.5, target_sz_bias = -10, obj_id=0)
'''

if __name__ == '__main__':
    run_tracking()
