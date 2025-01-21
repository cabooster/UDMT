
import datetime
import functools
import importlib
import itertools
import json
import os
import pickle
import random

import cv2
import numpy as np
from collections import OrderedDict

from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm

from .environment import env_settings
import time
import cv2 as cv
from ..utils.visdom import Visdom
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ..utils.plotting import draw_figure, overlay_mask
# from ..utils.convert_vot_anno_to_rect import convert_vot_anno_to_rect
# from ltr.data.bounding_box_utils import masks_to_bboxes
# from .multi_object_wrapper import MultiObjectWrapper
# from pathlib import Path
import torch
# from pytracking import dcf
from similaritymeasures import similaritymeasures
from .missing_object_detect import missing_object_detect,refine_pos,missing_object_cal,refine_pos_for_loss
from .single_object_track import single_compensate
from .set_ini_value import set_ini_value
from udmt.gui import BASE_DIR
from PySide6 import QtGui,QtWidgets
# from ..run_mot_challenge import mot_eval
from .covert_to_MOTA_fuc import convert_format
_tracker_disp_colors = {1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 0, 0),
                        4: (255, 255, 255), 5: (0, 0, 0), 6: (0, 255, 128),
                        7: (123, 123, 123), 8: (255, 128, 0), 9: (128, 0, 255)}
##########################
# test_img_num = 2000 # !!32000 13040 15020 17760 29550 27980 16000 25000 42115 41770
animal_species = 1 # !! 1  2 fish 3 mix 4 flies
# data_fps = 60
#### visualization ####
save_flag = True
vis_flag = True
combine_flag = False
speed_up_flag = True
DEBUG_FLAG = False
##########################
refine_pos_flag = True #!!!!!!!!!!!!!!!!
##########################
# MOT_eval_flag = False
# MOT_target_size = 70 # fish 55 mice 70
# MOT_dataset_name = None # '12-bbnc-fish-60hz-8000f'
# down_sample_fg in missing !!!


# result_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'results' ,'trdimp', 'trdimp'))

use_xmem = True
quality_check_flag = False
global search_scale_gl
global target_sz_bias_gl
min_correct_time = 10000
corresponding_miss_num = 10000
min_miss_time = 10000
min_loss_time = 10000

# global target_sz_ini
# global target_sz_uniform
# global area_in_first_frame
# target_sz_ini = 50 #fish 35 micro 120 white mice 50
# target_sz_uniform = 125 #fish 78 micro 145 white mice 125
# area_in_first_frame = 3218 #fish 691 micro 9846 white mice 3218
##########################
def debug_print(message):
    if DEBUG_FLAG:
        print(message)
def save_to_json(file_path, data):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)
def trackerlist(name: str, parameter_name: str, run_ids = None, display_name: str = None):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, run_id, display_name) for run_id in run_ids]


class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, run_id: int = None, display_name: str = None):
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.run_id = run_id
        self.display_name = display_name

        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
            self.segmentation_dir = '{}/{}/{}'.format(env.segmentation_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
            self.segmentation_dir = '{}/{}/{}_{:03d}'.format(env.segmentation_path, self.name, self.parameter_name, self.run_id)

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tracker', self.name))
        if os.path.isdir(tracker_module_abspath):
            tracker_module = importlib.import_module('pytracking.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

        self.visdom = None


    def _init_visdom(self, visdom_info, debug):
        visdom_info = {} if visdom_info is None else visdom_info
        self.pause_mode = False
        self.step = False
        if debug > 0 and visdom_info.get('use_visdom', True):
            try:
                self.visdom = Visdom(debug, {'handler': self._visdom_ui_handler, 'win_id': 'Tracking'},
                                     visdom_info=visdom_info)

                # Show help
                help_text = 'You can pause/unpause the tracker by pressing ''space'' with the ''Tracking'' window ' \
                            'selected. During paused mode, you can track for one frame by pressing the right arrow key.' \
                            'To enable/disable plotting of a data block, tick/untick the corresponding entry in ' \
                            'block list.'
                self.visdom.register(help_text, 'text', 1, 'Help')
            except:
                time.sleep(0.5)
                debug_print('!!! WARNING: Visdom could not start, so using matplotlib visualization instead !!!\n'
                      '!!! Start Visdom in a separate terminal window by typing \'visdom\' !!!')

    def _visdom_ui_handler(self, data):
        if data['event_type'] == 'KeyPress':
            if data['key'] == ' ':
                self.pause_mode = not self.pause_mode

            elif data['key'] == 'ArrowRight' and self.pause_mode:
                self.step = True


    def create_tracker(self, params):
        tracker = self.tracker_class(params)
        tracker.visdom = self.visdom
        return tracker

    def run_sequence(self, seq, visualization=None, debug=None, visdom_info=None, search_scale = 2,target_sz_bias=0 , multiobj_mode=None,gui_param = None):

        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            visdom_info: Visdom info.
            multiobj_mode: Which mode to use for multiple objects.
        """
        global search_scale_gl
        global target_sz_bias_gl
        target_sz_bias_gl = target_sz_bias
        search_scale_gl = search_scale
        params = self.get_parameters(search_scale_gl,gui_param)
        # debug_print('params.search_area_scale:',params.search_area_scale)
        visualization_ = visualization

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        if visualization is None:
            if debug is None:
                visualization_ = getattr(params, 'visualization', False)
            else:
                visualization_ = True if debug else False

        params.visualization = visualization_
        params.debug = debug_

        self._init_visdom(visdom_info, debug_)
        if visualization_ and self.visdom is None:
            self.init_visualization()

        # Get init information
        init_info = seq.init_info()
        ####################
        seq.multiobj_mode = True
        seq.object_num = seq.init_bbox_all.shape[0]
        ####################

        is_single_object = not seq.multiobj_mode

        if multiobj_mode is None:
            multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default' or is_single_object:
            tracker = self.create_tracker(params)
        elif multiobj_mode == 'parallel':
            # tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom)
            tracker = []
            for i in range(seq.object_num):
                if animal_species == 3:
                    if i == 0:
                        params_0 = self.get_parameters(search_scale_gl,gui_param)
                        params_0.search_area_scale = 2
                        tracker_id = self.create_tracker(params_0)
                    else:
                        tracker_id = self.create_tracker(params)
                else:
                    tracker_id = self.create_tracker(params)
                tracker.append(tracker_id)
            tracker_compensate = self.create_tracker(params)

        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        output, early_stop_flag = self._track_sequence(tracker,  seq, init_info, tracker_compensate = tracker_compensate,gui_param = gui_param)
        return output, early_stop_flag

    def _track_sequence(self, tracker, seq, init_info, tracker_compensate = None,gui_param = None):

        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i
        # segmentation[i] is the segmentation mask for frame i (numpy array)

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i
        # segmentation[i] is the multi-label segmentation mask for frame i (numpy array)
        global min_correct_time,min_miss_time,min_loss_time,corresponding_miss_num
        if DEBUG_FLAG:
            print('start min_correct_time: ',min_correct_time)
            print('start min_miss_time: ',min_miss_time)
            print('start min_loss_time: ',min_loss_time)
        #
        #
        # print('###########',random.random())
        output = {'target_bbox': [],
                  'time': [],
                  'segmentation': []}

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        cross_times = 0
        image = self._read_image(seq.frames[0])

        # if tracker.params.visualization and self.visdom is None:
        #     self.visualize(image, init_info.get('init_bbox'))

        start_time = time.time()

        bg = gui_param['project_folder'] + '/tmp/' + gui_param['video_name']+'/masks'

        ##################
        # global target_sz_ini
        # global target_sz_uniform
        # global area_in_first_frame
        start_point_corr = seq.init_bbox_all
        target_sz_ini, target_sz_uniform, area_in_first_frame, kernel, area_mean = set_ini_value(animal_species,seq.frames,start_point_corr,seq.object_num,bg)#
        true_bias = target_sz_ini * target_sz_bias_gl#debug 250113
        debug_print(f'target_sz_bias_gl:{target_sz_bias_gl}')
        debug_print(f'target_sz_ini: {target_sz_ini}')
        debug_print(f'true_bias:{true_bias}' )
        target_sz_ini += true_bias
        debug_print(f'target_sz_ini final: {target_sz_ini}')
        ##################
        output_bb_list = [[]for i in range(seq.object_num)]
        score_map_list = [[]for i in range(seq.object_num)]
        target_pos_mul = [[]for i in range(seq.object_num)]
        target_sz_mul = [[]for i in range(seq.object_num)]
        result_save_file_list = []
        lis = np.arange(seq.object_num)
        pairs = list(itertools.combinations(list(lis),2))
        pairs_num = len(pairs)
        cross_list_mul = [[-100] for i in range(pairs_num)]
        not_far_list_mul = [[-100] for i in range(pairs_num)]
        loss_list_mul = [[-100] for i in range(seq.object_num)]
        swap_time_list = [[0]for i in range(pairs_num)]
        pairs_num = len(pairs)
        if seq.multiobj_mode:
            array_nx2 = seq.init_bbox_all
            array_nx4 = np.zeros((seq.object_num, 4))
            array_nx4[:, :2] = array_nx2
            seq.init_bbox_all = array_nx4
            for i in range(seq.object_num):
                # init_info['init_bbox'] = np.array([178.0, 253.0, 35.0, 35.0])
                init_info['init_bbox'] = seq.init_bbox_all[i]
                if ((animal_species == 3) & (i == 0)):
                        seq.init_bbox_all[i][2:] = np.array([55., 55.])
                else:
                    seq.init_bbox_all[i][2:] = np.array([target_sz_ini, target_sz_ini])
                # aaa = np.asarray(seq.init_bbox_all[i][:2])
                # target_pos_mul[i].append(aaa)
                init_info['init_bbox'][2:] = np.array([target_sz_ini, target_sz_ini])
                target_sz_mul[i].append(np.asarray(seq.init_bbox_all[i][2:]))
                out = tracker[i].initialize(image, init_info)
                output_bb_list[i].append(np.asarray(seq.init_bbox_all[i]))
                target_pos_mul[i].append(output_bb_list[i][0][:2] + output_bb_list[i][0][2:]*1/2)
        else:
            out = tracker.initialize(image, init_info)


        if out is None:
            out = {}

        prev_output = OrderedDict(out)

        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time,
                        'segmentation': init_info.get('init_mask')}

        _store_outputs(out, init_default)



        cross_judgement = True
        correct_flag = True
        visualize_tracklet_flag = True
        cross_limit = False
        break_flag = False
        early_stop_flag  = False


        point_size = 1
        if animal_species == 4:
            thickness = 1
            font_size = 1
        else:
            thickness = 4
            font_size = 1.5
        swap_time = 0
        correct_cross_times = 0
        correct_loss_times = 0
        not_found_times = 0
        area_left = 0
        pair_id = 0
        miss_target_time_sum = 0
        fine_detection_mode = False

        # point_color = (0, 0, 255)  # BGR
        rect_color_mul = []
        rect_color_mul.append((0, 255, 255))
        rect_color_mul.append((255, 255, 0))
        rect_color_mul.append((0, 0, 255))
        rect_color_mul.append((0, 255, 0))
        rect_color_mul.append((255, 0, 0))

        rect_color_mul.append((125, 0, 125))
        rect_color_mul.append((125, 255, 0))
        rect_color_mul.append((0, 0, 125))
        rect_color_mul.append((0, 255, 125))
        rect_color_mul.append((125, 0, 0))

        rect_color_mul.append((0, 125, 125))
        rect_color_mul.append((0, 125, 0))
        rect_color_mul.append((125, 125, 0))
        rect_color_mul.append((255, 125, 0))
        rect_color_mul.append((125, 125, 125))

        rect_color_mul.append((255, 60, 0))
        rect_color_mul.append((125, 125, 60))
        rect_color_mul.append((60, 125, 0))
        rect_color_mul.append((255, 125, 60))
        rect_color_mul.append((60, 125, 125))
        
        rect_color_mul.append((255, 60, 125))
        rect_color_mul.append((255, 255, 60))
        rect_color_mul.append((123, 10, 55))
        rect_color_mul.append((153, 87, 200))
        rect_color_mul.append((60, 60, 60))

        img_width = image.shape[0]
        img_height = image.shape[1]
        if gui_param['status_flag'] == 1:
            result_dir = gui_param['project_folder'] + '/tmp/' + gui_param['video_name'] + '/train_set_results'
        elif gui_param['status_flag'] == 2:
            result_dir = gui_param['project_folder'] + '/tmp/' + gui_param['video_name'] + '/test_set_results'
        else:
            result_dir = gui_param['project_folder'] + '/tracking-results/' + gui_param['video_name']

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        formatted_target_sz = "{:.2f}".format(target_sz_ini)
        img_pre_scale = gui_param['resize_factor']
        if gui_param['status_flag'] == 3:
            base_results_path = result_dir + '/label_' + seq.name + '_' + formatted_target_sz + '_' + str(search_scale_gl) + '_pre_scale_' + str(img_pre_scale) + '_' + datetime.datetime.now().strftime("%Y%m%d%H%M")
        else:
            base_results_path = result_dir + '/label_' + seq.name + '_' + formatted_target_sz + '_' + str(search_scale_gl) + '_pre_scale_' + str(img_pre_scale)
        if not os.path.exists(base_results_path):
            os.makedirs(base_results_path)
        if save_flag:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            formatted_target_sz = "{:.2f}".format(target_sz_ini)
            if gui_param['status_flag'] == 3:
                result_video_dir = result_dir + '/tmp-videos'
            else:
                result_video_dir = result_dir
            if not os.path.exists(result_video_dir):
                os.makedirs(result_video_dir)
            out_save = cv2.VideoWriter(result_video_dir + '/' + seq.name + '_' + formatted_target_sz + '_' + str(search_scale_gl) + '_pre_scale_'+ str(img_pre_scale) + '_eval_new.avi', fourcc, 30, (img_height, img_width), True)
            # out_compensate = None
            out_compensate = cv2.VideoWriter(result_video_dir + '/' + seq.name + '_' + formatted_target_sz + '_' + str(search_scale_gl)  + '_pre_scale_'+ str(img_pre_scale)+ '_backward_track_new.avi', fourcc, 30, (img_height, img_width), True)

            ##################### save result##############################

            base_results_path = base_results_path + '/' + seq.name
            for animal_id in range(seq.object_num):
                result_path = '{}_{}_new.txt'.format(base_results_path, animal_id)
                result_save_file_list.append(result_path)

                if os.path.exists(result_path):
                    os.remove(result_path)

        if gui_param['frame_rate'] >= 60:
            data_fps = 60
        elif gui_param['frame_rate'] >= 30:
            data_fps = 30
        else:
            data_fps = gui_param['frame_rate']
        if data_fps >= 60:
            down_sample_fg = 2
        else:
            down_sample_fg = 1
        if data_fps <= 20:#debug in 1209
            time_interval = 15
        elif data_fps <= 30:
            time_interval = 25
        else:
            time_interval = 30
        # if animal_species == 1:
        #     time_interval = 30 # bbnc fish 50
        # if animal_species == 2:
        #     time_interval = 50
        reverse_frame_list = seq.frames[::-1]
        first_start_time = time.time()
        test_img_num = gui_param['frame_num']
        if gui_param['status_flag'] == 1:
            search_period = True
        elif gui_param['status_flag'] == 2:
            search_period = True
        else:
            search_period = False
        # for frame_num, frame_path in enumerate(seq.frames[1:test_img_num], start=1):
        for frame_num, frame_path in enumerate(tqdm(seq.frames[1:test_img_num], desc="Processing Frames"), start=1):
            show_text = str(frame_num)
            # while True:
            #     if not self.pause_mode:
            #         break
            #     elif self.step:
            #         self.step = False
            #         break
            #     else:
            #         time.sleep(0.1)

            image = self._read_image(frame_path)

            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            im_show = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # debug_print(image.shape[1])
            update_model_flag = False
            correct_frame_flag = False
            s_list = []
            update_judge_flag_list = []
            sample_scales_list = []
            sample_pos_list = []
            test_x_list = []
            x_clf_list = []
            scale_ind_list = []
            im_patches_list = []
            patch_coords_list = []
            tt1 = time.time()
            if seq.multiobj_mode:
                if not combine_flag:
                    if speed_up_flag:
                        for i in range(seq.object_num):
                            im_patches, patch_coords = tracker[i].extract_patch(image)
                            im_patches_list.append(im_patches)
                            patch_coords_list.append(patch_coords)
                        backbone_feat_list = tracker[0].extract_backbone_parallel(im_patches_list,seq.object_num)
                for i in range(seq.object_num):
                    if not combine_flag:
                        if speed_up_flag:
                            out, score_map, s, update_judge_flag, sample_scales, sample_pos, test_x, x_clf, scale_ind = tracker[i].track_speed_up(backbone_feat_list[i],patch_coords_list[i], im_patches_list[i],i)

                        # out, score_map = tracker[i].track(image, i)
                        else:
                            out, score_map, s, update_judge_flag, sample_scales, sample_pos, test_x, x_clf, scale_ind = tracker[i].track(image, i)

                        if update_judge_flag == 'not_found':
                            not_found_times += 1
                        # out, score_map = tracker[i].track(image, i, frame_num,target_pos_mul,target_sz_mul,bg, seq.name,info,current_frame=frame_num,animal_num=seq.object_num,animal_species=animal_species,area_in_first_frame=area_in_first_frame,kernel=kernel)
                        s_list.append(s)
                        update_judge_flag_list.append(update_judge_flag)
                        sample_scales_list.append(sample_scales)
                        sample_pos_list.append(sample_pos)
                        test_x_list.append(test_x)
                        x_clf_list.append(x_clf)
                        scale_ind_list.append(scale_ind)
                    else:
                        out, score_map = tracker[i].track_combine(image, i, frame_num)
                    score_map_list[i].append(score_map.float().cpu())
                    output_bb_list[i].append(np.asarray(out['target_bbox']))
                    # aaa = np.asarray(out['target_bbox'])[:2]+(np.asarray(out['target_bbox'])[2:])*1/2
                    # bbb = np.asarray(tracker[i].pos[[1,0]])
                    # debug_print('aaa',aaa)
                    # debug_print('bbb',bbb)
                    target_pos_mul[i].append(np.asarray(tracker[i].pos[[1,0]]))
                    target_sz_mul[i].append(np.asarray(out['target_bbox'])[2:])

            else:
                # out, score_map = tracker.track(image, info)
                debug_print('')
            # if frame_num > 15:
            #     score_judge_list = []
            #     score_sz = torch.Tensor(list(score_map_list[0][0].shape[-2:]))
            #     score_center = (score_sz - 1)/2
            #     debug_print('score_judge_list: range', frame_num-6, 'to', frame_num)
            #     for mouse_id in range(2):
            #         score_ = score_map_list[mouse_id][frame_num-6:frame_num]
            #         max_score = []
            #         for score_id in score_:
            #             max_score1, max_disp1 = dcf.max2d(score_id)
            #             max_score.append(max_score1.numpy())
            #             # max_disp1 = max_disp1[0,...].float().cpu().view(-1)
            #             # target_disp1 = max_disp1 - score_center
            #         max_score = np.asarray(max_score)
            #         score_judge_list.append(max_score.min())
            tt2 = time.time()
            if refine_pos_flag: #### debug 1012
                if ((frame_num > 10) & (frame_num % 1 == 0)):
                    # if frame_num < 2280:
                         # debug_print('tracker[0].pos',tracker[0].pos)
                         # debug_print(target_pos_mul[0][-1])
                         target_pos_mul, miss_target_time, miss_target_id_list, target_refine_list = refine_pos(image,target_pos_mul,target_sz_mul,bg, seq.name,current_frame=frame_num,animal_num=seq.object_num,animal_species=animal_species,area_in_first_frame=area_in_first_frame,kernel=kernel,down_sample_fg=down_sample_fg,concave_flag=gui_param['is_concave'])
                         miss_target_time_sum += miss_target_time
                         if not combine_flag:
                             for animal_id in range(seq.object_num): # debug in 1101
                                 # debug_print('animal_id:',animal_id)
                                 # debug_print('tracker.pos before', tracker[animal_id].pos)
                                 pos_after_refine = torch.tensor([target_pos_mul[animal_id][-1][1],target_pos_mul[animal_id][-1][0]],dtype=torch.float32)
                                 tracker[animal_id].update_post(update_judge_flag_list[animal_id],pos_after_refine,sample_scales_list[animal_id],sample_pos_list[animal_id],test_x_list[animal_id], x_clf_list[animal_id], s_list[animal_id], scale_ind_list[animal_id] ,animal_id, frame_num)
                                 # debug_print('tracker.pos after', tracker[animal_id].pos)
                             update_model_flag = True
                         # else:
                         #     for animal_id in range(seq.object_num): # debug in 1101
                         #         # debug_print('tracker.pos before', tracker[animal_id].pos)
                         #         tracker[animal_id].pos = torch.tensor([target_pos_mul[animal_id][-1][1],target_pos_mul[animal_id][-1][0]],dtype=torch.float32)
                         #         # debug_print('tracker.pos after', tracker[animal_id].pos)

                         #########################
                         if len(miss_target_id_list)>0:
                            for id in miss_target_id_list:
                                loss_list_mul[id].append(frame_num)
                         for animal_id in range(seq.object_num):
                            if data_fps >= 120:
                                loss_detect_length = 80
                            elif data_fps >= 60:
                                loss_detect_length = 40
                            elif data_fps >= 30:
                                loss_detect_length = 30 # 20
                            else:
                                loss_detect_length = 20
                            if len(loss_list_mul[animal_id]) > loss_detect_length:
                              # debug_print('animal_id',animal_id, "\033[0;31m", 'loss more than 10 frames', "\033[0m")
                              # debug_print('loss_list_mul[animal_id]', loss_list_mul[animal_id])
                              check_add_one = lambda arr:functools.reduce(lambda x,y:(x+1==y if isinstance(x,int) else x[0] and x[1]+1==y, y),arr)[0]
                              continue_loss = (check_add_one(loss_list_mul[animal_id][-loss_detect_length:])) & (frame_num - loss_list_mul[animal_id][-1] == 0)
                              # if frame_num > 1117:
                              #     if animal_id == 1:
                              #         aa = check_add_one(loss_list_mul[animal_id][-10:])
                              #         bb = frame_num - loss_list_mul[animal_id][-1] == 1
                              #         debug_print(continue_loss)
                            else:
                              continue_loss = False
                            #continue_loss = False ###############
                            if continue_loss:

                                 debug_print(f'continue_loss of id : {animal_id}')
                                 debug_print(loss_list_mul[animal_id])
                                 target_pos_mul,refine_loss_flag = refine_pos_for_loss(image,target_pos_mul,target_sz_mul,bg, seq.name,current_frame=frame_num,animal_num=seq.object_num,animal_species=animal_species,area_in_first_frame=area_in_first_frame, target_refine_list=target_refine_list,loss_animal_id=animal_id,kernel=kernel,down_sample_fg=down_sample_fg)
                                 ########################
                                 if refine_loss_flag:
                                     del tracker[animal_id]
                                     params = self.get_parameters(search_scale_gl,gui_param)
                                     tracker_new = self.create_tracker(params)
                                     debug_print(f'params.search_area_scale: {params.search_area_scale}')
                                     init_info = seq.init_info()
                                     init_info['init_bbox'] = np.array([target_pos_mul[animal_id][-1][0], target_pos_mul[animal_id][-1][1], target_sz_mul[animal_id][-1][0], target_sz_mul[animal_id][-1][1]])
                                     out = tracker_new.initialize(image, init_info)
                                     tracker.insert(animal_id,tracker_new)
                                     correct_loss_times += 1
                                 ########################
                         #############################
            tt3 = time.time()
            if not combine_flag:
                if update_model_flag == False:
                    for animal_id in range(seq.object_num): # debug in 1104
                         # debug_print('animal_id norefine:',animal_id)
                         # debug_print('tracker.pos before', tracker[animal_id].pos)
                         pos_after_refine = torch.tensor([target_pos_mul[animal_id][-1][1],target_pos_mul[animal_id][-1][0]],dtype=torch.float32)
                         tracker[animal_id].update_post(update_judge_flag_list[animal_id],pos_after_refine,sample_scales_list[animal_id],sample_pos_list[animal_id],test_x_list[animal_id], x_clf_list[animal_id], s_list[animal_id], scale_ind_list[animal_id] ,animal_id, frame_num)
                         # debug_print('tracker.pos update:', tracker[animal_id].pos)
            tt4 = time.time()
            if ((animal_species == 1) | (animal_species == 3)):
                if data_fps <= 20:
                    judgement_gap = 2
                elif data_fps <= 30:
                    judgement_gap = 5
                else:
                    judgement_gap = 10
            else:
                judgement_gap = 10
            if cross_judgement == True:
                if correct_flag == True:
                   if frame_num > time_interval:
                       if frame_num % judgement_gap == 0: # speed up/ accelerate
                         for pair in pairs:
                               tracklet1 = np.array(target_pos_mul[pair[0]])
                               tracklet1 = (tracklet1[frame_num-time_interval:frame_num] ) / np.array([img_width,img_height])
                               tracklet2 = np.array(target_pos_mul[pair[1]])
                               tracklet2 = (tracklet2[frame_num-time_interval:frame_num] ) / np.array([img_width,img_height])
                               P=tracklet1
                               Q=tracklet2
                               dtw, d = similaritymeasures.dtw(P, Q)
                               # dh, ind1, ind2 = directed_hausdorff(P, Q)
                               dtw_yuzhi = 0.8
                               if dtw < dtw_yuzhi:
                                   debug_print(f'frame:, {frame_num}, pair:, {pair}, coincide??')
                                   #####
                                   judge_distance_interval = 10
                                   distance_coincide_object = np.asarray(target_pos_mul[pair[0]][frame_num-judge_distance_interval:frame_num]) - np.asarray(target_pos_mul[pair[1]][frame_num-judge_distance_interval:frame_num])
                                   avg_distance_coincide_object = distance_coincide_object.mean(axis=0)
                                   distance_coincide_combine = np.sqrt(np.sum(avg_distance_coincide_object ** 2))
                                   debug_print(f'distance_coincide_combine:{distance_coincide_combine}' )
                                   if distance_coincide_combine < output_bb_list[0][0][0]*0.8:
                                      debug_print(f'frame:, {frame_num},distance confirm!!')
                                      debug_print(f'frame: {frame_num},frame num confirm! Coincide!')
                                      show_text = show_text + ' Coincide!'
                                      coincide_flag = 1
                                      coincide_pair = pair
                                      pair_id = pairs.index(coincide_pair)
                                      if (coincide_flag == 1) & (correct_flag == 1) :
                                          time_step1 = int(data_fps * 2. / judgement_gap)
                                          if len(cross_list_mul[pair_id]) > time_step1:
                                              debug_print("continue_cross judge")
                                              ############
                                              debug_print(f'frame: {frame_num}')
                                              debug_print(f'cross_list_mul[pair_id]:,{cross_list_mul[pair_id][-10:]}')
                                              ############
                                              check_add_one = lambda arr:functools.reduce(lambda x,y:(x+judgement_gap==y if isinstance(x,int) else x[0] and x[1]+judgement_gap==y, y),arr)[0]
                                              continue_cross = (check_add_one(cross_list_mul[pair_id][-time_step1:])) & (frame_num - cross_list_mul[pair_id][-1] == judgement_gap)

                                          else:
                                              continue_cross = False

                                          time_step2 = int(data_fps * 2. / judgement_gap) + 10
                                          if len(cross_list_mul[pair_id]) > time_step2: #
                                              debug_print('cross_limit judge')
                                              check_add_one = lambda arr:functools.reduce(lambda x,y:(x+judgement_gap==y if isinstance(x,int) else x[0] and x[1]+judgement_gap==y, y),arr)[0]
                                              cross_limit = (check_add_one(cross_list_mul[pair_id][-time_step2:])) & (frame_num - cross_list_mul[pair_id][-1] == judgement_gap)
                                              debug_print(f'time_step2:,{time_step2} cross_list_mul[pair_id]:{cross_list_mul[pair_id][-time_step2:]}')
                                              if cross_limit == True:
                                                  debug_print('if cross_limit == True:')
                                                  if search_period:
                                                      debug_print('stop because of frequent crossing')
                                                      break_flag = True


                                          detect_x, detect_y = missing_object_detect(image, target_pos_mul,
                                                                                         target_sz_mul, bg, seq.name,
                                                                                         current_frame=frame_num,
                                                                                         animal_num=seq.object_num,
                                                                                         animal_species=animal_species,
                                                                                         fine_detection_mode=fine_detection_mode,
                                                                                         area_in_first_frame=area_in_first_frame,
                                                                                         kernel=kernel,area_mean = area_mean,down_sample_fg=down_sample_fg
                                                                                         )
                                          if continue_cross: # if continue_cross:
                                              debug_print("swap try")
                                              if swap_time_list[pair_id][0] % 2 == 0:
                                                  debug_print('swap switch')
                                                  detect_x, detect_y = missing_object_detect(image, target_pos_mul,
                                                                                             target_sz_mul, bg, seq.name,
                                                                                             current_frame=frame_num,
                                                                                             animal_num=seq.object_num,
                                                                                             animal_species=animal_species,
                                                                                             fine_detection_mode=fine_detection_mode,
                                                                                             area_in_first_frame=area_in_first_frame,
                                                                                             kernel=kernel,area_mean = area_mean,down_sample_fg=down_sample_fg,
                                                                                             area_rank=-3)
                                              if swap_time_list[pair_id][0] % 3 == 0:
                                                  if DEBUG_FLAG:
                                                    print("\033[0;31m", 'swap switch', "\033[0m")
                                                  detect_x, detect_y = missing_object_detect(image, target_pos_mul,
                                                                                             target_sz_mul, bg, seq.name,
                                                                                             current_frame=frame_num,
                                                                                             animal_num=seq.object_num,
                                                                                             animal_species=animal_species,
                                                                                             fine_detection_mode=fine_detection_mode,
                                                                                             area_in_first_frame=area_in_first_frame,
                                                                                             kernel=kernel,area_mean = area_mean,down_sample_fg=down_sample_fg,
                                                                                             area_rank=-4)
                                              if swap_time_list[pair_id][0] > 10:
                                                  fine_detection_mode = True #### debug in 1012
                                                  if DEBUG_FLAG:
                                                    print("\033[0;31m", 'fine_detection_mode = True', "\033[0m")
                                              swap_time_list[pair_id][0] += 1
                                              if DEBUG_FLAG:
                                                print('swap_time:', swap_time_list[pair_id][0], 'of', pairs[pair_id])

                                          compensate_start = np.array([detect_x, detect_y])
                                          distance_between_detect_cross_list = []
                                          for mouse_id in coincide_pair:
                                              delta_between_detect_cross = compensate_start - np.array([target_pos_mul[mouse_id][frame_num][0],target_pos_mul[mouse_id][frame_num][1]])
                                              distance_between_detect_cross = np.sqrt(np.sum(delta_between_detect_cross ** 2))
                                              distance_between_detect_cross_list.append(distance_between_detect_cross)
                                          distance_between_detect_cross = min(distance_between_detect_cross_list)
                                          if distance_between_detect_cross > target_sz_mul[0][0][0] * 0.8: # important # 10 mice 0.8 ori 1
                                              if DEBUG_FLAG:
                                                 print('compensate_start:',compensate_start)
                                              ######

                                              cross_list_mul[pair_id].append(frame_num)
                                              if DEBUG_FLAG:
                                                 print('pair_id:',pair_id,'cross_list_mul[pair_id]',cross_list_mul[pair_id])
                                              ###############################
                                              start_cross_id = None
                                              if len(not_far_list_mul[pair_id]) > 1:
                                                  if frame_num - not_far_list_mul[pair_id][-1] == judgement_gap:

                                                        for frame_id in range(len(not_far_list_mul[pair_id]) - 1, 0, -1):
                                                            current = not_far_list_mul[pair_id][frame_id]
                                                            previous_value = not_far_list_mul[pair_id][frame_id-1]

                                                            if current - previous_value > judgement_gap:
                                                                start_cross_id = current

                                                                break



                                              ####################################
                                              if DEBUG_FLAG:
                                                print('1:',coincide_pair[0],target_pos_mul[coincide_pair[0]][-1])
                                              target_pos_mul, target_sz_mul,  final_compensate_id = single_compensate(tracker_compensate, compensate_start, target_pos_mul, target_sz_mul, score_map_list, seq, image,reverse_frame_list, frame_num, coincide_pair, out_compensate, animal_species,start_cross_id, data_fps,cross_limit=cross_limit, video_name=seq.name,gui_param = gui_param)

                                              cross_times = cross_times + 1
                                              coincide_flag = 0
                                              ######
                                              # if cross_limit == True:
                                              #     print('if cross_limit == True:')
                                              #     final_compensate_id = coincide_pair[0]
                                              if final_compensate_id != 10000:
                                                  if DEBUG_FLAG:
                                                     print('2:',final_compensate_id,target_pos_mul[final_compensate_id][-1])
                                                  del tracker[final_compensate_id]

                                                  params = self.get_parameters(search_scale_gl,gui_param)
                                                  if DEBUG_FLAG:
                                                    print('params.search_area_scale:',params.search_area_scale)
                                                  tracker_new = self.create_tracker(params)
                                                  init_info = seq.init_info()
                                                  init_info['init_bbox'] = np.array([target_pos_mul[final_compensate_id][-1][0], target_pos_mul[final_compensate_id][-1][1], target_sz_mul[final_compensate_id][-1][0], target_sz_mul[final_compensate_id][-1][1]])
                                                  out = tracker_new.initialize(image, init_info)
                                                  tracker.insert(final_compensate_id,tracker_new)
                                                  # print('ok!')
                                                  swap_time_list[pair_id][0] = 0
                                                  if fine_detection_mode == True:
                                                      fine_detection_mode = False
                                                      if DEBUG_FLAG:
                                                        print("\033[0;31m", 'fine_detection_mode = False', "\033[0m")
                                                  if DEBUG_FLAG:
                                                     print("\033[0;31m", 'swap stop', "\033[0m")
                                                  correct_cross_times  += 1
                                                  correct_frame_flag = True
                                          else:
                                              not_far_list_mul[pair_id].append(frame_num)
                                              if DEBUG_FLAG:
                                                  print('distance_between_detect_cross:', distance_between_detect_cross)
                                                  print('%%%% not far enough! %%%%')
                                              coincide_flag = 0
            tt5 = time.time()
            #print('whole time ---->')
            #print(tt2-tt1)
            #print(tt3-tt2)
            #print(tt4-tt3)
            #print(tt5-tt4)
            #print(tt5-tt1)
            if ((animal_species == 1) | (animal_species == 3)):
                if data_fps <= 20:
                    swap_time_max = 10
                elif data_fps <= 30:
                    swap_time_max = 15
                else:
                    swap_time_max = 20
            else:
                swap_time_max = 30
            if swap_time_list[pair_id][0] > swap_time_max / judgement_gap: # too long no found & consider change params
                if DEBUG_FLAG:
                     print('frame->', frame_num)
                     print("\033[0;31m", 'swap_time_list more than ', swap_time_max, "\033[0m")
                if continue_cross:
                     if DEBUG_FLAG:
                        print('continue_cross of id ', coincide_pair[0], 'and', coincide_pair[1])
                     continue_cross_id = coincide_pair[0]
                     target_pos_mul, refine_loss_flag = refine_pos_for_loss(image,target_pos_mul,target_sz_mul,bg, seq.name,current_frame=frame_num,animal_num=seq.object_num,animal_species=animal_species,area_in_first_frame=area_in_first_frame, target_refine_list=target_refine_list,loss_animal_id=continue_cross_id,kernel=kernel,down_sample_fg=down_sample_fg)
                     ################################################
                     if refine_loss_flag:
                         del tracker[continue_cross_id]
                         params = self.get_parameters(search_scale_gl,gui_param)
                         tracker_new = self.create_tracker(params)
                         if DEBUG_FLAG:
                            print('params.search_area_scale:',params.search_area_scale)
                         init_info = seq.init_info()
                         init_info['init_bbox'] = np.array([target_pos_mul[continue_cross_id][-1][0], target_pos_mul[continue_cross_id][-1][1], target_sz_mul[continue_cross_id][-1][0], target_sz_mul[continue_cross_id][-1][1]])
                         out = tracker_new.initialize(image, init_info)
                         tracker.insert(continue_cross_id,tracker_new)
                         correct_loss_times += 1
                         swap_time_list[pair_id][0] = 0
                         correct_frame_flag = True
                     ################################################





            for animal_id in range(seq.object_num):
               result_file_hand = open(result_save_file_list[animal_id], 'a')
               target_size = target_sz_mul[animal_id][frame_num-1][0]
               result_file_hand.write(','.join(['{:.2f}'.format(i) for i in target_pos_mul[animal_id][frame_num-1] - target_size/2]) + ',' + ','.join(['{:.2f}'.format(i) for i in target_sz_mul[animal_id][frame_num-1]]) + '\n')
               result_file_hand.close()
            #### visualization ####

            if vis_flag:

                for animal_id in range(seq.object_num):
                    # print(output_bb_list[animal_id][frame_num][:2])
                    # print(target_pos_mul[animal_id][frame_num][0]-1/2*target_sz_mul[animal_id][frame_num][0],target_pos_mul[animal_id][frame_num][1]-1/2*target_sz_mul[animal_id][frame_num][1])
                    #
                    # print(output_bb_list[animal_id][frame_num][2:])
                    # print(target_sz_mul[animal_id][frame_num][0],target_sz_mul[animal_id][frame_num][1])

                    # target_pos = output_bb_list[animal_id][frame_num][:2]
                    # target_sz = output_bb_list[animal_id][frame_num][2:]
                    target_pos = np.array([target_pos_mul[animal_id][frame_num][0]-1/2*target_sz_mul[animal_id][frame_num][0],target_pos_mul[animal_id][frame_num][1]-1/2*target_sz_mul[animal_id][frame_num][1]])
                    target_sz =  np.array(target_sz_mul[animal_id][frame_num])

                    if animal_species == 4:
                        cv2.rectangle(im_show, (int(target_pos[0]), int(target_pos[1])),
                                    (int(target_pos[0] + target_sz[0]), int(target_pos[1] + target_sz[1])),
                                    rect_color_mul[animal_id], 1)
                    else:
                        cv2.rectangle(im_show, (int(target_pos[0]), int(target_pos[1])),
                                    (int(target_pos[0] + target_sz[0]), int(target_pos[1] + target_sz[1])),
                                    rect_color_mul[animal_id], 3)

                    if visualize_tracklet_flag:
                        if frame_num > time_interval:
                            for tracklets_id in range(time_interval):
                                cv2.circle(im_show, (int(target_pos_mul[animal_id][frame_num-tracklets_id][0]),int(target_pos_mul[animal_id][frame_num-tracklets_id][1])), point_size, rect_color_mul[animal_id], thickness)



                if correct_frame_flag:
                    show_text = show_text + 'correct !!'
                cv2.putText(im_show, show_text , (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                for animal_id in range(seq.object_num):
                    target_pos = output_bb_list[animal_id][frame_num][:2]
                    show_id = str(animal_id)
                    cv2.putText(im_show,show_id, (int(target_pos[0])+20, int(target_pos[1])), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, rect_color_mul[animal_id], 1, cv2.LINE_AA)
                    ##################################display
                    play_video_widget = gui_param['video_display_widget1']
                    display_width = play_video_widget.width()
                    display_height = play_video_widget.height()
                    resized_im_show = cv2.resize(im_show, (display_width, display_height),
                                                 interpolation=cv2.INTER_AREA)
                    height, width, channel = resized_im_show.shape
                    bytes_per_line = 3 * width
                    qt_image = QtGui.QImage(resized_im_show.data, width, height, bytes_per_line,QtGui.QImage.Format_RGB888)

                    play_video_widget.setPixmap(QtGui.QPixmap.fromImage(qt_image))
                    QtWidgets.QApplication.processEvents()
                    #############################
                # cv2.imshow(seq.name, im_show)
                # cv2.waitKey(1)
            if save_flag:
                out_save.write(im_show)

            # quality check param choose


            if frame_num < 199:#199
                not_found_times_per_animal = not_found_times/seq.object_num
                # print(frame_num,'not_found_times_per_animal:',not_found_times_per_animal)
                if not_found_times_per_animal > 20:# mice20 500
                    if DEBUG_FLAG:
                        print(frame_num,'stop because of frequent not_found_times_per_animal early:', not_found_times_per_animal)
                    break_flag = True
                    early_stop_flag = True
                miss_target_time_per_animal = miss_target_time_sum/seq.object_num
                # print(frame_num,'miss_target_time_sum:',miss_target_time_sum)
                if miss_target_time_per_animal > 30:#mice 30 500
                    if DEBUG_FLAG:
                        print(frame_num,'stop because of frequent missing target miss_target_time_per_animal early:', miss_target_time_per_animal)
                    break_flag = True
                    early_stop_flag = True


            if search_period:
                if correct_cross_times > min_correct_time:
                    if DEBUG_FLAG:
                        print('correct_cross_times', correct_cross_times)
                        print('min_correct_time', min_correct_time)
                        print('stop because of frequent correction')
                    break_flag = True
                elif correct_cross_times == min_correct_time:
                    if correct_loss_times > min_loss_time:
                        if DEBUG_FLAG:
                            print('correct_loss_times', correct_loss_times)
                            print('min_loss_time', min_loss_time)
                            print('stop because of loss time')
                        break_flag = True
                    else:
                       if miss_target_time_sum >= min_miss_time:
                           if DEBUG_FLAG:
                             print('correct_cross_times',correct_cross_times)
                             print('min_correct_time', min_correct_time)
                             print('stop because of frequent correction')
                           break_flag = True
                else:
                    break_flag = False
                    #else:
                    #    print('stop because of frequent correction but Do not stop because of loss not frequent')
                miss_target_time_per_animal = miss_target_time_sum/seq.object_num
                if miss_target_time_per_animal > 4:
                    if miss_target_time_sum > min_miss_time:
                        if DEBUG_FLAG:
                            print('miss_target_time_sum', miss_target_time_sum)
                            print('min_miss_time', min_miss_time)
                            print('stop because of frequent missing')
                        break_flag = True
            if break_flag:
                break
            # if frame_num % 199 == 0:
            #     print('\033[1;31mper 199 frame not_found_times/seq.object_num---------->\033[0m', not_found_times/seq.object_num)
            if search_period:
                if frame_num == test_img_num-1:
                    if correct_loss_times < min_loss_time:
                        if DEBUG_FLAG:
                            print('update min loss times...')
                            print('min_loss_time before', min_loss_time)
                        min_loss_time = correct_loss_times
                        if DEBUG_FLAG:
                            print('min_correct_time after', min_loss_time)
                            print('correct_cross_times', correct_loss_times)
                    if correct_cross_times < min_correct_time:
                        if DEBUG_FLAG:
                            print('update min correct value...')
                            print('min_correct_time before', min_correct_time)
                        min_correct_time = correct_cross_times
                        if DEBUG_FLAG:
                            print('min_correct_time after', min_correct_time)
                            print('correct_cross_times', correct_cross_times)
                    if miss_target_time_sum < min_miss_time:
                        if DEBUG_FLAG:
                            print('update min missing value...')
                            print('min_miss_time before', min_miss_time)
                        min_miss_time = miss_target_time_sum
                        if DEBUG_FLAG:
                            print('min_miss_time after', min_miss_time)
                            print('miss_target_time_sum', miss_target_time_sum)
            used_time = time.time() - first_start_time
            metric_dict = {
                "target_sz_bias": target_sz_bias_gl,
                "search_scale": str(search_scale_gl),
                "target_sz": formatted_target_sz,
                "correct_number": correct_cross_times,
                "miss_target_time_sum": miss_target_time_sum,
                "current_correct_loss_number": correct_loss_times,
                "used_time": used_time
            }
            if frame_num % 1999 == 0:
                if DEBUG_FLAG:
                    print(frame_num,':','\033[1;31mQuality check -----> \033[0m')
                area_left += missing_object_cal(image,target_pos_mul,target_sz_uniform,bg, seq.name,current_frame=frame_num,animal_num=seq.object_num,animal_species=animal_species,kernel=kernel,down_sample_fg=down_sample_fg)
                area_left_pre = area_left/area_in_first_frame
                if DEBUG_FLAG:
                    print('\033[1;31mCurrent correct number---------->\033[0m', correct_cross_times)
                    print('\033[1;31mLocation precision: miss target time sum---------->\033[0m', miss_target_time_sum)
                    print('\033[1;31mArea left---------->\033[0m', area_left_pre)
                    print('\033[1;31mCurrent correct loss number---------->\033[0m', correct_loss_times)
                not_found_times_per_animal = not_found_times/seq.object_num
                if DEBUG_FLAG:
                    print('\033[1;31mCurrent not_found_times_per_animal---------->\033[0m', not_found_times_per_animal)
                    print('\033[1;31mUsed time---------->\033[0m', used_time)
                ##############
                if quality_check_flag:
                    formatted_target_sz = "{:.2f}".format(target_sz_ini)
                    result_file_hand = open(seq.name + '_quality_check.txt', 'a')
                    result_file_hand.write('frame_num:'+str(frame_num)+' target sz: '+formatted_target_sz + ' search scale: ' + str(search_scale_gl)+'\n'+'correct number: '+str(correct_cross_times)+'\n'+'miss target time sum: '
                                           + str(miss_target_time_sum) + '\n'+'Area left: '+ str(area_left_pre) + '\n' + 'Current correct loss number: ' +str(correct_loss_times)+'\n'+ 'Current not_found_times_per_animal: ' +str(not_found_times_per_animal)+'\n'+'Used time: '+str(used_time)+'\n'+'\n')
                    result_file_hand.close()
                if search_period == False:
                    gui_param['evaluation_metric'].append(metric_dict)
                    json_file_path = gui_param['project_folder'] + '/tmp/' + gui_param[
                        'video_name'] + "/evaluation_metric.json"
                    save_to_json(json_file_path, gui_param['evaluation_metric'])
                    print(f"Updated JSON file at: {json_file_path}")

            ############################### GUI ##############################
            if search_period == True:
                if frame_num == gui_param['frame_num'] - 1:
                    gui_param['evaluation_metric'].append(metric_dict)
                    # Append the new dictionary to the list
                    if gui_param['status_flag'] == 1:
                        json_file_path = gui_param['project_folder'] + '/tmp/' + gui_param['video_name'] + "/evaluation_metric_for_train.json"
                    else:
                        json_file_path = gui_param['project_folder'] + '/tmp/' + gui_param['video_name'] + "/evaluation_metric_for_test.json"
                    # Save the updated list to the JSON file
                    save_to_json(json_file_path, gui_param['evaluation_metric'])
                    print(f"Updated JSON file at: {json_file_path}")
                    ###############################



            # print(tracker[0].training_samples[0][-1])
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})
        if save_flag:
            out_save.release()

        if not break_flag:
            if seq.multiobj_mode:
                # result_dir = 'E:/01-LYX/new-research/TransformerTrack-main/TransformerTrack-main/results/trdimp/trdimp/'
                formatted_target_sz = "{:.2f}".format(target_sz_ini)

                # base_results_path = result_dir + 'label_' + seq.name + '_' + formatted_target_sz+'_'+ str(search_scale_gl) + '/' + seq.name
                # base_results_path = os.path.join(result_dir, seq.name)
                ############################### save final result##############################
                test_path = result_dir
                for animal_id in range(seq.object_num):
                    result_path = '{}_{}_new.txt'.format(base_results_path, animal_id)
                    # result_save_file_list.append(result_path)
                    if os.path.exists(result_path):
                        os.remove(result_path)

                for animal_id in range(seq.object_num):
                   result_path = '{}_{}_new.txt'.format(base_results_path, animal_id)
                   with open(result_path, 'w') as f:
                       for jj in range(test_img_num):
                          # for x in target_pos_mul[mouse_id]:
                          #     f.write(','.join(['{:.2f}'.format(i) for i in x]) + ',' +','.join(['{:.2f}'.format(i) for i in x]) + '\n')
                          target_size = target_sz_mul[animal_id][jj][0]
                          f.write(','.join(['{:.2f}'.format(i) for i in target_pos_mul[animal_id][jj] - target_size/2]) + ',' + ','.join(['{:.2f}'.format(i) for i in target_sz_mul[animal_id][jj]]) + '\n')
            #############################################################debug in 240817
            # score_map_path = ('./score_map/' + 'score_map_' + seq.name + '_' + formatted_target_sz + '_' +
            #                   str(search_scale_gl) + '_' + datetime.datetime.now().strftime("%Y%m%d%H%M"))
            # if not os.path.exists(score_map_path):
            #    os.makedirs(score_map_path)
            # with open(score_map_path+'/score_map_list.pkl', 'wb') as file:
            #    pickle.dump(score_map_list, file)
            #############################################################################

            '''
            base_results_path = os.path.join(result_dir, seq.name)
            def save_bb(file, data):
                tracked_bb = np.array(data).astype(int)
                np.savetxt(file, tracked_bb, delimiter=',', fmt='%.4f')
            for i in range(seq.object_num):
                bbox_file = '{}_{}_new.txt'.format(base_results_path, i)
                save_bb(bbox_file, output_bb_list[i])
            '''
        for key in ['target_bbox', 'segmentation']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)
        # if MOT_eval_flag:
        #     global MOT_dataset_name
        #     if MOT_dataset_name == None:
        #         MOT_dataset_name = seq.name
        #
        #     convert_format(seq.name,MOT_dataset_name,seq.object_num,test_img_num,result_save_file_list,mot_target_size = MOT_target_size)
        #     metrics = mot_eval(dataset_name = MOT_dataset_name)
        #     result_file_hand = open(seq.name + '_quality_check.txt', 'a')
        #     result_file_hand.write('HOTA: '+metrics[0] + ' MOTA: ' + metrics[1] +' IDF1: '+ metrics[2]+ '\n' + '\n')
        #     result_file_hand.close()

        del tracker

        return output, early_stop_flag


    def get_parameters(self,search_scale,gui_param):
        """Get parameters."""
        param_module = importlib.import_module('pytracking.parameter.{}.{}'.format(self.name, self.parameter_name))
        params = param_module.parameters(gui_param)
        params.search_area_scale = search_scale
        return params


    def init_visualization(self):
        self.pause_mode = False
        self.fig, self.ax = plt.subplots(1)
        self.fig.canvas.mpl_connect('key_press_event', self.press)
        plt.tight_layout()


    def visualize(self, image, state, segmentation=None):
        self.ax.cla()
        self.ax.imshow(image)
        if segmentation is not None:
            self.ax.imshow(segmentation, alpha=0.5)

        if isinstance(state, (OrderedDict, dict)):
            boxes = [v for k, v in state.items()]
        else:
            boxes = (state,)

        for i, box in enumerate(boxes, start=1):
            col = _tracker_disp_colors[i]
            col = [float(c) / 255.0 for c in col]
            rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor=col, facecolor='none')
            self.ax.add_patch(rect)

        if getattr(self, 'gt_state', None) is not None:
            gt_state = self.gt_state
            rect = patches.Rectangle((gt_state[0], gt_state[1]), gt_state[2], gt_state[3], linewidth=1, edgecolor='g', facecolor='none')
            self.ax.add_patch(rect)
        self.ax.set_axis_off()
        self.ax.axis('equal')
        draw_figure(self.fig)

        if self.pause_mode:
            keypress = False
            while not keypress:
                keypress = plt.waitforbuttonpress()

    def reset_tracker(self):
        pass

    def press(self, event):
        if event.key == 'p':
            self.pause_mode = not self.pause_mode
            print("Switching pause mode!")
        elif event.key == 'r':
            self.reset_tracker()
            print("Resetting target pos to gt!")

    def _read_image(self, image_file,scale_factor=1):
        im = cv.imread(image_file)
        # 获取原始图像的高度和宽度
        original_height, original_width = im.shape[:2]

        # 计算新的高度和宽度
        new_height = int(original_height * scale_factor)
        new_width = int(original_width * scale_factor)

        # 使用cv2.resize函数调整图像大小
        resized_image = cv2.resize(im, (new_width, new_height))
        return cv.cvtColor(resized_image, cv.COLOR_BGR2RGB)

if __name__ == "__main__":
    print("This is file1.py and it's being run as a script.")


