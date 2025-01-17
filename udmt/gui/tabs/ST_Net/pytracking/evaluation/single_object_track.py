import importlib
import itertools
import os

import cv2
import numpy as np
from collections import OrderedDict

from scipy.spatial.distance import directed_hausdorff

# from .pytracking.evaluation.environment import env_settings
# import time
import cv2 as cv
# from pytracking.utils.visdom import Visdom
import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from pytracking.utils.plotting import draw_figure, overlay_mask
# from pytracking.utils.convert_vot_anno_to_rect import convert_vot_anno_to_rect
# from ltr.data.bounding_box_utils import masks_to_bboxes
# from pytracking.evaluation.multi_object_wrapper import MultiObjectWrapper
# from pathlib import Path
# from .. import dcf
import torch
from scipy.linalg import hankel
import scipy.linalg.interpolative as sli
from scipy.signal import find_peaks
from scipy.spatial.distance import directed_hausdorff
from similaritymeasures import similaritymeasures

from udmt.gui.tabs.ST_Net.pytracking.libs import dcf
from PySide6 import QtWidgets
from PySide6 import QtGui
hankel_flag = False
use_algorithm = True
vel_change_flag = True
score_map_flag = True
def _read_image(image_file: str):
        im = cv.imread(image_file)
        # 获取原始图像的高度和宽度
        original_height, original_width = im.shape[:2]

        # 计算新的高度和宽度
        new_height = int(original_height)
        new_width = int(original_width)

        # 使用cv2.resize函数调整图像大小
        resized_image = cv2.resize(im, (new_width, new_height))
        return cv.cvtColor(resized_image, cv.COLOR_BGR2RGB)
def IoU(rec1, rec2):
    """
    computing IoU
    rec1: (x0, y0, x1, y1)
    rec2: (x0, y0, x1, y1)
    :return: scala value of IoU
    """
    # computing area of each rectangle
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])
    # print(top_line, left_line, right_line, bottom_line)

    # judge if there is an intersect area
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0
def optimize_tracklet(target_pos_mul, target_sz_mul,target_pos_compensate, target_sz_compensate,final_compensate_id,compensate_start_id):

    target_pos_mul[final_compensate_id][compensate_start_id:] = target_pos_compensate[0:]
    target_sz_mul[final_compensate_id][compensate_start_id:] = target_sz_compensate[0:]

    return target_pos_mul, target_sz_mul
def hankelize(xy):
        ncols = int(np.ceil(len(xy) * 2 / 3))
        nrows = len(xy) - ncols + 1
        mat = np.empty((2 * nrows, ncols))
        mat[::2] = hankel(xy[:nrows, 0], xy[-ncols:, 0])
        mat[1::2] = hankel(xy[:nrows, 1], xy[-ncols:, 1])
        return mat
def estimate_rank(tracklet, tol):
        """
        Estimate the (low) rank of a noisy matrix via
        hard thresholding of singular values.

        See Gavish & Donoho, 2013.
            The optimal hard threshold for singular values is 4/sqrt(3)
        """
        mat = hankelize(tracklet)
        # nrows, ncols = mat.shape
        # beta = nrows / ncols
        # omega = 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43
        _, s, _ = sli.svd(mat, min(10, min(mat.shape)))
        # return np.argmin(s > omega * np.median(s))
        eigen = s ** 2
        diff = np.abs(np.diff(eigen / eigen[0]))
        return np.argmin(diff > tol)
def dynamic_similarity_with(tracklet1, other_tracklet, tol=0.001):
        """
        Evaluate the complexity of the tracklets' underlying dynamics
        from the rank of their Hankel matrices, and assess whether
        they originate from the same track. The idea is that if two
        tracklets are part of the same track, they can be approximated
        by a low order regressor. Conversely, tracklets belonging to
        different tracks will require a higher order regressor.

        See Dicle et al., 2013.
            The Way They Move: Tracking Multiple Targets with Similar Appearance.
        """
        # TODO Add missing data imputation
        # joint_tracklet = tracklet1 + other_tracklet
        joint_tracklet = np.concatenate((tracklet1, other_tracklet))
        joint_rank = estimate_rank(joint_tracklet,tol)
        rank1 = estimate_rank(tracklet1,tol)
        rank2 = estimate_rank(other_tracklet,tol)
        return (rank1 + rank2) / joint_rank - 1
def dynamic_dissimilarity_with(tracklet1, other_tracklet):
        """
        Compute a dissimilarity score between Hankelets.
        This metric efficiently captures the degree of alignment of
        the subspaces spanned by the columns of both matrices.

        See Li et al., 2012.
            Cross-view Activity Recognition using Hankelets.
        """
        hk1 = hankelize(tracklet1)
        # hk1 = tracklet1.to_hankelet()
        hk1 /= np.linalg.norm(hk1)

        hk2 = hankelize(other_tracklet)
        # hk2 = other_tracklet.to_hankelet()
        hk2 /= np.linalg.norm(hk2)
        min_shape = min(hk1.shape + hk2.shape)
        temp1 = (hk1 @ hk1.T)[:min_shape, :min_shape]
        temp2 = (hk2 @ hk2.T)[:min_shape, :min_shape]
        return 2 - np.linalg.norm(temp1 + temp2)
def match_compensate_target(target_pos_mul, target_sz_mul,score_map_list,target_pos_compensate,target_sz_compensate,f,coincide_pair,mouse_num,image_files,target_pos_compensate_full,animal_species,start_cross_id,data_fps,IoU_min_frame):
    # cal by final_result_display.py
    # min_x = 53
    # max_x = 577
    # min_y = 63
    # max_y = 387
    # min_x = 19
    # max_x = 862
    # min_y = 25
    # max_y = 609
    print('match_compensate_target')

    point_size = 1
    thickness = 4
    rect_color_mul = []
    rect_color_mul.append((0, 255, 255))
    rect_color_mul.append((255, 255, 0))
    rect_color_mul.append((0, 0, 255))
    rect_color_mul.append((0, 255, 0))
    rect_color_mul.append((255, 0, 0))

    rect_color_mul.append((0, 255, 255))
    rect_color_mul.append((255, 255, 0))
    rect_color_mul.append((0, 0, 255))
    rect_color_mul.append((0, 255, 0))
    rect_color_mul.append((255, 0, 0))

    voting_arr = np.zeros(mouse_num)
    target_pos_compensate.reverse()
    target_sz_compensate.reverse()
    target_pos_compensate = np.asarray(target_pos_compensate)
    target_sz_compensate = np.asarray(target_sz_compensate)

    target_pos_compensate_full.reverse()
    target_pos_compensate_full = np.asarray(target_pos_compensate_full)
    f_pos_in_full = target_pos_compensate_full.shape[0] - target_pos_compensate.shape[0]

    ###############

    target_pos_mul = np.asarray(target_pos_mul)
    compensate_frame_length = target_pos_compensate.shape[0]
    print('compensate_frame_length:',compensate_frame_length)
    ###### calc_IoU
    # IoU_cross = []
    # box_lost = [int(target_pos_compensate[0][0] - target_sz_compensate[0][0] / 2), int(target_pos_compensate[0][1] - target_sz_compensate[0][1] / 2),int(target_pos_compensate[0][0] + target_sz_compensate[0][0] / 2), int(target_pos_compensate[0][1] + target_sz_compensate[0][1] / 2)]
    # # print(f)
    # # print('box_lost for calc_IoU:',box_lost)
    # for mouse_id in coincide_pair:
    #     box_coincide = [int(target_pos_mul[mouse_id][f][0] - target_sz_mul[mouse_id][f][0] / 2), int(target_pos_mul[mouse_id][f][1] - target_sz_mul[mouse_id][f][1] / 2),int(target_pos_mul[mouse_id][f][0] + target_sz_mul[mouse_id][f][0] / 2), int(target_pos_mul[mouse_id][f][1] + target_sz_mul[mouse_id][f][1] / 2)]
    #     IoU_cross.append(IoU(box_lost, box_coincide))
    # max_IoU = max(IoU_cross)
    # max_IoU_id = IoU_cross.index(max_IoU)
    # print('IoU_cross-------->',IoU_cross)
    # print('max_IoU:',max_IoU,'max_ID:',coincide_pair[max_IoU_id])
    # voting_arr[coincide_pair[max_IoU_id]] += 1

    # if hankel_flag:
    #     hankel_dis_mul = []
    #     hankel_sim_mul = []
    #     hankle_compensate_frame_length = 15
    #     # if slow_species == 2:
    #     #     hankle_compensate_frame_length = 20
    #
    #     if compensate_frame_length > hankle_compensate_frame_length:
    #         trackk2 = target_pos_compensate[:hankle_compensate_frame_length]
    #     else:
    #         print('compensate_frame_length is not enough for hankle')
    #         trackk2 = target_pos_compensate
    #     # vel2 = np.diff(trackk2, axis=0)
    #     # vel2 /= np.linalg.norm(vel2, axis=1, keepdims=True)
    #     # rank2 = estimate_rank(vel2)
    #     # print(rank2)
    #
    #     start_id = max(f-25, 0)
    #     end_id = f-10
    #     # for frame_id in abnormal_vel_frame:
    #     #     if (frame_id < end_id) & (frame_id > start_id-10):
    #     #          end_id = frame_id + 1
    #     #          start_id = max(end_id-15, 0)
    #     #          print('correct interval in hankel!!!!')
    #     #          break
    #     print('f:', f)
    #     # if slow_species == 2:
    #     #     start_id -= 5
    #     #     end_id -= 5
    #     print('range', start_id, 'to', end_id)
    #     for mouse_id in coincide_pair:
    #         trackk1 = target_pos_mul[mouse_id][start_id:end_id]
    #         trackk1 = np.asarray(trackk1)
    #         hal_dis = dynamic_dissimilarity_with(trackk1, trackk2)
    #         hankel_dis_mul.append(hal_dis)
    #         hal_sim = dynamic_similarity_with(trackk1, trackk2)
    #         hankel_sim_mul.append(hal_sim)
    #
    #     print('hal_dis:', hankel_dis_mul)
    #     print('hal_sim:', hankel_sim_mul)
    #     min_dis = min(hankel_dis_mul)
    #     min_dis_id = hankel_dis_mul.index(min_dis)
    #     print('min_dis_ID:',coincide_pair[min_dis_id])
    #     # voting_arr[coincide_pair[min_dis_id]] += 1

    if animal_species == 2:
        start_id_new = f - 20  # FISH
    else:
        if data_fps >= 60:
            start_id_new = f - 35 # 25
        else:
            start_id_new = f - 20
    if start_cross_id != None:
        if start_cross_id < start_id_new:
            start_id_new = start_cross_id - 10
            print('start_id_new refresh algorithm 1!!')
    if IoU_min_frame != None:
        print('IoU_min_frame:',IoU_min_frame,'start_id_new:',start_id_new)
        if IoU_min_frame < start_id_new:
            start_id_new = IoU_min_frame
            print('start_id_new refresh algorithm 2!!')
    if ((animal_species == 1) | (animal_species == 3)):
        if data_fps >= 60:
            end_id_new_vel = f + 10 # MICE
        else:
            end_id_new_vel = f + 5 # MICE
    else:
        end_id_new_vel = f + 10 # FISH

    if vel_change_flag:
        vel_change_list = []
        vel_max_pre_list =[]
        if start_id_new < 0:
            start_id_new = 0
        print('vel_change_flag: range', start_id_new, 'to', end_id_new_vel)
        for mouse_id in coincide_pair:
            trackk_ = target_pos_mul[mouse_id][start_id_new:end_id_new_vel]
            vel_trackk = np.diff(trackk_, axis=0)
            sum_vel = []
            for vel_id in vel_trackk:
                sum_vel.append(np.sqrt(np.sum(vel_id ** 2)))
            sum_vel = np.asarray(sum_vel)
            std_ = sum_vel.std()
            max_ = sum_vel.max()
            # median_ = np.median(sum_vel)
            mean_ = sum_vel.mean()
            vel_change_list.append(std_)

            vel_max_pre_list.append(int(max_/(mean_+0.000001)))

        print('vel_change:', vel_change_list)
        max_change = max(vel_change_list)
        max_change_id = vel_change_list.index(max_change)
        print("\033[0;31m", 'max_vel_change_ID std:',coincide_pair[max_change_id], "\033[0m")
        voting_arr[coincide_pair[max_change_id]] += 1

        print('vel_max_pre:', vel_max_pre_list)
        vel_max_pre = max(vel_max_pre_list)
        max_change_id = vel_max_pre_list.index(vel_max_pre)
        print("\033[0;31m", 'max_vel_change_ID max_/mean_:',coincide_pair[max_change_id], "\033[0m")
        voting_arr[coincide_pair[max_change_id]] += 1
    end_id_new_sc = f+10
    if score_map_flag:
        score_judge_list = []
        score_sz = torch.Tensor(list(score_map_list[0][0].shape[-2:]))
        score_center = (score_sz - 1)/2
        print('score_judge_list: range', start_id_new, 'to', end_id_new_sc)
        for mouse_id in coincide_pair:
            score_ = score_map_list[mouse_id][start_id_new:end_id_new_sc]
            max_score = []
            for score_id in score_:
                max_score1, max_disp1 = dcf.max2d(score_id)
                max_score.append(max_score1.numpy())
                # max_disp1 = max_disp1[0,...].float().cpu().view(-1)
                # target_disp1 = max_disp1 - score_center
            max_score = np.asarray(max_score)
            score_judge_list.append(max_score.min())
        print('score_judge_list:', score_judge_list)
        min_score = min(score_judge_list)
        min_score_id = score_judge_list.index(min_score)
        print("\033[0;31m", 'min_score_ID:', coincide_pair[min_score_id], "\033[0m")
        voting_arr[coincide_pair[min_score_id]] += 1

    # return final_compensate_id
    # final_compensate_id = coincide_pair[0]
    '''
        import random
        random_number = random.randint(0, 1)
        print('random_number',random_number)
        final_compensate_id = coincide_pair[random_number]
        print('random id',final_compensate_id)
        '''
    final_compensate_id = np.argmax(voting_arr)
    print("\033[0;31m", 'final_compensate_id:', final_compensate_id, "\033[0m")
    return final_compensate_id

def match_compensate_target_simple(target_pos_mul, score_map_list,f,coincide_pair,mouse_num,animal_species):
    # cal by final_result_display.py

    print('match_compensate_target')

    point_size = 1
    thickness = 4
    rect_color_mul = []
    rect_color_mul.append((0, 255, 255))
    rect_color_mul.append((255, 255, 0))
    rect_color_mul.append((0, 0, 255))
    rect_color_mul.append((0, 255, 0))
    rect_color_mul.append((255, 0, 0))

    rect_color_mul.append((0, 255, 255))
    rect_color_mul.append((255, 255, 0))
    rect_color_mul.append((0, 0, 255))
    rect_color_mul.append((0, 255, 0))
    rect_color_mul.append((255, 0, 0))

    voting_arr = np.zeros(mouse_num)


    target_pos_mul = np.asarray(target_pos_mul)


    start_id_new = f - 20 # MICE


    end_id_new = f+10
    if vel_change_flag:
        vel_change_list = []
        vel_max_pre_list =[]
        print('vel_change_flag: range', start_id_new, 'to', end_id_new)
        for mouse_id in coincide_pair:
            trackk_ = target_pos_mul[mouse_id][start_id_new:end_id_new]
            vel_trackk = np.diff(trackk_, axis=0)
            sum_vel = []
            for vel_id in vel_trackk:
                sum_vel.append(np.sqrt(np.sum(vel_id ** 2)))
            sum_vel = np.asarray(sum_vel)
            std_ = sum_vel.std()
            max_ = sum_vel.max()
            # median_ = np.median(sum_vel)
            mean_ = sum_vel.mean()
            vel_change_list.append(std_)

            vel_max_pre_list.append(int(max_/(mean_+0.000001)))

        print('vel_change:', vel_change_list)
        max_change = max(vel_change_list)
        max_change_id = vel_change_list.index(max_change)
        print("\033[0;31m", 'max_vel_change_ID std:',coincide_pair[max_change_id], "\033[0m")
        voting_arr[coincide_pair[max_change_id]] += 1

        print('vel_max_pre:', vel_max_pre_list)
        vel_max_pre = max(vel_max_pre_list)
        max_change_id = vel_max_pre_list.index(vel_max_pre)
        print("\033[0;31m", 'max_vel_change_ID max_/mean_:',coincide_pair[max_change_id], "\033[0m")
        voting_arr[coincide_pair[max_change_id]] += 1

    if score_map_flag:
        score_judge_list = []
        score_sz = torch.Tensor(list(score_map_list[0][0].shape[-2:]))
        score_center = (score_sz - 1)/2
        print('score_judge_list: range', start_id_new, 'to', end_id_new)
        for mouse_id in coincide_pair:
            score_ = score_map_list[mouse_id][start_id_new:end_id_new]
            max_score = []
            for score_id in score_:
                max_score1, max_disp1 = dcf.max2d(score_id)
                max_score.append(max_score1.numpy())
                # max_disp1 = max_disp1[0,...].float().cpu().view(-1)
                # target_disp1 = max_disp1 - score_center
            max_score = np.asarray(max_score)
            score_judge_list.append(max_score.min())
        print('score_judge_list:', score_judge_list)
        min_score = min(score_judge_list)
        min_score_id = score_judge_list.index(min_score)
        print("\033[0;31m", 'min_score_ID:', coincide_pair[min_score_id], "\033[0m")
        voting_arr[coincide_pair[min_score_id]] += 1

    # return final_compensate_id



    # final_compensate_id = coincide_pair[0]

    final_compensate_id = np.argmax(voting_arr)
    print("\033[0;31m", 'final_compensate_id:', final_compensate_id, "\033[0m")
    return final_compensate_id




def single_compensate(tracker_compensate, compensate_start, target_pos_mul, target_sz_mul, score_map_list, seq, image,reverse_frame_list, current_f, coincide_pair, out_compensate, animal_species,start_cross_id,data_fps,cross_limit=False, video_name=None,gui_param = None):
    vis_flag = True
    ######################
    save_flag = True
    ###################
    cross_point_detect_debug = False
    IoU_cross_flag = False
    init_info = seq.init_info()
    init_info['init_bbox'] = np.array([compensate_start[0], compensate_start[1], target_sz_mul[coincide_pair[0]][0][0], target_sz_mul[coincide_pair[0]][0][1]])
    target_sz_compensate = []
    target_pos_compensate = []
    score_map_compensate = []


    target_sz = np.array([target_sz_mul[coincide_pair[0]][0][0], target_sz_mul[coincide_pair[0]][0][1]])
    target_pos = compensate_start
    target_sz_compensate.append(target_sz)
    target_pos_compensate.append(target_pos)
    IoU_cross_between_pair = []
    IoU_cross_mul = [[]for a in range(seq.object_num)]
    distance_cross_mul = [[]for a in range(seq.object_num)]
    out = tracker_compensate.initialize(image, init_info)
    final_compensate_id = 10000
    if data_fps <= 30:
        backward_frame_num_min = 100  # change from 80 important 2.5special
        backward_frame_num_max = 130
    if (data_fps > 30) & (data_fps<=60):
        backward_frame_num_min = 100  # change from 60 important
        backward_frame_num_max = 200
    if data_fps>60:
        backward_frame_num_min = 130  
        backward_frame_num_max = 270

    for f in range(current_f-1, 0, -1):
        # print(seq.frames[f])
        image = _read_image(seq.frames[f])
        im_show = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        info = seq.frame_info(f)
        # out, score_compensate = tracker_compensate.track(image)
        out, score_compensate = tracker_compensate.track_combine(image)
        # print(f, out)
        target_sz = np.asarray(out['target_bbox'])[2:]
        target_pos = np.asarray(out['target_bbox'])[:2]+(np.asarray(out['target_bbox'])[2:])*1/2
        score_map_compensate.append(score_compensate)
        target_sz_compensate.append(target_sz)
        target_pos_compensate.append(target_pos)
        cv2.rectangle(im_show, (int(target_pos[0] - target_sz[0] / 2), int(target_pos[1] - target_sz[1] / 2)),
                          (int(target_pos[0] + target_sz[0] / 2), int(target_pos[1] + target_sz[1] / 2)),
                          (0, 255, 0), 3)
        point_size = 1
        point_color = (0, 0, 255)  # BGR
        thickness = 4
        cv2.circle(im_show, (int(target_pos[0]),int(target_pos[1])), point_size, point_color, thickness)
        cv2.putText(im_show, str(f), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        ##############################################################################################################

        if vis_flag:
            ##################################display
            play_video_widget = gui_param['video_display_widget2']
            display_width = play_video_widget.width()
            display_height = play_video_widget.height()
            resized_im_show = cv2.resize(im_show, (display_width, display_height),
                                         interpolation=cv2.INTER_AREA)
            height, width, channel = resized_im_show.shape
            bytes_per_line = 3 * width
            qt_image = QtGui.QImage(resized_im_show.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            # 更新 QLabel 显示
            play_video_widget.setPixmap(QtGui.QPixmap.fromImage(qt_image))
            QtWidgets.QApplication.processEvents()
            ##################################
            # cv2.imshow('coincide track back', im_show)
            # cv2.waitKey(10)
        if save_flag:
            if out_compensate != None:
                out_compensate.write(im_show)

        box_lost = [int(target_pos[0] - target_sz[0] / 2), int(target_pos[1] - target_sz[1] / 2),int(target_pos[0] + target_sz[0] / 2), int(target_pos[1] + target_sz[1] / 2)]

        for animal_id in coincide_pair:
            distance_cross = target_pos - target_pos_mul[animal_id][f]
            distance_cross_combine = np.sqrt(np.sum(distance_cross ** 2))

            box_coincide = [int(target_pos_mul[animal_id][f][0] - target_sz_mul[animal_id][f][0] / 2), int(target_pos_mul[animal_id][f][1] - target_sz_mul[animal_id][f][1] / 2),int(target_pos_mul[animal_id][f][0] + target_sz_mul[animal_id][f][0] / 2), int(target_pos_mul[animal_id][f][1] + target_sz_mul[animal_id][f][1] / 2)]
            IoU_cross = IoU(box_lost, box_coincide)
            ####################
            IoU_cross_mul[animal_id].append(IoU_cross)
            distance_cross_mul[animal_id].append(distance_cross_combine)
            ####################
        #####################0927
        animal_id = coincide_pair[0]
        box_coincide1 = [int(target_pos_mul[animal_id][f][0] - target_sz_mul[animal_id][f][0] / 2), int(target_pos_mul[animal_id][f][1] - target_sz_mul[animal_id][f][1] / 2),int(target_pos_mul[animal_id][f][0] + target_sz_mul[animal_id][f][0] / 2), int(target_pos_mul[animal_id][f][1] + target_sz_mul[animal_id][f][1] / 2)]
        animal_id = coincide_pair[1]
        box_coincide2 = [int(target_pos_mul[animal_id][f][0] - target_sz_mul[animal_id][f][0] / 2), int(target_pos_mul[animal_id][f][1] - target_sz_mul[animal_id][f][1] / 2),int(target_pos_mul[animal_id][f][0] + target_sz_mul[animal_id][f][0] / 2), int(target_pos_mul[animal_id][f][1] + target_sz_mul[animal_id][f][1] / 2)]
        IoU_cross_coincide = IoU(box_coincide1, box_coincide2)
        IoU_cross_between_pair.append(IoU_cross_coincide)
        #####################
        # speed up reverse

        if f < current_f-backward_frame_num_min:
            if IoU_cross_coincide < 0.1:
                print('frame:',f,'IoU_cross_coincide < 0.1')
                break
            # else:
            #     print('frame:',f,'IoU_cross_coincide > 0.1','IOU:',IoU_cross_coincide)
        if f < current_f-backward_frame_num_max:
                break

    if cross_point_detect_debug:
        plt.plot(np.asarray(distance_cross_mul[coincide_pair[0]]))
        plt.plot(np.asarray(distance_cross_mul[coincide_pair[1]]))


    mins_0, _ =find_peaks(np.asarray(distance_cross_mul[coincide_pair[0]])*-1,width=10)  # 纵轴局部最低点
    mins_1, _ =find_peaks(np.asarray(distance_cross_mul[coincide_pair[1]])*-1,width=10)  # 纵轴局部最低点

    if cross_point_detect_debug:
       plt.plot(mins_0, np.asarray(distance_cross_mul[coincide_pair[0]])[mins_0], 'x', label='mins')
       plt.plot(mins_1, np.asarray(distance_cross_mul[coincide_pair[1]])[mins_1], '*', label='mins')
       plt.savefig('distance_cross.png')
       plt.show()

    distance_cross_extrem = 10000
    compensate_limit = 0
    if (mins_0.size > 0):
        for dmc0 in mins_0:
            if distance_cross_mul[coincide_pair[0]][dmc0] < 100:
                distance_cross_extrem = dmc0
    if (mins_1.size > 0):
        for dmc1 in mins_1:
           if ((distance_cross_mul[coincide_pair[1]][dmc1] < 100) & (dmc1 < distance_cross_extrem)):
                distance_cross_extrem = dmc1

    print('distance_cross_extrem:',distance_cross_extrem)
    if distance_cross_extrem == 10000:
        print('no distance_cross_extrem!!!!!!!!!!!')

    if cross_point_detect_debug:
        plt.plot(np.asarray(IoU_cross_mul[coincide_pair[0]]))
        plt.plot(np.asarray(IoU_cross_mul[coincide_pair[1]]))
        plt.savefig('IoU_cross.jpg')
        plt.show()
    ###################################################cross_point_detect#############################################
    print('debug in 0927')
    for IoU_id in range(len(IoU_cross_between_pair)):
        # print('frame_id:',current_f - IoU_id,' IOU:',IoU_cross_between_pair[IoU_id])
        if IoU_cross_between_pair[IoU_id] < 0.1: # debug in 1106 for 30hz
            print('stop at ',current_f - IoU_id)
            IoU_min_frame = current_f - IoU_id
            break
        if IoU_id == len(IoU_cross_between_pair)-1:
            IoU_min_frame = None
            print('no found debug in 0927!!')


    for IoU_id in range(len(IoU_cross_mul[coincide_pair[0]])):
        aaa = IoU_cross_mul[coincide_pair[0]][IoU_id]
        # print(aaa)
        bbb = IoU_cross_mul[coincide_pair[1]][IoU_id]
        # print(bbb)
        ''''''
        if ((IoU_cross_mul[coincide_pair[0]][IoU_id]>0.15)|(IoU_cross_mul[coincide_pair[1]][IoU_id]>0.15)):# mouse #related to mouse size important
            if (IoU_id < distance_cross_extrem + 15):
                print('prepare for linking for',coincide_pair,'--------------------------------------->')
                compensate_limit = IoU_id + 1 #after debug
            else:
                print('IoU_id:', IoU_id)
                print('distance_cross_extrem', distance_cross_extrem)
                print('prepare for linking with for', coincide_pair, ' with distance_cross--------------------------------------->')
                compensate_limit = distance_cross_extrem
            print('compensate_limit:',compensate_limit)
            compensate_start_id = current_f - compensate_limit
            print('cross frame:',compensate_start_id)
            #############debug in 1004
            new_flag = False
            if ((animal_species == 1) | (animal_species == 3)):
                if data_fps >= 60:
                    end_f_bias = 10 # MICE
                else:
                    end_f_bias = 5 # MICE
            else:
                end_f_bias =  10 # FISH
            if IoU_id >= end_f_bias:
                if IoU_cross_between_pair[IoU_id - end_f_bias] < 0.5:
                    print('IoU_cross_between_pair[IoU_id] < 0.5')
                    for dd in range(IoU_id, 0, -1):
                        if IoU_cross_between_pair[dd - end_f_bias] > 0.5:
                            compensate_limit = dd - 5
                            new_flag = True
                            break
                    if new_flag == False:
                        print('new failed')
                    compensate_start_id = current_f - compensate_limit
                    print('cross frame new:',compensate_start_id)
            else:
                print('IoU_id < end_f_bias')
            ##############################
            target_pos_compensate_full = target_pos_compensate
            target_pos_compensate = target_pos_compensate[:compensate_limit+1]
            target_sz_compensate = target_sz_compensate[:compensate_limit+1]
            score_map_compensate = score_map_compensate[:compensate_limit+1]

            if use_algorithm:
                final_compensate_id = match_compensate_target(target_pos_mul, target_sz_mul, score_map_list, target_pos_compensate,target_sz_compensate, compensate_start_id, coincide_pair, seq.object_num, seq.frames,target_pos_compensate_full,animal_species,start_cross_id,data_fps,IoU_min_frame)
                # final_compensate_id = match_compensate_target_simple(target_pos_mul, score_map_list,compensate_start_id,coincide_pair,seq.object_num,animal_species)


            target_pos_mul, target_sz_mul = optimize_tracklet(target_pos_mul, target_sz_mul, target_pos_compensate,target_sz_compensate,final_compensate_id,compensate_start_id=compensate_start_id)
            IoU_cross_flag = True

        if IoU_cross_flag:
           break
        if IoU_id == len(IoU_cross_mul[coincide_pair[0]])-1:
            print('no found!!')
            if cross_limit:
                print('if cross_limit:')
                compesate_length = 100
                final_compensate_id = coincide_pair[0]
                target_pos_compensate = target_pos_compensate[:compesate_length]
                target_sz_compensate = target_sz_compensate[:compesate_length]
                target_pos_compensate.reverse()
                target_sz_compensate.reverse()
                target_pos_compensate = np.asarray(target_pos_compensate)
                target_sz_compensate = np.asarray(target_sz_compensate)
                target_pos_mul, target_sz_mul = optimize_tracklet(target_pos_mul, target_sz_mul, target_pos_compensate,target_sz_compensate,final_compensate_id,compensate_start_id=current_f-(compesate_length-1))
            IoU_cross_flag = True

    return target_pos_mul, target_sz_mul, final_compensate_id