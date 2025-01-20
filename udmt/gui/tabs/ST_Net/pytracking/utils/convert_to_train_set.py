import glob
import itertools
import os
from os.path import join, isdir
from os import makedirs
import argparse
import json
import numpy as np
from scipy.ndimage import generic_filter
from scipy.signal import medfilt
import torch
from os import listdir
import cv2
import time as time
######################
def count_files_in_folder(folder_path):
    # Initialize a counter for files
    file_count = 0

    # Iterate through all items in the folder
    for item in os.listdir(folder_path):
        # Build the full path to the item
        item_path = os.path.join(folder_path, item)

        # Check if the item is a file
        if os.path.isfile(item_path):
            file_count += 1

    return file_count
def create_train_label(dataset, trajectories_file_path, train_label_path):
    # dataset = '1-rat-2-mice-2-full'#1-rat-2-mice-2-full miniscope-5mice-male-1615-60hz 18-bbnc-flies-fullres-50hz  18-bbnc-flies-50hz-illumination-change '7-mice-full' #!! 7-mice-full-0.3bright  5-mice-full-47hz-15min-g2 7-mice-full 17-bbnc-fish -divide2 3-mice-full 1-rat-2-mice-2-full 17-bbnc-fish
    # dataset_name = '1-rat-2-mice-2-full-gt-2min-whole-filter30'#7-celegans-10hz-g1-gt-25min-new-whole-filter20-refine 5-mice-30hz-train-6000f  1-rat-2-mice-2-full-gt-2min miniscope-5mice-male-1615-60hz-5min -whole-refine 18-bbnc-flies-fullres-50hz-gt-5min 18-bbnc-flies-50hz-illumination-change-gt-2min '7-bbnc-mice-gt-7min-without_update' # 7-bbnc-mice-gt-bright0.3 17-bbnc-fish-60hz  1-rat-2-mice-2-full-gt-2min-whole
    # '7-bbnc-mice-gt-7min-without-transform' '7-bbnc-mice-gt-7min-finetune' '7-bbnc-mice-gt-7min-pretrained-model' '7-bbnc-mice-gt-7min-random-initialize'
    # trajectories_file_path = 'label_1-rat-2-mice-2-full_46.25_1.5_202310161549'#label_5-mice-full-divide2_47.20_2.5_202311041648 label_5-mice-full-divide2_52.20_2_202311041613

    for_train = True
    filter_flag = True # True
    filter_size = 5
    # tracker_name = 'ours'
    ######################
    # result_dir = os.path.abspath(os.path.join(os.getcwd(), '..','..','..', 'pytracking' ,'data','trackers','mot_challenge','MOT15-train', tracker_name,'data'))
    # if not os.path.exists(result_dir):
    #     os.makedirs(result_dir)
    ani_num = count_files_in_folder(trajectories_file_path)
    pos_plus_sz_mul_list = []
    for mouse_id in range(ani_num):
        result_path = join(trajectories_file_path, dataset+'_'+ str(mouse_id) + '_new.txt') # !!!!!!!!!!!!!!!!!!!!!!!
        pos_mul = np.loadtxt(result_path, dtype=float, delimiter=',')
        # pos_plus_sz_mul_list.append(pos_mul.tolist())
        pos_plus_sz_mul_list.append(pos_mul.tolist())
    frame_number = pos_mul.shape[0]
    pos_plus_sz_mul_arr = np.asarray(pos_plus_sz_mul_list)
    target_pos_mul = pos_plus_sz_mul_arr[:,:,:2]
    target_sz_mul = pos_plus_sz_mul_arr[:,:,2:]
    ################# filter
    def mean_filter(data):
        return np.mean(data)
    if filter_flag:
        for animal_id in range(ani_num):
            data = target_pos_mul[animal_id]
            filtered_x = generic_filter(data[:, 0], mean_filter, size=filter_size)
            filtered_y = generic_filter(data[:, 1], mean_filter, size=filter_size)

            filtered_time_series = np.column_stack((filtered_x, filtered_y))
            target_pos_mul[animal_id] = filtered_time_series
        #################################

    # result_save_path = result_dir + '/' + dataset_name + '.txt'
    ##################
    frame_count_real = 0
    # if 'whole' in dataset_name:
    #     frame_gap = 1

    # base_results_path = result_dir + '/' + 'label_' +dataset_name
    if not os.path.exists(train_label_path):
        os.makedirs(train_label_path)
    base_results_path = train_label_path + '/' + dataset
    if for_train:
        for animal_id in range(ani_num):
           result_path = '{}_{}_new.txt'.format(base_results_path, animal_id)
           with open(result_path, 'w') as f:
               for jj in range(frame_number):
                  # for x in target_pos_mul[mouse_id]:
                  #     f.write(','.join(['{:.2f}'.format(i) for i in x]) + ',' +','.join(['{:.2f}'.format(i) for i in x]) + '\n')
                  target_size = target_sz_mul[animal_id][jj][0]
                  f.write(','.join(['{:.2f}'.format(i) for i in target_pos_mul[animal_id][jj]]) + ',' + ','.join(['{:.2f}'.format(i) for i in target_sz_mul[animal_id][jj]]) + '\n')
        print('Train label is saved in ',train_label_path)
        print('The training dataset is successfully created.')
    # print(dataset_name)
    # print(trajectories_file_path)

# if __name__ == '__main__':
#     create_train_label(dataset='5-mice-1min',trajectories_file_path='E:/01-LYX/new-research/udmt_project/newwww-2025-01-13/tmp/5-mice-1min/train_set_results/label_5-mice-1min_58.00_2.0_pre_scale_0.8',
#                        train_label_path='E:/01-LYX/new-research/udmt_project/newwww-2025-01-13/training-datasets/5-mice-1min/label')