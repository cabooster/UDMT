import itertools
import os
from os.path import join, isdir
from os import makedirs
import argparse
import json
import numpy as np
import torch
from os import listdir
import cv2
import time as time
def convert_format(dataset,dataset_name, mouse_num, frame_number,result_save_file_list,mot_target_size):

    # dataset = '5-white-mice-30hz' #!!8-fish bbnc-fish 5-bbnc-mice-compress 17-bbnc-fish 5-white-mice-60hz rat-mice-30hz 2-miniscope-mice-30hz
    # mouse_num = 5 #!!
    # frame_number = 999
    # dataset_name = '12-bbnc-fish-60hz' # 5-bbnc-mice rat-mice-30hz 12-bbnc-fish-60hz 2-miniscope-mice-30hz
    tracker_name = 'ours'
    result_dir = 'E:/01-LYX/new-research/TransformerTrack-main/TransformerTrack-main/pytracking/data/trackers/mot_challenge/MOT15-train/'+tracker_name+'/data/'
    track_result_dir = 'E:/01-LYX/new-research/TransformerTrack-main/TransformerTrack-main/results/trdimp/trdimp/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_save_path = result_dir + dataset_name+'.txt'
    # test_path = join('result', dataset, 'DCFNet_test')
    # sub_set_base_path='./dataset/' + dataset
    # videos = sorted(listdir(sub_set_base_path))
    pos_plus_sz_mul_list = []
    for mouse_id in range(mouse_num):
        # result_path = join(track_result_dir + dataset +'_'+ str(mouse_id) + '_new.txt')
        pos_mul = np.loadtxt(result_save_file_list[mouse_id], dtype=np.int, delimiter=',')
        # pos_plus_sz_mul_list.append(pos_mul.tolist())
        pos_plus_sz_mul_list.append(pos_mul.tolist())
    pos_plus_sz_mul_arr = np.asarray(pos_plus_sz_mul_list)
    target_pos_mul = pos_plus_sz_mul_arr[:,:,:2]
    target_sz_mul = pos_plus_sz_mul_arr[:,:,2:]
    test_path = './tests/'+dataset
    # target_size = 70 #!! 70
    frame_count_real = 0
    # if not isdir(test_path): makedirs(test_path)
    # result_path = join(test_path, 'test.txt')
    with open(result_save_path, 'w') as f:
        for frame_id in range(frame_number): #!!
            if frame_id % 1 == 0:
                frame_count_real += 1
                # result_path = join(test_path, 'test.txt')
                for animal_id in range(mouse_num):
                    # target_size = 55
                    # if animal_id == 0:
                    #      target_size = 160
                    f.write('{},{},{},{},{},{},1,-1,-1,-1'.format(frame_count_real,animal_id+1,int(target_pos_mul[animal_id][frame_id][0]+target_sz_mul[animal_id][frame_id][0]/2-mot_target_size/2), int(target_pos_mul[animal_id][frame_id][1]+target_sz_mul[animal_id][frame_id][0]/2-mot_target_size/2),mot_target_size,mot_target_size) + '\n')
                    # target_size = target_sz_mul[animal_id][frame_id][0]
                    # f.write('{},{},{},{},{},{},1,-1,-1,-1'.format(frame_count_real,animal_id+1,int(target_pos_mul[animal_id][frame_id][0]), int(target_pos_mul[animal_id][frame_id][1]),target_size,target_size) + '\n')
    # print(frame_count_real)