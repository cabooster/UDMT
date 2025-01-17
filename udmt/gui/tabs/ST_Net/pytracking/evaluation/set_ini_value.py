import os
import cv2
import cv2 as cv
import numpy as np
import statistics
from collections import Counter
debug = False
print_flag = False
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
def set_ini_value(animal_species,img_name_list,start_point_corr,object_num,bg_path):
    n = 0
    point_size = 3
    thickness = 4
    # upper_thresh = 1.2
    # down_thresh = 0.5
    frame_id = "%07d" % n
    erode_flag = True
    # bg_img = os.path.abspath(os.path.join(os.getcwd(),  'bg', dataset_name))
    # bg_img = 'E:/01-LYX/new-research/TransformerTrack-main/TransformerTrack-main/pytracking/bg/'+ dataset_name
    detect_img = _read_image(img_name_list[n])
    # detect_img = _read_image('D:/tracking_datasets/Tracking/GOT-10k-test/test/' + dataset_name+ '/'+ frame_id_ + '.jpg')
    im_show = cv2.cvtColor(detect_img, cv2.COLOR_RGB2BGR)
    detect_img = cv2.cvtColor(detect_img,cv2.COLOR_RGB2GRAY)

    thresh = cv2.imread(bg_path + '/'+ frame_id + '.png', 0)

    thresh[thresh>0] = 255
    # _, _, stats_raw, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8, ltype=None)
    # area_mul_raw = stats_raw[:,4]
    # area_mul_raw = np.sort(area_mul_raw)
    # print(area_mul_raw)
    # print('area1',area_mul_raw[-2])

    # if animal_species == 1:
    #         kernel = 1
    #         erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel,kernel))
    #         dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel,kernel))
    #         cv2.erode(thresh, erode_kernel, thresh, iterations=2)
    #         #################
    #         # aa = thresh
    #         # _, _, stats_erode, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8, ltype=None)
    #         # area_mul_erode= stats_erode[:,4]
    #         # area_mul_erode = np.sort(area_mul_erode)
    #         # print(area_mul_erode)
    #         # print('area2',area_mul_erode[-2])
    #         # cv2.imshow("erode", aa)
    #         #################
    #         cv2.dilate(thresh, dilate_kernel, thresh, iterations=2)
    #         ########
    thresh = cv2.resize(thresh, (detect_img.shape[1], detect_img.shape[0]))
    # cv2.imwrite(file_name3, thresh)
    if debug:
        cv2.imshow("before dilate", thresh)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8, ltype=None)
    area_mul = stats[:,4]
    if print_flag:
        print(area_mul)
    ################
    area_list = []
    target_class_list = []
    for ani_id in range(object_num):
        target_pos = start_point_corr[ani_id]
        target_class = labels[int(min(target_pos[1],detect_img.shape[0]-1)),int(min(target_pos[0],detect_img.shape[1]-1))]
        if target_class != 0:
            area_list.append(area_mul[target_class])
            target_class_list.append(target_class)
    # Count the occurrences of each number in the list
    counter = Counter(target_class_list)
    # 创建一个新列表，其中每个元素是原始元素的出现次数
    count_list = [counter[item] for item in target_class_list]
    # Extract numbers that appear only once
    unique_numbers = [num for num, count in counter.items() if count == 1]
    area_pre_ani_list = [a / b if b != 0 else 'undefined' for a, b in zip(area_list, count_list)]
    mean_value = statistics.mean(area_pre_ani_list)
    area_sum_list = []
    target_size_list = []
    if len(unique_numbers) != 0:
        for id_ in unique_numbers:
            area_sum_list.append(area_mul[id_])
            st = stats[id_]
            cv2.rectangle(im_show, (st[0], st[1]), (st[0]+st[2], st[1]+st[3]), (0, 255, 0), 3)
            target_size_list.append(st[2])
            target_size_list.append(st[3])
    if print_flag:
        print('area_in_first_frame',mean_value)
    target_sz_ini = np.mean(target_size_list)
    target_sz_uniform = np.max(target_size_list)
    area_mean = sum(area_sum_list) / len(area_sum_list)
    # target_sz_ini_ = np.mean(target_size_list_)  # target_sz_ini_
    # print('target_sz_ini (mean target_size)__', target_sz_ini_)
    # # print('target_sz_uniform (max target_size)', target_sz_uniform)
    #
    # print('area mean__', mean_value, 'or', median_value)

    ###############################################################


    # print('area_in_first_frame',np.median(area_mul[1:]))
    area_in_first_frame = mean_value
    # area_in_first_frame = np.median(area_mul[1:]) #3220 1874 9846 np.median(area_mul[1:])
    area_rank = -2

    # if target_class != 0:
    #     if (area_mul[target_class] < area_in_first_frame * upper_thresh) & (area_mul[target_class] > area_in_first_frame * down_thresh):
    #         print(area_mul[target_class])
    #         centroids_refine = centroids[target_class].squeeze()
    #         cv2.circle(im_show, (int(centroids_refine[0]), int(centroids_refine[1])), point_size, (0, 0, 0), 6)
    #
    #         ####### refine
    #
    #         print('refine successfully!!')
    #     else:
    #         print('cross area cannot refine!!')
    # else:
    #     print('warning: target_pos not in any area..')
    #####################
    # area_sum = []
    # sort_area = np.sort(area_mul)
    # print(sort_area)
    # target_size_list = []
    # for i in range(area_mul.shape[0]):
    #     if (area_mul[i] < area_in_first_frame * upper_thresh) & (area_mul[i] > area_in_first_frame * down_thresh):
    #         centroids_ = centroids[i].squeeze()
    #         cv2.circle(im_show, (int(centroids_[0]), int(centroids_[1])), point_size, (255, 0, 255), thickness)
    #         print('{:.2f},{:.2f},55,55'.format(centroids_[0],centroids_[1]))
    #         area_sum.append(area_mul[i])
    #         st = stats[i]
    #         cv2.rectangle(im_show, (st[0], st[1]), (st[0]+st[2], st[1]+st[3]), (0, 255, 0), 3)
    #         target_size_list.append(st[2])
    #         target_size_list.append(st[3])
    # if len(target_size_list) != 0:
    #     target_sz_ini = np.mean(target_size_list)
    #     target_sz_uniform = np.max(target_size_list)
    #     area_mean = sum(area_sum) / len(area_sum)
    # else:
    #     raise ValueError('len(target_size_list) != 0!!!!!!you should set area your self!!')
    #     area_in_first_frame = 10065
    #     target_sz_ini = np.sqrt(area_in_first_frame)
    #     target_sz_uniform = np.sqrt(area_in_first_frame)
    #     area_mean = area_in_first_frame
    if print_flag:
        print('target_sz_ini (mean target_size)', target_sz_ini)
        # print('target_sz_uniform (max target_size)', target_sz_uniform)

        print('area mean', area_mean)
    if debug:
        cv2.imshow("result before process", im_show)
    kernel = int(area_mean/400)
    if kernel > 8:
        kernel = 8
    if kernel == 0:
        print('kernel == 0')
        erode_flag = False
    if animal_species == 3:
        kernel = 12
    
    if erode_flag == True:
        if print_flag:
            print('kernel:',kernel)
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel,kernel))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel,kernel))
        cv2.erode(thresh, erode_kernel, thresh, iterations=2)
        cv2.dilate(thresh, dilate_kernel, thresh, iterations=2)
    if debug:
        cv2.imshow("after dilate", thresh)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8, ltype=None)
    area_mul = stats[:,4]
    if print_flag:
        print(area_mul)
    im_show2 = cv2.cvtColor(detect_img, cv2.COLOR_RGB2BGR)
    # if len(unique_numbers) != 0:
    #     for id_ in unique_numbers:
    #         st = stats[id_]
    #         cv2.rectangle(im_show2, (st[0], st[1]), (st[0]+st[2], st[1]+st[3]), (0, 255, 0), 3)
    # for i in range(area_mul.shape[0]):
    #     if (area_mul[i] < area_in_first_frame * upper_thresh) & (area_mul[i] > area_in_first_frame * down_thresh):
    #         centroids_ = centroids[i].squeeze()
    #         cv2.circle(im_show2, (int(centroids_[0]), int(centroids_[1])), point_size, (255, 0, 255), thickness)
    #         st = stats[i]
    #         cv2.rectangle(im_show2, (st[0], st[1]), (st[0]+st[2], st[1]+st[3]), (0, 255, 0), 3)
    if debug:
        cv2.imshow("result after process", im_show2)
        cv2.waitKey(0)
    
    return target_sz_ini, target_sz_uniform, area_in_first_frame, kernel, area_mean
