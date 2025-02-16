import os

import cv2
import numpy as np
# from . import tracker_multi
# celegan_flag = False
point_size = 3
thickness = 4
detect_debug = False
debug_save = False
refine_debug_save = False
refine_print = False
upper_thresh = 1.2
down_thresh = 0.5

def missing_object_detect(detect_img,target_pos_mul,target_sz_mul,bg_img,seq_name,current_frame,animal_num,animal_species,fine_detection_mode,area_in_first_frame,kernel,area_mean,down_sample_fg,large_id,large_size, area_rank=-2):
    erode_flag = True
    # init_rect_mul = []
    point_size = 3
    thickness = 4
    im_show = cv2.cvtColor(detect_img, cv2.COLOR_RGB2BGR)
    detect_img = cv2.cvtColor(detect_img,cv2.COLOR_RGB2GRAY)
    file_dir = './debug/'+ seq_name
    # if not os.path.exists(file_dir):
    #     os.makedirs(file_dir)
    file_name1 = file_dir + '/thresh_' + str(current_frame) + '.jpg'
    file_name2 = file_dir + '/thresh_full_mask_' + str(current_frame) + '.jpg'
    file_name3 = file_dir + '/thresh_before_' + str(current_frame) + '.jpg'
    file_name4 = file_dir + '/thresh_track_' + str(current_frame) + '.jpg'
    file_name5 = file_dir + '/thresh_cutbg_' + str(current_frame) + '.jpg'
    if isinstance(bg_img,str):
        # print('1')
        n=current_frame

        frame_id = "%07d" % (n / down_sample_fg)

        thresh = cv2.imread(bg_img + '/'+ frame_id + '.png', 0)

        thresh[thresh>0] = 255
        ########
        # cv2.imwrite(file_name1, thresh)
    else:
        diff = cv2.absdiff(bg_img,detect_img)
        _, thresh = cv2.threshold(diff,40,255,cv2.THRESH_BINARY)
        if detect_debug:
            cv2.imshow('diff', diff)
            # cv2.imshow('thresh', thresh)
            # cv2.waitKey(0)
    thresh = cv2.resize(thresh, (detect_img.shape[1], detect_img.shape[0]))
    if kernel != 0:
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel,kernel))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel,kernel))
        cv2.erode(thresh, erode_kernel, thresh, iterations=2)
        cv2.dilate(thresh, dilate_kernel, thresh, iterations=2)

    time_frame = 10


    if fine_detection_mode:
            black_img = np.zeros((thresh.shape[0], thresh.shape[1]), np.uint8)
            for i in range(animal_num):
                for f in range(current_frame-time_frame,current_frame):
                    black_img[int(target_pos_mul[i][f][1]),int(target_pos_mul[i][f][0])] = 255
            for i in range(animal_num):
                vel_compensate = np.diff(target_pos_mul[i][current_frame-5:current_frame], axis=0)
                avg_vel_compensate = vel_compensate.mean(axis=0)
                for f in range(time_frame):
                    x_pos_pred = min(int(target_pos_mul[i][current_frame][1]+f*avg_vel_compensate[1]), black_img.shape[0] - 1)
                    y_pos_pred = min(int(target_pos_mul[i][current_frame][0]+f*avg_vel_compensate[0]), black_img.shape[1] - 1)
                    # print('x_pos_pred',x_pos_pred)
                    # print('black_img.shape[0]',black_img.shape[0])
                    # print('y_pos_pred',y_pos_pred)
                    # print('black_img.shape[1]',black_img.shape[1])
                    x_pos_pred = max(x_pos_pred,0)
                    y_pos_pred = max(y_pos_pred,0)
                    black_img[x_pos_pred,y_pos_pred] = 255

            black_img_1 = black_img
            if detect_debug:
                cv2.imshow("Black Image1", black_img_1)
            if debug_save:
                cv2.imwrite(file_name4, black_img)
            dilate_kernel_size = int(area_mean/170)
            if dilate_kernel_size > 10:
                dilate_kernel_size = 10
            # print('area_mean:',area_mean)
            # print('dilate_kernel_size:',dilate_kernel_size)
            if dilate_kernel_size != 0:
                dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(dilate_kernel_size,dilate_kernel_size)) # 4 4
                cv2.dilate(black_img, dilate_kernel, black_img, iterations=4)


            if detect_debug:
                cv2.imshow("Black Image2", black_img)
            # !!!!
            thresh[thresh < 125] = 0
            thresh[thresh >= 125] = 255
            black_img[black_img < 125] = 0
            black_img[black_img >= 125] = 255
            if debug_save:
                cv2.imwrite(file_name5, black_img)
            ddd = np.clip(thresh - black_img,0,255)
            # print(ddd.__contains__(1))
            ddd[ddd < 125] = 0
            # print(black_img.__contains__(254))
            # print(ddd.__contains__(1))
            thresh = ddd
            # print(thresh.__contains__(1))
            # erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
            #
            # cv2.erode(thresh, erode_kernel, thresh, iterations=2)

            '''
            for i in range(animal_num):
               thresh[int(target_pos_mul[i][current_frame][1]-target_sz_mul[i][current_frame][0]/2):int(target_pos_mul[i][current_frame][1]+target_sz_mul[i][current_frame][0]/2),int(target_pos_mul[i][current_frame][0]-target_sz_mul[i][current_frame][0]/2):int(target_pos_mul[i][current_frame][0]+target_sz_mul[i][current_frame][0]/2)] = 0
            '''
            if debug_save:
                cv2.imwrite(file_name1, thresh)
    else:
            if debug_save:
                cv2.imwrite(file_name3, thresh)
            # cv2.imwrite(file_name4, diff)

            for i in range(animal_num):
                if animal_species != 3:
                    thresh[int(target_pos_mul[i][current_frame][1]-target_sz_mul[i][current_frame][0]/2):int(target_pos_mul[i][current_frame][1]+target_sz_mul[i][current_frame][0]/2),int(target_pos_mul[i][current_frame][0]-target_sz_mul[i][current_frame][0]/2):int(target_pos_mul[i][current_frame][0]+target_sz_mul[i][current_frame][0]/2)] = 0
                else:
                    if i in large_id:
                        target_sz = large_size #180
                        thresh[int(target_pos_mul[i][current_frame][1]-target_sz/2):int(target_pos_mul[i][current_frame][1]+target_sz/2),int(target_pos_mul[i][current_frame][0]-target_sz/2):int(target_pos_mul[i][current_frame][0]+target_sz/2)] = 0
                    else:
                        thresh[int(target_pos_mul[i][current_frame][1]-target_sz_mul[i][current_frame][0]/2):int(target_pos_mul[i][current_frame][1]+target_sz_mul[i][current_frame][0]/2),int(target_pos_mul[i][current_frame][0]-target_sz_mul[i][current_frame][0]/2):int(target_pos_mul[i][current_frame][0]+target_sz_mul[i][current_frame][0]/2)] = 0
            if debug_save:
                cv2.imwrite(file_name2, thresh)

    # if ((animal_species == 1) | (animal_species == 3)):
    #     erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    #     dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
    #     cv2.erode(thresh, erode_kernel, thresh, iterations=2)
    #     cv2.dilate(thresh, dilate_kernel, thresh, iterations=2)
    #     ########
    #     # thresh = cv2.resize(thresh,(detect_img.shape[1], detect_img.shape[0]))
    #     cv2.imwrite(file_name3, thresh)

    # if ((animal_species == 1) | (animal_species == 3)):
    #     for i in range(animal_num):
    #        # print(target_pos_mul[i][current_frame])
    #        # print(target_sz_mul[i][current_frame])
    #        # print(thresh[int(target_pos_mul[i][current_frame][1]),int(target_pos_mul[i][current_frame][0])])
    #        thresh[int(target_pos_mul[i][current_frame][1]-target_sz_mul[i][current_frame][0]/2):int(target_pos_mul[i][current_frame][1]+target_sz_mul[i][current_frame][0]/2),int(target_pos_mul[i][current_frame][0]-target_sz_mul[i][current_frame][0]/2):int(target_pos_mul[i][current_frame][0]+target_sz_mul[i][current_frame][0]/2)] = 0
    #     cv2.imwrite(file_name2, thresh)

    if detect_debug:
        cv2.imshow("original", detect_img)
        cv2.imshow("dilate", thresh)
        # cv2.waitKey(0)
    # print(thresh.__contains__(1))
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8, ltype=None)
    area_mul = stats[:,4]
    sort_area = np.sort(area_mul)
    # if isinstance(bg_img,str):
    #     sort_area = sort_area[~((sort_area > area_in_first_frame * upper_thresh) | (sort_area < area_in_first_frame * down_thresh))]
    max_area_id = np.where(area_mul == sort_area[area_rank])
    max_area_id = np.array(max_area_id)
    if max_area_id.shape[1] > 1:
        max_area_id = max_area_id[:,0]
    # print(centroids[max_area_id])
    max_area_centroids = centroids[max_area_id].squeeze()
    # im_show = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    cv2.circle(im_show, (int(max_area_centroids[0]),int(max_area_centroids[1])), point_size, (0, 255, 255), thickness)
    # for i in range(centroids.shape[0]):
    #     cv2.circle(im_show, (int(centroids[i][0]),int(centroids[i][1])), 1, (0, 255, 255), thickness)
    if detect_debug:
        cv2.imshow("result", im_show)
        cv2.waitKey(0)
    return max_area_centroids[0],max_area_centroids[1]

def refine_pos_for_loss(detect_img,target_pos_mul,target_sz_mul,bg_img,seq_name,current_frame,animal_num,animal_species,area_in_first_frame, target_refine_list,loss_animal_id,kernel,down_sample_fg):
    refine_loss_flag = False
    # print('refine_pos_for_loss------------------>')
    im_show = cv2.cvtColor(detect_img, cv2.COLOR_RGB2BGR)
    detect_img = cv2.cvtColor(detect_img,cv2.COLOR_RGB2GRAY)
    n = current_frame

    frame_id = "%07d" % (n / down_sample_fg)

    thresh = cv2.imread(bg_img + '/'+ frame_id + '.png', 0)
    thresh[thresh>0] = 255

    # fix_miss_target_flag = False
    # refine_flag = False
    # miss_target_id = 0
    # miss_target_time = 0
    # miss_target_id_list = []
    thresh = cv2.resize(thresh,(detect_img.shape[1], detect_img.shape[0]))
    # if kernel > 8: #### debug 1012
    #     kernel = 8
    if kernel != 0:
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel,kernel))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel,kernel))
        cv2.erode(thresh, erode_kernel, thresh, iterations=2)
        cv2.dilate(thresh, dilate_kernel, thresh, iterations=2)



        # cv2.imwrite(file_name3, thresh)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8, ltype=None)


    area_mul = stats[:,4]
    # area_mul.sort()
    # area_mul = area_mul[::-1]
    target_class_list = [False for i in range(area_mul.shape[0])]
    target_class_list[0] = True
    target_loss_pos = target_pos_mul[loss_animal_id][-1]

    # print('before refine:',target_loss_pos)
    distance_list = []
    distance_id = []
    # delete small fg
    for i in range(area_mul.shape[0]):
         if ((area_mul[i] < area_in_first_frame * down_thresh) | (area_mul[i] > area_in_first_frame * upper_thresh)):
             target_class_list[i] = True
    if len(target_refine_list)>0:
        for id in target_refine_list:
            target_class_list[id] = True
    for i in range(area_mul.shape[0]):
        if target_class_list[i] == False:
            centroids_refine = centroids[i].squeeze()
            distance = np.linalg.norm(centroids_refine - target_loss_pos)
            distance_list.append(distance)
            distance_id.append(i)
    if len(distance_list)>0:
        min_value = min(distance_list)
        min_index = distance_list.index(min_value)
        miss_class = distance_id[min_index]
        centroids_refine = centroids[miss_class].squeeze()
        # print('refine pos:',centroids_refine)
        # print('area:',area_mul[target_class])
        ####### refine
        # centroids_refine = np.array([100,100])

        target_pos_mul[loss_animal_id][-1] = centroids_refine

        # print('after refine:',target_pos_mul[loss_animal_id][-1])
        # print('correct loss successfully!!')
        refine_loss_flag = True

    return target_pos_mul,refine_loss_flag

def refine_pos(detect_img,target_pos_mul,target_sz_mul,bg_img,seq_name,current_frame,animal_num,animal_species,area_in_first_frame,kernel,down_sample_fg,concave_flag):

    refine_debug = False
    if animal_species == 3:
        refine_judge_mode = 2 # 1: area threshold 2: number of animal (more strict)
    else:
        refine_judge_mode = 1
    # area_in_first_frame = 5968  # 3220 white or min area in size-differ img 1874 1685
    # print(target_pos_mul[0][-1])
    file_dir = './debug/'+ seq_name
    # if not os.path.exists(file_dir):
    #     os.makedirs(file_dir)
    file_name3 = file_dir + '/thresh_after_erode_dilate_' + str(current_frame) + '.jpg'
    im_show = cv2.cvtColor(detect_img, cv2.COLOR_RGB2BGR)
    detect_img = cv2.cvtColor(detect_img,cv2.COLOR_RGB2GRAY)
    n = current_frame

    frame_id = "%07d" % (n / down_sample_fg)

    thresh = cv2.imread(bg_img + '/' + frame_id + '.png', 0)
    thresh[thresh > 0] = 255
    
    fix_miss_target_flag = False
    refine_flag = False
    miss_target_id = 0
    miss_target_time = 0
    miss_target_id_list = []
    target_refine_list = []
    thresh = cv2.resize(thresh,(detect_img.shape[1], detect_img.shape[0]))
    # if kernel > 8: #### debug 1012
    #     kernel = 8
    # print('kernel:',kernel)
    if kernel != 0:
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel,kernel))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel,kernel))
        cv2.erode(thresh, erode_kernel, thresh, iterations=2)
        cv2.dilate(thresh, dilate_kernel, thresh, iterations=2)
        if refine_debug_save:
            cv2.imwrite(file_name3, thresh)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8, ltype=None)
    #####################
    ##########################################
    if concave_flag:
      rec_image = np.zeros((detect_img.shape[0], detect_img.shape[1]), dtype=np.uint8)
      for stats_id in range(stats.shape[0]):
          if stats_id != 0:
              stats_ = stats[stats_id]
              rec_image[stats_[1]:stats_[1]+stats_[3],stats_[0]:stats_[0]+stats_[2] ] = stats_id
      labels = rec_image
    ##########################################



    area_mul = stats[:,4]
    # area_mul.sort()

    target_class_list = [False for i in range(area_mul.shape[0])]
    target_class_list[0] = True
    # delete small fg
    for i in range(area_mul.shape[0]):
         if (area_mul[i] < area_in_first_frame * 0.4):
             target_class_list[i] = True
    if refine_print:
        print('frame num:',current_frame)
        print('area_mul:',area_mul)
    if refine_judge_mode == 2:
        if target_class_list.count(False) == animal_num:
                    refine_flag = True
    for animal_id in range(animal_num):
        if refine_judge_mode == 1:
            refine_flag = False
        target_pos = target_pos_mul[animal_id][-1]
        # target_pos = np.array([419,476])
        cv2.circle(im_show, (int(target_pos[0]),int(target_pos[1])), point_size, (0, 0, 255), 6)
        if refine_print:
            print("animal_id:",animal_id)
            print('class:',labels[int(target_pos[1]),int(target_pos[0])])
            print('ori pos:',target_pos)
        target_class = labels[int(min(target_pos[1],detect_img.shape[0]-1)),int(min(target_pos[0],detect_img.shape[1]-1))]
        # target_class = labels[int(target_pos[1]),int(target_pos[0])]

        if target_class != 0:
            target_class_list[target_class] = True
            if refine_judge_mode == 1:
                if (area_mul[target_class] < area_in_first_frame * upper_thresh) & (area_mul[target_class] > area_in_first_frame * down_thresh): # change from 0.7
                    refine_flag = True
            if refine_flag:
                centroids_refine = centroids[target_class].squeeze()
                # print('refine pos:',centroids_refine)
                cv2.circle(im_show, (int(centroids_refine[0]), int(centroids_refine[1])), point_size, (0, 0, 0), 3)
                # print('area:',area_mul[target_class])
                ####### refine
                # centroids_refine = np.array([100,100])

                target_pos_mul[animal_id][-1] = centroids_refine
                if refine_print:
                    print('after refine:',target_pos_mul[animal_id][-1])
                    print('refine successfully!!')
                target_refine_list.append(target_class)
            else:
                if refine_print:
                    print('cross area cannot refine!!')
                # print()
        else:
            fix_miss_target_flag = True
            miss_target_id = animal_id
            miss_target_id_list.append(miss_target_id)
            if refine_print:
                print('warning: target_pos not in any area..')
            miss_target_time += 1
    if refine_judge_mode == 2:
        if refine_flag:
            if fix_miss_target_flag:
                # print('fix_miss_target id', miss_target_id)
                if target_class_list.count(False) == 1:
                    miss_target_class = target_class_list.index(False)
                    centroids_refine = centroids[miss_target_class].squeeze()
                    # print('before refine:', target_pos_mul[miss_target_id][-1])
                    cv2.circle(im_show, (int(centroids_refine[0]), int(centroids_refine[1])), point_size, (0, 0, 0), 3)
                    # print(area_mul[miss_target_id])
                    ####### refine
                    # centroids_refine = np.array([100,100])

                    target_pos_mul[miss_target_id][-1] = centroids_refine
                    # print('after refine:', target_pos_mul[miss_target_id][-1])
                    # print('refine successfully!!')
                    target_refine_list.append(miss_target_class)
                # else:
                #     print('0 or more than 2 choice')


    # for i in range(area_mul.shape[0]):
    #       if (area_mul[i] < area_in_first_frame * 1.3) & (area_mul[i] > area_in_first_frame * 0.7):
    #           centroids_ = centroids[i].squeeze()
    #           cv2.circle(im_show, (int(centroids_[0]), int(centroids_[1])), point_size, (255, 0, 255), thickness)
    for st in stats[1:]:
            cv2.rectangle(im_show, (st[0], st[1]), (st[0]+st[2], st[1]+st[3]), (0, 255, 0), 3)
            # cv2.circle(im_show, (int(st[0]+st[2]*0.5), int(st[1]+st[3]*0.5)), point_size, (0, 0, 0), thickness)
    if refine_debug:
        cv2.imshow("dilate", thresh)
        cv2.imshow("result", im_show)
        cv2.waitKey(0)

    return target_pos_mul, miss_target_time, miss_target_id_list, target_refine_list

def missing_object_cal(detect_img,target_pos_mul,target_sz_uniform,bg_img,seq_name,current_frame,animal_num,animal_species,kernel,down_sample_fg):

    # init_rect_mul = []
    point_size = 3
    thickness = 4
    im_show = cv2.cvtColor(detect_img, cv2.COLOR_RGB2BGR)
    detect_img = cv2.cvtColor(detect_img,cv2.COLOR_RGB2GRAY)
    file_dir = './debug/'+ seq_name
    # if not os.path.exists(file_dir):
    #             os.makedirs(file_dir)
    # file_name1 = file_dir + '/thresh_' + str(current_frame) + '.jpg'
    file_name2 = file_dir + '/thresh_full_mask_' + str(current_frame) + '_cal.jpg'
    file_name3 = file_dir + '/thresh_before_' + str(current_frame) + '_cal.jpg'
    # file_name4 = file_dir + '/thresh_diff_' + str(current_frame) + '.jpg'
    if isinstance(bg_img,str):
        # print('1')
        n = current_frame

        frame_id = "%07d" % (n / down_sample_fg)

        thresh = cv2.imread(bg_img + '/'+ frame_id + '.png', 0)

        thresh[thresh>0] = 255
        ########
        # cv2.imwrite(file_name1, thresh)
    else:
        diff = cv2.absdiff(bg_img,detect_img)
        _, thresh = cv2.threshold(diff,40,255,cv2.THRESH_BINARY)
        if detect_debug:
            cv2.imshow('diff', diff)
            # cv2.imshow('thresh', thresh)
            # cv2.waitKey(0)



    # time_frame = 10
    # if (animal_species == 2):
    #     if fine_detection_mode:
    #         black_img = np.zeros((thresh.shape[0], thresh.shape[1]), np.uint8)
    #         for i in range(animal_num):
    #             for f in range(current_frame-time_frame,current_frame):
    #                 black_img[int(target_pos_mul[i][f][1]),int(target_pos_mul[i][f][0])] = 255
    #         for i in range(animal_num):
    #             vel_compensate = np.diff(target_pos_mul[i][current_frame-5:current_frame], axis=0)
    #             avg_vel_compensate = vel_compensate.mean(axis=0)
    #             for f in range(time_frame):
    #                 black_img[int(target_pos_mul[i][current_frame][1]+f*avg_vel_compensate[1]),int(target_pos_mul[i][current_frame][0]+f*avg_vel_compensate[0])] = 255
    #
    #         black_img_1 = black_img
    #         dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
    #         if detect_debug:
    #             cv2.imshow("Black Image1", black_img_1)
    #         cv2.dilate(black_img, dilate_kernel, black_img, iterations=4)

    #         if detect_debug:
    #             cv2.imshow("Black Image2", black_img)
    #         # !!!!
    #         ddd = np.clip(thresh - black_img,0,255)
    #         # print(ddd.__contains__(1))
    #         ddd[ddd < 125] = 0
    #         # print(black_img.__contains__(254))
    #         # print(ddd.__contains__(1))
    #         thresh = ddd
    #         # print(thresh.__contains__(1))
    #         # erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    #         #
    #         # cv2.erode(thresh, erode_kernel, thresh, iterations=2)
    #
    #         '''
    #         for i in range(animal_num):
    #            thresh[int(target_pos_mul[i][current_frame][1]-target_sz_mul[i][current_frame][0]/2):int(target_pos_mul[i][current_frame][1]+target_sz_mul[i][current_frame][0]/2),int(target_pos_mul[i][current_frame][0]-target_sz_mul[i][current_frame][0]/2):int(target_pos_mul[i][current_frame][0]+target_sz_mul[i][current_frame][0]/2)] = 0
    #         '''
    #         cv2.imwrite(file_name1, thresh)
    #     else:
    #         cv2.imwrite(file_name3, thresh)
    #         cv2.imwrite(file_name4, diff)
    #
    #         for i in range(animal_num):
    #             thresh[int(target_pos_mul[i][current_frame][1]-target_sz_uniform/2):int(target_pos_mul[i][current_frame][1]+target_sz_uniform/2),int(target_pos_mul[i][current_frame][0]-target_sz_uniform/2):int(target_pos_mul[i][current_frame][0]+target_sz_uniform/2)] = 0
    #
    #         cv2.imwrite(file_name2, thresh)


    thresh = cv2.resize(thresh,(detect_img.shape[1], detect_img.shape[0]))
    # if kernel > 8:#### debug 1012
    #     kernel = 8
    if kernel != 0:
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel,kernel))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel,kernel))
        cv2.erode(thresh, erode_kernel, thresh, iterations=2)
        cv2.dilate(thresh, dilate_kernel, thresh, iterations=2)

    if debug_save:
        cv2.imwrite(file_name3, thresh)

    for i in range(animal_num):
       # print(target_pos_mul[i][current_frame])
       # print(target_sz_mul[i][current_frame])
       # print(thresh[int(target_pos_mul[i][current_frame][1]),int(target_pos_mul[i][current_frame][0])])
       thresh[int(target_pos_mul[i][current_frame][1]-target_sz_uniform/2):int(target_pos_mul[i][current_frame][1]+target_sz_uniform/2),int(target_pos_mul[i][current_frame][0]-target_sz_uniform/2):int(target_pos_mul[i][current_frame][0]+target_sz_uniform/2)] = 0

    if debug_save:
        cv2.imwrite(file_name2, thresh)
    nonzero_count = np.count_nonzero(thresh)

    return nonzero_count
