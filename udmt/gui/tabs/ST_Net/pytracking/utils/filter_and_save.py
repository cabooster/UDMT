import glob
import itertools
import os
from os.path import join, isdir
from os import makedirs
import argparse
import json
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import generic_filter
from scipy.signal import medfilt
import torch
from os import listdir
import cv2
import time as time

from tqdm import tqdm


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
def post_process_results(dataset, filter_size, trajectories_file_path, final_result_path, filter_type = "mean"):
    ######################
    # dataset = '1-rat-2-mice-2-full'#1-rat-2-mice-2-full miniscope-5mice-male-1615-60hz 18-bbnc-flies-fullres-50hz  18-bbnc-flies-50hz-illumination-change '7-mice-full' #!! 7-mice-full-0.3bright  5-mice-full-47hz-15min-g2 7-mice-full 17-bbnc-fish -divide2 3-mice-full 1-rat-2-mice-2-full 17-bbnc-fish
    dataset_name = dataset + '-whole-filter'+ str(filter_size)#'1-rat-2-mice-2-full-gt-2min-whole-filter30'#7-celegans-10hz-g1-gt-25min-new-whole-filter20-refine 5-mice-30hz-train-6000f  1-rat-2-mice-2-full-gt-2min miniscope-5mice-male-1615-60hz-5min -whole-refine 18-bbnc-flies-fullres-50hz-gt-5min 18-bbnc-flies-50hz-illumination-change-gt-2min '7-bbnc-mice-gt-7min-without_update' # 7-bbnc-mice-gt-bright0.3 17-bbnc-fish-60hz  1-rat-2-mice-2-full-gt-2min-whole
    # '7-bbnc-mice-gt-7min-without-transform' '7-bbnc-mice-gt-7min-finetune' '7-bbnc-mice-gt-7min-pretrained-model' '7-bbnc-mice-gt-7min-random-initialize'
    # scale = 1
    # trajectories_file_path = 'label_1-rat-2-mice-2-full_46.25_1.5_202310161549'#label_5-mice-full-divide2_47.20_2.5_202311041648 label_5-mice-full-divide2_52.20_2_202311041613
    # mouse_num = 3 # !!
    filter_flag = True  # True
    ani_num = count_files_in_folder(trajectories_file_path)
    tracker_name = 'ours'
    ######################
    result_dir = final_result_path
    # image_file = 'D:/tracking_datasets/Tracking/GOT-10k-test/test/'+dataset+'/'# !!!!!!!!!!!!!!!!!!!!!!!
    # jpg_files = glob.glob(os.path.join(image_file, '*.jpg'))
    # jpg_file_count = len(jpg_files)
    # frame_number = 2000 # 32000 16000 27980 13990 29550 14775
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    pos_plus_sz_mul_list = []
    for mouse_id in range(ani_num):
        result_path = join(trajectories_file_path,
                           dataset + '_' + str(mouse_id) + '_new.txt')  # !!!!!!!!!!!!!!!!!!!!!!!
        pos_mul = np.loadtxt(result_path, dtype=float, delimiter=',')
        pos_mul = pos_mul
        # pos_plus_sz_mul_list.append(pos_mul.tolist())
        pos_plus_sz_mul_list.append(pos_mul.tolist())
    frame_number = pos_mul.shape[0]
    pos_plus_sz_mul_arr = np.asarray(pos_plus_sz_mul_list)
    target_pos_mul = pos_plus_sz_mul_arr[:, :, :2]
    target_sz_mul = pos_plus_sz_mul_arr[:, :, 2:]
    center_pos_mul = target_pos_mul + target_sz_mul / 2

    ################# filter
    def mean_filter(data):
        return np.mean(data)

    def median_filter(data):
        return np.median(data)

    if filter_type == "mean":
        filter_func = mean_filter
    else:
        filter_func = median_filter
    if filter_flag:
        for animal_id in range(ani_num):
            data = center_pos_mul[animal_id]
            filtered_x = generic_filter(data[:, 0], filter_func, size=filter_size)
            filtered_y = generic_filter(data[:, 1], filter_func, size=filter_size)

            filtered_time_series = np.column_stack((filtered_x, filtered_y))
            center_pos_mul[animal_id] = filtered_time_series

        #################################

    result_save_path = result_dir + '/' + dataset_name + '.npy'
    np.save(result_save_path, center_pos_mul)
    # loaded_data = np.load(result_save_path)

    return dataset_name, result_save_path
# post_process_results('5-mice-1min',5,'E:/01-LYX/new-research/udmt_project/newwww-2025-01-13/tracking-results/5-mice-1min/label_5-mice-1min_58.00_2.0_pre_scale_0.8_202501151920','E:/01-LYX/new-research/udmt_project/newwww-2025-01-13/tracking-results/5-mice-1min')
def create_tracking_video(video_frame_path, result_save_path, video_save_path,video_frame_rate, time_interval=40):
    """
    Create a video showing the positions of animals for each frame with different colors, IDs, and past positions.

    Parameters:
        video_frame_path (str): Path to the folder containing video frames (jpg only).
        center_pos_mul (numpy.ndarray): Array of shape (num_animals, num_frames, 2) indicating animal positions.
        video_save_path (str): Path to save the resulting video.
        time_interval (int): Number of past frames to show positions for each animal.
    """
    print("Generating tracking video. Please wait...")
    point_size = 2  # 1
    thickness = 3  # 2
    font_scale_frame = 1
    show_flag = False
    center_pos_mul = np.load(result_save_path)
    # Define colors for animals
    num_animals = center_pos_mul.shape[0]
    cmap = plt.cm.get_cmap('viridis', num_animals)
    colors = [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(num_animals)]

    # Get sorted list of frame file paths (jpg only)
    frame_files = sorted(
        [f for f in os.listdir(video_frame_path) if f.endswith(".jpg")],
        key=lambda x: int(os.path.splitext(x)[0])  # Assuming frames are named with numbers
    )

    # Check if there are enough frames
    num_frames = center_pos_mul.shape[1]
    if len(frame_files) < num_frames:
        raise ValueError("Not enough frames in the folder compared to center_pos_mul data.")

    frame_files = frame_files[:num_frames]  # Limit to the number of frames in center_pos_mul

    # Read the first frame to get video properties
    first_frame = cv2.imread(os.path.join(video_frame_path, frame_files[0]))
    if first_frame is None:
        raise FileNotFoundError("First frame could not be loaded.")

    frame_height, frame_width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for saving video
    out = cv2.VideoWriter(video_save_path, fourcc, int(video_frame_rate), (frame_width, frame_height))

    # Process each frame
    for i, frame_file in enumerate(tqdm(frame_files, desc="Processing frames")):
        frame_path = os.path.join(video_frame_path, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Could not read frame {frame_file}, skipping.")
            continue

        # overlay = frame.copy()

        # Draw each animal's position and ID
        for animal_idx in range(num_animals):
            x, y = center_pos_mul[animal_idx, i]

            # Draw past positions
            if i > time_interval:
                for tracklets_id in range(0, time_interval, 3):
                    target_pos_interval = center_pos_mul[animal_idx,i-tracklets_id]

                    overlay = frame.copy()

                    overlay = cv2.circle(overlay, (int(target_pos_interval[0]),
                                               int(target_pos_interval[1])), point_size,
                                     colors[animal_idx], thickness)
                    alpha = 0.026 * (time_interval - tracklets_id)  # Transparency factor.
                    # alpha = 1
                    ############ debug
                    # Following line overlays transparent rectangle over the image
                    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # for t in range(1, time_interval + 1, 2):
            #     if i - t < 0:
            #         break
            #     past_x, past_y = center_pos_mul[animal_idx, i - t]
            #     alpha = 0.026 * (time_interval - t + 1)  # Transparency factor
            #     overlay = cv2.circle(overlay, (int(past_x), int(past_y)), point_size, colors[animal_idx],thickness)
            #     frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # Draw current position and ID
            cv2.circle(frame, (int(x), int(y)), point_size, colors[animal_idx], thickness)
            cv2.putText(
                frame,
                f"ID {animal_idx}",
                (int(x) + 10, int(y) - 10),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                font_scale_frame,
                colors[animal_idx],
                1,
                cv2.LINE_AA
            )

        # Display current frame number
        cv2.putText(
            frame,
            f"Frame: {i+1}",(10, 30),cv2.FONT_HERSHEY_COMPLEX_SMALL, font_scale_frame, (255, 255, 255), 1, cv2.LINE_AA
        )
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if show_flag:
            # Show the frame
            cv2.imshow("Tracking", frame)
            cv2.waitKey(1)  # Short delay to display the frame

        # Write the frame to the video
        out.write(frame)

    # Release resources
    out.release()
    if show_flag:
        cv2.destroyAllWindows()
    return video_save_path
# create_tracking_video('E:/01-LYX/new-research/udmt_project/newwww-2025-01-13/tmp/5-mice-1min/extracted-images','E:/01-LYX/new-research/udmt_project/newwww-2025-01-13/tracking-results/5-mice-1min/5-mice-1min-whole-filter5.npy','E:/01-LYX/new-research/udmt_project/newwww-2025-01-13/tracking-results/5-mice-1min/5-mice-1min-whole-filter5.mp4',67)