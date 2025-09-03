"""
UDMT (https://cabooster.github.io/UDMT/)
Author: Yixin Li
https://github.com/cabooster/UDMT
Licensed under Non-Profit Open Software License 3.0
"""


import os


###############
DEBUG = True and "DEBUG" in os.environ and os.environ["DEBUG"]
from udmt.version import __version__, VERSION

# print(f"Loading UDMT {VERSION}...")
# from udmt.gui.tracklet_toolbox import refine_tracklets
# from udmt.gui.launch_script import launch_udmt

# from udmt.gui.widgets import SkeletonBuilder

from udmt.create_project import (
    create_new_project,
    # create_new_project_3d,
    # add_new_videos,
    # load_demo_data,
    # create_pretrained_project,
    # create_pretrained_human_project,
)
# from udmt.generate_training_dataset import (
#     check_labels,
#     create_training_dataset,
#     extract_frames,
#     mergeandsplit,
# )
# from udmt.generate_training_dataset import (
#     create_training_model_comparison,
#     create_multianimaltraining_dataset,
# )
# from udmt.generate_training_dataset import (
#     dropannotationfileentriesduetodeletedimages,
#     comparevideolistsanddatafolders,
#     dropimagesduetolackofannotation,
#     adddatasetstovideolistandviceversa,
#     dropduplicatesinannotatinfiles,
#     dropunlabeledframes,
# )
from udmt.utils import (
    # create_labeled_video,
    # create_video_with_all_detections,
    # plot_trajectories,
    auxiliaryfunctions,
    # convert2_maDLC,
    # convertcsv2h5,
    # analyze_videos_converth5_to_csv,
    # analyze_videos_converth5_to_nwb,
    # auxfun_videos,
)

# try:
#     from udmt.pose_tracking_pytorch import transformer_reID
# except ModuleNotFoundError as e:
#     import warnings
#
#     warnings.warn(
#         """
#         As PyTorch is not installed, unsupervised identity learning will not be available.
#         Please run `pip install torch`, or ignore this warning.
#         """
#     )

# from udmt.utils.auxfun_videos import (
#     ShortenVideo,
#     DownSampleVideo,
#     CropVideo,
#     check_video_integrity,
# )
#
# # Train, evaluate & predict functions / all require TF
# from udmt.pose_estimation_tensorflow import (
#     train_network,
#     return_train_network_path,
#     evaluate_network,
#     return_evaluate_network_data,
#     analyze_videos,
#     create_tracking_dataset,
#     analyze_time_lapse_frames,
#     convert_detections2tracklets,
#     extract_maps,
#     visualize_scoremaps,
#     visualize_locrefs,
#     visualize_paf,
#     extract_save_all_maps,
#     export_model,
#     video_inference_superanimal,
# )


# from udmt.pose_estimation_3d import (
#     calibrate_cameras,
#     check_undistortion,
#     triangulate,
#     create_labeled_video_3d,
# )

# from udmt.refine_training_dataset.stitch import stitch_tracklets
# from udmt.refine_training_dataset import (
#     extract_outlier_frames,
#     merge_datasets,
#     find_outliers_in_raw_data,
# )
# from udmt.post_processing import filterpredictions, analyzeskeleton
