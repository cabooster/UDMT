# 选择inference的参数以及模型文件名
from udmt.gui.tabs.ST_Net.pytracking.utils import TrackerParams
from udmt.gui.tabs.ST_Net.pytracking.features.net_wrappers import NetWithBackbone
# from pytracking.utils import TrackerParams
# from pytracking.features.net_wrappers import NetWithBackbone

def parameters(gui_param):
    params = TrackerParams()

    params.debug = 0
    params.visualization = False

    params.use_gpu = True

    params.image_sample_size = 22*16 #352
    #######
    params.search_area_scale = 2 # !!! mouse 2 fish 2 white 2
    params.train_skipping = 20 # 20hz 30hz: 20 60hz: 30  ###20
    #######

    # border
    params.border_mode = 'inside_major'
    params.patch_max_scale_change = 1.5

    # Learning parameters
    params.sample_memory_size = 50
    params.learning_rate = 0.01
    params.init_samples_minimum_weight = 0.25


    # Net optimization params
    params.update_classifier = True
    params.net_opt_iter = 10
    params.net_opt_update_iter = 2
    params.net_opt_hn_iter = 1

    # transformer memory update
    params.transformer_skipping = 5 # ! fish 5 mice 5
    params.transformer_memory_size = 15 # ! fish 15 mice 15

    # Detection parameters
    params.window_output = False

    # Init augmentation parameters
    params.use_augmentation = True
    params.augmentation = {'fliplr': True,
                           'rotate': [10, -10, 45, -45],
                           'blur': [(3,1), (1, 3), (2, 2)],
                           'relativeshift': [(0.6, 0.6), (-0.6, 0.6), (0.6, -0.6), (-0.6,-0.6)],
                           'dropout': (2, 0.2)}

    params.augmentation_expansion_factor = 2
    params.random_shift_factor = 1/3

    # Advanced localization parameters
    params.advanced_localization = True
    params.target_not_found_threshold = 0.25
    params.distractor_threshold = 0.8
    params.hard_negative_threshold = 0.5
    params.target_neighborhood_scale = 2.2
    params.dispalcement_scale = 0.8
    params.hard_negative_learning_rate = 0.02
    params.update_scale_when_uncertain = True

    # IoUnet parameters
    params.use_iou_net = False
    params.box_refinement_space = 'relative'
    params.iounet_augmentation = False      # Use the augmented samples to compute the modulation vector
    params.iounet_k = 3                     # Top-k average to estimate final box
    params.num_init_random_boxes = 9        # Num extra random boxes in addition to the classifier prediction
    params.box_jitter_pos = 0.1             # How much to jitter the translation for random boxes
    params.box_jitter_sz = 0.5              # How much to jitter the scale for random boxes
    params.maximal_aspect_ratio = 6         # Limit on the aspect ratio
    params.box_refinement_iter = 10          # Number of iterations for refining the boxes
    params.box_refinement_step_length = 2.5e-3  # Gradient step length in the bounding box refinement
    params.box_refinement_step_decay = 1    # Multiplicative step length decay (1 means no decay)
    # trdimp_net
    # DiMPnet_ep0083_finetune_5fish
    # DiMPnet_ep0072_finetune_5mice_withoutgt_change
    # DiMPnet_ep0073_finetune_5mice_withgt_change
    # DiMPnet_ep0072_finetune_5mice_withoutgt_change
    # DiMPnet_ep0050_finetune_5mice_mid_withoutgt_change # current
    # DiMPnet_ep0050_bbnc_fish # current
    # DiMPnet_ep0020_finetune_5whitemice_60hz_3 # current
    # DiMPnet_ep0020_finetune_rat_mice_30hz.pth.tar # current
    #  # current
    # DiMPnet_ep0050_1000fish
    # DiMPnet_ep0020_finetune_5mice_30hz_scale3
    # DiMPnet_ep0020_finetune_5mice_30hz_scale3_2
    # DiMPnet_ep0020_finetune_5mice_30hz_scale3_3 # current
    # DiMPnet_ep0020_finetune_5mice_60hz_scale3 # current
    # DiMPnet_ep0023_finetune_5mice_20hz_scale3
    network_name = gui_param['model_path'] #  DiMPnet_ep0054.pth.tar DiMPnet_ep0088_super DiMPnet_ep0099_unsuper_fromscratch  DiMPnet_ep0099_unsuper_pre3 trdimp_net.pth.tar DiMPnet_ep0083_finetune_5fish
    params.net = NetWithBackbone(net_path = network_name, use_gpu=params.use_gpu)
    print('using ', network_name)
    params.vot_anno_conversion_type = 'preserve_area'

    return params
