"""
A simple user interface for XMem
"""

import os
# fix for Windows
if 'QT_QPA_PLATFORM_PLUGIN_PATH' not in os.environ:
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''

import sys
from argparse import ArgumentParser

import torch

from .inference.interact.s2m_controller import S2MController
from .inference.interact.fbrs_controller import FBRSController
from .inference.interact.s2m.s2m_network import deeplabv3plus_resnet50 as S2M
from .inference.interact.sam_controller import Sam3PredictorAdapter, SamController
from PySide6.QtWidgets import QApplication
from .inference.interact.gui import App
from .inference.interact.resource_manager import ResourceManager
from pathlib import Path
from udmt.gui import BASE_DIR

# Import auto initial prediction module
try:
    from .auto_initial_prediction import run_auto_initial_prediction, check_dependencies
    AUTO_PREDICTION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Cannot import auto prediction module: {e}")
    AUTO_PREDICTION_AVAILABLE = False

torch.set_grad_enabled(False)
sam_checkpoint = os.environ.get(
    "UDMT_SAM3_CHECKPOINT",
    os.path.join(BASE_DIR, "tabs", "xmem", "sam_model", "sam3.pt"),
)
XMem_model_path = BASE_DIR +'/tabs/xmem/saves/XMem.pth'



def mask_seg_winclass(video_name, project_path, divide_num, enable_auto_prediction=True):
    file_path = Path(video_name)
    file_name_without_extension = file_path.stem
    workspace_name = project_path + '/tmp/' + file_name_without_extension
    # Arguments parsing
    parser = ArgumentParser()

    parser.add_argument('--model', default=XMem_model_path)
    parser.add_argument('--s2m_model', default=None)#'./tabs/xmem/saves/s2m.pth'
    parser.add_argument('--fbrs_model', default=None)#'./tabs/xmem/saves/fbrs.pth'

    """
    Priority 1: If a "images" folder exists in the workspace, we will read from that directory
    Priority 2: If --images is specified, we will copy/resize those images to the workspace
    Priority 3: If --video is specified, we will extract the frames to the workspace (in an "images" folder) and read from there
    
    In any case, if a "masks" folder exists in the workspace, we will use that to initialize the mask
    That way, you can continue annotation from an interrupted run as long as the same workspace is used.
    """
    parser.add_argument('--images', help='Folders containing input images.', default=None)
    parser.add_argument('--video', help='Video file readable by OpenCV.', default=video_name)
    parser.add_argument('--workspace', help='directory for storing buffered images (if needed) and output masks', default=workspace_name)

    parser.add_argument('--buffer_size', help='Correlate with CPU memory consumption', type=int, default=100)

    parser.add_argument('--num_objects', type=int, default=1)

    # Long-memory options
    # Defaults. Some can be changed in the GUI.
    parser.add_argument('--max_mid_term_frames', help='T_max in paper, decrease to save memory', type=int, default=100)
    parser.add_argument('--min_mid_term_frames', help='T_min in paper, decrease to save memory', type=int, default=90)
    parser.add_argument('--max_long_term_elements', help='LT_max in paper, increase if objects disappear for a long time',
                                                    type=int, default=10000)
    parser.add_argument('--num_prototypes', help='P in paper', type=int, default=128)

    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--mem_every', type=int, default=1000)
    parser.add_argument('--deep_update_every', help='Leave -1 normally to synchronize with mem_every', type=int, default=-1)
    parser.add_argument('--no_amp', help='Turn off AMP', action='store_true')
    parser.add_argument('--size', default=480, type=int,
            help='Resize the shorter side to this size. -1 to use original resolution. ')
    args = parser.parse_args()

    config = vars(args)
    config['enable_long_term'] = True
    config['enable_long_term_count_usage'] = True

    config["propagate_backend"] = "sam3"
    config["sam3_checkpoint"] = sam_checkpoint

    # SAM3 click + propagate run in bf16 inside the adapters.
    network = None

    # Loads the S2M model
    if args.s2m_model is not None:
        s2m_saved = torch.load(args.s2m_model)
        s2m_model = S2M().cuda().eval()
        s2m_model.load_state_dict(s2m_saved)
    else:
        s2m_model = None
    if not os.path.isfile(sam_checkpoint):
        raise FileNotFoundError(
            f"SAM3 checkpoint not found: {sam_checkpoint}. "
            "Place sam3.pt in ./udmt/gui/tabs/xmem/sam_model "
            "or set UDMT_SAM3_CHECKPOINT to the checkpoint path."
        )
    sam_predictor = Sam3PredictorAdapter(sam_checkpoint, device="cuda")
    sam_controller = SamController(sam_predictor, args.num_objects, ignore_class=255)
    # s2m_controller = S2MController(s2m_model, sam_predictor, args.num_objects, ignore_class=255)
    s2m_controller = None
    if args.fbrs_model is not None:
        fbrs_controller = FBRSController(args.fbrs_model)
    else:
        fbrs_controller = None

    # Manages most IO
    resource_manager = ResourceManager(config,divide_num)

    # Set up start point save path
    start_point_save_path = project_path + '/tmp/' + file_name_without_extension + '/extracted-images/'
    if not os.path.exists(start_point_save_path):
        os.makedirs(start_point_save_path)

    # Auto initial prediction feature
    if enable_auto_prediction and AUTO_PREDICTION_AVAILABLE:
        try:
            print("🤖 Running auto initial prediction...")

            # Check if first frame mask already exists
            first_mask_path = os.path.join(resource_manager.mask_dir, "0000000.png")
            if not os.path.exists(first_mask_path):
                print("📋 First frame mask not found, starting auto prediction...")

                # Check dependencies
                if check_dependencies():
                    # Get resource_manager image dimensions and resize scale
                    target_size = (resource_manager.h, resource_manager.w)
                    resize_scale = resource_manager.resize_scale
                    success, num_instances = run_auto_initial_prediction(
                        resource_manager.image_dir,
                        resource_manager.mask_dir,
                        target_size=target_size,
                        start_point_save_path=start_point_save_path,
                        resize_scale=resize_scale
                    )

                    if success and num_instances > 0:
                        print(f"✅ Auto prediction successful! Detected {num_instances} instances")
                        print("💡 Tip: You can check the prediction results and supplement missing animal instances")
                    elif success and num_instances == 0:
                        print("⚠️ Auto prediction completed but no animal instances detected")
                        print("💡 Tip: Please manually click the center of each animal")
                    else:
                        print("❌ Auto prediction failed, please initialize manually")
                else:
                    print("⚠️ Auto prediction dependencies check failed, will use manual mode")
            else:
                print("📋 First frame mask already exists, skipping auto prediction")

        except Exception as e:
            print(f"❌ Error during auto prediction: {e}")
            print("💡 Will continue with manual mode")

    ex = App(network, resource_manager, s2m_controller, fbrs_controller, sam_controller, config, start_point_save_path)

    # Pass auto prediction availability info to App
    ex.auto_prediction_available = AUTO_PREDICTION_AVAILABLE

    return ex

if __name__ == '__main__':
    # Example usage
    video_path = "path/to/your/video.mp4"  # Replace with your video path
    project_folder = "path/to/project"      # Replace with your project folder
    divide_num = 1                          # Replace with your desired divide number
    
    app = QApplication(sys.argv)
    mask_seg_window = mask_seg_winclass(video_path, project_folder, divide_num)
    mask_seg_window.show()
    sys.exit(app.exec_())

