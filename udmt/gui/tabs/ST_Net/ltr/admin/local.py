import os
import shutil


class EnvironmentSettings:
    def __init__(self,run_training_params):
        self.workspace_dir = run_training_params['project_folder'] + '/models/' + run_training_params['video_name'] # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.lasot_dir = ''
        self.got10k_dir = run_training_params['project_folder'] + '/training-datasets/' + run_training_params['video_name']
        self.trackingnet_dir = ''
        self.coco_dir = ''
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
        if os.path.exists(self.workspace_dir):
            # Iterate through all items in the folder
            for item in os.listdir(self.workspace_dir):
                # Build the full path
                item_path = os.path.join(self.workspace_dir, item)

                # Check if it's a file
                if os.path.isfile(item_path):
                    os.remove(item_path)  # Delete the file
                    print(f"Deleted file: {item_path}")

                # Check if it's a directory
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)  # Delete the subfolder and its contents
                    print(f"Deleted folder: {item_path}")
