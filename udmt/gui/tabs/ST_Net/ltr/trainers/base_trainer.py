import os
import glob
import torch
import traceback
# from ltr.admin import loading, multigpu
from udmt.gui.tabs.ST_Net.ltr.admin import loading, multigpu



class BaseTrainer:
    """Base trainer class. Contains functions for training and saving/loading chackpoints.
    Trainer classes should inherit from this one and overload the train_epoch function."""

    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        self.actor = actor
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loaders = loaders

        self.update_settings(settings)

        self.epoch = 0
        self.stats = {}

        self.device = getattr(settings, 'device', None)
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() and settings.use_gpu else "cpu")

        self.actor.to(self.device)

    def update_settings(self, settings=None):
        """Updates the trainer settings. Must be called to update internal settings."""
        if settings is not None:
            self.settings = settings

        if self.settings.env.workspace_dir is not None:
            self.settings.env.workspace_dir = os.path.expanduser(self.settings.env.workspace_dir)
            # self._checkpoint_dir = os.path.join(self.settings.env.workspace_dir, 'checkpoints')
            self._checkpoint_dir = self.settings.env.workspace_dir
            if not os.path.exists(self._checkpoint_dir):
                os.makedirs(self._checkpoint_dir)
        else:
            self._checkpoint_dir = None

    def delete_first_file_by_name(self, max_files):
        """
        Check if the number of files in the folder exceeds max_files.
        If it does, delete the first file based on filename sorting.
        """
        folder_path = self._checkpoint_dir
        # List all files in the folder
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        # print('List all files in the folder:', files)

        # Check if the number of files exceeds the limit
        if len(files) > max_files:
            # Sort the files by name
            files.sort()

            # Get the first file (based on sorted order)
            first_file = os.path.join(folder_path, files[0])

            # Delete the first file
            os.remove(first_file)
            # print(f"Deleted file: {first_file}")

    def train(self, max_epochs,max_save_snapshots,pretrained_model, load_latest=False, fail_safe=True):
        """Do training for the given number of epochs.
        args:
            max_epochs - Max number of training epochs,
            load_latest - Bool indicating whether to resume from latest epoch.
            fail_safe - Bool indicating whether the training to automatically restart in case of any crashes.
        """
        #pretrained_model = '/home/bbnclxy/ext16t/lyx-star/TransformerTrack-main/pytracking/networks/pretrained'
        # pretrained_model = None
        epoch = -1
        num_tries = 1
        for i in range(num_tries):
            try:
                if load_latest:
                    # print(pretrained_model) 
                    self.load_checkpoint(checkpoint=pretrained_model)
                    print('loading pretrained checkpoint')
                              
                # for param_id in range(6):
                #         print("\033[0;31m", "params={}, lr={}".format(param_id, self.optimizer.param_groups[0]['lr']),
                #               "\033[0m")
                for epoch in range(self.epoch+1, max_epochs+1):
                    # print("\033[0;31m", 'epoch={}'.format(epoch), "\033[0m")
                    self.epoch = epoch

                    self.train_epoch()

                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                    if self._checkpoint_dir:
                        self.save_checkpoint()
                    #############
                    self.delete_first_file_by_name(max_save_snapshots)
                    ###########

                    # for param_id in range(6):
                    #     print("\033[0;31m", "params={}, lr={}".format(param_id, self.optimizer.param_groups[0]['lr']),
                    #           "\033[0m")
            except:
                print('Training crashed at epoch {}'.format(epoch))
                if fail_safe:
                    self.epoch -= 1
                    load_latest = True
                    print('Traceback for the error!')
                    print(traceback.format_exc())
                    print('Restarting training from last epoch ...')
                else:
                    raise

        print('Finished training!')


    def train_epoch(self):
        raise NotImplementedError


    def save_checkpoint(self):
        """Saves a checkpoint of the network and other variables."""

        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        actor_type = type(self.actor).__name__
        net_type = type(net).__name__
        state = {
            'epoch': self.epoch,
            'actor_type': actor_type,
            'net_type': net_type,
            'net': net.state_dict(),
            'net_info': getattr(net, 'info', None),
            'constructor': getattr(net, 'constructor', None),
            'optimizer': self.optimizer.state_dict(),
            'stats': self.stats,
            'settings': self.settings
        }


        # directory = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path)
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        directory = self._checkpoint_dir

        # First save as a tmp file
        tmp_file_path = '{}/{}_ep{:04d}.tmp'.format(directory, net_type, self.epoch)
        torch.save(state, tmp_file_path)

        file_path = '{}/{}_ep{:04d}.pth.tar'.format(directory, net_type, self.epoch)

        # Now rename to actual checkpoint. os.rename seems to be atomic if files are on same filesystem. Not 100% sure
        os.rename(tmp_file_path, file_path)


    def load_checkpoint(self, checkpoint = None, fields = None, ignore_fields = None, load_constructor = False):
        """Loads a network checkpoint file.

        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the network at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """

        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        actor_type = type(self.actor).__name__
        net_type = type(net).__name__

        if checkpoint is None:
            # Load most recent checkpoint
            checkpoint_list = sorted(glob.glob('{}/{}/{}_bbep*.pth.tar'.format(self._checkpoint_dir,
                                                                             self.settings.project_path, net_type)))
            if checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
            else:
                print('No matching checkpoint file found')
                return
        elif isinstance(checkpoint, int):
            # Checkpoint is the epoch number
            checkpoint_path = '{}/{}/{}_ep{:04d}.pth.tar'.format(self._checkpoint_dir, self.settings.project_path,
                                                                 net_type, checkpoint)
        elif isinstance(checkpoint, str):
            # checkpoint is the path
            if os.path.isdir(checkpoint):
                checkpoint_list = sorted(glob.glob('{}/*_ep*.pth.tar'.format(checkpoint)))
                if checkpoint_list:
                    checkpoint_path = checkpoint_list[-1]
                else:
                    raise Exception('No checkpoint found')
            else:
                checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError
            
        print('checkpoint_path:',checkpoint_path)
        # Load network
        checkpoint_dict = loading.torch_load_legacy(checkpoint_path)

        assert net_type == checkpoint_dict['net_type'], 'Network is not of correct type.'

        if fields is None:
            fields = checkpoint_dict.keys()
        if ignore_fields is None:
            ignore_fields = ['settings']

        # Never load the scheduler. It exists in older checkpoints.
        ignore_fields.extend(['lr_scheduler', 'constructor', 'net_type', 'actor_type', 'net_info'])
        ''''''
        # 不载入filter层的权重
        del_key = []
        for key, _ in checkpoint_dict['net'].items():
            if ".filter" in key:
            #if ".filter" in key or "transformer" in key:
                del_key.append(key)
        for key in del_key:
            del checkpoint_dict['net'][key]
        # Load all fields
        for key in fields:
            if key in ignore_fields:
                continue
            if key == 'net':
                net.load_state_dict(checkpoint_dict[key], strict=False)
            # 注释后不载入之前的学习率
            '''
            elif key == 'optimizer':
                self.optimizer.load_state_dict(checkpoint_dict[key])
            
            else:
                setattr(self, key, checkpoint_dict[key])
            '''

        # Set the net info
        '''
        if load_constructor and 'constructor' in checkpoint_dict and checkpoint_dict['constructor'] is not None:
            net.constructor = checkpoint_dict['constructor']
        if 'net_info' in checkpoint_dict and checkpoint_dict['net_info'] is not None:
            net.info = checkpoint_dict['net_info']
        '''
        # Update the epoch in lr scheduler
        if 'epoch' in fields:
            self.lr_scheduler.last_epoch = self.epoch
            
        '''
        for name, value in net.named_parameters():
            if '.layer1' in name or '.layer2' in name or '.layer3' in name or 'extractor.conv1' in name or 'extractor.bn1' in name:
                value.requires_grad = False
            # print(name)
        for k, v in net.named_parameters():   # 查看是否冻结成功
            print('{}: {}'.format(k, v.requires_grad))
        '''
        
        

        return True
