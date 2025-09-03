import imageio.v2
import torch
import numpy as np
from segment_anything import SamPredictor

from ..interact.s2m.s2m_network import deeplabv3plus_resnet50 as S2M
from skimage import io
from ...util.tensor_util import pad_divide_by, unpad
# import cv2

class S2MController:
    """
    A controller for Scribble-to-Mask (for user interaction, not for DAVIS)
    Takes the image, previous mask, and scribbles to produce a new mask
    ignore_class is usually 255 
    0 is NOT the ignore class -- it is the label for the background
    """
    def __init__(self, s2m_net:S2M, sam_predictor, num_objects, ignore_class, device='cuda:0'):
        self.s2m_net = s2m_net
        self.sam_predictor = sam_predictor
        self.num_objects = num_objects
        self.ignore_class = ignore_class
        self.device = device


    def interact(self, image, prev_mask, scr_mask):
        image = image.to(self.device, non_blocking=True)
        prev_mask = prev_mask.unsqueeze(0)

        h, w = image.shape[-2:]
        unaggre_mask = torch.zeros((self.num_objects, h, w), dtype=torch.float32, device=image.device)

        for ki in range(1, self.num_objects+1):
            p_srb = (scr_mask==ki).astype(np.uint8)
            n_srb = ((scr_mask!=ki) * (scr_mask!=self.ignore_class)).astype(np.uint8)

            Rs = torch.from_numpy(np.stack([p_srb, n_srb], 0)).unsqueeze(0).float().to(image.device)
            #############
            # count = np.count_nonzero(p_srb)
            # print('count1:',count)
            # p_srb_save = p_srb
            # p_srb_save = p_srb_save * 255
            # io.imsave('p_srb_save.tif', p_srb_save)
            #############

            prev_ = (prev_mask==ki).float().unsqueeze(0)

            #############
            # prev_ = prev_.numpy()
            # count = np.count_nonzero(prev_)
            # print('count2:',count)
            # prev_save = prev_[0]
            # prev_save = prev_save * 255
            # io.imsave('prev_save.tif', prev_save)
            #############
            inputs = torch.cat([image, (prev_mask==ki).float().unsqueeze(0), Rs], 1)
            # inputs (1,6,h,w)
            inputs, pads = pad_divide_by(inputs, 16)

            unaggre_mask[ki-1] = unpad(torch.sigmoid(self.s2m_net(inputs)), pads)

        unaggre_mask_img = unaggre_mask.cpu().numpy()
        unaggre_mask_img = unaggre_mask_img * 255
        io.imsave('unaggre_mask_img.tif', unaggre_mask_img)
        return unaggre_mask