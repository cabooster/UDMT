import imageio.v2
import torch
import numpy as np
from segment_anything import SamPredictor

from ...dataset.range_transform import inv_im_trans
from skimage import io
from ...util.tensor_util import pad_divide_by, unpad
# import cv2
def torch_to_image(torch_img):
    torch_img = inv_im_trans(torch_img)
    torch_img = torch_img.cpu().numpy()
    torch_img = torch_img * 255
    torch_img = torch_img.transpose(1, 2, 0)
    # img = torch_img * 255
    img = torch_img.astype(np.uint8)

    # save_img = img.transpose(2, 0, 1)
    # io.imsave('save_img.tif', save_img)

    return img
class SamController:
    """
    A controller for Scribble-to-Mask (for user interaction, not for DAVIS)
    Takes the image, previous mask, and scribbles to produce a new mask
    ignore_class is usually 255
    0 is NOT the ignore class -- it is the label for the background
    """
    def __init__(self, sam_predictor, num_objects, ignore_class, device='cuda:0'):
        self.sam_predictor = sam_predictor
        self.num_objects = num_objects
        self.ignore_class = ignore_class
        self.device = device



    def interact(self, image, input_point, input_label):
        ori_img = torch_to_image(image)
        self.sam_predictor.set_image(ori_img)
        masks, scores, logits = self.sam_predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,)
        i = np.argmax(scores)
        mask = masks[i]
        save_mask = np.zeros((mask.shape[0], mask.shape[1]))
        save_mask[~mask] = 0
        save_mask[mask] = 1
        save_mask = torch.from_numpy(save_mask).float().to(device=self.device)
        save_mask = save_mask.unsqueeze(0).unsqueeze(0)
        return save_mask
        # image = image.to(self.device, non_blocking=True)
        # prev_mask = prev_mask.unsqueeze(0)
        #
        # h, w = image.shape[-2:]
        # unaggre_mask = torch.zeros((self.num_objects, h, w), dtype=torch.float32, device=image.device)
        #
        # for ki in range(1, self.num_objects+1):
        #     p_srb = (scr_mask==ki).astype(np.uint8)
        #     n_srb = ((scr_mask!=ki) * (scr_mask!=self.ignore_class)).astype(np.uint8)
        #     # image (1,3,h,w) 原图/255
        #     Rs = torch.from_numpy(np.stack([p_srb, n_srb], 0)).unsqueeze(0).float().to(image.device)
        #     #############
        #     # count = np.count_nonzero(p_srb)
        #     # print('count1:',count)
        #     # p_srb_save = p_srb
        #     # p_srb_save = p_srb_save * 255
        #     # io.imsave('p_srb_save.tif', p_srb_save)
        #     #############
        #     # Rs mark上去的点标签位置是1 别的位置是0 (1,2,h,w)
        #     prev_ = (prev_mask==ki).float().unsqueeze(0)
        #     # prev_ 之前标的点 mark上去的点标签位置是1 别的位置是0 (1,1,h,w)
        #     #############
        #     # prev_ = prev_.numpy()
        #     # count = np.count_nonzero(prev_)
        #     # print('count2:',count)
        #     # prev_save = prev_[0]
        #     # prev_save = prev_save * 255
        #     # io.imsave('prev_save.tif', prev_save)
        #     #############
        #     inputs = torch.cat([image, (prev_mask==ki).float().unsqueeze(0), Rs], 1)
        #     # inputs (1,6,h,w)
        #     inputs, pads = pad_divide_by(inputs, 16)
        #     # inputs (1,6,h,w) h,w是16的倍数
        #     unaggre_mask[ki-1] = unpad(torch.sigmoid(self.s2m_net(inputs)), pads)
        #
        # unaggre_mask_img = unaggre_mask.cpu().numpy()
        # unaggre_mask_img = unaggre_mask_img * 255
        # io.imsave('unaggre_mask_img.tif', unaggre_mask_img)
        # return unaggre_mask
