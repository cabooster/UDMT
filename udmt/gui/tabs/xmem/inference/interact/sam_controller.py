import contextlib
import gc

import torch
import numpy as np
from PIL import Image

from ...dataset.range_transform import inv_im_trans
import cv2


def _sam3_amp():
    # Official SAM3 is built for bf16. Cast activations to match bf16 weights.
    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def cast_sam3_to_bf16(module):
    """Cast floating parameters to bf16. Leave non-float / complex buffers alone."""
    if module is None:
        return module
    for param in module.parameters():
        if param.is_floating_point() and param.dtype != torch.bfloat16:
            param.data = param.data.to(dtype=torch.bfloat16)
    return module


def patch_official_sam3_fused_mlp():
    """GitHub SAM3 fused MLP does `mat1.to(bf16)` then `self.fc2(x)` in fp32.

    Keep the Linear + activation in the weight dtype so bf16 weights stay bf16
    and the older local checkout still works in whatever dtype it loaded.
    """
    try:
        import sam3.model.vitdet as vitdet
        import sam3.perflib.fused as fused
    except ImportError:
        return

    if getattr(fused, "_udmt_dtype_mlp", False):
        return

    def addmm_act(activation, linear, mat1):
        x = torch.nn.functional.linear(
            mat1.to(dtype=linear.weight.dtype), linear.weight, linear.bias
        )
        if activation in (torch.nn.functional.relu, torch.nn.ReLU):
            return torch.nn.functional.relu(x)
        if activation in (torch.nn.functional.gelu, torch.nn.GELU):
            return torch.nn.functional.gelu(x)
        raise ValueError(f"Unexpected activation {activation}")

    fused.addmm_act = addmm_act
    fused._udmt_dtype_mlp = True
    if getattr(vitdet, "addmm_act", None) is not None:
        vitdet.addmm_act = addmm_act


class Sam3PredictorAdapter:
    """Expose SAM3 point prompting through the SAM1 predictor interface."""

    def __init__(self, checkpoint_path, device="cuda"):
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        print(f"Loading SAM3 checkpoint: {checkpoint_path}")
        self.checkpoint_path = checkpoint_path
        self.device = device
        patch_official_sam3_fused_mlp()
        with _sam3_amp():
            self.model = build_sam3_image_model(
                checkpoint_path=checkpoint_path,
                enable_inst_interactivity=True,
                device=device,
            )
            cast_sam3_to_bf16(self.model)
            self.processor = Sam3Processor(self.model)
        self.inference_state = None

    def release(self):
        self.inference_state = None
        self.processor = None
        self.model = None

    def set_image(self, image):
        """Encode one RGB uint8 image, matching SamPredictor.set_image()."""
        with _sam3_amp():
            self.inference_state = self.processor.set_image(Image.fromarray(image))

    def predict(self, point_coords, point_labels, multimask_output=True):
        if self.inference_state is None:
            raise RuntimeError("set_image() must be called before predict().")

        with _sam3_amp():
            masks, scores, logits = self.model.predict_inst(
                self.inference_state,
                point_coords=np.asarray(point_coords, dtype=np.float32),
                point_labels=np.asarray(point_labels),
                multimask_output=multimask_output,
            )
        masks = np.asarray(masks)
        scores = np.asarray(scores).reshape(-1)
        if masks.ndim == 4 and masks.shape[0] == 1:
            masks = masks[0]
        return masks, scores, logits


def keep_blob_at(mask, x, y):
    """Drop whatever is not connected to the clicked pixel."""
    x, y = int(round(x)), int(round(y))
    mask_u8 = mask.astype(np.uint8)
    if not (0 <= y < mask_u8.shape[0] and 0 <= x < mask_u8.shape[1]) or mask_u8[y, x] == 0:
        return mask

    _, labels = cv2.connectedComponents(mask_u8)
    return labels == labels[y, x]


def touches_open_border(mask, box, shape):
    """True if the mask runs into a crop edge that is not an image edge, i.e. the
    crop was too tight and cut the animal in half."""
    x0, y0, x1, y1 = box
    h, w = shape
    return ((y0 > 0 and mask[0].any()) or (y1 < h and mask[-1].any())
            or (x0 > 0 and mask[:, 0].any()) or (x1 < w and mask[:, -1].any()))


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
        self.checkpoint_path = getattr(sam_predictor, "checkpoint_path", None)

    def unload_image_model(self):
        """Free the click-time SAM3 image model so propagate can use the VRAM."""
        if self.sam_predictor is None:
            return
        print("Unloading SAM3 image model to free GPU memory for propagate.")
        self.sam_predictor.release()
        self.sam_predictor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def ensure_image_model(self):
        if self.sam_predictor is not None:
            return
        if not self.checkpoint_path:
            raise RuntimeError("SAM3 image model was unloaded and no checkpoint path is stored.")
        print(f"Reloading SAM3 image model: {self.checkpoint_path}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam_predictor = Sam3PredictorAdapter(self.checkpoint_path, device=device)

    def release(self):
        predictor = getattr(self, "sam_predictor", None)
        if predictor is not None and hasattr(predictor, "release"):
            predictor.release()
        self.sam_predictor = None

    def interact(self, image, input_point, input_label, hires_image=None):
        self.ensure_image_model()
        ori_img = torch_to_image(image)
        h, w = ori_img.shape[:2]

        # SAM rescales its input to a 1024 long side, so prompting it on the whole
        # frame leaves a small animal with the same handful of pixels whatever the
        # source resolution was. Prompt it on a crop around the click instead, taken
        # from the full resolution frame when one is available.
        src = ori_img if hires_image is None else hires_image
        scale = np.asarray([src.shape[1]/w, src.shape[0]/h])
        mask = self.predict_on_crop(src, np.asarray(input_point)*scale, input_label)

        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h),
                              interpolation=cv2.INTER_AREA) > 0.5

        save_mask = torch.from_numpy(mask).float().to(device=self.device)
        save_mask = save_mask.unsqueeze(0).unsqueeze(0)
        return save_mask

    def predict_on_crop(self, img, point, input_label):
        """Run SAM on a window around the click, widening it until the mask stops
        being clipped by the window. Returns a full size boolean mask."""
        h, w = img.shape[:2]
        cx, cy = float(point[0, 0]), float(point[0, 1])
        crop = max(16, min(h, w)//4)

        while True:
            x0 = int(round(max(0, min(cx - crop/2, w - crop))))
            y0 = int(round(max(0, min(cy - crop/2, h - crop))))
            x1, y1 = min(w, x0 + crop), min(h, y0 + crop)

            self.sam_predictor.set_image(img[y0:y1, x0:x1])
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=point - np.asarray([x0, y0]),
                point_labels=input_label,
                multimask_output=True,)
            i = np.argmax(scores)
            mask = keep_blob_at(masks[i], cx - x0, cy - y0)

            if crop >= max(h, w) or not touches_open_border(mask, (x0, y0, x1, y1), (h, w)):
                full_mask = np.zeros((h, w), dtype=bool)
                full_mask[y0:y1, x0:x1] = mask
                return full_mask

            crop *= 2
        # image = image.to(self.device, non_blocking=True)
        # prev_mask = prev_mask.unsqueeze(0)
        #
        # h, w = image.shape[-2:]
        # unaggre_mask = torch.zeros((self.num_objects, h, w), dtype=torch.float32, device=image.device)
        #
        # for ki in range(1, self.num_objects+1):
        #     p_srb = (scr_mask==ki).astype(np.uint8)
        #     n_srb = ((scr_mask!=ki) * (scr_mask!=self.ignore_class)).astype(np.uint8)

        #     Rs = torch.from_numpy(np.stack([p_srb, n_srb], 0)).unsqueeze(0).float().to(image.device)
        #     #############
        #     # count = np.count_nonzero(p_srb)
        #     # print('count1:',count)
        #     # p_srb_save = p_srb
        #     # p_srb_save = p_srb_save * 255
        #     # io.imsave('p_srb_save.tif', p_srb_save)
        #     #############

        #     prev_ = (prev_mask==ki).float().unsqueeze(0)

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

        #     unaggre_mask[ki-1] = unpad(torch.sigmoid(self.s2m_net(inputs)), pads)
        #
        # unaggre_mask_img = unaggre_mask.cpu().numpy()
        # unaggre_mask_img = unaggre_mask_img * 255
        # io.imsave('unaggre_mask_img.tif', unaggre_mask_img)
        # return unaggre_mask
