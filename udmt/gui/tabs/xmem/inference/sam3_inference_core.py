"""SAM3 video tracker behind the XMem InferenceCore.step() interface."""
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class _Store:
    size = 0


class _DummyMemory:
    def __init__(self, config):
        self.min_mt_frames = config.get("min_mid_term_frames", 90)
        self.max_mt_frames = config.get("max_mid_term_frames", 100)
        self.max_long_elements = config.get("max_long_term_elements", 10000)
        self.num_prototypes = config.get("num_prototypes", 128)
        self.max_work_elements = 1
        self.work_mem = _Store()
        self.long_mem = _Store()

    def update_config(self, config):
        self.min_mt_frames = config.get("min_mid_term_frames", self.min_mt_frames)
        self.max_mt_frames = config.get("max_mid_term_frames", self.max_mt_frames)
        self.max_long_elements = config.get("max_long_term_elements", self.max_long_elements)
        self.num_prototypes = config.get("num_prototypes", self.num_prototypes)


class Sam3InferenceCore:
    """
    Drop-in replacement for XMem InferenceCore.

    First step(image, mask) seeds SAM3 with the current-frame mask (one object
    per connected component). Later step(image) calls yield the next frame.
    """

    def __init__(self, image_dir, checkpoint_path, config, device="cuda"):
        self.image_dir = image_dir
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.device = device
        self.mem_every = config.get("mem_every", 1000)
        self.memory = _DummyMemory(config)
        self.all_labels = None
        self.num_objects = int(config.get("num_objects", 1))

        self.tracker = None
        self._video_model = None
        self.state = None
        self._gen = None
        self._expected_hw = None

    def set_all_labels(self, all_labels):
        self.all_labels = list(all_labels)
        if self.all_labels:
            self.num_objects = len(self.all_labels)

    def update_config(self, config):
        self.config = config
        self.mem_every = config.get("mem_every", self.mem_every)
        self.memory.update_config(config)

    def clear_memory(self):
        self._gen = None
        self.state = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def release(self):
        self._gen = None
        self.state = None
        self.tracker = None
        self._video_model = None

    def _ensure_tracker(self):
        if self.tracker is not None:
            return
        from sam3.model_builder import build_sam3_video_model
        from .interact.sam_controller import (
            _sam3_amp,
            cast_sam3_to_bf16,
            patch_official_sam3_fused_mlp,
        )

        print(f"Loading SAM3 video tracker: {self.checkpoint_path}")
        patch_official_sam3_fused_mlp()
        with _sam3_amp():
            model = build_sam3_video_model(
                checkpoint_path=self.checkpoint_path,
                load_from_HF=False,
                device=self.device,
            )
            cast_sam3_to_bf16(model)
        self._video_model = model
        self.tracker = model.tracker
        self.tracker.backbone = model.detector.backbone
        self.tracker.to(self.device)
        self.tracker.backbone.to(self.device)
        self.tracker.eval()
        print("SAM3 video tracker ready (bf16).")

    def step(self, image, mask=None, valid_labels=None, end=False, frame_idx=None):
        from .interact.sam_controller import _sam3_amp

        with _sam3_amp():
            if mask is not None:
                start_idx = 0 if frame_idx is None else int(frame_idx)
                self._start_from_mask(mask, start_idx=start_idx)
            if self._gen is None:
                raise RuntimeError("SAM3 propagate has no seed mask. Confirm the first-frame mask first.")
            try:
                _, obj_ids, _, video_res_masks, _ = next(self._gen)
            except StopIteration:
                raise RuntimeError("SAM3 propagate finished before the GUI loop stopped.")
            return self._to_prob(obj_ids, video_res_masks, image)

    def _start_from_mask(self, mask, start_idx):
        self._ensure_tracker()
        index = self._mask_to_index(mask)
        self._expected_hw = index.shape
        components = self._split_instances(index)
        if not components:
            raise RuntimeError("No foreground in the current mask to propagate.")

        print(f"SAM3 mask propagate: {len(components)} object(s) from frame {start_idx}")
        self.state = self.tracker.init_state(
            video_path=self.image_dir,
            offload_video_to_cpu=True,
            async_loading_frames=True,
        )
        for obj_id, inst in components:
            self.tracker.add_new_mask(
                inference_state=self.state,
                frame_idx=start_idx,
                obj_id=obj_id,
                mask=torch.from_numpy(inst.astype(np.float32)),
            )
        self._gen = self.tracker.propagate_in_video(
            self.state,
            start_frame_idx=start_idx,
            max_frame_num_to_track=int(self.state["num_frames"]),
            reverse=False,
            tqdm_disable=True,
            propagate_preflight=True,
        )

    def _mask_to_index(self, mask):
        if torch.is_tensor(mask):
            mask_np = mask.detach().float().cpu().numpy()
        else:
            mask_np = np.asarray(mask)
        if mask_np.ndim == 3:
            if mask_np.shape[0] == 1:
                return (mask_np[0] > 0.5).astype(np.uint8)
            index = (mask_np.argmax(axis=0) + 1).astype(np.uint8)
            index[mask_np.max(axis=0) < 0.5] = 0
            return index
        return (mask_np > 0.5).astype(np.uint8)

    def _split_instances(self, index):
        labeled = index.astype(np.int32)
        if labeled.max() <= 1:
            binary = (labeled > 0).astype(np.uint8)
            n, cc = cv2.connectedComponents(binary)
            instances = []
            obj_id = 1
            for cid in range(1, n):
                inst = cc == cid
                if int(inst.sum()) < 8:
                    continue
                instances.append((obj_id, inst))
                obj_id += 1
            return instances

        instances = []
        for obj_id in range(1, int(labeled.max()) + 1):
            inst = labeled == obj_id
            if inst.any():
                instances.append((obj_id, inst))
        return instances

    def _to_prob(self, obj_ids, video_res_masks, image):
        if image is not None:
            h, w = image.shape[-2:]
        else:
            h, w = self._expected_hw
        device = image.device if torch.is_tensor(image) else torch.device(self.device)

        scores = []
        for i, _oid in enumerate(obj_ids):
            m = video_res_masks[i]
            if torch.is_tensor(m):
                if m.dim() == 3:
                    m = m[0]
                scores.append(m.float())
            else:
                scores.append(torch.from_numpy(np.asarray(m)).float())

        if self.num_objects <= 1:
            fg = torch.zeros((h, w), device=device)
            for score in scores:
                bin_mask = score > 0
                if tuple(bin_mask.shape[-2:]) != (h, w):
                    bin_mask = F.interpolate(
                        bin_mask.float()[None, None], size=(h, w), mode="nearest"
                    )[0, 0] > 0.5
                fg = torch.maximum(fg, bin_mask.to(device).float())
            return torch.stack([1.0 - fg, fg], dim=0)

        chans = [torch.ones((h, w), device=device)]
        for obj_id in range(1, self.num_objects + 1):
            chans.append(torch.zeros((h, w), device=device))
        for i, oid in enumerate(obj_ids):
            oid = int(oid)
            if 1 <= oid <= self.num_objects:
                bin_mask = scores[i] > 0
                if tuple(bin_mask.shape[-2:]) != (h, w):
                    bin_mask = F.interpolate(
                        bin_mask.float()[None, None], size=(h, w), mode="nearest"
                    )[0, 0] > 0.5
                chans[oid] = bin_mask.to(device).float()
        stacked = torch.stack(chans, dim=0)
        stacked[0] = (stacked[1:].sum(0) == 0).float()
        return stacked
