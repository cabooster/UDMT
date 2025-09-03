import torch.nn as nn
import torch
from torch.nn import functional as F


class LBHinge(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """
    def __init__(self, error_metric=nn.MSELoss(), threshold=None, clip=None):
        super().__init__()
        self.error_metric = error_metric
        self.threshold = threshold if threshold is not None else -100
        self.clip = clip

    def forward(self, prediction, label, target_bb=None):
        # print(prediction[0][0])
        # print(label[0][0])
        ''''''
        negative_mask = (label < self.threshold).float()
        positive_mask = (1.0 - negative_mask)

        prediction = negative_mask * F.relu(prediction) + positive_mask * prediction
        
        '''
        true_mask = positive_mask * label
        true_mask_ = true_mask.permute(1,0,2,3)
        prediction_ = prediction.permute(1,0,2,3)
        loss_sum = 0
        loss_list = []
        for i in range(10):
            loss_ = self.error_metric(prediction_[i], true_mask_[i]) # 10 3 23 23
            loss_sum += loss_
            loss_list.append(loss_)
        avg_loss = loss_sum / 10
        max_loss = max(loss_list)
        print(max_loss)
        '''
        
        loss = self.error_metric(prediction, positive_mask * label)
        # loss = self.error_metric(prediction, label)

        if self.clip is not None:
            loss = torch.min(loss, torch.tensor([self.clip], device=loss.device))
        return loss
