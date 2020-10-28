import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..registry import LOSSES
from .utils import weighted_loss


@weighted_loss
def imbalanced_l1_loss(pred,
                     target,
                     beta=1.0,
                     alpha=0.5,
                     gamma=1.5,
                     reduction='mean'):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0

    diff = torch.abs(pred - target)
    b = np.e**(gamma / alpha) - 1
    loss = torch.where(
        diff < beta, alpha / b *
        (b * diff + 1) * torch.log(b * diff / beta + 1) - alpha * diff,
        gamma * diff + gamma / b - alpha * beta)
    # print(loss.shape)
    return loss


@LOSSES.register_module
class ImBalancedL1Loss(nn.Module):
    """Balanced L1 Loss

    arXiv: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)
    """

    def __init__(self,
                 alpha=0.5,
                 gamma=1.5,
                 beta=1.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(ImBalancedL1Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                labels=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        # print("libra_loss")
        # print(labels.shape)
        no_of_classes = 6
        # 0.9
        # weights = torch.FloatTensor([1, 1, 1, 1, 1, 1]).cuda()
        # 0.99
        # weights = torch.FloatTensor([0.99742274, 1.01030905, 0.99742274, 0.99742274, 0.99742274, 0.99742274]).cuda()
        # 0.999
        # weights = torch.FloatTensor([0.72745181, 2.06528424, 0.72745181, 0.73440169, 0.72745181, 0.74541045]).cuda()
        # 0.9999
        # weights = torch.FloatTensor([0.15896092, 3.74256617, 0.15896092, 0.42687615, 0.16036038, 0.51123639]).cuda()
        # 0.99999
        weights = torch.FloatTensor([0.01988008, 4.06510283, 0.01988008, 0.38680778, 0.04662644, 0.48158286]).cuda()
        # 0.999999
        # weights = torch.FloatTensor([0.0091132, 4.09265612, 0.0091132, 0.38213106, 0.03835095, 0.47774866]).cuda()
        
        labels_one_hot = F.one_hot(labels, no_of_classes).float()
        # print(labels_one_hot.shape)
        weights = weights.unsqueeze(0)
        # print(weights.shape)
        weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
        weights = weights.sum(1)
        # print(weights)
        loss_bbox = weights * self.loss_weight * imbalanced_l1_loss(
            pred,
            target,
            weight,
            alpha=self.alpha,
            gamma=self.gamma,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox
