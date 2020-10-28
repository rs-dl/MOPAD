import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..registry import LOSSES
from .utils import weight_reduce_loss


def im_cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None):
    # element-wise losses
    loss = F.cross_entropy(pred, label, reduction='none')
    # print(label)
    # print(pred)
    # weight = torch.FloatTensor([0.15896092, 3.74256617, 0.15896092, 0.42687615, 0.16036038, 0.51123639]).cuda()
    # print(label.shape)
    # print(pred.shape)
    # print(weight.shape)
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(
            label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None):
    if pred.dim() != label.dim():
        label, weight = _expand_binary_labels(label, weight, pred.size(-1))
    # weight = torch.FloatTensor([0.15896092, 3.74256617, 0.15896092, 0.42687615, 0.16036038, 0.51123639]).cuda()
    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), weight, reduction='none')
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)

    return loss


def mask_cross_entropy(pred, target, label, reduction='mean', avg_factor=None):
    # TODO: handle these two reserved arguments
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, reduction='mean')[None]


@LOSSES.register_module
class ImCrossEntropyLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 loss_weight=1.0):
        super(ImCrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = im_cross_entropy

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        no_of_classes = 6
        # 0.9
        # weight = torch.FloatTensor([1, 1, 1, 1, 1, 1]).cuda()
        # 0.99
        # weight = torch.FloatTensor([0.99742274, 1.01030905, 0.99742274, 0.99742274, 0.99742274, 0.99742274]).cuda()
        # 0.999
        # weight = torch.FloatTensor([0.72745181, 2.06528424, 0.72745181, 0.73440169, 0.72745181, 0.74541045]).cuda()
        # 0.9999
        # weight = torch.FloatTensor([0.15896092, 3.74256617, 0.15896092, 0.42687615, 0.16036038, 0.51123639]).cuda()
        # 0.99999
        weight = torch.FloatTensor([0.01988008, 4.06510283, 0.01988008, 0.38680778, 0.04662644, 0.48158286]).cuda()
        # 0.999999
        # weight = torch.FloatTensor([0.0091132, 4.09265612, 0.0091132, 0.38213106, 0.03835095, 0.47774866]).cuda()
        labels_one_hot = F.one_hot(label, no_of_classes).float()
        weight = weight.unsqueeze(0)
        weight = weight.repeat(labels_one_hot.shape[0],1) * labels_one_hot
        weight = weight.sum(1)
        # weight = weight.unsqueeze(1)
        # weight = weight.repeat(1,no_of_classes)
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls
