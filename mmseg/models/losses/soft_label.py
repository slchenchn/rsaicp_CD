'''
Author: Shuailin Chen
Created Date: 2021-08-20
Last Modified: 2021-09-14
	content: 
'''

import torch 
from torch import nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import get_class_weight, weight_reduce_loss


@LOSSES.register_module()
class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
    NOTE: wrong implementation
    """
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

        
def cross_entropy_with_soft_label(pred,
                                label,
                                weight=None,
                                class_weight=None,
                                reduction='mean',
                                avg_factor=None,
                                ):
    """ Cross entropy loss with soft labels 
    NOTE: this function do not accept param `ignore_index`, set all entries to 0 instead.
    
    Args:
        label (Tensor): should has the same shape with `pred`
    """
    # class_weight is a manual rescaling weight given to each class.
    # If given, has to be a Tensor of size C element-wise losses
    
    assert pred.shape == label.shape, f'shape of prediction and label \
                                        shouldbe the same'
    logprobs = F.log_softmax(pred, dim=1)
    loss = logprobs * label

    if class_weight is not None:
        class_weight = class_weight.view(1, pred.shape[1], 1, 1)
        loss *= class_weight

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss

    
@LOSSES.register_module()
class CrossEntropyLossWithSoftLabel(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        super().__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)

        if self.use_sigmoid:
            raise NotImplementedError
            # self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            raise NotImplementedError
            # self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy_with_soft_label

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls
