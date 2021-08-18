'''
Author: Shuailin Chen
Created Date: 2021-06-13
Last Modified: 2021-08-18
	content: 
'''
from .accuracy import Accuracy, accuracy
from .cross_entropy_loss import (CrossEntropyLoss, binary_cross_entropy,
                                 cross_entropy, mask_cross_entropy)
from .dice_loss import DiceLoss
from .lovasz_loss import LovaszLoss
from .utils import reduce_loss, weight_reduce_loss, weighted_loss

from .loss_with_pre_process import LossWithPreProcess

# __all__ = [
#     'accuracy', 'Accuracy', 'cross_entropy', 'binary_cross_entropy',
#     'mask_cross_entropy', 'CrossEntropyLoss', 'reduce_loss',
#     'weight_reduce_loss', 'weighted_loss', 'LovaszLoss', 'DiceLoss'
# ]
