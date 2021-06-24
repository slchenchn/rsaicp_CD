'''
Author: Shuailin Chen
Created Date: 2021-06-20
Last Modified: 2021-06-21
	content: 
'''
from abc import ABCMeta, abstractmethod
import torch.nn as nn

from mmseg.utils import get_root_logger
from mmcv.runner import load_checkpoint
from mmcv.cnn import (UPSAMPLE_LAYERS, ConvModule, build_activation_layer,
                      build_norm_layer, constant_init, kaiming_init)
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmseg.utils import split_images

class BaseSiamNet(nn.Module):
    ''' Base class for siamese class '''

    __metaclass__ = ABCMeta

    def __init__(self):
        super().__init__()

    @abstractmethod
    def vanilla_forward(self, x1, x2):
        pass
        
    def forward(self, x):
        x1, x2 = split_images(x)
        return self.vanilla_forward(x1, x2)

    def init_weights(self, pretrained=None):
        """Initialize the weights

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')
