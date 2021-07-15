'''
Author: Shuailin Chen
Created Date: 2021-07-06
Last Modified: 2021-07-07
	content: 
'''

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      constant_init, kaiming_init)
from mmcv.runner import load_checkpoint
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmseg.utils import get_root_logger, split_images
from .. import builder
from ..builder import BACKBONES


@BACKBONES.register_module()
class SiamBackboneWrapper(nn.Module):
    ''' Wrapper for classical single-branch networks '''

    def __init__(self, *args, merge_method='conc', **kargs):
        assert isinstance(merge_method, str), f'merge_method shoud a str object, but got {type(merge_method)}'

        self.merge_method = merge_method
        super().__init__()
        kargs.update(type=kargs.pop('ori_type'))
        self.backbone = builder.build_backbone(kargs)

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained)

    def forward(self, x):
        x1, x2 = split_images(x)
        
        out1 = self.backbone(x1)
        out2 = self.backbone(x2)
        out = []
        for sub_out1, sub_out2 in zip(out1, out2):
            if self.merge_method.lower() in ('conc', 'concatenate'):
                sub_out = torch.cat((sub_out1, sub_out2), dim=1)
            
            elif self.merge_method.lower() == 'add':
                sub_out = sub_out1 + sub_out2

            elif self.merge_method.lower() in ('sub', 'subtraction'):
                sub_out = sub_out1 - sub_out2 # just not to take absolute value
                
            out.append(sub_out)
        
        return out