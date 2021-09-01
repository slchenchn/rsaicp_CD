'''
Author: Shuailin Chen
Created Date: 2021-07-06
Last Modified: 2021-09-01
	content: 
'''

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      constant_init, kaiming_init)
from mmcv.runner import load_checkpoint
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmcv.utils import print_log

from mmseg.utils import (get_root_logger, split_images, split_batches,
                        merge_batches)
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

        print_log(f'siamese backbone with `{merge_method}` merge method', 'mmseg')

    def init_weights(self, *args, **kargs):
        self.backbone.init_weights(*args, **kargs)

    def forward(self, x):

        ''' code of v1'''
        # x1, x2 = split_images(x)
        # out1 = self.backbone(x1)
        # out2 = self.backbone(x2)
        # out = []

        # for sub_out1, sub_out2 in zip(out1, out2):
        #     if self.merge_method.lower() in ('conc', 'concatenate'):
        #         sub_out = torch.cat((sub_out1, sub_out2), dim=1)
            
        #     elif self.merge_method.lower() == 'add':
        #         sub_out = sub_out1 + sub_out2

        #     elif self.merge_method.lower() in ('sub', 'subtraction'):
        #         sub_out = sub_out1 - sub_out2 # just not to take absolute value
                
        #     out.append(sub_out)
        
        ''' code of v2 '''
        x1, x2 = split_images(x)
        x = merge_batches(x1, x2)
        out = self.backbone(x)
        new_out = []
        for sub_out in out:
            batch_1, batch_2 = split_batches(sub_out)
            if self.merge_method.lower() in ('conc', 'concatenate'):
                new_sub_out = torch.cat((batch_1, batch_2), dim=1)
            elif self.merge_method.lower() in ('sub', 'subtraction'):
                # just not to take absolute 
                new_sub_out = batch_1 - batch_2   
            new_out.append(new_sub_out)

        return new_out