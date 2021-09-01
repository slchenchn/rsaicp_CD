'''
Author: Shuailin Chen
Created Date: 2021-09-01
Last Modified: 2021-09-01
	content: 
'''

import torch
from torch import nn
from mmcv.utils import print_log

from mmseg.utils import (get_root_logger, split_images, split_batches,
                        merge_batches)

from .. import builder
from ..builder import BACKBONES


@BACKBONES.register_module()
class BiTemporalTransformerBackbone(nn.Module):
	''' Perform self-attention on the two images simultaneously, not the paper `BIT` '''

	def __init__(self, *args, cat_dim=-1, merge_method='conc', **kargs):
		assert isinstance(merge_method, str), f'merge_method shoud a str object, but got {type(merge_method)}'

		super().__init__()
		kargs.update(type=kargs.pop('ori_type'))
		self.backbone = builder.build_backbone(kargs)
		self.cat_dim = cat_dim
		self.merge_method = merge_method

		print_log(f'siamese backbone with `{merge_method}` merge method, cat dim= {cat_dim}', 'mmseg')

	def init_weights(self, *args, **kargs):
		self.backbone.init_weights(*args, **kargs)

	def forward(self, x):
		x1, x2 = split_images(x)

		# cat the two images into spatial dimension
		x = torch.cat((x1, x2), dim=self.cat_dim)

		out = self.backbone(x)
		new_out = []
		for sub_out in out:
			batch_1, batch_2 = torch.split(sub_out, 
											sub_out.shape[self.cat_dim] // 2,
											dim=self.cat_dim)
			if self.merge_method.lower() in ('conc', 'concatenate'):
				new_sub_out = torch.cat((batch_1, batch_2), dim=1)
			elif self.merge_method.lower() in ('sub', 'subtraction'):
				# just not to take absolute 
				new_sub_out = batch_1 - batch_2   
			new_out.append(new_sub_out)
		
		return new_out