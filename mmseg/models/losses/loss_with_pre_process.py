'''
Author: Shuailin Chen
Created Date: 2021-08-18
Last Modified: 2021-08-18
	content: 
'''
from torch import nn

from ..builder import LOSSES, build_loss, build_post_process


@LOSSES.register_module()
class LossWithPreProcess(nn.Module):
	def __init__(self, ori_type, *args, **kargs):
		super().__init__()
		self.pre_process = build_post_process(kargs.pop('pre_process'))
		kargs.update(type=ori_type)
		self.loss = build_loss(kargs)

	def forward(self, cls_score, *args, **kwargs):
		cls_score = self.pre_process(cls_score)
		return self.loss(cls_score, *args, **kwargs)
