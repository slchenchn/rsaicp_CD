'''
Author: Shuailin Chen
Created Date: 2021-08-18
Last Modified: 2021-08-18
	content: 
'''
from torch import nn

from ..builder import POST_PROCESS

@POST_PROCESS.register_module()
class SetConstValue(nn.Module):
	''' Set constant value to the prediction logits, this is suitable for the senarios where the backgound is complex 

	Args:
		position (int): the class index to set to a constant value 
		value (float): the value to be setted
	'''
	def __init__(self, position, value):
		super().__init__()
		self.position = SetConstValue.check_list(position)
		self.value = SetConstValue.check_list(value)

	@staticmethod
	def check_list(val):
		''' check whether the input value is a list, if not, change it to a list 
		'''
		if not isinstance(val, (list, tuple)):
			val = [val]
		return val

	def forward(self, logits, *args, **kargs):
		for pos, val in zip(self.position, self.value):
			logits[:, pos, :, :] = val

		return logits