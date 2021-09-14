'''
Author: Shuailin Chen
Created Date: 2021-09-14
Last Modified: 2021-09-14
	content: 
'''

import torch
from torch import Tensor


def label_onehot(label, num_classes):
    ''' Convert indexed label to one-hot vector 
    
    Args:
        label (Tensor): in shape of BxCxHxW 
    '''

    batch_size, _, im_h, im_w = label.shape
    num_classes = num_classes
    
    assert torch.all(input>=0)

    outputs = torch.zeros([batch_size, num_classes, im_h, im_w]).to(label.device)
    return outputs.scatter_(1, label, 1.0)