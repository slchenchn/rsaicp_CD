'''
Author: Shuailin Chen
Created Date: 2021-06-20
Last Modified: 2021-06-20
	content: 
'''
from abc import ABCMeta, abstractmethod
import torch.nn as nn

class BaseSiamNet(nn.Module):
    ''' Base class for siamese class '''

    __metaclass__ = ABCMeta

    def __init__(self):
        super().__init__()

    def split_images(x):
        ''' Split a 2*c channels image into two c channels images, in order to adapt to MMsegmentation '''
        channels = x.shape[1]
        x1 = x[:, 0:channels, :, :]
        x2 = x[:, channels:, :, :]
        return x1, x2

    @abstractmethod
    def vanilla_forward(self, x1, x2):
        pass
        
    def forward(self, x):
        x1, x2 = self.split_images(x)
        return self.vanilla_forward(x1, x2)