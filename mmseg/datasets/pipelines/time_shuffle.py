'''
Author: Shuailin Chen
Created Date: 2021-07-11
Last Modified: 2021-07-11
	content: shuffle the first image and second image's order
'''

import os.path as osp

import mmcv
import numpy as np
import torch
import matplotlib.pyplot as plt

from mmseg.utils import split_images
from ..builder import PIPELINES

import mylib.image_utils as iu



@PIPELINES.register_module()
class TimeShuffle(object):
	''' Shuffle two images stacked in one array 
	
	Args:
		prob (float): probability to shuffle, must between 0 and 1
	'''

	def __init__(self, prob=0.5):
		assert prob>=0 and prob<=1, f'probability must in [0, 1], but got {prob}'
		self.prob = prob

	def __call__(self, results):
		if np.random.rand() < self.prob:
			# swap image
			img1, img2 = split_images(results['img'])
			# iu.save_image_by_cv2(img1, r'./tmp/1.jpg', if_norm=False)
			# iu.save_image_by_cv2(img2, r'./tmp/2.jpg', if_norm=False)
			results['img'] = np.concatenate((img2, img1), axis=-1)
			results['filename1'], results['filename2'] = results['filename2'], results['filename1']

			# change label, this is task specific
			ori_gt = results['gt_semantic_seg']
			new_gt = ori_gt.copy()
			new_gt[ori_gt==1] = 2
			new_gt[ori_gt==2] = 1
			results['gt_semantic_seg'] = new_gt
			
			# plt.matshow(ori_gt)
			# plt.savefig(r'./tmp/ori_gt.jpg')
			# plt.clf()
			# plt.matshow(new_gt)
			# plt.savefig(r'./tmp/new_gt.jpg')

		return results