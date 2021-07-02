'''
Author: Shuailin Chen
Created Date: 2021-07-02
Last Modified: 2021-07-02
	content: 
'''

import os
import os.path as osp
import cv2
from glob import glob
import numpy as np
import mmcv

def label2rsaicp(src_path, dst_path):
    ''' Convert the label file into the format rsaicp required
    '''

    label1 = np.zeros((1024, 1024, 3), dtype=np.uint8)
    label2 = np.zeros_like(label1)
    mmcv.mkdir_or_exist(osp.join(dst_path, 'label1'))
    mmcv.mkdir_or_exist(osp.join(dst_path, 'label2'))

    labelpaths = glob(osp.join(src_path, '*.png'))
    for labelpath in labelpaths:
        labelname = osp.basename(labelpath)
        label = cv2.imread(labelpath)
        label1[..., 0] = label[..., 0]
        label2[..., 2] = label[..., 2]
        cv2.imwrite(osp.join(dst_path, 'label1', labelname), label1)
        cv2.imwrite(osp.join(dst_path, 'label2', labelname), label2)


if __name__ == '__main__':
    label2rsaicp(src_path='show_dir', dst_path='show_dir')