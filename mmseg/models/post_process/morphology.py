'''
Author: Shuailin Chen
Created Date: 2021-08-30
Last Modified: 2021-08-30
	content: evaluate and analyze the model predictions
'''

import mylib.labelme_utils as lbl
import mylib.image_utils as iu
import cv2
import PIL
import os
import os.path as osp
import argparse
import numpy as np
from glob import glob
from mmcv.utils import Config
import mmcv

from mmseg.core import eval_metrics
from mmseg.datasets import build_dataset


def morphology_operation(src_folder, 
                        palette, 
                        kernel_size=5, 
                        save_folder=None,
                        operation=cv2.MORPH_OPEN
                        ):
    ''' Perform open operation (erosion+dialation) to model predictions 
    
    Args:
        src_folder (str): source folder of RGB labels to be processed
        palette (list | ndarray): palette of the label
        kernel_size (int): kernel size for the operation. Default: 5
        save_folder (str): folder to save the tranformed labels. if not
            specified, the results will not be saved. Default: None
    '''

    lables = read_s2looking_all_label(src_folder, palette=palette,
                                        label_format='rgb')

    # split one label image into two labels, and perform opening operation seperately
    new_labels = []
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    for label in lables:
        label1 = (label[..., 0] > 0).astype(np.uint8)
        label2 = (label[..., 2] > 0).astype(np.uint8)
        label1 = cv2.morphologyEx(label1, operation, kernel)
        label2 = cv2.morphologyEx(label2, operation, kernel)

        label[..., 0] = label1 * 255
        label[..., 2] = label2 * 255
        new_labels.append(label)
    
    if save_folder is not None:
        mmcv.mkdir_or_exist(save_folder)
        label_names = os.listdir(src_folder)
        for label, name in zip(new_labels, label_names):
            # lbl.lblsave(osp.join(save_folder, name), label, palette)
            label = cv2.cvtColor(label, cv2.COLOR_RGB2BGR)
            cv2.imwrite(osp.join(save_folder, name), label)

    return new_labels        
        

def read_s2looking_all_label(folderpath, palette, label_format='index'):
    ''' Read all the S2Looking dataset label in a folder

    Args:
        folderpath (str): folder path
        palette (list | ndarray): palette of the label
        label_format (str): 'index': return the indexed label data, 'rgb':
            return the raw label data in rgb format. Default: 'index'
    '''
    assert osp.isdir(folderpath)

    labels = []
    label_paths = glob(osp.join(folderpath, '*.png'))
    for label_path in label_paths:
        pred = read_s2looking_label(label_path, palette,
                                    label_format=label_format)
        labels.append(pred)

    return labels


def read_s2looking_label(filepath, palette, label_format='index'):
    ''' Read a label of the S2Looking dataset

    Args:
        filepath (str): file path
        palette (list | ndarray): palette of the label
        label_format (str): 'index': return the indexed label data, 'rgb':
            return the raw label data in rgb format. Default: 'index'
    '''

    label_rgb = cv2.imread(filepath)
    label_rgb = cv2.cvtColor(label_rgb, cv2.COLOR_RGB2BGR)

    if label_format == 'rgb':
        return label_rgb
    elif label_format == 'index':
        label = np.empty(shape=label_rgb.shape[:-1])
        for idx, color in enumerate(palette):
            color = np.array(color)[None, None, ...]
            label[(label_rgb==color).all(axis=2)] = idx

        return label
    else:
        raise ValueError


if __name__ == '__main__':
    PALETTE = [[0, 0, 0], [0, 0, 255], [255, 0, 0], [255, 0, 255]]
    pred_folder = r'show_dir'

    ''' morphology_operation '''
    save_folder = r'tmp/opening'
    morphology_operation(pred_folder, PALETTE, save_folder=save_folder,
                    kernel_size=15, operation=cv2.MORPH_OPEN)