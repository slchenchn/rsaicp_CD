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

from mmseg.core import eval_metrics
from mmseg.datasets import build_dataset


def eval_metrics_offline(cfg,
                        pred_folder, 
                        palette, 
                        metrics=['mFscoreCD', 'mFscore']
                        ):
    """Calculate evaluation metrics from offline gt and prediction images
    Args:
        
    """

    assert osp.isdir(pred_folder)

    preds = []
    pred_paths = glob(osp.join(pred_folder, '*.png'))
    for pred_path in pred_paths:
        pred_rgb = read_label_png(pred_path)
        pred = np.empty(shape=pred_rgb.shape[:-1])
        for idx, color in enumerate(palette):
            color = np.array(color)[None, None, ...]
            pred[(pred_rgb==color).all(axis=2)] = idx

        preds.append(pred)
    

    dataset = build_dataset(cfg.data.train)
    dataset.evaluate(preds, metric=metrics)


def read_label_png(src_path:str)->np.ndarray:
    '''读取 label图像的信息，这个文件的格式比较复杂，直接读取会有问题，需要特殊处理

    Args:
    src_path (str): label文件路径或者其文件夹

    Returns
    label_idx (ndarray): np.ndarray格式的label信息
    '''
    
    # read label.png, get the label index
    tmp = PIL.Image.open(src_path)
    label_idx = np.asarray(tmp)

    return label_idx


def get_args():
    parser = argparse.ArgumentParser('evaluate the model predictions')
    parser.add_argument('prediction_folder')
    parser.add_argument('--gt_folder', default=r'data/S2Looking/val/label_index')
    parser.add_argument('--out_folder', default=r'tmp')
    return parser.parse_args()


def show_confusion_pixels_on_image(pred_folder, gt_folder, palette, out_folder):
    ''' 
    '''

    assert osp.isdir(pred_folder)
    assert osp.isdir(gt_folder)

    gt_paths = glob(osp.join(gt_folder, '*.png'))
    for gt_path in gt_paths:
        pred_path = gt_path.replace(gt_folder, pred_folder)
        gt = read_label_png(gt_path)
        pred_rgb = read_label_png(pred_path)
        pred = np.empty_like(gt)
        for idx, color in enumerate(palette):
            color = np.array(color)[None, None, ...]
            pred[(pred_rgb==color).all(axis=2)] = idx

        err_path = osp.join(out_folder, osp.basename(gt_path).split('.')[0]+'_err.png')

        gt = gt > 0
        pred = pred > 0
        err = gt + pred * 2
        # true positive: white, true negative: black 
        # false positive: blue, false negative: red
        out_palette = np.array([[0, 0, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]])

        lbl.lblsave(err_path, err, out_palette)

        # # err = np.empty_like(pred_rgb)
        # for ii in range(len(palette)):
        #     idx = (gt==ii) and (pred==ii)
        #     err[idx] = 

        # err = (gt != pred).astype(np.uint8) * 255

        # # save err image
        # iu.save_image_by_cv2(err, err_path)


if __name__ == '__main__':
    PALETTE = [[0, 0, 0], [0, 0, 255], [255, 0, 0], [255, 0, 255]]

    # 
    


    # args = get_args()
    # show_confusion_pixels_on_image(args.prediction_folder, args.gt_folder, PALETTE, args.out_folder)