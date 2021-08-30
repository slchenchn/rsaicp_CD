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

from mmseg.models.post_process.morphology import read_s2looking_label
from mmseg.core import eval_metrics
from mmseg.datasets import build_dataset


def get_args():
    ''' CLI arg parser '''
    parser = argparse.ArgumentParser('evaluate the model predictions')
    parser.add_argument('prediction_folder')
    parser.add_argument('--gt_folder', default=r'data/S2Looking/val/label_index')
    parser.add_argument('--out_folder', default=r'tmp')
    return parser.parse_args()


def eval_metrics_offline(cfg,
                        pred_folder, 
                        palette, 
                        metrics=['mFscoreCD', 'mFscore']
                        ):
    """Calculate evaluation metrics from offline GT and prediction images, maily use the evaluate of dataset class

    Args:
        cfg (Config): config dict, must have the necessary items to build a
            dataset
        palette (list | ndarray): palette of the label
        metrics ([str]): metrics to be calculated. Default: 
            ['mFscoreCD', 'mFscore']

    Returns:
        returns of the evaluate method of the dataset class
    """

    assert osp.isdir(pred_folder)

    dataset = build_dataset(cfg)
    preds = []
    for img_info in dataset.img_infos:
        pred_path = osp.join(pred_folder, img_info['ann']['seg_map'])
        pred = read_s2looking_label(pred_path, palette)
        preds.append(pred)

    return dataset.evaluate(preds, metric=metrics)




def show_confusion_pixels_on_image(pred_folder, 
                                    gt_folder, 
                                    palette, 
                                    save_folder):
    ''' Show truly and wrongly predicted pixels in different color

    Args:
        pred_folder (str): folder where the predicted result stored
        gt_folder (str): folder where the ground truth stored
        palette (list | ndarray): palette of the label
        save_folder (str): folder where the confusion result image stored
    '''

    def read_label_png(src_path:str)->np.ndarray:
        '''读取 label图像的信息，这个文件的格式比较复杂，直接读取会有问题，需要特殊处理

        Args:
        src_path (str): label文件路径或者其文件夹

        Returns
        label_idx (ndarray): np.ndarray格式的label信息
        '''
        # read label.png, get the label index
        tmp = PIL.Image.open(src_path)
        label_idx = np.asarray(tmp).astype(np.uint8)
        return label_idx

    assert osp.isdir(pred_folder)
    assert osp.isdir(gt_folder)

    gt_paths = glob(osp.join(gt_folder, '*.png'))
    for gt_path in gt_paths:
        pred_path = gt_path.replace(gt_folder, pred_folder)
        gt = read_label_png(gt_path)
        pred_rgb = read_label_png(pred_path)
        pred = np.empty_like(gt)
        # chagne rgb label image into index image
        for idx, color in enumerate(palette):
            color = np.array(color)[None, None, ...]
            pred[(pred_rgb==color).all(axis=2)] = idx

        err_path = osp.join(save_folder, osp.basename(gt_path).split('.')[0]+'_err.png')

        gt = gt > 0
        pred = pred > 0
        err = gt + pred * 2
        # true positive: white, true negative: black 
        # false positive: blue, false negative: red
        out_palette = np.array([[0, 0, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]])

        lbl.lblsave(err_path, err, out_palette)
 

if __name__ == '__main__':
    PALETTE = [[0, 0, 0], [0, 0, 255], [255, 0, 0], [255, 0, 255]]
    cfg = r'configs/_base_/datasets/s2looking.py'
    cfg = Config.fromfile(cfg)

    pred_folder = r'show_dir'
    save_folder = r'tmp/opening'
    
    ''' evaluate metrics offline '''
    eval_metrics_offline(cfg.data.val, save_folder, PALETTE)


    ''' show_confusion_pixels_on_image '''
    # args = get_args()
    # show_confusion_pixels_on_image(args.prediction_folder, args.gt_folder, PALETTE, args.out_folder)