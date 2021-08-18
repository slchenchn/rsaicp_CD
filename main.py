'''
Author: Shuailin Chen
Created Date: 2021-07-02
Last Modified: 2021-08-17
	content: 
'''
import time
import argparse
import os
from os import system

import torch
from mmcv.utils import DictAction
from tools.test import main
from tools.label2rsaicp import label2rsaicp

def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('--config', help='test config file path', default='configs/bit/bit_pos_s2_dd8_512x512_80k_S2Looking.py')
    parser.add_argument('--checkpoint', help='checkpoint file',
            default='work_dirs/bit_pos_s2_dd8_512x512_80k_S2Looking/20210817_120553/latest.pth')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
        
    # 天智杯官方定义的数据集路径和输出路径
    parser.add_argument("--input_dir", default='/input_path', help="input path", type=str)

    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        # default=['mIoU', 'mFscoreCD'],
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir',"--output_dir", help='directory where painted images will be saved',
        default='/output_path'
        )
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options', default={
                # 'data.test.data_root': '/input_path',
                'data.test.img1_dir': 'Image1',
                'data.test.img2_dir': 'Image2',
                'data.workers_per_gpu': 1,
                })
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--opacity',
        type=float,
        default=1,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


if __name__ == '__main__':
    # arguments and environments setting
    start_time = time.time()
    args = parse_args()
    args.options.update({'data.test.data_root':args.input_dir})

    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # inferrence 
    print(f'config: {args.config}')
    main(args)

    # convert label format
    label2rsaicp(args.show_dir, args.show_dir)
    
    print('total time:', time.time() - start_time)