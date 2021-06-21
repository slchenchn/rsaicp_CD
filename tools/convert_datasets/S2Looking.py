'''
Author: Shuailin Chen
Created Date: 2021-06-15
Last Modified: 2021-06-15
	content: Convert S2Looking dataset to mmseg default format
    UNDONE
'''

import os.path as osp
import argparse
import mmcv


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert S2Looking annotations to TrainIds')
    parser.add_argument('S2Looking_path', help='S2Looking data path')
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    s2looking_path = args.S2Looking_path
    out_dir = args.out_dir if args.out_dir else s2looking_path
    mmcv.mkdir_or_exist(out_dir)


