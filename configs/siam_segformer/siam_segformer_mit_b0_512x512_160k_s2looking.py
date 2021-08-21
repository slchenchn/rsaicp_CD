'''
Author: Shuailin Chen
Created Date: 2021-07-06
Last Modified: 2021-08-21
	content: 
'''
_base_ = [
    '../_base_/models/siam_segformer_mit-b0.py', '../_base_/datasets/s2looking.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]


data = dict(
    samples_per_gpu=16,
    workers_per_gpu=16,
    )

evaluation = dict(metric=['mFscore', 'mFscoreCD'])